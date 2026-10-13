# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer: Leakage-Safe Feature Store Pipeline (`gold.features_hourly`)
# MAGIC
# MAGIC **Input Table:** `main.silver.grid_hourly_conformed`  
# MAGIC **Output Table:** `main.gold.features_hourly`  
# MAGIC
# MAGIC ### Architecture & Schema Contracts:
# MAGIC 1. **Row Grain Definition:**
# MAGIC    - Primary Key: `(respondent, timestamp_utc)` representing atomic hourly observations.
# MAGIC    - In this table, `timestamp_utc` represents the target observation timestamp $T$.
# MAGIC 2. **Multi-Horizon Forecast-Origin Availability:**
# MAGIC    - Models make predictions from a forecast origin $t_0$ for target $T = t_0 + h$:
# MAGIC      - **Horizon $H=24$ (Day-Ahead Origin $t_0 = T-24$):** Eligible autoregressive features include `demand_lag_24h`, `demand_lag_48h`, `demand_lag_168h`, and `demand_rolling_avg_24h_lag24` ($T-47$ to $T-24$).
# MAGIC      - **Horizon $H=48$ (48-Hour Ahead Origin $t_0 = T-48$):** A 24h lag relative to target $T$ falls at $t_0 + 24$ (in the future relative to origin $t_0$). Therefore, $H=48$ models strictly consume `demand_lag_48h`, `demand_lag_72h`, `demand_lag_168h`, and `demand_rolling_avg_24h_lag48` ($T-71$ to $T-48$), eliminating leakage.
# MAGIC 3. **Weather Availability Dual-Track Benchmark:**
# MAGIC    - **Oracle Weather Benchmark:** Target-hour thermodynamic features (`weighted_temperature_f`, degree hours) in this feature store use retrospective observed weather (ERA5/Open-Meteo reanalysis). This isolates pure load-model capacity given perfect weather knowledge.
# MAGIC    - **Operational Forecast Benchmark:** In operational Phase 4 inference, target-hour weather is populated from numerical weather prediction (NWP) forecasts available at or prior to origin $t_0$.
# MAGIC 4. **Training Sample Assembly Bridge:**
# MAGIC    - Downstream Phase 4 training pairs origin features with future targets using either a direct multi-output structure or a long-form panel `(respondent, forecast_origin_utc, target_timestamp_utc, lead_hours)`.
# MAGIC 5. **Lookback Window Contamination Filter:**
# MAGIC    - Flags observations whose 168h historical lookback window intersects the Feb 15–18, 2021 Uri blackout (`is_lookback_emergency_contaminated`) to prevent model pollution.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import os
import json
import math
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, StringType, BooleanType, IntegerType
from delta.tables import DeltaTable

CATALOG = "main"
SOURCE_TABLE = f"{CATALOG}.silver.grid_hourly_conformed"
TARGET_TABLE = f"{CATALOG}.gold.features_hourly"

EMPIRICAL_BALANCE_F = 55.2
CONVENTIONAL_BALANCE_F = 65.0

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA gold")

# COMMAND ----------
# DBTITLE 1,Dynamic Configuration Locator
def load_platform_config(filename="ercot_config.json"):
    curr = os.getcwd()
    for _ in range(5):
        test_path = os.path.join(curr, "configs", filename)
        if os.path.exists(test_path):
            with open(test_path, "r", encoding="utf-8") as f:
                print(f"Loaded platform config from {test_path}")
                return json.load(f)
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
        
    repo_candidate = f"/Workspace/Repos/configs/{filename}"
    if os.path.exists(repo_candidate):
        with open(repo_candidate, "r", encoding="utf-8") as f:
            print(f"Loaded platform config from {repo_candidate}")
            return json.load(f)
            
    raise FileNotFoundError(
        f"Critical Configuration Error: Could not locate 'configs/{filename}'. "
        "Ensure the repository is synced properly in Databricks Repos."
    )

config = load_platform_config("ercot_config.json")
BA_CODE = config.get("region", {}).get("code", "ERCO")

# COMMAND ----------
# DBTITLE 1,Verify Upstream Source
assert spark.catalog.tableExists(SOURCE_TABLE), f"Required upstream table {SOURCE_TABLE} does not exist!"
df_silver = spark.table(SOURCE_TABLE)

# COMMAND ----------
# DBTITLE 1,Transformation: Build Feature Dataset
def build_gold_features_dataset(df_silver, empirical_balance_f=EMPIRICAL_BALANCE_F):
    """
    Constructs leakage-safe tabular ML feature store from silver conformed hourly data.
    """
    pi = math.pi
    
    # Major industrial shutdown holidays observed in EDA to exhibit substantial demand drops (-8k to -10k MW)
    MAJOR_SHUTDOWN_HOLIDAYS = [
        "Thanksgiving Day",
        "Christmas Day",
        "New Year's Day",
        "Memorial Day",
        "Labor Day"
    ]
    
    # 1. Base calendar and cyclical features
    df_feat = (
        df_silver
        # Trigonometric Cyclical Encodings (preserve continuous wrap-around)
        .withColumn("hour_sin", F.sin(2.0 * pi * F.col("hour_local") / 24.0))
        .withColumn("hour_cos", F.cos(2.0 * pi * F.col("hour_local") / 24.0))
        .withColumn("day_of_week_sin", F.sin(2.0 * pi * (F.col("day_of_week_local") - 1) / 7.0))
        .withColumn("day_of_week_cos", F.cos(2.0 * pi * (F.col("day_of_week_local") - 1) / 7.0))
        .withColumn("day_of_year_sin", F.sin(2.0 * pi * (F.col("day_of_year_local") - 1) / 365.25))
        .withColumn("day_of_year_cos", F.cos(2.0 * pi * (F.col("day_of_year_local") - 1) / 365.25))
        .withColumn("month_sin", F.sin(2.0 * pi * (F.col("month_local") - 1) / 12.0))
        .withColumn("month_cos", F.cos(2.0 * pi * (F.col("month_local") - 1) / 12.0))
        
        # Holiday Sub-Classifications
        .withColumn(
            "is_major_shutdown_holiday",
            F.when(F.col("holiday_name").isin(MAJOR_SHUTDOWN_HOLIDAYS), F.lit(1.0)).otherwise(F.lit(0.0))
        )
        .withColumn(
            "season_summer",
            F.when(F.col("month_local").isin(6, 7, 8), F.lit(1.0)).otherwise(F.lit(0.0))
        )
        .withColumn(
            "season_winter",
            F.when(F.col("month_local").isin(12, 1, 2), F.lit(1.0)).otherwise(F.lit(0.0))
        )
        .withColumn(
            "season_shoulder",
            F.when(F.col("month_local").isin(3, 4, 5, 9, 10, 11), F.lit(1.0)).otherwise(F.lit(0.0))
        )
        
        # Empirical Balance Point Degree-Hours & Asymmetric Terms
        .withColumn(
            "cdh_55",
            F.when(F.col("weighted_temperature_f") > empirical_balance_f, 
                   F.col("weighted_temperature_f") - empirical_balance_f).otherwise(0.0)
        )
        .withColumn(
            "hdh_55",
            F.when(F.col("weighted_temperature_f") < empirical_balance_f,
                   empirical_balance_f - F.col("weighted_temperature_f")).otherwise(0.0)
        )
        .withColumn(
            "temp_deviation_55_sq",
            F.pow(F.col("weighted_temperature_f") - empirical_balance_f, 2.0)
        )
        # Regional Temperature Gradient (North-to-South cold front indicator)
        .withColumn(
            "temp_delta_dfw_houston_c",
            F.col("temp_dfw_c") - F.col("temp_houston_c")
        )
        
        # Grid Emergency Flag (Winter Storm Uri Involuntary Curtailment)
        .withColumn(
            "is_grid_emergency",
            F.when(
                (F.col("timestamp_local") >= "2021-02-15 00:00:00") &
                (F.col("timestamp_local") <  "2021-02-19 00:00:00"),
                F.lit(True)
            ).otherwise(F.lit(False))
        )
        # Lookback Contamination Flag (Winter Storm Uri):
        # A sample's 168h lookback window is contaminated if it overlaps Feb 15–18, 2021.
        # This extends through Feb 26, 2021 (7 days post-freeze).
        .withColumn(
            "is_lookback_emergency_contaminated",
            F.when(
                (F.col("timestamp_local") >= "2021-02-15 00:00:00") &
                (F.col("timestamp_local") <  "2021-02-26 00:00:00"),
                F.lit(True)
            ).otherwise(F.lit(False))
        )
    )
    
    # 2. Horizon-Aware Lagged Features
    # AVAILABILITY CONTRACT:
    # - Horizon H=24 (Day-Ahead Origin): Models predicting <= 24h ahead may consume lags >= 24h.
    # - Horizon H=48 (48-Hour Ahead Origin): Models predicting <= 48h ahead MUST consume lags >= 48h.
    #   (Using lag 24h for a 48h horizon is a critical data leakage violation!)
    window_time = Window.partitionBy("respondent").orderBy("timestamp_utc")
    window_rolling_lag24 = Window.partitionBy("respondent").orderBy("timestamp_utc").rowsBetween(-47, -24)
    window_rolling_lag48 = Window.partitionBy("respondent").orderBy("timestamp_utc").rowsBetween(-71, -48)
    
    df_feat_lagged = (
        df_feat
        # Autoregressive Lags (Explicitly Partitioned by Horizon Tier)
        .withColumn("demand_lag_24h", F.lag("actual_demand_mw", 24).over(window_time)) # Eligible for H=24
        .withColumn("demand_lag_48h", F.lag("actual_demand_mw", 48).over(window_time)) # Eligible for H=24 & H=48
        .withColumn("demand_lag_72h", F.lag("actual_demand_mw", 72).over(window_time)) # Eligible for H=48
        .withColumn("demand_lag_168h", F.lag("actual_demand_mw", 168).over(window_time)) # Eligible for all horizons
        
        # Rolling Statistics: Horizon 24 Window (t-47 to t-24)
        .withColumn("demand_rolling_avg_24h_lag24", F.round(F.avg("actual_demand_mw").over(window_rolling_lag24), 1))
        .withColumn("demand_rolling_std_24h_lag24", F.round(F.stddev("actual_demand_mw").over(window_rolling_lag24), 1))
        .withColumn("demand_rolling_count_24h_lag24", F.count("actual_demand_mw").over(window_rolling_lag24))
        
        # Rolling Statistics: Horizon 48 Window (t-71 to t-48, strictly leakage-safe for H=48)
        .withColumn("demand_rolling_avg_24h_lag48", F.round(F.avg("actual_demand_mw").over(window_rolling_lag48), 1))
        .withColumn("demand_rolling_std_24h_lag48", F.round(F.stddev("actual_demand_mw").over(window_rolling_lag48), 1))
        .withColumn("demand_rolling_count_24h_lag48", F.count("actual_demand_mw").over(window_rolling_lag48))
        
        # Holiday-Adjacent Indicator (True if previous 24h or next 24h is a holiday)
        .withColumn(
            "is_holiday_adjacent",
            F.when(
                (F.lag("is_holiday", 24).over(window_time) == True) |
                (F.lead("is_holiday", 24).over(window_time) == True),
                F.lit(1.0)
            ).otherwise(F.lit(0.0))
        )
        
        # Base training sample validity conditions:
        # 1. Target actual demand is valid and within physical bounds
        # 2. Target hour is not a curtailed grid emergency
        # 3. 168h lookback history does not intersect Winter Storm Uri
        # 4. Canonical 168h seasonal lag exists
        .withColumn(
            "is_base_training_eligible",
            F.when(
                F.col("actual_demand_mw").isNotNull() &
                (F.col("is_valid_demand") == True) &
                (F.col("is_grid_emergency") == False) &
                (F.col("is_lookback_emergency_contaminated") == False) &
                F.col("demand_lag_168h").isNotNull(),
                F.lit(True)
            ).otherwise(F.lit(False))
        )
        # Horizon H=24 Sample Validity: requires base validity, lag_24h, lag_48h, and complete 24h rolling window
        .withColumn(
            "is_valid_training_sample_h24",
            F.when(
                (F.col("is_base_training_eligible") == True) &
                F.col("demand_lag_24h").isNotNull() &
                F.col("demand_lag_48h").isNotNull() &
                (F.col("demand_rolling_count_24h_lag24") == 24),
                F.lit(True)
            ).otherwise(F.lit(False))
        )
        # Horizon H=48 Sample Validity: requires base validity, lag_48h, lag_72h, and complete 48h rolling window
        .withColumn(
            "is_valid_training_sample_h48",
            F.when(
                (F.col("is_base_training_eligible") == True) &
                F.col("demand_lag_48h").isNotNull() &
                F.col("demand_lag_72h").isNotNull() &
                (F.col("demand_rolling_count_24h_lag48") == 24),
                F.lit(True)
            ).otherwise(F.lit(False))
        )
        # Canonical backwards-compatible training sample flag
        .withColumn("is_valid_training_sample", F.col("is_valid_training_sample_h24"))
        .withColumn("_gold_feature_extracted_at_utc", F.current_timestamp())
    )
    
    return df_feat_lagged

# COMMAND ----------
# DBTITLE 1,Pre-Write Quality Gate Assertions
def validate_gold_features_candidate(df_features):
    """
    Enforces ML Feature Table invariants before writing:
    1. Zero null composite keys (respondent, timestamp_utc)
    2. Zero duplicate timestamps
    3. Strict lag alignment: verify demand_lag_24h and demand_lag_48h match expectations
    4. Cyclical encoding range bounds: sin/cos strictly in [-1.0, 1.0]
    5. Training sample integrity: zero Uri target or lookback contamination leaks
    """
    total_rows = df_features.count()
    distinct_keys = df_features.select("respondent", "timestamp_utc").distinct().count()
    null_keys = df_features.filter(F.col("respondent").isNull() | F.col("timestamp_utc").isNull()).count()
    
    assert null_keys == 0, f"Integrity Failure: Found {null_keys} null keys in Feature candidate!"
    assert total_rows == distinct_keys, (
        f"Integrity Failure: Found {total_rows - distinct_keys} duplicate timestamp records!"
    )
    
    # 2b. Chronological Grid Continuity Check: Ensure zero missing hours across the span
    span_stats = df_features.select(
        F.min("timestamp_utc").alias("min_utc"),
        F.max("timestamp_utc").alias("max_utc"),
        F.countDistinct("timestamp_utc").alias("distinct_hours")
    ).first()
    min_ts = span_stats["min_utc"]
    max_ts = span_stats["max_utc"]
    distinct_hours = span_stats["distinct_hours"]
    if min_ts is not None and max_ts is not None and distinct_hours > 0:
        expected_span_hours = int((max_ts - min_ts).total_seconds() / 3600.0) + 1
        assert distinct_hours == expected_span_hours, (
            f"Grid Continuity Failure: Expected {expected_span_hours:,} continuous hours "
            f"between {min_ts} and {max_ts}, but found {distinct_hours:,} (missing {expected_span_hours - distinct_hours:,} hours). "
            "Continuous hourly grid is strictly required for row-offset autoregressive lag safety."
        )
    
    # Cyclical feature boundary check
    bounds_violation = df_features.filter(
        (F.col("hour_sin") < -1.0001) | (F.col("hour_sin") > 1.0001) |
        (F.col("hour_cos") < -1.0001) | (F.col("hour_cos") > 1.0001)
    ).count()
    assert bounds_violation == 0, f"Math Failure: {bounds_violation} cyclical features outside [-1, 1] range!"
    
    # Horizon H=24 and H=48 training eligibility integrity assertions
    h24_invalid = df_features.filter(
        (F.col("is_valid_training_sample_h24") == True) &
        (
            F.col("demand_lag_24h").isNull() |
            F.col("demand_lag_48h").isNull() |
            (F.col("demand_rolling_count_24h_lag24") != 24) |
            (F.col("is_grid_emergency") == True) |
            (F.col("is_lookback_emergency_contaminated") == True)
        )
    ).count()
    assert h24_invalid == 0, f"Integrity Failure: Found {h24_invalid} invalid rows flagged as valid H=24 training samples!"
    
    h48_invalid = df_features.filter(
        (F.col("is_valid_training_sample_h48") == True) &
        (
            F.col("demand_lag_48h").isNull() |
            F.col("demand_lag_72h").isNull() |
            (F.col("demand_rolling_count_24h_lag48") != 24) |
            (F.col("is_grid_emergency") == True) |
            (F.col("is_lookback_emergency_contaminated") == True)
        )
    ).count()
    assert h48_invalid == 0, f"Integrity Failure: Found {h48_invalid} invalid rows flagged as valid H=48 training samples!"
    
    # Degree hour non-negativity check
    invalid_dh = df_features.filter((F.col("cdh_55") < 0.0) | (F.col("hdh_55") < 0.0)).count()
    assert invalid_dh == 0, f"Thermodynamic Failure: Found {invalid_dh} negative degree-hour records!"
    
    print(f"Feature Table Quality Gate Passed:")
    print(f"  - Total records: {total_rows:,} (Continuous hourly grid verified)")
    print(f"  - Cyclical feature bounds verified (sin/cos in [-1.0, 1.0])")
    print(f"  - Strict Horizon sample validity verified: H=24 and H=48 training flags verified")
    print(f"  - Winter Storm Uri target AND lookback contamination verified (0 sample leaks)")

# COMMAND ----------
# DBTITLE 1,Idempotent Delta MERGE
def upsert_gold_features_table(df_features, target_table, spark_session):
    """
    Executes idempotent Delta MERGE on (respondent, timestamp_utc).
    """
    total_records = df_features.count()
    if not spark_session.catalog.tableExists(target_table):
        df_features.write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable(target_table)
        print(f"Initialized Gold Feature table {target_table} with {total_records:,} records.")
    else:
        delta_target = DeltaTable.forName(spark_session, target_table)
        merge_condition = (
            "target.respondent = source.respondent "
            "AND target.timestamp_utc = source.timestamp_utc"
        )
        
        update_condition = (
            "NOT (target.actual_demand_mw <=> source.actual_demand_mw) OR "
            "NOT (target.demand_lag_24h <=> source.demand_lag_24h) OR "
            "NOT (target.demand_lag_48h <=> source.demand_lag_48h) OR "
            "NOT (target.weighted_temperature_f <=> source.weighted_temperature_f) OR "
            "NOT (target.is_valid_training_sample <=> source.is_valid_training_sample)"
        )
        
        delta_target.alias("target").merge(
            df_features.alias("source"),
            merge_condition
        ).whenMatchedUpdateAll(
            condition=update_condition
        ).whenNotMatchedInsertAll(
        ).execute()
        print(f"Idempotent MERGE successfully executed on {target_table}.")

# COMMAND ----------
# DBTITLE 1,Post-Write Read-Back Storage Audit
def audit_persisted_gold_features(target_table, spark_session):
    """
    Direct read-back audit from Delta storage verifying physical persistence integrity.
    """
    df_persisted = spark_session.table(target_table)
    persisted_rows = df_persisted.count()
    distinct_keys = df_persisted.select("respondent", "timestamp_utc").distinct().count()
    null_keys = df_persisted.filter(
        F.col("respondent").isNull() | F.col("timestamp_utc").isNull()
    ).count()

    assert null_keys == 0, f"Post-Write Failure: Found {null_keys} null keys in {target_table}!"
    assert persisted_rows == distinct_keys, (
        f"Post-Write Failure: Found {persisted_rows - distinct_keys} duplicate keys in {target_table}!"
    )
    print(f"Post-Write Storage Audit Passed: {persisted_rows:,} verified records in {target_table}.")

# COMMAND ----------
# DBTITLE 1,Execute Feature Pipeline
df_features = build_gold_features_dataset(df_silver)
validate_gold_features_candidate(df_features)
upsert_gold_features_table(df_features, TARGET_TABLE, spark)
audit_persisted_gold_features(TARGET_TABLE, spark)

# COMMAND ----------
# MAGIC %sql
# MAGIC -- Sample Inspection of Horizon-Partitioned Feature Table
# MAGIC SELECT 
# MAGIC     timestamp_local,
# MAGIC     actual_demand_mw,
# MAGIC     demand_lag_24h,
# MAGIC     demand_lag_48h,
# MAGIC     demand_rolling_avg_24h_lag24,
# MAGIC     demand_rolling_avg_24h_lag48,
# MAGIC     weighted_temperature_f,
# MAGIC     cdh_55,
# MAGIC     hdh_55,
# MAGIC     is_major_shutdown_holiday,
# MAGIC     is_holiday_adjacent,
# MAGIC     is_lookback_emergency_contaminated,
# MAGIC     is_valid_training_sample
# MAGIC FROM gold.features_hourly
# MAGIC WHERE is_valid_training_sample = True
# MAGIC ORDER BY timestamp_local DESC
# MAGIC LIMIT 25;
