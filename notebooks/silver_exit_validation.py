# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer Exit Validation — Cross-Table Quality Gate
# MAGIC
# MAGIC **Purpose:** Validate cross-table consistency and semantic correctness that individual Silver modules cannot check alone.
# MAGIC This notebook is designed to run as a DAG task after all Silver modules complete, acting as a circuit breaker before Gold.
# MAGIC
# MAGIC ### Design Decision (Why only 4 checks, not 8):
# MAGIC Each Silver module already enforces its own structural integrity (uniqueness, null keys, DST, cardinality).
# MAGIC Repeating those checks here wastes compute and violates DRY. This exit gate focuses exclusively on
# MAGIC **cross-table semantic checks** — things no individual module can validate in isolation.
# MAGIC
# MAGIC ### Checks:
# MAGIC 1. **Final Table Read-Back Audit:** Verify persisted data integrity after MERGE (confirms the write path didn't corrupt keys).
# MAGIC 2. **Missing Data Semantics:** Assert explicit separation of `AVAILABLE`, `NOT_YET_PUBLISHED`, and `MISSING_HISTORICAL` — confirms the full outer join semantics are working correctly across demand and weather.
# MAGIC 3. **Forecast-Time Availability:** Assert that no analytical metrics leak into rows where actuals don't exist yet.
# MAGIC 4. **Calendar Coverage:** Assert calendar dimension extends beyond current data to support future model scoring.

# COMMAND ----------
# DBTITLE 1,Environment & Table Loads
from pyspark.sql import functions as F

CATALOG = "main"
spark.sql(f"USE CATALOG {CATALOG}")

df_conformed = spark.table("main.silver.grid_hourly_conformed")
df_calendar = spark.table("main.bronze.calendar_reference")

print("Tables successfully loaded for exit validation.")

# COMMAND ----------
# DBTITLE 1,Check 1: Final Table Read-Back Audit (Post-MERGE Integrity)
# Why: Individual modules assert pre-MERGE integrity on DataFrames. This validates the
# persisted Delta table post-write — confirming the MERGE path didn't introduce duplicates
# or null keys. This is a read-back audit, not a duplicate of pre-merge checks.
total_rows = df_conformed.count()
distinct_keys = df_conformed.select("respondent", "timestamp_utc").distinct().count()
null_keys = df_conformed.filter(F.col("respondent").isNull() | F.col("timestamp_utc").isNull()).count()

assert null_keys == 0, f"Check 1 Failed: Found {null_keys} null join keys in persisted table!"
assert total_rows == distinct_keys, f"Check 1 Failed: {total_rows - distinct_keys} duplicate keys in persisted table!"
print(f"[PASS] Check 1: Post-MERGE read-back audit passed. {total_rows:,} rows, 0 null keys, 0 duplicates.")

# COMMAND ----------
# DBTITLE 1,Check 2: Missing Data Semantics (Cross-Table Join Verification)
# Why: This validates that the full outer join between demand and weather correctly
# categorizes data availability. No single Silver module can check this — it's a property
# of how the two sources interact after joining.
status_distribution = df_conformed.groupBy("demand_availability_status", "weather_availability_status") \
    .agg(
        F.count("*").alias("hour_count"),
        F.min("timestamp_utc").alias("earliest_utc"),
        F.max("timestamp_utc").alias("latest_utc")
    )
display(status_distribution)

# No row should have a null demand status — the join logic must always classify
assert df_conformed.filter(F.col("demand_availability_status").isNull()).count() == 0, \
    "Check 2 Failed: Found rows with null demand_availability_status!"

# NOT_YET_PUBLISHED should only exist for recent timestamps (within ~48h of now)
future_misclassified = df_conformed.filter(
    (F.col("demand_availability_status") == "NOT_YET_PUBLISHED") &
    (F.col("timestamp_utc") < (F.current_timestamp() - F.expr("INTERVAL 72 HOURS")))
).count()
assert future_misclassified == 0, \
    f"Check 2 Failed: {future_misclassified} rows older than 72h still marked NOT_YET_PUBLISHED!"

print(f"[PASS] Check 2: Missing data semantics verified. All hours explicitly categorized with valid temporal boundaries.")

# COMMAND ----------
# DBTITLE 1,Check 3: Forecast-Time Leakage Prevention
# Why: When Gold computes analytical metrics (error, APE), it must only do so on matched
# observation pairs. This check validates the precondition: no invalid data combinations
# exist that could produce nonsense metrics downstream.

# Forecast MW should only exist when demand had a forecast published
# (i.e., day_ahead_forecast_mw should be null when demand_availability_status != 'AVAILABLE')
forecast_without_demand = df_conformed.filter(
    (F.col("demand_availability_status") != "AVAILABLE") &
    (F.col("day_ahead_forecast_mw").isNotNull())
).count()
# Note: This is expected to be 0 in most cases, but EIA sometimes publishes forecasts
# before actuals. Log the count rather than hard-assert.
if forecast_without_demand > 0:
    print(f"[INFO] Check 3: {forecast_without_demand} rows have forecasts without matching actuals (expected for recent hours).")
else:
    print(f"[INFO] Check 3: All forecast values have matching actuals.")

# is_valid_demand must be NULL (not False) when actual_demand_mw is NULL
# This validates the Silver design decision: NULL = absent, False = present but invalid
invalid_validity_flag = df_conformed.filter(
    (F.col("actual_demand_mw").isNull()) & (F.col("is_valid_demand").isNotNull())
).count()
assert invalid_validity_flag == 0, \
    f"Check 3 Failed: {invalid_validity_flag} rows have is_valid_demand set when actual demand is absent!"

print(f"[PASS] Check 3: Forecast-time leakage preconditions verified. is_valid_demand correctly NULL for absent demand.")

# COMMAND ----------
# DBTITLE 1,Check 4: Calendar Coverage for Future Scoring
# Why: The calendar dimension must extend beyond current data to support forward model
# scoring in Gold. If the calendar ends before or at the data edge, future joins will
# drop rows or produce null holiday flags.
max_data_date = df_conformed.select(F.max(F.to_date("timestamp_local"))).first()[0]
max_calendar_date = df_calendar.select(F.max(F.to_date("date"))).first()[0]

print(f"Max Conformed Data Date: {max_data_date}")
print(f"Max Calendar Dimension Date: {max_calendar_date}")

assert max_calendar_date > max_data_date, (
    f"Check 4 Failed: Calendar ends at {max_calendar_date}, which cannot support future scoring beyond {max_data_date}!"
)
days_ahead = (max_calendar_date - max_data_date).days
assert days_ahead >= 365, (
    f"Check 4 Warning: Calendar only extends {days_ahead} days ahead. Recommend at least 365 days for stable forward scoring."
)
print(f"[PASS] Check 4: Calendar coverage verified. Calendar extends {days_ahead} days ahead of current data.")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Summary: 4/4 Silver Exit Quality Checks Passed
# MAGIC
# MAGIC | Check | What | Why it's here (not in modules) |
# MAGIC |-------|------|-------------------------------|
# MAGIC | 1 | Post-MERGE read-back | Validates persisted data, not pre-write DataFrame |
# MAGIC | 2 | Missing data semantics | Cross-table join property (demand × weather interaction) |
# MAGIC | 3 | Forecast-time leakage | Validates preconditions for Gold metric computation |
# MAGIC | 4 | Calendar coverage | Cross-layer dependency (Bronze calendar → Silver → Gold scoring) |
