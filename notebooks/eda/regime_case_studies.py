# Databricks notebook source
# MAGIC %md
# MAGIC # Grid Stress Regime Case Studies: Uri & Summer 2023
# MAGIC
# MAGIC **Core Objective:** Inspect selected extreme periods and the emergency-event flag used to
# MAGIC keep known supply-constrained observations identifiable during analysis and training.
# MAGIC Served demand alone cannot recover unserved load; any estimate of unconstrained demand must
# MAGIC be attributed to its external source and treated separately from this dataset's observations.
# MAGIC
# MAGIC **Feature Layer Implication:**
# MAGIC - The Gold table flags Feb 15–18, 2021 as an emergency period. The flag identifies a known
# MAGIC   regime; this demand-only view does not quantify involuntary load shedding.
# MAGIC - Summer 2023 values and the >95°F binned relationship are descriptive. They do not, by
# MAGIC   themselves, establish unconstrained demand or a general linear temperature response.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
from scipy import stats
from pyspark.sql import functions as F

SILVER_TABLE = "main.silver.grid_hourly_conformed"

df_full = spark.table(SILVER_TABLE).filter(F.col("actual_demand_mw").isNotNull())

# COMMAND ----------
# DBTITLE 1,1. Winter Storm Uri (Feb 13–20, 2021) — Load Shedding Audit
df_uri = (
    df_full.filter(
        (F.col("timestamp_local") >= "2021-02-13") &
        (F.col("timestamp_local") <  "2021-02-20")
    )
    .select("timestamp_local", "actual_demand_mw", "weighted_temperature_f")
    .orderBy("timestamp_local")
)

display(df_uri)

uri_stats = df_uri.select(
    F.round(F.min("weighted_temperature_f"), 1).alias("min_temp_f"),
    F.round(F.min("actual_demand_mw"), 1).alias("min_served_mw"),
    F.round(F.max("actual_demand_mw"), 1).alias("max_served_mw"),
).first()

if uri_stats:
    print(f"Uri Min Temp         : {uri_stats['min_temp_f']}°F")
    print(f"Uri Minimum Observed Demand: {uri_stats['min_served_mw']:,.0f} MW (served demand; not unconstrained demand)")
    print("Interpretation             : Gold marks the emergency window; quantify unserved load only from an attributed external source.")

# COMMAND ----------
# DBTITLE 1,2. Summer 2023 Heatwave & AC Saturation Linearity Test
df_s23 = (
    df_full.filter(
        (F.col("timestamp_local") >= "2023-07-15") &
        (F.col("timestamp_local") <  "2023-09-01")
    )
)

s23_stats = df_s23.select(
    F.round(F.max("weighted_temperature_f"), 1).alias("max_temp_f"),
    F.round(F.max("actual_demand_mw"), 1).alias("peak_mw"),
    F.count(F.when(F.col("actual_demand_mw") >= 80000, 1)).alias("hours_above_80gw"),
).first()

if s23_stats:
    print(f"Summer 2023 Max Temp   : {s23_stats['max_temp_f']}°F")
    print(f"Summer 2023 Peak Load  : {s23_stats['peak_mw']:,.0f} MW (observed demand; this table alone does not verify supply constraints)")
    print(f"Hours ≥ 80,000 MW      : {s23_stats['hours_above_80gw']:,} hours")

# AC Saturation test: verify if demand flattens above 95°F
df_ext = (
    df_full.filter(F.col("weighted_temperature_f") > 95.0)
    .withColumn("temp_bin_f", F.floor(F.col("weighted_temperature_f") / 2.0) * 2.0)
    .groupBy("temp_bin_f")
    .agg(
        F.count("*").alias("n"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_mw"),
    )
    .filter(F.col("n") >= 5)
    .orderBy("temp_bin_f")
)

display(df_ext)

pdf_ext = df_ext.toPandas()
if len(pdf_ext) >= 2:
    slope, _, r, _, _ = stats.linregress(pdf_ext["temp_bin_f"], pdf_ext["median_mw"])
    print(f"Slope >95°F : +{slope:,.1f} MW/°F (R² = {r**2:.2f})")
    print(f"Interpretation: Linear fit to temperature-bin medians only; descriptive and not evidence of a general causal response.")
