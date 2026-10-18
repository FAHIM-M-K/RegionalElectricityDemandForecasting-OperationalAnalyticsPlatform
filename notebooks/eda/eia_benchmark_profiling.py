# Databricks notebook source
# MAGIC %md
# MAGIC # EIA Day-Ahead Forecast Benchmark Profiling
# MAGIC
# MAGIC **Core Objective:** Profile ERCOT's published EIA Day-Ahead Forecast (`day_ahead_forecast_mw`)
# MAGIC across seasons, hours, and temperature extremes as a retrospective reference benchmark.
# MAGIC The conformed historical table does not preserve the original issue-time vintage for every
# MAGIC EIA forecast, so this profile is not an exact same-origin operational comparison.
# MAGIC
# MAGIC **Model promotion:** Candidate promotion is decided by the rolling, time-ordered naive
# MAGIC baselines in the validation workflow. Matched EIA hours are reported separately when coverage
# MAGIC is available; this descriptive notebook does not set a promotion threshold.

# COMMAND ----------
# DBTITLE 1,Imports & Synchronized Horizon Load
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

SILVER_TABLE = "main.silver.grid_hourly_conformed"

df_bench = (
    spark.table(SILVER_TABLE)
    .filter(
        F.col("actual_demand_mw").isNotNull() &
        (F.col("is_valid_demand") == True) &
        F.col("day_ahead_forecast_mw").isNotNull()
    )
    .withColumn("error_mw", F.col("day_ahead_forecast_mw") - F.col("actual_demand_mw"))
    .withColumn("abs_error_mw", F.abs(F.col("error_mw")))
    .withColumn("ape_pct", (F.abs(F.col("error_mw")) / F.col("actual_demand_mw")) * 100.0)
    .withColumn("sq_error", F.pow(F.col("error_mw"), 2))
    .withColumn(
        "season",
        F.when(F.col("month_local").isin(12, 1, 2), F.lit("Winter"))
         .when(F.col("month_local").isin(3, 4, 5), F.lit("Spring"))
         .when(F.col("month_local").isin(6, 7, 8), F.lit("Summer"))
         .otherwise(F.lit("Fall"))
    )
)

print(f"Synchronized Benchmark Observations: {df_bench.count():,} hours")

# COMMAND ----------
# DBTITLE 1,1. Global Benchmark Scorecard
g = df_bench.select(
    F.count("*").alias("hours"),
    F.round(F.avg("abs_error_mw"), 1).alias("mae"),
    F.round(F.sqrt(F.avg("sq_error")), 1).alias("rmse"),
    F.round(F.avg("ape_pct"), 2).alias("mape"),
    F.round(F.avg("error_mw"), 1).alias("bias"),
    F.round(F.expr("percentile_approx(abs_error_mw, 0.90)"), 1).alias("p90"),
).first()

if g:
    print(f"Retrospective EIA reference MAE : {g['mae']:,.1f} MW")
    print(f"EIA Day-Ahead RMSE : {g['rmse']:,.1f} MW")
    print(f"EIA Day-Ahead MAPE : {g['mape']:.2f}%")
    print(f"EIA Mean Bias      : {g['bias']:+,.1f} MW ({'Overforecast' if g['bias'] > 0 else 'Underforecast'})")
    print(f"90th Percentile |E|: {g['p90']:,.1f} MW")

# COMMAND ----------
# DBTITLE 1,2. Error Profile by Season & Hour of Day
df_season = (
    df_bench.groupBy("season")
    .agg(
        F.count("*").alias("hours"),
        F.round(F.avg("abs_error_mw"), 1).alias("mae"),
        F.round(F.avg("ape_pct"), 2).alias("mape"),
        F.round(F.avg("error_mw"), 1).alias("bias"),
    )
    .orderBy("mae", ascending=False)
)

display(df_season)

df_hour = (
    df_bench.groupBy("hour_local")
    .agg(
        F.round(F.avg("abs_error_mw"), 1).alias("mae"),
        F.round(F.sqrt(F.avg("sq_error")), 1).alias("rmse"),
    )
    .orderBy("hour_local")
)

pdf_hour = df_hour.toPandas()
plt.figure(figsize=(9, 4))
plt.plot(pdf_hour["hour_local"], pdf_hour["mae"], marker="o", color="#d62728", lw=2, label="MAE (MW)")
plt.plot(pdf_hour["hour_local"], pdf_hour["rmse"], marker="s", color="#1f77b4", lw=1.5, linestyle="--", label="RMSE (MW)")
plt.title("EIA Forecast Error by Hour of Day", fontsize=12, fontweight="bold")
plt.xlabel("Hour of Day (Local Time)")
plt.ylabel("Forecast Error (MW)")
plt.xticks(range(0, 24))
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------
# DBTITLE 1,3. Error Degradation by Temperature Decile
df_decile = (
    df_bench.withColumn("temp_decile", F.ntile(10).over(Window.orderBy("weighted_temperature_f")))
    .groupBy("temp_decile")
    .agg(
        F.round(F.min("weighted_temperature_f"), 1).alias("min_temp_f"),
        F.round(F.max("weighted_temperature_f"), 1).alias("max_temp_f"),
        F.round(F.avg("abs_error_mw"), 1).alias("mae"),
    )
    .orderBy("temp_decile")
)

display(df_decile)
