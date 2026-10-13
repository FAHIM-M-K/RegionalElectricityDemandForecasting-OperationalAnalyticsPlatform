# Databricks notebook source
# MAGIC %md
# MAGIC # EIA Day-Ahead Forecast Benchmark Profiling
# MAGIC
# MAGIC **Research Question:**  
# MAGIC How accurate is ERCOT's published EIA day-ahead demand forecast (`day_ahead_forecast_mw`),
# MAGIC and where does it fail — by season, by temperature regime, and by hour of day?
# MAGIC
# MAGIC **Why This Comes Before Building Our Model:**  
# MAGIC The EIA Day-Ahead Forecast (`DF`) is already a professional grid operator forecast —
# MAGIC our ML champion model must **beat it in a fair walk-forward evaluation** to justify
# MAGIC its existence. Without first understanding the baseline's strengths and failure modes,
# MAGIC we cannot make an informed claim that our model adds operational value.
# MAGIC
# MAGIC **Baseline Metrics Computed:**
# MAGIC - **MAE** (Mean Absolute Error) — primary metric; in MW, directly interpretable by operators
# MAGIC - **RMSE** (Root Mean Squared Error) — penalises large spikes; relevant for peak-hour risk
# MAGIC - **MAPE** (Mean Absolute Percentage Error) — normalised; comparable across load levels
# MAGIC - **Bias** (Mean Signed Error) — positive = systematic overforecast, negative = underforecast
# MAGIC
# MAGIC **These metrics become the champion model success criteria in the Gold/ML phase.**
# MAGIC
# MAGIC **Formal Claim Boundary:**  
# MAGIC EIA forecast performance reflects the ERCO balancing authority's internal operational
# MAGIC model, not a naive statistical baseline. We treat it as a professional benchmark, not
# MAGIC as a simple heuristic.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

SILVER_TABLE = "main.silver.grid_hourly_conformed"

# COMMAND ----------
# DBTITLE 1,Load Analysis Cohort & Derive Season
df = (
    spark.table(SILVER_TABLE)
    .filter(
        F.col("actual_demand_mw").isNotNull() &
        (F.col("is_valid_demand") == True) &
        (F.col("weighted_temperature_f").isNotNull())
    )
    .withColumn(
        "season",
        F.when(F.col("month_local").isin(12, 1, 2), F.lit("Winter"))
         .when(F.col("month_local").isin(3, 4, 5), F.lit("Spring"))
         .when(F.col("month_local").isin(6, 7, 8), F.lit("Summer"))
         .otherwise(F.lit("Fall"))
    )
)

# Restrict to hours where both actual demand and day-ahead forecast are present
df_bench = (
    df.filter(F.col("day_ahead_forecast_mw").isNotNull())
    .withColumn("error_mw",     F.col("day_ahead_forecast_mw") - F.col("actual_demand_mw"))
    .withColumn("abs_error_mw", F.abs(F.col("error_mw")))
    .withColumn("ape_pct",      (F.abs(F.col("error_mw")) / F.col("actual_demand_mw")) * 100.0)
    .withColumn("sq_error",     F.pow(F.col("error_mw"), 2))
)

bench_count = df_bench.count()
print(f"Benchmark evaluation horizon: {bench_count:,} synchronized hourly observations")
print("(Hours where both EIA actual demand and EIA day-ahead forecast are present)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Overall Baseline Scorecard
# MAGIC
# MAGIC This is the **target card** our ML model must beat in walk-forward evaluation.
# MAGIC Print it clearly — it will be referenced verbatim in the interview story.

# COMMAND ----------
# DBTITLE 1,Global EIA Day-Ahead Forecast Performance Scorecard
bench_global = df_bench.select(
    F.count("*").alias("evaluated_hours"),
    F.round(F.avg("abs_error_mw"), 1).alias("mae_mw"),
    F.round(F.sqrt(F.avg("sq_error")), 1).alias("rmse_mw"),
    F.round(F.avg("ape_pct"), 3).alias("mape_pct"),
    F.round(F.avg("error_mw"), 1).alias("mean_bias_mw"),
    F.round(F.expr("percentile_approx(abs_error_mw, 0.50)"), 1).alias("median_abs_error_mw"),
    F.round(F.expr("percentile_approx(abs_error_mw, 0.90)"), 1).alias("p90_abs_error_mw"),
    F.round(F.expr("percentile_approx(abs_error_mw, 0.95)"), 1).alias("p95_abs_error_mw"),
    F.round(F.max("abs_error_mw"), 1).alias("max_abs_error_mw"),
).first()

bias_direction = "Overforecast" if bench_global["mean_bias_mw"] > 0 else "Underforecast"

print("=" * 70)
print("  EIA DAY-AHEAD FORECAST — GLOBAL BENCHMARK SCORECARD")
print("=" * 70)
print(f"  Evaluated Observations   : {bench_global['evaluated_hours']:,} hours")
print()
print(f"  MAE                      : {bench_global['mae_mw']:>10,.1f} MW   ← primary target for ML champion")
print(f"  RMSE                     : {bench_global['rmse_mw']:>10,.1f} MW")
print(f"  MAPE                     : {bench_global['mape_pct']:>10.3f}%")
print(f"  Mean Bias                : {bench_global['mean_bias_mw']:>+10,.1f} MW   ({bias_direction})")
print()
print(f"  Median |Error|           : {bench_global['median_abs_error_mw']:>10,.1f} MW")
print(f"  90th Percentile |Error|  : {bench_global['p90_abs_error_mw']:>10,.1f} MW")
print(f"  95th Percentile |Error|  : {bench_global['p95_abs_error_mw']:>10,.1f} MW")
print(f"  Maximum |Error| Spike    : {bench_global['max_abs_error_mw']:>10,.1f} MW")
print()
print(f"  ML Champion Model Success Criteria:")
print(f"    Beat {bench_global['mae_mw']:.0f} MW MAE and {bench_global['mape_pct']:.2f}% MAPE on walk-forward 48h evaluation.")
print(f"    Specifically reduce 90th-percentile error ({bench_global['p90_abs_error_mw']:.0f} MW) — this is peak-risk territory.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Baseline Performance by Season
# MAGIC
# MAGIC The EIA forecast is likely strongest in mild-weather seasons (Spring, Fall)
# MAGIC where load is temperature-stable, and weakest in extreme-temperature seasons
# MAGIC (Summer, Winter) where demand is more volatile and harder to predict.

# COMMAND ----------
# DBTITLE 1,EIA Forecast Error Decomposed by Season
df_bench_season = (
    df_bench.groupBy("season")
    .agg(
        F.count("*").alias("hours"),
        F.round(F.avg("abs_error_mw"), 1).alias("mae_mw"),
        F.round(F.sqrt(F.avg("sq_error")), 1).alias("rmse_mw"),
        F.round(F.avg("ape_pct"), 3).alias("mape_pct"),
        F.round(F.avg("error_mw"), 1).alias("bias_mw"),
        F.round(F.expr("percentile_approx(abs_error_mw, 0.90)"), 1).alias("p90_error_mw"),
    )
    .orderBy("mae_mw", ascending=False)
)

display(df_bench_season)

pdf_season = df_bench_season.toPandas()
print("\n" + "=" * 70)
print("  SEASONAL BENCHMARK DECOMPOSITION")
print("=" * 70)
print(f"{'Season':<10} {'MAE (MW)':>10} {'RMSE (MW)':>11} {'MAPE %':>8} {'Bias (MW)':>10}")
print("-" * 55)
for _, row in pdf_season.iterrows():
    print(f"{row['season']:<10} {row['mae_mw']:>10,.1f} {row['rmse_mw']:>11,.1f} "
          f"{row['mape_pct']:>8.3f} {row['bias_mw']:>+10,.1f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Baseline Performance by Hour of Day
# MAGIC
# MAGIC Does the EIA forecast fail disproportionately at the critical afternoon peak window
# MAGIC (14:00–18:00 Central), which is when grid operators most need forecast accuracy
# MAGIC for dispatch and reserve planning?

# COMMAND ----------
# DBTITLE 1,EIA Forecast Error Decomposed by Hour of Day
df_bench_hour = (
    df_bench.groupBy("hour_local")
    .agg(
        F.count("*").alias("hours"),
        F.round(F.avg("abs_error_mw"), 1).alias("mae_mw"),
        F.round(F.sqrt(F.avg("sq_error")), 1).alias("rmse_mw"),
        F.round(F.avg("ape_pct"), 3).alias("mape_pct"),
        F.round(F.avg("error_mw"), 1).alias("bias_mw"),
    )
    .orderBy("hour_local")
)

display(df_bench_hour)

pdf_hour = df_bench_hour.toPandas()
peak_error_hour = pdf_hour.loc[pdf_hour["mae_mw"].idxmax()]
best_error_hour = pdf_hour.loc[pdf_hour["mae_mw"].idxmin()]

print("\n" + "=" * 70)
print("  INTRADAY ERROR PROFILE — HIGHEST/LOWEST FORECAST DIFFICULTY")
print("=" * 70)
print(f"Worst Forecast Accuracy Hour : {int(peak_error_hour['hour_local']):02d}:00 local  "
      f"(MAE: {peak_error_hour['mae_mw']:,.0f} MW)")
print(f"Best Forecast Accuracy Hour  : {int(best_error_hour['hour_local']):02d}:00 local  "
      f"(MAE: {best_error_hour['mae_mw']:,.0f} MW)")

# Visualize Intraday Forecast Error Profile
plt.figure(figsize=(10, 4.5))
plt.plot(pdf_hour["hour_local"], pdf_hour["mae_mw"], marker="o", color="#d62728", lw=2, label="MAE (MW)")
plt.plot(pdf_hour["hour_local"], pdf_hour["rmse_mw"], marker="s", color="#1f77b4", lw=1.5, linestyle="--", label="RMSE (MW)")
plt.title("EIA Day-Ahead Forecast Error by Hour of Day", fontsize=13, fontweight="bold")
plt.xlabel("Hour of Day (Local Central Time)", fontsize=11)
plt.ylabel("Forecast Error (MW)", fontsize=11)
plt.xticks(range(0, 24))
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(frameon=True)
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Baseline Performance by Temperature Decile
# MAGIC
# MAGIC Does EIA forecast accuracy degrade during extreme temperature events (the bottom 10%
# MAGIC cold hours and top 10% hot hours)? These are precisely the operationally critical
# MAGIC periods — peak demand hours where a bad forecast leads to reserve shortfalls.

# COMMAND ----------
# DBTITLE 1,EIA Forecast Error Conditioned on Temperature Deciles
df_bench_decile = (
    df_bench.withColumn(
        "temp_decile",
        F.ntile(10).over(Window.orderBy("weighted_temperature_f"))
    )
    .groupBy("temp_decile")
    .agg(
        F.round(F.min("weighted_temperature_f"), 1).alias("min_temp_f"),
        F.round(F.max("weighted_temperature_f"), 1).alias("max_temp_f"),
        F.count("*").alias("hours"),
        F.round(F.avg("abs_error_mw"), 1).alias("mae_mw"),
        F.round(F.sqrt(F.avg("sq_error")), 1).alias("rmse_mw"),
        F.round(F.avg("ape_pct"), 3).alias("mape_pct"),
        F.round(F.avg("error_mw"), 1).alias("bias_mw"),
    )
    .orderBy("temp_decile")
)

display(df_bench_decile)

pdf_decile = df_bench_decile.toPandas()

print("\n" + "=" * 70)
print("  TEMPERATURE DECILE ERROR PROFILE")
print("=" * 70)
print(f"{'Decile':>7} {'Temp Range (°F)':>18} {'MAE (MW)':>10} {'MAPE %':>8} {'Bias (MW)':>10}")
print("-" * 60)
for _, row in pdf_decile.iterrows():
    print(f"{int(row['temp_decile']):>7} {row['min_temp_f']:>6.1f}–{row['max_temp_f']:<9.1f} "
          f"{row['mae_mw']:>10,.1f} {row['mape_pct']:>8.3f} {row['bias_mw']:>+10,.1f}")

# Identify coldest and hottest decile performance
if not pdf_decile.empty:
    coldest = pdf_decile[pdf_decile["temp_decile"] == pdf_decile["temp_decile"].min()].iloc[0]
    hottest = pdf_decile[pdf_decile["temp_decile"] == pdf_decile["temp_decile"].max()].iloc[0]
    decile_5 = pdf_decile[pdf_decile["temp_decile"] == 5]
    median_decile = decile_5.iloc[0] if not decile_5.empty else pdf_decile.iloc[len(pdf_decile) // 2]

    median_mae = median_decile['mae_mw'] if median_decile['mae_mw'] > 0 else 1.0
    print()
    print("Key Finding:")
    print(f"  Coldest Decile (≤ {coldest['max_temp_f']:.0f}°F)  MAE: {coldest['mae_mw']:,.0f} MW "
          f"  ({coldest['mae_mw'] / median_mae:.1f}x median-decile error)")
    print(f"  Hottest Decile (≥ {hottest['min_temp_f']:.0f}°F)  MAE: {hottest['mae_mw']:,.0f} MW "
          f"  ({hottest['mae_mw'] / median_mae:.1f}x median-decile error)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Formal Benchmark Summary — Interview-Ready Claims
# MAGIC
# MAGIC This section consolidates the findings into concise, defensible claims for
# MAGIC the portfolio interview and the model evaluation section of the project.

# COMMAND ----------
# DBTITLE 1,Final Benchmark Summary & ML Target Definition
print("\n" + "=" * 70)
print("  BENCHMARK SUMMARY — EIA DAY-AHEAD FORECAST vs. ML CHAMPION TARGET")
print("=" * 70)
print(f"Benchmark (EIA Day-Ahead Forecast):")
print(f"  Overall MAE   : {bench_global['mae_mw']:,.0f} MW")
print(f"  Overall RMSE  : {bench_global['rmse_mw']:,.0f} MW")
print(f"  Overall MAPE  : {bench_global['mape_pct']:.2f}%")
print(f"  Overall Bias  : {bench_global['mean_bias_mw']:+,.0f} MW  ({bias_direction})")
print()

worst_season = pdf_season.loc[pdf_season["mae_mw"].idxmax()]
best_season  = pdf_season.loc[pdf_season["mae_mw"].idxmin()]
print(f"  Hardest Season for EIA  : {worst_season['season']} (MAE: {worst_season['mae_mw']:,.0f} MW)")
print(f"  Easiest Season for EIA  : {best_season['season']} (MAE: {best_season['mae_mw']:,.0f} MW)")
print()
print("Our ML Champion Model must:")
print(f"  1. Reduce overall MAE below  {bench_global['mae_mw'] * 0.85:.0f} MW (>15% improvement)")
print(f"  2. Reduce overall MAPE below {bench_global['mape_pct'] * 0.85:.2f}%")
print(f"  3. Materially reduce 90th-percentile error below {bench_global['p90_abs_error_mw'] * 0.85:.0f} MW")
print(f"     — this directly reduces peak-risk exposure for operators.")
print()
print("Non-Claim:")
print("  We do not claim the EIA forecast is a naive or weak baseline.")
print("  It is a professional grid operator forecast. Beating it by ≥15% is")
print("  meaningful operational improvement, not just statistical overfitting.")
