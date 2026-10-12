# Databricks notebook source
# MAGIC %md
# MAGIC # Temporal Seasonality & Diurnal Load Profile Analysis
# MAGIC
# MAGIC **Research Question:**  
# MAGIC How does the ERCOT hourly load shape vary by hour of day, season, and day of week?
# MAGIC Are there statistically distinct intraday demand regimes between summer and winter,
# MAGIC and between weekdays and weekends?
# MAGIC
# MAGIC **Why This Matters for the Gold Feature Layer:**  
# MAGIC - Hour-of-day and day-of-week are among the highest-importance features in all
# MAGIC   published electricity demand forecasting literature (Zheng et al., Hong et al.).
# MAGIC - Seasonal regime interactions (Summer×Hour, Winter×Hour) often outperform
# MAGIC   individual features when encoded as interaction terms in XGBoost.
# MAGIC - Quantifying intraday swing amplitude informs the forecast error budget —
# MAGIC   a model that misidentifies the peak hour by 2 hours incurs a much larger
# MAGIC   operational MW error in summer than in spring.
# MAGIC
# MAGIC **Formal Claim Boundary:**  
# MAGIC All patterns reported here are empirical associations in historical data.
# MAGIC We do not claim that observed diurnal patterns will persist unchanged under
# MAGIC structural economic or energy policy shifts.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import numpy as np
from pyspark.sql import functions as F

SILVER_TABLE = "main.silver.grid_hourly_conformed"

# COMMAND ----------
# DBTITLE 1,Load Analysis Cohort & Calendar Dimensions
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
    .withColumn(
        "day_name",
        F.when(F.col("day_of_week_local") == 1, F.lit("Sunday"))
         .when(F.col("day_of_week_local") == 2, F.lit("Monday"))
         .when(F.col("day_of_week_local") == 3, F.lit("Tuesday"))
         .when(F.col("day_of_week_local") == 4, F.lit("Wednesday"))
         .when(F.col("day_of_week_local") == 5, F.lit("Thursday"))
         .when(F.col("day_of_week_local") == 6, F.lit("Friday"))
         .when(F.col("day_of_week_local") == 7, F.lit("Saturday"))
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Intraday Diurnal Profiles by Meteorological Season
# MAGIC
# MAGIC We compute the **median** hourly demand profile for each meteorological season.
# MAGIC Median is preferred over mean for intraday profiles because it is robust to
# MAGIC extreme events (Uri freeze, summer 2023 records) that would skew seasonal means.
# MAGIC
# MAGIC **Expected Pattern:**
# MAGIC - **Summer:** Single unimodal peak at 16:00–18:00 Central, driven by peak commercial
# MAGIC   HVAC load coinciding with residential return-home air conditioning.
# MAGIC - **Winter:** Bimodal profile — morning warm-up peak (~07:00–09:00) and evening
# MAGIC   lighting/heating peak (~18:00–20:00).
# MAGIC - **Spring/Fall:** Flatter, lower amplitude, less weather-driven load.

# COMMAND ----------
# DBTITLE 1,Compute Seasonal Diurnal Profiles
df_diurnal = (
    df.groupBy("season", "hour_local")
    .agg(
        F.count("*").alias("hour_observations"),
        F.round(F.avg("actual_demand_mw"), 1).alias("mean_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.10)"), 1).alias("p10_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.90)"), 1).alias("p90_demand_mw"),
        F.round(F.avg("weighted_temperature_f"), 1).alias("avg_temp_f"),
    )
    .orderBy("season", "hour_local")
)

display(df_diurnal)

# COMMAND ----------
# DBTITLE 1,Diurnal Shape Diagnostics: Peak Hours & Intraday Swing Amplitudes
pdf_diurnal = df_diurnal.toPandas()

section("DIURNAL LOAD SHAPE DIAGNOSTICS BY METEOROLOGICAL SEASON")
for season in ["Summer", "Winter", "Spring", "Fall"]:
    sub = pdf_diurnal[pdf_diurnal["season"] == season].copy()

    peak_row = sub.loc[sub["median_demand_mw"].idxmax()]
    trough_row = sub.loc[sub["median_demand_mw"].idxmin()]
    swing_mw = peak_row["median_demand_mw"] - trough_row["median_demand_mw"]
    swing_pct = (swing_mw / trough_row["median_demand_mw"]) * 100.0

    # Check bimodality: split day at 13:00 and find the morning and afternoon peaks
    am_peak = sub[sub["hour_local"] < 13].loc[sub[sub["hour_local"] < 13]["median_demand_mw"].idxmax()]
    pm_peak = sub[sub["hour_local"] >= 13].loc[sub[sub["hour_local"] >= 13]["median_demand_mw"].idxmax()]
    bimodal_delta = abs(am_peak["median_demand_mw"] - pm_peak["median_demand_mw"])
    profile_type = "Bimodal" if bimodal_delta < 0.08 * peak_row["median_demand_mw"] else "Unimodal"

    print(f"Season: {season.upper()}")
    print(f"  Profile Type  : {profile_type}")
    print(f"  Overall Peak  : {int(peak_row['hour_local']):02d}:00 local  "
          f"(Median: {peak_row['median_demand_mw']:,.0f} MW, Avg Temp: {peak_row['avg_temp_f']:.1f}°F)")
    print(f"  Overnight Trough : {int(trough_row['hour_local']):02d}:00 local  "
          f"(Median: {trough_row['median_demand_mw']:,.0f} MW)")
    print(f"  Intraday Swing : {swing_mw:,.0f} MW  ({swing_pct:.1f}% amplitude)")
    if profile_type == "Bimodal":
        print(f"  AM Peak       : {int(am_peak['hour_local']):02d}:00 → {am_peak['median_demand_mw']:,.0f} MW")
        print(f"  PM Peak       : {int(pm_peak['hour_local']):02d}:00 → {pm_peak['median_demand_mw']:,.0f} MW")
    print("-" * 70)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Day-of-Week Load Variance
# MAGIC
# MAGIC **Research Question:**  
# MAGIC How much does aggregate ERCOT demand shift across the week?
# MAGIC Is the Monday ramp-up after Sunday significantly different from the midweek plateau?
# MAGIC
# MAGIC **Operational Relevance:**  
# MAGIC Day-of-week encodes commercial and industrial activity cycles. A model that ignores
# MAGIC the weekday/weekend gap will systematically under-forecast on Monday mornings and
# MAGIC over-forecast on Sundays.

# COMMAND ----------
# DBTITLE 1,Weekly Load Profile Aggregation
df_dow = (
    df.groupBy("day_of_week_local", "day_name")
    .agg(
        F.count("*").alias("hour_count"),
        F.round(F.avg("actual_demand_mw"), 1).alias("mean_demand_mw"),
        F.round(F.stddev("actual_demand_mw"), 1).alias("stddev_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.25)"), 1).alias("q1_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.75)"), 1).alias("q3_demand_mw"),
    )
    .orderBy("day_of_week_local")
)

display(df_dow)

# COMMAND ----------
# DBTITLE 1,Weekday vs. Weekend Load Gap Quantification
pdf_dow = df_dow.toPandas()

# Spark day_of_week_local: 1=Sunday, 2=Monday ... 6=Friday, 7=Saturday
weekday_mask = pdf_dow["day_of_week_local"].isin([2, 3, 4, 5, 6])  # Mon–Fri
weekend_mask = pdf_dow["day_of_week_local"].isin([1, 7])           # Sun, Sat

weekday_mean = pdf_dow[weekday_mask]["mean_demand_mw"].mean()
weekend_mean = pdf_dow[weekend_mask]["mean_demand_mw"].mean()
gap_mw = weekday_mean - weekend_mean
gap_pct = (gap_mw / weekend_mean) * 100.0

section("WEEKDAY vs. WEEKEND DEMAND GAP")
print(f"Weekday Average Demand (Mon–Fri) : {weekday_mean:,.1f} MW")
print(f"Weekend Average Demand (Sat–Sun) : {weekend_mean:,.1f} MW")
print(f"Absolute Gap                     : {gap_mw:,.1f} MW")
print(f"Relative Gap                     : {gap_pct:.1f}% higher load on weekdays vs. weekends")
print()
print("Per-day mean demand:")
for _, row in pdf_dow.iterrows():
    bar = "█" * int(row["mean_demand_mw"] / 2000)
    print(f"  {row['day_name']:<10}: {row['mean_demand_mw']:,.0f} MW  {bar}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Monthly Demand Trend & Annual Seasonal Cycle
# MAGIC
# MAGIC **Research Question:**  
# MAGIC How does aggregate demand vary month to month? Which months drive the system peak
# MAGIC and which represent the annual demand trough?
# MAGIC
# MAGIC **Operational Relevance:**  
# MAGIC Monthly patterns inform the 7-day peak-risk planning overlay (Phase 3 deliverable 2)
# MAGIC and help contextualise forecast errors reported to stakeholders.

# COMMAND ----------
# DBTITLE 1,Monthly Demand Distribution
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

df_monthly = (
    df.groupBy("month_local")
    .agg(
        F.count("*").alias("hour_count"),
        F.round(F.avg("actual_demand_mw"), 1).alias("mean_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.95)"), 1).alias("p95_demand_mw"),
        F.round(F.max("actual_demand_mw"), 1).alias("max_demand_mw"),
        F.round(F.avg("weighted_temperature_f"), 1).alias("avg_temp_f"),
    )
    .orderBy("month_local")
)

display(df_monthly)

# COMMAND ----------
# DBTITLE 1,Annual Cycle Summary
pdf_monthly = df_monthly.toPandas()
pdf_monthly["month_name"] = pdf_monthly["month_local"].map(MONTH_NAMES)

peak_month = pdf_monthly.loc[pdf_monthly["median_demand_mw"].idxmax()]
trough_month = pdf_monthly.loc[pdf_monthly["median_demand_mw"].idxmin()]

section("ANNUAL SEASONAL DEMAND CYCLE")
print(f"Peak Demand Month  : {peak_month['month_name']}  "
      f"(Median: {peak_month['median_demand_mw']:,.0f} MW, Avg Temp: {peak_month['avg_temp_f']:.1f}°F)")
print(f"Trough Demand Month: {trough_month['month_name']}  "
      f"(Median: {trough_month['median_demand_mw']:,.0f} MW, Avg Temp: {trough_month['avg_temp_f']:.1f}°F)")
print(f"Annual Peak-to-Trough Swing: "
      f"{peak_month['median_demand_mw'] - trough_month['median_demand_mw']:,.0f} MW")
print()
print("Monthly median demand profile:")
for _, row in pdf_monthly.iterrows():
    bar = "█" * int(row["median_demand_mw"] / 2000)
    print(f"  {row['month_name']}: {row['median_demand_mw']:,.0f} MW  {bar}")
