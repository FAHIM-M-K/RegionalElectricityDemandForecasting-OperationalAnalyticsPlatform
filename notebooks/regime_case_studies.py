# Databricks notebook source
# MAGIC %md
# MAGIC # Grid Stress Regime Case Studies: Uri & Summer 2023
# MAGIC
# MAGIC **Research Question:**  
# MAGIC Which historical periods in our dataset reflect **supply-constrained** (artificially capped)
# MAGIC served load rather than unconstrained consumer demand — and what are the implications
# MAGIC for machine learning training?
# MAGIC
# MAGIC **Why This Matters:**  
# MAGIC Machine learning models trained on electricity demand assume `actual_demand_mw` represents
# MAGIC what consumers **wanted** to consume at that hour. This assumption breaks catastrophically
# MAGIC during grid emergencies when involuntary load shedding forces demand below what it would
# MAGIC otherwise be. Including these distorted observations in model training without flagging
# MAGIC them biases learned relationships — particularly temperature-demand coefficients during
# MAGIC extreme cold, which is exactly the regime we most need to extrapolate to for risk.
# MAGIC
# MAGIC **Two Regimes Examined:**
# MAGIC
# MAGIC | Event | Period | Nature | Training Risk |
# MAGIC |---|---|---|---|
# MAGIC | Winter Storm Uri | Feb 13–19, 2021 | Supply curtailment (involuntary blackouts ~20 GW shed) | Demand artificially suppressed — cold-weather response is underestimated |
# MAGIC | Summer 2023 Heatwave | Jul–Aug 2023 | Unconstrained demand record (no load shedding) | True cooling saturation ceiling — model should reflect this |
# MAGIC
# MAGIC **Formal Claim Boundary:**  
# MAGIC We do not claim that ERCOT grid management decisions were correct or incorrect.
# MAGIC We document the data implications for our forecasting model exclusively.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import numpy as np
from pyspark.sql import functions as F

SILVER_TABLE = "main.silver.grid_hourly_conformed"

def section(title: str, width: int = 70) -> None:
    bar = "=" * width
    print(f"\n{bar}\n  {title}\n{bar}")

# COMMAND ----------
# DBTITLE 1,Load Full Conformed Table (Unrestricted for Regime Analysis)
# Note: We load the FULL silver table here — not the is_valid_demand-filtered cohort —
# because Uri observations were recorded and are "valid" data in the Silver sense
# (they passed physical bounds), but they are regime-distorted for modelling purposes.
# The regime analysis notebook explicitly characterises this distinction.
df_full = spark.table(SILVER_TABLE).filter(
    F.col("actual_demand_mw").isNotNull()
)

print(f"Full historical cohort (all available hours): {df_full.count():,} records")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Case Study 1: Winter Storm Uri — February 13–19, 2021
# MAGIC
# MAGIC ### Background
# MAGIC During the week of February 13–19, 2021, a historic polar vortex collapsed brought
# MAGIC temperatures as low as **−2°F in DFW and +12°F in Houston** — temperatures well outside
# MAGIC normal Texas building thermal tolerances. ERCOT's grid frequency dropped to within
# MAGIC **4.5 seconds** of a complete blackout (~59.3 Hz). Emergency firm load shedding
# MAGIC removed up to **20,000 MW** of load in rolling intervals to protect grid stability.
# MAGIC
# MAGIC ### Data Implication
# MAGIC EIA `actual_demand_mw` records the **served load** — what was actually consumed —
# MAGIC not the unconstrained consumer demand. During Uri, served load was artificially
# MAGIC capped at ~50,000 MW while ERCOT internally estimated unconstrained demand at
# MAGIC **>76,000 MW** had all consumers been able to receive power.
# MAGIC
# MAGIC ### Modelling Decision
# MAGIC Uri observations should be **flagged** in the Gold feature table (`is_grid_emergency`)
# MAGIC rather than silently dropped. During training, models can be evaluated with and without
# MAGIC Uri hours to measure the regime's influence on cold-weather response coefficients.

# COMMAND ----------
# DBTITLE 1,Uri Period: Hourly Temperature & Served Demand Profile
URI_START = "2021-02-13"
URI_END   = "2021-02-20"

df_uri = (
    df_full.filter(
        (F.col("timestamp_local") >= URI_START) &
        (F.col("timestamp_local") <  URI_END)
    )
    .select(
        "timestamp_local",
        "actual_demand_mw",
        "day_ahead_forecast_mw",
        "weighted_temperature_f",
        "temp_dfw_c",
        "temp_houston_c",
        "is_valid_demand",
    )
    .orderBy("timestamp_local")
)

display(df_uri)

# COMMAND ----------
# DBTITLE 1,Uri Regime Summary Statistics
uri_stats = df_uri.select(
    F.count("*").alias("total_uri_hours"),
    F.round(F.min("weighted_temperature_f"), 1).alias("min_regional_temp_f"),
    F.round(F.min("temp_dfw_c"), 1).alias("min_dfw_temp_c"),
    F.round(F.min("temp_houston_c"), 1).alias("min_houston_temp_c"),
    F.round(F.min("actual_demand_mw"), 1).alias("min_served_demand_mw"),
    F.round(F.max("actual_demand_mw"), 1).alias("max_served_demand_mw"),
    F.round(F.avg("actual_demand_mw"), 1).alias("avg_served_demand_mw"),
    F.round(F.max("day_ahead_forecast_mw"), 1).alias("max_da_forecast_mw"),
).first()

section("WINTER STORM URI (FEB 2021) REGIME AUDIT")
if uri_stats and uri_stats["total_uri_hours"] and uri_stats["total_uri_hours"] > 0:
    print(f"Total Hours Examined          : {uri_stats['total_uri_hours']}")
    print(f"Min Regional Temperature      : {uri_stats['min_regional_temp_f']:.1f}°F")
    print(f"Min DFW Temperature           : {uri_stats['min_dfw_temp_c']:.1f}°C  "
          f"({uri_stats['min_dfw_temp_c'] * 9/5 + 32:.1f}°F)")
    print(f"Min Houston Temperature       : {uri_stats['min_houston_temp_c']:.1f}°C  "
          f"({uri_stats['min_houston_temp_c'] * 9/5 + 32:.1f}°F)")
    print()
    print("Served Load During Uri Blackout Period:")
    print(f"  Minimum Served Demand       : {uri_stats['min_served_demand_mw']:,.0f} MW  ← artificially curtailed")
    print(f"  Maximum Served Demand       : {uri_stats['max_served_demand_mw']:,.0f} MW")
    print(f"  ERCOT-Estimated True Need   : ~76,000 MW  (PUCT testimony, March 2021)")
    print(f"  Estimated Shed Load         : ~{76000 - uri_stats['max_served_demand_mw']:,.0f} MW at peak blackout")
    print()
    print("Day-Ahead Forecast (pre-event):")
    print(f"  Maximum DA Forecast         : {uri_stats['max_da_forecast_mw']:,.0f} MW")
    print("  → EIA forecast was also not predictive of the true demand surge.")
else:
    print("No records available in dataset for the Feb 13–20, 2021 window.")
print()
print("Gold Layer Recommendation:")
print("  → Add `is_grid_emergency = True` flag for Feb 15–18 2021 in the Gold feature table.")
print("  → Run ablation study: train with Uri included vs. excluded, compare cold-weather RMSE.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Case Study 2: Summer 2023 Heatwave — All-Time Peak Demand Records
# MAGIC
# MAGIC ### Background
# MAGIC During July–August 2023, a prolonged heat dome kept temperatures above 100°F across
# MAGIC Texas for consecutive weeks. ERCOT set **10 new all-time demand records** during this
# MAGIC period, peaking above **85,000 MW**. Critically, **no firm load shedding occurred** —
# MAGIC this is unconstrained consumer demand at the system's physical ceiling.
# MAGIC
# MAGIC ### Data Implication
# MAGIC Unlike Uri, Summer 2023 observations are valid unconstrained demand and should
# MAGIC be retained in training. However, the prolonged extreme-heat plateau tests whether
# MAGIC air conditioning demand **saturates** at high temperatures (i.e., is the curve still
# MAGIC linear above 105°F, or does it flatten as all available AC units are already running?).

# COMMAND ----------
# DBTITLE 1,Summer 2023 Heatwave: Demand & Temperature Profile
S23_START = "2023-07-15"
S23_END   = "2023-09-01"

df_s23 = (
    df_full.filter(
        (F.col("timestamp_local") >= S23_START) &
        (F.col("timestamp_local") <  S23_END)
    )
    .select(
        "timestamp_local",
        "actual_demand_mw",
        "day_ahead_forecast_mw",
        "weighted_temperature_f",
        "is_valid_demand",
    )
    .orderBy("timestamp_local")
)

display(df_s23)

# COMMAND ----------
# DBTITLE 1,Summer 2023 Regime Summary Statistics
s23_stats = df_s23.select(
    F.count("*").alias("total_hours"),
    F.round(F.max("weighted_temperature_f"), 1).alias("max_regional_temp_f"),
    F.round(F.max("actual_demand_mw"), 1).alias("peak_demand_mw"),
    F.round(F.avg("actual_demand_mw"), 1).alias("avg_demand_mw"),
    F.count(F.when(F.col("actual_demand_mw") >= 80000, 1)).alias("hours_above_80gw"),
    F.count(F.when(F.col("actual_demand_mw") >= 85000, 1)).alias("hours_above_85gw"),
    F.round(F.avg(F.when(F.col("actual_demand_mw") >= 80000, F.col("weighted_temperature_f"))), 1).alias("avg_temp_during_80gw_hours"),
).first()

section("SUMMER 2023 HEATWAVE REGIME AUDIT")
if s23_stats and s23_stats["total_hours"] and s23_stats["total_hours"] > 0:
    print(f"Period Examined               : {S23_START} → {S23_END}")
    print(f"Total Hours in Window         : {s23_stats['total_hours']:,}")
    print(f"Maximum Regional Temperature  : {s23_stats['max_regional_temp_f']:.1f}°F")
    print(f"All-Time Peak Demand Recorded : {s23_stats['peak_demand_mw']:,.0f} MW")
    print(f"Average Demand in Period      : {s23_stats['avg_demand_mw']:,.0f} MW")
    print(f"Hours Above 80,000 MW         : {s23_stats['hours_above_80gw']:,} hours")
    print(f"Hours Above 85,000 MW         : {s23_stats['hours_above_85gw']:,} hours")
    avg_temp_80 = s23_stats['avg_temp_during_80gw_hours']
    print(f"Avg Temp During 80GW+ Hours   : {avg_temp_80:.1f}°F" if avg_temp_80 is not None else "Avg Temp During 80GW+ Hours   : N/A")
    print()
    print("No firm load shedding was required → observations are valid unconstrained demand.")
else:
    print("No records available in dataset for the Summer 2023 window.")

# COMMAND ----------
# DBTITLE 1,AC Saturation Test: High-Temperature Demand Linearity Check
# Pull extreme heat hours (>95°F) to driver to check if demand curve flattens
df_extreme_heat = (
    df_full.filter(F.col("weighted_temperature_f") > 95.0)
    .withColumn("temp_bin_f", F.floor(F.col("weighted_temperature_f") / 2.0) * 2.0)
    .groupBy("temp_bin_f")
    .agg(
        F.count("*").alias("hour_count"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_demand_mw"),
        F.round(F.avg("actual_demand_mw"), 1).alias("mean_demand_mw"),
    )
    .filter(F.col("hour_count") >= 5)
    .orderBy("temp_bin_f")
)

display(df_extreme_heat)

# Pull to driver for slope analysis
from scipy import stats
pdf_extreme = df_extreme_heat.toPandas()

section("HIGH-TEMPERATURE DEMAND LINEARITY (AC SATURATION TEST)")
if len(pdf_extreme) >= 2:
    slope, intercept, r, p, se = stats.linregress(
        pdf_extreme["temp_bin_f"], pdf_extreme["median_demand_mw"]
    )
    print(f"Temperature Range Analysed  : {pdf_extreme['temp_bin_f'].min():.0f}°F – {pdf_extreme['temp_bin_f'].max():.0f}°F")
    print(f"Linear Slope                : +{slope:,.1f} MW / °F above 95°F")
    print(f"R²                          : {r**2:.3f}")
    print(f"p-value                     : {p:.4e}")
    print()
    if r**2 > 0.85:
        print("Finding: Demand remains strongly linear above 95°F (R² > 0.85).")
        print("No evidence of saturation — AC demand continues to climb at extreme heat.")
        print("Implication: Linear + quadratic temperature features remain valid up to observed maximum.")
    elif r**2 > 0.5:
        print("Finding: Moderate linearity above 95°F — some evidence of demand plateau.")
        print("Implication: Consider a capped degree-hour feature (max(CDH, threshold)) in Gold.")
    else:
        print("Finding: Weak linearity above 95°F — demand curve appears to flatten at extreme heat.")
        print("Implication: Piecewise or quantile-based temperature feature may better capture saturation.")
else:
    print("Insufficient extreme heat records (>95°F) for linear regression (minimum 2 bins required).")
