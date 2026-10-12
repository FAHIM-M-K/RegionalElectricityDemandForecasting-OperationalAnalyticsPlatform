# Databricks notebook source
# MAGIC %md
# MAGIC # Empirical Temperature–Demand Curve & Thermal Balance Point Discovery
# MAGIC
# MAGIC **Research Question:**  
# MAGIC Does the Texas aggregate grid (ERCOT) balance at the conventional **65°F (18.33°C)**, or does the
# MAGIC empirical demand minimum occur at a different temperature due to Texas building stock,
# MAGIC humidity levels, and widespread air conditioning penetration?
# MAGIC
# MAGIC **Why This Matters for the Gold Feature Layer:**  
# MAGIC - The Silver layer computes `cdh_65` and `hdh_65` using the 65°F assumption.
# MAGIC - If the empirical balance point ($T_{\text{balance}}$) differs materially, we need custom
# MAGIC   degree-hour features at the observed pivot temperature in addition to the standard ones.
# MAGIC - The asymmetry between cooling and heating thermal gradients determines whether a
# MAGIC   quadratic temperature term or a two-slope piecewise linear feature is more appropriate.
# MAGIC
# MAGIC **Formal Claim Boundary:**  
# MAGIC We report correlation and association, not causation. Temperature changes co-vary with demand;
# MAGIC we make no claims about consumer responsiveness to price or regulatory interventions.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import numpy as np
from scipy import stats
from pyspark.sql import functions as F

SILVER_TABLE = "main.silver.grid_hourly_conformed"
MIN_BIN_HOURS = 15

def section(title: str, width: int = 70) -> None:
    bar = "=" * width
    print(f"\n{bar}\n  {title}\n{bar}")

# COMMAND ----------
# DBTITLE 1,Load Analysis Cohort
# Filter to valid historical records with both demand and weather observations
df = (
    spark.table(SILVER_TABLE)
    .filter(
        F.col("actual_demand_mw").isNotNull() &
        (F.col("is_valid_demand") == True) &
        (F.col("weighted_temperature_f").isNotNull())
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Binned Temperature–Demand Distribution (Non-Parametric)
# MAGIC
# MAGIC We aggregate demand into **2°F temperature bins** to observe the full empirical
# MAGIC distribution (median, 10th/90th percentile) across the observed temperature range.
# MAGIC Bins with fewer than `MIN_BIN_HOURS` observations are excluded as statistically unreliable.

# COMMAND ----------
# DBTITLE 1,Compute Binned Temperature–Demand Statistics
df_bins = (
    df.withColumn("temp_bin_f", F.floor(F.col("weighted_temperature_f") / 2.0) * 2.0)
    .groupBy("temp_bin_f")
    .agg(
        F.count("*").alias("hour_count"),
        F.round(F.min("actual_demand_mw"), 1).alias("min_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.10)"), 1).alias("p10_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_demand_mw"),
        F.round(F.avg("actual_demand_mw"), 1).alias("mean_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.90)"), 1).alias("p90_demand_mw"),
        F.round(F.max("actual_demand_mw"), 1).alias("max_demand_mw"),
        F.round(F.stddev("actual_demand_mw"), 1).alias("stddev_demand_mw"),
    )
    .filter(F.col("hour_count") >= MIN_BIN_HOURS)
    .orderBy("temp_bin_f")
)

display(df_bins)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Empirical Balance Point Optimization
# MAGIC
# MAGIC We fit a cubic polynomial to the median demand curve to locate:
# MAGIC
# MAGIC $$\hat{T}_{\text{balance}} = \arg\min_T \;\mathbb{E}\left[\text{Demand} \mid T\right]$$
# MAGIC
# MAGIC A cubic (degree-3) polynomial is used because it captures the known U-shaped convexity
# MAGIC while having enough flexibility for mild asymmetry, without overfitting to bin noise.

# COMMAND ----------
# DBTITLE 1,Derive Empirical Balance Point via Polynomial Minimisation
pdf_curve = df_bins.toPandas()

# Fit degree-3 polynomial to median demand curve
poly_coeffs = np.polyfit(pdf_curve["temp_bin_f"], pdf_curve["median_demand_mw"], deg=3)
poly_func = np.poly1d(poly_coeffs)

# Dense temperature vector for precise argmin resolution
temp_dense = np.linspace(pdf_curve["temp_bin_f"].min(), pdf_curve["temp_bin_f"].max(), 2000)
demand_dense = poly_func(temp_dense)
empirical_balance_f = float(temp_dense[np.argmin(demand_dense)])
empirical_balance_c = (empirical_balance_f - 32.0) * 5.0 / 9.0
min_basal_demand_mw = float(np.min(demand_dense))

section("EMPIRICAL BALANCE POINT ANALYSIS")
print(f"Conventional Standard Balance Point  : 65.0°F (18.3°C)")
print(f"Empirical Grid Balance Point (argmin): {empirical_balance_f:.1f}°F ({empirical_balance_c:.1f}°C)")
print(f"Minimum Basal Median Demand          : {min_basal_demand_mw:,.0f} MW")
delta_f = empirical_balance_f - 65.0
direction = "above" if delta_f > 0 else "below"
print(f"Deviation from 65°F standard         : {abs(delta_f):.1f}°F {direction} convention")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Asymmetric Thermal Response Gradients
# MAGIC
# MAGIC A key modelling insight is that the U-curve is **not symmetric**. Texas has:
# MAGIC - Massive penetration of central air conditioning (~99% of residential units).
# MAGIC - Relatively limited electric resistance heating (many homes use gas for space heat).
# MAGIC
# MAGIC This suggests the **cooling gradient** (MW per °F above balance) differs significantly
# MAGIC from the **heating gradient** (MW per °F below balance).
# MAGIC
# MAGIC We use `scipy.stats.linregress` on each arm of the curve independently.

# COMMAND ----------
# DBTITLE 1,Measure Cooling & Heating Thermal Gradients (Asymmetry Analysis)
cooling_arm = pdf_curve[pdf_curve["temp_bin_f"] > empirical_balance_f]
heating_arm = pdf_curve[pdf_curve["temp_bin_f"] < empirical_balance_f]

cooling_slope, cooling_intercept, r_cool, p_cool, se_cool = stats.linregress(
    cooling_arm["temp_bin_f"], cooling_arm["median_demand_mw"]
)
heating_slope, heating_intercept, r_heat, p_heat, se_heat = stats.linregress(
    heating_arm["temp_bin_f"], heating_arm["median_demand_mw"]
)

# Heating arm slope is negative (demand increases as temp falls below balance)
# We report the magnitude of the gradient for interpretability
heating_gradient_magnitude = abs(heating_slope)

section("ASYMMETRIC THERMAL RESPONSE GRADIENT")
print(f"Cooling Arm ({empirical_balance_f:.1f}°F → {pdf_curve['temp_bin_f'].max():.0f}°F):")
print(f"  Slope  : +{cooling_slope:,.1f} MW per °F increase in temperature")
print(f"  R²     : {r_cool**2:.3f}    p-value: {p_cool:.2e}")
print(f"  Std Err: {se_cool:.1f} MW/°F")
print()
print(f"Heating Arm ({pdf_curve['temp_bin_f'].min():.0f}°F → {empirical_balance_f:.1f}°F):")
print(f"  Slope  : -{heating_gradient_magnitude:,.1f} MW per °F decrease in temperature")
print(f"  R²     : {r_heat**2:.3f}    p-value: {p_heat:.2e}")
print(f"  Std Err: {se_heat:.1f} MW/°F")
print()
print("Asymmetry Ratio (Cooling / Heating):")
ratio = cooling_slope / heating_gradient_magnitude
print(f"  {ratio:.2f}x — {'cooling-dominant' if ratio > 1 else 'heating-dominant'}")
print()
print("Gold Feature Engineering Implication:")
print(f"  - If |empirical_balance_f - 65| > 2°F, recommend adding custom CDH/HDH at {empirical_balance_f:.0f}°F")
print(f"    alongside the existing cdh_65 / hdh_65 Silver features.")
print(f"  - Asymmetry ratio of {ratio:.2f}x supports a two-slope piecewise term in addition to quadratic T².")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Residual Demand Variance by Temperature Bin
# MAGIC
# MAGIC Temperature explains demand, but not fully. The within-bin standard deviation
# MAGIC measures **unexplained variance** — the portion of load driven by other factors:
# MAGIC hour of day, day of week, holidays, economic activity, and weather variables
# MAGIC beyond temperature (humidity, solar irradiance, wind). This motivates including
# MAGIC all those features in the Gold feature table rather than relying on temperature alone.

# COMMAND ----------
# DBTITLE 1,Within-Bin Demand Variance (Unexplained Demand Drivers)
section("WITHIN-BIN RESIDUAL VARIANCE — UNEXPLAINED DEMAND STRUCTURE")
high_variance_bins = pdf_curve.nlargest(10, "stddev_demand_mw")[
    ["temp_bin_f", "hour_count", "median_demand_mw", "stddev_demand_mw"]
].reset_index(drop=True)
print("Top 10 temperature bins by within-bin demand standard deviation:")
print(high_variance_bins.to_string(index=False))
print()
avg_stddev_mw = pdf_curve["stddev_demand_mw"].mean()
print(f"Average within-bin demand std dev: {avg_stddev_mw:,.0f} MW")
print("This residual variance motivates including hour-of-day, day-of-week,")
print("holiday flags, humidity, and solar irradiance as additional Gold features.")
