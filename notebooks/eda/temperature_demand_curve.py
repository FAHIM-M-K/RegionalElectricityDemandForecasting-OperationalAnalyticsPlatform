# Databricks notebook source
# MAGIC %md
# MAGIC # Exploratory Temperature–Demand Relationship
# MAGIC
# MAGIC **Core Objective:** Describe how observed demand varies across temperature bins and compare
# MAGIC the fitted slopes on either side of the observed minimum. This is a retrospective association:
# MAGIC hour-of-week, season, and other factors also affect demand, so this curve does not estimate a
# MAGIC causal temperature effect or a stable engineering balance point.
# MAGIC
# MAGIC **Feature Layer Implication:** Treat the fitted minimum and slopes as hypotheses. Add or tune
# MAGIC degree-hour features only if leakage-safe rolling validation shows improvement over the current
# MAGIC model and calendar features.

# COMMAND ----------
# DBTITLE 1,Imports & Data Load
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pyspark.sql import functions as F

SILVER_TABLE = "main.silver.grid_hourly_conformed"

df = (
    spark.table(SILVER_TABLE)
    .filter(
        F.col("actual_demand_mw").isNotNull() &
        (F.col("is_valid_demand") == True) &
        (F.col("weighted_temperature_f").isNotNull())
    )
)

# COMMAND ----------
# DBTITLE 1,Binned Temperature–Demand Distribution (2°F Bins)
df_bins = (
    df.withColumn("temp_bin_f", F.floor(F.col("weighted_temperature_f") / 2.0) * 2.0)
    .groupBy("temp_bin_f")
    .agg(
        F.count("*").alias("hour_count"),
        F.round(F.min("actual_demand_mw"), 1).alias("min_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.10)"), 1).alias("p10_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.90)"), 1).alias("p90_demand_mw"),
        F.round(F.max("actual_demand_mw"), 1).alias("max_demand_mw"),
    )
    .filter(F.col("hour_count") >= 15)
    .orderBy("temp_bin_f")
)

# Materialize the grouped Spark result once. Calling display(df_bins) and then
# toPandas() separately triggers the same distributed aggregation twice.
pdf_curve = df_bins.toPandas()
display(pdf_curve)

# COMMAND ----------
# DBTITLE 1,Exploratory Fitted Minimum
if len(pdf_curve) < 4:
    raise RuntimeError(
        "Temperature-demand curve needs at least four populated temperature bins "
        "for the cubic fit; reduce the bin-count filter or use a lower-degree fit."
    )

# Degree-3 polynomial fit to binned median demand
# Exploratory cubic fit to binned medians; it is not a causal temperature-response model.
poly_func = np.poly1d(np.polyfit(pdf_curve["temp_bin_f"], pdf_curve["median_demand_mw"], deg=3))
temp_dense = np.linspace(pdf_curve["temp_bin_f"].min(), pdf_curve["temp_bin_f"].max(), 1000)
demand_dense = poly_func(temp_dense)

# Restrict the exploratory fitted-minimum search to [48°F, 68°F].
interior_mask = (temp_dense >= 48.0) & (temp_dense <= 68.0)
if not interior_mask.any():
    raise RuntimeError(
        "The sampled temperature bins do not overlap the 48–68°F search range; "
        "no fitted minimum can be reported for this sample."
    )
fitted_minimum_f = float(temp_dense[interior_mask][np.argmin(demand_dense[interior_mask])])
min_demand_mw = float(np.min(demand_dense[interior_mask]))

print(f"Observed-bin fitted minimum : {fitted_minimum_f:.1f}°F (exploratory cubic fit; sensitive to the sampled period and binning)")
print(f"Fitted demand at minimum    : {min_demand_mw:,.0f} MW (descriptive fit, not a physical baseline)")

# Plot
plt.figure(figsize=(9, 5))
plt.scatter(pdf_curve["temp_bin_f"], pdf_curve["median_demand_mw"], color="#1f77b4", s=25, label="Observed Median Demand")
plt.plot(temp_dense, demand_dense, color="#d62728", lw=2, label="Cubic Fit")
plt.axvline(65.0, color="#7f7f7f", linestyle="--", label="65°F Standard")
plt.axvline(fitted_minimum_f, color="#2ca02c", linestyle="--", label=f"Fitted Minimum ({fitted_minimum_f:.1f}°F)")
plt.title("ERCOT Observed Temperature–Demand Relationship", fontsize=12, fontweight="bold")
plt.xlabel("Weighted Ambient Temperature (°F)")
plt.ylabel("Electricity Demand (MW)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------
# DBTITLE 1,Descriptive Slopes Around the Fitted Minimum
cooling = pdf_curve[pdf_curve["temp_bin_f"] > fitted_minimum_f]
heating = pdf_curve[pdf_curve["temp_bin_f"] < fitted_minimum_f]

if len(cooling) < 2 or len(heating) < 2:
    raise RuntimeError(
        "Temperature-demand curve needs at least two bins on each side of the fitted minimum "
        "to estimate separate descriptive slopes."
    )

slope_cool, _, r_cool, _, _ = stats.linregress(cooling["temp_bin_f"], cooling["median_demand_mw"])
slope_heat, _, r_heat, _, _ = stats.linregress(heating["temp_bin_f"], heating["median_demand_mw"])

ratio = slope_cool / abs(slope_heat) if slope_heat != 0 else float("nan")
print(f"Slope above fitted minimum : {slope_cool:+,.1f} MW/°F (R² = {r_cool**2:.2f})")
print(f"Slope below fitted minimum : {slope_heat:+,.1f} MW/°F (R² = {r_heat**2:.2f})")
print(f"Slope ratio (above / |below|): {ratio:.2f}")
print("Interpretation   : Descriptive slopes across temperature-bin medians; test degree-hour features with rolling validation before adoption.")
