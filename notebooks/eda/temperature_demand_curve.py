# Databricks notebook source
# MAGIC %md
# MAGIC # Empirical Temperature–Demand Curve & Thermal Balance Point Discovery
# MAGIC
# MAGIC **Core Objective:** Discover the empirical grid thermal balance point ($T^* = \arg\min \text{Demand}(T)$)
# MAGIC and quantify the cooling vs heating asymmetry to engineer optimal degree-hour features.
# MAGIC
# MAGIC **Feature Layer Implication:**
# MAGIC - Conventional degree-hours use 65°F. If the empirical balance is 55°F, custom `cdh_55` and `hdh_55` must be built.
# MAGIC - Asymmetry determines whether cooling and heating response should be modeled with separate slopes.

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

display(df_bins)

# COMMAND ----------
# DBTITLE 1,Empirical Balance Point Discovery & Fit
pdf_curve = df_bins.toPandas()

# Degree-3 polynomial fit to binned median demand
poly_func = np.poly1d(np.polyfit(pdf_curve["temp_bin_f"], pdf_curve["median_demand_mw"], deg=3))
temp_dense = np.linspace(pdf_curve["temp_bin_f"].min(), pdf_curve["temp_bin_f"].max(), 1000)
demand_dense = poly_func(temp_dense)

empirical_balance_f = float(temp_dense[np.argmin(demand_dense)])
min_demand_mw = float(np.min(demand_dense))

print(f"Empirical Grid Balance Point : {empirical_balance_f:.1f}°F (vs 65.0°F standard convention)")
print(f"Minimum Basal Demand         : {min_demand_mw:,.0f} MW")

# Plot
plt.figure(figsize=(9, 5))
plt.scatter(pdf_curve["temp_bin_f"], pdf_curve["median_demand_mw"], color="#1f77b4", s=25, label="Observed Median Demand")
plt.plot(temp_dense, demand_dense, color="#d62728", lw=2, label="Cubic Fit")
plt.axvline(65.0, color="#7f7f7f", linestyle="--", label="65°F Standard")
plt.axvline(empirical_balance_f, color="#2ca02c", linestyle="--", label=f"Empirical Balance ({empirical_balance_f:.1f}°F)")
plt.title("ERCOT Empirical Temperature–Demand Curve", fontsize=12, fontweight="bold")
plt.xlabel("Weighted Ambient Temperature (°F)")
plt.ylabel("Electricity Demand (MW)")
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------
# DBTITLE 1,Cooling vs Heating Gradient Asymmetry
cooling = pdf_curve[pdf_curve["temp_bin_f"] > empirical_balance_f]
heating = pdf_curve[pdf_curve["temp_bin_f"] < empirical_balance_f]

slope_cool, _, r_cool, _, _ = stats.linregress(cooling["temp_bin_f"], cooling["median_demand_mw"])
slope_heat, _, r_heat, _, _ = stats.linregress(heating["temp_bin_f"], heating["median_demand_mw"])

ratio = slope_cool / abs(slope_heat)
print(f"Cooling Gradient : +{slope_cool:,.1f} MW/°F (R² = {r_cool**2:.2f})")
print(f"Heating Gradient : -{abs(slope_heat):,.1f} MW/°F (R² = {r_heat**2:.2f})")
print(f"Asymmetry Ratio  : {ratio:.2f}x cooling-dominant")
print(f"Conclusion       : Empirically supports two-slope degree-hours (cdh_{empirical_balance_f:.0f} / hdh_{empirical_balance_f:.0f}).")
