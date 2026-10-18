# Databricks notebook source
# MAGIC %md
# MAGIC # Temporal Seasonality & Diurnal Load Profile Analysis
# MAGIC
# MAGIC **Core Objective:** Describe intraday load profiles and weekday/weekend and holiday
# MAGIC differences. These exploratory comparisons can motivate feature tests, but do not by
# MAGIC themselves establish that a feature improves forecast accuracy or that a pattern is causal.
# MAGIC
# MAGIC **Feature Layer Implication:**
# MAGIC - Describes patterns associated with hour, weekday/weekend, and holidays.
# MAGIC - Feature value must be confirmed with leakage-safe rolling validation or ablation; this
# MAGIC   notebook alone does not establish a statistical or causal justification.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
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
    .withColumn(
        "season",
        F.when(F.col("month_local").isin(12, 1, 2), F.lit("Winter"))
         .when(F.col("month_local").isin(3, 4, 5), F.lit("Spring"))
         .when(F.col("month_local").isin(6, 7, 8), F.lit("Summer"))
         .otherwise(F.lit("Fall"))
    )
    .withColumn("day_name", F.date_format("timestamp_local", "EEEE"))
)

# COMMAND ----------
# DBTITLE 1,1. Intraday Diurnal Profiles by Season
df_diurnal = (
    df.groupBy("season", "hour_local")
    .agg(
        F.count("*").alias("hours"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_demand_mw"),
        F.round(F.avg("weighted_temperature_f"), 1).alias("avg_temp_f"),
    )
    .orderBy("season", "hour_local")
)

display(df_diurnal)

pdf_diurnal = df_diurnal.toPandas()
plt.figure(figsize=(9, 4.5))
colors = {"Summer": "#d62728", "Winter": "#1f77b4", "Spring": "#2ca02c", "Fall": "#ff7f0e"}
for s in ["Summer", "Winter", "Spring", "Fall"]:
    sub = pdf_diurnal[pdf_diurnal["season"] == s]
    plt.plot(sub["hour_local"], sub["median_demand_mw"], marker="o", markersize=3, label=s, color=colors[s], lw=1.8)

plt.title("ERCOT Diurnal Load Profile by Season", fontsize=12, fontweight="bold")
plt.xlabel("Hour of Day (Local Time)")
plt.ylabel("Median Demand (MW)")
plt.xticks(range(0, 24))
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------
# DBTITLE 1,2. Weekday vs. Weekend Demand Gap
df_dow = (
    df.groupBy("day_of_week_local", "day_name")
    .agg(
        F.count("*").alias("hours"),
        F.round(F.avg("actual_demand_mw"), 1).alias("mean_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_demand_mw"),
    )
    .orderBy("day_of_week_local")
)

display(df_dow)

pdf_dow = df_dow.toPandas()
weekday_mean = pdf_dow[pdf_dow["day_of_week_local"].isin([2, 3, 4, 5, 6])]["mean_demand_mw"].mean()
weekend_mean = pdf_dow[pdf_dow["day_of_week_local"].isin([1, 7])]["mean_demand_mw"].mean()
gap_pct = ((weekday_mean - weekend_mean) / weekend_mean) * 100.0

print(f"Weekday Mean Load : {weekday_mean:,.0f} MW")
print(f"Weekend Mean Load : {weekend_mean:,.0f} MW")
print(f"Weekday Load Gap  : {gap_pct:+.1f}% relative to weekends (descriptive; not a feature test)")

# COMMAND ----------
# DBTITLE 1,3. Monthly Macro Seasonality
df_monthly = (
    df.groupBy("month_local")
    .agg(
        F.count("*").alias("hours"),
        F.round(F.avg("actual_demand_mw"), 1).alias("mean_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_demand_mw"),
        F.round(F.avg("weighted_temperature_f"), 1).alias("avg_temp_f"),
    )
    .orderBy("month_local")
)

display(df_monthly)

# COMMAND ----------
# DBTITLE 1,4. Date-Level Holiday Demand Suppression (Temperature-Controlled Regression)
df_weekdays = df.filter(F.col("day_of_week_local").isin(2, 3, 4, 5, 6))

# Date-level daily average aggregation to eliminate intraday serial autocorrelation (F-20)
df_daily = (
    df_weekdays.groupBy("date_local", "is_holiday")
    .agg(
        F.avg("actual_demand_mw").alias("daily_avg_demand_mw"),
        F.avg("weighted_temperature_f").alias("daily_avg_temp_f"),
    )
    .toPandas()
)

holiday_daily = df_daily[df_daily["is_holiday"] == True]["daily_avg_demand_mw"]
non_holiday_daily = df_daily[df_daily["is_holiday"] == False]["daily_avg_demand_mw"]

# 1. Unadjusted Welch's t-test
t_stat, p_val = stats.ttest_ind(holiday_daily, non_holiday_daily, equal_var=False)
raw_daily_drop = float(non_holiday_daily.mean() - holiday_daily.mean())

# 2. Temperature-controlled multiple regression (F-20)
# Model: Demand = beta0 + beta_holiday * is_holiday + beta_temp * Temp + beta_temp2 * (Temp - 55.2)^2
valid_mask = df_daily["daily_avg_demand_mw"].notnull() & df_daily["daily_avg_temp_f"].notnull()
df_reg = df_daily[valid_mask].copy()
temp_dev = df_reg["daily_avg_temp_f"] - 55.2

X = np.column_stack([
    np.ones(len(df_reg)),
    df_reg["is_holiday"].astype(float),
    df_reg["daily_avg_temp_f"],
    temp_dev ** 2
])
y = df_reg["daily_avg_demand_mw"].values
beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
adj_holiday_drop = -float(beta[1])

print(f"Date-Level Daily Observations  : {len(holiday_daily)} holiday dates vs {len(non_holiday_daily)} non-holiday weekdays.")
print(f"Unadjusted Mean Difference     : {raw_daily_drop:,.0f} MW (Welch's t-test p = {p_val:.3e})")
print(f"Temperature-Adjusted Drop      : {adj_holiday_drop:,.0f} MW (controlling for daily temp and nonlinear CDH/HDH response)")
if p_val < 0.05:
    print("Unadjusted Test                : Difference is statistically significant at the 0.05 level; this does not establish causality.")
else:
    print("Unadjusted Test                : Difference is not statistically significant at the 0.05 level.")
print("Adjusted Estimate              : Regression point estimate only; no confidence interval is calculated, so treat it as exploratory.")
print("Interpretation                 : Neither result proves a causal holiday effect or forecast-feature value.")
