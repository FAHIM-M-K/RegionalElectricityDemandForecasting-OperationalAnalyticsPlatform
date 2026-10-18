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
df_daily = df_daily.sort_values("date_local").reset_index(drop=True)

holiday_daily = df_daily[df_daily["is_holiday"] == True]["daily_avg_demand_mw"]
non_holiday_daily = df_daily[df_daily["is_holiday"] == False]["daily_avg_demand_mw"]

# 1. Unadjusted Welch's t-test
t_stat, p_val = stats.ttest_ind(holiday_daily, non_holiday_daily, equal_var=False)
raw_daily_drop = float(non_holiday_daily.mean() - holiday_daily.mean())

# 2. Temperature-controlled multiple regression (F-20)
# Model: Demand = beta0 + beta_holiday * is_holiday + beta_temp * Temp + beta_temp2 * (Temp - 55.2)^2
valid_mask = (
    df_daily["daily_avg_demand_mw"].notnull()
    & df_daily["daily_avg_temp_f"].notnull()
    & df_daily["is_holiday"].notnull()
)
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

# Moving-block bootstrap: resample contiguous weekday blocks to retain some
# short-term dependence between nearby daily observations. This interval is
# descriptive and does not remove confounding or establish a causal effect.
BOOTSTRAP_REPLICATES = 500
BOOTSTRAP_BLOCK_ROWS = 20  # approximately four consecutive work weeks
bootstrap_rng = np.random.default_rng(42)
bootstrap_adjusted_drops = []
row_count = len(df_reg)
block_starts = np.arange(max(1, row_count - BOOTSTRAP_BLOCK_ROWS + 1))
for _ in range(BOOTSTRAP_REPLICATES):
    sampled_positions = []
    while len(sampled_positions) < row_count:
        start = int(bootstrap_rng.choice(block_starts))
        sampled_positions.extend(range(start, min(start + BOOTSTRAP_BLOCK_ROWS, row_count)))
    sampled_positions = np.asarray(sampled_positions[:row_count], dtype=int)
    bootstrap_frame = df_reg.iloc[sampled_positions]
    bootstrap_temp = bootstrap_frame["daily_avg_temp_f"].to_numpy(float)
    bootstrap_temp_dev = bootstrap_temp - 55.2
    bootstrap_x = np.column_stack([
        np.ones(row_count),
        bootstrap_frame["is_holiday"].astype(float).to_numpy(),
        bootstrap_temp,
        bootstrap_temp_dev ** 2,
    ])
    bootstrap_y = bootstrap_frame["daily_avg_demand_mw"].to_numpy(float)
    bootstrap_beta, _, _, _ = np.linalg.lstsq(bootstrap_x, bootstrap_y, rcond=None)
    if np.isfinite(bootstrap_beta[1]):
        bootstrap_adjusted_drops.append(-float(bootstrap_beta[1]))

if len(bootstrap_adjusted_drops) >= BOOTSTRAP_REPLICATES * 0.9:
    adjusted_ci_low, adjusted_ci_high = np.percentile(bootstrap_adjusted_drops, [2.5, 97.5])
else:
    adjusted_ci_low, adjusted_ci_high = np.nan, np.nan

print(f"Date-Level Daily Observations  : {len(holiday_daily)} holiday dates vs {len(non_holiday_daily)} non-holiday weekdays.")
print(f"Unadjusted Mean Difference     : {raw_daily_drop:,.0f} MW (Welch's t-test p = {p_val:.3e})")
print(f"Temperature-Adjusted Drop      : {adj_holiday_drop:,.0f} MW (controlling for daily temp and nonlinear CDH/HDH response)")
if np.isfinite(adjusted_ci_low) and np.isfinite(adjusted_ci_high):
    print(
        f"Adjusted 95% block-bootstrap CI: [{adjusted_ci_low:,.0f}, "
        f"{adjusted_ci_high:,.0f}] MW ({len(bootstrap_adjusted_drops)} valid replicates; "
        f"20-weekday-row moving blocks)"
    )
else:
    print("Adjusted 95% block-bootstrap CI: unavailable (too many invalid bootstrap replicates).")
if p_val < 0.05:
    print("Unadjusted Test                : Difference is statistically significant at the 0.05 level; this does not establish causality.")
else:
    print("Unadjusted Test                : Difference is not statistically significant at the 0.05 level.")
print("Inference Caveat               : The Welch p-value assumes independent dates; use the block-bootstrap interval to account for short-term dependence.")
print("Interpretation                 : These adjusted observational results do not prove a causal holiday effect or forecast-feature value.")
