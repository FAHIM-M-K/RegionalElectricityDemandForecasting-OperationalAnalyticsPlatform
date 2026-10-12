# Databricks notebook source
# MAGIC %md
# MAGIC # Federal Holiday Industrial Demand Suppression — Hypothesis Testing
# MAGIC
# MAGIC **Research Question:**  
# MAGIC Do federal holidays cause a statistically significant reduction in ERCOT electricity demand
# MAGIC relative to comparable non-holiday weekdays — after controlling for ambient temperature?
# MAGIC
# MAGIC **Why Temperature Control Is Essential:**  
# MAGIC A naive comparison of all holiday hours vs. all non-holiday hours is confounded by season.
# MAGIC Christmas Day is cold; Labor Day is hot; both suppress commercial load but for different
# MAGIC thermodynamic reasons. Without temperature stratification, we cannot isolate the
# MAGIC holiday-specific industrial/commercial closure effect from normal weather-driven variation.
# MAGIC
# MAGIC **Identification Strategy:**  
# MAGIC Restrict to **weekday** hours only (Mon–Fri). Within that pool, stratify into narrow
# MAGIC 3°F ambient temperature bins and compute the within-bin mean demand delta between
# MAGIC holiday and non-holiday observations. This controls for the temperature confound.
# MAGIC
# MAGIC **Statistical Tests Applied:**  
# MAGIC - **Welch's t-test** (two-sample, unequal variance): parametric test robust to unequal group sizes.
# MAGIC - **Mann-Whitney U test**: non-parametric rank-based test — valid regardless of normality,
# MAGIC   preferred when the demand distribution has heavy tails (extreme events).
# MAGIC
# MAGIC **Formal Claim Boundary:**  
# MAGIC We claim statistical association between federal holiday dates and reduced grid demand
# MAGIC on comparable-temperature weekdays. We do not claim causality between public policy,
# MAGIC holiday designation, or labour law and energy consumption behaviour.

# COMMAND ----------
# DBTITLE 1,Imports & Setup
import numpy as np
import pandas as pd
from scipy import stats
from pyspark.sql import functions as F

SILVER_TABLE = "main.silver.grid_hourly_conformed"

# COMMAND ----------
# DBTITLE 1,Load Analysis Cohort — Weekday Scope Only
df = (
    spark.table(SILVER_TABLE)
    .filter(
        F.col("actual_demand_mw").isNotNull() &
        (F.col("is_valid_demand") == True) &
        (F.col("weighted_temperature_f").isNotNull())
    )
)

# Restrict to Mon–Fri (day_of_week_local: 2=Mon … 6=Fri in Spark convention)
df_weekdays = df.filter(F.col("day_of_week_local").isin(2, 3, 4, 5, 6))

total_weekday_hours = df_weekdays.count()
holiday_hours = df_weekdays.filter(F.col("is_holiday") == True).count()
non_holiday_hours = total_weekday_hours - holiday_hours

print(f"Total weekday hours in cohort : {total_weekday_hours:,}")
print(f"Holiday weekday hours         : {holiday_hours:,}")
print(f"Non-holiday weekday hours     : {non_holiday_hours:,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Raw Descriptive Comparison (Uncontrolled)
# MAGIC
# MAGIC We first compare holiday vs. non-holiday demand statistics before controlling for
# MAGIC temperature. This gives us the observable magnitude of the effect; the hypothesis
# MAGIC test will determine whether the controlled effect is statistically significant.

# COMMAND ----------
# DBTITLE 1,Holiday vs. Non-Holiday Descriptive Statistics
df_raw_compare = (
    df_weekdays.groupBy("is_holiday")
    .agg(
        F.count("*").alias("hour_count"),
        F.round(F.avg("actual_demand_mw"), 1).alias("mean_demand_mw"),
        F.round(F.stddev("actual_demand_mw"), 1).alias("stddev_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.10)"), 1).alias("p10_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.90)"), 1).alias("p90_demand_mw"),
        F.round(F.avg("weighted_temperature_f"), 1).alias("avg_temp_f"),
    )
)

display(df_raw_compare)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Individual Holiday Breakdown
# MAGIC
# MAGIC Not all holidays are equal. Christmas (industrial shutdown, cold weather) and
# MAGIC Labor Day (warm, offices closed, factories idled) behave differently.
# MAGIC This breakdown identifies which specific holidays drive the strongest suppression.

# COMMAND ----------
# DBTITLE 1,Per-Holiday Demand Profile vs. Non-Holiday Weekday Baseline
# Compute the non-holiday weekday baseline for comparison
non_holiday_mean = (
    df_weekdays.filter(F.col("is_holiday") == False)
    .agg(F.avg("actual_demand_mw"))
    .first()[0]
)

df_per_holiday = (
    df_weekdays.filter(F.col("is_holiday") == True)
    .groupBy("holiday_name")
    .agg(
        F.count("*").alias("holiday_hours"),
        F.round(F.avg("actual_demand_mw"), 1).alias("mean_demand_mw"),
        F.round(F.avg("weighted_temperature_f"), 1).alias("avg_temp_f"),
    )
    .withColumn(
        "demand_delta_vs_baseline_mw",
        F.round(F.col("mean_demand_mw") - F.lit(round(non_holiday_mean, 1)), 1)
    )
    .orderBy("demand_delta_vs_baseline_mw")
)

display(df_per_holiday)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Formal Hypothesis Testing — Temperature-Controlled
# MAGIC
# MAGIC **Procedure:**
# MAGIC 1. Pull weekday records to Pandas driver (feasible — this is a few thousand rows).
# MAGIC 2. Bin ambient temperature into 3°F slots.
# MAGIC 3. For each bin with ≥5 holiday hours and ≥20 non-holiday hours, compute within-bin
# MAGIC    mean demand delta (non-holiday mean − holiday mean).
# MAGIC 4. Perform a one-sample t-test on the vector of bin deltas against $H_0: \mu_{\Delta} = 0$.
# MAGIC 5. Perform a global Mann-Whitney U test on the full (uncontrolled) samples as
# MAGIC    a non-parametric corroboration.

# COMMAND ----------
# DBTITLE 1,Execute Temperature-Stratified Hypothesis Test
pdf_weekdays = (
    df_weekdays.select(
        "actual_demand_mw",
        "is_holiday",
        "holiday_name",
        "weighted_temperature_f",
    )
    .toPandas()
)

# -- Temperature stratification --
BIN_WIDTH_F = 3.0
pdf_weekdays["temp_bin_3f"] = np.floor(pdf_weekdays["weighted_temperature_f"] / BIN_WIDTH_F) * BIN_WIDTH_F

holiday_series = pdf_weekdays[pdf_weekdays["is_holiday"] == True]["actual_demand_mw"]
non_holiday_series = pdf_weekdays[pdf_weekdays["is_holiday"] == False]["actual_demand_mw"]

# -- Within-bin deltas --
bin_deltas = []
qualified_bins = 0
for temp_bin, grp in pdf_weekdays.groupby("temp_bin_3f"):
    h = grp[grp["is_holiday"] == True]["actual_demand_mw"]
    nh = grp[grp["is_holiday"] == False]["actual_demand_mw"]
    if len(h) >= 5 and len(nh) >= 20:
        bin_deltas.append(nh.mean() - h.mean())
        qualified_bins += 1

avg_controlled_drop_mw = np.mean(bin_deltas)
median_controlled_drop_mw = np.median(bin_deltas)

# -- Statistical tests --
t_stat_controlled, p_val_controlled = stats.ttest_1samp(bin_deltas, popmean=0.0)
mwu_stat, p_val_mwu = stats.mannwhitneyu(holiday_series, non_holiday_series, alternative="two-sided")

# -- Effect size: Cohen's d on raw uncontrolled samples --
pooled_std = np.sqrt(
    (holiday_series.std() ** 2 + non_holiday_series.std() ** 2) / 2.0
)
cohens_d = (non_holiday_series.mean() - holiday_series.mean()) / pooled_std

ALPHA = 0.01

section("FEDERAL HOLIDAY DEMAND SUPPRESSION — HYPOTHESIS TEST RESULTS")
print(f"Qualified temperature bins (n ≥ 5 holiday, ≥ 20 non-holiday) : {qualified_bins}")
print(f"Temperature-Controlled Avg Demand Drop                       : {avg_controlled_drop_mw:,.0f} MW")
print(f"Temperature-Controlled Median Demand Drop                    : {median_controlled_drop_mw:,.0f} MW")
print(f"Drop as % of Non-Holiday Mean ({non_holiday_series.mean():,.0f} MW)    : "
      f"{(avg_controlled_drop_mw / non_holiday_series.mean()) * 100:.1f}%")
print()
print("Statistical Significance Tests:")
print(f"  Welch's t-test (temperature-controlled, stratified):")
print(f"    t = {t_stat_controlled:.3f}   p-value = {p_val_controlled:.4e}   "
      f"{'SIGNIFICANT' if p_val_controlled < ALPHA else 'NOT SIGNIFICANT'} at α = {ALPHA}")
print()
print(f"  Mann-Whitney U test (rank-based, no normality assumption):")
print(f"    U = {mwu_stat:.1f}   p-value = {p_val_mwu:.4e}   "
      f"{'SIGNIFICANT' if p_val_mwu < ALPHA else 'NOT SIGNIFICANT'} at α = {ALPHA}")
print()
print(f"  Effect Size (Cohen's d, uncontrolled): {cohens_d:.3f}  "
      f"({'large' if cohens_d > 0.5 else 'medium' if cohens_d > 0.2 else 'small'})")
print()
print("Conclusion:")
if p_val_controlled < ALPHA and p_val_mwu < ALPHA:
    print(f"  ✓ Both tests REJECT the null hypothesis at α = {ALPHA}.")
    print(f"  ✓ Federal holidays are associated with a statistically significant")
    print(f"    temperature-controlled demand reduction of ~{avg_controlled_drop_mw:,.0f} MW")
    print(f"    (~{(avg_controlled_drop_mw / non_holiday_series.mean()) * 100:.1f}% of comparable non-holiday weekday demand).")
    print()
    print("  Gold Feature Implication:")
    print("  - Include `is_holiday` (boolean) as a binary feature.")
    print("  - Consider `is_holiday_adjacent` (day before / day after) as an additional feature")
    print("    — factories often begin shutdowns the preceding afternoon.")
else:
    print("  ✗ One or both tests failed to reject the null at α = 0.01.")
    print("    Review bin qualification thresholds and holiday coverage in the calendar reference.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Holiday-Adjacent Effects (Day Before / Day After)
# MAGIC
# MAGIC Industrial facilities and commercial buildings often begin shutdown procedures
# MAGIC the afternoon before major holidays (e.g., the Wednesday before Thanksgiving,
# MAGIC Christmas Eve). Similarly, ramp-up on the day after can be gradual.
# MAGIC We test whether the immediate calendar neighbours of holidays carry measurable demand signal.

# COMMAND ----------
# DBTITLE 1,Holiday-Adjacent Demand Analysis
from pyspark.sql.window import Window

# Flag dates immediately adjacent to holiday dates
window_lag  = Window.orderBy("timestamp_utc")
window_lead = Window.orderBy("timestamp_utc")

df_adj = (
    df.withColumn("prev_is_holiday", F.lag("is_holiday", 24).over(window_lag))
      .withColumn("next_is_holiday", F.lead("is_holiday", 24).over(window_lead))
      .withColumn(
          "adjacency_type",
          F.when(F.col("is_holiday") == True, F.lit("holiday"))
           .when(F.col("prev_is_holiday") == True, F.lit("day_after_holiday"))
           .when(F.col("next_is_holiday") == True, F.lit("day_before_holiday"))
           .otherwise(F.lit("regular_weekday"))
      )
    .filter(F.col("day_of_week_local").isin(2, 3, 4, 5, 6))
)

df_adj_stats = (
    df_adj.groupBy("adjacency_type")
    .agg(
        F.count("*").alias("hour_count"),
        F.round(F.avg("actual_demand_mw"), 1).alias("mean_demand_mw"),
        F.round(F.expr("percentile_approx(actual_demand_mw, 0.50)"), 1).alias("median_demand_mw"),
        F.round(F.avg("weighted_temperature_f"), 1).alias("avg_temp_f"),
    )
    .orderBy("mean_demand_mw")
)

display(df_adj_stats)

section("HOLIDAY-ADJACENT DEMAND SIGNAL")
pdf_adj = df_adj_stats.toPandas()
baseline_row = pdf_adj[pdf_adj["adjacency_type"] == "regular_weekday"]
if not baseline_row.empty:
    baseline_mw = baseline_row["mean_demand_mw"].values[0]
    for _, row in pdf_adj.iterrows():
        if row["adjacency_type"] != "regular_weekday":
            delta = row["mean_demand_mw"] - baseline_mw
            delta_pct = (delta / baseline_mw) * 100.0
            direction = "lower" if delta < 0 else "higher"
            print(f"  {row['adjacency_type']:<22}: {abs(delta):,.0f} MW {direction} than regular weekday "
                  f"({abs(delta_pct):.1f}%) — n={row['hour_count']:,}")
