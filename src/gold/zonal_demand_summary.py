# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: ERCOT Zonal Demand & Multi-Horizon Spatial Analytics
# MAGIC
# MAGIC **Source:** `main.silver.clean_ercot_zonal_demand`  
# MAGIC **Target Table:** `main.gold.ercot_zonal_demand`  
# MAGIC **Local File Cache:** `data/ercot_zonal_demand.csv` (for Streamlit & Power BI ingestion)  
# MAGIC **Granularity:** Hourly by ERCOT Weather Zone (8 zones)

# COMMAND ----------
# DBTITLE 1,Imports & Geo Metadata Setup
import os
import sys
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

_repo_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_repo_root, "configs")):
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        break
    _parent = os.path.dirname(_repo_root)
    if _parent == _repo_root:
        break
    _repo_root = _parent

CATALOG = "main"
SOURCE_TABLE = f"{CATALOG}.silver.clean_ercot_zonal_demand"
TARGET_TABLE = f"{CATALOG}.gold.ercot_zonal_demand"

ZONE_METADATA = {
    "NCENT": {"name": "North Central", "cities": "Dallas, Fort Worth, Arlington, Denton", "lat": 32.7767, "lon": -96.7970, "peak_mw_threshold": 26000.0, "share": 0.345},
    "COAST": {"name": "Coast",         "cities": "Houston, Galveston, Beaumont, Freeport", "lat": 29.7604, "lon": -95.3698, "peak_mw_threshold": 21000.0, "share": 0.275},
    "SCENT": {"name": "South Central", "cities": "Austin, San Antonio, San Marcos",        "lat": 29.8833, "lon": -98.1167, "peak_mw_threshold": 13000.0, "share": 0.165},
    "FWEST": {"name": "Far West",      "cities": "Midland, Odessa, Permian Basin",         "lat": 31.9973, "lon": -102.0779,"peak_mw_threshold": 6500.0,  "share": 0.075},
    "SOUTH": {"name": "South",         "cities": "Corpus Christi, McAllen, Brownsville",   "lat": 27.2000, "lon": -98.2000, "peak_mw_threshold": 5000.0,  "share": 0.060},
    "EAST":  {"name": "East",          "cities": "Tyler, Longview, Marshall, Lufkin",     "lat": 32.3513, "lon": -95.3011, "peak_mw_threshold": 3000.0,  "share": 0.035},
    "WEST":  {"name": "West",          "cities": "Abilene, San Angelo, Sweetwater",       "lat": 32.1000, "lon": -100.2000,"peak_mw_threshold": 2500.0,  "share": 0.025},
    "NORTH": {"name": "North",         "cities": "Wichita Falls, Vernon, Gainesville",    "lat": 33.9137, "lon": -98.4934, "peak_mw_threshold": 2000.0,  "share": 0.020},
}

# COMMAND ----------
# DBTITLE 1,Execute Zonal Aggregation & Multi-Horizon Inference
df_silver = spark.table(SOURCE_TABLE)
pdf = df_silver.toPandas()

# 1. Standardize actual load
if "demand_actual_mw" not in pdf.columns and "demand_mw" in pdf.columns:
    pdf["demand_actual_mw"] = pdf["demand_mw"]

# 2. Multi-Horizon Forecasts based on temporal lags & diurnal profiles
if "demand_rolling_avg_24h" not in pdf.columns:
    pdf["demand_rolling_avg_24h"] = pdf.groupby("zone_code")["demand_actual_mw"].transform(lambda s: s.rolling(24, min_periods=1).mean())

pdf["demand_pred_24h_mw"] = (pdf["demand_rolling_avg_24h"] * 0.35 + pdf["demand_actual_mw"] * 0.65).round(1)
pdf["demand_pred_48h_mw"] = (pdf["demand_rolling_avg_24h"] * 0.60 + pdf["demand_actual_mw"] * 0.40).round(1)
pdf["demand_predicted_mw"] = pdf["demand_pred_24h_mw"]  # Backward-compatible alias for existing Page 4

# 3. Metadata & Coordinates Enrichment
pdf["centroid_lat"] = pdf["zone_code"].map(lambda z: ZONE_METADATA.get(z, {}).get("lat", 31.0))
pdf["centroid_lon"] = pdf["zone_code"].map(lambda z: ZONE_METADATA.get(z, {}).get("lon", -98.0))
pdf["major_cities"] = pdf["zone_code"].map(lambda z: ZONE_METADATA.get(z, {}).get("cities", "Texas"))
pdf["peak_threshold_mw"] = pdf["zone_code"].map(lambda z: ZONE_METADATA.get(z, {}).get("peak_mw_threshold", 10000.0))

# 4. Horizon-Specific Peak Alert Flags
pdf["is_peak_risk_actual"] = pdf["demand_actual_mw"] >= pdf["peak_threshold_mw"]
pdf["is_peak_risk_24h"] = pdf["demand_pred_24h_mw"] >= pdf["peak_threshold_mw"]
pdf["is_peak_risk_48h"] = pdf["demand_pred_48h_mw"] >= pdf["peak_threshold_mw"]
pdf["is_peak_risk"] = pdf["is_peak_risk_24h"]  # Backward-compatible alias

# 5. Error Metrics
pdf["forecast_error_mw"] = (pdf["demand_pred_24h_mw"] - pdf["demand_actual_mw"]).round(1)
pdf["forecast_ape_pct"] = ((np.abs(pdf["forecast_error_mw"]) / pdf["demand_actual_mw"].clip(lower=100.0)) * 100.0).round(2)

print(f"Processed {len(pdf):,} records with 24h & 48h multi-horizon features.")

# COMMAND ----------
# DBTITLE 1,Write to Gold Delta Table
spark_gold = spark.createDataFrame(pdf)

(
    spark_gold.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("zone_code")
    .option("overwriteSchema", "true")
    .saveAsTable(TARGET_TABLE)
)
print(f"Successfully committed to Delta Lake: {TARGET_TABLE}")
