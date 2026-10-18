# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: ERCOT Zonal Demand Spatial Analytics
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

# ERCOT Weather Zone Metadata (display names and representative centroids)
ZONE_METADATA_ROWS = [
    ("NCENT", "North Central", "Dallas, Fort Worth, Arlington, Denton", 32.7767, -96.7970),
    ("COAST", "Coast",         "Houston, Galveston, Beaumont, Freeport", 29.7604, -95.3698),
    ("SCENT", "South Central", "Austin, San Antonio, San Marcos",        29.8833, -98.1167),
    ("FWEST", "Far West",      "Midland, Odessa, Permian Basin",         31.9973, -102.0779),
    ("SOUTH", "South",         "Corpus Christi, McAllen, Brownsville",   27.2000, -98.2000),
    ("EAST",  "East",          "Tyler, Longview, Marshall, Lufkin",     32.3513, -95.3011),
    ("WEST",  "West",          "Abilene, San Angelo, Sweetwater",       32.1000, -100.2000),
    ("NORTH", "North",         "Wichita Falls, Vernon, Gainesville",    33.9137, -98.4934),
]

# COMMAND ----------
# DBTITLE 1,Execute Pure-Spark Zonal Aggregation
df_silver = spark.table(SOURCE_TABLE)

df_meta = spark.createDataFrame(
    ZONE_METADATA_ROWS,
    ["zone_code", "zone_display_name", "major_cities", "centroid_lat", "centroid_lon"]
)

df_gold = (
    df_silver
    .join(F.broadcast(df_meta), on="zone_code", how="left")
    .withColumn("_gold_processed_at_utc", F.current_timestamp())
)

row_count = df_gold.count()
print(f"Native Spark Pipeline Processed: {row_count:,} zonal observed-demand records.")

# COMMAND ----------
# DBTITLE 1,Write to Gold Delta Table
(
    df_gold.write
    .format("delta")
    .mode("overwrite")
    .partitionBy("zone_code")
    .option("overwriteSchema", "true")
    .saveAsTable(TARGET_TABLE)
)
print(f"Successfully committed native Spark Gold dataset to Delta Lake: {TARGET_TABLE}")
