# Databricks notebook source
# MAGIC %md
# MAGIC # 01. EIA Electricity Load & Forecast Ingestion
# MAGIC
# MAGIC **Source:** U.S. Energy Information Administration (EIA) Open Data API v2 (Form EIA-930)  
# MAGIC **Target:** `bronze.eia_hourly_raw`  
# MAGIC **Series:** 
# MAGIC - `D` = Actual Electricity Demand (MWh)
# MAGIC - `DF` = Day-Ahead Demand Forecast (MWh)  
# MAGIC **Region:** ERCOT (`ERCO`)

# COMMAND ----------
# DBTITLE 1,Widgets & Parameters
dbutils.widgets.text("catalog", "main", "Catalog Name")
dbutils.widgets.text("eia_api_key", "", "EIA API Key")
dbutils.widgets.dropdown("ingest_mode", "incremental", ["incremental", "full_backfill"], "Ingest Mode")
dbutils.widgets.text("backfill_start_date", "2021-01-01", "Backfill Start Date (YYYY-MM-DD)")
dbutils.widgets.text("incremental_days", "14", "Incremental Lookback (Days)")

# COMMAND ----------
import os
import json
import time
import datetime
import urllib.request
import urllib.parse
from pyspark.sql import functions as F

CATALOG = dbutils.widgets.get("catalog").strip()
EIA_API_KEY = dbutils.widgets.get("eia_api_key").strip()
INGEST_MODE = dbutils.widgets.get("ingest_mode").strip()
BACKFILL_START = dbutils.widgets.get("backfill_start_date").strip()
INCREMENTAL_DAYS = int(dbutils.widgets.get("incremental_days").strip())

if not EIA_API_KEY:
    EIA_API_KEY = os.getenv("EIA_API_KEY", "")

assert EIA_API_KEY, "EIA_API_KEY is required to fetch EIA data."

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA bronze")

# COMMAND ----------
# DBTITLE 1,Compute Date Window
now_utc = datetime.datetime.now(datetime.timezone.utc)
end_date_str = now_utc.strftime("%Y-%m-%d")

if INGEST_MODE == "full_backfill":
    start_date_str = BACKFILL_START
else:
    # 14 days lookback captures restated / corrected EIA hours
    start_dt = now_utc - datetime.timedelta(days=INCREMENTAL_DAYS)
    start_date_str = start_dt.strftime("%Y-%m-%d")

start_param = f"{start_date_str}T00"
end_param = f"{end_date_str}T23"

print(f"Fetching EIA ERCOT data from {start_param} to {end_param} (Mode: {INGEST_MODE})")

# COMMAND ----------
# DBTITLE 1,Paginated EIA API Client
eia_base_url = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

def fetch_eia_records(api_key: str, start: str, end: str, page_size: int = 5000):
    records = []
    offset = 0
    
    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": "ERCO",
            "start": start,
            "end": end,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page_size
        }
        
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{eia_base_url}?{query}", headers={"User-Agent": "ERCOT-Platform/1.0"})
        
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    break
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"EIA API request failed at offset {offset}: {e}")
                time.sleep(2 * attempt)
                
        resp_obj = payload.get("response", {})
        batch = resp_obj.get("data", [])
        total = int(resp_obj.get("total", 0))
        
        if not batch:
            break
            
        records.extend(batch)
        offset += len(batch)
        print(f"  Retrieved {offset} / {total} records...")
        
        if offset >= total:
            break
        time.sleep(0.3)
        
    return records

# COMMAND ----------
# DBTITLE 1,Fetch & Land into Bronze Delta Table
eia_data = fetch_eia_records(EIA_API_KEY, start_param, end_param)
print(f"Total EIA records fetched: {len(eia_data)}")

if eia_data:
    df_raw = spark.createDataFrame(eia_data)
    
    # Audit metadata
    df_bronze = df_raw \
        .withColumn("_ingested_at_utc", F.current_timestamp()) \
        .withColumn("_source", F.lit("EIA_v2_region_data"))
    
    # Append to Bronze table
    df_bronze.write \
        .format("delta") \
        .mode("append") \
        .option("mergeSchema", "true") \
        .saveAsTable(f"{CATALOG}.bronze.eia_hourly_raw")
        
    print(f"Appended {len(eia_data)} rows to {CATALOG}.bronze.eia_hourly_raw")
else:
    print("Warning: No records found for the specified window.")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT type, type_name, count(*) as row_count, min(period) as min_period, max(period) as max_period
# MAGIC FROM bronze.eia_hourly_raw
# MAGIC GROUP BY type, type_name;
