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
# DBTITLE 1,Imports & Configuration
import os
import json
import time
import datetime
import urllib.request
import urllib.parse
from pyspark.sql import functions as F
from delta.tables import DeltaTable

CATALOG = "main"
TABLE_NAME = f"{CATALOG}.bronze.eia_hourly_raw"
FULL_HISTORY_START = "2021-01-01"  # Earliest date for initial full load
RESTATEMENT_BUFFER_DAYS = 5       # Overlap to catch EIA corrections on recent hours

# Retrieve API Key securely: Secret Scope -> Cluster Env Var
EIA_API_KEY = None
try:
    EIA_API_KEY = dbutils.secrets.get(scope="grid_platform", key="eia_api_key")
except Exception:
    pass

if not EIA_API_KEY:
    EIA_API_KEY = os.getenv("EIA_API_KEY", "")

assert EIA_API_KEY, (
    "EIA_API_KEY is required. Please ensure it is stored in Databricks Secret Scope "
    "(scope='grid_platform', key='eia_api_key') or as cluster environment variable 'EIA_API_KEY'."
)

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA bronze")

# COMMAND ----------
# DBTITLE 1,High-Water Mark: Auto-Detect Full Load vs Incremental
now_utc = datetime.datetime.now(datetime.timezone.utc)
end_date_str = now_utc.strftime("%Y-%m-%d")

if spark.catalog.tableExists(TABLE_NAME) and spark.table(TABLE_NAME).count() > 0:
    # Table exists with data: incremental from high-water mark minus restatement buffer
    max_period = spark.sql(f"SELECT max(period) FROM {TABLE_NAME}").first()[0]
    hwm_dt = datetime.datetime.strptime(max_period[:10], "%Y-%m-%d") - datetime.timedelta(days=RESTATEMENT_BUFFER_DAYS)
    start_date_str = hwm_dt.strftime("%Y-%m-%d")
    print(f"[INCREMENTAL] High-water mark: {max_period}. Fetching from {start_date_str} (with {RESTATEMENT_BUFFER_DAYS}-day restatement buffer).")
else:
    # First run or empty table: full historical load
    start_date_str = FULL_HISTORY_START
    print(f"[FULL LOAD] No existing data found. Loading full history from {start_date_str}.")

start_param = f"{start_date_str}T00"
end_param = f"{end_date_str}T23"

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
# DBTITLE 1,Land Raw to Volume & Append to Bronze Delta Table
eia_data = fetch_eia_records(EIA_API_KEY, start_param, end_param)
print(f"Total EIA records fetched: {len(eia_data)}")

if eia_data:
    # 1. Production Landing: Save raw JSON payload to Volume for replayability & audit
    raw_landing_dir = f"/Volumes/{CATALOG}/bronze/raw_landing/eia"
    os.makedirs(raw_landing_dir, exist_ok=True)
    timestamp_epoch = int(time.time())
    raw_file_name = f"eia_erco_{start_date_str}_{end_date_str}_{timestamp_epoch}.json"
    raw_file_path = f"{raw_landing_dir}/{raw_file_name}"
    
    with open(raw_file_path, "w", encoding="utf-8") as f:
        json.dump(eia_data, f)
    print(f"Archived raw source payload to Volume: {raw_file_path}")

    # 2. Convert to Spark DataFrame & sanitize hyphenated column names (e.g., type-name -> type_name)
    df_raw = spark.createDataFrame(eia_data)
    for c in df_raw.columns:
        if "-" in c:
            df_raw = df_raw.withColumnRenamed(c, c.replace("-", "_"))

    df_bronze = df_raw \
        .withColumn("_ingested_at_utc", F.current_timestamp()) \
        .withColumn("_source", F.lit("EIA_v2_region_data")) \
        .withColumn("_raw_payload_path", F.lit(raw_file_path))
    
    # 3. Idempotent Upsert into Bronze Delta table
    source_deduped = df_bronze.dropDuplicates(["respondent", "period", "type"])
    
    if not spark.catalog.tableExists(TABLE_NAME):
        source_deduped.write \
            .format("delta") \
            .mode("overwrite") \
            .saveAsTable(TABLE_NAME)
        print(f"Initialized table {TABLE_NAME} with {source_deduped.count()} records.")
    else:
        delta_table = DeltaTable.forName(spark, TABLE_NAME)
        merge_condition = (
            "target.respondent = source.respondent "
            "AND target.period = source.period "
            "AND target.type = source.type"
        )
        delta_table.alias("target").merge(
            source_deduped.alias("source"),
            merge_condition
        ).whenMatchedUpdateAll(
            condition="target.value != source.value OR (target.value IS NULL AND source.value IS NOT NULL)"
        ).whenNotMatchedInsertAll(
        ).execute()
        print(f"Idempotent MERGE completed on {TABLE_NAME}.")
else:
    print("Warning: No records found for the specified window.")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT type, type_name, count(*) as row_count, min(period) as min_period, max(period) as max_period
# MAGIC FROM bronze.eia_hourly_raw
# MAGIC GROUP BY type, type_name;
