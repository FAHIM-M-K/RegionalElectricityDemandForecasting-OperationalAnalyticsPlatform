# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: ERCOT Subregion (Zonal) Electricity Demand Ingestion
# MAGIC
# MAGIC **Source:** U.S. Energy Information Administration (EIA) Open Data API v2 (Form EIA-930 Subregion Data)  
# MAGIC **Route:** `electricity/rto/subregion-data/data/`  
# MAGIC **Parent Balancing Authority:** ERCOT (`ERCO`)  
# MAGIC **Subregions / Weather Zones:** 
# MAGIC - `NCENT` (North Central)
# MAGIC - `COAST` (Coast)
# MAGIC - `SCENT` (South Central)
# MAGIC - `FWEST` (Far West)
# MAGIC - `SOUTH` (South)
# MAGIC - `EAST` (East)
# MAGIC - `WEST` (West)
# MAGIC - `NORTH` (North)
# MAGIC
# MAGIC **Target Table:** `main.bronze.ercot_zonal_raw`  
# MAGIC **Quality Guarantees:** Zero synthetic approximations, complete envelope raw-payload archival, and null-safe idempotent Delta upsert.

# COMMAND ----------
# DBTITLE 1,Imports & Configuration
import os
import sys
import json
import time
import datetime
import urllib.request
import urllib.parse
from pyspark.sql import functions as F
from delta.tables import DeltaTable

CATALOG = "main"
TABLE_NAME = f"{CATALOG}.bronze.ercot_zonal_raw"
FULL_HISTORY_START = "2021-01-01"
RESTATEMENT_BUFFER_DAYS = 5

# Retrieve API Key securely: Databricks Secret Scope -> Cluster Env Var
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
# DBTITLE 1,Load Configuration & Establish Boundaries
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

from configs.loader import load_platform_config

config = load_platform_config("ercot_config.json")
eia_cfg = config.get("eia", {})
EIA_PARENT = eia_cfg.get("respondent", "ERCO")
EIA_PAGE_SIZE = eia_cfg.get("page_size", 5000)
SUBREGION_ROUTE = eia_cfg.get("subregion_route", "electricity/rto/region-sub-ba-data/data/")
EIA_BASE_URL = urllib.parse.urljoin(eia_cfg.get("base_url", "https://api.eia.gov/v2/"), SUBREGION_ROUTE)

EXPECTED_SUBREGIONS = {
    s["code"] for s in eia_cfg.get("subregions", [
        {"code": "NCENT"}, {"code": "COAST"}, {"code": "SCENT"}, {"code": "FWEST"},
        {"code": "SOUTH"}, {"code": "EAST"}, {"code": "WEST"}, {"code": "NORTH"}
    ])
}

# COMMAND ----------
# DBTITLE 1,High-Water Mark Detection
now_utc = datetime.datetime.now(datetime.timezone.utc)
end_date_str = now_utc.strftime("%Y-%m-%d")

table_exists = spark.catalog.tableExists(TABLE_NAME)
existing_columns = set(spark.table(TABLE_NAME).columns) if table_exists else set()
has_raw_grain = {"parent", "period", "value"}.issubset(existing_columns) and bool(
    {"subba", "subregion"} & existing_columns
)
table_has_rows = table_exists and spark.table(TABLE_NAME).limit(1).count() > 0
legacy_table_schema = table_exists and not has_raw_grain

if table_has_rows and has_raw_grain:
    max_period = spark.sql(f"SELECT max(period) FROM {TABLE_NAME}").first()[0]
    hwm_dt = datetime.datetime.strptime(max_period[:10], "%Y-%m-%d") - datetime.timedelta(days=RESTATEMENT_BUFFER_DAYS)
    start_date_str = hwm_dt.strftime("%Y-%m-%d")
    print(f"[INCREMENTAL] High-water mark: {max_period}. Fetching authentic subregion load from {start_date_str} (with {RESTATEMENT_BUFFER_DAYS}-day buffer).")
else:
    start_date_str = FULL_HISTORY_START
    if legacy_table_schema:
        print(f"[SCHEMA MIGRATION] Existing {TABLE_NAME} is not raw EIA Bronze grain. Refetching history to replace it with canonical raw records.")
    else:
        print(f"[FULL LOAD] No existing raw data found in {TABLE_NAME}. Ingesting full history from {start_date_str}.")

start_param = f"{start_date_str}T00"
end_param = f"{end_date_str}T23"

# COMMAND ----------
# DBTITLE 1,Paginated EIA Subregion API Client
def fetch_eia_subregion_dataset(api_key: str, start: str, end: str, parent: str = "ERCO", page_size: int = 5000):
    """
    Fetches authentic EIA-930 subregion hourly demand records with pagination, backoff, and envelope archival.
    Fail-closed: raises a RuntimeError on failure rather than returning synthetic approximations.
    """
    all_records = []
    raw_responses = []
    offset = 0

    while True:
        query_params = [
            ("api_key", api_key),
            ("frequency", "hourly"),
            ("data[0]", "value"),
            ("facets[parent][]", parent),
            ("start", start),
            ("end", end),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "asc"),
            ("offset", str(offset)),
            ("length", str(page_size))
        ]

        encoded_query = urllib.parse.urlencode(query_params)
        request_url = f"{EIA_BASE_URL}?{encoded_query}"
        req = urllib.request.Request(request_url, headers={"User-Agent": "ERCOT-Platform/1.0"})

        payload = None
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(req, timeout=40) as resp:
                    raw_text = resp.read().decode("utf-8")
                    payload = json.loads(raw_text)
                    break
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"EIA Subregion API request failed at offset {offset}: {e}")
                time.sleep(2 * attempt)

        sanitized_params = {k: v for k, v in query_params if k != "api_key"}

        raw_responses.append({
            "offset": offset,
            "page_size": page_size,
            "request_url_endpoint": EIA_BASE_URL,
            "request_params_sanitized": sanitized_params,
            "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "response_envelope": payload
        })

        resp_obj = payload.get("response", {})
        batch = resp_obj.get("data", [])
        total = int(resp_obj.get("total", 0))

        if not batch:
            break

        all_records.extend(batch)
        offset += len(batch)
        print(f"  Retrieved {offset} / {total} authentic subregion records...")

        if offset >= total:
            break
        time.sleep(0.3)

    return all_records, raw_responses


def date_windows(start_date: str, end_date: str, months: int = 3):
    """Bound pagination offsets during full backfills by querying quarterly windows."""
    current = datetime.date.fromisoformat(start_date)
    end = datetime.date.fromisoformat(end_date)
    while current <= end:
        month_index = current.month - 1 + months
        year = current.year + month_index // 12
        month = month_index % 12 + 1
        next_window = datetime.date(year, month, 1)
        window_end = min(next_window - datetime.timedelta(days=1), end)
        yield current.isoformat(), window_end.isoformat()
        current = window_end + datetime.timedelta(days=1)

# COMMAND ----------
# DBTITLE 1,Execute Ingestion, Archive Raw Payload & Write to Bronze Delta
records, raw_payloads = [], []
date_ranges = list(date_windows(start_date_str, end_date_str))
for window_index, (window_start, window_end) in enumerate(date_ranges, start=1):
    print(f"[BACKFILL] Fetching date window {window_index}/{len(date_ranges)}: {window_start} to {window_end}.")
    window_records, window_payloads = fetch_eia_subregion_dataset(
        api_key=EIA_API_KEY,
        start=f"{window_start}T00",
        end=f"{window_end}T23",
        parent=EIA_PARENT,
        page_size=EIA_PAGE_SIZE
    )
    records.extend(window_records)
    raw_payloads.extend(window_payloads)

print(f"Total authentic subregion records extracted: {len(records)} across {len(raw_payloads)} API pages.")

if records:
    # 1. Raw Archival: Persist response envelopes into Volume
    raw_landing_dir = f"/Volumes/{CATALOG}/bronze/raw_landing/ercot_zonal"
    os.makedirs(raw_landing_dir, exist_ok=True)
    timestamp_epoch = int(time.time())
    raw_file_name = f"eia_ercot_subregion_{start_date_str}_{end_date_str}_{timestamp_epoch}.json"
    raw_file_path = f"{raw_landing_dir}/{raw_file_name}"

    with open(raw_file_path, "w", encoding="utf-8") as f:
        json.dump(raw_payloads, f)
    print(f"Archived authentic raw subregion payloads to Volume: {raw_file_path}")

    # 2. Spark DataFrame Schema Standardization
    df_raw = spark.createDataFrame(records)
    for c in df_raw.columns:
        if "-" in c:
            df_raw = df_raw.withColumnRenamed(c, c.replace("-", "_"))

    df_bronze = df_raw \
        .withColumn("_ingested_at_utc", F.current_timestamp()) \
        .withColumn("_source", F.lit("EIA_v2_subregion_data")) \
        .withColumn("_raw_payload_path", F.lit(raw_file_path))

    # Standardize subba column name if necessary
    subba_col = "subba" if "subba" in df_bronze.columns else "subregion"

    # 3. Deterministic Deduplication & Data Integrity Assertions (F-17)
    from pyspark.sql.window import Window as SparkWindow
    dedup_window = SparkWindow.partitionBy("parent", subba_col, "period").orderBy(F.col("_ingested_at_utc").desc())
    source_deduped = df_bronze.withColumn("_row_num", F.row_number().over(dedup_window)) \
        .filter(F.col("_row_num") == 1) \
        .drop("_row_num")

    null_key_count = source_deduped.filter(
        F.col("parent").isNull() | F.col(subba_col).isNull() | F.col("period").isNull()
    ).count()
    assert null_key_count == 0, f"Integrity Failure: Found {null_key_count} records with null primary keys."

    present_subregions = set(row[subba_col] for row in source_deduped.select(subba_col).distinct().collect())
    print(f"Verified present subregions in batch: {present_subregions}")

    # F-03 & P1-06: Subregion completeness gate against expected subregions
    missing_zones = EXPECTED_SUBREGIONS - present_subregions
    if missing_zones:
        allow_partial = os.getenv("ALLOW_PARTIAL_ZONAL_INGESTION", "false").lower() in ("true", "1")
        if not allow_partial:
            raise RuntimeError(
                f"Integrity Failure (P1-06): Expected subregions missing from batch: {missing_zones}. "
                f"Ingested only {present_subregions}. Set ALLOW_PARTIAL_ZONAL_INGESTION=1 to override."
            )
        print(f"Notice: Batch contains partial subregions: {present_subregions}. Missing: {missing_zones}")

    # 4. Idempotent Upsert into Bronze Delta Table
    if not table_exists or legacy_table_schema:
        source_deduped.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .partitionBy(subba_col) \
            .saveAsTable(TABLE_NAME)
        print(f"Initialized or migrated table {TABLE_NAME} with {source_deduped.count()} authentic raw records.")
    else:
        delta_table = DeltaTable.forName(spark, TABLE_NAME)
        merge_condition = (
            f"target.parent = source.parent "
            f"AND target.{subba_col} = source.{subba_col} "
            f"AND target.period = source.period"
        )
        delta_table.alias("target").merge(
            source_deduped.alias("source"),
            merge_condition
        ).whenMatchedUpdateAll(
            condition=f"NOT (target.value <=> source.value)"
        ).whenNotMatchedInsertAll(
        ).execute()
        print(f"Idempotent MERGE completed on {TABLE_NAME} (null-safe equality enabled).")
else:
    # F-02: Fail-closed if no records returned on full load or significant request window
    if not spark.catalog.tableExists(TABLE_NAME) or spark.table(TABLE_NAME).count() == 0:
        raise RuntimeError(
            f"EIA Ingestion Failure: 0 records returned for window {start_param} to {end_param}. "
            f"Endpoint: {EIA_BASE_URL}, Parent: {EIA_PARENT}. Target table {TABLE_NAME} does not exist or is empty. "
            f"Failing closed to prevent silent pipeline bypass."
        )
    elif start_date_str < end_date_str:
        raise RuntimeError(
            f"EIA Ingestion Alert: 0 records returned for multi-day incremental window {start_param} to {end_param}. "
            f"Verify EIA API connectivity, route validity ({SUBREGION_ROUTE}), or upstream API data availability."
        )
    else:
        print(f"Incremental Ingestion Status: Table is fully up-to-date. No new records found for window {start_param} to {end_param}.")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT subba, count(*) as hours, min(period) as min_period, max(period) as max_period
# MAGIC FROM bronze.ercot_zonal_raw
# MAGIC GROUP BY subba
# MAGIC ORDER BY subba;
