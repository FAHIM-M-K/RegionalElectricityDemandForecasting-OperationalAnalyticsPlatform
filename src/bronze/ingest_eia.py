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
import hashlib
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
spark.conf.set("spark.sql.session.timeZone", "UTC")

# COMMAND ----------
# DBTITLE 1,Load Configuration & Establish Boundaries
import sys
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
EIA_RESPONDENT = eia_cfg.get("respondent", "ERCO")
EIA_SERIES_TYPES = eia_cfg.get("series_types", ["D", "DF"])
EIA_PAGE_SIZE = eia_cfg.get("page_size", 5000)
EIA_BASE_URL = urllib.parse.urljoin(eia_cfg.get("base_url", "https://api.eia.gov/v2/"), eia_cfg.get("route", "electricity/rto/region-data/data/"))

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
# DBTITLE 1,Paginated EIA API Client with Full Envelope Preservation
def fetch_eia_dataset(api_key: str, start: str, end: str, series_types: list, respondent: str = "ERCO", page_size: int = 5000):
    """
    Fetches EIA records with pagination and retry backoff.
    Returns:
      - all_records: extracted list of row dictionaries for Bronze ingestion
      - raw_responses: list of raw response JSON envelopes for raw landing archive
    """
    all_records = []
    raw_responses = []
    offset = 0

    while True:
        # Build query params including explicit series facets (e.g. facets[type][]=D & facets[type][]=DF)
        query_params = [
            ("api_key", api_key),
            ("frequency", "hourly"),
            ("data[0]", "value"),
            ("facets[respondent][]", respondent),
            ("start", start),
            ("end", end),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "asc"),
            ("offset", str(offset)),
            ("length", str(page_size))
        ]

        # Explicitly filter only targeted series types (D: Demand, DF: Day-ahead Forecast)
        for stype in series_types:
            query_params.append(("facets[type][]", stype))

        encoded_query = urllib.parse.urlencode(query_params)
        request_url = f"{EIA_BASE_URL}?{encoded_query}"
        req = urllib.request.Request(request_url, headers={"User-Agent": "ERCOT-Platform/1.0"})

        payload = None
        for attempt in range(1, 4):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw_text = resp.read().decode("utf-8")
                    payload = json.loads(raw_text)
                    break
            except Exception as e:
                if attempt == 3:
                    raise RuntimeError(f"EIA API request failed at offset {offset}: {e}")
                time.sleep(2 * attempt)

        # Preserve sanitized query params (excluding api_key) for complete audit trail & replayability
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
        print(f"  Retrieved {offset} / {total} records (Series: {series_types})...")

        if offset >= total:
            break
        time.sleep(0.3)

    return all_records, raw_responses

# COMMAND ----------
# DBTITLE 1,Execute Fetch, Land Raw Envelope to Volume & Upsert Bronze
records, raw_payloads = fetch_eia_dataset(
    api_key=EIA_API_KEY,
    start=start_param,
    end=end_param,
    series_types=EIA_SERIES_TYPES,
    respondent=EIA_RESPONDENT,
    page_size=EIA_PAGE_SIZE
)

print(f"Total EIA records retrieved: {len(records)} across {len(raw_payloads)} API pages.")

if not records:
    raise RuntimeError(
        f"Bronze Ingestion Failure: 0 EIA records retrieved for window {start_date_str} to {end_date_str}. "
        f"Pipeline failed closed to prevent downstream silent stale tables (C-03)."
    )

# 1. Raw Archival: Store COMPLETE response payloads (envelope + sanitized request params + pagination)
raw_landing_dir = f"/Volumes/{CATALOG}/bronze/raw_landing/eia"
os.makedirs(raw_landing_dir, exist_ok=True)
timestamp_epoch = int(time.time())
raw_file_name = f"eia_erco_{start_date_str}_{end_date_str}_{timestamp_epoch}.json"
raw_file_path = f"{raw_landing_dir}/{raw_file_name}"

with open(raw_file_path, "w", encoding="utf-8") as f:
    json.dump(raw_payloads, f)
print(f"Archived authentic raw API responses to Volume: {raw_file_path}")

# 2. Convert to Spark DataFrame & sanitize column names
df_raw = spark.createDataFrame(records)
for c in df_raw.columns:
    if "-" in c:
        df_raw = df_raw.withColumnRenamed(c, c.replace("-", "_"))

df_bronze = df_raw \
    .withColumn("_ingested_at_utc", F.current_timestamp()) \
    .withColumn("_source", F.lit("EIA_v2_region_data")) \
    .withColumn("_raw_payload_path", F.lit(raw_file_path))

# 3. Data Integrity & Completeness Assertions
duplicate_keys = (df_bronze.groupBy("respondent", "period", "type")
                  .count().filter(F.col("count") > 1).count())
assert duplicate_keys == 0, f"Integrity Failure: Found {duplicate_keys} duplicate EIA source keys in this batch."
source_deduped = df_bronze

null_key_count = source_deduped.filter(
    F.col("respondent").isNull() | F.col("period").isNull() | F.col("type").isNull()
).count()
assert null_key_count == 0, f"Integrity Failure: Found {null_key_count} records with null primary keys."

distinct_types = set(row["type"] for row in source_deduped.select("type").distinct().collect())

# Assert that no unexpected types entered Bronze
for t in distinct_types:
    assert t in EIA_SERIES_TYPES, f"Schema Leak: Found unexpected series type '{t}' not in {EIA_SERIES_TYPES}."

# Assert that ALL expected types are present in the batch
for expected_type in EIA_SERIES_TYPES:
    assert expected_type in distinct_types, f"Completeness Failure: Expected series type '{expected_type}' was not returned by EIA API."

print(f"Quality Assertions Passed: 0 null keys. All expected series types confirmed present: {distinct_types}")

# 4. Idempotent Upsert into Bronze Delta table using Null-Safe Equality (<=>)
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
    # Null-safe inequality: updates only when value actually changed (handles null -> val, val -> null, val -> new_val)
    delta_table.alias("target").merge(
        source_deduped.alias("source"),
        merge_condition
    ).whenMatchedUpdateAll(
        condition="NOT (target.value <=> source.value)"
    ).whenNotMatchedInsertAll(
    ).execute()
    print(f"Idempotent MERGE completed on {TABLE_NAME} (null-safe equality enabled).")

# COMMAND ----------
# DBTITLE 2,Capture an as-observed EIA day-ahead benchmark vintage
# Reuse the response already fetched above. This preserves what the pipeline
# actually saw at this run time without issuing a second EIA API request.
BENCHMARK_TABLE = f"{CATALOG}.bronze.eia_forecast_vintages"
df_eia_forecast = df_bronze.filter(F.col("type") == F.lit("DF"))
if df_eia_forecast.limit(1).count() == 0:
    raise RuntimeError("Benchmark capture failed: the EIA response contained no DF day-ahead forecasts.")

observed_at = datetime.datetime.now(datetime.timezone.utc)
vintage_id = f"eia_df_{observed_at.strftime('%Y%m%dT%H%M%S%fZ')}"
forecast_payload_for_hash = [
    (str(row["period"]), float(row["value"]))
    for row in df_eia_forecast.select("period", "value").orderBy("period").collect()
    if row["value"] is not None
]
payload_sha256 = hashlib.sha256(
    json.dumps(forecast_payload_for_hash, separators=(",", ":")).encode("utf-8")
).hexdigest()

benchmark_rows = (
    df_eia_forecast
    .filter(F.col("value").isNotNull())
    .select(
        F.lit("EIA").alias("benchmark_provider"),
        F.lit("EIA-930-DF").alias("product_id"),
        F.col("respondent"),
        F.lit(vintage_id).alias("vintage_id"),
        F.to_timestamp(F.col("period"), "yyyy-MM-dd'T'HH").alias("target_timestamp_utc"),
        F.col("value").cast("double").alias("forecast_mw"),
        F.lit(None).cast("timestamp").alias("published_at_utc"),
        F.lit(observed_at).cast("timestamp").alias("first_observed_at_utc"),
        F.lit(payload_sha256).alias("payload_sha256"),
        F.lit(raw_file_path).alias("raw_payload_path"),
    )
)
invalid_benchmark_rows = benchmark_rows.filter(
    F.col("respondent").isNull() | F.col("target_timestamp_utc").isNull()
    | F.col("forecast_mw").isNull() | (F.col("forecast_mw") <= 0)
).limit(1).count()
if invalid_benchmark_rows:
    raise RuntimeError("EIA benchmark vintage contains null keys or invalid forecast values.")

if not spark.catalog.tableExists(BENCHMARK_TABLE):
    benchmark_rows.write.format("delta").mode("overwrite").saveAsTable(BENCHMARK_TABLE)
else:
    benchmark_delta = DeltaTable.forName(spark, BENCHMARK_TABLE)
    benchmark_delta.alias("target").merge(
        benchmark_rows.alias("source"),
        "target.benchmark_provider = source.benchmark_provider "
        "AND target.respondent = source.respondent "
        "AND target.vintage_id = source.vintage_id "
        "AND target.target_timestamp_utc = source.target_timestamp_utc",
    ).whenNotMatchedInsertAll().execute()
print(
    f"Captured {len(forecast_payload_for_hash):,} EIA forecast vintage rows "
    f"as observed at {observed_at.isoformat()} in {BENCHMARK_TABLE}."
)

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT type, type_name, count(*) as row_count, min(period) as min_period, max(period) as max_period
# MAGIC FROM bronze.eia_hourly_raw
# MAGIC GROUP BY type, type_name;
