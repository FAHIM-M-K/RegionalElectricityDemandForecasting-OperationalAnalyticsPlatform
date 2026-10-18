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
import email.utils
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError
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
from src.shared.ercot_zones import canonical_zone_code, source_to_zone_code_map

SOURCE_TO_ZONE_CODE = source_to_zone_code_map(eia_cfg.get("subregions", []))

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
class RetryableEIARequestError(RuntimeError):
    """Transient EIA/API gateway failure after bounded retries."""


class IncompleteEIAWindowError(RuntimeError):
    """A completed response window omitted one or more configured zones."""


def validate_window_subregions(records, start: str, end: str):
    """Fail a window before caching/writing it if any configured zone is absent."""
    received = {
        canonical_zone_code(record.get("subba") or record.get("subregion") or "", SOURCE_TO_ZONE_CODE)
        for record in records
    }
    received.discard("")
    missing = EXPECTED_SUBREGIONS - received
    if missing:
        raise IncompleteEIAWindowError(
            f"EIA window {start} to {end} is incomplete: missing subregions {sorted(missing)}; "
            f"received {sorted(received)}. Window was not checkpointed or written."
        )


def fetch_eia_subregion_dataset(
    api_key: str,
    start: str,
    end: str,
    parent: str = "ERCO",
    page_size: int = 5000,
):
    """
    Fetches all authentic EIA-930 subregions for one parent and date window.

    Keep this request unfiltered by subba: the EIA route has returned empty
    results for valid ERCOT subba facets even though the parent-level query
    contains those zones. Validate returned subba codes after fetching instead
    of trusting a per-zone facet filter.
    Fail-closed: raises a RuntimeError on failure rather than returning synthetic approximations.
    """
    def fetch_page(offset: int):
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

        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw_text = resp.read().decode("utf-8")
                    payload = json.loads(raw_text)
                    break
            except HTTPError as e:
                retryable = e.code == 429 or 500 <= e.code <= 599
                if not retryable:
                    raise RuntimeError(f"EIA Subregion API request failed at offset {offset}: {e}")
                if attempt == max_attempts:
                    raise RetryableEIARequestError(
                        f"EIA Subregion API request failed at offset {offset} after {max_attempts} attempts: {e}"
                    ) from e
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    delay = float(retry_after) if retry_after else min(30, 2 ** attempt)
                except ValueError:
                    try:
                        retry_at = email.utils.parsedate_to_datetime(retry_after)
                        delay = max(0, (retry_at - datetime.datetime.now(retry_at.tzinfo)).total_seconds())
                    except (TypeError, ValueError, OverflowError):
                        delay = min(30, 2 ** attempt)
                delay = max(1, min(delay, 60))
                print(f"  Transient EIA HTTP {e.code} at offset {offset}; retry {attempt + 1}/{max_attempts} in {delay:g}s.")
                time.sleep(delay)
            except (URLError, TimeoutError, json.JSONDecodeError) as e:
                if attempt == max_attempts:
                    raise RetryableEIARequestError(
                        f"EIA Subregion API request failed at offset {offset} after {max_attempts} attempts: {e}"
                    ) from e
                delay = min(30, 2 ** attempt)
                print(f"  Transient EIA request error at offset {offset}; retry {attempt + 1}/{max_attempts} in {delay}s: {e}")
                time.sleep(delay)

        sanitized_params = {k: v for k, v in query_params if k != "api_key"}

        response = payload.get("response", {})
        return response.get("data", []), int(response.get("total", 0)), {
            "offset": offset,
            "page_size": page_size,
            "request_url_endpoint": EIA_BASE_URL,
            "request_params_sanitized": sanitized_params,
            "fetched_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "response_envelope": payload,
        }

    first_batch, total, first_raw = fetch_page(0)
    all_records = list(first_batch)
    raw_responses = [first_raw] if first_batch else []
    offsets = range(page_size, total, page_size)
    if offsets:
        # EIA pages within a fixed historical window are independent. Fetch a
        # few at once to reduce backfill wall time without launching an
        # unbounded request storm; each page still has bounded retries.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(fetch_page, offset): offset for offset in offsets}
            for future in as_completed(futures):
                batch, _, raw_page = future.result()
                all_records.extend(batch)
                raw_responses.append(raw_page)
                print(f"  Retrieved {len(all_records)} / {total} authentic subregion records...")
    return all_records, raw_responses


def fetch_window_with_split_fallback(start: str, end: str, depth: int = 0):
    """Split a date window after exhausted transient errors, then retry smaller queries."""
    try:
        return fetch_eia_subregion_dataset(
            EIA_API_KEY, f"{start}T00", f"{end}T23", EIA_PARENT, EIA_PAGE_SIZE
        )
    except RetryableEIARequestError:
        start_date = datetime.date.fromisoformat(start)
        end_date = datetime.date.fromisoformat(end)
        if depth >= 2 or start_date >= end_date:
            raise
        midpoint = start_date + (end_date - start_date) // 2
        left_start, left_end = start_date.isoformat(), midpoint.isoformat()
        right_start = (midpoint + datetime.timedelta(days=1)).isoformat()
        right_end = end_date.isoformat()
        print(f"  EIA remains unavailable for {start} to {end}; splitting into smaller date requests.")
        left_records, left_pages = fetch_window_with_split_fallback(left_start, left_end, depth + 1)
        right_records, right_pages = fetch_window_with_split_fallback(right_start, right_end, depth + 1)
        return left_records + right_records, left_pages + right_pages


def date_windows(start_date: str, end_date: str, months: int = 1):
    """Use monthly windows to keep EIA queries small and pagination shallow."""
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
checkpoint_dir = f"/Volumes/{CATALOG}/bronze/raw_landing/ercot_zonal/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
resume_backfill = (not table_has_rows) or legacy_table_schema
for window_index, (window_start, window_end) in enumerate(date_ranges, start=1):
    checkpoint_path = os.path.join(checkpoint_dir, f"{window_start}_{window_end}.json")
    window_records, window_payloads = [], []
    reused_checkpoint = False

    # A failed first-time/legacy-schema backfill must not repeat already
    # completed API windows. Reuse only checkpoints from the last 30 days;
    # regular incremental runs always refetch their restatement window.
    if resume_backfill and os.path.isfile(checkpoint_path):
        checkpoint_age_days = (time.time() - os.path.getmtime(checkpoint_path)) / 86400
        if checkpoint_age_days <= 30:
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as checkpoint_file:
                    checkpoint = json.load(checkpoint_file)
                if (checkpoint.get("version") == 1
                        and checkpoint.get("start") == window_start
                        and checkpoint.get("end") == window_end):
                    window_payloads = checkpoint["pages"]
                    window_records = [
                        record
                        for page in window_payloads
                        for record in page["response_envelope"]["response"]["data"]
                    ]
                    validate_window_subregions(window_records, window_start, window_end)
                    reused_checkpoint = bool(window_records)
            except (OSError, ValueError, KeyError, TypeError, IncompleteEIAWindowError):
                # An incomplete checkpoint from an interrupted write is safely
                # replaced by a fresh API fetch below.
                window_records, window_payloads, reused_checkpoint = [], [], False

    if reused_checkpoint:
        print(f"[RESUME] Reusing {len(window_records):,} cached records for {window_start} to {window_end}.")
    else:
        print(f"[BACKFILL] Fetching date window {window_index}/{len(date_ranges)}: {window_start} to {window_end}.")
        window_records, window_payloads = fetch_window_with_split_fallback(window_start, window_end)
        validate_window_subregions(window_records, window_start, window_end)
        if resume_backfill and window_records:
            temp_path = f"{checkpoint_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as checkpoint_file:
                json.dump({"version": 1, "start": window_start, "end": window_end,
                           "pages": window_payloads}, checkpoint_file)
            os.replace(temp_path, checkpoint_path)

    records.extend(window_records)
    raw_payloads.extend(window_payloads)
    print(f"  Retrieved {len(window_records):,} records across configured ERCOT subregions.")

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

    present_source_codes = set(row[subba_col] for row in source_deduped.select(subba_col).distinct().collect())
    present_subregions = {canonical_zone_code(code, SOURCE_TO_ZONE_CODE) for code in present_source_codes}
    print(f"Verified EIA source codes: {present_source_codes}; canonical zones: {present_subregions}")
    source_deduped.groupBy(subba_col).count().orderBy(subba_col).show(truncate=False)

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
