# Databricks notebook source
"""Capture ERCOT's official prospective seven-day system-load benchmark.

Product NP3-561-CD contains hourly forecasts by weather zone and system total.
The parser deliberately accepts only an explicitly identified system-total row;
it never adds regional forecasts together or fabricates an empty success result.

Credentials are read from Databricks secret scope ``grid_platform``:
``ercot_api_subscription_key``, ``ercot_api_username``, ``ercot_api_password``.
An optional ``ERCOT_BENCHMARK_ENDPOINT`` may point at the product artifact URL
shown by ERCOT's API metadata. No credentials or forecast data are hardcoded.
"""

import csv
import hashlib
import io
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

CATALOG = "main"
TARGET_TABLE = f"{CATALOG}.bronze.ercot_benchmark_raw"
PRODUCT_ID = "NP3-561-CD"
API_ROOT = "https://api.ercot.com/api/public-reports"
AUTH_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
AUTH_PARAMS = {
    "grant_type": "password",
    "scope": "openid fec253ea-0d06-4272-a5e6-b478baeecd70 offline_access",
    "client_id": "fec253ea-0d06-4272-a5e6-b478baeecd70",
    "response_type": "id_token",
}
CENTRAL = ZoneInfo("America/Chicago")


def _normalized(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _parse_timestamp(value):
    if value is None or str(value).strip() == "":
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _local_central_to_utc_unambiguous(local_naive: datetime) -> datetime:
    """Convert a local market interval only when DST maps it to one real hour."""
    first = local_naive.replace(tzinfo=CENTRAL, fold=0)
    second = local_naive.replace(tzinfo=CENTRAL, fold=1)
    first_valid = first.astimezone(timezone.utc).astimezone(CENTRAL).replace(tzinfo=None) == local_naive
    second_valid = second.astimezone(timezone.utc).astimezone(CENTRAL).replace(tzinfo=None) == local_naive
    if not first_valid and not second_valid:
        raise ValueError(
            f"ERCOT local interval {local_naive} does not exist during the spring-forward DST transition."
        )
    if first_valid and second_valid and first.utcoffset() != second.utcoffset():
        raise ValueError(
            f"ERCOT local interval {local_naive} is ambiguous during the fall-back DST transition; "
            "the artifact must provide an explicit UTC timestamp or fold indicator."
        )
    return (first if first_valid else second).astimezone(timezone.utc)


def _read_payload_records(raw_bytes: bytes, content_type: str = "") -> list[dict]:
    """Decode API JSON, CSV, XML, or a ZIP containing one supported artifact."""
    if raw_bytes.startswith(b"PK\x03\x04"):
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            supported = [n for n in archive.namelist() if n.lower().endswith((".csv", ".json", ".xml"))]
            if len(supported) != 1:
                raise ValueError(f"ERCOT ZIP must contain one CSV/JSON/XML data file; found {supported}.")
            raw_bytes = archive.read(supported[0])
    decoded = raw_bytes.decode("utf-8-sig")
    try:
        payload = json.loads(decoded)
    except json.JSONDecodeError:
        if decoded.lstrip().startswith("<"):
            root = ET.fromstring(decoded)
            candidates = []
            for element in root.iter():
                fields = {}
                for key, value in element.attrib.items():
                    fields[key.rsplit("}", 1)[-1]] = value
                for child in list(element):
                    if len(child) == 0 and child.text is not None:
                        fields[child.tag.rsplit("}", 1)[-1]] = child.text.strip()
                if fields:
                    candidates.append(fields)
            if not candidates:
                raise ValueError("ERCOT XML response contained no flat row records.")
            return candidates
        rows = list(csv.DictReader(io.StringIO(decoded)))
        if not rows:
            raise ValueError("ERCOT response was neither supported JSON nor non-empty CSV.")
        return rows

    def find_row_lists(node):
        if isinstance(node, list):
            if node and all(isinstance(item, dict) for item in node):
                yield node
            for item in node:
                yield from find_row_lists(item)
        elif isinstance(node, dict):
            for item in node.values():
                yield from find_row_lists(item)

    candidates = list(find_row_lists(payload))
    if not candidates:
        raise ValueError("ERCOT JSON response contained no row records.")
    # Select the largest row array; nested product metadata may contain smaller arrays.
    return max(candidates, key=len)


def _first(row: dict, names: tuple[str, ...]):
    normalized = {_normalized(k): v for k, v in row.items()}
    for name in names:
        if _normalized(name) in normalized:
            return normalized[_normalized(name)]
    return None


def parse_system_forecast_records(
    raw_bytes: bytes,
    retrieved_at: datetime | None = None,
    content_type: str = "",
    product_id: str = PRODUCT_ID,
) -> list[dict]:
    """Parse and validate system-wide forecast rows from an ERCOT artifact."""
    retrieved_at = retrieved_at or datetime.now(timezone.utc)
    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.replace(tzinfo=timezone.utc)
    retrieved_at = retrieved_at.astimezone(timezone.utc)
    payload_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    items = _read_payload_records(raw_bytes, content_type)
    normalized_rows = []
    seen_local_intervals = set()
    for row in items:
        # NP3-561-CD contains both weather-zone values and system total. If a
        # zone/type field exists, accept only a clearly labeled system total.
        scope_value = _first(row, (
            "weatherZone", "weatherZoneName", "zoneName", "forecastZone",
            "forecastZoneName", "areaName", "loadZone",
        ))
        if scope_value is not None:
            scope = _normalized(scope_value)
            if not any(token in scope for token in ("systemtotal", "ercottotal", "ercotsystem", "systemwide")):
                continue

        target_value = _first(row, (
            "target_timestamp_utc", "targetTimestampUtc", "utcTimestamp",
            "timestampUtc", "intervalStartUtc", "deliveryDateTimeUtc",
        ))
        target_ts = _parse_timestamp(target_value)
        if target_ts is None:
            local_date = _first(row, ("deliveryDate", "operatingDate", "date", "flowDate"))
            hour_ending = _first(row, ("hourEnding", "hourEndingValue", "he", "hour"))
            if local_date is None or hour_ending is None:
                continue
            date_text = str(local_date).strip()[:10]
            hour_text = str(hour_ending).strip()
            hour_match = re.search(r"\d{1,2}", hour_text)
            if not hour_match:
                continue
            hour = int(hour_match.group())
            # Hour-ending 1 means the interval starting at 00:00 local.
            # Reject DST gaps/folds unless the source supplies explicit UTC.
            if hour < 1 or hour > 24:
                continue
            local_interval = (date_text, hour)
            if local_interval in seen_local_intervals:
                raise ValueError(
                    "ERCOT artifact contains a repeated local hour without an explicit UTC timestamp; "
                    "cannot safely resolve the daylight-saving fold."
                )
            seen_local_intervals.add(local_interval)
            local_start_naive = datetime.fromisoformat(date_text) + timedelta(hours=hour - 1)
            target_ts = _local_central_to_utc_unambiguous(local_start_naive)

        value = _first(row, (
            "forecast_mw", "forecastDemandMw", "systemTotalForecastMw",
            "systemTotalLoadForecast", "forecastLoad", "loadForecast",
            "demandForecast", "totalForecastMw", "systemLoadMw", "systemTotal",
            "systemTotalMw", "ercotSystemTotal", "ercotTotalMw",
        ))
        if value is None:
            continue
        try:
            forecast_mw = float(str(value).replace(",", "").strip())
        except (TypeError, ValueError):
            continue
        if not 10_000 <= forecast_mw <= 110_000:
            raise ValueError(f"ERCOT system forecast value {forecast_mw} MW is outside physical bounds.")
        published_value = _first(row, (
            "published_at_utc", "postedDatetime", "publishedDatetime", "issueTimeUtc", "forecastIssueTime",
        ))
        published_at = _parse_timestamp(published_value)
        normalized_rows.append((target_ts, forecast_mw, published_at))

    if not normalized_rows:
        fields = sorted({_normalized(key) for row in items[:3] for key in row})
        raise ValueError(
            "ERCOT artifact has no recognizable system-total hourly rows. "
            f"Inspect product schema before enabling capture; sample fields={fields}."
        )
    issue_times = {published for _, _, published in normalized_rows if published is not None}
    if len(issue_times) > 1:
        raise ValueError("ERCOT artifact mixes forecast publication times; split it into provider issues first.")
    published_at = next(iter(issue_times), None)
    vintage_label = published_at.strftime("%Y%m%dT%H%M%SZ") if published_at else "unpublished"
    # Keep provider corrections that reuse a publication timestamp as separate
    # immutable vintages. Re-fetching identical bytes remains idempotent.
    vintage_id = f"ercot_{vintage_label}_{payload_sha256[:16]}"

    by_target = {}
    for target_ts, forecast_mw, _ in normalized_rows:
        if target_ts in by_target and by_target[target_ts] != forecast_mw:
            raise ValueError(f"ERCOT artifact has conflicting system forecasts for {target_ts}.")
        by_target[target_ts] = forecast_mw
    return [{
        "benchmark_provider": "ERCOT",
        "product_id": product_id,
        "respondent": "ERCO",
        "vintage_id": vintage_id,
        "target_timestamp_utc": target_ts,
        "forecast_mw": forecast_mw,
        "published_at_utc": published_at,
        "first_observed_at_utc": retrieved_at,
        "payload_sha256": payload_sha256,
    } for target_ts, forecast_mw in sorted(by_target.items())]


def _get_secret(name: str) -> str:
    try:
        return dbutils.secrets.get(scope="grid_platform", key=name)
    except Exception:
        return os.getenv(name.upper(), "")


def get_ercot_token(username: str, password: str) -> str:
    params = {**AUTH_PARAMS, "username": username, "password": password}
    request = urllib.request.Request(
        AUTH_URL,
        data=urllib.parse.urlencode(params).encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "ERCOT-demand-platform/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    # ERCOT Public API's own guide uses the ID token in the Bearer header.
    token = payload.get("id_token") or payload.get("access_token")
    if not token:
        raise RuntimeError("ERCOT authentication response contained no bearer token.")
    return token


def fetch_latest_artifact(subscription_key: str, bearer_token: str, endpoint: str | None = None):
    """Resolve NP3-561-CD artifact metadata, fetch an artifact, and return bytes."""
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Ocp-Apim-Subscription-Key": subscription_key,
        "Accept": "application/json, text/csv, application/zip, */*",
        "User-Agent": "ERCOT-demand-platform/1.0",
    }
    product_url = endpoint or f"{API_ROOT}/{PRODUCT_ID.lower()}"
    request = urllib.request.Request(product_url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
        final_url = response.geturl()

    # Metadata responses provide artifact links. Resolve exactly one artifact;
    # ambiguous products are rejected instead of choosing an arbitrary file.
    if "json" in content_type.lower() or body.lstrip().startswith(b"{"):
        try:
            metadata = json.loads(body.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            metadata = None
        if isinstance(metadata, dict):
            artifact_links = []
            product = metadata
            if "_embedded" in metadata:
                products = metadata.get("_embedded", {}).get("products", [])
                product = next((p for p in products if str(p.get("emilId", "")).upper() == PRODUCT_ID), {})
            for artifact in product.get("artifacts", []) if isinstance(product, dict) else []:
                href = artifact.get("_links", {}).get("endpoint", {}).get("href")
                if href:
                    artifact_links.append(href)
            if len(artifact_links) == 1:
                artifact_request = urllib.request.Request(artifact_links[0], headers=headers)
                with urllib.request.urlopen(artifact_request, timeout=60) as response:
                    return response.read(), response.headers.get("Content-Type", ""), response.geturl()
            if len(artifact_links) > 1:
                raise RuntimeError(f"ERCOT product metadata lists multiple artifacts: {artifact_links}.")
    return body, content_type, final_url


def _archive_and_merge(spark_session, raw_bytes: bytes, records: list[dict], retrieved_at: datetime, final_url: str):
    from pyspark.sql import functions as F
    from delta.tables import DeltaTable

    if not records:
        raise RuntimeError("ERCOT benchmark capture produced zero validated system-total rows.")
    raw_dir = f"/Volumes/{CATALOG}/bronze/raw_landing/ercot_benchmark"
    os.makedirs(raw_dir, exist_ok=True)
    file_path = f"{raw_dir}/{PRODUCT_ID.lower()}_{retrieved_at.strftime('%Y%m%dT%H%M%SZ')}_{records[0]['payload_sha256'][:12]}.bin"
    with open(file_path, "wb") as raw_file:
        raw_file.write(raw_bytes)

    incoming = spark_session.createDataFrame(records).withColumn("raw_payload_path", F.lit(file_path))
    duplicate_keys = incoming.groupBy("respondent", "vintage_id", "target_timestamp_utc").count().filter("count > 1").limit(1).count()
    if duplicate_keys:
        raise RuntimeError("ERCOT benchmark capture contains duplicate target rows within one vintage.")
    if not spark_session.catalog.tableExists(TARGET_TABLE):
        incoming.write.format("delta").mode("overwrite").saveAsTable(TARGET_TABLE)
    else:
        DeltaTable.forName(spark_session, TARGET_TABLE).alias("target").merge(
            incoming.alias("source"),
            "target.benchmark_provider = source.benchmark_provider "
            "AND target.respondent = source.respondent "
            "AND target.vintage_id = source.vintage_id "
            "AND target.target_timestamp_utc = source.target_timestamp_utc",
        ).whenNotMatchedInsertAll().execute()
    print(
        f"Captured {len(records)} validated ERCOT {PRODUCT_ID} system-total rows "
        f"from {final_url}; vintage={records[0]['vintage_id']}; archived={file_path}."
    )


if __name__ == "__main__" and "spark" in globals():
    subscription_key = _get_secret("ercot_api_subscription_key")
    username = _get_secret("ercot_api_username")
    password = _get_secret("ercot_api_password")
    if not (subscription_key and username and password):
        print(
            "ERCOT benchmark capture SKIPPED: configure Databricks secrets in scope "
            "grid_platform with keys ercot_api_subscription_key, ercot_api_username, "
            "and ercot_api_password. The hourly forecast path is independent of this task."
        )
    else:
        retrieved_at = datetime.now(timezone.utc)
        token = get_ercot_token(username, password)
        raw_bytes, content_type, artifact_url = fetch_latest_artifact(
            subscription_key, token, os.getenv("ERCOT_BENCHMARK_ENDPOINT")
        )
        rows = parse_system_forecast_records(raw_bytes, retrieved_at, content_type)
        _archive_and_merge(spark, raw_bytes, rows, retrieved_at, artifact_url)
