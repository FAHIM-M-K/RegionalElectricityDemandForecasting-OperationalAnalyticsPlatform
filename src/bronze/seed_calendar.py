# Databricks notebook source
# MAGIC %md
# MAGIC # 03. Calendar & Holiday Reference Table (Observed Federal Holidays)
# MAGIC
# MAGIC **Target:** `bronze.calendar_reference`  
# MAGIC **Purpose:** Provide an authoritative reference table of calendar dates (2021–2028), day-of-week, weekend flags, and recognized US federal holidays—including statutory observed rollover dates (e.g. Christmas on Sunday observed on Monday)—to support downstream feature engineering in Silver & Gold.

# COMMAND ----------
# DBTITLE 1,Widgets & Context
dbutils.widgets.text("catalog", "main", "Catalog Name")

CATALOG = dbutils.widgets.get("catalog").strip()
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA bronze")

# COMMAND ----------
# DBTITLE 1,Build Full Calendar with Observed Federal Holidays
import datetime
from pyspark.sql import functions as F

years = list(range(2021, 2029))  # Covers through 2028 for forward scoring
calendar_records = []

def get_federal_holidays(yr: int):
    """
    Returns dict of date_str -> holiday_name for year yr,
    incorporating federal holiday observance rules (Sunday -> Monday, Saturday -> Friday).
    """
    h_map = {}

    # 1. Fixed-date holidays with weekend observation rollover
    fixed_holidays = [
        (1, 1, "New Year's Day"),
        (6, 19, "Juneteenth National Independence Day"),
        (7, 4, "Independence Day"),
        (11, 11, "Veterans Day"),
        (12, 25, "Christmas Day")
    ]

    for m, day, name in fixed_holidays:
        actual_date = datetime.date(yr, m, day)
        # Mark actual date
        h_map[actual_date.strftime("%Y-%m-%d")] = name

        # Federal observation rule:
        # If Saturday -> observed on Friday
        # If Sunday -> observed on Monday
        if actual_date.weekday() == 5:  # Saturday
            obs_date = actual_date - datetime.timedelta(days=1)
            h_map[obs_date.strftime("%Y-%m-%d")] = f"{name} (Observed)"
        elif actual_date.weekday() == 6:  # Sunday
            obs_date = actual_date + datetime.timedelta(days=1)
            h_map[obs_date.strftime("%Y-%m-%d")] = f"{name} (Observed)"

    # 2. Floating Monday/Thursday holidays
    # MLK: 3rd Monday in Jan
    jan1 = datetime.date(yr, 1, 1)
    mlk_day = 1 + (0 - jan1.weekday() + 7) % 7 + 14
    h_map[f"{yr}-01-{mlk_day:02d}"] = "Martin Luther King Jr. Day"

    # Presidents' Day: 3rd Monday in Feb
    feb1 = datetime.date(yr, 2, 1)
    pres_day = 1 + (0 - feb1.weekday() + 7) % 7 + 14
    h_map[f"{yr}-02-{pres_day:02d}"] = "Presidents' Day"

    # Memorial Day: Last Monday in May
    may31 = datetime.date(yr, 5, 31)
    mem_day = 31 - (may31.weekday() - 0 + 7) % 7
    h_map[f"{yr}-05-{mem_day:02d}"] = "Memorial Day"

    # Labor Day: 1st Monday in Sept
    sep1 = datetime.date(yr, 9, 1)
    labor_day = 1 + (0 - sep1.weekday() + 7) % 7
    h_map[f"{yr}-09-{labor_day:02d}"] = "Labor Day"

    # Columbus Day / Indigenous Peoples' Day: 2nd Monday in Oct
    oct1 = datetime.date(yr, 10, 1)
    col_day = 1 + (0 - oct1.weekday() + 7) % 7 + 7
    h_map[f"{yr}-10-{col_day:02d}"] = "Columbus Day"

    # Thanksgiving Day: 4th Thursday in Nov
    nov1 = datetime.date(yr, 11, 1)
    thx_day = 1 + (3 - nov1.weekday() + 7) % 7 + 21
    h_map[f"{yr}-11-{thx_day:02d}"] = "Thanksgiving Day"
    # Day after Thanksgiving (Black Friday) - industrial suppression
    h_map[f"{yr}-11-{(thx_day + 1):02d}"] = "Day After Thanksgiving"

    return h_map

# Generate all days across the evaluation horizon
all_holidays = {}
for y in years:
    all_holidays.update(get_federal_holidays(y))

# Also capture New Year's Day observed if Dec 31 is Friday
for y in years:
    next_ny = datetime.date(y + 1, 1, 1)
    if next_ny.weekday() == 5:  # Jan 1 is Saturday -> Dec 31 is observed holiday
        all_holidays[f"{y}-12-31"] = "New Year's Day (Observed)"

start_date = datetime.date(2021, 1, 1)
end_date = datetime.date(2028, 12, 31)
curr = start_date

while curr <= end_date:
    d_str = curr.strftime("%Y-%m-%d")
    is_weekend = curr.weekday() >= 5
    holiday_name = all_holidays.get(d_str, None)
    
    calendar_records.append({
        "date": d_str,
        "year": curr.year,
        "month": curr.month,
        "day": curr.day,
        "day_of_week": curr.weekday(),       # 0 = Monday, 6 = Sunday (Python/Pandas standard)
        "day_of_week_spark": (curr.weekday() + 1) % 7 + 1, # 1 = Sunday, 7 = Saturday (PySpark standard)
        "is_weekend": is_weekend,
        "is_holiday": holiday_name is not None,
        "holiday_name": holiday_name
    })
    curr += datetime.timedelta(days=1)

cal_df = spark.createDataFrame(calendar_records)

(
    cal_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.bronze.calendar_reference")
)

print(f"Successfully populated {cal_df.count()} days into {CATALOG}.bronze.calendar_reference (2021-2028) with observed holiday rollover rules.")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT year, count(*) as days, sum(cast(is_holiday as int)) as holiday_count
# MAGIC FROM bronze.calendar_reference
# MAGIC GROUP BY year
# MAGIC ORDER BY year;
