# Databricks notebook source
# MAGIC %md
# MAGIC # 03. Calendar & Holiday Reference Table
# MAGIC
# MAGIC **Target:** `bronze.calendar_reference`  
# MAGIC **Purpose:** Provide a reference table of calendar dates (2021–2026), day-of-week, weekend flags, and recognized US federal holidays to support downstream feature engineering in Silver & Gold.

# COMMAND ----------
# DBTITLE 1,Widgets & Context
dbutils.widgets.text("catalog", "main", "Catalog Name")

CATALOG = dbutils.widgets.get("catalog").strip()
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql("USE SCHEMA bronze")

# COMMAND ----------
import datetime
from pyspark.sql import functions as F

years = list(range(2021, 2027))
calendar_records = []

for yr in years:
    holidays = {
        f"{yr}-01-01": "New Year's Day",
        f"{yr}-06-19": "Juneteenth",
        f"{yr}-07-04": "Independence Day",
        f"{yr}-11-11": "Veterans Day",
        f"{yr}-12-25": "Christmas Day"
    }
    
    d = datetime.date(yr, 1, 1)
    end_d = datetime.date(yr, 12, 31)
    while d <= end_d:
        d_str = d.strftime("%Y-%m-%d")
        is_weekend = d.weekday() >= 5
        holiday_name = holidays.get(d_str, None)
        
        # Floating holidays
        if d.month == 1 and d.weekday() == 0 and 15 <= d.day <= 21:
            holiday_name = "Martin Luther King Jr. Day"
        elif d.month == 2 and d.weekday() == 0 and 15 <= d.day <= 21:
            holiday_name = "Presidents' Day"
        elif d.month == 5 and d.weekday() == 0 and d.day >= 25:
            holiday_name = "Memorial Day"
        elif d.month == 9 and d.weekday() == 0 and d.day <= 7:
            holiday_name = "Labor Day"
        elif d.month == 11 and d.weekday() == 3 and 22 <= d.day <= 28:
            holiday_name = "Thanksgiving Day"
            
        calendar_records.append({
            "date": d_str,
            "year": yr,
            "month": d.month,
            "day": d.day,
            "day_of_week": d.weekday(), # 0=Monday, 6=Sunday
            "is_weekend": is_weekend,
            "is_holiday": holiday_name is not None,
            "holiday_name": holiday_name
        })
        d += datetime.timedelta(days=1)

cal_df = spark.createDataFrame(calendar_records)
cal_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(f"{CATALOG}.bronze.calendar_reference")

print(f"Successfully populated {cal_df.count()} days into {CATALOG}.bronze.calendar_reference")

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT year, count(*) as days, sum(cast(is_holiday as int)) as holiday_count
# MAGIC FROM bronze.calendar_reference
# MAGIC GROUP BY year
# MAGIC ORDER BY year;
