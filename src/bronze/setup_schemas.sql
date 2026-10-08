-- Databricks notebook source
-- MAGIC %md
-- MAGIC # 00. Unity Catalog & Medallion Schemas Setup
-- MAGIC
-- MAGIC **Order in Pipeline:** Step 1 (Prerequisite for all layers)
-- MAGIC **Purpose:** Ensure catalog `main`, medallion schemas (`bronze`, `silver`, `gold`), and staging volume exist.

-- COMMAND ----------
-- 1. Use / Create Catalog
CREATE CATALOG IF NOT EXISTS main;
USE CATALOG main;

-- COMMAND ----------
-- 2. Create Medallion Schemas
CREATE SCHEMA IF NOT EXISTS bronze
COMMENT 'Raw landing tables and unmodified API payloads';

CREATE SCHEMA IF NOT EXISTS silver
COMMENT 'Cleaned, validated, deduplicated, and timezone-aligned data';

CREATE SCHEMA IF NOT EXISTS gold
COMMENT 'Business-ready features, dimensions, metrics, and forecasts';

-- COMMAND ----------
-- 3. Create Raw Landing Volume for unstructured/raw backups
CREATE VOLUME IF NOT EXISTS bronze.raw_landing
COMMENT 'Volume storage for raw API JSON payloads';

-- COMMAND ----------
-- 4. Verification
SHOW SCHEMAS IN main;
