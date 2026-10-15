# Databricks notebook source
# MAGIC %md
# MAGIC # Run Streamlit Operational Dashboard inside Databricks Cluster
# MAGIC
# MAGIC This notebook launches the Streamlit app directly on the Databricks driver node and exposes the web interface via the Databricks Driver Proxy.

# COMMAND ----------
# MAGIC %pip install -r ../../requirements.txt --quiet

# COMMAND ----------
import subprocess
import os
import sys

# Get Workspace and Cluster identifiers
context_str = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
workspace_url = spark.conf.get("spark.databricks.workspaceUrl", "")
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId", "")
org_id = context_str.workspaceId().get()

# Construct Databricks Driver Proxy URL
proxy_url = f"https://{workspace_url}/driver-proxy/o/{org_id}/{cluster_id}/8501/"

print("=" * 80)
print(f"DATABRICKS STREAMLIT DASHBOARD PROXY URL:")
print(f"{proxy_url}")
print("=" * 80)

# COMMAND ----------
# DBTITLE 1,Start Streamlit Process on Driver Node
# Terminate any existing Streamlit instance on port 8501
subprocess.run("pkill -f 'streamlit run' || true", shell=True)

# Launch Streamlit pointing to app/main.py
process = subprocess.Popen([
    "streamlit", "run", "app/main.py",
    "--server.port", "8501",
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
    "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false"
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print(f"Streamlit process started with PID {process.pid}.")
print(f"Open this link in your browser to view the live dashboard:")
print(f"-> {proxy_url}")

# COMMAND ----------
# DBTITLE 1,Embedded View (Optional Inline Preview)
# MAGIC %md
# MAGIC If your browser allows iframes from Databricks, you can also view it directly below:

# COMMAND ----------
displayHTML(f"""
<div style="width: 100%; height: 900px; border: 1px solid #e2e8f0; border-radius: 6px; overflow: hidden;">
    <iframe src="{proxy_url}" width="100%" height="100%" style="border: none;"></iframe>
</div>
""")
