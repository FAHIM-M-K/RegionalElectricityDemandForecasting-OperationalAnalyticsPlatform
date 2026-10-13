"""
configs/loader.py
-----------------
Shared configuration loader for the ERCOT Grid Platform.

Locates and loads a JSON config file by walking up from the current working
directory (works for both local dev and Databricks Repos), then falls back to
the Databricks workspace path.

Usage in any notebook:
    import sys, os
    _root = os.getcwd()
    for _ in range(5):
        if os.path.isdir(os.path.join(_root, "configs")):
            if _root not in sys.path:
                sys.path.insert(0, _root)
            break
        _p = os.path.dirname(_root)
        if _p == _root:
            break
        _root = _p
    from configs.loader import load_platform_config
"""

import os
import json


def load_platform_config(filename="ercot_config.json"):
    """
    Locates and loads a JSON configuration file.

    Search order:
      1. Walk up from os.getcwd() looking for a 'configs/<filename>' path.
      2. Fall back to the Databricks Repos workspace root path.

    Raises FileNotFoundError if the config cannot be located.
    """
    curr = os.getcwd()
    for _ in range(5):
        test_path = os.path.join(curr, "configs", filename)
        if os.path.exists(test_path):
            with open(test_path, "r", encoding="utf-8") as f:
                print(f"Loaded platform config from {test_path}")
                return json.load(f)
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent

    repo_candidate = f"/Workspace/Repos/configs/{filename}"
    if os.path.exists(repo_candidate):
        with open(repo_candidate, "r", encoding="utf-8") as f:
            print(f"Loaded platform config from {repo_candidate}")
            return json.load(f)

    raise FileNotFoundError(
        f"Critical Configuration Error: Could not locate 'configs/{filename}'. "
        "Ensure the repository is synced properly in Databricks Repos."
    )
