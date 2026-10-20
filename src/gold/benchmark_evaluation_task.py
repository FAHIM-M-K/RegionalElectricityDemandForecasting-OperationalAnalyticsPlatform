# Databricks notebook source
"""Notebook entry point for the benchmark evaluation module."""

import os
import runpy
import sys

_repo_root = os.getcwd()
for _ in range(5):
    if os.path.isdir(os.path.join(_repo_root, "configs")):
        break
    _parent = os.path.dirname(_repo_root)
    if _parent == _repo_root:
        raise RuntimeError("Could not locate the project root containing configs/.")
    _repo_root = _parent

if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_benchmark_module = runpy.run_path(
    os.path.join(_repo_root, "src", "gold", "benchmark_evaluation.py"),
    run_name="benchmark_evaluation_module",
)
_benchmark_module["run_benchmark_evaluation"](spark)
