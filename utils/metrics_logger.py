"""
utils/metrics_logger.py
Shared logging helper for MediTwin AI.
Reused by every chapter to log model metrics to console and to a JSON file.
"""

import json
import os
from datetime import datetime


def log_metrics(component_name: str, metrics_dict: dict) -> None:
    """
    Log metrics for a given component.

    - Prints component name and all key-value pairs to the console.
    - Appends the same metrics (with a timestamp) as a JSON entry to
      logs/<component_name>_metrics.json, creating the file if needed.

    Args:
        component_name: Short identifier for the model/agent (e.g. "risk_prediction").
        metrics_dict:   Dictionary of metric names to values.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    # ── Console output ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Component : {component_name}")
    print(f"  Timestamp : {timestamp}")
    print(f"  Metrics")
    for key, value in metrics_dict.items():
        print(f"    {key}: {value}")
    print(f"{'='*60}\n")

    # ── JSON log file ─────────────────────────────────────────────────────────
    # Resolve the logs/ directory relative to this file's project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_path = os.path.join(logs_dir, f"{component_name}_metrics.json")

    entry = {"timestamp": timestamp, "component": component_name, **metrics_dict}

    # Append to existing list or create a new one
    if os.path.exists(log_path):
        with open(log_path, "r") as fh:
            try:
                existing = json.load(fh)
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    existing.append(entry)

    with open(log_path, "w") as fh:
        json.dump(existing, fh, indent=2)

    print(f"[metrics_logger] Saved → {log_path}")
