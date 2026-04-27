"""Scenario file loading for reproducible TerraScout runs."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from terrascout.sim.world import ScenarioConfig


def load_scenario_config(path: Path) -> ScenarioConfig:
    """Load a ``ScenarioConfig`` from a JSON file."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Scenario file must contain a JSON object")
    valid_fields = {field.name for field in fields(ScenarioConfig)}
    unknown = set(payload) - valid_fields
    if unknown:
        raise ValueError(f"Unknown scenario fields: {sorted(unknown)}")
    values = {key: _coerce_value(key, value) for key, value in payload.items()}
    return ScenarioConfig(
        rows=int(values.get("rows", ScenarioConfig.rows)),
        trees_per_row=int(values.get("trees_per_row", ScenarioConfig.trees_per_row)),
        row_spacing_m=float(values.get("row_spacing_m", ScenarioConfig.row_spacing_m)),
        tree_spacing_m=float(values.get("tree_spacing_m", ScenarioConfig.tree_spacing_m)),
        worker_count=int(values.get("worker_count", ScenarioConfig.worker_count)),
        width_margin_m=float(values.get("width_margin_m", ScenarioConfig.width_margin_m)),
        lidar_range_m=float(values.get("lidar_range_m", ScenarioConfig.lidar_range_m)),
        random_seed=int(values.get("random_seed", ScenarioConfig.random_seed)),
    )


def save_scenario_config(config: ScenarioConfig, path: Path) -> None:
    """Save a ``ScenarioConfig`` as formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {field.name: getattr(config, field.name) for field in fields(ScenarioConfig)}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _coerce_value(key: str, value: Any) -> int | float:
    integer_fields = {"rows", "trees_per_row", "worker_count", "random_seed"}
    if key in integer_fields:
        return int(value)
    return float(value)
