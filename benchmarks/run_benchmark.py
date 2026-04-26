"""Run a small deterministic benchmark suite for TerraScout."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terrascout.runner.mission import run_mission, write_metrics_csv


def main() -> None:
    seeds = [2, 3, 5, 7, 11]
    metrics = [run_mission(seed=seed) for seed in seeds]
    write_metrics_csv(metrics, Path("artifacts/benchmark.csv"))
    mean_success = sum(item.success_rate for item in metrics) / len(metrics)
    mean_wall = sum(item.wall_time_s for item in metrics) / len(metrics)
    print(f"runs={len(metrics)} mean_success={mean_success:.3f} mean_wall_time_s={mean_wall:.3f}")


if __name__ == "__main__":
    main()
