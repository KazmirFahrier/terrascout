"""Run a small deterministic benchmark suite for TerraScout."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terrascout.eval.benchmarks import run_mission_benchmark


def main() -> None:
    metrics = run_mission_benchmark()
    mean_success = sum(item.success_rate for item in metrics) / len(metrics)
    mean_wall = sum(item.wall_time_s for item in metrics) / len(metrics)
    print(f"runs={len(metrics)} mean_success={mean_success:.3f} mean_wall_time_s={mean_wall:.3f}")


if __name__ == "__main__":
    main()
