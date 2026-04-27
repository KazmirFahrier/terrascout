"""Compare grid A* and Hybrid A* on deterministic orchard plans."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrascout.eval.benchmarks import PlannerBenchmarkRow, run_planner_benchmark


def run(output: Path = Path("artifacts/planner_benchmark.csv")) -> list[PlannerBenchmarkRow]:
    return run_planner_benchmark(output)


def main() -> None:
    rows = run()
    grid_mean = sum(row.wall_time_ms for row in rows if row.planner == "grid_astar") / 5.0
    hybrid_mean = sum(row.wall_time_ms for row in rows if row.planner == "hybrid_astar") / 5.0
    print(f"grid_mean_ms={grid_mean:.2f} hybrid_mean_ms={hybrid_mean:.2f}")


if __name__ == "__main__":
    main()
