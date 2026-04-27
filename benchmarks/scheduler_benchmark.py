"""Run the TerraScout L5 scheduler oracle benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrascout.eval.benchmarks import SchedulerBenchmarkRow, run_scheduler_benchmark


def run(output: Path = Path("artifacts/scheduler_benchmark.csv")) -> list[SchedulerBenchmarkRow]:
    return run_scheduler_benchmark(output)


def main() -> None:
    rows = run()
    max_gap = max(row.optimality_gap_percent for row in rows)
    max_wall = max(row.wall_time_ms for row in rows)
    max_iterations = max(row.iterations for row in rows)
    print(f"max_gap_percent={max_gap:.3f} max_wall_ms={max_wall:.2f} max_iterations={max_iterations}")


if __name__ == "__main__":
    main()
