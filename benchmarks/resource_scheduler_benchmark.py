"""Run the TerraScout L5 resource-aware scheduler oracle benchmark."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrascout.eval.benchmarks import (
    ResourceSchedulerBenchmarkRow,
    run_resource_scheduler_benchmark,
)


def run(
    output: Path = Path("artifacts/resource_scheduler_benchmark.csv"),
) -> list[ResourceSchedulerBenchmarkRow]:
    return run_resource_scheduler_benchmark(output)


def main() -> None:
    rows = run()
    max_gap = max(row.optimality_gap_percent for row in rows)
    max_wall = max(row.wall_time_ms for row in rows)
    mean_inspected = sum(row.inspected_goals for row in rows) / len(rows)
    print(
        f"layouts={len(rows)} "
        f"max_gap_percent={max_gap:.3f} "
        f"max_wall_ms={max_wall:.2f} "
        f"mean_inspected={mean_inspected:.2f}"
    )


if __name__ == "__main__":
    main()
