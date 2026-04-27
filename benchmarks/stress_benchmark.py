"""Stress-test TerraScout mission modes across planners and pose sources."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrascout.eval.benchmarks import StressScenario, StressSummaryRow, run_stress_benchmark


def run(
    seeds: list[int] | None = None,
    scenarios: list[StressScenario] | None = None,
    detail_output: Path = Path("artifacts/stress_benchmark_detail.csv"),
    summary_output: Path = Path("artifacts/stress_benchmark_summary.csv"),
) -> list[StressSummaryRow]:
    return run_stress_benchmark(seeds, scenarios, detail_output, summary_output)


def main() -> None:
    rows = run()
    for row in rows:
        print(
            f"{row.scenario}: success={row.mean_success_rate:.3f} "
            f"collisions={row.total_collisions} wall={row.mean_wall_time_s:.2f}s"
        )


if __name__ == "__main__":
    main()
