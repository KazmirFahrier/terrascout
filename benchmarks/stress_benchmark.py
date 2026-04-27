"""Stress-test TerraScout mission modes across planners and pose sources."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrascout.runner.mission import MissionMetrics, run_mission


@dataclass(frozen=True)
class StressScenario:
    name: str
    planner: str
    pose_source: str
    workers: int = 1


@dataclass(frozen=True)
class StressSummaryRow:
    scenario: str
    planner: str
    pose_source: str
    runs: int
    mean_success_rate: float
    total_collisions: int
    mean_localization_error_m: float
    mean_wall_time_s: float
    max_scheduler_drops: int


DEFAULT_SCENARIOS = [
    StressScenario("grid_truth", "grid", "truth"),
    StressScenario("grid_particle", "grid", "particle"),
    StressScenario("grid_slam_clear", "grid", "slam", workers=0),
    StressScenario("hybrid_slam_clear", "hybrid", "slam", workers=0),
]


def run(
    seeds: list[int] | None = None,
    scenarios: list[StressScenario] | None = None,
    detail_output: Path = Path("artifacts/stress_benchmark_detail.csv"),
    summary_output: Path = Path("artifacts/stress_benchmark_summary.csv"),
) -> list[StressSummaryRow]:
    """Run deterministic stress scenarios and write detail/summary CSVs."""

    seeds = seeds or [2, 7, 11]
    scenarios = scenarios or DEFAULT_SCENARIOS
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[StressSummaryRow] = []

    for scenario in scenarios:
        metrics: list[MissionMetrics] = []
        for seed in seeds:
            result = run_mission(
                seed=seed,
                worker_count=scenario.workers,
                planner_kind=scenario.planner,
                pose_source=scenario.pose_source,
            )
            metrics.append(result)
            detail = asdict(result)
            detail["scenario"] = scenario.name
            detail_rows.append(detail)

        summary_rows.append(
            StressSummaryRow(
                scenario=scenario.name,
                planner=scenario.planner,
                pose_source=scenario.pose_source,
                runs=len(metrics),
                mean_success_rate=sum(item.success_rate for item in metrics) / len(metrics),
                total_collisions=sum(item.collisions for item in metrics),
                mean_localization_error_m=sum(
                    item.mean_localization_error_m for item in metrics
                )
                / len(metrics),
                mean_wall_time_s=sum(item.wall_time_s for item in metrics) / len(metrics),
                max_scheduler_drops=max(item.scheduler_dropped_goals for item in metrics),
            )
        )

    detail_output.parent.mkdir(parents=True, exist_ok=True)
    with detail_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    with summary_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(summary_rows[0]).keys()))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(asdict(row))

    return summary_rows


def main() -> None:
    rows = run()
    for row in rows:
        print(
            f"{row.scenario}: success={row.mean_success_rate:.3f} "
            f"collisions={row.total_collisions} wall={row.mean_wall_time_s:.2f}s"
        )


if __name__ == "__main__":
    main()
