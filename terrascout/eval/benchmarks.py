"""Reusable benchmark suites for TerraScout."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from math import hypot
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from terrascout.mapping.ekf_slam import EkfSlam
from terrascout.plan.astar import GridAStarPlanner
from terrascout.plan.hybrid_astar import HybridAStarPlanner
from terrascout.runner.mission import MissionMetrics, run_mission, write_metrics_csv
from terrascout.sim.geometry import Point2D, Pose2D
from terrascout.sim.world import OrchardWorld, ScenarioConfig


@dataclass(frozen=True)
class PlannerBenchmarkRow:
    seed: int
    planner: str
    waypoint_count: int
    path_length_m: float
    wall_time_ms: float


@dataclass(frozen=True)
class SlamBenchmarkRow:
    seed: int
    landmark_count: int
    mean_observations: float
    covariance_trace: float
    wall_time_ms: float


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


DEFAULT_BENCHMARK_SEEDS = [2, 3, 5, 7, 11]
DEFAULT_STRESS_SEEDS = [2, 7, 11]
DEFAULT_STRESS_SCENARIOS = [
    StressScenario("grid_truth", "grid", "truth"),
    StressScenario("grid_particle", "grid", "particle"),
    StressScenario("grid_slam_clear", "grid", "slam", workers=0),
    StressScenario("hybrid_slam_clear", "hybrid", "slam", workers=0),
]


def run_mission_benchmark(
    output: Path = Path("artifacts/benchmark.csv"),
    seeds: Sequence[int] | None = None,
) -> list[MissionMetrics]:
    """Run the default deterministic mission benchmark."""

    benchmark_seeds = list(seeds or DEFAULT_BENCHMARK_SEEDS)
    metrics = [run_mission(seed=seed) for seed in benchmark_seeds]
    write_metrics_csv(metrics, output)
    return metrics


def run_planner_benchmark(
    output: Path = Path("artifacts/planner_benchmark.csv"),
    seeds: Sequence[int] | None = None,
) -> list[PlannerBenchmarkRow]:
    """Compare grid A* and Hybrid A* on deterministic orchard plans."""

    rows: list[PlannerBenchmarkRow] = []
    for seed in list(seeds or DEFAULT_BENCHMARK_SEEDS):
        world = OrchardWorld(ScenarioConfig(rows=8, trees_per_row=14, worker_count=0, random_seed=seed))
        start = Pose2D(1.0, 0.8, 1.2)
        goal_point = world.row_goals[-1]
        goal_pose = Pose2D(goal_point.x, goal_point.y, 0.0)

        grid = GridAStarPlanner(world)
        started = perf_counter()
        grid_path = grid.plan(start, goal_point)
        rows.append(
            PlannerBenchmarkRow(
                seed=seed,
                planner="grid_astar",
                waypoint_count=len(grid_path),
                path_length_m=_point_length(grid_path),
                wall_time_ms=(perf_counter() - started) * 1000.0,
            )
        )

        hybrid = HybridAStarPlanner(world)
        started = perf_counter()
        hybrid_path = hybrid.plan(start, goal_pose)
        rows.append(
            PlannerBenchmarkRow(
                seed=seed,
                planner="hybrid_astar",
                waypoint_count=len(hybrid_path),
                path_length_m=_point_length(hybrid_path),
                wall_time_ms=(perf_counter() - started) * 1000.0,
            )
        )

    _write_csv(rows, output)
    return rows


def run_slam_benchmark(
    output: Path = Path("artifacts/slam_benchmark.csv"),
    seeds: Sequence[int] | None = None,
) -> list[SlamBenchmarkRow]:
    """Run a deterministic EKF-SLAM smoke benchmark."""

    rows: list[SlamBenchmarkRow] = []
    poses = [
        Pose2D(2.0, 2.0, 0.8),
        Pose2D(5.5, 7.0, 1.3),
        Pose2D(9.0, 12.0, 1.7),
        Pose2D(12.0, 18.0, 1.5),
    ]
    for seed in list(seeds or DEFAULT_BENCHMARK_SEEDS):
        world = OrchardWorld(ScenarioConfig(rows=6, trees_per_row=10, worker_count=0, random_seed=seed))
        slam = EkfSlam(poses[0])
        started = perf_counter()
        previous = poses[0]
        for pose in poses:
            slam.predict(
                linear_mps=((pose.x - previous.x) ** 2 + (pose.y - previous.y) ** 2) ** 0.5,
                angular_rps=pose.theta - previous.theta,
                dt=1.0,
            )
            slam.update(world.local_lidar_detections(pose, include_workers=False))
            previous = pose
        mean_observations = (
            sum(slam.landmark_observations) / len(slam.landmark_observations)
            if slam.landmark_observations
            else 0.0
        )
        rows.append(
            SlamBenchmarkRow(
                seed=seed,
                landmark_count=slam.landmark_count,
                mean_observations=mean_observations,
                covariance_trace=float(slam.covariance.trace()),
                wall_time_ms=(perf_counter() - started) * 1000.0,
            )
        )

    _write_csv(rows, output)
    return rows


def run_stress_benchmark(
    seeds: Sequence[int] | None = None,
    scenarios: Sequence[StressScenario] | None = None,
    detail_output: Path = Path("artifacts/stress_benchmark_detail.csv"),
    summary_output: Path = Path("artifacts/stress_benchmark_summary.csv"),
) -> list[StressSummaryRow]:
    """Run deterministic stress scenarios and write detail/summary CSVs."""

    benchmark_seeds = list(seeds or DEFAULT_STRESS_SEEDS)
    benchmark_scenarios = list(scenarios or DEFAULT_STRESS_SCENARIOS)
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[StressSummaryRow] = []

    for scenario in benchmark_scenarios:
        metrics: list[MissionMetrics] = []
        for seed in benchmark_seeds:
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
                mean_localization_error_m=sum(item.mean_localization_error_m for item in metrics)
                / len(metrics),
                mean_wall_time_s=sum(item.wall_time_s for item in metrics) / len(metrics),
                max_scheduler_drops=max(item.scheduler_dropped_goals for item in metrics),
            )
        )

    detail_output.parent.mkdir(parents=True, exist_ok=True)
    if detail_rows:
        with detail_output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    _write_csv(summary_rows, summary_output)
    return summary_rows


def _point_length(points: Sequence[Point2D | Pose2D]) -> float:
    return sum(hypot(b.x - a.x, b.y - a.y) for a, b in zip(points, points[1:]))


def _write_csv(rows: Sequence[Any], output: Path) -> None:
    if not rows:
        raise ValueError("Cannot write benchmark CSV with no rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
