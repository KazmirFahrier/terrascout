"""Reusable benchmark suites for TerraScout."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from itertools import permutations
from math import atan2, cos, hypot, pi, sin
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from terrascout.control.pid import DriveController
from terrascout.localize.particle import ParticleLocalizer
from terrascout.mapping.ekf_slam import EkfSlam
from terrascout.plan.astar import GridAStarPlanner
from terrascout.plan.hybrid_astar import HybridAStarPlanner
from terrascout.runner.mission import MissionMetrics, run_mission, write_metrics_csv
from terrascout.scheduler.value_iteration import InspectionScheduler
from terrascout.sim.geometry import Point2D, Pose2D, distance, wrap_angle
from terrascout.sim.rover import DifferentialDriveRover
from terrascout.sim.world import LidarDetection, OrchardWorld, ScenarioConfig
from terrascout.tracking.kalman import MultiObjectTracker


@dataclass(frozen=True)
class PlannerBenchmarkRow:
    seed: int
    planner: str
    waypoint_count: int
    path_length_m: float
    steering_effort_rad: float
    wall_time_ms: float


@dataclass(frozen=True)
class ControlBenchmarkRow:
    seed: int
    slip_fraction: float
    max_cross_track_error_m: float
    heading_settle_time_s: float
    heading_overshoot_fraction: float
    wall_time_ms: float


@dataclass(frozen=True)
class SlamBenchmarkRow:
    seed: int
    landmark_count: int
    mean_observations: float
    final_pose_error_m: float
    mean_landmark_error_m: float
    p95_landmark_error_m: float
    covariance_trace: float
    wall_time_ms: float


@dataclass(frozen=True)
class TrackingBenchmarkRow:
    seed: int
    worker_count: int
    track_count: int
    mean_prediction_error_m: float
    association_accuracy: float
    wall_time_ms: float


@dataclass(frozen=True)
class LocalizationBenchmarkRow:
    seed: int
    prior_position_error_m: float
    prior_heading_error_deg: float
    final_pose_error_m: float
    particle_count: int
    wall_time_ms: float


@dataclass(frozen=True)
class SchedulerBenchmarkRow:
    seed: int
    goal_count: int
    scheduler_value: float
    oracle_value: float
    optimality_gap_percent: float
    iterations: int
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


def run_control_benchmark(
    output: Path = Path("artifacts/control_benchmark.csv"),
    seeds: Sequence[int] | None = None,
) -> list[ControlBenchmarkRow]:
    """Evaluate L0 straight-line tracking and 90-degree heading-step response."""

    rows: list[ControlBenchmarkRow] = []
    for seed in list(seeds or range(10)):
        started = perf_counter()
        rng = np.random.default_rng(seed)
        slip_fraction = float(rng.uniform(0.0, 0.08))
        max_cross_track = _straight_line_cross_track_error(slip_fraction)
        settle_time, overshoot_fraction = _heading_step_response(slip_fraction)
        rows.append(
            ControlBenchmarkRow(
                seed=seed,
                slip_fraction=slip_fraction,
                max_cross_track_error_m=max_cross_track,
                heading_settle_time_s=settle_time,
                heading_overshoot_fraction=overshoot_fraction,
                wall_time_ms=(perf_counter() - started) * 1000.0,
            )
        )

    _write_csv(rows, output)
    return rows


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
                steering_effort_rad=_point_steering_effort(grid_path, start.theta),
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
                steering_effort_rad=_pose_steering_effort(hybrid_path),
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
    commands = [
        (0.9, 0.0, 8),
        (0.8, 0.18, 8),
        (0.9, 0.0, 8),
        (0.8, -0.18, 8),
        (0.9, 0.0, 8),
    ]
    for seed in list(seeds or DEFAULT_BENCHMARK_SEEDS):
        world = OrchardWorld(ScenarioConfig(rows=8, trees_per_row=14, worker_count=0, random_seed=seed))
        truth = Pose2D(2.0, 1.5, 1.1)
        slam = EkfSlam(truth)
        started = perf_counter()
        for linear_mps, angular_rps, steps in commands:
            for _ in range(steps):
                truth = _slam_truth_step(truth, linear_mps, angular_rps, dt=0.5)
                slam.predict(linear_mps=linear_mps, angular_rps=angular_rps, dt=0.5)
                slam.update(world.local_lidar_detections(truth, include_workers=False))
        mean_observations = (
            sum(slam.landmark_observations) / len(slam.landmark_observations)
            if slam.landmark_observations
            else 0.0
        )
        landmark_errors = _landmark_errors(slam.landmarks(), world.trees)
        rows.append(
            SlamBenchmarkRow(
                seed=seed,
                landmark_count=slam.landmark_count,
                mean_observations=mean_observations,
                final_pose_error_m=distance(slam.pose, truth),
                mean_landmark_error_m=_mean(landmark_errors),
                p95_landmark_error_m=percentile(landmark_errors, 95),
                covariance_trace=float(slam.covariance.trace()),
                wall_time_ms=(perf_counter() - started) * 1000.0,
            )
        )

    _write_csv(rows, output)
    return rows


def run_tracking_benchmark(
    output: Path = Path("artifacts/tracking_benchmark.csv"),
    seeds: Sequence[int] | None = None,
    worker_count: int = 10,
    steps: int = 80,
    dt: float = 0.1,
) -> list[TrackingBenchmarkRow]:
    """Evaluate 1-second worker prediction and ID continuity for up to 10 agents."""

    rows: list[TrackingBenchmarkRow] = []
    for seed in list(seeds or DEFAULT_BENCHMARK_SEEDS):
        started = perf_counter()
        rng = np.random.default_rng(seed)
        positions = _initial_worker_positions(rng, worker_count)
        velocities = np.column_stack(
            [
                rng.uniform(-0.25, 0.25, worker_count),
                rng.uniform(-0.20, 0.20, worker_count),
            ]
        )
        tracker = MultiObjectTracker(gate_m=1.0, max_missed=5)
        truth_to_track: dict[int, int] = {}
        correct_associations = 0
        total_associations = 0
        prediction_errors: list[float] = []

        for step in range(steps):
            detections = _worker_detections(rng, positions)
            tracker.update(detections, dt)
            assignments = _assign_tracks_to_truths(tracker, positions, max_distance_m=0.75)
            if step >= 10:
                future_positions = positions + velocities * 1.0
                predictions = {track_id: np.array([x, y]) for track_id, x, y in tracker.predicted_positions(1.0)}
                for truth_id, track_id in assignments.items():
                    if truth_id in truth_to_track:
                        total_associations += 1
                        correct_associations += int(truth_to_track[truth_id] == track_id)
                    else:
                        truth_to_track[truth_id] = track_id
                    if track_id in predictions:
                        prediction_errors.append(
                            float(np.linalg.norm(predictions[track_id] - future_positions[truth_id]))
                        )
            positions = _advance_workers(positions, velocities, dt)

        rows.append(
            TrackingBenchmarkRow(
                seed=seed,
                worker_count=worker_count,
                track_count=len(tracker.tracks),
                mean_prediction_error_m=_mean(prediction_errors),
                association_accuracy=(
                    correct_associations / total_associations if total_associations else 0.0
                ),
                wall_time_ms=(perf_counter() - started) * 1000.0,
            )
        )

    _write_csv(rows, output)
    return rows


def run_localization_benchmark(
    output: Path = Path("artifacts/localization_benchmark.csv"),
    seeds: Sequence[int] | None = None,
) -> list[LocalizationBenchmarkRow]:
    """Evaluate coarse-prior particle-filter pose refinement."""

    rows: list[LocalizationBenchmarkRow] = []
    for seed in list(seeds or DEFAULT_BENCHMARK_SEEDS):
        started = perf_counter()
        rng = np.random.default_rng(seed)
        world = OrchardWorld(ScenarioConfig(rows=4, trees_per_row=6, worker_count=0, random_seed=seed))
        truth = Pose2D(
            x=float(rng.uniform(3.0, 8.0)),
            y=float(rng.uniform(3.0, 10.0)),
            theta=float(rng.uniform(-1.0, 1.0)),
        )
        dx = float(rng.uniform(-0.5, 0.5))
        dy = float(rng.uniform(-0.5, 0.5))
        dtheta = float(rng.uniform(-pi / 18.0, pi / 18.0))
        prior = Pose2D(truth.x + dx, truth.y + dy, wrap_angle(truth.theta + dtheta))
        localizer = ParticleLocalizer.gaussian(
            1500,
            mean=prior,
            std=(0.8, 0.8, pi / 9.0),
            seed=seed + 100,
        )
        for _ in range(5):
            detections = world.local_lidar_detections(truth, include_workers=False)
            localizer.update(detections, world.trees)

        rows.append(
            LocalizationBenchmarkRow(
                seed=seed,
                prior_position_error_m=distance(prior, truth),
                prior_heading_error_deg=abs(wrap_angle(prior.theta - truth.theta)) * 180.0 / pi,
                final_pose_error_m=distance(localizer.estimate(), truth),
                particle_count=len(localizer.particles),
                wall_time_ms=(perf_counter() - started) * 1000.0,
            )
        )

    _write_csv(rows, output)
    return rows


def run_scheduler_benchmark(
    output: Path = Path("artifacts/scheduler_benchmark.csv"),
    seeds: Sequence[int] | None = None,
    goal_count: int = 7,
) -> list[SchedulerBenchmarkRow]:
    """Compare the MDP scheduler policy with a brute-force permutation oracle."""

    rows: list[SchedulerBenchmarkRow] = []
    start = Pose2D(0.0, 0.0, 0.0)
    for seed in list(seeds or DEFAULT_BENCHMARK_SEEDS):
        rng = np.random.default_rng(seed)
        goals = _random_scheduler_goals(rng, goal_count)
        priorities = [float(rng.uniform(0.7, 1.8)) for _ in goals]
        scheduler = InspectionScheduler()
        started = perf_counter()
        order = scheduler.plan_order(start, goals, priorities)
        scheduler_value = _schedule_order_value(start, goals, priorities, order)
        oracle_value = _scheduler_oracle_value(start, goals, priorities)
        wall_time_ms = (perf_counter() - started) * 1000.0
        gap = max(0.0, oracle_value - scheduler_value) / max(1.0, abs(oracle_value)) * 100.0
        rows.append(
            SchedulerBenchmarkRow(
                seed=seed,
                goal_count=goal_count,
                scheduler_value=scheduler_value,
                oracle_value=oracle_value,
                optimality_gap_percent=gap,
                iterations=scheduler.iterations,
                wall_time_ms=wall_time_ms,
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


def _point_steering_effort(points: Sequence[Point2D], start_theta: float) -> float:
    effort = 0.0
    previous_heading = start_theta
    previous = points[0] if points else Point2D(0.0, 0.0)
    for point in points:
        dx = point.x - previous.x
        dy = point.y - previous.y
        if dx * dx + dy * dy > 1e-9:
            heading = atan2(dy, dx)
            effort += abs(wrap_angle(heading - previous_heading))
            previous_heading = heading
        previous = point
    return effort


def _pose_steering_effort(poses: Sequence[Pose2D]) -> float:
    return sum(abs(wrap_angle(b.theta - a.theta)) for a, b in zip(poses, poses[1:]))


def _slam_truth_step(pose: Pose2D, linear_mps: float, angular_rps: float, dt: float) -> Pose2D:
    theta_mid = pose.theta + 0.5 * angular_rps * dt
    return Pose2D(
        x=pose.x + linear_mps * cos(theta_mid) * dt,
        y=pose.y + linear_mps * sin(theta_mid) * dt,
        theta=wrap_angle(pose.theta + angular_rps * dt),
    )


def _landmark_errors(estimated: Sequence[Point2D], truth: Sequence[Point2D]) -> list[float]:
    return [
        min(hypot(landmark.x - tree.x, landmark.y - tree.y) for tree in truth)
        for landmark in estimated
    ]


def _straight_line_cross_track_error(slip_fraction: float) -> float:
    controller = DriveController.default()
    rover = DifferentialDriveRover(
        pose=Pose2D(0.0, 0.02, 0.02),
        slip_fraction=slip_fraction,
    )
    goal = Point2D(8.0, 0.0)
    dt = 0.05
    max_cross_track = abs(rover.pose.y)
    for _ in range(180):
        left, right = controller.wheel_commands(rover.pose, goal, dt)
        rover.command(left, right)
        rover.step(dt)
        max_cross_track = max(max_cross_track, abs(rover.pose.y))
    return max_cross_track


def _heading_step_response(slip_fraction: float) -> tuple[float, float]:
    controller = DriveController.default()
    rover = DifferentialDriveRover(
        pose=Pose2D(0.0, 0.0, 0.0),
        slip_fraction=slip_fraction,
    )
    goal = Point2D(0.0, 5.0)
    dt = 0.05
    target_heading = pi / 2.0
    max_heading = rover.pose.theta
    settle_time = 5.0
    for step in range(100):
        left, right = controller.wheel_commands(rover.pose, goal, dt)
        rover.command(left, right)
        rover.step(dt)
        max_heading = max(max_heading, rover.pose.theta)
        if settle_time >= 5.0 and abs(wrap_angle(target_heading - rover.pose.theta)) < 0.035:
            settle_time = step * dt
    overshoot_fraction = max(0.0, max_heading - target_heading) / target_heading
    return settle_time, overshoot_fraction


def _initial_worker_positions(rng: np.random.Generator, worker_count: int) -> NDArray[np.float64]:
    x_positions = np.linspace(1.0, 19.0, worker_count)
    y_positions = np.linspace(2.0, 28.0, worker_count)
    return np.column_stack([x_positions, y_positions]) + rng.normal(0.0, 0.05, (worker_count, 2))


def _worker_detections(
    rng: np.random.Generator,
    positions: NDArray[np.float64],
) -> list[LidarDetection]:
    detections: list[LidarDetection] = []
    order = list(range(len(positions)))
    rng.shuffle(order)
    for idx in order:
        noise = rng.normal(0.0, 0.03, 2)
        detections.append(
            LidarDetection(
                x=float(positions[idx, 0] + noise[0]),
                y=float(positions[idx, 1] + noise[1]),
                kind="worker",
            )
        )
    return detections


def _assign_tracks_to_truths(
    tracker: MultiObjectTracker,
    positions: NDArray[np.float64],
    max_distance_m: float,
) -> dict[int, int]:
    assignments: dict[int, int] = {}
    used_tracks: set[int] = set()
    for truth_id in range(len(positions)):
        best: tuple[float, int] | None = None
        truth_xy = positions[truth_id]
        for track in tracker.tracks:
            if track.track_id in used_tracks:
                continue
            distance_m = float(np.linalg.norm(track.xy - truth_xy))
            if distance_m <= max_distance_m and (best is None or distance_m < best[0]):
                best = (distance_m, track.track_id)
        if best is not None:
            assignments[truth_id] = best[1]
            used_tracks.add(best[1])
    return assignments


def _advance_workers(
    positions: NDArray[np.float64],
    velocities: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    return positions + velocities * dt


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _random_scheduler_goals(rng: np.random.Generator, goal_count: int) -> list[Point2D]:
    return [
        Point2D(
            x=float(rng.uniform(2.0, 20.0)),
            y=float(rng.uniform(0.0, 18.0)),
        )
        for _ in range(goal_count)
    ]


def _schedule_order_value(
    start: Pose2D,
    goals: list[Point2D],
    priorities: list[float],
    order: Sequence[int],
) -> float:
    scheduler = InspectionScheduler()
    cfg = scheduler.config
    value = 0.0
    position = len(goals)
    for depth, action in enumerate(order):
        origin_x = start.x if position == len(goals) else goals[position].x
        origin_y = start.y if position == len(goals) else goals[position].y
        travel_m = hypot(goals[action].x - origin_x, goals[action].y - origin_y)
        reward = cfg.inspection_reward * priorities[action] - cfg.travel_cost_per_m * travel_m
        value += (cfg.discount**depth) * reward
        position = action
    return value


def _scheduler_oracle_value(
    start: Pose2D,
    goals: list[Point2D],
    priorities: list[float],
) -> float:
    return max(
        _schedule_order_value(start, goals, priorities, order)
        for order in permutations(range(len(goals)))
    )


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a percentile for summary metrics."""

    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=float), percentile_value))


def _write_csv(rows: Sequence[Any], output: Path) -> None:
    if not rows:
        raise ValueError("Cannot write benchmark CSV with no rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
