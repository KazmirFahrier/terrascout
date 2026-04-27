"""End-to-end TerraScout MVP mission runner."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from terrascout.control.pid import DriveController
from terrascout.localize.particle import ParticleLocalizer
from terrascout.mapping.ekf_slam import EkfSlam
from terrascout.mapping.landmarks import LandmarkMapper
from terrascout.plan.astar import GridAStarPlanner
from terrascout.plan.hybrid_astar import HybridAStarPlanner
from terrascout.safety.collision_guard import SafetySupervisor
from terrascout.scheduler.value_iteration import InspectionScheduler
from terrascout.sim.geometry import Point2D, Pose2D, distance
from terrascout.sim.battery import BatteryModel
from terrascout.sim.rover import DifferentialDriveRover
from terrascout.sim.scenario import load_scenario_config
from terrascout.sim.world import LidarDetection, OrchardWorld, ScenarioConfig
from terrascout.tracking.kalman import MultiObjectTracker


@dataclass(frozen=True)
class MissionMetrics:
    """Metrics emitted by a TerraScout mission run."""

    seed: int
    inspected_rows: int
    total_rows: int
    success_rate: float
    collisions: int
    path_length_m: float
    mission_time_s: float
    wall_time_s: float
    tracker_count: int
    mapped_landmarks: int
    slam_landmarks: int
    slam_covariance_trace: float
    mean_localization_error_m: float
    planner: str
    pose_source: str
    scheduler_value: float
    scheduler_dropped_goals: int
    battery_remaining_m: float
    daylight_remaining_s: float
    battery_soc_final: float
    battery_soc_min: float
    recharge_events: int
    safety_interventions: int
    safety_stops: int
    min_worker_clearance_m: float
    replans: int


@dataclass(frozen=True)
class MissionTrace:
    """Serializable path trace used by the renderer."""

    poses: list[tuple[float, float, float]]
    goals: list[tuple[float, float]]
    workers: list[list[tuple[float, float]]]


def run_mission(
    seed: int = 7,
    rows: int = 8,
    trees_per_row: int = 14,
    worker_count: int = 1,
    max_steps: int = 4200,
    dt: float = 0.05,
    planner_kind: str = "grid",
    pose_source: str = "truth",
    battery_budget_m: float = 140.0,
    daylight_budget_s: float = 180.0,
    max_goals: int | None = None,
    scenario_config: ScenarioConfig | None = None,
    trace_path: Path | None = None,
) -> MissionMetrics:
    """Run a complete deterministic orchard-inspection mission."""

    started = perf_counter()
    config = scenario_config or ScenarioConfig(
        rows=rows,
        trees_per_row=trees_per_row,
        worker_count=worker_count,
        random_seed=seed,
    )
    world = OrchardWorld(config)
    rover = DifferentialDriveRover(pose=Pose2D(x=1.0, y=0.8, theta=1.25), slip_fraction=0.04)
    battery = BatteryModel()
    controller = DriveController.default()
    tracker = MultiObjectTracker()
    mapper = LandmarkMapper()
    slam = EkfSlam(rover.pose)
    safety = SafetySupervisor()
    localizer = ParticleLocalizer.gaussian(
        500,
        mean=Pose2D(x=rover.pose.x + 0.25, y=rover.pose.y - 0.2, theta=rover.pose.theta + 0.08),
        std=(0.35, 0.35, 0.18),
        seed=seed + 101,
    )
    grid_planner = GridAStarPlanner(world)
    hybrid_planner = HybridAStarPlanner(world)

    if pose_source not in {"truth", "particle", "slam"}:
        raise ValueError(f"Unsupported pose_source: {pose_source}")
    if max_goals is not None and max_goals < 0:
        raise ValueError("max_goals must be non-negative")

    scheduler = InspectionScheduler()
    priorities = [1.0 + 0.25 * (idx % 3) for idx in range(len(world.row_goals))]
    candidate_indices = _candidate_goal_indices(priorities, max_goals)
    candidate_goals = [world.row_goals[idx] for idx in candidate_indices]
    candidate_priorities = [priorities[idx] for idx in candidate_indices]
    schedule = scheduler.plan_with_resources(
        rover.pose,
        candidate_goals,
        priorities=candidate_priorities,
        battery_budget_m=battery_budget_m,
        daylight_budget_s=daylight_budget_s,
    )
    ordered_goal_indices = schedule.order
    goals = [candidate_goals[idx] for idx in ordered_goal_indices]
    current_goal_idx = 0
    current_path: list[Point2D] = []
    current_waypoint_idx = 0
    replans = 0
    collisions = 0
    inspected: set[int] = set()
    path_length_m = 0.0
    previous_pose = rover.pose
    localization_error_sum = 0.0
    localization_error_count = 0
    safety_interventions = 0
    safety_stops = 0
    min_worker_clearance_m = float("inf")
    battery_soc_min = battery.soc_fraction
    recharge_events = 0
    was_recharging = False

    trace = MissionTrace(poses=[], goals=[(goal.x, goal.y) for goal in goals], workers=[])

    for step in range(max_steps):
        navigation_pose = _navigation_pose(pose_source, rover.pose, localizer.estimate(), slam.pose)
        detections = world.lidar_detections(rover.pose, include_workers=True, include_trees=True)
        local_detections = world.local_lidar_detections(rover.pose, include_workers=True, include_trees=True)
        tracker.update(detections, dt)
        if step % 2 == 0:
            mapper.update(rover.pose, local_detections)
        if step % 5 == 0:
            localizer.update(local_detections, world.trees)
            slam.update(local_detections)
        localization_error_sum += distance(localizer.estimate(), rover.pose)
        localization_error_count += 1

        if current_goal_idx >= len(goals):
            break
        goal = goals[current_goal_idx]
        if not current_path or current_waypoint_idx >= len(current_path) or step % 100 == 0:
            if planner_kind == "hybrid":
                hybrid_path = hybrid_planner.plan(
                    navigation_pose,
                    Pose2D(goal.x, goal.y, 0.0),
                    predicted_workers=tracker.predicted_positions(horizon_s=1.0),
                )
                current_path = [Point2D(pose.x, pose.y) for pose in hybrid_path]
            else:
                current_path = grid_planner.plan(
                    navigation_pose,
                    goal,
                    predicted_workers=tracker.predicted_positions(horizon_s=1.0),
                )
            current_waypoint_idx = 0
            replans += 1

        waypoint = current_path[current_waypoint_idx]
        if distance(navigation_pose, waypoint) < 0.38 and current_waypoint_idx < len(current_path) - 1:
            current_waypoint_idx += 1
            waypoint = current_path[current_waypoint_idx]

        left, right = controller.wheel_commands(navigation_pose, waypoint, dt)
        safety_decision = safety.supervise(
            pose=rover.pose,
            left_mps=left,
            right_mps=right,
            worker_detections=detections + _proximity_worker_detections(world, rover.pose, safety.slow_radius_m),
            predicted_workers=tracker.predicted_positions(horizon_s=1.0),
        )
        left, right = safety_decision.left_mps, safety_decision.right_mps
        safety_interventions += int(safety_decision.intervened)
        safety_stops += int(safety_decision.stopped)
        min_worker_clearance_m = min(min_worker_clearance_m, safety_decision.min_clearance_m)
        rover.command(left, right)
        localizer.predict(
            linear_mps=0.5 * (left + right),
            angular_rps=(right - left) / rover.wheel_base_m,
            dt=dt,
        )
        slam.predict(
            linear_mps=0.5 * (left + right),
            angular_rps=(right - left) / rover.wheel_base_m,
            dt=dt,
        )
        pose = rover.step(dt)
        world.step_workers(dt, avoid_pose=pose)

        step_distance_m = distance(previous_pose, pose)
        path_length_m += step_distance_m
        battery.consume(step_distance_m, dt)
        is_recharging = any(distance(pose, station) <= 0.75 for station in world.recharge_stations)
        if is_recharging:
            battery.recharge(dt)
            recharge_events += int(not was_recharging)
        was_recharging = is_recharging
        battery_soc_min = min(battery_soc_min, battery.soc_fraction)
        previous_pose = pose

        if world.collision_with_worker(pose):
            collisions += 1

        if distance(pose, goal) < 0.75:
            inspected.add(current_goal_idx)
            current_goal_idx += 1
            current_path = []
            current_waypoint_idx = 0
            controller.heading_pid.reset()
            controller.speed_pid.reset()

        if trace_path is not None and step % 4 == 0:
            trace.poses.append((pose.x, pose.y, pose.theta))
            trace.workers.append([(w.position.x, w.position.y) for w in world.workers])

    mission_time_s = min(max_steps * dt, (step + 1) * dt)
    metrics = MissionMetrics(
        seed=config.random_seed,
        inspected_rows=len(inspected),
        total_rows=len(goals),
        success_rate=len(inspected) / len(goals) if goals else 1.0,
        collisions=collisions,
        path_length_m=path_length_m,
        mission_time_s=mission_time_s,
        wall_time_s=perf_counter() - started,
        tracker_count=len(tracker.tracks),
        mapped_landmarks=len(mapper.landmarks),
        slam_landmarks=slam.landmark_count,
        slam_covariance_trace=float(slam.covariance.trace()),
        mean_localization_error_m=(
            localization_error_sum / localization_error_count if localization_error_count else 0.0
        ),
        planner=planner_kind,
        pose_source=pose_source,
        scheduler_value=schedule.expected_value,
        scheduler_dropped_goals=schedule.dropped_goals,
        battery_remaining_m=schedule.battery_remaining_m,
        daylight_remaining_s=schedule.time_remaining_s,
        battery_soc_final=battery.soc_fraction,
        battery_soc_min=battery_soc_min,
        recharge_events=recharge_events,
        safety_interventions=safety_interventions,
        safety_stops=safety_stops,
        min_worker_clearance_m=min_worker_clearance_m,
        replans=replans,
    )
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps({"metrics": asdict(metrics), "trace": asdict(trace)}, indent=2))
    return metrics


def _navigation_pose(
    pose_source: str,
    truth_pose: Pose2D,
    particle_pose: Pose2D,
    slam_pose: Pose2D,
) -> Pose2D:
    if pose_source == "particle":
        return particle_pose
    if pose_source == "slam":
        return slam_pose
    return truth_pose


def _candidate_goal_indices(priorities: list[float], max_goals: int | None) -> list[int]:
    if max_goals is None:
        return list(range(len(priorities)))
    ranked = sorted(range(len(priorities)), key=lambda idx: (-priorities[idx], idx))
    return ranked[:max_goals]


def _proximity_worker_detections(
    world: OrchardWorld,
    pose: Pose2D,
    radius_m: float,
) -> list[LidarDetection]:
    return [
        LidarDetection(worker.position.x, worker.position.y, "worker")
        for worker in world.workers
        if distance(pose, worker.position) <= radius_m
    ]


def write_metrics_csv(metrics: list[MissionMetrics], output: Path) -> None:
    """Write mission metrics as CSV."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for metric in metrics:
            writer.writerow(asdict(metric))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TerraScout MVP mission.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--trees-per-row", type=int, default=14)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--planner", choices=["grid", "hybrid"], default="grid")
    parser.add_argument("--pose-source", choices=["truth", "particle", "slam"], default="truth")
    parser.add_argument("--battery-budget-m", type=float, default=140.0)
    parser.add_argument("--daylight-budget-s", type=float, default=180.0)
    parser.add_argument("--max-goals", type=int, default=None)
    parser.add_argument("--scenario", type=Path, default=None)
    parser.add_argument("--trace", type=Path, default=Path("artifacts/mission_trace.json"))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    metrics = run_mission(
        seed=args.seed,
        rows=args.rows,
        trees_per_row=args.trees_per_row,
        worker_count=args.workers,
        planner_kind=args.planner,
        pose_source=args.pose_source,
        battery_budget_m=args.battery_budget_m,
        daylight_budget_s=args.daylight_budget_s,
        max_goals=args.max_goals,
        scenario_config=load_scenario_config(args.scenario) if args.scenario is not None else None,
        trace_path=args.trace,
    )
    print(json.dumps(asdict(metrics), indent=2))
    if args.csv is not None:
        write_metrics_csv([metrics], args.csv)


if __name__ == "__main__":
    main()
