"""End-to-end TerraScout MVP mission runner."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from math import cos, sin, isfinite
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
from terrascout.sim.world import LidarDetection, LocalLidarDetection, OrchardWorld, ScenarioConfig
from terrascout.sim.collision import ROVER_RADIUS_M, moving_clearance
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
    min_worker_clearance_m: float | None
    replans: int
    scheduled_goals: int = 0
    status: str = "not_started"
    tree_collisions: int = 0
    worker_collisions: int = 0
    boundary_collisions: int = 0
    collision_frames: int = 0
    plan_failures: int = 0
    footprint_stops: int = 0
    forecast_battery_remaining_m: float = 0.0
    forecast_daylight_remaining_s: float = 0.0
    particle_localization_error_m: float = 0.0
    slam_localization_error_m: float = 0.0
    observation_model: str = "synthetic_labeled_centroids"
    worker_behavior: str = "independent"


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
    pose_source: str = "particle",
    battery_budget_m: float = 140.0,
    daylight_budget_s: float = 180.0,
    max_goals: int | None = None,
    scenario_config: ScenarioConfig | None = None,
    trace_path: Path | None = None,
) -> MissionMetrics:
    """Run a complete deterministic orchard-inspection mission."""

    _validate_run(
        max_steps, dt, planner_kind, pose_source, battery_budget_m, daylight_budget_s, max_goals
    )
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
        min_particles=450,
    )
    grid_planner = GridAStarPlanner(world)
    hybrid_planner = HybridAStarPlanner(world)

    scheduler = InspectionScheduler()
    priorities = [1.0 + 0.25 * (idx % 3) for idx in range(len(world.row_goals))]
    candidate_indices = _candidate_goal_indices(priorities, max_goals)
    candidate_goals = [world.row_goals[idx] for idx in candidate_indices]
    if len(candidate_goals) > 12:
        raise ValueError("Exact scheduling supports at most 12 candidate goals; set max_goals")
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
    replans = plan_failures = footprint_stops = 0
    collisions = collision_frames = 0
    tree_collisions = worker_collisions = boundary_collisions = 0
    active_contacts: set[str] = set()
    inspected: set[int] = set()
    path_length_m = elapsed_s = service_elapsed_s = 0.0
    selected_error_sum = particle_error_sum = slam_error_sum = 0.0
    localization_error_count = 0
    safety_interventions = safety_stops = 0
    min_worker_clearance_m = float("inf")
    battery_soc_min = battery.soc_fraction
    measured_speed_mps = 0.0
    status = "step_limit" if goals else "no_feasible_goals"
    trace = MissionTrace(poses=[], goals=[(goal.x, goal.y) for goal in goals], workers=[])

    for step in range(max_steps):
        if current_goal_idx >= len(goals):
            status = (
                "completed"
                if len(inspected) == len(candidate_goals) and candidate_goals
                else "partial_schedule"
            )
            break
        if elapsed_s >= daylight_budget_s - 1e-9:
            status = "daylight_exhausted"
            break
        if path_length_m >= battery_budget_m - 1e-9 or battery.soc_wh <= 1e-9:
            status = "battery_exhausted"
            break
        tick_dt = min(dt, daylight_budget_s - elapsed_s)
        tick_dt = min(tick_dt, battery.soc_wh * 3600.0 / battery.idle_watts)
        # One synthetic local observation frame supplies every perception consumer.
        # True rover pose is used only inside the simulator and evaluator.
        local_detections = world.local_lidar_detections(rover.pose)
        observation_valid = all(
            isfinite(det.range_m) and det.range_m >= 0.0 and isfinite(det.bearing_rad)
            for det in local_detections
        )
        if not observation_valid:
            rover.command(0.0, 0.0)
            status = "sensor_fault"
            break
        if step % 5 == 0:
            # The centroid simulator has 0.02 m range and 0.4 degree bearing
            # noise (about 0.06 m lateral noise at maximum range). Use a
            # conservative 0.10 m residual scale for this observation model.
            localizer.update(local_detections, world.trees, max_detections=10, sigma_m=0.10)
            slam.update(local_detections)
        particle_pose, slam_pose = localizer.estimate(), slam.pose
        navigation_pose = _navigation_pose(pose_source, rover.pose, particle_pose, slam_pose)
        selected_error_sum += distance(navigation_pose, rover.pose)
        particle_error_sum += distance(particle_pose, rover.pose)
        slam_error_sum += distance(slam_pose, rover.pose)
        localization_error_count += 1
        detections = _global_detections(navigation_pose, local_detections)
        tracker.update(detections, tick_dt)
        if step % 2 == 0:
            mapper.update(navigation_pose, local_detections)
        predicted_workers = tracker.predicted_positions(horizon_s=1.0)
        goal = goals[current_goal_idx]
        if not current_path or current_waypoint_idx >= len(current_path) or step % 100 == 0:
            # Failed paths are explicit. Retry periodically while workers keep moving.
            if current_path or step % 20 == 0:
                if planner_kind == "hybrid":
                    hybrid_path = hybrid_planner.plan(
                        navigation_pose, Pose2D(goal.x, goal.y, 0.0), predicted_workers
                    )
                    current_path = [Point2D(pose.x, pose.y) for pose in hybrid_path]
                else:
                    current_path = grid_planner.plan(navigation_pose, goal, predicted_workers)
                current_waypoint_idx = 0
                replans += 1
                plan_failures += int(not current_path)
        left = right = 0.0
        if current_path:
            while (
                current_waypoint_idx < len(current_path) - 1
                and distance(navigation_pose, current_path[current_waypoint_idx]) < 0.12
            ):
                current_waypoint_idx += 1
            # Follow dense hybrid samples with a checked lookahead; braking for
            # every short sample artificially slowed the same geometric route.
            lookahead_end = (
                len(current_path) if planner_kind == "hybrid" else current_waypoint_idx + 1
            )
            for index in range(current_waypoint_idx + 1, lookahead_end):
                candidate = current_path[index]
                if distance(navigation_pose, candidate) > 1.5:
                    break
                if not grid_planner.segment_is_safe(
                    navigation_pose, candidate, predicted_workers, clearance_margin_m=0.15
                ):
                    break
                current_waypoint_idx = index
            waypoint = current_path[current_waypoint_idx]
            left, right = controller.wheel_commands(
                navigation_pose, waypoint, tick_dt, measured_speed_mps
            )
        servicing = distance(navigation_pose, goal) < 0.35
        if servicing:
            left = right = 0.0
        decision = safety.supervise(
            navigation_pose, left, right, detections, predicted_workers, observation_age_s=0.0
        )
        left, right = decision.left_mps, decision.right_mps
        safety_interventions += int(decision.intervened)
        safety_stops += int(decision.stopped)
        # The known-map guard uses the selected estimate, never actual rover pose.
        proposed = DifferentialDriveRover(navigation_pose)
        proposed.command(left, right)
        projected_pose = proposed.step(tick_dt)
        if not grid_planner.segment_is_safe(navigation_pose, projected_pose):
            left = right = 0.0
            current_path = []
            footprint_stops += 1
        # Clamp motion before integration so actual distance and energy cannot
        # exceed the remaining execution budgets, even at a partial final tick.
        rover.command(left, right)
        commanded_distance = (
            abs(0.5 * (rover.left_velocity_mps + rover.right_velocity_mps)) * tick_dt
        )
        remaining_distance = min(
            battery_budget_m - path_length_m,
            max(0.0, battery.soc_wh - battery.idle_watts * tick_dt / 3600.0)
            / battery.drive_wh_per_m,
        )
        if commanded_distance > remaining_distance:
            scale = remaining_distance / commanded_distance
            rover.command(rover.left_velocity_mps * scale, rover.right_velocity_mps * scale)
        before = rover.pose
        worker_starts = [worker.position for worker in world.workers]
        pose = rover.step(tick_dt)
        world.step_workers(tick_dt)
        encoders = world.encoder_sample(rover, tick_dt)
        imu = world.imu_sample(rover)
        measured_speed_mps = (encoders.left_delta_m + encoders.right_delta_m) / (2.0 * tick_dt)
        localizer.predict(measured_speed_mps, imu.yaw_rate_rps, tick_dt)
        slam.predict(measured_speed_mps, imu.yaw_rate_rps, tick_dt)
        step_distance_m = distance(before, pose)
        path_length_m += step_distance_m
        elapsed_s += tick_dt
        battery.consume(step_distance_m, tick_dt)
        battery_soc_min = min(battery_soc_min, battery.soc_fraction)
        contacts = world.swept_contacts(before, pose, worker_starts)
        new_contacts = contacts - active_contacts
        collisions += len(new_contacts)
        tree_collisions += sum(item.startswith("tree:") for item in new_contacts)
        worker_collisions += sum(item.startswith("worker:") for item in new_contacts)
        boundary_collisions += int("boundary" in new_contacts)
        collision_frames += int(bool(contacts))
        active_contacts = contacts
        for worker, worker_before in zip(world.workers, worker_starts, strict=True):
            min_worker_clearance_m = min(
                min_worker_clearance_m,
                moving_clearance(before, pose, worker_before, worker.position)
                - ROVER_RADIUS_M
                - worker.radius_m,
            )
        service_elapsed_s = (
            service_elapsed_s + tick_dt if servicing and not decision.stopped else 0.0
        )
        if service_elapsed_s >= scheduler.config.service_time_s:
            # Endpoint service is not proof of traversing or inspecting an entire row.
            if distance(pose, goal) < 0.75:
                inspected.add(current_goal_idx)
            current_goal_idx += 1
            current_path = []
            current_waypoint_idx = 0
            service_elapsed_s = 0.0
            controller.heading_pid.reset()
            controller.speed_pid.reset()
        if trace_path is not None and (step % 4 == 0 or contacts):
            trace.poses.append((pose.x, pose.y, pose.theta))
            trace.workers.append([(w.position.x, w.position.y) for w in world.workers])
        if contacts:
            status = "collision"
            break
    rover.command(0.0, 0.0)
    if not goals:
        status = "no_feasible_goals" if candidate_goals else "no_goals_requested"
    elif current_goal_idx == len(goals) and status != "collision":
        status = "completed" if len(inspected) == len(candidate_goals) else "partial_schedule"
    count = localization_error_count or 1
    metrics = MissionMetrics(
        seed=config.random_seed,
        inspected_rows=len(inspected),
        total_rows=len(candidate_goals),
        success_rate=len(inspected) / len(candidate_goals) if candidate_goals else 0.0,
        collisions=collisions,
        path_length_m=path_length_m,
        mission_time_s=elapsed_s,
        wall_time_s=perf_counter() - started,
        tracker_count=len(tracker.tracks),
        mapped_landmarks=len(mapper.landmarks),
        slam_landmarks=slam.landmark_count,
        slam_covariance_trace=float(slam.covariance.trace()),
        mean_localization_error_m=selected_error_sum / count,
        planner=planner_kind,
        pose_source=pose_source,
        scheduler_value=schedule.expected_value,
        scheduler_dropped_goals=schedule.dropped_goals,
        battery_remaining_m=max(0.0, battery_budget_m - path_length_m),
        daylight_remaining_s=max(0.0, daylight_budget_s - elapsed_s),
        battery_soc_final=battery.soc_fraction,
        battery_soc_min=battery_soc_min,
        recharge_events=0,
        safety_interventions=safety_interventions,
        safety_stops=safety_stops,
        min_worker_clearance_m=min_worker_clearance_m if isfinite(min_worker_clearance_m) else None,
        replans=replans,
        scheduled_goals=len(goals),
        status=status,
        tree_collisions=tree_collisions,
        worker_collisions=worker_collisions,
        boundary_collisions=boundary_collisions,
        collision_frames=collision_frames,
        plan_failures=plan_failures,
        footprint_stops=footprint_stops,
        forecast_battery_remaining_m=schedule.battery_remaining_m,
        forecast_daylight_remaining_s=schedule.time_remaining_s,
        particle_localization_error_m=particle_error_sum / count,
        slam_localization_error_m=slam_error_sum / count,
    )
    if trace_path is not None:
        if not trace.poses or trace.poses[-1] != (rover.pose.x, rover.pose.y, rover.pose.theta):
            trace.poses.append((rover.pose.x, rover.pose.y, rover.pose.theta))
            trace.workers.append([(w.position.x, w.position.y) for w in world.workers])
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(
                {"metrics": asdict(metrics), "trace": asdict(trace), "scenario": asdict(config)},
                indent=2,
                allow_nan=False,
            )
        )
    return metrics


def _validate_run(
    max_steps: int,
    dt: float,
    planner: str,
    pose_source: str,
    battery: float,
    daylight: float,
    max_goals: int | None,
) -> None:
    if planner not in {"grid", "hybrid"}:
        raise ValueError(f"Unsupported planner: {planner}")
    if pose_source not in {"truth", "particle", "slam"}:
        raise ValueError(f"Unsupported pose_source: {pose_source}")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if not isfinite(dt) or not 0.0 < dt <= 0.1:
        raise ValueError("dt must be finite and in (0, 0.1]")
    if any(not isfinite(value) or value < 0 for value in (battery, daylight)):
        raise ValueError("Resource budgets must be finite and nonnegative")
    if max_goals is not None and (
        isinstance(max_goals, bool) or not isinstance(max_goals, int) or max_goals < 0
    ):
        raise ValueError("max_goals must be a nonnegative integer")


def _global_detections(pose: Pose2D, detections: list[LocalLidarDetection]) -> list[LidarDetection]:
    return [
        LidarDetection(
            pose.x + det.range_m * cos(pose.theta + det.bearing_rad),
            pose.y + det.range_m * sin(pose.theta + det.bearing_rad),
            det.kind,
        )
        for det in detections
    ]


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


def write_metrics_csv(metrics: list[MissionMetrics], output: Path) -> None:
    """Write mission metrics as CSV."""

    if not metrics:
        raise ValueError("At least one mission metric is required")
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
    parser.add_argument("--pose-source", choices=["truth", "particle", "slam"], default="particle")
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
    print(json.dumps(asdict(metrics), indent=2, allow_nan=False))
    if args.csv is not None:
        write_metrics_csv([metrics], args.csv)


if __name__ == "__main__":
    main()
