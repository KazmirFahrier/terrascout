"""End-to-end TerraScout MVP mission runner."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

from terrascout.control.pid import DriveController
from terrascout.plan.astar import GridAStarPlanner
from terrascout.sim.geometry import Point2D, Pose2D, distance
from terrascout.sim.rover import DifferentialDriveRover
from terrascout.sim.world import OrchardWorld, ScenarioConfig
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
    trace_path: Path | None = None,
) -> MissionMetrics:
    """Run a complete deterministic orchard-inspection mission."""

    started = perf_counter()
    world = OrchardWorld(
        ScenarioConfig(
            rows=rows,
            trees_per_row=trees_per_row,
            worker_count=worker_count,
            random_seed=seed,
        )
    )
    rover = DifferentialDriveRover(pose=Pose2D(x=1.0, y=0.8, theta=1.25), slip_fraction=0.04)
    controller = DriveController.default()
    tracker = MultiObjectTracker()
    planner = GridAStarPlanner(world)

    goals = world.row_goals
    current_goal_idx = 0
    current_path: list[Point2D] = []
    current_waypoint_idx = 0
    replans = 0
    collisions = 0
    inspected: set[int] = set()
    path_length_m = 0.0
    previous_pose = rover.pose

    trace = MissionTrace(poses=[], goals=[(goal.x, goal.y) for goal in goals], workers=[])

    for step in range(max_steps):
        detections = world.lidar_detections(rover.pose, include_workers=True, include_trees=True)
        tracker.update(detections, dt)

        if current_goal_idx >= len(goals):
            break
        goal = goals[current_goal_idx]
        if not current_path or current_waypoint_idx >= len(current_path) or step % 100 == 0:
            current_path = planner.plan(
                rover.pose,
                goal,
                predicted_workers=tracker.predicted_positions(horizon_s=1.0),
            )
            current_waypoint_idx = 0
            replans += 1

        waypoint = current_path[current_waypoint_idx]
        if distance(rover.pose, waypoint) < 0.38 and current_waypoint_idx < len(current_path) - 1:
            current_waypoint_idx += 1
            waypoint = current_path[current_waypoint_idx]

        worker_clearance_m = min(
            (distance(rover.pose, worker.position) for worker in world.workers),
            default=float("inf"),
        )
        if worker_clearance_m < 1.45:
            left, right = 0.0, 0.0
        else:
            left, right = controller.wheel_commands(rover.pose, waypoint, dt)
        rover.command(left, right)
        pose = rover.step(dt)
        world.step_workers(dt)

        path_length_m += distance(previous_pose, pose)
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
        seed=seed,
        inspected_rows=len(inspected),
        total_rows=len(goals),
        success_rate=len(inspected) / len(goals),
        collisions=collisions,
        path_length_m=path_length_m,
        mission_time_s=mission_time_s,
        wall_time_s=perf_counter() - started,
        tracker_count=len(tracker.tracks),
        replans=replans,
    )
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps({"metrics": asdict(metrics), "trace": asdict(trace)}, indent=2))
    return metrics


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
    parser.add_argument("--trace", type=Path, default=Path("artifacts/mission_trace.json"))
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    metrics = run_mission(
        seed=args.seed,
        rows=args.rows,
        trees_per_row=args.trees_per_row,
        worker_count=args.workers,
        trace_path=args.trace,
    )
    print(json.dumps(asdict(metrics), indent=2))
    if args.csv is not None:
        write_metrics_csv([metrics], args.csv)


if __name__ == "__main__":
    main()
