"""Render TerraScout milestone demo artifacts.

The project plan promises one portfolio artifact per milestone. This script keeps
those artifacts reproducible by generating them from the simulator, estimators,
planner, and mission runner instead of hand-edited screenshots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import cos, pi, sin
from pathlib import Path
import sys
import tempfile
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from terrascout.control.pid import DriveController  # noqa: E402
from terrascout.localize.particle import ParticleLocalizer  # noqa: E402
from terrascout.mapping.ekf_slam import EkfSlam  # noqa: E402
from terrascout.plan.astar import GridAStarPlanner  # noqa: E402
from terrascout.plan.hybrid_astar import HybridAStarPlanner  # noqa: E402
from terrascout.runner.mission import run_mission  # noqa: E402
from terrascout.sim.geometry import Point2D, Pose2D, distance  # noqa: E402
from terrascout.sim.rover import DifferentialDriveRover  # noqa: E402
from terrascout.sim.world import LidarDetection, OrchardWorld, ScenarioConfig  # noqa: E402
from terrascout.tracking.kalman import MultiObjectTracker  # noqa: E402


DOCS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DOCS_DIR / "milestones"
GIF_KWARGS = {"writer": PillowWriter(fps=12), "dpi": 90}


@dataclass(frozen=True)
class DemoArtifact:
    milestone: str
    title: str
    path: Path


def render_all(output_dir: Path = DEFAULT_OUTPUT, still_only: bool = False) -> list[DemoArtifact]:
    """Render all milestone artifacts and return their paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    renderers: list[Callable[[Path, bool], DemoArtifact]] = [
        _render_m1_control,
        _render_m2_tracker,
        _render_m3_mcl,
        _render_m4_slam,
        _render_m5_planner,
        _render_m6_mission,
    ]
    return [renderer(output_dir, still_only) for renderer in renderers]


def _render_m1_control(output_dir: Path, still_only: bool) -> DemoArtifact:
    waypoints = _control_reference_path()
    rover = DifferentialDriveRover(Pose2D(0.0, 0.0, 0.0), slip_fraction=0.025)
    controller = DriveController.default()
    poses: list[Pose2D] = []
    waypoint_index = 0
    dt = 0.05
    for step in range(1800):
        target = waypoints[waypoint_index]
        if distance(rover.pose, target) < 0.28 and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1
            target = waypoints[waypoint_index]
        left, right = controller.wheel_commands(rover.pose, target, dt)
        rover.command(left, right)
        rover.step(dt)
        if step % 4 == 0:
            poses.append(rover.pose)

    path = output_dir / ("m1_control.png" if still_only else "m1_control.gif")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    _setup_axis(ax, "M1 PID control: square + figure-eight")
    ax.plot([p.x for p in waypoints], [p.y for p in waypoints], color="#bbbbbb", linewidth=1.4)
    actual_line, = ax.plot([], [], color="#2454a6", linewidth=2.0)
    rover_dot = ax.scatter([], [], s=50, color="#111111", zorder=3)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-2.4, 2.8)
    if still_only:
        actual_line.set_data([p.x for p in poses], [p.y for p in poses])
        rover_dot.set_offsets([[poses[-1].x, poses[-1].y]])
        fig.savefig(path, bbox_inches="tight")
    else:
        frames = _frame_indices(len(poses), 100)

        def update(frame: int) -> list[Artist]:
            idx = frames[frame]
            actual_line.set_data([p.x for p in poses[: idx + 1]], [p.y for p in poses[: idx + 1]])
            rover_dot.set_offsets([[poses[idx].x, poses[idx].y]])
            return [actual_line, rover_dot]

        FuncAnimation(fig, update, frames=len(frames), interval=75, blit=True).save(path, **GIF_KWARGS)
    plt.close(fig)
    return DemoArtifact("M1", "PID square and figure-eight tracking", path)


def _render_m2_tracker(output_dir: Path, still_only: bool) -> DemoArtifact:
    rng = np.random.default_rng(21)
    worker_count = 10
    positions = np.column_stack([rng.uniform(0.5, 7.5, worker_count), rng.uniform(0.5, 5.0, worker_count)])
    velocities = np.column_stack([rng.uniform(-0.25, 0.25, worker_count), rng.uniform(-0.2, 0.2, worker_count)])
    tracker = MultiObjectTracker(gate_m=1.0, max_missed=5)
    truth_history: list[NDArray[np.float64]] = []
    prediction_history: list[NDArray[np.float64]] = []
    dt = 0.1
    for _ in range(90):
        detections = [
            LidarDetection(float(x), float(y), "worker")
            for x, y in positions + rng.normal(0.0, 0.035, positions.shape)
        ]
        tracker.update(detections, dt)
        predictions = np.array([[x, y] for _, x, y in tracker.predicted_positions(1.0)], dtype=float)
        truth_history.append(positions.copy())
        prediction_history.append(predictions)
        positions = positions + velocities * dt
        bounced_x = (positions[:, 0] < 0.3) | (positions[:, 0] > 7.7)
        bounced_y = (positions[:, 1] < 0.3) | (positions[:, 1] > 5.2)
        velocities[bounced_x, 0] *= -1.0
        velocities[bounced_y, 1] *= -1.0
        positions[:, 0] = np.clip(positions[:, 0], 0.3, 7.7)
        positions[:, 1] = np.clip(positions[:, 1], 0.3, 5.2)

    path = output_dir / ("m2_tracker.png" if still_only else "m2_tracker.gif")
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    _setup_axis(ax, "M2 Kalman tracker: truth vs. 1s prediction")
    ax.set_xlim(0, 8.0)
    ax.set_ylim(0, 5.5)
    truth = ax.scatter([], [], s=45, color="#2454a6", label="truth")
    prediction = ax.scatter([], [], s=55, marker="x", color="#d12f2f", label="predicted")
    ax.legend(loc="upper right")
    if still_only:
        truth.set_offsets(truth_history[-1])
        prediction.set_offsets(prediction_history[-1])
        fig.savefig(path, bbox_inches="tight")
    else:
        frames = _frame_indices(len(truth_history), 80)

        def update(frame: int) -> list[Artist]:
            idx = frames[frame]
            truth.set_offsets(truth_history[idx])
            prediction.set_offsets(prediction_history[idx])
            return [truth, prediction]

        FuncAnimation(fig, update, frames=len(frames), interval=75, blit=True).save(path, **GIF_KWARGS)
    plt.close(fig)
    return DemoArtifact("M2", "Kalman worker tracking overlay", path)


def _render_m3_mcl(output_dir: Path, still_only: bool) -> DemoArtifact:
    world = OrchardWorld(ScenarioConfig(rows=4, trees_per_row=6, worker_count=0, random_seed=7))
    truth = Pose2D(6.13, 9.28, 0.55)
    localizer = ParticleLocalizer.gaussian(
        3000,
        mean=Pose2D(9.45, 7.35, 0.94),
        std=(2.2, 2.2, 0.63),
        seed=107,
    )
    detections = world.local_lidar_detections(truth, include_workers=False)
    snapshots: list[tuple[NDArray[np.float64], Pose2D]] = [(localizer.particles.copy(), localizer.estimate())]
    localizer.scan_match_reset(detections, world.trees)
    snapshots.append((localizer.particles.copy(), localizer.estimate()))
    for _ in range(5):
        localizer.update(detections, world.trees)
        snapshots.append((localizer.particles.copy(), localizer.estimate()))

    path = output_dir / ("m3_mcl.png" if still_only else "m3_mcl.gif")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    _setup_axis(ax, "M3 MCL: particle cloud convergence")
    ax.scatter([tree.x for tree in world.trees], [tree.y for tree in world.trees], s=20, color="#2f7d32")
    ax.scatter([truth.x], [truth.y], marker="*", s=140, color="#f1b82d", label="truth")
    particles = ax.scatter([], [], s=5, alpha=0.18, color="#2454a6")
    estimate = ax.scatter([], [], marker="x", s=80, color="#d12f2f", label="estimate")
    ax.set_xlim(-0.5, world.width_m + 4.0)
    ax.set_ylim(-0.5, world.height_m + 1.5)
    ax.legend(loc="upper right")
    if still_only:
        _set_particle_frame(particles, estimate, snapshots[-1])
        fig.savefig(path, bbox_inches="tight")
    else:
        def update(frame: int) -> list[Artist]:
            _set_particle_frame(particles, estimate, snapshots[frame])
            return [particles, estimate]

        FuncAnimation(fig, update, frames=len(snapshots), interval=500, blit=True).save(path, **GIF_KWARGS)
    plt.close(fig)
    return DemoArtifact("M3", "Particle-filter convergence heat map", path)


def _render_m4_slam(output_dir: Path, still_only: bool) -> DemoArtifact:
    seeds = [2, 3, 5, 7, 11]
    frames = [_slam_layout(seed) for seed in seeds]
    path = output_dir / ("m4_slam_overlay.png" if still_only else "m4_slam_overlay.gif")
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _setup_axis(ax, "M4 EKF-SLAM: estimated trunks vs. truth")
    truth_scatter = ax.scatter([], [], s=14, color="#2f7d32", label="truth trunks")
    estimate_scatter = ax.scatter([], [], marker="x", s=24, color="#d12f2f", label="EKF landmarks")
    path_line, = ax.plot([], [], color="#2454a6", linewidth=1.5, label="rover path")
    ax.legend(loc="upper right")
    if still_only:
        _set_slam_frame(ax, truth_scatter, estimate_scatter, path_line, frames[-1])
        fig.savefig(path, bbox_inches="tight")
    else:
        def update(frame: int) -> list[Artist]:
            _set_slam_frame(ax, truth_scatter, estimate_scatter, path_line, frames[frame])
            return [truth_scatter, estimate_scatter, path_line]

        FuncAnimation(fig, update, frames=len(frames), interval=900, blit=True).save(path, **GIF_KWARGS)
    plt.close(fig)
    return DemoArtifact("M4", "EKF-SLAM map overlay across layouts", path)


def _render_m5_planner(output_dir: Path, still_only: bool) -> DemoArtifact:
    seeds = [2, 3, 5, 7, 11]
    frames = [_planner_layout(seed) for seed in seeds]
    path = output_dir / ("m5_planner_comparison.png" if still_only else "m5_planner_comparison.gif")
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharex=True, sharey=True)
    for ax, title in zip(axes, ("Grid A*", "Hybrid A*")):
        _setup_axis(ax, f"M5 planner: {title}")
    tree_scatter = [ax.scatter([], [], s=12, color="#2f7d32") for ax in axes]
    grid_line, = axes[0].plot([], [], color="#2454a6", linewidth=2.0)
    hybrid_line, = axes[1].plot([], [], color="#d12f2f", linewidth=2.0)
    if still_only:
        _set_planner_frame(axes, tree_scatter, grid_line, hybrid_line, frames[-1])
        fig.savefig(path, bbox_inches="tight")
    else:
        def update(frame: int) -> list[Artist]:
            return _set_planner_frame(axes, tree_scatter, grid_line, hybrid_line, frames[frame])

        FuncAnimation(fig, update, frames=len(frames), interval=900, blit=True).save(path, **GIF_KWARGS)
    plt.close(fig)
    return DemoArtifact("M5", "Grid A* versus Hybrid A* planner", path)


def _render_m6_mission(output_dir: Path, still_only: bool) -> DemoArtifact:
    with tempfile.TemporaryDirectory() as tmp_dir:
        trace_path = Path(tmp_dir) / "m6_final_mission_trace.json"
        run_mission(
            seed=7,
            rows=30,
            trees_per_row=14,
            worker_count=1,
            battery_budget_m=700.0,
            daylight_budget_s=900.0,
            max_goals=10,
            trace_path=trace_path,
        )
        payload = json.loads(trace_path.read_text())
    poses = [Pose2D(*pose) for pose in payload["trace"]["poses"]]
    workers = payload["trace"]["workers"]
    goals = [Point2D(goal[0], goal[1]) for goal in payload["trace"]["goals"]]
    world = OrchardWorld(ScenarioConfig(rows=30, trees_per_row=14, worker_count=1, random_seed=7))
    path = output_dir / ("m6_final_mission.png" if still_only else "m6_final_mission.gif")
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    _setup_axis(ax, "M6 final demo: 30-row priority pass")
    ax.scatter([tree.x for tree in world.trees], [tree.y for tree in world.trees], s=8, color="#2f7d32")
    ax.scatter([goal.x for goal in goals], [goal.y for goal in goals], marker="*", s=70, color="#f1b82d")
    path_line, = ax.plot([], [], color="#2454a6", linewidth=1.9)
    rover_dot = ax.scatter([], [], s=55, color="#111111", zorder=3)
    worker_dot = ax.scatter([], [], marker="x", s=42, color="#d12f2f", zorder=3)
    ax.set_xlim(0, world.width_m + 0.5)
    ax.set_ylim(-0.5, world.height_m + 0.8)
    if still_only:
        _set_mission_frame(path_line, rover_dot, worker_dot, poses, workers, len(poses) - 1)
        fig.savefig(path, bbox_inches="tight")
    else:
        frames = _frame_indices(len(poses), 120)

        def update(frame: int) -> list[Artist]:
            return _set_mission_frame(path_line, rover_dot, worker_dot, poses, workers, frames[frame])

        FuncAnimation(fig, update, frames=len(frames), interval=75, blit=True).save(path, **GIF_KWARGS)
    plt.close(fig)
    return DemoArtifact("M6", "30-row end-to-end mission", path)


def _control_reference_path() -> list[Point2D]:
    square = [
        Point2D(2.0, 0.0),
        Point2D(2.0, 2.0),
        Point2D(0.0, 2.0),
        Point2D(0.0, 0.0),
        Point2D(1.2, 0.0),
    ]
    figure_eight = [
        Point2D(1.2 * sin(t), 1.0 + 0.85 * sin(t) * cos(t))
        for t in np.linspace(0.0, 2.0 * pi, 38)
    ]
    return [*square, *figure_eight]


def _slam_layout(seed: int) -> dict[str, object]:
    world = OrchardWorld(ScenarioConfig(rows=8, trees_per_row=14, worker_count=0, random_seed=seed))
    start = Pose2D(world.config.width_margin_m + 0.5 * world.config.row_spacing_m, 1.0, pi / 2.0)
    rover = DifferentialDriveRover(start)
    slam = EkfSlam(start)
    controller = DriveController.default()
    waypoints = [
        Point2D(world.config.width_margin_m + (lane + 0.5) * world.config.row_spacing_m, world.height_y - 1.0)
        for lane in range(3)
    ]
    poses: list[Pose2D] = []
    waypoint_index = 0
    dt = 0.5
    for step in range(150):
        waypoint = waypoints[waypoint_index]
        if distance(rover.pose, waypoint) < 0.8 and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1
            waypoint = waypoints[waypoint_index]
        left, right = controller.wheel_commands(rover.pose, waypoint, dt)
        linear = 0.5 * (left + right)
        angular = (right - left) / rover.wheel_base_m
        rover.command(left, right)
        rover.step(dt)
        slam.predict(linear, angular, dt)
        slam.update(world.local_lidar_detections(rover.pose, include_workers=False))
        if step % 3 == 0:
            poses.append(rover.pose)
    return {
        "world": world,
        "truth": np.array([[tree.x, tree.y] for tree in world.trees], dtype=float),
        "estimated": np.array([[lm.x, lm.y] for lm in slam.landmarks()], dtype=float),
        "path": np.array([[pose.x, pose.y] for pose in poses], dtype=float),
    }


def _planner_layout(seed: int) -> dict[str, object]:
    world = OrchardWorld(ScenarioConfig(rows=8, trees_per_row=14, worker_count=0, random_seed=seed))
    start = Pose2D(1.0, 0.8, 1.2)
    goal = world.row_goals[-1]
    goal_pose = Pose2D(goal.x, goal.y, 0.0)
    grid_path = GridAStarPlanner(world).plan(start, goal)
    hybrid_path = HybridAStarPlanner(world).plan(start, goal_pose)
    return {
        "world": world,
        "trees": np.array([[tree.x, tree.y] for tree in world.trees], dtype=float),
        "grid": np.array([[point.x, point.y] for point in grid_path], dtype=float),
        "hybrid": np.array([[pose.x, pose.y] for pose in hybrid_path], dtype=float),
    }


def _setup_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)


def _frame_indices(count: int, max_frames: int) -> list[int]:
    stride = max(1, count // max_frames)
    return list(range(0, count, stride))


def _set_particle_frame(
    particles: Artist,
    estimate: Artist,
    snapshot: tuple[NDArray[np.float64], Pose2D],
) -> None:
    cloud, pose = snapshot
    stride = max(1, len(cloud) // 700)
    particles.set_offsets(cloud[::stride, :2])
    estimate.set_offsets([[pose.x, pose.y]])


def _set_slam_frame(
    ax: plt.Axes,
    truth_scatter: Artist,
    estimate_scatter: Artist,
    path_line: Artist,
    frame: dict[str, object],
) -> None:
    world = frame["world"]
    assert isinstance(world, OrchardWorld)
    truth_scatter.set_offsets(frame["truth"])
    estimate_scatter.set_offsets(frame["estimated"])
    path_xy = frame["path"]
    path_line.set_data(path_xy[:, 0], path_xy[:, 1])
    ax.set_xlim(0, world.width_m + 0.5)
    ax.set_ylim(-0.5, world.height_m + 0.8)


def _set_planner_frame(
    axes: NDArray[np.object_] | list[plt.Axes],
    tree_scatter: list[Artist],
    grid_line: Artist,
    hybrid_line: Artist,
    frame: dict[str, object],
) -> list[Artist]:
    world = frame["world"]
    assert isinstance(world, OrchardWorld)
    trees = frame["trees"]
    grid = frame["grid"]
    hybrid = frame["hybrid"]
    for ax, scatter in zip(axes, tree_scatter):
        scatter.set_offsets(trees)
        ax.set_xlim(0, world.width_m + 0.5)
        ax.set_ylim(-0.5, world.height_m + 0.8)
    grid_line.set_data(grid[:, 0], grid[:, 1])
    hybrid_line.set_data(hybrid[:, 0], hybrid[:, 1])
    return [*tree_scatter, grid_line, hybrid_line]


def _set_mission_frame(
    path_line: Artist,
    rover_dot: Artist,
    worker_dot: Artist,
    poses: list[Pose2D],
    workers: list[list[list[float]]],
    idx: int,
) -> list[Artist]:
    path_line.set_data([pose.x for pose in poses[: idx + 1]], [pose.y for pose in poses[: idx + 1]])
    rover_dot.set_offsets([[poses[idx].x, poses[idx].y]])
    worker_dot.set_offsets(workers[idx] if idx < len(workers) and workers[idx] else np.empty((0, 2)))
    return [path_line, rover_dot, worker_dot]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render TerraScout milestone demo artifacts.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--still-only", action="store_true", help="Render PNG stills instead of animated GIFs.")
    args = parser.parse_args()
    artifacts = render_all(args.output_dir, still_only=args.still_only)
    for artifact in artifacts:
        print(f"{artifact.milestone}: {artifact.path}")


if __name__ == "__main__":
    main()
