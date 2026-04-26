"""Render a saved TerraScout mission trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.animation import FuncAnimation, PillowWriter

from terrascout.sim.world import OrchardWorld, ScenarioConfig


def render_trace(trace_json: Path, output_png: Path, seed: int = 7) -> None:
    """Render a static mission overview PNG."""

    payload = json.loads(trace_json.read_text())
    poses = payload["trace"]["poses"]
    goals = payload["trace"]["goals"]
    world = OrchardWorld(ScenarioConfig(random_seed=seed))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=140)
    ax.set_title("TerraScout MVP orchard inspection trace")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")

    ax.scatter([tree.x for tree in world.trees], [tree.y for tree in world.trees], s=18, c="#2f7d32")
    ax.scatter([goal[0] for goal in goals], [goal[1] for goal in goals], marker="*", s=100, c="#f1b82d")
    if poses:
        ax.plot([pose[0] for pose in poses], [pose[1] for pose in poses], c="#2454a6", linewidth=2.0)
        ax.scatter([poses[0][0]], [poses[0][1]], marker="o", s=70, c="#111111", label="start")
        ax.scatter([poses[-1][0]], [poses[-1][1]], marker="s", s=70, c="#d12f2f", label="finish")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)


def render_animation(trace_json: Path, output_gif: Path, seed: int = 7, max_frames: int = 140) -> None:
    """Render an animated GIF of a saved mission trace."""

    payload = json.loads(trace_json.read_text())
    poses = payload["trace"]["poses"]
    goals = payload["trace"]["goals"]
    workers = payload["trace"]["workers"]
    world = OrchardWorld(ScenarioConfig(random_seed=seed))
    if not poses:
        raise ValueError("Trace has no poses to render")

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, len(poses) // max_frames)
    frame_indices = list(range(0, len(poses), stride))

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=110)
    ax.set_title("TerraScout orchard inspection")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, world.width_m + 0.5)
    ax.set_ylim(-0.5, world.height_m + 0.8)
    ax.grid(True, alpha=0.22)
    ax.scatter([tree.x for tree in world.trees], [tree.y for tree in world.trees], s=16, c="#2f7d32")
    ax.scatter([goal[0] for goal in goals], [goal[1] for goal in goals], marker="*", s=90, c="#f1b82d")

    path_line, = ax.plot([], [], c="#2454a6", linewidth=2.3)
    rover_dot = ax.scatter([], [], marker="o", s=70, c="#111111", zorder=5)
    worker_dots = ax.scatter([], [], marker="x", s=45, c="#d12f2f", zorder=4)
    status = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")

    def update(frame_number: int) -> list[Artist]:
        idx = frame_indices[frame_number]
        path_line.set_data([pose[0] for pose in poses[: idx + 1]], [pose[1] for pose in poses[: idx + 1]])
        rover_dot.set_offsets([[poses[idx][0], poses[idx][1]]])
        if idx < len(workers) and workers[idx]:
            worker_dots.set_offsets(workers[idx])
        else:
            worker_dots.set_offsets(np.empty((0, 2)))
        status.set_text(f"t = {idx * 0.2:0.1f}s")
        return [path_line, rover_dot, worker_dots, status]

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=70, blit=True)
    animation.save(output_gif, writer=PillowWriter(fps=14))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a TerraScout mission trace PNG.")
    parser.add_argument("--trace", type=Path, default=Path("artifacts/mission_trace.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/mission_trace.png"))
    parser.add_argument("--gif", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    render_trace(args.trace, args.out, seed=args.seed)
    if args.gif is not None:
        render_animation(args.trace, args.gif, seed=args.seed)
    print(args.out)


if __name__ == "__main__":
    main()
