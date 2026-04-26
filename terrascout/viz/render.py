"""Render a saved TerraScout mission trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a TerraScout mission trace PNG.")
    parser.add_argument("--trace", type=Path, default=Path("artifacts/mission_trace.json"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/mission_trace.png"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    render_trace(args.trace, args.out, seed=args.seed)
    print(args.out)


if __name__ == "__main__":
    main()

