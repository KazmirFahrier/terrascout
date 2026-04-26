"""Compare grid A* and Hybrid A* on deterministic orchard plans."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from math import hypot
from pathlib import Path
from time import perf_counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from terrascout.plan.astar import GridAStarPlanner
from terrascout.plan.hybrid_astar import HybridAStarPlanner
from terrascout.sim.geometry import Point2D, Pose2D
from terrascout.sim.world import OrchardWorld, ScenarioConfig


@dataclass(frozen=True)
class PlannerBenchmarkRow:
    seed: int
    planner: str
    waypoint_count: int
    path_length_m: float
    wall_time_ms: float


def _point_length(points: list[Point2D] | list[Pose2D]) -> float:
    return sum(hypot(b.x - a.x, b.y - a.y) for a, b in zip(points, points[1:]))


def run(output: Path = Path("artifacts/planner_benchmark.csv")) -> list[PlannerBenchmarkRow]:
    rows: list[PlannerBenchmarkRow] = []
    for seed in [2, 3, 5, 7, 11]:
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

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    return rows


def main() -> None:
    rows = run()
    grid_mean = sum(row.wall_time_ms for row in rows if row.planner == "grid_astar") / 5.0
    hybrid_mean = sum(row.wall_time_ms for row in rows if row.planner == "hybrid_astar") / 5.0
    print(f"grid_mean_ms={grid_mean:.2f} hybrid_mean_ms={hybrid_mean:.2f}")


if __name__ == "__main__":
    main()

