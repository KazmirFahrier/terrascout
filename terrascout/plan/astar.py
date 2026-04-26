"""Grid A* planner for the MVP mission runner."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from terrascout.sim.geometry import Point2D, Pose2D
from terrascout.sim.world import OrchardWorld


@dataclass(frozen=True)
class PlannerConfig:
    """Discretization and obstacle inflation settings."""

    resolution_m: float = 0.4
    tree_radius_m: float = 0.38
    worker_radius_m: float = 0.75
    edge_margin_m: float = 0.8


class GridAStarPlanner:
    """A* planner over an inflated orchard occupancy grid."""

    def __init__(self, world: OrchardWorld, config: PlannerConfig | None = None) -> None:
        self.world = world
        self.config = config or PlannerConfig()
        self.width_cells = int(np.ceil(world.width_m / self.config.resolution_m)) + 1
        self.height_cells = int(np.ceil(world.height_m / self.config.resolution_m)) + 1

    def plan(
        self,
        start: Pose2D | Point2D,
        goal: Point2D,
        predicted_workers: Iterable[tuple[int, float, float]] = (),
    ) -> list[Point2D]:
        """Plan a waypoint sequence from start to goal."""

        blocked = self._occupancy_grid(predicted_workers)
        start_idx = self._to_cell(start.x, start.y)
        goal_idx = self._nearest_free(self._to_cell(goal.x, goal.y), blocked)
        start_idx = self._nearest_free(start_idx, blocked)
        cells = self._astar(start_idx, goal_idx, blocked)
        if not cells:
            return [Point2D(goal.x, goal.y)]
        sparse = self._sparsify(cells)
        return [self._to_point(cx, cy) for cx, cy in sparse]

    def _occupancy_grid(self, predicted_workers: Iterable[tuple[int, float, float]]) -> NDArray[np.bool_]:
        grid = np.zeros((self.width_cells, self.height_cells), dtype=bool)
        for tree in self.world.trees:
            self._inflate(grid, tree.x, tree.y, self.config.tree_radius_m)
        for _, x, y in predicted_workers:
            self._inflate(grid, x, y, self.config.worker_radius_m)
        return grid

    def _inflate(self, grid: NDArray[np.bool_], x: float, y: float, radius_m: float) -> None:
        cx, cy = self._to_cell(x, y)
        radius_cells = int(np.ceil(radius_m / self.config.resolution_m))
        for ix in range(max(0, cx - radius_cells), min(self.width_cells, cx + radius_cells + 1)):
            for iy in range(max(0, cy - radius_cells), min(self.height_cells, cy + radius_cells + 1)):
                px, py = self._to_point(ix, iy).x, self._to_point(ix, iy).y
                if hypot(px - x, py - y) <= radius_m:
                    grid[ix, iy] = True

    def _astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: NDArray[np.bool_],
    ) -> list[tuple[int, int]]:
        open_set: list[tuple[float, tuple[int, int]]] = []
        heappush(open_set, (0.0, start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {start: 0.0}

        while open_set:
            _, current = heappop(open_set)
            if current == goal:
                return self._reconstruct(came_from, current)
            for neighbor, cost in self._neighbors(current, blocked):
                tentative = g_score[current] + cost
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    priority = tentative + self._heuristic(neighbor, goal)
                    heappush(open_set, (priority, neighbor))
        return []

    def _neighbors(
        self, cell: tuple[int, int], blocked: NDArray[np.bool_]
    ) -> Iterable[tuple[tuple[int, int], float]]:
        x, y = cell
        for dx, dy, cost in (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.414),
            (-1, 1, 1.414),
            (1, -1, 1.414),
            (1, 1, 1.414),
        ):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width_cells and 0 <= ny < self.height_cells and not blocked[nx, ny]:
                yield (nx, ny), cost

    def _nearest_free(self, cell: tuple[int, int], blocked: NDArray[np.bool_]) -> tuple[int, int]:
        x, y = cell
        if self._in_bounds(x, y) and not blocked[x, y]:
            return cell
        for radius in range(1, 12):
            for nx in range(x - radius, x + radius + 1):
                for ny in range(y - radius, y + radius + 1):
                    if self._in_bounds(nx, ny) and not blocked[nx, ny]:
                        return (nx, ny)
        return (max(0, min(self.width_cells - 1, x)), max(0, min(self.height_cells - 1, y)))

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width_cells and 0 <= y < self.height_cells

    def _reconstruct(
        self,
        came_from: dict[tuple[int, int], tuple[int, int]],
        current: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _sparsify(self, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if len(cells) <= 2:
            return cells
        sparse = [cells[0]]
        last_dir: tuple[int, int] | None = None
        for prev, cur in zip(cells, cells[1:]):
            direction = (cur[0] - prev[0], cur[1] - prev[1])
            if last_dir is not None and direction != last_dir:
                sparse.append(prev)
            last_dir = direction
        sparse.append(cells[-1])
        return sparse

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return hypot(a[0] - b[0], a[1] - b[1])

    def _to_cell(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(round(x / self.config.resolution_m)),
            int(round(y / self.config.resolution_m)),
        )

    def _to_point(self, x: int, y: int) -> Point2D:
        return Point2D(x * self.config.resolution_m, y * self.config.resolution_m)
