"""Grid A* planner for the MVP mission runner."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import ceil, hypot, isfinite, sqrt
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from terrascout.sim.collision import (
    ROVER_RADIUS_M,
    TREE_RADIUS_M,
    WORKER_RADIUS_M,
    segment_clearance,
)
from terrascout.sim.geometry import Point2D, Pose2D
from terrascout.sim.world import OrchardWorld


@dataclass(frozen=True)
class PlannerConfig:
    """Discretization and obstacle inflation settings."""

    resolution_m: float = 0.4
    tree_radius_m: float = ROVER_RADIUS_M + TREE_RADIUS_M + 0.2
    worker_radius_m: float = ROVER_RADIUS_M + WORKER_RADIUS_M + 0.4
    edge_margin_m: float = ROVER_RADIUS_M + 0.1

    def __post_init__(self) -> None:
        for name in ("resolution_m", "tree_radius_m", "worker_radius_m", "edge_margin_m"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.tree_radius_m < ROVER_RADIUS_M + TREE_RADIUS_M:
            raise ValueError("Tree inflation must cover the rover and trunk radii")
        if self.worker_radius_m < ROVER_RADIUS_M + WORKER_RADIUS_M:
            raise ValueError("Worker inflation must cover both bodies")
        if self.edge_margin_m < ROVER_RADIUS_M:
            raise ValueError("Boundary margin must cover the rover body")


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
        """Return a checked polyline including exact endpoints, or [] if unreachable."""
        workers = list(predicted_workers)
        if not self.segment_is_safe(start, start, workers) or not self.segment_is_safe(
            goal, goal, workers
        ):
            return []
        blocked = self._occupancy_grid(workers)
        start_idx = self._connected_cell(start, blocked, workers)
        goal_idx = self._connected_cell(goal, blocked, workers)
        if start_idx is None or goal_idx is None:
            return []
        cells = self._astar(start_idx, goal_idx, blocked)
        if not cells:
            return []
        points = [Point2D(start.x, start.y)]
        points.extend(self._to_point(cx, cy) for cx, cy in self._sparsify(cells))
        points.append(Point2D(goal.x, goal.y))
        # Never teleport from a blocked endpoint to a nearby free cell.
        if not all(self.segment_is_safe(a, b, workers) for a, b in zip(points, points[1:])):
            return []
        return points

    def _connected_cell(
        self,
        point: Point2D | Pose2D,
        blocked: NDArray[np.bool_],
        workers: list[tuple[int, float, float]],
    ) -> tuple[int, int] | None:
        """Connect a physically free pose to the conservative grid without teleporting."""
        x, y = self._to_cell(point.x, point.y)
        candidates = [
            (nx, ny)
            for nx in range(x - 3, x + 4)
            for ny in range(y - 3, y + 4)
            if self._in_bounds(nx, ny) and not blocked[nx, ny]
        ]
        candidates.sort(
            key=lambda cell: hypot(
                self._to_point(*cell).x - point.x, self._to_point(*cell).y - point.y
            )
        )
        for cell in candidates:
            if self.segment_is_safe(point, self._to_point(*cell), workers):
                return cell
        return None

    def segment_is_safe(
        self,
        start: Point2D | Pose2D,
        end: Point2D | Pose2D,
        predicted_workers: Iterable[tuple[int, float, float]] = (),
        clearance_margin_m: float = 0.0,
    ) -> bool:
        """Check the full swept footprint against the known map and perceived workers."""
        if not isfinite(clearance_margin_m) or clearance_margin_m < 0:
            raise ValueError("Additional clearance margin must be finite and nonnegative")
        margin = self.config.edge_margin_m + clearance_margin_m
        for point in (start, end):
            if not (
                isfinite(point.x)
                and isfinite(point.y)
                and margin <= point.x <= self.world.width_m - margin
                and margin <= point.y <= self.world.height_m - margin
            ):
                return False
        if any(
            segment_clearance(start, end, tree) <= self.config.tree_radius_m + clearance_margin_m
            for tree in self.world.trees
        ):
            return False
        return all(
            segment_clearance(start, end, Point2D(x, y))
            > self.config.worker_radius_m + clearance_margin_m
            for _, x, y in predicted_workers
        )

    def grid_segment_is_free(
        self,
        start: Point2D | Pose2D,
        end: Point2D | Pose2D,
        blocked: NDArray[np.bool_],
    ) -> bool:
        """Check intermediate grid cells as well as the continuous static footprint."""
        if not self.segment_is_safe(start, end):
            return False
        samples = max(
            1, ceil(hypot(end.x - start.x, end.y - start.y) / (self.config.resolution_m / 4))
        )
        for idx in range(samples + 1):
            fraction = idx / samples
            x, y = self._to_cell(
                start.x + fraction * (end.x - start.x), start.y + fraction * (end.y - start.y)
            )
            if not self._in_bounds(x, y) or blocked[x, y]:
                return False
        return True

    def _occupancy_grid(
        self, predicted_workers: Iterable[tuple[int, float, float]]
    ) -> NDArray[np.bool_]:
        grid = np.zeros((self.width_cells, self.height_cells), dtype=bool)
        # A free cell represents a square, not just its centre.
        padding = self.config.resolution_m / sqrt(2.0)
        xs = np.arange(self.width_cells) * self.config.resolution_m
        ys = np.arange(self.height_cells) * self.config.resolution_m
        margin = self.config.edge_margin_m
        grid |= (xs[:, None] < margin) | (xs[:, None] > self.world.width_m - margin)
        grid |= (ys[None, :] < margin) | (ys[None, :] > self.world.height_m - margin)
        for tree in self.world.trees:
            self._inflate(grid, tree.x, tree.y, self.config.tree_radius_m + padding)
        for _, x, y in predicted_workers:
            self._inflate(grid, x, y, self.config.worker_radius_m + padding)
        return grid

    def _inflate(self, grid: NDArray[np.bool_], x: float, y: float, radius_m: float) -> None:
        cx, cy = self._to_cell(x, y)
        radius_cells = int(np.ceil(radius_m / self.config.resolution_m))
        for ix in range(max(0, cx - radius_cells), min(self.width_cells, cx + radius_cells + 1)):
            for iy in range(
                max(0, cy - radius_cells), min(self.height_cells, cy + radius_cells + 1)
            ):
                px, py = self._to_point(ix, iy).x, self._to_point(ix, iy).y
                if hypot(px - x, py - y) <= radius_m:
                    grid[ix, iy] = True

    def _astar(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: NDArray[np.bool_],
    ) -> list[tuple[int, int]]:
        if (
            not self._in_bounds(*start)
            or not self._in_bounds(*goal)
            or blocked[start]
            or blocked[goal]
        ):
            return []
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
                if dx and dy and (blocked[x + dx, y] or blocked[x, y + dy]):
                    continue
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
