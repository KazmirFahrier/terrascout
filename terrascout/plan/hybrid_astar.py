"""Heading-aware Hybrid A* planner.

This planner searches a coarse `(x, y, theta)` lattice with forward and reverse
arc primitives. It is intentionally lightweight, but it captures the core
constraint missing from grid A*: the rover cannot teleport its heading.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import atan2, cos, hypot, pi, sin
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from terrascout.plan.astar import GridAStarPlanner, PlannerConfig
from terrascout.sim.geometry import Point2D, Pose2D, wrap_angle
from terrascout.sim.world import OrchardWorld


@dataclass(frozen=True)
class HybridPlannerConfig:
    """Parameters for a compact Hybrid A* search."""

    grid: PlannerConfig = PlannerConfig(resolution_m=0.4)
    heading_bins: int = 24
    step_m: float = 0.8
    min_turn_radius_m: float = 1.2
    goal_tolerance_m: float = 0.75
    max_expansions: int = 18_000
    turn_cost: float = 0.12
    reverse_cost: float = 0.9
    heading_cost: float = 0.15


class HybridAStarPlanner:
    """Hybrid A* planner over an inflated orchard cost map."""

    def __init__(self, world: OrchardWorld, config: HybridPlannerConfig | None = None) -> None:
        self.world = world
        self.config = config or HybridPlannerConfig()
        self.grid_planner = GridAStarPlanner(world, self.config.grid)

    def plan(
        self,
        start: Pose2D,
        goal: Pose2D | Point2D,
        predicted_workers: Iterable[tuple[int, float, float]] = (),
    ) -> list[Pose2D]:
        """Plan a kinematically feasible sequence of poses."""

        goal_pose = goal if isinstance(goal, Pose2D) else Pose2D(goal.x, goal.y, 0.0)
        blocked = self.grid_planner._occupancy_grid(predicted_workers)
        start_key = self._key(start)
        start_key = self._nearest_free_state(start_key, blocked)

        open_set: list[tuple[float, int, tuple[int, int, int]]] = []
        heappush(open_set, (self._heuristic(start, goal_pose), 0, start_key))
        came_from: dict[tuple[int, int, int], tuple[int, int, int]] = {}
        poses: dict[tuple[int, int, int], Pose2D] = {start_key: self._pose_from_key(start_key, start.theta)}
        g_score: dict[tuple[int, int, int], float] = {start_key: 0.0}
        counter = 1
        expansions = 0

        while open_set and expansions < self.config.max_expansions:
            _, _, current_key = heappop(open_set)
            current_pose = poses[current_key]
            expansions += 1
            if self._reached(current_pose, goal_pose):
                return self._reconstruct(came_from, poses, current_key, goal_pose)

            for next_pose, primitive_cost in self._expand(current_pose):
                next_key = self._key(next_pose)
                if not self._is_free(next_key, blocked):
                    continue
                tentative = g_score[current_key] + primitive_cost
                if tentative >= g_score.get(next_key, float("inf")):
                    continue
                came_from[next_key] = current_key
                poses[next_key] = next_pose
                g_score[next_key] = tentative
                priority = tentative + self._heuristic(next_pose, goal_pose)
                heappush(open_set, (priority, counter, next_key))
                counter += 1

        return self._grid_fallback(start, goal_pose, predicted_workers)

    def _expand(self, pose: Pose2D) -> Iterable[tuple[Pose2D, float]]:
        step = self.config.step_m
        turn_delta = step / self.config.min_turn_radius_m
        for direction in (1.0, -1.0):
            for turn in (-turn_delta, 0.0, turn_delta):
                theta_mid = pose.theta + 0.5 * direction * turn
                x = pose.x + direction * step * cos(theta_mid)
                y = pose.y + direction * step * sin(theta_mid)
                theta = wrap_angle(pose.theta + direction * turn)
                cost = step
                if abs(turn) > 1e-9:
                    cost += self.config.turn_cost
                if direction < 0.0:
                    cost += self.config.reverse_cost
                yield Pose2D(x, y, theta), cost

    def _key(self, pose: Pose2D) -> tuple[int, int, int]:
        x, y = self.grid_planner._to_cell(pose.x, pose.y)
        heading = int(round((wrap_angle(pose.theta) + pi) / (2.0 * pi) * self.config.heading_bins))
        return x, y, heading % self.config.heading_bins

    def _pose_from_key(self, key: tuple[int, int, int], theta_hint: float | None = None) -> Pose2D:
        point = self.grid_planner._to_point(key[0], key[1])
        theta = theta_hint
        if theta is None:
            theta = (key[2] / self.config.heading_bins) * 2.0 * pi - pi
        return Pose2D(point.x, point.y, wrap_angle(theta))

    def _nearest_free_state(
        self,
        key: tuple[int, int, int],
        blocked: NDArray[np.bool_],
    ) -> tuple[int, int, int]:
        x, y = self.grid_planner._nearest_free((key[0], key[1]), blocked)
        return x, y, key[2]

    def _is_free(self, key: tuple[int, int, int], blocked: NDArray[np.bool_]) -> bool:
        x, y, _ = key
        return self.grid_planner._in_bounds(x, y) and not blocked[x, y]

    def _heuristic(self, pose: Pose2D, goal: Pose2D) -> float:
        distance_cost = hypot(goal.x - pose.x, goal.y - pose.y)
        heading_to_goal = atan2(goal.y - pose.y, goal.x - pose.x)
        heading_error = abs(wrap_angle(heading_to_goal - pose.theta))
        return distance_cost + self.config.heading_cost * heading_error

    def _reached(self, pose: Pose2D, goal: Pose2D) -> bool:
        return hypot(goal.x - pose.x, goal.y - pose.y) <= self.config.goal_tolerance_m

    def _reconstruct(
        self,
        came_from: dict[tuple[int, int, int], tuple[int, int, int]],
        poses: dict[tuple[int, int, int], Pose2D],
        current: tuple[int, int, int],
        goal: Pose2D,
    ) -> list[Pose2D]:
        path = [poses[current]]
        while current in came_from:
            current = came_from[current]
            path.append(poses[current])
        path.reverse()
        path.append(goal)
        return self._sparsify(path)

    def _sparsify(self, path: list[Pose2D]) -> list[Pose2D]:
        if len(path) <= 2:
            return path
        sparse = [path[0]]
        last_heading_bin = self._key(path[0])[2]
        for pose in path[1:-1]:
            heading_bin = self._key(pose)[2]
            if heading_bin != last_heading_bin:
                sparse.append(pose)
            last_heading_bin = heading_bin
        sparse.append(path[-1])
        return sparse

    def _grid_fallback(
        self,
        start: Pose2D,
        goal: Pose2D,
        predicted_workers: Iterable[tuple[int, float, float]],
    ) -> list[Pose2D]:
        points = self.grid_planner.plan(start, Point2D(goal.x, goal.y), predicted_workers)
        poses: list[Pose2D] = []
        prev = start
        for point in points:
            theta = atan2(point.y - prev.y, point.x - prev.x)
            poses.append(Pose2D(point.x, point.y, theta))
            prev = poses[-1]
        if not poses or hypot(poses[-1].x - goal.x, poses[-1].y - goal.y) > 1e-6:
            poses.append(goal)
        return poses
