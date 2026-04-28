"""Tree-trunk detection from planar lidar scans."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import atan2, cos, hypot, sin

import numpy as np
from numpy.typing import NDArray

from terrascout.sim.sensors import LidarScan
from terrascout.sim.world import LocalLidarDetection


@dataclass(frozen=True)
class TrunkDetectorConfig:
    """RANSAC and clustering settings for scan-space trunk detection."""

    min_cluster_points: int = 3
    max_cluster_gap_m: float = 0.24
    min_radius_m: float = 0.08
    max_radius_m: float = 0.28
    inlier_tolerance_m: float = 0.055
    min_inliers: int = 3
    max_ransac_models: int = 96


def detect_tree_trunks(
    scan: LidarScan,
    config: TrunkDetectorConfig | None = None,
) -> list[LocalLidarDetection]:
    """Detect circular tree trunks from a 2D lidar scan.

    Consecutive finite-return beams are clustered in the rover frame. Each cluster
    is fit with a deterministic three-point RANSAC circle model, and accepted
    clusters are returned as range/bearing tree detections for EKF-SLAM.
    """

    cfg = config or TrunkDetectorConfig()
    points = _scan_points(scan)
    clusters = _clusters(points, cfg)
    detections: list[LocalLidarDetection] = []
    for cluster in clusters:
        fit = _ransac_circle(cluster, cfg)
        if fit is None:
            continue
        center_x, center_y, radius = fit
        if cfg.min_radius_m <= radius <= cfg.max_radius_m:
            detections.append(
                LocalLidarDetection(
                    range_m=hypot(center_x, center_y),
                    bearing_rad=atan2(center_y, center_x),
                    kind="tree",
                )
            )
    detections.sort(key=lambda detection: detection.range_m)
    return detections


def _scan_points(scan: LidarScan) -> list[tuple[float, float] | None]:
    points: list[tuple[float, float] | None] = []
    for angle, range_m in zip(scan.angles_rad, scan.ranges_m):
        if range_m >= scan.max_range_m - 1e-6:
            points.append(None)
        else:
            points.append((range_m * cos(angle), range_m * sin(angle)))
    return points


def _clusters(
    points: list[tuple[float, float] | None],
    config: TrunkDetectorConfig,
) -> list[NDArray[np.float64]]:
    clusters: list[NDArray[np.float64]] = []
    current: list[tuple[float, float]] = []
    previous: tuple[float, float] | None = None
    for point in points:
        if point is None:
            _append_cluster(clusters, current, config)
            current = []
            previous = None
            continue
        if previous is not None and hypot(point[0] - previous[0], point[1] - previous[1]) > config.max_cluster_gap_m:
            _append_cluster(clusters, current, config)
            current = []
        current.append(point)
        previous = point
    _append_cluster(clusters, current, config)
    return clusters


def _append_cluster(
    clusters: list[NDArray[np.float64]],
    current: list[tuple[float, float]],
    config: TrunkDetectorConfig,
) -> None:
    if len(current) >= config.min_cluster_points:
        clusters.append(np.array(current, dtype=float))


def _ransac_circle(
    points: NDArray[np.float64],
    config: TrunkDetectorConfig,
) -> tuple[float, float, float] | None:
    best: tuple[int, float, float, float, float] | None = None
    for indices in _candidate_triples(len(points), config.max_ransac_models):
        model = _circle_from_three(points[list(indices)])
        if model is None:
            continue
        center_x, center_y, radius = model
        residuals = np.abs(np.linalg.norm(points - np.array([[center_x, center_y]]), axis=1) - radius)
        inliers = int(np.sum(residuals <= config.inlier_tolerance_m))
        mean_error = float(np.mean(residuals[residuals <= config.inlier_tolerance_m])) if inliers else float("inf")
        candidate = (inliers, -mean_error, center_x, center_y, radius)
        if best is None or candidate > best:
            best = candidate
    if best is None or best[0] < config.min_inliers:
        return None
    inlier_residuals = np.abs(np.linalg.norm(points - np.array([[best[2], best[3]]]), axis=1) - best[4])
    inlier_points = points[inlier_residuals <= config.inlier_tolerance_m]
    refined = _least_squares_circle(inlier_points)
    if refined is not None:
        return refined
    _, _, center_x, center_y, radius = best
    return center_x, center_y, radius


def _candidate_triples(point_count: int, max_models: int) -> list[tuple[int, int, int]]:
    total_combinations = point_count * (point_count - 1) * (point_count - 2) // 6
    if total_combinations <= max_models:
        return list(combinations(range(point_count), 3))

    sample_count = min(point_count, max(6, int(round((max_models * 6) ** (1.0 / 3.0))) + 4))
    sampled_indices = np.linspace(0, point_count - 1, num=sample_count, dtype=int).tolist()
    unique_indices = sorted(set(sampled_indices))
    sampled = []
    for triple in combinations(unique_indices, 3):
        sampled.append(triple)
        if len(sampled) >= max_models:
            break

    anchors = {
        (0, point_count // 2, point_count - 1),
        (0, max(1, point_count // 3), max(2, (2 * point_count) // 3)),
        (max(0, point_count // 4), point_count // 2, point_count - 1),
    }
    for triple in anchors:
        if len(set(triple)) == 3 and triple not in sampled:
            sampled.append(triple)
    return sampled


def _circle_from_three(points: NDArray[np.float64]) -> tuple[float, float, float] | None:
    a = np.array(
        [
            [2.0 * (points[1, 0] - points[0, 0]), 2.0 * (points[1, 1] - points[0, 1])],
            [2.0 * (points[2, 0] - points[0, 0]), 2.0 * (points[2, 1] - points[0, 1])],
        ],
        dtype=float,
    )
    b = np.array(
        [
            points[1, 0] ** 2 + points[1, 1] ** 2 - points[0, 0] ** 2 - points[0, 1] ** 2,
            points[2, 0] ** 2 + points[2, 1] ** 2 - points[0, 0] ** 2 - points[0, 1] ** 2,
        ],
        dtype=float,
    )
    if abs(float(np.linalg.det(a))) < 1e-8:
        return None
    center = np.linalg.solve(a, b)
    radius = float(np.mean(np.linalg.norm(points - center[None, :], axis=1)))
    return float(center[0]), float(center[1]), radius


def _least_squares_circle(points: NDArray[np.float64]) -> tuple[float, float, float] | None:
    if len(points) < 3:
        return None
    a = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
    b = points[:, 0] ** 2 + points[:, 1] ** 2
    try:
        solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    center_x = float(solution[0])
    center_y = float(solution[1])
    radius_sq = center_x**2 + center_y**2 + float(solution[2])
    if radius_sq <= 0.0:
        return None
    return center_x, center_y, radius_sq**0.5
