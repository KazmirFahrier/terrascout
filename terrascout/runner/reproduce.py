"""Single-command reproduction workflow for TerraScout."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from terrascout.eval.benchmarks import (
    ControlBenchmarkRow,
    LocalizationBenchmarkRow,
    PlannerBenchmarkRow,
    SlamBenchmarkRow,
    StressSummaryRow,
    TrackingBenchmarkRow,
    run_mission_benchmark,
    run_control_benchmark,
    run_localization_benchmark,
    run_planner_benchmark,
    run_slam_benchmark,
    run_stress_benchmark,
    run_tracking_benchmark,
)
from terrascout.runner.mission import MissionMetrics, run_mission, write_metrics_csv
from terrascout.viz.render import render_animation, render_trace


def run_reproduce(
    artifacts_dir: Path = Path("artifacts"),
    seed: int = 7,
    generate_gif: bool = True,
) -> dict[str, Any]:
    """Run the demo, render outputs, benchmarks, and write a summary JSON."""

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    trace_path = artifacts_dir / "mission_trace.json"
    png_path = artifacts_dir / "mission_trace.png"
    gif_path = artifacts_dir / "mission_trace.gif"
    mission_csv_path = artifacts_dir / "mission_metrics.csv"

    mission = run_mission(seed=seed, trace_path=trace_path)
    write_metrics_csv([mission], mission_csv_path)
    render_trace(trace_path, png_path, seed=seed)
    outputs = {
        "mission_trace_json": trace_path,
        "mission_trace_png": png_path,
        "mission_metrics_csv": mission_csv_path,
    }
    if generate_gif:
        render_animation(trace_path, gif_path, seed=seed)
        outputs["mission_trace_gif"] = gif_path

    mission_rows = run_mission_benchmark(artifacts_dir / "benchmark.csv")
    control_rows = run_control_benchmark(artifacts_dir / "control_benchmark.csv")
    tracking_rows = run_tracking_benchmark(artifacts_dir / "tracking_benchmark.csv")
    localization_rows = run_localization_benchmark(artifacts_dir / "localization_benchmark.csv")
    planner_rows = run_planner_benchmark(artifacts_dir / "planner_benchmark.csv")
    slam_rows = run_slam_benchmark(artifacts_dir / "slam_benchmark.csv")
    stress_rows = run_stress_benchmark(
        detail_output=artifacts_dir / "stress_benchmark_detail.csv",
        summary_output=artifacts_dir / "stress_benchmark_summary.csv",
    )

    summary = build_reproduce_summary(
        artifacts_dir=artifacts_dir,
        mission=mission,
        mission_rows=mission_rows,
        control_rows=control_rows,
        tracking_rows=tracking_rows,
        localization_rows=localization_rows,
        planner_rows=planner_rows,
        slam_rows=slam_rows,
        stress_rows=stress_rows,
        outputs=outputs,
    )
    summary_path = artifacts_dir / "reproduce_summary.json"
    summary["outputs"]["reproduce_summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def build_reproduce_summary(
    artifacts_dir: Path,
    mission: MissionMetrics,
    mission_rows: list[MissionMetrics],
    control_rows: list[ControlBenchmarkRow],
    tracking_rows: list[TrackingBenchmarkRow],
    localization_rows: list[LocalizationBenchmarkRow],
    planner_rows: list[PlannerBenchmarkRow],
    slam_rows: list[SlamBenchmarkRow],
    stress_rows: list[StressSummaryRow],
    outputs: dict[str, Path],
) -> dict[str, Any]:
    """Build the reviewer-facing JSON summary for a reproduce run."""

    return {
        "artifacts_dir": str(artifacts_dir),
        "mission": asdict(mission),
        "benchmark_summary": {
            "mission_runs": len(mission_rows),
            "mission_mean_success_rate": _mean([row.success_rate for row in mission_rows]),
            "mission_total_collisions": sum(row.collisions for row in mission_rows),
            "control_max_cross_track_error_m": max(
                (row.max_cross_track_error_m for row in control_rows),
                default=0.0,
            ),
            "control_max_heading_settle_time_s": max(
                (row.heading_settle_time_s for row in control_rows),
                default=0.0,
            ),
            "control_max_heading_overshoot_fraction": max(
                (row.heading_overshoot_fraction for row in control_rows),
                default=0.0,
            ),
            "tracking_mean_prediction_error_m": _mean(
                [row.mean_prediction_error_m for row in tracking_rows]
            ),
            "tracking_mean_association_accuracy": _mean(
                [row.association_accuracy for row in tracking_rows]
            ),
            "localization_mean_pose_error_m": _mean(
                [row.final_pose_error_m for row in localization_rows]
            ),
            "localization_p95_pose_error_m": _percentile(
                [row.final_pose_error_m for row in localization_rows],
                95,
            ),
            "planner_mean_wall_time_ms": {
                "grid_astar": _mean(
                    [row.wall_time_ms for row in planner_rows if row.planner == "grid_astar"]
                ),
                "hybrid_astar": _mean(
                    [row.wall_time_ms for row in planner_rows if row.planner == "hybrid_astar"]
                ),
            },
            "slam_mean_landmarks": _mean([row.landmark_count for row in slam_rows]),
            "stress_modes": [asdict(row) for row in stress_rows],
        },
        "outputs": {name: str(path) for name, path in outputs.items()},
    }


def _mean(values: list[float] | list[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile_value / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce TerraScout demo artifacts and benchmarks.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-gif", action="store_true", help="Skip animated GIF generation.")
    args = parser.parse_args()

    summary = run_reproduce(
        artifacts_dir=args.artifacts_dir,
        seed=args.seed,
        generate_gif=not args.skip_gif,
    )
    benchmark_summary = summary["benchmark_summary"]
    print(
        "reproduced "
        f"success={benchmark_summary['mission_mean_success_rate']:.3f} "
        f"collisions={benchmark_summary['mission_total_collisions']} "
        f"artifacts={summary['artifacts_dir']}"
    )


if __name__ == "__main__":
    main()
