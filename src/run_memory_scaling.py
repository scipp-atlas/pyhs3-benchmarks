from __future__ import annotations

import argparse
import math
import traceback
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_MODE,
    DEFAULT_N_RUNS,
    DEFAULT_TARGET,
    DEFAULT_WORKSPACE,
    PLOTS_DIR,
    RESULTS_DIR,
)
from .benchmark_stages import (
    DEFAULT_DISTRIBUTION,
    DEFAULT_N_EVALUATIONS,
    DEFAULT_N_SCAN_POINTS,
    DEFAULT_SCAN_MAX,
    DEFAULT_SCAN_MIN,
    DEFAULT_SCAN_PARAMETER,
    WORKFLOW_STAGES,
    build_stage_specs,
    resolve_stages,
    run_stage_isolated,
)
from .utils import (
    make_bar_plot,
    save_json,
    should_plot_metric,
)


BENCHMARK_NAME = "memory_scaling"

DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "memory_scaling_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME

RSS_KEYS = [
    "current_rss_before_mb",
    "current_rss_after_mb",
    "current_rss_delta_mb",
    "peak_rss_before_mb",
    "peak_rss_after_mb",
    "peak_rss_delta_mb",
]


def validate_workspace_path(workspace_path: Path) -> Path:
    """
    Validate that the workspace path points to an existing JSON file.
    """

    if not workspace_path.exists():
        raise FileNotFoundError(f"Workspace file does not exist: {workspace_path}")

    if not workspace_path.is_file():
        raise FileNotFoundError(f"Workspace path is not a file: {workspace_path}")

    return workspace_path


def validate_benchmark_config(
    target: str,
    mode: str,
    n_runs: int,
    n_evaluations: int,
    distribution: str,
    scan_parameter: str,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
) -> None:
    """
    Validate benchmark configuration before running workflow stages.
    """

    if not target:
        raise ValueError("target must be a non-empty string")

    if not mode:
        raise ValueError("mode must be a non-empty string")

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    if n_evaluations < 1:
        raise ValueError("n_evaluations must be at least 1")

    if not distribution:
        raise ValueError("distribution must be a non-empty string")

    if not scan_parameter:
        raise ValueError("scan_parameter must be a non-empty string")

    if not math.isfinite(scan_min):
        raise ValueError("scan_min must be finite")

    if not math.isfinite(scan_max):
        raise ValueError("scan_max must be finite")

    if scan_min >= scan_max:
        raise ValueError("scan_min must be smaller than scan_max")

    if n_scan_points < 2:
        raise ValueError("n_scan_points must be at least 2")


def verify_output_file(output_path: Path) -> None:
    """
    Verify that save_json created a regular output file.
    """

    if not output_path.exists():
        raise FileNotFoundError(f"Benchmark output file was not created: {output_path}")

    if not output_path.is_file():
        raise FileNotFoundError(f"Benchmark output path is not a file: {output_path}")


def extract_stage_memory(
    result: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    """
    Extract common RSS fields from a stage benchmark result.
    """

    record: dict[str, Any] = {
        "stage": stage,
        "status": result["status"],
        "benchmark": result["benchmark"],
    }

    for key in RSS_KEYS:
        record[key] = result.get(key)

    optional_keys = [
        "wall_time_seconds_mean",
        "wall_time_seconds_median",
        "wall_time_seconds_std",
        "average_runtime_seconds_per_evaluation",
        "throughput_evaluations_per_second",
        "runtime_per_scan_point_seconds",
        "throughput_scan_points_per_second",
        "total_runtime_seconds",
    ]

    for key in optional_keys:
        if key in result:
            record[key] = result[key]

    return record


def validate_stage_records(
    stage_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Validate that all selected stages succeeded and reported RSS fields.
    """

    missing_fields = []

    for record in stage_records:
        for key in RSS_KEYS:
            if record.get(key) is None:
                missing_fields.append(
                    {
                        "stage": record["stage"],
                        "missing_key": key,
                    }
                )

    return {
        "n_stages": len(stage_records),
        "all_stages_successful": all(
            record["status"] == "success" for record in stage_records
        ),
        "all_rss_fields_present": len(missing_fields) == 0,
        "missing_rss_fields": missing_fields,
    }


def summarize_memory(
    stage_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Summarize RSS usage across isolated workflow stages.
    """

    current_deltas = [
        record["current_rss_delta_mb"]
        for record in stage_records
        if record.get("current_rss_delta_mb") is not None
    ]

    peak_deltas = [
        record["peak_rss_delta_mb"]
        for record in stage_records
        if record.get("peak_rss_delta_mb") is not None
    ]

    peak_after_values = [
        record["peak_rss_after_mb"]
        for record in stage_records
        if record.get("peak_rss_after_mb") is not None
    ]

    return {
        "total_current_rss_delta_mb": sum(current_deltas),
        "total_peak_rss_delta_mb": sum(peak_deltas),
        "max_peak_rss_after_mb": max(peak_after_values) if peak_after_values else None,
    }


def run_single_benchmark(
    workspace_path: Path,
    target: str,
    mode: str,
    n_runs: int,
    n_evaluations: int,
    stages: list[str],
    distribution: str,
    scan_parameter: str,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
) -> dict[str, Any]:
    """
    Run RSS / memory scaling benchmark for one workspace, target, and mode.
    """

    validate_benchmark_config(
        target=target,
        mode=mode,
        n_runs=n_runs,
        n_evaluations=n_evaluations,
        distribution=distribution,
        scan_parameter=scan_parameter,
        scan_min=scan_min,
        scan_max=scan_max,
        n_scan_points=n_scan_points,
    )
    workspace_path = validate_workspace_path(workspace_path)

    selected_stages = resolve_stages(stages)

    stage_specs = build_stage_specs(
        selected_stages=selected_stages,
        workspace_path=workspace_path,
        target=target,
        mode=mode,
        n_runs=n_runs,
        n_evaluations=n_evaluations,
        distribution=distribution,
        scan_parameter=scan_parameter,
        scan_min=scan_min,
        scan_max=scan_max,
        n_scan_points=n_scan_points,
    )

    stage_results: dict[str, dict[str, Any]] = {}
    stage_records: list[dict[str, Any]] = []

    for stage_name, function, args in stage_specs:
        result = run_stage_isolated(function, args)
        stage_results[stage_name] = result
        stage_records.append(
            extract_stage_memory(
                result=result,
                stage=stage_name,
            )
        )

    validation_summary = validate_stage_records(stage_records)
    memory_summary = summarize_memory(stage_records)

    return {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "target": target,
        "mode": mode,
        "n_runs": n_runs,
        "n_evaluations": n_evaluations,
        "distribution": distribution,
        "scan_parameter": scan_parameter,
        "scan_min": scan_min,
        "scan_max": scan_max,
        "n_scan_points": n_scan_points,
        "selected_stages": selected_stages,
        "stages": stage_records,
        "stage_results": stage_results,
        **memory_summary,
        **validation_summary,
        "status": (
            "success"
            if validation_summary["all_stages_successful"]
            and validation_summary["all_rss_fields_present"]
            else "failed"
        ),
    }


def make_error_result(
    workspace_path: Path,
    target: str,
    mode: str,
    n_runs: int,
    n_evaluations: int,
    stages: list[str],
    distribution: str,
    scan_parameter: str,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
    exc: Exception,
) -> dict[str, Any]:
    """
    Build a structured benchmark result for a failed memory-scaling run.
    """

    selected_stages = list(stages)

    return {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "target": target,
        "mode": mode,
        "n_runs": n_runs,
        "n_evaluations": n_evaluations,
        "distribution": distribution,
        "scan_parameter": scan_parameter,
        "scan_min": scan_min,
        "scan_max": scan_max,
        "n_scan_points": n_scan_points,
        "selected_stages": selected_stages,
        "stages": [],
        "stage_results": {},
        "n_stages": len(selected_stages),
        "all_stages_successful": False,
        "all_rss_fields_present": False,
        "missing_rss_fields": [],
        "total_current_rss_delta_mb": 0.0,
        "total_peak_rss_delta_mb": 0.0,
        "max_peak_rss_after_mb": None,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def print_failed_result(result: dict[str, Any]) -> None:
    """
    Print a compact summary for a failed memory-scaling run.
    """

    print()
    print("=" * 72)
    print("RSS / memory scaling benchmark FAILED")
    print("=" * 72)
    print(f"Workspace: {result['workspace']}")
    print(f"Target:    {result['target']}")
    print(f"Mode:      {result['mode']}")
    print(f"Stages:    {', '.join(result['selected_stages'])}")
    print(
        "Reason:    "
        f"{result.get('error_type', 'UnknownError')}: "
        f"{result.get('error_message', '')}"
    )


def make_plot_records(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Make a list of records for plotting from the benchmark results.
    Each record is a dictionary with keys:
        - plot_label: str
        - workspace: str
        - stage: str
        - current_rss_delta_mb: float
        - peak_rss_delta_mb: float
        - peak_rss_after_mb: float
    """

    records = []

    for result in results:
        if result.get("status") != "success":
            continue

        for stage in result.get("stages", []):
            if any(stage.get(key) is None for key in RSS_KEYS):
                continue

            records.append(
                {
                    "plot_label": (f"{result['workspace']}\n{stage['stage']}"),
                    "workspace": result["workspace"],
                    "stage": stage["stage"],
                    "current_rss_delta_mb": stage["current_rss_delta_mb"],
                    "peak_rss_delta_mb": stage["peak_rss_delta_mb"],
                    "peak_rss_after_mb": stage["peak_rss_after_mb"],
                }
            )

    return records


def make_plots(
    results: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    """
    Make plots for the benchmark results.
    """

    plot_dir.mkdir(parents=True, exist_ok=True)

    records = make_plot_records(results)

    if not records:
        print("Skipping plots: no successful stage records to plot.")
        return

    if should_plot_metric(records, "current_rss_delta_mb"):
        make_bar_plot(
            results=records,
            output_path=plot_dir / "memory_scaling_current_rss_delta.png",
            title="Current RSS delta by workflow stage",
            metric_key="current_rss_delta_mb",
            metric_label="Current RSS delta [MB]",
        )

    if should_plot_metric(records, "peak_rss_delta_mb"):
        make_bar_plot(
            results=records,
            output_path=plot_dir / "memory_scaling_peak_rss_delta.png",
            title="Peak RSS delta by workflow stage",
            metric_key="peak_rss_delta_mb",
            metric_label="Peak RSS delta [MB]",
        )

    make_bar_plot(
        results=records,
        output_path=plot_dir / "memory_scaling_peak_rss_after.png",
        title="Peak RSS after workflow stage",
        metric_key="peak_rss_after_mb",
        metric_label="Peak RSS after stage [MB]",
    )


def print_result(result: dict[str, Any]) -> None:
    if result.get("status") != "success":
        print_failed_result(result)
        return

    print()
    print("=" * 72)
    print("RSS / memory scaling benchmark")
    print("=" * 72)
    print(f"Workspace:    {result['workspace']}")
    print(f"Target:       {result['target']}")
    print(f"Mode:         {result['mode']}")
    print(f"Stages:       {', '.join(result['selected_stages'])}")
    print(f"Status:       {result['status']}")

    print()
    print("Stages")
    for stage in result["stages"]:
        print()
        print(f"  {stage['stage']}")
        print(f"    status:              {stage['status']}")
        print(f"    current RSS before:  {stage['current_rss_before_mb']:.3f} MB")
        print(f"    current RSS after:   {stage['current_rss_after_mb']:.3f} MB")
        print(f"    current RSS delta:   {stage['current_rss_delta_mb']:.3f} MB")
        print(f"    peak RSS before:     {stage['peak_rss_before_mb']:.3f} MB")
        print(f"    peak RSS after:      {stage['peak_rss_after_mb']:.3f} MB")
        print(f"    peak RSS delta:      {stage['peak_rss_delta_mb']:.3f} MB")

    print()
    print("Summary")
    print(f"  total current RSS delta: {result['total_current_rss_delta_mb']:.3f} MB")
    print(f"  total peak RSS delta:    {result['total_peak_rss_delta_mb']:.3f} MB")
    print(f"  max peak RSS after:      {result['max_peak_rss_after_mb']:.3f} MB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RSS / memory scaling across PyHS3 workflow stages."
    )

    parser.add_argument(
        "--workspaces",
        nargs="+",
        type=Path,
        default=[DEFAULT_WORKSPACE],
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[DEFAULT_TARGET],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[DEFAULT_MODE],
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", *WORKFLOW_STAGES],
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
    )
    parser.add_argument(
        "--n-evaluations",
        type=int,
        default=DEFAULT_N_EVALUATIONS,
    )
    parser.add_argument(
        "--distribution",
        default=DEFAULT_DISTRIBUTION,
    )
    parser.add_argument(
        "--scan-parameter",
        default=DEFAULT_SCAN_PARAMETER,
    )
    parser.add_argument(
        "--scan-min",
        type=float,
        default=DEFAULT_SCAN_MIN,
    )
    parser.add_argument(
        "--scan-max",
        type=float,
        default=DEFAULT_SCAN_MAX,
    )
    parser.add_argument(
        "--n-scan-points",
        type=int,
        default=DEFAULT_N_SCAN_POINTS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.targets:
        raise ValueError("--targets must contain at least one value")

    if not args.modes:
        raise ValueError("--modes must contain at least one value")

    validate_benchmark_config(
        target=args.targets[0],
        mode=args.modes[0],
        n_runs=args.n_runs,
        n_evaluations=args.n_evaluations,
        distribution=args.distribution,
        scan_parameter=args.scan_parameter,
        scan_min=args.scan_min,
        scan_max=args.scan_max,
        n_scan_points=args.n_scan_points,
    )

    for workspace_path in args.workspaces:
        validate_workspace_path(workspace_path)

    for target in args.targets:
        if not target:
            raise ValueError("--targets must contain only non-empty strings")

    for mode in args.modes:
        if not mode:
            raise ValueError("--modes must contain only non-empty strings")

    selected_stages = resolve_stages(args.stages)

    results = []

    for workspace_path in args.workspaces:
        for target in args.targets:
            for mode in args.modes:
                print(
                    f"Running {workspace_path.name}, "
                    f"target={target}, "
                    f"mode={mode}, "
                    f"stages={selected_stages}",
                    flush=True,
                )

                try:
                    result = run_single_benchmark(
                        workspace_path=workspace_path,
                        target=target,
                        mode=mode,
                        n_runs=args.n_runs,
                        n_evaluations=args.n_evaluations,
                        stages=selected_stages,
                        distribution=args.distribution,
                        scan_parameter=args.scan_parameter,
                        scan_min=args.scan_min,
                        scan_max=args.scan_max,
                        n_scan_points=args.n_scan_points,
                    )
                except Exception as exc:
                    result = make_error_result(
                        workspace_path=workspace_path,
                        target=target,
                        mode=mode,
                        n_runs=args.n_runs,
                        n_evaluations=args.n_evaluations,
                        stages=selected_stages,
                        distribution=args.distribution,
                        scan_parameter=args.scan_parameter,
                        scan_min=args.scan_min,
                        scan_max=args.scan_max,
                        n_scan_points=args.n_scan_points,
                        exc=exc,
                    )

                results.append(result)
                print_result(result)

    output_data: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "description": "RSS / memory scaling across PyHS3 workflow stages.",
        "available_stages": WORKFLOW_STAGES,
        "selected_stages": selected_stages,
        "metrics": RSS_KEYS,
        "n_results": len(results),
        "results": results,
    }

    output_path = args.output_dir / args.output_name
    save_json(output_data, output_path)
    verify_output_file(output_path)

    print()
    print(f"Saved result to {output_path}")

    if args.plot:
        make_plots(
            results=results,
            plot_dir=args.plot_dir,
        )
        print(f"Saved plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
