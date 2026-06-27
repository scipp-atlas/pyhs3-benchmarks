from __future__ import annotations

import argparse
import csv
import math
import traceback
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_MODE,
    DEFAULT_N_RUNS,
    DEFAULT_TARGET,
    DEFAULT_WORKSPACE,
    PLOTS_DIR,
    REPORTS_DIR,
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
)
from .utils import (
    make_bar_plot,
    save_json,
    should_plot_metric,
)

BENCHMARK_NAME = "model_complexity_scaling"

DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "model_complexity_scaling_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME
DEFAULT_REPORT_DIR = REPORTS_DIR / BENCHMARK_NAME
DEFAULT_CSV_NAME = "model_complexity_scaling_summary.csv"

SETUP_STAGES = [
    "workspace_loading",
    "model_creation",
    "log_prob_construction",
    "log_prob_compilation",
]


def validate_workspace_path(workspace_path: Path) -> Path:
    """Validate that the workspace path points to an existing file."""

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
    """Validate benchmark configuration before running expensive work."""

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
    """Verify that an output writer created a regular file."""

    if not output_path.exists():
        raise FileNotFoundError(f"Output file was not created: {output_path}")

    if not output_path.is_file():
        raise FileNotFoundError(f"Output path is not a file: {output_path}")


def make_stage_error_result(
    stage_name: str,
    exc: Exception,
) -> dict[str, Any]:
    """Build a structured failed result for one workflow stage."""

    return {
        "benchmark": stage_name,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
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
    """Build a structured failed result for one scaling configuration."""

    workspace_path = Path(workspace_path)

    workspace_size_bytes: int | None
    try:
        workspace_size_bytes = workspace_path.stat().st_size
    except OSError:
        workspace_size_bytes = None

    return {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "workspace_size_bytes": workspace_size_bytes,
        "target": target,
        "mode": mode,
        "n_runs": n_runs,
        "n_evaluations": n_evaluations,
        "distribution": distribution,
        "scan_parameter": scan_parameter,
        "scan_min": scan_min,
        "scan_max": scan_max,
        "n_scan_points": n_scan_points,
        "selected_stages": stages,
        "stage_results": {},
        "total_setup_time_seconds": 0.0,
        "total_peak_rss_delta_mb": 0.0,
        "quickfit_reference_available": False,
        "quickfit_validation_status": "not_run",
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def print_failed_result(result: dict[str, Any]) -> None:
    """Print a readable summary for a failed scaling row."""

    print()
    print("=" * 72)
    print("Model complexity scaling benchmark FAILED")
    print("=" * 72)
    print(f"Workspace: {result.get('workspace')}")
    print(f"Target:    {result.get('target')}")
    print(f"Mode:      {result.get('mode')}")
    print(
        "Reason:    "
        f"{result.get('error_type', 'UnknownError')}: "
        f"{result.get('error_message', '')}"
    )


def summarize_stage(
    result: dict[str, Any],
    stage_name: str,
) -> dict[str, Any]:
    """
    Extract common timing and memory information from one benchmark stage.
    """

    summary: dict[str, Any] = {
        f"{stage_name}_status": result["status"],
        f"{stage_name}_current_rss_delta_mb": result.get(
            "current_rss_delta_mb",
            0.0,
        ),
        f"{stage_name}_peak_rss_delta_mb": result.get(
            "peak_rss_delta_mb",
            0.0,
        ),
    }

    if "wall_time_seconds_mean" in result:
        summary[f"{stage_name}_wall_time_seconds_mean"] = result[
            "wall_time_seconds_mean"
        ]
        summary[f"{stage_name}_wall_time_seconds_std"] = result.get(
            "wall_time_seconds_std",
            0.0,
        )

    if "average_runtime_seconds_per_evaluation" in result:
        summary[f"{stage_name}_average_runtime_seconds_per_evaluation"] = result[
            "average_runtime_seconds_per_evaluation"
        ]
        summary[f"{stage_name}_throughput_evaluations_per_second"] = result[
            "throughput_evaluations_per_second"
        ]

    if "runtime_per_scan_point_seconds" in result:
        summary[f"{stage_name}_runtime_per_scan_point_seconds"] = result[
            "runtime_per_scan_point_seconds"
        ]
        summary[f"{stage_name}_throughput_scan_points_per_second"] = result[
            "throughput_scan_points_per_second"
        ]

    if "total_runtime_seconds" in result:
        summary[f"{stage_name}_total_runtime_seconds"] = result[
            "total_runtime_seconds"
        ]

    return summary


def run_single_scaling_benchmark(
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
    Run selected benchmark stages for one workspace and collect one scaling row.
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

    row: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "workspace_size_bytes": workspace_path.stat().st_size,
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
        "quickfit_reference_available": False,
        "quickfit_validation_status": "not_run",
    }

    stage_results: dict[str, dict[str, Any]] = {}

    for stage_name, function, args in stage_specs:
        try:
            result = function(*args)
        except Exception as exc:
            result = make_stage_error_result(
                stage_name=stage_name,
                exc=exc,
            )

        stage_results[stage_name] = result
        row.update(
            summarize_stage(
                result=result,
                stage_name=stage_name,
            )
        )

        if stage_name == "compiled_evaluation" and result.get("status") == "success":
            row["compiled_evaluation_reference_output"] = result[
                "reference_output"
            ]
            row["compiled_evaluation_all_outputs_finite"] = result[
                "all_outputs_finite"
            ]

        if stage_name == "nll_scan":
            row["nll_scan_minimum_scan_value"] = result[
                "minimum_scan_value"
            ]
            row["nll_scan_minimum_nll_value"] = result[
                "minimum_nll_value"
            ]
            row["nll_scan_all_values_finite"] = result[
                "all_nll_values_finite"
            ]

    row["stage_results"] = stage_results

    row["total_setup_time_seconds"] = sum(
        row.get(f"{stage}_wall_time_seconds_mean", 0.0)
        for stage in SETUP_STAGES
    )

    row["total_peak_rss_delta_mb"] = sum(
        row.get(f"{stage}_peak_rss_delta_mb", 0.0)
        for stage in selected_stages
    )

    row["status"] = (
        "success"
        if all(
            row.get(f"{stage}_status") == "success"
            for stage in selected_stages
        )
        else "failed"
    )

    return row


def print_result(result: dict[str, Any]) -> None:
    """
    Print one scaling summary row.
    """

    print()
    print("=" * 72)
    print("Model complexity scaling benchmark")
    print("=" * 72)
    print(f"Workspace:        {result['workspace']}")
    print(f"Size:             {result['workspace_size_bytes']} bytes")
    print(f"Target:           {result['target']}")
    print(f"Mode:             {result['mode']}")
    print(f"Stages:           {', '.join(result['selected_stages'])}")
    print(f"Status:           {result['status']}")

    print()
    print("Timing")

    if "workspace_loading_wall_time_seconds_mean" in result:
        print(
            "  workspace load:        "
            f"{result['workspace_loading_wall_time_seconds_mean'] * 1000:.3f} ms"
        )

    if "model_creation_wall_time_seconds_mean" in result:
        print(
            "  model creation:        "
            f"{result['model_creation_wall_time_seconds_mean'] * 1000:.3f} ms"
        )

    if "log_prob_construction_wall_time_seconds_mean" in result:
        print(
            "  log_prob construction: "
            f"{result['log_prob_construction_wall_time_seconds_mean'] * 1000:.3f} ms"
        )

    if "log_prob_compilation_wall_time_seconds_mean" in result:
        print(
            "  log_prob compilation:  "
            f"{result['log_prob_compilation_wall_time_seconds_mean'] * 1000:.3f} ms"
        )

    print(
        "  total setup:           "
        f"{result['total_setup_time_seconds'] * 1000:.3f} ms"
    )

    if "compiled_evaluation_average_runtime_seconds_per_evaluation" in result:
        print(
            "  compiled eval / point: "
            f"{result['compiled_evaluation_average_runtime_seconds_per_evaluation'] * 1000:.6f} ms"
        )

    if "pdf_evaluation_average_runtime_seconds_per_evaluation" in result:
        print(
            "  PDF eval / point:      "
            f"{result['pdf_evaluation_average_runtime_seconds_per_evaluation'] * 1000:.6f} ms"
        )

    if "nll_scan_runtime_per_scan_point_seconds" in result:
        print(
            "  NLL scan / point:      "
            f"{result['nll_scan_runtime_per_scan_point_seconds'] * 1000:.6f} ms"
        )

    print()
    print("Memory")
    print(
        "  total peak RSS delta:  "
        f"{result['total_peak_rss_delta_mb']:.3f} MB"
    )

    print()
    print("Validation")

    if "compiled_evaluation_all_outputs_finite" in result:
        print(
            "  compiled output finite: "
            f"{result['compiled_evaluation_all_outputs_finite']}"
        )

    if "nll_scan_all_values_finite" in result:
        print(
            "  NLL values finite:      "
            f"{result['nll_scan_all_values_finite']}"
        )
        print(
            "  NLL minimum at:         "
            f"{result['nll_scan_minimum_scan_value']}"
        )


def write_summary_csv(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Write a CSV summary table using all keys observed in the result rows.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = sorted(
        {
            key
            for row in results
            for key in row.keys()
            if key != "stage_results"
        }
    )

    with output_path.open(
        "w",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in results:
            writer.writerow(
                {
                    key: value
                    for key, value in row.items()
                    if key != "stage_results"
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyHS3 scaling with workspace/model complexity."
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
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )
    parser.add_argument(
        "--csv-name",
        default=DEFAULT_CSV_NAME,
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


def make_plots(
    results: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    """
    Create scaling plots.
    """

    successful_results = [
        result
        for result in results
        if result.get("status") == "success"
    ]

    if len(successful_results) < 2:
        print("Skipping plots: at least two successful result entries are needed.")
        return

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_results = []

    for result in successful_results:
        plot_result = dict(result)

        plot_result["workspace_size_kb"] = (
            result["workspace_size_bytes"] / 1024.0
        )
        plot_result["total_setup_time_ms"] = (
            result["total_setup_time_seconds"] * 1000.0
        )

        if "compiled_evaluation_average_runtime_seconds_per_evaluation" in result:
            plot_result["compiled_evaluation_ms_per_eval"] = (
                result[
                    "compiled_evaluation_average_runtime_seconds_per_evaluation"
                ]
                * 1000.0
            )

        if "pdf_evaluation_average_runtime_seconds_per_evaluation" in result:
            plot_result["pdf_evaluation_ms_per_eval"] = (
                result[
                    "pdf_evaluation_average_runtime_seconds_per_evaluation"
                ]
                * 1000.0
            )

        if "nll_scan_runtime_per_scan_point_seconds" in result:
            plot_result["nll_scan_ms_per_point"] = (
                result["nll_scan_runtime_per_scan_point_seconds"] * 1000.0
            )

        plot_results.append(plot_result)

    make_bar_plot(
        results=plot_results,
        output_path=plot_dir / "model_complexity_total_setup_time.png",
        title="Model complexity scaling: total setup time",
        metric_key="total_setup_time_ms",
        metric_label="Total setup time [ms]",
    )

    if should_plot_metric(plot_results, "compiled_evaluation_ms_per_eval"):
        make_bar_plot(
            results=plot_results,
            output_path=plot_dir / "model_complexity_compiled_evaluation_time.png",
            title="Model complexity scaling: compiled evaluation time",
            metric_key="compiled_evaluation_ms_per_eval",
            metric_label="Compiled evaluation time per eval [ms]",
        )

    if should_plot_metric(plot_results, "pdf_evaluation_ms_per_eval"):
        make_bar_plot(
            results=plot_results,
            output_path=plot_dir / "model_complexity_pdf_evaluation_time.png",
            title="Model complexity scaling: PDF evaluation time",
            metric_key="pdf_evaluation_ms_per_eval",
            metric_label="PDF evaluation time per eval [ms]",
        )

    if should_plot_metric(plot_results, "nll_scan_ms_per_point"):
        make_bar_plot(
            results=plot_results,
            output_path=plot_dir / "model_complexity_nll_scan_time.png",
            title="Model complexity scaling: NLL scan time",
            metric_key="nll_scan_ms_per_point",
            metric_label="NLL scan time per point [ms]",
        )

    if should_plot_metric(plot_results, "total_peak_rss_delta_mb"):
        make_bar_plot(
            results=plot_results,
            output_path=plot_dir / "model_complexity_peak_rss_delta.png",
            title="Model complexity scaling: total peak RSS delta",
            metric_key="total_peak_rss_delta_mb",
            metric_label="Total peak RSS delta [MB]",
        )


def main() -> None:
    args = parse_args()

    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1")

    if args.n_evaluations < 1:
        raise ValueError("--n-evaluations must be at least 1")

    if args.n_scan_points < 2:
        raise ValueError("--n-scan-points must be at least 2")

    for workspace_path in args.workspaces:
        validate_workspace_path(workspace_path)

    for target in args.targets:
        if not target:
            raise ValueError("--targets must contain only non-empty strings")

    for mode in args.modes:
        if not mode:
            raise ValueError("--modes must contain only non-empty strings")

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

    selected_stages = resolve_stages(args.stages)

    results = []
    ctx = get_context("spawn")

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
                    with ctx.Pool(processes=1) as pool:
                        result = pool.apply(
                            run_single_scaling_benchmark,
                            args=(
                                workspace_path,
                                target,
                                mode,
                                args.n_runs,
                                args.n_evaluations,
                                selected_stages,
                                args.distribution,
                                args.scan_parameter,
                                args.scan_min,
                                args.scan_max,
                                args.n_scan_points,
                            ),
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

                if result.get("status") == "success":
                    print_result(result)
                else:
                    print_failed_result(result)

    output_data: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "available_stages": WORKFLOW_STAGES,
        "selected_stages": selected_stages,
        "n_results": len(results),
        "results": results,
    }

    output_path = args.output_dir / args.output_name
    save_json(
        output_data,
        output_path,
    )
    verify_output_file(output_path)

    csv_path = args.report_dir / args.csv_name
    write_summary_csv(
        results=results,
        output_path=csv_path,
    )
    verify_output_file(csv_path)

    print()
    print(f"Saved result to {output_path}")
    print(f"Saved CSV summary to {csv_path}")

    if args.plot:
        make_plots(
            results=results,
            plot_dir=args.plot_dir,
        )
        if len([result for result in results if result.get("status") == "success"]) >= 2:
            print(f"Saved plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
