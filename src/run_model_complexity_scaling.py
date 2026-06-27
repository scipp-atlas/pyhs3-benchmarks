from __future__ import annotations

import argparse
import csv
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
        result = function(*args)
        stage_results[stage_name] = result
        row.update(
            summarize_stage(
                result=result,
                stage_name=stage_name,
            )
        )

        if stage_name == "compiled_evaluation":
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

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_results = []

    for result in results:
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

                results.append(result)
                print_result(result)

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

    csv_path = args.report_dir / args.csv_name
    write_summary_csv(
        results=results,
        output_path=csv_path,
    )

    print()
    print(f"Saved result to {output_path}")
    print(f"Saved CSV summary to {csv_path}")

    if args.plot:
        if len(results) < 2:
            print("Skipping plots: at least two result entries are needed.")
        else:
            make_plots(
                results=results,
                plot_dir=args.plot_dir,
            )
            print(f"Saved plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
