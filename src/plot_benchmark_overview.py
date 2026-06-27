from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .config import PLOTS_DIR, RESULTS_DIR


BENCHMARK_NAME = "benchmark_overview"

DEFAULT_RESULTS_DIR = RESULTS_DIR
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME

AVAILABLE_PLOTS = [
    "status",
    "wall_time",
    "evaluation_time",
    "scan_time",
    "setup_time",
    "peak_rss",
    "total_peak_rss",
    "stage_timing",
    "stage_memory",
]

STAGE_TIME_KEYS = {
    "workspace_loading": "workspace_loading_wall_time_seconds_mean",
    "model_creation": "model_creation_wall_time_seconds_mean",
    "log_prob_construction": "log_prob_construction_wall_time_seconds_mean",
    "log_prob_compilation": "log_prob_compilation_wall_time_seconds_mean",
    "compiled_evaluation": (
        "compiled_evaluation_average_runtime_seconds_per_evaluation"
    ),
    "pdf_evaluation": (
        "pdf_evaluation_average_runtime_seconds_per_evaluation"
    ),
    "nll_scan": "nll_scan_runtime_per_scan_point_seconds",
}

STAGE_MEMORY_KEYS = {
    "workspace_loading": "workspace_loading_peak_rss_delta_mb",
    "model_creation": "model_creation_peak_rss_delta_mb",
    "log_prob_construction": "log_prob_construction_peak_rss_delta_mb",
    "log_prob_compilation": "log_prob_compilation_peak_rss_delta_mb",
    "compiled_evaluation": "compiled_evaluation_peak_rss_delta_mb",
    "pdf_evaluation": "pdf_evaluation_peak_rss_delta_mb",
    "nll_scan": "nll_scan_peak_rss_delta_mb",
}


def resolve_plots(plots: list[str]) -> list[str]:
    if "all" in plots:
        if len(plots) > 1:
            raise ValueError("--plots all cannot be combined with other plots")
        return AVAILABLE_PLOTS

    unknown_plots = [
        plot
        for plot in plots
        if plot not in AVAILABLE_PLOTS
    ]

    if unknown_plots:
        raise ValueError(
            f"Unknown plots: {unknown_plots}. "
            f"Available plots: {AVAILABLE_PLOTS}"
        )

    return plots


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def iter_result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/*_result.json"))


def normalize_benchmark_name(name: str) -> str:
    return name.replace("_", " ")


def get_workspace_label(result: dict[str, Any]) -> str:
    workspace = result.get("workspace", "unknown")
    target = result.get("target")

    if target:
        return f"{workspace}\n{target}"

    return str(workspace)


def extract_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results", [])

    if isinstance(results, list):
        return results

    return []


def maybe_to_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_overview_records(results_dir: Path) -> list[dict[str, Any]]:
    records = []

    for result_file in iter_result_files(results_dir):
        payload = load_json(result_file)
        benchmark = payload.get("benchmark", result_file.parent.name)

        for result in extract_results(payload):
            record: dict[str, Any] = {
                "benchmark": benchmark,
                "benchmark_label": normalize_benchmark_name(benchmark),
                "workspace": result.get("workspace"),
                "workspace_label": get_workspace_label(result),
                "target": result.get("target"),
                "mode": result.get("mode"),
                "source_file": str(result_file),
                "status": result.get("status", "unknown"),
                "n_runs": result.get("n_runs"),
                "n_evaluations": result.get("n_evaluations"),
                "n_scan_points": result.get("n_scan_points"),
            }

            metric_candidates = {
                "wall_time_seconds_mean": result.get(
                    "wall_time_seconds_mean"
                ),
                "average_runtime_seconds_per_evaluation": result.get(
                    "average_runtime_seconds_per_evaluation"
                ),
                "runtime_per_scan_point_seconds": result.get(
                    "runtime_per_scan_point_seconds"
                ),
                "total_runtime_seconds": result.get(
                    "total_runtime_seconds"
                ),
                "total_setup_time_seconds": result.get(
                    "total_setup_time_seconds"
                ),
                "current_rss_delta_mb": result.get(
                    "current_rss_delta_mb"
                ),
                "peak_rss_delta_mb": result.get(
                    "peak_rss_delta_mb"
                ),
                "total_peak_rss_delta_mb": result.get(
                    "total_peak_rss_delta_mb"
                ),
            }

            record.update(metric_candidates)

            for stage_name, key in STAGE_TIME_KEYS.items():
                value = result.get(key)
                if value is not None:
                    record[f"{stage_name}_time_ms"] = float(value) * 1000.0

            for stage_name, key in STAGE_MEMORY_KEYS.items():
                value = result.get(key)
                if value is not None:
                    record[f"{stage_name}_peak_rss_delta_mb"] = float(value)

            if record["wall_time_seconds_mean"] is not None:
                record["wall_time_ms"] = (
                    float(record["wall_time_seconds_mean"]) * 1000.0
                )

            if record["average_runtime_seconds_per_evaluation"] is not None:
                record["average_runtime_ms_per_evaluation"] = (
                    float(record["average_runtime_seconds_per_evaluation"])
                    * 1000.0
                )

            if record["runtime_per_scan_point_seconds"] is not None:
                record["runtime_ms_per_scan_point"] = (
                    float(record["runtime_per_scan_point_seconds"]) * 1000.0
                )

            if record["total_runtime_seconds"] is not None:
                record["total_runtime_ms"] = (
                    float(record["total_runtime_seconds"]) * 1000.0
                )

            if record["total_setup_time_seconds"] is not None:
                record["total_setup_time_ms"] = (
                    float(record["total_setup_time_seconds"]) * 1000.0
                )

            records.append(record)

    return records

def values_match(
    actual: Any,
    expected_values: list[str] | None,
) -> bool:
    if not expected_values:
        return True

    if actual is None:
        return False

    return str(actual) in expected_values


def numeric_value_matches(
    actual: Any,
    expected_values: list[int] | None,
) -> bool:
    if not expected_values:
        return True

    if actual is None:
        return False

    try:
        return int(actual) in expected_values
    except (TypeError, ValueError):
        return False


def filter_records(
    records: list[dict[str, Any]],
    benchmarks: list[str] | None,
    workspaces: list[str] | None,
    targets: list[str] | None,
    modes: list[str] | None,
    n_runs: list[int] | None,
    n_evaluations: list[int] | None,
    n_scan_points: list[int] | None,
    successful_only: bool,
) -> list[dict[str, Any]]:
    filtered = []

    for record in records:
        if successful_only and record.get("status") != "success":
            continue

        if not values_match(record.get("benchmark"), benchmarks):
            continue

        if not values_match(record.get("workspace"), workspaces):
            continue

        if not values_match(record.get("target"), targets):
            continue

        if not values_match(record.get("mode"), modes):
            continue

        if not numeric_value_matches(record.get("n_runs"), n_runs):
            continue

        if not numeric_value_matches(
            record.get("n_evaluations"),
            n_evaluations,
        ):
            continue

        if not numeric_value_matches(
            record.get("n_scan_points"),
            n_scan_points,
        ):
            continue

        filtered.append(record)

    return filtered


def compact_workspace_name(name: str | None) -> str:
    if not name:
        return "unknown"

    return (
        name.replace(".json", "")
        .replace("simple_workspace_", "")
        .replace("simple_workspace", "simple")
        .replace("normal_pdf_workspace", "normal")
        .replace("poisson_pdf_workspace", "poisson")
        .replace("exponential_pdf_workspace", "exponential")
    )


def make_label(record: dict[str, Any]) -> str:
    benchmark = normalize_benchmark_name(str(record["benchmark"]))
    workspace = compact_workspace_name(record.get("workspace"))

    return f"{benchmark}\n{workspace}"


def should_plot(records: list[dict[str, Any]], metric_key: str) -> bool:
    values = [
        maybe_to_float(record.get(metric_key))
        for record in records
    ]

    values = [
        value
        for value in values
        if value is not None
    ]

    return len(values) > 0 and any(value != 0.0 for value in values)


def make_simple_bar_plot(
    records: list[dict[str, Any]],
    output_path: Path,
    title: str,
    metric_key: str,
    metric_label: str,
) -> None:
    plot_records = [
        record
        for record in records
        if maybe_to_float(record.get(metric_key)) is not None
    ]

    if not should_plot(plot_records, metric_key):
        return

    labels = [
        make_label(record)
        for record in plot_records
    ]
    values = [
        float(record[metric_key])
        for record in plot_records
    ]

    fig_width = max(10.0, 0.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    x = np.arange(len(labels))
    ax.bar(x, values)

    ax.set_title(title)
    ax.set_ylabel(metric_label)
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_status_plot(
    records: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    status_counts: dict[str, int] = {}

    for record in records:
        key = f"{record['benchmark']}\n{record['status']}"
        status_counts[key] = status_counts.get(key, 0) + 1

    plot_records = [
        {
            "benchmark": key,
            "status_count": count,
        }
        for key, count in sorted(status_counts.items())
    ]

    if not plot_records:
        return

    labels = [
        record["benchmark"]
        for record in plot_records
    ]
    values = [
        record["status_count"]
        for record in plot_records
    ]

    fig_width = max(10.0, 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    x = np.arange(len(labels))
    ax.bar(x, values)

    ax.set_title("Benchmark overview: status counts")
    ax.set_ylabel("Result count")
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        plot_dir / "benchmark_overview_status.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

def format_segment_label(value: float, total: float, unit: str) -> str:
    if value == 0.0 or total == 0.0:
        return ""

    percentage = value / total * 100.0

    if percentage < 3.0:
        return ""

    return f"{value:.1f}{unit}\n{percentage:.0f}%"

def make_stage_timing_plot(
    records: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    plot_records = [
        record
        for record in records
        if record.get("benchmark") in {
            "model_complexity_scaling",
            "memory_scaling",
        }
    ]

    stage_names = list(STAGE_TIME_KEYS)

    rows = []
    for record in plot_records:
        values = [
            maybe_to_float(record.get(f"{stage}_time_ms")) or 0.0
            for stage in stage_names
        ]

        if any(value != 0.0 for value in values):
            rows.append(
                {
                    "label": compact_workspace_name(record.get("workspace")),
                    "values": values,
                }
            )

    if not rows:
        return

    labels = [row["label"] for row in rows]

    fig_width = max(11.0, 0.9 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    totals = np.array([sum(row["values"]) for row in rows])

    for stage_index, stage in enumerate(stage_names):
        values = np.array([row["values"][stage_index] for row in rows])

        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=stage,
        )

        for bar_index, bar in enumerate(bars):
            label = format_segment_label(
                value=values[bar_index],
                total=totals[bar_index],
                unit="ms",
            )

            if label:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[bar_index] + values[bar_index] / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        bottom += values

    for index, total in enumerate(totals):
        ax.text(
            x[index],
            total,
            f"{total:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Stacked PyHS3 stage timing overview")
    ax.set_ylabel("Time [ms]")
    ax.set_xlabel("Workspace")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        plot_dir / "benchmark_overview_stage_timing.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_stage_memory_plot(
    records: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    plot_records = [
        record
        for record in records
        if record.get("benchmark") in {
            "model_complexity_scaling",
            "memory_scaling",
        }
    ]

    stage_names = list(STAGE_MEMORY_KEYS)

    rows = []
    for record in plot_records:
        values = [
            maybe_to_float(record.get(f"{stage}_peak_rss_delta_mb")) or 0.0
            for stage in stage_names
        ]

        if any(value != 0.0 for value in values):
            rows.append(
                {
                    "label": compact_workspace_name(record.get("workspace")),
                    "values": values,
                }
            )

    if not rows:
        return

    labels = [row["label"] for row in rows]

    fig_width = max(11.0, 0.9 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    totals = np.array([sum(row["values"]) for row in rows])

    for stage_index, stage in enumerate(stage_names):
        values = np.array([row["values"][stage_index] for row in rows])

        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=stage,
        )

        for bar_index, bar in enumerate(bars):
            label = format_segment_label(
                value=values[bar_index],
                total=totals[bar_index],
                unit="MB",
            )

            if label:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[bar_index] + values[bar_index] / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        bottom += values

    for index, total in enumerate(totals):
        ax.text(
            x[index],
            total,
            f"{total:.1f} MB",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Stacked PyHS3 stage memory overview")
    ax.set_ylabel("Peak RSS delta [MB]")
    ax.set_xlabel("Workspace")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        plot_dir / "benchmark_overview_stage_memory.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create overview plots from benchmark result files."
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
    )

    parser.add_argument(
        "--plots",
        nargs="+",
        default=["all"],
        choices=["all", *AVAILABLE_PLOTS],
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
    )

    parser.add_argument(
        "--workspaces",
        nargs="+",
    )

    parser.add_argument(
        "--targets",
        nargs="+",
    )

    parser.add_argument(
        "--modes",
        nargs="+",
    )

    parser.add_argument(
        "--n-runs",
        nargs="+",
        type=int,
    )

    parser.add_argument(
        "--n-evaluations",
        nargs="+",
        type=int,
    )

    parser.add_argument(
        "--n-scan-points",
        nargs="+",
        type=int,
    )

    parser.add_argument(
        "--include-failed",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = collect_overview_records(
        args.results_dir,
    )

    if not records:
        raise ValueError(
            f"No benchmark results found in {args.results_dir}"
        )

    records = filter_records(
        records=records,
        benchmarks=args.benchmarks,
        workspaces=args.workspaces,
        targets=args.targets,
        modes=args.modes,
        n_runs=args.n_runs,
        n_evaluations=args.n_evaluations,
        n_scan_points=args.n_scan_points,
        successful_only=not args.include_failed,
    )

    if not records:
        raise ValueError(
            "No benchmark results remain after applying filters."
        )

    plot_dir = args.plot_dir
    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_plots = resolve_plots(args.plots)

    if "status" in selected_plots:
        make_status_plot(
            records,
            plot_dir,
        )

    if "wall_time" in selected_plots:
        make_simple_bar_plot(
            records,
            plot_dir / "benchmark_overview_wall_time.png",
            "Benchmark overview: wall time",
            "wall_time_ms",
            "Wall time [ms]",
        )

    if "evaluation_time" in selected_plots:
        make_simple_bar_plot(
            records,
            plot_dir / "benchmark_overview_average_evaluation_time.png",
            "Benchmark overview: evaluation time",
            "average_runtime_ms_per_evaluation",
            "Average evaluation time [ms]",
        )

    if "scan_time" in selected_plots:
        make_simple_bar_plot(
            records,
            plot_dir / "benchmark_overview_scan_time.png",
            "Benchmark overview: scan time",
            "runtime_ms_per_scan_point",
            "Runtime per scan point [ms]",
        )

    if "setup_time" in selected_plots:
        make_simple_bar_plot(
            records,
            plot_dir / "benchmark_overview_setup_time.png",
            "Benchmark overview: setup time",
            "total_setup_time_ms",
            "Setup time [ms]",
        )

    if "peak_rss" in selected_plots:
        make_simple_bar_plot(
            records,
            plot_dir / "benchmark_overview_peak_rss_delta.png",
            "Benchmark overview: peak RSS delta",
            "peak_rss_delta_mb",
            "Peak RSS delta [MB]",
        )

    if "total_peak_rss" in selected_plots:
        make_simple_bar_plot(
            records,
            plot_dir / "benchmark_overview_total_peak_rss_delta.png",
            "Benchmark overview: total peak RSS delta",
            "total_peak_rss_delta_mb",
            "Total peak RSS delta [MB]",
        )

    if "stage_timing" in selected_plots:
        make_stage_timing_plot(
            records,
            plot_dir,
        )

    if "stage_memory" in selected_plots:
        make_stage_memory_plot(
            records,
            plot_dir,
        )

    print()
    print("=" * 72)
    print("Benchmark overview")
    print("=" * 72)
    print(f"Loaded benchmark records : {len(records)}")
    print(f"Generated plots         : {', '.join(selected_plots)}")
    print(f"Saved plots to          : {plot_dir}")


if __name__ == "__main__":
    main()
