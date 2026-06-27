from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .config import PLOTS_DIR, RESULTS_DIR


BENCHMARK_NAME = "benchmark_overview"

DEFAULT_RESULTS_DIR = RESULTS_DIR
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME

AVAILABLE_PLOTS = [
    "performance_summary",
    "setup_summary",
    "evaluation_summary",
    "scan_summary",
    "stage_timing",
    "stage_memory",
    "diagnostics",
]

DEFAULT_PLOTS = [
    "performance_summary",
    "stage_timing",
    "stage_memory",
]

STAGE_TIME_KEYS = {
    "workspace_loading": "workspace_loading_wall_time_seconds_mean",
    "model_creation": "model_creation_wall_time_seconds_mean",
    "log_prob_construction": "log_prob_construction_wall_time_seconds_mean",
    "log_prob_compilation": "log_prob_compilation_wall_time_seconds_mean",
    "compiled_evaluation": "compiled_evaluation_average_runtime_seconds_per_evaluation",
    "pdf_evaluation": "pdf_evaluation_average_runtime_seconds_per_evaluation",
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

STAGE_LABELS = {
    "workspace_loading": "Workspace loading",
    "model_creation": "Model creation",
    "log_prob_construction": "Log-prob construction",
    "log_prob_compilation": "Log-prob compilation",
    "compiled_evaluation": "Compiled evaluation",
    "pdf_evaluation": "PDF evaluation",
    "nll_scan": "NLL scan",
}

# Stable, presentation-oriented colors. Keep the same stage colors everywhere.
STAGE_COLORS = {
    "workspace_loading": "#4E79A7",
    "model_creation": "#F28E2B",
    "log_prob_construction": "#59A14F",
    "log_prob_compilation": "#E15759",
    "compiled_evaluation": "#B07AA1",
    "pdf_evaluation": "#9C755F",
    "nll_scan": "#EDC948",
}

BENCHMARK_LABELS = {
    "workspace_loading": "Workspace loading",
    "model_creation": "Model creation",
    "log_prob_construction": "Log-prob construction",
    "log_prob_compilation": "Log-prob compilation",
    "compiled_evaluation": "Compiled evaluation",
    "pdf_evaluation": "PDF evaluation",
    "nll_scan": "NLL scan",
    "model_complexity_scaling": "Model complexity",
    "memory_scaling": "Memory scaling",
}

WORKSPACE_LABELS = {
    "simple_workspace": "Simple",
    "simple_workspace_nonp": "Simple non-parametric",
    "simple_workspace_generic": "Generic",
    "simple_workspace_generic_nonp": "Generic non-parametric",
    "normal_pdf_workspace": "Normal",
    "poisson_pdf_workspace": "Poisson",
    "exponential_pdf_workspace": "Exponential",
    "simple": "Simple",
    "nonp": "Simple non-parametric",
    "generic": "Generic",
    "generic_nonp": "Generic non-parametric",
}


def apply_cern_style() -> None:
    """Apply a clean HEP/paper-style matplotlib configuration."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.15",
            "axes.linewidth": 1.1,
            "axes.titlesize": 21,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,
            "font.family": "DejaVu Sans",
            "grid.color": "0.70",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.28,
            "savefig.dpi": 300,
        }
    )


def finalize_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.28)
    ax.tick_params(axis="both", which="major", length=5, width=1.0)


def resolve_plots(plots: list[str]) -> list[str]:
    if "all" in plots:
        if len(plots) > 1:
            raise ValueError("--plots all cannot be combined with other plots")
        return DEFAULT_PLOTS

    unknown_plots = [plot for plot in plots if plot not in AVAILABLE_PLOTS]
    if unknown_plots:
        raise ValueError(
            f"Unknown plots: {unknown_plots}. "
            f"Available plots: ['all', *{AVAILABLE_PLOTS}]"
        )

    return plots


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as file:
        return json.load(file)


def iter_result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/*_result.json"))


def normalize_benchmark_name(name: str) -> str:
    return BENCHMARK_LABELS.get(name, name.replace("_", " ").title())


def compact_workspace_name(name: str | None) -> str:
    if not name:
        return "Unknown"

    cleaned = str(name).replace(".json", "")
    return WORKSPACE_LABELS.get(cleaned, cleaned.replace("_", " ").title())


def get_workspace_label(result: dict[str, Any]) -> str:
    workspace = compact_workspace_name(result.get("workspace"))
    target = result.get("target")

    if target:
        return f"{workspace}\n{target}"

    return workspace


def extract_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results", [])
    return results if isinstance(results, list) else []


def maybe_to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_float):
        return None
    return value_float


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
                "wall_time_seconds_mean": result.get("wall_time_seconds_mean"),
                "average_runtime_seconds_per_evaluation": result.get(
                    "average_runtime_seconds_per_evaluation"
                ),
                "runtime_per_scan_point_seconds": result.get(
                    "runtime_per_scan_point_seconds"
                ),
                "total_runtime_seconds": result.get("total_runtime_seconds"),
                "total_setup_time_seconds": result.get("total_setup_time_seconds"),
                "current_rss_delta_mb": result.get("current_rss_delta_mb"),
                "peak_rss_delta_mb": result.get("peak_rss_delta_mb"),
                "total_peak_rss_delta_mb": result.get("total_peak_rss_delta_mb"),
            }
            record.update(metric_candidates)

            for stage_name, key in STAGE_TIME_KEYS.items():
                value = maybe_to_float(result.get(key))
                if value is not None:
                    record[f"{stage_name}_time_ms"] = value * 1000.0

            for stage_name, key in STAGE_MEMORY_KEYS.items():
                value = maybe_to_float(result.get(key))
                if value is not None:
                    record[f"{stage_name}_peak_rss_delta_mb"] = value

            if record["wall_time_seconds_mean"] is not None:
                value = maybe_to_float(record["wall_time_seconds_mean"])
                if value is not None:
                    record["wall_time_ms"] = value * 1000.0

            if record["average_runtime_seconds_per_evaluation"] is not None:
                value = maybe_to_float(record["average_runtime_seconds_per_evaluation"])
                if value is not None:
                    record["average_runtime_ms_per_evaluation"] = value * 1000.0

            if record["runtime_per_scan_point_seconds"] is not None:
                value = maybe_to_float(record["runtime_per_scan_point_seconds"])
                if value is not None:
                    record["runtime_ms_per_scan_point"] = value * 1000.0

            if record["total_runtime_seconds"] is not None:
                value = maybe_to_float(record["total_runtime_seconds"])
                if value is not None:
                    record["total_runtime_ms"] = value * 1000.0

            if record["total_setup_time_seconds"] is not None:
                value = maybe_to_float(record["total_setup_time_seconds"])
                if value is not None:
                    record["total_setup_time_ms"] = value * 1000.0

            records.append(record)

    return records


def values_match(actual: Any, expected_values: list[str] | None) -> bool:
    if not expected_values:
        return True
    if actual is None:
        return False
    return str(actual) in expected_values


def numeric_value_matches(actual: Any, expected_values: list[int] | None) -> bool:
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
        if not numeric_value_matches(record.get("n_evaluations"), n_evaluations):
            continue
        if not numeric_value_matches(record.get("n_scan_points"), n_scan_points):
            continue
        filtered.append(record)

    return filtered


def has_metric(records: list[dict[str, Any]], metric_key: str) -> bool:
    values = [maybe_to_float(record.get(metric_key)) for record in records]
    values = [value for value in values if value is not None]
    return bool(values) and any(value != 0.0 for value in values)


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def annotate_horizontal_bars(
    ax: plt.Axes,
    values: list[float],
    unit: str,
    *,
    precision: int = 1,
) -> None:
    if not values:
        return
    max_value = max(values)
    offset = 0.015 * max_value if max_value > 0 else 0.01
    for index, value in enumerate(values):
        if value <= 0:
            continue
        ax.text(
            value + offset,
            index,
            f"{value:.{precision}f} {unit}",
            va="center",
            ha="left",
            fontsize=11,
            fontweight="bold",
        )


def make_ranked_horizontal_plot(
    records: list[dict[str, Any]],
    output_path: Path,
    title: str,
    metric_key: str,
    metric_label: str,
    *,
    unit: str,
    max_rows: int = 12,
    benchmark_filter: set[str] | None = None,
) -> None:
    plot_records = []
    for record in records:
        if benchmark_filter and record.get("benchmark") not in benchmark_filter:
            continue
        value = maybe_to_float(record.get(metric_key))
        if value is None or value == 0.0:
            continue
        plot_records.append((record, value))

    if not plot_records:
        return

    plot_records = sorted(plot_records, key=lambda item: item[1], reverse=True)[:max_rows]
    plot_records = list(reversed(plot_records))

    labels = [
        f"{normalize_benchmark_name(str(record['benchmark']))} · "
        f"{compact_workspace_name(record.get('workspace'))}"
        for record, _value in plot_records
    ]
    values = [value for _record, value in plot_records]

    fig_height = max(5.0, 0.48 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(11.5, fig_height))

    y = np.arange(len(labels))
    ax.barh(y, values, color="#4E79A7", height=0.68)

    ax.set_title(title, loc="left", pad=16, fontweight="bold")
    ax.set_xlabel(metric_label)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    annotate_horizontal_bars(ax, values, unit)
    ax.set_xlim(0, max(values) * 1.18)
    finalize_axes(ax)

    save_figure(fig, output_path)


def aggregate_best_metric_by_workspace(
    records: list[dict[str, Any]],
    benchmark_name: str,
    metric_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        if record.get("benchmark") != benchmark_name:
            continue
        value = maybe_to_float(record.get(metric_key))
        if value is None or value == 0.0:
            continue
        rows.append(
            {
                "workspace": compact_workspace_name(record.get("workspace")),
                "value": value,
            }
        )

    # If several configurations exist for a workspace, report the median.
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["workspace"], []).append(row["value"])

    return [
        {"workspace": workspace, "value": float(np.median(values))}
        for workspace, values in grouped.items()
    ]


def make_grouped_metric_plot(
    records: list[dict[str, Any]],
    output_path: Path,
    title: str,
    benchmark_metric_pairs: list[tuple[str, str, str]],
    y_label: str,
) -> None:
    """Create grouped bars for comparable metrics across workspaces."""

    workspace_order: list[str] = []
    series_rows: list[tuple[str, list[dict[str, Any]]]] = []

    for benchmark, metric_key, label in benchmark_metric_pairs:
        rows = aggregate_best_metric_by_workspace(records, benchmark, metric_key)
        if not rows:
            continue
        for row in rows:
            if row["workspace"] not in workspace_order:
                workspace_order.append(row["workspace"])
        series_rows.append((label, rows))

    if not series_rows or not workspace_order:
        return

    fig_width = max(8.5, 1.5 * len(workspace_order) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 6.0))

    x = np.arange(len(workspace_order))
    width = min(0.72 / len(series_rows), 0.28)

    all_values: list[float] = []
    for series_index, (label, rows) in enumerate(series_rows):
        value_by_workspace = {row["workspace"]: row["value"] for row in rows}
        values = [value_by_workspace.get(workspace, 0.0) for workspace in workspace_order]
        all_values.extend([value for value in values if value > 0])
        offsets = x + (series_index - (len(series_rows) - 1) / 2) * width
        ax.bar(offsets, values, width=width, label=label)

    ax.set_title(title, loc="left", pad=16, fontweight="bold")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(workspace_order, rotation=20, ha="right")
    ax.legend(frameon=False)
    finalize_axes(ax)

    if all_values:
        ymax = max(all_values)
        ax.set_ylim(0, ymax * 1.18)

    save_figure(fig, output_path)


def make_performance_summary_plot(records: list[dict[str, Any]], plot_dir: Path) -> None:
    """Main overview: the few metrics that are safe to compare at a glance."""

    panels = [
        (
            "Setup time",
            "total_setup_time_ms",
            {"model_complexity_scaling", "memory_scaling"},
            "ms",
        ),
        (
            "Compiled evaluation time",
            "average_runtime_ms_per_evaluation",
            {"compiled_evaluation"},
            "ms/eval",
        ),
        (
            "PDF evaluation time",
            "average_runtime_ms_per_evaluation",
            {"pdf_evaluation"},
            "ms/eval",
        ),
        (
            "NLL scan time",
            "runtime_ms_per_scan_point",
            {"nll_scan"},
            "ms/point",
        ),
    ]

    available_panels = []
    for title, metric_key, benchmark_filter, unit in panels:
        panel_records = [
            record
            for record in records
            if record.get("benchmark") in benchmark_filter
            and maybe_to_float(record.get(metric_key)) is not None
            and maybe_to_float(record.get(metric_key)) != 0.0
        ]
        if panel_records:
            available_panels.append((title, metric_key, benchmark_filter, unit, panel_records))

    if not available_panels:
        return

    n_panels = len(available_panels)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(7.0, 4.4 * n_panels), 5.5),
        squeeze=False,
    )

    for ax, (title, metric_key, _benchmark_filter, unit, panel_records) in zip(
        axes[0], available_panels, strict=False
    ):
        grouped: dict[str, list[float]] = {}
        for record in panel_records:
            workspace = compact_workspace_name(record.get("workspace"))
            value = maybe_to_float(record.get(metric_key))
            if value is not None:
                grouped.setdefault(workspace, []).append(value)

        rows = sorted(
            (
                (workspace, float(np.median(values)))
                for workspace, values in grouped.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        labels = [workspace for workspace, _value in rows]
        values = [value for _workspace, value in rows]
        y = np.arange(len(labels))
        ax.barh(y, values, color="#4E79A7", height=0.62)
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        annotate_horizontal_bars(ax, values, unit, precision=3 if max(values) < 1 else 1)
        if values:
            ax.set_xlim(0, max(values) * 1.28)
        finalize_axes(ax)

    fig.suptitle("Benchmark performance summary", x=0.02, ha="left", fontsize=23, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, plot_dir / "benchmark_overview_performance_summary.png")


def make_setup_summary_plot(records: list[dict[str, Any]], plot_dir: Path) -> None:
    make_grouped_metric_plot(
        records=records,
        output_path=plot_dir / "benchmark_overview_setup_summary.png",
        title="Setup time by workspace",
        benchmark_metric_pairs=[
            ("model_complexity_scaling", "total_setup_time_ms", "Model complexity"),
            ("memory_scaling", "total_setup_time_ms", "Memory scaling"),
        ],
        y_label="Setup time [ms]",
    )


def make_evaluation_summary_plot(records: list[dict[str, Any]], plot_dir: Path) -> None:
    make_grouped_metric_plot(
        records=records,
        output_path=plot_dir / "benchmark_overview_evaluation_summary.png",
        title="Average evaluation time by workspace",
        benchmark_metric_pairs=[
            ("compiled_evaluation", "average_runtime_ms_per_evaluation", "Compiled"),
            ("pdf_evaluation", "average_runtime_ms_per_evaluation", "PDF"),
        ],
        y_label="Average time per evaluation [ms]",
    )


def make_scan_summary_plot(records: list[dict[str, Any]], plot_dir: Path) -> None:
    make_ranked_horizontal_plot(
        records=records,
        output_path=plot_dir / "benchmark_overview_scan_summary.png",
        title="NLL scan time per point",
        metric_key="runtime_ms_per_scan_point",
        metric_label="Runtime per scan point [ms]",
        unit="ms/point",
        benchmark_filter={"nll_scan"},
        max_rows=10,
    )


def stage_rows(
    records: list[dict[str, Any]],
    stage_keys: dict[str, str],
    suffix: str,
) -> list[dict[str, Any]]:
    plot_records = [
        record
        for record in records
        if record.get("benchmark") in {"model_complexity_scaling", "memory_scaling"}
    ]

    rows_by_workspace: dict[str, list[list[float]]] = {}
    stage_names = list(stage_keys)

    for record in plot_records:
        workspace = compact_workspace_name(record.get("workspace"))
        values = [
            maybe_to_float(record.get(f"{stage}{suffix}")) or 0.0
            for stage in stage_names
        ]
        if any(value != 0.0 for value in values):
            rows_by_workspace.setdefault(workspace, []).append(values)

    rows = []
    for workspace, workspace_values in rows_by_workspace.items():
        values_array = np.asarray(workspace_values, dtype=float)
        median_values = np.median(values_array, axis=0).tolist()
        if any(value != 0.0 for value in median_values):
            rows.append({"label": workspace, "values": median_values})

    return rows


def format_segment_label(value: float, total: float, unit: str) -> str:
    if value <= 0.0 or total <= 0.0:
        return ""

    percentage = value / total * 100.0
    if percentage < 7.0:
        return ""

    if unit == "ms":
        return f"{value:.1f}\n{percentage:.0f}%"

    return f"{value:.1f}\n{percentage:.0f}%"


def make_stacked_stage_plot(
    records: list[dict[str, Any]],
    plot_dir: Path,
    *,
    title: str,
    output_name: str,
    y_label: str,
    unit: str,
    suffix: str,
    stage_keys: dict[str, str],
) -> None:
    rows = stage_rows(records, stage_keys, suffix)
    if not rows:
        return

    stage_names = list(stage_keys)
    labels = [row["label"] for row in rows]
    totals = np.array([sum(row["values"]) for row in rows])
    order = np.argsort(-totals)

    rows = [rows[index] for index in order]
    labels = [labels[index] for index in order]
    totals = totals[order]

    fig_width = max(9.5, 1.45 * len(labels) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 6.4))

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for stage_index, stage in enumerate(stage_names):
        values = np.array([row["values"][stage_index] for row in rows])
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=STAGE_LABELS[stage],
            color=STAGE_COLORS[stage],
            width=0.64,
        )

        for bar_index, value in enumerate(values):
            label = format_segment_label(value, float(totals[bar_index]), unit)
            if label:
                ax.text(
                    x[bar_index],
                    bottom[bar_index] + value / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                )

        bottom += values

    for index, total in enumerate(totals):
        ax.text(
            x[index],
            total + max(totals) * 0.018,
            f"{total:.1f} {unit}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title(title, loc="left", pad=16, fontweight="bold")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Workspace")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(0, max(totals) * 1.17)
    finalize_axes(ax)

    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
    )

    save_figure(fig, plot_dir / output_name)


def make_stage_timing_plot(records: list[dict[str, Any]], plot_dir: Path) -> None:
    make_stacked_stage_plot(
        records,
        plot_dir,
        title="Stage timing breakdown",
        output_name="benchmark_overview_stage_timing.png",
        y_label="Time [ms]",
        unit="ms",
        suffix="_time_ms",
        stage_keys=STAGE_TIME_KEYS,
    )


def make_stage_memory_plot(records: list[dict[str, Any]], plot_dir: Path) -> None:
    make_stacked_stage_plot(
        records,
        plot_dir,
        title="Stage memory breakdown",
        output_name="benchmark_overview_stage_memory.png",
        y_label="Peak RSS delta [MB]",
        unit="MB",
        suffix="_peak_rss_delta_mb",
        stage_keys=STAGE_MEMORY_KEYS,
    )


def make_diagnostics_plot(records: list[dict[str, Any]], plot_dir: Path) -> None:
    """Optional plot for quick sanity checks, not meant for reports/papers."""

    status_counts: dict[str, int] = {}
    for record in records:
        key = f"{normalize_benchmark_name(str(record['benchmark']))} · {record['status']}"
        status_counts[key] = status_counts.get(key, 0) + 1

    if not status_counts:
        return

    labels = list(status_counts)
    values = [status_counts[label] for label in labels]

    fig_height = max(4.5, 0.42 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#4E79A7")
    ax.set_title("Diagnostic status counts", loc="left", pad=16, fontweight="bold")
    ax.set_xlabel("Result count")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    annotate_horizontal_bars(ax, [float(value) for value in values], "", precision=0)
    ax.set_xlim(0, max(values) * 1.18)
    finalize_axes(ax)
    save_figure(fig, plot_dir / "benchmark_overview_diagnostics_status.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-quality overview plots from benchmark result files."
    )

    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--plots",
        nargs="+",
        default=["all"],
        choices=["all", *AVAILABLE_PLOTS],
    )
    parser.add_argument("--benchmarks", nargs="+")
    parser.add_argument("--workspaces", nargs="+")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--modes", nargs="+")
    parser.add_argument("--n-runs", nargs="+", type=int)
    parser.add_argument("--n-evaluations", nargs="+", type=int)
    parser.add_argument("--n-scan-points", nargs="+", type=int)
    parser.add_argument("--include-failed", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_cern_style()

    records = collect_overview_records(args.results_dir)
    if not records:
        raise ValueError(f"No benchmark results found in {args.results_dir}")

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
        raise ValueError("No benchmark results remain after applying filters.")

    plot_dir = args.plot_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    selected_plots = resolve_plots(args.plots)

    if "performance_summary" in selected_plots:
        make_performance_summary_plot(records, plot_dir)
    if "setup_summary" in selected_plots:
        make_setup_summary_plot(records, plot_dir)
    if "evaluation_summary" in selected_plots:
        make_evaluation_summary_plot(records, plot_dir)
    if "scan_summary" in selected_plots:
        make_scan_summary_plot(records, plot_dir)
    if "stage_timing" in selected_plots:
        make_stage_timing_plot(records, plot_dir)
    if "stage_memory" in selected_plots:
        make_stage_memory_plot(records, plot_dir)
    if "diagnostics" in selected_plots:
        make_diagnostics_plot(records, plot_dir)

    print()
    print("=" * 72)
    print("Benchmark overview")
    print("=" * 72)
    print(f"Loaded benchmark records : {len(records)}")
    print(f"Generated plot set       : {', '.join(selected_plots)}")
    print(f"Saved plots to           : {plot_dir}")


if __name__ == "__main__":
    main()
