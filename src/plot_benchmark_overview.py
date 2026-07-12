from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

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
    "cross_framework_summary",
]

DEFAULT_PLOTS = [
    "performance_summary",
    "stage_timing",
    "stage_memory",
    "cross_framework_summary",
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
    "cross_scalar_pdf_evaluation": "Cross scalar PDF",
    "cross_scalar_pdf": "Cross scalar PDF",
    "cross_nll_scan": "Cross ΔNLL scan",
    "cross_nll": "Cross ΔNLL scan",
    "pyhs3_xroofit_benchmark": "PyHS3 vs xRooFit",
    "cross_model_complexity_scaling": "Cross model complexity",
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
    """Load and validate one benchmark result JSON file."""

    try:
        with path.open() as file:
            payload = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in result file {path}: {exc}") from exc
    except OSError as exc:
        raise OSError(f"Could not read result file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected top-level JSON object in result file {path}, "
            f"got {type(payload).__name__}"
        )

    return payload


def iter_result_files(results_dir: Path) -> list[Path]:
    """Return benchmark result JSON files from a results directory."""

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    if not results_dir.is_dir():
        raise NotADirectoryError(f"Results path is not a directory: {results_dir}")

    return sorted(
        path
        for path in results_dir.rglob("*.json")
        if path.name.endswith("_result.json")
    )


def normalize_benchmark_name(name: str) -> str:
    return BENCHMARK_LABELS.get(name, name.replace("_", " ").title())


def compact_workspace_name(name: str | None) -> str:
    if not name:
        return "Unknown"

    cleaned = Path(str(name)).name.removesuffix(".json").removesuffix(".root")
    if cleaned in WORKSPACE_LABELS:
        return WORKSPACE_LABELS[cleaned]

    parts = cleaned.split("_")
    if parts and parts[0].endswith("ch"):
        channel = parts[0]
        bkg = next((p.replace("bkg", "") for p in parts if p.startswith("bkg")), "")
        sig = next((p.replace("sig", "") for p in parts if p.startswith("sig")), "")
        np_state = next((p for p in parts if p.startswith("np")), "")
        constr = next((p for p in parts if p.startswith("constr")), "")
        yld = next((p for p in parts if p.startswith("yield")), "")

        line1 = channel
        line2 = " / ".join(p for p in [bkg, sig] if p)
        line3 = " / ".join(p for p in [np_state, constr, yld] if p)

        return "\n".join(line for line in [line1, line2, line3] if line)

    return cleaned.replace("_", "\n")


def workspace_from_result(
    payload: dict[str, Any], result: dict[str, Any]
) -> str | None:
    """Resolve a workspace name across old and new benchmark schemas."""

    for key in ("workspace", "workspace_name"):
        value = result.get(key)
        if value:
            return Path(str(value)).name

    for key in ("workspace_path", "json_path", "root_path"):
        value = result.get(key) or payload.get(key)
        if value:
            return Path(str(value)).name

    case = result.get("case")
    if case:
        return str(case)

    return None


def framework_from_result(result: dict[str, Any]) -> str | None:
    """Resolve framework/engine labels across legacy and current schemas."""

    for key in ("framework", "engine", "framework_label", "engine_label"):
        value = result.get(key)
        if value:
            return str(value)
    return None


def get_workspace_label(result: dict[str, Any]) -> str:
    workspace = compact_workspace_name(result.get("workspace"))
    target = result.get("target")
    framework = framework_from_result(result)

    pieces = [workspace]
    if target:
        pieces.append(str(target))
    if framework:
        pieces.append(str(framework))

    return "\n".join(pieces)


def flatten_nested_result(
    payload: dict[str, Any], result: dict[str, Any]
) -> list[dict[str, Any]]:
    """Flatten result schemas that store one row per case with nested frameworks."""

    flattened: list[dict[str, Any]] = []
    for framework in ("pyhs3", "roofit", "root", "xroofit"):
        nested = result.get(framework)
        if not isinstance(nested, dict):
            continue
        row = dict(nested)
        row.setdefault("framework", framework)
        row.setdefault("workspace", workspace_from_result(payload, result))
        row.setdefault("target", result.get("target"))
        row.setdefault("mode", result.get("mode") or payload.get("mode"))
        row.setdefault("status", nested.get("status", result.get("status", "unknown")))
        row.setdefault("case", result.get("case"))
        row.setdefault(
            "analysis", result.get("analysis") or result.get("analysis_name")
        )
        flattened.append(row)
    return flattened


def extract_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract valid result rows from a benchmark result payload."""

    results = payload.get("results", [])

    if not isinstance(results, list):
        return []

    extracted: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        nested = flatten_nested_result(payload, result)
        if nested:
            extracted.extend(nested)
        else:
            extracted.append(result)

    return extracted


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


def collect_overview_records(
    results_dir: Path,
    *,
    strict: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    Collect normalized benchmark records from result JSON files.

    Invalid files or malformed result rows are skipped by default, so one bad
    result does not prevent overview plotting. Use strict=True to fail fast.
    """

    records = []
    skipped_items: list[dict[str, str]] = []

    for result_file in iter_result_files(results_dir):
        try:
            payload = load_json(result_file)
        except (OSError, TypeError, ValueError) as exc:
            skipped_items.append(
                {
                    "path": str(result_file),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            if strict:
                raise
            print(f"Skipping result file {result_file}: {type(exc).__name__}: {exc}")
            continue

        benchmark = payload.get("benchmark")

        if benchmark is None:
            for parent in result_file.parents:
                if parent.name in BENCHMARK_LABELS:
                    benchmark = parent.name
                    break

        for result in extract_results(payload):
            try:
                resolved_workspace = workspace_from_result(payload, result)
                result_for_label = dict(result)
                result_for_label["workspace"] = resolved_workspace

                record: dict[str, Any] = {
                    "benchmark": benchmark,
                    "benchmark_label": normalize_benchmark_name(str(benchmark)),
                    "workspace": resolved_workspace,
                    "workspace_label": get_workspace_label(result_for_label),
                    "target": (
                        result.get("target")
                        or result.get("distribution")
                        or result.get("analysis")
                    ),
                    "mode": result.get("mode") or payload.get("mode"),
                    "framework": framework_from_result(result),
                    "engine": result.get("engine"),
                    "category": result.get("category_key") or result.get("category"),
                    "input_mode": result.get("input_mode"),
                    "source_file": str(result_file),
                    "status": result.get("status", "unknown"),
                    "n_runs": result.get("n_runs") or payload.get("n_runs"),
                    "n_evaluations": result.get("n_evaluations")
                    or payload.get("n_evaluations"),
                    "n_scan_points": (
                        result.get("n_scan_points")
                        or payload.get("n_points")
                        or payload.get("configuration", {}).get("n_mu_values")
                    ),
                }

                n_evaluations = maybe_to_float(record["n_evaluations"])
                warm_total = maybe_to_float(
                    result.get("warm_total_time_seconds")
                    or result.get("warm_total_seconds")
                    or result.get("warm_runtime_seconds")
                )
                derived_time_per_evaluation = None
                if warm_total is not None and n_evaluations and n_evaluations > 0:
                    derived_time_per_evaluation = warm_total / n_evaluations

                metric_candidates = {
                    "wall_time_seconds_mean": result.get("wall_time_seconds_mean"),
                    "average_runtime_seconds_per_evaluation": (
                        result.get("average_runtime_seconds_per_evaluation")
                        or result.get("time_per_evaluation_seconds")
                        or result.get("time_per_value_seconds")
                        or result.get("time_per_value_seconds_median")
                        or result.get("steady_state_seconds_median")
                        or result.get("warm_time_per_evaluation_seconds")
                        or result.get("warm_per_evaluation_seconds")
                        or derived_time_per_evaluation
                    ),
                    "runtime_per_scan_point_seconds": (
                        result.get("runtime_per_scan_point_seconds")
                        or result.get("time_per_scan_point_seconds")
                    ),
                    "total_runtime_seconds": (
                        result.get("total_runtime_seconds")
                        or result.get("full_scan_time_seconds")
                        or result.get("full_scan_time_seconds_median")
                        or result.get("scan_time_seconds")
                    ),
                    "total_setup_time_seconds": (
                        result.get("total_setup_time_seconds")
                        or result.get("cold_start_end_to_end_seconds")
                        or result.get("end_to_end_first_evaluation_seconds")
                        or result.get("model_build_time_seconds")
                        or result.get("build_time_seconds")
                    ),
                    "cold_first_evaluation_time_seconds": (
                        result.get("cold_first_evaluation_time_seconds")
                        or result.get("first_call_seconds")
                    ),
                    "warm_evaluation_time_seconds_mean": (
                        result.get("warm_evaluation_time_seconds_mean")
                        or (result.get("warm_evaluation") or {}).get("mean_seconds")
                    ),
                    "current_rss_delta_mb": (
                        result.get("current_rss_delta_mb") or result.get("rss_delta_mb")
                    ),
                    "peak_rss_delta_mb": result.get("peak_rss_delta_mb"),
                    "total_peak_rss_delta_mb": result.get("total_peak_rss_delta_mb"),
                    "throughput_evaluations_per_second": (
                        result.get("throughput_evaluations_per_second")
                        or result.get("scan_throughput_points_per_second")
                        or result.get("throughput")
                    ),
                    "delta_nll_shape_max_abs_diff": (
                        result.get("delta_nll_shape_max_abs_diff")
                        or result.get("delta_nll_max_abs_diff")
                        or (result.get("agreement") or {}).get("delta_nll_max_abs_diff")
                    ),
                    "minimum_mu_abs_diff": (
                        result.get("minimum_mu_abs_diff")
                        or result.get("minimum_poi_abs_diff")
                        or (result.get("agreement") or {}).get("minimum_poi_abs_diff")
                    ),
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
                    value = maybe_to_float(
                        record["average_runtime_seconds_per_evaluation"]
                    )
                    if value is not None:
                        record["average_runtime_ms_per_evaluation"] = value * 1000.0
                        record["time_per_evaluation_us"] = value * 1_000_000.0

                if record["runtime_per_scan_point_seconds"] is not None:
                    value = maybe_to_float(record["runtime_per_scan_point_seconds"])
                    if value is not None:
                        record["runtime_ms_per_scan_point"] = value * 1000.0
                        record["time_per_scan_point_us"] = value * 1_000_000.0

                if record["cold_first_evaluation_time_seconds"] is not None:
                    value = maybe_to_float(record["cold_first_evaluation_time_seconds"])
                    if value is not None:
                        record["cold_first_evaluation_ms"] = value * 1000.0

                if record["warm_evaluation_time_seconds_mean"] is not None:
                    value = maybe_to_float(record["warm_evaluation_time_seconds_mean"])
                    if value is not None:
                        record["warm_evaluation_us"] = value * 1_000_000.0

                if record["total_runtime_seconds"] is not None:
                    value = maybe_to_float(record["total_runtime_seconds"])
                    if value is not None:
                        record["total_runtime_ms"] = value * 1000.0

                if record["total_setup_time_seconds"] is not None:
                    value = maybe_to_float(record["total_setup_time_seconds"])
                    if value is not None:
                        record["total_setup_time_ms"] = value * 1000.0

                records.append(record)
            except (KeyError, TypeError, ValueError) as exc:
                skipped_items.append(
                    {
                        "path": str(result_file),
                        "error_type": type(exc).__name__,
                        "error_message": f"Could not normalize one result row: {exc}",
                    }
                )
                if strict:
                    raise
                print(
                    f"Skipping malformed result row in {result_file}: "
                    f"{type(exc).__name__}: {exc}"
                )

    return records, skipped_items


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
    """Save a matplotlib figure and verify that the file was created."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.savefig(output_path, bbox_inches="tight")
    except OSError as exc:
        raise OSError(f"Could not save plot to {output_path}: {exc}") from exc
    finally:
        plt.close(fig)

    if not output_path.exists() or not output_path.is_file():
        raise FileNotFoundError(f"Plot file was not created: {output_path}")


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

    plot_records = sorted(plot_records, key=lambda item: item[1], reverse=True)[
        :max_rows
    ]
    plot_records = list(reversed(plot_records))

    labels = []
    for record, _value in plot_records:
        label = (
            f"{normalize_benchmark_name(str(record['benchmark']))} · "
            f"{compact_workspace_name(record.get('workspace'))}"
        )
        if record.get("framework"):
            label += f" · {record['framework']}"
        labels.append(label)
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
        values = [
            value_by_workspace.get(workspace, 0.0) for workspace in workspace_order
        ]
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


def make_performance_summary_plot(
    records: list[dict[str, Any]], plot_dir: Path
) -> None:
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
            "Cross scalar PDF",
            "time_per_evaluation_us",
            {"cross_scalar_pdf_evaluation", "cross_scalar_pdf"},
            "µs/eval",
        ),
        (
            "NLL scan time",
            "runtime_ms_per_scan_point",
            {"nll_scan"},
            "ms/point",
        ),
        (
            "Cross ΔNLL scan",
            "time_per_scan_point_us",
            {"cross_nll_scan", "cross_nll", "pyhs3_xroofit_benchmark"},
            "µs/point",
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
            available_panels.append(
                (title, metric_key, benchmark_filter, unit, panel_records)
            )

    if not available_panels:
        return

    n_panels = len(available_panels)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(max(9.0, 5.4 * n_panels), 7.2),
        squeeze=False,
    )

    for panel_index, (
        ax,
        (title, metric_key, _benchmark_filter, unit, panel_records),
    ) in enumerate(zip(axes[0], available_panels, strict=False)):
        grouped: dict[str, list[float]] = {}
        for record in panel_records:
            workspace = compact_workspace_name(record.get("workspace"))
            if record.get("framework") and record.get("benchmark") in {
                "cross_scalar_pdf_evaluation",
                "cross_scalar_pdf",
                "cross_nll_scan",
                "cross_nll",
                "pyhs3_xroofit_benchmark",
            }:
                workspace = f"{workspace}\n{record['framework']}"
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
        if panel_index == 0:
            ax.set_yticklabels(labels, fontsize=9)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        annotate_horizontal_bars(
            ax, values, unit, precision=3 if max(values) < 1 else 1
        )
        if values:
            ax.set_xlim(0, max(values) * 1.28)
        finalize_axes(ax)

    fig.suptitle(
        "Benchmark performance summary",
        x=0.02,
        ha="left",
        fontsize=23,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.28, wspace=0.65)
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
        benchmark_filter={
            "nll_scan",
            "cross_nll_scan",
            "cross_nll",
            "pyhs3_xroofit_benchmark",
        },
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

    fig_width = max(12.0, 2.2 * len(labels) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 7.4))

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
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, max(totals) * 1.17)
    finalize_axes(ax)

    ax.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        borderaxespad=0.0,
    )

    fig.subplots_adjust(bottom=0.32, right=0.80)
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


def _canonical_cross_engine(record: dict[str, Any]) -> str | None:
    """Return one stable engine key for current and legacy result schemas."""

    raw = (
        record.get("engine")
        or record.get("framework")
        or record.get("engine_label")
        or record.get("framework_label")
    )
    if raw is None:
        return None

    normalized = str(raw).strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "pyhs3_noncompiled": "pyhs3_noncompiled",
        "pyhs3_non_compiled": "pyhs3_noncompiled",
        "pyhs3_(pytensor)": "pyhs3_noncompiled",
        "pyhs3_noncompiled_(pytensor)": "pyhs3_noncompiled",
        "pyhs3_non_compiled_(pytensor)": "pyhs3_noncompiled",
        "pyhs3_compiled": "pyhs3_compiled",
        "pyhs3_compiled_(jax)": "pyhs3_compiled",
        "roofit": "roofit",
        "root": "roofit",
    }
    return aliases.get(normalized)


def _cross_workspace_label(name: str | None) -> str:
    """Create a compact workspace label suitable for grouped overview plots."""

    if not name:
        return "Unknown"

    cleaned = Path(str(name)).name.removesuffix(".json").removesuffix(".root")
    parts = cleaned.split("_")

    if parts and parts[0].endswith("ch"):
        channel = parts[0]
        np_state = next((part for part in parts if part.startswith("np")), "")
        yield_scale = next((part for part in parts if part.startswith("yield")), "")
        details = " / ".join(part for part in (np_state, yield_scale) if part)
        return f"{channel}\n{details}" if details else channel

    return compact_workspace_name(name)


def _select_fastest_unique_cross_rows(
    records: list[dict[str, Any]],
    *,
    benchmark_names: set[str],
    metric_key: str,
    required_category: str | None = None,
    required_input_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Select one valid row per workspace and engine.

    Result files can contain several batch sizes, repeats, input modes, or
    compatibility aliases. For overview purposes, keep one row per
    workspace/engine. When several equivalent rows remain, use the row with the
    largest evaluation count and then the smallest measured latency.
    """

    candidates: list[dict[str, Any]] = []

    for record in records:
        if record.get("benchmark") not in benchmark_names:
            continue
        if record.get("status") != "success":
            continue
        if (
            required_category is not None
            and record.get("category") != required_category
        ):
            continue
        if (
            required_input_mode is not None
            and record.get("input_mode") != required_input_mode
        ):
            continue

        engine = _canonical_cross_engine(record)
        value = maybe_to_float(record.get(metric_key))
        if engine is None or value is None or value <= 0.0:
            continue

        row = dict(record)
        row["_canonical_engine"] = engine
        row["_metric_value"] = value
        row["_workspace_group"] = _cross_workspace_label(record.get("workspace"))
        row["_evaluation_count"] = maybe_to_float(record.get("n_evaluations")) or 0.0
        candidates.append(row)

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidates:
        key = (row["_workspace_group"], row["_canonical_engine"])
        previous = selected.get(key)
        if previous is None:
            selected[key] = row
            continue

        previous_count = float(previous["_evaluation_count"])
        current_count = float(row["_evaluation_count"])

        if current_count > previous_count:
            selected[key] = row
        elif (
            current_count == previous_count
            and row["_metric_value"] < previous["_metric_value"]
        ):
            selected[key] = row

    return list(selected.values())


def _make_cross_framework_grouped_plot(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str,
    y_label: str,
    value_suffix: str,
) -> None:
    """Plot three execution engines side by side for every workspace."""

    if not rows:
        return

    engine_order = ["pyhs3_noncompiled", "pyhs3_compiled", "roofit"]
    engine_labels = {
        "pyhs3_noncompiled": "pyHS3 non-compiled (PyTensor)",
        "pyhs3_compiled": "pyHS3 compiled (JAX)",
        "roofit": "RooFit",
    }
    engine_colors = {
        "pyhs3_noncompiled": "#1565C0",
        "pyhs3_compiled": "#EF6C00",
        "roofit": "#00897B",
    }

    workspace_order: list[str] = []
    for row in rows:
        workspace = str(row["_workspace_group"])
        if workspace not in workspace_order:
            workspace_order.append(workspace)

    # Prefer channel-count ordering for canonical workspaces.
    def workspace_sort_key(label: str) -> tuple[int, str]:
        first_line = label.splitlines()[0]
        digits = "".join(character for character in first_line if character.isdigit())
        return (int(digits) if digits else 10**9, label)

    workspace_order = sorted(workspace_order, key=workspace_sort_key)

    values_by_key = {
        (str(row["_workspace_group"]), str(row["_canonical_engine"])): float(
            row["_metric_value"]
        )
        for row in rows
    }

    fig_width = max(10.5, 2.4 * len(workspace_order) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 7.0))

    x = np.arange(len(workspace_order), dtype=float)
    width = 0.23
    all_values: list[float] = []

    for engine_index, engine in enumerate(engine_order):
        values = [
            values_by_key.get((workspace, engine), np.nan)
            for workspace in workspace_order
        ]
        offsets = x + (engine_index - 1) * width
        bars = ax.bar(
            offsets,
            values,
            width=width,
            label=engine_labels[engine],
            color=engine_colors[engine],
            edgecolor="white",
            linewidth=0.9,
        )

        for bar, value in zip(bars, values, strict=True):
            if not np.isfinite(value) or value <= 0.0:
                continue
            all_values.append(float(value))
            ax.annotate(
                f"{value:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                rotation=0,
            )

    if not all_values:
        plt.close(fig)
        return

    ratio = max(all_values) / min(all_values)
    if ratio >= 20.0:
        ax.set_yscale("log")
        ax.set_ylim(min(all_values) * 0.65, max(all_values) * 1.65)
    else:
        ax.set_ylim(0.0, max(all_values) * 1.28)

    ax.set_title(title, loc="left", pad=18, fontweight="bold")
    ax.set_ylabel(y_label)
    ax.set_xlabel("Workspace")
    ax.set_xticks(x)
    ax.set_xticklabels(workspace_order, fontsize=12)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    finalize_axes(ax)

    ax.text(
        1.0,
        1.015,
        f"Values shown in {value_suffix}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        color="0.35",
    )

    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.24)
    save_figure(fig, output_path)


def make_cross_framework_summary_plot(
    records: list[dict[str, Any]], plot_dir: Path
) -> None:
    """Create separate Scalar PDF and pointwise NLL cross-framework summaries.

    The two metrics intentionally use different output figures because a scalar
    PDF call and one complete NLL evaluation are not interchangeable units.
    Each figure groups results by workspace and compares the same three engines:
    non-compiled pyHS3, compiled pyHS3, and RooFit.
    """

    scalar_rows = _select_fastest_unique_cross_rows(
        records,
        benchmark_names={"cross_scalar_pdf_evaluation", "cross_scalar_pdf"},
        metric_key="time_per_evaluation_us",
        required_input_mode="varying",
    )
    _make_cross_framework_grouped_plot(
        scalar_rows,
        plot_dir / "benchmark_overview_cross_framework_scalar_pdf.png",
        title="Cross-framework Scalar PDF summary",
        y_label="Median time per scalar PDF evaluation [µs]",
        value_suffix="µs/evaluation",
    )

    pointwise_nll_rows = _select_fastest_unique_cross_rows(
        records,
        benchmark_names={"cross_nll_scan", "cross_nll"},
        metric_key="time_per_scan_point_us",
        required_category="pointwise_nll",
    )
    _make_cross_framework_grouped_plot(
        pointwise_nll_rows,
        plot_dir / "benchmark_overview_cross_framework_pointwise_nll.png",
        title="Cross-framework Pointwise NLL summary",
        y_label="Median time per complete NLL evaluation [µs]",
        value_suffix="µs/NLL evaluation",
    )


def make_diagnostics_plot(records: list[dict[str, Any]], plot_dir: Path) -> None:
    """Optional plot for quick sanity checks, not meant for reports/papers."""

    status_counts: dict[str, int] = {}
    for record in records:
        key = (
            f"{normalize_benchmark_name(str(record['benchmark']))} · {record['status']}"
        )
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


def run_plot_builder(
    plot_name: str,
    builder: Callable[[], None],
    *,
    strict: bool,
) -> bool:
    """Run one plot builder and report whether it completed successfully."""

    try:
        builder()
    except Exception as exc:
        if strict:
            raise
        print(f"Skipping plot '{plot_name}': {type(exc).__name__}: {exc}")
        return False

    return True


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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately instead of skipping malformed result files or failed plots.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_cern_style()

    records, skipped_items = collect_overview_records(
        args.results_dir,
        strict=args.strict,
    )
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

    plot_builders: dict[str, Callable[[], None]] = {
        "performance_summary": lambda: make_performance_summary_plot(records, plot_dir),
        "setup_summary": lambda: make_setup_summary_plot(records, plot_dir),
        "evaluation_summary": lambda: make_evaluation_summary_plot(records, plot_dir),
        "scan_summary": lambda: make_scan_summary_plot(records, plot_dir),
        "stage_timing": lambda: make_stage_timing_plot(records, plot_dir),
        "stage_memory": lambda: make_stage_memory_plot(records, plot_dir),
        "diagnostics": lambda: make_diagnostics_plot(records, plot_dir),
        "cross_framework_summary": lambda: make_cross_framework_summary_plot(
            records, plot_dir
        ),
    }

    completed_plots = []
    skipped_plots = []

    for plot_name in selected_plots:
        if run_plot_builder(
            plot_name=plot_name,
            builder=plot_builders[plot_name],
            strict=args.strict,
        ):
            completed_plots.append(plot_name)
        else:
            skipped_plots.append(plot_name)

    print()
    print("=" * 72)
    print("Benchmark overview")
    print("=" * 72)
    print(f"Loaded benchmark records : {len(records)}")
    print(f"Selected plot set        : {', '.join(selected_plots)}")
    print(
        f"Completed plot builders  : {', '.join(completed_plots) if completed_plots else 'none'}"
    )
    print(
        f"Skipped plot builders    : {', '.join(skipped_plots) if skipped_plots else 'none'}"
    )
    print(f"Skipped result items     : {len(skipped_items)}")
    print(f"Saved plots to           : {plot_dir}")


if __name__ == "__main__":
    main()
