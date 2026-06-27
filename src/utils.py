"""
This file contains utility functions for benchmarking pyHS3,
including memory usage tracking, timing, plotting, and workspace/model handling.
"""

from __future__ import annotations

import json
import math
import resource
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import psutil
from pyhs3 import jaxify
from pyhs3.model import Model
from pyhs3.transpile import JaxifiedGraph
from pyhs3.workspace import Workspace
from pytensor.tensor.variable import TensorVariable

from .config import WORKSPACE_LABELS


def get_current_rss_mb() -> float:
    """
    Return current process RSS usage in MB.
    """

    process = psutil.Process()

    return process.memory_info().rss / (1024 * 1024)


def get_peak_rss_mb() -> float:
    """
    Return peak process RSS usage in MB.

    On Linux, ru_maxrss is reported in KB.
    """

    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    return rss_kb / 1024.0


def run_repeated_timing(
    func,
    n_runs: int = 5,
    warmup_runs: int = 1,
) -> tuple[Any, list[float]]:
    """
    Run the given function multiple times and return
    the result of the last run along with the list of timings.
    """

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")

    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")

    timings = []

    try:
        for _ in range(warmup_runs):
            func()

        result = None

        for _ in range(n_runs):
            start = time.perf_counter()
            result = func()
            end = time.perf_counter()

            timings.append(end - start)

    except Exception as exc:
        raise RuntimeError(
            "Repeated timing failed while executing the benchmarked function"
        ) from exc

    return result, timings


def summarize_timings(timings: list[float]) -> dict[str, float]:
    """
    Summarize the list of timings.
    """

    if len(timings) == 0:
        raise ValueError("Cannot summarize an empty timing list")

    invalid_timings = [
        timing
        for timing in timings
        if not math.isfinite(timing) or timing <= 0
    ]

    if invalid_timings:
        raise ValueError(
            "Timing samples must be positive finite values. "
            f"Invalid samples: {invalid_timings}"
        )

    return {
        "wall_time_seconds_mean": statistics.mean(timings),
        "wall_time_seconds_median": statistics.median(timings),
        "wall_time_seconds_std": (
            statistics.stdev(timings)
            if len(timings) > 1
            else 0.0
        ),
    }


def save_json(data: dict[str, Any], output_path: Path) -> None:
    """
    Save the given data as JSON to the specified output path.
    """

    output_path = Path(output_path)

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open("w") as f:
            json.dump(
                data,
                f,
                indent=2,
                sort_keys=True,
            )

    except TypeError as exc:
        raise TypeError(
            f"Benchmark output is not JSON serializable: {output_path}"
        ) from exc

    except OSError as exc:
        raise OSError(
            f"Failed to write benchmark JSON output to {output_path}"
        ) from exc


def should_plot_metric(
    results: list[dict[str, Any]],
    metric_key: str,
) -> bool:
    """
    Return True if a metric exists and has at least one non-zero value.
    """

    values = []

    for result in results:
        if result.get("status") == "failed":
            continue

        value = result.get(metric_key)

        if value is None:
            continue

        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue

        if math.isfinite(value_float):
            values.append(value_float)

    return any(value != 0 for value in values)


def _apply_style() -> None:
    """
    Apply matplotlib style for benchmark plots.
    """

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.25",
            "axes.linewidth": 1.5,
            "axes.titlesize": 24,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "grid.color": "0.55",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.35,
            "savefig.dpi": 300,
        }
    )


def _result_label(result: dict[str, Any]) -> str:
    """
    Create a compact label for one benchmark result.
    """

    if "plot_label" in result:
        return str(result["plot_label"])

    workspace = result.get("workspace", "").replace(".json", "")
    workspace = WORKSPACE_LABELS.get(workspace, workspace)

    n_evaluations = result.get("n_evaluations")

    if n_evaluations is None:
        return workspace

    return f"{workspace}\n{n_evaluations}"


def _scaled_metric(
    results: list[dict[str, Any]],
    metric_key: str,
    metric_label: str,
) -> tuple[list[float], list[float] | None, str]:
    """
    Return values, optional errors, and an updated y-axis label.

    Timing means are stored in seconds but plotted in ms.
    """

    if len(results) == 0:
        raise ValueError("Cannot plot an empty result list")

    missing_results = [
        result.get("workspace", "<unknown>")
        for result in results
        if metric_key not in result
    ]

    if missing_results:
        raise KeyError(
            f"Metric '{metric_key}' is missing for results: {missing_results}"
        )

    try:
        values = [float(result[metric_key]) for result in results]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Metric '{metric_key}' contains non-numeric values") from exc

    if any(not math.isfinite(value) for value in values):
        raise ValueError(f"Metric '{metric_key}' contains non-finite values")

    std_key = None

    if metric_key.endswith("_mean"):
        candidate = metric_key.removesuffix("_mean") + "_std"

        if any(candidate in result for result in results):
            std_key = candidate

    errors = (
        [float(result.get(std_key, 0.0)) for result in results]
        if std_key is not None
        else None
    )

    if metric_key == "wall_time_seconds_mean":
        values = [value * 1000.0 for value in values]

        if errors is not None:
            errors = [error * 1000.0 for error in errors]

        metric_label = "Mean wall time [ms]"

    return values, errors, metric_label


def _format_value(value: float, metric_label: str) -> str:
    """
    Format bar labels in readable units.
    """

    if not math.isfinite(value):
        return "nan"

    if "[ms]" in metric_label:
        return f"{value:.3f}"

    if "[MB]" in metric_label:
        return f"{value:.3f}"

    if abs(value) >= 100:
        return f"{value:.0f}"

    return f"{value:.3f}"


def make_bar_plot(
    results: list[dict[str, Any]],
    output_path: Path,
    title: str,
    metric_key: str,
    metric_label: str,
) -> None:
    """
    Create a bar plot for a benchmark metric.
    """

    results = [
        result
        for result in results
        if result.get("status") != "failed"
    ]

    if len(results) == 0:
        raise ValueError("Cannot create a plot without successful benchmark results")

    _apply_style()

    labels = [_result_label(result) for result in results]

    values, errors, metric_label = _scaled_metric(
        results,
        metric_key,
        metric_label,
    )

    fig_width = max(14, len(labels) * 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, 9))

    bars = ax.bar(
        range(len(values)),
        values,
        yerr=errors,
        capsize=6 if errors is not None else 0,
    )

    ax.set_title(
        title,
        fontsize=24,
        pad=20,
        weight="bold",
    )

    ax.set_ylabel(
        metric_label,
        fontsize=18,
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(
        labels,
        rotation=30,
        ha="right",
        fontsize=14,
    )

    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.3)

    ymin = min(0.0, min(values))
    ymax = max(values)

    if errors is not None:
        ymax = max(
            value + error
            for value, error in zip(values, errors, strict=False)
        )

    span = ymax - ymin

    if span <= 0:
        span = 1.0

    ax.set_ylim(
        ymin - 0.08 * span,
        ymax + 0.35 * span,
    )

    has_nonzero_errors = (
        errors is not None
        and any(error_value != 0 for error_value in errors)
    )

    for index, (bar, value) in enumerate(zip(bars, values, strict=False)):
        error = errors[index] if errors is not None else 0.0
        offset = 0.035 * span

        if value >= 0:
            y = value + error + offset
            va = "bottom"
        else:
            y = value - error - offset
            va = "top"

        if has_nonzero_errors:
            label = (
                f"{_format_value(value, metric_label)} ± "
                f"{_format_value(error, metric_label)}"
            )
        else:
            label = _format_value(value, metric_label)

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va=va,
            fontsize=14,
            fontweight="bold",
        )

    fig.tight_layout()

    output_path = Path(output_path)

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(output_path)
    except OSError as exc:
        raise OSError(f"Failed to save plot to {output_path}") from exc
    finally:
        plt.close(fig)



def _grouped_result_parts(result: dict[str, Any]) -> tuple[str, int | None]:
    """
    Return workspace label and n_evaluations for grouped benchmark plots.
    """

    workspace = result.get("workspace", "").replace(".json", "")
    workspace = WORKSPACE_LABELS.get(workspace, workspace)
    n_evaluations = result.get("n_evaluations")

    return workspace, n_evaluations


def make_grouped_bar_plot(
    results: list[dict[str, Any]],
    output_path: Path,
    title: str,
    metric_key: str,
    metric_label: str,
) -> None:
    """
    Create a grouped bar plot where workspaces are groups and n_evaluations
    values are bars inside each group.
    """

    results = [
        result
        for result in results
        if result.get("status") != "failed"
    ]

    if len(results) == 0:
        raise ValueError("Cannot create a plot without successful benchmark results")

    _apply_style()

    values, errors, metric_label = _scaled_metric(
        results,
        metric_key,
        metric_label,
    )

    enriched_results = []
    for result, value in zip(results, values, strict=False):
        workspace, n_evaluations = _grouped_result_parts(result)
        enriched_results.append(
            {
                "workspace_label": workspace,
                "n_evaluations": n_evaluations,
                "value": value,
            }
        )

    workspace_labels = []
    for result in enriched_results:
        if result["workspace_label"] not in workspace_labels:
            workspace_labels.append(result["workspace_label"])

    n_values = sorted(
        {
            result["n_evaluations"]
            for result in enriched_results
            if result["n_evaluations"] is not None
        }
    )

    if not workspace_labels or not n_values:
        raise ValueError("Grouped bar plot requires workspace and n_evaluations values")

    value_by_group = {
        (result["workspace_label"], result["n_evaluations"]): result["value"]
        for result in enriched_results
    }

    x = np.arange(len(workspace_labels))
    width = min(0.8 / len(n_values), 0.18)

    fig_width = max(12, len(workspace_labels) * 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    all_plotted_values = []

    for index, n_evaluations in enumerate(n_values):
        offsets = x + (index - (len(n_values) - 1) / 2) * width
        plot_values = [
            value_by_group.get((workspace_label, n_evaluations), np.nan)
            for workspace_label in workspace_labels
        ]
        all_plotted_values.extend(
            value for value in plot_values if math.isfinite(float(value))
        )

        bars = ax.bar(
            offsets,
            plot_values,
            width=width,
            label=f"n={n_evaluations}",
        )

        for bar, value in zip(bars, plot_values, strict=False):
            if not math.isfinite(float(value)):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                _format_value(float(value), metric_label),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                rotation=90,
            )

    ax.set_title(title, fontsize=24, pad=20, weight="bold")
    ax.set_ylabel(metric_label, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(workspace_labels, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Evaluations", fontsize=12, title_fontsize=12)

    ymin = min(0.0, min(all_plotted_values))
    ymax = max(all_plotted_values)
    span = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.30 * span)

    fig.tight_layout()
    output_path = Path(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    except OSError as exc:
        raise OSError(f"Failed to save plot to {output_path}") from exc
    finally:
        plt.close(fig)


def make_line_plot_by_evaluations(
    results: list[dict[str, Any]],
    output_path: Path,
    title: str,
    metric_key: str,
    metric_label: str,
    *,
    log_x: bool = True,
) -> None:
    """
    Create a line plot with n_evaluations on the x-axis and one line per
    workspace.
    """

    results = [
        result
        for result in results
        if result.get("status") != "failed"
    ]

    if len(results) == 0:
        raise ValueError("Cannot create a plot without successful benchmark results")

    _apply_style()

    values, _errors, metric_label = _scaled_metric(
        results,
        metric_key,
        metric_label,
    )

    enriched_results = []
    for result, value in zip(results, values, strict=False):
        workspace, n_evaluations = _grouped_result_parts(result)
        if n_evaluations is None:
            continue
        enriched_results.append(
            {
                "workspace_label": workspace,
                "n_evaluations": int(n_evaluations),
                "value": float(value),
            }
        )

    workspace_labels = []
    for result in enriched_results:
        if result["workspace_label"] not in workspace_labels:
            workspace_labels.append(result["workspace_label"])

    if len(enriched_results) == 0:
        raise ValueError("Line plot requires at least one n_evaluations value")

    fig, ax = plt.subplots(figsize=(12, 8))

    all_values = []
    all_n_values = sorted({result["n_evaluations"] for result in enriched_results})

    for workspace_label in workspace_labels:
        rows = sorted(
            (
                result
                for result in enriched_results
                if result["workspace_label"] == workspace_label
            ),
            key=lambda result: result["n_evaluations"],
        )
        x_values = [result["n_evaluations"] for result in rows]
        y_values = [result["value"] for result in rows]
        all_values.extend(y_values)

        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=workspace_label,
        )

    ax.set_title(title, fontsize=24, pad=20, weight="bold")
    ax.set_xlabel("Number of evaluations", fontsize=18)
    ax.set_ylabel(metric_label, fontsize=18)
    ax.set_xticks(all_n_values)
    ax.set_xticklabels([str(value) for value in all_n_values])
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    if log_x and all(value > 0 for value in all_n_values):
        ax.set_xscale("log")

    ymin = min(0.0, min(all_values))
    ymax = max(all_values)
    span = ymax - ymin if ymax > ymin else 1.0
    ax.set_ylim(ymin - 0.08 * span, ymax + 0.22 * span)

    fig.tight_layout()
    output_path = Path(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    except OSError as exc:
        raise OSError(f"Failed to save plot to {output_path}") from exc
    finally:
        plt.close(fig)


def load_workspace(workspace_path: Path) -> Workspace:
    """
    Load a workspace from the given path.
    """

    workspace_path = Path(workspace_path)

    if not workspace_path.exists():
        raise FileNotFoundError(f"Workspace file does not exist: {workspace_path}")

    if not workspace_path.is_file():
        raise FileNotFoundError(f"Workspace path is not a file: {workspace_path}")

    try:
        return Workspace.load(workspace_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load workspace from {workspace_path}") from exc


def create_model(
    workspace: Workspace,
    target: str,
    mode: str,
) -> Model:
    """
    Create a model from the given workspace, target, and mode.
    """

    if workspace is None:
        raise ValueError("workspace must not be None")

    if not target:
        raise ValueError("target must be a non-empty string")

    if not mode:
        raise ValueError("mode must be a non-empty string")

    try:
        return workspace.model(
            target,
            progress=False,
            mode=mode,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create model for target='{target}', mode='{mode}'"
        ) from exc


def build_log_prob(
    workspace_path: Path,
    target: str,
    mode: str,
) -> tuple[Model, TensorVariable]:
    """
    Build a log_prob graph from the given workspace, target, and mode.
    """

    workspace = load_workspace(workspace_path)

    model = create_model(
        workspace=workspace,
        target=target,
        mode=mode,
    )

    try:
        log_prob = model.log_prob
    except Exception as exc:
        raise RuntimeError(
            f"Failed to build log_prob for {workspace_path}, "
            f"target='{target}', mode='{mode}'"
        ) from exc

    if log_prob is None:
        raise ValueError(
            f"log_prob construction returned None for {workspace_path}, "
            f"target='{target}', mode='{mode}'"
        )

    return model, log_prob


def compile_log_prob(log_prob: TensorVariable) -> JaxifiedGraph:
    """
    Compile a log_prob graph to a JaxifiedGraph.
    """

    if log_prob is None:
        raise ValueError("log_prob must not be None")

    try:
        compiled = jaxify(log_prob)
    except Exception as exc:
        raise RuntimeError("Failed to compile log_prob graph with jaxify") from exc

    if compiled is None:
        raise ValueError("jaxify returned None")

    return compiled


def build_validation_inputs(
    model: Model,
    compiled: JaxifiedGraph,
) -> dict[str, Any]:
    """
    Build validation inputs for the compiled graph.
    """

    if model is None:
        raise ValueError("model must not be None")

    if compiled is None:
        raise ValueError("compiled graph must not be None")

    available_inputs = {
        **model.data,
        **model.free_params,
    }

    missing_inputs = [
        name
        for name in compiled.input_names
        if name not in available_inputs
    ]

    if missing_inputs:
        raise KeyError(
            "Compiled graph inputs are missing from model data/free parameters: "
            f"{missing_inputs}. Available inputs: {list(available_inputs.keys())}"
        )

    return {
        name: available_inputs[name]
        for name in compiled.input_names
    }
