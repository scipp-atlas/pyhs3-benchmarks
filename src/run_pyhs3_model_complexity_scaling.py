from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyhs3.workspace import Workspace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import PLOTS_DIR, RESULTS_DIR
    from src.utils import get_current_rss_mb, get_peak_rss_mb, save_json
else:
    from .config import PLOTS_DIR, RESULTS_DIR
    from .utils import get_current_rss_mb, get_peak_rss_mb, save_json


BENCHMARK_NAME = "pyhs3_model_complexity_scaling"
DEFAULT_WORKSPACES = [
    "simple_workspace_nonp.json",
    "simple_workspace_generic_nonp.json",
    "simple_workspace.json",
    "simple_workspace_generic.json",
]
DEFAULT_N_RUNS = 100
DEFAULT_N_SCAN_POINTS = 101
DEFAULT_SCAN_PARAMETER = "mu_sig"
DEFAULT_SCAN_MIN = 0.0
DEFAULT_SCAN_MAX = 2.0

FRAMEWORK_COLOR = "#0B5CAD"
REFERENCE_COLOR = "#222222"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_existing_dir(path: Path, name: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist or is not a directory: {path}")
    return path


def validate_existing_file(path: Path, name: str) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")
    return path


def validate_positive_int(value: int, name: str, *, minimum: int = 1) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def validate_finite_float(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")


def validate_scan_values(values: Iterable[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


def validate_benchmark_config(
    *,
    input_dir: Path,
    workspace_names: list[str],
    n_runs: int,
    scan_parameter: str,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
) -> None:
    validate_existing_dir(input_dir, "Input directory")
    if not workspace_names:
        raise ValueError("At least one workspace must be provided")
    if not scan_parameter:
        raise ValueError("scan_parameter must not be empty")
    validate_positive_int(n_runs, "n_runs", minimum=1)
    validate_positive_int(n_scan_points, "n_scan_points", minimum=2)
    validate_finite_float(scan_min, "scan_min")
    validate_finite_float(scan_max, "scan_max")
    if scan_min >= scan_max:
        raise ValueError(
            f"scan_min must be smaller than scan_max, got {scan_min} >= {scan_max}"
        )


def validate_measurement_result(result: dict[str, Any]) -> None:
    required_positive = [
        "load_time_seconds",
        "build_time_seconds",
        "cold_first_evaluation_time_seconds",
        "warm_evaluation_time_seconds_mean",
        "scan_time_seconds",
        "time_per_scan_point_seconds",
    ]
    for key in required_positive:
        value = float(result[key])
        if not math.isfinite(value):
            raise ValueError(f"{key} is not finite")
        if (
            key
            in {
                "cold_first_evaluation_time_seconds",
                "warm_evaluation_time_seconds_mean",
                "scan_time_seconds",
                "time_per_scan_point_seconds",
            }
            and value <= 0.0
        ):
            raise ValueError(f"{key} must be positive")
        if key in {"load_time_seconds", "build_time_seconds"} and value < 0.0:
            raise ValueError(f"{key} must be non-negative")
    if not result.get("finite_values"):
        raise ValueError("NLL evaluation produced non-finite values")


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def channel_from_analysis(analysis_name: str) -> str:
    if not analysis_name.startswith("L_"):
        raise ValueError(f"Analysis name must start with 'L_', got {analysis_name!r}")
    channel = analysis_name.removeprefix("L_")
    if not channel:
        raise ValueError(f"Analysis name does not contain a channel: {analysis_name!r}")
    return channel


def target_from_analysis(analysis_name: str) -> str:
    return f"model_{channel_from_analysis(analysis_name)}"


def discover_analyses(workspace: Workspace) -> list[str]:
    analyses = getattr(getattr(workspace, "analyses", None), "root", None)
    if analyses is None:
        raise ValueError("Workspace does not contain a valid analyses section")
    names = [analysis.name for analysis in analyses]
    if not names:
        raise ValueError("Workspace does not contain any analyses")
    return names


def get_x_data(workspace: Workspace, analysis_name: str) -> np.ndarray:
    channel = channel_from_analysis(analysis_name)
    data_name = f"combData_{channel}"
    data_entries = getattr(getattr(workspace, "data", None), "root", None)
    if data_entries is None:
        raise ValueError("Workspace does not contain a valid data section")

    for data in data_entries:
        if getattr(data, "name", None) == data_name:
            values = np.asarray([entry[0] for entry in data.entries], dtype=np.float64)
            if values.size == 0:
                raise ValueError(f"Dataset {data_name!r} is empty")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Dataset {data_name!r} contains non-finite x values")
            return values

    raise KeyError(f"Could not find dataset {data_name!r}")


def get_eval_params(model: Any, x: np.ndarray) -> dict[str, Any]:
    free_params = getattr(model, "free_params", None)
    if not isinstance(free_params, dict):
        raise ValueError("PyHS3 model does not expose free_params as a dictionary")

    params: dict[str, Any] = {}
    for name, value in free_params.items():
        array = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Parameter {name!r} contains non-finite values")
        params[name] = array
    params["x"] = x
    return params


def evaluate_unbinned_nll(model: Any, target: str, params: dict[str, Any]) -> float:
    values = np.asarray(model.logpdf(target, **params), dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"logpdf for {target!r} returned no values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"logpdf for {target!r} returned non-finite values")
    nll = -float(np.sum(values))
    if not math.isfinite(nll):
        raise ValueError(f"NLL for {target!r} is non-finite")
    return nll


def summarize_timings(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Timing list must not be empty")
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Timing list contains invalid values")
    return {
        "mean_seconds": mean(values),
        "std_seconds": stdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def time_repeated(func: Any, *, n_runs: int) -> tuple[float, list[float]]:
    timings: list[float] = []
    value = float("nan")
    for _ in range(n_runs):
        start = time.perf_counter()
        value = float(func())
        end = time.perf_counter()
        if not math.isfinite(value):
            raise ValueError("Repeated evaluation produced a non-finite value")
        timings.append(end - start)
    return value, timings


def scan_nll(
    eval_func: Any,
    scan_parameter: str,
    scan_values: list[float],
    params: dict[str, Any],
) -> tuple[list[float], float]:
    values: list[float] = []
    start = time.perf_counter()
    for value in scan_values:
        eval_params = dict(params)
        eval_params[scan_parameter] = np.asarray(value, dtype=np.float64)
        values.append(float(eval_func(eval_params)))
    end = time.perf_counter()
    validate_scan_values(values, "scan_nll_values")
    return values, end - start


def delta_nll(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(validate_scan_values(values, "nll_values"), dtype=np.float64)
    return array - float(np.min(array))


def minimum_position(
    scan_values: list[float], nll_values: list[float]
) -> tuple[int, float]:
    if len(scan_values) != len(nll_values):
        raise ValueError("scan_values and nll_values must have the same length")
    index = int(np.argmin(np.asarray(nll_values, dtype=np.float64)))
    return index, float(scan_values[index])


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_workspace_analysis(
    *,
    workspace_path: Path,
    analysis_name: str,
    n_runs: int,
    scan_parameter: str,
    scan_values: list[float],
) -> dict[str, Any]:
    validate_existing_file(workspace_path, "Workspace file")
    scan_values = validate_scan_values(scan_values, "scan_values")

    rss_before = get_current_rss_mb()
    peak_before = get_peak_rss_mb()

    load_start = time.perf_counter()
    workspace = Workspace.load(workspace_path)
    load_end = time.perf_counter()

    build_start = time.perf_counter()
    model = workspace.model(analysis_name, progress=False, mode="FAST_RUN")
    build_end = time.perf_counter()

    x = get_x_data(workspace, analysis_name)
    target = target_from_analysis(analysis_name)
    params = get_eval_params(model, x)

    def eval_func(eval_params: dict[str, Any]) -> float:
        return evaluate_unbinned_nll(model, target, eval_params)

    cold_start = time.perf_counter()
    first_nll = eval_func(params)
    cold_end = time.perf_counter()

    warm_nll, warm_timings = time_repeated(lambda: eval_func(params), n_runs=n_runs)
    scan_nll_values, scan_time = scan_nll(
        eval_func, scan_parameter, scan_values, params
    )
    delta_shape = delta_nll(scan_nll_values).tolist()

    rss_after = get_current_rss_mb()
    peak_after = get_peak_rss_mb()

    minimum_index, minimum_parameter_value = minimum_position(
        scan_values, scan_nll_values
    )
    finite_values = bool(
        math.isfinite(first_nll)
        and math.isfinite(warm_nll)
        and all(math.isfinite(value) for value in scan_nll_values)
    )

    result = {
        "workspace": workspace_path.name,
        "analysis": analysis_name,
        "framework": "pyhs3",
        "plot_label": _case_label(workspace_path.stem, analysis_name),
        "target": target,
        "data_points": int(len(x)),
        "load_time_seconds": load_end - load_start,
        "build_time_seconds": build_end - build_start,
        "cold_first_evaluation_time_seconds": cold_end - cold_start,
        "warm_evaluation": summarize_timings(warm_timings),
        "warm_evaluation_time_seconds_mean": mean(warm_timings),
        "scan_time_seconds": scan_time,
        "time_per_scan_point_seconds": scan_time / len(scan_values),
        "current_rss_before_mb": rss_before,
        "current_rss_after_mb": rss_after,
        "current_rss_delta_mb": max(0.0, rss_after - rss_before),
        "peak_rss_before_mb": peak_before,
        "peak_rss_after_mb": peak_after,
        "peak_rss_delta_mb": max(0.0, peak_after - peak_before),
        "first_nll": first_nll,
        "warm_nll": warm_nll,
        "scan_nll_values": scan_nll_values,
        "delta_nll_shape": delta_shape,
        "scan_nll_min": min(scan_nll_values),
        "scan_nll_max": max(scan_nll_values),
        "minimum_index": minimum_index,
        "minimum_parameter_value": minimum_parameter_value,
        "finite_values": finite_values,
        "n_scan_points": len(scan_values),
        "n_runs": n_runs,
        "scan_parameter": scan_parameter,
        "status": "success",
    }
    validate_measurement_result(result)
    return result


def failed_result(
    workspace_name: str, analysis_name: str | None, exc: BaseException
) -> dict[str, Any]:
    return {
        "workspace": workspace_name,
        "analysis": analysis_name,
        "framework": "pyhs3",
        "target": target_from_analysis(analysis_name) if analysis_name else None,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _apply_cern_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 12,
            "axes.titlesize": 21,
            "axes.labelsize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "axes.linewidth": 1.4,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.grid": True,
            "grid.alpha": 0.35,
        }
    )


def _case_label(workspace_stem: str, analysis_name: str) -> str:
    channel = channel_from_analysis(analysis_name).replace("ch", "ch")
    name = workspace_stem.replace("simple_workspace_", "").replace(
        "simple_workspace", "base"
    )
    name = (
        name.replace("generic_nonp", "generic\nnonp")
        .replace("generic", "generic")
        .replace("nonp", "nonp")
    )
    return f"{name}\n{channel}"


def _successful_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if result.get("status") == "success"]


def _plot_floor(values: Iterable[float], *, floor: float = 1e-6) -> list[float]:
    return [max(float(value), floor) for value in values]


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, bbox_inches="tight")
    except OSError as exc:
        raise OSError(f"Failed to save plot to {output_path}: {exc}") from exc
    finally:
        plt.close(fig)


def make_runtime_scaling_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    if not successful:
        raise ValueError("No successful results available for runtime plot")

    labels = [result["plot_label"] for result in successful]
    values = [result["time_per_scan_point_seconds"] * 1e6 for result in successful]
    x = np.arange(len(successful))

    fig, ax = plt.subplots(figsize=(14.0, 5.8))
    bars = ax.bar(
        x,
        values,
        color=FRAMEWORK_COLOR,
        edgecolor="black",
        linewidth=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Time per scan point [μs]")
    ax.set_title(
        "PyHS3 model-complexity scaling: steady-state NLL evaluation",
        loc="left",
        weight="bold",
    )

    ymin = min(values) * 0.95
    ymax = max(values) * 1.12
    ax.set_ylim(ymin, ymax)

    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(False, axis="x")

    label_offset = (ymax - ymin) * 0.015
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + label_offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
            weight="bold",
        )

    _save_figure(fig, output_path)


def make_timing_profile_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    if not successful:
        raise ValueError("No successful results available for timing profile plot")

    labels = [result["plot_label"] for result in successful]
    x = np.arange(len(successful))
    width = 0.18
    metrics = [
        (
            "Load [ms]",
            [result["load_time_seconds"] * 1000.0 for result in successful],
            "",
        ),
        (
            "Build [ms]",
            [result["build_time_seconds"] * 1000.0 for result in successful],
            "//",
        ),
        (
            "Cold eval [μs]",
            [
                result["cold_first_evaluation_time_seconds"] * 1e6
                for result in successful
            ],
            "xx",
        ),
        (
            "Warm eval [μs]",
            [
                result["warm_evaluation_time_seconds_mean"] * 1e6
                for result in successful
            ],
            "..",
        ),
    ]

    fig, ax = plt.subplots(figsize=(14.2, 6.7))
    for idx, (label, values, hatch) in enumerate(metrics):
        offset = (idx - 1.5) * width
        ax.bar(
            x + offset,
            _plot_floor(values, floor=1e-3),
            width,
            color=FRAMEWORK_COLOR,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.7,
            hatch=hatch,
            label=label,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Timing (mixed units; log scale)")
    ax.set_title(
        "PyHS3 timing profile across model complexity cases", loc="left", weight="bold"
    )
    ax.grid(True, which="both", axis="y", alpha=0.35)
    ax.grid(False, axis="x")
    ax.legend(
        title="Metric", frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0)
    )
    fig.subplots_adjust(right=0.82)
    _save_figure(fig, output_path)


def make_memory_scaling_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    if not successful:
        raise ValueError("No successful results available for memory plot")

    labels = [result["plot_label"] for result in successful]
    current = [result["current_rss_delta_mb"] for result in successful]
    peak = [result["peak_rss_delta_mb"] for result in successful]
    x = np.arange(len(successful))
    width = 0.34

    fig, ax = plt.subplots(figsize=(14.0, 5.8))
    ax.bar(
        x - width / 2,
        _plot_floor(current, floor=1e-3),
        width,
        color=FRAMEWORK_COLOR,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.7,
        label="Current RSS delta",
    )
    ax.bar(
        x + width / 2,
        _plot_floor(peak, floor=1e-3),
        width,
        color=FRAMEWORK_COLOR,
        alpha=0.45,
        edgecolor="black",
        linewidth=0.7,
        hatch="//",
        label="Peak RSS delta",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Memory delta [MB] (log scale)")
    ax.set_title(
        "PyHS3 memory footprint across model complexity cases",
        loc="left",
        weight="bold",
    )
    ax.grid(True, which="both", axis="y", alpha=0.35)
    ax.grid(False, axis="x")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.subplots_adjust(right=0.80)
    _save_figure(fig, output_path)


def make_profile_examples_plot(
    results: list[dict[str, Any]], scan_values: list[float], output_path: Path
) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    if not successful:
        raise ValueError("No successful results available for profile examples plot")

    selected = successful[: min(4, len(successful))]
    nrows = len(selected)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=1, figsize=(10.8, 2.7 * nrows), sharex=True
    )
    if nrows == 1:
        axes = [axes]

    for ax, result in zip(axes, selected, strict=True):
        ax.plot(
            scan_values,
            result["delta_nll_shape"],
            color=FRAMEWORK_COLOR,
            linewidth=2.0,
            label="PyHS3",
        )
        ax.axvline(
            result["minimum_parameter_value"],
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            label="minimum",
        )
        ax.set_ylabel("ΔNLL")
        ax.set_title(
            f"{result['workspace']} / {result['analysis']}",
            loc="left",
            fontsize=12,
            weight="bold",
        )
        ax.legend(frameon=False, loc="upper right")
        ax.grid(True, alpha=0.35)

    axes[-1].set_xlabel(f"Signal strength {selected[0]['scan_parameter']}")
    fig.suptitle(
        "Representative PyHS3 ΔNLL profiles",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=22,
        weight="bold",
    )
    fig.subplots_adjust(top=0.91 if nrows > 1 else 0.82, hspace=0.42)
    _save_figure(fig, output_path)


def make_summary_table_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    if not successful:
        raise ValueError("No successful results available for summary table")

    headers = [
        "Workspace",
        "Analysis",
        "Data",
        "Build [ms]",
        "Cold [ms]",
        "Warm [μs]",
        "Scan [ms]",
        "μ min",
        "RSS Δ [MB]",
    ]
    rows = []
    for result in successful:
        rows.append(
            [
                result["workspace"].replace(".json", ""),
                result["analysis"],
                f"{result['data_points']}",
                f"{result['build_time_seconds'] * 1000.0:.2f}",
                f"{result['cold_first_evaluation_time_seconds'] * 1000.0:.2f}",
                f"{result['warm_evaluation_time_seconds_mean'] * 1e6:.2f}",
                f"{result['scan_time_seconds'] * 1000.0:.2f}",
                f"{result['minimum_parameter_value']:.3f}",
                f"{result['current_rss_delta_mb']:.2f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(15.5, max(4.5, 0.45 * len(rows) + 2.8)))
    ax.axis("off")
    ax.text(
        0.0,
        1.06,
        "PyHS3 model-complexity scaling summary",
        transform=ax.transAxes,
        fontsize=22,
        weight="bold",
        ha="left",
    )
    ax.text(
        0.0,
        0.965,
        "Measures internal PyHS3 scaling across unbinned HS3 workspaces.",
        transform=ax.transAxes,
        fontsize=13,
        ha="left",
    )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#B8B8B8")
        if row == 0:
            cell.set_facecolor("#262626")
            cell.set_text_props(color="white", weight="bold")
        elif col in {0, 1}:
            cell.set_text_props(weight="bold")
    _save_figure(fig, output_path)


def make_plots(
    results: list[dict[str, Any]], scan_values: list[float], plot_dir: Path
) -> None:
    successful = _successful_results(results)
    if not successful:
        raise ValueError("No successful results available for plotting")
    plot_dir.mkdir(parents=True, exist_ok=True)
    make_runtime_scaling_plot(
        successful, plot_dir / "pyhs3_model_complexity_runtime_scaling.png"
    )
    make_timing_profile_plot(
        successful, plot_dir / "pyhs3_model_complexity_timing_profile.png"
    )
    make_memory_scaling_plot(
        successful, plot_dir / "pyhs3_model_complexity_memory_scaling.png"
    )
    make_profile_examples_plot(
        successful,
        scan_values,
        plot_dir / "pyhs3_model_complexity_profile_examples.png",
    )
    make_summary_table_plot(
        successful, plot_dir / "pyhs3_model_complexity_summary_table.png"
    )


# ---------------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------------


def print_result(result: dict[str, Any]) -> None:
    print()
    print("-" * 80)
    print(f"{result['workspace']} / {result['analysis']} / {result['target']}")
    print("-" * 80)
    print("status:                  success")
    print(f"data points:             {result['data_points']}")
    print(f"load time:               {result['load_time_seconds'] * 1000.0:.3f} ms")
    print(f"model build:             {result['build_time_seconds'] * 1000.0:.3f} ms")
    print(
        f"cold first evaluation:   {result['cold_first_evaluation_time_seconds'] * 1000.0:.3f} ms"
    )
    print(
        "warm evaluation:         "
        f"{result['warm_evaluation_time_seconds_mean'] * 1e6:.3f} us "
        f"± {result['warm_evaluation']['std_seconds'] * 1e6:.3f} us"
    )
    print(f"full scan:               {result['scan_time_seconds'] * 1000.0:.3f} ms")
    print(
        f"time per scan point:     {result['time_per_scan_point_seconds'] * 1e6:.3f} us"
    )
    print(f"current RSS delta:       {result['current_rss_delta_mb']:.3f} MB")
    print(f"peak RSS delta:          {result['peak_rss_delta_mb']:.3f} MB")
    print(
        f"minimum {result['scan_parameter']}:       {result['minimum_parameter_value']:.15f}"
    )
    print(f"finite values:           {result['finite_values']}")


def print_failed_result(result: dict[str, Any]) -> None:
    print()
    print("-" * 80)
    print(f"{result['workspace']} / {result.get('analysis')}")
    print("-" * 80)
    print("status:                  failed")
    print(
        f"error:                   {result.get('error_type')}: {result.get('error_message')}"
    )


def build_failed_output(
    *,
    input_dir: Path,
    workspace_names: list[str],
    n_runs: int,
    scan_parameter: str,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "framework": "pyhs3",
        "input_dir": str(input_dir),
        "workspace_names": workspace_names,
        "n_runs": n_runs,
        "scan_parameter": scan_parameter,
        "scan_min": scan_min,
        "scan_max": scan_max,
        "n_scan_points": n_scan_points,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "results": [],
    }


def run(
    *,
    input_dir: Path,
    workspace_names: list[str],
    n_runs: int = DEFAULT_N_RUNS,
    scan_parameter: str = DEFAULT_SCAN_PARAMETER,
    scan_min: float = DEFAULT_SCAN_MIN,
    scan_max: float = DEFAULT_SCAN_MAX,
    n_scan_points: int = DEFAULT_N_SCAN_POINTS,
    output: Path = RESULTS_DIR / BENCHMARK_NAME / f"{BENCHMARK_NAME}_result.json",
    plot: bool = False,
    plot_dir: Path = PLOTS_DIR / BENCHMARK_NAME,
    continue_on_case_error: bool = True,
) -> dict[str, Any]:
    try:
        validate_benchmark_config(
            input_dir=input_dir,
            workspace_names=workspace_names,
            n_runs=n_runs,
            scan_parameter=scan_parameter,
            scan_min=scan_min,
            scan_max=scan_max,
            n_scan_points=n_scan_points,
        )
        scan_values = np.linspace(scan_min, scan_max, n_scan_points).tolist()
        results: list[dict[str, Any]] = []

        for workspace_name in workspace_names:
            workspace_path = validate_existing_file(
                input_dir / workspace_name, "Workspace file"
            )
            try:
                workspace = Workspace.load(workspace_path)
                analyses = discover_analyses(workspace)
            except Exception as exc:
                result = failed_result(workspace_name, None, exc)
                results.append(result)
                if not continue_on_case_error:
                    raise
                continue

            for analysis_name in analyses:
                try:
                    results.append(
                        measure_workspace_analysis(
                            workspace_path=workspace_path,
                            analysis_name=analysis_name,
                            n_runs=n_runs,
                            scan_parameter=scan_parameter,
                            scan_values=scan_values,
                        )
                    )
                except Exception as exc:
                    results.append(failed_result(workspace_name, analysis_name, exc))
                    if not continue_on_case_error:
                        raise

        successful_results = _successful_results(results)
        failed_results = [
            result for result in results if result.get("status") != "success"
        ]
        status = "success" if successful_results and not failed_results else "failed"

        output_data = {
            "benchmark": BENCHMARK_NAME,
            "framework": "pyhs3",
            "input_dir": str(input_dir),
            "workspace_names": workspace_names,
            "n_runs": n_runs,
            "scan_parameter": scan_parameter,
            "scan_min": scan_min,
            "scan_max": scan_max,
            "n_scan_points": n_scan_points,
            "status": status,
            "successful_runs": len(successful_results),
            "total_runs": len(results),
            "successful_cases": [
                f"{r['workspace']}/{r['analysis']}" for r in successful_results
            ],
            "failed_cases": [
                f"{r['workspace']}/{r.get('analysis')}" for r in failed_results
            ],
            "notes": (
                "Measures PyHS3 internal scaling across unbinned HS3 workspaces. "
                "Unbinned NLL is computed as -sum(logpdf(target, x=data, ...free_params))."
            ),
            "results": results,
        }

        print("=" * 80)
        print("PyHS3 model-complexity scaling benchmark")
        print("=" * 80)
        print(f"Input dir:       {input_dir}")
        print(f"Workspaces:      {', '.join(workspace_names)}")
        print(
            f"Grid:            {scan_parameter} in [{scan_min}, {scan_max}] with {n_scan_points} points"
        )
        print(f"Warm runs:       {n_runs}")
        print(f"Status:          {status}")
        print(f"Successful:      {len(successful_results)} / {len(results)}")

        for result in results:
            if result.get("status") == "success":
                print_result(result)
            else:
                print_failed_result(result)

        save_json(output_data, output)
        print()
        print(f"Saved result to {output}")

        if plot:
            make_plots(successful_results, scan_values, plot_dir)
            print(f"Saved plots to {plot_dir}")

        return output_data

    except Exception as exc:
        failed_output = build_failed_output(
            input_dir=input_dir,
            workspace_names=workspace_names,
            n_runs=n_runs,
            scan_parameter=scan_parameter,
            scan_min=scan_min,
            scan_max=scan_max,
            n_scan_points=n_scan_points,
            exc=exc,
        )
        try:
            save_json(failed_output, output)
        except Exception as save_exc:  # pragma: no cover
            print(
                f"Failed to save benchmark failure report: {save_exc}", file=sys.stderr
            )
        raise RuntimeError("PyHS3 model-complexity scaling benchmark failed") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyHS3 internal model-complexity scaling on unbinned HS3 workspaces."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("inputs"))
    parser.add_argument("--workspaces", nargs="+", default=DEFAULT_WORKSPACES)
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--scan-parameter", type=str, default=DEFAULT_SCAN_PARAMETER)
    parser.add_argument("--scan-min", type=float, default=DEFAULT_SCAN_MIN)
    parser.add_argument("--scan-max", type=float, default=DEFAULT_SCAN_MAX)
    parser.add_argument("--n-scan-points", type=int, default=DEFAULT_N_SCAN_POINTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / BENCHMARK_NAME / f"{BENCHMARK_NAME}_result.json",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=PLOTS_DIR / BENCHMARK_NAME)
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first workspace or analysis failure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        input_dir=args.input_dir,
        workspace_names=args.workspaces,
        n_runs=args.n_runs,
        scan_parameter=args.scan_parameter,
        scan_min=args.scan_min,
        scan_max=args.scan_max,
        n_scan_points=args.n_scan_points,
        output=args.output,
        plot=args.plot,
        plot_dir=args.plot_dir,
        continue_on_case_error=not args.fail_fast,
    )


if __name__ == "__main__":
    main()
