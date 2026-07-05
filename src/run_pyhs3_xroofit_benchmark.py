"""PyHS3 vs xRooFit NLL scan benchmark.

This benchmark compares NLL evaluation for a PyHS3 JSON workspace and a
matching RooFit/xRooFit ROOT workspace.  It is intentionally configurable so it
can be run on different workspace pairs by passing the analysis, target, data,
POI, and xRooFit model names on the command line.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

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

try:
    import ROOT
except ImportError:  # pragma: no cover - environment dependent
    ROOT = None


BENCHMARK_NAME = "pyhs3_xroofit_benchmark"
DEFAULT_OUTPUT = RESULTS_DIR / BENCHMARK_NAME / f"{BENCHMARK_NAME}_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME
DEFAULT_DELTA_TOLERANCE = 5e-8
DEFAULT_MINIMUM_TOLERANCE = 1e-12

FRAMEWORK_STYLE = {
    "pyhs3": {"label": "PyHS3", "color": "#0055A4", "marker": "s", "linestyle": "-"},
    "xroofit": {
        "label": "xRooFit",
        "color": "#CC6677",
        "marker": "D",
        "linestyle": "--",
    },
}


@dataclass(frozen=True)
class PyHS3Case:
    model: Any
    target: str
    params: dict[str, Any]


@dataclass(frozen=True)
class XRooFitCase:
    root_file: Any
    workspace: Any
    root_node: Any
    model_node: Any
    nll: Any
    poi: str


@dataclass(frozen=True)
class FrameworkSpec:
    name: str
    build_func: Callable[[], Any]
    eval_func: Callable[[Any, float], float]


def validate_existing_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def validate_positive_int(value: int, name: str, minimum: int = 1) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def validate_finite_float(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def validate_scan_config(
    *,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
    n_runs: int,
    delta_tolerance: float,
    minimum_tolerance: float,
) -> None:
    validate_finite_float(scan_min, "scan_min")
    validate_finite_float(scan_max, "scan_max")
    validate_finite_float(delta_tolerance, "delta_tolerance")
    validate_finite_float(minimum_tolerance, "minimum_tolerance")
    validate_positive_int(n_scan_points, "n_scan_points", minimum=2)
    validate_positive_int(n_runs, "n_runs", minimum=1)
    if scan_min >= scan_max:
        raise ValueError(
            f"scan_min must be smaller than scan_max, got {scan_min} >= {scan_max}"
        )
    if delta_tolerance <= 0.0:
        raise ValueError("delta_tolerance must be positive")
    if minimum_tolerance <= 0.0:
        raise ValueError("minimum_tolerance must be positive")


def channel_from_analysis(analysis_name: str) -> str:
    if not analysis_name.startswith("L_"):
        raise ValueError(
            "Cannot infer a channel from analysis name. Pass --target and --pyhs3-data-name explicitly, "
            f"or use an analysis name like L_ch0. Got: {analysis_name}"
        )
    return analysis_name.replace("L_", "", 1)


def default_target_from_analysis(analysis_name: str) -> str:
    return f"model_{channel_from_analysis(analysis_name)}"


def default_data_name_from_analysis(analysis_name: str) -> str:
    return f"combData_{channel_from_analysis(analysis_name)}"


def extract_parameter_point(
    workspace: Workspace, parameter_point: str | None
) -> dict[str, float]:
    try:
        points = workspace.parameter_points.root
    except AttributeError as exc:
        raise ValueError(
            "PyHS3 workspace does not contain parameter_points.root"
        ) from exc

    if not points:
        raise ValueError("PyHS3 workspace does not contain any parameter points")

    selected = None
    if parameter_point is None:
        selected = points[0]
    else:
        selected = next(
            (point for point in points if point.name == parameter_point), None
        )
        if selected is None:
            available = [getattr(point, "name", "<unnamed>") for point in points]
            raise KeyError(
                f"Could not find parameter point {parameter_point!r}. Available: {available}"
            )

    params: dict[str, float] = {}
    for parameter in selected.parameters:
        try:
            params[parameter.name] = float(parameter.value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Parameter {parameter.name!r} cannot be converted to float"
            ) from exc
    return params


def get_pyhs3_data_values(
    workspace: Workspace, data_name: str, observable_index: int = 0
) -> np.ndarray:
    try:
        data_entries = workspace.data.root
    except AttributeError as exc:
        raise ValueError("PyHS3 workspace does not contain data.root") from exc

    for data in data_entries:
        if data.name == data_name:
            values = np.asarray(
                [entry[observable_index] for entry in data.entries], dtype=np.float64
            )
            if values.size == 0:
                raise ValueError(f"PyHS3 data {data_name!r} is empty")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"PyHS3 data {data_name!r} contains non-finite values")
            return values

    available = [getattr(data, "name", "<unnamed>") for data in data_entries]
    raise KeyError(
        f"Could not find PyHS3 data {data_name!r}. Available data: {available}"
    )


def build_pyhs3_case(
    *,
    json_path: Path,
    analysis_name: str,
    target: str,
    data_name: str,
    poi: str,
    parameter_point: str | None,
    observable_name: str,
    observable_index: int,
) -> PyHS3Case:
    validate_existing_file(json_path, "PyHS3 JSON workspace")
    workspace = Workspace.load(json_path)
    model = workspace.model(analysis_name, progress=False, mode="FAST_RUN")

    params = extract_parameter_point(workspace, parameter_point)
    try:
        free_params = model.free_params
    except AttributeError:
        free_params = {}
    for name, value in free_params.items():
        params[name] = float(np.asarray(value))

    params[observable_name] = get_pyhs3_data_values(
        workspace, data_name, observable_index
    )
    if poi not in params and poi not in free_params:
        raise KeyError(f"POI {poi!r} is not present in PyHS3 parameters/free_params")

    return PyHS3Case(model=model, target=target, params=params)


def pyhs3_nll(case: PyHS3Case, poi: str, value: float) -> float:
    validate_finite_float(value, poi)
    eval_params = dict(case.params)
    eval_params[poi] = np.asarray(value, dtype=np.float64)
    logpdf = np.asarray(case.model.logpdf(case.target, **eval_params), dtype=np.float64)
    if logpdf.size == 0:
        raise ValueError(f"PyHS3 returned an empty logpdf array for {case.target}")
    if not np.all(np.isfinite(logpdf)):
        raise ValueError(f"PyHS3 returned non-finite logpdf values for {case.target}")
    return -float(np.sum(logpdf))


def require_xroofit(xroofit_library: str | None = "libxRooFit") -> Any:
    if ROOT is None:
        raise RuntimeError("ROOT is not available in this environment")

    if xroofit_library:
        load_status = int(ROOT.gSystem.Load(xroofit_library))
        if load_status < 0 and not hasattr(ROOT, "xRooNode"):
            raise RuntimeError(
                "Could not load xRooFit. Run `source external/xroofit-build/setup.sh` "
                "or pass --xroofit-library /path/to/libxRooFit.so."
            )

    if not hasattr(ROOT, "xRooNode"):
        raise RuntimeError(
            "xRooFit is not available in this ROOT/PyROOT session. "
            "Build xRooFit and source its setup.sh before running this benchmark."
        )
    return ROOT


def build_xroofit_case(
    *,
    root_path: Path,
    workspace_name: str,
    model_name: str,
    dataset_name: str,
    poi: str,
    xroofit_library: str | None,
) -> XRooFitCase:
    root = require_xroofit(xroofit_library)
    validate_existing_file(root_path, "RooFit ROOT workspace")

    root_file = root.TFile.Open(str(root_path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file {root_path}")

    workspace = root_file.Get(workspace_name)
    if workspace is None:
        root_file.Close()
        raise RuntimeError(
            f"Could not find RooWorkspace {workspace_name!r} in {root_path}"
        )

    root_node = root.xRooNode(workspace)
    model_node = root_node[model_name]
    if not model_node:
        root_file.Close()
        raise RuntimeError(f"Could not access xRooFit model node {model_name!r}")

    nll = model_node.nll(dataset_name)

    if not nll:
        raise RuntimeError(
            f"xRooFit returned a null NLL for model {model_name!r} "
            f"and dataset {dataset_name!r}. Try using the top-level model "
            "or ModelConfig, and verify the dataset is compatible."
        )

    return XRooFitCase(
        root_file=root_file,
        workspace=workspace,
        root_node=root_node,
        model_node=model_node,
        nll=nll,
        poi=poi,
    )


def _set_xroofit_parameter(case: XRooFitCase, value: float) -> None:
    # xRooNode wraps ROOT objects and forwards RooAbsArg-like methods.  Different
    # xRooFit builds expose the parameter either via model_node.pars()[name] or
    # nll.pars()[name], so try both before failing with a useful message.
    errors: list[str] = []
    for owner_name, owner in (("model", case.model_node), ("nll", case.nll)):
        try:
            parameter = owner.pars()[case.poi]
            parameter.setVal(float(value))
            return
        except Exception as exc:  # noqa: BLE001 - xRooFit/PyROOT can throw varied exception types
            errors.append(f"{owner_name}: {exc}")
    raise RuntimeError(
        f"Could not set xRooFit POI {case.poi!r}. Tried model.pars() and nll.pars(). "
        f"Errors: {'; '.join(errors)}"
    )


def xroofit_nll(case: XRooFitCase, value: float) -> float:
    validate_finite_float(value, case.poi)
    _set_xroofit_parameter(case, value)
    try:
        nll_value = float(case.nll.getVal())
    except AttributeError:
        nll_value = float(case.nll)
    if not math.isfinite(nll_value):
        raise ValueError(f"xRooFit returned non-finite NLL value: {nll_value}")
    return nll_value


def close_case(case: Any) -> None:
    if isinstance(case, XRooFitCase):
        try:
            case.root_file.Close()
        except Exception:
            pass


def validate_scan_values(values: list[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} contains non-finite values")


def delta_nll(values: list[float]) -> list[float]:
    validate_scan_values(values, "nll_values")
    minimum = min(values)
    return [float(value - minimum) for value in values]


def minimum_position(scan_values: list[float], nll_values: list[float]) -> float:
    if len(scan_values) != len(nll_values):
        raise ValueError("scan_values and nll_values must have the same length")
    return float(scan_values[int(np.argmin(np.asarray(nll_values, dtype=float)))])


def summarize_timings(values: list[float]) -> dict[str, float]:
    validate_scan_values(values, "timings")
    return {
        "mean_seconds": mean(values),
        "std_seconds": stdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def scan_nll(
    func: Callable[[float], float], scan_values: list[float]
) -> tuple[list[float], float]:
    values: list[float] = []
    start = time.perf_counter()
    for scan_value in scan_values:
        values.append(float(func(scan_value)))
    duration = time.perf_counter() - start
    validate_scan_values(values, "scan_nll_values")
    return values, duration


def measure_framework(
    *,
    name: str,
    build_func: Callable[[], Any],
    eval_func: Callable[[Any, float], float],
    scan_values: list[float],
    n_runs: int,
    poi_value: float,
) -> dict[str, Any]:
    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()
    case = None
    try:
        build_start = time.perf_counter()
        case = build_func()
        build_time = time.perf_counter() - build_start

        cold_start = time.perf_counter()
        first_nll = float(eval_func(case, poi_value))
        cold_time = time.perf_counter() - cold_start
        if not math.isfinite(first_nll):
            raise ValueError(f"{name} returned non-finite first NLL: {first_nll}")

        warm_timings: list[float] = []
        warm_nll = first_nll
        for _ in range(n_runs):
            start = time.perf_counter()
            warm_nll = float(eval_func(case, poi_value))
            duration = time.perf_counter() - start
            if duration <= 0.0 or not math.isfinite(duration):
                raise ValueError(f"{name} warm timing is invalid: {duration}")
            if not math.isfinite(warm_nll):
                raise ValueError(f"{name} returned non-finite warm NLL: {warm_nll}")
            warm_timings.append(duration)

        scan_values_nll, scan_time = scan_nll(lambda v: eval_func(case, v), scan_values)
        current_rss_after_mb = get_current_rss_mb()
        peak_rss_after_mb = get_peak_rss_mb()
        delta_shape = delta_nll(scan_values_nll)

        return {
            "framework": name,
            "framework_label": FRAMEWORK_STYLE[name]["label"],
            "status": "success",
            "n_runs": n_runs,
            "n_scan_points": len(scan_values),
            "build_time_seconds": build_time,
            "cold_first_evaluation_time_seconds": cold_time,
            "warm_nll": warm_nll,
            "warm_evaluation": summarize_timings(warm_timings),
            "warm_evaluation_time_seconds_mean": mean(warm_timings),
            "scan_time_seconds": scan_time,
            "time_per_scan_point_seconds": scan_time / len(scan_values),
            "current_rss_before_mb": current_rss_before_mb,
            "current_rss_after_mb": current_rss_after_mb,
            "current_rss_delta_mb": max(
                0.0, current_rss_after_mb - current_rss_before_mb
            ),
            "peak_rss_before_mb": peak_rss_before_mb,
            "peak_rss_after_mb": peak_rss_after_mb,
            "peak_rss_delta_mb": max(0.0, peak_rss_after_mb - peak_rss_before_mb),
            "first_nll": first_nll,
            "scan_nll_values": scan_values_nll,
            "delta_nll_shape": delta_shape,
            "minimum_poi": minimum_position(scan_values, scan_values_nll),
            "minimum_index": int(np.argmin(np.asarray(scan_values_nll, dtype=float))),
            "finite_values": bool(
                math.isfinite(first_nll)
                and math.isfinite(warm_nll)
                and all(math.isfinite(v) for v in scan_values_nll)
            ),
        }
    finally:
        close_case(case)


def add_agreement(
    pyhs3_result: dict[str, Any],
    xroofit_result: dict[str, Any],
    delta_tolerance: float,
    minimum_tolerance: float,
) -> dict[str, Any]:
    pyhs3_scan = np.asarray(pyhs3_result["scan_nll_values"], dtype=np.float64)
    xroofit_scan = np.asarray(xroofit_result["scan_nll_values"], dtype=np.float64)
    if pyhs3_scan.shape != xroofit_scan.shape:
        raise ValueError("Cannot compare scans with different shapes")

    pyhs3_delta = np.asarray(pyhs3_result["delta_nll_shape"], dtype=np.float64)
    xroofit_delta = np.asarray(xroofit_result["delta_nll_shape"], dtype=np.float64)
    raw_diff = xroofit_scan - pyhs3_scan
    delta_diff = xroofit_delta - pyhs3_delta
    delta_max = float(np.max(np.abs(delta_diff)))
    minimum_diff = abs(
        float(xroofit_result["minimum_poi"]) - float(pyhs3_result["minimum_poi"])
    )

    return {
        "raw_nll_abs_diff": abs(
            float(xroofit_result["first_nll"]) - float(pyhs3_result["first_nll"])
        ),
        "raw_scan_max_abs_diff": float(np.max(np.abs(raw_diff))),
        "raw_scan_mean_abs_diff": float(np.mean(np.abs(raw_diff))),
        "constant_offset_estimate": float(np.mean(raw_diff)),
        "delta_nll_max_abs_diff": delta_max,
        "minimum_poi_abs_diff": minimum_diff,
        "minimum_index_match": bool(
            pyhs3_result["minimum_index"] == xroofit_result["minimum_index"]
        ),
        "delta_shape_success": delta_max <= delta_tolerance,
        "minimum_poi_success": minimum_diff <= minimum_tolerance,
        "validation_status": "success"
        if delta_max <= delta_tolerance and minimum_diff <= minimum_tolerance
        else "failed",
        "delta_nll_difference": delta_diff.tolist(),
        "raw_nll_difference": raw_diff.tolist(),
    }


def failed_framework_result(name: str, exc: BaseException) -> dict[str, Any]:
    return {
        "framework": name,
        "framework_label": FRAMEWORK_STYLE.get(name, {"label": name})["label"],
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "axes.titlesize": 17,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def make_profile_plot(
    results: dict[str, dict[str, Any]], scan_values: list[float], output_path: Path
) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.asarray(scan_values, dtype=float)
    for name in ("pyhs3", "xroofit"):
        result = results[name]
        style = FRAMEWORK_STYLE[name]
        ax.plot(
            x,
            result["delta_nll_shape"],
            label=style["label"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=4,
            markevery=max(1, len(x) // 20),
        )
    ax.set_xlabel("Parameter of interest")
    ax.set_ylabel(r"$\Delta$NLL")
    ax.set_title("PyHS3 vs xRooFit NLL profile", loc="left", weight="bold")
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def make_runtime_plot(results: dict[str, dict[str, Any]], output_path: Path) -> None:
    _apply_plot_style()
    labels = [FRAMEWORK_STYLE[name]["label"] for name in ("pyhs3", "xroofit")]
    values = [
        results[name]["time_per_scan_point_seconds"] * 1e6
        for name in ("pyhs3", "xroofit")
    ]
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    bars = ax.bar(labels, values, edgecolor="black", alpha=0.9)
    ax.set_ylabel("Time per scan point [µs]")
    ax.set_title("NLL scan throughput", loc="left", weight="bold")
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(False, axis="x")
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.03,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            weight="bold",
        )
    _save_figure(fig, output_path)


def make_agreement_plot(
    agreement: dict[str, Any], delta_tolerance: float, output_path: Path
) -> None:
    _apply_plot_style()
    values = [agreement["delta_nll_max_abs_diff"], agreement["minimum_poi_abs_diff"]]
    labels = ["max ΔNLL diff", "minimum POI diff"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    plot_values = [max(v, 1e-16) for v in values]
    bars = ax.bar(labels, plot_values, edgecolor="black", alpha=0.9)
    ax.axhline(
        delta_tolerance,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"ΔNLL tol = {delta_tolerance:.0e}",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Absolute difference")
    ax.set_title("PyHS3 vs xRooFit numerical agreement", loc="left", weight="bold")
    ax.legend(frameon=False)
    for bar, raw in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f"{raw:.2e}",
            ha="center",
            va="bottom",
            weight="bold",
        )
    _save_figure(fig, output_path)


def make_plots(output_data: dict[str, Any], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    results_by_framework = {
        result["framework"]: result
        for result in output_data["results"]
        if result.get("status") == "success"
    }
    if "pyhs3" not in results_by_framework or "xroofit" not in results_by_framework:
        print(
            "Skipping plots because both PyHS3 and xRooFit did not complete successfully"
        )
        return
    make_profile_plot(
        results_by_framework,
        output_data["scan_values"],
        plot_dir / "pyhs3_xroofit_profile.png",
    )
    make_runtime_plot(results_by_framework, plot_dir / "pyhs3_xroofit_runtime.png")
    make_agreement_plot(
        output_data["agreement"],
        output_data["delta_tolerance"],
        plot_dir / "pyhs3_xroofit_agreement.png",
    )


def print_result(result: dict[str, Any]) -> None:
    print()
    print("-" * 72)
    print(result.get("framework_label", result.get("framework")))
    print("-" * 72)
    print(f"status:                 {result.get('status')}")
    if result.get("status") != "success":
        print(
            f"error:                  {result.get('error_type')}: {result.get('error_message')}"
        )
        return
    print(f"first NLL:              {result['first_nll']:.15f}")
    print(f"minimum POI:            {result['minimum_poi']:.15f}")
    print(f"build time:             {result['build_time_seconds'] * 1000.0:.3f} ms")
    print(
        f"cold first eval:        {result['cold_first_evaluation_time_seconds'] * 1e6:.3f} µs"
    )
    print(
        f"warm eval:              {result['warm_evaluation_time_seconds_mean'] * 1e6:.3f} µs"
    )
    print(f"scan time:              {result['scan_time_seconds'] * 1000.0:.3f} ms")
    print(
        f"time per scan point:    {result['time_per_scan_point_seconds'] * 1e6:.3f} µs"
    )
    print(f"current RSS delta:      {result['current_rss_delta_mb']:.3f} MB")
    print(f"peak RSS delta:         {result['peak_rss_delta_mb']:.3f} MB")


def run(
    *,
    json_path: Path,
    root_path: Path,
    analysis_name: str,
    target: str | None,
    pyhs3_data_name: str | None,
    xroofit_model_name: str | None,
    xroofit_dataset_name: str,
    root_workspace_name: str,
    poi: str,
    parameter_point: str | None,
    observable_name: str,
    observable_index: int,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
    n_runs: int,
    output: Path,
    plot: bool,
    plot_dir: Path,
    delta_tolerance: float,
    minimum_tolerance: float,
    xroofit_library: str | None,
) -> dict[str, Any]:
    validate_scan_config(
        scan_min=scan_min,
        scan_max=scan_max,
        n_scan_points=n_scan_points,
        n_runs=n_runs,
        delta_tolerance=delta_tolerance,
        minimum_tolerance=minimum_tolerance,
    )
    validate_existing_file(json_path, "PyHS3 JSON workspace")
    validate_existing_file(root_path, "RooFit ROOT workspace")

    resolved_target = target or default_target_from_analysis(analysis_name)
    resolved_pyhs3_data = pyhs3_data_name or default_data_name_from_analysis(
        analysis_name
    )
    resolved_xroofit_model = xroofit_model_name or resolved_target

    scan_values = [float(v) for v in np.linspace(scan_min, scan_max, n_scan_points)]
    specs = [
        FrameworkSpec(
            name="pyhs3",
            build_func=lambda: build_pyhs3_case(
                json_path=json_path,
                analysis_name=analysis_name,
                target=resolved_target,
                data_name=resolved_pyhs3_data,
                poi=poi,
                parameter_point=parameter_point,
                observable_name=observable_name,
                observable_index=observable_index,
            ),
            eval_func=lambda case, value: pyhs3_nll(case, poi, value),
        ),
        FrameworkSpec(
            name="xroofit",
            build_func=lambda: build_xroofit_case(
                root_path=root_path,
                workspace_name=root_workspace_name,
                model_name=resolved_xroofit_model,
                dataset_name=xroofit_dataset_name,
                poi=poi,
                xroofit_library=xroofit_library,
            ),
            eval_func=lambda case, value: xroofit_nll(case, value),
        ),
    ]

    results: list[dict[str, Any]] = []
    for spec in specs:
        try:
            results.append(
                measure_framework(
                    name=spec.name,
                    build_func=spec.build_func,
                    eval_func=spec.eval_func,
                    scan_values=scan_values,
                    n_runs=n_runs,
                    poi_value=1.0,
                )
            )
        except Exception as exc:  # noqa: BLE001 - preserve partial benchmark output
            results.append(failed_framework_result(spec.name, exc))

    successful = {
        result["framework"]: result
        for result in results
        if result.get("status") == "success"
    }
    agreement: dict[str, Any] | None = None
    if "pyhs3" in successful and "xroofit" in successful:
        agreement = add_agreement(
            successful["pyhs3"],
            successful["xroofit"],
            delta_tolerance=delta_tolerance,
            minimum_tolerance=minimum_tolerance,
        )
        status = "success" if agreement["validation_status"] == "success" else "failed"
    else:
        status = "failed"
        agreement = {"validation_status": "not_run"}

    output_data = {
        "benchmark": BENCHMARK_NAME,
        "json_path": str(json_path),
        "root_path": str(root_path),
        "analysis_name": analysis_name,
        "target": resolved_target,
        "pyhs3_data_name": resolved_pyhs3_data,
        "xroofit_model_name": resolved_xroofit_model,
        "xroofit_dataset_name": xroofit_dataset_name,
        "root_workspace_name": root_workspace_name,
        "poi": poi,
        "parameter_point": parameter_point,
        "observable_name": observable_name,
        "observable_index": observable_index,
        "scan_min": scan_min,
        "scan_max": scan_max,
        "n_scan_points": n_scan_points,
        "n_runs": n_runs,
        "delta_tolerance": delta_tolerance,
        "minimum_tolerance": minimum_tolerance,
        "scan_values": scan_values,
        "frameworks": ["pyhs3", "xroofit"],
        "status": status,
        "agreement": agreement,
        "results": results,
    }

    print("=" * 80)
    print("PyHS3 vs xRooFit benchmark")
    print("=" * 80)
    print(f"PyHS3 JSON:      {json_path}")
    print(f"ROOT workspace:  {root_path}")
    print(f"Analysis:        {analysis_name}")
    print(f"Target/model:    {resolved_target}")
    print(f"PyHS3 data:      {resolved_pyhs3_data}")
    print(f"xRooFit model:   {resolved_xroofit_model}")
    print(f"xRooFit data:    {xroofit_dataset_name}")
    print(f"POI:             {poi}")
    print(f"Grid:            [{scan_min}, {scan_max}] with {n_scan_points} points")
    print(f"Status:          {status}")
    for result in results:
        print_result(result)
    print()
    print("agreement")
    print(f"  validation:       {agreement.get('validation_status')}")
    if agreement.get("validation_status") != "not_run":
        print(f"  raw max diff:     {agreement['raw_scan_max_abs_diff']:.15e}")
        print(f"  delta max diff:   {agreement['delta_nll_max_abs_diff']:.15e}")
        print(f"  constant offset:  {agreement['constant_offset_estimate']:.15e}")
        print(f"  minimum POI diff: {agreement['minimum_poi_abs_diff']:.15e}")

    save_json(output_data, output)
    print(f"Saved result to {output}")
    if plot:
        make_plots(output_data, plot_dir)
        print(f"Saved plots to {plot_dir}")
    return output_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a configurable PyHS3 vs xRooFit NLL scan benchmark."
    )
    parser.add_argument("--json-workspace", type=Path, required=True)
    parser.add_argument("--root-workspace", type=Path, required=True)
    parser.add_argument(
        "--analysis", default="L_ch0", help="PyHS3 analysis name, for example L_ch0."
    )
    parser.add_argument(
        "--target",
        default=None,
        help="PyHS3 logpdf target. Defaults to model_<channel> inferred from --analysis.",
    )
    parser.add_argument(
        "--pyhs3-data-name",
        default=None,
        help="PyHS3 data name. Defaults to combData_<channel> inferred from --analysis.",
    )
    parser.add_argument(
        "--xroofit-model-name",
        default=None,
        help="xRooFit model node. Defaults to --target.",
    )
    parser.add_argument("--xroofit-dataset-name", default="combData")
    parser.add_argument("--root-workspace-name", default="combWS")
    parser.add_argument("--poi", default="mu_sig")
    parser.add_argument("--parameter-point", default=None)
    parser.add_argument("--observable-name", default="x")
    parser.add_argument("--observable-index", type=int, default=0)
    parser.add_argument("--scan-min", type=float, default=0.0)
    parser.add_argument("--scan-max", type=float, default=2.0)
    parser.add_argument("--n-scan-points", type=int, default=101)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--delta-tolerance", type=float, default=DEFAULT_DELTA_TOLERANCE
    )
    parser.add_argument(
        "--minimum-tolerance", type=float, default=DEFAULT_MINIMUM_TOLERANCE
    )
    parser.add_argument(
        "--xroofit-library",
        default="libxRooFit",
        help="Library to load before using xRooFit. Use an absolute libxRooFit.so path or empty string to skip loading.",
    )
    return parser.parse_args()


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args_from(argv)
    run(
        json_path=args.json_workspace,
        root_path=args.root_workspace,
        analysis_name=args.analysis,
        target=args.target,
        pyhs3_data_name=args.pyhs3_data_name,
        xroofit_model_name=args.xroofit_model_name,
        xroofit_dataset_name=args.xroofit_dataset_name,
        root_workspace_name=args.root_workspace_name,
        poi=args.poi,
        parameter_point=args.parameter_point,
        observable_name=args.observable_name,
        observable_index=args.observable_index,
        scan_min=args.scan_min,
        scan_max=args.scan_max,
        n_scan_points=args.n_scan_points,
        n_runs=args.n_runs,
        output=args.output,
        plot=args.plot,
        plot_dir=args.plot_dir,
        delta_tolerance=args.delta_tolerance,
        minimum_tolerance=args.minimum_tolerance,
        xroofit_library=args.xroofit_library or None,
    )


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *argv]
        return parse_args()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
