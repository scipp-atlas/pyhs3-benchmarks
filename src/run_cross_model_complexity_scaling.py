from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Iterable

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


BENCHMARK_NAME = "cross_model_complexity_scaling"
DEFAULT_OUTPUT = RESULTS_DIR / BENCHMARK_NAME / f"{BENCHMARK_NAME}_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME
DEFAULT_ANALYSES = ["L_ch0", "L_ch1", "L_ch2"]
# Generic RooFit expression workspaces show smooth floating-point-level
# differences of O(1e-8) with respect to PyHS3. Keep the default tolerance
# tight enough to catch meaningful regressions, but loose enough to avoid
# false failures for these known generic-expression effects.
DEFAULT_DELTA_TOLERANCE = 5e-8
DEFAULT_MINIMUM_TOLERANCE = 1e-12
DEFAULT_CASES = [
    "simple_workspace_nonp",
    "simple_workspace_generic_nonp",
    "simple_workspace",
    "simple_workspace_generic",
]

WORKSPACE_PAIRS = {
    "simple_workspace_nonp": {
        "json": "simple_workspace_nonp.json",
        "root": "simple_workspace_nonp.root",
    },
    "simple_workspace_generic_nonp": {
        "json": "simple_workspace_generic_nonp.json",
        "root": "simple_workspace_generic_nonp.root",
    },
    "simple_workspace": {
        "json": "simple_workspace.json",
        "root": "simple_workspace.root",
    },
    "simple_workspace_generic": {
        "json": "simple_workspace_generic.json",
        "root": "simple_workspace_generic.root",
    },
}

FRAMEWORK_STYLE = {
    "pyhs3": {"label": "PyHS3", "color": "#0055A4", "marker": "s", "linestyle": "-"},
    "roofit": {"label": "RooFit", "color": "#009E73", "marker": "D", "linestyle": "--"},
}


@dataclass(frozen=True)
class CaseSpec:
    case_name: str
    analysis_name: str
    json_path: Path
    root_path: Path


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_existing_dir(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{name} is not a directory: {path}")
    return path


def validate_existing_file(path: Path, name: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a file: {path}")
    return path


def validate_positive_int(value: int, name: str, minimum: int = 1) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def validate_finite_float(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def validate_benchmark_config(
    *,
    n_runs: int,
    mu_sig: float,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
    cases: list[str],
    analyses: list[str],
    delta_tolerance: float,
    minimum_tolerance: float,
) -> None:
    validate_positive_int(n_runs, "n_runs", minimum=1)
    validate_positive_int(n_scan_points, "n_scan_points", minimum=2)
    validate_finite_float(mu_sig, "mu_sig")
    validate_finite_float(scan_min, "scan_min")
    validate_finite_float(scan_max, "scan_max")
    validate_finite_float(delta_tolerance, "delta_tolerance")
    validate_finite_float(minimum_tolerance, "minimum_tolerance")

    if scan_min >= scan_max:
        raise ValueError(
            f"scan_min must be smaller than scan_max, got {scan_min} >= {scan_max}"
        )
    if delta_tolerance <= 0.0:
        raise ValueError("delta_tolerance must be positive")
    if minimum_tolerance <= 0.0:
        raise ValueError("minimum_tolerance must be positive")
    if not cases:
        raise ValueError("At least one case must be selected")
    if not analyses:
        raise ValueError("At least one analysis must be selected")

    unknown_cases = sorted(set(cases) - set(WORKSPACE_PAIRS))
    if unknown_cases:
        raise ValueError(
            "Unknown cases: "
            + ", ".join(unknown_cases)
            + f". Available cases: {', '.join(DEFAULT_CASES)}"
        )

    unknown_analyses = sorted(set(analyses) - set(DEFAULT_ANALYSES))
    if unknown_analyses:
        raise ValueError(
            "Unknown analyses: "
            + ", ".join(unknown_analyses)
            + f". Available analyses: {', '.join(DEFAULT_ANALYSES)}"
        )


def validate_scan_values(values: Iterable[float], name: str) -> None:
    values_list = list(values)
    if not values_list:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(float(value)) for value in values_list):
        raise ValueError(f"{name} contains non-finite values")


def validate_framework_result(result: dict[str, Any]) -> None:
    required_finite_fields = [
        "build_time_seconds",
        "cold_first_evaluation_time_seconds",
        "warm_evaluation_time_seconds_mean",
        "scan_time_seconds",
        "time_per_scan_point_seconds",
        "current_rss_delta_mb",
        "peak_rss_delta_mb",
        "first_nll",
        "warm_nll",
        "minimum_mu_sig",
    ]
    for field in required_finite_fields:
        value = float(result[field])
        if not math.isfinite(value):
            raise ValueError(
                f"{result['framework']} result field {field} is not finite"
            )
    if result["build_time_seconds"] < 0.0:
        raise ValueError("build_time_seconds must be non-negative")
    if result["cold_first_evaluation_time_seconds"] <= 0.0:
        raise ValueError("cold_first_evaluation_time_seconds must be positive")
    if result["warm_evaluation_time_seconds_mean"] <= 0.0:
        raise ValueError("warm_evaluation_time_seconds_mean must be positive")
    if result["scan_time_seconds"] <= 0.0:
        raise ValueError("scan_time_seconds must be positive")
    validate_scan_values(result["scan_nll_values"], "scan_nll_values")
    validate_scan_values(result["delta_nll_shape"], "delta_nll_shape")


# ---------------------------------------------------------------------------
# Case discovery and model helpers
# ---------------------------------------------------------------------------


def channel_from_analysis(analysis_name: str) -> str:
    if not analysis_name.startswith("L_"):
        raise ValueError(f"Analysis name must start with 'L_': {analysis_name}")
    return analysis_name.replace("L_", "", 1)


def target_from_analysis(analysis_name: str) -> str:
    return f"model_{channel_from_analysis(analysis_name)}"


def build_case_specs(
    *,
    json_input_dir: Path,
    root_input_dir: Path,
    cases: list[str],
    analyses: list[str],
) -> list[CaseSpec]:
    specs: list[CaseSpec] = []
    for case_name in cases:
        pair = WORKSPACE_PAIRS[case_name]
        json_path = json_input_dir / pair["json"]
        root_path = root_input_dir / pair["root"]
        for analysis_name in analyses:
            specs.append(
                CaseSpec(
                    case_name=case_name,
                    analysis_name=analysis_name,
                    json_path=json_path,
                    root_path=root_path,
                )
            )
    return specs


def get_pyhs3_x_data(workspace: Workspace, analysis_name: str) -> np.ndarray:
    data_name = f"combData_{channel_from_analysis(analysis_name)}"
    try:
        data_entries = workspace.data.root
    except AttributeError as exc:
        raise ValueError("Workspace does not contain a valid data section") from exc

    for data in data_entries:
        if data.name == data_name:
            values = np.asarray([entry[0] for entry in data.entries], dtype=np.float64)
            if values.size == 0:
                raise ValueError(f"PyHS3 data {data_name} is empty")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"PyHS3 data {data_name} contains non-finite values")
            return values

    raise KeyError(f"Could not find PyHS3 data {data_name}")


def get_pyhs3_params(model: Any, x: np.ndarray) -> dict[str, Any]:
    try:
        free_params = model.free_params
    except AttributeError as exc:
        raise ValueError("PyHS3 model does not expose free_params") from exc

    params = {
        name: np.asarray(value, dtype=np.float64) for name, value in free_params.items()
    }
    for name, value in params.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"PyHS3 free parameter {name} contains non-finite values")
    params["x"] = x
    return params


def pyhs3_nll(model: Any, target: str, params: dict[str, Any], mu_sig: float) -> float:
    validate_finite_float(mu_sig, "mu_sig")
    eval_params = dict(params)
    eval_params["mu_sig"] = np.asarray(mu_sig, dtype=np.float64)
    values = np.asarray(model.logpdf(target, **eval_params), dtype=np.float64)
    if values.size == 0:
        raise ValueError(f"PyHS3 returned an empty logpdf array for {target}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"PyHS3 returned non-finite logpdf values for {target}")
    return -float(np.sum(values))


def build_pyhs3_case(json_path: Path, analysis_name: str) -> dict[str, Any]:
    validate_existing_file(json_path, "PyHS3 JSON workspace")
    workspace = Workspace.load(json_path)
    model = workspace.model(analysis_name, progress=False, mode="FAST_RUN")
    x = get_pyhs3_x_data(workspace, analysis_name)
    target = target_from_analysis(analysis_name)
    params = get_pyhs3_params(model, x)
    return {
        "workspace": workspace,
        "model": model,
        "x": x,
        "target": target,
        "params": params,
    }


def require_root() -> Any:
    if ROOT is None:
        raise RuntimeError("ROOT is not available in this environment")
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    return ROOT


def get_roofit_workspace(root_path: Path) -> tuple[Any, Any]:
    root = require_root()
    validate_existing_file(root_path, "RooFit ROOT workspace")
    root_file = root.TFile.Open(str(root_path))
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file {root_path}")
    workspace = root_file.Get("combWS")
    if workspace is None:
        root_file.Close()
        raise RuntimeError(f"Could not find RooWorkspace combWS in {root_path}")
    return root_file, workspace


def get_roofit_channel_dataset(workspace: Any, analysis_name: str) -> Any:
    channel = channel_from_analysis(analysis_name)
    data = workspace.data("combData")
    if data is None:
        raise RuntimeError("Could not find RooFit dataset combData")
    index = workspace.cat("index")
    if index is None:
        raise RuntimeError("Could not find RooFit category index")
    reduced = data.reduce(f"index==index::{channel}")
    if reduced is None:
        raise RuntimeError(f"Could not reduce RooFit dataset for channel {channel}")
    if reduced.numEntries() <= 0:
        raise RuntimeError(f"RooFit dataset for channel {channel} is empty")
    return reduced


def build_roofit_case(root_path: Path, analysis_name: str) -> dict[str, Any]:
    root = require_root()
    root_file, workspace = get_roofit_workspace(root_path)
    workspace.loadSnapshot("nominal")
    target = target_from_analysis(analysis_name)
    pdf = workspace.pdf(target)
    if pdf is None:
        root_file.Close()
        raise RuntimeError(f"Could not find RooFit pdf {target}")
    data = get_roofit_channel_dataset(workspace, analysis_name)
    x = workspace.var("x")
    mu_sig = workspace.var("mu_sig")
    if x is None:
        root_file.Close()
        raise RuntimeError("Could not find RooFit variable x")
    if mu_sig is None:
        root_file.Close()
        raise RuntimeError("Could not find RooFit variable mu_sig")
    return {
        "root_file": root_file,
        "workspace": workspace,
        "pdf": pdf,
        "data": data,
        "x": x,
        "mu_sig": mu_sig,
        "target": target,
        "root": root,
    }


def roofit_nll(case: dict[str, Any], mu_sig_value: float) -> float:
    validate_finite_float(mu_sig_value, "mu_sig")
    pdf = case["pdf"]
    data = case["data"]
    x = case["x"]
    mu_sig = case["mu_sig"]
    root = case.get("root") or require_root()
    mu_sig.setVal(mu_sig_value)
    total = 0.0
    n_entries = data.numEntries()
    if n_entries <= 0:
        raise ValueError("RooFit dataset is empty")
    norm_set = root.RooArgSet(x)
    for index in range(n_entries):
        row = data.get(index)
        x_value = float(row.getRealValue("x"))
        if not math.isfinite(x_value):
            raise ValueError(
                f"RooFit dataset contains non-finite x value at entry {index}: {x_value}"
            )
        x.setVal(x_value)
        value = float(pdf.getVal(norm_set))
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(
                f"RooFit returned invalid PDF value at entry {index}: {value}"
            )
        total -= math.log(value)
    return total


# ---------------------------------------------------------------------------
# Timing, scans, agreement
# ---------------------------------------------------------------------------


def summarize_timings(values: list[float]) -> dict[str, float]:
    validate_scan_values(values, "timings")
    return {
        "mean_seconds": mean(values),
        "std_seconds": stdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
    }


def time_repeated(func: Callable[[], float], n_runs: int) -> tuple[float, list[float]]:
    validate_positive_int(n_runs, "n_runs", minimum=1)
    timings: list[float] = []
    value = float("nan")
    for _ in range(n_runs):
        start = time.perf_counter()
        value = float(func())
        end = time.perf_counter()
        if not math.isfinite(value):
            raise ValueError(
                f"Repeated evaluation returned non-finite NLL value: {value}"
            )
        timings.append(end - start)
    return value, timings


def scan_nll(
    func: Callable[[float], float], scan_values: list[float]
) -> tuple[list[float], float]:
    validate_scan_values(scan_values, "scan_values")
    values: list[float] = []
    start = time.perf_counter()
    for mu_value in scan_values:
        values.append(float(func(mu_value)))
    end = time.perf_counter()
    validate_scan_values(values, "scan_nll_values")
    return values, end - start


def delta_nll(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute delta NLL for an empty array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Cannot compute delta NLL for non-finite values")
    return arr - np.min(arr)


def minimum_position(scan_values: list[float], nll_values: list[float]) -> float:
    if len(scan_values) != len(nll_values):
        raise ValueError("scan_values and nll_values must have the same length")
    validate_scan_values(nll_values, "nll_values")
    return float(scan_values[int(np.argmin(np.asarray(nll_values, dtype=float)))])


def close_case(case: Any) -> None:
    if isinstance(case, dict) and case.get("root_file") is not None:
        try:
            case["root_file"].Close()
        except Exception:
            pass


def measure_framework(
    *,
    framework: str,
    build_func: Callable[[], Any],
    eval_func: Callable[[Any, float], float],
    scan_values: list[float],
    n_runs: int,
    mu_sig: float,
) -> dict[str, Any]:
    validate_scan_values(scan_values, "scan_values")
    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()
    case = None
    try:
        build_start = time.perf_counter()
        case = build_func()
        build_end = time.perf_counter()

        first_start = time.perf_counter()
        first_nll = float(eval_func(case, mu_sig))
        first_end = time.perf_counter()
        if not math.isfinite(first_nll):
            raise ValueError(
                f"{framework} returned non-finite cold first NLL: {first_nll}"
            )

        warm_nll, warm_timings = time_repeated(
            lambda: eval_func(case, mu_sig), n_runs=n_runs
        )
        scan_values_nll, scan_time = scan_nll(
            lambda scan_mu: eval_func(case, scan_mu), scan_values=scan_values
        )
        current_rss_after_mb = get_current_rss_mb()
        peak_rss_after_mb = get_peak_rss_mb()
        warm_summary = summarize_timings(warm_timings)
        result = {
            "framework": framework,
            "status": "success",
            "n_runs": n_runs,
            "n_scan_points": len(scan_values),
            "build_time_seconds": build_end - build_start,
            "cold_first_evaluation_time_seconds": first_end - first_start,
            "warm_evaluation": warm_summary,
            "warm_evaluation_time_seconds_mean": warm_summary["mean_seconds"],
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
            "warm_nll": warm_nll,
            "scan_nll_values": scan_values_nll,
            "delta_nll_shape": delta_nll(scan_values_nll).tolist(),
            "minimum_mu_sig": minimum_position(scan_values, scan_values_nll),
            "finite_values": bool(
                math.isfinite(first_nll)
                and math.isfinite(warm_nll)
                and all(math.isfinite(value) for value in scan_values_nll)
            ),
        }
        result["minimum_index"] = int(
            np.argmin(np.asarray(scan_values_nll, dtype=float))
        )
        validate_framework_result(result)
        return result
    finally:
        close_case(case)


def add_agreement_metrics(
    *,
    pyhs3_result: dict[str, Any],
    roofit_result: dict[str, Any],
    delta_tolerance: float,
    minimum_tolerance: float,
) -> dict[str, Any]:
    pyhs3_scan = np.asarray(pyhs3_result["scan_nll_values"], dtype=np.float64)
    roofit_scan = np.asarray(roofit_result["scan_nll_values"], dtype=np.float64)
    if pyhs3_scan.shape != roofit_scan.shape:
        raise ValueError("Cannot compare PyHS3 and RooFit scans with different shapes")
    pyhs3_delta = delta_nll(pyhs3_result["scan_nll_values"])
    roofit_delta = delta_nll(roofit_result["scan_nll_values"])
    raw_diff = roofit_scan - pyhs3_scan
    delta_diff = roofit_delta - pyhs3_delta
    minimum_mu_sig_abs_diff = abs(
        roofit_result["minimum_mu_sig"] - pyhs3_result["minimum_mu_sig"]
    )
    delta_nll_max_abs_diff = float(np.max(np.abs(delta_diff)))
    max_delta_index = int(np.argmax(np.abs(delta_diff)))
    agreement = {
        "raw_nll_abs_diff": abs(roofit_result["first_nll"] - pyhs3_result["first_nll"]),
        "constant_offset_estimate": float(np.mean(raw_diff)),
        "raw_scan_max_abs_diff": float(np.max(np.abs(raw_diff))),
        "raw_scan_mean_abs_diff": float(np.mean(np.abs(raw_diff))),
        "delta_nll_max_abs_diff": delta_nll_max_abs_diff,
        "centered_residual_max_abs_diff": float(
            np.max(np.abs(delta_diff - np.mean(delta_diff)))
        ),
        "minimum_index_match": bool(
            pyhs3_result["minimum_index"] == roofit_result["minimum_index"]
        ),
        "minimum_mu_sig_abs_diff": minimum_mu_sig_abs_diff,
        "delta_shape_success": delta_nll_max_abs_diff <= delta_tolerance,
        "minimum_mu_sig_success": minimum_mu_sig_abs_diff <= minimum_tolerance,
        "max_delta_nll_diff_index": max_delta_index,
        "max_delta_nll_diff_value": float(delta_diff[max_delta_index]),
        "pyhs3_delta_nll_shape": pyhs3_delta.tolist(),
        "roofit_delta_nll_shape": roofit_delta.tolist(),
        "delta_nll_difference": delta_diff.tolist(),
        "raw_nll_difference": raw_diff.tolist(),
    }
    agreement["validation_status"] = (
        "success"
        if agreement["delta_shape_success"] and agreement["minimum_mu_sig_success"]
        else "failed"
    )
    return agreement


def failed_case_result(spec: CaseSpec, exc: BaseException) -> dict[str, Any]:
    return {
        "case": spec.case_name,
        "analysis": spec.analysis_name,
        "target": target_from_analysis(spec.analysis_name),
        "json_path": str(spec.json_path),
        "root_path": str(spec.root_path),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def measure_case(
    *,
    spec: CaseSpec,
    scan_values: list[float],
    n_runs: int,
    mu_sig: float,
    delta_tolerance: float,
    minimum_tolerance: float,
) -> dict[str, Any]:
    pyhs3_result = measure_framework(
        framework="pyhs3",
        build_func=lambda: build_pyhs3_case(spec.json_path, spec.analysis_name),
        eval_func=lambda case, mu: pyhs3_nll(
            case["model"], case["target"], case["params"], mu
        ),
        scan_values=scan_values,
        n_runs=n_runs,
        mu_sig=mu_sig,
    )
    roofit_result = measure_framework(
        framework="roofit",
        build_func=lambda: build_roofit_case(spec.root_path, spec.analysis_name),
        eval_func=lambda case, mu: roofit_nll(case, mu),
        scan_values=scan_values,
        n_runs=n_runs,
        mu_sig=mu_sig,
    )
    agreement = add_agreement_metrics(
        pyhs3_result=pyhs3_result,
        roofit_result=roofit_result,
        delta_tolerance=delta_tolerance,
        minimum_tolerance=minimum_tolerance,
    )

    if agreement["validation_status"] != "success":
        return {
            "case": spec.case_name,
            "analysis": spec.analysis_name,
            "target": target_from_analysis(spec.analysis_name),
            "json_path": str(spec.json_path),
            "root_path": str(spec.root_path),
            "status": "failed",
            "error_type": "ValidationFailure",
            "error_message": (
                "Numerical agreement check failed "
                f"(delta-NLL={agreement['delta_nll_max_abs_diff']:.3e}, "
                f"minimum Δμ={agreement['minimum_mu_sig_abs_diff']:.3e}, "
                f"constant offset={agreement['constant_offset_estimate']:.3e}, "
                f"raw max diff={agreement['raw_scan_max_abs_diff']:.3e})"
            ),
            "pyhs3": pyhs3_result,
            "roofit": roofit_result,
            "agreement": agreement,
        }

    status = (
        "success"
        if pyhs3_result["finite_values"]
        and roofit_result["finite_values"]
        and agreement["validation_status"] == "success"
        else "failed"
    )
    return {
        "case": spec.case_name,
        "analysis": spec.analysis_name,
        "target": target_from_analysis(spec.analysis_name),
        "json_path": str(spec.json_path),
        "root_path": str(spec.root_path),
        "status": status,
        "pyhs3": pyhs3_result,
        "roofit": roofit_result,
        "agreement": agreement,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _apply_cern_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.4,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "font.size": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.width": 1.0,
            "ytick.minor.width": 1.0,
            "axes.grid": True,
            "grid.color": "0.82",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.65,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _style_for(framework: str) -> dict[str, str]:
    return FRAMEWORK_STYLE.get(
        framework,
        {"label": framework, "color": "#4D4D4D", "marker": "o", "linestyle": "-"},
    )


def _framework_label(framework: str) -> str:
    return _style_for(framework)["label"]


def _case_label(result: dict[str, Any]) -> str:
    return f"{result['case'].replace('simple_workspace_', '').replace('simple_workspace', 'base')}\n{result['analysis'].replace('L_', '')}"


def _successful_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if result.get("status") == "success"]


def _diagnostic_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        result
        for result in results
        if all(key in result for key in ("pyhs3", "roofit", "agreement"))
    ]


def _validation_failed_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        result
        for result in _diagnostic_results(results)
        if result.get("status") != "success"
        and result.get("error_type") == "ValidationFailure"
    ]


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path.with_suffix(".png"), dpi=300)
    except OSError as exc:
        raise OSError(f"Failed to save plot to {output_path}") from exc
    finally:
        plt.close(fig)


def _plot_floor(values: list[float], floor: float = 1e-12) -> list[float]:
    return [max(float(value), floor) for value in values]


def _collect_scaling_records(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in _successful_results(results):
        for framework in ("pyhs3", "roofit"):
            item = result[framework]
            records.append(
                {
                    "case": result["case"],
                    "analysis": result["analysis"],
                    "target": result["target"],
                    "plot_label": _case_label(result),
                    "framework": framework,
                    "build_ms": item["build_time_seconds"] * 1000.0,
                    "cold_eval_us": item["cold_first_evaluation_time_seconds"] * 1e6,
                    "warm_eval_us": item["warm_evaluation_time_seconds_mean"] * 1e6,
                    "scan_ms": item["scan_time_seconds"] * 1000.0,
                    "time_per_point_us": item["time_per_scan_point_seconds"] * 1e6,
                    "rss_mb": item["current_rss_delta_mb"],
                    "minimum_mu_sig": item["minimum_mu_sig"],
                }
            )
    return records


def make_runtime_scaling_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    labels = [_case_label(result) for result in successful]
    x = np.arange(len(successful))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    for offset, framework in [(-width / 2, "pyhs3"), (width / 2, "roofit")]:
        style = _style_for(framework)
        values = _plot_floor(
            [
                result[framework]["time_per_scan_point_seconds"] * 1e6
                for result in successful
            ],
            floor=1e-6,
        )
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=style["color"],
            label=style["label"],
            alpha=0.9,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.12,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Time per scan point [µs] (log scale)")
    ax.set_title(
        "Model-complexity scaling: steady-state NLL evaluation",
        loc="left",
        weight="bold",
    )
    ax.legend(
        frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Framework"
    )
    ax.grid(True, which="both", axis="y", alpha=0.45)
    ax.grid(False, axis="x")
    fig.subplots_adjust(right=0.84, bottom=0.27)
    _save_figure(fig, output_path)


def make_timing_breakdown_plot(
    results: list[dict[str, Any]], output_path: Path
) -> None:
    _apply_cern_style()
    records = _collect_scaling_records(results)
    labels = [
        f"{record['plot_label']}\n{_framework_label(record['framework'])}"
        for record in records
    ]
    x = np.arange(len(records))
    width = 0.22
    metrics = [
        ("build_ms", "Build [ms]", -width, ""),
        ("cold_eval_us", "Cold eval [µs]", 0.0, "//"),
        ("warm_eval_us", "Warm eval [µs]", width, "xx"),
    ]
    fig, ax = plt.subplots(figsize=(16.0, 7.5))
    for metric, label, offset, hatch in metrics:
        values = _plot_floor([record[metric] for record in records], floor=1e-6)
        colors = [_style_for(record["framework"])["color"] for record in records]
        ax.bar(
            x + offset,
            values,
            width,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            hatch=hatch,
            label=label,
            alpha=0.86,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("Timing (mixed units, log scale)")
    ax.set_title(
        "Timing profile across model complexity cases", loc="left", weight="bold"
    )
    metric_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="lightgray",
            edgecolor="black",
            hatch=hatch,
            label=label,
        )
        for _, label, _, hatch in metrics
    ]
    framework_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=_style_for(framework)["color"],
            edgecolor="black",
            label=_framework_label(framework),
        )
        for framework in ("pyhs3", "roofit")
    ]
    legend1 = ax.legend(
        handles=framework_handles,
        title="Framework",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.legend(
        handles=metric_handles,
        title="Metric",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.72),
    )
    ax.add_artist(legend1)
    ax.grid(True, which="both", axis="y", alpha=0.45)
    ax.grid(False, axis="x")
    fig.subplots_adjust(right=0.81, bottom=0.34)
    _save_figure(fig, output_path)


def make_memory_scaling_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    labels = [_case_label(result) for result in successful]
    x = np.arange(len(successful))
    width = 0.36
    fig, ax = plt.subplots(figsize=(14.5, 7.0))
    for offset, framework in [(-width / 2, "pyhs3"), (width / 2, "roofit")]:
        style = _style_for(framework)
        values = _plot_floor(
            [result[framework]["current_rss_delta_mb"] for result in successful],
            floor=1e-3,
        )
        ax.bar(
            x + offset,
            values,
            width,
            color=style["color"],
            label=style["label"],
            alpha=0.9,
        )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Current RSS delta [MB] (log scale)")
    ax.set_title(
        "Memory footprint across model complexity cases", loc="left", weight="bold"
    )
    ax.legend(
        frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="Framework"
    )
    ax.grid(True, which="both", axis="y", alpha=0.45)
    ax.grid(False, axis="x")
    fig.subplots_adjust(right=0.84, bottom=0.27)
    _save_figure(fig, output_path)


def make_agreement_plot(
    results: list[dict[str, Any]], delta_tolerance: float, output_path: Path
) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    labels = [_case_label(result) for result in successful]
    values = [result["agreement"]["delta_nll_max_abs_diff"] for result in successful]
    plot_values = _plot_floor(values, floor=max(1e-16, delta_tolerance * 1e-8))
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    x = np.arange(len(successful))
    bars = ax.bar(x, plot_values, color="#0055A4", alpha=0.88)
    ax.axhline(
        delta_tolerance,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"tolerance = {delta_tolerance:.0e}",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"max $|\Delta$NLL$_{RooFit}$ - $\Delta$NLL$_{PyHS3}|$")
    ax.set_title("PyHS3 vs RooFit numerical agreement", loc="left", weight="bold")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(True, which="both", axis="y", alpha=0.45)
    ax.grid(False, axis="x")
    ax.set_ylim(top=max(max(plot_values) * 12.0, delta_tolerance * 2.0))
    for bar, raw_value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.18,
            f"{raw_value:.1e}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    fig.subplots_adjust(right=0.82, bottom=0.27)
    _save_figure(fig, output_path)


def make_profile_examples_plot(
    results: list[dict[str, Any]], scan_values: list[float], output_path: Path
) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    if not successful:
        raise ValueError("No successful benchmark results available for profile plot")
    selected = successful[: min(4, len(successful))]
    x = np.asarray(scan_values, dtype=float)
    fig, axes = plt.subplots(
        len(selected), 1, figsize=(10.5, 3.1 * len(selected)), sharex=True
    )
    if len(selected) == 1:
        axes = [axes]
    for ax, result in zip(axes, selected, strict=True):
        for framework in ("pyhs3", "roofit"):
            style = _style_for(framework)
            values = np.asarray(result[framework]["delta_nll_shape"], dtype=float)
            ax.plot(
                x,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                label=style["label"],
            )
        ax.set_ylabel(r"$\Delta$NLL")
        ax.set_title(
            f"{result['case']} / {result['analysis']}",
            loc="left",
            fontsize=12,
            weight="bold",
        )
        ax.legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel(r"Signal strength $\mu_{sig}$")
    fig.suptitle(
        "Representative ΔNLL profile overlays",
        x=0.01,
        ha="left",
        weight="bold",
        fontsize=17,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, output_path)


def make_validation_failure_diagnostics_plot(
    results: list[dict[str, Any]],
    scan_values: list[float],
    output_path: Path,
) -> None:
    _apply_cern_style()
    failed = _validation_failed_results(results)
    if not failed:
        return

    x = np.asarray(scan_values, dtype=float)
    fig, axes = plt.subplots(
        len(failed), 1, figsize=(11.0, 3.4 * len(failed)), sharex=True
    )
    if len(failed) == 1:
        axes = [axes]

    for ax, result in zip(axes, failed, strict=True):
        agreement = result["agreement"]
        diff = np.asarray(agreement["delta_nll_difference"], dtype=float)
        pyhs3_delta = np.asarray(agreement["pyhs3_delta_nll_shape"], dtype=float)
        roofit_delta = np.asarray(agreement["roofit_delta_nll_shape"], dtype=float)
        max_index = int(agreement["max_delta_nll_diff_index"])

        ax.plot(
            x,
            diff,
            color="#D55E00",
            linewidth=2.2,
            label=r"RooFit $\Delta$NLL - PyHS3 $\Delta$NLL",
        )
        ax.axhline(0.0, color="black", linewidth=1.1, alpha=0.75)
        ax.scatter(
            [x[max_index]],
            [diff[max_index]],
            color="#D55E00",
            edgecolor="black",
            zorder=5,
            label=f"max |diff| = {agreement['delta_nll_max_abs_diff']:.2e}",
        )
        ax2 = ax.twinx()
        ax2.plot(
            x,
            pyhs3_delta,
            color=_style_for("pyhs3")["color"],
            alpha=0.25,
            linewidth=1.4,
            label="PyHS3 ΔNLL",
        )
        ax2.plot(
            x,
            roofit_delta,
            color=_style_for("roofit")["color"],
            alpha=0.25,
            linewidth=1.4,
            linestyle="--",
            label="RooFit ΔNLL",
        )
        ax2.set_ylabel(r"$\Delta$NLL", alpha=0.7)
        ax2.tick_params(axis="y", labelsize=9)

        ax.set_ylabel("Residual")
        ax.set_title(
            f"{result['case']} / {result['analysis']} validation failure",
            loc="left",
            fontsize=12,
            weight="bold",
        )
        ax.text(
            0.02,
            0.92,
            "constant offset = "
            f"{agreement['constant_offset_estimate']:.2e}; raw max diff = "
            f"{agreement['raw_scan_max_abs_diff']:.2e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(
            handles1 + handles2,
            labels1 + labels2,
            frameon=False,
            loc="lower right",
            fontsize=9,
        )

    axes[-1].set_xlabel(r"Signal strength $\mu_{sig}$")
    fig.suptitle(
        "Validation-failure diagnostics", x=0.01, ha="left", weight="bold", fontsize=17
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_figure(fig, output_path)


def make_summary_table_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    columns = [
        "Case",
        "Analysis",
        "Validation",
        "PyHS3 µs/pt",
        "RooFit µs/pt",
        "speed ratio",
        "ΔNLL diff",
        "min Δ",
    ]
    rows: list[list[str]] = []
    for result in successful:
        py_time = result["pyhs3"]["time_per_scan_point_seconds"] * 1e6
        roofit_time = result["roofit"]["time_per_scan_point_seconds"] * 1e6
        ratio = py_time / roofit_time if roofit_time > 0 else float("nan")
        rows.append(
            [
                result["case"],
                result["analysis"],
                result["agreement"]["validation_status"],
                f"{py_time:.2f}",
                f"{roofit_time:.2f}",
                f"{ratio:.2f}×",
                f"{result['agreement']['delta_nll_max_abs_diff']:.2e}",
                f"{result['agreement']['minimum_mu_sig_abs_diff']:.2e}",
            ]
        )
    fig, ax = plt.subplots(figsize=(14.5, 3.7 + 0.35 * len(rows)))
    ax.axis("off")
    ax.set_title(
        "Cross-framework model-complexity scaling summary",
        loc="left",
        weight="bold",
        pad=16,
    )
    ax.text(
        0.0,
        0.94,
        "Compares PyHS3 and RooFit on matched unbinned workspace pairs and validates ΔNLL shape agreement.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.65)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("0.15")
        elif col == 2 and cell.get_text().get_text() == "success":
            cell.set_text_props(weight="bold", color="#00843D")
    _save_figure(fig, output_path)


def make_plots(
    results: list[dict[str, Any]],
    scan_values: list[float],
    plot_dir: Path,
    delta_tolerance: float,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not _successful_results(results):
        raise ValueError("No successful benchmark results available for plotting")
    make_runtime_scaling_plot(
        results, plot_dir / "cross_model_complexity_runtime_scaling.png"
    )
    make_timing_breakdown_plot(
        results, plot_dir / "cross_model_complexity_timing_breakdown.png"
    )
    make_memory_scaling_plot(
        results, plot_dir / "cross_model_complexity_memory_scaling.png"
    )
    make_agreement_plot(
        results, delta_tolerance, plot_dir / "cross_model_complexity_agreement.png"
    )
    make_profile_examples_plot(
        results, scan_values, plot_dir / "cross_model_complexity_profile_examples.png"
    )
    make_validation_failure_diagnostics_plot(
        results,
        scan_values,
        plot_dir / "cross_model_complexity_validation_failure_diagnostics.png",
    )
    make_summary_table_plot(
        results, plot_dir / "cross_model_complexity_summary_table.png"
    )


# ---------------------------------------------------------------------------
# Output and CLI
# ---------------------------------------------------------------------------


def print_case(result: dict[str, Any]) -> None:
    print()
    print("-" * 80)
    print(f"{result['case']} / {result['analysis']} / {result['target']}")
    print("-" * 80)
    print(f"status: {result['status']}")
    if result["status"] != "success":
        print(f"error:  {result.get('error_type')}: {result.get('error_message')}")
        if not all(key in result for key in ("pyhs3", "roofit", "agreement")):
            return
        print(
            "diagnostics: available because both frameworks completed before validation failed"
        )
    for framework in ("pyhs3", "roofit"):
        item = result[framework]
        print()
        print(framework)
        print(f"  build time:           {item['build_time_seconds'] * 1000.0:.3f} ms")
        print(
            f"  cold first eval:      {item['cold_first_evaluation_time_seconds'] * 1e6:.3f} us"
        )
        print(
            f"  warm eval:            {item['warm_evaluation']['mean_seconds'] * 1e6:.3f} us"
        )
        print(f"  scan time:            {item['scan_time_seconds'] * 1000.0:.3f} ms")
        print(
            f"  time per scan point:  {item['time_per_scan_point_seconds'] * 1e6:.3f} us"
        )
        print(f"  current RSS delta:    {item['current_rss_delta_mb']:.3f} MB")
        print(f"  peak RSS delta:       {item['peak_rss_delta_mb']:.3f} MB")
        print(f"  minimum mu_sig:       {item['minimum_mu_sig']:.15f}")
        print(f"  finite values:        {item['finite_values']}")
    agreement = result["agreement"]
    print()
    print("agreement")
    print(f"  validation:             {agreement['validation_status']}")
    print(f"  raw NLL abs diff:       {agreement['raw_nll_abs_diff']:.15e}")
    print(f"  raw scan max abs diff:  {agreement['raw_scan_max_abs_diff']:.15e}")
    print(f"  delta-NLL max diff:     {agreement['delta_nll_max_abs_diff']:.15e}")
    print(f"  constant offset:        {agreement['constant_offset_estimate']:.15e}")
    print(f"  minimum mu_sig abs diff:{agreement['minimum_mu_sig_abs_diff']:.15e}")


def build_failed_output(
    *,
    json_input_dir: Path,
    root_input_dir: Path,
    n_runs: int,
    mu_sig: float,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
    cases: list[str],
    analyses: list[str],
    delta_tolerance: float,
    minimum_tolerance: float,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "frameworks": ["pyhs3", "roofit"],
        "json_input_dir": str(json_input_dir),
        "root_input_dir": str(root_input_dir),
        "n_runs": n_runs,
        "mu_sig": mu_sig,
        "scan_min": scan_min,
        "scan_max": scan_max,
        "n_scan_points": n_scan_points,
        "cases": cases,
        "analyses": analyses,
        "delta_tolerance": delta_tolerance,
        "minimum_tolerance": minimum_tolerance,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "results": [],
    }


def run(
    *,
    json_input_dir: Path,
    root_input_dir: Path,
    n_runs: int,
    mu_sig: float,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
    output: Path,
    plot: bool,
    plot_dir: Path,
    cases: list[str] | None = None,
    analyses: list[str] | None = None,
    delta_tolerance: float = DEFAULT_DELTA_TOLERANCE,
    minimum_tolerance: float = DEFAULT_MINIMUM_TOLERANCE,
    continue_on_case_error: bool = True,
) -> dict[str, Any]:
    selected_cases = cases or list(DEFAULT_CASES)
    selected_analyses = analyses or list(DEFAULT_ANALYSES)
    try:
        validate_benchmark_config(
            n_runs=n_runs,
            mu_sig=mu_sig,
            scan_min=scan_min,
            scan_max=scan_max,
            n_scan_points=n_scan_points,
            cases=selected_cases,
            analyses=selected_analyses,
            delta_tolerance=delta_tolerance,
            minimum_tolerance=minimum_tolerance,
        )
        validate_existing_dir(json_input_dir, "JSON input directory")
        validate_existing_dir(root_input_dir, "ROOT input directory")
        require_root()

        scan_values = [
            float(value) for value in np.linspace(scan_min, scan_max, n_scan_points)
        ]
        specs = build_case_specs(
            json_input_dir=json_input_dir,
            root_input_dir=root_input_dir,
            cases=selected_cases,
            analyses=selected_analyses,
        )
        results: list[dict[str, Any]] = []
        for spec in specs:
            try:
                validate_existing_file(spec.json_path, "PyHS3 JSON workspace")
                validate_existing_file(spec.root_path, "RooFit ROOT workspace")
                results.append(
                    measure_case(
                        spec=spec,
                        scan_values=scan_values,
                        n_runs=n_runs,
                        mu_sig=mu_sig,
                        delta_tolerance=delta_tolerance,
                        minimum_tolerance=minimum_tolerance,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - intentional case-level isolation
                if not continue_on_case_error:
                    raise RuntimeError(
                        f"Model-complexity case failed for {spec.case_name}/{spec.analysis_name}"
                    ) from exc
                results.append(failed_case_result(spec, exc))

        successful = _successful_results(results)
        failed = [result for result in results if result.get("status") != "success"]
        status = "success" if successful and not failed else "failed"
        output_data = {
            "benchmark": BENCHMARK_NAME,
            "frameworks": ["pyhs3", "roofit"],
            "json_input_dir": str(json_input_dir),
            "root_input_dir": str(root_input_dir),
            "n_runs": n_runs,
            "mu_sig": mu_sig,
            "scan_min": scan_min,
            "scan_max": scan_max,
            "n_scan_points": n_scan_points,
            "cases": selected_cases,
            "analyses": selected_analyses,
            "delta_tolerance": delta_tolerance,
            "minimum_tolerance": minimum_tolerance,
            "tolerance_rationale": (
                "The default delta-NLL tolerance is 5e-8 because generic RooFit "
                "expression workspaces can differ from PyHS3 at O(1e-8) while "
                "non-generic workspaces agree at machine precision and all fitted "
                "minimum positions remain identical."
            ),
            "status": status,
            "successful_cases": [
                f"{result['case']}/{result['analysis']}" for result in successful
            ],
            "failed_cases": [
                f"{result['case']}/{result['analysis']}" for result in failed
            ],
            "notes": (
                "Cross-framework model-complexity benchmark for matched unbinned HS3 JSON "
                "and RooFit ROOT workspace pairs. pyhf is not included because these models "
                "are not HistFactory-style binned likelihoods. The default delta-NLL "
                "tolerance is 5e-8 to account for smooth O(1e-8) floating-point "
                "differences observed in generic expression workspaces."
            ),
            "scan_values": scan_values,
            "results": results,
        }

        print("=" * 80)
        print("Cross-framework model-complexity scaling benchmark")
        print("=" * 80)
        print("Frameworks: PyHS3, RooFit")
        print(f"Cases:      {', '.join(selected_cases)}")
        print(f"Analyses:   {', '.join(selected_analyses)}")
        print(f"Grid:       [{scan_min}, {scan_max}] with {n_scan_points} points")
        print(f"Warm runs:  {n_runs}")
        print(
            f"Tolerance:  delta-NLL <= {delta_tolerance:.1e}, minimum Δμ <= {minimum_tolerance:.1e}"
        )
        print(f"Status:     {status}")
        print(f"Successful: {len(successful)} / {len(results)}")
        for result in results:
            print_case(result)

        save_json(output_data, output)
        print()
        print(f"Saved result to {output}")
        if plot and successful:
            make_plots(
                successful, scan_values, plot_dir, delta_tolerance=delta_tolerance
            )
            print(f"Saved plots to {plot_dir}")
        return output_data

    except Exception as exc:  # noqa: BLE001 - write structured failure report
        output_data = build_failed_output(
            json_input_dir=json_input_dir,
            root_input_dir=root_input_dir,
            n_runs=n_runs,
            mu_sig=mu_sig,
            scan_min=scan_min,
            scan_max=scan_max,
            n_scan_points=n_scan_points,
            cases=selected_cases,
            analyses=selected_analyses,
            delta_tolerance=delta_tolerance,
            minimum_tolerance=minimum_tolerance,
            exc=exc,
        )
        try:
            save_json(output_data, output)
        except Exception:  # noqa: BLE001
            print(
                "Failed to save benchmark failure report:\n" + traceback.format_exc(),
                file=sys.stderr,
            )
        raise RuntimeError(
            "Cross-framework model-complexity scaling benchmark failed"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a cross-framework model-complexity scaling benchmark for matched "
            "PyHS3 JSON and RooFit ROOT unbinned workspaces."
        )
    )
    parser.add_argument(
        "--json-input-dir", type=Path, default=Path("inputs/model_complexity")
    )
    parser.add_argument(
        "--root-input-dir", type=Path, default=Path("inputs/model_complexity_root")
    )
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--mu-sig", type=float, default=1.0)
    parser.add_argument("--scan-min", type=float, default=0.0)
    parser.add_argument("--scan-max", type=float, default=2.0)
    parser.add_argument("--n-scan-points", type=int, default=51)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--cases", nargs="+", choices=DEFAULT_CASES, default=DEFAULT_CASES
    )
    parser.add_argument(
        "--analyses", nargs="+", choices=DEFAULT_ANALYSES, default=DEFAULT_ANALYSES
    )
    parser.add_argument(
        "--delta-tolerance", type=float, default=DEFAULT_DELTA_TOLERANCE
    )
    parser.add_argument(
        "--minimum-tolerance", type=float, default=DEFAULT_MINIMUM_TOLERANCE
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop immediately when one case fails."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        json_input_dir=args.json_input_dir,
        root_input_dir=args.root_input_dir,
        n_runs=args.n_runs,
        mu_sig=args.mu_sig,
        scan_min=args.scan_min,
        scan_max=args.scan_max,
        n_scan_points=args.n_scan_points,
        output=args.output,
        plot=args.plot,
        plot_dir=args.plot_dir,
        cases=args.cases,
        analyses=args.analyses,
        delta_tolerance=args.delta_tolerance,
        minimum_tolerance=args.minimum_tolerance,
        continue_on_case_error=not args.fail_fast,
    )


if __name__ == "__main__":
    main()
