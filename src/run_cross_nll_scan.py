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
    import pyhf
except ImportError:
    pyhf = None

try:
    import ROOT
except ImportError:
    ROOT = None


BENCHMARK_NAME = "cross_nll_scan"
DEFAULT_FRAMEWORKS = ["manual", "pyhs3", "pyhf", "roofit"]
DEFAULT_OUTPUT = RESULTS_DIR / BENCHMARK_NAME / f"{BENCHMARK_NAME}_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME


@dataclass(frozen=True)
class FrameworkSpec:
    name: str
    build_func: Callable[[], Any]
    eval_func: Callable[[Any, float], float]


def validate_workspace_path(workspace_path: Path) -> Path:
    if not workspace_path.exists():
        raise FileNotFoundError(f"Workspace file does not exist: {workspace_path}")
    if not workspace_path.is_file():
        raise FileNotFoundError(f"Workspace path is not a file: {workspace_path}")
    return workspace_path


def validate_positive_int(value: int, name: str, minimum: int = 1) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def validate_finite_float(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def validate_benchmark_config(
    *,
    mu_min: float,
    mu_max: float,
    n_points: int,
    warmup_iterations: int,
    shape_tolerance: float,
    minimum_tolerance: float,
    frameworks: list[str],
) -> None:
    validate_positive_int(n_points, "n_points", minimum=2)
    validate_finite_float(mu_min, "mu_min")
    validate_finite_float(mu_max, "mu_max")
    validate_finite_float(shape_tolerance, "shape_tolerance")
    validate_finite_float(minimum_tolerance, "minimum_tolerance")

    if mu_min >= mu_max:
        raise ValueError(
            f"mu_min must be smaller than mu_max, got {mu_min} >= {mu_max}"
        )
    if shape_tolerance <= 0.0:
        raise ValueError("shape_tolerance must be positive")
    if minimum_tolerance <= 0.0:
        raise ValueError("minimum_tolerance must be positive")
    if not frameworks:
        raise ValueError("At least one framework must be selected")

    unknown = sorted(set(frameworks) - set(DEFAULT_FRAMEWORKS))
    if unknown:
        raise ValueError(
            "Unknown frameworks: "
            + ", ".join(unknown)
            + f". Available frameworks: {', '.join(DEFAULT_FRAMEWORKS)}"
        )
    if "manual" not in frameworks:
        raise ValueError(
            "manual framework is required because it is the validation reference"
        )


def validate_parameters(parameters: dict[str, float], n_bins: int) -> None:
    required = ["mu"]
    for i in range(n_bins):
        required.extend([f"signal_{i}", f"background_{i}", f"obs_{i}"])

    missing = [name for name in required if name not in parameters]
    if missing:
        raise KeyError(
            f"Workspace initial parameter point is missing parameters: {missing}"
        )

    non_finite = [name for name in required if not math.isfinite(parameters[name])]
    if non_finite:
        raise ValueError(f"Workspace parameters must be finite: {non_finite}")

    negative_expected_inputs = [
        name for name in required if name != "mu" and parameters[name] < 0.0
    ]
    if negative_expected_inputs:
        raise ValueError(
            f"Binned likelihood inputs must be non-negative: {negative_expected_inputs}"
        )


def validate_scan_values(values: list[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} contains non-finite values")


def extract_parameters(ws: Workspace) -> dict[str, float]:
    try:
        parameter_set = ws.parameter_points.root[0]
        parameters = {p.name: float(p.value) for p in parameter_set.parameters}
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("Could not extract initial parameters from workspace") from exc

    if not parameters:
        raise ValueError(
            "Workspace initial parameter point does not contain parameters"
        )
    return parameters


def infer_n_bins_from_parameters(parameters: dict[str, float]) -> int:
    signal_indices = sorted(
        int(name.removeprefix("signal_"))
        for name in parameters
        if name.startswith("signal_") and name.removeprefix("signal_").isdigit()
    )

    if not signal_indices:
        raise ValueError(
            "Could not infer n_bins: no signal_<index> parameters found "
            "in the workspace initial parameter point"
        )

    expected_indices = list(range(signal_indices[-1] + 1))
    if signal_indices != expected_indices:
        raise ValueError(
            "Could not infer n_bins: signal indices must be contiguous from 0. "
            f"Found indices: {signal_indices}"
        )

    missing = []
    for index in expected_indices:
        for prefix in ("background", "obs"):
            name = f"{prefix}_{index}"
            if name not in parameters:
                missing.append(name)

    if missing:
        raise ValueError(
            "Could not infer n_bins: matching background_<index> and obs_<index> "
            f"parameters are missing: {missing}"
        )

    return len(signal_indices)


def poisson_nll(observed: float, expected: float) -> float:
    if expected <= 0.0:
        raise ValueError(f"Poisson expected value must be positive, got {expected}")
    if observed < 0.0:
        raise ValueError(f"Poisson observed value must be non-negative, got {observed}")
    return expected - observed * math.log(expected) + math.lgamma(observed + 1.0)


def get_vectors(
    parameters: dict[str, float],
    n_bins: int | None,
) -> tuple[list[float], list[float], list[float]]:
    validate_parameters(parameters, n_bins)
    signal = [parameters[f"signal_{i}"] for i in range(n_bins)]
    background = [parameters[f"background_{i}"] for i in range(n_bins)]
    observed = [parameters[f"obs_{i}"] for i in range(n_bins)]
    return signal, background, observed


def build_manual_model(
    parameters: dict[str, float],
    n_bins: int,
) -> tuple[list[float], list[float], list[float]]:
    return get_vectors(parameters, n_bins)


def manual_nll(
    model: tuple[list[float], list[float], list[float]],
    mu: float,
) -> float:
    signal, background, observed = model
    return sum(
        poisson_nll(obs, mu * sig + bkg)
        for sig, bkg, obs in zip(signal, background, observed, strict=True)
    )


def build_pyhs3_model(workspace_path: Path) -> Any:
    workspace = Workspace.load(workspace_path)
    return workspace.model("analysis", progress=False, mode="FAST_RUN")


def pyhs3_nll(model: Any, n_bins: int, mu_value: float) -> float:
    mu = np.asarray(mu_value, dtype=np.float64)
    log_likelihood = 0.0

    for i in range(n_bins):
        value = float(np.asarray(model.pdf(f"poisson_{i}", mu=mu)).reshape(-1)[0])
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(f"PyHS3 returned invalid PDF value for bin {i}: {value}")
        log_likelihood += math.log(value)

    return -log_likelihood


def build_pyhf_spec(parameters: dict[str, float], n_bins: int) -> dict[str, Any]:
    signal, background, observed = get_vectors(parameters, n_bins)
    return {
        "channels": [
            {
                "name": "channel",
                "samples": [
                    {
                        "name": "signal",
                        "data": signal,
                        "modifiers": [
                            {
                                "name": "mu",
                                "type": "normfactor",
                                "data": None,
                            }
                        ],
                    },
                    {
                        "name": "background",
                        "data": background,
                        "modifiers": [],
                    },
                ],
            }
        ],
        "observations": [
            {
                "name": "channel",
                "data": observed,
            }
        ],
        "measurements": [
            {
                "name": "measurement",
                "config": {
                    "poi": "mu",
                    "parameters": [],
                },
            }
        ],
        "version": "1.0.0",
    }


def build_pyhf_model(parameters: dict[str, float], n_bins: int) -> tuple[Any, Any]:
    if pyhf is None:
        raise RuntimeError("pyhf is not available in this environment")

    workspace = pyhf.Workspace(build_pyhf_spec(parameters, n_bins))
    model = workspace.model()
    data = workspace.data(model)
    return model, data


def pyhf_nll(model_and_data: tuple[Any, Any], mu_value: float) -> float:
    model, data = model_and_data
    pars = model.config.suggested_init()
    mu_index = model.config.par_order.index("mu")
    pars[mu_index] = mu_value

    logpdf = model.logpdf(pars, data)
    value = -float(np.asarray(logpdf).squeeze())
    if not math.isfinite(value):
        raise ValueError(f"pyhf returned non-finite NLL value: {value}")
    return value


def build_roofit_model(
    parameters: dict[str, float],
    n_bins: int,
    mu_value: float,
) -> dict[str, Any]:
    if ROOT is None:
        raise RuntimeError("ROOT is not available in this environment")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

    signal, background, observed = get_vectors(parameters, n_bins)
    mu = ROOT.RooRealVar("mu", "mu", mu_value)
    mu.setConstant(False)

    pdfs = ROOT.RooArgList()
    keepalive = [mu]
    poissons = []

    upper_bound = max(100.0, max(observed + background + signal) * 5.0)

    for i, (sig, bkg, obs_value) in enumerate(
        zip(signal, background, observed, strict=True)
    ):
        obs = ROOT.RooRealVar(
            f"obs_{i}",
            f"obs_{i}",
            obs_value,
            0.0,
            upper_bound,
        )
        obs.setConstant(True)

        expected = ROOT.RooFormulaVar(
            f"expected_{i}",
            f"@0 * {sig:.17g} + {bkg:.17g}",
            ROOT.RooArgList(mu),
        )
        poisson = ROOT.RooPoisson(
            f"poisson_{i}",
            f"poisson_{i}",
            obs,
            expected,
        )

        pdfs.add(poisson)
        poissons.append(poisson)
        keepalive.extend([obs, expected, poisson])

    likelihood = ROOT.RooProdPdf("likelihood", "likelihood", pdfs)
    keepalive.append(likelihood)

    return {
        "mu": mu,
        "poissons": poissons,
        "likelihood": likelihood,
        "keepalive": keepalive,
    }


def roofit_nll(model: dict[str, Any], mu_value: float) -> float:
    model["mu"].setVal(mu_value)

    total = 0.0
    for index, poisson in enumerate(model["poissons"]):
        value = float(poisson.getVal())
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(
                f"RooFit returned invalid PDF value for bin {index}: {value}"
            )
        total -= math.log(value)

    return total


def summarize_values(values: list[float]) -> dict[str, float]:
    validate_scan_values(values, "values")
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def build_mu_grid(mu_min: float, mu_max: float, n_points: int) -> list[float]:
    validate_benchmark_config(
        mu_min=mu_min,
        mu_max=mu_max,
        n_points=n_points,
        warmup_iterations=0,
        shape_tolerance=1.0,
        minimum_tolerance=1.0,
        frameworks=["manual"],
    )
    return [float(value) for value in np.linspace(mu_min, mu_max, n_points)]


def run_scan(
    model: Any,
    eval_func: Callable[[Any, float], float],
    mu_grid: list[float],
) -> list[float]:
    validate_scan_values(mu_grid, "mu_grid")
    values = [float(eval_func(model, mu)) for mu in mu_grid]
    validate_scan_values(values, "nll_values")
    return values


def minimum_position(mu_grid: list[float], nll_values: list[float]) -> float:
    if len(mu_grid) != len(nll_values):
        raise ValueError("mu_grid and nll_values must have the same length")
    validate_scan_values(nll_values, "nll_values")
    index = int(np.argmin(nll_values))
    return float(mu_grid[index])


def delta_nll_shape(nll_values: list[float]) -> list[float]:
    validate_scan_values(nll_values, "nll_values")
    minimum = min(nll_values)
    return [float(value - minimum) for value in nll_values]


def max_abs_difference(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Cannot compare arrays with different lengths")
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def mean_offset(reference: list[float], values: list[float]) -> float:
    if len(reference) != len(values):
        raise ValueError("Cannot compare arrays with different lengths")
    return mean(value - ref for ref, value in zip(reference, values, strict=True))


def measure_framework_scan(
    name: str,
    build_func: Callable[[], Any],
    eval_func: Callable[[Any, float], float],
    mu_grid: list[float],
    warmup_iterations: int = 1,
) -> dict[str, Any]:
    validate_scan_values(mu_grid, "mu_grid")
    validate_positive_int(warmup_iterations, "warmup_iterations", minimum=0)

    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    build_start = time.perf_counter()
    model = build_func()
    build_end = time.perf_counter()

    # Measure the true cold first call separately. This captures lazy initialization
    # costs such as expression compilation, graph construction, or cache population.
    cold_first_start = time.perf_counter()
    cold_first_nll = float(eval_func(model, mu_grid[0]))
    cold_first_end = time.perf_counter()

    # Run optional unmeasured warm-up evaluations before timing the steady-state scan.
    # Use the middle of the scan range to exercise a representative point.
    warmup_mu = mu_grid[len(mu_grid) // 2]
    warmup_start = time.perf_counter()
    for _ in range(warmup_iterations):
        _ = float(eval_func(model, warmup_mu))
    warmup_end = time.perf_counter()

    first_start = time.perf_counter()
    first_nll = float(eval_func(model, mu_grid[0]))
    first_end = time.perf_counter()

    scan_start = time.perf_counter()
    nll_values = run_scan(model, eval_func, mu_grid)
    scan_end = time.perf_counter()

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    delta_shape = delta_nll_shape(nll_values)
    full_scan_time = scan_end - scan_start

    result = {
        "framework": name,
        "plot_label": name,
        "n_points": len(mu_grid),
        "warmup_iterations": warmup_iterations,
        "cold_first_nll": cold_first_nll,
        "first_nll": first_nll,
        "nll_values": nll_values,
        "delta_nll_shape": delta_shape,
        "minimum_mu": minimum_position(mu_grid, nll_values),
        "model_build_time_seconds": build_end - build_start,
        "cold_first_evaluation_time_seconds": cold_first_end - cold_first_start,
        "warmup_time_seconds": warmup_end - warmup_start,
        "first_evaluation_time_seconds": first_end - first_start,
        "full_scan_time_seconds": full_scan_time,
        "time_per_scan_point_seconds": full_scan_time / len(mu_grid),
        "current_rss_before_mb": current_rss_before_mb,
        "current_rss_after_mb": current_rss_after_mb,
        "current_rss_delta_mb": max(0.0, current_rss_after_mb - current_rss_before_mb),
        "peak_rss_before_mb": peak_rss_before_mb,
        "peak_rss_after_mb": peak_rss_after_mb,
        "peak_rss_delta_mb": max(0.0, peak_rss_after_mb - peak_rss_before_mb),
        "rss_delta_mb": max(0.0, current_rss_after_mb - current_rss_before_mb),
        "nll_summary": summarize_values(nll_values),
        "delta_nll_summary": summarize_values(delta_shape),
        "status": "success",
    }

    validate_framework_result(result)
    return result


def validate_framework_result(result: dict[str, Any]) -> None:
    required_finite_fields = [
        "first_nll",
        "minimum_mu",
        "model_build_time_seconds",
        "cold_first_evaluation_time_seconds",
        "warmup_time_seconds",
        "first_evaluation_time_seconds",
        "full_scan_time_seconds",
        "time_per_scan_point_seconds",
        "current_rss_delta_mb",
        "peak_rss_delta_mb",
    ]
    for field in required_finite_fields:
        value = result[field]
        if not math.isfinite(value):
            raise ValueError(
                f"{result['framework']} result field {field} is not finite"
            )

    if result["model_build_time_seconds"] < 0.0:
        raise ValueError("model_build_time_seconds must be non-negative")
    if result["cold_first_evaluation_time_seconds"] < 0.0:
        raise ValueError("cold_first_evaluation_time_seconds must be non-negative")
    if result["warmup_time_seconds"] < 0.0:
        raise ValueError("warmup_time_seconds must be non-negative")
    if result["first_evaluation_time_seconds"] < 0.0:
        raise ValueError("first_evaluation_time_seconds must be non-negative")
    if result["full_scan_time_seconds"] <= 0.0:
        raise ValueError("full_scan_time_seconds must be positive")
    validate_scan_values(result["nll_values"], "nll_values")
    validate_scan_values(result["delta_nll_shape"], "delta_nll_shape")


def add_scan_validation(
    results: list[dict[str, Any]],
    shape_tolerance: float,
    minimum_tolerance: float,
) -> None:
    if not results:
        raise ValueError("Cannot validate empty benchmark results")

    successful_results = [
        result for result in results if result.get("status") == "success"
    ]
    reference = next(
        (result for result in successful_results if result["framework"] == "manual"),
        None,
    )
    if reference is None:
        raise ValueError("Cannot validate cross-framework scan without manual result")

    reference_values = reference["nll_values"]
    reference_shape = reference["delta_nll_shape"]
    reference_minimum = reference["minimum_mu"]

    for result in results:
        if result.get("status") != "success":
            result["constant_offset_estimate"] = None
            result["delta_nll_shape_max_abs_diff"] = None
            result["minimum_mu_abs_diff"] = None
            result["delta_nll_shape_success"] = False
            result["minimum_mu_success"] = False
            result["validation_status"] = "not_run"
            continue

        result["constant_offset_estimate"] = mean_offset(
            reference_values,
            result["nll_values"],
        )
        result["delta_nll_shape_max_abs_diff"] = max_abs_difference(
            reference_shape,
            result["delta_nll_shape"],
        )
        result["minimum_mu_abs_diff"] = abs(result["minimum_mu"] - reference_minimum)
        result["delta_nll_shape_success"] = (
            result["delta_nll_shape_max_abs_diff"] <= shape_tolerance
        )
        result["minimum_mu_success"] = (
            result["minimum_mu_abs_diff"] <= minimum_tolerance
        )
        result["validation_status"] = (
            "success"
            if result["delta_nll_shape_success"] and result["minimum_mu_success"]
            else "failed"
        )


def failed_framework_result(
    framework: str,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "framework": framework,
        "plot_label": framework,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def make_framework_specs(
    frameworks: Iterable[str],
    parameters: dict[str, float],
    workspace_path: Path,
    n_bins: int,
    mu_min: float,
) -> list[FrameworkSpec]:
    specs: list[FrameworkSpec] = []

    for framework in frameworks:
        if framework == "manual":
            specs.append(
                FrameworkSpec(
                    name="manual",
                    build_func=lambda parameters=parameters: build_manual_model(
                        parameters,
                        n_bins,
                    ),
                    eval_func=lambda model, mu: manual_nll(model, mu),
                )
            )
        elif framework == "pyhs3":
            specs.append(
                FrameworkSpec(
                    name="pyhs3",
                    build_func=lambda workspace_path=workspace_path: build_pyhs3_model(
                        workspace_path,
                    ),
                    eval_func=lambda model, mu: pyhs3_nll(model, n_bins, mu),
                )
            )
        elif framework == "pyhf":
            specs.append(
                FrameworkSpec(
                    name="pyhf",
                    build_func=lambda parameters=parameters: build_pyhf_model(
                        parameters,
                        n_bins,
                    ),
                    eval_func=lambda model, mu: pyhf_nll(model, mu),
                )
            )
        elif framework == "roofit":
            specs.append(
                FrameworkSpec(
                    name="roofit",
                    build_func=lambda parameters=parameters: build_roofit_model(
                        parameters,
                        n_bins,
                        mu_min,
                    ),
                    eval_func=lambda model, mu: roofit_nll(model, mu),
                )
            )
        else:  # pragma: no cover - protected by validation
            raise ValueError(f"Unknown framework: {framework}")

    return specs


def _framework_order(results: list[dict[str, Any]]) -> list[str]:
    """Return successful framework names in execution order."""

    return [
        result["framework"] for result in results if result.get("status") == "success"
    ]


FRAMEWORK_STYLE = {
    "manual": {
        "label": "Manual reference",
        "color": "#1f1f1f",
        "marker": "o",
        "linestyle": "-",
    },
    "pyhs3": {"label": "PyHS3", "color": "#0055A4", "marker": "s", "linestyle": "--"},
    "pyhf": {"label": "pyhf", "color": "#E57200", "marker": "^", "linestyle": "-."},
    "roofit": {"label": "RooFit", "color": "#009E73", "marker": "D", "linestyle": ":"},
}


def _apply_cern_style() -> None:
    """Apply a clean HEP-paper style for benchmark plots."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.4,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
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


def _successful_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if result.get("status") == "success"]


def _style_for(framework: str) -> dict[str, str]:
    return FRAMEWORK_STYLE.get(
        framework,
        {"label": framework, "color": "#4D4D4D", "marker": "o", "linestyle": "-"},
    )


def _framework_label(framework: str) -> str:
    return _style_for(framework)["label"]


def _format_metric(value: float, unit: str = "") -> str:
    if not math.isfinite(value):
        return "nan"
    if value == 0:
        return f"0{unit}"
    abs_value = abs(value)
    if abs_value < 1e-3 or abs_value >= 1e4:
        return f"{value:.2e}{unit}"
    if abs_value < 10:
        return f"{value:.3f}{unit}"
    if abs_value < 100:
        return f"{value:.2f}{unit}"
    return f"{value:.1f}{unit}"


def _save_figure(fig: Any, output_path: Path) -> None:
    """Save a plot as a high-resolution PNG image."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_path.with_suffix(".png")

    try:
        fig.savefig(png_path, dpi=300)
    except OSError as exc:
        raise OSError(f"Failed to save plot to {png_path}") from exc
    finally:
        plt.close(fig)


def _reference_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    reference = next(
        (
            result
            for result in results
            if result.get("framework") == "manual" and result.get("status") == "success"
        ),
        None,
    )
    if reference is None:
        raise ValueError("Manual reference result is required for plotting")
    return reference


def make_nll_profile_plot(
    results: list[dict[str, Any]],
    mu_grid: list[float],
    output_path: Path,
) -> None:
    """Create the main physics plot: ΔNLL overlay plus residuals to manual."""

    _apply_cern_style()
    successful = _successful_results(results)
    reference = _reference_result(successful)
    reference_delta = np.asarray(reference["delta_nll_shape"], dtype=float)
    x = np.asarray(mu_grid, dtype=float)

    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.15], "hspace": 0.06},
    )

    for result in successful:
        framework = result["framework"]
        style = _style_for(framework)
        values = np.asarray(result["delta_nll_shape"], dtype=float)
        residual = values - reference_delta

        zorder = 5 if framework == "manual" else 4
        linewidth = 2.8 if framework == "manual" else 2.2
        alpha = 1.0 if framework == "manual" else 0.88

        ax.plot(
            x,
            values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=linewidth,
            alpha=alpha,
            label=style["label"],
            zorder=zorder,
        )
        ax.scatter(
            x[:: max(1, len(x) // 14)],
            values[:: max(1, len(x) // 14)],
            color=style["color"],
            marker=style["marker"],
            s=28,
            zorder=zorder + 1,
        )

        if framework != "manual":
            residual_ax.plot(
                x,
                residual,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                label=style["label"],
            )

    ax.set_ylabel(r"$\Delta$NLL")
    ax.set_title("Cross-framework NLL scan agreement", loc="left", weight="bold")
    ax.text(
        0.02,
        0.94,
        "PyHS3 benchmark · binned Poisson likelihood",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.62, 1.02))
    ax.margins(x=0.015)

    residual_ax.axhline(0.0, color="black", linewidth=1.2, alpha=0.75)
    residual_ax.set_xlabel(r"Signal strength $\mu$")
    residual_ax.set_ylabel("Residual\nvs manual")
    residual_ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    residual_ax.margins(x=0.015)

    fig.align_ylabels([ax, residual_ax])
    _save_figure(fig, output_path)


def make_timing_profile_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    """Compare setup and scan timings on one log-scale dot plot."""

    _apply_cern_style()
    successful = _successful_results(results)
    metrics = [
        ("Model build", "model_build_time_seconds", 1000.0, "ms"),
        ("Cold first eval", "cold_first_evaluation_time_seconds", 1000.0, "ms"),
        ("Warm first eval", "first_evaluation_time_seconds", 1000.0, "ms"),
        ("Full scan", "full_scan_time_seconds", 1000.0, "ms"),
        ("Per scan point", "time_per_scan_point_seconds", 1e6, "µs"),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    y_positions = np.arange(len(metrics))
    offsets = np.linspace(-0.24, 0.24, max(1, len(successful)))

    for framework_index, result in enumerate(successful):
        framework = result["framework"]
        style = _style_for(framework)
        values = [max(float(result[key]) * scale, 1e-9) for _, key, scale, _ in metrics]
        y = y_positions + offsets[framework_index]
        ax.scatter(
            values,
            y,
            s=95,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            zorder=4,
        )
        for value, y_value, (_, _key, _scale, unit) in zip(
            values, y, metrics, strict=True
        ):
            ax.text(
                value * 1.12,
                y_value,
                _format_metric(value, f" {unit}"),
                va="center",
                fontsize=10,
                color=style["color"],
            )

    ax.set_xscale("log")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([name for name, *_ in metrics])
    ax.invert_yaxis()
    ax.set_xlabel("Runtime (log scale; units shown in labels)")
    ax.set_title("Runtime profile by framework", loc="left", weight="bold")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    ax.grid(True, which="both", axis="x", alpha=0.45)
    ax.grid(False, axis="y")

    _save_figure(fig, output_path)


def make_relative_runtime_plot(
    results: list[dict[str, Any]], output_path: Path
) -> None:
    """Rank frameworks by warm scan throughput relative to the fastest one."""

    _apply_cern_style()
    successful = sorted(
        _successful_results(results),
        key=lambda result: result["time_per_scan_point_seconds"],
    )
    fastest = successful[0]["time_per_scan_point_seconds"]

    labels = [_framework_label(result["framework"]) for result in successful]
    relative = [
        result["time_per_scan_point_seconds"] / fastest for result in successful
    ]
    raw_us = [result["time_per_scan_point_seconds"] * 1e6 for result in successful]
    colors = [_style_for(result["framework"])["color"] for result in successful]

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    y = np.arange(len(successful))
    bars = ax.barh(y, relative, color=colors, alpha=0.92)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Relative time per scan point (fastest = 1×)")
    ax.set_title("NLL scan throughput ranking", loc="left", weight="bold")
    ax.grid(True, axis="x", alpha=0.45)
    ax.grid(False, axis="y")

    for bar, rel_value, us_value in zip(bars, relative, raw_us, strict=True):
        ax.text(
            bar.get_width() * 1.02,
            bar.get_y() + bar.get_height() / 2,
            f"{rel_value:.2f}×  ({us_value:.1f} µs/point)",
            va="center",
            fontsize=11,
            weight="bold",
        )

    ax.set_xlim(0.0, max(relative) * 1.35)
    _save_figure(fig, output_path)


def make_memory_profile_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    """Plot current and peak RSS deltas without letting zero-valued bars dominate."""

    _apply_cern_style()
    successful = _successful_results(results)
    labels = [_framework_label(result["framework"]) for result in successful]
    current = [float(result["rss_delta_mb"]) for result in successful]
    peak = [float(result["peak_rss_delta_mb"]) for result in successful]

    x = np.arange(len(successful))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.8, 6.6))
    current_plot = [max(value, 1e-3) for value in current]
    peak_plot = [max(value, 1e-3) for value in peak]

    current_bars = ax.bar(
        x - width / 2, current_plot, width, label="Current RSS delta", alpha=0.85
    )
    peak_bars = ax.bar(
        x + width / 2, peak_plot, width, label="Peak RSS delta", alpha=0.85
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Memory delta [MB] (log scale)")
    ax.set_title("Memory footprint during NLL scan", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(True, which="both", axis="y", alpha=0.45)
    ax.grid(False, axis="x")

    for bars, raw_values in ((current_bars, current), (peak_bars, peak)):
        for bar, raw in zip(bars, raw_values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15,
                _format_metric(raw, " MB"),
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
                rotation=0,
            )

    _save_figure(fig, output_path)


def make_numerical_agreement_plot(
    results: list[dict[str, Any]],
    shape_tolerance: float,
    output_path: Path,
) -> None:
    """Show numerical agreement in a way that remains readable near machine precision."""

    _apply_cern_style()
    successful = [
        result
        for result in _successful_results(results)
        if result["framework"] != "manual"
    ]
    labels = [_framework_label(result["framework"]) for result in successful]
    diffs = [float(result["delta_nll_shape_max_abs_diff"]) for result in successful]
    floor = max(shape_tolerance * 1e-6, 1e-16)
    plot_values = [max(value, floor) for value in diffs]
    colors = [_style_for(result["framework"])["color"] for result in successful]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(successful))
    bars = ax.bar(x, plot_values, color=colors, alpha=0.92)
    ax.axhline(
        shape_tolerance,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"tolerance = {shape_tolerance:.0e}",
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"max $|\Delta$NLL$_{framework}$ - $\Delta$NLL$_{manual}|$")
    ax.set_title("Numerical agreement with manual reference", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(True, which="both", axis="y", alpha=0.45)
    ax.grid(False, axis="x")

    for bar, raw_value in zip(bars, diffs, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.25,
            f"{raw_value:.2e}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    _save_figure(fig, output_path)


def make_summary_table_plot(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Create a compact report-card style summary for quick inspection."""

    _apply_cern_style()
    successful = _successful_results(results)
    columns = [
        "Framework",
        "Validation",
        "Build [ms]",
        "Cold eval [ms]",
        "Warm eval [ms]",
        "Scan [ms]",
        "µs / point",
        "RSS Δ [MB]",
        "max ΔNLL diff",
    ]
    rows = []
    for result in successful:
        rows.append(
            [
                _framework_label(result["framework"]),
                result.get("validation_status", "n/a"),
                f"{result['model_build_time_seconds'] * 1000.0:.2f}",
                f"{result['cold_first_evaluation_time_seconds'] * 1000.0:.2f}",
                f"{result['first_evaluation_time_seconds'] * 1000.0:.2f}",
                f"{result['full_scan_time_seconds'] * 1000.0:.2f}",
                f"{result['time_per_scan_point_seconds'] * 1e6:.2f}",
                f"{result['rss_delta_mb']:.2f}",
                f"{result.get('delta_nll_shape_max_abs_diff', 0.0):.2e}",
            ]
        )

    fig, ax = plt.subplots(figsize=(13.5, 3.6 + 0.35 * len(rows)))
    ax.axis("off")
    ax.set_title("Cross-framework NLL scan summary", loc="left", weight="bold", pad=16)
    ax.text(
        0.0,
        0.94,
        "Generated from one common binned Poisson workspace; manual implementation is the validation reference.",
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
    table.set_fontsize(10)
    table.scale(1.0, 1.65)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("0.75")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("0.15")
        elif col == 0:
            framework = successful[row - 1]["framework"]
            cell.set_text_props(weight="bold", color=_style_for(framework)["color"])
        elif col == 1 and cell.get_text().get_text() == "success":
            cell.set_text_props(weight="bold", color="#00843D")

    _save_figure(fig, output_path)


def make_plots(
    results: list[dict[str, Any]],
    mu_grid: list[float],
    plot_dir: Path,
    shape_tolerance: float = 1e-9,
) -> None:
    """Create the production plot set for the cross-framework NLL scan.

    This intentionally omits low-information plots such as raw constant offsets
    and best-fit mu bars when all frameworks agree exactly. The retained figures
    answer the questions that matter for the report: numerical agreement,
    runtime profile, throughput ranking, and memory footprint.
    """

    plot_dir.mkdir(parents=True, exist_ok=True)
    if not _successful_results(results):
        raise ValueError("No successful benchmark results available for plotting")

    make_nll_profile_plot(
        results=results,
        mu_grid=mu_grid,
        output_path=plot_dir / "cross_nll_scan_profile.png",
    )
    make_timing_profile_plot(
        results=results,
        output_path=plot_dir / "cross_nll_timing_profile.png",
    )
    make_relative_runtime_plot(
        results=results,
        output_path=plot_dir / "cross_nll_relative_runtime.png",
    )
    make_memory_profile_plot(
        results=results,
        output_path=plot_dir / "cross_nll_memory_profile.png",
    )
    make_numerical_agreement_plot(
        results=results,
        shape_tolerance=shape_tolerance,
        output_path=plot_dir / "cross_nll_numerical_agreement.png",
    )
    make_summary_table_plot(
        results=results,
        output_path=plot_dir / "cross_nll_summary_table.png",
    )


def print_result(result: dict[str, Any]) -> None:
    print()
    print("-" * 72)
    print(result["framework"])
    print("-" * 72)

    if result.get("status") != "success":
        print(f"status:                  {result.get('status')}")
        print(
            f"error:                   {result.get('error_type')}: {result.get('error_message')}"
        )
        return

    print(f"status:                  {result['status']}")
    print(f"validation:              {result['validation_status']}")
    print(f"minimum mu:              {result['minimum_mu']:.15f}")
    print(f"minimum mu abs diff:     {result['minimum_mu_abs_diff']:.15e}")
    print(f"shape max abs diff:      {result['delta_nll_shape_max_abs_diff']:.15e}")
    print(f"constant offset:         {result['constant_offset_estimate']:.15e}")
    print(
        f"model build:             {result['model_build_time_seconds'] * 1000.0:.3f} ms"
    )
    print(
        f"cold first evaluation:   {result['cold_first_evaluation_time_seconds'] * 1000.0:.3f} ms"
    )
    print(
        f"warm-up evaluations:     {result['warmup_iterations']} ({result['warmup_time_seconds'] * 1000.0:.3f} ms total)"
    )
    print(
        f"warm first evaluation:   {result['first_evaluation_time_seconds'] * 1e6:.3f} us"
    )
    print(
        f"full scan:               {result['full_scan_time_seconds'] * 1000.0:.3f} ms"
    )
    print(
        f"time per scan point:     {result['time_per_scan_point_seconds'] * 1e6:.3f} us"
    )
    print(f"current RSS delta:       {result['rss_delta_mb']:.3f} MB")
    print(f"peak RSS delta:          {result['peak_rss_delta_mb']:.3f} MB")


def build_failed_output(
    *,
    workspace_path: Path,
    n_bins: int | None,
    mu_min: float,
    mu_max: float,
    n_points: int,
    warmup_iterations: int,
    shape_tolerance: float,
    minimum_tolerance: float,
    frameworks: list[str],
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "n_bins": n_bins,
        "mu_min": mu_min,
        "mu_max": mu_max,
        "n_points": n_points,
        "warmup_iterations": warmup_iterations,
        "shape_tolerance": shape_tolerance,
        "minimum_tolerance": minimum_tolerance,
        "frameworks": frameworks,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "results": [],
    }


def run(
    workspace_path: Path,
    mu_min: float,
    mu_max: float,
    n_points: int,
    output: Path,
    plot: bool,
    plot_dir: Path,
    shape_tolerance: float,
    minimum_tolerance: float,
    frameworks: list[str] | None = None,
    continue_on_framework_error: bool = True,
    warmup_iterations: int = 1,
) -> dict[str, Any]:
    selected_frameworks = frameworks or list(DEFAULT_FRAMEWORKS)
    n_bins: int | None = None

    try:
        workspace_path = validate_workspace_path(workspace_path)
        validate_benchmark_config(
            mu_min=mu_min,
            mu_max=mu_max,
            n_points=n_points,
            warmup_iterations=warmup_iterations,
            shape_tolerance=shape_tolerance,
            minimum_tolerance=minimum_tolerance,
            frameworks=selected_frameworks,
        )
        validate_positive_int(warmup_iterations, "warmup_iterations", minimum=0)

        workspace = Workspace.load(workspace_path)
        parameters = extract_parameters(workspace)
        n_bins = infer_n_bins_from_parameters(parameters)
        validate_parameters(parameters, n_bins)
        mu_grid = build_mu_grid(mu_min=mu_min, mu_max=mu_max, n_points=n_points)

        results = []
        specs = make_framework_specs(
            selected_frameworks,
            parameters,
            workspace_path,
            n_bins,
            mu_min,
        )

        for spec in specs:
            try:
                results.append(
                    measure_framework_scan(
                        name=spec.name,
                        build_func=spec.build_func,
                        eval_func=spec.eval_func,
                        mu_grid=mu_grid,
                        warmup_iterations=warmup_iterations,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - intentional per-framework isolation
                if not continue_on_framework_error:
                    raise RuntimeError(
                        f"Framework benchmark failed for {spec.name}"
                    ) from exc
                results.append(failed_framework_result(spec.name, exc))

        add_scan_validation(
            results=results,
            shape_tolerance=shape_tolerance,
            minimum_tolerance=minimum_tolerance,
        )

        successful_results = [
            result for result in results if result.get("status") == "success"
        ]
        successful_validation = [
            result
            for result in successful_results
            if result.get("validation_status") == "success"
        ]

        status = (
            "success"
            if len(successful_validation) == len(selected_frameworks)
            else "failed"
        )

        output_data = {
            "benchmark": BENCHMARK_NAME,
            "workspace": workspace_path.name,
            "workspace_path": str(workspace_path),
            "n_bins": n_bins,
            "mu_min": mu_min,
            "mu_max": mu_max,
            "n_points": n_points,
            "warmup_iterations": warmup_iterations,
            "shape_tolerance": shape_tolerance,
            "minimum_tolerance": minimum_tolerance,
            "frameworks": selected_frameworks,
            "successful_frameworks": _framework_order(results),
            "failed_frameworks": [
                result["framework"]
                for result in results
                if result.get("status") != "success"
            ],
            "status": status,
            "mu_grid": mu_grid,
            "results": results,
        }

        print("=" * 80)
        print("Cross-framework NLL scan benchmark")
        print("=" * 80)
        print(f"Workspace:  {workspace_path.name}")
        print(f"Bins:       {n_bins}")
        print(f"Grid:       [{mu_min}, {mu_max}] with {n_points} points")
        print(
            f"Warm-up:    {warmup_iterations} unmeasured evaluation(s) after cold first call"
        )
        print(f"Frameworks: {', '.join(selected_frameworks)}")
        print(f"Status:     {status}")

        for result in results:
            print_result(result)

        save_json(output_data, output)
        print()
        print(f"Saved result to {output}")

        if plot:
            make_plots(results, mu_grid, plot_dir, shape_tolerance=shape_tolerance)
            print(f"Saved plots to {plot_dir}")

        return output_data

    except Exception as exc:  # noqa: BLE001 - write structured failure file
        output_data = build_failed_output(
            workspace_path=workspace_path,
            n_bins=n_bins,
            mu_min=mu_min,
            mu_max=mu_max,
            n_points=n_points,
            warmup_iterations=warmup_iterations,
            shape_tolerance=shape_tolerance,
            minimum_tolerance=minimum_tolerance,
            frameworks=selected_frameworks,
            exc=exc,
        )
        try:
            save_json(output_data, output)
        except Exception:  # noqa: BLE001
            print(
                "Failed to save benchmark failure report:\n" + traceback.format_exc(),
                file=sys.stderr,
            )
        raise RuntimeError("Cross-framework NLL scan benchmark failed") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a cross-framework NLL scan benchmark for the generated "
            "binned Poisson likelihood models."
        )
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--mu-min", type=float, default=0.0)
    parser.add_argument("--mu-max", type=float, default=2.0)
    parser.add_argument("--n-points", type=int, default=101)
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=1,
        help=(
            "Number of unmeasured warm-up evaluations to run after measuring "
            "the cold first call and before the timed scan."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--shape-tolerance", type=float, default=1e-9)
    parser.add_argument("--minimum-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--frameworks",
        nargs="+",
        choices=DEFAULT_FRAMEWORKS,
        default=DEFAULT_FRAMEWORKS,
        help=(
            "Frameworks to run. manual is required because it is used as "
            "the numerical reference."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one framework fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        workspace_path=args.workspace,
        mu_min=args.mu_min,
        mu_max=args.mu_max,
        n_points=args.n_points,
        output=args.output,
        plot=args.plot,
        plot_dir=args.plot_dir,
        shape_tolerance=args.shape_tolerance,
        minimum_tolerance=args.minimum_tolerance,
        frameworks=args.frameworks,
        continue_on_framework_error=not args.fail_fast,
        warmup_iterations=args.warmup_iterations,
    )


if __name__ == "__main__":
    main()
