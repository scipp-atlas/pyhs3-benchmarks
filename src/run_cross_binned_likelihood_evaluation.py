"""Cross-framework binned likelihood evaluation benchmark.

This benchmark compares negative log-likelihood (NLL) evaluation for equivalent
binned Poisson likelihood models across a manual reference implementation,
PyHS3, pyhf, and RooFit.

The benchmark measures model construction time, cold first evaluation time,
warm repeated evaluation time, memory usage, and numerical agreement for both
raw NLL and delta-NLL values.
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
import pyhf

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

from pyhs3.workspace import Workspace


BENCHMARK_NAME = "cross_binned_likelihood_evaluation"
BENCHMARK_TITLE = "Cross-framework binned likelihood evaluation benchmark"

DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = f"{BENCHMARK_NAME}_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME

SUPPORTED_FRAMEWORKS = ("manual", "pyhs3", "pyhf", "roofit")
DEFAULT_FRAMEWORKS = ["manual", "pyhs3", "pyhf", "roofit"]

PLOT_EPSILON = 1e-300
FRAMEWORK_STYLE: dict[str, dict[str, Any]] = {
    "manual": {"label": "Manual", "color": "#4d4d4d", "marker": "o", "hatch": ""},
    "pyhs3": {"label": "PyHS3", "color": "#1764ab", "marker": "s", "hatch": "//"},
    "pyhf": {"label": "pyhf", "color": "#f57c00", "marker": "^", "hatch": "\\\\"},
    "roofit": {"label": "RooFit", "color": "#009b77", "marker": "D", "hatch": ".."},
}


class BenchmarkConfigurationError(ValueError):
    """Raised when the benchmark configuration is invalid."""


class ValidationFailure(RuntimeError):
    """Raised when a framework result fails numerical validation."""


@dataclass(frozen=True)
class BenchmarkConfig:
    workspace_path: Path
    frameworks: list[str]
    n_bins: int | None
    mu: float
    delta_reference_mu: float
    n_runs: int
    raw_tolerance: float
    delta_tolerance: float
    output_dir: Path
    output_name: str
    plot: bool
    plot_dir: Path
    fail_fast: bool


@dataclass(frozen=True)
class BinnedVectors:
    signal: np.ndarray
    background: np.ndarray
    observed: np.ndarray

    @property
    def n_bins(self) -> int:
        return int(self.signal.size)


def _framework_label(framework: str) -> str:
    return FRAMEWORK_STYLE.get(framework, {"label": framework})["label"]


def _style_for(framework: str) -> dict[str, Any]:
    return FRAMEWORK_STYLE.get(
        framework,
        {"label": framework, "color": "#333333", "marker": "o", "hatch": ""},
    )


def _apply_cern_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 13,
            "axes.titlesize": 20,
            "axes.labelsize": 15,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _format_compact(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    abs_value = abs(value)
    if abs_value == 0:
        return "0"
    if abs_value >= 1000:
        return f"{value:.2g}"
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    return f"{value:.2g}"


def _format_scientific(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.1e}"


def _safe_log_value(value: float, floor: float = PLOT_EPSILON) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return floor
    return float(value)


def _successful_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if result.get("status") == "success"]


def validate_config(config: BenchmarkConfig) -> None:
    if not config.workspace_path.exists():
        raise FileNotFoundError(
            f"Workspace file does not exist: {config.workspace_path}"
        )
    if not config.workspace_path.is_file():
        raise BenchmarkConfigurationError(
            f"Workspace path is not a file: {config.workspace_path}"
        )

    if not config.frameworks:
        raise BenchmarkConfigurationError("At least one framework must be selected")
    invalid_frameworks = sorted(set(config.frameworks) - set(SUPPORTED_FRAMEWORKS))
    if invalid_frameworks:
        raise BenchmarkConfigurationError(
            f"Unsupported framework(s): {invalid_frameworks}. Supported: {list(SUPPORTED_FRAMEWORKS)}"
        )

    if config.n_bins is not None and config.n_bins < 1:
        raise BenchmarkConfigurationError("--n-bins must be at least 1 when provided")
    if not math.isfinite(config.mu) or config.mu < 0.0:
        raise BenchmarkConfigurationError("--mu must be a finite non-negative value")
    if not math.isfinite(config.delta_reference_mu) or config.delta_reference_mu < 0.0:
        raise BenchmarkConfigurationError(
            "--delta-reference-mu must be a finite non-negative value"
        )
    if math.isclose(config.mu, config.delta_reference_mu, rel_tol=0.0, abs_tol=1e-15):
        raise BenchmarkConfigurationError(
            "--mu and --delta-reference-mu must be different"
        )
    if config.n_runs < 1:
        raise BenchmarkConfigurationError("--n-runs must be at least 1")
    if not math.isfinite(config.raw_tolerance) or config.raw_tolerance <= 0.0:
        raise BenchmarkConfigurationError(
            "--raw-tolerance must be a positive finite value"
        )
    if not math.isfinite(config.delta_tolerance) or config.delta_tolerance <= 0.0:
        raise BenchmarkConfigurationError(
            "--delta-tolerance must be a positive finite value"
        )
    if "roofit" in config.frameworks and ROOT is None:
        raise BenchmarkConfigurationError(
            "RooFit was requested, but ROOT is not available in this environment"
        )


def load_workspace(workspace_path: Path) -> Workspace:
    return Workspace.load(workspace_path)


def extract_parameters(workspace: Workspace) -> dict[str, float]:
    try:
        parameter_set = workspace.parameter_points.root[0]
    except Exception as error:  # noqa: BLE001
        raise BenchmarkConfigurationError(
            "Workspace does not contain a root parameter point"
        ) from error

    parameters: dict[str, float] = {}
    for parameter in parameter_set.parameters:
        try:
            parameters[parameter.name] = float(parameter.value)
        except (TypeError, ValueError) as error:
            raise BenchmarkConfigurationError(
                f"Parameter '{parameter.name}' cannot be converted to float: {parameter.value!r}"
            ) from error
    return parameters


def infer_n_bins(parameters: dict[str, float]) -> int:
    signal_indices: set[int] = set()
    background_indices: set[int] = set()
    observed_indices: set[int] = set()

    for name in parameters:
        if name.startswith("signal_"):
            signal_indices.add(int(name.removeprefix("signal_")))
        elif name.startswith("background_"):
            background_indices.add(int(name.removeprefix("background_")))
        elif name.startswith("obs_"):
            observed_indices.add(int(name.removeprefix("obs_")))

    common = signal_indices & background_indices & observed_indices
    if not common:
        raise BenchmarkConfigurationError(
            "Could not infer n_bins from workspace parameters. Expected signal_i, background_i, obs_i."
        )

    n_bins = max(common) + 1
    expected = set(range(n_bins))
    if (
        signal_indices != expected
        or background_indices != expected
        or observed_indices != expected
    ):
        raise BenchmarkConfigurationError(
            "Workspace bin parameters are incomplete or non-contiguous. "
            "Expected signal_i, background_i, obs_i for every bin."
        )
    return n_bins


def validate_n_bins_against_parameters(
    parameters: dict[str, float], n_bins: int
) -> None:
    missing = []
    for index in range(n_bins):
        for prefix in ("signal", "background", "obs"):
            name = f"{prefix}_{index}"
            if name not in parameters:
                missing.append(name)
    if missing:
        raise BenchmarkConfigurationError(
            f"Workspace is missing required bin parameter(s): {', '.join(missing[:10])}"
            + (" ..." if len(missing) > 10 else "")
        )


def get_vectors(parameters: dict[str, float], n_bins: int) -> BinnedVectors:
    validate_n_bins_against_parameters(parameters, n_bins)
    signal = np.asarray([parameters[f"signal_{i}"] for i in range(n_bins)], dtype=float)
    background = np.asarray(
        [parameters[f"background_{i}"] for i in range(n_bins)], dtype=float
    )
    observed = np.asarray([parameters[f"obs_{i}"] for i in range(n_bins)], dtype=float)

    if np.any(signal < 0.0) or np.any(background < 0.0) or np.any(observed < 0.0):
        raise BenchmarkConfigurationError(
            "Signal, background, and observed bin values must be non-negative"
        )
    return BinnedVectors(signal=signal, background=background, observed=observed)


def poisson_nll(observed: float, expected: float) -> float:
    if expected <= 0.0 or not math.isfinite(expected):
        raise ValueError(f"Expected count must be positive and finite, got {expected}")
    if observed < 0.0 or not math.isfinite(observed):
        raise ValueError(
            f"Observed count must be non-negative and finite, got {observed}"
        )
    return expected - observed * math.log(expected) + math.lgamma(observed + 1.0)


def manual_nll_from_vectors(vectors: BinnedVectors, mu: float) -> float:
    expected = mu * vectors.signal + vectors.background
    if np.any(expected <= 0.0):
        raise ValueError(
            "Encountered non-positive expected count during manual NLL evaluation"
        )
    terms = [
        poisson_nll(float(obs), float(exp))
        for obs, exp in zip(vectors.observed, expected, strict=True)
    ]
    return float(sum(terms))


def build_manual_model(parameters: dict[str, float], n_bins: int) -> BinnedVectors:
    return get_vectors(parameters, n_bins)


def manual_nll(vectors: BinnedVectors, mu: float) -> float:
    return manual_nll_from_vectors(vectors, mu)


def build_pyhs3_model(workspace_path: Path) -> Any:
    workspace = Workspace.load(workspace_path)
    return workspace.model("analysis", progress=False, mode="FAST_RUN")


def pyhs3_nll(model: Any, n_bins: int, mu_value: float) -> float:
    mu = np.asarray(mu_value, dtype=np.float64)
    log_likelihood = 0.0
    for index in range(n_bins):
        value = float(np.asarray(model.pdf(f"poisson_{index}", mu=mu)).squeeze())
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(
                f"PyHS3 returned invalid PDF value for bin {index}: {value}"
            )
        log_likelihood += math.log(value)
    return -float(log_likelihood)


def build_pyhf_model(parameters: dict[str, float], n_bins: int) -> tuple[Any, Any]:
    vectors = get_vectors(parameters, n_bins)
    spec = {
        "channels": [
            {
                "name": "channel",
                "samples": [
                    {
                        "name": "signal",
                        "data": vectors.signal.tolist(),
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": "background",
                        "data": vectors.background.tolist(),
                        "modifiers": [],
                    },
                ],
            }
        ],
        "observations": [{"name": "channel", "data": vectors.observed.tolist()}],
        "measurements": [
            {"name": "measurement", "config": {"poi": "mu", "parameters": []}}
        ],
        "version": "1.0.0",
    }
    workspace = pyhf.Workspace(spec)
    model = workspace.model()
    data = workspace.data(model)
    return model, data


def pyhf_nll(model_and_data: tuple[Any, Any], mu_value: float) -> float:
    model, data = model_and_data
    pars = model.config.suggested_init()
    mu_index = model.config.par_order.index("mu")
    pars[mu_index] = mu_value
    logpdf = model.logpdf(pars, data)
    return -float(np.asarray(logpdf).squeeze())


def build_roofit_model(
    parameters: dict[str, float], n_bins: int, mu_value: float
) -> dict[str, Any]:
    if ROOT is None:
        raise RuntimeError("ROOT is not available in this environment")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

    mu = ROOT.RooRealVar("mu", "mu", float(mu_value), 0.0, 1.0e6)
    pdfs = ROOT.RooArgList()
    observables = ROOT.RooArgSet()
    keepalive: list[Any] = [mu, pdfs, observables]
    poissons: list[Any] = []

    for index in range(n_bins):
        obs_value = float(parameters[f"obs_{index}"])
        signal = float(parameters[f"signal_{index}"])
        background = float(parameters[f"background_{index}"])

        max_range = max(obs_value + 10.0 * math.sqrt(max(obs_value, 1.0)) + 10.0, 100.0)
        obs = ROOT.RooRealVar(f"obs_{index}", f"obs_{index}", obs_value, 0.0, max_range)
        obs.setConstant(True)

        arglist = ROOT.RooArgList(mu)
        expected = ROOT.RooFormulaVar(
            f"expected_{index}",
            f"@0 * {signal:.17g} + {background:.17g}",
            arglist,
        )
        poisson = ROOT.RooPoisson(f"poisson_{index}", f"poisson_{index}", obs, expected)

        pdfs.add(poisson)
        observables.add(obs)
        keepalive.extend([obs, arglist, expected, poisson])
        poissons.append(poisson)

    likelihood = ROOT.RooProdPdf("likelihood", "likelihood", pdfs)
    keepalive.append(likelihood)

    return {"mu": mu, "poissons": poissons, "keepalive": keepalive}


def roofit_nll(model: dict[str, Any], mu_value: float) -> float:
    model["mu"].setVal(float(mu_value))
    total = 0.0
    for index, poisson in enumerate(model["poissons"]):
        value = float(poisson.getVal())
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError(
                f"RooFit returned invalid PDF value for bin {index}: {value}"
            )
        total -= math.log(value)
    return float(total)


def validate_numeric_value(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def summarize_timings_seconds(timings: list[float]) -> dict[str, float]:
    if not timings:
        raise ValueError("Cannot summarize empty timing list")
    invalid = [
        timing for timing in timings if not math.isfinite(timing) or timing <= 0.0
    ]
    if invalid:
        raise ValueError(
            f"Timing samples must be positive finite values: {invalid[:5]}"
        )
    return {
        "mean_seconds": float(mean(timings)),
        "std_seconds": float(stdev(timings)) if len(timings) > 1 else 0.0,
        "min_seconds": float(min(timings)),
        "max_seconds": float(max(timings)),
    }


def measure_framework(
    *,
    name: str,
    build_func: Callable[[], Any],
    eval_func: Callable[[Any, float], float],
    mu: float,
    delta_reference_mu: float,
    n_runs: int,
) -> dict[str, Any]:
    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    build_start = time.perf_counter()
    model = build_func()
    model_build_time_seconds = time.perf_counter() - build_start

    first_start = time.perf_counter()
    raw_nll = float(eval_func(model, mu))
    first_evaluation_time_seconds = time.perf_counter() - first_start
    validate_numeric_value(raw_nll, f"{name} raw NLL")

    cache_bust_epsilon = 1e-9
    warm_timings: list[float] = []
    warm_nll = raw_nll
    for run_index in range(n_runs):
        eval_mu = mu if run_index % 2 == 0 else mu + cache_bust_epsilon
        start = time.perf_counter()
        warm_nll = float(eval_func(model, eval_mu))
        warm_timings.append(time.perf_counter() - start)
        validate_numeric_value(warm_nll, f"{name} warm NLL")

    reference_nll = float(eval_func(model, delta_reference_mu))
    validate_numeric_value(reference_nll, f"{name} reference NLL")
    delta_nll = raw_nll - reference_nll

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    return {
        "framework": name,
        "framework_label": _framework_label(name),
        "status": "success",
        "validation_status": "pending",
        "raw_nll": raw_nll,
        "warm_nll": warm_nll,
        "reference_nll": reference_nll,
        "delta_nll": delta_nll,
        "model_build_time_seconds": float(model_build_time_seconds),
        "first_evaluation_time_seconds": float(first_evaluation_time_seconds),
        "warm_evaluation": summarize_timings_seconds(warm_timings),
        "current_rss_before_mb": float(current_rss_before_mb),
        "current_rss_after_mb": float(current_rss_after_mb),
        "current_rss_delta_mb": float(
            max(0.0, current_rss_after_mb - current_rss_before_mb)
        ),
        "peak_rss_before_mb": float(peak_rss_before_mb),
        "peak_rss_after_mb": float(peak_rss_after_mb),
        "peak_rss_delta_mb": float(max(0.0, peak_rss_after_mb - peak_rss_before_mb)),
    }


def failed_framework_result(name: str, error: BaseException) -> dict[str, Any]:
    return {
        "framework": name,
        "framework_label": _framework_label(name),
        "status": "failed",
        "validation_status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def add_validation(
    results: list[dict[str, Any]], raw_tolerance: float, delta_tolerance: float
) -> None:
    successful = _successful_results(results)
    if not any(result["framework"] == "manual" for result in successful):
        raise ValidationFailure("Manual reference result is required for validation")

    reference = next(result for result in successful if result["framework"] == "manual")

    for result in results:
        if result.get("status") != "success":
            continue

        raw_abs_diff = abs(float(result["raw_nll"]) - float(reference["raw_nll"]))
        delta_abs_diff = abs(float(result["delta_nll"]) - float(reference["delta_nll"]))

        raw_success = raw_abs_diff <= raw_tolerance
        delta_success = delta_abs_diff <= delta_tolerance

        result["raw_nll_abs_diff"] = float(raw_abs_diff)
        result["delta_nll_abs_diff"] = float(delta_abs_diff)
        result["raw_nll_success"] = bool(raw_success)
        result["delta_nll_success"] = bool(delta_success)
        result["validation_status"] = (
            "success" if raw_success and delta_success else "failed"
        )

        if result["validation_status"] == "failed":
            result["error_type"] = "ValidationFailure"
            result["error_message"] = (
                "NLL agreement failed "
                f"(raw diff={raw_abs_diff:.3e}, delta diff={delta_abs_diff:.3e})"
            )


def summarize_status(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "success"]
    validated = [
        result for result in successful if result.get("validation_status") == "success"
    ]
    failed = [
        result
        for result in results
        if result.get("status") != "success"
        or result.get("validation_status") != "success"
    ]
    return {
        "status": "success" if not failed and len(results) > 0 else "failed",
        "n_results": len(results),
        "n_successful": len(successful),
        "n_validated": len(validated),
        "n_failed": len(failed),
        "failed_results": [
            {
                "framework": result.get("framework"),
                "status": result.get("status"),
                "validation_status": result.get("validation_status"),
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
            }
            for result in failed
        ],
    }


def print_result(result: dict[str, Any]) -> None:
    print()
    print("-" * 72)
    print(result.get("framework_label", result.get("framework")))
    print("-" * 72)
    print(f"status:                  {result.get('status')}")
    print(f"validation:              {result.get('validation_status', 'unknown')}")

    if result.get("status") != "success":
        print(
            f"error:                   {result.get('error_type')}: {result.get('error_message')}"
        )
        return

    print(f"raw NLL:                 {result['raw_nll']:.15f}")
    print(f"reference NLL:           {result['reference_nll']:.15f}")
    print(f"delta NLL:               {result['delta_nll']:.15f}")
    print(
        f"raw abs diff:            {result.get('raw_nll_abs_diff', float('nan')):.15e}"
    )
    print(
        f"delta abs diff:          {result.get('delta_nll_abs_diff', float('nan')):.15e}"
    )
    print(
        f"model build:             {result['model_build_time_seconds'] * 1000.0:.3f} ms"
    )
    print(
        f"first evaluation:        {result['first_evaluation_time_seconds'] * 1e6:.3f} us"
    )
    print(
        "warm evaluation:         "
        f"{result['warm_evaluation']['mean_seconds'] * 1e6:.3f} us "
        f"± {result['warm_evaluation']['std_seconds'] * 1e6:.3f} us"
    )
    print(f"current RSS delta:       {result['current_rss_delta_mb']:.3f} MB")
    print(f"peak RSS delta:          {result['peak_rss_delta_mb']:.3f} MB")


def print_final_summary(output_data: dict[str, Any]) -> None:
    summary = output_data["summary"]
    print()
    print("=" * 80)
    print(BENCHMARK_TITLE)
    print("=" * 80)
    print(f"Workspace:   {Path(output_data['workspace']).name}")
    print(f"Bins:        {output_data['n_bins']}")
    print(f"mu:          {output_data['mu']}")
    print(f"Reference:   {output_data['delta_reference_mu']}")
    print(f"Runs:        {output_data['n_runs']}")
    print(f"Status:      {summary['status']}")
    print(f"Validated:   {summary['n_validated']} / {summary['n_results']}")
    if summary["failed_results"]:
        print("Failed:")
        for failure in summary["failed_results"]:
            print(
                "  - "
                f"{failure.get('framework')}: "
                f"{failure.get('error_type') or failure.get('status')} "
                f"{failure.get('error_message') or ''}"
            )


def _plot_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _successful_results(results)


def make_timing_profile_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    selected = _plot_results(results)
    if not selected:
        raise ValueError("No successful results available for timing profile plot")

    labels = [_framework_label(result["framework"]) for result in selected]
    x = np.arange(len(selected))
    width = 0.26

    build_ms = [result["model_build_time_seconds"] * 1000.0 for result in selected]
    first_us = [result["first_evaluation_time_seconds"] * 1e6 for result in selected]
    warm_us = [result["warm_evaluation"]["mean_seconds"] * 1e6 for result in selected]
    colors = [_style_for(result["framework"])["color"] for result in selected]

    fig, ax = plt.subplots(figsize=(11.8, 6.2))
    build_bars = ax.bar(
        x - width,
        [_safe_log_value(value) for value in build_ms],
        width=width,
        color=colors,
        edgecolor="black",
        hatch="//",
        label="Model build [ms]",
    )
    first_bars = ax.bar(
        x,
        [_safe_log_value(value) for value in first_us],
        width=width,
        color=colors,
        edgecolor="black",
        hatch="..",
        alpha=0.85,
        label="First eval [µs]",
    )
    warm_bars = ax.bar(
        x + width,
        [_safe_log_value(value) for value in warm_us],
        width=width,
        color=colors,
        edgecolor="black",
        hatch="\\\\",
        alpha=0.72,
        label="Warm eval [µs]",
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Timing value (mixed units, log scale)")
    ax.set_title("Binned likelihood timing profile", loc="left", weight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(True, which="both", alpha=0.28)

    for bars, values in (
        (build_bars, build_ms),
        (first_bars, first_us),
        (warm_bars, warm_us),
    ):
        for bar, value in zip(bars, values, strict=True):
            ax.annotate(
                _format_compact(value),
                xy=(bar.get_x() + bar.get_width() / 2, _safe_log_value(value)),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
                clip_on=False,
            )

    _save_figure(fig, output_path)


def make_warm_evaluation_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    selected = _plot_results(results)
    if not selected:
        raise ValueError("No successful results available for warm evaluation plot")

    labels = [_framework_label(result["framework"]) for result in selected]
    values = [result["warm_evaluation"]["mean_seconds"] * 1e6 for result in selected]
    errors = [result["warm_evaluation"]["std_seconds"] * 1e6 for result in selected]
    colors = [_style_for(result["framework"])["color"] for result in selected]
    x = np.arange(len(selected))

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    bars = ax.bar(
        x,
        [_safe_log_value(value) for value in values],
        yerr=errors,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        capsize=4,
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Warm NLL evaluation [µs]")
    ax.set_title("Steady-state binned likelihood evaluation", loc="left", weight="bold")
    ax.grid(True, which="both", alpha=0.28)

    for bar, value in zip(bars, values, strict=True):
        ax.annotate(
            _format_compact(value),
            xy=(bar.get_x() + bar.get_width() / 2, _safe_log_value(value)),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )

    _save_figure(fig, output_path)


def make_memory_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    selected = _plot_results(results)
    if not selected:
        raise ValueError("No successful results available for memory plot")

    labels = [_framework_label(result["framework"]) for result in selected]
    current_values = [float(result["current_rss_delta_mb"]) for result in selected]
    peak_values = [float(result["peak_rss_delta_mb"]) for result in selected]
    current = [_safe_log_value(value, 1e-1) for value in current_values]
    peak = [_safe_log_value(value, 1e-1) for value in peak_values]
    colors = [_style_for(result["framework"])["color"] for result in selected]
    x = np.arange(len(selected))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.bar(
        x - width / 2,
        current,
        width=width,
        color=colors,
        edgecolor="black",
        label="Current RSS Δ",
    )
    ax.bar(
        x + width / 2,
        peak,
        width=width,
        color="#9ecae1",
        edgecolor="black",
        hatch="//",
        label="Peak RSS Δ",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Memory delta [MB]")
    ax.set_title("Binned likelihood memory footprint", loc="left", weight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(True, which="both", alpha=0.28)

    ymax = max(current + peak) * 1.45
    ymin = min(current + peak) / 1.8
    ax.set_ylim(max(1e-2, ymin), ymax)

    _save_figure(fig, output_path)


def make_nll_values_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    selected = _plot_results(results)
    if not selected:
        raise ValueError("No successful results available for NLL values plot")

    labels = [_framework_label(result["framework"]) for result in selected]
    raw = [float(result["raw_nll"]) for result in selected]
    colors = [_style_for(result["framework"])["color"] for result in selected]
    x = np.arange(len(selected))

    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    bars = ax.bar(x, raw, width=0.55, color=colors, edgecolor="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Raw NLL")
    ax.set_title("Raw binned likelihood NLL values", loc="left", weight="bold")
    ax.grid(True, axis="y", alpha=0.28)

    reference = next(
        (result for result in selected if result["framework"] == "manual"), None
    )
    if reference is not None:
        ax.axhline(
            float(reference["raw_nll"]),
            color="black",
            linestyle="--",
            linewidth=1.3,
            alpha=0.8,
            label="manual reference",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    spread = max(raw) - min(raw)
    if spread <= 1e-9:
        center = float(mean(raw))
        pad = max(abs(center) * 0.02, 1.0)
        ax.set_ylim(center - pad, center + pad)
    else:
        ax.set_ylim(min(raw) - 0.08 * spread, max(raw) + 0.12 * spread)

    for bar, value in zip(bars, raw, strict=True):
        ax.annotate(
            f"{value:.6g}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

    _save_figure(fig, output_path)


def make_agreement_plot(
    results: list[dict[str, Any]],
    output_path: Path,
    raw_tolerance: float,
    delta_tolerance: float,
) -> None:
    _apply_cern_style()
    selected = [
        result
        for result in _plot_results(results)
        if result.get("framework") != "manual"
    ]
    if not selected:
        raise ValueError(
            "No non-reference successful results available for agreement plot"
        )

    labels = [_framework_label(result["framework"]) for result in selected]
    raw_values = [float(result.get("raw_nll_abs_diff", 0.0)) for result in selected]
    delta_values = [float(result.get("delta_nll_abs_diff", 0.0)) for result in selected]
    raw = [_safe_log_value(value, 1e-18) for value in raw_values]
    delta = [_safe_log_value(value, 1e-18) for value in delta_values]
    colors = [_style_for(result["framework"])["color"] for result in selected]
    x = np.arange(len(selected))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    raw_bars = ax.bar(
        x - width / 2,
        raw,
        width=width,
        color=colors,
        edgecolor="black",
        label="Raw NLL diff",
    )
    delta_bars = ax.bar(
        x + width / 2,
        delta,
        width=width,
        color="#bdbdbd",
        edgecolor="black",
        hatch="//",
        label="ΔNLL diff",
    )
    ax.axhline(
        raw_tolerance,
        linestyle="--",
        color="black",
        linewidth=1.7,
        label=f"raw tol = {raw_tolerance:g}",
    )
    ax.axhline(
        delta_tolerance,
        linestyle=":",
        color="black",
        linewidth=1.9,
        label=f"Δ tol = {delta_tolerance:g}",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absolute difference from manual")
    ax.set_title("Numerical agreement with manual reference", loc="left", weight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.grid(True, which="both", alpha=0.28)

    smallest_positive = min(
        [
            value
            for value in raw + delta + [raw_tolerance, delta_tolerance]
            if value > 0.0
        ]
    )
    largest_value = max(raw + delta + [raw_tolerance, delta_tolerance])
    ax.set_ylim(smallest_positive / 5.0, largest_value * 4.0)

    for bars, values in ((raw_bars, raw_values), (delta_bars, delta_values)):
        for bar, value in zip(bars, values, strict=True):
            if value <= 0.0:
                continue
            ax.annotate(
                _format_scientific(value),
                xy=(bar.get_x() + bar.get_width() / 2, _safe_log_value(value, 1e-18)),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
            )

    _save_figure(fig, output_path)


def make_summary_table(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    selected = _plot_results(results)
    if not selected:
        raise ValueError("No successful results available for summary table")

    rows = []
    for result in selected:
        rows.append(
            [
                _framework_label(result["framework"]),
                f"{result['model_build_time_seconds'] * 1000.0:.3g}",
                f"{result['first_evaluation_time_seconds'] * 1e6:.3g}",
                f"{result['warm_evaluation']['mean_seconds'] * 1e6:.3g}",
                f"{result['current_rss_delta_mb']:.2f}",
                _format_scientific(result.get("raw_nll_abs_diff", 0.0)),
                _format_scientific(result.get("delta_nll_abs_diff", 0.0)),
                result.get("validation_status", "unknown"),
            ]
        )

    headers = [
        "Framework",
        "Build [ms]",
        "First [µs]",
        "Warm [µs]",
        "RSS Δ [MB]",
        "Raw diff",
        "ΔNLL diff",
        "Validation",
    ]

    fig_height = max(3.8, 0.45 * len(rows) + 2.2)
    fig, ax = plt.subplots(figsize=(14.0, fig_height))
    ax.axis("off")
    ax.set_title(
        "Cross-framework binned likelihood summary",
        loc="left",
        weight="bold",
        fontsize=24,
        pad=20,
    )

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.35)

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#bdbdbd")
        if row == 0:
            cell.set_facecolor("#262626")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")

    _save_figure(fig, output_path)


def make_plots(
    results: list[dict[str, Any]],
    plot_dir: Path,
    raw_tolerance: float,
    delta_tolerance: float,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    if len(_successful_results(results)) < 2:
        print("Skipping plots: at least two successful results are needed.")
        return

    make_timing_profile_plot(
        results, plot_dir / "cross_binned_likelihood_timing_profile.png"
    )
    make_warm_evaluation_plot(
        results, plot_dir / "cross_binned_likelihood_warm_evaluation.png"
    )
    make_memory_plot(results, plot_dir / "cross_binned_likelihood_memory.png")
    make_nll_values_plot(results, plot_dir / "cross_binned_likelihood_nll_values.png")
    make_agreement_plot(
        results,
        plot_dir / "cross_binned_likelihood_numerical_agreement.png",
        raw_tolerance=raw_tolerance,
        delta_tolerance=delta_tolerance,
    )
    make_summary_table(results, plot_dir / "cross_binned_likelihood_summary_table.png")


def build_framework_jobs(
    *,
    frameworks: list[str],
    parameters: dict[str, float],
    workspace_path: Path,
    n_bins: int,
    mu: float,
) -> dict[str, tuple[Callable[[], Any], Callable[[Any, float], float]]]:
    jobs: dict[str, tuple[Callable[[], Any], Callable[[Any, float], float]]] = {
        "manual": (
            lambda: build_manual_model(parameters, n_bins),
            lambda model, mu_value: manual_nll(model, mu_value),
        ),
        "pyhs3": (
            lambda: build_pyhs3_model(workspace_path),
            lambda model, mu_value: pyhs3_nll(model, n_bins, mu_value),
        ),
        "pyhf": (
            lambda: build_pyhf_model(parameters, n_bins),
            lambda model, mu_value: pyhf_nll(model, mu_value),
        ),
        "roofit": (
            lambda: build_roofit_model(parameters, n_bins, mu),
            lambda model, mu_value: roofit_nll(model, mu_value),
        ),
    }
    return {framework: jobs[framework] for framework in frameworks}


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    validate_config(config)

    workspace = load_workspace(config.workspace_path)
    parameters = extract_parameters(workspace)
    inferred_n_bins = infer_n_bins(parameters)
    n_bins = inferred_n_bins if config.n_bins is None else config.n_bins
    validate_n_bins_against_parameters(parameters, n_bins)

    jobs = build_framework_jobs(
        frameworks=config.frameworks,
        parameters=parameters,
        workspace_path=config.workspace_path,
        n_bins=n_bins,
        mu=config.mu,
    )

    results: list[dict[str, Any]] = []
    for framework, (build_func, eval_func) in jobs.items():
        try:
            result = measure_framework(
                name=framework,
                build_func=build_func,
                eval_func=eval_func,
                mu=config.mu,
                delta_reference_mu=config.delta_reference_mu,
                n_runs=config.n_runs,
            )
        except Exception as error:  # noqa: BLE001
            result = failed_framework_result(framework, error)
        results.append(result)

        if config.fail_fast and result.get("status") != "success":
            break

    add_validation(
        results,
        raw_tolerance=config.raw_tolerance,
        delta_tolerance=config.delta_tolerance,
    )
    summary = summarize_status(results)

    output_data: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "workspace": str(config.workspace_path),
        "n_bins": n_bins,
        "inferred_n_bins": inferred_n_bins,
        "mu": config.mu,
        "delta_reference_mu": config.delta_reference_mu,
        "n_runs": config.n_runs,
        "frameworks": config.frameworks,
        "raw_tolerance": config.raw_tolerance,
        "delta_tolerance": config.delta_tolerance,
        "summary": summary,
        "status": summary["status"],
        "results": results,
    }

    return output_data


def run(
    *,
    workspace_path: Path,
    frameworks: list[str],
    n_bins: int | None,
    mu: float,
    delta_reference_mu: float,
    n_runs: int,
    output_dir: Path,
    output_name: str,
    plot: bool,
    plot_dir: Path,
    raw_tolerance: float,
    delta_tolerance: float,
    fail_fast: bool = False,
) -> dict[str, Any]:
    config = BenchmarkConfig(
        workspace_path=workspace_path,
        frameworks=frameworks,
        n_bins=n_bins,
        mu=mu,
        delta_reference_mu=delta_reference_mu,
        n_runs=n_runs,
        raw_tolerance=raw_tolerance,
        delta_tolerance=delta_tolerance,
        output_dir=output_dir,
        output_name=output_name,
        plot=plot,
        plot_dir=plot_dir,
        fail_fast=fail_fast,
    )

    output_data = run_benchmark(config)

    print_final_summary(output_data)
    for result in output_data["results"]:
        print_result(result)

    output_path = output_dir / output_name
    save_json(output_data, output_path)
    print(f"Saved result to {output_path}")

    if plot:
        make_plots(
            output_data["results"],
            plot_dir,
            raw_tolerance=raw_tolerance,
            delta_tolerance=delta_tolerance,
        )
        print(f"Saved plots to {plot_dir}")

    return output_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=BENCHMARK_TITLE)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=DEFAULT_FRAMEWORKS,
        choices=SUPPORTED_FRAMEWORKS,
    )
    parser.add_argument("--n-bins", type=int, default=None)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--delta-reference-mu", type=float, default=0.0)
    parser.add_argument("--n-runs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--raw-tolerance", type=float, default=1e-10)
    parser.add_argument("--delta-tolerance", type=float, default=1e-10)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(
            workspace_path=args.workspace,
            frameworks=list(args.frameworks),
            n_bins=args.n_bins,
            mu=args.mu,
            delta_reference_mu=args.delta_reference_mu,
            n_runs=args.n_runs,
            output_dir=args.output_dir,
            output_name=args.output_name,
            plot=args.plot,
            plot_dir=args.plot_dir,
            raw_tolerance=args.raw_tolerance,
            delta_tolerance=args.delta_tolerance,
            fail_fast=args.fail_fast,
        )
    except Exception as error:
        raise RuntimeError(
            "Cross-framework binned likelihood benchmark failed"
        ) from error


if __name__ == "__main__":
    main(sys.argv[1:])
