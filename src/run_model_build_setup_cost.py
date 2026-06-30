from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyhf
from pyhs3.workspace import Workspace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import PLOTS_DIR, RESULTS_DIR
    from src.utils import get_current_rss_mb, save_json
else:
    from .config import PLOTS_DIR, RESULTS_DIR
    from .utils import get_current_rss_mb, save_json

try:
    import ROOT
except ImportError:
    ROOT = None


BENCHMARK_NAME = "model_build_setup_cost"
DEFAULT_WORKSPACE = Path("inputs/binned_likelihood_models/pyhs3_300bins.json")
DEFAULT_FRAMEWORKS = ["pyhs3", "pyhf", "roofit"]
DEFAULT_MU = 1.0
DEFAULT_WARMUP_ITERATIONS = 3
DEFAULT_AGREEMENT_TOLERANCE = 1e-9
FRAMEWORK_ORDER = {"pyhs3": 0, "pyhf": 1, "roofit": 2}
FRAMEWORK_STYLE = {
    "pyhs3": {"label": "PyHS3", "color": "#1764AB", "marker": "s"},
    "pyhf": {"label": "pyhf", "color": "#E67800", "marker": "^"},
    "roofit": {"label": "RooFit", "color": "#009E73", "marker": "D"},
}


def validate_existing_file(path: Path, name: str) -> Path:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    return path


def validate_positive_int(value: int, name: str, *, minimum: int = 1) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")


def validate_finite_float(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite; got {value}")


def validate_frameworks(frameworks: list[str]) -> list[str]:
    if not frameworks:
        raise ValueError("At least one framework must be requested")

    normalized: list[str] = []
    allowed = set(FRAMEWORK_ORDER)
    for framework in frameworks:
        framework = framework.lower()
        if framework not in allowed:
            raise ValueError(
                f"Unknown framework '{framework}'. Allowed values: {', '.join(sorted(allowed))}"
            )
        if framework not in normalized:
            normalized.append(framework)

    if "pyhf" not in normalized:
        raise ValueError(
            "pyhf must be included because it is used as the numerical reference"
        )

    return sorted(normalized, key=lambda item: FRAMEWORK_ORDER[item])


def validate_benchmark_config(
    *,
    workspace_path: Path,
    n_bins: int | None,
    mu: float,
    frameworks: list[str],
    warmup_iterations: int,
) -> list[str]:
    validate_existing_file(workspace_path, "Workspace file")
    if n_bins is not None:
        validate_positive_int(n_bins, "n_bins", minimum=1)
    validate_finite_float(mu, "mu")
    validate_positive_int(warmup_iterations, "warmup_iterations", minimum=0)
    return validate_frameworks(frameworks)


def validate_non_negative_seconds(value: float, name: str) -> None:
    validate_finite_float(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value}")


def validate_positive_seconds(value: float, name: str) -> None:
    validate_finite_float(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive; got {value}")


def validate_measurement_result(result: dict[str, Any]) -> None:
    if result.get("status") != "success":
        return

    for field in [
        "input_load_time_seconds",
        "model_construction_time_seconds",
        "warmup_time_seconds",
        "rss_delta_mb",
    ]:
        validate_non_negative_seconds(float(result[field]), field)

    for field in [
        "cold_first_evaluation_time_seconds",
        "warm_first_evaluation_time_seconds",
    ]:
        validate_positive_seconds(float(result[field]), field)

    validate_finite_float(float(result["value"]), "value")
    if result["rss_delta_mb"] < 0.0:
        raise ValueError("rss_delta_mb must be non-negative")


def extract_parameters(workspace: Workspace) -> dict[str, float]:
    try:
        parameter_points = workspace.parameter_points.root
    except AttributeError as exc:
        raise ValueError(
            "Workspace does not contain a valid parameter_points section"
        ) from exc

    if not parameter_points:
        raise ValueError("Workspace does not contain any parameter points")

    parameters = parameter_points[0].parameters
    if not parameters:
        raise ValueError("Initial parameter point does not contain parameters")

    extracted: dict[str, float] = {}
    for parameter in parameters:
        try:
            value = float(np.asarray(parameter.value).squeeze())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Parameter '{parameter.name}' is not scalar-like"
            ) from exc
        validate_finite_float(value, f"parameter {parameter.name}")
        extracted[parameter.name] = value

    return extracted


def infer_n_bins(parameters: dict[str, float]) -> int:
    indices: set[int] = set()
    for name in parameters:
        if name.startswith("obs_"):
            suffix = name.removeprefix("obs_")
            if suffix.isdigit():
                indices.add(int(suffix))

    if not indices:
        raise ValueError(
            "Could not infer n_bins from workspace parameters; no obs_<index> parameters found"
        )

    expected = set(range(max(indices) + 1))
    if indices != expected:
        missing = sorted(expected - indices)
        raise ValueError(
            f"Non-contiguous obs_<index> parameters; missing indices: {missing}"
        )

    return max(indices) + 1


def get_vectors(
    parameters: dict[str, float], n_bins: int
) -> tuple[list[float], list[float], list[float]]:
    validate_positive_int(n_bins, "n_bins", minimum=1)
    signal: list[float] = []
    background: list[float] = []
    observed: list[float] = []

    for index in range(n_bins):
        required = [f"signal_{index}", f"background_{index}", f"obs_{index}"]
        missing = [name for name in required if name not in parameters]
        if missing:
            raise KeyError(
                f"Missing required parameter(s) for bin {index}: {', '.join(missing)}"
            )

        sig = float(parameters[f"signal_{index}"])
        bkg = float(parameters[f"background_{index}"])
        obs = float(parameters[f"obs_{index}"])
        for name, value in [
            (f"signal_{index}", sig),
            (f"background_{index}", bkg),
            (f"obs_{index}", obs),
        ]:
            validate_finite_float(value, name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative; got {value}")
        signal.append(sig)
        background.append(bkg)
        observed.append(obs)

    return signal, background, observed


def poisson_nll_from_vectors(
    signal: list[float],
    background: list[float],
    observed: list[float],
    mu_value: float,
) -> float:
    validate_finite_float(mu_value, "mu")
    total = 0.0
    for index, (sig, bkg, obs) in enumerate(
        zip(signal, background, observed, strict=True)
    ):
        expected = mu_value * sig + bkg
        if expected <= 0.0:
            raise ValueError(
                f"Expected yield must be positive in bin {index}; got {expected}"
            )
        total += expected - obs * math.log(expected) + math.lgamma(obs + 1.0)
    validate_finite_float(total, "manual NLL")
    return total


def pyhs3_first_eval(model: Any, n_bins: int, mu_value: float) -> float:
    mu = np.asarray(mu_value, dtype=np.float64)
    log_likelihood = 0.0

    for index in range(n_bins):
        value = float(np.asarray(model.pdf(f"poisson_{index}", mu=mu)).squeeze())
        validate_finite_float(value, f"PyHS3 pdf value for bin {index}")
        if value <= 0.0:
            raise ValueError(
                f"PyHS3 pdf value must be positive in bin {index}; got {value}"
            )
        log_likelihood += math.log(value)

    nll = -log_likelihood
    validate_finite_float(nll, "PyHS3 NLL")
    return nll


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
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
                    },
                    {"name": "background", "data": background, "modifiers": []},
                ],
            }
        ],
        "observations": [{"name": "channel", "data": observed}],
        "measurements": [
            {"name": "measurement", "config": {"poi": "mu", "parameters": []}}
        ],
        "version": "1.0.0",
    }


def pyhf_first_eval(model: Any, data: Any, mu_value: float) -> float:
    pars = list(model.config.suggested_init())
    try:
        mu_index = model.config.par_order.index("mu")
    except ValueError as exc:
        raise ValueError("pyhf model does not expose a 'mu' parameter") from exc
    pars[mu_index] = mu_value

    logpdf = model.logpdf(pars, data)
    nll = -float(np.asarray(logpdf).squeeze())
    validate_finite_float(nll, "pyhf NLL")
    return nll


def roofit_first_eval(model: dict[str, Any], mu_value: float) -> float:
    model["mu"].setVal(mu_value)

    total = 0.0
    for index, poisson in enumerate(model["poissons"]):
        value = float(poisson.getVal())
        validate_finite_float(value, f"RooFit pdf value for bin {index}")
        if value <= 0.0:
            raise ValueError(
                f"RooFit pdf value must be positive in bin {index}; got {value}"
            )
        total -= math.log(value)

    validate_finite_float(total, "RooFit NLL")
    return total


def timed_call(function: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = function()
    duration = time.perf_counter() - start
    validate_non_negative_seconds(duration, "duration")
    return value, duration


def run_warmups(
    evaluator: Callable[[], float], warmup_iterations: int
) -> tuple[float, float]:
    validate_positive_int(warmup_iterations, "warmup_iterations", minimum=0)
    start = time.perf_counter()
    last_value = math.nan
    for _ in range(warmup_iterations):
        last_value = float(evaluator())
        validate_finite_float(last_value, "warmup NLL")
    warmup_time = time.perf_counter() - start
    validate_non_negative_seconds(warmup_time, "warmup_time")
    return last_value, warmup_time


def successful_result(
    *,
    framework: str,
    value: float,
    input_load_time_seconds: float,
    model_construction_time_seconds: float,
    cold_first_evaluation_time_seconds: float,
    warmup_time_seconds: float,
    warm_first_evaluation_time_seconds: float,
    rss_before_mb: float,
    rss_after_mb: float,
    stage_notes: str,
    warmup_iterations: int,
) -> dict[str, Any]:
    result = {
        "framework": framework,
        "plot_label": FRAMEWORK_STYLE[framework]["label"],
        "status": "success",
        "value": value,
        "input_load_time_seconds": input_load_time_seconds,
        "model_construction_time_seconds": model_construction_time_seconds,
        "cold_first_evaluation_time_seconds": cold_first_evaluation_time_seconds,
        "warmup_iterations": warmup_iterations,
        "warmup_time_seconds": warmup_time_seconds,
        "warm_first_evaluation_time_seconds": warm_first_evaluation_time_seconds,
        "rss_before_mb": rss_before_mb,
        "rss_after_mb": rss_after_mb,
        "rss_delta_mb": max(0.0, rss_after_mb - rss_before_mb),
        "stage_notes": stage_notes,
    }
    validate_measurement_result(result)
    return result


def failed_framework_result(framework: str, exc: BaseException) -> dict[str, Any]:
    return {
        "framework": framework,
        "plot_label": FRAMEWORK_STYLE.get(framework, {}).get("label", framework),
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def measure_manual(
    parameters: dict[str, float],
    n_bins: int,
    mu: float,
    warmup_iterations: int,
) -> dict[str, Any]:
    rss_before = get_current_rss_mb()

    vectors, load_time = timed_call(lambda: get_vectors(parameters, n_bins))
    signal, background, observed = vectors
    _, model_time = timed_call(lambda: None)

    def evaluator() -> float:
        return poisson_nll_from_vectors(signal, background, observed, mu)

    value, cold_time = timed_call(evaluator)
    _, warmup_time = run_warmups(evaluator, warmup_iterations)
    warm_value, warm_time = timed_call(evaluator)
    validate_finite_float(warm_value, "manual warm NLL")

    rss_after = get_current_rss_mb()
    return successful_result(
        framework="manual",
        value=value,
        input_load_time_seconds=load_time,
        model_construction_time_seconds=model_time,
        cold_first_evaluation_time_seconds=cold_time,
        warmup_time_seconds=warmup_time,
        warm_first_evaluation_time_seconds=warm_time,
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        stage_notes="Manual reference extracts HS3 vectors and evaluates the binned Poisson NLL directly.",
        warmup_iterations=warmup_iterations,
    )


def measure_pyhs3(
    workspace_path: Path,
    n_bins: int,
    mu: float,
    warmup_iterations: int,
) -> dict[str, Any]:
    rss_before = get_current_rss_mb()

    workspace, load_time = timed_call(lambda: Workspace.load(workspace_path))
    model, model_time = timed_call(
        lambda: workspace.model("analysis", progress=False, mode="FAST_RUN")
    )

    def evaluator() -> float:
        return pyhs3_first_eval(model, n_bins, mu)

    value, cold_time = timed_call(evaluator)
    _, warmup_time = run_warmups(evaluator, warmup_iterations)
    warm_value, warm_time = timed_call(evaluator)
    validate_finite_float(warm_value, "PyHS3 warm NLL")

    rss_after = get_current_rss_mb()
    return successful_result(
        framework="pyhs3",
        value=value,
        input_load_time_seconds=load_time,
        model_construction_time_seconds=model_time,
        cold_first_evaluation_time_seconds=cold_time,
        warmup_time_seconds=warmup_time,
        warm_first_evaluation_time_seconds=warm_time,
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        stage_notes=(
            "Workspace.load, Workspace.model, cold first pdf evaluation, then warm evaluation. "
            "Any PyHS3 lazy initialization is visible in the cold first-evaluation stage."
        ),
        warmup_iterations=warmup_iterations,
    )


def measure_pyhf(
    parameters: dict[str, float],
    n_bins: int,
    mu: float,
    warmup_iterations: int,
) -> dict[str, Any]:
    rss_before = get_current_rss_mb()

    spec, load_time = timed_call(lambda: build_pyhf_spec(parameters, n_bins))

    def build_model() -> tuple[Any, Any]:
        workspace = pyhf.Workspace(spec)
        model = workspace.model()
        data = workspace.data(model)
        return model, data

    (model, data), model_time = timed_call(build_model)

    def evaluator() -> float:
        return pyhf_first_eval(model, data, mu)

    value, cold_time = timed_call(evaluator)
    _, warmup_time = run_warmups(evaluator, warmup_iterations)
    warm_value, warm_time = timed_call(evaluator)
    validate_finite_float(warm_value, "pyhf warm NLL")

    rss_after = get_current_rss_mb()
    return successful_result(
        framework="pyhf",
        value=value,
        input_load_time_seconds=load_time,
        model_construction_time_seconds=model_time,
        cold_first_evaluation_time_seconds=cold_time,
        warmup_time_seconds=warmup_time,
        warm_first_evaluation_time_seconds=warm_time,
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        stage_notes="Input load builds a pyhf JSON spec from HS3 parameters; model construction builds pyhf Workspace, Model, and data.",
        warmup_iterations=warmup_iterations,
    )


def measure_roofit(
    parameters: dict[str, float],
    n_bins: int,
    mu_value: float,
    warmup_iterations: int,
) -> dict[str, Any]:
    if ROOT is None:
        raise RuntimeError("ROOT is not available in this environment")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    rss_before = get_current_rss_mb()

    vectors, load_time = timed_call(lambda: get_vectors(parameters, n_bins))
    signal, background, observed = vectors

    def build_model() -> dict[str, Any]:
        mu = ROOT.RooRealVar("mu", "mu", mu_value)
        mu.setConstant(False)

        pdfs = ROOT.RooArgList()
        keepalive = [mu]
        poissons = []

        for index, (sig, bkg, obs_value) in enumerate(
            zip(signal, background, observed, strict=True)
        ):
            upper = max(100.0, obs_value * 10.0 + 10.0)
            obs = ROOT.RooRealVar(f"obs_{index}", f"obs_{index}", obs_value, 0.0, upper)
            obs.setConstant(True)

            expected = ROOT.RooFormulaVar(
                f"expected_{index}",
                f"@0 * {sig:.17g} + {bkg:.17g}",
                ROOT.RooArgList(mu),
            )

            poisson = ROOT.RooPoisson(
                f"poisson_{index}", f"poisson_{index}", obs, expected
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

    model, model_time = timed_call(build_model)

    def evaluator() -> float:
        return roofit_first_eval(model, mu_value)

    value, cold_time = timed_call(evaluator)
    _, warmup_time = run_warmups(evaluator, warmup_iterations)
    warm_value, warm_time = timed_call(evaluator)
    validate_finite_float(warm_value, "RooFit warm NLL")

    rss_after = get_current_rss_mb()
    return successful_result(
        framework="roofit",
        value=value,
        input_load_time_seconds=load_time,
        model_construction_time_seconds=model_time,
        cold_first_evaluation_time_seconds=cold_time,
        warmup_time_seconds=warmup_time,
        warm_first_evaluation_time_seconds=warm_time,
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        stage_notes="Input load extracts vectors; model construction builds RooRealVar, RooFormulaVar, RooPoisson, and RooProdPdf objects.",
        warmup_iterations=warmup_iterations,
    )


def measure_framework(
    *,
    framework: str,
    workspace_path: Path,
    parameters: dict[str, float],
    n_bins: int,
    mu: float,
    warmup_iterations: int,
) -> dict[str, Any]:
    try:
        if framework == "pyhs3":
            return measure_pyhs3(workspace_path, n_bins, mu, warmup_iterations)
        if framework == "pyhf":
            return measure_pyhf(parameters, n_bins, mu, warmup_iterations)
        if framework == "roofit":
            return measure_roofit(parameters, n_bins, mu, warmup_iterations)
        raise ValueError(f"Unsupported framework: {framework}")
    except Exception as exc:
        return failed_framework_result(framework, exc)


def add_validation(results: list[dict[str, Any]], tolerance: float) -> None:
    successful = [result for result in results if result.get("status") == "success"]
    reference = next(
        (result for result in successful if result["framework"] == "pyhf"), None
    )
    if reference is None:
        raise ValueError(
            "Cannot validate setup-cost benchmark without a successful pyhf result"
        )

    reference_value = float(reference["value"])
    for result in successful:
        diff = abs(float(result["value"]) - reference_value)
        result["value_abs_diff_from_pyhf"] = diff
        result["validation_status"] = "success" if diff <= tolerance else "failed"
        result["validation_tolerance"] = tolerance

    for result in results:
        if result.get("status") != "success":
            result["value_abs_diff_from_pyhf"] = None
            result["validation_status"] = "failed"
            result["validation_tolerance"] = tolerance


def benchmark_status(results: list[dict[str, Any]]) -> str:
    return (
        "success"
        if results
        and all(
            result.get("status") == "success"
            and result.get("validation_status") == "success"
            for result in results
        )
        else "failed"
    )


def print_result(result: dict[str, Any]) -> None:
    print()
    print("-" * 72)
    print(
        FRAMEWORK_STYLE.get(result["framework"], {}).get("label", result["framework"])
    )
    print("-" * 72)
    print(f"status:                  {result.get('status')}")

    if result.get("status") != "success":
        print(
            f"error:                   {result.get('error_type')}: {result.get('error_message')}"
        )
        return

    print(f"validation:              {result['validation_status']}")
    print(f"NLL value:               {result['value']:.15f}")
    print(f"abs diff from pyhf:      {result['value_abs_diff_from_pyhf']:.15e}")
    print(
        f"input load:              {result['input_load_time_seconds'] * 1000.0:.3f} ms"
    )
    print(
        f"model construction:      {result['model_construction_time_seconds'] * 1000.0:.3f} ms"
    )
    print(
        f"cold first evaluation:   {result['cold_first_evaluation_time_seconds'] * 1000.0:.3f} ms"
    )
    print(
        f"warm-up evaluations:     {result['warmup_iterations']} "
        f"({result['warmup_time_seconds'] * 1000.0:.3f} ms total)"
    )
    print(
        f"warm first evaluation:   {result['warm_first_evaluation_time_seconds'] * 1e6:.3f} us"
    )
    print(f"RSS delta:               {result['rss_delta_mb']:.3f} MB")


def build_failed_output(
    *,
    workspace_path: Path,
    n_bins: int | None,
    mu: float,
    frameworks: list[str],
    warmup_iterations: int,
    agreement_tolerance: float,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "status": "failed",
        "workspace": str(workspace_path),
        "n_bins": n_bins,
        "mu": mu,
        "frameworks": frameworks,
        "warmup_iterations": warmup_iterations,
        "agreement_tolerance": agreement_tolerance,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "results": [],
    }


def _apply_cern_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "font.size": 14,
            "axes.titlesize": 24,
            "axes.labelsize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.linewidth": 1.6,
            "xtick.major.width": 1.4,
            "ytick.major.width": 1.4,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.grid": True,
            "grid.alpha": 0.28,
        }
    )


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(output_path, bbox_inches="tight")
    except OSError as exc:
        raise OSError(f"Failed to save plot to {output_path}") from exc
    finally:
        plt.close(fig)


def _successful_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if result.get("status") == "success"]


def _framework_colors(results: list[dict[str, Any]]) -> list[str]:
    return [FRAMEWORK_STYLE[result["framework"]]["color"] for result in results]


def _plot_floor(values: list[float], *, floor: float = 1e-6) -> list[float]:
    return [value if value > floor else floor for value in values]


def _format_compact_number(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.2g}"


def _add_bar_labels(
    ax: Any,
    bars: Any,
    values: list[float],
    formatter: Callable[[float], str],
    *,
    y_offset_points: int = 3,
) -> None:
    for bar, value in zip(bars, values, strict=True):
        height = max(float(bar.get_height()), 1e-30)
        ax.annotate(
            formatter(value),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, y_offset_points),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=8,
            weight="bold",
            clip_on=False,
        )


def make_setup_timing_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    if not successful:
        raise ValueError("No successful results available for setup timing plot")

    labels = [result["plot_label"] for result in successful]
    x = np.arange(len(successful))
    width = 0.22
    metrics = [
        ("input_load_time_seconds", "Input load [ms]", 1000.0, ""),
        ("model_construction_time_seconds", "Model build [ms]", 1000.0, "//"),
        ("cold_first_evaluation_time_seconds", "Cold first eval [ms]", 1000.0, "xx"),
    ]

    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    for offset, (field, label, scale, hatch) in zip(
        [-width, 0.0, width], metrics, strict=True
    ):
        values = [float(result[field]) * scale for result in successful]
        bars = ax.bar(
            x + offset,
            _plot_floor(values),
            width=width,
            label=label,
            color=[
                FRAMEWORK_STYLE[result["framework"]]["color"] for result in successful
            ],
            edgecolor="black",
            linewidth=0.7,
            hatch=hatch,
            alpha=0.9,
        )
        _add_bar_labels(ax, bars, values, _format_compact_number)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Timing [ms] (log scale)")
    ax.set_title("Model setup cost by framework", loc="left", weight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.grid(False, axis="x")
    _save_figure(fig, output_path)


def make_evaluation_latency_plot(
    results: list[dict[str, Any]], output_path: Path
) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    if not successful:
        raise ValueError("No successful results available for evaluation latency plot")

    labels = [result["plot_label"] for result in successful]
    x = np.arange(len(successful))
    width = 0.34
    cold = [
        float(result["cold_first_evaluation_time_seconds"]) * 1e6
        for result in successful
    ]
    warm = [
        float(result["warm_first_evaluation_time_seconds"]) * 1e6
        for result in successful
    ]

    fig, ax = plt.subplots(figsize=(11.0, 6.0))
    colors = _framework_colors(successful)
    cold_bars = ax.bar(
        x - width / 2,
        _plot_floor(cold),
        width,
        label="Cold first eval",
        color=colors,
        edgecolor="black",
        hatch="//",
    )
    warm_bars = ax.bar(
        x + width / 2,
        _plot_floor(warm),
        width,
        label="Warm eval",
        color=colors,
        edgecolor="black",
        hatch="..",
        alpha=0.8,
    )

    _add_bar_labels(ax, cold_bars, cold, _format_compact_number)
    _add_bar_labels(ax, warm_bars, warm, _format_compact_number)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Evaluation latency [μs] (log scale)")
    ax.set_title("Cold vs warm first NLL evaluation", loc="left", weight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.grid(False, axis="x")
    _save_figure(fig, output_path)


def make_memory_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    if not successful:
        raise ValueError("No successful results available for memory plot")

    labels = [result["plot_label"] for result in successful]
    values = [float(result["rss_delta_mb"]) for result in successful]
    x = np.arange(len(successful))

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(
        x,
        _plot_floor(values, floor=1e-3),
        color=_framework_colors(successful),
        edgecolor="black",
        linewidth=0.8,
    )
    _add_bar_labels(ax, bars, values, lambda value: f"{value:.3g} MB")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("RSS delta [MB] (log scale)")
    ax.set_title("Memory footprint during setup", loc="left", weight="bold")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.grid(False, axis="x")
    _save_figure(fig, output_path)


def make_value_agreement_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = [
        result
        for result in _successful_results(results)
        if result["framework"] != "pyhf"
    ]
    if not successful:
        raise ValueError(
            "No non-reference successful results available for agreement plot"
        )

    labels = [result["plot_label"] for result in successful]
    values = [float(result["value_abs_diff_from_pyhf"]) for result in successful]
    tolerance = float(successful[0]["validation_tolerance"])
    x = np.arange(len(successful))

    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    bars = ax.bar(
        x,
        _plot_floor(values, floor=1e-16),
        color=_framework_colors(successful),
        edgecolor="black",
        linewidth=0.8,
    )
    _add_bar_labels(ax, bars, values, lambda value: f"{value:.2e}")
    ax.axhline(
        tolerance,
        linestyle="--",
        color="black",
        linewidth=1.6,
        label=f"tolerance = {tolerance:.0e}",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$|NLL_{framework} - NLL_{pyhf}|$")
    ax.set_title("First-evaluation numerical agreement", loc="left", weight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.grid(False, axis="x")
    _save_figure(fig, output_path)


def make_summary_table_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = _successful_results(results)
    if not successful:
        raise ValueError("No successful results available for summary table")

    headers = [
        "Framework",
        "Validation",
        "Load [ms]",
        "Build [ms]",
        "Cold [ms]",
        "Warm [μs]",
        "RSS Δ [MB]",
        "NLL diff",
    ]
    rows = []
    for result in successful:
        rows.append(
            [
                result["plot_label"],
                result["validation_status"],
                f"{result['input_load_time_seconds'] * 1000.0:.3f}",
                f"{result['model_construction_time_seconds'] * 1000.0:.3f}",
                f"{result['cold_first_evaluation_time_seconds'] * 1000.0:.3f}",
                f"{result['warm_first_evaluation_time_seconds'] * 1e6:.3f}",
                f"{result['rss_delta_mb']:.3f}",
                f"{result['value_abs_diff_from_pyhf']:.2e}",
            ]
        )

    fig, ax = plt.subplots(figsize=(14.5, 4.6))
    ax.axis("off")
    ax.set_title("Model build / setup cost summary", loc="left", weight="bold", pad=20)
    ax.text(
        0.0,
        0.88,
        "Compares one-time setup costs and cold/warm first-evaluation latency across frameworks.",
        transform=ax.transAxes,
        fontsize=15,
        ha="left",
    )
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.7)

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#BDBDBD")
        if row == 0:
            cell.set_facecolor("#2B2B2B")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F7F7F7")

    _save_figure(fig, output_path)


def make_plots(results: list[dict[str, Any]], plot_dir: Path) -> None:
    if not _successful_results(results):
        raise ValueError("No successful results available for plotting")
    plot_dir.mkdir(parents=True, exist_ok=True)
    make_setup_timing_plot(results, plot_dir / "model_build_setup_timing.png")
    make_evaluation_latency_plot(
        results, plot_dir / "model_build_setup_evaluation_latency.png"
    )
    make_memory_plot(results, plot_dir / "model_build_setup_memory.png")
    make_value_agreement_plot(
        results, plot_dir / "model_build_setup_value_agreement.png"
    )
    make_summary_table_plot(results, plot_dir / "model_build_setup_summary_table.png")


def run(
    *,
    workspace_path: Path,
    n_bins: int | None = None,
    mu: float = DEFAULT_MU,
    frameworks: list[str] | None = None,
    warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
    agreement_tolerance: float = DEFAULT_AGREEMENT_TOLERANCE,
    output: Path | None = None,
    plot: bool = False,
    plot_dir: Path | None = None,
    continue_on_framework_error: bool = True,
) -> dict[str, Any]:
    frameworks = list(DEFAULT_FRAMEWORKS if frameworks is None else frameworks)
    output = (
        RESULTS_DIR / BENCHMARK_NAME / f"{BENCHMARK_NAME}_result.json"
        if output is None
        else output
    )
    plot_dir = PLOTS_DIR / BENCHMARK_NAME if plot_dir is None else plot_dir

    try:
        frameworks = validate_benchmark_config(
            workspace_path=workspace_path,
            n_bins=n_bins,
            mu=mu,
            frameworks=frameworks,
            warmup_iterations=warmup_iterations,
        )
        validate_finite_float(agreement_tolerance, "agreement_tolerance")
        if agreement_tolerance <= 0.0:
            raise ValueError("agreement_tolerance must be positive")

        workspace = Workspace.load(workspace_path)
        parameters = extract_parameters(workspace)
        inferred_n_bins = infer_n_bins(parameters)
        if n_bins is None:
            n_bins = inferred_n_bins
        elif n_bins != inferred_n_bins:
            raise ValueError(
                f"Provided n_bins={n_bins} does not match workspace-inferred n_bins={inferred_n_bins}"
            )

        results: list[dict[str, Any]] = []
        for framework in frameworks:
            result = measure_framework(
                framework=framework,
                workspace_path=workspace_path,
                parameters=parameters,
                n_bins=n_bins,
                mu=mu,
                warmup_iterations=warmup_iterations,
            )
            results.append(result)
            if result.get("status") != "success" and not continue_on_framework_error:
                raise RuntimeError(
                    f"{framework} measurement failed: {result.get('error_message')}"
                )

        add_validation(results, agreement_tolerance)
        status = benchmark_status(results)

        output_data = {
            "benchmark": BENCHMARK_NAME,
            "status": status,
            "workspace": str(workspace_path),
            "workspace_name": workspace_path.name,
            "n_bins": n_bins,
            "mu": mu,
            "frameworks": frameworks,
            "warmup_iterations": warmup_iterations,
            "agreement_tolerance": agreement_tolerance,
            "notes": "Stages are framework-specific but intentionally separated into input load, model construction, cold first evaluation, warm-up, and warm evaluation.",
            "results": results,
        }

        print("=" * 80)
        print("Model build / setup cost benchmark")
        print("=" * 80)
        print(f"Workspace:  {workspace_path.name}")
        print(f"Bins:       {n_bins}")
        print(f"mu:         {mu}")
        print(f"Warm-up:    {warmup_iterations} unmeasured evaluation(s)")
        print(f"Frameworks: {', '.join(frameworks)}")
        print(f"Status:     {status}")

        for result in results:
            print_result(result)

        save_json(output_data, output)
        print()
        print(f"Saved result to {output}")

        if plot:
            make_plots(results, plot_dir)
            print(f"Saved plots to {plot_dir}")

        return output_data

    except Exception as exc:
        output_data = build_failed_output(
            workspace_path=workspace_path,
            n_bins=n_bins,
            mu=mu,
            frameworks=frameworks,
            warmup_iterations=warmup_iterations,
            agreement_tolerance=agreement_tolerance,
            exc=exc,
        )
        try:
            save_json(output_data, output)
        except Exception as save_exc:
            print(
                f"Failed to save benchmark failure report: {save_exc}", file=sys.stderr
            )
        raise RuntimeError("Model build / setup cost benchmark failed") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure framework setup costs and cold/warm first-evaluation latency for binned Poisson models."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument(
        "--n-bins",
        type=int,
        default=None,
        help="Optional bin count. If omitted, it is inferred from obs_<index> parameters in the workspace.",
    )
    parser.add_argument("--mu", type=float, default=DEFAULT_MU)
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=DEFAULT_FRAMEWORKS,
        choices=sorted(FRAMEWORK_ORDER),
    )
    parser.add_argument(
        "--warmup-iterations", type=int, default=DEFAULT_WARMUP_ITERATIONS
    )
    parser.add_argument(
        "--agreement-tolerance", type=float, default=DEFAULT_AGREEMENT_TOLERANCE
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / BENCHMARK_NAME / f"{BENCHMARK_NAME}_result.json",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=PLOTS_DIR / BENCHMARK_NAME,
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort after the first framework failure instead of recording failed framework entries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        workspace_path=args.workspace,
        n_bins=args.n_bins,
        mu=args.mu,
        frameworks=args.frameworks,
        warmup_iterations=args.warmup_iterations,
        agreement_tolerance=args.agreement_tolerance,
        output=args.output,
        plot=args.plot,
        plot_dir=args.plot_dir,
        continue_on_framework_error=not args.fail_fast,
    )


if __name__ == "__main__":
    main()
