"""Cross-framework scalar PDF evaluation benchmark.

This benchmark measures repeated scalar PDF evaluation for simple probability
models across PyHS3, numba-stats, RooFit, and zfit.  It is complementary to the
vectorized PDF benchmark: here every framework is compared on the same scalar
workload, while the vectorized benchmark studies native vectorized APIs.
"""

from __future__ import annotations

import argparse
import gc
import math
import multiprocessing as mp
import queue
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import PLOTS_DIR, RESULTS_DIR
    from src.utils import get_current_rss_mb, get_peak_rss_mb, save_json
else:
    from .config import PLOTS_DIR, RESULTS_DIR
    from .utils import get_current_rss_mb, get_peak_rss_mb, save_json


BENCHMARK_NAME = "cross_scalar_pdf_evaluation"

DEFAULT_SCENARIOS = ["normal", "poisson"]
DEFAULT_FRAMEWORKS = ["pyhs3", "numba_stats", "root", "zfit"]
DEFAULT_N_EVALUATIONS = [1, 10, 100, 1000, 10000]
DEFAULT_N_POINTS = 1000
DEFAULT_PYHS3_WORKSPACE_DIR = Path("inputs/scalar_pdf_workspaces")
DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "cross_scalar_pdf_evaluation_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME

SUPPORTED_SCENARIOS = ("normal", "poisson", "exponential")
SUPPORTED_FRAMEWORKS = ("pyhs3", "numba_stats", "root", "zfit")
REFERENCE_FRAMEWORK = "manual"

FRAMEWORK_STYLE = {
    "pyhs3": {"label": "PyHS3", "color": "#0B5EA8", "marker": "s", "linestyle": "-"},
    "numba_stats": {
        "label": "numba-stats",
        "color": "#FF7F00",
        "marker": "^",
        "linestyle": "--",
    },
    "root": {"label": "RooFit", "color": "#009E73", "marker": "D", "linestyle": "-."},
    "zfit": {"label": "zfit", "color": "#CC79A7", "marker": "o", "linestyle": ":"},
}

SCENARIO_LABELS = {
    "normal": "Normal",
    "poisson": "Poisson",
    "exponential": "Exponential",
}


class BenchmarkConfigurationError(ValueError):
    """Raised when the benchmark configuration is invalid."""


class ValidationFailure(RuntimeError):
    """Raised when a framework output does not match the reference values."""


@dataclass(frozen=True)
class ScalarBenchmarkConfig:
    framework: str
    scenario: str
    n_evaluations: int
    n_points: int
    rtol: float
    atol: float
    pyhs3_workspace_dir: Path


def _framework_label(framework: str) -> str:
    return FRAMEWORK_STYLE.get(framework, {"label": framework})["label"]


def _scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario)


def _style_for(framework: str) -> dict[str, Any]:
    return FRAMEWORK_STYLE.get(
        framework,
        {"label": framework, "color": "#333333", "marker": "o", "linestyle": "-"},
    )


def _ordered_successful_results(
    results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [result for result in results if result.get("status") == "success"]


def _safe_positive(value: float, floor: float = 1e-300) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return floor
    return float(value)


def _format_seconds_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.3f} ms"


def _format_ns(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.2f} µs"
    return f"{value:.1f} ns"


def _format_scientific(value: float) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.1e}"


def validate_benchmark_config(
    *,
    frameworks: list[str],
    scenarios: list[str],
    n_evaluations: list[int],
    n_points: int,
    rtol: float,
    atol: float,
    timeout_seconds: float,
    pyhs3_workspace_dir: Path,
) -> None:
    if not frameworks:
        raise BenchmarkConfigurationError("At least one framework must be selected.")
    if not scenarios:
        raise BenchmarkConfigurationError("At least one scenario must be selected.")

    unknown_frameworks = sorted(set(frameworks) - set(SUPPORTED_FRAMEWORKS))
    if unknown_frameworks:
        raise BenchmarkConfigurationError(
            f"Unknown framework(s): {', '.join(unknown_frameworks)}"
        )

    unknown_scenarios = sorted(set(scenarios) - set(SUPPORTED_SCENARIOS))
    if unknown_scenarios:
        raise BenchmarkConfigurationError(
            f"Unknown scenario(s): {', '.join(unknown_scenarios)}"
        )

    if any(value < 1 for value in n_evaluations):
        raise BenchmarkConfigurationError(
            "All --n-evaluations values must be at least 1."
        )
    if n_points < 1:
        raise BenchmarkConfigurationError("--n-points must be at least 1.")
    if rtol < 0.0 or atol < 0.0:
        raise BenchmarkConfigurationError("--rtol and --atol must be non-negative.")
    if timeout_seconds <= 0.0:
        raise BenchmarkConfigurationError("--timeout-seconds must be positive.")

    if "pyhs3" in frameworks and not pyhs3_workspace_dir.exists():
        raise FileNotFoundError(
            f"PyHS3 workspace directory does not exist: {pyhs3_workspace_dir}"
        )


def normal_reference(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x.astype(float) ** 2) / math.sqrt(2.0 * math.pi)


def poisson_reference(x: np.ndarray) -> np.ndarray:
    mean = 5.0
    return np.asarray(
        [
            math.exp(k * math.log(mean) - mean - math.lgamma(k + 1.0))
            for k in x.astype(int)
        ],
        dtype=float,
    )


def exponential_reference(x: np.ndarray) -> np.ndarray:
    return np.exp(-x.astype(float))


def make_input_grid(scenario: str, n_points: int) -> np.ndarray:
    if n_points < 1:
        raise BenchmarkConfigurationError("n_points must be at least 1.")
    if scenario == "normal":
        return np.linspace(-5.0, 5.0, n_points, dtype=float)
    if scenario == "poisson":
        # Discrete support.  n_points is intentionally ignored so every framework
        # evaluates the same stable Poisson support.
        return np.arange(0, 30, dtype=int)
    if scenario == "exponential":
        return np.linspace(0.0, 10.0, n_points, dtype=float)
    raise ValueError(f"Unknown scenario: {scenario}")


def reference_values(scenario: str, x: np.ndarray) -> np.ndarray:
    if scenario == "normal":
        return normal_reference(x)
    if scenario == "poisson":
        return poisson_reference(x)
    if scenario == "exponential":
        return exponential_reference(x)
    raise ValueError(f"Unknown scenario: {scenario}")


def evaluate_pyhs3(
    scenario: str, x: np.ndarray, pyhs3_workspace_dir: Path
) -> np.ndarray:
    workspace_path = pyhs3_workspace_dir / f"{scenario}_pdf_workspace.json"
    if not workspace_path.exists():
        raise FileNotFoundError(
            f"PyHS3 workspace not found for scenario '{scenario}': {workspace_path}"
        )

    from pyhs3.workspace import Workspace

    workspace = Workspace.load(workspace_path)
    model = workspace.model("analysis", progress=False, mode="FAST_RUN")
    base_parameters = {
        name: np.asarray(value, dtype=float)
        for name, value in {**model.data, **model.free_params}.items()
    }

    values = np.empty(len(x), dtype=float)
    for index, value in enumerate(x):
        parameters = dict(base_parameters)
        parameters["x"] = np.asarray(float(value), dtype=float)
        result = model.pdf("pdf", **parameters)
        values[index] = float(np.asarray(result).reshape(-1)[0])
    return values


def evaluate_numba_stats(scenario: str, x: np.ndarray) -> np.ndarray:
    if scenario == "normal":
        from numba_stats import norm

        return np.asarray(norm.pdf(x, 0.0, 1.0), dtype=float)
    if scenario == "poisson":
        from numba_stats import poisson

        return np.asarray(poisson.pmf(x.astype(int), 5.0), dtype=float)
    if scenario == "exponential":
        from numba_stats import expon

        return np.asarray(expon.pdf(x, 0.0, 1.0), dtype=float)
    raise ValueError(f"Unknown scenario: {scenario}")


def evaluate_root(scenario: str, x: np.ndarray) -> np.ndarray:
    import ROOT

    xvar = ROOT.RooRealVar("x", "x", float(np.min(x)), float(np.max(x)))
    normalization_factor = 1.0

    if scenario == "normal":
        mu = ROOT.RooRealVar("mu", "mu", 0.0)
        sigma = ROOT.RooRealVar("sigma", "sigma", 1.0, 1e-6, 100.0)
        pdf = ROOT.RooGaussian("normal", "normal", xvar, mu, sigma)
        # getVal() without an explicit normalization set returns the Gaussian
        # shape.  Multiplying by 1/sqrt(2pi) gives the normalized standard normal
        # density expected by the benchmark reference.
        normalization_factor = 1.0 / math.sqrt(2.0 * math.pi)
        keepalive = [xvar, mu, sigma, pdf]
    elif scenario == "poisson":
        mean = ROOT.RooRealVar("mean", "mean", 5.0, 0.0, 1000.0)
        pdf = ROOT.RooPoisson("poisson", "poisson", xvar, mean)
        keepalive = [xvar, mean, pdf]
    elif scenario == "exponential":
        tau = ROOT.RooRealVar("tau", "tau", -1.0)
        pdf = ROOT.RooExponential("exponential", "exponential", xvar, tau)
        keepalive = [xvar, tau, pdf]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    values = np.empty(len(x), dtype=float)
    for index, value in enumerate(x):
        xvar.setVal(float(value))
        values[index] = float(pdf.getVal()) * normalization_factor

    # Keep RooFit inputs alive until all evaluations are complete.
    _ = keepalive
    return values


def evaluate_zfit(scenario: str, x: np.ndarray) -> np.ndarray:
    import zfit

    obs = zfit.Space("x", limits=(float(np.min(x)), float(np.max(x))))
    data = zfit.Data.from_numpy(obs=obs, array=np.asarray(x, dtype=float))

    if scenario == "normal":
        pdf = zfit.pdf.Gauss(obs=obs, mu=0.0, sigma=1.0)
    elif scenario == "exponential":
        pdf = zfit.pdf.Exponential(obs=obs, lam=-1.0)
    elif scenario == "poisson":
        raise NotImplementedError(
            "zfit Poisson scalar PDF is not included in this benchmark yet."
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return np.asarray(pdf.pdf(data).numpy(), dtype=float)


def evaluate_framework_once(
    *,
    framework: str,
    scenario: str,
    x: np.ndarray,
    pyhs3_workspace_dir: Path,
) -> np.ndarray:
    if framework == "pyhs3":
        return evaluate_pyhs3(scenario, x, pyhs3_workspace_dir)
    if framework == "numba_stats":
        return evaluate_numba_stats(scenario, x)
    if framework == "root":
        return evaluate_root(scenario, x)
    if framework == "zfit":
        return evaluate_zfit(scenario, x)
    raise ValueError(f"Unknown framework: {framework}")


def compute_agreement(
    observed: np.ndarray, reference: np.ndarray, rtol: float, atol: float
) -> dict[str, Any]:
    observed = np.asarray(observed, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if observed.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape}, reference {reference.shape}"
        )

    finite_mask = np.isfinite(observed)
    if not np.all(finite_mask):
        raise ValidationFailure(
            f"Framework returned {int((~finite_mask).sum())} non-finite PDF values"
        )

    abs_diff = np.abs(observed - reference)
    rel_diff = abs_diff / np.maximum(np.abs(reference), 1e-300)
    allclose_passed = bool(np.allclose(observed, reference, rtol=rtol, atol=atol))

    return {
        "n_values": int(observed.size),
        "n_finite_values": int(finite_mask.sum()),
        "all_values_finite": bool(np.all(finite_mask)),
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs_diff": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "max_rel_diff": float(np.max(rel_diff)) if rel_diff.size else 0.0,
        "mean_rel_diff": float(np.mean(rel_diff)) if rel_diff.size else 0.0,
        "allclose_passed": allclose_passed,
        "validation_status": "success" if allclose_passed else "failed",
    }


def run_single_framework_benchmark(config: ScalarBenchmarkConfig) -> dict[str, Any]:
    x = make_input_grid(config.scenario, config.n_points)
    reference = reference_values(config.scenario, x)

    gc.collect()
    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    cold_start_start = time.perf_counter()
    first_values = evaluate_framework_once(
        framework=config.framework,
        scenario=config.scenario,
        x=x,
        pyhs3_workspace_dir=config.pyhs3_workspace_dir,
    )
    cold_start_time_seconds = time.perf_counter() - cold_start_start

    agreement_summary = compute_agreement(
        first_values, reference, config.rtol, config.atol
    )
    if not agreement_summary["allclose_passed"]:
        raise ValidationFailure(
            "PDF value agreement failed "
            f"(max abs diff={agreement_summary['max_abs_diff']:.3e}, "
            f"max rel diff={agreement_summary['max_rel_diff']:.3e})"
        )

    warm_start = time.perf_counter()
    last_values = first_values
    for _ in range(config.n_evaluations):
        last_values = evaluate_framework_once(
            framework=config.framework,
            scenario=config.scenario,
            x=x,
            pyhs3_workspace_dir=config.pyhs3_workspace_dir,
        )
    total_runtime_seconds = time.perf_counter() - warm_start

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    average_runtime_seconds_per_evaluation = (
        total_runtime_seconds / config.n_evaluations
    )
    time_per_value_seconds = average_runtime_seconds_per_evaluation / len(x)
    throughput_values_per_second = (
        config.n_evaluations * len(x) / total_runtime_seconds
        if total_runtime_seconds > 0.0
        else float("inf")
    )

    del last_values
    gc.collect()

    return {
        "benchmark": BENCHMARK_NAME,
        "framework": config.framework,
        "framework_label": _framework_label(config.framework),
        "scenario": config.scenario,
        "scenario_label": _scenario_label(config.scenario),
        "n_evaluations": int(config.n_evaluations),
        "requested_n_points": int(config.n_points),
        "n_points": int(len(x)),
        "cold_start_time_seconds": float(cold_start_time_seconds),
        "total_runtime_seconds": float(total_runtime_seconds),
        "average_runtime_seconds_per_evaluation": float(
            average_runtime_seconds_per_evaluation
        ),
        "time_per_value_seconds": float(time_per_value_seconds),
        "time_per_value_ns": float(time_per_value_seconds * 1e9),
        "throughput_values_per_second": float(throughput_values_per_second),
        "current_rss_before_mb": float(current_rss_before_mb),
        "current_rss_after_mb": float(current_rss_after_mb),
        "current_rss_delta_mb": float(current_rss_after_mb - current_rss_before_mb),
        "peak_rss_before_mb": float(peak_rss_before_mb),
        "peak_rss_after_mb": float(peak_rss_after_mb),
        "peak_rss_delta_mb": float(peak_rss_after_mb - peak_rss_before_mb),
        "status": "success",
        **agreement_summary,
    }


def _failure_result(
    config: ScalarBenchmarkConfig, status: str, **extra: Any
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "framework": config.framework,
        "framework_label": _framework_label(config.framework),
        "scenario": config.scenario,
        "scenario_label": _scenario_label(config.scenario),
        "n_evaluations": int(config.n_evaluations),
        "requested_n_points": int(config.n_points),
        "n_points": int(make_input_grid(config.scenario, config.n_points).size)
        if config.scenario in SUPPORTED_SCENARIOS
        else int(config.n_points),
        "status": status,
        **extra,
    }


def run_worker(payload: dict[str, Any], output_queue: mp.Queue) -> None:
    config = ScalarBenchmarkConfig(
        framework=payload["framework"],
        scenario=payload["scenario"],
        n_evaluations=payload["n_evaluations"],
        n_points=payload["n_points"],
        rtol=payload["rtol"],
        atol=payload["atol"],
        pyhs3_workspace_dir=Path(payload["pyhs3_workspace_dir"]),
    )

    try:
        output_queue.put(run_single_framework_benchmark(config))
    except Exception as error:  # noqa: BLE001 - worker must serialize all failures
        output_queue.put(
            _failure_result(
                config,
                "failed",
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=traceback.format_exc(),
            )
        )


def run_with_timeout(payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    config = ScalarBenchmarkConfig(
        framework=payload["framework"],
        scenario=payload["scenario"],
        n_evaluations=payload["n_evaluations"],
        n_points=payload["n_points"],
        rtol=payload["rtol"],
        atol=payload["atol"],
        pyhs3_workspace_dir=Path(payload["pyhs3_workspace_dir"]),
    )

    ctx = mp.get_context("spawn")
    output_queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(target=run_worker, args=(payload, output_queue))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join()
        return _failure_result(
            config, "timeout", timeout_seconds=float(timeout_seconds)
        )

    if process.exitcode not in (0, None):
        return _failure_result(
            config,
            "failed",
            error_type="ProcessExitError",
            error_message=f"Worker exited with code {process.exitcode}",
        )

    try:
        return output_queue.get_nowait()
    except queue.Empty:
        return _failure_result(
            config,
            "failed",
            error_type="EmptyWorkerResult",
            error_message="Worker finished without returning a result.",
        )


def print_result(result: dict[str, Any]) -> None:
    print()
    print("-" * 72)
    print(
        f"{result.get('scenario_label', result.get('scenario'))} / "
        f"{result.get('framework_label', result.get('framework'))} / "
        f"evaluations={result.get('n_evaluations')}"
    )
    print("-" * 72)
    print(f"status:                  {result['status']}")

    if result["status"] != "success":
        print("validation:              failed")
        print(
            f"error:                   {result.get('error_type', result['status'])}: {result.get('error_message', '')}"
        )
        return

    print(f"validation:              {result['validation_status']}")
    print(f"points per evaluation:   {result['n_points']}")
    print(
        f"cold start:              {_format_seconds_ms(result['cold_start_time_seconds'])}"
    )
    print(
        f"warm total:              {_format_seconds_ms(result['total_runtime_seconds'])}"
    )
    print(
        "warm / evaluation:       "
        f"{result['average_runtime_seconds_per_evaluation'] * 1000.0:.6f} ms"
    )
    print(f"time per value:          {_format_ns(result['time_per_value_ns'])}")
    print(
        f"throughput:              {result['throughput_values_per_second']:.3e} values/s"
    )
    print(f"current RSS delta:       {result['current_rss_delta_mb']:.3f} MB")
    print(f"peak RSS delta:          {result['peak_rss_delta_mb']:.3f} MB")
    print(f"max abs diff:            {result['max_abs_diff']:.6e}")
    print(f"max rel diff:            {result['max_rel_diff']:.6e}")


def summarize_status(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "success"]
    failed = [result for result in results if result.get("status") != "success"]
    return {
        "status": "success" if len(successful) == len(results) else "failed",
        "n_results": len(results),
        "n_successful": len(successful),
        "n_failed": len(failed),
        "failed_results": [
            {
                "scenario": result.get("scenario"),
                "framework": result.get("framework"),
                "n_evaluations": result.get("n_evaluations"),
                "status": result.get("status"),
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
            }
            for result in failed
        ],
    }


def print_final_summary(results: list[dict[str, Any]]) -> None:
    summary = summarize_status(results)
    print()
    print("=" * 80)
    print("Cross-framework scalar PDF evaluation benchmark")
    print("=" * 80)
    print(f"Status:      {summary['status']}")
    print(f"Successful:  {summary['n_successful']} / {summary['n_results']}")
    if summary["failed_results"]:
        print("Failed:")
        for result in summary["failed_results"]:
            print(
                "  - "
                f"{result['scenario']} / {result['framework']} / "
                f"evals={result['n_evaluations']}: "
                f"{result.get('error_type') or result['status']} "
                f"{result.get('error_message') or ''}"
            )


def _success_dataframe(results: list[dict[str, Any]]):
    import pandas as pd

    rows = []
    for result in _ordered_successful_results(results):
        rows.append(
            {
                "scenario": result["scenario_label"],
                "scenario_key": result["scenario"],
                "framework": result["framework_label"],
                "framework_key": result["framework"],
                "n_evaluations": result["n_evaluations"],
                "n_points": result["n_points"],
                "cold_ms": result["cold_start_time_seconds"] * 1000.0,
                "warm_ms": result["average_runtime_seconds_per_evaluation"] * 1000.0,
                "ns_per_value": result["time_per_value_ns"],
                "throughput": result["throughput_values_per_second"],
                "current_rss_mb": max(result["current_rss_delta_mb"], 0.0),
                "peak_rss_mb": max(result["peak_rss_delta_mb"], 0.0),
                "max_abs_diff": result["max_abs_diff"],
                "max_rel_diff": result["max_rel_diff"],
            }
        )
    return pd.DataFrame(rows)


def _prepare_figure(figsize: tuple[float, float] = (12.0, 7.0)):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.18)
    ax.tick_params(axis="both", which="both", direction="in", width=1.2, length=6)
    ax.tick_params(axis="both", which="major", labelsize=13)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    return fig, ax


def _save_figure(fig: Any, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def make_throughput_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(df["scenario_key"]))
    fig, axes = plt.subplots(
        len(scenarios),
        1,
        figsize=(12.0, 5.8 * len(scenarios)),
        squeeze=False,
    )

    for ax, scenario in zip(axes.flat, scenarios, strict=False):
        subset = df[df["scenario_key"] == scenario]
        for framework in list(dict.fromkeys(subset["framework_key"])):
            framework_subset = subset[subset["framework_key"] == framework].sort_values(
                "n_evaluations"
            )
            style = _style_for(framework)
            ax.plot(
                framework_subset["n_evaluations"],
                framework_subset["throughput"],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.5,
                markersize=8,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            f"{_scenario_label(scenario)} PDF",
            loc="left",
            fontsize=18,
            fontweight="bold",
        )
        ax.set_xlabel("Number of repeated scalar evaluations", fontsize=15)
        ax.set_ylabel("Throughput [values/s]", fontsize=15)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.18)
        ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=13
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    fig.suptitle(
        "Scalar PDF evaluation throughput scaling",
        x=0.02,
        ha="left",
        fontsize=28,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_latency_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    """Plot cold-start and warm scalar-evaluation latency without overcrowded bar labels.

    The previous grouped-bar version produced one x-label per scenario/framework/evaluation
    combination, which became unreadable for the real run.  This view uses one row per
    scenario and two columns: cold first evaluation and warm steady-state evaluation.
    """
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(df["scenario_key"]))
    fig, axes = plt.subplots(
        len(scenarios),
        2,
        figsize=(15.5, 4.8 * len(scenarios)),
        squeeze=False,
        sharex=False,
    )

    for row_index, scenario in enumerate(scenarios):
        subset = df[df["scenario_key"] == scenario]
        scenario_label = _scenario_label(scenario)

        panels = [
            ("cold_ms", "Cold start [ms]"),
            ("warm_ms", "Warm / evaluation [ms]"),
        ]

        for col_index, (column, ylabel) in enumerate(panels):
            ax = axes[row_index][col_index]
            for framework in list(dict.fromkeys(subset["framework_key"])):
                framework_subset = subset[
                    subset["framework_key"] == framework
                ].sort_values("n_evaluations")
                style = _style_for(framework)
                ax.plot(
                    framework_subset["n_evaluations"],
                    np.maximum(framework_subset[column].to_numpy(dtype=float), 1e-12),
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2.4,
                    markersize=7,
                )

                # Annotate only the last point, not every point.  This keeps the
                # plot informative without the labels colliding.
                last = framework_subset.iloc[-1]
                last_value = max(float(last[column]), 1e-12)
                ax.annotate(
                    f"{last_value:.3g}",
                    (float(last["n_evaluations"]), last_value),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color=style["color"],
                )

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(
                f"{scenario_label} · {ylabel}",
                loc="left",
                fontsize=16,
                fontweight="bold",
            )
            ax.set_xlabel("Repeated scalar evaluations", fontsize=13)
            ax.set_ylabel(ylabel, fontsize=13)
            ax.grid(True, which="major", alpha=0.35)
            ax.grid(True, which="minor", alpha=0.18)
            ax.tick_params(
                axis="both", which="both", direction="in", width=1.2, length=6
            )
            for spine in ax.spines.values():
                spine.set_linewidth(1.4)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=max(1, len(labels)),
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(
        "Cold-start and warm scalar PDF evaluation latency",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=26,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_time_per_value_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(df["scenario_key"]))
    fig, axes = plt.subplots(
        len(scenarios), 1, figsize=(12.0, 5.8 * len(scenarios)), squeeze=False
    )
    for ax, scenario in zip(axes.flat, scenarios, strict=False):
        subset = df[df["scenario_key"] == scenario]
        for framework in list(dict.fromkeys(subset["framework_key"])):
            framework_subset = subset[subset["framework_key"] == framework].sort_values(
                "n_evaluations"
            )
            style = _style_for(framework)
            ax.plot(
                framework_subset["n_evaluations"],
                framework_subset["ns_per_value"],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.5,
                markersize=8,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            f"{_scenario_label(scenario)} PDF",
            loc="left",
            fontsize=18,
            fontweight="bold",
        )
        ax.set_xlabel("Number of repeated scalar evaluations", fontsize=15)
        ax.set_ylabel("Time/value [ns]", fontsize=15)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.18)
        ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=13
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    fig.suptitle(
        "Scalar PDF evaluation cost per value",
        x=0.02,
        ha="left",
        fontsize=28,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_memory_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    """Plot RSS memory deltas as trends instead of a dense labelled bar chart."""
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    scenarios = list(dict.fromkeys(df["scenario_key"]))
    fig, axes = plt.subplots(
        len(scenarios),
        1,
        figsize=(13.5, 4.7 * len(scenarios)),
        squeeze=False,
        sharex=False,
    )

    for ax, scenario in zip(axes.flat, scenarios, strict=False):
        subset = df[df["scenario_key"] == scenario]
        for framework in list(dict.fromkeys(subset["framework_key"])):
            framework_subset = subset[subset["framework_key"] == framework].sort_values(
                "n_evaluations"
            )
            style = _style_for(framework)
            x_values = framework_subset["n_evaluations"].to_numpy(dtype=float)
            current = np.maximum(
                framework_subset["current_rss_mb"].to_numpy(dtype=float), 1e-3
            )
            peak = np.maximum(
                framework_subset["peak_rss_mb"].to_numpy(dtype=float), 1e-3
            )

            ax.plot(
                x_values,
                current,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle="-",
                linewidth=2.4,
                markersize=7,
            )
            ax.plot(
                x_values,
                peak,
                color=style["color"],
                marker=style["marker"],
                linestyle="--",
                linewidth=1.8,
                markersize=5,
                alpha=0.75,
            )

            # Annotate one representative value at the largest n_evaluations.
            last_x = float(x_values[-1])
            last_value = float(current[-1])
            ax.annotate(
                f"{last_value:.0f} MB" if last_value >= 1 else f"{last_value:.2g} MB",
                (last_x, last_value),
                xytext=(6, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=style["color"],
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            f"{_scenario_label(scenario)} PDF",
            loc="left",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Repeated scalar evaluations", fontsize=13)
        ax.set_ylabel("Memory delta [MB]", fontsize=13)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.18)
        ax.tick_params(axis="both", which="both", direction="in", width=1.2, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    framework_handles, framework_labels = axes[0][0].get_legend_handles_labels()
    metric_handles = [
        Line2D(
            [0], [0], color="black", linestyle="-", linewidth=2.4, label="Current RSS Δ"
        ),
        Line2D(
            [0], [0], color="black", linestyle="--", linewidth=1.8, label="Peak RSS Δ"
        ),
    ]

    fig.legend(
        framework_handles + metric_handles,
        framework_labels + ["Current RSS Δ", "Peak RSS Δ"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=max(2, len(framework_labels) + 2),
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(
        "Memory footprint during scalar PDF evaluation",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=26,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_agreement_plot(
    results: list[dict[str, Any]], output_path: Path, tolerance: float
) -> None:
    """Plot numerical agreement as a clean per-scenario line plot."""
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    scenarios = list(dict.fromkeys(df["scenario_key"]))
    fig, axes = plt.subplots(
        len(scenarios),
        1,
        figsize=(13.5, 4.7 * len(scenarios)),
        squeeze=False,
        sharex=False,
    )

    floor = min(1e-18, max(tolerance, 1e-300) * 1e-8)

    for ax, scenario in zip(axes.flat, scenarios, strict=False):
        subset = df[df["scenario_key"] == scenario]
        for framework in list(dict.fromkeys(subset["framework_key"])):
            framework_subset = subset[subset["framework_key"] == framework].sort_values(
                "n_evaluations"
            )
            style = _style_for(framework)
            y_values = np.asarray(
                [
                    _safe_positive(value, floor)
                    for value in framework_subset["max_abs_diff"]
                ],
                dtype=float,
            )

            ax.plot(
                framework_subset["n_evaluations"],
                y_values,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.4,
                markersize=7,
            )

            max_raw = float(framework_subset["max_abs_diff"].max())
            label = "0" if max_raw == 0.0 else f"max {max_raw:.1e}"
            max_index = int(np.argmax(y_values))
            ax.annotate(
                label,
                (
                    float(framework_subset["n_evaluations"].iloc[max_index]),
                    float(y_values[max_index]),
                ),
                xytext=(6, 6),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=style["color"],
            )

        ax.axhline(
            tolerance,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label=f"tolerance = {tolerance:g}",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            f"{_scenario_label(scenario)} PDF",
            loc="left",
            fontsize=16,
            fontweight="bold",
        )
        ax.set_xlabel("Repeated scalar evaluations", fontsize=13)
        ax.set_ylabel("max |PDF - reference|", fontsize=13)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.18)
        ax.tick_params(axis="both", which="both", direction="in", width=1.2, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    handles, labels = axes[0][0].get_legend_handles_labels()
    # Deduplicate tolerance label across scenario panels.
    deduped = dict(zip(labels, handles, strict=False))
    fig.legend(
        deduped.values(),
        deduped.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=max(2, len(deduped)),
        frameon=False,
        fontsize=12,
    )
    fig.suptitle(
        "Scalar PDF numerical agreement",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=26,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_summary_table(results: list[dict[str, Any]], output_path: Path) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    rows = []
    for result in _ordered_successful_results(results):
        rows.append(
            [
                result["scenario_label"],
                result["framework_label"],
                str(result["n_evaluations"]),
                str(result["n_points"]),
                f"{result['cold_start_time_seconds'] * 1000.0:.3g}",
                f"{result['average_runtime_seconds_per_evaluation'] * 1000.0:.3g}",
                f"{result['time_per_value_ns']:.3g}",
                f"{result['throughput_values_per_second']:.2e}",
                f"{max(result['current_rss_delta_mb'], 0.0):.2f}",
                _format_scientific(result["max_abs_diff"]),
            ]
        )

    columns = [
        "Scenario",
        "Framework",
        "Evals",
        "Points",
        "Cold [ms]",
        "Warm [ms]",
        "ns/value",
        "Throughput",
        "RSS Δ [MB]",
        "max diff",
    ]
    fig_height = max(3.5, 0.45 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(15.0, fig_height))
    ax.axis("off")
    fig.text(
        0.02,
        0.93,
        "Cross-framework scalar PDF evaluation summary",
        fontsize=28,
        fontweight="bold",
        ha="left",
    )
    fig.text(
        0.02,
        0.84,
        "Repeated scalar PDF evaluations across equivalent simple PDFs.",
        fontsize=15,
        ha="left",
    )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.35)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#BDBDBD")
        if row == 0:
            cell.set_facecolor("#262626")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F5F5F5")

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_plots(results: list[dict[str, Any]], plot_dir: Path, tolerance: float) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    successful = _ordered_successful_results(results)
    if len(successful) < 2:
        print("Skipping plots: at least two successful results are needed.")
        return

    make_throughput_plot(results, plot_dir / "cross_scalar_pdf_throughput_scaling.png")
    make_time_per_value_plot(results, plot_dir / "cross_scalar_pdf_time_per_value.png")
    make_latency_plot(results, plot_dir / "cross_scalar_pdf_latency.png")
    make_memory_plot(results, plot_dir / "cross_scalar_pdf_memory.png")
    make_agreement_plot(
        results, plot_dir / "cross_scalar_pdf_numerical_agreement.png", tolerance
    )
    make_summary_table(results, plot_dir / "cross_scalar_pdf_summary_table.png")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-framework scalar PDF evaluation benchmark."
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=DEFAULT_FRAMEWORKS,
        choices=SUPPORTED_FRAMEWORKS,
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=DEFAULT_SCENARIOS, choices=SUPPORTED_SCENARIOS
    )
    parser.add_argument(
        "--n-evaluations", nargs="+", type=int, default=DEFAULT_N_EVALUATIONS
    )
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument(
        "--pyhs3-workspace-dir", type=Path, default=DEFAULT_PYHS3_WORKSPACE_DIR
    )
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return parser.parse_args(argv)


def build_payload(
    args: argparse.Namespace, framework: str, scenario: str, n_evaluations: int
) -> dict[str, Any]:
    return {
        "framework": framework,
        "scenario": scenario,
        "n_evaluations": int(n_evaluations),
        "n_points": int(args.n_points),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "pyhs3_workspace_dir": str(args.pyhs3_workspace_dir),
    }


def run(
    *,
    frameworks: list[str],
    scenarios: list[str],
    n_evaluations: list[int],
    n_points: int,
    pyhs3_workspace_dir: Path,
    rtol: float,
    atol: float,
    timeout_seconds: float,
    output_dir: Path,
    output_name: str,
    plot: bool,
    plot_dir: Path,
) -> dict[str, Any]:
    validate_benchmark_config(
        frameworks=frameworks,
        scenarios=scenarios,
        n_evaluations=n_evaluations,
        n_points=n_points,
        rtol=rtol,
        atol=atol,
        timeout_seconds=timeout_seconds,
        pyhs3_workspace_dir=pyhs3_workspace_dir,
    )

    args = argparse.Namespace(
        n_points=n_points,
        pyhs3_workspace_dir=pyhs3_workspace_dir,
        rtol=rtol,
        atol=atol,
    )

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        for framework in frameworks:
            for n_eval in n_evaluations:
                print(
                    f"Running scenario={scenario}, framework={framework}, n_evaluations={n_eval}",
                    flush=True,
                )
                result = run_with_timeout(
                    build_payload(args, framework, scenario, n_eval), timeout_seconds
                )
                results.append(result)
                print_result(result)

    summary = summarize_status(results)
    output_data: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "summary": summary,
        "configuration": {
            "frameworks": frameworks,
            "scenarios": scenarios,
            "n_evaluations": n_evaluations,
            "n_points": n_points,
            "rtol": rtol,
            "atol": atol,
            "timeout_seconds": timeout_seconds,
            "pyhs3_workspace_dir": str(pyhs3_workspace_dir),
        },
        "results": results,
    }

    output_path = output_dir / output_name
    save_json(output_data, output_path)
    print_final_summary(results)
    print(f"Saved result to {output_path}")

    if plot:
        make_plots(results, plot_dir, tolerance=atol)
        print(f"Saved plots to {plot_dir}")

    return output_data


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(
            frameworks=list(args.frameworks),
            scenarios=list(args.scenarios),
            n_evaluations=list(args.n_evaluations),
            n_points=args.n_points,
            pyhs3_workspace_dir=args.pyhs3_workspace_dir,
            rtol=args.rtol,
            atol=args.atol,
            timeout_seconds=args.timeout_seconds,
            output_dir=args.output_dir,
            output_name=args.output_name,
            plot=args.plot,
            plot_dir=args.plot_dir,
        )
    except Exception as error:
        raise RuntimeError(
            "Cross-framework scalar PDF evaluation benchmark failed"
        ) from error


if __name__ == "__main__":
    main(sys.argv[1:])
