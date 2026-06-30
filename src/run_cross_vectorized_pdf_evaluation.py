"""Cross-framework PDF vectorization benchmark.

This benchmark compares scalar-PDF evaluation throughput across frameworks that
support different levels of vectorization. PyHS3 is intentionally benchmarked in
its current non-vectorized mode, while numba-stats and zfit use native vectorized
APIs where available. ROOT/RooFit is evaluated point-by-point via RooAbsPdf.

The benchmark is therefore a vectorization study, not a strict apples-to-apples
implementation comparison. It is useful for quantifying the benefit of native
vectorized evaluation and for identifying future PyHS3 optimization targets.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
import traceback
from dataclasses import dataclass
from multiprocessing import TimeoutError, get_context
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import PLOTS_DIR, RESULTS_DIR
    from src.utils import get_current_rss_mb, get_peak_rss_mb, save_json
else:
    from .config import PLOTS_DIR, RESULTS_DIR
    from .utils import get_current_rss_mb, get_peak_rss_mb, save_json


BENCHMARK_NAME = "cross_vectorized_pdf_evaluation"
BENCHMARK_TITLE = "Cross-framework PDF vectorization benchmark"

SUPPORTED_SCENARIOS = ("normal", "poisson")
SUPPORTED_FRAMEWORKS = ("pyhs3", "numba_stats", "root", "zfit")
DEFAULT_SCENARIOS = ["normal", "poisson"]
DEFAULT_FRAMEWORKS = ["pyhs3", "numba_stats", "root", "zfit"]
DEFAULT_N_POINTS = [100, 1000, 10000, 100000]
DEFAULT_N_RUNS = 20
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_RTOL = 1e-7
DEFAULT_ATOL = 1e-10

DEFAULT_PYHS3_WORKSPACE_DIR = Path("inputs/scalar_pdf_workspaces")
FALLBACK_PYHS3_WORKSPACE_DIR = Path("benchmarking/inputs/scalar_pdf_workspaces")
DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = f"{BENCHMARK_NAME}_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME

PLOT_EPSILON = 1e-300
FRAMEWORK_STYLE = {
    "pyhs3": {"label": "PyHS3", "color": "#1764ab", "marker": "s", "linestyle": "-"},
    "numba_stats": {
        "label": "numba-stats",
        "color": "#f57c00",
        "marker": "^",
        "linestyle": "--",
    },
    "root": {"label": "RooFit", "color": "#009b77", "marker": "D", "linestyle": "-."},
    "zfit": {"label": "zfit", "color": "#7b3294", "marker": "o", "linestyle": ":"},
}
SCENARIO_LABELS = {"normal": "Normal", "poisson": "Poisson"}


@dataclass(frozen=True)
class BenchmarkSpec:
    framework: str
    scenario: str
    n_points: int
    n_runs: int
    rtol: float
    atol: float
    pyhs3_workspace_dir: Path


def _framework_label(framework: str) -> str:
    return FRAMEWORK_STYLE.get(framework, {"label": framework})["label"]


def _scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario)


def _apply_cern_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 13,
            "axes.titlesize": 22,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 1.5,
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


def _format_plain_number(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    abs_value = abs(value)
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    if abs_value >= 1:
        return f"{value:.2f}"
    if abs_value >= 1e-3:
        return f"{value:.3f}"
    return f"{value:.1e}"


def _format_memory_mb(value: float) -> str:
    if not math.isfinite(value):
        return "nan MB"
    if abs(value) >= 100:
        return f"{value:.0f} MB"
    if abs(value) >= 10:
        return f"{value:.1f} MB"
    return f"{value:.2f} MB"


def _add_bar_labels(
    ax: Any,
    bars: Any,
    values: list[float],
    formatter: Callable[[float], str] = _format_compact,
    *,
    x_offset_points: int = 0,
    y_offset_points: int = 3,
    fontsize: int = 8,
) -> None:
    for bar, value in zip(bars, values, strict=True):
        height = max(float(bar.get_height()), PLOT_EPSILON)
        ax.annotate(
            formatter(value),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(x_offset_points, y_offset_points),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=fontsize,
            weight="bold",
            clip_on=False,
        )


def validate_positive_int(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")


def validate_positive_float(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite value, got {value}")


def validate_probability_tolerance(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite value, got {value}")


def validate_choices(values: list[str], valid: tuple[str, ...], name: str) -> None:
    invalid = sorted(set(values) - set(valid))
    if invalid:
        raise ValueError(
            f"Unsupported {name}: {invalid}. Supported values: {list(valid)}"
        )


def validate_config(
    *,
    frameworks: list[str],
    scenarios: list[str],
    n_points: list[int],
    n_runs: int,
    timeout_seconds: float,
    rtol: float,
    atol: float,
    pyhs3_workspace_dir: Path,
) -> Path:
    validate_choices(frameworks, SUPPORTED_FRAMEWORKS, "frameworks")
    validate_choices(scenarios, SUPPORTED_SCENARIOS, "scenarios")
    if len(frameworks) == 0:
        raise ValueError("At least one framework must be selected")
    if len(scenarios) == 0:
        raise ValueError("At least one scenario must be selected")
    if len(n_points) == 0:
        raise ValueError("At least one --n-points value must be provided")
    for value in n_points:
        validate_positive_int(value, "--n-points")
    validate_positive_int(n_runs, "--n-runs")
    validate_positive_float(timeout_seconds, "--timeout-seconds")
    validate_probability_tolerance(rtol, "--rtol")
    validate_probability_tolerance(atol, "--atol")

    workspace_dir = pyhs3_workspace_dir
    if not workspace_dir.exists() and workspace_dir == DEFAULT_PYHS3_WORKSPACE_DIR:
        if FALLBACK_PYHS3_WORKSPACE_DIR.exists():
            workspace_dir = FALLBACK_PYHS3_WORKSPACE_DIR
    if "pyhs3" in frameworks and not workspace_dir.exists():
        raise FileNotFoundError(
            f"PyHS3 workspace directory does not exist: {workspace_dir}"
        )
    return workspace_dir


def make_input_grid(scenario: str, n_points: int) -> np.ndarray:
    validate_positive_int(n_points, "n_points")
    if scenario == "normal":
        return np.linspace(-5.0, 5.0, n_points, dtype=float)
    if scenario == "poisson":
        base = np.arange(0, 30, dtype=int)
        repeats = math.ceil(n_points / len(base))
        return np.tile(base, repeats)[:n_points]
    raise ValueError(f"Unknown scenario: {scenario}")


def normal_reference(x: np.ndarray) -> np.ndarray:
    return 1.0 / math.sqrt(2.0 * math.pi) * np.exp(-0.5 * x**2)


def poisson_reference(x: np.ndarray) -> np.ndarray:
    mean = 5.0
    return np.array(
        [
            math.exp(k * math.log(mean) - mean - math.lgamma(k + 1.0))
            for k in x.astype(int)
        ],
        dtype=float,
    )


def reference_values(scenario: str, x: np.ndarray) -> np.ndarray:
    if scenario == "normal":
        return normal_reference(x)
    if scenario == "poisson":
        return poisson_reference(x)
    raise ValueError(f"Unknown scenario: {scenario}")


class PyHS3Evaluator:
    """PyHS3 evaluator.

    PyHS3 currently does not expose the same native vectorized scalar-PDF API as
    libraries such as numba-stats. This evaluator therefore represents PyHS3's
    current available evaluation path for the same input grid.
    """

    is_native_vectorized = False

    def __init__(self, scenario: str, pyhs3_workspace_dir: Path) -> None:
        from pyhs3.workspace import Workspace

        workspace_path = pyhs3_workspace_dir / f"{scenario}_pdf_workspace.json"
        if not workspace_path.exists():
            raise FileNotFoundError(
                f"PyHS3 workspace not found for scenario '{scenario}': {workspace_path}"
            )
        workspace = Workspace.load(workspace_path)
        self.model = workspace.model("analysis", progress=False, mode="FAST_RUN")
        self.base_parameters = {
            name: np.asarray(value, dtype=float)
            for name, value in {**self.model.data, **self.model.free_params}.items()
        }

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        # PyHS3 currently does not support the same native vectorized scalar-PDF
        # path as numba-stats or zfit. Passing the full x array into model.pdf()
        # can trigger PyTensor/Numba compilation failures for scalar workspaces,
        # so this benchmark deliberately measures the current point-wise PyHS3
        # evaluation path.
        x_values = np.asarray(x, dtype=float).reshape(-1)
        values = np.empty(x_values.size, dtype=float)

        for index, x_value in enumerate(x_values):
            parameters = dict(self.base_parameters)
            parameters["x"] = np.asarray(float(x_value), dtype=float)
            values[index] = float(
                np.asarray(self.model.pdf("pdf", **parameters)).squeeze()
            )

        return values


class NumbaStatsEvaluator:
    is_native_vectorized = True

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        if scenario == "normal":
            from numba_stats import norm

            self.distribution = norm
        elif scenario == "poisson":
            from numba_stats import poisson

            self.distribution = poisson
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if self.scenario == "normal":
            return np.asarray(self.distribution.pdf(x, 0.0, 1.0), dtype=float)
        if self.scenario == "poisson":
            return np.asarray(self.distribution.pmf(x.astype(int), 5.0), dtype=float)
        raise ValueError(f"Unknown scenario: {self.scenario}")


class RootEvaluator:
    is_native_vectorized = False

    def __init__(self, scenario: str, x: np.ndarray) -> None:
        import ROOT

        self.scenario = scenario
        self.xvar = ROOT.RooRealVar("x", "x", float(np.min(x)), float(np.max(x)))
        self._normalization_factor = 1.0

        if scenario == "normal":
            self.mu = ROOT.RooRealVar("mu", "mu", 0.0)
            self.sigma = ROOT.RooRealVar("sigma", "sigma", 1.0, 1e-6, 100.0)
            self.pdf = ROOT.RooGaussian(
                "normal", "normal", self.xvar, self.mu, self.sigma
            )
            self._normalization_factor = 1.0 / math.sqrt(2.0 * math.pi)

            self._keepalive = [self.xvar, self.mu, self.sigma, self.pdf]

        elif scenario == "poisson":
            self.mean = ROOT.RooRealVar("mean", "mean", 5.0, 0.0, 1000.0)
            self.pdf = ROOT.RooPoisson("poisson", "poisson", self.xvar, self.mean)

            self._keepalive = [self.xvar, self.mean, self.pdf]

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        values = np.empty(len(x), dtype=float)

        for index, value in enumerate(x):
            self.xvar.setVal(float(value))
            values[index] = float(self.pdf.getVal()) * self._normalization_factor

        return values


class ZfitEvaluator:
    is_native_vectorized = True

    def __init__(self, scenario: str, x: np.ndarray) -> None:
        import zfit

        self.zfit = zfit
        self.scenario = scenario
        self.obs = zfit.Space("x", limits=(float(np.min(x)), float(np.max(x))))
        if scenario == "normal":
            self.pdf = zfit.pdf.Gauss(obs=self.obs, mu=0.0, sigma=1.0)
        elif scenario == "poisson":
            raise NotImplementedError(
                "zfit Poisson vectorized PDF is not included in this benchmark yet."
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        data = self.zfit.Data.from_numpy(obs=self.obs, array=np.asarray(x, dtype=float))
        return np.asarray(self.pdf.pdf(data).numpy(), dtype=float)


def create_evaluator(
    framework: str, scenario: str, x: np.ndarray, pyhs3_workspace_dir: Path
) -> Any:
    if framework == "pyhs3":
        return PyHS3Evaluator(
            scenario=scenario, pyhs3_workspace_dir=pyhs3_workspace_dir
        )
    if framework == "numba_stats":
        return NumbaStatsEvaluator(scenario=scenario)
    if framework == "root":
        return RootEvaluator(scenario=scenario, x=x)
    if framework == "zfit":
        return ZfitEvaluator(scenario=scenario, x=x)
    raise ValueError(f"Unknown framework: {framework}")


def validate_values(values: np.ndarray, expected_size: int, context: str) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size != expected_size:
        raise ValueError(
            f"{context} returned {array.size} values, expected {expected_size}"
        )
    if not np.all(np.isfinite(array)):
        n_bad = int(np.size(array) - np.isfinite(array).sum())
        raise ValueError(f"{context} returned {n_bad} non-finite values")
    return array


def compute_agreement(
    observed: np.ndarray, reference: np.ndarray, rtol: float, atol: float
) -> dict[str, Any]:
    observed = validate_values(observed, reference.size, "observed values")
    reference = validate_values(reference, reference.size, "reference values")
    abs_diff = np.abs(observed - reference)
    rel_diff = abs_diff / np.maximum(np.abs(reference), 1e-300)
    max_abs_diff = float(np.max(abs_diff))
    max_rel_diff = float(np.max(rel_diff))
    return {
        "n_values": int(observed.size),
        "n_finite_values": int(np.isfinite(observed).sum()),
        "all_values_finite": bool(np.all(np.isfinite(observed))),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": float(np.mean(abs_diff)),
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": float(np.mean(rel_diff)),
        "allclose_passed": bool(np.allclose(observed, reference, rtol=rtol, atol=atol)),
        "validation_status": "success"
        if np.allclose(observed, reference, rtol=rtol, atol=atol)
        else "failed",
    }


def summarize_timings(timings: list[float]) -> dict[str, float]:
    if not timings:
        raise ValueError("Cannot summarize empty timing list")
    invalid = [value for value in timings if not math.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError(f"Timing samples must be positive finite values: {invalid}")
    return {
        "mean_seconds": float(np.mean(timings)),
        "median_seconds": float(np.median(timings)),
        "std_seconds": float(np.std(timings, ddof=1)) if len(timings) > 1 else 0.0,
    }


def run_single_framework_benchmark(spec: BenchmarkSpec) -> dict[str, Any]:
    x = make_input_grid(spec.scenario, spec.n_points)
    reference = reference_values(spec.scenario, x)

    gc.collect()
    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    setup_start = time.perf_counter()
    evaluator = create_evaluator(
        spec.framework, spec.scenario, x, spec.pyhs3_workspace_dir
    )
    setup_seconds = time.perf_counter() - setup_start

    cold_start = time.perf_counter()
    first_values = validate_values(
        evaluator.evaluate(x), spec.n_points, f"{spec.framework} cold evaluation"
    )
    cold_seconds = time.perf_counter() - cold_start

    agreement = compute_agreement(first_values, reference, spec.rtol, spec.atol)

    warm_timings: list[float] = []
    for _ in range(spec.n_runs):
        start = time.perf_counter()
        values = validate_values(
            evaluator.evaluate(x), spec.n_points, f"{spec.framework} warm evaluation"
        )
        warm_timings.append(time.perf_counter() - start)
    warm_summary = summarize_timings(warm_timings)

    repeated_agreement = compute_agreement(values, reference, spec.rtol, spec.atol)
    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    warm_mean = warm_summary["mean_seconds"]
    throughput = spec.n_points / warm_mean if warm_mean > 0 else float("inf")
    seconds_per_value = warm_mean / spec.n_points if spec.n_points > 0 else float("inf")

    status = (
        "success"
        if agreement["validation_status"] == "success"
        and repeated_agreement["validation_status"] == "success"
        else "failed"
    )
    result: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "framework": spec.framework,
        "framework_label": _framework_label(spec.framework),
        "scenario": spec.scenario,
        "scenario_label": _scenario_label(spec.scenario),
        "n_points": int(spec.n_points),
        "n_runs": int(spec.n_runs),
        "native_vectorized": bool(getattr(evaluator, "is_native_vectorized", False)),
        "status": status,
        "validation_status": agreement["validation_status"],
        "setup_time_seconds": setup_seconds,
        "cold_vectorized_eval_time_seconds": cold_seconds,
        "warm_vectorized_eval_time_seconds_mean": warm_summary["mean_seconds"],
        "warm_vectorized_eval_time_seconds_median": warm_summary["median_seconds"],
        "warm_vectorized_eval_time_seconds_std": warm_summary["std_seconds"],
        "throughput_values_per_second": throughput,
        "time_per_value_seconds": seconds_per_value,
        "current_rss_before_mb": current_rss_before_mb,
        "current_rss_after_mb": current_rss_after_mb,
        "current_rss_delta_mb": current_rss_after_mb - current_rss_before_mb,
        "peak_rss_before_mb": peak_rss_before_mb,
        "peak_rss_after_mb": peak_rss_after_mb,
        "peak_rss_delta_mb": peak_rss_after_mb - peak_rss_before_mb,
        "agreement": agreement,
        "repeat_agreement": repeated_agreement,
        **{
            key: agreement[key]
            for key in [
                "max_abs_diff",
                "mean_abs_diff",
                "max_rel_diff",
                "mean_rel_diff",
                "allclose_passed",
            ]
        },
    }
    if status != "success":
        result["error_type"] = "ValidationFailure"
        result["error_message"] = (
            "PDF value agreement failed "
            f"(max abs diff={agreement['max_abs_diff']:.3e}, "
            f"max rel diff={agreement['max_rel_diff']:.3e})"
        )
    del values, first_values
    gc.collect()
    return result


def failed_result(
    spec: BenchmarkSpec, error: BaseException, status: str = "failed"
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "framework": spec.framework,
        "framework_label": _framework_label(spec.framework),
        "scenario": spec.scenario,
        "scenario_label": _scenario_label(spec.scenario),
        "n_points": int(spec.n_points),
        "n_runs": int(spec.n_runs),
        "status": status,
        "validation_status": "failed",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    spec = BenchmarkSpec(**payload)
    try:
        return run_single_framework_benchmark(spec)
    except Exception as error:
        return failed_result(spec, error)


def run_with_timeout(spec: BenchmarkSpec, timeout_seconds: float) -> dict[str, Any]:
    ctx = get_context("spawn")
    payload = {
        "framework": spec.framework,
        "scenario": spec.scenario,
        "n_points": spec.n_points,
        "n_runs": spec.n_runs,
        "rtol": spec.rtol,
        "atol": spec.atol,
        "pyhs3_workspace_dir": spec.pyhs3_workspace_dir,
    }
    with ctx.Pool(processes=1) as pool:
        async_result = pool.apply_async(run_worker, args=(payload,))
        try:
            return async_result.get(timeout=timeout_seconds)
        except TimeoutError:
            pool.terminate()
            pool.join()
            return {
                "benchmark": BENCHMARK_NAME,
                "framework": spec.framework,
                "framework_label": _framework_label(spec.framework),
                "scenario": spec.scenario,
                "scenario_label": _scenario_label(spec.scenario),
                "n_points": int(spec.n_points),
                "n_runs": int(spec.n_runs),
                "status": "timeout",
                "validation_status": "failed",
                "timeout_seconds": timeout_seconds,
                "error_type": "TimeoutError",
                "error_message": f"Benchmark exceeded timeout of {timeout_seconds:.1f} s",
            }


def print_result(result: dict[str, Any]) -> None:
    print()
    print("-" * 72)
    print(
        f"{result['scenario_label']} / {result['framework_label']} / n={result['n_points']}"
    )
    print("-" * 72)
    print(f"status:                  {result['status']}")
    print(f"validation:              {result.get('validation_status', 'unknown')}")
    if result["status"] != "success":
        print(
            f"error:                   {result.get('error_type')}: {result.get('error_message')}"
        )
        return
    mode = (
        "native vectorized" if result["native_vectorized"] else "point-wise/current API"
    )
    print(f"evaluation mode:         {mode}")
    print(f"setup:                   {result['setup_time_seconds'] * 1e3:.3f} ms")
    print(
        f"cold evaluation:         {result['cold_vectorized_eval_time_seconds'] * 1e3:.3f} ms"
    )
    print(
        f"warm evaluation mean:    {result['warm_vectorized_eval_time_seconds_mean'] * 1e3:.3f} ms"
    )
    print(
        f"warm evaluation median:  {result['warm_vectorized_eval_time_seconds_median'] * 1e3:.3f} ms"
    )
    print(f"time per value:          {result['time_per_value_seconds'] * 1e9:.3f} ns")
    print(
        f"throughput:              {result['throughput_values_per_second']:.3e} values/s"
    )
    print(f"current RSS delta:       {result['current_rss_delta_mb']:.3f} MB")
    print(f"peak RSS delta:          {result['peak_rss_delta_mb']:.3f} MB")
    print(f"max abs diff:            {result['max_abs_diff']:.6e}")
    print(f"max rel diff:            {result['max_rel_diff']:.6e}")


def successful_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if result.get("status") == "success"]


def _values_for(
    results: list[dict[str, Any]], scenario: str, framework: str, key: str
) -> tuple[list[int], list[float]]:
    selected = [
        result
        for result in results
        if result.get("status") == "success"
        and result["scenario"] == scenario
        and result["framework"] == framework
    ]
    selected.sort(key=lambda item: item["n_points"])
    return [int(item["n_points"]) for item in selected], [
        float(item[key]) for item in selected
    ]


def make_throughput_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = successful_results(results)
    if not successful:
        raise ValueError("No successful results available for throughput plot")
    scenarios = [
        scenario
        for scenario in SUPPORTED_SCENARIOS
        if any(r["scenario"] == scenario for r in successful)
    ]
    fig, axes = plt.subplots(
        len(scenarios), 1, figsize=(12.5, 4.9 * len(scenarios)), sharex=True
    )
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios, strict=True):
        for framework in SUPPORTED_FRAMEWORKS:
            xs, ys = _values_for(
                successful, scenario, framework, "throughput_values_per_second"
            )
            if not xs:
                continue
            style = FRAMEWORK_STYLE[framework]
            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=7,
                label=style["label"],
                color=style["color"],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Throughput [values/s]")
        ax.set_title(
            f"{_scenario_label(scenario)} PDF",
            loc="left",
            weight="bold",
            fontsize=15,
            pad=8,
        )
        ax.grid(True, which="both", alpha=0.28)
    axes[-1].set_xlabel("Number of evaluated points")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle(
        "PDF evaluation throughput scaling",
        x=0.06,
        ha="left",
        weight="bold",
        fontsize=22,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.80,
        top=0.83 if len(scenarios) == 1 else 0.91,
        bottom=0.16,
        hspace=0.36,
    )
    _save_figure(fig, output_path)


def make_time_per_value_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = successful_results(results)
    if not successful:
        raise ValueError("No successful results available for time-per-value plot")
    scenarios = [
        scenario
        for scenario in SUPPORTED_SCENARIOS
        if any(r["scenario"] == scenario for r in successful)
    ]
    fig, axes = plt.subplots(
        len(scenarios), 1, figsize=(12.5, 4.9 * len(scenarios)), sharex=True
    )
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios, strict=True):
        for framework in SUPPORTED_FRAMEWORKS:
            xs, ys = _values_for(
                successful, scenario, framework, "time_per_value_seconds"
            )
            if not xs:
                continue
            ns_values = [value * 1e9 for value in ys]
            style = FRAMEWORK_STYLE[framework]
            ax.plot(
                xs,
                ns_values,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=7,
                label=style["label"],
                color=style["color"],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("Time/value [ns]")
        ax.set_title(
            f"{_scenario_label(scenario)} PDF",
            loc="left",
            weight="bold",
            fontsize=15,
            pad=8,
        )
        ax.grid(True, which="both", alpha=0.28)
    axes[-1].set_xlabel("Number of evaluated points")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle(
        "PDF evaluation cost per value",
        x=0.06,
        ha="left",
        weight="bold",
        fontsize=22,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.80,
        top=0.83 if len(scenarios) == 1 else 0.91,
        bottom=0.16,
        hspace=0.36,
    )
    _save_figure(fig, output_path)


def make_agreement_plot(
    results: list[dict[str, Any]], output_path: Path, tolerance: float
) -> None:
    _apply_cern_style()
    successful = successful_results(results)
    if not successful:
        raise ValueError("No successful results available for agreement plot")
    selected = sorted(
        successful, key=lambda r: (r["scenario"], r["n_points"], r["framework"])
    )
    labels = [
        f"{_scenario_label(r['scenario'])}\n{_framework_label(r['framework'])}\n{r['n_points']}"
        for r in selected
    ]
    raw_values = [float(r["max_abs_diff"]) for r in selected]
    positive_values = [
        value for value in raw_values if value > 0 and math.isfinite(value)
    ]
    floor = (
        min(positive_values) / 10.0 if positive_values else max(tolerance * 1e-4, 1e-16)
    )
    floor = max(floor, max(tolerance * 1e-8, 1e-18))
    plot_values = [value if value > 0 else floor for value in raw_values]
    colors = [FRAMEWORK_STYLE[r["framework"]]["color"] for r in selected]
    fig, ax = plt.subplots(figsize=(max(12.5, len(labels) * 0.58), 6.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, plot_values, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(
        tolerance,
        linestyle="--",
        color="black",
        linewidth=1.7,
        label=f"tolerance = {tolerance:g}",
    )
    ax.set_yscale("log")
    y_min = min(plot_values + [tolerance]) / 4.0
    y_max = max(plot_values + [tolerance]) * 4.0
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("max |PDF - reference|")
    ax.set_title("PDF numerical agreement", loc="left", weight="bold", pad=10)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    label_values = raw_values
    _add_bar_labels(
        ax,
        bars,
        label_values,
        lambda v: "0" if v == 0 else f"{v:.1e}",
        y_offset_points=4,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.10, right=0.82, bottom=0.28, top=0.88)
    _save_figure(fig, output_path)


def make_memory_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = successful_results(results)
    if not successful:
        raise ValueError("No successful results available for memory plot")
    selected = sorted(
        successful, key=lambda r: (r["scenario"], r["n_points"], r["framework"])
    )
    labels = [
        f"{_scenario_label(r['scenario'])}\n{_framework_label(r['framework'])}\n{r['n_points']}"
        for r in selected
    ]
    raw_current = [float(r["current_rss_delta_mb"]) for r in selected]
    raw_peak = [float(r["peak_rss_delta_mb"]) for r in selected]
    current = [max(value, 1e-3) for value in raw_current]
    peak = [max(value, 1e-3) for value in raw_peak]
    x = np.arange(len(selected))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(12.5, len(labels) * 0.62), 6.6))
    bars_current = ax.bar(
        x - width / 2,
        current,
        width=width,
        label="Current RSS Δ",
        color="#1764ab",
        edgecolor="black",
        linewidth=0.7,
    )
    bars_peak = ax.bar(
        x + width / 2,
        peak,
        width=width,
        label="Peak RSS Δ",
        color="#9ecae1",
        edgecolor="black",
        linewidth=0.7,
        hatch="//",
    )
    ax.set_yscale("log")
    ax.set_ylim(min(current + peak) / 2.5, max(current + peak) * 1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Memory delta [MB]")
    ax.set_title(
        "Memory footprint during PDF evaluation", loc="left", weight="bold", pad=10
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _add_bar_labels(
        ax,
        bars_current,
        raw_current,
        _format_memory_mb,
        x_offset_points=-4,
        y_offset_points=4,
        fontsize=7,
    )
    _add_bar_labels(
        ax,
        bars_peak,
        raw_peak,
        _format_memory_mb,
        x_offset_points=4,
        y_offset_points=4,
        fontsize=7,
    )
    fig.subplots_adjust(left=0.10, right=0.82, bottom=0.28, top=0.88)
    _save_figure(fig, output_path)


def make_summary_table(results: list[dict[str, Any]], output_path: Path) -> None:
    _apply_cern_style()
    successful = successful_results(results)
    if not successful:
        raise ValueError("No successful results available for summary table")
    selected = sorted(
        successful, key=lambda r: (r["scenario"], r["n_points"], r["framework"])
    )
    headers = [
        "Scenario",
        "Framework",
        "Points",
        "Mode",
        "Warm [ms]",
        "ns/value",
        "Throughput",
        "RSS Δ [MB]",
        "max diff",
    ]
    rows = []
    for r in selected:
        rows.append(
            [
                _scenario_label(r["scenario"]),
                _framework_label(r["framework"]),
                str(r["n_points"]),
                "native" if r["native_vectorized"] else "current API",
                f"{r['warm_vectorized_eval_time_seconds_mean'] * 1e3:.3g}",
                f"{r['time_per_value_seconds'] * 1e9:.3g}",
                f"{r['throughput_values_per_second']:.2e}",
                f"{r['current_rss_delta_mb']:.2f}",
                f"{r['max_abs_diff']:.1e}",
            ]
        )
    fig_height = max(4.8, 0.34 * len(rows) + 2.5)
    fig, ax = plt.subplots(figsize=(15.5, fig_height))
    ax.axis("off")
    ax.set_title(
        "Cross-framework PDF vectorization summary",
        loc="left",
        weight="bold",
        fontsize=24,
        pad=16,
    )
    ax.text(
        0.0,
        0.88,
        "PyHS3 is evaluated through its current non-native-vectorized API; native vectorized frameworks are marked separately.",
        transform=ax.transAxes,
        fontsize=12,
    )
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.02, 1.0, 0.70],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.30)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#bdbdbd")
        if row == 0:
            cell.set_facecolor("#262626")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.04)
    _save_figure(fig, output_path)


def make_plots(results: list[dict[str, Any]], plot_dir: Path, atol: float) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    successful = successful_results(results)
    if len(successful) < 2:
        print("Skipping plots: at least two successful results are needed.")
        return
    make_throughput_plot(
        results, plot_dir / "cross_vectorized_pdf_throughput_scaling.png"
    )
    make_time_per_value_plot(
        results, plot_dir / "cross_vectorized_pdf_time_per_value.png"
    )
    make_agreement_plot(
        results,
        plot_dir / "cross_vectorized_pdf_numerical_agreement.png",
        tolerance=atol,
    )
    make_memory_plot(results, plot_dir / "cross_vectorized_pdf_memory.png")
    make_summary_table(results, plot_dir / "cross_vectorized_pdf_summary_table.png")


def build_specs(
    *,
    frameworks: list[str],
    scenarios: list[str],
    n_points: list[int],
    n_runs: int,
    rtol: float,
    atol: float,
    pyhs3_workspace_dir: Path,
) -> list[BenchmarkSpec]:
    return [
        BenchmarkSpec(
            framework, scenario, n_point, n_runs, rtol, atol, pyhs3_workspace_dir
        )
        for scenario in scenarios
        for framework in frameworks
        for n_point in n_points
    ]


def build_output(
    results: list[dict[str, Any]], args: argparse.Namespace, workspace_dir: Path
) -> dict[str, Any]:
    n_successful = sum(1 for result in results if result.get("status") == "success")
    return {
        "benchmark": BENCHMARK_NAME,
        "description": "Vectorization study: PyHS3 current PDF evaluation path vs frameworks with native vectorized APIs where available.",
        "status": "success" if n_successful == len(results) else "failed",
        "n_results": len(results),
        "n_successful": n_successful,
        "n_failed": len(results) - n_successful,
        "frameworks": args.frameworks,
        "scenarios": args.scenarios,
        "n_points": args.n_points,
        "n_runs": args.n_runs,
        "rtol": args.rtol,
        "atol": args.atol,
        "timeout_seconds": args.timeout_seconds,
        "pyhs3_workspace_dir": str(workspace_dir),
        "results": results,
    }


def print_summary(output_data: dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print(BENCHMARK_TITLE)
    print("=" * 80)
    print(f"Status:      {output_data['status']}")
    print(f"Successful:  {output_data['n_successful']} / {output_data['n_results']}")
    print("Note: PyHS3 is evaluated through its current non-native-vectorized API.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=BENCHMARK_TITLE)
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=DEFAULT_FRAMEWORKS,
        choices=SUPPORTED_FRAMEWORKS,
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=DEFAULT_SCENARIOS, choices=SUPPORTED_SCENARIOS
    )
    parser.add_argument("--n-points", nargs="+", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument(
        "--pyhs3-workspace-dir", type=Path, default=DEFAULT_PYHS3_WORKSPACE_DIR
    )
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument(
        "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def run(
    *,
    frameworks: list[str],
    scenarios: list[str],
    n_points: list[int],
    n_runs: int,
    pyhs3_workspace_dir: Path,
    rtol: float,
    atol: float,
    timeout_seconds: float,
    output_dir: Path,
    output_name: str,
    plot: bool,
    plot_dir: Path,
    fail_fast: bool = False,
) -> dict[str, Any]:
    class ArgsObj:
        pass

    args = ArgsObj()
    args.frameworks = frameworks
    args.scenarios = scenarios
    args.n_points = n_points
    args.n_runs = n_runs
    args.rtol = rtol
    args.atol = atol
    args.timeout_seconds = timeout_seconds

    workspace_dir = validate_config(
        frameworks=frameworks,
        scenarios=scenarios,
        n_points=n_points,
        n_runs=n_runs,
        timeout_seconds=timeout_seconds,
        rtol=rtol,
        atol=atol,
        pyhs3_workspace_dir=pyhs3_workspace_dir,
    )
    specs = build_specs(
        frameworks=frameworks,
        scenarios=scenarios,
        n_points=n_points,
        n_runs=n_runs,
        rtol=rtol,
        atol=atol,
        pyhs3_workspace_dir=workspace_dir,
    )
    results: list[dict[str, Any]] = []
    for spec in specs:
        print(
            f"Running scenario={spec.scenario}, framework={spec.framework}, n_points={spec.n_points}, n_runs={spec.n_runs}",
            flush=True,
        )
        result = run_with_timeout(spec, timeout_seconds=timeout_seconds)
        results.append(result)
        print_result(result)
        if fail_fast and result.get("status") != "success":
            break

    output_data = build_output(results, args, workspace_dir)
    output_path = output_dir / output_name
    save_json(output_data, output_path)
    print_summary(output_data)
    print(f"Saved result to {output_path}")
    if plot:
        make_plots(results, plot_dir, atol=atol)
        print(f"Saved plots to {plot_dir}")
    return output_data


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run(
        frameworks=args.frameworks,
        scenarios=args.scenarios,
        n_points=args.n_points,
        n_runs=args.n_runs,
        pyhs3_workspace_dir=args.pyhs3_workspace_dir,
        rtol=args.rtol,
        atol=args.atol,
        timeout_seconds=args.timeout_seconds,
        output_dir=args.output_dir,
        output_name=args.output_name,
        plot=args.plot,
        plot_dir=args.plot_dir,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
