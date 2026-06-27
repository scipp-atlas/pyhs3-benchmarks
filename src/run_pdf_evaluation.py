from __future__ import annotations

import argparse
import gc
import math
import time
import tracemalloc
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    DEFAULT_MODE,
    DEFAULT_TARGET,
    DEFAULT_WORKSPACE,
    PLOTS_DIR,
    RESULTS_DIR,
)
from .utils import (
    create_model,
    get_current_rss_mb,
    get_peak_rss_mb,
    load_workspace,
    make_bar_plot,
    save_json,
    should_plot_metric,
)

BENCHMARK_NAME = "pdf_evaluation"

DEFAULT_DISTRIBUTION = "sig_ch0"
DEFAULT_N_EVALUATIONS = [
    1,
    10,
    100,
    1000,
    10000,
]

DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "pdf_evaluation_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME


def build_parameter_inputs(model) -> dict[str, Any]:
    """
    Build keyword inputs for model.pdf(...) from the model data and free
    parameters.
    """

    return {
        key: np.asarray(value, dtype=float)
        for key, value in {
            **model.data,
            **model.free_params,
        }.items()
    }


def extract_scalar_output(result) -> float:
    """
    Convert a model.pdf(...) result into a scalar float for validation.
    """

    array = np.asarray(result)

    if array.size == 0:
        raise ValueError("PDF result is empty")

    return float(array.reshape(-1)[0])


def measure_cold_start_pdf_call(
    model,
    distribution: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Measure the first model.pdf(...) call separately.

    This captures cold-start cost, such as lazy compilation or cache setup,
    separately from repeated warm evaluations.
    """

    start = time.perf_counter()
    result = model.pdf(distribution, **parameters)
    end = time.perf_counter()

    output = extract_scalar_output(result)

    return {
        "cold_start_time_seconds": end - start,
        "cold_start_output": output,
    }


def evaluate_pdf(
    model,
    distribution: str,
    parameters: dict[str, Any],
    n_evaluations: int,
) -> list[float]:
    """
    Evaluate model.pdf(...) repeatedly and store scalar outputs for validation.
    """

    outputs = []

    for _ in range(n_evaluations):
        result = model.pdf(distribution, **parameters)
        outputs.append(extract_scalar_output(result))

    return outputs


def validate_pdf_outputs(outputs: list[float]) -> dict[str, Any]:
    """
    Validate PDF outputs and compute simple stability statistics.
    """

    if len(outputs) == 0:
        raise ValueError("No PDF outputs were produced")

    all_finite = all(math.isfinite(output) for output in outputs)
    reference_value = outputs[0]

    max_absolute_deviation = max(
        abs(output - reference_value)
        for output in outputs
    )

    return {
        "n_outputs": len(outputs),
        "all_outputs_finite": all_finite,
        "reference_output": reference_value,
        "max_absolute_deviation": max_absolute_deviation,
        "outputs_stable": max_absolute_deviation < 1e-12,
    }


def measure_pdf_evaluation_timing(
    model,
    distribution: str,
    parameters: dict[str, Any],
    n_evaluations: int,
) -> dict[str, Any]:
    """
    Measure repeated warm model.pdf(...) evaluation wall time.

    The cold-start call is measured separately before this function.
    """

    result = model.pdf(distribution, **parameters)
    first_output = extract_scalar_output(result)

    start = time.perf_counter()

    last_output = first_output
    for _ in range(n_evaluations):
        result = model.pdf(distribution, **parameters)
        last_output = extract_scalar_output(result)

    end = time.perf_counter()

    total_runtime_seconds = end - start
    average_runtime_seconds = total_runtime_seconds / n_evaluations
    throughput_evaluations_per_second = (
        n_evaluations / total_runtime_seconds
        if total_runtime_seconds > 0
        else float("inf")
    )

    return {
        "n_evaluations": n_evaluations,
        "total_runtime_seconds": total_runtime_seconds,
        "average_runtime_seconds_per_evaluation": average_runtime_seconds,
        "throughput_evaluations_per_second": throughput_evaluations_per_second,
        "first_timing_output": first_output,
        "last_timing_output": last_output,
    }


def measure_pdf_evaluation_memory(
    model,
    distribution: str,
    parameters: dict[str, Any],
    n_evaluations: int,
) -> dict[str, float | int]:
    """
    Measure memory change for repeated warm model.pdf(...) evaluation.
    """

    result = model.pdf(distribution, **parameters)
    extract_scalar_output(result)

    gc.collect()

    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    for _ in range(n_evaluations):
        result = model.pdf(distribution, **parameters)
        extract_scalar_output(result)

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    return {
        "memory_n_evaluations": n_evaluations,
        "current_rss_before_mb": current_rss_before_mb,
        "current_rss_after_mb": current_rss_after_mb,
        "current_rss_delta_mb": current_rss_after_mb - current_rss_before_mb,
        "peak_rss_before_mb": peak_rss_before_mb,
        "peak_rss_after_mb": peak_rss_after_mb,
        "peak_rss_delta_mb": peak_rss_after_mb - peak_rss_before_mb,
    }


def run_single_benchmark(
    workspace_path: Path,
    target: str,
    mode: str,
    distribution: str,
    n_evaluations: int,
) -> dict[str, Any]:
    """
    Run PDF evaluation benchmark for one workspace, target, mode, distribution,
    and evaluation count.

    Workspace loading and model creation are setup only.
    The measured operation is model.pdf(...).
    """

    workspace = load_workspace(workspace_path)

    model = create_model(
        workspace=workspace,
        target=target,
        mode=mode,
    )

    available_distributions = list(model.distributions.keys())

    print(
        f"Available distributions for {workspace_path.name}: "
        f"{available_distributions}",
        flush=True,
    )

    if distribution not in model.distributions:
        raise KeyError(
            f"Distribution '{distribution}' not found. "
            f"Available distributions: {available_distributions}"
        )

    parameters = build_parameter_inputs(model)

    cold_start_summary = measure_cold_start_pdf_call(
        model=model,
        distribution=distribution,
        parameters=parameters,
    )

    validation_outputs = evaluate_pdf(
        model=model,
        distribution=distribution,
        parameters=parameters,
        n_evaluations=n_evaluations,
    )
    validation_summary = validate_pdf_outputs(validation_outputs)

    del validation_outputs
    gc.collect()

    memory_summary = measure_pdf_evaluation_memory(
        model=model,
        distribution=distribution,
        parameters=parameters,
        n_evaluations=n_evaluations,
    )

    timing_summary = measure_pdf_evaluation_timing(
        model=model,
        distribution=distribution,
        parameters=parameters,
        n_evaluations=n_evaluations,
    )

    del model
    del workspace
    gc.collect()

    return {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "target": target,
        "mode": mode,
        "distribution": distribution,
        "n_evaluations": n_evaluations,
        "available_distributions": available_distributions,
        **cold_start_summary,
        **timing_summary,
        **memory_summary,
        "status": "success",
        **validation_summary,
    }


def print_result(result: dict[str, Any]) -> None:
    """
    Print the benchmark result in a human-readable format.
    """

    print()
    print("=" * 72)
    print("PDF evaluation benchmark")
    print("=" * 72)
    print(f"Workspace:     {result['workspace']}")
    print(f"Target:        {result['target']}")
    print(f"Mode:          {result['mode']}")
    print(f"Distribution:  {result['distribution']}")
    print(f"Evaluations:   {result['n_evaluations']}")
    print(f"Status:        {result['status']}")

    print()
    print("Cold start")
    print(f"  first call:       {result['cold_start_time_seconds'] * 1000:.6f} ms")
    print(f"  first output:     {result['cold_start_output']}")

    print()
    print("Warm timing")
    print(f"  total runtime:    {result['total_runtime_seconds'] * 1000:.3f} ms")
    print(
        "  average / eval:   "
        f"{result['average_runtime_seconds_per_evaluation'] * 1000:.6f} ms"
    )
    print(
        "  throughput:       "
        f"{result['throughput_evaluations_per_second']:.3f} eval/s"
    )

    print()
    print("Memory")
    print(f"  memory evaluations: {result['memory_n_evaluations']}")
    print(f"  current RSS delta:  {result['current_rss_delta_mb']:.3f} MB")
    print(f"  peak RSS delta:     {result['peak_rss_delta_mb']:.3f} MB")

    print()
    print("Validation")
    print(f"  outputs:            {result['n_outputs']}")
    print(f"  reference output:   {result['reference_output']}")
    print(f"  finite outputs:     {result['all_outputs_finite']}")
    print(f"  max abs deviation:  {result['max_absolute_deviation']}")
    print(f"  stable outputs:     {result['outputs_stable']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark repeated pyHS3 model.pdf(...) evaluation."
    )

    parser.add_argument(
        "--workspaces",
        nargs="+",
        type=Path,
        default=[DEFAULT_WORKSPACE],
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[DEFAULT_TARGET],
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[DEFAULT_MODE],
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        default=[DEFAULT_DISTRIBUTION],
    )
    parser.add_argument(
        "--n-evaluations",
        nargs="+",
        type=int,
        default=DEFAULT_N_EVALUATIONS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
    )

    return parser.parse_args()


def make_plots(results: list[dict[str, Any]], plot_dir: Path) -> None:
    """
    Create plots for the PDF evaluation benchmark.

    Plot labels are shortened by removing target and mode from the plot-only
    result dictionaries.
    """

    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_results = []
    for result in results:
        plot_result = {
            "workspace": result["workspace"],
            "n_evaluations": result["n_evaluations"],
            "cold_start_time_ms": result["cold_start_time_seconds"] * 1000.0,
            "average_runtime_ms_per_evaluation": (
                result["average_runtime_seconds_per_evaluation"] * 1000.0
            ),
            "throughput_evaluations_per_second": result[
                "throughput_evaluations_per_second"
            ],
            "current_rss_delta_mb": result["current_rss_delta_mb"],
            "peak_rss_delta_mb": result["peak_rss_delta_mb"],
        }
        plot_results.append(plot_result)

    make_bar_plot(
        results=plot_results,
        output_path=plot_dir / "pdf_evaluation_cold_start_time.png",
        title="PDF evaluation cold-start time",
        metric_key="cold_start_time_ms",
        metric_label="Cold-start time [ms]",
    )

    make_bar_plot(
        results=plot_results,
        output_path=plot_dir / "pdf_evaluation_average_time.png",
        title="PDF evaluation average warm wall time",
        metric_key="average_runtime_ms_per_evaluation",
        metric_label="Average warm time per evaluation [ms]",
    )

    make_bar_plot(
        results=plot_results,
        output_path=plot_dir / "pdf_evaluation_throughput.png",
        title="PDF evaluation warm throughput",
        metric_key="throughput_evaluations_per_second",
        metric_label="Throughput [evaluations / s]",
    )

    if should_plot_metric(plot_results, "current_rss_delta_mb"):
        make_bar_plot(
            results=plot_results,
            output_path=plot_dir / "pdf_evaluation_current_rss_delta.png",
            title="PDF evaluation current RSS delta",
            metric_key="current_rss_delta_mb",
            metric_label="Current RSS delta [MB]",
        )

    if should_plot_metric(plot_results, "peak_rss_delta_mb"):
        make_bar_plot(
            results=plot_results,
            output_path=plot_dir / "pdf_evaluation_peak_rss_delta.png",
            title="PDF evaluation peak RSS delta",
            metric_key="peak_rss_delta_mb",
            metric_label="Peak RSS delta [MB]",
        )


def main() -> None:
    args = parse_args()

    for n_evaluations in args.n_evaluations:
        if n_evaluations < 1:
            raise ValueError("--n-evaluations values must be at least 1")

    results = []
    ctx = get_context("spawn")

    for workspace_path in args.workspaces:
        for target in args.targets:
            for mode in args.modes:
                for distribution in args.distributions:
                    for n_evaluations in args.n_evaluations:
                        print(
                            f"Running {workspace_path.name}, "
                            f"target={target}, "
                            f"mode={mode}, "
                            f"distribution={distribution}, "
                            f"n_evaluations={n_evaluations}",
                            flush=True,
                        )

                        with ctx.Pool(processes=1) as pool:
                            result = pool.apply(
                                run_single_benchmark,
                                args=(
                                    workspace_path,
                                    target,
                                    mode,
                                    distribution,
                                    n_evaluations,
                                ),
                            )

                        results.append(result)
                        print_result(result)

    output_data: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "n_results": len(results),
        "results": results,
    }

    output_path = args.output_dir / args.output_name
    save_json(output_data, output_path)

    print()
    print(f"Saved result to {output_path}")

    if args.plot:
        if len(results) < 2:
            print("Skipping plots: at least two result entries are needed.")
        else:
            make_plots(results=results, plot_dir=args.plot_dir)
            print(f"Saved plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
