from __future__ import annotations

import argparse
import gc
import math
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import get_context
from pathlib import Path
from typing import Any

from .utils import (
    build_log_prob,
    build_validation_inputs,
    compile_log_prob,
    get_current_rss_mb,
    get_peak_rss_mb,
    save_json,
    make_bar_plot,
    should_plot_metric,
)

from .config import (
    DEFAULT_MODE,
    DEFAULT_TARGET,
    DEFAULT_WORKSPACE,
    PLOTS_DIR,
    RESULTS_DIR,
)

BENCHMARK_NAME = "compiled_evaluation"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "compiled_evaluation_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME

VALIDATION_N_EVALUATIONS = 3

DEFAULT_N_EVALUATIONS = [
    1,
    10,
    100,
    1000,
    10000,
]


def validate_workspace_path(workspace_path: Path) -> Path:
    """Validate that the workspace path points to an existing JSON file."""

    if not workspace_path.exists():
        raise FileNotFoundError(f"Workspace file does not exist: {workspace_path}")

    if not workspace_path.is_file():
        raise FileNotFoundError(f"Workspace path is not a file: {workspace_path}")

    return workspace_path


def validate_benchmark_config(target: str, mode: str, n_evaluations: int) -> None:
    """Validate benchmark configuration before running expensive work."""

    if not target:
        raise ValueError("target must be a non-empty string")

    if not mode:
        raise ValueError("mode must be a non-empty string")

    if n_evaluations < 1:
        raise ValueError("n_evaluations must be at least 1")


def verify_output_file(output_path: Path) -> None:
    """Verify that save_json created a regular output file."""

    if not output_path.exists():
        raise FileNotFoundError(f"Benchmark output file was not created: {output_path}")

    if not output_path.is_file():
        raise FileNotFoundError(f"Benchmark output path is not a file: {output_path}")


def extract_scalar_output(result) -> float:
    """
    Extract a scalar finite float from compiled log_prob output.

    pyHS3/JAX compiled outputs are expected to be returned as a tuple whose
    first element may be either scalar-like or array-like.
    """

    if not isinstance(result, tuple):
        raise TypeError(
            f"Expected compiled result to be a tuple, got {type(result).__name__}"
        )

    if len(result) == 0:
        raise ValueError("Compiled result tuple is empty")

    array = np.asarray(result[0])

    if array.size == 0:
        raise ValueError("Compiled result array is empty")

    try:
        value = float(array.reshape(-1)[0])
    except (TypeError, ValueError, IndexError) as exc:
        raise TypeError("Could not extract scalar float from compiled result") from exc

    if not math.isfinite(value):
        raise ValueError(f"Compiled result is not finite: {value}")

    return value


def prepare_compiled_graph(
    workspace_path: Path,
    target: str,
    mode: str,
):
    """
    Build and compile a log_prob graph and prepare evaluation inputs.
    """

    model, log_prob = build_log_prob(
        workspace_path=workspace_path,
        target=target,
        mode=mode,
    )

    compiled = compile_log_prob(log_prob)

    validation_inputs = build_validation_inputs(
        model=model,
        compiled=compiled,
    )

    return model, compiled, validation_inputs


def evaluate_compiled_graph(
    compiled,
    validation_inputs,
    n_evaluations: int,
) -> list[float]:
    """
    Evaluate the compiled log_prob graph multiple times and return the outputs.
    """

    outputs = []

    for _ in range(n_evaluations):
        result = compiled(**validation_inputs)
        outputs.append(extract_scalar_output(result))

    return outputs


def validate_evaluation(
    outputs: list[float],
) -> dict[str, Any]:
    """
    Validate the evaluation outputs and compute summary statistics.
    """

    if len(outputs) == 0:
        raise ValueError("No evaluation outputs were produced")

    all_finite = all(
        math.isfinite(output)
        for output in outputs
    )

    if not all_finite:
        raise ValueError("Evaluation outputs contain non-finite values")

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


def measure_evaluation_timing(
    compiled,
    validation_inputs,
    n_evaluations: int,
) -> dict[str, Any]:
    """
    Measure repeated compiled graph evaluation wall time.

    Outputs are not stored here so that timing reflects evaluation cost, not
    Python list accumulation.
    """

    if n_evaluations < 1:
        raise ValueError("n_evaluations must be at least 1")

    result = compiled(**validation_inputs)
    first_output = extract_scalar_output(result)

    start = time.perf_counter()

    last_output = first_output
    for _ in range(n_evaluations):
        result = compiled(**validation_inputs)
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


def measure_evaluation_memory(
    compiled,
    validation_inputs,
) -> dict[str, float]:
    """
    Measure memory change for a single compiled graph evaluation.

    This is separate from timing runs so memory is not reported as accumulation
    across many repeated evaluations.
    """

    result = compiled(**validation_inputs)
    extract_scalar_output(result)

    gc.collect()

    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    result = compiled(**validation_inputs)
    extract_scalar_output(result)

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    return {
        "memory_n_evaluations": 1,
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
    n_evaluations: int,
) -> dict[str, Any]:
    """
    Run compiled evaluation benchmark for one workspace, target, mode, and
    evaluation count.

    Workspace loading, model creation, log_prob construction, and compilation
    are setup only. The measured operation is only compiled graph evaluation.
    """

    validate_benchmark_config(target=target, mode=mode, n_evaluations=n_evaluations)
    workspace_path = validate_workspace_path(workspace_path)

    model, compiled, validation_inputs = prepare_compiled_graph(
        workspace_path=workspace_path,
        target=target,
        mode=mode,
    )

    validation_outputs = evaluate_compiled_graph(
        compiled=compiled,
        validation_inputs=validation_inputs,
        n_evaluations=VALIDATION_N_EVALUATIONS,
    )
    validation_summary = validate_evaluation(validation_outputs)

    del validation_outputs
    gc.collect()

    memory_summary = measure_evaluation_memory(
        compiled=compiled,
        validation_inputs=validation_inputs,
    )

    timing_summary = measure_evaluation_timing(
        compiled=compiled,
        validation_inputs=validation_inputs,
        n_evaluations=n_evaluations,
    )

    del compiled
    del model
    gc.collect()

    return {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "target": target,
        "mode": mode,
        "n_evaluations": n_evaluations,
        **timing_summary,
        **memory_summary,
        "status": "success",
        **validation_summary,
    }

def print_result(result: dict[str, Any]) -> None:
    """
    Print a readable benchmark summary.
    """

    print()
    print("=" * 72)
    print("Compiled evaluation benchmark")
    print("=" * 72)
    print(f"Workspace:    {result['workspace']}")
    print(f"Target:       {result['target']}")
    print(f"Mode:         {result['mode']}")
    print(f"Evaluations:  {result['n_evaluations']}")
    print(f"Status:       {result['status']}")

    print()
    print("Timing")
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
    print(f"  current RSS before: {result['current_rss_before_mb']:.3f} MB")
    print(f"  current RSS after:  {result['current_rss_after_mb']:.3f} MB")
    print(f"  current RSS delta:  {result['current_rss_delta_mb']:.3f} MB")
    print(f"  peak RSS before:    {result['peak_rss_before_mb']:.3f} MB")
    print(f"  peak RSS after:     {result['peak_rss_after_mb']:.3f} MB")
    print(f"  peak RSS delta:     {result['peak_rss_delta_mb']:.3f} MB")

    print()
    print("Validation")
    print(f"  outputs:            {result['n_outputs']}")
    print(f"  reference output:   {result['reference_output']}")
    print(f"  finite outputs:     {result['all_outputs_finite']}")
    print(f"  max abs deviation:  {result['max_absolute_deviation']}")
    print(f"  stable outputs:     {result['outputs_stable']}")



def make_error_result(
    workspace_path: Path,
    target: str,
    mode: str,
    n_evaluations: int,
    exc: Exception,
) -> dict[str, Any]:
    """Build a structured result for a failed compiled evaluation run."""

    return {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "target": target,
        "mode": mode,
        "n_evaluations": n_evaluations,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(
            traceback.format_exception(
                type(exc),
                exc,
                exc.__traceback__,
            )
        ),
    }


def print_failed_result(result: dict[str, Any]) -> None:
    """Print a readable failed benchmark summary."""

    print()
    print("=" * 72)
    print("Compiled evaluation benchmark FAILED")
    print("=" * 72)
    print(f"Workspace:    {result['workspace']}")
    print(f"Target:       {result['target']}")
    print(f"Mode:         {result['mode']}")
    print(f"Evaluations:  {result['n_evaluations']}")
    print(f"Status:       {result['status']}")
    print(f"Error:        {result['error_type']}: {result['error_message']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark repeated evaluation of compiled pyHS3 log_prob graphs."
    )

    parser.add_argument(
        "--workspaces",
        nargs="+",
        type=Path,
        default=[DEFAULT_WORKSPACE],
        help="Workspace JSON files to benchmark.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=[DEFAULT_TARGET],
        help="Workspace model targets, such as analysis or likelihood names.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[DEFAULT_MODE],
        help="PyTensor modes passed to workspace.model(...).",
    )
    parser.add_argument(
        "--n-evaluations",
        nargs="+",
        type=int,
        default=DEFAULT_N_EVALUATIONS,
        help="Numbers of repeated compiled evaluations to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where JSON benchmark output will be saved.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help="Name of the JSON benchmark output file.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create benchmark plots when multiple result entries are available.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=DEFAULT_PLOT_DIR,
        help="Directory where plots will be saved.",
    )

    return parser.parse_args()

def make_scaling_line_plot(
    results: list[dict[str, Any]],
    output_path: Path,
    metric_key: str,
    metric_label: str,
    title: str,
) -> None:
    """
    Create a scaling line plot for compiled evaluation metrics.

    The x-axis is the number of evaluations.
    Each line corresponds to one workspace.
    """

    results = [
        result
        for result in results
        if result.get("status") == "success" and metric_key in result
    ]

    if not results:
        print(f"Skipping {title}: no successful results with metric {metric_key}.")
        return

    grouped: dict[str, list[dict[str, Any]]] = {}

    for result in results:
        workspace = result["workspace"]
        grouped.setdefault(workspace, []).append(result)

    fig, ax = plt.subplots(figsize=(14, 9))

    for workspace, workspace_results in grouped.items():
        workspace_results = sorted(
            workspace_results,
            key=lambda item: item["n_evaluations"],
        )

        x_values = [
            result["n_evaluations"]
            for result in workspace_results
        ]
        y_values = [
            result[metric_key]
            for result in workspace_results
        ]

        if metric_key == "average_runtime_seconds_per_evaluation":
            y_values = [
                value * 1000.0
                for value in y_values
            ]

        label = workspace.replace(".json", "")

        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.5,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of evaluations")
    ax.set_ylabel(metric_label)
    ax.set_title(title, fontsize=24, weight="bold", pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_plots(
    results: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    """
    Create scaling plots for the compiled evaluation benchmark.
    """

    results = [result for result in results if result.get("status") == "success"]

    if len(results) < 2:
        print("Skipping plots: at least two successful result entries are needed.")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    make_scaling_line_plot(
        results=results,
        output_path=plot_dir / "compiled_evaluation_average_time.png",
        metric_key="average_runtime_seconds_per_evaluation",
        metric_label="Average wall time per evaluation [ms]",
        title="Compiled evaluation average wall time",
    )

    make_scaling_line_plot(
        results=results,
        output_path=plot_dir / "compiled_evaluation_throughput.png",
        metric_key="throughput_evaluations_per_second",
        metric_label="Throughput [evaluations / s]",
        title="Compiled evaluation throughput",
    )

    if should_plot_metric(results, "current_rss_delta_mb"):
        make_bar_plot(
            results=results,
            output_path=plot_dir / "compiled_evaluation_current_rss_delta.png",
            title="Compiled evaluation current RSS delta",
            metric_key="current_rss_delta_mb",
            metric_label="Current RSS delta [MB]",
        )
    else:
        print("Skipping current RSS plot: all values are zero.")

    if should_plot_metric(results, "peak_rss_delta_mb"):
        make_bar_plot(
            results=results,
            output_path=plot_dir / "compiled_evaluation_peak_rss_delta.png",
            title="Compiled evaluation peak RSS delta",
            metric_key="peak_rss_delta_mb",
            metric_label="Peak RSS delta [MB]",
        )
    else:
        print("Skipping peak RSS plot: all values are zero.")


def main() -> None:
    args = parse_args()

    if any(n_evaluations < 1 for n_evaluations in args.n_evaluations):
        raise ValueError("--n-evaluations values must be at least 1")

    for workspace_path in args.workspaces:
        validate_workspace_path(workspace_path)

    for target in args.targets:
        if not target:
            raise ValueError("--targets must contain only non-empty strings")

    for mode in args.modes:
        if not mode:
            raise ValueError("--modes must contain only non-empty strings")

    results = []

    ctx = get_context("spawn")

    for workspace_path in args.workspaces:
        for target in args.targets:
            for mode in args.modes:
                for n_evaluations in args.n_evaluations:
                    print(
                        f"Running {workspace_path.name}, "
                        f"target={target}, "
                        f"mode={mode}, "
                        f"n_evaluations={n_evaluations}",
                        flush=True,
                    )

                    try:
                        with ctx.Pool(processes=1) as pool:
                            result = pool.apply(
                                run_single_benchmark,
                                args=(
                                    workspace_path,
                                    target,
                                    mode,
                                    n_evaluations,
                                ),
                            )
                    except Exception as exc:
                        result = make_error_result(
                            workspace_path=workspace_path,
                            target=target,
                            mode=mode,
                            n_evaluations=n_evaluations,
                            exc=exc,
                        )

                    results.append(result)

                    if result["status"] == "success":
                        print_result(result)
                    else:
                        print_failed_result(result)

    output_data: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "n_results": len(results),
        "results": results,
    }

    output_path = args.output_dir / args.output_name
    save_json(output_data, output_path)
    verify_output_file(output_path)

    print()
    print(f"Saved result to {output_path}")

    if args.plot:
        make_plots(results=results, plot_dir=args.plot_dir)
        print(f"Saved plots to {args.plot_dir}")


if __name__ == "__main__":
    main()
