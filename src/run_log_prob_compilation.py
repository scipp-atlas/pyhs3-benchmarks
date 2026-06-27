from __future__ import annotations

import argparse
import gc
import math
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from pyhs3 import jaxify
from pyhs3.transpile import JaxifiedGraph
from pyhs3.workspace import Workspace
from .utils import (
    get_current_rss_mb,
    get_peak_rss_mb,
    make_bar_plot,
    save_json,
    summarize_timings,
    should_plot_metric,
    load_workspace,
    create_model,
    build_log_prob,
    compile_log_prob,
    build_validation_inputs,
)

from .config import (
    DEFAULT_MODE,
    DEFAULT_N_RUNS,
    DEFAULT_TARGET,
    DEFAULT_WORKSPACE,
    PLOTS_DIR,
    RESULTS_DIR,
)

BENCHMARK_NAME = "log_prob_compilation"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "log_prob_compilation_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME


def validate_workspace_path(workspace_path: Path) -> Path:
    """
    Validate that the workspace path points to an existing JSON file.
    """

    if not workspace_path.exists():
        raise FileNotFoundError(f"Workspace file does not exist: {workspace_path}")

    if not workspace_path.is_file():
        raise FileNotFoundError(f"Workspace path is not a file: {workspace_path}")

    return workspace_path


def validate_benchmark_config(target: str, mode: str, n_runs: int) -> None:
    """
    Validate benchmark configuration before running expensive work.
    """

    if not target:
        raise ValueError("target must be a non-empty string")

    if not mode:
        raise ValueError("mode must be a non-empty string")

    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")


def validate_timings(timings: list[float]) -> None:
    """
    Validate timing samples before summarizing them.
    """

    if len(timings) == 0:
        raise ValueError("Timing samples are empty")

    if any(timing <= 0 for timing in timings):
        raise ValueError("All timing samples must be positive")


def verify_output_file(output_path: Path) -> None:
    """
    Verify that save_json created a regular output file.
    """

    if not output_path.exists():
        raise FileNotFoundError(f"Benchmark output file was not created: {output_path}")

    if not output_path.is_file():
        raise FileNotFoundError(f"Benchmark output path is not a file: {output_path}")


def validate_compiled_graph(
    model,
    compiled: JaxifiedGraph,
) -> dict[str, Any]:
    """
    Validate that the compiled graph exists and can be called once.
    """

    if compiled is None:
        raise ValueError("Compilation returned None")

    if not isinstance(compiled, JaxifiedGraph):
        raise TypeError(
            f"Expected JaxifiedGraph, got {type(compiled).__name__}"
        )

    validation_inputs = build_validation_inputs(
        model=model,
        compiled=compiled,
    )

    result = compiled(**validation_inputs)

    if not isinstance(result, tuple):
        raise TypeError(
            f"Expected compiled result to be a tuple, got {type(result).__name__}"
        )

    if len(result) == 0:
        raise ValueError("Compiled result tuple is empty")

    result_value = result[0]
    first_value = float(result_value[0])

    if not math.isfinite(first_value):
        raise ValueError(f"Compiled result is not finite: {first_value}")

    return {
        "compiled_type": type(compiled).__name__,
        "n_compiled_inputs": len(compiled.input_names),
        "compiled_input_names": list(compiled.input_names),
        "validation_result_type": type(result_value).__name__,
        "validation_first_value": first_value,
        "validation_result_is_finite": True,
    }


def measure_compilation_memory(
    workspace_path: Path,
    target: str,
    mode: str,
) -> tuple[Any, JaxifiedGraph, dict[str, float]]:
    """
    Measure memory change for a single log_prob compilation.

    This is separate from repeated timing runs so memory is not reported as
    accumulation across many compiled graphs.
    """

    gc.collect()

    model, log_prob = build_log_prob(
        workspace_path=workspace_path,
        target=target,
        mode=mode,
    )

    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    compiled = compile_log_prob(log_prob)

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    return model, compiled, {
        "memory_n_runs": 1,
        "current_rss_before_mb": current_rss_before_mb,
        "current_rss_after_mb": current_rss_after_mb,
        "current_rss_delta_mb": current_rss_after_mb - current_rss_before_mb,
        "peak_rss_before_mb": peak_rss_before_mb,
        "peak_rss_after_mb": peak_rss_after_mb,
        "peak_rss_delta_mb": peak_rss_after_mb - peak_rss_before_mb,
    }


def measure_compilation_timing(
    workspace_path: Path,
    target: str,
    mode: str,
    n_runs: int,
) -> list[float]:
    """
    Measure repeated jaxify(model.log_prob) wall times.

    Each timing run rebuilds the model and log_prob before the timed section.
    The timed section itself only includes jaxify(log_prob).
    """

    timings: list[float] = []

    for _ in range(n_runs):
        model, log_prob = build_log_prob(
            workspace_path=workspace_path,
            target=target,
            mode=mode,
        )

        start = time.perf_counter()
        compiled = compile_log_prob(log_prob)
        end = time.perf_counter()

        timings.append(end - start)

        del compiled
        del log_prob
        del model
        gc.collect()

    return timings


def run_single_benchmark(
    workspace_path: Path,
    target: str,
    mode: str,
    n_runs: int,
) -> dict[str, Any]:
    """
    Run log_prob compilation benchmark for one workspace, target, and mode.

    Workspace loading, model creation, and log_prob construction are setup only.
    The timed operation is only jaxify(model.log_prob).
    """

    validate_benchmark_config(target=target, mode=mode, n_runs=n_runs)
    workspace_path = validate_workspace_path(workspace_path)

    model, compiled, memory_summary = measure_compilation_memory(
        workspace_path=workspace_path,
        target=target,
        mode=mode,
    )
    validation_summary = validate_compiled_graph(
        model=model,
        compiled=compiled,
    )

    del compiled
    del model
    gc.collect()

    timings = measure_compilation_timing(
        workspace_path=workspace_path,
        target=target,
        mode=mode,
        n_runs=n_runs,
    )
    validate_timings(timings)
    timing_summary = summarize_timings(timings)

    return {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "target": target,
        "mode": mode,
        "n_runs": n_runs,
        "timings_seconds": timings,
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
    print("log_prob compilation benchmark")
    print("=" * 72)
    print(f"Workspace: {result['workspace']}")
    print(f"Target:    {result['target']}")
    print(f"Mode:      {result['mode']}")
    print(f"Runs:      {result['n_runs']}")
    print(f"Status:    {result['status']}")

    print()
    print("Timing")
    print(f"  mean: {result['wall_time_seconds_mean'] * 1000:.3f} ms")
    print(f"  median: {result['wall_time_seconds_median'] * 1000:.3f} ms")
    print(f"  std:  {result['wall_time_seconds_std'] * 1000:.3f} ms")

    print()
    print("Memory")
    print(f"  memory runs:        {result['memory_n_runs']}")
    print(f"  current RSS before: {result['current_rss_before_mb']:.3f} MB")
    print(f"  current RSS after:  {result['current_rss_after_mb']:.3f} MB")
    print(f"  current RSS delta:  {result['current_rss_delta_mb']:.3f} MB")
    print(f"  peak RSS before:    {result['peak_rss_before_mb']:.3f} MB")
    print(f"  peak RSS after:     {result['peak_rss_after_mb']:.3f} MB")
    print(f"  peak RSS delta:     {result['peak_rss_delta_mb']:.3f} MB")

    print()
    print("Validation")
    print(f"  compiled type:       {result['compiled_type']}")
    print(f"  compiled inputs:     {result['n_compiled_inputs']}")
    print(f"  input names:         {result['compiled_input_names']}")
    print(f"  validation type:     {result['validation_result_type']}")
    print(f"  validation value:    {result['validation_first_value']}")
    print(f"  validation finite:   {result['validation_result_is_finite']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark JAX compilation/transpilation of pyHS3 log_prob graphs."
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
        "--n-runs",
        type=int,
        default=DEFAULT_N_RUNS,
        help="Number of repeated timing runs per workspace/target/mode.",
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

def make_plots(
    results: list[dict[str, Any]],
    plot_dir: Path,
) -> None:
    """
    Create standard plots for the log_prob compilation benchmark.
    """

    plot_dir.mkdir(parents=True, exist_ok=True)

    wall_time_plot_path = plot_dir / "log_prob_compilation_wall_time.png"
    make_bar_plot(
        results=results,
        output_path=plot_dir / "log_prob_compilation_wall_time.png",
        title="log_prob compilation wall time",
        metric_key="wall_time_seconds_mean",
        metric_label="Mean wall time [s]",
    )

    print(f"Saved plot to {wall_time_plot_path}")

    if should_plot_metric(results, "current_rss_delta_mb"):
        make_bar_plot(
            results=results,
            output_path=plot_dir / "log_prob_compilation_current_rss_delta.png",
            title="log_prob compilation RSS increase",
            metric_key="current_rss_delta_mb",
            metric_label="RSS increase [MB]",
        )

        print(f"Saved plot to {plot_dir / 'log_prob_compilation_current_rss_delta.png'}")
    else:
        print("Skipping current RSS plot: all values are zero.")

    if should_plot_metric(results, "peak_rss_delta_mb"):
        make_bar_plot(
            results=results,
            output_path=plot_dir / "log_prob_compilation_peak_rss_delta.png",
            title="log_prob compilation peak RSS increase",
            metric_key="peak_rss_delta_mb",
            metric_label="Peak RSS increase [MB]",
        )
        print(f"Saved plot to {plot_dir / 'log_prob_compilation_peak_rss_delta.png'}")
    else:
        print("Skipping peak RSS plot: all values are zero.")


def main() -> None:
    args = parse_args()

    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1")

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
                with ctx.Pool(processes=1) as pool:
                    result = pool.apply(
                        run_single_benchmark,
                        args=(workspace_path, target, mode, args.n_runs),
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
    verify_output_file(output_path)

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
