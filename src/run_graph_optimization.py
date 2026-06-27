from __future__ import annotations

import argparse
import gc
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, cast

from pytensor.compile import mode as _ptmode
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.traversal import explicit_graph_inputs

from .config import (
    DEFAULT_MODE,
    DEFAULT_N_RUNS,
    DEFAULT_TARGET,
    DEFAULT_WORKSPACE,
    PLOTS_DIR,
    RESULTS_DIR,
)
from .utils import (
    build_log_prob,
    get_current_rss_mb,
    get_peak_rss_mb,
    make_bar_plot,
    save_json,
    should_plot_metric,
    summarize_timings,
)

BENCHMARK_NAME = "graph_optimization"

DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "graph_optimization_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME


def build_function_graph(log_prob) -> FunctionGraph:
    """
    Build a FunctionGraph from the given log_prob variable.
    """

    inputs = cast(
        list,
        [
            variable
            for variable in explicit_graph_inputs([log_prob])
            if variable.name is not None
        ],
    )

    return FunctionGraph(
        inputs=list(inputs),
        outputs=[log_prob],
        clone=True,
    )


def optimize_graph(fgraph: FunctionGraph) -> FunctionGraph:
    """
    Optimize the given function graph using the JAX optimizer.
    """

    _ptmode.JAX.optimizer.rewrite(fgraph)
    return fgraph


def validate_optimized_graph(
    fgraph: FunctionGraph,
    n_apply_nodes_before: int,
) -> dict[str, Any]:
    """
    Validate the optimized function graph and return a summary of its properties.
    """

    n_apply_nodes_after = len(fgraph.apply_nodes)

    if len(fgraph.outputs) != 1:
        raise ValueError(
            f"Expected one graph output, got {len(fgraph.outputs)}"
        )

    if n_apply_nodes_after == 0:
        raise ValueError("Optimized graph has no apply nodes")

    return {
        "fgraph_type": type(fgraph).__name__,
        "n_graph_inputs": len(fgraph.inputs),
        "n_graph_outputs": len(fgraph.outputs),
        "n_apply_nodes_before": n_apply_nodes_before,
        "n_apply_nodes_after": n_apply_nodes_after,
        "apply_node_delta": n_apply_nodes_after - n_apply_nodes_before,
        "optimizer": "JAX",
    }


def measure_graph_optimization_memory(
    workspace_path: Path,
    target: str,
    mode: str,
) -> tuple[FunctionGraph, dict[str, float | int]]:
    """
    Measure the memory usage of optimizing the graph for the given workspace, target, and mode.
    Returns the optimized function graph and a dictionary containing memory usage statistics.
    """

    gc.collect()

    _, log_prob = build_log_prob(
        workspace_path=workspace_path,
        target=target,
        mode=mode,
    )
    fgraph = build_function_graph(log_prob)

    n_apply_nodes_before = len(fgraph.apply_nodes)

    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    optimized_fgraph = optimize_graph(fgraph)

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    return optimized_fgraph, {
        "memory_n_runs": 1,
        "current_rss_before_mb": current_rss_before_mb,
        "current_rss_after_mb": current_rss_after_mb,
        "current_rss_delta_mb": current_rss_after_mb - current_rss_before_mb,
        "peak_rss_before_mb": peak_rss_before_mb,
        "peak_rss_after_mb": peak_rss_after_mb,
        "peak_rss_delta_mb": peak_rss_after_mb - peak_rss_before_mb,
        "n_apply_nodes_before": n_apply_nodes_before,
    }


def measure_graph_optimization_timing(
    workspace_path: Path,
    target: str,
    mode: str,
    n_runs: int,
) -> list[float]:
    """
    Measure the time taken to optimize the graph for the given workspace, target, and mode.
    """

    timings = []

    for _ in range(n_runs):
        _, log_prob = build_log_prob(
            workspace_path=workspace_path,
            target=target,
            mode=mode,
        )
        fgraph = build_function_graph(log_prob)

        start = time.perf_counter()
        optimize_graph(fgraph)
        end = time.perf_counter()

        timings.append(end - start)

        del fgraph
        del log_prob
        gc.collect()

    return timings


def run_single_benchmark(
    workspace_path: Path,
    target: str,
    mode: str,
    n_runs: int,
) -> dict[str, Any]:
    """
    Run a single benchmark for the given workspace, target, and mode.
    Returns a dictionary containing the benchmark results.
    """

    fgraph, memory_summary = measure_graph_optimization_memory(
        workspace_path=workspace_path,
        target=target,
        mode=mode,
    )

    validation_summary = validate_optimized_graph(
        fgraph=fgraph,
        n_apply_nodes_before=int(memory_summary["n_apply_nodes_before"]),
    )

    del fgraph
    gc.collect()

    timings = measure_graph_optimization_timing(
        workspace_path=workspace_path,
        target=target,
        mode=mode,
        n_runs=n_runs,
    )

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
    Print the benchmark result in a human-readable format. 
    """

    print()
    print("=" * 72)
    print("Graph optimization benchmark")
    print("=" * 72)
    print(f"Workspace: {result['workspace']}")
    print(f"Target:    {result['target']}")
    print(f"Mode:      {result['mode']}")
    print(f"Runs:      {result['n_runs']}")
    print(f"Status:    {result['status']}")

    print()
    print("Timing")
    print(f"  mean:   {result['wall_time_seconds_mean'] * 1000:.3f} ms")
    print(f"  median: {result['wall_time_seconds_median'] * 1000:.3f} ms")
    print(f"  std:    {result['wall_time_seconds_std'] * 1000:.3f} ms")

    print()
    print("Memory")
    print(f"  current RSS delta: {result['current_rss_delta_mb']:.3f} MB")
    print(f"  peak RSS delta:    {result['peak_rss_delta_mb']:.3f} MB")

    print()
    print("Validation")
    print(f"  graph inputs:       {result['n_graph_inputs']}")
    print(f"  graph outputs:      {result['n_graph_outputs']}")
    print(f"  apply nodes before: {result['n_apply_nodes_before']}")
    print(f"  apply nodes after:  {result['n_apply_nodes_after']}")
    print(f"  apply node delta:   {result['apply_node_delta']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTensor/JAX graph optimization for pyHS3 log_prob graphs."
    )

    parser.add_argument("--workspaces", nargs="+", type=Path, default=[DEFAULT_WORKSPACE])
    parser.add_argument("--targets", nargs="+", default=[DEFAULT_TARGET])
    parser.add_argument("--modes", nargs="+", default=[DEFAULT_MODE])
    parser.add_argument("--n-runs", type=int, default=DEFAULT_N_RUNS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)

    return parser.parse_args()


def make_plots(results: list[dict[str, Any]], plot_dir: Path) -> None:
    """
    Create plots for the benchmark results.
    """

    plot_dir.mkdir(parents=True, exist_ok=True)

    make_bar_plot(
        results=results,
        output_path=plot_dir / "graph_optimization_wall_time.png",
        title="Graph optimization wall time",
        metric_key="wall_time_seconds_mean",
        metric_label="Mean wall time [s]",
    )

    if should_plot_metric(results, "current_rss_delta_mb"):
        make_bar_plot(
            results=results,
            output_path=plot_dir / "graph_optimization_current_rss_delta.png",
            title="Graph optimization current RSS delta",
            metric_key="current_rss_delta_mb",
            metric_label="Current RSS delta [MB]",
        )

    if should_plot_metric(results, "peak_rss_delta_mb"):
        make_bar_plot(
            results=results,
            output_path=plot_dir / "graph_optimization_peak_rss_delta.png",
            title="Graph optimization peak RSS delta",
            metric_key="peak_rss_delta_mb",
            metric_label="Peak RSS delta [MB]",
        )


def main() -> None:
    args = parse_args()

    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1")

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