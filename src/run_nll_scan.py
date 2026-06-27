from __future__ import annotations

import argparse
import gc
import math
import time
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
    build_log_prob,
    get_current_rss_mb,
    get_peak_rss_mb,
    make_bar_plot,
    save_json,
    should_plot_metric,
    build_validation_inputs,
    compile_log_prob,
)

BENCHMARK_NAME = "nll_scan"

DEFAULT_SCAN_PARAMETER = "mu_sig"
DEFAULT_SCAN_MIN = 0.0
DEFAULT_SCAN_MAX = 5.0
DEFAULT_N_SCAN_POINTS = 101

DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "nll_scan_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME


def extract_log_prob(result) -> float:
    """
    Extract the log probability value from the compiled result.
    """

    if not isinstance(result, tuple):
        raise TypeError(
            f"Expected compiled result to be a tuple, got {type(result).__name__}"
        )

    if len(result) == 0:
        raise ValueError("Compiled result tuple is empty")

    return float(np.asarray(result[0]).reshape(-1)[0])


def make_scan_values(
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
) -> list[float]:
    """
    Create a list of scan values for the given range and number of points.
    """

    if n_scan_points < 2:
        raise ValueError("n_scan_points must be at least 2")

    return np.linspace(
        scan_min,
        scan_max,
        n_scan_points,
    ).tolist()


def set_scan_parameter(
    inputs: dict[str, Any],
    parameter_name: str,
    value: float,
) -> dict[str, Any]:
    """
    Set the value of a scan parameter in the inputs dictionary.
    """

    if parameter_name not in inputs:
        raise KeyError(
            f"Scan parameter '{parameter_name}' not found in compiled inputs: "
            f"{list(inputs.keys())}"
        )

    updated_inputs = dict(inputs)
    original_value = inputs[parameter_name]
    original_array = np.asarray(original_value)

    if original_array.shape == ():
        updated_inputs[parameter_name] = value
    else:
        updated_inputs[parameter_name] = np.full_like(
            original_array,
            value,
            dtype=float,
        )

    return updated_inputs


def evaluate_nll_scan(
    compiled,
    base_inputs: dict[str, Any],
    scan_parameter: str,
    scan_values: list[float],
) -> list[float]:
    """
    Evaluate the negative log-likelihood (NLL) over a grid of scan values.
    """

    nll_values = []

    for value in scan_values:
        inputs = set_scan_parameter(
            inputs=base_inputs,
            parameter_name=scan_parameter,
            value=value,
        )

        log_prob = extract_log_prob(compiled(**inputs))
        nll_values.append(-log_prob)

    return nll_values


def validate_nll_scan(
    scan_values: list[float],
    nll_values: list[float],
) -> dict[str, Any]:
    """
    Validate the results of an NLL scan.
    """

    if len(scan_values) == 0:
        raise ValueError("Scan grid is empty")

    if len(scan_values) != len(nll_values):
        raise ValueError(
            f"Scan values and NLL values have different lengths: "
            f"{len(scan_values)} != {len(nll_values)}"
        )

    all_finite = all(math.isfinite(value) for value in nll_values)

    minimum_index = min(
        range(len(nll_values)),
        key=lambda index: nll_values[index],
    )

    return {
        "n_scan_outputs": len(nll_values),
        "all_nll_values_finite": all_finite,
        "minimum_index": minimum_index,
        "minimum_scan_value": scan_values[minimum_index],
        "minimum_nll_value": nll_values[minimum_index],
        "nll_min": min(nll_values),
        "nll_max": max(nll_values),
        "nll_range": max(nll_values) - min(nll_values),
    }


def measure_nll_scan_timing(
    compiled,
    base_inputs: dict[str, Any],
    scan_parameter: str,
    scan_values: list[float],
) -> dict[str, Any]:
    """
    Measure the timing of an NLL scan.
    """

    start = time.perf_counter()

    nll_values = evaluate_nll_scan(
        compiled=compiled,
        base_inputs=base_inputs,
        scan_parameter=scan_parameter,
        scan_values=scan_values,
    )

    end = time.perf_counter()

    total_runtime_seconds = end - start
    runtime_per_scan_point_seconds = total_runtime_seconds / len(scan_values)
    throughput_scan_points_per_second = (
        len(scan_values) / total_runtime_seconds
        if total_runtime_seconds > 0
        else float("inf")
    )

    return {
        "total_runtime_seconds": total_runtime_seconds,
        "runtime_per_scan_point_seconds": runtime_per_scan_point_seconds,
        "throughput_scan_points_per_second": throughput_scan_points_per_second,
        "first_nll_value": nll_values[0],
        "last_nll_value": nll_values[-1],
    }


def measure_nll_scan_memory(
    compiled,
    base_inputs: dict[str, Any],
    scan_parameter: str,
    scan_values: list[float],
) -> dict[str, float | int]:
    """
    Measure the memory usage of an NLL scan.
    """

    evaluate_nll_scan(
        compiled=compiled,
        base_inputs=base_inputs,
        scan_parameter=scan_parameter,
        scan_values=[scan_values[0]],
    )

    gc.collect()

    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    evaluate_nll_scan(
        compiled=compiled,
        base_inputs=base_inputs,
        scan_parameter=scan_parameter,
        scan_values=scan_values,
    )

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    return {
        "memory_n_scan_points": len(scan_values),
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
    scan_parameter: str,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
) -> dict[str, Any]:
    """
    Run a single NLL scan benchmark.
    """

    if n_scan_points < 2:
        raise ValueError("n_scan_points must be at least 2")

    model, log_prob = build_log_prob(
        workspace_path=workspace_path,
        target=target,
        mode=mode,
    )

    compiled = compile_log_prob(log_prob)
    base_inputs = build_validation_inputs(
        model=model,
        compiled=compiled,
    )

    scan_values = make_scan_values(
        scan_min=scan_min,
        scan_max=scan_max,
        n_scan_points=n_scan_points,
    )

    nll_values = evaluate_nll_scan(
        compiled=compiled,
        base_inputs=base_inputs,
        scan_parameter=scan_parameter,
        scan_values=scan_values,
    )
    validation_summary = validate_nll_scan(
        scan_values=scan_values,
        nll_values=nll_values,
    )

    memory_summary = measure_nll_scan_memory(
        compiled=compiled,
        base_inputs=base_inputs,
        scan_parameter=scan_parameter,
        scan_values=scan_values,
    )

    timing_summary = measure_nll_scan_timing(
        compiled=compiled,
        base_inputs=base_inputs,
        scan_parameter=scan_parameter,
        scan_values=scan_values,
    )

    del compiled
    del log_prob
    del model
    gc.collect()

    return {
        "benchmark": BENCHMARK_NAME,
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "target": target,
        "mode": mode,
        "scan_parameter": scan_parameter,
        "scan_min": scan_min,
        "scan_max": scan_max,
        "n_scan_points": n_scan_points,
        "scan_values": scan_values,
        "nll_values": nll_values,
        **timing_summary,
        **memory_summary,
        "status": "success",
        **validation_summary,
    }


def print_result(result: dict[str, Any]) -> None:
    """
    Print the result of a single NLL scan benchmark.
    """

    print()
    print("=" * 72)
    print("NLL scan benchmark")
    print("=" * 72)
    print(f"Workspace:       {result['workspace']}")
    print(f"Target:          {result['target']}")
    print(f"Mode:            {result['mode']}")
    print(f"Scan parameter:  {result['scan_parameter']}")
    print(f"Scan points:     {result['n_scan_points']}")
    print(f"Status:          {result['status']}")

    print()
    print("Timing")
    print(f"  full scan:     {result['total_runtime_seconds'] * 1000:.3f} ms")
    print(
        "  per point:     "
        f"{result['runtime_per_scan_point_seconds'] * 1000:.6f} ms"
    )
    print(
        "  throughput:    "
        f"{result['throughput_scan_points_per_second']:.3f} points/s"
    )

    print()
    print("Memory")
    print(f"  current RSS delta: {result['current_rss_delta_mb']:.3f} MB")
    print(f"  peak RSS delta:    {result['peak_rss_delta_mb']:.3f} MB")

    print()
    print("Validation")
    print(f"  finite NLL values: {result['all_nll_values_finite']}")
    print(f"  minimum at:        {result['minimum_scan_value']}")
    print(f"  minimum NLL:       {result['minimum_nll_value']}")
    print(f"  NLL range:         {result['nll_range']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyHS3 NLL scans over a parameter grid."
    )

    parser.add_argument("--workspaces", nargs="+", type=Path, default=[DEFAULT_WORKSPACE])
    parser.add_argument("--targets", nargs="+", default=[DEFAULT_TARGET])
    parser.add_argument("--modes", nargs="+", default=[DEFAULT_MODE])
    parser.add_argument("--scan-parameter", default=DEFAULT_SCAN_PARAMETER)
    parser.add_argument("--scan-min", type=float, default=DEFAULT_SCAN_MIN)
    parser.add_argument("--scan-max", type=float, default=DEFAULT_SCAN_MAX)
    parser.add_argument("--n-scan-points", nargs="+", type=int, default=[DEFAULT_N_SCAN_POINTS])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)

    return parser.parse_args()


def make_plots(results: list[dict[str, Any]], plot_dir: Path) -> None:
    """
    Create bar plots for the benchmark results.
    """

    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_results = []
    for result in results:
        plot_result = dict(result)
        plot_result["runtime_per_scan_point_ms"] = (
            result["runtime_per_scan_point_seconds"] * 1000.0
        )
        plot_result["total_runtime_ms"] = (
            result["total_runtime_seconds"] * 1000.0
        )
        plot_results.append(plot_result)

    make_bar_plot(
        results=plot_results,
        output_path=plot_dir / "nll_scan_total_runtime.png",
        title="NLL scan total runtime",
        metric_key="total_runtime_ms",
        metric_label="Full scan runtime [ms]",
    )

    make_bar_plot(
        results=plot_results,
        output_path=plot_dir / "nll_scan_runtime_per_point.png",
        title="NLL scan runtime per point",
        metric_key="runtime_per_scan_point_ms",
        metric_label="Runtime per scan point [ms]",
    )

    if should_plot_metric(plot_results, "current_rss_delta_mb"):
        make_bar_plot(
            results=plot_results,
            output_path=plot_dir / "nll_scan_current_rss_delta.png",
            title="NLL scan current RSS delta",
            metric_key="current_rss_delta_mb",
            metric_label="Current RSS delta [MB]",
        )

    if should_plot_metric(plot_results, "peak_rss_delta_mb"):
        make_bar_plot(
            results=plot_results,
            output_path=plot_dir / "nll_scan_peak_rss_delta.png",
            title="NLL scan peak RSS delta",
            metric_key="peak_rss_delta_mb",
            metric_label="Peak RSS delta [MB]",
        )


def main() -> None:
    args = parse_args()

    for n_scan_points in args.n_scan_points:
        if n_scan_points < 2:
            raise ValueError("--n-scan-points values must be at least 2")

    results = []
    ctx = get_context("spawn")

    for workspace_path in args.workspaces:
        for target in args.targets:
            for mode in args.modes:
                for n_scan_points in args.n_scan_points:
                    print(
                        f"Running {workspace_path.name}, "
                        f"target={target}, "
                        f"mode={mode}, "
                        f"scan_parameter={args.scan_parameter}, "
                        f"n_scan_points={n_scan_points}",
                        flush=True,
                    )

                    with ctx.Pool(processes=1) as pool:
                        result = pool.apply(
                            run_single_benchmark,
                            args=(
                                workspace_path,
                                target,
                                mode,
                                args.scan_parameter,
                                args.scan_min,
                                args.scan_max,
                                n_scan_points,
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
