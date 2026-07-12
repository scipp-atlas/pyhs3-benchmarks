from __future__ import annotations

import json
import math
import resource
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import psutil


def save_json(data: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True)


def current_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0


def finite_scalar(value: Any, *, label: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        raise ValueError(f"{label} returned an empty array")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"{label} returned a non-finite value: {result}")
    return result


def compiled_array(result: Any, *, label: str) -> np.ndarray:
    if not isinstance(result, tuple) or not result:
        raise TypeError(f"{label} must return a non-empty tuple")
    array = np.asarray(result[0], dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} returned invalid values")
    return array


def compiled_scalar(result: Any, *, label: str) -> float:
    return finite_scalar(compiled_array(result, label=label), label=label)


def time_once(operation: Callable[[], Any], *, label: str) -> tuple[Any, float]:
    start = time.perf_counter()
    output = operation()
    elapsed = time.perf_counter() - start
    if elapsed < 0.0 or not math.isfinite(elapsed):
        raise RuntimeError(f"Invalid timing for {label}: {elapsed}")
    return output, elapsed


def benchmark_batches(
    operation: Callable[[int], Any],
    *,
    batch_size: int,
    n_batches: int,
    warmup_batches: int,
) -> dict[str, Any]:
    if batch_size < 1 or n_batches < 1 or warmup_batches < 0:
        raise ValueError("Invalid batch benchmark configuration")

    for batch in range(warmup_batches):
        for index in range(batch_size):
            operation(batch * batch_size + index)

    samples: list[float] = []
    last_output: Any = None
    for batch in range(n_batches):
        start = time.perf_counter()
        for index in range(batch_size):
            last_output = operation(batch * batch_size + index)
        samples.append((time.perf_counter() - start) / batch_size)

    median = statistics.median(samples)
    return {
        "batch_size": int(batch_size),
        "n_batches": int(n_batches),
        "warmup_batches": int(warmup_batches),
        "timings_seconds_per_evaluation": [float(v) for v in samples],
        "steady_state_seconds_median": float(median),
        "steady_state_seconds_mean": float(statistics.mean(samples)),
        "steady_state_seconds_std": float(
            statistics.stdev(samples) if len(samples) > 1 else 0.0
        ),
        "throughput_evaluations_per_second": float(1.0 / median),
        "last_output": finite_scalar(last_output, label="timed operation"),
    }


def benchmark_scaling(
    operation: Callable[[int], Any],
    *,
    n_evaluations: Iterable[int],
    repeats: int,
    warmup_evaluations: int,
) -> list[dict[str, Any]]:
    counts = [int(v) for v in n_evaluations]
    if not counts or any(v < 1 for v in counts):
        raise ValueError("n_evaluations must contain positive integers")
    if repeats < 1 or warmup_evaluations < 0:
        raise ValueError("Invalid scaling configuration")

    for index in range(warmup_evaluations):
        operation(index)

    rows: list[dict[str, Any]] = []
    cursor = 0
    for count in counts:
        per_eval: list[float] = []
        totals: list[float] = []
        last_output: Any = None
        for _ in range(repeats):
            start = time.perf_counter()
            for local_index in range(count):
                last_output = operation(cursor + local_index)
            elapsed = time.perf_counter() - start
            cursor += count
            totals.append(elapsed)
            per_eval.append(elapsed / count)
        median = statistics.median(per_eval)
        rows.append(
            {
                "n_evaluations": count,
                "timing_repeats": repeats,
                "total_runtime_seconds_samples": [float(v) for v in totals],
                "time_per_value_seconds_samples": [float(v) for v in per_eval],
                "total_runtime_seconds_median": float(statistics.median(totals)),
                "time_per_value_seconds_median": float(median),
                "time_per_value_seconds_mean": float(statistics.mean(per_eval)),
                "time_per_value_seconds_std": float(
                    statistics.stdev(per_eval) if len(per_eval) > 1 else 0.0
                ),
                "time_per_value_ns": float(median * 1e9),
                "throughput_evaluations_per_second": float(1.0 / median),
                "last_output": finite_scalar(last_output, label="scaling operation"),
            }
        )
    return rows


def agreement_arrays(
    observed: Any, reference: Any, *, rtol: float, atol: float
) -> dict[str, Any]:
    left = np.asarray(observed, dtype=np.float64).reshape(-1)
    right = np.asarray(reference, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"Validation shape mismatch: {left.shape} != {right.shape}")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("Validation arrays contain non-finite values")
    absolute = np.abs(left - right)
    relative = absolute / np.maximum(np.abs(right), 1e-300)
    passed = bool(np.allclose(left, right, rtol=rtol, atol=atol))
    return {
        "n_validation_values": int(left.size),
        "all_values_finite": True,
        "max_abs_diff": float(np.max(absolute)) if absolute.size else 0.0,
        "mean_abs_diff": float(np.mean(absolute)) if absolute.size else 0.0,
        "max_rel_diff": float(np.max(relative)) if relative.size else 0.0,
        "mean_rel_diff": float(np.mean(relative)) if relative.size else 0.0,
        "allclose_passed": passed,
        "validation_status": "success" if passed else "mismatch",
    }


def delta_curve(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Cannot compute delta curve from invalid values")
    return array - np.min(array)


def style_axes(ax: Any, *, grid_axis: str = "both") -> None:
    ax.grid(True, which="major", axis=grid_axis, alpha=0.28)
    ax.grid(True, which="minor", axis=grid_axis, alpha=0.12)


def save_figure(fig: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
