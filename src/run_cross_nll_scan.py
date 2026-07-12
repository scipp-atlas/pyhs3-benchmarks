"""Cross-engine point-by-point and batched full-dataset NLL benchmarks.

``pointwise_nll`` compares one complete NLL objective evaluation at one POI
value using the same normalized scalar PDF, dataset, event count, and parameter
point. RooFit and non-compiled pyHS3 evaluate the scalar PDF event by event.
Compiled pyHS3 preserves the same point-by-point mathematical workflow, but
places the event mapping, logarithm, and reduction inside one precompiled JAX
program. This avoids measuring one Python-to-JAX dispatch per event while still
remaining distinct from the native array-input benchmark.

``batched_full_dataset_nll`` is intentionally separate: pyHS3 receives the
complete observable array directly as one graph input. It is a workflow and
vectorization benchmark rather than a pure RooFit-equivalent microbenchmark.

Every engine/category/workspace run executes in a fresh spawned process.
Workspace loading, model construction, graph preparation, compilation, first
execution, steady-state NLL evaluation, and full-scan timing are reported
separately. Compilation is never included in steady-state timings.
"""

from __future__ import annotations

import argparse
import gc
import math
import multiprocessing as mp
import queue
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pyhs3 import jaxify
from pyhs3.workspace import Workspace

try:
    import ROOT
except ImportError:  # pragma: no cover - depends on local ROOT installation
    ROOT = None

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.benchmark_modes import (
        ENGINE_LABELS,
        PYHS3_COMPILED,
        PYHS3_NONCOMPILED,
        ROOFIT,
    )
    from src.config import DEFAULT_MODE, PLOTS_DIR, RESULTS_DIR
    from src.cross_benchmark_utils import (
        agreement_arrays,
        benchmark_batches,
        current_rss_mb,
        delta_curve,
        finite_scalar,
        peak_rss_mb,
        save_figure,
        save_json,
        style_axes,
        time_once,
    )
else:
    from .benchmark_modes import (
        ENGINE_LABELS,
        PYHS3_COMPILED,
        PYHS3_NONCOMPILED,
        ROOFIT,
    )
    from .config import DEFAULT_MODE, PLOTS_DIR, RESULTS_DIR
    from .cross_benchmark_utils import (
        agreement_arrays,
        benchmark_batches,
        current_rss_mb,
        delta_curve,
        finite_scalar,
        peak_rss_mb,
        save_figure,
        save_json,
        style_axes,
        time_once,
    )

BENCHMARK_NAME = "cross_nll"
POINTWISE_NLL = "pointwise_nll"
BATCHED_NLL = "batched_full_dataset_nll"
SUPPORTED_ENGINES = (PYHS3_NONCOMPILED, PYHS3_COMPILED, ROOFIT)
CATEGORIES = (POINTWISE_NLL, BATCHED_NLL)

DEFAULT_WORKSPACES = [
    Path("inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json"),
    Path("inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json"),
    Path("inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json"),
]

ENGINE_COLORS = {
    PYHS3_NONCOMPILED: "#1565C0",
    PYHS3_COMPILED: "#EF6C00",
    ROOFIT: "#00897B",
}
ENGINE_STYLE = {
    PYHS3_NONCOMPILED: dict(marker="s", linestyle="-", linewidth=2.2),
    PYHS3_COMPILED: dict(marker="o", linestyle="--", linewidth=2.2),
    ROOFIT: dict(marker="D", linestyle="-.", linewidth=2.2),
}


@dataclass(frozen=True)
class Config:
    engine: str
    category: str
    workspace_path: Path
    root_workspace_path: Path | None
    analysis: str
    distribution: str
    data_name: str
    observable_name: str
    observable_index: int
    poi: str
    mode: str
    mu_values: np.ndarray
    batch_size: int
    n_batches: int
    warmup_batches: int
    scan_repeats: int
    rtol: float
    atol: float


def _workspace_data(
    workspace: Workspace,
    data_name: str,
    observable_index: int,
) -> np.ndarray:
    """Extract one observable column from a named pyHS3 dataset."""
    for datum in workspace.data.root:
        if datum.name != data_name:
            continue
        try:
            values = np.asarray(
                [entry[observable_index] for entry in datum.entries],
                dtype=np.float64,
            )
        except (IndexError, TypeError) as exc:
            raise IndexError(
                f"Observable index {observable_index} is invalid for data {data_name!r}"
            ) from exc
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"Invalid data array {data_name!r}")
        return values

    available = [getattr(datum, "name", "<unnamed>") for datum in workspace.data.root]
    raise KeyError(f"Data {data_name!r} not found; available: {available}")


def _model_defaults(model: Any) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for source in (getattr(model, "data", {}), getattr(model, "free_params", {})):
        for name, value in source.items():
            values[name] = np.asarray(value, dtype=np.float64)
    return values


def _set_scalar(container: dict[str, np.ndarray], name: str, value: float) -> None:
    if name not in container:
        raise KeyError(f"Input {name!r} is unavailable; available: {sorted(container)}")
    original = np.asarray(container[name])
    container[name] = (
        np.asarray([value], dtype=np.float64)
        if original.ndim > 0
        else np.asarray(value, dtype=np.float64)
    )


def _shared_inputs(config: Config) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Prepare the shared parameter point and dataset outside timed workers."""
    workspace = Workspace.load(config.workspace_path)
    model = workspace.model(config.analysis, progress=False, mode=config.mode)
    params = _model_defaults(model)
    data = _workspace_data(workspace, config.data_name, config.observable_index)
    missing = {config.observable_name, config.poi} - set(params)
    if missing:
        raise KeyError(
            f"Required inputs missing: {sorted(missing)}; available: {sorted(params)}"
        )
    return params, data


def _find_root_workspace(root_file: Any) -> Any:
    for key in root_file.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom(ROOT.RooWorkspace.Class()):
            return obj
    raise KeyError("No RooWorkspace found")


def _sync_root_parameters(
    root_workspace: Any,
    params: dict[str, np.ndarray],
    excluded: set[str],
) -> dict[str, Any]:
    """Synchronize RooFit parameters with the pyHS3 parameter point.

    A generated RooExponential may use ``exp(tau*x)`` while pyHS3 uses a
    positive decay rate in ``exp(-tau*x)``.  A sign flip is applied only when
    the direct value is outside the RooRealVar range and the negated value is
    inside it.  All skipped/transformed parameters are recorded in the output.
    """
    synchronized: list[str] = []
    transformed: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    for name, value in params.items():
        if name in excluded:
            skipped[name] = "controlled by the benchmark"
            continue

        variable = root_workspace.var(str(name))
        if variable is None:
            skipped[name] = "not present in RooWorkspace"
            continue
        try:
            if not bool(variable):
                skipped[name] = "null PyROOT proxy"
                continue
        except Exception:
            pass

        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        if flat.size != 1:
            skipped[name] = f"non-scalar value with shape {np.asarray(value).shape}"
            continue

        source_value = float(flat[0])
        if not np.isfinite(source_value):
            skipped[name] = "non-finite pyHS3 value"
            continue

        try:
            minimum = float(variable.getMin())
            maximum = float(variable.getMax())
        except Exception:
            minimum, maximum = float("-inf"), float("inf")

        target_value = source_value
        transformation: str | None = None
        direct_in_range = minimum <= source_value <= maximum
        negated_in_range = minimum <= -source_value <= maximum

        if not direct_in_range:
            if name.startswith("tau_") and negated_in_range:
                target_value = -source_value
                transformation = "rooexponential_sign_flip"
            else:
                skipped[name] = (
                    f"value {source_value} outside RooFit range "
                    f"[{minimum}, {maximum}] and no validated mapping applies"
                )
                continue

        try:
            variable.setVal(target_value)
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__} while setting {target_value}: {exc}"
            continue

        synchronized.append(name)
        if transformation is not None:
            transformed[name] = {
                "transformation": transformation,
                "pyhs3_value": source_value,
                "roofit_value": target_value,
                "roofit_range": [minimum, maximum],
            }

    return {
        "synchronized": synchronized,
        "transformed": transformed,
        "skipped": skipped,
    }


def _validate_pdf_array(pdf: Any, expected_size: int, *, label: str) -> np.ndarray:
    values = np.asarray(pdf, dtype=np.float64).reshape(-1)
    if values.size != expected_size:
        raise ValueError(
            f"{label} returned {values.size} values, expected {expected_size}"
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{label} returned non-finite or non-positive values")
    return values


def _prepare_pyhs3(
    config: Config,
    shared_data: np.ndarray,
) -> tuple[dict[str, Any], Any, dict[str, np.ndarray]]:
    workspace, load_time = time_once(
        lambda: Workspace.load(config.workspace_path),
        label="workspace loading",
    )
    model, model_time = time_once(
        lambda: workspace.model(config.analysis, progress=False, mode=config.mode),
        label="model construction",
    )
    params = _model_defaults(model)
    missing = {config.observable_name, config.poi} - set(params)
    if missing:
        raise KeyError(
            f"Required inputs missing: {sorted(missing)}; available: {sorted(params)}"
        )
    return (
        {
            "workspace_loading_seconds": load_time,
            "model_construction_seconds": model_time,
            "graph_lookup_seconds": 0.0,
            "jaxify_seconds": 0.0,
            "graph_preparation_seconds": 0.0,
            "compilation_seconds": 0.0,
            "compiled_input_names": [],
        },
        model,
        params,
    )


def _make_pyhs3_noncompiled(
    config: Config,
    shared_data: np.ndarray,
) -> tuple[dict[str, Any], Callable[[float], float], Callable[[], None]]:
    lifecycle, model, params = _prepare_pyhs3(config, shared_data)
    data = np.asarray(shared_data, dtype=np.float64)

    if config.category == POINTWISE_NLL:

        def evaluate(mu: float) -> float:
            _set_scalar(params, config.poi, mu)
            total = 0.0
            for x in data:
                _set_scalar(params, config.observable_name, float(x))
                value = finite_scalar(
                    model.pdf(config.distribution, **params),
                    label="pyHS3 non-compiled normalized PDF",
                )
                if value <= 0.0:
                    raise ValueError(f"Non-positive pyHS3 PDF value: {value}")
                total -= math.log(value)
            return total

    elif config.category == BATCHED_NLL:
        params[config.observable_name] = data.copy()

        def evaluate(mu: float) -> float:
            _set_scalar(params, config.poi, mu)
            pdf = _validate_pdf_array(
                model.pdf(config.distribution, **params),
                data.size,
                label="pyHS3 non-compiled batched PDF",
            )
            return -float(np.sum(np.log(pdf)))

    else:  # pragma: no cover - guarded by argparse/config validation
        raise ValueError(config.category)

    return lifecycle, evaluate, lambda: None


def _make_pyhs3_compiled(
    config: Config,
    shared_data: np.ndarray,
) -> tuple[dict[str, Any], Callable[[float], float], Callable[[], None]]:
    """Prepare the compiled pyHS3 workflow for one NLL category.

    ``pointwise_nll`` compiles one complete objective function. The underlying
    PDF is still evaluated as a scalar function at every event, but ``jax.vmap``
    performs those scalar evaluations inside one compiled executable and the
    logarithm/reduction also stay inside that executable. Therefore one timed
    NLL evaluation means one JAX dispatch and one synchronization, not one of
    each per event.

    ``batched_full_dataset_nll`` passes the entire observable array directly to
    the jaxified PDF graph. This category demonstrates native array execution
    and remains a separate workflow/vectorization benchmark.
    """
    lifecycle, model, params = _prepare_pyhs3(config, shared_data)
    data_np = np.asarray(shared_data, dtype=np.float64)

    expression, expression_time = time_once(
        lambda: model.distributions[config.distribution],
        label="distribution expression lookup",
    )
    jaxified_pdf, jaxify_time = time_once(
        lambda: jaxify(expression),
        label="PyTensor-to-JAX graph conversion",
    )

    lifecycle["graph_lookup_seconds"] = float(expression_time)
    lifecycle["jaxify_seconds"] = float(jaxify_time)
    lifecycle["graph_preparation_seconds"] = float(expression_time + jaxify_time)
    lifecycle["compiled_input_names"] = list(jaxified_pdf.input_names)

    missing = set(jaxified_pdf.input_names) - set(params)
    if missing:
        raise KeyError(
            f"Compiled inputs missing from model defaults: {sorted(missing)}"
        )
    if config.observable_name not in jaxified_pdf.input_names:
        raise KeyError(
            f"Observable {config.observable_name!r} is not a compiled graph input"
        )
    if config.poi not in jaxified_pdf.input_names:
        raise KeyError(f"POI {config.poi!r} is not a compiled graph input")

    base_inputs = {name: jnp.asarray(params[name]) for name in jaxified_pdf.input_names}

    poi_template = base_inputs[config.poi]
    if poi_template.ndim > 0 and poi_template.size != 1:
        raise ValueError(
            f"POI {config.poi!r} must be scalar-like, got shape {poi_template.shape}"
        )

    observable_default = np.asarray(
        params[config.observable_name],
        dtype=np.float64,
    )
    observable_dtype = base_inputs[config.observable_name].dtype
    observable_scalar_shape = () if observable_default.ndim == 0 else (1,)

    def scalar_like(template: Any, value: Any, *, label: str) -> Any:
        array = jnp.asarray(value, dtype=template.dtype)
        if template.ndim == 0:
            return jnp.reshape(array, ())
        if template.size != 1:
            raise ValueError(
                f"{label} requires a scalar or one-element template, "
                f"got shape {template.shape}"
            )
        return jnp.reshape(array, template.shape)

    def scalar_poi(value: Any) -> Any:
        return scalar_like(poi_template, value, label="POI replacement")

    def scalar_observable(value: Any) -> Any:
        array = jnp.asarray(value, dtype=observable_dtype)
        return jnp.reshape(array, observable_scalar_shape)

    mu_template = scalar_poi(float(np.asarray(params[config.poi]).reshape(-1)[0]))

    if config.category == POINTWISE_NLL:
        # Compile one complete point-by-point NLL objective. The scalar PDF
        # graph is mapped over the shared dataset inside XLA, so the benchmark
        # preserves event-by-event semantics without paying Python/JAX dispatch
        # and synchronization overhead once per event.
        base_inputs[config.observable_name] = scalar_observable(data_np[0])
        data_jax = jnp.asarray(data_np, dtype=observable_dtype)

        def scalar_pdf_function(mu_value: Any, x_value: Any) -> Any:
            call_inputs = dict(base_inputs)
            call_inputs[config.poi] = scalar_poi(mu_value)
            call_inputs[config.observable_name] = scalar_observable(x_value)
            raw = jaxified_pdf(**call_inputs)[0]
            return jnp.ravel(jnp.asarray(raw))[0]

        def pointwise_full_nll(mu_value: Any) -> Any:
            pdf_values = jax.vmap(
                lambda x_value: scalar_pdf_function(mu_value, x_value)
            )(data_jax)
            valid = jnp.all(jnp.isfinite(pdf_values) & (pdf_values > 0.0))
            nll = -jnp.sum(jnp.log(pdf_values))
            return jnp.where(valid, nll, jnp.inf)

        jitted_pointwise_nll = jax.jit(pointwise_full_nll)

        def compile_pointwise_nll() -> Any:
            return jitted_pointwise_nll.lower(mu_template).compile()

        compiled_nll, compilation_time = time_once(
            compile_pointwise_nll,
            label="point-by-point JAX NLL lowering and XLA compilation",
        )

        def evaluate(mu: float) -> float:
            result = compiled_nll(scalar_poi(mu))
            result = jax.block_until_ready(result)
            value = float(np.asarray(result, dtype=np.float64).reshape(-1)[0])
            if not math.isfinite(value):
                raise ValueError(
                    f"Compiled point-by-point NLL returned invalid value: {value}"
                )
            return value

        compiled_execution_mode = (
            "one complete point-by-point NLL executable per mu; scalar PDF "
            "evaluation is vmapped over events inside XLA, followed by "
            "compiled log and reduction"
        )
        compiled_program_scope = "pointwise_full_nll"

    elif config.category == BATCHED_NLL:
        data_jax = jnp.asarray(data_np)

        def full_dataset_nll(mu_value: Any) -> Any:
            call_inputs = dict(base_inputs)
            call_inputs[config.poi] = scalar_poi(mu_value)
            call_inputs[config.observable_name] = data_jax

            raw = jaxified_pdf(**call_inputs)[0]
            pdf_values = jnp.ravel(jnp.asarray(raw))
            valid = jnp.all(jnp.isfinite(pdf_values) & (pdf_values > 0.0))
            nll = -jnp.sum(jnp.log(pdf_values))
            return jnp.where(valid, nll, jnp.inf)

        jitted_nll = jax.jit(full_dataset_nll)

        def compile_batched_nll() -> Any:
            return jitted_nll.lower(mu_template).compile()

        compiled_nll, compilation_time = time_once(
            compile_batched_nll,
            label="full-dataset JAX NLL lowering and XLA compilation",
        )

        def evaluate(mu: float) -> float:
            result = compiled_nll(scalar_poi(mu))
            result = jax.block_until_ready(result)
            value = float(np.asarray(result, dtype=np.float64).reshape(-1)[0])
            if not math.isfinite(value):
                raise ValueError(
                    f"Compiled full-dataset NLL returned invalid value: {value}"
                )
            return value

        compiled_execution_mode = (
            "complete dataset PDF evaluation, logarithm, and reduction "
            "compiled as one JAX program"
        )
        compiled_program_scope = "full_dataset_nll"

    else:  # pragma: no cover
        raise ValueError(config.category)

    lifecycle["compilation_seconds"] = float(compilation_time)
    lifecycle["compiled_execution_mode"] = compiled_execution_mode
    lifecycle["compiled_program_scope"] = compiled_program_scope
    lifecycle["jax_backend"] = jax.default_backend()
    lifecycle["jax_enable_x64"] = bool(jax.config.jax_enable_x64)

    return lifecycle, evaluate, lambda: None


def _make_roofit(
    config: Config,
    shared_params: dict[str, np.ndarray],
    shared_data: np.ndarray,
) -> tuple[dict[str, Any], Callable[[float], float], Callable[[], None]]:
    if config.category != POINTWISE_NLL:
        raise NotImplementedError(
            "RooFit is intentionally not assigned an equivalent array call "
            "for the batched workflow benchmark"
        )
    if ROOT is None or config.root_workspace_path is None:
        raise RuntimeError("RooFit requested but ROOT/root workspace is unavailable")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
    root_file, load_time = time_once(
        lambda: ROOT.TFile.Open(str(config.root_workspace_path), "READ"),
        label="ROOT file loading",
    )
    if not root_file or root_file.IsZombie():
        raise FileNotFoundError(config.root_workspace_path)

    try:
        root_workspace, model_time = time_once(
            lambda: _find_root_workspace(root_file),
            label="RooWorkspace lookup",
        )
        sync = _sync_root_parameters(
            root_workspace,
            shared_params,
            {config.observable_name, config.poi},
        )
        pdf = root_workspace.pdf(config.distribution)
        observable = root_workspace.var(config.observable_name)
        poi = root_workspace.var(config.poi)
        if any(obj is None or not bool(obj) for obj in (pdf, observable, poi)):
            raise KeyError(
                "Missing RooFit PDF, observable, or POI: "
                f"{config.distribution}, {config.observable_name}, {config.poi}"
            )
        norm_set = ROOT.RooArgSet(observable)
    except Exception:
        root_file.Close()
        raise

    data = np.asarray(shared_data, dtype=np.float64)

    def evaluate(mu: float) -> float:
        poi.setVal(float(mu))
        total = 0.0
        for x in data:
            observable.setVal(float(x))
            value = finite_scalar(
                pdf.getVal(norm_set),
                label="RooFit normalized PDF",
            )
            if value <= 0.0:
                raise ValueError(f"Non-positive RooFit PDF value: {value}")
            total -= math.log(value)
        return total

    lifecycle = {
        "workspace_loading_seconds": load_time,
        "model_construction_seconds": model_time,
        "graph_lookup_seconds": 0.0,
        "jaxify_seconds": 0.0,
        "graph_preparation_seconds": 0.0,
        "compilation_seconds": 0.0,
        "compiled_input_names": [],
        "synchronized_root_parameters": sync["synchronized"],
        "transformed_root_parameters": sync["transformed"],
        "skipped_root_parameters": sync["skipped"],
    }
    return lifecycle, evaluate, root_file.Close


def _scan_timing(
    evaluate: Callable[[float], float],
    mu_values: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("scan_repeats must be positive")
    if mu_values.size < 1 or not np.all(np.isfinite(mu_values)):
        raise ValueError("mu_values must be a non-empty finite array")

    samples: list[float] = []
    last_values: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        last_values = [float(evaluate(float(mu))) for mu in mu_values]
        elapsed = time.perf_counter() - start
        if elapsed <= 0.0 or not math.isfinite(elapsed):
            raise RuntimeError(f"Invalid full-scan timing: {elapsed}")
        samples.append(elapsed)

    median = statistics.median(samples)
    delta = delta_curve(last_values)
    return {
        "full_scan_time_seconds_samples": [float(v) for v in samples],
        "full_scan_time_seconds_median": float(median),
        "full_scan_time_seconds_mean": float(statistics.mean(samples)),
        "full_scan_time_seconds_std": float(
            statistics.stdev(samples) if len(samples) > 1 else 0.0
        ),
        "time_per_scan_point_seconds": float(median / mu_values.size),
        "scan_throughput_points_per_second": float(mu_values.size / median),
        "nll_values": [float(v) for v in last_values],
        "delta_nll_values": [float(v) for v in delta],
        "minimum_mu": float(mu_values[int(np.argmin(last_values))]),
        "minimum_nll": float(np.min(last_values)),
    }


def run_engine(
    config: Config,
    reference_delta: np.ndarray | None,
    shared_params: dict[str, np.ndarray],
    shared_data: np.ndarray,
) -> dict[str, Any]:
    """Run one engine/category in the current (already isolated) process."""
    gc.collect()
    rss_before = current_rss_mb()
    peak_before = peak_rss_mb()

    if config.engine == PYHS3_NONCOMPILED:
        lifecycle, evaluate, cleanup = _make_pyhs3_noncompiled(config, shared_data)
    elif config.engine == PYHS3_COMPILED:
        lifecycle, evaluate, cleanup = _make_pyhs3_compiled(config, shared_data)
    elif config.engine == ROOFIT:
        lifecycle, evaluate, cleanup = _make_roofit(
            config,
            shared_params,
            shared_data,
        )
    else:
        raise ValueError(f"Unsupported engine: {config.engine}")

    try:
        first_output, first_time = time_once(
            lambda: evaluate(float(config.mu_values[0])),
            label="first NLL evaluation",
        )
        model_to_first = (
            sum(
                float(lifecycle.get(key, 0.0))
                for key in (
                    "model_construction_seconds",
                    "graph_preparation_seconds",
                    "compilation_seconds",
                )
            )
            + first_time
        )
        cold_start = (
            float(lifecycle.get("workspace_loading_seconds", 0.0)) + model_to_first
        )
        lifecycle["first_call_seconds"] = first_time
        lifecycle["model_to_first_evaluation_seconds"] = float(model_to_first)
        lifecycle["cold_start_end_to_end_seconds"] = float(cold_start)
        # Backward-compatible alias, explicitly defined as cold-start.
        lifecycle["end_to_end_first_evaluation_seconds"] = float(cold_start)

        def indexed(index: int) -> float:
            return evaluate(float(config.mu_values[index % config.mu_values.size]))

        steady = benchmark_batches(
            indexed,
            batch_size=config.batch_size,
            n_batches=config.n_batches,
            warmup_batches=config.warmup_batches,
        )
        scan = _scan_timing(evaluate, config.mu_values, config.scan_repeats)
    finally:
        cleanup()

    delta = np.asarray(scan["delta_nll_values"], dtype=np.float64)
    validation = (
        {}
        if reference_delta is None
        else agreement_arrays(
            delta, reference_delta, rtol=config.rtol, atol=config.atol
        )
    )

    gc.collect()
    current_after = current_rss_mb()
    peak_after = peak_rss_mb()
    status = (
        "success"
        if validation.get("validation_status", "success") == "success"
        else "validation_failed"
    )

    return {
        "benchmark": BENCHMARK_NAME,
        "category": (
            "Point-by-point NLL"
            if config.category == POINTWISE_NLL
            else "Batched full-dataset evaluation"
        ),
        "category_key": config.category,
        "comparison_class": (
            "same-objective cross-engine comparison: one complete NLL evaluation at one mu using the same scalar PDF, dataset, event count, and parameter point"
            if config.category == POINTWISE_NLL
            else "workflow benchmark; demonstrates native array execution/vectorization and is not a pure RooFit-equivalent microbenchmark"
        ),
        "execution_path": lifecycle.get(
            "compiled_execution_mode",
            (
                "event-by-event scalar PDF evaluation with host-side log/reduction"
                if config.category == POINTWISE_NLL
                else "full observable array passed to the pyHS3 PDF graph"
            ),
        ),
        "nll_definition": "-sum(log(normalized PDF(event | mu))) for the selected dataset and distribution; this is not the full constrained statistical-model likelihood",
        "measurement_isolation": "fresh_spawned_process",
        "engine": config.engine,
        "engine_label": ENGINE_LABELS[config.engine],
        "workspace": config.workspace_path.name,
        "workspace_path": str(config.workspace_path),
        "root_workspace_path": (
            str(config.root_workspace_path) if config.root_workspace_path else None
        ),
        "workspace_label": config.workspace_path.stem,
        "analysis": config.analysis,
        "distribution": config.distribution,
        "data_name": config.data_name,
        "observable_name": config.observable_name,
        "observable_index": config.observable_index,
        "poi": config.poi,
        "mode": config.mode,
        "n_events": int(shared_data.size),
        "n_scan_points": int(config.mu_values.size),
        "mu_values": [float(v) for v in config.mu_values],
        "first_output": float(first_output),
        "current_rss_before_mb": rss_before,
        "current_rss_after_mb": current_after,
        "current_rss_delta_mb": max(0.0, current_after - rss_before),
        "peak_rss_before_mb": peak_before,
        "peak_rss_after_mb": peak_after,
        "peak_rss_delta_mb": max(0.0, peak_after - peak_before),
        "status": status,
        **lifecycle,
        **steady,
        **scan,
        **validation,
    }


def _isolated_engine_worker(
    config: Config,
    reference_delta: np.ndarray | None,
    shared_params: dict[str, np.ndarray],
    shared_data: np.ndarray,
    result_queue: Any,
) -> None:
    try:
        result_queue.put(
            {
                "ok": True,
                "row": run_engine(
                    config,
                    reference_delta,
                    shared_params,
                    shared_data,
                ),
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def run_engine_isolated(
    config: Config,
    reference_delta: np.ndarray | None,
    shared_params: dict[str, np.ndarray],
    shared_data: np.ndarray,
) -> dict[str, Any]:
    """Run exactly one engine/category in a fresh spawned process."""
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_isolated_engine_worker,
        args=(
            config,
            reference_delta,
            shared_params,
            shared_data,
            result_queue,
        ),
    )
    process.start()
    process.join()

    try:
        payload = result_queue.get(timeout=5.0)
    except queue.Empty as exc:
        raise RuntimeError(
            "Isolated NLL benchmark worker produced no result "
            f"(exit code {process.exitcode})"
        ) from exc
    finally:
        result_queue.close()
        result_queue.join_thread()

    if process.exitcode not in (0, None) and payload.get("ok"):
        raise RuntimeError(
            f"Isolated worker exited with code {process.exitcode} after returning data"
        )
    if not payload.get("ok"):
        raise RuntimeError(
            f"{payload.get('error_type', 'WorkerError')}: "
            f"{payload.get('error_message', 'unknown worker failure')}\n"
            f"{payload.get('traceback', '')}"
        )
    return dict(payload["row"])


def _workspace_title(path: str) -> str:
    return Path(path).stem.replace("_", " / ")


def _workspace_multiline_label(workspace_label: str) -> str:
    return "\n".join(Path(workspace_label).stem.split("_"))


def _category_short_label(category: str) -> str:
    if category == POINTWISE_NLL:
        return "pointwise NLL"
    if category == BATCHED_NLL:
        return "batched NLL"
    return category.replace("_", " ")


def _engine_short_label(engine: str) -> str:
    if engine == PYHS3_NONCOMPILED:
        return "pyHS3 non-compiled"
    if engine == PYHS3_COMPILED:
        return "pyHS3 compiled"
    if engine == ROOFIT:
        return "RooFit"
    return engine


def _bar_label(row: dict[str, Any], *, include_engine: bool = True) -> str:
    parts = [
        _workspace_multiline_label(row["workspace_label"]),
        _category_short_label(row["category_key"]),
    ]
    if include_engine:
        parts.append(_engine_short_label(row["engine"]))
    return "\n".join(parts)


def _successful_rows(
    results: list[dict[str, Any]],
    category: str | None = None,
) -> list[dict[str, Any]]:
    return [
        row
        for row in results
        if row.get("status") == "success"
        and (category is None or row.get("category_key") == category)
    ]


def plot_scan_agreement(results: list[dict[str, Any]], output: Path) -> None:
    rows = _successful_rows(results, POINTWISE_NLL)
    workspaces = list(dict.fromkeys(row["workspace"] for row in rows))
    if not workspaces:
        return

    fig, axes = plt.subplots(
        len(workspaces),
        1,
        figsize=(14, 4.8 * len(workspaces)),
        squeeze=False,
    )
    fig.suptitle(
        "Cross-engine point-by-point ΔNLL agreement", fontsize=23, weight="bold"
    )
    for ax, workspace in zip(axes[:, 0], workspaces, strict=True):
        for engine in SUPPORTED_ENGINES:
            row = next(
                (
                    item
                    for item in rows
                    if item["workspace"] == workspace and item["engine"] == engine
                ),
                None,
            )
            if row is None:
                continue
            ax.plot(
                row["mu_values"],
                row["delta_nll_values"],
                label=ENGINE_LABELS[engine],
                color=ENGINE_COLORS[engine],
                **ENGINE_STYLE[engine],
            )
        ax.set_title(_workspace_title(workspace), loc="left", weight="bold")
        ax.set_xlabel("Signal strength μ")
        ax.set_ylabel("ΔNLL")
        style_axes(ax)
        ax.legend(frameon=False)
    save_figure(fig, output)


def plot_runtime(results: list[dict[str, Any]], output: Path) -> None:
    rows = _successful_rows(results)
    if not rows:
        return
    labels = [_bar_label(row) for row in rows]
    values = [float(row["steady_state_seconds_median"]) * 1e6 for row in rows]
    colors = [ENGINE_COLORS[row["engine"]] for row in rows]

    fig, ax = plt.subplots(figsize=(max(18, len(rows) * 1.65), 10))
    bars = ax.bar(np.arange(len(rows)), values, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Median time per complete NLL evaluation [µs]")
    ax.set_title("Cross-engine NLL steady-state runtime", fontsize=23, weight="bold")
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.tick_params(axis="x", pad=10)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    style_axes(ax, grid_axis="y")
    save_figure(fig, output)


def plot_memory(results: list[dict[str, Any]], output: Path) -> None:
    rows = _successful_rows(results)
    if not rows:
        return
    labels = [_bar_label(row) for row in rows]
    current = [float(row["current_rss_delta_mb"]) for row in rows]
    peak = [float(row["peak_rss_delta_mb"]) for row in rows]
    x = np.arange(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(18, len(rows) * 1.65), 10))
    ax.bar(x - width / 2, current, width, label="Current RSS delta")
    ax.bar(x + width / 2, peak, width, label="Peak RSS delta")
    ax.set_ylabel("Memory delta [MiB]")
    ax.set_title("Fresh-process NLL memory profile", fontsize=23, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.tick_params(axis="x", pad=10)
    style_axes(ax, grid_axis="y")
    ax.legend(frameon=False)
    save_figure(fig, output)


def plot_compiled_lifecycle(results: list[dict[str, Any]], output: Path) -> None:
    rows = [row for row in _successful_rows(results) if row["engine"] == PYHS3_COMPILED]
    if not rows:
        return

    labels = [_bar_label(row, include_engine=False) for row in rows]
    keys = [
        ("model_construction_seconds", "Model construction"),
        ("graph_preparation_seconds", "Graph preparation"),
        ("compilation_seconds", "Compilation"),
        ("first_call_seconds", "First call"),
    ]

    x = np.arange(len(rows))
    stacked_values = np.asarray(
        [[float(row.get(key, 0.0)) * 1e3 for key, _ in keys] for row in rows],
        dtype=np.float64,
    )
    totals = np.sum(stacked_values, axis=1)
    bottom = np.zeros(len(rows), dtype=np.float64)

    fig, ax = plt.subplots(figsize=(max(15, len(rows) * 3.2), 10))

    for key_index, (_, label) in enumerate(keys):
        values = stacked_values[:, key_index]
        bars = ax.bar(x, values, bottom=bottom, label=label)

        for row_index, (bar, value) in enumerate(zip(bars, values, strict=True)):
            total = totals[row_index]
            percentage = 0.0 if total <= 0.0 else 100.0 * value / total

            # Show percentages only for large sections to avoid overlap.
            if percentage >= 8.0 and value > 0.0:
                center = bottom[row_index] + value / 2.0
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    center,
                    f"{percentage:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                )

        bottom += values

    ax.set_yscale("log")
    ax.set_ylabel("Wall time [ms]")
    ax.set_title("pyHS3 compiled NLL lifecycle", fontsize=23, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.tick_params(axis="x", pad=10)
    style_axes(ax, grid_axis="y")
    ax.legend(frameon=False)
    save_figure(fig, output)


def plot_end_to_end_vs_steady(results: list[dict[str, Any]], output: Path) -> None:
    """Compare cold-start end-to-end latency with warm objective latency."""
    rows = _successful_rows(results)
    if not rows:
        return

    labels = [_bar_label(row) for row in rows]
    cold_ms = [float(row["cold_start_end_to_end_seconds"]) * 1e3 for row in rows]
    steady_ms = [float(row["steady_state_seconds_median"]) * 1e3 for row in rows]
    x = np.arange(len(rows))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(18, len(rows) * 1.65), 10))
    ax.bar(x - width / 2, cold_ms, width, label="End-to-end cold start")
    ax.bar(x + width / 2, steady_ms, width, label="Steady-state evaluation")
    ax.set_yscale("log")
    ax.set_ylabel("Wall time per complete NLL evaluation [ms]")
    ax.set_title("NLL end-to-end vs steady-state latency", fontsize=23, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.tick_params(axis="x", pad=10)
    style_axes(ax, grid_axis="y")
    ax.legend(frameon=False)
    save_figure(fig, output)


def _failure_row(config: Config, exc: Exception) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "category_key": config.category,
        "engine": config.engine,
        "engine_label": ENGINE_LABELS[config.engine],
        "workspace": config.workspace_path.name,
        "workspace_path": str(config.workspace_path),
        "root_workspace_path": (
            str(config.root_workspace_path) if config.root_workspace_path else None
        ),
        "workspace_label": config.workspace_path.stem,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspaces", nargs="+", type=Path, default=DEFAULT_WORKSPACES
    )
    parser.add_argument("--root-workspaces", nargs="+", type=Path)
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=SUPPORTED_ENGINES,
        default=list(SUPPORTED_ENGINES),
    )
    parser.add_argument(
        "--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES)
    )
    parser.add_argument("--analysis", default="L_ch0")
    parser.add_argument("--distribution", default="model_ch0")
    parser.add_argument("--data-name", default="combData_ch0")
    parser.add_argument("--observable-name", default="x")
    parser.add_argument("--observable-index", type=int, default=0)
    parser.add_argument("--poi", default="mu_sig")
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--mu-min", type=float, default=0.0)
    parser.add_argument("--mu-max", type=float, default=5.0)
    parser.add_argument("--n-mu-values", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-batches", type=int, default=9)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--scan-repeats", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / BENCHMARK_NAME / "result.json"
    )
    parser.add_argument("--plot-dir", type=Path, default=PLOTS_DIR / BENCHMARK_NAME)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_mu_values < 2:
        raise ValueError("--n-mu-values must be at least 2")
    if not np.isfinite(args.mu_min) or not np.isfinite(args.mu_max):
        raise ValueError("mu bounds must be finite")
    if args.mu_max <= args.mu_min:
        raise ValueError("--mu-max must be greater than --mu-min")
    if args.batch_size < 1 or args.n_batches < 1 or args.warmup_batches < 0:
        raise ValueError("Invalid steady-state batch configuration")
    if args.scan_repeats < 1:
        raise ValueError("--scan-repeats must be positive")
    if args.observable_index < 0:
        raise ValueError("--observable-index must be non-negative")
    if args.rtol < 0.0 or args.atol < 0.0:
        raise ValueError("Validation tolerances must be non-negative")


def main() -> None:
    args = parse_args()
    _validate_args(args)

    root_paths = args.root_workspaces or [
        workspace.with_suffix(".root") for workspace in args.workspaces
    ]
    if len(root_paths) != len(args.workspaces):
        raise ValueError(
            "--root-workspaces must contain one path per --workspaces entry"
        )

    mu_values = np.linspace(
        args.mu_min, args.mu_max, args.n_mu_values, dtype=np.float64
    )
    results: list[dict[str, Any]] = []
    reference_curves: dict[str, Any] = {}

    engine_order = [engine for engine in SUPPORTED_ENGINES if engine in args.engines]

    for workspace, root_workspace in zip(args.workspaces, root_paths, strict=True):
        base_config = Config(
            engine=PYHS3_NONCOMPILED,
            category=POINTWISE_NLL,
            workspace_path=workspace,
            root_workspace_path=root_workspace,
            analysis=args.analysis,
            distribution=args.distribution,
            data_name=args.data_name,
            observable_name=args.observable_name,
            observable_index=args.observable_index,
            poi=args.poi,
            mode=args.mode,
            mu_values=mu_values,
            batch_size=args.batch_size,
            n_batches=args.n_batches,
            warmup_batches=args.warmup_batches,
            scan_repeats=args.scan_repeats,
            rtol=args.rtol,
            atol=args.atol,
        )
        try:
            shared_params, shared_data = _shared_inputs(base_config)
        except Exception as exc:
            for category in args.categories:
                for engine in engine_order:
                    config = dataclass_replace(
                        base_config, engine=engine, category=category
                    )
                    results.append(_failure_row(config, exc))
            continue

        for category in args.categories:
            reference_delta: np.ndarray | None = None
            for engine in engine_order:
                if category == BATCHED_NLL and engine == ROOFIT:
                    continue

                config = dataclass_replace(
                    base_config, engine=engine, category=category
                )
                try:
                    row = run_engine_isolated(
                        config,
                        reference_delta,
                        shared_params,
                        shared_data,
                    )
                    if reference_delta is None:
                        reference_delta = np.asarray(
                            row["delta_nll_values"], dtype=np.float64
                        )
                        reference_curves[f"{workspace}:{category}"] = {
                            "reference_engine": engine,
                            "mu_values": [float(v) for v in mu_values],
                            "delta_nll_values": [float(v) for v in reference_delta],
                        }
                except Exception as exc:
                    row = _failure_row(config, exc)

                results.append(row)
                print(
                    f"{workspace.name} / {category} / {ENGINE_LABELS[engine]}: {row['status']}"
                )

    required = list(results)
    summary = {
        "n_results": len(results),
        "n_success": sum(row.get("status") == "success" for row in results),
        "n_validation_failed": sum(
            row.get("status") == "validation_failed" for row in results
        ),
        "n_failed": sum(row.get("status") == "failed" for row in results),
        "all_required_runs_passed": bool(required)
        and all(row.get("status") == "success" for row in required),
    }

    payload = {
        "benchmark": BENCHMARK_NAME,
        "methodology": {
            "pointwise_nll": (
                "same complete NLL objective at one mu using the same scalar "
                "normalized PDF, dataset, event count, and parameter point. "
                "RooFit and non-compiled pyHS3 use event-by-event host loops; "
                "compiled pyHS3 vmaps the same scalar PDF over events and "
                "performs log/reduction inside one precompiled executable, so "
                "steady-state timing contains one dispatch per NLL evaluation"
            ),
            "batched_full_dataset_nll": (
                "separate workflow benchmark: pyHS3 receives the whole "
                "observable array; compiled pyHS3 executes PDF, log, and "
                "reduction in one JAX program"
            ),
            "batched_roofit_policy": "unsupported by design; no equivalent array call is claimed",
            "measurement_isolation": "one fresh spawned process per workspace/category/engine",
            "compiled_timing": (
                "pointwise compiles one complete vmapped scalar-PDF NLL "
                "executable once; batched compiles one native array-input NLL "
                "executable once. Graph preparation, XLA compilation, first "
                "execution, and steady state are reported separately, and "
                "compilation is excluded from all steady-state sections"
            ),
            "end_to_end_metrics": (
                "model_to_first_evaluation excludes workspace loading; "
                "cold_start_end_to_end includes workspace loading"
            ),
            "validation": "delta-NLL curves are compared to the first successful engine in each workspace/category",
        },
        "configuration": vars(args)
        | {
            "workspaces": [str(path) for path in args.workspaces],
            "root_workspaces": [str(path) for path in root_paths],
            "output": str(args.output),
            "plot_dir": str(args.plot_dir),
        },
        "reference_curves": reference_curves,
        "results": results,
        "summary": summary,
    }
    save_json(payload, args.output)

    if args.plot:
        plot_scan_agreement(results, args.plot_dir / "cross_nll_scan_agreement.png")
        plot_runtime(results, args.plot_dir / "cross_nll_steady_state_runtime.png")
        plot_memory(results, args.plot_dir / "cross_nll_memory_profile.png")
        plot_compiled_lifecycle(
            results, args.plot_dir / "cross_nll_compiled_lifecycle.png"
        )
        plot_end_to_end_vs_steady(
            results, args.plot_dir / "cross_nll_end_to_end_vs_steady.png"
        )

    if not summary["all_required_runs_passed"]:
        raise SystemExit(
            "One or more required NLL benchmark runs or validations failed; "
            "inspect the JSON output."
        )


def dataclass_replace(config: Config, **changes: Any) -> Config:
    """Local typed replacement helper, avoiding mutation of frozen Config."""
    values = {field: getattr(config, field) for field in config.__dataclass_fields__}
    values.update(changes)
    return Config(**values)


if __name__ == "__main__":
    main()
