"""
Scalar PDF benchmark for pyHS3 and RooFit.

Each engine evaluates the same normalized scalar PDF at the same observable
values and synchronized parameter point.  The default ``varying`` mode changes x on every call. The optional ``fixed``
mode is retained only as an explicit cache diagnostic and is not part of the
primary performance comparison.
The compiled engine explicitly lowers and XLA-compiles one scalar PDF
function before first-call and steady-state timing. Graph preparation, XLA
compilation, first execution, and steady-state execution are reported
separately. Compilation and first-call cost are excluded from steady-state
timings.
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import queue
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
except ImportError:
    ROOT = None

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.benchmark_modes import (
        ENGINE_LABELS,
        FIXED_INPUT,
        INPUT_MODES,
        PYHS3_COMPILED,
        PYHS3_NONCOMPILED,
        ROOFIT,
        VARYING_INPUT,
    )
    from src.config import DEFAULT_MODE, PLOTS_DIR, RESULTS_DIR
    from src.cross_benchmark_utils import (
        agreement_arrays,
        benchmark_scaling,
        current_rss_mb,
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
        FIXED_INPUT,
        INPUT_MODES,
        PYHS3_COMPILED,
        PYHS3_NONCOMPILED,
        ROOFIT,
        VARYING_INPUT,
    )
    from .config import DEFAULT_MODE, PLOTS_DIR, RESULTS_DIR
    from .cross_benchmark_utils import (
        agreement_arrays,
        benchmark_scaling,
        current_rss_mb,
        finite_scalar,
        peak_rss_mb,
        save_figure,
        save_json,
        style_axes,
        time_once,
    )

BENCHMARK_NAME = "cross_scalar_pdf"
SUPPORTED_ENGINES = (PYHS3_NONCOMPILED, PYHS3_COMPILED, ROOFIT)
DEFAULT_WORKSPACES = [
    Path("inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json"),
    Path("inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json"),
    Path("inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json"),
]
DEFAULT_N_EVALUATIONS = [1, 10, 100, 1000, 10000]
ENGINE_COLORS = {
    PYHS3_NONCOMPILED: "#1565C0",
    PYHS3_COMPILED: "#EF6C00",
    ROOFIT: "#00897B",
}
ENGINE_STYLE = {
    PYHS3_NONCOMPILED: dict(
        marker="s",
        linestyle="-",
        color=ENGINE_COLORS[PYHS3_NONCOMPILED],
        linewidth=2.2,
        markersize=7,
    ),
    PYHS3_COMPILED: dict(
        marker="o",
        linestyle="--",
        color=ENGINE_COLORS[PYHS3_COMPILED],
        linewidth=2.2,
        markersize=7,
    ),
    ROOFIT: dict(
        marker="D",
        linestyle="-.",
        color=ENGINE_COLORS[ROOFIT],
        linewidth=2.2,
        markersize=7,
    ),
}

LIFECYCLE_COLORS = {
    "model_construction_seconds": "#1565C0",
    "graph_preparation_seconds": "#00ACC1",
    "compilation_seconds": "#EF6C00",
    "first_call_seconds": "#D81B60",
}


def _mode_label(mode: str) -> str:
    if mode == VARYING_INPUT:
        return "Changing observable"
    if mode == FIXED_INPUT:
        return "Fixed observable (cache diagnostic)"
    return mode.replace("_", " ").title()


def _style_bars(bars: Any) -> None:
    for bar in bars:
        bar.set_edgecolor("white")
        bar.set_linewidth(1.0)
        bar.set_alpha(1.0)


@dataclass(frozen=True)
class Config:
    engine: str
    workspace_path: Path
    root_workspace_path: Path | None
    target: str
    mode: str
    distribution: str
    observable_name: str
    input_mode: str
    n_evaluations: tuple[int, ...]
    timing_repeats: int
    warmup_evaluations: int
    validation_points: int
    rtol: float
    atol: float


def _model_defaults(model: Any) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for source in (getattr(model, "data", {}), getattr(model, "free_params", {})):
        for name, value in source.items():
            values[name] = np.asarray(value, dtype=np.float64)
    return values


def _set_scalar(container: dict[str, np.ndarray], name: str, value: float) -> None:
    original = np.asarray(container[name])
    container[name] = (
        np.asarray([value], dtype=np.float64)
        if original.ndim > 0
        else np.asarray(value, dtype=np.float64)
    )


def _shared_inputs(config: Config) -> tuple[dict[str, np.ndarray], np.ndarray]:
    workspace = Workspace.load(config.workspace_path)
    model = workspace.model(config.target, progress=False, mode=config.mode)
    params = _model_defaults(model)
    if config.observable_name not in params:
        raise KeyError(
            f"Observable {config.observable_name!r} missing; available: {sorted(params)}"
        )
    raw = np.asarray(params[config.observable_name], dtype=np.float64).reshape(-1)
    if raw.size >= 2 and np.ptp(raw) > 0.0:
        low, high = float(np.min(raw)), float(np.max(raw))
    elif raw.size:
        center = float(raw[0])
        low, high = center - 1.0, center + 1.0
    else:
        low, high = -1.0, 1.0
    values = np.linspace(low, high, config.validation_points, dtype=np.float64)
    return params, values


def _find_root_workspace(root_file: Any) -> Any:
    for key in root_file.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom(ROOT.RooWorkspace.Class()):
            return obj
    raise KeyError("No RooWorkspace found")


def _sync_root_parameters(
    root_workspace: Any,
    params: dict[str, np.ndarray],
    observable_name: str,
) -> dict[str, Any]:
    """Synchronize the RooFit parameter point with pyHS3 conventions.

    Generated RooExponential workspaces use opposite slope conventions:
    pyHS3 stores a positive decay rate ``tau`` for ``exp(-tau * x)``, while
    RooFit's ``RooExponential`` stores the coefficient of ``exp(tau * x)``.
    Therefore a positive pyHS3 ``tau_*`` maps to the negative RooFit value.

    The transformation is applied only when the direct value is outside the
    RooRealVar range and its negation is inside that range. This avoids
    hard-coding a sign flip for unrelated parameters that happen to contain
    ``tau`` in their name.
    """
    synchronized: list[str] = []
    transformed: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    for name, value in params.items():
        if name == observable_name:
            skipped[name] = "observable is controlled by the benchmark"
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
            minimum = float("-inf")
            maximum = float("inf")

        target_value = source_value
        transformation = None

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


def _prepare_engine(
    config: Config,
    shared_params: dict[str, np.ndarray],
    shared_values: np.ndarray,
) -> tuple[
    dict[str, Any],
    Callable[[int], float],
    Callable[[np.ndarray], np.ndarray],
    Callable[[], None],
]:
    """Prepare one engine inside a fresh worker process.

    ``shared_params`` and ``shared_values`` are prepared before spawning
    workers so that input construction is not accidentally included in
    any engine lifecycle measurement.  RSS baselines are taken before
    loading the engine workspace/model in this worker.
    """
    rss_before = current_rss_mb()
    peak_before = peak_rss_mb()

    if config.engine in (PYHS3_NONCOMPILED, PYHS3_COMPILED):
        workspace, load_time = time_once(
            lambda: Workspace.load(config.workspace_path), label="workspace loading"
        )
        model, model_time = time_once(
            lambda: workspace.model(config.target, progress=False, mode=config.mode),
            label="model construction",
        )
        params = _model_defaults(model)
        lifecycle = {
            "workspace_loading_seconds": load_time,
            "model_construction_seconds": model_time,
            "graph_lookup_seconds": 0.0,
            "jaxify_seconds": 0.0,
            "graph_preparation_seconds": 0.0,
            "compilation_seconds": 0.0,
        }

        if config.engine == PYHS3_NONCOMPILED:

            def scalar(index: int) -> float:
                x = (
                    shared_values[0]
                    if config.input_mode == FIXED_INPUT
                    else shared_values[index % shared_values.size]
                )
                _set_scalar(params, config.observable_name, float(x))
                return finite_scalar(
                    model.pdf(config.distribution, **params),
                    label="pyHS3 non-compiled PDF",
                )

            def evaluate_grid(values: np.ndarray) -> np.ndarray:
                return np.asarray(
                    [
                        scalar(i)
                        if config.input_mode == VARYING_INPUT
                        else _evaluate_noncompiled_at(model, params, config, float(x))
                        for i, x in enumerate(values)
                    ],
                    dtype=np.float64,
                )

            compiled_names: list[str] = []
        else:
            expression, expression_time = time_once(
                lambda: model.distributions[config.distribution],
                label="distribution graph lookup",
            )
            jaxified_pdf, jaxify_time = time_once(
                lambda: jaxify(expression),
                label="PyTensor-to-JAX graph conversion",
            )

            lifecycle["graph_lookup_seconds"] = float(expression_time)
            lifecycle["jaxify_seconds"] = float(jaxify_time)
            lifecycle["graph_preparation_seconds"] = float(
                expression_time + jaxify_time
            )

            missing = set(jaxified_pdf.input_names) - set(params)
            if missing:
                raise KeyError(
                    f"Compiled inputs missing from model defaults: {sorted(missing)}"
                )
            if config.observable_name not in jaxified_pdf.input_names:
                raise KeyError(
                    f"Observable {config.observable_name!r} is not a compiled graph input"
                )

            base_inputs = {
                name: jnp.asarray(params[name]) for name in jaxified_pdf.input_names
            }

            # The model default for the observable can be the full dataset
            # (for example shape (290,) or (26,)).  This benchmark evaluates
            # one scalar observable at a time, exactly like _set_scalar() in
            # the non-compiled path.  Therefore the compiled observable input
            # must be scalar-shaped: () for a scalar default, otherwise (1,).
            observable_default = np.asarray(
                params[config.observable_name],
                dtype=np.float64,
            )
            observable_dtype = base_inputs[config.observable_name].dtype
            observable_scalar_shape = () if observable_default.ndim == 0 else (1,)

            def scalar_observable(value: Any) -> Any:
                array = jnp.asarray(value, dtype=observable_dtype)
                return jnp.reshape(array, observable_scalar_shape)

            # Ensure the closed-over base dictionary itself also carries the
            # scalar observable shape.  The value is replaced on every call.
            base_inputs[config.observable_name] = scalar_observable(
                float(shared_values[0])
            )

            def scalar_pdf_function(x_value: Any) -> Any:
                call_inputs = dict(base_inputs)
                call_inputs[config.observable_name] = scalar_observable(x_value)
                raw = jaxified_pdf(**call_inputs)[0]
                value = jnp.ravel(jnp.asarray(raw))[0]
                return jnp.where(
                    jnp.isfinite(value) & (value > 0.0),
                    value,
                    jnp.nan,
                )

            jitted_pdf = jax.jit(scalar_pdf_function)
            x_template = scalar_observable(float(shared_values[0]))

            def compile_scalar_pdf() -> Any:
                lowered = jitted_pdf.lower(x_template)
                return lowered.compile()

            compiled_pdf, compilation_time = time_once(
                compile_scalar_pdf,
                label="scalar JAX PDF lowering and XLA compilation",
            )

            lifecycle["compilation_seconds"] = float(compilation_time)
            lifecycle["compiled_program_scope"] = "scalar_pdf"
            lifecycle["compiled_execution_mode"] = (
                "one explicitly XLA-compiled scalar PDF function taking only x"
            )
            lifecycle["jax_backend"] = jax.default_backend()
            lifecycle["jax_enable_x64"] = bool(jax.config.jax_enable_x64)

            def evaluate_compiled_x(x: float) -> float:
                x_input = scalar_observable(x)
                result = compiled_pdf(x_input)
                result = jax.block_until_ready(result)
                value = float(np.asarray(result, dtype=np.float64).reshape(-1)[0])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"pyHS3 compiled PDF returned invalid value: {value}"
                    )
                return value

            def scalar(index: int) -> float:
                x = (
                    shared_values[0]
                    if config.input_mode == FIXED_INPUT
                    else shared_values[index % shared_values.size]
                )
                return evaluate_compiled_x(float(x))

            def evaluate_grid(values: np.ndarray) -> np.ndarray:
                return np.asarray(
                    [evaluate_compiled_x(float(x)) for x in values],
                    dtype=np.float64,
                )

            compiled_names = list(jaxified_pdf.input_names)

        first_output, first_time = time_once(
            lambda: scalar(0), label="first engine call"
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
        lifecycle.update(
            {
                "first_call_seconds": first_time,
                "model_to_first_evaluation_seconds": float(model_to_first),
                "cold_start_end_to_end_seconds": float(cold_start),
                # Backward-compatible alias, explicitly defined as cold-start.
                "end_to_end_first_evaluation_seconds": float(cold_start),
                "first_output": first_output,
                "compiled_input_names": compiled_names,
            }
        )

        def cleanup() -> None:
            pass

    elif config.engine == ROOFIT:
        if ROOT is None or config.root_workspace_path is None:
            raise RuntimeError(
                "RooFit requested but ROOT/root workspace is unavailable"
            )
        ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
        root_file, load_time = time_once(
            lambda: ROOT.TFile.Open(str(config.root_workspace_path), "READ"),
            label="ROOT file loading",
        )
        if not root_file or root_file.IsZombie():
            raise FileNotFoundError(config.root_workspace_path)
        root_workspace, model_time = time_once(
            lambda: _find_root_workspace(root_file), label="RooWorkspace lookup"
        )
        sync_summary = _sync_root_parameters(
            root_workspace, shared_params, config.observable_name
        )
        pdf = root_workspace.pdf(config.distribution)
        observable = root_workspace.var(config.observable_name)
        if pdf is None or not bool(pdf) or observable is None or not bool(observable):
            root_file.Close()
            raise KeyError(
                f"Missing RooFit PDF/observable: {config.distribution}, {config.observable_name}"
            )
        norm_set = ROOT.RooArgSet(observable)

        def scalar(index: int) -> float:
            x = (
                shared_values[0]
                if config.input_mode == FIXED_INPUT
                else shared_values[index % shared_values.size]
            )
            observable.setVal(float(x))
            return finite_scalar(pdf.getVal(norm_set), label="RooFit normalized PDF")

        def evaluate_grid(values: np.ndarray) -> np.ndarray:
            output = []
            for x in values:
                observable.setVal(float(x))
                output.append(
                    finite_scalar(pdf.getVal(norm_set), label="RooFit normalized PDF")
                )
            return np.asarray(output, dtype=np.float64)

        first_output, first_time = time_once(
            lambda: scalar(0), label="first RooFit call"
        )
        lifecycle = {
            "workspace_loading_seconds": load_time,
            "model_construction_seconds": model_time,
            "graph_lookup_seconds": 0.0,
            "jaxify_seconds": 0.0,
            "graph_preparation_seconds": 0.0,
            "compilation_seconds": 0.0,
            "first_call_seconds": first_time,
            "model_to_first_evaluation_seconds": model_time + first_time,
            "cold_start_end_to_end_seconds": load_time + model_time + first_time,
            "end_to_end_first_evaluation_seconds": load_time + model_time + first_time,
            "first_output": first_output,
            "compiled_input_names": [],
            "synchronized_root_parameters": sync_summary["synchronized"],
            "transformed_root_parameters": sync_summary["transformed"],
            "skipped_root_parameters": sync_summary["skipped"],
        }
        cleanup = root_file.Close
    else:
        raise ValueError(config.engine)

    lifecycle["current_rss_before_mb"] = rss_before
    lifecycle["peak_rss_before_mb"] = peak_before
    return lifecycle, scalar, evaluate_grid, cleanup


def _evaluate_noncompiled_at(
    model: Any, params: dict[str, np.ndarray], config: Config, x: float
) -> float:
    _set_scalar(params, config.observable_name, x)
    return finite_scalar(
        model.pdf(config.distribution, **params), label="pyHS3 non-compiled PDF"
    )


def run_engine(
    config: Config,
    reference_values: np.ndarray | None,
    shared_params: dict[str, np.ndarray],
    shared_values: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Run one engine in the current process.

    The public driver executes this function only inside a fresh spawned
    process.  Consequently memory, startup, first-call, and compilation
    measurements are isolated from engines that ran earlier.
    """
    gc.collect()

    lifecycle, scalar, evaluate_grid, cleanup = _prepare_engine(
        config,
        shared_params,
        shared_values,
    )

    try:
        observed_grid = evaluate_grid(shared_values)

        validation = (
            {}
            if reference_values is None
            else agreement_arrays(
                observed_grid,
                reference_values,
                rtol=config.rtol,
                atol=config.atol,
            )
        )

        scaling = benchmark_scaling(
            scalar,
            n_evaluations=config.n_evaluations,
            repeats=config.timing_repeats,
            warmup_evaluations=config.warmup_evaluations,
        )
    finally:
        cleanup()

    gc.collect()

    current_after = current_rss_mb()
    peak_after = peak_rss_mb()

    memory = {
        "measurement_isolation": "fresh_spawned_process",
        "current_rss_after_mb": current_after,
        "current_rss_delta_mb": max(
            0.0,
            current_after - lifecycle["current_rss_before_mb"],
        ),
        "peak_rss_after_mb": peak_after,
        "peak_rss_delta_mb": max(
            0.0,
            peak_after - lifecycle["peak_rss_before_mb"],
        ),
    }

    rows: list[dict[str, Any]] = []

    for point in scaling:
        row = {
            "benchmark": BENCHMARK_NAME,
            "category": "Scalar PDF",
            "comparison_class": ("apples-to-apples engine-to-engine microbenchmark"),
            "measurement_isolation": "fresh_spawned_process",
            "engine": config.engine,
            "engine_label": ENGINE_LABELS[config.engine],
            "workspace": config.workspace_path.name,
            "workspace_path": str(config.workspace_path),
            "root_workspace_path": (
                str(config.root_workspace_path) if config.root_workspace_path else None
            ),
            "workspace_label": config.workspace_path.stem,
            "target": config.target,
            "mode": config.mode,
            "distribution": config.distribution,
            "observable_name": config.observable_name,
            "input_mode": config.input_mode,
            "status": (
                "success"
                if validation.get("validation_status", "success") == "success"
                else "validation_failed"
            ),
            **lifecycle,
            **memory,
            **point,
            **validation,
        }
        rows.append(row)

    return rows, observed_grid


def _isolated_engine_worker(
    config: Config,
    reference_values: np.ndarray | None,
    shared_params: dict[str, np.ndarray],
    shared_values: np.ndarray,
    result_queue: Any,
) -> None:
    """Child-process entry point for one workspace/engine/mode run."""
    try:
        rows, observed_grid = run_engine(
            config,
            reference_values,
            shared_params,
            shared_values,
        )
        result_queue.put(
            {
                "ok": True,
                "rows": rows,
                "observed_grid": observed_grid.tolist(),
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
    reference_values: np.ndarray | None,
    shared_params: dict[str, np.ndarray],
    shared_values: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Run exactly one engine in a fresh spawned process.

    A spawned process avoids inherited ROOT/JAX/PyTensor global state.  This
    makes RSS deltas, workspace/model construction, compilation, and first-call
    timings independent of the order in which engines are benchmarked.
    """
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)

    process = context.Process(
        target=_isolated_engine_worker,
        args=(
            config,
            reference_values,
            shared_params,
            shared_values,
            result_queue,
        ),
    )

    process.start()
    process.join()

    try:
        payload = result_queue.get(timeout=5.0)
    except queue.Empty as exc:
        raise RuntimeError(
            "Isolated benchmark worker produced no result "
            f"(exit code: {process.exitcode})."
        ) from exc
    finally:
        result_queue.close()
        result_queue.join_thread()

    if process.exitcode not in (0, None) and payload.get("ok"):
        raise RuntimeError(
            f"Isolated benchmark worker exited abnormally with code {process.exitcode}."
        )

    if not payload.get("ok"):
        raise RuntimeError(
            f"{payload.get('error_type', 'WorkerError')}: "
            f"{payload.get('error_message', 'unknown worker failure')}\n"
            f"{payload.get('traceback', '')}"
        )

    rows = payload["rows"]
    observed_grid = np.asarray(
        payload["observed_grid"],
        dtype=np.float64,
    )
    return rows, observed_grid


def _workspace_title(path: str) -> str:
    return path.replace(".json", "").replace("_", " / ")


def _successful(
    results: list[dict[str, Any]], mode: str | None = None
) -> list[dict[str, Any]]:
    return [
        r
        for r in results
        if r.get("status") == "success"
        and (mode is None or r.get("input_mode") == mode)
    ]


def _set_figure_header(
    fig: Any,
    title: str,
    mode: str | None = None,
) -> None:
    """Place the title and optional input-mode subtitle without overlap."""

    fig.suptitle(
        title,
        fontsize=23,
        fontweight="bold",
        y=0.985,
    )

    if mode is not None:
        fig.text(
            0.5,
            0.945,
            _mode_label(mode),
            ha="center",
            va="top",
            fontsize=13,
            color="0.38",
        )


def plot_time_per_value(
    results: list[dict[str, Any]],
    mode: str,
    output: Path,
) -> None:
    rows = _successful(results, mode)
    workspaces = list(dict.fromkeys(row["workspace"] for row in rows))

    if not workspaces:
        return

    fig, axes = plt.subplots(
        len(workspaces),
        1,
        figsize=(13, 4.6 * len(workspaces)),
        squeeze=False,
    )

    _set_figure_header(
        fig,
        title="Scalar PDF evaluation time",
        mode=mode,
    )

    for ax, workspace in zip(
        axes[:, 0],
        workspaces,
        strict=True,
    ):
        for engine in SUPPORTED_ENGINES:
            points = sorted(
                [
                    row
                    for row in rows
                    if row["workspace"] == workspace and row["engine"] == engine
                ],
                key=lambda row: row["n_evaluations"],
            )

            if not points:
                continue

            ax.plot(
                [point["n_evaluations"] for point in points],
                [point["time_per_value_ns"] for point in points],
                label=ENGINE_LABELS[engine],
                **ENGINE_STYLE[engine],
            )

        ax.set_title(
            _workspace_title(workspace),
            loc="left",
            fontweight="bold",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of repeated scalar evaluations")
        ax.set_ylabel("Median time per evaluation [ns]")

        style_axes(ax)
        ax.legend(frameon=False)

    fig.subplots_adjust(
        top=0.84,
        bottom=0.08,
        hspace=0.42,
    )
    save_figure(fig, output)


def plot_throughput(
    results: list[dict[str, Any]],
    mode: str,
    output: Path,
) -> None:
    rows = _successful(results, mode)
    workspaces = list(dict.fromkeys(row["workspace"] for row in rows))

    if not workspaces:
        return

    fig, axes = plt.subplots(
        len(workspaces),
        1,
        figsize=(13, 4.6 * len(workspaces)),
        squeeze=False,
    )

    _set_figure_header(
        fig,
        title="Scalar PDF throughput",
        mode=mode,
    )

    for ax, workspace in zip(
        axes[:, 0],
        workspaces,
        strict=True,
    ):
        for engine in SUPPORTED_ENGINES:
            points = sorted(
                [
                    row
                    for row in rows
                    if row["workspace"] == workspace and row["engine"] == engine
                ],
                key=lambda row: row["n_evaluations"],
            )

            if not points:
                continue

            ax.plot(
                [point["n_evaluations"] for point in points],
                [point["throughput_evaluations_per_second"] for point in points],
                label=ENGINE_LABELS[engine],
                **ENGINE_STYLE[engine],
            )

        ax.set_title(
            _workspace_title(workspace),
            loc="left",
            fontweight="bold",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of repeated scalar evaluations")
        ax.set_ylabel("Throughput [evaluations/s]")

        style_axes(ax)
        ax.legend(frameon=False)

    fig.subplots_adjust(
        top=0.84,
        bottom=0.08,
        hspace=0.42,
    )
    save_figure(fig, output)


def plot_latency(
    results: list[dict[str, Any]],
    mode: str,
    output: Path,
) -> None:
    rows = _successful(results, mode)
    workspaces = list(dict.fromkeys(row["workspace"] for row in rows))

    if not workspaces:
        return

    fig, axes = plt.subplots(
        len(workspaces),
        2,
        figsize=(18, 4.7 * len(workspaces)),
        squeeze=False,
    )

    _set_figure_header(
        fig,
        title="Scalar PDF startup and steady-state latency",
        mode=mode,
    )

    global_max_evaluations = max(row["n_evaluations"] for row in rows)

    for row_index, workspace in enumerate(workspaces):
        workspace_rows = [row for row in rows if row["workspace"] == workspace]

        unique = {row["engine"]: row for row in workspace_rows}

        available_engines = [engine for engine in SUPPORTED_ENGINES if engine in unique]

        labels = [ENGINE_LABELS[engine] for engine in available_engines]
        bar_colors = [ENGINE_COLORS[engine] for engine in available_engines]

        end_to_end_ms = [
            unique[engine]["cold_start_end_to_end_seconds"] * 1e3
            for engine in available_engines
        ]

        steady_state_us: list[float] = []

        for engine in available_engines:
            engine_rows = [row for row in workspace_rows if row["engine"] == engine]

            largest_batch_row = min(
                engine_rows,
                key=lambda row: abs(row["n_evaluations"] - global_max_evaluations),
            )

            steady_state_us.append(
                largest_batch_row["time_per_value_seconds_median"] * 1e6
            )

        ax_startup, ax_steady = axes[row_index]

        startup_bars = ax_startup.bar(
            labels,
            end_to_end_ms,
            color=bar_colors,
            width=0.68,
        )
        _style_bars(startup_bars)

        ax_startup.set_yscale("log")
        ax_startup.set_ylabel("Cold-start end-to-end [ms]")
        ax_startup.set_title(
            _workspace_title(workspace),
            loc="left",
            fontweight="bold",
        )

        for bar, value in zip(
            startup_bars,
            end_to_end_ms,
            strict=True,
        ):
            ax_startup.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.05,
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        style_axes(ax_startup, grid_axis="y")

        steady_bars = ax_steady.bar(
            labels,
            steady_state_us,
            color=bar_colors,
            width=0.68,
        )
        _style_bars(steady_bars)

        ax_steady.set_yscale("log")
        ax_steady.set_ylabel("Median time [µs/evaluation]")
        ax_steady.set_title(
            "Steady state at the largest batch size",
            loc="left",
            fontweight="bold",
        )

        for bar, value in zip(
            steady_bars,
            steady_state_us,
            strict=True,
        ):
            ax_steady.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.05,
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        style_axes(ax_steady, grid_axis="y")

        for ax in (ax_startup, ax_steady):
            ax.tick_params(
                axis="x",
                rotation=18,
            )

    fig.subplots_adjust(
        top=0.84,
        bottom=0.10,
        hspace=0.48,
        wspace=0.28,
    )
    save_figure(fig, output)


def plot_memory(results: list[dict[str, Any]], mode: str, output: Path) -> None:
    rows = _successful(results, mode)
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        unique[(r["workspace"], r["engine"])] = r
    if not unique:
        return
    labels, current, peak = [], [], []
    for workspace in dict.fromkeys(k[0] for k in unique):
        for engine in SUPPORTED_ENGINES:
            row = unique.get((workspace, engine))
            if row:
                labels.append(f"{workspace.split('ch_')[0]}ch\n{ENGINE_LABELS[engine]}")
                current.append(max(row["current_rss_delta_mb"], 1e-3))
                peak.append(max(row["peak_rss_delta_mb"], 1e-3))
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 1.6), 8))
    bars_current = ax.bar(
        x - width / 2,
        current,
        width,
        label="Current RSS increase",
        color="#1565C0",
    )
    bars_peak = ax.bar(
        x + width / 2,
        peak,
        width,
        label="Peak RSS increase",
        color="#EF6C00",
        hatch="//",
    )
    _style_bars(bars_current)
    _style_bars(bars_peak)
    ax.set_yscale("log")
    ax.set_ylabel("Memory delta [MB]")
    _set_figure_header(
        fig,
        title="Scalar PDF memory use",
        mode=mode,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    style_axes(ax, grid_axis="y")
    ax.legend(frameon=False)
    fig.subplots_adjust(
        top=0.84,
        bottom=0.18,
    )
    save_figure(fig, output)


def plot_agreement(
    results: list[dict[str, Any]],
    mode: str,
    output: Path,
    tolerance: float,
) -> None:
    rows = _successful(results, mode)

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        unique[(row["workspace"], row["engine"])] = row

    workspaces = list(dict.fromkeys(row["workspace"] for row in rows))

    if not workspaces:
        return

    fig, axes = plt.subplots(
        len(workspaces),
        1,
        figsize=(14, 4.2 * len(workspaces)),
        squeeze=False,
    )

    _set_figure_header(
        fig,
        title="Scalar PDF numerical agreement",
        mode=mode,
    )

    for ax, workspace in zip(
        axes[:, 0],
        workspaces,
        strict=True,
    ):
        available_engines: list[str] = []
        values: list[float] = []

        for engine in SUPPORTED_ENGINES:
            row = unique.get((workspace, engine))

            if row is None:
                continue

            available_engines.append(engine)
            values.append(
                max(
                    float(row.get("max_abs_diff", 0.0)),
                    1e-18,
                )
            )

        labels = [ENGINE_LABELS[engine] for engine in available_engines]
        colors = [ENGINE_COLORS[engine] for engine in available_engines]

        bars = ax.bar(
            labels,
            values,
            color=colors,
            width=0.68,
        )
        _style_bars(bars)

        ax.axhline(
            tolerance,
            linestyle="--",
            linewidth=1.8,
            color="0.15",
            label=f"Absolute tolerance = {tolerance:g}",
        )

        ax.set_yscale("log")
        ax.set_ylabel("Maximum absolute PDF difference")
        ax.set_title(
            _workspace_title(workspace),
            loc="left",
            fontweight="bold",
        )

        for bar, value in zip(
            bars,
            values,
            strict=True,
        ):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.08,
                f"{value:.2e}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        style_axes(ax, grid_axis="y")
        ax.legend(
            frameon=False,
            loc="upper right",
        )
        ax.tick_params(
            axis="x",
            rotation=15,
        )

    fig.subplots_adjust(
        top=0.84,
        bottom=0.10,
        hspace=0.48,
    )
    save_figure(fig, output)


def plot_compiled_lifecycle(
    results: list[dict[str, Any]],
    output: Path,
) -> None:
    rows = [
        row
        for row in _successful(results)
        if row["engine"] == PYHS3_COMPILED and row["input_mode"] == VARYING_INPUT
    ]

    unique = {row["workspace"]: row for row in rows}

    if not unique:
        return

    workspaces = list(unique)

    labels = [_workspace_title(workspace) for workspace in workspaces]

    phases = [
        ("model_construction_seconds", "Model construction"),
        ("graph_preparation_seconds", "Graph preparation"),
        ("compilation_seconds", "Compilation"),
        ("first_call_seconds", "First call"),
    ]

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels), dtype=float)

    totals_ms = np.asarray(
        [
            sum(float(unique[workspace][key]) for key, _ in phases) * 1e3
            for workspace in workspaces
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(max(12, len(labels) * 3.4), 8),
    )

    for key, phase_label in phases:
        values_ms = np.asarray(
            [float(unique[workspace][key]) * 1e3 for workspace in workspaces],
            dtype=float,
        )

        bars = ax.bar(
            x,
            values_ms,
            bottom=bottom,
            label=phase_label,
            color=LIFECYCLE_COLORS[key],
            width=0.68,
        )
        _style_bars(bars)

        percentages = (
            np.divide(
                values_ms,
                totals_ms,
                out=np.zeros_like(values_ms),
                where=totals_ms > 0.0,
            )
            * 100.0
        )

        for index, (value_ms, percentage) in enumerate(
            zip(
                values_ms,
                percentages,
                strict=True,
            )
        ):
            if percentage < 8.0:
                continue

            ax.text(
                x[index],
                bottom[index] + value_ms / 2.0,
                f"{percentage:.0f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=(
                    "white"
                    if key
                    in {
                        "model_construction_seconds",
                        "compilation_seconds",
                        "first_call_seconds",
                    }
                    else "0.15"
                ),
            )

        bottom += values_ms

    for index, total_ms in enumerate(totals_ms):
        ax.text(
            x[index],
            total_ms * 1.08,
            f"{total_ms:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_yscale("log")

    positive_totals = totals_ms[totals_ms > 0.0]
    if positive_totals.size:
        ax.set_ylim(
            max(
                float(np.min(positive_totals)) * 0.15,
                1e-3,
            ),
            float(np.max(positive_totals)) * 1.35,
        )

    ax.set_ylabel("Wall time [ms]")
    ax.set_title(
        "Compiled pyHS3 lifecycle",
        fontsize=22,
        fontweight="bold",
        pad=18,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=10,
        ha="right",
    )

    style_axes(ax, grid_axis="y")

    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        labelspacing=0.9,
        handlelength=1.8,
    )

    fig.subplots_adjust(
        top=0.88,
        right=0.78,
        bottom=0.22,
    )

    save_figure(fig, output)


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
        "--input-modes", nargs="+", choices=INPUT_MODES, default=[VARYING_INPUT]
    )
    parser.add_argument("--target", default="L_ch0")
    parser.add_argument("--distribution", default="sig_ch0")
    parser.add_argument("--observable-name", default="x")
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument(
        "--n-evaluations", nargs="+", type=int, default=DEFAULT_N_EVALUATIONS
    )
    parser.add_argument("--timing-repeats", type=int, default=7)
    parser.add_argument("--warmup-evaluations", type=int, default=100)
    parser.add_argument("--validation-points", type=int, default=257)
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--output", type=Path, default=RESULTS_DIR / BENCHMARK_NAME / "result.json"
    )
    parser.add_argument("--plot-dir", type=Path, default=PLOTS_DIR / BENCHMARK_NAME)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root_paths = args.root_workspaces or [
        path.with_suffix(".root") for path in args.workspaces
    ]

    if len(root_paths) != len(args.workspaces):
        raise ValueError("--root-workspaces must match --workspaces")

    results: list[dict[str, Any]] = []
    reference_values: dict[str, list[float]] = {}

    ordered_engines = [
        engine
        for engine in (
            PYHS3_NONCOMPILED,
            PYHS3_COMPILED,
            ROOFIT,
        )
        if engine in args.engines
    ]

    for workspace, root_workspace in zip(
        args.workspaces,
        root_paths,
        strict=True,
    ):
        # Prepare the canonical parameter point and observable grid once.
        # This setup is intentionally outside every isolated engine lifecycle.
        input_config = Config(
            engine=PYHS3_NONCOMPILED,
            workspace_path=workspace,
            root_workspace_path=root_workspace,
            target=args.target,
            mode=args.mode,
            distribution=args.distribution,
            observable_name=args.observable_name,
            input_mode=VARYING_INPUT,
            n_evaluations=tuple(args.n_evaluations),
            timing_repeats=args.timing_repeats,
            warmup_evaluations=args.warmup_evaluations,
            validation_points=args.validation_points,
            rtol=args.rtol,
            atol=args.atol,
        )
        shared_params, shared_values = _shared_inputs(input_config)

        for input_mode in args.input_modes:
            reference_grid: np.ndarray | None = None

            for engine in ordered_engines:
                config = Config(
                    engine=engine,
                    workspace_path=workspace,
                    root_workspace_path=root_workspace,
                    target=args.target,
                    mode=args.mode,
                    distribution=args.distribution,
                    observable_name=args.observable_name,
                    input_mode=input_mode,
                    n_evaluations=tuple(args.n_evaluations),
                    timing_repeats=args.timing_repeats,
                    warmup_evaluations=args.warmup_evaluations,
                    validation_points=args.validation_points,
                    rtol=args.rtol,
                    atol=args.atol,
                )

                try:
                    rows, observed = run_engine_isolated(
                        config,
                        reference_grid,
                        shared_params,
                        shared_values,
                    )

                    if reference_grid is None:
                        reference_grid = observed
                        reference_values[f"{workspace}:{input_mode}"] = [
                            float(value) for value in observed
                        ]

                    results.extend(rows)
                    status = rows[0]["status"]

                except Exception as exc:
                    status = "failed"
                    results.append(
                        {
                            "benchmark": BENCHMARK_NAME,
                            "engine": engine,
                            "engine_label": ENGINE_LABELS[engine],
                            "workspace": workspace.name,
                            "workspace_path": str(workspace),
                            "root_workspace_path": str(root_workspace),
                            "workspace_label": workspace.stem,
                            "input_mode": input_mode,
                            "measurement_isolation": ("fresh_spawned_process"),
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )

                print(
                    f"{workspace.name} / {input_mode} / "
                    f"{ENGINE_LABELS[engine]}: {status}"
                )

    summary = {
        "n_results": len(results),
        "n_success": sum(result.get("status") == "success" for result in results),
        "n_validation_failed": sum(
            result.get("status") == "validation_failed" for result in results
        ),
        "n_failed": sum(result.get("status") == "failed" for result in results),
        "all_required_runs_passed": all(
            result.get("status") == "success" for result in results
        ),
        "measurement_isolation": "fresh_spawned_process_per_engine",
    }

    payload = {
        "benchmark": BENCHMARK_NAME,
        "methodology": {
            "primary_mode": (
                "varying observable values; fixed observable is only a RooFit "
                "cache diagnostic"
            ),
            "compiled_timing": (
                "distribution lookup plus PyTensor-to-JAX conversion are "
                "reported as graph preparation; explicit XLA compilation is "
                "measured separately before first execution and steady state"
            ),
            "end_to_end_metrics": (
                "model_to_first_evaluation excludes workspace loading; "
                "cold_start_end_to_end includes workspace loading"
            ),
            "compiled_execution": (
                "one compiled scalar PDF function accepts only x; all other "
                "parameters are closed over as constant JAX inputs"
            ),
            "synchronization": (
                "every JAX timing waits for completed execution with "
                "jax.block_until_ready"
            ),
            "measurement_isolation": (
                "one fresh spawned process per workspace, input mode, and engine"
            ),
        },
        "configuration": vars(args)
        | {
            "workspaces": [str(path) for path in args.workspaces],
            "root_workspaces": [str(path) for path in root_paths],
            "output": str(args.output),
            "plot_dir": str(args.plot_dir),
            "measurement_isolation": ("fresh_spawned_process_per_engine"),
            "shared_input_setup_included_in_lifecycle": False,
        },
        "reference_values": reference_values,
        "results": results,
        "summary": summary,
    }

    save_json(payload, args.output)

    if args.plot:
        for mode in args.input_modes:
            plot_time_per_value(
                results,
                mode,
                args.plot_dir / f"scalar_pdf_{mode}_time_per_value.png",
            )
            plot_throughput(
                results,
                mode,
                args.plot_dir / f"scalar_pdf_{mode}_throughput.png",
            )
            plot_latency(
                results,
                mode,
                args.plot_dir / f"scalar_pdf_{mode}_latency.png",
            )
            plot_memory(
                results,
                mode,
                args.plot_dir / f"scalar_pdf_{mode}_memory.png",
            )
            plot_agreement(
                results,
                mode,
                args.plot_dir / f"scalar_pdf_{mode}_numerical_agreement.png",
                args.atol,
            )

        plot_compiled_lifecycle(
            results,
            args.plot_dir / "scalar_pdf_compiled_lifecycle.png",
        )

    if not summary["all_required_runs_passed"]:
        raise SystemExit(
            "One or more scalar benchmark runs or validations failed; "
            "inspect the JSON output."
        )


if __name__ == "__main__":
    main()
