"""Cross-framework scalar PDF evaluation benchmark for generated HS3 workspaces.

This benchmark compares repeated scalar PDF evaluation for matching PyHS3 and
ROOT workspaces on an apples-to-apples basis. It is intended for the generated benchmark workspace
collection stored directly in ``inputs/`` together with matching ``.root``
files.

For each selected workspace, framework, and number of repeated evaluations, the
benchmark measures cold-start latency, warm repeated-evaluation latency,
throughput, RSS memory deltas, and numerical agreement relative to PyHS3.

Important: this benchmark intentionally compares only scalar PDF evaluation
paths. PyHS3 compiled graph evaluation is measured by ``run_compiled_evaluation``
and is not mixed into this cross-framework scalar-PDF comparison because it is a
different operation.
"""

from __future__ import annotations

import argparse
import gc
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
    from src.config import DEFAULT_MODE, DEFAULT_TARGET, PLOTS_DIR, RESULTS_DIR
    from src.utils import get_current_rss_mb, get_peak_rss_mb, save_json
else:
    from .config import DEFAULT_MODE, DEFAULT_TARGET, PLOTS_DIR, RESULTS_DIR
    from .utils import get_current_rss_mb, get_peak_rss_mb, save_json


BENCHMARK_NAME = "cross_scalar_pdf_evaluation"

DEFAULT_FRAMEWORKS = ["pyhs3", "root"]
DEFAULT_N_EVALUATIONS = [1, 10, 100, 1000, 10000]
DEFAULT_DISTRIBUTION = "sig_ch0"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = "cross_scalar_pdf_evaluation_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME
DEFAULT_WORKSPACES = [
    Path("inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json"),
    Path("inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json"),
    Path("inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json"),
    Path("inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json"),
    Path("inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json"),
]

SUPPORTED_FRAMEWORKS = ("pyhs3", "root")
REFERENCE_FRAMEWORK = "pyhs3"

FRAMEWORK_STYLE = {
    "pyhs3": {
        "label": "PyHS3 (eager)",
        "color": "#0B5EA8",
        "marker": "s",
        "linestyle": "-",
    },
    "root": {"label": "RooFit", "color": "#009E73", "marker": "D", "linestyle": "-."},
}


class BenchmarkConfigurationError(ValueError):
    """Raised when the benchmark configuration is invalid."""


class ValidationFailure(RuntimeError):
    """Raised when a framework output is non-finite or cannot be compared."""


@dataclass(frozen=True)
class ScalarBenchmarkConfig:
    framework: str
    workspace_path: Path
    root_workspace_path: Path | None
    target: str
    mode: str
    distribution: str
    n_evaluations: int
    rtol: float
    atol: float
    reference_value: float | None = None


def _framework_label(framework: str) -> str:
    return FRAMEWORK_STYLE.get(framework, {"label": framework})["label"]


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


def workspace_stem(workspace_path: Path) -> str:
    return workspace_path.name.removesuffix(".json").removesuffix(".root")


def workspace_label(workspace_path: Path) -> str:
    return workspace_stem(workspace_path).replace("_", "\n")


def workspace_title(workspace: str) -> str:
    return workspace.replace("_", " / ")


def default_root_workspace_path(workspace_path: Path) -> Path:
    return workspace_path.with_suffix(".root")


def validate_benchmark_config(
    *,
    frameworks: list[str],
    workspaces: list[Path],
    root_workspaces: list[Path] | None,
    target: str,
    mode: str,
    distribution: str,
    n_evaluations: list[int],
    rtol: float,
    atol: float,
    timeout_seconds: float,
) -> None:
    if not frameworks:
        raise BenchmarkConfigurationError("At least one framework must be selected.")
    if not workspaces:
        raise BenchmarkConfigurationError("At least one workspace must be selected.")
    if not target:
        raise BenchmarkConfigurationError("--target must be a non-empty string.")
    if not mode:
        raise BenchmarkConfigurationError("--mode must be a non-empty string.")
    if not distribution:
        raise BenchmarkConfigurationError("--distribution must be a non-empty string.")

    unknown_frameworks = sorted(set(frameworks) - set(SUPPORTED_FRAMEWORKS))
    if unknown_frameworks:
        raise BenchmarkConfigurationError(
            f"Unknown framework(s): {', '.join(unknown_frameworks)}"
        )

    if any(value < 1 for value in n_evaluations):
        raise BenchmarkConfigurationError(
            "All --n-evaluations values must be at least 1."
        )
    if rtol < 0.0 or atol < 0.0:
        raise BenchmarkConfigurationError("--rtol and --atol must be non-negative.")
    if timeout_seconds <= 0.0:
        raise BenchmarkConfigurationError("--timeout-seconds must be positive.")

    for workspace in workspaces:
        if not workspace.exists():
            raise FileNotFoundError(f"Workspace file does not exist: {workspace}")
        if not workspace.is_file():
            raise FileNotFoundError(f"Workspace path is not a file: {workspace}")

    if root_workspaces is not None and len(root_workspaces) != len(workspaces):
        raise BenchmarkConfigurationError(
            "--root-workspaces must have the same number of entries as --workspaces."
        )

    if "root" in frameworks:
        root_paths = root_workspaces or [
            default_root_workspace_path(path) for path in workspaces
        ]
        for root_path in root_paths:
            if not root_path.exists():
                raise FileNotFoundError(
                    f"ROOT workspace file does not exist: {root_path}"
                )
            if not root_path.is_file():
                raise FileNotFoundError(
                    f"ROOT workspace path is not a file: {root_path}"
                )


def _pyhs3_default_parameters(model: Any) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    for source in (getattr(model, "data", {}), getattr(model, "free_params", {})):
        for name, value in source.items():
            parameters[name] = np.asarray(value, dtype=float)
    return parameters


def evaluate_pyhs3(
    workspace_path: Path,
    target: str,
    mode: str,
    distribution: str,
) -> float:
    from pyhs3.workspace import Workspace

    workspace = Workspace.load(workspace_path)
    model = workspace.model(target, progress=False, mode=mode)
    parameters = _pyhs3_default_parameters(model)

    result = model.pdf(distribution, **parameters)
    value = float(np.asarray(result).reshape(-1)[0])
    if not np.isfinite(value):
        raise ValidationFailure(f"PyHS3 returned a non-finite value: {value}")
    return value


def _find_root_workspace(root_file: Any) -> Any:
    import ROOT

    for key in root_file.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom(ROOT.RooWorkspace.Class()):
            return obj
    raise KeyError("No RooWorkspace object found in ROOT file.")


def _available_root_pdfs(workspace: Any) -> list[str]:
    return _root_collection_names(workspace.allPdfs())


def _available_root_vars(workspace: Any) -> list[str]:
    return _root_collection_names(workspace.allVars())


def _root_collection_names(collection: Any) -> list[str]:
    """Return RooFit collection object names across old and new PyROOT APIs.

    Some PyROOT versions expose ``createIterator()`` on RooArgSet/RooArgList,
    while newer pythonized versions support direct Python iteration and may not
    provide ``createIterator()``.  This helper supports both forms so plotting
    and validation diagnostics do not depend on a specific ROOT binding version.
    """

    names: list[str] = []

    try:
        for obj in collection:
            if obj is not None:
                names.append(str(obj.GetName()))
        return names
    except TypeError:
        pass

    iterator = collection.createIterator()
    obj = iterator.Next()
    while obj:
        names.append(str(obj.GetName()))
        obj = iterator.Next()
    return names


def _root_argset_names(argset: Any) -> list[str]:
    return _root_collection_names(argset)


def _is_root_observable_name(name: str) -> bool:
    """Heuristic for generated benchmark RooFit observables.

    The generated ROOT workspaces use the channel observable as ``x`` or an
    ``x_*``-style variable.  Shape parameters such as ``mean_ch0``,
    ``sigma_ch0``, ``alpha_sigma``, and ``sigma_nom_ch0`` must not be included
    in the normalization set, otherwise RooFit attempts an open-ended
    multi-dimensional integral over parameters instead of a one-dimensional
    PDF normalization over the observable.
    """

    return (
        name == "x"
        or name.startswith("x_")
        or name.startswith("xch")
        or name.startswith("x_ch")
    )


def _make_root_argset(root_workspace: Any, names: Iterable[str]) -> Any:
    import ROOT

    argset = ROOT.RooArgSet()
    for name in names:
        variable = root_workspace.var(str(name))
        if variable is not None:
            argset.add(variable)
    return argset


def _root_norm_set_for_pdf(root_workspace: Any, pdf: Any) -> Any:
    """Return the RooArgSet used to normalize a RooFit scalar PDF value.

    ``RooAbsPdf.getVal()`` without an explicit normalization set may return an
    unnormalized function value.  PyHS3 ``model.pdf(...)`` returns a normalized
    density/mass value, so the apples-to-apples RooFit call is
    ``pdf.getVal(norm_set)``.

    The important detail is that ``norm_set`` must contain only observables, not
    shape parameters.  ``pdf.getObservables(workspace.allVars())`` may return all
    variables appearing in a generic PDF, including parameters such as
    ``mean_ch0`` and ``sigma_ch0``.  Passing that full set to ``getVal`` makes
    RooFit try to normalize over an open-ended multi-dimensional parameter
    space.  For these generated scalar benchmark workspaces we therefore select
    the channel observable explicitly by name, normally ``x`` or ``x_*``.
    """

    candidates: list[str] = []

    try:
        pdf_variables = pdf.getObservables(root_workspace.allVars())
        candidates.extend(
            name
            for name in _root_argset_names(pdf_variables)
            if _is_root_observable_name(name)
        )
    except Exception:
        pass

    if not candidates:
        candidates.extend(
            name
            for name in _available_root_vars(root_workspace)
            if _is_root_observable_name(name)
        )

    # Preserve order while removing duplicates.
    candidates = list(dict.fromkeys(candidates))
    norm_set = _make_root_argset(root_workspace, candidates)

    if norm_set.getSize() > 0:
        return norm_set

    available = ", ".join(_available_root_vars(root_workspace))
    raise KeyError(
        "Could not determine RooFit normalization observable set for "
        f"PDF '{pdf.GetName()}'. Available variables: {available}"
    )


def _set_root_defaults_from_pyhs3(
    root_workspace: Any,
    workspace_path: Path,
    target: str,
    mode: str,
) -> None:
    """Best-effort synchronization of RooWorkspace variables with PyHS3 defaults.

    ROOT files normally store their own default values, so this function is not
    required for correctness in the common case.  When variable names match the
    HS3 parameter names, this makes the comparison more explicit and reproducible.
    Missing variables are silently ignored.
    """

    try:
        from pyhs3.workspace import Workspace

        workspace = Workspace.load(workspace_path)
        model = workspace.model(target, progress=False, mode=mode)
        parameters = _pyhs3_default_parameters(model)
    except Exception:
        return

    for name, value in parameters.items():
        root_var = root_workspace.var(str(name))
        if root_var is None:
            continue
        try:
            root_var.setVal(float(np.asarray(value).reshape(-1)[0]))
        except Exception:
            continue


def evaluate_root(
    root_workspace_path: Path,
    workspace_path: Path,
    target: str,
    mode: str,
    distribution: str,
) -> float:
    import ROOT

    root_file = ROOT.TFile.Open(str(root_workspace_path), "READ")
    if not root_file or root_file.IsZombie():
        raise FileNotFoundError(f"Could not open ROOT file: {root_workspace_path}")

    try:
        workspace = _find_root_workspace(root_file)
        _set_root_defaults_from_pyhs3(
            root_workspace=workspace,
            workspace_path=workspace_path,
            target=target,
            mode=mode,
        )

        pdf = workspace.pdf(distribution)
        if pdf is None:
            available = ", ".join(_available_root_pdfs(workspace))
            raise KeyError(
                f"PDF '{distribution}' was not found in {root_workspace_path}. "
                f"Available PDFs: {available}"
            )

        norm_set = _root_norm_set_for_pdf(workspace, pdf)
        value = float(pdf.getVal(norm_set))
        if not np.isfinite(value):
            raise ValidationFailure(f"ROOT returned a non-finite value: {value}")
        return value
    finally:
        root_file.Close()


def evaluate_framework_once(
    *,
    framework: str,
    workspace_path: Path,
    root_workspace_path: Path | None,
    target: str,
    mode: str,
    distribution: str,
) -> float:
    if framework == "pyhs3":
        return evaluate_pyhs3(workspace_path, target, mode, distribution)
    if framework == "root":
        if root_workspace_path is None:
            raise BenchmarkConfigurationError(
                "ROOT framework requires a ROOT workspace path."
            )
        return evaluate_root(
            root_workspace_path, workspace_path, target, mode, distribution
        )
    raise ValueError(f"Unknown framework: {framework}")


def compute_agreement(
    observed: float, reference: float, rtol: float, atol: float
) -> dict[str, Any]:
    if not np.isfinite(observed):
        raise ValidationFailure(
            f"Framework returned a non-finite PDF value: {observed}"
        )
    if not np.isfinite(reference):
        raise ValidationFailure(f"Reference PDF value is non-finite: {reference}")

    abs_diff = abs(observed - reference)
    rel_diff = abs_diff / max(abs(reference), 1e-300)
    allclose_passed = bool(np.allclose(observed, reference, rtol=rtol, atol=atol))

    return {
        "reference_framework": REFERENCE_FRAMEWORK,
        "reference_value": float(reference),
        "observed_value": float(observed),
        "n_values": 1,
        "n_finite_values": 1,
        "all_values_finite": True,
        "max_abs_diff": float(abs_diff),
        "mean_abs_diff": float(abs_diff),
        "max_rel_diff": float(rel_diff),
        "mean_rel_diff": float(rel_diff),
        "allclose_passed": allclose_passed,
        "validation_status": "success" if allclose_passed else "mismatch",
    }


def _prepare_pyhs3_evaluator(
    workspace_path: Path,
    target: str,
    mode: str,
    distribution: str,
):
    """Create a reusable PyHS3 scalar PDF evaluator.

    The expensive workspace loading and model construction are performed once.
    Warm timing then measures only repeated ``model.pdf(...)`` calls.
    """

    from pyhs3.workspace import Workspace

    workspace = Workspace.load(workspace_path)
    model = workspace.model(target, progress=False, mode=mode)
    parameters = _pyhs3_default_parameters(model)

    def evaluate() -> float:
        result = model.pdf(distribution, **parameters)
        value = float(np.asarray(result).reshape(-1)[0])
        if not np.isfinite(value):
            raise ValidationFailure(f"PyHS3 returned a non-finite value: {value}")
        return value

    return evaluate, None


def _prepare_root_evaluator(
    root_workspace_path: Path,
    workspace_path: Path,
    target: str,
    mode: str,
    distribution: str,
):
    """Create a reusable ROOT scalar PDF evaluator.

    The ROOT file, RooWorkspace, and RooAbsPdf are opened once and kept alive for
    the full timing loop.  Warm timing then measures only repeated ``getVal()``
    calls.
    """

    import ROOT

    root_file = ROOT.TFile.Open(str(root_workspace_path), "READ")
    if not root_file or root_file.IsZombie():
        raise FileNotFoundError(f"Could not open ROOT file: {root_workspace_path}")

    workspace = _find_root_workspace(root_file)
    _set_root_defaults_from_pyhs3(
        root_workspace=workspace,
        workspace_path=workspace_path,
        target=target,
        mode=mode,
    )

    pdf = workspace.pdf(distribution)
    if pdf is None:
        available = ", ".join(_available_root_pdfs(workspace))
        root_file.Close()
        raise KeyError(
            f"PDF '{distribution}' was not found in {root_workspace_path}. "
            f"Available PDFs: {available}"
        )

    norm_set = _root_norm_set_for_pdf(workspace, pdf)
    norm_set_names = _root_argset_names(norm_set)
    if not norm_set_names:
        raise KeyError(
            f"Could not determine normalization observables for ROOT PDF '{distribution}'."
        )

    def evaluate() -> float:
        value = float(pdf.getVal(norm_set))
        if not np.isfinite(value):
            raise ValidationFailure(f"ROOT returned a non-finite value: {value}")
        return value

    def close() -> None:
        root_file.Close()

    return evaluate, close


def _prepare_evaluator(config: ScalarBenchmarkConfig):
    if config.framework == "pyhs3":
        return _prepare_pyhs3_evaluator(
            workspace_path=config.workspace_path,
            target=config.target,
            mode=config.mode,
            distribution=config.distribution,
        )
    if config.framework == "root":
        if config.root_workspace_path is None:
            raise BenchmarkConfigurationError(
                "ROOT framework requires a ROOT workspace path."
            )
        return _prepare_root_evaluator(
            root_workspace_path=config.root_workspace_path,
            workspace_path=config.workspace_path,
            target=config.target,
            mode=config.mode,
            distribution=config.distribution,
        )
    raise ValueError(f"Unknown framework: {config.framework}")


def run_single_framework_benchmark(config: ScalarBenchmarkConfig) -> dict[str, Any]:
    reference_value = config.reference_value
    if reference_value is None:
        reference_value = evaluate_pyhs3(
            config.workspace_path,
            config.target,
            config.mode,
            config.distribution,
        )

    gc.collect()
    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    evaluator = None
    cleanup = None
    try:
        cold_start_start = time.perf_counter()
        evaluator, cleanup = _prepare_evaluator(config)
        first_value = evaluator()
        cold_start_time_seconds = time.perf_counter() - cold_start_start

        agreement_summary = compute_agreement(
            first_value,
            reference_value,
            config.rtol,
            config.atol,
        )

        warm_start = time.perf_counter()
        last_value = first_value
        for _ in range(config.n_evaluations):
            last_value = evaluator()
        total_runtime_seconds = time.perf_counter() - warm_start
    finally:
        if cleanup is not None:
            cleanup()

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()

    average_runtime_seconds_per_evaluation = (
        total_runtime_seconds / config.n_evaluations
    )
    throughput_evaluations_per_second = (
        config.n_evaluations / total_runtime_seconds
        if total_runtime_seconds > 0.0
        else float("inf")
    )

    output_stable = bool(
        np.allclose(last_value, first_value, rtol=config.rtol, atol=config.atol)
    )
    max_repeated_abs_diff = abs(float(last_value) - float(first_value))

    gc.collect()

    return {
        "benchmark": BENCHMARK_NAME,
        "framework": config.framework,
        "framework_label": _framework_label(config.framework),
        "workspace": config.workspace_path.name,
        "workspace_path": str(config.workspace_path),
        "root_workspace_path": str(config.root_workspace_path)
        if config.root_workspace_path
        else None,
        "workspace_label": workspace_label(config.workspace_path),
        "target": config.target,
        "mode": config.mode,
        "distribution": config.distribution,
        "n_evaluations": int(config.n_evaluations),
        "cold_start_time_seconds": float(cold_start_time_seconds),
        "total_runtime_seconds": float(total_runtime_seconds),
        "average_runtime_seconds_per_evaluation": float(
            average_runtime_seconds_per_evaluation
        ),
        "time_per_value_seconds": float(average_runtime_seconds_per_evaluation),
        "time_per_value_ns": float(average_runtime_seconds_per_evaluation * 1e9),
        "throughput_evaluations_per_second": float(throughput_evaluations_per_second),
        "current_rss_before_mb": float(current_rss_before_mb),
        "current_rss_after_mb": float(current_rss_after_mb),
        "current_rss_delta_mb": float(current_rss_after_mb - current_rss_before_mb),
        "peak_rss_before_mb": float(peak_rss_before_mb),
        "peak_rss_after_mb": float(peak_rss_after_mb),
        "peak_rss_delta_mb": float(peak_rss_after_mb - peak_rss_before_mb),
        "first_timing_output": float(first_value),
        "last_timing_output": float(last_value),
        "outputs_stable": output_stable,
        "max_repeated_abs_diff": float(max_repeated_abs_diff),
        "status": "success",
        **agreement_summary,
    }


def _error_result(
    config: ScalarBenchmarkConfig, status: str, **extra: Any
) -> dict[str, Any]:
    return {
        "benchmark": BENCHMARK_NAME,
        "framework": config.framework,
        "framework_label": _framework_label(config.framework),
        "workspace": config.workspace_path.name,
        "workspace_path": str(config.workspace_path),
        "root_workspace_path": str(config.root_workspace_path)
        if config.root_workspace_path
        else None,
        "workspace_label": workspace_label(config.workspace_path),
        "target": config.target,
        "mode": config.mode,
        "distribution": config.distribution,
        "n_evaluations": int(config.n_evaluations),
        "status": status,
        **extra,
    }


def _config_from_payload(payload: dict[str, Any]) -> ScalarBenchmarkConfig:
    return ScalarBenchmarkConfig(
        framework=payload["framework"],
        workspace_path=Path(payload["workspace_path"]),
        root_workspace_path=Path(payload["root_workspace_path"])
        if payload.get("root_workspace_path")
        else None,
        target=payload["target"],
        mode=payload["mode"],
        distribution=payload["distribution"],
        n_evaluations=payload["n_evaluations"],
        rtol=payload["rtol"],
        atol=payload["atol"],
        reference_value=payload.get("reference_value"),
    )


def run_worker(payload: dict[str, Any], output_queue: mp.Queue) -> None:
    config = _config_from_payload(payload)
    try:
        output_queue.put(run_single_framework_benchmark(config))
    except Exception as error:  # noqa: BLE001 - worker must serialize all errors
        output_queue.put(
            _error_result(
                config,
                "error",
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=traceback.format_exc(),
            )
        )


def run_with_timeout(payload: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
    config = _config_from_payload(payload)

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
        return _error_result(config, "timeout", timeout_seconds=float(timeout_seconds))

    if process.exitcode not in (0, None):
        return _error_result(
            config,
            "error",
            error_type="ProcessExitError",
            error_message=f"Worker exited with code {process.exitcode}",
        )

    try:
        return output_queue.get_nowait()
    except queue.Empty:
        return _error_result(
            config,
            "error",
            error_type="EmptyWorkerResult",
            error_message="Worker finished without returning a result.",
        )


def print_result(result: dict[str, Any]) -> None:
    print()
    print("-" * 72)
    print(
        f"{result.get('workspace')} / "
        f"{result.get('framework_label', result.get('framework'))} / "
        f"evaluations={result.get('n_evaluations')}"
    )
    print("-" * 72)
    print(f"status:                  {result['status']}")

    if result["status"] != "success":
        print("validation:              unavailable")
        print(
            f"error:                   {result.get('error_type', result['status'])}: "
            f"{result.get('error_message', '')}"
        )
        return

    print(f"validation:              {result['validation_status']}")
    print(f"target:                  {result['target']}")
    print(f"mode:                    {result['mode']}")
    print(f"distribution:            {result['distribution']}")
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
        "throughput:              "
        f"{result['throughput_evaluations_per_second']:.3e} evaluations/s"
    )
    print(f"current RSS delta:       {result['current_rss_delta_mb']:.3f} MB")
    print(f"peak RSS delta:          {result['peak_rss_delta_mb']:.3f} MB")
    print(f"reference value:         {result['reference_value']:.12g}")
    print(f"observed value:          {result['observed_value']:.12g}")
    print(f"max abs diff:            {result['max_abs_diff']:.6e}")
    print(f"max rel diff:            {result['max_rel_diff']:.6e}")


def summarize_status(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if result.get("status") == "success"]
    unsuccessful = [result for result in results if result.get("status") != "success"]
    return {
        "status": "success"
        if len(successful) == len(results)
        else "completed_with_errors",
        "n_results": len(results),
        "n_successful": len(successful),
        "n_unsuccessful": len(unsuccessful),
        "unsuccessful_results": [
            {
                "workspace": result.get("workspace"),
                "framework": result.get("framework"),
                "n_evaluations": result.get("n_evaluations"),
                "status": result.get("status"),
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
            }
            for result in unsuccessful
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
    if summary["unsuccessful_results"]:
        print("Unsuccessful:")
        for result in summary["unsuccessful_results"]:
            print(
                "  - "
                f"{result['workspace']} / {result['framework']} / "
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
                "workspace": result["workspace"],
                "workspace_label": result["workspace_label"],
                "workspace_key": workspace_stem(Path(result["workspace"])),
                "framework": result["framework_label"],
                "framework_key": result["framework"],
                "n_evaluations": result["n_evaluations"],
                "cold_ms": result["cold_start_time_seconds"] * 1000.0,
                "warm_ms": result["average_runtime_seconds_per_evaluation"] * 1000.0,
                "ns_per_value": result["time_per_value_ns"],
                "throughput": result["throughput_evaluations_per_second"],
                "current_rss_mb": max(result["current_rss_delta_mb"], 0.0),
                "peak_rss_mb": max(result["peak_rss_delta_mb"], 0.0),
                "max_abs_diff": result["max_abs_diff"],
                "max_rel_diff": result["max_rel_diff"],
                "validation_status": result["validation_status"],
            }
        )
    return pd.DataFrame(rows)


def make_throughput_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    workspaces = list(dict.fromkeys(df["workspace_key"]))
    fig, axes = plt.subplots(
        len(workspaces),
        1,
        figsize=(12.0, 4.6 * len(workspaces)),
        squeeze=False,
    )

    for ax, workspace in zip(axes.flat, workspaces, strict=False):
        subset = df[df["workspace_key"] == workspace]
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
            workspace_title(workspace), loc="left", fontsize=15, fontweight="bold"
        )
        ax.set_xlabel("Number of repeated scalar evaluations", fontsize=13)
        ax.set_ylabel("Throughput [evaluations/s]", fontsize=13)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.18)
        ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=12
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    fig.suptitle(
        "Cross-framework scalar PDF throughput scaling",
        x=0.02,
        ha="left",
        fontsize=26,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_latency_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    workspaces = list(dict.fromkeys(df["workspace_key"]))
    fig, axes = plt.subplots(
        len(workspaces),
        2,
        figsize=(15.5, 4.3 * len(workspaces)),
        squeeze=False,
        sharex=False,
    )

    for row_index, workspace in enumerate(workspaces):
        subset = df[df["workspace_key"] == workspace]
        panels = [("cold_ms", "Cold start [ms]"), ("warm_ms", "Warm / evaluation [ms]")]

        for col_index, (column, ylabel) in enumerate(panels):
            ax = axes[row_index][col_index]
            for framework in list(dict.fromkeys(subset["framework_key"])):
                framework_subset = subset[
                    subset["framework_key"] == framework
                ].sort_values("n_evaluations")
                style = _style_for(framework)
                values = np.maximum(
                    framework_subset[column].to_numpy(dtype=float), 1e-12
                )
                ax.plot(
                    framework_subset["n_evaluations"],
                    values,
                    label=style["label"],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2.4,
                    markersize=7,
                )
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
                f"{workspace_title(workspace)} · {ylabel}",
                loc="left",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xlabel("Repeated scalar evaluations", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
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
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_time_per_value_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    workspaces = list(dict.fromkeys(df["workspace_key"]))
    fig, axes = plt.subplots(
        len(workspaces),
        1,
        figsize=(12.0, 4.6 * len(workspaces)),
        squeeze=False,
    )
    for ax, workspace in zip(axes.flat, workspaces, strict=False):
        subset = df[df["workspace_key"] == workspace]
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
            workspace_title(workspace), loc="left", fontsize=15, fontweight="bold"
        )
        ax.set_xlabel("Number of repeated scalar evaluations", fontsize=13)
        ax.set_ylabel("Time/value [ns]", fontsize=13)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.18)
        ax.legend(
            loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=12
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    fig.suptitle(
        "Cross-framework scalar PDF evaluation cost",
        x=0.02,
        ha="left",
        fontsize=26,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_memory_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    workspaces = list(dict.fromkeys(df["workspace_key"]))
    fig, axes = plt.subplots(
        len(workspaces),
        1,
        figsize=(13.5, 4.2 * len(workspaces)),
        squeeze=False,
        sharex=False,
    )

    for ax, workspace in zip(axes.flat, workspaces, strict=False):
        subset = df[df["workspace_key"] == workspace]
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

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(
            workspace_title(workspace), loc="left", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Repeated scalar evaluations", fontsize=12)
        ax.set_ylabel("Memory delta [MB]", fontsize=12)
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
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_agreement_plot(
    results: list[dict[str, Any]], output_path: Path, tolerance: float
) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    workspaces = list(dict.fromkeys(df["workspace_key"]))
    fig, axes = plt.subplots(
        len(workspaces),
        1,
        figsize=(13.5, 4.2 * len(workspaces)),
        squeeze=False,
        sharex=False,
    )

    floor = min(1e-18, max(tolerance, 1e-300) * 1e-8)

    for ax, workspace in zip(axes.flat, workspaces, strict=False):
        subset = df[df["workspace_key"] == workspace]
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
            workspace_title(workspace), loc="left", fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Repeated scalar evaluations", fontsize=12)
        ax.set_ylabel("max |PDF - PyHS3|", fontsize=12)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.18)
        ax.tick_params(axis="both", which="both", direction="in", width=1.2, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(1.4)

    handles, labels = axes[0][0].get_legend_handles_labels()
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
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_summary_table(results: list[dict[str, Any]], output_path: Path) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    rows = []
    previous_workspace = None

    for result in _ordered_successful_results(results):
        workspace_name = workspace_stem(Path(result["workspace"]))
        workspace_cell = workspace_name if workspace_name != previous_workspace else ""
        previous_workspace = workspace_name

        rows.append(
            [
                workspace_cell,
                result["framework_label"],
                str(result["n_evaluations"]),
                f"{result['cold_start_time_seconds'] * 1000.0:.2f}",
                f"{result['average_runtime_seconds_per_evaluation'] * 1000.0:.4g}",
                f"{result['time_per_value_ns']:.3g}",
                f"{result['throughput_evaluations_per_second']:.2e}",
                f"{max(result['current_rss_delta_mb'], 0.0):.1f}",
                _format_scientific(result["max_abs_diff"]),
                result["validation_status"],
            ]
        )

    columns = [
        "Workspace",
        "Framework",
        "Evals",
        "Cold\n[ms]",
        "Warm\n[ms]",
        "ns/value",
        "Throughput\n[eval/s]",
        "RSS Δ\n[MB]",
        "Max diff",
        "Validation",
    ]

    n_rows = len(rows)
    fig_width = 15.8
    fig_height = max(4.8, 0.22 * n_rows + 1.35)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")

    fig.text(
        0.012,
        0.985,
        "Cross-framework scalar PDF evaluation summary",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.012,
        0.925,
        "Repeated scalar PDF evaluations across matching PyHS3 and ROOT workspaces.",
        fontsize=12.5,
        ha="left",
        va="top",
    )

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        bbox=[0.01, 0.04, 0.98, 0.80],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7.6)

    column_widths = {
        0: 0.245,
        1: 0.095,
        2: 0.058,
        3: 0.075,
        4: 0.075,
        5: 0.075,
        6: 0.105,
        7: 0.075,
        8: 0.075,
        9: 0.075,
    }

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c8c8c8")
        cell.set_linewidth(0.42)

        if col in column_widths:
            cell.set_width(column_widths[col])

        if row == 0:
            cell.set_facecolor("#2b2b2b")
            cell.set_text_props(color="white", weight="bold", fontsize=7.4)
        else:
            workspace_value = rows[row - 1][0]
            if workspace_value:
                cell.set_facecolor("#edf3f8")
                if col == 0:
                    cell.set_text_props(weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f7f7f7")
            else:
                cell.set_facecolor("white")

    fig.savefig(
        output_path, dpi=220, bbox_inches=None, pad_inches=0.03, facecolor="white"
    )
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
        help=(
            "Frameworks to compare. Defaults to PyHS3 eager scalar PDF and "
            "RooFit normalized scalar PDF. PyHS3 compiled graph evaluation is "
            "kept in run_compiled_evaluation.py because it is not the same "
            "scalar-PDF operation."
        ),
    )
    parser.add_argument(
        "--workspaces",
        nargs="+",
        type=Path,
        default=DEFAULT_WORKSPACES,
    )
    parser.add_argument(
        "--root-workspaces",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Optional ROOT files matching --workspaces. If omitted, each ROOT path "
            "is inferred by replacing the JSON suffix with .root."
        ),
    )
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--distribution", default=DEFAULT_DISTRIBUTION)
    parser.add_argument(
        "--n-evaluations", nargs="+", type=int, default=DEFAULT_N_EVALUATIONS
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
    *,
    framework: str,
    workspace_path: Path,
    root_workspace_path: Path | None,
    target: str,
    mode: str,
    distribution: str,
    n_evaluations: int,
    rtol: float,
    atol: float,
    reference_value: float,
) -> dict[str, Any]:
    return {
        "framework": framework,
        "workspace_path": str(workspace_path),
        "root_workspace_path": str(root_workspace_path)
        if root_workspace_path
        else None,
        "target": target,
        "mode": mode,
        "distribution": distribution,
        "n_evaluations": int(n_evaluations),
        "rtol": float(rtol),
        "atol": float(atol),
        "reference_value": float(reference_value),
    }


def run(
    *,
    frameworks: list[str],
    workspaces: list[Path],
    root_workspaces: list[Path] | None,
    target: str,
    mode: str,
    distribution: str,
    n_evaluations: list[int],
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
        workspaces=workspaces,
        root_workspaces=root_workspaces,
        target=target,
        mode=mode,
        distribution=distribution,
        n_evaluations=n_evaluations,
        rtol=rtol,
        atol=atol,
        timeout_seconds=timeout_seconds,
    )

    resolved_root_workspaces = root_workspaces or [
        default_root_workspace_path(workspace) for workspace in workspaces
    ]

    reference_values: dict[str, float] = {}
    for workspace in workspaces:
        print(f"Computing PyHS3 reference for {workspace.name}", flush=True)
        reference_values[str(workspace)] = evaluate_pyhs3(
            workspace,
            target,
            mode,
            distribution,
        )

    results: list[dict[str, Any]] = []
    for workspace, root_workspace in zip(
        workspaces, resolved_root_workspaces, strict=True
    ):
        for framework in frameworks:
            for n_eval in n_evaluations:
                print(
                    f"Running workspace={workspace.name}, framework={framework}, "
                    f"n_evaluations={n_eval}",
                    flush=True,
                )
                result = run_with_timeout(
                    build_payload(
                        framework=framework,
                        workspace_path=workspace,
                        root_workspace_path=root_workspace
                        if framework == "root"
                        else None,
                        target=target,
                        mode=mode,
                        distribution=distribution,
                        n_evaluations=n_eval,
                        rtol=rtol,
                        atol=atol,
                        reference_value=reference_values[str(workspace)],
                    ),
                    timeout_seconds,
                )
                results.append(result)
                print_result(result)

    summary = summarize_status(results)
    output_data: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "summary": summary,
        "configuration": {
            "frameworks": frameworks,
            "workspaces": [str(workspace) for workspace in workspaces],
            "root_workspaces": [
                str(workspace) for workspace in resolved_root_workspaces
            ],
            "target": target,
            "mode": mode,
            "distribution": distribution,
            "n_evaluations": n_evaluations,
            "rtol": rtol,
            "atol": atol,
            "timeout_seconds": timeout_seconds,
            "reference_framework": REFERENCE_FRAMEWORK,
        },
        "reference_values": reference_values,
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
            workspaces=list(args.workspaces),
            root_workspaces=list(args.root_workspaces)
            if args.root_workspaces
            else None,
            target=args.target,
            mode=args.mode,
            distribution=args.distribution,
            n_evaluations=list(args.n_evaluations),
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
            "Cross-framework scalar PDF evaluation benchmark did not complete"
        ) from error


if __name__ == "__main__":
    main(sys.argv[1:])
