"""Apples-to-apples PyHS3 compiled/non-compiled vs xRooFit ΔNLL benchmark.

The benchmark evaluates the same full extended unbinned likelihood for all
three engines on the same POI grid:

* PyHS3 non-compiled baseline: configurable PyTensor mode, default FAST_COMPILE.
* PyHS3 compiled/optimized: configurable PyTensor mode, default FAST_RUN.
* xRooFit: xRooNode(workspace)["pdfs/sim_pdf"].nll("combData").getVal().

For the generated workspaces the PyHS3 likelihood is evaluated as the sum of
the per-channel extended RooAddPdf-style likelihoods:

    NLL(mu) = Σ_channels [
        Nexp(mu) - Σ_events log(nsig(mu) * sig(x) + nbkg * bkg(x))
    ]

Numerical validation is pairwise:
* PyHS3 non-compiled vs PyHS3 compiled;
* PyHS3 non-compiled vs xRooFit;
* PyHS3 compiled vs xRooFit.

Raw NLL values may differ between frameworks by additive constants, therefore
cross-framework validation is based primarily on ΔNLL shapes and the minimum
POI position. The two PyHS3 modes are also checked in absolute NLL space.

Timing phases are kept separate:
* workspace loading;
* model/node construction;
* NLL construction;
* compilation/optimized graph construction;
* first evaluation;
* repeated steady-state single-point evaluations;
* repeated full scans.

Before running, activate xRooFit:

    source external/xroofit/build/setup.sh

Then run through the canonical pixi environment.
"""

from __future__ import annotations

import argparse
import gc
import math
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
from pyhs3.workspace import Workspace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import PLOTS_DIR, RESULTS_DIR
    from src.utils import get_current_rss_mb, get_peak_rss_mb, save_json
else:
    from .config import PLOTS_DIR, RESULTS_DIR
    from .utils import get_current_rss_mb, get_peak_rss_mb, save_json

try:
    import ROOT
except ImportError:
    ROOT = None


BENCHMARK_NAME = "pyhs3_xroofit_benchmark"
DEFAULT_OUTPUT = RESULTS_DIR / BENCHMARK_NAME / f"{BENCHMARK_NAME}_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME
DEFAULT_DELTA_TOLERANCE = 1e-6
DEFAULT_DELTA_RELATIVE_TOLERANCE = 1e-7
DEFAULT_ABSOLUTE_PYHS3_TOLERANCE = 1e-10
DEFAULT_MINIMUM_TOLERANCE = 1e-12

ENGINE_ORDER = (
    "pyhs3_noncompiled",
    "pyhs3_compiled",
    "xroofit",
)

ENGINE_STYLE = {
    "pyhs3_noncompiled": {
        "label": "PyHS3 non-compiled",
        "color": "#4477AA",
        "marker": "s",
        "linestyle": "-.",
    },
    "pyhs3_compiled": {
        "label": "PyHS3 compiled",
        "color": "#228833",
        "marker": "o",
        "linestyle": "-",
    },
    "xroofit": {
        "label": "xRooFit",
        "color": "#CC6677",
        "marker": "D",
        "linestyle": "--",
    },
}


class ValidationFailure(RuntimeError):
    """Raised when a benchmark returns numerically invalid data."""


@dataclass
class PyHS3Case:
    model: Any
    target: str
    params: dict[str, Any]
    poi: str
    nll_mode: str
    signal_pdf: str
    background_pdf: str
    signal_yield_param: str
    background_yield_param: str
    initial_poi: float
    engine_mode: str
    phase_timings: dict[str, float] = field(default_factory=dict)


@dataclass
class CombinedPyHS3Case:
    channels: tuple[PyHS3Case, ...]
    poi: str
    initial_poi: float
    engine_mode: str
    phase_timings: dict[str, float] = field(default_factory=dict)


@dataclass
class XRooFitCase:
    root_file: Any
    workspace: Any
    root_node: Any
    model_node: Any
    resolved_model_name: str
    nll: Any
    poi: str
    poi_var: Any
    initial_poi: float
    initial_constant: bool
    xroofit_node_python_type: str
    xroofit_model_node_python_type: str
    xroofit_nll_python_type: str
    xroofit_nll_cpp_class: str
    xroofit_runtime_verified: bool
    phase_timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineSpec:
    name: str
    build_func: Callable[[], Any]
    eval_func: Callable[[Any, float], float]
    restore_func: Callable[[Any], None]
    operational_definition: str


def validate_existing_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def validate_positive_int(value: int, name: str, minimum: int = 1) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")


def validate_finite_float(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")


def validate_scan_config(
    *,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
    n_warmup_evaluations: int,
    n_evaluation_runs: int,
    n_scan_runs: int,
    delta_tolerance: float,
    delta_relative_tolerance: float,
    absolute_pyhs3_tolerance: float,
    minimum_tolerance: float,
) -> None:
    for value, name in (
        (scan_min, "scan_min"),
        (scan_max, "scan_max"),
        (delta_tolerance, "delta_tolerance"),
        (delta_relative_tolerance, "delta_relative_tolerance"),
        (absolute_pyhs3_tolerance, "absolute_pyhs3_tolerance"),
        (minimum_tolerance, "minimum_tolerance"),
    ):
        validate_finite_float(value, name)

    validate_positive_int(n_scan_points, "n_scan_points", minimum=2)
    validate_positive_int(n_warmup_evaluations, "n_warmup_evaluations", minimum=0)
    validate_positive_int(n_evaluation_runs, "n_evaluation_runs", minimum=1)
    validate_positive_int(n_scan_runs, "n_scan_runs", minimum=1)

    if scan_min >= scan_max:
        raise ValueError(
            f"scan_min must be smaller than scan_max, got {scan_min} >= {scan_max}"
        )
    if delta_tolerance <= 0.0:
        raise ValueError("delta_tolerance must be positive")
    if delta_relative_tolerance < 0.0:
        raise ValueError("delta_relative_tolerance must be non-negative")
    if absolute_pyhs3_tolerance <= 0.0:
        raise ValueError("absolute_pyhs3_tolerance must be positive")
    if minimum_tolerance <= 0.0:
        raise ValueError("minimum_tolerance must be positive")


def channel_from_analysis(analysis_name: str) -> str:
    if not analysis_name.startswith("L_"):
        raise ValueError(
            "Cannot infer channel from analysis name. Use L_ch0 or pass explicit names. "
            f"Got: {analysis_name}"
        )
    return analysis_name.replace("L_", "", 1)


def default_target_from_analysis(analysis_name: str) -> str:
    return f"model_{channel_from_analysis(analysis_name)}"


def default_data_name_from_analysis(analysis_name: str) -> str:
    return f"combData_{channel_from_analysis(analysis_name)}"


def default_signal_pdf_from_analysis(analysis_name: str) -> str:
    return f"sig_{channel_from_analysis(analysis_name)}"


def default_background_pdf_from_analysis(analysis_name: str) -> str:
    return f"bkg_{channel_from_analysis(analysis_name)}"


def default_signal_yield_from_analysis(analysis_name: str) -> str:
    return f"nsig_{channel_from_analysis(analysis_name)}"


def default_background_yield_from_analysis(analysis_name: str) -> str:
    return f"nbkg_{channel_from_analysis(analysis_name)}"


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _scalar(value: Any, name: str) -> float:
    array = _as_array(value)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    result = float(array[0])
    validate_finite_float(result, name)
    return result


def extract_parameter_point(
    workspace: Workspace, parameter_point: str | None
) -> dict[str, Any]:
    try:
        points = workspace.parameter_points.root
    except AttributeError as exc:
        raise ValueError(
            "PyHS3 workspace does not contain parameter_points.root"
        ) from exc
    if not points:
        raise ValueError("PyHS3 workspace does not contain any parameter points")

    selected = (
        points[0]
        if parameter_point is None
        else next(
            (
                point
                for point in points
                if getattr(point, "name", None) == parameter_point
            ),
            None,
        )
    )
    if selected is None:
        available = [getattr(point, "name", "<unnamed>") for point in points]
        raise KeyError(
            f"Could not find parameter point {parameter_point!r}. Available: {available}"
        )

    params: dict[str, Any] = {}
    for parameter in selected.parameters:
        try:
            params[parameter.name] = np.asarray(
                float(parameter.value), dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Parameter {parameter.name!r} cannot be converted to float"
            ) from exc
    return params


def get_pyhs3_data_values(
    workspace: Workspace, data_name: str, observable_index: int = 0
) -> np.ndarray:
    try:
        data_entries = workspace.data.root
    except AttributeError as exc:
        raise ValueError("PyHS3 workspace does not contain data.root") from exc

    for data in data_entries:
        if data.name == data_name:
            values = np.asarray(
                [entry[observable_index] for entry in data.entries],
                dtype=np.float64,
            )
            if values.size == 0:
                raise ValueError(f"PyHS3 data {data_name!r} is empty")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"PyHS3 data {data_name!r} contains non-finite values")
            return values

    available = [getattr(data, "name", "<unnamed>") for data in data_entries]
    raise KeyError(
        f"Could not find PyHS3 data {data_name!r}. Available data: {available}"
    )


def infer_combined_channels(json_path: Path, prefix: str = "combData_ch") -> list[str]:
    workspace = Workspace.load(json_path)
    try:
        data_entries = workspace.data.root
    except AttributeError as exc:
        raise ValueError("PyHS3 workspace does not contain data.root") from exc

    channels: list[tuple[int, str]] = []
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for data in data_entries:
        name = getattr(data, "name", "")
        match = pattern.match(name)
        if match:
            channels.append((int(match.group(1)), f"ch{match.group(1)}"))

    if not channels:
        available = [getattr(data, "name", "<unnamed>") for data in data_entries]
        raise ValueError(
            "Could not infer combined PyHS3 channels from data names. "
            f"Expected names like {prefix}0. Available data: {available}"
        )

    return [channel for _, channel in sorted(channels)]


def resolve_parameter_name(
    available_names: Iterable[str],
    requested_name: str,
    *,
    context: str,
) -> str:
    """Resolve explicit, documented POI aliases.

    Generated JSON and ROOT workspaces may expose the same signal-strength POI
    as ``mu_sig`` and ``mu`` respectively. The benchmark accepts either spelling,
    prints the mapping, and stores the resolved name in each engine result.
    """

    available = {str(name) for name in available_names}
    if requested_name in available:
        return requested_name

    aliases = {
        "mu": ("mu_sig",),
        "mu_sig": ("mu",),
    }
    for candidate in aliases.get(requested_name, ()):
        if candidate in available:
            print(f"Resolved {context} POI alias: {requested_name!r} -> {candidate!r}")
            return candidate

    raise KeyError(
        f"POI {requested_name!r} is not present in {context}. "
        f"Available parameters: {sorted(available)}"
    )


def build_pyhs3_case_from_loaded_workspace(
    *,
    workspace: Workspace,
    analysis_name: str,
    target: str,
    data_name: str,
    poi: str,
    parameter_point: str | None,
    observable_name: str,
    observable_index: int,
    mode: str,
    nll_mode: str,
    signal_pdf: str,
    background_pdf: str,
    signal_yield_param: str,
    background_yield_param: str,
) -> PyHS3Case:
    model_start = time.perf_counter()
    model = workspace.model(analysis_name, progress=False, mode=mode)
    model_time = time.perf_counter() - model_start

    params = extract_parameter_point(workspace, parameter_point)
    try:
        for name, value in model.free_params.items():
            params[name] = np.asarray(value, dtype=np.float64)
    except AttributeError:
        pass

    params[observable_name] = get_pyhs3_data_values(
        workspace, data_name, observable_index
    )
    resolved_poi = resolve_parameter_name(
        params.keys(),
        poi,
        context=f"PyHS3 analysis {analysis_name!r}",
    )

    return PyHS3Case(
        model=model,
        target=target,
        params=params,
        poi=resolved_poi,
        nll_mode=nll_mode,
        signal_pdf=signal_pdf,
        background_pdf=background_pdf,
        signal_yield_param=signal_yield_param,
        background_yield_param=background_yield_param,
        initial_poi=_scalar(params[resolved_poi], resolved_poi),
        engine_mode=mode,
        phase_timings={
            "workspace_loading_seconds": 0.0,
            "model_construction_seconds": model_time,
            "nll_construction_seconds": 0.0,
            "compilation_seconds": 0.0,
        },
    )


def build_pyhs3_case(
    *,
    json_path: Path,
    analysis_name: str,
    target: str,
    data_name: str,
    poi: str,
    parameter_point: str | None,
    observable_name: str,
    observable_index: int,
    mode: str,
    nll_mode: str,
    signal_pdf: str,
    background_pdf: str,
    signal_yield_param: str,
    background_yield_param: str,
) -> PyHS3Case:
    validate_existing_file(json_path, "PyHS3 JSON workspace")
    load_start = time.perf_counter()
    workspace = Workspace.load(json_path)
    load_time = time.perf_counter() - load_start

    case = build_pyhs3_case_from_loaded_workspace(
        workspace=workspace,
        analysis_name=analysis_name,
        target=target,
        data_name=data_name,
        poi=poi,
        parameter_point=parameter_point,
        observable_name=observable_name,
        observable_index=observable_index,
        mode=mode,
        nll_mode=nll_mode,
        signal_pdf=signal_pdf,
        background_pdf=background_pdf,
        signal_yield_param=signal_yield_param,
        background_yield_param=background_yield_param,
    )
    case.phase_timings["workspace_loading_seconds"] = load_time
    return case


def build_combined_pyhs3_case(
    *,
    json_path: Path,
    channels: list[str],
    poi: str,
    parameter_point: str | None,
    observable_name: str,
    observable_index: int,
    mode: str,
    nll_mode: str,
) -> CombinedPyHS3Case:
    validate_existing_file(json_path, "PyHS3 JSON workspace")

    load_start = time.perf_counter()
    workspace = Workspace.load(json_path)
    load_time = time.perf_counter() - load_start

    cases: list[PyHS3Case] = []
    model_total = 0.0
    for channel in channels:
        analysis_name = f"L_{channel}"
        channel_case = build_pyhs3_case_from_loaded_workspace(
            workspace=workspace,
            analysis_name=analysis_name,
            target=f"model_{channel}",
            data_name=f"combData_{channel}",
            poi=poi,
            parameter_point=parameter_point,
            observable_name=observable_name,
            observable_index=observable_index,
            mode=mode,
            nll_mode=nll_mode,
            signal_pdf=f"sig_{channel}",
            background_pdf=f"bkg_{channel}",
            signal_yield_param=f"nsig_{channel}",
            background_yield_param=f"nbkg_{channel}",
        )
        model_total += channel_case.phase_timings["model_construction_seconds"]
        cases.append(channel_case)

    resolved_pois = {case.poi for case in cases}
    if len(resolved_pois) != 1:
        raise ValidationFailure(
            f"Channels resolved inconsistent POI names: {sorted(resolved_pois)}"
        )

    initial_values = {round(case.initial_poi, 15) for case in cases}
    if len(initial_values) != 1:
        raise ValidationFailure(
            f"Channels have inconsistent initial POI values: {sorted(initial_values)}"
        )

    return CombinedPyHS3Case(
        channels=tuple(cases),
        poi=cases[0].poi,
        initial_poi=cases[0].initial_poi,
        engine_mode=mode,
        phase_timings={
            "workspace_loading_seconds": load_time,
            "model_construction_seconds": model_total,
            "nll_construction_seconds": 0.0,
            "compilation_seconds": 0.0,
        },
    )


def pyhs3_logpdf_nll(case: PyHS3Case, value: float) -> float:
    eval_params = dict(case.params)
    eval_params[case.poi] = np.asarray(value, dtype=np.float64)
    logpdf = np.asarray(
        case.model.logpdf(case.target, **eval_params),
        dtype=np.float64,
    )
    if logpdf.size == 0 or not np.all(np.isfinite(logpdf)):
        raise ValidationFailure(
            f"PyHS3 returned invalid logpdf values for {case.target}"
        )
    return -float(np.sum(logpdf))


def pyhs3_extended_mixture_nll(case: PyHS3Case, value: float) -> float:
    validate_finite_float(value, case.poi)
    eval_params = dict(case.params)
    eval_params[case.poi] = np.asarray(value, dtype=np.float64)

    try:
        nominal_signal_yield = _scalar(
            eval_params[case.signal_yield_param], case.signal_yield_param
        )
        background_yield = _scalar(
            eval_params[case.background_yield_param], case.background_yield_param
        )
    except KeyError as exc:
        raise KeyError(
            "Missing yield parameter for extended-mixture NLL. "
            f"Need {case.signal_yield_param!r} and "
            f"{case.background_yield_param!r}."
        ) from exc

    signal_yield = float(value) * nominal_signal_yield
    expected_events = signal_yield + background_yield

    sig_pdf = _as_array(case.model.pdf(case.signal_pdf, **eval_params))
    bkg_pdf = _as_array(case.model.pdf(case.background_pdf, **eval_params))
    if sig_pdf.shape != bkg_pdf.shape:
        raise ValidationFailure(
            "Signal/background PDF arrays have different shapes: "
            f"{sig_pdf.shape} vs {bkg_pdf.shape}"
        )

    event_density = signal_yield * sig_pdf + background_yield * bkg_pdf
    if (
        event_density.size == 0
        or not np.all(np.isfinite(event_density))
        or np.any(event_density <= 0.0)
    ):
        raise ValidationFailure("PyHS3 extended-mixture event densities are invalid")

    return float(expected_events - np.sum(np.log(event_density)))


def pyhs3_nll(case: PyHS3Case | CombinedPyHS3Case, value: float) -> float:
    if isinstance(case, CombinedPyHS3Case):
        return float(
            sum(pyhs3_nll(channel_case, value) for channel_case in case.channels)
        )
    if case.nll_mode == "logpdf":
        return pyhs3_logpdf_nll(case, value)
    if case.nll_mode == "extended-mixture":
        return pyhs3_extended_mixture_nll(case, value)
    raise ValueError(f"Unknown PyHS3 NLL mode: {case.nll_mode}")


def restore_pyhs3_case(case: PyHS3Case | CombinedPyHS3Case) -> None:
    # PyHS3 evaluations use fresh parameter dictionaries and do not mutate
    # model state. This explicit check documents and verifies that invariant.
    if isinstance(case, CombinedPyHS3Case):
        for channel_case in case.channels:
            actual = _scalar(channel_case.params[channel_case.poi], channel_case.poi)
            if not math.isclose(
                actual,
                channel_case.initial_poi,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValidationFailure(
                    f"PyHS3 parameter dictionary mutated: "
                    f"{channel_case.poi}={actual}, expected "
                    f"{channel_case.initial_poi}"
                )
    else:
        actual = _scalar(case.params[case.poi], case.poi)
        if not math.isclose(
            actual,
            case.initial_poi,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValidationFailure(
                f"PyHS3 parameter dictionary mutated: {case.poi}={actual}, "
                f"expected {case.initial_poi}"
            )


def require_xroofit(xroofit_library: str | None = "libxRooFit") -> Any:
    if ROOT is None:
        raise RuntimeError("ROOT is not available in this environment")

    if xroofit_library:
        load_status = int(ROOT.gSystem.Load(xroofit_library))
        if load_status < 0 and not hasattr(ROOT, "xRooNode"):
            raise RuntimeError(
                "Could not load xRooFit. First run:\n"
                "  source external/xroofit/build/setup.sh\n"
                "Then execute the benchmark through pixi, or pass:\n"
                "  --xroofit-library /absolute/path/to/libxRooFit.so"
            )

    if not hasattr(ROOT, "xRooNode"):
        raise RuntimeError("xRooFit is not available in this ROOT/PyROOT session")
    return ROOT


def _is_valid_root_object(obj: Any) -> bool:
    if obj is None:
        return False
    try:
        return bool(obj)
    except Exception:
        return True


def _find_workspace(root_file: Any, workspace_name: str) -> Any:
    workspace = root_file.Get(workspace_name) if workspace_name else None
    if _is_valid_root_object(workspace):
        return workspace

    import ROOT

    for key in root_file.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom(ROOT.RooWorkspace.Class()):
            return obj
    raise RuntimeError(f"Could not find RooWorkspace {workspace_name!r}")


def _candidate_xroofit_model_paths(model_name: str) -> list[str]:
    if not model_name:
        raise ValueError("xRooFit model name must not be empty")
    if "/" in model_name:
        return [model_name]

    candidates = [model_name]
    if model_name == "ModelConfig" or model_name.startswith("L_"):
        candidates.insert(0, f"models/{model_name}")
    else:
        candidates.insert(0, f"pdfs/{model_name}")
    return list(dict.fromkeys(candidates))


def _get_xroofit_node(root_node: Any, model_name: str) -> tuple[Any, str]:
    errors: list[str] = []
    for candidate in _candidate_xroofit_model_paths(model_name):
        try:
            node = root_node[candidate]
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
        if _is_valid_root_object(node):
            return node, candidate
        errors.append(f"{candidate}: null/invalid xRooNode")

    raise RuntimeError(
        f"Could not access xRooFit model node {model_name!r}. "
        f"Tried: {', '.join(_candidate_xroofit_model_paths(model_name))}. "
        f"Details: {'; '.join(errors)}"
    )


def _set_root_defaults_from_pyhs3(
    workspace: Any,
    json_path: Path,
    parameter_point: str | None,
) -> dict[str, Any]:
    """Apply only JSON defaults that are valid ROOT parameter values.

    A JSON parameter can represent a different convention or transformation
    than the same-named RooRealVar. Values outside the ROOT range are not clipped
    or forced because that would silently change the statistical model.
    """

    pyhs3_workspace = Workspace.load(json_path)
    parameters = extract_parameter_point(pyhs3_workspace, parameter_point)
    applied: dict[str, float] = {}
    skipped: dict[str, str] = {}

    for name, value in parameters.items():
        var = workspace.var(str(name))
        if not _is_valid_root_object(var):
            skipped[name] = "variable not present in ROOT workspace"
            continue

        scalar_value = _scalar(value, name)
        try:
            minimum = float(var.getMin())
            maximum = float(var.getMax())
        except Exception:
            minimum = -math.inf
            maximum = math.inf

        if scalar_value < minimum or scalar_value > maximum:
            skipped[name] = (
                f"value {scalar_value} outside ROOT range [{minimum}, {maximum}]"
            )
            continue

        try:
            var.setVal(scalar_value)
            applied[name] = scalar_value
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"

    if skipped:
        print("ROOT defaults not applied for incompatible parameters:")
        for name, reason in sorted(skipped.items()):
            print(f"  {name}: {reason}")

    return {"applied": applied, "skipped": skipped}


def _construct_xroofit_nll(model_node: Any, dataset_name: str) -> Any:
    try:
        return model_node.nll(dataset_name)
    except TypeError as exc:
        raise RuntimeError(
            f"xRooFit could not construct an NLL from dataset {dataset_name!r}. "
            "Use a PDF node such as 'pdfs/sim_pdf'."
        ) from exc


def _get_workspace_poi(workspace: Any, poi: str) -> tuple[Any, str]:
    candidates = [poi]
    if poi == "mu":
        candidates.append("mu_sig")
    elif poi == "mu_sig":
        candidates.append("mu")

    for candidate in candidates:
        var = workspace.var(candidate)
        if _is_valid_root_object(var):
            if candidate != poi:
                print(f"Resolved xRooFit POI alias: {poi!r} -> {candidate!r}")
            return var, candidate

    raise RuntimeError(f"Could not find xRooFit/RooFit POI {poi!r}; tried {candidates}")


def _python_type_name(obj: Any) -> str:
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


def _root_cpp_class_name(obj: Any) -> str:
    """Return the dynamic C++ class name when ROOT exposes one."""

    for accessor in ("ClassName", "IsA"):
        try:
            value = getattr(obj, accessor)()
            if accessor == "IsA" and value:
                return str(value.GetName())
            if value:
                return str(value)
        except Exception:
            continue
    return "<unavailable>"


def _verify_xroofit_runtime(
    *,
    root_node: Any,
    model_node: Any,
    nll: Any,
) -> dict[str, Any]:
    """Verify that benchmark construction uses the xRooFit API path.

    The check intentionally rejects a direct RooFit-only path such as
    ``workspace.pdf(...).createNLL(...)``. It requires xRooNode wrappers and an
    NLL object produced by ``xRooNode.nll(...)``.
    """

    node_type = _python_type_name(root_node)
    model_node_type = _python_type_name(model_node)
    nll_type = _python_type_name(nll)
    nll_cpp_class = _root_cpp_class_name(nll)

    root_node_is_xroo = "xRooNode" in node_type
    model_node_is_xroo = "xRooNode" in model_node_type
    nll_looks_xroo = (
        "xRooNLL" in nll_type or "xRooNLL" in nll_cpp_class or "xRooNode" in nll_type
    )

    verified = root_node_is_xroo and model_node_is_xroo and nll_looks_xroo
    if not verified:
        raise RuntimeError(
            "xRooFit runtime verification failed. "
            f"root_node={node_type}, model_node={model_node_type}, "
            f"nll_python_type={nll_type}, nll_cpp_class={nll_cpp_class}. "
            "The benchmark requires ROOT.xRooNode(...).nll(...), not a direct "
            "RooFit createNLL path."
        )

    return {
        "root_node_python_type": node_type,
        "model_node_python_type": model_node_type,
        "nll_python_type": nll_type,
        "nll_cpp_class": nll_cpp_class,
        "verified": True,
    }


def build_xroofit_case(
    *,
    root_path: Path,
    json_path: Path,
    workspace_name: str,
    model_name: str,
    dataset_name: str,
    poi: str,
    parameter_point: str | None,
    xroofit_library: str | None,
) -> XRooFitCase:
    root = require_xroofit(xroofit_library)
    validate_existing_file(root_path, "xRooFit ROOT workspace")

    load_start = time.perf_counter()
    root_file = root.TFile.Open(str(root_path), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file {root_path}")
    workspace = _find_workspace(root_file, workspace_name)
    root_default_sync = _set_root_defaults_from_pyhs3(
        workspace, json_path, parameter_point
    )
    load_time = time.perf_counter() - load_start

    try:
        model_start = time.perf_counter()
        root_node = root.xRooNode(workspace)
        model_node, resolved_model_name = _get_xroofit_node(root_node, model_name)
        model_time = time.perf_counter() - model_start

        poi_var, resolved_poi = _get_workspace_poi(workspace, poi)
        initial_poi = float(poi_var.getVal())
        initial_constant = bool(poi_var.isConstant())
        poi_var.setConstant(True)

        nll_start = time.perf_counter()
        nll = _construct_xroofit_nll(model_node, dataset_name)
        nll_time = time.perf_counter() - nll_start
        xroofit_identity = _verify_xroofit_runtime(
            root_node=root_node,
            model_node=model_node,
            nll=nll,
        )

        return XRooFitCase(
            root_file=root_file,
            workspace=workspace,
            root_node=root_node,
            model_node=model_node,
            resolved_model_name=resolved_model_name,
            nll=nll,
            poi=resolved_poi,
            poi_var=poi_var,
            initial_poi=initial_poi,
            initial_constant=initial_constant,
            xroofit_node_python_type=xroofit_identity["root_node_python_type"],
            xroofit_model_node_python_type=xroofit_identity["model_node_python_type"],
            xroofit_nll_python_type=xroofit_identity["nll_python_type"],
            xroofit_nll_cpp_class=xroofit_identity["nll_cpp_class"],
            xroofit_runtime_verified=xroofit_identity["verified"],
            phase_timings={
                "workspace_loading_seconds": load_time,
                "model_construction_seconds": model_time,
                "nll_construction_seconds": nll_time,
                "compilation_seconds": 0.0,
                "root_defaults_applied_count": len(root_default_sync["applied"]),
                "root_defaults_skipped_count": len(root_default_sync["skipped"]),
            },
        )
    except Exception:
        root_file.Close()
        raise


def _set_xroofit_parameter(case: XRooFitCase, value: float) -> None:
    validate_finite_float(value, case.poi)
    case.poi_var.setVal(float(value))
    case.poi_var.setConstant(True)


def xroofit_nll(case: XRooFitCase, value: float) -> float:
    _set_xroofit_parameter(case, value)
    try:
        nll_value = float(case.nll.getVal())
    except AttributeError:
        nll_value = float(case.nll)

    if not math.isfinite(nll_value):
        raise ValidationFailure(f"xRooFit returned non-finite NLL value: {nll_value}")
    return nll_value


def restore_xroofit_case(case: XRooFitCase) -> None:
    case.poi_var.setVal(case.initial_poi)
    case.poi_var.setConstant(case.initial_constant)

    restored_value = float(case.poi_var.getVal())
    restored_constant = bool(case.poi_var.isConstant())
    if not math.isclose(
        restored_value,
        case.initial_poi,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValidationFailure(
            f"xRooFit POI was not restored: {restored_value} != {case.initial_poi}"
        )
    if restored_constant != case.initial_constant:
        raise ValidationFailure(
            "xRooFit POI constant state was not restored: "
            f"{restored_constant} != {case.initial_constant}"
        )


def close_case(case: Any) -> None:
    if isinstance(case, XRooFitCase):
        try:
            restore_xroofit_case(case)
        finally:
            try:
                case.root_file.Close()
            except Exception:
                pass


def validate_numeric_sequence(values: Iterable[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must not be empty")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


def delta_nll(values: Iterable[float]) -> list[float]:
    checked = validate_numeric_sequence(values, "nll_values")
    minimum = min(checked)
    return [float(value - minimum) for value in checked]


def minimum_position(scan_values: list[float], nll_values: list[float]) -> float:
    if len(scan_values) != len(nll_values):
        raise ValueError("scan_values and nll_values must have the same length")
    return float(scan_values[int(np.argmin(np.asarray(nll_values, dtype=np.float64)))])


def summarize_timings(values: Iterable[float]) -> dict[str, float]:
    checked = validate_numeric_sequence(values, "timings")
    array = np.asarray(checked, dtype=np.float64)
    q1, q3 = np.percentile(array, [25.0, 75.0])
    return {
        "count": int(array.size),
        "mean_seconds": mean(checked),
        "median_seconds": median(checked),
        "std_seconds": stdev(checked) if len(checked) > 1 else 0.0,
        "q1_seconds": float(q1),
        "q3_seconds": float(q3),
        "iqr_seconds": float(q3 - q1),
        "min_seconds": min(checked),
        "max_seconds": max(checked),
    }


def scan_nll(
    func: Callable[[float], float],
    scan_values: list[float],
) -> tuple[list[float], float]:
    values: list[float] = []
    start = time.perf_counter()
    for scan_value in scan_values:
        values.append(float(func(scan_value)))
    duration = time.perf_counter() - start
    validate_numeric_sequence(values, "scan_nll_values")
    return values, duration


def assert_scans_repeat(
    reference: list[float],
    candidate: list[float],
    *,
    engine_name: str,
    run_index: int,
    absolute_tolerance: float,
) -> None:
    if not np.allclose(
        np.asarray(reference, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
        rtol=0.0,
        atol=absolute_tolerance,
    ):
        max_diff = float(
            np.max(
                np.abs(
                    np.asarray(reference, dtype=np.float64)
                    - np.asarray(candidate, dtype=np.float64)
                )
            )
        )
        raise ValidationFailure(
            f"{engine_name} full scan run {run_index} differs from the "
            f"reference scan; max absolute difference={max_diff:.3e}"
        )


def build_steady_state_poi_values(
    scan_values: list[float],
    n_values: int,
    *,
    avoid_first: float | None = None,
) -> list[float]:
    """Create a deterministic changing-POI sequence for timing.

    Consecutive repeated evaluations at an identical POI can exercise RooFit
    caches rather than the full engine. This sequence cycles through distinct
    scan-grid values and avoids consecutive duplicates.
    """

    if not scan_values:
        raise ValueError("scan_values must not be empty")
    if n_values < 1:
        raise ValueError("n_values must be positive")

    candidates = [float(value) for value in scan_values]
    if avoid_first is not None and len(candidates) > 1:
        candidates = [
            value
            for value in candidates
            if not math.isclose(value, avoid_first, rel_tol=0.0, abs_tol=1e-15)
        ] or candidates

    # Prefer interior points to avoid pathological boundaries when possible.
    if len(candidates) > 4:
        interior = candidates[1:-1]
        ordered = interior[::2] + interior[1::2]
    else:
        ordered = candidates

    values: list[float] = []
    index = 0
    while len(values) < n_values:
        candidate = ordered[index % len(ordered)]
        if values and math.isclose(candidate, values[-1], rel_tol=0.0, abs_tol=1e-15):
            index += 1
            continue
        values.append(candidate)
        index += 1

    return values


def measure_engine(
    *,
    spec: EngineSpec,
    scan_values: list[float],
    n_warmup_evaluations: int,
    n_evaluation_runs: int,
    n_scan_runs: int,
    poi_value: float,
    repeat_tolerance: float,
) -> dict[str, Any]:
    gc.collect()
    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()
    case = None

    try:
        total_build_start = time.perf_counter()
        case = spec.build_func()
        total_build_time = time.perf_counter() - total_build_start

        phase_timings = dict(getattr(case, "phase_timings", {}))
        phase_timings["total_build_seconds"] = total_build_time

        first_start = time.perf_counter()
        first_nll = float(spec.eval_func(case, poi_value))
        first_evaluation_time = time.perf_counter() - first_start
        if not math.isfinite(first_nll):
            raise ValidationFailure(
                f"{spec.name} returned non-finite first NLL: {first_nll}"
            )

        # Force lazy initialization with changing POI values so RooFit cannot
        # satisfy the benchmark from a repeated-value cache.
        warmup_poi_values = build_steady_state_poi_values(
            scan_values,
            max(1, n_warmup_evaluations),
            avoid_first=poi_value,
        )
        for warmup_poi in warmup_poi_values[:n_warmup_evaluations]:
            warmup_nll = float(spec.eval_func(case, warmup_poi))
            if not math.isfinite(warmup_nll):
                raise ValidationFailure(f"{spec.name} returned non-finite warm-up NLL")

        steady_poi_values = build_steady_state_poi_values(
            scan_values,
            n_evaluation_runs,
            avoid_first=warmup_poi_values[-1] if warmup_poi_values else poi_value,
        )
        steady_timings: list[float] = []
        steady_nll_values: list[float] = []
        for steady_poi in steady_poi_values:
            start = time.perf_counter()
            value = float(spec.eval_func(case, steady_poi))
            duration = time.perf_counter() - start
            if duration <= 0.0 or not math.isfinite(duration):
                raise ValidationFailure(
                    f"{spec.name} steady-state timing is invalid: {duration}"
                )
            if not math.isfinite(value):
                raise ValidationFailure(
                    f"{spec.name} returned non-finite steady-state NLL"
                )
            steady_timings.append(duration)
            steady_nll_values.append(value)

        # Re-evaluate the same changing sequence outside the timed section and
        # require exact repeatability within the configured engine tolerance.
        repeated_values = [
            float(spec.eval_func(case, steady_poi)) for steady_poi in steady_poi_values
        ]
        if not np.allclose(
            np.asarray(steady_nll_values, dtype=np.float64),
            np.asarray(repeated_values, dtype=np.float64),
            rtol=0.0,
            atol=repeat_tolerance,
        ):
            raise ValidationFailure(
                f"{spec.name} changing-POI steady-state NLL values are not stable"
            )

        scan_timings: list[float] = []
        reference_scan: list[float] | None = None
        for run_index in range(n_scan_runs):
            scan_result, scan_duration = scan_nll(
                lambda value: spec.eval_func(case, value),
                scan_values,
            )
            scan_timings.append(scan_duration)
            if reference_scan is None:
                reference_scan = scan_result
            else:
                assert_scans_repeat(
                    reference_scan,
                    scan_result,
                    engine_name=spec.name,
                    run_index=run_index,
                    absolute_tolerance=repeat_tolerance,
                )
            spec.restore_func(case)

        if reference_scan is None:
            raise RuntimeError("No full scan was executed")

        current_rss_after_mb = get_current_rss_mb()
        peak_rss_after_mb = get_peak_rss_mb()
        scan_summary = summarize_timings(scan_timings)
        evaluation_summary = summarize_timings(steady_timings)

        runtime_identity: dict[str, Any] = {}
        if isinstance(case, XRooFitCase):
            runtime_identity = {
                "xroofit_runtime_verified": case.xroofit_runtime_verified,
                "xroofit_node_python_type": case.xroofit_node_python_type,
                "xroofit_model_node_python_type": (case.xroofit_model_node_python_type),
                "xroofit_nll_python_type": case.xroofit_nll_python_type,
                "xroofit_nll_cpp_class": case.xroofit_nll_cpp_class,
                "xroofit_api_path": (
                    "ROOT.xRooNode(workspace)[model].nll(dataset).getVal()"
                ),
                "direct_roofit_create_nll_used": False,
            }

        return {
            "engine": spec.name,
            "framework": spec.name,
            "engine_label": ENGINE_STYLE[spec.name]["label"],
            "framework_label": ENGINE_STYLE[spec.name]["label"],
            "operational_definition": spec.operational_definition,
            "status": "success",
            "n_warmup_evaluations": n_warmup_evaluations,
            "n_evaluation_runs": n_evaluation_runs,
            "n_scan_runs": n_scan_runs,
            "n_scan_points": len(scan_values),
            "phase_timings": phase_timings,
            "build_time_seconds": total_build_time,
            "workspace_loading_time_seconds": phase_timings.get(
                "workspace_loading_seconds", 0.0
            ),
            "model_construction_time_seconds": phase_timings.get(
                "model_construction_seconds", 0.0
            ),
            "nll_construction_time_seconds": phase_timings.get(
                "nll_construction_seconds", 0.0
            ),
            "compilation_time_seconds": phase_timings.get("compilation_seconds", 0.0),
            "cold_first_evaluation_time_seconds": first_evaluation_time,
            "first_nll": first_nll,
            "steady_state_nll": steady_nll_values[-1],
            "steady_state_poi_values": steady_poi_values,
            "steady_state_uses_changing_poi": True,
            "steady_state_evaluation": evaluation_summary,
            "warm_evaluation": evaluation_summary,
            "warm_evaluation_time_seconds_mean": evaluation_summary["mean_seconds"],
            "warm_evaluation_time_seconds_median": evaluation_summary["median_seconds"],
            "full_scan": scan_summary,
            "scan_time_seconds": scan_summary["median_seconds"],
            "scan_time_seconds_median": scan_summary["median_seconds"],
            "time_per_scan_point_seconds": (
                scan_summary["median_seconds"] / len(scan_values)
            ),
            "current_rss_before_mb": current_rss_before_mb,
            "current_rss_after_mb": current_rss_after_mb,
            "current_rss_delta_mb": max(
                0.0, current_rss_after_mb - current_rss_before_mb
            ),
            "peak_rss_before_mb": peak_rss_before_mb,
            "peak_rss_after_mb": peak_rss_after_mb,
            "peak_rss_delta_mb": max(0.0, peak_rss_after_mb - peak_rss_before_mb),
            "scan_nll_values": reference_scan,
            "delta_nll_shape": delta_nll(reference_scan),
            "minimum_poi": minimum_position(scan_values, reference_scan),
            "minimum_index": int(
                np.argmin(np.asarray(reference_scan, dtype=np.float64))
            ),
            "finite_values": True,
            "parameters_restored": True,
            **runtime_identity,
        }
    finally:
        close_case(case)


def pairwise_agreement(
    *,
    left_result: dict[str, Any],
    right_result: dict[str, Any],
    delta_tolerance: float,
    delta_relative_tolerance: float,
    minimum_tolerance: float,
    raw_tolerance: float | None,
) -> dict[str, Any]:
    left_scan = np.asarray(
        left_result["scan_nll_values"],
        dtype=np.float64,
    )
    right_scan = np.asarray(
        right_result["scan_nll_values"],
        dtype=np.float64,
    )
    if left_scan.shape != right_scan.shape:
        raise ValueError("Cannot compare scans with different shapes")

    left_delta = np.asarray(
        left_result["delta_nll_shape"],
        dtype=np.float64,
    )
    right_delta = np.asarray(
        right_result["delta_nll_shape"],
        dtype=np.float64,
    )

    raw_diff = right_scan - left_scan
    delta_diff = right_delta - left_delta
    delta_abs = np.abs(delta_diff)
    minimum_diff = abs(
        float(right_result["minimum_poi"]) - float(left_result["minimum_poi"])
    )
    offset = float(np.mean(raw_diff))
    centered_raw_diff = raw_diff - offset

    delta_scale = np.maximum(
        np.maximum(np.abs(left_delta), np.abs(right_delta)),
        np.finfo(np.float64).eps,
    )
    delta_relative = delta_abs / delta_scale
    delta_success = bool(
        np.allclose(
            left_delta,
            right_delta,
            rtol=delta_relative_tolerance,
            atol=delta_tolerance,
        )
    )
    minimum_success = minimum_diff <= minimum_tolerance
    raw_success = (
        True
        if raw_tolerance is None
        else float(np.max(np.abs(raw_diff))) <= raw_tolerance
    )

    validation_success = delta_success and minimum_success and raw_success

    return {
        "left_engine": left_result["engine"],
        "right_engine": right_result["engine"],
        "left_label": left_result["engine_label"],
        "right_label": right_result["engine_label"],
        "raw_nll_max_abs_diff": float(np.max(np.abs(raw_diff))),
        "raw_nll_mean_abs_diff": float(np.mean(np.abs(raw_diff))),
        "constant_offset_estimate": offset,
        "centered_raw_nll_max_abs_diff": float(np.max(np.abs(centered_raw_diff))),
        "delta_nll_max_abs_diff": float(np.max(delta_abs)),
        "delta_nll_max_relative_diff": float(np.max(delta_relative)),
        "delta_nll_mean_abs_diff": float(np.mean(delta_abs)),
        "delta_nll_rms_diff": float(np.sqrt(np.mean(delta_diff**2))),
        "minimum_poi_abs_diff": minimum_diff,
        "minimum_index_match": bool(
            left_result["minimum_index"] == right_result["minimum_index"]
        ),
        "raw_absolute_check_required": raw_tolerance is not None,
        "raw_tolerance": raw_tolerance,
        "delta_absolute_tolerance": delta_tolerance,
        "delta_relative_tolerance": delta_relative_tolerance,
        "raw_absolute_success": raw_success,
        "delta_shape_success": delta_success,
        "minimum_poi_success": minimum_success,
        "validation_status": "success" if validation_success else "failed",
        "delta_nll_difference": delta_diff.tolist(),
        "raw_nll_difference": raw_diff.tolist(),
    }


def build_all_agreements(
    successful: dict[str, dict[str, Any]],
    *,
    delta_tolerance: float,
    delta_relative_tolerance: float,
    absolute_pyhs3_tolerance: float,
    minimum_tolerance: float,
) -> dict[str, Any]:
    required = set(ENGINE_ORDER)
    if not required.issubset(successful):
        return {
            "validation_status": "not_run",
            "missing_engines": sorted(required - set(successful)),
            "comparisons": {},
        }

    comparisons = {
        "pyhs3_noncompiled_vs_pyhs3_compiled": pairwise_agreement(
            left_result=successful["pyhs3_noncompiled"],
            right_result=successful["pyhs3_compiled"],
            delta_tolerance=delta_tolerance,
            delta_relative_tolerance=delta_relative_tolerance,
            minimum_tolerance=minimum_tolerance,
            raw_tolerance=absolute_pyhs3_tolerance,
        ),
        "pyhs3_noncompiled_vs_xroofit": pairwise_agreement(
            left_result=successful["pyhs3_noncompiled"],
            right_result=successful["xroofit"],
            delta_tolerance=delta_tolerance,
            delta_relative_tolerance=delta_relative_tolerance,
            minimum_tolerance=minimum_tolerance,
            raw_tolerance=None,
        ),
        "pyhs3_compiled_vs_xroofit": pairwise_agreement(
            left_result=successful["pyhs3_compiled"],
            right_result=successful["xroofit"],
            delta_tolerance=delta_tolerance,
            delta_relative_tolerance=delta_relative_tolerance,
            minimum_tolerance=minimum_tolerance,
            raw_tolerance=None,
        ),
    }

    overall = all(
        item["validation_status"] == "success" for item in comparisons.values()
    )
    return {
        "validation_status": "success" if overall else "failed",
        "comparisons": comparisons,
    }


def failed_engine_result(name: str, exc: BaseException) -> dict[str, Any]:
    return {
        "engine": name,
        "framework": name,
        "engine_label": ENGINE_STYLE.get(name, {"label": name})["label"],
        "framework_label": ENGINE_STYLE.get(name, {"label": name})["label"],
        "status": "failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save_figure(fig: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def make_profile_plot(
    results: dict[str, dict[str, Any]],
    scan_values: list[float],
    poi: str,
    output_path: Path,
) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.asarray(scan_values, dtype=np.float64)

    for name in ENGINE_ORDER:
        result = results[name]
        style = ENGINE_STYLE[name]
        ax.plot(
            x,
            result["delta_nll_shape"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            markersize=4,
            markevery=max(1, len(x) // 20),
        )

    ax.set_xlabel(poi)
    ax.set_ylabel(r"$\Delta$NLL")
    ax.set_title(
        "ΔNLL profile comparison",
        loc="left",
        weight="bold",
    )
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def make_residual_plot(
    agreements: dict[str, Any],
    scan_values: list[float],
    poi: str,
    output_path: Path,
) -> None:
    """Plot absolute pointwise ΔNLL differences with explicit engine names."""

    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.asarray(scan_values, dtype=np.float64)

    comparison_styles = {
        "pyhs3_noncompiled_vs_pyhs3_compiled": {
            "label": ("|PyHS3 compiled − PyHS3 non-compiled|"),
            "linestyle": "-",
            "color": "#4477AA",
        },
        "pyhs3_noncompiled_vs_xroofit": {
            "label": ("|xRooFit − PyHS3 non-compiled|"),
            "linestyle": "--",
            "color": "#EE7733",
        },
        "pyhs3_compiled_vs_xroofit": {
            "label": ("|xRooFit − PyHS3 compiled|"),
            "linestyle": "-.",
            "color": "#228833",
        },
    }

    positive_values: list[np.ndarray] = []
    for key, style in comparison_styles.items():
        comparison = agreements["comparisons"][key]
        absolute_difference = np.abs(
            np.asarray(
                comparison["delta_nll_difference"],
                dtype=np.float64,
            )
        )
        positive_values.append(absolute_difference)
        ax.plot(
            x,
            absolute_difference,
            label=style["label"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=1.8,
        )

    all_values = np.concatenate(positive_values)
    strictly_positive = all_values[all_values > 0.0]
    if strictly_positive.size:
        dynamic_range = float(np.max(strictly_positive)) / float(
            np.min(strictly_positive)
        )
        if dynamic_range >= 1e3:
            ax.set_yscale("log")

    ax.set_xlabel(poi)
    ax.set_ylabel(r"Absolute pointwise $\Delta$NLL difference")
    ax.set_title(
        "Pointwise ΔNLL differences across the scan",
        loc="left",
        weight="bold",
    )
    ax.legend(frameon=False)
    _save_figure(fig, output_path)


def _bar_with_iqr(
    *,
    ax: Any,
    results: dict[str, dict[str, Any]],
    summary_key: str,
    scale: float,
    ylabel: str,
    title: str,
) -> None:
    labels = [ENGINE_STYLE[name]["label"] for name in ENGINE_ORDER]
    medians = [
        results[name][summary_key]["median_seconds"] * scale for name in ENGINE_ORDER
    ]
    lower = [
        (
            results[name][summary_key]["median_seconds"]
            - results[name][summary_key]["q1_seconds"]
        )
        * scale
        for name in ENGINE_ORDER
    ]
    upper = [
        (
            results[name][summary_key]["q3_seconds"]
            - results[name][summary_key]["median_seconds"]
        )
        * scale
        for name in ENGINE_ORDER
    ]
    colors = [ENGINE_STYLE[name]["color"] for name in ENGINE_ORDER]

    bars = ax.bar(
        labels,
        medians,
        color=colors,
        edgecolor="black",
        yerr=np.asarray([lower, upper]),
        capsize=5,
    )
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", weight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")

    for bar, value in zip(bars, medians, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def make_steady_runtime_plot(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    _bar_with_iqr(
        ax=ax,
        results=results,
        summary_key="steady_state_evaluation",
        scale=1e6,
        ylabel="Median time per NLL evaluation [µs]",
        title="Steady-state NLL evaluation time (median ± IQR)",
    )
    _save_figure(fig, output_path)


def make_scan_runtime_plot(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    _bar_with_iqr(
        ax=ax,
        results=results,
        summary_key="full_scan",
        scale=1e3,
        ylabel="Median full-scan time [ms]",
        title="Full ΔNLL scan time (median ± IQR)",
    )
    _save_figure(fig, output_path)


def make_phase_breakdown_plot(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    _apply_plot_style()
    labels = [ENGINE_STYLE[name]["label"] for name in ENGINE_ORDER]
    phase_keys = (
        ("workspace_loading_time_seconds", "Workspace loading"),
        ("model_construction_time_seconds", "Model/node construction"),
        ("nll_construction_time_seconds", "NLL construction"),
        ("cold_first_evaluation_time_seconds", "First evaluation / lazy compilation"),
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bottom = np.zeros(len(ENGINE_ORDER), dtype=np.float64)

    phase_values: list[np.ndarray] = []
    for key, _ in phase_keys:
        phase_values.append(
            np.asarray(
                [results[name].get(key, 0.0) * 1e3 for name in ENGINE_ORDER],
                dtype=np.float64,
            )
        )

    totals = np.sum(np.vstack(phase_values), axis=0)
    percentage_label_threshold = 10.0
    absolute_height_threshold_ms = max(float(np.max(totals)) * 0.025, 40.0)

    for (key, label), values in zip(phase_keys, phase_values, strict=True):
        bars = ax.bar(
            labels,
            values,
            bottom=bottom,
            label=label,
            edgecolor="black",
        )

        for index, (bar, value) in enumerate(zip(bars, values, strict=True)):
            total = totals[index]
            percentage = 0.0 if total <= 0.0 else 100.0 * value / total

            # Label only visually meaningful segments to avoid overlaps.
            if (
                percentage >= percentage_label_threshold
                and value >= absolute_height_threshold_ms
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[index] + value / 2,
                    f"{percentage:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

        bottom += values

    ax.set_ylabel("Time [ms]")
    ax.set_title(
        "Setup and first-evaluation time breakdown",
        loc="left",
        weight="bold",
    )
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")
    _save_figure(fig, output_path)


def make_agreement_plot(
    agreements: dict[str, Any],
    delta_tolerance: float,
    delta_relative_tolerance: float,
    output_path: Path,
) -> None:
    _apply_plot_style()
    comparison_order = (
        "pyhs3_noncompiled_vs_pyhs3_compiled",
        "pyhs3_noncompiled_vs_xroofit",
        "pyhs3_compiled_vs_xroofit",
    )
    labels = [
        "non-compiled\nvs compiled",
        "non-compiled\nvs xRooFit",
        "compiled\nvs xRooFit",
    ]
    values = [
        agreements["comparisons"][key]["delta_nll_max_abs_diff"]
        for key in comparison_order
    ]
    plot_values = [max(value, 1e-18) for value in values]

    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    bars = ax.bar(labels, plot_values, edgecolor="black")
    ax.axhline(
        delta_tolerance,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=(
            f"Absolute tolerance = {delta_tolerance:.0e}; "
            f"relative tolerance = {delta_relative_tolerance:.0e}"
        ),
    )
    ax.set_yscale("log")
    ax.set_ylabel("Maximum absolute ΔNLL difference")
    ax.set_title(
        "Numerical agreement between engines",
        loc="left",
        weight="bold",
    )
    ax.legend(frameon=False)

    for bar, raw in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.15,
            f"{raw:.2e}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save_figure(fig, output_path)


def make_plots(output_data: dict[str, Any], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    results = {
        item["engine"]: item
        for item in output_data["results"]
        if item.get("status") == "success"
    }

    if not set(ENGINE_ORDER).issubset(results):
        print("Skipping plots because all three engines did not complete successfully")
        return
    if output_data["agreement"].get("validation_status") == "not_run":
        print("Skipping agreement plots because validation was not run")
        return

    make_profile_plot(
        results,
        output_data["scan_values"],
        output_data["poi"],
        plot_dir / "delta_nll_profile",
    )
    make_residual_plot(
        output_data["agreement"],
        output_data["scan_values"],
        output_data["poi"],
        plot_dir / "delta_nll_absolute_differences",
    )
    make_steady_runtime_plot(
        results,
        plot_dir / "steady_state_runtime",
    )
    make_scan_runtime_plot(
        results,
        plot_dir / "full_scan_runtime",
    )
    make_phase_breakdown_plot(
        results,
        plot_dir / "timing_phase_breakdown",
    )
    make_agreement_plot(
        output_data["agreement"],
        output_data["delta_tolerance"],
        output_data["delta_relative_tolerance"],
        plot_dir / "numerical_agreement",
    )


def print_result(result: dict[str, Any]) -> None:
    print("\n" + "-" * 80)
    print(result.get("engine_label", result.get("engine")))
    print("-" * 80)
    print(f"status:                          {result.get('status')}")
    if result.get("status") != "success":
        print(
            "error:                           "
            f"{result.get('error_type')}: {result.get('error_message')}"
        )
        return

    print(f"first NLL:                       {result['first_nll']:.15f}")
    print(f"minimum POI:                     {result['minimum_poi']:.15f}")
    print(
        "workspace loading:                "
        f"{result['workspace_loading_time_seconds'] * 1e3:.3f} ms"
    )
    print(
        "model/node construction:          "
        f"{result['model_construction_time_seconds'] * 1e3:.3f} ms"
    )
    print(
        "NLL construction:                 "
        f"{result['nll_construction_time_seconds'] * 1e3:.3f} ms"
    )
    print(
        "cold first evaluation:             "
        f"{result['cold_first_evaluation_time_seconds'] * 1e6:.3f} µs"
    )
    print(
        "steady-state evaluation median:    "
        f"{result['steady_state_evaluation']['median_seconds'] * 1e6:.3f} µs"
    )
    print(
        "steady-state evaluation IQR:       "
        f"{result['steady_state_evaluation']['iqr_seconds'] * 1e6:.3f} µs"
    )
    print(
        "full scan median:                  "
        f"{result['full_scan']['median_seconds'] * 1e3:.3f} ms"
    )
    print(
        "full scan IQR:                     "
        f"{result['full_scan']['iqr_seconds'] * 1e3:.3f} ms"
    )
    print(
        "time per scan point:               "
        f"{result['time_per_scan_point_seconds'] * 1e6:.3f} µs"
    )
    print(f"current RSS delta:                 {result['current_rss_delta_mb']:.3f} MB")
    print(f"peak RSS delta:                    {result['peak_rss_delta_mb']:.3f} MB")
    if result.get("xroofit_runtime_verified"):
        print("xRooFit runtime verified:         True")
        print(f"xRooFit node type:                {result['xroofit_node_python_type']}")
        print(f"xRooFit NLL Python type:          {result['xroofit_nll_python_type']}")
        print(f"xRooFit NLL C++ class:            {result['xroofit_nll_cpp_class']}")


def print_agreement(agreement: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("Numerical agreement between engines")
    print("=" * 80)
    print(f"overall validation: {agreement.get('validation_status')}")

    for name, comparison in agreement.get("comparisons", {}).items():
        print(f"\n{name}")
        print(f"  validation:             {comparison['validation_status']}")
        print(f"  raw max abs diff:       {comparison['raw_nll_max_abs_diff']:.15e}")
        print(
            "  centered raw max diff:  "
            f"{comparison['centered_raw_nll_max_abs_diff']:.15e}"
        )
        print(f"  ΔNLL max abs diff:      {comparison['delta_nll_max_abs_diff']:.15e}")
        print(
            "  ΔNLL max relative diff: "
            f"{comparison['delta_nll_max_relative_diff']:.15e}"
        )
        print(f"  ΔNLL RMS diff:          {comparison['delta_nll_rms_diff']:.15e}")
        print(
            f"  constant offset:        {comparison['constant_offset_estimate']:.15e}"
        )
        print(f"  minimum POI diff:       {comparison['minimum_poi_abs_diff']:.15e}")


def run(
    *,
    json_path: Path,
    root_path: Path,
    analysis_name: str,
    target: str | None,
    pyhs3_data_name: str | None,
    pyhs3_combined: bool,
    pyhs3_channels: str | None,
    xroofit_model_name: str | None,
    xroofit_dataset_name: str,
    root_workspace_name: str,
    poi: str,
    parameter_point: str | None,
    observable_name: str,
    observable_index: int,
    pyhs3_noncompiled_mode: str,
    pyhs3_compiled_mode: str,
    pyhs3_nll_mode: str,
    signal_pdf: str | None,
    background_pdf: str | None,
    signal_yield_param: str | None,
    background_yield_param: str | None,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
    n_warmup_evaluations: int,
    n_evaluation_runs: int,
    n_scan_runs: int,
    poi_timing_value: float,
    output: Path,
    plot: bool,
    plot_dir: Path,
    delta_tolerance: float,
    delta_relative_tolerance: float,
    absolute_pyhs3_tolerance: float,
    minimum_tolerance: float,
    xroofit_library: str | None,
) -> dict[str, Any]:
    validate_scan_config(
        scan_min=scan_min,
        scan_max=scan_max,
        n_scan_points=n_scan_points,
        n_warmup_evaluations=n_warmup_evaluations,
        n_evaluation_runs=n_evaluation_runs,
        n_scan_runs=n_scan_runs,
        delta_tolerance=delta_tolerance,
        delta_relative_tolerance=delta_relative_tolerance,
        absolute_pyhs3_tolerance=absolute_pyhs3_tolerance,
        minimum_tolerance=minimum_tolerance,
    )
    validate_existing_file(json_path, "PyHS3 JSON workspace")
    validate_existing_file(root_path, "xRooFit ROOT workspace")
    validate_finite_float(poi_timing_value, "poi_timing_value")

    resolved_target = target or default_target_from_analysis(analysis_name)
    resolved_pyhs3_data = pyhs3_data_name or default_data_name_from_analysis(
        analysis_name
    )
    resolved_xroofit_model = xroofit_model_name or "pdfs/sim_pdf"
    resolved_pyhs3_combined = pyhs3_combined or (
        resolved_xroofit_model in {"pdfs/sim_pdf", "sim_pdf"}
        and xroofit_dataset_name == "combData"
    )
    resolved_pyhs3_channels = (
        [channel.strip() for channel in pyhs3_channels.split(",") if channel.strip()]
        if pyhs3_channels
        else infer_combined_channels(json_path)
        if resolved_pyhs3_combined
        else []
    )

    if resolved_pyhs3_combined and not resolved_pyhs3_channels:
        raise ValueError("Combined PyHS3 mode requires at least one channel")

    resolved_signal_pdf = signal_pdf or default_signal_pdf_from_analysis(analysis_name)
    resolved_background_pdf = background_pdf or default_background_pdf_from_analysis(
        analysis_name
    )
    resolved_signal_yield = signal_yield_param or default_signal_yield_from_analysis(
        analysis_name
    )
    resolved_background_yield = (
        background_yield_param or default_background_yield_from_analysis(analysis_name)
    )

    scan_values = [
        float(value) for value in np.linspace(scan_min, scan_max, n_scan_points)
    ]

    def build_pyhs3(mode: str) -> PyHS3Case | CombinedPyHS3Case:
        if resolved_pyhs3_combined:
            return build_combined_pyhs3_case(
                json_path=json_path,
                channels=resolved_pyhs3_channels,
                poi=poi,
                parameter_point=parameter_point,
                observable_name=observable_name,
                observable_index=observable_index,
                mode=mode,
                nll_mode=pyhs3_nll_mode,
            )
        return build_pyhs3_case(
            json_path=json_path,
            analysis_name=analysis_name,
            target=resolved_target,
            data_name=resolved_pyhs3_data,
            poi=poi,
            parameter_point=parameter_point,
            observable_name=observable_name,
            observable_index=observable_index,
            mode=mode,
            nll_mode=pyhs3_nll_mode,
            signal_pdf=resolved_signal_pdf,
            background_pdf=resolved_background_pdf,
            signal_yield_param=resolved_signal_yield,
            background_yield_param=resolved_background_yield,
        )

    specs = [
        EngineSpec(
            name="pyhs3_noncompiled",
            build_func=lambda: build_pyhs3(pyhs3_noncompiled_mode),
            eval_func=lambda case, value: pyhs3_nll(case, value),
            restore_func=restore_pyhs3_case,
            operational_definition=(
                "PyHS3 unoptimized/non-compiled baseline using "
                f"Workspace.model(mode={pyhs3_noncompiled_mode!r})."
            ),
        ),
        EngineSpec(
            name="pyhs3_compiled",
            build_func=lambda: build_pyhs3(pyhs3_compiled_mode),
            eval_func=lambda case, value: pyhs3_nll(case, value),
            restore_func=restore_pyhs3_case,
            operational_definition=(
                "PyHS3 compiled/optimized engine using "
                f"Workspace.model(mode={pyhs3_compiled_mode!r})."
            ),
        ),
        EngineSpec(
            name="xroofit",
            build_func=lambda: build_xroofit_case(
                root_path=root_path,
                json_path=json_path,
                workspace_name=root_workspace_name,
                model_name=resolved_xroofit_model,
                dataset_name=xroofit_dataset_name,
                poi=poi,
                parameter_point=parameter_point,
                xroofit_library=xroofit_library,
            ),
            eval_func=lambda case, value: xroofit_nll(case, value),
            restore_func=restore_xroofit_case,
            operational_definition=(
                "Verified xRooFit API path: "
                "ROOT.xRooNode(workspace)[model].nll(dataset).getVal(); "
                "direct RooFit createNLL is not used."
            ),
        ),
    ]

    results: list[dict[str, Any]] = []
    for spec in specs:
        try:
            result = measure_engine(
                spec=spec,
                scan_values=scan_values,
                n_warmup_evaluations=n_warmup_evaluations,
                n_evaluation_runs=n_evaluation_runs,
                n_scan_runs=n_scan_runs,
                poi_value=poi_timing_value,
                repeat_tolerance=absolute_pyhs3_tolerance,
            )
        except Exception as exc:
            result = failed_engine_result(spec.name, exc)
        results.append(result)

    successful = {
        result["engine"]: result
        for result in results
        if result.get("status") == "success"
    }
    agreement = build_all_agreements(
        successful,
        delta_tolerance=delta_tolerance,
        delta_relative_tolerance=delta_relative_tolerance,
        absolute_pyhs3_tolerance=absolute_pyhs3_tolerance,
        minimum_tolerance=minimum_tolerance,
    )
    status = (
        "success"
        if len(successful) == len(ENGINE_ORDER)
        and agreement["validation_status"] == "success"
        else "failed"
    )

    output_data = {
        "benchmark": BENCHMARK_NAME,
        "benchmark_mode": (
            "full_extended_nll_pyhs3_noncompiled_vs_compiled_vs_xroofit"
        ),
        "status": status,
        "json_path": str(json_path),
        "root_path": str(root_path),
        "analysis_name": analysis_name,
        "target": resolved_target,
        "pyhs3_data_name": resolved_pyhs3_data,
        "pyhs3_combined": resolved_pyhs3_combined,
        "pyhs3_channels": resolved_pyhs3_channels,
        "xroofit_model_name": resolved_xroofit_model,
        "xroofit_dataset_name": xroofit_dataset_name,
        "root_workspace_name": root_workspace_name,
        "poi": poi,
        "parameter_point": parameter_point,
        "observable_name": observable_name,
        "observable_index": observable_index,
        "pyhs3_noncompiled_mode": pyhs3_noncompiled_mode,
        "pyhs3_compiled_mode": pyhs3_compiled_mode,
        "pyhs3_nll_mode": pyhs3_nll_mode,
        "signal_pdf": resolved_signal_pdf,
        "background_pdf": resolved_background_pdf,
        "signal_yield_param": resolved_signal_yield,
        "background_yield_param": resolved_background_yield,
        "scan_min": scan_min,
        "scan_max": scan_max,
        "n_scan_points": n_scan_points,
        "n_warmup_evaluations": n_warmup_evaluations,
        "n_evaluation_runs": n_evaluation_runs,
        "n_scan_runs": n_scan_runs,
        "poi_timing_value": poi_timing_value,
        "delta_tolerance": delta_tolerance,
        "delta_relative_tolerance": delta_relative_tolerance,
        "absolute_pyhs3_tolerance": absolute_pyhs3_tolerance,
        "minimum_tolerance": minimum_tolerance,
        "scan_values": scan_values,
        "engines": list(ENGINE_ORDER),
        "frameworks": list(ENGINE_ORDER),
        "agreement": agreement,
        "results": results,
        "methodology": {
            "numerical_comparability": (
                "All engines use the same ordered POI grid and the same five "
                "channel datasets. The two PyHS3 modes evaluate the identical "
                "Python likelihood implementation and must agree in absolute "
                "NLL. The xRooFit path evaluates the corresponding ROOT "
                "workspace model and may differ by additive constants or "
                "parameter conventions; therefore cross-framework validation "
                "requires matching ΔNLL shapes and best-fit grid positions "
                "under combined absolute and relative tolerances."
            ),
            "engine_to_engine_comparability": (
                "Steady-state timing measures one full NLL evaluation after "
                "changing the POI between consecutive calls. Full-scan timing "
                "uses the same ordered POI grid and number of repetitions for "
                "all engines. These are comparable workflow-level engine "
                "operations. Setup phases are reported separately and are not "
                "claimed to represent identical internal work across engines."
            ),
            "xroofit_runtime_identity": (
                "The ROOT path is required at runtime to use "
                "ROOT.xRooNode(workspace)[model].nll(dataset).getVal(). "
                "The benchmark rejects a direct RooFit-only createNLL path and "
                "records the dynamic xRooFit wrapper/NLL types in the JSON."
            ),
            "comparability_limitations": (
                "The benchmark is not a bit-for-bit comparison of identical "
                "internal kernels. PyHS3 explicitly sums per-channel extended "
                "mixture terms, while xRooFit constructs an NLL over RooFit "
                "objects through the xRooFit API. Skipped transformed ROOT "
                "defaults and the observed constant raw-NLL offset are recorded. "
                "Consequently the strongest apples-to-apples claim applies to "
                "the POI scan workflow and ΔNLL result, not every internal "
                "operation or raw NLL convention."
            ),
            "parameter_policy": (
                "The POI is fixed while scanning. xRooFit POI value and "
                "constant state are snapshotted and restored. PyHS3 uses fresh "
                "evaluation dictionaries and verifies that stored defaults "
                "are unchanged."
            ),
            "timing_statistics": (
                "Steady-state evaluations and full scans report mean, median, "
                "standard deviation, quartiles and IQR. Plots use median with "
                "IQR error bars."
            ),
            "xroofit_activation": (
                "Run 'source external/xroofit/build/setup.sh' before the pixi "
                "benchmark command."
            ),
        },
    }

    print("=" * 80)
    print("PyHS3 compiled/non-compiled and xRooFit ΔNLL benchmark")
    print("=" * 80)
    print(f"PyHS3 JSON:                 {json_path}")
    print(f"ROOT workspace:             {root_path}")
    print(f"Analysis:                   {analysis_name}")
    print(f"PyHS3 target:               {resolved_target}")
    print(f"PyHS3 combined:             {resolved_pyhs3_combined}")
    if resolved_pyhs3_combined:
        print(f"PyHS3 channels:             {','.join(resolved_pyhs3_channels)}")
    print(f"PyHS3 NLL mode:             {pyhs3_nll_mode}")
    print(f"PyHS3 non-compiled mode:    {pyhs3_noncompiled_mode}")
    print(f"PyHS3 compiled mode:        {pyhs3_compiled_mode}")
    print(f"xRooFit model:              {resolved_xroofit_model}")
    print(f"xRooFit data:               {xroofit_dataset_name}")
    print(f"POI:                        {poi}")
    print(
        "Grid:                       "
        f"[{scan_min}, {scan_max}] with {n_scan_points} points"
    )
    print(f"Warm-up evaluations:        {n_warmup_evaluations}")
    print(f"Steady-state runs:          {n_evaluation_runs}")
    print(f"Full-scan runs:             {n_scan_runs}")
    print(f"Status:                     {status}")

    for result in results:
        print_result(result)
    print_agreement(agreement)

    save_json(output_data, output)
    print(f"\nSaved result to {output}")
    if plot:
        make_plots(output_data, plot_dir)
        print(f"Saved PNG plots to {plot_dir}")

    return output_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run PyHS3 compiled/non-compiled and real xRooFit full-NLL scans.")
    )
    parser.add_argument("--json-workspace", type=Path, required=True)
    parser.add_argument("--root-workspace", type=Path, required=True)
    parser.add_argument("--analysis", default="L_ch0")
    parser.add_argument("--target", default=None)
    parser.add_argument("--pyhs3-data-name", default=None)
    parser.add_argument(
        "--pyhs3-combined",
        action="store_true",
        help=(
            "Sum PyHS3 channel likelihoods to match xRooFit pdfs/sim_pdf on combData."
        ),
    )
    parser.add_argument(
        "--pyhs3-channels",
        default=None,
        help=(
            "Comma-separated channel names, e.g. ch0,ch1,ch2,ch3,ch4. "
            "Defaults to channels inferred from combData_chN datasets."
        ),
    )
    parser.add_argument(
        "--xroofit-model-name",
        default="pdfs/sim_pdf",
    )
    parser.add_argument(
        "--xroofit-dataset-name",
        default="combData",
    )
    parser.add_argument("--root-workspace-name", default="combWS")
    parser.add_argument("--poi", default="mu_sig")
    parser.add_argument("--parameter-point", default=None)
    parser.add_argument("--observable-name", default="x")
    parser.add_argument("--observable-index", type=int, default=0)
    parser.add_argument(
        "--pyhs3-noncompiled-mode",
        default="FAST_COMPILE",
        help=(
            "PyTensor mode used as the non-compiled/unoptimized PyHS3 "
            "baseline. Default: FAST_COMPILE."
        ),
    )
    parser.add_argument(
        "--pyhs3-compiled-mode",
        default="FAST_RUN",
        help=(
            "PyTensor mode used as the compiled/optimized PyHS3 engine. "
            "Default: FAST_RUN."
        ),
    )
    parser.add_argument(
        "--pyhs3-nll-mode",
        choices=["extended-mixture", "logpdf"],
        default="extended-mixture",
    )
    parser.add_argument("--signal-pdf", default=None)
    parser.add_argument("--background-pdf", default=None)
    parser.add_argument("--signal-yield-param", default=None)
    parser.add_argument("--background-yield-param", default=None)
    parser.add_argument("--scan-min", type=float, default=0.0)
    parser.add_argument("--scan-max", type=float, default=2.0)
    parser.add_argument("--n-scan-points", type=int, default=101)
    parser.add_argument("--n-warmup-evaluations", type=int, default=3)
    parser.add_argument("--n-evaluation-runs", type=int, default=20)
    parser.add_argument("--n-scan-runs", type=int, default=10)
    parser.add_argument("--poi-timing-value", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument(
        "--delta-tolerance",
        type=float,
        default=DEFAULT_DELTA_TOLERANCE,
    )
    parser.add_argument(
        "--delta-relative-tolerance",
        type=float,
        default=DEFAULT_DELTA_RELATIVE_TOLERANCE,
    )
    parser.add_argument(
        "--absolute-pyhs3-tolerance",
        type=float,
        default=DEFAULT_ABSOLUTE_PYHS3_TOLERANCE,
    )
    parser.add_argument(
        "--minimum-tolerance",
        type=float,
        default=DEFAULT_MINIMUM_TOLERANCE,
    )
    parser.add_argument("--xroofit-library", default="libxRooFit")
    return parser.parse_args()


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *argv]
        return parse_args()
    finally:
        sys.argv = original_argv


def main(argv: list[str] | None = None) -> None:
    args = parse_args() if argv is None else parse_args_from(argv)
    run(
        json_path=args.json_workspace,
        root_path=args.root_workspace,
        analysis_name=args.analysis,
        target=args.target,
        pyhs3_data_name=args.pyhs3_data_name,
        pyhs3_combined=args.pyhs3_combined,
        pyhs3_channels=args.pyhs3_channels,
        xroofit_model_name=args.xroofit_model_name,
        xroofit_dataset_name=args.xroofit_dataset_name,
        root_workspace_name=args.root_workspace_name,
        poi=args.poi,
        parameter_point=args.parameter_point,
        observable_name=args.observable_name,
        observable_index=args.observable_index,
        pyhs3_noncompiled_mode=args.pyhs3_noncompiled_mode,
        pyhs3_compiled_mode=args.pyhs3_compiled_mode,
        pyhs3_nll_mode=args.pyhs3_nll_mode,
        signal_pdf=args.signal_pdf,
        background_pdf=args.background_pdf,
        signal_yield_param=args.signal_yield_param,
        background_yield_param=args.background_yield_param,
        scan_min=args.scan_min,
        scan_max=args.scan_max,
        n_scan_points=args.n_scan_points,
        n_warmup_evaluations=args.n_warmup_evaluations,
        n_evaluation_runs=args.n_evaluation_runs,
        n_scan_runs=args.n_scan_runs,
        poi_timing_value=args.poi_timing_value,
        output=args.output,
        plot=args.plot,
        plot_dir=args.plot_dir,
        delta_tolerance=args.delta_tolerance,
        delta_relative_tolerance=args.delta_relative_tolerance,
        absolute_pyhs3_tolerance=args.absolute_pyhs3_tolerance,
        minimum_tolerance=args.minimum_tolerance,
        xroofit_library=args.xroofit_library or None,
    )


if __name__ == "__main__":
    main()
