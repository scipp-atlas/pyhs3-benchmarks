from __future__ import annotations

import argparse
import gc
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import PLOTS_DIR, RESULTS_DIR
    from src.utils import get_current_rss_mb, get_peak_rss_mb, save_json
else:
    from .config import PLOTS_DIR, RESULTS_DIR
    from .utils import get_current_rss_mb, get_peak_rss_mb, save_json

from pyhs3.workspace import Workspace

try:
    import ROOT
except ImportError:  # pragma: no cover - environment dependent
    ROOT = None


BENCHMARK_NAME = "cross_nll_scan"
BENCHMARK_TITLE = "Cross-framework NLL scan benchmark"

DEFAULT_OUTPUT_DIR = RESULTS_DIR / BENCHMARK_NAME
DEFAULT_OUTPUT_NAME = f"{BENCHMARK_NAME}_result.json"
DEFAULT_PLOT_DIR = PLOTS_DIR / BENCHMARK_NAME

DEFAULT_WORKSPACES = [
    Path("inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json"),
    Path("inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json"),
    Path("inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json"),
]
DEFAULT_FRAMEWORKS = ["pyhs3", "roofit"]
SUPPORTED_FRAMEWORKS = ("pyhs3", "roofit")
REFERENCE_FRAMEWORK = "pyhs3"

PLOT_EPSILON = 1e-300
FRAMEWORK_STYLE = {
    "pyhs3": {
        "label": "PyHS3",
        "color": "#0B5EA8",
        "marker": "s",
        "linestyle": "-",
    },
    "roofit": {
        "label": "RooFit",
        "color": "#009E73",
        "marker": "D",
        "linestyle": "-.",
    },
}


class BenchmarkConfigurationError(ValueError):
    """Raised when the benchmark configuration is invalid."""


class ValidationFailure(RuntimeError):
    """Raised when a framework result is non-finite or numerically inconsistent."""


@dataclass(frozen=True)
class NLLScanConfig:
    framework: str
    workspace_path: Path
    root_workspace_path: Path | None
    analysis: str
    target: str
    pyhs3_data_name: str
    root_pdf_name: str
    root_data_name: str
    parameter_point: str | None
    observable_name: str
    observable_index: int
    poi: str
    mode: str
    mu_grid: list[float]
    shape_tolerance: float
    minimum_tolerance: float
    reference_delta_nll: list[float] | None = None
    reference_minimum_mu: float | None = None


def _framework_label(framework: str) -> str:
    return FRAMEWORK_STYLE.get(framework, {"label": framework})["label"]


def _style_for(framework: str) -> dict[str, Any]:
    return FRAMEWORK_STYLE.get(
        framework,
        {"label": framework, "color": "#333333", "marker": "o", "linestyle": "-"},
    )


def workspace_stem(workspace_path: Path) -> str:
    return workspace_path.name.removesuffix(".json").removesuffix(".root")


def workspace_label(workspace_path: Path) -> str:
    return workspace_stem(workspace_path).replace("_", "\n")


def workspace_title(workspace: str) -> str:
    return workspace.replace("_", " / ")


def channel_from_analysis(analysis_name: str) -> str:
    if not analysis_name.startswith("L_"):
        raise BenchmarkConfigurationError(
            "Cannot infer channel from analysis name. Use an analysis like L_ch0 "
            f"or pass explicit --target/--pyhs3-data-name. Got: {analysis_name}"
        )
    return analysis_name.replace("L_", "", 1)


def default_target_from_analysis(analysis_name: str) -> str:
    return f"model_{channel_from_analysis(analysis_name)}"


def default_data_name_from_analysis(analysis_name: str) -> str:
    return f"combData_{channel_from_analysis(analysis_name)}"


def default_root_workspace_path(workspace_path: Path) -> Path:
    return workspace_path.with_suffix(".root")


def build_mu_grid(mu_min: float, mu_max: float, n_points: int) -> list[float]:
    if not math.isfinite(mu_min) or not math.isfinite(mu_max):
        raise BenchmarkConfigurationError("--mu-min and --mu-max must be finite")
    if mu_min >= mu_max:
        raise BenchmarkConfigurationError("--mu-min must be smaller than --mu-max")
    if n_points < 2:
        raise BenchmarkConfigurationError("--n-points must be at least 2")
    return [float(value) for value in np.linspace(mu_min, mu_max, n_points)]


def validate_benchmark_config(
    *,
    frameworks: list[str],
    workspaces: list[Path],
    root_workspaces: list[Path] | None,
    analysis: str,
    target: str,
    pyhs3_data_name: str,
    root_pdf_name: str,
    root_data_name: str,
    observable_name: str,
    observable_index: int,
    poi: str,
    mode: str,
    mu_grid: list[float],
    shape_tolerance: float,
    minimum_tolerance: float,
) -> None:
    if not frameworks:
        raise BenchmarkConfigurationError("At least one framework must be selected")
    unknown = sorted(set(frameworks) - set(SUPPORTED_FRAMEWORKS))
    if unknown:
        raise BenchmarkConfigurationError(
            f"Unsupported framework(s): {unknown}. Supported: {list(SUPPORTED_FRAMEWORKS)}"
        )
    if not workspaces:
        raise BenchmarkConfigurationError("At least one workspace must be selected")
    if root_workspaces is not None and len(root_workspaces) != len(workspaces):
        raise BenchmarkConfigurationError(
            "--root-workspaces must have the same number of entries as --workspaces"
        )
    for workspace in workspaces:
        if not workspace.exists():
            raise FileNotFoundError(f"Workspace file does not exist: {workspace}")
        if not workspace.is_file():
            raise FileNotFoundError(f"Workspace path is not a file: {workspace}")
    if "roofit" in frameworks:
        if ROOT is None:
            raise BenchmarkConfigurationError(
                "RooFit was requested, but ROOT is not importable in this environment"
            )
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
    for name, value in {
        "analysis": analysis,
        "target": target,
        "pyhs3_data_name": pyhs3_data_name,
        "root_pdf_name": root_pdf_name,
        "root_data_name": root_data_name,
        "observable_name": observable_name,
        "poi": poi,
        "mode": mode,
    }.items():
        if not value:
            raise BenchmarkConfigurationError(f"{name} must be a non-empty string")
    if observable_index < 0:
        raise BenchmarkConfigurationError("--observable-index must be non-negative")
    if not mu_grid:
        raise BenchmarkConfigurationError("mu grid must not be empty")
    if shape_tolerance <= 0.0 or not math.isfinite(shape_tolerance):
        raise BenchmarkConfigurationError(
            "--shape-tolerance must be positive and finite"
        )
    if minimum_tolerance <= 0.0 or not math.isfinite(minimum_tolerance):
        raise BenchmarkConfigurationError(
            "--minimum-tolerance must be positive and finite"
        )


def extract_parameter_point(
    workspace: Workspace,
    parameter_point: str | None,
) -> dict[str, Any]:
    try:
        points = workspace.parameter_points.root
    except AttributeError as exc:
        raise BenchmarkConfigurationError(
            "Workspace does not contain parameter_points.root"
        ) from exc

    if not points:
        raise BenchmarkConfigurationError("Workspace does not contain parameter points")

    if parameter_point is None:
        selected = points[0]
    else:
        selected = next(
            (
                point
                for point in points
                if getattr(point, "name", None) == parameter_point
            ),
            None,
        )
        if selected is None:
            available = [getattr(point, "name", "<unnamed>") for point in points]
            raise BenchmarkConfigurationError(
                f"Could not find parameter point {parameter_point!r}. Available: {available}"
            )

    params: dict[str, Any] = {}
    for parameter in selected.parameters:
        try:
            params[parameter.name] = np.asarray(
                float(parameter.value), dtype=np.float64
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkConfigurationError(
                f"Parameter {parameter.name!r} cannot be converted to float: "
                f"{parameter.value!r}"
            ) from exc
    return params


def get_pyhs3_data_values(
    workspace: Workspace,
    data_name: str,
    observable_index: int,
) -> np.ndarray:
    try:
        data_entries = workspace.data.root
    except AttributeError as exc:
        raise BenchmarkConfigurationError(
            "Workspace does not contain data.root"
        ) from exc

    for data in data_entries:
        if data.name == data_name:
            values = np.asarray(
                [entry[observable_index] for entry in data.entries],
                dtype=np.float64,
            )
            if values.size == 0:
                raise BenchmarkConfigurationError(f"PyHS3 data {data_name!r} is empty")
            if not np.all(np.isfinite(values)):
                raise BenchmarkConfigurationError(
                    f"PyHS3 data {data_name!r} contains non-finite values"
                )
            return values

    available = [getattr(data, "name", "<unnamed>") for data in data_entries]
    raise BenchmarkConfigurationError(
        f"Could not find PyHS3 data {data_name!r}. Available data: {available}"
    )


def prepare_pyhs3_case(config: NLLScanConfig) -> tuple[Any, dict[str, Any]]:
    workspace = Workspace.load(config.workspace_path)
    model = workspace.model(config.analysis, progress=False, mode=config.mode)

    params = extract_parameter_point(workspace, config.parameter_point)

    try:
        for name, value in model.free_params.items():
            params[name] = np.asarray(value, dtype=np.float64)
    except AttributeError:
        pass

    params[config.observable_name] = get_pyhs3_data_values(
        workspace,
        config.pyhs3_data_name,
        config.observable_index,
    )

    if config.poi not in params:
        raise BenchmarkConfigurationError(
            f"POI {config.poi!r} is not present in PyHS3 parameters. "
            f"Available parameters: {sorted(params)}"
        )

    return model, params


def evaluate_pyhs3_nll(
    model: Any, params: dict[str, Any], target: str, poi: str, value: float
) -> float:
    eval_params = dict(params)
    eval_params[poi] = np.asarray(value, dtype=np.float64)
    logpdf = np.asarray(model.logpdf(target, **eval_params), dtype=np.float64)
    if logpdf.size == 0:
        raise ValidationFailure(f"PyHS3 returned an empty logpdf array for {target}")
    if not np.all(np.isfinite(logpdf)):
        raise ValidationFailure(f"PyHS3 returned non-finite logpdf values for {target}")
    nll = -float(np.sum(logpdf))
    if not math.isfinite(nll):
        raise ValidationFailure(f"PyHS3 NLL is non-finite: {nll}")
    return nll


def _root_collection_names(collection: Any) -> list[str]:
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


def _find_root_workspace(root_file: Any) -> Any:
    import ROOT

    for key in root_file.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom(ROOT.RooWorkspace.Class()):
            return obj
    raise KeyError("No RooWorkspace object found in ROOT file")


def _available_root_objects(root_workspace: Any) -> dict[str, list[str]]:
    names: dict[str, list[str]] = {"pdfs": [], "data": [], "vars": []}
    try:
        names["pdfs"] = _root_collection_names(root_workspace.allPdfs())
    except Exception:
        pass
    try:
        names["vars"] = _root_collection_names(root_workspace.allVars())
    except Exception:
        pass
    try:
        data_names: list[str] = []
        all_data = root_workspace.allData()
        for data in all_data:
            if data is not None:
                data_names.append(str(data.GetName()))
        names["data"] = data_names
    except Exception:
        pass
    return names


def _candidate_names(primary: str, fallbacks: Iterable[str]) -> list[str]:
    return list(dict.fromkeys([primary, *fallbacks]))


def _is_valid_root_object(obj: Any) -> bool:
    """Return False for missing/null PyROOT proxy objects.

    Newer PyROOT may return a typed null-pointer proxy instead of plain None
    for missing RooWorkspace objects.  Such objects compare as false in boolean
    context, but ``obj is not None`` is still true.
    """

    if obj is None:
        return False
    try:
        return bool(obj)
    except Exception:
        return True


def _make_single_observable_norm_set(
    root_workspace: Any, observable_name: str
) -> tuple[Any, Any]:
    """Return (observable, norm_set) for normalized RooFit PDF evaluation.

    For apples-to-apples comparison with ``PyHS3 model.logpdf(target, ...)`` on
    Alexx-generated unbinned workspaces, RooFit is evaluated point-by-point as a
    normalized PDF value over the same observable values from the HS3 JSON data.
    This deliberately avoids ``createNLL`` because RooFit may silently choose an
    extended-likelihood convention that does not match the PyHS3 target logpdf.
    """

    import ROOT

    observable = root_workspace.var(str(observable_name))
    if not _is_valid_root_object(observable):
        available = ", ".join(_available_root_objects(root_workspace)["vars"])
        raise KeyError(
            f"RooFit observable {observable_name!r} not found. "
            f"Available variables: {available}"
        )

    norm_set = ROOT.RooArgSet(observable)
    if not _is_valid_root_object(norm_set) or norm_set.getSize() != 1:
        raise KeyError(
            f"Could not build RooFit normalization set for {observable_name!r}"
        )

    return observable, norm_set


def _get_root_pdf(
    root_workspace: Any, requested: str, target: str, analysis: str
) -> Any:
    candidates = _candidate_names(
        requested,
        [target, analysis, "sim_pdf", "model_ch0", "likelihood"],
    )
    for name in candidates:
        pdf = root_workspace.pdf(name)
        if _is_valid_root_object(pdf):
            return pdf
    available = ", ".join(_available_root_objects(root_workspace)["pdfs"])
    raise KeyError(
        f"Could not find RooFit PDF. Tried {candidates}. Available PDFs: {available}"
    )


def _get_root_data(root_workspace: Any, requested: str, pyhs3_data_name: str) -> Any:
    channel = pyhs3_data_name.replace("combData_", "", 1)
    candidates = _candidate_names(
        requested,
        [pyhs3_data_name, "combData", f"combData_{channel}", "data"],
    )
    for name in candidates:
        data = root_workspace.data(name)
        if _is_valid_root_object(data):
            return data
    available = ", ".join(_available_root_objects(root_workspace)["data"])
    raise KeyError(
        f"Could not find RooFit data. Tried {candidates}. Available data: {available}"
    )


def _set_root_defaults_from_pyhs3(
    root_workspace: Any,
    workspace_path: Path,
    parameter_point: str | None,
) -> None:
    try:
        workspace = Workspace.load(workspace_path)
        parameters = extract_parameter_point(workspace, parameter_point)
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


def prepare_roofit_case(
    config: NLLScanConfig,
) -> tuple[Any, Any, Any, Any, np.ndarray, Any]:
    if ROOT is None:
        raise BenchmarkConfigurationError("ROOT is not available")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

    if config.root_workspace_path is None:
        raise BenchmarkConfigurationError(
            "RooFit requires a matching ROOT workspace path"
        )

    root_file = ROOT.TFile.Open(str(config.root_workspace_path), "READ")
    if not root_file or root_file.IsZombie():
        raise FileNotFoundError(
            f"Could not open ROOT file: {config.root_workspace_path}"
        )

    try:
        root_workspace = _find_root_workspace(root_file)
        _set_root_defaults_from_pyhs3(
            root_workspace=root_workspace,
            workspace_path=config.workspace_path,
            parameter_point=config.parameter_point,
        )

        pdf = _get_root_pdf(
            root_workspace,
            requested=config.root_pdf_name,
            target=config.target,
            analysis=config.analysis,
        )
        poi_var = root_workspace.var(config.poi)
        if not _is_valid_root_object(poi_var):
            available = ", ".join(_available_root_objects(root_workspace)["vars"])
            raise KeyError(
                f"RooFit POI {config.poi!r} not found. Available variables: {available}"
            )

        observable, norm_set = _make_single_observable_norm_set(
            root_workspace,
            config.observable_name,
        )

        # Use the exact same unbinned observable values as PyHS3.  This is more
        # robust than relying on RooWorkspace.data(...), whose object names differ
        # between exported ROOT files and whose createNLL convention may include
        # extended terms that are not part of the PyHS3 target logpdf.
        pyhs3_workspace = Workspace.load(config.workspace_path)
        data_values = get_pyhs3_data_values(
            pyhs3_workspace,
            config.pyhs3_data_name,
            config.observable_index,
        )

    except Exception:
        root_file.Close()
        raise

    keepalive = (root_file, root_workspace, pdf, poi_var, observable, norm_set)
    return keepalive, pdf, poi_var, observable, data_values, root_file


def evaluate_roofit_nll(
    pdf: Any,
    poi_var: Any,
    observable: Any,
    data_values: np.ndarray,
    value: float,
) -> float:
    poi_var.setVal(float(value))

    total_logpdf = 0.0
    norm_set = None
    try:
        import ROOT

        norm_set = ROOT.RooArgSet(observable)
    except Exception as error:
        raise ValidationFailure(
            f"Could not create RooFit normalization set: {error}"
        ) from error

    for index, x_value in enumerate(
        np.asarray(data_values, dtype=np.float64).reshape(-1)
    ):
        observable.setVal(float(x_value))
        pdf_value = float(pdf.getVal(norm_set))
        if pdf_value <= 0.0 or not math.isfinite(pdf_value):
            raise ValidationFailure(
                f"RooFit returned invalid normalized PDF value at event {index}: {pdf_value}"
            )
        total_logpdf += math.log(pdf_value)

    nll_value = -float(total_logpdf)
    if not math.isfinite(nll_value):
        raise ValidationFailure(f"RooFit NLL is non-finite: {nll_value}")
    return nll_value


def delta_nll_shape(nll_values: list[float]) -> list[float]:
    if not nll_values:
        raise ValidationFailure("Cannot compute delta NLL from empty values")
    if not all(math.isfinite(value) for value in nll_values):
        raise ValidationFailure("NLL values contain non-finite entries")
    minimum = min(nll_values)
    return [float(value - minimum) for value in nll_values]


def minimum_position(mu_grid: list[float], nll_values: list[float]) -> float:
    if len(mu_grid) != len(nll_values):
        raise ValidationFailure("mu grid and NLL values must have the same length")
    return float(mu_grid[int(np.argmin(np.asarray(nll_values, dtype=float)))])


def max_abs_difference(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValidationFailure("Cannot compare arrays with different lengths")
    return float(max(abs(a - b) for a, b in zip(left, right, strict=True)))


def mean_offset(reference: list[float], observed: list[float]) -> float:
    if len(reference) != len(observed):
        raise ValidationFailure("Cannot compare arrays with different lengths")
    return float(
        mean(value - ref for ref, value in zip(reference, observed, strict=True))
    )


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(mean(values)),
    }


def run_single_framework_scan(config: NLLScanConfig) -> dict[str, Any]:
    gc.collect()
    current_rss_before_mb = get_current_rss_mb()
    peak_rss_before_mb = get_peak_rss_mb()

    cleanup = None
    try:
        build_start = time.perf_counter()
        if config.framework == "pyhs3":
            model, params = prepare_pyhs3_case(config)

            def evaluate(mu: float) -> float:
                return evaluate_pyhs3_nll(model, params, config.target, config.poi, mu)

        elif config.framework == "roofit":
            keepalive, pdf, poi_var, observable, data_values, root_file = (
                prepare_roofit_case(config)
            )
            cleanup = root_file.Close

            def evaluate(mu: float) -> float:
                return evaluate_roofit_nll(pdf, poi_var, observable, data_values, mu)

        else:
            raise BenchmarkConfigurationError(f"Unknown framework: {config.framework}")

        model_build_time_seconds = time.perf_counter() - build_start

        cold_start = time.perf_counter()
        cold_first_nll = evaluate(config.mu_grid[0])
        cold_first_evaluation_time_seconds = time.perf_counter() - cold_start

        warmup_mu = config.mu_grid[len(config.mu_grid) // 2]
        warmup_start = time.perf_counter()
        _ = evaluate(warmup_mu)
        warmup_time_seconds = time.perf_counter() - warmup_start

        first_start = time.perf_counter()
        first_nll = evaluate(config.mu_grid[0])
        first_evaluation_time_seconds = time.perf_counter() - first_start

        scan_start = time.perf_counter()
        nll_values = [float(evaluate(mu)) for mu in config.mu_grid]
        full_scan_time_seconds = time.perf_counter() - scan_start

    finally:
        if cleanup is not None:
            cleanup()

    current_rss_after_mb = get_current_rss_mb()
    peak_rss_after_mb = get_peak_rss_mb()
    gc.collect()

    delta_shape = delta_nll_shape(nll_values)
    min_mu = minimum_position(config.mu_grid, nll_values)

    result: dict[str, Any] = {
        "benchmark": BENCHMARK_NAME,
        "framework": config.framework,
        "framework_label": _framework_label(config.framework),
        "workspace": config.workspace_path.name,
        "workspace_path": str(config.workspace_path),
        "root_workspace_path": str(config.root_workspace_path)
        if config.root_workspace_path
        else None,
        "workspace_label": workspace_label(config.workspace_path),
        "analysis": config.analysis,
        "target": config.target,
        "pyhs3_data_name": config.pyhs3_data_name,
        "root_pdf_name": config.root_pdf_name,
        "root_data_name": config.root_data_name,
        "observable_name": config.observable_name,
        "observable_index": config.observable_index,
        "poi": config.poi,
        "mode": config.mode,
        "n_points": len(config.mu_grid),
        "mu_min": float(config.mu_grid[0]),
        "mu_max": float(config.mu_grid[-1]),
        "cold_first_nll": float(cold_first_nll),
        "first_nll": float(first_nll),
        "nll_values": [float(value) for value in nll_values],
        "delta_nll_shape": delta_shape,
        "minimum_mu": float(min_mu),
        "model_build_time_seconds": float(model_build_time_seconds),
        "cold_first_evaluation_time_seconds": float(cold_first_evaluation_time_seconds),
        "warmup_time_seconds": float(warmup_time_seconds),
        "first_evaluation_time_seconds": float(first_evaluation_time_seconds),
        "full_scan_time_seconds": float(full_scan_time_seconds),
        "time_per_scan_point_seconds": float(
            full_scan_time_seconds / len(config.mu_grid)
        ),
        "current_rss_before_mb": float(current_rss_before_mb),
        "current_rss_after_mb": float(current_rss_after_mb),
        "current_rss_delta_mb": float(
            max(0.0, current_rss_after_mb - current_rss_before_mb)
        ),
        "peak_rss_before_mb": float(peak_rss_before_mb),
        "peak_rss_after_mb": float(peak_rss_after_mb),
        "peak_rss_delta_mb": float(max(0.0, peak_rss_after_mb - peak_rss_before_mb)),
        "rss_delta_mb": float(max(0.0, current_rss_after_mb - current_rss_before_mb)),
        "nll_summary": summarize(nll_values),
        "delta_nll_summary": summarize(delta_shape),
        "status": "success",
    }

    if (
        config.reference_delta_nll is not None
        and config.reference_minimum_mu is not None
    ):
        diff = max_abs_difference(config.reference_delta_nll, delta_shape)
        minimum_diff = abs(min_mu - config.reference_minimum_mu)
        result.update(
            {
                "reference_framework": REFERENCE_FRAMEWORK,
                "constant_offset_estimate": mean_offset(
                    config.reference_delta_nll, delta_shape
                ),
                "delta_nll_shape_max_abs_diff": float(diff),
                "minimum_mu_abs_diff": float(minimum_diff),
                "delta_nll_shape_success": bool(diff <= config.shape_tolerance),
                "minimum_mu_success": bool(minimum_diff <= config.minimum_tolerance),
                "validation_status": "success"
                if diff <= config.shape_tolerance
                and minimum_diff <= config.minimum_tolerance
                else "failed",
            }
        )
    else:
        result.update(
            {
                "reference_framework": REFERENCE_FRAMEWORK,
                "constant_offset_estimate": 0.0,
                "delta_nll_shape_max_abs_diff": 0.0,
                "minimum_mu_abs_diff": 0.0,
                "delta_nll_shape_success": True,
                "minimum_mu_success": True,
                "validation_status": "success",
            }
        )

    if result["validation_status"] != "success":
        result["error_type"] = "ValidationFailure"
        result["error_message"] = (
            "Delta-NLL agreement failed "
            f"(shape diff={result['delta_nll_shape_max_abs_diff']:.3e}, "
            f"minimum diff={result['minimum_mu_abs_diff']:.3e})"
        )

    return result


def error_result(config: NLLScanConfig, error: BaseException) -> dict[str, Any]:
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
        "analysis": config.analysis,
        "target": config.target,
        "pyhs3_data_name": config.pyhs3_data_name,
        "root_pdf_name": config.root_pdf_name,
        "root_data_name": config.root_data_name,
        "observable_name": config.observable_name,
        "observable_index": config.observable_index,
        "poi": config.poi,
        "mode": config.mode,
        "n_points": len(config.mu_grid),
        "status": "error",
        "validation_status": "not_run",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }


def print_result(result: dict[str, Any]) -> None:
    print()
    print("-" * 72)
    print(
        f"{result.get('workspace')} / "
        f"{result.get('framework_label', result.get('framework'))}"
    )
    print("-" * 72)
    print(f"status:                  {result.get('status')}")
    print(f"validation:              {result.get('validation_status', 'unknown')}")
    if result.get("status") != "success":
        print(
            f"error:                   {result.get('error_type')}: {result.get('error_message')}"
        )
        return
    print(f"minimum mu:              {result['minimum_mu']:.15f}")
    print(f"minimum mu abs diff:     {result['minimum_mu_abs_diff']:.15e}")
    print(f"shape max abs diff:      {result['delta_nll_shape_max_abs_diff']:.15e}")
    print(f"constant offset:         {result['constant_offset_estimate']:.15e}")
    print(
        f"model build:             {result['model_build_time_seconds'] * 1000.0:.3f} ms"
    )
    print(
        f"cold first evaluation:   {result['cold_first_evaluation_time_seconds'] * 1000.0:.3f} ms"
    )
    print(
        f"warm first evaluation:   {result['first_evaluation_time_seconds'] * 1e6:.3f} us"
    )
    print(
        f"full scan:               {result['full_scan_time_seconds'] * 1000.0:.3f} ms"
    )
    print(
        f"time per scan point:     {result['time_per_scan_point_seconds'] * 1e6:.3f} us"
    )
    print(f"current RSS delta:       {result['rss_delta_mb']:.3f} MB")
    print(f"peak RSS delta:          {result['peak_rss_delta_mb']:.3f} MB")


def successful_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if result.get("status") == "success"]


def summarize_status(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = successful_results(results)
    validated = [
        result for result in successful if result.get("validation_status") == "success"
    ]
    failed = [
        result
        for result in results
        if result.get("status") != "success"
        or result.get("validation_status") != "success"
    ]
    return {
        "status": "success" if not failed and results else "completed_with_errors",
        "n_results": len(results),
        "n_successful": len(successful),
        "n_validated": len(validated),
        "n_failed": len(failed),
        "failed_results": [
            {
                "workspace": result.get("workspace"),
                "framework": result.get("framework"),
                "status": result.get("status"),
                "validation_status": result.get("validation_status"),
                "error_type": result.get("error_type"),
                "error_message": result.get("error_message"),
            }
            for result in failed
        ],
    }


def _success_dataframe(results: list[dict[str, Any]]):
    import pandas as pd

    rows = []
    for result in successful_results(results):
        rows.append(
            {
                "workspace": result["workspace"],
                "workspace_key": workspace_stem(Path(result["workspace"])),
                "workspace_label": result["workspace_label"],
                "framework": result["framework_label"],
                "framework_key": result["framework"],
                "minimum_mu": result["minimum_mu"],
                "minimum_mu_abs_diff": result["minimum_mu_abs_diff"],
                "shape_diff": result["delta_nll_shape_max_abs_diff"],
                "build_ms": result["model_build_time_seconds"] * 1000.0,
                "cold_ms": result["cold_first_evaluation_time_seconds"] * 1000.0,
                "first_us": result["first_evaluation_time_seconds"] * 1e6,
                "scan_ms": result["full_scan_time_seconds"] * 1000.0,
                "us_per_point": result["time_per_scan_point_seconds"] * 1e6,
                "current_rss_mb": max(result["rss_delta_mb"], 0.0),
                "peak_rss_mb": max(result["peak_rss_delta_mb"], 0.0),
                "validation_status": result["validation_status"],
            }
        )
    return pd.DataFrame(rows)


def make_profile_plot(
    results: list[dict[str, Any]], mu_grid: list[float], output_path: Path
) -> None:
    df_results = successful_results(results)
    if not df_results:
        return

    workspaces = list(dict.fromkeys(result["workspace"] for result in df_results))
    fig, axes = plt.subplots(
        len(workspaces),
        1,
        figsize=(12.8, 4.8 * len(workspaces)),
        squeeze=False,
    )
    x = np.asarray(mu_grid, dtype=float)

    for ax, workspace_name in zip(axes.flat, workspaces, strict=False):
        subset = [
            result for result in df_results if result["workspace"] == workspace_name
        ]
        reference = next(
            (result for result in subset if result["framework"] == REFERENCE_FRAMEWORK),
            subset[0],
        )
        reference_delta = np.asarray(reference["delta_nll_shape"], dtype=float)

        for result in subset:
            framework = result["framework"]
            style = _style_for(framework)
            values = np.asarray(result["delta_nll_shape"], dtype=float)
            ax.plot(
                x,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=max(1, len(x) // 12),
                linewidth=2.4,
                markersize=6,
                label=style["label"],
            )
            if framework != REFERENCE_FRAMEWORK:
                max_residual = float(np.max(np.abs(values - reference_delta)))
                ax.text(
                    0.02,
                    0.90 - 0.07 * len(ax.texts),
                    f"{style['label']} max residual: {max_residual:.2e}",
                    transform=ax.transAxes,
                    fontsize=10,
                    color=style["color"],
                    weight="bold",
                )
        ax.set_title(
            workspace_title(workspace_stem(Path(workspace_name))),
            loc="left",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Signal strength μ", fontsize=12)
        ax.set_ylabel("ΔNLL", fontsize=12)
        ax.grid(True, which="major", alpha=0.35)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.3)

    fig.suptitle(
        "Cross-framework ΔNLL scan agreement",
        x=0.02,
        ha="left",
        fontsize=25,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_timing_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15.2, max(5.2, 1.0 + 0.45 * len(df["workspace_key"].unique()))),
        squeeze=False,
    )
    metrics = [("build_ms", "Model build [ms]"), ("scan_ms", "Full scan [ms]")]

    for ax, (column, title) in zip(axes.flat, metrics, strict=True):
        labels = []
        values = []
        colors = []
        for _, row in df.iterrows():
            labels.append(f"{row['workspace_key'].split('_')[0]}\n{row['framework']}")
            values.append(max(float(row[column]), 1e-9))
            colors.append(_style_for(row["framework_key"])["color"])
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0)
        ax.grid(True, which="both", axis="y", alpha=0.32)
        for bar, raw in zip(bars, values, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.12,
                f"{raw:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
                weight="bold",
            )

    fig.suptitle(
        "NLL scan runtime profile", x=0.02, ha="left", fontsize=25, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_relative_runtime_plot(
    results: list[dict[str, Any]], output_path: Path
) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    import matplotlib.pyplot as plt

    workspaces = list(dict.fromkeys(df["workspace_key"]))
    fig, axes = plt.subplots(
        len(workspaces),
        1,
        figsize=(11.8, 3.8 * len(workspaces)),
        squeeze=False,
    )

    for ax, workspace in zip(axes.flat, workspaces, strict=False):
        subset = df[df["workspace_key"] == workspace].sort_values("us_per_point")
        fastest = max(float(subset["us_per_point"].iloc[0]), 1e-300)
        labels = list(subset["framework"])
        values = [float(value) / fastest for value in subset["us_per_point"]]
        colors = [
            _style_for(framework)["color"] for framework in subset["framework_key"]
        ]
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("Relative time per scan point (fastest = 1×)")
        ax.set_title(
            workspace_title(workspace), loc="left", fontsize=13, fontweight="bold"
        )
        ax.grid(True, axis="x", alpha=0.35)
        for bar, rel, us in zip(bars, values, subset["us_per_point"], strict=True):
            ax.text(
                bar.get_width() * 1.02,
                bar.get_y() + bar.get_height() / 2,
                f"{rel:.2f}× ({float(us):.2f} µs/point)",
                va="center",
                fontsize=10,
                weight="bold",
            )
        ax.set_xlim(0.0, max(values) * 1.35)

    fig.suptitle(
        "NLL scan throughput ranking", x=0.02, ha="left", fontsize=25, fontweight="bold"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_memory_plot(results: list[dict[str, Any]], output_path: Path) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(max(12.8, 0.55 * len(df)), 6.2))
    labels = [
        f"{row['workspace_key'].split('_')[0]}\n{row['framework']}"
        for _, row in df.iterrows()
    ]
    x = np.arange(len(labels))
    width = 0.36
    current = [max(float(value), 1e-3) for value in df["current_rss_mb"]]
    peak = [max(float(value), 1e-3) for value in df["peak_rss_mb"]]

    ax.bar(
        x - width / 2, current, width=width, label="Current RSS Δ", edgecolor="black"
    )
    ax.bar(
        x + width / 2,
        peak,
        width=width,
        label="Peak RSS Δ",
        edgecolor="black",
        hatch="//",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Memory delta [MB]")
    ax.set_title(
        "Memory footprint during NLL scan", loc="left", fontsize=18, fontweight="bold"
    )
    ax.grid(True, which="both", axis="y", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_agreement_plot(
    results: list[dict[str, Any]], output_path: Path, tolerance: float
) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    selected = df[df["framework_key"] != REFERENCE_FRAMEWORK]
    if selected.empty:
        return

    fig, ax = plt.subplots(figsize=(max(10.8, 0.7 * len(selected)), 5.8))
    labels = [
        f"{row['workspace_key'].split('_')[0]}\n{row['framework']}"
        for _, row in selected.iterrows()
    ]
    raw = [float(value) for value in selected["shape_diff"]]
    floor = max(tolerance * 1e-6, 1e-18)
    values = [max(value, floor) for value in raw]
    colors = [_style_for(framework)["color"] for framework in selected["framework_key"]]
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.7)
    ax.axhline(
        tolerance,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=f"tolerance = {tolerance:g}",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("max |ΔNLL - ΔNLL(PyHS3)|")
    ax.set_title(
        "Numerical agreement with PyHS3", loc="left", fontsize=18, fontweight="bold"
    )
    ax.grid(True, which="both", axis="y", alpha=0.35)
    ax.legend(frameon=False)
    for bar, value in zip(bars, raw, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.18,
            f"{value:.1e}",
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_summary_table(results: list[dict[str, Any]], output_path: Path) -> None:
    df = _success_dataframe(results)
    if df.empty:
        return

    rows = []
    previous_workspace = None
    for result in successful_results(results):
        workspace_name = workspace_stem(Path(result["workspace"]))
        rows.append(
            [
                workspace_name if workspace_name != previous_workspace else "",
                result["framework_label"],
                result["n_points"],
                f"{result['model_build_time_seconds'] * 1000.0:.3g}",
                f"{result['full_scan_time_seconds'] * 1000.0:.3g}",
                f"{result['time_per_scan_point_seconds'] * 1e6:.3g}",
                f"{result['minimum_mu']:.3g}",
                f"{result['delta_nll_shape_max_abs_diff']:.1e}",
                f"{result['rss_delta_mb']:.2f}",
                result["validation_status"],
            ]
        )
        previous_workspace = workspace_name

    columns = [
        "Workspace",
        "Framework",
        "Points",
        "Build\n[ms]",
        "Scan\n[ms]",
        "µs/point",
        "min μ",
        "max ΔNLL diff",
        "RSS Δ\n[MB]",
        "Validation",
    ]

    fig_height = max(4.8, 0.28 * len(rows) + 1.7)
    fig, ax = plt.subplots(figsize=(16.0, fig_height))
    ax.axis("off")
    fig.text(
        0.012,
        0.985,
        "Cross-framework NLL scan summary",
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.012,
        0.925,
        "ΔNLL scans across matching PyHS3 and ROOT workspaces; PyHS3 is the numerical reference.",
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
    table.set_fontsize(7.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#c8c8c8")
        cell.set_linewidth(0.42)
        if row == 0:
            cell.set_facecolor("#2b2b2b")
            cell.set_text_props(color="white", weight="bold")
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
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_plots(
    results: list[dict[str, Any]],
    mu_grid: list[float],
    plot_dir: Path,
    shape_tolerance: float,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    if len(successful_results(results)) < 2:
        print("Skipping plots: at least two successful results are needed.")
        return
    make_profile_plot(results, mu_grid, plot_dir / "cross_nll_scan_profile.png")
    make_timing_plot(results, plot_dir / "cross_nll_timing_profile.png")
    make_relative_runtime_plot(results, plot_dir / "cross_nll_relative_runtime.png")
    make_memory_plot(results, plot_dir / "cross_nll_memory_profile.png")
    make_agreement_plot(
        results, plot_dir / "cross_nll_numerical_agreement.png", shape_tolerance
    )
    make_summary_table(results, plot_dir / "cross_nll_summary_table.png")


def run(
    *,
    frameworks: list[str],
    workspaces: list[Path],
    root_workspaces: list[Path] | None,
    analysis: str,
    target: str | None,
    pyhs3_data_name: str | None,
    root_pdf_name: str | None,
    root_data_name: str | None,
    parameter_point: str | None,
    observable_name: str,
    observable_index: int,
    poi: str,
    mode: str,
    mu_min: float,
    mu_max: float,
    n_points: int,
    shape_tolerance: float,
    minimum_tolerance: float,
    output_dir: Path,
    output_name: str,
    plot: bool,
    plot_dir: Path,
    fail_fast: bool = False,
) -> dict[str, Any]:
    resolved_target = target or default_target_from_analysis(analysis)
    resolved_pyhs3_data_name = pyhs3_data_name or default_data_name_from_analysis(
        analysis
    )
    resolved_root_pdf_name = root_pdf_name or resolved_target
    resolved_root_data_name = root_data_name or resolved_pyhs3_data_name
    mu_grid = build_mu_grid(mu_min, mu_max, n_points)

    validate_benchmark_config(
        frameworks=frameworks,
        workspaces=workspaces,
        root_workspaces=root_workspaces,
        analysis=analysis,
        target=resolved_target,
        pyhs3_data_name=resolved_pyhs3_data_name,
        root_pdf_name=resolved_root_pdf_name,
        root_data_name=resolved_root_data_name,
        observable_name=observable_name,
        observable_index=observable_index,
        poi=poi,
        mode=mode,
        mu_grid=mu_grid,
        shape_tolerance=shape_tolerance,
        minimum_tolerance=minimum_tolerance,
    )

    resolved_root_workspaces = root_workspaces or [
        default_root_workspace_path(path) for path in workspaces
    ]

    results: list[dict[str, Any]] = []
    reference_delta_by_workspace: dict[str, list[float]] = {}
    reference_min_by_workspace: dict[str, float] = {}

    for workspace, root_workspace in zip(
        workspaces, resolved_root_workspaces, strict=True
    ):
        print(f"Computing PyHS3 reference ΔNLL scan for {workspace.name}", flush=True)
        reference_config = NLLScanConfig(
            framework=REFERENCE_FRAMEWORK,
            workspace_path=workspace,
            root_workspace_path=None,
            analysis=analysis,
            target=resolved_target,
            pyhs3_data_name=resolved_pyhs3_data_name,
            root_pdf_name=resolved_root_pdf_name,
            root_data_name=resolved_root_data_name,
            parameter_point=parameter_point,
            observable_name=observable_name,
            observable_index=observable_index,
            poi=poi,
            mode=mode,
            mu_grid=mu_grid,
            shape_tolerance=shape_tolerance,
            minimum_tolerance=minimum_tolerance,
        )
        reference_result = run_single_framework_scan(reference_config)
        reference_delta_by_workspace[str(workspace)] = list(
            reference_result["delta_nll_shape"]
        )
        reference_min_by_workspace[str(workspace)] = float(
            reference_result["minimum_mu"]
        )

    for workspace, root_workspace in zip(
        workspaces, resolved_root_workspaces, strict=True
    ):
        for framework in frameworks:
            config = NLLScanConfig(
                framework=framework,
                workspace_path=workspace,
                root_workspace_path=root_workspace if framework == "roofit" else None,
                analysis=analysis,
                target=resolved_target,
                pyhs3_data_name=resolved_pyhs3_data_name,
                root_pdf_name=resolved_root_pdf_name,
                root_data_name=resolved_root_data_name,
                parameter_point=parameter_point,
                observable_name=observable_name,
                observable_index=observable_index,
                poi=poi,
                mode=mode,
                mu_grid=mu_grid,
                shape_tolerance=shape_tolerance,
                minimum_tolerance=minimum_tolerance,
                reference_delta_nll=reference_delta_by_workspace[str(workspace)],
                reference_minimum_mu=reference_min_by_workspace[str(workspace)],
            )
            print(
                f"Running workspace={workspace.name}, framework={framework}", flush=True
            )
            try:
                result = run_single_framework_scan(config)
            except Exception as error:  # noqa: BLE001
                result = error_result(config, error)
            results.append(result)
            print_result(result)
            if fail_fast and (
                result.get("status") != "success"
                or result.get("validation_status") != "success"
            ):
                break

    summary = summarize_status(results)
    output_data = {
        "benchmark": BENCHMARK_NAME,
        "benchmark_mode": "alexx_workspace_pyhs3_vs_roofit",
        "summary": summary,
        "configuration": {
            "frameworks": frameworks,
            "workspaces": [str(path) for path in workspaces],
            "root_workspaces": [str(path) for path in resolved_root_workspaces],
            "analysis": analysis,
            "target": resolved_target,
            "pyhs3_data_name": resolved_pyhs3_data_name,
            "root_pdf_name": resolved_root_pdf_name,
            "root_data_name": resolved_root_data_name,
            "parameter_point": parameter_point,
            "observable_name": observable_name,
            "observable_index": observable_index,
            "poi": poi,
            "mode": mode,
            "mu_min": mu_min,
            "mu_max": mu_max,
            "n_points": n_points,
            "shape_tolerance": shape_tolerance,
            "minimum_tolerance": minimum_tolerance,
            "reference_framework": REFERENCE_FRAMEWORK,
        },
        "mu_grid": mu_grid,
        "results": results,
    }

    output_path = output_dir / output_name
    save_json(output_data, output_path)

    print()
    print("=" * 80)
    print(BENCHMARK_TITLE)
    print("=" * 80)
    print(f"Status:      {summary['status']}")
    print(f"Validated:   {summary['n_validated']} / {summary['n_results']}")
    if summary["failed_results"]:
        print("Failed:")
        for failure in summary["failed_results"]:
            print(
                "  - "
                f"{failure.get('workspace')} / {failure.get('framework')}: "
                f"{failure.get('error_type') or failure.get('status')} "
                f"{failure.get('error_message') or ''}"
            )
    print(f"Saved result to {output_path}")

    if plot:
        make_plots(results, mu_grid, plot_dir, shape_tolerance=shape_tolerance)
        print(f"Saved plots to {plot_dir}")

    return output_data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=BENCHMARK_TITLE)
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=DEFAULT_FRAMEWORKS,
        choices=SUPPORTED_FRAMEWORKS,
    )
    parser.add_argument(
        "--workspaces", nargs="+", type=Path, default=DEFAULT_WORKSPACES
    )
    parser.add_argument("--root-workspaces", nargs="+", type=Path, default=None)
    parser.add_argument("--analysis", default="L_ch0")
    parser.add_argument("--target", default=None)
    parser.add_argument("--pyhs3-data-name", default=None)
    parser.add_argument("--root-pdf-name", default=None)
    parser.add_argument("--root-data-name", default=None)
    parser.add_argument("--parameter-point", default=None)
    parser.add_argument("--observable-name", default="x")
    parser.add_argument("--observable-index", type=int, default=0)
    parser.add_argument("--poi", default="mu_sig")
    parser.add_argument("--mode", default="FAST_RUN")
    parser.add_argument("--mu-min", type=float, default=0.0)
    parser.add_argument("--mu-max", type=float, default=2.0)
    parser.add_argument("--n-points", type=int, default=101)
    parser.add_argument("--shape-tolerance", type=float, default=1e-7)
    parser.add_argument("--minimum-tolerance", type=float, default=1e-12)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(
            frameworks=list(args.frameworks),
            workspaces=list(args.workspaces),
            root_workspaces=list(args.root_workspaces)
            if args.root_workspaces
            else None,
            analysis=args.analysis,
            target=args.target,
            pyhs3_data_name=args.pyhs3_data_name,
            root_pdf_name=args.root_pdf_name,
            root_data_name=args.root_data_name,
            parameter_point=args.parameter_point,
            observable_name=args.observable_name,
            observable_index=args.observable_index,
            poi=args.poi,
            mode=args.mode,
            mu_min=args.mu_min,
            mu_max=args.mu_max,
            n_points=args.n_points,
            shape_tolerance=args.shape_tolerance,
            minimum_tolerance=args.minimum_tolerance,
            output_dir=args.output_dir,
            output_name=args.output_name,
            plot=args.plot,
            plot_dir=args.plot_dir,
            fail_fast=args.fail_fast,
        )
    except Exception as error:
        raise RuntimeError(
            "Cross-framework NLL scan benchmark did not complete"
        ) from error


if __name__ == "__main__":
    main(sys.argv[1:])
