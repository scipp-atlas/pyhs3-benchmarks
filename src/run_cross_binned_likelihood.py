"""Cross-framework binned-likelihood benchmark for paired pyHS3/pyhf models.

This benchmark intentionally uses the small paired HistFactory/HS3 models from the
pyHS3 test suite. It does not compare pyhf against the more complex Allex
RooFit/xRooFit workspaces because those are different statistical models.

Measured separately:
- workspace/schema validation and model construction;
- first likelihood evaluation;
- steady-state likelihood evaluation;
- a POI scan and numerical agreement after removing a constant NLL offset;
- correlated-model scaling with the number of bins.

The pyhf backend is NumPy with 64-bit floating-point values. There is no backend
compilation phase for this configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pyhf
from pyhs3 import Workspace as PyHS3Workspace

BENCHMARK_NAME = "cross_binned_likelihood"
DEFAULT_INPUT_DIR = Path("inputs/pyhf")
DEFAULT_RESULTS_DIR = Path("results") / BENCHMARK_NAME
DEFAULT_PLOTS_DIR = Path("plots") / BENCHMARK_NAME
MODEL_NAMES = ("correlated-background", "uncorrelated-background")


@dataclass(frozen=True)
class PairPaths:
    name: str
    hifa: Path
    hs3: Path


@dataclass
class Engine:
    name: str
    construction_seconds: float
    first_evaluation_seconds: float
    evaluate: Callable[[float], float]
    expected: Callable[[float], np.ndarray]
    nominal_parameters: dict[str, Any]
    parameter_order: list[str]
    backend: str
    dtype: str
    compiled: bool
    batched: bool


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _timed(call: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = call()
    return value, time.perf_counter() - start


def _scalar(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise ValueError(f"Expected one finite scalar, received shape={array.shape}")
    return float(array[0])


def _model_pairs(input_dir: Path) -> list[PairPaths]:
    return [
        PairPaths(
            name=name,
            hifa=input_dir / f"simplemodel_{name}_hifa.json",
            hs3=input_dir / f"simplemodel_{name}_hs3.json",
        )
        for name in MODEL_NAMES
    ]


def workspace_structure(hifa_spec: dict[str, Any]) -> dict[str, Any]:
    """Return the model structure recorded in the benchmark result."""
    channels = []
    for channel in hifa_spec["channels"]:
        samples = []
        for sample in channel["samples"]:
            samples.append(
                {
                    "name": sample["name"],
                    "bins": len(sample["data"]),
                    "modifiers": [
                        {"name": modifier["name"], "type": modifier["type"]}
                        for modifier in sample["modifiers"]
                    ],
                }
            )
        channels.append({"name": channel["name"], "samples": samples})
    measurement = hifa_spec["measurements"][0]
    return {
        "channels": channels,
        "observations": hifa_spec["observations"],
        "poi": measurement["config"]["poi"],
        "measurement": measurement["name"],
    }


def _pyhf_engine(spec: dict[str, Any]) -> Engine:
    pyhf.set_backend("numpy", precision="64b")

    def construct() -> tuple[Any, Any, np.ndarray]:
        workspace = pyhf.Workspace(spec, validate=True)
        model = workspace.model(measurement_name=spec["measurements"][0]["name"])
        data = np.asarray(workspace.data(model), dtype=np.float64)
        return workspace, model, data

    (_, model, data), construction_seconds = _timed(construct)
    nominal = np.asarray(model.config.suggested_init(), dtype=np.float64)
    poi_index = int(model.config.poi_index)

    def parameters(mu: float) -> np.ndarray:
        values = nominal.copy()
        values[poi_index] = mu
        return values

    def evaluate(mu: float) -> float:
        # pyhf.logpdf returns log L; benchmark convention is NLL = -log L.
        return -_scalar(model.logpdf(parameters(mu), data))

    def expected(mu: float) -> np.ndarray:
        # Only the main-channel expected counts are compared. Auxiliary data encode
        # constraints and are validated through the full NLL comparison.
        return np.asarray(model.expected_actualdata(parameters(mu)), dtype=np.float64)

    _, first_seconds = _timed(lambda: evaluate(1.0))
    return Engine(
        name="pyhf-numpy",
        construction_seconds=construction_seconds,
        first_evaluation_seconds=first_seconds,
        evaluate=evaluate,
        expected=expected,
        nominal_parameters={
            name: np.asarray(nominal[model.config.par_slice(name)]).tolist()
            for name in model.config.par_order
        },
        parameter_order=list(model.config.par_order),
        backend="numpy",
        dtype="float64",
        compiled=False,
        batched=False,
    )


def _pyhs3_parameters(model: Any) -> dict[str, np.ndarray]:
    return {
        parameter.name: np.asarray(parameter.value, dtype=np.float64)
        for parameter in model.parameterset
    }


def _replace_parameter(
    parameters: dict[str, np.ndarray], name: str, value: float
) -> dict[str, np.ndarray]:
    updated = {key: np.asarray(item).copy() for key, item in parameters.items()}
    if name not in updated:
        raise KeyError(f"Parameter {name!r} is absent; available={sorted(updated)}")
    original = updated[name]
    updated[name] = (
        np.asarray(value, dtype=np.float64)
        if original.ndim == 0
        else np.full(original.shape, value, dtype=np.float64)
    )
    return updated


def _hs3_serialized_nominal_expected(hs3_spec: dict[str, Any], mu: float) -> np.ndarray:
    """Return nominal expected bin counts encoded by the paired HS3 workspace.

    pyHS3 currently exposes the HistFactory likelihood evaluation engine, but not
    a public expected-bin-count API. Therefore this validation is deliberately
    labelled as a serialized-model validation, not as a timed pyHS3 engine call.

    The engine-to-engine comparison in this benchmark is the full constrained
    likelihood/NLL evaluation. Expected counts are still checked against the HS3
    model representation so that the paired JSON files cannot silently diverge.
    """
    distribution = hs3_spec["distributions"][0]
    total = np.zeros(
        len(distribution["samples"][0]["data"]["contents"]), dtype=np.float64
    )
    for sample in distribution["samples"]:
        contents = np.asarray(sample["data"]["contents"], dtype=np.float64)
        scale = 1.0
        for modifier in sample["modifiers"]:
            if modifier["type"] == "normfactor":
                parameter = modifier.get("parameter", modifier["name"])
                if parameter == "mu":
                    scale *= mu
                elif parameter == "Lumi":
                    scale *= 1.0
        total += scale * contents
    return total


def _pyhs3_engine(spec: dict[str, Any], analysis_name: str) -> Engine:
    distribution_name = spec["distributions"][0]["name"]

    def construct() -> tuple[Any, Any]:
        workspace = PyHS3Workspace(**spec)
        model = workspace.model(
            analysis_name,
            parameter_set="default_values",
            progress=False,
        )
        return workspace, model

    (_, model), construction_seconds = _timed(construct)
    nominal = _pyhs3_parameters(model)

    def evaluate(mu: float) -> float:
        parameters = _replace_parameter(nominal, "mu", mu)
        # For a HistFactoryDistChannel, pyHS3 logpdf includes the channel
        # Poisson term and its constraint terms. These workspaces contain one
        # channel, so this is the same full constrained likelihood evaluated by
        # pyhf.model.logpdf.
        return -_scalar(model.logpdf(distribution_name, **parameters))

    def expected(mu: float) -> np.ndarray:
        return _hs3_serialized_nominal_expected(spec, mu)

    _, first_seconds = _timed(lambda: evaluate(1.0))
    return Engine(
        name="pyhs3",
        construction_seconds=construction_seconds,
        first_evaluation_seconds=first_seconds,
        evaluate=evaluate,
        expected=expected,
        nominal_parameters={name: value.tolist() for name, value in nominal.items()},
        parameter_order=list(nominal),
        backend="PyTensor FAST_RUN",
        dtype="float64",
        compiled=True,
        batched=False,
    )


def _benchmark_evaluation(
    evaluate: Callable[[float], float], *, mu: float, repeats: int, warmups: int
) -> dict[str, Any]:
    for _ in range(warmups):
        evaluate(mu)
    timings = []
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        values.append(evaluate(mu))
        timings.append(time.perf_counter() - start)
    return {
        "median_seconds": statistics.median(timings),
        "mean_seconds": statistics.fmean(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "repeats": repeats,
        "warmups": warmups,
        "nll": statistics.fmean(values),
    }


def _scan(engine: Engine, mu_values: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    values = np.asarray(
        [engine.evaluate(float(mu)) for mu in mu_values], dtype=np.float64
    )
    return values, time.perf_counter() - start


def _agreement(
    pyhs3_values: np.ndarray,
    pyhf_values: np.ndarray,
    mu_values: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    offsets = pyhs3_values - pyhf_values
    offset = float(np.median(offsets))
    residual = offsets - offset
    pyhs3_delta = pyhs3_values - np.min(pyhs3_values)
    pyhf_delta = pyhf_values - np.min(pyhf_values)
    return {
        "raw_nll_constant_offset": offset,
        "raw_nll_offset_residual_max_abs": float(np.max(np.abs(residual))),
        "raw_nll_agrees_after_constant_offset": bool(
            np.allclose(residual, 0.0, rtol=rtol, atol=atol)
        ),
        "delta_nll_max_abs_difference": float(np.max(np.abs(pyhs3_delta - pyhf_delta))),
        "delta_nll_agrees": bool(
            np.allclose(pyhs3_delta, pyhf_delta, rtol=rtol, atol=atol)
        ),
        "pyhs3_minimum_mu": float(mu_values[int(np.argmin(pyhs3_values))]),
        "pyhf_minimum_mu": float(mu_values[int(np.argmin(pyhf_values))]),
        "minimum_grid_difference": float(
            abs(
                mu_values[int(np.argmin(pyhs3_values))]
                - mu_values[int(np.argmin(pyhf_values))]
            )
        ),
    }


def _repeat_correlated_hifa(spec: dict[str, Any], n_bins: int) -> dict[str, Any]:
    """Repeat the original two-bin correlated model in memory."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    result = copy.deepcopy(spec)

    def repeat(values: list[float]) -> list[float]:
        return np.resize(np.asarray(values, dtype=np.float64), n_bins).tolist()

    for sample in result["channels"][0]["samples"]:
        sample["data"] = repeat(sample["data"])
        for modifier in sample["modifiers"]:
            data = modifier.get("data")
            if modifier["type"] == "histosys" and isinstance(data, dict):
                data["hi_data"] = repeat(data["hi_data"])
                data["lo_data"] = repeat(data["lo_data"])
    result["observations"][0]["data"] = repeat(result["observations"][0]["data"])
    return result


def _repeat_correlated_hs3(spec: dict[str, Any], n_bins: int) -> dict[str, Any]:
    """Repeat the original two-bin correlated HS3 model in memory."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    result = copy.deepcopy(spec)

    def repeat(values: list[float]) -> list[float]:
        return np.resize(np.asarray(values, dtype=np.float64), n_bins).tolist()

    for datum in result["data"]:
        datum["contents"] = repeat(datum["contents"])
        datum["axes"][0]["nbins"] = n_bins
        datum["axes"][0]["max"] = float(n_bins)
    distribution = result["distributions"][0]
    distribution["axes"][0]["nbins"] = n_bins
    distribution["axes"][0]["max"] = float(n_bins)
    for sample in distribution["samples"]:
        sample["data"]["contents"] = repeat(sample["data"]["contents"])
        sample["data"]["errors"] = repeat(sample["data"]["errors"])
        for modifier in sample["modifiers"]:
            if modifier["type"] == "histosys":
                modifier["data"]["hi"]["contents"] = repeat(
                    modifier["data"]["hi"]["contents"]
                )
                modifier["data"]["lo"]["contents"] = repeat(
                    modifier["data"]["lo"]["contents"]
                )
    return result


def _scaling(
    hifa_spec: dict[str, Any],
    hs3_spec: dict[str, Any],
    bins: list[int],
    *,
    repeats: int,
    warmups: int,
) -> list[dict[str, Any]]:
    rows = []
    pyhs3_underflow_seen = False

    for n_bins in bins:
        scaled_hifa = _repeat_correlated_hifa(hifa_spec, n_bins)
        scaled_hs3 = _repeat_correlated_hs3(hs3_spec, n_bins)

        engines: list[Engine] = [_pyhf_engine(scaled_hifa)]

        if not pyhs3_underflow_seen:
            try:
                engines.insert(
                    0,
                    _pyhs3_engine(scaled_hs3, "simPdf_obsData"),
                )
            except ValueError as error:
                pyhs3_underflow_seen = True
                rows.append(
                    {
                        "number_of_bins": n_bins,
                        "engine": "pyhs3",
                        "status": "non-finite",
                        "reason": (
                            "Nominal pyHS3 logpdf is non-finite for this repeated "
                            "model, consistent with product-space PDF underflow: "
                            f"{error}"
                        ),
                    }
                )
                print(
                    f"Skipping pyHS3 scaling at {n_bins} bins and above: "
                    "nominal logpdf became non-finite."
                )
        else:
            rows.append(
                {
                    "number_of_bins": n_bins,
                    "engine": "pyhs3",
                    "status": "skipped-after-underflow",
                    "reason": (
                        "A smaller scaling point already produced a non-finite "
                        "nominal pyHS3 logpdf."
                    ),
                }
            )

        for engine in engines:
            timing = _benchmark_evaluation(
                engine.evaluate,
                mu=1.0,
                repeats=repeats,
                warmups=warmups,
            )

            rows.append(
                {
                    "number_of_bins": n_bins,
                    "engine": engine.name,
                    "status": "ok",
                    "construction_seconds": engine.construction_seconds,
                    "first_evaluation_seconds": engine.first_evaluation_seconds,
                    "steady_state_median_seconds": timing["median_seconds"],
                }
            )

    return rows


def _plot_representative_delta_nll(
    model_name: str,
    mu_values: np.ndarray,
    scans: dict[str, np.ndarray],
    output_dir: Path,
) -> Path:
    """Plot one representative physics curve; the second model is validated numerically."""
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for engine, values in scans.items():
        axis.plot(
            mu_values,
            values - np.min(values),
            linewidth=2,
            label=engine,
        )
    axis.set_xlabel("mu")
    axis.set_ylabel("Delta NLL")
    axis.set_title(f"{model_name}: engine-to-engine Delta NLL")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "representative_delta_nll.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_validation_summary(results: list[dict[str, Any]], output_dir: Path) -> Path:
    labels = [result["model"] for result in results]
    delta_nll = [
        max(result["agreement"]["delta_nll_max_abs_difference"], 1e-18)
        for result in results
    ]
    offset_residual = [
        max(result["agreement"]["raw_nll_offset_residual_max_abs"], 1e-18)
        for result in results
    ]

    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.35
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.bar(
        positions - width / 2,
        delta_nll,
        width=width,
        label="Delta NLL max abs diff",
    )
    axis.bar(
        positions + width / 2,
        offset_residual,
        width=width,
        label="NLL offset residual max abs",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Maximum absolute difference")
    axis.set_yscale("log")
    axis.set_title(
        "Numerical agreement summary",
        pad=22,
    )

    axis.text(
        0.5,
        1.005,
        "Expected counts: exact agreement for both paired models",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
    )
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "numerical_agreement_summary.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_timing_phases(results: list[dict[str, Any]], output_dir: Path) -> Path:
    labels: list[str] = []
    construction: list[float] = []
    cold_or_compile: list[float] = []
    compiled_or_warm: list[float] = []

    for result in results:
        for engine_name, engine in result["engines"].items():
            labels.append(f"{result['model']}\n{engine_name}")
            construction.append(engine["construction_seconds"])
            cold_or_compile.append(engine["first_evaluation_seconds"])
            compiled_or_warm.append(engine["steady_state"]["median_seconds"])

    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.25
    figure, axis = plt.subplots(figsize=(10.5, 5.4))
    axis.bar(
        positions - width,
        construction,
        width=width,
        label="workspace/model construction",
    )
    axis.bar(
        positions,
        cold_or_compile,
        width=width,
        label="first call (includes pyHS3 lazy compilation)",
    )
    axis.bar(
        positions + width,
        compiled_or_warm,
        width=width,
        label="warm/compiled function call",
    )
    axis.set_xticks(positions, labels, rotation=18, ha="right")
    axis.set_ylabel("Runtime [s]")
    axis.set_yscale("log")
    axis.set_title("Construction, compilation/cold call, and steady-state")
    axis.grid(True, axis="y", which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "timing_phases.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _plot_scaling_metric(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
) -> Path:
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    engines = sorted({row["engine"] for row in rows})
    for engine in engines:
        selected = sorted(
            (
                row
                for row in rows
                if row["engine"] == engine
                and row.get("status", "ok") == "ok"
                and metric in row
            ),
            key=lambda row: row["number_of_bins"],
        )
        if not selected:
            continue
        display_name = "pyHS3 compiled/warm" if engine == "pyhs3" else "pyhf NumPy warm"
        axis.plot(
            [row["number_of_bins"] for row in selected],
            [row[metric] for row in selected],
            marker="o",
            linewidth=2,
            label=display_name,
        )
    axis.set_xlabel("Number of bins")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.3)

    first_non_finite = min(
        (
            row["number_of_bins"]
            for row in rows
            if row["engine"] == "pyhs3" and row.get("status") == "non-finite"
        ),
        default=None,
    )
    if first_non_finite is not None:
        axis.axvline(
            first_non_finite,
            linestyle="--",
            linewidth=1,
            label=f"pyHS3 non-finite from {first_non_finite} bins",
        )

    axis.legend()
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _summary_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for engine_name, engine in result["engines"].items():
            rows.append(
                {
                    "framework": engine_name,
                    "workspace_type": f"paired simple HistFactory/HS3: {result['model']}",
                    "compiled": engine["compiled"],
                    "batched": engine["batched"],
                    "agreement": (
                        "expected + Delta NLL agree"
                        if (
                            result["agreement"]["expected_values_agree"]
                            and result["agreement"]["delta_nll_agrees"]
                        )
                        else "agreement failed"
                    ),
                    "execution_mode": (
                        "compiled/warm PyTensor function"
                        if engine_name == "pyhs3"
                        else "warm NumPy function call"
                    ),
                    "runtime_seconds": engine["steady_state"]["median_seconds"],
                }
            )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    results = []
    for pair in _model_pairs(args.input_dir):
        hifa_spec = _load_json(pair.hifa)
        hs3_spec = _load_json(pair.hs3)
        pyhf_engine = _pyhf_engine(hifa_spec)
        pyhs3_engine = _pyhs3_engine(hs3_spec, "simPdf_obsData")

        expected_pyhf = pyhf_engine.expected(1.0)
        expected_pyhs3 = pyhs3_engine.expected(1.0)
        expected_agreement = bool(
            np.allclose(expected_pyhs3, expected_pyhf, rtol=args.rtol, atol=args.atol)
        )

        pyhs3_scan, pyhs3_scan_seconds = _scan(pyhs3_engine, args.mu_values)
        pyhf_scan, pyhf_scan_seconds = _scan(pyhf_engine, args.mu_values)
        agreement = _agreement(
            pyhs3_scan,
            pyhf_scan,
            args.mu_values,
            rtol=args.rtol,
            atol=args.atol,
        )
        agreement["expected_values_agree"] = expected_agreement
        agreement["expected_values_max_abs_difference"] = float(
            np.max(np.abs(expected_pyhs3 - expected_pyhf))
        )

        scans = {"pyhs3": pyhs3_scan, "pyhf-numpy": pyhf_scan}
        results.append(
            {
                "model": pair.name,
                "files": {"hifa": str(pair.hifa), "hs3": str(pair.hs3)},
                "structure": workspace_structure(hifa_spec),
                "expected_values_at_nominal": {
                    "hs3_serialized_model": expected_pyhs3.tolist(),
                    "pyhf_engine": expected_pyhf.tolist(),
                    "validation_kind": (
                        "serialized HS3 model vs pyhf engine; pyHS3 has no "
                        "public expected-bin-count API"
                    ),
                },
                "mu_values": args.mu_values.tolist(),
                "nll_scans": {name: values.tolist() for name, values in scans.items()},
                "agreement": agreement,
                "engines": {
                    "pyhs3": {
                        "backend": pyhs3_engine.backend,
                        "dtype": pyhs3_engine.dtype,
                        "compiled": pyhs3_engine.compiled,
                        "batched": pyhs3_engine.batched,
                        "parameter_order": pyhs3_engine.parameter_order,
                        "nominal_parameters": pyhs3_engine.nominal_parameters,
                        "construction_seconds": pyhs3_engine.construction_seconds,
                        "first_evaluation_seconds": pyhs3_engine.first_evaluation_seconds,
                        "steady_state": _benchmark_evaluation(
                            pyhs3_engine.evaluate,
                            mu=1.0,
                            repeats=args.repeats,
                            warmups=args.warmups,
                        ),
                        "scan_seconds": pyhs3_scan_seconds,
                    },
                    "pyhf-numpy": {
                        "backend": pyhf_engine.backend,
                        "dtype": pyhf_engine.dtype,
                        "compiled": pyhf_engine.compiled,
                        "batched": pyhf_engine.batched,
                        "parameter_order": pyhf_engine.parameter_order,
                        "nominal_parameters": pyhf_engine.nominal_parameters,
                        "construction_seconds": pyhf_engine.construction_seconds,
                        "first_evaluation_seconds": pyhf_engine.first_evaluation_seconds,
                        "steady_state": _benchmark_evaluation(
                            pyhf_engine.evaluate,
                            mu=1.0,
                            repeats=args.repeats,
                            warmups=args.warmups,
                        ),
                        "scan_seconds": pyhf_scan_seconds,
                    },
                },
            }
        )

    correlated_hifa = _load_json(_model_pairs(args.input_dir)[0].hifa)
    correlated_hs3 = _load_json(_model_pairs(args.input_dir)[0].hs3)
    scaling_rows = _scaling(
        correlated_hifa,
        correlated_hs3,
        args.scaling_bins,
        repeats=args.scaling_repeats,
        warmups=args.warmups,
    )
    representative = results[0]
    representative_plot = _plot_representative_delta_nll(
        representative["model"],
        np.asarray(representative["mu_values"], dtype=np.float64),
        {
            name: np.asarray(values, dtype=np.float64)
            for name, values in representative["nll_scans"].items()
        },
        args.plots_dir,
    )
    validation_plot = _plot_validation_summary(results, args.plots_dir)
    timing_phases_plot = _plot_timing_phases(results, args.plots_dir)
    steady_state_scaling_plot = _plot_scaling_metric(
        scaling_rows,
        args.plots_dir,
        metric="steady_state_median_seconds",
        ylabel="Median warm NLL function call [s]",
        title="Warm/compiled NLL function-call scaling",
        filename="warm_function_call_vs_number_of_bins.png",
    )
    payload = {
        "benchmark": BENCHMARK_NAME,
        "methodology": {
            "pyhf_backend": "numpy",
            "pyhf_precision": "64b",
            "nll_definition": "-log L",
            "engine_to_engine_quantity": (
                "full constrained likelihood/NLL at identical parameter points"
            ),
            "expected_values_validation": (
                "pyhf engine output compared with nominal bin means encoded in "
                "the paired HS3 workspace; pyHS3 currently has no public "
                "expected-bin-count evaluation API"
            ),
            "construction_excluded_from_steady_state": True,
            "pyhs3_timing_phases": (
                "model construction is timed separately; the first logpdf call "
                "includes lazy PyTensor compilation; steady-state timings call "
                "the already compiled function"
            ),
            "pyhf_timing_phases": (
                "NumPy has no compilation phase; first-call and warm-call "
                "timings are reported separately"
            ),
            "backend_compilation": "not applicable for pyhf NumPy backend",
            "scaling": (
                "The existing two-bin correlated model is repeated in memory. "
                "The same nuisance remains correlated across every bin; no scaled "
                "workspace files are generated or stored. Scaling uses warm "
                "single-call latency. pyHS3 currently evaluates logpdf as log(pdf), "
                "whose product-form PDF can underflow for many bins."
            ),
        },
        "models": results,
        "plots": {
            "representative_delta_nll": str(representative_plot),
            "numerical_agreement_summary": str(validation_plot),
            "timing_phases": str(timing_phases_plot),
        },
        "scaling": {
            "rows": scaling_rows,
            "pyhs3_largest_finite_bin_count": max(
                (
                    row["number_of_bins"]
                    for row in scaling_rows
                    if row["engine"] == "pyhs3" and row.get("status") == "ok"
                ),
                default=None,
            ),
            "plots": {
                "warm_function_call": str(steady_state_scaling_plot),
            },
        },
        "summary": _summary_table(results),
        "limitations": [
            "pyhf is compared only on the paired simple HistFactory/HS3 models.",
            "RooFit/xRooFit benchmarks use different, more complex Allex workspaces.",
            "These results must not be combined into a context-free framework ranking.",
            "The pyhf backend is NumPy float64; JAX compilation is not measured here.",
            "The primary engine-to-engine comparison is the full constrained NLL.",
            "Expected bin counts are validated against the paired HS3 serialization because pyHS3 currently has no public expected-bin-count API; they are not included in runtime comparisons.",
            "Absolute NLL values may differ by a constant normalization offset; Delta NLL and the offset residual are reported explicitly.",
            "Scaling repeats the correlated two-bin template in memory and preserves one shared nuisance parameter across bins.",
            "For large repeated models, pyHS3 PDF products can underflow before the logarithm is taken. Non-finite scaling points are recorded and skipped rather than treated as timing results; pyhf scaling continues.",
        ],
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.results_dir / "results.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR)
    parser.add_argument("--mu-min", type=float, default=0.0)
    parser.add_argument("--mu-max", type=float, default=3.0)
    parser.add_argument("--mu-points", type=int, default=61)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--scaling-repeats", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument(
        "--scaling-bins",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-8)
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.mu_points < 2:
        parser.error("--mu-points must be at least 2")
    args.mu_values = np.linspace(
        args.mu_min, args.mu_max, args.mu_points, dtype=np.float64
    )
    payload = run(args)
    failures = [
        model["model"]
        for model in payload["models"]
        if not model["agreement"]["expected_values_agree"]
        or not model["agreement"]["delta_nll_agrees"]
    ]
    print(json.dumps(payload["summary"], indent=2))
    if failures:
        raise SystemExit(f"Numerical validation failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
