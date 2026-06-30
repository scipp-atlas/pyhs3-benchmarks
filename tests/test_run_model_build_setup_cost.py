from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from src import run_model_build_setup_cost as benchmark


class FakeParameter:
    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value


class FakeParameterPoint:
    def __init__(self, parameters: list[FakeParameter]) -> None:
        self.parameters = parameters


class FakeWorkspace:
    def __init__(
        self, parameters: dict[str, Any] | None = None, model: Any | None = None
    ) -> None:
        if parameters is None:
            parameters = valid_parameters()
        self.parameter_points = SimpleNamespace(
            root=[
                FakeParameterPoint(
                    [FakeParameter(name, value) for name, value in parameters.items()]
                )
            ]
        )
        self.model_obj = model if model is not None else FakePyhs3Model()
        self.model_calls: list[tuple[str, bool, str]] = []

    def model(self, target: str, progress: bool, mode: str) -> Any:
        self.model_calls.append((target, progress, mode))
        return self.model_obj


class FakePyhs3Model:
    def __init__(self, values: list[float] | None = None) -> None:
        self.values = values if values is not None else [0.25, 0.125]
        self.calls: list[tuple[str, Any]] = []

    def pdf(self, name: str, mu: Any) -> float:
        self.calls.append((name, mu))
        index = int(name.removeprefix("poisson_"))
        return self.values[index]


class FakePyhfConfig:
    def __init__(self, par_order: list[str] | None = None) -> None:
        self.par_order = par_order if par_order is not None else ["mu"]

    def suggested_init(self) -> list[float]:
        return [1.0 for _ in self.par_order]


class FakePyhfModel:
    def __init__(self, value: float = -3.0, par_order: list[str] | None = None) -> None:
        self.value = value
        self.config = FakePyhfConfig(par_order)
        self.pars_seen: list[list[float]] = []

    def logpdf(self, pars: list[float], data: Any) -> float:
        self.pars_seen.append(list(pars))
        return self.value


class FakePyhfWorkspace:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.model_obj = FakePyhfModel()

    def model(self) -> FakePyhfModel:
        return self.model_obj

    def data(self, model: FakePyhfModel) -> list[float]:
        return [1.0, 2.0]


class FakeRooRealVar:
    def __init__(self, name: str, title: str, value: float, *bounds: float) -> None:
        self.name = name
        self.value = value
        self.constant = False
        self.bounds = bounds

    def setConstant(self, value: bool) -> None:
        self.constant = value

    def setVal(self, value: float) -> None:
        self.value = value


class FakeRooArgList(list):
    def __init__(self, *args: Any) -> None:
        super().__init__(args)

    def add(self, value: Any) -> None:
        self.append(value)


class FakeRooFormulaVar:
    def __init__(self, *args: Any) -> None:
        self.args = args


class FakeRooPoisson:
    def __init__(self, *args: Any) -> None:
        self.args = args
        self.value = 0.5

    def getVal(self) -> float:
        return self.value


class FakeRooProdPdf:
    def __init__(self, *args: Any) -> None:
        self.args = args


class FakeRootModule:
    class RooFit:
        ERROR = object()

    class RooMsgService:
        @staticmethod
        def instance() -> "FakeRootModule.RooMsgService":
            return FakeRootModule.RooMsgService()

        def setGlobalKillBelow(self, level: object) -> None:
            self.level = level

    RooRealVar = FakeRooRealVar
    RooArgList = FakeRooArgList
    RooFormulaVar = FakeRooFormulaVar
    RooPoisson = FakeRooPoisson
    RooProdPdf = FakeRooProdPdf


def valid_parameters() -> dict[str, float]:
    return {
        "mu": 1.0,
        "signal_0": 2.0,
        "background_0": 10.0,
        "obs_0": 12.0,
        "signal_1": 3.0,
        "background_1": 11.0,
        "obs_1": 14.0,
    }


@pytest.fixture
def workspace_file(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


@pytest.fixture
def success_result() -> dict[str, Any]:
    return {
        "framework": "pyhf",
        "plot_label": "pyhf",
        "status": "success",
        "value": 3.0,
        "input_load_time_seconds": 0.001,
        "model_construction_time_seconds": 0.002,
        "cold_first_evaluation_time_seconds": 0.003,
        "warmup_iterations": 2,
        "warmup_time_seconds": 0.004,
        "warm_first_evaluation_time_seconds": 0.005,
        "rss_before_mb": 10.0,
        "rss_after_mb": 11.0,
        "rss_delta_mb": 1.0,
        "stage_notes": "notes",
        "value_abs_diff_from_pyhf": 0.0,
        "validation_status": "success",
        "validation_tolerance": 1e-9,
    }


@pytest.fixture
def all_success_results(success_result: dict[str, Any]) -> list[dict[str, Any]]:
    pyhf = dict(success_result, framework="pyhf", plot_label="pyhf", value=3.0)
    pyhs3 = dict(success_result, framework="pyhs3", plot_label="PyHS3", value=3.0)
    roofit = dict(success_result, framework="roofit", plot_label="RooFit", value=3.0)
    return [pyhs3, pyhf, roofit]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def test_validate_existing_file_success(workspace_file: Path) -> None:
    assert (
        benchmark.validate_existing_file(workspace_file, "Workspace file")
        == workspace_file
    )


@pytest.mark.parametrize("kind", ["missing", "dir"])
def test_validate_existing_file_rejects_invalid(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "item"
    if kind == "dir":
        path.mkdir()
    with pytest.raises(FileNotFoundError, match="Workspace file"):
        benchmark.validate_existing_file(path, "Workspace file")


@pytest.mark.parametrize(("value", "minimum"), [(0, 0), (1, 1), (3, 2)])
def test_validate_positive_int_success(value: int, minimum: int) -> None:
    benchmark.validate_positive_int(value, "value", minimum=minimum)


def test_validate_positive_int_rejects_too_small() -> None:
    with pytest.raises(ValueError, match="value must be at least 2"):
        benchmark.validate_positive_int(1, "value", minimum=2)


@pytest.mark.parametrize("value", [0.0, -1.0, 2.5])
def test_validate_finite_float_success(value: float) -> None:
    benchmark.validate_finite_float(value, "value")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_validate_finite_float_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        benchmark.validate_finite_float(value, "value")


def test_validate_frameworks_normalizes_deduplicates_and_sorts() -> None:
    assert benchmark.validate_frameworks(["ROOFIT", "pyhf", "pyhs3", "pyhf"]) == [
        "pyhs3",
        "pyhf",
        "roofit",
    ]


@pytest.mark.parametrize(
    ("frameworks", "message"),
    [
        ([], "At least one"),
        (["bad", "pyhf"], "Unknown framework"),
        (["pyhs3"], "pyhf must be included"),
    ],
)
def test_validate_frameworks_rejects_invalid(
    frameworks: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_frameworks(frameworks)


def test_validate_benchmark_config_success(workspace_file: Path) -> None:
    assert benchmark.validate_benchmark_config(
        workspace_path=workspace_file,
        n_bins=2,
        mu=1.0,
        frameworks=["pyhf", "pyhs3"],
        warmup_iterations=0,
    ) == ["pyhs3", "pyhf"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_bins": 0}, "n_bins must be at least 1"),
        ({"mu": float("nan")}, "mu must be finite"),
        ({"warmup_iterations": -1}, "warmup_iterations must be at least 0"),
        ({"frameworks": ["pyhs3"]}, "pyhf must be included"),
    ],
)
def test_validate_benchmark_config_rejects_invalid(
    workspace_file: Path, override: dict[str, Any], message: str
) -> None:
    kwargs = {
        "workspace_path": workspace_file,
        "n_bins": 2,
        "mu": 1.0,
        "frameworks": ["pyhf"],
        "warmup_iterations": 0,
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        benchmark.validate_benchmark_config(**kwargs)


@pytest.mark.parametrize("value", [0.0, 0.1])
def test_validate_non_negative_seconds_success(value: float) -> None:
    benchmark.validate_non_negative_seconds(value, "duration")


def test_validate_non_negative_seconds_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        benchmark.validate_non_negative_seconds(-0.1, "duration")


@pytest.mark.parametrize("value", [0.1, 1.0])
def test_validate_positive_seconds_success(value: float) -> None:
    benchmark.validate_positive_seconds(value, "duration")


@pytest.mark.parametrize("value", [0.0, -0.1])
def test_validate_positive_seconds_rejects_non_positive(value: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        benchmark.validate_positive_seconds(value, "duration")


def test_validate_measurement_result_ignores_failed_result() -> None:
    benchmark.validate_measurement_result({"status": "failed"})


def test_validate_measurement_result_success(success_result: dict[str, Any]) -> None:
    benchmark.validate_measurement_result(success_result)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("input_load_time_seconds", -0.1, "non-negative"),
        ("model_construction_time_seconds", -0.1, "non-negative"),
        ("warmup_time_seconds", -0.1, "non-negative"),
        ("rss_delta_mb", -0.1, "non-negative"),
        ("cold_first_evaluation_time_seconds", 0.0, "positive"),
        ("warm_first_evaluation_time_seconds", 0.0, "positive"),
        ("value", float("nan"), "value must be finite"),
    ],
)
def test_validate_measurement_result_rejects_invalid(
    success_result: dict[str, Any], field: str, value: float, message: str
) -> None:
    result = dict(success_result)
    result[field] = value
    with pytest.raises(ValueError, match=message):
        benchmark.validate_measurement_result(result)


# ---------------------------------------------------------------------------
# Workspace/statistical helpers
# ---------------------------------------------------------------------------


def test_extract_parameters_success() -> None:
    workspace = FakeWorkspace(
        {"mu": np.asarray([1.0]), "obs_0": 2, "signal_0": 3, "background_0": 4}
    )
    assert benchmark.extract_parameters(workspace) == {
        "mu": 1.0,
        "obs_0": 2.0,
        "signal_0": 3.0,
        "background_0": 4.0,
    }


@pytest.mark.parametrize(
    ("workspace", "message"),
    [
        (SimpleNamespace(), "valid parameter_points"),
        (
            SimpleNamespace(parameter_points=SimpleNamespace(root=[])),
            "any parameter points",
        ),
        (
            SimpleNamespace(
                parameter_points=SimpleNamespace(root=[SimpleNamespace(parameters=[])])
            ),
            "does not contain parameters",
        ),
        (FakeWorkspace({"bad": object()}), "not scalar-like"),
        (FakeWorkspace({"bad": float("nan")}), "parameter bad"),
    ],
)
def test_extract_parameters_rejects_invalid(workspace: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.extract_parameters(workspace)


@pytest.mark.parametrize(
    ("parameters", "expected"),
    [({"obs_0": 1.0}, 1), ({"obs_0": 1.0, "obs_1": 2.0, "other": 3.0}, 2)],
)
def test_infer_n_bins_success(parameters: dict[str, float], expected: int) -> None:
    assert benchmark.infer_n_bins(parameters) == expected


@pytest.mark.parametrize(
    ("parameters", "message"),
    [({"signal_0": 1.0}, "no obs"), ({"obs_0": 1.0, "obs_2": 2.0}, "Non-contiguous")],
)
def test_infer_n_bins_rejects_invalid(
    parameters: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.infer_n_bins(parameters)


def test_get_vectors_success() -> None:
    assert benchmark.get_vectors(valid_parameters(), 2) == (
        [2.0, 3.0],
        [10.0, 11.0],
        [12.0, 14.0],
    )


@pytest.mark.parametrize(
    ("parameters", "n_bins", "exc_type", "message"),
    [
        (valid_parameters(), 0, ValueError, "n_bins"),
        ({"signal_0": 1.0, "background_0": 2.0}, 1, KeyError, "Missing required"),
        ({**valid_parameters(), "signal_0": float("nan")}, 2, ValueError, "signal_0"),
        ({**valid_parameters(), "background_0": -1.0}, 2, ValueError, "non-negative"),
    ],
)
def test_get_vectors_rejects_invalid(
    parameters: dict[str, float], n_bins: int, exc_type: type[Exception], message: str
) -> None:
    with pytest.raises(exc_type, match=message):
        benchmark.get_vectors(parameters, n_bins)


def test_poisson_nll_from_vectors_success() -> None:
    value = benchmark.poisson_nll_from_vectors([2.0], [10.0], [12.0], 1.0)
    assert math.isfinite(value)


@pytest.mark.parametrize(
    ("signal", "background", "observed", "mu", "message"),
    [
        ([1.0], [-2.0], [1.0], 1.0, "Expected yield"),
        ([1.0], [1.0], [1.0], float("nan"), "mu must be finite"),
    ],
)
def test_poisson_nll_from_vectors_rejects_invalid(
    signal: list[float],
    background: list[float],
    observed: list[float],
    mu: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.poisson_nll_from_vectors(signal, background, observed, mu)


def test_pyhs3_first_eval_success() -> None:
    assert benchmark.pyhs3_first_eval(
        FakePyhs3Model([0.5, 0.25]), 2, 1.0
    ) == pytest.approx(-math.log(0.5) - math.log(0.25))


@pytest.mark.parametrize("values", [[0.5, 0.0], [0.5, float("nan")]])
def test_pyhs3_first_eval_rejects_invalid_pdf(values: list[float]) -> None:
    with pytest.raises(ValueError):
        benchmark.pyhs3_first_eval(FakePyhs3Model(values), 2, 1.0)


def test_build_pyhf_spec() -> None:
    spec = benchmark.build_pyhf_spec(valid_parameters(), 2)
    assert spec["version"] == "1.0.0"
    assert spec["measurements"][0]["config"]["poi"] == "mu"
    assert spec["channels"][0]["samples"][0]["data"] == [2.0, 3.0]


def test_pyhf_first_eval_success() -> None:
    model = FakePyhfModel(value=-4.0)
    assert benchmark.pyhf_first_eval(model, data=[1.0], mu_value=2.5) == 4.0
    assert model.pars_seen[-1] == [2.5]


def test_pyhf_first_eval_rejects_missing_mu_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="mu"):
        benchmark.pyhf_first_eval(FakePyhfModel(par_order=["theta"]), [], 1.0)
    with pytest.raises(ValueError, match="pyhf NLL"):
        benchmark.pyhf_first_eval(FakePyhfModel(value=float("nan")), [], 1.0)


def test_roofit_first_eval_success() -> None:
    mu = FakeRooRealVar("mu", "mu", 1.0)
    model = {"mu": mu, "poissons": [FakeRooPoisson(), FakeRooPoisson()]}
    assert benchmark.roofit_first_eval(model, 2.0) == pytest.approx(-2 * math.log(0.5))
    assert mu.value == 2.0


@pytest.mark.parametrize("value", [0.0, float("nan")])
def test_roofit_first_eval_rejects_invalid_pdf(value: float) -> None:
    poisson = FakeRooPoisson()
    poisson.value = value
    with pytest.raises(ValueError):
        benchmark.roofit_first_eval(
            {"mu": FakeRooRealVar("mu", "mu", 1.0), "poissons": [poisson]}, 1.0
        )


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def test_timed_call_success(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([1.0, 1.25])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    assert benchmark.timed_call(lambda: "ok") == ("ok", pytest.approx(0.25))


def test_timed_call_rejects_negative_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([2.0, 1.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    with pytest.raises(ValueError, match="non-negative"):
        benchmark.timed_call(lambda: None)


def test_run_warmups_success(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([1.0, 1.2])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    value, duration = benchmark.run_warmups(lambda: 3.0, warmup_iterations=2)
    assert value == 3.0
    assert duration == pytest.approx(0.2)


def test_run_warmups_zero_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([1.0, 1.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    value, duration = benchmark.run_warmups(lambda: 3.0, warmup_iterations=0)
    assert math.isnan(value)
    assert duration == 0.0


def test_run_warmups_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="warmup NLL"):
        benchmark.run_warmups(lambda: float("nan"), warmup_iterations=1)


def test_successful_result_and_failed_framework_result(
    success_result: dict[str, Any],
) -> None:
    result = benchmark.successful_result(
        framework="pyhf",
        value=3.0,
        input_load_time_seconds=0.001,
        model_construction_time_seconds=0.002,
        cold_first_evaluation_time_seconds=0.003,
        warmup_time_seconds=0.004,
        warm_first_evaluation_time_seconds=0.005,
        rss_before_mb=10.0,
        rss_after_mb=12.0,
        stage_notes="notes",
        warmup_iterations=2,
    )
    assert result["plot_label"] == "pyhf"
    assert result["rss_delta_mb"] == 2.0

    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        failed = benchmark.failed_framework_result("unknown", exc)
    assert failed["plot_label"] == "unknown"
    assert failed["status"] == "failed"
    assert failed["error_type"] == "RuntimeError"


def patch_time_and_memory(monkeypatch: pytest.MonkeyPatch, n_calls: int = 20) -> None:
    values = iter([float(i) / 10.0 for i in range(n_calls)])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(values))
    rss_values = iter([10.0, 12.0] * 10)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: next(rss_values))


def test_measure_manual_currently_fails_for_missing_manual_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_time_and_memory(monkeypatch)
    with pytest.raises(KeyError, match="manual"):
        benchmark.measure_manual(valid_parameters(), 2, 1.0, 1)


def test_measure_pyhs3_success(
    monkeypatch: pytest.MonkeyPatch, workspace_file: Path
) -> None:
    patch_time_and_memory(monkeypatch)
    monkeypatch.setattr(
        benchmark.Workspace,
        "load",
        lambda path: FakeWorkspace(model=FakePyhs3Model([0.5, 0.25])),
    )
    result = benchmark.measure_pyhs3(
        workspace_file, n_bins=2, mu=1.0, warmup_iterations=1
    )
    assert result["framework"] == "pyhs3"
    assert result["status"] == "success"


def test_measure_pyhf_success(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_time_and_memory(monkeypatch)
    monkeypatch.setattr(benchmark.pyhf, "Workspace", FakePyhfWorkspace)
    result = benchmark.measure_pyhf(
        valid_parameters(), n_bins=2, mu=1.0, warmup_iterations=1
    )
    assert result["framework"] == "pyhf"
    assert result["status"] == "success"


def test_measure_roofit_rejects_missing_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "ROOT", None)
    with pytest.raises(RuntimeError, match="ROOT is not available"):
        benchmark.measure_roofit(valid_parameters(), 2, 1.0, 1)


def test_measure_roofit_success(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_time_and_memory(monkeypatch, n_calls=30)
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    result = benchmark.measure_roofit(
        valid_parameters(), n_bins=2, mu_value=1.0, warmup_iterations=1
    )
    assert result["framework"] == "roofit"
    assert result["status"] == "success"


@pytest.mark.parametrize("framework", ["pyhs3", "pyhf", "roofit"])
def test_measure_framework_dispatch_success(
    monkeypatch: pytest.MonkeyPatch, workspace_file: Path, framework: str
) -> None:
    monkeypatch.setattr(
        benchmark,
        "measure_pyhs3",
        lambda *args: {"framework": "pyhs3", "status": "success"},
    )
    monkeypatch.setattr(
        benchmark,
        "measure_pyhf",
        lambda *args: {"framework": "pyhf", "status": "success"},
    )
    monkeypatch.setattr(
        benchmark,
        "measure_roofit",
        lambda *args: {"framework": "roofit", "status": "success"},
    )
    result = benchmark.measure_framework(
        framework=framework,
        workspace_path=workspace_file,
        parameters=valid_parameters(),
        n_bins=2,
        mu=1.0,
        warmup_iterations=1,
    )
    assert result["framework"] == framework
    assert result["status"] == "success"


def test_measure_framework_returns_failed_for_unknown(workspace_file: Path) -> None:
    result = benchmark.measure_framework(
        framework="unknown",
        workspace_path=workspace_file,
        parameters=valid_parameters(),
        n_bins=2,
        mu=1.0,
        warmup_iterations=1,
    )
    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"


# ---------------------------------------------------------------------------
# Validation/reporting
# ---------------------------------------------------------------------------


def test_add_validation_success_and_failure(
    all_success_results: list[dict[str, Any]],
) -> None:
    results = [dict(item) for item in all_success_results]
    results[0]["value"] = 3.1
    results.append({"framework": "bad", "status": "failed"})
    benchmark.add_validation(results, tolerance=0.05)
    assert results[1]["validation_status"] == "success"
    assert results[0]["validation_status"] == "failed"
    assert results[-1]["validation_status"] == "failed"
    assert results[-1]["value_abs_diff_from_pyhf"] is None


def test_add_validation_rejects_missing_pyhf() -> None:
    with pytest.raises(ValueError, match="without a successful pyhf"):
        benchmark.add_validation(
            [{"framework": "pyhs3", "status": "success", "value": 1.0}], tolerance=1e-9
        )


def test_benchmark_status() -> None:
    assert benchmark.benchmark_status([]) == "failed"
    assert (
        benchmark.benchmark_status(
            [{"status": "success", "validation_status": "success"}]
        )
        == "success"
    )
    assert (
        benchmark.benchmark_status(
            [{"status": "success", "validation_status": "failed"}]
        )
        == "failed"
    )


def test_print_result_success_and_failed(
    capsys: pytest.CaptureFixture[str], success_result: dict[str, Any]
) -> None:
    benchmark.print_result(success_result)
    output = capsys.readouterr().out
    assert "status:" in output
    assert "NLL value" in output

    benchmark.print_result(
        {
            "framework": "unknown",
            "status": "failed",
            "error_type": "X",
            "error_message": "bad",
        }
    )
    output = capsys.readouterr().out
    assert "error:" in output


def test_build_failed_output(workspace_file: Path) -> None:
    try:
        raise ValueError("bad")
    except ValueError as exc:
        output = benchmark.build_failed_output(
            workspace_path=workspace_file,
            n_bins=None,
            mu=1.0,
            frameworks=["pyhf"],
            warmup_iterations=1,
            agreement_tolerance=1e-9,
            exc=exc,
        )
    assert output["benchmark"] == benchmark.BENCHMARK_NAME
    assert output["status"] == "failed"
    assert output["error_type"] == "ValueError"
    assert output["results"] == []


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plot_helpers(success_result: dict[str, Any]) -> None:
    assert benchmark._successful_results([success_result, {"status": "failed"}]) == [
        success_result
    ]
    assert benchmark._framework_colors([success_result]) == [
        benchmark.FRAMEWORK_STYLE["pyhf"]["color"]
    ]
    assert benchmark._plot_floor([0.0, 2.0], floor=0.1) == [0.1, 2.0]
    assert benchmark._format_compact_number(1000.0) == "1000"
    assert benchmark._format_compact_number(100.0) == "100"
    assert benchmark._format_compact_number(10.0) == "10.0"
    assert benchmark._format_compact_number(1.0) == "1.00"
    assert benchmark._format_compact_number(0.01) == "0.01"


def test_add_bar_labels() -> None:
    fig, ax = benchmark.plt.subplots()
    bars = ax.bar([0], [1.0])
    benchmark._add_bar_labels(ax, bars, [1.0], lambda value: f"{value}")
    assert ax.texts
    benchmark.plt.close(fig)


def test_save_figure_creates_png(tmp_path: Path) -> None:
    fig, _ax = benchmark.plt.subplots()
    output = tmp_path / "figure_without_suffix"
    benchmark._save_figure(fig, output)
    assert (tmp_path / "figure_without_suffix.png").exists()


def test_save_figure_wraps_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BadFig:
        def savefig(self, *args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

    closed: list[Any] = []
    monkeypatch.setattr(benchmark.plt, "close", lambda fig: closed.append(fig))
    with pytest.raises(OSError, match="Failed to save plot"):
        benchmark._save_figure(BadFig(), tmp_path / "plot.png")
    assert closed


@pytest.mark.parametrize(
    ("plot_func", "filename"),
    [
        (benchmark.make_setup_timing_plot, "setup.png"),
        (benchmark.make_evaluation_latency_plot, "latency.png"),
        (benchmark.make_memory_plot, "memory.png"),
        (benchmark.make_value_agreement_plot, "agreement.png"),
        (benchmark.make_summary_table_plot, "summary.png"),
    ],
)
def test_individual_plot_functions_create_png(
    tmp_path: Path,
    all_success_results: list[dict[str, Any]],
    plot_func: Any,
    filename: str,
) -> None:
    output = tmp_path / filename
    plot_func(all_success_results, output)
    assert output.exists()


@pytest.mark.parametrize(
    "plot_func",
    [
        benchmark.make_setup_timing_plot,
        benchmark.make_evaluation_latency_plot,
        benchmark.make_memory_plot,
        benchmark.make_summary_table_plot,
    ],
)
def test_plot_functions_reject_no_success(tmp_path: Path, plot_func: Any) -> None:
    with pytest.raises(ValueError, match="No successful"):
        plot_func([{"status": "failed"}], tmp_path / "plot.png")


def test_value_agreement_plot_rejects_only_reference(
    tmp_path: Path, success_result: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="No non-reference"):
        benchmark.make_value_agreement_plot([success_result], tmp_path / "plot.png")


def test_make_plots_creates_expected_pngs(
    tmp_path: Path, all_success_results: list[dict[str, Any]]
) -> None:
    benchmark.make_plots(all_success_results, tmp_path)
    expected = {
        "model_build_setup_timing.png",
        "model_build_setup_evaluation_latency.png",
        "model_build_setup_memory.png",
        "model_build_setup_value_agreement.png",
        "model_build_setup_summary_table.png",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})


def test_make_plots_rejects_no_success(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No successful"):
        benchmark.make_plots([{"status": "failed"}], tmp_path)


# ---------------------------------------------------------------------------
# Runner / CLI
# ---------------------------------------------------------------------------


def test_run_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_file: Path,
    all_success_results: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(benchmark.Workspace, "load", lambda path: FakeWorkspace())
    sequence = iter([dict(item) for item in all_success_results])
    monkeypatch.setattr(benchmark, "measure_framework", lambda **kwargs: next(sequence))
    plot_calls: list[Any] = []
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda *args, **kwargs: plot_calls.append((args, kwargs)),
    )

    output = tmp_path / "result.json"
    result = benchmark.run(
        workspace_path=workspace_file,
        n_bins=None,
        mu=1.0,
        frameworks=["pyhf", "pyhs3", "roofit"],
        warmup_iterations=1,
        agreement_tolerance=1e-9,
        output=output,
        plot=True,
        plot_dir=tmp_path / "plots",
    )

    assert result["status"] == "success"
    assert result["n_bins"] == 2
    assert result["frameworks"] == ["pyhs3", "pyhf", "roofit"]
    assert json.loads(output.read_text())["status"] == "success"
    assert plot_calls


def test_run_fails_on_n_bins_mismatch(
    monkeypatch: pytest.MonkeyPatch, workspace_file: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(benchmark.Workspace, "load", lambda path: FakeWorkspace())
    output = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="Model build"):
        benchmark.run(
            workspace_path=workspace_file,
            n_bins=3,
            frameworks=["pyhf"],
            output=output,
        )
    payload = json.loads(output.read_text())
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ValueError"


def test_run_fail_fast_on_framework_failure(
    monkeypatch: pytest.MonkeyPatch, workspace_file: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(benchmark.Workspace, "load", lambda path: FakeWorkspace())
    monkeypatch.setattr(
        benchmark,
        "measure_framework",
        lambda **kwargs: {
            "framework": kwargs["framework"],
            "status": "failed",
            "error_message": "boom",
        },
    )
    output = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="Model build"):
        benchmark.run(
            workspace_path=workspace_file,
            frameworks=["pyhf"],
            output=output,
            continue_on_framework_error=False,
        )
    assert json.loads(output.read_text())["status"] == "failed"


def test_run_continues_on_framework_failure_but_validation_fails(
    monkeypatch: pytest.MonkeyPatch, workspace_file: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(benchmark.Workspace, "load", lambda path: FakeWorkspace())
    monkeypatch.setattr(
        benchmark,
        "measure_framework",
        lambda **kwargs: {
            "framework": kwargs["framework"],
            "status": "failed",
            "error_message": "boom",
        },
    )
    output = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="Model build"):
        benchmark.run(
            workspace_path=workspace_file,
            frameworks=["pyhf"],
            output=output,
            continue_on_framework_error=True,
        )
    assert json.loads(output.read_text())["error_type"] == "ValueError"


def test_run_rejects_invalid_agreement_tolerance(
    workspace_file: Path, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="Model build"):
        benchmark.run(
            workspace_path=workspace_file,
            frameworks=["pyhf"],
            agreement_tolerance=0.0,
            output=tmp_path / "result.json",
        )


def test_run_handles_failure_report_save_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_file: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cannot save")),
    )
    with pytest.raises(RuntimeError, match="Model build"):
        benchmark.run(
            workspace_path=tmp_path / "missing.json",
            frameworks=["pyhf"],
            output=tmp_path / "result.json",
        )
    assert "Failed to save benchmark failure report" in capsys.readouterr().err


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_model_build_setup_cost.py"])
    args = benchmark.parse_args()
    assert args.workspace == benchmark.DEFAULT_WORKSPACE
    assert args.frameworks == benchmark.DEFAULT_FRAMEWORKS
    assert args.mu == benchmark.DEFAULT_MU
    assert args.plot is False
    assert args.fail_fast is False


def test_parse_args_custom_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_build_setup_cost.py",
            "--workspace",
            str(tmp_path / "workspace.json"),
            "--n-bins",
            "2",
            "--mu",
            "1.5",
            "--frameworks",
            "pyhf",
            "pyhs3",
            "--warmup-iterations",
            "0",
            "--agreement-tolerance",
            "1e-8",
            "--output",
            str(tmp_path / "out.json"),
            "--plot",
            "--plot-dir",
            str(tmp_path / "plots"),
            "--fail-fast",
        ],
    )
    args = benchmark.parse_args()
    assert args.workspace == tmp_path / "workspace.json"
    assert args.n_bins == 2
    assert args.mu == 1.5
    assert args.frameworks == ["pyhf", "pyhs3"]
    assert args.warmup_iterations == 0
    assert args.agreement_tolerance == 1e-8
    assert args.output == tmp_path / "out.json"
    assert args.plot is True
    assert args.plot_dir == tmp_path / "plots"
    assert args.fail_fast is True


def test_parse_args_rejects_unknown_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["run_model_build_setup_cost.py", "--frameworks", "bad"]
    )
    with pytest.raises(SystemExit):
        benchmark.parse_args()


def test_main_passes_cli_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_build_setup_cost.py",
            "--workspace",
            str(tmp_path / "workspace.json"),
            "--n-bins",
            "2",
            "--frameworks",
            "pyhf",
            "--output",
            str(tmp_path / "out.json"),
            "--fail-fast",
        ],
    )
    monkeypatch.setattr(benchmark, "run", lambda **kwargs: calls.append(kwargs))
    benchmark.main()
    assert calls[0]["workspace_path"] == tmp_path / "workspace.json"
    assert calls[0]["n_bins"] == 2
    assert calls[0]["frameworks"] == ["pyhf"]
    assert calls[0]["continue_on_framework_error"] is False
