from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src import run_cross_nll_regression as benchmark


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


@pytest.fixture
def parameters() -> dict[str, float]:
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
def mu_grid() -> list[float]:
    return [0.0, 1.0, 2.0]


def _metrics(
    framework: str,
    *,
    validation_status: str = "success",
    delta: float = 0.0,
) -> dict[str, Any]:
    return {
        "framework": framework,
        "max_abs_diff": delta,
        "max_rel_diff": delta,
        "mean_abs_diff": delta,
        "std_abs_diff": 0.0,
        "constant_offset": 0.0,
        "centered_residual_max_abs_diff": delta,
        "delta_nll_max_abs_diff": delta,
        "minimum_mu": 1.0,
        "reference_minimum_mu": 1.0,
        "minimum_mu_abs_diff": 0.0,
        "allclose_passed": delta == 0.0,
        "finite_values": True,
        "delta_shape_success": validation_status == "success",
        "minimum_mu_success": True,
        "validation_status": validation_status,
    }


@pytest.fixture
def successful_results() -> list[dict[str, Any]]:
    base = {
        "status": "success",
        "n_points": 3,
        "nll_values": [3.0, 1.0, 3.0],
        "delta_nll_shape": [2.0, 0.0, 2.0],
        "minimum_mu": 1.0,
        "model_build_time_seconds": 0.001,
        "full_scan_time_seconds": 0.003,
        "time_per_scan_point_seconds": 0.001,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 101.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 122.0,
        "peak_rss_delta_mb": 2.0,
        "nll_summary": {"mean": 7.0 / 3.0, "std": 1.1547, "min": 1.0, "max": 3.0},
    }
    manual = {**base, "framework": "manual", "metrics": _metrics("manual")}
    pyhs3 = {
        **base,
        "framework": "pyhs3",
        "nll_values": [3.0, 1.0, 3.0],
        "delta_nll_shape": [2.0, 0.0, 2.0],
        "current_rss_delta_mb": 1.5,
        "metrics": _metrics("pyhs3"),
    }
    pyhf = {
        **base,
        "framework": "pyhf",
        "nll_values": [3.0000000001, 1.0000000001, 3.0000000001],
        "delta_nll_shape": [2.0, 0.0, 2.0],
        "current_rss_delta_mb": 1.7,
        "metrics": _metrics("pyhf", delta=1e-12),
    }
    return [manual, pyhs3, pyhf]


def test_framework_spec_dataclass() -> None:
    spec = benchmark.FrameworkSpec("manual", lambda: "model", lambda model, mu: 1.0)
    assert spec.name == "manual"
    assert spec.build_func() == "model"
    assert spec.eval_func(object(), 1.0) == 1.0


def test_validate_workspace_path_success(workspace_path: Path) -> None:
    assert benchmark.validate_workspace_path(workspace_path) == workspace_path


def test_validate_workspace_path_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.validate_workspace_path(tmp_path / "missing.json")


def test_validate_workspace_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace path is not a file"):
        benchmark.validate_workspace_path(tmp_path)


@pytest.mark.parametrize(
    ("value", "minimum"),
    [(1, 1), (2, 2)],
)
def test_validate_positive_int_accepts_valid_values(value: int, minimum: int) -> None:
    benchmark.validate_positive_int(value, "value", minimum=minimum)


@pytest.mark.parametrize("value", [0, -1])
def test_validate_positive_int_rejects_invalid_values(value: int) -> None:
    with pytest.raises(ValueError, match="value must be at least 1"):
        benchmark.validate_positive_int(value, "value")


def test_validate_finite_float_success() -> None:
    benchmark.validate_finite_float(1.25, "x")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_finite_float_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="x must be finite"):
        benchmark.validate_finite_float(value, "x")


def test_validate_benchmark_config_success() -> None:
    benchmark.validate_benchmark_config(
        mu_min=0.0,
        mu_max=2.0,
        n_points=3,
        rtol=0.0,
        atol=0.0,
        delta_tolerance=1e-9,
        minimum_tolerance=1e-12,
        frameworks=["manual", "pyhs3"],
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_points": 1}, "n_points must be at least 2"),
        ({"mu_min": float("nan")}, "mu_min must be finite"),
        ({"mu_max": float("inf")}, "mu_max must be finite"),
        ({"rtol": float("nan")}, "rtol must be finite"),
        ({"atol": float("nan")}, "atol must be finite"),
        ({"delta_tolerance": float("nan")}, "delta_tolerance must be finite"),
        ({"minimum_tolerance": float("nan")}, "minimum_tolerance must be finite"),
        ({"mu_min": 2.0, "mu_max": 1.0}, "mu_min must be smaller"),
        ({"rtol": -1.0}, "rtol must be non-negative"),
        ({"atol": -1.0}, "atol must be non-negative"),
        ({"delta_tolerance": 0.0}, "delta_tolerance must be positive"),
        ({"minimum_tolerance": 0.0}, "minimum_tolerance must be positive"),
        ({"frameworks": []}, "At least one framework"),
        ({"frameworks": ["manual", "unknown"]}, "Unknown frameworks"),
        ({"frameworks": ["pyhs3"]}, "manual framework is required"),
    ],
)
def test_validate_benchmark_config_rejects_invalid_values(
    overrides: dict[str, Any],
    message: str,
) -> None:
    kwargs = {
        "mu_min": 0.0,
        "mu_max": 2.0,
        "n_points": 3,
        "rtol": 0.0,
        "atol": 0.0,
        "delta_tolerance": 1e-9,
        "minimum_tolerance": 1e-12,
        "frameworks": ["manual"],
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        benchmark.validate_benchmark_config(**kwargs)


def test_validate_parameters_success(parameters: dict[str, float]) -> None:
    benchmark.validate_parameters(parameters, 2)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"mu": 1.0}, "missing parameters"),
        (
            {"mu": 1.0, "signal_0": float("nan"), "background_0": 1.0, "obs_0": 1.0},
            "must be finite",
        ),
        (
            {"mu": 1.0, "signal_0": -1.0, "background_0": 1.0, "obs_0": 1.0},
            "non-negative",
        ),
    ],
)
def test_validate_parameters_rejects_invalid(
    params: dict[str, float], message: str
) -> None:
    with pytest.raises((KeyError, ValueError), match=message):
        benchmark.validate_parameters(params, 1)


def test_validate_parameters_allows_negative_mu(parameters: dict[str, float]) -> None:
    parameters = dict(parameters)
    parameters["mu"] = -1.0
    benchmark.validate_parameters(parameters, 2)


def test_validate_scan_values_success() -> None:
    benchmark.validate_scan_values((value for value in [1.0, 2.0]), "values")


@pytest.mark.parametrize("values", [[], [1.0, float("nan")], [float("inf")]])
def test_validate_scan_values_rejects_invalid(values: list[float]) -> None:
    with pytest.raises(ValueError):
        benchmark.validate_scan_values(values, "values")


def test_extract_parameters_success() -> None:
    ws = SimpleNamespace(
        parameter_points=SimpleNamespace(
            root=[SimpleNamespace(parameters=[SimpleNamespace(name="mu", value="1.0")])]
        )
    )
    assert benchmark.extract_parameters(ws) == {"mu": 1.0}


@pytest.mark.parametrize(
    "ws",
    [
        SimpleNamespace(parameter_points=SimpleNamespace(root=[])),
        SimpleNamespace(
            parameter_points=SimpleNamespace(root=[SimpleNamespace(parameters=None)])
        ),
    ],
)
def test_extract_parameters_rejects_malformed_workspace(ws: Any) -> None:
    with pytest.raises(ValueError, match="Could not extract initial parameters"):
        benchmark.extract_parameters(ws)


def test_extract_parameters_rejects_empty_parameter_set() -> None:
    ws = SimpleNamespace(
        parameter_points=SimpleNamespace(root=[SimpleNamespace(parameters=[])])
    )
    with pytest.raises(ValueError, match="does not contain parameters"):
        benchmark.extract_parameters(ws)


def test_infer_n_bins_from_parameters_success(parameters: dict[str, float]) -> None:
    assert benchmark.infer_n_bins_from_parameters(parameters) == 2


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"mu": 1.0}, "no signal"),
        ({"signal_1": 1.0, "background_1": 1.0, "obs_1": 1.0}, "contiguous"),
        ({"signal_0": 1.0}, "parameters are missing"),
    ],
)
def test_infer_n_bins_from_parameters_rejects_invalid(
    params: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.infer_n_bins_from_parameters(params)


def test_poisson_nll_matches_formula() -> None:
    assert benchmark.poisson_nll(3.0, 2.5) == pytest.approx(
        2.5 - 3.0 * math.log(2.5) + math.lgamma(4.0)
    )


@pytest.mark.parametrize(
    ("observed", "expected", "message"),
    [(1.0, 0.0, "positive"), (-1.0, 1.0, "non-negative")],
)
def test_poisson_nll_rejects_invalid_inputs(
    observed: float, expected: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.poisson_nll(observed, expected)


def test_get_vectors_and_manual_model(parameters: dict[str, float]) -> None:
    vectors = benchmark.get_vectors(parameters, 2)
    assert vectors == ([2.0, 3.0], [10.0, 11.0], [12.0, 14.0])
    assert benchmark.build_manual_model(parameters, 2) == vectors


def test_manual_nll_sums_bin_nlls() -> None:
    model = ([2.0], [10.0], [12.0])
    assert benchmark.manual_nll(model, 1.0) == pytest.approx(
        benchmark.poisson_nll(12.0, 12.0)
    )


def test_build_pyhs3_model_uses_workspace_load(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    model = object()
    fake_workspace = SimpleNamespace(model=lambda *args, **kwargs: model)
    monkeypatch.setattr(benchmark.Workspace, "load", lambda path: fake_workspace)
    assert benchmark.build_pyhs3_model(workspace_path) is model


def test_pyhs3_nll_success() -> None:
    class FakeModel:
        def pdf(self, name: str, **kwargs: Any) -> list[float]:
            assert name in {"poisson_0", "poisson_1"}
            assert "mu" in kwargs
            return [0.5]

    assert benchmark.pyhs3_nll(FakeModel(), 2, 1.0) == pytest.approx(
        -2.0 * math.log(0.5)
    )


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_pyhs3_nll_rejects_invalid_pdf_value(value: float) -> None:
    class FakeModel:
        def pdf(self, name: str, **kwargs: Any) -> list[float]:
            return [value]

    with pytest.raises(ValueError, match="PyHS3 returned invalid PDF"):
        benchmark.pyhs3_nll(FakeModel(), 1, 1.0)


def test_build_pyhf_spec_uses_input_arrays(parameters: dict[str, float]) -> None:
    spec = benchmark.build_pyhf_spec(parameters, 2)
    assert spec["version"] == "1.0.0"
    assert spec["channels"][0]["samples"][0]["data"] == [2.0, 3.0]
    assert spec["channels"][0]["samples"][0]["modifiers"][0]["name"] == "mu"
    assert spec["channels"][0]["samples"][1]["data"] == [10.0, 11.0]
    assert spec["observations"][0]["data"] == [12.0, 14.0]
    assert spec["measurements"][0]["config"]["poi"] == "mu"


def test_build_pyhf_model_rejects_missing_pyhf(
    monkeypatch: pytest.MonkeyPatch, parameters: dict[str, float]
) -> None:
    monkeypatch.setattr(benchmark, "pyhf", None)
    with pytest.raises(RuntimeError, match="pyhf is not available"):
        benchmark.build_pyhf_model(parameters, 1)


def test_build_pyhf_model_success(
    monkeypatch: pytest.MonkeyPatch, parameters: dict[str, float]
) -> None:
    fake_model = object()
    fake_data = [1, 2, 3]

    class FakeWorkspace:
        def __init__(self, spec: dict[str, Any]) -> None:
            self.spec = spec

        def model(self) -> object:
            return fake_model

        def data(self, model: object) -> list[int]:
            assert model is fake_model
            return fake_data

    monkeypatch.setattr(benchmark, "pyhf", SimpleNamespace(Workspace=FakeWorkspace))
    assert benchmark.build_pyhf_model(parameters, 1) == (fake_model, fake_data)


def test_pyhf_nll_success() -> None:
    class Config:
        par_order = ["alpha", "mu"]

        def suggested_init(self) -> list[float]:
            return [0.0, 1.0]

    class Model:
        config = Config()

        def logpdf(self, pars: list[float], data: Any) -> list[float]:
            assert pars == [0.0, 2.5]
            assert data == ["data"]
            return [-12.5]

    assert benchmark.pyhf_nll((Model(), ["data"]), 2.5) == pytest.approx(12.5)


def test_pyhf_nll_rejects_non_finite() -> None:
    class Config:
        par_order = ["mu"]

        def suggested_init(self) -> list[float]:
            return [1.0]

    class Model:
        config = Config()

        def logpdf(self, pars: list[float], data: Any) -> list[float]:
            return [float("nan")]

    with pytest.raises(ValueError, match="pyhf returned non-finite"):
        benchmark.pyhf_nll((Model(), []), 1.0)


def test_build_roofit_model_rejects_missing_root(
    monkeypatch: pytest.MonkeyPatch, parameters: dict[str, float]
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", None)
    with pytest.raises(RuntimeError, match="ROOT is not available"):
        benchmark.build_roofit_model(parameters, 1, 0.0)


def test_build_roofit_model_success(
    monkeypatch: pytest.MonkeyPatch, parameters: dict[str, float]
) -> None:
    class RooFit:
        ERROR = object()

    class MsgService:
        @staticmethod
        def instance() -> "MsgService":
            return MsgService()

        def setGlobalKillBelow(self, level: object) -> None:
            self.level = level

    class RooRealVar:
        def __init__(self, name: str, title: str, value: float, *bounds: float) -> None:
            self.name = name
            self.value = value
            self.constant = False

        def setConstant(self, value: bool) -> None:
            self.constant = value

        def setVal(self, value: float) -> None:
            self.value = value

    class RooArgList(list):
        def __init__(self, *args: Any) -> None:
            super().__init__(args)

    class RooFormulaVar:
        def __init__(self, *args: Any) -> None:
            self.args = args

    class RooPoisson:
        def __init__(self, *args: Any) -> None:
            self.args = args

        def getVal(self) -> float:
            return 0.5

    fake_root = SimpleNamespace(
        RooFit=RooFit,
        RooMsgService=MsgService,
        RooRealVar=RooRealVar,
        RooArgList=RooArgList,
        RooFormulaVar=RooFormulaVar,
        RooPoisson=RooPoisson,
    )
    monkeypatch.setattr(benchmark, "ROOT", fake_root)

    model = benchmark.build_roofit_model(parameters, 2, 0.0)
    assert set(model) == {"mu", "poissons", "keepalive"}
    assert len(model["poissons"]) == 2
    assert len(model["keepalive"]) >= 1


def test_roofit_nll_success() -> None:
    class Mu:
        def __init__(self) -> None:
            self.value = None

        def setVal(self, value: float) -> None:
            self.value = value

    class Poisson:
        def getVal(self) -> float:
            return 0.25

    model = {"mu": Mu(), "poissons": [Poisson(), Poisson()]}
    assert benchmark.roofit_nll(model, 2.0) == pytest.approx(-2.0 * math.log(0.25))
    assert model["mu"].value == 2.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_roofit_nll_rejects_invalid_pdf_value(value: float) -> None:
    class Mu:
        def setVal(self, value: float) -> None:
            pass

    class Poisson:
        def getVal(self) -> float:
            return value

    with pytest.raises(ValueError, match="RooFit returned invalid PDF"):
        benchmark.roofit_nll({"mu": Mu(), "poissons": [Poisson()]}, 1.0)


def test_build_mu_grid_success() -> None:
    assert benchmark.build_mu_grid(0.0, 1.0, 3) == [0.0, 0.5, 1.0]


def test_build_mu_grid_reuses_config_validation() -> None:
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        benchmark.build_mu_grid(0.0, 1.0, 1)


def test_run_scan_success() -> None:
    assert benchmark.run_scan("model", lambda model, mu: mu + 1.0, [0.0, 1.0]) == [
        1.0,
        2.0,
    ]


def test_run_scan_rejects_non_finite_outputs() -> None:
    with pytest.raises(ValueError, match="nll_values"):
        benchmark.run_scan("model", lambda model, mu: float("nan"), [0.0])


def test_delta_nll_success() -> None:
    np.testing.assert_allclose(
        benchmark.delta_nll(np.asarray([3.0, 1.0, 2.0])), [2.0, 0.0, 1.0]
    )


@pytest.mark.parametrize("values", [np.asarray([]), np.asarray([1.0, float("nan")])])
def test_delta_nll_rejects_invalid(values: np.ndarray) -> None:
    with pytest.raises(ValueError):
        benchmark.delta_nll(values)


def test_minimum_position_success() -> None:
    assert benchmark.minimum_position([0.0, 1.0, 2.0], [3.0, 1.0, 2.0]) == 1.0


def test_minimum_position_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        benchmark.minimum_position([0.0], [1.0, 2.0])


def test_summarize_values_one_and_many() -> None:
    assert benchmark.summarize_values([2.0]) == {
        "mean": 2.0,
        "std": 0.0,
        "min": 2.0,
        "max": 2.0,
    }
    result = benchmark.summarize_values([1.0, 3.0])
    assert result["mean"] == 2.0
    assert result["std"] == pytest.approx(math.sqrt(2.0))


def test_make_framework_specs_builds_requested_specs(
    parameters: dict[str, float], workspace_path: Path
) -> None:
    specs = benchmark.make_framework_specs(
        ["manual", "pyhs3", "pyhf", "roofit"], parameters, workspace_path, 2, 0.0
    )
    assert [spec.name for spec in specs] == ["manual", "pyhs3", "pyhf", "roofit"]


def test_make_framework_specs_manual_callable(
    parameters: dict[str, float], workspace_path: Path
) -> None:
    spec = benchmark.make_framework_specs(
        ["manual"], parameters, workspace_path, 2, 0.0
    )[0]
    model = spec.build_func()
    assert spec.eval_func(model, 1.0) == pytest.approx(benchmark.manual_nll(model, 1.0))


def test_make_framework_specs_unknown_framework_raises(
    parameters: dict[str, float], workspace_path: Path
) -> None:
    with pytest.raises(ValueError, match="Unknown framework"):
        benchmark.make_framework_specs(["unknown"], parameters, workspace_path, 2, 0.0)


def test_compute_metrics_success_and_failed_status(mu_grid: list[float]) -> None:
    result = benchmark.compute_metrics(
        framework="pyhs3",
        reference_values=[3.0, 1.0, 3.0],
        values=[3.0, 1.0, 3.0],
        mu_grid=mu_grid,
        rtol=0.0,
        atol=0.0,
        delta_tolerance=1e-12,
        minimum_tolerance=1e-12,
    )
    assert result["validation_status"] == "success"
    assert result["allclose_passed"] is True
    assert result["delta_nll_max_abs_diff"] == 0.0

    failed = benchmark.compute_metrics(
        framework="pyhs3",
        reference_values=[3.0, 1.0, 3.0],
        values=[3.0, 1.0, 4.0],
        mu_grid=mu_grid,
        rtol=0.0,
        atol=0.0,
        delta_tolerance=1e-12,
        minimum_tolerance=1e-12,
    )
    assert failed["validation_status"] == "failed"
    assert failed["allclose_passed"] is False


def test_compute_metrics_rejects_shape_mismatch(mu_grid: list[float]) -> None:
    with pytest.raises(ValueError, match="different shapes"):
        benchmark.compute_metrics(
            framework="x",
            reference_values=[1.0],
            values=[1.0, 2.0],
            mu_grid=mu_grid,
            rtol=0.0,
            atol=0.0,
            delta_tolerance=1.0,
            minimum_tolerance=1.0,
        )


def test_measure_framework_regression_success(
    monkeypatch: pytest.MonkeyPatch, mu_grid: list[float]
) -> None:
    monkeypatch.setattr(benchmark, "get_current_rss_mb", iter([100.0, 103.0]).__next__)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", iter([120.0, 125.0]).__next__)
    monkeypatch.setattr(
        benchmark.time, "perf_counter", iter([1.0, 1.2, 2.0, 2.6]).__next__
    )

    result = benchmark.measure_framework_regression(
        name="manual",
        build_func=lambda: "model",
        eval_func=lambda model, mu: {0.0: 3.0, 1.0: 1.0, 2.0: 3.0}[mu],
        mu_grid=mu_grid,
    )

    assert result["framework"] == "manual"
    assert result["status"] == "success"
    assert result["model_build_time_seconds"] == pytest.approx(0.2)
    assert result["full_scan_time_seconds"] == pytest.approx(0.6)
    assert result["time_per_scan_point_seconds"] == pytest.approx(0.2)
    assert result["current_rss_delta_mb"] == pytest.approx(3.0)
    assert result["peak_rss_delta_mb"] == pytest.approx(5.0)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"minimum_mu": float("nan")}, "minimum_mu"),
        (
            {"model_build_time_seconds": -0.1},
            "model_build_time_seconds must be non-negative",
        ),
        ({"full_scan_time_seconds": 0.0}, "full_scan_time_seconds must be positive"),
        ({"nll_values": [float("nan")]}, "nll_values contains non-finite"),
        ({"delta_nll_shape": []}, "delta_nll_shape must not be empty"),
    ],
)
def test_validate_framework_result_rejects_invalid(
    updates: dict[str, Any], message: str
) -> None:
    result = {
        "framework": "manual",
        "minimum_mu": 1.0,
        "model_build_time_seconds": 0.1,
        "full_scan_time_seconds": 0.2,
        "time_per_scan_point_seconds": 0.1,
        "current_rss_delta_mb": 0.0,
        "peak_rss_delta_mb": 0.0,
        "nll_values": [1.0],
        "delta_nll_shape": [0.0],
    }
    result.update(updates)
    with pytest.raises(ValueError, match=message):
        benchmark.validate_framework_result(result)


def test_failed_framework_result_contains_error() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        result = benchmark.failed_framework_result("pyhf", exc)

    assert result["framework"] == "pyhf"
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["validation_status"] == "not_run"
    assert "boom" in result["error_message"]
    assert "RuntimeError" in result["traceback"]


def test_add_regression_metrics_success_and_failed_framework(
    mu_grid: list[float],
) -> None:
    results = [
        {"framework": "manual", "status": "success", "nll_values": [3.0, 1.0, 3.0]},
        {"framework": "pyhs3", "status": "success", "nll_values": [3.0, 1.0, 3.0]},
        {"framework": "pyhf", "status": "failed"},
    ]
    for result in results[:2]:
        result["delta_nll_shape"] = [2.0, 0.0, 2.0]

    benchmark.add_regression_metrics(
        results=results,
        mu_grid=mu_grid,
        rtol=0.0,
        atol=0.0,
        delta_tolerance=1e-12,
        minimum_tolerance=1e-12,
    )

    assert results[0]["validation_status"] == "success"
    assert results[1]["metrics"]["framework"] == "pyhs3"
    assert results[2]["metrics"] is None


def test_add_regression_metrics_requires_manual(mu_grid: list[float]) -> None:
    with pytest.raises(ValueError, match="without manual"):
        benchmark.add_regression_metrics(
            results=[{"framework": "pyhs3", "status": "success", "nll_values": [1.0]}],
            mu_grid=mu_grid,
            rtol=0.0,
            atol=0.0,
            delta_tolerance=1.0,
            minimum_tolerance=1.0,
        )


def test_apply_cern_style_and_style_helpers() -> None:
    benchmark._apply_cern_style()
    assert plt.rcParams["figure.facecolor"] == "white"
    assert benchmark._style_for("manual")["label"] == "Manual reference"
    assert benchmark._style_for("new")["label"] == "new"
    assert benchmark._framework_label("pyhs3") == "PyHS3"


def test_successful_and_reference_results(
    successful_results: list[dict[str, Any]],
) -> None:
    mixed = [*successful_results, {"framework": "bad", "status": "failed"}]
    assert [result["framework"] for result in benchmark._successful_results(mixed)] == [
        "manual",
        "pyhs3",
        "pyhf",
    ]
    assert benchmark._reference_result(mixed)["framework"] == "manual"


def test_reference_result_requires_manual() -> None:
    with pytest.raises(ValueError, match="Manual reference"):
        benchmark._reference_result([{"framework": "pyhs3", "status": "success"}])


def test_save_figure_creates_png(tmp_path: Path) -> None:
    fig, _ax = plt.subplots()
    output = tmp_path / "plot_without_suffix"
    benchmark._save_figure(fig, output)
    assert output.with_suffix(".png").exists()


def test_save_figure_wraps_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fig, _ax = plt.subplots()

    def failing_savefig(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(fig, "savefig", failing_savefig)
    with pytest.raises(OSError, match="Failed to save plot"):
        benchmark._save_figure(fig, tmp_path / "bad.png")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "nan"),
        (0.0, "0 ms"),
        (1e-4, "1.00e-04 ms"),
        (1.2345, "1.234 ms"),
        (12.345, "12.35 ms"),
        (123.45, "123.5 ms"),
        (1e5, "1.00e+05 ms"),
    ],
)
def test_format_metric(value: float, expected: str) -> None:
    assert benchmark._format_metric(value, " ms") == expected


@pytest.mark.parametrize(
    ("plot_func", "filename", "extra_args"),
    [
        (benchmark.make_regression_profile_plot, "profile.png", None),
        (benchmark.make_error_envelope_plot, "envelope.png", None),
        (benchmark.make_metric_summary_plot, "metrics.png", {"delta_tolerance": 1e-9}),
        (benchmark.make_offset_vs_shape_plot, "offset.png", None),
        (benchmark.make_summary_table_plot, "summary.png", None),
    ],
)
def test_individual_plot_functions_create_png(
    plot_func: Any,
    filename: str,
    extra_args: dict[str, Any] | None,
    tmp_path: Path,
    successful_results: list[dict[str, Any]],
    mu_grid: list[float],
) -> None:
    output_path = tmp_path / filename
    if plot_func in {
        benchmark.make_regression_profile_plot,
        benchmark.make_error_envelope_plot,
    }:
        plot_func(successful_results, mu_grid, output_path)
    elif plot_func is benchmark.make_metric_summary_plot:
        plot_func(successful_results, extra_args["delta_tolerance"], output_path)
    else:
        plot_func(successful_results, output_path)
    assert output_path.with_suffix(".png").exists()


def test_make_plots_creates_all_pngs(
    tmp_path: Path, successful_results: list[dict[str, Any]], mu_grid: list[float]
) -> None:
    benchmark.make_plots(successful_results, mu_grid, tmp_path, delta_tolerance=1e-9)
    expected = {
        "cross_nll_regression_profile.png",
        "cross_nll_regression_residual_envelope.png",
        "cross_nll_regression_agreement.png",
        "cross_nll_regression_offset_vs_shape.png",
        "cross_nll_regression_summary_table.png",
    }
    assert expected == {path.name for path in tmp_path.glob("*.png")}


def test_make_plots_requires_successful_results(
    tmp_path: Path, mu_grid: list[float]
) -> None:
    with pytest.raises(ValueError, match="No successful"):
        benchmark.make_plots(
            [{"framework": "manual", "status": "failed"}], mu_grid, tmp_path, 1e-9
        )


def test_framework_order(successful_results: list[dict[str, Any]]) -> None:
    assert benchmark._framework_order(
        [*successful_results, {"framework": "bad", "status": "failed"}]
    ) == [
        "manual",
        "pyhs3",
        "pyhf",
    ]


def test_print_result_success_and_failure(
    capsys: pytest.CaptureFixture[str], successful_results: list[dict[str, Any]]
) -> None:
    benchmark.print_result(successful_results[1])
    benchmark.print_result(
        {
            "framework": "pyhf",
            "status": "failed",
            "error_type": "RuntimeError",
            "error_message": "boom",
        }
    )
    output = capsys.readouterr().out
    assert "pyhs3" in output
    assert "validation:" in output
    assert "allclose passed" in output
    assert "RuntimeError: boom" in output


def test_build_failed_output(workspace_path: Path) -> None:
    try:
        raise RuntimeError("bad")
    except RuntimeError as exc:
        output = benchmark.build_failed_output(
            workspace_path=workspace_path,
            n_bins=2,
            mu_min=0.0,
            mu_max=2.0,
            n_points=3,
            rtol=1e-9,
            atol=1e-9,
            delta_tolerance=1e-9,
            minimum_tolerance=1e-12,
            frameworks=["manual"],
            exc=exc,
        )
    assert output["benchmark"] == benchmark.BENCHMARK_NAME
    assert output["workspace"] == workspace_path.name
    assert output["status"] == "failed"
    assert output["error_type"] == "RuntimeError"
    assert output["results"] == []


def _fake_workspace(parameters: dict[str, float]) -> SimpleNamespace:
    parameter_objects = [
        SimpleNamespace(name=name, value=value) for name, value in parameters.items()
    ]
    return SimpleNamespace(
        parameter_points=SimpleNamespace(
            root=[SimpleNamespace(parameters=parameter_objects)]
        )
    )


def test_run_success_with_mocked_frameworks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    parameters: dict[str, float],
) -> None:
    saved: list[tuple[dict[str, Any], Path]] = []
    monkeypatch.setattr(
        benchmark.Workspace, "load", lambda path: _fake_workspace(parameters)
    )
    monkeypatch.setattr(benchmark, "get_current_rss_mb", iter([100.0, 103.0]).__next__)
    monkeypatch.setattr(
        benchmark, "save_json", lambda data, output: saved.append((data, output))
    )

    def fake_measure(
        name: str, build_func: Any, eval_func: Any, mu_grid: list[float]
    ) -> dict[str, Any]:
        nll = [3.0, 1.0, 3.0]
        return {
            "framework": name,
            "status": "success",
            "n_points": len(mu_grid),
            "nll_values": nll,
            "delta_nll_shape": [2.0, 0.0, 2.0],
            "minimum_mu": 1.0,
            "model_build_time_seconds": 0.001,
            "full_scan_time_seconds": 0.003,
            "time_per_scan_point_seconds": 0.001,
            "current_rss_delta_mb": 1.0,
            "peak_rss_delta_mb": 2.0,
            "nll_summary": {"mean": 7.0 / 3.0, "std": 1.0, "min": 1.0, "max": 3.0},
        }

    monkeypatch.setattr(benchmark, "measure_framework_regression", fake_measure)
    plot_calls: list[tuple[list[dict[str, Any]], list[float], Path, float]] = []
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda results, mu_grid, plot_dir, delta_tolerance: plot_calls.append(
            (results, mu_grid, plot_dir, delta_tolerance)
        ),
    )

    output = tmp_path / "result.json"
    result = benchmark.run(
        workspace_path=workspace_path,
        mu_min=0.0,
        mu_max=2.0,
        n_points=3,
        output=output,
        plot=True,
        plot_dir=tmp_path / "plots",
        rtol=1e-9,
        atol=1e-9,
        delta_tolerance=1e-9,
        minimum_tolerance=1e-12,
        frameworks=["manual", "pyhs3"],
    )

    assert result["status"] == "success"
    assert result["successful_frameworks"] == ["manual", "pyhs3"]
    assert result["failed_frameworks"] == []
    assert result["rss_delta_mb"] == pytest.approx(3.0)
    assert saved == [(result, output)]
    assert len(plot_calls) == 1


def test_run_continues_after_framework_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    parameters: dict[str, float],
) -> None:
    monkeypatch.setattr(
        benchmark.Workspace, "load", lambda path: _fake_workspace(parameters)
    )
    monkeypatch.setattr(benchmark, "get_current_rss_mb", iter([100.0, 100.0]).__next__)
    monkeypatch.setattr(benchmark, "save_json", lambda data, output: None)

    def fake_measure(
        name: str, build_func: Any, eval_func: Any, mu_grid: list[float]
    ) -> dict[str, Any]:
        if name == "pyhs3":
            raise RuntimeError("framework boom")
        return {
            "framework": name,
            "status": "success",
            "n_points": len(mu_grid),
            "nll_values": [3.0, 1.0, 3.0],
            "delta_nll_shape": [2.0, 0.0, 2.0],
            "minimum_mu": 1.0,
            "model_build_time_seconds": 0.001,
            "full_scan_time_seconds": 0.003,
            "time_per_scan_point_seconds": 0.001,
            "current_rss_delta_mb": 0.0,
            "peak_rss_delta_mb": 0.0,
            "nll_summary": {"mean": 7.0 / 3.0, "std": 1.0, "min": 1.0, "max": 3.0},
        }

    monkeypatch.setattr(benchmark, "measure_framework_regression", fake_measure)

    result = benchmark.run(
        workspace_path=workspace_path,
        mu_min=0.0,
        mu_max=2.0,
        n_points=3,
        output=tmp_path / "result.json",
        plot=False,
        plot_dir=tmp_path / "plots",
        rtol=1e-9,
        atol=1e-9,
        delta_tolerance=1e-9,
        minimum_tolerance=1e-12,
        frameworks=["manual", "pyhs3"],
        continue_on_framework_error=True,
    )

    assert result["status"] == "failed"
    assert result["successful_frameworks"] == ["manual"]
    assert result["failed_frameworks"] == ["pyhs3"]
    failed = next(item for item in result["results"] if item["framework"] == "pyhs3")
    assert failed["validation_status"] == "not_run"
    assert failed["metrics"] is None


def test_run_fail_fast_wraps_framework_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    parameters: dict[str, float],
) -> None:
    saved: list[dict[str, Any]] = []
    monkeypatch.setattr(
        benchmark.Workspace, "load", lambda path: _fake_workspace(parameters)
    )
    monkeypatch.setattr(benchmark, "save_json", lambda data, output: saved.append(data))
    monkeypatch.setattr(
        benchmark,
        "measure_framework_regression",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(
        RuntimeError, match="Cross-framework numerical regression benchmark failed"
    ):
        benchmark.run(
            workspace_path=workspace_path,
            mu_min=0.0,
            mu_max=2.0,
            n_points=3,
            output=tmp_path / "result.json",
            plot=False,
            plot_dir=tmp_path / "plots",
            rtol=1e-9,
            atol=1e-9,
            delta_tolerance=1e-9,
            minimum_tolerance=1e-12,
            frameworks=["manual"],
            continue_on_framework_error=False,
        )

    assert saved[0]["status"] == "failed"
    assert saved[0]["error_type"] == "RuntimeError"


def test_run_failure_report_save_error_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_save_json(data: dict[str, Any], output: Path) -> None:
        raise OSError("cannot write")

    monkeypatch.setattr(benchmark, "save_json", failing_save_json)

    with pytest.raises(
        RuntimeError, match="Cross-framework numerical regression benchmark failed"
    ):
        benchmark.run(
            workspace_path=tmp_path / "missing.json",
            mu_min=0.0,
            mu_max=2.0,
            n_points=3,
            output=tmp_path / "result.json",
            plot=False,
            plot_dir=tmp_path / "plots",
            rtol=1e-9,
            atol=1e-9,
            delta_tolerance=1e-9,
            minimum_tolerance=1e-12,
            frameworks=["manual"],
        )

    assert "Failed to save benchmark failure report" in capsys.readouterr().err


def test_parse_args_custom_values(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cross_nll_regression.py",
            "--workspace",
            str(workspace_path),
            "--mu-min",
            "0.1",
            "--mu-max",
            "3.0",
            "--n-points",
            "7",
            "--output",
            str(tmp_path / "out.json"),
            "--plot",
            "--plot-dir",
            str(tmp_path / "plots"),
            "--rtol",
            "1e-6",
            "--atol",
            "1e-7",
            "--delta-tolerance",
            "1e-5",
            "--minimum-tolerance",
            "1e-4",
            "--frameworks",
            "manual",
            "pyhs3",
            "--fail-fast",
        ],
    )
    args = benchmark.parse_args()
    assert args.workspace == workspace_path
    assert args.mu_min == 0.1
    assert args.mu_max == 3.0
    assert args.n_points == 7
    assert args.output == tmp_path / "out.json"
    assert args.plot is True
    assert args.plot_dir == tmp_path / "plots"
    assert args.rtol == 1e-6
    assert args.atol == 1e-7
    assert args.delta_tolerance == 1e-5
    assert args.minimum_tolerance == 1e-4
    assert args.frameworks == ["manual", "pyhs3"]
    assert args.fail_fast is True


def test_parse_args_requires_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_cross_nll_regression.py"])
    with pytest.raises(SystemExit):
        benchmark.parse_args()


def test_main_passes_parsed_args(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cross_nll_regression.py",
            "--workspace",
            str(workspace_path),
            "--output",
            str(tmp_path / "out.json"),
            "--frameworks",
            "manual",
            "--fail-fast",
        ],
    )
    monkeypatch.setattr(benchmark, "run", lambda **kwargs: calls.append(kwargs))
    benchmark.main()
    assert calls[0]["workspace_path"] == workspace_path
    assert calls[0]["output"] == tmp_path / "out.json"
    assert calls[0]["frameworks"] == ["manual"]
    assert calls[0]["continue_on_framework_error"] is False
