from __future__ import annotations

import json
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

from src import run_cross_nll_scan as benchmark


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


def make_result(
    framework: str = "manual",
    *,
    status: str = "success",
    offset: float = 0.0,
    validation_status: str = "success",
    time_per_point: float = 0.001,
) -> dict[str, Any]:
    nll_values = [3.0 + offset, 1.0 + offset, 2.0 + offset]
    delta = [2.0, 0.0, 1.0]
    result = {
        "framework": framework,
        "plot_label": framework,
        "status": status,
    }
    if status != "success":
        result.update(
            {
                "error_type": "RuntimeError",
                "error_message": "boom",
                "traceback": "traceback",
            }
        )
        return result
    result.update(
        {
            "n_points": 3,
            "warmup_iterations": 1,
            "cold_first_nll": nll_values[0],
            "first_nll": nll_values[0],
            "nll_values": nll_values,
            "delta_nll_shape": delta,
            "minimum_mu": 0.5,
            "model_build_time_seconds": 0.001,
            "cold_first_evaluation_time_seconds": 0.002,
            "warmup_time_seconds": 0.003,
            "first_evaluation_time_seconds": 0.004,
            "full_scan_time_seconds": 0.03,
            "time_per_scan_point_seconds": time_per_point,
            "current_rss_before_mb": 100.0,
            "current_rss_after_mb": 101.0,
            "current_rss_delta_mb": 1.0,
            "peak_rss_before_mb": 120.0,
            "peak_rss_after_mb": 122.0,
            "peak_rss_delta_mb": 2.0,
            "rss_delta_mb": 1.0,
            "nll_summary": {
                "mean": 2.0 + offset,
                "std": 1.0,
                "min": 1.0 + offset,
                "max": 3.0 + offset,
            },
            "delta_nll_summary": {"mean": 1.0, "std": 1.0, "min": 0.0, "max": 2.0},
            "constant_offset_estimate": offset,
            "delta_nll_shape_max_abs_diff": 0.0,
            "minimum_mu_abs_diff": 0.0,
            "delta_nll_shape_success": True,
            "minimum_mu_success": True,
            "validation_status": validation_status,
        }
    )
    return result


@pytest.fixture
def successful_results() -> list[dict[str, Any]]:
    return [
        make_result("manual", time_per_point=0.002),
        make_result("pyhs3", offset=5.0, time_per_point=0.001),
    ]


class FakePdfModel:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.calls: list[tuple[str, float]] = []

    def pdf(self, name: str, **kwargs: Any) -> list[float]:
        self.calls.append((name, float(np.asarray(kwargs["mu"]))))
        index = int(name.split("_")[-1])
        return [self.values[index]]


class FakeMu:
    def __init__(self) -> None:
        self.value = 0.0

    def setVal(self, value: float) -> None:
        self.value = value


class FakePoisson:
    def __init__(self, value: float) -> None:
        self.value = value

    def getVal(self) -> float:
        return self.value


def test_validate_workspace_path_success(workspace_path: Path) -> None:
    assert benchmark.validate_workspace_path(workspace_path) == workspace_path


def test_validate_workspace_path_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.validate_workspace_path(tmp_path / "missing.json")


def test_validate_workspace_path_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace path is not a file"):
        benchmark.validate_workspace_path(tmp_path)


@pytest.mark.parametrize(
    ("value", "name", "minimum"),
    [(1, "n_points", 1), (0, "warmup_iterations", 0)],
)
def test_validate_positive_int_accepts_valid_values(
    value: int, name: str, minimum: int
) -> None:
    benchmark.validate_positive_int(value, name, minimum=minimum)


@pytest.mark.parametrize(
    ("value", "name", "minimum", "message"),
    [
        (0, "n_points", 1, "n_points must be at least 1"),
        (-1, "warmup_iterations", 0, "warmup_iterations must be at least 0"),
    ],
)
def test_validate_positive_int_rejects_invalid_values(
    value: int,
    name: str,
    minimum: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_positive_int(value, name, minimum=minimum)


@pytest.mark.parametrize("value", [0.0, 1.25, -2.0])
def test_validate_finite_float_accepts_finite_values(value: float) -> None:
    benchmark.validate_finite_float(value, "value")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_finite_float_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        benchmark.validate_finite_float(value, "value")


def valid_config(**overrides: Any) -> dict[str, Any]:
    config = {
        "mu_min": 0.0,
        "mu_max": 2.0,
        "n_points": 3,
        "warmup_iterations": 1,
        "shape_tolerance": 1e-9,
        "minimum_tolerance": 1e-12,
        "frameworks": ["manual"],
    }
    config.update(overrides)
    return config


def test_validate_benchmark_config_success() -> None:
    benchmark.validate_benchmark_config(**valid_config())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_points": 1}, "n_points must be at least 2"),
        ({"mu_min": float("nan")}, "mu_min must be finite"),
        ({"mu_max": float("inf")}, "mu_max must be finite"),
        ({"shape_tolerance": float("nan")}, "shape_tolerance must be finite"),
        ({"minimum_tolerance": float("nan")}, "minimum_tolerance must be finite"),
        ({"mu_min": 2.0, "mu_max": 2.0}, "mu_min must be smaller"),
        ({"shape_tolerance": 0.0}, "shape_tolerance must be positive"),
        ({"minimum_tolerance": 0.0}, "minimum_tolerance must be positive"),
        ({"frameworks": []}, "At least one framework"),
        ({"frameworks": ["manual", "bad"]}, "Unknown frameworks"),
        ({"frameworks": ["pyhs3"]}, "manual framework is required"),
    ],
)
def test_validate_benchmark_config_rejects_invalid_values(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_benchmark_config(**valid_config(**overrides))


def test_validate_parameters_success(parameters: dict[str, float]) -> None:
    benchmark.validate_parameters(parameters, 2)


@pytest.mark.parametrize(
    ("updates", "message", "error_type"),
    [
        ({"signal_1": None}, "missing parameters", KeyError),
        ({"background_0": float("nan")}, "must be finite", ValueError),
        ({"obs_1": -1.0}, "must be non-negative", ValueError),
    ],
)
def test_validate_parameters_rejects_bad_inputs(
    parameters: dict[str, float],
    updates: dict[str, float | None],
    message: str,
    error_type: type[Exception],
) -> None:
    modified = dict(parameters)
    for key, value in updates.items():
        if value is None:
            modified.pop(key)
        else:
            modified[key] = value

    with pytest.raises(error_type, match=message):
        benchmark.validate_parameters(modified, 2)


@pytest.mark.parametrize("values", [[1.0], [0.0, 2.0]])
def test_validate_scan_values_success(values: list[float]) -> None:
    benchmark.validate_scan_values(values, "values")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "must not be empty"),
        ([1.0, float("nan")], "contains non-finite"),
        ([float("inf")], "contains non-finite"),
    ],
)
def test_validate_scan_values_rejects_invalid_values(
    values: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_scan_values(values, "values")


def test_extract_parameters_success(parameters: dict[str, float]) -> None:
    fake_parameters = [
        SimpleNamespace(name=name, value=value) for name, value in parameters.items()
    ]
    workspace = SimpleNamespace(
        parameter_points=SimpleNamespace(
            root=[SimpleNamespace(parameters=fake_parameters)]
        )
    )

    assert benchmark.extract_parameters(workspace) == parameters


@pytest.mark.parametrize(
    "workspace",
    [
        SimpleNamespace(),
        SimpleNamespace(parameter_points=SimpleNamespace(root=[])),
        SimpleNamespace(
            parameter_points=SimpleNamespace(
                root=[
                    SimpleNamespace(parameters=[SimpleNamespace(name="x", value="bad")])
                ]
            )
        ),
    ],
)
def test_extract_parameters_rejects_malformed_workspace(workspace: Any) -> None:
    with pytest.raises(ValueError, match="Could not extract initial parameters"):
        benchmark.extract_parameters(workspace)


def test_extract_parameters_rejects_empty_parameter_set() -> None:
    workspace = SimpleNamespace(
        parameter_points=SimpleNamespace(root=[SimpleNamespace(parameters=[])])
    )

    with pytest.raises(ValueError, match="does not contain parameters"):
        benchmark.extract_parameters(workspace)


def test_infer_n_bins_from_parameters_success(parameters: dict[str, float]) -> None:
    assert benchmark.infer_n_bins_from_parameters(parameters) == 2


@pytest.mark.parametrize(
    ("bad_parameters", "message"),
    [
        ({"mu": 1.0}, "no signal"),
        ({"mu": 1.0, "signal_1": 2.0, "background_1": 1.0, "obs_1": 1.0}, "contiguous"),
        ({"mu": 1.0, "signal_0": 2.0, "obs_0": 1.0}, "parameters are missing"),
    ],
)
def test_infer_n_bins_from_parameters_rejects_invalid_shapes(
    bad_parameters: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.infer_n_bins_from_parameters(bad_parameters)


def test_poisson_nll_matches_formula() -> None:
    expected = 12.0 - 10.0 * math.log(12.0) + math.lgamma(11.0)
    assert benchmark.poisson_nll(10.0, 12.0) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("observed", "expected", "message"),
    [
        (1.0, 0.0, "expected value must be positive"),
        (-1.0, 1.0, "observed value must be non-negative"),
    ],
)
def test_poisson_nll_rejects_invalid_values(
    observed: float, expected: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.poisson_nll(observed, expected)


def test_get_vectors_and_manual_model(parameters: dict[str, float]) -> None:
    vectors = benchmark.get_vectors(parameters, 2)
    assert vectors == ([2.0, 3.0], [10.0, 11.0], [12.0, 14.0])
    assert benchmark.build_manual_model(parameters, 2) == vectors


def test_manual_nll_sums_poisson_terms() -> None:
    model = ([2.0], [10.0], [12.0])
    assert benchmark.manual_nll(model, 1.0) == pytest.approx(
        benchmark.poisson_nll(12.0, 12.0)
    )


def test_pyhs3_nll_success() -> None:
    model = FakePdfModel([0.25, 0.5])

    result = benchmark.pyhs3_nll(model, n_bins=2, mu_value=1.5)

    assert result == pytest.approx(-(math.log(0.25) + math.log(0.5)))
    assert model.calls == [("poisson_0", 1.5), ("poisson_1", 1.5)]


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan")])
def test_pyhs3_nll_rejects_invalid_pdf_values(bad_value: float) -> None:
    with pytest.raises(ValueError, match="PyHS3 returned invalid PDF value"):
        benchmark.pyhs3_nll(FakePdfModel([bad_value]), n_bins=1, mu_value=1.0)


def test_build_pyhf_spec_uses_expected_structure(parameters: dict[str, float]) -> None:
    spec = benchmark.build_pyhf_spec(parameters, 2)

    assert spec["version"] == "1.0.0"
    assert spec["channels"][0]["name"] == "channel"
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
        benchmark.build_pyhf_model(parameters, 2)


def test_build_pyhf_model_success_with_fake_pyhf(
    monkeypatch: pytest.MonkeyPatch, parameters: dict[str, float]
) -> None:
    class FakeWorkspace:
        def __init__(self, spec: dict[str, Any]) -> None:
            self.spec = spec

        def model(self) -> str:
            return "model"

        def data(self, model: str) -> list[float]:
            assert model == "model"
            return [1.0]

    fake_pyhf = SimpleNamespace(Workspace=FakeWorkspace)
    monkeypatch.setattr(benchmark, "pyhf", fake_pyhf)

    assert benchmark.build_pyhf_model(parameters, 2) == ("model", [1.0])


def test_pyhf_nll_success() -> None:
    class FakeConfig:
        par_order = ["alpha", "mu"]

        def suggested_init(self) -> list[float]:
            return [0.0, 1.0]

    class FakeModel:
        config = FakeConfig()

        def logpdf(self, pars: list[float], data: list[float]) -> list[float]:
            assert pars == [0.0, 2.5]
            assert data == [1.0]
            return [-3.0]

    assert benchmark.pyhf_nll((FakeModel(), [1.0]), 2.5) == pytest.approx(3.0)


def test_pyhf_nll_rejects_non_finite_value() -> None:
    class FakeConfig:
        par_order = ["mu"]

        def suggested_init(self) -> list[float]:
            return [1.0]

    class FakeModel:
        config = FakeConfig()

        def logpdf(self, pars: list[float], data: list[float]) -> float:
            return float("nan")

    with pytest.raises(ValueError, match="pyhf returned non-finite"):
        benchmark.pyhf_nll((FakeModel(), []), 1.0)


def test_build_roofit_model_rejects_missing_root(
    monkeypatch: pytest.MonkeyPatch, parameters: dict[str, float]
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", None)

    with pytest.raises(RuntimeError, match="ROOT is not available"):
        benchmark.build_roofit_model(parameters, 1, 0.0)


def test_roofit_nll_success() -> None:
    model = {"mu": FakeMu(), "poissons": [FakePoisson(0.25), FakePoisson(0.5)]}

    result = benchmark.roofit_nll(model, 1.75)

    assert model["mu"].value == 1.75
    assert result == pytest.approx(-(math.log(0.25) + math.log(0.5)))


@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan")])
def test_roofit_nll_rejects_invalid_pdf_values(bad_value: float) -> None:
    model = {"mu": FakeMu(), "poissons": [FakePoisson(bad_value)]}

    with pytest.raises(ValueError, match="RooFit returned invalid PDF value"):
        benchmark.roofit_nll(model, 1.0)


def test_summarize_values_single_and_multiple() -> None:
    assert benchmark.summarize_values([2.0]) == {
        "mean": 2.0,
        "std": 0.0,
        "min": 2.0,
        "max": 2.0,
    }
    multiple = benchmark.summarize_values([1.0, 2.0, 3.0])
    assert multiple["mean"] == pytest.approx(2.0)
    assert multiple["std"] == pytest.approx(1.0)
    assert multiple["min"] == 1.0
    assert multiple["max"] == 3.0


def test_build_mu_grid_success() -> None:
    assert benchmark.build_mu_grid(0.0, 1.0, 3) == [0.0, 0.5, 1.0]


def test_build_mu_grid_rejects_invalid_grid() -> None:
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        benchmark.build_mu_grid(0.0, 1.0, 1)


def test_run_scan_success() -> None:
    values = benchmark.run_scan("model", lambda model, mu: mu + 1.0, [0.0, 1.0])
    assert values == [1.0, 2.0]


def test_run_scan_rejects_non_finite_outputs() -> None:
    with pytest.raises(ValueError, match="nll_values contains non-finite"):
        benchmark.run_scan("model", lambda model, mu: float("nan"), [1.0])


def test_minimum_position_success() -> None:
    assert benchmark.minimum_position([0.0, 1.0, 2.0], [3.0, 1.0, 2.0]) == 1.0


def test_minimum_position_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        benchmark.minimum_position([0.0], [1.0, 2.0])


def test_delta_nll_shape_success() -> None:
    assert benchmark.delta_nll_shape([3.0, 1.0, 2.0]) == [2.0, 0.0, 1.0]


def test_max_abs_difference_and_mean_offset_success() -> None:
    assert benchmark.max_abs_difference([1.0, 2.0], [1.5, 1.0]) == pytest.approx(1.0)
    assert benchmark.mean_offset([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)


@pytest.mark.parametrize("func", [benchmark.max_abs_difference, benchmark.mean_offset])
def test_comparison_helpers_reject_length_mismatch(func: Any) -> None:
    with pytest.raises(ValueError, match="different lengths"):
        func([1.0], [1.0, 2.0])


def test_measure_framework_scan_success(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([0.0, 0.1, 0.1, 0.2, 0.2, 0.25, 0.25, 0.3, 0.3, 0.6])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    current_rss = iter([100.0, 105.0])
    peak_rss = iter([120.0, 130.0])
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: next(current_rss))
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: next(peak_rss))

    calls: list[float] = []

    def eval_func(model: str, mu: float) -> float:
        assert model == "model"
        calls.append(mu)
        return (mu - 1.0) ** 2

    result = benchmark.measure_framework_scan(
        name="manual",
        build_func=lambda: "model",
        eval_func=eval_func,
        mu_grid=[0.0, 1.0, 2.0],
        warmup_iterations=2,
    )

    assert result["framework"] == "manual"
    assert result["status"] == "success"
    assert result["model_build_time_seconds"] == pytest.approx(0.1)
    assert result["cold_first_evaluation_time_seconds"] == pytest.approx(0.1)
    assert result["warmup_time_seconds"] == pytest.approx(0.05)
    assert result["first_evaluation_time_seconds"] == pytest.approx(0.05)
    assert result["full_scan_time_seconds"] == pytest.approx(0.3)
    assert result["time_per_scan_point_seconds"] == pytest.approx(0.1)
    assert result["current_rss_delta_mb"] == 5.0
    assert result["peak_rss_delta_mb"] == 10.0
    assert result["minimum_mu"] == 1.0
    assert result["delta_nll_shape"] == [1.0, 0.0, 1.0]
    assert calls == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("first_nll", float("nan"), "field first_nll is not finite"),
        (
            "model_build_time_seconds",
            -1.0,
            "model_build_time_seconds must be non-negative",
        ),
        (
            "cold_first_evaluation_time_seconds",
            -1.0,
            "cold_first_evaluation_time_seconds must be non-negative",
        ),
        ("warmup_time_seconds", -1.0, "warmup_time_seconds must be non-negative"),
        (
            "first_evaluation_time_seconds",
            -1.0,
            "first_evaluation_time_seconds must be non-negative",
        ),
        ("full_scan_time_seconds", 0.0, "full_scan_time_seconds must be positive"),
    ],
)
def test_validate_framework_result_rejects_invalid_fields(
    field: str, value: float, message: str
) -> None:
    result = make_result("manual")
    result[field] = value

    with pytest.raises(ValueError, match=message):
        benchmark.validate_framework_result(result)


def test_add_scan_validation_success_and_failed_result() -> None:
    results = [
        make_result("manual"),
        make_result("pyhs3", offset=5.0),
        make_result("pyhf", status="failed"),
    ]

    benchmark.add_scan_validation(
        results, shape_tolerance=1e-9, minimum_tolerance=1e-12
    )

    assert results[0]["validation_status"] == "success"
    assert results[1]["constant_offset_estimate"] == pytest.approx(5.0)
    assert results[1]["delta_nll_shape_max_abs_diff"] == pytest.approx(0.0)
    assert results[1]["validation_status"] == "success"
    assert results[2]["validation_status"] == "not_run"
    assert results[2]["delta_nll_shape_success"] is False


def test_add_scan_validation_marks_numerical_failure() -> None:
    manual = make_result("manual")
    shifted = make_result("pyhs3")
    shifted["delta_nll_shape"] = [2.1, 0.0, 1.0]
    shifted["minimum_mu"] = 1.5
    results = [manual, shifted]

    benchmark.add_scan_validation(results, shape_tolerance=0.01, minimum_tolerance=0.1)

    assert shifted["delta_nll_shape_success"] is False
    assert shifted["minimum_mu_success"] is False
    assert shifted["validation_status"] == "failed"


@pytest.mark.parametrize("results", [[], [make_result("pyhs3")]])
def test_add_scan_validation_rejects_missing_manual_reference(
    results: list[dict[str, Any]],
) -> None:
    with pytest.raises(ValueError, match="manual result|empty benchmark"):
        benchmark.add_scan_validation(results, 1e-9, 1e-12)


def test_failed_framework_result() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        result = benchmark.failed_framework_result("pyhf", exc)

    assert result["framework"] == "pyhf"
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "boom"
    assert "RuntimeError" in result["traceback"]


def test_make_framework_specs_builds_requested_specs(
    monkeypatch: pytest.MonkeyPatch,
    parameters: dict[str, float],
    workspace_path: Path,
) -> None:
    monkeypatch.setattr(benchmark, "build_pyhs3_model", lambda path: ("pyhs3", path))
    monkeypatch.setattr(
        benchmark, "build_pyhf_model", lambda params, n_bins: ("pyhf", n_bins)
    )
    monkeypatch.setattr(
        benchmark, "build_roofit_model", lambda params, n_bins, mu: {"roofit": mu}
    )
    monkeypatch.setattr(benchmark, "pyhs3_nll", lambda model, n_bins, mu: 10.0 + mu)
    monkeypatch.setattr(benchmark, "pyhf_nll", lambda model, mu: 20.0 + mu)
    monkeypatch.setattr(benchmark, "roofit_nll", lambda model, mu: 30.0 + mu)

    specs = benchmark.make_framework_specs(
        ["manual", "pyhs3", "pyhf", "roofit"],
        parameters,
        workspace_path,
        n_bins=2,
        mu_min=0.5,
    )

    assert [spec.name for spec in specs] == ["manual", "pyhs3", "pyhf", "roofit"]
    assert specs[0].eval_func(specs[0].build_func(), 1.0) == pytest.approx(
        benchmark.manual_nll(benchmark.build_manual_model(parameters, 2), 1.0)
    )
    assert specs[1].build_func() == ("pyhs3", workspace_path)
    assert specs[1].eval_func("model", 1.0) == 11.0
    assert specs[2].build_func() == ("pyhf", 2)
    assert specs[2].eval_func("model", 1.0) == 21.0
    assert specs[3].build_func() == {"roofit": 0.5}
    assert specs[3].eval_func("model", 1.0) == 31.0


def test_make_framework_specs_rejects_unknown_framework(
    parameters: dict[str, float], workspace_path: Path
) -> None:
    with pytest.raises(ValueError, match="Unknown framework"):
        benchmark.make_framework_specs(["unknown"], parameters, workspace_path, 2, 0.0)


def test_framework_order_returns_successful_frameworks_only() -> None:
    assert benchmark._framework_order(
        [
            make_result("manual"),
            make_result("pyhf", status="failed"),
            make_result("pyhs3"),
        ]
    ) == ["manual", "pyhs3"]


def test_style_helpers_and_format_metric() -> None:
    assert benchmark._framework_label("manual") == "Manual reference"
    assert benchmark._style_for("unknown")["label"] == "unknown"
    assert benchmark._format_metric(float("nan")) == "nan"
    assert benchmark._format_metric(0.0, " ms") == "0 ms"
    assert benchmark._format_metric(1e-4, " s") == "1.00e-04 s"
    assert benchmark._format_metric(1.23456, " ms") == "1.235 ms"
    assert benchmark._format_metric(12.3456, " ms") == "12.35 ms"
    assert benchmark._format_metric(123.456, " ms") == "123.5 ms"


def test_successful_results_filters_failed(
    successful_results: list[dict[str, Any]],
) -> None:
    results = successful_results + [make_result("pyhf", status="failed")]
    assert [
        result["framework"] for result in benchmark._successful_results(results)
    ] == ["manual", "pyhs3"]


def test_reference_result_success(successful_results: list[dict[str, Any]]) -> None:
    assert benchmark._reference_result(successful_results)["framework"] == "manual"


def test_reference_result_rejects_missing_manual() -> None:
    with pytest.raises(ValueError, match="Manual reference result"):
        benchmark._reference_result([make_result("pyhs3")])


def test_save_figure_creates_png(tmp_path: Path) -> None:
    fig, _ax = plt.subplots()
    output_path = tmp_path / "plot_without_suffix"

    benchmark._save_figure(fig, output_path)

    assert (tmp_path / "plot_without_suffix.png").exists()


def test_save_figure_wraps_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fig, _ax = plt.subplots()

    def failing_savefig(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(fig, "savefig", failing_savefig)

    with pytest.raises(OSError, match="Failed to save plot"):
        benchmark._save_figure(fig, tmp_path / "plot.png")


@pytest.mark.parametrize(
    "plot_func,filename,args",
    [
        (benchmark.make_nll_profile_plot, "profile.png", ([0.0, 0.5, 1.0],)),
        (benchmark.make_timing_profile_plot, "timing.png", ()),
        (benchmark.make_relative_runtime_plot, "relative.png", ()),
        (benchmark.make_memory_profile_plot, "memory.png", ()),
        (benchmark.make_numerical_agreement_plot, "agreement.png", (1e-9,)),
        (benchmark.make_summary_table_plot, "summary.png", ()),
    ],
)
def test_individual_plot_functions_create_png(
    tmp_path: Path,
    successful_results: list[dict[str, Any]],
    plot_func: Any,
    filename: str,
    args: tuple[Any, ...],
) -> None:
    output_path = tmp_path / filename

    if plot_func is benchmark.make_nll_profile_plot:
        plot_func(successful_results, args[0], output_path)
    elif plot_func is benchmark.make_numerical_agreement_plot:
        plot_func(successful_results, args[0], output_path)
    else:
        plot_func(successful_results, output_path)

    assert output_path.exists()


def test_make_plots_calls_all_plot_builders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    successful_results: list[dict[str, Any]],
) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        benchmark, "make_nll_profile_plot", lambda **kwargs: calls.append("profile")
    )
    monkeypatch.setattr(
        benchmark, "make_timing_profile_plot", lambda **kwargs: calls.append("timing")
    )
    monkeypatch.setattr(
        benchmark,
        "make_relative_runtime_plot",
        lambda **kwargs: calls.append("relative"),
    )
    monkeypatch.setattr(
        benchmark, "make_memory_profile_plot", lambda **kwargs: calls.append("memory")
    )
    monkeypatch.setattr(
        benchmark,
        "make_numerical_agreement_plot",
        lambda **kwargs: calls.append("agreement"),
    )
    monkeypatch.setattr(
        benchmark, "make_summary_table_plot", lambda **kwargs: calls.append("summary")
    )

    benchmark.make_plots(
        successful_results, [0.0, 0.5, 1.0], tmp_path, shape_tolerance=1e-6
    )

    assert calls == ["profile", "timing", "relative", "memory", "agreement", "summary"]
    assert tmp_path.exists()


def test_make_plots_rejects_no_successful_results(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No successful benchmark results"):
        benchmark.make_plots(
            [make_result("pyhf", status="failed")], [0.0, 1.0], tmp_path
        )


def test_print_result_success_and_failure(capsys: pytest.CaptureFixture[str]) -> None:
    benchmark.print_result(make_result("manual"))
    benchmark.print_result(make_result("pyhf", status="failed"))

    output = capsys.readouterr().out
    assert "manual" in output
    assert "validation:" in output
    assert "pyhf" in output
    assert "error:" in output


def test_build_failed_output(workspace_path: Path) -> None:
    try:
        raise RuntimeError("top-level boom")
    except RuntimeError as exc:
        output = benchmark.build_failed_output(
            workspace_path=workspace_path,
            n_bins=2,
            mu_min=0.0,
            mu_max=2.0,
            n_points=3,
            warmup_iterations=1,
            shape_tolerance=1e-9,
            minimum_tolerance=1e-12,
            frameworks=["manual"],
            exc=exc,
        )

    assert output["benchmark"] == benchmark.BENCHMARK_NAME
    assert output["workspace"] == workspace_path.name
    assert output["n_bins"] == 2
    assert output["status"] == "failed"
    assert output["error_type"] == "RuntimeError"
    assert output["results"] == []


def fake_workspace(parameters: dict[str, float]) -> Any:
    fake_parameters = [
        SimpleNamespace(name=name, value=value) for name, value in parameters.items()
    ]
    return SimpleNamespace(
        parameter_points=SimpleNamespace(
            root=[SimpleNamespace(parameters=fake_parameters)]
        )
    )


def test_run_success_manual_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    parameters: dict[str, float],
) -> None:
    saved_payloads: list[dict[str, Any]] = []
    plot_calls: list[tuple[list[dict[str, Any]], list[float], Path]] = []

    monkeypatch.setattr(
        benchmark.Workspace, "load", lambda path: fake_workspace(parameters)
    )
    monkeypatch.setattr(
        benchmark,
        "measure_framework_scan",
        lambda **kwargs: make_result(kwargs["name"]),
    )
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, output: (
            saved_payloads.append(payload)
            or output.parent.mkdir(parents=True, exist_ok=True)
            or output.write_text(json.dumps(payload))
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda results, mu_grid, plot_dir, shape_tolerance: plot_calls.append(
            (results, mu_grid, plot_dir)
        ),
    )

    output_path = tmp_path / "result.json"
    plot_dir = tmp_path / "plots"
    result = benchmark.run(
        workspace_path=workspace_path,
        mu_min=0.0,
        mu_max=1.0,
        n_points=3,
        output=output_path,
        plot=True,
        plot_dir=plot_dir,
        shape_tolerance=1e-9,
        minimum_tolerance=1e-12,
        frameworks=["manual"],
        warmup_iterations=0,
    )

    assert result["status"] == "success"
    assert result["n_bins"] == 2
    assert result["mu_grid"] == [0.0, 0.5, 1.0]
    assert result["successful_frameworks"] == ["manual"]
    assert result["failed_frameworks"] == []
    assert saved_payloads[0] == result
    assert len(plot_calls) == 1


def test_run_continues_after_framework_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    parameters: dict[str, float],
) -> None:
    def fake_measure(**kwargs: Any) -> dict[str, Any]:
        if kwargs["name"] == "pyhs3":
            raise RuntimeError("framework failed")
        return make_result(kwargs["name"])

    monkeypatch.setattr(
        benchmark.Workspace, "load", lambda path: fake_workspace(parameters)
    )
    monkeypatch.setattr(benchmark, "measure_framework_scan", fake_measure)
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, output: (
            output.parent.mkdir(parents=True, exist_ok=True)
            or output.write_text(json.dumps(payload))
        ),
    )

    result = benchmark.run(
        workspace_path=workspace_path,
        mu_min=0.0,
        mu_max=1.0,
        n_points=3,
        output=tmp_path / "result.json",
        plot=False,
        plot_dir=tmp_path / "plots",
        shape_tolerance=1e-9,
        minimum_tolerance=1e-12,
        frameworks=["manual", "pyhs3"],
        continue_on_framework_error=True,
        warmup_iterations=0,
    )

    assert result["status"] == "failed"
    assert result["successful_frameworks"] == ["manual"]
    assert result["failed_frameworks"] == ["pyhs3"]
    failed = result["results"][1]
    assert failed["status"] == "failed"
    assert failed["validation_status"] == "not_run"


def test_run_fail_fast_wraps_framework_failure_and_writes_failed_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    parameters: dict[str, float],
) -> None:
    saved_payloads: list[dict[str, Any]] = []

    monkeypatch.setattr(
        benchmark.Workspace, "load", lambda path: fake_workspace(parameters)
    )

    def fake_measure(**kwargs: Any) -> dict[str, Any]:
        if kwargs["name"] == "pyhs3":
            raise RuntimeError("framework failed")
        return make_result(kwargs["name"])

    monkeypatch.setattr(benchmark, "measure_framework_scan", fake_measure)
    monkeypatch.setattr(
        benchmark, "save_json", lambda payload, output: saved_payloads.append(payload)
    )

    with pytest.raises(RuntimeError, match="Cross-framework NLL scan benchmark failed"):
        benchmark.run(
            workspace_path=workspace_path,
            mu_min=0.0,
            mu_max=1.0,
            n_points=3,
            output=tmp_path / "result.json",
            plot=False,
            plot_dir=tmp_path / "plots",
            shape_tolerance=1e-9,
            minimum_tolerance=1e-12,
            frameworks=["manual", "pyhs3"],
            continue_on_framework_error=False,
            warmup_iterations=0,
        )

    assert saved_payloads[-1]["status"] == "failed"
    assert saved_payloads[-1]["error_type"] == "RuntimeError"


def test_run_failure_report_save_error_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_save_json(payload: dict[str, Any], output: Path) -> None:
        raise OSError("cannot save")

    monkeypatch.setattr(benchmark, "save_json", failing_save_json)

    with pytest.raises(RuntimeError, match="Cross-framework NLL scan benchmark failed"):
        benchmark.run(
            workspace_path=tmp_path / "missing.json",
            mu_min=0.0,
            mu_max=1.0,
            n_points=3,
            output=tmp_path / "result.json",
            plot=False,
            plot_dir=tmp_path / "plots",
            shape_tolerance=1e-9,
            minimum_tolerance=1e-12,
            frameworks=["manual"],
        )

    assert "Failed to save benchmark failure report" in capsys.readouterr().err


def test_parse_args_custom_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cross_nll_scan.py",
            "--workspace",
            str(tmp_path / "workspace.json"),
            "--mu-min",
            "0.1",
            "--mu-max",
            "1.5",
            "--n-points",
            "7",
            "--warmup-iterations",
            "3",
            "--output",
            str(tmp_path / "out.json"),
            "--plot",
            "--plot-dir",
            str(tmp_path / "plots"),
            "--shape-tolerance",
            "1e-6",
            "--minimum-tolerance",
            "1e-5",
            "--frameworks",
            "manual",
            "pyhs3",
            "--fail-fast",
        ],
    )

    args = benchmark.parse_args()

    assert args.workspace == tmp_path / "workspace.json"
    assert args.mu_min == 0.1
    assert args.mu_max == 1.5
    assert args.n_points == 7
    assert args.warmup_iterations == 3
    assert args.output == tmp_path / "out.json"
    assert args.plot is True
    assert args.plot_dir == tmp_path / "plots"
    assert args.shape_tolerance == pytest.approx(1e-6)
    assert args.minimum_tolerance == pytest.approx(1e-5)
    assert args.frameworks == ["manual", "pyhs3"]
    assert args.fail_fast is True


def test_parse_args_requires_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_cross_nll_scan.py"])

    with pytest.raises(SystemExit):
        benchmark.parse_args()


def test_main_passes_parsed_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    args = SimpleNamespace(
        workspace=tmp_path / "workspace.json",
        mu_min=0.0,
        mu_max=1.0,
        n_points=3,
        output=tmp_path / "result.json",
        plot=True,
        plot_dir=tmp_path / "plots",
        shape_tolerance=1e-9,
        minimum_tolerance=1e-12,
        frameworks=["manual"],
        fail_fast=True,
        warmup_iterations=2,
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(benchmark, "run", lambda **kwargs: calls.append(kwargs))

    benchmark.main()

    assert calls == [
        {
            "workspace_path": args.workspace,
            "mu_min": 0.0,
            "mu_max": 1.0,
            "n_points": 3,
            "output": args.output,
            "plot": True,
            "plot_dir": args.plot_dir,
            "shape_tolerance": 1e-9,
            "minimum_tolerance": 1e-12,
            "frameworks": ["manual"],
            "continue_on_framework_error": False,
            "warmup_iterations": 2,
        }
    ]


class FakeRooRealVar:
    def __init__(self, name: str, title: str, value: float, *args: float) -> None:
        self.name = name
        self.title = title
        self.value = value
        self.args = args
        self.constant = False

    def setConstant(self, value: bool) -> None:
        self.constant = value

    def setVal(self, value: float) -> None:
        self.value = value


class FakeRooArgList(list):
    def __init__(self, *args: Any) -> None:
        super().__init__(args)

    def add(self, item: Any) -> None:
        self.append(item)


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


def test_build_pyhs3_model_loads_workspace(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    class FakeWorkspaceForLoad:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool, str]] = []

        def model(self, target: str, progress: bool, mode: str) -> str:
            self.calls.append((target, progress, mode))
            return "pyhs3-model"

    fake_workspace = FakeWorkspaceForLoad()
    monkeypatch.setattr(benchmark.Workspace, "load", lambda path: fake_workspace)

    assert benchmark.build_pyhs3_model(workspace_path) == "pyhs3-model"
    assert fake_workspace.calls == [("analysis", False, "FAST_RUN")]


def test_compiled_pyhs3_model_matches_manual_nll(
    workspace_path: Path, parameters: dict[str, float]
) -> None:
    model = benchmark.build_pyhs3_compiled_model(workspace_path, parameters)

    assert model.description.startswith("JAX-compiled")
    assert benchmark.pyhs3_compiled_nll(model, 1.0) == pytest.approx(
        benchmark.manual_nll(benchmark.build_manual_model(parameters, 2), 1.0)
    )


def test_pyhs3_compiled_nll_rejects_non_finite_output() -> None:
    model = benchmark.CompiledPyHS3Model(
        compiled_nll=lambda _mu: np.asarray(float("nan")),
        description="fake",
    )

    with pytest.raises(ValueError, match="Compiled PyHS3 NLL is not finite"):
        benchmark.pyhs3_compiled_nll(model, 1.0)


def test_build_roofit_model_success(
    monkeypatch: pytest.MonkeyPatch, parameters: dict[str, float]
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)

    model = benchmark.build_roofit_model(parameters, n_bins=2, mu_value=1.0)

    assert isinstance(model["mu"], FakeRooRealVar)
    assert len(model["poissons"]) == 2
    assert isinstance(model["likelihood"], FakeRooProdPdf)
    assert len(model["keepalive"]) >= 1 + 3 * 2 + 1
    assert benchmark.roofit_nll(model, 1.5) == pytest.approx(-2.0 * math.log(0.5))


def test_make_framework_specs_includes_pyhs3_compiled(
    monkeypatch: pytest.MonkeyPatch,
    parameters: dict[str, float],
    workspace_path: Path,
) -> None:
    compiled_model = benchmark.CompiledPyHS3Model(
        compiled_nll=lambda mu: np.asarray(float(mu) + 10.0),
        description="fake",
    )
    monkeypatch.setattr(
        benchmark,
        "build_pyhs3_compiled_model",
        lambda path, params: compiled_model,
    )

    specs = benchmark.make_framework_specs(
        ["manual", "pyhs3_compiled"],
        parameters,
        workspace_path,
        n_bins=2,
        mu_min=0.0,
    )

    assert [spec.name for spec in specs] == ["manual", "pyhs3_compiled"]
    assert specs[1].build_func() is compiled_model
    assert specs[1].eval_func(compiled_model, 2.0) == pytest.approx(12.0)
