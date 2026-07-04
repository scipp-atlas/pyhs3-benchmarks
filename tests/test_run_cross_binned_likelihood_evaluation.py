from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from src import run_cross_binned_likelihood_evaluation as benchmark


class FakeParameter:
    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value


class FakeParameterSet:
    def __init__(self, parameters: list[FakeParameter]) -> None:
        self.parameters = parameters


class FakePyHS3Model:
    def __init__(self, value: float = 0.5) -> None:
        self.value = value
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def pdf(self, name: str, **params: Any) -> np.ndarray:
        self.calls.append((name, params))
        return np.asarray(self.value)


class FakeWorkspace:
    def __init__(
        self, parameters: dict[str, Any] | None = None, model: Any | None = None
    ) -> None:
        params = make_parameters(2) if parameters is None else parameters
        self.parameter_points = SimpleNamespace(
            root=[
                FakeParameterSet(
                    [FakeParameter(name, value) for name, value in params.items()]
                )
            ]
        )
        self.model_obj = model or FakePyHS3Model()

    def model(self, analysis: str, progress: bool, mode: str) -> Any:
        assert analysis == "analysis"
        assert progress is False
        assert mode == "FAST_RUN"
        return self.model_obj


class FakePyhfConfig:
    par_order = ["mu"]

    @staticmethod
    def suggested_init() -> list[float]:
        return [1.0]


class FakePyhfModel:
    def __init__(self, value: float = -3.0, has_mu: bool = True) -> None:
        self.config = FakePyhfConfig()
        if not has_mu:
            self.config = SimpleNamespace(
                par_order=["theta"], suggested_init=lambda: [0.0]
            )
        self.value = value
        self.calls: list[tuple[list[float], Any]] = []

    def logpdf(self, pars: list[float], data: Any) -> np.ndarray:
        self.calls.append((pars, data))
        return np.asarray(self.value)


class FakePyhfWorkspace:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.spec = spec
        self.model_obj = FakePyhfModel()
        self.data_obj = ["data"]

    def model(self) -> FakePyhfModel:
        return self.model_obj

    def data(self, model: FakePyhfModel) -> list[str]:
        assert model is self.model_obj
        return self.data_obj


class FakeRooRealVar:
    def __init__(self, name: str, title: str, value: float, *bounds: float) -> None:
        self.name = name
        self.value = value
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


class FakeRooArgSet(list):
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
    RooArgSet = FakeRooArgSet
    RooFormulaVar = FakeRooFormulaVar
    RooPoisson = FakeRooPoisson
    RooProdPdf = FakeRooProdPdf


def make_parameters(n_bins: int = 2) -> dict[str, float]:
    params: dict[str, float] = {}
    for index in range(n_bins):
        params[f"signal_{index}"] = float(index + 1)
        params[f"background_{index}"] = float(index + 10)
        params[f"obs_{index}"] = float(index + 12)
    return params


@pytest.fixture
def workspace_file(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


@pytest.fixture
def valid_config(workspace_file: Path, tmp_path: Path) -> benchmark.BenchmarkConfig:
    return benchmark.BenchmarkConfig(
        workspace_path=workspace_file,
        frameworks=["manual", "pyhs3", "pyhf"],
        n_bins=None,
        mu=1.0,
        delta_reference_mu=0.0,
        n_runs=2,
        warmup_iterations=1,
        raw_tolerance=1e-9,
        delta_tolerance=1e-9,
        output_dir=tmp_path / "out",
        output_name="result.json",
        plot=False,
        plot_dir=tmp_path / "plots",
        fail_fast=False,
    )


@pytest.fixture
def success_result() -> dict[str, Any]:
    return {
        "framework": "manual",
        "framework_label": "Manual",
        "status": "success",
        "validation_status": "success",
        "raw_nll": 10.0,
        "warm_nll": 10.0,
        "reference_nll": 9.0,
        "delta_nll": 1.0,
        "raw_nll_abs_diff": 0.0,
        "delta_nll_abs_diff": 0.0,
        "raw_nll_success": True,
        "delta_nll_success": True,
        "input_load_time_seconds": 0.0005,
        "model_build_time_seconds": 0.001,
        "first_evaluation_time_seconds": 0.002,
        "warmup_iterations": 1,
        "warmup_time_seconds": 0.0015,
        "warm_evaluation": {
            "mean_seconds": 0.003,
            "std_seconds": 0.0001,
            "min_seconds": 0.002,
            "max_seconds": 0.004,
        },
        "current_rss_before_mb": 10.0,
        "current_rss_after_mb": 11.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 20.0,
        "peak_rss_after_mb": 22.0,
        "peak_rss_delta_mb": 2.0,
    }


def test_label_style_and_format_helpers() -> None:
    assert benchmark._framework_label("pyhs3") == "PyHS3"
    assert benchmark._framework_label("unknown") == "unknown"
    assert benchmark._style_for("unknown")["label"] == "unknown"
    assert benchmark._format_compact(float("nan")) == "nan"
    assert benchmark._format_compact(0.0) == "0"
    assert benchmark._format_scientific(0.0) == "0"
    assert benchmark._safe_log_value(-1.0) == benchmark.PLOT_EPSILON
    assert benchmark._successful_results(
        [{"status": "success"}, {"status": "failed"}]
    ) == [{"status": "success"}]


def test_validate_config_success(
    valid_config: benchmark.BenchmarkConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", object())
    benchmark.validate_config(valid_config)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"frameworks": []}, "At least one framework"),
        ({"frameworks": ["bad"]}, "Unsupported framework"),
        ({"n_bins": 0}, "--n-bins"),
        ({"mu": -1.0}, "--mu"),
        ({"mu": float("nan")}, "--mu"),
        ({"delta_reference_mu": -1.0}, "--delta-reference-mu"),
        ({"mu": 1.0, "delta_reference_mu": 1.0}, "must be different"),
        ({"n_runs": 0}, "--n-runs"),
        ({"raw_tolerance": 0.0}, "--raw-tolerance"),
        ({"delta_tolerance": float("nan")}, "--delta-tolerance"),
    ],
)
def test_validate_config_rejects_invalid(
    valid_config: benchmark.BenchmarkConfig,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
    message: str,
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", object())
    config = benchmark.BenchmarkConfig(**{**valid_config.__dict__, **override})
    with pytest.raises(benchmark.BenchmarkConfigurationError, match=message):
        benchmark.validate_config(config)


def test_validate_config_rejects_missing_directory_and_roofit_without_root(
    valid_config: benchmark.BenchmarkConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", object())
    missing = benchmark.BenchmarkConfig(
        **{**valid_config.__dict__, "workspace_path": tmp_path / "missing.json"}
    )
    with pytest.raises(FileNotFoundError):
        benchmark.validate_config(missing)

    directory = tmp_path / "dir"
    directory.mkdir()
    not_file = benchmark.BenchmarkConfig(
        **{**valid_config.__dict__, "workspace_path": directory}
    )
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="not a file"):
        benchmark.validate_config(not_file)

    monkeypatch.setattr(benchmark, "ROOT", None)
    roofit = benchmark.BenchmarkConfig(
        **{**valid_config.__dict__, "frameworks": ["manual", "roofit"]}
    )
    with pytest.raises(
        benchmark.BenchmarkConfigurationError, match="ROOT is not available"
    ):
        benchmark.validate_config(roofit)


def test_load_workspace_delegates(
    monkeypatch: pytest.MonkeyPatch, workspace_file: Path
) -> None:
    fake_workspace = FakeWorkspace()
    monkeypatch.setattr(benchmark.Workspace, "load", lambda path: fake_workspace)
    assert benchmark.load_workspace(workspace_file) is fake_workspace


def test_extract_parameters_success_and_errors() -> None:
    params = benchmark.extract_parameters(FakeWorkspace(make_parameters(2)))
    assert params["signal_0"] == 1.0
    assert params["background_1"] == 11.0

    with pytest.raises(
        benchmark.BenchmarkConfigurationError, match="root parameter point"
    ):
        benchmark.extract_parameters(SimpleNamespace())

    workspace = SimpleNamespace(
        parameter_points=SimpleNamespace(
            root=[FakeParameterSet([FakeParameter("bad", object())])]
        )
    )
    with pytest.raises(
        benchmark.BenchmarkConfigurationError, match="cannot be converted"
    ):
        benchmark.extract_parameters(workspace)


def test_infer_n_bins_success_and_errors() -> None:
    assert benchmark.infer_n_bins(make_parameters(3)) == 3
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="Could not infer"):
        benchmark.infer_n_bins({"signal_0": 1.0})
    params = make_parameters(3)
    del params["obs_1"]
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="incomplete"):
        benchmark.infer_n_bins(params)


def test_validate_n_bins_get_vectors_and_manual_nll() -> None:
    benchmark.validate_n_bins_against_parameters(make_parameters(2), 2)
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="missing required"):
        benchmark.validate_n_bins_against_parameters(make_parameters(1), 2)

    vectors = benchmark.get_vectors(make_parameters(2), 2)
    assert vectors.n_bins == 2
    assert vectors.signal.tolist() == [1.0, 2.0]
    params = make_parameters(1)
    params["signal_0"] = -1.0
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="non-negative"):
        benchmark.get_vectors(params, 1)

    assert benchmark.poisson_nll(3.0, 2.0) == pytest.approx(
        2.0 - 3.0 * math.log(2.0) + math.lgamma(4.0)
    )
    with pytest.raises(ValueError, match="Expected count"):
        benchmark.poisson_nll(1.0, 0.0)
    with pytest.raises(ValueError, match="Observed count"):
        benchmark.poisson_nll(-1.0, 1.0)
    assert math.isfinite(benchmark.manual_nll_from_vectors(vectors, 1.0))
    bad_vectors = benchmark.BinnedVectors(
        signal=np.asarray([0.0]),
        background=np.asarray([0.0]),
        observed=np.asarray([1.0]),
    )
    with pytest.raises(ValueError, match="non-positive expected"):
        benchmark.manual_nll_from_vectors(bad_vectors, 1.0)


def test_build_manual_and_pyhs3_helpers(
    monkeypatch: pytest.MonkeyPatch, workspace_file: Path
) -> None:
    model = benchmark.build_manual_model(make_parameters(1), 1)
    assert isinstance(model, benchmark.BinnedVectors)
    assert benchmark.manual_nll(model, 1.0) == benchmark.manual_nll_from_vectors(
        model, 1.0
    )

    fake_model = FakePyHS3Model(0.5)
    monkeypatch.setattr(
        benchmark.Workspace, "load", lambda path: FakeWorkspace(model=fake_model)
    )
    assert benchmark.build_pyhs3_model(workspace_file) is fake_model
    assert benchmark.pyhs3_nll(fake_model, 2, 1.0) == pytest.approx(-2 * math.log(0.5))
    with pytest.raises(ValueError, match="invalid PDF"):
        benchmark.pyhs3_nll(FakePyHS3Model(0.0), 1, 1.0)


def test_pyhf_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark.pyhf, "Workspace", FakePyhfWorkspace)
    model, data = benchmark.build_pyhf_model(make_parameters(2), 2)
    assert isinstance(model, FakePyhfModel)
    assert data == ["data"]
    assert benchmark.pyhf_nll((model, data), 2.0) == pytest.approx(3.0)
    assert model.calls[-1][0] == [2.0]
    with pytest.raises(ValueError):
        benchmark.pyhf_nll((FakePyhfModel(has_mu=False), ["data"]), 1.0)


def test_roofit_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "ROOT", None)
    with pytest.raises(RuntimeError, match="ROOT is not available"):
        benchmark.build_roofit_model(make_parameters(1), 1, 1.0)

    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    model = benchmark.build_roofit_model(make_parameters(2), 2, 1.0)
    assert len(model["poissons"]) == 2
    assert benchmark.roofit_nll(model, 2.0) == pytest.approx(-2 * math.log(0.5))
    model["poissons"][0].value = 0.0
    with pytest.raises(ValueError, match="invalid PDF"):
        benchmark.roofit_nll(model, 1.0)


def test_numeric_and_timing_validation() -> None:
    benchmark.validate_numeric_value(1.0, "value")
    with pytest.raises(ValueError, match="finite"):
        benchmark.validate_numeric_value(float("nan"), "value")
    assert benchmark.summarize_timings_seconds([0.1])["std_seconds"] == 0.0
    assert benchmark.summarize_timings_seconds([0.1, 0.3])[
        "mean_seconds"
    ] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="empty"):
        benchmark.summarize_timings_seconds([])
    with pytest.raises(ValueError, match="positive finite"):
        benchmark.summarize_timings_seconds([0.0, float("nan")])


def test_measure_framework_success_and_nonfinite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = {"value": 0.0}

    def fake_perf_counter() -> float:
        counter["value"] += 0.1
        return counter["value"]

    monkeypatch.setattr(benchmark.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 10.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 20.0)
    result = benchmark.measure_framework(
        name="manual",
        input_load_func=lambda: {"input": object()},
        build_func=lambda data: {"model": data},
        eval_func=lambda model, mu: mu + 1.0,
        mu=1.0,
        delta_reference_mu=0.0,
        n_runs=2,
        warmup_iterations=1,
    )
    assert result["raw_nll"] == 2.0
    assert result["reference_nll"] == 1.0
    assert result["delta_nll"] == 1.0

    with pytest.raises(ValueError, match="raw NLL"):
        benchmark.measure_framework(
            name="manual",
            input_load_func=lambda: object(),
            build_func=lambda data: data,
            eval_func=lambda _model, _mu: float("nan"),
            mu=1.0,
            delta_reference_mu=0.0,
            n_runs=1,
            warmup_iterations=0,
        )


def test_failed_result_validation_and_summary(success_result: dict[str, Any]) -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        result = benchmark.failed_framework_result("manual", exc)
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"

    manual = dict(success_result, framework="manual", raw_nll=10.0, delta_nll=1.0)
    pyhf = dict(success_result, framework="pyhf", raw_nll=10.0, delta_nll=1.0)
    failed = {"framework": "pyhs3", "status": "failed", "validation_status": "failed"}
    results = [manual, pyhf, failed]
    benchmark.add_validation(results, raw_tolerance=1e-9, delta_tolerance=1e-9)
    assert pyhf["validation_status"] == "success"

    bad = dict(success_result, framework="pyhs3", raw_nll=10.1, delta_nll=1.2)
    results = [manual, bad]
    benchmark.add_validation(results, raw_tolerance=1e-9, delta_tolerance=1e-9)
    assert bad["validation_status"] == "failed"
    assert bad["error_type"] == "ValidationFailure"
    with pytest.raises(benchmark.ValidationFailure, match="Manual reference"):
        benchmark.add_validation([dict(success_result, framework="pyhf")], 1e-9, 1e-9)

    assert (
        benchmark.summarize_status([dict(success_result, validation_status="success")])[
            "status"
        ]
        == "success"
    )
    assert benchmark.summarize_status([])["status"] == "failed"


def test_printing(
    capsys: pytest.CaptureFixture[str],
    success_result: dict[str, Any],
    workspace_file: Path,
) -> None:
    benchmark.print_result(success_result)
    assert "raw NLL" in capsys.readouterr().out
    benchmark.print_result(
        {
            "framework": "bad",
            "framework_label": "Bad",
            "status": "failed",
            "validation_status": "failed",
            "error_type": "X",
            "error_message": "bad",
        }
    )
    assert "error:" in capsys.readouterr().out
    output_data = {
        "workspace": str(workspace_file),
        "n_bins": 2,
        "mu": 1.0,
        "delta_reference_mu": 0.0,
        "n_runs": 2,
        "summary": {
            "status": "failed",
            "n_validated": 1,
            "n_results": 2,
            "failed_results": [
                {"framework": "x", "error_type": "X", "error_message": "bad"}
            ],
        },
    }
    benchmark.print_final_summary(output_data)
    assert "Failed:" in capsys.readouterr().out


def plot_results(success_result: dict[str, Any]) -> list[dict[str, Any]]:
    manual = dict(
        success_result,
        framework="manual",
        framework_label="Manual",
        raw_nll_abs_diff=0.0,
        delta_nll_abs_diff=0.0,
    )
    pyhs3 = dict(
        success_result,
        framework="pyhs3",
        framework_label="PyHS3",
        raw_nll=10.000000001,
        raw_nll_abs_diff=1e-9,
        delta_nll_abs_diff=1e-10,
    )
    pyhf = dict(
        success_result,
        framework="pyhf",
        framework_label="pyhf",
        raw_nll=10.0,
        raw_nll_abs_diff=0.0,
        delta_nll_abs_diff=0.0,
    )
    return [manual, pyhs3, pyhf]


@pytest.mark.parametrize(
    ("plot_func", "filename", "extra"),
    [
        (benchmark.make_timing_profile_plot, "timing.png", ()),
        (benchmark.make_warm_evaluation_plot, "warm.png", ()),
        (benchmark.make_memory_plot, "memory.png", ()),
        (benchmark.make_nll_values_plot, "nll.png", ()),
        (benchmark.make_agreement_plot, "agreement.png", (1e-9, 1e-9)),
        (benchmark.make_summary_table, "summary.png", ()),
    ],
)
def test_plot_functions_create_png(
    tmp_path: Path,
    success_result: dict[str, Any],
    plot_func: Any,
    filename: str,
    extra: tuple[Any, ...],
) -> None:
    output = tmp_path / filename
    plot_func(plot_results(success_result), output, *extra)
    assert output.exists()


@pytest.mark.parametrize(
    "plot_func",
    [
        benchmark.make_timing_profile_plot,
        benchmark.make_warm_evaluation_plot,
        benchmark.make_memory_plot,
        benchmark.make_nll_values_plot,
        benchmark.make_summary_table,
    ],
)
def test_plot_functions_reject_no_success(tmp_path: Path, plot_func: Any) -> None:
    with pytest.raises(ValueError, match="No successful"):
        plot_func([{"status": "failed"}], tmp_path / "plot.png")


def test_agreement_plot_rejects_no_non_reference(
    tmp_path: Path, success_result: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="No non-reference"):
        benchmark.make_agreement_plot(
            [dict(success_result, framework="manual")],
            tmp_path / "agreement.png",
            1e-9,
            1e-9,
        )


def test_make_plots_and_save_figure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], success_result: dict[str, Any]
) -> None:
    benchmark.make_plots(
        plot_results(success_result), tmp_path, raw_tolerance=1e-9, delta_tolerance=1e-9
    )
    assert (tmp_path / "cross_binned_likelihood_timing_profile.png").exists()
    benchmark.make_plots(
        [success_result], tmp_path / "skip", raw_tolerance=1e-9, delta_tolerance=1e-9
    )
    assert "Skipping plots" in capsys.readouterr().out
    fig, _ax = benchmark.plt.subplots()
    output = tmp_path / "nested" / "plot.png"
    benchmark._save_figure(fig, output)
    assert output.exists()


def test_build_framework_jobs(
    monkeypatch: pytest.MonkeyPatch, workspace_file: Path
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    jobs = benchmark.build_framework_jobs(
        frameworks=["manual", "pyhs3", "pyhf", "roofit"],
        parameters=make_parameters(1),
        workspace_path=workspace_file,
        n_bins=1,
        mu=1.0,
    )
    assert list(jobs) == ["manual", "pyhs3", "pyhf", "roofit"]
    assert isinstance(jobs["manual"][0](), benchmark.BinnedVectors)


def test_run_benchmark_success(
    monkeypatch: pytest.MonkeyPatch, valid_config: benchmark.BenchmarkConfig
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", object())
    monkeypatch.setattr(
        benchmark, "load_workspace", lambda path: FakeWorkspace(make_parameters(1))
    )
    measured = [
        {
            "framework": "manual",
            "framework_label": "Manual",
            "status": "success",
            "validation_status": "pending",
            "raw_nll": 10.0,
            "reference_nll": 9.0,
            "delta_nll": 1.0,
            "input_load_time_seconds": 0.001,
            "model_build_time_seconds": 0.001,
            "warmup_iterations": 1,
            "warmup_time_seconds": 0.001,
            "first_evaluation_time_seconds": 0.001,
            "warm_evaluation": {
                "mean_seconds": 0.001,
                "std_seconds": 0.0,
                "min_seconds": 0.001,
                "max_seconds": 0.001,
            },
            "current_rss_delta_mb": 0.0,
            "peak_rss_delta_mb": 0.0,
        },
        {
            "framework": "pyhf",
            "framework_label": "pyhf",
            "status": "success",
            "validation_status": "pending",
            "raw_nll": 10.0,
            "reference_nll": 9.0,
            "delta_nll": 1.0,
            "input_load_time_seconds": 0.001,
            "model_build_time_seconds": 0.001,
            "warmup_iterations": 1,
            "warmup_time_seconds": 0.001,
            "first_evaluation_time_seconds": 0.001,
            "warm_evaluation": {
                "mean_seconds": 0.001,
                "std_seconds": 0.0,
                "min_seconds": 0.001,
                "max_seconds": 0.001,
            },
            "current_rss_delta_mb": 0.0,
            "peak_rss_delta_mb": 0.0,
        },
    ]
    monkeypatch.setattr(
        benchmark, "measure_framework", lambda **kwargs: measured.pop(0)
    )
    config = benchmark.BenchmarkConfig(
        **{**valid_config.__dict__, "frameworks": ["manual", "pyhf"]}
    )
    output = benchmark.run_benchmark(config)
    assert output["status"] == "success"
    assert output["n_bins"] == 1
    assert output["summary"]["n_validated"] == 2


def test_run_benchmark_records_failure_and_fail_fast(
    monkeypatch: pytest.MonkeyPatch, valid_config: benchmark.BenchmarkConfig
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", object())
    monkeypatch.setattr(
        benchmark, "load_workspace", lambda path: FakeWorkspace(make_parameters(1))
    )
    monkeypatch.setattr(
        benchmark,
        "measure_framework",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    config = benchmark.BenchmarkConfig(
        **{**valid_config.__dict__, "frameworks": ["manual"], "fail_fast": True}
    )
    with pytest.raises(benchmark.ValidationFailure, match="Manual reference"):
        benchmark.run_benchmark(config)


def test_run_writes_output_and_calls_plots(
    monkeypatch: pytest.MonkeyPatch, workspace_file: Path, tmp_path: Path
) -> None:
    output_data = {
        "benchmark": benchmark.BENCHMARK_NAME,
        "workspace": str(workspace_file),
        "n_bins": 1,
        "mu": 1.0,
        "delta_reference_mu": 0.0,
        "n_runs": 1,
        "summary": {
            "status": "success",
            "n_validated": 1,
            "n_results": 1,
            "failed_results": [],
        },
        "status": "success",
        "results": [
            {
                "framework": "manual",
                "framework_label": "Manual",
                "status": "success",
                "validation_status": "success",
                "raw_nll": 1.0,
                "reference_nll": 0.5,
                "delta_nll": 0.5,
                "raw_nll_abs_diff": 0.0,
                "delta_nll_abs_diff": 0.0,
                "model_build_time_seconds": 0.001,
                "first_evaluation_time_seconds": 0.001,
                "warm_evaluation": {"mean_seconds": 0.001, "std_seconds": 0.0},
                "current_rss_delta_mb": 0.0,
                "peak_rss_delta_mb": 0.0,
            }
        ],
    }
    monkeypatch.setattr(benchmark, "run_benchmark", lambda config: output_data)
    plot_calls: list[Any] = []
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda *args, **kwargs: plot_calls.append((args, kwargs)),
    )
    result = benchmark.run(
        workspace_path=workspace_file,
        frameworks=["manual"],
        n_bins=None,
        mu=1.0,
        delta_reference_mu=0.0,
        n_runs=1,
        output_dir=tmp_path / "out",
        output_name="result.json",
        plot=True,
        plot_dir=tmp_path / "plots",
        raw_tolerance=1e-9,
        delta_tolerance=1e-9,
    )
    assert result is output_data
    assert (tmp_path / "out" / "result.json").exists()
    assert plot_calls


def test_parse_args_and_main(
    monkeypatch: pytest.MonkeyPatch, workspace_file: Path, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args([])
    args = benchmark.parse_args(["--workspace", str(workspace_file)])
    assert args.workspace == workspace_file
    assert args.frameworks == benchmark.DEFAULT_FRAMEWORKS
    args = benchmark.parse_args(
        [
            "--workspace",
            str(workspace_file),
            "--frameworks",
            "manual",
            "pyhf",
            "--n-bins",
            "2",
            "--mu",
            "1.5",
            "--delta-reference-mu",
            "0.5",
            "--n-runs",
            "3",
            "--output-dir",
            str(tmp_path / "out"),
            "--output-name",
            "x.json",
            "--plot",
            "--plot-dir",
            str(tmp_path / "plots"),
            "--raw-tolerance",
            "1e-8",
            "--delta-tolerance",
            "1e-7",
            "--fail-fast",
        ]
    )
    assert args.frameworks == ["manual", "pyhf"]
    assert args.n_bins == 2
    assert args.fail_fast is True

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(benchmark, "run", lambda **kwargs: calls.append(kwargs))
    benchmark.main(
        [
            "--workspace",
            str(workspace_file),
            "--frameworks",
            "manual",
            "--n-runs",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
            "--fail-fast",
        ]
    )
    assert calls[0]["workspace_path"] == workspace_file
    assert calls[0]["frameworks"] == ["manual"]
    assert calls[0]["fail_fast"] is True

    monkeypatch.setattr(
        benchmark, "run", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError, match="Cross-framework binned likelihood"):
        benchmark.main(["--workspace", str(workspace_file)])
