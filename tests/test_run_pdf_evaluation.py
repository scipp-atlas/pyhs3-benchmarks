from __future__ import annotations

import json
import math
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src import run_pdf_evaluation as benchmark


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


class FakeModel:
    def __init__(self, outputs: list[Any] | None = None) -> None:
        self.data = {"x": np.array([1.0])}
        self.free_params = {"mu": np.array([2.0])}
        self.distributions = {"sig_ch0": object(), "bkg_ch0": object()}
        self.outputs = outputs or [np.array([0.5])]
        self.calls = 0

    def pdf(self, distribution: str, **parameters: Any) -> Any:
        if distribution not in self.distributions:
            raise KeyError(distribution)
        output = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return output


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def valid_result() -> dict[str, Any]:
    return {
        "benchmark": "pdf_evaluation",
        "workspace": "workspace.json",
        "workspace_path": "/tmp/workspace.json",
        "target": "analysis",
        "mode": "FAST_RUN",
        "distribution": "sig_ch0",
        "n_evaluations": 3,
        "available_distributions": ["sig_ch0", "bkg_ch0"],
        "cold_start_time_seconds": 0.01,
        "cold_start_output": 0.5,
        "total_runtime_seconds": 0.3,
        "average_runtime_seconds_per_evaluation": 0.1,
        "throughput_evaluations_per_second": 10.0,
        "first_timing_output": 0.5,
        "last_timing_output": 0.5,
        "memory_n_evaluations": 3,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 101.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 122.0,
        "peak_rss_delta_mb": 2.0,
        "status": "success",
        "n_outputs": 3,
        "all_outputs_finite": True,
        "reference_output": 0.5,
        "max_absolute_deviation": 0.0,
        "outputs_stable": True,
    }


def test_validate_workspace_path_success(workspace_path: Path) -> None:
    assert benchmark.validate_workspace_path(workspace_path) == workspace_path


def test_validate_workspace_path_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.validate_workspace_path(tmp_path / "missing.json")


def test_validate_workspace_path_directory_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace path is not a file"):
        benchmark.validate_workspace_path(tmp_path)


def test_validate_benchmark_config_success() -> None:
    benchmark.validate_benchmark_config(
        target="analysis",
        mode="FAST_RUN",
        distribution="sig_ch0",
        n_evaluations=1,
    )


@pytest.mark.parametrize(
    ("target", "mode", "distribution", "n_evaluations", "message"),
    [
        ("", "FAST_RUN", "sig_ch0", 1, "target must be a non-empty string"),
        ("analysis", "", "sig_ch0", 1, "mode must be a non-empty string"),
        ("analysis", "FAST_RUN", "", 1, "distribution must be a non-empty string"),
        ("analysis", "FAST_RUN", "sig_ch0", 0, "n_evaluations must be at least 1"),
    ],
)
def test_validate_benchmark_config_rejects_invalid_values(
    target: str,
    mode: str,
    distribution: str,
    n_evaluations: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_benchmark_config(target, mode, distribution, n_evaluations)


def test_verify_output_file_success(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"
    output_path.write_text("{}")
    benchmark.verify_output_file(output_path)


def test_verify_output_file_missing(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError, match="Benchmark output file was not created"
    ):
        benchmark.verify_output_file(tmp_path / "missing.json")


def test_verify_output_file_directory_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Benchmark output path is not a file"):
        benchmark.verify_output_file(tmp_path)


def test_build_parameter_inputs(fake_model: FakeModel) -> None:
    result = benchmark.build_parameter_inputs(fake_model)

    assert set(result) == {"x", "mu"}
    assert isinstance(result["x"], np.ndarray)
    assert result["x"].dtype == float
    assert result["mu"].dtype == float


def test_build_parameter_inputs_propagates_data_error() -> None:
    class BadModel:
        @property
        def data(self):
            raise RuntimeError("data failed")

        free_params = {"mu": 1.0}

    with pytest.raises(RuntimeError, match="data failed"):
        benchmark.build_parameter_inputs(BadModel())


def test_extract_scalar_output_success() -> None:
    assert benchmark.extract_scalar_output(np.array([[1.25]])) == 1.25


@pytest.mark.parametrize("result", [np.array([]), []])
def test_extract_scalar_output_rejects_empty_result(result: Any) -> None:
    with pytest.raises(ValueError, match="PDF result is empty"):
        benchmark.extract_scalar_output(result)


@pytest.mark.parametrize("result", [np.array([math.nan]), np.array([math.inf])])
def test_extract_scalar_output_rejects_non_finite_result(result: Any) -> None:
    with pytest.raises(ValueError, match="PDF result is not finite"):
        benchmark.extract_scalar_output(result)


def test_measure_cold_start_pdf_call_success(
    monkeypatch: pytest.MonkeyPatch, fake_model: FakeModel
) -> None:
    perf_counter_values = iter([1.0, 1.25])
    monkeypatch.setattr(
        benchmark.time, "perf_counter", lambda: next(perf_counter_values)
    )

    result = benchmark.measure_cold_start_pdf_call(
        model=fake_model,
        distribution="sig_ch0",
        parameters={"x": np.array([1.0])},
    )

    assert result == {
        "cold_start_time_seconds": 0.25,
        "cold_start_output": 0.5,
    }


def test_measure_cold_start_pdf_call_propagates_pdf_error(
    fake_model: FakeModel,
) -> None:
    with pytest.raises(KeyError):
        benchmark.measure_cold_start_pdf_call(fake_model, "missing", {})


def test_evaluate_pdf_success(fake_model: FakeModel) -> None:
    outputs = benchmark.evaluate_pdf(fake_model, "sig_ch0", {}, n_evaluations=3)

    assert outputs == [0.5, 0.5, 0.5]


def test_evaluate_pdf_zero_evaluations_returns_empty(fake_model: FakeModel) -> None:
    assert benchmark.evaluate_pdf(fake_model, "sig_ch0", {}, n_evaluations=0) == []


def test_evaluate_pdf_propagates_pdf_error(fake_model: FakeModel) -> None:
    with pytest.raises(KeyError):
        benchmark.evaluate_pdf(fake_model, "missing", {}, n_evaluations=1)


def test_validate_pdf_outputs_success() -> None:
    result = benchmark.validate_pdf_outputs([0.5, 0.5, 0.5])

    assert result == {
        "n_outputs": 3,
        "all_outputs_finite": True,
        "reference_output": 0.5,
        "max_absolute_deviation": 0.0,
        "outputs_stable": True,
    }


def test_validate_pdf_outputs_detects_unstable_outputs() -> None:
    result = benchmark.validate_pdf_outputs([0.5, 0.5001])

    assert result["outputs_stable"] is False
    assert result["max_absolute_deviation"] == pytest.approx(0.0001)


def test_validate_pdf_outputs_rejects_empty_outputs() -> None:
    with pytest.raises(ValueError, match="No PDF outputs were produced"):
        benchmark.validate_pdf_outputs([])


@pytest.mark.parametrize("outputs", [[0.5, math.nan], [math.inf], [-math.inf]])
def test_validate_pdf_outputs_rejects_non_finite_outputs(outputs: list[float]) -> None:
    with pytest.raises(ValueError, match="PDF outputs contain non-finite values"):
        benchmark.validate_pdf_outputs(outputs)


def test_measure_pdf_evaluation_timing_success(
    monkeypatch: pytest.MonkeyPatch,
    fake_model: FakeModel,
) -> None:
    perf_counter_values = iter([1.0, 1.3])
    monkeypatch.setattr(
        benchmark.time, "perf_counter", lambda: next(perf_counter_values)
    )

    result = benchmark.measure_pdf_evaluation_timing(
        fake_model, "sig_ch0", {}, n_evaluations=3
    )

    assert result["n_evaluations"] == 3
    assert result["total_runtime_seconds"] == pytest.approx(0.3)
    assert result["average_runtime_seconds_per_evaluation"] == pytest.approx(0.1)
    assert result["throughput_evaluations_per_second"] == pytest.approx(10.0)
    assert result["first_timing_output"] == 0.5
    assert result["last_timing_output"] == 0.5


def test_measure_pdf_evaluation_timing_rejects_invalid_n_evaluations(
    fake_model: FakeModel,
) -> None:
    with pytest.raises(ValueError, match="n_evaluations must be at least 1"):
        benchmark.measure_pdf_evaluation_timing(fake_model, "sig_ch0", {}, 0)


def test_measure_pdf_evaluation_timing_handles_zero_runtime(
    monkeypatch: pytest.MonkeyPatch,
    fake_model: FakeModel,
) -> None:
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: 1.0)

    result = benchmark.measure_pdf_evaluation_timing(
        fake_model, "sig_ch0", {}, n_evaluations=2
    )

    assert result["total_runtime_seconds"] == 0.0
    assert result["throughput_evaluations_per_second"] == float("inf")


def test_measure_pdf_evaluation_timing_propagates_pdf_error(
    fake_model: FakeModel,
) -> None:
    with pytest.raises(KeyError):
        benchmark.measure_pdf_evaluation_timing(fake_model, "missing", {}, 1)


def test_measure_pdf_evaluation_memory_success(
    monkeypatch: pytest.MonkeyPatch,
    fake_model: FakeModel,
) -> None:
    current_rss_values = iter([100.0, 103.0])
    peak_rss_values = iter([120.0, 125.0])
    monkeypatch.setattr(
        benchmark, "get_current_rss_mb", lambda: next(current_rss_values)
    )
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: next(peak_rss_values))

    result = benchmark.measure_pdf_evaluation_memory(
        fake_model, "sig_ch0", {}, n_evaluations=3
    )

    assert result == {
        "memory_n_evaluations": 3,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 103.0,
        "current_rss_delta_mb": 3.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 125.0,
        "peak_rss_delta_mb": 5.0,
    }


def test_measure_pdf_evaluation_memory_rejects_invalid_n_evaluations(
    fake_model: FakeModel,
) -> None:
    with pytest.raises(ValueError, match="n_evaluations must be at least 1"):
        benchmark.measure_pdf_evaluation_memory(fake_model, "sig_ch0", {}, 0)


def test_measure_pdf_evaluation_memory_propagates_pdf_error(
    fake_model: FakeModel,
) -> None:
    with pytest.raises(KeyError):
        benchmark.measure_pdf_evaluation_memory(fake_model, "missing", {}, 1)


def test_run_single_benchmark_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    fake_model: FakeModel,
) -> None:
    workspace = SimpleNamespace()
    monkeypatch.setattr(benchmark, "load_workspace", lambda path: workspace)
    monkeypatch.setattr(
        benchmark, "create_model", lambda workspace, target, mode: fake_model
    )
    monkeypatch.setattr(
        benchmark, "build_parameter_inputs", lambda model: {"x": np.array([1.0])}
    )
    monkeypatch.setattr(
        benchmark,
        "measure_cold_start_pdf_call",
        lambda model, distribution, parameters: {
            "cold_start_time_seconds": 0.01,
            "cold_start_output": 0.5,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_pdf",
        lambda model, distribution, parameters, n_evaluations: [0.5, 0.5],
    )
    monkeypatch.setattr(
        benchmark,
        "validate_pdf_outputs",
        lambda outputs: {
            "n_outputs": 2,
            "all_outputs_finite": True,
            "reference_output": 0.5,
            "max_absolute_deviation": 0.0,
            "outputs_stable": True,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "measure_pdf_evaluation_memory",
        lambda model, distribution, parameters, n_evaluations: {
            "memory_n_evaluations": n_evaluations,
            "current_rss_before_mb": 100.0,
            "current_rss_after_mb": 101.0,
            "current_rss_delta_mb": 1.0,
            "peak_rss_before_mb": 120.0,
            "peak_rss_after_mb": 122.0,
            "peak_rss_delta_mb": 2.0,
        },
    )
    monkeypatch.setattr(
        benchmark,
        "measure_pdf_evaluation_timing",
        lambda model, distribution, parameters, n_evaluations: {
            "n_evaluations": n_evaluations,
            "total_runtime_seconds": 0.2,
            "average_runtime_seconds_per_evaluation": 0.1,
            "throughput_evaluations_per_second": 10.0,
            "first_timing_output": 0.5,
            "last_timing_output": 0.5,
        },
    )

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        distribution="sig_ch0",
        n_evaluations=2,
    )

    assert result["benchmark"] == "pdf_evaluation"
    assert result["workspace"] == "workspace.json"
    assert result["target"] == "analysis"
    assert result["mode"] == "FAST_RUN"
    assert result["distribution"] == "sig_ch0"
    assert result["n_evaluations"] == 2
    assert result["status"] == "success"
    assert result["available_distributions"] == ["sig_ch0", "bkg_ch0"]


def test_run_single_benchmark_rejects_invalid_config(workspace_path: Path) -> None:
    with pytest.raises(ValueError, match="n_evaluations must be at least 1"):
        benchmark.run_single_benchmark(
            workspace_path, "analysis", "FAST_RUN", "sig_ch0", 0
        )


def test_run_single_benchmark_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.run_single_benchmark(
            tmp_path / "missing.json", "analysis", "FAST_RUN", "sig_ch0", 1
        )


def test_run_single_benchmark_rejects_missing_distribution(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    fake_model: FakeModel,
) -> None:
    monkeypatch.setattr(benchmark, "load_workspace", lambda path: SimpleNamespace())
    monkeypatch.setattr(
        benchmark, "create_model", lambda workspace, target, mode: fake_model
    )

    with pytest.raises(KeyError, match="Distribution 'missing' not found"):
        benchmark.run_single_benchmark(
            workspace_path, "analysis", "FAST_RUN", "missing", 1
        )


def test_run_single_benchmark_propagates_load_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "load_workspace",
        lambda path: (_ for _ in ()).throw(RuntimeError("load failed")),
    )

    with pytest.raises(RuntimeError, match="load failed"):
        benchmark.run_single_benchmark(
            workspace_path, "analysis", "FAST_RUN", "sig_ch0", 1
        )


def test_print_result_outputs_summary(
    capsys: pytest.CaptureFixture[str], valid_result: dict[str, Any]
) -> None:
    benchmark.print_result(valid_result)

    output = capsys.readouterr().out

    assert "PDF evaluation benchmark" in output
    assert "Cold start" in output
    assert "Warm timing" in output
    assert "Memory" in output
    assert "Validation" in output
    assert "workspace.json" in output
    assert "sig_ch0" in output


def test_print_error_result_outputs_failed_summary(
    capsys: pytest.CaptureFixture[str],
    workspace_path: Path,
) -> None:
    result = benchmark.make_error_result(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        distribution="sig_ch0",
        n_evaluations=2,
        exc=RuntimeError("pdf failed"),
    )

    benchmark.print_error_result(result)

    output = capsys.readouterr().out

    assert "PDF evaluation benchmark FAILED" in output
    assert "workspace.json" in output
    assert "analysis" in output
    assert "FAST_RUN" in output
    assert "sig_ch0" in output
    assert "RuntimeError: pdf failed" in output


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_pdf_evaluation.py"])

    args = benchmark.parse_args()

    assert args.workspaces == [benchmark.DEFAULT_WORKSPACE]
    assert args.targets == [benchmark.DEFAULT_TARGET]
    assert args.modes == [benchmark.DEFAULT_MODE]
    assert args.distributions == [benchmark.DEFAULT_DISTRIBUTION]
    assert args.n_evaluations == benchmark.DEFAULT_N_EVALUATIONS
    assert args.output_dir == benchmark.DEFAULT_OUTPUT_DIR
    assert args.output_name == benchmark.DEFAULT_OUTPUT_NAME
    assert args.plot is False


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pdf_evaluation.py",
            "--workspaces",
            "a.json",
            "b.json",
            "--targets",
            "analysis",
            "likelihood",
            "--modes",
            "FAST_RUN",
            "FAST_COMPILE",
            "--distributions",
            "sig_ch0",
            "bkg_ch0",
            "--n-evaluations",
            "1",
            "10",
            "--output-dir",
            "results/custom",
            "--output-name",
            "custom.json",
            "--plot",
            "--plot-dir",
            "plots/custom",
        ],
    )

    args = benchmark.parse_args()

    assert args.workspaces == [Path("a.json"), Path("b.json")]
    assert args.targets == ["analysis", "likelihood"]
    assert args.modes == ["FAST_RUN", "FAST_COMPILE"]
    assert args.distributions == ["sig_ch0", "bkg_ch0"]
    assert args.n_evaluations == [1, 10]
    assert args.output_dir == Path("results/custom")
    assert args.output_name == "custom.json"
    assert args.plot is True
    assert args.plot_dir == Path("plots/custom")


def test_make_plots_calls_plot_helpers_for_available_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    grouped_calls = []
    line_calls = []

    monkeypatch.setattr(
        benchmark,
        "make_grouped_bar_plot",
        lambda **kwargs: grouped_calls.append(kwargs),
    )
    monkeypatch.setattr(
        benchmark,
        "make_line_plot_by_evaluations",
        lambda **kwargs: line_calls.append(kwargs),
    )
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda results, metric: True)

    second_result = valid_result.copy()
    second_result["n_evaluations"] = 10

    benchmark.make_plots([valid_result, second_result], tmp_path)

    assert len(grouped_calls) == 3
    assert len(line_calls) == 2
    assert grouped_calls[0]["metric_key"] == "cold_start_time_ms"
    assert line_calls[0]["metric_key"] == "average_runtime_ms_per_evaluation"
    assert line_calls[1]["metric_key"] == "throughput_evaluations_per_second"
    assert grouped_calls[1]["metric_key"] == "current_rss_delta_mb"
    assert grouped_calls[2]["metric_key"] == "peak_rss_delta_mb"


def test_make_plots_skips_optional_memory_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    grouped_calls = []
    line_calls = []

    monkeypatch.setattr(
        benchmark,
        "make_grouped_bar_plot",
        lambda **kwargs: grouped_calls.append(kwargs),
    )
    monkeypatch.setattr(
        benchmark,
        "make_line_plot_by_evaluations",
        lambda **kwargs: line_calls.append(kwargs),
    )
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda results, metric: False)

    second_result = valid_result.copy()
    second_result["n_evaluations"] = 10

    benchmark.make_plots([valid_result, second_result], tmp_path)

    assert len(grouped_calls) == 1
    assert len(line_calls) == 2
    assert grouped_calls[0]["metric_key"] == "cold_start_time_ms"


def test_make_plots_skips_when_fewer_than_two_successful_results(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    failed_result = valid_result.copy()
    failed_result["status"] = "failed"

    benchmark.make_plots([valid_result, failed_result], tmp_path)

    output = capsys.readouterr().out

    assert (
        "Skipping plots: at least two successful result entries are needed." in output
    )
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


class FakePool:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def __enter__(self) -> "FakePool":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def apply(self, func: Any, args: tuple[Any, ...]) -> dict[str, Any]:
        return self.result


class FakeContext:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def Pool(self, processes: int) -> FakePool:
        assert processes == 1
        return FakePool(self.result)


def test_main_saves_json_and_verifies_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    output_dir = tmp_path / "results"
    output_name = "result.json"
    saved_payloads = []
    verified_paths = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pdf_evaluation.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "analysis",
            "--modes",
            "FAST_RUN",
            "--distributions",
            "sig_ch0",
            "--n-evaluations",
            "2",
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
        ],
    )
    monkeypatch.setattr(
        benchmark, "get_context", lambda method: FakeContext(valid_result)
    )
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)

    def fake_save_json(payload: dict[str, Any], output_path: Path) -> None:
        saved_payloads.append(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}")

    monkeypatch.setattr(benchmark, "save_json", fake_save_json)
    monkeypatch.setattr(
        benchmark,
        "verify_output_file",
        lambda output_path: verified_paths.append(output_path),
    )

    benchmark.main()

    assert saved_payloads == [
        {
            "benchmark": "pdf_evaluation",
            "n_results": 1,
            "n_successful_results": 1,
            "n_failed_results": 0,
            "results": [valid_result],
        }
    ]
    assert verified_paths == [output_dir / output_name]


def test_main_records_invalid_n_evaluations_as_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    saved_payloads = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pdf_evaluation.py",
            "--workspaces",
            str(workspace_path),
            "--n-evaluations",
            "0",
            "--output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, output_path: (
            saved_payloads.append(payload)
            or output_path.parent.mkdir(parents=True, exist_ok=True)
            or output_path.write_text("{}")
        ),
    )
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)
    monkeypatch.setattr(benchmark, "print_error_result", lambda result: None)

    benchmark.main()

    assert saved_payloads[0]["n_results"] == 1
    assert saved_payloads[0]["n_successful_results"] == 0
    assert saved_payloads[0]["n_failed_results"] == 1
    result = saved_payloads[0]["results"][0]
    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"
    assert "n_evaluations must be at least 1" in result["error_message"]


def test_main_records_missing_workspace_as_failed_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    missing_workspace = tmp_path / "missing.json"
    saved_payloads = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pdf_evaluation.py",
            "--workspaces",
            str(missing_workspace),
            "--output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, output_path: (
            saved_payloads.append(payload)
            or output_path.parent.mkdir(parents=True, exist_ok=True)
            or output_path.write_text("{}")
        ),
    )
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)
    monkeypatch.setattr(benchmark, "print_error_result", lambda result: None)

    benchmark.main()

    payload = saved_payloads[0]
    assert payload["n_results"] == len(benchmark.DEFAULT_N_EVALUATIONS)
    assert payload["n_successful_results"] == 0
    assert payload["n_failed_results"] == len(benchmark.DEFAULT_N_EVALUATIONS)
    assert all(result["status"] == "failed" for result in payload["results"])
    assert all(
        result["error_type"] == "FileNotFoundError" for result in payload["results"]
    )
    assert all(
        "Workspace file does not exist" in result["error_message"]
        for result in payload["results"]
    )


def test_main_skips_plots_for_single_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pdf_evaluation.py",
            "--workspaces",
            str(workspace_path),
            "--output-dir",
            str(tmp_path / "results"),
            "--n-evaluations",
            "1",
            "--plot",
        ],
    )
    monkeypatch.setattr(
        benchmark, "get_context", lambda method: FakeContext(valid_result)
    )
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, output_path: (
            output_path.parent.mkdir(parents=True, exist_ok=True)
            or output_path.write_text("{}")
        ),
    )
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)
    monkeypatch.setattr(
        benchmark, "make_plots", lambda *args, **kwargs: calls.append(kwargs)
    )

    benchmark.main()

    assert len(calls) == 1
    assert len(calls[0]["results"]) == 1
    assert calls[0]["results"] == [valid_result]
    assert calls[0]["plot_dir"] == benchmark.DEFAULT_PLOT_DIR


def test_main_creates_plots_for_multiple_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pdf_evaluation.py",
            "--workspaces",
            str(workspace_path),
            "--n-evaluations",
            "1",
            "2",
            "--output-dir",
            str(tmp_path / "results"),
            "--plot",
        ],
    )
    monkeypatch.setattr(
        benchmark, "get_context", lambda method: FakeContext(valid_result)
    )
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, output_path: (
            output_path.parent.mkdir(parents=True, exist_ok=True)
            or output_path.write_text("{}")
        ),
    )
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda results, plot_dir: calls.append((results, plot_dir)),
    )

    benchmark.main()

    assert len(calls) == 1
    assert len(calls[0][0]) == 2


def test_module_entrypoint_calls_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    import multiprocessing

    class EntrypointContext:
        def Pool(self, processes: int) -> FakePool:
            assert processes == 1
            return FakePool(valid_result)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pdf_evaluation.py",
            "--output-dir",
            str(tmp_path / "results"),
            "--output-name",
            "entrypoint.json",
            "--n-evaluations",
            "1",
        ],
    )
    monkeypatch.setattr(
        multiprocessing, "get_context", lambda method: EntrypointContext()
    )

    runpy.run_module("src.run_pdf_evaluation", run_name="__main__")


def test_run_single_benchmark_real_workspace() -> None:
    result = benchmark.run_single_benchmark(
        workspace_path=benchmark.DEFAULT_WORKSPACE,
        target=benchmark.DEFAULT_TARGET,
        mode=benchmark.DEFAULT_MODE,
        distribution=benchmark.DEFAULT_DISTRIBUTION,
        n_evaluations=1,
    )

    assert result["status"] == "success"
    assert result["benchmark"] == "pdf_evaluation"
    assert result["n_evaluations"] == 1
    assert result["distribution"] == benchmark.DEFAULT_DISTRIBUTION
    assert result["n_outputs"] == 1
    assert result["all_outputs_finite"] is True
    assert result["cold_start_time_seconds"] >= 0
    assert result["average_runtime_seconds_per_evaluation"] >= 0
    assert result["throughput_evaluations_per_second"] >= 0
    assert result["current_rss_before_mb"] >= 0
    assert result["peak_rss_before_mb"] >= 0


def test_main_real_run_writes_output_json_and_uses_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_name = "pdf_evaluation_result.json"
    output_path = output_dir / output_name

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pdf_evaluation.py",
            "--workspaces",
            str(benchmark.DEFAULT_WORKSPACE),
            "--targets",
            benchmark.DEFAULT_TARGET,
            "--modes",
            benchmark.DEFAULT_MODE,
            "--distributions",
            benchmark.DEFAULT_DISTRIBUTION,
            "--n-evaluations",
            "1",
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
        ],
    )

    benchmark.main()

    assert output_path.exists()
    assert output_path.is_file()

    with output_path.open() as file:
        payload = json.load(file)

    assert payload["benchmark"] == "pdf_evaluation"
    assert payload["n_results"] == 1
    assert len(payload["results"]) == 1
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][0]["distribution"] == benchmark.DEFAULT_DISTRIBUTION
    assert payload["results"][0]["n_evaluations"] == 1


def test_make_plots_real_png_files_created(
    tmp_path: Path, valid_result: dict[str, Any]
) -> None:
    second_result = valid_result.copy()
    second_result["n_evaluations"] = 10

    benchmark.make_plots([valid_result, second_result], tmp_path)

    assert (tmp_path / "pdf_evaluation_cold_start_time_grouped.png").exists()
    assert (tmp_path / "pdf_evaluation_average_time_lines.png").exists()
    assert (tmp_path / "pdf_evaluation_throughput_lines.png").exists()
    assert (tmp_path / "pdf_evaluation_current_rss_delta_grouped.png").exists()
    assert (tmp_path / "pdf_evaluation_peak_rss_delta_grouped.png").exists()
