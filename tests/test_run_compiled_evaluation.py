from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import run_compiled_evaluation as benchmark


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


class FakeCompiled:
    def __init__(self, value: float = 1.25) -> None:
        self.value = value
        self.calls = 0

    def __call__(self, **kwargs: Any) -> tuple[list[float]]:
        self.calls += 1
        return ([self.value],)


@pytest.fixture
def compiled() -> FakeCompiled:
    return FakeCompiled(1.25)


@pytest.fixture
def validation_inputs() -> dict[str, float]:
    return {"x": 1.0, "mu": 1.0}


@pytest.fixture
def valid_result() -> dict[str, Any]:
    return {
        "benchmark": "compiled_evaluation",
        "workspace": "workspace.json",
        "workspace_path": "/tmp/workspace.json",
        "target": "analysis",
        "mode": "FAST_RUN",
        "n_evaluations": 10,
        "total_runtime_seconds": 0.5,
        "average_runtime_seconds_per_evaluation": 0.05,
        "throughput_evaluations_per_second": 20.0,
        "first_timing_output": 1.25,
        "last_timing_output": 1.25,
        "memory_n_evaluations": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 101.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 122.0,
        "peak_rss_delta_mb": 2.0,
        "status": "success",
        "n_outputs": 3,
        "all_outputs_finite": True,
        "reference_output": 1.25,
        "max_absolute_deviation": 0.0,
        "outputs_stable": True,
    }


def make_second_result(result: dict[str, Any]) -> dict[str, Any]:
    second = dict(result)
    second["workspace"] = "workspace2.json"
    second["workspace_path"] = "/tmp/workspace2.json"
    second["n_evaluations"] = 100
    second["total_runtime_seconds"] = 4.0
    second["average_runtime_seconds_per_evaluation"] = 0.04
    second["throughput_evaluations_per_second"] = 25.0
    return second


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
        target="analysis", mode="FAST_RUN", n_evaluations=1
    )


@pytest.mark.parametrize(
    ("target", "mode", "n_evaluations", "message"),
    [
        ("", "FAST_RUN", 1, "target must be a non-empty string"),
        ("analysis", "", 1, "mode must be a non-empty string"),
        ("analysis", "FAST_RUN", 0, "n_evaluations must be at least 1"),
    ],
)
def test_validate_benchmark_config_rejects_invalid_values(
    target: str,
    mode: str,
    n_evaluations: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_benchmark_config(target, mode, n_evaluations)


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


def test_extract_scalar_output_success() -> None:
    assert benchmark.extract_scalar_output(([1.25],)) == 1.25


@pytest.mark.parametrize("result", [1.25, [1.25], {"x": 1.25}])
def test_extract_scalar_output_rejects_non_tuple(result: Any) -> None:
    with pytest.raises(TypeError, match="Expected compiled result to be a tuple"):
        benchmark.extract_scalar_output(result)


def test_extract_scalar_output_rejects_empty_tuple() -> None:
    with pytest.raises(ValueError, match="Compiled result tuple is empty"):
        benchmark.extract_scalar_output(())


def test_extract_scalar_output_accepts_scalar_tuple() -> None:
    assert benchmark.extract_scalar_output((1.25,)) == 1.25


def test_extract_scalar_output_rejects_invalid_scalar_value() -> None:
    with pytest.raises(TypeError, match="Could not extract scalar float"):
        benchmark.extract_scalar_output((["not-a-number"],))


def test_extract_scalar_output_rejects_empty_array() -> None:
    with pytest.raises(ValueError, match="Compiled result array is empty"):
        benchmark.extract_scalar_output(([],))


def test_prepare_compiled_graph_success(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    model = SimpleNamespace(name="model")
    log_prob = SimpleNamespace(name="log_prob")
    compiled = FakeCompiled(1.25)
    inputs = {"x": 1.0}

    monkeypatch.setattr(benchmark, "build_log_prob", lambda **kwargs: (model, log_prob))
    monkeypatch.setattr(benchmark, "compile_log_prob", lambda lp: compiled)
    monkeypatch.setattr(
        benchmark, "build_validation_inputs", lambda model, compiled: inputs
    )

    result = benchmark.prepare_compiled_graph(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
    )

    assert result == (model, compiled, inputs)


def test_prepare_compiled_graph_propagates_build_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_build_log_prob(**kwargs: Any) -> None:
        raise RuntimeError("build failed")

    monkeypatch.setattr(benchmark, "build_log_prob", failing_build_log_prob)

    with pytest.raises(RuntimeError, match="build failed"):
        benchmark.prepare_compiled_graph(workspace_path, "analysis", "FAST_RUN")


def test_prepare_compiled_graph_propagates_compile_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    monkeypatch.setattr(
        benchmark, "build_log_prob", lambda **kwargs: (object(), object())
    )

    def failing_compile_log_prob(log_prob: Any) -> None:
        raise RuntimeError("compile failed")

    monkeypatch.setattr(benchmark, "compile_log_prob", failing_compile_log_prob)

    with pytest.raises(RuntimeError, match="compile failed"):
        benchmark.prepare_compiled_graph(workspace_path, "analysis", "FAST_RUN")


def test_evaluate_compiled_graph_success(
    compiled: FakeCompiled, validation_inputs: dict[str, float]
) -> None:
    outputs = benchmark.evaluate_compiled_graph(
        compiled, validation_inputs, n_evaluations=3
    )

    assert outputs == [1.25, 1.25, 1.25]
    assert compiled.calls == 3


def test_evaluate_compiled_graph_zero_evaluations_returns_empty(
    compiled: FakeCompiled,
    validation_inputs: dict[str, float],
) -> None:
    assert benchmark.evaluate_compiled_graph(compiled, validation_inputs, 0) == []


def test_evaluate_compiled_graph_propagates_compiled_error(
    validation_inputs: dict[str, float],
) -> None:
    def failing_compiled(**kwargs: Any) -> None:
        raise RuntimeError("evaluation failed")

    with pytest.raises(RuntimeError, match="evaluation failed"):
        benchmark.evaluate_compiled_graph(failing_compiled, validation_inputs, 1)


def test_validate_evaluation_success() -> None:
    result = benchmark.validate_evaluation([1.25, 1.25, 1.25])

    assert result == {
        "n_outputs": 3,
        "all_outputs_finite": True,
        "reference_output": 1.25,
        "max_absolute_deviation": 0.0,
        "outputs_stable": True,
    }


def test_validate_evaluation_detects_unstable_outputs() -> None:
    result = benchmark.validate_evaluation([1.0, 1.0 + 1e-6])

    assert result["all_outputs_finite"] is True
    assert result["outputs_stable"] is False
    assert result["max_absolute_deviation"] == pytest.approx(1e-6)


def test_validate_evaluation_rejects_empty_outputs() -> None:
    with pytest.raises(ValueError, match="No evaluation outputs were produced"):
        benchmark.validate_evaluation([])


@pytest.mark.parametrize(
    "outputs", [[float("nan")], [float("inf")], [1.0, float("-inf")]]
)
def test_validate_evaluation_rejects_non_finite_outputs(outputs: list[float]) -> None:
    with pytest.raises(
        ValueError, match="Evaluation outputs contain non-finite values"
    ):
        benchmark.validate_evaluation(outputs)


def test_measure_evaluation_timing_success(
    monkeypatch: pytest.MonkeyPatch,
    compiled: FakeCompiled,
    validation_inputs: dict[str, float],
) -> None:
    perf_counter_values = iter([10.0, 10.5])

    monkeypatch.setattr(
        benchmark.time, "perf_counter", lambda: next(perf_counter_values)
    )

    result = benchmark.measure_evaluation_timing(
        compiled, validation_inputs, n_evaluations=5
    )

    assert result["n_evaluations"] == 5
    assert result["total_runtime_seconds"] == pytest.approx(0.5)
    assert result["average_runtime_seconds_per_evaluation"] == pytest.approx(0.1)
    assert result["throughput_evaluations_per_second"] == pytest.approx(10.0)
    assert result["first_timing_output"] == 1.25
    assert result["last_timing_output"] == 1.25
    assert compiled.calls == 6


def test_measure_evaluation_timing_rejects_invalid_n_evaluations(
    compiled: FakeCompiled,
    validation_inputs: dict[str, float],
) -> None:
    with pytest.raises(ValueError, match="n_evaluations must be at least 1"):
        benchmark.measure_evaluation_timing(
            compiled, validation_inputs, n_evaluations=0
        )


def test_measure_evaluation_timing_handles_zero_runtime(
    monkeypatch: pytest.MonkeyPatch,
    compiled: FakeCompiled,
    validation_inputs: dict[str, float],
) -> None:
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: 10.0)

    result = benchmark.measure_evaluation_timing(
        compiled, validation_inputs, n_evaluations=2
    )

    assert result["total_runtime_seconds"] == 0.0
    assert result["average_runtime_seconds_per_evaluation"] == 0.0
    assert result["throughput_evaluations_per_second"] == float("inf")


def test_measure_evaluation_timing_propagates_compiled_error(
    validation_inputs: dict[str, float],
) -> None:
    def failing_compiled(**kwargs: Any) -> None:
        raise RuntimeError("timing failed")

    with pytest.raises(RuntimeError, match="timing failed"):
        benchmark.measure_evaluation_timing(
            failing_compiled, validation_inputs, n_evaluations=1
        )


def test_measure_evaluation_memory_success(
    monkeypatch: pytest.MonkeyPatch,
    compiled: FakeCompiled,
    validation_inputs: dict[str, float],
) -> None:
    current_rss_values = iter([100.0, 104.0])
    peak_rss_values = iter([120.0, 125.5])

    monkeypatch.setattr(
        benchmark, "get_current_rss_mb", lambda: next(current_rss_values)
    )
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: next(peak_rss_values))

    result = benchmark.measure_evaluation_memory(compiled, validation_inputs)

    assert result == {
        "memory_n_evaluations": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 104.0,
        "current_rss_delta_mb": 4.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 125.5,
        "peak_rss_delta_mb": 5.5,
    }
    assert compiled.calls == 2


def test_measure_evaluation_memory_propagates_compiled_error(
    validation_inputs: dict[str, float],
) -> None:
    def failing_compiled(**kwargs: Any) -> None:
        raise RuntimeError("memory failed")

    with pytest.raises(RuntimeError, match="memory failed"):
        benchmark.measure_evaluation_memory(failing_compiled, validation_inputs)


def test_run_single_benchmark_success(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    model = SimpleNamespace(name="model")
    compiled = FakeCompiled(1.25)
    inputs = {"x": 1.0}
    memory_summary = {
        "memory_n_evaluations": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 101.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 122.0,
        "peak_rss_delta_mb": 2.0,
    }
    timing_summary = {
        "n_evaluations": 10,
        "total_runtime_seconds": 0.5,
        "average_runtime_seconds_per_evaluation": 0.05,
        "throughput_evaluations_per_second": 20.0,
        "first_timing_output": 1.25,
        "last_timing_output": 1.25,
    }

    monkeypatch.setattr(
        benchmark, "prepare_compiled_graph", lambda **kwargs: (model, compiled, inputs)
    )
    monkeypatch.setattr(
        benchmark, "evaluate_compiled_graph", lambda **kwargs: [1.25, 1.25, 1.25]
    )
    monkeypatch.setattr(
        benchmark, "measure_evaluation_memory", lambda **kwargs: memory_summary
    )
    monkeypatch.setattr(
        benchmark, "measure_evaluation_timing", lambda **kwargs: timing_summary
    )

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        n_evaluations=10,
    )

    assert result["benchmark"] == "compiled_evaluation"
    assert result["workspace"] == "workspace.json"
    assert result["workspace_path"] == str(workspace_path)
    assert result["target"] == "analysis"
    assert result["mode"] == "FAST_RUN"
    assert result["n_evaluations"] == 10
    assert result["total_runtime_seconds"] == 0.5
    assert result["memory_n_evaluations"] == 1
    assert result["status"] == "success"
    assert result["n_outputs"] == 3
    assert result["outputs_stable"] is True


def test_run_single_benchmark_rejects_invalid_config(workspace_path: Path) -> None:
    with pytest.raises(ValueError, match="n_evaluations must be at least 1"):
        benchmark.run_single_benchmark(workspace_path, "analysis", "FAST_RUN", 0)


def test_run_single_benchmark_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.run_single_benchmark(
            tmp_path / "missing.json", "analysis", "FAST_RUN", 1
        )


def test_run_single_benchmark_propagates_prepare_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_prepare_compiled_graph(**kwargs: Any) -> None:
        raise RuntimeError("prepare failed")

    monkeypatch.setattr(
        benchmark, "prepare_compiled_graph", failing_prepare_compiled_graph
    )

    with pytest.raises(RuntimeError, match="prepare failed"):
        benchmark.run_single_benchmark(workspace_path, "analysis", "FAST_RUN", 1)


def test_run_single_benchmark_propagates_validation_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "prepare_compiled_graph",
        lambda **kwargs: (SimpleNamespace(), FakeCompiled(1.25), {"x": 1.0}),
    )
    monkeypatch.setattr(benchmark, "evaluate_compiled_graph", lambda **kwargs: [])

    with pytest.raises(ValueError, match="No evaluation outputs were produced"):
        benchmark.run_single_benchmark(workspace_path, "analysis", "FAST_RUN", 1)


def test_print_result_outputs_summary(
    capsys: pytest.CaptureFixture[str], valid_result: dict[str, Any]
) -> None:
    benchmark.print_result(valid_result)

    output = capsys.readouterr().out

    assert "Compiled evaluation benchmark" in output
    assert "Timing" in output
    assert "Memory" in output
    assert "Validation" in output
    assert "workspace.json" in output
    assert "analysis" in output
    assert "FAST_RUN" in output


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_compiled_evaluation.py"])

    args = benchmark.parse_args()

    assert args.workspaces == [benchmark.DEFAULT_WORKSPACE]
    assert args.targets == [benchmark.DEFAULT_TARGET]
    assert args.modes == [benchmark.DEFAULT_MODE]
    assert args.n_evaluations == benchmark.DEFAULT_N_EVALUATIONS
    assert args.output_dir == benchmark.DEFAULT_OUTPUT_DIR
    assert args.output_name == benchmark.DEFAULT_OUTPUT_NAME
    assert args.plot is False


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_compiled_evaluation.py",
            "--workspaces",
            "a.json",
            "b.json",
            "--targets",
            "analysis",
            "likelihood",
            "--modes",
            "FAST_RUN",
            "FAST_COMPILE",
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
    assert args.n_evaluations == [1, 10]
    assert args.output_dir == Path("results/custom")
    assert args.output_name == "custom.json"
    assert args.plot is True
    assert args.plot_dir == Path("plots/custom")


def test_make_scaling_line_plot_real_png_created(
    tmp_path: Path, valid_result: dict[str, Any]
) -> None:
    output_path = tmp_path / "scaling.png"

    benchmark.make_scaling_line_plot(
        results=[valid_result],
        output_path=output_path,
        metric_key="average_runtime_seconds_per_evaluation",
        metric_label="Average wall time per evaluation [ms]",
        title="Compiled evaluation average wall time",
    )

    assert output_path.exists()
    assert output_path.is_file()


def test_make_plots_calls_expected_plotters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    scaling_calls = []
    bar_calls = []

    def fake_make_scaling_line_plot(**kwargs: Any) -> None:
        scaling_calls.append(kwargs)

    def fake_make_bar_plot(**kwargs: Any) -> None:
        bar_calls.append(kwargs)

    monkeypatch.setattr(
        benchmark, "make_scaling_line_plot", fake_make_scaling_line_plot
    )
    monkeypatch.setattr(benchmark, "make_bar_plot", fake_make_bar_plot)
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda results, metric: True)

    benchmark.make_plots([valid_result, make_second_result(valid_result)], tmp_path)

    assert len(scaling_calls) == 2
    assert scaling_calls[0]["metric_key"] == "average_runtime_seconds_per_evaluation"
    assert scaling_calls[1]["metric_key"] == "throughput_evaluations_per_second"
    assert len(bar_calls) == 2
    assert bar_calls[0]["metric_key"] == "current_rss_delta_mb"
    assert bar_calls[1]["metric_key"] == "peak_rss_delta_mb"


def test_make_plots_skips_optional_memory_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    scaling_calls = []
    bar_calls = []

    monkeypatch.setattr(
        benchmark,
        "make_scaling_line_plot",
        lambda **kwargs: scaling_calls.append(kwargs),
    )
    monkeypatch.setattr(
        benchmark, "make_bar_plot", lambda **kwargs: bar_calls.append(kwargs)
    )
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda results, metric: False)

    benchmark.make_plots([valid_result, make_second_result(valid_result)], tmp_path)

    output = capsys.readouterr().out

    assert len(scaling_calls) == 2
    assert bar_calls == []
    assert "Skipping current RSS plot" in output
    assert "Skipping peak RSS plot" in output


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
            "run_compiled_evaluation.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "analysis",
            "--modes",
            "FAST_RUN",
            "--n-evaluations",
            "10",
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
            "benchmark": "compiled_evaluation",
            "n_results": 1,
            "results": [valid_result],
        }
    ]
    assert verified_paths == [output_dir / output_name]


def test_main_rejects_invalid_n_evaluations(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_compiled_evaluation.py",
            "--workspaces",
            str(workspace_path),
            "--n-evaluations",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="--n-evaluations values must be at least 1"):
        benchmark.main()


def test_main_rejects_missing_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing_workspace = tmp_path / "missing.json"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_compiled_evaluation.py", "--workspaces", str(missing_workspace)],
    )

    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.main()


def test_main_skips_plots_for_single_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    make_plots_calls = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_compiled_evaluation.py",
            "--workspaces",
            str(workspace_path),
            "--n-evaluations",
            "1",
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
        benchmark, "make_plots", lambda *args, **kwargs: make_plots_calls.append(kwargs)
    )

    benchmark.main()

    assert len(make_plots_calls) == 1
    assert len(make_plots_calls[0]["results"]) == 1


def test_main_creates_plots_for_multiple_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    make_plots_calls = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_compiled_evaluation.py",
            "--workspaces",
            str(workspace_path),
            "--n-evaluations",
            "1",
            "10",
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
        lambda results, plot_dir: make_plots_calls.append((results, plot_dir)),
    )

    benchmark.main()

    assert len(make_plots_calls) == 1
    assert len(make_plots_calls[0][0]) == 2


def test_run_single_benchmark_real_workspace() -> None:
    result = benchmark.run_single_benchmark(
        workspace_path=benchmark.DEFAULT_WORKSPACE,
        target=benchmark.DEFAULT_TARGET,
        mode=benchmark.DEFAULT_MODE,
        n_evaluations=1,
    )

    assert result["status"] == "success"
    assert result["target"] == benchmark.DEFAULT_TARGET
    assert result["mode"] == benchmark.DEFAULT_MODE
    assert result["n_evaluations"] == 1
    assert result["n_outputs"] == benchmark.VALIDATION_N_EVALUATIONS
    assert result["all_outputs_finite"] is True
    assert result["outputs_stable"] is True
    assert result["total_runtime_seconds"] >= 0
    assert result["average_runtime_seconds_per_evaluation"] >= 0
    assert result["throughput_evaluations_per_second"] >= 0
    assert result["current_rss_before_mb"] >= 0
    assert result["current_rss_after_mb"] >= 0
    assert result["peak_rss_before_mb"] >= 0
    assert result["peak_rss_after_mb"] >= 0


def test_main_real_run_writes_output_json_and_uses_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_name = "compiled_evaluation_result.json"
    output_path = output_dir / output_name

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_compiled_evaluation.py",
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

    assert payload["benchmark"] == "compiled_evaluation"
    assert payload["n_results"] == 1
    assert len(payload["results"]) == 1
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][0]["workspace"] == benchmark.DEFAULT_WORKSPACE.name
    assert payload["results"][0]["target"] == benchmark.DEFAULT_TARGET
    assert payload["results"][0]["mode"] == benchmark.DEFAULT_MODE
    assert payload["results"][0]["n_evaluations"] == 1


def test_make_plots_real_png_files_created(
    tmp_path: Path, valid_result: dict[str, Any]
) -> None:
    benchmark.make_plots(
        results=[valid_result, make_second_result(valid_result)], plot_dir=tmp_path
    )

    assert (tmp_path / "compiled_evaluation_average_time.png").exists()
    assert (tmp_path / "compiled_evaluation_throughput.png").exists()
    assert (tmp_path / "compiled_evaluation_current_rss_delta.png").exists()
    assert (tmp_path / "compiled_evaluation_peak_rss_delta.png").exists()


def test_extract_scalar_output_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="Compiled result is not finite"):
        benchmark.extract_scalar_output(([float("inf")],))


def test_make_scaling_line_plot_skips_without_successful_metric(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark.make_scaling_line_plot(
        results=[{"status": "failed", "workspace": "failed.json"}],
        output_path=tmp_path / "missing.png",
        metric_key="average_runtime_seconds_per_evaluation",
        metric_label="Average wall time per evaluation [ms]",
        title="Missing metric plot",
    )

    output = capsys.readouterr().out

    assert "Skipping Missing metric plot" in output
    assert not (tmp_path / "missing.png").exists()


def test_make_plots_skips_when_fewer_than_two_successes(
    tmp_path: Path,
    valid_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark.make_plots(
        results=[valid_result, {"status": "failed", "workspace": "failed.json"}],
        plot_dir=tmp_path,
    )

    output = capsys.readouterr().out

    assert (
        "Skipping plots: at least two successful result entries are needed." in output
    )
    assert not any(tmp_path.glob("*.png"))


def test_make_error_result_includes_traceback(workspace_path: Path) -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        result = benchmark.make_error_result(
            workspace_path=workspace_path,
            target="analysis",
            mode="FAST_RUN",
            n_evaluations=10,
            exc=exc,
        )

    assert result["benchmark"] == "compiled_evaluation"
    assert result["workspace"] == "workspace.json"
    assert result["workspace_path"] == str(workspace_path)
    assert result["target"] == "analysis"
    assert result["mode"] == "FAST_RUN"
    assert result["n_evaluations"] == 10
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "boom"
    assert "RuntimeError: boom" in result["traceback"]


def test_print_failed_result_outputs_summary(
    capsys: pytest.CaptureFixture[str],
    workspace_path: Path,
) -> None:
    result = benchmark.make_error_result(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        n_evaluations=10,
        exc=ValueError("bad run"),
    )

    benchmark.print_failed_result(result)

    output = capsys.readouterr().out

    assert "Compiled evaluation benchmark FAILED" in output
    assert "workspace.json" in output
    assert "analysis" in output
    assert "FAST_RUN" in output
    assert "failed" in output
    assert "ValueError: bad run" in output


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--targets", ""], "--targets must contain only non-empty strings"),
        (["--modes", ""], "--modes must contain only non-empty strings"),
    ],
)
def test_main_rejects_empty_targets_and_modes(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    argv: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_compiled_evaluation.py", "--workspaces", str(workspace_path), *argv],
    )

    with pytest.raises(ValueError, match=message):
        benchmark.main()


class FailingPool:
    def __enter__(self) -> "FailingPool":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def apply(self, func: Any, args: tuple[Any, ...]) -> dict[str, Any]:
        raise RuntimeError("worker failed")


class FailingContext:
    def Pool(self, processes: int) -> FailingPool:
        assert processes == 1
        return FailingPool()


def test_main_records_failed_result_when_worker_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    saved_payloads = []
    printed_failures = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_compiled_evaluation.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "analysis",
            "--modes",
            "FAST_RUN",
            "--n-evaluations",
            "10",
            "--output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.setattr(benchmark, "get_context", lambda method: FailingContext())
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)
    monkeypatch.setattr(
        benchmark,
        "print_failed_result",
        lambda result: printed_failures.append(result),
    )

    def fake_save_json(payload: dict[str, Any], output_path: Path) -> None:
        saved_payloads.append(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}")

    monkeypatch.setattr(benchmark, "save_json", fake_save_json)
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)

    benchmark.main()

    assert len(saved_payloads) == 1
    result = saved_payloads[0]["results"][0]
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "worker failed"
    assert printed_failures == [result]
