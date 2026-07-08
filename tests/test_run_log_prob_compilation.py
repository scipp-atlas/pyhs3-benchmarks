from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src import run_log_prob_compilation as benchmark


class FakeCompiledGraph:
    def __init__(
        self,
        result: object = None,
        input_names: list[str] | None = None,
    ) -> None:
        self.input_names = input_names or ["x", "mu"]
        self.result = result if result is not None else (np.array([1.25]),)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self.result


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


@pytest.fixture
def fake_model() -> SimpleNamespace:
    return SimpleNamespace(name="model")


@pytest.fixture
def fake_compiled_graph() -> FakeCompiledGraph:
    return FakeCompiledGraph()


@pytest.fixture
def valid_result() -> dict[str, Any]:
    return {
        "benchmark": "log_prob_compilation",
        "workspace": "workspace.json",
        "workspace_path": "/tmp/workspace.json",
        "target": "analysis",
        "mode": "FAST_RUN",
        "n_runs": 3,
        "timings_seconds": [0.1, 0.2, 0.3],
        "wall_time_seconds_mean": 0.2,
        "wall_time_seconds_median": 0.2,
        "wall_time_seconds_std": 0.08165,
        "memory_n_runs": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 105.0,
        "current_rss_delta_mb": 5.0,
        "peak_rss_before_mb": 110.0,
        "peak_rss_after_mb": 120.0,
        "peak_rss_delta_mb": 10.0,
        "status": "success",
        "compiled_type": "FakeCompiledGraph",
        "n_compiled_inputs": 2,
        "compiled_input_names": ["x", "mu"],
        "validation_result_type": "ndarray",
        "validation_first_value": 1.25,
        "validation_result_is_finite": True,
    }


def test_validate_workspace_path_success(workspace_path: Path) -> None:
    assert benchmark.validate_workspace_path(workspace_path) == workspace_path


def test_validate_workspace_path_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.validate_workspace_path(missing_path)


def test_validate_workspace_path_directory_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace path is not a file"):
        benchmark.validate_workspace_path(tmp_path)


def test_validate_benchmark_config_success() -> None:
    benchmark.validate_benchmark_config(target="analysis", mode="FAST_RUN", n_runs=1)


@pytest.mark.parametrize(
    ("target", "mode", "n_runs", "message"),
    [
        ("", "FAST_RUN", 1, "target must be a non-empty string"),
        ("analysis", "", 1, "mode must be a non-empty string"),
        ("analysis", "FAST_RUN", 0, "n_runs must be at least 1"),
    ],
)
def test_validate_benchmark_config_rejects_invalid_values(
    target: str,
    mode: str,
    n_runs: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_benchmark_config(target=target, mode=mode, n_runs=n_runs)


def test_validate_timings_success() -> None:
    benchmark.validate_timings([0.1, 0.2, 0.3])


def test_validate_timings_empty() -> None:
    with pytest.raises(ValueError, match="Timing samples are empty"):
        benchmark.validate_timings([])


@pytest.mark.parametrize("timings", [[0.1, 0.0], [0.1, -0.2]])
def test_validate_timings_rejects_non_positive_values(timings: list[float]) -> None:
    with pytest.raises(ValueError, match="All timing samples must be positive"):
        benchmark.validate_timings(timings)


def test_verify_output_file_success(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"
    output_path.write_text("{}")

    benchmark.verify_output_file(output_path)


def test_verify_output_file_missing(tmp_path: Path) -> None:
    output_path = tmp_path / "missing.json"

    with pytest.raises(
        FileNotFoundError, match="Benchmark output file was not created"
    ):
        benchmark.verify_output_file(output_path)


def test_verify_output_file_directory_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Benchmark output path is not a file"):
        benchmark.verify_output_file(tmp_path)


def test_validate_compiled_graph_success(
    monkeypatch: pytest.MonkeyPatch,
    fake_model: SimpleNamespace,
    fake_compiled_graph: FakeCompiledGraph,
) -> None:
    monkeypatch.setattr(benchmark, "JaxifiedGraph", FakeCompiledGraph)
    monkeypatch.setattr(
        benchmark,
        "build_validation_inputs",
        lambda model, compiled: {"x": np.array([1.0]), "mu": np.array([1.0])},
    )

    result = benchmark.validate_compiled_graph(
        model=fake_model,
        compiled=fake_compiled_graph,
    )

    assert result == {
        "compiled_type": "FakeCompiledGraph",
        "n_compiled_inputs": 2,
        "compiled_input_names": ["x", "mu"],
        "validation_result_type": "ndarray",
        "validation_first_value": 1.25,
        "validation_result_is_finite": True,
    }
    assert fake_compiled_graph.calls == [{"x": np.array([1.0]), "mu": np.array([1.0])}]


def test_validate_compiled_graph_rejects_none(fake_model: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="Compilation returned None"):
        benchmark.validate_compiled_graph(model=fake_model, compiled=None)


def test_validate_compiled_graph_rejects_wrong_type(
    fake_model: SimpleNamespace,
) -> None:
    with pytest.raises(TypeError, match="Expected JaxifiedGraph"):
        benchmark.validate_compiled_graph(model=fake_model, compiled=object())


def test_validate_compiled_graph_rejects_non_tuple_result(
    monkeypatch: pytest.MonkeyPatch,
    fake_model: SimpleNamespace,
) -> None:
    monkeypatch.setattr(benchmark, "JaxifiedGraph", FakeCompiledGraph)
    monkeypatch.setattr(
        benchmark, "build_validation_inputs", lambda model, compiled: {}
    )

    compiled = FakeCompiledGraph(result=np.array([1.0]))

    with pytest.raises(TypeError, match="Expected compiled result to be a tuple"):
        benchmark.validate_compiled_graph(model=fake_model, compiled=compiled)


def test_validate_compiled_graph_rejects_empty_tuple(
    monkeypatch: pytest.MonkeyPatch,
    fake_model: SimpleNamespace,
) -> None:
    monkeypatch.setattr(benchmark, "JaxifiedGraph", FakeCompiledGraph)
    monkeypatch.setattr(
        benchmark, "build_validation_inputs", lambda model, compiled: {}
    )

    compiled = FakeCompiledGraph(result=())

    with pytest.raises(ValueError, match="Compiled result tuple is empty"):
        benchmark.validate_compiled_graph(model=fake_model, compiled=compiled)


def test_validate_compiled_graph_rejects_non_finite_result(
    monkeypatch: pytest.MonkeyPatch,
    fake_model: SimpleNamespace,
) -> None:
    monkeypatch.setattr(benchmark, "JaxifiedGraph", FakeCompiledGraph)
    monkeypatch.setattr(
        benchmark, "build_validation_inputs", lambda model, compiled: {}
    )

    compiled = FakeCompiledGraph(result=(np.array([np.inf]),))

    with pytest.raises(ValueError, match="Compiled result is not finite"):
        benchmark.validate_compiled_graph(model=fake_model, compiled=compiled)


def test_validate_compiled_graph_propagates_input_build_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_model: SimpleNamespace,
    fake_compiled_graph: FakeCompiledGraph,
) -> None:
    monkeypatch.setattr(benchmark, "JaxifiedGraph", FakeCompiledGraph)

    def failing_build_validation_inputs(model: Any, compiled: Any) -> dict[str, Any]:
        raise RuntimeError("input build failed")

    monkeypatch.setattr(
        benchmark,
        "build_validation_inputs",
        failing_build_validation_inputs,
    )

    with pytest.raises(RuntimeError, match="input build failed"):
        benchmark.validate_compiled_graph(
            model=fake_model,
            compiled=fake_compiled_graph,
        )


def test_measure_compilation_memory_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    fake_model: SimpleNamespace,
    fake_compiled_graph: FakeCompiledGraph,
) -> None:
    current_rss_values = iter([100.0, 104.5])
    peak_rss_values = iter([120.0, 130.0])
    log_prob = object()

    monkeypatch.setattr(
        benchmark,
        "build_log_prob",
        lambda workspace_path, target, mode: (fake_model, log_prob),
    )
    monkeypatch.setattr(
        benchmark, "compile_log_prob", lambda graph: fake_compiled_graph
    )
    monkeypatch.setattr(
        benchmark, "get_current_rss_mb", lambda: next(current_rss_values)
    )
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: next(peak_rss_values))

    model, compiled, memory_summary = benchmark.measure_compilation_memory(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
    )

    assert model is fake_model
    assert compiled is fake_compiled_graph
    assert memory_summary == {
        "memory_n_runs": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 104.5,
        "current_rss_delta_mb": 4.5,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 130.0,
        "peak_rss_delta_mb": 10.0,
    }


def test_measure_compilation_memory_propagates_build_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_build_log_prob(workspace_path: Path, target: str, mode: str) -> None:
        raise RuntimeError("build failed")

    monkeypatch.setattr(benchmark, "build_log_prob", failing_build_log_prob)

    with pytest.raises(RuntimeError, match="build failed"):
        benchmark.measure_compilation_memory(
            workspace_path=workspace_path,
            target="analysis",
            mode="FAST_RUN",
        )


def test_measure_compilation_memory_propagates_compile_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    fake_model: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "build_log_prob",
        lambda workspace_path, target, mode: (fake_model, object()),
    )

    def failing_compile_log_prob(log_prob: Any) -> None:
        raise RuntimeError("compile failed")

    monkeypatch.setattr(benchmark, "compile_log_prob", failing_compile_log_prob)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 120.0)

    with pytest.raises(RuntimeError, match="compile failed"):
        benchmark.measure_compilation_memory(
            workspace_path=workspace_path,
            target="analysis",
            mode="FAST_RUN",
        )


def test_measure_compilation_timing_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    fake_model: SimpleNamespace,
    fake_compiled_graph: FakeCompiledGraph,
) -> None:
    perf_counter_values = iter([1.0, 1.2, 2.0, 2.3, 3.0, 3.4])
    build_calls = []
    compile_calls = []

    def fake_build_log_prob(
        workspace_path: Path, target: str, mode: str
    ) -> tuple[Any, Any]:
        log_prob = object()
        build_calls.append((workspace_path, target, mode, log_prob))
        return fake_model, log_prob

    def fake_compile_log_prob(log_prob: Any) -> FakeCompiledGraph:
        compile_calls.append(log_prob)
        return fake_compiled_graph

    monkeypatch.setattr(
        benchmark.time, "perf_counter", lambda: next(perf_counter_values)
    )
    monkeypatch.setattr(benchmark, "build_log_prob", fake_build_log_prob)
    monkeypatch.setattr(benchmark, "compile_log_prob", fake_compile_log_prob)

    timings = benchmark.measure_compilation_timing(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        n_runs=3,
    )

    assert timings == pytest.approx([0.2, 0.3, 0.4])
    assert len(build_calls) == 3
    assert len(compile_calls) == 3


def test_measure_compilation_timing_propagates_build_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_build_log_prob(workspace_path: Path, target: str, mode: str) -> None:
        raise RuntimeError("timing build failed")

    monkeypatch.setattr(benchmark, "build_log_prob", failing_build_log_prob)

    with pytest.raises(RuntimeError, match="timing build failed"):
        benchmark.measure_compilation_timing(
            workspace_path=workspace_path,
            target="analysis",
            mode="FAST_RUN",
            n_runs=1,
        )


def test_measure_compilation_timing_propagates_compile_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    fake_model: SimpleNamespace,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "build_log_prob",
        lambda workspace_path, target, mode: (fake_model, object()),
    )

    def failing_compile_log_prob(log_prob: Any) -> None:
        raise RuntimeError("timing compile failed")

    monkeypatch.setattr(benchmark, "compile_log_prob", failing_compile_log_prob)

    with pytest.raises(RuntimeError, match="timing compile failed"):
        benchmark.measure_compilation_timing(
            workspace_path=workspace_path,
            target="analysis",
            mode="FAST_RUN",
            n_runs=1,
        )


def test_run_single_benchmark_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    fake_model: SimpleNamespace,
    fake_compiled_graph: FakeCompiledGraph,
) -> None:
    memory_summary = {
        "memory_n_runs": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 101.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 122.0,
        "peak_rss_delta_mb": 2.0,
    }
    validation_summary = {
        "compiled_type": "FakeCompiledGraph",
        "n_compiled_inputs": 2,
        "compiled_input_names": ["x", "mu"],
        "validation_result_type": "ndarray",
        "validation_first_value": 1.25,
        "validation_result_is_finite": True,
    }
    timings = [0.1, 0.2, 0.3]
    timing_summary = {
        "wall_time_seconds_mean": 0.2,
        "wall_time_seconds_median": 0.2,
        "wall_time_seconds_std": 0.08165,
    }

    monkeypatch.setattr(
        benchmark,
        "measure_compilation_memory",
        lambda workspace_path, target, mode: (
            fake_model,
            fake_compiled_graph,
            memory_summary,
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "validate_compiled_graph",
        lambda model, compiled: validation_summary,
    )
    monkeypatch.setattr(
        benchmark,
        "measure_compilation_timing",
        lambda workspace_path, target, mode, n_runs: timings,
    )
    monkeypatch.setattr(benchmark, "summarize_timings", lambda samples: timing_summary)

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        n_runs=3,
    )

    assert result["benchmark"] == "log_prob_compilation"
    assert result["workspace"] == "workspace.json"
    assert result["workspace_path"] == str(workspace_path)
    assert result["target"] == "analysis"
    assert result["mode"] == "FAST_RUN"
    assert result["n_runs"] == 3
    assert result["timings_seconds"] == timings
    assert result["status"] == "success"
    assert result["wall_time_seconds_mean"] == 0.2
    assert result["compiled_type"] == "FakeCompiledGraph"
    assert result["validation_result_is_finite"] is True


def test_run_single_benchmark_rejects_invalid_config(workspace_path: Path) -> None:
    with pytest.raises(ValueError, match="n_runs must be at least 1"):
        benchmark.run_single_benchmark(
            workspace_path=workspace_path,
            target="analysis",
            mode="FAST_RUN",
            n_runs=0,
        )


def test_run_single_benchmark_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.run_single_benchmark(
            workspace_path=tmp_path / "missing.json",
            target="analysis",
            mode="FAST_RUN",
            n_runs=1,
        )


def test_run_single_benchmark_rejects_empty_timings(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    fake_model: SimpleNamespace,
    fake_compiled_graph: FakeCompiledGraph,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "measure_compilation_memory",
        lambda workspace_path, target, mode: (fake_model, fake_compiled_graph, {}),
    )
    monkeypatch.setattr(
        benchmark, "validate_compiled_graph", lambda model, compiled: {}
    )
    monkeypatch.setattr(
        benchmark,
        "measure_compilation_timing",
        lambda workspace_path, target, mode, n_runs: [],
    )

    with pytest.raises(ValueError, match="Timing samples are empty"):
        benchmark.run_single_benchmark(
            workspace_path=workspace_path,
            target="analysis",
            mode="FAST_RUN",
            n_runs=3,
        )


def test_run_single_benchmark_rejects_non_positive_timings(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    fake_model: SimpleNamespace,
    fake_compiled_graph: FakeCompiledGraph,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "measure_compilation_memory",
        lambda workspace_path, target, mode: (fake_model, fake_compiled_graph, {}),
    )
    monkeypatch.setattr(
        benchmark, "validate_compiled_graph", lambda model, compiled: {}
    )
    monkeypatch.setattr(
        benchmark,
        "measure_compilation_timing",
        lambda workspace_path, target, mode, n_runs: [0.1, 0.0],
    )

    with pytest.raises(ValueError, match="All timing samples must be positive"):
        benchmark.run_single_benchmark(
            workspace_path=workspace_path,
            target="analysis",
            mode="FAST_RUN",
            n_runs=3,
        )


def test_run_single_benchmark_propagates_memory_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_measure_compilation_memory(
        workspace_path: Path,
        target: str,
        mode: str,
    ) -> None:
        raise RuntimeError("memory failed")

    monkeypatch.setattr(
        benchmark,
        "measure_compilation_memory",
        failing_measure_compilation_memory,
    )

    with pytest.raises(RuntimeError, match="memory failed"):
        benchmark.run_single_benchmark(
            workspace_path=workspace_path,
            target="analysis",
            mode="FAST_RUN",
            n_runs=1,
        )


def test_print_result_outputs_summary(
    capsys: pytest.CaptureFixture[str],
    valid_result: dict[str, Any],
) -> None:
    benchmark.print_result(valid_result)

    output = capsys.readouterr().out

    assert "log_prob compilation benchmark" in output
    assert "Timing" in output
    assert "Memory" in output
    assert "Validation" in output
    assert "workspace.json" in output
    assert "analysis" in output
    assert "FAST_RUN" in output


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_log_prob_compilation.py"])

    args = benchmark.parse_args()

    assert args.workspaces == [benchmark.DEFAULT_WORKSPACE]
    assert args.targets == [benchmark.DEFAULT_TARGET]
    assert args.modes == [benchmark.DEFAULT_MODE]
    assert args.n_runs == benchmark.DEFAULT_N_RUNS
    assert args.output_dir == benchmark.DEFAULT_OUTPUT_DIR
    assert args.output_name == benchmark.DEFAULT_OUTPUT_NAME
    assert args.plot is False


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_log_prob_compilation.py",
            "--workspaces",
            "a.json",
            "b.json",
            "--targets",
            "analysis",
            "likelihood",
            "--modes",
            "FAST_RUN",
            "FAST_COMPILE",
            "--n-runs",
            "25",
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
    assert args.n_runs == 25
    assert args.output_dir == Path("results/custom")
    assert args.output_name == "custom.json"
    assert args.plot is True
    assert args.plot_dir == Path("plots/custom")


def test_make_plots_calls_make_bar_plot_for_available_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    calls = []

    def fake_make_bar_plot(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(benchmark, "make_bar_plot", fake_make_bar_plot)
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda results, metric: True)
    second_result = dict(valid_result)
    second_result["workspace"] = "workspace2.json"
    second_result["workspace_path"] = "/tmp/workspace2.json"

    benchmark.make_plots(results=[valid_result, second_result], plot_dir=tmp_path)

    assert len(calls) == 3
    assert calls[0]["metric_key"] == "wall_time_seconds_mean"
    assert calls[1]["metric_key"] == "current_rss_delta_mb"
    assert calls[2]["metric_key"] == "peak_rss_delta_mb"


def test_make_plots_skips_optional_memory_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = []

    def fake_make_bar_plot(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(benchmark, "make_bar_plot", fake_make_bar_plot)
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda results, metric: False)
    second_result = dict(valid_result)
    second_result["workspace"] = "workspace2.json"
    second_result["workspace_path"] = "/tmp/workspace2.json"

    benchmark.make_plots(results=[valid_result, second_result], plot_dir=tmp_path)

    output = capsys.readouterr().out

    assert len(calls) == 1
    assert calls[0]["metric_key"] == "wall_time_seconds_mean"
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
            "run_log_prob_compilation.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "analysis",
            "--modes",
            "FAST_RUN",
            "--n-runs",
            "3",
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

    def fake_verify_output_file(output_path: Path) -> None:
        verified_paths.append(output_path)

    monkeypatch.setattr(benchmark, "save_json", fake_save_json)
    monkeypatch.setattr(benchmark, "verify_output_file", fake_verify_output_file)

    benchmark.main()

    assert saved_payloads == [
        {
            "benchmark": "log_prob_compilation",
            "n_results": 1,
            "results": [valid_result],
        }
    ]
    assert verified_paths == [output_dir / output_name]


def test_main_rejects_invalid_n_runs(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_log_prob_compilation.py",
            "--workspaces",
            str(workspace_path),
            "--n-runs",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="--n-runs must be at least 1"):
        benchmark.main()


def test_main_records_missing_workspace_as_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_workspace = tmp_path / "missing.json"
    output_dir = tmp_path / "results"
    output_name = "missing.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_log_prob_compilation.py",
            "--workspaces",
            str(missing_workspace),
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
        ],
    )

    benchmark.main()

    with (output_dir / output_name).open() as handle:
        payload = json.load(handle)

    result = payload["results"][0]
    assert result["status"] == "failed"
    assert result["error_type"] == "FileNotFoundError"
    assert "Workspace file does not exist" in result["error_message"]


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
            "run_log_prob_compilation.py",
            "--workspaces",
            str(workspace_path),
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
        lambda *args, **kwargs: make_plots_calls.append(kwargs),
    )

    benchmark.main()

    assert len(make_plots_calls) == 1
    assert make_plots_calls[0]["results"] == [valid_result]
    assert make_plots_calls[0]["plot_dir"] == benchmark.DEFAULT_PLOT_DIR


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
            "run_log_prob_compilation.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "analysis",
            "likelihood",
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


def test_run_single_benchmark_mocked_workspace(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    fake_model: SimpleNamespace,
    fake_compiled_graph: FakeCompiledGraph,
) -> None:
    memory_summary = {
        "memory_n_runs": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 101.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 122.0,
        "peak_rss_delta_mb": 2.0,
    }
    validation_summary = {
        "compiled_type": "FakeCompiledGraph",
        "n_compiled_inputs": 2,
        "compiled_input_names": ["x", "mu"],
        "validation_result_type": "ndarray",
        "validation_first_value": 1.25,
        "validation_result_is_finite": True,
    }
    monkeypatch.setattr(
        benchmark,
        "measure_compilation_memory",
        lambda workspace_path, target, mode: (
            fake_model,
            fake_compiled_graph,
            memory_summary,
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "validate_compiled_graph",
        lambda model, compiled: validation_summary,
    )
    monkeypatch.setattr(
        benchmark,
        "measure_compilation_timing",
        lambda workspace_path, target, mode, n_runs: [0.1],
    )
    monkeypatch.setattr(
        benchmark,
        "summarize_timings",
        lambda timings: {
            "wall_time_seconds_mean": 0.1,
            "wall_time_seconds_median": 0.1,
            "wall_time_seconds_std": 0.0,
        },
    )

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target=benchmark.DEFAULT_TARGET,
        mode=benchmark.DEFAULT_MODE,
        n_runs=1,
    )

    assert result["status"] == "success"
    assert result["workspace"] == workspace_path.name
    assert result["target"] == benchmark.DEFAULT_TARGET
    assert result["mode"] == benchmark.DEFAULT_MODE
    assert result["n_runs"] == 1
    assert result["wall_time_seconds_mean"] == pytest.approx(0.1)
    assert result["validation_result_is_finite"] is True
    assert result["current_rss_before_mb"] >= 0
    assert result["current_rss_after_mb"] >= 0
    assert result["peak_rss_before_mb"] >= 0
    assert result["peak_rss_after_mb"] >= 0


def test_main_mocked_run_writes_output_json_and_uses_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    output_dir = tmp_path / "results"
    output_name = "log_prob_compilation_result.json"
    output_path = output_dir / output_name

    result = dict(valid_result)
    result["workspace"] = workspace_path.name
    result["workspace_path"] = str(workspace_path)
    result["target"] = benchmark.DEFAULT_TARGET
    result["mode"] = benchmark.DEFAULT_MODE

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_log_prob_compilation.py",
            "--workspaces",
            str(workspace_path),
            "--n-runs",
            "1",
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
        ],
    )
    monkeypatch.setattr(benchmark, "get_context", lambda method: FakeContext(result))
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)

    benchmark.main()

    assert output_path.exists()
    assert output_path.is_file()

    with output_path.open() as file:
        payload = json.load(file)

    assert payload["benchmark"] == "log_prob_compilation"
    assert payload["n_results"] == 1
    assert len(payload["results"]) == 1
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][0]["workspace"] == workspace_path.name
    assert payload["results"][0]["target"] == benchmark.DEFAULT_TARGET
    assert payload["results"][0]["mode"] == benchmark.DEFAULT_MODE


def test_make_plots_real_png_files_created(
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    second_result = dict(valid_result)
    second_result["workspace"] = "workspace2.json"
    second_result["workspace_path"] = "/tmp/workspace2.json"

    benchmark.make_plots(results=[valid_result, second_result], plot_dir=tmp_path)

    assert (tmp_path / "log_prob_compilation_wall_time.png").exists()
    assert (tmp_path / "log_prob_compilation_current_rss_delta.png").exists()
    assert (tmp_path / "log_prob_compilation_peak_rss_delta.png").exists()


def test_validate_compiled_graph_rejects_empty_array(
    monkeypatch: pytest.MonkeyPatch,
    fake_model: SimpleNamespace,
) -> None:
    monkeypatch.setattr(benchmark, "JaxifiedGraph", FakeCompiledGraph)
    monkeypatch.setattr(
        benchmark, "build_validation_inputs", lambda model, compiled: {}
    )

    compiled = FakeCompiledGraph(result=(np.array([]),))

    with pytest.raises(ValueError, match="Compiled result array is empty"):
        benchmark.validate_compiled_graph(model=fake_model, compiled=compiled)


def test_make_error_result_contains_traceback(workspace_path: Path) -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        result = benchmark.make_error_result(
            workspace_path=workspace_path,
            target="analysis",
            mode="FAST_RUN",
            n_runs=3,
            exc=exc,
        )

    assert result["benchmark"] == "log_prob_compilation"
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "boom"
    assert "RuntimeError: boom" in result["traceback"]


def test_print_error_result_outputs_summary(
    capsys: pytest.CaptureFixture[str],
    workspace_path: Path,
) -> None:
    result = benchmark.make_error_result(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        n_runs=1,
        exc=ValueError("bad config"),
    )

    benchmark.print_error_result(result)

    output = capsys.readouterr().out
    assert "log_prob compilation benchmark FAILED" in output
    assert "workspace.json" in output
    assert "ValueError: bad config" in output


def test_make_plots_skips_when_fewer_than_two_successful_results(
    tmp_path: Path,
    valid_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark.make_plots(results=[valid_result], plot_dir=tmp_path)

    output = capsys.readouterr().out
    assert "Skipping plots" in output
    assert not any(tmp_path.glob("*.png"))


def test_make_plots_ignores_failed_results(
    tmp_path: Path,
    valid_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    failed_result = dict(valid_result)
    failed_result["status"] = "failed"

    benchmark.make_plots(results=[valid_result, failed_result], plot_dir=tmp_path)

    output = capsys.readouterr().out
    assert "Skipping plots" in output
    assert not any(tmp_path.glob("*.png"))


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--targets", ""], "--targets must contain only non-empty strings"),
        (["--modes", ""], "--modes must contain only non-empty strings"),
    ],
)
def test_main_rejects_empty_target_or_mode(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    argv: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_log_prob_compilation.py", "--workspaces", str(workspace_path), *argv],
    )

    with pytest.raises(ValueError, match=message):
        benchmark.main()


def test_main_records_pool_error_as_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_name = "pool_error.json"
    saved_payloads = []

    class FailingPool:
        def __enter__(self) -> "FailingPool":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def apply(self, func: Any, args: tuple[Any, ...]) -> dict[str, Any]:
            raise RuntimeError("pool failed")

    class FailingContext:
        def Pool(self, processes: int) -> FailingPool:
            assert processes == 1
            return FailingPool()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_log_prob_compilation.py",
            "--workspaces",
            str(workspace_path),
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
        ],
    )
    monkeypatch.setattr(benchmark, "get_context", lambda method: FailingContext())
    monkeypatch.setattr(benchmark, "print_error_result", lambda result: None)

    def fake_save_json(payload: dict[str, Any], output_path: Path) -> None:
        saved_payloads.append(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}")

    monkeypatch.setattr(benchmark, "save_json", fake_save_json)
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)

    benchmark.main()

    result = saved_payloads[0]["results"][0]
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "pool failed"


def test_main_with_multiple_results_invokes_plots_and_prints_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "results"
    plot_dir = tmp_path / "plots"
    make_plots_calls = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_log_prob_compilation.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "analysis",
            "likelihood",
            "--output-dir",
            str(output_dir),
            "--plot",
            "--plot-dir",
            str(plot_dir),
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
        lambda *args, **kwargs: make_plots_calls.append(kwargs),
    )

    benchmark.main()

    output = capsys.readouterr().out
    assert len(make_plots_calls) == 1
    assert len(make_plots_calls[0]["results"]) == 2
    assert make_plots_calls[0]["plot_dir"] == plot_dir
    assert f"Saved plots to {plot_dir}" in output


def test_module_main_guard_rejects_invalid_n_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_log_prob_compilation.py", "--n-runs", "0"],
    )

    with pytest.raises(ValueError, match="--n-runs must be at least 1"):
        runpy.run_module("src.run_log_prob_compilation", run_name="__main__")
