from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytensor.tensor as pt

from src import run_graph_optimization as benchmark


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


@pytest.fixture
def fake_fgraph() -> SimpleNamespace:
    return SimpleNamespace(
        inputs=[SimpleNamespace(name="x")],
        outputs=[SimpleNamespace(name="out")],
        apply_nodes=[object(), object(), object()],
    )


@pytest.fixture
def valid_result() -> dict[str, Any]:
    return {
        "benchmark": "graph_optimization",
        "workspace": "workspace.json",
        "workspace_path": "/tmp/workspace.json",
        "target": "L_ch0",
        "mode": "FAST_RUN",
        "n_runs": 3,
        "timings_seconds": [0.1, 0.2, 0.3],
        "wall_time_seconds_mean": 0.2,
        "wall_time_seconds_median": 0.2,
        "wall_time_seconds_std": 0.08165,
        "memory_n_runs": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 104.0,
        "current_rss_delta_mb": 4.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 127.0,
        "peak_rss_delta_mb": 7.0,
        "status": "success",
        "fgraph_type": "FunctionGraph",
        "n_graph_inputs": 1,
        "n_graph_outputs": 1,
        "n_apply_nodes_before": 4,
        "n_apply_nodes_after": 3,
        "apply_node_delta": -1,
        "optimizer": "JAX",
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
    benchmark.validate_benchmark_config(target="L_ch0", mode="FAST_RUN", n_runs=1)


@pytest.mark.parametrize(
    ("target", "mode", "n_runs", "message"),
    [
        ("", "FAST_RUN", 1, "target must be a non-empty string"),
        ("L_ch0", "", 1, "mode must be a non-empty string"),
        ("L_ch0", "FAST_RUN", 0, "n_runs must be at least 1"),
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
    benchmark.validate_timings([0.1, 0.2])


def test_validate_timings_empty() -> None:
    with pytest.raises(ValueError, match="Timing samples are empty"):
        benchmark.validate_timings([])


@pytest.mark.parametrize("timings", [[0.1, 0.0], [0.1, -0.2]])
def test_validate_timings_rejects_non_positive_values(timings: list[float]) -> None:
    with pytest.raises(ValueError, match="All timing samples must be positive"):
        benchmark.validate_timings(timings)


def test_verify_output_file_success(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("{}")
    benchmark.verify_output_file(path)


def test_verify_output_file_missing(tmp_path: Path) -> None:
    with pytest.raises(
        FileNotFoundError, match="Benchmark output file was not created"
    ):
        benchmark.verify_output_file(tmp_path / "missing.json")


def test_verify_output_file_directory_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Benchmark output path is not a file"):
        benchmark.verify_output_file(tmp_path)


def test_build_function_graph_success() -> None:
    x = pt.scalar("x")
    log_prob = x + 1

    fgraph = benchmark.build_function_graph(log_prob)

    assert len(fgraph.outputs) == 1
    assert len(fgraph.inputs) == 1
    assert fgraph.inputs[0].name == "x"


def test_build_function_graph_rejects_none() -> None:
    with pytest.raises(ValueError, match="log_prob must not be None"):
        benchmark.build_function_graph(None)


def test_optimize_graph_success(
    monkeypatch: pytest.MonkeyPatch, fake_fgraph: SimpleNamespace
) -> None:
    calls = []

    class FakeOptimizer:
        def rewrite(self, fgraph: Any) -> None:
            calls.append(fgraph)

    fake_jax_mode = SimpleNamespace(optimizer=FakeOptimizer())
    monkeypatch.setattr(benchmark._ptmode, "JAX", fake_jax_mode)

    result = benchmark.optimize_graph(fake_fgraph)  # type: ignore[arg-type]

    assert result is fake_fgraph
    assert calls == [fake_fgraph]


def test_optimize_graph_rejects_none() -> None:
    with pytest.raises(ValueError, match="FunctionGraph must not be None"):
        benchmark.optimize_graph(None)  # type: ignore[arg-type]


def test_optimize_graph_propagates_rewrite_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_fgraph: SimpleNamespace,
) -> None:
    class FakeOptimizer:
        def rewrite(self, fgraph: Any) -> None:
            raise RuntimeError("rewrite failed")

    fake_jax_mode = SimpleNamespace(optimizer=FakeOptimizer())
    monkeypatch.setattr(benchmark._ptmode, "JAX", fake_jax_mode)

    with pytest.raises(RuntimeError, match="rewrite failed"):
        benchmark.optimize_graph(fake_fgraph)  # type: ignore[arg-type]


def test_validate_optimized_graph_success(fake_fgraph: SimpleNamespace) -> None:
    result = benchmark.validate_optimized_graph(
        fake_fgraph,  # type: ignore[arg-type]
        n_apply_nodes_before=5,
    )

    assert result == {
        "fgraph_type": "SimpleNamespace",
        "n_graph_inputs": 1,
        "n_graph_outputs": 1,
        "n_apply_nodes_before": 5,
        "n_apply_nodes_after": 3,
        "apply_node_delta": -2,
        "optimizer": "JAX",
    }


def test_validate_optimized_graph_rejects_none() -> None:
    with pytest.raises(ValueError, match="Optimized graph is None"):
        benchmark.validate_optimized_graph(None, 1)  # type: ignore[arg-type]


def test_validate_optimized_graph_rejects_negative_before(
    fake_fgraph: SimpleNamespace,
) -> None:
    with pytest.raises(ValueError, match="n_apply_nodes_before must be non-negative"):
        benchmark.validate_optimized_graph(fake_fgraph, -1)  # type: ignore[arg-type]


def test_validate_optimized_graph_rejects_multiple_outputs(
    fake_fgraph: SimpleNamespace,
) -> None:
    fake_fgraph.outputs = [object(), object()]

    with pytest.raises(ValueError, match="Expected one graph output"):
        benchmark.validate_optimized_graph(fake_fgraph, 3)  # type: ignore[arg-type]


def test_validate_optimized_graph_rejects_no_apply_nodes(
    fake_fgraph: SimpleNamespace,
) -> None:
    fake_fgraph.apply_nodes = []

    with pytest.raises(ValueError, match="Optimized graph has no apply nodes"):
        benchmark.validate_optimized_graph(fake_fgraph, 3)  # type: ignore[arg-type]


def test_measure_graph_optimization_memory_success(
    monkeypatch: pytest.MonkeyPatch,
    fake_fgraph: SimpleNamespace,
    workspace_path: Path,
) -> None:
    current_values = iter([100.0, 103.5])
    peak_values = iter([120.0, 125.0])
    log_prob = object()

    monkeypatch.setattr(
        benchmark, "build_log_prob", lambda **kwargs: (object(), log_prob)
    )
    monkeypatch.setattr(benchmark, "build_function_graph", lambda log_prob: fake_fgraph)
    monkeypatch.setattr(benchmark, "optimize_graph", lambda fgraph: fgraph)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: next(current_values))
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: next(peak_values))

    fgraph, summary = benchmark.measure_graph_optimization_memory(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
    )

    assert fgraph is fake_fgraph
    assert summary == {
        "memory_n_runs": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 103.5,
        "current_rss_delta_mb": 3.5,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 125.0,
        "peak_rss_delta_mb": 5.0,
        "n_apply_nodes_before": 3,
    }


def test_measure_graph_optimization_memory_propagates_build_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_build_log_prob(**kwargs: Any) -> None:
        raise RuntimeError("build failed")

    monkeypatch.setattr(benchmark, "build_log_prob", failing_build_log_prob)

    with pytest.raises(RuntimeError, match="build failed"):
        benchmark.measure_graph_optimization_memory(workspace_path, "L_ch0", "FAST_RUN")


def test_measure_graph_optimization_memory_propagates_optimize_error(
    monkeypatch: pytest.MonkeyPatch,
    fake_fgraph: SimpleNamespace,
    workspace_path: Path,
) -> None:
    monkeypatch.setattr(
        benchmark, "build_log_prob", lambda **kwargs: (object(), object())
    )
    monkeypatch.setattr(benchmark, "build_function_graph", lambda log_prob: fake_fgraph)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 120.0)

    def failing_optimize_graph(fgraph: Any) -> None:
        raise RuntimeError("optimize failed")

    monkeypatch.setattr(benchmark, "optimize_graph", failing_optimize_graph)

    with pytest.raises(RuntimeError, match="optimize failed"):
        benchmark.measure_graph_optimization_memory(workspace_path, "L_ch0", "FAST_RUN")


def test_measure_graph_optimization_timing_success(
    monkeypatch: pytest.MonkeyPatch,
    fake_fgraph: SimpleNamespace,
    workspace_path: Path,
) -> None:
    perf_values = iter([1.0, 1.1, 2.0, 2.3, 3.0, 3.4])
    build_calls = []
    optimize_calls = []

    def fake_build_log_prob(**kwargs: Any) -> tuple[object, object]:
        build_calls.append(kwargs)
        return object(), object()

    def fake_optimize_graph(fgraph: Any) -> Any:
        optimize_calls.append(fgraph)
        return fgraph

    monkeypatch.setattr(benchmark, "build_log_prob", fake_build_log_prob)
    monkeypatch.setattr(benchmark, "build_function_graph", lambda log_prob: fake_fgraph)
    monkeypatch.setattr(benchmark, "optimize_graph", fake_optimize_graph)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(perf_values))

    timings = benchmark.measure_graph_optimization_timing(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=3,
    )

    assert timings == pytest.approx([0.1, 0.3, 0.4])
    assert len(build_calls) == 3
    assert optimize_calls == [fake_fgraph, fake_fgraph, fake_fgraph]


def test_measure_graph_optimization_timing_rejects_invalid_n_runs(
    workspace_path: Path,
) -> None:
    with pytest.raises(ValueError, match="n_runs must be at least 1"):
        benchmark.measure_graph_optimization_timing(
            workspace_path, "L_ch0", "FAST_RUN", 0
        )


def test_measure_graph_optimization_timing_propagates_build_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_build_log_prob(**kwargs: Any) -> None:
        raise RuntimeError("timing build failed")

    monkeypatch.setattr(benchmark, "build_log_prob", failing_build_log_prob)

    with pytest.raises(RuntimeError, match="timing build failed"):
        benchmark.measure_graph_optimization_timing(
            workspace_path, "L_ch0", "FAST_RUN", 1
        )


def test_run_single_benchmark_success(
    monkeypatch: pytest.MonkeyPatch,
    fake_fgraph: SimpleNamespace,
    workspace_path: Path,
) -> None:
    memory_summary = {
        "memory_n_runs": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 103.0,
        "current_rss_delta_mb": 3.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 124.0,
        "peak_rss_delta_mb": 4.0,
        "n_apply_nodes_before": 4,
    }
    timings = [0.1, 0.2, 0.3]
    timing_summary = {
        "wall_time_seconds_mean": 0.2,
        "wall_time_seconds_median": 0.2,
        "wall_time_seconds_std": 0.08165,
    }

    monkeypatch.setattr(
        benchmark,
        "measure_graph_optimization_memory",
        lambda workspace_path, target, mode: (fake_fgraph, memory_summary),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_graph_optimization_timing",
        lambda workspace_path, target, mode, n_runs: timings,
    )
    monkeypatch.setattr(benchmark, "summarize_timings", lambda samples: timing_summary)

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=3,
    )

    assert result["benchmark"] == "graph_optimization"
    assert result["workspace"] == "workspace.json"
    assert result["target"] == "L_ch0"
    assert result["n_runs"] == 3
    assert result["timings_seconds"] == timings
    assert result["status"] == "success"
    assert result["n_apply_nodes_before"] == 4
    assert result["n_apply_nodes_after"] == 3
    assert result["wall_time_seconds_mean"] == 0.2


def test_run_single_benchmark_rejects_invalid_config(workspace_path: Path) -> None:
    with pytest.raises(ValueError, match="n_runs must be at least 1"):
        benchmark.run_single_benchmark(workspace_path, "L_ch0", "FAST_RUN", 0)


def test_run_single_benchmark_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.run_single_benchmark(
            tmp_path / "missing.json", "L_ch0", "FAST_RUN", 1
        )


def test_run_single_benchmark_rejects_empty_timings(
    monkeypatch: pytest.MonkeyPatch,
    fake_fgraph: SimpleNamespace,
    workspace_path: Path,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "measure_graph_optimization_memory",
        lambda workspace_path, target, mode: (fake_fgraph, {"n_apply_nodes_before": 3}),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_graph_optimization_timing",
        lambda workspace_path, target, mode, n_runs: [],
    )

    with pytest.raises(ValueError, match="Timing samples are empty"):
        benchmark.run_single_benchmark(workspace_path, "L_ch0", "FAST_RUN", 1)


def test_run_single_benchmark_rejects_non_positive_timings(
    monkeypatch: pytest.MonkeyPatch,
    fake_fgraph: SimpleNamespace,
    workspace_path: Path,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "measure_graph_optimization_memory",
        lambda workspace_path, target, mode: (fake_fgraph, {"n_apply_nodes_before": 3}),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_graph_optimization_timing",
        lambda workspace_path, target, mode, n_runs: [0.1, 0.0],
    )

    with pytest.raises(ValueError, match="All timing samples must be positive"):
        benchmark.run_single_benchmark(workspace_path, "L_ch0", "FAST_RUN", 1)


def test_run_single_benchmark_propagates_memory_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_memory(**kwargs: Any) -> None:
        raise RuntimeError("memory failed")

    monkeypatch.setattr(benchmark, "measure_graph_optimization_memory", failing_memory)

    with pytest.raises(RuntimeError, match="memory failed"):
        benchmark.run_single_benchmark(workspace_path, "L_ch0", "FAST_RUN", 1)


def test_print_result_outputs_summary(
    capsys: pytest.CaptureFixture[str], valid_result: dict[str, Any]
) -> None:
    benchmark.print_result(valid_result)

    output = capsys.readouterr().out

    assert "Graph optimization benchmark" in output
    assert "Timing" in output
    assert "Memory" in output
    assert "Validation" in output
    assert "apply nodes before" in output
    assert "workspace.json" in output


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_graph_optimization.py"])

    args = benchmark.parse_args()

    assert args.workspaces == [benchmark.DEFAULT_WORKSPACE]
    assert args.targets == [benchmark.DEFAULT_TARGET]
    assert args.modes == [benchmark.DEFAULT_MODE]
    assert args.n_runs == benchmark.DEFAULT_N_RUNS
    assert args.output_dir == benchmark.DEFAULT_OUTPUT_DIR
    assert args.output_name == benchmark.DEFAULT_OUTPUT_NAME
    assert args.plot is False
    assert args.plot_dir == benchmark.DEFAULT_PLOT_DIR


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_graph_optimization.py",
            "--workspaces",
            "a.json",
            "b.json",
            "--targets",
            "L_ch0",
            "L_ch1",
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
    assert args.targets == ["L_ch0", "L_ch1"]
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
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda results, key: True)

    second = valid_result.copy()
    second["workspace"] = "workspace2.json"
    benchmark.make_plots(results=[valid_result, second], plot_dir=tmp_path)

    assert [call["metric_key"] for call in calls] == [
        "wall_time_seconds_mean",
        "current_rss_delta_mb",
        "peak_rss_delta_mb",
    ]


def test_make_plots_skips_optional_memory_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    calls = []

    def fake_make_bar_plot(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(benchmark, "make_bar_plot", fake_make_bar_plot)
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda results, key: False)

    second = valid_result.copy()
    second["workspace"] = "workspace2.json"
    benchmark.make_plots(results=[valid_result, second], plot_dir=tmp_path)

    assert [call["metric_key"] for call in calls] == ["wall_time_seconds_mean"]


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
            "run_graph_optimization.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "L_ch0",
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

    monkeypatch.setattr(benchmark, "save_json", fake_save_json)
    monkeypatch.setattr(
        benchmark, "verify_output_file", lambda path: verified_paths.append(path)
    )

    benchmark.main()

    assert saved_payloads == [
        {
            "benchmark": "graph_optimization",
            "n_results": 1,
            "results": [valid_result],
        }
    ]
    assert verified_paths == [output_dir / output_name]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--n-runs", "0"], "--n-runs must be at least 1"),
        (["--targets", ""], "--targets must contain only non-empty strings"),
        (["--modes", ""], "--modes must contain only non-empty strings"),
    ],
)
def test_main_rejects_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    argv: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_graph_optimization.py", "--workspaces", str(workspace_path), *argv],
    )

    with pytest.raises(ValueError, match=message):
        benchmark.main()


def test_main_rejects_missing_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_graph_optimization.py", "--workspaces", str(tmp_path / "missing.json")],
    )

    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.main()


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
            "run_graph_optimization.py",
            "--workspaces",
            str(workspace_path),
            "--output-dir",
            str(tmp_path / "results"),
            "--n-runs",
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
            "run_graph_optimization.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "L_ch0",
            "L_ch1",
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


def test_make_plots_real_png_files_created(
    tmp_path: Path, valid_result: dict[str, Any]
) -> None:
    second = valid_result.copy()
    second["workspace"] = "workspace2.json"
    benchmark.make_plots(results=[valid_result, second], plot_dir=tmp_path)

    assert (tmp_path / "graph_optimization_wall_time.png").exists()
    assert (tmp_path / "graph_optimization_current_rss_delta.png").exists()
    assert (tmp_path / "graph_optimization_peak_rss_delta.png").exists()


def test_run_single_benchmark_real_workspace() -> None:
    result = benchmark.run_single_benchmark(
        workspace_path=Path("inputs/simple_workspace.json"),
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=1,
    )

    assert result["status"] == "success"
    assert result["benchmark"] == "graph_optimization"
    assert result["workspace"] == "simple_workspace.json"
    assert result["target"] == "L_ch0"
    assert result["mode"] == "FAST_RUN"
    assert result["n_runs"] == 1
    assert result["wall_time_seconds_mean"] > 0
    assert result["n_graph_outputs"] == 1
    assert result["n_apply_nodes_before"] > 0
    assert result["n_apply_nodes_after"] > 0
    assert result["current_rss_before_mb"] >= 0
    assert result["peak_rss_before_mb"] >= 0


def test_main_real_run_writes_output_json_and_uses_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_name = "graph_optimization_result.json"
    output_path = output_dir / output_name

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_graph_optimization.py",
            "--workspaces",
            "inputs/simple_workspace.json",
            "--targets",
            "L_ch0",
            "--modes",
            "FAST_RUN",
            "--n-runs",
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

    assert payload["benchmark"] == "graph_optimization"
    assert payload["n_results"] == 1
    assert len(payload["results"]) == 1
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][0]["workspace"] == "simple_workspace.json"
    assert payload["results"][0]["target"] == "L_ch0"


def test_make_error_result_contains_structured_failure(workspace_path: Path) -> None:
    try:
        raise RuntimeError("optimization failed")
    except RuntimeError as exc:
        result = benchmark.make_error_result(
            workspace_path=workspace_path,
            target="L_ch0",
            mode="FAST_RUN",
            n_runs=2,
            exc=exc,
        )

    assert result["benchmark"] == "graph_optimization"
    assert result["workspace"] == "workspace.json"
    assert result["workspace_path"] == str(workspace_path)
    assert result["target"] == "L_ch0"
    assert result["mode"] == "FAST_RUN"
    assert result["n_runs"] == 2
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "optimization failed"
    assert "RuntimeError" in result["traceback"]


def test_print_error_result_outputs_summary(
    capsys: pytest.CaptureFixture[str],
    workspace_path: Path,
) -> None:
    result = benchmark.make_error_result(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=1,
        exc=ValueError("bad graph"),
    )

    benchmark.print_error_result(result)

    output = capsys.readouterr().out
    assert "Graph optimization benchmark FAILED" in output
    assert "workspace.json" in output
    assert "L_ch0" in output
    assert "FAST_RUN" in output
    assert "ValueError: bad graph" in output


def test_make_plots_skips_when_fewer_than_two_successful_results(
    tmp_path: Path,
    valid_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark.make_plots(results=[valid_result], plot_dir=tmp_path)

    output = capsys.readouterr().out
    assert "Skipping plots" in output
    assert not any(tmp_path.glob("*.png"))


def test_main_records_pool_error_as_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "results"
    output_name = "result.json"
    saved_payloads = []

    class FailingPool:
        def __enter__(self) -> "FailingPool":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def apply(self, func: Any, args: tuple[Any, ...]) -> None:
            raise RuntimeError("pool failed")

    class FailingContext:
        def Pool(self, processes: int) -> FailingPool:
            assert processes == 1
            return FailingPool()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_graph_optimization.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "L_ch0",
            "--modes",
            "FAST_RUN",
            "--n-runs",
            "1",
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
        ],
    )
    monkeypatch.setattr(benchmark, "get_context", lambda method: FailingContext())
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)

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
    assert result["error_message"] == "pool failed"
    assert "Graph optimization benchmark FAILED" in capsys.readouterr().out


def test_module_main_guard_runs_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_graph_optimization.py", "--n-runs", "0"],
    )

    with pytest.raises(ValueError, match="--n-runs must be at least 1"):
        runpy.run_module("src.run_graph_optimization", run_name="__main__")
