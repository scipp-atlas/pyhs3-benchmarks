from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytensor.tensor as pt
from pytensor.tensor.variable import TensorVariable

from src import run_log_prob_construction as benchmark


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


@pytest.fixture
def valid_log_prob() -> TensorVariable:
    return pt.vector("log_prob")


@pytest.fixture
def valid_result() -> dict[str, Any]:
    return {
        "benchmark": "log_prob_construction",
        "workspace": "workspace.json",
        "workspace_path": "/tmp/workspace.json",
        "target": "analysis",
        "mode": "FAST_RUN",
        "n_runs": 3,
        "wall_time_seconds_samples": [0.1, 0.2, 0.3],
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
        "log_prob_type": "TensorVariable",
        "log_prob_name": "log_prob",
        "log_prob_ndim": 1,
        "log_prob_dtype": "float64",
        "can_proceed_to_compilation": True,
    }


def _first_collection_name(collection: Any) -> str | None:
    if collection is None:
        return None

    if hasattr(collection, "_map"):
        keys = list(collection._map.keys())
        if keys:
            return str(keys[0])

    if hasattr(collection, "keys"):
        try:
            keys = list(collection.keys())
        except TypeError:
            keys = []
        if keys:
            return str(keys[0])

    try:
        items = list(collection)
    except TypeError:
        items = []

    for item in items:
        name = getattr(item, "name", None)
        if name:
            return str(name)

    return None


def _real_likelihood_target() -> str:
    workspace = benchmark.load_workspace(Path("inputs/simple_workspace.json"))

    target = _first_collection_name(getattr(workspace, "analyses", None))
    if target is not None:
        return target

    target = _first_collection_name(getattr(workspace, "likelihoods", None))
    if target is not None:
        return target

    pytest.skip("inputs/simple_workspace.json does not expose analyses or likelihoods")


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


def test_construct_log_prob_returns_model_log_prob(
    valid_log_prob: TensorVariable,
) -> None:
    model = SimpleNamespace(log_prob=valid_log_prob)

    assert benchmark.construct_log_prob(model) is valid_log_prob


def test_construct_log_prob_propagates_property_error() -> None:
    class FailingModel:
        @property
        def log_prob(self) -> TensorVariable:
            raise RuntimeError("log_prob failed")

    with pytest.raises(RuntimeError, match="log_prob failed"):
        benchmark.construct_log_prob(FailingModel())


def test_validate_log_prob_success(valid_log_prob: TensorVariable) -> None:
    result = benchmark.validate_log_prob(valid_log_prob)

    assert result["log_prob_type"] == "TensorVariable"
    assert result["log_prob_name"] == str(valid_log_prob)
    assert result["log_prob_ndim"] == valid_log_prob.ndim
    assert result["log_prob_dtype"] == str(valid_log_prob.dtype)
    assert result["can_proceed_to_compilation"] is True


def test_validate_log_prob_rejects_none() -> None:
    with pytest.raises(ValueError, match="log_prob construction returned None"):
        benchmark.validate_log_prob(None)


def test_validate_log_prob_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="Expected TensorVariable, got str"):
        benchmark.validate_log_prob("not a tensor")


def test_measure_log_prob_construction_memory_success(
    monkeypatch: pytest.MonkeyPatch,
    valid_log_prob: TensorVariable,
) -> None:
    current_rss_values = iter([100.0, 104.5])
    peak_rss_values = iter([120.0, 130.0])
    model = SimpleNamespace(log_prob=valid_log_prob)

    monkeypatch.setattr(
        benchmark, "get_current_rss_mb", lambda: next(current_rss_values)
    )
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: next(peak_rss_values))

    log_prob, memory_summary = benchmark.measure_log_prob_construction_memory(model)

    assert log_prob is valid_log_prob
    assert memory_summary == {
        "memory_n_runs": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 104.5,
        "current_rss_delta_mb": 4.5,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 130.0,
        "peak_rss_delta_mb": 10.0,
    }


def test_measure_log_prob_construction_memory_propagates_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel:
        @property
        def log_prob(self) -> TensorVariable:
            raise RuntimeError("memory failed")

    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 120.0)

    with pytest.raises(RuntimeError, match="memory failed"):
        benchmark.measure_log_prob_construction_memory(FailingModel())


def test_measure_log_prob_construction_timing_success(
    monkeypatch: pytest.MonkeyPatch,
    valid_log_prob: TensorVariable,
) -> None:
    perf_counter_values = iter([1.0, 1.2, 2.0, 2.3, 3.0, 3.4])
    create_calls = []

    def fake_create_model(workspace: Any, target: str, mode: str) -> SimpleNamespace:
        create_calls.append((workspace, target, mode))
        return SimpleNamespace(log_prob=valid_log_prob)

    monkeypatch.setattr(
        benchmark.time, "perf_counter", lambda: next(perf_counter_values)
    )
    monkeypatch.setattr(benchmark, "create_model", fake_create_model)

    workspace = SimpleNamespace()
    timings = benchmark.measure_log_prob_construction_timing(
        workspace=workspace,
        target="analysis",
        mode="FAST_RUN",
        n_runs=3,
    )

    assert timings == pytest.approx([0.2, 0.3, 0.4])
    assert create_calls == [
        (workspace, "analysis", "FAST_RUN"),
        (workspace, "analysis", "FAST_RUN"),
        (workspace, "analysis", "FAST_RUN"),
    ]


def test_measure_log_prob_construction_timing_propagates_create_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_create_model(workspace: Any, target: str, mode: str) -> None:
        raise RuntimeError("create failed")

    monkeypatch.setattr(benchmark, "create_model", failing_create_model)

    with pytest.raises(RuntimeError, match="create failed"):
        benchmark.measure_log_prob_construction_timing(
            workspace=SimpleNamespace(),
            target="analysis",
            mode="FAST_RUN",
            n_runs=1,
        )


def test_measure_log_prob_construction_timing_propagates_log_prob_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel:
        @property
        def log_prob(self) -> TensorVariable:
            raise RuntimeError("log_prob timing failed")

    monkeypatch.setattr(
        benchmark,
        "create_model",
        lambda workspace, target, mode: FailingModel(),
    )

    with pytest.raises(RuntimeError, match="log_prob timing failed"):
        benchmark.measure_log_prob_construction_timing(
            workspace=SimpleNamespace(),
            target="analysis",
            mode="FAST_RUN",
            n_runs=1,
        )


def test_run_single_benchmark_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    valid_log_prob: TensorVariable,
) -> None:
    workspace = SimpleNamespace()
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
        "log_prob_type": "TensorVariable",
        "log_prob_name": "log_prob",
        "log_prob_ndim": 1,
        "log_prob_dtype": "float64",
        "can_proceed_to_compilation": True,
    }
    timings = [0.1, 0.2, 0.3]
    timing_summary = {
        "wall_time_seconds_mean": 0.2,
        "wall_time_seconds_median": 0.2,
        "wall_time_seconds_std": 0.08165,
    }

    monkeypatch.setattr(benchmark, "load_workspace", lambda path: workspace)
    monkeypatch.setattr(
        benchmark,
        "create_model",
        lambda workspace, target, mode: SimpleNamespace(log_prob=valid_log_prob),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_log_prob_construction_memory",
        lambda model: (valid_log_prob, memory_summary),
    )
    monkeypatch.setattr(
        benchmark, "validate_log_prob", lambda log_prob: validation_summary
    )
    monkeypatch.setattr(
        benchmark,
        "measure_log_prob_construction_timing",
        lambda workspace, target, mode, n_runs: timings,
    )
    monkeypatch.setattr(benchmark, "summarize_timings", lambda samples: timing_summary)

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        n_runs=3,
    )

    assert result["benchmark"] == "log_prob_construction"
    assert result["workspace"] == "workspace.json"
    assert result["workspace_path"] == str(workspace_path)
    assert result["target"] == "analysis"
    assert result["mode"] == "FAST_RUN"
    assert result["n_runs"] == 3
    assert result["wall_time_seconds_samples"] == timings
    assert result["status"] == "success"
    assert result["wall_time_seconds_mean"] == 0.2
    assert result["log_prob_type"] == "TensorVariable"


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
    valid_log_prob: TensorVariable,
) -> None:
    monkeypatch.setattr(benchmark, "load_workspace", lambda path: SimpleNamespace())
    monkeypatch.setattr(
        benchmark,
        "create_model",
        lambda workspace, target, mode: SimpleNamespace(log_prob=valid_log_prob),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_log_prob_construction_memory",
        lambda model: (valid_log_prob, {}),
    )
    monkeypatch.setattr(benchmark, "validate_log_prob", lambda log_prob: {})
    monkeypatch.setattr(
        benchmark,
        "measure_log_prob_construction_timing",
        lambda workspace, target, mode, n_runs: [],
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
    valid_log_prob: TensorVariable,
) -> None:
    monkeypatch.setattr(benchmark, "load_workspace", lambda path: SimpleNamespace())
    monkeypatch.setattr(
        benchmark,
        "create_model",
        lambda workspace, target, mode: SimpleNamespace(log_prob=valid_log_prob),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_log_prob_construction_memory",
        lambda model: (valid_log_prob, {}),
    )
    monkeypatch.setattr(benchmark, "validate_log_prob", lambda log_prob: {})
    monkeypatch.setattr(
        benchmark,
        "measure_log_prob_construction_timing",
        lambda workspace, target, mode, n_runs: [0.1, 0.0],
    )

    with pytest.raises(ValueError, match="All timing samples must be positive"):
        benchmark.run_single_benchmark(
            workspace_path=workspace_path,
            target="analysis",
            mode="FAST_RUN",
            n_runs=3,
        )


def test_run_single_benchmark_propagates_load_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_load_workspace(path: Path) -> None:
        raise RuntimeError("load failed")

    monkeypatch.setattr(benchmark, "load_workspace", failing_load_workspace)

    with pytest.raises(RuntimeError, match="load failed"):
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

    assert "log_prob construction benchmark" in output
    assert "Timing" in output
    assert "Memory" in output
    assert "Validation" in output
    assert "workspace.json" in output
    assert "analysis" in output
    assert "FAST_RUN" in output
    assert "TensorVariable" in output


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_log_prob_construction.py"])

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
            "run_log_prob_construction.py",
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

    assert len(calls) == 1
    assert calls[0]["metric_key"] == "wall_time_seconds_mean"


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
            "run_log_prob_construction.py",
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
            "benchmark": "log_prob_construction",
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
            "run_log_prob_construction.py",
            "--workspaces",
            str(workspace_path),
            "--n-runs",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="--n-runs must be at least 1"):
        benchmark.main()


def test_main_rejects_missing_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing_workspace = tmp_path / "missing.json"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_log_prob_construction.py", "--workspaces", str(missing_workspace)],
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
            "run_log_prob_construction.py",
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
        lambda *args, **kwargs: make_plots_calls.append(args),
    )

    benchmark.main()

    assert make_plots_calls == []


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
            "run_log_prob_construction.py",
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
    valid_log_prob: TensorVariable,
) -> None:
    workspace = SimpleNamespace()
    model = SimpleNamespace(log_prob=valid_log_prob)

    monkeypatch.setattr(benchmark, "load_workspace", lambda path: workspace)
    monkeypatch.setattr(
        benchmark,
        "create_model",
        lambda workspace, target, mode: model,
    )
    monkeypatch.setattr(
        benchmark,
        "measure_log_prob_construction_memory",
        lambda model: (
            valid_log_prob,
            {
                "memory_n_runs": 1,
                "current_rss_before_mb": 100.0,
                "current_rss_after_mb": 101.0,
                "current_rss_delta_mb": 1.0,
                "peak_rss_before_mb": 120.0,
                "peak_rss_after_mb": 121.0,
                "peak_rss_delta_mb": 1.0,
            },
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_log_prob_construction_timing",
        lambda workspace, target, mode, n_runs: [0.1],
    )
    monkeypatch.setattr(
        benchmark,
        "summarize_timings",
        lambda samples: {
            "wall_time_seconds_mean": 0.1,
            "wall_time_seconds_median": 0.1,
            "wall_time_seconds_std": 0.0,
        },
    )

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        n_runs=1,
    )

    assert result["status"] == "success"
    assert result["workspace"] == "workspace.json"
    assert result["target"] == "analysis"
    assert result["mode"] == "FAST_RUN"
    assert result["wall_time_seconds_mean"] == pytest.approx(0.1)
    assert result["log_prob_type"] == "TensorVariable"
    assert result["can_proceed_to_compilation"] is True


def test_main_mocked_run_writes_output_json_and_uses_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    output_dir = tmp_path / "results"
    output_name = "log_prob_construction_result.json"
    output_path = output_dir / output_name

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_log_prob_construction.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "analysis",
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
    monkeypatch.setattr(
        benchmark, "get_context", lambda method: FakeContext(valid_result)
    )
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)

    benchmark.main()

    assert output_path.exists()
    assert output_path.is_file()

    with output_path.open() as file:
        payload = json.load(file)

    assert payload["benchmark"] == "log_prob_construction"
    assert payload["n_results"] == 1
    assert len(payload["results"]) == 1
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][0]["workspace"] == "workspace.json"
    assert payload["results"][0]["target"] == "analysis"
    assert payload["results"][0]["mode"] == "FAST_RUN"


def test_make_plots_real_png_files_created(
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    second_result = dict(valid_result)
    second_result["workspace"] = "workspace2.json"
    second_result["workspace_path"] = "/tmp/workspace2.json"

    benchmark.make_plots(results=[valid_result, second_result], plot_dir=tmp_path)

    assert (tmp_path / "log_prob_construction_wall_time.png").exists()
    assert (tmp_path / "log_prob_construction_current_rss_delta.png").exists()
    assert (tmp_path / "log_prob_construction_peak_rss_delta.png").exists()


def test_make_error_result_contains_failed_payload(workspace_path: Path) -> None:
    exc = RuntimeError("boom")

    result = benchmark.make_error_result(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        n_runs=3,
        exc=exc,
    )

    assert result == {
        "benchmark": "log_prob_construction",
        "workspace": "workspace.json",
        "workspace_path": str(workspace_path),
        "target": "analysis",
        "mode": "FAST_RUN",
        "n_runs": 3,
        "status": "failed",
        "error_type": "RuntimeError",
        "error_message": "boom",
    }


def test_print_error_result_outputs_summary(
    capsys: pytest.CaptureFixture[str],
    workspace_path: Path,
) -> None:
    result = benchmark.make_error_result(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        n_runs=3,
        exc=RuntimeError("boom"),
    )

    benchmark.print_error_result(result)

    output = capsys.readouterr().out
    assert "log_prob construction benchmark FAILED" in output
    assert "workspace.json" in output
    assert "RuntimeError: boom" in output


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
        ["run_log_prob_construction.py", "--workspaces", str(workspace_path), *argv],
    )

    with pytest.raises(ValueError, match=message):
        benchmark.main()


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


def test_main_records_pool_error_as_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_name = "result.json"
    saved_payloads = []
    error_results = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_log_prob_construction.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "analysis",
            "--modes",
            "FAST_RUN",
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
        ],
    )
    monkeypatch.setattr(benchmark, "get_context", lambda method: FailingContext())
    monkeypatch.setattr(
        benchmark, "print_error_result", lambda result: error_results.append(result)
    )

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
    assert error_results == [result]


def test_main_with_multiple_results_invokes_plots_and_prints_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    plot_calls = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_log_prob_construction.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "analysis",
            "likelihood",
            "--output-dir",
            str(tmp_path / "results"),
            "--plot",
            "--plot-dir",
            str(tmp_path / "plots"),
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
        lambda results, plot_dir: plot_calls.append((results, plot_dir)),
    )

    benchmark.main()

    output = capsys.readouterr().out
    assert "Saved plots to" in output
    assert len(plot_calls) == 1
    assert len(plot_calls[0][0]) == 2
