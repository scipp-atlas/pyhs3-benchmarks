from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import run_model_creation as benchmark


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


@pytest.fixture
def valid_model() -> SimpleNamespace:
    return SimpleNamespace(
        log_prob=lambda *args, **kwargs: 0.0,
        data={"obs": 1.0},
        free_params=["mu"],
    )


@pytest.fixture
def valid_result() -> dict[str, Any]:
    return {
        "benchmark": "model_creation",
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
        "model_type": "SimpleNamespace",
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


def test_validate_model_success(valid_model: SimpleNamespace) -> None:
    result = benchmark.validate_model(valid_model)

    assert result == {
        "model_type": "SimpleNamespace",
    }


def test_validate_model_rejects_none() -> None:
    with pytest.raises(ValueError, match="Model creation returned None"):
        benchmark.validate_model(None)


def test_measure_model_creation_memory_success(
    monkeypatch: pytest.MonkeyPatch,
    valid_model: SimpleNamespace,
) -> None:
    workspace = SimpleNamespace()
    created_models = []
    current_rss_values = iter([100.0, 104.5])
    peak_rss_values = iter([120.0, 130.0])

    def fake_create_model(workspace: Any, target: str, mode: str) -> SimpleNamespace:
        model = SimpleNamespace(
            log_prob=lambda *args, **kwargs: 0.0,
            data={"obs": 1.0},
            free_params=["mu"],
        )
        created_models.append(model)
        return model

    monkeypatch.setattr(benchmark, "create_model", fake_create_model)
    monkeypatch.setattr(
        benchmark, "get_current_rss_mb", lambda: next(current_rss_values)
    )
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: next(peak_rss_values))

    model, memory_summary = benchmark.measure_model_creation_memory(
        workspace=workspace,
        target="analysis",
        mode="FAST_RUN",
    )

    assert model is created_models[1]
    assert len(created_models) == 2
    assert memory_summary == {
        "memory_n_runs": 1,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 104.5,
        "current_rss_delta_mb": 4.5,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 130.0,
        "peak_rss_delta_mb": 10.0,
    }


def test_measure_model_creation_memory_propagates_create_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_create_model(workspace: Any, target: str, mode: str) -> None:
        raise RuntimeError("model creation failed")

    monkeypatch.setattr(benchmark, "create_model", failing_create_model)

    with pytest.raises(RuntimeError, match="model creation failed"):
        benchmark.measure_model_creation_memory(
            workspace=SimpleNamespace(),
            target="analysis",
            mode="FAST_RUN",
        )


def test_measure_model_creation_timing_success(
    monkeypatch: pytest.MonkeyPatch,
    valid_model: SimpleNamespace,
) -> None:
    perf_counter_values = iter([1.0, 1.2, 2.0, 2.3, 3.0, 3.4])
    create_calls = []

    def fake_create_model(workspace: Any, target: str, mode: str) -> SimpleNamespace:
        create_calls.append((workspace, target, mode))
        return valid_model

    monkeypatch.setattr(
        benchmark.time, "perf_counter", lambda: next(perf_counter_values)
    )
    monkeypatch.setattr(benchmark, "create_model", fake_create_model)

    workspace = SimpleNamespace()
    timings = benchmark.measure_model_creation_timing(
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


def test_measure_model_creation_timing_propagates_create_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_create_model(workspace: Any, target: str, mode: str) -> None:
        raise RuntimeError("timing failed")

    monkeypatch.setattr(benchmark, "create_model", failing_create_model)

    with pytest.raises(RuntimeError, match="timing failed"):
        benchmark.measure_model_creation_timing(
            workspace=SimpleNamespace(),
            target="analysis",
            mode="FAST_RUN",
            n_runs=1,
        )


def test_run_single_benchmark_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    valid_model: SimpleNamespace,
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
    timings = [0.1, 0.2, 0.3]
    timing_summary = {
        "wall_time_seconds_mean": 0.2,
        "wall_time_seconds_median": 0.2,
        "wall_time_seconds_std": 0.08165,
    }

    monkeypatch.setattr(benchmark, "load_workspace", lambda path: workspace)
    monkeypatch.setattr(
        benchmark,
        "measure_model_creation_memory",
        lambda workspace, target, mode: (valid_model, memory_summary),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_model_creation_timing",
        lambda workspace, target, mode, n_runs: timings,
    )
    monkeypatch.setattr(benchmark, "summarize_timings", lambda samples: timing_summary)

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_RUN",
        n_runs=3,
    )

    assert result["benchmark"] == "model_creation"
    assert result["workspace"] == "workspace.json"
    assert result["workspace_path"] == str(workspace_path)
    assert result["target"] == "analysis"
    assert result["mode"] == "FAST_RUN"
    assert result["n_runs"] == 3
    assert result["wall_time_seconds_samples"] == timings
    assert result["status"] == "success"
    assert result["wall_time_seconds_mean"] == 0.2
    assert result["model_type"] == "SimpleNamespace"


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
    valid_model: SimpleNamespace,
) -> None:
    monkeypatch.setattr(benchmark, "load_workspace", lambda path: SimpleNamespace())
    monkeypatch.setattr(
        benchmark,
        "measure_model_creation_memory",
        lambda workspace, target, mode: (valid_model, {}),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_model_creation_timing",
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
    valid_model: SimpleNamespace,
) -> None:
    monkeypatch.setattr(benchmark, "load_workspace", lambda path: SimpleNamespace())
    monkeypatch.setattr(
        benchmark,
        "measure_model_creation_memory",
        lambda workspace, target, mode: (valid_model, {}),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_model_creation_timing",
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

    assert "Model creation benchmark" in output
    assert "Timing" in output
    assert "Memory" in output
    assert "Validation" in output
    assert "workspace.json" in output
    assert "analysis" in output
    assert "FAST_RUN" in output


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_model_creation.py"])

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
            "run_model_creation.py",
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
            "--plot-name",
            "plot.png",
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
    assert args.plot_name == "plot.png"


def test_make_plots_calls_make_bar_plot_three_times(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    calls = []

    def fake_make_bar_plot(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(benchmark, "make_bar_plot", fake_make_bar_plot)

    second_result = valid_result.copy()
    second_result["workspace"] = "workspace_2.json"

    benchmark.make_plots(
        results=[valid_result, second_result],
        plot_dir=tmp_path,
        wall_time_plot_name="wall_time.png",
    )

    assert len(calls) == 3
    assert calls[0]["metric_key"] == "wall_time_seconds_mean"
    assert calls[1]["metric_key"] == "current_rss_delta_mb"
    assert calls[2]["metric_key"] == "peak_rss_delta_mb"


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
            "run_model_creation.py",
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
            "benchmark": "model_creation",
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
            "run_model_creation.py",
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
        ["run_model_creation.py", "--workspaces", str(missing_workspace)],
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
            "run_model_creation.py",
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
        lambda payload, output_path: output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        or output_path.write_text("{}"),
    )
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda *args, **kwargs: make_plots_calls.append(kwargs),
    )

    benchmark.main()

    assert len(make_plots_calls) == 1
    assert len(make_plots_calls[0]["results"]) == 1
    assert make_plots_calls[0]["plot_dir"] == benchmark.DEFAULT_PLOT_DIR
    assert make_plots_calls[0]["wall_time_plot_name"] == benchmark.DEFAULT_PLOT_NAME


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
            "run_model_creation.py",
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
        lambda payload, output_path: output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        or output_path.write_text("{}"),
    )
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda results, plot_dir, wall_time_plot_name: make_plots_calls.append(
            (results, plot_dir, wall_time_plot_name)
        ),
    )

    benchmark.main()

    assert len(make_plots_calls) == 1
    assert len(make_plots_calls[0][0]) == 2


def test_make_plots_skips_with_less_than_two_successes(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    failed_result = valid_result.copy()
    failed_result["status"] = "failed"

    benchmark.make_plots(
        results=[valid_result, failed_result],
        plot_dir=tmp_path,
        wall_time_plot_name="wall_time.png",
    )

    assert "Skipping plots" in capsys.readouterr().out
    assert not (tmp_path / "wall_time.png").exists()


def test_make_plots_filters_failed_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    calls = []
    second_result = valid_result.copy()
    second_result["workspace"] = "workspace_2.json"
    failed_result = valid_result.copy()
    failed_result["workspace"] = "failed.json"
    failed_result["status"] = "failed"

    monkeypatch.setattr(
        benchmark, "make_bar_plot", lambda **kwargs: calls.append(kwargs)
    )

    benchmark.make_plots(
        results=[valid_result, second_result, failed_result],
        plot_dir=tmp_path,
        wall_time_plot_name="wall_time.png",
    )

    assert len(calls) == 3
    assert all(len(call["results"]) == 2 for call in calls)
    assert all(result["status"] == "success" for result in calls[0]["results"])


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

    assert result["benchmark"] == "model_creation"
    assert result["workspace"] == "workspace.json"
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "boom"
    assert any("RuntimeError: boom" in line for line in result["traceback"])


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
    assert "Model creation benchmark FAILED" in output
    assert "RuntimeError: boom" in output
    assert "workspace.json" in output


def test_main_rejects_empty_target(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_creation.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "",
        ],
    )

    with pytest.raises(
        ValueError, match="--targets must contain only non-empty strings"
    ):
        benchmark.main()


def test_main_rejects_empty_mode(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_creation.py",
            "--workspaces",
            str(workspace_path),
            "--modes",
            "",
        ],
    )

    with pytest.raises(ValueError, match="--modes must contain only non-empty strings"):
        benchmark.main()


def test_main_records_pool_error_as_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
) -> None:
    output_dir = tmp_path / "results"
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
            "run_model_creation.py",
            "--workspaces",
            str(workspace_path),
            "--output-dir",
            str(output_dir),
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


def test_main_propagates_save_json_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_creation.py",
            "--workspaces",
            str(workspace_path),
            "--output-dir",
            str(tmp_path / "results"),
        ],
    )
    monkeypatch.setattr(
        benchmark, "get_context", lambda method: FakeContext(valid_result)
    )
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, output_path: (_ for _ in ()).throw(OSError("save failed")),
    )

    with pytest.raises(OSError, match="save failed"):
        benchmark.main()


def test_main_propagates_verify_output_file_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_creation.py",
            "--workspaces",
            str(workspace_path),
            "--output-dir",
            str(tmp_path / "results"),
        ],
    )
    monkeypatch.setattr(
        benchmark, "get_context", lambda method: FakeContext(valid_result)
    )
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, output_path: output_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        or output_path.write_text("{}"),
    )
    monkeypatch.setattr(
        benchmark,
        "verify_output_file",
        lambda output_path: (_ for _ in ()).throw(FileNotFoundError("verify failed")),
    )

    with pytest.raises(FileNotFoundError, match="verify failed"):
        benchmark.main()


def test_run_single_benchmark_real_workspace() -> None:
    workspace_path = Path("inputs/simple_workspace.json")

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="default_domain",
        mode="FAST_RUN",
        n_runs=1,
    )

    assert result["status"] == "success"
    assert result["target"] == "default_domain"
    assert result["mode"] == "FAST_RUN"
    assert result["n_distributions"] > 0 if "n_distributions" in result else True
    assert result["wall_time_seconds_mean"] > 0
    assert result["current_rss_before_mb"] >= 0
    assert result["current_rss_after_mb"] >= 0
    assert result["peak_rss_before_mb"] >= 0
    assert result["peak_rss_after_mb"] >= 0


def test_main_real_run_writes_output_json_and_uses_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_name = "model_creation_result.json"
    output_path = output_dir / output_name

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_creation.py",
            "--workspaces",
            "inputs/simple_workspace.json",
            "--targets",
            "default_domain",
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

    assert payload["benchmark"] == "model_creation"
    assert payload["n_results"] == 1
    assert len(payload["results"]) == 1
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][0]["workspace"] == "simple_workspace.json"
    assert payload["results"][0]["mode"] == "FAST_RUN"
    assert payload["results"][0]["target"] == "default_domain"


def test_make_plots_real_png_files_created(
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    second_result = valid_result.copy()
    second_result["workspace"] = "workspace_2.json"

    benchmark.make_plots(
        results=[valid_result, second_result],
        plot_dir=tmp_path,
        wall_time_plot_name="model_creation_wall_time.png",
    )

    assert (tmp_path / "model_creation_wall_time.png").exists()
    assert (tmp_path / "model_creation_current_rss_delta.png").exists()
    assert (tmp_path / "model_creation_peak_rss_delta.png").exists()
