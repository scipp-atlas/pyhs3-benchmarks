from __future__ import annotations

import sys
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src import run_workspace_loading as benchmark


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


@pytest.fixture
def valid_workspace() -> SimpleNamespace:
    return SimpleNamespace(
        metadata=SimpleNamespace(hs3_version="0.2"),
        distributions=["dist"],
        likelihoods=["likelihood"],
        data=["data"],
        domains=["domain"],
        parameter_points=["init"],
    )


@pytest.fixture
def valid_result() -> dict[str, Any]:
    return {
        "benchmark": "workspace_loading",
        "workspace": "workspace.json",
        "workspace_path": "/tmp/workspace.json",
        "n_runs": 3,
        "wall_time_seconds_samples": [0.1, 0.2, 0.3],
        "wall_time_seconds_mean": 0.2,
        "wall_time_seconds_median": 0.2,
        "wall_time_seconds_std": 0.08165,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 105.0,
        "current_rss_delta_mb": 5.0,
        "peak_rss_before_mb": 110.0,
        "peak_rss_after_mb": 120.0,
        "peak_rss_delta_mb": 10.0,
        "metadata_hs3_version": "0.2",
        "n_distributions": 1,
        "n_likelihoods": 1,
        "n_data": 1,
        "n_domains": 1,
        "n_parameter_points": 1,
        "status": "success",
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


def test_validate_workspace_success(valid_workspace: SimpleNamespace) -> None:
    result = benchmark.validate_workspace(valid_workspace)

    assert result == {
        "metadata_hs3_version": "0.2",
        "n_distributions": 1,
        "n_likelihoods": 1,
        "n_data": 1,
        "n_domains": 1,
        "n_parameter_points": 1,
    }


def test_validate_workspace_allows_missing_domains(
    valid_workspace: SimpleNamespace,
) -> None:
    valid_workspace.domains = None

    result = benchmark.validate_workspace(valid_workspace)

    assert result["n_domains"] == 0


def test_validate_workspace_allows_missing_parameter_points(
    valid_workspace: SimpleNamespace,
) -> None:
    valid_workspace.parameter_points = None

    result = benchmark.validate_workspace(valid_workspace)

    assert result["n_parameter_points"] == 0


def test_validate_workspace_rejects_missing_distributions(
    valid_workspace: SimpleNamespace,
) -> None:
    valid_workspace.distributions = None

    with pytest.raises(ValueError, match="Workspace does not contain distributions"):
        benchmark.validate_workspace(valid_workspace)


def test_validate_workspace_rejects_empty_distributions(
    valid_workspace: SimpleNamespace,
) -> None:
    valid_workspace.distributions = []

    with pytest.raises(ValueError, match="Workspace does not contain distributions"):
        benchmark.validate_workspace(valid_workspace)


def test_validate_workspace_rejects_missing_likelihoods(
    valid_workspace: SimpleNamespace,
) -> None:
    valid_workspace.likelihoods = None

    with pytest.raises(ValueError, match="Workspace does not contain likelihoods"):
        benchmark.validate_workspace(valid_workspace)


def test_validate_workspace_rejects_empty_likelihoods(
    valid_workspace: SimpleNamespace,
) -> None:
    valid_workspace.likelihoods = []

    with pytest.raises(ValueError, match="Workspace does not contain likelihoods"):
        benchmark.validate_workspace(valid_workspace)


def test_validate_workspace_rejects_missing_data(
    valid_workspace: SimpleNamespace,
) -> None:
    valid_workspace.data = None

    with pytest.raises(ValueError, match="Workspace does not contain data"):
        benchmark.validate_workspace(valid_workspace)


def test_validate_workspace_rejects_empty_data(
    valid_workspace: SimpleNamespace,
) -> None:
    valid_workspace.data = []

    with pytest.raises(ValueError, match="Workspace does not contain data"):
        benchmark.validate_workspace(valid_workspace)


def test_measure_single_load_memory_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    valid_workspace: SimpleNamespace,
) -> None:
    current_rss_values = iter([100.0, 104.5])
    peak_rss_values = iter([120.0, 130.0])

    monkeypatch.setattr(benchmark, "load_workspace", lambda path: valid_workspace)
    monkeypatch.setattr(
        benchmark, "get_current_rss_mb", lambda: next(current_rss_values)
    )
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: next(peak_rss_values))

    result = benchmark.measure_single_load_memory(workspace_path)

    assert result["current_rss_before_mb"] == 100.0
    assert result["current_rss_after_mb"] == 104.5
    assert result["current_rss_delta_mb"] == 4.5
    assert result["peak_rss_before_mb"] == 120.0
    assert result["peak_rss_after_mb"] == 130.0
    assert result["peak_rss_delta_mb"] == 10.0
    assert result["n_distributions"] == 1


def test_measure_single_load_memory_propagates_load_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_load_workspace(path: Path) -> None:
        raise RuntimeError("load failed")

    monkeypatch.setattr(benchmark, "load_workspace", failing_load_workspace)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 120.0)

    with pytest.raises(RuntimeError, match="load failed"):
        benchmark.measure_single_load_memory(workspace_path)


def test_run_single_benchmark_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    memory_summary = {
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 101.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 122.0,
        "peak_rss_delta_mb": 2.0,
        "metadata_hs3_version": "0.2",
        "n_distributions": 1,
        "n_likelihoods": 1,
        "n_data": 1,
        "n_domains": 0,
        "n_parameter_points": 0,
    }
    timings = [0.1, 0.2, 0.3]
    timing_summary = {
        "wall_time_seconds_mean": 0.2,
        "wall_time_seconds_median": 0.2,
        "wall_time_seconds_std": 0.08165,
    }

    monkeypatch.setattr(
        benchmark, "measure_single_load_memory", lambda path: memory_summary
    )
    monkeypatch.setattr(
        benchmark,
        "run_repeated_timing",
        lambda func, n_runs: (object(), timings),
    )
    monkeypatch.setattr(benchmark, "summarize_timings", lambda samples: timing_summary)

    result = benchmark.run_single_benchmark(workspace_path, n_runs=3)

    assert result["benchmark"] == "workspace_loading"
    assert result["workspace"] == "workspace.json"
    assert result["workspace_path"] == str(workspace_path)
    assert result["n_runs"] == 3
    assert result["wall_time_seconds_samples"] == timings
    assert result["status"] == "success"
    assert result["wall_time_seconds_mean"] == 0.2
    assert result["n_distributions"] == 1


def test_run_single_benchmark_rejects_invalid_n_runs(workspace_path: Path) -> None:
    result = benchmark.run_single_benchmark(workspace_path, n_runs=0)

    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"
    assert "n_runs must be at least 1" in result["error_message"]


def test_run_single_benchmark_rejects_empty_timings(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    monkeypatch.setattr(benchmark, "measure_single_load_memory", lambda path: {})
    monkeypatch.setattr(
        benchmark,
        "run_repeated_timing",
        lambda func, n_runs: (object(), []),
    )

    result = benchmark.run_single_benchmark(workspace_path, n_runs=3)

    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"
    assert "Timing samples are empty" in result["error_message"]


def test_run_single_benchmark_rejects_non_positive_timings(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    monkeypatch.setattr(benchmark, "measure_single_load_memory", lambda path: {})
    monkeypatch.setattr(
        benchmark,
        "run_repeated_timing",
        lambda func, n_runs: (object(), [0.1, 0.0]),
    )

    result = benchmark.run_single_benchmark(workspace_path, n_runs=3)

    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"
    assert "All timing samples must be positive" in result["error_message"]


def test_run_single_benchmark_propagates_timing_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_run_repeated_timing(func: Any, n_runs: int) -> None:
        raise RuntimeError("timing failed")

    monkeypatch.setattr(benchmark, "measure_single_load_memory", lambda path: {})
    monkeypatch.setattr(benchmark, "run_repeated_timing", failing_run_repeated_timing)

    result = benchmark.run_single_benchmark(workspace_path, n_runs=3)

    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert "timing failed" in result["error_message"]


def test_print_result_outputs_summary(
    capsys: pytest.CaptureFixture[str],
    valid_result: dict[str, Any],
) -> None:
    benchmark.print_result(valid_result)

    output = capsys.readouterr().out

    assert "Workspace loading benchmark" in output
    assert "Timing" in output
    assert "Memory" in output
    assert "Validation" in output
    assert "workspace.json" in output


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_workspace_loading.py"])

    args = benchmark.parse_args()

    assert args.workspaces == [benchmark.DEFAULT_WORKSPACE]
    assert args.n_runs == benchmark.DEFAULT_N_RUNS
    assert args.output_dir == benchmark.DEFAULT_OUTPUT_DIR
    assert args.output_name == benchmark.DEFAULT_OUTPUT_NAME
    assert args.plot is False


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_workspace_loading.py",
            "--workspaces",
            "a.json",
            "b.json",
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

    second_result = {**valid_result, "workspace": "workspace_2.json"}

    benchmark.make_plots(
        results=[valid_result, second_result],
        plot_dir=tmp_path,
        wall_time_plot_name="wall_time.png",
    )

    assert len(calls) == 3
    assert calls[0]["metric_key"] == "wall_time_seconds_mean"
    assert calls[1]["metric_key"] == "peak_rss_delta_mb"
    assert calls[2]["metric_key"] == "current_rss_delta_mb"


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
            "run_workspace_loading.py",
            "--workspaces",
            str(workspace_path),
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
            "benchmark": "workspace_loading",
            "n_workspaces": 1,
            "n_successful_workspaces": 1,
            "n_failed_workspaces": 0,
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
            "run_workspace_loading.py",
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
    saved_payloads = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_workspace_loading.py",
            "--workspaces",
            str(missing_workspace),
            "--output-dir",
            str(output_dir),
        ],
    )

    def fake_save_json(payload: dict[str, Any], output_path: Path) -> None:
        saved_payloads.append(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}")

    monkeypatch.setattr(benchmark, "save_json", fake_save_json)
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)

    benchmark.main()

    assert saved_payloads[0]["n_workspaces"] == 1
    assert saved_payloads[0]["n_successful_workspaces"] == 0
    assert saved_payloads[0]["n_failed_workspaces"] == 1
    assert saved_payloads[0]["results"][0]["status"] == "failed"
    assert saved_payloads[0]["results"][0]["error_type"] == "FileNotFoundError"
    assert (
        "Workspace file does not exist"
        in saved_payloads[0]["results"][0]["error_message"]
    )


def test_main_skips_plots_for_single_workspace(
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
            "run_workspace_loading.py",
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
        benchmark, "make_plots", lambda *args, **kwargs: make_plots_calls.append(args)
    )

    benchmark.main()

    assert len(make_plots_calls) == 1


def test_main_creates_plots_for_multiple_workspaces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    workspace_a = tmp_path / "a.json"
    workspace_b = tmp_path / "b.json"
    workspace_a.write_text("{}")
    workspace_b.write_text("{}")
    make_plots_calls = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_workspace_loading.py",
            "--workspaces",
            str(workspace_a),
            str(workspace_b),
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


def test_run_single_benchmark_real_workspace() -> None:
    workspace_path = Path("inputs/simple_workspace.json")

    result = benchmark.run_single_benchmark(workspace_path, n_runs=1)

    assert result["status"] == "success"
    assert result["benchmark"] == "workspace_loading"
    assert result["workspace"] == "simple_workspace.json"
    assert result["n_runs"] == 1
    assert result["n_distributions"] > 0
    assert result["n_likelihoods"] > 0
    assert result["n_data"] > 0
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
    output_name = "workspace_loading_result.json"
    output_path = output_dir / output_name

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_workspace_loading.py",
            "--workspaces",
            "inputs/simple_workspace.json",
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

    assert payload["benchmark"] == "workspace_loading"
    assert payload["n_workspaces"] == 1
    assert len(payload["results"]) == 1
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][0]["workspace"] == "simple_workspace.json"


def test_make_plots_real_png_files_created(
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    second_result = {**valid_result, "workspace": "workspace_2.json"}

    benchmark.make_plots(
        results=[valid_result, second_result],
        plot_dir=tmp_path,
        wall_time_plot_name="workspace_loading_wall_time.png",
    )

    assert (tmp_path / "workspace_loading_wall_time.png").exists()
    assert (tmp_path / "workspace_loading_peak_rss_delta.png").exists()
    assert (tmp_path / "workspace_loading_current_rss_delta.png").exists()


def test_validate_workspace_rejects_none_workspace() -> None:
    with pytest.raises(ValueError, match="Workspace loading returned None"):
        benchmark.validate_workspace(None)


def test_validate_workspace_rejects_missing_metadata(
    valid_workspace: SimpleNamespace,
) -> None:
    valid_workspace.metadata = None

    with pytest.raises(ValueError, match="Workspace does not contain metadata"):
        benchmark.validate_workspace(valid_workspace)


def test_make_plots_skips_when_less_than_two_successful_results(
    tmp_path: Path,
    valid_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark.make_plots(
        results=[valid_result],
        plot_dir=tmp_path,
        wall_time_plot_name="wall_time.png",
    )

    output = capsys.readouterr().out

    assert "Skipping plots" in output
    assert not (tmp_path / "wall_time.png").exists()


class RaisingContext:
    def Pool(self, processes: int) -> None:
        assert processes == 1
        raise RuntimeError("pool failed")


def test_main_records_pool_error_as_failed_result(
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
            "run_workspace_loading.py",
            "--workspaces",
            str(workspace_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.setattr(benchmark, "get_context", lambda method: RaisingContext())
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)

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
    assert "pool failed" in result["error_message"]


def test_main_wraps_json_write_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    output_dir = tmp_path / "results"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_workspace_loading.py",
            "--workspaces",
            str(workspace_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    monkeypatch.setattr(
        benchmark, "get_context", lambda method: FakeContext(valid_result)
    )
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, output_path: (_ for _ in ()).throw(OSError("write failed")),
    )

    with pytest.raises(RuntimeError, match="Failed to write benchmark result JSON"):
        benchmark.main()


def test_main_wraps_plot_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    output_dir = tmp_path / "results"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_workspace_loading.py",
            "--workspaces",
            str(workspace_path),
            "--output-dir",
            str(output_dir),
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
        lambda results, plot_dir, wall_time_plot_name: (_ for _ in ()).throw(
            OSError("plot failed")
        ),
    )

    with pytest.raises(RuntimeError, match="Failed to create workspace loading plots"):
        benchmark.main()


def test_module_runs_as_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "script_results"
    output_name = "script_result.json"
    output_path = output_dir / output_name

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_workspace_loading.py",
            "--workspaces",
            "inputs/simple_workspace.json",
            "--n-runs",
            "1",
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
        ],
    )

    runpy.run_module("src.run_workspace_loading", run_name="__main__")

    assert output_path.exists()
