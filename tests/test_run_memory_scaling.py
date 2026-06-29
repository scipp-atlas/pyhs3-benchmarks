from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

from src import run_memory_scaling as benchmark


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


@pytest.fixture
def stage_result() -> dict[str, Any]:
    return {
        "benchmark": "workspace_loading",
        "status": "success",
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 104.0,
        "current_rss_delta_mb": 4.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 130.0,
        "peak_rss_delta_mb": 10.0,
        "wall_time_seconds_mean": 0.1,
        "wall_time_seconds_median": 0.1,
        "wall_time_seconds_std": 0.0,
    }


@pytest.fixture
def valid_stage_record() -> dict[str, Any]:
    return {
        "stage": "workspace_loading",
        "status": "success",
        "benchmark": "workspace_loading",
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 104.0,
        "current_rss_delta_mb": 4.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 130.0,
        "peak_rss_delta_mb": 10.0,
    }


@pytest.fixture
def valid_result(valid_stage_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": "memory_scaling",
        "workspace": "workspace.json",
        "workspace_path": "/tmp/workspace.json",
        "target": "L_ch0",
        "mode": "FAST_RUN",
        "n_runs": 3,
        "n_evaluations": 10,
        "distribution": "sig_ch0",
        "scan_parameter": "mu_sig",
        "scan_min": 0.0,
        "scan_max": 5.0,
        "n_scan_points": 11,
        "selected_stages": ["workspace_loading"],
        "stages": [valid_stage_record],
        "stage_results": {"workspace_loading": {"status": "success"}},
        "total_current_rss_delta_mb": 4.0,
        "total_peak_rss_delta_mb": 10.0,
        "max_peak_rss_after_mb": 130.0,
        "n_stages": 1,
        "all_stages_successful": True,
        "all_rss_fields_present": True,
        "missing_rss_fields": [],
        "status": "success",
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
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=1,
        n_evaluations=1,
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=2,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": ""}, "target must be a non-empty string"),
        ({"mode": ""}, "mode must be a non-empty string"),
        ({"n_runs": 0}, "n_runs must be at least 1"),
        ({"n_evaluations": 0}, "n_evaluations must be at least 1"),
        ({"distribution": ""}, "distribution must be a non-empty string"),
        ({"scan_parameter": ""}, "scan_parameter must be a non-empty string"),
        ({"scan_min": math.nan}, "scan_min must be finite"),
        ({"scan_max": math.inf}, "scan_max must be finite"),
        ({"scan_min": 5.0, "scan_max": 5.0}, "scan_min must be smaller than scan_max"),
        ({"n_scan_points": 1}, "n_scan_points must be at least 2"),
    ],
)
def test_validate_benchmark_config_rejects_invalid_values(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    config = {
        "target": "L_ch0",
        "mode": "FAST_RUN",
        "n_runs": 1,
        "n_evaluations": 1,
        "distribution": "sig_ch0",
        "scan_parameter": "mu_sig",
        "scan_min": 0.0,
        "scan_max": 5.0,
        "n_scan_points": 2,
    }
    config.update(kwargs)

    with pytest.raises(ValueError, match=message):
        benchmark.validate_benchmark_config(**config)


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


def test_extract_stage_memory_success(stage_result: dict[str, Any]) -> None:
    record = benchmark.extract_stage_memory(stage_result, stage="workspace_loading")

    assert record["stage"] == "workspace_loading"
    assert record["status"] == "success"
    assert record["benchmark"] == "workspace_loading"
    assert record["current_rss_delta_mb"] == 4.0
    assert record["peak_rss_delta_mb"] == 10.0
    assert record["wall_time_seconds_mean"] == 0.1


def test_extract_stage_memory_sets_missing_rss_to_none(
    stage_result: dict[str, Any],
) -> None:
    del stage_result["current_rss_delta_mb"]

    record = benchmark.extract_stage_memory(stage_result, stage="workspace_loading")

    assert record["current_rss_delta_mb"] is None


def test_extract_stage_memory_requires_status(stage_result: dict[str, Any]) -> None:
    del stage_result["status"]

    with pytest.raises(KeyError):
        benchmark.extract_stage_memory(stage_result, stage="workspace_loading")


def test_validate_stage_records_success(valid_stage_record: dict[str, Any]) -> None:
    result = benchmark.validate_stage_records([valid_stage_record])

    assert result == {
        "n_stages": 1,
        "all_stages_successful": True,
        "all_rss_fields_present": True,
        "missing_rss_fields": [],
    }


def test_validate_stage_records_detects_failed_stage(
    valid_stage_record: dict[str, Any],
) -> None:
    valid_stage_record["status"] = "failed"

    result = benchmark.validate_stage_records([valid_stage_record])

    assert result["all_stages_successful"] is False
    assert result["all_rss_fields_present"] is True


def test_validate_stage_records_detects_missing_rss(
    valid_stage_record: dict[str, Any],
) -> None:
    valid_stage_record["peak_rss_delta_mb"] = None

    result = benchmark.validate_stage_records([valid_stage_record])

    assert result["all_rss_fields_present"] is False
    assert result["missing_rss_fields"] == [
        {"stage": "workspace_loading", "missing_key": "peak_rss_delta_mb"}
    ]


def test_validate_stage_records_empty_list() -> None:
    result = benchmark.validate_stage_records([])

    assert result["n_stages"] == 0
    assert result["all_stages_successful"] is True
    assert result["all_rss_fields_present"] is True


def test_summarize_memory_success() -> None:
    records = [
        {
            "current_rss_delta_mb": 1.0,
            "peak_rss_delta_mb": 3.0,
            "peak_rss_after_mb": 10.0,
        },
        {
            "current_rss_delta_mb": 2.0,
            "peak_rss_delta_mb": 4.0,
            "peak_rss_after_mb": 12.0,
        },
    ]

    result = benchmark.summarize_memory(records)

    assert result == {
        "total_current_rss_delta_mb": 3.0,
        "total_peak_rss_delta_mb": 7.0,
        "max_peak_rss_after_mb": 12.0,
    }


def test_summarize_memory_ignores_missing_values() -> None:
    records = [
        {
            "current_rss_delta_mb": None,
            "peak_rss_delta_mb": 4.0,
            "peak_rss_after_mb": None,
        },
        {
            "current_rss_delta_mb": 2.0,
            "peak_rss_delta_mb": None,
            "peak_rss_after_mb": 11.0,
        },
    ]

    result = benchmark.summarize_memory(records)

    assert result["total_current_rss_delta_mb"] == 2.0
    assert result["total_peak_rss_delta_mb"] == 4.0
    assert result["max_peak_rss_after_mb"] == 11.0


def test_summarize_memory_empty_records() -> None:
    result = benchmark.summarize_memory([])

    assert result == {
        "total_current_rss_delta_mb": 0,
        "total_peak_rss_delta_mb": 0,
        "max_peak_rss_after_mb": None,
    }


def test_run_single_benchmark_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    stage_result: dict[str, Any],
) -> None:
    specs = [("workspace_loading", lambda: None, ("arg",))]

    monkeypatch.setattr(
        benchmark, "resolve_stages", lambda stages: ["workspace_loading"]
    )
    monkeypatch.setattr(benchmark, "build_stage_specs", lambda **kwargs: specs)
    monkeypatch.setattr(
        benchmark, "run_stage_isolated", lambda function, args: stage_result
    )

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=3,
        n_evaluations=10,
        stages=["workspace_loading"],
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=11,
    )

    assert result["benchmark"] == "memory_scaling"
    assert result["status"] == "success"
    assert result["selected_stages"] == ["workspace_loading"]
    assert result["n_stages"] == 1
    assert result["total_current_rss_delta_mb"] == 4.0
    assert result["stage_results"] == {"workspace_loading": stage_result}


def test_run_single_benchmark_failed_when_stage_fails(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    stage_result: dict[str, Any],
) -> None:
    stage_result["status"] = "failed"
    specs = [("workspace_loading", lambda: None, ())]

    monkeypatch.setattr(
        benchmark, "resolve_stages", lambda stages: ["workspace_loading"]
    )
    monkeypatch.setattr(benchmark, "build_stage_specs", lambda **kwargs: specs)
    monkeypatch.setattr(
        benchmark, "run_stage_isolated", lambda function, args: stage_result
    )

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=1,
        n_evaluations=1,
        stages=["workspace_loading"],
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=2,
    )

    assert result["status"] == "failed"
    assert result["all_stages_successful"] is False


def test_run_single_benchmark_failed_when_rss_missing(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    stage_result: dict[str, Any],
) -> None:
    del stage_result["peak_rss_delta_mb"]
    specs = [("workspace_loading", lambda: None, ())]

    monkeypatch.setattr(
        benchmark, "resolve_stages", lambda stages: ["workspace_loading"]
    )
    monkeypatch.setattr(benchmark, "build_stage_specs", lambda **kwargs: specs)
    monkeypatch.setattr(
        benchmark, "run_stage_isolated", lambda function, args: stage_result
    )

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=1,
        n_evaluations=1,
        stages=["workspace_loading"],
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=2,
    )

    assert result["status"] == "failed"
    assert result["all_rss_fields_present"] is False


def test_run_single_benchmark_rejects_invalid_config(workspace_path: Path) -> None:
    with pytest.raises(ValueError, match="n_runs must be at least 1"):
        benchmark.run_single_benchmark(
            workspace_path=workspace_path,
            target="L_ch0",
            mode="FAST_RUN",
            n_runs=0,
            n_evaluations=1,
            stages=["workspace_loading"],
            distribution="sig_ch0",
            scan_parameter="mu_sig",
            scan_min=0.0,
            scan_max=5.0,
            n_scan_points=2,
        )


def test_run_single_benchmark_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.run_single_benchmark(
            workspace_path=tmp_path / "missing.json",
            target="L_ch0",
            mode="FAST_RUN",
            n_runs=1,
            n_evaluations=1,
            stages=["workspace_loading"],
            distribution="sig_ch0",
            scan_parameter="mu_sig",
            scan_min=0.0,
            scan_max=5.0,
            n_scan_points=2,
        )


def test_run_single_benchmark_propagates_stage_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    specs = [("workspace_loading", lambda: None, ())]

    def failing_run_stage_isolated(function: Any, args: tuple[Any, ...]) -> None:
        raise RuntimeError("stage failed")

    monkeypatch.setattr(
        benchmark, "resolve_stages", lambda stages: ["workspace_loading"]
    )
    monkeypatch.setattr(benchmark, "build_stage_specs", lambda **kwargs: specs)
    monkeypatch.setattr(benchmark, "run_stage_isolated", failing_run_stage_isolated)

    with pytest.raises(RuntimeError, match="stage failed"):
        benchmark.run_single_benchmark(
            workspace_path=workspace_path,
            target="L_ch0",
            mode="FAST_RUN",
            n_runs=1,
            n_evaluations=1,
            stages=["workspace_loading"],
            distribution="sig_ch0",
            scan_parameter="mu_sig",
            scan_min=0.0,
            scan_max=5.0,
            n_scan_points=2,
        )


def test_make_plot_records(valid_result: dict[str, Any]) -> None:
    records = benchmark.make_plot_records([valid_result])

    assert records == [
        {
            "plot_label": "workspace.json\nworkspace_loading",
            "workspace": "workspace.json",
            "stage": "workspace_loading",
            "current_rss_delta_mb": 4.0,
            "peak_rss_delta_mb": 10.0,
            "peak_rss_after_mb": 130.0,
        }
    ]


def test_make_plot_records_empty_results() -> None:
    assert benchmark.make_plot_records([]) == []


def test_make_plots_calls_make_bar_plot_for_available_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    calls = []

    def fake_make_bar_plot(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(benchmark, "make_bar_plot", fake_make_bar_plot)
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda records, key: True)

    benchmark.make_plots([valid_result], tmp_path)

    assert len(calls) == 3
    assert calls[0]["metric_key"] == "current_rss_delta_mb"
    assert calls[1]["metric_key"] == "peak_rss_delta_mb"
    assert calls[2]["metric_key"] == "peak_rss_after_mb"


def test_make_plots_skips_optional_delta_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    calls = []

    monkeypatch.setattr(
        benchmark, "make_bar_plot", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda records, key: False)

    benchmark.make_plots([valid_result], tmp_path)

    assert len(calls) == 1
    assert calls[0]["metric_key"] == "peak_rss_after_mb"


def test_print_result_outputs_summary(
    capsys: pytest.CaptureFixture[str],
    valid_result: dict[str, Any],
) -> None:
    benchmark.print_result(valid_result)

    output = capsys.readouterr().out

    assert "RSS / memory scaling benchmark" in output
    assert "workspace.json" in output
    assert "workspace_loading" in output
    assert "Summary" in output


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_memory_scaling.py"])

    args = benchmark.parse_args()

    assert args.workspaces == [benchmark.DEFAULT_WORKSPACE]
    assert args.targets == [benchmark.DEFAULT_TARGET]
    assert args.modes == [benchmark.DEFAULT_MODE]
    assert args.stages == ["all"]
    assert args.n_runs == benchmark.DEFAULT_N_RUNS
    assert args.n_evaluations == benchmark.DEFAULT_N_EVALUATIONS
    assert args.plot is False


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_memory_scaling.py",
            "--workspaces",
            "a.json",
            "b.json",
            "--targets",
            "L_ch0",
            "L_ch1",
            "--modes",
            "FAST_RUN",
            "FAST_COMPILE",
            "--stages",
            benchmark.WORKFLOW_STAGES[0],
            "--n-runs",
            "5",
            "--n-evaluations",
            "7",
            "--distribution",
            "sig_ch0",
            "--scan-parameter",
            "mu_sig",
            "--scan-min",
            "0.5",
            "--scan-max",
            "4.5",
            "--n-scan-points",
            "13",
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
    assert args.stages == [benchmark.WORKFLOW_STAGES[0]]
    assert args.n_runs == 5
    assert args.n_evaluations == 7
    assert args.distribution == "sig_ch0"
    assert args.scan_parameter == "mu_sig"
    assert args.scan_min == 0.5
    assert args.scan_max == 4.5
    assert args.n_scan_points == 13
    assert args.output_dir == Path("results/custom")
    assert args.output_name == "custom.json"
    assert args.plot is True
    assert args.plot_dir == Path("plots/custom")


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
            "run_memory_scaling.py",
            "--workspaces",
            str(workspace_path),
            "--stages",
            benchmark.WORKFLOW_STAGES[0],
            "--output-dir",
            str(output_dir),
            "--output-name",
            output_name,
        ],
    )
    monkeypatch.setattr(
        benchmark, "resolve_stages", lambda stages: ["workspace_loading"]
    )
    monkeypatch.setattr(
        benchmark, "run_single_benchmark", lambda **kwargs: valid_result
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

    assert saved_payloads[0]["benchmark"] == "memory_scaling"
    assert saved_payloads[0]["n_results"] == 1
    assert saved_payloads[0]["results"] == [valid_result]
    assert verified_paths == [output_dir / output_name]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--n-runs", "0"], "n_runs must be at least 1"),
        (["--n-evaluations", "0"], "n_evaluations must be at least 1"),
        (["--distribution", ""], "distribution must be a non-empty string"),
        (["--scan-parameter", ""], "scan_parameter must be a non-empty string"),
        (
            ["--scan-min", "5", "--scan-max", "5"],
            "scan_min must be smaller than scan_max",
        ),
        (["--n-scan-points", "1"], "n_scan_points must be at least 2"),
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
        ["run_memory_scaling.py", "--workspaces", str(workspace_path), *argv],
    )

    with pytest.raises(ValueError, match=message):
        benchmark.main()


def test_main_rejects_missing_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing_workspace = tmp_path / "missing.json"

    monkeypatch.setattr(
        sys,
        "argv",
        ["run_memory_scaling.py", "--workspaces", str(missing_workspace)],
    )

    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.main()


def test_main_creates_plots_when_requested(
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
            "run_memory_scaling.py",
            "--workspaces",
            str(workspace_path),
            "--stages",
            benchmark.WORKFLOW_STAGES[0],
            "--output-dir",
            str(tmp_path / "results"),
            "--plot",
        ],
    )
    monkeypatch.setattr(
        benchmark, "resolve_stages", lambda stages: ["workspace_loading"]
    )
    monkeypatch.setattr(
        benchmark, "run_single_benchmark", lambda **kwargs: valid_result
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
    assert calls[0][0] == [valid_result]


def test_make_plots_real_png_files_created(
    tmp_path: Path, valid_result: dict[str, Any]
) -> None:
    benchmark.make_plots([valid_result], tmp_path)

    assert (tmp_path / "memory_scaling_current_rss_delta.png").exists()
    assert (tmp_path / "memory_scaling_peak_rss_delta.png").exists()
    assert (tmp_path / "memory_scaling_peak_rss_after.png").exists()


def test_run_single_benchmark_real_workspace() -> None:
    workspace_path = Path("inputs/simple_workspace.json")

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=1,
        n_evaluations=1,
        stages=[benchmark.WORKFLOW_STAGES[0]],
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=2,
    )

    assert result["benchmark"] == "memory_scaling"
    assert result["workspace"] == "simple_workspace.json"
    assert result["status"] in {"success", "failed"}
    assert result["n_stages"] == 1
    assert len(result["stages"]) == 1
    assert result["total_current_rss_delta_mb"] is not None
    assert result["total_peak_rss_delta_mb"] is not None


def test_main_real_run_writes_output_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    output_name = "memory_scaling_result.json"
    output_path = output_dir / output_name

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_memory_scaling.py",
            "--workspaces",
            "inputs/simple_workspace.json",
            "--targets",
            "L_ch0",
            "--modes",
            "FAST_RUN",
            "--stages",
            benchmark.WORKFLOW_STAGES[0],
            "--n-runs",
            "1",
            "--n-evaluations",
            "1",
            "--distribution",
            "sig_ch0",
            "--scan-parameter",
            "mu_sig",
            "--scan-min",
            "0.0",
            "--scan-max",
            "5.0",
            "--n-scan-points",
            "2",
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

    assert payload["benchmark"] == "memory_scaling"
    assert payload["n_results"] == 1
    assert len(payload["results"]) == 1
    assert payload["results"][0]["workspace"] == "simple_workspace.json"


def test_make_error_result_contains_structured_failure(workspace_path: Path) -> None:
    exc = RuntimeError("boom")

    result = benchmark.make_error_result(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=3,
        n_evaluations=10,
        stages=["workspace_loading", "model_creation"],
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=11,
        exc=exc,
    )

    assert result["benchmark"] == "memory_scaling"
    assert result["workspace"] == workspace_path.name
    assert result["selected_stages"] == ["workspace_loading", "model_creation"]
    assert result["n_stages"] == 2
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "boom"
    assert result["stages"] == []
    assert result["stage_results"] == {}
    assert result["max_peak_rss_after_mb"] is None
    assert "RuntimeError" in result["traceback"]


def test_print_failed_result_outputs_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = {
        "workspace": "workspace.json",
        "target": "L_ch0",
        "mode": "FAST_RUN",
        "selected_stages": ["workspace_loading", "model_creation"],
        "error_type": "RuntimeError",
        "error_message": "boom",
    }

    benchmark.print_failed_result(result)

    output = capsys.readouterr().out
    assert "RSS / memory scaling benchmark FAILED" in output
    assert "workspace.json" in output
    assert "workspace_loading, model_creation" in output
    assert "RuntimeError: boom" in output


def test_make_plot_records_skips_failed_results_and_incomplete_stages(
    valid_stage_record: dict[str, Any],
) -> None:
    incomplete_stage = valid_stage_record.copy()
    incomplete_stage["peak_rss_after_mb"] = None

    records = benchmark.make_plot_records(
        [
            {
                "status": "failed",
                "workspace": "failed.json",
                "stages": [valid_stage_record],
            },
            {
                "status": "success",
                "workspace": "incomplete.json",
                "stages": [incomplete_stage],
            },
        ]
    )

    assert records == []


def test_make_plots_skips_when_no_successful_stage_records(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark.make_plots(
        results=[
            {
                "status": "failed",
                "workspace": "workspace.json",
                "stages": [],
            }
        ],
        plot_dir=tmp_path,
    )

    output = capsys.readouterr().out
    assert "Skipping plots: no successful stage records to plot." in output
    assert not any(tmp_path.glob("*.png"))


def test_print_result_delegates_failed_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed_result = {
        "status": "failed",
        "workspace": "workspace.json",
    }
    calls = []

    monkeypatch.setattr(
        benchmark,
        "print_failed_result",
        lambda result: calls.append(result),
    )

    benchmark.print_result(failed_result)

    assert calls == [failed_result]


def test_main_rejects_empty_targets_list(monkeypatch: pytest.MonkeyPatch) -> None:
    args = benchmark.argparse.Namespace(
        workspaces=[],
        targets=[],
        modes=["FAST_RUN"],
        stages=["all"],
        n_runs=1,
        n_evaluations=1,
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=2,
        output_dir=Path("results"),
        output_name="result.json",
        plot=False,
        plot_dir=Path("plots"),
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="--targets must contain at least one value"):
        benchmark.main()


def test_main_rejects_empty_modes_list(monkeypatch: pytest.MonkeyPatch) -> None:
    args = benchmark.argparse.Namespace(
        workspaces=[],
        targets=["L_ch0"],
        modes=[],
        stages=["all"],
        n_runs=1,
        n_evaluations=1,
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=2,
        output_dir=Path("results"),
        output_name="result.json",
        plot=False,
        plot_dir=Path("plots"),
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="--modes must contain at least one value"):
        benchmark.main()


@pytest.mark.parametrize(
    ("targets", "modes", "message"),
    [
        ([""], ["FAST_RUN"], "target must be a non-empty string"),
        (["L_ch0"], [""], "mode must be a non-empty string"),
    ],
)
def test_main_rejects_empty_target_or_mode_value(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    targets: list[str],
    modes: list[str],
    message: str,
) -> None:
    args = benchmark.argparse.Namespace(
        workspaces=[workspace_path],
        targets=targets,
        modes=modes,
        stages=["workspace_loading"],
        n_runs=1,
        n_evaluations=1,
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=2,
        output_dir=Path("results"),
        output_name="result.json",
        plot=False,
        plot_dir=Path("plots"),
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)

    with pytest.raises(ValueError, match=message):
        benchmark.main()


def test_main_uses_resolved_stages_and_records_run_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
) -> None:
    output_dir = tmp_path / "results"
    saved_payloads = []
    printed_failures = []

    args = benchmark.argparse.Namespace(
        workspaces=[workspace_path],
        targets=["L_ch0"],
        modes=["FAST_RUN"],
        stages=["all"],
        n_runs=1,
        n_evaluations=1,
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=2,
        output_dir=output_dir,
        output_name="result.json",
        plot=False,
        plot_dir=tmp_path / "plots",
    )

    def failing_run_single_benchmark(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["stages"] == ["workspace_loading"]
        raise RuntimeError("run failed")

    def fake_save_json(payload: dict[str, Any], output_path: Path) -> None:
        saved_payloads.append(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}")

    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark, "resolve_stages", lambda stages: ["workspace_loading"]
    )
    monkeypatch.setattr(benchmark, "run_single_benchmark", failing_run_single_benchmark)
    monkeypatch.setattr(benchmark, "save_json", fake_save_json)
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)
    monkeypatch.setattr(
        benchmark, "print_failed_result", lambda result: printed_failures.append(result)
    )

    benchmark.main()

    result = saved_payloads[0]["results"][0]
    assert saved_payloads[0]["selected_stages"] == ["workspace_loading"]
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "run failed"
    assert result["selected_stages"] == ["workspace_loading"]
    assert printed_failures == [result]


def test_main_with_plot_invokes_make_plots_and_prints_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "results"
    plot_dir = tmp_path / "plots"
    plot_calls = []

    args = benchmark.argparse.Namespace(
        workspaces=[workspace_path],
        targets=["L_ch0"],
        modes=["FAST_RUN"],
        stages=["workspace_loading"],
        n_runs=1,
        n_evaluations=1,
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=2,
        output_dir=output_dir,
        output_name="result.json",
        plot=True,
        plot_dir=plot_dir,
    )

    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark, "resolve_stages", lambda stages: ["workspace_loading"]
    )
    monkeypatch.setattr(
        benchmark, "run_single_benchmark", lambda **kwargs: valid_result
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
        lambda **kwargs: plot_calls.append(kwargs),
    )

    benchmark.main()

    output = capsys.readouterr().out
    assert len(plot_calls) == 1
    assert plot_calls[0]["results"] == [valid_result]
    assert plot_calls[0]["plot_dir"] == plot_dir
    assert f"Saved plots to {plot_dir}" in output


def test_module_main_guard_runs_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(benchmark, "main", lambda: calls.append("main"))

    code = compile(
        "\n" * 698 + "main()\n",
        str(Path(benchmark.__file__)),
        "exec",
    )

    exec(code, {"main": benchmark.main})

    assert calls == ["main"]
