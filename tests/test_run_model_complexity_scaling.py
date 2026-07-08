from __future__ import annotations

import csv
import json
import math
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from src import run_model_complexity_scaling as benchmark


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text('{"workspace": true}')
    return path


@pytest.fixture
def stage_result() -> dict[str, Any]:
    return {
        "benchmark": "workspace_loading",
        "status": "success",
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 102.0,
        "current_rss_delta_mb": 2.0,
        "peak_rss_before_mb": 110.0,
        "peak_rss_after_mb": 114.0,
        "peak_rss_delta_mb": 4.0,
        "wall_time_seconds_mean": 0.1,
        "wall_time_seconds_std": 0.01,
    }


@pytest.fixture
def valid_result(workspace_path: Path) -> dict[str, Any]:
    return {
        "benchmark": "model_complexity_scaling",
        "workspace": workspace_path.name,
        "workspace_path": str(workspace_path),
        "workspace_size_bytes": workspace_path.stat().st_size,
        "target": "L_ch0",
        "mode": "FAST_RUN",
        "n_runs": 1,
        "n_evaluations": 1,
        "distribution": "sig_ch0",
        "scan_parameter": "mu_sig",
        "scan_min": 0.0,
        "scan_max": 5.0,
        "n_scan_points": 3,
        "selected_stages": ["workspace_loading", "compiled_evaluation", "nll_scan"],
        "quickfit_reference_available": False,
        "quickfit_validation_status": "not_run",
        "workspace_loading_status": "success",
        "workspace_loading_wall_time_seconds_mean": 0.1,
        "workspace_loading_wall_time_seconds_std": 0.01,
        "workspace_loading_current_rss_delta_mb": 2.0,
        "workspace_loading_peak_rss_delta_mb": 4.0,
        "compiled_evaluation_status": "success",
        "compiled_evaluation_current_rss_delta_mb": 1.0,
        "compiled_evaluation_peak_rss_delta_mb": 3.0,
        "compiled_evaluation_average_runtime_seconds_per_evaluation": 0.002,
        "compiled_evaluation_throughput_evaluations_per_second": 500.0,
        "compiled_evaluation_total_runtime_seconds": 0.002,
        "compiled_evaluation_reference_output": -1.0,
        "compiled_evaluation_all_outputs_finite": True,
        "nll_scan_status": "success",
        "nll_scan_current_rss_delta_mb": 1.5,
        "nll_scan_peak_rss_delta_mb": 2.5,
        "nll_scan_runtime_per_scan_point_seconds": 0.003,
        "nll_scan_throughput_scan_points_per_second": 333.0,
        "nll_scan_total_runtime_seconds": 0.009,
        "nll_scan_minimum_scan_value": 1.0,
        "nll_scan_minimum_nll_value": 2.0,
        "nll_scan_all_values_finite": True,
        "stage_results": {},
        "total_setup_time_seconds": 0.1,
        "total_peak_rss_delta_mb": 9.5,
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
    with pytest.raises(FileNotFoundError, match="Output file was not created"):
        benchmark.verify_output_file(tmp_path / "missing.json")


def test_verify_output_file_directory_is_invalid(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Output path is not a file"):
        benchmark.verify_output_file(tmp_path)


def test_summarize_stage_basic(stage_result: dict[str, Any]) -> None:
    result = benchmark.summarize_stage(stage_result, "workspace_loading")

    assert result["workspace_loading_status"] == "success"
    assert result["workspace_loading_current_rss_delta_mb"] == 2.0
    assert result["workspace_loading_peak_rss_delta_mb"] == 4.0
    assert result["workspace_loading_wall_time_seconds_mean"] == 0.1
    assert result["workspace_loading_wall_time_seconds_std"] == 0.01


def test_summarize_stage_defaults_missing_rss_to_zero() -> None:
    result = benchmark.summarize_stage({"status": "success"}, "stage")

    assert result["stage_current_rss_delta_mb"] == 0.0
    assert result["stage_peak_rss_delta_mb"] == 0.0


def test_summarize_stage_requires_status() -> None:
    with pytest.raises(KeyError):
        benchmark.summarize_stage({}, "stage")


def test_summarize_stage_includes_evaluation_metrics() -> None:
    result = benchmark.summarize_stage(
        {
            "status": "success",
            "average_runtime_seconds_per_evaluation": 0.002,
            "throughput_evaluations_per_second": 500.0,
        },
        "compiled_evaluation",
    )

    assert result["compiled_evaluation_average_runtime_seconds_per_evaluation"] == 0.002
    assert result["compiled_evaluation_throughput_evaluations_per_second"] == 500.0


def test_summarize_stage_includes_scan_metrics() -> None:
    result = benchmark.summarize_stage(
        {
            "status": "success",
            "runtime_per_scan_point_seconds": 0.003,
            "throughput_scan_points_per_second": 333.0,
            "total_runtime_seconds": 0.009,
        },
        "nll_scan",
    )

    assert result["nll_scan_runtime_per_scan_point_seconds"] == 0.003
    assert result["nll_scan_throughput_scan_points_per_second"] == 333.0
    assert result["nll_scan_total_runtime_seconds"] == 0.009


def test_run_single_scaling_benchmark_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def stage_a(*args: Any) -> dict[str, Any]:
        return {
            "benchmark": "workspace_loading",
            "status": "success",
            "wall_time_seconds_mean": 0.1,
            "wall_time_seconds_std": 0.01,
            "current_rss_delta_mb": 1.0,
            "peak_rss_delta_mb": 2.0,
        }

    def stage_b(*args: Any) -> dict[str, Any]:
        return {
            "benchmark": "compiled_evaluation",
            "status": "success",
            "average_runtime_seconds_per_evaluation": 0.002,
            "throughput_evaluations_per_second": 500.0,
            "total_runtime_seconds": 0.002,
            "current_rss_delta_mb": 3.0,
            "peak_rss_delta_mb": 4.0,
            "reference_output": -1.0,
            "all_outputs_finite": True,
        }

    monkeypatch.setattr(benchmark, "resolve_stages", lambda stages: stages)
    monkeypatch.setattr(
        benchmark,
        "build_stage_specs",
        lambda **kwargs: [
            ("workspace_loading", stage_a, ()),
            ("compiled_evaluation", stage_b, ()),
        ],
    )

    result = benchmark.run_single_scaling_benchmark(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=1,
        n_evaluations=1,
        stages=["workspace_loading", "compiled_evaluation"],
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=3,
    )

    assert result["status"] == "success"
    assert result["workspace_size_bytes"] == workspace_path.stat().st_size
    assert result["total_setup_time_seconds"] == 0.1
    assert result["total_peak_rss_delta_mb"] == 6.0
    assert result["compiled_evaluation_reference_output"] == -1.0
    assert result["compiled_evaluation_all_outputs_finite"] is True
    assert set(result["stage_results"]) == {"workspace_loading", "compiled_evaluation"}


def test_run_single_scaling_benchmark_includes_nll_scan_summary(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def nll_stage(*args: Any) -> dict[str, Any]:
        return {
            "benchmark": "nll_scan",
            "status": "success",
            "current_rss_delta_mb": 1.0,
            "peak_rss_delta_mb": 2.0,
            "minimum_scan_value": 1.0,
            "minimum_nll_value": 2.0,
            "all_nll_values_finite": True,
            "runtime_per_scan_point_seconds": 0.003,
            "throughput_scan_points_per_second": 333.0,
        }

    monkeypatch.setattr(benchmark, "resolve_stages", lambda stages: stages)
    monkeypatch.setattr(
        benchmark, "build_stage_specs", lambda **kwargs: [("nll_scan", nll_stage, ())]
    )

    result = benchmark.run_single_scaling_benchmark(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=1,
        n_evaluations=1,
        stages=["nll_scan"],
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=3,
    )

    assert result["nll_scan_minimum_scan_value"] == 1.0
    assert result["nll_scan_minimum_nll_value"] == 2.0
    assert result["nll_scan_all_values_finite"] is True


def test_run_single_scaling_benchmark_failed_when_stage_fails(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_stage(*args: Any) -> dict[str, Any]:
        return {
            "benchmark": "workspace_loading",
            "status": "failed",
            "peak_rss_delta_mb": 1.0,
        }

    monkeypatch.setattr(benchmark, "resolve_stages", lambda stages: stages)
    monkeypatch.setattr(
        benchmark,
        "build_stage_specs",
        lambda **kwargs: [("workspace_loading", failing_stage, ())],
    )

    result = benchmark.run_single_scaling_benchmark(
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
        n_scan_points=3,
    )

    assert result["status"] == "failed"


def test_run_single_scaling_benchmark_rejects_invalid_config(
    workspace_path: Path,
) -> None:
    with pytest.raises(ValueError, match="n_runs must be at least 1"):
        benchmark.run_single_scaling_benchmark(
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
            n_scan_points=3,
        )


def test_run_single_scaling_benchmark_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.run_single_scaling_benchmark(
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
            n_scan_points=3,
        )


def test_run_single_scaling_benchmark_records_stage_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def bad_stage(*args: Any) -> dict[str, Any]:
        raise RuntimeError("stage failed")

    monkeypatch.setattr(benchmark, "resolve_stages", lambda stages: stages)
    monkeypatch.setattr(
        benchmark,
        "build_stage_specs",
        lambda **kwargs: [("workspace_loading", bad_stage, ())],
    )

    result = benchmark.run_single_scaling_benchmark(
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
        n_scan_points=3,
    )

    assert result["status"] == "failed"
    assert result["workspace_loading_status"] == "failed"
    assert result["stage_results"]["workspace_loading"]["error_type"] == "RuntimeError"
    assert (
        "stage failed" in result["stage_results"]["workspace_loading"]["error_message"]
    )


def test_write_summary_csv_success(
    tmp_path: Path, valid_result: dict[str, Any]
) -> None:
    output_path = tmp_path / "reports" / "summary.csv"

    benchmark.write_summary_csv([valid_result], output_path)

    assert output_path.exists()
    with output_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["benchmark"] == "model_complexity_scaling"
    assert "stage_results" not in rows[0]


def test_write_summary_csv_empty_results(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.csv"

    benchmark.write_summary_csv([], output_path)

    assert output_path.read_text().strip() == ""


def test_print_result_outputs_summary(
    capsys: pytest.CaptureFixture[str], valid_result: dict[str, Any]
) -> None:
    benchmark.print_result(valid_result)

    output = capsys.readouterr().out

    assert "Model complexity scaling benchmark" in output
    assert "Timing" in output
    assert "Memory" in output
    assert "Validation" in output
    assert valid_result["workspace"] in output


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_model_complexity_scaling.py"])

    args = benchmark.parse_args()

    assert args.workspaces == [benchmark.DEFAULT_WORKSPACE]
    assert args.targets == [benchmark.DEFAULT_TARGET]
    assert args.modes == [benchmark.DEFAULT_MODE]
    assert args.stages == ["all"]
    assert args.n_runs == benchmark.DEFAULT_N_RUNS
    assert args.n_evaluations == benchmark.DEFAULT_N_EVALUATIONS
    assert args.output_dir == benchmark.DEFAULT_OUTPUT_DIR
    assert args.report_dir == benchmark.DEFAULT_REPORT_DIR
    assert args.csv_name == benchmark.DEFAULT_CSV_NAME
    assert args.plot is False


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_complexity_scaling.py",
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
            "workspace_loading",
            "model_creation",
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
            "2.5",
            "--n-scan-points",
            "11",
            "--output-dir",
            "results/custom",
            "--output-name",
            "custom.json",
            "--report-dir",
            "reports/custom",
            "--csv-name",
            "custom.csv",
            "--plot",
            "--plot-dir",
            "plots/custom",
        ],
    )

    args = benchmark.parse_args()

    assert args.workspaces == [Path("a.json"), Path("b.json")]
    assert args.targets == ["L_ch0", "L_ch1"]
    assert args.modes == ["FAST_RUN", "FAST_COMPILE"]
    assert args.stages == ["workspace_loading", "model_creation"]
    assert args.n_runs == 5
    assert args.n_evaluations == 7
    assert args.distribution == "sig_ch0"
    assert args.scan_parameter == "mu_sig"
    assert args.scan_min == 0.5
    assert args.scan_max == 2.5
    assert args.n_scan_points == 11
    assert args.output_dir == Path("results/custom")
    assert args.output_name == "custom.json"
    assert args.report_dir == Path("reports/custom")
    assert args.csv_name == "custom.csv"
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

    second_result = valid_result.copy()
    second_result["workspace"] = "workspace2.json"
    second_result["workspace_path"] = "/tmp/workspace2.json"
    second_result["workspace_size_bytes"] = valid_result["workspace_size_bytes"] * 2

    benchmark.make_plots([valid_result, second_result], tmp_path)

    metric_keys = [call["metric_key"] for call in calls]
    assert metric_keys == [
        "total_setup_time_ms",
        "compiled_evaluation_ms_per_eval",
        "pdf_evaluation_ms_per_eval",
        "nll_scan_ms_per_point",
        "total_peak_rss_delta_mb",
    ]


def test_make_plots_skips_optional_metric_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    calls = []
    minimal_result = {
        "workspace": "workspace.json",
        "workspace_size_bytes": 1024,
        "total_setup_time_seconds": 0.1,
        "total_peak_rss_delta_mb": 0.0,
        "status": "success",
    }
    second_result = minimal_result.copy()
    second_result["workspace"] = "workspace2.json"
    second_result["workspace_size_bytes"] = 2048

    monkeypatch.setattr(
        benchmark, "make_bar_plot", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda results, key: False)

    benchmark.make_plots([minimal_result, second_result], tmp_path)

    assert len(calls) == 1
    assert calls[0]["metric_key"] == "total_setup_time_ms"


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


def test_main_saves_json_csv_and_verifies_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    output_dir = tmp_path / "results"
    report_dir = tmp_path / "reports"
    saved_payloads = []
    verified_paths = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_complexity_scaling.py",
            "--workspaces",
            str(workspace_path),
            "--stages",
            "workspace_loading",
            "--output-dir",
            str(output_dir),
            "--output-name",
            "result.json",
            "--report-dir",
            str(report_dir),
            "--csv-name",
            "summary.csv",
        ],
    )
    monkeypatch.setattr(
        benchmark, "get_context", lambda method: FakeContext(valid_result)
    )
    monkeypatch.setattr(benchmark, "resolve_stages", lambda stages: stages)
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)

    def fake_save_json(payload: dict[str, Any], output_path: Path) -> None:
        saved_payloads.append(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}")

    def fake_write_summary_csv(
        results: list[dict[str, Any]], output_path: Path
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("benchmark\nmodel_complexity_scaling\n")

    def fake_verify_output_file(output_path: Path) -> None:
        verified_paths.append(output_path)

    monkeypatch.setattr(benchmark, "save_json", fake_save_json)
    monkeypatch.setattr(benchmark, "write_summary_csv", fake_write_summary_csv)
    monkeypatch.setattr(benchmark, "verify_output_file", fake_verify_output_file)

    benchmark.main()

    assert saved_payloads[0]["benchmark"] == "model_complexity_scaling"
    assert saved_payloads[0]["n_results"] == 1
    assert saved_payloads[0]["results"] == [valid_result]
    assert verified_paths == [output_dir / "result.json", report_dir / "summary.csv"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--n-runs", "0"], "--n-runs must be at least 1"),
        (["--n-evaluations", "0"], "--n-evaluations must be at least 1"),
        (["--distribution", ""], "distribution must be a non-empty string"),
        (["--scan-parameter", ""], "scan_parameter must be a non-empty string"),
        (
            ["--scan-min", "5", "--scan-max", "5"],
            "scan_min must be smaller than scan_max",
        ),
        (["--n-scan-points", "1"], "--n-scan-points must be at least 2"),
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
        ["run_model_complexity_scaling.py", "--workspaces", str(workspace_path), *argv],
    )

    with pytest.raises(ValueError, match=message):
        benchmark.main()


def test_main_rejects_missing_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_complexity_scaling.py",
            "--workspaces",
            str(tmp_path / "missing.json"),
        ],
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
            "run_model_complexity_scaling.py",
            "--workspaces",
            str(workspace_path),
            "--stages",
            "workspace_loading",
            "--output-dir",
            str(tmp_path / "results"),
            "--report-dir",
            str(tmp_path / "reports"),
            "--plot",
        ],
    )
    monkeypatch.setattr(
        benchmark, "get_context", lambda method: FakeContext(valid_result)
    )
    monkeypatch.setattr(benchmark, "resolve_stages", lambda stages: stages)
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, output_path: (
            output_path.parent.mkdir(parents=True, exist_ok=True)
            or output_path.write_text("{}")
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "write_summary_csv",
        lambda results, output_path: (
            output_path.parent.mkdir(parents=True, exist_ok=True)
            or output_path.write_text("benchmark\n")
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
    valid_result: dict[str, Any],
) -> None:
    workspace_a = tmp_path / "a.json"
    workspace_b = tmp_path / "b.json"
    workspace_a.write_text("{}")
    workspace_b.write_text("{}")
    calls = []

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_complexity_scaling.py",
            "--workspaces",
            str(workspace_a),
            str(workspace_b),
            "--stages",
            "workspace_loading",
            "--output-dir",
            str(tmp_path / "results"),
            "--report-dir",
            str(tmp_path / "reports"),
            "--plot",
        ],
    )
    monkeypatch.setattr(
        benchmark, "get_context", lambda method: FakeContext(valid_result)
    )
    monkeypatch.setattr(benchmark, "resolve_stages", lambda stages: stages)
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, output_path: (
            output_path.parent.mkdir(parents=True, exist_ok=True)
            or output_path.write_text("{}")
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "write_summary_csv",
        lambda results, output_path: (
            output_path.parent.mkdir(parents=True, exist_ok=True)
            or output_path.write_text("benchmark\n")
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
    second_result = valid_result.copy()
    second_result["workspace"] = "workspace2.json"
    second_result["workspace_path"] = "/tmp/workspace2.json"
    second_result["workspace_size_bytes"] = valid_result["workspace_size_bytes"] * 2

    benchmark.make_plots([valid_result, second_result], tmp_path)

    assert (tmp_path / "model_complexity_total_setup_time.png").exists()
    assert (tmp_path / "model_complexity_compiled_evaluation_time.png").exists()
    assert (tmp_path / "model_complexity_nll_scan_time.png").exists()
    assert (tmp_path / "model_complexity_peak_rss_delta.png").exists()


def test_run_single_scaling_benchmark_with_mocked_stage(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def workspace_loading_stage(*args: Any) -> dict[str, Any]:
        return {
            "benchmark": "workspace_loading",
            "status": "success",
            "wall_time_seconds_mean": 0.01,
            "wall_time_seconds_std": 0.0,
            "current_rss_delta_mb": 1.0,
            "peak_rss_delta_mb": 2.0,
        }

    monkeypatch.setattr(benchmark, "resolve_stages", lambda stages: stages)
    monkeypatch.setattr(
        benchmark,
        "build_stage_specs",
        lambda **kwargs: [("workspace_loading", workspace_loading_stage, ())],
    )

    result = benchmark.run_single_scaling_benchmark(
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
        n_scan_points=3,
    )

    assert result["status"] == "success"
    assert result["workspace"] == workspace_path.name
    assert result["selected_stages"] == ["workspace_loading"]
    assert result["workspace_size_bytes"] == workspace_path.stat().st_size
    assert result["total_setup_time_seconds"] == pytest.approx(0.01)
    assert result["total_peak_rss_delta_mb"] == pytest.approx(2.0)


def test_main_mocked_run_writes_output_json_and_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    output_dir = tmp_path / "results"
    report_dir = tmp_path / "reports"
    output_path = output_dir / "model_complexity_scaling_result.json"
    csv_path = report_dir / "model_complexity_scaling_summary.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_complexity_scaling.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "L_ch0",
            "--modes",
            "FAST_RUN",
            "--stages",
            "workspace_loading",
            "--n-runs",
            "1",
            "--n-evaluations",
            "1",
            "--n-scan-points",
            "3",
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(report_dir),
        ],
    )
    monkeypatch.setattr(
        benchmark, "get_context", lambda method: FakeContext(valid_result)
    )
    monkeypatch.setattr(benchmark, "resolve_stages", lambda stages: stages)
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)

    benchmark.main()

    assert output_path.exists()
    assert csv_path.exists()

    with output_path.open() as file:
        payload = json.load(file)

    assert payload["benchmark"] == "model_complexity_scaling"
    assert payload["n_results"] == 1
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][0]["selected_stages"] == valid_result["selected_stages"]

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["benchmark"] == "model_complexity_scaling"


def test_print_result_with_all_optional_fields(
    capsys: pytest.CaptureFixture[str],
    valid_result: dict[str, Any],
) -> None:
    result = dict(valid_result)
    result.update(
        {
            "compiled_evaluation_average_runtime_seconds_per_evaluation": 0.001,
            "pdf_evaluation_average_runtime_seconds_per_evaluation": 0.002,
            "nll_scan_runtime_per_scan_point_seconds": 0.003,
            "compiled_evaluation_all_outputs_finite": True,
            "nll_scan_all_values_finite": True,
            "nll_scan_minimum_scan_value": 1.0,
        }
    )

    benchmark.print_result(result)

    output = capsys.readouterr().out

    assert "compiled eval / point" in output
    assert "PDF eval / point" in output
    assert "NLL scan / point" in output
    assert "compiled output finite" in output
    assert "NLL values finite" in output
    assert "NLL minimum at" in output


def test_make_plots_calls_all_optional_metric_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    calls = []

    def fake_make_bar_plot(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(benchmark, "make_bar_plot", fake_make_bar_plot)
    monkeypatch.setattr(benchmark, "should_plot_metric", lambda results, key: True)

    result = dict(valid_result)
    result.update(
        {
            "compiled_evaluation_average_runtime_seconds_per_evaluation": 0.001,
            "pdf_evaluation_average_runtime_seconds_per_evaluation": 0.002,
            "nll_scan_runtime_per_scan_point_seconds": 0.003,
            "total_peak_rss_delta_mb": 5.0,
        }
    )

    second_result = result.copy()
    second_result["workspace"] = "workspace2.json"
    second_result["workspace_path"] = "/tmp/workspace2.json"
    second_result["workspace_size_bytes"] = result["workspace_size_bytes"] * 2

    benchmark.make_plots(results=[result, second_result], plot_dir=tmp_path)

    metric_keys = [call["metric_key"] for call in calls]

    assert metric_keys == [
        "total_setup_time_ms",
        "compiled_evaluation_ms_per_eval",
        "pdf_evaluation_ms_per_eval",
        "nll_scan_ms_per_point",
        "total_peak_rss_delta_mb",
    ]


def test_main_creates_real_plots_for_multiple_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    output_dir = tmp_path / "results"
    report_dir = tmp_path / "reports"
    plot_dir = tmp_path / "plots"

    workspace_a = workspace_path
    workspace_b = tmp_path / "workspace_b.json"
    workspace_b.write_text("{}")

    result_a = dict(valid_result)
    result_a["workspace"] = workspace_a.name
    result_a["workspace_path"] = str(workspace_a)

    result_b = dict(valid_result)
    result_b["workspace"] = workspace_b.name
    result_b["workspace_path"] = str(workspace_b)

    class MultiResultPool:
        def __init__(self) -> None:
            self.index = 0

        def __enter__(self) -> "MultiResultPool":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def apply(self, func: Any, args: tuple[Any, ...]) -> dict[str, Any]:
            workspace = args[0]
            return result_a if workspace == workspace_a else result_b

    class MultiResultContext:
        def Pool(self, processes: int) -> MultiResultPool:
            assert processes == 1
            return MultiResultPool()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_complexity_scaling.py",
            "--workspaces",
            str(workspace_a),
            str(workspace_b),
            "--stages",
            "workspace_loading",
            "--n-runs",
            "1",
            "--n-evaluations",
            "1",
            "--n-scan-points",
            "2",
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(report_dir),
            "--plot",
            "--plot-dir",
            str(plot_dir),
        ],
    )
    monkeypatch.setattr(benchmark, "get_context", lambda method: MultiResultContext())
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)

    benchmark.main()

    assert (output_dir / benchmark.DEFAULT_OUTPUT_NAME).exists()
    assert (report_dir / benchmark.DEFAULT_CSV_NAME).exists()
    assert (plot_dir / "model_complexity_total_setup_time.png").exists()


def test_make_stage_error_result_contains_structured_error() -> None:
    try:
        raise RuntimeError("stage exploded")
    except RuntimeError as exc:
        result = benchmark.make_stage_error_result("workspace_loading", exc)

    assert result["benchmark"] == "workspace_loading"
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "stage exploded"
    assert "RuntimeError" in result["traceback"]


def test_make_error_result_handles_missing_workspace_size(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    try:
        raise FileNotFoundError("missing workspace")
    except FileNotFoundError as exc:
        result = benchmark.make_error_result(
            workspace_path=missing,
            target="L_ch0",
            mode="FAST_RUN",
            n_runs=1,
            n_evaluations=1,
            stages=["workspace_loading"],
            distribution="sig_ch0",
            scan_parameter="mu_sig",
            scan_min=0.0,
            scan_max=5.0,
            n_scan_points=3,
            exc=exc,
        )

    assert result["status"] == "failed"
    assert result["workspace_size_bytes"] is None
    assert result["error_type"] == "FileNotFoundError"
    assert result["selected_stages"] == ["workspace_loading"]


def test_print_failed_result_outputs_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark.print_failed_result(
        {
            "workspace": "workspace.json",
            "target": "L_ch0",
            "mode": "FAST_RUN",
            "error_type": "RuntimeError",
            "error_message": "boom",
        }
    )

    output = capsys.readouterr().out
    assert "Model complexity scaling benchmark FAILED" in output
    assert "workspace.json" in output
    assert "RuntimeError: boom" in output


def test_print_result_with_full_setup_stage_timings(
    capsys: pytest.CaptureFixture[str], valid_result: dict[str, Any]
) -> None:
    result = dict(valid_result)
    result.update(
        {
            "model_creation_wall_time_seconds_mean": 0.2,
            "log_prob_construction_wall_time_seconds_mean": 0.3,
            "log_prob_compilation_wall_time_seconds_mean": 0.4,
        }
    )

    benchmark.print_result(result)

    output = capsys.readouterr().out
    assert "model creation" in output
    assert "log_prob construction" in output
    assert "log_prob compilation" in output
    assert "200.000 ms" in output
    assert "300.000 ms" in output
    assert "400.000 ms" in output


def test_make_plots_skips_when_fewer_than_two_successful_results(
    tmp_path: Path,
    valid_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark.make_plots([valid_result], tmp_path)

    output = capsys.readouterr().out
    assert "Skipping plots" in output
    assert not (tmp_path / "model_complexity_total_setup_time.png").exists()
    assert not (tmp_path / "model_complexity_compiled_evaluation_time.png").exists()
    assert not (tmp_path / "model_complexity_pdf_evaluation_time.png").exists()
    assert not (tmp_path / "model_complexity_nll_scan_time.png").exists()
    assert not (tmp_path / "model_complexity_peak_rss_delta.png").exists()


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
        [
            "run_model_complexity_scaling.py",
            "--workspaces",
            str(workspace_path),
            *argv,
        ],
    )

    with pytest.raises(ValueError, match=message):
        benchmark.main()


def test_main_records_pool_error_as_failed_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
) -> None:
    saved_payloads: list[dict[str, Any]] = []
    printed_failed_results: list[dict[str, Any]] = []
    output_dir = tmp_path / "results"
    report_dir = tmp_path / "reports"

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

    def fake_save_json(payload: dict[str, Any], output_path: Path) -> None:
        saved_payloads.append(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_complexity_scaling.py",
            "--workspaces",
            str(workspace_path),
            "--stages",
            "workspace_loading",
            "--n-runs",
            "1",
            "--n-evaluations",
            "1",
            "--n-scan-points",
            "2",
            "--output-dir",
            str(output_dir),
            "--report-dir",
            str(report_dir),
        ],
    )
    monkeypatch.setattr(benchmark, "get_context", lambda method: FailingContext())
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)
    monkeypatch.setattr(
        benchmark,
        "print_failed_result",
        lambda result: printed_failed_results.append(result),
    )
    monkeypatch.setattr(benchmark, "save_json", fake_save_json)
    monkeypatch.setattr(
        benchmark,
        "write_summary_csv",
        lambda results, output_path: (
            output_path.parent.mkdir(parents=True, exist_ok=True)
            or output_path.write_text("benchmark\n")
        ),
    )
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)

    benchmark.main()

    assert len(saved_payloads) == 1
    result = saved_payloads[0]["results"][0]
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "pool failed"
    assert printed_failed_results == [result]


def test_module_main_guard_runs_main(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model_complexity_scaling.py",
            "--n-runs",
            "0",
        ],
    )

    with pytest.raises(ValueError, match="--n-runs must be at least 1"):
        runpy.run_module("src.run_model_complexity_scaling", run_name="__main__")
