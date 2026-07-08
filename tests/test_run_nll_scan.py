from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src import run_nll_scan as benchmark


class FakeCompiled:
    def __init__(self, log_prob: float = -1.0) -> None:
        self.log_prob = log_prob
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **inputs: Any) -> tuple[np.ndarray]:
        self.calls.append(inputs)
        return (np.asarray([self.log_prob], dtype=float),)


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    return path


@pytest.fixture
def base_inputs() -> dict[str, Any]:
    return {
        "mu_sig": np.asarray([1.0]),
        "obs": np.asarray([2.0]),
    }


@pytest.fixture
def valid_result() -> dict[str, Any]:
    return {
        "benchmark": "nll_scan",
        "workspace": "workspace.json",
        "workspace_path": "/tmp/workspace.json",
        "target": "L_ch0",
        "mode": "FAST_RUN",
        "scan_parameter": "mu_sig",
        "scan_min": 0.0,
        "scan_max": 2.0,
        "n_scan_points": 3,
        "scan_values": [0.0, 1.0, 2.0],
        "nll_values": [2.0, 1.0, 3.0],
        "total_runtime_seconds": 0.3,
        "runtime_per_scan_point_seconds": 0.1,
        "throughput_scan_points_per_second": 10.0,
        "first_nll_value": 2.0,
        "last_nll_value": 3.0,
        "memory_n_scan_points": 3,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 101.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 122.0,
        "peak_rss_delta_mb": 2.0,
        "status": "success",
        "n_scan_outputs": 3,
        "all_nll_values_finite": True,
        "minimum_index": 1,
        "minimum_scan_value": 1.0,
        "minimum_nll_value": 1.0,
        "nll_min": 1.0,
        "nll_max": 3.0,
        "nll_range": 2.0,
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
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=2,
    )


@pytest.mark.parametrize(
    (
        "target",
        "mode",
        "scan_parameter",
        "scan_min",
        "scan_max",
        "n_scan_points",
        "message",
    ),
    [
        ("", "FAST_RUN", "mu_sig", 0.0, 5.0, 2, "target must be a non-empty string"),
        ("L_ch0", "", "mu_sig", 0.0, 5.0, 2, "mode must be a non-empty string"),
        (
            "L_ch0",
            "FAST_RUN",
            "",
            0.0,
            5.0,
            2,
            "scan_parameter must be a non-empty string",
        ),
        ("L_ch0", "FAST_RUN", "mu_sig", math.nan, 5.0, 2, "scan_min must be finite"),
        ("L_ch0", "FAST_RUN", "mu_sig", 0.0, math.inf, 2, "scan_max must be finite"),
        (
            "L_ch0",
            "FAST_RUN",
            "mu_sig",
            5.0,
            5.0,
            2,
            "scan_min must be smaller than scan_max",
        ),
        (
            "L_ch0",
            "FAST_RUN",
            "mu_sig",
            0.0,
            5.0,
            1,
            "n_scan_points must be at least 2",
        ),
    ],
)
def test_validate_benchmark_config_rejects_invalid_values(
    target: str,
    mode: str,
    scan_parameter: str,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_benchmark_config(
            target=target,
            mode=mode,
            scan_parameter=scan_parameter,
            scan_min=scan_min,
            scan_max=scan_max,
            n_scan_points=n_scan_points,
        )


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


def test_extract_log_prob_success() -> None:
    assert benchmark.extract_log_prob((np.asarray([[2.5]]),)) == 2.5


@pytest.mark.parametrize("result", [1.0, [1.0], {"x": 1.0}])
def test_extract_log_prob_rejects_non_tuple(result: Any) -> None:
    with pytest.raises(TypeError, match="Expected compiled result to be a tuple"):
        benchmark.extract_log_prob(result)


def test_extract_log_prob_rejects_empty_tuple() -> None:
    with pytest.raises(ValueError, match="Compiled result tuple is empty"):
        benchmark.extract_log_prob(())


@pytest.mark.parametrize("result", [(np.asarray([]),), ([math.nan],), ([math.inf],)])
def test_extract_log_prob_rejects_empty_or_non_finite(result: tuple[Any, ...]) -> None:
    with pytest.raises(ValueError):
        benchmark.extract_log_prob(result)


def test_make_scan_values_success() -> None:
    assert benchmark.make_scan_values(0.0, 2.0, 3) == [0.0, 1.0, 2.0]


def test_make_scan_values_rejects_invalid_points() -> None:
    with pytest.raises(ValueError, match="n_scan_points must be at least 2"):
        benchmark.make_scan_values(0.0, 1.0, 1)


def test_make_scan_values_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="scan_min must be smaller than scan_max"):
        benchmark.make_scan_values(2.0, 1.0, 3)


def test_set_scan_parameter_scalar() -> None:
    inputs = {"mu_sig": 1.0, "other": 3.0}

    updated = benchmark.set_scan_parameter(inputs, "mu_sig", 2.5)

    assert updated["mu_sig"] == 2.5
    assert inputs["mu_sig"] == 1.0
    assert updated["other"] == 3.0


def test_set_scan_parameter_array(base_inputs: dict[str, Any]) -> None:
    updated = benchmark.set_scan_parameter(base_inputs, "mu_sig", 2.5)

    assert np.allclose(updated["mu_sig"], np.asarray([2.5]))
    assert np.allclose(base_inputs["mu_sig"], np.asarray([1.0]))


def test_set_scan_parameter_rejects_empty_name(base_inputs: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="parameter_name must be a non-empty string"):
        benchmark.set_scan_parameter(base_inputs, "", 1.0)


def test_set_scan_parameter_rejects_non_finite_value(
    base_inputs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="Scan value must be finite"):
        benchmark.set_scan_parameter(base_inputs, "mu_sig", math.nan)


def test_set_scan_parameter_rejects_missing_parameter(
    base_inputs: dict[str, Any],
) -> None:
    with pytest.raises(KeyError, match="Scan parameter 'missing' not found"):
        benchmark.set_scan_parameter(base_inputs, "missing", 1.0)


def test_evaluate_nll_scan_success(base_inputs: dict[str, Any]) -> None:
    compiled = FakeCompiled(log_prob=-2.0)

    nll_values = benchmark.evaluate_nll_scan(
        compiled=compiled,
        base_inputs=base_inputs,
        scan_parameter="mu_sig",
        scan_values=[0.0, 1.0, 2.0],
    )

    assert nll_values == [2.0, 2.0, 2.0]
    assert [float(call["mu_sig"][0]) for call in compiled.calls] == [0.0, 1.0, 2.0]


def test_evaluate_nll_scan_empty_grid_returns_empty(
    base_inputs: dict[str, Any],
) -> None:
    assert benchmark.evaluate_nll_scan(FakeCompiled(), base_inputs, "mu_sig", []) == []


def test_evaluate_nll_scan_propagates_compiled_error(
    base_inputs: dict[str, Any],
) -> None:
    class FailingCompiled:
        def __call__(self, **inputs: Any) -> tuple[np.ndarray]:
            raise RuntimeError("compiled failed")

    with pytest.raises(RuntimeError, match="compiled failed"):
        benchmark.evaluate_nll_scan(
            compiled=FailingCompiled(),
            base_inputs=base_inputs,
            scan_parameter="mu_sig",
            scan_values=[1.0],
        )


def test_validate_nll_scan_success() -> None:
    result = benchmark.validate_nll_scan(
        scan_values=[0.0, 1.0, 2.0],
        nll_values=[2.0, 1.0, 3.0],
    )

    assert result["n_scan_outputs"] == 3
    assert result["all_nll_values_finite"] is True
    assert result["minimum_index"] == 1
    assert result["minimum_scan_value"] == 1.0
    assert result["minimum_nll_value"] == 1.0
    assert result["nll_range"] == 2.0


def test_validate_nll_scan_rejects_empty_grid() -> None:
    with pytest.raises(ValueError, match="Scan grid is empty"):
        benchmark.validate_nll_scan([], [])


def test_validate_nll_scan_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        benchmark.validate_nll_scan([0.0, 1.0], [1.0])


@pytest.mark.parametrize(
    "nll_values", [[1.0, math.nan], [1.0, math.inf], [1.0, -math.inf]]
)
def test_validate_nll_scan_rejects_non_finite_values(nll_values: list[float]) -> None:
    with pytest.raises(ValueError, match="NLL scan produced non-finite values"):
        benchmark.validate_nll_scan([0.0, 1.0], nll_values)


def test_measure_nll_scan_timing_success(
    monkeypatch: pytest.MonkeyPatch,
    base_inputs: dict[str, Any],
) -> None:
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([1.0, 1.3]).__next__)

    result = benchmark.measure_nll_scan_timing(
        compiled=FakeCompiled(log_prob=-1.0),
        base_inputs=base_inputs,
        scan_parameter="mu_sig",
        scan_values=[0.0, 1.0, 2.0],
    )

    assert result["total_runtime_seconds"] == pytest.approx(0.3)
    assert result["runtime_per_scan_point_seconds"] == pytest.approx(0.1)
    assert result["throughput_scan_points_per_second"] == pytest.approx(10.0)
    assert result["first_nll_value"] == 1.0
    assert result["last_nll_value"] == 1.0


def test_measure_nll_scan_timing_rejects_empty_grid(
    base_inputs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="Scan grid is empty"):
        benchmark.measure_nll_scan_timing(FakeCompiled(), base_inputs, "mu_sig", [])


def test_measure_nll_scan_timing_handles_zero_runtime(
    monkeypatch: pytest.MonkeyPatch,
    base_inputs: dict[str, Any],
) -> None:
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: 1.0)

    result = benchmark.measure_nll_scan_timing(
        compiled=FakeCompiled(),
        base_inputs=base_inputs,
        scan_parameter="mu_sig",
        scan_values=[0.0, 1.0],
    )

    assert math.isinf(result["throughput_scan_points_per_second"])


def test_measure_nll_scan_timing_propagates_compiled_error(
    base_inputs: dict[str, Any],
) -> None:
    class FailingCompiled:
        def __call__(self, **inputs: Any) -> tuple[np.ndarray]:
            raise RuntimeError("scan failed")

    with pytest.raises(RuntimeError, match="scan failed"):
        benchmark.measure_nll_scan_timing(
            FailingCompiled(), base_inputs, "mu_sig", [1.0]
        )


def test_measure_nll_scan_memory_success(
    monkeypatch: pytest.MonkeyPatch,
    base_inputs: dict[str, Any],
) -> None:
    current_values = iter([100.0, 103.0])
    peak_values = iter([120.0, 125.0])
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: next(current_values))
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: next(peak_values))

    result = benchmark.measure_nll_scan_memory(
        compiled=FakeCompiled(),
        base_inputs=base_inputs,
        scan_parameter="mu_sig",
        scan_values=[0.0, 1.0],
    )

    assert result["memory_n_scan_points"] == 2
    assert result["current_rss_delta_mb"] == 3.0
    assert result["peak_rss_delta_mb"] == 5.0


def test_measure_nll_scan_memory_rejects_empty_grid(
    base_inputs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="Scan grid is empty"):
        benchmark.measure_nll_scan_memory(FakeCompiled(), base_inputs, "mu_sig", [])


def test_measure_nll_scan_memory_propagates_compiled_error(
    base_inputs: dict[str, Any],
) -> None:
    class FailingCompiled:
        def __call__(self, **inputs: Any) -> tuple[np.ndarray]:
            raise RuntimeError("memory failed")

    with pytest.raises(RuntimeError, match="memory failed"):
        benchmark.measure_nll_scan_memory(
            FailingCompiled(), base_inputs, "mu_sig", [1.0]
        )


def test_run_single_benchmark_success(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    base_inputs: dict[str, Any],
) -> None:
    model = object()
    log_prob = object()
    compiled = FakeCompiled(log_prob=-1.0)
    memory_summary = {
        "memory_n_scan_points": 3,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 101.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 122.0,
        "peak_rss_delta_mb": 2.0,
    }
    timing_summary = {
        "total_runtime_seconds": 0.3,
        "runtime_per_scan_point_seconds": 0.1,
        "throughput_scan_points_per_second": 10.0,
        "first_nll_value": 1.0,
        "last_nll_value": 1.0,
    }

    monkeypatch.setattr(benchmark, "build_log_prob", lambda **kwargs: (model, log_prob))
    monkeypatch.setattr(benchmark, "compile_log_prob", lambda log_prob: compiled)
    monkeypatch.setattr(
        benchmark, "build_validation_inputs", lambda model, compiled: base_inputs
    )
    monkeypatch.setattr(
        benchmark, "measure_nll_scan_memory", lambda **kwargs: memory_summary
    )
    monkeypatch.setattr(
        benchmark, "measure_nll_scan_timing", lambda **kwargs: timing_summary
    )

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=2.0,
        n_scan_points=3,
    )

    assert result["benchmark"] == "nll_scan"
    assert result["status"] == "success"
    assert result["workspace"] == "workspace.json"
    assert result["scan_values"] == [0.0, 1.0, 2.0]
    assert result["nll_values"] == [1.0, 1.0, 1.0]
    assert result["n_scan_outputs"] == 3
    assert result["minimum_index"] == 0


def test_run_single_benchmark_rejects_invalid_config(workspace_path: Path) -> None:
    with pytest.raises(ValueError, match="n_scan_points must be at least 2"):
        benchmark.run_single_benchmark(
            workspace_path=workspace_path,
            target="L_ch0",
            mode="FAST_RUN",
            scan_parameter="mu_sig",
            scan_min=0.0,
            scan_max=1.0,
            n_scan_points=1,
        )


def test_run_single_benchmark_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.run_single_benchmark(
            workspace_path=tmp_path / "missing.json",
            target="L_ch0",
            mode="FAST_RUN",
            scan_parameter="mu_sig",
            scan_min=0.0,
            scan_max=1.0,
            n_scan_points=2,
        )


def test_run_single_benchmark_propagates_build_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    def failing_build_log_prob(**kwargs: Any) -> tuple[object, object]:
        raise RuntimeError("build failed")

    monkeypatch.setattr(benchmark, "build_log_prob", failing_build_log_prob)

    with pytest.raises(RuntimeError, match="build failed"):
        benchmark.run_single_benchmark(
            workspace_path=workspace_path,
            target="L_ch0",
            mode="FAST_RUN",
            scan_parameter="mu_sig",
            scan_min=0.0,
            scan_max=1.0,
            n_scan_points=2,
        )


def test_run_single_benchmark_propagates_scan_parameter_error(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
) -> None:
    monkeypatch.setattr(
        benchmark, "build_log_prob", lambda **kwargs: (object(), object())
    )
    monkeypatch.setattr(benchmark, "compile_log_prob", lambda log_prob: FakeCompiled())
    monkeypatch.setattr(
        benchmark, "build_validation_inputs", lambda model, compiled: {"other": 1.0}
    )

    with pytest.raises(KeyError, match="Scan parameter 'mu_sig' not found"):
        benchmark.run_single_benchmark(
            workspace_path=workspace_path,
            target="L_ch0",
            mode="FAST_RUN",
            scan_parameter="mu_sig",
            scan_min=0.0,
            scan_max=1.0,
            n_scan_points=2,
        )


def test_print_result_outputs_summary(
    capsys: pytest.CaptureFixture[str],
    valid_result: dict[str, Any],
) -> None:
    benchmark.print_result(valid_result)

    output = capsys.readouterr().out

    assert "NLL scan benchmark" in output
    assert "Timing" in output
    assert "Memory" in output
    assert "Validation" in output
    assert "mu_sig" in output


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_nll_scan.py"])

    args = benchmark.parse_args()

    assert args.workspaces == [benchmark.DEFAULT_WORKSPACE]
    assert args.targets == [benchmark.DEFAULT_TARGET]
    assert args.modes == [benchmark.DEFAULT_MODE]
    assert args.scan_parameter == benchmark.DEFAULT_SCAN_PARAMETER
    assert args.scan_min == benchmark.DEFAULT_SCAN_MIN
    assert args.scan_max == benchmark.DEFAULT_SCAN_MAX
    assert args.n_scan_points == [benchmark.DEFAULT_N_SCAN_POINTS]
    assert args.output_dir == benchmark.DEFAULT_OUTPUT_DIR
    assert args.output_name == benchmark.DEFAULT_OUTPUT_NAME
    assert args.plot is False


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nll_scan.py",
            "--workspaces",
            "a.json",
            "b.json",
            "--targets",
            "L_ch0",
            "L_ch1",
            "--modes",
            "FAST_RUN",
            "FAST_COMPILE",
            "--scan-parameter",
            "mu_sig",
            "--scan-min",
            "0.5",
            "--scan-max",
            "2.5",
            "--n-scan-points",
            "3",
            "5",
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
    assert args.scan_parameter == "mu_sig"
    assert args.scan_min == 0.5
    assert args.scan_max == 2.5
    assert args.n_scan_points == [3, 5]
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
    monkeypatch.setattr(
        benchmark, "make_bar_plot", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        benchmark, "should_plot_metric", lambda results, metric_key: True
    )

    second_result = valid_result.copy()
    second_result["n_scan_points"] = 5

    benchmark.make_plots([valid_result, second_result], tmp_path)

    assert len(calls) == 4
    assert calls[0]["metric_key"] == "total_runtime_ms"
    assert calls[1]["metric_key"] == "runtime_per_scan_point_ms"
    assert calls[2]["metric_key"] == "current_rss_delta_mb"
    assert calls[3]["metric_key"] == "peak_rss_delta_mb"


def test_make_plots_skips_optional_memory_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    calls = []
    monkeypatch.setattr(
        benchmark, "make_bar_plot", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        benchmark, "should_plot_metric", lambda results, metric_key: False
    )

    second_result = valid_result.copy()
    second_result["n_scan_points"] = 5

    benchmark.make_plots([valid_result, second_result], tmp_path)

    assert len(calls) == 2
    assert [call["metric_key"] for call in calls] == [
        "total_runtime_ms",
        "runtime_per_scan_point_ms",
    ]


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
            "run_nll_scan.py",
            "--workspaces",
            str(workspace_path),
            "--n-scan-points",
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
            "benchmark": "nll_scan",
            "n_results": 1,
            "results": [valid_result],
        }
    ]
    assert verified_paths == [output_dir / output_name]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["run_nll_scan.py", "--scan-parameter", ""],
            "--scan-parameter must be a non-empty string",
        ),
        (["run_nll_scan.py", "--scan-min", "nan"], "--scan-min must be finite"),
        (["run_nll_scan.py", "--scan-max", "inf"], "--scan-max must be finite"),
        (
            ["run_nll_scan.py", "--scan-min", "5", "--scan-max", "5"],
            "--scan-min must be smaller",
        ),
        (
            ["run_nll_scan.py", "--n-scan-points", "1"],
            "--n-scan-points values must be at least 2",
        ),
    ],
)
def test_main_rejects_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match=message):
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
            "run_nll_scan.py",
            "--workspaces",
            str(missing_workspace),
            "--n-scan-points",
            "3",
            "--output-dir",
            str(output_dir),
        ],
    )

    class ApplyingPool:
        def __enter__(self) -> "ApplyingPool":
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            return None

        def apply(self, func: Any, args: tuple[Any, ...]) -> dict[str, Any]:
            return func(*args)

    class ApplyingContext:
        def Pool(self, processes: int) -> ApplyingPool:
            assert processes == 1
            return ApplyingPool()

    def fake_save_json(payload: dict[str, Any], output_path: Path) -> None:
        saved_payloads.append(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}")

    monkeypatch.setattr(benchmark, "get_context", lambda method: ApplyingContext())
    monkeypatch.setattr(benchmark, "save_json", fake_save_json)
    monkeypatch.setattr(benchmark, "verify_output_file", lambda output_path: None)

    benchmark.main()

    assert saved_payloads[0]["n_results"] == 1
    result = saved_payloads[0]["results"][0]
    assert result["status"] == "failed"
    assert result["error_type"] == "FileNotFoundError"
    assert "Workspace file does not exist" in result["error_message"]
    assert result["workspace"] == "missing.json"


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
            "run_nll_scan.py",
            "--workspaces",
            str(workspace_path),
            "--n-scan-points",
            "3",
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
        benchmark, "make_plots", lambda *args, **kwargs: calls.append(kwargs)
    )

    benchmark.main()

    assert len(calls) == 1
    assert len(calls[0]["results"]) == 1
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
            "run_nll_scan.py",
            "--workspaces",
            str(workspace_path),
            "--n-scan-points",
            "3",
            "5",
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


def test_make_plots_skips_when_less_than_two_successful_results(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    benchmark.make_plots([valid_result], tmp_path)

    output = capsys.readouterr().out
    assert "Skipping plots" in output
    assert list(tmp_path.iterdir()) == []


def test_print_error_result_outputs_failure_summary(
    capsys: pytest.CaptureFixture[str],
    workspace_path: Path,
) -> None:
    result = benchmark.make_error_result(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=1.0,
        n_scan_points=3,
        exc=RuntimeError("boom"),
    )

    benchmark.print_error_result(result)

    output = capsys.readouterr().out
    assert "NLL scan benchmark FAILED" in output
    assert "RuntimeError: boom" in output
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert "RuntimeError" in result["traceback"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["run_nll_scan.py", "--targets", ""],
            "--targets must contain only non-empty strings",
        ),
        (
            ["run_nll_scan.py", "--modes", ""],
            "--modes must contain only non-empty strings",
        ),
    ],
)
def test_main_rejects_empty_targets_and_modes(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    message: str,
) -> None:
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ValueError, match=message):
        benchmark.main()


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
            "run_nll_scan.py",
            "--workspaces",
            str(workspace_path),
            "--n-scan-points",
            "3",
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


def test_main_module_entrypoint_reaches_main(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    monkeypatch.setattr(sys, "argv", ["run_nll_scan.py", "--scan-parameter", ""])

    with pytest.raises(ValueError, match="--scan-parameter must be a non-empty string"):
        runpy.run_module("src.run_nll_scan", run_name="__main__")


def test_run_single_benchmark_mocked_workspace(
    monkeypatch: pytest.MonkeyPatch,
    workspace_path: Path,
    base_inputs: dict[str, Any],
) -> None:
    compiled = FakeCompiled(log_prob=-2.0)

    monkeypatch.setattr(
        benchmark, "build_log_prob", lambda **kwargs: (object(), object())
    )
    monkeypatch.setattr(benchmark, "compile_log_prob", lambda log_prob: compiled)
    monkeypatch.setattr(
        benchmark, "build_validation_inputs", lambda model, compiled: base_inputs
    )
    monkeypatch.setattr(
        benchmark,
        "measure_nll_scan_memory",
        lambda **kwargs: {
            "memory_n_scan_points": 3,
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
        "measure_nll_scan_timing",
        lambda **kwargs: {
            "total_runtime_seconds": 0.3,
            "runtime_per_scan_point_seconds": 0.1,
            "throughput_scan_points_per_second": 10.0,
            "first_nll_value": 2.0,
            "last_nll_value": 2.0,
        },
    )

    result = benchmark.run_single_benchmark(
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=1.0,
        n_scan_points=3,
    )

    assert result["status"] == "success"
    assert result["workspace"] == "workspace.json"
    assert result["target"] == "L_ch0"
    assert result["scan_parameter"] == "mu_sig"
    assert result["n_scan_points"] == 3
    assert result["scan_values"] == [0.0, 0.5, 1.0]
    assert result["nll_values"] == [2.0, 2.0, 2.0]
    assert result["all_nll_values_finite"] is True
    assert result["runtime_per_scan_point_seconds"] == 0.1


def test_main_mocked_run_writes_output_json_and_uses_spawn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    valid_result: dict[str, Any],
) -> None:
    output_dir = tmp_path / "results"
    output_name = "nll_scan_result.json"
    output_path = output_dir / output_name

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nll_scan.py",
            "--workspaces",
            str(workspace_path),
            "--targets",
            "L_ch0",
            "--modes",
            "FAST_RUN",
            "--scan-parameter",
            "mu_sig",
            "--scan-min",
            "0",
            "--scan-max",
            "1",
            "--n-scan-points",
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

    benchmark.main()

    assert output_path.exists()
    assert output_path.is_file()

    with output_path.open() as file:
        payload = json.load(file)

    assert payload["benchmark"] == "nll_scan"
    assert payload["n_results"] == 1
    assert len(payload["results"]) == 1
    assert payload["results"][0]["status"] == "success"
    assert payload["results"][0]["workspace"] == "workspace.json"
    assert payload["results"][0]["scan_parameter"] == "mu_sig"


def test_make_plots_real_png_files_created(
    tmp_path: Path,
    valid_result: dict[str, Any],
) -> None:
    second_result = valid_result.copy()
    second_result["n_scan_points"] = 5

    benchmark.make_plots([valid_result, second_result], tmp_path)

    assert (tmp_path / "nll_scan_total_runtime.png").exists()
    assert (tmp_path / "nll_scan_runtime_per_point.png").exists()
    assert (tmp_path / "nll_scan_current_rss_delta.png").exists()
    assert (tmp_path / "nll_scan_peak_rss_delta.png").exists()
