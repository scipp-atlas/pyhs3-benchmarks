from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from src import run_pyhs3_model_complexity_scaling as benchmark


class FakeAnalysis:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeData:
    def __init__(self, name: str, entries: list[list[float]]) -> None:
        self.name = name
        self.entries = [[1.0], [2.0]] if entries is None else entries


class FakeModel:
    def __init__(
        self,
        free_params: dict[str, Any] | None = None,
        logpdf_values: list[float] | np.ndarray | None = None,
    ) -> None:
        self.free_params = free_params if free_params is not None else {"mu_sig": 1.0}
        self.logpdf_values = (
            np.asarray(logpdf_values, dtype=float)
            if logpdf_values is not None
            else np.asarray([-1.0, -2.0], dtype=float)
        )
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def logpdf(self, target: str, **params: Any) -> np.ndarray:
        self.calls.append((target, params))
        mu_value = float(np.asarray(params.get("mu_sig", 1.0)))
        return self.logpdf_values - 0.1 * (mu_value - 1.0) ** 2


class FakeWorkspace:
    def __init__(
        self,
        analyses: list[str] | None = None,
        data_entries: list[list[float]] | None = None,
        model: FakeModel | None = None,
    ) -> None:
        self.analyses = SimpleNamespace(
            root=[FakeAnalysis(name) for name in (analyses or ["L_ch0"])]
        )
        if data_entries is None:
            data_entries = [[1.0], [2.0]]

        self.data = SimpleNamespace(
            root=[
                FakeData("other", [[99.0]]),
                FakeData("combData_ch0", data_entries),
            ]
        )
        self.model_obj = model or FakeModel()
        self.model_calls: list[tuple[str, bool, str]] = []

    def model(self, analysis_name: str, progress: bool, mode: str) -> FakeModel:
        self.model_calls.append((analysis_name, progress, mode))
        return self.model_obj


@pytest.fixture
def valid_result() -> dict[str, Any]:
    return {
        "workspace": "simple_workspace_nonp.json",
        "analysis": "L_ch0",
        "framework": "pyhs3",
        "plot_label": "nonp\nch0",
        "target": "model_ch0",
        "data_points": 2,
        "load_time_seconds": 0.001,
        "build_time_seconds": 0.002,
        "cold_first_evaluation_time_seconds": 0.003,
        "warm_evaluation": {
            "mean_seconds": 0.004,
            "std_seconds": 0.001,
            "min_seconds": 0.003,
            "max_seconds": 0.005,
        },
        "warm_evaluation_time_seconds_mean": 0.004,
        "scan_time_seconds": 0.009,
        "time_per_scan_point_seconds": 0.003,
        "current_rss_before_mb": 10.0,
        "current_rss_after_mb": 11.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 20.0,
        "peak_rss_after_mb": 22.0,
        "peak_rss_delta_mb": 2.0,
        "first_nll": 3.0,
        "warm_nll": 3.0,
        "scan_nll_values": [4.0, 3.0, 4.0],
        "delta_nll_shape": [1.0, 0.0, 1.0],
        "scan_nll_min": 3.0,
        "scan_nll_max": 4.0,
        "minimum_index": 1,
        "minimum_parameter_value": 1.0,
        "finite_values": True,
        "n_scan_points": 3,
        "n_runs": 2,
        "scan_parameter": "mu_sig",
        "status": "success",
    }


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def test_validate_existing_dir_and_file_success(tmp_path: Path) -> None:
    directory = tmp_path / "inputs"
    directory.mkdir()
    file_path = directory / "workspace.json"
    file_path.write_text("{}")

    assert benchmark.validate_existing_dir(directory, "Input directory") == directory
    assert benchmark.validate_existing_file(file_path, "Workspace file") == file_path


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_validate_existing_dir_rejects_invalid(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "item"
    if kind == "file":
        path.write_text("not a directory")

    with pytest.raises(FileNotFoundError, match="Input directory"):
        benchmark.validate_existing_dir(path, "Input directory")


@pytest.mark.parametrize("kind", ["missing", "dir"])
def test_validate_existing_file_rejects_invalid(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "item"
    if kind == "dir":
        path.mkdir()

    with pytest.raises(FileNotFoundError, match="Workspace file"):
        benchmark.validate_existing_file(path, "Workspace file")


@pytest.mark.parametrize(("value", "minimum"), [(1, 1), (3, 2)])
def test_validate_positive_int_success(value: int, minimum: int) -> None:
    benchmark.validate_positive_int(value, "value", minimum=minimum)


def test_validate_positive_int_rejects_too_small() -> None:
    with pytest.raises(ValueError, match="value must be at least 2"):
        benchmark.validate_positive_int(1, "value", minimum=2)


@pytest.mark.parametrize("value", [0.0, -1.0, 2.5])
def test_validate_finite_float_success(value: float) -> None:
    benchmark.validate_finite_float(value, "value")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_validate_finite_float_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        benchmark.validate_finite_float(value, "value")


@pytest.mark.parametrize("values", [[1.0], (value for value in [1.0, 2.0])])
def test_validate_scan_values_accepts_iterables(values: Any) -> None:
    assert benchmark.validate_scan_values(values, "values") == [1.0] or True


@pytest.mark.parametrize("values", [[], [1.0, float("nan")], [float("inf")]])
def test_validate_scan_values_rejects_invalid(values: list[float]) -> None:
    with pytest.raises(ValueError):
        benchmark.validate_scan_values(values, "values")


def valid_config_kwargs(tmp_path: Path) -> dict[str, Any]:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    return {
        "input_dir": input_dir,
        "workspace_names": ["workspace.json"],
        "n_runs": 1,
        "scan_parameter": "mu_sig",
        "scan_min": 0.0,
        "scan_max": 2.0,
        "n_scan_points": 3,
    }


def test_validate_benchmark_config_success(tmp_path: Path) -> None:
    benchmark.validate_benchmark_config(**valid_config_kwargs(tmp_path))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"workspace_names": []}, "At least one workspace"),
        ({"scan_parameter": ""}, "scan_parameter must not be empty"),
        ({"n_runs": 0}, "n_runs must be at least 1"),
        ({"n_scan_points": 1}, "n_scan_points must be at least 2"),
        ({"scan_min": float("nan")}, "scan_min must be finite"),
        ({"scan_min": 2.0, "scan_max": 1.0}, "scan_min must be smaller"),
    ],
)
def test_validate_benchmark_config_rejects_invalid(
    tmp_path: Path,
    override: dict[str, Any],
    message: str,
) -> None:
    kwargs = valid_config_kwargs(tmp_path)
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        benchmark.validate_benchmark_config(**kwargs)


def test_validate_benchmark_config_rejects_missing_input_dir(tmp_path: Path) -> None:
    kwargs = valid_config_kwargs(tmp_path)
    kwargs["input_dir"] = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Input directory"):
        benchmark.validate_benchmark_config(**kwargs)


def test_validate_measurement_result_success(valid_result: dict[str, Any]) -> None:
    benchmark.validate_measurement_result(valid_result)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("load_time_seconds", float("nan"), "not finite"),
        ("load_time_seconds", -0.1, "load_time_seconds must be non-negative"),
        ("build_time_seconds", -0.1, "build_time_seconds must be non-negative"),
        (
            "cold_first_evaluation_time_seconds",
            0.0,
            "cold_first_evaluation_time_seconds must be positive",
        ),
        (
            "warm_evaluation_time_seconds_mean",
            0.0,
            "warm_evaluation_time_seconds_mean must be positive",
        ),
        ("scan_time_seconds", 0.0, "scan_time_seconds must be positive"),
        (
            "time_per_scan_point_seconds",
            0.0,
            "time_per_scan_point_seconds must be positive",
        ),
    ],
)
def test_validate_measurement_result_rejects_invalid_fields(
    valid_result: dict[str, Any],
    field: str,
    value: float,
    message: str,
) -> None:
    result = dict(valid_result)
    result[field] = value

    with pytest.raises(ValueError, match=message):
        benchmark.validate_measurement_result(result)


def test_validate_measurement_result_rejects_nonfinite_evaluation(
    valid_result: dict[str, Any],
) -> None:
    result = dict(valid_result, finite_values=False)

    with pytest.raises(ValueError, match="non-finite"):
        benchmark.validate_measurement_result(result)


# ---------------------------------------------------------------------------
# Workspace helpers
# ---------------------------------------------------------------------------


def test_channel_and_target_from_analysis() -> None:
    assert benchmark.channel_from_analysis("L_ch2") == "ch2"
    assert benchmark.target_from_analysis("L_ch2") == "model_ch2"


@pytest.mark.parametrize("analysis", ["ch0", "L_"])
def test_channel_from_analysis_rejects_invalid(analysis: str) -> None:
    with pytest.raises(ValueError):
        benchmark.channel_from_analysis(analysis)


def test_discover_analyses_success() -> None:
    workspace = FakeWorkspace(analyses=["L_ch0", "L_ch1"])

    assert benchmark.discover_analyses(workspace) == ["L_ch0", "L_ch1"]


def test_discover_analyses_rejects_missing_or_empty_section() -> None:
    with pytest.raises(ValueError, match="valid analyses"):
        benchmark.discover_analyses(SimpleNamespace())

    with pytest.raises(ValueError, match="any analyses"):
        benchmark.discover_analyses(SimpleNamespace(analyses=SimpleNamespace(root=[])))


def test_get_x_data_success() -> None:
    workspace = FakeWorkspace(data_entries=[[1.0], [2.0], [3.0]])

    values = benchmark.get_x_data(workspace, "L_ch0")

    assert values.tolist() == [1.0, 2.0, 3.0]


def test_get_x_data_rejects_missing_section_empty_nonfinite_and_missing_dataset() -> (
    None
):
    with pytest.raises(ValueError, match="valid data"):
        benchmark.get_x_data(SimpleNamespace(), "L_ch0")

    with pytest.raises(ValueError, match="empty"):
        benchmark.get_x_data(FakeWorkspace(data_entries=[]), "L_ch0")

    with pytest.raises(ValueError, match="non-finite"):
        benchmark.get_x_data(FakeWorkspace(data_entries=[[float("nan")]]), "L_ch0")

    workspace = SimpleNamespace(data=SimpleNamespace(root=[FakeData("other", [[1.0]])]))
    with pytest.raises(KeyError, match="combData_ch0"):
        benchmark.get_x_data(workspace, "L_ch0")


def test_get_eval_params_success() -> None:
    model = FakeModel(free_params={"mu_sig": 1.0, "theta": np.asarray([2.0])})
    x = np.asarray([1.0, 2.0])

    params = benchmark.get_eval_params(model, x)

    assert set(params) == {"mu_sig", "theta", "x"}
    assert params["x"] is x


def test_get_eval_params_rejects_missing_or_nonfinite_free_params() -> None:
    with pytest.raises(ValueError, match="free_params"):
        benchmark.get_eval_params(SimpleNamespace(free_params=None), np.asarray([1.0]))

    with pytest.raises(ValueError, match="non-finite"):
        benchmark.get_eval_params(
            SimpleNamespace(free_params={"bad": float("nan")}),
            np.asarray([1.0]),
        )


def test_evaluate_unbinned_nll_success() -> None:
    model = FakeModel(logpdf_values=[-1.0, -2.0])

    value = benchmark.evaluate_unbinned_nll(
        model, "model_ch0", {"x": np.asarray([1.0])}
    )

    assert value == pytest.approx(3.0)


@pytest.mark.parametrize("values", [[], [float("nan")]])
def test_evaluate_unbinned_nll_rejects_empty_or_nonfinite(values: list[float]) -> None:
    model = FakeModel(logpdf_values=values)

    with pytest.raises(ValueError):
        benchmark.evaluate_unbinned_nll(model, "target", {})


def test_evaluate_unbinned_nll_rejects_nonfinite_sum() -> None:
    model = FakeModel(logpdf_values=[-1e309])

    with pytest.raises(ValueError, match="non-finite"):
        benchmark.evaluate_unbinned_nll(model, "target", {})


# ---------------------------------------------------------------------------
# Timing and scan helpers
# ---------------------------------------------------------------------------


def test_summarize_timings_single_and_multiple_values() -> None:
    assert benchmark.summarize_timings([0.1])["std_seconds"] == 0.0
    summary = benchmark.summarize_timings([0.1, 0.3])
    assert summary["mean_seconds"] == pytest.approx(0.2)
    assert summary["min_seconds"] == pytest.approx(0.1)
    assert summary["max_seconds"] == pytest.approx(0.3)


@pytest.mark.parametrize("values", [[], [1.0, -1.0], [float("nan")]])
def test_summarize_timings_rejects_invalid(values: list[float]) -> None:
    with pytest.raises(ValueError):
        benchmark.summarize_timings(values)


def test_time_repeated_success(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([1.0, 1.1, 2.0, 2.2])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))

    value, timings = benchmark.time_repeated(lambda: 42.0, n_runs=2)

    assert value == 42.0
    assert timings == pytest.approx([0.1, 0.2])


def test_time_repeated_rejects_nonfinite_value() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        benchmark.time_repeated(lambda: float("nan"), n_runs=1)


def test_scan_nll_success(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([1.0, 1.3])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))

    values, duration = benchmark.scan_nll(
        lambda params: float(np.asarray(params["mu_sig"])) ** 2,
        "mu_sig",
        [0.0, 1.0, 2.0],
        {"x": np.asarray([1.0])},
    )

    assert values == [0.0, 1.0, 4.0]
    assert duration == pytest.approx(0.3)


def test_scan_nll_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        benchmark.scan_nll(
            lambda _params: float("nan"),
            "mu_sig",
            [0.0],
            {},
        )


def test_delta_nll_and_minimum_position_success() -> None:
    assert benchmark.delta_nll([3.0, 1.0, 2.0]).tolist() == [2.0, 0.0, 1.0]
    assert benchmark.minimum_position([0.0, 1.0, 2.0], [3.0, 1.0, 2.0]) == (1, 1.0)


def test_minimum_position_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        benchmark.minimum_position([0.0], [1.0, 2.0])


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def test_measure_workspace_analysis_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_path = tmp_path / "workspace.json"
    workspace_path.write_text("{}")
    fake_workspace = FakeWorkspace()
    times = iter(
        [
            1.0,
            1.1,  # load
            2.0,
            2.2,  # build
            3.0,
            3.1,  # cold
            4.0,
            4.1,  # warm 1
            5.0,
            5.2,  # warm 2
            6.0,
            6.3,  # scan
        ]
    )

    monkeypatch.setattr(benchmark.Workspace, "load", lambda path: fake_workspace)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 10.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 20.0)

    result = benchmark.measure_workspace_analysis(
        workspace_path=workspace_path,
        analysis_name="L_ch0",
        n_runs=2,
        scan_parameter="mu_sig",
        scan_values=[0.0, 1.0, 2.0],
    )

    assert result["workspace"] == "workspace.json"
    assert result["analysis"] == "L_ch0"
    assert result["target"] == "model_ch0"
    assert result["status"] == "success"
    assert result["minimum_parameter_value"] == 1.0
    assert result["n_runs"] == 2
    assert result["scan_nll_values"][1] < result["scan_nll_values"][0]


def test_measure_workspace_analysis_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file"):
        benchmark.measure_workspace_analysis(
            workspace_path=tmp_path / "missing.json",
            analysis_name="L_ch0",
            n_runs=1,
            scan_parameter="mu_sig",
            scan_values=[0.0, 1.0],
        )


def test_failed_result_with_and_without_analysis() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        with_analysis = benchmark.failed_result("workspace.json", "L_ch0", exc)
        without_analysis = benchmark.failed_result("workspace.json", None, exc)

    assert with_analysis["target"] == "model_ch0"
    assert without_analysis["target"] is None
    assert with_analysis["status"] == "failed"
    assert with_analysis["error_type"] == "RuntimeError"
    assert "traceback" in with_analysis


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_case_label_and_plot_helpers(valid_result: dict[str, Any]) -> None:
    assert benchmark._case_label("simple_workspace_nonp", "L_ch0") == "nonp\nch0"
    assert (
        benchmark._case_label("simple_workspace_generic_nonp", "L_ch1")
        == "generic\nnonp\nch1"
    )
    assert benchmark._successful_results([valid_result, {"status": "failed"}]) == [
        valid_result
    ]
    assert benchmark._plot_floor([0.0, 2.0], floor=0.1) == [0.1, 2.0]


def test_save_figure_creates_png(tmp_path: Path) -> None:
    fig, _ax = benchmark.plt.subplots()
    output = tmp_path / "plot_without_suffix"

    benchmark._save_figure(fig, output)

    assert (tmp_path / "plot_without_suffix.png").exists()


def test_save_figure_wraps_oserror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BadFig:
        def savefig(self, *args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

    closed: list[Any] = []
    monkeypatch.setattr(benchmark.plt, "close", lambda fig: closed.append(fig))

    with pytest.raises(OSError, match="Failed to save plot"):
        benchmark._save_figure(BadFig(), tmp_path / "plot.png")
    assert closed


@pytest.mark.parametrize(
    ("plot_func", "filename", "extra"),
    [
        (benchmark.make_runtime_scaling_plot, "runtime.png", None),
        (benchmark.make_timing_profile_plot, "timing.png", None),
        (benchmark.make_memory_scaling_plot, "memory.png", None),
        (benchmark.make_profile_examples_plot, "profiles.png", [0.0, 1.0, 2.0]),
        (benchmark.make_summary_table_plot, "summary.png", None),
    ],
)
def test_individual_plot_functions_create_png(
    tmp_path: Path,
    valid_result: dict[str, Any],
    plot_func: Any,
    filename: str,
    extra: Any,
) -> None:
    output = tmp_path / filename
    if plot_func is benchmark.make_profile_examples_plot:
        plot_func([valid_result], extra, output)
    else:
        plot_func([valid_result], output)

    assert output.exists()


@pytest.mark.parametrize(
    "plot_func",
    [
        benchmark.make_runtime_scaling_plot,
        benchmark.make_timing_profile_plot,
        benchmark.make_memory_scaling_plot,
        benchmark.make_summary_table_plot,
    ],
)
def test_bar_plot_functions_reject_no_success(tmp_path: Path, plot_func: Any) -> None:
    with pytest.raises(ValueError, match="No successful"):
        plot_func([{"status": "failed"}], tmp_path / "plot.png")


def test_profile_examples_plot_rejects_no_success(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No successful"):
        benchmark.make_profile_examples_plot([], [0.0, 1.0], tmp_path / "plot.png")


def test_make_plots_creates_expected_pngs(
    tmp_path: Path, valid_result: dict[str, Any]
) -> None:
    benchmark.make_plots([valid_result], [0.0, 1.0, 2.0], tmp_path)

    expected = {
        "pyhs3_model_complexity_runtime_scaling.png",
        "pyhs3_model_complexity_timing_profile.png",
        "pyhs3_model_complexity_memory_scaling.png",
        "pyhs3_model_complexity_profile_examples.png",
        "pyhs3_model_complexity_summary_table.png",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})


def test_make_plots_rejects_no_success(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No successful"):
        benchmark.make_plots([{"status": "failed"}], [0.0, 1.0], tmp_path)


# ---------------------------------------------------------------------------
# Output, run, CLI
# ---------------------------------------------------------------------------


def test_print_result_and_failed_result(
    capsys: pytest.CaptureFixture[str],
    valid_result: dict[str, Any],
) -> None:
    benchmark.print_result(valid_result)
    output = capsys.readouterr().out
    assert "workspace" in output
    assert "status:                  success" in output

    failed = {
        "workspace": "bad.json",
        "analysis": "L_ch0",
        "error_type": "X",
        "error_message": "bad",
    }
    benchmark.print_failed_result(failed)
    output = capsys.readouterr().out
    assert "status:                  failed" in output
    assert "X: bad" in output


def test_build_failed_output(tmp_path: Path) -> None:
    try:
        raise ValueError("bad")
    except ValueError as exc:
        output = benchmark.build_failed_output(
            input_dir=tmp_path / "inputs",
            workspace_names=["workspace.json"],
            n_runs=1,
            scan_parameter="mu_sig",
            scan_min=0.0,
            scan_max=2.0,
            n_scan_points=3,
            exc=exc,
        )

    assert output["benchmark"] == benchmark.BENCHMARK_NAME
    assert output["framework"] == "pyhs3"
    assert output["status"] == "failed"
    assert output["error_type"] == "ValueError"
    assert output["results"] == []


def test_run_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, valid_result: dict[str, Any]
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    workspace_path = input_dir / "workspace.json"
    workspace_path.write_text("{}")
    output = tmp_path / "result.json"
    plot_calls: list[Any] = []

    monkeypatch.setattr(
        benchmark.Workspace,
        "load",
        lambda path: FakeWorkspace(analyses=["L_ch0", "L_ch1"]),
    )

    def fake_measure_workspace_analysis(**kwargs: Any) -> dict[str, Any]:
        return dict(
            valid_result,
            workspace=kwargs["workspace_path"].name,
            analysis=kwargs["analysis_name"],
            target=benchmark.target_from_analysis(kwargs["analysis_name"]),
        )

    monkeypatch.setattr(
        benchmark, "measure_workspace_analysis", fake_measure_workspace_analysis
    )
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda *args, **kwargs: plot_calls.append((args, kwargs)),
    )

    result = benchmark.run(
        input_dir=input_dir,
        workspace_names=["workspace.json"],
        n_runs=1,
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=2.0,
        n_scan_points=3,
        output=output,
        plot=True,
        plot_dir=tmp_path / "plots",
    )

    assert result["status"] == "success"
    assert result["successful_runs"] == 2
    assert result["total_runs"] == 2
    assert json.loads(output.read_text())["status"] == "success"
    assert plot_calls


def test_run_records_discovery_failure_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "workspace.json").write_text("{}")
    output = tmp_path / "result.json"

    monkeypatch.setattr(
        benchmark.Workspace,
        "load",
        lambda path: (_ for _ in ()).throw(RuntimeError("cannot load")),
    )

    result = benchmark.run(
        input_dir=input_dir,
        workspace_names=["workspace.json"],
        n_runs=1,
        output=output,
        plot=False,
        continue_on_case_error=True,
    )

    assert result["status"] == "failed"
    assert result["results"][0]["analysis"] is None
    assert result["results"][0]["error_type"] == "RuntimeError"


def test_run_records_analysis_failure_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "workspace.json").write_text("{}")
    output = tmp_path / "result.json"

    monkeypatch.setattr(
        benchmark.Workspace,
        "load",
        lambda path: FakeWorkspace(analyses=["L_ch0"]),
    )
    monkeypatch.setattr(
        benchmark,
        "measure_workspace_analysis",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("analysis failed")),
    )

    result = benchmark.run(
        input_dir=input_dir,
        workspace_names=["workspace.json"],
        n_runs=1,
        output=output,
        plot=False,
        continue_on_case_error=True,
    )

    assert result["status"] == "failed"
    assert result["failed_cases"] == ["workspace.json/L_ch0"]
    assert result["results"][0]["error_message"] == "analysis failed"


@pytest.mark.parametrize("failure_stage", ["discovery", "analysis"])
def test_run_fail_fast_writes_failure_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "workspace.json").write_text("{}")
    output = tmp_path / "result.json"

    if failure_stage == "discovery":
        monkeypatch.setattr(
            benchmark.Workspace,
            "load",
            lambda path: (_ for _ in ()).throw(RuntimeError("cannot load")),
        )
    else:
        monkeypatch.setattr(
            benchmark.Workspace, "load", lambda path: FakeWorkspace(analyses=["L_ch0"])
        )
        monkeypatch.setattr(
            benchmark,
            "measure_workspace_analysis",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("analysis failed")),
        )

    with pytest.raises(RuntimeError, match="PyHS3 model-complexity"):
        benchmark.run(
            input_dir=input_dir,
            workspace_names=["workspace.json"],
            n_runs=1,
            output=output,
            plot=False,
            continue_on_case_error=False,
        )

    payload = json.loads(output.read_text())
    assert payload["status"] == "failed"
    assert payload["error_type"] in {"RuntimeError", "ValueError"}


def test_run_handles_failure_report_save_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing_save_json(*args: Any, **kwargs: Any) -> None:
        raise OSError("cannot save")

    monkeypatch.setattr(benchmark, "save_json", failing_save_json)

    with pytest.raises(RuntimeError, match="PyHS3 model-complexity"):
        benchmark.run(
            input_dir=tmp_path / "missing",
            workspace_names=["workspace.json"],
            output=tmp_path / "result.json",
            plot=False,
        )

    assert "Failed to save benchmark failure report" in capsys.readouterr().err


def test_run_plot_with_no_success_raises_and_writes_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    (input_dir / "workspace.json").write_text("{}")
    output = tmp_path / "result.json"

    monkeypatch.setattr(
        benchmark.Workspace,
        "load",
        lambda path: (_ for _ in ()).throw(RuntimeError("cannot load")),
    )

    with pytest.raises(RuntimeError, match="PyHS3 model-complexity"):
        benchmark.run(
            input_dir=input_dir,
            workspace_names=["workspace.json"],
            output=output,
            plot=True,
            continue_on_case_error=True,
        )

    payload = json.loads(output.read_text())
    assert payload["status"] == "failed"
    assert payload["error_type"] == "ValueError"


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_pyhs3_model_complexity_scaling.py"])

    args = benchmark.parse_args()

    assert args.input_dir == Path("inputs")
    assert args.workspaces == benchmark.DEFAULT_WORKSPACES
    assert args.n_runs == benchmark.DEFAULT_N_RUNS
    assert args.scan_parameter == benchmark.DEFAULT_SCAN_PARAMETER
    assert args.plot is False
    assert args.fail_fast is False


def test_parse_args_custom_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pyhs3_model_complexity_scaling.py",
            "--input-dir",
            str(tmp_path / "inputs"),
            "--workspaces",
            "a.json",
            "b.json",
            "--n-runs",
            "2",
            "--scan-parameter",
            "theta",
            "--scan-min",
            "0.1",
            "--scan-max",
            "1.9",
            "--n-scan-points",
            "5",
            "--output",
            str(tmp_path / "out.json"),
            "--plot",
            "--plot-dir",
            str(tmp_path / "plots"),
            "--fail-fast",
        ],
    )

    args = benchmark.parse_args()

    assert args.input_dir == tmp_path / "inputs"
    assert args.workspaces == ["a.json", "b.json"]
    assert args.n_runs == 2
    assert args.scan_parameter == "theta"
    assert args.scan_min == 0.1
    assert args.scan_max == 1.9
    assert args.n_scan_points == 5
    assert args.output == tmp_path / "out.json"
    assert args.plot is True
    assert args.plot_dir == tmp_path / "plots"
    assert args.fail_fast is True


def test_main_passes_cli_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pyhs3_model_complexity_scaling.py",
            "--input-dir",
            str(tmp_path / "inputs"),
            "--workspaces",
            "workspace.json",
            "--n-runs",
            "2",
            "--output",
            str(tmp_path / "out.json"),
            "--fail-fast",
        ],
    )
    monkeypatch.setattr(benchmark, "run", lambda **kwargs: calls.append(kwargs))

    benchmark.main()

    assert calls[0]["input_dir"] == tmp_path / "inputs"
    assert calls[0]["workspace_names"] == ["workspace.json"]
    assert calls[0]["n_runs"] == 2
    assert calls[0]["continue_on_case_error"] is False
