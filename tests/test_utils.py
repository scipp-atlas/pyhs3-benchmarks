from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import pytest

from src import utils


def test_get_current_rss_mb_returns_positive_number() -> None:
    value = utils.get_current_rss_mb()

    assert isinstance(value, float)
    assert value > 0


def test_get_peak_rss_mb_returns_positive_number() -> None:
    value = utils.get_peak_rss_mb()

    assert isinstance(value, float)
    assert value > 0


def test_run_repeated_timing_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    times = iter([1.0, 1.1, 2.0, 2.2, 3.0, 3.3])

    def func() -> str:
        calls.append("call")
        return f"result-{len(calls)}"

    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(times))

    result, timings = utils.run_repeated_timing(func, n_runs=3, warmup_runs=1)

    assert result == "result-4"
    assert len(calls) == 4
    assert timings == pytest.approx([0.1, 0.2, 0.3])


@pytest.mark.parametrize(
    ("n_runs", "warmup_runs", "message"),
    [
        (0, 0, "n_runs must be at least 1"),
        (1, -1, "warmup_runs must be non-negative"),
    ],
)
def test_run_repeated_timing_rejects_invalid_counts(
    n_runs: int,
    warmup_runs: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        utils.run_repeated_timing(lambda: None, n_runs=n_runs, warmup_runs=warmup_runs)


def test_run_repeated_timing_wraps_function_error() -> None:
    def func() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="Repeated timing failed") as exc_info:
        utils.run_repeated_timing(func, n_runs=1, warmup_runs=0)

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("timings", "expected"),
    [
        ([1.0], {"mean": 1.0, "median": 1.0, "std": 0.0}),
        ([1.0, 2.0, 3.0], {"mean": 2.0, "median": 2.0}),
    ],
)
def test_summarize_timings_success(
    timings: list[float], expected: dict[str, float]
) -> None:
    summary = utils.summarize_timings(timings)

    assert summary["wall_time_seconds_mean"] == pytest.approx(expected["mean"])
    assert summary["wall_time_seconds_median"] == pytest.approx(expected["median"])
    if "std" in expected:
        assert summary["wall_time_seconds_std"] == expected["std"]
    else:
        assert summary["wall_time_seconds_std"] > 0


def test_summarize_timings_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="empty timing list"):
        utils.summarize_timings([])


@pytest.mark.parametrize("timings", [[0.0], [-1.0], [float("nan")], [float("inf")]])
def test_summarize_timings_rejects_non_positive_or_non_finite(
    timings: list[float],
) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        utils.summarize_timings(timings)


def test_save_json_creates_parent_directories_and_sorted_json(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "result.json"

    utils.save_json({"b": 2, "a": 1}, output_path)

    assert output_path.exists()
    text = output_path.read_text()
    assert text.index('"a"') < text.index('"b"')
    assert json.loads(text) == {"a": 1, "b": 2}


def test_save_json_rejects_non_serializable_data(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"

    with pytest.raises(TypeError, match="not JSON serializable"):
        utils.save_json({"bad": {1, 2, 3}}, output_path)


def test_save_json_wraps_os_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "result.json"

    def fail_open(*args: Any, **kwargs: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", fail_open)

    with pytest.raises(OSError, match="Failed to write benchmark JSON output"):
        utils.save_json({"ok": True}, output_path)


@pytest.mark.parametrize(
    ("results", "metric_key", "expected"),
    [
        ([{"status": "success", "x": 1.0}], "x", True),
        ([{"status": "success", "x": 0.0}], "x", False),
        ([{"status": "failed", "x": 1.0}], "x", False),
        ([{"status": "success", "x": None}], "x", False),
        ([{"status": "success", "x": "abc"}], "x", False),
        ([{"status": "success", "x": float("nan")}], "x", False),
    ],
)
def test_should_plot_metric(
    results: list[dict[str, Any]], metric_key: str, expected: bool
) -> None:
    assert utils.should_plot_metric(results, metric_key) is expected


def test_apply_style_updates_rcparams() -> None:
    utils._apply_style()

    assert plt.rcParams["figure.facecolor"] == "white"
    assert plt.rcParams["axes.linewidth"] == 1.5


def test_result_label_uses_explicit_plot_label() -> None:
    assert utils._result_label({"plot_label": "custom"}) == "custom"


def test_result_label_uses_workspace_only() -> None:
    label = utils._result_label(
        {
            "workspace": "simple_workspace.json",
            "n_evaluations": 10,
        }
    )

    assert label == "simple\nworkspace"
    assert "10" not in label


def test_grouped_result_parts_uses_workspace_and_evaluations() -> None:
    workspace, n_evaluations = utils._grouped_result_parts(
        {
            "workspace": "simple_workspace.json",
            "n_evaluations": 10,
        }
    )

    assert workspace
    assert n_evaluations == 10


def test_scaled_metric_scales_wall_time_and_errors() -> None:
    results = [
        {
            "workspace": "a.json",
            "wall_time_seconds_mean": 0.1,
            "wall_time_seconds_std": 0.01,
        }
    ]

    values, errors, label = utils._scaled_metric(
        results,
        "wall_time_seconds_mean",
        "Mean wall time [s]",
    )

    assert values == pytest.approx([100.0])
    assert errors == pytest.approx([10.0])
    assert label == "Mean wall time [ms]"


def test_scaled_metric_without_errors() -> None:
    values, errors, label = utils._scaled_metric(
        [{"workspace": "a.json", "metric": 2.5}],
        "metric",
        "Metric",
    )

    assert values == [2.5]
    assert errors is None
    assert label == "Metric"


def test_scaled_metric_uses_missing_std_as_zero() -> None:
    values, errors, _label = utils._scaled_metric(
        [
            {"workspace": "a.json", "wall_time_seconds_mean": 0.1},
            {
                "workspace": "b.json",
                "wall_time_seconds_mean": 0.2,
                "wall_time_seconds_std": 0.02,
            },
        ],
        "wall_time_seconds_mean",
        "Mean wall time [s]",
    )

    assert values == pytest.approx([100.0, 200.0])
    assert errors == pytest.approx([0.0, 20.0])


def test_scaled_metric_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="empty result list"):
        utils._scaled_metric([], "metric", "Metric")


def test_scaled_metric_rejects_missing_metric() -> None:
    with pytest.raises(KeyError, match="Metric 'metric' is missing"):
        utils._scaled_metric([{"workspace": "a.json"}], "metric", "Metric")


@pytest.mark.parametrize("value", ["abc", object()])
def test_scaled_metric_rejects_non_numeric_values(value: Any) -> None:
    with pytest.raises(ValueError, match="contains non-numeric"):
        utils._scaled_metric(
            [{"workspace": "a.json", "metric": value}], "metric", "Metric"
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_scaled_metric_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="contains non-finite"):
        utils._scaled_metric(
            [{"workspace": "a.json", "metric": value}], "metric", "Metric"
        )


@pytest.mark.parametrize(
    ("value", "metric_label", "expected"),
    [
        (1.23456, "Time [ms]", "1.235"),
        (1.23456, "Memory [MB]", "1.235"),
        (150.9, "Count", "151"),
        (1.23456, "Value", "1.235"),
        (float("nan"), "Value", "nan"),
    ],
)
def test_format_value(value: float, metric_label: str, expected: str) -> None:
    assert utils._format_value(value, metric_label) == expected


def valid_plot_results() -> list[dict[str, Any]]:
    return [
        {
            "workspace": "simple_workspace.json",
            "status": "success",
            "wall_time_seconds_mean": 0.01,
            "wall_time_seconds_std": 0.001,
        },
        {
            "workspace": "simple_workspace_nonp.json",
            "status": "success",
            "wall_time_seconds_mean": 0.02,
            "wall_time_seconds_std": 0.002,
        },
    ]


def test_make_bar_plot_creates_png(tmp_path: Path) -> None:
    output_path = tmp_path / "plots" / "bar.png"

    utils.make_bar_plot(
        results=valid_plot_results(),
        output_path=output_path,
        title="Wall time",
        metric_key="wall_time_seconds_mean",
        metric_label="Mean wall time [s]",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_make_bar_plot_filters_failed_results(tmp_path: Path) -> None:
    output_path = tmp_path / "bar.png"
    results = valid_plot_results() + [
        {
            "workspace": "failed.json",
            "status": "failed",
            "wall_time_seconds_mean": 1000.0,
        }
    ]

    utils.make_bar_plot(
        results=results,
        output_path=output_path,
        title="Wall time",
        metric_key="wall_time_seconds_mean",
        metric_label="Mean wall time [s]",
    )

    assert output_path.exists()


def test_make_bar_plot_rejects_no_successful_results(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without successful benchmark results"):
        utils.make_bar_plot(
            results=[{"status": "failed", "metric": 1.0}],
            output_path=tmp_path / "bar.png",
            title="Metric",
            metric_key="metric",
            metric_label="Metric",
        )


def test_make_bar_plot_handles_negative_values(tmp_path: Path) -> None:
    output_path = tmp_path / "negative.png"

    utils.make_bar_plot(
        results=[{"workspace": "a.json", "status": "success", "metric": -1.0}],
        output_path=output_path,
        title="Negative",
        metric_key="metric",
        metric_label="Metric",
    )

    assert output_path.exists()


def test_make_bar_plot_wraps_save_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_savefig(self: Any, *args: Any, **kwargs: Any) -> None:
        raise OSError("cannot write")

    monkeypatch.setattr(plt.Figure, "savefig", fail_savefig)

    with pytest.raises(OSError, match="Failed to save plot"):
        utils.make_bar_plot(
            results=valid_plot_results(),
            output_path=tmp_path / "bar.png",
            title="Wall time",
            metric_key="wall_time_seconds_mean",
            metric_label="Mean wall time [s]",
        )


def test_grouped_result_parts() -> None:
    workspace, n_evaluations = utils._grouped_result_parts(
        {"workspace": "simple_workspace.json", "n_evaluations": 10}
    )

    assert workspace
    assert n_evaluations == 10


def grouped_results() -> list[dict[str, Any]]:
    return [
        {
            "workspace": "simple_workspace.json",
            "status": "success",
            "n_evaluations": 1,
            "average_runtime_seconds_per_evaluation": 0.001,
        },
        {
            "workspace": "simple_workspace.json",
            "status": "success",
            "n_evaluations": 10,
            "average_runtime_seconds_per_evaluation": 0.002,
        },
        {
            "workspace": "simple_workspace_nonp.json",
            "status": "success",
            "n_evaluations": 1,
            "average_runtime_seconds_per_evaluation": 0.003,
        },
    ]


def test_make_grouped_bar_plot_creates_png(tmp_path: Path) -> None:
    output_path = tmp_path / "grouped.png"

    utils.make_grouped_bar_plot(
        results=grouped_results(),
        output_path=output_path,
        title="Grouped",
        metric_key="average_runtime_seconds_per_evaluation",
        metric_label="Average runtime [s]",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_make_grouped_bar_plot_rejects_no_successful_results(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="without successful benchmark results"):
        utils.make_grouped_bar_plot(
            results=[{"status": "failed", "metric": 1.0}],
            output_path=tmp_path / "grouped.png",
            title="Grouped",
            metric_key="metric",
            metric_label="Metric",
        )


def test_make_grouped_bar_plot_requires_n_evaluations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires workspace and n_evaluations"):
        utils.make_grouped_bar_plot(
            results=[{"workspace": "a.json", "status": "success", "metric": 1.0}],
            output_path=tmp_path / "grouped.png",
            title="Grouped",
            metric_key="metric",
            metric_label="Metric",
        )


def test_make_grouped_bar_plot_wraps_save_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_savefig(self: Any, *args: Any, **kwargs: Any) -> None:
        raise OSError("cannot write")

    monkeypatch.setattr(plt.Figure, "savefig", fail_savefig)

    with pytest.raises(OSError, match="Failed to save plot"):
        utils.make_grouped_bar_plot(
            results=grouped_results(),
            output_path=tmp_path / "grouped.png",
            title="Grouped",
            metric_key="average_runtime_seconds_per_evaluation",
            metric_label="Average runtime [s]",
        )


def test_make_line_plot_by_evaluations_creates_png(tmp_path: Path) -> None:
    output_path = tmp_path / "line.png"

    utils.make_line_plot_by_evaluations(
        results=grouped_results(),
        output_path=output_path,
        title="Line",
        metric_key="average_runtime_seconds_per_evaluation",
        metric_label="Average runtime [s]",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_make_line_plot_by_evaluations_without_log_x(tmp_path: Path) -> None:
    output_path = tmp_path / "line_linear.png"

    utils.make_line_plot_by_evaluations(
        results=grouped_results(),
        output_path=output_path,
        title="Line",
        metric_key="average_runtime_seconds_per_evaluation",
        metric_label="Average runtime [s]",
        log_x=False,
    )

    assert output_path.exists()


def test_make_line_plot_by_evaluations_rejects_no_successful_results(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="without successful benchmark results"):
        utils.make_line_plot_by_evaluations(
            results=[{"status": "failed", "metric": 1.0}],
            output_path=tmp_path / "line.png",
            title="Line",
            metric_key="metric",
            metric_label="Metric",
        )


def test_make_line_plot_by_evaluations_requires_n_evaluations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires at least one n_evaluations"):
        utils.make_line_plot_by_evaluations(
            results=[{"workspace": "a.json", "status": "success", "metric": 1.0}],
            output_path=tmp_path / "line.png",
            title="Line",
            metric_key="metric",
            metric_label="Metric",
        )


def test_make_line_plot_by_evaluations_wraps_save_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_savefig(self: Any, *args: Any, **kwargs: Any) -> None:
        raise OSError("cannot write")

    monkeypatch.setattr(plt.Figure, "savefig", fail_savefig)

    with pytest.raises(OSError, match="Failed to save plot"):
        utils.make_line_plot_by_evaluations(
            results=grouped_results(),
            output_path=tmp_path / "line.png",
            title="Line",
            metric_key="average_runtime_seconds_per_evaluation",
            metric_label="Average runtime [s]",
        )


def test_load_workspace_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_path = tmp_path / "workspace.json"
    workspace_path.write_text("{}")
    loaded = object()

    monkeypatch.setattr(utils.Workspace, "load", lambda path: loaded)

    assert utils.load_workspace(workspace_path) is loaded


def test_load_workspace_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        utils.load_workspace(tmp_path / "missing.json")


def test_load_workspace_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not a file"):
        utils.load_workspace(tmp_path)


def test_load_workspace_wraps_load_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_path = tmp_path / "workspace.json"
    workspace_path.write_text("{}")

    def fail_load(path: Path) -> Any:
        raise RuntimeError("bad workspace")

    monkeypatch.setattr(utils.Workspace, "load", fail_load)

    with pytest.raises(RuntimeError, match="Failed to load workspace"):
        utils.load_workspace(workspace_path)


def test_create_model_success() -> None:
    model = object()
    workspace = SimpleNamespace(model=lambda target, progress, mode: model)

    assert utils.create_model(workspace, target="analysis", mode="FAST_RUN") is model


@pytest.mark.parametrize(
    ("workspace", "target", "mode", "message"),
    [
        (None, "analysis", "FAST_RUN", "workspace must not be None"),
        (object(), "", "FAST_RUN", "target must be a non-empty string"),
        (object(), "analysis", "", "mode must be a non-empty string"),
    ],
)
def test_create_model_rejects_invalid_inputs(
    workspace: Any,
    target: str,
    mode: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        utils.create_model(workspace, target=target, mode=mode)


def test_create_model_wraps_model_error() -> None:
    def fail_model(target: str, progress: bool, mode: str) -> Any:
        raise RuntimeError("bad target")

    workspace = SimpleNamespace(model=fail_model)

    with pytest.raises(RuntimeError, match="Failed to create model"):
        utils.create_model(workspace, target="analysis", mode="FAST_RUN")


def test_build_log_prob_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = SimpleNamespace(log_prob="log_prob")

    monkeypatch.setattr(utils, "load_workspace", lambda workspace_path: "workspace")
    monkeypatch.setattr(
        utils,
        "create_model",
        lambda workspace, target, mode: model,
    )

    result_model, log_prob = utils.build_log_prob(
        tmp_path / "workspace.json",
        target="analysis",
        mode="FAST_RUN",
    )

    assert result_model is model
    assert log_prob == "log_prob"


def test_build_log_prob_wraps_property_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BadModel:
        @property
        def log_prob(self) -> Any:
            raise RuntimeError("missing likelihood")

    monkeypatch.setattr(utils, "load_workspace", lambda workspace_path: "workspace")
    monkeypatch.setattr(
        utils, "create_model", lambda workspace, target, mode: BadModel()
    )

    with pytest.raises(RuntimeError, match="Failed to build log_prob"):
        utils.build_log_prob(
            tmp_path / "workspace.json", target="analysis", mode="FAST_RUN"
        )


def test_build_log_prob_rejects_none_log_prob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(utils, "load_workspace", lambda workspace_path: "workspace")
    monkeypatch.setattr(
        utils,
        "create_model",
        lambda workspace, target, mode: SimpleNamespace(log_prob=None),
    )

    with pytest.raises(ValueError, match="log_prob construction returned None"):
        utils.build_log_prob(
            tmp_path / "workspace.json", target="analysis", mode="FAST_RUN"
        )


def test_compile_log_prob_success(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = object()
    monkeypatch.setattr(utils, "jaxify", lambda log_prob: compiled)

    assert utils.compile_log_prob("log_prob") is compiled


def test_compile_log_prob_rejects_none() -> None:
    with pytest.raises(ValueError, match="log_prob must not be None"):
        utils.compile_log_prob(None)


def test_compile_log_prob_wraps_jaxify_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_jaxify(log_prob: Any) -> Any:
        raise RuntimeError("compile failed")

    monkeypatch.setattr(utils, "jaxify", fail_jaxify)

    with pytest.raises(RuntimeError, match="Failed to compile log_prob"):
        utils.compile_log_prob("log_prob")


def test_compile_log_prob_rejects_none_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "jaxify", lambda log_prob: None)

    with pytest.raises(ValueError, match="jaxify returned None"):
        utils.compile_log_prob("log_prob")


def test_build_validation_inputs_success() -> None:
    model = SimpleNamespace(
        data={"obs": 1.0},
        free_params={"mu": 2.0},
    )
    compiled = SimpleNamespace(input_names=["mu", "obs"])

    inputs = utils.build_validation_inputs(model, compiled)

    assert inputs == {"mu": 2.0, "obs": 1.0}


@pytest.mark.parametrize(
    ("model", "compiled", "message"),
    [
        (None, SimpleNamespace(input_names=[]), "model must not be None"),
        (
            SimpleNamespace(data={}, free_params={}),
            None,
            "compiled graph must not be None",
        ),
    ],
)
def test_build_validation_inputs_rejects_none_inputs(
    model: Any,
    compiled: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        utils.build_validation_inputs(model, compiled)


def test_build_validation_inputs_rejects_missing_compiled_inputs() -> None:
    model = SimpleNamespace(data={"obs": 1.0}, free_params={"mu": 2.0})
    compiled = SimpleNamespace(input_names=["mu", "missing"])

    with pytest.raises(KeyError, match="Compiled graph inputs are missing"):
        utils.build_validation_inputs(model, compiled)
