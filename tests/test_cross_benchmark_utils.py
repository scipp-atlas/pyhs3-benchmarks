from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.cross_benchmark_utils as utils


def test_save_json_creates_parent_and_sorted_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"

    utils.save_json({"b": 2, "a": 1}, output)

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    text = output.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"b"')


def test_current_rss_mb(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_process = SimpleNamespace(
        memory_info=lambda: SimpleNamespace(rss=10 * 1024 * 1024)
    )
    monkeypatch.setattr(utils.psutil, "Process", lambda: fake_process)

    assert utils.current_rss_mb() == pytest.approx(10.0)


def test_peak_rss_mb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        utils.resource,
        "getrusage",
        lambda _: SimpleNamespace(ru_maxrss=2048.0),
    )

    assert utils.peak_rss_mb() == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3.5, 3.5),
        ([2.0], 2.0),
        (np.asarray([[4.0]]), 4.0),
        ([7.0, 8.0], 7.0),
    ],
)
def test_finite_scalar_valid(value: Any, expected: float) -> None:
    assert utils.finite_scalar(value, label="value") == pytest.approx(expected)


def test_finite_scalar_empty() -> None:
    with pytest.raises(ValueError, match="value returned an empty array"):
        utils.finite_scalar([], label="value")


@pytest.mark.parametrize("value", [[np.nan], [np.inf], [-np.inf]])
def test_finite_scalar_nonfinite(value: Any) -> None:
    with pytest.raises(ValueError, match="value returned a non-finite value"):
        utils.finite_scalar(value, label="value")


@pytest.mark.parametrize(
    "result",
    [
        None,
        [],
        [np.asarray([1.0])],
        "not-a-tuple",
    ],
)
def test_compiled_array_requires_nonempty_tuple(result: Any) -> None:
    with pytest.raises(TypeError, match="compiled must return a non-empty tuple"):
        utils.compiled_array(result, label="compiled")


@pytest.mark.parametrize(
    "result",
    [
        ([],),
        ([np.nan],),
        ([np.inf],),
        ([1.0, np.nan],),
    ],
)
def test_compiled_array_rejects_invalid_values(result: Any) -> None:
    with pytest.raises(ValueError, match="compiled returned invalid values"):
        utils.compiled_array(result, label="compiled")


def test_compiled_array_and_scalar_valid() -> None:
    result = (np.asarray([1.5, 2.5]), "extra")

    array = utils.compiled_array(result, label="compiled")
    np.testing.assert_allclose(array, [1.5, 2.5])
    assert utils.compiled_scalar(result, label="compiled") == pytest.approx(1.5)


def test_time_once_success(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([10.0, 10.25])
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(times))

    output, elapsed = utils.time_once(lambda: "ok", label="operation")

    assert output == "ok"
    assert elapsed == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("times", "message"),
    [
        ([2.0, 1.0], "Invalid timing for operation: -1.0"),
        ([1.0, np.inf], "Invalid timing for operation: inf"),
    ],
)
def test_time_once_rejects_invalid_elapsed(
    monkeypatch: pytest.MonkeyPatch,
    times: list[float],
    message: str,
) -> None:
    iterator = iter(times)
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(iterator))

    with pytest.raises(RuntimeError, match=message):
        utils.time_once(lambda: None, label="operation")


@pytest.mark.parametrize(
    ("batch_size", "n_batches", "warmup_batches"),
    [
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, -1),
    ],
)
def test_benchmark_batches_invalid_configuration(
    batch_size: int,
    n_batches: int,
    warmup_batches: int,
) -> None:
    with pytest.raises(ValueError, match="Invalid batch benchmark configuration"):
        utils.benchmark_batches(
            lambda index: index,
            batch_size=batch_size,
            n_batches=n_batches,
            warmup_batches=warmup_batches,
        )


def test_benchmark_batches_single_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    times = iter([1.0, 1.4])
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(times))

    result = utils.benchmark_batches(
        lambda index: calls.append(index) or float(index + 1),
        batch_size=2,
        n_batches=1,
        warmup_batches=1,
    )

    assert calls == [0, 1, 0, 1]
    assert result["batch_size"] == 2
    assert result["n_batches"] == 1
    assert result["warmup_batches"] == 1
    assert result["timings_seconds_per_evaluation"] == pytest.approx([0.2])
    assert result["steady_state_seconds_median"] == pytest.approx(0.2)
    assert result["steady_state_seconds_mean"] == pytest.approx(0.2)
    assert result["steady_state_seconds_std"] == pytest.approx(0.0)
    assert result["throughput_evaluations_per_second"] == pytest.approx(5.0)
    assert result["last_output"] == pytest.approx(2.0)


def test_benchmark_batches_multiple_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([0.0, 0.2, 1.0, 1.4])
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(times))

    result = utils.benchmark_batches(
        lambda index: float(index),
        batch_size=2,
        n_batches=2,
        warmup_batches=0,
    )

    assert result["timings_seconds_per_evaluation"] == pytest.approx([0.1, 0.2])
    assert result["steady_state_seconds_median"] == pytest.approx(0.15)
    assert result["steady_state_seconds_mean"] == pytest.approx(0.15)
    assert result["steady_state_seconds_std"] > 0.0
    assert result["last_output"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "counts",
    [
        [],
        [0],
        [-1, 2],
    ],
)
def test_benchmark_scaling_invalid_counts(counts: list[int]) -> None:
    with pytest.raises(
        ValueError,
        match="n_evaluations must contain positive integers",
    ):
        utils.benchmark_scaling(
            lambda index: index,
            n_evaluations=counts,
            repeats=1,
            warmup_evaluations=0,
        )


@pytest.mark.parametrize(
    ("repeats", "warmups"),
    [
        (0, 0),
        (1, -1),
    ],
)
def test_benchmark_scaling_invalid_configuration(
    repeats: int,
    warmups: int,
) -> None:
    with pytest.raises(ValueError, match="Invalid scaling configuration"):
        utils.benchmark_scaling(
            lambda index: index,
            n_evaluations=[1],
            repeats=repeats,
            warmup_evaluations=warmups,
        )


def test_benchmark_scaling_single_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    times = iter([0.0, 0.4])
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(times))

    rows = utils.benchmark_scaling(
        lambda index: calls.append(index) or float(index + 1),
        n_evaluations=[2],
        repeats=1,
        warmup_evaluations=2,
    )

    assert calls == [0, 1, 0, 1]
    row = rows[0]
    assert row["n_evaluations"] == 2
    assert row["timing_repeats"] == 1
    assert row["total_runtime_seconds_samples"] == pytest.approx([0.4])
    assert row["time_per_value_seconds_samples"] == pytest.approx([0.2])
    assert row["total_runtime_seconds_median"] == pytest.approx(0.4)
    assert row["time_per_value_seconds_median"] == pytest.approx(0.2)
    assert row["time_per_value_seconds_mean"] == pytest.approx(0.2)
    assert row["time_per_value_seconds_std"] == pytest.approx(0.0)
    assert row["time_per_value_ns"] == pytest.approx(2e8)
    assert row["throughput_evaluations_per_second"] == pytest.approx(5.0)
    assert row["last_output"] == pytest.approx(2.0)


def test_benchmark_scaling_multiple_counts_and_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter(
        [
            0.0,
            0.2,
            1.0,
            1.4,
            2.0,
            2.6,
            3.0,
            3.8,
        ]
    )
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(times))

    rows = utils.benchmark_scaling(
        lambda index: float(index),
        n_evaluations=[1, 2],
        repeats=2,
        warmup_evaluations=0,
    )

    assert len(rows) == 2
    assert rows[0]["time_per_value_seconds_samples"] == pytest.approx([0.2, 0.4])
    assert rows[0]["time_per_value_seconds_std"] > 0.0
    assert rows[1]["time_per_value_seconds_samples"] == pytest.approx([0.3, 0.4])
    assert rows[1]["last_output"] == pytest.approx(5.0)


def test_agreement_arrays_success() -> None:
    result = utils.agreement_arrays(
        [1.0, 2.0],
        [1.0, 2.0 + 1e-10],
        rtol=1e-8,
        atol=1e-8,
    )

    assert result["n_validation_values"] == 2
    assert result["all_values_finite"] is True
    assert result["allclose_passed"] is True
    assert result["validation_status"] == "success"
    assert result["max_abs_diff"] >= 0.0
    assert result["mean_abs_diff"] >= 0.0
    assert result["max_rel_diff"] >= 0.0
    assert result["mean_rel_diff"] >= 0.0


def test_agreement_arrays_mismatch() -> None:
    result = utils.agreement_arrays(
        [1.0, 3.0],
        [1.0, 2.0],
        rtol=0.0,
        atol=0.0,
    )

    assert result["allclose_passed"] is False
    assert result["validation_status"] == "mismatch"
    assert result["max_abs_diff"] == pytest.approx(1.0)


def test_agreement_arrays_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Validation shape mismatch"):
        utils.agreement_arrays(
            [1.0],
            [1.0, 2.0],
            rtol=1e-8,
            atol=1e-8,
        )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([np.nan], [1.0]),
        ([1.0], [np.inf]),
    ],
)
def test_agreement_arrays_nonfinite(
    left: list[float],
    right: list[float],
) -> None:
    with pytest.raises(
        ValueError,
        match="Validation arrays contain non-finite values",
    ):
        utils.agreement_arrays(left, right, rtol=1e-8, atol=1e-8)


def test_agreement_arrays_empty() -> None:
    result = utils.agreement_arrays([], [], rtol=1e-8, atol=1e-8)

    assert result == {
        "n_validation_values": 0,
        "all_values_finite": True,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "mean_rel_diff": 0.0,
        "allclose_passed": True,
        "validation_status": "success",
    }


def test_agreement_arrays_zero_reference_uses_safe_denominator() -> None:
    result = utils.agreement_arrays(
        [1e-301],
        [0.0],
        rtol=0.0,
        atol=0.0,
    )

    assert np.isfinite(result["max_rel_diff"])
    assert result["max_rel_diff"] == pytest.approx(0.1)


def test_delta_curve() -> None:
    np.testing.assert_allclose(
        utils.delta_curve([3.0, 1.0, 2.0]),
        [2.0, 0.0, 1.0],
    )


@pytest.mark.parametrize(
    "values",
    [
        [],
        [np.nan],
        [np.inf],
    ],
)
def test_delta_curve_invalid(values: list[float]) -> None:
    with pytest.raises(
        ValueError,
        match="Cannot compute delta curve from invalid values",
    ):
        utils.delta_curve(values)


@pytest.mark.parametrize("grid_axis", ["both", "x", "y"])
def test_style_axes(grid_axis: str) -> None:
    figure, axis = plt.subplots()
    try:
        utils.style_axes(axis, grid_axis=grid_axis)
        figure.canvas.draw()
    finally:
        plt.close(figure)


def test_save_figure(tmp_path: Path) -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    output = tmp_path / "nested" / "plot.png"
    utils.save_figure(figure, output)

    assert output.is_file()
    assert output.stat().st_size > 0
    assert not plt.fignum_exists(figure.number)


def test_save_figure_calls_expected_methods(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Any]] = []

    class FakeFigure:
        def tight_layout(self) -> None:
            calls.append(("tight_layout", None))

        def savefig(self, path: Path, **kwargs: Any) -> None:
            calls.append(("savefig", (Path(path), kwargs)))
            Path(path).write_bytes(b"png")

    figure = FakeFigure()
    monkeypatch.setattr(
        utils.plt,
        "close",
        lambda fig: calls.append(("close", fig)),
    )

    output = tmp_path / "plot.png"
    utils.save_figure(figure, output)

    assert calls[0] == ("tight_layout", None)
    assert calls[1][0] == "savefig"
    saved_path, kwargs = calls[1][1]
    assert saved_path == output
    assert kwargs == {"dpi": 300, "bbox_inches": "tight"}
    assert calls[2] == ("close", figure)
