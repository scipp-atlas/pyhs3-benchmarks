from __future__ import annotations

import json
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from src import run_cross_scalar_pdf_evaluation as benchmark


@pytest.fixture
def workspace_dir(tmp_path: Path) -> Path:
    path = tmp_path / "workspaces"
    path.mkdir()
    return path


@pytest.fixture
def success_result() -> dict[str, Any]:
    return {
        "benchmark": benchmark.BENCHMARK_NAME,
        "framework": "numba_stats",
        "framework_label": "numba-stats",
        "scenario": "normal",
        "scenario_label": "Normal",
        "n_evaluations": 10,
        "requested_n_points": 5,
        "n_points": 5,
        "cold_start_time_seconds": 0.001,
        "total_runtime_seconds": 0.010,
        "average_runtime_seconds_per_evaluation": 0.001,
        "time_per_value_seconds": 0.0002,
        "time_per_value_ns": 2.0e5,
        "throughput_values_per_second": 5000.0,
        "current_rss_before_mb": 10.0,
        "current_rss_after_mb": 11.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 20.0,
        "peak_rss_after_mb": 22.0,
        "peak_rss_delta_mb": 2.0,
        "status": "success",
        "n_values": 5,
        "n_finite_values": 5,
        "all_values_finite": True,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "max_rel_diff": 0.0,
        "mean_rel_diff": 0.0,
        "allclose_passed": True,
        "validation_status": "success",
    }


@pytest.fixture
def second_success_result(success_result: dict[str, Any]) -> dict[str, Any]:
    return {
        **success_result,
        "framework": "pyhs3",
        "framework_label": "PyHS3",
        "scenario": "poisson",
        "scenario_label": "Poisson",
        "n_evaluations": 100,
        "n_points": 30,
        "requested_n_points": 1000,
        "throughput_values_per_second": 3000.0,
        "time_per_value_ns": 500.0,
        "current_rss_delta_mb": 0.0,
        "peak_rss_delta_mb": 0.0,
        "max_abs_diff": 1e-12,
        "max_rel_diff": 1e-10,
    }


def test_label_style_and_format_helpers() -> None:
    assert benchmark._framework_label("pyhs3") == "PyHS3"
    assert benchmark._framework_label("unknown") == "unknown"
    assert benchmark._scenario_label("normal") == "Normal"
    assert benchmark._scenario_label("custom") == "custom"
    assert benchmark._style_for("root")["label"] == "RooFit"
    assert benchmark._style_for("custom")["color"] == "#333333"
    assert benchmark._safe_positive(0.0) == pytest.approx(1e-300)
    assert benchmark._safe_positive(float("nan")) == pytest.approx(1e-300)
    assert benchmark._safe_positive(2.0) == pytest.approx(2.0)
    assert benchmark._format_seconds_ms(0.001) == "1.000 ms"
    assert benchmark._format_ns(999.0) == "999.0 ns"
    assert benchmark._format_ns(1000.0) == "1.00 µs"
    assert benchmark._format_scientific(0.0) == "0"
    assert benchmark._format_scientific(1e-5) == "1.0e-05"


def test_ordered_successful_results() -> None:
    success = {"status": "success"}
    failed = {"status": "failed"}
    assert benchmark._ordered_successful_results([failed, success]) == [success]


def test_validate_benchmark_config_success_without_pyhs3(workspace_dir: Path) -> None:
    benchmark.validate_benchmark_config(
        frameworks=["numba_stats", "root"],
        scenarios=["normal", "poisson"],
        n_evaluations=[1, 10],
        n_points=5,
        rtol=0.0,
        atol=0.0,
        timeout_seconds=1.0,
        pyhs3_workspace_dir=workspace_dir / "missing_is_ok_without_pyhs3",
    )


def test_validate_benchmark_config_success_with_pyhs3(workspace_dir: Path) -> None:
    benchmark.validate_benchmark_config(
        frameworks=["pyhs3"],
        scenarios=["normal"],
        n_evaluations=[1],
        n_points=1,
        rtol=1e-7,
        atol=1e-10,
        timeout_seconds=1.0,
        pyhs3_workspace_dir=workspace_dir,
    )


@pytest.mark.parametrize(
    ("override", "exception", "message"),
    [
        (
            {"frameworks": []},
            benchmark.BenchmarkConfigurationError,
            "At least one framework",
        ),
        (
            {"scenarios": []},
            benchmark.BenchmarkConfigurationError,
            "At least one scenario",
        ),
        (
            {"frameworks": ["bad"]},
            benchmark.BenchmarkConfigurationError,
            "Unknown framework",
        ),
        (
            {"scenarios": ["bad"]},
            benchmark.BenchmarkConfigurationError,
            "Unknown scenario",
        ),
        (
            {"n_evaluations": [0]},
            benchmark.BenchmarkConfigurationError,
            "n-evaluations",
        ),
        ({"n_points": 0}, benchmark.BenchmarkConfigurationError, "n-points"),
        ({"rtol": -1.0}, benchmark.BenchmarkConfigurationError, "rtol"),
        ({"atol": -1.0}, benchmark.BenchmarkConfigurationError, "atol"),
        ({"timeout_seconds": 0.0}, benchmark.BenchmarkConfigurationError, "timeout"),
        ({"frameworks": ["pyhs3"]}, FileNotFoundError, "workspace directory"),
    ],
)
def test_validate_benchmark_config_rejects_invalid(
    tmp_path: Path,
    override: dict[str, Any],
    exception: type[Exception],
    message: str,
) -> None:
    kwargs = {
        "frameworks": ["numba_stats"],
        "scenarios": ["normal"],
        "n_evaluations": [1],
        "n_points": 5,
        "rtol": 1e-7,
        "atol": 1e-10,
        "timeout_seconds": 1.0,
        "pyhs3_workspace_dir": tmp_path / "missing",
    }
    kwargs.update(override)
    with pytest.raises(exception, match=message):
        benchmark.validate_benchmark_config(**kwargs)


@pytest.mark.parametrize(
    ("scenario", "n_points", "expected_len"),
    [("normal", 5, 5), ("poisson", 5, 30), ("exponential", 7, 7)],
)
def test_make_input_grid(scenario: str, n_points: int, expected_len: int) -> None:
    grid = benchmark.make_input_grid(scenario, n_points)
    assert len(grid) == expected_len
    assert np.all(np.isfinite(grid))


def test_make_input_grid_rejects_invalid() -> None:
    with pytest.raises(benchmark.BenchmarkConfigurationError):
        benchmark.make_input_grid("normal", 0)
    with pytest.raises(ValueError, match="Unknown scenario"):
        benchmark.make_input_grid("bad", 1)


@pytest.mark.parametrize("scenario", ["normal", "poisson", "exponential"])
def test_reference_values_match_expected_shape(scenario: str) -> None:
    x = benchmark.make_input_grid(scenario, 8)
    values = benchmark.reference_values(scenario, x)
    assert values.shape == x.shape
    assert np.all(np.isfinite(values))


def test_reference_values_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        benchmark.reference_values("bad", np.asarray([1.0]))


def test_compute_agreement_success_and_failure() -> None:
    reference = np.asarray([1.0, 2.0, 3.0])
    observed = reference + 1e-12
    agreement = benchmark.compute_agreement(observed, reference, rtol=1e-9, atol=1e-9)
    assert agreement["validation_status"] == "success"
    assert agreement["allclose_passed"] is True
    assert agreement["n_values"] == 3

    failed = benchmark.compute_agreement(observed + 1.0, reference, rtol=0.0, atol=0.0)
    assert failed["validation_status"] == "failed"
    assert failed["allclose_passed"] is False


def test_compute_agreement_rejects_shape_mismatch_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        benchmark.compute_agreement(
            np.asarray([1.0]), np.asarray([1.0, 2.0]), 1e-7, 1e-10
        )

    with pytest.raises(benchmark.ValidationFailure, match="non-finite"):
        benchmark.compute_agreement(
            np.asarray([float("nan")]), np.asarray([1.0]), 1e-7, 1e-10
        )


def test_evaluate_framework_once_dispatch(
    monkeypatch: pytest.MonkeyPatch, workspace_dir: Path
) -> None:
    x = np.asarray([1.0, 2.0])
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        benchmark,
        "evaluate_pyhs3",
        lambda scenario, x, pyhs3_workspace_dir: (
            calls.append(("pyhs3", scenario)) or np.asarray([1.0, 2.0])
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_numba_stats",
        lambda scenario, x: (
            calls.append(("numba_stats", scenario)) or np.asarray([1.0, 2.0])
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_root",
        lambda scenario, x: calls.append(("root", scenario)) or np.asarray([1.0, 2.0]),
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_zfit",
        lambda scenario, x: calls.append(("zfit", scenario)) or np.asarray([1.0, 2.0]),
    )

    for framework in benchmark.SUPPORTED_FRAMEWORKS:
        values = benchmark.evaluate_framework_once(
            framework=framework,
            scenario="normal",
            x=x,
            pyhs3_workspace_dir=workspace_dir,
        )
        assert values.tolist() == [1.0, 2.0]

    assert [framework for framework, _ in calls] == list(benchmark.SUPPORTED_FRAMEWORKS)

    with pytest.raises(ValueError, match="Unknown framework"):
        benchmark.evaluate_framework_once(
            framework="bad",
            scenario="normal",
            x=x,
            pyhs3_workspace_dir=workspace_dir,
        )


def test_evaluate_pyhs3_success(
    monkeypatch: pytest.MonkeyPatch, workspace_dir: Path
) -> None:
    (workspace_dir / "normal_pdf_workspace.json").write_text("{}")

    class Model:
        data = {"x": 0.0}
        free_params = {"mu": 0.0}

        def pdf(self, name: str, **params: Any) -> np.ndarray:
            assert name == "pdf"
            return np.asarray(float(np.asarray(params["x"])) + 1.0)

    fake_workspace = SimpleNamespace(model=lambda *args, **kwargs: Model())

    class FakeWorkspace:
        @staticmethod
        def load(path: Path) -> Any:
            assert path.name == "normal_pdf_workspace.json"
            return fake_workspace

    monkeypatch.setitem(
        sys.modules, "pyhs3.workspace", SimpleNamespace(Workspace=FakeWorkspace)
    )

    values = benchmark.evaluate_pyhs3("normal", np.asarray([0.0, 1.0]), workspace_dir)

    assert values.tolist() == [1.0, 2.0]


def test_evaluate_pyhs3_rejects_missing_workspace(workspace_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PyHS3 workspace not found"):
        benchmark.evaluate_pyhs3("normal", np.asarray([0.0]), workspace_dir)


def test_evaluate_numba_stats_with_fake_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_norm = SimpleNamespace(pdf=lambda x, loc, scale: np.asarray(x) * 0.0 + 0.1)
    fake_poisson = SimpleNamespace(pmf=lambda x, mean: np.asarray(x) * 0.0 + 0.2)
    fake_expon = SimpleNamespace(pdf=lambda x, loc, scale: np.asarray(x) * 0.0 + 0.3)
    monkeypatch.setitem(
        sys.modules,
        "numba_stats",
        SimpleNamespace(norm=fake_norm, poisson=fake_poisson, expon=fake_expon),
    )

    assert benchmark.evaluate_numba_stats(
        "normal", np.asarray([0.0, 1.0])
    ).tolist() == [0.1, 0.1]
    assert benchmark.evaluate_numba_stats("poisson", np.asarray([0, 1])).tolist() == [
        0.2,
        0.2,
    ]
    assert benchmark.evaluate_numba_stats(
        "exponential", np.asarray([0.0, 1.0])
    ).tolist() == [0.3, 0.3]

    with pytest.raises(ValueError, match="Unknown scenario"):
        benchmark.evaluate_numba_stats("bad", np.asarray([0.0]))


def test_evaluate_root_with_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class RooRealVar:
        def __init__(self, *args: Any) -> None:
            self.value = args[2] if len(args) > 2 else 0.0

        def setVal(self, value: float) -> None:
            self.value = value

    class Pdf:
        def __init__(self, *args: Any) -> None:
            pass

        def getVal(self) -> float:
            return 1.0

    fake_root = SimpleNamespace(
        RooRealVar=RooRealVar,
        RooGaussian=Pdf,
        RooPoisson=Pdf,
        RooExponential=Pdf,
    )
    monkeypatch.setitem(sys.modules, "ROOT", fake_root)

    assert benchmark.evaluate_root("normal", np.asarray([0.0, 1.0])).shape == (2,)
    assert benchmark.evaluate_root("poisson", np.asarray([0.0, 1.0])).tolist() == [
        1.0,
        1.0,
    ]
    assert benchmark.evaluate_root("exponential", np.asarray([0.0, 1.0])).tolist() == [
        1.0,
        1.0,
    ]

    with pytest.raises(ValueError, match="Unknown scenario"):
        benchmark.evaluate_root("bad", np.asarray([0.0]))


def test_evaluate_zfit_with_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeData:
        @staticmethod
        def from_numpy(obs: Any, array: np.ndarray) -> np.ndarray:
            return np.asarray(array, dtype=float)

    class FakePdf:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def pdf(self, data: np.ndarray) -> Any:
            return SimpleNamespace(numpy=lambda: np.asarray(data) * 0.0 + 0.5)

    fake_zfit = SimpleNamespace(
        Space=lambda *args, **kwargs: object(),
        Data=FakeData,
        pdf=SimpleNamespace(Gauss=FakePdf, Exponential=FakePdf),
    )
    monkeypatch.setitem(sys.modules, "zfit", fake_zfit)

    assert benchmark.evaluate_zfit("normal", np.asarray([0.0, 1.0])).tolist() == [
        0.5,
        0.5,
    ]
    assert benchmark.evaluate_zfit("exponential", np.asarray([0.0, 1.0])).tolist() == [
        0.5,
        0.5,
    ]

    with pytest.raises(NotImplementedError, match="Poisson"):
        benchmark.evaluate_zfit("poisson", np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="Unknown scenario"):
        benchmark.evaluate_zfit("bad", np.asarray([0.0]))


def test_run_single_framework_benchmark_success(
    monkeypatch: pytest.MonkeyPatch, workspace_dir: Path
) -> None:
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 10.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 20.0)
    times = iter([1.0, 1.2, 2.0, 2.5])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))

    def fake_eval(**kwargs: Any) -> np.ndarray:
        x = kwargs["x"]
        return benchmark.reference_values(kwargs["scenario"], x)

    monkeypatch.setattr(benchmark, "evaluate_framework_once", fake_eval)

    config = benchmark.ScalarBenchmarkConfig(
        framework="numba_stats",
        scenario="normal",
        n_evaluations=1,
        n_points=3,
        rtol=1e-7,
        atol=1e-10,
        pyhs3_workspace_dir=workspace_dir,
    )
    result = benchmark.run_single_framework_benchmark(config)

    assert result["status"] == "success"
    assert result["validation_status"] == "success"
    assert result["n_points"] == 3
    assert result["throughput_values_per_second"] == pytest.approx(6.0)


def test_run_single_framework_benchmark_raises_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    workspace_dir: Path,
) -> None:
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 10.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 20.0)
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: 1.0)
    monkeypatch.setattr(
        benchmark,
        "evaluate_framework_once",
        lambda **kwargs: np.zeros_like(kwargs["x"], dtype=float),
    )

    config = benchmark.ScalarBenchmarkConfig(
        framework="numba_stats",
        scenario="normal",
        n_evaluations=1,
        n_points=3,
        rtol=0.0,
        atol=0.0,
        pyhs3_workspace_dir=workspace_dir,
    )

    with pytest.raises(benchmark.ValidationFailure, match="PDF value agreement failed"):
        benchmark.run_single_framework_benchmark(config)


def test_failure_result(workspace_dir: Path) -> None:
    config = benchmark.ScalarBenchmarkConfig(
        "numba_stats", "poisson", 5, 10, 1e-7, 1e-10, workspace_dir
    )
    result = benchmark._failure_result(
        config, "failed", error_type="X", error_message="bad"
    )
    assert result["status"] == "failed"
    assert result["scenario_label"] == "Poisson"
    assert result["n_points"] == 30
    assert result["error_type"] == "X"


def test_run_worker_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, workspace_dir: Path
) -> None:
    class Queue:
        def __init__(self) -> None:
            self.items: list[Any] = []

        def put(self, item: Any) -> None:
            self.items.append(item)

    payload = {
        "framework": "numba_stats",
        "scenario": "normal",
        "n_evaluations": 1,
        "n_points": 2,
        "rtol": 1e-7,
        "atol": 1e-10,
        "pyhs3_workspace_dir": str(workspace_dir),
    }

    q = Queue()
    monkeypatch.setattr(
        benchmark,
        "run_single_framework_benchmark",
        lambda config: {"status": "success"},
    )
    benchmark.run_worker(payload, q)
    assert q.items == [{"status": "success"}]

    q = Queue()
    monkeypatch.setattr(
        benchmark,
        "run_single_framework_benchmark",
        lambda config: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    benchmark.run_worker(payload, q)
    assert q.items[0]["status"] == "failed"
    assert q.items[0]["error_type"] == "RuntimeError"


def test_run_with_timeout_success(
    monkeypatch: pytest.MonkeyPatch, workspace_dir: Path
) -> None:
    class FakeQueue:
        def __init__(self, maxsize: int = 1) -> None:
            self.item = {"status": "success"}

        def get_nowait(self) -> dict[str, Any]:
            return self.item

    class FakeProcess:
        exitcode = 0

        def __init__(self, target: Any, args: tuple[Any, ...]) -> None:
            self.target = target
            self.args = args

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    class FakeContext:
        Queue = FakeQueue
        Process = FakeProcess

    monkeypatch.setattr(benchmark.mp, "get_context", lambda method: FakeContext())

    payload = {
        "framework": "numba_stats",
        "scenario": "normal",
        "n_evaluations": 1,
        "n_points": 2,
        "rtol": 1e-7,
        "atol": 1e-10,
        "pyhs3_workspace_dir": str(workspace_dir),
    }

    assert benchmark.run_with_timeout(payload, 1.0) == {"status": "success"}


def test_run_with_timeout_timeout_exit_error_and_empty(
    monkeypatch: pytest.MonkeyPatch, workspace_dir: Path
) -> None:
    payload = {
        "framework": "numba_stats",
        "scenario": "normal",
        "n_evaluations": 1,
        "n_points": 2,
        "rtol": 1e-7,
        "atol": 1e-10,
        "pyhs3_workspace_dir": str(workspace_dir),
    }

    class EmptyQueue:
        def __init__(self, maxsize: int = 1) -> None:
            pass

        def get_nowait(self) -> Any:
            raise queue.Empty

    class AliveThenDeadProcess:
        exitcode = 0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.alive_calls = 0

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

        def is_alive(self) -> bool:
            self.alive_calls += 1
            return self.alive_calls == 1

    class ExitCodeProcess(AliveThenDeadProcess):
        exitcode = 7

        def is_alive(self) -> bool:
            return False

    class EmptyProcess(AliveThenDeadProcess):
        exitcode = 0

        def is_alive(self) -> bool:
            return False

    class Context:
        Queue = EmptyQueue
        Process = AliveThenDeadProcess

    monkeypatch.setattr(benchmark.mp, "get_context", lambda method: Context())
    assert benchmark.run_with_timeout(payload, 0.01)["status"] == "timeout"

    Context.Process = ExitCodeProcess
    result = benchmark.run_with_timeout(payload, 0.01)
    assert result["status"] == "failed"
    assert result["error_type"] == "ProcessExitError"

    Context.Process = EmptyProcess
    result = benchmark.run_with_timeout(payload, 0.01)
    assert result["status"] == "failed"
    assert result["error_type"] == "EmptyWorkerResult"


def test_print_result_success_and_failure(
    capsys: pytest.CaptureFixture[str], success_result: dict[str, Any]
) -> None:
    benchmark.print_result(success_result)
    output = capsys.readouterr().out
    assert "Normal / numba-stats" in output
    assert "throughput" in output

    failed = {
        "scenario": "normal",
        "scenario_label": "Normal",
        "framework": "numba_stats",
        "framework_label": "numba-stats",
        "n_evaluations": 1,
        "status": "failed",
        "error_type": "RuntimeError",
        "error_message": "boom",
    }
    benchmark.print_result(failed)
    output = capsys.readouterr().out
    assert "validation:              failed" in output
    assert "RuntimeError: boom" in output


def test_summarize_status_and_final_summary(
    capsys: pytest.CaptureFixture[str], success_result: dict[str, Any]
) -> None:
    failed = {
        "status": "timeout",
        "scenario": "poisson",
        "framework": "root",
        "n_evaluations": 1,
        "error_type": "TimeoutError",
        "error_message": "slow",
    }
    summary = benchmark.summarize_status([success_result, failed])
    assert summary["status"] == "failed"
    assert summary["n_successful"] == 1
    assert summary["failed_results"][0]["error_type"] == "TimeoutError"

    benchmark.print_final_summary([success_result, failed])
    output = capsys.readouterr().out
    assert "Failed:" in output
    assert "poisson / root" in output


def test_success_dataframe(
    success_result: dict[str, Any], second_success_result: dict[str, Any]
) -> None:
    df = benchmark._success_dataframe(
        [success_result, {"status": "failed"}, second_success_result]
    )
    assert list(df["framework_key"]) == ["numba_stats", "pyhs3"]
    assert "throughput" in df.columns


def test_prepare_figure_and_save(tmp_path: Path) -> None:
    fig, ax = benchmark._prepare_figure()
    assert fig is not None
    assert ax is not None
    output = tmp_path / "figure.png"
    benchmark._save_figure(fig, output)
    assert output.exists()


@pytest.mark.parametrize(
    "plot_func",
    [
        benchmark.make_throughput_plot,
        benchmark.make_latency_plot,
        benchmark.make_time_per_value_plot,
        benchmark.make_memory_plot,
        benchmark.make_summary_table,
    ],
)
def test_plot_functions_create_png(
    tmp_path: Path,
    success_result: dict[str, Any],
    second_success_result: dict[str, Any],
    plot_func: Any,
) -> None:
    output = tmp_path / f"{plot_func.__name__}.png"
    plot_func([success_result, second_success_result], output)
    assert output.exists()


def test_make_agreement_plot_creates_png(
    tmp_path: Path,
    success_result: dict[str, Any],
    second_success_result: dict[str, Any],
) -> None:
    output = tmp_path / "agreement.png"
    benchmark.make_agreement_plot(
        [success_result, second_success_result], output, tolerance=1e-10
    )
    assert output.exists()


@pytest.mark.parametrize(
    "plot_func",
    [
        benchmark.make_throughput_plot,
        benchmark.make_latency_plot,
        benchmark.make_time_per_value_plot,
        benchmark.make_memory_plot,
        benchmark.make_agreement_plot,
        benchmark.make_summary_table,
    ],
)
def test_plot_functions_return_on_empty_success(tmp_path: Path, plot_func: Any) -> None:
    output = tmp_path / f"{plot_func.__name__}.png"
    if plot_func is benchmark.make_agreement_plot:
        plot_func([{"status": "failed"}], output, tolerance=1e-10)
    else:
        plot_func([{"status": "failed"}], output)
    assert not output.exists()


def test_make_plots_creates_all_pngs(
    tmp_path: Path,
    success_result: dict[str, Any],
    second_success_result: dict[str, Any],
) -> None:
    benchmark.make_plots(
        [success_result, second_success_result], tmp_path, tolerance=1e-10
    )
    expected = {
        "cross_scalar_pdf_throughput_scaling.png",
        "cross_scalar_pdf_time_per_value.png",
        "cross_scalar_pdf_latency.png",
        "cross_scalar_pdf_memory.png",
        "cross_scalar_pdf_numerical_agreement.png",
        "cross_scalar_pdf_summary_table.png",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})


def test_make_plots_skips_when_less_than_two_successes(
    tmp_path: Path,
    success_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark.make_plots([success_result], tmp_path, tolerance=1e-10)
    assert "Skipping plots" in capsys.readouterr().out


def test_build_payload(workspace_dir: Path) -> None:
    args = SimpleNamespace(
        n_points=12, rtol=1e-7, atol=1e-10, pyhs3_workspace_dir=workspace_dir
    )
    payload = benchmark.build_payload(args, "numba_stats", "normal", 5)
    assert payload == {
        "framework": "numba_stats",
        "scenario": "normal",
        "n_evaluations": 5,
        "n_points": 12,
        "rtol": 1e-7,
        "atol": 1e-10,
        "pyhs3_workspace_dir": str(workspace_dir),
    }


def test_run_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_dir: Path,
    success_result: dict[str, Any],
    second_success_result: dict[str, Any],
) -> None:
    returned = [success_result, second_success_result]
    calls: list[dict[str, Any]] = []

    def fake_run_with_timeout(
        payload: dict[str, Any], timeout_seconds: float
    ) -> dict[str, Any]:
        calls.append(payload)
        return returned[len(calls) - 1]

    monkeypatch.setattr(benchmark, "run_with_timeout", fake_run_with_timeout)
    plot_calls: list[Any] = []
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda *args, **kwargs: plot_calls.append((args, kwargs)),
    )

    output = benchmark.run(
        frameworks=["numba_stats"],
        scenarios=["normal"],
        n_evaluations=[10, 100],
        n_points=5,
        pyhs3_workspace_dir=workspace_dir,
        rtol=1e-7,
        atol=1e-10,
        timeout_seconds=1.0,
        output_dir=tmp_path,
        output_name="result.json",
        plot=True,
        plot_dir=tmp_path / "plots",
    )

    assert output["summary"]["status"] == "success"
    assert output["summary"]["n_results"] == 2
    assert (
        json.loads((tmp_path / "result.json").read_text())["summary"]["status"]
        == "success"
    )
    assert len(calls) == 2
    assert plot_calls


def test_main_wraps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "parse_args",
        lambda argv=None: SimpleNamespace(
            frameworks=["numba_stats"],
            scenarios=["normal"],
            n_evaluations=[1],
            n_points=1,
            pyhs3_workspace_dir=Path("missing"),
            rtol=1e-7,
            atol=1e-10,
            timeout_seconds=1.0,
            output_dir=Path("out"),
            output_name="result.json",
            plot=False,
            plot_dir=Path("plots"),
        ),
    )
    monkeypatch.setattr(
        benchmark, "run", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="Cross-framework scalar PDF"):
        benchmark.main([])


def test_main_passes_cli_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(benchmark, "run", lambda **kwargs: calls.append(kwargs))
    benchmark.main(
        [
            "--frameworks",
            "numba_stats",
            "--scenarios",
            "normal",
            "--n-evaluations",
            "1",
            "2",
            "--n-points",
            "3",
            "--pyhs3-workspace-dir",
            str(tmp_path),
            "--rtol",
            "1e-6",
            "--atol",
            "1e-9",
            "--timeout-seconds",
            "5",
            "--output-dir",
            str(tmp_path / "out"),
            "--output-name",
            "out.json",
            "--plot",
            "--plot-dir",
            str(tmp_path / "plots"),
        ]
    )

    assert calls[0]["frameworks"] == ["numba_stats"]
    assert calls[0]["scenarios"] == ["normal"]
    assert calls[0]["n_evaluations"] == [1, 2]
    assert calls[0]["n_points"] == 3
    assert calls[0]["plot"] is True
