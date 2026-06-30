from __future__ import annotations

import math
import sys
from multiprocessing import TimeoutError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from src import run_cross_vectorized_pdf_evaluation as benchmark


def success_result(
    framework: str = "pyhs3",
    scenario: str = "normal",
    n_points: int = 100,
    *,
    native_vectorized: bool = False,
    max_abs_diff: float = 0.0,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark.BENCHMARK_NAME,
        "framework": framework,
        "framework_label": benchmark._framework_label(framework),
        "scenario": scenario,
        "scenario_label": benchmark._scenario_label(scenario),
        "n_points": n_points,
        "n_runs": 2,
        "native_vectorized": native_vectorized,
        "status": "success",
        "validation_status": "success",
        "setup_time_seconds": 0.01,
        "cold_vectorized_eval_time_seconds": 0.02,
        "warm_vectorized_eval_time_seconds_mean": 0.03,
        "warm_vectorized_eval_time_seconds_median": 0.025,
        "warm_vectorized_eval_time_seconds_std": 0.001,
        "throughput_values_per_second": n_points / 0.03,
        "time_per_value_seconds": 0.03 / n_points,
        "current_rss_before_mb": 10.0,
        "current_rss_after_mb": 11.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 20.0,
        "peak_rss_after_mb": 22.0,
        "peak_rss_delta_mb": 2.0,
        "agreement": {
            "n_values": n_points,
            "n_finite_values": n_points,
            "all_values_finite": True,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": max_abs_diff / 2.0,
            "max_rel_diff": 0.0,
            "mean_rel_diff": 0.0,
            "allclose_passed": True,
            "validation_status": "success",
        },
        "repeat_agreement": {
            "n_values": n_points,
            "n_finite_values": n_points,
            "all_values_finite": True,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": max_abs_diff / 2.0,
            "max_rel_diff": 0.0,
            "mean_rel_diff": 0.0,
            "allclose_passed": True,
            "validation_status": "success",
        },
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": max_abs_diff / 2.0,
        "max_rel_diff": 0.0,
        "mean_rel_diff": 0.0,
        "allclose_passed": True,
    }


@pytest.fixture
def plot_results() -> list[dict[str, Any]]:
    return [
        success_result(
            "pyhs3", "normal", 100, native_vectorized=False, max_abs_diff=0.0
        ),
        success_result(
            "numba_stats", "normal", 1000, native_vectorized=True, max_abs_diff=1e-13
        ),
        success_result(
            "root", "poisson", 100, native_vectorized=False, max_abs_diff=2e-13
        ),
        success_result(
            "zfit", "poisson", 1000, native_vectorized=True, max_abs_diff=0.0
        ),
    ]


def test_label_helpers_use_known_and_fallback_values() -> None:
    assert benchmark._framework_label("pyhs3") == "PyHS3"
    assert benchmark._framework_label("unknown") == "unknown"
    assert benchmark._scenario_label("normal") == "Normal"
    assert benchmark._scenario_label("custom") == "custom"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "nan"),
        (0.0, "0"),
        (1500.0, "1.5e+03"),
        (150.0, "150"),
        (15.5, "15.5"),
        (1.234, "1.23"),
        (0.01234, "0.012"),
    ],
)
def test_format_compact(value: float, expected: str) -> None:
    assert benchmark._format_compact(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "nan"),
        (150.0, "150"),
        (15.5, "15.5"),
        (1.234, "1.23"),
        (0.01234, "0.012"),
        (1e-5, "1.0e-05"),
    ],
)
def test_format_plain_number(value: float, expected: str) -> None:
    assert benchmark._format_plain_number(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (float("nan"), "nan MB"),
        (150.0, "150 MB"),
        (15.25, "15.2 MB"),
        (1.234, "1.23 MB"),
    ],
)
def test_format_memory_mb(value: float, expected: str) -> None:
    assert benchmark._format_memory_mb(value) == expected


def test_add_bar_labels_adds_one_annotation_per_bar() -> None:
    fig, ax = benchmark.plt.subplots()
    bars = ax.bar([0, 1], [0.0, 2.0])
    benchmark._add_bar_labels(ax, bars, [0.0, 2.0])
    assert len(ax.texts) == 2
    benchmark.plt.close(fig)


def test_save_figure_creates_parent_directory(tmp_path: Path) -> None:
    fig, _ax = benchmark.plt.subplots()
    output = tmp_path / "nested" / "figure.png"
    benchmark._save_figure(fig, output)
    assert output.exists()


@pytest.mark.parametrize("value", [1, 5])
def test_validate_positive_int_success(value: int) -> None:
    benchmark.validate_positive_int(value, "value")


def test_validate_positive_int_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="value must be at least 1"):
        benchmark.validate_positive_int(0, "value")


@pytest.mark.parametrize("value", [0.1, 1.0])
def test_validate_positive_float_success(value: float) -> None:
    benchmark.validate_positive_float(value, "value")


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_validate_positive_float_rejects_invalid(value: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        benchmark.validate_positive_float(value, "value")


@pytest.mark.parametrize("value", [0.0, 1e-6, 1.0])
def test_validate_probability_tolerance_success(value: float) -> None:
    benchmark.validate_probability_tolerance(value, "tol")


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_validate_probability_tolerance_rejects_invalid(value: float) -> None:
    with pytest.raises(ValueError, match="non-negative finite"):
        benchmark.validate_probability_tolerance(value, "tol")


def test_validate_choices_success_and_failure() -> None:
    benchmark.validate_choices(["normal"], benchmark.SUPPORTED_SCENARIOS, "scenarios")
    with pytest.raises(ValueError, match="Unsupported scenarios"):
        benchmark.validate_choices(["bad"], benchmark.SUPPORTED_SCENARIOS, "scenarios")


def test_validate_config_success_without_pyhs3(tmp_path: Path) -> None:
    result = benchmark.validate_config(
        frameworks=["numba_stats"],
        scenarios=["normal"],
        n_points=[10],
        n_runs=1,
        timeout_seconds=1.0,
        rtol=0.0,
        atol=0.0,
        pyhs3_workspace_dir=tmp_path / "missing",
    )
    assert result == tmp_path / "missing"


def test_validate_config_uses_fallback_workspace_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    default = tmp_path / "default"
    fallback = tmp_path / "fallback"
    fallback.mkdir()
    monkeypatch.setattr(benchmark, "DEFAULT_PYHS3_WORKSPACE_DIR", default)
    monkeypatch.setattr(benchmark, "FALLBACK_PYHS3_WORKSPACE_DIR", fallback)

    result = benchmark.validate_config(
        frameworks=["pyhs3"],
        scenarios=["normal"],
        n_points=[10],
        n_runs=1,
        timeout_seconds=1.0,
        rtol=0.0,
        atol=0.0,
        pyhs3_workspace_dir=default,
    )

    assert result == fallback


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"frameworks": ["bad"]}, "Unsupported frameworks"),
        ({"scenarios": ["bad"]}, "Unsupported scenarios"),
        ({"frameworks": []}, "At least one framework"),
        ({"scenarios": []}, "At least one scenario"),
        ({"n_points": []}, "At least one --n-points"),
        ({"n_points": [0]}, "--n-points"),
        ({"n_runs": 0}, "--n-runs"),
        ({"timeout_seconds": 0.0}, "--timeout-seconds"),
        ({"rtol": -1.0}, "--rtol"),
        ({"atol": -1.0}, "--atol"),
    ],
)
def test_validate_config_rejects_invalid(
    tmp_path: Path, override: dict[str, Any], message: str
) -> None:
    kwargs: dict[str, Any] = {
        "frameworks": ["numba_stats"],
        "scenarios": ["normal"],
        "n_points": [10],
        "n_runs": 1,
        "timeout_seconds": 1.0,
        "rtol": 0.0,
        "atol": 0.0,
        "pyhs3_workspace_dir": tmp_path,
    }
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        benchmark.validate_config(**kwargs)


def test_validate_config_rejects_missing_pyhs3_workspace_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PyHS3 workspace directory"):
        benchmark.validate_config(
            frameworks=["pyhs3"],
            scenarios=["normal"],
            n_points=[10],
            n_runs=1,
            timeout_seconds=1.0,
            rtol=0.0,
            atol=0.0,
            pyhs3_workspace_dir=tmp_path / "missing",
        )


def test_make_input_grid_normal_and_poisson() -> None:
    normal = benchmark.make_input_grid("normal", 5)
    poisson = benchmark.make_input_grid("poisson", 35)

    assert normal.tolist() == pytest.approx([-5.0, -2.5, 0.0, 2.5, 5.0])
    assert poisson.shape == (35,)
    assert poisson[:30].tolist() == list(range(30))
    assert poisson[30:].tolist() == list(range(5))


def test_make_input_grid_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        benchmark.make_input_grid("bad", 3)


def test_reference_values_normal_and_poisson() -> None:
    x_normal = np.asarray([0.0])
    x_poisson = np.asarray([0, 1, 2])
    assert benchmark.reference_values("normal", x_normal)[0] == pytest.approx(
        1.0 / math.sqrt(2.0 * math.pi)
    )
    assert benchmark.reference_values("poisson", x_poisson).shape == (3,)


def test_reference_values_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="Unknown scenario"):
        benchmark.reference_values("bad", np.asarray([1.0]))


def test_pyhs3_evaluator_success_and_missing_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace_path = tmp_path / "normal_pdf_workspace.json"
    workspace_path.write_text("{}")

    class FakeModel:
        data = {"x": 0.0}
        free_params = {"mu": 0.0}

        def pdf(self, name: str, **params: Any) -> np.ndarray:
            assert name == "pdf"
            return np.asarray(0.25 + float(np.asarray(params["x"])) * 0.0)

    fake_workspace = SimpleNamespace(
        model=lambda *args, **kwargs: FakeModel(),
    )

    class FakeWorkspace:
        @staticmethod
        def load(path: Path) -> Any:
            assert path == workspace_path
            return fake_workspace

    monkeypatch.setitem(
        sys.modules, "pyhs3.workspace", SimpleNamespace(Workspace=FakeWorkspace)
    )
    evaluator = benchmark.PyHS3Evaluator("normal", tmp_path)

    assert evaluator.is_native_vectorized is False
    assert evaluator.evaluate(np.asarray([1.0, 2.0])).tolist() == [0.25, 0.25]

    with pytest.raises(FileNotFoundError, match="PyHS3 workspace not found"):
        benchmark.PyHS3Evaluator("poisson", tmp_path)


def test_numba_stats_evaluator_normal_and_poisson(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_norm = SimpleNamespace(
        pdf=lambda x, mu, sigma: np.ones_like(x, dtype=float) * 0.5
    )
    fake_poisson = SimpleNamespace(
        pmf=lambda x, mean: np.ones_like(x, dtype=float) * 0.25
    )
    monkeypatch.setitem(
        sys.modules,
        "numba_stats",
        SimpleNamespace(norm=fake_norm, poisson=fake_poisson),
    )

    assert benchmark.NumbaStatsEvaluator("normal").evaluate(
        np.asarray([0.0, 1.0])
    ).tolist() == [0.5, 0.5]
    assert benchmark.NumbaStatsEvaluator("poisson").evaluate(
        np.asarray([0, 1])
    ).tolist() == [0.25, 0.25]

    with pytest.raises(ValueError, match="Unknown scenario"):
        benchmark.NumbaStatsEvaluator("bad")


def test_numba_stats_evaluator_evaluate_rejects_unknown_runtime_scenario(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_norm = SimpleNamespace(pdf=lambda x, mu, sigma: np.ones_like(x, dtype=float))
    monkeypatch.setitem(sys.modules, "numba_stats", SimpleNamespace(norm=fake_norm))
    evaluator = benchmark.NumbaStatsEvaluator("normal")
    evaluator.scenario = "bad"
    with pytest.raises(ValueError, match="Unknown scenario"):
        evaluator.evaluate(np.asarray([1.0]))


def test_root_evaluator_normal_and_poisson(monkeypatch: pytest.MonkeyPatch) -> None:
    class RooRealVar:
        def __init__(self, *args: Any) -> None:
            self.value = args[2] if len(args) > 2 else 0.0

        def setVal(self, value: float) -> None:
            self.value = value

    class Pdf:
        def __init__(self, *args: Any) -> None:
            self.args = args

        def getVal(self) -> float:
            return 1.0

    fake_root = SimpleNamespace(
        RooRealVar=RooRealVar,
        RooGaussian=Pdf,
        RooPoisson=Pdf,
    )
    monkeypatch.setitem(sys.modules, "ROOT", fake_root)

    normal = benchmark.RootEvaluator("normal", np.asarray([0.0, 1.0]))
    poisson = benchmark.RootEvaluator("poisson", np.asarray([0, 1]))
    assert normal.evaluate(np.asarray([0.0, 1.0])).tolist() == pytest.approx(
        [1 / math.sqrt(2 * math.pi)] * 2
    )
    assert poisson.evaluate(np.asarray([0, 1])).tolist() == [1.0, 1.0]

    with pytest.raises(ValueError, match="Unknown scenario"):
        benchmark.RootEvaluator("bad", np.asarray([0.0]))


def test_zfit_evaluator_normal_and_unsupported_poisson(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Data:
        @staticmethod
        def from_numpy(obs: Any, array: np.ndarray) -> np.ndarray:
            return np.asarray(array, dtype=float)

    class FakePdfResult:
        def __init__(self, size: int) -> None:
            self.size = size

        def numpy(self) -> np.ndarray:
            return np.ones(self.size, dtype=float) * 0.75

    class Gauss:
        def __init__(self, obs: Any, mu: float, sigma: float) -> None:
            self.obs = obs

        def pdf(self, data: np.ndarray) -> FakePdfResult:
            return FakePdfResult(len(data))

    fake_zfit = SimpleNamespace(
        Space=lambda name, limits: ("space", name, limits),
        Data=Data,
        pdf=SimpleNamespace(Gauss=Gauss),
    )
    monkeypatch.setitem(sys.modules, "zfit", fake_zfit)

    evaluator = benchmark.ZfitEvaluator("normal", np.asarray([0.0, 1.0]))
    assert evaluator.evaluate(np.asarray([0.0, 1.0])).tolist() == [0.75, 0.75]

    with pytest.raises(NotImplementedError, match="zfit Poisson"):
        benchmark.ZfitEvaluator("poisson", np.asarray([0, 1]))
    with pytest.raises(ValueError, match="Unknown scenario"):
        benchmark.ZfitEvaluator("bad", np.asarray([0.0]))


def test_create_evaluator_dispatches_and_rejects_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, str]] = []

    class FakePyHS3:
        def __init__(self, scenario: str, pyhs3_workspace_dir: Path) -> None:
            calls.append(("pyhs3", scenario))

    class FakeNumba:
        def __init__(self, scenario: str) -> None:
            calls.append(("numba_stats", scenario))

    class FakeRoot:
        def __init__(self, scenario: str, x: np.ndarray) -> None:
            calls.append(("root", scenario))

    class FakeZfit:
        def __init__(self, scenario: str, x: np.ndarray) -> None:
            calls.append(("zfit", scenario))

    monkeypatch.setattr(benchmark, "PyHS3Evaluator", FakePyHS3)
    monkeypatch.setattr(benchmark, "NumbaStatsEvaluator", FakeNumba)
    monkeypatch.setattr(benchmark, "RootEvaluator", FakeRoot)
    monkeypatch.setattr(benchmark, "ZfitEvaluator", FakeZfit)

    for framework in benchmark.SUPPORTED_FRAMEWORKS:
        benchmark.create_evaluator(framework, "normal", np.asarray([1.0]), tmp_path)

    assert calls == [
        ("pyhs3", "normal"),
        ("numba_stats", "normal"),
        ("root", "normal"),
        ("zfit", "normal"),
    ]
    with pytest.raises(ValueError, match="Unknown framework"):
        benchmark.create_evaluator("bad", "normal", np.asarray([1.0]), tmp_path)


def test_validate_values_success_and_failures() -> None:
    assert benchmark.validate_values([1.0, 2.0], 2, "ctx").tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="returned 1 values"):
        benchmark.validate_values([1.0], 2, "ctx")
    with pytest.raises(ValueError, match="non-finite"):
        benchmark.validate_values([1.0, float("nan")], 2, "ctx")


def test_compute_agreement_success_and_failure() -> None:
    observed = np.asarray([1.0, 2.0])
    reference = np.asarray([1.0, 2.0])
    agreement = benchmark.compute_agreement(observed, reference, rtol=0.0, atol=0.0)
    assert agreement["validation_status"] == "success"
    assert agreement["max_abs_diff"] == 0.0

    failed = benchmark.compute_agreement(
        np.asarray([1.0, 2.1]), reference, rtol=0.0, atol=0.0
    )
    assert failed["validation_status"] == "failed"
    assert failed["allclose_passed"] is False


def test_summarize_timings_success_single_and_multiple() -> None:
    assert benchmark.summarize_timings([0.1])["std_seconds"] == 0.0
    summary = benchmark.summarize_timings([0.1, 0.3])
    assert summary["mean_seconds"] == pytest.approx(0.2)
    assert summary["median_seconds"] == pytest.approx(0.2)
    assert summary["std_seconds"] > 0.0


@pytest.mark.parametrize("timings", [[], [0.0], [-1.0], [float("nan")]])
def test_summarize_timings_rejects_invalid(timings: list[float]) -> None:
    with pytest.raises(ValueError):
        benchmark.summarize_timings(timings)


def test_run_single_framework_benchmark_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeEvaluator:
        is_native_vectorized = True

        def evaluate(self, x: np.ndarray) -> np.ndarray:
            return benchmark.reference_values("normal", x)

    monkeypatch.setattr(
        benchmark, "create_evaluator", lambda *args, **kwargs: FakeEvaluator()
    )
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 10.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 20.0)

    spec = benchmark.BenchmarkSpec("numba_stats", "normal", 5, 2, 0.0, 0.0, tmp_path)
    result = benchmark.run_single_framework_benchmark(spec)

    assert result["status"] == "success"
    assert result["native_vectorized"] is True
    assert result["n_points"] == 5
    assert result["max_abs_diff"] == 0.0


def test_run_single_framework_benchmark_records_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class BadEvaluator:
        is_native_vectorized = False

        def evaluate(self, x: np.ndarray) -> np.ndarray:
            return np.zeros_like(x, dtype=float)

    monkeypatch.setattr(
        benchmark, "create_evaluator", lambda *args, **kwargs: BadEvaluator()
    )

    spec = benchmark.BenchmarkSpec("root", "normal", 5, 1, 0.0, 0.0, tmp_path)
    result = benchmark.run_single_framework_benchmark(spec)

    assert result["status"] == "failed"
    assert result["error_type"] == "ValidationFailure"
    assert "PDF value agreement failed" in result["error_message"]


def test_failed_result_contains_traceback(tmp_path: Path) -> None:
    spec = benchmark.BenchmarkSpec("root", "normal", 10, 1, 1e-7, 1e-10, tmp_path)
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        result = benchmark.failed_result(spec, exc, status="custom")

    assert result["status"] == "custom"
    assert result["error_type"] == "RuntimeError"
    assert result["framework_label"] == "RooFit"
    assert "traceback" in result


def test_run_worker_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "framework": "numba_stats",
        "scenario": "normal",
        "n_points": 10,
        "n_runs": 1,
        "rtol": 1e-7,
        "atol": 1e-10,
        "pyhs3_workspace_dir": tmp_path,
    }
    monkeypatch.setattr(
        benchmark,
        "run_single_framework_benchmark",
        lambda spec: {"status": "success", "framework": spec.framework},
    )
    assert benchmark.run_worker(payload)["status"] == "success"

    monkeypatch.setattr(
        benchmark,
        "run_single_framework_benchmark",
        lambda spec: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    failed = benchmark.run_worker(payload)
    assert failed["status"] == "failed"
    assert failed["error_type"] == "RuntimeError"


def test_run_with_timeout_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeAsync:
        def get(self, timeout: float) -> dict[str, Any]:
            assert timeout == 3.0
            return {"status": "success"}

    class FakePool:
        def __init__(self, processes: int) -> None:
            assert processes == 1

        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def apply_async(self, func: Any, args: tuple[Any, ...]) -> FakeAsync:
            assert func is benchmark.run_worker
            assert args[0]["framework"] == "numba_stats"
            return FakeAsync()

    monkeypatch.setattr(
        benchmark, "get_context", lambda method: SimpleNamespace(Pool=FakePool)
    )
    spec = benchmark.BenchmarkSpec(
        "numba_stats", "normal", 10, 1, 1e-7, 1e-10, tmp_path
    )

    assert benchmark.run_with_timeout(spec, timeout_seconds=3.0) == {
        "status": "success"
    }


def test_run_with_timeout_returns_timeout_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeAsync:
        def get(self, timeout: float) -> dict[str, Any]:
            raise TimeoutError

    class FakePool:
        terminated = False
        joined = False

        def __init__(self, processes: int) -> None:
            pass

        def __enter__(self) -> "FakePool":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def apply_async(self, func: Any, args: tuple[Any, ...]) -> FakeAsync:
            return FakeAsync()

        def terminate(self) -> None:
            self.terminated = True

        def join(self) -> None:
            self.joined = True

    monkeypatch.setattr(
        benchmark, "get_context", lambda method: SimpleNamespace(Pool=FakePool)
    )
    spec = benchmark.BenchmarkSpec("root", "poisson", 10, 1, 1e-7, 1e-10, tmp_path)

    result = benchmark.run_with_timeout(spec, timeout_seconds=0.5)

    assert result["status"] == "timeout"
    assert result["error_type"] == "TimeoutError"
    assert result["timeout_seconds"] == 0.5


def test_print_result_success_and_failure(capsys: pytest.CaptureFixture[str]) -> None:
    benchmark.print_result(success_result("pyhs3", "normal", 100))
    output = capsys.readouterr().out
    assert "evaluation mode" in output
    assert "throughput" in output

    failed = {
        "scenario_label": "Normal",
        "framework_label": "PyHS3",
        "n_points": 100,
        "status": "failed",
        "validation_status": "failed",
        "error_type": "X",
        "error_message": "bad",
    }
    benchmark.print_result(failed)
    output = capsys.readouterr().out
    assert "error:                   X: bad" in output


def test_successful_results_filters() -> None:
    ok = success_result()
    assert benchmark.successful_results([ok, {"status": "failed"}]) == [ok]


def test_values_for_sorts_by_n_points() -> None:
    results = [
        success_result("pyhs3", "normal", 1000),
        success_result("pyhs3", "normal", 100),
        success_result("root", "normal", 100),
        {
            "status": "failed",
            "framework": "pyhs3",
            "scenario": "normal",
            "n_points": 10,
        },
    ]
    xs, ys = benchmark._values_for(
        results, "normal", "pyhs3", "throughput_values_per_second"
    )
    assert xs == [100, 1000]
    assert ys[0] > 0.0


@pytest.mark.parametrize(
    ("plot_func", "filename", "extra_args"),
    [
        (benchmark.make_throughput_plot, "throughput.png", ()),
        (benchmark.make_time_per_value_plot, "time_per_value.png", ()),
        (benchmark.make_agreement_plot, "agreement.png", (1e-10,)),
        (benchmark.make_memory_plot, "memory.png", ()),
        (benchmark.make_summary_table, "summary.png", ()),
    ],
)
def test_individual_plot_functions_create_png(
    tmp_path: Path,
    plot_results: list[dict[str, Any]],
    plot_func: Any,
    filename: str,
    extra_args: tuple[Any, ...],
) -> None:
    output = tmp_path / filename
    plot_func(plot_results, output, *extra_args)
    assert output.exists()


@pytest.mark.parametrize(
    ("plot_func", "extra_args"),
    [
        (benchmark.make_throughput_plot, ()),
        (benchmark.make_time_per_value_plot, ()),
        (benchmark.make_agreement_plot, (1e-10,)),
        (benchmark.make_memory_plot, ()),
        (benchmark.make_summary_table, ()),
    ],
)
def test_plot_functions_reject_no_success(
    tmp_path: Path, plot_func: Any, extra_args: tuple[Any, ...]
) -> None:
    with pytest.raises(ValueError, match="No successful"):
        plot_func([{"status": "failed"}], tmp_path / "plot.png", *extra_args)


def test_make_plots_skips_when_too_few_successes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    benchmark.make_plots([success_result()], tmp_path, atol=1e-10)
    assert "Skipping plots" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_make_plots_creates_expected_files(
    tmp_path: Path, plot_results: list[dict[str, Any]]
) -> None:
    benchmark.make_plots(plot_results, tmp_path, atol=1e-10)

    expected = {
        "cross_vectorized_pdf_throughput_scaling.png",
        "cross_vectorized_pdf_time_per_value.png",
        "cross_vectorized_pdf_numerical_agreement.png",
        "cross_vectorized_pdf_memory.png",
        "cross_vectorized_pdf_summary_table.png",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})


def test_build_specs_order(tmp_path: Path) -> None:
    specs = benchmark.build_specs(
        frameworks=["pyhs3", "root"],
        scenarios=["normal", "poisson"],
        n_points=[10, 20],
        n_runs=3,
        rtol=1e-7,
        atol=1e-10,
        pyhs3_workspace_dir=tmp_path,
    )

    assert len(specs) == 8
    assert specs[0] == benchmark.BenchmarkSpec(
        "pyhs3", "normal", 10, 3, 1e-7, 1e-10, tmp_path
    )
    assert specs[-1] == benchmark.BenchmarkSpec(
        "root", "poisson", 20, 3, 1e-7, 1e-10, tmp_path
    )


def test_build_output_success_and_failure(tmp_path: Path) -> None:
    args = SimpleNamespace(
        frameworks=["pyhs3"],
        scenarios=["normal"],
        n_points=[10],
        n_runs=1,
        rtol=1e-7,
        atol=1e-10,
        timeout_seconds=2.0,
    )
    output = benchmark.build_output(
        [success_result(), {"status": "failed"}], args, tmp_path
    )

    assert output["status"] == "failed"
    assert output["n_results"] == 2
    assert output["n_successful"] == 1
    assert output["n_failed"] == 1
    assert output["pyhs3_workspace_dir"] == str(tmp_path)


def test_print_summary(capsys: pytest.CaptureFixture[str]) -> None:
    benchmark.print_summary({"status": "success", "n_successful": 1, "n_results": 1})
    output = capsys.readouterr().out
    assert benchmark.BENCHMARK_TITLE in output
    assert "Successful:  1 / 1" in output


def test_parse_args_defaults_and_custom_values(tmp_path: Path) -> None:
    defaults = benchmark.parse_args([])
    assert defaults.frameworks == benchmark.DEFAULT_FRAMEWORKS
    assert defaults.scenarios == benchmark.DEFAULT_SCENARIOS

    custom = benchmark.parse_args(
        [
            "--frameworks",
            "pyhs3",
            "root",
            "--scenarios",
            "normal",
            "--n-points",
            "3",
            "4",
            "--n-runs",
            "2",
            "--pyhs3-workspace-dir",
            str(tmp_path / "workspaces"),
            "--rtol",
            "1e-5",
            "--atol",
            "1e-8",
            "--timeout-seconds",
            "7",
            "--output-dir",
            str(tmp_path / "out"),
            "--output-name",
            "result.json",
            "--plot",
            "--plot-dir",
            str(tmp_path / "plots"),
            "--fail-fast",
        ]
    )
    assert custom.frameworks == ["pyhs3", "root"]
    assert custom.n_points == [3, 4]
    assert custom.n_runs == 2
    assert custom.plot is True
    assert custom.fail_fast is True


def test_parse_args_rejects_unknown_choice() -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--frameworks", "bad"])


def test_run_success_and_plot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plot_results: list[dict[str, Any]],
) -> None:
    workspace_dir = tmp_path / "workspaces"
    workspace_dir.mkdir()
    output_dir = tmp_path / "out"
    plot_dir = tmp_path / "plots"
    returned = iter(plot_results[:2])
    calls: list[benchmark.BenchmarkSpec] = []

    def fake_run_with_timeout(
        spec: benchmark.BenchmarkSpec, timeout_seconds: float
    ) -> dict[str, Any]:
        calls.append(spec)
        return next(returned)

    plot_calls: list[Any] = []
    monkeypatch.setattr(benchmark, "run_with_timeout", fake_run_with_timeout)
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda *args, **kwargs: plot_calls.append((args, kwargs)),
    )

    output = benchmark.run(
        frameworks=["pyhs3", "root"],
        scenarios=["normal"],
        n_points=[10],
        n_runs=1,
        pyhs3_workspace_dir=workspace_dir,
        rtol=1e-7,
        atol=1e-10,
        timeout_seconds=3.0,
        output_dir=output_dir,
        output_name="result.json",
        plot=True,
        plot_dir=plot_dir,
    )

    assert output["status"] == "success"
    assert output["n_successful"] == 2
    assert len(calls) == 2
    assert (output_dir / "result.json").exists()
    assert plot_calls


def test_run_fail_fast_stops_after_first_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace_dir = tmp_path / "workspaces"
    workspace_dir.mkdir()
    calls: list[benchmark.BenchmarkSpec] = []

    def fake_run_with_timeout(
        spec: benchmark.BenchmarkSpec, timeout_seconds: float
    ) -> dict[str, Any]:
        calls.append(spec)
        return {
            "benchmark": benchmark.BENCHMARK_NAME,
            "framework": spec.framework,
            "framework_label": benchmark._framework_label(spec.framework),
            "scenario": spec.scenario,
            "scenario_label": benchmark._scenario_label(spec.scenario),
            "n_points": spec.n_points,
            "n_runs": spec.n_runs,
            "status": "failed",
            "validation_status": "failed",
            "error_type": "RuntimeError",
            "error_message": "boom",
        }

    monkeypatch.setattr(benchmark, "run_with_timeout", fake_run_with_timeout)

    output = benchmark.run(
        frameworks=["numba_stats", "root"],
        scenarios=["normal"],
        n_points=[10, 20],
        n_runs=1,
        pyhs3_workspace_dir=workspace_dir,
        rtol=1e-7,
        atol=1e-10,
        timeout_seconds=3.0,
        output_dir=tmp_path / "out",
        output_name="result.json",
        plot=False,
        plot_dir=tmp_path / "plots",
        fail_fast=True,
    )

    assert output["status"] == "failed"
    assert len(calls) == 1
    assert output["n_results"] == 1


def test_main_passes_cli_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        benchmark,
        "run",
        lambda **kwargs: calls.append(kwargs),
    )

    benchmark.main(
        [
            "--frameworks",
            "numba_stats",
            "--scenarios",
            "normal",
            "--n-points",
            "5",
            "--n-runs",
            "2",
            "--pyhs3-workspace-dir",
            str(tmp_path / "workspaces"),
            "--output-dir",
            str(tmp_path / "out"),
            "--output-name",
            "result.json",
            "--plot",
            "--plot-dir",
            str(tmp_path / "plots"),
            "--fail-fast",
        ]
    )

    assert calls[0]["frameworks"] == ["numba_stats"]
    assert calls[0]["scenarios"] == ["normal"]
    assert calls[0]["n_points"] == [5]
    assert calls[0]["n_runs"] == 2
    assert calls[0]["plot"] is True
    assert calls[0]["fail_fast"] is True
