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
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "5ch_case.json"
    path.write_text("{}")
    return path


@pytest.fixture
def root_workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "5ch_case.root"
    path.write_text("root")
    return path


def make_config(
    workspace_path: Path, **overrides: Any
) -> benchmark.ScalarBenchmarkConfig:
    values = {
        "framework": "pyhs3",
        "workspace_path": workspace_path,
        "root_workspace_path": None,
        "target": "L_ch0",
        "mode": "FAST_RUN",
        "distribution": "sig_ch0",
        "n_evaluations": 3,
        "rtol": 1e-7,
        "atol": 1e-10,
        "reference_value": None,
    }
    values.update(overrides)
    return benchmark.ScalarBenchmarkConfig(**values)


def make_result(
    framework: str = "pyhs3",
    *,
    workspace: str = "5ch_case.json",
    status: str = "success",
    n_evaluations: int = 10,
) -> dict[str, Any]:
    if status != "success":
        return {
            "benchmark": benchmark.BENCHMARK_NAME,
            "framework": framework,
            "framework_label": benchmark._framework_label(framework),
            "workspace": workspace,
            "workspace_label": benchmark.workspace_label(Path(workspace)),
            "target": "L_ch0",
            "mode": "FAST_RUN",
            "distribution": "sig_ch0",
            "n_evaluations": n_evaluations,
            "status": status,
            "error_type": "RuntimeError",
            "error_message": "boom",
        }
    return {
        "benchmark": benchmark.BENCHMARK_NAME,
        "framework": framework,
        "framework_label": benchmark._framework_label(framework),
        "workspace": workspace,
        "workspace_path": workspace,
        "root_workspace_path": None,
        "workspace_label": benchmark.workspace_label(Path(workspace)),
        "target": "L_ch0",
        "mode": "FAST_RUN",
        "distribution": "sig_ch0",
        "n_evaluations": n_evaluations,
        "cold_start_time_seconds": 0.002,
        "total_runtime_seconds": 0.01,
        "average_runtime_seconds_per_evaluation": 0.001,
        "time_per_value_seconds": 0.001,
        "time_per_value_ns": 1_000_000.0,
        "throughput_evaluations_per_second": 1000.0,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 101.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 122.0,
        "peak_rss_delta_mb": 2.0,
        "first_timing_output": 0.5,
        "last_timing_output": 0.5,
        "outputs_stable": True,
        "max_repeated_abs_diff": 0.0,
        "status": "success",
        "reference_framework": benchmark.REFERENCE_FRAMEWORK,
        "reference_value": 0.5,
        "observed_value": 0.5,
        "n_values": 1,
        "n_finite_values": 1,
        "all_values_finite": True,
        "max_abs_diff": 0.0 if framework == "pyhs3" else 1e-12,
        "mean_abs_diff": 0.0 if framework == "pyhs3" else 1e-12,
        "max_rel_diff": 0.0 if framework == "pyhs3" else 2e-12,
        "mean_rel_diff": 0.0 if framework == "pyhs3" else 2e-12,
        "allclose_passed": True,
        "validation_status": "success",
    }


class FakeModel:
    def __init__(self, value: float = 0.5) -> None:
        self.data = {"x": 1.0}
        self.free_params = {"mu_sig": 1.0}
        self.value = value
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def pdf(self, distribution: str, **params: Any) -> np.ndarray:
        self.calls.append((distribution, params))
        return np.asarray([self.value])


class FakeWorkspaceLoader:
    model = FakeModel()

    @staticmethod
    def load(path: Path) -> Any:
        return SimpleNamespace(
            model=lambda target, progress, mode: FakeWorkspaceLoader.model
        )


class FakeRootObject:
    def __init__(self, name: str = "x", value: float = 0.0) -> None:
        self.name = name
        self.value = value

    def GetName(self) -> str:
        return self.name

    def setVal(self, value: float) -> None:
        self.value = float(value)


class FakeRootCollection:
    def __init__(self, objects: list[Any]) -> None:
        self.objects = objects

    def __iter__(self):
        return iter(self.objects)


class FakeIteratorOnlyCollection:
    def __init__(self, objects: list[Any]) -> None:
        self.objects = list(objects)

    def __iter__(self):
        raise TypeError

    def createIterator(self):
        values = list(self.objects)

        class Iterator:
            def Next(self_inner):
                return values.pop(0) if values else None

        return Iterator()


class FakeArgSet:
    def __init__(self, *objects: Any) -> None:
        self.objects: list[Any] = list(objects)

    def add(self, obj: Any) -> None:
        self.objects.append(obj)

    def getSize(self) -> int:
        return len(self.objects)

    def __iter__(self):
        return iter(self.objects)


class FakePdf:
    def __init__(
        self,
        name: str = "sig_ch0",
        value: float = 0.5,
        variables: list[str] | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.variables = variables if variables is not None else ["x", "mu_sig"]

    def GetName(self) -> str:
        return self.name

    def getVal(self, norm_set: Any) -> float:
        return self.value

    def getObservables(self, all_vars: Any) -> FakeRootCollection:
        return FakeRootCollection([FakeRootObject(name) for name in self.variables])


class FakeRootWorkspace:
    def __init__(self) -> None:
        self.variables = {
            "x": FakeRootObject("x"),
            "mu_sig": FakeRootObject("mu_sig"),
            "sigma_ch0": FakeRootObject("sigma_ch0"),
        }
        self.pdfs = {"sig_ch0": FakePdf("sig_ch0")}

    def var(self, name: str) -> Any:
        return self.variables.get(name)

    def pdf(self, name: str) -> Any:
        return self.pdfs.get(name)

    def allPdfs(self) -> Any:
        return FakeRootCollection([FakeRootObject(name) for name in self.pdfs])

    def allVars(self) -> Any:
        return FakeRootCollection([FakeRootObject(name) for name in self.variables])


class FakeRootFile:
    def __init__(self, workspace: Any | None = None, zombie: bool = False) -> None:
        self.workspace = workspace or FakeRootWorkspace()
        self.zombie = zombie
        self.closed = False

    def __bool__(self) -> bool:
        return True

    def IsZombie(self) -> bool:
        return self.zombie

    def Close(self) -> None:
        self.closed = True


class FakeRootModule:
    RooArgSet = FakeArgSet

    class RooWorkspace:
        @staticmethod
        def Class() -> object:
            return object()

    class TFile:
        opened_file = FakeRootFile()

        @staticmethod
        def Open(path: str, mode: str) -> FakeRootFile:
            return FakeRootModule.TFile.opened_file


def test_helpers_and_formatters() -> None:
    assert benchmark._framework_label("pyhs3") == "PyHS3 (eager)"
    assert benchmark._framework_label("unknown") == "unknown"
    assert benchmark._style_for("root")["label"] == "RooFit"
    assert benchmark._style_for("unknown")["marker"] == "o"
    assert benchmark._ordered_successful_results(
        [{"status": "failed"}, {"status": "success"}]
    ) == [{"status": "success"}]
    assert benchmark._safe_positive(0.0) == pytest.approx(1e-300)
    assert benchmark._safe_positive(float("nan")) == pytest.approx(1e-300)
    assert benchmark._safe_positive(2.0) == 2.0
    assert benchmark._format_seconds_ms(0.001) == "1.000 ms"
    assert benchmark._format_ns(999.0) == "999.0 ns"
    assert benchmark._format_ns(1000.0) == "1.00 µs"
    assert benchmark._format_scientific(0.0) == "0"
    assert benchmark._format_scientific(1e-5) == "1.0e-05"


def test_workspace_helpers() -> None:
    path = Path("inputs/5ch_case.json")
    assert benchmark.workspace_stem(path) == "5ch_case"
    assert benchmark.workspace_stem(Path("inputs/5ch_case.root")) == "5ch_case"
    assert benchmark.workspace_label(path) == "5ch\ncase"
    assert benchmark.workspace_title("5ch_case") == "5ch / case"
    assert benchmark.default_root_workspace_path(path) == Path("inputs/5ch_case.root")


def valid_config_kwargs(
    workspace_path: Path, root_workspace_path: Path | None = None, **overrides: Any
) -> dict[str, Any]:
    values = {
        "frameworks": ["pyhs3"],
        "workspaces": [workspace_path],
        "root_workspaces": None,
        "target": "L_ch0",
        "mode": "FAST_RUN",
        "distribution": "sig_ch0",
        "n_evaluations": [1, 10],
        "rtol": 1e-7,
        "atol": 1e-10,
        "timeout_seconds": 1.0,
    }
    if root_workspace_path is not None:
        values["frameworks"] = ["pyhs3", "root"]
        values["root_workspaces"] = [root_workspace_path]
    values.update(overrides)
    return values


def test_validate_benchmark_config_success(
    workspace_path: Path, root_workspace_path: Path
) -> None:
    benchmark.validate_benchmark_config(**valid_config_kwargs(workspace_path))
    benchmark.validate_benchmark_config(
        **valid_config_kwargs(workspace_path, root_workspace_path)
    )


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        (
            {"frameworks": []},
            benchmark.BenchmarkConfigurationError,
            "At least one framework",
        ),
        (
            {"workspaces": []},
            benchmark.BenchmarkConfigurationError,
            "At least one workspace",
        ),
        ({"target": ""}, benchmark.BenchmarkConfigurationError, "target"),
        ({"mode": ""}, benchmark.BenchmarkConfigurationError, "mode"),
        ({"distribution": ""}, benchmark.BenchmarkConfigurationError, "distribution"),
        (
            {"frameworks": ["bad"]},
            benchmark.BenchmarkConfigurationError,
            "Unknown framework",
        ),
        (
            {"n_evaluations": [0]},
            benchmark.BenchmarkConfigurationError,
            "n-evaluations",
        ),
        ({"rtol": -1.0}, benchmark.BenchmarkConfigurationError, "non-negative"),
        ({"atol": -1.0}, benchmark.BenchmarkConfigurationError, "non-negative"),
        ({"timeout_seconds": 0.0}, benchmark.BenchmarkConfigurationError, "timeout"),
    ],
)
def test_validate_benchmark_config_rejects_invalid(
    workspace_path: Path,
    overrides: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        benchmark.validate_benchmark_config(
            **valid_config_kwargs(workspace_path, **overrides)
        )


def test_validate_benchmark_config_rejects_paths(
    tmp_path: Path, workspace_path: Path
) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.validate_benchmark_config(
            **valid_config_kwargs(tmp_path / "missing.json")
        )
    with pytest.raises(FileNotFoundError, match="Workspace path is not a file"):
        benchmark.validate_benchmark_config(**valid_config_kwargs(tmp_path))
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="same number"):
        benchmark.validate_benchmark_config(
            **valid_config_kwargs(
                workspace_path, root_workspaces=[Path("a.root"), Path("b.root")]
            )
        )
    with pytest.raises(FileNotFoundError, match="ROOT workspace file does not exist"):
        benchmark.validate_benchmark_config(
            **valid_config_kwargs(
                workspace_path,
                frameworks=["root"],
                root_workspaces=[tmp_path / "missing.root"],
            )
        )
    root_dir = tmp_path / "case.root"
    root_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="ROOT workspace path is not a file"):
        benchmark.validate_benchmark_config(
            **valid_config_kwargs(
                workspace_path, frameworks=["root"], root_workspaces=[root_dir]
            )
        )


def test_pyhs3_default_parameters() -> None:
    model = SimpleNamespace(data={"x": [1, 2]}, free_params={"mu_sig": 1.0})
    params = benchmark._pyhs3_default_parameters(model)
    assert np.allclose(params["x"], [1, 2])
    assert params["mu_sig"] == pytest.approx(1.0)


def test_evaluate_pyhs3_success_and_nonfinite(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    FakeWorkspaceLoader.model = FakeModel(0.25)
    monkeypatch.setitem(
        sys.modules, "pyhs3.workspace", SimpleNamespace(Workspace=FakeWorkspaceLoader)
    )
    assert benchmark.evaluate_pyhs3(
        workspace_path, "L_ch0", "FAST_RUN", "sig_ch0"
    ) == pytest.approx(0.25)
    assert FakeWorkspaceLoader.model.calls[-1][0] == "sig_ch0"

    FakeWorkspaceLoader.model = FakeModel(float("nan"))
    with pytest.raises(benchmark.ValidationFailure, match="non-finite"):
        benchmark.evaluate_pyhs3(workspace_path, "L_ch0", "FAST_RUN", "sig_ch0")


def test_root_collection_names_iteration_and_iterator() -> None:
    assert benchmark._root_collection_names(
        FakeRootCollection([FakeRootObject("a"), None, FakeRootObject("b")])
    ) == ["a", "b"]
    assert benchmark._root_collection_names(
        FakeIteratorOnlyCollection([FakeRootObject("x"), FakeRootObject("y")])
    ) == ["x", "y"]
    assert benchmark._root_argset_names(FakeRootCollection([FakeRootObject("z")])) == [
        "z"
    ]


def test_root_helpers_and_norm_set(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    workspace = FakeRootWorkspace()
    assert benchmark._available_root_pdfs(workspace) == ["sig_ch0"]
    assert set(benchmark._available_root_vars(workspace)) == {
        "x",
        "mu_sig",
        "sigma_ch0",
    }
    assert benchmark._is_root_observable_name("x")
    assert benchmark._is_root_observable_name("x_ch0")
    assert not benchmark._is_root_observable_name("sigma_ch0")
    argset = benchmark._make_root_argset(workspace, ["x", "missing"])
    assert argset.getSize() == 1
    norm_set = benchmark._root_norm_set_for_pdf(workspace, workspace.pdfs["sig_ch0"])
    assert benchmark._root_argset_names(norm_set) == ["x"]


def test_root_norm_set_raises_when_no_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    workspace = FakeRootWorkspace()
    workspace.variables = {"mu_sig": FakeRootObject("mu_sig")}
    pdf = FakePdf(variables=["mu_sig"])
    with pytest.raises(KeyError, match="normalization observable"):
        benchmark._root_norm_set_for_pdf(workspace, pdf)


def test_set_root_defaults_from_pyhs3_best_effort(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    workspace = FakeRootWorkspace()
    FakeWorkspaceLoader.model = FakeModel(0.5)
    monkeypatch.setitem(
        sys.modules, "pyhs3.workspace", SimpleNamespace(Workspace=FakeWorkspaceLoader)
    )
    benchmark._set_root_defaults_from_pyhs3(
        workspace, workspace_path, "L_ch0", "FAST_RUN"
    )
    assert workspace.variables["mu_sig"].value == pytest.approx(1.0)


def test_evaluate_root_success_and_errors(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, root_workspace_path: Path
) -> None:
    workspace = FakeRootWorkspace()
    root_file = FakeRootFile(workspace)
    FakeRootModule.TFile.opened_file = root_file
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    monkeypatch.setattr(
        benchmark, "_find_root_workspace", lambda root_file_arg: workspace
    )
    monkeypatch.setattr(
        benchmark, "_set_root_defaults_from_pyhs3", lambda **kwargs: None
    )
    assert benchmark.evaluate_root(
        root_workspace_path, workspace_path, "L_ch0", "FAST_RUN", "sig_ch0"
    ) == pytest.approx(0.5)
    assert root_file.closed is True

    FakeRootModule.TFile.opened_file = FakeRootFile(workspace, zombie=True)
    with pytest.raises(FileNotFoundError, match="Could not open"):
        benchmark.evaluate_root(
            root_workspace_path, workspace_path, "L_ch0", "FAST_RUN", "sig_ch0"
        )

    root_file = FakeRootFile(workspace)
    FakeRootModule.TFile.opened_file = root_file
    with pytest.raises(KeyError, match="Available PDFs"):
        benchmark.evaluate_root(
            root_workspace_path, workspace_path, "L_ch0", "FAST_RUN", "missing"
        )

    workspace.pdfs["sig_ch0"] = FakePdf(value=float("inf"))
    with pytest.raises(benchmark.ValidationFailure, match="non-finite"):
        benchmark.evaluate_root(
            root_workspace_path, workspace_path, "L_ch0", "FAST_RUN", "sig_ch0"
        )


def test_evaluate_framework_once_dispatch(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, root_workspace_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        benchmark, "evaluate_pyhs3", lambda *args: calls.append("pyhs3") or 0.5
    )
    monkeypatch.setattr(
        benchmark, "evaluate_root", lambda *args: calls.append("root") or 0.5
    )
    assert (
        benchmark.evaluate_framework_once(
            framework="pyhs3",
            workspace_path=workspace_path,
            root_workspace_path=None,
            target="L_ch0",
            mode="FAST_RUN",
            distribution="sig_ch0",
        )
        == 0.5
    )
    assert (
        benchmark.evaluate_framework_once(
            framework="root",
            workspace_path=workspace_path,
            root_workspace_path=root_workspace_path,
            target="L_ch0",
            mode="FAST_RUN",
            distribution="sig_ch0",
        )
        == 0.5
    )
    assert calls == ["pyhs3", "root"]
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="requires"):
        benchmark.evaluate_framework_once(
            framework="root",
            workspace_path=workspace_path,
            root_workspace_path=None,
            target="L_ch0",
            mode="FAST_RUN",
            distribution="sig_ch0",
        )
    with pytest.raises(ValueError, match="Unknown framework"):
        benchmark.evaluate_framework_once(
            framework="bad",
            workspace_path=workspace_path,
            root_workspace_path=None,
            target="L_ch0",
            mode="FAST_RUN",
            distribution="sig_ch0",
        )


def test_compute_agreement_success_mismatch_and_invalid() -> None:
    ok = benchmark.compute_agreement(0.5, 0.5 + 1e-12, rtol=1e-7, atol=1e-10)
    assert ok["validation_status"] == "success"
    assert ok["allclose_passed"] is True
    mismatch = benchmark.compute_agreement(0.6, 0.5, rtol=0.0, atol=0.0)
    assert mismatch["validation_status"] == "mismatch"
    assert mismatch["allclose_passed"] is False
    with pytest.raises(benchmark.ValidationFailure, match="non-finite"):
        benchmark.compute_agreement(float("nan"), 0.5, 1e-7, 1e-10)
    with pytest.raises(benchmark.ValidationFailure, match="Reference"):
        benchmark.compute_agreement(0.5, float("inf"), 1e-7, 1e-10)


def test_prepare_evaluators(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, root_workspace_path: Path
) -> None:
    FakeWorkspaceLoader.model = FakeModel(0.3)
    monkeypatch.setitem(
        sys.modules, "pyhs3.workspace", SimpleNamespace(Workspace=FakeWorkspaceLoader)
    )
    evaluate, cleanup = benchmark._prepare_pyhs3_evaluator(
        workspace_path, "L_ch0", "FAST_RUN", "sig_ch0"
    )
    assert cleanup is None
    assert evaluate() == pytest.approx(0.3)

    root_file = FakeRootFile(FakeRootWorkspace())
    FakeRootModule.TFile.opened_file = root_file
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    monkeypatch.setattr(
        benchmark, "_find_root_workspace", lambda root_file_arg: root_file.workspace
    )
    monkeypatch.setattr(
        benchmark, "_set_root_defaults_from_pyhs3", lambda **kwargs: None
    )
    evaluate, cleanup = benchmark._prepare_root_evaluator(
        root_workspace_path, workspace_path, "L_ch0", "FAST_RUN", "sig_ch0"
    )
    assert evaluate() == pytest.approx(0.5)
    assert cleanup is not None
    cleanup()
    assert root_file.closed is True


def test_prepare_evaluator_dispatch_and_errors(
    workspace_path: Path, root_workspace_path: Path
) -> None:
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="requires"):
        benchmark._prepare_evaluator(make_config(workspace_path, framework="root"))
    with pytest.raises(ValueError, match="Unknown framework"):
        benchmark._prepare_evaluator(make_config(workspace_path, framework="bad"))


def test_run_single_framework_benchmark_success_and_mismatch(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 120.0)
    times = iter([1.0, 1.2, 2.0, 2.5])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(benchmark, "evaluate_pyhs3", lambda *args: 0.5)
    monkeypatch.setattr(
        benchmark, "_prepare_evaluator", lambda config: (lambda: 0.5, None)
    )
    result = benchmark.run_single_framework_benchmark(
        make_config(workspace_path, n_evaluations=3)
    )
    assert result["status"] == "success"
    assert result["validation_status"] == "success"
    assert result["throughput_evaluations_per_second"] == pytest.approx(6.0)
    assert result["workspace"] == workspace_path.name

    times = iter([1.0, 1.1, 2.0, 2.2])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(
        benchmark, "_prepare_evaluator", lambda config: (lambda: 0.7, None)
    )
    mismatch = benchmark.run_single_framework_benchmark(
        make_config(workspace_path, reference_value=0.5, rtol=0.0, atol=0.0)
    )
    assert mismatch["validation_status"] == "mismatch"
    assert mismatch["allclose_passed"] is False


def test_run_single_framework_benchmark_cleanup_on_error(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    closed: list[bool] = []
    monkeypatch.setattr(benchmark, "evaluate_pyhs3", lambda *args: 0.5)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 120.0)

    def prepare(config: Any):
        def evaluate() -> float:
            raise RuntimeError("boom")

        return evaluate, lambda: closed.append(True)

    monkeypatch.setattr(benchmark, "_prepare_evaluator", prepare)
    with pytest.raises(RuntimeError, match="boom"):
        benchmark.run_single_framework_benchmark(
            make_config(workspace_path, reference_value=0.5)
        )
    assert closed == [True]


def test_error_result_and_config_from_payload(
    workspace_path: Path, root_workspace_path: Path
) -> None:
    config = make_config(
        workspace_path,
        framework="root",
        root_workspace_path=root_workspace_path,
        reference_value=0.5,
    )
    result = benchmark._error_result(config, "timeout", timeout_seconds=1.0)
    assert result["status"] == "timeout"
    assert result["workspace"] == workspace_path.name
    assert result["root_workspace_path"] == str(root_workspace_path)
    payload = {
        "framework": "root",
        "workspace_path": str(workspace_path),
        "root_workspace_path": str(root_workspace_path),
        "target": "L_ch0",
        "mode": "FAST_RUN",
        "distribution": "sig_ch0",
        "n_evaluations": 2,
        "rtol": 1e-7,
        "atol": 1e-10,
        "reference_value": 0.5,
    }
    parsed = benchmark._config_from_payload(payload)
    assert parsed.framework == "root"
    assert parsed.workspace_path == workspace_path
    assert parsed.root_workspace_path == root_workspace_path


def test_run_worker_success_and_error(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    class Queue:
        def __init__(self) -> None:
            self.items: list[Any] = []

        def put(self, item: Any) -> None:
            self.items.append(item)

    payload = benchmark.build_payload(
        framework="pyhs3",
        workspace_path=workspace_path,
        root_workspace_path=None,
        target="L_ch0",
        mode="FAST_RUN",
        distribution="sig_ch0",
        n_evaluations=1,
        rtol=1e-7,
        atol=1e-10,
        reference_value=0.5,
    )
    q = Queue()
    monkeypatch.setattr(
        benchmark,
        "run_single_framework_benchmark",
        lambda config: {"status": "success"},
    )
    benchmark.run_worker(payload, q)  # type: ignore[arg-type]
    assert q.items == [{"status": "success"}]

    q = Queue()
    monkeypatch.setattr(
        benchmark,
        "run_single_framework_benchmark",
        lambda config: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    benchmark.run_worker(payload, q)  # type: ignore[arg-type]
    assert q.items[0]["status"] == "error"
    assert q.items[0]["error_type"] == "RuntimeError"


def test_run_with_timeout_success_timeout_exit_and_empty(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    payload = benchmark.build_payload(
        framework="pyhs3",
        workspace_path=workspace_path,
        root_workspace_path=None,
        target="L_ch0",
        mode="FAST_RUN",
        distribution="sig_ch0",
        n_evaluations=1,
        rtol=1e-7,
        atol=1e-10,
        reference_value=0.5,
    )

    class GoodQueue:
        def __init__(self, maxsize: int = 1) -> None:
            self.item = {"status": "success"}

        def get_nowait(self) -> dict[str, Any]:
            return self.item

    class EmptyQueue:
        def __init__(self, maxsize: int = 1) -> None:
            pass

        def get_nowait(self) -> Any:
            raise queue.Empty

    class Process:
        exitcode = 0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.alive = False

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.alive = False

        def kill(self) -> None:
            self.alive = False

    class AliveProcess(Process):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calls = 0

        def is_alive(self) -> bool:
            self.calls += 1
            return self.calls == 1

    class ExitProcess(Process):
        exitcode = 7

    class Ctx:
        pass

    Ctx.Queue = GoodQueue
    Ctx.Process = Process

    monkeypatch.setattr(benchmark.mp, "get_context", lambda method: Ctx())
    assert benchmark.run_with_timeout(payload, 1.0)["status"] == "success"
    Ctx.Process = AliveProcess
    Ctx.Queue = EmptyQueue
    assert benchmark.run_with_timeout(payload, 0.01)["status"] == "timeout"
    Ctx.Process = ExitProcess
    result = benchmark.run_with_timeout(payload, 0.01)
    assert result["status"] == "error"
    assert result["error_type"] == "ProcessExitError"
    Ctx.Process = Process
    result = benchmark.run_with_timeout(payload, 0.01)
    assert result["status"] == "error"
    assert result["error_type"] == "EmptyWorkerResult"


def test_print_and_summary(capsys: pytest.CaptureFixture[str]) -> None:
    success = make_result("pyhs3")
    benchmark.print_result(success)
    out = capsys.readouterr().out
    assert "5ch_case.json" in out
    assert "throughput" in out
    failed = make_result("root", status="error")
    benchmark.print_result(failed)
    out = capsys.readouterr().out
    assert "validation:              unavailable" in out
    summary = benchmark.summarize_status([success, failed])
    assert summary["status"] == "completed_with_errors"
    assert summary["n_successful"] == 1
    assert summary["unsuccessful_results"][0]["framework"] == "root"
    benchmark.print_final_summary([success, failed])
    out = capsys.readouterr().out
    assert "Unsuccessful" in out


def test_success_dataframe_and_plots(tmp_path: Path) -> None:
    results = [
        make_result("pyhs3", n_evaluations=1),
        make_result("root", n_evaluations=10),
        make_result("pyhs3", workspace="10ch_case.json", n_evaluations=1),
    ]
    df = benchmark._success_dataframe(results + [{"status": "failed"}])
    assert list(df["framework_key"])[:2] == ["pyhs3", "root"]
    plotters = [
        benchmark.make_throughput_plot,
        benchmark.make_latency_plot,
        benchmark.make_time_per_value_plot,
        benchmark.make_memory_plot,
        benchmark.make_summary_table,
    ]
    for plotter in plotters:
        output = tmp_path / f"{plotter.__name__}.png"
        plotter(results, output)
        assert output.exists()
    agreement = tmp_path / "agreement.png"
    benchmark.make_agreement_plot(results, agreement, tolerance=1e-10)
    assert agreement.exists()
    benchmark.make_plots(results, tmp_path / "all", tolerance=1e-10)
    assert (tmp_path / "all" / "cross_scalar_pdf_summary_table.png").exists()


def test_plot_functions_return_on_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for plotter in [
        benchmark.make_throughput_plot,
        benchmark.make_latency_plot,
        benchmark.make_time_per_value_plot,
        benchmark.make_memory_plot,
        benchmark.make_summary_table,
    ]:
        output = tmp_path / f"{plotter.__name__}.png"
        plotter([{"status": "failed"}], output)
        assert not output.exists()
    output = tmp_path / "agreement.png"
    benchmark.make_agreement_plot([{"status": "failed"}], output, tolerance=1e-10)
    assert not output.exists()
    benchmark.make_plots([make_result("pyhs3")], tmp_path / "plots", tolerance=1e-10)
    assert "at least two successful" in capsys.readouterr().out


def test_parse_args_defaults_and_custom(
    workspace_path: Path, root_workspace_path: Path
) -> None:
    args = benchmark.parse_args([])
    assert args.frameworks == benchmark.DEFAULT_FRAMEWORKS
    assert args.workspaces == benchmark.DEFAULT_WORKSPACES
    assert args.n_evaluations == benchmark.DEFAULT_N_EVALUATIONS
    args = benchmark.parse_args(
        [
            "--frameworks",
            "pyhs3",
            "root",
            "--workspaces",
            str(workspace_path),
            "--root-workspaces",
            str(root_workspace_path),
            "--target",
            "A",
            "--mode",
            "FAST_COMPILE",
            "--distribution",
            "pdf",
            "--n-evaluations",
            "1",
            "2",
            "--rtol",
            "0.1",
            "--atol",
            "0.2",
            "--timeout-seconds",
            "3",
            "--output-dir",
            "out",
            "--output-name",
            "x.json",
            "--plot",
            "--plot-dir",
            "plots",
        ]
    )
    assert args.frameworks == ["pyhs3", "root"]
    assert args.workspaces == [workspace_path]
    assert args.root_workspaces == [root_workspace_path]
    assert args.target == "A"
    assert args.mode == "FAST_COMPILE"
    assert args.distribution == "pdf"
    assert args.n_evaluations == [1, 2]
    assert args.rtol == pytest.approx(0.1)
    assert args.atol == pytest.approx(0.2)
    assert args.plot is True


def test_build_payload(workspace_path: Path, root_workspace_path: Path) -> None:
    payload = benchmark.build_payload(
        framework="root",
        workspace_path=workspace_path,
        root_workspace_path=root_workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        distribution="sig_ch0",
        n_evaluations=2,
        rtol=1e-7,
        atol=1e-10,
        reference_value=0.5,
    )
    assert payload["framework"] == "root"
    assert payload["workspace_path"] == str(workspace_path)
    assert payload["root_workspace_path"] == str(root_workspace_path)
    assert payload["reference_value"] == 0.5


def test_run_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    workspace_path: Path,
    root_workspace_path: Path,
) -> None:
    saved: list[tuple[dict[str, Any], Path]] = []
    monkeypatch.setattr(benchmark, "evaluate_pyhs3", lambda *args: 0.5)
    monkeypatch.setattr(
        benchmark,
        "run_with_timeout",
        lambda payload, timeout: make_result(
            payload["framework"],
            workspace=Path(payload["workspace_path"]).name,
            n_evaluations=payload["n_evaluations"],
        ),
    )
    monkeypatch.setattr(benchmark, "print_result", lambda result: None)
    monkeypatch.setattr(benchmark, "print_final_summary", lambda results: None)
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda payload, path: (
            saved.append((payload, path)),
            path.parent.mkdir(parents=True, exist_ok=True),
            path.write_text(json.dumps(payload)),
        ),
    )
    plot_calls: list[Any] = []
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda results, plot_dir, tolerance: plot_calls.append(
            (results, plot_dir, tolerance)
        ),
    )
    output = benchmark.run(
        frameworks=["pyhs3", "root"],
        workspaces=[workspace_path],
        root_workspaces=[root_workspace_path],
        target="L_ch0",
        mode="FAST_RUN",
        distribution="sig_ch0",
        n_evaluations=[1],
        rtol=1e-7,
        atol=1e-10,
        timeout_seconds=1.0,
        output_dir=tmp_path,
        output_name="out.json",
        plot=True,
        plot_dir=tmp_path / "plots",
    )
    assert output["benchmark"] == benchmark.BENCHMARK_NAME
    assert output["summary"]["n_successful"] == 2
    assert len(output["results"]) == 2
    assert saved[0][1] == tmp_path / "out.json"
    assert plot_calls


def test_main_passes_cli_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workspace_path: Path
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"benchmark": benchmark.BENCHMARK_NAME}

    monkeypatch.setattr(benchmark, "run", fake_run)
    benchmark.main(
        [
            "--frameworks",
            "pyhs3",
            "--workspaces",
            str(workspace_path),
            "--output-dir",
            str(tmp_path),
            "--n-evaluations",
            "1",
        ]
    )
    assert calls[0]["frameworks"] == ["pyhs3"]
    assert calls[0]["workspaces"] == [workspace_path]
    assert calls[0]["root_workspaces"] is None
    assert calls[0]["output_dir"] == tmp_path


def test_main_wraps_errors(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    monkeypatch.setattr(
        benchmark, "run", lambda **kwargs: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(RuntimeError, match="did not complete"):
        benchmark.main(["--frameworks", "pyhs3", "--workspaces", str(workspace_path)])


class FakeRootKey:
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def ReadObj(self) -> Any:
        return self.obj


class FakeWorkspaceObj:
    def __init__(self, inherits: bool = True) -> None:
        self.inherits = inherits

    def InheritsFrom(self, cls: Any) -> bool:
        return self.inherits


def test_find_root_workspace_success_and_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    good = FakeWorkspaceObj(True)
    root_file = SimpleNamespace(
        GetListOfKeys=lambda: [FakeRootKey(FakeWorkspaceObj(False)), FakeRootKey(good)]
    )
    assert benchmark._find_root_workspace(root_file) is good

    empty_file = SimpleNamespace(
        GetListOfKeys=lambda: [FakeRootKey(FakeWorkspaceObj(False))]
    )
    with pytest.raises(KeyError, match="No RooWorkspace"):
        benchmark._find_root_workspace(empty_file)


def test_root_norm_set_falls_back_when_get_observables_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)

    class BadPdf(FakePdf):
        def getObservables(self, all_vars: Any) -> Any:
            raise RuntimeError("cannot inspect")

    workspace = FakeRootWorkspace()
    norm_set = benchmark._root_norm_set_for_pdf(workspace, BadPdf())
    assert norm_set.getSize() == 1
    assert norm_set.objects[0].GetName() == "x"


def test_set_root_defaults_best_effort_branches(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    # Import/load failures are intentionally ignored.
    monkeypatch.setitem(sys.modules, "pyhs3.workspace", SimpleNamespace())
    benchmark._set_root_defaults_from_pyhs3(
        FakeRootWorkspace(), workspace_path, "L_ch0", "FAST_RUN"
    )

    class BadSetVal(FakeRootObject):
        def setVal(self, value: float) -> None:
            raise RuntimeError("cannot set")

    workspace = FakeRootWorkspace()
    workspace.variables["bad"] = BadSetVal("bad")

    class Model:
        data = {"missing": 1.0, "bad": 2.0}
        free_params = {"mu_sig": 3.0}

    class WorkspaceLoader:
        @staticmethod
        def load(path: Path) -> Any:
            return SimpleNamespace(model=lambda target, progress, mode: Model())

    monkeypatch.setitem(
        sys.modules, "pyhs3.workspace", SimpleNamespace(Workspace=WorkspaceLoader)
    )
    benchmark._set_root_defaults_from_pyhs3(
        workspace, workspace_path, "L_ch0", "FAST_RUN"
    )
    assert workspace.variables["mu_sig"].value == pytest.approx(3.0)


def test_prepare_pyhs3_evaluator_rejects_nonfinite(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    class BadModel(FakeModel):
        def pdf(self, distribution: str, **params: Any) -> np.ndarray:
            return np.asarray([float("nan")])

    class Loader:
        @staticmethod
        def load(path: Path) -> Any:
            return SimpleNamespace(model=lambda target, progress, mode: BadModel())

    monkeypatch.setitem(
        sys.modules, "pyhs3.workspace", SimpleNamespace(Workspace=Loader)
    )
    evaluator, cleanup = benchmark._prepare_pyhs3_evaluator(
        workspace_path, "L_ch0", "FAST_RUN", "sig_ch0"
    )
    assert cleanup is None
    with pytest.raises(benchmark.ValidationFailure, match="non-finite"):
        evaluator()


def test_prepare_root_evaluator_error_branches(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, root_workspace_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)

    FakeRootModule.TFile.opened_file = FakeRootFile(zombie=True)
    with pytest.raises(FileNotFoundError, match="Could not open ROOT file"):
        benchmark._prepare_root_evaluator(
            root_workspace_path, workspace_path, "L_ch0", "FAST_RUN", "sig_ch0"
        )

    workspace = FakeRootWorkspace()
    workspace.pdfs = {}
    root_file = FakeRootFile(workspace)
    FakeRootModule.TFile.opened_file = root_file
    monkeypatch.setattr(
        benchmark, "_find_root_workspace", lambda root_file_arg: workspace
    )
    with pytest.raises(KeyError, match="Available PDFs"):
        benchmark._prepare_root_evaluator(
            root_workspace_path, workspace_path, "L_ch0", "FAST_RUN", "missing"
        )
    assert root_file.closed is True

    workspace = FakeRootWorkspace()
    root_file = FakeRootFile(workspace)
    FakeRootModule.TFile.opened_file = root_file
    monkeypatch.setattr(
        benchmark, "_find_root_workspace", lambda root_file_arg: workspace
    )
    monkeypatch.setattr(
        benchmark, "_root_norm_set_for_pdf", lambda ws, pdf: FakeArgSet()
    )
    with pytest.raises(KeyError, match="normalization observables"):
        benchmark._prepare_root_evaluator(
            root_workspace_path, workspace_path, "L_ch0", "FAST_RUN", "sig_ch0"
        )


def test_prepare_root_evaluator_runtime_nonfinite(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, root_workspace_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    workspace = FakeRootWorkspace()
    workspace.pdfs["sig_ch0"] = FakePdf("sig_ch0", value=float("inf"))
    root_file = FakeRootFile(workspace)
    FakeRootModule.TFile.opened_file = root_file
    monkeypatch.setattr(
        benchmark, "_find_root_workspace", lambda root_file_arg: workspace
    )
    evaluator, cleanup = benchmark._prepare_root_evaluator(
        root_workspace_path, workspace_path, "L_ch0", "FAST_RUN", "sig_ch0"
    )
    with pytest.raises(benchmark.ValidationFailure, match="non-finite"):
        evaluator()
    cleanup()
    assert root_file.closed is True


def test_prepare_evaluator_dispatch_branches(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, root_workspace_path: Path
) -> None:
    pyhs3_called = []
    root_called = []
    monkeypatch.setattr(
        benchmark,
        "_prepare_pyhs3_evaluator",
        lambda **kwargs: pyhs3_called.append(kwargs) or (lambda: 0.5, None),
    )
    monkeypatch.setattr(
        benchmark,
        "_prepare_root_evaluator",
        lambda **kwargs: root_called.append(kwargs) or (lambda: 0.5, lambda: None),
    )

    benchmark._prepare_evaluator(make_config(workspace_path, framework="pyhs3"))
    assert pyhs3_called

    with pytest.raises(benchmark.BenchmarkConfigurationError, match="requires"):
        benchmark._prepare_evaluator(
            make_config(workspace_path, framework="root", root_workspace_path=None)
        )

    benchmark._prepare_evaluator(
        make_config(
            workspace_path, framework="root", root_workspace_path=root_workspace_path
        )
    )
    assert root_called

    with pytest.raises(ValueError, match="Unknown framework"):
        benchmark._prepare_evaluator(make_config(workspace_path, framework="bad"))


def test_run_with_timeout_kills_stubborn_process(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    payload = benchmark.build_payload(
        framework="pyhs3",
        workspace_path=workspace_path,
        root_workspace_path=None,
        target="L_ch0",
        mode="FAST_RUN",
        distribution="sig_ch0",
        n_evaluations=1,
        rtol=1e-7,
        atol=1e-10,
        reference_value=0.5,
    )

    class FakeQueue:
        def __init__(self, maxsize: int = 1) -> None:
            pass

    class StubbornProcess:
        exitcode = 0

        def __init__(self, target: Any, args: tuple[Any, ...]) -> None:
            self.kill_called = False
            self.alive_checks = 0

        def start(self) -> None:
            pass

        def join(self, timeout: float | None = None) -> None:
            pass

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            self.kill_called = True

        def is_alive(self) -> bool:
            self.alive_checks += 1
            return self.alive_checks <= 2

    class Context:
        Queue = FakeQueue
        Process = StubbornProcess

    monkeypatch.setattr(benchmark.mp, "get_context", lambda method: Context())
    result = benchmark.run_with_timeout(payload, 0.001)
    assert result["status"] == "timeout"
    assert result["timeout_seconds"] == pytest.approx(0.001)


def test_make_summary_table_hits_repeated_workspace_white_row(tmp_path: Path) -> None:
    # Three rows for the same workspace make the repeated empty workspace cell land
    # on both alternating-row branches in the table styling loop.
    results = [
        make_result("pyhs3", workspace="same.json", n_evaluations=1),
        make_result("root", workspace="same.json", n_evaluations=10),
        make_result("pyhs3", workspace="same.json", n_evaluations=100),
    ]
    output = tmp_path / "summary.png"
    benchmark.make_summary_table(results, output)
    assert output.exists()
