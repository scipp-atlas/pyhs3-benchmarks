from __future__ import annotations

import math
import queue
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from src import run_cross_nll_scan as benchmark


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "5ch_case.json"
    path.write_text("{}")
    return path


@pytest.fixture
def config(workspace_path: Path) -> benchmark.Config:
    return benchmark.Config(
        engine=benchmark.PYHS3_NONCOMPILED,
        category=benchmark.POINTWISE_NLL,
        workspace_path=workspace_path,
        root_workspace_path=workspace_path.with_suffix(".root"),
        analysis="L_ch0",
        distribution="model_ch0",
        data_name="combData_ch0",
        observable_name="x",
        observable_index=0,
        poi="mu_sig",
        mode="FAST_RUN",
        mu_values=np.asarray([0.0, 1.0, 2.0]),
        batch_size=2,
        n_batches=2,
        warmup_batches=1,
        scan_repeats=2,
        rtol=1e-7,
        atol=1e-7,
    )


class FakeModel:
    def __init__(self) -> None:
        self.data = {"x": np.asarray([0.0, 1.0])}
        self.free_params = {"mu_sig": np.asarray(1.0), "tau_ch0": np.asarray(0.2)}
        self.distributions = {"model_ch0": object()}

    def pdf(self, _distribution: str, **params: Any) -> np.ndarray:
        x = np.asarray(params["x"], dtype=float)
        mu = float(np.asarray(params["mu_sig"]).reshape(-1)[0])
        return np.exp(-0.5 * (x - mu) ** 2) + 0.1


class FakeWorkspace:
    model_object = FakeModel()
    data = SimpleNamespace(
        root=[SimpleNamespace(name="combData_ch0", entries=[[0.0], [1.0], [2.0]])]
    )

    @classmethod
    def load(cls, _path: Path) -> "FakeWorkspace":
        return cls()

    def model(self, *_args: Any, **_kwargs: Any) -> FakeModel:
        return self.model_object


class FakeVar:
    def __init__(
        self, minimum: float = -10.0, maximum: float = 10.0, truthy: bool = True
    ) -> None:
        self.minimum = minimum
        self.maximum = maximum
        self.truthy = truthy
        self.value = 0.0
        self.raise_set = False

    def __bool__(self) -> bool:
        return self.truthy

    def getMin(self) -> float:
        return self.minimum

    def getMax(self) -> float:
        return self.maximum

    def setVal(self, value: float) -> None:
        if self.raise_set:
            raise ValueError("cannot set")
        self.value = float(value)


class FakePdf:
    def __init__(self, observable: FakeVar, poi: FakeVar) -> None:
        self.observable = observable
        self.poi = poi

    def __bool__(self) -> bool:
        return True

    def getVal(self, _norm: Any) -> float:
        return math.exp(-0.5 * (self.observable.value - self.poi.value) ** 2) + 0.1


class FakeRootWorkspace:
    def __init__(self) -> None:
        self.vars = {
            "x": FakeVar(),
            "mu_sig": FakeVar(),
            "tau_ch0": FakeVar(-1.0, -0.01),
        }
        self.pdf_obj = FakePdf(self.vars["x"], self.vars["mu_sig"])

    def var(self, name: str) -> Any:
        return self.vars.get(name)

    def pdf(self, name: str) -> Any:
        return self.pdf_obj if name == "model_ch0" else None

    def InheritsFrom(self, _cls: Any) -> bool:
        return True


class FakeKey:
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def ReadObj(self) -> Any:
        return self.obj


class FakeRootFile:
    def __init__(
        self, workspace: Any | None = None, *, zombie: bool = False, truthy: bool = True
    ) -> None:
        self.workspace = workspace or FakeRootWorkspace()
        self.zombie = zombie
        self.truthy = truthy
        self.closed = False

    def __bool__(self) -> bool:
        return self.truthy

    def IsZombie(self) -> bool:
        return self.zombie

    def GetListOfKeys(self) -> list[FakeKey]:
        return [FakeKey(self.workspace)]

    def Close(self) -> None:
        self.closed = True


class FakeROOT:
    opened = FakeRootFile()

    class RooWorkspace:
        @staticmethod
        def Class() -> object:
            return object()

    class RooFit:
        WARNING = 1

    class RooMsgService:
        @staticmethod
        def instance() -> Any:
            return SimpleNamespace(setGlobalKillBelow=lambda _level: None)

    class RooArgSet:
        def __init__(self, *args: Any) -> None:
            self.args = args

    class TFile:
        @staticmethod
        def Open(_path: str, _mode: str) -> FakeRootFile:
            return FakeROOT.opened


def fake_time_once(fn: Any, *, label: str) -> tuple[Any, float]:
    return fn(), {
        "workspace loading": 0.1,
        "model construction": 0.2,
        "distribution expression lookup": 0.03,
        "PyTensor-to-JAX graph conversion": 0.04,
        "point-by-point JAX NLL lowering and XLA compilation": 0.4,
        "full-dataset JAX NLL lowering and XLA compilation": 0.5,
        "ROOT file loading": 0.11,
        "RooWorkspace lookup": 0.12,
        "first NLL evaluation": 0.01,
    }[label]


def test_workspace_data_and_errors() -> None:
    ws = FakeWorkspace()
    assert np.allclose(benchmark._workspace_data(ws, "combData_ch0", 0), [0, 1, 2])
    with pytest.raises(KeyError):
        benchmark._workspace_data(ws, "missing", 0)
    with pytest.raises(IndexError):
        benchmark._workspace_data(ws, "combData_ch0", 3)
    bad = SimpleNamespace(
        data=SimpleNamespace(root=[SimpleNamespace(name="d", entries=[[np.nan]])])
    )
    with pytest.raises(ValueError):
        benchmark._workspace_data(bad, "d", 0)


def test_model_defaults_set_scalar_and_validate_pdf_array() -> None:
    values = benchmark._model_defaults(FakeModel())
    assert set(values) == {"x", "mu_sig", "tau_ch0"}
    benchmark._set_scalar(values, "x", 4.0)
    benchmark._set_scalar(values, "mu_sig", 2.0)
    assert np.allclose(values["x"], [4.0])
    assert float(values["mu_sig"]) == 2.0
    with pytest.raises(KeyError):
        benchmark._set_scalar(values, "missing", 1.0)
    assert np.allclose(
        benchmark._validate_pdf_array([1.0, 2.0], 2, label="pdf"), [1, 2]
    )
    with pytest.raises(ValueError, match="expected"):
        benchmark._validate_pdf_array([1.0], 2, label="pdf")
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        benchmark._validate_pdf_array([0.0, 1.0], 2, label="pdf")


def test_shared_inputs(
    monkeypatch: pytest.MonkeyPatch, config: benchmark.Config
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    params, data = benchmark._shared_inputs(config)
    assert "mu_sig" in params and np.allclose(data, [0, 1, 2])
    FakeWorkspace.model_object = SimpleNamespace(data={"x": [1]}, free_params={})
    with pytest.raises(KeyError, match="Required inputs missing"):
        benchmark._shared_inputs(config)
    FakeWorkspace.model_object = FakeModel()


def test_find_and_sync_root_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeROOT)
    ws = FakeRootWorkspace()
    assert benchmark._find_root_workspace(FakeRootFile(ws)) is ws
    empty = SimpleNamespace(GetListOfKeys=lambda: [])
    with pytest.raises(KeyError):
        benchmark._find_root_workspace(empty)

    result = benchmark._sync_root_parameters(
        ws,
        {
            "x": np.asarray([1.0]),
            "mu_sig": np.asarray(2.0),
            "tau_ch0": np.asarray(0.2),
            "missing": np.asarray(1.0),
            "array": np.asarray([1.0, 2.0]),
            "nan": np.asarray(np.nan),
        },
        {"x"},
    )
    assert ws.vars["mu_sig"].value == pytest.approx(2.0)
    assert ws.vars["tau_ch0"].value == pytest.approx(-0.2)
    assert "tau_ch0" in result["transformed"]
    assert result["skipped"]["x"] == "controlled by the benchmark"
    assert "missing" in result["skipped"]


def test_noncompiled_pointwise_and_batched(
    monkeypatch: pytest.MonkeyPatch, config: benchmark.Config
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)
    monkeypatch.setattr(
        benchmark,
        "finite_scalar",
        lambda v, *, label: float(np.asarray(v).reshape(-1)[0]),
    )

    lifecycle, evaluate, cleanup = benchmark._make_pyhs3_noncompiled(
        config, np.asarray([0.0, 1.0])
    )
    expected = -sum(math.log(math.exp(-0.5 * (x - 1.0) ** 2) + 0.1) for x in [0.0, 1.0])
    assert evaluate(1.0) == pytest.approx(expected)
    assert lifecycle["model_construction_seconds"] == 0.2
    assert cleanup() is None

    batched = benchmark.dataclass_replace(config, category=benchmark.BATCHED_NLL)
    _, evaluate_batched, _ = benchmark._make_pyhs3_noncompiled(
        batched, np.asarray([0.0, 1.0])
    )
    assert evaluate_batched(1.0) == pytest.approx(expected)

    bad = benchmark.dataclass_replace(config, category="bad")
    with pytest.raises(ValueError):
        benchmark._make_pyhs3_noncompiled(bad, np.asarray([0.0]))


def test_make_roofit(monkeypatch: pytest.MonkeyPatch, config: benchmark.Config) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeROOT)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)
    monkeypatch.setattr(benchmark, "finite_scalar", lambda v, *, label: float(v))
    FakeROOT.opened = FakeRootFile(FakeRootWorkspace())
    lifecycle, evaluate, cleanup = benchmark._make_roofit(
        config,
        {"mu_sig": np.asarray(0.0)},
        np.asarray([0.0, 1.0]),
    )
    assert math.isfinite(evaluate(1.0))
    assert lifecycle["workspace_loading_seconds"] == 0.11
    cleanup()
    assert FakeROOT.opened.closed

    batched = benchmark.dataclass_replace(config, category=benchmark.BATCHED_NLL)
    with pytest.raises(NotImplementedError):
        benchmark._make_roofit(batched, {}, np.asarray([0.0]))
    monkeypatch.setattr(benchmark, "ROOT", None)
    with pytest.raises(RuntimeError):
        benchmark._make_roofit(config, {}, np.asarray([0.0]))


def test_scan_timing(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([0.0, 1.0, 2.0, 4.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    result = benchmark._scan_timing(
        lambda mu: (mu - 1.0) ** 2, np.asarray([0.0, 1.0, 2.0]), 2
    )
    assert result["full_scan_time_seconds_median"] == pytest.approx(1.5)
    assert result["minimum_mu"] == 1.0
    with pytest.raises(ValueError):
        benchmark._scan_timing(lambda x: x, np.asarray([]), 1)
    with pytest.raises(ValueError):
        benchmark._scan_timing(lambda x: x, np.asarray([1.0]), 0)


def test_run_engine_success_and_validation(
    monkeypatch: pytest.MonkeyPatch, config: benchmark.Config
) -> None:
    lifecycle = {
        "workspace_loading_seconds": 0.1,
        "model_construction_seconds": 0.2,
        "graph_preparation_seconds": 0.0,
        "compilation_seconds": 0.0,
    }
    monkeypatch.setattr(
        benchmark,
        "_make_pyhs3_noncompiled",
        lambda *_: (lifecycle.copy(), lambda mu: (mu - 1) ** 2 + 1, lambda: None),
    )
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)
    monkeypatch.setattr(
        benchmark,
        "benchmark_batches",
        lambda *a, **k: {"steady_state_seconds_median": 1e-4},
    )
    monkeypatch.setattr(
        benchmark,
        "_scan_timing",
        lambda *a, **k: {
            "full_scan_time_seconds_samples": [0.1],
            "full_scan_time_seconds_median": 0.1,
            "full_scan_time_seconds_mean": 0.1,
            "full_scan_time_seconds_std": 0.0,
            "time_per_scan_point_seconds": 0.01,
            "scan_throughput_points_per_second": 100.0,
            "nll_values": [2.0, 1.0, 2.0],
            "delta_nll_values": [1.0, 0.0, 1.0],
            "minimum_mu": 1.0,
            "minimum_nll": 1.0,
        },
    )
    monkeypatch.setattr(benchmark, "current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "peak_rss_mb", lambda: 120.0)
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    row = benchmark.run_engine(
        config, np.asarray([1.0, 0.0, 1.0]), {}, np.asarray([0.0, 1.0])
    )
    assert row["status"] == "success"
    assert row["cold_start_end_to_end_seconds"] == pytest.approx(0.31)
    assert row["category_key"] == benchmark.POINTWISE_NLL


def test_worker_and_isolation(
    monkeypatch: pytest.MonkeyPatch, config: benchmark.Config
) -> None:
    q: queue.Queue[Any] = queue.Queue()
    monkeypatch.setattr(benchmark, "run_engine", lambda *a: {"status": "success"})
    benchmark._isolated_engine_worker(config, None, {}, np.asarray([1.0]), q)
    assert q.get()["ok"] is True
    monkeypatch.setattr(
        benchmark, "run_engine", lambda *a: (_ for _ in ()).throw(ValueError("bad"))
    )
    benchmark._isolated_engine_worker(config, None, {}, np.asarray([1.0]), q)
    assert q.get()["ok"] is False


def test_labels_failure_replace_and_validate_args(config: benchmark.Config) -> None:
    assert benchmark._workspace_title("a_b.json") == "a / b"
    assert benchmark._workspace_multiline_label("a_b") == "a\nb"
    assert benchmark._category_short_label(benchmark.POINTWISE_NLL) == "pointwise NLL"
    assert benchmark._engine_short_label(benchmark.ROOFIT) == "RooFit"
    row = {
        "workspace_label": "a_b",
        "category_key": benchmark.POINTWISE_NLL,
        "engine": benchmark.ROOFIT,
    }
    assert "RooFit" in benchmark._bar_label(row)
    changed = benchmark.dataclass_replace(config, engine=benchmark.ROOFIT)
    assert changed.engine == benchmark.ROOFIT and config.engine != changed.engine
    failure = benchmark._failure_row(config, RuntimeError("boom"))
    assert failure["status"] == "failed"

    valid = SimpleNamespace(
        n_mu_values=2,
        mu_min=0.0,
        mu_max=1.0,
        batch_size=1,
        n_batches=1,
        warmup_batches=0,
        scan_repeats=1,
        observable_index=0,
        rtol=0.0,
        atol=0.0,
    )
    benchmark._validate_args(valid)
    for attr, value in [
        ("n_mu_values", 1),
        ("mu_max", 0.0),
        ("batch_size", 0),
        ("scan_repeats", 0),
        ("observable_index", -1),
        ("rtol", -1.0),
    ]:
        bad = SimpleNamespace(**vars(valid))
        setattr(bad, attr, value)
        with pytest.raises(ValueError):
            benchmark._validate_args(bad)


def _result(engine: str, category: str, workspace: str = "a.json") -> dict[str, Any]:
    return {
        "status": "success",
        "engine": engine,
        "category_key": category,
        "workspace": workspace,
        "workspace_label": Path(workspace).stem,
        "mu_values": [0.0, 1.0],
        "delta_nll_values": [1.0, 0.0],
        "steady_state_seconds_median": 1e-4,
        "current_rss_delta_mb": 1.0,
        "peak_rss_delta_mb": 2.0,
        "model_construction_seconds": 0.1,
        "graph_preparation_seconds": 0.02,
        "compilation_seconds": 0.2,
        "first_call_seconds": 0.01,
        "cold_start_end_to_end_seconds": 0.4,
    }


def test_plot_functions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    saved: list[Path] = []
    monkeypatch.setattr(benchmark, "save_figure", lambda fig, path: saved.append(path))
    rows = [
        _result(benchmark.PYHS3_NONCOMPILED, benchmark.POINTWISE_NLL),
        _result(benchmark.PYHS3_COMPILED, benchmark.POINTWISE_NLL),
        _result(benchmark.ROOFIT, benchmark.POINTWISE_NLL),
        _result(benchmark.PYHS3_NONCOMPILED, benchmark.BATCHED_NLL),
        _result(benchmark.PYHS3_COMPILED, benchmark.BATCHED_NLL),
    ]
    benchmark.plot_scan_agreement(rows, tmp_path / "a.png")
    benchmark.plot_runtime(rows, tmp_path / "b.png")
    benchmark.plot_memory(rows, tmp_path / "c.png")
    benchmark.plot_compiled_lifecycle(rows, tmp_path / "d.png")
    benchmark.plot_end_to_end_vs_steady(rows, tmp_path / "e.png")
    assert len(saved) == 5
    assert benchmark._successful_rows(rows, benchmark.POINTWISE_NLL)


class FakeJaxifiedNLL:
    input_names = ["x", "mu_sig", "tau_ch0"]

    def __call__(self, **inputs: Any) -> list[Any]:
        x = np.asarray(inputs["x"], dtype=float)
        mu = float(np.asarray(inputs["mu_sig"]).reshape(-1)[0])
        return [np.exp(-0.5 * (x - mu) ** 2) + 0.1]


class FakeLowered:
    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def compile(self) -> Any:
        return self.fn


class FakeJitted:
    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def lower(self, *_args: Any, **_kwargs: Any) -> FakeLowered:
        return FakeLowered(self.fn)


def patch_fake_jax(
    monkeypatch: pytest.MonkeyPatch, jaxified: Any | None = None
) -> None:
    monkeypatch.setattr(
        benchmark, "jaxify", lambda _expr: jaxified or FakeJaxifiedNLL()
    )
    monkeypatch.setattr(benchmark.jax, "jit", lambda fn: FakeJitted(fn))
    monkeypatch.setattr(
        benchmark.jax,
        "vmap",
        lambda fn: (
            lambda values: np.asarray([fn(value) for value in np.asarray(values)])
        ),
    )
    monkeypatch.setattr(benchmark.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(benchmark.jax, "default_backend", lambda: "cpu")


def test_compiled_pointwise_and_batched_paths(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)
    patch_fake_jax(monkeypatch)

    lifecycle, evaluate, cleanup = benchmark._make_pyhs3_compiled(
        benchmark.dataclass_replace(config, engine=benchmark.PYHS3_COMPILED),
        np.asarray([0.0, 1.0]),
    )
    expected = -sum(math.log(math.exp(-0.5 * (x - 1.0) ** 2) + 0.1) for x in [0.0, 1.0])
    assert evaluate(1.0) == pytest.approx(expected)
    assert lifecycle["compiled_program_scope"] == "pointwise_full_nll"
    assert lifecycle["graph_preparation_seconds"] == pytest.approx(0.07)
    assert lifecycle["compilation_seconds"] == pytest.approx(0.4)
    assert lifecycle["jax_backend"] == "cpu"
    assert cleanup() is None

    batched = benchmark.dataclass_replace(
        config,
        engine=benchmark.PYHS3_COMPILED,
        category=benchmark.BATCHED_NLL,
    )
    lifecycle_batched, evaluate_batched, _ = benchmark._make_pyhs3_compiled(
        batched,
        np.asarray([0.0, 1.0]),
    )
    assert evaluate_batched(1.0) == pytest.approx(expected)
    assert lifecycle_batched["compiled_program_scope"] == "full_dataset_nll"
    assert lifecycle_batched["compilation_seconds"] == pytest.approx(0.5)


def test_compiled_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)

    class Inputs:
        def __init__(self, names: list[str]) -> None:
            self.input_names = names

        def __call__(self, **inputs: Any) -> list[Any]:
            return [np.asarray([1.0])]

    patch_fake_jax(monkeypatch, Inputs(["x", "mu_sig", "missing"]))
    with pytest.raises(KeyError, match="Compiled inputs missing"):
        benchmark._make_pyhs3_compiled(config, np.asarray([0.0]))

    patch_fake_jax(monkeypatch, Inputs(["mu_sig", "tau_ch0"]))
    with pytest.raises(KeyError, match="Observable"):
        benchmark._make_pyhs3_compiled(config, np.asarray([0.0]))

    patch_fake_jax(monkeypatch, Inputs(["x", "tau_ch0"]))
    with pytest.raises(KeyError, match="POI"):
        benchmark._make_pyhs3_compiled(config, np.asarray([0.0]))

    FakeWorkspace.model_object = FakeModel()
    FakeWorkspace.model_object.free_params["mu_sig"] = np.asarray([1.0, 2.0])
    patch_fake_jax(monkeypatch)
    with pytest.raises(ValueError, match="must be scalar-like"):
        benchmark._make_pyhs3_compiled(config, np.asarray([0.0]))
    FakeWorkspace.model_object = FakeModel()

    bad_category = benchmark.dataclass_replace(config, category="invalid")
    patch_fake_jax(monkeypatch)
    with pytest.raises(ValueError, match="invalid"):
        benchmark._make_pyhs3_compiled(bad_category, np.asarray([0.0]))


def test_compiled_invalid_results(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)

    class BadJaxified(FakeJaxifiedNLL):
        def __call__(self, **inputs: Any) -> list[Any]:
            x = np.asarray(inputs["x"])
            return [np.full_like(x, np.nan, dtype=float)]

    patch_fake_jax(monkeypatch, BadJaxified())
    _, evaluate, _ = benchmark._make_pyhs3_compiled(config, np.asarray([0.0]))
    with pytest.raises(ValueError, match="invalid value"):
        evaluate(1.0)

    batched = benchmark.dataclass_replace(config, category=benchmark.BATCHED_NLL)
    _, evaluate_batched, _ = benchmark._make_pyhs3_compiled(batched, np.asarray([0.0]))
    with pytest.raises(ValueError, match="invalid value"):
        evaluate_batched(1.0)


def test_noncompiled_and_roofit_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)
    monkeypatch.setattr(benchmark, "finite_scalar", lambda *_a, **_k: 0.0)
    _, evaluate, _ = benchmark._make_pyhs3_noncompiled(config, np.asarray([0.0]))
    with pytest.raises(ValueError, match="Non-positive"):
        evaluate(1.0)

    monkeypatch.setattr(benchmark, "ROOT", FakeROOT)
    FakeROOT.opened = FakeRootFile(FakeRootWorkspace())
    _, roofit_evaluate, _ = benchmark._make_roofit(config, {}, np.asarray([0.0]))
    with pytest.raises(ValueError, match="Non-positive"):
        roofit_evaluate(1.0)


def test_prepare_and_roofit_error_branches(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)
    FakeWorkspace.model_object = SimpleNamespace(data={"x": [1.0]}, free_params={})
    with pytest.raises(KeyError, match="Required inputs missing"):
        benchmark._prepare_pyhs3(config, np.asarray([0.0]))
    FakeWorkspace.model_object = FakeModel()

    monkeypatch.setattr(benchmark, "ROOT", FakeROOT)
    FakeROOT.opened = FakeRootFile(zombie=True)
    with pytest.raises(FileNotFoundError):
        benchmark._make_roofit(config, {}, np.asarray([0.0]))

    ws = FakeRootWorkspace()
    ws.pdf_obj = None
    FakeROOT.opened = FakeRootFile(ws)
    with pytest.raises(KeyError, match="Missing RooFit"):
        benchmark._make_roofit(config, {}, np.asarray([0.0]))
    assert FakeROOT.opened.closed


def test_run_engine_dispatch_and_failures(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    common = (
        {
            "workspace_loading_seconds": 0.0,
            "model_construction_seconds": 0.0,
            "graph_preparation_seconds": 0.0,
            "compilation_seconds": 0.0,
        },
        lambda mu: float(mu),
        lambda: None,
    )
    monkeypatch.setattr(benchmark, "_make_pyhs3_compiled", lambda *_: common)
    monkeypatch.setattr(benchmark, "_make_roofit", lambda *_: common)
    monkeypatch.setattr(benchmark, "time_once", lambda fn, *, label: (fn(), 0.01))
    monkeypatch.setattr(
        benchmark,
        "benchmark_batches",
        lambda *a, **k: {"steady_state_seconds_median": 0.001},
    )
    monkeypatch.setattr(
        benchmark,
        "_scan_timing",
        lambda *a, **k: {
            "full_scan_time_seconds_samples": [0.1],
            "full_scan_time_seconds_median": 0.1,
            "full_scan_time_seconds_mean": 0.1,
            "full_scan_time_seconds_std": 0.0,
            "time_per_scan_point_seconds": 0.01,
            "scan_throughput_points_per_second": 100.0,
            "nll_values": [0.0, 1.0, 2.0],
            "delta_nll_values": [0.0, 1.0, 2.0],
            "minimum_mu": 0.0,
            "minimum_nll": 0.0,
        },
    )
    monkeypatch.setattr(benchmark, "current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "peak_rss_mb", lambda: 120.0)
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)

    compiled = benchmark.dataclass_replace(config, engine=benchmark.PYHS3_COMPILED)
    assert (
        benchmark.run_engine(compiled, None, {}, np.asarray([0.0]))["status"]
        == "success"
    )
    roofit = benchmark.dataclass_replace(config, engine=benchmark.ROOFIT)
    assert (
        benchmark.run_engine(roofit, None, {}, np.asarray([0.0]))["engine"]
        == benchmark.ROOFIT
    )

    invalid = benchmark.dataclass_replace(config, engine="invalid")
    with pytest.raises(ValueError, match="Unsupported engine"):
        benchmark.run_engine(invalid, None, {}, np.asarray([0.0]))

    monkeypatch.setattr(
        benchmark, "agreement_arrays", lambda *a, **k: {"validation_status": "failed"}
    )
    failed_validation = benchmark.run_engine(
        compiled, np.asarray([0.0, 1.0, 2.0]), {}, np.asarray([0.0])
    )
    assert failed_validation["status"] == "validation_failed"


def test_run_engine_isolated_payload_branches(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    class FakeQueue:
        def __init__(self, payload: Any = None, empty: bool = False) -> None:
            self.payload, self.empty = payload, empty
            self.closed = False

        def get(self, timeout: float) -> Any:
            if self.empty:
                raise queue.Empty
            return self.payload

        def close(self) -> None:
            self.closed = True

        def join_thread(self) -> None:
            pass

    class FakeProcess:
        def __init__(self, exitcode: int, **_kwargs: Any) -> None:
            self.exitcode = exitcode

        def start(self) -> None:
            pass

        def join(self) -> None:
            pass

    class FakeContext:
        def __init__(
            self, payload: Any, exitcode: int = 0, empty: bool = False
        ) -> None:
            self.q = FakeQueue(payload, empty)
            self.exitcode = exitcode

        def Queue(self, maxsize: int) -> FakeQueue:
            return self.q

        def Process(self, **kwargs: Any) -> FakeProcess:
            return FakeProcess(self.exitcode)

    monkeypatch.setattr(
        benchmark.mp,
        "get_context",
        lambda *_: FakeContext({"ok": True, "row": {"x": 1}}),
    )
    assert benchmark.run_engine_isolated(config, None, {}, np.asarray([1.0])) == {
        "x": 1
    }

    monkeypatch.setattr(
        benchmark.mp,
        "get_context",
        lambda *_: FakeContext(
            {
                "ok": False,
                "error_type": "ValueError",
                "error_message": "bad",
                "traceback": "tb",
            }
        ),
    )
    with pytest.raises(RuntimeError, match="ValueError: bad"):
        benchmark.run_engine_isolated(config, None, {}, np.asarray([1.0]))

    monkeypatch.setattr(
        benchmark.mp,
        "get_context",
        lambda *_: FakeContext({"ok": True, "row": {}}, exitcode=2),
    )
    with pytest.raises(RuntimeError, match="exited with code"):
        benchmark.run_engine_isolated(config, None, {}, np.asarray([1.0]))

    monkeypatch.setattr(
        benchmark.mp, "get_context", lambda *_: FakeContext(None, empty=True)
    )
    with pytest.raises(RuntimeError, match="produced no result"):
        benchmark.run_engine_isolated(config, None, {}, np.asarray([1.0]))


def test_parse_args_and_empty_plot_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--n-mu-values", "3", "--mu-max", "2"])
    args = benchmark.parse_args()
    assert args.n_mu_values == 3 and args.mu_max == 2.0

    saved: list[Path] = []
    monkeypatch.setattr(benchmark, "save_figure", lambda fig, path: saved.append(path))
    benchmark.plot_scan_agreement([], tmp_path / "a.png")
    benchmark.plot_runtime([], tmp_path / "b.png")
    benchmark.plot_memory([], tmp_path / "c.png")
    benchmark.plot_compiled_lifecycle([], tmp_path / "d.png")
    benchmark.plot_end_to_end_vs_steady([], tmp_path / "e.png")
    assert not saved
    assert benchmark._category_short_label("custom_category") == "custom category"
    assert benchmark._engine_short_label("custom") == "custom"
    assert "RooFit" not in benchmark._bar_label(
        {
            "workspace_label": "a",
            "category_key": benchmark.POINTWISE_NLL,
            "engine": benchmark.ROOFIT,
        },
        include_engine=False,
    )


def test_main_success_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "a.json"
    workspace.write_text("{}")
    output = tmp_path / "result.json"
    plot_dir = tmp_path / "plots"
    args = SimpleNamespace(
        workspaces=[workspace],
        root_workspaces=None,
        engines=[
            benchmark.PYHS3_NONCOMPILED,
            benchmark.PYHS3_COMPILED,
            benchmark.ROOFIT,
        ],
        categories=[benchmark.POINTWISE_NLL, benchmark.BATCHED_NLL],
        analysis="L_ch0",
        distribution="model_ch0",
        data_name="combData_ch0",
        observable_name="x",
        observable_index=0,
        poi="mu_sig",
        mode="FAST_RUN",
        mu_min=0.0,
        mu_max=1.0,
        n_mu_values=2,
        batch_size=1,
        n_batches=1,
        warmup_batches=0,
        scan_repeats=1,
        rtol=1e-7,
        atol=1e-7,
        output=output,
        plot_dir=plot_dir,
        plot=True,
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark,
        "_shared_inputs",
        lambda *_: (
            {"x": np.asarray([0.0]), "mu_sig": np.asarray(0.0)},
            np.asarray([0.0]),
        ),
    )

    def fake_isolated(
        config: benchmark.Config, reference: Any, *_args: Any
    ) -> dict[str, Any]:
        return {
            "status": "success",
            "delta_nll_values": [0.0, 1.0],
            "engine": config.engine,
        }

    monkeypatch.setattr(benchmark, "run_engine_isolated", fake_isolated)
    payloads: list[Any] = []
    monkeypatch.setattr(
        benchmark, "save_json", lambda payload, path: payloads.append(payload)
    )
    for name in (
        "plot_scan_agreement",
        "plot_runtime",
        "plot_memory",
        "plot_compiled_lifecycle",
        "plot_end_to_end_vs_steady",
    ):
        monkeypatch.setattr(benchmark, name, lambda *a, **k: None)
    benchmark.main()
    assert payloads[-1]["summary"]["all_required_runs_passed"] is True
    assert len(payloads[-1]["results"]) == 5

    bad_args = SimpleNamespace(**vars(args))
    bad_args.root_workspaces = [tmp_path / "a.root", tmp_path / "b.root"]
    monkeypatch.setattr(benchmark, "parse_args", lambda: bad_args)
    with pytest.raises(ValueError, match="one path per"):
        benchmark.main()

    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark,
        "_shared_inputs",
        lambda *_: (_ for _ in ()).throw(RuntimeError("bad inputs")),
    )
    with pytest.raises(SystemExit):
        benchmark.main()


# Additional defensive-branch coverage for the NLL benchmark.


def test_sync_root_parameters_defensive_branches() -> None:
    class BoolRaisesVar(FakeVar):
        def __bool__(self) -> bool:
            raise RuntimeError("bool failed")

    class RangeRaisesVar(FakeVar):
        def getMin(self) -> float:
            raise RuntimeError("no min")

        def getMax(self) -> float:
            raise RuntimeError("no max")

    ws = FakeRootWorkspace()
    ws.vars.update(
        {
            "false_proxy": FakeVar(truthy=False),
            "bool_raises": BoolRaisesVar(),
            "range_raises": RangeRaisesVar(),
            "outside": FakeVar(-1.0, 1.0),
            "set_fails": FakeVar(),
            "array_value": FakeVar(),
            "nonfinite": FakeVar(),
        }
    )
    ws.vars["set_fails"].raise_set = True

    result = benchmark._sync_root_parameters(
        ws,
        {
            "false_proxy": np.asarray(1.0),
            "bool_raises": np.asarray(0.5),
            "range_raises": np.asarray(3.0),
            "outside": np.asarray(4.0),
            "set_fails": np.asarray(0.2),
            "array_value": np.asarray([1.0, 2.0]),
            "nonfinite": np.asarray(np.inf),
        },
        set(),
    )

    assert result["skipped"]["false_proxy"] == "null PyROOT proxy"
    assert "bool_raises" in result["synchronized"]
    assert "range_raises" in result["synchronized"]
    assert "outside RooFit range" in result["skipped"]["outside"]
    assert "ValueError" in result["skipped"]["set_fails"]
    assert "non-scalar value" in result["skipped"]["array_value"]
    assert result["skipped"]["nonfinite"] == "non-finite pyHS3 value"


def test_compiled_scalar_like_rejects_non_scalar_template(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)

    model = FakeModel()
    model.free_params["mu_sig"] = np.asarray([1.0, 2.0])
    FakeWorkspace.model_object = model

    class FakeJaxified:
        input_names = ["x", "mu_sig", "tau_ch0"]

        def __call__(self, **inputs: Any) -> list[Any]:
            return [np.asarray([1.0])]

    monkeypatch.setattr(benchmark, "jaxify", lambda _expr: FakeJaxified())

    with pytest.raises(ValueError, match="must be scalar-like"):
        benchmark._make_pyhs3_compiled(config, np.asarray([0.0]))

    FakeWorkspace.model_object = FakeModel()


def test_scan_timing_rejects_invalid_elapsed(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([1.0, 1.0])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(RuntimeError, match="Invalid full-scan timing"):
        benchmark._scan_timing(lambda mu: mu, np.asarray([0.0]), 1)


def test_run_engine_indexed_callback_and_unsupported_engine(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    lifecycle = {
        "workspace_loading_seconds": 0.1,
        "model_construction_seconds": 0.2,
        "graph_preparation_seconds": 0.0,
        "compilation_seconds": 0.0,
    }
    calls: list[float] = []

    def evaluate(mu: float) -> float:
        calls.append(mu)
        return (mu - 1.0) ** 2 + 1.0

    monkeypatch.setattr(
        benchmark,
        "_make_pyhs3_noncompiled",
        lambda *_: (lifecycle.copy(), evaluate, lambda: None),
    )
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)

    def fake_batches(fn: Any, **_kwargs: Any) -> dict[str, float]:
        assert fn(0) == pytest.approx(2.0)
        assert fn(4) == pytest.approx(1.0)  # 4 % 3 == 1
        return {"steady_state_seconds_median": 1e-4}

    monkeypatch.setattr(benchmark, "benchmark_batches", fake_batches)
    monkeypatch.setattr(
        benchmark,
        "_scan_timing",
        lambda *_a, **_k: {
            "full_scan_time_seconds_samples": [0.1],
            "full_scan_time_seconds_median": 0.1,
            "full_scan_time_seconds_mean": 0.1,
            "full_scan_time_seconds_std": 0.0,
            "time_per_scan_point_seconds": 0.01,
            "scan_throughput_points_per_second": 100.0,
            "nll_values": [2.0, 1.0, 2.0],
            "delta_nll_values": [1.0, 0.0, 1.0],
            "minimum_mu": 1.0,
            "minimum_nll": 1.0,
        },
    )
    monkeypatch.setattr(benchmark, "current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "peak_rss_mb", lambda: 120.0)
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)

    row = benchmark.run_engine(config, None, {}, np.asarray([0.0]))
    assert row["status"] == "success"
    assert 0.0 in calls and 1.0 in calls

    bad = benchmark.dataclass_replace(config, engine="unsupported")
    with pytest.raises(ValueError, match="Unsupported engine"):
        benchmark.run_engine(bad, None, {}, np.asarray([0.0]))


def test_misc_label_fallbacks_and_validation_finite_bounds(
    config: benchmark.Config,
) -> None:
    assert benchmark._category_short_label("custom_category") == "custom category"
    assert benchmark._engine_short_label("custom_engine") == "custom_engine"
    row = {
        "workspace_label": "a_b",
        "category_key": benchmark.POINTWISE_NLL,
        "engine": benchmark.ROOFIT,
    }
    label = benchmark._bar_label(row, include_engine=False)
    assert "RooFit" not in label

    base = SimpleNamespace(
        n_mu_values=2,
        mu_min=0.0,
        mu_max=1.0,
        batch_size=1,
        n_batches=1,
        warmup_batches=0,
        scan_repeats=1,
        observable_index=0,
        rtol=0.0,
        atol=0.0,
    )
    for attr, value in [("mu_min", np.nan), ("mu_max", np.inf)]:
        bad = SimpleNamespace(**vars(base))
        setattr(bad, attr, value)
        with pytest.raises(ValueError, match="finite"):
            benchmark._validate_args(bad)


def test_plot_scan_agreement_skips_missing_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    saved: list[Path] = []
    monkeypatch.setattr(benchmark, "save_figure", lambda fig, path: saved.append(path))
    rows = [_result(benchmark.PYHS3_NONCOMPILED, benchmark.POINTWISE_NLL)]
    benchmark.plot_scan_agreement(rows, tmp_path / "agreement.png")
    assert saved == [tmp_path / "agreement.png"]


def test_main_records_engine_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "a.json"
    workspace.write_text("{}")
    root = tmp_path / "a.root"
    root.write_text("root")

    args = SimpleNamespace(
        workspaces=[workspace],
        root_workspaces=[root],
        engines=[benchmark.PYHS3_NONCOMPILED],
        categories=[benchmark.POINTWISE_NLL],
        analysis="L_ch0",
        distribution="model_ch0",
        data_name="combData_ch0",
        observable_name="x",
        observable_index=0,
        poi="mu_sig",
        mode="FAST_RUN",
        mu_min=0.0,
        mu_max=1.0,
        n_mu_values=2,
        batch_size=1,
        n_batches=1,
        warmup_batches=0,
        scan_repeats=1,
        rtol=1e-7,
        atol=1e-7,
        output=tmp_path / "result.json",
        plot_dir=tmp_path / "plots",
        plot=False,
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark,
        "_shared_inputs",
        lambda _config: (
            {"x": np.asarray([0.0]), "mu_sig": np.asarray(1.0)},
            np.asarray([0.0]),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "run_engine_isolated",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("engine failed")),
    )
    payloads: list[dict[str, Any]] = []
    monkeypatch.setattr(
        benchmark, "save_json", lambda payload, _path: payloads.append(payload)
    )

    with pytest.raises(SystemExit, match="required NLL benchmark"):
        benchmark.main()

    assert payloads[0]["results"][0]["status"] == "failed"
