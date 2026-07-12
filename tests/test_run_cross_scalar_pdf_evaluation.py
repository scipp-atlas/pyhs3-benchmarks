from __future__ import annotations

import queue
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
    p = tmp_path / "5ch_case.json"
    p.write_text("{}")
    return p


@pytest.fixture
def config(workspace_path: Path) -> benchmark.Config:
    return benchmark.Config(
        engine=benchmark.PYHS3_NONCOMPILED,
        workspace_path=workspace_path,
        root_workspace_path=workspace_path.with_suffix(".root"),
        target="L_ch0",
        mode="FAST_RUN",
        distribution="model_ch0",
        observable_name="x",
        input_mode=benchmark.VARYING_INPUT,
        n_evaluations=(1, 10),
        timing_repeats=2,
        warmup_evaluations=1,
        validation_points=5,
        rtol=1e-7,
        atol=1e-7,
    )


class FakeModel:
    def __init__(self, raw: Any = None) -> None:
        self.data = {"x": np.asarray([0.0, 1.0] if raw is None else raw)}
        self.free_params = {"mu": np.asarray(1.0), "tau_ch0": np.asarray(0.2)}
        self.distributions = {"model_ch0": object()}

    def pdf(self, _distribution: str, **params: Any) -> np.ndarray:
        x = float(np.asarray(params["x"]).reshape(-1)[0])
        return np.asarray([2.0 + x])


class FakeWorkspace:
    model_object = FakeModel()

    @classmethod
    def load(cls, _path: Path) -> "FakeWorkspace":
        return cls()

    def model(self, *_args: Any, **_kwargs: Any) -> FakeModel:
        return self.model_object


class FakeVar:
    def __init__(self, lo=-10.0, hi=10.0, truthy=True) -> None:
        self.lo, self.hi, self.truthy, self.value = lo, hi, truthy, 0.0
        self.fail = False

    def __bool__(self):
        return self.truthy

    def getMin(self):
        return self.lo

    def getMax(self):
        return self.hi

    def setVal(self, v):
        if self.fail:
            raise ValueError("bad")
        self.value = float(v)


class FakePdf:
    def __init__(self, x: FakeVar) -> None:
        self.x = x

    def __bool__(self):
        return True

    def getVal(self, _norm):
        return 2.0 + self.x.value


class FakeRootWorkspace:
    def __init__(self) -> None:
        self.vars = {"x": FakeVar(), "mu": FakeVar(), "tau_ch0": FakeVar(-1.0, -0.01)}
        self.pdf_obj = FakePdf(self.vars["x"])

    def var(self, name):
        return self.vars.get(name)

    def pdf(self, name):
        return self.pdf_obj if name == "model_ch0" else None

    def InheritsFrom(self, _cls):
        return True


class FakeKey:
    def __init__(self, obj):
        self.obj = obj

    def ReadObj(self):
        return self.obj


class FakeFile:
    def __init__(self, ws=None, zombie=False, truthy=True):
        self.ws, self.zombie, self.truthy, self.closed = (
            ws or FakeRootWorkspace(),
            zombie,
            truthy,
            False,
        )

    def __bool__(self):
        return self.truthy

    def IsZombie(self):
        return self.zombie

    def GetListOfKeys(self):
        return [FakeKey(self.ws)]

    def Close(self):
        self.closed = True


class FakeROOT:
    opened = FakeFile()

    class RooWorkspace:
        @staticmethod
        def Class():
            return object()

    class RooFit:
        WARNING = 1

    class RooMsgService:
        @staticmethod
        def instance():
            return SimpleNamespace(setGlobalKillBelow=lambda _x: None)

    class RooArgSet:
        def __init__(self, *x):
            self.values = x

    class TFile:
        @staticmethod
        def Open(_p, _m):
            return FakeROOT.opened


def fake_time_once(fn: Any, *, label: str):
    return fn(), {
        "workspace loading": 0.1,
        "model construction": 0.2,
        "distribution graph lookup": 0.03,
        "PyTensor-to-JAX graph conversion": 0.04,
        "scalar JAX PDF lowering and XLA compilation": 0.4,
        "first engine call": 0.05,
        "ROOT file loading": 0.11,
        "RooWorkspace lookup": 0.12,
        "first RooFit call": 0.06,
    }[label]


def test_labels_style_and_helpers() -> None:
    assert benchmark._mode_label(benchmark.VARYING_INPUT) == "Changing observable"
    assert "cache" in benchmark._mode_label(benchmark.FIXED_INPUT)
    assert benchmark._mode_label("my_mode") == "My Mode"
    bars = [
        SimpleNamespace(
            set_edgecolor=lambda x: None,
            set_linewidth=lambda x: None,
            set_alpha=lambda x: None,
        )
    ]
    benchmark._style_bars(bars)
    values = benchmark._model_defaults(FakeModel())
    benchmark._set_scalar(values, "x", 3.0)
    benchmark._set_scalar(values, "mu", 2.0)
    assert np.allclose(values["x"], [3.0]) and float(values["mu"]) == 2.0


def test_shared_inputs_ranges(
    monkeypatch: pytest.MonkeyPatch, config: benchmark.Config
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    for raw, expected in [([-2, 2], [-2, 2]), ([3], [2, 4]), ([], [-1, 1])]:
        FakeWorkspace.model_object = FakeModel(raw)
        params, values = benchmark._shared_inputs(config)
        assert (
            "x" in params
            and values[0] == pytest.approx(expected[0])
            and values[-1] == pytest.approx(expected[1])
        )
    FakeWorkspace.model_object = SimpleNamespace(data={"y": [1]}, free_params={})
    with pytest.raises(KeyError):
        benchmark._shared_inputs(config)
    FakeWorkspace.model_object = FakeModel()


def test_find_and_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeROOT)
    ws = FakeRootWorkspace()
    assert benchmark._find_root_workspace(FakeFile(ws)) is ws
    with pytest.raises(KeyError):
        benchmark._find_root_workspace(SimpleNamespace(GetListOfKeys=lambda: []))
    result = benchmark._sync_root_parameters(
        ws,
        {
            "x": np.asarray([1.0]),
            "mu": np.asarray(2.0),
            "tau_ch0": np.asarray(0.2),
            "missing": np.asarray(1.0),
            "arr": np.asarray([1.0, 2.0]),
            "nan": np.asarray(np.nan),
        },
        "x",
    )
    assert ws.vars["mu"].value == 2 and ws.vars["tau_ch0"].value == pytest.approx(-0.2)
    assert "tau_ch0" in result["transformed"] and "missing" in result["skipped"]


def test_prepare_noncompiled_varying_fixed(
    monkeypatch: pytest.MonkeyPatch, config: benchmark.Config
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)
    monkeypatch.setattr(benchmark, "current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "peak_rss_mb", lambda: 120.0)
    monkeypatch.setattr(
        benchmark,
        "finite_scalar",
        lambda v, *, label: float(np.asarray(v).reshape(-1)[0]),
    )
    params = benchmark._model_defaults(FakeWorkspace.model_object)
    vals = np.asarray([0.0, 1.0, 2.0])
    lifecycle, scalar, grid, cleanup = benchmark._prepare_engine(config, params, vals)
    assert scalar(0) == 2 and scalar(1) == 3 and np.allclose(grid(vals), [2, 3, 4])
    assert lifecycle["cold_start_end_to_end_seconds"] == pytest.approx(0.35)
    assert cleanup() is None
    fixed = benchmark.Config(**{**config.__dict__, "input_mode": benchmark.FIXED_INPUT})
    _, scalar_fixed, grid_fixed, _ = benchmark._prepare_engine(fixed, params, vals)
    assert scalar_fixed(2) == 2 and np.allclose(grid_fixed(vals), [2, 3, 4])


def test_prepare_roofit(
    monkeypatch: pytest.MonkeyPatch, config: benchmark.Config
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeROOT)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)
    monkeypatch.setattr(benchmark, "current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "peak_rss_mb", lambda: 120.0)
    monkeypatch.setattr(benchmark, "finite_scalar", lambda v, *, label: float(v))
    FakeROOT.opened = FakeFile(FakeRootWorkspace())
    roofit = benchmark.Config(**{**config.__dict__, "engine": benchmark.ROOFIT})
    lifecycle, scalar, grid, cleanup = benchmark._prepare_engine(
        roofit, {"mu": np.asarray(1.0)}, np.asarray([0.0, 1.0])
    )
    assert scalar(1) == 3 and np.allclose(grid(np.asarray([0.0, 2.0])), [2, 4])
    assert lifecycle["first_call_seconds"] == 0.06
    cleanup()
    assert FakeROOT.opened.closed
    monkeypatch.setattr(benchmark, "ROOT", None)
    with pytest.raises(RuntimeError):
        benchmark._prepare_engine(roofit, {}, np.asarray([0.0]))


def test_evaluate_noncompiled_at(config: benchmark.Config) -> None:
    params = benchmark._model_defaults(FakeModel())
    assert benchmark._evaluate_noncompiled_at(
        FakeModel(), params, config, 2.0
    ) == pytest.approx(4.0)


def test_run_engine(monkeypatch: pytest.MonkeyPatch, config: benchmark.Config) -> None:
    lifecycle = {
        "current_rss_before_mb": 100.0,
        "peak_rss_before_mb": 120.0,
        "first_output": 2.0,
        "workspace_loading_seconds": 0.1,
        "model_construction_seconds": 0.2,
        "graph_preparation_seconds": 0.0,
        "compilation_seconds": 0.0,
        "first_call_seconds": 0.05,
        "model_to_first_evaluation_seconds": 0.25,
        "cold_start_end_to_end_seconds": 0.35,
        "end_to_end_first_evaluation_seconds": 0.35,
        "compiled_input_names": [],
    }
    monkeypatch.setattr(
        benchmark,
        "_prepare_engine",
        lambda *a: (
            lifecycle.copy(),
            lambda i: 2.0 + i,
            lambda v: np.asarray([2.0 + i for i, _ in enumerate(v)]),
            lambda: None,
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_scaling",
        lambda *a, **k: [
            {
                "n_evaluations": 10,
                "time_per_value_ns": 100.0,
                "time_per_value_seconds_median": 1e-7,
                "throughput_evaluations_per_second": 1e7,
            }
        ],
    )
    monkeypatch.setattr(
        benchmark,
        "agreement_arrays",
        lambda *a, **k: {"validation_status": "success", "max_abs_diff": 0.0},
    )
    monkeypatch.setattr(benchmark, "current_rss_mb", lambda: 110.0)
    monkeypatch.setattr(benchmark, "peak_rss_mb", lambda: 130.0)
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    rows, observed = benchmark.run_engine(
        config, np.asarray([2.0, 3.0]), {}, np.asarray([0.0, 1.0])
    )
    assert rows[0]["status"] == "success" and np.allclose(observed, [2, 3])
    assert rows[0]["current_rss_delta_mb"] == 10


def test_worker(monkeypatch: pytest.MonkeyPatch, config: benchmark.Config) -> None:
    q: queue.Queue[Any] = queue.Queue()
    monkeypatch.setattr(
        benchmark, "run_engine", lambda *a: ([{"status": "success"}], np.asarray([1.0]))
    )
    benchmark._isolated_engine_worker(config, None, {}, np.asarray([1.0]), q)
    assert q.get()["ok"]
    monkeypatch.setattr(
        benchmark, "run_engine", lambda *a: (_ for _ in ()).throw(ValueError("bad"))
    )
    benchmark._isolated_engine_worker(config, None, {}, np.asarray([1.0]), q)
    assert not q.get()["ok"]


def _row(engine: str, mode: str, workspace="a.json", n=10000):
    return {
        "status": "success",
        "engine": engine,
        "input_mode": mode,
        "workspace": workspace,
        "workspace_label": Path(workspace).stem,
        "n_evaluations": n,
        "time_per_value_ns": 100.0,
        "throughput_evaluations_per_second": 1e7,
        "cold_start_end_to_end_seconds": 0.5,
        "time_per_value_seconds_median": 1e-6,
        "current_rss_delta_mb": 1.0,
        "peak_rss_delta_mb": 2.0,
        "max_abs_diff": 1e-10,
        "model_construction_seconds": 0.1,
        "graph_preparation_seconds": 0.02,
        "compilation_seconds": 0.2,
        "first_call_seconds": 0.01,
    }


def test_plot_functions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    saved = []
    monkeypatch.setattr(benchmark, "save_figure", lambda fig, p: saved.append(p))
    rows = [
        _row(benchmark.PYHS3_NONCOMPILED, benchmark.VARYING_INPUT),
        _row(benchmark.PYHS3_COMPILED, benchmark.VARYING_INPUT),
        _row(benchmark.ROOFIT, benchmark.VARYING_INPUT),
    ]
    benchmark.plot_time_per_value(rows, benchmark.VARYING_INPUT, tmp_path / "a.png")
    benchmark.plot_throughput(rows, benchmark.VARYING_INPUT, tmp_path / "b.png")
    benchmark.plot_latency(rows, benchmark.VARYING_INPUT, tmp_path / "c.png")
    benchmark.plot_memory(rows, benchmark.VARYING_INPUT, tmp_path / "d.png")
    benchmark.plot_agreement(rows, benchmark.VARYING_INPUT, tmp_path / "e.png", 1e-7)
    benchmark.plot_compiled_lifecycle(rows, tmp_path / "f.png")
    assert len(saved) == 6 and benchmark._successful(rows, benchmark.VARYING_INPUT)
    assert benchmark._workspace_title("a_b.json") == "a / b"


class FakeJaxifiedScalar:
    input_names = ["x", "mu", "tau_ch0"]

    def __call__(self, **inputs: Any) -> list[Any]:
        x = float(np.asarray(inputs["x"]).reshape(-1)[0])
        return [np.asarray([2.0 + x])]


class FakeLoweredScalar:
    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def compile(self) -> Any:
        return self.fn


class FakeJittedScalar:
    def __init__(self, fn: Any) -> None:
        self.fn = fn

    def lower(self, *_args: Any, **_kwargs: Any) -> FakeLoweredScalar:
        return FakeLoweredScalar(self.fn)


def patch_scalar_jax(
    monkeypatch: pytest.MonkeyPatch, jaxified: Any | None = None
) -> None:
    monkeypatch.setattr(
        benchmark, "jaxify", lambda _expr: jaxified or FakeJaxifiedScalar()
    )
    monkeypatch.setattr(benchmark.jax, "jit", lambda fn: FakeJittedScalar(fn))
    monkeypatch.setattr(benchmark.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(benchmark.jax, "default_backend", lambda: "cpu")


def test_prepare_compiled_engine(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)
    monkeypatch.setattr(benchmark, "current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "peak_rss_mb", lambda: 120.0)
    patch_scalar_jax(monkeypatch)
    compiled = benchmark.Config(
        **{**config.__dict__, "engine": benchmark.PYHS3_COMPILED}
    )
    params = benchmark._model_defaults(FakeWorkspace.model_object)
    lifecycle, scalar, grid, cleanup = benchmark._prepare_engine(
        compiled, params, np.asarray([0.0, 1.0, 2.0])
    )
    assert scalar(1) == pytest.approx(3.0)
    assert np.allclose(grid(np.asarray([0.0, 2.0])), [2.0, 4.0])
    assert lifecycle["graph_preparation_seconds"] == pytest.approx(0.07)
    assert lifecycle["compilation_seconds"] == pytest.approx(0.4)
    assert lifecycle["compiled_program_scope"] == "scalar_pdf"
    assert lifecycle["jax_backend"] == "cpu"
    assert cleanup() is None


def test_compiled_engine_error_branches(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspace)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)
    monkeypatch.setattr(benchmark, "current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "peak_rss_mb", lambda: 120.0)
    compiled = benchmark.Config(
        **{**config.__dict__, "engine": benchmark.PYHS3_COMPILED}
    )

    class Inputs:
        def __init__(self, names: list[str], value: float = 1.0) -> None:
            self.input_names, self.value = names, value

        def __call__(self, **inputs: Any) -> list[Any]:
            return [np.asarray([self.value])]

    patch_scalar_jax(monkeypatch, Inputs(["x", "mu", "missing"]))
    with pytest.raises(KeyError, match="Compiled inputs missing"):
        benchmark._prepare_engine(compiled, {}, np.asarray([0.0]))

    patch_scalar_jax(monkeypatch, Inputs(["mu", "tau_ch0"]))
    with pytest.raises(KeyError, match="Observable"):
        benchmark._prepare_engine(
            compiled,
            {"mu": np.asarray(1.0), "tau_ch0": np.asarray(0.2)},
            np.asarray([0.0]),
        )

    patch_scalar_jax(monkeypatch, Inputs(["x", "mu", "tau_ch0"], value=np.nan))
    params = benchmark._model_defaults(FakeWorkspace.model_object)
    with pytest.raises(ValueError, match="invalid value"):
        benchmark._prepare_engine(compiled, params, np.asarray([0.0]))

    invalid = benchmark.Config(**{**config.__dict__, "engine": "invalid"})
    with pytest.raises(ValueError, match="invalid"):
        benchmark._prepare_engine(invalid, {}, np.asarray([0.0]))


def test_sync_additional_branches() -> None:
    class WeirdVar(FakeVar):
        def __bool__(self):
            raise RuntimeError("bool")

        def getMin(self):
            raise RuntimeError("range")

        def getMax(self):
            raise RuntimeError("range")

    ws = FakeRootWorkspace()
    ws.vars.update(
        {
            "false": FakeVar(truthy=False),
            "outside": FakeVar(-1.0, 1.0),
            "set_fail": FakeVar(),
            "weird": WeirdVar(),
            "nonfinite": FakeVar(),
            "array": FakeVar(),
        }
    )
    ws.vars["set_fail"].fail = True
    result = benchmark._sync_root_parameters(
        ws,
        {
            "x": np.asarray(0.0),
            "false": np.asarray(1.0),
            "outside": np.asarray(5.0),
            "set_fail": np.asarray(1.0),
            "weird": np.asarray(2.0),
            "nonfinite": np.asarray(np.inf),
            "array": np.asarray([1.0, 2.0]),
        },
        "x",
    )
    assert "null PyROOT proxy" in result["skipped"]["false"]
    assert "outside RooFit range" in result["skipped"]["outside"]
    assert "ValueError" in result["skipped"]["set_fail"]
    assert "weird" in result["synchronized"]
    assert "non-finite" in result["skipped"]["nonfinite"]
    assert "non-scalar" in result["skipped"]["array"]


def test_roofit_file_and_object_errors(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeROOT)
    monkeypatch.setattr(benchmark, "time_once", fake_time_once)
    monkeypatch.setattr(benchmark, "current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "peak_rss_mb", lambda: 120.0)
    roofit = benchmark.Config(**{**config.__dict__, "engine": benchmark.ROOFIT})

    FakeROOT.opened = FakeFile(zombie=True)
    with pytest.raises(FileNotFoundError):
        benchmark._prepare_engine(roofit, {}, np.asarray([0.0]))

    ws = FakeRootWorkspace()
    ws.pdf_obj = None
    FakeROOT.opened = FakeFile(ws)
    with pytest.raises(KeyError, match="Missing RooFit"):
        benchmark._prepare_engine(roofit, {}, np.asarray([0.0]))
    assert FakeROOT.opened.closed


def test_run_engine_validation_failed_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    cleaned: list[bool] = []
    lifecycle = {
        "current_rss_before_mb": 100.0,
        "peak_rss_before_mb": 120.0,
        "first_output": 2.0,
        "workspace_loading_seconds": 0.1,
        "model_construction_seconds": 0.2,
        "graph_preparation_seconds": 0.0,
        "compilation_seconds": 0.0,
        "first_call_seconds": 0.05,
        "model_to_first_evaluation_seconds": 0.25,
        "cold_start_end_to_end_seconds": 0.35,
        "end_to_end_first_evaluation_seconds": 0.35,
        "compiled_input_names": [],
    }
    monkeypatch.setattr(
        benchmark,
        "_prepare_engine",
        lambda *a: (
            lifecycle.copy(),
            lambda i: 2.0,
            lambda values: np.asarray([2.0] * len(values)),
            lambda: cleaned.append(True),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "benchmark_scaling",
        lambda *a, **k: [
            {
                "n_evaluations": 1,
                "time_per_value_ns": 100.0,
                "time_per_value_seconds_median": 1e-7,
                "throughput_evaluations_per_second": 1e7,
            }
        ],
    )
    monkeypatch.setattr(
        benchmark, "agreement_arrays", lambda *a, **k: {"validation_status": "failed"}
    )
    monkeypatch.setattr(benchmark, "current_rss_mb", lambda: 99.0)
    monkeypatch.setattr(benchmark, "peak_rss_mb", lambda: 119.0)
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    rows, _ = benchmark.run_engine(config, np.asarray([2.0] * 5), {}, np.arange(5.0))
    assert rows[0]["status"] == "validation_failed"
    assert rows[0]["current_rss_delta_mb"] == 0.0
    assert cleaned


def test_run_engine_isolated_branches(
    monkeypatch: pytest.MonkeyPatch,
    config: benchmark.Config,
) -> None:
    class FakeQueue:
        def __init__(self, payload: Any = None, empty: bool = False):
            self.payload, self.empty = payload, empty

        def get(self, timeout: float):
            if self.empty:
                raise queue.Empty
            return self.payload

        def close(self):
            pass

        def join_thread(self):
            pass

    class FakeProcess:
        def __init__(self, exitcode=0, **kwargs):
            self.exitcode = exitcode

        def start(self):
            pass

        def join(self):
            pass

    class FakeContext:
        def __init__(self, payload, exitcode=0, empty=False):
            self.q = FakeQueue(payload, empty)
            self.exitcode = exitcode

        def Queue(self, maxsize):
            return self.q

        def Process(self, **kwargs):
            return FakeProcess(self.exitcode)

    monkeypatch.setattr(
        benchmark.mp,
        "get_context",
        lambda *_: FakeContext(
            {"ok": True, "rows": [{"x": 1}], "observed_grid": [1.0]}
        ),
    )
    rows, grid = benchmark.run_engine_isolated(config, None, {}, np.asarray([0.0]))
    assert rows == [{"x": 1}] and np.allclose(grid, [1.0])

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
        benchmark.run_engine_isolated(config, None, {}, np.asarray([0.0]))

    monkeypatch.setattr(
        benchmark.mp,
        "get_context",
        lambda *_: FakeContext(
            {"ok": True, "rows": [], "observed_grid": []}, exitcode=2
        ),
    )
    with pytest.raises(RuntimeError, match="exited abnormally"):
        benchmark.run_engine_isolated(config, None, {}, np.asarray([0.0]))

    monkeypatch.setattr(
        benchmark.mp, "get_context", lambda *_: FakeContext(None, empty=True)
    )
    with pytest.raises(RuntimeError, match="produced no result"):
        benchmark.run_engine_isolated(config, None, {}, np.asarray([0.0]))


def test_parse_args_empty_plots_and_titles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import sys

    monkeypatch.setattr(
        sys, "argv", ["prog", "--n-evaluations", "1", "2", "--validation-points", "3"]
    )
    args = benchmark.parse_args()
    assert args.n_evaluations == [1, 2] and args.validation_points == 3

    saved: list[Any] = []
    monkeypatch.setattr(benchmark, "save_figure", lambda *a: saved.append(a))
    benchmark.plot_time_per_value([], benchmark.VARYING_INPUT, tmp_path / "a")
    benchmark.plot_throughput([], benchmark.VARYING_INPUT, tmp_path / "b")
    benchmark.plot_latency([], benchmark.VARYING_INPUT, tmp_path / "c")
    benchmark.plot_memory([], benchmark.VARYING_INPUT, tmp_path / "d")
    benchmark.plot_agreement([], benchmark.VARYING_INPUT, tmp_path / "e", 1e-7)
    benchmark.plot_compiled_lifecycle([], tmp_path / "f")
    assert not saved
    assert benchmark._workspace_title("a_b.json") == "a / b"
    assert benchmark._successful([{"status": "failed"}], None) == []


def test_main_success_failure_and_validation_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "a.json"
    workspace.write_text("{}")
    args = SimpleNamespace(
        workspaces=[workspace],
        root_workspaces=None,
        engines=[
            benchmark.PYHS3_NONCOMPILED,
            benchmark.PYHS3_COMPILED,
            benchmark.ROOFIT,
        ],
        target="L_ch0",
        mode="FAST_RUN",
        distribution="model_ch0",
        observable_name="x",
        input_modes=[benchmark.VARYING_INPUT, benchmark.FIXED_INPUT],
        n_evaluations=[1],
        timing_repeats=1,
        warmup_evaluations=0,
        validation_points=2,
        rtol=1e-7,
        atol=1e-7,
        output=tmp_path / "result.json",
        plot_dir=tmp_path / "plots",
        plot=True,
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark,
        "_shared_inputs",
        lambda *_: (
            {"x": np.asarray([0.0]), "mu": np.asarray(1.0)},
            np.asarray([0.0, 1.0]),
        ),
    )

    def fake_isolated(config: benchmark.Config, reference: Any, *_args: Any):
        return (
            [
                {
                    "status": "success",
                    "engine": config.engine,
                    "input_mode": config.input_mode,
                }
            ],
            np.asarray([2.0, 3.0]),
        )

    monkeypatch.setattr(benchmark, "run_engine_isolated", fake_isolated)
    payloads: list[Any] = []
    monkeypatch.setattr(
        benchmark, "save_json", lambda payload, path: payloads.append(payload)
    )
    for name in (
        "plot_time_per_value",
        "plot_throughput",
        "plot_latency",
        "plot_memory",
        "plot_agreement",
        "plot_compiled_lifecycle",
    ):
        monkeypatch.setattr(benchmark, name, lambda *a, **k: None)
    benchmark.main()
    assert payloads[-1]["summary"]["all_required_runs_passed"] is True
    assert len(payloads[-1]["results"]) == 6

    bad_args = SimpleNamespace(**vars(args))
    bad_args.root_workspaces = [tmp_path / "a.root", tmp_path / "b.root"]
    monkeypatch.setattr(benchmark, "parse_args", lambda: bad_args)
    with pytest.raises(ValueError, match="must match"):
        benchmark.main()

    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(
        benchmark,
        "run_engine_isolated",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    with pytest.raises(SystemExit):
        benchmark.main()
