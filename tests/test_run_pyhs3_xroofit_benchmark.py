from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from src import run_pyhs3_xroofit_benchmark as benchmark


@pytest.fixture
def json_workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.json"
    path.write_text("{}", encoding="utf-8")
    return path


@pytest.fixture
def root_workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "workspace.root"
    path.write_text("root", encoding="utf-8")
    return path


class FakeParameter:
    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value


class FakeParameterPoint:
    def __init__(
        self,
        name: str = "nominal",
        parameters: list[FakeParameter] | None = None,
    ) -> None:
        self.name = name
        self.parameters = parameters or [
            FakeParameter("mu_sig", 1.0),
            FakeParameter("nsig_ch0", 2.0),
            FakeParameter("nbkg_ch0", 3.0),
        ]


class FakeData:
    def __init__(
        self,
        name: str,
        entries: list[list[float]] | None = None,
    ) -> None:
        self.name = name
        self.entries = entries if entries is not None else [[0.1], [0.2]]


class FakePyHS3Model:
    def __init__(
        self,
        *,
        free_params: dict[str, Any] | None = None,
        logpdf_values: Any = None,
        pdf_values: dict[str, Any] | None = None,
    ) -> None:
        self.free_params = free_params or {"mu_sig": 1.0}
        self.logpdf_values = (
            np.asarray([-1.0, -2.0], dtype=np.float64)
            if logpdf_values is None
            else logpdf_values
        )
        self.pdf_values = pdf_values or {
            "sig_ch0": np.asarray([0.4, 0.5]),
            "bkg_ch0": np.asarray([0.2, 0.3]),
        }
        self.model_calls: list[tuple[str, dict[str, Any]]] = []

    def logpdf(self, target: str, **params: Any) -> Any:
        self.model_calls.append((target, params))
        return self.logpdf_values

    def pdf(self, target: str, **params: Any) -> Any:
        self.model_calls.append((target, params))
        return self.pdf_values[target]


class FakePyHS3Workspace:
    def __init__(
        self,
        *,
        model: FakePyHS3Model | None = None,
        points: list[FakeParameterPoint] | None = None,
        data: list[FakeData] | None = None,
    ) -> None:
        self._model = model or FakePyHS3Model()
        self.parameter_points = SimpleNamespace(
            root=points if points is not None else [FakeParameterPoint()]
        )
        self.data = SimpleNamespace(
            root=data if data is not None else [FakeData("combData_ch0")]
        )
        self.model_args: list[tuple[str, bool, str]] = []

    def model(self, analysis: str, progress: bool, mode: str) -> FakePyHS3Model:
        self.model_args.append((analysis, progress, mode))
        return self._model


class FakeWorkspaceLoader:
    loaded_workspace = FakePyHS3Workspace()

    @staticmethod
    def load(path: Path) -> FakePyHS3Workspace:
        return FakeWorkspaceLoader.loaded_workspace


class FakeRootVar:
    def __init__(
        self,
        name: str,
        value: float = 1.0,
        minimum: float = -10.0,
        maximum: float = 10.0,
        constant: bool = False,
        fail_set: bool = False,
    ) -> None:
        self.name = name
        self.value = float(value)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.constant = bool(constant)
        self.fail_set = fail_set

    def __bool__(self) -> bool:
        return True

    def setVal(self, value: float) -> None:
        if self.fail_set:
            raise RuntimeError("set failed")
        self.value = float(value)

    def getVal(self) -> float:
        return self.value

    def setConstant(self, value: bool) -> None:
        self.constant = bool(value)

    def isConstant(self) -> bool:
        return self.constant

    def getMin(self) -> float:
        return self.minimum

    def getMax(self) -> float:
        return self.maximum


class FakeNLL:
    def __init__(self, value: float = 2.5, cpp_class: str = "xRooNLLVar") -> None:
        self.value = value
        self.cpp_class = cpp_class

    def getVal(self) -> float:
        return self.value

    def ClassName(self) -> str:
        return self.cpp_class


class FakeXRooNode:
    def __init__(
        self,
        workspace: Any = None,
        *,
        children: dict[str, Any] | None = None,
        nll: Any = None,
    ) -> None:
        self.workspace = workspace
        self.children = children or {}
        self._nll = nll if nll is not None else FakeNLL()

    def __bool__(self) -> bool:
        return True

    def __getitem__(self, key: str) -> Any:
        value = self.children.get(key)
        if isinstance(value, BaseException):
            raise value
        return value

    def nll(self, dataset: str) -> Any:
        if isinstance(self._nll, BaseException):
            raise self._nll
        return self._nll


class FakeRootKeyObject:
    def __init__(self, workspace, is_workspace=True):
        self.workspace = workspace
        self.is_workspace = is_workspace

    def InheritsFrom(self, cls):
        return self.is_workspace

    def __getattr__(self, name):
        return getattr(self.workspace, name)


class FakeRootKey:
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def ReadObj(self) -> Any:
        return self.obj


class FakeRootWorkspace:
    def __init__(
        self,
        variables: dict[str, FakeRootVar] | None = None,
    ) -> None:
        self.variables = (
            variables
            if variables is not None
            else {
                "mu_sig": FakeRootVar("mu_sig", 1.0),
                "alpha": FakeRootVar("alpha", 0.0),
            }
        )

    def __bool__(self) -> bool:
        return True

    def InheritsFrom(self, cls: Any) -> bool:
        return True

    def var(self, name: str) -> Any:
        return self.variables.get(name)


class FakeRootFile:
    def __init__(
        self,
        workspace: Any | None = None,
        *,
        zombie: bool = False,
        direct_get: bool = True,
    ) -> None:
        self.workspace = workspace or FakeRootWorkspace()
        self.zombie = zombie
        self.direct_get = direct_get
        self.closed = False

    def __bool__(self) -> bool:
        return True

    def IsZombie(self) -> bool:
        return self.zombie

    def Get(self, name: str) -> Any:
        return self.workspace if self.direct_get else None

    def GetListOfKeys(self) -> list[FakeRootKey]:
        return [FakeRootKey(self.workspace)]

    def Close(self) -> None:
        self.closed = True


class FakeGSystem:
    def __init__(self, load_status: int = 0) -> None:
        self.load_status = load_status
        self.loaded: list[str] = []

    def Load(self, library: str) -> int:
        self.loaded.append(library)
        return self.load_status


class FakeRootModule:
    def __init__(
        self,
        *,
        root_file: FakeRootFile | None = None,
        load_status: int = 0,
        has_xroonode: bool = True,
    ) -> None:
        self.gSystem = FakeGSystem(load_status)
        self._root_file = root_file or FakeRootFile()

        class TFile:
            @staticmethod
            def Open(path: str, mode: str) -> FakeRootFile:
                return self._root_file

        class RooWorkspace:
            @staticmethod
            def Class() -> object:
                return object()

        self.TFile = TFile
        self.RooWorkspace = RooWorkspace
        if has_xroonode:
            self.xRooNode = lambda workspace: FakeXRooNode(
                workspace,
                children={"pdfs/sim_pdf": FakeXRooNode()},
            )


def make_pyhs3_case(
    *,
    model: FakePyHS3Model | None = None,
    nll_mode: str = "extended-mixture",
    poi: str = "mu_sig",
    initial_poi: float = 1.0,
) -> benchmark.PyHS3Case:
    return benchmark.PyHS3Case(
        model=model or FakePyHS3Model(),
        target="model_ch0",
        params={
            poi: np.asarray(initial_poi),
            "x": np.asarray([0.1, 0.2]),
            "nsig_ch0": np.asarray(2.0),
            "nbkg_ch0": np.asarray(3.0),
        },
        poi=poi,
        nll_mode=nll_mode,
        signal_pdf="sig_ch0",
        background_pdf="bkg_ch0",
        signal_yield_param="nsig_ch0",
        background_yield_param="nbkg_ch0",
        initial_poi=initial_poi,
        engine_mode="FAST_RUN",
        phase_timings={
            "workspace_loading_seconds": 0.1,
            "model_construction_seconds": 0.2,
            "nll_construction_seconds": 0.0,
            "compilation_seconds": 0.0,
        },
    )


def make_xroofit_case(
    *,
    nll: Any | None = None,
    poi_var: FakeRootVar | None = None,
) -> benchmark.XRooFitCase:
    var = poi_var or FakeRootVar("mu_sig", 1.0, constant=False)
    root_file = FakeRootFile(FakeRootWorkspace({"mu_sig": var}))
    return benchmark.XRooFitCase(
        root_file=root_file,
        workspace=root_file.workspace,
        root_node=FakeXRooNode(),
        model_node=FakeXRooNode(),
        resolved_model_name="pdfs/sim_pdf",
        nll=nll or FakeNLL(),
        poi="mu_sig",
        poi_var=var,
        initial_poi=1.0,
        initial_constant=False,
        xroofit_node_python_type="ROOT.xRooNode",
        xroofit_model_node_python_type="ROOT.xRooNode",
        xroofit_nll_python_type="ROOT.xRooNLLVar",
        xroofit_nll_cpp_class="xRooNLLVar",
        xroofit_runtime_verified=True,
        phase_timings={
            "workspace_loading_seconds": 0.1,
            "model_construction_seconds": 0.2,
            "nll_construction_seconds": 0.3,
            "compilation_seconds": 0.0,
        },
    )


def make_engine_result(
    engine: str,
    scan: list[float],
    *,
    status: str = "success",
    minimum_poi: float = 1.0,
) -> dict[str, Any]:
    if status != "success":
        return {
            "engine": engine,
            "status": status,
            "engine_label": benchmark.ENGINE_STYLE[engine]["label"],
        }
    delta = benchmark.delta_nll(scan)
    return {
        "engine": engine,
        "framework": engine,
        "engine_label": benchmark.ENGINE_STYLE[engine]["label"],
        "framework_label": benchmark.ENGINE_STYLE[engine]["label"],
        "status": "success",
        "scan_nll_values": scan,
        "delta_nll_shape": delta,
        "minimum_poi": minimum_poi,
        "minimum_index": int(np.argmin(scan)),
        "steady_state_evaluation": {
            "median_seconds": 1e-4,
            "q1_seconds": 9e-5,
            "q3_seconds": 1.1e-4,
            "iqr_seconds": 2e-5,
        },
        "full_scan": {
            "median_seconds": 1e-2,
            "q1_seconds": 9e-3,
            "q3_seconds": 1.1e-2,
            "iqr_seconds": 2e-3,
        },
        "first_nll": float(scan[0]),
        "steady_state_nll": float(scan[-1]),
        "workspace_loading_time_seconds": 0.01,
        "model_construction_time_seconds": 0.02,
        "nll_construction_time_seconds": 0.03,
        "cold_first_evaluation_time_seconds": 0.04,
        "time_per_scan_point_seconds": 0.001,
        "current_rss_delta_mb": 1.0,
        "peak_rss_delta_mb": 2.0,
        "finite_values": True,
        "parameters_restored": True,
    }


def valid_run_kwargs(
    json_workspace_path: Path,
    root_workspace_path: Path,
    tmp_path: Path,
    **overrides: Any,
) -> dict[str, Any]:
    values = {
        "json_path": json_workspace_path,
        "root_path": root_workspace_path,
        "analysis_name": "L_ch0",
        "target": None,
        "pyhs3_data_name": None,
        "pyhs3_combined": False,
        "pyhs3_channels": None,
        "xroofit_model_name": "pdfs/sim_pdf",
        "xroofit_dataset_name": "combData",
        "root_workspace_name": "combWS",
        "poi": "mu_sig",
        "parameter_point": None,
        "observable_name": "x",
        "observable_index": 0,
        "pyhs3_noncompiled_mode": "FAST_COMPILE",
        "pyhs3_compiled_mode": "FAST_RUN",
        "pyhs3_nll_mode": "extended-mixture",
        "signal_pdf": None,
        "background_pdf": None,
        "signal_yield_param": None,
        "background_yield_param": None,
        "scan_min": 0.0,
        "scan_max": 2.0,
        "n_scan_points": 3,
        "n_warmup_evaluations": 1,
        "n_evaluation_runs": 2,
        "n_scan_runs": 1,
        "poi_timing_value": 1.0,
        "output": tmp_path / "result.json",
        "plot": False,
        "plot_dir": tmp_path / "plots",
        "delta_tolerance": 1e-6,
        "delta_relative_tolerance": 1e-7,
        "absolute_pyhs3_tolerance": 1e-10,
        "minimum_tolerance": 1e-12,
        "xroofit_library": "libxRooFit",
    }
    values.update(overrides)
    return values


def test_validate_existing_file_and_scalar_validators(tmp_path: Path) -> None:
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    assert benchmark.validate_existing_file(file_path, "input") == file_path

    with pytest.raises(FileNotFoundError, match="does not exist"):
        benchmark.validate_existing_file(tmp_path / "missing", "input")
    with pytest.raises(FileNotFoundError, match="is not a file"):
        benchmark.validate_existing_file(tmp_path, "input")

    benchmark.validate_positive_int(2, "count", minimum=2)
    with pytest.raises(ValueError, match="at least 2"):
        benchmark.validate_positive_int(1, "count", minimum=2)

    benchmark.validate_finite_float(1.5, "value")
    with pytest.raises(ValueError, match="must be finite"):
        benchmark.validate_finite_float(float("nan"), "value")


def test_validate_scan_config_success() -> None:
    benchmark.validate_scan_config(
        scan_min=0.0,
        scan_max=2.0,
        n_scan_points=3,
        n_warmup_evaluations=0,
        n_evaluation_runs=1,
        n_scan_runs=1,
        delta_tolerance=1e-6,
        delta_relative_tolerance=1e-7,
        absolute_pyhs3_tolerance=1e-10,
        minimum_tolerance=1e-12,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"scan_min": 2.0, "scan_max": 1.0}, "scan_min"),
        ({"n_scan_points": 1}, "n_scan_points"),
        ({"n_warmup_evaluations": -1}, "n_warmup_evaluations"),
        ({"n_evaluation_runs": 0}, "n_evaluation_runs"),
        ({"n_scan_runs": 0}, "n_scan_runs"),
        ({"delta_tolerance": 0.0}, "delta_tolerance"),
        ({"delta_relative_tolerance": -1.0}, "delta_relative_tolerance"),
        ({"absolute_pyhs3_tolerance": 0.0}, "absolute_pyhs3_tolerance"),
        ({"minimum_tolerance": 0.0}, "minimum_tolerance"),
        ({"scan_min": float("nan")}, "scan_min"),
    ],
)
def test_validate_scan_config_rejects_invalid(
    overrides: dict[str, Any],
    message: str,
) -> None:
    kwargs = {
        "scan_min": 0.0,
        "scan_max": 2.0,
        "n_scan_points": 3,
        "n_warmup_evaluations": 0,
        "n_evaluation_runs": 1,
        "n_scan_runs": 1,
        "delta_tolerance": 1e-6,
        "delta_relative_tolerance": 1e-7,
        "absolute_pyhs3_tolerance": 1e-10,
        "minimum_tolerance": 1e-12,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        benchmark.validate_scan_config(**kwargs)


def test_analysis_name_helpers() -> None:
    assert benchmark.channel_from_analysis("L_ch3") == "ch3"
    assert benchmark.default_target_from_analysis("L_ch3") == "model_ch3"
    assert benchmark.default_data_name_from_analysis("L_ch3") == "combData_ch3"
    assert benchmark.default_signal_pdf_from_analysis("L_ch3") == "sig_ch3"
    assert benchmark.default_background_pdf_from_analysis("L_ch3") == "bkg_ch3"
    assert benchmark.default_signal_yield_from_analysis("L_ch3") == "nsig_ch3"
    assert benchmark.default_background_yield_from_analysis("L_ch3") == "nbkg_ch3"
    with pytest.raises(ValueError, match="Cannot infer channel"):
        benchmark.channel_from_analysis("ch3")


def test_array_and_scalar_helpers() -> None:
    assert np.allclose(benchmark._as_array([[1.0], [2.0]]), [1.0, 2.0])
    assert benchmark._scalar([3.0], "x") == 3.0
    with pytest.raises(ValueError, match="empty"):
        benchmark._scalar([], "x")
    with pytest.raises(ValueError, match="finite"):
        benchmark._scalar([float("inf")], "x")


def test_extract_parameter_point_success_and_named_selection() -> None:
    workspace = FakePyHS3Workspace(
        points=[
            FakeParameterPoint("first", [FakeParameter("a", 1.0)]),
            FakeParameterPoint("chosen", [FakeParameter("b", 2.0)]),
        ]
    )
    first = benchmark.extract_parameter_point(workspace, None)
    chosen = benchmark.extract_parameter_point(workspace, "chosen")
    assert first["a"] == pytest.approx(1.0)
    assert chosen["b"] == pytest.approx(2.0)


def test_extract_parameter_point_errors() -> None:
    with pytest.raises(ValueError, match="parameter_points.root"):
        benchmark.extract_parameter_point(SimpleNamespace(), None)

    workspace = SimpleNamespace(parameter_points=SimpleNamespace(root=[]))
    with pytest.raises(ValueError, match="does not contain any"):
        benchmark.extract_parameter_point(workspace, None)

    workspace = FakePyHS3Workspace(points=[FakeParameterPoint("a")])
    with pytest.raises(KeyError, match="Available"):
        benchmark.extract_parameter_point(workspace, "missing")

    workspace = FakePyHS3Workspace(
        points=[FakeParameterPoint("bad", [FakeParameter("x", "bad")])]
    )
    with pytest.raises(ValueError, match="cannot be converted"):
        benchmark.extract_parameter_point(workspace, None)


def test_get_pyhs3_data_values_success_and_errors() -> None:
    workspace = FakePyHS3Workspace(data=[FakeData("data", [[1.0, 5.0], [2.0, 6.0]])])
    assert np.allclose(benchmark.get_pyhs3_data_values(workspace, "data", 1), [5, 6])

    with pytest.raises(KeyError, match="Available data"):
        benchmark.get_pyhs3_data_values(workspace, "missing")

    with pytest.raises(ValueError, match="is empty"):
        benchmark.get_pyhs3_data_values(
            FakePyHS3Workspace(data=[FakeData("data", [])]),
            "data",
        )

    with pytest.raises(ValueError, match="non-finite"):
        benchmark.get_pyhs3_data_values(
            FakePyHS3Workspace(data=[FakeData("data", [[float("nan")]])]),
            "data",
        )

    with pytest.raises(ValueError, match="data.root"):
        benchmark.get_pyhs3_data_values(SimpleNamespace(), "data")


def test_infer_combined_channels(
    monkeypatch: pytest.MonkeyPatch,
    json_workspace_path: Path,
) -> None:
    FakeWorkspaceLoader.loaded_workspace = FakePyHS3Workspace(
        data=[
            FakeData("combData_ch10"),
            FakeData("other"),
            FakeData("combData_ch2"),
            FakeData("combData_ch0"),
        ]
    )
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspaceLoader)
    assert benchmark.infer_combined_channels(json_workspace_path) == [
        "ch0",
        "ch2",
        "ch10",
    ]

    FakeWorkspaceLoader.loaded_workspace = FakePyHS3Workspace(data=[FakeData("other")])
    with pytest.raises(ValueError, match="Could not infer"):
        benchmark.infer_combined_channels(json_workspace_path)


def test_resolve_parameter_name(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        benchmark.resolve_parameter_name(["mu_sig", "x"], "mu_sig", context="test")
        == "mu_sig"
    )
    assert (
        benchmark.resolve_parameter_name(["mu_sig"], "mu", context="test") == "mu_sig"
    )
    assert "Resolved test POI alias" in capsys.readouterr().out
    with pytest.raises(KeyError, match="Available parameters"):
        benchmark.resolve_parameter_name(["x"], "mu", context="test")


def test_build_pyhs3_case_from_loaded_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([1.0, 1.25])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    workspace = FakePyHS3Workspace()
    case = benchmark.build_pyhs3_case_from_loaded_workspace(
        workspace=workspace,
        analysis_name="L_ch0",
        target="model_ch0",
        data_name="combData_ch0",
        poi="mu",
        parameter_point=None,
        observable_name="x",
        observable_index=0,
        mode="FAST_RUN",
        nll_mode="extended-mixture",
        signal_pdf="sig_ch0",
        background_pdf="bkg_ch0",
        signal_yield_param="nsig_ch0",
        background_yield_param="nbkg_ch0",
    )
    assert case.poi == "mu_sig"
    assert case.engine_mode == "FAST_RUN"
    assert case.phase_timings["model_construction_seconds"] == pytest.approx(0.25)
    assert np.allclose(case.params["x"], [0.1, 0.2])


def test_build_pyhs3_case(
    monkeypatch: pytest.MonkeyPatch,
    json_workspace_path: Path,
) -> None:
    FakeWorkspaceLoader.loaded_workspace = FakePyHS3Workspace()
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspaceLoader)
    times = iter([1.0, 1.1, 2.0, 2.2])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))

    case = benchmark.build_pyhs3_case(
        json_path=json_workspace_path,
        analysis_name="L_ch0",
        target="model_ch0",
        data_name="combData_ch0",
        poi="mu_sig",
        parameter_point=None,
        observable_name="x",
        observable_index=0,
        mode="FAST_RUN",
        nll_mode="extended-mixture",
        signal_pdf="sig_ch0",
        background_pdf="bkg_ch0",
        signal_yield_param="nsig_ch0",
        background_yield_param="nbkg_ch0",
    )
    assert case.phase_timings["workspace_loading_seconds"] == pytest.approx(0.1)


def test_build_combined_pyhs3_case_success(
    monkeypatch: pytest.MonkeyPatch,
    json_workspace_path: Path,
) -> None:
    workspace = FakePyHS3Workspace(
        data=[FakeData("combData_ch0"), FakeData("combData_ch1")]
    )
    FakeWorkspaceLoader.loaded_workspace = workspace
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspaceLoader)
    times = iter([1.0, 1.1, 2.0, 2.2, 3.0, 3.3])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))

    case = benchmark.build_combined_pyhs3_case(
        json_path=json_workspace_path,
        channels=["ch0", "ch1"],
        poi="mu_sig",
        parameter_point=None,
        observable_name="x",
        observable_index=0,
        mode="FAST_RUN",
        nll_mode="extended-mixture",
    )
    assert len(case.channels) == 2
    assert case.poi == "mu_sig"
    assert case.phase_timings["workspace_loading_seconds"] == pytest.approx(0.1)


def test_build_combined_pyhs3_case_rejects_inconsistent_cases(
    monkeypatch: pytest.MonkeyPatch,
    json_workspace_path: Path,
) -> None:
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspaceLoader)

    calls = iter(
        [
            make_pyhs3_case(poi="mu_sig", initial_poi=1.0),
            make_pyhs3_case(poi="mu", initial_poi=1.0),
        ]
    )
    monkeypatch.setattr(
        benchmark,
        "build_pyhs3_case_from_loaded_workspace",
        lambda **kwargs: next(calls),
    )
    with pytest.raises(benchmark.ValidationFailure, match="POI names"):
        benchmark.build_combined_pyhs3_case(
            json_path=json_workspace_path,
            channels=["ch0", "ch1"],
            poi="mu_sig",
            parameter_point=None,
            observable_name="x",
            observable_index=0,
            mode="FAST_RUN",
            nll_mode="extended-mixture",
        )

    calls = iter(
        [
            make_pyhs3_case(initial_poi=1.0),
            make_pyhs3_case(initial_poi=2.0),
        ]
    )
    monkeypatch.setattr(
        benchmark,
        "build_pyhs3_case_from_loaded_workspace",
        lambda **kwargs: next(calls),
    )
    with pytest.raises(benchmark.ValidationFailure, match="initial POI"):
        benchmark.build_combined_pyhs3_case(
            json_path=json_workspace_path,
            channels=["ch0", "ch1"],
            poi="mu_sig",
            parameter_point=None,
            observable_name="x",
            observable_index=0,
            mode="FAST_RUN",
            nll_mode="extended-mixture",
        )


def test_pyhs3_logpdf_nll_success_and_invalid() -> None:
    case = make_pyhs3_case(nll_mode="logpdf")
    assert benchmark.pyhs3_logpdf_nll(case, 1.5) == pytest.approx(3.0)

    case.model.logpdf_values = np.asarray([])
    with pytest.raises(benchmark.ValidationFailure, match="invalid logpdf"):
        benchmark.pyhs3_logpdf_nll(case, 1.0)

    case.model.logpdf_values = np.asarray([float("nan")])
    with pytest.raises(benchmark.ValidationFailure, match="invalid logpdf"):
        benchmark.pyhs3_logpdf_nll(case, 1.0)


def test_pyhs3_extended_mixture_nll_success() -> None:
    case = make_pyhs3_case()
    value = benchmark.pyhs3_extended_mixture_nll(case, 1.0)
    expected = 5.0 - np.sum(np.log([1.4, 1.9]))
    assert value == pytest.approx(expected)


def test_pyhs3_extended_mixture_nll_errors() -> None:
    case = make_pyhs3_case()
    del case.params["nsig_ch0"]
    with pytest.raises(KeyError, match="Missing yield parameter"):
        benchmark.pyhs3_extended_mixture_nll(case, 1.0)

    case = make_pyhs3_case(
        model=FakePyHS3Model(
            pdf_values={
                "sig_ch0": np.asarray([0.4]),
                "bkg_ch0": np.asarray([0.2, 0.3]),
            }
        )
    )
    with pytest.raises(benchmark.ValidationFailure, match="different shapes"):
        benchmark.pyhs3_extended_mixture_nll(case, 1.0)

    case = make_pyhs3_case(
        model=FakePyHS3Model(
            pdf_values={
                "sig_ch0": np.asarray([-10.0, -10.0]),
                "bkg_ch0": np.asarray([0.0, 0.0]),
            }
        )
    )
    with pytest.raises(benchmark.ValidationFailure, match="densities are invalid"):
        benchmark.pyhs3_extended_mixture_nll(case, 1.0)


def test_pyhs3_nll_dispatch_and_combined() -> None:
    log_case = make_pyhs3_case(nll_mode="logpdf")
    extended_case = make_pyhs3_case()
    combined = benchmark.CombinedPyHS3Case(
        channels=(extended_case, extended_case),
        poi="mu_sig",
        initial_poi=1.0,
        engine_mode="FAST_RUN",
    )
    assert benchmark.pyhs3_nll(log_case, 1.0) == pytest.approx(3.0)
    assert benchmark.pyhs3_nll(combined, 1.0) == pytest.approx(
        2 * benchmark.pyhs3_extended_mixture_nll(extended_case, 1.0)
    )
    extended_case.nll_mode = "bad"
    with pytest.raises(ValueError, match="Unknown PyHS3 NLL mode"):
        benchmark.pyhs3_nll(extended_case, 1.0)


def test_restore_pyhs3_case_success_and_mutation() -> None:
    case = make_pyhs3_case()
    benchmark.restore_pyhs3_case(case)
    case.params["mu_sig"] = np.asarray(2.0)
    with pytest.raises(benchmark.ValidationFailure, match="mutated"):
        benchmark.restore_pyhs3_case(case)

    good = make_pyhs3_case()
    bad = make_pyhs3_case()
    bad.params["mu_sig"] = np.asarray(2.0)
    combined = benchmark.CombinedPyHS3Case(
        channels=(good, bad),
        poi="mu_sig",
        initial_poi=1.0,
        engine_mode="FAST_RUN",
    )
    with pytest.raises(benchmark.ValidationFailure, match="mutated"):
        benchmark.restore_pyhs3_case(combined)


def test_require_xroofit(monkeypatch: pytest.MonkeyPatch) -> None:
    root = FakeRootModule()
    monkeypatch.setattr(benchmark, "ROOT", root)
    assert benchmark.require_xroofit("libxRooFit") is root
    assert root.gSystem.loaded == ["libxRooFit"]

    monkeypatch.setattr(benchmark, "ROOT", None)
    with pytest.raises(RuntimeError, match="ROOT is not available"):
        benchmark.require_xroofit()

    root = FakeRootModule(load_status=-1, has_xroonode=False)
    monkeypatch.setattr(benchmark, "ROOT", root)
    with pytest.raises(RuntimeError, match="Could not load"):
        benchmark.require_xroofit("libxRooFit")

    root = FakeRootModule(has_xroonode=False)
    monkeypatch.setattr(benchmark, "ROOT", root)
    with pytest.raises(RuntimeError, match="xRooFit is not available"):
        benchmark.require_xroofit(None)


def test_root_object_and_workspace_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    assert benchmark._is_valid_root_object(object())
    assert not benchmark._is_valid_root_object(None)

    class BadBool:
        def __bool__(self) -> bool:
            raise RuntimeError

    assert benchmark._is_valid_root_object(BadBool())

    workspace = FakeRootWorkspace()
    direct = FakeRootFile(workspace, direct_get=True)
    assert benchmark._find_workspace(direct, "combWS") is workspace

    module = FakeRootModule()
    monkeypatch.setitem(sys.modules, "ROOT", module)
    fallback = FakeRootFile(workspace, direct_get=False)
    found = benchmark._find_workspace(fallback, "missing")

    assert found is workspace

    non_workspace = FakeRootFile(workspace, direct_get=False)
    non_workspace.GetListOfKeys = lambda: [
        FakeRootKey(FakeRootKeyObject(workspace, is_workspace=False))
    ]
    with pytest.raises(RuntimeError, match="Could not find RooWorkspace"):
        benchmark._find_workspace(non_workspace, "missing")

    empty = FakeRootFile(workspace, direct_get=False)
    empty.GetListOfKeys = lambda: []
    with pytest.raises(RuntimeError, match="Could not find RooWorkspace"):
        benchmark._find_workspace(empty, "missing")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("pdfs/sim_pdf", ["pdfs/sim_pdf"]),
        ("ModelConfig", ["models/ModelConfig", "ModelConfig"]),
        ("L_ch0", ["models/L_ch0", "L_ch0"]),
        ("sim_pdf", ["pdfs/sim_pdf", "sim_pdf"]),
    ],
)
def test_candidate_xroofit_model_paths(name: str, expected: list[str]) -> None:
    assert benchmark._candidate_xroofit_model_paths(name) == expected


def test_candidate_xroofit_model_paths_empty() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        benchmark._candidate_xroofit_model_paths("")


def test_get_xroofit_node_success_and_failure() -> None:
    node = FakeXRooNode(children={"pdfs/sim_pdf": FakeXRooNode()})
    result, path = benchmark._get_xroofit_node(node, "sim_pdf")
    assert isinstance(result, FakeXRooNode)
    assert path == "pdfs/sim_pdf"

    node = FakeXRooNode(
        children={
            "pdfs/sim_pdf": RuntimeError("bad"),
            "sim_pdf": None,
        }
    )
    with pytest.raises(RuntimeError, match="Could not access"):
        benchmark._get_xroofit_node(node, "sim_pdf")


def test_set_root_defaults_from_pyhs3(
    monkeypatch: pytest.MonkeyPatch,
    json_workspace_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    points = [
        FakeParameterPoint(
            parameters=[
                FakeParameter("inside", 1.0),
                FakeParameter("outside", 5.0),
                FakeParameter("missing", 2.0),
                FakeParameter("fails", 1.0),
            ]
        )
    ]
    FakeWorkspaceLoader.loaded_workspace = FakePyHS3Workspace(points=points)
    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspaceLoader)

    workspace = FakeRootWorkspace(
        {
            "inside": FakeRootVar("inside", minimum=0, maximum=2),
            "outside": FakeRootVar("outside", minimum=0, maximum=1),
            "fails": FakeRootVar("fails", fail_set=True),
        }
    )
    result = benchmark._set_root_defaults_from_pyhs3(
        workspace,
        json_workspace_path,
        None,
    )
    assert result["applied"] == {"inside": 1.0}
    assert set(result["skipped"]) == {"outside", "missing", "fails"}
    assert "ROOT defaults not applied" in capsys.readouterr().out


def test_construct_xroofit_nll() -> None:
    model = FakeXRooNode(nll=FakeNLL())
    assert isinstance(benchmark._construct_xroofit_nll(model, "data"), FakeNLL)
    model = FakeXRooNode(nll=TypeError("bad"))
    with pytest.raises(RuntimeError, match="could not construct"):
        benchmark._construct_xroofit_nll(model, "data")


def test_get_workspace_poi(capsys: pytest.CaptureFixture[str]) -> None:
    workspace = FakeRootWorkspace({"mu_sig": FakeRootVar("mu_sig")})
    var, name = benchmark._get_workspace_poi(workspace, "mu")
    assert name == "mu_sig"
    assert var is workspace.variables["mu_sig"]
    assert "Resolved xRooFit POI alias" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="Could not find"):
        benchmark._get_workspace_poi(FakeRootWorkspace({}), "mu")


def test_type_helpers() -> None:
    node = FakeXRooNode()
    assert "FakeXRooNode" in benchmark._python_type_name(node)
    assert benchmark._root_cpp_class_name(FakeNLL()) == "xRooNLLVar"

    class IsAOnly:
        def ClassName(self) -> None:
            return None

        def IsA(self) -> Any:
            return SimpleNamespace(GetName=lambda: "SpecialClass")

    assert benchmark._root_cpp_class_name(IsAOnly()) == "SpecialClass"
    assert benchmark._root_cpp_class_name(object()) == "<unavailable>"


def test_verify_xroofit_runtime_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark,
        "_python_type_name",
        lambda obj: "ROOT.xRooNLLVar" if isinstance(obj, FakeNLL) else "ROOT.xRooNode",
    )
    result = benchmark._verify_xroofit_runtime(
        root_node=FakeXRooNode(),
        model_node=FakeXRooNode(),
        nll=FakeNLL(),
    )
    assert result["verified"] is True

    monkeypatch.setattr(benchmark, "_python_type_name", lambda obj: "ROOT.RooAbsPdf")
    with pytest.raises(RuntimeError, match="runtime verification failed"):
        benchmark._verify_xroofit_runtime(
            root_node=object(),
            model_node=object(),
            nll=object(),
        )


def test_build_xroofit_case_success(
    monkeypatch: pytest.MonkeyPatch,
    json_workspace_path: Path,
    root_workspace_path: Path,
) -> None:
    root_workspace = FakeRootWorkspace({"mu_sig": FakeRootVar("mu_sig", 1.2)})
    root_file = FakeRootFile(root_workspace)
    root = FakeRootModule(root_file=root_file)
    monkeypatch.setattr(benchmark, "ROOT", root)
    monkeypatch.setattr(benchmark, "require_xroofit", lambda library: root)
    monkeypatch.setattr(
        benchmark,
        "_set_root_defaults_from_pyhs3",
        lambda *args, **kwargs: {"applied": {"mu_sig": 1.0}, "skipped": {}},
    )
    monkeypatch.setattr(
        benchmark,
        "_get_xroofit_node",
        lambda node, name: (FakeXRooNode(nll=FakeNLL()), "pdfs/sim_pdf"),
    )
    monkeypatch.setattr(
        benchmark,
        "_verify_xroofit_runtime",
        lambda **kwargs: {
            "root_node_python_type": "ROOT.xRooNode",
            "model_node_python_type": "ROOT.xRooNode",
            "nll_python_type": "ROOT.xRooNLLVar",
            "nll_cpp_class": "xRooNLLVar",
            "verified": True,
        },
    )
    times = iter([1.0, 1.1, 2.0, 2.2, 3.0, 3.3])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))

    case = benchmark.build_xroofit_case(
        root_path=root_workspace_path,
        json_path=json_workspace_path,
        workspace_name="combWS",
        model_name="pdfs/sim_pdf",
        dataset_name="combData",
        poi="mu_sig",
        parameter_point=None,
        xroofit_library="libxRooFit",
    )
    assert case.xroofit_runtime_verified is True
    assert case.initial_poi == pytest.approx(1.2)
    assert case.poi_var.isConstant() is True


def test_build_xroofit_case_errors_close_file(
    monkeypatch: pytest.MonkeyPatch,
    json_workspace_path: Path,
    root_workspace_path: Path,
) -> None:
    root_file = FakeRootFile(zombie=True)
    root = FakeRootModule(root_file=root_file)
    monkeypatch.setattr(benchmark, "require_xroofit", lambda library: root)
    with pytest.raises(RuntimeError, match="Could not open"):
        benchmark.build_xroofit_case(
            root_path=root_workspace_path,
            json_path=json_workspace_path,
            workspace_name="combWS",
            model_name="pdfs/sim_pdf",
            dataset_name="combData",
            poi="mu_sig",
            parameter_point=None,
            xroofit_library=None,
        )

    root_file = FakeRootFile()
    root = FakeRootModule(root_file=root_file)
    monkeypatch.setattr(benchmark, "require_xroofit", lambda library: root)
    monkeypatch.setattr(
        benchmark,
        "_set_root_defaults_from_pyhs3",
        lambda *args, **kwargs: {"applied": {}, "skipped": {}},
    )
    monkeypatch.setattr(
        benchmark,
        "_get_xroofit_node",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError, match="boom"):
        benchmark.build_xroofit_case(
            root_path=root_workspace_path,
            json_path=json_workspace_path,
            workspace_name="combWS",
            model_name="pdfs/sim_pdf",
            dataset_name="combData",
            poi="mu_sig",
            parameter_point=None,
            xroofit_library=None,
        )
    assert root_file.closed is True


def test_xroofit_nll_and_restore() -> None:
    case = make_xroofit_case(nll=FakeNLL(4.2))
    assert benchmark.xroofit_nll(case, 1.5) == pytest.approx(4.2)
    assert case.poi_var.getVal() == pytest.approx(1.5)
    assert case.poi_var.isConstant() is True

    benchmark.restore_xroofit_case(case)
    assert case.poi_var.getVal() == pytest.approx(1.0)
    assert case.poi_var.isConstant() is False

    case.nll = SimpleNamespace(getVal=lambda: float("nan"))
    with pytest.raises(benchmark.ValidationFailure, match="non-finite"):
        benchmark.xroofit_nll(case, 1.0)


def test_xroofit_nll_float_fallback() -> None:
    class FloatNLL:
        def __float__(self) -> float:
            return 3.5

    case = make_xroofit_case(nll=FloatNLL())
    assert benchmark.xroofit_nll(case, 1.0) == pytest.approx(3.5)


def test_restore_xroofit_case_detects_failures() -> None:
    case = make_xroofit_case()

    def bad_set(value: float) -> None:
        case.poi_var.value = value + 1.0

    case.poi_var.setVal = bad_set
    with pytest.raises(benchmark.ValidationFailure, match="not restored"):
        benchmark.restore_xroofit_case(case)

    case = make_xroofit_case()
    original_set_constant = case.poi_var.setConstant

    def wrong_constant(value: bool) -> None:
        original_set_constant(not value)

    case.poi_var.setConstant = wrong_constant
    with pytest.raises(benchmark.ValidationFailure, match="constant state"):
        benchmark.restore_xroofit_case(case)


def test_close_case() -> None:
    case = make_xroofit_case()
    benchmark.close_case(case)
    assert case.root_file.closed is True
    benchmark.close_case(make_pyhs3_case())


def test_numeric_sequence_delta_minimum_and_summary() -> None:
    assert benchmark.validate_numeric_sequence([1, 2], "x") == [1.0, 2.0]
    with pytest.raises(ValueError, match="must not be empty"):
        benchmark.validate_numeric_sequence([], "x")
    with pytest.raises(ValueError, match="non-finite"):
        benchmark.validate_numeric_sequence([float("nan")], "x")

    assert benchmark.delta_nll([3.0, 1.0, 2.0]) == [2.0, 0.0, 1.0]
    assert benchmark.minimum_position([0.0, 1.0, 2.0], [3.0, 1.0, 2.0]) == 1.0
    with pytest.raises(ValueError, match="same length"):
        benchmark.minimum_position([0.0], [1.0, 2.0])

    summary = benchmark.summarize_timings([1.0, 2.0, 3.0, 4.0])
    assert summary["count"] == 4
    assert summary["median_seconds"] == pytest.approx(2.5)
    assert summary["mean_seconds"] == pytest.approx(2.5)
    assert summary["iqr_seconds"] == pytest.approx(1.5)

    one = benchmark.summarize_timings([2.0])
    assert one["std_seconds"] == 0.0


def test_scan_nll(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([1.0, 1.5])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    values, duration = benchmark.scan_nll(lambda x: x * x, [1.0, 2.0])
    assert values == [1.0, 4.0]
    assert duration == pytest.approx(0.5)


def test_assert_scans_repeat() -> None:
    benchmark.assert_scans_repeat(
        [1.0, 2.0],
        [1.0, 2.0 + 1e-12],
        engine_name="x",
        run_index=1,
        absolute_tolerance=1e-10,
    )
    with pytest.raises(benchmark.ValidationFailure, match="differs"):
        benchmark.assert_scans_repeat(
            [1.0, 2.0],
            [1.0, 3.0],
            engine_name="x",
            run_index=1,
            absolute_tolerance=1e-10,
        )


def test_build_steady_state_poi_values() -> None:
    values = benchmark.build_steady_state_poi_values(
        [0.0, 0.5, 1.0, 1.5, 2.0],
        6,
        avoid_first=1.0,
    )
    assert len(values) == 6
    assert all(a != b for a, b in zip(values, values[1:]))
    assert values[0] != 1.0

    with pytest.raises(ValueError, match="must not be empty"):
        benchmark.build_steady_state_poi_values([], 1)
    with pytest.raises(ValueError, match="must be positive"):
        benchmark.build_steady_state_poi_values([1.0], 0)


def test_measure_engine_pyhs3_success(monkeypatch: pytest.MonkeyPatch) -> None:
    case = make_pyhs3_case(nll_mode="logpdf")
    spec = benchmark.EngineSpec(
        name="pyhs3_compiled",
        build_func=lambda: case,
        eval_func=lambda c, value: value**2 + 1.0,
        restore_func=lambda c: None,
        operational_definition="test",
    )
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 120.0)
    counter = iter(np.linspace(0.0, 10.0, 1000))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: float(next(counter)))

    result = benchmark.measure_engine(
        spec=spec,
        scan_values=[0.0, 1.0, 2.0],
        n_warmup_evaluations=1,
        n_evaluation_runs=2,
        n_scan_runs=2,
        poi_value=1.0,
        repeat_tolerance=1e-10,
    )
    assert result["status"] == "success"
    assert result["steady_state_uses_changing_poi"] is True
    assert result["n_scan_runs"] == 2
    assert result["parameters_restored"] is True


def test_measure_engine_xroofit_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    case = make_xroofit_case()
    spec = benchmark.EngineSpec(
        name="xroofit",
        build_func=lambda: case,
        eval_func=lambda c, value: value + 1.0,
        restore_func=lambda c: None,
        operational_definition="test",
    )
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 120.0)
    counter = iter(np.linspace(0.0, 10.0, 1000))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: float(next(counter)))

    result = benchmark.measure_engine(
        spec=spec,
        scan_values=[0.0, 1.0],
        n_warmup_evaluations=0,
        n_evaluation_runs=1,
        n_scan_runs=1,
        poi_value=1.0,
        repeat_tolerance=1e-10,
    )
    assert result["xroofit_runtime_verified"] is True
    assert result["direct_roofit_create_nll_used"] is False
    assert "xRooNode" in result["xroofit_api_path"]


def test_measure_engine_unstable_repeated_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_pyhs3_case()
    calls = {"count": 0}

    def evaluate(c: Any, value: float) -> float:
        calls["count"] += 1
        return value + calls["count"] * 0.1

    spec = benchmark.EngineSpec(
        name="pyhs3_noncompiled",
        build_func=lambda: case,
        eval_func=evaluate,
        restore_func=lambda c: None,
        operational_definition="test",
    )
    monkeypatch.setattr(benchmark.gc, "collect", lambda: None)
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 0.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 0.0)
    counter = iter(np.linspace(0.0, 10.0, 1000))
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: float(next(counter)))

    with pytest.raises(benchmark.ValidationFailure, match="not stable"):
        benchmark.measure_engine(
            spec=spec,
            scan_values=[0.0, 1.0],
            n_warmup_evaluations=0,
            n_evaluation_runs=1,
            n_scan_runs=1,
            poi_value=1.0,
            repeat_tolerance=1e-12,
        )


def test_pairwise_agreement_success_and_failures() -> None:
    left = make_engine_result("pyhs3_noncompiled", [2.0, 1.0, 3.0])
    right = make_engine_result("pyhs3_compiled", [2.0, 1.0, 3.0])
    agreement = benchmark.pairwise_agreement(
        left_result=left,
        right_result=right,
        delta_tolerance=1e-6,
        delta_relative_tolerance=1e-7,
        minimum_tolerance=1e-12,
        raw_tolerance=1e-10,
    )
    assert agreement["validation_status"] == "success"
    assert agreement["minimum_index_match"] is True

    shifted = make_engine_result("xroofit", [12.0, 11.0, 13.0])
    agreement = benchmark.pairwise_agreement(
        left_result=left,
        right_result=shifted,
        delta_tolerance=1e-6,
        delta_relative_tolerance=1e-7,
        minimum_tolerance=1e-12,
        raw_tolerance=None,
    )
    assert agreement["validation_status"] == "success"
    assert agreement["constant_offset_estimate"] == pytest.approx(10.0)

    mismatched = make_engine_result("xroofit", [1.0, 4.0])
    with pytest.raises(ValueError, match="different shapes"):
        benchmark.pairwise_agreement(
            left_result=left,
            right_result=mismatched,
            delta_tolerance=1e-6,
            delta_relative_tolerance=1e-7,
            minimum_tolerance=1e-12,
            raw_tolerance=None,
        )


def test_build_all_agreements() -> None:
    missing = benchmark.build_all_agreements(
        {"pyhs3_noncompiled": make_engine_result("pyhs3_noncompiled", [1, 2])},
        delta_tolerance=1e-6,
        delta_relative_tolerance=1e-7,
        absolute_pyhs3_tolerance=1e-10,
        minimum_tolerance=1e-12,
    )
    assert missing["validation_status"] == "not_run"
    assert "xroofit" in missing["missing_engines"]

    successful = {
        "pyhs3_noncompiled": make_engine_result("pyhs3_noncompiled", [2.0, 1.0, 3.0]),
        "pyhs3_compiled": make_engine_result("pyhs3_compiled", [2.0, 1.0, 3.0]),
        "xroofit": make_engine_result("xroofit", [12.0, 11.0, 13.0]),
    }
    agreement = benchmark.build_all_agreements(
        successful,
        delta_tolerance=1e-6,
        delta_relative_tolerance=1e-7,
        absolute_pyhs3_tolerance=1e-10,
        minimum_tolerance=1e-12,
    )
    assert agreement["validation_status"] == "success"
    assert len(agreement["comparisons"]) == 3


def test_failed_engine_result() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        result = benchmark.failed_engine_result("xroofit", exc)
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert "boom" in result["error_message"]
    assert "RuntimeError" in result["traceback"]


def test_save_figure_png_only(tmp_path: Path) -> None:
    fig, _ = benchmark.plt.subplots()
    output = tmp_path / "figure"
    benchmark._save_figure(fig, output)
    assert output.with_suffix(".png").exists()
    assert not output.with_suffix(".pdf").exists()


def test_plot_functions_generate_pngs(tmp_path: Path) -> None:
    results = {
        "pyhs3_noncompiled": make_engine_result("pyhs3_noncompiled", [2.0, 1.0, 3.0]),
        "pyhs3_compiled": make_engine_result("pyhs3_compiled", [2.0, 1.0, 3.0]),
        "xroofit": make_engine_result("xroofit", [12.0, 11.0, 13.0]),
    }
    agreement = benchmark.build_all_agreements(
        results,
        delta_tolerance=1e-6,
        delta_relative_tolerance=1e-7,
        absolute_pyhs3_tolerance=1e-10,
        minimum_tolerance=1e-12,
    )
    scan_values = [0.0, 1.0, 2.0]

    benchmark.make_profile_plot(results, scan_values, "mu", tmp_path / "profile")
    benchmark.make_residual_plot(agreement, scan_values, "mu", tmp_path / "residual")
    benchmark.make_steady_runtime_plot(results, tmp_path / "steady")
    benchmark.make_scan_runtime_plot(results, tmp_path / "scan")
    benchmark.make_phase_breakdown_plot(results, tmp_path / "phases")
    benchmark.make_agreement_plot(
        agreement,
        1e-6,
        1e-7,
        tmp_path / "agreement",
    )

    for name in ("profile", "residual", "steady", "scan", "phases", "agreement"):
        assert (tmp_path / f"{name}.png").exists()


def test_make_plots_success_and_skips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        make_engine_result("pyhs3_noncompiled", [2.0, 1.0, 3.0]),
        make_engine_result("pyhs3_compiled", [2.0, 1.0, 3.0]),
        make_engine_result("xroofit", [12.0, 11.0, 13.0]),
    ]
    agreement = benchmark.build_all_agreements(
        {item["engine"]: item for item in results},
        delta_tolerance=1e-6,
        delta_relative_tolerance=1e-7,
        absolute_pyhs3_tolerance=1e-10,
        minimum_tolerance=1e-12,
    )
    output = {
        "results": results,
        "agreement": agreement,
        "scan_values": [0.0, 1.0, 2.0],
        "poi": "mu",
        "delta_tolerance": 1e-6,
        "delta_relative_tolerance": 1e-7,
    }
    benchmark.make_plots(output, tmp_path)
    assert (tmp_path / "delta_nll_profile.png").exists()

    output["results"] = results[:2]
    benchmark.make_plots(output, tmp_path / "missing")
    assert "Skipping plots" in capsys.readouterr().out

    output["results"] = results
    output["agreement"] = {"validation_status": "not_run"}
    benchmark.make_plots(output, tmp_path / "not_run")
    assert "Skipping agreement plots" in capsys.readouterr().out


def test_print_helpers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = make_engine_result("pyhs3_compiled", [2.0, 1.0, 3.0])
    result.update(
        {
            "first_nll": 1.0,
            "workspace_loading_time_seconds": 0.1,
            "model_construction_time_seconds": 0.2,
            "nll_construction_time_seconds": 0.3,
            "cold_first_evaluation_time_seconds": 0.4,
            "steady_state_evaluation": {
                "median_seconds": 0.01,
                "iqr_seconds": 0.001,
            },
            "full_scan": {
                "median_seconds": 0.1,
                "iqr_seconds": 0.01,
            },
            "time_per_scan_point_seconds": 0.001,
            "current_rss_delta_mb": 1.0,
            "peak_rss_delta_mb": 2.0,
        }
    )
    benchmark.print_result(result)
    assert "PyHS3 compiled" in capsys.readouterr().out

    failed = {
        "engine": "xroofit",
        "engine_label": "xRooFit",
        "status": "failed",
        "error_type": "RuntimeError",
        "error_message": "boom",
    }
    benchmark.print_result(failed)
    assert "boom" in capsys.readouterr().out

    agreement = benchmark.build_all_agreements(
        {
            "pyhs3_noncompiled": make_engine_result(
                "pyhs3_noncompiled", [2.0, 1.0, 3.0]
            ),
            "pyhs3_compiled": make_engine_result("pyhs3_compiled", [2.0, 1.0, 3.0]),
            "xroofit": make_engine_result("xroofit", [12.0, 11.0, 13.0]),
        },
        delta_tolerance=1e-6,
        delta_relative_tolerance=1e-7,
        absolute_pyhs3_tolerance=1e-10,
        minimum_tolerance=1e-12,
    )
    benchmark.print_agreement(agreement)
    assert "Numerical agreement" in capsys.readouterr().out


def test_run_orchestration_success(
    monkeypatch: pytest.MonkeyPatch,
    json_workspace_path: Path,
    root_workspace_path: Path,
    tmp_path: Path,
) -> None:
    def fake_measure_engine(
        *, spec: benchmark.EngineSpec, **kwargs: Any
    ) -> dict[str, Any]:
        if spec.name == "xroofit":
            return make_engine_result("xroofit", [12.0, 11.0, 13.0])
        return make_engine_result(spec.name, [2.0, 1.0, 3.0])

    saved: dict[str, Any] = {}
    monkeypatch.setattr(benchmark, "measure_engine", fake_measure_engine)
    monkeypatch.setattr(
        benchmark,
        "save_json",
        lambda data, path: saved.update({"data": data, "path": path}),
    )
    monkeypatch.setattr(benchmark, "make_plots", lambda data, path: None)

    result = benchmark.run(
        **valid_run_kwargs(
            json_workspace_path,
            root_workspace_path,
            tmp_path,
            pyhs3_combined=True,
            pyhs3_channels="ch0,ch1",
            plot=True,
        )
    )
    assert result["status"] == "success"
    assert result["pyhs3_channels"] == ["ch0", "ch1"]
    assert result["agreement"]["validation_status"] == "success"
    assert saved["path"] == tmp_path / "result.json"


def test_run_orchestration_failed_engine(
    monkeypatch: pytest.MonkeyPatch,
    json_workspace_path: Path,
    root_workspace_path: Path,
    tmp_path: Path,
) -> None:
    def fake_measure_engine(
        *, spec: benchmark.EngineSpec, **kwargs: Any
    ) -> dict[str, Any]:
        if spec.name == "xroofit":
            raise RuntimeError("boom")
        return make_engine_result(spec.name, [2.0, 1.0, 3.0])

    monkeypatch.setattr(benchmark, "measure_engine", fake_measure_engine)
    monkeypatch.setattr(benchmark, "save_json", lambda data, path: None)

    result = benchmark.run(
        **valid_run_kwargs(
            json_workspace_path,
            root_workspace_path,
            tmp_path,
            pyhs3_combined=True,
            pyhs3_channels="ch0",
        )
    )
    assert result["status"] == "failed"
    assert result["agreement"]["validation_status"] == "not_run"
    failed = next(item for item in result["results"] if item["engine"] == "xroofit")
    assert failed["error_message"] == "boom"


def test_run_infers_combined_channels(
    monkeypatch: pytest.MonkeyPatch,
    json_workspace_path: Path,
    root_workspace_path: Path,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(benchmark, "infer_combined_channels", lambda path: ["ch0"])
    monkeypatch.setattr(
        benchmark,
        "measure_engine",
        lambda spec, **kwargs: (
            make_engine_result("xroofit", [12.0, 11.0, 13.0])
            if spec.name == "xroofit"
            else make_engine_result(spec.name, [2.0, 1.0, 3.0])
        ),
    )
    monkeypatch.setattr(benchmark, "save_json", lambda data, path: None)

    result = benchmark.run(
        **valid_run_kwargs(
            json_workspace_path,
            root_workspace_path,
            tmp_path,
            pyhs3_combined=False,
            pyhs3_channels=None,
            xroofit_model_name="pdfs/sim_pdf",
            xroofit_dataset_name="combData",
        )
    )
    assert result["pyhs3_combined"] is True
    assert result["pyhs3_channels"] == ["ch0"]


def test_run_rejects_empty_combined_channels(
    json_workspace_path: Path,
    root_workspace_path: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires at least one channel"):
        benchmark.run(
            **valid_run_kwargs(
                json_workspace_path,
                root_workspace_path,
                tmp_path,
                pyhs3_combined=True,
                pyhs3_channels=", ,",
            )
        )


def test_parse_args_from(
    json_workspace_path: Path,
    root_workspace_path: Path,
) -> None:
    args = benchmark.parse_args_from(
        [
            "--json-workspace",
            str(json_workspace_path),
            "--root-workspace",
            str(root_workspace_path),
            "--pyhs3-combined",
            "--plot",
            "--n-scan-points",
            "5",
            "--delta-relative-tolerance",
            "1e-8",
        ]
    )
    assert args.json_workspace == json_workspace_path
    assert args.root_workspace == root_workspace_path
    assert args.pyhs3_combined is True
    assert args.plot is True
    assert args.n_scan_points == 5
    assert args.delta_relative_tolerance == pytest.approx(1e-8)


def test_main_dispatches_run(
    monkeypatch: pytest.MonkeyPatch,
    json_workspace_path: Path,
    root_workspace_path: Path,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        benchmark,
        "run",
        lambda **kwargs: captured.update(kwargs) or {},
    )
    benchmark.main(
        [
            "--json-workspace",
            str(json_workspace_path),
            "--root-workspace",
            str(root_workspace_path),
            "--xroofit-library",
            "",
        ]
    )
    assert captured["json_path"] == json_workspace_path
    assert captured["root_path"] == root_workspace_path
    assert captured["xroofit_library"] is None
