from __future__ import annotations

import math
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
    path = tmp_path / "5ch_example.json"
    path.write_text("{}")
    return path


@pytest.fixture
def root_workspace_path(tmp_path: Path) -> Path:
    path = tmp_path / "5ch_example.root"
    path.write_text("root")
    return path


def make_config(workspace_path: Path, **overrides: Any) -> benchmark.NLLScanConfig:
    data = {
        "framework": "pyhs3",
        "workspace_path": workspace_path,
        "root_workspace_path": None,
        "analysis": "L_ch0",
        "target": "model_ch0",
        "pyhs3_data_name": "combData_ch0",
        "root_pdf_name": "model_ch0",
        "root_data_name": "combData_ch0",
        "parameter_point": None,
        "observable_name": "x",
        "observable_index": 0,
        "poi": "mu_sig",
        "mode": "FAST_RUN",
        "mu_grid": [0.0, 1.0, 2.0],
        "shape_tolerance": 1e-7,
        "minimum_tolerance": 1e-12,
        "reference_delta_nll": None,
        "reference_minimum_mu": None,
    }
    data.update(overrides)
    return benchmark.NLLScanConfig(**data)


def make_result(
    framework: str = "pyhs3",
    *,
    workspace: str = "5ch_example.json",
    status: str = "success",
    validation_status: str = "success",
    shape_diff: float = 0.0,
    minimum_diff: float = 0.0,
    time_per_point: float = 1e-6,
) -> dict[str, Any]:
    if status != "success":
        return {
            "benchmark": benchmark.BENCHMARK_NAME,
            "framework": framework,
            "framework_label": benchmark._framework_label(framework),
            "workspace": workspace,
            "workspace_label": benchmark.workspace_label(Path(workspace)),
            "status": status,
            "validation_status": "not_run",
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
        "analysis": "L_ch0",
        "target": "model_ch0",
        "pyhs3_data_name": "combData_ch0",
        "root_pdf_name": "model_ch0",
        "root_data_name": "combData_ch0",
        "observable_name": "x",
        "observable_index": 0,
        "poi": "mu_sig",
        "mode": "FAST_RUN",
        "status": status,
        "n_points": 3,
        "mu_min": 0.0,
        "mu_max": 2.0,
        "cold_first_nll": 2.0,
        "first_nll": 2.0,
        "nll_values": [2.0, 0.0, 1.0],
        "delta_nll_shape": [2.0, 0.0, 1.0],
        "minimum_mu": 1.0,
        "model_build_time_seconds": 0.002,
        "cold_first_evaluation_time_seconds": 0.001,
        "warmup_time_seconds": 0.001,
        "first_evaluation_time_seconds": 0.0005,
        "full_scan_time_seconds": time_per_point * 3,
        "time_per_scan_point_seconds": time_per_point,
        "current_rss_before_mb": 100.0,
        "current_rss_after_mb": 101.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 120.0,
        "peak_rss_after_mb": 122.0,
        "peak_rss_delta_mb": 2.0,
        "rss_delta_mb": 1.0,
        "nll_summary": {"min": 0.0, "max": 2.0, "mean": 1.0},
        "delta_nll_summary": {"min": 0.0, "max": 2.0, "mean": 1.0},
        "reference_framework": benchmark.REFERENCE_FRAMEWORK,
        "constant_offset_estimate": 0.0,
        "delta_nll_shape_max_abs_diff": shape_diff,
        "minimum_mu_abs_diff": minimum_diff,
        "delta_nll_shape_success": shape_diff <= 1e-7,
        "minimum_mu_success": minimum_diff <= 1e-12,
        "validation_status": validation_status,
    }


@pytest.fixture
def successful_results() -> list[dict[str, Any]]:
    return [
        make_result("pyhs3", workspace="5ch_example.json", time_per_point=1e-6),
        make_result("roofit", workspace="5ch_example.json", time_per_point=2e-6),
        make_result("pyhs3", workspace="10ch_example.json", time_per_point=1.5e-6),
        make_result("roofit", workspace="10ch_example.json", time_per_point=2.5e-6),
    ]


class FakeModel:
    def __init__(
        self, values: list[float], free_params: dict[str, Any] | None = None
    ) -> None:
        self.values = values
        self.free_params = free_params or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def logpdf(self, target: str, **params: Any) -> list[float]:
        self.calls.append((target, params))
        return self.values


class FakeRootObject:
    def __init__(self, name: str = "obj", truthy: bool = True) -> None:
        self._name = name
        self.truthy = truthy
        self.value = 0.0

    def __bool__(self) -> bool:
        return self.truthy

    def GetName(self) -> str:
        return self._name

    def setVal(self, value: float) -> None:
        self.value = float(value)


class FakeRootCollection:
    def __init__(self, objects: list[Any]) -> None:
        self.objects = objects

    def __iter__(self):
        return iter(self.objects)


class FakeIteratorOnlyCollection:
    def __init__(self, objects: list[Any]) -> None:
        self.objects = objects

    def __iter__(self):
        raise TypeError

    def createIterator(self):
        objects = list(self.objects)

        class Iterator:
            def Next(self_inner):
                return objects.pop(0) if objects else None

        return Iterator()


class FakeNormSet:
    def __init__(self, *objects: Any) -> None:
        self.objects = objects

    def __bool__(self) -> bool:
        return True

    def getSize(self) -> int:
        return len(self.objects)


class FakePdf:
    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.index = 0

    def getVal(self, _norm_set: Any) -> float:
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


class FakeRootWorkspace:
    def __init__(self) -> None:
        self.variables: dict[str, Any] = {
            "x": FakeRootObject("x"),
            "mu_sig": FakeRootObject("mu_sig"),
        }
        self.pdfs: dict[str, Any] = {"model_ch0": FakePdf([0.5, 0.25])}
        self.datasets: dict[str, Any] = {"combData_ch0": FakeRootObject("combData_ch0")}

    def var(self, name: str) -> Any:
        return self.variables.get(name)

    def pdf(self, name: str) -> Any:
        return self.pdfs.get(name)

    def data(self, name: str) -> Any:
        return self.datasets.get(name)

    def allPdfs(self) -> Any:
        return FakeRootCollection([FakeRootObject(name) for name in self.pdfs])

    def allVars(self) -> Any:
        return FakeRootCollection([FakeRootObject(name) for name in self.variables])

    def allData(self) -> Any:
        return [FakeRootObject(name) for name in self.datasets]


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
    class RooFit:
        ERROR = 1

    class RooMsgService:
        @staticmethod
        def instance() -> Any:
            return SimpleNamespace(setGlobalKillBelow=lambda level: None)

    RooArgSet = FakeNormSet

    class TFile:
        opened_file = FakeRootFile()

        @staticmethod
        def Open(_path: str, _mode: str) -> FakeRootFile:
            return FakeRootModule.TFile.opened_file


def test_name_helpers() -> None:
    path = Path("inputs/5ch_model.json")
    assert benchmark.workspace_stem(path) == "5ch_model"
    assert benchmark.workspace_stem(Path("inputs/5ch_model.root")) == "5ch_model"
    assert benchmark.workspace_label(path) == "5ch\nmodel"
    assert benchmark.workspace_title("5ch_model") == "5ch / model"
    assert benchmark.default_root_workspace_path(path) == Path("inputs/5ch_model.root")


def test_analysis_helpers() -> None:
    assert benchmark.channel_from_analysis("L_ch0") == "ch0"
    assert benchmark.default_target_from_analysis("L_ch0") == "model_ch0"
    assert benchmark.default_data_name_from_analysis("L_ch0") == "combData_ch0"
    with pytest.raises(
        benchmark.BenchmarkConfigurationError, match="Cannot infer channel"
    ):
        benchmark.channel_from_analysis("bad")


def test_framework_style_helpers() -> None:
    assert benchmark._framework_label("pyhs3") == "PyHS3"
    assert benchmark._framework_label("unknown") == "unknown"
    assert benchmark._style_for("roofit")["label"] == "RooFit"
    assert benchmark._style_for("unknown")["marker"] == "o"


@pytest.mark.parametrize(
    ("mu_min", "mu_max", "n_points", "expected"),
    [(0.0, 1.0, 2, [0.0, 1.0]), (0.0, 2.0, 3, [0.0, 1.0, 2.0])],
)
def test_build_mu_grid_success(
    mu_min: float, mu_max: float, n_points: int, expected: list[float]
) -> None:
    assert benchmark.build_mu_grid(mu_min, mu_max, n_points) == expected


@pytest.mark.parametrize(
    ("mu_min", "mu_max", "n_points", "message"),
    [
        (float("nan"), 1.0, 3, "finite"),
        (0.0, float("inf"), 3, "finite"),
        (2.0, 2.0, 3, "smaller"),
        (2.0, 1.0, 3, "smaller"),
        (0.0, 1.0, 1, "at least 2"),
    ],
)
def test_build_mu_grid_rejects_invalid_values(
    mu_min: float, mu_max: float, n_points: int, message: str
) -> None:
    with pytest.raises(benchmark.BenchmarkConfigurationError, match=message):
        benchmark.build_mu_grid(mu_min, mu_max, n_points)


def valid_validate_config(workspace_path: Path, **overrides: Any) -> dict[str, Any]:
    config = {
        "frameworks": ["pyhs3"],
        "workspaces": [workspace_path],
        "root_workspaces": None,
        "analysis": "L_ch0",
        "target": "model_ch0",
        "pyhs3_data_name": "combData_ch0",
        "root_pdf_name": "model_ch0",
        "root_data_name": "combData_ch0",
        "observable_name": "x",
        "observable_index": 0,
        "poi": "mu_sig",
        "mode": "FAST_RUN",
        "mu_grid": [0.0, 1.0],
        "shape_tolerance": 1e-7,
        "minimum_tolerance": 1e-12,
    }
    config.update(overrides)
    return config


def test_validate_benchmark_config_success(workspace_path: Path) -> None:
    benchmark.validate_benchmark_config(**valid_validate_config(workspace_path))


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        (
            {"frameworks": []},
            benchmark.BenchmarkConfigurationError,
            "At least one framework",
        ),
        (
            {"frameworks": ["bad"]},
            benchmark.BenchmarkConfigurationError,
            "Unsupported framework",
        ),
        (
            {"workspaces": []},
            benchmark.BenchmarkConfigurationError,
            "At least one workspace",
        ),
        (
            {"root_workspaces": [Path("a.root"), Path("b.root")]},
            benchmark.BenchmarkConfigurationError,
            "same number",
        ),
        ({"analysis": ""}, benchmark.BenchmarkConfigurationError, "analysis"),
        ({"target": ""}, benchmark.BenchmarkConfigurationError, "target"),
        (
            {"pyhs3_data_name": ""},
            benchmark.BenchmarkConfigurationError,
            "pyhs3_data_name",
        ),
        ({"root_pdf_name": ""}, benchmark.BenchmarkConfigurationError, "root_pdf_name"),
        (
            {"root_data_name": ""},
            benchmark.BenchmarkConfigurationError,
            "root_data_name",
        ),
        (
            {"observable_name": ""},
            benchmark.BenchmarkConfigurationError,
            "observable_name",
        ),
        ({"poi": ""}, benchmark.BenchmarkConfigurationError, "poi"),
        ({"mode": ""}, benchmark.BenchmarkConfigurationError, "mode"),
        (
            {"observable_index": -1},
            benchmark.BenchmarkConfigurationError,
            "non-negative",
        ),
        ({"mu_grid": []}, benchmark.BenchmarkConfigurationError, "must not be empty"),
        (
            {"shape_tolerance": 0.0},
            benchmark.BenchmarkConfigurationError,
            "positive and finite",
        ),
        (
            {"shape_tolerance": float("nan")},
            benchmark.BenchmarkConfigurationError,
            "positive and finite",
        ),
        (
            {"minimum_tolerance": 0.0},
            benchmark.BenchmarkConfigurationError,
            "positive and finite",
        ),
    ],
)
def test_validate_benchmark_config_rejects_bad_values(
    workspace_path: Path,
    overrides: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        benchmark.validate_benchmark_config(
            **valid_validate_config(workspace_path, **overrides)
        )


def test_validate_benchmark_config_rejects_missing_workspace(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file does not exist"):
        benchmark.validate_benchmark_config(
            **valid_validate_config(tmp_path / "missing.json")
        )


def test_validate_benchmark_config_rejects_workspace_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace path is not a file"):
        benchmark.validate_benchmark_config(**valid_validate_config(tmp_path))


def test_validate_benchmark_config_requires_root_when_roofit_requested(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", None)
    with pytest.raises(
        benchmark.BenchmarkConfigurationError, match="ROOT is not importable"
    ):
        benchmark.validate_benchmark_config(
            **valid_validate_config(workspace_path, frameworks=["roofit"])
        )


def test_validate_benchmark_config_checks_root_workspace_path(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    with pytest.raises(FileNotFoundError, match="ROOT workspace file does not exist"):
        benchmark.validate_benchmark_config(
            **valid_validate_config(
                workspace_path,
                frameworks=["roofit"],
                root_workspaces=[tmp_path / "missing.root"],
            )
        )


def fake_workspace(
    parameter_points: list[Any] | None = None, data_entries: list[Any] | None = None
) -> Any:
    return SimpleNamespace(
        parameter_points=SimpleNamespace(
            root=parameter_points if parameter_points is not None else []
        ),
        data=SimpleNamespace(root=data_entries if data_entries is not None else []),
    )


def test_extract_parameter_point_default_and_named() -> None:
    points = [
        SimpleNamespace(
            name="nominal", parameters=[SimpleNamespace(name="mu_sig", value="1.0")]
        ),
        SimpleNamespace(
            name="alt", parameters=[SimpleNamespace(name="mu_sig", value="2.0")]
        ),
    ]
    workspace = fake_workspace(parameter_points=points)
    assert benchmark.extract_parameter_point(workspace, None)[
        "mu_sig"
    ] == pytest.approx(1.0)
    assert benchmark.extract_parameter_point(workspace, "alt")[
        "mu_sig"
    ] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("workspace", "parameter_point", "message"),
    [
        (SimpleNamespace(), None, "parameter_points.root"),
        (fake_workspace(parameter_points=[]), None, "parameter points"),
        (
            fake_workspace(
                parameter_points=[SimpleNamespace(name="nominal", parameters=[])]
            ),
            "missing",
            "Could not find",
        ),
        (
            fake_workspace(
                parameter_points=[
                    SimpleNamespace(
                        name="nominal",
                        parameters=[SimpleNamespace(name="mu", value="bad")],
                    )
                ]
            ),
            None,
            "cannot be converted",
        ),
    ],
)
def test_extract_parameter_point_rejects_malformed_workspace(
    workspace: Any, parameter_point: str | None, message: str
) -> None:
    with pytest.raises(benchmark.BenchmarkConfigurationError, match=message):
        benchmark.extract_parameter_point(workspace, parameter_point)


def test_get_pyhs3_data_values_success() -> None:
    workspace = fake_workspace(
        data_entries=[
            SimpleNamespace(name="other", entries=[[99.0]]),
            SimpleNamespace(name="combData_ch0", entries=[[10.0, 1.0], [11.0, 2.0]]),
        ]
    )
    assert np.allclose(
        benchmark.get_pyhs3_data_values(workspace, "combData_ch0", 0), [10.0, 11.0]
    )
    assert np.allclose(
        benchmark.get_pyhs3_data_values(workspace, "combData_ch0", 1), [1.0, 2.0]
    )


@pytest.mark.parametrize(
    ("workspace", "message"),
    [
        (SimpleNamespace(), "data.root"),
        (
            fake_workspace(
                data_entries=[SimpleNamespace(name="combData_ch0", entries=[])]
            ),
            "is empty",
        ),
        (
            fake_workspace(
                data_entries=[
                    SimpleNamespace(name="combData_ch0", entries=[[float("nan")]])
                ]
            ),
            "non-finite",
        ),
        (
            fake_workspace(
                data_entries=[SimpleNamespace(name="other", entries=[[1.0]])]
            ),
            "Could not find",
        ),
    ],
)
def test_get_pyhs3_data_values_rejects_bad_data(workspace: Any, message: str) -> None:
    with pytest.raises(benchmark.BenchmarkConfigurationError, match=message):
        benchmark.get_pyhs3_data_values(workspace, "combData_ch0", 0)


def test_prepare_pyhs3_case_merges_parameters(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    workspace = fake_workspace()
    model = SimpleNamespace(free_params={"theta": 2.5})
    workspace.model = lambda analysis, progress, mode: model

    class FakeWorkspaceLoader:
        @staticmethod
        def load(path: Path) -> Any:
            assert path == workspace_path
            return workspace

    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspaceLoader)
    monkeypatch.setattr(
        benchmark,
        "extract_parameter_point",
        lambda ws, point: {"mu_sig": np.asarray(1.0)},
    )
    monkeypatch.setattr(
        benchmark,
        "get_pyhs3_data_values",
        lambda ws, name, idx: np.asarray([10.0, 11.0]),
    )

    result_model, params = benchmark.prepare_pyhs3_case(make_config(workspace_path))

    assert result_model is model
    assert params["mu_sig"] == pytest.approx(1.0)
    assert params["theta"] == pytest.approx(2.5)
    assert np.allclose(params["x"], [10.0, 11.0])


def test_prepare_pyhs3_case_rejects_missing_poi(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    workspace = fake_workspace()
    workspace.model = lambda analysis, progress, mode: SimpleNamespace(free_params={})

    class FakeWorkspaceLoader:
        @staticmethod
        def load(path: Path) -> Any:
            return workspace

    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspaceLoader)
    monkeypatch.setattr(
        benchmark,
        "extract_parameter_point",
        lambda ws, point: {"other": np.asarray(1.0)},
    )
    monkeypatch.setattr(
        benchmark, "get_pyhs3_data_values", lambda ws, name, idx: np.asarray([10.0])
    )

    with pytest.raises(benchmark.BenchmarkConfigurationError, match="POI 'mu_sig'"):
        benchmark.prepare_pyhs3_case(make_config(workspace_path))


def test_evaluate_pyhs3_nll_success() -> None:
    model = FakeModel([math.log(0.25), math.log(0.5)])
    result = benchmark.evaluate_pyhs3_nll(
        model, {"mu_sig": 1.0, "x": np.asarray([1.0])}, "model_ch0", "mu_sig", 2.0
    )
    assert result == pytest.approx(-(math.log(0.25) + math.log(0.5)))
    assert np.asarray(model.calls[0][1]["mu_sig"]) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "empty logpdf"),
        ([float("nan")], "non-finite"),
        ([float("inf")], "non-finite"),
    ],
)
def test_evaluate_pyhs3_nll_rejects_invalid_outputs(
    values: list[float], message: str
) -> None:
    with pytest.raises(benchmark.ValidationFailure, match=message):
        benchmark.evaluate_pyhs3_nll(
            FakeModel(values), {"mu_sig": 1.0}, "model", "mu_sig", 0.0
        )


def test_root_collection_names_supports_iteration_and_iterator() -> None:
    assert benchmark._root_collection_names(
        FakeRootCollection([FakeRootObject("a"), None, FakeRootObject("b")])
    ) == ["a", "b"]
    assert benchmark._root_collection_names(
        FakeIteratorOnlyCollection([FakeRootObject("x"), FakeRootObject("y")])
    ) == ["x", "y"]


def test_candidate_names_deduplicates_and_preserves_order() -> None:
    assert benchmark._candidate_names("a", ["b", "a", "c"]) == ["a", "b", "c"]


@pytest.mark.parametrize(
    "obj, expected",
    [
        (None, False),
        (FakeRootObject(truthy=True), True),
        (FakeRootObject(truthy=False), False),
    ],
)
def test_is_valid_root_object(obj: Any, expected: bool) -> None:
    assert benchmark._is_valid_root_object(obj) is expected


def test_available_root_objects_collects_names() -> None:
    workspace = FakeRootWorkspace()
    assert benchmark._available_root_objects(workspace) == {
        "pdfs": ["model_ch0"],
        "data": ["combData_ch0"],
        "vars": ["x", "mu_sig"],
    }


def test_make_single_observable_norm_set_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    observable, norm_set = benchmark._make_single_observable_norm_set(
        FakeRootWorkspace(), "x"
    )
    assert observable.GetName() == "x"
    assert norm_set.getSize() == 1


def test_make_single_observable_norm_set_rejects_missing_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    with pytest.raises(KeyError, match="observable"):
        benchmark._make_single_observable_norm_set(FakeRootWorkspace(), "missing")


def test_get_root_pdf_and_data_support_fallbacks() -> None:
    workspace = FakeRootWorkspace()
    assert (
        benchmark._get_root_pdf(workspace, "missing", "model_ch0", "L_ch0")
        is workspace.pdfs["model_ch0"]
    )
    assert (
        benchmark._get_root_data(workspace, "missing", "combData_ch0")
        is workspace.datasets["combData_ch0"]
    )


def test_get_root_pdf_and_data_report_available_names() -> None:
    workspace = FakeRootWorkspace()
    workspace.pdfs = {}
    workspace.datasets = {}
    with pytest.raises(KeyError, match="Available PDFs"):
        benchmark._get_root_pdf(workspace, "missing", "target", "analysis")
    with pytest.raises(KeyError, match="Available data"):
        benchmark._get_root_data(workspace, "missing", "combData_ch0")


def test_set_root_defaults_from_pyhs3_best_effort(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    workspace = FakeRootWorkspace()
    monkeypatch.setattr(benchmark.Workspace, "load", lambda path: fake_workspace())
    monkeypatch.setattr(
        benchmark,
        "extract_parameter_point",
        lambda ws, point: {"mu_sig": np.asarray(1.5), "missing": np.asarray(7.0)},
    )
    benchmark._set_root_defaults_from_pyhs3(workspace, workspace_path, None)
    assert workspace.variables["mu_sig"].value == pytest.approx(1.5)


def test_prepare_roofit_case_rejects_missing_root(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", None)
    with pytest.raises(
        benchmark.BenchmarkConfigurationError, match="ROOT is not available"
    ):
        benchmark.prepare_roofit_case(make_config(workspace_path, framework="roofit"))


def test_prepare_roofit_case_success(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, root_workspace_path: Path
) -> None:
    root_file = FakeRootFile(FakeRootWorkspace())
    FakeRootModule.TFile.opened_file = root_file
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    monkeypatch.setattr(
        benchmark, "_find_root_workspace", lambda root_file_arg: root_file.workspace
    )
    monkeypatch.setattr(
        benchmark, "_set_root_defaults_from_pyhs3", lambda **kwargs: None
    )
    monkeypatch.setattr(benchmark.Workspace, "load", lambda path: fake_workspace())
    monkeypatch.setattr(
        benchmark,
        "get_pyhs3_data_values",
        lambda ws, name, idx: np.asarray([10.0, 11.0]),
    )

    keepalive, pdf, poi_var, observable, data_values, returned_file = (
        benchmark.prepare_roofit_case(
            make_config(
                workspace_path,
                framework="roofit",
                root_workspace_path=root_workspace_path,
            )
        )
    )

    assert keepalive[0] is root_file
    assert pdf is root_file.workspace.pdfs["model_ch0"]
    assert poi_var is root_file.workspace.variables["mu_sig"]
    assert observable is root_file.workspace.variables["x"]
    assert np.allclose(data_values, [10.0, 11.0])
    assert returned_file is root_file


def test_evaluate_roofit_nll_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    pdf = FakePdf([0.5, 0.25])
    poi = FakeRootObject("mu_sig")
    observable = FakeRootObject("x")
    result = benchmark.evaluate_roofit_nll(
        pdf, poi, observable, np.asarray([10.0, 11.0]), 1.75
    )
    assert poi.value == pytest.approx(1.75)
    assert observable.value == pytest.approx(11.0)
    assert result == pytest.approx(-(math.log(0.5) + math.log(0.25)))


@pytest.mark.parametrize("bad_pdf_value", [0.0, -1.0, float("nan"), float("inf")])
def test_evaluate_roofit_nll_rejects_invalid_pdf(
    monkeypatch: pytest.MonkeyPatch, bad_pdf_value: float
) -> None:
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    with pytest.raises(
        benchmark.ValidationFailure, match="invalid normalized PDF value"
    ):
        benchmark.evaluate_roofit_nll(
            FakePdf([bad_pdf_value]),
            FakeRootObject("mu"),
            FakeRootObject("x"),
            np.asarray([1.0]),
            1.0,
        )


def test_evaluate_roofit_nll_wraps_norm_set_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadRootModule:
        @staticmethod
        def RooArgSet(_observable: Any) -> Any:
            raise RuntimeError("bad norm")

    monkeypatch.setitem(sys.modules, "ROOT", BadRootModule)
    with pytest.raises(benchmark.ValidationFailure, match="Could not create"):
        benchmark.evaluate_roofit_nll(
            FakePdf([0.5]),
            FakeRootObject("mu"),
            FakeRootObject("x"),
            np.asarray([1.0]),
            1.0,
        )


def test_delta_nll_shape_minimum_and_comparison_helpers() -> None:
    assert benchmark.delta_nll_shape([3.0, 1.0, 2.0]) == [2.0, 0.0, 1.0]
    assert benchmark.minimum_position([0.0, 1.0, 2.0], [3.0, 1.0, 2.0]) == 1.0
    assert benchmark.max_abs_difference([1.0, 2.0], [2.5, 2.5]) == pytest.approx(1.5)
    assert benchmark.mean_offset([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.5)
    assert benchmark.summarize([1.0, 2.0, 3.0]) == {"min": 1.0, "max": 3.0, "mean": 2.0}


@pytest.mark.parametrize("values", [[], [1.0, float("nan")], [float("inf")]])
def test_delta_nll_shape_rejects_invalid_values(values: list[float]) -> None:
    with pytest.raises(benchmark.ValidationFailure):
        benchmark.delta_nll_shape(values)


@pytest.mark.parametrize(
    "func",
    [benchmark.minimum_position, benchmark.max_abs_difference, benchmark.mean_offset],
)
def test_array_helpers_reject_length_mismatch(func: Any) -> None:
    with pytest.raises(
        benchmark.ValidationFailure, match="same length|different lengths"
    ):
        func([1.0], [1.0, 2.0])


def test_run_single_framework_scan_pyhs3_success(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 120.0)
    monkeypatch.setattr(
        benchmark,
        "prepare_pyhs3_case",
        lambda config: ("model", {"mu_sig": np.asarray(1.0)}),
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_pyhs3_nll",
        lambda model, params, target, poi, mu: (mu - 1.0) ** 2 + 10.0,
    )

    result = benchmark.run_single_framework_scan(make_config(workspace_path))

    assert result["status"] == "success"
    assert result["framework"] == "pyhs3"
    assert result["nll_values"] == [11.0, 10.0, 11.0]
    assert result["delta_nll_shape"] == [1.0, 0.0, 1.0]
    assert result["minimum_mu"] == 1.0
    assert result["validation_status"] == "success"


def test_run_single_framework_scan_marks_validation_failure(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 120.0)
    monkeypatch.setattr(
        benchmark,
        "prepare_pyhs3_case",
        lambda config: ("model", {"mu_sig": np.asarray(1.0)}),
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_pyhs3_nll",
        lambda model, params, target, poi, mu: (mu - 2.0) ** 2,
    )

    result = benchmark.run_single_framework_scan(
        make_config(
            workspace_path,
            reference_delta_nll=[1.0, 0.0, 1.0],
            reference_minimum_mu=1.0,
            shape_tolerance=0.1,
            minimum_tolerance=0.1,
        )
    )

    assert result["validation_status"] == "failed"
    assert result["error_type"] == "ValidationFailure"
    assert result["delta_nll_shape_success"] is False
    assert result["minimum_mu_success"] is False


def test_run_single_framework_scan_roofit_closes_file(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, root_workspace_path: Path
) -> None:
    root_file = FakeRootFile()
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 100.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 120.0)
    monkeypatch.setattr(
        benchmark,
        "prepare_roofit_case",
        lambda config: (
            (root_file,),
            "pdf",
            "poi",
            "obs",
            np.asarray([1.0]),
            root_file,
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_roofit_nll",
        lambda pdf, poi, observable, data_values, mu: (mu - 1.0) ** 2,
    )

    result = benchmark.run_single_framework_scan(
        make_config(
            workspace_path, framework="roofit", root_workspace_path=root_workspace_path
        )
    )

    assert result["framework"] == "roofit"
    assert result["status"] == "success"
    assert root_file.closed is True


def test_run_single_framework_scan_rejects_unknown_framework(
    workspace_path: Path,
) -> None:
    with pytest.raises(
        benchmark.BenchmarkConfigurationError, match="Unknown framework"
    ):
        benchmark.run_single_framework_scan(
            make_config(workspace_path, framework="unknown")
        )


def test_error_result_contains_context(workspace_path: Path) -> None:
    config = make_config(
        workspace_path, framework="roofit", root_workspace_path=Path("file.root")
    )
    error = RuntimeError("boom")
    result = benchmark.error_result(config, error)
    assert result["status"] == "error"
    assert result["validation_status"] == "not_run"
    assert result["framework"] == "roofit"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "boom"


def test_successful_results_and_summarize_status() -> None:
    results = [
        make_result("pyhs3"),
        make_result("roofit", validation_status="failed"),
        make_result("roofit", status="error"),
    ]
    assert [row["framework"] for row in benchmark.successful_results(results)] == [
        "pyhs3",
        "roofit",
    ]
    summary = benchmark.summarize_status(results)
    assert summary["status"] == "completed_with_errors"
    assert summary["n_results"] == 3
    assert summary["n_successful"] == 2
    assert summary["n_validated"] == 1
    assert summary["n_failed"] == 2


def test_summarize_status_success() -> None:
    summary = benchmark.summarize_status([make_result("pyhs3"), make_result("roofit")])
    assert summary["status"] == "success"
    assert summary["failed_results"] == []


def test_success_dataframe(successful_results: list[dict[str, Any]]) -> None:
    df = benchmark._success_dataframe(
        successful_results + [make_result("bad", status="error")]
    )
    assert set(df["framework_key"]) == {"pyhs3", "roofit"}
    assert "us_per_point" in df.columns


def test_plot_functions_create_files(
    tmp_path: Path, successful_results: list[dict[str, Any]]
) -> None:
    mu_grid = [0.0, 1.0, 2.0]
    plot_calls = [
        (
            benchmark.make_profile_plot,
            (successful_results, mu_grid, tmp_path / "profile.png"),
        ),
        (benchmark.make_timing_plot, (successful_results, tmp_path / "timing.png")),
        (
            benchmark.make_relative_runtime_plot,
            (successful_results, tmp_path / "relative.png"),
        ),
        (benchmark.make_memory_plot, (successful_results, tmp_path / "memory.png")),
        (
            benchmark.make_agreement_plot,
            (successful_results, tmp_path / "agreement.png", 1e-7),
        ),
        (benchmark.make_summary_table, (successful_results, tmp_path / "summary.png")),
    ]
    for func, args in plot_calls:
        func(*args)
        assert args[-1].exists() if isinstance(args[-1], Path) else args[-2].exists()


def test_plot_functions_return_on_empty_inputs(tmp_path: Path) -> None:
    benchmark.make_profile_plot([], [0.0, 1.0], tmp_path / "profile.png")
    benchmark.make_timing_plot([], tmp_path / "timing.png")
    benchmark.make_relative_runtime_plot([], tmp_path / "relative.png")
    benchmark.make_memory_plot([], tmp_path / "memory.png")
    benchmark.make_agreement_plot([], tmp_path / "agreement.png", 1e-7)
    benchmark.make_summary_table([], tmp_path / "summary.png")
    assert not any(tmp_path.iterdir())


def test_make_plots_calls_all_builders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    successful_results: list[dict[str, Any]],
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        benchmark, "make_profile_plot", lambda *args, **kwargs: calls.append("profile")
    )
    monkeypatch.setattr(
        benchmark, "make_timing_plot", lambda *args, **kwargs: calls.append("timing")
    )
    monkeypatch.setattr(
        benchmark,
        "make_relative_runtime_plot",
        lambda *args, **kwargs: calls.append("relative"),
    )
    monkeypatch.setattr(
        benchmark, "make_memory_plot", lambda *args, **kwargs: calls.append("memory")
    )
    monkeypatch.setattr(
        benchmark,
        "make_agreement_plot",
        lambda *args, **kwargs: calls.append("agreement"),
    )
    monkeypatch.setattr(
        benchmark, "make_summary_table", lambda *args, **kwargs: calls.append("summary")
    )

    benchmark.make_plots(
        successful_results, [0.0, 1.0, 2.0], tmp_path, shape_tolerance=1e-7
    )
    assert calls == ["profile", "timing", "relative", "memory", "agreement", "summary"]


def test_make_plots_skips_with_too_few_successes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        benchmark,
        "make_profile_plot",
        lambda *args, **kwargs: pytest.fail("should not be called"),
    )
    benchmark.make_plots(
        [make_result("pyhs3")], [0.0, 1.0], tmp_path, shape_tolerance=1e-7
    )
    assert "Skipping plots" in capsys.readouterr().out


def test_print_result_success_and_error(capsys: pytest.CaptureFixture[str]) -> None:
    benchmark.print_result(make_result("pyhs3"))
    benchmark.print_result(make_result("roofit", status="error"))
    out = capsys.readouterr().out
    assert "PyHS3" in out
    assert "minimum mu" in out
    assert "error:" in out


def test_run_orchestrates_reference_validation_and_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workspace_path: Path
) -> None:
    saved: dict[str, Any] = {}
    calls: list[tuple[str, list[float] | None]] = []

    def fake_scan(config: benchmark.NLLScanConfig) -> dict[str, Any]:
        calls.append((config.framework, config.reference_delta_nll))
        result = make_result(config.framework, workspace=config.workspace_path.name)
        if config.reference_delta_nll is None:
            result["delta_nll_shape"] = [1.0, 0.0, 1.0]
            result["minimum_mu"] = 1.0
        return result

    def fake_save_json(data: dict[str, Any], path: Path) -> None:
        saved.update(data)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")

    monkeypatch.setattr(benchmark, "run_single_framework_scan", fake_scan)
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    (tmp_path / "workspace.root").write_text("root")
    monkeypatch.setattr(benchmark, "save_json", fake_save_json)
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda *args, **kwargs: saved.setdefault("plots_called", True),
    )

    output = benchmark.run(
        frameworks=["pyhs3", "roofit"],
        workspaces=[workspace_path],
        root_workspaces=[tmp_path / "workspace.root"],
        analysis="L_ch0",
        target=None,
        pyhs3_data_name=None,
        root_pdf_name=None,
        root_data_name=None,
        parameter_point=None,
        observable_name="x",
        observable_index=0,
        poi="mu_sig",
        mode="FAST_RUN",
        mu_min=0.0,
        mu_max=2.0,
        n_points=3,
        shape_tolerance=1e-7,
        minimum_tolerance=1e-12,
        output_dir=tmp_path,
        output_name="result.json",
        plot=True,
        plot_dir=tmp_path / "plots",
    )

    assert saved.pop("plots_called") is True
    assert output == saved
    assert output["configuration"]["target"] == "model_ch0"
    assert output["configuration"]["pyhs3_data_name"] == "combData_ch0"
    assert output["mu_grid"] == [0.0, 1.0, 2.0]
    assert calls[0] == ("pyhs3", None)
    assert calls[1][1] == [1.0, 0.0, 1.0]
    assert calls[2][1] == [1.0, 0.0, 1.0]


def test_run_records_framework_error_and_fail_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workspace_path: Path
) -> None:
    def fake_scan(config: benchmark.NLLScanConfig) -> dict[str, Any]:
        if config.reference_delta_nll is None:
            return make_result("pyhs3", workspace=config.workspace_path.name)
        if config.framework == "pyhs3":
            raise RuntimeError("boom")
        return make_result(config.framework, workspace=config.workspace_path.name)

    monkeypatch.setattr(benchmark, "run_single_framework_scan", fake_scan)
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    (tmp_path / "workspace.root").write_text("root")
    monkeypatch.setattr(
        benchmark, "save_json", lambda data, path: path.write_text("{}")
    )

    output = benchmark.run(
        frameworks=["pyhs3", "roofit"],
        workspaces=[workspace_path],
        root_workspaces=[tmp_path / "workspace.root"],
        analysis="L_ch0",
        target="model_ch0",
        pyhs3_data_name="combData_ch0",
        root_pdf_name="model_ch0",
        root_data_name="combData_ch0",
        parameter_point=None,
        observable_name="x",
        observable_index=0,
        poi="mu_sig",
        mode="FAST_RUN",
        mu_min=0.0,
        mu_max=2.0,
        n_points=3,
        shape_tolerance=1e-7,
        minimum_tolerance=1e-12,
        output_dir=tmp_path,
        output_name="result.json",
        plot=False,
        plot_dir=tmp_path / "plots",
        fail_fast=True,
    )

    assert len(output["results"]) == 1
    assert output["results"][0]["status"] == "error"
    assert output["summary"]["status"] == "completed_with_errors"


def test_parse_args_defaults_and_custom_values(
    workspace_path: Path, root_workspace_path: Path, tmp_path: Path
) -> None:
    args = benchmark.parse_args(
        [
            "--frameworks",
            "pyhs3",
            "roofit",
            "--workspaces",
            str(workspace_path),
            "--root-workspaces",
            str(root_workspace_path),
            "--analysis",
            "L_ch1",
            "--target",
            "model_ch1",
            "--pyhs3-data-name",
            "combData_ch1",
            "--root-pdf-name",
            "pdf_ch1",
            "--root-data-name",
            "data_ch1",
            "--parameter-point",
            "nominal",
            "--observable-name",
            "mass",
            "--observable-index",
            "1",
            "--poi",
            "mu",
            "--mode",
            "FAST_COMPILE",
            "--mu-min",
            "0.5",
            "--mu-max",
            "1.5",
            "--n-points",
            "5",
            "--shape-tolerance",
            "1e-6",
            "--minimum-tolerance",
            "1e-9",
            "--output-dir",
            str(tmp_path),
            "--output-name",
            "out.json",
            "--plot",
            "--plot-dir",
            str(tmp_path / "plots"),
            "--fail-fast",
        ]
    )

    assert args.frameworks == ["pyhs3", "roofit"]
    assert args.workspaces == [workspace_path]
    assert args.root_workspaces == [root_workspace_path]
    assert args.analysis == "L_ch1"
    assert args.target == "model_ch1"
    assert args.pyhs3_data_name == "combData_ch1"
    assert args.root_pdf_name == "pdf_ch1"
    assert args.root_data_name == "data_ch1"
    assert args.parameter_point == "nominal"
    assert args.observable_name == "mass"
    assert args.observable_index == 1
    assert args.poi == "mu"
    assert args.mode == "FAST_COMPILE"
    assert args.mu_min == pytest.approx(0.5)
    assert args.mu_max == pytest.approx(1.5)
    assert args.n_points == 5
    assert args.plot is True
    assert args.fail_fast is True


def test_parse_args_rejects_unknown_framework() -> None:
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--frameworks", "bad"])


def test_main_calls_run(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(benchmark, "run", fake_run)
    benchmark.main(
        [
            "--frameworks",
            "pyhs3",
            "--workspaces",
            str(workspace_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert captured["frameworks"] == ["pyhs3"]
    assert captured["workspaces"] == [workspace_path]
    assert captured["root_workspaces"] is None
    assert captured["analysis"] == "L_ch0"


def test_validate_benchmark_config_rejects_root_workspace_directory(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, tmp_path: Path
) -> None:
    root_dir = tmp_path / "root_dir.root"
    root_dir.mkdir()
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    with pytest.raises(FileNotFoundError, match="ROOT workspace path is not a file"):
        benchmark.validate_benchmark_config(
            **valid_validate_config(
                workspace_path,
                frameworks=["roofit"],
                root_workspaces=[root_dir],
            )
        )


def test_prepare_pyhs3_case_allows_model_without_free_params(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    workspace = fake_workspace()
    model = object()
    workspace.model = lambda analysis, progress, mode: model

    class FakeWorkspaceLoader:
        @staticmethod
        def load(path: Path) -> Any:
            return workspace

    monkeypatch.setattr(benchmark, "Workspace", FakeWorkspaceLoader)
    monkeypatch.setattr(
        benchmark,
        "extract_parameter_point",
        lambda ws, point: {"mu_sig": np.asarray(1.0)},
    )
    monkeypatch.setattr(
        benchmark, "get_pyhs3_data_values", lambda ws, name, idx: np.asarray([10.0])
    )

    result_model, params = benchmark.prepare_pyhs3_case(make_config(workspace_path))
    assert result_model is model
    assert "x" in params


class BoolRaisesRootObject(FakeRootObject):
    def __bool__(self) -> bool:
        raise RuntimeError("bool broken")


def test_is_valid_root_object_treats_bool_errors_as_valid() -> None:
    assert benchmark._is_valid_root_object(BoolRaisesRootObject()) is True


class FakeRootKey:
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def ReadObj(self) -> Any:
        return self.obj


class FakeWorkspaceClass:
    @staticmethod
    def Class() -> str:
        return "RooWorkspaceClass"


class FakeRootWorkspaceObject:
    def InheritsFrom(self, cls: Any) -> bool:
        return cls == "RooWorkspaceClass"


class FakeNonWorkspaceObject:
    def InheritsFrom(self, cls: Any) -> bool:
        return False


def test_find_root_workspace_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RootWithWorkspace:
        RooWorkspace = FakeWorkspaceClass

    monkeypatch.setitem(sys.modules, "ROOT", RootWithWorkspace)
    root_file = SimpleNamespace(
        GetListOfKeys=lambda: [
            FakeRootKey(FakeNonWorkspaceObject()),
            FakeRootKey(FakeRootWorkspaceObject()),
        ]
    )
    assert isinstance(
        benchmark._find_root_workspace(root_file), FakeRootWorkspaceObject
    )

    root_file = SimpleNamespace(
        GetListOfKeys=lambda: [FakeRootKey(FakeNonWorkspaceObject())]
    )
    with pytest.raises(KeyError, match="No RooWorkspace"):
        benchmark._find_root_workspace(root_file)


class BrokenRootWorkspace(FakeRootWorkspace):
    def allPdfs(self) -> Any:
        raise RuntimeError("pdfs broken")

    def allVars(self) -> Any:
        raise RuntimeError("vars broken")

    def allData(self) -> Any:
        raise RuntimeError("data broken")


def test_available_root_objects_handles_collection_errors() -> None:
    assert benchmark._available_root_objects(BrokenRootWorkspace()) == {
        "pdfs": [],
        "data": [],
        "vars": [],
    }


def test_make_single_observable_norm_set_rejects_bad_norm_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadNormSet(FakeNormSet):
        def __bool__(self) -> bool:
            return False

    class BadRootModule(FakeRootModule):
        RooArgSet = BadNormSet

    monkeypatch.setitem(sys.modules, "ROOT", BadRootModule)
    with pytest.raises(KeyError, match="normalization set"):
        benchmark._make_single_observable_norm_set(FakeRootWorkspace(), "x")


def test_set_root_defaults_from_pyhs3_ignores_load_and_set_errors(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path
) -> None:
    workspace = FakeRootWorkspace()
    monkeypatch.setattr(
        benchmark.Workspace,
        "load",
        lambda path: (_ for _ in ()).throw(RuntimeError("load failed")),
    )
    benchmark._set_root_defaults_from_pyhs3(workspace, workspace_path, None)

    class BadVar(FakeRootObject):
        def setVal(self, value: float) -> None:
            raise RuntimeError("set failed")

    workspace.variables["mu_sig"] = BadVar("mu_sig")
    monkeypatch.setattr(benchmark.Workspace, "load", lambda path: fake_workspace())
    monkeypatch.setattr(
        benchmark,
        "extract_parameter_point",
        lambda ws, point: {"mu_sig": np.asarray(1.5)},
    )
    benchmark._set_root_defaults_from_pyhs3(workspace, workspace_path, None)


def test_prepare_roofit_case_rejects_missing_root_path_and_zombie(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, root_workspace_path: Path
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    with pytest.raises(benchmark.BenchmarkConfigurationError, match="matching ROOT"):
        benchmark.prepare_roofit_case(
            make_config(workspace_path, framework="roofit", root_workspace_path=None)
        )

    FakeRootModule.TFile.opened_file = FakeRootFile(zombie=True)
    with pytest.raises(FileNotFoundError, match="Could not open ROOT file"):
        benchmark.prepare_roofit_case(
            make_config(
                workspace_path,
                framework="roofit",
                root_workspace_path=root_workspace_path,
            )
        )


def test_prepare_roofit_case_closes_file_on_setup_error(
    monkeypatch: pytest.MonkeyPatch, workspace_path: Path, root_workspace_path: Path
) -> None:
    root_file = FakeRootFile(FakeRootWorkspace())
    root_file.workspace.variables.pop("mu_sig")
    FakeRootModule.TFile.opened_file = root_file
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)
    monkeypatch.setattr(
        benchmark, "_find_root_workspace", lambda root_file_arg: root_file.workspace
    )
    monkeypatch.setattr(
        benchmark, "_set_root_defaults_from_pyhs3", lambda **kwargs: None
    )
    with pytest.raises(KeyError, match="POI"):
        benchmark.prepare_roofit_case(
            make_config(
                workspace_path,
                framework="roofit",
                root_workspace_path=root_workspace_path,
            )
        )
    assert root_file.closed is True


def test_evaluate_pyhs3_nll_rejects_nonfinite_sum() -> None:
    class HugeModel:
        def logpdf(self, target: str, **params: Any) -> list[float]:
            return [1e309]

    with pytest.raises(benchmark.ValidationFailure, match="non-finite logpdf"):
        benchmark.evaluate_pyhs3_nll(
            HugeModel(), {"mu_sig": 1.0}, "model", "mu_sig", 1.0
        )


def test_evaluate_roofit_nll_rejects_nonfinite_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "ROOT", FakeRootModule)

    class GoodPdf:
        def getVal(self, norm_set: Any) -> float:
            return 0.5

    monkeypatch.setattr(benchmark.math, "log", lambda value: float("-inf"))
    with pytest.raises(benchmark.ValidationFailure, match="RooFit NLL is non-finite"):
        benchmark.evaluate_roofit_nll(
            GoodPdf(), FakeRootObject("mu"), FakeRootObject("x"), np.asarray([1.0]), 1.0
        )


def test_make_agreement_plot_returns_for_reference_only(tmp_path: Path) -> None:
    path = tmp_path / "agreement.png"
    benchmark.make_agreement_plot([make_result("pyhs3")], path, 1e-7)
    assert not path.exists()


def test_make_summary_table_alternating_rows(tmp_path: Path) -> None:
    path = tmp_path / "table.png"
    rows = [
        make_result("pyhs3", workspace="a.json"),
        make_result("roofit", workspace="a.json"),
        make_result("pyhs3", workspace="b.json"),
    ]
    benchmark.make_summary_table(rows, path)
    assert path.exists()


def test_run_orchestrates_first_case_with_root_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, workspace_path: Path
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    root_path = tmp_path / "workspace.root"
    root_path.write_text("root")
    monkeypatch.setattr(
        benchmark,
        "run_single_framework_scan",
        lambda config: make_result(
            config.framework, workspace=config.workspace_path.name
        ),
    )
    monkeypatch.setattr(
        benchmark, "save_json", lambda data, path: path.write_text("{}")
    )
    monkeypatch.setattr(benchmark, "make_plots", lambda *args, **kwargs: None)
    output = benchmark.run(
        frameworks=["pyhs3"],
        workspaces=[workspace_path],
        root_workspaces=[root_path],
        analysis="L_ch0",
        target=None,
        pyhs3_data_name=None,
        root_pdf_name=None,
        root_data_name=None,
        parameter_point=None,
        observable_name="x",
        observable_index=0,
        poi="mu_sig",
        mode="FAST_RUN",
        mu_min=0.0,
        mu_max=2.0,
        n_points=3,
        shape_tolerance=1e-7,
        minimum_tolerance=1e-12,
        output_dir=tmp_path,
        output_name="result.json",
        plot=True,
        plot_dir=tmp_path / "plots",
    )
    assert output["summary"]["status"] == "success"


def test_main_wraps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "parse_args",
        lambda argv=None: SimpleNamespace(
            frameworks=["pyhs3"],
            workspaces=[Path("missing.json")],
            root_workspaces=None,
            analysis="bad",
            target=None,
            pyhs3_data_name=None,
            root_pdf_name=None,
            root_data_name=None,
            parameter_point=None,
            observable_name="x",
            observable_index=0,
            poi="mu",
            mode="FAST_RUN",
            mu_min=0.0,
            mu_max=1.0,
            n_points=2,
            shape_tolerance=1e-7,
            minimum_tolerance=1e-12,
            output_dir=Path("."),
            output_name="out.json",
            plot=False,
            plot_dir=Path("."),
            fail_fast=False,
        ),
    )
    with pytest.raises(RuntimeError, match="did not complete"):
        benchmark.main([])
