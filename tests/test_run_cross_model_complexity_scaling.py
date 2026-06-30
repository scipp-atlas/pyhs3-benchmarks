from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from src import run_cross_model_complexity_scaling as benchmark


@pytest.fixture
def tmp_existing_dirs(tmp_path: Path) -> tuple[Path, Path]:
    json_dir = tmp_path / "json"
    root_dir = tmp_path / "root"
    json_dir.mkdir()
    root_dir.mkdir()
    return json_dir, root_dir


@pytest.fixture
def valid_framework_result() -> dict[str, Any]:
    return {
        "framework": "pyhs3",
        "status": "success",
        "n_runs": 2,
        "n_scan_points": 3,
        "build_time_seconds": 0.01,
        "cold_first_evaluation_time_seconds": 0.02,
        "warm_evaluation": {
            "mean_seconds": 0.03,
            "std_seconds": 0.001,
            "min_seconds": 0.02,
            "max_seconds": 0.04,
        },
        "warm_evaluation_time_seconds_mean": 0.03,
        "scan_time_seconds": 0.09,
        "time_per_scan_point_seconds": 0.03,
        "current_rss_before_mb": 10.0,
        "current_rss_after_mb": 11.0,
        "current_rss_delta_mb": 1.0,
        "peak_rss_before_mb": 20.0,
        "peak_rss_after_mb": 21.0,
        "peak_rss_delta_mb": 1.0,
        "first_nll": 10.0,
        "warm_nll": 10.0,
        "scan_nll_values": [2.0, 1.0, 2.0],
        "delta_nll_shape": [1.0, 0.0, 1.0],
        "minimum_mu_sig": 1.0,
        "minimum_index": 1,
        "finite_values": True,
    }


@pytest.fixture
def successful_case(valid_framework_result: dict[str, Any]) -> dict[str, Any]:
    pyhs3 = dict(valid_framework_result, framework="pyhs3")
    roofit = dict(
        valid_framework_result, framework="roofit", time_per_scan_point_seconds=0.06
    )
    agreement = benchmark.add_agreement_metrics(
        pyhs3_result=pyhs3,
        roofit_result=roofit,
        delta_tolerance=1e-9,
        minimum_tolerance=1e-12,
    )
    return {
        "case": "simple_workspace_nonp",
        "analysis": "L_ch0",
        "target": "model_ch0",
        "json_path": "json/simple_workspace_nonp.json",
        "root_path": "root/simple_workspace_nonp.root",
        "status": "success",
        "pyhs3": pyhs3,
        "roofit": roofit,
        "agreement": agreement,
    }


@pytest.fixture
def validation_failed_case(successful_case: dict[str, Any]) -> dict[str, Any]:
    case = json.loads(json.dumps(successful_case))
    case["status"] = "failed"
    case["error_type"] = "ValidationFailure"
    case["error_message"] = "Numerical agreement check failed"
    case["agreement"]["validation_status"] = "failed"
    case["agreement"]["delta_nll_difference"] = [0.0, 2e-6, 0.0]
    case["agreement"]["delta_nll_max_abs_diff"] = 2e-6
    case["agreement"]["max_delta_nll_diff_index"] = 1
    case["agreement"]["pyhs3_delta_nll_shape"] = [1.0, 0.0, 1.0]
    case["agreement"]["roofit_delta_nll_shape"] = [1.0, 2e-6, 1.0]
    return case


class FakeRow:
    def __init__(self, value: float) -> None:
        self.value = value

    def getRealValue(self, name: str) -> float:
        assert name == "x"
        return self.value


class FakeDataset:
    def __init__(self, values: list[float] | None = None) -> None:
        if values is None:
            values = [1.0, 2.0]
        self.values = values
        self.reduce_expression: str | None = None

    def reduce(self, expression: str) -> "FakeDataset":
        self.reduce_expression = expression
        return self

    def numEntries(self) -> int:
        return len(self.values)

    def get(self, index: int) -> FakeRow:
        return FakeRow(self.values[index])


class FakeVar:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def setVal(self, value: float) -> None:
        self.value = value


class FakePdf:
    def __init__(self, value: float = 0.5) -> None:
        self.value = value

    def getVal(self, norm_set: Any) -> float:
        return self.value


class FakeRootFile:
    def __init__(self, workspace: Any | None = None, zombie: bool = False) -> None:
        self.workspace = workspace
        self.zombie = zombie
        self.closed = False

    def IsZombie(self) -> bool:
        return self.zombie

    def Get(self, name: str) -> Any:
        assert name == "combWS"
        return self.workspace

    def Close(self) -> None:
        self.closed = True


class FakeRooFitWorkspace:
    def __init__(self) -> None:
        self.dataset = FakeDataset()
        self.pdf_obj = FakePdf()
        self.x = FakeVar()
        self.mu_sig = FakeVar()
        self.loaded_snapshot: str | None = None

    def loadSnapshot(self, name: str) -> None:
        self.loaded_snapshot = name

    def pdf(self, name: str) -> Any:
        return self.pdf_obj

    def data(self, name: str) -> Any:
        return self.dataset

    def cat(self, name: str) -> Any:
        return object()

    def var(self, name: str) -> Any:
        if name == "x":
            return self.x
        if name == "mu_sig":
            return self.mu_sig
        return None


class FakeRootModule:
    class RooFit:
        ERROR = object()

    class RooMsgService:
        @staticmethod
        def instance() -> "FakeRootModule.RooMsgService":
            return FakeRootModule.RooMsgService()

        def setGlobalKillBelow(self, level: object) -> None:
            self.level = level

    class RooArgSet(tuple):
        def __new__(cls, *args: Any) -> "FakeRootModule.RooArgSet":
            return tuple.__new__(cls, args)

    class TFile:
        root_file: FakeRootFile | None = None

        @staticmethod
        def Open(path: str) -> FakeRootFile | None:
            return FakeRootModule.TFile.root_file


def test_validate_existing_dir_and_file_success(tmp_path: Path) -> None:
    directory = tmp_path / "inputs"
    directory.mkdir()
    file_path = directory / "workspace.json"
    file_path.write_text("{}")

    assert benchmark.validate_existing_dir(directory, "Input directory") == directory
    assert benchmark.validate_existing_file(file_path, "Input file") == file_path


@pytest.mark.parametrize("path_kind", ["missing", "file"])
def test_validate_existing_dir_rejects_invalid(tmp_path: Path, path_kind: str) -> None:
    path = tmp_path / "path"
    if path_kind == "file":
        path.write_text("not a dir")
    with pytest.raises(FileNotFoundError, match="Input directory"):
        benchmark.validate_existing_dir(path, "Input directory")


@pytest.mark.parametrize("path_kind", ["missing", "dir"])
def test_validate_existing_file_rejects_invalid(tmp_path: Path, path_kind: str) -> None:
    path = tmp_path / "path"
    if path_kind == "dir":
        path.mkdir()
    with pytest.raises(FileNotFoundError, match="Input file"):
        benchmark.validate_existing_file(path, "Input file")


@pytest.mark.parametrize(
    ("value", "minimum"),
    [(1, 1), (2, 2)],
)
def test_validate_positive_int_success(value: int, minimum: int) -> None:
    benchmark.validate_positive_int(value, "value", minimum=minimum)


def test_validate_positive_int_rejects_too_small() -> None:
    with pytest.raises(ValueError, match="value must be at least 2"):
        benchmark.validate_positive_int(1, "value", minimum=2)


@pytest.mark.parametrize("value", [0.0, -1.5, 2.0])
def test_validate_finite_float_success(value: float) -> None:
    benchmark.validate_finite_float(value, "value")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_validate_finite_float_rejects_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="value must be finite"):
        benchmark.validate_finite_float(value, "value")


def valid_config_kwargs() -> dict[str, Any]:
    return {
        "n_runs": 1,
        "mu_sig": 1.0,
        "scan_min": 0.0,
        "scan_max": 2.0,
        "n_scan_points": 3,
        "cases": ["simple_workspace_nonp"],
        "analyses": ["L_ch0"],
        "delta_tolerance": 1e-8,
        "minimum_tolerance": 1e-12,
    }


def test_validate_benchmark_config_success() -> None:
    benchmark.validate_benchmark_config(**valid_config_kwargs())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_runs": 0}, "n_runs must be at least 1"),
        ({"n_scan_points": 1}, "n_scan_points must be at least 2"),
        ({"mu_sig": float("nan")}, "mu_sig must be finite"),
        ({"scan_min": 2.0, "scan_max": 1.0}, "scan_min must be smaller"),
        ({"delta_tolerance": 0.0}, "delta_tolerance must be positive"),
        ({"minimum_tolerance": 0.0}, "minimum_tolerance must be positive"),
        ({"cases": []}, "At least one case"),
        ({"analyses": []}, "At least one analysis"),
        ({"cases": ["missing"]}, "Unknown cases"),
        ({"analyses": ["L_missing"]}, "Unknown analyses"),
    ],
)
def test_validate_benchmark_config_rejects_invalid(
    override: dict[str, Any], message: str
) -> None:
    kwargs = valid_config_kwargs()
    kwargs.update(override)
    with pytest.raises(ValueError, match=message):
        benchmark.validate_benchmark_config(**kwargs)


@pytest.mark.parametrize("values", [[1.0], (v for v in [1.0, 2.0])])
def test_validate_scan_values_accepts_non_empty_finite_iterables(values: Any) -> None:
    benchmark.validate_scan_values(values, "values")


@pytest.mark.parametrize("values", [[], [1.0, float("nan")], [float("inf")]])
def test_validate_scan_values_rejects_invalid(values: list[float]) -> None:
    with pytest.raises(ValueError):
        benchmark.validate_scan_values(values, "values")


def test_validate_framework_result_success(
    valid_framework_result: dict[str, Any],
) -> None:
    benchmark.validate_framework_result(valid_framework_result)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("build_time_seconds", float("nan"), "not finite"),
        ("build_time_seconds", -0.01, "build_time_seconds must be non-negative"),
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
    ],
)
def test_validate_framework_result_rejects_invalid_fields(
    valid_framework_result: dict[str, Any],
    field: str,
    value: float,
    message: str,
) -> None:
    result = dict(valid_framework_result)
    result[field] = value
    with pytest.raises(ValueError, match=message):
        benchmark.validate_framework_result(result)


def test_channel_and_target_from_analysis() -> None:
    assert benchmark.channel_from_analysis("L_ch2") == "ch2"
    assert benchmark.target_from_analysis("L_ch2") == "model_ch2"


def test_channel_from_analysis_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="must start"):
        benchmark.channel_from_analysis("ch0")


def test_build_case_specs(tmp_path: Path) -> None:
    specs = benchmark.build_case_specs(
        json_input_dir=tmp_path / "json",
        root_input_dir=tmp_path / "root",
        cases=["simple_workspace_nonp", "simple_workspace"],
        analyses=["L_ch0", "L_ch1"],
    )

    assert len(specs) == 4
    assert specs[0] == benchmark.CaseSpec(
        case_name="simple_workspace_nonp",
        analysis_name="L_ch0",
        json_path=tmp_path / "json" / "simple_workspace_nonp.json",
        root_path=tmp_path / "root" / "simple_workspace_nonp.root",
    )


def test_get_pyhs3_x_data_success() -> None:
    workspace = SimpleNamespace(
        data=SimpleNamespace(
            root=[
                SimpleNamespace(name="other", entries=[[99.0]]),
                SimpleNamespace(name="combData_ch0", entries=[[1.0], [2.0], [3.0]]),
            ]
        )
    )

    values = benchmark.get_pyhs3_x_data(workspace, "L_ch0")

    assert values.tolist() == [1.0, 2.0, 3.0]


def test_get_pyhs3_x_data_rejects_invalid_data_section() -> None:
    with pytest.raises(ValueError, match="valid data section"):
        benchmark.get_pyhs3_x_data(SimpleNamespace(), "L_ch0")


def test_get_pyhs3_x_data_rejects_empty_and_nonfinite() -> None:
    empty_workspace = SimpleNamespace(
        data=SimpleNamespace(root=[SimpleNamespace(name="combData_ch0", entries=[])])
    )
    with pytest.raises(ValueError, match="empty"):
        benchmark.get_pyhs3_x_data(empty_workspace, "L_ch0")

    bad_workspace = SimpleNamespace(
        data=SimpleNamespace(
            root=[SimpleNamespace(name="combData_ch0", entries=[[float("nan")]])]
        )
    )
    with pytest.raises(ValueError, match="non-finite"):
        benchmark.get_pyhs3_x_data(bad_workspace, "L_ch0")


def test_get_pyhs3_x_data_rejects_missing_dataset() -> None:
    workspace = SimpleNamespace(data=SimpleNamespace(root=[]))
    with pytest.raises(KeyError, match="combData_ch0"):
        benchmark.get_pyhs3_x_data(workspace, "L_ch0")


def test_get_pyhs3_params_success() -> None:
    model = SimpleNamespace(free_params={"mu_sig": 1.0, "theta": np.asarray([2.0])})
    x = np.asarray([1.0, 2.0])

    params = benchmark.get_pyhs3_params(model, x)

    assert set(params) == {"mu_sig", "theta", "x"}
    assert params["x"] is x


def test_get_pyhs3_params_rejects_missing_or_nonfinite_params() -> None:
    with pytest.raises(ValueError, match="free_params"):
        benchmark.get_pyhs3_params(SimpleNamespace(), np.asarray([1.0]))

    with pytest.raises(ValueError, match="non-finite"):
        benchmark.get_pyhs3_params(
            SimpleNamespace(free_params={"bad": float("nan")}), np.asarray([1.0])
        )


def test_pyhs3_nll_success() -> None:
    class Model:
        def logpdf(self, target: str, **params: Any) -> np.ndarray:
            assert target == "model_ch0"
            assert "mu_sig" in params
            return np.asarray([-1.0, -2.0])

    value = benchmark.pyhs3_nll(
        Model(), "model_ch0", {"x": np.asarray([1.0, 2.0])}, 1.5
    )

    assert value == pytest.approx(3.0)


@pytest.mark.parametrize("output", [np.asarray([]), np.asarray([float("nan")])])
def test_pyhs3_nll_rejects_empty_or_nonfinite(output: np.ndarray) -> None:
    class Model:
        def logpdf(self, target: str, **params: Any) -> np.ndarray:
            return output

    with pytest.raises(ValueError):
        benchmark.pyhs3_nll(Model(), "target", {}, 1.0)


def test_build_pyhs3_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "workspace.json"
    path.write_text("{}")
    fake_workspace = SimpleNamespace(
        data=SimpleNamespace(
            root=[SimpleNamespace(name="combData_ch0", entries=[[1.0], [2.0]])]
        ),
        model=lambda *args, **kwargs: SimpleNamespace(free_params={"mu_sig": 1.0}),
    )
    monkeypatch.setattr(benchmark.Workspace, "load", lambda input_path: fake_workspace)

    case = benchmark.build_pyhs3_case(path, "L_ch0")

    assert case["workspace"] is fake_workspace
    assert case["target"] == "model_ch0"
    assert case["x"].tolist() == [1.0, 2.0]
    assert "params" in case


def test_require_root_rejects_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "ROOT", None)
    with pytest.raises(RuntimeError, match="ROOT is not available"):
        benchmark.require_root()


def test_require_root_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    assert benchmark.require_root() is FakeRootModule


def test_get_roofit_workspace_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root_path = tmp_path / "workspace.root"
    root_path.write_text("fake")
    fake_workspace = FakeRooFitWorkspace()
    FakeRootModule.TFile.root_file = FakeRootFile(fake_workspace)
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)

    root_file, workspace = benchmark.get_roofit_workspace(root_path)

    assert root_file.workspace is fake_workspace
    assert workspace is fake_workspace


def test_get_roofit_workspace_rejects_zombie_and_missing_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root_path = tmp_path / "workspace.root"
    root_path.write_text("fake")
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)

    FakeRootModule.TFile.root_file = FakeRootFile(FakeRooFitWorkspace(), zombie=True)
    with pytest.raises(RuntimeError, match="Could not open ROOT file"):
        benchmark.get_roofit_workspace(root_path)

    missing_file = FakeRootFile(None)
    FakeRootModule.TFile.root_file = missing_file
    with pytest.raises(RuntimeError, match="combWS"):
        benchmark.get_roofit_workspace(root_path)
    assert missing_file.closed


def test_get_roofit_channel_dataset_success() -> None:
    workspace = FakeRooFitWorkspace()

    dataset = benchmark.get_roofit_channel_dataset(workspace, "L_ch0")

    assert dataset is workspace.dataset
    assert dataset.reduce_expression == "index==index::ch0"


def test_get_roofit_channel_dataset_rejects_missing_parts() -> None:
    class NoData(FakeRooFitWorkspace):
        def data(self, name: str) -> None:
            return None

    with pytest.raises(RuntimeError, match="combData"):
        benchmark.get_roofit_channel_dataset(NoData(), "L_ch0")

    class NoIndex(FakeRooFitWorkspace):
        def cat(self, name: str) -> None:
            return None

    with pytest.raises(RuntimeError, match="category"):
        benchmark.get_roofit_channel_dataset(NoIndex(), "L_ch0")

    class BadDataset(FakeDataset):
        def reduce(self, expression: str) -> None:
            return None

    class BadReduction(FakeRooFitWorkspace):
        def __init__(self) -> None:
            super().__init__()
            self.dataset = BadDataset()

    with pytest.raises(RuntimeError, match="reduce"):
        benchmark.get_roofit_channel_dataset(BadReduction(), "L_ch0")

    class EmptyReduction(FakeRooFitWorkspace):
        def __init__(self) -> None:
            super().__init__()
            self.dataset = FakeDataset([])

    with pytest.raises(RuntimeError, match="empty"):
        benchmark.get_roofit_channel_dataset(EmptyReduction(), "L_ch0")


def test_build_roofit_case_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root_path = tmp_path / "workspace.root"
    root_path.write_text("fake")
    fake_workspace = FakeRooFitWorkspace()
    FakeRootModule.TFile.root_file = FakeRootFile(fake_workspace)
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)

    case = benchmark.build_roofit_case(root_path, "L_ch0")

    assert case["workspace"] is fake_workspace
    assert case["target"] == "model_ch0"
    assert case["pdf"] is fake_workspace.pdf_obj
    assert case["root"] is FakeRootModule


def test_build_roofit_case_rejects_missing_pdf_or_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root_path = tmp_path / "workspace.root"
    root_path.write_text("fake")
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)

    class NoPdf(FakeRooFitWorkspace):
        def pdf(self, name: str) -> None:
            return None

    root_file = FakeRootFile(NoPdf())
    FakeRootModule.TFile.root_file = root_file
    with pytest.raises(RuntimeError, match="pdf"):
        benchmark.build_roofit_case(root_path, "L_ch0")
    assert root_file.closed

    class NoX(FakeRooFitWorkspace):
        def var(self, name: str) -> Any:
            if name == "x":
                return None
            return super().var(name)

    root_file = FakeRootFile(NoX())
    FakeRootModule.TFile.root_file = root_file
    with pytest.raises(RuntimeError, match="variable x"):
        benchmark.build_roofit_case(root_path, "L_ch0")
    assert root_file.closed

    class NoMu(FakeRooFitWorkspace):
        def var(self, name: str) -> Any:
            if name == "mu_sig":
                return None
            return super().var(name)

    root_file = FakeRootFile(NoMu())
    FakeRootModule.TFile.root_file = root_file
    with pytest.raises(RuntimeError, match="mu_sig"):
        benchmark.build_roofit_case(root_path, "L_ch0")
    assert root_file.closed


def test_roofit_nll_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    case = {
        "pdf": FakePdf(0.5),
        "data": FakeDataset([1.0, 2.0]),
        "x": FakeVar(),
        "mu_sig": FakeVar(),
        "root": FakeRootModule,
    }

    value = benchmark.roofit_nll(case, 1.25)

    assert value == pytest.approx(-2 * math.log(0.5))
    assert case["mu_sig"].value == 1.25


def test_roofit_nll_rejects_empty_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)

    class EmptyDataset(FakeDataset):
        def __init__(self) -> None:
            super().__init__([])

        def numEntries(self) -> int:
            return 0

    case = {
        "pdf": FakePdf(),
        "data": EmptyDataset(),
        "x": FakeVar(),
        "mu_sig": FakeVar(),
        "root": FakeRootModule,
    }
    with pytest.raises(ValueError, match="empty"):
        benchmark.roofit_nll(case, 1.0)


def test_roofit_nll_rejects_nonfinite_x_and_invalid_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "ROOT", FakeRootModule)
    bad_x_case = {
        "pdf": FakePdf(),
        "data": FakeDataset([float("nan")]),
        "x": FakeVar(),
        "mu_sig": FakeVar(),
        "root": FakeRootModule,
    }
    with pytest.raises(ValueError, match="non-finite x"):
        benchmark.roofit_nll(bad_x_case, 1.0)

    bad_pdf_case = {
        "pdf": FakePdf(0.0),
        "data": FakeDataset([1.0]),
        "x": FakeVar(),
        "mu_sig": FakeVar(),
        "root": FakeRootModule,
    }
    with pytest.raises(ValueError, match="invalid PDF"):
        benchmark.roofit_nll(bad_pdf_case, 1.0)


def test_summarize_timings_single_and_multiple_values() -> None:
    assert benchmark.summarize_timings([0.1])["std_seconds"] == 0.0
    summary = benchmark.summarize_timings([0.1, 0.3])
    assert summary["mean_seconds"] == pytest.approx(0.2)
    assert summary["min_seconds"] == pytest.approx(0.1)
    assert summary["max_seconds"] == pytest.approx(0.3)


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

    values, duration = benchmark.scan_nll(lambda mu: mu * mu, [0.0, 1.0, 2.0])

    assert values == [0.0, 1.0, 4.0]
    assert duration == pytest.approx(0.3)


@pytest.mark.parametrize("values", [[], [1.0, float("nan")]])
def test_delta_nll_rejects_invalid(values: list[float]) -> None:
    with pytest.raises(ValueError):
        benchmark.delta_nll(values)


def test_delta_nll_success() -> None:
    assert benchmark.delta_nll([3.0, 1.0, 2.0]).tolist() == [2.0, 0.0, 1.0]


def test_minimum_position_success_and_length_mismatch() -> None:
    assert benchmark.minimum_position([0.0, 1.0, 2.0], [3.0, 1.0, 2.0]) == 1.0
    with pytest.raises(ValueError, match="same length"):
        benchmark.minimum_position([0.0], [1.0, 2.0])


def test_close_case_closes_root_file_and_swallows_errors() -> None:
    root_file = FakeRootFile()
    benchmark.close_case({"root_file": root_file})
    assert root_file.closed

    class BadFile:
        def Close(self) -> None:
            raise RuntimeError("close failed")

    benchmark.close_case({"root_file": BadFile()})
    benchmark.close_case(None)


def test_measure_framework_success(monkeypatch: pytest.MonkeyPatch) -> None:
    times = iter([1.0, 1.1, 2.0, 2.1, 3.0, 3.1, 4.0, 4.2])
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 10.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 20.0)

    result = benchmark.measure_framework(
        framework="pyhs3",
        build_func=lambda: {"model": object()},
        eval_func=lambda case, mu: (mu - 1.0) ** 2,
        scan_values=[0.0, 1.0, 2.0],
        n_runs=1,
        mu_sig=1.0,
    )

    assert result["framework"] == "pyhs3"
    assert result["status"] == "success"
    assert result["scan_nll_values"] == [1.0, 0.0, 1.0]
    assert result["minimum_mu_sig"] == 1.0


def test_measure_framework_closes_case_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_file = FakeRootFile()
    case = {"root_file": root_file}
    monkeypatch.setattr(benchmark, "get_current_rss_mb", lambda: 10.0)
    monkeypatch.setattr(benchmark, "get_peak_rss_mb", lambda: 20.0)

    with pytest.raises(RuntimeError, match="boom"):
        benchmark.measure_framework(
            framework="roofit",
            build_func=lambda: case,
            eval_func=lambda _case, _mu: (_ for _ in ()).throw(RuntimeError("boom")),
            scan_values=[0.0, 1.0],
            n_runs=1,
            mu_sig=1.0,
        )

    assert root_file.closed


def test_add_agreement_metrics_success_and_failure(
    valid_framework_result: dict[str, Any],
) -> None:
    pyhs3 = dict(valid_framework_result, framework="pyhs3")
    roofit = dict(valid_framework_result, framework="roofit")

    agreement = benchmark.add_agreement_metrics(
        pyhs3_result=pyhs3,
        roofit_result=roofit,
        delta_tolerance=1e-12,
        minimum_tolerance=1e-12,
    )

    assert agreement["validation_status"] == "success"
    assert agreement["minimum_index_match"] is True
    assert agreement["delta_nll_max_abs_diff"] == pytest.approx(0.0)

    roofit_bad = dict(
        roofit, scan_nll_values=[2.0, 1.2, 2.0], minimum_index=0, minimum_mu_sig=0.0
    )
    failed = benchmark.add_agreement_metrics(
        pyhs3_result=pyhs3,
        roofit_result=roofit_bad,
        delta_tolerance=1e-12,
        minimum_tolerance=1e-12,
    )
    assert failed["validation_status"] == "failed"
    assert failed["minimum_index_match"] is False


def test_add_agreement_metrics_rejects_shape_mismatch(
    valid_framework_result: dict[str, Any],
) -> None:
    pyhs3 = dict(valid_framework_result, framework="pyhs3")
    roofit = dict(valid_framework_result, framework="roofit", scan_nll_values=[1.0])
    with pytest.raises(ValueError, match="different shapes"):
        benchmark.add_agreement_metrics(
            pyhs3_result=pyhs3,
            roofit_result=roofit,
            delta_tolerance=1e-9,
            minimum_tolerance=1e-12,
        )


def test_failed_case_result(tmp_path: Path) -> None:
    spec = benchmark.CaseSpec("case", "L_ch0", tmp_path / "a.json", tmp_path / "a.root")
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        result = benchmark.failed_case_result(spec, exc)

    assert result["case"] == "case"
    assert result["target"] == "model_ch0"
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert "traceback" in result


def test_measure_case_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_framework_result: dict[str, Any],
) -> None:
    spec = benchmark.CaseSpec(
        "simple_workspace_nonp", "L_ch0", tmp_path / "a.json", tmp_path / "a.root"
    )
    pyhs3 = dict(valid_framework_result, framework="pyhs3")
    roofit = dict(valid_framework_result, framework="roofit")
    calls: list[str] = []

    def fake_measure_framework(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["framework"])
        return pyhs3 if kwargs["framework"] == "pyhs3" else roofit

    monkeypatch.setattr(benchmark, "measure_framework", fake_measure_framework)

    result = benchmark.measure_case(
        spec=spec,
        scan_values=[0.0, 1.0, 2.0],
        n_runs=1,
        mu_sig=1.0,
        delta_tolerance=1e-9,
        minimum_tolerance=1e-12,
    )

    assert calls == ["pyhs3", "roofit"]
    assert result["status"] == "success"
    assert result["agreement"]["validation_status"] == "success"


def test_measure_case_returns_failed_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    valid_framework_result: dict[str, Any],
) -> None:
    spec = benchmark.CaseSpec(
        "simple_workspace_nonp", "L_ch0", tmp_path / "a.json", tmp_path / "a.root"
    )
    pyhs3 = dict(valid_framework_result, framework="pyhs3")
    roofit = dict(
        valid_framework_result, framework="roofit", scan_nll_values=[2.0, 1.2, 2.0]
    )

    monkeypatch.setattr(
        benchmark,
        "measure_framework",
        lambda **kwargs: pyhs3 if kwargs["framework"] == "pyhs3" else roofit,
    )

    result = benchmark.measure_case(
        spec=spec,
        scan_values=[0.0, 1.0, 2.0],
        n_runs=1,
        mu_sig=1.0,
        delta_tolerance=1e-12,
        minimum_tolerance=1e-12,
    )

    assert result["status"] == "failed"
    assert result["error_type"] == "ValidationFailure"
    assert "Numerical agreement check failed" in result["error_message"]


def test_style_label_helpers(successful_case: dict[str, Any]) -> None:
    assert benchmark._framework_label("pyhs3") == "PyHS3"
    assert benchmark._framework_label("unknown") == "unknown"
    assert benchmark._case_label(successful_case) == "nonp\nch0"
    assert benchmark._successful_results([successful_case, {"status": "failed"}]) == [
        successful_case
    ]
    assert benchmark._diagnostic_results([successful_case, {"status": "failed"}]) == [
        successful_case
    ]


def test_validation_failed_results(
    validation_failed_case: dict[str, Any], successful_case: dict[str, Any]
) -> None:
    assert benchmark._validation_failed_results(
        [validation_failed_case, successful_case]
    ) == [validation_failed_case]


def test_plot_floor() -> None:
    assert benchmark._plot_floor([0.0, 2.0], floor=0.1) == [0.1, 2.0]


def test_collect_scaling_records(successful_case: dict[str, Any]) -> None:
    records = benchmark._collect_scaling_records([successful_case])
    assert [record["framework"] for record in records] == ["pyhs3", "roofit"]
    assert records[0]["plot_label"] == "nonp\nch0"
    assert records[0]["build_ms"] == pytest.approx(10.0)


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
        (benchmark.make_timing_breakdown_plot, "timing.png", None),
        (benchmark.make_memory_scaling_plot, "memory.png", None),
        (benchmark.make_agreement_plot, "agreement.png", 1e-9),
        (benchmark.make_profile_examples_plot, "profiles.png", [0.0, 1.0, 2.0]),
        (benchmark.make_summary_table_plot, "summary.png", None),
    ],
)
def test_individual_plot_functions_create_png(
    tmp_path: Path,
    successful_case: dict[str, Any],
    plot_func: Any,
    filename: str,
    extra: Any,
) -> None:
    output = tmp_path / filename
    if plot_func is benchmark.make_agreement_plot:
        plot_func([successful_case], extra, output)
    elif plot_func is benchmark.make_profile_examples_plot:
        plot_func([successful_case], extra, output)
    else:
        plot_func([successful_case], output)
    assert output.exists()


def test_profile_examples_plot_rejects_no_success(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No successful"):
        benchmark.make_profile_examples_plot([], [0.0, 1.0], tmp_path / "plot.png")


def test_validation_failure_diagnostics_plot_returns_when_no_failures(
    tmp_path: Path, successful_case: dict[str, Any]
) -> None:
    output = tmp_path / "diagnostics.png"
    benchmark.make_validation_failure_diagnostics_plot(
        [successful_case], [0.0, 1.0, 2.0], output
    )
    assert not output.exists()


def test_validation_failure_diagnostics_plot_creates_png(
    tmp_path: Path,
    validation_failed_case: dict[str, Any],
) -> None:
    output = tmp_path / "diagnostics.png"
    benchmark.make_validation_failure_diagnostics_plot(
        [validation_failed_case], [0.0, 1.0, 2.0], output
    )
    assert output.exists()


def test_make_plots_creates_expected_pngs(
    tmp_path: Path, successful_case: dict[str, Any]
) -> None:
    benchmark.make_plots(
        [successful_case], [0.0, 1.0, 2.0], tmp_path, delta_tolerance=1e-9
    )

    expected = {
        "cross_model_complexity_runtime_scaling.png",
        "cross_model_complexity_timing_breakdown.png",
        "cross_model_complexity_memory_scaling.png",
        "cross_model_complexity_agreement.png",
        "cross_model_complexity_profile_examples.png",
        "cross_model_complexity_summary_table.png",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})


def test_make_plots_rejects_no_success(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No successful"):
        benchmark.make_plots(
            [{"status": "failed"}], [0.0, 1.0], tmp_path, delta_tolerance=1e-9
        )


def test_print_case_success_and_failure(
    capsys: pytest.CaptureFixture[str], successful_case: dict[str, Any]
) -> None:
    benchmark.print_case(successful_case)
    output = capsys.readouterr().out
    assert "simple_workspace_nonp / L_ch0 / model_ch0" in output
    assert "agreement" in output

    benchmark.print_case(
        {
            "case": "case",
            "analysis": "L_ch0",
            "target": "model_ch0",
            "status": "failed",
            "error_type": "X",
            "error_message": "bad",
        }
    )
    output = capsys.readouterr().out
    assert "error:  X: bad" in output

    failed_with_diagnostics = dict(
        successful_case,
        status="failed",
        error_type="ValidationFailure",
        error_message="bad",
    )
    benchmark.print_case(failed_with_diagnostics)
    output = capsys.readouterr().out
    assert "diagnostics: available" in output


def test_build_failed_output(tmp_path: Path) -> None:
    try:
        raise ValueError("bad")
    except ValueError as exc:
        output = benchmark.build_failed_output(
            json_input_dir=tmp_path / "json",
            root_input_dir=tmp_path / "root",
            n_runs=1,
            mu_sig=1.0,
            scan_min=0.0,
            scan_max=2.0,
            n_scan_points=3,
            cases=["simple_workspace_nonp"],
            analyses=["L_ch0"],
            delta_tolerance=1e-9,
            minimum_tolerance=1e-12,
            exc=exc,
        )

    assert output["benchmark"] == benchmark.BENCHMARK_NAME
    assert output["status"] == "failed"
    assert output["error_type"] == "ValueError"
    assert output["results"] == []


def test_run_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, successful_case: dict[str, Any]
) -> None:
    json_dir, root_dir = tmp_path / "json", tmp_path / "root"
    json_dir.mkdir()
    root_dir.mkdir()
    for pair in benchmark.WORKSPACE_PAIRS.values():
        (json_dir / pair["json"]).write_text("{}")
        (root_dir / pair["root"]).write_text("root")
    output = tmp_path / "result.json"
    plot_dir = tmp_path / "plots"

    monkeypatch.setattr(benchmark, "require_root", lambda: FakeRootModule)
    monkeypatch.setattr(benchmark, "measure_case", lambda **kwargs: successful_case)
    plot_calls: list[Any] = []
    monkeypatch.setattr(
        benchmark,
        "make_plots",
        lambda *args, **kwargs: plot_calls.append((args, kwargs)),
    )

    result = benchmark.run(
        json_input_dir=json_dir,
        root_input_dir=root_dir,
        n_runs=1,
        mu_sig=1.0,
        scan_min=0.0,
        scan_max=2.0,
        n_scan_points=3,
        output=output,
        plot=True,
        plot_dir=plot_dir,
        cases=["simple_workspace_nonp"],
        analyses=["L_ch0"],
        delta_tolerance=1e-9,
        minimum_tolerance=1e-12,
    )

    assert result["status"] == "success"
    assert result["successful_cases"] == ["simple_workspace_nonp/L_ch0"]
    assert json.loads(output.read_text())["status"] == "success"
    assert plot_calls


def test_run_continues_on_case_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    json_dir, root_dir = tmp_path / "json", tmp_path / "root"
    json_dir.mkdir()
    root_dir.mkdir()
    (json_dir / "simple_workspace_nonp.json").write_text("{}")
    (root_dir / "simple_workspace_nonp.root").write_text("root")
    output = tmp_path / "result.json"

    monkeypatch.setattr(benchmark, "require_root", lambda: FakeRootModule)
    monkeypatch.setattr(
        benchmark,
        "measure_case",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("case failed")),
    )

    result = benchmark.run(
        json_input_dir=json_dir,
        root_input_dir=root_dir,
        n_runs=1,
        mu_sig=1.0,
        scan_min=0.0,
        scan_max=2.0,
        n_scan_points=3,
        output=output,
        plot=True,
        plot_dir=tmp_path / "plots",
        cases=["simple_workspace_nonp"],
        analyses=["L_ch0"],
        continue_on_case_error=True,
    )

    assert result["status"] == "failed"
    assert result["failed_cases"] == ["simple_workspace_nonp/L_ch0"]
    assert result["results"][0]["error_type"] == "RuntimeError"


def test_run_fail_fast_writes_failure_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    json_dir, root_dir = tmp_path / "json", tmp_path / "root"
    json_dir.mkdir()
    root_dir.mkdir()
    (json_dir / "simple_workspace_nonp.json").write_text("{}")
    (root_dir / "simple_workspace_nonp.root").write_text("root")
    output = tmp_path / "result.json"

    monkeypatch.setattr(benchmark, "require_root", lambda: FakeRootModule)
    monkeypatch.setattr(
        benchmark,
        "measure_case",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("case failed")),
    )

    with pytest.raises(RuntimeError, match="Cross-framework model-complexity"):
        benchmark.run(
            json_input_dir=json_dir,
            root_input_dir=root_dir,
            n_runs=1,
            mu_sig=1.0,
            scan_min=0.0,
            scan_max=2.0,
            n_scan_points=3,
            output=output,
            plot=False,
            plot_dir=tmp_path / "plots",
            cases=["simple_workspace_nonp"],
            analyses=["L_ch0"],
            continue_on_case_error=False,
        )

    payload = json.loads(output.read_text())
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"


def test_run_handles_failure_report_save_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    json_dir = tmp_path / "missing_json"
    root_dir = tmp_path / "missing_root"

    def failing_save_json(*args: Any, **kwargs: Any) -> None:
        raise OSError("cannot save")

    monkeypatch.setattr(benchmark, "save_json", failing_save_json)

    with pytest.raises(RuntimeError, match="Cross-framework model-complexity"):
        benchmark.run(
            json_input_dir=json_dir,
            root_input_dir=root_dir,
            n_runs=1,
            mu_sig=1.0,
            scan_min=0.0,
            scan_max=2.0,
            n_scan_points=3,
            output=tmp_path / "result.json",
            plot=False,
            plot_dir=tmp_path / "plots",
            cases=["simple_workspace_nonp"],
            analyses=["L_ch0"],
        )

    assert "Failed to save benchmark failure report" in capsys.readouterr().err


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_cross_model_complexity_scaling.py"])
    args = benchmark.parse_args()
    assert args.json_input_dir == Path("inputs/model_complexity")
    assert args.root_input_dir == Path("inputs/model_complexity_root")
    assert args.n_runs == 100
    assert args.cases == benchmark.DEFAULT_CASES
    assert args.analyses == benchmark.DEFAULT_ANALYSES
    assert args.fail_fast is False


def test_parse_args_custom_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cross_model_complexity_scaling.py",
            "--json-input-dir",
            str(tmp_path / "json"),
            "--root-input-dir",
            str(tmp_path / "root"),
            "--n-runs",
            "2",
            "--mu-sig",
            "1.5",
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
            "--cases",
            "simple_workspace_nonp",
            "--analyses",
            "L_ch0",
            "--delta-tolerance",
            "1e-7",
            "--minimum-tolerance",
            "1e-11",
            "--fail-fast",
        ],
    )
    args = benchmark.parse_args()
    assert args.json_input_dir == tmp_path / "json"
    assert args.root_input_dir == tmp_path / "root"
    assert args.n_runs == 2
    assert args.mu_sig == 1.5
    assert args.plot is True
    assert args.cases == ["simple_workspace_nonp"]
    assert args.analyses == ["L_ch0"]
    assert args.fail_fast is True


def test_parse_args_rejects_unknown_case(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys, "argv", ["run_cross_model_complexity_scaling.py", "--cases", "missing"]
    )
    with pytest.raises(SystemExit):
        benchmark.parse_args()


def test_main_passes_cli_arguments(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_cross_model_complexity_scaling.py",
            "--json-input-dir",
            str(tmp_path / "json"),
            "--root-input-dir",
            str(tmp_path / "root"),
            "--n-runs",
            "2",
            "--output",
            str(tmp_path / "out.json"),
            "--cases",
            "simple_workspace_nonp",
            "--analyses",
            "L_ch0",
            "--fail-fast",
        ],
    )
    monkeypatch.setattr(benchmark, "run", lambda **kwargs: calls.append(kwargs))

    benchmark.main()

    assert calls[0]["json_input_dir"] == tmp_path / "json"
    assert calls[0]["root_input_dir"] == tmp_path / "root"
    assert calls[0]["continue_on_case_error"] is False
