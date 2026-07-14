from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import src.run_cross_binned_likelihood as benchmark


def _hifa_spec() -> dict[str, Any]:
    return {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "signal",
                        "data": [12.0, 11.0],
                        "modifiers": [
                            {"name": "mu", "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": "background",
                        "data": [50.0, 52.0],
                        "modifiers": [
                            {
                                "name": "correlated_bkg_uncertainty",
                                "type": "histosys",
                                "data": {
                                    "hi_data": [45.0, 55.0],
                                    "lo_data": [55.0, 45.0],
                                },
                            }
                        ],
                    },
                ],
            }
        ],
        "measurements": [
            {
                "name": "measurement",
                "config": {"poi": "mu", "parameters": []},
            }
        ],
        "observations": [{"name": "singlechannel", "data": [51.0, 48.0]}],
        "version": "1.0.0",
    }


def _hs3_spec() -> dict[str, Any]:
    return {
        "analyses": [
            {
                "name": "simPdf_obsData",
                "likelihood": "simPdf_obsData",
                "domains": [],
                "parameters_of_interest": ["mu"],
            }
        ],
        "data": [
            {
                "name": "obsData_singlechannel",
                "type": "binned",
                "axes": [
                    {
                        "name": "obs_x_singlechannel",
                        "min": 0.0,
                        "max": 2.0,
                        "nbins": 2,
                    }
                ],
                "contents": [51.0, 48.0],
            },
            {
                "name": "asimovData_singlechannel",
                "type": "binned",
                "axes": [
                    {
                        "name": "obs_x_singlechannel",
                        "min": 0.0,
                        "max": 2.0,
                        "nbins": 2,
                    }
                ],
                "contents": [62.0, 63.0],
            },
        ],
        "distributions": [
            {
                "name": "model_singlechannel",
                "type": "histfactory_dist",
                "axes": [
                    {
                        "name": "obs_x_singlechannel",
                        "min": 0.0,
                        "max": 2.0,
                        "nbins": 2,
                    }
                ],
                "samples": [
                    {
                        "name": "signal",
                        "data": {
                            "contents": [12.0, 11.0],
                            "errors": [3.5, 3.3],
                        },
                        "modifiers": [
                            {
                                "name": "Lumi",
                                "parameter": "Lumi",
                                "type": "normfactor",
                            },
                            {
                                "name": "mu",
                                "parameter": "mu",
                                "type": "normfactor",
                            },
                        ],
                    },
                    {
                        "name": "background",
                        "data": {
                            "contents": [50.0, 52.0],
                            "errors": [7.0, 7.2],
                        },
                        "modifiers": [
                            {
                                "name": "Lumi",
                                "parameter": "Lumi",
                                "type": "normfactor",
                            },
                            {
                                "name": "correlated_bkg_uncertainty",
                                "parameter": "alpha_correlated_bkg_uncertainty",
                                "type": "histosys",
                                "data": {
                                    "hi": {"contents": [45.0, 55.0]},
                                    "lo": {"contents": [55.0, 45.0]},
                                },
                            },
                        ],
                    },
                ],
            }
        ],
        "domains": [],
        "likelihoods": [
            {
                "name": "simPdf_obsData",
                "data": ["obsData_singlechannel"],
                "distributions": ["model_singlechannel"],
            }
        ],
        "metadata": {"hs3_version": "0.2", "packages": []},
        "misc": {},
        "parameter_points": [
            {
                "name": "default_values",
                "parameters": [
                    {"name": "Lumi", "value": 1.0, "const": True},
                    {"name": "alpha_correlated_bkg_uncertainty", "value": 0.0},
                    {"name": "mu", "value": 1.0},
                ],
            }
        ],
    }


def _fake_engine(
    name: str,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    construction: float = 0.01,
    first: float = 0.02,
) -> benchmark.Engine:
    def evaluate(mu: float) -> float:
        return scale * (mu - 0.3) ** 2 + offset

    def expected(mu: float) -> np.ndarray:
        return np.asarray([50.0 + 12.0 * mu, 52.0 + 11.0 * mu])

    return benchmark.Engine(
        name=name,
        construction_seconds=construction,
        first_evaluation_seconds=first,
        evaluate=evaluate,
        expected=expected,
        nominal_parameters={"mu": [1.0]},
        parameter_order=["mu"],
        backend="fake",
        dtype="float64",
        compiled=name == "pyhs3",
        batched=False,
    )


def test_load_json_and_model_pairs(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text('{"answer": 42}', encoding="utf-8")
    assert benchmark._load_json(path) == {"answer": 42}

    pairs = benchmark._model_pairs(tmp_path)
    assert [pair.name for pair in pairs] == list(benchmark.MODEL_NAMES)
    assert pairs[0].hifa == tmp_path / "simplemodel_correlated-background_hifa.json"
    assert pairs[1].hs3 == tmp_path / "simplemodel_uncorrelated-background_hs3.json"


def test_timed_returns_value_and_nonnegative_duration() -> None:
    value, duration = benchmark._timed(lambda: "ok")
    assert value == "ok"
    assert duration >= 0.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3.5, 3.5),
        ([2.0], 2.0),
        (np.asarray([[4.0]]), 4.0),
    ],
)
def test_scalar_accepts_one_finite_value(value: Any, expected: float) -> None:
    assert benchmark._scalar(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        [],
        [1.0, 2.0],
        [np.nan],
        [np.inf],
    ],
)
def test_scalar_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(ValueError, match="Expected one finite scalar"):
        benchmark._scalar(value)


def test_workspace_structure() -> None:
    structure = benchmark.workspace_structure(_hifa_spec())
    assert structure["poi"] == "mu"
    assert structure["measurement"] == "measurement"
    assert structure["observations"][0]["data"] == [51.0, 48.0]
    assert structure["channels"][0]["samples"] == [
        {
            "name": "signal",
            "bins": 2,
            "modifiers": [{"name": "mu", "type": "normfactor"}],
        },
        {
            "name": "background",
            "bins": 2,
            "modifiers": [
                {
                    "name": "correlated_bkg_uncertainty",
                    "type": "histosys",
                }
            ],
        },
    ]


def test_replace_parameter_scalar_and_vector_do_not_mutate_input() -> None:
    original = {
        "mu": np.asarray(1.0),
        "vector": np.asarray([1.0, 2.0]),
    }

    scalar = benchmark._replace_parameter(original, "mu", 3.0)
    vector = benchmark._replace_parameter(original, "vector", 4.0)

    assert scalar["mu"].item() == 3.0
    np.testing.assert_allclose(vector["vector"], [4.0, 4.0])
    assert original["mu"].item() == 1.0
    np.testing.assert_allclose(original["vector"], [1.0, 2.0])


def test_replace_parameter_missing_name() -> None:
    with pytest.raises(KeyError, match="Parameter 'mu' is absent"):
        benchmark._replace_parameter({"alpha": np.asarray(0.0)}, "mu", 1.0)


def test_hs3_serialized_nominal_expected() -> None:
    spec = _hs3_spec()
    np.testing.assert_allclose(
        benchmark._hs3_serialized_nominal_expected(spec, 0.0),
        [50.0, 52.0],
    )
    np.testing.assert_allclose(
        benchmark._hs3_serialized_nominal_expected(spec, 2.0),
        [74.0, 74.0],
    )


def test_pyhs3_parameters() -> None:
    model = SimpleNamespace(
        parameterset=[
            SimpleNamespace(name="mu", value=1.0),
            SimpleNamespace(name="alpha", value=[0.0, 1.0]),
        ]
    )
    result = benchmark._pyhs3_parameters(model)
    assert result["mu"].dtype == np.float64
    np.testing.assert_allclose(result["alpha"], [0.0, 1.0])


def test_benchmark_evaluation_and_scan() -> None:
    calls: list[float] = []

    def evaluate(mu: float) -> float:
        calls.append(mu)
        return mu**2

    result = benchmark._benchmark_evaluation(
        evaluate,
        mu=2.0,
        repeats=4,
        warmups=3,
    )
    assert len(calls) == 7
    assert result["nll"] == 4.0
    assert result["repeats"] == 4
    assert result["warmups"] == 3
    assert result["min_seconds"] >= 0.0
    assert result["min_seconds"] <= result["median_seconds"] <= result["max_seconds"]

    engine = _fake_engine("fake")
    values, duration = benchmark._scan(
        engine,
        np.asarray([0.0, 1.0, 2.0]),
    )
    np.testing.assert_allclose(
        values,
        [engine.evaluate(0.0), engine.evaluate(1.0), engine.evaluate(2.0)],
    )
    assert duration >= 0.0


def test_agreement_with_constant_offset() -> None:
    mu_values = np.asarray([0.0, 0.5, 1.0])
    reference = np.asarray([1.0, 0.0, 1.0])
    shifted = reference + 7.5

    result = benchmark._agreement(
        shifted,
        reference,
        mu_values,
        rtol=1e-12,
        atol=1e-12,
    )

    assert result["raw_nll_constant_offset"] == pytest.approx(7.5)
    assert result["raw_nll_agrees_after_constant_offset"] is True
    assert result["delta_nll_agrees"] is True
    assert result["minimum_grid_difference"] == 0.0
    assert result["pyhs3_minimum_mu"] == 0.5
    assert result["pyhf_minimum_mu"] == 0.5


def test_agreement_detects_shape_difference() -> None:
    result = benchmark._agreement(
        np.asarray([0.0, 1.0, 5.0]),
        np.asarray([0.0, 1.0, 4.0]),
        np.asarray([0.0, 1.0, 2.0]),
        rtol=0.0,
        atol=1e-12,
    )
    assert result["delta_nll_agrees"] is False
    assert result["raw_nll_agrees_after_constant_offset"] is False


@pytest.mark.parametrize("n_bins", [1, 3, 5])
def test_repeat_correlated_hifa(n_bins: int) -> None:
    original = _hifa_spec()
    result = benchmark._repeat_correlated_hifa(original, n_bins)

    assert len(result["channels"][0]["samples"][0]["data"]) == n_bins
    assert len(result["channels"][0]["samples"][1]["data"]) == n_bins
    assert (
        len(result["channels"][0]["samples"][1]["modifiers"][0]["data"]["hi_data"])
        == n_bins
    )
    assert len(result["observations"][0]["data"]) == n_bins

    assert len(original["channels"][0]["samples"][0]["data"]) == 2


@pytest.mark.parametrize("n_bins", [1, 3, 5])
def test_repeat_correlated_hs3(n_bins: int) -> None:
    original = _hs3_spec()
    result = benchmark._repeat_correlated_hs3(original, n_bins)

    for datum in result["data"]:
        assert datum["axes"][0]["nbins"] == n_bins
        assert datum["axes"][0]["max"] == float(n_bins)
        assert len(datum["contents"]) == n_bins

    distribution = result["distributions"][0]
    assert distribution["axes"][0]["nbins"] == n_bins
    for sample in distribution["samples"]:
        assert len(sample["data"]["contents"]) == n_bins
        assert len(sample["data"]["errors"]) == n_bins

    histosys = distribution["samples"][1]["modifiers"][1]
    assert len(histosys["data"]["hi"]["contents"]) == n_bins
    assert len(histosys["data"]["lo"]["contents"]) == n_bins
    assert original["distributions"][0]["axes"][0]["nbins"] == 2


@pytest.mark.parametrize(
    "function",
    [
        benchmark._repeat_correlated_hifa,
        benchmark._repeat_correlated_hs3,
    ],
)
def test_repeat_rejects_nonpositive_bins(function: Any) -> None:
    spec = (
        _hifa_spec() if function is benchmark._repeat_correlated_hifa else _hs3_spec()
    )
    with pytest.raises(ValueError, match="n_bins must be positive"):
        function(spec, 0)


def test_scaling_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "_pyhf_engine",
        lambda spec: _fake_engine("pyhf-numpy"),
    )
    monkeypatch.setattr(
        benchmark,
        "_pyhs3_engine",
        lambda spec, analysis_name: _fake_engine("pyhs3"),
    )

    rows = benchmark._scaling(
        _hifa_spec(),
        _hs3_spec(),
        [2, 4],
        repeats=2,
        warmups=1,
    )

    assert len(rows) == 4
    assert all(row["status"] == "ok" for row in rows)
    assert {(row["number_of_bins"], row["engine"]) for row in rows} == {
        (2, "pyhs3"),
        (2, "pyhf-numpy"),
        (4, "pyhs3"),
        (4, "pyhf-numpy"),
    }


def test_scaling_underflow_and_skip_branch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        benchmark,
        "_pyhf_engine",
        lambda spec: _fake_engine("pyhf-numpy"),
    )

    calls = 0

    def fake_pyhs3(spec: dict[str, Any], analysis_name: str) -> benchmark.Engine:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("non-finite")
        return _fake_engine("pyhs3")

    monkeypatch.setattr(benchmark, "_pyhs3_engine", fake_pyhs3)

    rows = benchmark._scaling(
        _hifa_spec(),
        _hs3_spec(),
        [256, 512],
        repeats=1,
        warmups=0,
    )

    assert calls == 1
    assert rows[0]["status"] == "non-finite"
    assert rows[0]["number_of_bins"] == 256
    assert rows[2]["status"] == "skipped-after-underflow"
    assert rows[2]["number_of_bins"] == 512
    assert "Skipping pyHS3 scaling at 256 bins and above" in capsys.readouterr().out


def test_plot_functions_create_files(tmp_path: Path) -> None:
    mu_values = np.asarray([0.0, 1.0, 2.0])
    scans = {
        "pyhs3": np.asarray([1.0, 0.0, 1.0]),
        "pyhf-numpy": np.asarray([2.0, 1.0, 2.0]),
    }

    representative = benchmark._plot_representative_delta_nll(
        "correlated-background",
        mu_values,
        scans,
        tmp_path,
    )
    assert representative.is_file()

    results = [
        {
            "model": "correlated-background",
            "agreement": {
                "delta_nll_max_abs_difference": 1e-14,
                "raw_nll_offset_residual_max_abs": 2e-14,
            },
            "engines": {
                "pyhs3": {
                    "construction_seconds": 0.1,
                    "first_evaluation_seconds": 1.0,
                    "steady_state": {"median_seconds": 1e-5},
                },
                "pyhf-numpy": {
                    "construction_seconds": 0.01,
                    "first_evaluation_seconds": 0.001,
                    "steady_state": {"median_seconds": 1e-4},
                },
            },
        },
        {
            "model": "uncorrelated-background",
            "agreement": {
                "delta_nll_max_abs_difference": 2e-14,
                "raw_nll_offset_residual_max_abs": 3e-14,
            },
            "engines": {
                "pyhs3": {
                    "construction_seconds": 0.2,
                    "first_evaluation_seconds": 2.0,
                    "steady_state": {"median_seconds": 2e-5},
                },
                "pyhf-numpy": {
                    "construction_seconds": 0.02,
                    "first_evaluation_seconds": 0.002,
                    "steady_state": {"median_seconds": 2e-4},
                },
            },
        },
    ]

    validation = benchmark._plot_validation_summary(results, tmp_path)
    timing = benchmark._plot_timing_phases(results, tmp_path)
    assert validation.is_file()
    assert timing.is_file()

    rows = [
        {
            "number_of_bins": 2,
            "engine": "pyhs3",
            "status": "ok",
            "steady_state_median_seconds": 1e-5,
        },
        {
            "number_of_bins": 2,
            "engine": "pyhf-numpy",
            "status": "ok",
            "steady_state_median_seconds": 1e-4,
        },
        {
            "number_of_bins": 4,
            "engine": "pyhs3",
            "status": "non-finite",
        },
        {
            "number_of_bins": 4,
            "engine": "pyhf-numpy",
            "status": "ok",
            "steady_state_median_seconds": 2e-4,
        },
        {
            "number_of_bins": 8,
            "engine": "unused-engine",
            "status": "skipped-after-underflow",
        },
    ]
    scaling = benchmark._plot_scaling_metric(
        rows,
        tmp_path,
        metric="steady_state_median_seconds",
        ylabel="Runtime",
        title="Scaling",
        filename="scaling.png",
    )
    assert scaling.is_file()


def test_summary_table_success_and_failure() -> None:
    base_engine = {
        "compiled": True,
        "batched": False,
        "steady_state": {"median_seconds": 1e-5},
    }
    results = [
        {
            "model": "good",
            "agreement": {
                "expected_values_agree": True,
                "delta_nll_agrees": True,
            },
            "engines": {
                "pyhs3": dict(base_engine),
                "pyhf-numpy": {
                    **base_engine,
                    "compiled": False,
                },
            },
        },
        {
            "model": "bad",
            "agreement": {
                "expected_values_agree": False,
                "delta_nll_agrees": True,
            },
            "engines": {"pyhs3": dict(base_engine)},
        },
    ]

    rows = benchmark._summary_table(results)
    assert rows[0]["agreement"] == "expected + Delta NLL agree"
    assert rows[0]["execution_mode"] == "compiled/warm PyTensor function"
    assert rows[1]["execution_mode"] == "warm NumPy function call"
    assert rows[2]["agreement"] == "agreement failed"


def test_pyhf_engine_with_mocked_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeConfig:
        poi_index = 0
        par_order = ["mu", "alpha"]

        @staticmethod
        def suggested_init() -> list[float]:
            return [1.0, 0.0]

        @staticmethod
        def par_slice(name: str) -> slice:
            return {"mu": slice(0, 1), "alpha": slice(1, 2)}[name]

    class FakeModel:
        config = FakeConfig()

        @staticmethod
        def logpdf(parameters: np.ndarray, data: np.ndarray) -> np.ndarray:
            assert data.tolist() == [51.0, 48.0]
            return np.asarray([-((parameters[0] - 1.0) ** 2) - 3.0])

        @staticmethod
        def expected_actualdata(parameters: np.ndarray) -> np.ndarray:
            return np.asarray([50.0 + parameters[0], 52.0 + parameters[0]])

    class FakeWorkspace:
        def __init__(self, spec: dict[str, Any], validate: bool) -> None:
            assert validate is True

        @staticmethod
        def model(measurement_name: str) -> FakeModel:
            assert measurement_name == "measurement"
            return FakeModel()

        @staticmethod
        def data(model: FakeModel) -> list[float]:
            return [51.0, 48.0]

    monkeypatch.setattr(benchmark.pyhf, "set_backend", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark.pyhf, "Workspace", FakeWorkspace)

    engine = benchmark._pyhf_engine(_hifa_spec())
    assert engine.name == "pyhf-numpy"
    assert engine.evaluate(1.0) == pytest.approx(3.0)
    np.testing.assert_allclose(engine.expected(2.0), [52.0, 54.0])
    assert engine.nominal_parameters == {"mu": [1.0], "alpha": [0.0]}
    assert engine.parameter_order == ["mu", "alpha"]


def test_pyhs3_engine_with_mocked_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        parameterset = [
            SimpleNamespace(name="Lumi", value=1.0),
            SimpleNamespace(name="alpha_correlated_bkg_uncertainty", value=0.0),
            SimpleNamespace(name="mu", value=1.0),
        ]

        @staticmethod
        def logpdf(name: str, **parameters: np.ndarray) -> np.ndarray:
            assert name == "model_singlechannel"
            mu = float(np.asarray(parameters["mu"]))
            return np.asarray([-((mu - 1.0) ** 2) - 4.0])

    class FakeWorkspace:
        def __init__(self, **spec: Any) -> None:
            assert "distributions" in spec

        @staticmethod
        def model(
            analysis_name: str,
            parameter_set: str,
            progress: bool,
        ) -> FakeModel:
            assert analysis_name == "simPdf_obsData"
            assert parameter_set == "default_values"
            assert progress is False
            return FakeModel()

    monkeypatch.setattr(benchmark, "PyHS3Workspace", FakeWorkspace)

    engine = benchmark._pyhs3_engine(_hs3_spec(), "simPdf_obsData")
    assert engine.name == "pyhs3"
    assert engine.evaluate(1.0) == pytest.approx(4.0)
    np.testing.assert_allclose(engine.expected(1.0), [62.0, 63.0])
    assert engine.compiled is True
    assert engine.parameter_order == [
        "Lumi",
        "alpha_correlated_bkg_uncertainty",
        "mu",
    ]


def _write_model_files(input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for name in benchmark.MODEL_NAMES:
        (input_dir / f"simplemodel_{name}_hifa.json").write_text(
            json.dumps(_hifa_spec()),
            encoding="utf-8",
        )
        (input_dir / f"simplemodel_{name}_hs3.json").write_text(
            json.dumps(_hs3_spec()),
            encoding="utf-8",
        )


def test_run_end_to_end_with_mock_engines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    results_dir = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    _write_model_files(input_dir)

    monkeypatch.setattr(
        benchmark,
        "_pyhf_engine",
        lambda spec: _fake_engine("pyhf-numpy", offset=0.0),
    )
    monkeypatch.setattr(
        benchmark,
        "_pyhs3_engine",
        lambda spec, analysis_name: _fake_engine("pyhs3", offset=5.0),
    )

    args = argparse.Namespace(
        input_dir=input_dir,
        results_dir=results_dir,
        plots_dir=plots_dir,
        mu_values=np.asarray([0.0, 0.5, 1.0]),
        repeats=2,
        warmups=1,
        scaling_repeats=1,
        scaling_bins=[2, 4],
        rtol=1e-12,
        atol=1e-12,
    )

    payload = benchmark.run(args)

    assert payload["benchmark"] == benchmark.BENCHMARK_NAME
    assert len(payload["models"]) == 2
    assert len(payload["summary"]) == 4
    assert payload["scaling"]["pyhs3_largest_finite_bin_count"] == 4

    output = results_dir / "results.json"
    assert output.is_file()
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["models"][0]["agreement"]["delta_nll_agrees"] is True

    assert (plots_dir / "representative_delta_nll.png").is_file()
    assert (plots_dir / "numerical_agreement_summary.png").is_file()
    assert (plots_dir / "timing_phases.png").is_file()
    assert (plots_dir / "warm_function_call_vs_number_of_bins.png").is_file()


def test_parser_defaults() -> None:
    parser = benchmark._parser()
    args = parser.parse_args([])
    assert args.input_dir == benchmark.DEFAULT_INPUT_DIR
    assert args.mu_points == 61
    assert args.scaling_bins == [2, 4, 8, 16, 32, 64, 128]


def test_main_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        mu_min=0.0,
        mu_max=3.0,
        mu_points=3,
    )
    monkeypatch.setattr(
        benchmark,
        "_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: args,
            error=lambda message: pytest.fail(message),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "run",
        lambda parsed: {
            "summary": [{"framework": "fake"}],
            "models": [
                {
                    "model": "good",
                    "agreement": {
                        "expected_values_agree": True,
                        "delta_nll_agrees": True,
                    },
                }
            ],
        },
    )

    benchmark.main()
    assert '"framework": "fake"' in capsys.readouterr().out
    np.testing.assert_allclose(args.mu_values, [0.0, 1.5, 3.0])


def test_main_numerical_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    args = argparse.Namespace(
        mu_min=0.0,
        mu_max=3.0,
        mu_points=2,
    )
    monkeypatch.setattr(
        benchmark,
        "_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: args,
            error=lambda message: pytest.fail(message),
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "run",
        lambda parsed: {
            "summary": [],
            "models": [
                {
                    "model": "bad-model",
                    "agreement": {
                        "expected_values_agree": True,
                        "delta_nll_agrees": False,
                    },
                }
            ],
        },
    )

    with pytest.raises(SystemExit, match="Numerical validation failed for: bad-model"):
        benchmark.main()


def test_main_rejects_too_few_mu_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = argparse.Namespace(mu_points=1)

    def error(message: str) -> None:
        raise RuntimeError(message)

    monkeypatch.setattr(
        benchmark,
        "_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: args,
            error=error,
        ),
    )

    with pytest.raises(RuntimeError, match="--mu-points must be at least 2"):
        benchmark.main()
