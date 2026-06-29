from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pyhs3.workspace import Workspace

from src import generate_scalar_pdf_workspaces as generator


@pytest.mark.parametrize(
    ("scenario", "x_value"),
    [
        ("normal", 0.0),
        ("poisson", 5.0),
        ("exponential", 1.0),
    ],
)
def test_make_workspace_basic_structure(scenario: str, x_value: float) -> None:
    workspace = generator.make_workspace(scenario)
    payload = workspace.model_dump(mode="json", exclude_none=True)

    assert payload["metadata"]["hs3_version"] == "0.2"
    assert scenario in payload["metadata"]["description"]
    assert len(payload["distributions"]) == 1
    assert len(payload["domains"]) == 1
    assert len(payload["data"]) == 1
    assert len(payload["likelihoods"]) == 1
    assert len(payload["analyses"]) == 1

    assert payload["distributions"][0]["name"] == "pdf"
    assert payload["domains"][0]["name"] == "domain_x"
    assert payload["data"][0]["name"] == "data_x"
    assert payload["data"][0]["value"] == x_value
    assert payload["likelihoods"][0]["name"] == "likelihood"
    assert payload["likelihoods"][0]["distributions"] == ["pdf"]
    assert payload["likelihoods"][0]["data"] == ["data_x"]
    assert payload["analyses"][0]["name"] == "analysis"
    assert payload["analyses"][0]["likelihood"] == "likelihood"
    assert payload["analyses"][0]["parameters_of_interest"] == ["x"]
    assert payload["analyses"][0]["domains"] == ["domain_x"]
    assert payload["analyses"][0]["init"] == "init"


@pytest.mark.parametrize(
    ("scenario", "expected_min", "expected_max"),
    [
        ("normal", -10.0, 10.0),
        ("poisson", 0.0, 30.0),
        ("exponential", 0.0, 10.0),
    ],
)
def test_make_workspace_domain_ranges(
    scenario: str,
    expected_min: float,
    expected_max: float,
) -> None:
    workspace = generator.make_workspace(scenario)
    payload = workspace.model_dump(mode="json", exclude_none=True)

    axis = payload["domains"][0]["axes"][0]

    assert axis["name"] == "x"
    assert axis["min"] == expected_min
    assert axis["max"] == expected_max


@pytest.mark.parametrize(
    ("scenario", "expected_parameters"),
    [
        (
            "normal",
            {
                "x": {"value": 0.0, "const": False},
                "mu": {"value": 0.0, "const": True},
                "sigma": {"value": 1.0, "const": True},
            },
        ),
        (
            "poisson",
            {
                "x": {"value": 5.0, "const": False},
                "mean": {"value": 5.0, "const": True},
            },
        ),
        (
            "exponential",
            {
                "x": {"value": 1.0, "const": False},
                "c": {"value": -1.0, "const": True},
            },
        ),
    ],
)
def test_make_workspace_parameter_points(
    scenario: str,
    expected_parameters: dict[str, dict[str, Any]],
) -> None:
    workspace = generator.make_workspace(scenario)
    payload = workspace.model_dump(mode="json", exclude_none=True)

    parameter_point = payload["parameter_points"][0]
    parameters = {
        parameter["name"]: parameter for parameter in parameter_point["parameters"]
    }

    assert parameter_point["name"] == "init"
    assert set(parameters) == set(expected_parameters)

    for name, expected in expected_parameters.items():
        assert parameters[name]["value"] == expected["value"]
        assert parameters[name]["const"] is expected["const"]


@pytest.mark.parametrize("scenario", ["normal", "poisson", "exponential"])
def test_make_workspace_can_create_model_and_pdf(scenario: str) -> None:
    workspace = generator.make_workspace(scenario)
    model = workspace.model("analysis", progress=False, mode="FAST_RUN")

    assert "pdf" in model.distributions

    parameters = {
        name: np.asarray(value, dtype=float)
        for name, value in {
            **model.data,
            **model.free_params,
        }.items()
    }

    result = model.pdf("pdf", **parameters)
    result_array = np.asarray(result, dtype=float)

    assert result_array.size > 0
    assert np.all(np.isfinite(result_array))


def test_make_workspace_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="Unknown scenario: unknown"):
        generator.make_workspace("unknown")


@pytest.mark.parametrize("scenario", ["normal", "poisson", "exponential"])
def test_save_and_load_workspace(tmp_path: Path, scenario: str) -> None:
    workspace = generator.make_workspace(scenario)
    output_path = tmp_path / f"{scenario}_pdf_workspace.json"

    generator.save_workspace(workspace, output_path)

    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["analyses"][0]["name"] == "analysis"
    assert payload["distributions"][0]["name"] == "pdf"

    loaded = Workspace.load(output_path)
    model = loaded.model("analysis", progress=False, mode="FAST_RUN")
    assert "pdf" in model.distributions


def test_save_workspace_creates_parent_directories_and_sorted_json(
    tmp_path: Path,
) -> None:
    workspace = generator.make_workspace("normal")
    output_path = tmp_path / "nested" / "normal_pdf_workspace.json"

    generator.save_workspace(workspace, output_path)

    assert output_path.exists()
    text = output_path.read_text()
    assert json.loads(text)["analyses"][0]["name"] == "analysis"
    assert text.index('"analyses"') < text.index('"data"')


@pytest.mark.parametrize("scenario", ["normal", "poisson", "exponential"])
def test_validate_workspace_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    scenario: str,
) -> None:
    workspace = generator.make_workspace(scenario)
    output_path = tmp_path / f"{scenario}_pdf_workspace.json"
    generator.save_workspace(workspace, output_path)

    generator.validate_workspace(output_path=output_path, scenario=scenario)

    output = capsys.readouterr().out
    assert f"{scenario}: validation output" in output


def test_validate_workspace_rejects_missing_pdf_distribution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_model = SimpleNamespace(distributions={})
    fake_workspace = SimpleNamespace(
        model=lambda *args, **kwargs: fake_model,
    )

    monkeypatch.setattr(
        generator.Workspace,
        "load",
        lambda output_path: fake_workspace,
    )

    with pytest.raises(ValueError, match="does not expose distribution 'pdf'"):
        generator.validate_workspace(
            output_path=tmp_path / "broken.json",
            scenario="normal",
        )


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["generate_scalar_pdf_workspaces.py"])

    args = generator.parse_args()

    assert args.output_dir == generator.DEFAULT_OUTPUT_DIR
    assert args.scenarios == ["normal", "poisson", "exponential"]
    assert args.validate is False


def test_parse_args_custom_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_scalar_pdf_workspaces.py",
            "--output-dir",
            str(tmp_path),
            "--scenarios",
            "normal",
            "poisson",
            "--validate",
        ],
    )

    args = generator.parse_args()

    assert args.output_dir == tmp_path
    assert args.scenarios == ["normal", "poisson"]
    assert args.validate is True


def test_parse_args_rejects_unknown_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_scalar_pdf_workspaces.py",
            "--scenarios",
            "unknown",
        ],
    )

    with pytest.raises(SystemExit):
        generator.parse_args()


def test_main_creates_default_workspace_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_scalar_pdf_workspaces.py",
            "--output-dir",
            str(tmp_path),
        ],
    )

    generator.main()

    for scenario in ["normal", "poisson", "exponential"]:
        output_path = tmp_path / f"{scenario}_pdf_workspace.json"
        assert output_path.exists()
        payload = json.loads(output_path.read_text())
        assert payload["analyses"][0]["name"] == "analysis"
        assert payload["distributions"][0]["name"] == "pdf"


def test_main_creates_only_selected_scenarios(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_scalar_pdf_workspaces.py",
            "--output-dir",
            str(tmp_path),
            "--scenarios",
            "normal",
        ],
    )

    generator.main()

    assert (tmp_path / "normal_pdf_workspace.json").exists()
    assert not (tmp_path / "poisson_pdf_workspace.json").exists()
    assert not (tmp_path / "exponential_pdf_workspace.json").exists()


def test_main_with_validate_calls_validate_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, str]] = []

    def fake_validate_workspace(output_path: Path, scenario: str) -> None:
        calls.append((output_path, scenario))

    monkeypatch.setattr(generator, "validate_workspace", fake_validate_workspace)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_scalar_pdf_workspaces.py",
            "--output-dir",
            str(tmp_path),
            "--scenarios",
            "normal",
            "exponential",
            "--validate",
        ],
    )

    generator.main()

    assert calls == [
        (tmp_path / "normal_pdf_workspace.json", "normal"),
        (tmp_path / "exponential_pdf_workspace.json", "exponential"),
    ]


def test_main_prints_saved_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_scalar_pdf_workspaces.py",
            "--output-dir",
            str(tmp_path),
            "--scenarios",
            "normal",
        ],
    )

    generator.main()

    output = capsys.readouterr().out
    assert "Saved normal workspace" in output
    assert str(tmp_path / "normal_pdf_workspace.json") in output


@pytest.mark.parametrize("scenario", ["normal", "poisson", "exponential"])
def test_validate_scenario_accepts_known_scenarios(scenario: str) -> None:
    assert generator.validate_scenario(scenario) == scenario


def test_validate_scenario_rejects_unknown_scenario() -> None:
    with pytest.raises(ValueError, match="Unknown scenario: unknown"):
        generator.validate_scenario("unknown")


def test_validate_scenarios_accepts_multiple_values() -> None:
    assert generator.validate_scenarios(["normal", "poisson"]) == ["normal", "poisson"]


def test_validate_scenarios_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="--scenarios must contain at least one value"):
        generator.validate_scenarios([])


def test_verify_output_file_success(tmp_path: Path) -> None:
    output_path = tmp_path / "workspace.json"
    output_path.write_text("{}")

    generator.verify_output_file(output_path)


def test_verify_output_file_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace file was not created"):
        generator.verify_output_file(tmp_path / "missing.json")


def test_verify_output_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Workspace output path is not a file"):
        generator.verify_output_file(tmp_path)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (0.25, 0.25),
        ([0.5], 0.5),
        (np.asarray([[0.75]]), 0.75),
    ],
)
def test_extract_scalar_output_success(result: Any, expected: float) -> None:
    assert generator.extract_scalar_output(result) == pytest.approx(expected)


def test_extract_scalar_output_rejects_empty_output() -> None:
    with pytest.raises(ValueError, match="Validation PDF output is empty"):
        generator.extract_scalar_output([])


@pytest.mark.parametrize("result", [float("nan"), float("inf"), [float("-inf")]])
def test_extract_scalar_output_rejects_non_finite_output(result: Any) -> None:
    with pytest.raises(ValueError, match="Validation PDF output is not finite"):
        generator.extract_scalar_output(result)


def test_generate_single_workspace_without_validation(tmp_path: Path) -> None:
    result = generator.generate_single_workspace(
        scenario="normal",
        output_dir=tmp_path,
        validate=False,
    )

    assert result == {
        "scenario": "normal",
        "output_path": tmp_path / "normal_pdf_workspace.json",
        "status": "success",
    }
    assert (tmp_path / "normal_pdf_workspace.json").is_file()


def test_generate_single_workspace_with_validation_calls_validate_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, str]] = []

    def fake_validate_workspace(output_path: Path, scenario: str) -> None:
        calls.append((output_path, scenario))

    monkeypatch.setattr(generator, "validate_workspace", fake_validate_workspace)

    result = generator.generate_single_workspace(
        scenario="poisson",
        output_dir=tmp_path,
        validate=True,
    )

    assert result["scenario"] == "poisson"
    assert calls == [(tmp_path / "poisson_pdf_workspace.json", "poisson")]


def test_print_success_outputs_saved_path(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "normal_pdf_workspace.json"

    generator.print_success(
        {
            "scenario": "normal",
            "output_path": output_path,
            "status": "success",
        }
    )

    output = capsys.readouterr().out
    assert "Saved normal workspace" in output
    assert str(output_path) in output


def test_print_failure_outputs_error_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    generator.print_failure("normal", RuntimeError("boom"))

    output = capsys.readouterr().out
    assert "Scalar PDF workspace generation FAILED" in output
    assert "Scenario: normal" in output
    assert "RuntimeError: boom" in output


def test_main_reports_failures_and_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_parse_args() -> SimpleNamespace:
        return SimpleNamespace(
            output_dir=tmp_path,
            scenarios=["normal", "poisson"],
            validate=False,
        )

    def fake_generate_single_workspace(
        scenario: str,
        output_dir: Path,
        validate: bool,
    ) -> dict[str, Any]:
        if scenario == "normal":
            raise RuntimeError("generation failed")
        return {
            "scenario": scenario,
            "output_path": output_dir / f"{scenario}_pdf_workspace.json",
            "status": "success",
        }

    monkeypatch.setattr(generator, "parse_args", fake_parse_args)
    monkeypatch.setattr(
        generator, "generate_single_workspace", fake_generate_single_workspace
    )

    with pytest.raises(SystemExit) as exc_info:
        generator.main()

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Scalar PDF workspace generation FAILED" in output
    assert "Scalar PDF workspace generation summary" in output
    assert "Succeeded: 1" in output
    assert "Failed:    1" in output
    assert "normal: RuntimeError: generation failed" in output
    assert "Saved poisson workspace" in output


def test_main_rejects_empty_scenario_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        generator,
        "parse_args",
        lambda: SimpleNamespace(output_dir=tmp_path, scenarios=[], validate=False),
    )

    with pytest.raises(ValueError, match="--scenarios must contain at least one value"):
        generator.main()


def test_module_main_guard_runs_successfully(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_scalar_pdf_workspaces.py",
            "--output-dir",
            str(tmp_path),
            "--scenarios",
            "normal",
        ],
    )

    runpy.run_module("src.generate_scalar_pdf_workspaces", run_name="__main__")

    assert (tmp_path / "normal_pdf_workspace.json").is_file()
