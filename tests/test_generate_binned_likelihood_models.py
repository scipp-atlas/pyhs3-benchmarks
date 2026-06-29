from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from pyhs3.workspace import Workspace

from src import generate_binned_likelihood_models as generator


@pytest.mark.parametrize("n_bins", [1, 3, 30])
def test_make_arrays_has_expected_lengths(n_bins: int) -> None:
    arrays = generator.make_arrays(n_bins=n_bins)

    assert set(arrays) == {"signal", "background", "observation"}
    assert len(arrays["signal"]) == n_bins
    assert len(arrays["background"]) == n_bins
    assert len(arrays["observation"]) == n_bins


@pytest.mark.parametrize("n_bins", [3, 30])
def test_make_arrays_is_deterministic(n_bins: int) -> None:
    first = generator.make_arrays(n_bins=n_bins)
    second = generator.make_arrays(n_bins=n_bins)

    assert first == second


def test_make_arrays_observation_matches_rounded_signal_plus_background() -> None:
    arrays = generator.make_arrays(n_bins=5)

    expected_observation = [
        float(round(signal + background))
        for signal, background in zip(
            arrays["signal"],
            arrays["background"],
            strict=True,
        )
    ]

    assert arrays["observation"] == expected_observation


def test_make_arrays_values_are_in_expected_ranges() -> None:
    arrays = generator.make_arrays(n_bins=10)

    assert all(2.0 <= value <= 10.0 for value in arrays["signal"])
    assert all(15.0 <= value <= 35.0 for value in arrays["background"])
    assert all(value >= 0.0 for value in arrays["observation"])


def test_make_pyhf_spec_structure() -> None:
    model = {
        "signal": [2.0, 3.0],
        "background": [10.0, 11.0],
        "observation": [12.0, 14.0],
    }

    spec = generator.make_pyhf_spec(model)

    assert spec["version"] == "1.0.0"
    assert len(spec["channels"]) == 1
    assert len(spec["observations"]) == 1
    assert len(spec["measurements"]) == 1

    channel = spec["channels"][0]
    assert channel["name"] == "singlechannel"
    assert [sample["name"] for sample in channel["samples"]] == ["signal", "background"]

    measurement = spec["measurements"][0]
    assert measurement["name"] == "measurement"
    assert measurement["config"]["poi"] == "mu"


def test_make_pyhf_spec_uses_input_arrays() -> None:
    model = {
        "signal": [2.0, 3.0],
        "background": [10.0, 11.0],
        "observation": [12.0, 14.0],
    }

    spec = generator.make_pyhf_spec(model)
    signal_sample = spec["channels"][0]["samples"][0]
    background_sample = spec["channels"][0]["samples"][1]

    assert signal_sample["data"] == model["signal"]
    assert background_sample["data"] == model["background"]
    assert spec["observations"][0]["data"] == model["observation"]


def test_make_pyhf_spec_signal_has_mu_normfactor() -> None:
    model = generator.make_arrays(n_bins=3)

    spec = generator.make_pyhf_spec(model)
    signal_sample = spec["channels"][0]["samples"][0]

    assert signal_sample["modifiers"] == [
        {
            "name": "mu",
            "type": "normfactor",
            "data": None,
        }
    ]


def test_make_pyhs3_workspace_basic_structure() -> None:
    n_bins = 3
    model = generator.make_arrays(n_bins=n_bins)

    workspace = generator.make_pyhs3_workspace(model=model, n_bins=n_bins)
    payload = workspace.model_dump(mode="json", exclude_none=True)

    assert payload["metadata"]["hs3_version"] == "0.2"
    assert len(payload["distributions"]) == n_bins
    assert len(payload["data"]) == n_bins
    assert len(payload["domains"]) == n_bins
    assert len(payload["likelihoods"]) == 1
    assert len(payload["analyses"]) == 1
    assert payload["analyses"][0]["name"] == "analysis"
    assert payload["analyses"][0]["likelihood"] == "likelihood"
    assert payload["analyses"][0]["parameters_of_interest"] == ["mu"]


def test_make_pyhs3_workspace_creates_expected_per_bin_objects() -> None:
    n_bins = 3
    model = generator.make_arrays(n_bins=n_bins)

    workspace = generator.make_pyhs3_workspace(model=model, n_bins=n_bins)
    payload = workspace.model_dump(mode="json", exclude_none=True)

    distribution_names = {item["name"] for item in payload["distributions"]}
    function_names = {item["name"] for item in payload["functions"]}
    data_names = {item["name"] for item in payload["data"]}
    domain_names = {item["name"] for item in payload["domains"]}

    for index in range(n_bins):
        assert f"poisson_{index}" in distribution_names
        assert f"scaled_signal_{index}" in function_names
        assert f"expected_{index}" in function_names
        assert f"data_{index}" in data_names
        assert f"domain_{index}" in domain_names


@pytest.mark.parametrize("n_bins", [1, 3])
def test_save_and_load_pyhs3_workspace(tmp_path: Path, n_bins: int) -> None:
    model = generator.make_arrays(n_bins=n_bins)
    workspace = generator.make_pyhs3_workspace(model=model, n_bins=n_bins)
    output_path = tmp_path / f"pyhs3_{n_bins}bins.json"

    generator.save_workspace(workspace, output_path)

    assert output_path.exists()
    loaded = Workspace.load(output_path)
    loaded_model = loaded.model("analysis", progress=False, mode="FAST_RUN")

    assert len(loaded_model.distributions) == n_bins


def test_save_json_creates_parent_directories_and_sorted_json(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "model.json"

    generator.save_json({"b": 2, "a": 1}, output_path)

    assert output_path.exists()
    assert json.loads(output_path.read_text()) == {"a": 1, "b": 2}
    assert output_path.read_text().index('"a"') < output_path.read_text().index('"b"')


def test_validate_workspace_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    n_bins = 3
    model = generator.make_arrays(n_bins=n_bins)
    workspace = generator.make_pyhs3_workspace(model=model, n_bins=n_bins)
    output_path = tmp_path / "pyhs3_3bins.json"
    generator.save_workspace(workspace, output_path)

    generator.validate_workspace(output_path, n_bins=n_bins)

    output = capsys.readouterr().out
    assert "3 bins: validation successful" in output


def test_validate_workspace_rejects_wrong_distribution_count(tmp_path: Path) -> None:
    n_bins = 3
    model = generator.make_arrays(n_bins=n_bins)
    workspace = generator.make_pyhs3_workspace(model=model, n_bins=n_bins)
    output_path = tmp_path / "pyhs3_3bins.json"
    generator.save_workspace(workspace, output_path)

    with pytest.raises(ValueError, match="Expected 4 distributions"):
        generator.validate_workspace(output_path, n_bins=4)


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["generate_binned_likelihood_models.py"])

    args = generator.parse_args()

    assert args.output_dir == generator.DEFAULT_OUTPUT_DIR
    assert args.bin_counts == generator.DEFAULT_BIN_COUNTS
    assert args.validate is False


def test_parse_args_custom_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_binned_likelihood_models.py",
            "--output-dir",
            str(tmp_path),
            "--bin-counts",
            "2",
            "4",
            "--validate",
        ],
    )

    args = generator.parse_args()

    assert args.output_dir == tmp_path
    assert args.bin_counts == [2, 4]
    assert args.validate is True


def test_main_creates_all_expected_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_binned_likelihood_models.py",
            "--output-dir",
            str(tmp_path),
            "--bin-counts",
            "2",
            "4",
        ],
    )

    generator.main()

    for n_bins in [2, 4]:
        assert (tmp_path / f"common_{n_bins}bins.json").exists()
        assert (tmp_path / f"pyhf_{n_bins}bins.json").exists()
        assert (tmp_path / f"pyhs3_{n_bins}bins.json").exists()

        common = json.loads((tmp_path / f"common_{n_bins}bins.json").read_text())
        pyhf = json.loads((tmp_path / f"pyhf_{n_bins}bins.json").read_text())
        pyhs3 = json.loads((tmp_path / f"pyhs3_{n_bins}bins.json").read_text())

        assert common["n_bins"] == n_bins
        assert len(common["signal"]) == n_bins
        assert pyhf["measurements"][0]["config"]["poi"] == "mu"
        assert pyhs3["analyses"][0]["name"] == "analysis"


def test_main_with_validate_calls_validate_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, int]] = []

    def fake_validate_workspace(workspace_path: Path, n_bins: int) -> None:
        calls.append((workspace_path, n_bins))

    monkeypatch.setattr(generator, "validate_workspace", fake_validate_workspace)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_binned_likelihood_models.py",
            "--output-dir",
            str(tmp_path),
            "--bin-counts",
            "2",
            "--validate",
        ],
    )

    generator.main()

    assert calls == [(tmp_path / "pyhs3_2bins.json", 2)]


def test_main_prints_saved_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_binned_likelihood_models.py",
            "--output-dir",
            str(tmp_path),
            "--bin-counts",
            "2",
        ],
    )

    generator.main()

    output = capsys.readouterr().out
    assert "Saved common model" in output
    assert "Saved pyhf model" in output
    assert "Saved pyhs3 model" in output


def test_validate_bin_count_success() -> None:
    assert generator.validate_bin_count(3) == 3


@pytest.mark.parametrize(
    ("value", "exc_type", "message"),
    [
        (0, ValueError, "n_bins must be at least 1"),
        (-2, ValueError, "n_bins must be at least 1"),
        (1.5, TypeError, "n_bins must be an integer"),
        ("3", TypeError, "n_bins must be an integer"),
    ],
)
def test_validate_bin_count_rejects_invalid_values(
    value: Any,
    exc_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exc_type, match=message):
        generator.validate_bin_count(value)  # type: ignore[arg-type]


def test_validate_bin_counts_success() -> None:
    assert generator.validate_bin_counts([1, 3, 30]) == [1, 3, 30]


def test_validate_bin_counts_rejects_empty_list() -> None:
    with pytest.raises(
        ValueError, match="--bin-counts must contain at least one value"
    ):
        generator.validate_bin_counts([])


@pytest.mark.parametrize(
    ("model", "message"),
    [
        ({"signal": [1.0], "background": [2.0]}, "missing arrays"),
        (
            {"signal": [1.0], "background": [2.0], "observation": [3.0, 4.0]},
            "Array 'observation' has length 2, expected 1",
        ),
        (
            {"signal": [float("nan")], "background": [2.0], "observation": [3.0]},
            "Array 'signal' contains non-finite values",
        ),
        (
            {"signal": [1.0], "background": [float("inf")], "observation": [3.0]},
            "Array 'background' contains non-finite values",
        ),
    ],
)
def test_validate_model_arrays_rejects_invalid_model(
    model: dict[str, list[float]],
    message: str,
) -> None:
    with pytest.raises((KeyError, ValueError), match=message):
        generator.validate_model_arrays(model, n_bins=1)


def test_make_arrays_rejects_invalid_bin_count() -> None:
    with pytest.raises(ValueError, match="n_bins must be at least 1"):
        generator.make_arrays(0)


def test_make_pyhs3_workspace_rejects_invalid_bin_count() -> None:
    with pytest.raises(ValueError, match="n_bins must be at least 1"):
        generator.make_pyhs3_workspace(
            model={"signal": [], "background": [], "observation": []},
            n_bins=0,
        )


def test_make_pyhs3_workspace_rejects_invalid_model_arrays() -> None:
    with pytest.raises(KeyError, match="missing arrays"):
        generator.make_pyhs3_workspace(
            model={"signal": [1.0], "background": [2.0]},
            n_bins=1,
        )


def test_verify_output_file_success(tmp_path: Path) -> None:
    path = tmp_path / "output.json"
    path.write_text("{}")

    generator.verify_output_file(path)


def test_verify_output_file_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Expected output file was not created"):
        generator.verify_output_file(tmp_path / "missing.json")


def test_verify_output_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Expected output path is not a file"):
        generator.verify_output_file(tmp_path)


def test_generate_single_model_success_without_validation(tmp_path: Path) -> None:
    result = generator.generate_single_model(
        n_bins=2,
        output_dir=tmp_path,
        validate=False,
    )

    assert result == {
        "n_bins": 2,
        "common_path": tmp_path / "common_2bins.json",
        "pyhf_path": tmp_path / "pyhf_2bins.json",
        "pyhs3_path": tmp_path / "pyhs3_2bins.json",
        "status": "success",
    }
    assert result["common_path"].exists()
    assert result["pyhf_path"].exists()
    assert result["pyhs3_path"].exists()


def test_generate_single_model_with_validation_calls_validate_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, int]] = []

    def fake_validate_workspace(workspace_path: Path, n_bins: int) -> None:
        calls.append((workspace_path, n_bins))

    monkeypatch.setattr(generator, "validate_workspace", fake_validate_workspace)

    result = generator.generate_single_model(
        n_bins=2,
        output_dir=tmp_path,
        validate=True,
    )

    assert result["status"] == "success"
    assert calls == [(tmp_path / "pyhs3_2bins.json", 2)]


def test_generate_single_model_rejects_invalid_bin_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_bins must be at least 1"):
        generator.generate_single_model(
            n_bins=0,
            output_dir=tmp_path,
            validate=False,
        )


def test_print_success_outputs_all_paths(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    generator.print_success(
        {
            "common_path": tmp_path / "common.json",
            "pyhf_path": tmp_path / "pyhf.json",
            "pyhs3_path": tmp_path / "pyhs3.json",
        }
    )

    output = capsys.readouterr().out
    assert "Saved common model" in output
    assert "Saved pyhf model" in output
    assert "Saved pyhs3 model" in output


def test_print_failure_outputs_error(capsys: pytest.CaptureFixture[str]) -> None:
    generator.print_failure(3, RuntimeError("boom"))

    output = capsys.readouterr().out
    assert "Binned likelihood generation FAILED" in output
    assert "Bins:   3" in output
    assert "RuntimeError: boom" in output


def test_main_rejects_empty_bin_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        generator,
        "parse_args",
        lambda: type(
            "Args",
            (),
            {
                "output_dir": Path("unused"),
                "bin_counts": [],
                "validate": False,
            },
        )(),
    )

    with pytest.raises(
        ValueError, match="--bin-counts must contain at least one value"
    ):
        generator.main()


def test_main_prints_failure_summary_and_exits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_generate_single_model(
        n_bins: int, output_dir: Path, validate: bool
    ) -> dict[str, Any]:
        if n_bins == 2:
            raise RuntimeError("boom")
        return {
            "n_bins": n_bins,
            "common_path": output_dir / f"common_{n_bins}bins.json",
            "pyhf_path": output_dir / f"pyhf_{n_bins}bins.json",
            "pyhs3_path": output_dir / f"pyhs3_{n_bins}bins.json",
            "status": "success",
        }

    monkeypatch.setattr(generator, "generate_single_model", fake_generate_single_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_binned_likelihood_models.py",
            "--output-dir",
            str(tmp_path),
            "--bin-counts",
            "1",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        generator.main()

    assert exc_info.value.code == 1
    output = capsys.readouterr().out
    assert "Saved common model" in output
    assert "Binned likelihood generation FAILED" in output
    assert "Binned likelihood generation summary" in output
    assert "Succeeded: 1" in output
    assert "Failed:    1" in output
    assert "2 bins: RuntimeError: boom" in output


def test_main_continues_after_failed_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[int] = []

    def fake_generate_single_model(
        n_bins: int, output_dir: Path, validate: bool
    ) -> dict[str, Any]:
        calls.append(n_bins)
        if n_bins == 1:
            raise RuntimeError("first failed")
        return {
            "n_bins": n_bins,
            "common_path": output_dir / f"common_{n_bins}bins.json",
            "pyhf_path": output_dir / f"pyhf_{n_bins}bins.json",
            "pyhs3_path": output_dir / f"pyhs3_{n_bins}bins.json",
            "status": "success",
        }

    monkeypatch.setattr(generator, "generate_single_model", fake_generate_single_model)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_binned_likelihood_models.py",
            "--output-dir",
            str(tmp_path),
            "--bin-counts",
            "1",
            "2",
        ],
    )

    with pytest.raises(SystemExit):
        generator.main()

    assert calls == [1, 2]


def test_module_main_guard_runs_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import runpy

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_binned_likelihood_models.py",
            "--output-dir",
            str(tmp_path),
            "--bin-counts",
            "1",
        ],
    )

    runpy.run_module("src.generate_binned_likelihood_models", run_name="__main__")

    assert (tmp_path / "common_1bins.json").exists()
    assert (tmp_path / "pyhf_1bins.json").exists()
    assert (tmp_path / "pyhs3_1bins.json").exists()
