from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from pyhs3.analyses import Analyses, Analysis
from pyhs3.data import Data, PointData
from pyhs3.distributions import (
    Distributions,
    ExponentialDist,
    GaussianDist,
    PoissonDist,
)
from pyhs3.domains import DomainCoordinateAxis, Domains, ProductDomain
from pyhs3.likelihoods import Likelihood, Likelihoods
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoint, ParameterPoints
from pyhs3.workspace import Workspace


DEFAULT_OUTPUT_DIR = Path("inputs/scalar_pdf_workspaces")

SCENARIO_X_VALUES = {
    "normal": 0.0,
    "poisson": 5.0,
    "exponential": 1.0,
}

AVAILABLE_SCENARIOS = tuple(SCENARIO_X_VALUES)


def validate_scenario(scenario: str) -> str:
    """
    Validate a scalar PDF scenario name.
    """

    if scenario not in SCENARIO_X_VALUES:
        raise ValueError(
            f"Unknown scenario: {scenario}. "
            f"Available scenarios: {list(AVAILABLE_SCENARIOS)}"
        )

    return scenario


def validate_scenarios(scenarios: list[str]) -> list[str]:
    """
    Validate all requested scalar PDF scenarios.
    """

    if not scenarios:
        raise ValueError("--scenarios must contain at least one value")

    return [validate_scenario(scenario) for scenario in scenarios]


def make_workspace(
    scenario: str,
) -> Workspace:
    """
    Generate a minimal HS3 workspace for a scalar PDF benchmark.
    """

    scenario = validate_scenario(scenario)

    metadata = Metadata(
        hs3_version="0.2",
        authors=["pyHS3 benchmark"],
        description=f"Minimal scalar PDF workspace for {scenario}.",
    )

    x_value = SCENARIO_X_VALUES[scenario]

    domain = ProductDomain(
        name="domain_x",
        axes=[
            DomainCoordinateAxis(
                name="x",
                min=-10.0 if scenario == "normal" else 0.0,
                max=30.0 if scenario == "poisson" else 10.0,
            )
        ],
    )

    data = PointData(
        name="data_x",
        value=x_value,
    )

    if scenario == "normal":
        distribution = GaussianDist(
            name="pdf",
            mean="mu",
            sigma="sigma",
            x="x",
        )
        parameter_points = [
            ParameterPoint(name="x", value=x_value, const=False),
            ParameterPoint(name="mu", value=0.0, const=True),
            ParameterPoint(name="sigma", value=1.0, const=True),
        ]

    elif scenario == "poisson":
        distribution = PoissonDist(
            name="pdf",
            mean="mean",
            x="x",
        )
        parameter_points = [
            ParameterPoint(name="x", value=x_value, const=False),
            ParameterPoint(name="mean", value=5.0, const=True),
        ]

    elif scenario == "exponential":
        distribution = ExponentialDist(
            name="pdf",
            x="x",
            c="c",
        )
        parameter_points = [
            ParameterPoint(name="x", value=x_value, const=False),
            ParameterPoint(name="c", value=-1.0, const=True),
        ]

    likelihood = Likelihood(
        name="likelihood",
        distributions=["pdf"],
        data=["data_x"],
    )

    analysis = Analysis(
        name="analysis",
        likelihood="likelihood",
        parameters_of_interest=["x"],
        domains=["domain_x"],
        init="init",
    )

    return Workspace(
        metadata=metadata,
        distributions=Distributions([distribution]),
        domains=Domains([domain]),
        data=Data([data]),
        likelihoods=Likelihoods([likelihood]),
        analyses=Analyses([analysis]),
        parameter_points=ParameterPoints(
            [
                {
                    "name": "init",
                    "parameters": parameter_points,
                }
            ]
        ),
    )


def save_workspace(
    workspace: Workspace,
    output_path: Path,
) -> None:
    """
    Save a pyHS3 workspace as a JSON file.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as output_file:
        json.dump(
            workspace.model_dump(mode="json", exclude_none=True),
            output_file,
            indent=2,
            sort_keys=True,
        )


def verify_output_file(output_path: Path) -> None:
    """
    Verify that the expected workspace file was created.
    """

    if not output_path.exists():
        raise FileNotFoundError(f"Workspace file was not created: {output_path}")

    if not output_path.is_file():
        raise FileNotFoundError(f"Workspace output path is not a file: {output_path}")


def extract_scalar_output(result: Any) -> float:
    """
    Extract a finite scalar value from model.pdf(...) output for validation.
    """

    array = np.asarray(result)

    if array.size == 0:
        raise ValueError("Validation PDF output is empty")

    value = float(array.reshape(-1)[0])

    if not math.isfinite(value):
        raise ValueError(f"Validation PDF output is not finite: {value}")

    return value


def validate_workspace(
    output_path: Path,
    scenario: str,
) -> None:
    """
    Validate a pyHS3 workspace for a scalar PDF benchmark.
    """

    scenario = validate_scenario(scenario)
    workspace = Workspace.load(output_path)
    model = workspace.model("analysis", progress=False, mode="FAST_RUN")

    if "pdf" not in model.distributions:
        raise ValueError(
            f"Generated {scenario} workspace does not expose distribution 'pdf'"
        )

    parameters = {
        name: np.asarray(value, dtype=float)
        for name, value in {
            **model.data,
            **model.free_params,
        }.items()
    }

    result = model.pdf("pdf", **parameters)
    output = extract_scalar_output(result)

    print(f"{scenario}: validation output = {output}")


def generate_single_workspace(
    scenario: str,
    output_dir: Path,
    validate: bool,
) -> dict[str, Any]:
    """
    Generate and optionally validate one scalar PDF workspace.
    """

    scenario = validate_scenario(scenario)
    workspace = make_workspace(scenario)
    output_path = output_dir / f"{scenario}_pdf_workspace.json"

    save_workspace(
        workspace=workspace,
        output_path=output_path,
    )
    verify_output_file(output_path)

    if validate:
        validate_workspace(
            output_path=output_path,
            scenario=scenario,
        )

    return {
        "scenario": scenario,
        "output_path": output_path,
        "status": "success",
    }


def print_success(result: dict[str, Any]) -> None:
    print(f"Saved {result['scenario']} workspace to {result['output_path']}")


def print_failure(scenario: str, exc: Exception) -> None:
    print()
    print("=" * 72)
    print("Scalar PDF workspace generation FAILED")
    print("=" * 72)
    print(f"Scenario: {scenario}")
    print(f"Error:    {type(exc).__name__}: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate minimal HS3 workspaces for scalar PDF benchmarks."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(AVAILABLE_SCENARIOS),
        choices=list(AVAILABLE_SCENARIOS),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = validate_scenarios(args.scenarios)

    failures = []

    for scenario in scenarios:
        try:
            result = generate_single_workspace(
                scenario=scenario,
                output_dir=args.output_dir,
                validate=args.validate,
            )
        except Exception as exc:
            failures.append(
                {
                    "scenario": scenario,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            print_failure(scenario=scenario, exc=exc)
            continue

        print_success(result)

    if failures:
        print()
        print("=" * 72)
        print("Scalar PDF workspace generation summary")
        print("=" * 72)
        print(f"Succeeded: {len(scenarios) - len(failures)}")
        print(f"Failed:    {len(failures)}")
        for failure in failures:
            print(
                f"  - {failure['scenario']}: "
                f"{failure['error_type']}: {failure['error_message']}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
