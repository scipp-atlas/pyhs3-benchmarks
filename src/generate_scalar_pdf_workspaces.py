from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

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


def make_workspace(
    scenario: str,
) -> Workspace:
    """
    Generate a minimal HS3 workspace for a scalar PDF benchmark.
    """

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

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

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


def validate_workspace(
    output_path: Path,
    scenario: str,
) -> None:
    """
    Validate a pyHS3 workspace for a scalar PDF benchmark.
    """

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

    print(f"{scenario}: validation output = {result}")


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
        default=["normal", "poisson", "exponential"],
        choices=["normal", "poisson", "exponential"],
    )
    parser.add_argument(
        "--validate",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for scenario in args.scenarios:
        workspace = make_workspace(scenario)
        output_path = args.output_dir / f"{scenario}_pdf_workspace.json"

        save_workspace(
            workspace=workspace,
            output_path=output_path,
        )

        print(f"Saved {scenario} workspace to {output_path}")

        if args.validate:
            validate_workspace(
                output_path=output_path,
                scenario=scenario,
            )


if __name__ == "__main__":
    main()
