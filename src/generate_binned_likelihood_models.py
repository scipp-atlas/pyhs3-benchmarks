from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyhs3.analyses import Analyses, Analysis
from pyhs3.data import Data, PointData
from pyhs3.distributions import Distributions, PoissonDist
from pyhs3.domains import DomainCoordinateAxis, Domains, ProductDomain
from pyhs3.functions import Functions, ProductFunction, SumFunction
from pyhs3.likelihoods import Likelihood, Likelihoods
from pyhs3.metadata import Metadata
from pyhs3.parameter_points import ParameterPoint, ParameterPoints
from pyhs3.workspace import Workspace


DEFAULT_OUTPUT_DIR = Path("inputs/binned_likelihood_models")
DEFAULT_BIN_COUNTS = [3, 30, 300]


def make_arrays(n_bins: int) -> dict[str, list[float]]:
    """
    Generate deterministic arrays for a binned likelihood model.
    """

    rng = np.random.default_rng(12345 + n_bins)

    signal = rng.uniform(2.0, 10.0, size=n_bins)
    background = rng.uniform(15.0, 35.0, size=n_bins)
    observation = np.rint(signal + background).astype(float)

    return {
        "signal": signal.tolist(),
        "background": background.tolist(),
        "observation": observation.tolist(),
    }


def make_pyhf_spec(model: dict[str, list[float]]) -> dict:
    """
    Generate a pyhf spec for a binned likelihood model.
    """

    return {
        "channels": [
            {
                "name": "singlechannel",
                "samples": [
                    {
                        "name": "signal",
                        "data": model["signal"],
                        "modifiers": [
                            {
                                "name": "mu",
                                "type": "normfactor",
                                "data": None,
                            }
                        ],
                    },
                    {
                        "name": "background",
                        "data": model["background"],
                        "modifiers": [],
                    },
                ],
            }
        ],
        "observations": [
            {
                "name": "singlechannel",
                "data": model["observation"],
            }
        ],
        "measurements": [
            {
                "name": "measurement",
                "config": {
                    "poi": "mu",
                    "parameters": [],
                },
            }
        ],
        "version": "1.0.0",
    }


def make_pyhs3_workspace(
    model: dict[str, list[float]],
    n_bins: int,
) -> Workspace:
    """
    Generate a pyHS3 workspace for a binned likelihood model.
    """

    metadata = Metadata(
        hs3_version="0.2",
        authors=["pyHS3 benchmark"],
        description=f"Minimal binned likelihood model with {n_bins} bins.",
    )

    distributions = []
    functions = []
    parameter_points = [
        ParameterPoint(name="mu", value=1.0, const=False),
    ]

    data = []
    domains = []

    for index in range(n_bins):
        sig_name = f"signal_{index}"
        bkg_name = f"background_{index}"
        obs_name = f"obs_{index}"
        scaled_signal_name = f"scaled_signal_{index}"
        expected_name = f"expected_{index}"
        poisson_name = f"poisson_{index}"

        parameter_points.extend(
            [
                ParameterPoint(
                    name=sig_name,
                    value=float(model["signal"][index]),
                    const=True,
                ),
                ParameterPoint(
                    name=bkg_name,
                    value=float(model["background"][index]),
                    const=True,
                ),
                ParameterPoint(
                    name=obs_name,
                    value=float(model["observation"][index]),
                    const=True,
                ),
            ]
        )

        functions.append(
            ProductFunction(
                name=scaled_signal_name,
                factors=["mu", sig_name],
            )
        )
        functions.append(
            SumFunction(
                name=expected_name,
                summands=[scaled_signal_name, bkg_name],
            )
        )

        distributions.append(
            PoissonDist(
                name=poisson_name,
                mean=expected_name,
                x=obs_name,
            )
        )

        data.append(
            PointData(
                name=f"data_{index}",
                value=float(model["observation"][index]),
            )
        )

        domains.append(
            ProductDomain(
                name=f"domain_{index}",
                axes=[
                    DomainCoordinateAxis(
                        name=obs_name,
                        min=0.0,
                        max=max(100.0, float(model["observation"][index]) * 2.0),
                    )
                ],
            )
        )

    likelihood = Likelihood(
        name="likelihood",
        distributions=[f"poisson_{index}" for index in range(n_bins)],
        data=[f"data_{index}" for index in range(n_bins)],
    )

    analysis = Analysis(
        name="analysis",
        likelihood="likelihood",
        parameters_of_interest=["mu"],
        domains=[f"domain_{index}" for index in range(n_bins)],
        init="init",
    )

    return Workspace(
        metadata=metadata,
        distributions=Distributions(distributions),
        functions=Functions(functions),
        domains=Domains(domains),
        data=Data(data),
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


def save_json(data: dict, path: Path) -> None:
    """
    Save a dictionary to a JSON file.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as output_file:
        json.dump(
            data,
            output_file,
            indent=2,
            sort_keys=True,
        )


def save_workspace(workspace: Workspace, path: Path) -> None:
    """
    Save a pyHS3 workspace to a JSON file.
    """

    save_json(
        workspace.model_dump(mode="json", exclude_none=True),
        path,
    )


def validate_workspace(
    workspace_path: Path,
    n_bins: int,
) -> None:
    """
    Validate a pyHS3 workspace by loading it and checking the number of distributions.
    """

    workspace = Workspace.load(workspace_path)

    model = workspace.model(
        "analysis",
        progress=False,
        mode="FAST_RUN",
    )

    if len(model.distributions) != n_bins:
        raise ValueError(
            f"Expected {n_bins} distributions, "
            f"got {len(model.distributions)}"
        )

    print(
        f"{n_bins} bins: validation successful "
        f"({len(model.distributions)} distributions)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate binned likelihood models for cross-framework benchmarks."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--bin-counts",
        nargs="+",
        type=int,
        default=DEFAULT_BIN_COUNTS,
    )
    parser.add_argument(
        "--validate",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for n_bins in args.bin_counts:
        model = make_arrays(n_bins=n_bins)

        common_path = args.output_dir / f"common_{n_bins}bins.json"
        pyhf_path = args.output_dir / f"pyhf_{n_bins}bins.json"
        pyhs3_path = args.output_dir / f"pyhs3_{n_bins}bins.json"

        save_json(
            {
                "n_bins": n_bins,
                **model,
            },
            common_path,
        )

        save_json(
            make_pyhf_spec(model),
            pyhf_path,
        )

        save_workspace(
            make_pyhs3_workspace(
                model=model,
                n_bins=n_bins,
            ),
            pyhs3_path,
        )

        if args.validate:
            validate_workspace(
                workspace_path=pyhs3_path,
                n_bins=n_bins,
            )

        print(f"Saved common model to {common_path}")
        print(f"Saved pyhf model to {pyhf_path}")
        print(f"Saved pyhs3 model to {pyhs3_path}")


if __name__ == "__main__":
    main()
