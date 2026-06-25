from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent

DEFAULT_SIMPLE_WORKSPACES = [
    "inputs/simple_workspace.json",
    "inputs/simple_workspace_nonp.json",
    "inputs/simple_workspace_generic.json",
    "inputs/simple_workspace_generic_nonp.json",
]

DEFAULT_SCALAR_WORKSPACES = [
    "inputs/scalar_pdf_workspaces/normal_pdf_workspace.json",
    "inputs/scalar_pdf_workspaces/poisson_pdf_workspace.json",
    "inputs/scalar_pdf_workspaces/exponential_pdf_workspace.json",
]

PRESETS = {
    "smoke": {
        "n_runs": 1,
        "n_evaluations": 1,
        "n_scan_points": 11,
        "plot": False,
    },
    "default": {
        "n_runs": 20,
        "n_evaluations": 1000,
        "n_scan_points": 1001,
        "plot": True,
    },
    "full": {
        "n_runs": 200,
        "n_evaluations": 10000,
        "n_scan_points": 5001,
        "plot": True,
    },
}


@dataclass(frozen=True)
class BenchmarkCommand:
    """
    Container for a benchmark name and execution command.
    """

    name: str
    command: list[str]


def script_path(script_name: str) -> str:
    """
    Return the absolute path to a benchmark script.
    """

    return str(SRC_DIR / script_name)


def run_command(
    benchmark: BenchmarkCommand,
    dry_run: bool,
) -> bool:
    """
    Execute a benchmark command and report its status.
    """

    print()
    print("=" * 80)
    print(f"Running benchmark: {benchmark.name}")
    print("=" * 80)
    print(" ".join(benchmark.command))

    if dry_run:
        return True

    start = time.perf_counter()

    completed = subprocess.run(
        benchmark.command,
        cwd=REPO_ROOT,
        check=False,
    )

    end = time.perf_counter()

    if completed.returncode == 0:
        print(f"Finished {benchmark.name} in {end - start:.2f} s")
        return True

    print(f"FAILED {benchmark.name} with exit code {completed.returncode}")
    return False


def build_core_benchmarks(
    n_runs: int,
    n_evaluations: int,
    workspaces: list[str],
    target: str,
    mode: str,
    plot: bool,
) -> list[BenchmarkCommand]:
    """
    Create the core benchmark commands.
    """

    plot_flag = ["--plot"] if plot else []

    return [
        BenchmarkCommand(
            name="workspace_loading",
            command=[
                sys.executable,
                script_path("run_workspace_loading.py"),
                "--workspaces",
                *workspaces,
                "--n-runs",
                str(n_runs),
                *plot_flag,
            ],
        ),
        BenchmarkCommand(
            name="model_creation",
            command=[
                sys.executable,
                script_path("run_model_creation.py"),
                "--workspaces",
                *workspaces,
                "--targets",
                target,
                "--modes",
                mode,
                "--n-runs",
                str(n_runs),
                *plot_flag,
            ],
        ),
        BenchmarkCommand(
            name="log_prob_construction",
            command=[
                sys.executable,
                script_path("run_log_prob_construction.py"),
                "--workspaces",
                *workspaces,
                "--targets",
                target,
                "--modes",
                mode,
                "--n-runs",
                str(n_runs),
                *plot_flag,
            ],
        ),
        BenchmarkCommand(
            name="log_prob_compilation",
            command=[
                sys.executable,
                script_path("run_log_prob_compilation.py"),
                "--workspaces",
                *workspaces,
                "--targets",
                target,
                "--modes",
                mode,
                "--n-runs",
                str(n_runs),
                *plot_flag,
            ],
        ),
        BenchmarkCommand(
            name="compiled_evaluation",
            command=[
                sys.executable,
                script_path("run_compiled_evaluation.py"),
                "--workspaces",
                *workspaces,
                "--targets",
                target,
                "--modes",
                mode,
                "--n-evaluations",
                str(n_evaluations),
                *plot_flag,
            ],
        ),
    ]


def build_pdf_benchmarks(
    n_evaluations: int,
    plot: bool,
) -> list[BenchmarkCommand]:
    """
    Create benchmarks for PDF evaluation.
    """

    plot_flag = ["--plot"] if plot else []

    return [
        BenchmarkCommand(
            name="pdf_evaluation_simple",
            command=[
                sys.executable,
                script_path("run_pdf_evaluation.py"),
                "--workspaces",
                *DEFAULT_SIMPLE_WORKSPACES,
                "--targets",
                "L_ch0",
                "--distributions",
                "sig_ch0",
                "--n-evaluations",
                str(n_evaluations),
                *plot_flag,
                "--plot-dir",
                "plots/pdf_evaluation_simple",
            ],
        ),
        BenchmarkCommand(
            name="pdf_evaluation_scalar",
            command=[
                sys.executable,
                script_path("run_pdf_evaluation.py"),
                "--workspaces",
                *DEFAULT_SCALAR_WORKSPACES,
                "--targets",
                "analysis",
                "--distributions",
                "pdf",
                "--n-evaluations",
                str(n_evaluations),
                *plot_flag,
                "--plot-dir",
                "plots/pdf_evaluation_scalar",
            ],
        ),
    ]


def build_nll_scan_benchmark(
    n_scan_points: int,
    plot: bool,
) -> BenchmarkCommand:
    """
    Create a benchmark for NLL scanning.
    """

    plot_flag = ["--plot"] if plot else []

    return BenchmarkCommand(
        name="nll_scan",
        command=[
            sys.executable,
            script_path("run_nll_scan.py"),
            "--workspaces",
            *DEFAULT_SIMPLE_WORKSPACES,
            "--targets",
            "L_ch0",
            "--scan-parameter",
            "mu_sig",
            "--n-scan-points",
            str(n_scan_points),
            *plot_flag,
        ],
    )


def build_scaling_benchmarks(
    n_runs: int,
    n_evaluations: int,
    n_scan_points: int,
    plot: bool,
) -> list[BenchmarkCommand]:
    """
    Create scaling-related benchmark commands.
    """

    plot_flag = ["--plot"] if plot else []

    return [
        BenchmarkCommand(
            name="memory_scaling",
            command=[
                sys.executable,
                script_path("run_memory_scaling.py"),
                "--workspaces",
                *DEFAULT_SIMPLE_WORKSPACES,
                "--targets",
                "L_ch0",
                "--n-runs",
                str(n_runs),
                "--n-evaluations",
                str(n_evaluations),
                "--n-scan-points",
                str(n_scan_points),
                *plot_flag,
                "--plot-dir",
                "plots/memory_scaling_all_stages",
            ],
        ),
        BenchmarkCommand(
            name="model_complexity_scaling",
            command=[
                sys.executable,
                script_path("run_model_complexity_scaling.py"),
                "--workspaces",
                *DEFAULT_SIMPLE_WORKSPACES,
                "--targets",
                "L_ch0",
                "--stages",
                "all",
                "--distribution",
                "sig_ch0",
                "--scan-parameter",
                "mu_sig",
                "--n-runs",
                str(n_runs),
                "--n-evaluations",
                str(n_evaluations),
                "--n-scan-points",
                str(n_scan_points),
                *plot_flag,
                "--plot-dir",
                "plots/model_complexity_all_stages",
            ],
        ),
    ]


def build_graph_benchmarks(
    n_runs: int,
    plot: bool,
) -> list[BenchmarkCommand]:
    """
    Create graph processing benchmark commands.
    """

    plot_flag = ["--plot"] if plot else []

    return [
        BenchmarkCommand(
            name="graph_canonicalization",
            command=[
                sys.executable,
                script_path("run_graph_canonicalization.py"),
                "--workspaces",
                *DEFAULT_SIMPLE_WORKSPACES,
                "--targets",
                "L_ch0",
                "--n-runs",
                str(n_runs),
                *plot_flag,
                "--plot-dir",
                "plots/graph_canonicalization_simple",
            ],
        ),
        BenchmarkCommand(
            name="graph_optimization",
            command=[
                sys.executable,
                script_path("run_graph_optimization.py"),
                "--workspaces",
                *DEFAULT_SIMPLE_WORKSPACES,
                "--targets",
                "L_ch0",
                "--n-runs",
                str(n_runs),
                *plot_flag,
                "--plot-dir",
                "plots/graph_optimization_simple",
            ],
        ),
    ]


def select_benchmarks(
    benchmark_names: list[str],
    all_benchmarks: list[BenchmarkCommand],
) -> list[BenchmarkCommand]:
    """
    Select benchmarks requested by the user.
    """

    if benchmark_names == ["all"]:
        return all_benchmarks

    available = {benchmark.name: benchmark for benchmark in all_benchmarks}

    unknown = [
        name
        for name in benchmark_names
        if name not in available
    ]

    if unknown:
        raise ValueError(
            f"Unknown benchmark(s): {unknown}. "
            f"Available benchmarks: {sorted(available)}"
        )

    return [
        available[name]
        for name in benchmark_names
    ]

def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    """
    Apply a benchmark preset.

    Explicit command-line overrides take priority over preset values.
    """

    if args.preset is None:
        return args

    preset = PRESETS[args.preset]

    if args.n_runs is None:
        args.n_runs = preset["n_runs"]

    if args.n_evaluations is None:
        args.n_evaluations = preset["n_evaluations"]

    if args.n_scan_points is None:
        args.n_scan_points = preset["n_scan_points"]

    if not args.no_plot_was_set:
        args.no_plot = not preset["plot"]

    return args

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PyHS3 benchmark suite."
    )

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        help="Benchmarks to run, or 'all'.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n-evaluations",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n-scan-points",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--target",
        default="L_ch0",
    )
    parser.add_argument(
        "--mode",
        default="FAST_RUN",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
    )

    args = parser.parse_args()
    args.no_plot_was_set = "--no-plot" in sys.argv
    return args


def main() -> None:
    args = parse_args()

    args = apply_preset(args)

    if args.n_runs is None:
        args.n_runs = 5

    if args.n_evaluations is None:
        args.n_evaluations = 1000

    if args.n_scan_points is None:
        args.n_scan_points = 1001

    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1")

    if args.n_evaluations < 1:
        raise ValueError("--n-evaluations must be at least 1")

    if args.n_scan_points < 2:
        raise ValueError("--n-scan-points must be at least 2")

    plot = not args.no_plot

    all_benchmarks = [
        *build_core_benchmarks(
            n_runs=args.n_runs,
            n_evaluations=args.n_evaluations,
            workspaces=DEFAULT_SIMPLE_WORKSPACES,
            target=args.target,
            mode=args.mode,
            plot=plot,
        ),
        *build_pdf_benchmarks(
            n_evaluations=args.n_evaluations,
            plot=plot,
        ),
        build_nll_scan_benchmark(
            n_scan_points=args.n_scan_points,
            plot=plot,
        ),
        *build_scaling_benchmarks(
            n_runs=args.n_runs,
            n_evaluations=args.n_evaluations,
            n_scan_points=args.n_scan_points,
            plot=plot,
        ),
        *build_graph_benchmarks(
            n_runs=args.n_runs,
            plot=plot,
        ),
    ]

    selected_benchmarks = select_benchmarks(
        benchmark_names=args.benchmarks,
        all_benchmarks=all_benchmarks,
    )

    print("Selected benchmarks:")
    for benchmark in selected_benchmarks:
        print(f"  - {benchmark.name}")

    failures = []

    start = time.perf_counter()

    for benchmark in selected_benchmarks:
        success = run_command(
            benchmark=benchmark,
            dry_run=args.dry_run,
        )

        if not success:
            failures.append(benchmark.name)

            if not args.continue_on_failure:
                break

    end = time.perf_counter()

    print()
    print("=" * 80)
    print("Benchmark suite summary")
    print("=" * 80)
    print(f"Total time: {end - start:.2f} s")
    print(f"Succeeded:  {len(selected_benchmarks) - len(failures)}")
    print(f"Failed:     {len(failures)}")

    if failures:
        print()
        print("Failed benchmarks:")
        for failure in failures:
            print(f"  - {failure}")

        raise SystemExit(1)

    print()
    print("All selected benchmarks completed successfully.")


if __name__ == "__main__":
    main()
