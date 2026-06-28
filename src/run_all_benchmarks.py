from __future__ import annotations

import argparse
import json
import shlex
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
    """Container for a benchmark name and execution command."""

    name: str
    command: list[str]


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Structured result for one benchmark subprocess."""

    name: str
    command: list[str]
    status: str
    returncode: int | None
    duration_seconds: float
    error_type: str | None = None
    error_message: str | None = None


def module_name(script_name: str) -> str:
    """Return the importable module name for a benchmark script."""

    if not script_name.endswith(".py"):
        raise ValueError(f"Expected a Python script name, got: {script_name}")

    return f"src.{script_name.removesuffix('.py')}"


def module_command(script_name: str) -> list[str]:
    """Build a command that runs a benchmark as a package module.

    The benchmark files use relative imports such as ``from .config import ...``,
    so they must be executed with ``python -m src.<module>`` from the repository
    root rather than as direct script paths.
    """

    return [sys.executable, "-m", module_name(script_name)]


def validate_positive_int(value: int, name: str, minimum: int = 1) -> None:
    """Validate that an integer CLI/configuration value is at least a minimum."""

    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def validate_target_and_mode(target: str, mode: str) -> None:
    """Validate common target and mode options."""

    if not isinstance(target, str) or not target.strip():
        raise ValueError("--target must be a non-empty string")

    if not isinstance(mode, str) or not mode.strip():
        raise ValueError("--mode must be a non-empty string")


def validate_benchmark_config(n_runs: int, n_evaluations: int, n_scan_points: int, target: str, mode: str) -> None:
    """Validate top-level benchmark-suite configuration."""

    validate_positive_int(n_runs, "--n-runs")
    validate_positive_int(n_evaluations, "--n-evaluations")
    validate_positive_int(n_scan_points, "--n-scan-points", minimum=2)
    validate_target_and_mode(target, mode)


def command_has_plot_flag(command: list[str]) -> bool:
    """Return whether a generated command includes --plot."""

    return "--plot" in command

def format_command(command: list[str]) -> str:
    """Return a shell-readable command string for logging."""

    return " ".join(shlex.quote(part) for part in command)


def run_command(
    benchmark: BenchmarkCommand,
    dry_run: bool,
) -> BenchmarkRunResult:
    """Execute one benchmark command and return a structured status."""

    print()
    print("=" * 80)
    print(f"Running benchmark: {benchmark.name}")
    print("=" * 80)
    print(format_command(benchmark.command))

    if dry_run:
        return BenchmarkRunResult(
            name=benchmark.name,
            command=benchmark.command,
            status="skipped_dry_run",
            returncode=0,
            duration_seconds=0.0,
        )

    start = time.perf_counter()

    try:
        completed = subprocess.run(
            benchmark.command,
            cwd=REPO_ROOT,
            check=False,
        )
    except OSError as exc:
        end = time.perf_counter()
        print(
            f"FAILED {benchmark.name}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return BenchmarkRunResult(
            name=benchmark.name,
            command=benchmark.command,
            status="failed",
            returncode=None,
            duration_seconds=end - start,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    end = time.perf_counter()
    duration_seconds = end - start

    if completed.returncode == 0:
        print(f"Finished {benchmark.name} in {duration_seconds:.2f} s")
        return BenchmarkRunResult(
            name=benchmark.name,
            command=benchmark.command,
            status="success",
            returncode=completed.returncode,
            duration_seconds=duration_seconds,
        )

    print(f"FAILED {benchmark.name} with exit code {completed.returncode}")
    return BenchmarkRunResult(
        name=benchmark.name,
        command=benchmark.command,
        status="failed",
        returncode=completed.returncode,
        duration_seconds=duration_seconds,
        error_type="CalledProcessError",
        error_message=f"Benchmark exited with code {completed.returncode}",
    )


def save_suite_summary(
    run_results: list[BenchmarkRunResult],
    output_path: Path,
    total_time_seconds: float,
) -> None:
    """Save a machine-readable summary for the full benchmark suite."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "benchmark": "benchmark_suite",
        "total_time_seconds": total_time_seconds,
        "n_results": len(run_results),
        "n_success": sum(result.status == "success" for result in run_results),
        "n_failed": sum(result.status == "failed" for result in run_results),
        "n_skipped_dry_run": sum(
            result.status == "skipped_dry_run"
            for result in run_results
        ),
        "results": [
            {
                "name": result.name,
                "command": result.command,
                "command_string": format_command(result.command),
                "status": result.status,
                "returncode": result.returncode,
                "duration_seconds": result.duration_seconds,
                "error_type": result.error_type,
                "error_message": result.error_message,
            }
            for result in run_results
        ],
    }

    with output_path.open("w") as file:
        json.dump(payload, file, indent=2, sort_keys=True)



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
                *module_command("run_workspace_loading.py"),
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
                *module_command("run_model_creation.py"),
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
                *module_command("run_log_prob_construction.py"),
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
                *module_command("run_log_prob_compilation.py"),
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
                *module_command("run_compiled_evaluation.py"),
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
                *module_command("run_pdf_evaluation.py"),
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
                *module_command("run_pdf_evaluation.py"),
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
            *module_command("run_nll_scan.py"),
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
                *module_command("run_memory_scaling.py"),
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
                *module_command("run_model_complexity_scaling.py"),
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
                *module_command("run_graph_canonicalization.py"),
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
                *module_command("run_graph_optimization.py"),
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
        "--summary-output",
        type=Path,
        default=Path("results/benchmark_suite/benchmark_suite_summary.json"),
        help="Path where a JSON summary of the benchmark suite will be saved.",
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

    validate_benchmark_config(
        n_runs=args.n_runs,
        n_evaluations=args.n_evaluations,
        n_scan_points=args.n_scan_points,
        target=args.target,
        mode=args.mode,
    )

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

    run_results: list[BenchmarkRunResult] = []
    failures: list[str] = []

    start = time.perf_counter()

    try:
        for benchmark in selected_benchmarks:
            result = run_command(
                benchmark=benchmark,
                dry_run=args.dry_run,
            )
            run_results.append(result)

            if result.status == "failed":
                failures.append(benchmark.name)

                if not args.continue_on_failure:
                    break
    except KeyboardInterrupt:
        end = time.perf_counter()
        total_time_seconds = end - start
        save_suite_summary(
            run_results=run_results,
            output_path=args.summary_output,
            total_time_seconds=total_time_seconds,
        )
        print()
        print("Benchmark suite interrupted by user.")
        print(f"Saved partial summary to {args.summary_output}")
        raise SystemExit(130)

    end = time.perf_counter()
    total_time_seconds = end - start

    save_suite_summary(
        run_results=run_results,
        output_path=args.summary_output,
        total_time_seconds=total_time_seconds,
    )

    n_success = sum(result.status == "success" for result in run_results)
    n_dry_run = sum(result.status == "skipped_dry_run" for result in run_results)

    print()
    print("=" * 80)
    print("Benchmark suite summary")
    print("=" * 80)
    print(f"Total time: {total_time_seconds:.2f} s")
    print(f"Selected:   {len(selected_benchmarks)}")
    print(f"Executed:   {len(run_results)}")
    print(f"Succeeded:  {n_success}")
    print(f"Dry-run:    {n_dry_run}")
    print(f"Failed:     {len(failures)}")
    print(f"Summary:    {args.summary_output}")

    if failures:
        print()
        print("Failed benchmarks:")
        for failure in failures:
            print(f"  - {failure}")

        raise SystemExit(1)

    print()
    if args.dry_run:
        print("Dry run completed successfully.")
    else:
        print("All selected benchmarks completed successfully.")


if __name__ == "__main__":
    main()
