from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    group: str
    kind: str
    module: str
    uses_workspace_matrix: bool
    requires_root_pair: bool = False
    run_once: bool = False


@dataclass
class RunRecord:
    benchmark: str
    group: str
    workspace: str | None
    root_workspace: str | None
    command: list[str]
    status: str
    returncode: int | None
    duration_seconds: float
    stdout_path: str
    stderr_path: str
    error: str | None = None


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "workspace_loading": BenchmarkSpec(
        name="workspace_loading",
        group="pyhs3",
        kind="multi_workspace",
        module="src.run_workspace_loading",
        uses_workspace_matrix=True,
    ),
    "model_creation": BenchmarkSpec(
        name="model_creation",
        group="pyhs3",
        kind="multi_workspace",
        module="src.run_model_creation",
        uses_workspace_matrix=True,
    ),
    "log_prob_construction": BenchmarkSpec(
        name="log_prob_construction",
        group="pyhs3",
        kind="multi_workspace",
        module="src.run_log_prob_construction",
        uses_workspace_matrix=True,
    ),
    "log_prob_compilation": BenchmarkSpec(
        name="log_prob_compilation",
        group="pyhs3",
        kind="multi_workspace",
        module="src.run_log_prob_compilation",
        uses_workspace_matrix=True,
    ),
    "graph_canonicalization": BenchmarkSpec(
        name="graph_canonicalization",
        group="pyhs3",
        kind="multi_workspace",
        module="src.run_graph_canonicalization",
        uses_workspace_matrix=True,
    ),
    "graph_optimization": BenchmarkSpec(
        name="graph_optimization",
        group="pyhs3",
        kind="multi_workspace",
        module="src.run_graph_optimization",
        uses_workspace_matrix=True,
    ),
    "compiled_evaluation": BenchmarkSpec(
        name="compiled_evaluation",
        group="pyhs3",
        kind="multi_workspace",
        module="src.run_compiled_evaluation",
        uses_workspace_matrix=True,
    ),
    "pdf_evaluation": BenchmarkSpec(
        name="pdf_evaluation",
        group="pyhs3",
        kind="multi_workspace",
        module="src.run_pdf_evaluation",
        uses_workspace_matrix=True,
    ),
    "nll_scan": BenchmarkSpec(
        name="nll_scan",
        group="pyhs3",
        kind="multi_workspace",
        module="src.run_nll_scan",
        uses_workspace_matrix=True,
    ),
    "memory_scaling": BenchmarkSpec(
        name="memory_scaling",
        group="pyhs3",
        kind="multi_workspace",
        module="src.run_memory_scaling",
        uses_workspace_matrix=True,
    ),
    "model_complexity_scaling": BenchmarkSpec(
        name="model_complexity_scaling",
        group="pyhs3",
        kind="multi_workspace",
        module="src.run_model_complexity_scaling",
        uses_workspace_matrix=True,
    ),
    "cross_nll_scan": BenchmarkSpec(
        name="cross_nll_scan",
        group="cross",
        kind="multi_workspace",
        module="src.run_cross_nll_scan",
        uses_workspace_matrix=True,
        requires_root_pair=True,
    ),
    "pyhs3_xroofit_benchmark": BenchmarkSpec(
        name="pyhs3_xroofit_benchmark",
        group="cross",
        kind="json_root_pair",
        module="src.run_pyhs3_xroofit_benchmark",
        uses_workspace_matrix=True,
        requires_root_pair=True,
    ),
    "cross_binned_likelihood_evaluation": BenchmarkSpec(
        name="cross_binned_likelihood_evaluation",
        group="cross",
        kind="single_workspace",
        module="src.run_cross_binned_likelihood_evaluation",
        uses_workspace_matrix=True,
    ),
    "cross_model_complexity_scaling": BenchmarkSpec(
        name="cross_model_complexity_scaling",
        group="cross",
        kind="run_once",
        module="src.run_cross_model_complexity_scaling",
        uses_workspace_matrix=False,
        run_once=True,
    ),
    "cross_vectorized_pdf_evaluation": BenchmarkSpec(
        name="cross_vectorized_pdf_evaluation",
        group="cross",
        kind="run_once",
        module="src.run_cross_vectorized_pdf_evaluation",
        uses_workspace_matrix=False,
        run_once=True,
    ),
    "cross_scalar_pdf_evaluation": BenchmarkSpec(
        name="cross_scalar_pdf_evaluation",
        group="scalar",
        kind="scalar_workspace_dir",
        module="src.run_cross_scalar_pdf_evaluation",
        uses_workspace_matrix=False,
        run_once=True,
    ),
    "benchmark_overview": BenchmarkSpec(
        name="benchmark_overview",
        group="overview",
        kind="overview",
        module="src.plot_benchmark_overview",
        uses_workspace_matrix=False,
        run_once=True,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a configurable benchmark matrix over selected workspaces."
    )

    parser.add_argument("--workspace-dir", type=Path, default=Path("inputs"))
    parser.add_argument("--root-workspace-dir", type=Path, default=None)
    parser.add_argument("--workspace-glob", default="*.json")
    parser.add_argument("--workspace-regex", default=None)
    parser.add_argument("--workspaces", nargs="+", type=Path, default=None)
    parser.add_argument("--exclude-workspaces", nargs="+", default=[])
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["all"],
        choices=["all", *BENCHMARKS.keys()],
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all"],
        choices=["all", "pyhs3", "cross", "scalar", "overview"],
    )
    parser.add_argument(
        "--exclude-benchmarks",
        nargs="+",
        default=[],
        choices=list(BENCHMARKS.keys()),
    )

    parser.add_argument("--targets", nargs="+", default=["L_ch0"])
    parser.add_argument("--modes", nargs="+", default=["FAST_RUN"])
    parser.add_argument("--stages", nargs="+", default=["all"])

    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument(
        "--n-evaluations",
        nargs="+",
        type=int,
        default=[100],
    )
    parser.add_argument("--n-scan-points", nargs="+", type=int, default=[101])
    parser.add_argument("--n-points", nargs="+", type=int, default=[101])
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=7)
    parser.add_argument("--warmup-evaluations", type=int, default=100)
    parser.add_argument("--validation-points", type=int, default=257)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-batches", type=int, default=9)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--scan-repeats", type=int, default=5)
    parser.add_argument("--input-modes", nargs="+", default=["varying"])
    parser.add_argument(
        "--categories", nargs="+", default=["pointwise_nll", "batched_full_dataset_nll"]
    )

    parser.add_argument("--distribution", default="sig_ch0")
    parser.add_argument("--nll-distribution", default="model_ch0")
    parser.add_argument("--scan-parameter", default="mu_sig")
    parser.add_argument("--scan-min", type=float, default=0.0)
    parser.add_argument("--scan-max", type=float, default=2.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--delta-reference-mu", type=float, default=0.0)

    parser.add_argument("--frameworks", nargs="+", default=None)
    parser.add_argument("--scalar-frameworks", nargs="+", default=None)
    parser.add_argument("--scenarios", nargs="+", default=None)

    parser.add_argument("--analysis", default="L_ch0")
    parser.add_argument("--pyhs3-data-name", default=None)
    parser.add_argument("--xroofit-model-name", default=None)
    parser.add_argument("--xroofit-dataset-name", default="combData")
    parser.add_argument("--root-workspace-name", default="combWS")
    parser.add_argument("--poi", default="mu_sig")
    parser.add_argument("--xroofit-library", default="libxRooFit")
    parser.add_argument(
        "--xroofit-pyhs3-combined",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate all inferred channels in the PyHS3 side of the xRooFit benchmark.",
    )
    parser.add_argument(
        "--xroofit-pyhs3-channels",
        default=None,
        help="Optional comma-separated channel list, for example ch0,ch1,ch2.",
    )
    parser.add_argument("--pyhs3-noncompiled-mode", default="FAST_COMPILE")
    parser.add_argument("--pyhs3-compiled-mode", default="FAST_RUN")
    parser.add_argument(
        "--pyhs3-nll-mode",
        choices=["logpdf", "extended-mixture"],
        default="extended-mixture",
    )
    parser.add_argument("--n-warmup-evaluations", type=int, default=3)
    parser.add_argument("--n-evaluation-runs", type=int, default=20)
    parser.add_argument("--n-scan-runs", type=int, default=10)
    parser.add_argument("--poi-timing-value", type=float, default=1.0)
    parser.add_argument("--delta-tolerance", type=float, default=1e-6)
    parser.add_argument("--delta-relative-tolerance", type=float, default=1e-7)
    parser.add_argument("--absolute-pyhs3-tolerance", type=float, default=1e-10)
    parser.add_argument("--minimum-tolerance", type=float, default=1e-12)

    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/benchmark_matrix")
    )
    parser.add_argument("--plot-dir", type=Path, default=Path("plots/benchmark_matrix"))
    parser.add_argument("--report-name", default="matrix_summary.json")

    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=None)

    return parser.parse_args()


def selected_benchmarks(args: argparse.Namespace) -> list[BenchmarkSpec]:
    if "all" in args.benchmarks:
        selected = list(BENCHMARKS.values())
    else:
        selected = [BENCHMARKS[name] for name in args.benchmarks]

    if "all" not in args.groups:
        selected = [spec for spec in selected if spec.group in set(args.groups)]

    excluded = set(args.exclude_benchmarks)
    return [spec for spec in selected if spec.name not in excluded]


def discover_workspaces(args: argparse.Namespace) -> list[Path]:
    if args.workspaces is not None:
        paths = [path for path in args.workspaces]
    else:
        paths = sorted(args.workspace_dir.glob(args.workspace_glob))

    paths = [path for path in paths if path.suffix == ".json"]

    if args.workspace_regex:
        pattern = re.compile(args.workspace_regex)
        paths = [path for path in paths if pattern.search(path.name)]

    for exclude_pattern in args.exclude_workspaces:
        paths = [
            path for path in paths if not fnmatch.fnmatch(path.name, exclude_pattern)
        ]

    if args.limit is not None:
        paths = paths[: args.limit]

    return paths


def paired_root_path(json_workspace: Path, args: argparse.Namespace) -> Path | None:
    if args.root_workspace_dir is not None:
        candidate = args.root_workspace_dir / f"{json_workspace.stem}.root"
    else:
        candidate = json_workspace.with_suffix(".root")

    return candidate if candidate.exists() else None


def make_output_paths(
    args: argparse.Namespace,
    benchmark: str,
    workspace: Path | None,
    repeat_index: int,
) -> tuple[Path, Path, str]:
    workspace_part = workspace.stem if workspace is not None else "global"
    run_part = f"repeat_{repeat_index:03d}"

    output_dir = args.output_dir / benchmark / workspace_part / run_part
    plot_dir = args.plot_dir / benchmark / workspace_part / run_part
    output_name = f"{benchmark}_result.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, plot_dir, output_name


def make_batch_output_paths(
    args: argparse.Namespace,
    benchmark: str,
    repeat_index: int,
) -> tuple[Path, Path, str]:
    """
    Return output and plot paths for a benchmark run that receives all selected
    workspaces in one subprocess.

    Batch mode is used for documentation-style comparison plots. Individual
    workspace runs cannot create comparison plots because each subprocess sees
    only one successful workspace.
    """

    run_part = f"repeat_{repeat_index:03d}"

    output_dir = args.output_dir / benchmark / "global" / run_part
    plot_dir = args.plot_dir / benchmark
    output_name = f"{benchmark}_result.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, plot_dir, output_name


def base_command(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def command_for_multi_workspace(
    spec: BenchmarkSpec,
    workspace: Path,
    args: argparse.Namespace,
    output_dir: Path,
    plot_dir: Path,
    output_name: str,
) -> list[str]:
    cmd = base_command(spec.module)

    if spec.name == "cross_nll_scan":
        root_workspace = paired_root_path(workspace, args)
        if root_workspace is None:
            raise FileNotFoundError(f"Missing matching ROOT workspace for {workspace}")
        cmd += [
            "--workspaces",
            str(workspace),
            "--root-workspaces",
            str(root_workspace),
            "--engines",
            *(args.frameworks or ["pyhs3_noncompiled", "pyhs3_compiled", "roofit"]),
            "--categories",
            *args.categories,
            "--analysis",
            args.analysis,
            "--distribution",
            args.nll_distribution,
            "--data-name",
            args.pyhs3_data_name or "combData_ch0",
            "--poi",
            args.poi,
            "--mode",
            args.modes[0],
            "--mu-min",
            str(args.scan_min),
            "--mu-max",
            str(args.scan_max),
            "--n-mu-values",
            str(args.n_points[0]),
            "--batch-size",
            str(args.batch_size),
            "--n-batches",
            str(args.n_batches),
            "--warmup-batches",
            str(args.warmup_batches),
            "--scan-repeats",
            str(args.scan_repeats),
            "--output",
            str(output_dir / output_name),
        ]
        if args.plot:
            cmd += ["--plot", "--plot-dir", str(plot_dir)]
        return cmd

    cmd += ["--workspaces", str(workspace)]
    cmd += ["--output-dir", str(output_dir), "--output-name", output_name]

    if spec.name not in {"workspace_loading"}:
        cmd += ["--targets", *args.targets]

    if spec.name not in {"workspace_loading"}:
        cmd += ["--modes", *args.modes]

    if spec.name in {
        "workspace_loading",
        "model_creation",
        "log_prob_construction",
        "log_prob_compilation",
        "graph_canonicalization",
        "graph_optimization",
        "memory_scaling",
        "model_complexity_scaling",
    }:
        cmd += ["--n-runs", str(args.n_runs)]

    if spec.name in {
        "compiled_evaluation",
        "pdf_evaluation",
        "memory_scaling",
        "model_complexity_scaling",
    }:
        cmd += ["--n-evaluations", *[str(value) for value in args.n_evaluations]]

    if spec.name in {"pdf_evaluation", "memory_scaling", "model_complexity_scaling"}:
        cmd += ["--distribution", args.distribution]

    if spec.name in {"nll_scan", "memory_scaling", "model_complexity_scaling"}:
        cmd += [
            "--scan-parameter",
            args.scan_parameter,
            "--scan-min",
            str(args.scan_min),
            "--scan-max",
            str(args.scan_max),
        ]

    if spec.name == "nll_scan":
        cmd += ["--n-scan-points", *[str(value) for value in args.n_scan_points]]

    if spec.name in {"memory_scaling", "model_complexity_scaling"}:
        cmd += ["--n-scan-points", str(args.n_scan_points[0])]
        cmd += ["--stages", *args.stages]

    if spec.name == "model_complexity_scaling":
        report_dir = output_dir / "reports"
        cmd += ["--report-dir", str(report_dir)]

    if args.plot:
        cmd += ["--plot", "--plot-dir", str(plot_dir)]

    return cmd


def command_for_multi_workspace_batch(
    spec: BenchmarkSpec,
    workspaces: list[Path],
    args: argparse.Namespace,
    output_dir: Path,
    plot_dir: Path,
    output_name: str,
) -> list[str]:
    """
    Build a command that passes all selected workspaces to one benchmark process.

    This is required for comparison plots: benchmark modules such as
    ``run_workspace_loading`` create bar plots only when they receive multiple
    successful workspace results in the same execution.
    """

    if not workspaces:
        raise ValueError("Batch benchmark execution requires at least one workspace")

    cmd = base_command(spec.module)

    if spec.name == "cross_nll_scan":
        root_workspaces = [
            paired_root_path(workspace, args) for workspace in workspaces
        ]
        missing = [
            str(workspace)
            for workspace, root_workspace in zip(
                workspaces, root_workspaces, strict=True
            )
            if root_workspace is None
        ]
        if missing:
            raise FileNotFoundError(
                "Missing matching ROOT workspaces for: " + ", ".join(missing)
            )
        cmd += [
            "--workspaces",
            *[str(workspace) for workspace in workspaces],
            "--root-workspaces",
            *[str(path) for path in root_workspaces if path is not None],
            "--engines",
            *(args.frameworks or ["pyhs3_noncompiled", "pyhs3_compiled", "roofit"]),
            "--categories",
            *args.categories,
            "--analysis",
            args.analysis,
            "--distribution",
            args.nll_distribution,
            "--data-name",
            args.pyhs3_data_name or "combData_ch0",
            "--poi",
            args.poi,
            "--mode",
            args.modes[0],
            "--mu-min",
            str(args.scan_min),
            "--mu-max",
            str(args.scan_max),
            "--n-mu-values",
            str(args.n_points[0]),
            "--batch-size",
            str(args.batch_size),
            "--n-batches",
            str(args.n_batches),
            "--warmup-batches",
            str(args.warmup_batches),
            "--scan-repeats",
            str(args.scan_repeats),
            "--output",
            str(output_dir / output_name),
        ]
        if args.plot:
            cmd += ["--plot", "--plot-dir", str(plot_dir)]
        return cmd

    cmd += ["--workspaces", *[str(workspace) for workspace in workspaces]]
    cmd += ["--output-dir", str(output_dir), "--output-name", output_name]

    if spec.name not in {"workspace_loading"}:
        cmd += ["--targets", *args.targets]

    if spec.name not in {"workspace_loading"}:
        cmd += ["--modes", *args.modes]

    if spec.name in {
        "workspace_loading",
        "model_creation",
        "log_prob_construction",
        "log_prob_compilation",
        "graph_canonicalization",
        "graph_optimization",
        "memory_scaling",
        "model_complexity_scaling",
    }:
        cmd += ["--n-runs", str(args.n_runs)]

    if spec.name in {
        "compiled_evaluation",
        "pdf_evaluation",
        "memory_scaling",
        "model_complexity_scaling",
    }:
        cmd += ["--n-evaluations", *[str(value) for value in args.n_evaluations]]

    if spec.name in {"pdf_evaluation", "memory_scaling", "model_complexity_scaling"}:
        cmd += ["--distribution", args.distribution]

    if spec.name in {"nll_scan", "memory_scaling", "model_complexity_scaling"}:
        cmd += [
            "--scan-parameter",
            args.scan_parameter,
            "--scan-min",
            str(args.scan_min),
            "--scan-max",
            str(args.scan_max),
        ]

    if spec.name == "nll_scan":
        cmd += ["--n-scan-points", *[str(value) for value in args.n_scan_points]]

    if spec.name in {"memory_scaling", "model_complexity_scaling"}:
        cmd += ["--n-scan-points", str(args.n_scan_points[0])]
        cmd += ["--stages", *args.stages]

    if spec.name == "model_complexity_scaling":
        report_dir = output_dir / "reports"
        cmd += ["--report-dir", str(report_dir)]

    if args.plot:
        cmd += ["--plot", "--plot-dir", str(plot_dir)]

    return cmd


def command_for_single_workspace(
    spec: BenchmarkSpec,
    workspace: Path,
    args: argparse.Namespace,
    output_dir: Path,
    plot_dir: Path,
    output_name: str,
) -> list[str]:
    cmd = base_command(spec.module)

    if spec.name == "cross_binned_likelihood_evaluation":
        cmd += [
            "--workspace",
            str(workspace),
            "--frameworks",
            *(args.frameworks or ["pyhs3", "pyhf", "roofit"]),
            "--target",
            args.targets[0],
            "--n-points",
            str(args.n_points[0]),
            "--output",
            str(output_dir / output_name),
        ]
        if getattr(args, "fail_fast", False):
            cmd.append("--fail-fast")

    elif spec.name == "cross_nll_scan":
        cmd += [
            "--workspace",
            str(workspace),
            "--frameworks",
            *(args.frameworks or ["pyhs3", "pyhf", "roofit"]),
            "--analysis",
            args.analysis,
            "--distribution",
            getattr(args, "nll_distribution", "model_ch0"),
            "--poi",
            args.poi,
            "--n-points",
            str(args.n_points[0]),
            "--scan-min",
            str(args.scan_min),
            "--scan-max",
            str(args.scan_max),
            "--output",
            str(output_dir / output_name),
        ]
        if getattr(args, "fail_fast", False):
            cmd.append("--fail-fast")

    else:
        raise ValueError(f"Unsupported single-workspace benchmark: {spec.name}")

    if args.plot:
        cmd += ["--plot", "--plot-dir", str(plot_dir)]

    return cmd


def command_for_json_root_pair(
    spec: BenchmarkSpec,
    json_workspace: Path,
    root_workspace: Path,
    args: argparse.Namespace,
    output_dir: Path,
    plot_dir: Path,
    output_name: str,
) -> list[str]:
    """Build a command for the current compiled/non-compiled PyHS3 vs xRooFit benchmark."""

    if spec.name != "pyhs3_xroofit_benchmark":
        raise ValueError(f"Unsupported JSON/ROOT-pair benchmark: {spec.name}")

    cmd = base_command(spec.module)
    cmd += [
        "--json-workspace",
        str(json_workspace),
        "--root-workspace",
        str(root_workspace),
        "--analysis",
        args.analysis,
        "--target",
        args.targets[0],
        "--xroofit-dataset-name",
        args.xroofit_dataset_name,
        "--root-workspace-name",
        args.root_workspace_name,
        "--poi",
        args.poi,
        "--pyhs3-noncompiled-mode",
        getattr(args, "pyhs3_noncompiled_mode", "FAST_COMPILE"),
        "--pyhs3-compiled-mode",
        getattr(args, "pyhs3_compiled_mode", "FAST_RUN"),
        "--pyhs3-nll-mode",
        getattr(args, "pyhs3_nll_mode", "extended-mixture"),
        "--scan-min",
        str(args.scan_min),
        "--scan-max",
        str(args.scan_max),
        "--n-scan-points",
        str(args.n_points[0]),
        "--n-warmup-evaluations",
        str(getattr(args, "n_warmup_evaluations", 3)),
        "--n-evaluation-runs",
        str(getattr(args, "n_evaluation_runs", 20)),
        "--n-scan-runs",
        str(getattr(args, "n_scan_runs", 10)),
        "--poi-timing-value",
        str(getattr(args, "poi_timing_value", 1.0)),
        "--delta-tolerance",
        str(getattr(args, "delta_tolerance", 1e-6)),
        "--delta-relative-tolerance",
        str(getattr(args, "delta_relative_tolerance", 1e-7)),
        "--absolute-pyhs3-tolerance",
        str(getattr(args, "absolute_pyhs3_tolerance", 1e-10)),
        "--minimum-tolerance",
        str(getattr(args, "minimum_tolerance", 1e-12)),
        "--output",
        str(output_dir / output_name),
        "--xroofit-library",
        args.xroofit_library,
    ]

    if getattr(args, "xroofit_pyhs3_combined", True):
        cmd.append("--pyhs3-combined")
    if getattr(args, "xroofit_pyhs3_channels", None):
        cmd += ["--pyhs3-channels", getattr(args, "xroofit_pyhs3_channels", None)]
    if args.pyhs3_data_name:
        cmd += ["--pyhs3-data-name", args.pyhs3_data_name]
    if args.xroofit_model_name:
        cmd += ["--xroofit-model-name", args.xroofit_model_name]
    if args.plot:
        cmd += ["--plot", "--plot-dir", str(plot_dir)]

    return cmd


def command_for_run_once(
    spec: BenchmarkSpec,
    args: argparse.Namespace,
    output_dir: Path,
    plot_dir: Path,
    output_name: str,
    workspaces: list[Path] | None = None,
) -> list[str]:
    cmd = base_command(spec.module)

    if spec.name == "cross_scalar_pdf_evaluation":
        cmd += [
            "--workspace-dir",
            str(args.workspace_dir),
        ]
        if args.root_workspace_dir is not None:
            cmd += ["--root-workspace-dir", str(args.root_workspace_dir)]
        cmd += [
            "--frameworks",
            *list(
                dict.fromkeys(
                    [
                        *(
                            args.scalar_frameworks
                            or ["pyhs3_noncompiled", "pyhs3_compiled"]
                        ),
                        "root",
                    ]
                )
            ),
            "--input-modes",
            *getattr(args, "input_modes", ["varying"]),
            "--distribution",
            args.distribution,
            "--target",
            args.targets[0],
            "--mode",
            args.modes[0],
            "--n-evaluations",
            *[str(v) for v in args.n_evaluations],
            "--timing-repeats",
            str(getattr(args, "timing_repeats", 7)),
            "--warmup-evaluations",
            str(getattr(args, "warmup_evaluations", 100)),
            "--validation-points",
            str(getattr(args, "validation_points", 257)),
            "--output",
            str(output_dir / output_name),
        ]

    elif spec.name == "cross_vectorized_pdf_evaluation":
        cmd += [
            "--workspace-dir",
            str(args.workspace_dir),
            "--scenarios",
            *(args.scenarios or []),
            "--n-points",
            str(args.n_points[0]),
            "--output",
            str(output_dir / output_name),
        ]

    elif spec.name == "cross_model_complexity_scaling":
        cmd += [
            "--json-input-dir",
            str(args.workspace_dir),
            "--root-input-dir",
            str(args.root_workspace_dir or args.workspace_dir),
            "--output",
            str(output_dir / output_name),
        ]
        if getattr(args, "fail_fast", False):
            cmd.append("--fail-fast")

    elif spec.name == "benchmark_overview":
        cmd += [
            "--results-dir",
            str(args.output_dir),
            "--plot-dir",
            str(plot_dir),
            "--include-failed",
        ]

    else:
        raise ValueError(f"Unsupported run-once benchmark: {spec.name}")

    if args.plot and spec.name != "benchmark_overview":
        cmd += ["--plot", "--plot-dir", str(plot_dir)]

    return cmd


def run_command(
    *,
    command: list[str],
    spec: BenchmarkSpec,
    workspace: Path | None,
    root_workspace: Path | None,
    output_dir: Path,
    timeout_seconds: float | None,
    dry_run: bool,
) -> RunRecord:
    stdout_path = output_dir / "stdout.txt"
    stderr_path = output_dir / "stderr.txt"

    if dry_run:
        return RunRecord(
            benchmark=spec.name,
            group=spec.group,
            workspace=str(workspace) if workspace else None,
            root_workspace=str(root_workspace) if root_workspace else None,
            command=command,
            status="dry_run",
            returncode=None,
            duration_seconds=0.0,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    start = time.perf_counter()

    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_seconds,
        )
        duration = time.perf_counter() - start

        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        status = "success" if completed.returncode == 0 else "failed"

        return RunRecord(
            benchmark=spec.name,
            group=spec.group,
            workspace=str(workspace) if workspace else None,
            root_workspace=str(root_workspace) if root_workspace else None,
            command=command,
            status=status,
            returncode=completed.returncode,
            duration_seconds=duration,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error=None if status == "success" else completed.stderr[-4000:],
        )

    except subprocess.TimeoutExpired as exc:
        duration = time.perf_counter() - start
        stderr_path.write_text(str(exc), encoding="utf-8")

        return RunRecord(
            benchmark=spec.name,
            group=spec.group,
            workspace=str(workspace) if workspace else None,
            root_workspace=str(root_workspace) if root_workspace else None,
            command=command,
            status="timeout",
            returncode=None,
            duration_seconds=duration,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error=str(exc),
        )

    except Exception as exc:
        duration = time.perf_counter() - start
        stderr_path.write_text(repr(exc), encoding="utf-8")

        return RunRecord(
            benchmark=spec.name,
            group=spec.group,
            workspace=str(workspace) if workspace else None,
            root_workspace=str(root_workspace) if root_workspace else None,
            command=command,
            status="error",
            returncode=None,
            duration_seconds=duration,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error=repr(exc),
        )


def write_summary(args: argparse.Namespace, records: list[RunRecord]) -> None:
    summary = {
        "total": len(records),
        "success": sum(record.status == "success" for record in records),
        "failed": sum(record.status == "failed" for record in records),
        "timeout": sum(record.status == "timeout" for record in records),
        "error": sum(record.status == "error" for record in records),
        "dry_run": sum(record.status == "dry_run" for record in records),
        "skipped": sum(record.status == "skipped" for record in records),
        "records": [asdict(record) for record in records],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / args.report_name).write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    failed_lines = []
    for record in records:
        if record.status in {"failed", "error", "timeout"}:
            failed_lines.append(
                "\n".join(
                    [
                        "=" * 80,
                        f"benchmark: {record.benchmark}",
                        f"workspace:  {record.workspace}",
                        f"status:     {record.status}",
                        f"stderr:     {record.stderr_path}",
                        f"error:      {record.error or ''}",
                    ]
                )
            )

    (args.output_dir / "failed_summary.txt").write_text(
        "\n\n".join(failed_lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    specs = selected_benchmarks(args)
    workspaces = discover_workspaces(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    records: list[RunRecord] = []

    print(f"Selected benchmarks: {', '.join(spec.name for spec in specs)}")
    print(f"Selected workspaces: {len(workspaces)}")
    print()

    for repeat_index in range(args.repeat):
        print("=" * 80)
        print(f"Repeat {repeat_index + 1} / {args.repeat}")
        print("=" * 80)

        for spec in specs:
            if spec.kind == "multi_workspace" and args.plot:
                output_dir, plot_dir, output_name = make_batch_output_paths(
                    args, spec.name, repeat_index
                )
                command = command_for_multi_workspace_batch(
                    spec, workspaces, args, output_dir, plot_dir, output_name
                )

                print(f"[{spec.group}] {spec.name} all selected workspaces")
                print(" ".join(command))

                record = run_command(
                    command=command,
                    spec=spec,
                    workspace=None,
                    root_workspace=None,
                    output_dir=output_dir,
                    timeout_seconds=args.timeout_seconds,
                    dry_run=args.dry_run,
                )
                records.append(record)
                print(f"  -> {record.status}")

                write_summary(args, records)

                if args.fail_fast and record.status not in {"success", "dry_run"}:
                    write_summary(args, records)
                    raise SystemExit(1)

                continue

            if spec.run_once:
                output_dir, plot_dir, output_name = make_output_paths(
                    args, spec.name, None, repeat_index
                )
                command = command_for_run_once(
                    spec, args, output_dir, plot_dir, output_name, workspaces
                )

                print(f"[{spec.group}] {spec.name}")
                print(" ".join(command))

                record = run_command(
                    command=command,
                    spec=spec,
                    workspace=None,
                    root_workspace=None,
                    output_dir=output_dir,
                    timeout_seconds=args.timeout_seconds,
                    dry_run=args.dry_run,
                )
                records.append(record)
                print(f"  -> {record.status}")

                write_summary(args, records)

                if args.fail_fast and record.status not in {"success", "dry_run"}:
                    write_summary(args, records)
                    raise SystemExit(1)

                continue

            for workspace in workspaces:
                output_dir, plot_dir, output_name = make_output_paths(
                    args, spec.name, workspace, repeat_index
                )

                root_workspace = paired_root_path(workspace, args)

                if spec.requires_root_pair and root_workspace is None:
                    record = RunRecord(
                        benchmark=spec.name,
                        group=spec.group,
                        workspace=str(workspace),
                        root_workspace=None,
                        command=[],
                        status="skipped",
                        returncode=None,
                        duration_seconds=0.0,
                        stdout_path=str(output_dir / "stdout.txt"),
                        stderr_path=str(output_dir / "stderr.txt"),
                        error="Missing matching ROOT workspace.",
                    )
                    records.append(record)
                    print(f"[{spec.group}] {spec.name} {workspace.name} -> skipped")
                    write_summary(args, records)
                    continue

                if spec.kind == "multi_workspace":
                    command = command_for_multi_workspace(
                        spec, workspace, args, output_dir, plot_dir, output_name
                    )
                elif spec.kind == "single_workspace":
                    command = command_for_single_workspace(
                        spec, workspace, args, output_dir, plot_dir, output_name
                    )
                elif spec.kind == "json_root_pair":
                    assert root_workspace is not None
                    command = command_for_json_root_pair(
                        spec,
                        workspace,
                        root_workspace,
                        args,
                        output_dir,
                        plot_dir,
                        output_name,
                    )
                else:
                    raise ValueError(f"Unsupported benchmark kind: {spec.kind}")

                print(f"[{spec.group}] {spec.name} {workspace.name}")
                print(" ".join(command))

                record = run_command(
                    command=command,
                    spec=spec,
                    workspace=workspace,
                    root_workspace=root_workspace,
                    output_dir=output_dir,
                    timeout_seconds=args.timeout_seconds,
                    dry_run=args.dry_run,
                )
                records.append(record)
                print(f"  -> {record.status}")

                write_summary(args, records)

                if args.fail_fast and record.status not in {"success", "dry_run"}:
                    write_summary(args, records)
                    raise SystemExit(1)

    write_summary(args, records)
    print()
    print(f"Saved matrix summary to {args.output_dir / args.report_name}")


if __name__ == "__main__":
    main()
