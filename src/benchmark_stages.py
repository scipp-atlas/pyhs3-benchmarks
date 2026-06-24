from __future__ import annotations

from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable

from run_compiled_evaluation import run_single_benchmark as run_compiled_evaluation
from run_log_prob_compilation import run_single_benchmark as run_log_prob_compilation
from run_log_prob_construction import run_single_benchmark as run_log_prob_construction
from run_model_creation import run_single_benchmark as run_model_creation
from run_nll_scan import run_single_benchmark as run_nll_scan
from run_pdf_evaluation import run_single_benchmark as run_pdf_evaluation
from run_workspace_loading import run_single_benchmark as run_workspace_loading

WORKFLOW_STAGES = [
    "workspace_loading",
    "model_creation",
    "log_prob_construction",
    "log_prob_compilation",
    "compiled_evaluation",
    "pdf_evaluation",
    "nll_scan",
]

DEFAULT_N_EVALUATIONS = 100
DEFAULT_DISTRIBUTION = "sig_ch0"

DEFAULT_SCAN_PARAMETER = "mu_sig"
DEFAULT_SCAN_MIN = 0.0
DEFAULT_SCAN_MAX = 5.0
DEFAULT_N_SCAN_POINTS = 101

def resolve_stages(
    stages: list[str],
    available_stages: list[str] = WORKFLOW_STAGES,
) -> list[str]:
    """
    Resolve the list of stages to run.

    Passing ``["all"]`` expands to *available_stages*.  Any unknown stage
    name raises :exc:`ValueError`.
    """

    if "all" in stages:
        if len(stages) > 1:
            raise ValueError("--stages all cannot be combined with other stages")
        return list(available_stages)

    unknown_stages = [s for s in stages if s not in available_stages]

    if unknown_stages:
        raise ValueError(
            f"Unknown stages: {unknown_stages}. "
            f"Available stages: {available_stages}"
        )

    return stages


def build_stage_specs(
    selected_stages: list[str],
    workspace_path: Path,
    target: str,
    mode: str,
    n_runs: int,
    n_evaluations: int,
    distribution: str,
    scan_parameter: str,
    scan_min: float,
    scan_max: float,
    n_scan_points: int,
) -> list[tuple[str, Callable[..., dict[str, Any]], tuple[Any, ...]]]:
    """
    Build a list of ``(stage_name, function, args)`` triples for the
    *selected_stages*, ready to be executed sequentially.
    """

    specs: dict[str, tuple[Callable[..., dict[str, Any]], tuple[Any, ...]]] = {
        "workspace_loading": (
            run_workspace_loading,
            (workspace_path, n_runs),
        ),
        "model_creation": (
            run_model_creation,
            (workspace_path, target, mode, n_runs),
        ),
        "log_prob_construction": (
            run_log_prob_construction,
            (workspace_path, target, mode, n_runs),
        ),
        "log_prob_compilation": (
            run_log_prob_compilation,
            (workspace_path, target, mode, n_runs),
        ),
        "compiled_evaluation": (
            run_compiled_evaluation,
            (workspace_path, target, mode, n_evaluations),
        ),
        "pdf_evaluation": (
            run_pdf_evaluation,
            (workspace_path, target, mode, distribution, n_evaluations),
        ),
        "nll_scan": (
            run_nll_scan,
            (
                workspace_path,
                target,
                mode,
                scan_parameter,
                scan_min,
                scan_max,
                n_scan_points,
            ),
        ),
    }

    return [(stage, specs[stage][0], specs[stage][1]) for stage in selected_stages]


def run_stage_isolated(
    function: Callable[..., dict[str, Any]],
    args: tuple[Any, ...],
) -> dict[str, Any]:
    """
    Run one benchmark stage in a fresh child process (spawn context).
    """

    ctx = get_context("spawn")

    with ctx.Pool(processes=1) as pool:
        return pool.apply(function, args=args)
