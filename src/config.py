"""
Configuration file for benchmarking scripts.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WORKSPACE = REPO_ROOT / "inputs" / "simple_workspace_nonp.json"

DEFAULT_TARGET = "L_ch0"
DEFAULT_MODE = "FAST_RUN"
DEFAULT_N_RUNS = 5

RESULTS_DIR = REPO_ROOT / "results"
PLOTS_DIR = REPO_ROOT / "plots"
REPORTS_DIR = REPO_ROOT / "reports"

WORKSPACE_LABELS = {
    "simple_workspace_nonp": "Simple nonp",
    "simple_workspace": "Simple",
    "simple_workspace_generic_nonp": "Generic nonp",
    "simple_workspace_generic": "Generic",
}
