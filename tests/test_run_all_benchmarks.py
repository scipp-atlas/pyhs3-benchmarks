from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src import run_all_benchmarks as benchmark


@pytest.fixture
def command() -> benchmark.BenchmarkCommand:
    return benchmark.BenchmarkCommand(name="demo", command=[sys.executable, "demo.py"])


def make_result(
    name: str = "demo",
    status: str = "success",
    returncode: int | None = 0,
    duration_seconds: float = 1.0,
    command: list[str] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
) -> benchmark.BenchmarkRunResult:
    return benchmark.BenchmarkRunResult(
        name=name,
        command=command or [sys.executable, "-m", f"src.{name}"],
        status=status,
        returncode=returncode,
        duration_seconds=duration_seconds,
        error_type=error_type,
        error_message=error_message,
    )


def make_args(**overrides: Any) -> argparse.Namespace:
    values = {
        "preset": None,
        "n_runs": None,
        "n_evaluations": None,
        "n_scan_points": None,
        "no_plot": False,
        "no_plot_was_set": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_benchmark_command_dataclass(command: benchmark.BenchmarkCommand) -> None:
    assert command.name == "demo"
    assert command.command == [sys.executable, "demo.py"]


def test_benchmark_run_result_dataclass() -> None:
    result = make_result(
        name="demo",
        status="failed",
        returncode=7,
        error_type="CalledProcessError",
        error_message="Benchmark exited with code 7",
    )

    assert result.name == "demo"
    assert result.status == "failed"
    assert result.returncode == 7
    assert result.error_type == "CalledProcessError"
    assert result.error_message == "Benchmark exited with code 7"


def test_module_name_success() -> None:
    assert (
        benchmark.module_name("run_workspace_loading.py") == "src.run_workspace_loading"
    )


def test_module_name_rejects_non_python_script() -> None:
    with pytest.raises(ValueError, match="Expected a Python script name"):
        benchmark.module_name("run_workspace_loading")


def test_module_command_uses_python_module_execution() -> None:
    command = benchmark.module_command("run_workspace_loading.py")

    assert command == [sys.executable, "-m", "src.run_workspace_loading"]


@pytest.mark.parametrize(
    ("value", "name", "minimum"),
    [(1, "--n-runs", 1), (2, "--n-scan-points", 2)],
)
def test_validate_positive_int_success(value: int, name: str, minimum: int) -> None:
    benchmark.validate_positive_int(value, name, minimum)


@pytest.mark.parametrize(
    ("value", "name", "minimum", "message"),
    [
        (0, "--n-runs", 1, "--n-runs must be at least 1"),
        (1, "--n-scan-points", 2, "--n-scan-points must be at least 2"),
    ],
)
def test_validate_positive_int_rejects_invalid(
    value: int,
    name: str,
    minimum: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_positive_int(value, name, minimum)


def test_validate_target_and_mode_success() -> None:
    benchmark.validate_target_and_mode("L_ch0", "FAST_RUN")


@pytest.mark.parametrize(
    ("target", "mode", "message"),
    [
        ("", "FAST_RUN", "target must be a non-empty string"),
        ("L_ch0", "", "mode must be a non-empty string"),
        ("   ", "FAST_RUN", "target must be a non-empty string"),
        (object(), "FAST_RUN", "target must be a non-empty string"),
        ("L_ch0", object(), "mode must be a non-empty string"),
    ],
)
def test_validate_target_and_mode_rejects_invalid(
    target: Any,
    mode: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_target_and_mode(target, mode)


def test_validate_benchmark_config_success() -> None:
    benchmark.validate_benchmark_config(
        n_runs=1,
        n_evaluations=1,
        n_scan_points=2,
        target="L_ch0",
        mode="FAST_RUN",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {
                "n_runs": 0,
                "n_evaluations": 1,
                "n_scan_points": 2,
                "target": "L_ch0",
                "mode": "FAST_RUN",
            },
            "--n-runs must be at least 1",
        ),
        (
            {
                "n_runs": 1,
                "n_evaluations": 0,
                "n_scan_points": 2,
                "target": "L_ch0",
                "mode": "FAST_RUN",
            },
            "--n-evaluations must be at least 1",
        ),
        (
            {
                "n_runs": 1,
                "n_evaluations": 1,
                "n_scan_points": 1,
                "target": "L_ch0",
                "mode": "FAST_RUN",
            },
            "--n-scan-points must be at least 2",
        ),
        (
            {
                "n_runs": 1,
                "n_evaluations": 1,
                "n_scan_points": 2,
                "target": "",
                "mode": "FAST_RUN",
            },
            "target must be a non-empty string",
        ),
        (
            {
                "n_runs": 1,
                "n_evaluations": 1,
                "n_scan_points": 2,
                "target": "L_ch0",
                "mode": "",
            },
            "mode must be a non-empty string",
        ),
    ],
)
def test_validate_benchmark_config_rejects_invalid(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark.validate_benchmark_config(**kwargs)


def test_command_has_plot_flag() -> None:
    assert benchmark.command_has_plot_flag(["python", "script.py", "--plot"])
    assert not benchmark.command_has_plot_flag(["python", "script.py"])


def test_format_command_quotes_special_characters() -> None:
    assert benchmark.format_command(["python", "script.py", "value with spaces"]) == (
        "python script.py 'value with spaces'"
    )


def test_run_command_dry_run_does_not_call_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    command: benchmark.BenchmarkCommand,
) -> None:
    def fail_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setattr(benchmark.subprocess, "run", fail_run)

    result = benchmark.run_command(command, dry_run=True)

    assert result == benchmark.BenchmarkRunResult(
        name="demo",
        command=[sys.executable, "demo.py"],
        status="skipped_dry_run",
        returncode=0,
        duration_seconds=0.0,
    )


def test_run_command_success(
    monkeypatch: pytest.MonkeyPatch,
    command: benchmark.BenchmarkCommand,
) -> None:
    calls = []

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args[0], returncode=0)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([10.0, 12.0]).__next__)

    result = benchmark.run_command(command, dry_run=False)

    assert result.status == "success"
    assert result.returncode == 0
    assert result.duration_seconds == pytest.approx(2.0)
    assert calls[0][1]["cwd"] == benchmark.REPO_ROOT
    assert calls[0][1]["check"] is False


def test_run_command_failure(
    monkeypatch: pytest.MonkeyPatch,
    command: benchmark.BenchmarkCommand,
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=args[0], returncode=7)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([1.0, 1.5]).__next__)

    result = benchmark.run_command(command, dry_run=False)

    assert result.status == "failed"
    assert result.returncode == 7
    assert result.duration_seconds == pytest.approx(0.5)
    assert result.error_type == "CalledProcessError"
    assert result.error_message == "Benchmark exited with code 7"


def test_run_command_handles_os_error(
    monkeypatch: pytest.MonkeyPatch,
    command: benchmark.BenchmarkCommand,
) -> None:
    def failing_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("cannot execute")

    monkeypatch.setattr(benchmark.subprocess, "run", failing_run)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([2.0, 2.25]).__next__)

    result = benchmark.run_command(command, dry_run=False)

    assert result.status == "failed"
    assert result.returncode is None
    assert result.duration_seconds == pytest.approx(0.25)
    assert result.error_type == "OSError"
    assert result.error_message == "cannot execute"


def test_save_suite_summary_writes_expected_json(tmp_path: Path) -> None:
    output_path = tmp_path / "summary" / "suite.json"
    results = [
        make_result(name="ok", status="success", returncode=0, duration_seconds=1.5),
        make_result(
            name="bad",
            status="failed",
            returncode=7,
            duration_seconds=0.5,
            error_type="CalledProcessError",
            error_message="Benchmark exited with code 7",
        ),
        make_result(
            name="skip", status="skipped_dry_run", returncode=0, duration_seconds=0.0
        ),
    ]

    benchmark.save_suite_summary(
        run_results=results,
        output_path=output_path,
        total_time_seconds=2.0,
    )

    with output_path.open() as file:
        payload = json.load(file)

    assert payload["benchmark"] == "benchmark_suite"
    assert payload["total_time_seconds"] == 2.0
    assert payload["n_results"] == 3
    assert payload["n_success"] == 1
    assert payload["n_failed"] == 1
    assert payload["n_skipped_dry_run"] == 1
    assert payload["results"][0]["command_string"] == benchmark.format_command(
        results[0].command
    )
    assert payload["results"][1]["error_type"] == "CalledProcessError"


def assert_module_command(command: list[str], script_name: str) -> None:
    assert command[0] == sys.executable
    assert command[1] == "-m"
    assert command[2] == benchmark.module_name(script_name)


def test_build_core_benchmarks_with_plot() -> None:
    commands = benchmark.build_core_benchmarks(
        n_runs=3,
        n_evaluations=5,
        workspaces=["a.json", "b.json"],
        target="L_ch0",
        mode="FAST_RUN",
        plot=True,
    )

    assert [command.name for command in commands] == [
        "workspace_loading",
        "model_creation",
        "log_prob_construction",
        "log_prob_compilation",
        "compiled_evaluation",
    ]
    assert_module_command(commands[0].command, "run_workspace_loading.py")
    assert_module_command(commands[-1].command, "run_compiled_evaluation.py")
    assert all("--plot" in command.command for command in commands)
    assert commands[0].command[-2:] == ["3", "--plot"]
    assert "--n-evaluations" in commands[-1].command
    assert "5" in commands[-1].command


def test_build_core_benchmarks_without_plot() -> None:
    commands = benchmark.build_core_benchmarks(
        n_runs=3,
        n_evaluations=5,
        workspaces=["a.json"],
        target="L_ch0",
        mode="FAST_RUN",
        plot=False,
    )

    assert all("--plot" not in command.command for command in commands)


def test_build_pdf_benchmarks() -> None:
    commands = benchmark.build_pdf_benchmarks(n_evaluations=10, plot=True)

    assert [command.name for command in commands] == [
        "pdf_evaluation_simple",
        "pdf_evaluation_scalar",
    ]
    assert all(command.command[1] == "-m" for command in commands)
    assert all(command.command[2] == "src.run_pdf_evaluation" for command in commands)
    assert all("--plot" in command.command for command in commands)
    assert "plots/pdf_evaluation_simple" in commands[0].command
    assert "plots/pdf_evaluation_scalar" in commands[1].command


def test_build_nll_scan_benchmark() -> None:
    command = benchmark.build_nll_scan_benchmark(n_scan_points=11, plot=False)

    assert command.name == "nll_scan"
    assert_module_command(command.command, "run_nll_scan.py")
    assert "--plot" not in command.command
    assert command.command[-1] == "11"


def test_build_scaling_benchmarks() -> None:
    commands = benchmark.build_scaling_benchmarks(
        n_runs=2,
        n_evaluations=3,
        n_scan_points=4,
        plot=True,
    )

    assert [command.name for command in commands] == [
        "memory_scaling",
        "model_complexity_scaling",
    ]
    assert_module_command(commands[0].command, "run_memory_scaling.py")
    assert_module_command(commands[1].command, "run_model_complexity_scaling.py")
    assert all("--plot" in command.command for command in commands)
    assert "plots/memory_scaling_all_stages" in commands[0].command
    assert "plots/model_complexity_all_stages" in commands[1].command


def test_build_graph_benchmarks() -> None:
    commands = benchmark.build_graph_benchmarks(n_runs=3, plot=True)

    assert [command.name for command in commands] == [
        "graph_canonicalization",
        "graph_optimization",
    ]
    assert_module_command(commands[0].command, "run_graph_canonicalization.py")
    assert_module_command(commands[1].command, "run_graph_optimization.py")
    assert all("--plot" in command.command for command in commands)
    assert "plots/graph_canonicalization_simple" in commands[0].command
    assert "plots/graph_optimization_simple" in commands[1].command


def test_build_graph_benchmarks_without_plot() -> None:
    commands = benchmark.build_graph_benchmarks(n_runs=3, plot=False)

    assert all("--plot" not in command.command for command in commands)


def test_select_benchmarks_all() -> None:
    commands = [
        benchmark.BenchmarkCommand("a", ["a"]),
        benchmark.BenchmarkCommand("b", ["b"]),
    ]
    assert benchmark.select_benchmarks(["all"], commands) == commands


def test_select_benchmarks_subset_preserves_requested_order() -> None:
    commands = [
        benchmark.BenchmarkCommand("a", ["a"]),
        benchmark.BenchmarkCommand("b", ["b"]),
    ]
    assert [cmd.name for cmd in benchmark.select_benchmarks(["b", "a"], commands)] == [
        "b",
        "a",
    ]


def test_select_benchmarks_rejects_unknown() -> None:
    commands = [benchmark.BenchmarkCommand("a", ["a"])]
    with pytest.raises(ValueError, match="Unknown benchmark"):
        benchmark.select_benchmarks(["missing"], commands)


def test_apply_preset_none_returns_args() -> None:
    args = make_args()
    assert benchmark.apply_preset(args) is args


def test_apply_preset_smoke_sets_missing_values() -> None:
    args = make_args(preset="smoke")
    updated = benchmark.apply_preset(args)

    assert updated.n_runs == 1
    assert updated.n_evaluations == 1
    assert updated.n_scan_points == 11
    assert updated.no_plot is True


def test_apply_preset_default_enables_plots_when_not_explicitly_disabled() -> None:
    args = make_args(preset="default", no_plot=True, no_plot_was_set=False)
    updated = benchmark.apply_preset(args)

    assert updated.n_runs == 20
    assert updated.n_evaluations == 1000
    assert updated.n_scan_points == 1001
    assert updated.no_plot is False


def test_apply_preset_respects_explicit_overrides() -> None:
    args = make_args(
        preset="full",
        n_runs=8,
        n_evaluations=9,
        n_scan_points=10,
        no_plot=True,
        no_plot_was_set=True,
    )
    updated = benchmark.apply_preset(args)

    assert updated.n_runs == 8
    assert updated.n_evaluations == 9
    assert updated.n_scan_points == 10
    assert updated.no_plot is True


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_all_benchmarks.py"])

    args = benchmark.parse_args()

    assert args.benchmarks == ["all"]
    assert args.preset is None
    assert args.n_runs is None
    assert args.n_evaluations is None
    assert args.n_scan_points is None
    assert args.target == "L_ch0"
    assert args.mode == "FAST_RUN"
    assert args.no_plot is False
    assert args.summary_output == Path(
        "results/benchmark_suite/benchmark_suite_summary.json"
    )
    assert args.dry_run is False
    assert args.continue_on_failure is False
    assert args.no_plot_was_set is False


def test_parse_args_custom_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary_output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--benchmarks",
            "workspace_loading",
            "nll_scan",
            "--preset",
            "smoke",
            "--n-runs",
            "2",
            "--n-evaluations",
            "3",
            "--n-scan-points",
            "4",
            "--target",
            "analysis",
            "--mode",
            "FAST_COMPILE",
            "--no-plot",
            "--summary-output",
            str(summary_output),
            "--dry-run",
            "--continue-on-failure",
        ],
    )

    args = benchmark.parse_args()

    assert args.benchmarks == ["workspace_loading", "nll_scan"]
    assert args.preset == "smoke"
    assert args.n_runs == 2
    assert args.n_evaluations == 3
    assert args.n_scan_points == 4
    assert args.target == "analysis"
    assert args.mode == "FAST_COMPILE"
    assert args.no_plot is True
    assert args.summary_output == summary_output
    assert args.dry_run is True
    assert args.continue_on_failure is True
    assert args.no_plot_was_set is True


def test_main_dry_run_smoke_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--preset",
            "smoke",
            "--summary-output",
            str(summary_output),
            "--dry-run",
        ],
    )

    benchmark.main()

    output = capsys.readouterr().out

    assert "Selected benchmarks" in output
    assert "Dry run completed successfully" in output
    with summary_output.open() as file:
        payload = json.load(file)
    assert payload["n_failed"] == 0
    assert payload["n_skipped_dry_run"] == payload["n_results"]


def test_main_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all_benchmarks.py", "--n-runs", "0", "--dry-run"],
    )

    with pytest.raises(ValueError, match="--n-runs must be at least 1"):
        benchmark.main()


def test_main_rejects_unknown_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_all_benchmarks.py", "--benchmarks", "missing", "--dry-run"],
    )

    with pytest.raises(ValueError, match="Unknown benchmark"):
        benchmark.main()


def test_main_runs_selected_benchmark(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    summary_output = tmp_path / "summary.json"

    def fake_run_command(**kwargs: Any) -> benchmark.BenchmarkRunResult:
        benchmark_command = kwargs["benchmark"]
        dry_run = kwargs["dry_run"]
        calls.append((benchmark_command.name, dry_run, benchmark_command.command))
        return make_result(
            name=benchmark_command.name,
            status="success",
            returncode=0,
            duration_seconds=0.25,
            command=benchmark_command.command,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--benchmarks",
            "workspace_loading",
            "--summary-output",
            str(summary_output),
            "--dry-run",
        ],
    )
    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([1.0, 2.0]).__next__)

    benchmark.main()

    assert len(calls) == 1
    assert calls[0][0] == "workspace_loading"
    assert calls[0][1] is True
    with summary_output.open() as file:
        payload = json.load(file)
    assert payload["n_success"] == 1
    assert payload["total_time_seconds"] == 1.0


def test_main_uses_default_values_when_no_preset_or_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls = []
    summary_output = tmp_path / "summary.json"

    def fake_run_command(**kwargs: Any) -> benchmark.BenchmarkRunResult:
        benchmark_command = kwargs["benchmark"]
        calls.append(benchmark_command)
        return make_result(
            name=benchmark_command.name,
            command=benchmark_command.command,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--benchmarks",
            "compiled_evaluation",
            "--summary-output",
            str(summary_output),
        ],
    )
    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([0.0, 1.0]).__next__)

    benchmark.main()

    assert len(calls) == 1
    command = calls[0].command
    assert "--n-evaluations" in command
    assert command[command.index("--n-evaluations") + 1] == "1000"


def test_main_stops_on_first_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    summary_output = tmp_path / "summary.json"

    def fake_run_command(**kwargs: Any) -> benchmark.BenchmarkRunResult:
        benchmark_command = kwargs["benchmark"]
        calls.append(benchmark_command.name)
        return make_result(
            name=benchmark_command.name,
            status="failed",
            returncode=7,
            duration_seconds=0.5,
            command=benchmark_command.command,
            error_type="CalledProcessError",
            error_message="Benchmark exited with code 7",
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--benchmarks",
            "workspace_loading",
            "model_creation",
            "--summary-output",
            str(summary_output),
            "--dry-run",
        ],
    )
    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([1.0, 2.0]).__next__)

    with pytest.raises(SystemExit) as exc_info:
        benchmark.main()

    assert exc_info.value.code == 1
    assert calls == ["workspace_loading"]
    with summary_output.open() as file:
        payload = json.load(file)
    assert payload["n_failed"] == 1
    assert payload["n_results"] == 1


def test_main_continue_on_failure_runs_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []
    summary_output = tmp_path / "summary.json"

    def fake_run_command(**kwargs: Any) -> benchmark.BenchmarkRunResult:
        benchmark_command = kwargs["benchmark"]
        calls.append(benchmark_command.name)
        if benchmark_command.name == "workspace_loading":
            return make_result(
                name=benchmark_command.name,
                status="failed",
                returncode=7,
                duration_seconds=0.5,
                command=benchmark_command.command,
                error_type="CalledProcessError",
                error_message="Benchmark exited with code 7",
            )
        return make_result(
            name=benchmark_command.name,
            status="success",
            returncode=0,
            duration_seconds=0.5,
            command=benchmark_command.command,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--benchmarks",
            "workspace_loading",
            "model_creation",
            "--summary-output",
            str(summary_output),
            "--dry-run",
            "--continue-on-failure",
        ],
    )
    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([1.0, 2.0]).__next__)

    with pytest.raises(SystemExit) as exc_info:
        benchmark.main()

    assert exc_info.value.code == 1
    assert calls == ["workspace_loading", "model_creation"]
    with summary_output.open() as file:
        payload = json.load(file)
    assert payload["n_failed"] == 1
    assert payload["n_success"] == 1
    assert payload["n_results"] == 2


def test_main_handles_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary_output = tmp_path / "summary.json"

    def interrupting_run_command(**kwargs: Any) -> benchmark.BenchmarkRunResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--benchmarks",
            "workspace_loading",
            "--summary-output",
            str(summary_output),
            "--dry-run",
        ],
    )
    monkeypatch.setattr(benchmark, "run_command", interrupting_run_command)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([5.0, 7.0]).__next__)

    with pytest.raises(SystemExit) as exc_info:
        benchmark.main()

    assert exc_info.value.code == 130
    with summary_output.open() as file:
        payload = json.load(file)
    assert payload["n_results"] == 0
    assert payload["total_time_seconds"] == 2.0


def test_main_non_dry_run_success_prints_completed_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary_output = tmp_path / "summary.json"

    def fake_run_command(**kwargs: Any) -> benchmark.BenchmarkRunResult:
        benchmark_command = kwargs["benchmark"]
        assert kwargs["dry_run"] is False
        return make_result(
            name=benchmark_command.name,
            status="success",
            returncode=0,
            duration_seconds=0.5,
            command=benchmark_command.command,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--benchmarks",
            "workspace_loading",
            "--summary-output",
            str(summary_output),
        ],
    )
    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([3.0, 4.5]).__next__)

    benchmark.main()

    output = capsys.readouterr().out
    assert "All selected benchmarks completed successfully." in output

    with summary_output.open() as file:
        payload = json.load(file)

    assert payload["n_success"] == 1
    assert payload["n_failed"] == 0
    assert payload["total_time_seconds"] == 1.5


def test_module_main_guard_runs_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary_output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--preset",
            "smoke",
            "--summary-output",
            str(summary_output),
            "--dry-run",
        ],
    )

    runpy.run_module("src.run_all_benchmarks", run_name="__main__")

    assert summary_output.exists()
    with summary_output.open() as file:
        payload = json.load(file)
    assert payload["benchmark"] == "benchmark_suite"
    assert payload["n_failed"] == 0
