from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src import run_all_benchmarks as benchmark


def make_args(**overrides: Any) -> argparse.Namespace:
    values: dict[str, Any] = {
        "workspace_dir": Path("inputs"),
        "root_workspace_dir": None,
        "workspace_glob": "*.json",
        "workspace_regex": None,
        "workspaces": None,
        "exclude_workspaces": [],
        "limit": None,
        "benchmarks": ["all"],
        "groups": ["all"],
        "exclude_benchmarks": [],
        "targets": ["L_ch0"],
        "modes": ["FAST_RUN"],
        "stages": ["all"],
        "n_runs": 3,
        "n_evaluations": [100],
        "n_scan_points": [101],
        "n_points": [101],
        "warmup_iterations": 1,
        "distribution": "sig_ch0",
        "scan_parameter": "mu_sig",
        "scan_min": 0.0,
        "scan_max": 2.0,
        "mu": 1.0,
        "delta_reference_mu": 0.0,
        "frameworks": None,
        "scalar_frameworks": None,
        "scenarios": None,
        "analysis": "L_ch0",
        "pyhs3_data_name": None,
        "xroofit_model_name": None,
        "xroofit_dataset_name": "combData",
        "root_workspace_name": "combWS",
        "poi": "mu_sig",
        "xroofit_library": "libxRooFit",
        "output_dir": Path("results/benchmark_matrix"),
        "plot_dir": Path("plots/benchmark_matrix"),
        "report_name": "matrix_summary.json",
        "plot": False,
        "dry_run": False,
        "fail_fast": False,
        "repeat": 1,
        "timeout_seconds": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def spec(name: str) -> benchmark.BenchmarkSpec:
    return benchmark.BENCHMARKS[name]


def test_benchmark_spec_and_run_record_dataclasses(tmp_path: Path) -> None:
    benchmark_spec = benchmark.BenchmarkSpec(
        name="demo",
        group="pyhs3",
        kind="multi_workspace",
        module="src.demo",
        uses_workspace_matrix=True,
        requires_root_pair=True,
        run_once=False,
    )
    assert benchmark_spec.name == "demo"
    assert benchmark_spec.requires_root_pair is True

    record = benchmark.RunRecord(
        benchmark="demo",
        group="pyhs3",
        workspace="a.json",
        root_workspace=None,
        command=[sys.executable, "-m", "src.demo"],
        status="success",
        returncode=0,
        duration_seconds=0.25,
        stdout_path=str(tmp_path / "stdout.txt"),
        stderr_path=str(tmp_path / "stderr.txt"),
    )
    assert record.benchmark == "demo"
    assert record.error is None


def test_parse_args_defaults_and_custom(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_all_benchmarks.py"])
    args = benchmark.parse_args()
    assert args.workspace_dir == Path("inputs")
    assert args.benchmarks == ["all"]
    assert args.groups == ["all"]
    assert args.n_runs == 3
    assert args.n_evaluations == [100]
    assert args.n_scan_points == [101]
    assert args.plot is False

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--workspace-dir",
            str(tmp_path),
            "--workspace-regex",
            "5ch",
            "--benchmarks",
            "workspace_loading",
            "nll_scan",
            "--groups",
            "pyhs3",
            "--exclude-benchmarks",
            "nll_scan",
            "--targets",
            "A",
            "B",
            "--modes",
            "FAST_COMPILE",
            "--n-runs",
            "7",
            "--n-evaluations",
            "1",
            "2",
            "--n-scan-points",
            "3",
            "4",
            "--n-points",
            "5",
            "6",
            "--plot",
            "--dry-run",
            "--fail-fast",
            "--repeat",
            "2",
            "--timeout-seconds",
            "9.5",
        ],
    )
    args = benchmark.parse_args()
    assert args.workspace_dir == tmp_path
    assert args.workspace_regex == "5ch"
    assert args.benchmarks == ["workspace_loading", "nll_scan"]
    assert args.exclude_benchmarks == ["nll_scan"]
    assert args.targets == ["A", "B"]
    assert args.modes == ["FAST_COMPILE"]
    assert args.n_runs == 7
    assert args.n_evaluations == [1, 2]
    assert args.n_scan_points == [3, 4]
    assert args.n_points == [5, 6]
    assert args.plot is True
    assert args.dry_run is True
    assert args.fail_fast is True
    assert args.repeat == 2
    assert args.timeout_seconds == 9.5


def test_selected_benchmarks_all_group_and_exclusions() -> None:
    all_specs = benchmark.selected_benchmarks(make_args())
    assert len(all_specs) == len(benchmark.BENCHMARKS)

    pyhs3_specs = benchmark.selected_benchmarks(make_args(groups=["pyhs3"]))
    assert pyhs3_specs
    assert {item.group for item in pyhs3_specs} == {"pyhs3"}

    selected = benchmark.selected_benchmarks(
        make_args(
            benchmarks=["workspace_loading", "nll_scan", "benchmark_overview"],
            groups=["pyhs3", "overview"],
            exclude_benchmarks=["nll_scan"],
        )
    )
    assert [item.name for item in selected] == [
        "workspace_loading",
        "benchmark_overview",
    ]


def test_discover_workspaces_filters_regex_exclude_limit_and_explicit(
    tmp_path: Path,
) -> None:
    names = ["1ch_keep.json", "2ch_skip.json", "notes.txt", "3ch_keep.root"]
    for name in names:
        (tmp_path / name).write_text("{}")

    args = make_args(
        workspace_dir=tmp_path,
        workspace_glob="*",
        workspace_regex="ch_",
        exclude_workspaces=["2ch*"],
        limit=1,
    )
    assert benchmark.discover_workspaces(args) == [tmp_path / "1ch_keep.json"]

    explicit = [tmp_path / "notes.txt", tmp_path / "1ch_keep.json"]
    assert benchmark.discover_workspaces(make_args(workspaces=explicit)) == [
        tmp_path / "1ch_keep.json"
    ]


def test_paired_root_path_default_and_custom_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "case.json"
    workspace.write_text("{}")
    assert benchmark.paired_root_path(workspace, make_args()) is None

    default_root = tmp_path / "case.root"
    default_root.write_text("root")
    assert benchmark.paired_root_path(workspace, make_args()) == default_root

    custom_dir = tmp_path / "root"
    custom_dir.mkdir()
    custom_root = custom_dir / "case.root"
    custom_root.write_text("root")
    assert (
        benchmark.paired_root_path(workspace, make_args(root_workspace_dir=custom_dir))
        == custom_root
    )


def test_output_path_helpers_create_directories(tmp_path: Path) -> None:
    args = make_args(output_dir=tmp_path / "results", plot_dir=tmp_path / "plots")
    output_dir, plot_dir, output_name = benchmark.make_output_paths(
        args, "demo", tmp_path / "workspace.json", 12
    )
    assert output_dir == tmp_path / "results" / "demo" / "workspace" / "repeat_012"
    assert plot_dir == tmp_path / "plots" / "demo" / "workspace" / "repeat_012"
    assert output_name == "demo_result.json"
    assert output_dir.is_dir()
    assert plot_dir.is_dir()

    output_dir, plot_dir, output_name = benchmark.make_batch_output_paths(
        args, "demo", 2
    )
    assert output_dir == tmp_path / "results" / "demo" / "global" / "repeat_002"
    assert plot_dir == tmp_path / "plots" / "demo"
    assert output_name == "demo_result.json"


def test_base_command() -> None:
    assert benchmark.base_command("src.demo") == [sys.executable, "-m", "src.demo"]


@pytest.mark.parametrize(
    ("name", "expected_flags"),
    [
        ("workspace_loading", ["--workspaces", "--n-runs"]),
        ("model_creation", ["--targets", "--modes", "--n-runs"]),
        ("compiled_evaluation", ["--n-evaluations"]),
        ("pdf_evaluation", ["--n-evaluations", "--distribution"]),
        (
            "nll_scan",
            ["--scan-parameter", "--scan-min", "--scan-max", "--n-scan-points"],
        ),
        (
            "memory_scaling",
            ["--n-runs", "--n-evaluations", "--distribution", "--stages"],
        ),
        ("model_complexity_scaling", ["--report-dir", "--stages"]),
        ("graph_canonicalization", ["--n-runs"]),
        ("graph_optimization", ["--n-runs"]),
    ],
)
def test_command_for_multi_workspace_variants(
    name: str, expected_flags: list[str], tmp_path: Path
) -> None:
    args = make_args(plot=True, n_evaluations=[10, 20], n_scan_points=[11, 12])
    command = benchmark.command_for_multi_workspace(
        spec(name),
        tmp_path / "case.json",
        args,
        tmp_path / "out",
        tmp_path / "plots",
        "out.json",
    )
    assert command[:3] == [sys.executable, "-m", spec(name).module]
    assert "--output-dir" in command
    assert "--output-name" in command
    assert "--plot" in command
    assert "--plot-dir" in command
    for flag in expected_flags:
        assert flag in command
    if name == "workspace_loading":
        assert "--targets" not in command
        assert "--modes" not in command


def test_command_for_multi_workspace_batch_requires_workspaces_and_accepts_many(
    tmp_path: Path,
) -> None:
    args = make_args(plot=True)
    with pytest.raises(ValueError, match="at least one workspace"):
        benchmark.command_for_multi_workspace_batch(
            spec("workspace_loading"),
            [],
            args,
            tmp_path / "out",
            tmp_path / "plots",
            "x.json",
        )

    workspaces = [tmp_path / "a.json", tmp_path / "b.json"]
    command = benchmark.command_for_multi_workspace_batch(
        spec("workspace_loading"),
        workspaces,
        args,
        tmp_path / "out",
        tmp_path / "plots",
        "x.json",
    )
    assert command.count("--workspaces") == 1
    assert str(workspaces[0]) in command
    assert str(workspaces[1]) in command
    assert "--plot" in command


def test_command_for_single_workspace_supported_and_unsupported(tmp_path: Path) -> None:
    args = make_args(frameworks=["pyhs3"], plot=True, fail_fast=True, n_points=[17])
    workspace = tmp_path / "case.json"

    binned = benchmark.command_for_single_workspace(
        spec("cross_binned_likelihood_evaluation"),
        workspace,
        args,
        tmp_path / "out",
        tmp_path / "plots",
        "result.json",
    )
    assert binned[:3] == [
        sys.executable,
        "-m",
        "src.run_cross_binned_likelihood_evaluation",
    ]
    assert "--workspace" in binned
    assert "--frameworks" in binned
    assert "--fail-fast" in binned
    assert "--plot" in binned

    nll = benchmark.command_for_single_workspace(
        spec("cross_nll_scan"),
        workspace,
        args,
        tmp_path / "out",
        tmp_path / "plots",
        "result.json",
    )
    assert nll[:3] == [sys.executable, "-m", "src.run_cross_nll_scan"]
    assert "--n-points" in nll
    assert "17" in nll
    assert "--output" in nll

    bad_spec = benchmark.BenchmarkSpec(
        "bad", "cross", "single_workspace", "src.bad", True
    )
    with pytest.raises(ValueError, match="Unsupported single-workspace benchmark"):
        benchmark.command_for_single_workspace(
            bad_spec,
            workspace,
            args,
            tmp_path / "out",
            tmp_path / "plots",
            "result.json",
        )


def test_command_for_json_root_pair_includes_optional_fields(tmp_path: Path) -> None:
    args = make_args(
        plot=True,
        pyhs3_data_name="combData_ch0",
        xroofit_model_name="sim_pdf",
        n_points=[13],
        n_runs=5,
    )
    command = benchmark.command_for_json_root_pair(
        spec("pyhs3_xroofit_benchmark"),
        tmp_path / "case.json",
        tmp_path / "case.root",
        args,
        tmp_path / "out",
        tmp_path / "plots",
        "result.json",
    )
    assert command[:3] == [sys.executable, "-m", "src.run_pyhs3_xroofit_benchmark"]
    assert "--json-workspace" in command
    assert "--root-workspace" in command
    assert "--pyhs3-data-name" in command
    assert "combData_ch0" in command
    assert "--xroofit-model-name" in command
    assert "sim_pdf" in command
    assert "--plot" in command


@pytest.mark.parametrize(
    "name",
    [
        "cross_model_complexity_scaling",
        "cross_scalar_pdf_evaluation",
        "cross_vectorized_pdf_evaluation",
        "benchmark_overview",
    ],
)
def test_command_for_run_once_variants(name: str, tmp_path: Path) -> None:
    args = make_args(
        workspace_dir=tmp_path / "inputs",
        root_workspace_dir=tmp_path / "roots",
        plot=True,
        scalar_frameworks=["pyhs3"],
        scenarios=["small"],
        fail_fast=True,
        n_points=[9],
        n_evaluations=[3, 4],
    )
    command = benchmark.command_for_run_once(
        spec(name), args, tmp_path / "out", tmp_path / "plots", "result.json"
    )
    assert command[:3] == [sys.executable, "-m", spec(name).module]
    if name == "benchmark_overview":
        assert "--include-failed" in command
        assert command.count("--plot") == 0
    else:
        assert "--plot" in command
        assert "--plot-dir" in command
    if name == "cross_scalar_pdf_evaluation":
        assert "--frameworks" in command
        assert "root" in command
        assert "--n-evaluations" in command
    if name == "cross_vectorized_pdf_evaluation":
        assert "--scenarios" in command
        assert "small" in command
    if name == "cross_model_complexity_scaling":
        assert "--json-input-dir" in command
        assert "--root-input-dir" in command
        assert "--fail-fast" in command


def test_command_for_run_once_rejects_unknown(tmp_path: Path) -> None:
    bad_spec = benchmark.BenchmarkSpec(
        "unknown", "other", "other", "src.unknown", False, run_once=True
    )
    with pytest.raises(ValueError, match="Unsupported run-once benchmark"):
        benchmark.command_for_run_once(
            bad_spec, make_args(), tmp_path / "out", tmp_path / "plots", "result.json"
        )


def test_run_command_dry_run(tmp_path: Path) -> None:
    record = benchmark.run_command(
        command=["python", "-V"],
        spec=spec("workspace_loading"),
        workspace=tmp_path / "case.json",
        root_workspace=None,
        output_dir=tmp_path,
        timeout_seconds=None,
        dry_run=True,
    )
    assert record.status == "dry_run"
    assert record.returncode is None
    assert record.duration_seconds == 0.0
    assert record.workspace == str(tmp_path / "case.json")


def test_run_command_success_and_failure_write_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Any, Any]] = []

    def fake_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="ok", stderr=""
        )

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([1.0, 2.5]).__next__)
    record = benchmark.run_command(
        command=["cmd"],
        spec=spec("workspace_loading"),
        workspace=None,
        root_workspace=None,
        output_dir=tmp_path,
        timeout_seconds=12.0,
        dry_run=False,
    )
    assert record.status == "success"
    assert record.returncode == 0
    assert record.duration_seconds == pytest.approx(1.5)
    assert (tmp_path / "stdout.txt").read_text() == "ok"
    assert calls[0][1]["timeout"] == 12.0
    assert calls[0][1]["check"] is False

    def failing_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0], returncode=7, stdout="", stderr="x" * 5000
        )

    monkeypatch.setattr(benchmark.subprocess, "run", failing_run)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([3.0, 4.0]).__next__)
    record = benchmark.run_command(
        command=["cmd"],
        spec=spec("workspace_loading"),
        workspace=None,
        root_workspace=None,
        output_dir=tmp_path,
        timeout_seconds=None,
        dry_run=False,
    )
    assert record.status == "failed"
    assert record.returncode == 7
    assert record.error == "x" * 4000
    assert (tmp_path / "stderr.txt").read_text() == "x" * 5000


def test_run_command_timeout_and_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def timeout_run(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="cmd", timeout=1.0)

    monkeypatch.setattr(benchmark.subprocess, "run", timeout_run)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([0.0, 2.0]).__next__)
    record = benchmark.run_command(
        command=["cmd"],
        spec=spec("workspace_loading"),
        workspace=None,
        root_workspace=None,
        output_dir=tmp_path,
        timeout_seconds=1.0,
        dry_run=False,
    )
    assert record.status == "timeout"
    assert record.duration_seconds == pytest.approx(2.0)
    assert "timed out" in record.error

    def error_run(*args: Any, **kwargs: Any) -> None:
        raise OSError("no executable")

    monkeypatch.setattr(benchmark.subprocess, "run", error_run)
    monkeypatch.setattr(benchmark.time, "perf_counter", iter([2.0, 2.25]).__next__)
    record = benchmark.run_command(
        command=["cmd"],
        spec=spec("workspace_loading"),
        workspace=None,
        root_workspace=None,
        output_dir=tmp_path,
        timeout_seconds=None,
        dry_run=False,
    )
    assert record.status == "error"
    assert record.error == "OSError('no executable')"
    assert "no executable" in (tmp_path / "stderr.txt").read_text()


def test_write_summary_counts_and_failed_summary(tmp_path: Path) -> None:
    args = make_args(output_dir=tmp_path, report_name="summary.json")
    records = [
        benchmark.RunRecord(
            "ok", "pyhs3", None, None, ["ok"], "success", 0, 1.0, "out", "err"
        ),
        benchmark.RunRecord(
            "bad",
            "pyhs3",
            "a.json",
            None,
            ["bad"],
            "failed",
            1,
            2.0,
            "out",
            "err",
            "boom",
        ),
        benchmark.RunRecord(
            "slow",
            "pyhs3",
            None,
            None,
            ["slow"],
            "timeout",
            None,
            3.0,
            "out",
            "err",
            "timeout",
        ),
        benchmark.RunRecord(
            "oops",
            "pyhs3",
            None,
            None,
            ["oops"],
            "error",
            None,
            4.0,
            "out",
            "err",
            "error",
        ),
        benchmark.RunRecord(
            "dry", "pyhs3", None, None, ["dry"], "dry_run", None, 0.0, "out", "err"
        ),
        benchmark.RunRecord(
            "skip", "pyhs3", None, None, [], "skipped", None, 0.0, "out", "err"
        ),
    ]
    benchmark.write_summary(args, records)
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["total"] == 6
    assert payload["success"] == 1
    assert payload["failed"] == 1
    assert payload["timeout"] == 1
    assert payload["error"] == 1
    assert payload["dry_run"] == 1
    assert payload["skipped"] == 1
    failed_summary = (tmp_path / "failed_summary.txt").read_text()
    assert "benchmark: bad" in failed_summary
    assert "benchmark: slow" in failed_summary
    assert "benchmark: oops" in failed_summary
    assert "benchmark: ok" not in failed_summary


def test_main_multi_workspace_plot_uses_batch_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ws1 = tmp_path / "a.json"
    ws2 = tmp_path / "b.json"
    ws1.write_text("{}")
    ws2.write_text("{}")
    args = make_args(
        benchmarks=["workspace_loading"],
        workspaces=[ws1, ws2],
        plot=True,
        dry_run=True,
        output_dir=tmp_path / "results",
        plot_dir=tmp_path / "plots",
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    benchmark.main()
    payload = json.loads((args.output_dir / args.report_name).read_text())
    assert payload["total"] == 1
    record = payload["records"][0]
    assert record["workspace"] is None
    assert record["status"] == "dry_run"
    assert str(ws1) in record["command"]
    assert str(ws2) in record["command"]


def test_main_non_plot_runs_each_workspace_and_skips_missing_root_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ws1 = tmp_path / "a.json"
    ws2 = tmp_path / "b.json"
    ws1.write_text("{}")
    ws2.write_text("{}")
    (tmp_path / "a.root").write_text("root")
    args = make_args(
        benchmarks=["pyhs3_xroofit_benchmark"],
        workspaces=[ws1, ws2],
        dry_run=True,
        output_dir=tmp_path / "results",
        plot_dir=tmp_path / "plots",
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    benchmark.main()
    payload = json.loads((args.output_dir / args.report_name).read_text())
    assert payload["total"] == 2
    statuses = [record["status"] for record in payload["records"]]
    assert statuses == ["dry_run", "skipped"]
    assert payload["records"][0]["root_workspace"] == str(tmp_path / "a.root")
    assert payload["records"][1]["error"] == "Missing matching ROOT workspace."


def test_main_run_once_and_repeats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = make_args(
        benchmarks=["benchmark_overview"],
        dry_run=True,
        repeat=2,
        output_dir=tmp_path / "results",
        plot_dir=tmp_path / "plots",
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    benchmark.main()
    payload = json.loads((args.output_dir / args.report_name).read_text())
    assert payload["total"] == 2
    assert all(
        record["benchmark"] == "benchmark_overview" for record in payload["records"]
    )
    assert {
        Path(record["stdout_path"]).parent.name for record in payload["records"]
    } == {
        "repeat_000",
        "repeat_001",
    }


def test_main_fail_fast_stops_after_failed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ws1 = tmp_path / "a.json"
    ws2 = tmp_path / "b.json"
    ws1.write_text("{}")
    ws2.write_text("{}")
    args = make_args(
        benchmarks=["workspace_loading"],
        workspaces=[ws1, ws2],
        fail_fast=True,
        output_dir=tmp_path / "results",
        plot_dir=tmp_path / "plots",
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)

    def fake_run_command(**kwargs: Any) -> benchmark.RunRecord:
        output_dir = kwargs["output_dir"]
        return benchmark.RunRecord(
            benchmark=kwargs["spec"].name,
            group=kwargs["spec"].group,
            workspace=str(kwargs["workspace"]) if kwargs["workspace"] else None,
            root_workspace=None,
            command=kwargs["command"],
            status="failed",
            returncode=7,
            duration_seconds=0.1,
            stdout_path=str(output_dir / "stdout.txt"),
            stderr_path=str(output_dir / "stderr.txt"),
            error="boom",
        )

    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    with pytest.raises(SystemExit) as exc_info:
        benchmark.main()
    assert exc_info.value.code == 1
    payload = json.loads((args.output_dir / args.report_name).read_text())
    assert payload["total"] == 1
    assert payload["failed"] == 1


def test_main_raises_for_unknown_benchmark_kind(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "a.json"
    workspace.write_text("{}")
    bad_spec = benchmark.BenchmarkSpec(
        name="bad",
        group="pyhs3",
        kind="mystery",
        module="src.bad",
        uses_workspace_matrix=True,
    )
    args = make_args(
        workspaces=[workspace],
        output_dir=tmp_path / "results",
        plot_dir=tmp_path / "plots",
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    monkeypatch.setattr(benchmark, "selected_benchmarks", lambda _args: [bad_spec])
    with pytest.raises(ValueError, match="Unsupported benchmark kind"):
        benchmark.main()


def test_command_for_multi_workspace_batch_covers_all_optional_branches(
    tmp_path: Path,
) -> None:
    args = make_args(
        plot=True,
        n_runs=6,
        n_evaluations=[7, 8],
        n_scan_points=[9, 10],
        stages=["workspace_loading", "nll_scan"],
        distribution="custom_dist",
    )
    workspaces = [tmp_path / "a.json", tmp_path / "b.json"]

    model_creation = benchmark.command_for_multi_workspace_batch(
        spec("model_creation"),
        workspaces,
        args,
        tmp_path / "out",
        tmp_path / "plots",
        "result.json",
    )
    assert "--targets" in model_creation
    assert "--modes" in model_creation
    assert "--n-runs" in model_creation

    compiled = benchmark.command_for_multi_workspace_batch(
        spec("compiled_evaluation"),
        workspaces,
        args,
        tmp_path / "out",
        tmp_path / "plots",
        "result.json",
    )
    assert "--targets" in compiled
    assert "--modes" in compiled
    assert "--n-evaluations" in compiled
    assert "7" in compiled and "8" in compiled

    pdf = benchmark.command_for_multi_workspace_batch(
        spec("pdf_evaluation"),
        workspaces,
        args,
        tmp_path / "out",
        tmp_path / "plots",
        "result.json",
    )
    assert "--distribution" in pdf
    assert "custom_dist" in pdf

    nll = benchmark.command_for_multi_workspace_batch(
        spec("nll_scan"),
        workspaces,
        args,
        tmp_path / "out",
        tmp_path / "plots",
        "result.json",
    )
    assert "--scan-parameter" in nll
    assert "--scan-min" in nll
    assert "--scan-max" in nll
    assert "--n-scan-points" in nll
    assert "9" in nll and "10" in nll

    memory = benchmark.command_for_multi_workspace_batch(
        spec("memory_scaling"),
        workspaces,
        args,
        tmp_path / "out",
        tmp_path / "plots",
        "result.json",
    )
    assert "--n-runs" in memory
    assert "--n-evaluations" in memory
    assert "--distribution" in memory
    assert "--scan-parameter" in memory
    assert "--n-scan-points" in memory
    assert memory[memory.index("--n-scan-points") + 1] == "9"
    assert "--stages" in memory

    complexity = benchmark.command_for_multi_workspace_batch(
        spec("model_complexity_scaling"),
        workspaces,
        args,
        tmp_path / "out",
        tmp_path / "plots",
        "result.json",
    )
    assert "--report-dir" in complexity
    assert str(tmp_path / "out" / "reports") in complexity


def test_main_batch_fail_fast_stops_after_failed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ws1 = tmp_path / "a.json"
    ws2 = tmp_path / "b.json"
    ws1.write_text("{}")
    ws2.write_text("{}")
    args = make_args(
        benchmarks=["workspace_loading"],
        workspaces=[ws1, ws2],
        plot=True,
        fail_fast=True,
        output_dir=tmp_path / "results",
        plot_dir=tmp_path / "plots",
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)

    def fake_run_command(**kwargs: Any) -> benchmark.RunRecord:
        output_dir = kwargs["output_dir"]
        return benchmark.RunRecord(
            benchmark=kwargs["spec"].name,
            group=kwargs["spec"].group,
            workspace=None,
            root_workspace=None,
            command=kwargs["command"],
            status="failed",
            returncode=1,
            duration_seconds=0.2,
            stdout_path=str(output_dir / "stdout.txt"),
            stderr_path=str(output_dir / "stderr.txt"),
            error="batch failed",
        )

    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    with pytest.raises(SystemExit) as exc_info:
        benchmark.main()
    assert exc_info.value.code == 1
    payload = json.loads((args.output_dir / args.report_name).read_text())
    assert payload["total"] == 1
    assert payload["records"][0]["workspace"] is None
    assert payload["failed"] == 1


def test_main_run_once_fail_fast_stops_after_failed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    args = make_args(
        benchmarks=["benchmark_overview"],
        fail_fast=True,
        output_dir=tmp_path / "results",
        plot_dir=tmp_path / "plots",
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)

    def fake_run_command(**kwargs: Any) -> benchmark.RunRecord:
        output_dir = kwargs["output_dir"]
        return benchmark.RunRecord(
            benchmark=kwargs["spec"].name,
            group=kwargs["spec"].group,
            workspace=None,
            root_workspace=None,
            command=kwargs["command"],
            status="error",
            returncode=None,
            duration_seconds=0.2,
            stdout_path=str(output_dir / "stdout.txt"),
            stderr_path=str(output_dir / "stderr.txt"),
            error="overview failed",
        )

    monkeypatch.setattr(benchmark, "run_command", fake_run_command)
    with pytest.raises(SystemExit) as exc_info:
        benchmark.main()
    assert exc_info.value.code == 1
    payload = json.loads((args.output_dir / args.report_name).read_text())
    assert payload["total"] == 1
    assert payload["error"] == 1


def test_main_single_workspace_branch_runs_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "a.json"
    workspace.write_text("{}")
    args = make_args(
        benchmarks=["cross_binned_likelihood_evaluation"],
        workspaces=[workspace],
        dry_run=True,
        frameworks=["pyhs3"],
        output_dir=tmp_path / "results",
        plot_dir=tmp_path / "plots",
    )
    monkeypatch.setattr(benchmark, "parse_args", lambda: args)
    benchmark.main()
    payload = json.loads((args.output_dir / args.report_name).read_text())
    assert payload["total"] == 1
    record = payload["records"][0]
    assert record["benchmark"] == "cross_binned_likelihood_evaluation"
    assert record["workspace"] == str(workspace)
    assert "--workspace" in record["command"]


def test_module_main_guard_runs_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "a.json"
    workspace.write_text("{}")
    summary_dir = tmp_path / "results"
    plot_dir = tmp_path / "plots"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_benchmarks.py",
            "--benchmarks",
            "workspace_loading",
            "--workspaces",
            str(workspace),
            "--output-dir",
            str(summary_dir),
            "--plot-dir",
            str(plot_dir),
            "--dry-run",
        ],
    )

    import runpy

    runpy.run_module("src.run_all_benchmarks", run_name="__main__")

    payload = json.loads((summary_dir / "matrix_summary.json").read_text())
    assert payload["dry_run"] == 1
    assert payload["total"] == 1
