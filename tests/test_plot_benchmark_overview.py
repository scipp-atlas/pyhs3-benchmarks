from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src import plot_benchmark_overview as benchmark


@pytest.fixture
def sample_result() -> dict[str, Any]:
    return {
        "benchmark": "model_complexity_scaling",
        "workspace": "simple_workspace.json",
        "target": "L_ch0",
        "mode": "FAST_RUN",
        "status": "success",
        "n_runs": 3,
        "n_evaluations": 10,
        "n_scan_points": 5,
        "total_setup_time_seconds": 0.123,
        "total_peak_rss_delta_mb": 4.5,
        "workspace_loading_wall_time_seconds_mean": 0.001,
        "model_creation_wall_time_seconds_mean": 0.002,
        "log_prob_construction_wall_time_seconds_mean": 0.003,
        "log_prob_compilation_wall_time_seconds_mean": 0.004,
        "workspace_loading_peak_rss_delta_mb": 1.0,
        "model_creation_peak_rss_delta_mb": 2.0,
        "log_prob_construction_peak_rss_delta_mb": 3.0,
        "log_prob_compilation_peak_rss_delta_mb": 4.0,
    }


@pytest.fixture
def overview_records(sample_result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(sample_result),
        {
            "benchmark": "compiled_evaluation",
            "workspace": "simple_workspace.json",
            "target": "L_ch0",
            "mode": "FAST_RUN",
            "status": "success",
            "n_evaluations": 10,
            "average_runtime_ms_per_evaluation": 0.12,
            "average_runtime_seconds_per_evaluation": 0.00012,
            "peak_rss_delta_mb": 1.2,
        },
        {
            "benchmark": "pdf_evaluation",
            "workspace": "simple_workspace.json",
            "target": "L_ch0",
            "mode": "FAST_RUN",
            "status": "success",
            "n_evaluations": 10,
            "average_runtime_ms_per_evaluation": 0.15,
            "average_runtime_seconds_per_evaluation": 0.00015,
            "peak_rss_delta_mb": 1.5,
        },
        {
            "benchmark": "nll_scan",
            "workspace": "simple_workspace.json",
            "target": "L_ch0",
            "mode": "FAST_RUN",
            "status": "success",
            "n_scan_points": 5,
            "runtime_ms_per_scan_point": 0.5,
            "runtime_per_scan_point_seconds": 0.0005,
            "peak_rss_delta_mb": 2.5,
        },
        {
            "benchmark": "memory_scaling",
            "workspace": "simple_workspace.json",
            "target": "L_ch0",
            "mode": "FAST_RUN",
            "status": "failed",
            "total_setup_time_ms": 130.0,
            "total_setup_time_seconds": 0.13,
            "total_peak_rss_delta_mb": 6.0,
            "workspace_loading_time_ms": 1.0,
            "model_creation_time_ms": 2.0,
            "workspace_loading_peak_rss_delta_mb": 1.0,
            "model_creation_peak_rss_delta_mb": 2.0,
        },
    ]


def write_result_file(
    results_dir: Path, benchmark_name: str, results: list[dict[str, Any]]
) -> Path:
    result_dir = results_dir / benchmark_name
    result_dir.mkdir(parents=True, exist_ok=True)
    path = result_dir / f"{benchmark_name}_result.json"
    path.write_text(json.dumps({"benchmark": benchmark_name, "results": results}))
    return path


def test_resolve_plots_all_returns_default() -> None:
    assert benchmark.resolve_plots(["all"]) == benchmark.DEFAULT_PLOTS


def test_resolve_plots_rejects_all_combined() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        benchmark.resolve_plots(["all", "diagnostics"])


def test_resolve_plots_rejects_unknown_plot() -> None:
    with pytest.raises(ValueError, match="Unknown plots"):
        benchmark.resolve_plots(["unknown"])


def test_resolve_plots_custom_selection() -> None:
    assert benchmark.resolve_plots(["diagnostics", "scan_summary"]) == [
        "diagnostics",
        "scan_summary",
    ]


def test_load_json(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text('{"a": 1}')
    assert benchmark.load_json(path) == {"a": 1}


def test_iter_result_files(tmp_path: Path) -> None:
    first = write_result_file(tmp_path, "a", [])
    second = write_result_file(tmp_path, "b", [])
    assert benchmark.iter_result_files(tmp_path) == [first, second]


def test_normalize_benchmark_name_known_and_unknown() -> None:
    assert benchmark.normalize_benchmark_name("model_creation") == "Model creation"
    assert benchmark.normalize_benchmark_name("custom_benchmark") == "Custom Benchmark"


def test_compact_workspace_name_known_unknown_and_none() -> None:
    assert benchmark.compact_workspace_name("simple_workspace.json") == "Simple"
    assert benchmark.compact_workspace_name("my_workspace.json") == "my\nworkspace"
    assert benchmark.compact_workspace_name(None) == "Unknown"


def test_get_workspace_label_with_and_without_target() -> None:
    assert (
        benchmark.get_workspace_label(
            {"workspace": "simple_workspace.json", "target": "L_ch0"}
        )
        == "Simple\nL_ch0"
    )
    assert (
        benchmark.get_workspace_label({"workspace": "simple_workspace.json"})
        == "Simple"
    )


def test_extract_results_accepts_only_lists() -> None:
    assert benchmark.extract_results({"results": [{"a": 1}]}) == [{"a": 1}]
    assert benchmark.extract_results({"results": {"a": 1}}) == []
    assert benchmark.extract_results({}) == []


@pytest.mark.parametrize("value", [None, "abc", float("nan"), float("inf")])
def test_maybe_to_float_rejects_invalid_values(value: Any) -> None:
    assert benchmark.maybe_to_float(value) is None


def test_maybe_to_float_success() -> None:
    assert benchmark.maybe_to_float("1.25") == 1.25


def test_collect_overview_records(
    tmp_path: Path, sample_result: dict[str, Any]
) -> None:
    write_result_file(tmp_path, "model_complexity_scaling", [sample_result])

    records, skipped_items = benchmark.collect_overview_records(tmp_path)

    assert skipped_items == []
    assert len(records) == 1
    record = records[0]
    assert record["benchmark"] == "model_complexity_scaling"
    assert record["benchmark_label"] == "Model complexity"
    assert record["workspace_label"] == "Simple\nL_ch0"
    assert record["total_setup_time_ms"] == pytest.approx(123.0)
    assert record["workspace_loading_time_ms"] == pytest.approx(1.0)
    assert record["model_creation_peak_rss_delta_mb"] == pytest.approx(2.0)


def test_collect_overview_records_ignores_non_list_results(tmp_path: Path) -> None:
    result_dir = tmp_path / "bad"
    result_dir.mkdir()
    (result_dir / "bad_result.json").write_text(
        json.dumps({"benchmark": "bad", "results": {"x": 1}})
    )
    records, skipped_items = benchmark.collect_overview_records(tmp_path)
    assert records == []
    assert skipped_items == []


def test_values_match() -> None:
    assert benchmark.values_match("a", None)
    assert benchmark.values_match("a", ["a", "b"])
    assert not benchmark.values_match("c", ["a", "b"])
    assert not benchmark.values_match(None, ["a"])


def test_numeric_value_matches() -> None:
    assert benchmark.numeric_value_matches("10", [10])
    assert benchmark.numeric_value_matches(10, [10])
    assert not benchmark.numeric_value_matches("bad", [10])
    assert not benchmark.numeric_value_matches(None, [10])
    assert benchmark.numeric_value_matches(None, None)


def test_filter_records_successful_only_and_filters(
    overview_records: list[dict[str, Any]],
) -> None:
    filtered = benchmark.filter_records(
        records=overview_records,
        benchmarks=["compiled_evaluation"],
        workspaces=["simple_workspace.json"],
        targets=["L_ch0"],
        modes=["FAST_RUN"],
        n_runs=None,
        n_evaluations=[10],
        n_scan_points=None,
        successful_only=True,
    )

    assert len(filtered) == 1
    assert filtered[0]["benchmark"] == "compiled_evaluation"


def test_filter_records_can_include_failed(
    overview_records: list[dict[str, Any]],
) -> None:
    filtered = benchmark.filter_records(
        records=overview_records,
        benchmarks=["memory_scaling"],
        workspaces=None,
        targets=None,
        modes=None,
        n_runs=None,
        n_evaluations=None,
        n_scan_points=None,
        successful_only=False,
    )
    assert len(filtered) == 1
    assert filtered[0]["status"] == "failed"


def test_has_metric() -> None:
    assert benchmark.has_metric([{"metric": 0}, {"metric": 2.0}], "metric")
    assert not benchmark.has_metric([{"metric": 0}, {"metric": None}], "metric")


def test_apply_cern_style_updates_rcparams() -> None:
    benchmark.apply_cern_style()
    assert plt.rcParams["figure.facecolor"] == "white"


def test_finalize_axes() -> None:
    fig, ax = plt.subplots()
    benchmark.finalize_axes(ax)
    assert not ax.spines["top"].get_visible()
    assert not ax.spines["right"].get_visible()
    plt.close(fig)


def test_save_figure_creates_file(tmp_path: Path) -> None:
    fig, _ax = plt.subplots()
    output_path = tmp_path / "nested" / "plot.png"
    benchmark.save_figure(fig, output_path)
    assert output_path.exists()


def test_annotate_horizontal_bars_adds_labels() -> None:
    fig, ax = plt.subplots()
    benchmark.annotate_horizontal_bars(ax, [1.0, 0.0, 2.0], "ms")
    assert len(ax.texts) == 2
    plt.close(fig)


def test_annotate_horizontal_bars_empty_values() -> None:
    fig, ax = plt.subplots()
    benchmark.annotate_horizontal_bars(ax, [], "ms")
    assert len(ax.texts) == 0
    plt.close(fig)


def test_make_ranked_horizontal_plot_creates_png(
    tmp_path: Path, overview_records: list[dict[str, Any]]
) -> None:
    output_path = tmp_path / "ranked.png"
    benchmark.make_ranked_horizontal_plot(
        records=overview_records,
        output_path=output_path,
        title="Ranked",
        metric_key="average_runtime_ms_per_evaluation",
        metric_label="ms",
        unit="ms",
        benchmark_filter={"compiled_evaluation", "pdf_evaluation"},
    )
    assert output_path.exists()


def test_make_ranked_horizontal_plot_no_records_returns(tmp_path: Path) -> None:
    output_path = tmp_path / "ranked.png"
    benchmark.make_ranked_horizontal_plot(
        [], output_path, "Ranked", "metric", "Metric", unit="ms"
    )
    assert not output_path.exists()


def test_aggregate_best_metric_by_workspace_uses_median() -> None:
    records = [
        {
            "benchmark": "compiled_evaluation",
            "workspace": "simple_workspace.json",
            "metric": 1.0,
        },
        {
            "benchmark": "compiled_evaluation",
            "workspace": "simple_workspace.json",
            "metric": 3.0,
        },
        {"benchmark": "other", "workspace": "simple_workspace.json", "metric": 99.0},
    ]
    rows = benchmark.aggregate_best_metric_by_workspace(
        records, "compiled_evaluation", "metric"
    )
    assert rows == [{"workspace": "Simple", "value": 2.0}]


def test_make_grouped_metric_plot_creates_png(
    tmp_path: Path, overview_records: list[dict[str, Any]]
) -> None:
    output_path = tmp_path / "grouped.png"
    benchmark.make_grouped_metric_plot(
        records=overview_records,
        output_path=output_path,
        title="Grouped",
        benchmark_metric_pairs=[
            ("compiled_evaluation", "average_runtime_ms_per_evaluation", "Compiled"),
            ("pdf_evaluation", "average_runtime_ms_per_evaluation", "PDF"),
        ],
        y_label="ms",
    )
    assert output_path.exists()


def test_make_grouped_metric_plot_no_rows_returns(tmp_path: Path) -> None:
    output_path = tmp_path / "grouped.png"
    benchmark.make_grouped_metric_plot(
        [], output_path, "Grouped", [("a", "b", "c")], "ms"
    )
    assert not output_path.exists()


def test_make_performance_summary_plot_creates_png(
    tmp_path: Path, overview_records: list[dict[str, Any]]
) -> None:
    benchmark.make_performance_summary_plot(overview_records, tmp_path)
    assert (tmp_path / "benchmark_overview_performance_summary.png").exists()


def test_make_performance_summary_plot_no_metrics_returns(tmp_path: Path) -> None:
    benchmark.make_performance_summary_plot([{"benchmark": "unknown"}], tmp_path)
    assert not (tmp_path / "benchmark_overview_performance_summary.png").exists()


def test_make_setup_evaluation_and_scan_summary_plots(
    tmp_path: Path, overview_records: list[dict[str, Any]]
) -> None:
    benchmark.make_setup_summary_plot(overview_records, tmp_path)
    benchmark.make_evaluation_summary_plot(overview_records, tmp_path)
    benchmark.make_scan_summary_plot(overview_records, tmp_path)

    assert (tmp_path / "benchmark_overview_setup_summary.png").exists()
    assert (tmp_path / "benchmark_overview_evaluation_summary.png").exists()
    assert (tmp_path / "benchmark_overview_scan_summary.png").exists()


def test_stage_rows_returns_median_rows(overview_records: list[dict[str, Any]]) -> None:
    rows = benchmark.stage_rows(overview_records, benchmark.STAGE_TIME_KEYS, "_time_ms")
    assert len(rows) == 1
    assert rows[0]["label"] == "Simple"
    assert any(value > 0 for value in rows[0]["values"])


def test_stage_rows_empty_when_no_stage_metrics() -> None:
    assert (
        benchmark.stage_rows(
            [{"benchmark": "compiled_evaluation"}],
            benchmark.STAGE_TIME_KEYS,
            "_time_ms",
        )
        == []
    )


@pytest.mark.parametrize(
    ("value", "total", "expected"),
    [
        (0.0, 10.0, ""),
        (1.0, 100.0, ""),
        (10.0, 100.0, "10.0\n10%"),
    ],
)
def test_format_segment_label(value: float, total: float, expected: str) -> None:
    assert benchmark.format_segment_label(value, total, "ms") == expected


def test_make_stacked_stage_plot_creates_png(
    tmp_path: Path, overview_records: list[dict[str, Any]]
) -> None:
    benchmark.make_stacked_stage_plot(
        records=overview_records,
        plot_dir=tmp_path,
        title="Stages",
        output_name="stages.png",
        y_label="Time [ms]",
        unit="ms",
        suffix="_time_ms",
        stage_keys=benchmark.STAGE_TIME_KEYS,
    )
    assert (tmp_path / "stages.png").exists()


def test_make_stacked_stage_plot_no_rows_returns(tmp_path: Path) -> None:
    benchmark.make_stacked_stage_plot(
        records=[],
        plot_dir=tmp_path,
        title="Stages",
        output_name="stages.png",
        y_label="Time [ms]",
        unit="ms",
        suffix="_time_ms",
        stage_keys=benchmark.STAGE_TIME_KEYS,
    )
    assert not (tmp_path / "stages.png").exists()


def test_make_stage_timing_and_memory_plots(
    tmp_path: Path, overview_records: list[dict[str, Any]]
) -> None:
    benchmark.make_stage_timing_plot(overview_records, tmp_path)
    benchmark.make_stage_memory_plot(overview_records, tmp_path)
    assert (tmp_path / "benchmark_overview_stage_timing.png").exists()
    assert (tmp_path / "benchmark_overview_stage_memory.png").exists()


def test_make_diagnostics_plot_creates_png(
    tmp_path: Path, overview_records: list[dict[str, Any]]
) -> None:
    benchmark.make_diagnostics_plot(overview_records, tmp_path)
    assert (tmp_path / "benchmark_overview_diagnostics_status.png").exists()


def test_make_diagnostics_plot_no_records_returns(tmp_path: Path) -> None:
    benchmark.make_diagnostics_plot([], tmp_path)
    assert not (tmp_path / "benchmark_overview_diagnostics_status.png").exists()


def test_parse_args_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["plot_benchmark_overview.py"])
    args = benchmark.parse_args()
    assert args.results_dir == benchmark.DEFAULT_RESULTS_DIR
    assert args.plot_dir == benchmark.DEFAULT_PLOT_DIR
    assert args.plots == ["all"]
    assert args.include_failed is False


def test_parse_args_custom_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_benchmark_overview.py",
            "--results-dir",
            "results/custom",
            "--plot-dir",
            "plots/custom",
            "--plots",
            "diagnostics",
            "scan_summary",
            "--benchmarks",
            "nll_scan",
            "--workspaces",
            "simple_workspace.json",
            "--targets",
            "L_ch0",
            "--modes",
            "FAST_RUN",
            "--n-runs",
            "3",
            "--n-evaluations",
            "10",
            "--n-scan-points",
            "5",
            "--include-failed",
        ],
    )
    args = benchmark.parse_args()
    assert args.results_dir == Path("results/custom")
    assert args.plot_dir == Path("plots/custom")
    assert args.plots == ["diagnostics", "scan_summary"]
    assert args.benchmarks == ["nll_scan"]
    assert args.workspaces == ["simple_workspace.json"]
    assert args.targets == ["L_ch0"]
    assert args.modes == ["FAST_RUN"]
    assert args.n_runs == [3]
    assert args.n_evaluations == [10]
    assert args.n_scan_points == [5]
    assert args.include_failed is True


def test_main_rejects_no_result_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_benchmark_overview.py",
            "--results-dir",
            str(tmp_path),
            "--plot-dir",
            str(tmp_path / "plots"),
        ],
    )
    with pytest.raises(ValueError, match="No benchmark results found"):
        benchmark.main()


def test_main_rejects_empty_after_filters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    write_result_file(
        tmp_path,
        "compiled_evaluation",
        [
            {
                "benchmark": "compiled_evaluation",
                "workspace": "simple_workspace.json",
                "target": "L_ch0",
                "mode": "FAST_RUN",
                "status": "success",
                "average_runtime_seconds_per_evaluation": 0.001,
            }
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_benchmark_overview.py",
            "--results-dir",
            str(tmp_path),
            "--plot-dir",
            str(tmp_path / "plots"),
            "--benchmarks",
            "nll_scan",
        ],
    )
    with pytest.raises(ValueError, match="No benchmark results remain"):
        benchmark.main()


def test_main_creates_default_overview_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_result: dict[str, Any],
) -> None:
    results_dir = tmp_path / "results"
    plot_dir = tmp_path / "plots"
    write_result_file(results_dir, "model_complexity_scaling", [sample_result])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_benchmark_overview.py",
            "--results-dir",
            str(results_dir),
            "--plot-dir",
            str(plot_dir),
        ],
    )

    benchmark.main()

    assert (plot_dir / "benchmark_overview_performance_summary.png").exists()
    assert (plot_dir / "benchmark_overview_stage_timing.png").exists()
    assert (plot_dir / "benchmark_overview_stage_memory.png").exists()


def test_main_creates_selected_plots_and_includes_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    plot_dir = tmp_path / "plots"
    write_result_file(
        results_dir,
        "nll_scan",
        [
            {
                "benchmark": "nll_scan",
                "workspace": "simple_workspace.json",
                "target": "L_ch0",
                "mode": "FAST_RUN",
                "status": "failed",
                "runtime_per_scan_point_seconds": 0.001,
                "n_scan_points": 5,
            }
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_benchmark_overview.py",
            "--results-dir",
            str(results_dir),
            "--plot-dir",
            str(plot_dir),
            "--plots",
            "scan_summary",
            "diagnostics",
            "--include-failed",
        ],
    )

    benchmark.main()

    assert (plot_dir / "benchmark_overview_scan_summary.png").exists()
    assert (plot_dir / "benchmark_overview_diagnostics_status.png").exists()


def test_collect_overview_records_converts_all_optional_metrics(tmp_path: Path) -> None:
    result_dir = tmp_path / "results" / "mixed_metrics"
    result_dir.mkdir(parents=True)

    payload = {
        "benchmark": "mixed_metrics",
        "results": [
            {
                "workspace": "simple_workspace.json",
                "target": "L_ch0",
                "mode": "FAST_RUN",
                "status": "success",
                "n_runs": "3",
                "n_evaluations": "10",
                "n_scan_points": "5",
                "wall_time_seconds_mean": 0.5,
                "average_runtime_seconds_per_evaluation": 0.01,
                "runtime_per_scan_point_seconds": 0.02,
                "total_runtime_seconds": 1.5,
                "total_setup_time_seconds": 2.0,
                "workspace_loading_wall_time_seconds_mean": 0.1,
                "model_creation_wall_time_seconds_mean": 0.2,
                "log_prob_construction_wall_time_seconds_mean": 0.3,
                "log_prob_compilation_wall_time_seconds_mean": 0.4,
                "compiled_evaluation_average_runtime_seconds_per_evaluation": 0.005,
                "pdf_evaluation_average_runtime_seconds_per_evaluation": 0.006,
                "nll_scan_runtime_per_scan_point_seconds": 0.007,
                "workspace_loading_peak_rss_delta_mb": 1.0,
                "model_creation_peak_rss_delta_mb": 2.0,
                "log_prob_construction_peak_rss_delta_mb": 3.0,
                "log_prob_compilation_peak_rss_delta_mb": 4.0,
                "compiled_evaluation_peak_rss_delta_mb": 5.0,
                "pdf_evaluation_peak_rss_delta_mb": 6.0,
                "nll_scan_peak_rss_delta_mb": 7.0,
            }
        ],
    }

    (result_dir / "mixed_metrics_result.json").write_text(json.dumps(payload))

    records, skipped_items = benchmark.collect_overview_records(tmp_path / "results")

    assert skipped_items == []
    assert len(records) == 1
    record = records[0]

    assert record["wall_time_ms"] == 500.0
    assert record["average_runtime_ms_per_evaluation"] == 10.0
    assert record["runtime_ms_per_scan_point"] == 20.0
    assert record["total_runtime_ms"] == 1500.0
    assert record["total_setup_time_ms"] == 2000.0
    assert record["workspace_loading_time_ms"] == 100.0
    assert record["model_creation_time_ms"] == 200.0
    assert record["log_prob_construction_time_ms"] == 300.0
    assert record["log_prob_compilation_time_ms"] == 400.0
    assert record["compiled_evaluation_time_ms"] == 5.0
    assert record["pdf_evaluation_time_ms"] == 6.0
    assert record["nll_scan_time_ms"] == 7.0
    assert record["workspace_loading_peak_rss_delta_mb"] == 1.0
    assert record["nll_scan_peak_rss_delta_mb"] == 7.0


@pytest.mark.parametrize(
    ("actual", "expected_values", "expected"),
    [
        (None, [1], False),
        ("abc", [1], False),
        ("5", [5], True),
        (5.0, [5], True),
        (5, [4], False),
        (5, None, True),
        (5, [], True),
    ],
)
def test_numeric_value_matches_corner_cases(
    actual: Any,
    expected_values: list[int] | None,
    expected: bool,
) -> None:
    assert benchmark.numeric_value_matches(actual, expected_values) is expected


def test_make_ranked_horizontal_plot_returns_for_zero_only_records(
    tmp_path: Path,
) -> None:
    records = [
        {
            "benchmark": "nll_scan",
            "workspace": "simple_workspace.json",
            "runtime_ms_per_scan_point": 0.0,
        }
    ]

    output_path = tmp_path / "ranked.png"

    benchmark.make_ranked_horizontal_plot(
        records=records,
        output_path=output_path,
        title="No non-zero data",
        metric_key="runtime_ms_per_scan_point",
        metric_label="Runtime",
        unit="ms",
        benchmark_filter={"nll_scan"},
    )

    assert not output_path.exists()


def test_main_creates_diagnostics_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    plot_dir = tmp_path / "plots"
    result_dir = results_dir / "workspace_loading"
    result_dir.mkdir(parents=True)

    payload = {
        "benchmark": "workspace_loading",
        "results": [
            {
                "workspace": "simple_workspace.json",
                "target": "L_ch0",
                "mode": "FAST_RUN",
                "status": "success",
                "n_runs": 1,
                "wall_time_seconds_mean": 0.1,
            }
        ],
    }

    (result_dir / "workspace_loading_result.json").write_text(json.dumps(payload))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_benchmark_overview.py",
            "--results-dir",
            str(results_dir),
            "--plot-dir",
            str(plot_dir),
            "--plots",
            "diagnostics",
        ],
    )

    benchmark.main()

    assert (plot_dir / "benchmark_overview_diagnostics_status.png").exists()


def test_load_json_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{")

    with pytest.raises(ValueError, match="Invalid JSON"):
        benchmark.load_json(path)


def test_load_json_rejects_non_object_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]))

    with pytest.raises(TypeError, match="Expected top-level JSON object"):
        benchmark.load_json(path)


def test_iter_result_files_rejects_missing_and_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Results directory does not exist"):
        benchmark.iter_result_files(tmp_path / "missing")

    file_path = tmp_path / "not_a_directory"
    file_path.write_text("x")
    with pytest.raises(NotADirectoryError, match="Results path is not a directory"):
        benchmark.iter_result_files(file_path)


def test_collect_overview_records_records_skipped_invalid_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_dir = tmp_path / "results" / "bad"
    result_dir.mkdir(parents=True)
    bad_file = result_dir / "bad_result.json"
    bad_file.write_text("{")

    records, skipped_items = benchmark.collect_overview_records(tmp_path / "results")

    output = capsys.readouterr().out
    assert records == []
    assert len(skipped_items) == 1
    assert skipped_items[0]["path"] == str(bad_file)
    assert skipped_items[0]["error_type"] == "ValueError"
    assert "Skipping result file" in output


def test_collect_overview_records_strict_reraises_invalid_file(tmp_path: Path) -> None:
    result_dir = tmp_path / "results" / "bad"
    result_dir.mkdir(parents=True)
    (result_dir / "bad_result.json").write_text("{")

    with pytest.raises(ValueError, match="Invalid JSON"):
        benchmark.collect_overview_records(tmp_path / "results", strict=True)


def test_collect_overview_records_uses_parent_directory_as_benchmark_when_missing(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "results" / "model_creation"
    result_dir.mkdir(parents=True)
    (result_dir / "model_creation_result.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "workspace": "simple_workspace.json",
                        "target": "L_ch0",
                        "status": "success",
                        "wall_time_seconds_mean": 0.1,
                    }
                ]
            }
        )
    )

    records, skipped_items = benchmark.collect_overview_records(tmp_path / "results")

    assert skipped_items == []
    assert len(records) == 1
    assert records[0]["benchmark"] == "model_creation"
    assert records[0]["benchmark_label"] == "Model creation"
    assert records[0]["wall_time_ms"] == pytest.approx(100.0)


def test_filter_records_rejects_all_filter_mismatches(
    overview_records: list[dict[str, Any]],
) -> None:
    assert (
        benchmark.filter_records(
            records=overview_records,
            benchmarks=["missing"],
            workspaces=None,
            targets=None,
            modes=None,
            n_runs=None,
            n_evaluations=None,
            n_scan_points=None,
            successful_only=False,
        )
        == []
    )
    assert (
        benchmark.filter_records(
            records=overview_records,
            benchmarks=None,
            workspaces=["missing.json"],
            targets=None,
            modes=None,
            n_runs=None,
            n_evaluations=None,
            n_scan_points=None,
            successful_only=False,
        )
        == []
    )
    assert (
        benchmark.filter_records(
            records=overview_records,
            benchmarks=None,
            workspaces=None,
            targets=["missing"],
            modes=None,
            n_runs=None,
            n_evaluations=None,
            n_scan_points=None,
            successful_only=False,
        )
        == []
    )
    assert (
        benchmark.filter_records(
            records=overview_records,
            benchmarks=None,
            workspaces=None,
            targets=None,
            modes=["missing"],
            n_runs=None,
            n_evaluations=None,
            n_scan_points=None,
            successful_only=False,
        )
        == []
    )
    assert (
        benchmark.filter_records(
            records=overview_records,
            benchmarks=None,
            workspaces=None,
            targets=None,
            modes=None,
            n_runs=[999],
            n_evaluations=None,
            n_scan_points=None,
            successful_only=False,
        )
        == []
    )
    assert (
        benchmark.filter_records(
            records=overview_records,
            benchmarks=None,
            workspaces=None,
            targets=None,
            modes=None,
            n_runs=None,
            n_evaluations=[999],
            n_scan_points=None,
            successful_only=False,
        )
        == []
    )
    assert (
        benchmark.filter_records(
            records=overview_records,
            benchmarks=None,
            workspaces=None,
            targets=None,
            modes=None,
            n_runs=None,
            n_evaluations=None,
            n_scan_points=[999],
            successful_only=False,
        )
        == []
    )


def test_save_figure_raises_when_file_is_not_created(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fig, _ax = plt.subplots()
    monkeypatch.setattr(fig, "savefig", lambda *args, **kwargs: None)

    with pytest.raises(FileNotFoundError, match="Plot file was not created"):
        benchmark.save_figure(fig, tmp_path / "missing.png")


def test_save_figure_wraps_oserror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fig, _ax = plt.subplots()

    def failing_savefig(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(fig, "savefig", failing_savefig)

    with pytest.raises(OSError, match="Could not save plot"):
        benchmark.save_figure(fig, tmp_path / "plot.png")


def test_has_metric_ignores_non_numeric_and_non_finite_values() -> None:
    assert not benchmark.has_metric(
        [
            {"metric": "not-a-number"},
            {"metric": float("nan")},
            {"metric": None},
        ],
        "metric",
    )


def test_aggregate_best_metric_by_workspace_ignores_missing_zero_and_other_benchmarks() -> (
    None
):
    records = [
        {
            "benchmark": "compiled_evaluation",
            "workspace": "simple_workspace.json",
            "metric": 0.0,
        },
        {"benchmark": "compiled_evaluation", "workspace": "simple_workspace.json"},
        {"benchmark": "other", "workspace": "simple_workspace.json", "metric": 2.0},
    ]
    assert (
        benchmark.aggregate_best_metric_by_workspace(
            records,
            "compiled_evaluation",
            "metric",
        )
        == []
    )


def test_make_grouped_metric_plot_skips_empty_series(tmp_path: Path) -> None:
    output_path = tmp_path / "grouped.png"
    benchmark.make_grouped_metric_plot(
        records=[
            {
                "benchmark": "compiled_evaluation",
                "workspace": "simple_workspace.json",
                "metric": 0.0,
            }
        ],
        output_path=output_path,
        title="Grouped",
        benchmark_metric_pairs=[("compiled_evaluation", "metric", "Compiled")],
        y_label="ms",
    )
    assert not output_path.exists()


def test_make_stacked_stage_plot_returns_for_zero_only_rows(tmp_path: Path) -> None:
    records = [
        {
            "benchmark": "model_complexity_scaling",
            "workspace": "simple_workspace.json",
            "workspace_loading_time_ms": 0.0,
        }
    ]
    benchmark.make_stacked_stage_plot(
        records=records,
        plot_dir=tmp_path,
        title="Stages",
        output_name="stages.png",
        y_label="Time [ms]",
        unit="ms",
        suffix="_time_ms",
        stage_keys=benchmark.STAGE_TIME_KEYS,
    )
    assert not (tmp_path / "stages.png").exists()


def test_run_plot_builder_success_and_non_strict_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert benchmark.run_plot_builder("ok", lambda: None, strict=True) is True

    def failing_builder() -> None:
        raise RuntimeError("plot failed")

    assert benchmark.run_plot_builder("bad", failing_builder, strict=False) is False
    assert "Skipping plot 'bad'" in capsys.readouterr().out


def test_run_plot_builder_strict_reraises() -> None:
    def failing_builder() -> None:
        raise RuntimeError("plot failed")

    with pytest.raises(RuntimeError, match="plot failed"):
        benchmark.run_plot_builder("bad", failing_builder, strict=True)


def test_main_strict_reraises_plot_builder_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_result: dict[str, Any],
) -> None:
    results_dir = tmp_path / "results"
    plot_dir = tmp_path / "plots"
    write_result_file(results_dir, "model_complexity_scaling", [sample_result])

    def failing_plot(records: list[dict[str, Any]], plot_dir: Path) -> None:
        raise RuntimeError("plot failed")

    monkeypatch.setattr(benchmark, "make_performance_summary_plot", failing_plot)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_benchmark_overview.py",
            "--results-dir",
            str(results_dir),
            "--plot-dir",
            str(plot_dir),
            "--plots",
            "performance_summary",
            "--strict",
        ],
    )

    with pytest.raises(RuntimeError, match="plot failed"):
        benchmark.main()


def test_main_non_strict_skips_plot_builder_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_dir = tmp_path / "results"
    plot_dir = tmp_path / "plots"
    write_result_file(results_dir, "model_complexity_scaling", [sample_result])

    def failing_plot(records: list[dict[str, Any]], plot_dir: Path) -> None:
        raise RuntimeError("plot failed")

    monkeypatch.setattr(benchmark, "make_performance_summary_plot", failing_plot)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_benchmark_overview.py",
            "--results-dir",
            str(results_dir),
            "--plot-dir",
            str(plot_dir),
            "--plots",
            "performance_summary",
        ],
    )

    benchmark.main()

    output = capsys.readouterr().out
    assert "Skipping plot 'performance_summary'" in output
    assert "Skipped plot builders" in output
    assert "performance_summary" in output


def test_main_reports_skipped_result_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    sample_result: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_dir = tmp_path / "results"
    plot_dir = tmp_path / "plots"
    write_result_file(results_dir, "model_complexity_scaling", [sample_result])
    bad_dir = results_dir / "bad"
    bad_dir.mkdir()
    (bad_dir / "bad_result.json").write_text("{")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_benchmark_overview.py",
            "--results-dir",
            str(results_dir),
            "--plot-dir",
            str(plot_dir),
            "--plots",
            "diagnostics",
        ],
    )

    benchmark.main()

    output = capsys.readouterr().out
    assert "Skipped result items     : 1" in output


def test_load_json_wraps_oserror_for_directory(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="Could not read result file"):
        benchmark.load_json(tmp_path)


class BadString:
    def __str__(self) -> str:
        raise ValueError("bad workspace")


def test_collect_overview_records_skips_malformed_result_row(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_file = write_result_file(tmp_path, "malformed", [])

    monkeypatch_payload = {
        "benchmark": "malformed",
        "results": [
            {
                "workspace": BadString(),
                "status": "success",
            }
        ],
    }

    original_load_json = benchmark.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        if path == result_file:
            return monkeypatch_payload
        return original_load_json(path)

    from pytest import MonkeyPatch

    mp = MonkeyPatch()
    mp.setattr(benchmark, "load_json", fake_load_json)
    try:
        records, skipped_items = benchmark.collect_overview_records(tmp_path)
    finally:
        mp.undo()

    output = capsys.readouterr().out
    assert records == []
    assert len(skipped_items) == 1
    assert skipped_items[0]["path"] == str(result_file)
    assert skipped_items[0]["error_type"] == "ValueError"
    assert "Could not normalize one result row" in skipped_items[0]["error_message"]
    assert "Skipping malformed result row" in output


def test_collect_overview_records_strict_reraises_malformed_result_row(
    tmp_path: Path,
) -> None:
    result_file = write_result_file(tmp_path, "malformed", [])

    monkeypatch_payload = {
        "benchmark": "malformed",
        "results": [
            {
                "workspace": BadString(),
                "status": "success",
            }
        ],
    }

    original_load_json = benchmark.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        if path == result_file:
            return monkeypatch_payload
        return original_load_json(path)

    from pytest import MonkeyPatch

    mp = MonkeyPatch()
    mp.setattr(benchmark, "load_json", fake_load_json)
    try:
        with pytest.raises(ValueError, match="bad workspace"):
            benchmark.collect_overview_records(tmp_path, strict=True)
    finally:
        mp.undo()


def test_module_main_guard_creates_diagnostics_plot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    plot_dir = tmp_path / "plots"
    write_result_file(
        results_dir,
        "workspace_loading",
        [
            {
                "workspace": "simple_workspace.json",
                "target": "L_ch0",
                "mode": "FAST_RUN",
                "status": "success",
                "n_runs": 1,
                "wall_time_seconds_mean": 0.1,
            }
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_benchmark_overview.py",
            "--results-dir",
            str(results_dir),
            "--plot-dir",
            str(plot_dir),
            "--plots",
            "diagnostics",
        ],
    )

    runpy.run_module("src.plot_benchmark_overview", run_name="__main__")

    assert (plot_dir / "benchmark_overview_diagnostics_status.png").exists()


def test_extra_compact_workspace_generated_and_unknown_none() -> None:
    assert benchmark.compact_workspace_name(None) == "Unknown"
    label = benchmark.compact_workspace_name(
        "5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json"
    )
    assert label.splitlines() == [
        "5ch",
        "RooExp / Generic",
        "npOn / constrGauss / yield10x",
    ]


def test_extra_workspace_from_result_path_case_and_none() -> None:
    assert (
        benchmark.workspace_from_result({}, {"workspace_path": "dir/a.json"})
        == "a.json"
    )
    assert benchmark.workspace_from_result({"json_path": "dir/b.json"}, {}) == "b.json"
    assert benchmark.workspace_from_result({}, {"case": "case-A"}) == "case-A"
    assert benchmark.workspace_from_result({}, {}) is None


def test_extra_framework_and_workspace_label_framework_only() -> None:
    assert benchmark.framework_from_result({"framework": "pyhs3"}) == "pyhs3"
    assert benchmark.framework_from_result({"framework_label": "PyHS3"}) == "PyHS3"
    assert (
        benchmark.get_workspace_label(
            {
                "workspace": "simple_workspace.json",
                "target": "L_ch0",
                "framework": "pyhs3",
            }
        )
        == "Simple\nL_ch0\npyhs3"
    )


def test_extra_flatten_nested_result_and_extract_results() -> None:
    payload = {"mode": "FAST_RUN"}
    result = {
        "workspace_path": "dir/workspace.json",
        "target": "L_ch0",
        "analysis_name": "analysis",
        "case": "case1",
        "pyhs3": {"status": "success", "metric": 1.0},
        "root": {"metric": 2.0},
    }
    flattened = benchmark.flatten_nested_result(payload, result)
    assert [row["framework"] for row in flattened] == ["pyhs3", "root"]
    assert flattened[0]["workspace"] == "workspace.json"
    assert flattened[0]["mode"] == "FAST_RUN"
    assert flattened[0]["analysis"] == "analysis"
    assert flattened[1]["status"] == "unknown"

    extracted = benchmark.extract_results({"results": ["bad", result, {"plain": True}]})
    assert len(extracted) == 3
    assert extracted[-1] == {"plain": True}


def test_extra_collect_overview_records_derives_warm_and_optional_timing_metrics(
    tmp_path: Path,
) -> None:
    result_dir = tmp_path / "results" / "model_creation"
    result_dir.mkdir(parents=True)
    (result_dir / "model_creation_result.json").write_text(
        json.dumps(
            {
                "results": [
                    {
                        "workspace": "simple_workspace.json",
                        "target": "L_ch0",
                        "status": "success",
                        "n_evaluations": 5,
                        "warm_total_seconds": 0.5,
                        "cold_first_evaluation_time_seconds": 0.25,
                        "warm_evaluation": {"mean_seconds": 0.125},
                    }
                ]
            }
        )
    )

    records, skipped_items = benchmark.collect_overview_records(tmp_path / "results")

    assert skipped_items == []
    assert len(records) == 1
    record = records[0]
    assert record["benchmark"] == "model_creation"
    assert record["average_runtime_seconds_per_evaluation"] == pytest.approx(0.1)
    assert record["average_runtime_ms_per_evaluation"] == pytest.approx(100.0)
    assert record["cold_first_evaluation_ms"] == pytest.approx(250.0)
    assert record["warm_evaluation_us"] == pytest.approx(125000.0)


def test_extra_make_ranked_horizontal_plot_includes_framework_label(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "ranked_framework.png"
    benchmark.make_ranked_horizontal_plot(
        records=[
            {
                "benchmark": "cross_nll_scan",
                "workspace": "simple_workspace.json",
                "framework": "pyhs3",
                "runtime_ms_per_scan_point": 1.0,
            }
        ],
        output_path=output_path,
        title="Ranked",
        metric_key="runtime_ms_per_scan_point",
        metric_label="ms",
        unit="ms",
        benchmark_filter={"cross_nll_scan"},
    )
    assert output_path.exists()


def test_extra_make_performance_summary_plot_cross_framework_panel(
    tmp_path: Path,
) -> None:
    records = [
        {
            "benchmark": "cross_nll_scan",
            "workspace": "simple_workspace.json",
            "framework": "roofit",
            "status": "success",
            "time_per_scan_point_us": 4.2,
        }
    ]
    benchmark.make_performance_summary_plot(records, tmp_path)
    assert (tmp_path / "benchmark_overview_performance_summary.png").exists()


def test_extra_make_cross_framework_summary_plot_creates_png(tmp_path: Path) -> None:
    records = [
        {
            "benchmark": "cross_scalar_pdf_evaluation",
            "workspace": "5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json",
            "framework": "pyhs3",
            "status": "success",
            "time_per_evaluation_us": 2.0,
        },
        {
            "benchmark": "cross_scalar_pdf_evaluation",
            "workspace": "5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json",
            "framework": "pyhs3",
            "status": "success",
            "time_per_evaluation_us": 1.5,
        },
        {
            "benchmark": "cross_scalar_pdf_evaluation",
            "workspace": "5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json",
            "framework": "root",
            "status": "success",
            "time_per_evaluation_us": 0.8,
        },
        {
            "benchmark": "cross_nll_scan",
            "workspace": "5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json",
            "framework": "roofit",
            "status": "success",
            "time_per_scan_point_us": 3.0,
        },
        {
            "benchmark": "pyhs3_xroofit_benchmark",
            "workspace": "5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json",
            "framework": "xroofit",
            "status": "success",
            "time_per_scan_point_us": 5.0,
        },
        {
            "benchmark": "cross_nll_scan",
            "workspace": "bad.json",
            "framework": "roofit",
            "status": "success",
            "time_per_scan_point_us": 0.0,
        },
    ]

    benchmark.make_cross_framework_summary_plot(records, tmp_path)

    assert (tmp_path / "benchmark_overview_cross_framework_summary.png").exists()


def test_extra_make_cross_framework_summary_plot_no_usable_rows_returns(
    tmp_path: Path,
) -> None:
    benchmark.make_cross_framework_summary_plot(
        [
            {
                "benchmark": "cross_scalar_pdf_evaluation",
                "workspace": "simple_workspace.json",
                "framework": "pyhs3",
                "status": "success",
                "time_per_evaluation_us": 0.0,
            }
        ],
        tmp_path,
    )
    assert not (tmp_path / "benchmark_overview_cross_framework_summary.png").exists()
