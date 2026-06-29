from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src import benchmark_stages as stages


@pytest.fixture
def workspace_path(tmp_path: Path) -> Path:
    return tmp_path / "workspace.json"


def test_resolve_stages_all_returns_available_stages() -> None:
    assert stages.resolve_stages(["all"], available_stages=["a", "b"]) == ["a", "b"]


def test_resolve_stages_preserves_requested_order() -> None:
    assert stages.resolve_stages(
        ["model_creation", "workspace_loading"],
        available_stages=["workspace_loading", "model_creation"],
    ) == ["model_creation", "workspace_loading"]


@pytest.mark.parametrize(
    ("selected", "available", "message"),
    [
        ([], ["a"], "At least one stage must be selected"),
        (["a"], [], "available_stages must not be empty"),
        (["all", "a"], ["a"], "--stages all cannot be combined"),
        (["missing"], ["a"], "Unknown stages"),
    ],
)
def test_resolve_stages_rejects_invalid_inputs(
    selected: list[str],
    available: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        stages.resolve_stages(selected, available_stages=available)


def test_build_stage_specs_for_all_stages(workspace_path: Path) -> None:
    specs = stages.build_stage_specs(
        selected_stages=["all"],
        workspace_path=workspace_path,
        target="L_ch0",
        mode="FAST_RUN",
        n_runs=5,
        n_evaluations=100,
        distribution="sig_ch0",
        scan_parameter="mu_sig",
        scan_min=0.0,
        scan_max=5.0,
        n_scan_points=101,
    )

    assert [
        stage_name for stage_name, _function, _args in specs
    ] == stages.WORKFLOW_STAGES

    args_by_stage = {stage_name: args for stage_name, _function, args in specs}
    assert args_by_stage["workspace_loading"] == (workspace_path, 5)
    assert args_by_stage["model_creation"] == (workspace_path, "L_ch0", "FAST_RUN", 5)
    assert args_by_stage["log_prob_construction"] == (
        workspace_path,
        "L_ch0",
        "FAST_RUN",
        5,
    )
    assert args_by_stage["log_prob_compilation"] == (
        workspace_path,
        "L_ch0",
        "FAST_RUN",
        5,
    )
    assert args_by_stage["compiled_evaluation"] == (
        workspace_path,
        "L_ch0",
        "FAST_RUN",
        100,
    )
    assert args_by_stage["pdf_evaluation"] == (
        workspace_path,
        "L_ch0",
        "FAST_RUN",
        "sig_ch0",
        100,
    )
    assert args_by_stage["nll_scan"] == (
        workspace_path,
        "L_ch0",
        "FAST_RUN",
        "mu_sig",
        0.0,
        5.0,
        101,
    )


def test_build_stage_specs_selected_subset_preserves_order(
    workspace_path: Path,
) -> None:
    specs = stages.build_stage_specs(
        selected_stages=["nll_scan", "compiled_evaluation"],
        workspace_path=workspace_path,
        target="analysis",
        mode="FAST_COMPILE",
        n_runs=2,
        n_evaluations=3,
        distribution="pdf",
        scan_parameter="x",
        scan_min=-1.0,
        scan_max=1.0,
        n_scan_points=7,
    )

    assert [stage_name for stage_name, _function, _args in specs] == [
        "nll_scan",
        "compiled_evaluation",
    ]
    assert specs[0][2] == (
        workspace_path,
        "analysis",
        "FAST_COMPILE",
        "x",
        -1.0,
        1.0,
        7,
    )
    assert specs[1][2] == (workspace_path, "analysis", "FAST_COMPILE", 3)


def test_build_stage_specs_rejects_unknown_stage(workspace_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown stages"):
        stages.build_stage_specs(
            selected_stages=["missing"],
            workspace_path=workspace_path,
            target="L_ch0",
            mode="FAST_RUN",
            n_runs=1,
            n_evaluations=1,
            distribution="sig_ch0",
            scan_parameter="mu_sig",
            scan_min=0.0,
            scan_max=5.0,
            n_scan_points=2,
        )


def named_stage_function() -> dict[str, Any]:
    return {"benchmark": "demo", "status": "success"}


class CallableWithoutName:
    def __call__(self) -> dict[str, Any]:
        return {"benchmark": "callable_object"}


def test_make_stage_error_result_uses_function_name() -> None:
    try:
        raise RuntimeError("stage failed")
    except RuntimeError as exc:
        result = stages.make_stage_error_result(
            stage_name="demo_stage",
            function=named_stage_function,
            exc=exc,
        )

    assert result["benchmark"] == "demo_stage"
    assert result["stage_function"] == "named_stage_function"
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["error_message"] == "stage failed"
    assert "RuntimeError: stage failed" in result["traceback"]


def test_make_stage_error_result_falls_back_to_callable_type_name() -> None:
    callable_object = CallableWithoutName()

    try:
        raise ValueError("bad callable")
    except ValueError as exc:
        result = stages.make_stage_error_result(
            stage_name="callable_stage",
            function=callable_object,
            exc=exc,
        )

    assert result["stage_function"] == "CallableWithoutName"
    assert result["error_type"] == "ValueError"
    assert result["error_message"] == "bad callable"


class FakePool:
    def __init__(self, *, result: Any = None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.apply_calls: list[tuple[Any, tuple[Any, ...]]] = []

    def __enter__(self) -> "FakePool":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def apply(self, function: Any, args: tuple[Any, ...]) -> Any:
        self.apply_calls.append((function, args))
        if self.exc is not None:
            raise self.exc
        return self.result


class FakeContext:
    def __init__(self, pool: FakePool) -> None:
        self.pool = pool
        self.processes_seen: list[int] = []

    def Pool(self, processes: int) -> FakePool:
        self.processes_seen.append(processes)
        return self.pool


def test_run_stage_isolated_returns_success_result_and_preserves_existing_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {"benchmark": "existing", "status": "custom", "value": 42}
    pool = FakePool(result=result)
    context = FakeContext(pool)

    monkeypatch.setattr(stages, "get_context", lambda method: context)

    output = stages.run_stage_isolated(
        function=named_stage_function,
        args=("a", "b"),
        stage_name="explicit_stage",
    )

    assert output is result
    assert output == {"benchmark": "existing", "status": "custom", "value": 42}
    assert context.processes_seen == [1]
    assert pool.apply_calls == [(named_stage_function, ("a", "b"))]


def test_run_stage_isolated_adds_missing_benchmark_and_success_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = {"value": 42}
    pool = FakePool(result=result)
    context = FakeContext(pool)

    monkeypatch.setattr(stages, "get_context", lambda method: context)

    output = stages.run_stage_isolated(
        function=named_stage_function,
        args=(),
        stage_name="filled_stage",
    )

    assert output is result
    assert output["benchmark"] == "filled_stage"
    assert output["status"] == "success"
    assert output["value"] == 42


def test_run_stage_isolated_uses_function_name_when_stage_name_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = FakePool(result={})
    context = FakeContext(pool)

    monkeypatch.setattr(stages, "get_context", lambda method: context)

    output = stages.run_stage_isolated(
        function=named_stage_function,
        args=(),
    )

    assert output["benchmark"] == "named_stage_function"
    assert output["status"] == "success"


def test_run_stage_isolated_falls_back_to_unknown_stage_for_unnamed_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NamelessCallable:
        def __call__(self) -> dict[str, Any]:
            return {}

    callable_object = NamelessCallable()
    pool = FakePool(result={})
    context = FakeContext(pool)

    monkeypatch.setattr(stages, "get_context", lambda method: context)

    output = stages.run_stage_isolated(
        function=callable_object,
        args=(),
    )

    assert output["benchmark"] == "unknown_stage"
    assert output["status"] == "success"


def test_run_stage_isolated_returns_failure_for_pool_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = FakePool(exc=RuntimeError("child process failed"))
    context = FakeContext(pool)

    monkeypatch.setattr(stages, "get_context", lambda method: context)

    output = stages.run_stage_isolated(
        function=named_stage_function,
        args=(1, 2),
        stage_name="failing_stage",
    )

    assert output["benchmark"] == "failing_stage"
    assert output["stage_function"] == "named_stage_function"
    assert output["status"] == "failed"
    assert output["error_type"] == "RuntimeError"
    assert output["error_message"] == "child process failed"
    assert "RuntimeError: child process failed" in output["traceback"]


@pytest.mark.parametrize(
    ("bad_result", "type_name"),
    [
        (None, "NoneType"),
        (["not", "a", "dict"], "list"),
        ("not-a-dict", "str"),
    ],
)
def test_run_stage_isolated_returns_failure_for_non_dict_result(
    monkeypatch: pytest.MonkeyPatch,
    bad_result: Any,
    type_name: str,
) -> None:
    pool = FakePool(result=bad_result)
    context = FakeContext(pool)

    monkeypatch.setattr(stages, "get_context", lambda method: context)

    output = stages.run_stage_isolated(
        function=named_stage_function,
        args=(),
        stage_name="bad_result_stage",
    )

    assert output["benchmark"] == "bad_result_stage"
    assert output["stage_function"] == "named_stage_function"
    assert output["status"] == "failed"
    assert output["error_type"] == "TypeError"
    assert (
        output["error_message"]
        == f"Stage returned a non-dictionary result: {type_name}"
    )
