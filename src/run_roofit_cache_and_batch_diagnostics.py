from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from pyhs3.workspace import Workspace

try:
    import ROOT
except ImportError:
    ROOT = None


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def version_info() -> dict[str, Any]:
    info: dict[str, Any] = {}

    try:
        import pyhs3

        info["pyhs3_version"] = getattr(pyhs3, "__version__", None)
    except Exception as exc:
        info["pyhs3_version_error"] = str(exc)

    try:
        info["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except Exception as exc:
        info["git_sha_error"] = str(exc)

    if ROOT is not None:
        try:
            info["root_version"] = str(ROOT.gROOT.GetVersion())
        except Exception as exc:
            info["root_version_error"] = str(exc)

    return info


def find_root_workspace(root_file: Any) -> Any:
    for key in root_file.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom(ROOT.RooWorkspace.Class()):
            return obj
    raise RuntimeError("No RooWorkspace found in ROOT file")


def is_valid_root_object(obj: Any) -> bool:
    if obj is None:
        return False
    try:
        return bool(obj)
    except Exception:
        return True


def parameter_point(
    workspace: Workspace, selected_name: str | None = None
) -> dict[str, Any]:
    points = workspace.parameter_points.root
    selected = (
        points[0]
        if selected_name is None
        else next(
            point for point in points if getattr(point, "name", None) == selected_name
        )
    )

    params: dict[str, Any] = {}
    for parameter in selected.parameters:
        params[parameter.name] = np.asarray(float(parameter.value), dtype=np.float64)
    return params


def pyhs3_data_values(
    workspace: Workspace,
    data_name: str,
    observable_index: int,
) -> np.ndarray:
    for data in workspace.data.root:
        if data.name == data_name:
            return np.asarray(
                [entry[observable_index] for entry in data.entries],
                dtype=np.float64,
            )
    available = [getattr(data, "name", "<unnamed>") for data in workspace.data.root]
    raise RuntimeError(f"Data {data_name!r} not found. Available: {available}")


def prepare_pyhs3(
    workspace_path: Path,
    analysis: str,
    data_name: str,
    observable_name: str,
    observable_index: int,
    mode: str,
    parameter_point_name: str | None,
):
    workspace = Workspace.load(workspace_path)
    model = workspace.model(analysis, progress=False, mode=mode)
    params = parameter_point(workspace, parameter_point_name)

    try:
        for name, value in model.free_params.items():
            params[name] = np.asarray(value, dtype=np.float64)
    except AttributeError:
        pass

    x_values = pyhs3_data_values(workspace, data_name, observable_index)
    params[observable_name] = x_values
    return model, params, x_values


def prepare_roofit(
    root_workspace_path: Path,
    pdf_name: str,
    poi_name: str,
    observable_name: str,
):
    if ROOT is None:
        raise RuntimeError("ROOT is not importable")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)

    root_file = ROOT.TFile.Open(str(root_workspace_path), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file {root_workspace_path}")

    root_workspace = find_root_workspace(root_file)
    pdf = root_workspace.pdf(pdf_name)
    if not is_valid_root_object(pdf):
        raise RuntimeError(f"Could not find PDF {pdf_name!r}")

    poi = root_workspace.var(poi_name)
    if not is_valid_root_object(poi):
        raise RuntimeError(f"Could not find POI {poi_name!r}")

    observable = root_workspace.var(observable_name)
    if not is_valid_root_object(observable):
        raise RuntimeError(f"Could not find observable {observable_name!r}")

    norm_set = ROOT.RooArgSet(observable)
    return root_file, pdf, poi, observable, norm_set


def median_time_seconds(fn, n_repeats: int) -> tuple[float, list[float]]:
    timings = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return float(statistics.median(timings)), timings


def diagnostic_roofit_cache(
    pdf: Any,
    observable: Any,
    norm_set: Any,
    x_values: np.ndarray,
    n_repeats: int,
) -> dict[str, Any]:
    x0 = float(x_values[0])
    x1 = float(x_values[len(x_values) // 2])

    observable.setVal(x0)

    first_same_x, first_same_samples = median_time_seconds(
        lambda: float(pdf.getVal(norm_set)),
        n_repeats,
    )

    second_same_x, second_same_samples = median_time_seconds(
        lambda: float(pdf.getVal(norm_set)),
        n_repeats,
    )

    def changed_x_call():
        observable.setVal(x1)
        return float(pdf.getVal(norm_set))

    changed_x, changed_samples = median_time_seconds(changed_x_call, n_repeats)

    xs = np.asarray([x0, x1], dtype=np.float64)
    counter = {"i": 0}

    def alternating_x_call():
        i = counter["i"]
        observable.setVal(float(xs[i % 2]))
        counter["i"] = i + 1
        return float(pdf.getVal(norm_set))

    alternating_x, alternating_samples = median_time_seconds(
        alternating_x_call,
        n_repeats,
    )

    return {
        "n_repeats": n_repeats,
        "x0": x0,
        "x1": x1,
        "first_same_x_median_seconds": first_same_x,
        "second_same_x_median_seconds": second_same_x,
        "changed_x_median_seconds": changed_x,
        "alternating_x_median_seconds": alternating_x,
        "first_same_x_samples_seconds": first_same_samples,
        "second_same_x_samples_seconds": second_same_samples,
        "changed_x_samples_seconds": changed_samples,
        "alternating_x_samples_seconds": alternating_samples,
    }


def diagnostic_nll_batching(
    model: Any,
    params: dict[str, Any],
    target: str,
    poi_name: str,
    x_values: np.ndarray,
    observable_name: str,
    mu_value: float,
    n_repeats: int,
) -> dict[str, Any]:
    def pyhs3_batched():
        eval_params = dict(params)
        eval_params[poi_name] = np.asarray(mu_value, dtype=np.float64)
        eval_params[observable_name] = x_values
        logpdf = np.asarray(model.logpdf(target, **eval_params), dtype=np.float64)
        return -float(np.sum(logpdf))

    def pyhs3_nonbatched():
        total = 0.0
        for x in x_values:
            eval_params = dict(params)
            eval_params[poi_name] = np.asarray(mu_value, dtype=np.float64)
            eval_params[observable_name] = np.asarray([float(x)], dtype=np.float64)
            logpdf = np.asarray(model.logpdf(target, **eval_params), dtype=np.float64)
            total += float(np.asarray(logpdf).reshape(-1)[0])
        return -total

    batched_median, batched_samples = median_time_seconds(pyhs3_batched, n_repeats)
    nonbatched_median, nonbatched_samples = median_time_seconds(
        pyhs3_nonbatched,
        n_repeats,
    )

    batched_nll = pyhs3_batched()
    nonbatched_nll = pyhs3_nonbatched()

    return {
        "n_repeats": n_repeats,
        "n_events": int(len(x_values)),
        "mu": float(mu_value),
        "pyhs3_batched_nll": float(batched_nll),
        "pyhs3_nonbatched_nll": float(nonbatched_nll),
        "pyhs3_abs_nll_diff": float(abs(batched_nll - nonbatched_nll)),
        "pyhs3_batched_median_seconds": batched_median,
        "pyhs3_nonbatched_median_seconds": nonbatched_median,
        "pyhs3_nonbatched_over_batched": float(nonbatched_median / batched_median)
        if batched_median > 0
        else math.inf,
        "pyhs3_batched_samples_seconds": batched_samples,
        "pyhs3_nonbatched_samples_seconds": nonbatched_samples,
    }


def diagnostic_roofit_pointwise_nll(
    pdf: Any,
    poi: Any,
    observable: Any,
    norm_set: Any,
    x_values: np.ndarray,
    mu_value: float,
    n_repeats: int,
) -> dict[str, Any]:
    def roofit_pointwise():
        poi.setVal(float(mu_value))
        total = 0.0
        for x in x_values:
            observable.setVal(float(x))
            pdf_value = float(pdf.getVal(norm_set))
            if pdf_value <= 0.0 or not math.isfinite(pdf_value):
                raise RuntimeError(f"Invalid RooFit PDF value: {pdf_value}")
            total += math.log(pdf_value)
        return -total

    median, samples = median_time_seconds(roofit_pointwise, n_repeats)
    nll = roofit_pointwise()

    return {
        "n_repeats": n_repeats,
        "n_events": int(len(x_values)),
        "mu": float(mu_value),
        "roofit_pointwise_nll": float(nll),
        "roofit_pointwise_median_seconds": median,
        "roofit_pointwise_samples_seconds": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostics for RooFit getVal caching and PyHS3 batching."
    )
    parser.add_argument("--json-workspace", type=Path, required=True)
    parser.add_argument("--root-workspace", type=Path, required=True)
    parser.add_argument("--analysis", default="L_ch0")
    parser.add_argument("--target", default="model_ch0")
    parser.add_argument("--pyhs3-data-name", default="combData_ch0")
    parser.add_argument("--root-pdf-name", default="model_ch0")
    parser.add_argument("--observable-name", default="x")
    parser.add_argument("--observable-index", type=int, default=0)
    parser.add_argument("--poi", default="mu_sig")
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--mode", default="FAST_RUN")
    parser.add_argument("--parameter-point", default=None)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/pyhs3_roofit_diagnostics/diagnostics_result.json"),
    )
    args = parser.parse_args()

    model, params, x_values = prepare_pyhs3(
        workspace_path=args.json_workspace,
        analysis=args.analysis,
        data_name=args.pyhs3_data_name,
        observable_name=args.observable_name,
        observable_index=args.observable_index,
        mode=args.mode,
        parameter_point_name=args.parameter_point,
    )

    root_file, pdf, poi, observable, norm_set = prepare_roofit(
        root_workspace_path=args.root_workspace,
        pdf_name=args.root_pdf_name,
        poi_name=args.poi,
        observable_name=args.observable_name,
    )

    try:
        root_cache = diagnostic_roofit_cache(
            pdf=pdf,
            observable=observable,
            norm_set=norm_set,
            x_values=x_values,
            n_repeats=args.n_repeats,
        )

        pyhs3_batching = diagnostic_nll_batching(
            model=model,
            params=params,
            target=args.target,
            poi_name=args.poi,
            x_values=x_values,
            observable_name=args.observable_name,
            mu_value=args.mu,
            n_repeats=args.n_repeats,
        )

        roofit_pointwise = diagnostic_roofit_pointwise_nll(
            pdf=pdf,
            poi=poi,
            observable=observable,
            norm_set=norm_set,
            x_values=x_values,
            mu_value=args.mu,
            n_repeats=args.n_repeats,
        )

        result = {
            "benchmark": "roofit_cache_and_pyhs3_batching_diagnostics",
            "configuration": {
                "json_workspace": str(args.json_workspace),
                "root_workspace": str(args.root_workspace),
                "analysis": args.analysis,
                "target": args.target,
                "pyhs3_data_name": args.pyhs3_data_name,
                "root_pdf_name": args.root_pdf_name,
                "observable_name": args.observable_name,
                "observable_index": args.observable_index,
                "poi": args.poi,
                "mu": args.mu,
                "mode": args.mode,
                "n_repeats": args.n_repeats,
                "n_events": int(len(x_values)),
            },
            "versions": version_info(),
            "roofit_cache_check": root_cache,
            "pyhs3_batching_check": pyhs3_batching,
            "roofit_pointwise_nll_check": roofit_pointwise,
        }

        save_json(result, args.output)

        print()
        print("=" * 80)
        print("RooFit cache + PyHS3 batching diagnostics")
        print("=" * 80)
        print(f"n events: {len(x_values)}")
        print(f"output:   {args.output}")
        print()
        print("RooFit getVal cache check [median per call]")
        print(
            f"  first same x:   {root_cache['first_same_x_median_seconds'] * 1e6:.3f} us"
        )
        print(
            f"  second same x:  {root_cache['second_same_x_median_seconds'] * 1e6:.3f} us"
        )
        print(
            f"  changed x:      {root_cache['changed_x_median_seconds'] * 1e6:.3f} us"
        )
        print(
            f"  alternating x:  {root_cache['alternating_x_median_seconds'] * 1e6:.3f} us"
        )
        print()
        print("PyHS3 batch vs non-batch NLL [median per NLL]")
        print(
            f"  batched:        {pyhs3_batching['pyhs3_batched_median_seconds'] * 1e6:.3f} us"
        )
        print(
            f"  non-batched:    {pyhs3_batching['pyhs3_nonbatched_median_seconds'] * 1e6:.3f} us"
        )
        print(
            f"  ratio:          {pyhs3_batching['pyhs3_nonbatched_over_batched']:.2f}x"
        )
        print(f"  abs NLL diff:   {pyhs3_batching['pyhs3_abs_nll_diff']:.6e}")
        print()
        print("RooFit pointwise NLL [median per NLL]")
        print(
            f"  pointwise:      {roofit_pointwise['roofit_pointwise_median_seconds'] * 1e6:.3f} us"
        )
        print()

    finally:
        root_file.Close()


if __name__ == "__main__":
    main()
