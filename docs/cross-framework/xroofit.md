# xRooFit Benchmark

On this page, you will learn how PyHS3 and xRooFit are compared using statistically equivalent likelihood models and how to interpret the benchmark results.

The **xRooFit Benchmark** compares complete negative log-likelihood evaluation workflows between **PyHS3** and **xRooFit** using equivalent HS3 and ROOT workspaces.

Unlike the **Cross-Framework Scalar PDF Evaluation** benchmark, which measures isolated PDF evaluation, this benchmark compares complete statistical workflows including likelihood construction, repeated evaluation, and ΔNLL scans.

---

# Why xRooFit?

xRooFit provides a high-level interface for statistical workflows built on top of RooFit.

Rather than interacting directly with RooFit objects, users work through the `xRooNode` API, which simplifies likelihood construction, parameter scans, and model inspection.

The benchmark intentionally evaluates the public xRooFit API instead of calling RooFit internals directly.

Official repository:

https://gitlab.cern.ch/will/xroofit

---

# Benchmark Goals

The benchmark is designed to

- validate ΔNLL agreement between PyHS3 and xRooFit;
- compare compiled and non-compiled PyHS3 execution;
- measure complete likelihood evaluation performance;
- separate initialization costs from repeated execution.

---

# Execution Engines

The benchmark compares

| Engine | Description |
|---------|-------------|
| **PyHS3 (PyTensor)** | Non-compiled execution. |
| **PyHS3 (JAX)** | Compiled execution. |
| **xRooFit** | ROOT likelihood evaluation through the xRooFit API. |

All engines evaluate statistically equivalent models using identical scan parameters.

---

# Benchmark Workflow

```text
HS3 Workspace              ROOT Workspace
      │                           │
      ▼                           ▼
PyHS3                    xRooFit API
      │                           │
      └────────────┬──────────────┘
                   ▼
         Numerical Validation
                   ▼
         Cold-Start Evaluation
                   ▼
      Repeated Likelihood Evaluation
                   │
                   ├── Timing
                   ├── Memory
                   ├── ΔNLL Agreement
                   └── Full Scan Runtime
                   ▼
            Comparison Plots
```

---

# Installation

Clone and build xRooFit.

```bash
git clone https://gitlab.cern.ch/will/xroofit.git external/xroofit

cd external/xroofit
mkdir build
cd build

cmake ..
make -j$(nproc)
```

---

# Activating xRooFit

Activate the xRooFit environment before running the benchmark.

```bash
source external/xroofit/build/setup.sh
```

This step must be repeated for every new terminal session.

---

# Running the Benchmark

```bash
pixi run python -m src.run_pyhs3_xroofit_benchmark \
    ...
    --plot
```

The benchmark

- constructs equivalent likelihoods;
- validates numerical agreement;
- measures startup and steady-state performance;
- performs ΔNLL scans;
- generates comparison plots.

---

# Numerical Validation

Performance comparisons are interpreted only after numerical agreement has been verified.

Validation includes

- ΔNLL profile agreement;
- best-fit parameter agreement;
- maximum absolute difference;
- maximum relative difference;
- RMS ΔNLL difference.

ΔNLL is compared instead of raw NLL values to remove framework-dependent additive constants.

---

# Results

## ΔNLL Profile

![](../assets/plots/pyhs3_xroofit_benchmark/delta_nll_profile.png)

Equivalent ΔNLL profiles demonstrate that all execution engines evaluate statistically equivalent likelihoods.

---

## Pointwise ΔNLL Differences

![](../assets/plots/pyhs3_xroofit_benchmark/delta_nll_absolute_differences.png)

Pointwise differences quantify numerical agreement between execution engines.

---

## Steady-State Runtime

![](../assets/plots/pyhs3_xroofit_benchmark/steady_state_runtime.png)

Steady-state timing compares repeated likelihood evaluation after initialization.

---

## Full ΔNLL Scan Runtime

![](../assets/plots/pyhs3_xroofit_benchmark/full_scan_runtime.png)

This benchmark measures the runtime of complete likelihood scans under identical benchmark conditions.

---

## Setup Breakdown

![](../assets/plots/pyhs3_xroofit_benchmark/timing_phase_breakdown.png)

Initialization is separated into workspace loading, model construction, NLL construction, and first evaluation.

---

## Numerical Agreement Summary

![](../assets/plots/pyhs3_xroofit_benchmark/numerical_agreement.png)

The benchmark summarizes maximum ΔNLL differences together with the configured validation tolerances.

---

# Runtime Verification

To ensure that the benchmark evaluates the intended interface, runtime metadata verifies that the likelihood is constructed through

```python
ROOT.xRooNode(workspace)[model].nll(dataset)
```

rather than directly through RooFit.

---

# Limitations

This benchmark compares equivalent statistical workflows rather than identical internal implementations.

Consequently

- initialization procedures differ;
- workspace loading differs;
- graph construction differs;
- raw NLL values may differ by additive constants.

The strongest engine-to-engine comparison is therefore provided by

- ΔNLL agreement;
- repeated steady-state evaluation;
- complete ΔNLL scan performance.

---

# Related Documentation

See also

- **Cross-Framework Benchmarks**
- **Cross-Framework Scalar PDF Evaluation**
- **Cross-Framework ΔNLL Benchmark**
- **Benchmark Methodology**
- **Benchmark Results**
