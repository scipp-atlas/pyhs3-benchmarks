# Cross-Framework Benchmarks

On this page, you will learn how the cross-framework benchmark suite is organized, which frameworks are compared, and how statistically equivalent comparisons are performed.

The **Cross-Framework Benchmarks** compare PyHS3 with other statistical inference frameworks using equivalent statistical models, identical benchmark configurations, and validated numerical agreement.

Unlike the workflow benchmarks, which measure individual stages of the PyHS3 execution pipeline, these benchmarks evaluate how different frameworks perform the same statistical computation.

---

# Apples-to-Apples Methodology

Meaningful framework comparisons require every implementation to evaluate the same mathematical problem.

Whenever supported by the participating frameworks, benchmarks use

- identical statistical models;
- identical observed datasets;
- identical parameter values;
- identical benchmark configurations;
- equivalent mathematical definitions;
- isolated benchmark processes.

This minimizes differences unrelated to framework implementation and ensures that reported performance reflects execution rather than benchmark setup.

---

# Common Benchmark Dataset

All cross-framework benchmarks use statistically equivalent benchmark workspaces generated with the `workspace-scripts` repository.

The complete workspace collection is documented in **Benchmark Workspaces**.

---

# Execution Engines

Depending on the benchmark, one or more of the following execution engines are compared.

| Engine | Description |
|---------|-------------|
| **PyHS3 (PyTensor)** | Non-compiled execution. |
| **PyHS3 (JAX)** | Compiled execution. |
| **RooFit** | ROOT RooFit implementation. |
| **xRooFit** | Workflow interface built on RooFit. |
| **pyhf** | HistFactory implementation using the NumPy backend. |

Not every benchmark includes every execution engine.

---

# Benchmark Categories

The benchmark suite covers four complementary comparison types.

- Scalar probability density evaluation.
- Likelihood evaluation.
- Binned likelihood evaluation.
- Complete workflow comparisons.

Together these benchmarks evaluate both numerical agreement and execution performance across multiple statistical frameworks.

---

# Available Benchmarks

## Scalar PDF Evaluation

Compares scalar probability density evaluation across PyHS3 and RooFit.

Measures

- cold-start latency;
- warm evaluation latency;
- throughput;
- memory usage;
- numerical agreement.

---

## ΔNLL Benchmark

Compares equivalent negative log-likelihood evaluation across statistically equivalent models.

Measures

- point-by-point ΔNLL evaluation;
- batched evaluation;
- numerical agreement;
- execution performance.

---

## xRooFit Benchmark

Compares complete likelihood evaluation workflows between PyHS3 and xRooFit.

Measures

- workflow initialization;
- cold-start execution;
- steady-state execution;
- scan performance;
- memory usage.

---

## Cross-Framework Binned Likelihood

Compares equivalent HistFactory models implemented in PyHS3 and pyhf.

Measures

- model construction;
- likelihood evaluation;
- scaling with histogram size;
- numerical agreement.

---

# Numerical Validation

Performance comparisons are interpreted only after numerical agreement has been verified.

Depending on the benchmark, validation includes

- scalar PDF agreement;
- ΔNLL agreement;
- likelihood residuals;
- best-fit parameter agreement;
- configurable numerical tolerances.

---

# Why These Benchmarks Matter

Cross-framework benchmarks

- independently validate PyHS3 implementations;
- verify that performance optimizations preserve numerical correctness;
- identify implementation-specific performance characteristics;
- provide reproducible comparisons using common benchmark inputs;
- separate initialization costs from steady-state execution.

---

# Related Documentation

See also

- **Benchmark Methodology**
- **Benchmark Workspaces**
- **Benchmark Results**
- **Workflow Benchmarks**
