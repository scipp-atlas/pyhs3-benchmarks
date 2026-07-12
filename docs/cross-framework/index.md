# Cross-Framework Benchmarks

The **Cross-Framework** benchmark suite evaluates equivalent statistical computations across multiple statistical inference frameworks.

Unlike the workflow benchmarks, which characterize the internal performance of PyHS3, the benchmarks in this section compare PyHS3 with external implementations while ensuring that every execution engine performs the **same statistical computation**.

The primary goals of these benchmarks are to

- validate numerical agreement between implementations;
- compare performance under equivalent statistical conditions;
- identify implementation-specific performance characteristics;
- ensure that performance optimizations do not compromise statistical correctness.

All comparisons follow an **apples-to-apples** methodology using matching benchmark workspaces, identical parameter values, identical statistical models, and equivalent benchmark configurations.

---

# Philosophy

Cross-framework benchmarking is fundamentally different from traditional performance benchmarking.

Comparing statistical inference frameworks is meaningful only if every engine evaluates the **same mathematical quantity** using the **same statistical model**.

For this reason, every benchmark in this section is designed to eliminate differences arising from

- model construction;
- workspace contents;
- parameter initialization;
- observed datasets;
- normalization conventions;
- benchmark configuration.

Only after these quantities have been aligned are execution time, memory consumption, and numerical agreement compared.

---

# Common Benchmark Dataset

All cross-framework benchmarks operate on the canonical benchmark workspace collection described in **Benchmark Workspaces**.

Each HS3 workspace has a corresponding ROOT workspace generated from the same statistical model using the `workspace-scripts` repository.

This guarantees that every execution engine evaluates statistically equivalent models and that any observed differences originate from the framework implementation rather than differences in the benchmark inputs.

---

# Execution Engines

Depending on the benchmark, one or more of the following execution engines are compared.

| Engine | Description |
|---------|-------------|
| **PyHS3 non-compiled (PyTensor)** | Eager execution without graph compilation. |
| **PyHS3 compiled (JAX)** | JAX-compiled execution after graph compilation. Cold-start and steady-state execution are reported separately. |
| **RooFit** | ROOT RooFit implementation using statistically equivalent ROOT workspaces. |
| **xRooFit** | xRooFit likelihood evaluation built on top of RooFit. |

Not every benchmark includes every execution engine. Each benchmark page specifies which engines participate in the comparison.

---

# Benchmark Categories

The cross-framework benchmark suite intentionally compares different levels of statistical computation.

Together, the benchmarks cover

- engine-level scalar probability density function evaluation;
- complete point-by-point likelihood evaluation;
- batched full-dataset likelihood evaluation;
- complete statistical workflows evaluated through xRooFit.

These complementary benchmarks provide both fair engine-to-engine comparisons and realistic workflow-level performance measurements, allowing PyHS3 to be evaluated from multiple perspectives while maintaining statistical equivalence whenever a direct comparison is possible.

---

# Available Benchmarks

The repository currently provides the following cross-framework benchmarks.

## Scalar PDF Evaluation

Compares repeated normalized scalar probability density function evaluation across

- **PyHS3 non-compiled (PyTensor)**;
- **PyHS3 compiled (JAX)**;
- **RooFit**.

The benchmark reports

- cold-start end-to-end latency;
- steady-state scalar evaluation latency;
- throughput;
- memory usage;
- numerical agreement;
- compiled execution lifecycle.

The benchmark isolates scalar PDF evaluation from likelihood construction, fitting and minimization, providing an engine-level comparison under identical statistical conditions.

---

## ΔNLL Benchmark

Compares complete negative log-likelihood evaluation using statistically equivalent HS3 and ROOT workspaces.

All participating engines

- evaluate the same observed dataset;
- scan the same parameter of interest;
- use identical parameter values;
- compute the same likelihood quantity.

The benchmark distinguishes between

- **point-by-point likelihood evaluation**, providing the primary apples-to-apples engine-to-engine comparison; and
- **batched full-dataset evaluation**, demonstrating the native vectorized execution model of pyHS3.

Numerical agreement is verified before any performance measurements are interpreted.

---

## xRooFit Benchmark

Compares complete likelihood evaluation workflows between **PyHS3** and **xRooFit** using statistically equivalent HS3 and ROOT workspaces.

The benchmark evaluates

- PyHS3 non-compiled execution;
- PyHS3 compiled execution;
- xRooFit likelihood evaluation.

All engines execute equivalent ΔNLL scans using identical datasets, parameter values, scan grids and benchmark configurations.

In addition to numerical validation, the benchmark separately reports

- workspace loading;
- model construction;
- likelihood construction;
- first (cold-start) evaluation;
- steady-state evaluation;
- complete scan performance;
- memory usage.

This benchmark therefore provides both an engine-to-engine workflow comparison and an independent validation of PyHS3 against the public xRooFit API.

---

# Apples-to-Apples Methodology

The benchmarks in this section are designed to compare equivalent statistical computations whenever a direct engine-to-engine comparison is possible.

Whenever supported by the participating frameworks, all engines use

- identical statistical models;
- identical observed datasets;
- identical parameter values;
- identical observable values;
- identical scan grids;
- identical benchmark configurations;
- identical numerical tolerances;
- equivalent mathematical definitions;
- isolated benchmark processes.

This methodology minimizes systematic differences unrelated to implementation and allows benchmark results to be interpreted with confidence.

Where an exact one-to-one comparison is not possible—for example, for batched full-dataset evaluation—the corresponding benchmark documentation explicitly describes the methodological differences and explains how the resulting performance measurements should be interpreted.

---

# Numerical Validation

Performance measurements alone are not sufficient for cross-framework benchmarking.

Every benchmark therefore validates that the compared execution engines produce numerically equivalent statistical results **before any performance measurements are interpreted**.

Depending on the benchmark, validation may include

- scalar PDF agreement;
- negative log-likelihood agreement;
- ΔNLL profile agreement;
- point-by-point residuals;
- best-fit parameter agreement;
- configurable numerical tolerances.

Only benchmark runs that satisfy the corresponding validation criteria should be interpreted as meaningful performance comparisons.

---

# Why Cross-Framework Benchmarks?

Cross-framework benchmarking provides several important benefits.

- Independent verification of PyHS3 implementations.
- Confidence that performance optimizations preserve numerical correctness.
- Identification of performance trade-offs across execution engines.
- Reproducible comparisons using common benchmark inputs.
- Transparent reporting of both numerical agreement and execution performance.
- Separation of cold-start initialization costs from steady-state execution performance.
- Direct comparison between compiled and non-compiled PyHS3 execution.

Together, these benchmarks complement the workflow benchmarks by demonstrating not only how efficiently PyHS3 executes statistical computations, but also how its numerical results compare with established statistical inference frameworks.

---

# Available Documentation

This section currently includes

- **Scalar PDF Evaluation**
- **ΔNLL Benchmark**
- **xRooFit Benchmark**

Each benchmark page describes

- benchmark methodology;
- benchmark lifecycle;
- execution procedure;
- command-line interface;
- generated outputs;
- numerical validation strategy;
- interpretation of the reported performance;
- benchmark limitations where applicable.

---

# Related Documentation

See also

- **Benchmark Methodology**
- **Benchmark Workspaces**
- **Benchmark Results**
- **Outputs**
