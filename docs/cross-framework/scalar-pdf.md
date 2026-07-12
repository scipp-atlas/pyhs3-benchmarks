# Cross-framework Scalar PDF Evaluation

The **Scalar PDF Evaluation** benchmark provides an apples-to-apples comparison of the cost of evaluating a **single normalized probability density function (PDF)** across multiple statistical inference engines.

Unlike the likelihood benchmarks, which measure complete statistical workflows, this benchmark isolates **scalar PDF evaluation only**, allowing the execution overhead of each framework to be studied independently of model fitting, minimization, or likelihood construction.

The benchmark compares

- **pyHS3 non-compiled (PyTensor)**
- **pyHS3 compiled (JAX)**
- **ROOT RooFit**

using statistically equivalent HS3 and ROOT workspaces generated from the same underlying statistical models.

---

# Benchmark goals

This benchmark has four primary objectives.

- Verify numerical agreement between all execution engines.
- Measure the cost of a single scalar PDF evaluation.
- Separate one-time initialization costs from repeated execution.
- Compare compiled and non-compiled pyHS3 under identical statistical conditions.

The benchmark intentionally avoids model fitting or likelihood minimization in order to isolate the execution characteristics of scalar PDF evaluation.

---

# Benchmark methodology

Every execution engine evaluates exactly the same statistical quantity.

For every benchmark workspace the benchmark

1. loads the statistical workspace;
2. constructs the statistical model;
3. synchronizes model parameters across frameworks;
4. evaluates the same normalized scalar PDF;
5. validates numerical agreement;
6. measures startup latency;
7. measures steady-state evaluation performance;
8. measures memory consumption.

The benchmark therefore compares framework implementations rather than different statistical workflows.

---

# Execution engines

The benchmark evaluates three execution engines.

| Engine | Description |
|---------|-------------|
| **pyHS3 non-compiled (PyTensor)** | Standard eager execution without graph compilation. |
| **pyHS3 compiled (JAX)** | JAX-compiled execution after graph compilation. Startup and steady-state execution are reported separately. |
| **ROOT RooFit** | Scalar normalized PDF evaluation using statistically equivalent ROOT workspaces. |

Every engine evaluates

- the same statistical model;
- the same parameter point;
- the same observable;
- the same normalization;
- the same probability density function.

This ensures that any observed performance differences originate from the implementation rather than differences in the statistical model.

---

# Benchmark lifecycle

Every benchmark is executed inside a **fresh spawned subprocess**.

Each subprocess independently performs

1. workspace loading;
2. model construction;
3. graph preparation;
4. JAX compilation (compiled engine only);
5. first scalar PDF evaluation;
6. steady-state evaluation;
7. memory measurement.

Running each engine in an isolated process prevents measurements from being affected by

- previous ROOT initialization,
- JAX compilation cache,
- PyTensor internal state,
- allocator reuse,
- memory allocated by another execution engine.

Fresh-process isolation therefore provides reproducible startup, runtime and memory measurements across all compared frameworks.

---

# Input modes

The benchmark supports two observable input modes.

## Varying observable (default)

The observable value changes before every scalar PDF evaluation.

This represents the typical workload encountered during likelihood scans, minimization and repeated statistical inference.

All primary performance comparisons use this mode.

---

## Fixed observable

The observable remains unchanged throughout the benchmark.

This mode is provided exclusively as a **cache diagnostic**.

It allows framework-specific caching behaviour to be investigated, particularly for RooFit, but is **not** intended as the primary performance comparison.

---

# Numerical validation

Performance measurements are interpreted only after numerical agreement has been verified.

The benchmark first evaluates the **pyHS3 non-compiled** implementation and stores its output as the numerical reference.

Compiled pyHS3 and RooFit are then evaluated on the identical observable grid.

For every engine the benchmark computes

- maximum absolute difference;
- maximum relative difference.

Validation uses configurable

- absolute tolerance (`--atol`);
- relative tolerance (`--rtol`).

Only benchmark runs satisfying these tolerances should be interpreted as meaningful performance comparisons.

---

# Startup versus steady-state execution

One of the primary goals of this benchmark is to distinguish **initialization cost** from **repeated execution performance**.

The compiled implementation therefore reports two different execution metrics.

## Cold-start end-to-end latency

This metric includes every operation required before the first successful PDF evaluation.

For the compiled engine this includes

- workspace loading;
- model construction;
- graph preparation;
- JAX compilation;
- first evaluation.

Cold-start latency is the relevant metric for one-off workflows where initialization cannot be amortized.

---

## Steady-state evaluation

Steady-state timing measures only repeated scalar PDF evaluations performed **after graph compilation has completed**.

Compilation is intentionally excluded from this measurement.

This metric represents the execution cost encountered during

- likelihood scans,
- repeated evaluations,
- iterative minimization,
- inference workflows.

Separating these two metrics prevents compilation time from artificially inflating repeated evaluation performance.

---

# Generated figures

## Scalar PDF startup and steady-state latency

![](../assets/plots/cross_scalar_pdf/scalar_pdf_varying_latency.png)

This figure compares cold-start execution with steady-state scalar evaluation.

The cold-start measurement includes all initialization required before the first evaluation, while the steady-state measurement reports only repeated execution after initialization has completed.

---

## Scalar PDF execution time

![](../assets/plots/cross_scalar_pdf/scalar_pdf_varying_time_per_value.png)

Shows the median runtime required for a single scalar PDF evaluation as the number of repeated evaluations increases.

This figure characterizes the sustained execution cost of scalar PDF evaluation across the compared execution engines.

---

## Scalar PDF throughput

![](../assets/plots/cross_scalar_pdf/scalar_pdf_varying_throughput.png)

Shows the sustained evaluation throughput measured in evaluations per second.

Increasing throughput corresponds to improved steady-state execution performance.

---

## Memory consumption

![](../assets/plots/cross_scalar_pdf/scalar_pdf_varying_memory.png)

Reports the memory increase observed during benchmark execution.

Both

- Current RSS increase
- Peak RSS increase

are measured inside isolated subprocesses, ensuring directly comparable memory measurements across all execution engines.

---

## Numerical agreement

![](../assets/plots/cross_scalar_pdf/scalar_pdf_varying_numerical_agreement.png)

Shows the maximum numerical deviation relative to the pyHS3 non-compiled reference implementation.

The horizontal tolerance line indicates the configured validation threshold.

Successful validation confirms that performance comparisons are performed on statistically equivalent computations.

---

## Compiled execution lifecycle

![](../assets/plots/cross_scalar_pdf/scalar_pdf_compiled_lifecycle.png)

The compiled lifecycle separates the startup cost into

- model construction;
- graph preparation;
- JAX compilation;
- first evaluation.

This visualization illustrates where initialization time is spent before compiled execution reaches steady-state performance.

---

# Cache diagnostic

The benchmark additionally produces equivalent figures for the fixed observable mode.

These measurements are intended only to investigate framework-specific caching behaviour.

The corresponding figures include

- fixed observable latency;
- fixed observable throughput;
- fixed observable memory usage;
- fixed observable numerical agreement;
- fixed observable execution time.

These results should not be interpreted as representative likelihood evaluation performance.

---

# Benchmark outputs

The benchmark produces

- JSON benchmark results;
- numerical validation summaries;
- startup timing measurements;
- steady-state timing measurements;
- lifecycle measurements;
- throughput measurements;
- memory measurements;
- publication-quality figures.

These outputs provide a complete characterization of scalar PDF evaluation across all supported execution engines.

---

# Interpretation

The scalar PDF benchmark should be interpreted as an engine-level comparison.

It intentionally excludes

- likelihood construction,
- minimization,
- parameter estimation,
- fitting workflows.

Consequently, it measures the execution characteristics of the PDF evaluation itself rather than complete statistical inference workflows.

For workflow-level comparisons, see the ΔNLL benchmark documentation.
