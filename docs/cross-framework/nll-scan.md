# Cross-framework ΔNLL Benchmark

The **Cross-framework ΔNLL Benchmark** compares complete likelihood evaluation across multiple statistical inference engines using statistically equivalent HS3 and ROOT workspaces.

Unlike the scalar PDF benchmark, which isolates the cost of evaluating a single normalized probability density function, this benchmark measures the execution of a **complete negative log-likelihood (NLL) evaluation** and validates that every framework produces the same statistical result before comparing performance.

The benchmark compares

- **pyHS3 non-compiled (PyTensor)**
- **pyHS3 compiled (JAX)**
- **ROOT RooFit**

using identical statistical models, observed datasets, parameter values, and scan grids.

---

# Benchmark goals

The benchmark has five primary objectives.

- Validate numerical agreement between pyHS3 and RooFit.
- Compare complete NLL evaluation performance across execution engines.
- Separate initialization costs from repeated execution performance.
- Compare compiled and non-compiled pyHS3 under identical statistical conditions.
- Demonstrate the performance benefits of batched vectorized execution.

Unlike workflow benchmarks, this benchmark is designed as an **engine-to-engine comparison**, ensuring that every framework evaluates the same statistical quantity whenever a direct comparison is possible.

---

# Statistical quantity

For every scan point

\[
\mu_i
\]

the benchmark evaluates

\[
\mathrm{NLL}(\mu_i)
=
-\sum_k \log p(x_k|\mu_i)
\]

using the identical observed dataset.

The reported scan is

\[
\Delta\mathrm{NLL}(\mu)
=
\mathrm{NLL}(\mu)
-
\min_\mu \mathrm{NLL}(\mu).
\]

Using ΔNLL rather than the absolute likelihood removes constant offsets and allows direct comparison of the statistical behaviour of every implementation.

---

# Execution engines

The benchmark evaluates three execution engines.

| Engine | Description |
|---------|-------------|
| **pyHS3 non-compiled (PyTensor)** | Standard eager execution without graph compilation. |
| **pyHS3 compiled (JAX)** | JAX-compiled execution after graph compilation. Startup and steady-state execution are reported separately. |
| **ROOT RooFit** | Equivalent likelihood evaluation using statistically matched ROOT workspaces. |

Every execution engine uses

- the same statistical model;
- the same observed dataset;
- the same parameter values;
- the same scan grid;
- the same normalization convention.

This allows differences in execution time to be attributed to implementation rather than differences in the statistical model.

---

# Two benchmark categories

The benchmark intentionally distinguishes between two different execution patterns.

## Point-by-point NLL evaluation

This benchmark represents the typical workload encountered during

- likelihood scans,
- profile likelihood evaluation,
- minimization,
- repeated objective-function evaluation.

For every parameter value

\[
\mu
\]

each engine performs one complete NLL evaluation using the identical observed dataset.

### RooFit

RooFit evaluates the normalized PDF for every observed event and computes

\[
-\sum \log(\mathrm{PDF}).
\]

### pyHS3 non-compiled

The non-compiled implementation performs the same calculation using eager PyTensor execution.

### pyHS3 compiled

The compiled implementation performs one complete compiled NLL evaluation for every parameter value.

The scalar PDF is evaluated over the complete dataset inside a single compiled JAX executable using vectorized evaluation.

Graph preparation and JAX compilation occur **before** any timed measurements and are never included in steady-state performance.

This benchmark therefore compares equivalent statistical computations while avoiding repeated compilation or repeated JAX dispatch overhead.

---

## Batched full-dataset evaluation

The batched benchmark measures a different execution strategy.

Instead of evaluating scalar PDF values one event at a time, pyHS3 receives the complete observable array and evaluates the likelihood using its native vectorized execution model.

This benchmark intentionally demonstrates one of the primary strengths of pyHS3.

Because RooFit does not provide an equivalent array-oriented execution model, this benchmark is **not** a pure apples-to-apples microbenchmark.

Instead, it should be interpreted as a **workflow benchmark** illustrating the benefits of vectorization and batched execution.

---

# Benchmark lifecycle

Every benchmark run is executed inside a fresh spawned subprocess.

Each subprocess independently performs

1. workspace loading;
2. model construction;
3. graph preparation;
4. JAX compilation (compiled engine only);
5. first NLL evaluation;
6. repeated steady-state evaluations;
7. memory measurement.

Running every engine in a separate process prevents measurements from being affected by

- previously initialized ROOT state;
- JAX compilation cache;
- PyTensor internal state;
- allocator reuse;
- memory allocated by another execution engine.

Fresh-process isolation ensures reproducible startup, runtime and memory measurements across all execution engines.

---

# Cold-start versus steady-state

Compiled execution naturally contains one-time initialization costs that are irrelevant during repeated likelihood evaluation.

To avoid mixing these costs with repeated execution performance, the benchmark reports two complementary metrics.

## Cold-start end-to-end latency

Cold-start latency includes every operation required before the first successful likelihood evaluation.

For the compiled engine this includes

- workspace loading;
- model construction;
- graph preparation;
- JAX compilation;
- first NLL evaluation.

This metric is representative of one-off workflows where initialization cannot be amortized.

---

## Steady-state evaluation

Steady-state timing measures only repeated likelihood evaluations after graph compilation has completed.

Compilation is intentionally excluded.

This metric represents the execution cost encountered during

- likelihood scans;
- repeated minimization;
- profile likelihood evaluation;
- statistical inference.

Separating these measurements prevents compilation time from artificially inflating repeated execution performance.

---

# Numerical validation

Performance comparisons are meaningful only if every framework produces statistically equivalent results.

The benchmark therefore validates every execution engine before interpreting performance measurements.

For every workspace the benchmark verifies

- identical ΔNLL minimum;
- identical ΔNLL profile;
- point-by-point numerical agreement;
- agreement within configurable tolerances.

Only benchmark runs satisfying these validation criteria should be interpreted as valid performance comparisons.

---

# Generated figures

## Point-by-point ΔNLL agreement

![](../assets/plots/cross_nll/cross_nll_scan_agreement.png)

This figure compares the ΔNLL profiles produced by every execution engine.

The curves should overlap within numerical tolerance, demonstrating that all frameworks evaluate statistically equivalent likelihoods.

---

## Steady-state runtime

![](../assets/plots/cross_nll/cross_nll_steady_state_runtime.png)

Shows the median execution time required for one complete NLL evaluation after initialization has completed.

The figure separately reports

- point-by-point evaluation;
- batched full-dataset evaluation.

These measurements characterize sustained likelihood evaluation performance.

---

## Cold-start versus steady-state

![](../assets/plots/cross_nll/cross_nll_end_to_end_vs_steady.png)

Separates initialization cost from repeated execution.

Cold-start includes workspace loading, model construction, graph preparation, JAX compilation and the first successful likelihood evaluation.

Steady-state measures only repeated NLL evaluations performed after initialization has completed.

---

## Compiled execution lifecycle

![](../assets/plots/cross_nll/cross_nll_compiled_lifecycle.png)

Breaks the compiled execution lifecycle into

- model construction;
- graph preparation;
- JAX compilation;
- first function call.

The figure illustrates where startup time is spent before compiled execution reaches steady-state performance.

---

## Memory profile

![](../assets/plots/cross_nll/cross_nll_memory_profile.png)

Reports the increase in memory usage observed during benchmark execution.

Both

- Current RSS increase
- Peak RSS increase

are measured inside isolated subprocesses, ensuring directly comparable measurements across execution engines.

---

# Benchmark outputs

The benchmark produces

- JSON benchmark results;
- ΔNLL validation summaries;
- cold-start timing measurements;
- steady-state timing measurements;
- compiled lifecycle measurements;
- memory measurements;
- publication-quality figures.

Together these outputs provide a complete characterization of likelihood evaluation across all supported execution engines.

---

# Interpretation

The benchmark intentionally reports two complementary likelihood evaluation strategies.

**Point-by-point NLL** represents the repeated objective-function evaluations encountered during likelihood scans and minimization and provides the primary apples-to-apples comparison across frameworks.

**Batched full-dataset evaluation** demonstrates the native vectorized execution model of pyHS3. Because RooFit evaluates events individually, this benchmark should be interpreted as a workflow comparison rather than a strict engine-to-engine microbenchmark.

Consequently, the two benchmark categories answer different questions:

- **Point-by-point NLL** measures equivalent statistical computations across execution engines.
- **Batched full-dataset evaluation** demonstrates the performance advantages achievable through vectorized execution.

Together they provide both a fair cross-framework comparison and a realistic picture of pyHS3 performance in modern statistical inference workflows.
