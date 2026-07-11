# Cross-Framework Benchmarks

The **Cross-Framework** benchmark suite evaluates equivalent statistical computations across multiple statistical inference frameworks.

Unlike the workflow benchmarks, which measure the internal performance of PyHS3, the benchmarks in this section compare PyHS3 with external implementations while ensuring that every framework performs the **same statistical computation**.

The primary goals of these benchmarks are to

- validate numerical agreement between implementations;
- compare performance under equivalent conditions;
- identify implementation-specific performance characteristics;
- ensure that optimization does not compromise statistical correctness.

All comparisons follow an **apples-to-apples** methodology using matching benchmark workspaces, identical parameter values, identical scan configurations, and equivalent statistical models.

---

# Philosophy

Cross-framework benchmarking is fundamentally different from traditional performance benchmarking.

Comparing two statistical frameworks is meaningful only if they evaluate the **same mathematical quantity** using the **same statistical model**.

For this reason, every benchmark in this section is designed to eliminate differences arising from

- model construction;
- workspace contents;
- parameter initialization;
- observed datasets;
- benchmark configuration.

Only after these quantities have been aligned are execution time, memory consumption, or numerical agreement compared.

---

# Common Benchmark Dataset

All cross-framework benchmarks operate on the canonical benchmark workspace collection described in **Benchmark Workspaces**.

Each HS3 workspace has a corresponding ROOT workspace generated from the same statistical model using the `workspace-scripts` repository.

This ensures that every framework evaluates statistically equivalent models and that observed differences originate from the framework implementation rather than differences in the benchmark inputs.

---

# Current Benchmark Suites

The repository currently includes the following cross-framework benchmarks.

## Scalar PDF Evaluation

Compares repeated normalized scalar probability density function evaluation between PyHS3 and RooFit.

This benchmark measures

- cold-start latency;
- warm evaluation latency;
- throughput;
- resident memory usage;
- numerical agreement.

Unlike the compiled evaluation benchmark, this comparison intentionally evaluates only eager scalar PDF execution to preserve an apples-to-apples comparison.

---

## ΔNLL Scan

Compares the evaluation of identical ΔNLL scans using matching HS3 and ROOT workspaces.

Both frameworks

- evaluate the same observed events;
- scan the same parameter of interest;
- use identical parameter values;
- compute the same likelihood quantity.

The benchmark reports both numerical agreement and performance characteristics.

---

## xRooFit Benchmark

Compares complete likelihood evaluation workflows between **PyHS3** and **xRooFit** using matching HS3 and ROOT workspaces.

The benchmark evaluates

- PyHS3 non-compiled execution;
- PyHS3 compiled execution;
- xRooFit NLL evaluation.

All engines execute an equivalent ΔNLL scan over the same parameter of interest using identical datasets, scan ranges, and benchmark configurations.

Besides numerical validation, the benchmark separately reports

- workspace loading;
- model construction;
- NLL construction;
- first (cold) evaluation;
- steady-state evaluation;
- complete scan performance;
- memory usage.

This benchmark therefore provides both an engine-to-engine performance comparison and a numerical validation of equivalent statistical workflows.

---

# Apples-to-Apples Methodology

Every benchmark in this section follows the same validation principles.

Whenever possible, all compared frameworks use

- identical statistical models;
- identical observed datasets;
- identical parameter values;
- identical scan grids;
- identical benchmark configurations;
- equivalent mathematical definitions.

This methodology minimizes systematic differences unrelated to implementation and allows benchmark results to be interpreted with confidence.

Where an exact one-to-one comparison is not possible, the corresponding benchmark documentation explicitly describes the assumptions and limitations.

---

# Numerical Validation

Performance measurements alone are not sufficient for cross-framework benchmarking.

Every benchmark therefore validates that the compared frameworks produce numerically equivalent results before drawing conclusions about performance.

Depending on the benchmark, validation may include

- scalar PDF values;
- likelihood values;
- ΔNLL profiles;
- point-by-point residuals;
- agreement within predefined numerical tolerances.

Only validated benchmark results should be interpreted as meaningful performance comparisons.

---

# Why Cross-Framework Benchmarks?

Cross-framework benchmarking provides several important benefits.

- Independent verification of PyHS3 implementations.
- Confidence that optimizations preserve numerical correctness.
- Identification of performance trade-offs across frameworks.
- Reproducible comparisons using common benchmark inputs.
- Transparent reporting of both performance and numerical agreement.

Together, these benchmarks complement the workflow benchmarks by demonstrating not only how fast PyHS3 performs, but also how its results compare with established statistical frameworks.

---

# Available Documentation

This section currently includes

- **Scalar PDF Evaluation**
- **ΔNLL Scan**
- **xRooFit Benchmark**

Each benchmark page describes the benchmark methodology, execution procedure, generated outputs, validation strategy, and interpretation of the reported results.

---

# Related Documentation

See also

- **Benchmark Methodology**
- **Benchmark Workspaces**
- **Benchmark Results**
- **Outputs**
