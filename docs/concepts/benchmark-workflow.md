# Benchmark Workflow

Every benchmark in the PyHS3 Benchmarks repository follows a common execution workflow.

Although individual benchmark suites measure different stages of the statistical inference pipeline, they all share the same high-level execution model. This consistency simplifies benchmark development, ensures comparable outputs, and enables the benchmark runner to execute every benchmark using the same infrastructure.

---

# Workflow Overview

A typical benchmark execution consists of the following stages.

```text
Benchmark Configuration
        │
        ▼
Load Benchmark Workspace
        │
        ▼
Prepare Benchmark State
        │
        ▼
Execute Benchmark
        │
        ▼
Validate Results
        │
        ▼
Generate JSON Report
        │
        ▼
Generate Plots (optional)
```

Each stage has a clearly defined responsibility, making the benchmark pipeline predictable and reproducible.

---

# Benchmark Configuration

Every benchmark begins by reading its execution configuration.

Typical configuration parameters include

- benchmark workspaces;
- benchmark targets;
- execution mode;
- number of benchmark iterations;
- output directory;
- plotting options.

Individual benchmark suites may introduce additional benchmark-specific parameters while preserving the same overall execution model.

---

# Workspace Preparation

The benchmark loads one or more HS3 workspaces from the `inputs/` directory.

Depending on the benchmark, the workspace may then be used to

- construct a statistical model;
- build a symbolic log-probability graph;
- optimize or compile the graph;
- evaluate the compiled model.

The exact preparation steps depend on the benchmark being executed.

---

# Benchmark Execution

Once the benchmark state has been prepared, the benchmark performs the operation it is designed to measure.

Examples include

- loading an HS3 workspace;
- constructing a statistical model;
- building a symbolic graph;
- compiling the graph;
- evaluating a compiled likelihood;
- performing an NLL scan.

Each benchmark focuses on a single workflow stage whenever possible. This isolation makes it easier to identify performance bottlenecks and evaluate optimization strategies.

---

# Result Validation

Before benchmark results are recorded, benchmark-specific validation is performed.

Depending on the benchmark, validation may verify

- successful execution;
- finite numerical outputs;
- valid symbolic graphs;
- successful compilation;
- numerical agreement with reference values.

Validation ensures that reported performance measurements correspond to successful and meaningful computations.

---

# Result Generation

Successful benchmark executions produce a structured JSON report.

Benchmark reports typically contain

- benchmark metadata;
- execution configuration;
- timing statistics;
- memory statistics;
- validation results;
- benchmark-specific measurements.

These reports provide a consistent interface for downstream analysis and visualization.

---

# Plot Generation

Most benchmark suites can optionally generate publication-quality figures from the produced JSON reports.

Typical figures include

- execution time comparisons;
- memory profiles;
- throughput measurements;
- scaling studies;
- likelihood scan visualizations.

Because figures are generated from benchmark reports rather than directly from benchmark execution, they can be reproduced without repeating expensive computations.

---

# Workflow Variations

Although every benchmark follows the same overall execution model, different benchmark suites measure different workflow stages.

For example,

- **Workspace Loading** measures only workspace deserialization.
- **Model Creation** begins with an already loaded workspace.
- **Compiled Evaluation** assumes that graph construction and compilation have already completed.
- **Cross-framework benchmarks** execute equivalent workflows across multiple statistical frameworks.

Each benchmark page documents any workflow stages that are intentionally excluded from its measurements.

---

# Why a Common Workflow?

Using a shared execution workflow provides several advantages.

- Benchmark implementations remain consistent.
- Benchmark outputs follow a common structure.
- Plot generation can be shared across benchmark suites.
- The benchmark runner can execute all benchmark suites using the same infrastructure.
- New benchmark suites can be integrated with minimal additional code.

This common workflow is one of the key design principles of the PyHS3 Benchmarks repository.

---

# Related Documentation

For more information, see

- **Benchmark Methodology** for the measurement strategy and benchmarking principles.
- **Workspace Lifecycle** for the lifecycle of an HS3 workspace during benchmark execution.
- **Benchmark Runner** for the shared benchmark execution infrastructure.
- **Benchmark Results** for generated reports and visualization outputs.
