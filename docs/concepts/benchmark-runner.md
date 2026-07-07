# Benchmark Runner

The **Benchmark Runner** provides the common execution infrastructure used throughout the PyHS3 Benchmarks repository.

Rather than executing benchmark suites independently, the benchmark runner coordinates benchmark execution, manages benchmark campaigns, collects results, and optionally generates visualizations. This shared infrastructure ensures that all benchmark suites follow the same execution model, produce consistent outputs, and can be executed together as part of large-scale performance studies.

The benchmark runner is implemented in `src/run_all_benchmarks.py`.

---

# Purpose

The repository contains many benchmark suites that evaluate different stages of the PyHS3 statistical inference workflow.

Although each benchmark measures a different operation, they all share the same high-level execution pattern.

The benchmark runner exists to provide a single interface for

- executing multiple benchmark suites;
- managing collections of benchmark workspaces;
- coordinating benchmark execution;
- collecting benchmark outputs;
- generating benchmark summaries;
- producing publication-quality plots.

Without a shared runner, every benchmark would need to implement its own execution logic, output management, and reporting infrastructure.

---

# Execution Model

The benchmark runner follows a simple execution model.

```text
Benchmark Configuration
          │
          ▼
Select Benchmark Suites
          │
          ▼
Load Benchmark Workspaces
          │
          ▼
Execute Benchmarks
          │
          ▼
Collect Results
          │
          ▼
Generate Summary
          │
          ▼
Generate Plots (optional)
```

Each benchmark executes independently while the runner manages the overall benchmark campaign.

---

# Benchmark Discovery

Each workflow benchmark is implemented as an independent executable module under `src/`.

Typical benchmark entry points include

```text
run_workspace_loading.py
run_model_creation.py
run_log_prob_construction.py
run_graph_canonicalization.py
run_graph_optimization.py
run_log_prob_compilation.py
run_compiled_evaluation.py
run_pdf_evaluation.py
run_nll_scan.py
run_memory_scaling.py
run_model_complexity_scaling.py
```

Because benchmark implementations remain independent, they can be executed either directly or through the benchmark runner.

This separation keeps individual benchmark implementations focused on measurement while allowing the runner to coordinate complete benchmark campaigns.

---

# Benchmark Campaigns

A benchmark campaign consists of

- one or more benchmark suites;
- one or more benchmark workspaces;
- a shared execution configuration.

For example,

```text
Benchmarks
    Workspace Loading
    Model Creation
    PDF Evaluation

Workspaces
    1-channel
    5-channel
    30-channel
```

The benchmark runner automatically executes every requested benchmark for every selected workspace.

Using the same benchmark configuration throughout the campaign ensures that benchmark results remain directly comparable.

---

# Workspace Management

The benchmark runner manages benchmark workspaces centrally.

Each benchmark receives the same workspace collection, avoiding inconsistencies between benchmark suites.

This approach provides

- reproducible benchmark campaigns;
- consistent benchmark inputs;
- straightforward scalability studies;
- comparable performance measurements.

---

# Result Collection

After each benchmark completes, the runner collects the generated benchmark reports.

Individual benchmark outputs remain independent and are stored within their corresponding result directories.

The runner additionally produces a benchmark campaign summary describing

- executed benchmark suites;
- processed benchmark workspaces;
- execution status;
- generated result locations.

This summary provides a convenient overview of the entire benchmark campaign.

---

# Plot Generation

When plotting is enabled, benchmark figures are generated from the produced benchmark reports.

The runner coordinates plot generation but does not perform any numerical measurements itself.

Separating measurement from visualization provides several advantages.

- Benchmark execution remains deterministic.
- Figures can be regenerated without rerunning benchmarks.
- Plotting improvements do not require repeating expensive benchmark campaigns.

---

# Error Handling

Benchmark failures are isolated.

If one benchmark encounters an error, previously completed benchmark results remain available.

Whenever possible, the runner records benchmark status together with diagnostic information so that failed benchmark executions can be investigated without repeating the complete benchmark campaign.

This behavior is particularly useful during benchmark development and large performance studies.

---

# Why Use the Benchmark Runner?

Individual benchmark modules are useful during development and debugging.

For routine performance evaluation, however, the benchmark runner provides significant advantages.

- A single command executes complete benchmark campaigns.
- Benchmark configuration remains consistent across benchmark suites.
- Outputs follow a common directory structure.
- Result aggregation is performed automatically.
- Plot generation is integrated into the execution workflow.

This shared infrastructure greatly simplifies repository maintenance while ensuring consistent benchmark execution.

---

# Relationship to Individual Benchmarks

The benchmark runner orchestrates benchmark execution but does not replace individual benchmark implementations.

Each benchmark remains responsible for

- performing the measured computation;
- collecting benchmark statistics;
- validating benchmark outputs;
- generating its own benchmark report.

The benchmark runner coordinates these independent components into a single, reproducible workflow.

---

# Typical Development Workflow

During development, benchmark implementations are usually executed directly until they behave as expected.

```text
Develop Benchmark
        │
        ▼
Run Individual Benchmark
        │
        ▼
Inspect Results
        │
        ▼
Register with Benchmark Runner
        │
        ▼
Execute Benchmark Campaign
```

This iterative workflow simplifies development while ensuring that newly implemented benchmarks integrate naturally with the rest of the repository.

---

# Related Documentation

See also

- **Benchmark Workflow** for the benchmark execution lifecycle.
- **Workspace Lifecycle** for the lifecycle of benchmark workspaces.
- **Benchmark Methodology** for the measurement strategy used throughout the repository.
- **Benchmark Results** for the generated reports and benchmark artifacts.
- **Development** for extending the benchmark suite.
