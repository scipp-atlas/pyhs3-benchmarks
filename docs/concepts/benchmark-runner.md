# Benchmark Runner

On this page, you will learn the role of the **Benchmark Runner** and how it coordinates benchmark execution across the repository.

The Benchmark Runner provides the shared execution infrastructure used by all benchmark suites. Rather than performing benchmark measurements itself, it orchestrates benchmark execution, coordinates benchmark campaigns, and ensures that benchmark suites follow a consistent execution model.

The runner is implemented in

```text
src/run_all_benchmarks.py
```

---

## Purpose

The repository contains many benchmark suites targeting different stages of the statistical inference workflow.

Instead of each benchmark implementing its own execution infrastructure, the Benchmark Runner provides a common framework for

- coordinating benchmark execution;
- managing benchmark campaigns;
- sharing benchmark configuration;
- collecting benchmark outputs;
- generating campaign summaries.

This shared infrastructure keeps benchmark implementations focused on measurement rather than execution management.

---

## Execution Model

Every benchmark campaign follows the same high-level execution flow.

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

The Benchmark Runner coordinates this workflow while allowing each benchmark suite to execute independently.

---

## Benchmark Campaigns

A benchmark campaign combines

- one or more benchmark suites;
- one or more benchmark workspaces;
- a shared execution configuration.

Using a single configuration across an entire campaign ensures that benchmark results remain directly comparable across benchmark suites and workspace collections.

For practical examples of configuring and executing benchmark campaigns, see **Benchmark Matrix Runner**.

---

## Workspace Coordination

The Benchmark Runner provides every benchmark with the same workspace collection.

This guarantees that benchmark suites operate on consistent benchmark inputs, simplifying comparison across workflow stages and scalability studies.

Details on workspace discovery, filtering, and selection are documented in **Benchmark Matrix Runner**.

---

## Result Collection

Each benchmark remains responsible for generating its own benchmark report.

The Benchmark Runner collects these independent outputs into a single benchmark campaign while preserving the separation between benchmark suites.

Generated reports and figures are documented in **Outputs**.

---

## Plot Coordination

The Benchmark Runner coordinates figure generation after benchmark execution completes.

Benchmark measurements and visualization remain separate processes, allowing plots to be regenerated without repeating benchmark execution.

See **Outputs** for generated artifacts and **Benchmark Methodology** for the measurement strategy.

---

## Error Isolation

Benchmark suites execute independently.

If one benchmark fails, completed benchmark results remain available while failures are recorded as part of the benchmark campaign summary.

This design simplifies debugging and allows long-running benchmark campaigns to continue whenever possible.

---

## Relationship to Individual Benchmarks

Individual benchmark implementations and the Benchmark Runner have different responsibilities.

| Individual Benchmark | Benchmark Runner |
|----------------------|------------------|
| Performs the measured computation | Coordinates benchmark execution |
| Collects benchmark statistics | Manages benchmark campaigns |
| Generates benchmark reports | Collects campaign outputs |
| Implements benchmark-specific logic | Provides shared execution infrastructure |

This separation keeps benchmark implementations simple while allowing the repository to scale to many benchmark suites.

---

## Related Documentation

See also

- **Benchmark Matrix Runner**
- **Benchmark Methodology**
- **Outputs**
- **Development**
- **Repository Structure**
