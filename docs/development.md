# Development

This guide describes the development workflow used throughout the PyHS3 Benchmarks project.

The repository is designed around a modular benchmarking architecture in which benchmark implementations, reporting, plotting, and documentation remain largely independent. This separation simplifies maintenance while making it straightforward to extend the benchmark suite.

---

# Development Philosophy

Development follows a few simple principles.

- Every benchmark measures one well-defined workflow stage.
- Benchmark implementations remain independent.
- Common infrastructure is reused whenever possible.
- Results are generated in a consistent format.
- Documentation is generated from reproducible benchmark outputs.

Keeping benchmarks small and independent makes it easier to maintain the repository while allowing new benchmark suites to be added without affecting existing workflows.

---

# Repository Architecture

Most benchmark implementations follow the same execution model.

```text
Workspace
      │
      ▼
run_<benchmark>.py
      │
      ▼
JSON Report
      │
      ▼
Plots
      │
      ▼
Documentation
```

Each benchmark is implemented as an executable Python module while sharing common infrastructure for reporting, plotting, configuration, and benchmark execution.

---

# Implementing a New Benchmark

A new benchmark typically consists of

- a new executable module under `src/`;
- benchmark-specific measurement logic;
- JSON result generation;
- optional plot generation;
- documentation under `docs/benchmarks/`;
- automated tests.

Whenever possible, new benchmark implementations should reuse the existing reporting and plotting infrastructure rather than introducing benchmark-specific output formats.

---

# Registering the Benchmark

To make a benchmark available through the benchmark runner,

- add the benchmark implementation;
- register it with the benchmark runner;
- ensure that benchmark outputs follow the standard reporting conventions.

Once registered, the benchmark can be executed individually or as part of a larger benchmark campaign.

---

# Result Generation

Benchmark implementations should generate structured JSON outputs rather than directly producing figures.

JSON reports act as the canonical representation of benchmark results and provide the input for

- visualization;
- regression analysis;
- documentation;
- automated reporting.

Separating measurement from visualization improves reproducibility and simplifies development.

---

# Plot Generation

Plots should be generated from benchmark reports instead of directly from benchmark execution.

This allows

- figures to be regenerated without rerunning benchmarks;
- plotting improvements without repeating measurements;
- reproducible documentation updates.

---

# Testing

Every benchmark should include automated tests covering

- successful execution;
- expected benchmark outputs;
- numerical validation where applicable;
- benchmark-specific functionality.

Shared infrastructure should be tested independently of individual benchmark implementations whenever possible.

---

# Documentation

Each benchmark should include corresponding documentation describing

- benchmark purpose;
- measured quantities;
- benchmark configuration;
- generated outputs;
- interpretation of benchmark results.

Keeping implementation and documentation synchronized helps ensure that benchmark behavior remains understandable as the project evolves.

---

# Typical Development Workflow

Most development follows the same sequence.

```text
Implement Benchmark
        │
        ▼
Execute Benchmark
        │
        ▼
Inspect JSON Results
        │
        ▼
Generate Plots
        │
        ▼
Update Documentation
        │
        ▼
Run Tests
```

Following this workflow helps maintain consistency across benchmark implementations and simplifies long-term maintenance.

---

# Related Documentation

See also

- **Benchmark Methodology**
- **Benchmark Runner**
- **Benchmark Results**
- **Repository Structure**
