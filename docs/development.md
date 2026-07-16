# Development

On this page, you will learn how to extend **PyHS3 Benchmarks** by implementing, registering, testing, and documenting new benchmark suites.

The repository is built around a modular benchmarking architecture where benchmark implementations, reporting, plotting, and documentation remain largely independent. This separation simplifies maintenance while making it straightforward to extend the benchmark suite.

---

## Development Principles

Development follows a few simple principles.

- Reuse shared infrastructure whenever possible.
- Keep benchmark implementations independent.
- Generate benchmark results in a consistent format.
- Build documentation from reproducible benchmark outputs.

For details on benchmark execution and measurement, see **Benchmark Methodology**.

---

## Repository Architecture

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

Each benchmark is implemented as an executable Python module while sharing common infrastructure for configuration, reporting, plotting, and execution.

---

## Implementing a New Benchmark

A new benchmark typically consists of

- a new executable module under `src/`;
- benchmark-specific measurement logic;
- JSON result generation;
- optional plot generation;
- documentation under `docs/benchmarks/`;
- automated tests.

Whenever possible, benchmark implementations should reuse the existing infrastructure instead of introducing benchmark-specific output formats or utilities.

---

## Registering a Benchmark

To make a benchmark available through the benchmark matrix runner,

- add the benchmark implementation;
- register it in the benchmark registry;
- ensure that benchmark outputs follow the standard reporting conventions.

Once registered, the benchmark can be executed individually or as part of a benchmark campaign.

---

## Benchmark Outputs

Benchmark implementations should generate structured JSON reports as the canonical benchmark output.

These reports provide the input for

- visualization;
- regression analysis;
- documentation;
- automated reporting.

See **Outputs** for the report format and generated artifacts.

---

## Plot Generation

Plots should be generated from benchmark reports rather than directly from benchmark execution.

This allows

- figures to be regenerated without rerunning benchmarks;
- plotting improvements without repeating measurements;
- reproducible documentation updates.

---

## Testing

Every benchmark should include automated tests covering

- successful execution;
- expected benchmark outputs;
- numerical validation where applicable;
- benchmark-specific functionality.

Shared infrastructure should be tested independently whenever possible.

---

## Documentation

Every benchmark should include corresponding user-facing documentation that is consistent with the repository documentation style.

Documentation should clearly describe

- what the benchmark measures;
- how to execute it;
- generated outputs;
- how to interpret the results.

---

## Typical Development Workflow

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

This workflow keeps benchmark implementations, generated outputs, and documentation synchronized.

---

## Related Documentation

See also

- **Benchmark Methodology**
- **Benchmark Matrix Runner**
- **API Reference**
- **Outputs**
- **Repository Structure**
