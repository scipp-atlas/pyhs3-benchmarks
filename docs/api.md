# API Reference

The PyHS3 Benchmarks repository is primarily a command-line benchmarking framework rather than a traditional Python library.

Most functionality is exposed through executable benchmark modules instead of a public Python API.

This page provides an overview of the repository interfaces available to users and developers.

---

# Command-Line Interface

The primary interface is the command-line benchmark runner.

Individual benchmark suites can be executed using

```bash
python -m src.run_<benchmark>
```

Examples include

```bash
python -m src.run_workspace_loading

python -m src.run_model_creation

python -m src.run_pdf_evaluation

python -m src.run_compiled_evaluation
```

Each benchmark accepts its own command-line options while sharing a common execution model and reporting infrastructure.

---

# Benchmark Runner

Complete benchmark campaigns are executed using

```bash
python -m src.run_all_benchmarks
```

The benchmark runner coordinates

- benchmark execution;
- workspace management;
- report generation;
- plot generation.

See **Benchmark Runner** for details.

---

# Configuration

Shared configuration is provided through

```text
src/config.py
```

This module defines repository-wide configuration such as

- benchmark directories;
- output locations;
- plotting paths;
- repository constants.

Benchmark implementations reuse this configuration to ensure consistent behavior across the repository.

---

# Benchmark Stages

Workflow stages are defined in

```text
src/benchmark_stages.py
```

This module provides a common representation of benchmark stages used throughout the benchmarking pipeline.

---

# Utilities

Shared helper functions are implemented in

```text
src/utils.py
```

These utilities are reused across benchmark implementations to avoid duplication and maintain consistent behavior.

---

# Result Format

Benchmark outputs are exposed through structured JSON reports.

These reports provide the primary programmatic interface for downstream analysis, visualization, and automated reporting.

See **Benchmark Results** for the report format and generated artifacts.

---

# Public Interfaces

The repository exposes four primary interfaces.

| Interface | Purpose |
|-----------|---------|
| Command-line benchmarks | Execute individual benchmark suites |
| Benchmark runner | Execute complete benchmark campaigns |
| JSON reports | Machine-readable benchmark results |
| Generated figures | Human-readable performance visualizations |

Together these interfaces support interactive benchmarking, automated performance analysis, and reproducible documentation.

---

# Internal Modules

Most modules inside `src/` are considered implementation details.

Although they can be imported directly, they are primarily intended to support the command-line benchmark infrastructure rather than provide a stable public Python API.

Users are therefore encouraged to interact with the repository through the documented benchmark commands instead of relying on internal implementation modules.

---

# Related Documentation

See also

- **Getting Started**
- **Benchmark Runner**
- **Benchmark Methodology**
- **Benchmark Results**
- **Development**
