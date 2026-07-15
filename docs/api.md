# API Reference

On this page, you will learn which interfaces are exposed by **PyHS3 Benchmarks** and which modules are considered internal implementation details.

PyHS3 Benchmarks is primarily a command-line benchmarking framework rather than a traditional Python library. Most functionality is exposed through benchmark commands and structured benchmark outputs instead of a stable public Python API.

---

## Repository Interfaces

The repository exposes four primary interfaces.

| Interface | Purpose |
|-----------|---------|
| Command-line benchmarks | Execute individual benchmark suites |
| Benchmark runner | Execute complete benchmark campaigns |
| JSON reports | Machine-readable benchmark results |
| Generated figures | Human-readable performance visualizations |

These interfaces support interactive benchmarking, automated performance analysis, and reproducible reporting.

---

## Command-Line Interface

Individual benchmark suites can be executed using

```bash
python -m src.run_<benchmark>
```

See **Getting Started** for benchmark-specific examples and common command-line options.

---

## Benchmark Runner

Complete benchmark campaigns can be executed using

```bash
python -m src.run_all_benchmarks
```

The benchmark runner coordinates benchmark execution, workspace management, report generation, and plot generation.

See **Benchmark Runner** for implementation details.

---

## Repository Modules

Several modules provide shared functionality across benchmark suites.

### Configuration

```text
src/config.py
```

Provides repository-wide configuration, including benchmark directories, output locations, plotting paths, and shared constants.

### Benchmark Stages

```text
src/benchmark_stages.py
```

Defines the common representation of workflow stages used throughout the benchmarking pipeline.

### Utilities

```text
src/utils.py
```

Provides shared helper functions reused across benchmark implementations.

---

## Benchmark Outputs

Benchmark results are exposed through structured JSON reports together with generated figures.

See **Outputs** for the report format and generated artifacts.

---

## Internal API

Most modules inside `src/` are implementation details rather than a stable public Python API.

Although they can be imported directly, users are encouraged to interact with the repository through the documented command-line interfaces instead of relying on internal modules.

---

## Related Documentation

See also

- **Getting Started**
- **Benchmark Runner**
- **Benchmark Methodology**
- **Outputs**
- **Development**
