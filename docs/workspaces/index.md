# Benchmark Workspaces

Benchmark workspaces define the statistical models used throughout the PyHS3 Benchmarks repository.

The repository is built around a single benchmark workspace collection that serves as the primary input for nearly every benchmark suite. Using a unified dataset ensures that performance measurements remain reproducible, directly comparable, and representative across the entire benchmarking framework.

Rather than maintaining separate benchmark datasets for different workflow stages, the repository uses the same workspace collection whenever possible. This allows benchmark results from different suites to be interpreted consistently and enables meaningful comparisons between workflow stages and statistical frameworks.

---

# Workspace Collections

The repository currently contains two workspace collections.

## Benchmark Workspaces

The primary benchmark dataset.

These workspaces are used by virtually every workflow benchmark, scalability benchmark, and cross-framework benchmark implemented in the repository.

The collection spans a wide range of statistical models with varying complexity, including different numbers of channels, signal parameterizations, background models, nuisance parameter configurations, and constraint models.

See **Benchmark Workspaces** for a detailed description.

---

## PyHF Workspaces

The repository also contains benchmark inputs targeting the PyHF ecosystem.

These workspaces are documented separately because their structure and intended use differ from the primary benchmark workspace collection.

See **PyHF Workspaces** for additional details.

---

# Why a Single Benchmark Dataset?

Using one benchmark workspace collection throughout the repository provides several advantages.

- consistent benchmark inputs across workflow stages;
- reproducible performance measurements;
- directly comparable benchmark results;
- simplified benchmark maintenance;
- identical statistical models for cross-framework validation.

As a result, benchmark differences reflect implementation differences rather than differences in benchmark inputs.

---

# Matrix Benchmarking

The matrix runner automatically discovers compatible benchmark workspaces and executes benchmark suites across the complete collection.

This enables large-scale benchmark campaigns with a single command while preserving a consistent directory structure for benchmark outputs.

---

# Related Documentation

- Benchmark Workspaces
- PyHF Workspaces
