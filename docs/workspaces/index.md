# Workspaces

Benchmark workspaces define the statistical models used throughout the PyHS3 Benchmarks repository.

The repository is built around a single canonical benchmark workspace collection that serves as the common input for nearly every benchmark suite. Using one shared dataset ensures that performance measurements remain reproducible, directly comparable, and representative across the entire benchmarking framework.

All benchmark workspaces are generated automatically using the **workspace-scripts** repository:

https://github.com/scipp-atlas/workspace-scripts

The generation process produces statistically equivalent **HS3** and **ROOT** workspaces from the same underlying statistical models. These paired workspaces form the benchmark dataset used throughout the repository.

---

# Benchmark Workspace Collection

The benchmark workspace collection spans a range of statistical models with varying complexity.

The collection includes variations in

- analysis channel count;
- background parameterization;
- signal parameterization;
- nuisance parameter configuration;
- auxiliary constraint model;
- expected signal yield.

Every workspace follows the same naming convention and is generated from a common baseline statistical model.

See **Benchmark Workspaces** for a complete description of the dataset design and naming convention.

---

# Why a Common Benchmark Dataset?

Using a single benchmark workspace collection provides several advantages.

- reproducible benchmark inputs;
- consistent benchmark configuration;
- directly comparable benchmark results;
- simplified benchmark maintenance;
- statistically equivalent HS3 and ROOT models for cross-framework validation.

As a result, observed benchmark differences reflect implementation characteristics rather than differences in benchmark inputs.

---

# Relationship to Cross-Framework Benchmarks

Cross-framework benchmarks operate on matching HS3 and ROOT workspaces generated from the same statistical models.

Using paired workspaces allows PyHS3 and ROOT-based frameworks to evaluate equivalent likelihoods while maintaining an apples-to-apples comparison methodology.

---

# Related Documentation

- Benchmark Workspaces
- Benchmark Methodology
- Cross-Framework Benchmarks
