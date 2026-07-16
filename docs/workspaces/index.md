# Workspaces

On this page, you will learn what benchmark workspaces are, how they are generated, and where to find the complete workspace catalog.

Benchmark workspaces define the statistical models used throughout **PyHS3 Benchmarks**. They provide the common benchmark inputs used across workflow benchmarks, scalability studies, and cross-framework comparisons.

All benchmark workspaces are generated using the **workspace-scripts** repository:

https://github.com/scipp-atlas/workspace-scripts

---

## Workspace Collection

The repository uses a curated collection of benchmark workspaces derived from a common statistical model.

Workspace variants differ in

- channel count;
- background parameterization;
- signal model;
- nuisance-parameter configuration;
- constraint model;
- expected signal yield.

Using a shared benchmark dataset ensures that benchmark results remain reproducible and directly comparable across benchmark suites.

The complete catalog of benchmark workspaces, including filenames, supported frameworks, and intended use, is provided in **Benchmark Workspaces**.

---

## Supported Workspace Formats

The primary benchmark dataset consists of HS3 workspaces stored in

```text
inputs/
```

For cross-framework binned likelihood benchmarks, equivalent **pyhf** workspaces are provided in

```text
inputs/
└── pyhf/
```

These workspaces are generated from the same statistical models and are used exclusively by the **Cross-Framework Binned Likelihood** benchmark.

Equivalent workspace formats are generated using the **workspace-scripts** repository.

---

## Where to Go Next

- **Benchmark Workspaces** — complete workspace catalog and naming convention.
- **Workspace Lifecycle** — how a workspace is transformed inside PyHS3.
- **Cross-Framework Benchmarks** — framework comparisons using equivalent workspaces.
- **Benchmark Matrix Runner** — workspace selection and benchmark campaigns.
