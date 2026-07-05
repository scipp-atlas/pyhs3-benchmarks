# Getting Started

This guide introduces the basic workflow for working with the PyHS3 Benchmarks repository.

By the end of this guide you will be able to

- run your first benchmark;
- execute multiple benchmark suites;
- understand the generated outputs;
- generate benchmark plots;
- navigate the repository documentation.

If you have not yet installed the project, see the **Installation** guide before continuing.

---

# Benchmark Workflow

A typical benchmarking session consists of four steps.

```text
Install Dependencies
        │
        ▼
Run Benchmarks
        │
        ▼
Inspect Results
        │
        ▼
Analyze Plots
```

The repository automates this workflow through the benchmark runner and the built-in plotting utilities.

---

# Running Your First Benchmark

The simplest way to become familiar with the repository is to execute a single benchmark.

```bash
pixi run python -m src.run_workspace_loading \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --n-runs 30 \
    --plot
```

This command

- benchmarks workspace loading performance;
- measures execution time and memory usage;
- generates a JSON report;
- produces comparison plots for the selected workspaces.

For a complete description of this benchmark, see **Workspace Loading**.

---

# Running Multiple Benchmarks

The recommended way to execute several benchmark suites is to use the benchmark runner.

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks \
        workspace_loading \
        model_creation \
        compiled_evaluation \
    --n-runs 30 \
    --plot
```

The benchmark runner executes each selected benchmark, stores the generated JSON reports, and produces the corresponding comparison plots.

---

# Running the Complete Benchmark Suite

To execute every available workflow benchmark on the selected workspaces, run

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks all \
    --n-runs 30 \
    --plot
```

This is the recommended approach for comprehensive benchmarking campaigns and performance regression testing.

---

# Benchmark Outputs

Benchmark execution produces structured JSON reports together with optional plots.

Typical outputs include

```text
results/
└── workspace_loading/
    └── workspace_loading_result.json
```

and

```text
docs/
└── assets/
    └── plots/
        └── workspace_loading/
            ├── workspace_loading_wall_time.png
            ├── workspace_loading_current_rss_delta.png
            └── workspace_loading_peak_rss_delta.png
```

The exact outputs depend on the selected benchmark.

For a complete description of the generated artifacts, see **Benchmark Results**.

---

# Typical Development Workflow

The recommended workflow during benchmark development is

```text
Modify Benchmark
        │
        ▼
Run Benchmark
        │
        ▼
Inspect JSON Results
        │
        ▼
Review Generated Plots
        │
        ▼
Verify Performance
```

Using the same workflow throughout the repository ensures that benchmark results remain reproducible and directly comparable.

---

# Where to Go Next

Once you have successfully executed your first benchmark, the following pages provide more detailed information:

- **Benchmark Suite** — overview of all available benchmarks.
- **Benchmark Methodology** — benchmarking principles and measurement methodology.
- **Workspace Loading** — detailed documentation for the first workflow benchmark.
- **Benchmark Results** — generated JSON reports and plots.
- **Repository Structure** — project organization and directory layout.
