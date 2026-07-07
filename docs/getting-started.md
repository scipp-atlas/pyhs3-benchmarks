# Getting Started

This guide introduces the basic workflow for using the **PyHS3 Benchmarks** repository.

By the end of this guide, you will be able to

- install and configure the benchmarking environment;
- execute your first benchmark;
- run multiple benchmark suites;
- understand the generated outputs;
- navigate the project documentation.

If you have not yet installed the project, begin with the **Installation** guide before continuing.

---

# Typical Benchmark Workflow

A typical benchmarking session consists of four stages.

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
Analyze Performance
```

Most benchmark workflows in this repository follow this sequence. The benchmark runner automates benchmark execution, report generation, and plot creation, allowing individual benchmarks to be executed consistently across different workspaces.

---

# Running Your First Benchmark

The recommended first benchmark is **Workspace Loading**, which measures the cost of loading HS3 workspaces before any statistical models are constructed.

Run

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
- measures execution time and memory consumption;
- generates a structured JSON report;
- creates comparison plots for the selected workspaces.

For a detailed description of this benchmark, see **Workspace Loading**.

---

# Running Multiple Benchmarks

Once you are familiar with individual benchmarks, the recommended workflow is to use the benchmark runner.

For example,

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

The benchmark runner executes each requested benchmark using the same workspace collection, stores the generated reports, and optionally produces comparison plots.

---

# Running the Complete Benchmark Suite

To execute every available workflow benchmark for the selected workspaces, run

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

This approach is recommended for complete benchmark campaigns and performance regression studies.

---

# Understanding the Generated Outputs

Every benchmark produces machine-readable outputs that can be inspected directly or used to generate visualizations.

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

The exact files depend on the benchmark being executed.

A detailed description of benchmark outputs is provided in **Benchmark Results**.

---

# Typical Development Workflow

During benchmark development, the recommended workflow is

```text
Modify Benchmark
        │
        ▼
Execute Benchmark
        │
        ▼
Inspect JSON Results
        │
        ▼
Review Generated Plots
        │
        ▼
Evaluate Performance
```

Following the same workflow throughout the repository helps ensure that benchmark results remain reproducible and directly comparable across different revisions.

---

# Where to Go Next

After successfully running your first benchmark, the following pages provide a natural next step:

- **Installation** — managing and updating the project environment.
- **Repository Structure** — understanding the organization of the repository.
- **Benchmark Methodology** — benchmarking principles and measurement strategy.
- **Benchmark Suite** — overview of all available workflow benchmarks.
- **Benchmark Results** — generated reports and visualization outputs.
- **Workspace Loading** — detailed documentation for the first workflow benchmark.
