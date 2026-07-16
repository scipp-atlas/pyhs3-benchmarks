# Quick Start

On this page, you will learn how to run your first benchmark, execute multiple benchmark suites, and understand the generated outputs.

This guide assumes that the project has already been installed. If not, begin with the **Installation** guide before continuing.

---

## Typical Benchmark Workflow

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

This workflow provides the quickest way to become familiar with the repository before exploring individual benchmark suites.

---

## Running Your First Benchmark

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

This command executes the benchmark, generates a structured JSON report, and creates comparison plots for the selected workspaces.

For details on what this benchmark measures, see **Workspace Loading**.

---

## Running Multiple Benchmarks

Once you are familiar with individual benchmark suites, you can execute several benchmarks in a single run.

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

The benchmark matrix runner executes each selected benchmark using the same workspace collection and organizes the generated outputs automatically.

For additional execution modes and campaign management, see **Benchmark Matrix Runner**.

---

## Running the Complete Benchmark Suite

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

## Understanding the Generated Outputs

Benchmark execution produces structured JSON reports together with benchmark plots.

Typical outputs include

```text
results/
└── workspace_loading/
    └── workspace_loading_result.json
```

and

```text
plots/
└── workspace_loading/
    ├── workspace_loading_wall_time.png
    ├── workspace_loading_current_rss_delta.png
    └── workspace_loading_peak_rss_delta.png
```

See **Outputs** for a complete description of generated reports and figures.

---

## Where to Go Next

After successfully running your first benchmark, the following pages provide a natural next step:

- **Benchmark Methodology** — understand how benchmarks are executed, measured, and validated.
- **Benchmark Suite** — explore individual benchmark implementations.
- **Benchmark Workspaces** — learn about the available benchmark workspaces.
- **Outputs** — inspect generated reports and figures.
- **Benchmark Matrix Runner** — run complete benchmark campaigns.
- **Development** — extend the repository with new benchmark suites.
