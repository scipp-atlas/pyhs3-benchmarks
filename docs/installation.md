# Installation

On this page, you will learn how to install **PyHS3 Benchmarks**, verify the installation, and prepare a reproducible benchmarking environment.

PyHS3 Benchmarks uses **Pixi** to provide a reproducible software environment with all dependencies required to execute the benchmark suite.

---

## Prerequisites

Before installing the repository, ensure that the following tools are available.

- **Git**
- **Pixi**

All Python packages and project dependencies are installed automatically by Pixi.

---

## Installation Overview

Installing the repository consists of four steps.

1. Clone the repository.
2. Create the project environment.
3. Verify the installation.
4. Confirm that benchmark execution works correctly.

---

## Clone the Repository

Clone the repository from GitHub.

```bash
git clone https://github.com/scipp-atlas/pyhs3-benchmarks.git

cd pyhs3-benchmarks
```

---

## Create the Benchmark Environment

Create the project environment.

```bash
pixi install
```

This command installs

- PyHS3 Benchmarks;
- PyHS3 and its dependencies;
- numerical computation libraries;
- plotting libraries;
- benchmarking utilities.

The environment is managed entirely by Pixi, ensuring that benchmark results are reproducible across systems.

---

## Verify the Installation

Run a simple benchmark to verify that the environment was created successfully.

```bash
pixi run python -m src.run_workspace_loading \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --n-runs 10 \
    --plot
```

A successful execution loads the selected workspaces, executes the benchmark, generates a JSON report, and creates benchmark plots.

For details on benchmark execution, see **Getting Started**.

---

## Verify the Benchmark Runner

After confirming that an individual benchmark executes correctly, verify the benchmark runner.

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks workspace_loading \
    --n-runs 10 \
    --plot
```

This confirms that the shared benchmark infrastructure is configured correctly.

For advanced execution modes, see **Benchmark Matrix Runner**.

---

## Expected Output

A successful installation produces benchmark reports similar to

```text
results/
└── workspace_loading/
    └── workspace_loading_result.json
```

and benchmark figures in

```text
plots/
└── workspace_loading/
    ├── workspace_loading_wall_time.png
    ├── workspace_loading_current_rss_delta.png
    └── workspace_loading_peak_rss_delta.png
```

See **Outputs** for a complete description of generated reports and figures.

---

## Updating the Environment

If project dependencies change, synchronize the local environment.

```bash
pixi install
```

Pixi automatically updates the environment to match the repository configuration.

---

## Troubleshooting

### Pixi is not installed

Install Pixi before creating the project environment.

---

### Installation fails

Run

```bash
pixi install
```

again to ensure that all required dependencies have been installed successfully.

---

### Benchmark execution fails

Verify that

- the selected workspace files exist in the `inputs/` directory;
- the benchmark name is valid;
- the Pixi environment was created successfully.

If problems persist, verify that the repository was cloned correctly and that the installation completed without errors.

---

## Next Steps

Your benchmarking environment is now ready.

Continue with

- **Getting Started** to run your first benchmark.
- **Benchmark Methodology** to understand how benchmarks are executed and measured.
- **Benchmark Workspaces** to explore the available benchmark inputs.
- **Outputs** to inspect generated reports and figures.
- **Benchmark Matrix Runner** to execute complete benchmark campaigns.
