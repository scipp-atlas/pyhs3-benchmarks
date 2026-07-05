# Installation

This guide explains how to install the **PyHS3 Benchmarks** repository and verify that the benchmarking environment is configured correctly.

By the end of this guide you will have

- cloned the repository;
- installed all required dependencies;
- verified the installation by running a benchmark;
- generated your first benchmark plots.

---

# Prerequisites

Before installing the project, ensure that the following software is available on your system.

- Git
- Pixi

Python and all required packages are managed automatically by Pixi, so no manual Python installation or virtual environment setup is required.

---

# Clone the Repository

Clone the repository from GitHub.

```bash
git clone https://github.com/scipp-atlas/pyhs3-benchmarks.git

cd pyhs3-benchmarks
```

---

# Install Dependencies

Create the project environment.

```bash
pixi install
```

Pixi installs all required dependencies and creates an isolated environment for running benchmarks.

---

# Verify the Installation

Verify that the installation completed successfully by running the Workspace Loading benchmark.

```bash
pixi run python -m src.run_workspace_loading \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --n-runs 10 \
    --plot
```

Successful execution should

- load the selected workspaces;
- print benchmark statistics to the terminal;
- generate a JSON benchmark report;
- generate comparison plots.

---

# Verify the Benchmark Runner

Next, verify that the benchmark runner is working correctly.

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

This command executes the benchmark through the common benchmark runner and verifies that the shared benchmarking infrastructure is functioning correctly.

---

# Expected Output

After a successful installation, the repository should produce benchmark reports similar to

```text
results/
└── workspace_loading/
    └── workspace_loading_result.json
```

and benchmark plots in

```text
docs/
└── assets/
    └── plots/
        └── workspace_loading/
```

---

# Updating the Environment

If project dependencies change, synchronize the local environment by running

```bash
pixi install
```

Pixi automatically updates the environment to match the project configuration.

---

# Troubleshooting

## Pixi is not installed

Install Pixi by following the official installation instructions before continuing.

## Installation fails

Run

```bash
pixi install
```

again to ensure that all required dependencies are installed correctly.

## Benchmark execution fails

Verify that

- the selected workspace files exist in the `inputs/` directory;
- the benchmark name is valid;
- the Pixi environment was created successfully.

If benchmark execution still fails, verify that the repository was cloned correctly and that all dependencies were installed without errors.

---

# Next Steps

After completing the installation, continue with

- **Getting Started** — run your first benchmark.
- **Benchmark Suite** — overview of all available benchmarks.
- **Benchmark Methodology** — measurement principles and reporting conventions.
- **Workspace Loading** — detailed documentation for the first workflow benchmark.
