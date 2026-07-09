# Installation

This guide explains how to install the **PyHS3 Benchmarks** repository and create a fully reproducible benchmarking environment.

By the end of this guide, you will have

- cloned the repository;
- installed all required dependencies;
- verified that the benchmarking environment is working correctly;
- executed your first benchmark.

---

# What Will Be Installed?

The PyHS3 Benchmarks repository uses **Pixi** to create a fully reproducible software environment.

Running

```bash
pixi install
```

installs and configures everything required to execute the benchmark suite, including

- the PyHS3 Benchmarks framework;
- the PyHS3 library and its dependencies;
- numerical computation and compilation backends;
- plotting and visualization libraries;
- benchmarking utilities;
- the Python environment used throughout the repository.

Because the environment is managed entirely by Pixi, no manual Python installation, virtual environment setup, or dependency management is required after cloning the repository.

Every benchmark, example, and figure presented throughout this documentation is intended to run inside this environment.

---

# Prerequisites

Before installing the repository, ensure that the following tools are available on your system.

- **Git**, for cloning the repository.
- **Pixi**, for creating and managing the project environment.

No additional software is required. Python and all project dependencies are installed automatically by Pixi.

---

# Installation Overview

Installing the repository consists of four steps.

1. Clone the repository.
2. Create the project environment.
3. Verify the installation.
4. Execute a benchmark.

---

# Clone the Repository

Clone the repository from GitHub.

```bash
git clone https://github.com/scipp-atlas/pyhs3-benchmarks.git

cd pyhs3-benchmarks
```

---

# Create the Benchmark Environment

Create the project environment by running

```bash
pixi install
```

Pixi automatically resolves and installs all project dependencies, creating an isolated environment that matches the repository configuration.

---

# Verify the Installation

Verify that the installation completed successfully by executing the **Workspace Loading** benchmark.

```bash
pixi run python -m src.run_workspace_loading \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --n-runs 10 \
    --plot
```

A successful execution should

- load the selected benchmark workspaces;
- print benchmark statistics to the terminal;
- generate a JSON benchmark report;
- generate comparison plots.

---

# Verify the Benchmark Runner

After confirming that an individual benchmark executes successfully, verify the shared benchmarking infrastructure by running the benchmark runner.

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

This verifies that the common benchmark execution framework is functioning correctly.

---

# Expected Output

A successful installation produces benchmark reports similar to

```text
results/
└── workspace_loading/
    └── workspace_loading_result.json
```

and generated figures in

```text
plots/
└── workspace_loading/
    ├── workspace_loading_wall_time.png
    ├── workspace_loading_current_rss_delta.png
    └── workspace_loading_peak_rss_delta.png
```

The exact outputs depend on the benchmark being executed.

---

# Updating the Environment

If project dependencies change, synchronize the local environment by running

```bash
pixi install
```

Pixi automatically updates the environment to match the repository configuration.

---

# Troubleshooting

## Pixi is not installed

Install Pixi before continuing with the installation.

---

## Installation fails

Run

```bash
pixi install
```

again to ensure that all required dependencies have been installed successfully.

---

## Benchmark execution fails

Verify that

- the selected workspace files exist in the `inputs/` directory;
- the requested benchmark name is valid;
- the Pixi environment was created successfully.

If benchmark execution still fails, verify that the repository was cloned correctly and that the installation completed without errors.

---

# Next Steps

Your benchmarking environment is now ready.

Continue with

- **Getting Started** to execute your first benchmarking workflow.
- **Repository Structure** to understand the organization of the repository.
- **Benchmark Methodology** to learn how benchmark measurements are performed.
- **Benchmark Suite** for an overview of all available benchmarks.
