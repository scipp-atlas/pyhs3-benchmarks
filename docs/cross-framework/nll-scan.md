# Cross-framework ΔNLL Scan Benchmark

This benchmark performs an **apples-to-apples numerical and performance comparison** between **PyHS3** and **RooFit** by evaluating the same ΔNLL scan on matching generated workspaces.

Unlike the generic performance benchmarks, this benchmark is designed to verify that both frameworks evaluate **the same statistical model**, using the **same parameter values**, **same observed events**, and **same scan grid** before comparing execution time.

---

## What is compared?

For each workspace, the benchmark

- loads the matching **PyHS3 JSON workspace**;
- loads the corresponding **ROOT RooWorkspace**;
- extracts the identical observed events from the HS3 workspace;
- scans the same POI (`mu_sig`);
- evaluates the normalized PDF at every observed event;
- computes

\[
\mathrm{NLL}(\mu)
=
-\sum_i \log p(x_i \mid \mu)
\]

- converts the result into

\[
\Delta\mathrm{NLL}(\mu)
=
\mathrm{NLL}(\mu)
-
\min_\mu \mathrm{NLL}(\mu)
\]

The PyHS3 ΔNLL curve is used as the numerical reference.

---

## Apples-to-apples methodology

The benchmark intentionally evaluates **exactly the same mathematical quantity** in both frameworks.

For every scan point:

| PyHS3 | RooFit |
|--------|---------|
| evaluates `model.logpdf(...)` | evaluates `pdf.getVal(normSet)` |
| uses the same observed events | uses the same observed events |
| uses identical μ values | uses identical μ values |
| computes `-Σ log(pdf)` | computes `-Σ log(pdf)` |

No framework-specific likelihood builders (`createNLL()`) are used, since those may introduce additional extended or constraint terms depending on the workspace implementation.

Instead, both frameworks evaluate the normalized PDF directly, producing an equivalent unbinned likelihood calculation.

---

## Validation

For every workspace the benchmark verifies

- identical ΔNLL minimum;
- identical ΔNLL shape;
- maximum point-by-point residual;
- numerical agreement within tolerance.

The default tolerances are

```
maximum ΔNLL residual < 1e-7
minimum position agreement < 1e-12
```

---

## Default benchmark inputs

```
inputs/
├── 5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json
├── 10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json
└── 30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json
```

with the corresponding ROOT workspaces

```
inputs/
├── 5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.root
├── 10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.root
└── 30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.root
```

---

## Running the benchmark

Single workspace

```bash
pixi run python -m src.run_cross_nll_scan \
    --workspaces inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
    --frameworks pyhs3 roofit \
    --analysis L_ch0 \
    --poi mu_sig \
    --mode FAST_RUN \
    --mu-min 0.0 \
    --mu-max 2.0 \
    --n-points 101 \
    --output-dir results/docs_examples/cross_nll_scan \
    --plot-dir docs/assets/plots/cross_nll_scan
```

Multiple workspaces

```bash
pixi run python -m src.run_cross_nll_scan \
    --workspaces \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --frameworks pyhs3 roofit \
    --analysis L_ch0 \
    --poi mu_sig \
    --mode FAST_RUN \
    --mu-min 0.0 \
    --mu-max 2.0 \
    --n-points 101 \
    --output-dir results/docs_examples/cross_nll_scan \
    --plot-dir docs/assets/plots/cross_nll_scan
```

---

---

## Command-line Arguments

The benchmark supports the following command-line arguments.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--frameworks` | `str ...` | `pyhs3 roofit` | Frameworks to compare. Supported values are `pyhs3` and `roofit`. |
| `--workspaces` | `Path ...` | Benchmark workspace set | One or more HS3 workspace JSON files used for the ΔNLL comparison. |
| `--root-workspaces` | `Path ...` | inferred automatically | Optional ROOT workspace files corresponding to `--workspaces`. If omitted, each `.root` file is inferred automatically from the JSON workspace path. |
| `--analysis` | `str` | `L_ch0` | Analysis (likelihood) name used to construct the statistical model. |
| `--target` | `str` | inferred from `--analysis` | Target PDF evaluated by PyHS3 and RooFit. By default this is derived automatically from the analysis name (for example `model_ch0`). |
| `--pyhs3-data-name` | `str` | inferred from `--analysis` | Name of the observed dataset inside the HS3 workspace. |
| `--root-pdf-name` | `str` | inferred from `--target` | Name of the RooFit PDF inside the ROOT workspace. |
| `--root-data-name` | `str` | inferred from `--pyhs3-data-name` | Name of the RooFit dataset. |
| `--parameter-point` | `str` | first available | Optional parameter point used to initialize the model before scanning the POI. |
| `--observable-name` | `str` | `x` | Observable used during normalized PDF evaluation. |
| `--observable-index` | `int` | `0` | Observable index within multidimensional HS3 datasets. |
| `--poi` | `str` | `mu_sig` | Parameter of interest scanned during the ΔNLL evaluation. |
| `--mode` | `str` | `FAST_RUN` | PyTensor compilation mode used when constructing the PyHS3 model. |
| `--mu-min` | `float` | `0.0` | Lower bound of the POI scan. |
| `--mu-max` | `float` | `2.0` | Upper bound of the POI scan. |
| `--n-points` | `int` | `101` | Number of uniformly spaced scan points between `mu-min` and `mu-max`. |
| `--shape-tolerance` | `float` | `1e-7` | Maximum allowed point-by-point ΔNLL difference relative to the PyHS3 reference. |
| `--minimum-tolerance` | `float` | `1e-12` | Maximum allowed difference between the positions of the ΔNLL minima. |
| `--output-dir` | `Path` | `results/cross_nll_scan/` | Directory where the benchmark JSON results are written. |
| `--output-name` | `str` | `cross_nll_scan_result.json` | Name of the JSON output file. |
| `--plot` | flag | disabled | Generate comparison plots summarizing runtime, memory usage, ΔNLL agreement, and numerical validation. |
| `--plot-dir` | `Path` | `docs/assets/plots/cross_nll_scan/` | Directory where generated plots are stored. |
| `--fail-fast` | flag | disabled | Stop the benchmark immediately after the first failed benchmark or validation error. |

## Notes

- At least one framework and one workspace must be provided.
- If the `roofit` framework is selected, matching ROOT workspaces must be available. When `--root-workspaces` is omitted, the corresponding `.root` files are inferred automatically from the JSON workspace paths.
- The benchmark first computes a complete PyHS3 ΔNLL scan for every workspace. These scans serve as the numerical reference for all subsequent RooFit comparisons.
- A separate benchmark is executed for every combination of workspace and framework.
- The POI scan is constructed using a uniformly spaced grid between `--mu-min` and `--mu-max` containing `--n-points` values.
- The benchmark evaluates normalized PDFs directly for every observed event and computes
  \[
  \mathrm{NLL} = -\sum_i \log p(x_i|\mu),
  \]
  avoiding framework-specific likelihood builders such as `createNLL()` to ensure an apples-to-apples comparison.
- Numerical validation requires both the ΔNLL profile shape and the position of the minimum to satisfy the specified tolerances.
- `--n-points` must be at least **2**, `--mu-min` must be smaller than `--mu-max`, and both validation tolerances must be positive.

---

## Generated plots

### Cross-framework ΔNLL agreement

Shows the ΔNLL curves produced by both frameworks.

![Cross-framework ΔNLL agreement](../assets/plots/cross_nll_scan/cross_nll_scan_profile.png)

---

### Runtime profile

Separately reports

- model construction time;
- full ΔNLL scan time.

![Runtime profile](../assets/plots/cross_nll_scan/cross_nll_timing_profile.png)

---

### Relative throughput

Ranks frameworks according to scan throughput (μs per scan point).

![Relative runtime](../assets/plots/cross_nll_scan/cross_nll_relative_runtime.png)

---

### Numerical agreement

Reports the maximum residual with respect to the PyHS3 reference.

The dashed line indicates the validation tolerance.

![Numerical agreement](../assets/plots/cross_nll_scan/cross_nll_numerical_agreement.png)

---

### Memory footprint

Reports the current and peak RSS increase during the benchmark.

![Memory footprint](../assets/plots/cross_nll_scan/cross_nll_memory_profile.png)

---

## Output

The benchmark stores

```
results/docs_examples/cross_nll_scan/
└── cross_nll_scan_result.json
```

which contains

- benchmark configuration;
- scan grid;
- timing measurements;
- memory measurements;
- ΔNLL curves;
- numerical validation;
- summary status.

---

## Interpretation

A successful benchmark reports

```
Status: success
Validated: N / N
```

meaning that

- both frameworks produced identical ΔNLL minima;
- ΔNLL curves agree within numerical precision;
- both implementations evaluated the same statistical quantity.

Any validation failure indicates a genuine numerical disagreement rather than a performance difference.
