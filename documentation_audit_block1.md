# PyHS3 Benchmarks Documentation Audit — Block 1

## Scope

- 35 Markdown pages reviewed.
- Approximately 28,579 words.
- Audit labels: necessary, duplicated, too detailed, outdated/unverified, needs example, needs shortening.
- `assets/software_status/*` pages are included in the audit even though they are not present in the Zensical navigation.

## Executive findings

1. The documentation is technically rich but organized as a reference manual rather than a 15–20 minute onboarding path.
2. The largest source of repetition is the common benchmark-page template: purpose, workflow, direct command, runner command, full CLI table, outputs, JSON schema, plots, implementation details, limitations.
3. The lifecycle is explained repeatedly in `index.md`, `benchmark-methodology.md`, `concepts/workspace-lifecycle.md`, `benchmarks/index.md`, and many benchmark pages.
4. Runner behavior is split across `getting-started.md`, `concepts/benchmark-runner.md`, and the much more detailed `benchmark-matrix-runner.md`.
5. Workspace documentation explains the dataset design but does not provide the required enumerated inventory of actual files and framework support.
6. Cross-framework pages repeat methodology, lifecycle, cold/warm definitions, numerical-validation language, and plot interpretation.
7. Developer material is mixed into the primary user navigation.
8. Some content is demonstrably inconsistent: `benchmarks/overview.md` documents `docs/assets/images/plots/benchmark_overview/`, while the uploaded files use `docs/assets/plots/benchmark_overview/`.

## Page-by-page audit

Legend: **Yes** means the label applies. “Partial” means only some content should be retained or moved.

| Page | Main purpose | Necessary | Duplicated | Too detailed | Outdated / unverified | Needs example | Needs shortening | Recommended action |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `index.md` | Project overview, feature list, documentation guide, lifecycle, quick start | Yes | Yes | No | Partial | No | Yes | Rewrite as the compact **Overview** page. Keep what/why/frameworks/metrics and a 3-step quick start; remove the long documentation tour and duplicate lifecycle. |
| `installation.md` | Prerequisites, clone/install/verify/troubleshoot | Yes | Yes | Partial | Unverified | No | Yes | Merge installation and first-run verification into **Quick Start**. Keep only Git, Pixi, clone, `pixi install`, one verified command, expected output. Move troubleshooting to a collapsible or developer section. |
| `getting-started.md` | First benchmark, multiple benchmarks, full suite, outputs | Yes | Yes | Partial | Unverified | No | Yes | Merge with `installation.md`. Use one canonical command only; link to runner reference for advanced selections. Remove “typical development workflow” from user onboarding. |
| `repository-structure.md` | Directory layout and architectural philosophy | Partial | Yes | No | Unverified | No | Yes | Remove from main user path. Merge concise directory map into **Developer Notes / Architecture** with `development.md`. |
| `benchmark-methodology.md` | Shared measurement principles and metrics | Yes | Yes | Partial | Partial | Yes | Yes | Make this the single canonical **Methodology** page. Add exact defaults or explain that they are benchmark-specific; define cold/warm, repetitions, summary statistics, batching, correctness checks, machine metadata. Remove generic goals and repeated output descriptions. |
| `concepts/workspace-lifecycle.md` | Seven stages from loading to evaluation | Yes | Yes | No | No | Yes | Yes | Merge into **Methodology** as one lifecycle diagram/table. Do not keep a separate page unless developer-level details are added. |
| `concepts/benchmark-runner.md` | Conceptual runner purpose and campaign workflow | Partial | Yes | No | Unverified | No | Yes | Merge the user-facing 1–2 paragraph explanation into **Quick Start / Reproducing Results**. Move architecture details to the matrix-runner developer page. |
| `outputs.md` | JSON reports, matrix summary, figures, status, interpretation | Yes | Yes | Partial | Unverified | Yes | Yes | Merge practical output locations into **Quick Start** and **Reproducing Results**. Keep a short schema/artefact reference in Developer Notes if needed. Show one real JSON fragment. |
| `workspaces/index.md` | Dataset philosophy and paired HS3/ROOT overview | Partial | Yes | No | No | No | Yes | Merge into `workspaces/benchmark-workspaces.md`; it is currently an introductory duplicate. |
| `workspaces/benchmark-workspaces.md` | Baseline, variants, naming convention, ROOT counterparts | Yes | Partial | Partial | Unverified | Yes | Yes | Rewrite as **Workspaces**. Preserve naming-key explanations, but add the required complete enumerated table of actual files, channels, bins/events, distributions, constraints, purpose, format, and framework support. |
| `benchmarks/index.md` | Benchmark categories, pipeline, benchmark inventory, commands | Yes | Yes | Partial | Partial | No | Yes | Convert into the single compact **Benchmarks** page. Keep a one-row-per-benchmark semantics table. Remove full workspace commands and claims for undocumented benchmark categories unless verified. |
| `benchmarks/overview.md` | How summary plots are generated and interpreted | Yes | Partial | Partial | Yes | No | Yes | Move plot-generation command to **Reproducing Results** and key figures/interpretation to **Results**. Correct the documented `assets/images/plots` path to `assets/plots`. Remove internal architecture from main path. |
| `benchmarks/workspace-loading.md` | Loading timing/memory semantics, CLI, results | Partial | Yes | Yes | Unverified | No | Yes | Collapse into a row/short subsection under **Benchmarks**. Retain inclusions/exclusions, warm-up, runs, statistic, outputs, and one result sentence. Move full CLI/schema/implementation notes to developer reference or generated `--help`. |
| `benchmarks/model-creation.md` | Model-construction timing/memory semantics and results | Partial | Yes | Yes | Unverified | No | Yes | Same consolidation as workspace loading. Strongly duplicates its structure and explanatory language. Preserve the important exclusion of workspace loading and isolated memory measurement. |
| `benchmarks/log-prob-construction.md` | Symbolic log-probability graph construction | Partial | Yes | Yes | Unverified | No | Yes | Collapse into **Benchmarks**. Keep exact measured boundary and validation; remove repeated runner command, full CLI table, output tree, and generic interpretation. |
| `benchmarks/graph-canonicalization.md` | Canonicalization timing/memory and graph simplification | Partial | Yes | Yes | Unverified | Yes | Yes | Collapse into **Benchmarks**. Add one small before/after conceptual example if canonicalization remains unfamiliar; move implementation details to developer notes. |
| `benchmarks/graph-optimization.md` | Optimization timing/memory and graph validation | Partial | Yes | Yes | Unverified | Yes | Yes | Collapse into **Benchmarks**, paired with canonicalization. Clarify how it differs from canonicalization with one concrete example; remove repeated CLI/result scaffolding. |
| `benchmarks/log-prob-compilation.md` | Compilation timing/memory | Partial | Yes | Yes | Unverified | No | Yes | Collapse into **Benchmarks**. Explicitly distinguish compilation from construction and warm evaluation; keep only measurement boundary and key plot. |
| `benchmarks/compiled-evaluation.md` | Warm compiled likelihood evaluation and throughput | Yes | Yes | Partial | Unverified | No | Yes | Keep as a compact subsection in **Benchmarks** because it is central. Consolidate with PDF evaluation where terminology overlaps; explicitly state batching and warm-up. |
| `benchmarks/pdf-evaluation.md` | Cold/warm PDF latency, throughput, memory | Yes | Yes | Partial | Unverified | No | Yes | Keep compactly in **Benchmarks**. Define scalar/vector/batch input and what cold-start includes. Remove full CLI argument reference from core docs. |
| `benchmarks/nll-scan.md` | NLL scan runtime per point, memory, validation | Yes | Yes | Yes | Unverified | No | Yes | Keep compactly in **Benchmarks**. State point count, parameter range, batching, setup inclusion, warm-up, and summary statistic; move generic output/schema text out. |
| `benchmarks/memory-scaling.md` | RSS growth across model sizes/stages | Yes | Yes | Yes | Unverified | Yes | Yes | Keep one concise benchmark definition and one key plot in **Benchmarks/Results**. Add a short warning about RSS interpretation and process isolation. |
| `benchmarks/model-complexity-scaling.md` | Runtime/memory versus workspace complexity | Yes | Yes | Yes | Unverified | Yes | Yes | Merge with memory scaling into a “Scaling” subsection. Explicitly define the complexity axis and workspace set; avoid repeating every stage’s result narrative. |
| `cross-framework/index.md` | Philosophy, engines, categories, numerical validation | Yes | Yes | Partial | Partial | No | Yes | Use as the base for a concise cross-framework subsection in **Benchmarks/Methodology**. Remove repeated lifecycle and “why” text. Verify listed engines/available benchmarks against the current runner. |
| `cross-framework/scalar-pdf.md` | Scalar PDF methodology, cold/warm metrics, plots | Yes | Yes | Yes | Unverified | No | Yes | Retain a concise semantics table and 2–3 key figures in **Results**. Move diagnostic/cache details and full plot catalogue to developer/reference notes. |
| `cross-framework/nll-scan.md` | Pointwise/batched ΔNLL comparison and lifecycle | Yes | Yes | Yes | Unverified | No | Yes | Merge overlapping methodology with scalar PDF and xRooFit into one cross-framework methodology section. Keep benchmark-specific statistical quantity, scan setup, batching, agreement tolerance, and key plots. |
| `cross-framework/xroofit.md` | Full PyHS3 vs xRooFit setup, installation, methodology, results | Yes | Yes | Yes | Partial | Yes | Yes | Largest page (~3,042 words). Reduce heavily. Keep xRooFit-specific setup in **Reproducing Results** or Developer Notes; move shared methodology out; retain exact engine definitions, installation caveat, validation tolerance, timing boundaries, and 2–3 key results. |
| `cross-framework/binned-likelihood.md` | HistFactory/pyhf-style binned likelihood validation and timing | Yes | Yes | Yes | Unverified | No | Yes | Keep a concise benchmark-specific section. Move shared cold/warm, numerical-validation, and output explanations to Methodology. Clearly state workspace format/framework support and construction versus warm call. |
| `benchmark-matrix-runner.md` | Complete runner registry, modes, selection, paths, retries, logs | Yes for developers | Yes | Yes | Unverified | Yes | Yes | Move under **Developer Notes / Runner Reference**. Keep one short user recipe in Reproducing Results. This page should not be required reading. Consider generating CLI options from `--help` to prevent drift. |
| `development.md` | How to add/register/test/document a benchmark | Yes for developers | Yes | Partial | Unverified | Yes | Yes | Make the Developer Notes landing page. Merge repository architecture from `repository-structure.md`; link to runner, tests, profiling, and API. Add one concrete minimal benchmark-registration example. |
| `api.md` | CLI/internal module interface overview | Partial | Yes | No | Unverified | Yes | Yes | Rename to **Internal API / Code Reference** or remove. It is not a true API reference. Merge module overview into Developer Notes and avoid presenting internal modules as stable public API. |
| `scalene-profiling.md` | Profiling rationale, commands, metrics, workflow | Yes for developers | Partial | Partial | Partial | Yes | Yes | Keep under Developer Notes. Verify references to JAX versus the actual backend; provide one complete command with benchmark arguments and expected output. Remove generic profiling prose. |
| `tests.md` | Test, coverage, lint, CI commands and organization | Yes for developers | Partial | Partial | Unverified | No | Yes | Keep under Developer Notes. Reduce the long list of individual test examples and file tree; keep canonical commands and contribution checks. Verify Pixi task names. |
| `assets/software_status/benchmark_environment.md` | Frozen machine/software/commit metadata | Yes as result metadata | No | No | Yes by design | No | No | Do not place in normal navigation. Treat as a generated/snapshot appendix linked from Results. Update automatically for each published benchmark campaign. Current branch/commit values are inherently time-bound. |
| `assets/software_status/current_benchmark_status.md` | Working/in-progress/known issues snapshot | Partial | No | No | Yes by design | No | No | Keep out of the main docs or replace with issue tracker/project status. It is dated and quickly becomes stale; current content says documentation cleanup and cross-framework work are still in progress. |

## Section-level duplication map

### 1. Project purpose and feature overview

Repeated in:

- `index.md`
- `benchmark-methodology.md`
- `benchmarks/index.md`
- `cross-framework/index.md`
- `repository-structure.md`

**Resolution:** explain project purpose once in **Overview**. Other pages should start immediately with page-specific information.

### 2. Statistical workspace lifecycle

Repeated in:

- `index.md`
- `benchmark-methodology.md`
- `concepts/workspace-lifecycle.md`
- `benchmarks/index.md`
- individual benchmark pages

**Resolution:** one canonical lifecycle diagram/table in **Methodology**. Individual benchmarks should link to their lifecycle stage rather than restating the whole pipeline.

### 3. Benchmark runner and campaign execution

Repeated in:

- `getting-started.md`
- `installation.md`
- `concepts/benchmark-runner.md`
- `benchmark-matrix-runner.md`
- `outputs.md`
- `benchmarks/index.md`
- most individual benchmark pages

**Resolution:** one simple command in **Quick Start**, exact campaign commands in **Reproducing Results**, full registry/modes/reference in **Developer Notes**.

### 4. Outputs and directory layout

Repeated in:

- `installation.md`
- `getting-started.md`
- `outputs.md`
- `repository-structure.md`
- `concepts/benchmark-runner.md`
- `benchmark-matrix-runner.md`
- individual benchmark pages

**Resolution:** show output locations once in Quick Start and the complete reproducible tree once in Reproducing Results. Benchmark pages only list benchmark-specific filenames when necessary.

### 5. Warm-up, repetitions, aggregation, and memory semantics

Repeated in:

- `benchmark-methodology.md`
- individual workflow benchmark pages
- cross-framework benchmark pages

**Resolution:** define common semantics in Methodology. Each benchmark gets a compact deviations table: included time, excluded time, compilation, batching, warm-up, runs, statistic, validation.

### 6. Numerical validation and apples-to-apples language

Repeated in:

- `benchmark-methodology.md`
- `cross-framework/index.md`
- `cross-framework/scalar-pdf.md`
- `cross-framework/nll-scan.md`
- `cross-framework/xroofit.md`
- `cross-framework/binned-likelihood.md`

**Resolution:** define the general policy once. Each benchmark specifies only compared quantity, parameter points/data, tolerance, and known non-equivalences.

### 7. Benchmark-page scaffolding

Strongly repeated across nearly every file in `benchmarks/`:

- what is measured;
- how to run directly;
- how to run through the runner;
- full CLI argument table;
- notes;
- output tree;
- JSON fields;
- plot-by-plot results;
- implementation details;
- limitations.

**Resolution:** replace the 11 workflow pages in the main path with one compact benchmark catalogue. Preserve detailed pages only as optional reference pages, or generate CLI/schema sections automatically.

### 8. Workspace dataset philosophy

Repeated in:

- `workspaces/index.md`
- `workspaces/benchmark-workspaces.md`
- `cross-framework/index.md`
- several cross-framework pages

**Resolution:** one **Workspaces** page with an exact inventory table. Cross-framework pages should reference workspace IDs rather than re-explain the dataset.

## Confirmed or likely stale/inconsistent items

1. `benchmarks/overview.md` uses `docs/assets/images/plots/benchmark_overview/`, but uploaded plot files are under `docs/assets/plots/benchmark_overview/`.
2. `assets/software_status/benchmark_environment.md` pins a branch and commit; it must be treated as a dated benchmark snapshot, not evergreen documentation.
3. `assets/software_status/current_benchmark_status.md` is explicitly dated and lists work in progress; it should not be relied upon as permanent documentation.
4. Several pages contain extensive CLI option/default lists. These are high-risk for drift and must be verified against current `argparse` definitions or generated from `--help`.
5. `benchmarks/index.md` lists cross-framework categories such as vectorized PDF and model-complexity scaling without corresponding pages in the current navigation. Verify whether these are implemented, intentionally undocumented, or stale.
6. `api.md` describes an “API Reference” but explicitly says the project has no stable public Python API. The title and placement are misleading.
7. `scalene-profiling.md` mentions NumPy, JAX, and compiled kernels, while other pages emphasize PyTensor. Verify the current backend wording.

## Pages that need examples most urgently

- **Methodology:** one annotated benchmark record showing warm-up, runs, statistic, and included/excluded work.
- **Workspaces:** one decoded filename plus the complete workspace inventory.
- **Graph canonicalization vs optimization:** one small before/after graph or expression example.
- **Outputs:** one real, shortened JSON result.
- **Development:** one minimal benchmark registration example.
- **Runner reference:** one example showing how a `BenchmarkSpec` changes command construction.

## Recommended disposition summary

### Core user path

Rewrite/merge into seven pages:

1. Overview
2. Quick Start
3. Methodology
4. Workspaces
5. Benchmarks
6. Results
7. Reproducing Results

### Optional developer path

Retain, shorten, and reorganize:

- Developer Notes / Architecture
- Matrix Runner Reference
- Adding a Benchmark
- Testing
- Profiling
- Internal API / Result Schema

### Pages to merge or remove from navigation

- `repository-structure.md`
- `concepts/workspace-lifecycle.md`
- `concepts/benchmark-runner.md`
- `workspaces/index.md`
- most individual workflow benchmark pages from the core navigation
- status snapshot pages

## Block 1 completion status

- [x] All pages and sections inventoried.
- [x] Every page classified for necessity, duplication, excessive detail, staleness risk, examples, and shortening.
- [x] Major duplication clusters mapped.
- [x] Confirmed internal path inconsistency recorded.
- [x] Pages suitable for the core path versus Developer Notes identified.
