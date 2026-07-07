# Workspace Lifecycle

Every benchmark in the PyHS3 Benchmarks repository operates on one or more HS3 workspaces.

A workspace progresses through several well-defined stages during benchmark execution, from a serialized JSON description to an executable statistical model. Different benchmark suites measure different stages of this lifecycle, allowing the overall workflow to be analyzed in a modular and reproducible way.

---

# Lifecycle Overview

The complete workspace lifecycle is illustrated below.

```text
HS3 Workspace (JSON)
          │
          ▼
Workspace Loading
          │
          ▼
Model Creation
          │
          ▼
Log-Probability Construction
          │
          ▼
Graph Canonicalization
          │
          ▼
Graph Optimization
          │
          ▼
Compilation
          │
          ▼
Compiled Model
          │
          ├────────────► PDF Evaluation
          │
          ├────────────► NLL Scan
          │
          ├────────────► Memory Scaling
          │
          └────────────► Cross-Framework Benchmarks
```

Each benchmark focuses on one or more stages of this lifecycle while reusing the outputs produced by previous stages.

---

# Stage 1 — Workspace Loading

The lifecycle begins with an HS3 workspace stored as a JSON document.

During this stage, the serialized workspace is deserialized into an in-memory `Workspace` object.

This stage is measured by the **Workspace Loading** benchmark.

Typical operations include

- reading the workspace file;
- validating the workspace structure;
- constructing the in-memory workspace representation.

---

# Stage 2 — Model Creation

Once the workspace has been loaded, PyHS3 constructs the corresponding statistical model.

During this stage,

- model components are instantiated;
- parameters are created;
- observables are connected;
- probability density functions are assembled.

This stage is measured independently by the **Model Creation** benchmark.

---

# Stage 3 — Log-Probability Construction

The statistical model is transformed into a symbolic log-probability representation.

This stage constructs the computational graph that will later be optimized and evaluated.

The resulting graph represents the mathematical operations required to evaluate the statistical model.

This stage is measured by the **Log-Probability Construction** benchmark.

---

# Stage 4 — Graph Canonicalization

The symbolic graph is transformed into a canonical representation.

Canonicalization simplifies graph structure and establishes a consistent representation that is independent of implementation details.

This stage improves reproducibility and prepares the graph for subsequent optimization.

---

# Stage 5 — Graph Optimization

The canonical graph is optimized before numerical execution.

Typical optimizations include simplifying symbolic expressions and removing redundant computations where possible.

The optimized graph forms the basis for efficient compiled execution.

---

# Stage 6 — Compilation

The optimized graph is compiled into an executable representation.

Compilation transforms the symbolic graph into a form suitable for repeated numerical evaluation.

Compilation is performed only once and is measured separately from execution to distinguish setup costs from runtime performance.

---

# Stage 7 — Numerical Evaluation

After compilation, the workspace is ready for numerical computations.

Depending on the benchmark, this stage may include

- evaluating probability density functions;
- evaluating compiled log-probabilities;
- scanning the negative log-likelihood;
- performing cross-framework comparisons.

Because the expensive setup stages have already completed, these benchmarks measure execution performance independently of initialization costs.

---

# Why Separate the Lifecycle?

The statistical inference workflow consists of multiple computational stages with different performance characteristics.

Measuring each stage independently makes it possible to

- identify computational bottlenecks;
- evaluate optimization strategies;
- distinguish setup costs from execution costs;
- study memory usage throughout the workflow;
- compare implementations at equivalent stages.

This modular approach provides significantly more insight than measuring only the total execution time.

---

# Relationship to Benchmark Suites

Each benchmark suite corresponds to one or more stages of the workspace lifecycle.

| Lifecycle stage | Benchmark |
|-----------------|-----------|
| Workspace Loading | Workspace Loading |
| Model Creation | Model Creation |
| Log-Probability Construction | Log-Probability Construction |
| Graph Canonicalization | Graph Canonicalization |
| Graph Optimization | Graph Optimization |
| Compilation | Log-Probability Compilation |
| Numerical Evaluation | Compiled Evaluation, PDF Evaluation, NLL Scan |
| Cross-Framework Evaluation | Cross-Framework Benchmarks |

Together these benchmark suites provide complete coverage of the statistical workflow implemented by PyHS3.

---

# Related Documentation

See also

- **Benchmark Workflow** for the overall benchmark execution process.
- **Benchmark Methodology** for the measurement strategy used throughout the repository.
- **Benchmark Suite** for detailed descriptions of each benchmark.
- **Benchmark Results** for generated reports and visualizations.
