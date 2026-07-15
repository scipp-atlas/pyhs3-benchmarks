# Workspace Lifecycle

On this page, you will learn how an HS3 workspace is transformed from a serialized model into an executable statistical representation used throughout the PyHS3 workflow.

Rather than describing benchmark execution, this page focuses on the evolution of a workspace itself. Each stage transforms the statistical model into a representation suitable for the next stage of the inference pipeline.

---

# Lifecycle Overview

Each stage transforms the workspace into a richer internal representation that can be reused by subsequent stages.

---

# Stage 1 — Workspace Loading

The lifecycle begins with a serialized HS3 workspace stored as JSON.

During this stage the workspace is read from disk and converted into an in-memory `Workspace` object that represents the statistical model.

---

# Stage 2 — Model Creation

The workspace is transformed into a collection of statistical model objects.

Parameters, observables, probability density functions, and model components are connected to form an executable representation of the statistical model.

---

# Stage 3 — Log-Probability Construction

The statistical model is converted into a symbolic log-probability graph.

Instead of performing numerical evaluation immediately, this stage builds a computational representation that can later be optimized and executed efficiently.

---

# Stage 4 — Graph Canonicalization

The symbolic graph is rewritten into a canonical representation.

This normalization step provides a consistent graph structure independent of how the original model was specified.

---

# Stage 5 — Graph Optimization

The canonical graph is simplified before numerical execution.

Expression simplification and elimination of redundant computations prepare the graph for efficient evaluation.

---

# Stage 6 — Compilation

The optimized graph is compiled into an executable representation.

Compilation converts the symbolic computation into a form that can be evaluated repeatedly with minimal overhead.

---

# Stage 7 — Numerical Evaluation

Once compilation is complete, the model can be evaluated efficiently.

Depending on the analysis, this executable representation may be used for

- probability density evaluation;
- likelihood evaluation;
- likelihood scans;
- memory studies;
- cross-framework comparisons.

These analyses all reuse the same compiled statistical model.

---

# Why the Lifecycle Matters

Representing the workflow as a sequence of independent transformations provides several advantages.

- Each stage has a clearly defined responsibility.
- Intermediate representations can be inspected independently.
- Expensive preparation stages are performed only once.
- Later stages reuse the outputs of earlier transformations.

This layered design simplifies both optimization and maintenance of the statistical inference pipeline.

---

# Relationship to Benchmark Suites

Many benchmark suites measure the performance of one lifecycle stage.

| Workspace lifecycle stage | Corresponding benchmark |
|---------------------------|-------------------------|
| Workspace Loading | Workspace Loading |
| Model Creation | Model Creation |
| Log-Probability Construction | Log-Probability Construction |
| Graph Canonicalization | Graph Canonicalization |
| Graph Optimization | Graph Optimization |
| Compilation | Log-Probability Compilation |
| Numerical Evaluation | Compiled Evaluation, PDF Evaluation, NLL Scan |
| Cross-Framework Evaluation | Cross-Framework Benchmarks |

This mapping allows individual stages of the workspace lifecycle to be evaluated independently without changing the underlying statistical workflow.

---

# Related Documentation

See also

- **Benchmark Methodology** — how benchmark measurements are performed.
- **Benchmark Suite** — detailed descriptions of each benchmark.
- **Benchmark Workspaces** — available benchmark input models.
- **Benchmark Results** — generated reports and visualizations.
