# Profiling with Scalene

Performance benchmarking measures **how fast** an operation executes.

Profiling answers a different question:

> **Where is execution time spent?**

For this purpose, the benchmark suite uses **Scalene**, a high-performance profiler for Python that provides detailed CPU and memory attribution.

Unlike traditional profilers, Scalene distinguishes between

- Python execution time;
- native extension execution time;
- system time;
- memory allocation;
- memory growth;
- memory copy volume.

This information is invaluable when identifying optimization opportunities within PyHS3.

---

# Why Scalene?

Many PyHS3 operations involve

- NumPy;
- JAX;
- compiled numerical kernels;
- Python orchestration;
- memory-intensive graph construction.

Traditional profilers often attribute all execution time to Python function calls, making it difficult to understand where performance is actually spent.

Scalene separates

- Python CPU time;
- native CPU time;
- system CPU time;
- memory allocation;
- memory growth.

This makes it significantly easier to identify performance bottlenecks and understand whether they originate from Python code or compiled libraries.

---

# What We Use Scalene For

Within this repository, Scalene is primarily used to

- investigate performance bottlenecks;
- identify expensive Python code paths;
- study memory allocation patterns;
- understand memory growth;
- guide optimization efforts.

Scalene is intended for **performance analysis**, not benchmark reporting.

The benchmark figures presented throughout this documentation are produced by the benchmarking framework itself. Scalene serves as a complementary tool for understanding *why* a benchmark performs the way it does.

---

# Running Scalene

Any benchmark script can be profiled with Scalene.

For example, to profile the model creation benchmark:

```bash
pixi run scalene \
    src/run_model_creation.py \
    -o results/scalene/model_creation.json
```

The `-o` option stores the profiling results in JSON format, allowing them to be archived or analyzed programmatically.

Alternatively, Scalene can generate an interactive HTML report:

```bash
pixi run scalene src/run_model_creation.py
```

After execution, Scalene automatically creates an HTML report that can be viewed in a web browser.

---

# Typical Profiling Workflow

A common optimization workflow is

1. Run a benchmark.
2. Identify a slow stage.
3. Profile that stage using Scalene.
4. Locate Python bottlenecks.
5. Optimize the implementation.
6. Re-run the benchmark to quantify the improvement.

This separates **performance measurement** from **performance diagnosis**, resulting in a reproducible optimization workflow.

---

# Understanding the Output

Scalene reports several useful metrics.

| Metric | Description |
|---------|-------------|
| Python CPU | Time spent executing Python code |
| Native CPU | Time spent inside compiled libraries |
| System CPU | Operating system overhead |
| Memory Allocation | Memory allocated by each line |
| Memory Growth | Net memory increase |
| Copy Volume | Amount of memory copied between objects |

Together, these metrics provide a comprehensive view of application performance.

---

# JSON Output

Within this repository, Scalene profiles are commonly stored as JSON files in

```text
results/scalene/
```

Storing profiles as JSON allows profiling results to be archived, compared across benchmark runs, and analyzed programmatically.

A typical profile contains

- execution metadata;
- command-line arguments;
- elapsed runtime;
- per-file statistics;
- line-level CPU utilization;
- memory allocation information.

For example, the repository includes a Scalene profile generated while executing the `model_creation` benchmark, illustrating the structure of Scalene's JSON output. :contentReference[oaicite:0]{index=0}

---

# When Should Scalene Be Used?

Scalene is particularly useful when

- optimizing benchmark implementations;
- investigating unexpectedly slow benchmarks;
- identifying excessive memory allocations;
- validating the impact of performance optimizations.

Routine benchmark execution does not require Scalene. It is an optional tool intended for performance investigation during development and optimization.

---

# Benchmarking vs Profiling

Although closely related, benchmarking and profiling serve different purposes.

| Benchmarking | Profiling |
|--------------|-----------|
| Measures performance | Explains performance |
| Produces reproducible metrics | Identifies bottlenecks |
| Generates benchmark reports | Guides optimization |
| Compares implementations | Analyzes implementation details |

In this repository, benchmarking is used to measure performance across benchmark suites, while Scalene is used to investigate and optimize the underlying implementation.

---

# Further Reading

Additional information about Scalene is available in the official project documentation:

https://github.com/plasma-umass/scalene
