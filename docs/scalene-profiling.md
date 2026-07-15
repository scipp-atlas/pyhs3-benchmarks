# Profiling with Scalene

On this page, you will learn how to use **Scalene** to profile benchmark implementations, identify performance bottlenecks, and guide optimization efforts.

While benchmarking measures **how fast** a benchmark executes, profiling explains **where execution time and memory are spent**. Within this repository, Scalene is an optional developer tool used to investigate benchmark performance rather than to produce benchmark results.

---

# Why Scalene?

PyHS3 Benchmarks combines Python orchestration with numerical libraries such as

- NumPy;
- JAX;
- compiled numerical kernels.

Traditional profilers often attribute most execution time to Python function calls, making it difficult to distinguish Python overhead from compiled execution.

Scalene reports

- Python CPU time;
- native CPU time;
- system CPU time;
- memory allocation;
- memory growth;
- memory copy volume.

This makes it easier to identify whether a performance bottleneck originates in Python code or compiled libraries.

---

# Typical Use Cases

Within this repository, Scalene is primarily used to

- investigate performance bottlenecks;
- identify expensive Python code paths;
- analyze memory allocation patterns;
- understand memory growth;
- evaluate the impact of performance optimizations.

Benchmark reports and publication-quality figures are produced by the benchmarking framework itself. Scalene complements these results by explaining *why* a benchmark behaves as it does.

---

# Running Scalene

Any benchmark can be profiled using Scalene.

For example, to profile the Model Creation benchmark:

```bash
pixi run scalene \
    src/run_model_creation.py \
    -o results/scalene/model_creation.json
```

The `-o` option stores the profile in JSON format for later inspection or automated analysis.

To generate an interactive HTML report instead:

```bash
pixi run scalene src/run_model_creation.py
```

After execution, Scalene automatically generates an HTML report that can be explored in a web browser.

---

# Typical Profiling Workflow

A common optimization workflow is

1. Run a benchmark.
2. Identify a slow stage.
3. Profile that stage with Scalene.
4. Locate the performance bottleneck.
5. Optimize the implementation.
6. Re-run the benchmark to measure the improvement.

This workflow separates **performance measurement** from **performance diagnosis**.

---

# Understanding the Output

Scalene reports several useful performance metrics.

| Metric | Description |
|---------|-------------|
| Python CPU | Time spent executing Python code |
| Native CPU | Time spent inside compiled libraries |
| System CPU | Operating system overhead |
| Memory Allocation | Memory allocated by each line |
| Memory Growth | Net memory increase |
| Copy Volume | Amount of memory copied between objects |

Together, these metrics provide a detailed view of CPU and memory behavior during benchmark execution.

---

# JSON Output

Scalene profiles are commonly stored under

```text
results/scalene/
```

A typical profile contains

- execution metadata;
- command-line arguments;
- elapsed runtime;
- per-file statistics;
- line-level CPU utilization;
- memory allocation information.

Because profiles are stored in JSON format, they can be archived, compared across optimization iterations, and analyzed programmatically.

---

# When Should Scalene Be Used?

Scalene is particularly useful when

- optimizing benchmark implementations;
- investigating unexpectedly slow benchmarks;
- identifying excessive memory allocations;
- validating optimization efforts.

Routine benchmark execution does **not** require Scalene. It is intended as a development and performance analysis tool.

---

# Benchmarking vs Profiling

Although closely related, benchmarking and profiling answer different questions.

| Benchmarking | Profiling |
|--------------|-----------|
| Measures performance | Explains performance |
| Produces reproducible metrics | Identifies bottlenecks |
| Generates benchmark reports | Guides optimization |
| Compares implementations | Analyzes implementation details |

See **Benchmark Methodology** for details on how benchmark measurements are collected and reported.

---

# Further Reading

For installation instructions, advanced usage, and additional examples, see the official **Scalene** repository:

https://github.com/plasma-umass/scalene

---

# Related Documentation

See also

- **Benchmark Methodology**
- **Development**
- **Outputs**
- **Benchmark Matrix Runner**
