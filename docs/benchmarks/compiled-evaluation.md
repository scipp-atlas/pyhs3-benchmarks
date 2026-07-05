# Compiled Evaluation Benchmark

This benchmark measures the execution cost of evaluating a previously compiled pyHS3 log-probability graph.

Unlike the graph construction or compilation benchmarks, this benchmark excludes workspace loading, model creation, graph construction, and graph compilation. These steps are performed once during setup and are not included in the reported timings. The measured operation is repeated execution of the compiled graph.

## What is measured

For every workspace, target, mode, and evaluation count, the benchmark reports:

- Average wall time per evaluation
- Throughput (evaluations per second)
- Current RSS memory delta
- Peak RSS memory delta

Before timing begins, the benchmark validates that repeated evaluations produce stable finite outputs.

## Running

Run the benchmark directly:

```bash
pixi run python -m src.run_compiled_evaluation \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-evaluations 1 10 100 1000 10000 \
    --output-dir results/docs_examples/function_compilation \
    --plot \
    --plot-dir docs/assets/plots/function_compilation
```

The benchmark can also be executed through the benchmark runner:

```bash
pixi run python -m src.run_all_benchmarks \
    --workspaces \
        inputs/1ch_bkgRooExp_sigGauss_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/3ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
        inputs/5ch_bkgRooExp_sigGeneric_shapeFloat_npOn_constrGauss_yield10x.json \
        inputs/10ch_bkgRooExp_sigGeneric_shapeFloat_npOff_constrGauss_yield1x.json \
        inputs/30ch_bkgGenPoly_sigGeneric_shapeFloat_npOn_constrGauss_yield1x.json \
    --benchmarks compiled_evaluation \
    --targets L_ch0 \
    --modes FAST_RUN \
    --n-evaluations 1 10 100 1000 10000 \
    --output-dir results/docs_examples \
    --plot \
    --plot-dir docs/assets
```

## Example results

### Average evaluation time

![Compiled evaluation average wall time](../assets/plots/compiled_evaluation/compiled_evaluation_average_time.png)

The average evaluation time remains nearly constant for small and moderate numbers of evaluations. As the number of repeated evaluations increases to 10,000, the average execution time increases across all tested workspaces, indicating reduced throughput during long-running evaluation loops.

### Evaluation throughput

![Compiled evaluation throughput](../assets/plots/compiled_evaluation/compiled_evaluation_throughput.png)

Throughput is largely stable between 1 and 100 evaluations and then gradually decreases as the number of repeated evaluations increases. The 10-channel workspace without nuisance parameters achieves the highest throughput across all tested evaluation counts.

### Memory usage

For all tested workspaces, both current RSS delta and peak RSS delta remain equal to zero during repeated compiled graph evaluation. This indicates that evaluating an already compiled graph does not allocate additional persistent memory beyond the initial setup phase.
