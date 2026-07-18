# Profiling Environment

## Run identity

- Collection date: 2026-07-18
- Collection timestamp: 2026-07-18T06:21:49+00:00
- Time zone recorded by environment: UTC (+00:00)
- Benchmark repository: `scipp-atlas/pyhs3-benchmarks`
- Benchmark repository branch: `main`
- Benchmark repository commit: `ee5f11f4fc367e592bbd6e3fb5e27bb0b26623cb`
- Benchmark repository commit message: `finalized documentation`
- Repository state before profiling directory creation: clean
- Repository state at the start of environment collection: clean
- pyHS3 version: `0.4.3.dev2+g195fd75f4`
- pyHS3 commit: `195fd75f4f45c843689993c78198107e94105634`
- pyHS3 requested Git revision: `main`
- pyHS3 installation type: Pixi/conda source build from Git
- pyHS3 editable installation: no
- pyHS3 installed location: `.pixi/envs/default/lib/python3.12/site-packages/pyhs3`

## Machine

- Execution environment: Windows Subsystem for Linux 2
- CPU: 12th Gen Intel(R) Core(TM) i5-12500
- CPU architecture: `x86_64`
- CPU vendor: GenuineIntel
- Physical CPU cores visible to WSL: 6
- Logical CPUs visible to WSL: 12
- Threads per core: 2
- CPU sockets: 1
- L1 data cache: 288 KiB total, 6 instances
- L1 instruction cache: 192 KiB total, 6 instances
- L2 cache: 7.5 MiB total, 6 instances
- L3 cache: 18 MiB, 1 instance
- Hypervisor: Microsoft
- NUMA nodes: 1

## Memory

- RAM visible to WSL: 7.6 GiB
- Exact MemTotal: 7,998,092 kB
- MemAvailable at collection time: 6,728,540 kB
- MemFree at collection time: 6,409,944 kB
- Swap visible to WSL: 2.0 GiB

The reported memory values describe the resources visible to WSL and may not
equal the total physical RAM installed in the Windows host.

## Operating system

- Linux distribution: Ubuntu 24.04.4 LTS
- Distribution codename: Noble Numbat
- Kernel: `6.18.33.2-microsoft-standard-WSL2`
- Pixi platform: `linux-64`
- glibc virtual package: `2.39`
- WSL distribution: Ubuntu
- CPU frequency governor: not exposed by this WSL environment
- `cpupower`: not installed

## Python environment

- Python: 3.12.13
- Python implementation: CPython
- Python distribution: conda-forge
- Python build compiler: GCC 14.3.0
- Python executable: `.pixi/envs/default/bin/python`
- Pixi: 0.70.2
- Pixi workspace: `pyhs3-benchmarks`
- Pixi environment: `default`
- Pixi channels: `conda-forge`
- Pixi architecture specification: `skylake`

The active Conda shell environment is named `iris-hep`, but benchmark commands
run through `pixi run` and therefore use the Pixi `default` environment.

## Numerical libraries

- NumPy: 2.4.6
- SciPy: 1.18.0
- PyTensor: 3.1.2
- JAX: 0.10.2
- JAXlib: 0.10.2
- ROOT: 6.40.02
- pandas: 3.0.3
- pyhf: 0.7.6
- numba-stats: 0.0.0

## NumPy and BLAS build

- NumPy BLAS detection: successful
- NumPy-reported BLAS compatibility version: 3.9.0
- NumPy LAPACK detection: successful
- NumPy-reported LAPACK compatibility version: 3.9.0
- Conda BLAS provider: Intel MKL
- `blas` package build: `mkl`
- `libblas` package build: `mkl`
- `libcblas` package build: `mkl`
- NumPy build compiler: GCC 14.3.0
- SIMD baseline: X86_V2
- SIMD detected: X86_V3
- SIMD not detected: X86_V4, AVX512_ICL, AVX512_SPR

NumPy's build report exposes the generic BLAS/LAPACK compatibility interface,
while the resolved Pixi/Conda packages show that the concrete runtime provider
is Intel MKL. No explicit MKL thread limit was set during environment
collection.

## Profilers

- Scalene: 1.5.51
- PyInstrument: not installed

## JAX runtime configuration

- Selected JAX backend: CPU
- JAX device count: 1
- JAX device 0 platform: `cpu`
- JAX device 0 kind: `cpu`
- CUDA virtual package visible to Pixi: 12.6
- CUDA-enabled JAXlib installed: no
- GPU used by JAX benchmarks: no

JAX reports that an NVIDIA GPU may be present, but the installed JAXlib does
not contain CUDA support, so execution falls back to CPU.

## PyTensor runtime configuration

- PyTensor version: 3.1.2
- `config.mode`: `Mode`
- `config.floatX`: `float64`
- `config.optimizer`: `o4`
- `config.linker`: `auto`
- `config.cxx`: `.pixi/envs/default/bin/g++`
- `config.allow_gc`: `True`
- `config.exception_verbosity`: `low`
- `config.on_opt_error`: `warn`
- `PYTENSOR_FLAGS`: not set
- Resolved mode class: `Mode`
- Resolved default linker: `NumbaLinker()`
- Resolved optimizer: `RewriteDatabaseQuery`
- Default compiled execution enabled: yes, through the Numba linker
- Benchmark-specific mode overrides: pending stage-boundary audit
- Compilation included in measured regions: pending stage-boundary audit

The value `config.linker = auto` resolves to `NumbaLinker()` in the current
environment. Therefore, PyTensor's default execution path compiles the graph
through Numba. Individual benchmark functions may still override the default
mode or linker, which must be verified from the benchmark implementation.

PyTensor 3.1.2 does not expose all configuration attributes used by older
PyTensor versions, including `pytensor.config.device`.

## Thread and backend environment variables

The following variables were not explicitly set during environment collection:

- `OMP_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `MKL_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`
- `BLIS_NUM_THREADS`
- `JAX_*`
- `XLA_*`
- `PYTENSOR_FLAGS`

## Benchmark policy

- Process isolation: pending definition
- CPU affinity: pending definition
- Thread limits: pending definition
- Warm-up policy: pending definition
- Cold-start policy: pending definition
- First-evaluation policy: pending definition
- Steady-state policy: pending definition
- Batch-evaluation policy: pending definition
- Number of repetitions: pending definition
- Summary statistics: median, mean, standard deviation and/or interquartile range
- Outlier policy: pending definition

## Environment snapshots

- `profiling/environment/environment-raw.txt`
- `profiling/environment/pixi-info.txt`
- `profiling/environment/pixi-packages.txt`
- `profiling/environment/conda-list.txt`
- `profiling/environment/python-packages.txt`
- `profiling/environment/pytensor-config.txt`
