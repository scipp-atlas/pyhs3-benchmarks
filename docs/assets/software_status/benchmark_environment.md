# Benchmark Environment

This document describes the reference environment used to develop and evaluate the pyHS3 benchmarking suite.

## Benchmark repository

Repository:
pyhs3-benchmarks

Current benchmark branch:
finalize-benchmarks-before-profiling

Benchmark repository commit:
893c162668754df5e5c74fdfa025d19a9359b6f0

---

## Software versions

Python
3.12.13

ROOT
6.40.02

PyTensor
3.1.2

pyHS3

Version:
0.4.3.dev2+g195fd75f4

Git commit:
195fd75f4f45c843689993c78198107e94105634

xRooFit

Version:
v0.0.4-17-gea0cfde

Git commit:
ea0cfdea6f6ac7f6943eb3b15f530fd2be5e1360

---

## Runtime environment

All benchmarks are executed using

pixi run

to guarantee a reproducible software environment.

---

## Platform

Ubuntu 24.04.4 LTS (WSL2)

CPU

Intel Core i5-12500

12 logical CPUs

RAM

7.6 GiB

---

## Benchmark assumptions

The benchmark environment satisfies the following requirements.

- Environment installs successfully following the project README.

- Minimal benchmark executes successfully.

- All CLI arguments are current.

- No benchmark depends on machine-specific absolute paths.

- All workspace paths are relative to the repository.
