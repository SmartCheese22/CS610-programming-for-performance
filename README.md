# CS610 – Programming for Performance

A collection of programming assignments completed as part of **CS610 (Programming for Performance)** at **IIT Kanpur**.

The repository explores modern techniques for improving application performance through multithreading, SIMD vectorization, OpenMP, GPU programming with CUDA, and low-level optimization on Linux systems.

---

## Overview

This repository contains implementations and experiments focused on writing high-performance software by leveraging both hardware and software optimizations.

Topics covered include:

- Multithreaded programming using C++ threads
- Synchronization with mutexes, condition variables, and atomics
- SIMD vectorization using SSE4 and AVX2 intrinsics
- OpenMP parallel programming
- CUDA programming for GPU acceleration
- Loop transformations and compiler optimizations
- Performance benchmarking and analysis

---

## Repository Structure

```text
.
├── A1/
│   ├── Producer-Consumer implementation
│   └── Thread synchronization
│
├── A2/
│   ├── Performance programming assignments
│   └── C++ implementations
│
├── A3/
│   ├── SIMD (SSE4 / AVX2)
│   ├── OpenMP optimization
│   ├── Parallel grid search
│   └── Compiler optimization experiments
│
├── A4/
│   ├── CUDA programming
│   ├── GPU kernels
│   ├── OpenMP implementations
│   └── Performance evaluation
│
└── README.md
```

---

## Highlights

### Assignment 1
- Producer–Consumer implementation
- Thread-safe bounded buffer
- Mutexes and condition variables
- Atomic synchronization
- Multi-threaded file processing

### Assignment 2
- Performance-oriented C++ programming
- Algorithm optimization exercises

### Assignment 3
- Prefix sum implementation using:
  - Scalar baseline
  - OpenMP SIMD
  - SSE4 intrinsics
  - AVX2 intrinsics
- Parallel grid-search optimization
- OpenMP scheduling and loop transformations

### Assignment 4
- CUDA implementations of computational kernels
- GPU acceleration
- Performance comparison against CPU implementations

---

## Technologies

- C
- C++
- CUDA
- OpenMP
- SSE4 Intrinsics
- AVX2 Intrinsics
- POSIX Threads
- Linux
- GCC
- Make

---

## Building

Most assignments include a Makefile.

```bash
cd A1
make
```

or

```bash
cd A3/submission/problem4-dir
make
```

CUDA assignments require the NVIDIA CUDA Toolkit.

---

## Learning Outcomes

This repository explores several important concepts in high-performance computing, including:

- Thread-level parallelism
- Data-level parallelism (SIMD)
- Shared-memory parallel programming
- GPU computing
- Synchronization primitives
- Compiler-assisted optimization
- Cache-efficient programming
- Performance measurement and benchmarking

---

## Course Information

**Course:** CS610 – Programming for Performance

**Institution:** Indian Institute of Technology Kanpur

---

## License

This repository is intended for educational purposes.
