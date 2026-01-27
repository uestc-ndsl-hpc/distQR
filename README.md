[English](README.md)｜[简体中文](README_zh.md)

# Project Overview

This project is a distributed QR decomposition solver using MPI and CUDA.

## Project Structure

The project is organized into the following directories:

- `src/`: Source code for the project.
- `include/`: Header files for the project.
- `test/`: Test files for the project.
- `bench/`: Benchmarks for TSQR vs cuSOLVER QR.
- `third_party/`: Third-party libraries for the project.

## Third Party

We depends on git submodules for some third-party libraries.
You can initialize the submodules by running:
```bash
git submodule update --init --recursive
```

## Bench

Build and run the TSQR vs cuSOLVER GEQRF benchmark:
```bash
cmake -S . -B build
cmake --build build
./build/bench/bench_tsqr --m 32768 --iters 20 --type float
```
