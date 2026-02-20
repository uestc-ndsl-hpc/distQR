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

Build and run the blocked QR benchmark:
```bash
./build/bench/bench_qr --m 16384 --n 16384 --nb 1024 --iters 10 --warmup 2 --type float
```

`bench_qr` options:
- `--m <int>`: matrix rows (`m`), default `16384`.
- `--n <int>`: matrix cols (`n`), default `16384`.
- `--nb <int>`: outer blocked-QR width (`nb`), default `1024`.
- `--iters <int>`: benchmark iterations, default `10`.
- `--warmup <int>`: warmup iterations, default `2`.
- `--type <float|double|fp64>`: data type, default `float`.
- `--run_geqrf <bool>` or `--run-geqrf <bool>`: enable/disable cuSOLVER GEQRF path, default `true`.
- `--no-geqrf`: shorthand for disabling GEQRF path.
- `--with-q`: include explicit-Q generation timings.
- `--with-q-batched`: add StridedBatchedGEMM explicit-Q timings (implies `--with-q`).
- `--trail-one-shot`: use one-shot trailing-update GEMM.
- `--trail-tiled`: use tiled trailing-update GEMM.
- `--trail-tile-cols <int>`: tile width for tiled trailing-update GEMM. If omitted (or set to `0`), it defaults to `nb`.

`bench_qr` argument constraints:
- `m > 0`, `n > 0`, `nb > 0`.
- `m >= n`.
- `n` and `nb` must be multiples of `32`.
- `trail_tile_cols` must be `>= 0` (`0` means "use `nb`").
- If `--with-q` is used with `--run_geqrf 0`, cuSOLVER GEQRF/ORGQR timings are skipped.

Example with custom tiled width:
```bash
./build/bench/bench_qr --m 2048 --n 2048 --nb 256 --trail-tiled --trail-tile-cols 96 --iters 10 --warmup 2
```

Example with explicit-Q timings:
```bash
./build/bench/bench_qr --m 8192 --n 8192 --nb 512 --with-q --iters 10 --warmup 2
```
