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

Build and run the distributed col-blockcyclic blocked QR benchmark:
```bash
mpirun -np 4 ./build/bench/bench_dist_blocked_qr_col_blockcyclic \
  --m 32768 --n 32768 --nb 1024 --block_cols 4096 \
  --iters 5 --warmup 2 --type float
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

`bench_dist_blocked_qr_col_blockcyclic` options:
- `--m <int>`: matrix rows (`m`), default `16384`.
- `--n <int>`: matrix cols (`n`), default `1024`.
- `--nb <int>`: outer blocked-QR width (`nb`), default `1024`.
- `--block_cols <int>`: column-blockcyclic block width. If omitted (or set to `0`), it defaults to `nb`.
- `--overlap_tile <int>`: tiled trailing-update width. `0` means one-shot trailing update.
- `--iters <int>`: benchmark iterations, default `3`.
- `--warmup <int>`: warmup iterations, default `1`.
- `--type <float|double|fp64>`: data type, default `float`.
- `--print_per_rank`: print per-rank elapsed time.
- `--print_comm_bw`: print coarse communication event time, bytes, and derived bandwidth.
- `--print_phase_timing`: print per-rank phase timing for panel factorization, WY build, communication, trailing update, and trailing-update GEMM TFLOPS.
- `--panel-comm <sendrecv|broadcast>`: select panel communication path.
- `--broadcast-mode <panel|block>`: when `--panel-comm broadcast` is used, select per-panel broadcast or one-broadcast-per-block mode. Default is `block`.

`bench_dist_blocked_qr_col_blockcyclic` argument constraints:
- `m >= n > 0`.
- `nb > 0`, `nb <= n`, and both `n` and `nb` must be multiples of `32`.
- `block_cols > 0`, `block_cols <= n`, `block_cols % nb == 0`, and `block_cols` must be a multiple of `32`.
- `warmup >= 0`, `iters > 0`.
- `broadcast_mode` is only used when `--panel-comm broadcast`; otherwise it is ignored.

Example comparing panel-broadcast vs block-broadcast:
```bash
mpirun -np 8 ./build/bench/bench_dist_blocked_qr_col_blockcyclic \
  --m 131072 --n 131072 --nb 1024 --block_cols 4096 \
  --panel-comm broadcast --broadcast-mode panel --print_phase_timing

mpirun -np 8 ./build/bench/bench_dist_blocked_qr_col_blockcyclic \
  --m 131072 --n 131072 --nb 1024 --block_cols 4096 \
  --panel-comm broadcast --broadcast-mode block --print_phase_timing
```
