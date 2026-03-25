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

## Distributed Col-Blockcyclic

The repository also provides distributed blocked-QR and explicit-Q benchmarks for the 1D column block-cyclic layout.

Distributed blocked-QR benchmark:
```bash
mpirun -np 4 --mca pml ucx --mca coll_hcoll_enable 0 \
  ./build/bench/bench_dist_blocked_qr_col_blockcyclic \
  --m 65536 --n 65536 --nb 1024 --block_cols 1024 \
  --iters 2 --warmup 3 --panel-buffers 2 \
  --panel-comm broadcast --broadcast-mode block \
  --update_tile 1024 --store-wy none
```

Distributed explicit-Q benchmark:
```bash
mpirun -np 4 --mca pml ucx --mca coll_hcoll_enable 0 \
  ./build/bench/bench_dist_orgqr_col_blockcyclic \
  --m 8192 --n 8192 --nb 1024 --block_cols 1024 \
  --iters 2 --warmup 1 --panel-buffers 2 \
  --panel-comm broadcast --broadcast-mode block --store-wy compact --e2e
```

Useful options for the distributed col-blockcyclic benches:
- `--block_cols <int>`: block-cyclic ownership granularity. It must be a positive multiple of `nb`.
- `--panel-comm <sendrecv|broadcast>`: select panel communication mode.
- `--broadcast-mode <panel|block|block-a>`: select panel-level broadcast, full block `W/Y` broadcast, or block-`A`-only broadcast for the factorization path. `block-a` halves block-broadcast traffic by sending only factorized block `A`, then rebuilding block `W/Y` locally on receivers. This applies to `bench_dist_blocked_qr_col_blockcyclic` and to `bench_dist_orgqr_col_blockcyclic --e2e`.
- `--overlap_tile <int>` or `--update_tile <int>`: trailing-update tile width. `0` means one-shot update.
- `--panel-buffers <int>`: number of packed panel buffers. Must be at least `2`.
- `--compact-local-gemm` / `--segmented-local-gemm`: choose local trailing-update implementation.
- `--store-wy <none|dense|compact>`: persistent WY storage mode.

`--store-wy` behavior:
- `none`: do not allocate persistent `W/Y`. This is the default for `bench_dist_blocked_qr_col_blockcyclic` and is the lowest-memory choice for factorization-only runs.
- `dense`: keep the original dense local `W/Y` layout.
- `compact`: store persistent `W/Y` in a compact arena, grouped by owned outer blocks. This is the default for `bench_dist_orgqr_col_blockcyclic`.

Notes:
- `bench_dist_orgqr_col_blockcyclic` runs explicit-Q only by default. Add `--e2e` to include factorization in the timed region.
- `bench_dist_orgqr_col_blockcyclic` requires `--store-wy dense` or `--store-wy compact`; `--store-wy none` is rejected.

Distributed correctness test:
```bash
mpirun -np 4 --mca pml ucx --mca coll_hcoll_enable 0 \
  ./build/test/test_dist_blocked_qr_col_blockcyclic_correctness
```
