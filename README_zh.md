[English](README.md)｜[简体中文](README_zh.md)

# 项目概述

这个项目是一个使用 MPI 和 CUDA 的分布式 QR 分解求解器。

## 项目结构

项目的目录结构如下：
- src/：项目的源代码。
- include/：项目的头文件。
- test/：项目的测试文件。
- bench/：TSQR 与 cuSOLVER QR 的基准测试。
- third_party/：项目的第三方库。

## 第三方库

我们依赖 Git 子模块来管理一些第三方库。
你可以通过运行以下命令来初始化子模块：

```
git submodule update --init --recursive
```

## 基准测试

构建并运行 TSQR 与 cuSOLVER GEQRF 的性能对比：
```
cmake -S . -B build
cmake --build build
./build/bench/bench_tsqr --m 32768 --iters 20 --type float
```

构建并运行分块 QR 基准测试：
```bash
./build/bench/bench_qr --m 16384 --n 16384 --nb 1024 --iters 10 --warmup 2 --type float
```

`bench_qr` 参数：
- `--m <int>`：矩阵行数（`m`），默认 `16384`。
- `--n <int>`：矩阵列数（`n`），默认 `16384`。
- `--nb <int>`：分块 QR 的外层块宽（`nb`），默认 `1024`。
- `--iters <int>`：基准测试迭代次数，默认 `10`。
- `--warmup <int>`：预热迭代次数，默认 `2`。
- `--type <float|double|fp64>`：数据类型，默认 `float`。
- `--run_geqrf <bool>` 或 `--run-geqrf <bool>`：开启/关闭 cuSOLVER GEQRF 路径，默认 `true`。
- `--no-geqrf`：关闭 GEQRF 路径的简写。
- `--with-q`：包含显式 Q 生成的计时。
- `--with-q-batched`：额外加入 StridedBatchedGEMM 的显式 Q 计时（隐含 `--with-q`）。
- `--panel-backend <tsqr|cusolver>`：选择 blocked QR 的 panel 分解后端。`cusolver` 会对每个 panel 执行 `geqrf+orgqr`，把输出对齐到现有 TSQR/WY 路径，适合做消融实验。
- `--trail-one-shot`：使用 one-shot 的 trailing-update GEMM。
- `--trail-tiled`：使用分块（tiled）的 trailing-update GEMM。
- `--trail-tile-cols <int>`：分块 trailing-update GEMM 的列方向 tile 宽度。未指定（或设置为 `0`）时，默认使用 `nb`。

`bench_qr` 参数约束：
- `m > 0`，`n > 0`，`nb > 0`。
- `m >= n`。
- `n` 与 `nb` 必须是 `32` 的倍数。
- `trail_tile_cols` 必须满足 `>= 0`（`0` 表示“使用 `nb`”）。
- 当 `--with-q` 与 `--run_geqrf 0` 同时使用时，会跳过 cuSOLVER GEQRF/ORGQR 计时。

自定义 tiled 宽度示例：
```bash
./build/bench/bench_qr --m 2048 --n 2048 --nb 256 --trail-tiled --trail-tile-cols 96 --iters 10 --warmup 2
```

包含显式 Q 计时的示例：
```bash
./build/bench/bench_qr --m 8192 --n 8192 --nb 512 --with-q --iters 10 --warmup 2
```

使用 cuSOLVER panel backend 做消融的示例：
```bash
./build/bench/bench_qr --m 8192 --n 8192 --nb 512 --panel-backend cusolver --iters 10 --warmup 2
```

## 分布式 Col-Blockcyclic

仓库里也提供了 1D 列 block-cyclic 布局下的分布式 blocked QR 和 explicit Q 基准测试。

分布式 blocked QR 基准测试：
```bash
mpirun -np 4 --mca pml ucx --mca coll_hcoll_enable 0 \
  ./build/bench/bench_dist_blocked_qr_col_blockcyclic \
  --m 65536 --n 65536 --nb 1024 --block_cols 1024 \
  --iters 2 --warmup 3 --panel-buffers 2 \
  --panel-comm broadcast --broadcast-mode block \
  --update_tile 1024 --store-wy none
```

分布式 explicit Q 基准测试：
```bash
mpirun -np 4 --mca pml ucx --mca coll_hcoll_enable 0 \
  ./build/bench/bench_dist_orgqr_col_blockcyclic \
  --m 8192 --n 8192 --nb 1024 --block_cols 1024 \
  --iters 2 --warmup 1 --panel-buffers 2 \
  --panel-comm broadcast --broadcast-mode block --store-wy compact --e2e
```

分布式 col-blockcyclic benchmark 常用参数：
- `--block_cols <int>`：block-cyclic 的拥有粒度，必须是正数并且是 `nb` 的整数倍。
- `--panel-comm <sendrecv|broadcast>`：选择 panel 通信方式。
- `--broadcast-mode <panel|block|block-a|block-yt>`：为 factorization 路径选择 panel 级广播、完整 block `W/Y` 广播、只广播 block `A`，或广播 block `Y/T`。`block-a` 只发送 factorize 后的 block `A`，由接收方本地重建 block `W/Y`，因此 block 级通信量减半。`block-yt` 发送 compact `Y` 和较小的三角 `T`，接收方本地重建 `W = Y * T^T`。它既适用于 `bench_dist_blocked_qr_col_blockcyclic`，也适用于 `bench_dist_orgqr_col_blockcyclic --e2e`。
- `--overlap_tile <int>` 或 `--update_tile <int>`：尾面板更新的 tile 宽度。`0` 表示 one-shot update。
- `--panel-buffers <int>`：panel pack buffer 的个数，至少为 `2`。
- `--compact-local-gemm` / `--segmented-local-gemm`：选择本地尾面板更新实现。
- `--store-wy <none|dense|compact>`：persistent WY 的存储模式。

`--store-wy` 的行为：
- `none`：不分配 persistent `W/Y`。这是 `bench_dist_blocked_qr_col_blockcyclic` 的默认模式，适合只测 factorization、显存占用最低。
- `dense`：保留原来的 dense 本地 `W/Y` 布局。
- `compact`：把 persistent `W/Y` 按本 rank 拥有的 outer block 压缩存进一个 arena。这是 `bench_dist_orgqr_col_blockcyclic` 的默认模式。

说明：
- `bench_dist_orgqr_col_blockcyclic` 默认只测 explicit Q。加上 `--e2e` 后会把 factorization 也计入计时。
- `bench_dist_orgqr_col_blockcyclic` 要求 `--store-wy dense` 或 `--store-wy compact`，不接受 `--store-wy none`。

分布式正确性测试：
```bash
mpirun -np 4 --mca pml ucx --mca coll_hcoll_enable 0 \
  ./build/test/test_dist_blocked_qr_col_blockcyclic_correctness
```
