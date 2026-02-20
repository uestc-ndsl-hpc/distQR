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
