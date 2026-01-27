[English](README.md)｜[简体中文](README_zh.md)

# 项目概述

这个项目是一个使用 MPI 和 CUDA 的分布式 QR 分解求解器。

## 项目结构

项目的目录结构如下：
- src/：项目的源代码。
- include/：项目的头文件。
- test/：项目的测试文件。
- third_party/：项目的第三方库。

## 第三方库

我们依赖 Git 子模块来管理一些第三方库。
你可以通过运行以下命令来初始化子模块：

```
git submodule update --init --recursive
```