# Kernel Launch 头开销测试指南

## 概述

本指南说明如何在 Ascend NPU 环境上复测 add 和 grouped_matmul 两个算子的 kernel launch 头开销数据（满核）。

**原理**：在 kernel 计算主流程前插入 `return;`（early return），让 kernel 只跑初始化部分（tiling 解析 / buffer 初始化），跳过实际计算。用 msprof capture 模式采集的 Task Duration 即为"头开销"。

## 环境要求

| 项目 | 要求 |
|------|------|
| NPU | Ascend 950PR (A5) 或 Ascend910 (A3) |
| CANN | 9.1.0+ (需含 msprof) |
| torch | 2.7.1+ |
| torch_npu | 2.7.1.post4+ |
| Python | 3.12+ |

> **注意**：A5 需要 torch_npu 2.7.1.post4+（支持 Ascend950）。A3 对 torch_npu 版本要求较低。

## 一键执行

```bash
# 1. Clone 仓库（token 已内嵌）
git clone -b head-overhead-clean https://oauth2:NpWjsPrRrAxRsyv5sYqKhCMi@gitcode.com/liulinxiang/ops-transformer.git
cd ops-transformer/examples/fast_kernel_launch_example

# 2. Source CANN 环境
source <CANN安装路径>/set_env.sh

# 3. 一键执行
bash run_head_overhead.sh
```

脚本会自动完成：
1. npu-smi 自动检测平台（A5→ascend950, A3→ascend910_93）
2. 查询设备核数（Cube / Vector）
3. 通过 sed 修改源码核数，构建满核 wheel
4. 用 msprof capture 模式采集 add + grouped_matmul 两组头开销数据（101 次循环/组）
5. 汇总输出结果表 + `summary.csv`

采集结果默认保存在 `head_overhead_results/` 目录下。

## 参考数据（Ascend950PR, dav-3510, capture 模式, 满核）

> 设备：28 Cube 核 / 56 Vector 核

| 算子 | 类型 | 核数 | 稳态 mean(us) |
|------|------|------|-------------|
| **add** | Vector | 56核 | 3.16 |
| **grouped_matmul** | Cube | 28核 | 2.56 |

> A3 平台：add 满核 48核，gmm 满核 24核。

**关键发现**：
- add 满核头开销 3.17us，主要来自 3 次 `InitBuffer`（UB 内存分配 + 同步信号量注册）
- gmm 满核头开销 2.56us，主要来自 `mm.Init`（Matmul tiling 下发到硬件）
- gmm 头开销低于 add，因为 Cube 核的 L0 buffer 是硬件固定的，`Init` 只传指针；Vector 核的 UB 需要运行时切分

## 目录结构

```
examples/fast_kernel_launch_example/
├── csrc/                          # 算子源码
│   ├── add/ascend950/             # add 算子 (A5)
│   ├── add/ascend910_93/          # add 算子 (A3)
│   ├── grouped_matmul/ascend950/  # grouped_matmul 算子 (A5)
│   └── grouped_matmul/ascend910_93/ # grouped_matmul 算子 (A3)
├── tests/                         # 测试脚本 (101次循环, 无精度校验)
│   ├── add/test_add.py
│   └── grouped_matmul/test_groupedmatmul.py
├── run_head_overhead.sh           # 一键测试脚本
├── scan_replay.py                 # NPUGraph capture 采集脚本
├── HEAD_OVERHEAD_GUIDE.md         # 本文档
├── 手动操作步骤.md                  # 简短操作步骤
├── CMakeLists.txt
├── setup.py
└── build_and_test.sh              # 原始构建+测试脚本
```
