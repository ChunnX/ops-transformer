# URMA P2P Benchmark

## 工具概述

URMA P2P Benchmark 是面向昇腾（Ascend）NPU 平台的 **URMA 协议 Point-to-Point 通信基准测试工具**。该工具专注于测量 NPU 间 URMA 单链路通信的底层性能指标，包括：

- **头开销（Overhead）**：URMA 通信的固定启动成本
- **时延（Latency）**：不同数据包大小下的端到端通信时延及百分位分布
- **带宽（Bandwidth）**：不同数据规模下的 URMA 单链路吞吐量

与现有 `bandwidth_test` 工具（面向 MOE AlltoAll Dispatch 场景的多对多聚合带宽测量）不同，本工具聚焦于 **指定 rank 间的一对一 P2P 链路性能基线**，可更精确地反映 URMA 协议本身的通信能力。

## 目录结构

```
urma_p2p_benchmark/
├── CMakeLists.txt              # 顶层 CMake 构建配置
├── 测试报告模板.md             # 中文测试报告模板
├── docs/
│   └── urma_p2p_benchmark.md   # 工具说明文档
├── op_api/                     # aclnn 算子 API（Host侧调度入口）
├── op_graph/                   # 图模式相关（算子注册/任务生成）
├── op_host/                    # Host侧 Tiling 计算
├── op_kernel/                  # 设备核 kernel 实现
└── tests/                      # 测试代码（需 ENABLE_TEST 开启）
```

## 三种测试模式

### 1. 头开销测试（overhead）

测量 URMA P2P 通信的固定启动成本。通过发送极小数据包（如 1 Byte），多次重复执行并统计通信时间，剥离出与数据量无关的固定开销部分。

- 输出指标：平均时间、最小时间、最大时间、标准差

### 2. 时延测试（latency）

测量不同数据包大小下的端到端通信时延。遍历预定义的数据包大小列表，对每个大小执行多次通信并统计时延分布。

- 输出指标：平均时延、最小时延、最大时延、标准差、P50、P90、P99
- 支持抖动分析

### 3. 带宽测试（bandwidth）

测量不同数据规模下的 URMA 单链路吞吐量。遍历预定义的数据大小列表（从 KB 级到 MB 级），统计各规模的带宽表现。

- 输出指标：平均带宽、最大带宽、平均时间、最小时间
- 支持带宽趋势分析

## 使用方法

### 基本命令

```bash
# 头开销测试
python urma_p2p_benchmark.py --test_type overhead --world_size 2 --dst_rank 1

# 时延测试
python urma_p2p_benchmark.py --test_type latency --world_size 2 --dst_rank 1 --data_sizes 1,4,16,64,256,1024,4096,16384,65536

# 带宽测试
python urma_p2p_benchmark.py --test_type bandwidth --world_size 2 --dst_rank 1 --data_sizes 4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608,16777216,33554432,67108864,134217728
```

### 关键参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--test_type` | 测试类型：overhead / latency / bandwidth | bandwidth |
| `--world_size` | 通信域 rank 总数 | 2 |
| `--rank_per_dev` | 每设备 rank 数 | 1 |
| `--dst_rank` | 目标 rank 索引（P2P 通信对端） | 1 |
| `--data_sizes` | 数据大小列表（Bytes） | bandwidth 模式默认 16 组 |
| `--iterations` | 每组测试的迭代次数 | 100 |
| `--warmup` | 预热迭代次数 | 10 |
| `--dtype` | 数据类型：float16 / bfloat16 / float32 | float16 |

## 编译说明

本工具作为 CANN 算子项目的一部分，遵循标准的 CANN 算子编译流程：

1. 确保 CANN 开发环境已正确安装
2. 在项目根目录执行编译：

```bash
# 编译整个项目（包含 urma_p2p_benchmark）
cd ops-trans-bandwidth
bash build.sh

# 或仅编译本工具
cd mc2/tools/urma_p2p_benchmark
# 通过顶层 CMakeLists.txt 的 add_subdirectory 机制自动纳入编译
```

3. 编译产物将随 CANN 算子包一起部署到 NPU 设备