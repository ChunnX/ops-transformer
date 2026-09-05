# Bandwidth Test 使用说明

## 概述

`bandwidth_test.py` 是一个用于测试 NPU 带宽测试算子 (`bandwidth_test`) 的脚本，支持精度验证和性能测试两种模式，可进行单配置或多配置批量测试。

## 功能特性

- **精度测试**：验证算子输出与 Golden 数据的一致性
- **带宽测试**：测量通信带宽性能，支持 profiling 性能采集
- **多配置测试**：支持多组数据大小的批量测试，自动生成性能汇总报告
- **通信域复用**：多配置测试时共享通信域，减少创建/销毁开销

## 环境要求

- Python 3.8+
- PyTorch
- TorchNPU
- cann_ops_transformer（安装仓库根目录的 torch_extension）
- pandas, numpy

## 快速开始

Torch 接口、context 生命周期和 `aclnn_graph` 示例见
[Torch Extension 说明](torch_extension/README.md)。

```bash
python bandwidth_test.py
```

## 配置说明

`ModelConfig.comm_alg` 可设为 `"urma"`，使用 Torch context、URMA batch 接口和 Win 中转区。
每个目标卡固定由一个 AIV 负责，handle 在该卡的发送循环外创建一次并复用。
精度模式和带宽模式沿用原配置开关，URMA 的窗口容量和依赖见
[Torch Extension 说明](torch_extension/README.md)。

以下配置统一在同目录的 `config.py` 中修改，仍运行 `python bandwidth_test.py`。
配置按用途分为 `ShapeConfig`、`ParallelConfig`、`ModelConfig`、`RunConfig` 和
`DiagnosticsConfig`，每个字段都注明用途。直接修改对应类的字段默认值即可，脚本通过
`CONFIG.shape`、`CONFIG.parallel`、`CONFIG.model`、`CONFIG.run`、`CONFIG.diagnostics` 读取。
以下代码块展示要修改的字段值，不要另加同名的模块级变量或给冻结的 `CONFIG` 字段赋值。
核数、world size、算子 mode 和 HCCL 缓冲区大小由测试脚本根据基础配置自动计算。
图模式外层采集次数、采样条数、通信端口、输出目录和精度容差也在 `config.py` 中配置。
`profiling_path` 请使用专用结果目录：每次带宽测试会清理该目录及其 `_config_数字` 同级目录。

### 测试模式配置

修改 `RunConfig` 中的字段。

```python
# 带宽测试与精度测试不能同时使能
is_precision_test = True   # 是否进行精度测试
is_bandwidth_test = False # 是否进行带宽测试
is_single_test = True     # True: 单配置测试; False: 多配置批量测试
```

### 核心配置

`graph_type`、`comm_alg` 和 `batch_tokens` 位于 `ModelConfig`，核数配置位于 `ParallelConfig`。

```python
graph_type = 0            # 0: 单算子模式; 3: aclgraph模式
comm_alg = "urma"        # mte 或 urma
batch_tokens = 0         # URMA: 0 自动最大批量（最高 2048）；正数超限由 tiling 报最大值
is_full_core = True      # True: 使用全部核心; False: 使用部分核心
partial_core_num = 16    # 非全核时的 AIC 数量，AIV 数量由脚本自动取其两倍
aiv_num = None          # 算子 AIV 核数，None 使用自动值，也可单独指定
```

`batch_tokens` 表示每张目标卡每次 BatchCommit 的 token 数，只有最后一批允许不足。
BS 不受单批容量限制，tiling 的最大提交数也不按 BS 截断；MTE 使用 `batch_tokens=0`。
调用方必须保证通信参数有效，并自行确认选定批量小于所有使用的 SQ 深度（包括 0/最大值模式）。
kernel 不做参数校验、打印或自动缩小批量；UB 预算与完成流控说明见 [Torch 扩展说明](torch_extension/README.md)。

### 性能采集配置

修改 `RunConfig` 中的字段。

```python
loop = 100               # 总迭代次数
repeat = 1               # 重复次数
warmup = 5               # 预热迭代次数
profiling_path = './Profiling_Result_bandwidth_test'  # Profiling 结果路径
```

### 通信域配置

修改 `ParallelConfig` 中的字段。

```python
server_num = 1           # 服务器数量
is_cloud = False         # False: 本地环境; True: 云环境
rank_per_dev = 8         # 每台服务器的 NPU 数量
# world_size 由 server_num * rank_per_dev 自动计算，无需配置
```

**云环境配置**：当 `is_cloud = 1` 时，需要设置环境变量：

- `VC_TASK_INDEX`: 当前任务索引
- `VC_WORKER_HOSTS`: Worker 主机列表

### 数据配置

修改 `ShapeConfig` 中的字段。

```python
x_dtype = 'fp16'         # 数据类型: 'fp16'
random_seed = 90         # 随机种子
```

### 算子配置

修改 `ModelConfig` 中的字段。

```python
# mode 由 is_precision_test 自动决定：精度为 0，带宽为 1
comm_alg = "mte"         # 通信算法
```

### 测试数据配置

在 `RunConfig` 中选择 `is_single_test`，输入尺寸和列表在 `ShapeConfig` 中修改。
多配置用例编辑 `test_configs` 的 `default_factory` 返回列表。

**单配置测试** (`is_single_test = True`)：

```python
bs = 2                   # Batch size
data_size = 8192         # 数据维度 H
max_bs = 2               # 与 bs 相等；不受 URMA 单批提交容量限制
```

**多配置测试** (`is_single_test = False`)：

```python
test_configs = [
    {'bs': 1, 'data_size': 1024, 'max_bs': 1},     # 2KB
    {'bs': 2, 'data_size': 1024, 'max_bs': 2},     # 4KB
    {'bs': 2, 'data_size': 2048, 'max_bs': 2},     # 8KB
    # ... 更多配置
]
```

### 其他配置

`is_save_tensor_to_bin` 位于 `DiagnosticsConfig`，`is_check_setting` 位于 `RunConfig`。

```python
is_save_tensor_to_bin = False  # 是否保存 tensor 到 bin 文件
is_check_setting = True        # 是否检查配置有效性
```

## 使用示例

下列模式开关修改 `RunConfig` 的字段默认值，`bs`、`data_size`、`max_bs` 修改 `ShapeConfig` 的字段默认值。

### 示例 1：单配置精度测试

```python
is_precision_test = True
is_bandwidth_test = False
is_single_test = True
bs = 4
data_size = 4096
max_bs = 4
```

### 示例 2：单配置带宽测试

```python
is_precision_test = False
is_bandwidth_test = True
is_single_test = True
bs = 64
data_size = 8192
max_bs = 64
```

### 示例 3：多配置批量带宽测试

```python
is_precision_test = False
is_bandwidth_test = True
is_single_test = False
# 使用预设的多组配置
```

## 输出说明

### 精度测试输出

```
[PASS] rank_0_y 精度对比通过
[PASS] rank_0_receive_cnt 精度对比通过
```

### 带宽测试输出

```
============================================================
Bandwidth Test Performance Results
============================================================
Single rank data: 128.0000 KB
Total data per iteration: 256.00 MB
World size: 2
BS per rank: 64
Data size (H): 4096
------------------------------------------------------------
Mean execution time: 123.56 us
Min execution time:  60.00 us
Max execution time:  633.05 us
------------------------------------------------------------
Mean bandwidth: 77.44 GB/s
Max bandwidth:  113.33 GB/s
============================================================
```

### 多配置测试汇总表

```
==========================================================================================
Bandwidth Test Summary - All Configurations
==========================================================================================
No.   BS       H          Rank Data(MB)    Batch Tokens   Batch Data(MB)   Mean BW(GB/s)    Max BW(GB/s)     Mean Time(us)
------------------------------------------------------------------------------------------
1     1        1024       0.0020           256            0.5000           45.23            48.12            88.56
2     2        1024       0.0039           256            0.5000           78.45            82.31            102.34
...
```

示例使用 FP16、URMA、`batch_tokens=256`。MB 按 `1024²` 字节计算：
`Rank Data = BS × H × 元素字节数`，包含本卡目标数据；
`Batch Data = batch_tokens × H × 元素字节数`，表示每个目标卡的配置满批数据量，实际尾批可以更小。
自动批量 `0` 显示 `Auto`，容量由 tiling 决定且最高 2048；MTE 显示 `N/A`。
带宽数值继续沿用原来的跨卡数据量计算口径。

采集循环只运行 BandwidthTest，已去掉辅助张量和 `torch.exp` 负载。
profiler 使用 `Level0` 和 `AiCoreNone`，关闭额外的 AI Core 指标与内存采集；
输入准备、参数检查、日志和结果校验位于 kernel 计时之外。
报告基于 `op_summary` 的算子执行时间，包含必要的搬运、通信及完成等待，并非裸链路峰值。

## 核心函数说明

| 函数名 | 功能 |
|--------|------|
| `run_precision_test` | 运行精度测试 |
| `run_bandwidth_test` | 运行单配置带宽测试 |
| `run_multi_config_bandwidth_test` | 运行多配置带宽测试（共享通信域） |
| `generate_inputs` | 生成测试输入数据 |
| `gen_golden_data` | 生成 Golden 参考数据 |
| `compare_result` | 对比 NPU 输出与 Golden 数据 |
| `get_bandwidth_performance` | 从 profiling 结果提取性能数据 |

## 注意事项

1. **通信域创建**：多配置测试时会复用通信域，避免重复创建销毁的开销
2. **HCCL_BUFFSIZE**：脚本会根据测试配置自动计算并设置 `HCCL_BUFFSIZE` 环境变量
3. **Profiling 路径**：多配置测试时，每个配置有独立的 profiling 子目录
4. **内存占用**：大数据量测试时注意 NPU 内存限制
