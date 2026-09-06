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

默认使用 `ModelConfig.comm_alg = "urma"`、`ModelConfig.mode = 3`，运行多配置带宽测试。
模式 3 使用 Torch context、URMA batch 接口和 Win 直接打流。
默认每个目标卡固定由一个 AIV 负责；开启共享 jetty 后，tiling 仍按原有 `aiv_num` 规则启动 AIV，
仅前 `shared_jetty_num` 个核按 BS 分片并各自独占一个共享 jetty，其余核参与状态等待和接收搬运。
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
is_precision_test = False # 是否进行精度测试
is_bandwidth_test = True  # 是否进行带宽测试
is_single_test = False    # True: 单配置测试; False: 多配置批量测试
```

### 核心配置

`graph_type`、`mode`、`comm_alg`、`batch_tokens`、`shared_jetty` 和 `shared_jetty_num`
位于 `ModelConfig`，核数配置位于 `ParallelConfig`。

```python
graph_type = 3           # 0: 单算子模式; 3: aclgraph模式
comm_alg = "urma"        # mte 或 urma
shared_jetty = False     # True: URMA 共享 jetty；MTE 不支持
shared_jetty_num = 1     # 共享 jetty 及发送核数量；tiling 校验不超过所选 AIV 核数且最多 56
mode = 3                # URMA直接Win打流；精度测试改为0或None
batch_tokens = 0         # URMA: 0 自动最大批量（最高 2048）；正数超限由 tiling 报最大值
is_full_core = True      # True: 使用全部核心; False: 使用部分核心
partial_core_num = 16    # 非全核时的 AIC 数量，AIV 数量由脚本自动取其两倍
aiv_num = None          # 算子 AIV 核数，None 使用自动值，也可单独指定
```

`batch_tokens` 表示每张目标卡每次 BatchCommit 的 token 数，只有最后一批允许不足。
BS 不受单批容量限制，tiling 的最大提交数也不按 BS 截断；MTE 使用 `batch_tokens=0`。
调用方必须保证通信参数有效，并自行确认选定批量小于所有使用的 SQ 和 CQ 深度（包括 0/最大值模式）。
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
# ModelConfig.mode 为 None 时自动决定：精度为 0，带宽为 1；显式设为 3 启用 URMA Win 打流
comm_alg = "urma"        # 默认通信算法
mode = 3                # 默认URMA Win打流
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

下列测试开关修改 `RunConfig`，`mode` 修改 `ModelConfig`，`bs`、`data_size`、`max_bs` 修改 `ShapeConfig`。

### 示例 1：单配置精度测试

```python
mode = 0
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
1     1        1024       0.0000           0              0.0000           0.00             0.00             5.00
2     2048     8192       28.0000          256            4.0000           136.72           151.91           200.00
3     8192     8192       112.0000         512            8.0000           136.72           151.91           800.00
...
```

示例仅演示报告格式，使用 FP16、URMA 模式 3、8 卡、`batch_tokens=512`，耗时和带宽为示意数值。
MB 按 `1024²` 字节计算，`Rank Data = (BS // world_size) × (world_size - 1) × H × 元素字节数`。
`Batch Tokens = min(选定批量, BS // world_size)`，`Batch Data = Batch Tokens × H × 元素字节数`，
表示该用例实际最大的单次远端数据提交；有多次提交时，最后一批仍可能更小。
自动批量 `0` 先按当前 tiling 规则选择 2048，再按实际目标卡数据量计算显示值；请求值仍在表头日志中显示。
无远端数据时两列显示 0；MTE 显示 `N/A`。
模式 0/1 的 `Rank Data` 仍包含本卡目标数据，`Batch Tokens = min(选定批量, ceil(BS/world_size))`，与脚本均衡分配的输入一致。

采集循环只运行 BandwidthTest，已去掉辅助张量和 `torch.exp` 负载。
profiler 配置恢复到已验证可采集的 `124d318a` 基线：`Level1`、`PipeUtilization`、`profile_memory=True`；
输入准备、参数检查、日志和结果校验位于 kernel 计时之外。
报告基于 `op_summary` 的算子执行时间，包含必要的搬运、通信及完成等待，并非裸链路峰值。

## 核心函数说明

### URMA 极限带宽模式 3

`config.py` 默认设置 `ModelConfig.comm_alg = "urma"`、`ModelConfig.mode = 3`，
开启带宽测试并关闭精度测试。`graph_type` 仍独立选择 eager 或 aclgraph。
调节 `batch_tokens`（0 自动选择 2048）和 `ShapeConfig.data_size`（H）观察带宽；BS 决定发送持续量，
应足够大，使 `BS // world_size` 能形成多次满批提交。

普通 URMA 下每个目标卡由一个 AIV 核负责，目标卡数多于核数时按核号步进处理。
共享 jetty 下由前 N 个核共同拆分每个对端的 BS，N 为 `shared_jetty_num`。
每个远端目标发送 `BS // world_size` 个 token，自卡跳过，除法余数不参与传输。
不足一条 token 时只交换完成 flag，数据带宽为 0。
`Rank Data` 和带宽分子均使用实际远端字节数：
`(BS // world_size) * (world_size - 1) * H * 2`；`Batch Data` 按实际最大单次提交计算，不超过每个目标卡的数据量。

kernel 直接从已注册 Win 的只读源区向远端独立接收区逐 token 调用 Batch Write，
只分配 WQE buffer，不读取 x 或目标 ID，不做 token 搬运、统计、输出拷贝或双缓冲初始化。
所有 URMA 模式直接调用 `hcomm_.WriteNbi(handle, dst, src, len)`，不传 WQE 配置，完全使用 SDK 默认值。
所有 URMA 模式通过公开 Hcomm 接口提交通信，不读取 channel entity、队列深度或 handle 内部字段。
每次凑够 batch 就 Commit，再通过 Drain 确认该批完成；不根据私有队列信息决定在途批次数。
普通 URMA 模式 3 的尾批未满时，将 flag 接在尾批后一起 Commit，数据与 flag 合计不超过配置批量；
SDK 默认的 fence 保证 flag 在前面的数据完成后到达，最终一次 Drain 收齐尾批数据和 flag 的完成通知。
数据恰好满批时先完成该批，再单独提交并完成 flag。共享 jetty 模式始终先完成数据批，
再单独 batch 提交并完成 flag。两条路径的报告都只计算数据 token 和字节，不计 flag。
最后参与发送的各核等待负责的来源卡 flag，全部核退出后算子才结束。

flag 使用 Win 中独立的递增序号，避免清零和旧 flag 误判；Win 在通信接口初始化时已清零。
同一 context 的调用必须串行，所有 rank 的模式、shape、当次核数和调用顺序必须一致。
普通 URMA 路径改变核数需要重建 context；共享 jetty 路径使用公共双缓冲位，同步完成后可以改变
`aiv_num`，但仍须满足 `aiv_num >= shared_jetty_num`。
此模式不校验数据内容，y 和 receive_cnt 不保证有效；脚本不初始化输入内容、不保存或回传无效输出。
输出 tensor 形状沿用接口约定。窗口最小值为 `1 MiB + 2 * BS * H * 2`，包含分离的源区和接收区。
Batch buffer 复用不代表网络完成；报告时间包含 CQE 回收及 flag 等待。
普通 URMA 的模式 3 与模式 0/1 共用 `BandwidthTestUrma`，在 `Process()` 的 `mode_ == 3` 分支执行；
共享 jetty 使用 `BandwidthTestUrmaSharedJetty` 和共享 tiling key。两者的 `Init()` 都只读取该模式需要的
通信参数便返回，不初始化 mode 0/1 的业务缓冲区。
超出目标卡范围的核在读取通信 context 前返回；只负责自卡的核不分配 WQE buffer 或更新序号。

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
