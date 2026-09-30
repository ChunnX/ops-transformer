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
单共享 Jetty 连接多个远端时的因素拆分、实验矩阵和结果解释见
[单共享 Jetty 多对端带宽测试方案](docs/shared_jetty_multi_peer_benchmark_plan.md)。

```bash
python bandwidth_test.py
```

## ROCE 多 SGE 功能测试

`config.py` 当前默认单机两卡、`server_index=0`、`master_ip="127.0.0.1"`，选择 ROCE、mode 0、
eager、FP16、非共享 channel，关闭拓扑扫描和 profiling。直接运行 `python bandwidth_test.py`
即进入多 SGE 功能测试，不要求额外参数。`--roce-multi-sge` 仍可显式选择此入口。
通信拓扑读取 `config.py` 的 `parallel` 字段，也可用下面的命令行参数覆盖；此入口不读取云环境拓扑变量。

**先编译并安装这版算子。** 目标 asc-comm/CANN 必须提供用户约定的 ROCE BatchHandle 接口：
`MakeBatchHandle(channel, ub, bytes, remoteWin, localWin)`、
`WriteNbi(batch, dst, const AscendC::BufDesc *, count)`、`BatchCommit(batch)`、`Drain(batch)`，
以及支持跨核状态交接的 `Lock(channel)` / `Unlock(channel)`。
本地参考 asc-comm 的现有实现只在 URMA 上支持这些操作；旧 ROCE SDK 不能编译本版，
不会自动降级为单 SGE。先让编译环境使用新增 ROCE 实现的头文件/库，再重新生成并安装
`bandwidth_test` 算子包（host tiling 和 kernel）；Torch 包的安装步骤见
[Torch 扩展构建说明](../../../torch_extension/README.md)。
确认实际加载的是此次生成的算子包，旧 kernel 的数值结果相同，不能凭脚本 PASS 判断旧包执行了多 SGE。

发送时每个 SGE 指向一个注册 Win token 槽，同一目标 rank 最多 4 个槽合成一个 WQE。
槽宽为 `align_up(H * 2, 512)` 字节，padding 在发送前清零；接收侧仍只输出每行有效的 H 个元素。
交错路由使源槽之间存在空隙，而远端按 SGE 顺序连续落地。保持原有两核依次发送、核 0 接收的流程，
每个请求均按 `Lock → MakeBatchHandle → WriteNbi → BatchCommit → Drain → Unlock` 执行，
不积累在途请求；count 完成后才发布 ready flag。`Unlock` 写回通道队列计数，
使另一核后续创建 batch 时能够使用更新后的状态；`Drain` 和核间同步不能代替这一步。
这版用于功能验证，不用于带宽比较。新增独立的 256B batch UB（参考 URMA 最大 WQE 容量），
目标 ROCE SDK 需满足 1～4 SGE、单请求及 Drain scratch 能容纳于该缓冲区的约定；接口出错会 Trap。

单机两卡（同机两卡间必须有 ROCE link）：

```bash
cd mc2/tools/bandwidth_test
unset BANDWIDTH_TEST_CASE
python -u bandwidth_test.py
```

双机、每机 8 卡：两台机器分别执行，`MASTER_IP` 都填机器 0 的可达 IP：

```bash
# 机器 0
python -u bandwidth_test.py --roce-multi-sge \
  --server-num 2 --rank-per-dev 8 --server-index 0 \
  --master-ip "$MASTER_IP" --master-port 5001

# 机器 1
python -u bandwidth_test.py --roce-multi-sge \
  --server-num 2 --rank-per-dev 8 --server-index 1 \
  --master-ip "$MASTER_IP" --master-port 5001
```

脚本自行启动本机 rank 进程，不要再套 `torchrun`。所有 rank 对都需要支持 ROCE，包括同机卡之间。
默认只运行一组用例、调用一次，尺寸读取 `config.py` 中 `ShapeConfig` 的 `bs/data_size/max_bs`，
当前为 `BS=max_bs=512`、`H=8192`。可通过 `--iterations N` 增加同组用例的重复次数；
重复时复用同一通信域和注册 Win，并改变输入值。每次检查 `receive_cnt` 和有效输出与 CPU golden 完全一致。

| 用例 | BS | H | 覆盖内容 |
| --- | ---: | ---: | --- |
| scattered | 512 | 8192 | 交错路由、非连续源地址、4 SGE 请求及自发自收 |

最后应打印 `ROCE multi-SGE: PASS`，默认每 rank 1 次检查。
本机子进程报错、计数或数据不一致均以非零状态退出；双机时两台机器都 PASS 才表示全部 rank 通过。
功能入口默认给本机 worker 共 600 秒（含初始化和全部用例），超时会终止本机 worker 并以非零状态退出，
用于发现对端失败或设备等待卡住；首次 JIT 构建较慢时可用 `--timeout-seconds N` 放宽。
`HCCL_BUFFSIZE` 在创建通信组前按全套用例的最大窗口需求自动设置。
本地 Windows 环境的静态检查不等于 CANN 编译或多卡 NPU 功能验证。

## 全 UB_RTP 连续批跑

切回 URMA 带宽扫描时，将 `comm_alg="urma"`、`shared_jetty=True`、`mode=3`、`graph_type=3`，
以及 `sweep_enabled=True`、`is_precision_test=False`、`is_bandwidth_test=True`，并按实际拓扑配置服务器和卡数。
扫描按原有拓扑矩阵和全部 shape 连续运行；所有用例统一使用
`ParallelConfig.master_port=5001`，不按用例递增端口。
保留全 UB_RTP（comm_level=2）、batch_tokens=0、drain_threshold=8192，
图模式每个用例 105 轮（5 轮预热、100 轮采集）。临时设备诊断日志已删除。

两台服务器分别配置 server_index=0/1，master_ip 填同一个正确的主节点地址，再各自运行：

```bash
python -u bandwidth_test.py 2>&1 | tee bandwidth_sweep.log
```

静态图目录为 `static_kernel`。本次修改了 host tiling，安装过 batch 扫描实验版本或诊断版本的服务器，
需重新构建安装算子，并重新生成对应静态内核。

共享 mode 3 忽略显式 batch_tokens，直接取 min(BS, 248 KiB / 64, drain_threshold)，
不再受 2048 的软件截断；当前 UB 布局最多容纳 3968 个 WQE，BS=4096 按 3968+128 提交。
其他模式保留原 2048 上限。3968 是源码缓冲容量，目标 SDK 在这一容量下的执行能力仍需实机验证。
当前恢复原 12 组拓扑、17 个 shape（仅移除 BS=8192），不再单独扫描 batch。

## 配置说明

普通打流可使用 `ModelConfig.comm_alg = "urma"`、`ModelConfig.mode = 2`；当前默认配置为上面的单机两卡 ROCE 功能测试。
模式 2 保留原有 Torch context、URMA batch 接口和 Win 直接打流逻辑。共享 Jetty 下还支持模式 3/4：
模式 3 的每个 Jetty 都向所有远端卡发送完整 `BS` 个 token；模式 4 固定建立
`world_size - 1` 个 Jetty，每个 Jetty 负责一张远端卡并发送完整 BS。
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

`graph_type`、`mode`、`comm_alg`、`batch_tokens`、`shared_jetty`、`shared_jetty_num` 和
`shared_jetty_comm_level`
位于 `ModelConfig`，核数配置位于 `ParallelConfig`。

```python
graph_type = 3           # 0: 单算子模式; 3: aclgraph模式
comm_alg = "urma"        # mte 或 urma
shared_jetty = False     # True: URMA 共享 jetty；MTE 不支持
shared_jetty_num = 1     # 共享 jetty 数量；mode 4 忽略此值并固定使用 world_size - 1
shared_jetty_comm_level = 0  # 仅共享模式生效：0=域内UB_CTP/域间UB_RTP，1=全UB_CTP，2=全UB_RTP
mode = 2                # 2=原Win打流，3=每Jetty发所有卡，4=每远端卡一个Jetty
batch_tokens = 0         # mode 3忽略；mode 4取min(配置或BS, BS, tiling上限)
is_full_core = True      # True: 使用全部核心; False: 使用部分核心
partial_core_num = 16    # 非全核时的 AIC 数量，AIV 数量由脚本自动取其两倍
aiv_num = None          # 算子 AIV 核数，None 使用自动值，也可单独指定
```

`batch_tokens` 通常表示每张目标卡每次 BatchCommit 的 token 数，只有最后一批允许不足。
模式 3 不读取该配置，有效值为 `min(BS, tiling最大值, drain_threshold)`；能一次提交完就一次提交，
否则按 tiling 最大值分批。模式 4 的有效值为 `min(batch_tokens or BS, BS, tiling最大值, drain_threshold)`。
MTE 使用 `batch_tokens=0`。
`shared_jetty_comm_level=0` 沿用拓扑自动选择；1 和 2 分别要求所有远端能够在同一层通过
UB_CTP 或 UB_RTP 建立共享 Jetty。该配置不作为算子属性下发，只用于共享 context 建链。
调用方必须保证通信参数有效，并自行确认选定批量小于所有使用的 SQ 和 CQ 深度（包括 0/最大值模式）。
kernel 不重复校验参数范围、context 与 tiling 的 Jetty 数量或通信返回值；模式 4 的 BS 截断由 tiling 完成。
host 参数和资源检查保留；设备端断言、通信错误分支及 ROCE 调试打印已移除，避免在性能路径中执行。
UB 预算与完成流控说明见 [Torch 扩展说明](torch_extension/README.md)。

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
# ModelConfig.mode 为 None 时自动决定：精度为 0，带宽为 1；显式设为 2 启用 URMA Win 打流
comm_alg = "urma"        # 默认通信算法
mode = 2                # 默认URMA Win打流
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

示例仅演示报告格式，使用 FP16、URMA 模式 2、8 卡、`batch_tokens=512`，耗时和带宽为示意数值。
MB 按 `1024²` 字节计算，`Rank Data = (BS // world_size) × (world_size - 1) × H × 元素字节数`。
`Batch Tokens = min(选定批量, BS // world_size)`，`Batch Data = Batch Tokens × H × 元素字节数`，
表示该用例实际最大的单次远端数据提交；有多次提交时，最后一批仍可能更小。
mode 0/1/2 的自动批量 `0` 先按当前 tiling 规则选择 2048，再按实际目标卡数据量计算显示值；共享 mode 3 直接取 BS、3968 和 Drain 阈值的最小值，不受 2048 截断。请求值仍在表头日志中显示。
无远端数据时两列显示 0；MTE 显示 `N/A`。
模式 0/1 的 `Rank Data` 仍包含本卡目标数据，`Batch Tokens = min(选定批量, ceil(BS/world_size))`，与脚本均衡分配的输入一致。

采集循环只运行 BandwidthTest，已去掉辅助张量和 `torch.exp` 负载。
profiler 配置恢复到已验证可采集的 `124d318a` 基线：`Level1`、`PipeUtilization`、`profile_memory=True`；
输入准备、参数检查、日志和结果校验位于 kernel 计时之外。
报告基于 `op_summary` 的算子执行时间，包含必要的搬运、通信及完成等待，并非裸链路峰值。

## 核心函数说明

### URMA Win 打流模式 2/3/4

`config.py` 默认设置 `ModelConfig.comm_alg = "urma"`、`ModelConfig.mode = 2`，
开启带宽测试并关闭精度测试。`graph_type` 仍独立选择 eager 或 aclgraph。
模式 2 逻辑保持不变：调节 `batch_tokens`（0 自动选择 2048）和 `ShapeConfig.data_size`（H）观察带宽。

普通 URMA 下每个目标卡由一个 AIV 核负责，目标卡数多于核数时按核号步进处理。
共享 Jetty 模式 2 下由前 N 个核共同拆分每个对端的 BS，N 为 `shared_jetty_num`。
每个远端目标发送 `BS // world_size` 个 token，自卡跳过，除法余数不参与传输。
模式 2 不足一条 token 时只交换完成 flag，数据带宽为 0。
`Rank Data` 和带宽分子均使用实际远端字节数：
`(BS // world_size) * (world_size - 1) * H * 2`；`Batch Data` 按实际最大单次提交计算，不超过每个目标卡的数据量。

共享 Jetty 模式 3 不再拆分对端数据。前 N 个核各自独占一个 Jetty，每个核都遍历全部远端卡，
向每张远端卡发送完整 `BS` 个 token；`batch_tokens` 被忽略。实际 batch 为
`min(BS, tiling最大值, drain_threshold)`，因此放得下时一次提交，否则按最大容量分批。
单 rank 流量为 `N * BS * (world_size - 1) * H * 2`，带宽分子使用该实际发送量，不包含 flag。
本卡跳过；允许 `BS >= 1`，仍要求 `bs == max_bs` 和 `world_size > 1`。
接收区由所有来源 rank 和 Jetty 共用，允许覆盖；同一 shape 的 Win 容量不随卡数或 Jetty 数增长。

共享 Jetty 模式 4 忽略配置的 `shared_jetty_num`，脚本建立 `world_size - 1` 个 Jetty，并按跳过本卡的
rank 顺序一一绑定远端卡。每个 Jetty 向负责卡发送完整 BS，单 rank 流量为
`BS * (world_size - 1) * H * 2`。有效批量为
`min(batch_tokens or BS, BS, tiling最大值, drain_threshold)`，超过 BS 或最大容量时直接取对应上限。

kernel 直接从已注册 Win 的只读源区向远端公共接收区逐 token 调用 Batch Write，
只分配 WQE buffer，不读取 x 或目标 ID，不做 token 搬运、统计、输出拷贝或双缓冲初始化。
所有 URMA 模式直接调用 `hcomm_.WriteNbi(handle, dst, src, len)`，不传 WQE 配置，完全使用 SDK 默认值。
所有 URMA 模式通过公开 Hcomm 接口提交通信，不读取 channel entity、队列深度或 handle 内部字段。
每次凑够 batch 就 Commit；共享 Jetty 按 `drain_threshold` 累计在途 WQE，下一批将超过阈值时先 Drain，发送结束再 Drain。
普通 URMA 模式 2 的尾批未满时，将 flag 接在尾批后一起 Commit，数据与 flag 合计不超过配置批量；
SDK 默认的 fence 保证 flag 在前面的数据完成后到达，最终一次 Drain 收齐尾批数据和 flag 的完成通知。
数据恰好满批时先提交该批，再单独提交 flag，按阈值及通道释放前 Drain。共享 Jetty 先提交数据批，
再单独 batch 提交 flag，按阈值及发送结束统一 Drain。两条路径的报告都只计算数据 token 和字节，不计 flag。
模式 2/3 的每个发送核等待所有来源卡同编号 Jetty 的 flag；模式 4 的每个发送核等待一张配对来源卡的 flag。

flag 使用 Win 中独立的递增序号，避免清零和旧 flag 误判；Win 在通信接口初始化时已清零。
同一 context 的调用必须串行，所有 rank 的模式、shape、当次核数和调用顺序必须一致。
普通 URMA 路径改变核数需要重建 context；共享 jetty 路径使用公共双缓冲位，同步完成后可以改变
`aiv_num`，但仍须满足 `aiv_num >= shared_jetty_num`。
此模式不校验数据内容，y 和 receive_cnt 不保证有效；脚本不初始化输入内容、不保存或回传无效输出。
输出 tensor 形状沿用接口约定。模式 2/3/4 的窗口最小值统一为 `1 MiB + 2 * BS * H * 2`：
前 1 MiB 为状态区，随后是 BS 个 token 的只读源区，再后是 BS 个 token 的公共接收区。
源 token `i` 写入目标接收区的 token `i`；模式 2 保留目标分片和分核规则，模式 3/4 发送完整 BS。
不执行 `X → UB → Win` 搬运，接收数据允许并发覆盖且不校验；完成 flag 仍按来源和核独立存储。
带宽分子统计每次实际发送的字节数，包括覆盖写，不使用接收区最终保留的数据量。
Batch buffer 复用不代表网络完成；报告时间包含 CQE 回收及 flag 等待。
普通 URMA 的模式 2 与模式 0/1 共用 `BandwidthTestUrma`，在 `Process()` 的 `mode_ == 2` 分支执行；
共享 jetty 使用 `BandwidthTestUrmaSharedJetty` 和共享 tiling key。Win 打流分支的 `Init()` 只读取需要的
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

### URMA 的 Drain 阈值

`config.py` 中 `ModelConfig.drain_threshold = 8192` 通过 eager/图模式、Torch、ACLNN/GE 和 tiling 传到普通 URMA 和共享 Jetty kernel。
该值为自上次 Drain 后累计提交的 WQE 数上限；普通 URMA 在当前持锁通道内跨 batch 累计，共享 Jetty 还跨目标卡累计；每个 token 为一个 WQE，flag/count 也各占一个，但不计入 payload 带宽。
若累计值加下一批数量大于阈值，先 Drain 已提交请求，再准备并提交下一批；等于阈值允许提交。单批容量同时受阈值限制。
发送结束再 Drain；mode 0/1 还在发布状态前完成数据阶段的 Drain。临时 Drain 失败 Trap 已移除。
阈值适用于普通 URMA mode 0/1/2 和共享 Jetty mode 0/1/2/3/4；MTE、ROCE 的提交等待策略不变。阈值范围为 `[1, UINT32_MAX]`，实际值须符合部署 SDK 的 SQ/CQ 可用容量。
新增可选算子属性后，需重新生成/安装算子包和 Torch 扩展，并重建静态图内核。

普通 URMA 的累计量在持锁期间跨批保留，最终 Drain 后才释放锁并切换通道；共享 Jetty 跨目标卡继续累计。
普通 mode 0/1 扫描 ID 时才知道尾批大小，因此开始新批时按满批预留预算（已知为单条尾批时按 1）；可能提前 Drain，避免额外扫描。
普通 mode 2 按实际下一批 WQE 数检查，尾批合入 flag 时预留其 1 个 WQE。所有 URMA 单批容量均受 drain_threshold 限制。


## 一次批跑 Jetty 数量和发送对端数

切换为 URMA 带宽配置并设置 `RunConfig.sweep_enabled=True` 后，执行 `python bandwidth_test.py` 完成拓扑扫描。
操作步骤、轮转公式、报告字段及重建要求见 [拓扑批跑说明](docs/topology_sweep.md)。
`False` 恢复原单拓扑流程；mode 3 新增 `ModelConfig.peers_per_jetty`，0 表示本文原有的全部远端，正数表示实际发送对端数。
此时每 rank payload 为 `Jetty数 × peers_per_jetty × BS × H × dtype_bytes`，接收等待仅覆盖对应发送方。
