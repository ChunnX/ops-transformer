# 单共享 Jetty 多对端带宽测试方案

> 执行策略更新：共享 Jetty 已改用 `drain_threshold`（默认 8192）跨批、跨 peer 累计 WQE。
> 下一批将超过阈值时先 Drain，最终发送完成后再 Drain；flag/count 同样占预算。
> 下文逐批/逐 peer Drain 次数和由此推导的台阶属于旧基线，不能直接用于新实现；Commit 次数仍按 payload 分批计算。
> 新实验需固定并记录 drain_threshold，Drain 次数按实际提交序列累计推导，不能直接等同 Commit 次数。


## 1. 目标与适用范围

本文档用于规划和解释“一个共享 Jetty 连接多个远端”的带宽测试，目标是区分以下因素对性能的影响：

- 单条 WQE 的 payload 大小；
- 每个对端的 token 数和单批容量；
- 对端数量及 endpoint 切换开销；
- UB_CTP、UB_RTP 以及混合拓扑；
- 单 Jetty 串行和多 Jetty 并行的差异；
- AIV 数量、eager/aclgraph、profiling 和系统状态造成的测量差异。

主测试使用共享 Jetty mode 3：每个 Jetty 向每个远端发送完整 BS，本卡跳过。
本文公式和矩阵均按此语义计算；`BS` 不再按 world size 放大。

```python
comm_alg = "urma"
shared_jetty = True
shared_jetty_num = 1
mode = 3
```

mode 3 中，一个 Jetty 所属的 AIV 按 rank 顺序逐个处理所有远端。每个远端的数据经过一到多次
`BatchCommit + Drain`，随后单独提交并完成该远端的 flag；所有远端发送完成后，再等待所有来源端的
flag。因此当前测试测量的是：

> 一条共享 Jetty/SQ/CQ 串行切换多个 endpoint 时的 payload goodput，包含 payload Drain、每对端
> flag 提交以及最慢来源端等待。

它不是裸链路峰值，也不是每个对端独立并发的带宽。mode 2/3/4 不校验数据内容，输出张量不能用于
精度判断。每张卡使用 `1 MiB + 2*BS*H*2` 的 Win：状态区后是 BS 源区和 BS 公共接收区。
各来源和 Jetty 直接从 Win 发送，按源 token 编号覆盖公共接收区；带宽统计全部实际发送字节。
多个写入者对同一地址的覆盖可能影响 GM 写入争用，测量结果应按这种公共接收区模型解释。

## 2. 测试模型与计算口径

统一使用以下符号描述工作量，避免只用 BS 导致不同 world size 的测试不可比。

| 符号 | 含义 |
|---|---|
| `W` | `world_size` |
| `P = W - 1` | 每个 rank 的远端数量 |
| `H` | 每个 token 的元素数 |
| `T = H * 2` | 每个 token 的字节数；FP16/BF16 均为 2 字节 |
| `n = BS` | mode 3 中每个 Jetty 发给每个远端的 token 数 |
| `B = min(n, 2048)` | 当前 mode 3 的实际 batch token 数 |
| `D_peer = n * T` | 每个远端的 payload |
| `D_rank = P * n * T` | 单 rank 总 payload |
| `C_peer = ceil(n / B)` | 每个远端的数据 Commit/Drain 次数 |

单 Jetty mode 3 的提交规模为：

```text
Payload WQE/rank       = (W - 1) * n
Payload Commit/rank    = (W - 1) * ceil(n / B)
Flag Commit/rank       = W - 1
总 WQE/rank            = (W - 1) * (n + 1)
```

脚本报告的带宽等价于：

```text
D_rank / kernel_time / 2^30
```

脚本列名当前为 `GB/s`，但计算除数为 `1024^3`，严格来说数值单位是 `GiB/s`。报告中的 payload
字节数不包含 flag、CQE、协议头等控制流量，所以它是应用 payload goodput，不是实际链路总字节率。

`Mean BW` 使用所有选中样本的平均执行时间计算；`Max BW` 使用最短执行时间计算，容易受到偶然快样本
影响，不应单独作为主要结论。

## 3. 需要控制和扫描的因素

### 3.1 单条 WQE 的 payload 大小

`T = H * 2`。建议使用以下点位：

| H | 每个 token/WQE 的 payload |
|---:|---:|
| 256 | 512 B |
| 512 | 1 KiB |
| 1024 | 2 KiB |
| 2048 | 4 KiB |
| 4096 | 8 KiB |
| 8192 | 16 KiB |
| 16384 | 32 KiB |

H 必须保持 16 的倍数并小于 32768。小 payload 更容易受到 WQE、CQE、Commit、Drain 和 endpoint
切换的固定成本影响；payload 增大后，性能会逐渐由链路和 GM 访问能力主导。

mode 3 直接从注册 Win 发数据，现阶段应固定 `x_dtype="fp16"`。FP16 与 BF16 的元素大小相同，扫描
两种 dtype 对主带宽矩阵的解释价值较低，可作为模板路径一致性检查单独测试。

### 3.2 每个对端的 token 数

为了保持每个对端的 token 数明确，应设置：

```text
BS = n
max_bs = BS
```

建议扫描：

```text
n = 1, 2, 4, 8, 16, 32, 64, 128,
    256, 512, 1024,
    2047, 2048, 2049,
    4095, 4096, 4097,
    8192, 16384
```

其中 `2047/2048/2049` 和 `4095/4096/4097` 必须成组测试：

```text
n = 2048：1 次 payload Drain/peer
n = 2049：2 次 payload Drain/peer
n = 4096：2 次 payload Drain/peer
n = 4097：3 次 payload Drain/peer
```

这样可以识别新增尾批和额外 Drain 导致的台阶式性能变化。

### 3.3 endpoint 数量

建议至少测试：

```text
W = 2, 4, 8
```

有足够多机资源时再增加 `W = 16, 32`。endpoint 扩展必须使用两种口径。

固定每个远端 payload：

```text
固定 n、H
BS = n
D_rank = (W - 1) * n * T
```

该口径观察 endpoint 增加时，单 rank 总注入 goodput 能否增长。

固定单 rank 总 payload：

```text
固定 D_rank、H
n = D_rank / ((W - 1) * T)
BS = n
```

该口径主要观察总字节相同时，endpoint 切换、flag 和 Drain 数量增加的代价。应选择可以被
`W - 1` 整除的总 token 数，避免整数除法引入额外差异。

### 3.4 batch 大小

mode 3 忽略 `batch_tokens`，有效值固定为：

```text
min(BS, 2048)
```

因此不能在同一个 mode 3 shape 下独立扫描 batch。要保持“单共享 Jetty、多对端”的数据路径并扫描
batch，可使用：

```python
shared_jetty = True
shared_jetty_num = 1
mode = 2
```

Jetty 数为 1 时，用 mode 2 的 `BS=n*W` 对比 mode 3 的 `BS=n`，可保持每个远端的
payload 相同。两者的 rank 遍历顺序和 flag 路径一致，但源数据范围、Win 大小和写入偏移不同，
因此该对照不能单独隔离 batch 的影响；mode 2 允许显式设置 batch。

推荐固定：

```text
W = 8
H = 2048          # 4 KiB/token
n = 4096
BS = 32768       # mode 2；mode 3 对照使用 BS = 4096
```

分别启动：

```text
batch_tokens = 1, 4, 16, 64, 256, 512, 1024, 2048
```

`batch_tokens` 是全局 ModelConfig 字段，每个值必须单独启动一次，不能在一份 `test_configs` 中逐项
变化。mode 2 的显式值超过 tiling maximum 会失败。最后保持 H 不变，用 `BS=4096` 跑 mode 3，与 mode 2 的
`batch_tokens=2048` 对照。

### 3.5 拓扑和通信协议

在 W 保持为 8 时，建议使用以下拓扑矩阵：

| 拓扑 | `server_num * rank_per_dev` | 每个 rank 的远端组成 | 建议 comm level |
|---|---:|---|---:|
| 纯机内 | `1 * 8` | 7 个机内 peer | 1，强制 UB_CTP |
| 混合 | `2 * 4` | 3 个机内 + 4 个跨机 peer | 0，AUTO |
| 更偏跨机 | `4 * 2` | 1 个机内 + 6 个跨机 peer | 0，AUTO |
| 纯跨机 | `8 * 1` | 7 个跨机 peer | 2，强制 UB_RTP |

`shared_jetty_comm_level` 的含义为：

```text
0：拓扑自动选择，机内 UB_CTP、跨域 UB_RTP
1：所有远端强制 UB_CTP
2：所有远端强制 UB_RTP
```

AUTO 混合 CTP/RTP 时，一个 Jetty 必须能在兼容 layer 和同一 local endpoint 上覆盖全部远端。
如果 context 构建阶段报告没有兼容 layer 或 local endpoint 不一致，应记录为“拓扑不支持该共享
Jetty 组合”，不能作为带宽为 0 的性能样本。

混合场景可能出现队头阻塞：当前单 Jetty 每处理一个 peer 都会 Drain，一个慢 peer 会阻止后续 peer
继续提交，最后的全来源 flag 等待还会放大最慢 rank 的影响。

### 3.6 Jetty 数量和并发方式

主测试固定：

```python
shared_jetty_num = 1
```

建议增加 mode 4 对照，比较“一个 Jetty 串行多个 peer”和“每个 peer 一个 Jetty”。为保持流量一致：

```text
mode 3：shared_jetty_num = 1，BS = n
mode 4：Jetty 数自动为 W - 1，BS = n
```

两者的单 rank payload 都是：

```text
D_rank = (W - 1) * n * T
```

如果 mode 4 明显更快，说明单 Jetty/SQ/CQ、endpoint 串行切换或单 AIV 提交能力可能是主要瓶颈。
如果 mode 4 也不能提升，更可能受到设备总注入、链路或 GM 能力限制。

还可以在 mode 3 下增加 Jetty，但必须缩小每个 Jetty 的 n，以保持总 payload 不变。例如每个 peer
总共发送 4096 token：

| Jetty 数 J | 每个 Jetty/peer 的 n | mode 3 BS |
|---:|---:|---:|
| 1 | 4096 | `4096` |
| 2 | 2048 | `2048` |
| 4 | 1024 | `1024` |

该对照会同时改变每个 Jetty 的 batch/Drain 数量，应作为并发扩展性分析，不能代替纯 batch 实验。

### 3.7 AIV 数量

单 Jetty mode 3 只有第一个 Jetty owning AIV 执行通信，其他 AIV 会提前返回。主基准建议：

```python
aiv_num = 1
```

随后可以用 `aiv_num = 1, 8, 64` 做少量对照。如果 `shared_jetty_num=1` 时性能随 `aiv_num` 明显
变化，该变化不是通信并发带来的，应检查空闲 AIV 调度、kernel block 数、图编译配置和 profiler
统计。

### 3.8 eager 与 aclgraph

主矩阵建议固定：

```python
graph_type = 0
```

随后只选择以下三个代表点对比 `graph_type=0` 和 `graph_type=3`：

- 小消息：`n=16, T=512 B`；
- 一个满批：`n=2048, T=4 KiB`；
- 多批大消息：`n=4096, T=16 KiB`。

当前时间来自 `op_summary` 的 `Task Duration(us)`，主要反映 NPU task，而不是 Python 调用时间。如果
两种 graph type 差异较大，应进一步检查静态 kernel、核数配置和 profiler 文件选择，不应直接归因于
共享 Jetty。

## 4. 推荐实验矩阵

不建议做所有因素的全笛卡尔积。应先执行下面的主效应测试，再根据拐点补充局部组合。

### 4.1 阶段 A：固定每个远端总字节，观察 WQE 粒度

固定：

```text
W = 8
D_peer = 8 MiB
D_rank = 56 MiB
```

| H | T | n/peer | BS | payload batch/peer |
|---:|---:|---:|---:|---:|
| 256 | 512 B | 16384 | 16384 | 8 |
| 512 | 1 KiB | 8192 | 8192 | 4 |
| 1024 | 2 KiB | 4096 | 4096 | 2 |
| 2048 | 4 KiB | 2048 | 2048 | 1 |
| 4096 | 8 KiB | 1024 | 1024 | 1 |
| 8192 | 16 KiB | 512 | 512 | 1 |
| 16384 | 32 KiB | 256 | 256 | 1 |

该组每个 peer payload、单 rank payload 和 mode 3 Win 需求基本相同，只改变 WQE 数量、每条 WQE
payload 和 Drain 次数，最适合区分 WQE/CQE 能力与字节带宽。

### 4.2 阶段 B：固定 WQE 数，观察单 WQE payload

固定：

```text
W = 8
n = 2048
BS = 2048
```

扫描：

```text
T = 512 B, 1 KiB, 2 KiB, 4 KiB, 8 KiB, 16 KiB, 32 KiB
H = T / 2
```

每个 peer 都只有一个满批，Commit/Drain 数量固定，主要观察 payload 增长后的带宽饱和曲线。

### 4.3 阶段 C：batch 边界

固定：

```text
W = 8
H = 2048
T = 4 KiB
```

扫描：

```text
n = 1, 4, 16, 64, 256, 1024,
    2047, 2048, 2049,
    4095, 4096, 4097,
    8192
BS = n
```

### 4.4 阶段 D：endpoint 扩展性

固定 `H=2048, n=2048`：

| W | BS | 每个 peer payload | 单 rank payload |
|---:|---:|---:|---:|
| 2 | 2048 | 8 MiB | 8 MiB |
| 4 | 2048 | 8 MiB | 24 MiB |
| 8 | 2048 | 8 MiB | 56 MiB |
| 16 | 2048 | 8 MiB | 120 MiB |

这组测试总注入扩展性。随后再补一组固定 `D_rank`、改变 W 的实验，测试纯 endpoint 切换代价。

### 4.5 阶段 E：拓扑组成

保持 `W=8`、H、n、BS 完全相同，比较：

```text
1*8、2*4、4*2、8*1
```

每种拓扑至少选择以下三个负载：

- `n=16, H=256`：固定开销主导；
- `n=2048, H=2048`：一个满批；
- `n=4096, H=8192`：大 payload、多批。

### 4.6 阶段 F：对照实验

- shared mode 2、`J=1`：扫描 batch；
- shared mode 3、`J=1`：自动 batch；
- mode 4：每个 peer 一个 Jetty，保持 `D_peer` 一致；
- 可选 ordinary mode 2、`shared_jetty=False, aiv_num=1`：观察共享队列与普通 channel 的差异；
- 选三个代表点比较 eager 和 aclgraph；
- 选三个代表点比较 `aiv_num=1, 8, 64`。

## 5. 基线配置

第一轮建议使用以下基线：

```python
# ParallelConfig
server_num = 1
rank_per_dev = 8
is_full_core = True
aiv_num = 1

# ModelConfig
graph_type = 0
comm_alg = "urma"
shared_jetty = True
shared_jetty_num = 1
shared_jetty_comm_level = 1  # 单机明确测试 UB_CTP
mode = 3
batch_tokens = 0             # mode 3 忽略

# RunConfig
is_precision_test = False
is_bandwidth_test = True
is_single_test = False
loop = 120
repeat = 1
warmup = 20
performance_sample_count = 100

# ShapeConfig
x_dtype = "fp16"
```

每个 mode 3 case 应满足：

```text
max_bs = BS
BS = n
BS >= 1, W > 1
```

一份 `test_configs` 可以同时改变 `bs`、`data_size` 和 `max_bs`。下列参数在一次启动中全局共享，改变
后必须重新启动：

- server/rank 拓扑和 world size；
- mode；
- shared Jetty 数；
- comm level；
- batch tokens；
- AIV 数；
- graph type；
- dtype。

脚本按同一批 `test_configs` 中最大的 Win 需求计算 `HCCL_BUFFSIZE`，并在多个 case 之间复用 context。
比较不同列表时，应让各组包含相同的最大 Win case，或者至少记录实际 `HCCL_BUFFSIZE`，避免窗口容量
和显存布局成为隐藏变量。

## 6. 结果记录要求

### 6.1 配置和派生量

每个样本必须记录：

- run ID、时间和 profiling 目录；
- CANN、Hcomm、torch、torch_npu 和算子包版本；
- server 数、每 server rank 数、world size；
- 本地 peer 数和跨域 peer 数；
- comm level、mode、Jetty 数、AIV 数和 graph type；
- H、token bytes、BS、`n=BS`；
- 实际 batch、每 peer payload、单 rank payload；
- payload WQE 数、payload Commit/Drain 数和 flag Commit/Drain 数；
- HCCL_BUFFSIZE；
- context 构建、kernel 执行和 profiler 解析是否成功。

### 6.2 时间统计

当前汇总表会合并样本。详细分析时应保留原始 `op_summary`，按 rank/设备目录分别计算：

- mean；
- p50、p90、p99；
- min、max；
- 标准差；
- `CV = std / mean`；
- 最慢 rank 与最快 rank 的差距；
- 三次独立作业之间的差距。

主要结论使用 mean/p50；p90、p99 和 rank skew 用于判断抖动。不要只使用由最短时间得到的 Max BW。

### 6.3 归一化指标

除 payload 带宽外，应同时计算：

```text
Payload bandwidth = D_rank / time
WQE rate          = payload_WQE_count / time
Commit rate       = payload_commit_count / time
```

典型解释方式：

- 字节带宽随 H 增长而 WQE/s 接近平台：WQE/CQE 能力可能受限；
- H 增长后字节带宽进入平台：链路或 GM 带宽可能饱和；
- 2049 相比 2048 明显下降：额外尾批 Drain 成本明显；
- W 增长但总带宽不增长：单 Jetty 串行或共享队列可能成为瓶颈；
- mode 4 显著高于 mode 3：多 Jetty/多 endpoint 并行有效；
- mode 4 也不提升：更可能受底层网络、GM 或设备总注入能力限制；
- mixed topology 的 rank skew 明显增大：跨域慢 peer 或队头阻塞可能主导完成时间。

## 7. 环境控制和重复策略

每个测试点至少执行三次独立作业，并固定：

- 软件和算子包版本；
- NPU 频率、功耗模式和温度范围；
- 机器上的其他 NPU/网络负载；
- rank 到设备的映射；
- HCCL 相关环境变量；
- 网络路由、交换设备和服务器位置；
- profiler level 和采样配置；
- shape 列表和执行顺序。

建议分别按正序和倒序运行 shape 列表。如果两种顺序都是后半段普遍变慢，应优先检查温度、频率、
资源泄漏或环境竞争，不应直接归因于 shape。

每次运行使用独立 `profiling_path` 或立即归档原始目录。测试脚本启动时会清理配置目录及对应的
`_config_数字` 目录。

失败样本要分类记录：

- context/layer/local endpoint 不兼容；
- Hcomm/Batch API 返回失败或 kernel Trap；
- SQ/CQ 深度不支持所选 batch；
- 子进程退出但没有 `op_summary`；
- `op_summary` 存在但没有 `BandwidthTest`；
- profiler 有数据但样本数不足。

不能把建链失败、进程失败或 profiler 解析失败当作带宽为 0。

## 8. 当前不改代码无法直接获得的数据

当前实现不能直接给出：

- 每个 peer 独立的开始/结束时间和实际带宽；
- CTP 与 RTP 各自的耗时；
- endpoint 切换的单独耗时；
- payload、Commit、Drain、flag 提交和 flag 等待的阶段占比；
- 同一 mode 3 shape 下独立调节 batch；
- 单源多目标、其他 rank 只接收的单向流量；
- 任意选择部分 peer；
- SQ/CQ 实际深度、游标和占用率。

不改代码时，应通过纯机内、纯跨机、混合拓扑和 mode 2/3/4 的差分测试推断主要瓶颈。完成上述矩阵
后，如果仍需定位到具体 peer 或阶段，再考虑增加以下轻量打点：

```text
peer begin/end
payload commit begin/end
drain begin/end
flag commit/drain
wait flag begin/end
```

在增加打点前，不应把整体 kernel 时间解释成某一个 peer 或某一种协议的独立耗时。
