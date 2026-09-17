# D&C性能优化

本文记录 MoE Dispatch / Combine 四个算子的性能优化方案，供后续逐项实现和验证。

当前状态：方案记录，尚未实施本文中的计数接口调整。第一章针对非跨超（Direct）路径；跨超（Hybrid）路径需要单独检查兼容性。性能收益尚未进行设备测量。

## 第一章：moe_ep_dispatch

### 1.1 优化目标

取消 `numRecvPerRank`、`numRecvPerExpert` 作为 dispatch 的两个输出 Tensor，以及作为 dispatch epilogue 的两个输入 Tensor的中间传递。

接收计数继续保留在 HCCL buffer 中，由 epilogue 等待、读取并在内部生成所需统计。第一步保留现有计数通信，仅调整计数的消费位置和算子接口。

预期减少中间 Tensor 的创建、GM 写回与再次读取，并解除 dispatch 中接收计数等待对后续发送阶段的约束。是否获得端到端收益，需要结合 epilogue 增加的工作和 Host 同步方式测量。

### 1.2 当前非跨超路径

当前 dispatch 主要执行：

```text
BufferInit
    → CalSendCnt：计算 per-rank / per-expert 发送计数
    → 核间同步
    → Communication：向目标 rank 发送计数和计数就绪状态
    → GetRecvCount：等待接收计数，生成两个输出 Tensor，并清理计数状态
    → SendPhase：生成发送索引、slot 编号和 metadata
    → SetStatus
    → 核间同步
    → WriteToRemoteWindow：发送 token、metadata 和 payload 状态
```

`GetRecvCount` 由最后一个 AIV 执行。其他 AIV 可以继续自己的准备工作，但后续全核同步会使最终发送阶段受到该等待的约束。

发送端按目标 rank 对 token 去重：同一个 token 命中同一张卡上的多个专家，只向该卡发送一份 x，metadata 仍保留完整 top-k expert IDs、可选 weights、来源 rank 和原始 token ID。接收端根据 metadata 将 token 展开到各本地专家的输出区域。

### 1.3 两类计数的含义与当前用途

| 计数 | 含义 | 当前消费者 |
| --- | --- | --- |
| `numRecvPerRank[srcRank]` | 从该来源 rank 接收的去重 token 数，即有效 slot 数 | epilogue 的 `CountHits` 和 `CopyFromWindowByExpert` 据此确定遍历及分核范围 |
| `numRecvPerExpert[localExpert]` | 本地该专家接收的 token 路由数量 | epilogue 计算专家输出前缀和、总输出行数；Cached 路径用总行数分配处理范围 |

例如，一个 token 同时命中目标卡上的两个专家：该来源的 per-rank 计数增加 1，两个专家的 per-expert 计数各增加 1，最终展开成两行专家输入。

计数说明有效数据范围和输出空间大小；每个专家具体需要哪些 token，由 slot metadata 中的 top-k expert IDs 决定。

### 1.4 HCCL buffer 中已有的信息

当前计数通信已将以下信息写入目标 rank 的 HCCL buffer：

- 每个来源 rank 的计数通知记录：高 32 位是去重 token 数，低 32 位是计数就绪状态。
- 每个来源 rank 发给本卡各专家的计数数组。

计数通知的打包方式为：

```cpp
uint64_t notifyVal = (static_cast<uint64_t>(sendCnt) << 32) | 1U;
```

因此，epilogue 可以直接读取并生成：

```text
每个来源 rank 的有效 slot 数
    = 对应计数通知记录中的 sendCnt

每个本地专家的总接收数
    = 所有来源 rank 对该专家的计数之和
```

当前窗口按来源 rank 预留固定大小的接收区域。发送端计算目标写入地址依赖本地路由、来源 rank 和发送 slot，不依赖本卡收到的接收计数。

窗口偏移和对齐继续使用公共 window layout 中的定义，不在 kernel 中重新硬编码。

### 1.5 已确定的第一步方案

```text
dispatch
    计算本地发送计数
    → 将计数与就绪状态写入各目标 rank 的 HCCL buffer
    → 准备并发送 token 和 metadata
    → 发布 payload 状态

dispatch epilogue
    等待计数就绪并读取 HCCL buffer
    → 在 UB 中生成 per-rank 计数、per-expert 计数及专家前缀和
    → 等待 payload 就绪
    → 按有效 slot 扫描 metadata，并展开到各专家输出区域
    → 在安全时点完成本轮计数与状态清理
```

具体调整：

1. 保留 dispatch 的发送计数计算与计数发送。`CalSendCnt` 的结果还用于本地发送索引、slot 分配和实际 payload 发送。
2. 将接收计数的等待、汇总和输出消费从 dispatch 移到 epilogue；dispatch 不再提前清理这些接收状态。
3. epilogue 直接填充已有的 `ubRecvCnt_`、`ubExpertPfx_` 等内部数据，不再通过两个中间 GM Tensor 传递计数。
4. 同步调整 dispatch 输出、epilogue 输入及对应的 Python、Torch C++、ACLNN、tiling 和调用方。任何其他消费者都需追踪后处理。
5. 接收端仍需知道有效 slot 范围，不能直接扫描全部预留窗口并把旧数据视为本轮有效数据。

### 1.6 实施前必须确认的依赖

**Host 输出容量与 `DoCpuSync`**

当前 `DoCpuSync` 路径将接收专家计数总和写入 Host pinned 地址。实施前必须追踪 Host 是否用这个实际行数，为 epilogue 分配输出或决定其他操作。

- 如果输出按最大容量预分配，可以考虑由 epilogue 内部统计实际行数。
- 如果 Host 依赖提前获得精确行数，则需保留或重新安排提供总数的途径。
- 在上述依赖确认前，不能直接删除所有接收计数处理和 Host 通知。

**计数就绪、payload 就绪与下一轮复用**

计数与 payload 可能经不同 channel 发送。第一步方案保留对计数就绪状态的明确等待，不假设某个 payload channel 完成就代表所有计数已经可读。

计数和状态的清理责任移到 epilogue，需要确保：本轮所有消费者已完成读取，清理写入已经完成，下一轮才复用对应区域。多核同步与实际内存写入完成的作用需要分别核对。

**Cached 与 Hybrid**

Cached epilogue 当前仍依赖 per-expert 总数得到 `expertSum_`，用于分配缓存 metadata 的处理范围，不能漏改。共享接口若同时服务 Hybrid 路径，也必须检查该路径的消费者；本章尚未确定其具体改造方案。

### 1.7 后续可选优化，尚未确定实施

接收端可以从完整 metadata 中的 top-k expert IDs 重新统计 per-expert 数量，进一步减少 per-expert 计数通信。

这需要调整 epilogue 的扫描、汇总和前缀和顺序，并重新处理 Host 提前获知实际输出行数的依赖，不属于第一步仅去掉中间计数 Tensor 的范围。

per-rank 有效 slot 数也可考虑与最终完成通知合并传递；同样需要单独设计通知格式、跨 channel 顺序和完成语义。

### 1.8 后续验证要点

- 对比优化前后的有效输出：recv x、weights、scales、metadata、专家分组和 combine 使用的映射关系。
- 覆盖同 token 命中同卡多个专家、跨多个目标 rank、无效 expert ID、零发送量及不均衡路由。
- 覆盖 Cached / 非 Cached、`DoCpuSync`、不同 rank 数量及连续多轮窗口复用。
- 核查所有调用方的输入输出数量、参数顺序、类型和容量依赖，避免只改 kernel。
- 分别测量 dispatch、epilogue 及二者串联耗时，同时观察 Host 等待；不能仅凭 dispatch 变快认定整体收益。

### 1.9 代码入口

- [dispatch kernel](moe_ep_dispatch/op_kernel/moe_ep_dispatch.h)：`CalSendCnt`、`Communication`、`GetRecvCount`、`SendPhase`、`WriteToRemoteWindow`。
- [dispatch epilogue kernel](moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue.h)：`Init`、`ComputePrefixSums`、`CountHits`、`CopyFromWindowByExpert`、`CopyFromWindowByCachedMeta`。
- [公共窗口布局](common/op_host/op_tiling/moe_ep_window_layout.h)及[布局计算](common/op_host/op_tiling/moe_ep_window_layout.cpp)。
- [Torch C++ 调用](common/torch_extension/csrc/elastic_buffer.cpp)及[Python 接口](common/torch_extension/elastic_buffer.py)。

### 1.10 当前共享 Jetty 穿刺版本

本节记录通信接口迁移，与前面的计数优化方案分开实施。当前已完成源码接入，尚未进行 CANN 编译和 NPU 验证；两个计数 Tensor、去重和专家还原逻辑保持原样。

- Direct dispatch、dispatch epilogue 使用共享 context；Hybrid 和 combine 两个算子继续使用普通 context。Python 参数和输出不变。直接调用 ACLNN 的 Direct 调用方必须传入 `MoeSharedJettyContext`，旧 context 会被 tiling 的长度检查拒绝。
- Host 固定建立 56 个共享 Jetty。保留按目标 rank 分组、组内切分 token 的调度，实际发送核数不超过 56，每个发送核按 `aivId` 独占一个 Jetty。原 `channelIndex` 继续表示组内分片和通知槽位。
- 计数、payload 和空分片通知均使用 `MakeBatchHandle → GetHandleRef → WriteNbi / WriteWithNotifyNbi → BatchCommit`。对端索引按升序跳过本卡；本卡仍走本地拷贝。本版不合并跨对端 batch，也不改成所有核遍历所有对端。
- epilogue 保留原等待条件与通知清理，只适配共享 context 的窗口访问。发送核不足以覆盖所有 rank 时，按实际发送核数重新划分 rank，避免漏发。
- 共享窗口使用 HCCL 内置 buffer，容量由通信域的 `HCCL_BUFFSIZE` 配置决定，接入时检查容量。首次缓存未命中时，在建链前清零整个本地窗口；缓存命中不清零。通信域须持续存活，构建失败后沿用现有约定废弃通信域。
- 迁移期间仍会建立普通 context，供拓扑识别、Hybrid 和 combine 使用，因此不是最终资源占用方案。共享内置窗口在 dispatch 到 epilogue 完成之间不可被其他通信操作覆盖。
- 同步适配 Device 诊断、Host 异常回调和离线 dump 解析，避免将较小的共享 context 当作普通结构读取。

本地 asc-comm 接口核对结果：batch 形式的 `WriteNbi`、`WriteWithNotifyNbi`、`ReadNbi` 支持由 `GetHandleRef` 选择共享 Jetty 内的对端，均包含连续地址和 scatter/gather 重载；`BatchCommit`、`Drain` 有对应 batch 重载。单次 scatter/gather 请求不能跨对端。当前公开的 `WriteValueNbi`、`WriteReduceNbi`、`AtomicFAA`、`AtomicCAS`、`Commit`、`Lock`、`Unlock` 只有普通 `ChannelHandle` 重载，不能直接传入 `MultiChannelHandle`。

`BatchCommit` 表示提交，不等同于远端接收完成。当前仍由接收侧通知协议判断 payload 就绪；没有因本次迁移新增 CQE 或 Drain 等待。

设备验收至少覆盖：8 卡与 16 卡、空分片、零发送、非均衡路由、Cached / 非 Cached、`DoCpuSync`、连续多轮，以及 dispatch → epilogue → combine → combine epilogue。静态核对不能替代上述设备验证。

## 第二章：moe_ep_dispatch_epilogue

待后续讨论并补充该算子的独立优化方案。与计数读取、汇总、前缀和及接口调整有关的配套要求，目前统一记录在第一章。

## 第三章：moe_ep_combine

待后续讨论并记录优化方案。

## 第四章：moe_ep_combine_epilogue

待后续讨论并记录优化方案。
