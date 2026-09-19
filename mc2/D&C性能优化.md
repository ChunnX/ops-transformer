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

- 四个算子统一使用共享 context，旧普通 MoE context builder、topology 分流和单对端 kernel 路径已弃用。Python 参数和输出不变。world size 由运行时参数和 communicator context 共同校验，不再限定 8P；当前显式支持范围为 2～1024，设备可行范围仍受完整 shape/UB/window 约束。直接 ACLNN 调用须传入 `MoeSharedJettyContext`；topology 属性仅为兼容主线 ACLNN ABI 而保留。
- Host 每个 rank 固定建立 32 个共享 Jetty，每个 Jetty 均包含按 rank 升序跳过自身的全部远端 peer。sender AIV 0～31 与 Jetty 0～31 一一绑定，每个 owner 遍历所有 peer；核/Jetty 映射不随卡数改变。
- Dispatch 固定由 0～31 号核绑定 32 个 Jetty，并将 BS 独立均分为 32 个发送区间；其余 `aivNum - 32` 个 prepare 核另行均分 BS。sender 只等待与其区间相交的 prepare，利用原 count 前缀和 ready 分片内的 source-token lower_bound 定位连续远端 slot；卡数不参与分核。Dispatch 要求 `33 <= aivNum <= 100`，上限来自固定的每核诊断区容量。Combine 对每个 peer 的 metadata 区间做 32 路无重叠切分。
- count 使用 `MakeBatchHandle → GetHandleRef → WriteWithNotifyNbi → BatchCommit`，payload 和空分片通知使用 `WriteNbi`；对端索引按升序跳过本卡，本卡仍走本地拷贝。完成标志按 `[source rank][Jetty owner]` 布局，epilogue 等待 `worldSize × 32` 个标志，并以固定 chunk 分块搬入 UB。
- shared-Jetty builder 按算子参数计算的容量自行申请、清零并通过 `HcclCommMemReg` 注册窗口；channel descriptor 绑定该内存句柄，并通过首个共享 channel 按 tag 获取各 peer 的远端窗口地址。该路径不使用 HCCL 内置 buffer，也不依赖 `HCCL_BUFFSIZE`。同一通信域首次建立的容量固定，后续请求不得超过首建容量；各 rank 的窗口参数须一致。通信域须持续存活，注册后的资源构建失败时沿用现有约定废弃通信域。
- 四个算子复用同一份 HCCL 共享 context 和独立注册窗口，不再提前建立普通 MoE context。注册窗口在算子链完成之前不可被其他调用覆盖。Engram 的独立通信实现不在本次删除范围。
- 同步适配 Device 诊断、Host 异常回调和离线 dump 解析，避免将较小的共享 context 当作普通结构读取。

本地 asc-comm 接口核对结果：batch 形式的 `WriteNbi`、`WriteWithNotifyNbi`、`ReadNbi` 支持由 `GetHandleRef` 选择共享 Jetty 内的对端，均包含连续地址和 scatter/gather 重载；`BatchCommit`、`Drain` 有对应 batch 重载。单次 scatter/gather 请求不能跨对端。当前公开的 `WriteValueNbi`、`WriteReduceNbi`、`AtomicFAA`、`AtomicCAS`、`Commit`、`Lock`、`Unlock` 只有普通 `ChannelHandle` 重载，不能直接传入 `MultiChannelHandle`。

`BatchCommit` 表示提交，不等同于远端接收完成，也不对累计未完成的 SQ 请求自动节流。Dispatch payload 跨 peer 复用同一个 owner batch，按实际 WQEBB 计量：首批32、后续512、最后剩余请求强制提交；不新增 CQE 或 Drain。Combine 同样只提交 payload 和最终有序标记，接收侧仍由各自 epilogue 等待全部 payload 就绪。本版本要求实际资源为 32 Jetty、SQ 4096 WQEBB、CQ 64，并在 device 构造 batch 后严格核对；当前 HCCL 公共建链接口没有显式深度参数，因此该深度映射仍是平台资源策略的外部约束，必须在 NPU 建链时验证。本轮按要求不增加 SQ 累计水位保护。

设备验收至少覆盖 2P/4P/8P/16P、不同 `aivNum - 32` prepare 核数、BS 小于 prepare 核数、BS 不能被 prepare 核数整除（典型 64 AIV 时即 32）、空分片、零发送、非均衡路由、Cached / 非 Cached、`DoCpuSync`、连续多轮，以及 dispatch → epilogue → combine → combine epilogue。静态核对不能替代设备验证。

## 第二章：moe_ep_dispatch_epilogue

待后续讨论并补充该算子的独立优化方案。与计数读取、汇总、前缀和及接口调整有关的配套要求，目前统一记录在第一章。

## 第三章：moe_ep_combine

### 3.1 共享 Jetty 最小穿刺

combine 仅使用 dispatch 的 `MoeSharedJettyContext`，context 仍由 HCCL 管理，Torch 仅持有其 Tensor。不再保留普通 context/channel 分支或 `SharedJetty` 双模板；公开算子参数不变。

Combine 不再按目标 rank 分核：AIV 0～31 分别独占 Jetty 0～31，每个 owner 遍历全部 peer，并把该 peer 的 metadata 区间按 32 份切分。token/weight 写入位置及主线输出排布不变；通过 `MakeBatchHandle → GetHandleRef → WriteNbi → BatchCommit` 切换对端并批量提交。对端索引按 rank 排序并跳过自身。共享最终标记保持 `fence=1, cqe=0`，在各 peer payload 之后提交但不在 combine 中等待；combine epilogue 观察全部 `worldSize × 32` 标记后再读取数据。

本次不修改 token 排布、接收 metadata 或算子输入输出，也不实施第一章的计数 Tensor 优化。直接调用 ACLNN 的调用方在 Direct 模式下须传共享 context；tiling 会拒绝旧普通 context。

### 3.2 验证边界

本轮 CPU 测试直接编译 Dispatch 使用的区间划分函数，覆盖1～68个 prepare 核、空/小BS及4095/4096/4097，验证独立32路发送区间与ready交集；随机路由模型将计数前缀加边界二分与全表 lower_bound 对照，验证slot完整且无重叠。另已静态检查 Combine 每 peer 的32路metadata覆盖及C++/Python window公式。CPU测试和模型不是AscendC/Hcomm设备执行，尚未进行CANN编译或NPU精度验证。

设备验收应覆盖 dispatch → dispatch epilogue → combine → combine epilogue 连续执行、连续多轮复用、空分片、零 token、非均衡路由、带/不带 topk weight及低 AIV 多对端。修改 kernel 后须重新编译安装自定义算子包；修改 Torch 扩展后须重新构建并安装 wheel。

## 第四章：moe_ep_combine_epilogue

### 4.1 共享 context 配套接入

与 combine 使用同一份共享窗口，只解析共享 context。异常 metadata 和核诊断显式使用共享类型，沿用已有共享 context 离线解析。

完成标记固定按 `[source rank][Jetty owner]` 布局为 `epWorldSize × 32`；Combine Epilogue 分块等待全部标记后再清零。接收归约仍使用运行时 `aivNum`，保留原来的归约分核、topk weight 处理和输出排布。

本章与第三章一起验证；静态检查不代表设备运行或精度通过。
