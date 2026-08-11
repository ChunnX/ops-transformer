# hybrid-perf 分支代码检视报告

## 1. 检视范围

- 基线分支：`master`
- 基线提交及共同祖先：`db91bb43d0bfc8d7b7f444512ff0f943da19764c`
- 检视分支：`hybrid-perf`
- 分支提交：`e4c87e213f48e805574b49226205d2ff7cb065db`
- 对比范围：`master...hybrid-perf`
- 主要涉及：dispatch hybrid kernel、dispatch/epilogue/combine tiling、aclnn 原型及 Torch 调试入口

本次以静态代码检视为主。当前环境没有多卡、多机 NPU 验证条件，因此没有完成设备侧编译、精度、死锁和性能实测。

## 2. 结论摘要

| 编号 | 级别 | 归属 | 结论 |
| --- | --- | --- | --- |
| A1 | 阻塞 | 本分支新增 | fanout 来源状态初始化后缺少 `V -> S` 同步，可能读取旧状态并卡死 |
| A2 | 阻塞 | `master` 已存在 | FP8 场景下三个算子计算的 hybrid win 区偏移不一致 |
| B1 | 高 | 本分支新增 | 调试环境变量可绕过接口校验并制造不受支持的 hybrid 拓扑 |
| B2 | 中 | 本分支新增 | 新 fanout 调度轮询没有超时或错误出口，协议异常时永久占核 |
| P1 | 高，性能 | 本分支新增 | 每个 scaleup owner 核都重复读取并扫描同一份路由信息 |
| P2 | 高，性能 | 基线保留 | fanout/scaleup 每发送一个 payload 都执行一次 `Drain`，通信被串行化 |
| T1 | 中 | 本分支缺口 | 调度协议变化较大，但分支没有增加自动化测试用例 |

当前版本不建议直接合入。至少需要先修复 A1，并统一 A2 所涉及的跨算子 win 区布局契约。`e4c87e213` 是独立的调试提交，正式合入前建议直接移除或限制其生效范围。

## 3. 详细问题

### A1. fanout 来源状态初始化缺少 `V -> S` 同步

位置：`mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_hybrid.h:1517`

`SendFanoutPayloads` 先通过 `Duplicate<int64_t>` 将 `sourceStateTensor_` 初始化为 `SOURCE_STATE_UNREAD`，随后马上由 `GetSourceState` 使用标量 `GetValue` 读取同一块 UB。这里存在明确的 Vector 写、Scalar 读依赖，但两者之间没有 `SyncFunc<AscendC::HardEvent::V_S>()`。

可能产生的结果：

- Scalar 读到旧值或未初始化值，错误跳过尚未处理的来源。
- 若旧值表现为“剩余 slot 数为 0”，来源会被持续跳过，但 `completedSourceCount` 不增加，外层循环无法退出。
- 非法状态还可能形成错误的 slot 下标，导致读错数据或越界。
- 后续状态清理假设已发布的 slot 数非零；若调度错误完成，还存在计数下溢风险。

建议在 `Duplicate` 后、第一次 `GetSourceState` 前加入明确的 `V -> S` 同步。该问题由 `d55991489` 引入。

### A2. FP8 hybrid win 区布局在三个算子之间不一致

该问题已经存在于 `master`，不是 `hybrid-perf` 新增，但会直接影响本分支功能完整性。

相关位置：

- dispatch：`mc2/moe_ep_dispatch/op_host/op_tiling/moe_ep_dispatch_tiling.cpp:597`、`:746`、`:833`
- epilogue：`mc2/moe_ep_dispatch_epilogue/op_host/op_tiling/moe_ep_dispatch_epilogue_tiling.cpp:481`
- combine：`mc2/moe_ep_combine/op_host/op_tiling/moe_ep_combine_tiling.cpp:362`

dispatch 使用实际输入类型和实际 scales 大小计算 `perSlotBytes`，并据此计算 scaleout slot stride 以及其后的 scaleup、status、combine、stash 区域偏移。epilogue 和 combine 无法获得同样的 FP8 scales 信息，仍按固定的 `hidden * 2` 公式重建 dispatch slot 大小。

BF16/FP16 下两套公式一致，FP8 下则不同。例如：

- `hidden=5701`
- `topk=32`
- FP8 输入
- 45 个 `float32` scales

此时 dispatch 的实际 `perSlotBytes` 为 6656 B、scaleout slot stride 为 7168 B；epilogue/combine 推导出的 dispatch slot 为 11776 B、scaleout slot stride 为 12288 B，每个 slot 相差 5120 B。即使 Python 侧申请的总 win 区足够大，也无法修复各算子使用不同偏移的问题。

可能产生的结果包括 `recv_x`、scales、metadata 错位，以及后续算子等待错误状态而卡死。

建议建立唯一的跨算子布局契约：

1. dispatch 也使用 epilogue/combine 能重建的固定保留 stride；实际 payload 搬运长度仍可使用真实长度。
2. 或将精确的公共 slot stride 显式传给三个算子，并保证三者只使用该值定位共享区域。

### B1. 调试环境变量可造成接口校验与实际 tiling 不一致

位置包括：

- `mc2/moe_ep_dispatch/op_host/op_tiling/moe_ep_dispatch_tiling.cpp:91`
- dispatch epilogue/combine 对应 tiling 文件
- `torch_extension/cann_ops_transformer/ops/csrc/elastic_buffer.cpp`
- `torch_extension/cann_ops_transformer/ops/elastic_buffer.py`

提交 `e4c87e213` 增加了 `MOE_EP_DEBUG_FORCE_NETWORK_MODE` 和 `MOE_EP_DEBUG_RANK_NUM_PER_SERVER`。当前实现允许 tiling 在 aclnn 参数检查完成后单独覆盖拓扑，因此接口检查看到的模式可能与最终选择的 kernel 不同。

具体风险：

- `serverNum == 1` 时仍可强制进入 hybrid；此时 `remoteServerCount_ == 0`，部分 buffer 可能以 0 字节初始化，该路径没有完整验证。
- 非 Torch 调用方可能按 direct 参数通过 aclnn 检查，但 tiling 被环境变量切为 hybrid，最终缺少 hybrid 所需的缓存路由输入。
- Python sizing 只按环境变量当时选择的拓扑申请 win 区；若 sizing 后改变环境变量，实际 kernel 可能使用更大的布局。

建议正式版本移除该调试提交。若必须保留，应在接口校验前统一解析覆盖值，并增加 `serverNum > 1`、输入完整性和 sizing 一致性检查。

### B2. 新 fanout 来源调度采用无界轮询

位置：`mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_hybrid.h:1523`

`SendFanoutPayloads` 会反复轮询各远端来源的 slot count 和 ready 状态。当前只有短暂的 cycle 等待，没有超时、错误标志或退出条件。任何发送端缺失通知、状态地址错误或同步问题都会使 AIV 永久停留在该循环。

通信协议正常时无界等待可以工作，但不利于定位多机偶现问题。建议调试版本保留可配置超时，并输出来源 server、slot、期望状态和实际状态；正式版本是否保留超时可按通信错误处理策略决定。

### P1. 每个 scaleup owner 核重复扫描全部来源路由

位置：`mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_hybrid.h:1375`

`PrepareFanoutDestinations` 对每个收到的 scaleout slot 执行以下操作：

1. 初始化完整的 `rankNumPerServer_` rank-slot 映射。
2. 读取该 slot 的 topk 和目标 slot 信息。
3. 扫描全部 `axisK_` 路由。
4. 最后才按当前 owner 核负责的 rank 范围过滤。

当存在多个 scaleup owner 核时，同一个来源 slot 会被多个核重复读取和扫描，计算量近似随 owner 核数量增长。该设计减少了共享 fanout entry 表并允许按来源 ready 状态推进，但在 `rankNumPerServer` 或 `topk` 较大时可能成为主要瓶颈。

建议至少只初始化 owner 核负责的 rank 范围。进一步优化可由少量扫描核构造紧凑的 ready entry，再由 owner 核发送，但需要评估新增状态和同步是否值得。

### P2. fanout/scaleup 逐 payload `Drain`

位置：

- `SendFanoutPayloadToRank`：`mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_hybrid.h:1418`
- `SendEntries`：`mc2/moe_ep_dispatch/op_kernel/moe_ep_dispatch_hybrid.h:1589`

每个 payload 都执行 `WriteWithNotifyNbi` 后立即 `Drain`，NBI 请求无法形成有效流水，发送延迟会随 payload 数量线性累积。scaleout 到 proxy 的路径则是同一目标批量提交后只 `Drain` 一次。

该行为属于稳定性问题后的保守实现，并非本分支新引入。后续优化不能直接删除 `Drain`，应先确认 HComm CQ、channel owner、notify 可见性和 epilogue 消费协议，再按目标 rank 或安全批次合并提交与完成等待。

### T1. 缺少与新协议对应的测试增量

`master...hybrid-perf` 没有测试文件变化，但分支修改了 fanout 的调度顺序、分核方式、状态轮询和布局数据。静态检查无法覆盖跨核可见性、跨卡快慢差以及性能循环中的状态复用。

合入前至少应覆盖：

- direct 模式回归。
- BF16、FP16、FP8 输入。
- cache 与非 cache。
- 无远端路由、仅远端路由、远端和本地混合路由。
- 不同 `rankNumPerServer`、server 数和 AIV 数。
- dispatch、epilogue、combine 单跑和 continue 连跑。
- 性能模式多轮运行，用于捕获偶现同步和状态复用问题。

## 4. 已检查但未发现明确问题的部分

- workspace 中 counter、send-entry 等区域的 offset 和容量计算内部一致。
- 每个 token 对同一目标 rank/server 最多产生一条对应 entry，按 token 分片预留的 entry 容量满足该上界。
- scaleout 与 scaleup owner 的目标范围互斥，同一阶段没有发现多个 owner 核并发使用同一目标 channel 的情况。
- 跨核发布的 GM 数据在消费者读取前存在 `PipeBarrier` 和 `SyncAll`，未发现新的明确 GM 发布缺口。
- fanout 状态协议中 slot 0 表示总数加一，后续 slot 表示 payload ready；零 payload 的通知值和清理范围相互匹配。
- HComm 实现会在 SQ 接近满时轮询 CQ，因此 scaleout 路径按 proxy 批量提交后 `Drain` 一次，没有发现明确的队列溢出问题。
- 按 tiling 极限估算，count 阶段 UB 峰值约 240.6 KiB，低于 248 KiB，但余量约 7.4 KiB，后续新增 UB 分配时必须重新核算。
- `GetExpertFreq` 的输入、输出使用方式没有发现新的覆盖问题。
- direct kernel 的差异主要是字段和命名适配，未发现本分支引入的 direct 功能改动。
- aclnn 部分主要为常量名称调整，未发现新增行为差异。

## 5. 本地静态验证

- `git diff --check master...hybrid-perf`：通过。
- `python3 -m py_compile torch_extension/cann_ops_transformer/ops/elastic_buffer.py`：通过。
- 未执行 AscendC 设备侧编译。
- 未执行 NPU 单机/多机功能、稳定性及性能测试。

## 6. 建议处理顺序

1. 修复 A1 的 `V -> S` 同步，并重新检查来源状态生命周期。
2. 统一 dispatch、epilogue、combine 的 FP8 hybrid win 区布局契约。
3. 正式合入前移除或收紧调试环境变量提交。
4. 补齐多轮 continue 和 FP8 大 shape 测试，确认无卡死与错位。
5. 用设备侧打点量化重复路由扫描和逐 payload `Drain`，再决定下一轮性能优化。

