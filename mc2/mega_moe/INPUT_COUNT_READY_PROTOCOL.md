# MTE 输入 count 就绪协议

MTE 路径删除共享专家前的入口跨卡握手，以量化硬件通知和 count epoch 传递输入依赖。
保留共享入口的 `PipeBarrier<PIPE_ALL>()`、`SyncAll<false>()` 及出口跨卡同步。
Dispatch 仍等待完整 count 表，远端 24-bit count + 8-bit epoch 布局和 VF cumsum 不变。

## 修改前后对比

原实现中，每个 AIV 完成自身量化后生成并发送 index/count，接着清零 workspace、
整理共享输入。所有核经过本卡 `SyncAll<false>()` 后，AIV 还要完成入口跨卡握手及
其末尾的 `SyncAll<true>()`，才能进入共享激活或 Dispatch。AIC 不执行跨卡握手，
但其共享 GMM1 与 AIV 激活存在 UB/ping-pong 交接，仍会受到 AIV 跨卡等待的影响。

第一版移除了入口跨卡握手，通过量化后的 `MTE3_S` 和逐 AIV GM 计数轮询保护 count。
本次进一步把通知改为硬件事件，并去掉 count 发送内部的标量完成等待。

| 项目 | 原实现 | 本次实现 |
|---|---|---|
| 量化依赖 | 入口本卡/跨卡同步收口 | 量化后发 mode 0 MTE3 事件，index 后、count 前在 MTE3 等待 |
| 量化发布端 | 无独立通知 | 无 `MTE3_S`；硬件在量化写回完成后上报 |
| count 发布端 | 非对齐段重排并等待，函数末尾等待 MTE3 | 每项 count 使用独立 32B UB 槽，Normal 二维搬运异步返回 |
| 入口跨卡握手 | AIV 参与，末尾另有 AIV 全核同步 | 删除 |
| 共享计算入口 | 配对 AIV 可能阻塞在跨卡握手 | 原有本卡 `SyncAll<false>()` 后推进 |
| 接收 count | 逐槽验证 epoch，整表就绪后 VF cumsum | 不变；epoch 同时证明源数据可读 |
| 本卡同步、出口跨卡同步 | 存在 | 保留 |

与第一版相比，不再用 GM 槽表示量化 ready，也不逐核轮询。原逐 AIV 计数槽只用于
跨轮次 epoch 记账，入口和出口各推进一次，兼容原 count 和出口协议。

## 发送与消费顺序

```text
每个 AIV（包括空 token / 无专家任务）：
量化自己的 token，下发 token/scale 写回
  → 标量推进本核输入 epoch，保留返回值
  → CrossCoreSetFlag<0, PIPE_MTE3>(8)
  → 生成、发送自己负责的 compact index
  → 收口 route ring 的完成事件
  → CrossCoreWaitFlag<0, PIPE_MTE3>(8)
  → PIPE_MTE3 barrier
  → 给 count 添加 epoch，S_MTE3 保护标量写入的 UB 数据
  → 异步发布 count
  → workspace 清零、共享输入整理
  → 原有 PIPE_ALL + SyncAll<false>（完成搬运，允许后续 UB 复用）
  → 共享 GMM1/激活及 MoE 阶段

接收 AIV1：
本卡 SyncAll<false>
  → 读取本轮 epoch
  → 逐槽确认 count epoch，计算 prefix
  → Dispatch 拉取源卡 token/scale
```

量化按 token 分工，路由按专家分工。某专家可以引用其他 AIV 量化的 token，故 count
发布必须等待全体量化生产者。硬件 mode 0 汇合本卡所有 AIV 的 MTE3 完成通知。
无专家任务的核在返回路由函数前也必须消费一次事件；AIC 不发、不等该事件。

在 Ascend 950 上，Wait 的 `PIPE_MTE3` 参数生效：等待进入 MTE3 队列，不要求标量
同步等完再下发 count。Set 位于 index 之前，因此慢核仍可生成 index，快核只在发布
count 的 MTE3 前等待。后续同一 MTE3 队列中的清零等搬运仍受该等待影响；这不保证
所有后续流水都完全独立，现有 route ring、共享输入搬运依赖也保留。

发送端已有 route ring 的 `MTE3_V` 完成等待，并在 count 前保留 `PIPE_MTE3` barrier。
协议依赖上述操作在设备上保证 index/量化写回先于 count 可见；CPU 模型不能证明
NPU 互连可见性，仍需多卡设备验证。

## count 的 UB 布局和生命周期

```text
发送端 UB： [count0:4B][padding:28B][count1:4B][padding:28B] ...
远端 GM：  [localExpert][sourceRank]，每项仍为 4B tagged count
```

`MTE_COUNT_UB_STRIDE=8` 同时用于 host 预算、kernel 分配和索引。Normal 模式下每个
block 的 UB 步长为 32B；每个 block 仅发送 4B，GM 的额外步长为 `(worldSize-1)*4B`。
任意专家分段的 UB 起点都对齐，不再复用 `topkIndexTensor` 重排 count，段间和函数末尾
均无需 `MTE3_S`。每个目的 rank 仍合并为一次二维搬运，不新增跨卡状态消息。

发送端 count 固定占用从 `align32(maxOwnedExperts*4)` 改为 `maxOwnedExperts*32` 字节。
Host 自适应 route buffer 预算同步扣除这部分占用，因此部分 shape 的 route batch 或
ring 深度可能变化，需要一起评估性能。

Count 源 UB 是输入阶段尾部的独立区域，不与 workspace 清零源或共享输入双缓冲重叠。
所有 tagged count 在搬运下发前写好，返回后直到入口 `PIPE_ALL` 都不再改写。该同步
完成之后后续阶段才重用 UB。未来若移除共享入口同步，必须重新建立这个完成边界。

## 硬件事件与 epoch 生命周期

量化硬件事件使用 mode 0、flag ID 8。避开本地 SDK `SyncAll` 使用的 11--14；
GMM 的 mode 4 事件使用独立计数器空间。所有 AIV 每轮恰好 Set/Wait 各一次，
入口 `PIPE_ALL` 保证等待已执行，不能让无专家任务的核积累未消费事件。
不依赖不同 flag ID 的 Set 执行先后关系。

原逐 AIV 计数槽保持：

```text
rankSyncBase + 48 KiB + aivIndex * 64B
```

仅所属 AIV 标量读写，不清零、不使用 DMA 写入。若上次出口计数为 `s`：

| 时刻 | 槽值 | 含义 |
|---|---|---|
| 本轮入口记账之前 | s | 上轮出口序列 |
| 量化写回下发后 | s + 1 | 本轮 epoch 已分配；不代表量化已经完成 |
| 硬件量化等待完成后 | s + 1 | 此时才允许 MTE3 发布 count |
| 出口同步 | s + 2 | 本轮出口序列 |

发送函数直接使用发布函数返回的完整序列；count epoch 为 `(sequence & 0x7f) | 0x80`。
递增使用无符号加法，保持符号边界及 uint32 回绕行为。

保留的输入同步确保所有本核 epoch 记账完成。接收端在其后读取 slot 0；MoE 阶段末尾
AIV 同步保证所有接收者读完后，才可能在出口更新 slot 0。现有整表等待和出口握手继续
约束各 rank 的跨轮次进度。首次窗口初始化、rank 调用顺序和活动核集合约定沿用原协议。

## 验证

```bash
python3 mc2/mega_moe/tests/ut/op_kernel/arch35/run_input_sync_test.py
```

测试直接编译生产通知、等待和 count 发布函数，以确定性 CPU MTE3 队列代替设备流水。
覆盖 3120 轮调用、延迟量化、空 token/无专家任务、index 先推进、异步 count 返回、
跨目的 rank 的非对齐专家分段、零/最大 count、epoch/符号/uint32 回绕、单写者及
UB/GM 哨兵。DMA 在队列执行时才读 UB，检查源未被提前覆盖；AIC 测试禁止状态与事件访问。
这验证协议和地址计算，不代表真实 NPU 性能或硬件内存顺序。

本地验证（2026-09-08）：上述 CPU 协议测试通过；host tiling 翻译单元编译通过。
CANN 9.2 beta2、Ascend950PR_9599 下 A8W8/W4 两组共 36 个 MTE tiling key 编译通过，
包含 combine 量化和 topk weights prefetch 变体。内核编译使用验证专用 Hcomm shim
屏蔽当前 SDK 缺失的无关 URMA API；URMA 和真实多卡运行不在本次验证范围内。
提交检查随后仅调整格式和注释。

设备验收仍需覆盖共享专家开关、A8W8/A8W4/A4W4、空专家、小 bs、连续调用及多卡精度。
性能对比需同时查看算子耗时、量化后的标量停顿和 route buffer 配置变化。

## 接口依据（CANN 9.2 beta2）

- [CrossCoreSetFlag](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreSetFlag_ISASI.md)
- [CrossCoreWaitFlag](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/CrossCoreWaitFlag_ISASI.md)
- [SyncAll](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta2/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md)
