---
name: kernel-optimization-discipline
description: Ascend C kernel 性能优化纪律。适用于冗余 DataCopy/LoadData/Fixpipe/MMAD-copy、resident buffer 物化、cache clean/DCCI、同步边界、tiny MatMul/MMAD tiling、性能与精度 gate 判断。
---

# Kernel 优化纪律

在修改 Ascend C kernel 数据流前使用本 skill。目标是删除真实冗余，同时不降低目标算法形态，也不把正确性边界藏在 copy、cache clean、sync 或副作用里。

主 skill 只放通用规则。某个算子的阶段号、临时文件名、baseline、SHA、失败 shape 等细节放到 `references/`。

## 核心流程

1. 先命名消费者，再保留搬运。
   - 每个 GM/L1/L0/L0C 物化都要写清楚下一个消费者和所需 layout。
   - 没有明确消费者的 copy，应删除、后移或作为诊断路径保留。
   - 如果写回只承担副作用，先命名副作用，再决定是否替换。

2. 保持目标形态。
   - packed MMAD、resident buffer、merged block execution 可以就是目标。
   - 不要为了快速 PASS 把 packed/resident 目标退回 per-block fallback，除非用户明确要求临时诊断。
   - 冗余通常是目标周围重复的 per-block 搬运、copy-out、layout-copy、identity/zero 生成，而不是 packed 目标本身。

3. 优先合并同形同 stride 的搬运。
   - 若多个 block 源 stride 和目的 stride 恒定，先评估一个 DataCopy/LoadData/Fixpipe 候选。
   - 合并前必须确认 source offset、dst offset、transpose 模式、fractal layout、stride 单位和硬件支持。
   - 不要因为 L0A/L0B 形状相同就合并；load 方向和 transpose 语义可能不同。

4. 不把 MMAD 默认当 layout-copy 引擎。
   - 如果操作本质是 `X @ I` 或 `I @ X`，先找直接 LoadData/Fixpipe/DataCopy 方案。
   - identity/zero tile 要有明确生命周期，在已验证的 buffer epoch 内复用，不要在每个 helper 里重建。

5. cache clean/flush 是最后的可见性工具。
   - 先检查 producer/consumer 顺序：`SetFlag/WaitFlag`、`SyncMte3ToMte2`、`SyncFixToMte2`、`SyncFixToS`、跨核 flag、Fixpipe/MTE event wait。
   - 不要因为“怀疑 AIC/AIV 看不到”就加 `DataCacheCleanAndInvalid` 或 flush。
   - reference/origin 路径只靠 flag/wait 能正确工作时，不要默认加 DCCI。
   - 删除 DCCI 和删除 sync 是两个优化，必须分开验证。

6. 开 double buffer 前先定义被重叠的两段流水。
   - double buffer 是两个物理 slot 加一组 producer/consumer 事件，不只是把变量改成 `Buf1/Buf2`。
   - 先确认下一 tile 可提前生产、当前 tile 消费不依赖下一 tile、两个 slot 不 alias，再写 ping-pong。
   - L0A/L0B、L1/UB、Fixpipe copy-out 的事件方向不同，不要照抄 flag 类型。

7. 一次只验证一个变化。
   - 每个数据流变化后 build/install。
   - 先跑当前任务要求的精度 ladder。
   - 精度 PASS 后再跑性能。
   - 精度 PASS 但性能回退，不应作为继续优化的性能 baseline，除非用户接受该取舍。

## 性能判断

- 长序列行为是长上下文 kernel 的主要验收标准。
- 短序列收益可能来自启动噪声或小 shape 偶然性，不能单独支撑 baseline。
- MTE 下降但 scalar 上升时，优先查 per-row cache clean、小 helper 循环、identity/zero writer、重复状态设置。
- 记录性能时说明收益来自哪里：更少 MMAD、更少 MTE/Fixpipe、更少 scalar setup、更好 tiling，还是只有噪声级变化。

## 文档规则

- 主 skill 只写可复用规则：producer/consumer、内存层级、同步边界、验证顺序、profile 解读。
- 具体失败、hash、阶段标签、算子路径写入 reference。
- reference 应回答：改了什么、什么 gate 通过或失败、学到什么边界、哪些 exact retry 应避免。

## 参考资料

按问题只读相关 reference：

- `references/redundant_movement.md`：冗余搬运、物化、`X @ I`、identity/zero 生命周期。
- `references/double_buffer.md`：通用 double buffer / ping-pong 实施规则、事件生命周期和排错清单。
- `references/cache_clean_and_sync.md`：DCCI/cache clean/flush 与同步边界。
- `references/tiny_mmad_and_tiling.md`：tiny MatMul/MMAD tiling 与固定开销陷阱。
- `references/chunk_gdr_packed64_case_notes.md`：packed64 Neumann 案例，仅作证据，不是通用策略。
- `references/chunk_gdr_tail_padding_a131_case.md`：A131 tail padding 优化、未对齐 Duplicate 边界、artifact 防污染和 gate 案例。
