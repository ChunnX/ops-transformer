# Chunk GDR Packed64 案例笔记

这是通用优化纪律的算子案例证据。不要把这里的阶段号、buffer 名和路径直接搬进通用规则；只有遇到相同 API/硬件边界时才复用结论。

## 可复用教训

- target path 可以精度正确但性能不被接受。多个候选通过 build/install、边界 smoke、full compare 后，在 medium/long shape 上回退。
- 短 shape win 不足以支撑长上下文 kernel baseline；只改善 `S=64` 的候选经常回退 `S=512/2048/10240`。
- source、build output、installed package、active selector 可能互相漂移。有未跟踪变体或多写者时，先用 hash 或可核对 artifact 证明运行对象。
- 不要在 failing candidate 上叠加新优化。恢复 PASS surface 只是清理工作面，失败边界要记录并后续正向解决。

## 搬运边界

- Packed execution 是目标，目标周围重复的 per-`16x16` load/store/copy 才是冗余。
- 同形 block 合并前必须检查 load direction、transpose mode、stride 单位、source layout、destination layout。
- GM materialization 有时不是数学输入，但承担 order/lifecycle side effect；删除前要替换该 side effect。
- L0C->L1 Fixpipe 和 GM/L1 DataCopy 是不同 API surface，不能假设 `CFG_NZ` copy-out 支持 row-major multi-ND scatter。
- 分析 RHS 或 matmul operand 搬运时，必须拆成两层：GM->L1 `DataCopy/Nd2Nz` 和 L1->L0 `LoadData`。GM->L1 能一条 `DataCopy` 搬完整 `64x64`，不代表 L1->L0 也能一条 `LoadData` 覆盖完整 operand。
- L1->L0 是否能一条 `LoadData` 覆盖完整 B tile，取决于 L1 源布局是否已按 B operand 需要连续。arch32 参考 `attention/common/op_kernel/arch32/fia_block_cube_nonquant_gqa.h::LoadBTransposeToL0()`：当 `nL1Size == subNSize` 时可用 `repeatTimes = Kblocks * Nblocks` 单条 `LoadData`；否则按 K 维循环多条。不要先验否定，也不要不看 layout 直接套用。
- L0A/L0B 可优化点可能不对称。A operand 可考虑 `LoadData3DParamsV2`，B operand 可考虑 `LoadData2DParams` 的连续 B tile 分支；同一个 `64x64` 形状不代表两边能用同一种合并方式。

## Identity/Zero 生命周期

- 更长生命周期的 `prepared` flag 不是设计；要先证明 buffer 分配域、alias、`pipe_->Reset()`、local-buffer init、scratch 复用。
- 常量 identity tile 应放在不被 scratch/result/materialization helper 复用的 buffer；如果本地 buffer reset 会覆盖它，就按已验证 epoch 重建。
- 删除 zero-init 和删除 sync 要分开；zero 内容正确不代表同步边界可删。

## RHS/Compose 案例边界

- `FusedRhsAICProcess()` 中 RHS 不是 Neumann power 矩阵，而是 `gBKHalfWsGm_` 或 `vBetaHalfWsGm_`。`aicHalfL1TilePower_` 在 RHS 阶段只是复用为 L1 ping-pong stage，不能按名字判断数据语义。
- 当前常见验证形态 `dk=dv=64` 时，`gBK` 与 `vBeta` 都是单个 `64x64` RHS tile；若未来 `dk/dv > 64`，才需要区分 first tile 和后续 tile。
- RHS GM->L1 prefetch 若想掩盖 inverse 或前一 RHS MMAD，优先考虑新增独立 RHS prefetch L1 tile，而不是复用 `aicHalfL1TileInv_` / `aicHalfL1TilePower_`。这些 tile 在 inverse 和 RHS ping-pong 中已有消费者，强行复用容易引入覆盖风险。
- 预取 `gBK` 与 `vBeta` 时要分别检查 ready flag：`gBKHalfWsGm_` 由 AIV set `0x6` 后可读，`vBetaHalfWsGm_` 由 AIV set `0x5` 后可读。`vBeta` 的 prefetch 通常更可能被 `INV @ gBK` 覆盖，而不一定能被 inverse 覆盖。
- 若资源允许，方案 A 可以直接新增两个 `64x64 half` prefetch tile，分别给 `gBK` 和 `vBeta`，先做资源/编译验证，再替换 RHS 首个 tile 的 GM->L1 来源。该 phase 不应同时改 L1->L0 `LoadData` 合并。
- Odd/even compose 中第二轮 MMAD 的 L0A 常可提前装载，但 L0B 依赖第一轮 `cTmp -> tmp64` 的 Fixpipe 结果，只能等 L0C->L1 完成后再装。这里的核心瓶颈是 `cTmp -> tmp64 -> L0B` 往返，不是再换一套 DB 事件。

## AIV 半块计算

- 若两个 AIV sub-block 各自只消费 `halfChunkSize x chunkSize`，不要让每个 AIV 都生成完整 `chunkSize x chunkSize` 中间矩阵。
- `chunk_gated_delta_rule_C128` 的 `GammaCompute()` 是可参考模式：`divShape = {halfChunkSize, chunkSize}`，`gShape = {halfChunkSize, 1}`，输入从 `gCumSumUbFloat[subOffset_]` 开始，mask 从 `subOffset_ * chunkSize` 开始。Neumann 路径若仍 broadcast 完整 `64x64` 再只消费半块，就是明确冗余。
- 将完整 broadcast 改成半块 broadcast 是独立优化，应与 RHS prefetch、L1->L0 合并分开验证。

## Sync/Cache

- 删除 DCCI 不等于删除 sync。DCCI cleanup 保留 `Sync*` 可以 PASS，更大范围 DCCI+sync 删除可能长 shape FAIL。
- 统计 cache clean 时按真实 cacheline 操作计数，不只看 helper 调用次数。
- 小/中 shape PASS、长 shape FAIL 时，按真实边界失败处理。

## 记录模板

每个候选记录：

- 起始 baseline 或 active artifact；
- 唯一数据流变化；
- build/install 状态；
- boundary smoke ladder；
- full compare 结果；
- 精度 PASS 后的性能结果；
- 候选最终状态：accepted、restored、deferred 或 abandoned。
