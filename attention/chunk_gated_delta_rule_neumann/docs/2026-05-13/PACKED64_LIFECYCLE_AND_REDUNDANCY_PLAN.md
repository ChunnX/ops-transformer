# Packed64 生命周期与冗余拆解计划

Date: 2026-05-13

## 当前基线

- Active path: `p3l1_p8a / C64_1616_FP16`。
- Active Stage1: `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1.h`。
- Continuation baseline: A102。
- Source SHA256: `9cd969fc8edef610266c2dd974b70bb214eeb6c9f3d3ebfd758607736d50ac01`。
- Correctness: clean rebuild + isolated OPP install PASS, `S=64/256/1024` PASS, full compare `16 passed in 6.20s`。
- Performance ratios: `1.082 / 1.434 / 2.182 / 2.209`。

本计划只覆盖 Stage1 inverse/writeback/RHS exposure 路径。不要修改 Stage2、Stage3、QK、`gBK/vBeta` 生成和 final RHS 数学。

## cache clean 说明

之前说的“4 个 cache clean”应精确理解为：

- `CleanPacked64DiagonalHalfBlocks(gm)` 对 `(0,0)/(16,16)/(32,32)/(48,48)` 四个 diagonal `16x16` block 各调用一次 `CleanHalfMatrixCacheLines()`。
- `CleanHalfMatrixCacheLines()` 内部按 half cacheline 清理。对一个 `16x16` half block，`HALF_CACHELINE_ELEMS=16`，因此每行 1 条 cacheline，合计 16 次 `DataCacheCleanAndInvalid<half, ..., DcciDst::CACHELINE_UB>()`。
- P7 strict diagonal preload 对两个源各调用一次 packed diagonal load：
  - `L_powHalfWsGm_ -> aicHalfL1TilePower_`
  - `INVInitHalfWsGm_ -> aicHalfL1P3Result_`
- 所以每个 chunk 的 P7 diagonal preload 是 8 个 block-level clean 调用，约 128 条 single-cacheline DCCI。

这些 clean 是 AIC 侧在 GM->L1 读取前做的可见性/一致性边界，用来读取 AIV 刚写入并 flush 出来的 half workspace。A95 直接删除 strict P7 diagonal preload AIC-side clean 已经精度 PASS 但性能退化，因此不按原样重试。

## tri_inverse 生命周期参考

参考 `experimental/attention/tri_inverse` 的可复用结论，而不是照搬实现：

- Phase22: AIC 每个 `ProcessParaTiles` batch 加载一次 `identity32`，batch 内复用。
- Phase23: AIV 只在每个 batch 的首 tile 写 `identity32` scratch，AIC batch 内复用。
- Phase27: 尝试提升到 core lifetime，并同时 hoist raw-MMAD init/restore/reset；正确但总时延不优，不能把它作为“identity core lifetime 一定更快”的证明。
- Phase29a: 隔离 first-batch-only L1 reuse，不 hoist init/reset；这是更接近当前 CGDR 可借鉴的生命周期拆法。
- Phase31: P7 resident boundary 必须先成立，`cInv=aicL0C_` 语义持有四个 diagonal lane 后，再进入 P8 `cTmp + cInv` accumulation。

映射到 CGDR：

- 当前 `identity64` 是 raw-MMAD inverse-window epoch 级：`ParaChunkAIC()` 入口会清 `packed64IdentityPrepared_`。
- 不再重试 A30 式单纯延长 prepared flag；它已在 `S=1024` 失败。
- 后续如果做 operator/core lifetime identity64，必须先证明 L1 identity64 buffer 不会被 `pipe_->Reset()`、`InitBuffer()`、scratch alias 或 RHS raw-MMAD 复用覆盖。
- 生命周期优化要拆成两件事分别验证：
  - identity 内容在 L1 的复用范围。
  - raw-MMAD state/init/reset 的复用范围。

## 性能放大判断

A101 profile 显示大 shape 的 2.x 主要来自 per-64-chunk 重复项放大，尤其 AIC scalar 时间：

```text
S=64:    normal 113us,  neumann 188us,   +75us
S=2048:  normal 1962us, neumann 6478us,  +4516us
S=10240: normal 9663us, neumann 32260us, +22597us
```

因此优先处理“每个 chunk 都重复做、且不参与当前 RHS 数学”的项。不要先重试已经证明只是 small-shape win 或 medium/large 退化的同步删除。

## A102 结果：raw-MMAD epoch 辅助 tile 生命周期

A102 已完成并采用。

- 单点：`InitRawMmadLocalBuffers()` 建立 raw-MMAD local buffer epoch 后调用 `PrepareRawMmadAuxiliaryTiles()`。
- `PrepareRawMmadAuxiliaryTiles()` 一次准备：
  - `EnsureHalfL1Zero(C64_PACKED_SIZE_U32 * C64_PACKED_SIZE_U32)`
  - `EnsurePacked64Identity()`
- active full64 helper 不再各自 lazy 生成 zero/identity。
- 生命周期边界：这是 raw-MMAD local buffer epoch 级，不是整算子 lifetime。`RestoreStage1MatmulLocalBuffers()` / `pipe_->Reset()` 后仍重新建立。
- baseline: `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phaseA102_raw_epoch_auxiliary_tiles_baseline.h`。
- package SHA256: `e4b17d8eba153286e381f42310274ab2a5a3f4531ab8aade1a59b1af0a6044c6`。
- isolated OPP: `/tmp/cgdr_a102_opp_20260513_1006/vendors/custom_transformer`。
- Performance: `tests/pytest/prof_out/packed64_phaseA102_raw_epoch_auxiliary_tiles_valid_20260513/performance_compare.csv`。

注意：`tests/pytest/prof_out/packed64_phaseA102_raw_epoch_auxiliary_tiles_20260513/` 是默认 `custom_transformer` op_api 被符号污染期间产生的无效 profiling 目录，不能用于比较。

## A103+ 执行计划

每个 phase 只改一个数据流。每个 phase 必须走 build/install、`S=64/256/1024`、full compare；PASS 后再跑 performance，失败则记录并恢复到最近 PASS baseline。

### A103: 删除 BF16 `attnWsGm_` 真实数据写

目标：验证当前 active RHS 不消费 `attnWsGm_` 后，能否删除 inverse BF16 GM materialization 的真实写出。

范围：

- 只动 `StorePackedL1InverseToBf16Attn()` 的调用/主体。
- 不改 P7/P8 公式。
- 不改 `StorePackedL1InverseToResidentL1()`。
- 不改 `FusedRhsAICProcess()`。
- 不改 cross-core flag 语义。

推进方式：

1. A102a：先删除真实 BF16 GM 数据写出，但保留必要的本地 ordering/sync 骨架；如果没有明确需要，不保留任何 GM 写。
2. A102b：若 A102a PASS，再继续拆剩余空同步；每次只删一类 sync。
3. A103 fail 时，只定位 BF16 materialization 携带的是哪类边界：Fixpipe 可见性、M/S ordering、workspace 清零、还是隐藏 consumer。不得把 `attnWsGm_` 重新描述成 RHS 数学输入。

预期收益：随 chunk 数放大的 GM/Fixpipe/zero-rect materialization 成本下降。

### A104: 求逆输入 CC 整块 MTE2 搬入 L1

前置：A103 PASS。

目标：替换当前 strict full64 入口处的“P7 只搬 diagonal + P8 再搬 lower blocks”模式，改为求逆开始时整块 `64x64` CC 输入各搬一次到 L1，并按 CC 大小做 cache clean。

当前 strict full64 路径的重复点：

- P7 diagonal preload 只从 `L_powHalfWsGm_` / `INVInitHalfWsGm_` 取四个 diagonal `16x16`。
- P8 lower preload 又从 `INVInitHalfWsGm_` 取 `(16,0)`、`(48,32)`、`(32,0)`。
- 两类 preload 分别做 block-level cache clean，导致同一个 CC 输入在求逆阶段被分散清理和搬运。

A104 单点改法：

- 新增 full64 GM->packed L1 helper。
  - `CleanHalfMatrixCacheLines(gm, chunkSize_, chunkSize_, chunkSize_)` 清整块 CC。
  - 一次 `Nd2NzParams DataCopy` 搬 `64x64` 到 packed64 L1。
- strict full64 入口准备 L1 resident 输入：
  - `INVInitHalfWsGm_ -> aicHalfL1Scratch_`，作为完整 `INVInit/lowerNeg` CC 输入。
  - `L_powHalfWsGm_ -> aicHalfL1TilePower_` 仍只搬 diagonal packed lanes，因为 P7 power 只消费 diagonal；把 `L_pow` off-diagonal 也搬入 L1 是新冗余，不纳入 A104。
- P7 改为从 `aicHalfL1Scratch_` 读 diagonal `INVInit`，并继续把 diagonal inverse 写到 `aicHalfL1P3Result_`。
- P8 继续读 `aicHalfL1Scratch_` 里的 lower blocks，不再 active replay GM->L1 lower preload。

生命周期约束：

- `aicHalfL1P3Result_` 仍是 result64，不承载完整 `INVInit` 输入。
- `aicHalfL1Scratch_` 在 strict full64 path 中已经是 P8 lowerWork64；整块 `INVInit` 放这里不会改变 P8 消费布局。
- `aicHalfL1TilePower_` 仍放 packed diagonal `L_pow`；后续 power 中间态覆盖该 buffer 是允许的。
- 不改 P7/P8 公式，不改 final RHS，不改 `StorePackedL1InverseToResidentL1()`。
- 失败时只定位 full CC preload 的 layout/cache/sync，不叠加 direct resident writeback 或 identity64 lifetime。

预期收益：

- `INVInit/lowerNeg` 只在求逆入口做一次 full CC preload，P7 diagonal 和 P8 lower 均从同一份 L1 resident 输入取数。
- 用 `INVInit` 整块 CC cache clean 替换其 diagonal/lower block-level clean。
- 删除 strict full64 path 上 P8 lower GM replay 的 MTE2 和 sync。

原 A104 `final cInv direct resident L1 writeback` 后移为 A105，等待 A104 PASS 后再单独评估。

### A105: final `cInv` 直接写 resident L1

前置：A104 PASS。

目标：减少 `cInv -> aicHalfL1P3Result_ -> aicHalfL1ResidentInv_` 的二次 L1 materialization/copy。

候选方向：

- second-layer P8 最终 `cInv` 不再先 full64 Fixpipe 到 `aicHalfL1P3Result_` 后再 row-prefix copy 到 resident。
- 直接把最终 `cInv` Fixpipe 到 `aicHalfL1ResidentInv_[paraIdx]`，并验证 upper-zero 语义是否已由 L0C 内容保证。

约束：

- RHS 真实输入仍是 `aicHalfL1ResidentInv_`。
- 若 resident 需要 compact/row-prefix 特殊布局，先保留 `result64` 中转，不做猜测性改写。
- 失败时区分 layout 错、upper 未清零、Fixpipe 可见性错，而不是回退到 BF16 GM fallback。

### A106: identity64 生命周期按 tri_inverse 方式拆解

前置：A103/A104 至少有一个稳定 PASS baseline；不要在 failing surface 上做 lifecycle。

目标：从 raw-MMAD epoch 级 identity64，推进到更长生命周期，但不重试 A30 裸 flag。

计划：

1. A104a：源码层确认 `aicHalfL1Identity64_` 的分配域、alias、`pipe_->Reset()` 覆盖关系和 RHS raw-MMAD 使用关系，写入本文件。
2. A104b：仿 tri_inverse Phase29a，只做“首 batch 生成、后续 batch L1 复用 identity64”，不 hoist raw-MMAD init/reset。
3. A104c：若 A104b PASS 且 profile 指向有效，再单独验证 raw-MMAD state/init/reset hoist；不要和 identity 内容生命周期混在一起。

禁止：

- 不重试 A30 直接移除 `ParaChunkAIC()` 的 reset。
- 不重试 same-GM-region reload。
- 不用 AIV/GM dump 来证明 identity 内容。

### A107: P7 diagonal preload clean 拆分

前置：A103/A104/A105 中至少完成当前最确定收益项后再做。

目标：降低 P7 diagonal preload 前的 AIC-side DCCI 成本，但不重试 A95 exact 删除。

拆分：

- A106a：只处理 `L_powHalfWsGm_` diagonal clean。
- A106b：只处理 `INVInitHalfWsGm_` diagonal clean。
- A106c：若二者单独 PASS，再尝试合并策略或依赖 AIV-side flush 的更小 clean 范围。

每一步都要看 performance，不只看 correctness；A95 已证明 correctness PASS 不等于收益。

### A108: 后置 L0C copy-out 专项

目标：回到 A42-deferred / A45-deferred 的 L0C partial copy-out 问题。

当前已知：

- DAV220 L0C->L1 `Fixpipe<..., CFG_NZ>` 没有 GM row-major 路径那种多 ND scatter 参数。
- 非零 L0C source offset/partial rectangle 多次 timeout 或 full compare fail。

策略：

- 不在 A103-A106 上叠加 A42/A45。
- 等前面收敛后，单独建立 copy-out support matrix：full64、top-left 32x32、nonzero offset 32x32、diagonal-only。
- 若硬件/API 边界不支持 diagonal scatter，明确记录 unsupported boundary，并保留 full64 copy-out 或选择合法矩形形态。

## 下一步

立即执行队列：

1. A104：strict full64 求逆入口整块 preload `INVInit` 到 L1，`L_pow` 继续 diagonal preload。
2. 删除 active strict path 的 P8 lower GM replay，但保留非 strict/tail path 不变。
3. Gate: build/install, `S=64 -> 256 -> 1024`, full compare。
4. PASS 后跑 A103b 同款 performance，对比 `1.035 / 1.359 / 2.169 / 2.190`。
5. A104 PASS 后再进入 direct resident writeback；不跨 phase 叠加。
