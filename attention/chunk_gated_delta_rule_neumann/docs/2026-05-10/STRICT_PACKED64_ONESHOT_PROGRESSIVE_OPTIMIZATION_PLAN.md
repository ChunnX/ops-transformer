# Strict Packed64 One-shot Progressive Optimization Plan

## 2026-05-11 覆盖规则

- 后续执行以 `todo.md` 为主；本文件保留 strict packed64 one-shot 设计背景。
- 精度失败必须在当前目标上正向定位和修复，不能把回滚到 PASS baseline 当成任务完成。
- Packed MMAD 是 P7/P8 的目标，不是冗余；优化对象是重复搬运、copy-out、layout-copy、identity 构造和同步，不是把 packed 计算退回逐 `16x16`。
- `identity64` 目标是第一次有效生成后直接在 L1 复用，但必须先定义 raw-MMAD buffer epoch / `pipe_->Reset()` 生命周期；不能只加一个跨 epoch 的 prepared flag。
- L0C copy-out 若被硬件/API 边界卡住，可以记录为后置项，但不算 PASS，后续仍要单独解决或明确放弃。
- 所有计划完成并通过 gate 后，先保存新 baseline，再用 `msprof` 实测或 simulator/仿真分析寻找新的优化点。

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 目标

把旧 `P3L1/P8A` 融合计划、`packed64` 递进计划和当前 strict one-shot 定位结果合并为新的推进主线。

最终目标不是 Phase 10 的 packed64 容器 + 四个 `16x16` diagonal solve，而是：

1. P7 diagonal inverse 必须走 strict packed `64x64` one-shot。
2. one-shot 不能带 correctness 定位阶段遗留的冗余 identity 构造、重复 `inv @ I`、runtime scalar offset 和 compact bridge。
3. P8 compose 继续保持 lower-triangular 公式，但吸收 `ref/ref.cpp` 的 packed 抽取思想。
4. lower-left GM->L1 preload、P7 MMAD/Fixpipe、P8 compose、RHS 入口尽量做流水掩盖。
5. final RHS 数学语义不变，先保 resident L1 消费，再在生命周期明确后削减 BF16 GM inverse 写回。

本计划的 A0-A10 是必须完成的最低优化集，不是可选项。不能在 Phase 10 回退点、`S=64/256` 部分通过状态、或只完成其中几个优化点后宣布停止。A0-A10 全部实现并通过 gate 后，仍要根据 profile 继续判断是否存在新的 A11+ 优化点；如果性能仍未达到目标或 profiling 暴露明确瓶颈，继续拆新 phase 推进。

## 当前状态和基线

当前要分清两个状态：

1. correctness-safe 回退点仍是 Phase 10 baseline，不是最终目标。
2. 当前工作树里的 strict one-shot 修补候选只恢复到 `S=64/256` 通过，`S=1024` 仍失败，不能保存为 baseline。

- Selector: `p3l1_p8a`
- Active Stage1: `op_kernel/p8afullocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a.h`
- Safe baseline: `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_packed64_phase10_drop_wait0d_baseline.h`
- Safe baseline correctness: build/install PASS，`S=64/256/1024` smoke PASS，compare PASS。
- Performance: `tests/pytest/prof_out/packed64_phase10_drop_wait0d_20260510/performance_compare.csv`
- Current partial candidate:
  - build/install PASS。
  - `S=64` PASS：out `3.173694e-03`，state `5.276948e-04`。
  - `S=256` PASS：out `3.173694e-03`，state `9.178072e-04`。
  - `S=1024` FAIL：out `1.118436e-01`，state PASS，worst actual 接近 0。
  - 该候选只能作为继续修 strict one-shot 的工作现场，不能作为 correctness baseline。

strict one-shot 已经证明 L1->L0 placement、Fixpipe offset、identity placement 可以 correctness PASS，但当时实现保留了明显性能负担：

- 每个 `paraIdx` 重建 `identity64`。
- 每轮用 `result64 @ identity64` 做 C 初始化/拷贝。
- `scratch64` 和 `lowerWork64` 共用 `aicHalfL1Scratch_`，阻断 lower-left preload overlap。
- helper 仍有较多 runtime scalar offset/shape 分支。
- P8 仍是 correctness-first direct compose，没有迁入 ref-style odd/even packed 抽取。
- BF16 `attnWsGm_` 写回仍保留；直接删除或跳过 upper zero writeback 已在 `S=1024` 失败。

## 不变边界

- 只改 Stage1 inverse path：
  - FP16 `L/INV`
  - diagonal inverse
  - P8 lower-triangular compose
  - inverse resident L1 / writeback 时序
- 不改 Stage2、Stage3、QK、`gBK/vBeta` 生成语义。
- P7/P8 必须严格参考 `ref/ref.cpp` 的 pack/layout 方式实现。
  - 主线按 packed block 抽取、packed block 写回组织 L0A/L0B/L0C。
  - 不再把最终 P7/P8 主线退回到单个 `16x16` 求解再拼接。
  - P8 只参考 ref 的 packed layout，不照搬 ref 的 upper-triangular 公式。
  - P8 block 顺序按当前 lower-triangular 公式排：`tmp = lower @ topInv`，`out = bottomInv @ tmp`。
- final RHS 数学语义不变：
  - `AICProcess(attnWsGm_, gBKWsGm_, outKCumdecayGm_, ...)`
  - `AICProcess(attnWsGm_, vBetaWsGm_, outVInnerGm_, ...)`
  - 或当前 fused resident-L1 等价路径，但不能改变 RHS 公式。
- 只测 64-aligned `S=64/256/1024`。tail case 保持注释或不启用，直到 tail 支持单独恢复。
- 每个 phase 只解决一个问题。失败时先在当前优化点上正向定位和修复，不叠加下一项优化；需要恢复上一个 PASS baseline 时，只作为放弃候选或重建干净工作面的动作。
- 每个 phase 做完且通过完整 correctness gate 后，都要保存一个阶段 baseline；未通过 `S=64/256/1024 + compare` 的候选不得进 `op_kernel/baseline/`。
- 若 L0C 累加是必须的，允许第一轮 `INV @ I` 作为 bootstrap，但 `identity64` 只能在算子生命周期内生成一次并缓存复用，不能每个 `paraIdx` 重建。
- 后续必须把 identity 准备收敛为一次性路径：当前 `aicHalfL1Identity_` 的按 blockSize/lifecycle 重建只能作为过渡，不能作为最终形态。

## Stage1 旧做法与当前做法

### 原来 Stage1 怎么做

Stage1 的 AIV 侧负责生成两个 FP16 GM workspace：

- `L_powHalfWsGm_`：由 `InverseCompute()` 写出的 lower/power 输入。
- `INVInitHalfWsGm_`：由 `InverseCompute()` 写出的 initial inverse / lower-left 输入。

AIC 侧等待 AIV 的同步 flag 后消费这两个 workspace：

- 从 `L_powHalfWsGm_` / `INVInitHalfWsGm_` 搬 diagonal `16x16` 或 packed diagonal block 到 L1。
- 用 AIC 本地 `aicHalfL1Identity_` 做 MMAD copy/累加的 identity operand。
- `aicHalfL1Identity_` 不是 AIV 生成的 GM workspace，而是 AIC local L1/TSCM 里的 identity/cache；旧 helper `PrepareHalfL1IdentityByMmad(blockSize)` 会先写 `tmpWsGm_`，再用 MMAD 转成 half identity 放到 `aicHalfL1Identity_`。

### 当前 strict one-shot 候选怎么做

当前 AIV 侧仍然只生成同样两个 GM workspace：`L_powHalfWsGm_` 和 `INVInitHalfWsGm_`。变化在 AIC 侧：

- AIC 把四个 diagonal `16x16` block 定点搬入 packed `64x64` 容器：`result64` 和 `diagPower64`。
- P7 strict one-shot 尝试直接在 packed `64x64` 上做 diagonal-only Neumann。
- P8 lower-left 从 `INVInitHalfWsGm_` 搬到 `lowerWork64[(16,0)/(48,32)/(32,0)]`，compose 公式仍按 lower-triangular：`tmp = lower @ topInv`，`out = bottomInv @ tmp`。
- 当前候选里的 `PreparePacked64IdentityFromPacked16()` 仍是 correctness 定位遗留：它会把 16x16 identity 组装成 `identity64`，并且和 `PrepareHalfL1IdentityByMmad(16/32/64)` 共用 `aicHalfL1Identity_` 这块可变 cache。后续 A1 要把 `identity64` 改成算子生命周期内一次性生成并稳定复用，避免被不同 blockSize 的 identity 覆盖。
- 这个优化必须落地为“只做一次”的 identity 路径：identity64 准备完成后，P7/P8 和 copy/accumulate helper 不应再反复走 scalar 写 GM、GM->L1 MMAD 转 half、或 16x16 identity 拼 64x64。

## 继承旧计划的有效结论

### 旧 P3L1/P8A 计划继承点

- P3 diagonal result 可以落在 packed L1。
- P8 的 `32x32` 和 `64x64` compose 可以走 L1-resident。
- final RHS 已有 resident L1 消费桥。
- 但旧计划没有解决 strict packed `64x64` one-shot diagonal solve。

### packed64 计划继承点

- `result64 / diagPower64 / lowerWork64` 三块主资源模型成立。
- GM->L1 定点搬入四个 diagonal `16x16` 到 `(0,0)/(16,16)/(32,32)/(48,48)` 已通过。
- lower-left 可以单独放入 `lowerWork64` 的 `(16,0)/(48,32)/(32,0)`。
- compose output 可以直接写回 `result64`。
- AIC/AIV 同步已有 Phase 10 回退点。

### strict one-shot 定位继承点

- one-shot placement correctness 已过，不要重复把它当成未知问题。
- 后续重点是把 correctness 版本的性能负担拆掉。
- one-shot 必须成为新 active 目标，不再用四个 `16x16` solve 作为最终实现。

## 资源模型

长期 L1/TSCM half tile：

| 名称 | 当前候选 tensor | 语义 |
|---|---|---|
| `result64` | `aicHalfL1P3Result_` | diagonal inverse 初值、one-shot 输出、P8 compose 最新结果 |
| `diagPower64` | `aicHalfL1TilePower_` | `P/P^2/P^4/P^8` power 递推 |
| `lowerWork64` | `aicHalfL1Scratch_` 或重分配 alias | P8 lower-left 输入，不再长期承担 diagonal scratch |
| `diagScratch64` | 待 Phase 3 决定 | one-shot 临时 power/output scratch；不能和 overlap 阶段的 `lowerWork64` 冲突 |
| `identity64` | `aicHalfL1Identity_` 或一次性 cached | 只允许缓存/常量化，不允许每 para 重建 |

L0C 逻辑分区：

- `cInv`: inverse accumulate / bootstrap。
- `cPower`: `diagPower64 @ diagPower64` 输出。
- `cTmp`: P8 `tmp = lower @ topInv`。
- `cOut`: P8 output 或备用。

## 推进阶段

### Phase A0: 重新接入 strict one-shot 目标路径

解决的问题：当前 active 已回到 Phase 10，one-shot 不在 active 路径。

任务：

1. 以 Phase 10 baseline 为起点，只重新接入 strict packed64 one-shot P7。
2. 保留 P8 compose、BF16 GM 写回、resident L1 RHS 与 Phase 10 一致。
3. one-shot 入口必须是显式 compile-time 开关或明确函数路径，不能和旧四个 `16x16` solve 混成不可读分支。
4. 不做性能优化，只恢复可验证的 target path。
5. P7/P8 pack 方式必须按 ref-style packed layout 落地，不允许把主线实现回落成单个 `16x16` 求解拼接。

验收：

- build/install PASS。
- `S=64 -> 256 -> 1024` PASS。
- compare PASS。
- 记录性能，但不因性能退化回避 one-shot 主线；退化作为后续 phase 要解决的问题。
- 保存阶段 baseline，例如 `op_kernel/baseline/..._packed64_phaseA0_strict_oneshot_baseline.h`。

### Phase A1: 去掉每 para 重建 `identity64`

解决的问题：`PreparePacked64IdentityFromPacked16()` 是 one-shot correctness 定位遗留，重建 identity 的 GM 写、GM->L1 转换、四次 `I16 @ I16` 都是冗余。

任务：

1. 删除 one-shot P7 内的 per-para identity 构造。
2. 改为算子生命周期内一次性准备 `identity64`，或用已有 `aicHalfL1Identity_` 的 stable cached 版本。
3. 若复用 `aicHalfL1Identity_`，必须保证后续 `PrepareHalfL1IdentityByMmad(16/32/64)` 不会覆盖 active `identity64`；否则独立出 `identity64` cache。
4. 最终必须收敛到只做一次；若 identity L1 生命周期不稳定，可以先做一次 per-core/per-workspace cached 作为过渡，但不得把 per-blockSize/per-para 重建留到收口版本。
5. 本阶段不改 `inv @ I` 的数学使用方式，只解决 identity 构造来源。

当前补充边界：

- active path 已有独立 `aicHalfL1Identity64_` tile，但它仍受 `packed64IdentityPrepared_` / `packed64IdentityGmPrepared_` 生命周期控制，不等价于真正算子生命周期常驻。
- `16x16` half identity 在 BF16 writeback、resident copy、pack/compact copy 中只是 `X @ I16` 搬运替身；收口版本应改为直接搬运/专门 layout copy。过渡期若还必须用 `I16`，也应从常驻 `identity64` 的 diagonal fractal 派生，不再单独生成。
- A30 只允许验证删除 `ParaChunkAIC()` 中 per-para-group `packed64IdentityPrepared_ = false;` 这一点，不把 BF16 GM side effect、resident copy、`RestoreRawMmadState()` 删除叠进同一 phase。

验收：

- correctness gate 全过。
- profile 重点看 scalar、GM/MTE2、identity 相关 MMAD 是否消失。
- 保存阶段 baseline，例如 `op_kernel/baseline/..._packed64_phaseA1_identity_once_baseline.h`。

### Phase A2: 去掉重复 `result64 @ identity64`

解决的问题：one-shot 每轮用 `result64 @ I` 做 C 初始化/拷贝，导致 full-64 MMAD 被当成 copy 用。

任务：

1. 将 Neumann round 从：
   - `cInv = result64 @ I`
   - `cInv += result64 @ power`
   改为一次 bootstrap 后复用，或改成 `result64 @ (I + power)` 的单 MMAD 形式。
2. 允许保留第一轮必要的 bootstrap，但不允许每轮重复 `result64 @ I`；如果 L0C 累加不可避免，`INV @ I` 只保留一次起步。
3. 如果使用 `I + power`，必须保证只在 diagonal block 加 identity，off-diagonal/padding 仍为 0。
4. 本阶段不改 P8、不改 lower-left preload、不改 helper 模板化。

验收：

- correctness gate 全过。
- performance 必须单独记录，与 A1 比较 full-64 MMAD 次数和总时延。
- 保存阶段 baseline，例如 `op_kernel/baseline/..._packed64_phaseA2_no_repeat_inv_i_baseline.h`。

### Phase A3: strict one-shot helper 模板化和 offset 常量化

解决的问题：当前 one-shot helper 仍有过多 runtime shape/offset/scalar 计算。

任务：

1. 固化 `packedSize=64`、`blockSize=64`、`fractal=16`、diagonal bases `{0,16,32,48}` 为 `constexpr` 或模板参数。
2. 对 L1->L0 diagonal-only load 采用 fixed four-fractal unroll。
3. 对 Fixpipe packed NZ 写回采用 compile-time offset helper，避免运行时重复算 `CompactMatrixNzOffset`。
4. 拆出清晰的 one-shot 专用 helper，例如：
   - `LoadPacked64DiagOnlyToL0A<>`
   - `LoadPacked64DiagOnlyToL0B<>`
   - `MmadPacked64DiagOnly<>`
   - `FixpipePacked64DiagOnly<>`
5. 不引入新的 buffer 生命周期变化。

验收：

- correctness gate 全过。
- profile 重点看 scalar 时间、MTE1 时间和编译生成代码是否减少循环/分支。
- 保存阶段 baseline，例如 `op_kernel/baseline/..._packed64_phaseA3_constexpr_helper_baseline.h`。

### Phase A4: 分离 `diagScratch64` 与 `lowerWork64` 生命周期

解决的问题：`aicHalfL1Scratch_` 同时当 diagonal scratch 和 lower-left work，导致 lower-left preload 不能和 P7 one-shot 并行。

任务：

1. 明确 one-shot P7 需要的临时 scratch：
   - power output scratch
   - optional `I + power` operand
   - temporary Fixpipe target
2. 明确 P8 需要的 `lowerWork64`：
   - `(16,0)`
   - `(48,32)`
   - `(32,0)`
3. 通过重排生命周期或重新 alias，使 P7 scratch 和 P8 lower-left 不写同一 L1 区域。
4. 本阶段只做 buffer ownership，不做 preload overlap。

验收：

- correctness gate 全过。
- 日志中写清每块 L1 tile 的 producer/consumer 和最后使用点。
- 保存阶段 baseline，例如 `op_kernel/baseline/..._packed64_phaseA4_l1_lifetime_baseline.h`。

### Phase A5: lower-left GM->L1 preload 与 one-shot 计算重叠

解决的问题：`INVInitHalfWsGm_ -> lowerWork64` 现在没有掩盖在 diagonal one-shot MMAD/Fixpipe 后面。

任务：

1. 在 A4 的资源分离后，把 lower-left preload 安排到 P7 one-shot 计算窗口内。
2. preload 只搬：
   - `INVInitHalfWsGm_ -> lowerWork64[(16,0)]`
   - `INVInitHalfWsGm_ -> lowerWork64[(48,32)]`
   - `INVInitHalfWsGm_ -> lowerWork64[(32,0)]`
3. 事件上保证：
   - MTE 写 lowerWork64 不踩 P7 scratch/result。
   - P8 读 lowerWork64 前等待对应搬运完成。
   - P7 Fixpipe 写 result64 与 preload 不竞争同一地址。
4. 不改 P8 公式和 P8 packing。

验收：

- correctness gate 全过。
- profile 重点看 lower-left MTE2/MTE1 时间是否被 P7 M/FIX 掩盖。
- 保存阶段 baseline，例如 `op_kernel/baseline/..._packed64_phaseA5_lower_preload_overlap_baseline.h`。

### Phase A6: P8 第一层迁入 ref-style packed 抽取

解决的问题：当前 `(16,0)` 和 `(48,32)` 两个 independent `16x16` compose 仍分别执行，没有按 ref 思路合成 `32x32` packed MMAD。

任务：

1. 参考 `ref/ref.cpp::CopyOddOrEvenBlocksL1ToL0()` 的 layout 思路，只迁移 packing，不照搬 upper-triangular 公式。
2. block 顺序按当前 lower-triangular 公式排，不按 ref upper-triangular 公式排。
3. 对两组 independent lower-triangular compose 做 packed `32x32`：
   - `tmp0 = lowerWork64[(16,0)] @ result64[(0,0)]`
   - `tmp1 = lowerWork64[(48,32)] @ result64[(32,32)]`
   - 合成一次 packed `32x32` MMAD。
4. 再做第二个 packed `32x32` MMAD：
   - `out0 = result64[(16,16)] @ tmp0`
   - `out1 = result64[(48,48)] @ tmp1`
5. output 直接 Fixpipe 到：
   - `result64[(16,0)]`
   - `result64[(48,32)]`
6. 本阶段不改 `32x32 -> 64x64` compose。

验收：

- correctness gate 全过。
- 失败时只查 first-layer packed L0A/L0B extraction、output offset，不同时改 second layer。
- 保存阶段 baseline，例如 `op_kernel/baseline/..._packed64_phaseA6_p8_first_layer_packed_baseline.h`。

### Phase A7: P8 第二层 `32x32 -> 64x64` packed compose 收口

解决的问题：第一层 packed 后，第二层仍是 direct `32x32` compose，没有形成完整 P8 packed compose 结构。

任务：

1. 保持 lower-triangular 公式：
   - `tmp = lowerWork64[(32,0)] @ result64[(0,0)]`
   - `out = result64[(32,32)] @ tmp`
2. 将 L1->L0 extraction 和 Fixpipe 写回统一到 ref-style packed helper 的 `block_size=32` 形态。
3. output 直接写 `result64[(32,0)]`。
4. 不改 P7，不改 BF16 GM 写回。

验收：

- correctness gate 全过。
- compare PASS 后才跑 performance。
- 保存阶段 baseline，例如 `op_kernel/baseline/..._packed64_phaseA7_p8_second_layer_packed_baseline.h`。

### Phase A8: RHS / BF16 GM inverse 写回生命周期定位

解决的问题：当前理论上应在 L1 消费 inverse，但直接删 BF16 `attnWsGm_` 写回或 upper zero 写回会在 `S=1024` 失败。

任务：

1. 不直接删写回，先梳理 `attnWsGm_` 的 producer/consumer：
   - lower-triangular BF16 block 写回。
   - upper zero block 写回。
   - resident L1 copy。
   - fused RHS 读取边界。
   - 跨 chunk workspace 复用边界。
2. 将失败过的两个尝试列为禁区，除非本阶段给出新的生命周期屏障：
   - 删除完整 BF16 GM 写回。
   - 跳过 upper zero writeback。
3. 若要试验，只允许一次改一个边界：
   - 先改清零位置。
   - 再改写回范围。
   - 最后改 RHS 输入来源。
4. 本阶段目标是找出 BF16 GM 写回的真实依赖，不以性能收益为验收。

验收：

- 必须包含 `S=1024`。
- 若任何子尝试失败，先停在该子尝试上定位具体 side effect 或生命周期缺口；只有明确放弃该候选或需要重建干净工作面时，active 才恢复到 A7 PASS 状态。
- 输出明确结论：GM 写回依赖是清零、同步、跨 chunk 可见性，还是 RHS fallback 消费。
- 保存阶段 baseline，例如 `op_kernel/baseline/..._packed64_phaseA8_bf16_lifecycle_baseline.h`。

### Phase A9: resident L1-only inverse 消费

解决的问题：在 A8 明确生命周期后，削减或删除 BF16 GM inverse 写回，让 inverse 主要在 L1 被 RHS 消费。

任务：

1. 保持 final RHS 数学语义不变。
2. 若 A8 证明 upper zero 只承担清零作用，则替换为更小范围或更早的清零。
3. 若 A8 证明 BF16 GM 完全不被 RHS 正常路径消费，则删除或延后完整 GM 写回。
4. 若 A8 证明仍有跨 chunk 依赖，则只做范围缩减，不强删。

验收：

- correctness gate 全过。
- performance 必须对比 A7/A8，重点看 MTE2、Fixpipe、总时延。
- 保存阶段 baseline，例如 `op_kernel/baseline/..._packed64_phaseA9_l1_only_rhs_baseline.h`。

### Phase A10: 收口清理和新 baseline

解决的问题：阶段性 helper、开关、fallback 容易把未验证路径留在 active 中。

任务：

1. 删除或隔离不再使用的 one-shot diagnostic helper。
2. active path 必须清楚表达：
   - strict packed64 one-shot P7
   - ref-style P8 lower-triangular packed compose
   - resident L1 RHS 消费
   - 已验证的 BF16 GM 写回策略
3. 保存新的 baseline 只能在 full compare + performance 对比完成后进行。
4. 文档更新：
   - `todo.md`
   - `docs/2026-05-10/WORK_LOG.md`
   - `docs/2026-05-10/PROGRESS_AND_ISSUES.md`

验收：

- build/install PASS。
- `S=64/256/1024` PASS。
- compare PASS。
- performance 至少不差于 Phase 10，若仍差，必须明确剩余瓶颈，不保存为性能 baseline。

## 停止条件

只有同时满足以下条件，才能认为本轮 strict packed64 one-shot 优化收口：

1. A0-A10 全部实现，且没有用 Phase 10、四个 `16x16` solve、GM fallback、single-block loop 或 compact-only bridge 代替目标实现。
2. `identity64` 已收敛为一次性准备并复用，不再按 `paraIdx`、blockSize 或 helper 调用反复重建。
3. P7 为 strict packed `64x64` one-shot active path。
4. P8 为 ref-style packed layout + 当前 lower-triangular 公式。
5. BF16 GM 写回和 resident L1-only 消费生命周期已经明确，最终策略通过完整 correctness gate。
6. 每个 PASS phase 都已保存阶段 baseline，最终 active baseline 通过 build/install、`S=64/256/1024` 和 compare。
7. 已完成 performance 对比；如果仍差于 Phase 10 或 profile 仍有明确瓶颈，不能停止，必须继续提出并推进 A11+ 优化点。

## 验证门槛

每个 phase 都执行同一组 correctness gate。只有 compare PASS 后才跑 performance。

```bash
bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16
../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp
timeout 90s env CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer python3 tests/pytest/compare_chunk_gdr_three_way.py --seed 0 --batch-size 1 --seqlen 64 --nk 4 --nv 4 --dk 64 --dv 64
timeout 90s env CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer python3 tests/pytest/compare_chunk_gdr_three_way.py --seed 0 --batch-size 1 --seqlen 256 --nk 4 --nv 4 --dk 64 --dv 64
timeout 90s env CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer python3 tests/pytest/compare_chunk_gdr_three_way.py --seed 0 --batch-size 1 --seqlen 1024 --nk 4 --nv 4 --dk 64 --dv 64
CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer bash tests/pytest/test_run.sh compare
```

performance 记录要求：

- 每个 PASS phase 单独输出到 `tests/pytest/prof_out/<phase_name>_YYYYMMDD/`。
- 至少对比：
  - Phase 10 baseline
  - 上一个 PASS phase
  - 当前 phase
- 关注字段：
  - total latency
  - `aic_mac_time_us`
  - `aic_mte1_time_us`
  - `aic_mte2_time_us`
  - `aic_fixpipe_time_us`
  - `aic_scalar_time_us`

## 禁止项

- 不把旧四个 `16x16` diagonal solve 描述成 one-shot 完成。
- 不用 GM fallback、single-block loop、compact-only intermediate 代替 strict packed64 target。
- 不在 BF16 GM 写回生命周期未明确前重复删除写回。
- 不把 correctness diagnostic PASS 当性能采纳。
- 不在一个 phase 同时改 P7、P8、RHS。
- 不保存未通过 full compare 的实验文件为 baseline。
