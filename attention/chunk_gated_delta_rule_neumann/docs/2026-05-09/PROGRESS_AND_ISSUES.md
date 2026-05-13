# 2026-05-09 进度与问题

## 已完成

- 已确认 active path：
  - selector: `op_kernel/selected/stage1_selected.h`
  - config: `op_kernel/selected/stage1_selected_config.h`
  - active Stage1: `op_kernel/fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocalnew.h`
  - fused helper: `op_kernel/fulllocalnew/fast_strictri_inverse.h`
- 已确认 `validLen == chunkSize_ == 64` 时进入 `FastStrictTriInverse64::Compute()` 的 FULL 路径。
- 已完成一次 build/install/S=64 smoke：
  - build PASS。
  - install PASS。
  - smoke FAIL，但不再卡死。
- 已明确禁止 `dumpdebug` 和 out-dump 类诊断。
- 已补一条纯对照线：
  - selector 切到 `p8a_baseline`。
  - P3 仅保留单个 `16x16` Neumann 逆写回 GM。
  - P8 继续走原始 GM compose。
  - `S=64` smoke PASS。

## 当前问题

- 完整 fused cube inverse 当前不能通过 `S=64` smoke。
- 失败指标：
  - `CPU vs ChunkGDRNeumann out max_abs=3.430384e-01`
  - `CPU vs ChunkGDRNeumann state max_abs=1.301773e-01`
- 失败特征：
  - state 最坏点 actual 近似 0。
  - 更像 block 丢失、错位、packed layout 错误或后续消费到空块。
  - 不像单纯半精度数值误差。
- 已排除/弱化：
  - FULL path timeout：当前能返回。
  - 最终 GM 写回单独作为主因：此前 scalar row-major 写回实验没有改变失败形态。
  - `CopyPackedL1BlockToL0()` 单点 compact L0 offset：本轮改动后 smoke 指标完全不变。
  - P3 单块逆本身：纯对照线 `p8a_baseline` 已 PASS，说明它不是主因。

## 约束

- 不改 Stage2/Stage3。
- 不改 QK、`gBK/vBeta` 生成和 final RHS。
- final RHS 保持 BF16 GM：
  - `AICProcess(attnWsGm_, gBKWsGm_, outKCumdecayGm_, ...)`
  - `AICProcess(attnWsGm_, vBetaWsGm_, outVInnerGm_, ...)`
- 不回退到 P3/P8 fallback 作为解决方案。
- 不使用 `dumpdebug`。
- 不通过把中间结果写到 `out/attnWsGm_` 的方式调试。
- 当前只验 64-aligned sequence length。

## 下一步

1. 继续静态对照 `ref/fast_strictri_inverse.h` 与 active `op_kernel/fulllocalnew/fast_strictri_inverse.h`。
2. 优先检查 `ComposeTwo32From16()`：
   - 第一次 `RawMmadOddLower16WithEvenDiag()` 是否生成 compact 32x32 `tmp`。
   - 第二次 `RawMmadOddDiag16WithCompactTmp()` 后，两个 16x16 lower block 是否写回到 `(16,0)` 与 `(48,32)`。
   - 当前用 `CopyPackedL1SubBlock()` 从 whole 32x32 scratch 拷回 result，需继续确认 packed source 坐标是否与 L0C/Fixpipe 输出一致。
3. 检查 `Compose64From32()`：
   - 第一段是否为 `tmp = lower @ topInv`，source 为 `(32,0)` 与 `(0,0)`。
   - 第二段是否为 `out = bottomInv @ tmp`，source 为 `(32,32)` 与 tmp `(32,0)`。
   - 最终写回是否只覆盖 result 的 `(32,0)` 32x32 lower-left block。
4. 检查 `CopyL0CMatrixToPackedL1()` / `CopyL0CSubMatrixToPackedL1()`：
   - `srcStride` 是否匹配 L0C matrix size。
   - `dstStride` 是否匹配 packed L1 size。
   - L0C sub-offset 是否与 Fixpipe `CFG_NZ` 语义一致。
5. 用现有 debug mode 只观察最终正常输出形态，定位第一个引入稳定 FAIL 的阶段；不读中间 dump。
6. 每次只改一个数据流，之后执行 build/install/S=64 smoke，再决定下一步。

## 新递进计划

- `aa.txt` 的 5 步递进已落成新的推进计划，旧的 `todo.md` 计划已暂停但保留。
- 当前 `p3l1_p8a` 是第 1/2 步的诊断桥：
  - P3 16x16 对角逆先写入 L1。
  - P8 的对角 operand 改为从 L1 读取。
  - 最终输出链仍保持原 GM 消费，不改 Stage2/Stage3/final RHS。
- 这条桥在 `S=64` 上仍 FAIL，因此下一步不继续在最终 RHS 上打补丁，而是进入更深一级的 L1-resident compose / 32x32 抽取路线。
- 新计划的推进门槛维持为 `S=64 / 256 / 1024` 全部通过后再进入下一步。

## 最新边界

- `p3l1_p8a` 已进一步拆分失败边界：
  - packed L1 -> GM diagonal 写回参与正常链路时，`S=64` 仍 FAIL，且保持旧签名：
    - `out max_abs=3.430384e-01`
    - `state max_abs=1.301773e-01`
  - P3 仍写一份 packed L1 side copy，但 GM diagonal 写回恢复为 compact tile 直接写回、P8 使用 GM direct compose 时，`S=64` PASS：
    - `CPU vs ChunkGDRNeumann out max_abs=3.173694e-03`
    - `CPU vs ChunkGDRNeumann state max_abs=7.139370e-04`
- 当前判断：
  - 主因不是 P3 单块逆，也不是 P8 GM direct compose 或 final RHS。
  - 主因集中在 packed L1 结果回读并写回 GM diagonal 的链路。
  - step 3 还不能算完成；下一步应先修 packed L1 -> compact/GM 的读出语义，再重新启用 32x32 L1-resident compose。

## 18:41 最新边界

- 已修复 packed L1 -> compact/GM 读回语义：
  - 不再用 scalar `GetValue/SetValue` 做 packed/compact L1 中转。
  - 改为 packed source 乘 identity，经 L0C/Fixpipe 写入目标 L1，再写 GM。
- 当前 active 仍是 `p3l1_p8a`。
- 当前通过内容：
  - P3 `16x16` diagonal inverse 结果落到 `aicHalfL1P3Result_`。
  - GM diagonal 正常链路从 `aicHalfL1P3Result_` 写回。
  - P8 compose 仍为已验证 GM direct。
  - Stage2/Stage3/QK/`gBK`/`vBeta`/final RHS 未改。
- build/install PASS。
- `S=64 / 256 / 1024` smoke 均 PASS。
- 最新结论：
  - packed L1 side copy 本身可以作为后续 32x32 L1-resident compose 的输入来源。
  - 旧的 `out max_abs=3.430384e-01` / `state max_abs=1.301773e-01` 签名来自 scalar packed readback 语义。
  - 下一步进入 step 3 时，只允许替换 32x32 compose 的 diagonal/operand 提取与计算路径；64x64 compose 和 final RHS 继续保持不变。

## 18:48 最新边界

- step 3 已通过：
  - `32x32 from 16x16` compose 已切到 L1-resident `P8Compose32From16SingleDirect()`。
  - `64x64 from 32x32` 仍保持 GM direct。
  - final RHS 未改。
- build/install PASS。
- `S=64 / 256 / 1024` smoke 均 PASS。
- 当前判断：
  - P3 packed side copy、packed readback、32x32 L1 compose 均可作为通过路径。
  - 还未完成完整 P8fuse；剩余关键边界是 `64x64 from 32x32` compose。
  - 下一步只切 `P8Compose64From32GmDirect32()` -> `P8Compose64From32Direct32()`，若失败则先正向定位和修复该 direct L1 compose 的布局/同步/Fixpipe 问题；只有明确放弃候选或需要重建干净工作面时，才恢复到本条 step 3 PASS 边界。

## 18:54 最新边界

- step 4 已切到完整 P8fuse compose：
  - `32x32 from 16x16` compose: L1-resident。
  - `64x64 from 32x32` compose: L1-resident。
  - final RHS 未改。
- build/install PASS。
- `S=64 / 256 / 1024` smoke 均 PASS。
- 观察到的风险：
  - `out` max_abs 相对 step 3 明显上升：
    - `S=64`: `3.169131e-02`
    - `S=256`: `6.747369e-02`
    - `S=1024`: `6.747369e-02`
  - `state` max_abs 仍和 step 3 同量级。
- 下一步：
  - 跑当前 64-aligned compare set。
  - 若 compare PASS，step 4 可作为 active 目标通过边界，但文档中必须保留 `out` 误差上升风险。
  - 若 compare FAIL，回退到 step 3 PASS 边界。

## 18:56 最新边界

- 当前 64-aligned compare set 已 PASS：
  - `16 passed in 6.18s`
- 当前 active 通过边界：
  - P3 diagonal inverse -> packed L1 side copy。
  - packed L1 readback -> GM diagonal。
  - `32x32 from 16x16` L1-resident compose。
  - `64x64 from 32x32` L1-resident compose。
  - final RHS 保持 BF16 GM 原路径。
- 剩余风险：
  - step 4 的 `out` 误差明显高于 step 3，但未越过当前 compare 阈值。
  - 非 64-aligned tail cases 仍未恢复。
- 下一步可选：
  - 若继续按 `aa.txt`，进入 step 5：P3 改 P7 布局。
  - 在进入 step 5 前，建议先保留当前 step 4 active 作为可回退通过边界。
