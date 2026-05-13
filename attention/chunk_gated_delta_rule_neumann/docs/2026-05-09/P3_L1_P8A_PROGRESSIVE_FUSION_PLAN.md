# 2026-05-09 P3_L1_P8A Progressive Fusion Plan

## 2026-05-11 覆盖规则

- 后续执行以 `todo.md` 为主；本文件保留历史阶段和已验证边界。
- 精度失败必须在当前目标上正向定位和修复，不能把回滚到 PASS baseline 当成任务完成。
- Packed MMAD 是 P7/P8 的目标，不是冗余；优化对象是重复搬运、copy-out、layout-copy、identity 构造和同步，不是把 packed 计算退回逐 `16x16`。
- `identity64` 目标是第一次有效生成后直接在 L1 复用，但必须先定义 raw-MMAD buffer epoch / `pipe_->Reset()` 生命周期；不能只加一个跨 epoch 的 prepared flag。
- L0C copy-out 若被硬件/API 边界卡住，可以记录为后置项，但不算 PASS，后续仍要单独解决或明确放弃。
- 所有计划完成并通过 gate 后，先保存新 baseline，再用 `msprof` 实测或 simulator/仿真分析寻找新的优化点。

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 2026-05-09 初始状态（历史）

- 当前 selector 已切到 `p3l1_p8a`。
- build/install 已通过。
- `S=64` smoke 仍 FAIL，失败形态和旧版一致：
  - `CPU vs ChunkGDRNeumann out max_abs=3.430384e-01`
  - `CPU vs ChunkGDRNeumann state max_abs=1.301773e-01`
  - state worst actual 仍接近 `0`
- 当前诊断桥已经做到：
  - P3 的 `16x16` 对角逆先落到独立 packed L1 tile。
  - P8 的对角 operand 从该 L1 tile 取。
  - P8 的 lower-left 输入和最终 GM 消费链仍保持原样。

## 2026-05-09 初始结论（历史）

- 仅做 P3->L1 和 P8->L1 的桥接，不足以恢复精度。
- 下一步不能继续动 final RHS，也不能回退到 fallback。
- 接下来要沿 `aa.txt` 的递进顺序继续推进更深一级的 L1-resident compose。

## 递进顺序

1. `P3` 求逆结果放到 `L1` 上，其他不变。
2. `P3` 求逆结果放到 `L1` 上的同时，放在 `64x64` 矩阵的对角块对应位置，参考 `P7`，其他不变。
3. `32x32` 求逆时所有矩阵从 `L1` 提取，提前把 `inv` 矩阵加载到 `L1`，参考 `P8fuse`，其他不变。
4. 完全实现 `P8fuse`。
5. 实现 `P3 -> P7`，调整矩阵结果存放的位置，应与第 2 步一致。

## 执行规则

- 一次只改一个数据流。
- 每一步必须先完成 build -> install -> smoke。
- 只有 `S=64 / 256 / 1024` 都通过后，才进入下一步。
- 保持 `Stage2 / Stage3 / QK / gBK / vBeta / final RHS` 不变。
- 禁止 `dumpdebug`，禁止把中间结果塞进 `out` 或 `attnWsGm_` 作为调试手段。

## 2026-05-09 18:28 更新

- 已确认 `p3l1_p8a` 的最新通过边界：
  - P3 求逆结果复制到 packed L1 side copy。
  - GM diagonal 写回使用 compact tile 直接写回。
  - P8 compose 保持已验证 GM direct。
  - `S=64` smoke PASS。
- 失败边界：
  - 只要正常链路改为从 packed L1 结果回读再写回 GM diagonal，`S=64` 仍恢复旧失败签名：
    - `out max_abs=3.430384e-01`
    - `state max_abs=1.301773e-01`
- 因此第 3 步不能继续直接叠 L1-resident compose；应先修 packed L1 -> compact/GM 读出语义，再重新进入 `32x32` 从 L1 提取的验证。

## 2026-05-09 18:41 更新

- packed L1 -> compact/GM 读出语义已修复：
  - packed/compact L1 block copy 改用 identity MMAD + Fixpipe，而不是 scalar `GetValue/SetValue`。
  - GM diagonal 正常链路已切到从 `aicHalfL1P3Result_` 写回。
  - P8 compose 仍保持 GM direct，final RHS 未改。
- 验证：
  - build PASS。
  - install PASS。
  - `S=64 / 256 / 1024` smoke 均 PASS。
- 现在可重新进入第 3 步：
  - 仅改 `32x32` compose 的 L1 operand 提取/计算。
  - 不同时改 `64x64` compose。
  - 不改 final RHS。

## 2026-05-09 18:48 更新

- 第 3 步已通过：
  - 两处 `32x32 from 16x16` compose 已切到 L1-resident。
  - `64x64 from 32x32` compose 仍保持 GM direct。
  - final RHS 未改。
- 验证：
  - build PASS。
  - install PASS。
  - `S=64 / 256 / 1024` smoke 均 PASS。
- 下一步进入第 4 步：
  - 只切 `64x64 from 32x32` compose 到 L1-resident。
  - 若失败，先定位当前 P8 direct L1 compose 的布局/同步/Fixpipe 问题并正向修复；只有明确放弃该候选或需要重建干净工作面时，才恢复到本条第 3 步 PASS 边界。

## 2026-05-09 18:54 更新

- 第 4 步已完成初验：
  - `64x64 from 32x32` compose 已切到 L1-resident。
  - final RHS 未改。
- 验证：
  - build PASS。
  - install PASS。
  - `S=64 / 256 / 1024` smoke 均 PASS。
- 注意：
  - `out` max_abs 明显高于第 3 步，最高到 `6.747369e-02`。
  - `state` max_abs 仍稳定在 `~1e-03` 内。
- 下一步执行当前 64-aligned compare set，再决定第 4 步是否作为 active 通过边界。

## 2026-05-09 18:56 更新

- 第 4 步已通过当前 compare gate：
  - `16 passed in 6.18s`
- 当前 active 可视为完整 P8fuse compose 通过边界：
  - P3 diagonal result 在 packed L1 / GM diagonal 均可用。
  - 32x32 compose 和 64x64 compose 均为 L1-resident。
  - final RHS 未改。
- 仍需保留的风险：
  - `out` max_abs 高于第 3 步，compare 中最高约 `7.5e-02`。
  - 非 64-aligned tail cases 仍未恢复。
- 后续如果继续第 5 步，应先以本状态为回退点，再单独切 P3/P7 布局。

## 2026-05-10 收口更新

- 后续推进已转入 `todo.md` 和 `docs/2026-05-10/WORK_LOG.md` 记录。
- 所有 planned phases 已完成到 Phase 6b：
  - Phase 5 final RHS fusion 已通过 target correctness gate。
  - Phase 6b group load + two `16x16` P7 solves 已通过 target correctness gate。
  - `S=64 / 256 / 1024` smoke PASS。
  - 当前 64-aligned compare set PASS：`16 passed in 6.51s`。
- 已保存当前 clean baseline：
  - `op_kernel/baseline/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_p3l1_p8a_phase6b_baseline.h`
- 重要边界：
  - Phase 6b 不是性能收益，normal 仍更快。
  - Phase 6b 不是 strict `32x32/64x64` group-MMAD diagonal solver；该 strict diagonal-only raw-MMAD body 仍失败，未保存为 baseline。
