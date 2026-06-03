# Packed64 Diagonal Inverse Progressive Plan

## 2026-05-11 覆盖规则

- 后续执行以 `todo.md` 为主；本文件保留 packed64 推进历史和设计背景。
- 精度失败必须在当前目标上正向定位和修复，不能把回滚到 PASS baseline 当成任务完成。
- Packed MMAD 是 P7/P8 的目标，不是冗余；优化对象是重复搬运、copy-out、layout-copy、identity 构造和同步，不是把 packed 计算退回逐 `16x16`。
- `identity64` 目标是第一次有效生成后直接在 L1 复用，但必须先定义 raw-MMAD buffer epoch / `pipe_->Reset()` 生命周期；不能只加一个跨 epoch 的 prepared flag。
- L0C copy-out 若被硬件/API 边界卡住，可以记录为后置项，但不算 PASS，后续仍要单独解决或明确放弃。
- 所有计划完成并通过 gate 后，先保存新 baseline，再用 `msprof` 实测或 simulator/仿真分析寻找新的优化点。

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 目标

当前 `p3l1_p8a` Phase 6b 已通过 correctness gate，但性能仍慢。下一轮主线是把 diagonal inverse 从 `group load + two 16x16 solves` 推进到真正的 packed `64x64` diagonal solve，并同步减少：

- packed/compact identity-MMAD copy。
- repeated `inv @ I` 初始化。
- compose lower-left 的 GM on-demand load。
- diagonal / compose 结果写回的串行暴露。

只改 Stage1 inverse path：FP16 `L/INV`、diagonal inverse、P8 lower-triangular compose、inverse packed L1 residency、写回时序。不改 Stage2、Stage3、QK、`gBK/vBeta` 生成语义和 final RHS 数学语义。

## 新资源模型

长期必须独立的 L1/TSCM half tile 从 5 块收敛为 3 块：

1. `result64`
   - 初始装入 `INVInitHalfWsGm_` 的四个 diagonal `16x16` block。
   - diagonal solve 后仍作为最新 inverse 的 L1 版本。
   - compose 后成为最终 packed `64x64` lower-triangular inverse。
   - 后续 BF16 `attnWsGm_` 写回、resident L1 copy、fused RHS 都消费它。
2. `diagPower64`
   - 初始装入 `L_powHalfWsGm_` 的四个 diagonal `16x16` power block。
   - 可 in-place 更新为 `L^2/L^4/L^8`，前提是 LoadData/MMAD 已完成后再 Fixpipe 覆盖。
3. `lowerWork64`
   - diagonal solve 阶段可作为保守 scratch 或调试 scratch。
   - compose 阶段覆盖为 lower-left 输入：
     - `(16,0)` for top `16+16 -> 32`
     - `(48,32)` for bottom `16+16 -> 32`
     - `(32,0)` for `32+32 -> 64`

`diagInv64` 不再单独存在，直接是 `result64` 的 diagonal 区域。`diagScratch64` 不是必需长期 buffer，只作为保守/debug 选项。

## L0C 分区模型

`aicL0CBuff_` 当前为 `64KB`。一个 `64x64 FP32 C` 为 `16KB`，理论上可划分 4 个逻辑 C 分区：

- `cInv`: 保存 inverse 累积 C。
- `cPower`: 保存 `power @ power` 输出。
- `cComposeTmp`: 保存 compose 的 `tmp = lower @ topInv`。
- `cFree` 或 compose output 备用。

Phase 初期只要求双 L0C：

- `cInv`：第一次 `result64 @ I` 初始化 C，后续 `result64 @ power` accumulate 到同一 C。
- `cPower`：计算 `diagPower64 @ diagPower64`，再 Fixpipe 回 `diagPower64`。

注意：即使 `cInv` 在 L0C 中持续累积，每轮仍需要把 `cInv` Fixpipe 回 `result64`，因为下一轮 MMAD 的 A 操作数必须从 L1/L0A 来，不能直接把 L0C 当 A 输入。

## Phase 0: 静态资源与别名确认

不改数学，只确认资源安排。

任务：

1. 明确当前 L1 分配：
   - `result64` 复用 `aicHalfL1P3Result_`。
   - `diagPower64` 复用 `aicHalfL1TilePower_`。
   - `lowerWork64` 优先复用 `aicHalfL1Scratch_`。
   - `aicHalfL1TileInv_` 暂时保留为 fallback/debug compact work，不作为新主线长期结果。
2. 确认 `AIC_LOW_L1_BYTES` 不需要扩容；不动 `residentInv[MAX_PARALLEL_NUM]`，避免影响 Phase 5 fused RHS。
3. 设计 L0C offset：
   - `cInv = aicL0C_[0 * 64 * 64]`
   - `cPower = aicL0C_[1 * 64 * 64]`
   - `cComposeTmp = aicL0C_[2 * 64 * 64]`
   - `cFree = aicL0C_[3 * 64 * 64]`
4. 静态检查所有 `Mmad()` / `Fixpipe()` 是否能接受带 offset 的 `LocalTensor<float>`。

验收：只 build，不跑性能。若只做文档/静态检查，不进入代码。

## Phase 1: P7-style packed64 初始化和 GM 定点搬入

目的：先把 Stage1 inverse 入口改成 P7 风格的 `64x64` packed 容器：初始化全 0，再从 GM 把需要的 block 搬到 L1 对应位置。本阶段不碰求解算法。

实现：

1. 先把 `result64 / diagPower64 / lowerWork64` 初始化全 0，保证 upper、未用 lower 和 padding 都是确定值。
2. `L_powHalfWsGm_ -> diagPower64` 的 `(0,0)/(16,16)/(32,32)/(48,48)`。
3. `INVInitHalfWsGm_ -> result64` 的 `(0,0)/(16,16)/(32,32)/(48,48)`。
4. 仍按 proven `16x16` solve 跑四个 diagonal block：
   - 从 `result64/diagPower64` 对应 diagonal 子块取出。
   - 调用旧 `16x16` direct solve。
   - 写回 `result64` diagonal 位置。
5. 本阶段只验证 P7-style `64x64` pack 的 GM->L1 定点搬入和零初始化，不启用 packed64 one-shot 求逆。

验收：

1. build/install。
2. smoke: `S=64 -> S=256 -> S=1024`。
3. compare set。
4. compare PASS 后跑 performance，作为 Phase 1 新基线。

## Phase 2: lower-left 输入单独搬入 lowerWork64

目的：compose 输入从 result 和 lower-left 分离，后续 diagonal solve / result 写回不会覆盖 lower-left。

实现：

1. diagonal solve 之后或之前均可搬运 lower-left；若 `lowerWork64` 在 diagonal solve 阶段用作 scratch，则必须 diagonal solve 完成后搬。
2. `INVInitHalfWsGm_ -> lowerWork64`：
   - `(16,0)` 16x16
   - `(48,32)` 16x16
   - `(32,0)` 32x32
3. P8 compose helper 改为：
   - `topInv/bottomInv` 从 `result64` 读。
   - `lower` 从 `lowerWork64` 读。
   - output 写回 `result64`。

验收同 Phase 1。若失败，优先查 `lowerWork64` packed offset 和 `CopyPackedHalfL1BlockToL0()` source base。

## Phase 3: packed64 diagonal-only solve

目的：把四个 `16x16` 逐次求解替换为 packed `64x64` raw-MMAD。由于 Phase 1 已把 packed64 容器初始化全 0，并只填四个 diagonal `16x16` block，所以 L1 中已经是 block-diagonal 矩阵；本阶段不再抽取 diagonal fractals，而是直接 full-load `64x64` packed tile 到 L0A/L0B。

第一版保持保守，不做 L0C 累积优化：

1. 输入：
   - `result64` diagonal 区域是 inv。
   - `diagPower64` diagonal 区域是 power。
   - 两者都来自 Phase 1 的全 0 packed64 容器和 GM 定点搬入。
2. `power = power @ power`：
   - `diagPower64` full packed64 L1->L0A/L0B。
   - MMAD 到 `cPower`。
   - Fixpipe 回 `diagPower64`。
3. `inv = inv + inv @ power`：
   - 仍可沿用当前安全模式：`inv @ I` initC + `inv @ power` accumulate。
   - Fixpipe 回 `result64`。
4. 重复 3 轮。

验收：

- 必须单独通过 `S=64/256/1024` smoke 和 compare。
- 若失败，不能保留为 active path；记录第一个失败 shape，重点查 off-diagonal/padding 是否全 0、L1->L0 full packed placement、Fixpipe `CFG_NZ` offset。

## Phase 4: L0C 累积，减少 repeated inv @ I

目的：在 Phase 3 正确后，再减少每轮 `inv @ I`。

实现：

1. 第一次：
   - `cInv = result64 @ I`，`initC=true`。
2. 每轮 power：
   - `cPower = diagPower64 @ diagPower64`。
   - Fixpipe `cPower -> diagPower64`，得到下一阶 power。
3. 每轮 inv add：
   - `cInv += result64 @ diagPower64`，`initC=false`。
   - Fixpipe `cInv -> result64`，给下一轮作为 A 操作数。
4. 不再每轮执行 `result64 @ I`。

注意：

- `cInv` 和 `cPower` 必须是不同 L0C 分区。
- 每轮仍要 `cInv -> result64`，否则下一轮 MMAD 取不到最新 inv 的 L1 A 输入。
- 若双 L0C PASS，再评估是否需要 4 分区减少 compose 中间 C 冲突。

验收同 Phase 3，且 compare PASS 后必须跑 performance，看 `aic_mac_time_us`、`aic_fixpipe_time_us`、`aic_scalar_time_us` 是否实际下降。

## Phase 5: packed64 求逆内部搬运/计算并行

目的：在 packed64 correctness PASS 后，再引入搬运/计算重叠。该阶段只做性能流水，不改数学。

候选改造：

1. L0A/L0B ping-pong：
   - 当前计算使用 L0 half ping/pong buffer。
   - 将下一次 `result64/diagPower64 -> L0A/L0B` 搬运与当前 MMAD 重叠。
   - 事件模型按 `M_MTE1`、`MTE1_M` 分离 L0 buffer 释放和数据就绪。
2. power square 与 inv accumulate 解耦：
   - `cPower -> diagPower64` 的 Fixpipe 完成后，下一次 power 的 L1->L0 搬运可以提前。
   - `cInv -> result64` 的 Fixpipe 和下一轮 `diagPower64 -> L0` 要避免写读冲突。
3. lowerWork64 preload 掩盖：
   - 若 `lowerWork64` 在 diagonal solve 阶段不用作 scratch，可在 diagonal MMAD/Fixpipe 间隙预取 `(16,0)/(48,32)/(32,0)`。
   - 若 `lowerWork64` 被用作 scratch，则只能在 diagonal solve 完成后 preload。
4. Fixpipe 与后续 GM/L1 搬运解耦：
   - 只在确认目标 tile 后续不被立即读时做 overlap。
   - 不允许为了 overlap 引入额外 GM fallback。

验收：

- 必须以 Phase 3/4 correctness PASS 版本为起点。
- build/install，`S=64/256/1024`，compare。
- compare PASS 后跑 performance。
- profile 重点看：
  - `aic_mte1_time_us`
  - `aic_fixpipe_time_us`
  - `aic_scalar_time_us`
  - 总时延

## Phase 6: compose output 直接落回 result64

目的：去掉 compose 中 compact output -> packed result 的 identity-MMAD copy。

实现顺序：

1. `16+16 -> 32`：
   - `tmp = lowerWork64[(16,0)] @ result64[(0,0)]`
   - `out = result64[(16,16)] @ tmp`
   - Fixpipe 直接到 `result64[(16,0)]`
2. bottom `16+16 -> 32` 同理写 `result64[(48,32)]`。
3. `32+32 -> 64`：
   - lower 从 `lowerWork64[(32,0)]` 读。
   - output Fixpipe 直接到 `result64[(32,0)]`。
4. 保留 half GM scratch writeback 一版用于对照；PASS 后再删。

验收同 Phase 3。若失败，重点查 `CFG_NZ` offset、dstStride、full/compact NZ layout。

## Phase 7: 写回与 compose/RHS 减少暴露

目的：降低最终写回的串行成本。

候选：

1. 在 compose 使用不到某些 diagonal 后，提前写 `StoreP3DiagonalResultsToGm()`。
2. 分离 BF16 `attnWsGm_` 写回和 resident L1 copy。
3. 若 final RHS 已完全消费 `result64/residentInv`，评估是否能按语义减少 BF16 GM 写回范围或延后写回。

约束：

- 这一步必须看 profile，不能只看代码少了 copy。
- 指标关注：`aic_fixpipe_time_us`、`aic_mte2_time_us`、`aic_scalar_time_us`、总时延。

## 验证门槛

每个 phase 单独验证，不叠加失败实现：

```bash
bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16
../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp
timeout 90s env CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer python3 tests/pytest/compare_chunk_gdr_three_way.py --seed 0 --batch-size 1 --seqlen 64 --nk 4 --nv 4 --dk 64 --dv 64
timeout 90s env CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer python3 tests/pytest/compare_chunk_gdr_three_way.py --seed 0 --batch-size 1 --seqlen 256 --nk 4 --nv 4 --dk 64 --dv 64
timeout 90s env CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer python3 tests/pytest/compare_chunk_gdr_three_way.py --seed 0 --batch-size 1 --seqlen 1024 --nk 4 --nv 4 --dk 64 --dv 64
CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer bash tests/pytest/test_run.sh compare
```

只有 compare PASS 后再跑 performance。
