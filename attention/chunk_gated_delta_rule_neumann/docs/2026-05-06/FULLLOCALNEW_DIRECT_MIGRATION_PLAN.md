# 2026-05-06 fulllocalnew 直接搬运迁移计划

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 目标

将 `attention/chunk_gated_delta_rule` 当前新实现直接迁移到
`attention/chunk_gated_delta_rule_neumann`，然后在新代码基础上实现两个 FP16 Neumann
fulllocal 方案：

- `fulllocalnew/C16_FP16`
- `fulllocalnew/C64_1616_FP16`

核心原则：

- 先搬新 normal 的 BF16 主链路，不在旧 Stage1 上继续补丁式改造。
- 新 Stage1 方案只放入 `op_kernel/fulllocalnew/`。
- Neumann 矩阵展开求逆必须使用 FP16 cube 路径。
- 不在算子输入输出头尾做全局 BF16/FP16 转换，只在 Stage1 inverse 局部桥接。
- final RHS 先不 resident/raw 化，使用新代码 BF16 Matmul 消费最终 inverse。

## 直接搬运范围

`git diff -- attention/chunk_gated_delta_rule` 当前显示 normal 新代码改动为 6 个文件：

1. `op_host/chunk_gated_delta_rule_tiling.cpp`
2. `op_kernel/chunk_gated_delta_rule.h`
3. `op_kernel/chunk_gated_delta_rule_stage1.h`
4. `op_kernel/chunk_gated_delta_rule_stage2.h`
5. `op_kernel/chunk_gated_delta_rule_stage3.h`
6. `op_kernel/chunk_gated_delta_rule_utils.h`

Neumann 对应迁移落点：

1. `op_host/chunk_gated_delta_rule_neumann_tiling.cpp`
2. `op_kernel/chunk_gated_delta_rule_neumann.h`
3. `op_kernel/chunk_gated_delta_rule_neumann_stage2.h`
4. `op_kernel/chunk_gated_delta_rule_neumann_stage3.h`
5. `op_kernel/chunk_gated_delta_rule_neumann_utils.h`
6. `op_kernel/fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocalnew.h`
7. `op_kernel/fulllocalnew/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocalnew.h`

不迁移到 `baseline/`、`halflocal/`、现有 `fulllocal/`，避免污染已验证版本。

## 新代码主链路变化

normal 新实现需要整体搬到 Neumann，否则 Stage1 与 Stage2/Stage3 tensor 类型会不匹配：

- `stageOneParaNum` 从 `2` 改为 `4`，Stage1 每核并行处理多个 chunk。
- Stage1/Stage2/Stage3 主 Matmul 改成 BF16/BF16/BF16。
- 中间 workspace 的 `kCumDecay`、`vInner`、`qPrime`、`kg`、`qkt` 改为 BF16。
- Stage2 直接以 BF16 `initState/finalState` 递推，不再走单独 FP32 `highState`。
- Stage3 直接输出 BF16，不再通过 FP32 `attnInter` 后 cast。

## FP16 inverse 转换方案

不做算子头尾全局转换。只在 Stage1 inverse 内局部处理：

1. AIC 用新 BF16 Matmul 计算 `kkWsGm_ = key @ key.T`，`kkWsGm_` 为 BF16 GM。
2. AIV 读取 BF16 `kkWsGm_` 到 UB，cast 到 float，乘 `beta/gamma/mask` 得到 float `L`。
3. AIV 写 FP32 inverse scratch：
   - `L_powWsGm_ = L`
   - `INVWsGm_ = I - L`
4. AIC 将 FP32 scratch 转为 half L1/TSCM：
   - `FP32 GM @ I -> L0C float -> Fixpipe half NZ -> half L1/TSCM`
   - 这个路径同时完成 dtype cast 和 cube 可消费的 NZ layout 生成。
5. AIC 在 half L1/TSCM + L0A/L0B/L0C 内做 FP16 Neumann。
6. AIC 将最终 inverse 写回 BF16 `attnWsGm_`。
7. final RHS 使用新 BF16 Matmul：
   - `attnWsGm_ @ gBKWsGm_ -> outKCumDecay`
   - `attnWsGm_ @ vBetaWsGm_ -> outVInner`

这意味着 FP32 GM scratch 只是 inverse 局部桥接，不代表级数展开仍用 FP32 Matmul。

## Workspace 与 tiling 要点

- host `stageOneParaNum` 同步改为 `4`。
- host `matmulTilingFp32` 字段名可暂时保留，但主 tiling 类型应与新 normal 一致：BF16/BF16/BF16。
- Stage1 主 workspace 按 BF16 解释：
  - `gBKWsGm_`
  - `queryConGm_`
  - `keyConGm_`
  - `kkWsGm_`
  - `attnWsGm_`
  - `vBetaWsGm_`
- Neumann scratch 需要按 `coreNum * paraNum * c * c` 分配：
  - `L_powWsGm_`
  - `tmpWsGm_`
  - `INVWsGm_`
- parallel chunk 的 scratch offset 必须用 `i * ccOffset_`，否则 `stageOneParaNum=4` 会覆盖 workspace。
- `chunk_gated_delta_rule_neumann_tiling_data.h` 先保留 restored superset；正式单方案放大性能前再裁剪无关 `TilingData/MatmulTiling` 字段。

## 实施步骤

### Step 1：建立 fulllocalnew 目录

- 新建 `op_kernel/fulllocalnew/`。
- 新增两个 Stage1 文件：
  - `chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocalnew.h`
  - `chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocalnew.h`
- selector 暂不切换，先保证文件落点清晰。

### Step 2：迁移 normal 新 BF16 主链路

- 将 normal 新 `chunk_gated_delta_rule.h` 的 workspace/state 递推逻辑迁移到 Neumann wrapper。
- 将 normal 新 Stage2/Stage3 迁移到 Neumann Stage2/Stage3。
- 将 normal 新 utils 增量迁移到 Neumann utils。
- 将 normal host tiling 的 `stageOneParaNum=4` 和 BF16 main matmul tiling 迁移到 Neumann host tiling。
- 保留 Neumann 额外 FP16/L1 tiling 字段和 Matmul object，供 `fulllocalnew` 使用。

验收：构建通过。

### Step 3：实现 C16_FP16 fulllocalnew

- 以 normal 新 Stage1 为底座。
- 保留新 `ParaChunkAIC/ParaChunkAIV` 4 路 parallel chunk 结构。
- AIV `KKBetaCompute + InverseCompute` 改为生成 FP32 `L_powWsGm_/INVWsGm_` scratch。
- AIC 在 `CrossCoreWaitFlag(0x7)` 后调用 `C16InverseBy16x16Neumann(i)`。
- 复用当前 `fulllocal/C16_FP16` 已验证的 half L1/TSCM resident inverse helper，offset 改为 parallel chunk aware。
- 最终 inverse 写回 BF16 `attnWsGm_[i * ccOffset_]`。
- final RHS 继续走新 BF16 `AICProcess(attnWsGm_, gBKWsGm_/vBetaWsGm_)`。

验收：

```bash
bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16
bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp
CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare
```

通过标准：`20 passed`。

### Step 4：实现 C64_1616_FP16 fulllocalnew

- 以 C16 通过版本为基础。
- selector chunk size 切为 `64`。
- AIV 为完整 64x64 生成 FP32 `L_powWsGm_/INVWsGm_` scratch。
- AIC 对 64x64 拆 16x16 子块：
  - diagonal 16x16 block 走 FP16 resident Neumann。
  - lower block compose 先复用当前 `fulllocal/C64_1616_FP16` 已验证顺序。
- 最终 64x64 inverse 写回 BF16 `attnWsGm_`。
- final RHS 继续新 BF16 Matmul。

验收：

```bash
bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16
bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp
CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare
```

通过标准：`20 passed`。

### Step 5：简单性能 smoke

精度通过后跑简单 smoke：

```bash
python3 tests/pytest/run_chunk_gdr_performance_msprof.py \
  --output tests/pytest/prof_out/fulllocalnew_c16_c64_1616_smoke_20260506 \
  --case prefill_b8_s16_d64_h16_hkv4 \
  --case prefill_b8_s64_d64_h16_hkv4 \
  --ops both
```

未裁剪无关 `TilingData/MatmulTiling` 前，结果只作为 smoke，不作为正式性能结论。

## 风险

- BF16 主链路迁移会触碰 wrapper、Stage2、Stage3，不能只改 Stage1。
- `stageOneParaNum=4` 后 scratch offset 是高风险点。
- 最终 inverse 写回 BF16 `attnWsGm_` 的 raw/Fixpipe helper 需要单独验证。
- C64_1616 compose 先保证精度，full matrix resident 和 fast inverse packed 优化后置。
