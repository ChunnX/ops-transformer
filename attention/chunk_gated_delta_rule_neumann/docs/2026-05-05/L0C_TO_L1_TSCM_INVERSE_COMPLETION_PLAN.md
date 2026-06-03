# 2026-05-05 完成算子计划：L0C -> L1/TSCM 求逆闭环

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 目标

把当前默认 `C64_3232_FP32_CVInverse` 从“CMatrix/MMAD 累加后写回 GM”推进到“CMatrix/MMAD 累加后写到 L1/TSCM，再由下一轮 LoadData 消费”的最小正确闭环。

当前已经完成：

- `INV += INV@P` 在 L0C/CMatrix 内通过 `cmatrixInitVal=false` 累加。
- lower-triangular power 保持精度正确。
- 默认线全量三方精度通过：`20 passed in 6.14s`。

尚未完成：

- updated `INV` 跨 round 仍走 `L0C -> Fixpipe -> GM -> LoadData -> L0A/B`。
- final RHS `INV @ k_cumdecay` / `INV @ v_beta` 仍走 FP32 GM `MatmulImpl`，今日明确不改。

## 今日交付口径

今日交付不承诺一次性完成全算子最终性能超越 normal；final RHS 先不改，今日目标只覆盖 Neumann inverse 矩阵生成片上闭环的正确性门槛：

1. 建立 L1/TSCM tile 数值探针。
2. 打通单个 16x16 updated `INV` 的 `L0C -> Fixpipe -> L1/TSCM -> LoadData -> MMAD` 回读验证。
3. 通过后再扩到 32x32 compose。
4. 只有全量 compare 通过后才跑 msprof smoke。

## 执行步骤

### Step 1：冻结当前稳定基线

- 保持 selector：`C64_3232_FP32_CVInverse`。
- 保持当前 CMatrix 累加实现不变。
- 不再启用 full-tile power、`INV@P` k-range clipping、标量 AddLocalMatrix 默认线。
- 记录当前全量 compare 结果作为回退基线。

验收：无需新增代码，确认 build/compare 已有 `20 passed in 6.14s`。

### Step 2：新增 L1/TSCM 最小探针，不改默认路径

目标：只验证 layout，不影响默认精度线。

实现建议：

- 新增 debug/PoC helper，单独分支启用，不作为默认路径。
- 只处理一个 16x16 tile。
- 用已知矩阵或现有 `INV` tile：
  - L0C 计算 `INV@I + INV@P`。
  - Fixpipe 写到 L1/TSCM buffer。
  - 下一次 `LoadData` 从 L1/TSCM 读回 L0A/L0B。
  - 做一次 `readback @ I` 或 `readback @ zero/identity` 写 GM，与 GM baseline 比较。

验收：单 16x16 deterministic smoke PASS。

### Step 3：把 16x16 Neumann round 的 ping-pong buffer 改成 L1/TSCM PoC

目标：只替换 16x16 block 内 updated `INV` 的中转位置。

方案：

- 为 `INV` 和 `power` 准备两个 L1/TSCM ping-pong tile buffer。
- Round 内：
  - `power` 仍可先保持 GM，降低变量数；先只替换 updated `INV` 回读链路。
  - updated `INV` 从 L0C/Fixpipe 写 L1/TSCM。
  - 下一次 `INV@P` 从 L1/TSCM LoadData 到 L0A。
- 如果 16x16 通过，再考虑 power 也 resident。

验收：`S=16/17` Neumann-only smoke PASS。

### Step 4：扩展到 32x32 compose

目标：验证 `Compose32From16()` 的 lower-left block 在 L1/TSCM compact layout 下 offset 正确。

执行：

- 先只让两个 16x16 对角块 resident，compose 仍 GM baseline。
- 再把 `Compose32From16()` 的中间结果改到 L1/TSCM。
- 最后验证 `validLen=33/49/64`。

验收：`S=16/17/33/49/64` smoke PASS。

### Step 5：全量 compare 与性能 smoke

前提：Step 2-4 全部 PASS。

执行：

- 构建安装：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `bash ../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 精度：
  - `source /home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer/bin/set_env.bash && CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
- 性能 smoke：
  - 只跑 `B=8,S=16` 和 `B=8,S=64`。

验收：全量 compare 必须 20/20 PASS；性能只作为趋势，不作为今日唯一交付。

## 风险与回退

- 最大风险：Fixpipe 到 L1/TSCM 的格式不是后续 `LoadData` 期望的格式。
- 第二风险：32x32/64x64 compose 子块 offset 在 compact NZ 下错位。
- 第三风险：L1/TSCM 空间与 MatmulImpl/raw local buffers 冲突，需要 pipe reset 或独立 buffer 管理。
- 回退方式：selector 保持当前稳定文件，PoC 分支用 compile-time flag 控制；精度失败立即回到 GM baseline。

## 今日不做

- 不修改 final RHS：`INV @ k_cumdecay` 和 `INV @ v_beta` 都保持 FP32 GM `MatmulImpl`。
- 不做 full prefill 32 shape msprof，除非全量 compare 已通过。
- 不推送远端分支。

## 用户最新约束

- final RHS 可以先不用改；今日实现范围只覆盖 Neumann inverse 矩阵生成链路。
- `INV @ k_cumdecay` / `INV @ v_beta` 保持现有 FP32 GM `MatmulImpl`，作为稳定消费端。
