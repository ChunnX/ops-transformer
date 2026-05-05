# 2026-05-05 Fulllocal 单点突破计划

## 背景

当前已完成三套 stage1 目录拆分：

- `baseline`：AIV 只构造初始 `L` / `I-L`，后续 Neumann 轮次用 AIC Matmul/Fixpipe atomic add。
- `halflocal`：`old_INV + INV@P` 在 L0C/CMatrix 内完成，随后普通 Fixpipe 写回 GM，下一轮仍从 GM 读回。
- `fulllocal`：目录和候选文件已创建，但尚未作为完整 suite 通过精度。

当前 selector 停在 `halflocal/C64_3232_FP32`，这是本轮 baseline/halflocal 性能测试中的最佳 Neumann 版本。`baseline` 和 `halflocal` 的 `3*2` variant 均已构建、安装、三方精度通过，后续作为回退和对照基线，不再在 fulllocal 调试中修改。

## 单点选择

首个单点选择 `fulllocal/C64_3232_FP16`，必要时用 `C64_3232_FP16_localtscm_fulllocal` 作为隔离 PoC 文件。

选择原因：

- `C64_3232` 是当前 halflocal 最优方向，`S=16/64/125/2048` ratio 为 `1.305x / 1.651x / 1.873x / 2.332x`。
- 当前 DAV2201 上 `float L0C -> float L1/TSCM` 不支持，fulllocal resident 可行方向应先验证 `float L0C -> half/bfloat16_t L1 NZ -> LoadData -> L0A/L0B`。
- `C64_1616` 当前性能极差，且 power/compose 仍含 MatmulImpl fallback 和大量 16x16 tiny 调度，不适合作为第一突破点。

## 非目标

- 不先扩展 `3*2` fulllocal 矩阵；单点精度不过前不复制实现。
- 不修改 `baseline` / `halflocal` 文件。
- 不把 final RHS `INV @ k_cumdecay` / `INV @ v_beta` 纳入本轮 resident 验收，仍保持 FP32 GM `MatmulImpl` 稳定消费端。
- 不把未通过精度的 resident 分支切到默认 selector。

## 实施步骤

### Step 1：16x16 resident 布局探针

目标是独立验证 `L0C -> Fixpipe CFG_NZ -> L1/TSCM -> LoadData -> MMAD` 的数值闭环，不接入完整 Neumann。

动作：

- 用固定 16x16 tile 做 `A @ B -> L0C`。
- 用 `Fixpipe<half/bfloat16_t, float, CFG_NZ>` 写入 L1/TSCM。
- 分别按 A 输入和 B 输入语义 `LoadData` 到 L0A/L0B。
- 做 `resident @ I`、`I @ resident` 或等价 MMAD 写回 GM，与 GM baseline 对比。

验收：

- A/B 两条 LoadData 路径分别通过。
- 确认 `ifTranspose`、stride、compact NZ offset 不混用。

### Step 2：接入单 16x16 Neumann

目标是只覆盖 `validLen <= 16` 的最小 Neumann 路径。

动作：

- 在 `C64_3232_FP16` fulllocal 文件中接入 16x16 resident path。
- 只跑 `S=16` smoke 和少量 tail case。
- final RHS 仍从 GM 侧读取稳定结果，避免同时调试两条链路。

验收：

- `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare` 的单 case smoke 通过。
- 若出现 `1e-1` 级误差，优先回到 Step 1 排查 layout，不先改 Neumann 公式。

### Step 3：扩到 32x32 compose

目标是验证两个 16x16 对角块和 lower-left block 的 compact NZ offset。

动作：

- 覆盖 `S=17/33/64` smoke。
- 分别验证 `Compose32From16()` 的 lower-left block 读写位置。
- 继续保留 GM readback debug 开关，便于对比 resident 与 halflocal 中间矩阵。

验收：

- `S=17/33/64` smoke 精度通过。
- lower-left compose 的 readback 与 halflocal GM baseline 一致到可接受精度。

### Step 4：C64_3232 fulllocal 全路径

目标是完成 `C64_3232_FP16` 单点 fulllocal 的完整 Neumann inverse resident。

动作：

- 接入 64 内两个 32 block 和 `Compose64From32()`。
- 跑全量三方 compare。
- compare 通过后再跑 `S=16/64/125/2048` 性能。

验收：

- 全量 compare `20 passed`。
- 性能至少优于当前 `halflocal/C64_3232_FP16` 或明确定位剩余瓶颈。

### Step 5：再扩展到 3*2

只有 `C64_3232_FP16` 单点通过后，再决定是否复制到：

- `C64_3232_FP32`：受限于 FP32 L1 写回能力，默认不作为当前架构首选。
- `C16_FP16/FP32`：用于验证短块 resident 收益。
- `C64_1616_FP16/FP32`：放在最后，除非先去掉 power/compose MatmulImpl fallback 和过量同步。

## 风险与回退

- `float L0C -> half/bfloat16 L1` 会引入量化误差，必须用精度结果决定是否可接受。
- `CFG_NZ` 是 C/output 侧 layout，作为 A/B 输入时需要分别验证 L0A/L0B load 参数。
- `C64_3232_FP16_LocalTscm` 历史 PoC 曾出现 `17 failed, 3 passed`，不能直接视为可用实现。
- 回退方式固定为 selector 切回 `halflocal/C64_3232_FP32`。
