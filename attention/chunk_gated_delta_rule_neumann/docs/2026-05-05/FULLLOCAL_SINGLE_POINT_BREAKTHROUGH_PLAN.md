# 2026-05-05 Fulllocal 单点突破计划

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 背景

当前已完成三套 stage1 目录拆分：

- `baseline`：AIV 只构造初始 `L` / `I-L`，后续 Neumann 轮次用 AIC Matmul/Fixpipe atomic add。
- `halflocal`：`old_INV + INV@P` 在 L0C/CMatrix 内完成，随后普通 Fixpipe 写回 GM，下一轮仍从 GM 读回。
- `fulllocal`：单点文件已创建，2026-05-06 已完成 `C64_3232_FP16` 精度闭环和指定性能点测试；随后按用户要求追加迁移到 `C16_FP16` 看收益。两个 fulllocal 口径性能仍慢于 normal。

计划创建时 selector 停在 `halflocal/C64_3232_FP32`，这是本轮 baseline/halflocal 性能测试中的最佳 Neumann 版本。2026-05-06 先切到 `fulllocal/C64_3232_FP16` 完成单点闭环，随后最新 active selector 已切到 `fulllocal/C16_FP16`。`baseline` 和 `halflocal` 的 `3*2` variant 均已构建、安装、三方精度通过，后续作为回退和对照基线，不再在 fulllocal 调试中修改。

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

## 2026-05-06 执行结果

- 当前 selector：
  - `op_kernel/selected/stage1_selected.h` include `../fulllocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP16_fulllocal.h`
  - `CGDR_NEUMANN_SELECTED_SUITE="fulllocal"`
  - `CGDR_NEUMANN_SELECTED_VARIANT="C64_3232_FP16"`
  - `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`
- 已恢复 `Compose32From16()` / `Compose64From32()` resident 分支中的 scalar full-L1 backfill；删除该 backfill 会导致全量 compare 退化。
- 构建安装通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 精度结果：
  - `S=16` smoke PASS。
  - `S=17` smoke PASS。
  - `CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare` 最终 `20 passed in 6.17s`。
- 性能结果：
  - CSV: `tests/pytest/prof_out/fulllocal_C64_3232_FP16_selected_s_20260506/performance_compare.csv`
  - Neumann/normal task ratio: `S=16 1.307x`、`S=64 1.957x`、`S=125 2.286x`、`S=2048 3.011x`。
- 结论：
  - `fulllocal/C64_3232_FP16` 单点已完成精度和性能闭环。
  - 性能没有突破 normal，也没有超过当前最佳 halflocal 口径。
  - direct 32x32 resident 路线曾在 `S=17` 出现大误差，`LoadData2DParamsV2` 尝试出现 `nan`，当前不作为主线。

## 风险与回退

- `float L0C -> half/bfloat16 L1` 会引入量化误差，必须用精度结果决定是否可接受。
- `CFG_NZ` 是 C/output 侧 layout，作为 A/B 输入时需要分别验证 L0A/L0B load 参数。
- `C64_3232_FP16_LocalTscm` 历史 PoC 曾出现 `17 failed, 3 passed`，不能直接视为可用实现。
- 回退方式固定为 selector 切回 `halflocal/C64_3232_FP32`。这里的 selector 是编译期入口，即 `op_kernel/selected/stage1_selected.h` 和 `stage1_selected_config.h`；回退后必须重新 build/install/compare，不是运行时动态切换。

## 2026-05-06 追加：C16 fulllocal 迁移

用户追加要求：先迁移到 `C16` 的 Stage1 实现看收益，`finalRHS` 暂不处理。

当前 selector 已更新为：

- `op_kernel/selected/stage1_selected.h` include `../fulllocal/chunk_gated_delta_rule_neumann_stage1_C16_FP16_fulllocal.h`
- `CGDR_NEUMANN_SELECTED_SUITE="fulllocal"`
- `CGDR_NEUMANN_SELECTED_VARIANT="C16_FP16"`
- `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=16`

实现边界：

- `Neumann16InverseBlock()` 复用 C64 fulllocal 已通过的 half L1/TSCM resident inverse 路径。
- `FP16_L1_RESIDENT_INVERSE=true`，`FP32_SCALAR_RESIDENT_INVERSE=false`。
- C16 没有 `Compose32From16()` / `Compose64From32()`，因此当前只验证 16x16 inverse resident 本身。
- `final RHS` 暂不 resident，仍通过 `AICProcess(INVWsGm_, gBKWsGm_, ...)` 和 `AICProcess(INVWsGm_, vBetaWsGm_, ...)` 从 GM 消费 inverse。

验证结果：

- 构建安装通过。
- `CGDR_GOLDEN_CHUNK_SIZE=16 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`：`20 passed in 5.98s`。
- 性能 CSV：
  - `tests/pytest/prof_out/stage1_fulllocal_C16_FP16_s_selected_20260506/performance_compare.csv`

四点性能：

| case | Neumann/normal | fulllocal/halflocal |
|---|---:|---:|
| `B=8,S=16,D=64,Hq=16,Hkv=4` | `1.084x` | `1.000x` |
| `B=8,S=64,D=64,Hq=16,Hkv=4` | `2.140x` | `1.007x` |
| `B=8,S=125,D=64,Hq=16,Hkv=4` | `2.503x` | `1.011x` |
| `B=8,S=2048,D=128,Hq=16,Hkv=4` | `4.076x` | `0.996x` |

结论：

- `C16_FP16` fulllocal inverse resident 已完成精度闭环。
- 性能相比 `C16_FP16` halflocal 基本持平，没有明显收益。
- 当前瓶颈边界很清楚：inverse resident 后仍写回 GM，`final RHS` 又从 GM 读取；若不把 RHS 消费端也接到 resident `INV`，当前 C16 版本很难产生实质性性能突破。

## 2026-05-06 追加：只保留 FP16 C16/C64_1616 fulllocal

用户后续将当前目标收窄为 FP16 `C16` 与 FP16 `C64_1616` 两条 fulllocal，`final RHS` 继续不处理。

当前 active selector：

- `op_kernel/selected/stage1_selected.h` include `../fulllocal/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_fulllocal.h`
- `CGDR_NEUMANN_SELECTED_SUITE="fulllocal"`
- `CGDR_NEUMANN_SELECTED_VARIANT="C64_1616_FP16"`
- `CGDR_NEUMANN_SELECTED_CHUNK_SIZE=64`

当前实现边界：

- `C16_FP16`：只做 16x16 inverse resident，final RHS 仍从 GM 消费 `INVWsGm_`。
- `C64_1616_FP16`：`FP16_L1_RESIDENT_INVERSE=true`，`FP16_L1_RESIDENT_COMPOSE16=true`，`FP16_L1_RESIDENT_COMPOSE32=false`；即 16x16 inverse 和 16x16 compose 走 half L1/TSCM，32x32 compose 仍走稳定 raw/GM fallback。
- 两条线均未引入 RHS resident/raw final，`gBK/vBeta` 仍是 FP32 GM workspace，QK、Stage2、Stage3 未修改。

恢复 `TilingData` 后的复验结果：

| 方案 | chunk | build/install | compare | smoke Neumann/normal |
|---|---:|---|---|---|
| `fulllocal/C16_FP16` | 16 | 通过 | `CGDR_GOLDEN_CHUNK_SIZE=16`，`20 passed in 6.01s` | `S=16 1.129x`，`S=64 2.184x` |
| `fulllocal/C64_1616_FP16` | 64 | 通过 | `CGDR_GOLDEN_CHUNK_SIZE=64`，`20 passed in 6.20s` | `S=16 1.234x`，`S=64 1.722x` |

性能说明：

- smoke CSV 分别为：
  - `tests/pytest/prof_out/fulllocal_c16_fp16_smoke_20260506/performance_compare.csv`
  - `tests/pytest/prof_out/fulllocal_c64_1616_fp16_smoke_after_tiling_restore_20260506/performance_compare.csv`
- 本次 smoke 未裁剪恢复后的完整 `TilingData/MatmulTiling` 结构体，只作为简单性能观察。
- 后续若要做正式单方案放大性能，必须先按 `AGENTS.md` 规则裁剪无关 tiling 字段，并记录保留字段。
