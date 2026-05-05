# 2026-05-05 进度与问题

## 当前状态

- 默认线：`C64_3232_FP32_CVInverse`。
- Neumann inverse 的 `INV += INV@P` 已由底层 CMatrix/MMAD 累加控制，不是标量循环，也不是 Fixpipe atomic add。
- power 保持 lower-triangular raw MMAD；full-tile power 和 `INV@P` k-range 裁剪仍因 `1e-2` 级 out 漂移保持回退。
- final RHS `INV @ k_cumdecay` / `INV @ v_beta` 仍走 FP32 GM `MatmulImpl`。

## 主要耗时判断

- lower-triangular power smoke 中 Neumann 仍慢于 normal：S16 为 `1.323x`，S64 为 `1.664x`。
- 主要放大项是 AIC scalar、MTE1/MTE2、Fixpipe、同步和 GM workspace 往返；MAC time 不是唯一瓶颈。
- 当前实现仍需要每轮把 L0C 结果写回 GM 后再读入 L0A/B，尚未做到 local/TSCM resident。

## 阻塞项

- `L0C` 不能直接作为下一轮 A/B；必须经 `Fixpipe -> L1/TSCM/GM -> LoadData`。
- Fixpipe 到 L1/TSCM 的 compact NZ/stride/layout 尚未验证通过。
- `C64_3232_FP16_LocalTscm_CVInverse` PoC 仍不是默认线，之前全量精度未过。

## 下一动作

1. 若继续优化性能，优先做 local/TSCM compact tile 数值探针。
2. 保留当前 CMatrix/MMAD 累加主线作为精度正确基线。
3. raw final RHS 不混入 inverse 矩阵生成验收，单独排查。

## 纯 cube 核内状态

- 当前还不是纯 cube 核内 resident。
- 已完成：`INV += INV@P` 在 L0C/CMatrix 内通过 `cmatrixInitVal=false` 累加。
- 未完成：跨 round 保持在 L1/TSCM 或 L0；当前仍需 GM workspace 往返。
- 后续目标：验证 `L0C -> Fixpipe -> L1/TSCM -> LoadData`，减少或删除 `GM` 中转。

## CMatrix 累加 vs GM atomic add

- 当前 CMatrix 累加和旧 GM atomic add 的主要区别是加法发生位置：
  - GM atomic add：`tmp=INV@P` 后，Fixpipe/Matmul 写 GM 时执行 `INVWsGm += tmp`。
  - CMatrix 累加：`INV@I` 初始化 L0C，`INV@P` 用 `cmatrixInitVal=false` 直接累加到 L0C，GM 只接收最终 updated `INV`。
- 当前仍需 GM 中转，因此性能上还会保留 Fixpipe/GM/LoadData 固定开销。
- CMatrix 路线是走向片上融合的必要前置，但还不是完整片上 resident。

## 下一步方向：L1/TSCM 承接 updated INV

- 下一步应验证把累加后的 updated `INV` 从 L0C/Fixpipe 写到 L1/TSCM，而不是 GM。
- 目标是删除 `GM -> LoadData` 往返，保留必要的 `L0C -> Fixpipe -> L1/TSCM -> LoadData -> L0A/B`。
- 阻塞点不是理论公式，而是片上 layout：
  - Fixpipe 到 L1/TSCM 的格式和 stride。
  - 下一轮 `LoadData` 对该 layout 的解释。
  - 16x16/32x32 子块 offset 与 compose lower-left block 的 compact NZ 位置。

## 今日计划状态

- 后台任务检查：无仍在运行的构建、pytest、msprof、opc 或安装任务。
- 已创建今日计划：`docs/2026-05-05/L0C_TO_L1_TSCM_INVERSE_COMPLETION_PLAN.md`。
- 今日主线：先打通 `L0C -> Fixpipe -> L1/TSCM -> LoadData` 的 16x16 正确性探针，再扩到 32x32 compose。
- 精度门槛：PoC smoke PASS 后才进入全量 compare；全量 compare PASS 后才跑性能 smoke。

## 用户约束更新：final RHS 暂不修改

- 用户明确 final RHS 可以先不用改。
- 今日后续实现范围收窄为 Neumann inverse 矩阵生成 resident 链路。
- `INV @ k_cumdecay` / `INV @ v_beta` 继续保持 FP32 GM `MatmulImpl`，避免与 L1/TSCM inverse 闭环调试混在一起。

## 计划执行结果更新

### 已完成

- 已按 5.05 计划核对默认线和子代理结论。
- 已确认默认 selector 仍为 `C64_3232_FP32_CVInverse`，final RHS 保持 FP32 GM `MatmulImpl`。
- 已尝试实现 FP32 `L0C -> L1/TSCM` resident inverse 链路，并定位到 CANN/架构类型限制。
- 已从默认文件中移除不可编译 resident 路径，恢复到可构建、可运行的 GM 中转基线。
- 已完成验证：
  - 单算子包构建通过。
  - 双算子包构建通过。
  - custom OPP 安装通过。
  - 全量三方精度 `20 passed in 6.22s`。

### 阻塞项

- `ascend910_93` / DAV2201 当前不支持 `Fixpipe<float, float, CFG_NZ>(LocalTensor<float>, L0C)` 这类 `float L0C -> float L1/TSCM` 写回。
- 因此 5.05 计划中的 FP32 resident 闭环不能作为默认算子实现。
- 可用替代路线会改变精度或适用架构：
  - half/bfloat16 resident inverse，需要重新验证精度。
  - arch35/后续架构如果支持 FP32 L1 写回，可重新启用 FP32 resident 设计。

### 当前默认状态

- 默认算子：`C64_3232_FP32_CVInverse`。
- Neumann inverse：
  - power 仍走 lower-triangular raw MMAD。
  - `INV += INV@P` 在 L0C/CMatrix 内用 `cmatrixInitVal=false` 累加。
  - updated `INV` 仍普通 Fixpipe 写回 `INVWsGm_`，下一轮从 GM 读回。
- final RHS：
  - `INV @ k_cumdecay` 和 `INV @ v_beta` 均保持 FP32 GM `MatmulImpl`。

### 下一动作

1. 若目标仍是 resident inverse，建议先做 half/bfloat16 L1 resident PoC，并把精度门槛作为第一验收项。
2. 若必须保持 FP32 inverse 精度，当前架构上继续沿 GM 中转基线优化同步、cache flush、tiny MMAD 调度开销。
3. 不把不可编译的 FP32 L1 resident 代码保留在默认路径，避免破坏后续构建和 compare。

## L0C -> L1 口径修正

- 已修正结论：不是 `L0C -> L1` 不支持，而是当前 DAV2201 上 `float L0C -> float L1/TSCM` 不支持。
- 可继续验证的 resident 方向是 `float L0C -> half/bfloat16_t L1 NZ -> LoadData -> L0A/L0B -> Mmad(float accum)`。
- 当前新增阻塞点：
  - `Fixpipe<half/bfloat16_t, float, CFG_NZ>` 写到 L1 后是 C/output 的 NZ layout。
  - 下一轮作为 A 或 B 输入时，必须分别按 L0A/L0B 的 `LoadData` 参数解释；A/B 侧不能共用一套 `ifTranspose/stride`。
  - 16x16/32x32 compact tile 的 offset 仍需单独数值探针确认。
- 下一步若继续 half/bfloat16 resident PoC，优先写最小 16x16 闭环验证：`A@B -> L0C -> Fixpipe CFG_NZ 到 L1 -> 分别 LoadData 到 L0A/L0B -> 再 Mmad`，先验证布局再接入完整 Neumann。

## 最新状态复核

### 已确认

- 默认 selector 仍指向 `C64_3232_FP32_CVInverse`，默认四个 FP16/L1 resident 相关开关均为 `false`。
- 默认算子仍处于稳定 FP32 GM 中转基线：Neumann inverse 已使用 raw MMAD/CMatrix 累加，但跨 round 仍写 GM，再由下一轮读回。
- 子代理确认 `L0C -> L1/TSCM` 的可行低精度写回格式是 `CFG_NZ`，不是 `CFG_ZN`；A/B 作为 MMAD 输入时需要分别按 L0A/L0B load 语义处理。

### 仍未完成

- half/bfloat16 L1 resident 的 16x16 闭环探针尚未完成。
- 32x32/64x64 compose 的 compact NZ 子块 offset 尚未完成设备侧验证。
- 默认稳定线未启用 `FP16_L1_RESIDENT_INVERSE`，因此当前性能仍保留 GM workspace 往返开销。

## Baseline/Halflocal/Fulllocal 拆分进度

### 已完成

- baseline 口径已按用户纠正收敛：AIV 只构造初始 `L` / `I-L`，后续 Neumann 轮次使用 AIC Matmul/Fixpipe atomic add。
- `op_kernel/baseline`、`op_kernel/halflocal`、`op_kernel/fulllocal` 已拆分完成，并通过 `op_kernel/selected` selector 选择当前 stage1 实现。
- stage1 `.h` 新文件名已去除 `CVInverse/cvinverse`；`op_kernel/op_host/tests/pytest` 检索无该命名残留。
- C16 halflocal/fulllocal 残留 C64 compose 代码已清理，避免重复函数和未定义 tail 常量。
- baseline `3*2` 与 halflocal `3*2` 均已构建通过，三方精度均 `20 passed`。

### 性能结果

- 已完成指定 S 集合 `S=16/64/125/2048` 的 baseline、halflocal 与 original normal 对比。
- 汇总文件：
  - `docs/2026-05-05/BASELINE_HALFLOCAL_PERFORMANCE_SUMMARY.md`
  - `tests/pytest/prof_out/stage1_baseline_halflocal_s_selected_summary_20260505.csv`
- 当前最佳：`halflocal/C64_3232_FP32`，Neumann/normal ratio 为 `1.305x / 1.651x / 1.873x / 2.332x`，平均 `1.790x`。
- normal 在本轮所有被测点仍更快。

### 阻塞/风险

- `C64_1616` 方案在 `S>=64` 性能极差，ratio 达 `31x+`、`133x+`，主要来自 16x16 子块数量和 tiny MMAD 固定开销放大。
- halflocal 仍是 L0C/CMatrix 内完成 `old_INV + INV@P` 后写回 GM，下一轮仍从 GM 读回，不是 fulllocal resident。
- `halflocal/C16_FP16/S=2048` 首次 msprof 出现一次 `507015`，retry 成功；需要后续重复跑确认稳定性。
- fulllocal 版本当前只是目录和候选文件，不作为本轮精度/性能结论。

### 下一动作

1. 若继续追性能，优先推进 `C64_3232` 的 fulllocal resident PoC，而不是 `C64_1616`。
2. 对 `L0C -> half/bfloat16 L1/TSCM -> LoadData` 先做 16x16 数值探针，确认 NZ layout 和 A/B LoadData 参数。
3. 当前 selector 已复核为 `halflocal/C64_3232_FP32`，这是本轮性能表中的最佳 Neumann 版本；后续 fulllocal 调试前以此作为默认回退点。

## 本轮最新状态与计划

### 当前状态

- 当前 selector：`halflocal/C64_3232_FP32`，chunk size `64`。
- 当前 include：`op_kernel/selected/stage1_selected.h` 指向 `../halflocal/chunk_gated_delta_rule_neumann_stage1_C64_3232_FP32_halflocal.h`。
- 当前无仍在运行的构建、安装、pytest、msprof 或 variant matrix 任务。
- source stage1 `.h` 已去除 `CVInverse/cvinverse` 命名残留；旧 docs/AGENTS 中相关名字只作为历史记录存在。

### fulllocal 进度

- `op_kernel/fulllocal` 已有 `C16_FP16/FP32`、`C64_1616_FP16/FP32`、`C64_3232_FP16/FP32` 以及 `C64_3232_FP16_localtscm` 候选文件。
- fulllocal 尚未作为 suite 完成精度验证；历史 `C64_3232_FP16_LocalTscm` PoC 全量 compare 未过。
- 新增计划文档：`docs/2026-05-05/FULLLOCAL_SINGLE_POINT_BREAKTHROUGH_PLAN.md`。
- 单点突破选择：`fulllocal/C64_3232_FP16`。

### C64_1616 性能问题

- `halflocal/C64_1616_FP32` 在 `S=16/64/125/2048` 的 task ratio 为 `5.240x / 31.341x / 43.869x / 133.229x`。
- 同样四点，`halflocal/C64_3232_FP32` 为 `1.305x / 1.651x / 1.873x / 2.332x`。
- `C64_1616_FP32` 的 AIC MAC ratio 只有 `1.087x / 1.609x / 1.609x / 1.224x`，说明极端慢不是 MAC 主导。
- 主要原因：
  - 每个 C64 chunk 拆成 4 个 16x16 Neumann block，再做 2 次 32 compose 和 1 次 64 compose。
  - 当前 `C64_1616` 仍打开 power/compose MatmulImpl fallback，tiny 16x16 MatmulImpl 调度、Fixpipe/MTE2、cache flush、同步成本被多次放大。
  - AIV 侧大量时间表现为等待/轮询放大，不是有效 vector 计算。
- 处理策略：不把 `C64_1616` 作为 fulllocal 第一突破点；先在 `C64_3232_FP16` 打通 resident 布局和精度，再决定是否回头压 `C64_1616` 的 fallback 和同步。

### FP16/FP32 性能口径

- 当前 `halflocal/C64_3232_FP16` 与 `halflocal/C64_3232_FP32` 的 stage1 代码除文件名/header guard 外无实际差异。
- 当前 `FP16_L1_RESIDENT_INVERSE=false`，所以 FP16 variant 没有真正启用 half L1 resident inverse。
- 当前核心求逆和 final RHS 仍走 FP32 GM/raw MMAD/MatmulImpl 路线。
- 四点结果中 `FP16` 只有 `S=16` 明显慢于 `FP32`，`S=64/125/2048` 基本持平或略快：
  - FP32：`1.305x / 1.651x / 1.873x / 2.332x`。
  - FP16：`1.383x / 1.647x / 1.864x / 2.319x`。
- 当前不能得出“FP16 计算比 FP32 慢”的结论；更准确说是 FP16 文件尚未启用真实低精度 resident 路线，现有差异属于固定开销和测量波动主导。

### FP16 最小实验

- 已尝试仅打开 `FP16_L1_RESIDENT_INVERSE` 的最小修改实验。
- 该实验可以构建并安装，但全量 compare 失败，20 个 case 全部不通过。
- 说明单纯切开关不足以形成可用 FP16 resident 版本；后续若继续 FP16，必须补齐 resident 布局和读写链路，不应把这个最小开关视为最终实现。
- 实验后 `FP16_L1_RESIDENT_INVERSE` 已恢复为 `false`，保留可构建但未启用的单点文件。

### 最终回归

- 已删除其它未构建/未通过精度的 fulllocal 候选，`op_kernel/fulllocal` 当前只保留 `C64_3232_FP16_fulllocal` 单点文件。
- 当前 selector 已恢复为 `halflocal/C64_3232_FP32`，作为 baseline/halflocal 性能表中的最佳稳定回退点。
- 双算子包构建通过，custom OPP 安装通过。
- 稳定线全量三方精度复跑通过：`CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext bash tests/pytest/test_run.sh compare`，结果 `20 passed in 6.15s`。
