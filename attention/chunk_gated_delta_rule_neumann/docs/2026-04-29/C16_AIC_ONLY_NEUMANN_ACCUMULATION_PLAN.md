# C16 AIC-only Neumann Accumulation Plan - 2026-04-29

## 失败处理规则

- 精度不过时，必须先在当前优化点上正向定位和修复；不能把回滚到 PASS baseline 当成任务完成。
- 不能在失败候选上继续叠加下一项优化。需要恢复 baseline 时，只能作为明确放弃候选或重建干净工作面的动作，并记录失败现象、根因假设和下一步正向修复入口。


## 背景

当前 C16 Neumann stage1 中，`INV += tmp` 三轮累加由 AIV 执行：AIC 先计算 `tmp = INV @ L_pow` 写 GM，然后通过 flag 通知 AIV；AIV 再从 GM 读 `tmp` 和 `INV`，完成 vector add 后写回 `INV`。profiling 已显示 `S=16` 慢点主要不是 MAC，而是 tiny matmul 调度、AIC/AIV 同步和 GM 往返累加开销。

## 方案A目标

保留 AIV 初始矩阵生成逻辑：

- AIV 继续执行 `InverseCompute()`，生成并写出完整 `L_powWsGm_ = L` 和 `INVWsGm_ = I-L`。
- AIC 等待 flag `0x7` 后，三轮 Neumann 倍增和 `INV` 累加全部在 AIC/FixPipe 路径完成。
- AIV 不再参与三轮 `INV += tmp`，可以直接进入 `GBKCompute()` 和 `VBetaCompute()`，与 AIC Neumann 求逆形成并行。

## 已实现改动

- 基线副本曾用于全量精度前保护旧路径；全量精度通过后已按要求删除。
- 修改实验文件：`op_kernel/chunk_gated_delta_rule_neumann_stage1_C16_FP32_CVInverse.h`。
- AIC 路径改为：
  - `L² = L @ L` 写 `attnWsGm_`。
  - `INV += INV @ L²`，通过 `mmFp32.IterateAll(INVWsGm_, 1)` 走 matmul atomic add。
  - `L⁴ = L² @ L²` 写 `L_powWsGm_`。
  - `INV += INV @ L⁴`，同样 AIC atomic add。
  - `L⁸ = L⁴ @ L⁴` 写 `attnWsGm_`。
  - `INV += INV @ L⁸`，同样 AIC atomic add。
- 删除 `ProcessOneChunk()` 中三轮 AIC↔AIV 累加同步：`0x1/0x2`、`0x3/0x4`、`0xa/0xb`。
- `NeumannAddCompute()` 暂未删除，保留为旧路径参考；当前方案A主流程不调用。
- selector 已正式切到 `chunk_gated_delta_rule_neumann_stage1_C16_FP32_CVInverse.h`。

## 预期性能收益

1. **减少三轮跨核同步**：去掉 AIC 通知 AIV tmp 就绪、AIC 等待 AIV add 完成的三组 flag 往返。
2. **减少 GM 往返搬运**：每轮不再写 `tmpWsGm_` 后由 AIV 读；AIV 也不再读 `INV`、写回 `INV`。
3. **释放 AIV 关键路径**：AIV 完成 `InverseCompute()` 后可直接算 `GBKCompute()` 和 `VBetaCompute()`，与 AIC 的 Neumann matmul 重叠。
4. **降低 AIV vec/scalar/MTE 压力**：profiling 中 S=16 的 AIV add 与 MTE2 开销应下降。
5. **后续可减少 workspace**：方案稳定后，`tmpWsGm_` 可从 Neumann workspace 删除，`neumannWsSz` 可由 `3*c*c` 降为 `2*c*c`。

## 风险与待验证项

- 当前实现使用 `INVWsGm_` 同时作为 matmul 输入 A 和 atomic add 输出 C。C16 单 tile smoke 已通过，但需要全量精度和 profiling 确认该同址读写在所有 shape 下稳定。
- atomic add 可能增加 FixPipe 开销，需要用 msprof 对比 AIV add 旧路径；判断指标应重点看 `aic_fixpipe_time_us`、`aic_scalar_time_us`、`aiv_vec_time_us`、`aiv_mte2_time_us` 和端到端耗时。
- selector 已永久切到 `C16_FP32_CVInverse`，当前默认构建即使用方案A。
- 若同址 atomic add 在大批量或不同 tiling 下出现不稳定，需要退一步改为 AIC 输出 `tmpWsGm_` 后用 AIC vector/UB 或非同址双 buffer 累加。

## 验证记录

- selector 已正式切到 `chunk_gated_delta_rule_neumann_stage1_C16_FP32_CVInverse.h`。
- 构建通过：`bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`。
- 安装后 smoke 通过：`python3 compare_chunk_gdr_three_way.py --batch-size 1 --seqlen 64 --nk 4 --nv 4 --dk 64 --dv 64`。
- 全量三方精度通过：`bash test_run.sh compare`，20 case 全部 PASS，耗时 `5.96s`。
- 全量精度通过后已删除基线副本 `stage1_C16_FP32_CVInverse_AIVAddBaseline_20260429.h`。

## 下一步

1. 重新跑包含 `S=16/S=64` 的全量 32 shape msprof 批测。
2. 对比旧路径 profiling，确认收益是否来自 AIV add/MTE2/同步下降。
3. 若方案A性能稳定，再删除 `tmpWsGm_` 相关 workspace 和不再调用的 AIV add UB 占用。
