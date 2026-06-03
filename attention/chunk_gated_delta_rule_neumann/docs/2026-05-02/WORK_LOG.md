# 2026-05-02 工作日志

## 当前状态核对

- 当前默认 selector 仍指向 `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`，host 主 chunk 为 `64`，对应 `C64_1616_NeumannCompose`。
- 当前源码已按精度优先状态落地并重新构建安装，`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare` 全量三方精度 `20 passed in 6.18s`。
- 当前开关组合：
  - `RAW_MMAD_USE_MATMULIMPL_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_POWER_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_COMPOSE_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_KCUM_FINAL_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK = true`
- 当前精度已通过；raw `INV @ v_beta -> outVInnerGm_` 仍未修好，若重新关闭 `vinner final` fallback，需要从关键失败 case 重新验证。

## 修复前失败状态

- 小范围 raw inverse 链路此前已通过：`S=16/17/33/49/64, dk=dv=16` 和 `S=16, dk=dv=32` 均 PASS，证明 `LoadData3DParamsV2<float>` 修复了 `L1->L0A/L0B` 布局问题。
- 修复前关键失败 case：
  - `B=2,S=160,nk=8,nv=32,dk=dv=64,seed=40`
  - `CGDR_GOLDEN_CHUNK_SIZE=64`
  - `atol=rtol=0.08`
- 修复前表现：
  - 普通 `ChunkGDR` PASS。
  - `ChunkGDRNeumann` 的 `state` PASS，但 `out` FAIL。
  - 使用 `Partial=2/Final=3` unit flag 未修复。
  - 打开 HF32 后误差反而更差，应回退为 `HF32Mode::DISABLE`。

## 当前判断

- `state` PASS 且 `kCumdecay` final 仍为 fallback，说明 Stage2 状态链路和 raw inverse 主体不是当前唯一问题。
- 失败主要集中在 raw final `INV @ v_beta -> outVInnerGm_`，或 raw final 写 GM 对后续 Stage3 workspace/缓存状态造成副作用。
- 已验证过 raw final 写到 `tmpWsGm_` 再由 fallback 覆盖真实 `outVInnerGm_` 后仍可能导致 `out` 失败，说明不能只看目标输出数值，还需要排查 scratch workspace 与 Stage3 复用关系。

## 后续 raw final 定位计划

1. 保持 `C64_1616` selector 不变，围绕 raw final `RawMmad64ByN()` 单独定位。
2. 增加不污染 `tmpWsGm_` / `INVWsGm_` / Stage3 复用区的安全 scratch，隔离“raw final 写副作用”和“raw final 数值错误”。
3. 重新打开 `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK=false` 后，先跑关键失败 case。
4. 如果隔离后 fallback PASS，则优先修 raw final 写回/缓存/stride；如果仍 FAIL，则继续查 raw MMAD 状态恢复或 shared stage workspace 副作用。
5. raw final 通过后，再逐步关闭 `kcum final`、compose、power、base inverse fallback，并运行全量 `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`。

## HF32 回退后复测

- 已将 `C64_1616_FP32_CVInverse` 中 raw MMAD 路径的 `SetHF32Mode(HF32Mode::ENABLE)` 回退为 `DISABLE`。
- 重新构建双算子包通过：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 安装通过：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 复测关键 case：
  - `B=2,S=160,nk=8,nv=32,dk=dv=64,seed=40`
  - `CGDR_GOLDEN_CHUNK_SIZE=64`
  - `atol=rtol=0.08`
- 结果：
  - `CPU vs ChunkGDR` out/state PASS，out/state max_abs 为 `9.762645e-04 / 4.881620e-04`。
  - `CPU vs ChunkGDRNeumann` out FAIL，`max_abs=1.037243e-01`；state PASS，`max_abs=4.881620e-04`。
  - `ChunkGDR vs ChunkGDRNeumann` out FAIL，`max_abs=1.037598e-01`；state PASS，`max_abs=9.313226e-10`。
- 结论：
  - HF32 回退后误差回到此前 `~1.037e-01` 级别，说明 HF32 enable 是负向实验而不是根因。
  - `state` 与 normal 几乎完全一致，当前问题继续集中在 raw `INV @ v_beta -> outVInnerGm_` 及 Stage3 读取该结果的链路。

## 修复前换窗口交接摘要

- 修复前不是“所有阶段都 fused，只剩 bug”：
  - Stage2/Stage3 未做 fused 改造，仍使用原 MatmulImpl 流程。
  - Stage1 的 `C64_1616` raw MMAD fused inverse/final 路线已有实现骨架，小范围 smoke 曾通过。
  - 为定位大 case，base inverse、power、compose、`kcum final` 临时 fallback 到 MatmulImpl，`vinner final` 当时走 raw。
- 当前关键失败点：
  - `RawMmad64ByN(INVWsGm_, vBetaWsGm_, outVInnerGm_, dv_)`
  - 失败表现为 `out` FAIL，但 `state` PASS 且与 normal 几乎完全一致。
- 重要注意：
  - `tmpWsGm_` 会被 Stage3 作为 `masked_qkt` 临时矩阵使用，不能作为干净 scratch 做 raw final 隔离。
  - 后续若继续隔离 raw final，需要新增或确认一块不被 Stage2/Stage3 使用的 scratch。

## 精度恢复修复

- 复现关键失败 case：
  - `B=2,S=160,nk=8,nv=32,dk=dv=64,seed=40`
  - `CGDR_GOLDEN_CHUNK_SIZE=64`
  - `atol=rtol=0.08`
- 修复前本轮实测表现：
  - `CPU vs ChunkGDRNeumann out` FAIL，`max_abs=1.584216e-01`
  - `CPU vs ChunkGDRNeumann state` PASS
  - `ChunkGDR vs ChunkGDRNeumann state` 仍几乎一致，说明问题集中在 Stage1 `vInner` 输出或 Stage3 消费链路。
- 小切片验证结论：
  - `B=1,S=160,nk=8,nv=32,dk=dv=64` 在阈值内 PASS。
  - `B=2`、多 head/multi chunk 更容易触发，继续支持“raw final 写回/状态污染或跨任务问题”的判断。
- 本次精度修复：
  - 将 `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK` 从 `false` 改为 `true`。
  - 保持 `C64_1616_FP32_CVInverse` selector 与 host `chunkSize=64` 不变。
  - 该修复是精度优先回退：`INV @ v_beta -> outVInnerGm_` 暂时回到 `MatmulImpl`，不是 raw final fused 完成。
- 构建安装：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 修复后关键 case：
  - `CPU vs ChunkGDRNeumann out` PASS，`max_abs=9.762645e-04`
  - `CPU vs ChunkGDRNeumann state` PASS，`max_abs=4.881620e-04`
  - `ChunkGDR vs ChunkGDRNeumann out` PASS，`max_abs=0`
  - `ChunkGDR vs ChunkGDRNeumann state` PASS，`max_abs=9.313226e-10`
- 全量三方精度：
  - 从算子根目录运行：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - 本轮源码落地后验证 `20 passed in 6.18s`。

## 测试脚本修复

- 修复 `tests/pytest/test_run.sh`：
  - 进入脚本所在目录后再执行 pytest，避免从算子根目录运行时找不到 `compare_chunk_gdr_three_way.py`。
  - `single/compare/performance` 分支现在透传实际命令退出码，避免 pytest 失败但 shell 脚本末尾 `exit 0` 造成假通过。

## 当前真实状态

- 当前默认实现仍是 `C64_1616_FP32_CVInverse`。
- 当前精度状态：全量三方精度 20/20 PASS。
- 当前 fused 状态：
  - base inverse、power、compose、`kcum final`、`vinner final` 均仍有 MatmulImpl fallback。
  - raw `INV @ v_beta -> outVInnerGm_` 仍未修好，后续若继续追 fused 性能，需要重新打开 `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK=false` 并单独定位。
