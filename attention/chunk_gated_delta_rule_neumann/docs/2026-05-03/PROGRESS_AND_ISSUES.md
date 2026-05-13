# 2026-05-03 进度与问题

## 已完成

- 保持默认 selector 为 `C64_1616_FP32_CVInverse`，host 主 chunk 为 `64`。
- 关闭 `RAW_MMAD_DEBUG_RAW_FINAL_TO_ATTN_THEN_FALLBACK` 调试探针。
- 将 final 两路统一收口到 `MatmulImpl`：
  - `RAW_MMAD_USE_MATMULIMPL_FINAL_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_KCUM_FINAL_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK = true`
- 重新构建双算子包成功：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 重新安装 custom OPP 成功：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 完成 raw final 二分探针：
  - 安全 scratch 探针仍 FAIL，排除 `tmpWsGm_` / `attnWsGm_` 真实输出复用污染。
  - Stage2/Stage3 `RestoreCubeState()` 探针仍 FAIL，排除常规 cube 状态恢复可修复该问题。
  - raw final 多 K 分段 unitFlag `2/3` 探针仍 FAIL，排除单纯 unitFlag 未更新。
  - `RAW_MMAD_DEBUG_SKIP_RAW_FINAL=true` 探针 PASS，确认问题由 `RawMmad64ByN()` 执行本身触发。
- 关键失败 case 已恢复 PASS：
  - `B=2,S=160,nk=8,nv=32,dk=dv=64,seed=40`
  - Neumann out/state `max_abs=9.762645e-04/4.881620e-04`
  - Neumann 与 normal out `max_abs=0.000000e+00`
- 全量三方精度已通过：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
  - 最新复跑结果 `20 passed in 6.22s`（同配置此前结果 `20 passed in 6.24s`）

## 进行中

- 当前以精度通过为优先状态，不再保留 raw final 写 `attnWsGm_` 后 fallback 的调试路径。
- 当前 `C64_1616_FP32_CVInverse` 仍保留 raw/MMAD 调试骨架，后续可继续定位 raw final。
- 后续若追性能，需要逐项关闭 fallback 并记录每轮开关组合和验证结果。

## 阻塞项

- raw `RawMmad64ByN(INVWsGm_, vBetaWsGm_, outVInnerGm_, dv_)` 仍未证明可通过关键 multi-batch/multi-head case。
- raw final 即使写入额外安全 scratch 后再 fallback 覆盖真实输出，仍会导致 out FAIL，说明 raw final 执行本身存在副作用。
- Stage2/Stage3 显式恢复 atomic/padding/ND/HF32/layout transform 未修复；下一步需从 `RawMmad64ByN()` 的 L0C/FixPipe/cache 可见性、GM 写回范围或跨阶段副作用继续查。

## 风险

- 当前精度 PASS 依赖 final MatmulImpl fallback，不能表述为 raw final fused 完成。
- 当前文件中仍有大量 debug 常量，后续测试前必须核对开关，避免误用调试状态跑性能结论。
- 若后续将 `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK` 改回 `false`，应预期关键 case 可能再次失败。

## 下一动作

- 精度提交口径：
  - 保留当前 final fallback 配置。
  - 提交说明中明确 raw final fused 尚未完成，本次是精度收口。
- raw final 后续定位口径：
  1. 基于 `RAW_MMAD_DEBUG_SKIP_RAW_FINAL=true` PASS 的结论，继续收缩 `RawMmad64ByN()` 内部执行段。
  2. 优先二分 `RawMmadAccumulateL0C()` 与 `StoreAccumulatedL0C()`，确认副作用来自 GM->L1/L1->L0、MMAD、Fixpipe 还是 cache clean。
  3. 仅打开 raw `vinner final`，先跑 `seed=40,B=2,S=160,nv=32,dk=dv=64`。
  4. 每轮最终运行 `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`。
