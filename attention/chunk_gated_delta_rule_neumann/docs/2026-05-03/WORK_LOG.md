# 2026-05-03 工作日志

## 本轮目标

- 按用户要求继续到精度通过为止。
- 当前默认 selector 保持 `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP32_CVInverse.h`。
- 当前 host 主 chunk 保持 `chunkSize=64`，对应 `C64_1616_NeumannCompose` 调试线。

## 当前开关收口

- 关闭 raw final 调试探针：
  - `RAW_MMAD_DEBUG_RAW_FINAL_TO_ATTN_THEN_FALLBACK = false`
- final 两路统一回退到 `MatmulImpl`：
  - `RAW_MMAD_USE_MATMULIMPL_FINAL_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_KCUM_FINAL_FALLBACK = true`
  - `RAW_MMAD_USE_MATMULIMPL_VINNER_FINAL_FALLBACK = true`
- 仍保留当前 raw/MMAD 调试骨架和 Stage1 `C64_1616` selector，未切换到其他方案。

## 修复背景

- 修复前关键失败 case：
  - `B=2,S=160,nk=8,nv=32,dk=dv=64,seed=40`
  - `CGDR_GOLDEN_CHUNK_SIZE=64`
  - `atol=rtol=0.08`
- 失败形态：
  - 普通 `ChunkGDR` out/state PASS。
  - `ChunkGDRNeumann` state PASS 且与 normal 几乎一致。
  - `ChunkGDRNeumann` out FAIL，曾出现 `max_abs` 约 `1.037e-01`。
- 判断：
  - 当前问题集中在 raw `INV @ v_beta -> outVInnerGm_` final，或 raw final 执行后对后续 Stage3/MatmulImpl 状态造成副作用。
  - 为满足“精度先通过”，本轮不继续冒险推进 raw final，而是将 final 两路收口回 `MatmulImpl`。

## 构建安装

- 构建命令：
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- 构建结果：
  - 通过，生成 `../../build_out/cann-ops-transformer-custom_linux-aarch64.run`
- 安装命令：
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- 安装结果：
  - `SUCCESS`

## 精度验证

### 关键失败 case 复测

- 命令：
  - `python3 tests/pytest/compare_chunk_gdr_three_way.py --seed 40 -B 2 --seqlen 160 --nk 8 --nv 32 --dk 64 --dv 64 --atol 0.08 --rtol 0.08`
- 环境：
  - `ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer:${ASCEND_CUSTOM_OPP_PATH}`
  - `LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}`
  - `TORCH_EXTENSIONS_DIR=/tmp/torch_extensions_cgdr_neumann`
  - `CGDR_GOLDEN_CHUNK_SIZE=64`
- 结果：
  - `CPU vs ChunkGDRNeumann out` PASS，`max_abs=9.762645e-04`
  - `CPU vs ChunkGDRNeumann state` PASS，`max_abs=4.881620e-04`
  - `ChunkGDR vs ChunkGDRNeumann out` PASS，`max_abs=0.000000e+00`
  - `ChunkGDR vs ChunkGDRNeumann state` PASS，`max_abs=9.313226e-10`
  - 脚本最终 `PASSED`

### 全量三方 compare

- 命令：
  - `CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`
- 结果：
  - `20 passed in 6.24s`
  - 按用户追加要求继续复跑同一全量 compare，最新结果 `20 passed in 6.22s`
- 覆盖：
  - 默认 case、变量长度、tail、GQA、多 batch、多 head、`dk/dv` 组合和关键 `B=2,S=160,nv=32` case。

## 继续定位记录

- 安全 scratch 探针：
  - 在 host tiling 中为 raw final 增加 `rawFinalScratchSz = sizeHigh * aiCoreNum * c * max(dk,dv)`。
  - Stage1 增加 `rawFinalScratchGm_`，让 raw `INV @ v_beta` 先写安全 scratch，再用 `MatmulImpl` 覆盖真实 `outVInnerGm_`。
  - 关键 case 仍 FAIL，`CPU vs ChunkGDRNeumann out max_abs=1.037243e-01`，`state` PASS。
  - 结论：不是 `tmpWsGm_` / `attnWsGm_` 复用污染真实输出；只要执行 raw final，就会影响后续 out 链路。
- Stage2/Stage3 cube 状态恢复探针：
  - 在 Stage2 `CalAttnInter`、`CalVPrime`、`CalStateNew` 和 Stage3 `AICProcess` 前增加 `RestoreCubeState()`，重置 atomic、padding、ND 参数、HF32、layout transform。
  - 关键 case 仍 FAIL，误差与 safe scratch 实验一致。
  - 结论：常规 cube/FixPipe 状态恢复不能消除 raw final 执行后的 out 偏差。
- raw MMAD unitFlag 探针：
  - 将 raw final 多 K 分段的 `RAW_MMAD_NON_FINAL_UNIT_FLAG/RAW_MMAD_FINAL_UNIT_FLAG` 从 `0/0` 改为 `2/3`，对齐仓库 grouped matmul、lightning indexer 的累加控制用法。
  - 关键 case 仍 FAIL，说明问题不只是 unitFlag 未关闭/未更新。
- skip raw final 探针：
  - 设置 `RAW_MMAD_DEBUG_SKIP_RAW_FINAL=true`，保留 fallback 覆盖链路，但完全跳过 raw final 执行。
  - 关键 case PASS，Neumann out/state `max_abs=9.762645e-04/4.881620e-04`，Neumann 与 normal out `max_abs=0`。
  - 结论：失败被收敛到 `RawMmad64ByN()` 执行本身；不是 final `MatmulImpl` fallback、Stage2 state 或 Stage3 常规计算路径。

## 当前真实状态

- 精度状态：
  - 当前源码构建安装后，全量三方精度通过。
  - 最新全量结果：`CGDR_GOLDEN_CHUNK_SIZE=64 bash tests/pytest/test_run.sh compare`，`20 passed in 6.22s`。
- fused 状态：
  - raw final fused 未完成。
  - 当前 `INV @ k_cumdecay` 与 `INV @ v_beta` final 走 `MatmulImpl`，这是精度优先收口。
  - base inverse、power、compose 等路径仍保留此前 fallback/debug 组合，不能宣称完整 raw MMAD fused。
- 后续定位建议：
  - 继续定位 `RawMmad64ByN()` 时，应优先检查 raw final 对 L0C/FixPipe/GM cache 或跨阶段 workspace 可见性的副作用。
  - 每次关闭 fallback 后先跑关键 `seed=40,B=2,S=160,nv=32,dk=dv=64` case，再跑全量 compare。
