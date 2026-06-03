---
name: fulllocal-compose-precision-debugging
description: Use when debugging fulllocal or compose precision issues in chunk_gated_delta_rule_neumann, especially when a resident/L1/TSCM path diverges from a GM fallback, a boundary shape first fails, or a safe baseline rollback is needed.
---

# Fulllocal/Compose 精度定位

用于 `chunk_gated_delta_rule_neumann` 中 fulllocal、compose、resident/L1/TSCM、GM fallback 等路径的精度定位。核心原则：一次只改一个数据流，先缩小失败边界，再用可信 fallback 隔离问题；未通过精度的实现不得作为后续基线。

## 通用定位流程

1. 确认当前实验入口：
   - `op_kernel/selected/stage1_selected.h`
   - `op_kernel/selected/stage1_selected_config.h`
   - host chunk、golden chunk、selector chunk 必须一致。
2. 找到最小失败边界：
   - 先跑只覆盖单个子块的短序列。
   - 再跑首次触发 compose、tail、跨块、跨 batch/head 的边界 shape。
   - 记录第一个失败 shape 的 `out/state max_abs`，不要只看全量 compare 的汇总。
3. 建立可信对照：
   - 保留上一个全量 compare 通过的 baseline。
   - 用 GM fallback、已通过 compose、或原 BF16 主链路替换单个可疑环节。
   - 如果 fallback PASS、resident FAIL，问题就在被替换的 resident 环节附近，不要继续叠加优化。
4. 缩小失败环节：
   - 先区分是输入 materialize、resident 计算、写回、cast、还是后续消费。
   - 对 compose 类问题，重点检查 full/compact NZ offset、block 顺序、padding、tail 行列清零、Fixpipe 写回后读可见性。
   - 对跨 AIC/AIV 问题，重点检查 flag 语义、等待顺序、cache clean/flush 和 pipe reset 后状态恢复。
5. 回退和沉淀：
   - 当前优化失败时，active path 回退到上一个全量精度通过版本。
   - 保存 `_baseline` 或 `_pN` 时，只保存可通过的干净路径，不保存未启用 probe/helper。
   - 将失败 shape、fallback 结论、回退点写入当日 `WORK_LOG.md` 和 `PROGRESS_AND_ISSUES.md`。

## 本次案例

- 当时的目标是继续推进 fulllocal compose，但 direct resident `32->64` compose 在 `S=33` 首次失败：`CPU vs ChunkGDRNeumann state max_abs=9.669919e-02`，`ChunkGDR vs ChunkGDRNeumann state max_abs=9.655762e-02`。
- 诊断 fallback 将 full-L1 中间块写回 FP16 GM，再调用已验证 GM compose，`S=16/17/33` PASS。
- 由此排除 diagonal 输入、最终写回、AIV half->BF16 cast 作为主因，问题收敛到 resident compose 本体。
- 最终 active 回退到 P3：16x16 Neumann 和 compose 中间态保留在 FP16 GM scratch，full compare `20 passed in 6.53s`。

## 验证命令模板

```bash
bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16
../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp
CGDR_GOLDEN_CHUNK_SIZE=<16|64> TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer python3 tests/pytest/compare_chunk_gdr_three_way.py --seed 0 --batch-size 1 --seqlen <boundary_s> --nk 4 --nv 4 --dk 64 --dv 64
CGDR_GOLDEN_CHUNK_SIZE=<16|64> TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer bash tests/pytest/test_run.sh compare
```

定位期间只跑精度；完整阶段 active path 通过 smoke/full compare 后，再按用户规则决定是否跑一次阶段性能。
