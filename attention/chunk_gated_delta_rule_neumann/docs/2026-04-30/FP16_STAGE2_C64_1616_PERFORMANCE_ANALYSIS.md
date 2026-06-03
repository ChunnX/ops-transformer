# 2026-04-30 FP16 方案2：C64_1616 NeumannCompose 阶段总结

## 方案定义

- 方案名称：`C64_1616_FP16_CVInverse`
- 默认 selector：`op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_CVInverse.h`
- host chunk：`chunkSize=64`
- 求逆结构：64×64 下三角矩阵按 4 个 16×16 对角块做 Neumann，再按 32×32/64×64 block compose 得到整块 inverse。
- FP16 范围：Neumann inverse 内部 `L/INV/tmp` 使用 half workspace，16×16/32×32 子块 matmul 使用 `half × half -> half`；final `INV @ gBK/vBeta` 使用 `half × half -> float`。
- Stage2/Stage3：未修改。

## 关键实现点

1. `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` 默认切到 `C64_1616_FP16_CVInverse`。
2. `op_kernel/chunk_gated_delta_rule_neumann_stage1_C64_1616_FP16_CVInverse.h` 新增 FP16 MatmulType：
   - `StageOneFp16MT`：`half × half -> half`
   - `StageOneFp16OutMT`：`half × half -> float`
3. AIV 侧显式生成 half 副本：
   - `L_powHalfWsGm_`
   - `INVHalfWsGm_`
   - `tmpHalfWsGm_`
   - `gBKHalfWsGm_`
   - `vBetaHalfWsGm_`
4. `op_host/chunk_gated_delta_rule_neumann_tiling.cpp` 新增 FP16 tiling：
   - `matmulTilingFp16Out`
   - `matmulTilingFp16C16`
   - `matmulTilingFp16C32`
5. `op_kernel/chunk_gated_delta_rule_neumann.h` 初始化 FP16 matmul 实例，并传入 Stage1。

## 构建与安装

构建命令：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
```

说明：本机实测需要 `ascend910_93` 包。最初安装 `ascend910b` 包会导致普通 `ChunkGatedDeltaRule` 报 `Op ChunkGatedDeltaRule does not has any binary`。

## 精度结果

执行命令：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_neumann/tests/pytest
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=/mnt/workspace/gitCode/cann/ops-transformer/torch_extension:${PYTHONPATH}
CGDR_GOLDEN_CHUNK_SIZE=64 bash test_run.sh compare
```

结果：`20 passed in 6.14s`。

精度门槛：`atol=0.08`、`rtol=0.08`。

代表性误差：

| case | CPU vs Neumann out max_abs | CPU vs Neumann state max_abs | ChunkGDR vs Neumann out max_abs | ChunkGDR vs Neumann state max_abs |
|---|---:|---:|---:|---:|
| `default_b1_t64_d128` | `9.736717e-04` | `5.198121e-04` | `9.765625e-04` | `9.765625e-04` |
| `b4_t32_h16_h4_d64` | `9.779036e-04` | `5.685687e-04` | `1.953125e-03` | `9.765625e-04` |
| `b2_t160_h32_h8_d64` | `9.855628e-04` | `5.408376e-04` | `1.953125e-03` | `9.765625e-04` |

结论：FP16 方案2在当前 20 个三方精度 case 下全部通过，误差远低于 `0.08/0.08` 门槛。

## 性能测试

测试方法：继续使用 `msprof --application`，payload 内部执行 operator warmup、`torch.mm(10240×10240)` flush、`ReduceMax(96×1024×1024)` flush、最后单次 measurement。

输出目录：

```text
tests/pytest/prof_out/chunk_gdr_phase2_c64_1616_fp16_seq_sweep/
```

CSV：

```text
tests/pytest/prof_out/chunk_gdr_phase2_c64_1616_fp16_seq_sweep/performance_compare.csv
tests/pytest/prof_out/chunk_gdr_phase2_c64_1616_fp16_seq_sweep/performance_raw.csv
```

执行 case：B=8，遍历 `S=16/64/128/256/512/1024/2048`，其中 `S=512` 同时覆盖 `D=64` 和 `D=128`。

| S | D | Hq | Hkv | normal us | FP16 Neumann us | neumann/normal | faster |
|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 64 | 16 | 4 | 636.66 | 729.92 | 1.146 | normal |
| 64 | 64 | 16 | 4 | 657.46 | 1267.62 | 1.928 | normal |
| 128 | 64 | 16 | 4 | 898.32 | 2102.52 | 2.341 | normal |
| 256 | 64 | 16 | 4 | 1361.46 | 3641.30 | 2.675 | normal |
| 512 | 64 | 32 | 8 | 3379.78 | 10679.52 | 3.160 | normal |
| 512 | 128 | 32 | 8 | 3961.60 | 11254.82 | 2.841 | normal |
| 1024 | 128 | 32 | 8 | 7488.88 | 22065.10 | 2.946 | normal |
| 2048 | 128 | 16 | 4 | 7619.46 | 22190.38 | 2.912 | normal |

汇总：

- 平均 `neumann/normal = 2.494x`
- 最好 case：`S=16`，`1.146x`
- 最差 case：`S=512,D=64,Hq=32,Hkv=8`，`3.160x`
- 8/8 个 seq sweep case 均为 normal 更快

## Profiling 指标观察

| S | D | FP16 Neumann aic_mac_ratio | aic_scalar us | aic_mte2 us | aic_fixpipe us | aiv_vec_ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 64 | 0.008 | 168.52 | 46.82 | 37.54 | 0.028 |
| 64 | 64 | 0.013 | 349.86 | 128.80 | 93.80 | 0.016 |
| 128 | 64 | 0.015 | 644.08 | 242.07 | 184.49 | 0.019 |
| 256 | 64 | 0.017 | 1213.00 | 456.15 | 335.16 | 0.021 |
| 512 | 64 | 0.024 | 4651.98 | 1746.23 | 1301.10 | 0.029 |
| 512 | 128 | 0.042 | 5024.55 | 1969.84 | 1724.07 | 0.032 |
| 1024 | 128 | 0.043 | 10022.57 | 3968.45 | 3496.47 | 0.033 |
| 2048 | 128 | 0.042 | 10023.70 | 4074.78 | 3567.52 | 0.032 |

结论：FP16 降低了部分 Cube 输入/输出位宽，但当前实现仍是 `MatmulImpl + GM half workspace + FIX/MTE2` 串联。耗时主因仍在 AIC scalar、MTE2、FixPipe 随 chunk/子块调用数放大，不是单纯 MAC dtype。

## 阶段结论

1. 精度：通过，20/20 PASS。
2. 性能：未达预期，seq sweep 平均仍为 normal 的 `2.494x`。
3. FP16 本身有意义，但当前收益被 GM workspace 往返和通用 MatmulImpl tiny 调度开销吞掉。
4. 后续若继续方案2，应优先减少 `AICProcessFp16/AICBlockProcessFp16` 的 GM 中间落地次数，而不是继续只改 dtype。
5. 下一阶段建议对比 `C64_3232_FP16_CVInverse`，因为 32×32 compose 的 Matmul 调用数少，理论上更可能受益于 FP16。
