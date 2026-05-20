# C128 当前 TODO

更新时间：2026-05-20 09:55 CST

## 当前状态

- N9.13.0 到 N9.13.4 已全部完成。
- active Stage1：
  `op_kernel/chunk_gated_delta_rule_C128_stage1.h`
- final baseline：
  `op_kernel/baseline/chunk_gated_delta_rule_C128_stage1_n9_13_4_final_pass_20260519.h`
- active Stage1 SHA256：
  `ffa00a838eef0958dca49d5ca9ae1648ef2c1b04cde745dec11ea2965ce0b2f8`
- build package SHA256：
  `0de1bc3c3f3fcb53984871f7c843c8cd1b8e8c6b2b6be83679c4ca6faa0f46ef`
- installed C128 object SHA256：
  `b5e5d86694885a363742da36f738c6c600e26ec30585d00cec78c29e228b38e8`

## 已完成修改

### N9.13.1 resident footprint 重整

- full128 L1 resident 明确拆成 7 个 tile：
  `invInit / diagWork / powerWork / identity / zero / packedInv / residentInv`。
- `Full128L1TileByteOffset()` 固化 tile offset，避免手写 offset 漂移。
- inverse epoch 后，`invInit` 和 `powerWork` 复用为 RHS 首块 prefetch buffer：
  `GetRhsPrefetchGbk128()` / `GetRhsPrefetchVBeta128()`。
- 未扩大 `C128_FULL_L1_TILE_COUNT`，仍保持 7 tile。

### N9.13.2 RHS prefetch / double-buffer

- GBK 首块走 `GetRhsPrefetchGbk128()`。
- VBeta 首块走 `GetRhsPrefetchVBeta128()`。
- `ComputeRhsPacked128Bf16Tiles()` 保持 C64 式顺序：
  首块 issue，当前块 wait，循环内准备 next tile / next para，并用 `stageIdx ^ 1U` ping-pong。
- tail RHS 也显式接入对应 first prefetch buffer。
- dtype 边界未扩大：仍是 BF16 resident + BF16 RHS。

### N9.13.3 restore 生命周期收口

- `ReleaseRawL0AbDbEvents()` 统一做 wait + `ResetRawL0AbDbState()`。
- `InitRawMmadLocalBuffers()` 和 `RestoreStage1MatmulLocalBuffers()` 都走统一封装。
- `RestoreStage1MatmulLocalBuffers()` 保留 `pipe_->Reset()` / `mm.Init()`，并在 reset 后显式清
  `aicLocalBuffersReady_` 与 packed128 aux state，避免下一轮复用 reset 前的 local tensor 句柄。
- 结论：`ResetRawL0AbDbState()`、`pipe_->Reset()`、`mm.Init()` 都不能删；可接受优化边界是封装生命周期，而不是直接删除。

### N9.13.4 合并复核

- 已完成 build / 4-case / ladder / 30-case / C128 vs Neumann profiling。
- 结果已同步到 `performance_baselines.md` Step 018。

## 验证结果

build：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann,chunk_gated_delta_rule_C128 -j16
```

结果：PASS。

4 个 known regression case：

- `b4_s256_d64_h16_h4`：PASS
- `b2_s160_d64_h32_h8`：PASS
- `b1_s384_d128_h8_h4`：PASS
- `b1_s16384_d128_h4_h4`：PASS

full / tail ladder：

- `S=64/80/96/112/128/160/256/1024`：全部 PASS

30-case CPU golden：

```bash
python3 -m pytest -rA -s tests/pytest/compare_chunk_gdr_C128_cpu_golden.py -v -m ci
```

结果：`30 passed, 2 warnings in 7.67s`。

清理后复测：

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_C128/tests/pytest
python3 -m pytest -rA -s compare_chunk_gdr_C128_cpu_golden.py -v -m ci -W ignore::UserWarning -W ignore::DeprecationWarning
```

结果：`30 passed in 9.41s`。

## 性能结果

输出目录：

- `tests/pytest/prof_out/neumann_vs_c128_n9_13_final_20260519/`
- C128 raw：
  `tests/pytest/prof_out/neumann_vs_c128_n9_13_final_20260519/c128_summary/performance_raw.csv`
- Neumann raw：
  `tests/pytest/prof_out/neumann_vs_c128_n9_13_final_20260519/neumann_summary/performance_raw.csv`
- 对比表：
  `tests/pytest/prof_out/neumann_vs_c128_n9_13_final_20260519/c128_vs_neumann_compare.csv`

8-shape 长序列对比：

| S | C128_us | Neumann_us | C128/Neumann |
| ---: | ---: | ---: | ---: |
| 1024 | 2536.36 | 2391.70 | 1.0605 |
| 2048 | 4330.28 | 4282.28 | 1.0112 |
| 4096 | 8175.22 | 8612.04 | 0.9493 |
| 8192 | 16398.84 | 16793.96 | 0.9765 |
| 10240 | 20372.36 | 20447.50 | 0.9963 |
| 12288 | 24133.48 | 24590.90 | 0.9814 |
| 16384 | 32333.44 | 32932.06 | 0.9818 |
| 20480 | 40235.24 | 41141.30 | 0.9780 |

结论：C128 平均 `0.9919x` Neumann；`S=1024/2048` 仍慢，`S>=4096` 整体快于 Neumann。

## 后续可选方向

- `S=1024/2048` 仍慢，主要看 `aic_mac_ratio` 与 `aiv_vec_ratio`，不是简单 MTE2/Fixpipe 问题。
- 不再继续尝试删除 reset / `mm.Init()`；若要优化 restore，只能继续缩小进入 raw-MMAD epoch 的次数或生命周期边界。
- 下一轮若继续优化，建议单独从 `S=1024/2048` 的 inverse / compose MAC 开销入手。
