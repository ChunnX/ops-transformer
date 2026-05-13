# 2026-04-30 C64 方案最终性能表

## 数据来源

- `C64_1616_sync_reduced`: `tests/pytest/prof_out/chunk_gdr_c64_1616_sync_reduced_all_shapes`，这是当前用于对比的 C64 16x16 子块方案全量数据。

- `C64_3232_hybrid`: `tests/pytest/prof_out/chunk_gdr_c64_3232_hybrid_all_shapes`，这是当前 32x32 主路径 + tail shortcut 的 hybrid 方案全量数据。


## 方案说明与 CSV 对应关系

| 方案 | 含义 | Profiling 原始汇总目录 | 对应 CSV |
| --- | --- | --- | --- |
| `C64_1616_sync_reduced` | 主 chunk 保持 `C=64`；64x64 对角求逆拆成 16x16 Neumann 子块，再通过 block compose 得到 64x64 inverse；相比早期 1616 版本减少部分同步，是当前用于最终对比的 16x16 子块版本。 | `tests/pytest/prof_out/chunk_gdr_c64_1616_sync_reduced_all_shapes` | 总表：`docs/2026-04-30/C64_FINAL_SCHEME_SUMMARY.csv`；按 S：`docs/2026-04-30/C64_FINAL_BY_S.csv`；逐 case：`docs/2026-04-30/C64_FINAL_CASE_DETAIL.csv` 中 `scheme=C64_1616_sync_reduced` |
| `C64_3232_hybrid` | 主 chunk 保持 `C=64`；常规路径用 32x32 Neumann 子块，最后 compose 成 64x64 inverse；短尾 `validLen<=16` 走 16x16 shortcut，`32<validLen<=48` 走 32+16 混合，所以叫 hybrid。 | `tests/pytest/prof_out/chunk_gdr_c64_3232_hybrid_all_shapes` | 总表：`docs/2026-04-30/C64_FINAL_SCHEME_SUMMARY.csv`；按 S：`docs/2026-04-30/C64_FINAL_BY_S.csv`；逐 case：`docs/2026-04-30/C64_FINAL_CASE_DETAIL.csv` 中 `scheme=C64_3232_hybrid` |

### CSV 文件用途

- `docs/2026-04-30/C64_FINAL_SCHEME_SUMMARY.csv`: 每个方案一行，给 overall 平均耗时比和 AIC/AIV 指标比值。
- `docs/2026-04-30/C64_FINAL_BY_S.csv`: 每个方案按 `S` 分组汇总，方便看 `S=16/64/128/...` 的趋势。
- `docs/2026-04-30/C64_FINAL_CASE_DETAIL.csv`: 每个具体 shape 一行，保留 `B/S/D/Hq/Hkv` 和该 case 的比值。

## 总表

| scheme | cases | neumann_faster_cases | avg_neumann_over_normal | min_neumann_over_normal | max_neumann_over_normal | avg_aic_mac_ratio | avg_aic_scalar_ratio | avg_aic_mte2_ratio | avg_aic_fixpipe_ratio | avg_aiv_vec_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C64_1616_sync_reduced | 32 | 4 | 2.339 | 0.921 | 3.010 | 1.494 | 3.082 | 2.604 | 2.061 | 0.403 |
| C64_3232_hybrid | 32 | 3 | 1.778 | 0.981 | 2.198 | 1.382 | 2.202 | 1.979 | 1.638 | 0.404 |

## 按 S 汇总

| scheme | S | cases | neumann_faster_cases | avg_neumann_over_normal | avg_aic_mac_ratio | avg_aic_scalar_ratio | avg_aic_mte2_ratio | avg_aic_fixpipe_ratio | avg_aiv_vec_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C64_1616_sync_reduced | 16 | 4 | 4 | 0.946 | 1.146 | 1.249 | 1.334 | 1.329 | 0.392 |
| C64_1616_sync_reduced | 64 | 4 | 0 | 1.773 | 1.725 | 2.894 | 2.637 | 2.619 | 0.399 |
| C64_1616_sync_reduced | 128 | 4 | 0 | 2.124 | 1.725 | 3.329 | 2.810 | 2.378 | 0.392 |
| C64_1616_sync_reduced | 256 | 4 | 0 | 2.550 | 1.725 | 3.655 | 2.985 | 2.378 | 0.388 |
| C64_1616_sync_reduced | 512 | 8 | 0 | 2.858 | 1.514 | 3.573 | 3.058 | 2.170 | 0.406 |
| C64_1616_sync_reduced | 1024 | 4 | 0 | 2.820 | 1.302 | 3.196 | 2.581 | 1.766 | 0.423 |
| C64_1616_sync_reduced | 2048 | 4 | 0 | 2.782 | 1.302 | 3.184 | 2.373 | 1.677 | 0.421 |
| C64_3232_hybrid | 16 | 4 | 3 | 0.994 | 1.144 | 1.239 | 1.271 | 1.252 | 0.394 |
| C64_3232_hybrid | 64 | 4 | 0 | 1.391 | 1.554 | 2.077 | 1.927 | 1.884 | 0.399 |
| C64_3232_hybrid | 128 | 4 | 0 | 1.638 | 1.554 | 2.320 | 2.058 | 1.813 | 0.393 |
| C64_3232_hybrid | 256 | 4 | 0 | 1.923 | 1.554 | 2.515 | 2.241 | 1.835 | 0.389 |
| C64_3232_hybrid | 512 | 8 | 0 | 2.095 | 1.393 | 2.474 | 2.259 | 1.712 | 0.406 |
| C64_3232_hybrid | 1024 | 4 | 0 | 2.057 | 1.231 | 2.263 | 1.968 | 1.474 | 0.422 |
| C64_3232_hybrid | 2048 | 4 | 0 | 2.033 | 1.231 | 2.251 | 1.846 | 1.424 | 0.421 |

## 结论

- `C64_3232_hybrid` 明显优于 `C64_1616_sync_reduced`：平均耗时比从 `2.339x` 降到 `1.778x`。

- `C64_3232_hybrid` 仍未整体超过 normal；主要优势只出现在 `S=16`，`S>=64` 仍慢。

- `C64_1616_sync_reduced` 的问题是 16x16 tiny MatMul 数量更多，AIC scalar/MTE2/FixPipe 固定开销放大。
