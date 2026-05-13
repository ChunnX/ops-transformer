# Neumann Scheme Summary Skill

用于总结、对比和汇报 `chunk_gated_delta_rule_neumann` 的 Neumann 求逆方案。触发场景：用户要求总结方案、解释 C16/C64、解释 1616/3232、整理 profiling 表格、确认最终性能口径。

## 固定总结口径

后续只把方案归为三种最终方案；中间优化过程不得作为最终方案单独列出。

| 方案编号 | 最终方案名 | 主 chunk | 求逆结构 | 说明 |
|---|---|---:|---|---|
| 1 | `C16_1616_Neumann` | 16 | 直接 16×16 Neumann 求逆 | 主 chunk 就是 16，host tiling 必须 `chunkSize=16`。当前优化版把 `INV += INV@Lpow` 放到 AIC/FixPipe atomic add，但仍落 GM，不是真正 fused。 |
| 2 | `C64_1616_NeumannCompose` | 64 | 16×16 Neumann 子块 + block compose | host 主 chunk 是 64；`1616` 是内部求逆子块。最终数据口径使用 `chunk_gdr_c64_1616_sync_reduced_all_shapes`。 |
| 3 | `C64_3232_NeumannCompose` | 64 | 32×32 Neumann 子块 + block compose | host 主 chunk 是 64；当前最终实现是 hybrid，短尾可走 16×16 shortcut。最终数据口径使用 `chunk_gdr_c64_3232_hybrid_all_shapes`。 |

## 命名规则

- `C16` / `C64` 表示 host tiling 的主 `chunkSize`。
- `1616` / `3232` 表示 Neumann 求逆内部子矩阵大小。
- `sync_reduced`、`tail_specialized`、`cacheline`、`current` 是优化过程标签，不是最终方案名。
- `hybrid` 是 `C64_3232_NeumannCompose` 的最终实现细节：按 `validLen` 选择 16×16/32×32/32+16 路径。

## 当前最终性能数据

C64 最终表：

- Markdown：`docs/2026-04-30/C64_FINAL_PERFORMANCE_TABLE.md`
- 总表 CSV：`docs/2026-04-30/C64_FINAL_SCHEME_SUMMARY.csv`
- 按 S CSV：`docs/2026-04-30/C64_FINAL_BY_S.csv`
- 逐 case CSV：`docs/2026-04-30/C64_FINAL_CASE_DETAIL.csv`

Profiling 目录映射：

- `C64_1616_NeumannCompose` → `tests/pytest/prof_out/chunk_gdr_c64_1616_sync_reduced_all_shapes`
- `C64_3232_NeumannCompose` → `tests/pytest/prof_out/chunk_gdr_c64_3232_hybrid_all_shapes`
- `C16_1616_Neumann` → `tests/pytest/prof_out/chunk_gdr_prefill_all_shapes_aic_accum`

## 已知结论

- C64 3232 当前优于 C64 1616：平均耗时比 normal 从约 `2.339x` 降到约 `1.778x`。
- C16 在 `S=16` 附近可接近 normal，但长序列整体慢，平均约 `3.015x`。
- 主要瓶颈不是理论 MAC，而是 `MatmulImpl + GM workspace + FIX/MTE2` 串 tiny MatMul 的固定开销。
- 真正 fused AIC inverse 的目标是让 `L/power/INV` 尽量驻留 L1/L0/register，最后只写最终 `INV`；当前实现还未达到。

## 汇报模板

```text
本轮只按三种最终方案总结：
1. C16_1616_Neumann：主 chunk=16，16×16 Neumann，当前 AIC atomic-add 优化版。
2. C64_1616_NeumannCompose：主 chunk=64，16×16 子块 Neumann + compose，最终数据来自 ...sync_reduced_all_shapes。
3. C64_3232_NeumannCompose：主 chunk=64，32×32 子块 Neumann + compose，hybrid 短尾优化，最终数据来自 ...3232_hybrid_all_shapes。
```
