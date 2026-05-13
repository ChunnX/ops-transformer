# Performance Profiling with msprof Skill

用于 `chunk_gated_delta_rule` / `chunk_gated_delta_rule_neumann` 性能测试、批测、结果汇总。触发场景：用户要求性能对比、profiling、msprof、批量 shape、cache flush、提取 AIC/AIV 指标。

## 强制原则

- 性能测试必须使用 `msprof --application=... --task-time=on`。
- 不用 `torch.npu.Event`、Python wall time 或自写 benchmark 作为最终性能结论。
- 单次 measurement 前必须先 warm operator，再 flush cache。
- 后续默认 shape 集合包含 `S=16`、`S=64` 和 prefill shapes，不能只测原 24 个 shape。

## 标准流程

1. Phase 1: operator warmup
   - 跑一次目标算子，加载 kernel binary，warm instruction cache。
2. Phase 2: MatMul flush
   - 用大矩阵 `torch.mm` 刷 L2/cache，例如 `10240x10240` FP16。
3. Phase 3: ReduceMax flush
   - 用大 tensor reduce 进一步 flush，例如 `96x1024x1024`。
4. Phase 4: measurement
   - 同输入 shape 再跑一次目标算子，作为最终 op_summary 目标行。

## 脚本入口

在 `tests/pytest` 下运行：

```bash
python3 run_chunk_gdr_performance_msprof.py \
  --output ./prof_out/<output_name> \
  --ops both
```

常用输出目录：

- `chunk_gdr_prefill_all_shapes_aic_accum`: `C16_1616_Neumann` 最终数据。
- `chunk_gdr_c64_1616_sync_reduced_all_shapes`: `C64_1616_NeumannCompose` 最终数据。
- `chunk_gdr_c64_3232_hybrid_all_shapes`: `C64_3232_NeumannCompose` 最终数据。

## 汇总规则

- 使用 `summarize_chunk_gdr_performance_msprof.py` 读取 `op_summary_*.csv`。
- 目标行取 Phase4 measurement，即目标 op type 最后一次出现。
- `performance_compare.csv` 使用三行 long format：normal、neumann、`neumann/normal` ratio。
- ratio 字段只保留必要项：`aic_mac_ratio` 和 `aiv_vec_ratio`；其他时间类字段直接用 time ratio。

## 必提指标

最终汇总至少包含：

- `task_duration_us`
- `aicore_time_us`
- `aic_mac_time_us`
- `aic_mac_ratio`
- `aic_scalar_time_us`
- `aic_mte1_time_us`
- `aic_mte2_time_us`
- `aic_fixpipe_time_us`
- `aic_icache_miss_rate`
- `aiv_time_us`
- `aiv_vec_time_us`
- `aiv_vec_ratio`
- `aiv_scalar_time_us`
- `aiv_mte2_time_us`
- `aiv_mte3_time_us`
- `aiv_icache_miss_rate`

## 最终数据路径

- `C16_1616_Neumann`: `tests/pytest/prof_out/chunk_gdr_prefill_all_shapes_aic_accum/performance_compare.csv`
- `C64_1616_NeumannCompose`: `tests/pytest/prof_out/chunk_gdr_c64_1616_sync_reduced_all_shapes/performance_compare.csv`
- `C64_3232_NeumannCompose`: `tests/pytest/prof_out/chunk_gdr_c64_3232_hybrid_all_shapes/performance_compare.csv`

## 注意事项

- 每轮更换 kernel 方案后必须重新 build/install，再跑 msprof。
- 不要把 smoke、cacheline、tail_specialized、current 等中间目录当作最终结论。
- 用户未要求推送时不要 `git push`。
