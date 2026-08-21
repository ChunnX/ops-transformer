# quantMatmulReduceScatter Profiling 数据归档

> 采集日期: 2026-08-21
> 平台: Ascend 950PR (dav-3510), CANN 9.2.0
> 代码版本: 方案B — 后R核通信 + 前(28-R)核计算 (严格分离, winOffset=128, totalJobs=rankSize)

## 目录结构

```
profiling/
├── README.md                              # 本文件
├── st_direct/                             # ST直调 (Kernel<<<>>>直调, 方案B)
│   ├── msprof_summary.log                 # 12 case msprof 解析汇总 (中位数, 跨rank max)
│   └── raw/                               # msprof op_summary 原始 CSV (仅保留 csv, 删除 db/json/log)
│       ├── m8192k1536n12288r2/            # 每 case 一个目录
│       │   └── op_summary_dev{0..R-1}.csv # 每个 device 的 op_summary (10 次迭代, 含 pipe 级指标)
│       └── ... (12 个目录, 共 36 个 CSV)
├── mc2_test/                              # 融合算子路径 (aclnnMatmulReduceScatterV2)
│   ├── aclnnMatmulReduceScatterV2_performance_result.csv  # 完整结果 (mc2_prof + golden_prof 列)
│   ├── mc2_aclnnMatmulReduceScatterV2_pro_result.csv      # mc2 专用结果
│   └── raw/                               # 原始 op_summary CSV (torch profiler + msprof 采集)
│       └── m{M}k{K}n{N}r{R}/              # 同 st_direct/raw 结构, 与 ST直调同口径 pipe 级对比
├── hccl_mm/                               # 分步路径 (aclnnQuantMatmulV5 + HcclReduceScatter)
│   ├── hccl_mm_aclnnMatmulReduceScatterV2_pro_result.csv  # hccl_mm 专用结果 (含时间分解)
│   └── raw/                               # 原始 op_summary CSV (含 matmul + HcclReduceScatter 两个 op)
│       └── m{M}k{K}n{N}r{R}/              # hccl_mm 的 op_summary 含两个 op, 可分析 matmul 与通信时间分布
└── comparison/
    └── mc2_test_vs_st_comparison.log      # 三路径性能对比报告
```

## 12 Case 性能汇总

| Case | mc2_test (us) | hccl_mm (us) | ST直调 (us) | mc2/ST | hccl/ST |
|------|-------------|-------------|------------|--------|---------|
| 8192x1536x12288 R=2 | 2768.0 | 2970.8 | 2467 | 1.12x | 1.20x |
| 8192x1536x12288 R=4 | 1798.6 | 1716.8 | 1383 | 1.30x | 1.24x |
| 8192x3696x8192 R=2 | 1954.7 | 2472.2 | 1758 | 1.11x | 1.41x |
| 8192x3696x8192 R=4 | 1337.5 | 1653.2 | 1317 | 1.02x | 1.25x |
| 8192x1024x8192 R=2 | 1896.4 | 1880.4 | 1626 | 1.17x | 1.16x |
| 8192x1024x8192 R=4 | 1220.4 | 1058.9 | 926 | 1.32x | 1.14x |
| 4096x3584x4096 R=2 | 568.5 | 588.1 | 469 | 1.21x | 1.25x |
| 4096x3584x4096 R=4 | 451.6 | 410.1 | 305 | 1.48x | 1.34x |
| 2048x3584x4096 R=2 | 331.0 | 305.0 | 268 | 1.24x | 1.14x |
| 2048x3584x4096 R=4 | 296.1 | 232.1 | 176 | 1.68x | 1.32x |
| 4032x8192x4096 R=2 | 778.0 | 798.4 | 570 | 1.37x | 1.40x |
| 4032x8192x4096 R=4 | 625.1 | 620.9 | 612 | 1.02x | 1.01x |

- **mc2/ST 平均**: 1.24x (ST直调比融合算子快 1.24x)
- **hccl/ST 平均**: 1.23x (ST直调比分步路径快 1.23x)

## 采集方法

### ST直调 (st_direct/)

```bash
# MC² 多 rank 算子专用采集 (task-based, PipeUtilization)
# 前置: 先用 gen_data.py 生成各 case 测试数据到 data_m{M}k{K}n{N}r{R}/
msprof --output={prof_dir} \
       --ai-core=on \
       --aic-mode=task-based \
       --aic-metrics=PipeUtilization \
       --task-time=on \
       --ascendcl=on \
       ./quant_matmul_reduce_scatter_st {M} {K} {N} {R} perf {headMSize}
```

- 自适应 headMSize: R=2 → 512, R=4 → 128
- perf 模式循环 10 次, 前 5 次 warmup 丢弃
- 稳态取值: 后 5 次 Task Duration **中位数** (排除 msprof spike 干扰)
- 跨 rank 取 max (木桶效应)
- 采集后只保留 `op_summary_*.csv`, 删除 db/json/log 中间文件 (节省 99% 空间)

一键脚本: `bash run_12cases_msprof.sh`

### mc2_test (mc2_test/)

```bash
# 融合算子路径 (aclnnMatmulReduceScatterV2, MC2 通算融合)
# 采集方式: torch_npu.profiler.profile 上下文中循环 10 次, 取稳态
python3 mc2_test.py --xlsx {excel} --scene one_hccl_mm --world_size {R}
```

- 汇总数据: `aclnnMatmulReduceScatterV2_performance_result.csv` 的 `mc2_prof` 列
- 原始 op_summary: `raw/m{M}k{K}n{N}r{R}/op_summary_dev{0..R-1}.csv`
  - 来源: torch profiler 采集的 PROF 层 `mindstudio_profiler_output/op_summary_*.csv`
  - 与 ST直调同结构, 支持三路径 pipe 级对比
  - 12 个对比用例的 case_name 后缀为 `rs_st`
  - 注: `m8192k3696n8192r4` 的 dev3 原始采集缺失 op_summary (仅 3 个 device)

### hccl_mm (hccl_mm/)

```bash
# 分步路径 (aclnnQuantMatmulV5 matmul + HcclReduceScatter 通信)
python3 mc2_test.py --xlsx {excel} --scene one_hccl_mm --world_size {R}
```

- 汇总数据: 同 CSV 的 `golden_prof` 列, `hccl_mm_aclnnMatmulReduceScatterV2_pro_result.csv` 含时间分解 (avg_hccl + avg_matmul + avg_aicpu)
- 原始 op_summary: `raw/m{M}k{K}n{N}r{R}/op_summary_dev{0..R-1}.csv`
  - hccl_mm 的 op_summary 含**两个 op** (aclnnQuantMatmulV5 matmul + HcclReduceScatter), 可分析 matmul 与通信的时间分布

## op_summary CSV 字段说明

每个 `op_summary_dev{N}.csv` 包含 10 行 (10 次迭代), 关键列:

| 列名 | 说明 |
|------|------|
| Device_id | 设备 ID (0-3) |
| Op Name | 算子名 (QuantMatmulReduceScatterKernelE4M3E4M3) |
| Task Duration(us) | 端到端 Task 时间 (关键指标) |
| aicore_time(us) | AIC 核时间 |
| aic_mac_time(us) | MAC 流水线时间 |
| aic_mte2_time(us) | MTE2 流水线时间 (内存读取, 主要瓶颈) |
| aic_fixpipe_time(us) | FixPipe 时间 |
| aiv_time(us) | AIV 核时间 |
| aiv_vec_time(us) | Vector 流水线时间 |
| cube_utilization(%) | Cube 利用率 |
