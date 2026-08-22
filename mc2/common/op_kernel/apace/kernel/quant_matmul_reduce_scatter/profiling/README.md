# quantMatmulReduceScatter Profiling 数据归档

> 采集日期: 2026-08-22
> 平台: Ascend 950PR (dav-3510), CANN 9.2.0
> 代码版本: 方案B — 后R核通信 + 前(28-R)核计算 (严格分离, winOffset=128, totalJobs=rankSize)
> 采集脚本: `tests/st/quant_matmul_reduce_scatter/run_12cases_msprof.sh`

## 目录结构

```
profiling/
├── README.md                              # 本文件
├── st_direct/                             # ST直调 (Kernel<<<>>>直调, 方案B)
│   ├── msprof_summary.log                 # 12 case msprof 解析汇总 (中位数, 跨rank max)
│   └── raw/                               # msprof op_summary 原始 CSV (仅保留 csv, 删除 db/json/log)
│       ├── m2048k3584n4096r2/             # 每 case 一个目录
│       │   └── op_summary_dev{0..R-1}.csv # 每个 device 的 op_summary (10 次迭代, 含 pipe 级指标)
│       └── ... (12 个目录, 共 36 个 CSV)
├── mc2_test/                              # 融合算子路径 (aclnnMatmulReduceScatterV2)
│   ├── aclnnMatmulReduceScatterV2_performance_result.csv  # 完整结果
│   ├── mc2_aclnnMatmulReduceScatterV2_pro_result.csv      # mc2 专用结果
│   └── raw/                               # 原始 op_summary CSV
│       └── m{M}k{K}n{N}r{R}/              # 同 st_direct/raw 结构
├── hccl_mm/                               # 分步路径 (aclnnQuantMatmulV5 + HcclReduceScatter)
│   ├── hccl_mm_aclnnMatmulReduceScatterV2_pro_result.csv  # 含时间分解
│   └── raw/                               # 原始 op_summary CSV (含 matmul + HcclReduceScatter 两个 op)
│       └── m{M}k{K}n{N}r{R}/
├── ANALYSIS.md                            # 三路径对比 + 加速比覆盖率 + 隔离测试分析
└── comparison/
    └── mc2_test_vs_st_comparison.log      # 三路径性能对比报告
```

> 注: `mc2_test/`、`hccl_mm/`、`comparison/`、`ANALYSIS.md` 为 cann_skills 仓同步数据，`st_direct/` 为本仓独立采集。

## 12 Case ST直调性能汇总

| Case | Task Duration (us) |
|------|--------------------|
| 2048x3584x4096 R=2 | 272.6 |
| 2048x3584x4096 R=4 | 179.6 |
| 4096x3584x4096 R=2 | 473.4 |
| 4096x3584x4096 R=4 | 308.2 |
| 4032x8192x4096 R=2 | 574.9 |
| 4032x8192x4096 R=4 | 613.5 |
| 8192x1024x8192 R=2 | 1623.4 |
| 8192x1024x8192 R=4 | 921.4 |
| 8192x3696x8192 R=2 | 1771.9 |
| 8192x3696n8192 R=4 | 1322.5 |
| 8192x1536x12288 R=2 | 2438.3 |
| 8192x1536x12288 R=4 | 1337.3 |

## 采集方法

### ST直调 (st_direct/)

```bash
# MC² 多 rank 算子专用采集 (task-based, PipeUtilization)
# 前置: gen_data.py 生成测试数据
cd tests/st/quant_matmul_reduce_scatter
bash run_12cases_msprof.sh
```

底层调用：
```bash
msprof --output={prof_dir} \
       --ai-core=on \
       --aic-mode=task-based \
       --aic-metrics=PipeUtilization \
       --task-time=on \
       --ascendcl=on \
       ./quant_matmul_reduce_scatter {M} {K} {N} {R} perf {headMSize}
```

- 自适应 headMSize: R=2 → 512, R=4 → 128
- perf 模式循环 10 次, 前 5 次 warmup 丢弃
- 稳态取值: 后 5 次 Task Duration **中位数** (排除 msprof spike 干扰)
- 跨 rank 取 max (木桶效应)
- 采集后只保留 `op_summary_*.csv`, 删除 db/json/log 中间文件

## op_summary CSV 字段说明

每个 `op_summary_dev{N}.csv` 包含 10 行 (10 次迭代), 关键列:

| 列名 | 说明 |
|------|------|
| Device_id | 设备 ID (0-3) |
| Op Name | 算子名 (QuantMatmulReduceScatterKernelE4M3E4M3) |
| Task Duration(us) | 端到端 Task 时间 (关键指标) |
| aicore_time(us) | AIC 核时间 |
| aic_mac_time(us) | MAC 流水线时间 |
| aic_mte2_time(us) | MTE2 流水线时间 (内存读取) |
| aic_fixpipe_time(us) | FixPipe 时间 |
| aiv_time(us) | AIV 核时间 |
| aiv_vec_time(us) | Vector 流水线时间 |
| cube_utilization(%) | Cube 利用率 |
