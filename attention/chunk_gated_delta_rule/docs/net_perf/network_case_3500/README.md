# B1 / T3500 / Nk16 / Nv32 / gate

[完整分析报告](../network_case_3500_20260908.md)记录正式测量：CANN原版898.227µs、当前优化版148.505µs，20次预热+100次测量，6.048×加速。

## 流水图入口

| 图 | 展示范围 |
|---|---|
| [原优化版片段图](figures/pipeline_timeline.png) | core0核组的8条选定管线；0～20µs、50～65µs两个片段 |
| [CANN原版时间线](figures/pipeline_timeline_baseline.png) | core0核组的13条管线；完整trace及相同的两个片段 |
| [原版/优化版完整对照](figures/pipeline_timeline_comparison_full.png) | 两版完整trace，13条管线一致，各用自己的横轴刻度 |
| [相同窗口放大对照](figures/pipeline_timeline_comparison_zoom.png) | 上排原版、下排优化版；左列0～20µs、右列50～65µs |

新图均有同名PDF，便于放大。两版原始trace都采样core0～5共6个核组，图中只画core0，不能当作32个核组的全局汇总。

每行表示一条硬件管线，色块表示记录到的活动区间；横向重叠表示不同管线有并行活动。蓝色为矩阵计算，绿色为向量计算，紫色为输入搬运，橙色为Cube结果转换/搬出，粉色为UB输出搬运，灰色为Scalar控制。空白只表示该管线没有活动记录，不能直接定位某个等待信号。一个色块不对应固定的一个chunk或一个VF函数。

**原图是片段图。** 0～20µs和50～65µs来自同一次被采集kernel；两个窗口并不是两次独立调用。新增原版50～65µs窗口仍是早期工作，不能假设它与优化版处在相同算法进度。

**trace轴与算子耗时口径不同。** 本次原版trace末事件约1430.864µs，OpBasicInfo为878.439µs；优化版分别为140.033和153.321µs。保留原始时间戳，只比较流水结构，不按trace范围推导加速比。图中横轴未拉伸，也未归一化。正式性能仍使用上面的msprof重复测量值。

## 数据

- [逐轮统计](comparison.csv)、[逐次样本](timing_samples.csv)、[PMU逐角色汇总](core_metrics.csv)。
- [时间线来源、哈希和窗口元数据](timeline_comparison.json)、[本轮时间线审计](timeline_validation.json)。
- [源码分析](static_analysis.md)、[msopprof审阅](msopprof_review.md)、[gate消融](probe_summary.json)。

原始trace.json、visualize_data.bin及绘图脚本保留在算子`tests/pytest/output/rdv_perf_20260908/network_case_3500/`，具体路径见元数据。
