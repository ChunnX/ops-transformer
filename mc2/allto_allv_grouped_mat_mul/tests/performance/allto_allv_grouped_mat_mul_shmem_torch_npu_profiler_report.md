# AllToAllV GroupedMatMul SHMEM 性能评估结果

**状态**：NOT_RUN（当前无可用 Ascend950 SHMEM 多卡编译/执行环境）

本算子是 aclnn C++ 多进程 SHMEM 接口，当前仓库没有等价 PyTorch 注册，因此本轮没有伪造
`torch_npu.profiler` 数据。计划采集固定使用 `warmup=5`、`active=5`、`repeat=1`，每步调用
`prof.step()`；自定义路径与源 `mingzhea2a_udma` 路径分别独立采集。

## 性能对比

| Case | Shape | DType | 自定义算子(us) | 标杆(us) | 加速比 |
|---|---|---|---:|---:|---:|
| 全部 10 条计划用例 | 见 JSONL | bfloat16 | N/A | N/A | N/A |

## 全量汇总

| 指标 | 值 |
|---|---:|
| 计划用例数 | 10 |
| 已执行用例数 | 0 |
| 平均加速比 | N/A |
| 自定义算子更优 | N/A |
| 标杆更优 | N/A |

### 按数据类型汇总

| DType | 计划用例数 | 已执行 | 平均加速比 | 自定义算子更优 | 标杆更优 |
|---|---:|---:|---:|---:|---:|
| bfloat16 | 10 | 0 | N/A | N/A | N/A |

## 简短分析

- 当前仅完成性能用例元数据与采集口径定义，没有可支持性能结论的实测数据。
- 用例覆盖 EP=1/2/4/8、两个 M0 tiling 家族、源参考 shape、残块、零 peer 和高度不均匀路由。
- 实测时应同时保留迁移实现和源实现的双路径数据，不能以单路径耗时替代对比报告。
- 指标取各自 `op_statistic.csv` 的 `Total Time(us)` 总和除以 5，并保留通信、GMM 和总耗时（trace 可区分时）。
