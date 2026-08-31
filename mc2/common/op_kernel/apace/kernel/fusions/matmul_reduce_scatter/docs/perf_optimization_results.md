# quantMatmulReduceScatter 性能优化结果报告

> 日期: 2026-08-25 (ST直调重新采集, 机器空闲)
> 平台: Ascend 950PR (dav-3510), CANN 9.2.0
> 代码版本: 方案B — 后R核通信 + 前(28-R)核计算 (严格分离, winOffset=128, totalJobs=rankSize)
> 数据来源: `profiling/st_direct/msprof_summary.log` (本仓独立采集, 8/25 22:03)

## 一、历史优化尝试 (totalJobs=1 旧版本)

以下优化在 totalJobs=1 单核通信版本上尝试，均无效或恶化：

### 优化1: dbL0c=2 (双缓冲 L0C)

**结果**: 持平 (284us → 285us)
**原因**: L0C 双缓冲占用 L0A/L0B 空间, 引发更多 L1 重载, fixpipe 反而增加

### 优化2: stepK=4 (K-steps 减半)

**结果**: 持平 (284us → 285us)
**原因**: MTE2 与 MAC 已高度重叠 (90%+), K-step 减少被其他开销抵消

### 优化3: baseN=128 (N-tiles 翻倍)

**结果**: 恶化 (+7%)
**原因**: N-tiles 翻倍导致 fixpipe 调用次数翻倍, 固定开销累加

### 优化4-6: 组合优化

**结果**: 均无效
**原因**: AIC MTE2 是不可逾越的计算瓶颈 (内存带宽受限)

## 二、多核并行通信 (方案B, 当前版本)

### 方案B 核心改进

| 改进 | 说明 |
|------|------|
| winOffset=128 | PUT 数据跳过 win 区前 128B barrier counter, 修复假通过隐患 |
| totalJobs=rankSize | 后 R 核各负责 1 个 target, 并行 PUT, 通信时间降为 1/R |
| teamBarrier totalJobs=1 | CrossDevice step=1 正确轮询所有 remote |
| Wait\<BARRIER_NONE\> | Drain only, 不依赖框架 CrossDevice |
| 手动 CrossDevice | SyncAll 后 block 0 调 teamBarrier_.CrossDevice() |
| 严格分离 | 后 R 核仅通信, 前 (28-R) 核仅计算, 无 Phase 2 串行 |
| rsCoreNum=28-R | reduceSum 核数从 28 降到 28-R |
| DoReduceSumBatch | commTurn<=1 时批量 reduceSum, 减少 flag 操作次数 |

### msprof 12 case 性能数据 (Task Duration 中位数, 跨 rank max)

| Case | 旧基线 (us) | 方案B (us) | 变化 |
|------|------------|-----------|------|
| 8192x1536x12288 R=2 | 2669 | 2435.5 | -8.7% |
| 8192x1536x12288 R=4 | 1696 | 1336.3 | -21.2% |
| 8192x3696x8192 R=2 | 1947 | 1772.0 | -9.0% |
| 8192x3696x8192 R=4 | 1328 | 1317.8 | -0.8% |
| 8192x1024x8192 R=2 | 1799 | 1622.5 | -9.8% |
| 8192x1024x8192 R=4 | 1207 | 918.7 | -23.9% |
| 4096x3584x4096 R=2 | 526 | 472.3 | -10.2% |
| 4096x3584x4096 R=4 | 415 | 307.5 | -25.9% |
| 2048x3584x4096 R=2 | 286 | 272.6 | -4.7% |
| 2048x3584x4096 R=4 | 222 | 175.6 | -20.9% |
| 4032x8192x4096 R=2 | 604 | 575.0 | -4.8% |
| 4032x8192x4096 R=4 | 609 | 615.4 | +1.1% |

> 注: 旧基线数据来自 cann_skills 仓 totalJobs=1 版本历史采集；方案B数据来自本仓 `profiling/st_direct/msprof_summary.log` (8/25 重新采集)

- **R=2 平均**: -7.9%
- **R=4 平均**: -15.3%
- **总体平均**: -11.6%
- **最佳 case**: 4096x3584x4096 R=4 = 307.5us vs 基线 415us (-25.9%)
- **唯一退化**: 4032x8192x4096 R=4 = 615.4us vs 基线 609us (+1.1%), 原因: headMSize=128 (tileCnt=7) 非最优, SyncAll 过多 + GM 带宽争抢。headMSize=252 (tileCnt=4) 验证 TD=506.6us (-16.8%), 覆盖率从 0.07 提升至 0.54。全 6 R=4 case headMSize 搜索完成, 仅 4032 R=4 显著受益, 其余 case headMSize=128 基本合理 (改善 <1%) (详见 `profiling/ANALYSIS.md` 6.6 节)

### 方案A vs 方案B 对比

| 维度 | 方案A (通信核也计算) | 方案B (严格分离) |
|------|---------------------|-----------------|
| Phase 2 | 有 (通信核补做 reduceSum) | 无 |
| reduceSum 核数 | 28 | 28-R |
| 关键路径 | max(AT, RS_phase1) + RS_phase2 | max(AT, RS_strict) |
| B vs A 性能 | — | 平均 -7.5% |

方案B 消除 Phase 2 串行的收益 > 减少 reduceSum 核数的代价。

## 三、三路径性能对比

三路径对比数据（mc2_test 融合算子路径、hccl_mm 分步路径、ST 直调路径）来自 cann_skills 仓同步，详见 `profiling/comparison/mc2_test_vs_st_comparison.log` 和 `profiling/ANALYSIS.md`。

| 指标 | 值 |
|------|-----|
| mc2/ST (融合算子/直调) | 平均 1.25x |
| hccl/ST (分步/直调) | 平均 1.24x |
| ST直调 vs mc2_test | 快 1.25x |
| ST直调 vs hccl_mm | 快 1.24x |

## 四、当前性能瓶颈与未来方向

### 瓶颈: AIC MTE2 (内存带宽)

- 11/12 用例为 MTE2 bound (MTE2/MAC > 1.0)
- AIV 完全被 AIC 掩盖 (82% 时间在 WaitFlag)
- tiling 参数 (baseM/baseN/baseK/stepK/dbL0c) 已穷尽, 无法突破 MTE2 瓶颈

### 未来优化方向

| 方向 | 预期收益 | 难度 |
|------|---------|------|
| **SplitK** | ~1.9x (每卡算 K/R, AIC 减半) | 高 (需 AllReduce 替代 ReduceScatter) |
| 框架修复 CrossDeviceExecute | 消除手动 CrossDevice 开销 | 中 (框架侧修改) |
| GET 模式 | 小 (PUT 已稳定) | 低 (4+rank 稳定性) |
| bias 支持 | 功能扩展 | 低 |
