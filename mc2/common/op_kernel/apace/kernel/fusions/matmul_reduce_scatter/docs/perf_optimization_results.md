# quantMatmulReduceScatter 性能优化结果报告

> **最新更新**: 2026-09-22（三轮评审后 v4 全量 101 用例精度回归 + 15 用例性能复测，见第六节）
> **历史更新**: 2026-09-12（97 用例完整对比 + 隔离测试，见第五节）
> **历史基线**: 2026-08-25（方案B 定版，12 用例，历史数据已随早期 profiling 清理，仅存结论）
> 平台: Ascend 950PR (dav-3510), CANN 9.2.0
> 代码版本: 方案B — 后R核通信 + 前(28-R)核计算 (严格分离, winOffset=128, totalJobs=rankSize)

## 零、三轮评审后 v4 版本 (2026-09-22)

### 变更内容 (相对 v2)

| 变更 | 来源 | 说明 |
|------|------|------|
| tid flag | 三轮评审意见1 | remote tile 通知 flagid 直接用 tid（替代 t%11 轮转），T ≤ 14 |
| tiling 切换 L1 排空 | 三轮评审意见3 | 段间/段内 SetFlag/WaitFlag(MTE1_MTE2) 排空 L1 |
| 并行跨卡 fence | 三轮评审意见2 | teamBarrier totalJobs=rankSize，R 核各 fence 1 个远端；win 数据区起点 alignUp(2R*32,128) |
| 段级 tiling | v2 存量 bug 修复 | 每段 tiling m = 段实际逻辑 M（REMOTE (R-1)*tileM / LOCAL tileM） |
| reduceSum slot 固定布局 | v2 存量 bug 修复 | slot 布局按 maxSegLen 固定，消除尾段收缩重叠竞态 |

### 精度回归 (本机 Ascend915, CANN 9.2.0)

**101/101 全部 PASS**（含 v2 曾崩溃的 m=4032 tail 系列 2 例与 n=16384/n=10240 k 族 22 例）。

### 性能复测 (15 代表用例, msprof task-based, 10 iter / 3 warmup / 7 valid, 跨 rank 取均值)

| M | K | N | R | headM | v4(us) | v1 历史(us)* | v4/v1 | 备注 |
|------|------|-------|---|-------|---------|-------------|--------|------|
| 8192 | 1536 | 12288 | 2 | 512 | 2845.0 | 2415.7 | +17.8% | head 未变 |
| 8192 | 1536 | 12288 | 4 | 256 | 1782.7 | 1357.6 | +31.3% | head 128→256 (tid flag 上限) |
| 4096 | 3584 | 4096 | 2 | 512 | 526.0 | 451.2 | +16.6% | head 未变 |
| 4096 | 3584 | 4096 | 4 | 128 | 380.8 | 300.3 | +26.8% | head 未变 |
| 4032 | 8192 | 4096 | 2 | 512 | 662.6 | 575.2 | +15.2% | v2 崩溃, v4 修复 |
| 4032 | 8192 | 4096 | 4 | 128 | 652.3 | 597.0 | +9.3% | v2 崩溃, v4 修复 |
| 4096 | 5120 | 640 | 2 | 512 | 161.3 | 147.7 | +9.2% | 窄 N |
| 2048 | 2048 | 16384 | 2 | 512 | 971.5 | 951.8 | +2.1% | v2 崩溃, v4 修复 |
| 4096 | 256 | 16384 | 2 | 512 | 1854.1 | 1648.9 | +12.4% | v2 崩溃, v4 修复 |
| 6144 | 1536 | 12288 | 2 | 512 | 2149.0 | 1831.3 | +17.3% | head 未变 |
| 6144 | 1536 | 12288 | 4 | 128 | 1442.2 | 1029.8 | +40.1% | head 未变 |
| 8192 | 4096 | 12288 | 2 | 512 | 2906.7 | 2529.3 | +14.9% | head 未变 |
| 16384 | 512 | 4096 | 2 | 1024 | 1994.5 | 1613.9 | +23.6% | head 512→1024 |
| 16384 | 4096 | 4096 | 4 | 512 | 1408.6 | — | — | v2 新增(崩溃), 无基线 |
| 32768 | 4096 | 4096 | 4 | 1024 | 2601.1 | — | — | v2 新增(崩溃), 无基线 |

\* v1 历史数据采集自 **另一台主机**（Ascend806 安装路径, 见 environment.md），跨机对比仅供参考。

### 与 v1 差异的归因分析 (待同机 A/B 验证)

1. **跨机环境差异**: v1 基线为 Ascend806 主机采集，本机为 Ascend915 路径，主机/驱动/频点状态不同，
   全量 +2%~+40% 的偏移不能直接归因于代码变更（未做同机 v2 A/B 对照）。
2. **headMSize 调整的独立影响**（3 例）: tid flag 4bit 空间约束 T ≤ 14，m=8192/R=4（head 128→256,
   T 16→8）、m=16384/R=2（512→1024）、m=16384/R=4（128→512）三例流水粒度变粗。
3. **段级 tiling 的 SWAT 参数变化**: v4 REMOTE 段 tiling 问题规模从 headMSize 变为 (R-1)*headMSize，
   SWAT 选择的 baseM/dbL0c/stepK 与 v2 subMmT 不同（如 row59: baseM 128→256、dbL0c 2→1），
   粗 tile 降低负载均衡度，为 row59 +40% 的可能主因，属可调优项（后续可对 remoteHead 段
   强制沿用小 baseM 策略复测）。
4. **结构性代价**: 3 处 L1 排空为正确性修复的必要代价（仅在 tiling 切换点，非逐 tile）。

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
| 逐行 ExecuteReduceSum | commTurn<=1 单 tile 路径（曾用批量接口 DoReduceSumBatch，因时序竞态已删除） |

### 方案B 定版性能（历史，12 用例，2026-08-25 采集）

> 注: 定版时 12 用例 vs 旧基线（totalJobs=1）平均 **-11.6%**（R=2 -7.9%, R=4 -15.3%），
> 最佳 case 4096x3584x4096 R=4 = -25.9%。原始数据已随早期 profiling 目录清理，结论保留。

### 方案A vs 方案B 对比

| 维度 | 方案A (通信核也计算) | 方案B (严格分离) |
|------|---------------------|-----------------|
| Phase 2 | 有 (通信核补做 reduceSum) | 无 |
| reduceSum 核数 | 28 | 28-R |
| 关键路径 | max(AT, RS_phase1) + RS_phase2 | max(AT, RS_strict) |
| B vs A 性能 | — | 平均 -7.5% |

方案B 消除 Phase 2 串行的收益 > 减少 reduceSum 核数的代价。

## 三、97 用例完整对比（最新，2026-09-11 采集）

用例来源：Excel level0/level0_4p（world≤4）97 用例，与级联基线、MC2 融合算子同环境对比。

详细报告：[profiling/reports/comparison_report.md](../../../tests/st/matmul_reduce_scatter/profiling/reports/comparison_report.md)

| 对比维度 | 结果 |
|------|-----|
| **vs 小算子级联基线** (npu_quant_matmul + reduce_scatter) | **93/97 更快**，中位 **1.23x**，avg 1.25x，max 1.65x |
| **vs MC2 融合算子** (ACLNN npu_quant_mm_reduce_scatter) | **88/97 更快**，中位 **1.14x**，avg 1.17x，max 1.69x |
| **隔离测试加速比** (T_serial/T_fused, 12 代表用例) | 中位 **1.20x**，max 1.51x |
| **流水掩盖率** (修正口径，12 代表用例) | 中位 **0.70**，通信 Bound 场景中位 0.83 |

Bound 分布（基于级联基线分解）：**70% 用例为通信 Bound**（R<0.67），融合掩盖收益显著。

## 四、已知劣势与根因

| 模式 | 用例 | 根因 | 性质 |
|---|---|---|---|
| **极窄 N=640** | 2/97 | URMA PUT 小消息固定开销 > HCCL；隔离测试证实融合同步开销 > 重叠收益（负掩盖率 -0.63，4 次复测确认） | 架构性 trade-off |
| **K=5472/6144 大 K × rank4** | 7/97 | matmul 效率差距 ~10%（K%64==32 padding + tiling 参数）；隔离测试证实 R≈1.5 计算 Bound，瓶颈在 matmul 本身 | 可优化项 |

## 五、当前性能瓶颈与未来方向

### 瓶颈分析（隔离测试更新，2026-09-12）

- 早期结论 "11/12 用例 MTE2 bound" 在当前 97 用例集下修正为：
  **70% 用例通信 Bound**（级联基线口径），融合后通信被流水掩盖（掩盖率中位 0.70）
- 剩余瓶颈集中在 **大 K 场景 matmul 效率**（K=5472/6144，隔离测试 R≈1.5）

### 未来优化方向

| 方向 | 预期收益 | 难度 |
|------|---------|------|
| **大 K tiling 调优** (baseK/stepK/scaleKL1, K>4608) | 追回 ~5-10% | 中 |
| **小 N 通信优化** (URMA PUT batching, N≤1024) | 消除 N=640 劣势 | 中 |
| **SplitK** | ~1.9x (每卡算 K/R, AIC 减半) | 高 (需 AllReduce 替代 ReduceScatter) |
| 框架修复 CrossDeviceExecute | 消除手动 CrossDevice 开销 | 中 (框架侧修改) |
| GET 模式 | 小 (PUT 已稳定) | 低 (4+rank 稳定性) |
| bias 支持 | 功能扩展 | 低 |
