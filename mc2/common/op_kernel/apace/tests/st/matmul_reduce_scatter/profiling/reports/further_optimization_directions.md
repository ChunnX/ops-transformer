# quantMatmulReduceScatter 后续性能优化方向分析

- **日期**: 2026-09-23（基于 v6 A/B 数据 + 隔离测试报告 + op_summary 流水指标）
- **数据来源**: `sync_ab_v6_report.md`（10 case A/B）、`isolation_metrics_report.md`（四分量分解）、
  v6 PROF `op_summary` 逐 case 流水利用率（`prof_v6_batch1/2`）
- **环境**: Ascend950PR ×4, CANN 9.2.0, msprof task-based（PipeUtilization 指标）

## 一、瓶颈证据链（v6 实测）

### 1. AIV 是全局关键路径

10 个 case 全部满足 `aiv_time ≈ Task Duration`（98~99.5%），AIC 均更早完成。
**任何只优化 AIC/matmul 的改动，只在计算 bound case 上可见。**

### 2. AIV 大部分时间在等待（idle 42~69%）

| case 族 | AIV idle | AIV vec | AIV mte2 | AIC mte2 |
|---|---|---|---|---|
| row1/86 (rank2 大通信量) | 62~69% | 8% | 18~23% | 64~80% |
| row10/27/47 (rank4 常规) | 54~63% | 8~9% | 24~33% | 86~95% |
| row16/17 (小 N=640) | 42~50% | 7~10% | 35~37% | 90~91% |
| row87/89/93 (小 K 大 M) | 32~52% | 13% | 42~47% | 69~92% |

AIV 空闲 = WaitFlag(AIC) + SyncAll + fence 轮询 + MTE2 深度不足的气泡。
有效工作中**搬运(mte2)是计算(vec)的 2~5 倍** —— RS 访存与 fence 轮询主导 AIV。

### 3. 单 UDMA 通道 ~40GB/s 封顶（结构性瓶颈）

由隔离报告 T_a2a 反推每通道有效带宽（每目标 rank 1 通道、1 通信核）:

| case | 每通道数据量 | T_a2a | 有效带宽 |
|---|---|---|---|
| row1 (8192,1536,12288,r2) | 96MB | 2269.6us | **42GB/s** |
| row13 (4096,640,5120,r2) | 21MB | 490.8us | **43GB/s** |
| row86 (16384,512,4096,r2) | 64MB | ~1602us | **40GB/s** |
| row87 (16384,512,4096,r4) | 32MB | 813.9us | **39GB/s** |
| row17 (4096,5120,640,r4) | 1.31MB | 103.6us | 13GB/s（小消息延迟受限） |

rank2 case 通信量全部走单通道 → 通信时间 ≈ 数据量/40GB/s，这是 row1/86 等
通信 bound case 的硬上限；r4 case 3 通道聚合也仅 ~120GB/s。

### 4. AIC 侧访存接近饱和（次要路径）

AIC mte2_ratio 80~95%（A/B 矩阵加载），大 K case 尤甚（row93: 92%）。
仅对计算 bound case（row17: cube 80%、row67/81/97）端到端可见。

## 二、优化方向（按优先级）

### P0: PUT 通道带宽扩展 — 通信 bound case 结构性收益（预期 rank2 case ~30~40%）

- **方案**: 每目标 rank 开多条 UDMA channel（tile 按行块拆分到 2~4 通道），或通信核分配从
  1 核/目标改为 K 核/目标（K 通道并行）
- **依据**: 单通道 40GB/s 封顶；row1 T_a2a=2269.6us → 双通道理论上 ~1150us，
  端到端 2431→~1400us
- **风险**: 中。需确认 hcomm 框架 channel 配额与创建方式；通信核增多挤占 RS 计算核
  （rsCoreNum = usedCoreNum - rankSize 需重新权衡）
- **验证**: 先在 row1/86/13 三个 rank2 case 上 A/B

### P1: PUT 深度流水化 — 消除通道空闲（预期通信 bound case 5~10%）

- **方案**: ① Commit 提前 + drain 惰性化：PUT(t+1) 在 PUT(t) drain 之前 post
  （URMA 队列天然按序），仅在 fence 前统一 drain；② fence 后移一轮：fence(t) 与
  PUT(t+1) 传输完全重叠，RS(t) 推迟至 turn t+2 消费（流水深度 +1，结构不变）
- **依据**: 当前 `Commit(t)→Wait(t)→fence(t)` 串行，drain/fence 期间通道空闲；
  AIV idle 30~69% 说明流水气泡充足
- **风险**: 低-中。协议不变、纯时序重排；fence 传播链论证需按 v6 注释框架重做
  （fence(t) 须在 RS(t) 前到达，即 #1(t+2) 之前）
- **验证**: row27/47（fence 占比高的 r4 通信 bound）

### P1: 小 N case tile 尺寸策略（预期 row16/17 通信段 10~25%）

- **方案**: tiling 侧对 N 小的 case 增大 headMSize（N=640 时 win 仅 ~10MB 远未占满，
  tileM=128 导致每轮 PUT 仅 160KB、T=8 轮固定开销占比过高）
- **依据**: row17 单通道仅 12.6GB/s（vs 40GB/s 能力）——小消息延迟受限；
  隔离报告 row17 融合回退 -0.63 的根因
- **风险**: 低。纯 tiling 参数，但 headMSize 同时是 matmul tile 粒度（受 AIC L1 约束），
  需联调重跑精度
- **验证**: row16/17 + 全量回归

### P2: RS 访存效率（预期均衡 case 5~15%，通信 bound 收益被掩盖）

- **方案**: RS 读块增大（UB 双缓冲 + 更大 MTE2 burst）；win 布局按消费序重排
  （rank 维连续 → 大块读，当前 rowStride 跨度读）；对齐消除碎片读
- **依据**: AIV mte2 18~47% busy 但 vec 仅 7~14%；row47 RS 读带宽 ~170GB/s
  远低于 HBM 峰值 —— 延迟/块大小受限而非带宽受限
- **风险**: 低。RS 是独立组件（reduce_sum_ref.h），不动同步协议
- **验证**: row87（T_rs=474us/1346us serial 占比最高）+ row8/10（均衡）

### P2: AIC 侧 Blaze kernel 访存调优（预期计算 bound case 5~10%）

- **方案**: A 矩阵加载块增大 / L2 复用布局；baseK/stepK/scaleKL1 tiling 调优
  （承接 deep_analysis 报告既有结论：K=5472/6144 matmul 效率差 ~10%）
- **依据**: AIC mte2_ratio 80~95%；row17 cube 80% 为最高计算占比 case
- **风险**: 低-中。Blaze kernel 内部参数，有既有调优方法论（aiss-tiling-solver）
- **验证**: row17/67/81/97 + 既有 97 case 大 K 子集

### P3: fence 协议本地化 — 数据内嵌序列号（每轮省几 us，与 P1 叠加）

- **方案**: PUT 尾部附带 tile 序号 word（数据+序号同 channel 两次 WriteNbi，依赖同通道
  写序保证），接收侧本地轮询 win 尾字替代远端 flag 握手，fence 变纯本地 GM 读
- **依据**: v6 已将 fence 重叠化，但通信核到达 #1(t+1) 时刻仍受 fence 尾延制约；
  row17 逐迭代仍有 2~4us/轮量级同步残值
- **风险**: 中-高。需 URMA 同通道写序硬件语义确认；改动共享 TeamBarrier/AllToAll 协议层，
  波及 all_to_all/all_gather 两个兄弟算子，需全量回归
- **前置**: 建议 P0/P1 落地后评估残余同步占比再决定是否投入

## 三、建议实施路线

```
第一批 (低风险快赢):   P1-PUT深度流水化 + P1-小N tile策略      → 预期整体 3~8%
第二批 (结构性大收益):  P0-多通道 PUT 扩展                     → 预期 rank2 case 30%+
第三批 (按 case 选择):  P2-RS 访存 + P2-AIC tiling             → 预期特定族 5~15%
观察项:               P3-fence 本地化 (视前两批残余同步占比)
```

每批实施后以本报告同口径（msprof task-based、10 iter、A/B 双二进制、典型 case 集
row1/10/16/17/27/47/86/87/89/93）复测对比。
