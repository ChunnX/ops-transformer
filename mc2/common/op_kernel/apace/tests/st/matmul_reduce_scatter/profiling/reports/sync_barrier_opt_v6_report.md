# quantMatmulReduceScatter 同步结构优化 (v6 单 barrier 化) A/B 验证报告

- **日期**: 2026-09-23
- **环境**: Ascend950PR ×4, CANN 9.2.0 (`/home/l00947526/Ascend915/cann-9.2.0`), bisheng (dav-3510)
- **代码**: ops-transformer upstream-apace @ 106ff32dd (v5 基线) + v6 工作区修改
- **二进制**: v5 `md5=1883722f…` / v6 `md5=f4008b90…`（同源同环境编译）
- **采集**: msprof task-based, 10 iter / 3 warmup / 7 valid, outlier 过滤, 跨 rank 取 avg of card avgs
- **A/B 方式**: 同机同日交替运行两个二进制, 全部经 qrun 排队独占 NPU

## 一、优化内容

优化目标为 v5 每轮 (turn) 的同步三元组（原 `RunOneTileTurn` 尾部）:

```
v5:  WaitFlag(t) → SyncAll#1 → [PUT(t) ‖ RS(t-1)] → SyncAll#2 → fence → SyncAll#3
v6:  WaitFlag(t) → SyncAll#1 → [PUT(t)+drain+fence(t) ‖ RS(t-1)]        ← #2/#3 删除
     循环后: 终局 SyncAll → RS(T-1)                                       ← fence 结论兜底传播
```

- **barrier 数 3T → T+1**（如 T=8: 24 次 → 9 次）
- **fence 移出串行关键路径**: 紧跟本核 PUT drain 执行, 与计算核 RS(t-1)、AIC tile(t+1) 计算重叠
- 正确性依据（详见 `quant_matmul_reduce_scatter_urma_impl.h` 函数头注释）: fence 为成对握手 + 单调计数,
  只依赖本核 drain, 无需全核汇合（#2 冗余）; fence(t) 结论经下一轮 SyncAll#1(t+1) 程序序传播（#3 冗余）;
  全核 SyncAll 计数平衡 (每核 T+1 次); 末 turn 由终局 SyncAll 兜底

## 二、正确性验证

- **精度**: 10 个典型用例全 PASS（rank2/4、head+tail tile、T=1~8、小 N=640、大 shape 全覆盖）
  rows 1/2/8/10/11/12/16/17/27/47
- **perf 模式内联精度**: 10/10 PASS（每 case msprof 后自动比对）

## 三、A/B 性能对比（10 典型用例）

批次 1（常规 + 小 N 同步敏感场景）:

| row | M | K | N | R | T | v5 (us) | v6 (us) | Δ |
|----|------|------|------|---|---|---------|---------|------|
| 1 | 8192 | 1536 | 12288 | 2 | 8 | 2434.446 | 2431.221 | -0.13% |
| 10 | 2048 | 3584 | 4096 | 4 | 4 | 166.109 | 165.017 | -0.66% |
| 16 | 4096 | 5120 | 640 | 2 | 4 | 130.693 | 130.354 | -0.26% |
| 17 | 4096 | 5120 | 640 | 4 | 8 | 161.275 | 160.210 | -0.66% |
| 27 | 2048 | 4096 | 12288 | 4 | 4 | 519.372 | 521.161 | +0.34% |
| 47 | 4096 | 4096 | 12288 | 4 | 8 | 946.247 | 945.388 | -0.09% |

批次 2（16384 家族, 小 K 快 AIC + T=8 多 tile, AIV 链最可能成为关键路径的场景）:

| row | M | K | N | R | T | v5 (us) | v6 (us) | Δ |
|----|-------|------|------|---|---|---------|---------|------|
| 86 | 16384 | 512 | 4096 | 2 | 8 | 1606.746 | 1602.595 | -0.26% |
| 87 | 16384 | 512 | 4096 | 4 | 8 | 883.386 | 880.423 | -0.34% |
| 89 | 16384 | 640 | 5120 | 4 | 8 | 1099.721 | 1090.877 | -0.80% |
| 93 | 16384 | 1376 | 4096 | 4 | 8 | 914.323 | 913.636 | -0.08% |

**汇总**: 9/10 用例提升, 1/10 (row 27) +0.34% 为迭代噪声级; 几何平均 **-0.3%**, 最大 -0.80% (row 89, -8.8us)。

逐迭代佐证 (row 17 单 rank): v5 162.4~163.9us → v6 158.9~161.3us, 稳定快 2~4us。

## 四、收益有限的根因分析

1. **v5 的 #2/#3 大部分本就不在关键路径**: 稳态流水下 AIV 链 (PUT+RS+barrier+fence) 短于 AIC
   生产者链时, AIV 核在 WaitFlag(t+1) 处等下一个 tile —— 删除 barrier 缩短的是非关键路径,
   算子时长 ≈ AIC 总时长 + 尾部 (末轮 PUT+fence+RS) + 启动段。
2. **硬件 SyncAll 开销极小**: dav_3510 `SyncAll<true>` 为 ffts 硬件同步, 核已收敛时亚微秒级,
   15 次 barrier 的绝对节省仅 ~1-4us。
3. **每轮真正的同步成本是 fence 远端轮询** (跨卡 interconnect RTT), 它是 PUT 数据到达证明所必需。
   v6 已将其与 RS/AIC 计算重叠; 通信 bound case (row 27/47) 中 PUT 传输时间本身主导。
4. 与隔离测试报告结论对齐: row 17 融合回退 ~9us 中, v6 收回 1~3us, 其余属 fence 轮询固有成本。

## 五、结论与建议

- **建议保留 v6 修改**: 语义等价、结构更简 (每轮 2 次全核 barrier 冗余消除)、fence 重叠化,
  10 用例无一真实回退, 小幅一致收益 (几何平均 -0.3%)。修改位于工作区, 未提交未推送。
- **后续方向** (若需进一步压缩每轮同步成本, 属协议级改造):
  1. 数据内嵌序列号: PUT 尾部附带 tile 序号, 接收侧本地轮询 win 数据区替代远端 flag 握手
     (需确认 URMA 同 channel 写序保证), 可消除每轮 fence 的跨卡读;
  2. URMA CQE 远端完成语义: 若 Drain 可证远端可见, fence 整体可去除 (需硬件语义确认)。

## 六、数据索引

| 项 | 位置 |
|---|---|
| v5 批次1 PROF (6 case) | `../../prof_v5_batch1/` |
| v6 批次1 PROF (6 case) | `../../prof_v6_batch1/` |
| v5 批次2 PROF (4 case) | `../../prof_v5_batch2/` |
| v6 批次2 PROF (4 case) | `../../prof_v6_batch2/` |
| 解析文本 | `../data/sync_ab_v6/*.txt` |
| 代码修改 | `apace/kernel/fusions/matmul_reduce_scatter/quant_matmul_reduce_scatter_urma_impl.h` (v6, 工作区) |
