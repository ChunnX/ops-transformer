---
name: matmul-reduce-scatter-v2-intro
description: 小白向的MatmulReduceScatterV2算子入门讲解
metadata:
  type: reference
---

# MatmulReduceScatterV2 算子入门笔记

## 一句话
把"矩阵乘法(AIC计算)" + "跨卡数据求和分发(AIV通信)" 融合成一个流水线操作，专门用于Ascend NPU上的MoE模型分布式训练。

## 核心架构
- **AIC核心**（AI Core）：做矩阵乘法（使用Catlass模板库），类似"重型机床"
- **AIV核心**（AI Vector Core）：做Padding、反量化、ReduceScatter通信，类似"传送带和叉车"
- **双缓冲(Ping-Pong)**：AIC算完一块 → AIV立刻处理 → AIC算下一块，计算和通信完全重叠

## 5阶段流水线
1. **Init**：初始化参数、地址映射
2. **Padding**（AIV）：对不满足512B对齐的矩阵补零
3. **CatlassMatmul**（AIC）：分块矩阵乘法，写入peermem
4. **Dequant**（AIV）：int32→float16反量化（per-channel/per-token）
5. **ReduceScatter**（AIV）：从其他rank读取数据，原子累加到本地输出

## 两种模式
| 模式 | 条件 | 特点 |
|------|------|------|
| 大M(默认) | M>2048或N≤512 | CrossRankSyncV1/V2屏障同步，所有AIV相同分工 |
| 小M(SmallM) | M≤2048且N>512 | AIV分两组(通知组/搬运组)，按列分块通信，预初始化输出为0 |

## 关键技术
- **CRTP模式**：父类模板参数=子类，避免虚函数开销
- **Swizzle分块调度**：特殊访问顺序提高L1 Cache命中率
- **Peermem**：每卡共享内存区域，其他卡可直接读取
- **Flag同步**：跨卡通过peermem末尾区域的flag值做信号同步
- **Tiling调优**：Host侧根据M/K/N/rankSize查Code Map找最优分块参数

## 重点文件
- `op_kernel/matmul_reduce_scatter_aiv_mode.h` — 大M主类
- `op_kernel/matmul_reduce_scatter_aiv_mode_smallM.h` — 小M子类
- `op_kernel/matmul_reduce_scatter_aiv_mode_util.h` — 通信工具(CommBase等)
- `op_host/op_tiling/arch32/matmul_reduce_scatter_v2_aiv_mode_tiling.cpp` — 参数调优

**学习路径建议**：README.md → util.h（基础设施） → aiv_mode.h（主流程） → smallM.h（变体）
