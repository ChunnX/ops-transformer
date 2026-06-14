# 昇腾950 NPU架构白皮书硬件导读

> 来源:昇腾950 NPU架构白皮书.pdf
> 用途:作为 fa_arch35_design.md 的硬件依据索引。筛出对 arch35 FlashAttention / FlashAttentionGrad 读码有帮助的硬件事实,并补上“为什么影响 FA/FAG”。
> 注:本文为配套硬件事实索引,正文量化推导见 fa_arch35_design.md §2.5。详细页码与逐章笔记以你本地抽取版为准;此处保留供设计书交叉引用的稳定锚点。

## 0. 一页结论(对 FA/FAG 最有价值的硬件约束组合)

| 硬件事实 | 白皮书位置 | 对 FA/FAG 的意义 | 代码落点 |
|---|---|---|---|
| 每个 AI 子系统 = 1 Cube Core + 2 Vector Core;完整 36 个 AI 子系统 | p13 | 1C:2V 发块、单 AIV 吃半 S1 block | `KERNEL_TYPE_MIX_AIC_1_2`, `CV_RATIO=2` |
| UB=512KB,L1=512KB,L0A/B=64KB,L0C=256KB per AI Core | p25 | block size、N-buffer、BMM2 写 UB/GM、FAG DKV L0C resident | `GetC2Position`, `IS_DKV_RESIDENT_L0C` |
| L2 up to 128MB,512B line,4*128B sector,Cache Hint/CMO,2-Die UMA 但有局部亲和 | p15,p25-27 | splitCoreMode 围绕 L2 工作集与 K/V 时间局部性设计 | `CalcThresholdForS2Size` |
| 片上内存 PR 128GB/1.6TB/s;DT 144GB/4TB/s | p10,p13-14,p26 | MTE/HBM bound 背景;单核有效带宽需 profile | §4.3 / §2.5 |
| Cube 原生 HiF8/MXFP8/FP8(=2× FP16)、MXFP4(=4×) | p17 | FP8/HIFLOAT8 低精度模板与特殊 block size | quant block |
| Vector 双发射 Register-Based SIMD,SIMD/SIMT 混合,VF 可选 SIMD/SIMT | p11,p21-22 | mask/tail/dropout 可能免物化;VF lowering 需汇编 | `vf/*`, `vector_api/*`, `__simd_vf__` |
| Vector 针对 Softmax/GELU 优化,UB↔ALU 间加 RegFile | p20-21 | softmax/online update 能被藏进 Cube 流水的硬件根因 | `ProcessVec1/2` |
| Cube L1 Buffer 与 Vector UB 直接 CV 通道,明确为 FA 带宽瓶颈而加 | p22 | C/V 数据通路不走 L2/GM 往返 | BMM1→UB, P→L1 |
| CV 传输随路精度/排布转换;L0C→UB 随路量化 | p17,p22 | Fixpipe 是核心路径,cast/quant 放 L0C 出口 | `FixpipeParamsC310` |
| NDDMA 最多 5 维重排,搬运同时转置/重排,细粒度读聚合为 128B | p22-23 | ND/NZ、varlen/tail、多维 DataCopy 的硬件亲和 | `DataCopy`, NDDMA lowering 待查 |
| BufferID 同步;STARS2.0 大量同步标志、ns 级 HSCB、Group/Die 亲和调度 | p23,p27-28 | 新同步/发射背景;当前仍用 CrossCore flag | `CrossCoreSetFlag/WaitFlag` |

## 1. Cube:Vector 峰值算力(p13-14)

| 产品 | Cube BF16/FP16 TFLOPS | Vector FP16/BF16 TFLOPS | 粗略比 |
|---|---|---|---|
| 950PR | 432/378 | 54/47 | 约 8:1 |
| 950DT | 486/432/378 | 60/54/47 | 约 8:1 |

注意:8:1 是峰值矩阵 FLOPS 比,不是 FA 阶段耗时比;Vector 的 exp/特殊函数有效吞吐白皮书未给(设计书 §2.5(5) / C8 待 profile)。

## 2. 结论分级回填规则

| 分类 | 含义 |
|---|---|
| 白皮书确认 | 硬件能力/容量可直接引用(1C:2V、UB/L1/L0 容量、SIMD/SIMT、双发、NDDMA、L2 128MB) |
| 源码确认 | 代码已读到实现(`KERNEL_TYPE_MIX_AIC_1_2`、P 写 L1、`GetC2Position`、`splitAxis`) |
| 待 profile/汇编 | 白皮书+C++ 都不足以定量或确认 lowering(VF 是否落 SIMT、双发槽位、FIFO 深度、单核有效带宽、Vector exp 吞吐、跨 Die L2 代价) |

回填位置见设计书 §2.2 / §2.4 / §2.5 / §7 / §8 / §9 / §14 Track C。