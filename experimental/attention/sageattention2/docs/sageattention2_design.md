# 需求背景（required）

## 需求来源

- 任务来源：昇腾社区开源算子开发大赛 / ops-transformer 社区算子开发任务。
- 交付目标仓库：https://gitcode.com/cann/ops-transformer ，合入路径为 `experimental/attention/sageattention2`。
- 对标参考：SageAttention2（ICML 2025，arXiv:2411.10958），GitHub 开源仓 `thu-ml/SageAttention` main 分支（`sageattention==2.2.0`），对标 `sageattn_qk_int8_pv_fp8_cuda`。
- 性能基线：ops-transformer 同仓 FIA（FusedInferAttentionScore），针对 Atlas 950PR 对标 `aclnnFusedInferAttentionScoreV5` 与 `torch_npu.npu_fused_infer_attention_score`。

## 背景介绍

### SageAttention2 算子实现优化

- Attention 计算是现代大语言模型与多模态扩散模型（如 CogVideoX、HunyuanVideo、Flux 等）计算与显存访问的核心瓶颈。
- SageAttention2 提出了 INT8 QK^T 与 FP8 PV 的混合低精度量化注意力方案，结合精细粒度量化、两级累加抑制截断误差、以及序列维均值扣减（smooth_k）消除离群值。
- 在昇腾 Atlas 950PR 架构上，基于原生 FP8 / MXFP8 计算能力与 AscendC + CATLASS 框架，实现高效、高精度的 SageAttention2 算子，显著提升推理吞吐。

### 对标实现现状分析

- 开源 GPU 实现特点：
  - 基于 Triton 与 CUDA 实现 INT8 QK^T 与 FP8 PV；
  - 量化粒度支持 per_thread（交织步长分组）与 per_warp（连续 32 token 分组）；
  - 支持 smooth_k 与 smooth_v 离群值平滑；
  - 累加模式支持 fp32、fp32+fp32、fp32+fp16。
- 昇腾 950PR 架构映射：
  - 硬件原生支持 INT8 与 FP8(e4m3) MMAD 矩阵乘指令；
  - 采用 1C2V（1 个 Cube + 2 个 Vector）异构协同架构；
  - 利用 L0C 临时累加器与 FixPipe 随路排布/精度转换硬件通道，消灭中间全量矩阵的 GM 往返。

# 需求分析（required）

## 需求描述

- 基于 AscendC + CATLASS 联合开发 SageAttention2 算子（仅前向 INT8 QK^T + FP8 PV 路径），提供统一 aclnn 接口与 PyTorch 适配层接口。
- 性能硬性门禁：在 Atlas 950PR、CANN 9.2.0-beta.1 环境下，同机同 shape 同 dtype 相对 FIA，PyTorch 与 aclnn 双路径 Speedup 均达到 1.6 倍以上（Speedup >= 1.6x）。
- 精度标准：满足《生态算子开源精度标准》，以 GPU 侧 `sageattn_qk_int8_pv_fp8_cuda` 为 L-A 主标杆（混合容差 matched_ratio >= 0.99），以 FP32 SDPA 为 L-B 辅助标杆。

## 需求拆解

1. PyTorch 层接口：`sageattn` 自动分发与 `sageattn_qk_int8_pv_fp8_asc` 显式入口，参数逐项对齐开源。
2. aclnn 两段式接口：`aclnnSageAttention2GetWorkspaceSize` 与 `aclnnSageAttention2`。
3. 量化与平滑方案：自研 Vector INT8 per_thread / per_warp 量化，smooth_k 序列维扣减均值，FP8(e4m3) V 量化（scale_max=2.25/448.0）。
4. 累加模式：支持 fp32、fp32+fp32、fp32+fp16（默认验收档），实现低精度 MMA 临时累加与定期归约。
5. 特性支撑：head_dim pad（48补到64、96补到128、大于128报错并自动裁切）、GQA（Hq 整除 Hk）、causal 掩码、`return_lse`（返回自然对数域对齐的 logsumexp）。

# 详细设计（required）

## 算子分析

### 数学公式

- 主计算流：
  - O = softmax(QK^T * s) * V
- 量化与平滑：
  - K' = K - k_mean
  - Q_hat = quant_int8(Q), K_hat = quant_int8(K'), V_hat = quant_fp8(V)
  - P = softmax(Q_hat * K_hat^T * scale_q * scale_k * s)
  - O = P * V_hat * scale_v
- LSE 语义与校正：
  - LSE = lse_raw / 1.44269504 + (Q * k_mean) * s（GQA 下 k_mean 经 repeat_interleave 广播）

### 支持数据类型

- 输入 Q/K/V：torch.float16、torch.bfloat16（三者 dtype 必须一致）
- 量化内部类型：Q/K 为 INT8，V 为 FP8(e4m3)
- 输出 O：与 Q 相同 dtype（FP16 或 BF16）
- 输出 LSE（当 return_lse=True 时）：Float32

### 支持形状

- HND 布局：
  - Q: [batch, num_qo_heads, qo_len, head_dim]
  - K/V: [batch, num_kv_heads, kv_len, head_dim]
- NHD 布局：
  - Q: [batch, qo_len, num_qo_heads, head_dim]
  - K/V: [batch, kv_len, num_kv_heads, head_dim]
- 维度约束：
  - head_dim 原始维度属于 (0, 128]，内部 pad 到 64 或 128；
  - num_qo_heads 能够被 num_kv_heads 整除；
  - causal 模式下要求 qo_len == kv_len。

## 算子实现

### 实现方案

#### host 侧设计

##### 分核策略

- 满核利用原则：面向 950PR 标称 32 AI Core（兼容 28 核减配档）进行满核负载均衡。
- 任务网格划分：将 (batch_size, num_kv_heads, q_blocks) 映射为一维任务流，按核均匀分配。
- GQA 聚合优化：将 Hq / Hkv 个 Q 头在 M 维聚合成批，保证单核 M 维分块达到 64 整数倍，打满 MMAD 算力。
- 双 Die 亲和调度（DieTask）：在多 Die 场景下优先保证单核任务在本 Die 内部完成，降低跨 Die 互联流量。

##### 数据分块和内存优化策略

- 片上存储预算（严格受 RES-01 资源台账校验）：
  - L1 Bank0 Q 区：常驻当前 Q tile 与 scale，上限 256KB；
  - L1 Bank1 KV 区：K（INT8）与 V（FP8）双缓冲 ping-pong 区，上限 256KB；
  - L0C 累加器：用于 QK^T（INT32）与 PV（FP32）临时乘积累加，上限 256KB；
  - V0 UB 区：输入暂存双缓冲、smooth 扣减与量化中间区、ND2NZ 重排区，上限 256KB；
  - V1 UB 区：O 主累加器（FP32，64KB 常驻）、FixPipe 接收区与 softmax exp2 结果区（64KB）、两级累加归约区（128KB），上限 256KB。
- 512B 访存对齐法则：
  - innermost 维度与分块 stride 均按 512B 整倍数对齐，触发 MTE2/MTE3 full-burst 突发读写；
  - scale 数组存储按 512B line 边界对齐，消除跨 cache line 拆包。

##### tilingkey 规划策略

- 32 位整型 tiling_key 编码结构：
  - bits 0-3: gran（2=per_warp, 3=per_thread）
  - bits 4-7: accum（0=fp32, 1=fp32+fp32, 2=fp32+fp16）
  - bits 8-11: layout（0=NHD, 1=HND）
  - bits 12-15: causal（0=非因果, 1=因果）
  - bits 16-19: pad（0=d_pad 64 档, 1=d_pad 128 档；编码见 op_host SetTilingKey 的 `d_pad == 128u ? 1u : 0u`）
  - bits 20-23: input_dtype（1=fp16, 2=bf16）
- M5 规划：依据 tiling_key 做编译期模板特化（ASCENDC_TPL_ARGS_DECL）零分支直调；当前 kernel 按 GET_TILING_DATA_WITH_STRUCT 的运行时字段消费 tiling_key 不参与核内分派。

#### kernel 侧设计

- 1C2V 协同流水：
  - Cube 引擎：负责 INT8 MMAD（QK^T）与 FP8 MMAD（PV），由 CATLASS 矩阵乘模板驱动；
  - CATLASS 接入边界（2026-10-03 收敛）：分形常量唯一定义源为官方 cann/catlass 仓
    `<catlass/catlass.hpp>`（sage_tile.hpp include + static_assert 绑定，构建接线见
    cmake/SageAttnDeps.cmake）；LoadData2D/Fixpipe 数据通路因本算子 L1 为 ND 行布局
    + Data/Scale 相邻契约（与官方 TileCopyTla 的 zN 前置不匹配），保持 AscendC 自研，
    为任务书"按 Cube/CATLASS 微调分块"授权范围；
  - V0 向量核：负责 MTE2 搬入、smooth_k 均值扣减、INT8/FP8 量化、scale 提取与 ND2NZ 排布转换；
  - V1 向量核：负责 L0C FixPipe 接收、exp2 域 online softmax、行 max 与 skip-scale 动态重缩放、两级累加归约与 LSE 累计；
- 延迟一拍软件流水：
  - Cube 计算与 Vector 计算深度交叠，消灭 Vector 成为关键路径的等待气泡；
- 同步与流水守卫：
  - 采用现役 11 flag 有界缓冲协议（`sync_guard.h`）实现核间/核内定向通知；
  - 引入类型状态守卫（`pipeline_guard.h`，RES-05）确保 DMA 搬运与计算状态机严格保序。

## 支持硬件

- Atlas 950PR（SoC Version: Ascend950PR，NpuArch=3510），兼容 28 核与 32 核配置。

## 算子约束限制

- 仅支持前向推理计算，不支持反向传播。
- 仅支持 INT8 QK^T + FP8 PV 路径（不支持 FP16 PV 与纯 INT4）。
- 仅支持连续末维（`stride(-1) == 1`）。
- head_dim 原始维度属于 (0, 128]，超过 128 明确报错。
- 不支持任意 `attn_mask` 传入（非空即明确报错，因果模式走 `is_causal=True` 优化掩码）。

# 可维可测分析（required）

## 精度标准/性能标准

### 精度标准

- 遵循《生态算子开源精度标准》experimental_standard 规范；
- L-A 主标杆（GPU `sageattn_qk_int8_pv_fp8_cuda` 同配置）：
  - FP16 混合容差：rtol=2^-9, atol=2^-9, matched_ratio >= 0.99；
  - BF16 混合容差：rtol=2^-6, atol=2^-6, matched_ratio >= 0.99；
- L-B 辅标杆（FP32 SDPA）：相对误差比例 max <= 2.0, mean <= 1.2, RMSE <= 1.2；
- 25 桶全配置覆盖与 `tests/test_accumulator_probe.py` 4 项整数精确域探针保障。

### 性能标准

- 相对 ops-transformer 同仓 FIA 基线（`aclnnFusedInferAttentionScoreV5` 与 `npu_fused_infer_attention_score`）；
- P-01 至 P-07 全部 7 个基准 shape，在 aclnn 接口与 PyTorch 接口两条对比路径上，单 case Speedup 与全量几何平均 Speedup 均达到 1.6 倍以上（Speedup >= 1.6x）。

## 兼容性分析

- 新增独立算子模块，合入路径为 `experimental/attention/sageattention2/`；
- 接口与 SageAttention1/3 保持命名空间隔离，不修改既有 FIA 算子任何实现，保持 100% 独立与生态解耦。
