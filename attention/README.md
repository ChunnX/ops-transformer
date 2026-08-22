## 1 attention layer 算子分类原则

attention layer 算子按是否使用 Softmax 分为两大类：Softmax Attention（基于 Softmax 的注意力）和 Linear Attention（线性注意力）。

| 大类 | 子类 | 核心特征 | 典型网络 | 对应算子 |
| :--- | :--- | :--- | :--- | :--- |
| Softmax Attention | GQA | 标准 MHA/GQA 稠密注意力，对全部 KV 执行 Softmax(QK^T/√d)V，无稀疏筛选；覆盖训练正向/反向与推理全量/增量场景 | Llama、Qwen、GPT | flash_attention_score、flash_attention_score_grad、flash_attn、fused_infer_attention_score、quant_flash_attn |
| Softmax Attention | MLA | Multi-head Latent Attention，K/V 共享同一份低维 latent 表示，通过下采样+RmsNorm+上采样+RoPE 前处理生成 query/query_rope 并更新 KV cache | DeepSeek-V2/V3.1 | mla_prolog、mla_prolog_v2、mla_prolog_v3 |
| Softmax Attention | 稀疏 | 稀疏 Flash Attention / Sparse Flash MLA，先通过 Lightning Indexer 选取 Top-K 关键 KV 子集，再对选中 KV 执行注意力；含 KV 压缩（Compressor）、双路稀疏等变体 | DeepSeek-V3.2/V4（SMLA/QLI） | lightning_indexer、lightning_indexer_grad、quant_lightning_indexer、sparse_flash_attention、sparse_flash_attention_grad、kv_quant_sparse_flash_attention、compressor、compressor_grad、quant_compressor、lightning_indexer_v2、lightning_indexer_v2_metadata、quant_lightning_indexer_v2、quant_lightning_indexer_v2_metadata、sparse_flash_mla、sparse_flash_mla_grad、sparse_flash_mla_metadata、sparse_flash_mla_grad_metadata、sparse_flash_mla_softmax_l1_norm、sparse_flash_mla_softmax_l1_norm_metadata、quant_sparse_flash_mla、quant_sparse_flash_mla_metadata、mixed_quant_sparse_flash_mla、mixed_quant_sparse_flash_mla_metadata |
| Linear Attention | GDN | 基于 Gated DeltaNet 论文的 Gated Delta Rule，通过门控状态递归更新；分 Recurrent（Decode 逐步递归）和 Chunk（Prefill 切块并行）两种计算模式 | Gated DeltaNet、GLA | recurrent_gated_delta_rule、chunk_gated_delta_rule |
| Linear Attention | KDA | Kernel Delta Attention，基于分块核方法将 delta rule 的递归状态更新 kernel 化实现并行，适用于 Prefill 阶段 | DeltaNet、FLA | chunk_kda_fwd |

---

## 2 对外接口

下表列出 attention layer 算子的全部算子，算子名与算子目录名保持一致。

| 算子名 | 算子概述 | sample |
| :--- | :--- | :--- |
| flash_attention_score | 训练场景 FlashAttention 正向计算，执行 Dropout(Softmax(Mask(scale·Q·K^T+pse)))·V，支持 pse/mask/dropout/RoPE | [test_aclnn_flash_attention_score.cpp](./flash_attention_score/examples/test_aclnn_flash_attention_score.cpp) |
| flash_attention_score_grad | FlashAttentionScore 的反向梯度计算，求 Q/K/V 的梯度 | [test_aclnn_flash_attention_score_grad_v4_fp16.cpp](./flash_attention_score_grad/examples/test_aclnn_flash_attention_score_grad_v4_fp16.cpp) |
| flash_attn + flash_attn_metadata | 统一训练/推理 FlashAttention 接口，支持 BSND/BNSD/TND 布局、分页注意力、变长、滑动窗口；metadata 为其预计算 tiling 切分方案 | [test_flash_attn.py](./flash_attn/tests/pytests/test_flash_attn.py) |
| fused_infer_attention_score | 融合推理注意力，统一支持全量(Prompt)与增量(Incre)场景，支持量化/反量化/分页/共享前缀/RoPE | [test_aclnn_fused_infer_attention_score_v5.cpp](./fused_infer_attention_score/examples/arch35/test_aclnn_fused_infer_attention_score_v5.cpp) |
| incre_flash_attention | 自回归推理增量注意力，query 的 S 轴固定为 1，KV 叠加历史 state；950 已不支持，推荐使用 flash_attn | [test_aclnn_incre_flash_attention.cpp](./incre_flash_attention/examples/test_aclnn_incre_flash_attention.cpp) |
| prompt_flash_attention | 全量推理 Prefill 阶段 FlashAttention，支持 sparse 优化、INT8 量化；950 已不支持，推荐使用 flash_attn | [test_aclnn_prompt_flash_attention.cpp](./prompt_flash_attention/examples/test_aclnn_prompt_flash_attention.cpp) |
| quant_flash_attn + quant_flash_attn_metadata | MXFP8 量化版 FlashAttention，Q/K/V 为 FP8_E4M3，softmax 在 FP32 计算，输出 BF16；metadata 为其预计算 tiling 切分方案；prompt_flash_attention / incre_flash_attention 量化场景推荐使用 quant_flash_attn | [test_quant_flash_attn_func_rdv.py](./quant_flash_attn/tests/pytest/qfa_mxfp8_test/test_quant_flash_attn_func_rdv.py) |
| lightning_indexer | 基于 Q@K 相关性 + ReLU + 权重 W 选取每个 token 的 Top-K 稀疏位置索引，作为 SparseFlashAttention 前处理 | [test_aclnn_lightning_indexer.cpp](./lightning_indexer/examples/test_aclnn_lightning_indexer.cpp) |
| lightning_indexer_grad | lightning_indexer 的反向算子，计算 query/key/weights 的梯度 | [test_aclnn_lightning_indexer_grad.cpp](./lightning_indexer_grad/examples/test_aclnn_lightning_indexer_grad.cpp) |
| quant_lightning_indexer | 量化版 Lightning Indexer，query/key 为 INT8/FP8/HIF8 量化输入，存8算8 | [test_aclnn_quant_lightning_indexer.cpp](./quant_lightning_indexer/examples/test_aclnn_quant_lightning_indexer.cpp) |
| sparse_flash_attention | 面向大序列推理的稀疏注意力，通过 sparse_indices 选取关键 KV 执行 Softmax(QK^T/√d)V | [test_aclnn_sparse_flash_attention_v2.cpp](./sparse_flash_attention/examples/test_aclnn_sparse_flash_attention_v2.cpp) |
| sparse_flash_attention_grad | sparse_flash_attention 的反向算子，根据 sparseIndices 重排 KV 后计算 dQ/dK/dV | [test_aclnn_sparse_flash_attention_grad.cpp](./sparse_flash_attention_grad/examples/test_aclnn_sparse_flash_attention_grad.cpp) |
| kv_quant_sparse_flash_attention | 支持 Per-Token-Head-Tile-128 量化 KV 输入的稀疏注意力，仅 IR 图模式算子，无 aclnn C 接口 | [test_npu_kv_quant_sparse_flash_attention.py](./kv_quant_sparse_flash_attention/examples/test_npu_kv_quant_sparse_flash_attention.py) |
| mla_prolog | MLA 前处理：下采样 + RmsNorm + 上采样 + RoPE，生成 query/query_rope 并更新 kv_cache/kr_cache | [test_aclnn_mla_prolog_nq_bsh.cpp](./mla_prolog/examples/test_aclnn_mla_prolog_nq_bsh.cpp) |
| mla_prolog_v2 | MLA 前处理 v2，新增全量化场景支持（token_x/weight 均可量化），输出 dequant_scale_q_nope | [test_aclnn_mla_prolog_v2_fq.cpp](./mla_prolog_v2/examples/test_aclnn_mla_prolog_v2_fq.cpp) |
| mla_prolog_v3 | MLA 前处理 v3，新增尺度矫正因子(αq/αkv)、query_norm 输出、per-token/per-group 量化模式 | [test_aclnn_mla_prolog_v3.cpp](./mla_prolog_v3/examples/test_aclnn_mla_prolog_v3.cpp) |
| compressor | KV 压缩前处理，将每 4 或 128 个 token 的 KV cache 压缩成 1 个，输出 cmp_kv | [test_aclnn_compressor.cpp](./compressor/examples/test_aclnn_compressor.cpp) |
| compressor_grad | compressor 的反向算子，计算 X、W^KV/W^Gate 与 Ape 的梯度 | [test_aclnn_compressor_grad.cpp](./compressor_grad/examples/test_aclnn_compressor_grad.cpp) |
| quant_compressor | 量化版 Compressor，输入 x/W^KV/W^Gate 为 HIFLOAT8，硬件原生 Matmul 后反量化 | [test_aclnn_quant_compressor.cpp](./quant_compressor/examples/arch35/test_aclnn_quant_compressor.cpp) |
| lightning_indexer_v2 + lightning_indexer_v2_metadata | V2 版稀疏索引选取，支持 maskMode/cmpRatio，基于 Q@K 相关性 + Top-K 选出稀疏位置；metadata 为其生成负载均衡方案 | [test_aclnn_lightning_indexer_v2.cpp](./lightning_indexer_v2/examples/test_aclnn_lightning_indexer_v2.cpp) |
| quant_lightning_indexer_v2 + quant_lightning_indexer_v2_metadata | 量化版 V2 稀疏索引，query/key 为 INT8/FP8/HIF8 量化输入；metadata 为其生成负载均衡方案 | [test_aclnn_quant_lightning_indexer_v2.cpp](./quant_lightning_indexer_v2/examples/test_aclnn_quant_lightning_indexer_v2.cpp) |
| sparse_flash_mla + sparse_flash_mla_metadata | MLA 稀疏注意力主算，支持双路（ori 原始 KV + cmp 压缩 KV）稀疏注意力，支持 SWA/CSA/HCA 掩码；metadata 为其生成负载均衡方案 | [test_aclnn_sparse_flash_mla.cpp](./sparse_flash_mla/examples/test_aclnn_sparse_flash_mla.cpp) |
| sparse_flash_mla_grad + sparse_flash_mla_grad_metadata | sparse_flash_mla 训练场景反向算子；metadata 为其生成含 Batch/Head/Q/K 分块索引的任务列表 | [test_aclnn_sparse_flash_mla_grad.cpp](./sparse_flash_mla_grad/examples/test_aclnn_sparse_flash_mla_grad.cpp) |
| sparse_flash_mla_softmax_l1_norm + sparse_flash_mla_softmax_l1_norm_metadata | 计算 sparse_flash_mla 注意力的 Softmax L1Norm 结果，为 KL Loss Grad 反向配套；metadata 为其生成前置分核方案 | [test_aclnn_sparse_flash_mla_softmax_l1_norm.cpp](./sparse_flash_mla_softmax_l1_norm/examples/test_aclnn_sparse_flash_mla_softmax_l1_norm.cpp) |
| quant_sparse_flash_mla + quant_sparse_flash_mla_metadata | 全量化版 MLA 稀疏注意力，Q/KV 为 HIFLOAT8 per-tensor 量化；metadata 为其生成负载均衡方案 | [test_aclnn_quant_sparse_flash_mla.cpp](./quant_sparse_flash_mla/examples/test_aclnn_quant_sparse_flash_mla.cpp) |
| mixed_quant_sparse_flash_mla + mixed_quant_sparse_flash_mla_metadata | 混合量化版 MLA 稀疏注意力，支持 per-token-group 量化 KV；metadata 为其生成负载均衡方案 | [test_aclnn_mixed_quant_sparse_flash_mla.cpp](./mixed_quant_sparse_flash_mla/examples/test_aclnn_mixed_quant_sparse_flash_mla.cpp) |
| recurrent_gated_delta_rule | 逐时间步递归更新隐状态 S_t 并输出 o_t，适用于 Decode 阶段；支持变步长、状态索引 | [test_aclnn_recurrent_gated_delta_rule.cpp](./recurrent_gated_delta_rule/examples/test_aclnn_recurrent_gated_delta_rule.cpp) |
| chunk_gated_delta_rule | 序列切块并行计算 Gated Delta Rule，长上下文效率高于 Recurrent 版，适用于 Prefill 阶段；输出每步 o_t 及最终状态 S_L | [test_aclnn_chunk_gated_delta_rule.cpp](./chunk_gated_delta_rule/examples/test_aclnn_chunk_gated_delta_rule.cpp) |
| chunk_kda_fwd | KDA 分块前向计算，通过分块核方法将 delta rule 递归状态更新 kernel 化实现并行，适用于 Prefill 阶段 | [test_chunk_kda_fwd.py](./chunk_kda_fwd/tests/pytest/test_chunk_kda_fwd.py) |

---

## 3 Metadata 算子机制

### 3.1 概述

Metadata 算子是与主算子配套的 AICPU 前置算子（命名以 `_metadata` 结尾），运行在 device 侧的 AI CPU 上，负责根据输入的序列元信息（`cuSeqlens`、`seqused`、batch_size、max_seqlen 等）计算负载均衡的分核切分方案，输出固定 shape 的 metadata tensor 供主算子消费。

典型调用流程为两段式：

```python
import torch
import torch_npu

# 第一步：调用 metadata 算子生成分核方案（AICPU 执行）
metadata = torch_npu.npu_sparse_flash_mla_metadata(
    cu_seqlens_q, cu_seqlens_kv,
    seqused_q, seqused_kv,
    batch_size, max_seqlen_q, max_seqlen_kv,
    num_heads_q, num_heads_kv, head_dim,
    ...
)

# 第二步：调用主算子，将 metadata 传入（AICore 执行）
attn_out = torch_npu.npu_sparse_flash_mla(
    query, ori_kv, cmp_kv,
    sparse_indices, block_table,
    cu_seqlens_q, seqused_kv,
    metadata=metadata,
    ...
)
```

### 3.2 解决的核心问题：Tiling 下沉

#### 3.2.1 什么是 Tiling 下沉

传统模式下，主算子的分核切分在 host 端 `GetWorkspaceSize`（tiling）阶段完成。Tiling 下沉是一种图执行模式，将原本在 host 端 tiling 阶段做的分核计算，下沉到 device 侧（AICPU）执行。Metadata 算子就是 tiling 下沉的具体实现手段。

#### 3.2.2 为什么需要 Tiling 下沉

| 问题 | 说明 |
| :--- | :--- |
| **Host 端无法读取 device tensor 的值** | 变长序列（TND）场景下，`cuSeqlens`、`seqused` 等 tensor 在 device 侧，host 端 tiling 阶段拿不到实际数值，无法做精准负载均衡 |
| **减少 host 端开销** | 复杂的分核逻辑在 host 端执行会阻塞 launch 流水，下沉到 AICPU 后主算子 tiling 阶段可直接跳过切分计算，减少对 host array 的访问 |
| **支持 aclgraph 图模式** | aclgraph 要求静态编译时确定执行图，而 tiling 依赖运行时数据。通过 AICPU 算子提前算好 metadata，使主算子能在图模式下执行 |

#### 3.2.3 整体下沉

"整体下沉"指将 tiling 计算从 host 端整体下沉到 AICPU 算子中，主算子 tiling 阶段不再自行计算切分方案，而是直接消费 AICPU 算子的输出结果。主算子文档将 metadata 描述为"tiling 下沉的 AICPU 算子输出结果"。

### 3.3 Metadata 算子与主算子的值依赖及分工

Metadata 算子的输出 tensor 直接作为主算子的输入 tensor（`metadataOptional` 参数），是 device 侧的运行时数据依赖，而非 host 侧属性依赖。主算子 tiling 阶段读取 metadata tensor 填充分核结构，kernel 阶段按分核结果执行计算。两者分工如下：

| 维度 | Metadata 算子（AICPU） | 主算子（AICore） |
| :--- | :--- | :--- |
| 执行位置 | Device 侧 AI CPU | Device 侧 AI Core / Vector |
| 功能 | 负载均衡分核、任务划分 | Attention 数值计算 |
| 数据访问 | 可读 device tensor 实际值（cuSeqlens、seqused 等） | 按 metadata 指示执行 |
| 执行时机 | 主算子之前（前置算子） | 消费 metadata 后执行 |
| workspace | 通常为 0 | 需要较大 workspace |
| 流水关系 | 两者可流水并行，metadata 可复用 | 消费前置结果 |

### 3.4 当前 Metadata 算子清单

| Metadata 算子 | 配套主算子 |
| :--- | :--- |
| flash_attn_metadata | flash_attn |
| quant_flash_attn_metadata | quant_flash_attn |
| sparse_flash_mla_metadata | sparse_flash_mla |
| sparse_flash_mla_grad_metadata | sparse_flash_mla_grad |
| sparse_flash_mla_softmax_l1_norm_metadata | sparse_flash_mla_softmax_l1_norm |
| quant_sparse_flash_mla_metadata | quant_sparse_flash_mla |
| mixed_quant_sparse_flash_mla_metadata | mixed_quant_sparse_flash_mla |
| lightning_indexer_v2_metadata | lightning_indexer_v2 |
| quant_lightning_indexer_v2_metadata | quant_lightning_indexer_v2 |

---

## 4 公共概念

### 4.1 Layout（数据布局）

attention layer 算子中，输入输出张量的排布格式通过 `inputLayout` 参数指定。维度字母含义如下：

| 字母 | 含义 |
| :--- | :--- |
| B | Batch Size，输入样本批量大小 |
| S | Sequence Length，单条序列长度 |
| T | Total Tokens，所有 Batch 序列长度的累加和（T = ΣS_i） |
| N | Head Num，多头数 |
| D | Head Dim，单个 head 的隐藏维度 |
| H | Head Size，隐藏层大小（H = N × D） |

各布局格式定义如下：

| 格式名 | shape | 特点 | 适用场景 |
| :--- | :--- | :--- | :--- |
| BSH | `[B, S, H]`，H = N × D | N 和 D 合并为一个维度 H，3 维格式，最简单 | FlashAttentionScore 训练；PromptFlashAttention / IncreFlashAttention / FusedInferAttentionScore 推理（推荐默认布局） |
| BSND | `[B, S, N, D]` | N 和 D 拆分为独立维度，4 维格式 | 训练和推理全场景通用 |
| BNSD | `[B, N, S, D]` | 同 BSND，但 N 和 S 轴互换 | 训练和推理全场景通用 |
| TND | `[T, N, D]` | 将多个 batch 的序列拼接为一维张量，消除 padding 浪费；需配合 `cuSeqlens` 标记 batch 边界 | 变长序列（VarLen）场景，训练和推理均支持 |
| NTD | `[N, T, D]` | 同 TND，但 N 和 T 轴互换 | 量化注意力全量化场景；常以 `NTD_TND`（输入 NTD、输出 TND）组合使用 |
| BnBsH | `[BlockNum, BlockSize, H]` | PA 场景 KV Cache 布局，H = N×D，3 维最简格式；H 超过 65535 时不可用 | PageAttention KV Cache 存储 |
| PA_BSND / BnNBsD / PA_BBND | `[BlockNum, BlockSize, N, D]` 或 `[BlockNum, N, BlockSize, D]` | PA 场景 KV Cache 布局，D 维连续存放，4 维格式；性能优于 BnBsH，推荐优先选择 | PageAttention KV Cache 存储 |
| PA_NZ / NZ | `[BlockNum, N, D/D0, BlockSize, D0]` | PA 场景 KV Cache 布局，D 维拆分为 D/D0 × D0 的 NZ 分形结构，5 维；最内层 D0=32（FP8/INT8）或 16（FP16/BF16/keyRope）；量化场景专用，INT8 时**仅支持** NZ | PageAttention 量化场景 KV Cache 存储 |

> **补充说明**：
> - **BNSD_BSND**：特殊组合布局，输入为 BNSD 时输出格式为 BSND。
> - PA_BSND 与 PA_NZ 的核心区别：PA_BSND 的 D 维连续存放（4 维），PA_NZ 将 D 维拆分为 NZ 分形（5 维），量化场景下对 Cube 指令更友好。

### 4.2 PageAttention（分页注意力）

#### 4.2.1 原理与设计动机

在大模型自回归推理中，每个 token 生成时都需读取历史 KV Cache。传统 KV Cache 要求每个 batch 在 S 维（序列长度）上预留一段连续物理内存，并按 `maxSeqLen` 预先 padding，导致显存碎片化和浪费。

PagedAttention 借鉴操作系统虚拟内存分页机制，将 KV Cache 切分为固定大小的 block，存储在一片连续物理内存中，通过 `blockTable`（页表）建立"逻辑序列顺序 → 物理块地址"的映射。其核心价值：

- **消除显存碎片**：按需分配 block，无需预分配整段连续内存
- **提升显存利用率**：不同 batch 共享同一片物理 block 池，支持更大 batch 或更长序列
- **逻辑连续、物理离散**：逻辑上连续的 KV 序列可散布在不连续的物理 block 中

#### 4.2.2 工作机制

PagedAttention 的开启必要条件是 `blockTable` 存在且有效，同时 key/value 按 `blockTable` 中的索引在连续内存中排布。开启后 key/value 的 `inputLayout` 参数失效，KV Cache 的排布由其自身 shape 维度决定。

**blockTable 查表过程**（以 `B=2`、`blockSize=128`、`actualSeqLengthsKv=[256, 512]` 为例）：

`maxBlockNumPerSeq = ceil(512/128) = 4`，blockTable shape = `[2, 4]`：

```
blockTable = [
  [ 3,  7,  1,  0],   # batch 0: 逻辑块0→物理block 3, 逻辑块1→物理block 7 (实际只用2个block)
  [ 5,  2,  9,  4],   # batch 1: 逻辑块0→物理block 5, 逻辑块1→物理block 2, 逻辑块2→物理block 9, 逻辑块3→物理block 4
]
```

访问 batch 0 的 token 0~255 时：
- token 0~127 → 逻辑块 0 → 查表得 block_id=3 → 读物理 `kvCache[3]`
- token 128~255 → 逻辑块 1 → 查表得 block_id=7 → 读物理 `kvCache[7]`

block id 可乱序分配、可跨 batch 共享物理池，用户需自行保证 block id 合法性（host 侧不校验）。

#### 4.2.3 blockTable

| 项目 | 内容 |
| :--- | :--- |
| 数据类型 | INT32 |
| shape | `[B, maxBlockNumPerSeq]` |
| 第一维 | 等于 Batch 数 B |
| 第二维 | 不小于最大 KV 序列长度对应的 block 数，即 `ceil(max_KV_S / blockSize)` |
| 元素含义 | block id，指向 KV Cache 物理池中第几个 block |
| 约束 | block id 合法性由用户自行保证，host 侧不校验 |

#### 4.2.4 blockSize

| 项目 | 内容 |
| :--- | :--- |
| 含义 | 每个 block 中的最大 token 数，用户自定义参数 |
| 通用对齐 | FP16/BF16 需 16 对齐；INT8/HIFLOAT8/FP8_E4M3 需 32 对齐；INT4/FLOAT4_E2M1 需 64 对齐 |
| 最大值 | 1024（部分早期版本为 512） |
| 推荐值 | 128（兼容性最好，多数全量化场景的强制值） |
| 性能影响 | 调大 blockSize 有一定性能收益（减少查表次数、提升单次搬运数据量），但会增加尾部 block 显存浪费 |

#### 4.2.5 PA 模式与普通模式对比

| 维度 | 普通模式 | PA 模式 |
| :--- | :--- | :--- |
| KV 物理排布 | S 轴连续 | 按 block 散布，靠 blockTable 映射 |
| inputLayout（KV） | 生效 | 失效 |
| blockTable | 不传 | 必须传 |
| actualSeqLengthsKv | 可选 | 必须传 |
| tensorlist | 支持 | 不支持 |
| 左 padding | 支持 | 不支持 |
| 公共前缀 | 支持 | 不支持 |
| D 不等长 | 支持 | 不支持 |
| KV Cache 布局 | BSH/BSND/BNSD/TND 等 | BnBsH / BnNBsD / NZ |

#### 4.2.6 性能特点

- **吞吐量提升 vs 单次延迟下降**：PA 通过提升显存利用率支持更大 batch，从而提高吞吐量；但 blockTable 间接寻址带来额外查表开销和非连续访问，单次推理延迟略有下降
- **KV Cache 排布影响性能**：BnNBsD 性能优于 BnBsH（N 轴前置便于按 head 拉取连续 D 数据），推荐优先选择 BnNBsD 格式
- **BnBsH 限制**：当 KV_N × D 超过 65535 时，受硬件指令约束报错，需改用 BnNBsD 格式

### 4.3 Mask Mode（掩码模式）

Mask Mode（`sparseMode` / `mask_mode`）控制注意力计算中的掩码策略。不同算子中参数名不同但语义一致。

#### 4.3.1 取值定义

| 取值 | 名称 | 含义 | attenMask 要求 | preTokens/nextTokens |
| :---: | :--- | :--- | :--- | :--- |
| 0 | defaultMask | 默认模式；不传 mask 则全计算；传 mask 则按 preTokens/nextTokens 做 causal 或 band | 可选 | 生效 |
| 1 | allMask | 传入完整 attenMask 矩阵 | 必传完整矩阵 | 不生效 |
| 2 | leftUpCausal | 左上顶点划分的下三角场景 | 压缩下三角矩阵（2048×2048） | 不生效 |
| 3 | rightDownCausal | 右下顶点划分的下三角场景 | 压缩下三角矩阵（2048×2048） | 不生效 |
| 4 | band | 计算 preTokens 和 nextTokens 之间的部分，起点为右下角 | 压缩下三角矩阵（2048×2048） | 生效 |
| 5 | prefix（非压缩） | rightDownCausal 基础上左侧加宽矩阵 | 矩形矩阵 | 不生效 |
| 6 | prefix（压缩） | 压缩下三角 + 矩形矩阵 | 矩阵（3072×2048） | 不生效 |
| 7 | varlen rightDownCausal | VarLen 外切场景，基于 mode=3 外切 | 压缩矩阵 | 不生效 |
| 8 | varlen leftUpCausal | VarLen 外切场景，基于 mode=2 外切 | 压缩矩阵 | 不生效 |
| 9 | treeMask | 推测解码（speculative decoding）树形注意力掩码 | 树形 mask | 不生效 |

#### 4.3.2 各算子支持范围

| 算子 | 支持的 sparseMode | 备注 |
| :--- | :--- | :--- |
| flash_attention_score | 0、1、2、3、4、5、6 | 默认 0；mode=0/4 时 preTokens/nextTokens 生效 |
| flash_attention_score VarLen | 0~8（不支持 5） | 传 rope 时不支持 6；mode=7 不支持 realShiftOptional |
| flash_attn | 0、2、3、4 | maskMode 参数 |
| fused_infer_attention_score | 0~4 | 默认 0；Decode MLA 仅支持 0/3/4；GQA head dim 不等时仅支持 0/2/3 |
| prompt_flash_attention | 0、3 | TND 场景仅支持 0（不传 mask）或 3（传 mask） |
| sparse_flash_attention | 0、3 | 0=全计算，3=rightDownCausal |
| sparse_flash_mla | ori_mask_mode: 0/3/4；cmp_mask_mode: 0/3 | 拆分为两个独立参数分别控制原始 KV 和压缩 KV 的掩码 |
| lightning_indexer_v2 | 0、3 | 0=No mask，3=rightDownCausal |

#### 4.3.3 SparseFlashMla 的双 mask_mode

SparseFlashMla 将掩码拆分为两个独立参数：

| 参数 | 支持值 | 含义 |
| :--- | :--- | :--- |
| `ori_mask_mode` | 0、3、4 | 控制原始 KV 路径：0=No Mask，3=RightDownCausal，4=Band |
| `cmp_mask_mode` | 0、3 | 控制压缩 KV 路径：0=No Mask，3=RightDownCausal |

交叉规则：
- SWA 场景（`ori_mask_mode=4`）：要求 `cmp_mask_mode=0`，需传入非负 `ori_win_left/ori_win_right`
- HCA/CSA 场景：要求 `cmp_mask_mode=3`
- 非 Sliding Window 场景（`ori_mask_mode≠4`）：要求 `ori_win_left=ori_win_right=-1`

### 4.4 cuSeqlens & seqused

这两个参数用于描述变长序列中每个 batch 的实际有效长度。不同 layout 场景下使用方式不同：TND 布局将多个 batch 的序列拼接为一维张量，需通过 `cuSeqlens` 标记 batch 边界；非 TND 布局（BSND/BNSD/BSH 等）每个 batch 有独立的 S 维度，需通过 `seqused` 标记实际有效长度。

#### 4.4.1 cuSeqlens（累积序列长度）

| 项目 | 内容 |
| :--- | :--- |
| 含义 | 前缀和（cumulative sum），标记各 batch 的边界；每个元素为当前 batch 及前序 batch 有效 token 数的累加值 |
| shape | `(B+1,)` |
| 数据类型 | INT32 |
| 数值约束 | 首元素固定为 0；单调非递减；末元素等于 T（总 token 数） |
| 适用场景 | **TND 布局必传**（数据按 batch 紧凑排列，需 cuSeqlens 标记边界）；非 TND 布局禁止传入 |

#### 4.4.2 seqused（每 batch 有效长度）

| 项目 | 内容 |
| :--- | :--- |
| 含义 | 每个 batch 实际参与计算的 token 数 |
| shape | `(B,)` |
| 数据类型 | INT32 |
| 适用场景 | TND 可选（总长度由 cuSeqlens 表达，seqused 可显式覆盖）；非 TND 布局可选（用于标记每个 batch 的有效长度，不超过 shape 中的 S） |

#### 4.4.3 两者的关系

| 维度 | cuSeqlens | seqused |
| :--- | :--- | :--- |
| 语义 | 累积前缀和，标记 batch 边界 | 每个 batch 的有效长度 |
| shape | `(B+1,)` | `(B,)` |
| 首元素 | 固定为 0 | 无要求 |
| 数值关系 | `cuSeqlens[i+1] - cuSeqlens[i] = seqused[i]` | 可由 cuSeqlens 差分得到 |
| TND 布局 | 必传 | 可选（覆盖有效长度） |
| 非 TND 布局 | 禁止传入 | 可选 |

### 4.5 稀疏索引参数（topk / sparse_indices / cmp_ratio）

稀疏注意力算子通过 TopK 选取关键 KV 子集，仅对选中 KV 执行注意力计算。涉及三个核心参数：`topk` 控制选取数量，`sparse_indices` 记录选取的位置索引，`cmp_ratio` 控制压缩倍率。

#### 4.5.1 topk（TopK 选取数量）

`topk` 是 Lightning Indexer 系列算子的输入/属性参数，决定每个 query token 经相关性打分后保留多少个 KV 位置索引。

| 项目 | 内容 |
| :--- | :--- |
| 含义 | TopK 阶段需要保留的 block/索引数量 |
| 数据类型 | INT32 或 INT64（因算子而异） |
| 取值范围 | [1, 2048]，以及 3072/4096/5120/6144/7168/8192；950PR/950DT 支持 [1, 8192] |
| 默认值 | 2048 |
| 直接控制 | `sparse_indices` 输出的最后一维大小 |

> 不同算子命名不同：lightning_indexer / quant_lightning_indexer 使用 `sparse_count`（属性 INT32），lightning_indexer_v2 使用 `topk`（输入 INT64），quant_lightning_indexer_v2 使用 `topk`（属性 INT32）。语义一致。

#### 4.5.2 sparse_indices（稀疏位置索引）

`sparse_indices` 是 Lightning Indexer 的输出、Sparse Flash Attention / Sparse Flash MLA 的输入，记录每个 query token 应关注的 KV 位置索引。

| 项目 | 内容 |
| :--- | :--- |
| 数据类型 | INT32 |
| shape（BSND） | `[B, Q_S, KV_N, sparse_size]` |
| shape（TND） | `[Q_T, KV_N, sparse_size]` |
| `sparse_size` | 对应 lightning_indexer 的 `topk` / `sparse_count` |
| 无效值约定 | 无效部分填充 -1；消费侧约定有效值在前半部分、无效值在后半部分 |
| 生产侧 | lightning_indexer / lightning_indexer_v2 / quant_lightning_indexer / quant_lightning_indexer_v2 输出 |
| 消费侧 | sparse_flash_attention / kv_quant_sparse_flash_attention / sparse_flash_mla 输入 |

**生产-消费链路**：

```
lightning_indexer(query, key, weights, topk) → sparse_indices (输出)
                                                    ↓
sparse_flash_attention(query, key, value, sparse_indices) → attn_out (输入)
```

Lightning Indexer 计算 `ReLU(Q@K^T)` 得到相关性分数，乘以权重后沿 group 聚合，TopK 选取前 `topk` 个索引输出。Sparse Flash Attention 据此离散 gather KV 后执行 `Softmax(Q@K^T)@V`。

#### 4.5.3 Sparse Flash MLA 的双路稀疏索引

Sparse Flash MLA 采用双路 KV 设计（ori 原始 KV + cmp 压缩 KV），对应两组稀疏索引：

| 参数 | 作用对象 | shape 最后一维 | 当前状态 | 使用场景 |
| :--- | :--- | :--- | :--- | :--- |
| `ori_sparse_indices` | ori_kv（原始 KV） | 未明确 | 暂不支持，必须传 nullptr | 无（预留） |
| `cmp_sparse_indices` | cmp_kv（压缩 KV） | K2（TopK 稀疏选择数） | CSA 场景必传 | CSA |

三种注意力场景的输入要求：

| 场景 | ori_kv | cmp_kv | cmp_sparse_indices | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| SWA（滑动窗口） | 必传 | 不传 | 不传 | 仅滑动窗口注意力 |
| CSA（压缩稀疏） | 必传 | 必传 | **必传** | 滑动窗口 + TopK 稀疏压缩 KV |
| HCA（稠密压缩） | 必传 | 必传 | 不传 | 滑动窗口 + 稠密压缩 KV（不稀疏） |

#### 4.5.4 cmp_ratio（压缩倍率）

`cmp_ratio` 表示 cmp_kv 相对于压缩前 KV 长度的压缩倍率，用于在 mask 计算时将压缩后的 cmp_kv 长度还原为压缩前的逻辑 KV 长度。

| 项目 | 内容 |
| :--- | :--- |
| 含义 | cmp_kv 相对于压缩前 KV 长度的压缩倍率 |
| 数据类型 | INT |
| 取值范围 | [1, 128] |
| 场景约束 | SWA 保持默认值 1；CSA 支持传入 4；HCA 支持传入 128 |
| 配套参数 | `cmp_residual_kv`（shape `(B,)`），压缩 KV 余数，按 `cmp_len * cmp_ratio + residual` 恢复压缩前 KV 长度 |

> `topk_length` 仅在 sparse_flash_mla 中作为预留输入出现，当前版本不支持传入非空 Tensor，必须传 nullptr 或空 Tensor。

### 4.6 量化场景概念

attention layer 算子按 Q/K/V 是否量化及量化粒度，分为非量化、伪量化和全量化三类场景。

#### 4.6.1 量化类型划分

| 量化类型 | Q | K/V | 反量化参数 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| 非量化 | FP16/BF16 | FP16/BF16 | 无 | Q/K/V 均为高精度，直接参与计算 |
| 伪量化 | FP16/BF16 | INT8/FP8/HIF8 等 | `antiquantScale` / `antiquantOffset` | 仅 K/V 量化存储，算子内反量化后参与计算；Q 与输出仍为高精度 |
| 全量化 | INT8/FP8/HIF8 等 | INT8/FP8/HIF8 等 | `dequantScaleQuery` + `antiquantScale` | Q/K/V 均量化输入，算子内全部反量化后计算 |

#### 4.6.2 反量化参数体系

| 参数 | 作用 | 适用场景 |
| :--- | :--- | :--- |
| `antiquantScale` / `antiquantOffset` | 对 K/V 伪量化的反量化因子和偏移 | 伪量化（对称/非对称） |
| `keyAntiquantScale` / `valueAntiquantScale` | K/V 分离反量化因子（KV 可使用不同 scale） | 伪量化/全量化（KV 分离模式） |
| `dequantScaleQuery` | 对 Q 反量化的因子 | **全量化专用** |
| `deqScale1` / `quantScale1` | 对 QK 结果反量化 / 对 P 量化 | INT8 在线量化（Q 高精度场景） |
| `deqScale2` / `quantScale2` / `quantOffset2` | 对 PV 结果反量化 / 对输出量化及偏移 | INT8 在线量化（输出 INT8 场景） |

#### 4.6.3 GQA 全量化（FP8）

GQA 全量化场景下 Q/K/V 均为 FLOAT8_E4M3FN 格式，通过 FLOAT32 反量化因子在算子内恢复高精度计算。

| 项目 | 内容 |
| :--- | :--- |
| Q/K/V 数据类型 | FLOAT8_E4M3FN |
| 反量化因子 dtype | FLOAT32 |
| 量化粒度 | `queryQuantMode=3`（per-token 叠加 per-head），`keyAntiquantMode=3`（per-token 叠加 per-head），`valueAntiquantMode=2`（per-tensor 叠加 per-head） |
| HeadDim 约束 | 仅支持 128 |
| blockSize 约束 | 仅支持 128 |
| 典型算子 | fused_infer_attention_score |

#### 4.6.4 MXFP8 全量化

MXFP8（Microscaling FP8）是 FP8_E4M3FN 数据配合 FLOAT8_E8M0 缩放因子的量化格式，每 32 个元素共享一个 E8M0 scale。

| 项目 | 内容 |
| :--- | :--- |
| Q/K/V 数据类型 | FLOAT8_E4M3FN |
| 缩放因子 dtype | FLOAT8_E8M0 |
| 量化粒度 | `queryQuantMode/keyAntiquantMode=6`（per-token-group），`valueAntiquantMode=8`（per-channel-group） |
| scale shape | `(1, B, N, >=KV_S, D/32)`，每 32 个 D 元素共享一个 E8M0 scale |
| HeadDim 约束 | 仅支持 64 或 128 |
| blockSize 约束 | 支持 64/128/256/512/1024 |
| 典型算子 | quant_flash_attn、fused_infer_attention_score |

### 4.7 Batch 一致性

Batch 一致性（确定性计算）是指算子在不同 batch 大小、不同 batch 组合下执行时，计算结果保持一致的能力。默认情况下，部分算子（如 compressor / quant_compressor）的计算结果可能与输入 batch 在整体数据中的位置和 batch 划分方式有关，即同一组数据在不同 batch 划分下可能产生微小数值差异。

通过 `aclrtSetSysParamOpt()` 配置 `ACL_OPT_DETERMINISTIC=3` 可开启 batch 一致性模式，开启后算子内部采用确定性计算策略，保证计算结果与所在批次大小、位置无关。

| 项目 | 内容 |
| :--- | :--- |
| 含义 | 计算结果与 batch 大小、batch 在数据中的位置无关 |
| 开启方式 | `aclrtSetSysParamOpt(ACL_OPT_DETERMINISTIC, 3)` |
| 适用算子 | compressor、quant_compressor 等 |
| 默认状态 | 关闭（非确定性） |
| 代价 | 开启后可能带来一定性能下降 |

---

## 5 算子典型场景样例

下表列出第 2 章各算子对应的典型场景样例文件，点击链接可跳转至源文件。

### 5.1 GQA

| 算子名 | 典型样例 |
| :--- | :--- |
| flash_attention_score | [test_aclnn_flash_attention_score.cpp](./flash_attention_score/examples/test_aclnn_flash_attention_score.cpp) |
| flash_attention_score_grad | [test_aclnn_flash_attention_score_grad_v4_fp16.cpp](./flash_attention_score_grad/examples/test_aclnn_flash_attention_score_grad_v4_fp16.cpp) |
| flash_attn | [test_flash_attn.py](./flash_attn/tests/pytests/test_flash_attn.py) |
| flash_attn_metadata | [test_aclnn_flash_attn_metadata.cpp](./flash_attn_metadata/examples/test_aclnn_flash_attn_metadata.cpp) |
| fused_infer_attention_score | [test_aclnn_fused_infer_attention_score_v5.cpp](./fused_infer_attention_score/examples/arch35/test_aclnn_fused_infer_attention_score_v5.cpp) |
| incre_flash_attention | [test_aclnn_incre_flash_attention.cpp](./incre_flash_attention/examples/test_aclnn_incre_flash_attention.cpp) |
| prompt_flash_attention | [test_aclnn_prompt_flash_attention.cpp](./prompt_flash_attention/examples/test_aclnn_prompt_flash_attention.cpp) |
| quant_flash_attn | [test_quant_flash_attn_func_rdv.py](./quant_flash_attn/tests/pytest/qfa_mxfp8_test/test_quant_flash_attn_func_rdv.py) |
| quant_flash_attn_metadata | [test_aclnn_quant_flash_attn_metadata.cpp](./quant_flash_attn_metadata/examples/test_aclnn_quant_flash_attn_metadata.cpp) |

### 5.2 MLA

| 算子名 | 典型样例 |
| :--- | :--- |
| mla_prolog | [test_aclnn_mla_prolog_nq_bsh.cpp](./mla_prolog/examples/test_aclnn_mla_prolog_nq_bsh.cpp) |
| mla_prolog_v2 | [test_aclnn_mla_prolog_v2_fq.cpp](./mla_prolog_v2/examples/test_aclnn_mla_prolog_v2_fq.cpp) |
| mla_prolog_v3 | [test_aclnn_mla_prolog_v3.cpp](./mla_prolog_v3/examples/test_aclnn_mla_prolog_v3.cpp) |

### 5.3 稀疏

| 算子名 | 典型样例 |
| :--- | :--- |
| lightning_indexer | [test_aclnn_lightning_indexer.cpp](./lightning_indexer/examples/test_aclnn_lightning_indexer.cpp) |
| lightning_indexer_grad | [test_aclnn_lightning_indexer_grad.cpp](./lightning_indexer_grad/examples/test_aclnn_lightning_indexer_grad.cpp) |
| quant_lightning_indexer | [test_aclnn_quant_lightning_indexer.cpp](./quant_lightning_indexer/examples/test_aclnn_quant_lightning_indexer.cpp) |
| sparse_flash_attention | [test_aclnn_sparse_flash_attention_v2.cpp](./sparse_flash_attention/examples/test_aclnn_sparse_flash_attention_v2.cpp) |
| sparse_flash_attention_grad | [test_aclnn_sparse_flash_attention_grad.cpp](./sparse_flash_attention_grad/examples/test_aclnn_sparse_flash_attention_grad.cpp) |
| kv_quant_sparse_flash_attention | [test_npu_kv_quant_sparse_flash_attention.py](./kv_quant_sparse_flash_attention/examples/test_npu_kv_quant_sparse_flash_attention.py) |
| compressor | [test_aclnn_compressor.cpp](./compressor/examples/test_aclnn_compressor.cpp) |
| compressor_grad | [test_aclnn_compressor_grad.cpp](./compressor_grad/examples/test_aclnn_compressor_grad.cpp) |
| quant_compressor | [test_aclnn_quant_compressor.cpp](./quant_compressor/examples/arch35/test_aclnn_quant_compressor.cpp) |
| lightning_indexer_v2 | [test_aclnn_lightning_indexer_v2.cpp](./lightning_indexer_v2/examples/test_aclnn_lightning_indexer_v2.cpp) |
| lightning_indexer_v2_metadata | [test_aclnn_lightning_indexer_v2_metadata.cpp](./lightning_indexer_v2_metadata/examples/test_aclnn_lightning_indexer_v2_metadata.cpp) |
| quant_lightning_indexer_v2 | [test_aclnn_quant_lightning_indexer_v2.cpp](./quant_lightning_indexer_v2/examples/test_aclnn_quant_lightning_indexer_v2.cpp) |
| quant_lightning_indexer_v2_metadata | [test_aclnn_quant_lightning_indexer_v2_metadata.cpp](./quant_lightning_indexer_v2_metadata/examples/test_aclnn_quant_lightning_indexer_v2_metadata.cpp) |
| sparse_flash_mla | [test_aclnn_sparse_flash_mla.cpp](./sparse_flash_mla/examples/test_aclnn_sparse_flash_mla.cpp) |
| sparse_flash_mla_grad | [test_aclnn_sparse_flash_mla_grad.cpp](./sparse_flash_mla_grad/examples/test_aclnn_sparse_flash_mla_grad.cpp) |
| sparse_flash_mla_metadata | [test_aclnn_sparse_flash_mla_metadata.cpp](./sparse_flash_mla_metadata/examples/test_aclnn_sparse_flash_mla_metadata.cpp) |
| sparse_flash_mla_grad_metadata | [test_aclnn_sparse_flash_mla_grad_metadata.cpp](./sparse_flash_mla_grad_metadata/examples/test_aclnn_sparse_flash_mla_grad_metadata.cpp) |
| sparse_flash_mla_softmax_l1_norm | [test_aclnn_sparse_flash_mla_softmax_l1_norm.cpp](./sparse_flash_mla_softmax_l1_norm/examples/test_aclnn_sparse_flash_mla_softmax_l1_norm.cpp) |
| sparse_flash_mla_softmax_l1_norm_metadata | 暂无样例 |
| quant_sparse_flash_mla | [test_aclnn_quant_sparse_flash_mla.cpp](./quant_sparse_flash_mla/examples/test_aclnn_quant_sparse_flash_mla.cpp) |
| quant_sparse_flash_mla_metadata | [test_aclnn_quant_sparse_flash_mla_metadata.cpp](./quant_sparse_flash_mla_metadata/examples/test_aclnn_quant_sparse_flash_mla_metadata.cpp) |
| mixed_quant_sparse_flash_mla | [test_aclnn_mixed_quant_sparse_flash_mla.cpp](./mixed_quant_sparse_flash_mla/examples/test_aclnn_mixed_quant_sparse_flash_mla.cpp) |
| mixed_quant_sparse_flash_mla_metadata | [test_aclnn_mixed_quant_sparse_flash_mla_metadata.cpp](./mixed_quant_sparse_flash_mla_metadata/examples/test_aclnn_mixed_quant_sparse_flash_mla_metadata.cpp) |

### 5.4 GDN

| 算子名 | 典型样例 |
| :--- | :--- |
| recurrent_gated_delta_rule | [test_aclnn_recurrent_gated_delta_rule.cpp](./recurrent_gated_delta_rule/examples/test_aclnn_recurrent_gated_delta_rule.cpp) |
| chunk_gated_delta_rule | [test_aclnn_chunk_gated_delta_rule.cpp](./chunk_gated_delta_rule/examples/test_aclnn_chunk_gated_delta_rule.cpp) |

### 5.5 KDA

| 算子名 | 典型样例 |
| :--- | :--- |
| chunk_kda_fwd | [test_chunk_kda_fwd.py](./chunk_kda_fwd/tests/pytest/test_chunk_kda_fwd.py) |

