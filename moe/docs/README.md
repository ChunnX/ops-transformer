# MoE 算子库分析报告总览

本目录包含 `moe/` 下所有算子的全面分析报告，共 **31 份报告**，覆盖 **28 个算子目录**。

## 按功能分类

### 1. 门控/选路算子 (Gating & Routing)

| 算子 | 报告文件 | 功能概述 |
|------|----------|----------|
| moe_gating_top_k | [moe_gating_top_k_full_analysis.md](moe_gating_top_k_full_analysis.md) | 门控分组TopK，支持Sigmoid/Softmax+分组选择+多路归并，DeepSeekV3高性能路径 |
| moe_gating_top_k_softmax | [moe_gating_top_k_softmax_full_analysis.md](moe_gating_top_k_softmax_full_analysis.md) | Softmax+TopK融合，推理场景，支持finished(EOS)处理和row_idx源行映射 |
| moe_gating_top_k_softmax_v2 | [moe_gating_top_k_softmax_v2_full_analysis.md](moe_gating_top_k_softmax_v2_full_analysis.md) | V2增强版，新增renorm重归一化、可选softmax_result输出，公式化TilingKey编码 |
| moe_fused_topk | [moe_fused_topk_full_analysis.md](moe_fused_topk_full_analysis.md) | 融合分组TopK，组内TopN求和+广播掩码，mappingTable物理-逻辑专家映射 |
| moe_compute_expert_tokens | [moe_compute_expert_tokens_full_analysis.md](moe_compute_expert_tokens_full_analysis.md) | 二分查找已排序专家数组的每个专家最后出现位置，三种模板(SS/M/L) |

### 2. 路由初始化算子 (Init Routing)

| 算子 | 报告文件 | 功能概述 |
|------|----------|----------|
| moe_init_routing | [moe_init_routing_full_analysis.md](moe_init_routing_full_analysis.md) | V1基础版，排序+Gather |
| moe_init_routing_v2 | [moe_init_routing_v2_full_analysis.md](moe_init_routing_v2_full_analysis.md) | V2增强版，支持Drop/Pad、Expert Token Count |
| moe_init_routing_v3 | [moe_init_routing_v3_full_analysis.md](moe_init_routing_v3_full_analysis.md) | V3全面版，7种量化模式(FP8/HIF8)、部分专家路由、Scatter索引、性能打点 |
| moe_init_routing_quant | [moe_init_routing_quant_full_analysis.md](moe_init_routing_quant_full_analysis.md) | 量化V1版，静态量化 |
| moe_init_routing_quant_v2 | [moe_init_routing_quant_v2_full_analysis.md](moe_init_routing_quant_v2_full_analysis.md) | 量化V2版，动态量化、INT4 |
| moe_init_routing_v2_grad | [moe_init_routing_v2_grad_full_analysis.md](moe_init_routing_v2_grad_full_analysis.md) | V2反向传播，BinaryAdd二分累加树，Dropless/Activate/PositionPad0三模式 |

### 3. 路由完成算子 (Finalize Routing)

| 算子 | 报告文件 | 功能概述 |
|------|----------|----------|
| moe_finalize_routing | [moe_finalize_routing_full_analysis.md](moe_finalize_routing_full_analysis.md) | 7输入，19种TilingKey，14个kernel变体，BF16乒乓并行 |
| moe_finalize_routing_v2 | [moe_finalize_routing_v2_full_analysis.md](moe_finalize_routing_v2_full_analysis.md) | V2版，5个OPTIONAL输入，4种drop_pad_mode，混合精度，50种TilingKey |
| moe_finalize_routing_v2_grad | [moe_finalize_routing_v2_grad_full_analysis.md](moe_finalize_routing_v2_grad_full_analysis.md) | V2反向，两阶段并行，Scatter-Add无原子操作，FP32精度提升 |

### 4. Token 重排算子 (Token Permute)

| 算子 | 报告文件 | 功能概述 |
|------|----------|----------|
| moe_token_permute | [moe_token_permute_full_analysis.md](moe_token_permute_full_analysis.md) | 基础版，Bitonic Sort + Merge Sort排序后搬运token |
| moe_token_permute_grad | [moe_token_permute_grad_full_analysis.md](moe_token_permute_grad_full_analysis.md) | 基础反向，按sorted_indices逆向累加梯度 |
| moe_token_permute_with_ep | [moe_token_permute_with_ep_full_analysis.md](moe_token_permute_with_ep_full_analysis.md) | EP变体，新增probs输入输出，range[start,end)区间语义 |
| moe_token_permute_with_ep_grad | [moe_token_permute_with_ep_grad_full_analysis.md](moe_token_permute_with_ep_grad_full_analysis.md) | EP反向，两阶段计算(token_grad/probs_grad)，range约束偏移量 |
| moe_token_permute_with_routing_map | [moe_token_permute_with_routing_map_full_analysis.md](moe_token_permute_with_routing_map_full_analysis.md) | 路由矩阵变体，MaskedSelectV3提取有效位，支持drop_and_pad |
| moe_token_permute_with_routing_map_grad | [moe_token_permute_with_routing_map_grad_full_analysis.md](moe_token_permute_with_routing_map_grad_full_analysis.md) | 路由矩阵反向，专用DropPad逻辑按capacity/expertNum恢复梯度 |

### 5. Token 反重排算子 (Token Unpermute)

| 算子 | 报告文件 | 功能概述 |
|------|----------|----------|
| moe_token_unpermute | [moe_token_unpermute_full_analysis.md](moe_token_unpermute_full_analysis.md) | 基础版正向，9种数据类型组合，三层循环kernel(token分块->hidden分块->topK累加) |
| moe_token_unpermute_grad | [moe_token_unpermute_grad_full_analysis.md](moe_token_unpermute_grad_full_analysis.md) | 基础版反向，ping-pong double buffer + ReduceSum计算prob_grad |
| moe_token_unpermute_with_ep | [moe_token_unpermute_with_ep_full_analysis.md](moe_token_unpermute_with_ep_full_analysis.md) | EP版正向，range属性实现局部expert反排列 |
| moe_token_unpermute_with_ep_grad | [moe_token_unpermute_with_ep_grad_full_analysis.md](moe_token_unpermute_with_ep_grad_full_analysis.md) | EP版反向，复用基础版kernel，通过tiling参数(start/end)体现EP差异 |
| moe_token_unpermute_with_routing_map | [moe_token_unpermute_with_routing_map_full_analysis.md](moe_token_unpermute_with_routing_map_full_analysis.md) | RoutingMap版正向，两阶段(MaskedSelect+Unpermute)，drop_and_pad用独立Pad kernel |
| moe_token_unpermute_with_routing_map_grad | [moe_token_unpermute_with_routing_map_grad_full_analysis.md](moe_token_unpermute_with_routing_map_grad_full_analysis.md) | RoutingMap版反向，4种子kernel(prob_none/not_none x pad_true/false)+混精变体 |

### 6. 重路由算子 (Re-Routing)

| 算子 | 报告文件 | 功能概述 |
|------|----------|----------|
| moe_re_routing | [moe_re_routing_full_analysis.md](moe_re_routing_full_analysis.md) | rank-major到expert-major重排，双内核(membase/regbase)，三步tiling策略 |

### 7. 第三方算子 (3rd Party)

| 算子 | 报告文件 | 功能概述 |
|------|----------|----------|
| moe_gather_v2 | [3rd_moe_gather_v2_full_analysis.md](3rd_moe_gather_v2_full_analysis.md) | 轻量级L0 op，复用CANN内置GatherV2，无自定义tiling/kernel |
| moe_inplace_index_add | [3rd_moe_inplace_index_add_full_analysis.md](3rd_moe_inplace_index_add_full_analysis.md) | 最复杂第三方算子，5套Tiling策略，SIMD/SIMT双计算模式 |
| moe_inplace_index_add_with_sorted | [3rd_moe_inplace_index_add_with_sorted_full_analysis.md](3rd_moe_inplace_index_add_with_sorted_full_analysis.md) | 预排序版index_add，双kernel(Avg/Fix)，当前仅axis=0 |
| moe_masked_scatter | [3rd_moe_masked_scatter_full_analysis.md](3rd_moe_masked_scatter_full_analysis.md) | 掩码散射，前缀和优化，两阶段执行(Phase1前缀和+Phase2 SIMT Scatter) |

## 算子演进与变体关系

```
moe_init_routing (v1)
  ├── moe_init_routing_quant (量化v1)
  ├── moe_init_routing_v2 (增强v2)
  │   └── moe_init_routing_v2_grad (反向)
  ├── moe_init_routing_quant_v2 (量化v2)
  └── moe_init_routing_v3 (全面v3, 含量化/部分路由/Scatter)

moe_finalize_routing (v1)
  ├── moe_finalize_routing_v2 (增强v2)
  └── moe_finalize_routing_v2_grad (反向)

moe_token_permute (基础)
  ├── moe_token_permute_grad (反向)
  ├── moe_token_permute_with_ep (EP变体)
  │   └── moe_token_permute_with_ep_grad (反向)
  ├── moe_token_permute_with_routing_map (RoutingMap变体)
  │   └── moe_token_permute_with_routing_map_grad (反向)

moe_token_unpermute (基础)
  ├── moe_token_unpermute_grad (反向)
  ├── moe_token_unpermute_with_ep (EP变体)
  │   └── moe_token_unpermute_with_ep_grad (反向)
  ├── moe_token_unpermute_with_routing_map (RoutingMap变体)
  │   └── moe_token_unpermute_with_routing_map_grad (反向)

moe_gating_top_k (基础)
  ├── moe_gating_top_k_softmax (Softmax融合)
  │   └── moe_gating_top_k_softmax_v2 (V2增强)
  └── moe_fused_topk (融合分组TopK)
```

## 分析方法

每份报告均包含以下标准化分析内容：
- 算子概述（功能、支持平台）
- 算子定义（输入/输出/属性、数据类型映射）
- Tiling数据结构（ASCII层级图、字段说明）
- Tiling策略（TilingKey编码规则、切分策略）
- Shape推导规则
- Kernel执行流（数据流ASCII图、核心算法）
- Host-Kernel接口
- 完整数据流
- 依赖关系
- 与同族变体的核心差异（如适用）
