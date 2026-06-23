# AscendC 算子列表

## 判定规则

本文件只把“后续可以在本仓库内修改 AscendC 源码并优化性能”的目录计入主列表。静态判定规则如下：

- 目录位于主线算子域：`attention/`、`ffn/`、`gmm/`、`mc2/`、`moe/`、`posembedding/`。
- 目录下存在 `op_kernel/`。
- `op_kernel/` 中存在 `.cpp`、`.cc`、`.cxx`、`.h` 或 `.hpp` 源文件。
- 源码命中 AscendC 特征，例如 `__aicore__`、`AscendC::`、`using namespace AscendC`、`extern "C" __global__`、`KERNEL_TASK_TYPE_AICORE`。
- 默认排除 `tests/`、`examples/`、`common/`、`3rd/`、`experimental/`、模板目录和纯 AICPU 调度目录。

按上述规则，本轮静态扫描得到主线 AscendC 算子 117 个。`scripts/ci/ascend910b/ops_transformer_operator_list.yaml` 覆盖其中 99 个，是当前 Ascend 910B/B3 优化的优先候选集合；但最终仍要以真实 build、example、UT 和 profiling 结果确认。

## 数量汇总

| 域 | 主线 AscendC 算子数 | 910B CI 列表覆盖 | 910_93 CI 列表覆盖 | 950 CI 列表覆盖 |
| --- | ---: | ---: | ---: | ---: |
| `attention` | 40 | 38 | 38 | 17 |
| `ffn` | 5 | 4 | 2 | 0 |
| `gmm` | 7 | 5 | 5 | 5 |
| `mc2` | 27 | 14 | 22 | 18 |
| `moe` | 27 | 27 | 27 | 17 |
| `posembedding` | 11 | 11 | 10 | 7 |
| 合计 | 117 | 99 | 104 | 64 |

## 910B CI 未覆盖但仓库存在源码的主线算子

以下目录有 AscendC 源码，但未出现在 `scripts/ci/ascend910b/ops_transformer_operator_list.yaml` 中。后续若在 B3 上优化这些算子，要先确认是否支持 `ascend910b` 构建和运行。

- `attention/scatter_pa_cache`
- `attention/swin_attention_score_quant`
- `ffn/swin_transformer_ln_qkv_quant`
- `gmm/quant_grouped_matmul_dequant`
- `gmm/quant_grouped_matmul_inplace_add`
- `mc2/allto_all_all_gather_batch_mat_mul`
- `mc2/allto_allv_grouped_mat_mul`
- `mc2/attention_to_ffn`
- `mc2/batch_mat_mul_reduce_scatter_allto_all`
- `mc2/distribute_barrier`
- `mc2/ffn_to_attention`
- `mc2/grouped_mat_mul_allto_allv`
- `mc2/moe_distribute_combine_add_rms_norm`
- `mc2/moe_distribute_combine_v3`
- `mc2/moe_distribute_dispatch_v3`
- `mc2/moe_update_expert`
- `mc2/quant_all_reduce`
- `mc2/quant_reduce_scatter`

## 主线算子明细

表中“源码数”是静态扫描到的 `op_kernel` 源文件数量，用来估算源码复杂度；不是性能复杂度结论。

### attention

| 算子 | 源码目录 | 源码数 | 910B CI | 接口/验证线索 |
| --- | --- | ---: | --- | --- |
| `attention_update` | `attention/attention_update` | 6 | 是 | `op_api`, `op_graph`, `st`, `utest` |
| `attention_worker_combine` | `attention/attention_worker_combine` | 5 | 是 | `op_graph`, `ut` |
| `block_sparse_attention` | `attention/block_sparse_attention` | 53 | 是 | `op_api`, examples |
| `block_sparse_attention_grad` | `attention/block_sparse_attention_grad` | 54 | 是 | `op_api`, examples, `ut` |
| `dense_lightning_indexer_grad_kl_loss` | `attention/dense_lightning_indexer_grad_kl_loss` | 8 | 是 | examples |
| `dense_lightning_indexer_softmax_lse` | `attention/dense_lightning_indexer_softmax_lse` | 7 | 是 | examples |
| `flash_attention_score` | `attention/flash_attention_score` | 18 | 是 | `op_api`, `op_graph`, examples, `ut`, pytest |
| `flash_attention_score_grad` | `attention/flash_attention_score_grad` | 87 | 是 | `op_api`, examples, `ut`, pytest |
| `fused_floyd_attention` | `attention/fused_floyd_attention` | 3 | 是 | examples, `utest` |
| `fused_floyd_attention_grad` | `attention/fused_floyd_attention_grad` | 4 | 是 | examples, `ut` |
| `fused_infer_attention_score` | `attention/fused_infer_attention_score` | 61 | 是 | `op_api`, `op_graph`, examples, `st`, `ut`, pytest |
| `gather_pa_kv_cache` | `attention/gather_pa_kv_cache` | 6 | 是 | `op_graph`, examples, `ut` |
| `incre_flash_attention` | `attention/incre_flash_attention` | 36 | 是 | `op_api`, `op_graph`, examples, `st`, `ut` |
| `kv_quant_sparse_flash_attention` | `attention/kv_quant_sparse_flash_attention` | 9 | 是 | `ut` |
| `lightning_indexer` | `attention/lightning_indexer` | 7 | 是 | `op_api`, examples, `ut` |
| `lightning_indexer_grad` | `attention/lightning_indexer_grad` | 9 | 是 | `op_api`, `utest` |
| `mla_preprocess` | `attention/mla_preprocess` | 23 | 是 | examples, `ut` |
| `mla_preprocess_v2` | `attention/mla_preprocess_v2` | 1 | 是 | examples, `ut` |
| `mla_prolog` | `attention/mla_prolog` | 22 | 是 | `op_api`, examples, `ut` |
| `mla_prolog_v2` | `attention/mla_prolog_v2` | 1 | 是 | `op_api`, examples, `ut`, pytest |
| `mla_prolog_v3` | `attention/mla_prolog_v3` | 1 | 是 | `op_api`, examples, `ut` |
| `nsa_compress` | `attention/nsa_compress` | 5 | 是 | `op_api`, examples, `utest` |
| `nsa_compress_attention` | `attention/nsa_compress_attention` | 4 | 是 | `op_api`, examples, `utest` |
| `nsa_compress_attention_infer` | `attention/nsa_compress_attention_infer` | 3 | 是 | `op_api`, examples, `utest` |
| `nsa_compress_grad` | `attention/nsa_compress_grad` | 4 | 是 | examples, `utest` |
| `nsa_compress_with_cache` | `attention/nsa_compress_with_cache` | 1 | 是 | `op_api`, examples, `utest` |
| `nsa_selected_attention` | `attention/nsa_selected_attention` | 3 | 是 | `op_api`, examples, `utest` |
| `nsa_selected_attention_grad` | `attention/nsa_selected_attention_grad` | 4 | 是 | `op_api`, examples, `utest` |
| `nsa_selected_attention_infer` | `attention/nsa_selected_attention_infer` | 3 | 是 | `op_api`, `op_graph`, examples, `utest` |
| `prompt_flash_attention` | `attention/prompt_flash_attention` | 70 | 是 | `op_api`, `op_graph`, examples, `st`, `ut` |
| `quant_lightning_indexer` | `attention/quant_lightning_indexer` | 10 | 是 | `ut` |
| `rain_fusion_attention` | `attention/rain_fusion_attention` | 53 | 是 | `op_api`, examples |
| `recurrent_gated_delta_rule` | `attention/recurrent_gated_delta_rule` | 3 | 是 | `op_api`, examples, `ut` |
| `ring_attention_update` | `attention/ring_attention_update` | 8 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `scatter_pa_cache` | `attention/scatter_pa_cache` | 1 | 否 | `op_graph`, examples, `ut` |
| `scatter_pa_kv_cache` | `attention/scatter_pa_kv_cache` | 19 | 是 | `op_graph`, examples, `ut` |
| `sparse_flash_attention` | `attention/sparse_flash_attention` | 6 | 是 | `op_api`, examples, `ut` |
| `sparse_flash_attention_grad` | `attention/sparse_flash_attention_grad` | 22 | 是 | kernel 源码 |
| `sparse_lightning_indexer_grad_kl_loss` | `attention/sparse_lightning_indexer_grad_kl_loss` | 21 | 是 | examples |
| `swin_attention_score_quant` | `attention/swin_attention_score_quant` | 4 | 否 | `op_graph`, examples, `ut` |

### ffn

| 算子 | 源码目录 | 源码数 | 910B CI | 接口/验证线索 |
| --- | --- | ---: | --- | --- |
| `ffn` | `ffn/ffn` | 11 | 是 | `op_api`, examples, `utest` |
| `ffn_worker_batching` | `ffn/ffn_worker_batching` | 16 | 是 | `op_graph`, examples, `ut` |
| `swin_attention_ffn` | `ffn/swin_attention_ffn` | 1 | 是 | `op_graph`, examples, `ut` |
| `swin_transformer_ln_qkv` | `ffn/swin_transformer_ln_qkv` | 1 | 是 | `op_graph`, examples, `ut` |
| `swin_transformer_ln_qkv_quant` | `ffn/swin_transformer_ln_qkv_quant` | 2 | 否 | `op_graph`, examples, `ut` |

### gmm

| 算子 | 源码目录 | 源码数 | 910B CI | 接口/验证线索 |
| --- | --- | ---: | --- | --- |
| `grouped_matmul` | `gmm/grouped_matmul` | 97 | 是 | `op_api`, examples, `ut` |
| `grouped_matmul_add` | `gmm/grouped_matmul_add` | 4 | 是 | `op_api`, examples, `ut` |
| `grouped_matmul_finalize_routing` | `gmm/grouped_matmul_finalize_routing` | 11 | 是 | `op_api`, examples, `ut` |
| `grouped_matmul_swiglu_quant` | `gmm/grouped_matmul_swiglu_quant` | 8 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `grouped_matmul_swiglu_quant_v2` | `gmm/grouped_matmul_swiglu_quant_v2` | 14 | 是 | `op_api`, examples, `ut` |
| `quant_grouped_matmul_dequant` | `gmm/quant_grouped_matmul_dequant` | 5 | 否 | `op_api`, `op_graph`, examples, `ut` |
| `quant_grouped_matmul_inplace_add` | `gmm/quant_grouped_matmul_inplace_add` | 6 | 否 | `op_api`, examples, `utest` |

### mc2

| 算子 | 源码目录 | 源码数 | 910B CI | 接口/验证线索 |
| --- | --- | ---: | --- | --- |
| `all_gather_matmul` | `mc2/all_gather_matmul` | 5 | 是 | `op_api`, `op_graph`, examples, `st`, `ut` |
| `all_gather_matmul_v2` | `mc2/all_gather_matmul_v2` | 21 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `allto_all_all_gather_batch_mat_mul` | `mc2/allto_all_all_gather_batch_mat_mul` | 5 | 否 | `op_api`, `op_graph`, examples, `ut` |
| `allto_all_matmul` | `mc2/allto_all_matmul` | 17 | 是 | `op_api`, `op_graph`, examples, `st`, `ut` |
| `allto_allv_grouped_mat_mul` | `mc2/allto_allv_grouped_mat_mul` | 13 | 否 | `op_api`, `op_graph`, examples, `ut` |
| `attention_to_ffn` | `mc2/attention_to_ffn` | 4 | 否 | `op_api`, `op_graph`, examples, `ut` |
| `batch_mat_mul_reduce_scatter_allto_all` | `mc2/batch_mat_mul_reduce_scatter_allto_all` | 5 | 否 | `op_api`, `op_graph`, examples, `ut` |
| `distribute_barrier` | `mc2/distribute_barrier` | 3 | 否 | `op_api`, `op_graph`, examples, `ut` |
| `ffn_to_attention` | `mc2/ffn_to_attention` | 4 | 否 | `op_api`, `op_graph`, examples, `ut` |
| `grouped_mat_mul_all_reduce` | `mc2/grouped_mat_mul_all_reduce` | 4 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `grouped_mat_mul_allto_allv` | `mc2/grouped_mat_mul_allto_allv` | 4 | 否 | `op_api`, `op_graph`, examples, `ut` |
| `inplace_matmul_all_reduce_add_rms_norm` | `mc2/inplace_matmul_all_reduce_add_rms_norm` | 1 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `matmul_all_reduce` | `mc2/matmul_all_reduce` | 53 | 是 | `op_api`, `op_graph`, examples, `st`, `ut` |
| `matmul_all_reduce_add_rms_norm` | `mc2/matmul_all_reduce_add_rms_norm` | 13 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `matmul_allto_all` | `mc2/matmul_allto_all` | 37 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `matmul_reduce_scatter` | `mc2/matmul_reduce_scatter` | 6 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `matmul_reduce_scatter_v2` | `mc2/matmul_reduce_scatter_v2` | 22 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `moe_distribute_combine` | `mc2/moe_distribute_combine` | 3 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `moe_distribute_combine_add_rms_norm` | `mc2/moe_distribute_combine_add_rms_norm` | 1 | 否 | `op_api`, `op_graph`, examples, `ut` |
| `moe_distribute_combine_v2` | `mc2/moe_distribute_combine_v2` | 11 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `moe_distribute_combine_v3` | `mc2/moe_distribute_combine_v3` | 1 | 否 | `op_api`, `op_graph` |
| `moe_distribute_dispatch` | `mc2/moe_distribute_dispatch` | 3 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `moe_distribute_dispatch_v2` | `mc2/moe_distribute_dispatch_v2` | 19 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `moe_distribute_dispatch_v3` | `mc2/moe_distribute_dispatch_v3` | 1 | 否 | `op_api`, `op_graph` |
| `moe_update_expert` | `mc2/moe_update_expert` | 4 | 否 | `op_api`, `op_graph`, examples, `ut` |
| `quant_all_reduce` | `mc2/quant_all_reduce` | 4 | 否 | `op_api`, `op_graph`, examples, `ut` |
| `quant_reduce_scatter` | `mc2/quant_reduce_scatter` | 7 | 否 | `op_api`, `op_graph`, examples, `ut` |

### moe

| 算子 | 源码目录 | 源码数 | 910B CI | 接口/验证线索 |
| --- | --- | ---: | --- | --- |
| `moe_compute_expert_tokens` | `moe/moe_compute_expert_tokens` | 4 | 是 | `op_graph`, examples, `ut` |
| `moe_finalize_routing` | `moe/moe_finalize_routing` | 16 | 是 | `op_graph`, examples, `ut` |
| `moe_finalize_routing_v2` | `moe/moe_finalize_routing_v2` | 22 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `moe_finalize_routing_v2_grad` | `moe/moe_finalize_routing_v2_grad` | 13 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `moe_fused_topk` | `moe/moe_fused_topk` | 2 | 是 | `op_api`, examples, `ut` |
| `moe_gating_top_k` | `moe/moe_gating_top_k` | 8 | 是 | `op_graph`, examples, `ut` |
| `moe_gating_top_k_softmax` | `moe/moe_gating_top_k_softmax` | 7 | 是 | `op_graph`, examples, `ut` |
| `moe_gating_top_k_softmax_v2` | `moe/moe_gating_top_k_softmax_v2` | 10 | 是 | `op_graph`, examples, `ut` |
| `moe_init_routing` | `moe/moe_init_routing` | 23 | 是 | `op_graph`, examples, `ut` |
| `moe_init_routing_quant` | `moe/moe_init_routing_quant` | 10 | 是 | `op_graph`, examples, `ut` |
| `moe_init_routing_quant_v2` | `moe/moe_init_routing_quant_v2` | 32 | 是 | `op_graph`, examples, `ut` |
| `moe_init_routing_v2` | `moe/moe_init_routing_v2` | 24 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `moe_init_routing_v2_grad` | `moe/moe_init_routing_v2_grad` | 10 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `moe_init_routing_v3` | `moe/moe_init_routing_v3` | 42 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `moe_re_routing` | `moe/moe_re_routing` | 6 | 是 | `op_graph`, examples, `ut` |
| `moe_token_permute` | `moe/moe_token_permute` | 9 | 是 | `op_api`, examples, `ut` |
| `moe_token_permute_grad` | `moe/moe_token_permute_grad` | 2 | 是 | `op_api`, examples, `ut` |
| `moe_token_permute_with_ep` | `moe/moe_token_permute_with_ep` | 9 | 是 | examples, `ut` |
| `moe_token_permute_with_ep_grad` | `moe/moe_token_permute_with_ep_grad` | 2 | 是 | examples, `ut` |
| `moe_token_permute_with_routing_map` | `moe/moe_token_permute_with_routing_map` | 14 | 是 | `op_api`, examples, `ut` |
| `moe_token_permute_with_routing_map_grad` | `moe/moe_token_permute_with_routing_map_grad` | 3 | 是 | `op_api`, examples, `ut` |
| `moe_token_unpermute` | `moe/moe_token_unpermute` | 3 | 是 | `op_api`, examples, `ut` |
| `moe_token_unpermute_grad` | `moe/moe_token_unpermute_grad` | 4 | 是 | `op_api`, examples, `ut` |
| `moe_token_unpermute_with_ep` | `moe/moe_token_unpermute_with_ep` | 2 | 是 | examples, `ut` |
| `moe_token_unpermute_with_ep_grad` | `moe/moe_token_unpermute_with_ep_grad` | 4 | 是 | examples, `ut` |
| `moe_token_unpermute_with_routing_map` | `moe/moe_token_unpermute_with_routing_map` | 5 | 是 | `op_api`, examples, `ut` |
| `moe_token_unpermute_with_routing_map_grad` | `moe/moe_token_unpermute_with_routing_map_grad` | 12 | 是 | examples, `ut` |

### posembedding

| 算子 | 源码目录 | 源码数 | 910B CI | 接口/验证线索 |
| --- | --- | ---: | --- | --- |
| `apply_rotary_pos_emb` | `posembedding/apply_rotary_pos_emb` | 10 | 是 | `op_api`, `op_graph`, examples, `st`, `ut` |
| `dequant_rope_quant_kvcache` | `posembedding/dequant_rope_quant_kvcache` | 2 | 是 | `op_graph`, examples, `ut` |
| `interleave_rope` | `posembedding/interleave_rope` | 5 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `kv_rms_norm_rope_cache` | `posembedding/kv_rms_norm_rope_cache` | 17 | 是 | `op_graph`, examples, `ut` |
| `norm_rope_concat` | `posembedding/norm_rope_concat` | 4 | 是 | `op_api`, examples, `ut` |
| `norm_rope_concat_grad` | `posembedding/norm_rope_concat_grad` | 3 | 是 | `op_api`, examples, `ut` |
| `qkv_rms_norm_rope_cache` | `posembedding/qkv_rms_norm_rope_cache` | 3 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `rope_quant_kvcache` | `posembedding/rope_quant_kvcache` | 2 | 是 | `op_graph`, examples, `ut` |
| `rope_with_sin_cos_cache` | `posembedding/rope_with_sin_cos_cache` | 8 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `rotary_position_embedding` | `posembedding/rotary_position_embedding` | 17 | 是 | `op_api`, `op_graph`, examples, `ut` |
| `rotary_position_embedding_grad` | `posembedding/rotary_position_embedding_grad` | 16 | 是 | `op_graph`, examples, `ut` |

## 非主线但存在 AscendC 源码的目录

以下目录本轮不计入主线算子列表，但后续可能需要按任务进入：

- `experimental/`：包含实验性算子和专项 benchmark，例如 experimental attention、GMM、MC2、MoE 等。只有用户明确要求优化 experimental 算子时才纳入。
- `examples/`：包含 `add_example`、`mc2/all_gather_add`、`fast_kernel_launch_example` 等样例 AscendC 源码，主要用于开发示范，不作为产品算子列表。
- `attention/common/`、`mc2/3rd/`、`moe/3rd/` 等公共或第三方目录：可能影响多个算子，但默认作为依赖库处理，优化前要确认调用关系和影响面。
- `scripts/opgen/template/`：模板代码，不是可直接优化的仓库算子。

## 与官方 op list 的关系

`docs/zh/op_list.md` 是公开算子清单，适合确认用户可见接口；但它不是源码覆盖的完整列表。本轮主列表以仓库内 `op_kernel` AscendC 源码为准，因此会包含部分未出现在公开 op list 中、但源码实际存在的目录。
