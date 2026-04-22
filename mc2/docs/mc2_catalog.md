# mc2 算子库全面分析 — 总览目录

## 分析分组

| 分组 | 算子 | 报告文件 | 状态 |
|------|------|---------|------|
| **通信原语+Pipeline** | distribute_barrier/, quant_all_reduce/, quant_reduce_scatter/, attention_to_ffn/, ffn_to_attention/ | [basic_ops_batch1_analysis.md](basic_ops_batch1_analysis.md) | 已完成 |
| **Matmul+Comm融合** | matmul_all_reduce, matmul_allto_all, matmul_reduce_scatter/v2, all_gather_matmul/v2, matmul_all_reduce_add_rms_norm, inplace_matmul_all_reduce_add_rms_norm | [matmul_comm_fusion_batch2_analysis.md](matmul_comm_fusion_batch2_analysis.md) | 已完成 |
| **GroupedMatmul+复杂融合** | grouped_mat_mul_all_reduce, grouped_mat_mul_allto_allv, quant_grouped_mat_mul_allto_allv, allto_all_all_gather_batch_mat_mul, batch_mat_mul_reduce_scatter_allto_all, allto_allv_grouped_mat_mul, allto_allv_quant_grouped_mat_mul, allto_all_matmul | [grouped_complex_fusion_batch3_analysis.md](grouped_complex_fusion_batch3_analysis.md) | 已完成 |
| **MOE 生命周期+Dispatch/Combine** | moe_distribute_dispatch (v1/v3), moe_distribute_combine (v1/v3), moe_*_setup, moe_*_teardown, moe_distribute_combine_add_rms_norm, moe_update_expert | [moe_lifecycle_batch4_analysis.md](moe_lifecycle_batch4_analysis.md) | 已完成 |

## 已完成的分析

- [moe_distribute_dispatch_v2_op_kernel_analysis.md](moe_distribute_dispatch_v2_op_kernel_analysis.md) — DispatchV2 op_kernel 详细分析
- [moe_distribute_dispatch_a2_layered_merged.md](moe_distribute_dispatch_a2_layered_merged.md) — A2 Layered Dispatch 分析
- [moe_distribute_combine_a2_layered_analysis.md](moe_distribute_combine_a2_layered_analysis.md) — A2 Layered Combine 分析
- [moe_distribute_combine_v2_op_kernel_full_analysis.md](moe_distribute_combine_v2_op_kernel_full_analysis.md) — CombineV2 op_kernel 全量分析
