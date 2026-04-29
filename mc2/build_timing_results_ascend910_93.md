# MC2算子编译耗时统计（A3）

## 编译环境
- 芯片：A3 (ascend910_93)
- 模式：--pkg
- 算子数：22
- 时间：2026-05-08 11:03:09

| 序号 | 算子名称 | 编译耗时 | 状态 | 产物大小 |
|------|---------|---------|------|---------|
| 1 | all_gather_matmul | 0分9秒 | ✓ | 3.5M |
| 2 | all_gather_matmul_v2 | 0分9秒 | ✓ | 3.6M |
| 3 | allto_all_all_gather_batch_mat_mul | 0分9秒 | ✓ | 6.0M |
| 4 | allto_all_matmul | 0分10秒 | ✓ | 3.7M |
| 5 | allto_allv_grouped_mat_mul | 0分9秒 | ✓ | 3.9M |
| 6 | attention_to_ffn | 0分9秒 | ✓ | 2.9M |
| 7 | batch_mat_mul_reduce_scatter_allto_all | 0分9秒 | ✓ | 3.9M |
| 8 | distribute_barrier | 0分9秒 | ✓ | 2.9M |
| 9 | ffn_to_attention | 0分9秒 | ✓ | 2.9M |
| 10 | grouped_mat_mul_all_reduce | 0分9秒 | ✓ | 2.9M |
| 11 | grouped_mat_mul_allto_allv | 0分10秒 | ✓ | 3.4M |
| 12 | matmul_allto_all | 0分9秒 | ✓ | 3.4M |
| 13 | matmul_reduce_scatter | 0分10秒 | ✓ | 3.4M |
| 14 | matmul_reduce_scatter_v2 | 0分9秒 | ✓ | 3.7M |
| 15 | moe_distribute_combine | 0分11秒 | ✓ | 3.7M |
| 16 | moe_distribute_combine_add_rms_norm | 0分11秒 | ✓ | 3.7M |
| 17 | moe_distribute_combine_v2 | 0分11秒 | ✓ | 3.8M |
| 18 | moe_distribute_combine_v3 | 0分10秒 | ✓ | 3.5M |
| 19 | moe_distribute_dispatch | 0分10秒 | ✓ | 3.7M |
| 20 | moe_distribute_dispatch_v2 | 0分13秒 | ✓ | 4.5M |
| 21 | moe_distribute_dispatch_v3 | 0分11秒 | ✓ | 3.6M |
| 22 | moe_update_expert | 0分8秒 | ✓ | 2.9M |

## 统计汇总
- 总算子：22
- 成功：22 ✓
- 失败：0 ✗
- 成功率：100.00%
- 总耗时：0时3分35秒
- 平均耗时：0分9秒


## 编译时间排序（从快到慢）

| 排名 | 算子名称 | 编译耗时 | 产物大小 |
|------|---------|---------|---------|
| 1 | moe_update_expert | 0分8秒 | 2.9M |
| 2 | all_gather_matmul | 0分9秒 | 3.5M |
| 3 | all_gather_matmul_v2 | 0分9秒 | 3.6M |
| 4 | allto_all_all_gather_batch_mat_mul | 0分9秒 | 6.0M |
| 5 | allto_allv_grouped_mat_mul | 0分9秒 | 3.9M |
| 6 | attention_to_ffn | 0分9秒 | 2.9M |
| 7 | batch_mat_mul_reduce_scatter_allto_all | 0分9秒 | 3.9M |
| 8 | distribute_barrier | 0分9秒 | 2.9M |
| 9 | ffn_to_attention | 0分9秒 | 2.9M |
| 10 | grouped_mat_mul_all_reduce | 0分9秒 | 2.9M |
| 11 | matmul_allto_all | 0分9秒 | 3.4M |
| 12 | matmul_reduce_scatter_v2 | 0分9秒 | 3.7M |
| 13 | allto_all_matmul | 0分10秒 | 3.7M |
| 14 | grouped_mat_mul_allto_allv | 0分10秒 | 3.4M |
| 15 | matmul_reduce_scatter | 0分10秒 | 3.4M |
| 16 | moe_distribute_combine_v3 | 0分10秒 | 3.5M |
| 17 | moe_distribute_dispatch | 0分10秒 | 3.7M |
| 18 | moe_distribute_combine | 0分11秒 | 3.7M |
| 19 | moe_distribute_combine_add_rms_norm | 0分11秒 | 3.7M |
| 20 | moe_distribute_combine_v2 | 0分11秒 | 3.8M |
| 21 | moe_distribute_dispatch_v3 | 0分11秒 | 3.6M |
| 22 | moe_distribute_dispatch_v2 | 0分13秒 | 4.5M |
