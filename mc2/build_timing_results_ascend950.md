# MC2算子编译耗时统计（A5）

## 编译环境
- 芯片：A5 (ascend950)
- 模式：--pkg
- 算子数：22
- 时间：2026-05-08 11:27:17

| 序号 | 算子名称 | 编译耗时 | 状态 | 产物大小 |
|------|---------|---------|------|---------|
| 1 | all_gather_matmul | 0分9秒 | ✓ | 3.1M |
| 2 | all_gather_matmul_v2 | 0分12秒 | ✓ | 19M |
| 3 | allto_all_matmul | 0分17秒 | ✓ | 23M |
| 4 | allto_allv_grouped_mat_mul | 0分9秒 | ✓ | 3.6M |
| 5 | allto_allv_quant_grouped_mat_mul | 0分11秒 | ✓ | 11M |
| 6 | distribute_barrier | 0分9秒 | ✓ | 3.0M |
| 7 | grouped_mat_mul_allto_allv | 0分9秒 | ✓ | 3.4M |
| 8 | matmul_all_reduce | 0分49秒 | ✓ | 44M |
| 9 | matmul_allto_all | 0分10秒 | ✓ | 7.0M |
| 10 | matmul_reduce_scatter | 0分10秒 | ✓ | 3.1M |
| 11 | matmul_reduce_scatter_v2 | 0分18秒 | ✓ | 36M |
| 12 | moe_distribute_combine | 0分11秒 | ✓ | 3.7M |
| 13 | moe_distribute_combine_add_rms_norm | 0分11秒 | ✓ | 3.9M |
| 14 | moe_distribute_combine_v2 | 0分10秒 | ✓ | 4.1M |
| 15 | moe_distribute_combine_v3 | 0分9秒 | ✓ | 3.8M |
| 16 | moe_distribute_dispatch | 0分9秒 | ✓ | 3.5M |
| 17 | moe_distribute_dispatch_v2 | 0分20秒 | ✓ | 19M |
| 18 | moe_distribute_dispatch_v3 | 0分12秒 | ✓ | 9.3M |
| 19 | moe_update_expert | 0分8秒 | ✓ | 2.9M |
| 20 | quant_all_reduce | 0分10秒 | ✓ | 3.1M |
| 21 | quant_grouped_mat_mul_allto_allv | 0分16秒 | ✓ | 17M |
| 22 | quant_reduce_scatter | 0分9秒 | ✓ | 3.1M |

## 统计汇总
- 总算子：22
- 成功：22 ✓
- 失败：0 ✗
- 成功率：100.00%
- 总耗时：0时4分53秒
- 平均耗时：0分13秒


## 编译时间排序（从快到慢）

| 排名 | 算子名称 | 编译耗时 | 产物大小 |
|------|---------|---------|---------|
| 1 | moe_update_expert | 0分8秒 | 2.9M |
| 2 | all_gather_matmul | 0分9秒 | 3.1M |
| 3 | allto_allv_grouped_mat_mul | 0分9秒 | 3.6M |
| 4 | distribute_barrier | 0分9秒 | 3.0M |
| 5 | grouped_mat_mul_allto_allv | 0分9秒 | 3.4M |
| 6 | moe_distribute_combine_v3 | 0分9秒 | 3.8M |
| 7 | moe_distribute_dispatch | 0分9秒 | 3.5M |
| 8 | quant_reduce_scatter | 0分9秒 | 3.1M |
| 9 | matmul_allto_all | 0分10秒 | 7.0M |
| 10 | matmul_reduce_scatter | 0分10秒 | 3.1M |
| 11 | moe_distribute_combine_v2 | 0分10秒 | 4.1M |
| 12 | quant_all_reduce | 0分10秒 | 3.1M |
| 13 | allto_allv_quant_grouped_mat_mul | 0分11秒 | 11M |
| 14 | moe_distribute_combine | 0分11秒 | 3.7M |
| 15 | moe_distribute_combine_add_rms_norm | 0分11秒 | 3.9M |
| 16 | all_gather_matmul_v2 | 0分12秒 | 19M |
| 17 | moe_distribute_dispatch_v3 | 0分12秒 | 9.3M |
| 18 | quant_grouped_mat_mul_allto_allv | 0分16秒 | 17M |
| 19 | allto_all_matmul | 0分17秒 | 23M |
| 20 | matmul_reduce_scatter_v2 | 0分18秒 | 36M |
| 21 | moe_distribute_dispatch_v2 | 0分20秒 | 19M |
| 22 | matmul_all_reduce | 0分49秒 | 44M |
