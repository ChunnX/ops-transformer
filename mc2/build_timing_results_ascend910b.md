# MC2算子编译耗时统计（A2）

## 编译环境
- 芯片：A2 (ascend910b)
- 模式：--pkg
- 算子数：14
- 时间：2026-05-08 11:07:14

| 序号 | 算子名称 | 编译耗时 | 状态 | 产物大小 |
|------|---------|---------|------|---------|
| 1 | all_gather_matmul | 0分9秒 | ✓ | 3.5M |
| 2 | all_gather_matmul_v2 | 0分9秒 | ✓ | 3.6M |
| 3 | allto_all_matmul | 0分12秒 | ✓ | 6.3M |
| 4 | grouped_mat_mul_all_reduce | 0分8秒 | ✓ | 2.9M |
| 5 | inplace_matmul_all_reduce_add_rms_norm | 0分10秒 | ✓ | 4.0M |
| 6 | matmul_all_reduce | 0分19秒 | ✓ | 18M |
| 7 | matmul_all_reduce_add_rms_norm | 0分9秒 | ✓ | 3.9M |
| 8 | matmul_allto_all | 0分10秒 | ✓ | 3.6M |
| 9 | matmul_reduce_scatter | 0分9秒 | ✓ | 3.4M |
| 10 | matmul_reduce_scatter_v2 | 0分10秒 | ✓ | 3.7M |
| 11 | moe_distribute_combine | 0分10秒 | ✓ | 3.7M |
| 12 | moe_distribute_combine_v2 | 0分10秒 | ✓ | 3.6M |
| 13 | moe_distribute_dispatch | 0分9秒 | ✓ | 3.7M |
| 14 | moe_distribute_dispatch_v2 | 0分11秒 | ✓ | 4.0M |

## 统计汇总
- 总算子：14
- 成功：14 ✓
- 失败：0 ✗
- 成功率：100.00%
- 总耗时：0时2分30秒
- 平均耗时：0分10秒


## 编译时间排序（从快到慢）

| 排名 | 算子名称 | 编译耗时 | 产物大小 |
|------|---------|---------|---------|
| 1 | grouped_mat_mul_all_reduce | 0分8秒 | 2.9M |
| 2 | all_gather_matmul | 0分9秒 | 3.5M |
| 3 | all_gather_matmul_v2 | 0分9秒 | 3.6M |
| 4 | matmul_all_reduce_add_rms_norm | 0分9秒 | 3.9M |
| 5 | matmul_reduce_scatter | 0分9秒 | 3.4M |
| 6 | moe_distribute_dispatch | 0分9秒 | 3.7M |
| 7 | inplace_matmul_all_reduce_add_rms_norm | 0分10秒 | 4.0M |
| 8 | matmul_allto_all | 0分10秒 | 3.6M |
| 9 | matmul_reduce_scatter_v2 | 0分10秒 | 3.7M |
| 10 | moe_distribute_combine | 0分10秒 | 3.7M |
| 11 | moe_distribute_combine_v2 | 0分10秒 | 3.6M |
| 12 | moe_distribute_dispatch_v2 | 0分11秒 | 4.0M |
| 13 | allto_all_matmul | 0分12秒 | 6.3M |
| 14 | matmul_all_reduce | 0分19秒 | 18M |
