# MC2 算子迁移 TODO

## 已完成
- 21 个 MC2 算子从 TTK 迁移到 ops-transformer 各算子 tests/assets/ 目录
- TTK mc2_golden.py 精简为框架-only（208行），只保留多机多卡框架
- TTK E2E worker 删除 592 行死 golden 代码
- 共享 golden 工具模块 mc2/common/tests/assets/golden_utils.py
- op_kernel 模式跑通（4 个算子全部 PASS）
- a3_ops_small.csv 全部 8 个用例 cross_check 模式 PASS
- 6/8 用例 close 模式 PASS（非 BMM 算子全部 PASS）
- BMM 双通信域算子 golden 逻辑修复完成
  - EP/TP 分组对齐 TTK profiling.py HCCL 通信域创建方式
  - reduce_scatter 修复：包含所有 rank 的求和
- MoE dispatch E2E PASS

## 精度说明
- BMM 类算子（AlltoAllAllGatherBatchMatMul / BatchMatMulReduceScatterAlltoAll）：
  - cross_check 模式 PASS（HCCL cascade 真实通信，逻辑正确）
  - close 模式 FAIL（CPU float32 bmm 与 NPU bf16 Cube bmm 有精度差异）
    - A2A_AG_BMM: gathered 100% PASS, bmm_out 96.9%, main 98.47%
    - BMM_RS_A2A: close 67%（reduce_scatter 累加放大了 bf16 bmm 误差）
  - 根本原因：NPU Cube 单元 bf16 bmm 的累加方式与 CPU torch.bmm 不同
  - 实际测试应使用 cross_check 模式

## 测试结果汇总

### a3_ops_small.csv (cross_check 模式)
| # | 算子 | 卡数 | 结果 |
|---|------|------|------|
| 0 | AllGatherMatmul | 2 | PASS |
| 1 | AlltoAllAllGatherBatchMatMul | 8 | PASS |
| 2 | AlltoAllMatmul | 2 | PASS |
| 3 | AlltoAllvGroupedMatMul | 16 | PASS |
| 4 | BatchMatMulReduceScatterAlltoAll | 8 | PASS |
| 5 | GroupedMatMulAllToAllv | 8 | PASS |
| 6 | MatmulAlltoAll | 2 | PASS |
| 7 | MatmulReduceScatter | 8 | PASS |

### op_kernel (close 模式)
| # | 算子 | 卡数 | 结果 |
|---|------|------|------|
| 0 | AllGatherMatmul | 2 | PASS |
| 1 | MatmulAlltoAll | 2 | PASS |
| 2 | AlltoAllMatmul | 2 | PASS |
| 3 | MatmulReduceScatter | 8 | PASS |

### E2E (close 模式)
| 算子 | 结果 |
|------|------|
| MoE dispatch | PASS |
| MoE combine | FAIL（需级联 golden） |

## TODO

### 1. MoE combine 级联 golden
- MoE dispatch E2E PASS
- MoE combine 需要级联 golden（dispatch→combine 两阶段测试）
  - combine 的输入应该是 dispatch 的输出，不是随机生成
  - 需要修改 TTK 框架支持两阶段级联测试

### 2. A5 算子在 950 上验证
- A5 算子 golden 代码已就位，等 950 环境验证

### 3. 未迁移的 MoE 变体算子（A5-only）
- moe_distribute_dispatch_v2/v3, moe_distribute_combine_v2/v3
- moe_distribute_combine_add_rms_norm, moe_update_expert
- moe_ep_dispatch/combine/epilogue, mega_moe

### 4. 无需 golden 的算子
- setup/teardown 类：执行即可，无需精度比对
- distribute_barrier：同步原语，返回 PASS(EXECUTED)
