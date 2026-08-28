# 任务：分析 FP8 golden 能否覆盖 BSA 量化全部特性

## 目标
对照 ops-transformer_9.1 下 BSA 量化实现，判断 `new_fp8_aclnn_blocksparseattention.py` 的特性覆盖缺口。

## 待办事项
- [ ] 梳理 9.1 BSA 量化：dtype、quantMode、layout、mask、seq、innerPrecise、LSE、GQA
- [ ] 对照 FP8 golden 实际实现与 ATK 用例
- [ ] 对照 kernel/tiling 已实例化的量化分支
- [ ] 给出覆盖结论与缺口清单

## 进度
0/4
