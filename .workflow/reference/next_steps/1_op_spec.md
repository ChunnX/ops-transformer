# 任务 1：编写算子原型描述文档

## 背景

GMM A16W4(INT4) weight ND 数据流需要在 Ascend 950 平台上支持 pergroup 伪量化场景。在开始编码之前，需要先完成算子原型描述文档，明确该数据流的输入输出规格、计算公式、约束条件等，作为后续 tiling 和 kernel 实现的输入依据。

本任务参照已有的 `gmmfr_op.md`（GMMFR MxA8W4 weightNz 算子规格文档）的格式和详细程度来编写。

## 任务详情

### 目标

完成 `.workflow/raw/spec/gmm_a16w4nd_op.md`，作为 GMM A16W4 weight ND pergroup 数据流的算子原型描述文档。

### 参考文档

| 文档 | 用途 |
|------|------|
| `.workflow/raw/spec/gmmfr_op.md` | **格式参照**：算子规格文档的写作格式和详细程度 |
| `gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md` | **参数来源**：A2/A3 A16W4 场景约束章节，提供 dtype、shape、约束等参数规格 |
| `gmm/grouped_matmul/op_host/grouped_matmul_def.cpp` | **参数定义**：算子输入参数的 OpDef 定义（Input/Output/Attr 的 dtype、format、paramType） |
| `.workflow/raw/spec/gmm_a16w4_950_requirement.md` | **需求约束**：用户需求中的关键约束（group_size、K 轴对齐、antiquant_scale shape 等） |

### 文档应包含的内容

参照 `gmmfr_op.md` 的结构，文档应包含：

1. **需求背景**
   - 数据流类型说明（A16W4 伪量化 pergroup ND）
   - 计算公式：$y_i = x_i \times (weight_i + antiquant\_offset_i) * antiquant\_scale_i + bias_i$
   - 与 A2/A3 已有场景的关系（兼容性扩展）

2. **输入规格 — Tensor 参数**
   - x: FLOAT16/BFLOAT16, ND, shape (m, k) 或多多多模式
   - weight: INT4, ND, 支持转置/非转置
   - antiquant_scale: FLOAT16/BFLOAT16, ND, shape (e, k/gs, n) pergroup 或 (e, n) perchannel
   - antiquant_offset: FLOAT16/BFLOAT16, ND, shape 同 antiquant_scale（可选）
   - bias: FLOAT16/BFLOAT32/null
   - groupList: INT64, ND
   - 其余参数（scale、offset、perTokenScale 等）在本场景为 null

3. **输入规格 — Scalar / 属性参数**
   - splitItem, groupType, groupListType, actType 等
   - group_size 约束：仅 32/64/128/256

4. **约束条件**
   - K % group_size == 0
   - weight 最后一维为偶数
   - 单单单 / 多多多 两种 tensor 模式的具体约束
   - 转置/非转置约束

5. **输出规格**
   - out: FLOAT16/BFLOAT16, ND

### 完成标准

- [ ] 文档格式与 `gmmfr_op.md` 一致
- [ ] 所有参数规格与 `aclnnGroupedMatmulV5.md` A2/A3 A16W4 章节一致
- [ ] 约束条件完整覆盖用户需求中的所有约束
- [ ] 计算公式正确

## 任务计划

1. 读取 `gmmfr_op.md` 确认文档格式
2. 读取 `aclnnGroupedMatmulV5.md` 中 A2/A3 A16W4 场景约束章节，提取参数规格
3. 读取 `grouped_matmul_def.cpp` 确认 OpDef 参数定义
4. 编写 `gmm_a16w4nd_op.md`
5. 自检：对照需求约束逐项检查完整性
