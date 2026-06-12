# GMM A16W4(INT4) Weight ND 数据流 — Ascend950 场景需求

> 用户原始需求，不可修改。

## 需求描述

给GMM（gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md）支持 A16W4(INT4) weight ND数据流，weight支持转置和非转置，group_size仅支持32/64/128/256，K轴为group_size的倍数。

该数据流中A2/A3场景约束已存在，现在需要补充Ascend950场景的A16W4场景，对原有A2/A3场景做兼容，输入支持要求支持x、weight、out有单单单和多多多两种。

## 关键约束提取

| 约束项 | 说明 |
|--------|------|
| 目标平台 | Ascend 950PR / Ascend 950DT |
| 数据流类型 | A16W4 伪量化（pseudo-quantization） |
| x 数据类型 | FLOAT16 / BFLOAT16（A16） |
| weight 数据类型 | INT4 |
| weight 数据格式 | ND（非NZ） |
| weight 转置 | 支持转置和非转置 |
| group_size | 仅支持 32 / 64 / 128 / 256 |
| K轴约束 | K 必须为 group_size 的倍数 |
| antiquant_scale | (e, k/gs, n) |
| antiquantOffset | (e, k/gs, n) 可选 |
| weight 数据类型 | INT4 |
| 输入模式 | 单单单（x单tensor, weight单tensor, out单tensor）和 多多多（x多tensor, weight多tensor, out多tensor） |
| 兼容性 | 对原有 A2/A3 场景做兼容，不能破坏已有功能 |

## 参考文档

- 接口规格：`gmm/grouped_matmul/docs/aclnnGroupedMatmulV5.md`
- 现有 A2/A3 A16W4 约束：参见 aclnnGroupedMatmulV5.md 中 "A16W4场景约束" 章节的约束说明

## 重要澄清

- **范围边界**：本任务仅关注 **tiling + kernel** 部分的实现。Aclnn 通路（API checker、infershape）由其他人完成。
- **参照实现**：`GMMWeightQuantBasicController`（A16W8 多多多），位于 `gmm/grouped_matmul/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h`。大部分代码可能不需要修改，核心是适配 pergroup 计算、地址偏移、ND反量化VF函数实现。
- **本任务核心**：在 arch35 的 `weight_quant_basic_block` 体系中，参照 A16W8 多多多 Controller，适配 A16W4 pergroup ND 场景。

## 现有 A2/A3 A16W4 场景约束摘要（供参考）

- x: FLOAT16/BFLOAT16, weight: INT4(ND), out: FLOAT16/BFLOAT16
- x不支持转置
- 仅支持GroupType=-1、0，actType=0，groupListType=0/1/2
- 对称量化支持perchannel和pergroup，非对称量化仅支持perchannel
- pergroup数G必须能整除k_i
- 若weight为多tensor，所有pergroup长度s_i = k_i/G_i 必须相等
- antiquantScale/antiquantOffset shape约束见规格文档

## 现有 Ascend 950 伪量化约束摘要（供参考）

- groupType: -1/0（伪量化场景）
- weight INT4时：最后一维大小应是偶数
- antiquantScale shape: groupType=-1 weight多tensor时为(n_i)，groupType=0 weight单tensor时为(E, N)
- 支持场景：多多多（groupType=-1）和 单单单（groupType=0）

## 实现方案

- tiling 策略参考GMM A16W8 多多多策略，x: `256*256` L1切分，weight：`256*256`，kL1=256
- kernel参考GMMWeightQuantBasicController，用多多多计算方式，同时实现单单单
- kernel的cube主要流程与A16W8多多多方案一致
- vec反量化写NK/KN两份功能实现代码，代码参考 [wqbmmv2的weight ND int4的反量化方法](/home/sunduiyang/Code/ops-nn/matmul/weight_quant_batch_matmul_v2/op_kernel/arch35/weight_quant_batch_matmul_v2_reg_base_common.h)，核心函数为AntiquantComputeW4NKPerGroup和AntiquantComputeKNPerGroup。UB划分参考GMM A16W8实现，暂时不修改
- 每个基本块计算完成后地址计算更新按int4计算字节偏移，字节大小需要减半