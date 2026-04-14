# npu_flash_attn 拦截模块任务拆分

按照参数组拆分模块，每个参数组对应一个独立的Checker类，每个Checker类需要实现以下四个方法：
- CheckSinglePara: 单参数约束
- CheckParaExistence: 存在性约束
- CheckFeature: 特性交叉约束
- CheckMultiPara: 一致性约束

---

## 模块 1: ShapeChecker（公共参数组）

### 任务 1.1: ShapeChecker::CheckSinglePara
**指令**: 实现单参数约束验证：
- 无单参数约束
- 直接返回 GRAPH_SUCCESS

**实现状态**: ✅ 已完成
**代码位置**: shape_checker.cpp:1061-1071

### 任务 1.2: ShapeChecker::CheckParaExistence
**指令**: 实现存在性约束验证：
- 检查 query、key、value、attentionOut 是否存在 nullptr
- 若存在 nullptr，返回 GRAPH_FAILED
- 否则返回 GRAPH_SUCCESS

**实现状态**: ✅ 已完成
**代码位置**: shape_checker.cpp:45-60 (CheckParaExistenceImpl), 1073-1079 (CheckParaExistence)

### 任务 1.3: ShapeChecker::CheckFeature
**指令**: 实现特性交叉约束验证：
- 验证 PagedAttention 场景下 Key/Value 的排布格式：
  - 当 layout_kv 为 PA_BBND、PA_BNBD 或 PA_Nz 时，验证排布格式是否正确
- 否则返回 GRAPH_SUCCESS

**实现状态**: ✅ 已完成
**代码位置**: shape_checker.cpp:80-205 (CheckPAKeyValue), 1081-1106 (CheckFeature)

### 任务 1.4: ShapeChecker::CheckMultiPara
**指令**: 实现一致性约束验证：
- 验证 q、k、v 张量的维度一致性约束：
  - B轴：支持小于等于65536
  - N轴：无限制
  - S轴：无限制
  - D轴：小于等于512，且16对齐
  - N_Q 是 N_KV 的整数倍
- 验证 query、key、value、attentionOut 的数据类型一致性（BF16/FP16）
- 验证 attentionOut 的 D 维度与 value 的 D 维度一致
- 验证 attentionOut 的其余维度与 query 的 shape 一致

**实现状态**: ✅ 已完成
**代码位置**: 
- shape_checker.cpp:427-482 (CheckAxis - B/D轴验证)
- shape_checker.cpp:826-846 (CheckNonQuantHeadNum - N_Q是N_KV整数倍验证)
- shape_checker.cpp:798-814 (CheckNonQuantDataType - 数据类型一致性)
- shape_checker.cpp:534-574 (CheckQueryOutConsistency - attentionOut与query shape一致性)
- shape_checker.cpp:1108-1123 (CheckMultiPara)

---

## 模块 2: MaskChecker（Mask参数组）

### 任务 2.1: MaskChecker::CheckSinglePara
**指令**: 实现单参数约束验证：
- 验证 mask_mode 参数的取值范围：
  - 支持 1、3、4
  - 默认值为 1
  - 其他值返回 GRAPH_FAILED
- 验证 win_left、win_right 参数为 int 类型
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

### 任务 2.2: MaskChecker::CheckParaExistence
**指令**: 实现存在性约束验证：
- 无存在性约束
- 直接返回 GRAPH_SUCCESS

### 任务 2.3: MaskChecker::CheckFeature
**指令**: 实现特性交叉约束验证：
- 无特性交叉约束
- 直接返回 GRAPH_SUCCESS

### 任务 2.4: MaskChecker::CheckMultiPara
**指令**: 实现一致性约束验证：
- 无一致性约束
- 直接返回 GRAPH_SUCCESS

---

## 模块 3: ActualSeqLenChecker（SeqLengths参数组）

### 任务 3.1: ActualSeqLenChecker::CheckSinglePara
**指令**: 实现单参数约束验证：
- 验证 seqused_qused_q 参数：
  - 长度应等于 batch 数
  - 入参中的数值应为非负整数
- 验证 seqused_kvused_kv 参数：
  - 长度应等于 batch 数
  - 入参中的数值应为非负整数
- 验证 cu_seqlens_qcu_seqlens_q 参数：
  - 长度应等于 batch 数 + 1
  - 入参中的数值应为非负整数
- 验证 cu_seqlens_kvcu_seqlens_kv 参数：
  - 长度应等于 batch 数 + 1
  - 入参中的数值应为非负整数
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

### 任务 3.2: ActualSeqLenChecker::CheckParaExistence
**指令**: 实现存在性约束验证：
- PagedAttention 场景下，必须传入 seqused_kv，否则返回 GRAPH_FAILED
- 当 layout_q 为 TND 时，必须传入 cu_seqlens_q，否则返回 GRAPH_FAILED
- 当 layout_kv 为 TND 时，必须传入 cu_seqlens_kv，否则返回 GRAPH_FAILED
- 否则返回 GRAPH_SUCCESS

### 任务 3.3: ActualSeqLenChecker::CheckFeature
**指令**: 实现特性交叉约束验证：
- 无特性交叉约束
- 直接返回 GRAPH_SUCCESS

### 任务 3.4: ActualSeqLenChecker::CheckMultiPara
**指令**: 实现一致性约束验证：
- 验证 cu_seqlens_qcu_seqlens_q 参数的数值约束：
  - 当 layout_q 为 TND 时：
    - 其值应递增（大于等于前一个值）排列
    - 第一个元素为0且最后一个元素应等于 T_Q
- 验证 cu_seqlens_kvcu_seqlens_kv 参数的数值约束：
  - 当 layout_kv 为 TND 时：
    - 其值应递增（大于等于前一个值）排列
    - 第一个元素为0且最后一个元素应等于 T_KV
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

---

## 模块 4: PagedAttentionChecker（Paged Attention参数组）

### 任务 4.1: PagedAttentionChecker::CheckSinglePara
**指令**: 实现单参数约束验证：
- 验证 blockTable 参数：
  - tensor dtype 必须为 int32 类型
  - tensor shape 为2维
  - 每一维 dim value 取值均不能为0
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

### 任务 4.2: PagedAttentionChecker::CheckParaExistence
**指令**: 实现存在性约束验证：
- PagedAttention 使能情况下（layout_kv 为 PA_BBND、PA_BNBD 或 PA_Nz），必须传入 seqused_kv
- 否则返回 GRAPH_FAILED
- 否则返回 GRAPH_SUCCESS

### 任务 4.3: PagedAttentionChecker::CheckFeature
**指令**: 实现特性交叉约束验证：
- 无特性交叉约束
- 直接返回 GRAPH_SUCCESS

### 任务 4.4: PagedAttentionChecker::CheckMultiPara
**指令**: 实现一致性约束验证：
- 验证 blockTable 的 shape：
  - 第一维长度需等于 Batch size
  - 第二维长度不能小于 maxBlockNumPerSeq（maxBlockNumPerSeq 为每个 batch 中最大 actualSeqLengthsKv 对应的 block 数量）
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

---

## 模块 5: LearnableSinkChecker（LearnableSink参数组）

### 任务 5.1: LearnableSinkChecker::CheckSinglePara
**指令**: 实现单参数约束验证：
- 验证 sinks 参数：
  - tensor dtype 必须为 FP32
  - tensor shape 必须为 (N_Q)
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

### 任务 5.2: LearnableSinkChecker::CheckParaExistence
**指令**: 实现存在性约束验证：
- 无存在性约束
- 直接返回 GRAPH_SUCCESS

### 任务 5.3: LearnableSinkChecker::CheckFeature
**指令**: 实现特性交叉约束验证：
- 无特性交叉约束
- 直接返回 GRAPH_SUCCESS

### 任务 5.4: LearnableSinkChecker::CheckMultiPara
**指令**: 实现一致性约束验证：
- 无一致性约束
- 直接返回 GRAPH_SUCCESS

---

## 模块 6: SoftmaxLSEChecker（SoftmaxLSE参数组）

### 任务 6.1: SoftmaxLSEChecker::CheckSinglePara
**指令**: 实现单参数约束验证：
- 验证输出 lseOut 参数：
  - 仅支持数据类型 FP32
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

### 任务 6.2: SoftmaxLSEChecker::CheckParaExistence
**指令**: 实现存在性约束验证：
- softmaxLSE 场景下（return_softmax_lse 为 1），输出 lseverOut 不应为空
- 否则返回 GRAPH_FAILED
- 否则返回 GRAPH_SUCCESS

### 任务 6.3: SoftmaxLSEChecker::CheckFeature
**指令**: 实现特性交叉约束验证：
- 无特性交叉约束
- 直接返回 GRAPH_SUCCESS

### 任务 6.4: SoftmaxLSEChecker::CheckMultiPara
**指令**: 实现一致性约束验证：
- 验证 lseverOut 的输出维度和 shape：
  - softmaxLSE 且非空 tensor 场景下：
    - 如输出 layout 为 TND，则 lseverOut 输出维度应为 2，且 shape 匹配 [N_Q，T_Q]
    - 如输出 layout 不为 TND，则 lseverOut 输出维度应为 3，且 shape 匹配 [B，N_Q，Q_S]
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

---

## 模块 7: LayoutChecker（Layout匹配关系验证）

### 任务 7.1: LayoutChecker::CheckSinglePara
**指令**: 实现单参数约束验证：
- 验证 layout_q 参数：支持 "BNSD"、"BSND"、"TND"
- 验证 layout_kv 参数：支持 "BNSD"、"BSND"、"TND"、"PA_BBND"、"PA_BNBD"、"PA_Nz"
- 验证 layout_out 参数：支持 "BNSD"、"BSND"、"TND"
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

### 任务 7.2: LayoutChecker::CheckParaExistence
**指令**: 实现存在性约束验证：
- 无存在性约束
- 直接返回 GRAPH_SUCCESS

### 任务 7.3: LayoutChecker::CheckFeature
**指令**: 实现特性交叉约束验证：
- 无特性交叉约束
- 直接返回 GRAPH_SUCCESS

### 任务 7.4: LayoutChecker::CheckMultiPara
**指令**: 实现一致性约束验证：
- 验证 layout 匹配关系：
  - 当 layout_q 为 BNSD 时：
    - layout_kv 支持：BNSD、PA_BBND、PA_BNBD、PA_Nz
    - layout_out 支持：BNSD、BSND
  - 当 layout_q 为 BSND 时：
    - layout_kv 支持：BSND、PA_BBND、PA_BNBD、PA_Nz
    - layout_out 支持：BSND
  - 当 layout_q 为 TND 时：
    - layout_kv 支持：TND、PA_BBND、PA_BNBD、PA_Nz
    - layout_out 支持：TND
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

---

## 模块 8: EmptyTensorChecker（空Tensor场景处理）

### 任务 8.1: EmptyTensorChecker::CheckSinglePara
**指令**: 实现单参数约束验证：
- 无单参数约束
- 直接返回 GRAPH_SUCCESS

### 任务 8.2: EmptyTensorChecker::CheckParaExistence
**指令**: 实现存在性约束验证：
- 无存在性约束
- 直接返回 GRAPH_SUCCESS

### 任务 8.3: EmptyTensorChecker::CheckFeature
**指令**: 实现特性交叉约束验证：
- 实现空Tensor检测逻辑，判断是否属于空Tensor场景：
  - query、attentionOut 所有 tensor 的 shapeSize 为 0，属于空Tensor
  - query、attentionOut 所有 tensor 的 shapeSize 不为 0，若有 lsever 且 lsever 不为空，并且 key、value 中所有 tensor 的 shapeSize 为 0，属于空Tensor
  - attentionOut 和 lsever 都为空时，属于空Tensor
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

### 任务 8.4: EmptyTensorChecker::CheckMultiPara
**指令**: 实现一致性约束验证：
- 实现空Tensor场景下的返回值逻辑：
  - 若 attentionOut 为空，返回空
  - 否则返回全0
  - 若有 lsever 且 lsever 为空时返回空
  - lsever 不为空则返回全inf
- 返回 GRAPH_SUCCESS 或 GRAPH_FAILED

---

## 按实现优先级排序的任务清单

### 第一阶段：基础Checker框架（必须实现）
1. ✅ 任务 1.1: ShapeChecker::CheckSinglePara - 已完成
2. ✅ 任务 1.2: ShapeChecker::CheckParaExistence - 已完成
3. ✅ 任务 1.3: ShapeChecker::CheckFeature - 已完成
4. ✅ 任务 1.4: ShapeChecker::CheckMultiPara - 已完成

### 第二阶段：参数组Checker实现（核心功能）
5. 任务 2.1: MaskChecker::CheckSinglePara
6. 任务 2.2: MaskChecker::CheckParaExistence
7. 任务 2.3: MaskChecker::CheckFeature
8. 任务 2.4: MaskChecker::CheckMultiPara

9. 任务 3.1: ActualSeqLenChecker::CheckSinglePara
10. 任务 3.2: ActualSeqLenChecker::CheckParaExistence
11. 任务 3.3: ActualSeqLenChecker::CheckFeature
12. 任务 3.4: ActualSeqLenChecker::CheckMultiPara

13. 任务 4.1: PagedAttentionChecker::CheckSinglePara
14. 任务 4.2: PagedAttentionChecker::CheckParaExistence
15. 任务 4.3: PagedAttentionChecker::CheckFeature
16. 任务 4.4: PagedAttentionChecker::CheckMultiPara

17. 任务 5.1: LearnableSinkChecker::CheckSinglePara
18. 任务 5.2: LearnableSinkChecker::CheckParaExistence
19. 任务 5.3: LearnableSinkChecker::CheckFeature
20. 任务 5.4: LearnableSinkChecker::CheckMultiPara

21. 任务 6.1: SoftmaxLSEChecker::CheckSinglePara
22. 任务 6.2: SoftmaxLSEChecker::CheckParaExistence
23. 任务 6.3: SoftmaxLSEChecker::CheckFeature
24. 任务 6.4: SoftmaxLSEChecker::CheckMultiPara

### 第三阶段：高级Checker实现
25. 任务 7.1: LayoutChecker::CheckSinglePara
26. 任务 7.2: LayoutChecker::CheckParaExistence
27. 任务 7.3: LayoutChecker::CheckFeature
28. 任务 7.4: LayoutChecker::CheckMultiPara

29. 任务 8.1: EmptyTensorChecker::CheckSinglePara
30. 任务 8.2: EmptyTensorChecker::CheckParaExistence
31. 任务 8.3: EmptyTensorChecker::CheckFeature
32. 任务 8.4: EmptyTensorChecker::CheckMultiPara

---

## 使用说明

1. 每个Checker类需要继承自基类（如果有的话）
2. 每个Checker类需要实现四个方法：CheckSinglePara、CheckParaExistence、CheckFeature、CheckMultiPara
3. FAChecker::Process 方法会按顺序调用所有Checker的四个方法
4. 建议按照上述优先级顺序逐步实现和测试
5. 每个任务都可以直接作为实现指令提供给开发人员

## 实现进度总结

- ✅ ShapeChecker: 100% 完成 (4/4 任务)
- ⏳ MaskChecker: 0% 完成 (0/4 任务)
- ⏳ ActualSeqLenChecker: 0% 完成 (0/4 任务)
- ⏳ PagedAttentionChecker: 0% 完成 (0/4 任务)
- ⏳ LearnableSinkChecker: 0% 完成 (0/4 任务)
- ⏳ SoftmaxLSEChecker: 0% 完成 (0/4 任务)
- ⏳ LayoutChecker: 0% 完成 (0/4 任务)
- ⏳ EmptyTensorChecker: 0% 完成 (0/4 任务)

**总体进度**: 12.5% (4/32 任务)
