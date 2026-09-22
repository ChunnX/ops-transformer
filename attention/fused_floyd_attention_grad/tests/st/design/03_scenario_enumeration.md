# aclnnFusedFloydAttentionGrad 场景枚举

## 约束关系清单

| ID | 约束描述 | 涉及参数 | 关键词 | 需要枚举 | @solves target |
|----|---------|---------|--------|---------|---------------|
| C1 | B 轴取值范围 [1,2048] | _b.value | 取值范围 | 否 | _b.value |
| C2 | H 轴取值范围 [1,256] | _h.value | 取值范围 | 否 | _h.value（factor-domain） |
| C3 | N 轴取值范围 [16,1M] 且 %16==0 | _n.value | 取值范围 | 否 | _n.value |
| C4 | M 轴取值范围 [128,1M] 且 %128==0 | _m.value | 取值范围 | 否 | _m.value |
| C5 | K 轴取值范围 [128,1M] 且 %128==0 | _k.value | 取值范围 | 否 | _k.value |
| C6 | D 轴枚举 {32,64,128} | _d.value | 枚举 | 否 | _d.value（factor-domain） |
| C7 | query 与 key1 第 0/2/4 轴相同 | query.shape, key1.shape | 相同 | 否 | key1.shape |
| C8 | key1 与 value1 shape 相同 | key1.shape, value1.shape | 相同 | 否 | value1.shape |
| C9 | key2 与 value2 shape 相同 | key2.shape, value2.shape | 相同 | 否 | value2.shape |
| C10 | query 与 dy shape 相同 | query.shape, dy.shape | 相同 | 否 | dy.shape |
| C11 | query 与 attentionIn shape 相同 | query.shape, attentionIn.shape | 相同 | 否 | attentionIn.shape |
| C12 | softmaxMax 与 softmaxSum shape 相同 | softmaxMax.shape, softmaxSum.shape | 相同 | 否 | softmaxSum.shape |
| C13 | attenMask shape=[B,1,N,1,K] | attenMaskOptional.shape, attenMaskOptional.exist | 必须等于 | 否 | attenMaskOptional.shape |
| C14 | softmaxMax/Sum shape=[B,H,N,M,8] | softmaxMax.shape, softmaxSum.shape | 必须等于 | 否 | softmaxMax.shape |
| C15 | 五输入 dtype 一致 | query/key1/value1/key2/value2 .dtype | 一致 | 否 | key1/value1/key2/value2.dtype |
| C16 | attentionIn dtype 与 query 一致 | attentionIn.dtype, query.dtype | 一致 | 否 | attentionIn.dtype |
| C17 | dy dtype 与 query 一致 | dy.dtype, query.dtype | 一致 | 否 | dy.dtype |
| C18 | 输出 dtype 与 query 一致 | dqOut/dk1Out/dv1Out/dk2Out/dv2Out.dtype, query.dtype | 一致 | 否 | 各输出 .dtype |
| C19 | 输出 shape 与对应输入相同 | 各输出.shape, 对应输入.shape | 相同 | 否 | 各输出 .shape |
| C20 | attenMask 元素值域 {0,1} | attenMaskOptional.value_range_bool/uint8 | 元素取值范围 | 否 | factor-domain + validate |

**子场景枚举说明**：本算子全部约束（C1~C20）均为单场景约束（等值/范围/模板/元素级值域），按 §0.1 模式速查表均判定"不需要枚举"（关键词：一致/相同/等于/取值范围/元素取值范围，子场景数=1）。约束的正确性由 @solves 正向生成 + validate_constraints 反向校验双层保障；mask 取数场景（全参与/全掩蔽/混合）通过 `attenMaskOptional.value_range_bool/uint8` 的三段取值 [[0,0],[1,1],[0,1]] 在 L0 因子值覆盖中实现，无需独立子场景 ID。

## 空tensor维度场景

无 ET 场景。依据：本算子所有输入 tensor 的 shape 各轴均有正下界（R1~R6：B≥1、H≥1、N≥16、M≥128、K≥128、D∈{32,64,128}），shape 不可能含 0 轴，空 tensor 属于异常场景（host 代码 tiling_common.cpp "input shape error, got 0 in bhmnkd" 拒绝），已在 02_test_factors.yaml 对全部输入 tensor 设置 `support_empty_tensor: false`，对应异常场景见 EX-S1~EX-S5。

## 边界值场景

| 子场景 ID | locked因子 | 约束来源 | 测试重点 |
|----------|-----------|---------|---------|
| BD-S1 | _b.value_range=[1,1] | 01 R1 "B：取值范围为1~2K" | B 下界（B=1） |
| BD-S2 | _b.value_range=[2048,2048], _h.value=1, _d.value=32, _n.value_range=[16,16], _m.value_range=[128,128], _k.value_range=[128,128] | 01 R1 "B：取值范围为1~2K" | B 上界（B=2048，相关中间变量同步锁定最小值保证 2G 预算内可求解） |
| BD-S3 | _h.value=1 | 01 R2 "H：取值范围为1~256" | H 下界（H=1） |
| BD-S4 | _h.value=256, _b.value_range=[1,1], _d.value=32, _n.value_range=[16,16], _m.value_range=[128,128], _k.value_range=[128,128] | 01 R2 "H：取值范围为1~256" | H 上界（H=256，相关中间变量同步锁定） |
| BD-S5 | _n.value_range=[16,16] | 01 R3 "N：取值范围为16~1M且N%16==0" | N 下界（N=16，最小 16 对齐值） |
| BD-S6 | _m.value_range=[128,128] | 01 R4 "M：取值范围为128~1M且M%128==0" | M 下界（M=128，最小 128 对齐值） |
| BD-S7 | _k.value_range=[128,128] | 01 R5 "K：取值范围为128~1M且K%128==0" | K 下界（K=128，最小 128 对齐值） |
| BD-S8 | _d.value=32 | 01 R6 "D：取值范围为32/64/128" | D 枚举值 32 |
| BD-S9 | _d.value=64 | 01 R6 "D：取值范围为32/64/128" | D 枚举值 64 |
| BD-S10 | _d.value=128 | 01 R6 "D：取值范围为32/64/128" | D 枚举值 128 |
| BD-S11 | _d.value=128, _m.value_range=[512,512], _b.value_range=[1,1], _h.value=1, _n.value_range=[16,16], _k.value_range=[128,128] | 01 约束说明 "当M*D>=65536...会出现明显性能下降" | 性能语义边界：M*D=65536（降级侧，M=512×D=128） |
| BD-S12 | _d.value=128, _m.value_range=[256,256], _b.value_range=[1,1], _h.value=1, _n.value_range=[16,16], _k.value_range=[128,128] | 01 约束说明（同上） | 性能语义边界：M*D=32768（正常侧） |
| BD-S13 | _d.value=128, _k.value_range=[512,512], _b.value_range=[1,1], _h.value=1, _n.value_range=[16,16], _m.value_range=[128,128] | 01 约束说明 "当...K*D>=65536时，会出现明显性能下降" | 性能语义边界：K*D=65536（降级侧） |
| BD-S14 | _d.value=128, _k.value_range=[256,256], _b.value_range=[1,1], _h.value=1, _n.value_range=[16,16], _m.value_range=[128,128] | 01 约束说明（同上） | 性能语义边界：K*D=32768（正常侧） |

> N/M/K 上界（1M）说明：在 2G shape 元素预算（B*H*N*M*D ≤ 2G 等）下，N 上界可达值受其他轴下界约束（如 N ≤ 2G/(B*H*D*128)，B=H=1、D=32 时最大 524288 < 1M），文档上界 1M 的极端组合超出单 tensor 2G 元素限制，不可构造可执行用例；上界方向覆盖由 `_n/_m/_k.value_range` 的大值段（[65536,1048576]/[131072,1048576]）在预算内随机采样实现。

## 异常场景

> 预期错误码说明：接口文档错误码表中 shape/tiling 类约束违反对应 ACLNN_ERR_INNER_TILING_ERROR(561002)，框架统一映射为 ACL_ERROR_INVALID_DIMENSION；dtype/format 类对应 ACLNN_ERR_PARAM_INVALID(161002)，由引擎自动生成（unsupported_dtype/unsupported_format）。
> 锁定机制说明：EX 场景锁定 tensor 级 `.shape` 因子（L2 管道对 `_*` 前缀中间因子锁不传入引擎求解，违反值必须落在 tensor shape 上才可在 CSV 中可见）；派生 shape 锁定值经引擎 L2 fallback 机制保留（与引擎自动生成的 output_shape_mismatch 用例同机制），生成日志出现 `[WARN] Scenario locks derived factor` 信息性提示属预期行为。

| 子场景 ID | locked因子 | 预期错误码 | 测试重点 |
|----------|-----------|-----------|---------|
| EX-S1 | query.shape=[0,1,16,128,32], dy.shape=[0,1,16,128,32], attentionIn.shape=[0,1,16,128,32], dqOut.shape=[0,1,16,128,32], key1.shape=[0,1,16,128,32], value1.shape=[0,1,16,128,32], dk1Out.shape=[0,1,16,128,32], dv1Out.shape=[0,1,16,128,32], key2.shape=[0,1,128,128,32], value2.shape=[0,1,128,128,32], dk2Out.shape=[0,1,128,128,32], dv2Out.shape=[0,1,128,128,32], softmaxMax.shape=[0,1,16,128,8], softmaxSum.shape=[0,1,16,128,8], attenMaskOptional.exist=True, attenMaskOptional.shape=[0,1,16,1,128] | ACL_ERROR_INVALID_DIMENSION | B=0 空轴（R1 违反，tiling "got 0 in bhmnkd"，561002） |
| EX-S2 | query.shape=[1,0,16,128,32], dy.shape=[1,0,16,128,32], attentionIn.shape=[1,0,16,128,32], dqOut.shape=[1,0,16,128,32], key1.shape=[1,0,16,128,32], value1.shape=[1,0,16,128,32], dk1Out.shape=[1,0,16,128,32], dv1Out.shape=[1,0,16,128,32], key2.shape=[1,0,128,128,32], value2.shape=[1,0,128,128,32], dk2Out.shape=[1,0,128,128,32], dv2Out.shape=[1,0,128,128,32], softmaxMax.shape=[1,0,16,128,8], softmaxSum.shape=[1,0,16,128,8], attenMaskOptional.exist=True, attenMaskOptional.shape=[1,1,16,1,128] | ACL_ERROR_INVALID_DIMENSION | H=0 空轴（R2 违反，561002；attenMask 第 1 轴固定为 1 不受 H 影响） |
| EX-S3 | query.shape=[1,1,0,128,32], dy.shape=[1,1,0,128,32], attentionIn.shape=[1,1,0,128,32], dqOut.shape=[1,1,0,128,32], key1.shape=[1,1,0,128,32], value1.shape=[1,1,0,128,32], dk1Out.shape=[1,1,0,128,32], dv1Out.shape=[1,1,0,128,32], key2.shape=[1,1,128,128,32], value2.shape=[1,1,128,128,32], dk2Out.shape=[1,1,128,128,32], dv2Out.shape=[1,1,128,128,32], softmaxMax.shape=[1,1,0,128,8], softmaxSum.shape=[1,1,0,128,8], attenMaskOptional.exist=True, attenMaskOptional.shape=[1,1,0,1,128] | ACL_ERROR_INVALID_DIMENSION | N=0 空轴（R3 违反，561002；key2 族不含 N 轴保持正常） |
| EX-S4 | query.shape=[1,1,16,0,32], dy.shape=[1,1,16,0,32], attentionIn.shape=[1,1,16,0,32], dqOut.shape=[1,1,16,0,32], key1.shape=[1,1,16,128,32], value1.shape=[1,1,16,128,32], dk1Out.shape=[1,1,16,128,32], dv1Out.shape=[1,1,16,128,32], key2.shape=[1,1,128,0,32], value2.shape=[1,1,128,0,32], dk2Out.shape=[1,1,128,0,32], dv2Out.shape=[1,1,128,0,32], softmaxMax.shape=[1,1,16,0,8], softmaxSum.shape=[1,1,16,0,8], attenMaskOptional.exist=True, attenMaskOptional.shape=[1,1,16,1,128] | ACL_ERROR_INVALID_DIMENSION | M=0 空轴（R4 违反，561002；key1 族与 attenMask 不含 M 轴保持正常） |
| EX-S5 | query.shape=[1,1,16,128,32], dy.shape=[1,1,16,128,32], attentionIn.shape=[1,1,16,128,32], dqOut.shape=[1,1,16,128,32], key1.shape=[1,1,16,0,32], value1.shape=[1,1,16,0,32], dk1Out.shape=[1,1,16,0,32], dv1Out.shape=[1,1,16,0,32], key2.shape=[1,1,0,128,32], value2.shape=[1,1,0,128,32], dk2Out.shape=[1,1,0,128,32], dv2Out.shape=[1,1,0,128,32], softmaxMax.shape=[1,1,16,128,8], softmaxSum.shape=[1,1,16,128,8], attenMaskOptional.exist=True, attenMaskOptional.shape=[1,1,16,1,0] | ACL_ERROR_INVALID_DIMENSION | K=0 空轴（R5 违反，561002；query 族与 softmax 族不含 K 轴保持正常） |
| EX-S6 | query.shape=[1,1,17,128,32], dy.shape=[1,1,17,128,32], attentionIn.shape=[1,1,17,128,32], dqOut.shape=[1,1,17,128,32], key1.shape=[1,1,17,128,32], value1.shape=[1,1,17,128,32], dk1Out.shape=[1,1,17,128,32], dv1Out.shape=[1,1,17,128,32], key2.shape=[1,1,128,128,32], value2.shape=[1,1,128,128,32], dk2Out.shape=[1,1,128,128,32], dv2Out.shape=[1,1,128,128,32], softmaxMax.shape=[1,1,17,128,8], softmaxSum.shape=[1,1,17,128,8], attenMaskOptional.exist=True, attenMaskOptional.shape=[1,1,17,1,128] | ACL_ERROR_INVALID_DIMENSION | N 非 16 对齐（N=17，R3 对齐违反，tiling "16-byte alignment"，561002） |
| EX-S7 | query.shape=[1,1,16,129,32], dy.shape=[1,1,16,129,32], attentionIn.shape=[1,1,16,129,32], dqOut.shape=[1,1,16,129,32], key1.shape=[1,1,16,128,32], value1.shape=[1,1,16,128,32], dk1Out.shape=[1,1,16,128,32], dv1Out.shape=[1,1,16,128,32], key2.shape=[1,1,128,129,32], value2.shape=[1,1,128,129,32], dk2Out.shape=[1,1,128,129,32], dv2Out.shape=[1,1,128,129,32], softmaxMax.shape=[1,1,16,129,8], softmaxSum.shape=[1,1,16,129,8], attenMaskOptional.exist=True, attenMaskOptional.shape=[1,1,16,1,128] | ACL_ERROR_INVALID_DIMENSION | M 非 128 对齐（M=129，R4 对齐违反，561002） |
| EX-S8 | query.shape=[1,1,16,128,32], dy.shape=[1,1,16,128,32], attentionIn.shape=[1,1,16,128,32], dqOut.shape=[1,1,16,128,32], key1.shape=[1,1,16,100,32], value1.shape=[1,1,16,100,32], dk1Out.shape=[1,1,16,100,32], dv1Out.shape=[1,1,16,100,32], key2.shape=[1,1,100,128,32], value2.shape=[1,1,100,128,32], dk2Out.shape=[1,1,100,128,32], dv2Out.shape=[1,1,100,128,32], softmaxMax.shape=[1,1,16,128,8], softmaxSum.shape=[1,1,16,128,8], attenMaskOptional.exist=True, attenMaskOptional.shape=[1,1,16,1,100] | ACL_ERROR_INVALID_DIMENSION | K 低于下界且非对齐（K=100，R5 违反，561002） |
| EX-S9 | query.shape=[1,1,16,128,48], dy.shape=[1,1,16,128,48], attentionIn.shape=[1,1,16,128,48], dqOut.shape=[1,1,16,128,48], key1.shape=[1,1,16,128,48], value1.shape=[1,1,16,128,48], dk1Out.shape=[1,1,16,128,48], dv1Out.shape=[1,1,16,128,48], key2.shape=[1,1,128,128,48], value2.shape=[1,1,128,128,48], dk2Out.shape=[1,1,128,128,48], dv2Out.shape=[1,1,128,128,48], softmaxMax.shape=[1,1,16,128,8], softmaxSum.shape=[1,1,16,128,8], attenMaskOptional.exist=True, attenMaskOptional.shape=[1,1,16,1,128] | ACL_ERROR_INVALID_DIMENSION | D 非法值（D=48，R6 违反，tiling "Headdim only supports 32/64/128"，561002；softmax 尾轴固定 8 不含 D） |
| EX-S10 | query.shape=[2049,1,16,128,32], dy.shape=[2049,1,16,128,32], attentionIn.shape=[2049,1,16,128,32], dqOut.shape=[2049,1,16,128,32], key1.shape=[2049,1,16,128,32], value1.shape=[2049,1,16,128,32], dk1Out.shape=[2049,1,16,128,32], dv1Out.shape=[2049,1,16,128,32], key2.shape=[2049,1,128,128,32], value2.shape=[2049,1,128,128,32], dk2Out.shape=[2049,1,128,128,32], dv2Out.shape=[2049,1,128,128,32], softmaxMax.shape=[2049,1,16,128,8], softmaxSum.shape=[2049,1,16,128,8], attenMaskOptional.exist=True, attenMaskOptional.shape=[2049,1,16,1,128] | ACL_ERROR_INVALID_DIMENSION | B 超上界（B=2049，R1 违反，561002） |
| EX-S11 | query.shape=[1,257,16,128,32], dy.shape=[1,257,16,128,32], attentionIn.shape=[1,257,16,128,32], dqOut.shape=[1,257,16,128,32], key1.shape=[1,257,16,128,32], value1.shape=[1,257,16,128,32], dk1Out.shape=[1,257,16,128,32], dv1Out.shape=[1,257,16,128,32], key2.shape=[1,257,128,128,32], value2.shape=[1,257,128,128,32], dk2Out.shape=[1,257,128,128,32], dv2Out.shape=[1,257,128,128,32], softmaxMax.shape=[1,257,16,128,8], softmaxSum.shape=[1,257,16,128,8], attenMaskOptional.exist=True, attenMaskOptional.shape=[1,1,16,1,128] | ACL_ERROR_INVALID_DIMENSION | H 超上界（H=257，R2 违反，561002；attenMask 第 1 轴固定为 1 不受 H 影响） |
| EX-S12 | query.shape=[1,1,16,128,32], softmaxMax.shape=[1,1,16,128,16] | ACL_ERROR_INVALID_DIMENSION | softmaxMax 尾轴≠8（期望 [1,1,16,128,8]，实际尾轴 16，R14 违反，561002） |
| EX-S13 | key1.shape=[1,1,16,128,32], value1.shape=[1,1,16,256,32] | ACL_ERROR_INVALID_DIMENSION | key1 与 value1 shape 不同（K 轴 128 vs 256，R8 违反，tiling CheckSameShape，561002） |
| EX-S14 | attenMaskOptional.exist=True, query.shape=[1,1,16,128,32], key1.shape=[1,1,16,128,32], attenMaskOptional.shape=[1,2,16,1,128] | ACL_ERROR_INVALID_DIMENSION | attenMask shape 模板违反（期望 [1,1,16,1,128]，实际第 1 轴=2，R13 违反，561002） |
| EX-S15 | query.shape=[1,1,16,128,32], dy.shape=[1,1,16,256,32] | ACL_ERROR_INVALID_DIMENSION | dy 与 query shape 不同（M 轴 128 vs 256，R10 违反，561002） |
| EX-S16 | query.shape=[1,1,16,128,32], key1.shape=[1,1,32,128,32] | ACL_ERROR_INVALID_DIMENSION | key1 与 query 第 2 轴不同（N 轴 16 vs 32，R7 违反，561002） |

> 补充说明：
> 1. dtype 域外异常（不支持的 dtype/format）、输入间 dtype 不匹配（R15/R17 等值关系违反）、输出 shape 不匹配、维度边界异常、空 tensor（shape 含 0 轴，各输入 tensor 逐一生成）由引擎 L2 自动生成（_generate_out_of_domain_cases / _generate_constraint_violation_cases / _generate_dimension_boundary_exceptions / _generate_empty_tensor_exceptions），不在此重复定义。
> 2. N/M/K 超上界（>1M）异常不可构造：任何 N>524288（或 M/K 超预算上界）的组合超出单 tensor 2G 元素限制，测试框架无法分配输入内存，故不设该类 EX 场景。
> 3. EX 场景锁定完整一致的 shape 组（16 个 tensor shape + exist），确保违反值在 CSV 的 tensor_view_shapes 列可见且单一违反意图明确；其余因子（dtype/scaleValue/mask 值域等）由引擎 fallback 填充。

## 场景覆盖验证

（步骤 4 阶段 A/B 执行后回填）

| 子场景 ID | 是否出现 | 出现次数 | 备注 |
|----------|---------|---------|------|
| BD-S1 | 是 | 1 | L0 场景覆盖（4B 日志 [SCENARIO-COVERAGE] + Q14_bd_case_coverage 确认） |
| BD-S2 | 是 | 1 | L0 场景覆盖 |
| BD-S3 | 是 | 1 | L0 场景覆盖 |
| BD-S4 | 是 | 1 | L0 场景覆盖 |
| BD-S5 | 是 | 1 | L0 场景覆盖 |
| BD-S6 | 是 | 1 | L0 场景覆盖 |
| BD-S7 | 是 | 1 | L0 场景覆盖 |
| BD-S8 | 是 | 1 | L0 场景覆盖 |
| BD-S9 | 是 | 1 | L0 场景覆盖 |
| BD-S10 | 是 | 1 | L0 场景覆盖 |
| BD-S11 | 是 | 1 | L0 场景覆盖（性能边界 M*D=65536） |
| BD-S12 | 是 | 1 | L0 场景覆盖（性能边界 M*D=32768） |
| BD-S13 | 是 | 1 | L0 场景覆盖（性能边界 K*D=65536） |
| BD-S14 | 是 | 1 | L0 场景覆盖（性能边界 K*D=32768） |
| EX-S1~EX-S16 | 是 | 各 1 | L2 CSV（aclnnFusedFloydAttentionGrad_l2_exception.csv）中逐条出现 |

> 覆盖结论：BD-S1~BD-S14 全部出现在 L0（引擎 [SCENARIO-COVERAGE] 输出 + qa_verify_oracle Q14_bd_case_coverage 14 条记录确认）；EX-S1~EX-S16 全部出现在 L2（CSV remark 列 _scenario_id 逐条确认）。ET 场景无（空 tensor 不支持，见"空tensor维度场景"章节说明）。
