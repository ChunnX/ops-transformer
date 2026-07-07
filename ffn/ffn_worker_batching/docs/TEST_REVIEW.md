# FfnWorkerBatching 测试设计评审报告（新增 Ascend950/A5 支持）

**状态**: ✅通过

**评审对象**: FfnWorkerBatching A5(Ascend950/arch35) 测试设计（TEST.md + L0/L1/L2 用例 + coverage_report）
**真值源**: `docs/spec.yaml`（L0 数学契约，SPEC_REVIEW ✅通过）；`docs/REQUIREMENTS.md`（需求背景）
**评审场景**: scene: test-design-review（CP2 前置，不触达用户）
**评审方法**: 仅评审 Markdown 测试设计文档与测试用例表（CSV/YAML），未评审代码文件
**评审时间**: 2026-07-06

---

**条款总数**: 7 | 通过: 7 | 发现问题(HIGH): 0 | 需关注(MED): 2 | 提示(LOW): 2

---

## spec.yaml 测试映射核对

| spec 字段 | TEST.md 承接位置 | 状态 |
|-----------|-----------------|------|
| dtype_policy.supported_combinations（P0 fp16 / P1 bf16 / P2 int8） | §2 映射表、§5.1 dtype 矩阵 | ✓ |
| outputs[].shape_rule（Y=A×BS×(topK+1)、[expert_num,2]、[Y]、[1]） | §2 映射表、§5.2 shape 关系 | ✓ |
| boundary_conditions[]（spec 实为 11 条） | §2 映射表、§4.1 边界覆盖表（14 行） | ✓（覆盖全，计数标签误写，见 TEST-SPEC-3 / 问题#1） |
| extreme_inputs[]（spec 实为 3 条） | §2 映射表、§4.2 极端输入表（3 行） | ✓（覆盖全，计数标签误写，见 问题#2） |
| numerical_tolerance.per_dtype（全 dtype bitwise_equal, rtol=0/atol=0） | §5.3 精度验收标准 | ✓ |
| math_semantics.reference_oracle（absent=true, framework=numpy） | §6 oracle 与 golden 来源（显式声明替代 golden） | ✓ |
| determinism（required=true, sequential, bitwise_reproducible） | §7 确定性测试 | ✓ |

> 结论：TEST.md「§2 spec.yaml 测试映射」章节齐备（TEST-SPEC-1 通过）。最易误用 5 字段
> （dtype_policy.supported_combinations / outputs[].shape_rule / numerical_tolerance.per_dtype /
> boundary_conditions[] / extreme_inputs[]）均直接取自 spec.yaml，未从 REQUIREMENTS 重新解释。

---

## 用例覆盖核对

### dtype 组合（spec.dtype_policy.supported_combinations，3 profile）

| spec 项 | 期望覆盖 | 实际用例 | 覆盖状态 |
|---------|---------|---------|---------|
| P0: schedule_context=int8, token_payload=fp16 → y=fp16 | L0 + L1 | L0_001/004/005 + L1_PRF_P0_*（NORM/RECV）+ 162 分布 | ✓ |
| P1: schedule_context=int8, token_payload=bf16 → y=bf16 | L0 + L1 | L0_002 + L1_PRF_P1_*（NORM/RECV）+ 175 分布 | ✓ |
| P2: schedule_context=int8, token_payload=int8 → y=int8 + dynamic_scale 有效 | L0 + L1 | L0_003 + L1_PRF_P2_*（NORM/RECV）+ L1_EXT_dynamic_quant_int8_scale + 178 分布 | ✓ |
| 功能路径 6 组合（token_dtype × need_schedule） | L1 全覆盖 | 6 条 PRF 矩阵用例，coverage_report 6 桶均 ≥27 | ✓ |

### boundary_conditions（spec 11 条）

| spec 项 | 期望级别 | 承接用例 | 覆盖状态 |
|---------|---------|---------|---------|
| k=0（全 mask，actual_token_num=0） | L1 | L1_BND_k_zero_all_masked | ✓ |
| k=N（A=1024 上界 + topK+1=64 上界） | L1 | L1_BND_k_eq_n_A1024_topk64（max_out_shape=[1024,4,64,128]） | ✓ |
| same_bin + expert_num=8192 上界 | L1 | L1_BND_same_bin_expert8192（expert_num=8192） | ✓ |
| expert_num>8192 → 报错 | L2 | L2_ERR_attr_expert_num_over_8193 | ✓ |
| topK+1>64 → 报错 | L2 | L2_ERR_attr_topk_over_65（max_out_shape=[8,4,65,128]） | ✓ |
| A>1024 → 报错 | L2 | L2_ERR_attr_A_over_1025（max_out_shape=[1025,4,8,128]） | ✓ |
| max_out_shape 长度≠4 → 报错 | L2 | L2_ERR_attr_max_out_shape_len3（[8,4,8]） | ✓ |
| schedule_context dtype≠INT8 → 报错 | L2 | L2_ERR_dtype_not_supported_fp16 / _int32 | ✓ |
| schedule_context 非 1 维 → 报错 | L2 | L2_ERR_shape_mismatch_2d_2x512 / _2d_1024x1 | ✓ |
| need_schedule=0 NORM（TilingKey 100） | L1 | L1_PRF_P0_fp16_NORM（+ 341 分布） | ✓ |
| need_schedule=1 RECV（TilingKey 101） | L1 | L1_PRF_P0_fp16_RECV（+ 174 分布） | ✓ |
| (额外) expert_num=1 下界 | L1 | L1_BND_expert_num_lower_1 | ✓（增强） |
| (额外) topK+1=64 / A=1024 内部上界 | L1 | L1_BND_topk64_upper / L1_BND_A1024_upper | ✓（增强） |

> spec 11 条 boundary_conditions 全部被 L1（非报错项）+ L2（报错项）覆盖，另含 3 条增强边界用例。

### extreme_inputs（spec 3 条）

| spec 项 | 期望级别 | 承接用例 | 覆盖状态 |
|---------|---------|---------|---------|
| all_same_index（全同专家，极端访存） | L1 | L1_EXT_all_same_index（expert_num=8192） | ✓ |
| token_dtype=2 连续 scale（尾部 32B FP32） | L1 | L1_EXT_dynamic_quant_int8_scale（token_dtype=2） | ✓ |
| 稀疏 mask（≈50% mask，被 mask 排末尾） | L1 | L1_EXT_sparse_50pct_mask_recv（need_schedule=1） | ✓ |

### L2 错误码（⊆ spec.error_codes）

| 错误码（spec） | 用例数 | 用例 | 覆盖状态 |
|---------|-------|------|---------|
| null_input | 1 | L2_ERR_null_input_schedule_context | ✓ |
| dtype_not_supported | 2 | L2_ERR_dtype_not_supported_fp16 / _int32 | ✓ |
| shape_mismatch | 2 | L2_ERR_shape_mismatch_2d_2x512 / _2d_1024x1 | ✓ |
| attribute_value_out_of_range | 8 | expert_num_over_8193 / expert_num_zero / topk_over_65 / A_over_1025 / max_out_shape_len3 / token_dtype_over_3 / need_schedule_over_2 / layer_num_gt_expert | ✓ |

> L2 共 13 条，全部映射到 spec.error_codes 四类，**无非 spec 错误码混入**。

---

## 黑盒机器证据核对（workflow validator 前置）

| 检查项 | 期望 | 实测 | 状态 |
|--------|------|------|------|
| blackbox_case_targets marker（TEST.md 文末） | L0:5 / L1:515 / L2:13 | 存在，值一致 | ✓ |
| CSV 数据行数（不含表头） | L0=5 / L1=515 / L2=13 | 6/516/14 总行 → 5/515/13 数据行 | ✓ |
| testcase_name 唯一（三 CSV 合并） | 全唯一 | 533 条 0 重复 | ✓ |
| level 可推断（命名含 _L0_/_L1_/_L2_） | 100% | 533/533 可推断 | ✓ |
| 每行 tensor 列结构一致 | 9 shape / 9 dtype / 8 output_index | L1 全 515 行校验 0 异常 | ✓ |

---

## 逐条款结论

| 条款 ID | 结论 | 证据 | spec.yaml 依据 |
|---------|------|------|---------------|
| TEST-SPEC-1（spec 一致性） | ✅通过 | TEST.md §2 映射表 7 行齐备，dtype/shape/boundary/extreme/tolerance/oracle/determinism 全承接 | 全 spec-owned 字段 |
| TEST-SPEC-2（dtype 覆盖） | ✅通过 | L0 覆盖 3 profile；L1 6 条 PRF 矩阵 + 分布 162/175/178；6 功能路径桶均 ≥27 | dtype_policy.supported_combinations |
| TEST-SPEC-3（边界/极端覆盖） | ✅通过 | 11 boundary + 3 extreme 逐条有承接用例（+3 增强边界）；A/topK+1/expert_num 上下界、空-mask、被 mask 排末尾、3 profile、need_schedule 0/1 全覆盖 | boundary_conditions[] / extreme_inputs[] |
| TEST-SPEC-4（精度判据） | ✅通过 | §5.3 全 dtype rtol=0/atol=0/bitwise_equal，逐项等于 spec per_dtype；6 dtype 全一致 | numerical_tolerance.per_dtype |
| TEST-SPEC-5（oracle 一致性） | ✅通过 | spec reference_oracle.absent=true；TEST.md §6 显式声明 CPU numpy 三步参考 + np.array_equal 替代 golden，已记录 | math_semantics.reference_oracle |
| TEST-COV-1（用例分级） | ✅通过 | L0 门槛（单因子 100%，核心 fp16 + expert_num 5 档）；L1 功能（pairwise+boundary+extreme+profile）；L2 异常（4 错误码）分级合理 | — |
| TEST-REQ-1（需求承接） | ✅通过 | REQUIREMENTS §4.4/§4.5/§4.6/§8.3/§9 验收口径（对齐 A2、二进制一致、确定性、6 功能路径、shape 上限）均可追溯到用例 | REQUIREMENTS 需求追溯表 |

---

## 问题清单

| # | 条款 ID | 严重度 | 证据（TEST.md 位置） | spec.yaml 依据 | 修复建议 |
|---|---------|--------|--------------------|---------------|----------|
| 1 | TEST-SPEC-3 | MED | 行 80「boundary_conditions[] 共 10 条 case（spec）」 | spec boundary_conditions 实为 **11 条**（need_schedule=0 与 =1 为两条独立 case，L162-289） | 计数标签订正为 11 条。**仅标签误写，覆盖已完整**（L1_PRF_P0_fp16_NORM/RECV 两路均在），不构成覆盖缺口 |
| 2 | TEST-SPEC-1 | MED | 行 34「extreme_inputs[]（4 条：all_same_index / token_dtype=2 连续 scale / 稀疏 mask）」 | spec extreme_inputs 实为 **3 条**（L291-305），该行括号内亦仅列 3 项 | 计数标签订正为 3 条（与括号列举项、§4.2 表 3 行一致）。**仅标签误写，覆盖已完整** |
| 3 | TEST-SPEC-4 | LOW | 三 CSV 的 `precision_tolerances` / `absolute_precision` 列全空 | numerical_tolerance.per_dtype 全 bitwise_equal | 交接测试工程开发（场景二）：须确认下游框架将空容差列解释为「精确相等 np.array_equal」，而非套用默认 rtol，以贯彻二进制一致口径 |
| 4 | TEST-COV-1 | LOW | L2_ERR_null_input_schedule_context 行结构与正常用例同形 | error_codes: null_input | 交接测试工程开发：null 语义需由测试框架按命名约定在开发期注入（置空 aclTensor 指针），设计层用例名已正确标注，非设计缺陷 |

---

## 冲突检查（REQUIREMENTS ↔ spec.yaml spec-owned 字段）

逐项核对属性 machine_constraint：

| 字段 | spec.yaml | REQUIREMENTS | 结论 |
|------|-----------|--------------|------|
| expert_num | [1, 8192] | (0, 8192] | ✓ 等价（整数域 1..8192） |
| max_out_shape.length | 4 | 长度必须为 4 | ✓ 一致 |
| token_dtype | [0, 2] | [0,2] | ✓ 一致 |
| need_schedule | [0, 1] | [0,1] | ✓ 一致 |
| layer_num | machine [0, 8192]，semantics [0, expert_num] | [0, expert_num] | ✓ 一致；语义关系 layer_num≤expert_num 由 TEST.md §10 处理（取值域收敛 {0,1}，越界由 L2_ERR_attr_layer_num_gt_expert 覆盖） |

**结论：REQUIREMENTS 与 spec.yaml 在 spec-owned 字段上无冲突。**

---

## 评审结论

**状态**: ✅通过

FfnWorkerBatching A5 测试设计满足 CP2 前置要求：

1. TEST.md「spec.yaml 测试映射」章节齐备，最易误用 5 字段直接取自 spec.yaml（TEST-SPEC-1/-2/-3/-4/-5 全通过）。
2. 精度口径全 dtype bitwise_equal（rtol=atol=0, np.array_equal），与 spec.numerical_tolerance.per_dtype 逐项一致，符合非计算类算子标准。
3. spec 11 条 boundary_conditions + 3 条 extreme_inputs 全被 L1/L2 用例覆盖；L2 13 条异常用例全 ⊆ spec.error_codes 四类，无非 spec 错误码混入。
4. 黑盒机器证据（blackbox_case_targets marker、CSV 行数 5/515/13、testcase_name 533 条 0 重复、level 全可推断、tensor 列结构一致）满足 workflow validator 校验。
5. REQUIREMENTS 与 spec.yaml 无 spec-owned 字段冲突。

**遗留项（不阻断 CP2，建议场景一 test-design 顺手订正）**：2 处 MED 计数标签误写（boundary 应 11 非 10、extreme 应 3 非 4，均为标签笔误、覆盖已完整）；2 处 LOW 交接提示（空容差列语义、null_input 开发期注入）转测试工程开发阶段处理。
