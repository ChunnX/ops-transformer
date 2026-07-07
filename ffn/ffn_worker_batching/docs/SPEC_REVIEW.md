# FfnWorkerBatching — spec 自审报告（1.2.5R）

**状态**: ✅通过

**spec.yaml 路径**: ffn/ffn_worker_batching/docs/spec.yaml
**REQUIREMENTS.md 路径**: ffn/ffn_worker_batching/docs/REQUIREMENTS.md

**评审范围**: 新增 Ascend950(arch35/A5) 支持，A5 对外支持范围对齐 A2（非计算类算子：sort→多核 gather→单核 group_listing）
**9-stage 校验**: overall PASS（stage 1-7 ✓ PASS；stage 8 SKIP=formula_kind=textual_only；stage 9 SKIP=reference_oracle.absent；--strict 零告警）

## 13 条 SPEC-* 条款评审

| 条款 ID | 状态 | spec 字段值 | REQUIREMENTS 来源 | 证据 / 备注 |
|---------|------|-------------|------------------|------------|
| SPEC-CHIP-1 | ✓ | `supported_chips: [Ascend910B, Ascend950PR, Ascend950DT]` | §2 运行环境：A2=Ascend910B / A3=Ascend910_93 / A5=Ascend950(DT/PR) | 子集关系成立：spec 三芯片全部 ⊆ REQUIREMENTS 目标芯片集。A3(Ascend910_93) 本次范围外（保留不动，已知建模决策 #2），未纳入不破坏 ⊆ 判定；实测 Ascend950DT_9584 被 Ascend950DT 覆盖。 |
| SPEC-DAV-1 | ⚠ | v1 暂缓 | §2 编译宏 DAV_2201/DAV_3510 | `dav_macros` 尚未纳入 schema，由 REQUIREMENTS §2 表 + §9 承载；非阻塞。 |
| SPEC-DTYPE-1 | ✓ | 3 profile：token_payload ∈ {float16, bfloat16, int8}，schedule_context 恒 int8 | §4.4 dtype 矩阵 P0(fp16)/P1(bf16)/P2(int8) | 集合相等：spec `supported_combinations` 三组 = REQUIREMENTS 三 profile；y dtype 随 token_dtype (0→fp16/1→bf16/2→int8)，与 §4.4 一致；A5 全量对齐 A2，无新增/删减（不引入 arch35 独有 FP8/MXFP8/HiF8）。 |
| SPEC-DTYPE-2 | ✓ | `inputs[0].schedule_context.dtype_set: [int8]` | §4.2 输入表：schedule_context INT8 | 物理输入唯一 schedule_context=INT8，spec 覆盖 REQUIREMENTS §4.2；token_payload 为建模键非独立 IR 输入（已知建模决策 #3），在 dtype_policy 表达 y dtype 随属性变化，不计入 inputs。 |
| SPEC-IO-1 | ✓ | inputs=1 (schedule_context)；outputs=8 (y, group_list, session_ids, micro_batch_ids, token_ids, expert_offsets, dynamic_scale, actual_token_num) | §5.1/§5.2 ACLNN 参数列表 | 输入名字集合对齐：ACLNN 唯一 aclTensor 输入 scheduleContext（其余 expertNum/maxOutShape/tokenDtype/needSchedule/layerNum 为属性）。输出 8 个名字一一对齐（camelCase↔snake_case 同名）：y/groupList/sessionIds/microBatchIds/tokenIds/expertOffsets/dynamicScale/actualTokenNum。数量与集合均相等。 |
| SPEC-ARG-1 | ⚠ | v1 暂缓 | §5.1 参数顺序 | `interface_binding.arg_order` 尚未纳入 schema；ACLNN INPUT→ATTR→OUTPUT 顺序由 REQUIREMENTS §5.1 承载；非阻塞。 |
| SPEC-ERROR-1 | ✓ | `error_codes: [null_input, shape_mismatch, dtype_not_supported, attribute_value_out_of_range]` | §4.2/§4.3/§8.1 硬校验：dtype 必 INT8、shape 1 维、属性取值范围 | 包含关系成立：REQUIREMENTS 隐含错误码全被覆盖——dtype≠INT8→dtype_not_supported、非 1 维→shape_mismatch、expert_num/max_out_shape/topK+1/A 越界→attribute_value_out_of_range；null_input 为标准入参校验。boundary_conditions 6 条 raises_error 用例均引用上述码，闭环。 |
| SPEC-PERF-1 | ⚠ | v1 暂缓 | §7 性能要求 | `performance_baseline` 尚未纳入 schema；且 §7 明确「本次无量化性能指标」，性能达标为可选验收；非阻塞。 |
| SPEC-RES-1 | ⚠ | v1 暂缓 | §8.2 资源约束 | `performance_budget` 尚未纳入 schema；workspace 公式/核数/UB 动态取值由 REQUIREMENTS §8.2 承载；非阻塞。 |
| SPEC-FORMULA-1 | ⚠ | `formula_kind: textual_only`；formula 为三步文字语义（sort→gather→group_listing），引用 token_data/expert_id/gather_idx 等 FfnArea 解析字段 | §4.1 三步算法语义 | 唯一物理输入 `schedule_context` 未以字面串出现在 formula（formula 引用的是其内 FfnArea 逐字节解析出的 token_data/expert_id 等派生量）。属已知建模决策 #1 的合法降级：单 1024B INT8 结构体入参无法在 numpy 沙箱忠实重建，语义由三步描述 + structural invariants + boundary/extreme 承载，stage 8 SKIP 印证。非阻塞（⚠），不构成 ❌。 |
| SPEC-PARADIGM-1 | ✓ | `category: index_gather`；`paradigms: [IndexGather, SortSelect, Histogram]` | §4.1 融合类：Sort 排序 + IndexGather 索引 gather + Scan/分组计数 | 集合差为空：category(index_gather) 隐含 IndexGather；REQUIREMENTS 修饰范式 Sort→SortSelect、分组计数→Histogram 全部被 spec.paradigms 覆盖；boundary/extreme 用例按 SortSelect/Histogram/IndexGather 三族最低集组织，佐证一致。 |
| SPEC-LIFECYCLE-1 | ✓ | `lifecycle: experimental` | §6 proto 标注「THIS FUNCTION IS EXPERIMENTAL」，A5 沿用 | 字符串匹配：spec=experimental = REQUIREMENTS §6 生命周期 experimental。 |
| SPEC-INTERFACE-1 | ⚠ | v1 暂缓 | §5/§6 ACLNN + GE IR 绑定 | `interface_binding.*` 尚未纳入 schema；两段式 aclnn + GE IR (REG_OP) 由 REQUIREMENTS §5/§6 承载；非阻塞。 |

## 附加机器可判项交叉核对（证据补强，非独立条款）

| 维度 | spec 字段值 | REQUIREMENTS 来源 | 一致性 |
|------|-------------|------------------|--------|
| 输出 shape 关系式 | `y=(A*BS*(topK+1), H)`、`group_list=(expert_num,2)`、id 类=(Y,)、`actual_token_num=(1,)` | §4.2/§4.6：Y=A×BS×(topK+1)、H=max_out_shape[3] | ✓ 一一对应，静态推导，A5 与代次无关全对齐 |
| shape 约束上限 | A∈(0,1024]、topK+1∈(0,64]、expert_num∈(0,8192]、M≤64 | §4.6：MAX_SESSION_NUM=1024/MAX_K_NUM=64/EXPERT_IDX_MAX=8192/micro_batch≤64 | ✓ 阈值全对齐 |
| 属性表（5 个） | expert_num/max_out_shape/token_dtype/need_schedule/layer_num + machine_constraint | §4.3 属性表 + 取值范围 | ✓ 名称/类型/默认值/范围一致 |
| 精度容差 | per_dtype 全部 `bitwise_equal (rtol=0, atol=0)`（fp16/bf16/int8/int32/int64/fp32） | §4.4-精度：非计算类二进制一致（np.array_equal） | ✓ 全 dtype 二进制一致，A5 与 A2 同口径 |
| 功能路径 | boundary need_schedule=0/1 两路（TilingKey 100/101） | §4.5 6 条组合（need_schedule×token_dtype） | ✓ 路径覆盖对齐（TilingKey 由 need_schedule 决定） |
| 确定性 | `determinism.required: true`、bitwise_reproducible、sequential | §8.3 默认支持、稳定序、整数精确累加 | ✓ 一致，A5 沿用同一策略 |

## 汇总

- **活跃条款**（已纳入 schema）：SPEC-CHIP-1 / DTYPE-1 / DTYPE-2 / IO-1 / ERROR-1 / PARADIGM-1 / LIFECYCLE-1 = 7 条 ✓；SPEC-FORMULA-1 = 1 条 ⚠（已知建模决策 #1 的合法 textual_only 降级，非阻塞）。
- **v1 暂缓条款**（schema 未定义，⚠ 非阻塞）：SPEC-DAV-1 / ARG-1 / PERF-1 / RES-1 / INTERFACE-1 = 5 条。
- **❌ 数量**：0。
- 按强制规则 R4：无任一 ❌，全部 ✓/⚠ → **状态=✅通过**。

## 问题清单

无 ❌ 级问题。⚠ 项均为 schema 未纳入的暂缓字段或已知建模决策的合法降级，不阻塞进入方案设计 / 测试设计。
