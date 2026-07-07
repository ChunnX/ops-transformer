# FfnWorkerBatching 测试设计文档（新增 Ascend950/A5 支持）

**状态**: ✅通过

**算子**: FfnWorkerBatching（snake：`ffn_worker_batching`；OpType：`FfnWorkerBatching`）
**本次范围**: 新增 Ascend950(arch35/A5) 支持，A5 对外支持范围对齐 A2；A2/A3 分支不改。
**真值源**: `docs/spec.yaml`（L0 数学契约，SPEC_REVIEW ✅通过）；`docs/REQUIREMENTS.md`（需求背景/调用方式）。
**算子类别**: 非计算类（Sort 排序 + IndexGather 索引 gather + Scan/分组计数），无浮点算术，属搬移/重排类。
**精度口径**: 二进制一致（Bitwise Match, `np.array_equal`，rtol=atol=0），取自 `spec.yaml numerical_tolerance.per_dtype`（全 dtype `bitwise_equal`）。

---

## 1. 输入优先级与字段所有权

以下 spec-owned 字段直接取自 `docs/spec.yaml`，未从 REQUIREMENTS 重新解释：

- `dtype_policy.supported_combinations`（3 profile）
- `outputs[].shape_rule`（Y=A×BS×(topK+1) 等静态推导）
- `numerical_tolerance.per_dtype`（bitwise_equal）
- `boundary_conditions[]`
- `extreme_inputs[]`

REQUIREMENTS 与 spec.yaml 在 spec-owned 字段上**无冲突**（SPEC_REVIEW 已交叉核对，13 条 SPEC-* 条款 0 个 ❌）。

---

## 2. spec.yaml 测试映射

| spec.yaml 字段 | 测试来源 / 承接 | 承接位置 |
|----------------|-----------------|----------|
| `dtype_policy.supported_combinations`（P0 fp16 / P1 bf16 / P2 int8） | token_dtype∈{0,1,2} 枚举因子 → y.dtype 随动（0→fp16/1→bf16/2→int8）；schedule_context 恒 int8 | §5.1 dtype 矩阵；L0（3 profile 全覆盖）、L1（`_PRF_*` 6 条 profile×path 矩阵 + 500 pairwise）、design `07_因子值.csv` Y-DTYPE-001 |
| `outputs[].shape_rule`（y=[Y,H]、group_list=[expert_num,2]、id=[Y]、actual_token_num=[1]，Y=A×BS×(topK+1)） | 约束 `Y-SHAPE-001/GL-SHAPE-001/SID/MBID/TID/EOFF/DS/ATN-SHAPE-001` 由 max_out_shape/expert_num 静态推导 | §5.2 shape 关系；全部用例 tensor_view_shapes 静态推导 |
| `boundary_conditions[]`（11 条：k=0 / k=N(A=1024,topK+1=64) / same_bin(expert_num=8192) / expert_num 越界 / topK+1 越界 / A 越界 / max_out_shape 长度≠4 / dtype≠INT8 / 非 1 维 / need_schedule=0 / need_schedule=1） | boundary 用例逐条覆盖 | §4.1 边界覆盖表（L1_BND_* 非报错项 + L2_ERR_* 报错项） |
| `extreme_inputs[]`（3 条：all_same_index / token_dtype=2 连续 scale / 稀疏 mask） | extreme 用例覆盖 | §4.2 极端输入表（L1_EXT_*） |
| `math_semantics.reference_oracle`（`absent: true`, framework=numpy） | CPU numpy golden 精确重排 + `np.array_equal` 对拍（见 §6 oracle 选择） | §6 oracle 与 golden 来源 |
| `numerical_tolerance.per_dtype`（全 dtype bitwise_equal, rtol=0/atol=0） | 精度断言 = 二进制一致 | §5.3 精度验收标准 |
| `determinism`（required=true, sequential, bitwise_reproducible） | 确定性/重复执行用例 | §7 确定性测试 |

> `blackbox_case_targets` 机器可判段见文末。

---

## 3. 测试设计流程与产物（ascendc-st-design 默认流程）

| 阶段 | 产物 | 路径 |
|------|------|------|
| 参数定义 | 1 输入(schedule_context) + 5 属性 + 8 输出 | `tests/st/design/03_参数定义.yaml` |
| 测试因子 | 53 因子 | `tests/st/design/04_测试因子.yaml` |
| 约束定义 | 输出 shape/dtype 静态推导（calculate 型） | `tests/st/design/05_约束定义.yaml` |
| 求解配置 | 拓扑分层（4 level） | `tests/st/design/06_求解配置.yaml` |
| 因子值 | 10000 组（约束全满足） | `tests/st/design/07_因子值.csv` |
| L0 用例 | 门槛（单因子覆盖 100%） | `tests/st/testcases/aclnnFfnWorkerBatching_l0_test_cases.csv` |
| L1 用例 | 功能（pairwise + 显式 boundary/extreme/profile） | `tests/st/testcases/aclnnFfnWorkerBatching_l1_test_cases.csv` |
| L2 用例 | 异常（4 spec error_codes） | `tests/st/testcases/aclnnFfnWorkerBatching_l2_test_cases.csv` |
| 覆盖报告 | L0/L1 覆盖度 | `tests/st/testcases/aclnnFfnWorkerBatching_l{0,1}_coverage_report.yaml` |

---

## 4. 边界与极端输入覆盖（逐条对齐 spec.yaml）

### 4.1 boundary_conditions 覆盖表

| # | spec.yaml boundary case | 关键构造（属性 / schedule_context） | 判据 | 承接用例 | 级别 |
|---|--------------------------|-------------------------------------|------|----------|------|
| 1 | k=0：全部 token 被 mask，无有效 token | token_dtype=0, need_schedule=0；所有 expert_id 置极大值排末尾 | actual_token_num=0；group_list 全 [0,0] | `L1_BND_k_zero_all_masked` | L1 |
| 2 | k=N：无 mask 全有效，A 上界=1024、topK+1 上界=64 | max_out_shape=[1024,4,64,128] | output_shape_static_contract | `L1_BND_k_eq_n_A1024_topk64` | L1 |
| 3 | same_bin：全落同一专家 + expert_num 上界=8192 | expert_num=8192, token_dtype=0 | output_shape_static_contract | `L1_BND_same_bin_expert8192` | L1 |
| 4 | expert_num 下界=1 | expert_num=1 | output_shape_static_contract | `L1_BND_expert_num_lower_1` | L1 |
| 5 | topK+1 上界=64（内部） | max_out_shape=[8,4,64,128] | output_shape_static_contract | `L1_BND_topk64_upper` | L1 |
| 6 | A 上界=1024（内部） | max_out_shape=[1024,4,8,64] | output_shape_static_contract | `L1_BND_A1024_upper` | L1 |
| 7 | expert_num 越界 (>8192) → 报错 | expert_num=8193 | raises_error: attribute_value_out_of_range | `L2_ERR_attr_expert_num_over_8193` | L2 |
| 8 | topK+1 越界 (>64) → 报错 | max_out_shape=[8,4,65,128] | raises_error: attribute_value_out_of_range | `L2_ERR_attr_topk_over_65` | L2 |
| 9 | A 越界 (>1024) → 报错 | max_out_shape=[1025,4,8,128] | raises_error: attribute_value_out_of_range | `L2_ERR_attr_A_over_1025` | L2 |
| 10 | max_out_shape 长度≠4 → 报错 | max_out_shape=[8,4,8] | raises_error: attribute_value_out_of_range | `L2_ERR_attr_max_out_shape_len3` | L2 |
| 11 | schedule_context dtype≠INT8 → 报错 | schedule_context.dtype=float16 | raises_error: dtype_not_supported | `L2_ERR_dtype_not_supported_fp16` | L2 |
| 12 | schedule_context 非 1 维 → 报错 | schedule_context.shape=[2,512] | raises_error: shape_mismatch | `L2_ERR_shape_mismatch_2d_2x512` | L2 |
| 13 | need_schedule=0 NORM 路径 (TilingKey 100) | need_schedule=0, token_dtype=0 | output_shape_static_contract | `L1_PRF_P0_fp16_NORM` | L1 |
| 14 | need_schedule=1 RECV 路径 (TilingKey 101) | need_schedule=1, token_dtype=0 | output_shape_static_contract | `L1_PRF_P0_fp16_RECV` | L1 |

> boundary_conditions[] 共 11 条 case（spec）；上表逐条覆盖全部 11 条——第 1–12 行覆盖前 9 条（k=0 / k=N / same_bin / expert_num·topK+1·A 越界 / max_out_shape 长度≠4 / dtype≠INT8 / 非 1 维，另含 expert_num 下界、内部 topK+1·A 上界补强），第 13/14 行分别对应 need_schedule=0 / need_schedule=1 两条独立 boundary case（TilingKey 100/101），并由 500 pairwise 冗余覆盖。

### 4.2 extreme_inputs 覆盖表

| # | spec.yaml extreme case | 关键构造 | 判据 | 承接用例 | 级别 |
|---|-------------------------|----------|------|----------|------|
| 1 | all_same_index：全相同 gather 索引/全同专家（极端访存） | expert_num=8192, token_dtype=0；所有有效 token expert_id 相同 | output_shape_static_contract；gather+group_listing 二进制一致 | `L1_EXT_all_same_index` | L1 |
| 2 | token_dtype=2：INT8 + FP32 scale 连续排布（每 token 尾部 32B），y=INT8 且 dynamic_scale 有效 | token_dtype=2；ffn_wb_gather_out_all 按 BLOCK_BYTES=32 连续搬运 | output_shape_static_contract；dynamic_scale 原样字节 gather | `L1_EXT_dynamic_quant_int8_scale` | L1 |
| 3 | 稀疏有效 token：约 50% mask，无效排末尾，有效按专家升序聚合，actual_token_num<Y | token_dtype=0, need_schedule=1 | output_shape_static_contract | `L1_EXT_sparse_50pct_mask_recv` | L1 |

### 4.3 schedule_context (1024B 结构体) 数据构造约束（测试开发依据）

本算子唯一物理输入为 `schedule_context`（INT8，1024 字节），token 数据经 FfnArea 指针间接读取。测试数据须按 `ScheduleContext` 结构逐字节构造（CommonArea/ControlArea/AttentionArea/FfnArea），并单独分配 `token_data_buf`。关键约束：

- CommonArea：`session_num`(=A) / `micro_batch_num`(=M≤64) / `micro_batch_size`(=BS) / `selected_expert_num`(=topK+1) / `expert_num` 须与属性一致。
- FfnArea：`token_info_buf`([A,M,F] 含 flag/layer_id/expert_ids)、`token_data_buf`([A,M,BS,K+1,HS])、输出区各 buf 指针须指向合法分配区。
- 被 mask 的无效 token：expert_id 初始化为极大值（排到末尾）。
- token_dtype=2：每 token 尾部额外 `BLOCK_BYTES=32` 承载 FP32 dynamic scale，与 INT8 数据连续。
- need_schedule=1：需填充 token_info 供 RECV 路径扫描接收。

> 该逐字节构造为测试工程开发阶段（场景二）的核心工作，本设计阶段仅定义参数矩阵与语义约束。

---

## 5. dtype / shape / 精度

### 5.1 dtype 矩阵（3 profile，取自 spec.dtype_policy.supported_combinations）

| profile | token_dtype | schedule_context | y | group_list | session/micro_batch/token/expert_offsets | dynamic_scale | actual_token_num |
|---------|-------------|------------------|-----|-----------|------------------------------------------|---------------|------------------|
| P0 | 0 | INT8 | **FP16** | INT64 | INT32 | FP32(空) | INT64 |
| P1 | 1 | INT8 | **BF16** | INT64 | INT32 | FP32(空) | INT64 |
| P2 | 2 | INT8 | **INT8** | INT64 | INT32 | **FP32(有效)** | INT64 |

A5 三 profile 全量对齐 A2，无新增/删减 dtype（不引入 arch35 独有 FP8/MXFP8/HiF8）。

### 5.2 shape 关系（取自 spec.outputs[].shape_rule）

- `Y = A × BS × (topK+1) = max_out_shape[0] × max_out_shape[1] × max_out_shape[2]`，`H = max_out_shape[3]`
- `y=[Y,H]`、`group_list=[expert_num,2]`、`session_ids/micro_batch_ids/token_ids/expert_offsets/dynamic_scale=[Y]`、`actual_token_num=[1]`
- 上限：A∈(0,1024]、topK+1∈(0,64]、M≤64、expert_num∈(0,8192]、BS/Y/H 泛化。
- 所有输出 shape 由属性静态推导（infershape），与代次无关，A5 与 A2 全对齐。

### 5.3 精度验收标准（取自 spec.numerical_tolerance.per_dtype）

| 输出 | dtype | 精度判定（rtol/atol/metric） |
|------|-------|------------------------------|
| y | FP16 / BF16 / INT8 | 0 / 0 / **bitwise_equal** |
| group_list | INT64 | 0 / 0 / bitwise_equal |
| session_ids / micro_batch_ids / token_ids / expert_offsets | INT32 | 0 / 0 / bitwise_equal |
| dynamic_scale | FP32 | 0 / 0 / bitwise_equal（原样字节 gather，非重算） |
| actual_token_num | INT64 | 0 / 0 / bitwise_equal |

全部输出为整数索引或原样字节搬运，无浮点算术误差，验收 = CPU golden **二进制一致（`np.array_equal`）**；A5 与 A2 同一口径。精度标准来源 `ops-precision-standard → 非计算类标准`。

---

## 6. oracle 与 golden 来源

spec.yaml `math_semantics.reference_oracle.absent: true`（framework=numpy）——非计算类算子无单一 torch/numpy callable 可表达「解析 1024B 调度上下文 + 多字段 gather + group_listing」。

**替代 golden 来源（显式声明）**：CPU numpy 参考实现按三步语义精确重排——
1. 对 expert_id 稳定排序（被 mask 置极大值排末尾）生成 gather_idx；
2. 按 gather_idx 从 token_data 原样字节搬运 y / dynamic_scale，并查表 session/micro_batch/token/expert_offset；
3. 单核扫描专家跳变点生成 group_list 与 actual_token_num。

输出用 `np.array_equal` 逐输出二进制对拍（无数值容差）。语义由 spec.invariants（`output_shape_static_contract` / `no_leak_intermediate_indices`）+ boundary/extreme 机器断言共同锚定。此为 spec `reference_oracle.absent=true` 的合法替代，已记录于本节。

---

## 7. 确定性测试（取自 spec.determinism）

- `required: true`、`accumulation_order: sequential`、`bitwise_reproducible: true`。
- 测试项：相同 schedule_context 重复执行 N 次，y / group_list / 各 id / dynamic_scale / actual_token_num 须完全二进制一致。
- 排序 key = 专家 ID + 稳定序；group_list 整数精确累加；多核 gather 切分后按固定专家聚合顺序合并。
- A5 沿用同一确定性策略（见 REQUIREMENTS §8.3）。

---

## 8. 用例分级

| 级别 | 定位 | 覆盖 | 用例数 | 文件 |
|------|------|------|--------|------|
| L0 | 门槛 | 基础 shape + 单 dtype，单因子覆盖 100%（30/30 因子值） | 5 | `..._l0_test_cases.csv` |
| L1 | 功能 | 500 pairwise（6 功能路径全覆盖 + expert_num 5 档）+ 6 boundary + 6 profile×path 矩阵 + 3 extreme | 515 | `..._l1_test_cases.csv` |
| L2 | 异常 | 4 spec error_codes：null_input(1) / dtype_not_supported(2) / shape_mismatch(2) / attribute_value_out_of_range(8) | 13 | `..._l2_test_cases.csv` |

L2 错误码映射（全部 ⊆ spec.error_codes `[null_input, shape_mismatch, dtype_not_supported, attribute_value_out_of_range]`）：

| 错误码 | 用例 |
|--------|------|
| null_input | `L2_ERR_null_input_schedule_context` |
| dtype_not_supported | `L2_ERR_dtype_not_supported_fp16` / `..._int32` |
| shape_mismatch | `L2_ERR_shape_mismatch_2d_2x512` / `..._2d_1024x1` |
| attribute_value_out_of_range | `L2_ERR_attr_expert_num_over_8193` / `..._expert_num_zero` / `..._topk_over_65` / `..._A_over_1025` / `..._max_out_shape_len3` / `..._token_dtype_over_3` / `..._need_schedule_over_2` / `..._layer_num_gt_expert` |

> TilingKey 102 (RECV_GATHER) 不测：A2 从不启用（tiling 仅 SetTilingKey 100/101），本次不移植（对齐 REQUIREMENTS §4.5）。

---

## 9. A5 与 A2 对齐性说明（测试视角）

对外可见支持范围（dtype 矩阵 / shape 约束 / 功能路径 / 精度口径 / 确定性）A5 全对齐 A2，测试矩阵不区分代次。差异仅在内部实现（arch35 全新 kernel/tiling、UB 容量运行时查询导致内部切分阈值重算），不改变任何黑盒可见行为，故黑盒用例 A5 复用同一套设计。A5 实机（Ascend950DT_9584, DAV_3510）执行 Real 模式精度对拍。

---

## 10. 已知说明

- `layer_num` 取值域收敛为 {0,1}（保证任意 (layer_num, expert_num) 组合恒满足 spec 语义 `layer_num≤expert_num`，避免 L0/L1 正常用例出现越界）；`layer_num>expert_num` 越界由 L2 `attribute_value_out_of_range` 覆盖。
- L1 pairwise 覆盖 94/100：未达组合为 `layer_num≤expert_num` 语义约束下不可行的离散对，非缺陷。
- 空 tensor 场景不适用：`schedule_context` 为固定 1024B 结构体（aclnn 接口明确「不支持空Tensor」），故 L0/L1 跳过空 tensor 派生。

---

blackbox_case_targets:
  L0: 5
  L1: 515
  L2: 13

