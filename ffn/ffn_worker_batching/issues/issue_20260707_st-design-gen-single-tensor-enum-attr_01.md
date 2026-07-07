# issue: ascendc-st-design generate_test_cases.py 对「单张量入参 + 枚举属性驱动」类算子的 4 处缺陷

- **编号**: issue_20260707_st-design-gen-single-tensor-enum-attr_01
- **发现时间**: 2026-07-07
- **发现场景**: 1.4 测试设计（FfnWorkerBatching 新增 Ascend950/A5 支持）
- **归属组件**: 共享 skill `ascendc-st-design`（脚本 `scripts/generate_test_cases.py`）
- **严重度**: 中（阻塞该类算子跑通默认 ST 设计流程；本次已用临时补丁绕过，未影响交付物正确性）
- **状态**: 待回补 skill 主线

## 1. 现象

FfnWorkerBatching 的参数结构为「**单一 aclTensor 输入（schedule_context, INT8, [1024]）+ 5 个枚举标量属性（expert_num/max_out_shape/token_dtype/need_schedule/layer_num）+ 8 个 aclTensor 输出**」，即算子行为变化几乎全部由标量/数组属性枚举驱动，而非传统的 tensor dtype/shape 变化。

用 `ascendc-st-design` 默认流程（`generate_test_cases.py` 生成 L0/L1/L2）时，L0/L1/L2 三个级别分别在不同代码路径触发 4 处报错/逻辑缺陷，导致无法直接产出用例。传统「多 tensor 输入、dtype/shape 驱动」算子不易触发这些路径，故此前未暴露。

## 2. 四处缺陷定位

### BUG-1: `TensorProcessData.tensor_list_lengths` 为必填字段，aclTensor 路径构造时缺参

- **文件/位置**: `scripts/generate_test_cases.py`，`@dataclass class TensorProcessData`（约 L74-83）；触发于 `_process_single_ttk_case` 内 aclTensor 分支构造 `TensorProcessData(...)`（约 L1513）。
- **触发条件**: 用例含 aclTensor 参数但**不含** aclTensorList 时，aclTensor 分支构造 `TensorProcessData` 未传 `tensor_list_lengths`。
- **报错**: `TypeError: TensorProcessData.__init__() missing 1 required positional argument: 'tensor_list_lengths'`。
- **根因**: `tensor_list_lengths` 声明为必填字段，但仅 aclTensorList 分支传该参数；纯 aclTensor 用例的构造点漏传。

### BUG-2: L1 pairwise 只识别 `.dtype/.dimensions/.exist/.format` 与硬编码 `cubeMathType.value`，枚举标量属性 `.value` 被过滤

- **文件/位置**: `scripts/generate_test_cases.py`，`generate_pairwise_combinations` 内 `is_discrete_factor`（约 L787-811）与 `select_cases_L1` 内同名闭包（约 L889-913）；判定行 `if factor_name.endswith('.value') and not factor_name == 'cubeMathType.value': return False`（L795、L897）。
- **触发条件**: 算子变化维度来自枚举标量属性（`expert_num.value`/`token_dtype.value`/`need_schedule.value`/`layer_num.value` 等），而非 tensor dtype/dimensions。
- **报错/后果**: 离散多值因子仅剩 1 个 → 两两组合数=0 → `select_cases_L1` 选中 0 用例 → `pad_cases` 收到空序列 `IndexError: Cannot choose from an empty sequence`；且 `select_cases_L1` 内 `len(pairwise_combinations)=0` 触发 `ZeroDivisionError`。
- **根因**: 离散因子白名单硬编码，仅认 `cubeMathType.value` 一个枚举属性，未泛化到任意枚举标量属性 `.value`。

### BUG-3: `generate_format_error_cases` 引用未定义变量 `case_name_str`

- **文件/位置**: `scripts/generate_test_cases.py`，`generate_format_error_cases`（约 L2174-2211），`ErrorCaseConfig(... case_name=case_name_str ...)`（L2199）。
- **触发条件**: 生成 L2 时算子仅支持单一 format（如仅 `ND`），脚本判定存在「不支持的 format」并进入格式异常用例分支。
- **报错**: `NameError: name 'case_name_str' is not defined`。
- **根因**: 该函数从未定义 `case_name_str` 局部变量即直接引用（疑似重构遗漏），其它同类函数均先构造 `case['case_name'] = f"..."`。
- **附注**: 对本算子而言，`format_not_supported` 并不在 spec.error_codes（`null_input/shape_mismatch/dtype_not_supported/attribute_value_out_of_range`）内，属脚本对「单 format 算子」的误判额外用例；本次交付已改为按 spec.error_codes 重建 L2。

### BUG-4: `pd.isna` / `pd.notna` 在 list/ndarray 值上产生数组布尔歧义

- **文件/位置**: `scripts/generate_test_cases.py`，`format_ttk_attr_value`（`if pd.isna(value):` L1683）与 `create_error_ttk_dataframe` 调用链上的 `_process_scalar_types`（`if pd.notna(value) and value != '':` L2643/L2648）。
- **触发条件**: 属性为数组类型（如 `aclIntArray max_out_shape`，value 为 `[8,4,8,128]` 列表/ndarray）时，`pd.isna(list_or_ndarray)` 返回逐元素布尔数组。
- **报错**: `ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()`。
- **根因**: 标量空值判定 `pd.isna/pd.notna` 未先分流 list/tuple/ndarray 类型。

## 3. 本次临时绕过方式

- **不改动共享 skill**：将 `scripts/*.py` 复制到临时工作区 `/home/zl/workspace/ffn_wb_a5_tmp/scripts/`，仅对副本打补丁后运行；`generate_test_factors.py` / `generate_implicit_constraints.py` / `generate_solver_config.py` / `generate_factor_values.py` 仍直接调用共享 skill 原件（无 bug）。
- **补丁副本路径**: `/home/zl/workspace/ffn_wb_a5_tmp/scripts/generate_test_cases.py`。
- **补丁要点（与上文 BUG 一一对应）**:
  - BUG-1: `TensorProcessData.tensor_list_lengths` 改为 `= None` 默认值，`__post_init__` 中置空列表。
  - BUG-2: 在两处 `is_discrete_factor` 增加枚举属性白名单 `{expert_num.value, token_dtype.value, need_schedule.value, layer_num.value}`（并入 `discrete_patterns`），排除判定改为 `factor_name not in _enum_attr_value_whitelist`；`max_out_shape.value`（list 派生因子）不列入。
  - BUG-3: 在 `generate_format_error_cases` 内补 `case_name_str = f"{aclnn_name}_{supported_dtype}_{format_name}_{param_name}_format_exception"`。
  - BUG-4: `format_ttk_attr_value` 与 `_process_scalar_types` 先对 list/tuple/ndarray 分流（新增 `_has_scalar_value` 辅助），再对标量走 `pd.isna/pd.notna`。
- **交付说明**: L2 未沿用脚本自动产出（含误判的 `format_not_supported`），改为按 spec.yaml `error_codes` 手工重建 13 条（null_input/dtype_not_supported/shape_mismatch/attribute_value_out_of_range）；L1 在 500 pairwise 基础上手工补齐 6 boundary + 6 profile×path + 3 extreme 显式用例（参数化域 [1,64] 不可达 A=1024/topK+1=64 上界）。

## 4. 对下游 / 其它同类算子的影响

- **本算子**: 交付物（TEST.md、L0=5/L1=515/L2=13、design 中间产物）已通过补丁副本正确产出，1.4R 评审 ✅通过，无功能影响。
- **其它同类算子**: 凡属「单/少 tensor 输入 + 枚举标量或数组属性驱动」形态（如部分调度/路由/重排类、含结构体入参的融合算子），直接用共享 skill 默认流程会命中同样 4 处缺陷，无法产出 L1/L2。建议这类算子在 skill 回补前，参照本 issue 的补丁副本方式绕过。
- **风险**: 补丁副本为临时产物（位于 `/home/zl/workspace/`，非仓内交付），不随算子目录留存；若后续需重跑本算子 ST 设计，需重新准备补丁副本或等待 skill 主线修复。

## 5. 建议修复方向（回补 skill 主线）

1. **BUG-1**: 将 `TensorProcessData.tensor_list_lengths` 改为可选（默认 `None` + `__post_init__` 置 `[]`），与 `TtkCaseData` 一致。
2. **BUG-2**: 泛化离散因子识别——对 `is_enum: true` 的标量属性 `.value`（单值或多值枚举）纳入 pairwise 离散因子；数组类 `.value`（list）继续排除；避免硬编码 `cubeMathType.value`。同时对 `len(pairwise_combinations)==0` 增加保护分支（回退单因子或直接补齐），避免 `ZeroDivisionError` / 空序列 `IndexError`。
3. **BUG-3**: 补齐 `generate_format_error_cases` 的 `case_name_str` 定义；并考虑：当算子仅支持单一合法 format 且 `format_not_supported` 不在算子 error_codes 时，不生成该类误判用例。
4. **BUG-4**: 统一空值判定工具函数，先分流 list/tuple/ndarray（判 `len/size`），再对标量走 `pd.isna/pd.notna`，覆盖 `format_ttk_attr_value` 与 `_process_scalar_types` 等所有调用点。
5. **回归**: 建议在 skill 测试集补充一个「单 aclTensor 输入 + 枚举标量属性 + aclIntArray 属性 + 多 aclTensor 输出」的最小算子样例，作为该形态的回归基线。
