# pytest 测试进展记录

> 记录测试执行命令和重构进展，供新会话快速了解当前状态

## 执行测试命令

### 黑盒测试（保障重构前后功能一致性）

```bash
# 在 pytest 目录下执行
cd /home/panjie/workspace/codes/ops-transformer/attention/kv_quant_sparse_flash_attention/tests/pytest

# 运行所有黑盒测试（推荐）
pytest -v test_blackbox_contract.py test_blackbox_runtime.py test_blackbox_snapshot.py

# 单独运行各文件
pytest -v test_blackbox_contract.py    # 参数契约测试
pytest -v test_blackbox_runtime.py     # 运行时行为测试
pytest -v test_blackbox_snapshot.py    # 数据生成形状测试
```

### 验证导入

```bash
# 检查模块导入是否正常（无循环依赖）
python3 -c "import utils"
python3 -c "import generate_tensor_data"
python3 -c "import kv_quant_sparse_flash_attention_golden"
```

### 检查函数唯一性

```bash
# 确认公共函数只定义一次
grep -r "^def get_np_dtype" --include="*.py" . | wc -l  # 预期: 1
grep -r "^def print_log" --include="*.py" . | wc -l     # 预期: 1
```

## 重构进展

### 已完成

| 时间 | 任务 | 状态 |
|------|------|------|
| Session 1 | 将 `tools.py` 函数迁移到 `utils.py` | ✅ 完成 |
| Session 1 | 删除 `tools.py` 文件 | ✅ 完成 |
| Session 1 | 统一 `get_np_dtype` 函数定义 | ✅ 完成 |
| Session 1 | 统一 `print_log` 函数定义 | ✅ 完成 |
| Session 1 | 修复 SyntaxError 和 IndentationError | ✅ 完成 |
| Session 1 | 消除循环依赖 | ✅ 完成 |

### 当前测试结果

```
============================= test session starts ==============================
collected 21 items

test_blackbox_contract.py::test_check_valid_param_accepts_representative_pa_case PASSED
test_blackbox_contract.py::test_check_valid_param_rejects_non_pa_layout_mismatch PASSED
test_blackbox_contract.py::test_check_valid_param_rejects_actual_seq_length_mismatch PASSED
test_blackbox_contract.py::test_convert_param_combination_to_cs_format_keeps_case_name_and_auto_block_num PASSED
test_blackbox_contract.py::test_combin_params_expands_cartesian_product_without_changing_values PASSED
test_blackbox_contract.py::test_baseline_md5_covers_all_enabled_single_cases PASSED

test_blackbox_runtime.py::test_qsfa_run_npu_records_skipcompare_when_cpu_golden_is_unavailable PASSED
test_blackbox_runtime.py::test_compare_output_bin_md5_detects_match_and_mismatch PASSED
test_blackbox_runtime.py::test_qsfa_run_cpu_and_save_md5_returns_cpu_bin_metadata PASSED
test_blackbox_runtime.py::test_is_baseline_update_enabled_defaults_to_false PASSED
test_blackbox_runtime.py::test_is_baseline_update_enabled_accepts_explicit_switch PASSED
test_blackbox_runtime.py::test_verify_md5_with_baseline_reports_missing_baseline PASSED

test_blackbox_snapshot.py::test_generate_and_save_testdata_bsnd_pa_shapes_match_contract PASSED
test_blackbox_snapshot.py::test_generate_and_save_testdata_tnd_converts_layout_and_prefix_sum PASSED
test_blackbox_snapshot.py::test_generate_and_save_testdata_is_repeatable_under_fixed_seed PASSED
test_blackbox_snapshot.py::test_generate_and_save_testdata_case_matrix_contract[case0] PASSED
test_blackbox_snapshot.py::test_generate_and_save_testdata_case_matrix_contract[case1] PASSED
test_blackbox_snapshot.py::test_generate_and_save_testdata_case_matrix_contract[case2] PASSED
test_blackbox_snapshot.py::test_generate_and_save_testdata_case_matrix_contract[case3] PASSED
test_blackbox_snapshot.py::test_generate_and_save_testdata_case_matrix_contract[case4] PASSED
test_blackbox_snapshot.py::test_generate_and_save_testdata_case_matrix_contract[case5] PASSED

============================== 21 passed in 3.83s ==============================
```

**状态**: ✅ 全部通过

## 重构变更清单

### 文件变更

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `utils.py` | 新增函数 | `get_np_dtype`, `print_log`, `dump_file`, `modify_alcnn_input_file` |
| `generate_tensor_data.py` | 修改 import | `import utils` + `utils.print_log()` |
| `kv_quant_sparse_flash_attention_golden.py` | 修改 import | `import utils` + `utils.xxx()` |
| `batch/kv_quant_sparse_flash_attention_process.py` | 修改 import | `import utils` + `utils.get_np_dtype()` |
| `tools.py` | 删除 | 功能已迁移 |

### 函数迁移

| 函数 | 原位置 | 新位置 |
|------|--------|--------|
| `get_np_dtype` | tools.py, generate_tensor_data.py, kv_quant_sparse_flash_attention_golden.py | utils.py |
| `print_log` | tools.py, generate_tensor_data.py, kv_quant_sparse_flash_attention_golden.py | utils.py |
| `dump_file` | tools.py | utils.py |
| `modify_alcnn_input_file` | tools.py | utils.py |

## 遇到的问题与解决

### 问题 1: 循环依赖

**表现**: `utils.py` 导入 `kv_quant_sparse_flash_attention_golden.py`，后者又导入 `utils.py`

**解决**: 将无依赖的基础函数（`get_np_dtype`, `print_log`）放在 `utils.py` 顶部，其他模块通过 `import utils` 使用

### 问题 2: SyntaxError - unexpected indent

**表现**: 替换函数调用时格式错误
```
k_np_dtype = import utils
    utils.get_np_dtype(...)
```

**解决**: 正确替换为 `k_np_dtype = utils.get_np_dtype(...)`

### 问题 3: 函数重复定义

**表现**: `utils.py` 中有两处 `def get_np_dtype`

**解决**: 删除 line 884 的重复定义，保留 line 51 的定义

### 问题 4: IndentationError

**表现**: 函数体缩进不一致

**解决**: 确保 `utils.print_log` 的缩进与原代码一致

## 后续维护建议

1. **新增公共函数时**: 在 `utils.py` 中定义，其他模块通过 `import utils` 使用
2. **避免循环依赖**: 公共函数尽量不依赖其他模块，必要时延迟导入
3. **定期检查**: `grep -r "^def xxx" --include="*.py"` 确认函数唯一性
4. **测试保障**: 每次重构后运行黑盒测试确认功能一致性

## 当前工作区增量进展

> 记录本轮尚未提交的最新修改，和上面的历史记录并存

### 本轮变更摘要

1. 将原先放在 `utils.py` 的用例执行逻辑移出，避免 `utils` 继续承载测试编排职责
2. 在 `test_kv_quant_sparse_flash_attention_single.py` 中集中承接单用例执行入口
3. 将输入张量批量生成逻辑下沉到 `generate_tensor_data.py`
4. 在 `kv_quant_sparse_flash_attention_golden.py` 中补充 `route()`，统一 CPU/NPU 前的张量路由
5. 更新 batch、runtime 黑盒测试和 pt save 入口，统一改为调用 `single` 模块中的执行函数

### 当前文件变更

| 文件 | 当前修改 | 说明 |
|------|----------|------|
| `utils.py` | 删除函数 | 移除了 `sub_process_cpu`、`route`、`aclnn_op_func`、`qsfa`、`qsfa_run_npu`、`qsfa_run_cpu_and_save_md5` |
| `generate_tensor_data.py` | 新增函数 | 新增 `generate_input_tensor_list(params)`，负责生成输入 tensor 并落盘 |
| `kv_quant_sparse_flash_attention_golden.py` | 新增函数 | 新增 `route(input_tensor_list, params, device)`，承接设备前的数据转换 |
| `test_kv_quant_sparse_flash_attention_single.py` | 新增执行逻辑 | 集中实现 `sub_process_cpu`、`qsfa`、`qsfa_run_npu`、`qsfa_run_cpu_and_save_md5` |
| `test_blackbox_runtime.py` | 调整依赖 | 从调用 `utils` 改为调用 `test_kv_quant_sparse_flash_attention_single` |
| `test_kv_quant_sparse_flash_attention_batch.py` | 调整依赖 | batch 回放改为调用 `single.qsfa_run_npu` |
| `batch/test_kv_quant_sparse_flash_attention_pt_save.py` | 调整依赖 | pt 生成入口改为调用 `single.qsfa` |

### 设计意图

- `utils.py` 保留为底层工具模块，只放格式转换、MD5、文件写入、结果保存等通用能力
- 用例编排和执行流程收口到 `test_kv_quant_sparse_flash_attention_single.py`
- 数据生成与设备路由分别落在 `generate_tensor_data.py` 和 `kv_quant_sparse_flash_attention_golden.py`，职责边界更清楚

### 当前状态

- `git status --short -- attention/kv_quant_sparse_flash_attention/tests/pytest` 显示当前有 9 个已修改文件
- 本轮已验证：黑盒测试 21/21 通过，函数唯一性检查通过，模块导入正常

### 本轮测试结果

```
============================== 21 passed in 3.48s ==============================
```

**状态**: ✅ 全部通过

---

## Session 4: 统一使用 torch tensor 替代 numpy

### 变更摘要

将 `input_tensors` 中的 numpy 数组统一转换为 torch tensor，消除 numpy/torch 混用导致的复杂转换逻辑。

### 文件变更

| 文件 | 当前修改 | 说明 |
|------|----------|------|
| `utils.py` | 新增函数 | `numpy_to_torch()` 处理 numpy→torch 转换，支持 uint64/bfloat16/float8 等特殊类型 |
| `utils.py` | 新增函数 | `get_torch_dtype()` 返回 torch dtype |
| `generate_tensor_data.py` | 修改函数 | `generate_input_tensors()` 生成后调用 `numpy_to_torch()` 转换 |
| `kv_quant_sparse_flash_attention_golden.py` | 简化函数 | `route()` 移除两套分支，统一使用 torch tensor 处理 |
| `test_kv_quant_sparse_flash_attention_single.py` | 简化函数 | `sub_process_cpu()` 移除 numpy→torch 更新逻辑 |

### numpy_to_torch 处理的特殊类型

```python
"uint64"     → torch.int64 (torch 不支持 uint64)
"bfloat16"   → torch.bfloat16
"float8_e4m3fn" → torch.float8_e4m3fn
"float8_e5m2"   → torch.float8_e5m2
"float16"    → torch.float16
其他         → torch.from_numpy 直接转换
```

### 设计意图

- 统一数据类型：`input_tensors` 全部使用 torch tensor
- 简化逻辑：`route()` 从 30+ 行简化到 15 行
- 保持兼容：`gen_tensor_data()` 内部仍用 numpy 处理特殊 dtype 转换，输出后再统一转 torch

---

## Session 3: 统一 params 和 input_tensors 为字典格式

### 变更摘要

将原先使用 `params[0]`、`input_tensor_list[i]` 等索引方式的代码统一改为字典键名访问，提高可读性。同时将 `input_data["input"]` 分拆为 `input_tensors` 和 `input_attrs`，与 `generate_input_tensors` 返回的结构统一。

### 文件变更

| 文件 | 当前修改 | 说明 |
|------|----------|------|
| `utils.py` | 新增常量 | `INPUT_TENSOR_NAMES`、`INPUT_ATTR_NAMES` 定义张量和属性字段 |
| `utils.py` | 修改函数 | `save_result()` 改用字典键名访问 params |
| `generate_tensor_data.py` | 修改函数 | `generate_input_tensors()` 返回完整 `test_case_data` 结构 |
| `kv_quant_sparse_flash_attention_golden.py` | 修改函数 | `generate_and_save_testdata()` 返回统一结构，分拆 `input_tensors` 和 `input_attrs` |
| `kv_quant_sparse_flash_attention_golden.py` | 修改函数 | `route()` 使用字典遍历 |
| `test_kv_quant_sparse_flash_attention_single.py` | 修改函数 | `sub_process_cpu()`、`qsfa_run_cpu_and_save_md5()` 返回统一结构 |
| `test_blackbox_snapshot.py` | 修改测试 | 使用 `test_data["input_tensors"]` 和 `test_data["input_attrs"]` |
| `test_blackbox_runtime.py` | 修改测试 | 适配新的 `test_case_data` 结构 |
| `batch/kv_quant_sparse_flash_attention_process.py` | 新增函数 | `_merge_input_for_npu()` 合并 input_tensors 和 input_attrs |

### 统一 TestCaseData 结构

```python
test_case_data = {
    "testcase_name": "...",       # 测试用例名称
    "params": {...},              # 参数字典（完整参数）
    "input_tensors": {            # 输入张量（numpy 或 torch）
        "query": ...,
        "key": ...,
        "value": ...,
        "sparse_indices": ...,
        "block_table": ...,
        "actual_seq_lengths_query": ...,
        "actual_seq_lengths_kv": ...,
        "key_dequant_scale": ...,
        "value_dequant_scale": ...,
    },
    "input_attrs": {              # 运行时属性（非张量参数）
        "scale_value": ...,
        "layout_query": ...,
        "layout_kv": ...,
        "sparse_mode": ...,
        "key_quant_mode": ...,
        "value_quant_mode": ...,
        "sparse_block_size": ...,
        "attention_mode": ...,
        "quant_scale_repo_mode": ...,
        "tile_size": ...,
        "rope_head_dim": ...,
        "kv_dtype": ...,
        "pre_tokens": ...,
        "next_tokens": ...,
    },
    "cpu_output": ...,            # CPU 输出结果
}
```

### 设计意图

- 统一数据结构：`generate_input_tensors` 和 `generate_and_save_testdata` 都返回 `test_case_data` 结构
- 分离张量和属性：`input_tensors` 存放张量数据，`input_attrs` 存放运行时参数
- 提高可读性：避免 `params[18]`、`input_data["input"]["query"]` 等难以理解的访问方式
