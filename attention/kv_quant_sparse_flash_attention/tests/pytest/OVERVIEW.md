# pytest 测试框架概览

> 本文件供新会话快速了解测试框架结构、职责划分和当前工作区状态

## 目录结构

```text
pytest/
├── utils.py                                    # 公共工具模块（格式转换、numpy→torch、结果保存、MD5 等）
├── generate_tensor_data.py                     # 输入张量生成（numpy 生成后转 torch）
├── kv_quant_sparse_flash_attention_golden.py   # CPU golden 与张量路由
├── kv_quant_sparse_flash_attention_paramset.py # 参数配置
├── check_valid_param.py                        # 参数校验
├── result_compare_method.py                    # 结果对比方法
│
├── test_blackbox_contract.py                   # 参数契约测试（6个）
├── test_blackbox_runtime.py                    # 运行时行为测试（6个）
├── test_blackbox_snapshot.py                   # 数据生成形状测试（9个）
├── test_kv_quant_sparse_flash_attention_single.py # 单用例执行主入口
├── test_kv_quant_sparse_flash_attention_batch.py  # 批量回放入口
│
├── batch/
│   ├── kv_quant_sparse_flash_attention_process.py   # NPU 执行流程
│   └── test_kv_quant_sparse_flash_attention_pt_save.py # PT 文件生成入口
│
├── README.md                                   # 测试框架说明
├── OVERVIEW.md                                 # 本文件
├── TEST_PROGRESS.md                            # 测试进展记录
└── pytest.ini                                  # pytest 配置
```

## 核心模块职责

| 模块 | 职责 | 主要函数 |
|------|------|----------|
| `utils.py` | 底层公共工具 | `get_np_dtype`, `numpy_to_torch`, `get_torch_dtype`, `print_log`, `dump_file`, `save_result`, `INPUT_TENSOR_NAMES`, `INPUT_ATTR_NAMES` |
| `generate_tensor_data.py` | 生成和落盘输入张量 | `gen_tensor_data`, `generate_input_tensors` (返回 test_case_data) |
| `kv_quant_sparse_flash_attention_golden.py` | CPU golden 和张量设备路由 | `generate_and_save_testdata`, `route` (统一处理 torch tensor) |
| `test_kv_quant_sparse_flash_attention_single.py` | 单用例执行编排 | `sub_process_cpu`, `qsfa`, `qsfa_run_npu`, `qsfa_run_cpu_and_save_md5` |
| `check_valid_param.py` | 参数校验 | `check_valid_param` |

## 统一 TestCaseData 结构

```python
test_case_data = {
    "testcase_name": "...",       # 测试用例名称
    "params": {...},              # 参数字典（完整参数）
    "input_tensors": {            # 输入张量（torch.Tensor）
        "query": torch.Tensor,
        "key": torch.Tensor,
        "value": torch.Tensor,
        "sparse_indices": torch.Tensor,
        "block_table": torch.Tensor,
        "actual_seq_lengths_query": torch.Tensor,
        "actual_seq_lengths_kv": torch.Tensor,
        "key_dequant_scale": torch.Tensor,
        "value_dequant_scale": torch.Tensor,
    },
    "input_attrs": {              # 运行时属性（非张量参数）
        "scale_value": float,
        "layout_query": str,
        "layout_kv": str,
        "sparse_mode": int,
        "key_quant_mode": int,
        "value_quant_mode": int,
        "sparse_block_size": int,
        "attention_mode": int,
        "quant_scale_repo_mode": int,
        "tile_size": int,
        "rope_head_dim": int,
        "kv_dtype": str,
        "pre_tokens": int,
        "next_tokens": int,
    },
    "cpu_output": torch.Tensor,   # CPU 输出结果
}
```

### 设计要点

1. **统一数据类型**: `input_tensors` 全部使用 `torch.Tensor`
2. **分离张量和属性**: 张量数据在 `input_tensors`，运行时参数在 `input_attrs`
3. **统一访问方式**: 使用字典键名 `test_case_data["input_tensors"]["query"]`
4. **numpy 仅用于生成**: `gen_tensor_data()` 内部用 numpy 处理特殊 dtype，输出后调用 `numpy_to_torch()` 转换

## numpy_to_torch 转换规则

```python
# utils.py 中定义的特殊类型转换
"uint64"        → torch.int64    # torch 不支持 uint64
"bfloat16"      → torch.bfloat16
"float16"       → torch.float16
"float8_e4m3fn" → torch.float8_e4m3fn
"float8_e5m2"   → torch.float8_e5m2
其他            → torch.from_numpy 直接转换
```

## 当前职责边界

- `utils.py`: 通用工具能力，包括 numpy→torch 转换、文件操作、MD5 计算
- `generate_tensor_data.py`: 输入数据生成，返回统一 `test_case_data` 结构
- `kv_quant_sparse_flash_attention_golden.py`: CPU golden 计算，`route()` 统一处理 torch tensor
- `test_kv_quant_sparse_flash_attention_single.py`: 单用例主入口，参数组装、CPU 生成、NPU 回放
- `batch/kv_quant_sparse_flash_attention_process.py`: `_merge_input_for_npu()` 合并 tensors + attrs 供 NPU 调用

## 依赖关系

```text
utils.py
    ├── 被 generate_tensor_data.py 导入（numpy_to_torch, INPUT_TENSOR_NAMES）
    ├── 被 kv_quant_sparse_flash_attention_golden.py 导入（numpy_to_torch, INPUT_TENSOR_NAMES）
    ├── 被 test_kv_quant_sparse_flash_attention_single.py 导入
    └── 被黑盒测试导入

generate_tensor_data.py
    └── 被 test_kv_quant_sparse_flash_attention_single.py 调用

kv_quant_sparse_flash_attention_golden.py
    └── 被 test_kv_quant_sparse_flash_attention_single.py 调用（route）

test_kv_quant_sparse_flash_attention_single.py
    ├── 被 test_blackbox_runtime.py 复用
    ├── 被 test_kv_quant_sparse_flash_attention_batch.py 复用
    └── 被 batch/test_kv_quant_sparse_flash_attention_pt_save.py 复用
```

## 测试用例分类

### 黑盒测试（21个）

| 文件 | 数量 | 测试内容 |
|------|------|----------|
| `test_blackbox_contract.py` | 6 | 参数契约、格式转换、基线覆盖 |
| `test_blackbox_runtime.py` | 6 | 运行时行为、MD5 对比、基线更新 |
| `test_blackbox_snapshot.py` | 9 | 数据生成形状契约、参数矩阵 |

### 功能测试

| 文件 | 说明 |
|------|------|
| `test_kv_quant_sparse_flash_attention_single.py` | 单用例执行与 MD5 基线校验 |
| `test_kv_quant_sparse_flash_attention_batch.py` | 加载 pt 文件并批量执行回放 |
| `batch/test_kv_quant_sparse_flash_attention_pt_save.py` | 根据参数组合生成 pt 文件 |

## 快速命令

```bash
# 运行所有黑盒测试
pytest -v test_blackbox_contract.py test_blackbox_runtime.py test_blackbox_snapshot.py

# 运行单用例主入口
pytest -v test_kv_quant_sparse_flash_attention_single.py

# 运行批量回放
pytest -v test_kv_quant_sparse_flash_attention_batch.py

# 运行 PT 生成入口
pytest -v batch/test_kv_quant_sparse_flash_attention_pt_save.py

# 检查导入是否正常
python3 -c "import utils; import generate_tensor_data; import kv_quant_sparse_flash_attention_golden; import test_kv_quant_sparse_flash_attention_single"

# 检查底层工具是否仍然唯一
grep -r "^def get_np_dtype" --include="*.py" . | wc -l  # 应为 1
grep -r "^def numpy_to_torch" --include="*.py" . | wc -l  # 应为 1
```

## 当前状态

- **已完成的重构**:
  1. Session 1: `tools.py` 迁移到 `utils.py`
  2. Session 2: 测试执行逻辑迁移到 `test_kv_quant_sparse_flash_attention_single.py`
  3. Session 3: 统一 `test_case_data` 结构，分拆 `input_tensors` 和 `input_attrs`
  4. Session 4: 统一使用 torch tensor，简化 `route()` 逻辑

- **当前测试状态**: 黑盒测试 21/21 通过

- **代码状态**: 重构完成，等待提交

## 重构历史

详见 `TEST_PROGRESS.md` 中的 Session 1-4 记录。