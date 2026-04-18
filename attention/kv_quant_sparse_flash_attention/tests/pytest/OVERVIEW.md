# pytest 测试框架概览

> 本文件供新会话快速了解测试框架结构、职责划分和当前工作区状态

## 目录结构

```text
pytest/
├── utils.py                                    # 公共工具模块（格式转换、结果保存、MD5 等）
├── generate_tensor_data.py                     # 输入张量生成
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
| `utils.py` | 底层公共工具 | `get_np_dtype`, `print_log`, `dump_file`, `modify_alcnn_input_file`, `convert_param_combination_to_cs_format`, `save_result` |
| `generate_tensor_data.py` | 生成和落盘输入张量 | `gen_tensor_data`, `gen_uniform_data`, `gen_normal_data`, `generate_input_tensor_list` |
| `kv_quant_sparse_flash_attention_golden.py` | CPU golden 和张量设备路由 | `aclnn_op_func_cpu`, `generate_and_save_testdata`, `route` |
| `test_kv_quant_sparse_flash_attention_single.py` | 单用例执行编排 | `sub_process_cpu`, `qsfa`, `qsfa_run_npu`, `qsfa_run_cpu_and_save_md5` |
| `check_valid_param.py` | 参数校验 | `check_valid_param` |

## 当前职责边界

- `utils.py` 只保留通用工具能力，不再承载测试执行编排。
- `generate_tensor_data.py` 负责输入数据构造与输入 bin 落盘。
- `kv_quant_sparse_flash_attention_golden.py` 负责 CPU golden 计算，以及 `route()` 中的设备前数据整理。
- `test_kv_quant_sparse_flash_attention_single.py` 是当前单用例主入口，负责参数组装、CPU 生成、NPU 回放和 MD5 基线验证。
- `test_kv_quant_sparse_flash_attention_batch.py` 和 `batch/test_kv_quant_sparse_flash_attention_pt_save.py` 通过导入 `single` 模块复用执行入口。

## 依赖关系

```text
utils.py
    ├── 被 generate_tensor_data.py 导入
    ├── 被 kv_quant_sparse_flash_attention_golden.py 导入
    ├── 被 test_kv_quant_sparse_flash_attention_single.py 导入
    └── 被黑盒测试导入

generate_tensor_data.py
    └── 被 test_kv_quant_sparse_flash_attention_single.py 调用生成输入

kv_quant_sparse_flash_attention_golden.py
    └── 被 test_kv_quant_sparse_flash_attention_single.py 调用执行 CPU route/golden

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
```

## 当前状态

- 已完成的稳定状态：
  `tools.py` 的公共能力已迁移到 `utils.py`，黑盒测试曾跑通 `21/21`。
- 当前工作区正在进行的新一轮调整：
  测试执行逻辑正从 `utils.py` 迁移到 `test_kv_quant_sparse_flash_attention_single.py`。
- 当前未完成事项：
  代码仍处于未提交状态，需要在本轮修改后重新验证。
- 当前不要假设：
  历史 `21 passed` 结果能自动覆盖这轮最新改动。

## 下一步可能的工作

1. 重新执行黑盒测试，确认 `single` 入口迁移后运行时行为未回归。
2. 评估 `test_kv_quant_sparse_flash_attention_single.py` 中新增执行逻辑是否还应继续拆分，避免测试文件承担过多基础设施职责。
3. 补充 batch/PT save 路径的回归验证，确保调用 `single` 后行为一致。
