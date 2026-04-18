# pytest 测试框架概览

> 本文件供新会话快速了解测试框架结构和当前状态

## 目录结构

```
pytest/
├── utils.py                          # 核心工具模块（公共函数定义）
├── generate_tensor_data.py           # 测试数据生成器
├── kv_quant_sparse_flash_attention_golden.py  # CPU golden 实现
├── kv_quant_sparse_flash_attention_paramset.py # 参数配置
├── check_valid_param.py              # 参数校验
├── result_compare_method.py          # 结果对比方法
│
├── test_blackbox_contract.py         # 参数契约测试（6个）
├── test_blackbox_runtime.py          # 运行时行为测试（6个）
├── test_blackbox_snapshot.py         # 数据生成形状测试（9个）
├── test_kv_quant_sparse_flash_attention_single.py # 单用例测试
├── test_kv_quant_sparse_flash_attention_batch.py  # 批量用例测试
│
├── batch/
│   ├── kv_quant_sparse_flash_attention_process.py  # NPU 执行流程
│   └── test_kv_quant_sparse_flash_attention_pt_save.py # PT 文件保存
│
├── README.md                         # 测试框架说明
├── OVERVIEW.md                       # 本文件
├── TEST_PROGRESS.md                  # 测试进展记录
└── pytest.ini                        # pytest 配置
```

## 核心模块职责

| 模块 | 职责 | 主要函数 |
|------|------|----------|
| `utils.py` | 公共工具函数 | `get_np_dtype`, `print_log`, `dump_file`, `modify_alcnn_input_file` |
| `generate_tensor_data.py` | 生成测试数据 | `gen_tensor_data`, `gen_uniform_data`, `gen_normal_data` |
| `kv_quant_sparse_flash_attention_golden.py` | CPU 计算 | `aclnn_op_func_cpu`, `generate_and_save_testdata` |
| `check_valid_param.py` | 参数校验 | `check_valid_param` |

## 依赖关系（重构后）

```
utils.py (无循环依赖，顶层公共模块)
    ├── 被 generate_tensor_data.py 导入
    ├── 被 kv_quant_sparse_flash_attention_golden.py 导入
    └── 被 batch/kv_quant_sparse_flash_attention_process.py 导入
```

**重构关键点**: 
- 所有公共函数（`get_np_dtype`, `print_log` 等）统一在 `utils.py` 定义
- 其他模块通过 `import utils` + `utils.xxx()` 方式调用
- 消除了原有的循环依赖问题

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
| `test_kv_quant_sparse_flash_attention_single.py` | 单用例执行验证 |
| `test_kv_quant_sparse_flash_attention_batch.py` | 批量用例执行验证 |

## 快速命令

```bash
# 运行所有黑盒测试
pytest -v test_blackbox_contract.py test_blackbox_runtime.py test_blackbox_snapshot.py

# 运行单个测试文件
pytest -v test_blackbox_contract.py

# 检查导入是否正常
python3 -c "import utils; import generate_tensor_data; import kv_quant_sparse_flash_attention_golden"

# 检查函数唯一性
grep -r "^def get_np_dtype" --include="*.py" . | wc -l  # 应为 1
```

## 当前状态

- ✅ 重构完成：`tools.py` 功能已迁移到 `utils.py`
- ✅ 循环依赖已消除
- ✅ 所有测试用例通过（21/21）
- ✅ 函数定义已统一（无重复）

## 下一步可能的工作

1. 添加更多测试用例覆盖边界场景
2. 完善 NPU 执行流程测试
3. 添加性能基准测试