# QFA MXFP8 Pytest 代码适配 TTK 框架计划

## 1. TTK 框架架构详解

### 1.1 核心架构

TTK (Ops Test Tool Kit) 是华为 CANN 算子测试框架，通过 `python3 -m ttk e2e` 命令执行端到端测试。其核心设计是**插件系统**：算子开发者只需在 `tests/assets/` 目录下提供 4 个插件文件，TTK 框架自动加载并执行测试。

### 1.2 四大插件组件

| 文件           | 位置                             | 注册字典              | 核心职责                                                       |
| ------------ | ------------------------------ | ----------------- | ---------------------------------------------------------- |
| `spec.py`    | `tests/assets/spec.py`         | `__spec__`        | 算子规格描述：定义 golden 函数、input 函数、tolerance、compare 函数          |
| `golden.py`  | `tests/assets/impl/golden.py`  | `__golden__`      | CPU 参考实现：接收 NPU API 格式的输入，返回 CPU golden 输出                 |
| `inputs.py`  | `tests/assets/impl/inputs.py`  | `__input__`       | 输入预处理：对 TTK 生成的随机输入进行合法性修正（如 sparse\_indices、block\_table） |
| `compare.py` | `tests/assets/impl/compare.py` | `__compare__`（可选） | 自定义精度比较：替代 TTK 默认比较逻辑                                      |

### 1.3 spec.py 结构详解

`spec.py` 是 TTK 的入口文件，定义 `__spec__` 注册字典：

```python
import importlib.util
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

def load_impl_module(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"qfa_assets_impl_{stem}_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")

class QuantFlashAttnSpec:
    golden = golden_module.cpu_quant_flash_attn          # CPU golden 函数引用
    customize_inputs = inputs_module.generate_qfa_inputs  # 输入预处理函数引用
    tolerance = {                                          # 精度阈值
        "float16": {"standard": "IsClose", "rtol": 0.005, "ptol": 0.005, "atol": 0.000025},
        "bfloat16": {"standard": "IsClose", "rtol": 0.0078125, "ptol": 0.005, "atol": 0.0001},
    }
    # 可选：自定义 compare 方法
    # def compare(*outputs, **kwargs):
    #     return compare_module.compare(*outputs, ...)

__spec__ = {
    "torch_npu.quant_flash_attn_metadata": QuantFlashAttnSpec,  # 算子 API 名称
}
```

**关键设计点：**

* `__spec__` 字典的 key 是算子的 torch\_npu API 名称（如 `"torch_npu.npu_sparse_flash_attention"`）

* Spec 类的 `golden` 属性直接引用 impl/golden.py 中的函数

* `customize_inputs` 属性引用 impl/inputs.py 中的函数

* `tolerance` 定义每种 dtype 的精度标准

* `compare` 是可选的类方法，用于自定义比较逻辑

### 1.4 golden.py 结构详解

`golden.py` 定义 `__golden__` 注册字典和 CPU golden 函数：

```python
__golden__ = {
    "e2e": {
        "torch_npu.npu_sparse_flash_attention": "cpu_sparse_flash_attention"
    }
}
```

**关键设计模式：**

1. **复用 pytest golden**：PR #8185 的核心模式是 `load_pytest_golden_module()` — 通过 `importlib.util` 动态加载 `tests/pytest/` 下的 golden 文件，避免代码重复
2. **TTK 适配层**：golden.py 中只写 TTK API → pytest golden 的适配代码（layout 转换、PA→BNSD 转换、量化 scale 处理等）
3. **函数签名匹配 NPU API**：golden 函数的参数名和顺序与 NPU 算子 API 一致，TTK 框架会按 CSV 中的参数名自动映射

### 1.5 inputs.py 结构详解

`inputs.py` 定义 `__input__` 注册字典和输入预处理函数：

```python
__input__ = {
    "e2e": {
        "torch_npu.npu_sparse_flash_attention": "generate_valid_sparse_indices"
    }
}
```

**关键职责：**

* 对 TTK 随机生成的输入进行合法性修正（如 sparse\_indices 填充有效值、block\_table 随机填充、量化 scale 初始化）

* 函数签名与 NPU API 一致，TTK 框架传入所有输入张量和属性参数

* 使用 `**kwargs` 接收额外参数

* 直接修改传入的张量（in-place 操作），不需要返回值

### 1.6 compare.py 结构详解（可选）

`compare.py` 定义自定义比较逻辑，用于替代 TTK 默认的 IsClose 比较：

```python
def compare(*outputs, scores=None):
    # outputs = [npu_out1, npu_out2, ..., golden_out1, golden_out2, ...]
    # 前半是 NPU 输出，后半是 golden 输出
    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]
    ...
```

**适用场景：**

* TopK 索引比较（LI/QLI 算子）

* Attention 输出需要按 valid token 区域比较

* 不同输出需要不同精度阈值

### 1.7 数据流

```
CSV 测试用例 → TTK 解析 → 生成随机输入张量 → inputs.py 预处理 →
    → NPU 执行（e2e 模式） → NPU 输出
    → golden.py CPU 计算 → CPU golden 输出
    → compare.py / TTK 默认比较 → 结果 CSV
```

### 1.8 E2E 模式

PR #8185 中所有注册字典都使用 `"e2e"` 子字典（而非 `"kernel"` 或 `"aclnn"`），说明这是端到端测试模式：

```bash
python3 -m ttk e2e -i ${test_csv} -o ${result_csv} --plugin ${ops_test_path} -c --pc=8
```

E2E 模式下 TTK 会：

1. 从 CSV 读取算子名称和参数
2. 调用 `torch_npu.xxx()` 算子 API 执行 NPU 计算
3. 调用 `__spec__` 中注册的 golden 函数计算 CPU 参考
4. 调用 `__spec__` 中注册的 compare 函数或使用 tolerance 进行比较

## 2. 当前状态分析

### 2.1 quant\_flash\_attn 目录结构

```
attention/quant_flash_attn/
├── op_api/          # C++ 算子 API 实现
├── op_host/         # C++ tiling 实现
├── op_kernel/       # C++ kernel 实现
└── tests/
    └── pytest/      # 当前只有 pytest 测试
        └── fia_fullquant_mxfp8_test/
            ├── common/
            │   ├── fia_fullquant_mxfp8_golden.py  # 已重构为双算子
            │   ├── golden_cache.py
            │   ├── perf_parser.py
            │   ├── result_compare_method.py
            │   └── test_runner.py
            ├── conftest.py
            ├── fia_fullquant_mxfp8_paramset_*.py
            ├── test_fia_fullquant_mxfp8_*.py
            └── pytest.ini
```

**缺失：** 没有 `tests/assets/` 目录，没有 TTK 插件文件。

### 2.2 QFA 双算子接口

QFA 是双算子模式，需要两次 NPU 调用：

1. `torch_npu.quant_flash_attn_metadata()` → 返回 metadata tensor
2. `quant_flash_attn()` → 返回 (attn\_out, lse)

**TTK E2E 模式的挑战：** TTK 的 e2e 模式通常只调用单个算子 API。QFA 是双算子，需要特殊处理。

### 2.3 QFA 算子 API 名称

根据 `quant_flash_attn.py` 的实现：

* 第一个算子：`torch_npu.quant_flash_attn_metadata`（通过 `torch_npu.npu_quant_flash_attn_metadata` 注册）

* 第二个算子：`quant_flash_attn`（Python 函数，内部调用 `torch_npu.npu_quant_flash_attn`）

## 3. 适配方案

### 3.1 目录结构

```
attention/quant_flash_attn/tests/
├── pytest/                          # 保留现有 pytest 测试（不修改）
│   └── fia_fullquant_mxfp8_test/
└── assets/                          # 新增 TTK 插件目录
    ├── spec.py                      # 算子规格描述
    └── impl/
        ├── golden.py                # CPU golden 适配层
        ├── inputs.py                # 输入预处理
        └── compare.py              # 自定义精度比较（可选）
```

### 3.2 spec.py

```python
"""TestSpec adapter for QuantFlashAttn assets."""

import importlib.util
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

def load_impl_module(stem):
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(
        f"qfa_assets_impl_{stem}_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

golden_module = load_impl_module("golden")
inputs_module = load_impl_module("inputs")
compare_module = load_impl_module("compare")

class QuantFlashAttnSpec:
    golden = golden_module.cpu_quant_flash_attn
    customize_inputs = inputs_module.generate_qfa_inputs
    tolerance = {
        "float16": {"standard": "IsClose", "rtol": 0.005, "ptol": 0.005, "atol": 0.000025},
        "bfloat16": {"standard": "IsClose", "rtol": 0.0078125, "ptol": 0.005, "atol": 0.0001},
    }

    def compare(*outputs, **kwargs):
        return compare_module.compare(*outputs)

__spec__ = {
    "torch_npu.npu_quant_flash_attn_metadata": QuantFlashAttnSpec,
}
```

**注意：** `__spec__` 的 key 使用 `torch_npu.npu_quant_flash_attn_metadata`，因为这是 QFA 双算子的入口算子。TTK e2e 模式会调用这个算子获取 metadata，然后需要额外逻辑调用 `quant_flash_attn`。

### 3.3 golden.py

**核心设计：复用 pytest golden，只写适配层。**

golden.py 的 `cpu_quant_flash_attn` 函数需要：

1. 接收 TTK 传入的 NPU API 格式参数（与 `quant_flash_attn_metadata` + `quant_flash_attn` 的参数一致）
2. 将参数转换为 pytest golden 的 `Network.forward()` 所需格式
3. 调用 pytest golden 的 CPU 计算逻辑

**关键适配点：**

* TTK 传入的 Q/K/V 是 FP8 量化后的张量 + descale scale

* pytest golden 的 `cpu_mxfp8_golden()` 接收的是 FP32 的 Q/K/V + 量化参数

* 需要在 golden.py 中做 FP8→FP32 反量化，然后调用 pytest golden

**函数签名设计：**

```python
def cpu_quant_flash_attn(
    query, key, value,
    query_descale_scale, key_descale_scale, value_descale_scale,
    quant_mode,
    *,
    cu_seqlens_q=None, cu_seqlens_kv=None,
    seqused_q=None, seqused_kv=None,
    mask_mode=None,
    layout_q=None, layout_kv=None, layout_out=None,
    block_table=None, pa_block_size=None,
    scale=None,
    **kwargs):
```

**适配逻辑：**

1. FP8→FP32 反量化：`q_float = query.float() * query_descale_scale`
2. Layout 转换：TTK 传入的 layout 字符串 → pytest golden 的 layout 格式
3. 序列长度转换：TTK 的 `cu_seqlens_q`/`seqused_q` → pytest golden 的 `actual_seq_lengths_q`
4. PA 模式：TTK 的 PA KV cache → BNSD 格式
5. 调用 pytest golden 的 `cpu_mxfp8_golden()` 或 `Network.forward()`

### 3.4 inputs.py

**核心职责：对 TTK 随机生成的输入进行合法性修正。**

```python
__input__ = {
    "e2e": {
        "torch_npu.npu_quant_flash_attn_metadata": "generate_qfa_inputs"
    }
}

def generate_qfa_inputs(
    query, key, value,
    query_descale_scale, key_descale_scale, value_descale_scale,
    quant_mode,
    *,
    cu_seqlens_q=None, cu_seqlens_kv=None,
    seqused_q=None, seqused_kv=None,
    mask_mode=None,
    layout_q=None, layout_kv=None, layout_out=None,
    block_table=None, pa_block_size=None,
    scale=None,
    **kwargs):
    # 1. 填充 cu_seqlens / seqused
    # 2. PA 模式：随机填充 block_table
    # 3. 量化 scale 初始化（避免全零）
    # 4. FP8 张量范围修正
```

### 3.5 compare.py

**QFA 输出是 (attn\_out, lse)，需要分别比较：**

```python
def compare(*outputs, **kwargs):
    # outputs = [npu_attn_out, npu_lse, golden_attn_out, golden_lse]
    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]

    # attn_out: 使用 IsClose 比较，考虑 valid token 区域
    # lse: 使用更宽松的阈值比较
    attn_result = attn_compare(npu_outputs[0], golden_outputs[0])
    lse_result = lse_compare(npu_outputs[1], golden_outputs[1])
    return [attn_result, lse_result]
```

### 3.6 双算子 E2E 模式的特殊处理

**问题：** TTK e2e 模式通常只调用单个算子，但 QFA 需要双算子调用。

**解决方案（参考 PR #8185 模式）：**

PR #8185 中所有算子都是单算子 API（`npu_sparse_flash_attention`、`npu_lightning_indexer` 等）。QFA 的双算子模式需要特殊处理：

**方案 A：在 golden.py 中模拟双算子流程**

* golden 函数内部先计算 metadata（CPU 模拟 tiling），再计算 attention

* TTK 只需调用一次 golden 函数，golden 内部完成双算子的完整流程

* `__spec__` 注册在 `torch_npu.npu_quant_flash_attn_metadata` 上

**方案 B：TTK 调用 Python 封装函数**

* 在 `quant_flash_attn.py` 中已有 `quant_flash_attn()` Python 函数，内部调用双算子

* TTK e2e 模式可以直接调用这个 Python 函数

* `__spec__` 注册在 Python 函数名上

**推荐方案 A**，因为：

1. 与 PR #8185 的模式一致（golden 函数内部完成完整计算流程）
2. TTK e2e 模式下 NPU 执行会调用 `quant_flash_attn_metadata` + `quant_flash_attn` 双算子
3. golden 函数只需返回最终结果 (attn\_out, lse)

## 4. 实施步骤

### Step 1: 创建目录结构

* 创建 `attention/quant_flash_attn/tests/assets/`

* 创建 `attention/quant_flash_attn/tests/assets/impl/`

### Step 2: 编写 spec.py

* 定义 `QuantFlashAttnSpec` 类

* 注册 `__spec__` 字典

* 加载 impl 模块

### Step 3: 编写 golden.py

* 实现 `load_pytest_golden_module()` 加载 `fia_fullquant_mxfp8_golden.py`

* 实现 FP8→FP32 反量化适配

* 实现 layout/序列长度/PA 转换适配

* 实现 `cpu_quant_flash_attn()` golden 函数

* 注册 `__golden__` 字典

### Step 4: 编写 inputs.py

* 实现 `generate_qfa_inputs()` 输入预处理函数

* 处理 cu\_seqlens/seqused 填充

* 处理 block\_table 随机填充

* 处理量化 scale 初始化

* 注册 `__input__` 字典

### Step 5: 编写 compare.py（可选）

* 实现 attn\_out 和 lse 的分别比较

* 处理 valid token 区域比较

* 处理不同精度阈值

### Step 6: 验证

* 确保 TTK 能正确加载插件

* 确保 golden 函数输出与 pytest golden 一致

* 确保 inputs 预处理后的输入合法

## 5. 关键决策与假设

1. **复用 pytest golden**：遵循 PR #8185 模式，golden.py 只写适配层，核心计算逻辑复用 `fia_fullquant_mxfp8_golden.py`
2. **E2E 模式**：使用 `"e2e"` 子字典注册，与 PR #8185 一致
3. **双算子处理**：golden 函数内部模拟完整双算子流程，TTK 只调用一次 golden
4. **compare.py**：QFA 输出有 attn\_out 和 lse 两个输出，需要自定义 compare 分别处理
5. **不修改 pytest 测试**：TTK 插件与 pytest 测试并存，互不影响
6. **FP8 反量化**：TTK 传入 FP8 张量，golden.py 需先反量化再调用 pytest golden

