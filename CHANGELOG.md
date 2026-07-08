Install and configure oh-my-opencode by following the instructions here:
https://raw.githubusercontent.com/code-yeongyu/oh-my-openagent/refs/heads/dev/docs/guide/installation.md
# TTK 算子适配教程

ttk蓝区仓：https://gitcode.com/cann/ops-test-kit

---

## 一、模式概览与执行命令

### 1.1 torch_npu 直调（E2E）

```bash
python3 -m ttk e2e \
  -i cases.csv \
  --backend npu \
  --plugin ops-transformer/attention/{op_name}/tests/assets \
  -o result.csv
```

### 1.2 入图（Graph）模式

在 E2E 命令基础上加 `-c`（静态图）或 `-d`（动态图）。`-c` / `-d` 开启时，TTK 会在 eager 执行之后，额外走 `torch.compile` 编译 + 执行 + 比对。

**不需要定义 `torch_graph` 也能跑入图**：框架会自动用 `GraphNetwork` 包裹 API 函数走 `torch.compile`。只有当默认图捕获不满足需求时，才需要在 spec.py 中定义自定义 `torch_graph`（`torch.nn.Module` 子类）。

```bash
# 静态图
python3 -m ttk e2e -i cases.csv --backend npu --plugin tests/assets -o result.csv -c
# 动态图
python3 -m ttk e2e -i cases.csv --backend npu --plugin tests/assets -o result.csv -d
```

### 1.3 ACLNN 模式

```bash
python3 -m ttk aclnn \
  -i cases_aclnn.csv \
  --plugin ops-transformer/attention/{op_name}/tests/assets \
  -o result_aclnn.csv
```

> **注意**：aclnn 模式暂未接入 TestSpec 体系，仍使用传统的 `assets/golden.py` + `assets/inputs.py` 中 `__golden__` / `__input__` 字典注册。后期商议是否接入 TestSpec 体系

### 1.4 常用参数说明

| 参数                                           | 说明                                             |
| ---------------------------------------------- | ------------------------------------------------ |
| `-i`                                           | 输入 CSV 文件路径                                |
| `-o`                                           | 输出结果 CSV 文件路径                            |
| `--backend`                                    | 后端，NPU 测试固定`npu`                          |
| `--plugin`                                     | 插件目录，指向算子的`tests/assets`               |
| `-c`                                           | 开启静态 shape 图模式                            |
| `-d`                                           | 开启动态 shape 图模式                            |
| `-t case_name`                                 | 只跑指定用例（逗号分隔多个）                     |
| `--compare`                                    | 比较器，默认`close`                              |
| `--dev N`                                      | 使用 N 张 NPU                                    |
| `--pc N`                                       | 每张 NPU 启动 N 个 worker 进程                   |
| `--print false`                                | 关闭周期性汇总打印，减少日志噪音                 |
| `--task-prof false`                            | 关闭任务级耗时统计                               |
| `--single-log`                                 | 每条 case 单独日志文件                           |
| `--proc-no-reuse`                              | 每条 case 后重启执行进程，避免 NPU 状态污染      |
| `--dump full --dump-format npy --dump-on-fail` | 失败时 dump 全量 input/output/golden 到 npy 文件 |

### 1.5 失败用例 dump 定位

单条复跑 + dump 命令模板：

```bash
python3 -m ttk e2e \
  -i cases.csv \
  -t "FAILED_CASE_NAME" \
  --backend npu \
  --plugin tests/assets \
  -o /tmp/failed_result.csv \
  --print true --task-prof false \
  --single-log --proc-no-reuse \
  --dump full --dump-format npy --dump-on-fail
```

dump 产物：

- `{case_name}_input_{idx}.npy` — NPU 输入
- `{case_name}_output_{idx}.npy` — NPU 输出
- `{case_name}_golden_{idx}.npy` — CPU golden 输出

常见失败类型定位：

- `PARAM_PLAN_FAILURE` → API 签名匹配失败，检查 CSV `api_name` 和 `tensor_view_shapes` 数量
- `INPUT_GEN_FAILURE` → 默认输入生成或 input plugin 异常
- `GOLDEN_FAILURE` → golden 计算阶段异常，优先检查 golden 函数的能力边界
- `COMPARE_FAILURE` → compare 阶段自身异常（不是精度不达标）
- `precision_status=FAIL` → 数值精度不达标，用 dump 的 npy 离线复算定位

---

## 二、目录结构与文件作用

```
ops-transformer/attention/{op_name}/tests/
├── assets/                        # TTK 插件目录（--plugin 指向这里）
│   ├── spec.py                    # TestSpec 入口（torch_npu 直调 + 入图）
│   ├── golden.py                  # 传统 golden 入口（aclnn 模式）
│   ├── inputs.py                  # 传统 input 入口（aclnn 模式）
│   └── impl/                      # 算子独立公共模块
│       ├── golden.py              # 算子 CPU golden 实现（不绑定任何框架）
│       ├── inputs.py              # customize_inputs 实现
│       ├── graph.py               # 入图 torch.nn.Module 定义（可选）
│       └── compare.py             # 自定义 compare 逻辑（可选）
├── pytest/                        # Pytest 测试
└── ...
```

> **说明**：`assets/` 下应该是算子独立的 golden、compare等模块的参考实现，**不依赖任何框架**。当前因一些开发排期原因暂时可以考虑从 pytest 加载公式，后续应逐步迁移为独立的公共模块。

---

## 三、CSV 用例字段详解

### 3.1 E2E（torch_npu 直调）CSV 字段

| 字段                    | 类型         | 必填 | 说明                                             | 示例                                     |
| ----------------------- | ------------ | :--: | ------------------------------------------------ | ---------------------------------------- |
| `testcase_name`         | string       |  ✅  | 用例名称，唯一标识                               | `my_op_BSND_BF16_1_4_1_4_5_16`           |
| `api_name`              | string       |  ✅  | torch_npu API 全名                               | `torch_npu.npu_my_op`                    |
| `tensor_view_shapes`    | nested tuple |  ✅  | 每个 tensor 的 view shape，按 API 签名顺序       | `((1,4,1,16),(1,5,1,16),...)`            |
| `tensor_dtypes`         | nested tuple |  ✅  | 每个 tensor 的 dtype 字符串                      | `('bfloat16','bfloat16',...)`            |
| `attributes`            | dict         |  否  | 算子属性，含 layout/sparse_mode 等非 tensor 参数 | `{'layout':'BSND','softmax_scale':0.25}` |
| `input_data_ranges`     | nested tuple |  否  | 每个 tensor 的随机值范围`(low, high)`            | `((-1,1),(-1,1),...)`                    |
| `precision_tolerances`  | nested tuple |  否  | `(rtol, ptol)` 精度容忍度，单个值广播到所有输出  | `(0.005, 0.005)`                         |
| `absolute_precision`    | float/tuple  |  否  | atol 绝对精度，默认`1e-8`                        | `0.0001`                                 |
| `is_enabled`            | bool         |  否  | 是否启用该用例，默认`True`                       | `True`                                   |
| `soc_series`            | string       |  否  | 限定芯片系列                                     | `Ascend950`                              |
| `golden_api`            | string       |  否  | 覆盖 golden 使用的 CPU API                       | `""`                                     |
| `tensor_formats`        | nested tuple |  否  | tensor format（如 ND/NZ），默认空                | `()`                                     |
| `output_tensor_indexes` | tuple        |  否  | 哪些顶层 tensor 是`out=` 输出容器                | `()`                                     |

**optional tensor 占位规则**：optional tensor 不传但后续 tensor 要传时，必须用 `None` 占位：

```python
# query, key, value, pse_shift(None), mask
tensor_view_shapes = ((1,4,16), (1,5,16), (1,5,16), None, (1,1,4,5))
tensor_dtypes = ('float16', 'float16', 'float16', None, 'bool')
```

### 3.2 ACLNN CSV 字段

| 字段                   | 类型         | 必填 | 说明            |
| ---------------------- | ------------ | :--: | --------------- |
| `testcase_name`        | string       |  ✅  | 用例名称        |
| `api_name`             | string       |  ✅  | `aclnnXxx` 格式 |
| `tensor_view_shapes`   | nested tuple |  ✅  | 同 E2E          |
| `tensor_dtypes`        | nested tuple |  ✅  | 同 E2E          |
| `attributes`           | dict         |  否  | 同 E2E          |
| `input_data_ranges`    | nested tuple |  否  | 同 E2E          |
| `precision_tolerances` | nested tuple |  否  | 同 E2E          |
| `absolute_precision`   | float/tuple  |  否  | 同 E2E          |
| `is_enabled`           | bool         |  否  | 同 E2E          |
| `soc_series`           | string       |  否  | 同 E2E          |

---

## 四、环境要求

- 已安装 CANN 环境的 NPU 机器
- Python 环境已安装 `torch_npu` 包

```bash
export PYTHONPATH=/path/ops-test-kit:$PYTHONPATH
```

---

## 五、torch_npu 直调（E2E）实现

### 5.1 spec.py — TestSpec 入口

```python
# ops-transformer/attention/{op_name}/tests/assets/spec.py

# 从 impl/ 加载各模块
# golden_module  = load_impl_module("golden")  # impl/golden.py
# inputs_module  = load_impl_module("inputs")  # impl/inputs.py
# graph_module   = load_impl_module("graph")   # impl/graph.py（入图可选）
# compare_module = load_impl_module("compare") # impl/compare.py（可选）


class MyOpSpec:
    """算子测试规范"""

    # 必选：CPU golden 参考实现
    golden = golden_module.cpu_my_op

    # 可选：定制输入修正
    customize_inputs = inputs_module.generate_my_op_inputs

    # 可选：精度标准（当前仅元数据层，框架默认 compare 尚未自动消费）
    tolerance = {
        "float16": {"standard": "IsClose", "rtol": 0.005, "ptol": 0.005, "atol": 0.000025},
        "bfloat16": {"standard": "IsClose", "rtol": 0.0078125, "ptol": 0.005, "atol": 0.0001},
    }

    # 可选：自定义 compare（如 TopK 语义，返回 dict 或 list[dict]）
    # def compare(*outputs, **kwargs):
    #     return compare_module.compare(*outputs, ...)

    # 可选：入图 torch.nn.Module（见第六章，从 impl/graph.py 加载）
    # torch_graph = graph_module.MyOpGraphModule

    # 也可保留传统 __golden__ 映射，辅助兼容：
    # __golden__ = {"e2e": {"torch_npu.npu_my_op": "cpu_my_op"}}


__spec__ = {
    "torch_npu.npu_my_op": MyOpSpec,
}
```

**关键点**：

- `__spec__` 的 key 是完整的 `torch_npu.npu_xxx` API 名
- `golden` 指向 `impl/golden.py` 中的 CPU golden 函数
- golden 函数签名（参数名和顺序）需与 torch_npu API 保持一致，TTK 按 `ParamPlan.build_args()` 组装参数传入
- 不定义 `compare` 时走框架默认 close 比较器
- 传统 `__golden__` 映射可以保留在 spec.py 中，用于兼容 `get_plugin_function` 的降级查找

### 5.2 impl/golden.py — CPU Golden 参考实现

Golden 是算子独立的 CPU 参考实现模块

```python
# ops-transformer/attention/{op_name}/tests/assets/impl/golden.py

import torch
import numpy as np


# 如果当前暂未完成独立公式，可以临时从 pytest 加载
# ...


def cpu_my_op(query, key, value, *,
              layout="BSND", softmax_scale=None, **kwargs):
    """
    CPU 参考实现 for torch_npu.npu_my_op。

    函数参数签名与 torch_npu API 保持一致（参数名和顺序）。
    TTK 通过 ParamPlan.build_args() 传入 tensor 和 attributes。

    Args:
        query:  torch tensor，shape 由 CSV tensor_view_shapes 定义
        key:    torch tensor
        value:  torch tensor
        layout: str, 输入布局
        softmax_scale: float, softmax 缩放因子

    Returns:
        torch tensor 或 tuple of torch tensors（与 API 输出一致）
    """

    # 将 tensor 转 CPU（TTK 传入的 tensor 可能在 NPU 上）
    query_cpu = query.detach().cpu().float() if torch.is_tensor(query) else query
    key_cpu = key.detach().cpu().float() if torch.is_tensor(key) else key

    …………

    # 返回与 API 一致的输出
    return output.to(query.dtype)
```

**关键结构**：

- 函数命名建议前缀 `cpu_`，如 `cpu_my_op`，与 NPU 侧 `npu_xxx` 区分
- 函数参数签名与 torch_npu API 一致（TTK 按 ParamPlan 传参）
- 接收的 tensor 可能是 NPU device tensor，需要先 `.detach().cpu()`
- 返回 torch tensor（或 tuple），与 API 输出签名一致

### 5.3 impl/inputs.py — 定制输入修正

```python
# ops-transformer/attention/{op_name}/tests/assets/impl/inputs.py

import torch


def _fill_tensor_from_value(tensor, value):
    """用 Python 值覆盖 tensor 内容"""
    if tensor is None or value is None or not torch.is_tensor(tensor):
        return
    data = torch.tensor(value, dtype=tensor.dtype, device=tensor.device)
    tensor.copy_(data.reshape(tensor.shape))


def generate_my_op_inputs(query, key, value, *,
                           actual_seq_lengths_query=None,
                           actual_seq_lengths_kv=None,
                           layout="BSND", **kwargs):
    """
    定制输入修正。

    TTK 已生成随机 tensor，此函数做原地确定性修正：
    1. 用 attributes 中的真实值覆盖 shape tensor
    2. 补充算子特有的语义约束（如 MLA-absorb V=K、scale 填充等）
    3. 缩放随机值到稳定范围，避免 softmax 饱和
    """

    # ① 覆盖 shape tensor（actual_seq 等确定性元数据）
    _fill_tensor_from_value(actual_seq_lengths_query,
                            kwargs.get('actual_seq_lengths_query'))
    _fill_tensor_from_value(actual_seq_lengths_kv,
                            kwargs.get('actual_seq_lengths_kv'))

    # ② 缩放输入到稳定范围（避免极端值导致 softmax 饱和）
    if query is not None and query.is_floating_point():
        query.mul_(0.125)
    if key is not None and key.is_floating_point():
        key.mul_(0.125)

    # ③ 算子特有约束（按需要实现）
    # 例如：PA 场景生成 block_table
    # layout_kv = kwargs.get('layout_kv', layout)
    # if layout_kv == "PA_BSND":
    #     _fill_random_block_table(...)
```

**关键点**：

- 函数签名与 API 签名一致（TTK 按 ParamPlan 组装参数传入）
- **原地修改** tensor（`tensor[:] = ...` / `tensor.copy_(...)` / `tensor.mul_(...)`）
- 不要返回任何值（返回值会被忽略）
- 同一份 tensor 被 device 和 golden 共用，保证双侧输入一致

### 5.4 CSV 用例示例

```csv
testcase_name,api_name,tensor_view_shapes,tensor_dtypes,attributes,input_data_ranges,precision_tolerances,absolute_precision,is_enabled
my_op_BSND_BF16_1_4_1_4_5_16,torch_npu.npu_my_op,"((1,4,1,16),(1,5,1,16),(1,5,1,16))","('bfloat16','bfloat16','bfloat16')","{'layout':'BSND','softmax_scale':0.25}","((-1,1),(-1,1),(-1,1))","(0.005,0.005)",0.0001,True
```

---

## 六、入图（Graph）模式实现

### 6.1 默认方式：不写代码即可跑

不定义 `torch_graph` 时，TTK 自动用 `GraphNetwork` 包裹 API 函数：

```python
# 框架内部等价于：
class GraphNetwork(torch.nn.Module):
    def __init__(self, api_func):
        super().__init__()
        self.api_func = api_func
    def forward(self, *args, **kwargs):
        return self.api_func(*args, **kwargs)
```

所有参数打平进 `forward`，`torch.compile` 统一捕获。大多数算子默认方式即可跑通。

### 6.2 自定义 torch_graph（impl/graph.py）

在 `impl/graph.py` 中定义 `torch.nn.Module` 子类，`__init__` 放非 tensor 参数，`forward` 只接收 tensor：

**impl/graph.py：**

```python
# ops-transformer/attention/{op_name}/tests/assets/impl/graph.py

import torch
import torch_npu


class MyOpGraphModule(torch.nn.Module):
    """自定义图模块：torch.nn.Module 子类"""

    def __init__(self, layout="BSND", softmax_scale=None):
        super().__init__()
        self.layout = layout
        self.softmax_scale = softmax_scale

    def forward(self, query, key, value, *optional_tensors):
        return torch_npu.npu_my_op(
            query, key, value,
            *optional_tensors,
            layout=self.layout,
            softmax_scale=self.softmax_scale,
        )
```

**关键点**：

- `torch_graph` 必须是 `torch.nn.Module` 子类（框架强校验，非 Module 子类会抛 `InvalidSpecError`）
- `__init__` 参数名需与 API 关键字参数名一致，框架通过 `split_params` 按签名匹配拆分
- 图模式完全复用 spec.py 的 golden / customize_inputs / compare，不需要额外实现
- 不定义 `torch_graph` 时默认 `GraphNetwork` 够用，大多数算子不需要自定义

---

## 七、ACLNN 模式实现

ACLNN 暂未接入 TestSpec 体系，使用传统 `assets/golden.py` + `assets/inputs.py` 中 `__golden__` / `__input__` 字典注册。

### 7.1 assets/golden.py

```python
# ops-transformer/attention/{op_name}/tests/assets/golden.py

import numpy as np

__golden__ = {
    "aclnn": {
        "aclnnMyOp": "aclnnMyOpGolden",
    }
}


def aclnnMyOpGolden(query, key, value, *,
                    layout="BSND", softmax_scale=None, **kwargs):
    """
    ACLNN CPU golden。

    接收 numpy array（不是 torch tensor）。
    返回值与 aclnn API 输出一致。
    """
  
	……
    return output
```

### 7.2 assets/inputs.py

```python
# ops-transformer/attention/{op_name}/tests/assets/inputs.py

import numpy as np

__input__ = {
    "aclnn": {
        "aclnnMyOp": "aclnnMyOpInput",
    }
}


def aclnnMyOpInput(query, key, value, *,
                   layout="BSND", softmax_scale=None, **kwargs):
    """ACLNN 定制输入修正。接收 numpy array，返回修正后的数组。"""
    query = (query * 0.125).astype(query.dtype)
    key = (key * 0.125).astype(key.dtype)
    return query, key, value
```

**关键点**：

- ACLNN 的 golden/input 函数接收 **numpy array**
- 用 `__golden__` / `__input__` 字典注册，key 为 `"aclnn"`，value 为 `{api_name: func_name}` 映射
- input 函数需 **返回修改后的数组**（与 E2E 的原地修改不同）

---
