# Validation 文件生成规则

生成 `validation/` 目录下的四个文件。核心原则：验证算子用户可见语义，允许为 msopgen/pybind 限制做接口桥接，但不通过放宽 golden、跳过 case、改 PyTorch 参考语义掩盖 kernel 问题。

## 生成顺序

四个文件有强依赖关系，**必须按以下顺序生成**（修改任何文件后需同步检查相关联文件）：

1. **pybind.cpp** — 参数类型由 OpDef 属性类型决定（先理解 answer 的 OpDef）
2. **module.py** — `ModelNew.forward()` 签名须与 pybind.cpp 一致
3. **prepare_inputs.py** — 返回值须与 Model/ModelNew 的 forward 签名匹配
4. **test_cases.csv** — 列名须与 `prepare_inputs.py` 中 `param['字段名']` 一致

## 1. pybind.cpp

### 1.1 核心规则

- **始终使用占位符** `aclnnCustomOp` 和 `custom_pybind_api`——框架在编译时自动渲染：
  - `aclnnCustomOp` → `aclnn<OpName>`
  - `custom_pybind_api` → `<kernel_name>`（snake_case）
- **EXEC_NPU_CMD 参数顺序**：输入张量 → 属性参数 → **输出张量必须放最后**
- **输出张量**必须事先创建（用 `torch::empty_like(input)` 或 `torch::zeros(shape)`）
- 输入/输出顺序与 OpDef 中 `Input()`/`Output()` 声明顺序匹配

### 1.2 模板

```cpp
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

// 多输入/多输出示例（根据实际算子调整参数数量）：
at::Tensor custom_pybind_api(
    at::Tensor &input1,
    at::Tensor &input2,
    int64_t attr1,
    float attr2)
{
    at::Tensor result = torch::empty_like(input1);
    // 参数顺序：输入张量 → 属性参数 → 输出张量（必须在最后）
    EXEC_NPU_CMD(aclnnCustomOp, input1, input2, attr1, attr2, result);
    return result;
}

// 多输出示例：
// std::vector<at::Tensor> custom_pybind_api(
//     at::Tensor &input1,
//     at::Tensor &input2)
// {
//     at::Tensor output1 = torch::empty_like(input1);
//     at::Tensor output2 = torch::empty_like(input2);
//     // 输出张量必须在最后
//     EXEC_NPU_CMD(aclnnCustomOp, input1, input2, output1, output2);
//     return {output1, output2};
// }

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
```

### 1.3 OpDef 属性类型 → C++ 类型映射

| OpDef 类型 | C++ 类型 | 注意事项 |
|:---|:---|:---|
| `Int` | `int64_t` | |
| `Float` | `float`（必须与 autogen 头一致） | 查看 autogen 头的 `GetWorkspaceSize` 签名确认是 `float` 还是 `double` |
| `Bool` | `bool` | |
| `String` | `char*` 传入 `EXEC_NPU_CMD` | 使用 `const_cast<char*>(str.c_str())`，autogen 通常期望 `char*` 而非 `const char*` |
| `ListInt` | `at::IntArrayRef` | |
| `ListFloat` | **不存在桥接，必须拆分为独立 `Float` 属性** | 见第5节 |
| `ListBool` | `at::IntArrayRef` | 传入 int8 数组 |

### 1.4 标量类型必须与 autogen 匹配

pybind.cpp 中标量参数的类型必须与 autogen 生成的 `aclnn<OpName>GetWorkspaceSize` 函数签名中对应参数类型完全一致。

检查方法：
```bash
find runs -path "*<OpName>*" -name "aclnn_*.h" -exec head -60 {} \;
# 查看 GetWorkspaceSize 签名的参数类型
```

### 1.5 NPU 张量访问规则

`torch::Tensor.accessor<T, N>()` 仅适用于 CPU 张量。如果需从 NPU 张量读取元数据，必须先转到 CPU：
```cpp
auto size_cpu = size.to(torch::kCPU).contiguous();
auto size_acc = size_cpu.accessor<int32_t, 1>();
int64_t out_h = size_acc[0];
```

### 1.6 OPTIONAL 输入处理

**方式一（推荐）：`c10::optional<at::Tensor>`**
```cpp
at::Tensor custom_pybind_api(
    at::Tensor &required_input,
    c10::optional<at::Tensor> optional_input)
{
    EXEC_NPU_CMD(aclnnCustomOp, required_input, optional_input, result);
}
```

**方式二：裸 `at::Tensor` + `.defined()` 判断**
当 autogen API 不接受 `c10::optional` 时使用：
```cpp
at::Tensor custom_pybind_api(
    at::Tensor &required_input,
    at::Tensor optional_input)
{
    if (optional_input.defined()) {
        EXEC_NPU_CMD(aclnnCustomOp, required_input, optional_input, result);
    } else {
        // dummy 张量的 dtype 应与 OpDef 中该 OPTIONAL 输入的 DataType 声明一致
        at::Tensor dummy = torch::zeros({1}, torch::kFloat32);  // 根据实际 OpDef 调整 dtype
        EXEC_NPU_CMD(aclnnCustomOp, required_input, dummy, result);
    }
}
```

### 1.7 布局转换（NHWC ← → NCHW）

若算子内部用通道最后布局，在 pybind.cpp 中完成 permute：
```cpp
// NCHW → NHWC
input = input.permute({0, 2, 3, 1}).contiguous();
// ... EXEC_NPU_CMD ...
result = result.permute({0, 3, 1, 2}).contiguous();
```

---

**推导参数**是指仅从用户可见参数计算得到的中间值（如从输入 shape 计算的 reshape 维度、tiling 策略选择标志），在 `ModelNew.forward()` 内部计算后传入 `kernel_gen_ops`。它们不出现在 `test_cases.csv` 或 `get_inputs()` 中。

## 2. module.py

### 2.1 模板

```python
import torch
import torch_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch 原生算子参考实现。必须复制算子的精度提升逻辑。"""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, y, attr1, attr2):
        # 使用 PyTorch 原生 API 实现相同语义
        # 包含精度提升逻辑：如 int16→float32→计算→转回
        return torch.add(x, y)


class ModelNew(nn.Module):
    """自定义 kernel 版本。与 Model 相同签名。"""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, y, attr1, attr2):
        # 推导参数在此计算后传入
        result = kernel_gen_ops.<kernel_name>(x, y, attr1, attr2)
        return result
```

### 2.2 关键规则

- `Model.forward()` 和 `ModelNew.forward()` **签名必须相同**
- **推导参数**在 `ModelNew.forward()` 中计算，不暴露给 `get_inputs()`
- `Model` 必须复制算子的精度提升逻辑
- 调用 `kernel_gen_ops.<kernel_name>(...)`

### 2.3 NC1HWC0 格式转换

**OpDef 侧**：`FORMAT_NC1HWC0` 在 msopgen 中替换为 `FORMAT_ND`。

**Python 侧**：严禁使用 `npu_format_cast`，必须手动 reshape+permute：
```python
B, C, H, W = x.shape
C1, C0 = C // 16, 16
x_5d = x.reshape(B, C1, C0, H, W).permute(0, 1, 3, 4, 2).unsqueeze(3)
# → (B, C1, H, W, 1, C0)
result = kernel_gen_ops.<kernel_name>(x_5d, ...)
result = result.squeeze(3).permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
```

---

## 3. prepare_inputs.py

### 3.1 模板

```python
import torch


def get_init_inputs(param, device=None):
    """返回 Model(*init_args) 和 ModelNew(*init_args) 的初始化参数。"""
    return []


def get_inputs(param, device=None):
    """
    根据 test_cases.csv 的一行生成输入张量。
    返回值与 Model.forward() 和 ModelNew.forward() 签名匹配。
    不含推导参数——推导参数由 ModelNew.forward() 内部计算。
    """
    shape = eval(param['shape'])
    dtype = getattr(torch, param['dtype'])
    device = device or "cpu"

    x = torch.randn(shape, dtype=dtype, device=device)
    return (x,)


def custom_check_precision(param, ref_output, test_output):
    """可选。若存在，框架自动发现并优先使用它替代默认逐元素误差检查。
    返回 (is_pass: bool, abs_diff: torch.Tensor, rel_diff: torch.Tensor)"""
    abs_diff = torch.abs(ref_output - test_output)
    rel_diff = abs_diff / (torch.abs(ref_output) + 1e-7)
    # ... 判断逻辑 ...
    return is_pass, abs_diff.reshape(-1), rel_diff.reshape(-1)
```

### 3.2 规则

- **严禁生成违反算子约束的数据**（sqrt 负值、gather 无效索引等）
- `get_inputs()` 只返回用户可见参数，不含推导参数
- CSV bool 字段兼容处理：
  ```python
  ac = param['align_corners']
  align_corners = bool(ac) if not isinstance(ac, str) else (ac.lower() == 'true')
  ```
- `custom_check_precision` 返回的 `abs_diff` 和 `rel_diff` 必须为 `torch.Tensor` 类型，不能是 Python `float`

### 3.3 精度容差参考

| 数据类型 | RTOL | 说明 |
|:---|:---|:---|
| 整型 | 0 | 必须精确匹配 |
| float16 | 1/512 | ~0.00195 |
| bfloat16 | 1/256 + 1/16384 | ~0.004 |
| float32 | 1/2048 + 1/16384 | ~0.0005 |
| complex64 | 1/2048 + 1/16384 | ~0.0005 |

---

## 4. test_cases.csv

### 4.1 格式

- **分号 `;` 分隔**
- **首列必须为 `case_id`，从 0 开始**
- 后续列名为算子参数，**列名必须与 `prepare_inputs.py` 中 `get_inputs(param, device)` 通过 `param['字段名']` 访问的字段名一致**（如 shape、dtype 等）

### 4.2 覆盖要求

- **dtype 覆盖**：算子支持的所有 dtype
- **shape 覆盖**：小（<100 元素）、中（100-10K）、大（>10K）
- **边界条件**：标量（shape=[]）、1D、高维、对齐/非对齐
- **每个显式分支至少 1 个 case**

### 4.3 示例

```csv
case_id;shape;dtype
0;[2, 4];float32
1;[16, 64, 128];float16
2;[1, 1];float32
3;[64, 256, 512];bfloat16
```

---

## 5. 专项：ListFloat 属性拆分

**判断条件**：OpDef 中有 `.ListFloat()`

**原因**：`EXEC_NPU_CMD` 底层无 `std::vector<float>` → `aclFloatArray*` 转换。

**_def.cpp 变更**：拆分为独立 `.Float()` 属性。

**_tiling.cpp 变更**：
```cpp
// 改前
const gert::ContinuousVector *arr = attrs->GetAttrPointer<gert::ContinuousVector>(IDX);
const float *data = reinterpret_cast<const float *>(arr->GetData());
float v = data[0];

// 改后
const float *ptr = attrs->GetAttrPointer<float>(IDX);
float v = (ptr != nullptr) ? *ptr : 0.0f;
```

**pybind.cpp 变更**：`std::vector<float>` → 独立的 `float` 参数。
**module.py 变更**：`ModelNew.forward()` 中拆分列表为独立参数。

---

## 6. 专项：ListInt 负值默认值

当 OpDef 有 `.AttrType(OPTIONAL).ListInt({-1, ...})` 且 tiling 对负值做 offset 处理时，`prepare_inputs.py` 中将负值替换为有效范围：

```python
if range_start < 0 or range_end < 0:
    range_vals = [0, total_size]
```

---

## 7. tiling_cases.csv（可选）

如果算子有 `validation/tiling_cases.csv`，它是 `test_cases.csv` 的子集，用于 tiling key 覆盖率测试。迁移时可以从 `test_cases.csv` 中挑选覆盖每个 tiling key 的一个代表性 case 来生成。

---

## 8. Validation 检查清单

- [ ] `pybind.cpp` 使用占位符 `aclnnCustomOp` 和 `custom_pybind_api`
- [ ] `pybind.cpp` EXEC_NPU_CMD 输出张量在最后
- [ ] 标量参数类型与 autogen 头一致
- [ ] ListFloat 属性已拆分
- [ ] `module.py` 中两个 Model 的 forward 签名相同
- [ ] `module.py` 调用 `kernel_gen_ops.<kernel_name>()`
- [ ] `Model` 复制了精度提升逻辑
- [ ] `prepare_inputs.py` 不生成违反算子约束的数据
- [ ] `test_cases.csv` 覆盖多种 dtype、shape、边界条件
