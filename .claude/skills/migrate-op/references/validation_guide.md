# 验证体系代码生成规范

本文档覆盖 validation/ 目录下所有文件的生成规则：`pybind.cpp`、`module.py`、`prepare_inputs.py`、`test_cases.csv`。
同时包含 `ListFloat` 属性拆分、标量类型匹配等跨文件专项规则。

---

## 1. pybind.cpp

### 1.1 模板

```cpp
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(at::Tensor &input1, ...)
{
    at::Tensor result = torch::empty_like(input1);
    // 参数顺序：输入张量 → 属性参数 → 输出张量（必须在最后）
    EXEC_NPU_CMD(aclnnCustomOp, input1, ..., result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
```

### 1.2 核心规则

**复杂算子必须使用具体 ACLNN 函数名**：对于 `extract_complex` 提取的复杂算子，`EXEC_NPU_CMD` 第一个参数必须是 autogen 生成的实际函数名（如 `aclnnBatchMatMulV3`），而不是 `aclnnCustomOp`。可通过 `nm -D libcust_opapi.so | grep aclnn` 查看导出的实际函数名。简单算子（`extract` 提取）仍可使用 `aclnnCustomOp`。

- module.py 中使用 `kernel_gen_ops.<opname>(...)`，**不是** `custom_pybind_api`
- **EXEC_NPU_CMD 参数顺序错误是最常见 bug**，会导致 `Check out != nullptr`。输出张量必须放最后
- 输入/输出顺序与 OpDef 中 `Input()`/`Output()` 声明顺序匹配

### 1.3 OpDef 属性类型映射

| OpDef 类型 | C++ 类型 | 示例 |
|:---|:---|:---|
| `ListInt` | `at::IntArrayRef` | `at::IntArrayRef output_size` |
| `Int` | `int64_t` | `int64_t dim` |
| `Float` | `float` 或 `double` | `float scale` |
| `Bool` | `bool` | `bool align_corners` |
| `String` | `std::string`（传 EXEC_NPU_CMD 用 `const_cast<char*>(str.c_str())`） | `std::string mode` |

### 1.4 标量参数类型必须与 autogen API 精确匹配

**问题**：pybind.cpp 中 `float scale` → EXEC_NPU_CMD → autogen 生成函数指针签名含 `float` → 但 autogen API 期望 `double` → ABI 不匹配 → scale 被解析为 0。

**规则**：pybind.cpp 中标量参数的类型必须与 autogen 生成的 `aclnnXxxGetWorkspaceSize` 函数签名中对应参数类型**完全一致**。

**检查方法**：打开 `build_out/autogen/aclnn_xxx.h`，查看 `GetWorkspaceSize` 签名中每个参数的类型。

**调试特征**：若 tiling 日志显示属性值恒为 0（如 `scale: 0.000000`），而测试传入非零值，则几乎一定是此类型不匹配问题。

### 1.5 NPU 张量的 CPU 访问规则

`torch::Tensor.accessor<T, N>()` 仅适用于 CPU 张量。若需从 NPU 张量读取元数据（如 `size` 张量中的输出尺寸），必须先 `.to(torch::kCPU).contiguous()` 再使用 accessor。示例：

```cpp
auto size_cpu = size.to(torch::kCPU).contiguous();
auto size_acc = size_cpu.accessor<int32_t, 1>();
int64_t out_h = size_acc[0];
```

注意：`.to(torch::kCPU)` 创建新张量，不修改原 tensor，原 NPU tensor 仍可正常传给 `EXEC_NPU_CMD`。

### 1.6 数据布局转换

若算子用通道最后布局（NHWC/NDHWC），在 pybind.cpp 中完成 permute。详见第8.2节。

---

## 2. module.py

### 2.1 模板

```python
from typing import List
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    """PyTorch 原生算子参考实现（golden model）。"""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, ...):
        # 必须复制算子的精度提升逻辑（如 int16→float32→计算→转回）
        return result

class ModelNew(nn.Module):
    """自定义 kernel 版本。"""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, ...):
        # 与 Model.forward() 相同签名
        # 推导参数在此计算后传入
        return kernel_gen_ops.<opname>(...)
```

### 2.2 推导参数规则

- `Model.forward()` 只接收用户可见参数，用 PyTorch 原生 API
- `ModelNew.forward()` 与 `Model.forward()` **相同签名**，内部推导额外参数
- `get_inputs()` 返回值与 `Model.forward()` 匹配，**不含推导参数**
- NC1HWC0 格式转换放在 `ModelNew` 中（调用 kernel 前 reshape+permute，返回前逆转换）

### 2.3 常见错误

- Model 未复制精度提升逻辑
- 推导参数泄露到 `get_inputs()`
- 使用 `custom_pybind_api` 而非 `<opname>`

---

## 3. prepare_inputs.py

### 3.1 模板

```python
import torch

def get_inputs(param, device=None):
    """根据 test_cases.csv 行生成输入张量。返回与 Model.forward() 签名匹配的元组。"""
    ...

def get_init_inputs(param, device=None):
    return []

def custom_check_precision(param, outputs, outputs_new):
    """返回 (is_pass, all_abs_diff, all_rel_diff)。"""
    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
```

### 3.2 规则

- 严禁生成违反算子约束的数据（sqrt 负值、gather 无效索引等）
- 索引/形状相关算子严格遵守约束
- `get_inputs()` 只返回张量输入和用户直接指定的参数；推导参数由 `ModelNew.forward()` 计算

### 3.3 精度容差参考

| 数据类型 | RTOL |
|:---|:---|
| 整型 | 0 |
| float16 | 1/512 |
| bfloat16 | 1/256 + 1/16384 |
| float32 | 1/2048 + 1/16384 |
| complex64 | 1/2048 + 1/16384 |

### 3.4 常见错误

- `all_abs_diff`/`all_rel_diff` 必须是 `torch.Tensor`，不能是 Python `float`
- CSV bool 字段可能被解析为 `int`（1/0），兼容写法：
  ```python
  ac = param['align_corners']
  align_corners = bool(ac) if not isinstance(ac, str) else (ac.lower() == 'true')
  ```

---

## 4. test_cases.csv

- 分号 `;` 分隔，首列 `case_id` 从 0 开始
- 覆盖：多种 dtype（float16/float32 + 算子支持的 bfloat16/整型）、多种形状（小/中/大）、所有分支条件
- 每个分支至少 5 个用例
- 算子所有明确分支条件必须有对应用例

---

## 5. 专项：ListFloat 属性拆分

**问题**：`pytorch_npu_helper.hpp` 的 `EXEC_NPU_CMD` 宏依赖 `ConvertType` 函数族，**没有** `std::vector<float>` → `aclFloatArray*` 的转换重载。未匹配类型会导致运行时 `std::bad_alloc` 或 `MemoryError`。

**判断**：若源码 OpDef 中包含 `Attr("xxx").AttrType(OPTIONAL).ListFloat()`，必须拆分为独立的 `Float` 属性。

### 5.1 _def.cpp 变更

```cpp
// 替换前
this->Attr("scales").AttrType(OPTIONAL).ListFloat();
// 替换后
this->Attr("scales_h").AttrType(OPTIONAL).Float();
this->Attr("scales_w").AttrType(OPTIONAL).Float();
```

### 5.2 _tiling.cpp 变更

将 `gert::ContinuousVector` 读取方式替换为独立 float 指针读取：

```cpp
// 替换前
const gert::ContinuousVector *scalesAttr = attrs->GetAttrPointer<gert::ContinuousVector>(SCALES_ATTR);
const float *scalesArray = reinterpret_cast<const float *>(scalesAttr->GetData());
scale_h = scalesArray[0];
scale_w = scalesArray[1];

// 替换后
const float *scalesHPtr = attrs->GetAttrPointer<float>(SCALES_H_ATTR);
const float *scalesWPtr = attrs->GetAttrPointer<float>(SCALES_W_ATTR);
scale_h = (scalesHPtr != nullptr) ? *scalesHPtr : 0.0f;
scale_w = (scalesWPtr != nullptr) ? *scalesWPtr : 0.0f;
```

**注意**：属性索引常量必须与 `_def.cpp` 中的声明顺序一致（索引从 0 开始）。

### 5.3 pybind.cpp 变更

```cpp
// 替换前
at::Tensor custom_pybind_api(at::Tensor &input, ..., std::vector<float> scales)
// 替换后
at::Tensor custom_pybind_api(at::Tensor &input, ..., float scales_h, float scales_w)
```

### 5.4 module.py 变更

`ModelNew.forward()` 中分别计算并传递独立参数：
```python
return kernel_gen_ops.<opname>(input, size, align_corners, scales_h, scales_w)
```

**适用场景**：所有使用 `ListFloat` 属性的算子。常见于上采样类算子（scales 列表）、归一化类算子（axes 列表但类型为 float）等。

---

## 6. 专项：OPTIONAL 输入的 pybind 处理

**判断条件**：OpDef 中有 `.ParamType(OPTIONAL)` 的 Input

**规则**：pybind.cpp 中对应的参数类型必须使用 `c10::optional<at::Tensor>`，不能用 `at::Tensor&`。

```cpp
// 正确
std::tuple<at::Tensor, at::Tensor> op_func(
    at::Tensor &required_input,
    c10::optional<at::Tensor> optional_input,  // OPTIONAL 输入
    ...)

// 错误 - 会将空张量传递给 EXEC_NPU_CMD 而非 nullptr
std::tuple<at::Tensor, at::Tensor> op_func(
    at::Tensor &required_input,
    at::Tensor &optional_input,  // 编译通过但运行时行为错误
    ...)
```

**原因**：`EXEC_NPU_CMD` 需要区分"未提供"（nullptr）和"提供了空张量"。`c10::optional` 在 Python 传入 `None` 时为 `c10::nullopt`，框架会传 nullptr 给 CANN API。

**prepare_inputs.py** 中对应参数在"未提供"时应传 `None`：
```python
if has_optional_input:
    optional_input = torch.randn(...)
else:
    optional_input = None
```

---

## 7. 专项：ListInt 属性含负值默认值

**判断条件**：OpDef 中有 `.AttrType(OPTIONAL).ListInt({-1, ...})` 形式的属性

**规则**：当 tiling 代码对负值做 offset 处理（如 `end = (end < 0) ? end + totalNum : end`）时，prepare_inputs.py 中必须将负值默认值替换为有效的实际范围，否则 tiling 计算出错误的范围（如 start == end，导致空范围）。

```python
# 当 CSV 中 range_start=-1, range_end=-1 表示"不限制"时：
if range_start < 0 or range_end < 0:
    range_vals = [0, total_size]  # 传入完整范围
else:
    range_vals = [range_start, range_end]
```

---

## 8. 专项：数据格式与布局适配

本节覆盖算子中涉及的特殊数据格式与内存布局转换，包括 NC1HWC0 五维格式和通道优先/通道最后布局的互转规则。

### 8.1 NC1HWC0 格式全流程适配

**判断**：`_def.cpp` 中 Input/Output Format 为 `FORMAT_NC1HWC0`

**注意**：`_def.cpp` 和 question 骨架的 `FORMAT_NC1HWC0 → FORMAT_ND` 修改需在步骤4（op_host 适配）中按 `conversion_rules.md` 第9节执行，Tiling 代码清理也在步骤4中完成。本节仅描述 Python 侧的转换规则。

**Python 侧**（`module.py` 的 `ModelNew.forward()`）：
- 调用 kernel 前：ND 数据 reshape+permute 为 5D 布局
- 示例：(B, C, N) → (B, C1=C/16, C0=16, N) → permute(B, C1, N, C0) → unsqueeze(B, C1, N, 1, C0)
- 输出：逆转换回来

**严禁使用 `npu_format_cast`**（编译期符号找不到或运行时格式元数据未生效），必须用 ND 格式 + 手动 reshape。

### 8.2 通道优先/通道最后布局转换

**判断**：tiling 代码 `GetInputShape` 的 dim[1] 是空间维度（D/H/W）而非通道维度（C）

**转换规则**：
- PyTorch 默认 NCHW/NCDHW，若算子内部用 NHWC/NDHWC，须在 **pybind.cpp** 中 permute
- 2D：`tensor.permute({0, 2, 3, 1}).contiguous()` (NCHW→NHWC)
- 3D：`tensor.permute({0, 2, 3, 4, 1}).contiguous()` (NCDHW→NDHWC)
- 算子输出必须逆变换回 PyTorch 默认布局

---

## 变更记录

- 2026-04-23: 第8.1节拆分——_def.cpp 和 question 骨架的格式修改规则移至 `conversion_rules.md` 第9节，本节仅保留 Python 侧 reshape/permute 规则
- 2026-04-23: 修正复杂算子 ACLNN 函数命名规则（必须用具体函数名而非 aclnnCustomOp）
- 2026-04-22: 合并原 data_layout.md 为第8节
- 2026-04-09: 新增第6节 OPTIONAL 输入 pybind 处理规则、第7节 ListInt 负值默认值规则
