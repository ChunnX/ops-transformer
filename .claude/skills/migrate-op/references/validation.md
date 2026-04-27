# Validation 与题目文件

生成 `question/`、`validation/` 和 `question/api_desc.md` 时读取本文。

## 1. question

`question/` 是给 LLM 的骨架，不参与 `answer/0` 参考实现编译。它要暴露正确接口，但不包含完整答案。

`question/op_host/<kernel_name>.cpp`：

- 沿用 answer 的 Input/Output/Attr 定义和 `OpName`
- 保留 `ascend910b`、`ascend910_93`
- 简单占位 tiling 返回 `GRAPH_SUCCESS`
- 不包含复杂 tiling 算法

`question/op_kernel/<kernel_name>.cpp`：

- kernel 函数名和 `GM_ADDR` 参数顺序与 answer 完全一致
- 函数体留空或最小占位

`question/api_desc.md`：

- 写中文说明
- 包含功能、接口、参数、返回值、约束和一个 Python 调用示例
- Python 接口名使用 `kernel_gen_ops.<kernel_name>()`

## 2. pybind.cpp

模板占位符：

```cpp
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

at::Tensor custom_pybind_api(at::Tensor &x, ...)
{
    at::Tensor y = torch::empty_like(x);
    EXEC_NPU_CMD(aclnnCustomOp, x, ..., y);
    return y;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.def("custom_pybind_api", &custom_pybind_api, "");
}
```

规则：

- 源文件只保留 `aclnnCustomOp` 和 `custom_pybind_api` 占位符，框架会自动渲染；不要 include、复制或手写 `aclnn_*.h`/`aclnn_*.cpp`
- `EXEC_NPU_CMD` 参数顺序为输入张量、属性参数、输出张量
- 输出张量放在最后
- 输入/输出顺序与 OpDef 一致
- 标量 C++ 类型必须与编译后 `build_out/autogen/aclnn_*.h` 中 `GetWorkspaceSize` 签名一致，先编译生成 autogen 头再核对
- 从 NPU 张量读取 shape/size 值时，先 `.to(torch::kCPU).contiguous()`
- OPTIONAL 输入优先使用 `c10::optional<at::Tensor>`

常用映射：

| OpDef | pybind C++ |
|---|---|
| `Int` | `int64_t` |
| `Float` | `float` 或按 autogen 签名调整 |
| `Bool` | `bool` |
| `ListInt` | `at::IntArrayRef` |
| `String` | `std::string`，传入 C API 时用 `const_cast<char*>(s.c_str())` |

## 3. module.py

结构：

```python
import torch
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def forward(self, ...):
        return torch_reference(...)

class ModelNew(nn.Module):
    def forward(self, ...):
        return kernel_gen_ops.<kernel_name>(...)
```

规则：

- `Model.forward()` 和 `ModelNew.forward()` 签名一致
- `Model` 使用 PyTorch CPU golden，包含算子的精度提升逻辑和用户可见语义
- `ModelNew` 内部计算推导参数、布局转换参数或 pybind 桥接参数
- `get_inputs()` 返回值只匹配 `Model.forward()` 的用户可见参数
- 自定义算子调用名使用渲染后的 `kernel_name`

## 4. prepare_inputs.py

必须导出：

```python
def get_init_inputs(param, device=None):
    return []

def get_inputs(param, device=None):
    return (...)
```

可选导出：

```python
def custom_check_precision(param, outputs, outputs_new):
    return is_pass, all_abs_diff, all_rel_diff
```

规则：

- 生成满足算子约束的数据，例如合法 index、合法 shape、合法 dtype
- bool 字段兼容 CSV 中的 `0/1` 和 `true/false`
- `all_abs_diff`、`all_rel_diff` 返回 `torch.Tensor`
- 不把推导参数放进 `get_inputs()`

## 5. test_cases.csv

- 使用分号分隔
- 首列 `case_id` 从 0 开始
- 覆盖 dtype、shape、边界条件和 tiling 分支
- 先写少量稳定 case；迁移通过后再补充分支覆盖

## 6. 常见桥接

### ListFloat

`EXEC_NPU_CMD` 不稳定支持 `std::vector<float>` 到 `aclFloatArray*` 的转换。OpDef 中 `ListFloat` 属性可拆成固定数量的 `Float` 属性；`Model.forward()` 保持原用户语义，`ModelNew.forward()` 内部拆分后调用 pybind。

### OPTIONAL 输入

Python 传 `None` 时，pybind 用 `c10::optional<at::Tensor>` 表达缺省输入。OpDef 保持 `.ParamType(OPTIONAL)`。

### 布局转换

如果 kernel 使用 NHWC/NDHWC/NC1HWC0 等内部布局，在 `ModelNew` 或 pybind 中完成输入转换和输出还原。`Model` 保持 PyTorch 用户布局。
