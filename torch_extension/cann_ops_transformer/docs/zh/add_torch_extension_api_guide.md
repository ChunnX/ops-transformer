# 在 ops-transformer 中新增 Torch API 的开发指引

本文以 `MoeTokenPermute` 为例，说明当不允许在 `op-plugin` 新增或修改 `torch_npu` API 时，如何在 `ops-transformer-public/torch_extension/cann_ops_transformer/ops` 下新增一个可被 PyTorch 调用的 API。

## 适用场景

当满足以下条件时，推荐采用本文方式：

- 不允许修改 `op-plugin/op_plugin/config/op_plugin_functions.yaml`。
- 不希望新增 `torch_npu.xxx` 命名空间下的 API。
- 算子已经提供 aclnn 接口，例如 `aclnnMoeTokenPermute`。
- 调用路径只需要覆盖 aclnn 直调，不需要支持图模式或单算子 `*.def` 生成路径。
- 希望 API 跟随 `ops-transformer-public` 仓发布。

最终调用路径如下：

```text
Python 用户代码
  -> cann_ops_transformer.ops.npu_moe_token_permute
  -> torch.ops.cann_ops_transformer.npu_moe_token_permute
  -> torch_extension C++ bridge
  -> aclnnMoeTokenPermute
  -> 算子侧按平台分流，例如 Ascend 950 调 MoeInitRoutingV3
```

## 目录结构

新增一个 torch extension API 通常需要三个文件：

```text
torch_extension/cann_ops_transformer/ops/
  __init__.py
  moe_token_permute.py
  csrc/
    moe_token_permute.cpp
```

职责划分：

- `ops/<op_name>.py`：定义 PyTorch dispatcher schema、Meta 实现、PrivateUse1 注册、用户侧 Python 包装函数。
- `ops/csrc/<op_name>.cpp`：分配输出 Tensor，并通过 `ACLNN_CMD` 调用 aclnn 接口。
- `ops/__init__.py`：导出用户可 import 的 Python API。

## Python Builder 设计

参考 `ops/moe_distribute_combine.py`，新增一个继承 `OpBuilder` 的 builder。

示例：

```python
class MoeTokenPermuteOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeTokenPermuteOpBuilder, self).__init__("npu_moe_token_permute")

    def sources(self):
        return ["ops/csrc/moe_token_permute.cpp"]

    def schema(self) -> str:
        return "npu_moe_token_permute(Tensor tokens, Tensor indices, " \
            "int? num_out_tokens=None, bool padded_mode=False, int quant_mode=-1) -> (Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_token_permute_meta(tokens, indices, num_out_tokens=None, padded_mode=False, quant_mode=-1):
            ...
```

关键点：

- builder 名称 `npu_moe_token_permute` 会成为 `torch.ops.cann_ops_transformer.npu_moe_token_permute` 的 op 名。
- `sources()` 返回 C++ 文件相对 `cann_ops_transformer` 包的路径。
- `schema()` 必须和 Python Meta 函数、PrivateUse1 函数、C++ pybind 入参保持一致。
- `register_meta()` 负责 FakeTensor、编译期 shape/dtype 推导。

## Meta 实现原则

Meta 实现只做 shape/dtype 推导，不真正调用 NPU。

`MoeTokenPermute` 的推导规则：

```text
flatten_size = indices.numel()

if num_out_tokens is None:
    num_out_tokens_value = 0
else:
    num_out_tokens_value = num_out_tokens

if num_out_tokens_value > 0:
    actual_num_out_tokens = min(num_out_tokens_value, flatten_size)
else:
    actual_num_out_tokens = max(num_out_tokens_value + flatten_size, 0)
```

输出规则：

```text
quant_mode = -1:
  permuted_tokens: [actual_num_out_tokens, hidden_size], dtype = tokens.dtype
  sorted_indices: [indices.numel()], dtype = int32
  expanded_scale: [0], dtype = float32

quant_mode = 2:
  permuted_tokens: [actual_num_out_tokens, hidden_size], dtype = float8_e5m2
  expanded_scale: [actual_num_out_tokens, AlignUp(CeilDiv(hidden_size, 32), 2)], dtype = float8_e8m0fnu

quant_mode = 3:
  permuted_tokens: [actual_num_out_tokens, hidden_size], dtype = float8_e4m3fn
  expanded_scale: [actual_num_out_tokens, AlignUp(CeilDiv(hidden_size, 32), 2)], dtype = float8_e8m0fnu

quant_mode = 9:
  permuted_tokens: [actual_num_out_tokens, hidden_size / 2], dtype = uint8
  expanded_scale: [actual_num_out_tokens, CeilDiv(hidden_size, 64), 2], dtype = uint8
```

说明：

- MXFP4 的 PyTorch 侧物理 dtype 使用 `uint8` 承载。
- C++ 调 aclnn 时再通过 `TensorWrapper` 把逻辑 dtype 覆盖成 `ACL_FLOAT4_E2M1` 和 `ACL_FLOAT8_E8M0`。
- Python Meta 不可靠地感知真实 SoC，因此只按入参推导。平台相关的静默忽略策略放在 C++ runtime 层处理。

## PrivateUse1 注册

在 Python 文件中加载 C++ extension，并注册 NPU backend 的 dispatcher 实现：

```python
moe_token_permute_op_builder = MoeTokenPermuteOpBuilder()
op_module = moe_token_permute_op_builder.load()


@impl(AS_LIBRARY, moe_token_permute_op_builder.name, "PrivateUse1")
def _npu_moe_token_permute(tokens, indices, num_out_tokens=None, padded_mode=False, quant_mode=-1):
    return op_module.npu_moe_token_permute(tokens, indices, num_out_tokens, padded_mode, quant_mode)
```

对外再包一层普通 Python 函数：

```python
def npu_moe_token_permute(tokens, indices, num_out_tokens=None, padded_mode=False, quant_mode=-1):
    return torch.ops.cann_ops_transformer.npu_moe_token_permute(
        tokens,
        indices,
        num_out_tokens,
        padded_mode,
        quant_mode,
    )
```

## C++ Bridge 设计

C++ 文件负责运行时检查、输出分配、aclnn 调用。

基本结构：

```cpp
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor, at::Tensor> npu_moe_token_permute(
    const at::Tensor &tokens,
    const at::Tensor &indices,
    c10::optional<int64_t> num_out_tokens,
    bool padded_mode,
    int64_t quant_mode)
{
    ...
    ACLNN_CMD(aclnnMoeTokenPermute, ...);
    return std::make_tuple(permuted_tokens, sorted_indices, expanded_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_moe_token_permute", &npu_moe_token_permute, "npu_moe_token_permute");
}
}  // namespace op_api
```

运行时检查建议：

- `tokens.dim() == 2`
- `indices.dim() == 1 || indices.dim() == 2`
- `quant_mode` 只允许 `-1, 2, 3, 9`
- `quant_mode == 9` 时，`hidden_size` 必须为偶数
- 量化场景不支持 autograd 时，应显式检查 `tokens.requires_grad()`

## 平台分流策略

如果需求要求“非 Ascend 950 静默忽略量化参数”，应在 C++ runtime 层处理：

```cpp
const char *soc_name = aclrtGetSocName();
bool is_ascend950 = soc_name != nullptr && std::strstr(soc_name, "Ascend950") != nullptr;
int64_t effective_quant_mode = is_ascend950 ? quant_mode : QUANT_MODE_NONE;
```

后续所有输出分配和 aclnn 调用都使用 `effective_quant_mode`。

这样可以保证：

- Ascend 950 下，`quant_mode=2/3/9` 进入量化路径。
- 非 Ascend 950 下，即使用户传了量化参数，也按 `quant_mode=-1` 行为处理。
- 不需要修改 `*.def`，也不引入图模式/单算子路径。

## 输出分配

非量化：

```cpp
permuted_tokens = at::empty({actual_num_out_tokens, hidden_size}, tokens.options());
expanded_scale = at::empty({0}, tokens.options().dtype(at::kFloat));
```

MXFP8：

```cpp
permuted_tokens = at::empty(
    {actual_num_out_tokens, hidden_size},
    tokens.options().dtype(at::kFloat8_e5m2));  // 或 at::kFloat8_e4m3fn

expanded_scale = at::empty(
    {actual_num_out_tokens, AlignUp(CeilDiv(hidden_size, 32), 2)},
    tokens.options().dtype(at::kFloat8_e8m0fnu));
```

MXFP4：

```cpp
permuted_tokens = at::empty(
    {actual_num_out_tokens, hidden_size / 2},
    tokens.options().dtype(at::kByte));

expanded_scale = at::empty(
    {actual_num_out_tokens, CeilDiv(hidden_size, 64), 2},
    tokens.options().dtype(at::kByte));
```

MXFP4 需要用 `TensorWrapper` 覆盖逻辑 dtype：

```cpp
TensorWrapper permuted_tokens_wrapper = {
    permuted_tokens, ConvertToAclDataType(permuted_tokens.scalar_type())};
TensorWrapper expanded_scale_wrapper = {
    expanded_scale, ConvertToAclDataType(expanded_scale.scalar_type())};

if (effective_quant_mode == QUANT_MODE_MXFP4_E2M1) {
    permuted_tokens_wrapper.dtype = aclDataType::ACL_FLOAT4_E2M1;
    expanded_scale_wrapper.dtype = aclDataType::ACL_FLOAT8_E8M0;
}
```

## aclnn 调用

`MoeTokenPermute` 的 aclnn 调用形式：

```cpp
ACLNN_CMD(aclnnMoeTokenPermute,
    tokens,
    indices,
    actual_num_out_tokens,
    padded_mode,
    effective_quant_mode,
    permuted_tokens_wrapper,
    sorted_indices,
    expanded_scale_wrapper);
```

注意：

- 传入 aclnn 的是 `effective_quant_mode`，不是原始 `quant_mode`。
- 非 Ascend 950 下必须传 `-1`，以保持兼容。
- `expanded_scale` 即使非量化也要传入 shape `[0]` 的 float32 Tensor，因为 aclnn 新接口已经包含该输出。

## 导出 API

在 `ops/__init__.py` 增加：

```python
from .moe_token_permute import npu_moe_token_permute
```

这样用户可以：

```python
from cann_ops_transformer.ops import npu_moe_token_permute
```

也可以：

```python
import cann_ops_transformer

cann_ops_transformer.ops.npu_moe_token_permute(...)
```

底层 dispatcher 入口为：

```python
torch.ops.cann_ops_transformer.npu_moe_token_permute(...)
```

## 用户调用示例

安装：

```bash
cd ops-transformer-public/torch_extension
python3 -m pip install -r requirements.txt
python3 -m build --wheel -n
python3 -m pip install dist/*.whl --force-reinstall --no-deps
```

非量化：

```python
import torch
import torch_npu
from cann_ops_transformer.ops import npu_moe_token_permute

tokens = torch.randn((8, 128), dtype=torch.float16, device="npu")
indices = torch.randint(0, 4, (8, 2), dtype=torch.int32, device="npu")

permuted_tokens, sorted_indices, expanded_scale = npu_moe_token_permute(
    tokens,
    indices,
    num_out_tokens=None,
    padded_mode=False,
    quant_mode=-1,
)
```

Ascend 950 MXFP8：

```python
permuted_tokens, sorted_indices, expanded_scale = npu_moe_token_permute(
    tokens,
    indices,
    num_out_tokens=20,
    padded_mode=False,
    quant_mode=2,
)
```

Ascend 950 MXFP4：

```python
tokens = torch.randn((8, 128), dtype=torch.float16, device="npu")

permuted_tokens, sorted_indices, expanded_scale = npu_moe_token_permute(
    tokens,
    indices,
    num_out_tokens=20,
    padded_mode=False,
    quant_mode=9,
)
```

## 验证建议

基础检查：

```bash
PYTHONPYCACHEPREFIX=/tmp/python-pycache python3 -m py_compile \
  torch_extension/cann_ops_transformer/ops/moe_token_permute.py
```

代码格式检查：

```bash
git diff --check
```

功能验证建议覆盖：

- `quant_mode=-1`，`indices` 为 1D。
- `quant_mode=-1`，`indices` 为 2D。
- `num_out_tokens=None`。
- `num_out_tokens > indices.numel()`。
- `num_out_tokens < indices.numel()`。
- Ascend 950 上 `quant_mode=2`。
- Ascend 950 上 `quant_mode=3`。
- Ascend 950 上 `quant_mode=9`。
- 非 Ascend 950 上传 `quant_mode=2/3/9`，确认静默按非量化输出。

## 常见问题

### 这个 API 是否会出现在 torch_npu 下？

不会。本文方式注册的是：

```python
torch.ops.cann_ops_transformer.npu_moe_token_permute
```

对外推荐入口是：

```python
from cann_ops_transformer.ops import npu_moe_token_permute
```

### 是否需要修改 op-plugin？

不需要。本文方式绕开 `op-plugin` 自动生成链路，直接在 `ops-transformer-public` 的 torch extension 包里定义 API。

### 是否需要修改 *.def？

不需要。本文方式只覆盖 aclnn 直调路径，不引入图模式或单算子调用。

### 第一次调用为什么慢？

`OpBuilder.load()` 会触发 JIT 编译 C++ extension。首次调用会生成并加载 `.so`，后续调用会复用缓存。

### MXFP4 为什么 Python dtype 是 uint8？

PyTorch 侧没有统一稳定地直接承载所有 CANN FP4 逻辑 dtype 的方式。当前采用 `uint8` 作为物理存储，C++ 调 aclnn 时通过 `TensorWrapper` 将逻辑 dtype 覆盖为 `ACL_FLOAT4_E2M1`。

## 开发检查清单

- [ ] 新增 `ops/<op_name>.py`。
- [ ] 新增 `ops/csrc/<op_name>.cpp`。
- [ ] Python schema、Meta、PrivateUse1、C++ pybind 入参一致。
- [ ] Meta 输出 shape/dtype 与 C++ 输出分配一致。
- [ ] C++ 使用 `ACLNN_CMD` 调 aclnn。
- [ ] 非目标平台的兼容策略在 C++ runtime 层处理。
- [ ] `ops/__init__.py` 导出用户 API。
- [ ] 不修改 `op-plugin`。
- [ ] 不修改 `*.def`，除非明确要求支持图模式/单算子路径。
