# CANN Ops Transformer

`cann_ops_transformer` 是专为昇腾 NPU 设计的高性能算子扩展库。它利用即时编译（JIT）技术，将 PyTorch 函数接口与 ACLNN 库进行桥接。

## 编译与安装

### 前置条件

- 操作系统：Linux
- Python：3.8+
- 编译器：GCC 9.4.0+
- 框架：
  - PyTorch>=2.6.0
  - torch_npu（与 PyTorch 版本匹配）
- 工具包：昇腾 CANN Toolkit

### 安装步骤

1. 安装依赖：

    ```sh
    python3 -m pip install -r requirements.txt
    ```

2. 构建 Wheel 包：

    ```sh
    # -n: 非隔离构建（使用现有环境）
    python3 -m build --wheel -n
    ```

3. 安装软件包：

    ```sh
    python3 -m pip install dist/*.whl --force-reinstall --no-deps
    ```

## 快速开始

使用 `cann_ops_transformer` 非常便捷。您可以直接通过库的算子集调用 NPU 加速的算子。

```python
import torch
import torch_npu
import cann_ops_transformer

# 在 NPU 上初始化数据
x = torch.randn(10, 32, dtype=torch.float32).npu()

# 调用自定义 NPU 算子
# 第一次调用时会触发 JIT 编译
npu_result = cann_ops_transformer.ops.abs(x)

# 与 CPU ATen 实现进行验证
cpu_x = x.cpu()
cpu_result = torch.ops.aten.abs(cpu_x)

assert torch.allclose(cpu_result, npu_result.cpu(), rtol=1e-6)
print("验证成功！")
```

## 开发者指南：添加新算子

> 完整的算子开发规范 —— 目录结构、命名规范、各层实现（C++ / Python / torchair 图模式）、文档字符串和 DeviceGuard 要求 —— 请参见 [torch_extension 开发规范](cann_ops_transformer/docs/torch_extension_guidelines.md)。

要实现一个新算子（例如 `abs`），您需要提供两个组件：C++ 内核封装和 Python JIT 构建器。

### 1. C++ 后端（`ops/csrc/<OP_NAME>.cpp`）

该文件将 PyTorch tensor 桥接到 ACLNN C-API。

```cpp
#include <torch/extension.h>
#include "aclnn_common.h"

/**
 * @brief aclnnAbs 的 ACLNN 封装
 * @param x 输入 Tensor（在 NPU 上）
 * @return 结果 Tensor
 */
at::Tensor npu_abs(const at::Tensor &x)
{
    // 1. 手动分配输出 tensor（PyTorch 标准实践）
    at::Tensor y = at::empty_like(x);

    // 2. 使用辅助宏启动 ACLNN 内核
    ACLNN_CMD(aclnnAbs, x, y);

    return y;
}

// 将 C++ 函数绑定到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_abs", &npu_abs, "abs");
}
```

### 2. Python 前端（`ops/<OP_NAME>.py`）

该文件管理 JIT 编译逻辑，并将算子注册到 PyTorch Dispatcher。

```python
import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY

class AbsOpBuilder(OpBuilder):
    def __init__(self):
        super(AbsOpBuilder, self).__init__("abs")

    def sources(self):
        """C++ 源代码路径"""
        return ['ops/csrc/abs.cpp']

    def schema(self) -> str:
        """PyTorch 算子签名"""
        return "abs(Tensor x) -> Tensor"

    def register_meta(self):
        """
        注册 Meta 实现（形状/数据类型推导）。
        对于 Autograd 和 FakeTensor 支持至关重要。
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def abs_meta(x):
            return torch.empty_like(x)

# 实例化构建器
abs_op_builder = AbsOpBuilder()

@impl(AS_LIBRARY, abs_op_builder.name, "PrivateUse1")
def abs(x):
    """
    NPU 的 Dispatcher 实现。
    'PrivateUse1' 是自定义 NPU 后端使用的分发键。
    """
    op_module = abs_op_builder.load()  # 编译/加载 .so 文件
    return op_module.npu_abs(x)
```

### 技术说明

| 组件 | 职责 |
| --- | --- |
| **OpBuilder** | 使用 `ninja` 处理 C++ 源代码的 JIT 编译。 |
| **Meta Dispatch** | 使 PyTorch 能够在无需运行 NPU 代码的情况下推导输出形状/类型。 |
| **PrivateUse1** | PyTorch 用于路由 NPU 特定操作的特定后端键。 |