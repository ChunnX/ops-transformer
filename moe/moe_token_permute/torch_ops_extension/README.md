# moe_token_permute — PyTorch 接入层 (TorchNPU / torch_ops_extension)

将 `moe_token_permute` 算子桥接到 PyTorch，注册为 `torch.ops.custom.npu_moe_token_permute`
（同时挂载到 `TorchNPU.npu_moe_token_permute`），底层调用 ACLNN 接口 `aclnnMoeTokenPermuteV2`。

> 结构与配置对齐参考实现 `ops_adv_open/torch_ops_extension`：编译式 `NpuExtension`
> （`custom_ops.custom_ops_lib`）+ `TORCH_LIBRARY`/`TORCH_LIBRARY_IMPL` + `EXEC_NPU_CMD_V1`。
> `csrc/ops_common.h`、`csrc/ops_common.cpp` 为该参考的整文件拷贝（自包含）。

## 目录结构

```
torch_ops_extension/
├── setup.py                    # NpuExtension custom_ops.custom_ops_lib，编译 csrc/*.cpp
├── build_and_install.sh        # 构建 wheel 并 pip 安装
├── README.md
└── custom_ops/
    ├── __init__.py             # 导入 custom_ops_lib(.so) + converter，并挂载到 TorchNPU
    ├── csrc/
    │   ├── ops_def_registration.cpp       # TORCH_LIBRARY(custom): npu_moe_token_permute schema
    │   ├── ops_common.h / ops_common.cpp  # EXEC_NPU_CMD_V1 / ConvertType 基础设施（拷贝）
    │   └── npu_moe_token_permute.cpp      # NPU/Meta 前向实现 + TORCH_LIBRARY_IMPL
    └── converter/
        ├── __init__.py
        └── npu_moe_token_permute.py       # torch.compile (GE) 成图 converter
```

## 前置条件

- Linux；Python 3.8+；GCC 9.4.0+
- PyTorch >= 2.6.0 与匹配版本的 `TorchNPU`
- Ascend CANN Toolkit（`ASCEND_HOME_PATH` 已设置）
- 已部署 `aclnnMoeTokenPermuteV2` 算子包，运行时可经
  `ASCEND_CUSTOM_OPP_PATH` / `ASCEND_OPP_PATH` 检索到 `libcust_opapi.so`

## 构建与安装

```sh
# 方式一：构建 wheel 并安装（推荐）
bash build_and_install.sh

# 方式二：就地编译（.so 生成在 custom_ops/custom_ops_lib*.so）
python3 setup.py build_ext --inplace
```

## 用法

### 非量化调用

```python
import torch
import torch_npu
import custom_ops   # 注册 torch.ops.custom.npu_moe_token_permute 并挂载到 torch_npu

tokens = torch.randn(4, 128, dtype=torch.float16, device="npu")
indices = torch.tensor(
    [[1, 0], [0, 1], [1, 0], [0, 1]],
    dtype=torch.int32,
    device="npu",
)

permuted_tokens, sorted_indices, expanded_scale = torch.ops.custom.npu_moe_token_permute(
    tokens,
    indices,
    num_out_tokens=6,
)

print(permuted_tokens.shape)  # torch.Size([6, 128])
print(sorted_indices.shape)   # torch.Size([8])
print(expanded_scale.shape)   # torch.Size([0])
# 也可：torch_npu.npu_moe_token_permute(...)
```

### Ascend 950 MXFP8 调用

```python
permuted_tokens, sorted_indices, expanded_scale = torch.ops.custom.npu_moe_token_permute(
    tokens,
    indices,
    quant_mode=2,
)

print(permuted_tokens.shape)  # torch.Size([8, 128])
print(expanded_scale.shape)   # torch.Size([8, 4])
```

### Ascend 950 MXFP4 调用

```python
tokens = torch.randn(4, 128, dtype=torch.float16, device="npu")
permuted_tokens, sorted_indices, expanded_scale = torch.ops.custom.npu_moe_token_permute(
    tokens,
    indices,
    quant_mode=9,
)

print(permuted_tokens.shape)  # torch.Size([8, 64])，物理隐藏维 H / 2
print(expanded_scale.shape)   # torch.Size([8, 2, 2])
```

## 接口与 IR

- 入参/属性/输出与算子原型 `op_host/op_api/aclnn_moe_token_permute_v2.h` 一一对应。
- `EXEC_NPU_CMD_V1` 实参按 IR 声明顺序传入（输入→属性→输出）。
- 非量化：`permuted_tokens` dtype 与 `tokens` 相同；`expanded_scale` 为空 Tensor。
- 量化模式（quant_mode=2/3/9）：`permuted_tokens` 为 float8/uint8；`expanded_scale` 为 per-token 分块 scale。
- MXFP4 通过 uint8 物理存储，调用 aclnn 时 dtype 被重写为 `ACL_FLOAT4_E2M1` / `ACL_FLOAT8_E8M0`。
