# moe_finalize_routing — PyTorch 接入层 (TorchNPU / torch_ops_extension)

将 `moe_finalize_routing` 算子桥接到 PyTorch，注册为 `torch.ops.custom.npu_moe_finalize_routing`
（同时挂载到 `TorchNPU.npu_moe_finalize_routing`），底层调用 ACLNN 接口 `aclnnMoeFinalizeRoutingV4`。

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
    │   ├── ops_def_registration.cpp       # TORCH_LIBRARY(custom): npu_moe_finalize_routing schema
    │   ├── ops_common.h / ops_common.cpp  # EXEC_NPU_CMD_V1 / ConvertType 基础设施
    │   └── npu_moe_finalize_routing.cpp   # NPU/Meta 前向实现 + TORCH_LIBRARY_IMPL
    └── converter/
        ├── __init__.py
        └── npu_moe_finalize_routing.py    # torch.compile (GE) 成图 converter
```

## 前置条件

- Linux；Python 3.8+；GCC 9.4.0+
- PyTorch >= 2.6.0 与匹配版本的 `TorchNPU`
- Ascend CANN Toolkit（`ASCEND_HOME_PATH` 已设置）
- 已部署 `aclnnMoeFinalizeRoutingV4` 算子包，运行时可经
  `ASCEND_CUSTOM_OPP_PATH` / `ASCEND_OPP_PATH` 检索到 `libcust_opapi.so`

## 构建与安装

```sh
# 方式一：构建 wheel 并安装（推荐）
bash build_and_install.sh

# 方式二：就地编译（.so 生成在 custom_ops/custom_ops_lib*.so）
python3 setup.py build_ext --inplace
```

## 用法

### eager / 单算子

```python
import torch
import torch_npu
import custom_ops

NUM_ROWS = 4
K = 2
H = 8
E = 8
CE_NUM = 2  # constant_expert_range_num = end - start

expanded_x = torch.randn(NUM_ROWS * K, H, dtype=torch.float32, device="npu")
row_idx = torch.arange(NUM_ROWS * K, dtype=torch.int32, device="npu")
scales = torch.randn(NUM_ROWS, K, dtype=torch.float32, device="npu")
x1 = torch.randn(NUM_ROWS, H, dtype=torch.float32, device="npu")
x2 = torch.randn(NUM_ROWS, H, dtype=torch.float32, device="npu")
bias = torch.randn(E, H, dtype=torch.float32, device="npu")
expert_idx = torch.tensor(
    [[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.int32, device="npu"
)
x = torch.randn(NUM_ROWS, H, dtype=torch.float32, device="npu")
alpha1 = torch.randn(CE_NUM, H, dtype=torch.float32, device="npu")
alpha2 = torch.randn(CE_NUM, H, dtype=torch.float32, device="npu")
v = torch.randn(CE_NUM, H, dtype=torch.float32, device="npu")

out = torch.ops.custom.npu_moe_finalize_routing(
    expanded_x,
    row_idx,
    x1=x1,
    x2=x2,
    bias=bias,
    scales=scales,
    expert_idx=expert_idx,
    x=x,
    alpha1=alpha1,
    alpha2=alpha2,
    v=v,
    drop_pad_mode=0,
    zero_expert_range=[0, 2],
    copy_expert_range=[2, 4],
    constant_expert_range=[6, 8],
    k=K,
)
print(out.shape)  # torch.Size([4, 8])
# 也可：torch_npu.npu_moe_finalize_routing(...)
```

## 接口与 IR

- 入参/属性/输出与算子原型 `moe_finalize_routing_v2/op_host/op_api/aclnn_moe_finalize_routing_v4.h` 一一对应。
- `EXEC_NPU_CMD_V1` 实参按 IR 声明顺序传入（输入→属性→输出）。
- 返回单个 Tensor，shape 为 `(NUM_ROWS, H)`，dtype 与 `expanded_x` 一致。
