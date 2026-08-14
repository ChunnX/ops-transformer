# TorchOpsExtension for FusedGdnGating

## 概述

本目录提供 `fused_gdn_gating` 算子的 PyTorch 绑定，将 `torch.ops.custom.npu_fused_gdn_gating` 注册到 PyTorch 自定义算子命名空间，底层通过 ACLNN `aclnnFusedGdnGating` 接口在 NPU 上执行。

## 目录结构

```
torch_ops_extension/
├── build_and_install.sh                  # 编译安装脚本
├── setup.py                              # 构建配置（NpuExtension）
└── custom_ops/
    ├── __init__.py                       # Python 镜像，将 torch.ops.custom.* 挂载到 torch_npu
    ├── csrc/
    │   ├── ops_def_registration.cpp      # TORCH_LIBRARY(custom, m) 算子 schema 定义
    │   ├── npu_fused_gdn_gating.cpp      # NPU 实现 + Meta 实现 + TORCH_LIBRARY_IMPL 注册
    │   ├── ops_common.h                  # 共享基础设施（EXEC_NPU_CMD_V1 宏、类型转换器）
    │   └── ops_common.cpp               # 共享辅助函数实现
    └── converter/
        └── __init__.py                   # GE 转换器占位（暂无实现）
```

## 注册流程（5步）

1. `TORCH_LIBRARY(custom, m)` — 定义算子 schema
2. `npu_fused_gdn_gating_npu()` — NPU 前向实现，调用 `EXEC_NPU_CMD_V1(aclnnFusedGdnGating, ...)`
3. `npu_fused_gdn_gating_meta()` — Meta 设备实现（仅 shape 推导）
4. `TORCH_LIBRARY_IMPL(custom, PrivateUse1, m)` — 注册 NPU 实现
5. `TORCH_LIBRARY_IMPL(custom, Meta, m)` — 注册 Meta 实现

## 算子接口

```python
torch.ops.custom.npu_fused_gdn_gating(
    A_log,      # [num_heads], FP32/BF16/FP16
    a,          # [batch, num_heads], BF16/FP16
    b,          # [batch, num_heads], BF16/FP16
    dt_bias,    # [num_heads], FP32/BF16/FP16
    beta=1.0,   # float
    threshold=20.0,  # float
) -> (g, beta_output)
    # g:           [1, batch, num_heads], FP32
    # beta_output: [1, batch, num_heads], 同 b dtype
```

## 前置条件

- Linux 系统，Python 3.8+
- GCC 9.4+
- PyTorch >= 2.6
- TorchNPU
- CANN 工具包

## 编译安装

```bash
cd torch_ops_extension
bash build_and_install.sh
```

或手动编译：

```bash
python3 setup.py build_ext --inplace
```

## pytest 集成

测试框架通过 `custom_ops.py` 加载器自动发现并加载 `.so`，无需手动设置 `PYTHONPATH`。

环境变量 `FUSED_GDN_GATING_CUSTOM_OPS_PATH` 可指定额外的扩展搜索路径。
