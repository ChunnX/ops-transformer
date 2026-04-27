# Ascend C 自定义算子 aclnn 单算子直调测试

## 概述

从零开始调用 Ascend C 自定义算子（aclnn API）的完整流程，包括环境配置、编译安装、测试代码编写、运行验证。

## 适用场景

- 新开发的自定义算子需要验证功能正确性
- 算子已编译但 Python API 封装未完成时的替代测试方案
- 调试算子 kernel 问题时的快速验证手段

## 完整工作流

### Phase 1: 环境检查与准备

```bash
# 检查 NPU 设备和 SoC 类型（关键步骤）
npu-smi info
python3 -c "import torch_npu; print(torch_npu.npu.get_device_name(0))"

# 设置环境变量
source ${ASCEND_HOME_PATH}/bin/setenv.bash
export ASCEND_CUSTOM_OPP_PATH=${ASCEND_OPP_PATH}/vendors/custom_transformer
```

### Phase 2: 算子编译与安装

```bash
# 编译算子（必须使用正确的 SoC 版本）
bash build.sh --pkg --soc=<正确的soc版本> --ops=<算子名> -j8

# 安装算子包
./build/cann-ops-transformer-custom_linux-aarch64.run --quiet --nox11
```

### Phase 3: 测试代码编写

创建测试文件 `${算子目录}/examples/test_aclnn_${算子名}.cpp`

关键步骤：
1. 初始化 ACL (`aclInit`, `aclrtSetDevice`, `aclrtCreateStream`)
2. 创建输入 aclTensor (`aclCreateTensor`)
3. 获取 workspace 大小 (`aclnn${算子名}GetWorkspaceSize`)
4. 执行算子 (`aclnn${算子名}`)
5. 同步获取结果 (`aclrtSynchronizeStream`)
6. 释放资源

详细代码模板参见 reference/code_template.md

### Phase 4: 编译测试程序

```bash
source ${ASCEND_HOME_PATH}/bin/setenv.bash

ARCH_INFO=$(uname -m)
INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
EAGER_INCLUDE_OPP_ACLNNOP_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include/aclnnop"
EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/lib64"
CUST_INCLUDE_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/include"
CUST_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib"
TILING_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"

g++ ${算子目录}/examples/test_aclnn_${算子名}.cpp \
    -I ${CUST_INCLUDE_PATH} -I ${CUST_INCLUDE_PATH}/aclnnop \
    -I ${INCLUDE_PATH} -I ${EAGER_INCLUDE_OPP_ACLNNOP_PATH} \
    -L ${CUST_LIBRARY_PATH} -L ${EAGER_LIBRARY_PATH} \
    -lcust_opapi -lascendcl -lnnopbase -lc_sec \
    -Wl,-rpath=${CUST_LIBRARY_PATH} -Wl,-rpath=${EAGER_LIBRARY_PATH} \
    -o test_aclnn_${算子名} \
    -std=c++17 -O2
```

### Phase 5: 运行测试

```bash
export LD_LIBRARY_PATH=${TILING_LIBRARY_PATH}:${CUST_LIBRARY_PATH}:${EAGER_LIBRARY_PATH}:${LD_LIBRARY_PATH}
./test_aclnn_${算子名}
```

成功标志：
```
Workspace size: xxx bytes    ← GetWorkspaceSize 成功
Test completed successfully! ← 算子执行成功
```

## 最佳实践

### 输入数据准备

推荐使用 Python 生成测试数据，参考算子 UT 测试的默认参数：

```python
import torch
import torch_npu

torch_npu.npu.set_device(0)

# 使用 UT 测试的默认参数
B, seqlen, nk, nv, dk, dv = 2, 64, 4, 4, 64, 64
T = B * seqlen

q = torch.rand((T, nk, dk), dtype=torch.bfloat16, device='npu')
k = torch.rand((T, nk, dk), dtype=torch.bfloat16, device='npu')
v = torch.rand((T, nv, dv), dtype=torch.bfloat16, device='npu')

# L2 normalize（attention 算子必须）
q = torch.nn.functional.normalize(q, p=2, dim=-1)
k = torch.nn.functional.normalize(k, p=2, dim=-1)
```

### 参数对照

测试参数应与算子 UT 测试保持一致，参考 `tests/ut/op_kernel/test_${算子名}.cpp` 的默认参数。

### 精度验证流程

1. 运行 CPU golden（Python pytest）获取真值
2. 运行 NPU aclnn 测试获取实测值
3. 对比结果（计算相对误差、最大误差等指标）

## Reference 文件清单

- `reference/troubleshooting.md` - 常见问题排查指南
- `reference/soc_mapping.md` - SoC 类型映射表
- `reference/file_paths.md` - 关键文件路径模板
- `reference/error_codes.md` - 错误码参考
- `reference/code_template.md` - 测试代码完整模板
- `reference/example_solution.md` - chunk_gated_delta_rule 实例