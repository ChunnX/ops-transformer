# MatmulAllReduceV2 算子

## 概述

MatmulAllReduceV2 是矩阵乘法与全Reduce融合算子的第二个版本。**功能与 MatmulAllReduce 完全一致**，但只保留三个核心接口。

## 核心特性

### 保留的接口

MatmulAllReduceV2 只保留以下三个接口：

1. **aclnn_matmul_all_reduce_v3** - 非量化矩阵乘法全Reduce接口
2. **aclnn_quant_matmul_all_reduce_v5** - 量化矩阵乘法全Reduce接口
3. **aclnn_weight_quant_matmul_all_reduce_v2** - 权重量化矩阵乘法全Reduce接口

### 代码复用策略

MatmulAllReduceV2 的实现策略：

- **功能一致性**: 与 MatmulAllReduce 功能完全相同
- **代码复用**: 通过引用原算子代码实现，避免代码重复
- **独立注册**: 注册为 `MatmulAllReduceV2`，与原算子独立

## 实现架构

### op_host 层

- `matmul_all_reduce_v2_def.cpp` - 算子定义，引用原算子定义逻辑
- `matmul_all_reduce_v2_infershape.cpp` - InferShape，引用原算子实现
- `matmul_all_reduce_v2_tiling.cpp` - Tiling入口，引用原算子tiling函数

### op_api 层

- `aclnn_matmul_all_reduce_v3.cpp` - 非量化接口，引用原算子实现
- `aclnn_quant_matmul_all_reduce_v5.cpp` - 量化接口，引用原算子实现
- `aclnn_weight_quant_matmul_all_reduce_v2.cpp` - 权重量化接口，引用原算子实现

### op_kernel 层

- `matmul_all_reduce_v2_kernel.cpp` - Kernel入口，引用原算子kernel实现
- `matmul_all_reduce_v2_tiling_key.h` - TilingKey定义，引用原算子定义

### op_graph 层

- `matmul_all_reduce_v2_proto.h` - 图算子定义
- `matmul_all_reduce_v2_gen_task.cpp` - 任务生成，引用原算子实现
- `fallback_matmul_all_reduce_v2.cpp` - Fallback实现，引用原算子实现

## 功能说明

### 算子功能

MatmulAllReduceV2 实现矩阵乘法与通信Reduce操作的融合：

```
Output = Reduce(Matmul(x1, x2) + bias + x3)
```

支持的Reduce操作：
- `sum`: 求和（目前唯一支持）
- `min`: 最小值（未来支持）
- `max`: 最大值（未来支持）
- `prod`: 乘积（未来支持）

### 数据类型支持

- **非量化模式**: FP16, BF16
- **量化模式**: INT8, FP8 (E4M3FN, E5M2), HIFLOAT8, FP4 (E2M1)
- **权重量化模式**: INT8, INT4, FP8, HIFLOAT8

### 架构支持

- Atlas A2 Training Series Product
- Atlas 800I A2 Inference Product  
- Atlas 200I A2 Box Heterogeneous Component
- 支持 1、2、4、8 卡配置

## 使用示例

### 非量化接口 (aclnn_matmul_all_reduce_v3)

```cpp
#include "aclnn_matmul_all_reduce_v3.h"

aclTensor* x1 = ...;  // [batch, m, k]
aclTensor* x2 = ...;  // [k, n]
aclTensor* bias = nullptr;  // Optional
aclTensor* x3 = nullptr;  // Optional
const char* group = "group0";
const char* reduceOp = "sum";
const char* commMode = "default";
int64_t commTurn = 0;
int64_t streamMode = 0;
aclTensor* output = ...;  // [batch, m, n]

// 调用接口
aclnnMatmulAllReduceV3GetWorkspaceSize(x1, x2, bias, x3, group, reduceOp, 
    commMode, commTurn, streamMode, output, &workspaceSize, &executor);
aclnnMatmulAllReduceV3(workspace, workspaceSize, executor, stream);
```

### 量化接口 (aclnn_quant_matmul_all_reduce_v5)

```cpp
#include "aclnn_quant_matmul_all_reduce_v5.h"

aclTensor* x1 = ...;  // INT8 [batch, m, k]
aclTensor* x2 = ...;  // INT8 [k, n]
aclTensor* x2Scale = ...;  // Dequant scale
aclTensor* bias = nullptr;  // Optional INT32
aclTensor* x3 = nullptr;  // Optional
aclTensor* output = ...;  // FP16/BF16 [batch, m, n]
const char* group = "group0";
const char* reduceOp = "sum";
int64_t streamMode = 0;

// 调用接口
aclnnQuantMatmulAllReduceV5GetWorkspaceSize(x1, x2, x2Scale, bias, x3, 
    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
    group, reduceOp, 0, 0, streamMode, output, &workspaceSize, &executor);
aclnnQuantMatmulAllReduceV5(workspace, workspaceSize, executor, stream);
```

### 权重量化接口 (aclnn_weight_quant_matmul_all_reduce_v2)

```cpp
#include "aclnn_weight_quant_matmul_all_reduce_v2.h"

aclTensor* x1 = ...;  // FP16/BF16 [batch, m, k]
aclTensor* x2 = ...;  // INT8/INT4 [k, n]
aclTensor* antiquantScale = ...;  // Antiquant scale
aclTensor* antiquantOffset = nullptr;  // Optional
aclTensor* bias = nullptr;  // Optional
aclTensor* x3 = nullptr;  // Optional
aclTensor* output = ...;  // FP16/BF16 [batch, m, n]
const char* group = "group0";
const char* reduceOp = "sum";
int64_t antiquantGroupSize = 0;
int64_t commTurn = 0;
int64_t streamMode = 0;

// 调用接口
aclnnWeightQuantMatmulAllReduceV2GetWorkspaceSize(x1, x2, antiquantScale, 
    antiquantOffset, bias, x3, group, reduceOp, antiquantGroupSize, 
    commTurn, streamMode, output, &workspaceSize, &executor);
aclnnWeightQuantMatmulAllReduceV2(workspace, workspaceSize, executor, stream);
```

## 与 MatmulAllReduce 的区别

| 特性 | MatmulAllReduce | MatmulAllReduceV2 |
|------|-----------------|-------------------|
| **功能** | 完整功能 | 完整功能（完全一致） |
| **接口数量** | 多个版本接口 | 只保留三个核心接口 |
| **代码策略** | 原始实现 | 引用原算子代码 |
| **注册名称** | MatmulAllReduce | MatmulAllReduceV2 |
| **代码复用** | 原始实现 | 复用原算子实现 |

## 构建与安装

### 构建步骤

```bash
cd mc2/matmul_all_reduce_v2
mkdir build && cd build
cmake ..
make
make install
```

### 编译依赖

- CANN Toolkit
- CANN Ops
- HCCL (通信库)
- CMake >= 3.16
- GCC >= 9.0

## 测试

### 单元测试 (UT)

```bash
cd tests/ut
./matmul_all_reduce_v3_ut
./quant_matmul_all_reduce_v5_ut
./weight_quant_matmul_all_reduce_v2_ut
```

### 系统测试 (ST)

```bash
cd tests/st
./matmul_all_reduce_v3_st
./quant_matmul_all_reduce_v5_st
./weight_quant_matmul_all_reduce_v2_st
```

## 性能优化

MatmulAllReduceV2 通过融合矩阵乘法和通信操作，显著降低通信开销：

- **减少通信次数**: 从两次通信（Matmul输出+Reduce）减少到一次
- **降低数据量**: Matmul输出直接用于通信，无需中间存储
- **流水优化**: Matmul计算与通信可以流水执行

## 注意事项

1. **通信组配置**: 需要正确配置HCCL通信组
2. **数据类型**: 确保输入输出数据类型匹配
3. **Shape约束**: k维度必须匹配
4. **性能优化**: 小批次场景可能性能不佳
5. **内存管理**: 需要足够的workspace内存

## 文档资源

- [算子设计文档](docs/DESIGN.md)
- [接口文档](docs/API.md)
- [测试文档](docs/TEST.md)
- [性能报告](docs/PERFORMANCE.md)

## 版本历史

- **v2.0** (2025-05): MatmulAllReduceV2 发布，保留三个核心接口

## 许可证

CANN Open Software License Agreement Version 2.0

## 联系方式

如有问题，请联系华为技术支持。