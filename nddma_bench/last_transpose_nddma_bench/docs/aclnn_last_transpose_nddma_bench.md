# aclnnLastTransposeNddmaBench

## 支持的产品型号

- Ascend 950PR / 950DT（arch35 / dav-3510）。

## 功能描述

- **算子功能**：对 4D 输入做最后两轴转置 `permute[0,1,3,2]`，用一条 UB 内 NDDMA 完成
  `[dim2,dim3]→[dim3,dim2]` 的重排。
- **计算公式**：
  $$y[i_0,i_1,i_3,i_2] = x[i_0,i_1,i_2,i_3]$$
  纯索引重排、无数值运算，输出与 `x.transpose(-1,-2)` bitwise 全等。

## 接口原型

两段式接口，必须先调用 `aclnnLastTransposeNddmaBenchGetWorkspaceSize` 计算 workspace 大小并申请，
再调用 `aclnnLastTransposeNddmaBench` 执行计算。

```cpp
aclnnStatus aclnnLastTransposeNddmaBenchGetWorkspaceSize(
    const aclTensor* x, int64_t isMte3, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnLastTransposeNddmaBench(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

## aclnnLastTransposeNddmaBenchGetWorkspaceSize

| 参数 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|------|-----------|------|----------|----------|-------------|
| x | 输入 | device 侧输入张量 | FLOAT16 / FLOAT / INT8 | ND | 4D `(dim0,dim1,dim2,dim3)` |
| isMte3 | 输入(host) | `1`=转置结果写回 GM（精度对比，**默认**）；`0`=去掉 MTE3 写回（纯 NDDMA 读带宽） | INT64 | - | 标量 |
| y | 输出 | device 侧输出张量，dtype 与 x 一致 | FLOAT16 / FLOAT / INT8 | ND | 4D `(dim0,dim1,dim3,dim2)` |
| workspaceSize | 输出 | 返回需在 device 侧申请的 workspace 字节数 | uint64_t* | - | - |
| executor | 输出 | 返回 op 执行器 | aclOpExecutor** | - | - |

**返回值**：`aclnnStatus`。参数为空指针、dtype/format 不支持、x 与 y shape 不满足转置关系时返回对应错误码。

## aclnnLastTransposeNddmaBench

| 参数 | 输入/输出 | 描述 | 数据类型 |
|------|-----------|------|----------|
| workspace | 输入 | device 侧 workspace 内存起址 | void* |
| workspaceSize | 输入 | workspace 字节数（由第一段接口返回） | uint64_t |
| executor | 输入 | op 执行器（由第一段接口返回） | aclOpExecutor* |
| stream | 输入 | acl stream 流 | aclrtStream |

**返回值**：`aclnnStatus`。

## 约束与限制

- 仅支持 arch35（Ascend 950PR/950DT），格式仅 ND。
- `x`、`y` dtype 必须一致，取值 ∈ {FLOAT16, FLOAT, INT8}。
- `y` 的 shape 必须为 `x` 末两轴互换后的结果 `(dim0,dim1,dim3,dim2)`。
- `isMte3` 为 OPTIONAL attr，缺省 `1`；`0` 用于 benchmark 隔离纯读带宽（不产出有效 `y`）。
- 单块转置元素数 `dim2*dim3` 受 UB 容量约束（见 `docs/README.md` 的 UB 规划表）。

## 调用示例

完整可编译示例见 [`examples/test_aclnn_last_transpose_nddma_bench.cpp`](../examples/test_aclnn_last_transpose_nddma_bench.cpp)
（自造输入 + host golden 逐字节比对）。核心两段式调用：

```cpp
#include "aclnnop/aclnn_last_transpose_nddma_bench.h"

// x: (dim0,dim1,dim2,dim3)  y: (dim0,dim1,dim3,dim2)  dtype 一致
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
int64_t isMte3 = 1;   // 写回 GM 做精度比对

aclnnLastTransposeNddmaBenchGetWorkspaceSize(x, isMte3, y, &workspaceSize, &executor);
void* workspaceAddr = nullptr;
if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclnnLastTransposeNddmaBench(workspaceAddr, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

> PyTorch 直调（不经 aclnn）的 benchmark 路径见 [`pta/README.md`](../pta/README.md)。
