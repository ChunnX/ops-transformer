# aclnnNlastTransposeNddmaBench

## 支持的产品型号

- Ascend 950PR / 950DT（arch35 / dav-3510）。

## 功能描述

- **算子功能**：对 5D 输入交换中间两轴 `permute[0,1,3,2,4]`，保留最内连续轴 `dim4`，
  用一条 UB 内 3D NDDMA 完成 `(dim2,dim3,dim4)→(dim3,dim2,dim4)` 的重排。
- **计算公式**：
  $$y[i_0,i_1,i_3,i_2,i_4] = x[i_0,i_1,i_2,i_3,i_4]$$
  纯索引重排、无数值运算，输出与 `x.permute(0,1,3,2,4)` bitwise 全等。

## 接口原型

两段式接口，必须先调用 `aclnnNlastTransposeNddmaBenchGetWorkspaceSize` 计算 workspace 大小并申请，
再调用 `aclnnNlastTransposeNddmaBench` 执行计算。

```cpp
aclnnStatus aclnnNlastTransposeNddmaBenchGetWorkspaceSize(
    const aclTensor* x, int64_t isMte3, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnNlastTransposeNddmaBench(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

## aclnnNlastTransposeNddmaBenchGetWorkspaceSize

| 参数 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|------|-----------|------|----------|----------|-------------|
| x | 输入 | device 侧输入张量 | FLOAT16 / FLOAT / INT8 | ND | 5D `(dim0,dim1,dim2,dim3,dim4)` |
| isMte3 | 输入(host) | `1`=转置结果写回 GM（精度对比，**默认**）；`0`=去掉 MTE3 写回（纯 NDDMA 读带宽） | INT64 | - | 标量 |
| y | 输出 | device 侧输出张量，dtype 与 x 一致 | FLOAT16 / FLOAT / INT8 | ND | 5D `(dim0,dim1,dim3,dim2,dim4)` |
| workspaceSize | 输出 | 返回需在 device 侧申请的 workspace 字节数 | uint64_t* | - | - |
| executor | 输出 | 返回 op 执行器 | aclOpExecutor** | - | - |

**返回值**：`aclnnStatus`。参数为空指针、dtype/format 不支持、x 与 y shape 不满足中间两轴互换关系时返回对应错误码。

## aclnnNlastTransposeNddmaBench

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
- `y` 的 shape 必须为 `x` 中间两轴互换、`dim4` 保留后的结果 `(dim0,dim1,dim3,dim2,dim4)`。
- `isMte3` 为 OPTIONAL attr，缺省 `1`；`0` 用于 benchmark 隔离纯读带宽（不产出有效 `y`）。
- 单块转置元素数 `dim2*dim3*dim4` 受 UB 容量约束（见 `docs/README.md` 的 UB 规划表）。

## 调用示例

核心两段式调用（完整流程与 last_transpose 示例同构，仅 shape 为 5D）：

```cpp
#include "aclnnop/aclnn_nlast_transpose_nddma_bench.h"

// x: (dim0,dim1,dim2,dim3,dim4)  y: (dim0,dim1,dim3,dim2,dim4)  dtype 一致
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
int64_t isMte3 = 1;   // 写回 GM 做精度比对

aclnnNlastTransposeNddmaBenchGetWorkspaceSize(x, isMte3, y, &workspaceSize, &executor);
void* workspaceAddr = nullptr;
if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclnnNlastTransposeNddmaBench(workspaceAddr, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

> PyTorch 直调（不经 aclnn）的 benchmark 路径见 [`pta/README.md`](../pta/README.md)。
