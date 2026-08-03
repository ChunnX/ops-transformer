# aclnnBroadcastNddmaBench

## 支持的产品型号

- Ascend 950PR / 950DT（arch35 / dav-3510）。

## 功能描述

- **算子功能**：沿广播轴复制搬运，用一条 UB 内 3D NDDMA（广播轴 `srcStride=0`）把广播轴长度=1 的
  5D 输入展开成满 5D 输出。广播模式由 `scene` 选择（AB/BA/BAB/ABA）。
- **计算等价**：输出与 `torch.broadcast_to(x, (dim0,dim1,dim2,dim3,dim4))` bitwise 全等。
  输入广播轴长度=1、无法从 input 恢复真实长度，故满 5D 的 `dim2/dim3/dim4` 由 attr 传入。

### 广播场景（scene）

| scene | 值 | 广播轴 | 输入广播轴长度=1 的形态 |
|-------|----|--------|-------------------------|
| AB | 0 | dim4 | `(dim0,dim1,dim2,dim3,1)` |
| BA | 1 | dim3 | `(dim0,dim1,dim2,1,dim4)` |
| BAB | 2 | dim2+dim4 | `(dim0,dim1,1,dim3,1)` |
| ABA | 3 | dim3（≡ BA，见 kernel §9.1） | `(dim0,dim1,dim2,1,dim4)` |

## 接口原型

两段式接口，必须先调用 `aclnnBroadcastNddmaBenchGetWorkspaceSize` 计算 workspace 大小并申请，
再调用 `aclnnBroadcastNddmaBench` 执行计算。

```cpp
aclnnStatus aclnnBroadcastNddmaBenchGetWorkspaceSize(
    const aclTensor* x, int64_t isMte3, int64_t scene,
    int64_t dim2, int64_t dim3, int64_t dim4, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnBroadcastNddmaBench(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

## aclnnBroadcastNddmaBenchGetWorkspaceSize

| 参数 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|------|-----------|------|----------|----------|-------------|
| x | 输入 | device 侧输入张量，广播轴长度=1 的 5D | FLOAT16 / FLOAT / INT8 | ND | 5D（随 scene，见上表） |
| isMte3 | 输入(host) | `1`=广播结果写回 GM（精度对比，**默认**）；`0`=去掉 MTE3 写回（纯 NDDMA 读带宽） | INT64 | - | 标量 |
| scene | 输入(host) | 广播场景 `0=AB / 1=BA / 2=BAB / 3=ABA`，缺省 `0` | INT64 | - | 标量 |
| dim2 | 输入(host) | 输出满 5D 的 `dim2`（NDDMA 最外轴），缺省 `0` | INT64 | - | 标量 |
| dim3 | 输入(host) | 输出满 5D 的 `dim3`（NDDMA 中轴），缺省 `0` | INT64 | - | 标量 |
| dim4 | 输入(host) | 输出满 5D 的 `dim4`（NDDMA 最内连续轴），缺省 `0` | INT64 | - | 标量 |
| y | 输出 | device 侧输出张量，dtype 与 x 一致 | FLOAT16 / FLOAT / INT8 | ND | 5D `(dim0,dim1,dim2,dim3,dim4)` |
| workspaceSize | 输出 | 返回需在 device 侧申请的 workspace 字节数 | uint64_t* | - | - |
| executor | 输出 | 返回 op 执行器 | aclOpExecutor** | - | - |

> **attr 顺序**：`is_mte3, scene, dim2, dim3, dim4`（即 aclnn 入参顺序，也是 kernel tiling 的 attr index 顺序）。
> 输出 shape 的 `dim0/dim1` 取自输入（广播从不折叠这两维），`dim2/dim3/dim4` 取自上述 attr。

**返回值**：`aclnnStatus`。参数为空指针、dtype/format 不支持、scene 非法、attr 维度与输出不符时返回对应错误码。

## aclnnBroadcastNddmaBench

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
- kernel Init 三断言（host 配错会触发）：`dim2/dim3/dim4 ≥ 2`；`dim4*sizeof(T)` 须 8B 对齐；
  单条 3D tile `dim2*dim3*dim4 ≤ DB halfCap`（见 `docs/README.md`）。
- `isMte3` 为 OPTIONAL attr，缺省 `1`；`0` 用于 benchmark 隔离纯读带宽（不产出有效 `y`）。

## 调用示例

核心两段式调用（以 scene=AB、输出 `(dim0,dim1,dim2,dim3,dim4)` 为例）：

```cpp
#include "aclnnop/aclnn_broadcast_nddma_bench.h"

// x: 广播轴=1 的 5D（scene=AB 时 (dim0,dim1,dim2,dim3,1)）；y: 满 5D (dim0,dim1,dim2,dim3,dim4)
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
int64_t isMte3 = 1, scene = 0;   // AB：dim4 广播

aclnnBroadcastNddmaBenchGetWorkspaceSize(x, isMte3, scene, dim2, dim3, dim4, y,
                                         &workspaceSize, &executor);
void* workspaceAddr = nullptr;
if (workspaceSize > 0) aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclnnBroadcastNddmaBench(workspaceAddr, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);
```

> PyTorch 直调（不经 aclnn）的 benchmark 路径见 [`pta/README.md`](../pta/README.md)。
