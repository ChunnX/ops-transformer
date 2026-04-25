# aclnnAllGatherMatmul

## 功能描述

### 算子功能

实现 AllGather 通信与 MatMul 计算融合，在分布式环境下高效完成矩阵乘法前的 AllGather 通信操作。

### 计算公式

$$
y = AllGather(x1) @ x2 + bias
$$

$$
gatherOut = AllGather(x1)
$$

其中：
- `AllGather(x1)` 将每卡的 x1 矩阵沿第 0 维拼接，得到 `(m * rank_size, k)` 的矩阵
- `@` 表示矩阵乘法
- `bias` 为可选偏置向量

## 接口定义

### Python 接口

通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.all_gather_matmul()` 函数形式提供：

```python
def all_gather_matmul(
    x1: Tensor,
    x2: Tensor,
    bias: Optional[Tensor],
    group: str,
    gather_index: int,
    comm_turn: int,
    stream_mode: int
) -> List[Tensor]:
    """
    实现 AllGather + MatMul 融合操作。

    参数:
        x1 (Tensor): matmul 左矩阵，shape (m, k)，dtype float16/bfloat16
        x2 (Tensor): matmul 右矩阵，shape (k, n)，dtype float16/bfloat16
        bias (Optional[Tensor]): 可选偏置，shape (n,)
        group (str): 通信域名称，通过 HcclGetCommName 获取
        gather_index (int): gather 目标，0 表示 x1，1 表示 x2，当前仅支持 0
        comm_turn (int): 通信数据切分数，当前仅支持 0
        stream_mode (int): 须与 CANN 侧 `NUM_ACL_STOP_ON_FAILURE` 一致，**当前有效取值为 1**（`aclnn_all_gather_matmul` 的 `CheckAttr`）

    返回:
        List[Tensor]: [output, gatherOut]
            - output: 计算结果，shape (m * rank_size, n)
            - gatherOut: AllGather 结果，shape (m * rank_size, k)
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

m, k, n = 1024, 1024, 1024
x1 = torch.randn((m, k), dtype=torch.float16)
x2 = torch.randn((k, n), dtype=torch.float16)

output, gather_out = kernel_gen_ops.all_gather_matmul(
    x1, x2, None, "hccl_world_group", 0, 0, 1
)
```

## 约束与限制

- **本文档仅描述算子功能、接口与数据约束**
- x1、x2 支持 FLOAT16、BFLOAT16 数据类型
- x1 为 2 维，shape (m, k)；x2 为 2 维，shape (k, n)
- k 轴取值范围为 [256, 65535)
- x1、x2、output、gatherOut 数据类型需保持一致
- bias 可选，当前版本仅支持一维输入
- gather_index 当前版本仅支持 0
- comm_turn 当前版本仅支持 0
- 输出 shape: output 为 (m * rank_size, n)，gatherOut 为 (m * rank_size, k)
- 支持 2、4、8、16、32、64 卡（根据平台不同）