# aclnnMoeInitRoutingV2

## 功能描述

### 算子功能
该算子对应MoE（Mixture of Experts，混合专家模型）中的Routing计算，以aclnnMoeGatingTopKSoftmax算子的计算结果作为输入，并输出Routing矩阵expandedX等结果供后续计算使用。

### 计算公式

1. 将输入shape为[numRows, k]或[numRows]的expertIdx展平为一行做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx，其中numRows为token个数，k表示token数据元素对应权重最高的k个专家，当expertIdx为1维时记k为1：

  $$
sortedExpertIdx, sortedRowIdx=keyValueSort(\text{flatten}(expertIdx))
$$

2. 以sortedRowIdx做位置映射得出expandedRowIdx：

  $$
expandedRowIdx[sortedRowIdx[i]]=i
$$

3. 按照sortedRowIdx将token按专家顺序排列，在dropPadMode为1时将每个专家需要处理的Token个数对齐为expertCapacity个，超过expertCapacity个的Token会被Drop，不足的会用0填充。得出expandedX：

  $$
expandedX[expandedRowIdx[i]]=x[i//k]
$$

4. 对sortedExpertIdx的每个专家统计直方图结果，再进行Cumsum，得出expertTokensCountOrCumsum：

  $$
expertTokensCountOrCumsum[i]=Cumsum(Histogram(sortedExpertIdx))
$$

5. 对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensBeforeCapacity：

  $$
expertTokensBeforeCapacity[i]=Histogram(sortedExpertIdx)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_init_routing_v2()` 函数形式提供：

```python
def moe_init_routing_v2(x: Tensor, 
                       expertIdx: Tensor, 
                       activeNum: int, 
                       expertCapacity: int, 
                       expertNum: int, 
                       dropPadMode: int, 
                       expertTokensCountOrCumsumFlag: int, 
                       expertTokensBeforeCapacityFlag: bool):
    """
    实现 MoE 中的 Routing 计算，生成路由矩阵供后续计算使用。

    参数:
        x (Tensor): MOE的输入，即token特征输入，形状为[numRows, h]
        expertIdx (Tensor): 为每个Token对应的k个处理专家的序号，形状为[numRows, k]或[numRows]
        activeNum (int): 表示是否为Active场景，值范围大于等于0，0表示Dropless场景，大于0时表示Active场景
        expertCapacity (int): 表示每个专家能够处理的tokens数，值范围大于等于0
        expertNum (int): 表示专家总数，值范围大于等于0
        dropPadMode (int): 表示是否为Drop/Pad场景，取值为0或1
        expertTokensCountOrCumsumFlag (int): 控制是否输出expertTokensCountOrCumsum，取值为0、1和2
        expertTokensBeforeCapacityFlag (bool): 控制是否输出expertTokensBeforeCapacity

    返回:
        Tuple[Tensor, Tensor, Tensor, Tensor]: 包含四个张量的元组
            - expandedX: 根据expertIdx进行扩展过的特征
            - expandedRowIdx: expandedX和x的索引映射关系
            - expertTokensCountOrCumsum: 输出每个专家处理的token数量的统计结果及累加值
            - expertTokensBeforeCapacity: 输出drop之前每个专家处理的token数量的统计结果
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 场景1：Dropless模式，输出expertTokensCountOrCumsum
# 创建输入张量
numRows = 32
h = 512
k = 4
expertNum = 16

x = torch.randn(numRows, h, dtype=torch.float16)
expertIdx = torch.randint(0, expertNum, (numRows, k), dtype=torch.int32)

# 执行计算
expandedX, expandedRowIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity = kernel_gen_ops.moe_init_routing_v2(
    x, expertIdx, 0, 0, expertNum, 0, 1, False
)

print(f"expandedX shape: {expandedX.shape}")
print(f"expandedRowIdx shape: {expandedRowIdx.shape}")
print(f"expertTokensCountOrCumsum shape: {expertTokensCountOrCumsum.shape}")
print(f"expertTokensBeforeCapacity shape: {expertTokensBeforeCapacity.shape}")

# 场景2：Drop/Pad模式，输出expertTokensBeforeCapacity
numRows = 64
h = 1024
k = 8
expertNum = 32
expertCapacity = 16

x = torch.randn(numRows, h, dtype=torch.float32)
expertIdx = torch.randint(0, expertNum, (numRows, k), dtype=torch.int32)

# 执行计算
expandedX, expandedRowIdx, expertTokensCountOrCumsum, expertTokensBeforeCapacity = kernel_gen_ops.moe_init_routing_v2(
    x, expertIdx, 0, expertCapacity, expertNum, 1, 0, True
)

print(f"expandedX shape: {expandedX.shape}")
print(f"expandedRowIdx shape: {expandedRowIdx.shape}")
print(f"expertTokensCountOrCumsum shape: {expertTokensCountOrCumsum.shape}")
print(f"expertTokensBeforeCapacity shape: {expertTokensBeforeCapacity.shape}")
```

## 约束与限制

- 输入张量x数据类型支持：FLOAT16、BFLOAT16、FLOAT32
- 输入张量expertIdx数据类型支持：INT32、INT64
- expertIdx的形状支持：1D或2D
- activeNum、expertNum、expertCapacity的值范围大于等于0
- dropPadMode的取值为0或1
- expertTokensCountOrCumsumFlag的取值为0、1和2
- expertTokensBeforeCapacityFlag的取值为false和true
- 在Drop/Pad场景下或者非Drop/Pad场景下且需要输出expertTokensCountOrCumsum时，要求expertIdx值域范围是[0, expertNum - 1]
- 在Drop/Pad场景下，expertCapacity的值域范围(0, numRows]
- 在Drop/Pad场景下或者expertTokensCountOrCumsumFlag大于0需要输出expertTokensCountOrCumsum时，expertNum需大于0
- 输出张量expandedX的数据类型需与x相同
- 输出张量expandedRowIdx的数据类型为INT32
- 输出张量expertTokensCountOrCumsum的数据类型为INT32
- 输出张量expertTokensBeforeCapacity的数据类型为INT32
- 确定性计算：aclnnMoeInitRoutingV2默认确定性实现
