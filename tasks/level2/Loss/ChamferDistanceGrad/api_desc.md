# aclnnChamferDistanceGrad

## 功能描述

### 算子功能
该AscendC算子用于计算ChamferDistance（倒角距离）的反向，根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。


### 计算公式
假设有两个点集：  xyz1=[B,N,2], xyz2=[B,M,2]

  - ChamferDistance（倒角距离）正向算子计算公式为：

    $dist1_i=Min((x_{1_i}−x_2)^2+(y_{1_i}−y_2)^2)，x_2, y_2∈xyz2$
    $dist2_i=Min((x_{2_i}-x_1)^2+(y_{2_i}-y_1)^2)，x_1,y_1∈xyz1$

  - 反向算子即为对该公式求导，计算公式为：
    - $dist1_i 对x_{1_i} $的导数$=2*grad\_dist1*(x_{1_i}-x_2)$

      其中：$x_{1_i}∈xyz1$，$x_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的横坐标，单点求导公式如上，因为单点梯度更新的位置是连续的，所以考虑多点并行计算。

    - $dist1_i 对y_{1_i} $的导数$=2*grad\_dist1*(y_{1_i}-y_2)$

      其中$y_{1_i}∈xyz1$，$y_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的纵坐标，单点求导公式如上，因为单点梯度更新的位置是连续的，所以也可以考虑多点并行计算。

    - $dist1_i 对x_2 $的导数$=-2*grad\_dist1*(x_{1_i}-x_2)$

      其中$x_{1_i}∈xyz1，x_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的横坐标，单点求导公式如上，因为单点梯度需要根据最小距离值对应的索引值去更新，所以这块无法并行只能单点计算。

    - $dist1_i 对y_2 $的导数$=-2*grad\_dist1*(y_{1_i}-y_2)$

      其中$y_{1_i}∈xyz1，y_2$是根据正向输出的id1的索引值从xyz2中取出距离最小的点的纵坐标，单点求导公式如上，因为单点梯度需要根据最小值对应的索引值去更新，所以这块也无法并行只能单点计算。

对应$dist2_i$对$x_{2_i} 、x_1、y_{2_i} 、y_1$的导数和上述过程类似，这里不再赘述。

最终计算公式如下，i∈[0,n)：

$grad_xyz1[2*i] = 2*grad\_dist_1*(x_{1_i}-x_2) - 2*grad\_dist_1*(x_{1_i}-x_2)$

$grad_xyz1[2*i+1] = 2*grad\_dist1*(y_{1_i}-y_2) - 2*grad\_dist1*(y_{1_i}-y_2)$

$grad_xyz2[2*i] = 2*grad\_dist2*(x_{1_i}-x_2) - 2*grad\_dist2*(x_{1_i}-x_2)$

$grad_xyz2[2*i+1] = 2*grad\_dist2*(y_{1_i}-y_2) - 2*grad\_dist2*(y_{1_i}-y_2)$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.chamfer_distance_grad()` 函数形式提供：

```python
def chamfer_distance_grad(predict, label, reduction='mean'):
    """
    计算预测值和目标值之间的均方误差损失。
    
    参数:
        xyz1（Tensor): 点集1的坐标张量，形状为 [B, N, 2]，表示 B 个批次中每个批次包含 N 个二维点，数据类型支持 float32 或 float16；
        xyz2（Tensor): 点集2的坐标张量，形状为 [B, N, 2]，表示 B 个批次中每个批次包含 N 个二维点，数据类型与 xyz1 一致；
        idx1（Tensor): 正向传播中，点集1中每个点到点集2的最近点的索引，形状为 [B, N]，数据类型为 int32；
        idx2（Tensor): 正向传播中，点集2中每个点到点集1的最近点的索引，形状为 [B, N]，数据类型为 int32；
        gradDist1: 正向输出 dist1 的梯度（即上游反向梯度），形状为 [B, N]，与 xyz1、xyz2 数据类型一致；
        gradDist2: 正向输出 dist2 的梯度（即上游反向梯度），形状为 [B, N]，与 xyz1、xyz2 数据类型一致；
        
    返回:
        grad_xyz1（Tensor): 输入 xyz1 对应的梯度，形状为 [B, N, 2]，表示每个点的坐标梯度，用于反向传播更新；
        grad_xyz2（Tensor): 输入 xyz2 对应的梯度，形状为 [B, N, 2]，表示每个点的坐标梯度，用于反向传播更新；
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops
import numpy as np

# 创建输入张量
np.random.seed(1234)
xyz1_shape = [2,3,2]
xyz2_shape = [2,4,2]
dtype = torch.float32
xyz1 = torch.rand(xyz1_shape, dtype=dtype).cpu().numpy()
xyz2 = torch.rand(xyz2_shape, dtype=dtype).cpu().numpy()
idx1 = np.random.randint(0, xyz2_shape[1], size=idx1_shape, dtype=np.int32)
idx2 = np.random.randint(0, xyz1_shape[1], size=idx2_shape, dtype=np.int32)

gradDist1 = torch.rand(xyz1_shape[:-1], dtype=dtype).cpu().numpy()
gradDist2 = torch.rand(xyz2_shape[:-1], dtype=dtype).cpu().numpy()

# 3. 转为 tensor（用于 golden）
xyz1_t = torch.from_numpy(xyz1).to(device).to(dtype)
xyz2_t = torch.from_numpy(xyz2).to(device).to(dtype)
idx1_t = torch.from_numpy(idx1).to(device).to(torch.int32)
idx2_t = torch.from_numpy(idx2).to(device).to(torch.int32)
gradDist1_t = torch.from_numpy(gradDist1).to(device).to(dtype)
gradDist2_t = torch.from_numpy(gradDist2).to(device).to(dtype)


# 使用gelu执行计算
result = kernel_gen_ops.chamfer_distance_grad(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2)
```

## 约束与限制

- 无