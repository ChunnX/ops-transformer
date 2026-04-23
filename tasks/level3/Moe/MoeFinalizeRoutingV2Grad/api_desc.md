# MoeFinalizeRoutingV2Grad 算子接口说明

## 功能描述
MoeFinalizeRoutingV2Grad 是 MoeFinalizeRoutingV2 的反向传播算子，用于计算 MoE 最终路由操作的梯度。

## 接口定义
```python
def moe_finalize_routing_v2_grad(
    gradY: Tensor,
    expandedRowIdx: Tensor,
    expandedXOptional: Tensor,
    scalesOptional: Tensor,
    expertIdxOptional: Tensor,
    biasOptional: Tensor,
    dropPadMode: int,
    activeNum: int,
    expertNum: int,
    expertCapacity: int
) -> Tuple[Tensor, Tensor]:
    """
    实现 moe_finalize_routing_v2_grad 操作，计算 MoE 最终路由的反向传播梯度。
    
    参数:
        gradY (Tensor): 输出梯度，Device侧的张量，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND，shape为(R, H)，其中R为batch_size，H为hidden_size。
        expandedRowIdx (Tensor): 扩展行索引，Device侧的张量，数据类型为INT32，数据格式支持ND，shape为(R*K)，其中K为topk。
        expandedXOptional (Tensor): 扩展输入X的可选张量，Device侧的张量，数据类型与gradY一致，数据格式支持ND，shape根据dropPadMode不同而不同：
            - dropPadMode=0（drop less）：shape为(activeNum, H)或(R*K, H)
            - dropPadMode=1（drop pad）：shape为(E, C, H)，其中E为expert_num，C为expert_capacity
        scalesOptional (Tensor): 缩放因子的可选张量，Device侧的张量，数据类型与gradY一致，数据格式支持ND，shape为(R, K)。
        expertIdxOptional (Tensor): 专家索引的可选张量，Device侧的张量，数据类型为INT32，数据格式支持ND，shape为(R, K)，当use_bias=1时有效。
        biasOptional (Tensor): 偏置的可选张量，Device侧的张量，数据类型与gradY一致，数据格式支持ND，shape为(E, H)，当use_bias=1时有效。
        dropPadMode (int): 丢弃模式，0表示drop less，1表示drop pad。
        activeNum (int): 激活数量，仅在dropPadMode=0时有效，当activeNum>0且activeNum<R*K时，只处理前activeNum个元素。
        expertNum (int): 专家数量，仅在dropPadMode=1时有效。
        expertCapacity (int): 专家容量，仅在dropPadMode=1时有效。
    
    返回值:
        Tuple[Tensor, Tensor]: 
            - gradExpandedXOut (Tensor): expandedXOptional的梯度，Device侧的张量，数据类型与gradY一致，数据格式支持ND，shape与expandedXOptional相同。
            - gradScalesOut (Tensor): scalesOptional的梯度，Device侧的张量，数据类型与gradY一致，数据格式支持ND，shape与scalesOptional相同。
    """
```

## 分块策略
MoeFinalizeRoutingV2Grad 算子根据以下条件选择不同的分块分支：

1. **数据类型（dtype）**：
   - float16
   - bfloat16
   - float32

2. **丢弃模式（dropPadMode）**：
   - 0：drop less
   - 1：drop pad

3. **专家数量（expertNum）**：
   - 0
   - > 0

4. **激活数量（activeNum）**：
   - 0
   - > 0

## 使用示例
```python
import torch
import torch_npu
import kernel_gen_ops

# 设置设备
device = torch.device('npu:0')

# 准备输入数据
batch_size = 16
hidden_size = 256
k = 2
use_bias = 1
dropPadMode = 0
active_num = 0
expert_num = 0
expert_capacity = 0

gradY = torch.randn(batch_size, hidden_size, dtype=torch.float16, device=device)
expandedRowIdx = torch.randperm(batch_size * k, dtype=torch.int32, device=device)
expandedXOptional = torch.randn(batch_size * k, hidden_size, dtype=torch.float16, device=device)
scalesOptional = torch.randn(batch_size, k, dtype=torch.float16, device=device)
expertIdxOptional = torch.randint(0, expert_num, (batch_size, k), dtype=torch.int32, device=device) if use_bias else torch.zeros((0, 0), dtype=torch.int32, device=device)
biasOptional = torch.randn(expert_num, hidden_size, dtype=torch.float16, device=device) if use_bias else torch.zeros((0, 0), dtype=torch.float16, device=device)

# 调用算子
gradExpandedXOut, gradScalesOut = kernel_gen_ops.moe_finalize_routing_v2_grad(
    gradY, expandedRowIdx, expandedXOptional, scalesOptional, expertIdxOptional, biasOptional,
    dropPadMode, activeNum, expertNum, expertCapacity
)

print(f"gradExpandedXOut shape: {gradExpandedXOut.shape}")
print(f"gradScalesOut shape: {gradScalesOut.shape}")
```

## 注意事项
1. 输入张量必须都在 NPU 设备上。
2. expandedXOptional、scalesOptional、expertIdxOptional、biasOptional 为可选张量，但在实际使用中需要根据具体情况提供。
3. dropPadMode、activeNum、expertNum、expertCapacity 参数需要与前向计算时保持一致。
4. 当 use_bias=0 时，expertIdxOptional 和 biasOptional 可以为空张量，但数据类型需要与其他输入张量保持一致。
5. 对于 bfloat16 数据类型，由于 numpy 不支持，在 CPU 基准测试中会使用 float32 进行计算，然后转换回 bfloat16。

## 精度说明
| 数据类型 | 相对误差阈值 | 绝对误差阈值 | 总绝对误差阈值 |
|----------|--------------|--------------|----------------|
| float32  | 1e-5         | 1e-5         | 1e-4           |
| float16  | 1e-3         | 1e-3         | 1e-3           |
| bfloat16 | 1e-2         | 1e-2         | 1e-2           |

## 依赖说明
- 依赖 PyTorch 和 torch_npu 库。
- 依赖 Ascend NPU 硬件和相关驱动。
- 依赖 ACLNN 库提供的底层实现。