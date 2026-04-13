# aclnnMoeInitRoutingQuantV2

## 功能描述

### 算子功能
该算子对应MoE（Mixture of Experts，混合专家模型）中的Routing计算，以aclnnMoeGatingTopKSoftmax算子的计算结果作为输入，并输出量化后的Routing矩阵expandedXOut等结果供后续计算使用。

### 计算公式

1. 将输入shape为[NUM_ROWS, K]的expertIdx展平为一行做排序，其中NUM_ROWS为输入token个数，K为token选择的专家个数，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

  $$
sortedExpertIdx, sortedRowIdx=keyValueSort(\text{flatten}(expertIdx))
$$

2. 以sortedRowIdx做位置映射得出expandedRowIdxOut：

  $$
expandedRowIdxOut[sortedRowIdx[i]]=i
$$

3. 在dropless模式下，对sortedExpertIdx的每个专家统计直方图结果，再进行Cumsum，得出expertTokensCountOrCumsumOutOptional：

  $$
expertTokensCountOrCumsumOutOptional[i]=Cumsum(Histogram(sortedExpertIdx))
$$

4. 在drop模式下，对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensBeforeCapacityOutOptional：

  $$
expertTokensBeforeCapacityOutOptional[i]=Histogram(sortedExpertIdx)
$$

5. 计算quant结果：

  - 静态quant：

    $$
    quantResult = round((x * scaleOptional) + offsetOptional)
    $$

  - 动态quant：
    - 若不输入scale：

      $$
      dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
      $$

      $$
      quantResult = round(x / dynamicQuantScaleOutOptional)
      $$

    - 若输入scale:

      $$
      dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
      $$

      $$
      quantResult = round(x * scaleOptional / dynamicQuantScaleOutOptional)
      $$

6. 根据quantResult得出expandedXOut：

  $$
expandedXOut[expandedRowIdxOut[i]]=quantResult[i // K]
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_init_routing_quant_v2()` 函数形式提供：

```python
def moe_init_routing_quant_v2(x: Tensor, 
                             expert_idx: Tensor, 
                             scale_optional: Optional[Tensor] = None, 
                             offset_optional: Optional[Tensor] = None, 
                             active_num: int = 0, 
                             expert_capacity: int = 0, 
                             expert_num: int = 0, 
                             drop_pad_mode: int = 0, 
                             expert_tokens_count_or_cumsum_flag: int = 0, 
                             expert_tokens_before_capacity_flag: bool = False, 
                             quant_mode: int = 0):
    """
    实现 MoE 中 Routing 的量化计算，生成量化后的路由矩阵供后续计算使用。

    参数:
        x (Tensor): MOE的输入，即token特征输入，形状为[NUM_ROWS, H]
        expert_idx (Tensor): 每个Token对应的K个处理专家的序号，形状为[NUM_ROWS, K]
        scale_optional (Optional[Tensor]): 用于计算quant结果的参数，静态quant场景下必须输入
        offset_optional (Optional[Tensor]): 用于计算quant结果的偏移值，静态quant场景下必须输入
        active_num (int): 表示是否为Active场景，值范围大于等于0
        expert_capacity (int): 表示每个专家能够处理的tokens数，值范围大于等于0
        expert_num (int): 表示专家数，值范围大于等于0
        drop_pad_mode (int): 表示是否为Drop/Pad场景，取值为0或1
        expert_tokens_count_or_cumsum_flag (int): 控制是否输出expertTokensCountOrCumsumOutOptional，取值为0、1和2
        expert_tokens_before_capacity_flag (bool): 控制是否输出expertTokensBeforeCapacityOutOptional
        quant_mode (int): 表示quant计算模式，取值为0（静态quant）和1（动态quant）

    返回:
        Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]]: 包含五个张量的元组
            - expanded_x_out: 根据expert_idx进行扩展过的量化特征
            - expanded_row_idx_out: expanded_x_out和x的索引映射关系
            - expert_tokens_count_or_cumsum_out_optional: 输出每个专家处理的token数量的统计结果及累加值
            - expert_tokens_before_capacity_out_optional: 输出drop之前每个专家处理的token数量的统计结果
            - dynamic_quant_scale_out_optional: 输出动态quant计算过程中的中间值
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 场景1：静态quant，Dropless模式
# 创建输入张量
num_rows = 32
k = 4
h = 512
expert_num = 16

x = torch.randn(num_rows, h, dtype=torch.float16)
expert_idx = torch.randint(0, expert_num, (num_rows, k), dtype=torch.int32)
scale = torch.tensor([0.1], dtype=torch.float32)
offset = torch.tensor([0.0], dtype=torch.float32)

# 执行计算
expanded_x_out, expanded_row_idx_out, expert_tokens_count_or_cumsum, expert_tokens_before_capacity, dynamic_scale = kernel_gen_ops.moe_init_routing_quant_v2(
    x, expert_idx, scale, offset, 0, 0, expert_num, 0, 1, False, 0
)

print(f"expanded_x_out shape: {expanded_x_out.shape}")
print(f"expanded_row_idx_out shape: {expanded_row_idx_out.shape}")
print(f"expert_tokens_count_or_cumsum shape: {expert_tokens_count_or_cumsum.shape}")
print(f"expert_tokens_before_capacity: {expert_tokens_before_capacity}")  # 应为None
print(f"dynamic_scale: {dynamic_scale}")  # 应为None

# 场景2：动态quant，Drop/Pad模式
num_rows = 64
k = 8
h = 1024
expert_num = 32
expert_capacity = 16

x = torch.randn(num_rows, h, dtype=torch.float32)
expert_idx = torch.randint(0, expert_num, (num_rows, k), dtype=torch.int32)

# 执行计算
expanded_x_out, expanded_row_idx_out, expert_tokens_count_or_cumsum, expert_tokens_before_capacity, dynamic_scale = kernel_gen_ops.moe_init_routing_quant_v2(
    x, expert_idx, None, None, 0, expert_capacity, expert_num, 1, 0, True, 1
)

print(f"expanded_x_out shape: {expanded_x_out.shape}")
print(f"expanded_row_idx_out shape: {expanded_row_idx_out.shape}")
print(f"expert_tokens_count_or_cumsum: {expert_tokens_count_or_cumsum}")  # 应为None
print(f"expert_tokens_before_capacity shape: {expert_tokens_before_capacity.shape}")
print(f"dynamic_scale shape: {dynamic_scale.shape}")
```

## 约束与限制

- 输入张量x数据类型支持：FLOAT16、BFLOAT16、FLOAT32
- 输入张量expert_idx数据类型支持：INT32
- 输入张量scale_optional和offset_optional数据类型支持：FLOAT32
- 输出张量expanded_x_out数据类型支持：INT8、INT4
- 输出张量expanded_row_idx_out数据类型支持：INT32
- 输出张量expert_tokens_count_or_cumsum_out_optional数据类型支持：INT32
- 输出张量expert_tokens_before_capacity_out_optional数据类型支持：INT32
- 输出张量dynamic_quant_scale_out_optional数据类型支持：FLOAT32
- active_num、expert_num、expert_capacity的值范围大于等于0
- drop_pad_mode的取值为0或1
- expert_tokens_count_or_cumsum_flag的取值为0、1和2
- expert_tokens_before_capacity_flag的取值为False和True
- quant_mode的取值为0（静态quant）和1（动态quant）
- 静态quant场景下必须输入scale_optional和offset_optional
- 动态quant场景下scale_optional为可选输入
- Drop/Pad场景下expertCapacity的值域范围(0, NUM_ROWS]
- 当expanded_x_out的数据类型为INT4时：
  - dropPadMode仅支持0
  - H必须可以被2整除
  - 传入scale_optional时scale_optional的shape仅支持为[1, H]
- Ascend 950PR/Ascend 950DT：输出expanded_x_out数据类型仅支持INT8
- 确定性计算：aclnnMoeInitRoutingQuantV2默认确定性实现
