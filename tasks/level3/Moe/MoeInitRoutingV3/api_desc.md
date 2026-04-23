# aclnnMoeInitRoutingV3

## 功能描述

### 算子功能
该算子实现MoE（Mixture of Experts）模型的routing计算，根据aclnnMoeGatingTopKSoftmaxV2的计算结果做routing处理，支持不量化、静态量化和动态量化模式。

### 计算公式

1. 对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

   $$
sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
   $$

2. 以sortedRowIdx做位置映射得出expandedRowIdxOut：
   - rowIdxType等于1时, 输出scatter索引

     $$
expandedRowIdxOut[i]=sortedRowIdx[i]
     $$

   - rowIdxType等于0时, 输出gather索引

     $$
expandedRowIdxOut[sortedRowIdx[i]]=i
     $$
     
3. 对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsumOutOptional：

   $$
expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
   $$

4. 如果quantMode不等于-1, 计算quant结果：
   - 静态quant

   $$
quantResult=round((x∗scaleOptional)+offsetOptional)
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
quantResult = round(x / dynamicQuantScaleOutOptional)
          $$

5. 若活跃的expert范围为全专家范围时，按照Scatter索引搬运token；反之按照Gather索引搬运token。在dropPadMode为1时将每个专家需要处理的Token个数对齐为expertCapacity个，超过expertCapacity个的Token会被Drop，不足的会用0填充。得出expandedXOut：
   - 非量化场景
     - 按照Scatter索引搬运

     $$
expandedXOut[i]=x[scatterRowIdx[i] // K]
     $$

     - 按照Gather索引搬运

     $$
expandedXOut[gatherRowIdx[i]]=x[i // K]
     $$

   - 量化场景
     - 按照Scatter索引搬运

     $$
expandedXOut[i]=quantResult[scatterRowIdx[i] // K]
     $$

     - 按照Gather索引搬运

     $$
expandedXOut[gatherRowIdx[i]]=quantResult[i // K]
     $$

6. expandedRowIdxOut的有效元素数量availableIdxNum，计算方式为expertIdx中activeExpertRangeOptional范围内的元素的个数

   $$
availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
   $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.moe_init_routing_v3()` 函数形式提供：

```python
def moe_init_routing_v3(x: Tensor, 
                       expert_idx: Tensor, 
                       scale_optional: Optional[Tensor] = None,
                       offset_optional: Optional[Tensor] = None,
                       active_num: int = 0,
                       expert_capacity: int = 0,
                       expert_num: int = 0,
                       drop_pad_mode: int = 0,
                       expert_tokens_num_type: int = 0,
                       expert_tokens_num_flag: bool = False,
                       quant_mode: int = -1,
                       active_expert_range_optional: Optional[List[int]] = None,
                       row_idx_type: int = 0):
    """
    实现 MoE 初始化路由操作，根据expertIdx对token进行重排和处理。

    参数:
        x (Tensor): MOE的输入，即token特征输入，形状为(NUM_ROWS, H)
        expert_idx (Tensor): 每一行特征对应的K个处理专家，形状为(NUM_ROWS, K)
        scale_optional (Optional[Tensor]): 用于计算量化结果的参数
        offset_optional (Optional[Tensor]): 用于计算quant结果的偏移值
        active_num (int): 总的最大处理row数
        expert_capacity (int): 每个专家能够处理的tokens数
        expert_num (int): 专家数
        drop_pad_mode (int): 是否为DropPad场景，0表示Dropless，1表示DropPad
        expert_tokens_num_type (int): 直方图的不同模式，0表示comsum，1表示count，2表示key_value
        expert_tokens_num_flag (bool): 是否输出expertTokensCountOrCumsumOut
        quant_mode (int): 量化模式，-1表示不量化，0表示静态量化，1表示动态量化
        active_expert_range_optional (Optional[List[int]]): 活跃的expert范围，格式为[expertStart, expertEnd]
        row_idx_type (int): expandedRowIdxOut使用的索引类型，0表示gather，1表示scatter

    返回:
        Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]: 包含四个张量的元组
            - expanded_x_out: 根据expertIdx进行扩展过的特征
            - expanded_row_idx_out: expandedXOut和x的索引映射关系
            - expert_tokens_count_or_cumsum_out: 每个专家处理的token数量的统计结果或累加值
            - expanded_scale_out: 不同量化过程中scaleOptional的中间值
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 场景1：基本使用
# 创建输入张量
tokens_num = 32
hidden_size = 512
topk_num = 4

x = torch.randn(tokens_num, hidden_size, dtype=torch.float32)
expert_idx = torch.randint(0, 16, (tokens_num, topk_num), dtype=torch.int32)

# 执行路由计算
expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = kernel_gen_ops.moe_init_routing_v3(
    x, expert_idx, expert_num=16
)

print(f"expanded_x shape: {expanded_x.shape}")
print(f"expanded_row_idx shape: {expanded_row_idx.shape}")
print(f"expert_tokens_count: {expert_tokens_count}")
print(f"expanded_scale: {expanded_scale}")

# 场景2：带量化的使用
x = torch.randn(tokens_num, hidden_size, dtype=torch.float32)
expert_idx = torch.randint(0, 16, (tokens_num, topk_num), dtype=torch.int32)

# 执行带动态量化的路由计算
expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale = kernel_gen_ops.moe_init_routing_v3(
    x, expert_idx, expert_num=16, quant_mode=1, expert_tokens_num_flag=True, expert_tokens_num_type=1
)

print(f"expanded_x shape: {expanded_x.shape}")
print(f"expanded_x dtype: {expanded_x.dtype}")  # 应为INT8
print(f"expert_tokens_count shape: {expert_tokens_count.shape}")
print(f"expanded_scale shape: {expanded_scale.shape}")
```

## 约束与限制

- 输入张量数据类型支持：
  - x: FLOAT16、BFLOAT16、FLOAT32、INT8、HIFLOAT8
  - expert_idx: INT32
  - scale_optional: FLOAT32
  - offset_optional: FLOAT32

- 输出张量数据类型：
  - expanded_x_out: 非量化场景下同x，量化场景下为INT8、FLOAT8_E5M2、FLOAT8_E4M3FN或HIFLOAT8
  - expanded_row_idx_out: INT32
  - expert_tokens_count_or_cumsum_out: INT64
  - expanded_scale_out: FLOAT32或FLOAT8_E8M0

- 专家数限制：
  - expert_tokens_num_type为key_value模式时，expert_num取值范围为[0, 5120]
  - 其它模式时，expert_num取值范围为[0, 10240]

- 量化模式支持：
  - Atlas A2/A3系列产品：支持-1（不量化）、0（静态量化）、1（动态量化）
  - Ascend 950PR/Ascend 950DT：支持-1、1、2（MXFP8_E5M2）、3（MXFP8_E4M3FN）、6（HIF8直转）、7（HIF8 PERTENSOR）、8（HIF8 PERTOKEN）

- 特殊参数限制：
  - Ascend 950PR/Ascend 950DT仅支持：
    - activeNum等于NUM_ROWS*K
    - dropPadMode取值为0
    - expertTokensNumType取值为1或2
    - expertTokensNumFlag取值为true

- 性能模板准入条件：
  - 低时延性能模板：x shape=(1, 7168)、expertIdx shape=(1, 8)、scaleOptional shape=(256, 7168)，x dtype=BFLOAT16，activeExpertRangeOptional=[0, 256]，quantMode=1，expertTokensNumType=2，expertNum=256
  - 大batch性能模板：NUM_ROWS范围[384, 8192]，K=8，expertNum=256，expertEnd-expertStart<=32，quantMode=-1，rowIdxType=1，expertTokensNumType=1
  - 全载性能模板：dropPadMode=0

- 确定性计算：aclnnMoeInitRoutingV3默认确定性实现