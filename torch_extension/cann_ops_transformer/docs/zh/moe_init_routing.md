# moe_init_routing

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- API功能：MoE的routing计算，根据moe_gating_top_k_softmax的计算结果做routing处理，支持非量化、静态量化和动态量化模式。支持输入`topk_weight`和输出`expanded_topk_weight`，支持对topkWeight按排序后的索引进行重排，使得`expanded_topk_weight`与`expanded_x`一一对应。`topk_weight`与`expanded_topk_weight`必须同时传入或同时不传入。

- 计算公式：

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：
    - rowIdxType等于1时, 输出scatter索引

      $$
      expandedRowIdxOut[i]=sortedRowIdx[i]
      $$

    - rowIdxType等于0时, 输出gather索引

      $$
      expandedRowIdxOut[sortedRowIdx[i]]=i
      $$

  3.对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.如果quantMode不等于-1, 计算quant结果：
     - 静态quant

     $$
     quantResult=round((x∗scaleOptional)+offsetOptional)
     $$

    - 动态quant：
        - 若不输入scale：

            $$
            dynamicQuantScaleOutOptional = row_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - 若输入scale:

            $$
            dynamicQuantScaleOutOptional = row_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$

        - 当quantMode为13时，动态量化使用对称量化范围[-8, 7]，scale计算中的分母为7，量化结果沿H维每两个INT4值打包为1个字节。

  5.若活跃的expert范围为全专家范围时，按照Scatter索引搬运token；反之按照Gather索引搬运token。在dropPadMode为1时将每个专家需要处理的Token个数对齐为expertCapacity个，超过expertCapacity个的Token会被Drop，不足的会用0填充。得出expandedXOut：
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

  6.若输入topkWeight，按照排序后的索引对topkWeight进行重排，使得expandedTopkWeightOut与expandedXOut一一对应：
    - Dropless场景（dropPadMode=0）

      $$
      expandedTopkWeightOut[i] = topkWeight[sortedRowIdx[i]], \quad 0 \le i < effectiveNum
      $$

      其中effectiveNum为：

      $$
      effectiveNum = \min(activeNum, availableIdxNum)
      $$

    - DropPad场景（dropPadMode=1）

      $$
      expandedTopkWeightOut[dstIdx] = topkWeight[globalSortIdx], \quad dstIdx \ne -1
      $$

      其中globalSortIdx为expertIdx在flatten布局中的位置（即globalSortIdx = rowIdx \* K + colIdx，对应topkWeight[rowIdx, colIdx]），dstIdx为expandedRowIdxOut中对应位置的目标索引，未被分配的专家容量位置expandedTopkWeightOut填充为0。

  7.expandedRowIdxOut的有效元素数量availableIdxNum，计算方式为expertIdx中activeExpertRangeOptional范围内的元素的个数

    $$
    availableIdxNum = |\{x\in expertIdx| expert_start \le x<expert_end \ \}|
    $$

## 函数原型

```python
def moe_init_routing(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    *,
    scale: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    topk_weight: Optional[torch.Tensor] = None,
    active_num: Union[int, torch.SymInt] = -1,
    expert_capacity: int = -1,
    expert_num: int = -1,
    drop_pad_mode: int = 0,
    expert_tokens_num_type: int = 0,
    expert_tokens_num_flag: bool = False,
    quant_mode: int = -1,
    active_expert_range: Optional[List[int]] = None,
    row_idx_type: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

## 参数说明

- **x** (`Tensor`)：必选参数，表示MoE的输入即token特征输入，要求为2维张量，shape为(NUM_ROWS, H)。数据类型支持`float16`、`bfloat16`、`float32`、`int8`，数据格式要求为$ND$。
- **expert_idx** (`Tensor`)：必选参数，表示moe_gating_top_k_softmax输出每一行特征对应的K个处理专家，要求是2维张量，shape为(NUM_ROWS, K)，且专家id不能超过专家数。数据类型支持`int32`，数据格式要求为$ND$。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **scale** (`Tensor`)：可选参数，默认为None，用于计算量化结果的参数。数据类型支持`float32`，数据格式要求为$ND$。如果不输入表示计算时不使用`scale`，且输出`expanded_scale`中的值无意义。
    - 非量化场景下，如果输入则要求为1维张量，shape为(NUM_ROWS,)。
    - 静态量化场景必须输入，输入要求为1D的Tensor，shape为(1,)
    - 动态量化场景下，如果输入则要求为2维张量，shape为(expert_end-expert_start, H)或(1, H)。
    - quantMode为1的INT8动态量化场景下为可选输入，如果输入则要求为2D的Tensor，shape为(expert_end-expert_start, H)；quantMode为13的INT4动态量化场景下为可选输入，如果输入则要求shape为(1, H)，表示按H维广播的smooth scale。
    - MXFP8量化场景下（quantMode为2、3）不输入。

- **offset** (`Tensor`)：可选参数，默认为None，用于计算量化结果的偏移值。数据类型支持`float32`，数据格式要求为$ND$。
    - 在非量化场景下不输入。
    - 静态量化场景必须输入，输入要求为1维张量，shape为(1,)
    - 动态量化、MXFP8量化场景下不输入。

- **topk_weight** (`Tensor`)：可选参数，默认为None，表示topk专家的路由权重，用于按排序索引重排以与`expanded_x`一一对应。
    - 可选输入，不输入表示不输出`expanded_topk_weight`。
    - 输入shape为(NUM_ROWS, K)，数据类型仅支持`float32`，数据格式要求为$ND$。
    - 与`expanded_topk_weight`联动：必须同时传入或同时不传入。

- **active_num** (`Union[int, torch.SymInt]`)：可选参数，默认值为-1，表示总的最大处理row数，输出`expanded_x`只有这么多行是有效的，入参校验需大于等于0，0表示Dropless场景，大于0时表示Active场景，约束所有专家共同处理tokens总量。支持`int`或`torch.SymInt`类型，图模式下传入`torch.SymInt`（如`x.size(0) * k`）可避免被固化为常量，实现动态shape编译缓存复用。eager模式下传入`int`即可，上层调用脚本无需感知类型差异，同一份代码在两种模式下均可正常运行。
- **expert_capacity** (`int`)：可选参数，默认值为-1，表示每个专家能够处理的tokens数，入参校验大于0小于NUM_ROWS。
- **expert_num** (`int`)：可选参数，默认值为-1，表示专家数。`expert_tokens_num_type`为key_value模式时，取值范围为[0, 5120]；其他模式取值范围为[0, 10240]。
- **drop_pad_mode** (`int`)：可选参数，默认值为0，表示是否为drop_pad场景。0表示Dropless场景，该场景下不校验`expert_capacity`。1表示Drop_Pad场景。
- **expert_tokens_num_type** (`int`)：可选参数，默认值为0，表示直方图的不同模式。取值为0、1和2。0表示cumsum模式；1表示count模式；2表示key_value模式。
- **expert_tokens_num_flag** (`bool`)：可选参数，默认值为False，取值为False和True，表示是否输出`expert_token_cumsum_or_count`。
- **quant_mode** (`int`)：可选参数，默认值为-1，表示量化模式，支持取值为-1、0、1、2、3、4、5、6、7、8、9、11、12、13、14、15、16、17。
    - -1：表示非量化场景；
    - 0：表示静态量化场景；
    - 1：表示动态量化场景，`expanded_x`量化到INT8；
    - 2：表示MXFP8量化场景，`expanded_x`量化到float8_e5m2；
    - 3：表示MXFP8量化场景，`expanded_x`量化到float8_e4m3fn；
    - 4：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale），`expanded_x`量化到float8_e5m2；
    - 5：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale），`expanded_x`量化到float8_e4m3fn；
    - 6：表示HIF8直转量化场景，`expanded_x`量化到hifloat8；
    - 7：表示HIF8 PERTENSOR量化场景；
    - 8：表示HIF8 PERTOKEN量化场景；
    - 9：表示MXFP4量化场景，`expanded_x`量化到float4_e2m1；
    - 11：表示FP8 PerBlock量化场景（BlockSize=128），`expanded_x`量化到float8_e5m2；
    - 12：表示FP8 PerBlock量化场景（BlockSize=128），`expanded_x`量化到float8_e4m3fn；
    - 13：表示INT4动态量化场景，`expanded_x`量化到INT4；
    - 14：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale+Amax），`expanded_x`量化到float8_e5m2；
    - 15：表示FP8 PerGroup量化场景（GroupSize=128，RoundScale+Amax），`expanded_x`量化到float8_e4m3fn；
    - 16：表示MXFP8 RoundScale+Amax量化场景，`expanded_x`量化到float8_e5m2；
    - 17：表示MXFP8 RoundScale+Amax量化场景，`expanded_x`量化到float8_e4m3fn。
- **active_expert_range** (`List[int]`)：可选参数，默认为None，表示活跃expert的范围。长度为2，数组内的值为[expert_start, expert_end]，左闭右开，要求值大于等于0，并且expert_end不大于`expert_num`。Drop/Pad场景下，expert_start等于0, expert_end等于`expert_num`。传入None时，视为活跃的expert范围在0到`expert_num`之间。
- **row_idx_type** (`int`)：可选参数，默认为0，表示输出`expanded_row_idx`使用的索引类型，支持取值0和1。0表示gather类型的索引；1表示scatter类型的索引。

## 返回值说明

- **expanded_x** (`Tensor`)：根据`expert_idx`进行扩展过的特征。Dropless场景shape为[NUM_ROWS \* K, H]。Active场景shape为[min(activeNum, NUM_ROWS \* K), H]。Drop/Pad场景下shape为[expertNum \* expertCapacity, H]。非量化场景下数据类型同`x`；量化场景下数据类型根据quantMode确定（详见参数说明中quant_mode）。数据格式要求为$ND$。量化场景下，当`x`的数据类型为`int8`时，输出值无意义。
- **expanded_row_idx** (`Tensor`)：`expanded_x`和`x`的映射关系，shape与`expanded_x`第一维一致，数据类型支持`int32`，数据格式要求为$ND$。前availableIdxNum个元素为有效数据，其余无效数据由`row_idx_type`决定：当`row_idx_type`为0时，无效数据由-1填充；当`row_idx_type`为1时，无效数据未初始化。
- **expert_token_cumsum_or_count** (`Tensor`)：表示输出每个专家处理的token数量的统计结果或累加值。数据类型支持`int64`，数据格式要求为$ND$。
    - 在`expert_tokens_num_type`为0时，表示`active_expert_range`范围内expert在排序后处理token总数的前缀和。
    - 在`expert_tokens_num_type`为1的场景下，shape为[expert_end-expert_start]。
    - 在`expert_tokens_num_type`为2的场景下，shape为[expert_num, 2]，表示token总数为非0的expert及其处理token的总数。

- **expanded_scale** (`Tensor`)：输出不同量化过程中scale的中间值。数据类型支持`float32`或`float8_e8m0`，数据格式要求为$ND$。
    - 非量化场景下，当`scale`输入时，shape为[NUM_ROWS\*K, 1]，前availableIdxNum个元素为有效数据，输出`float32`类型。当输入x数据类型为`float8_e5m2`、`float8_e4m3fn`或`float4_e2m1`时，如果`scale`输入，则shape为[NUM_ROWS\*K, CeilDiv(H, 64), 2]，输出`float8_e8m0`类型。
    - 动态量化场景下，当`scale`输入时，前availableIdxNum个元素为有效数据。
    - 静态量化场景下、HIF8直转量化场景下、HIF8 PERTENSOR量化场景下，输出为空tensor。
    - HIF8 PERTOKEN量化场景下，shape为[NUM_ROWS\*K, 1]，输出`float32`类型。
    - MXFP8量化场景下（quantMode为2、3、16、17），输出`float8_e8m0`类型，Shape为[NUM_ROWS\*K, M]，其中M=CeilAlign(CeilDiv(H, 32), 2)，前availableIdxNum行为有效数据。
    - MXFP4量化场景下，输出`float8_e8m0`类型，Shape为[NUM_ROWS\*K, M, 2]，其中M=CeilDiv(H, 64)，前availableIdxNum行为有效数据。
    - FP8 PerGroup量化场景下（quantMode为4、5、14、15），输出`float32`类型，Shape为[NUM_ROWS\*K, CeilDiv(H, 128)]，前availableIdxNum行为有效数据。
    - FP8 PerBlock量化场景下（quantMode为11、12），输出`float32`类型，Shape为[NUM_ROWS\*K, CeilDiv(H, 256), 2]，前availableIdxNum行为有效数据。

- **expanded_topk_weight** (`Tensor`)：按排序索引重排后的路由权重，与`expanded_x`一一对应。数据类型仅支持`float32`，数据格式要求为$ND$。
    - 可选输出，`topk_weight`输入时必须同时输出；`topk_weight`未输入时输出为空tensor（shape为(0,)）。
    - Dropless场景下shape为[min(activeNum, NUM_ROWS\*K), 1]，前effectiveNum个元素为有效数据，其余未初始化，effectiveNum=min(activeNum, availableIdxNum)。
    - DropPad场景下shape为[expertNum\*expertCapacity, 1]，已分配容量的位置为有效数据，未分配容量的位置填充为0。

## 约束说明

1. 该接口支持推理场景下使用。
2. 该接口支持单算子模式和图模式调用。
3. `topk_weight`不受`quant_mode`影响，`expanded_topk_weight`数据类型始终为`float32`。
4. `expert_num`必须大于0。
5. `drop_pad_mode=1`时，`row_idx_type`仅支持取值为0（gather索引），`quant_mode`仅支持-1（非量化）。
6. `expert_tokens_num_flag`仅支持取值为true。
7. `active_num`仅支持值等于NUM_ROWS*K。
8. quantMode为13的INT4动态量化场景，需同时满足：`x`数据类型为`float32`或`bfloat16`；H为偶数。
9. 空tensor处理：当输入的x首个维度的值为0时，DropPadMode必须为0，expanded_x、expanded_row_idx和expanded_scale为空tensor，expert_token_cumsum_or_count返回全0的tensor。

## 调用示例

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import moe_init_routing

    n = 3
    h = 2
    k = 4
    expert_num = 256
    expert_capacity = -1
    drop_pad_mode = 0
    expert_tokens_num_type = 1
    expert_tokens_num_flag = True
    quant_mode = -1
    active_expert_range = [0, 4]
    row_idx_type = 1

    x = torch.randn((n, h), dtype=torch.float32).npu()
    expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32).npu()
    scale = torch.randn((n,), dtype=torch.float32).npu()

    # 不传入topk_weight，expanded_topk_weight为空tensor
    expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
        moe_init_routing(x, expert_idx, scale=scale,
                         active_num=-1, expert_capacity=expert_capacity,
                         expert_num=expert_num, drop_pad_mode=drop_pad_mode,
                         expert_tokens_num_type=expert_tokens_num_type,
                         expert_tokens_num_flag=expert_tokens_num_flag,
                         quant_mode=quant_mode, active_expert_range=active_expert_range,
                         row_idx_type=row_idx_type)

    # 传入topk_weight，expanded_topk_weight为有效数据
    topk_weight = torch.randn((n, k), dtype=torch.float32).npu()
    expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
        moe_init_routing(x, expert_idx, topk_weight=topk_weight, scale=scale,
                         active_num=-1, expert_capacity=expert_capacity,
                         expert_num=expert_num, drop_pad_mode=drop_pad_mode,
                         expert_tokens_num_type=expert_tokens_num_type,
                         expert_tokens_num_flag=expert_tokens_num_flag,
                         quant_mode=quant_mode, active_expert_range=active_expert_range,
                         row_idx_type=row_idx_type)
    ```

- 图模式调用（静态量化场景，quant_mode=0）

    ```python
    import torch
    import torch.nn as nn
    import torch_npu
    import torchair as tng
    from torchair.configs.compiler_config import CompilerConfig
    from cann_ops_transformer.ops import moe_init_routing

    config = CompilerConfig()
    npu_backend = tng.get_npu_backend(compiler_config=config)

    class MoeInitRoutingModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, expert_idx, *, topk_weight=None, scale=None, offset=None,
                    active_num=-1, expert_capacity=-1, expert_num=-1, drop_pad_mode=0,
                    expert_tokens_num_type=0, expert_tokens_num_flag=False, quant_mode=-1,
                    active_expert_range=None, row_idx_type=0):
            return moe_init_routing(x, expert_idx, topk_weight=topk_weight,
                                    scale=scale, offset=offset, active_num=active_num,
                                    expert_capacity=expert_capacity, expert_num=expert_num,
                                    drop_pad_mode=drop_pad_mode,
                                    expert_tokens_num_type=expert_tokens_num_type,
                                    expert_tokens_num_flag=expert_tokens_num_flag,
                                    quant_mode=quant_mode,
                                    active_expert_range=active_expert_range,
                                    row_idx_type=row_idx_type)

    def main():
        n = 3
        h = 2
        k = 4
        expert_num = 256
        active_expert_range = [0, 4]

        x = torch.randn((n, h), dtype=torch.float32).npu()
        expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32).npu()
        topk_weight = torch.randn((n, k), dtype=torch.float32).npu()
        scale = torch.randn((1,), dtype=torch.float32).npu()
        offset = torch.randn((1,), dtype=torch.float32).npu()

        model = MoeInitRoutingModel().npu()
        model = torch.compile(model, backend=npu_backend, dynamic=False)
        expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
            model(x, expert_idx, topk_weight=topk_weight, scale=scale, offset=offset,
                  active_num=-1, expert_capacity=-1, expert_num=expert_num,
                  drop_pad_mode=0, expert_tokens_num_type=1,
                  expert_tokens_num_flag=True, quant_mode=0,
                  active_expert_range=active_expert_range, row_idx_type=1)

    if __name__ == '__main__':
        main()
    ```
