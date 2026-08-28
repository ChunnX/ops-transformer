# all_gather_quant_matmul

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

- **接口功能**：

  `cann_ops_transformer.all_gather_quant_matmul`完成AllGather通信与MX量化矩阵乘法的融合计算。在多卡场景下，各卡先通过URMA all-gather收集所有卡的左矩阵及其MX缩放因子，再与本地的右矩阵做MX FP8/FP4量化矩阵乘法，输出BF16或FP16结果。底层调用`aclnnAllGatherQuantMatmulV3`接口完成计算。
- **计算公式**：

  本算子仅支持MX量化场景：`x1`为`(M, K)`、`x2`为`(N, K)`，`x1_scale`为`(M, ceilDiv(K, 64), 2)`、`x2_scale`为`(N, ceilDiv(K, 64), 2)`。入参`x1`和`x1_scale`进行AllGather后，对`x1`、`x2`进行MatMul计算，然后按K轴分组（group size = 32）进行dequant操作：

  $$
  output=\sum_{0}^{\left \lfloor \frac{K}{blockSize=32} \right \rfloor}(AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1\_scale)_{pr}*x2\_scale_{rq})) + bias
  $$

  其中 rankSize 为 NPU 卡数，即 world_size；`bias`为可选累加项，输出形状为`(M * rankSize, N)`。

## 函数原型

```python
cann_ops_transformer.all_gather_quant_matmul(
    x1,
    x2,
    group,
    *,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    group_sizes=None,
    x1_dtype=None,
    x2_dtype=None,
    x1_scale_dtype=None,
    x2_scale_dtype=None,
    y_dtype=None,
    comm_mode="ai_cpu",
) -> (Tensor, Tensor, Tensor)
```

## 参数说明

| 参数名             | 参数类型     | 可选/必选 | 描述                                                                                                                                                                                                                                                                | 数据类型                                                                                      | 维度(shape)              |
| ------------------ | ------------ | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------ |
| `x1`             | Tensor       | 必选      | 左矩阵，对应各卡本地的`x1[M, K]`。MXFP4场景以uint8打包存储（2个fp4占1字节），实际传入的K轴大小为`K/2`，需配合`x1_dtype=296`。                                                                                                                                 | `torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch.float4_e2m1fn_x2`（或uint8打包存储） | `(M, K)`               |
| `x2`             | Tensor       | 必选      | 右矩阵，对应公式中X2，须传入`[N, K]`存储的`.t()`转置视图。MXFP4场景以uint8打包存储，需配合`x2_dtype=296`，且`x1`/`x2`必须同为FP4。                                                                                                                        | `torch.float8_e4m3fn`、`torch.float8_e5m2`、`torch.float4_e2m1fn_x2`（或uint8打包存储） | `(N, K)`               |
| `group`          | ProcessGroup | 必选      | torch.distributed通信域，用于AllGather通信。                                                                                                                                                                                                                        | -                                                                                             | -                        |
| `bias`           | Tensor       | 可选      | 偏置项，矩阵乘后累加；传入`None`时不叠加偏置。                                                                                                                                                                                                                    | `torch.float32`                                                                             | `(N,)`                 |
| `x1_scale`       | Tensor       | 可选      | `x1`的MX量化scale，MX量化场景必传。                                                                                                                                                                                                                               | `torch.float8_e8m0fnu`                                                                      | `(M, ceil(K / 64), 2)` |
| `x2_scale`       | Tensor       | 可选      | `x2`的MX量化scale，MX量化场景必传。                                                                                                                                                                                                                               | `torch.float8_e8m0fnu`                                                                      | `(N, ceil(K / 64), 2)` |
| `group_sizes`    | List[int]    | 可选      | MX量化group大小，MX场景仅支持`[1, 1, 32]`；传入`None`时按公式推导：groupSizeM = M / scaleM（需保证`x1`的M能被`x1_scale`的M维整除），groupSizeN = N / scaleN（需保证`x2`的N能被`x2_scale`的N维整除），groupSizeK = K / scaleK（需保证K能被scaleK整除）。 | int                                                                                           | -                        |
| `x1_dtype`       | int          | 可选      | `x1`的dtype wrapper覆盖值，传入`None`时使用输入`x1`的dtype。FP8场景传291（fp8_e5m2）或292（fp8_e4m3fn），MXFP4场景传296（fp4_e2m1）。                                                                                                                         | int                                                                                           | -                        |
| `x2_dtype`       | int          | 可选      | `x2`的dtype wrapper覆盖值，传入`None`时使用输入`x2`的dtype。FP8场景传291（fp8_e5m2）或292（fp8_e4m3fn），MXFP4场景传296（fp4_e2m1）。                                                                                                                         | int                                                                                           | -                        |
| `x1_scale_dtype` | int          | 可选      | `x1_scale`的dtype wrapper覆盖值，传入`None`时使用输入`x1_scale`的dtype。uint8存储时传293（fp8_e8m0）。                                                                                                                                                        | int                                                                                           | -                        |
| `x2_scale_dtype` | int          | 可选      | `x2_scale`的dtype wrapper覆盖值，传入`None`时使用输入`x2_scale`的dtype。uint8存储时传293（fp8_e8m0）。                                                                                                                                                        | int                                                                                           | -                        |
| `y_dtype`        | int          | 可选      | 输出`y`的数据类型，仅支持传15（BF16）或5（FP16）；传入`None`时默认为BF16。                                                                                                                                                                                      | int                                                                                           | -                        |
| `comm_mode`      | str          | 可选      | 通信模式，默认值为`"ai_cpu"`；当前仅支持`"urma"`，需显式传入，缺省时默认`"ai_cpu"`会被接口拦截报错。                                                                                                                                                          | string                                                                                        | -                        |

## 返回值说明

| 参数名         | 参数类型 | 可选/必选 | 描述                                                                                   | 数据类型                        | 维度(shape)           |
| -------------- | -------- | --------- | -------------------------------------------------------------------------------------- | ------------------------------- | --------------------- |
| `y`          | Tensor   | 必选      | AllGather与矩阵乘融合计算输出。                                                        | 由`y_dtype`指定（BF16或FP16） | `(M * rankSize, N)` |
| `gather_out` | Tensor   | 可选      | AllGather通信中间输出，预留参数，当前版本未实现（返回shape为`(0,)`的空占位tensor）。 | 同`x1`                        | `(0,)`              |
| `amax_out`   | Tensor   | 可选      | amax输出，预留参数，当前版本未实现（返回shape为`(0,)`的空占位tensor）。              | `torch.float32`               | `(0,)`              |

## 约束说明

- 适用场景：该接口支持训练、推理场景下使用。
- 调用方式：该接口支持单算子模式调用，需要在多卡通信域下使用。
- 仅支持Ascend 950系列产品。
- 输入输出Tensor的数据格式仅支持ND。
- 输入`x1`为2维，shape为`(M, K)`；输入`x2`为2维，须传入`[N, K]`存储的`.t()`转置视图（is_trans_b必须为true）。`x1`与`x2`的K轴相等，`K`取值范围为[256, 65535)；FP4场景`K`须为偶数。
- `M`（单卡M轴长度）、`N`取值范围为[1, 2147483647]（INT32_MAX）。
- `bias`为1维，shape为`(N,)`，dtype为`torch.float32`。
- `x1_scale`、`x2_scale`为3维，shape分别为`(M, ceil(K/64), 2)`、`(N, ceil(K/64), 2)`，dtype为`torch.float8_e8m0fnu`（或uint8存储搭配scale_dtype=293）。
- 输出`y`为2维，shape为`(M * rank_size, N)`，dtype仅支持BF16/FP16。
- `x1`与`x2`需同为FP8（float8_e4m3fn与float8_e5m2可混用）或同为FP4。
- `rank_size`需为2、4、8、16之一，且不超过设备实际可用计算核数。
- `comm_mode`当前仅支持`"urma"`。
- `group_sizes`在MX场景仅支持`[1, 1, 32]`（显式传入或自动推导结果均须满足）。
- 通信域`group`名称长度需在[1, 128)范围内；`hccl_buffer_size`需为正整数。
- 通信buffer大小约束：本算子使用HCCL内置通信buffer完成AllGather通信，该buffer大小由环境变量`HCCL_BUFFSIZE`（单位MB）控制。要求`通信数据总量 + 2MB <= hccl_buffer_size`，其中通信数据总量为`rank_size * M * (x1数据 + x1_scale)`；该约束在torch接口层与tiling层双重拦截，不满足时tiling阶段报错并提示建议的`HCCL_BUFFSIZE`取值，可通过`export HCCL_BUFFSIZE=<取值>`调大buffer后重试。

## 调用说明

- 单算子模式调用（MX FP8 场景）：

  ```python
  import math
  import torch
  import torch_npu
  import cann_ops_transformer
  import torch.distributed as dist

  # 初始化多卡通信域
  dist.init_process_group(backend="hccl")
  group = dist.group.WORLD
  world_size = dist.get_world_size()

  M = 2048
  K = 4096
  N = 2560

  x1 = torch.randn(M, K).to(torch.float8_e4m3fn).npu()
  x2 = torch.randn(N, K).to(torch.float8_e4m3fn).npu()
  x1_scale = torch.randint(0, 256, (M, math.ceil(K / 64), 2), dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)
  x2_scale = torch.randint(0, 256, (N, math.ceil(K / 64), 2), dtype=torch.uint8).npu().view(torch.float8_e8m0fnu)

  y, _, _ = cann_ops_transformer.all_gather_quant_matmul(
      x1,
      x2.t(),                # 传入 [N, K] 存储的 .t() 转置视图
      group,
      x1_scale=x1_scale,
      x2_scale=x2_scale,
      group_sizes=[1, 1, 32],
      y_dtype=15,            # BF16
      comm_mode="urma",
  )
  ```
