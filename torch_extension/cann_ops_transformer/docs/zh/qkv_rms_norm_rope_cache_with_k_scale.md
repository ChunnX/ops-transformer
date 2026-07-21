# qkv_rms_norm_rope_cache_with_k_scale

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：本文档包含`qkv_rms_norm_rope_cache_with_k_scale_`和`qkv_rms_norm_rope_cache_with_k_scale`两个torch_extension接口，均封装`aclnnQkvRmsNormRopeCacheWithKScale`，用于大语言模型推理场景下的Q/K/V预处理与PagedAttention KV Cache更新。接口支持以下两种完整场景：
  - legacy场景：使用`query_start_loc`和`seq_lens`生成普通RoPE位置，对Q/K执行RMSNorm、RoPE和共享`rotation`矩阵乘，并将Q/K动态量化为FP8 E4M3FN；返回FP8 `q_out`和FP32 `q_scale`，K写入FP8 Cache。
  - M-RoPE场景：使用`mrope_position`读取A/T/W三路位置，并由`mrope_section`分别交织cos和sin。Q不量化，返回BF16 `q_out`，固定的`q_scale`返回位置为`None`；K量化为INT8并写入INT8 Cache。
- `qkv_rms_norm_rope_cache_with_k_scale_`原地更新调用方传入的`k_cache`、`v_cache`和`k_scale_cache`，返回`q_out`和可选的`q_scale`。
- `qkv_rms_norm_rope_cache_with_k_scale`内部先拷贝三个cache，再对副本执行更新，返回`q_out`、可选的`q_scale`和更新后的三个cache；调用方传入的cache保持不变。

- 基础计算公式：

  按`head_nums=[Nq,Nk,Nv]`从`qkv`拆分Q、K、V：

  $$
  q, k, v = split(qkv, [Nq, Nk, Nv])
  $$

  Q/K分支分别使用`q_gamma`和`k_gamma`做RMSNorm：

  $$
  y = \frac{x}{\sqrt{mean(x^2) + epsilon}} * gamma
  $$

  Q/K分支执行RoPE，V分支不执行RoPE：

  $$
  y_{rope} = concat(y_{low} * cos - y_{high} * sin,\ y_{high} * cos + y_{low} * sin)
  $$

  Q/K共享`rotation`矩阵：

  $$
  q_{rot} = q_{rope} @ rotation,\quad k_{rot} = k_{rope} @ rotation
  $$

- legacy场景的RoPE与量化：

  第`b`个batch中第`i`个token的RoPE位置由`query_start_loc`和`seq_lens`确定：

  $$
  position = seq\_lens[b] - (query\_start\_loc[b + 1] - query\_start\_loc[b]) + i
  $$

  `cos_sin[..., :D/2]`为cos，`cos_sin[..., D/2:]`为sin。Q/K按每个token和head做动态量化，FP8 E4M3FN最大有限值为448：

  $$
  scale = max(abs(x)) / 448,\quad x_{fp8} = cast(x / scale)
  $$

- M-RoPE场景的cos/sin交织与量化：

  令`H=D/2`。对token `t`，按照`mrope_position`完整读取A/T/W三行：

  ```text
  A[t,d] = cos_sin[mrope_position[0,t], d]
  Taxis[t,d] = cos_sin[mrope_position[1,t], d]
  W[t,d] = cos_sin[mrope_position[2,t], d]
  d in [0,D)
  ```

  `mrope_section=[m,n,k]`中第0项`m`不参与计算。对cos和sin各自长度为`H`的half独立应用相同的交织规则：

  ```text
  for h in [0,H):
      triplet = h // 3
      lane = h % 3
      if lane == 1 and triplet < n:
          source_axis(h) = Taxis
      elif lane == 2 and triplet < k:
          source_axis(h) = W
      else:
          source_axis(h) = A

  cos_eff[t,h] = source_axis(h)[t,h]
  sin_eff[t,h] = source_axis(h)[t,H+h]
  ```

  A/T/W三行始终完整采集；超出T/W覆盖范围的lane回退到A的同一列。输出维度仍为`D`，不会扩展为`3D`。Q分支不量化，直接输出BF16且不创建`q_scale`；K分支使用INT8最大值127做动态量化：

  $$
  k\_scale = max(abs(k)) / 127,\quad k_{int8} = cast(k / k\_scale)
  $$

  V分支在两种场景中均按`v_scale`缩放并量化为FP8 E4M3FN：

  $$
  v_{fp8} = cast(v * v\_scale)
  $$

  Cache写回位置由`slot_mapping`决定：

  $$
  blockId = slot\_mapping[t] / BlockSize,\quad blockOffset = slot\_mapping[t]\ \%\ BlockSize
  $$

## 函数原型

`qkv_rms_norm_rope_cache_with_k_scale_`为原地cache更新接口；`qkv_rms_norm_rope_cache_with_k_scale`为函数式变体。两个接口输入参数一致。

```python
cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc,
    seq_lens,
    head_nums,
    *,
    rotation=None,
    v_scale=None,
    layout_qkv="TND",
    layout_q_out="NTD",
    epsilon=0.000001,
    mrope_position=None,
    mrope_section=None,
    q_quant_mode="PerTokenPerHead",
    q_out_dtype=36,
) -> (Tensor, Optional[Tensor])
```

```python
cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    query_start_loc,
    seq_lens,
    head_nums,
    *,
    rotation=None,
    v_scale=None,
    layout_qkv="TND",
    layout_q_out="NTD",
    epsilon=0.000001,
    mrope_position=None,
    mrope_section=None,
    q_quant_mode="PerTokenPerHead",
    q_out_dtype=36,
) -> (Tensor, Optional[Tensor], Tensor, Tensor, Tensor)
```

`query_start_loc`和`seq_lens`在算子接口上是可选Tensor，但由于它们后面仍有必选参数`head_nums`，两个位置不设置Python默认值。legacy调用表达式及既有参数顺序保持不变；M-RoPE调用需要在这两个固定位置显式传入`None`。新增参数位于keyword-only参数尾部。

## 参数说明

以下两个接口输入参数一致。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| qkv | Tensor | 必选 | Q/K/V融合输入。`layout_qkv="TND"`时shape为`[T,Nq+Nk+Nv,D]`，`layout_qkv="NTD"`时shape为`[Nq+Nk+Nv,T,D]`。 | torch.bfloat16 | 3维 |
| q_gamma | Tensor | 必选 | Q分支RMSNorm权重。 | torch.float32 | `[D]` |
| k_gamma | Tensor | 必选 | K分支RMSNorm权重。 | torch.float32 | `[D]` |
| cos_sin | Tensor | 必选 | RoPE位置编码表，前`D/2`列为cos，后`D/2`列为sin。 | torch.float32 | `[MaxSeqLen,D]` |
| slot_mapping | Tensor | 必选 | 每个token写入cache的slot索引。 | torch.int32 | `[T]` |
| k_cache | Tensor | 必选 | K Cache。原地接口直接更新该Tensor；functional接口更新其副本并返回。legacy为FP8 E4M3FN，M-RoPE为INT8。 | torch.float8_e4m3fn或torch.int8 | `[BlockNum,Nk,BlockSize,D]` |
| v_cache | Tensor | 必选 | V Cache。原地接口直接更新该Tensor；functional接口更新其副本并返回。 | torch.float8_e4m3fn | `[BlockNum,Nv,BlockSize,D]` |
| k_scale_cache | Tensor | 必选 | K动态量化scale cache。原地接口直接更新该Tensor；functional接口更新其副本并返回。 | torch.float32 | `[BlockNum,Nk,BlockSize,1]` |
| query_start_loc | Optional[Tensor] | 可选（接口级） | legacy位置源之一，表示当前调用内各batch token数的前缀和；legacy必须与`seq_lens`同时传入，M-RoPE必须显式传`None`。 | torch.int32 | `[Batch+1]` |
| seq_lens | Optional[Tensor] | 可选（接口级） | legacy位置源之一，表示每个batch追加本次token后的实际序列长度；legacy必须与`query_start_loc`同时传入，M-RoPE必须显式传`None`。 | torch.int32 | `[Batch]` |
| head_nums | List[int] | 必选 | Q/K/V头数数组，必须按`[Nq,Nk,Nv]`传入。 | int | 长度为3 |
| rotation | Optional[Tensor] | 可选 | Q/K共享矩阵乘权重。默认值为`None`，但当前支持场景要求传入有效Tensor。 | torch.bfloat16 | `[D,D]` |
| v_scale | Optional[Tensor] | 可选 | V分支量化缩放因子。默认值为`None`，但当前支持场景要求传入有效Tensor。 | torch.float32 | `[Nv]` |
| layout_qkv | Optional[str] | 可选 | `qkv`的N/T轴布局标识，默认值为`"TND"`；传入`None`或空字符串时按默认值处理。大小写敏感，仅支持`"TND"`和`"NTD"`。 | str | - |
| layout_q_out | Optional[str] | 可选 | `q_out`和存在时的`q_scale`的N/T轴布局标识，默认值为`"NTD"`；传入`None`或空字符串时按默认值处理。大小写敏感，仅支持`"TND"`和`"NTD"`。 | str | - |
| epsilon | float | 可选 | RMSNorm防除零参数，默认值为`1e-6`。 | float | - |
| mrope_position | Optional[Tensor] | 可选（接口级） | M-RoPE唯一位置源。三行分别记录每个token的A/T/W三路`cos_sin`行下标；legacy必须为`None`。公开参数名仅为`mrope_position`，不支持旧草案名`position`。 | torch.int32 | `[3,T]` |
| mrope_section | Optional[List[int]] | 可选（接口级） | M-RoPE的A/T/W lane容量参数`[m,n,k]`；legacy必须为`None`。该参数是ListInt属性，不是int32 Tensor；公开参数名仅为`mrope_section`，不支持旧草案名`msection`。 | int | 长度为3 |
| q_quant_mode | str | 可选 | Q分支量化模式，默认值为`"PerTokenPerHead"`。大小写敏感，仅支持`"PerTokenPerHead"`和`"NoQuant"`。 | str | - |
| q_out_dtype | int | 可选 | `q_out`的数据类型枚举，默认值为`ge::DT_FLOAT8_E4M3FN`（通常为36）。当前仅支持`ge::DT_FLOAT8_E4M3FN`和`ge::DT_BF16`（通常为27）。 | int | - |

## 返回值说明

### qkv_rms_norm_rope_cache_with_k_scale_

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| q_out | Tensor | 必选 | Q分支输出。legacy为FP8 E4M3FN；M-RoPE为BF16。`layout_q_out="TND"`时shape为`[T,Nq,D]`，`layout_q_out="NTD"`时shape为`[Nq,T,D]`。 | torch.float8_e4m3fn或torch.bfloat16 | 3维 |
| q_scale | Optional[Tensor] | 可选（接口级） | legacy为每个token/head对应的FP32动态量化scale；M-RoPE固定返回`None`，不创建空Tensor或占位Tensor。 | torch.float32或None | legacy为2维；M-RoPE为None |

### qkv_rms_norm_rope_cache_with_k_scale

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| q_out | Tensor | 必选 | Q分支输出，shape和dtype规则与原地接口一致。 | torch.float8_e4m3fn或torch.bfloat16 | 3维 |
| q_scale | Optional[Tensor] | 可选（接口级） | legacy返回FP32 scale；M-RoPE在固定的第2个返回位置返回`None`，后续cache返回位置不前移。 | torch.float32或None | legacy为2维；M-RoPE为None |
| k_cache_out | Tensor | 必选 | 更新后的K Cache，shape和dtype与输入`k_cache`一致。 | torch.float8_e4m3fn或torch.int8 | `[BlockNum,Nk,BlockSize,D]` |
| v_cache_out | Tensor | 必选 | 更新后的V Cache，shape和dtype与输入`v_cache`一致。 | torch.float8_e4m3fn | `[BlockNum,Nv,BlockSize,D]` |
| k_scale_cache_out | Tensor | 必选 | 更新后的K动态量化scale cache，shape和dtype与输入`k_scale_cache`一致。 | torch.float32 | `[BlockNum,Nk,BlockSize,1]` |

## 约束说明

- 该接口支持推理场景下的单算子模式调用，当前不支持TorchAir图模式调用。
- 原地接口会修改调用方传入的三个cache；函数式接口会克隆输入cache并返回更新后的副本。
- 当前只支持下面两种完整合同，不支持在两者之间自由组合参数：

  | 条件 | legacy | M-RoPE |
  |---|---|---|
  | query_start_loc / seq_lens | 两者都存在 | 两者都为None |
  | mrope_position / mrope_section | 两者都为None | 两者都存在 |
  | q_quant_mode | `"PerTokenPerHead"` | `"NoQuant"` |
  | q_out_dtype / q_out dtype | FP8 E4M3FN | BF16 |
  | q_scale | FP32 Tensor | None |
  | k_cache dtype | FP8 E4M3FN | INT8 |
  | v_cache dtype | FP8 E4M3FN | FP8 E4M3FN |
  | layout_qkv -> layout_q_out | NTD->NTD、TND->TND、TND->NTD | 仅TND->TND |

- 接口根据四个可选参数的实际presence识别场景，不提供独立`scene`参数。`query_start_loc`与`seq_lens`只存在一个、`mrope_position`与`mrope_section`只存在一个、两组位置源同时存在或同时缺失均非法。零元素Tensor或空List仍视为已提供，并按shape或长度非法处理。
- 当前实现仅支持`D=128`，且`Nv`必须等于`Nk`。
- M-RoPE下令`H=D/2=64`，`mrope_section`三项分别满足`0 <= value <= [22,21,21]`。一般形式为`[ceil(H/3),floor((H+1)/3),floor(H/3)]`。第0项虽需满足`[0,22]`，但其数值不参与交织公式、场景判断或A路采集范围；A/T/W三路始终全部采集。三项之间没有和值约束。
- M-RoPE中cos和sin分别应用`mrope_section`，两者使用相同的half内lane规则；不得把`mrope_section`当作整行连续分段，也不得把结果扩为`3D`。
- `mrope_position`三路下标可以不同、非单调或重复，但每个值必须满足`0 <= value < MaxSeqLen`。
- legacy支持`layout_qkv="TND", layout_q_out="TND"`、`layout_qkv="TND", layout_q_out="NTD"`和`layout_qkv="NTD", layout_q_out="NTD"`；两种场景均不支持`NTD->TND`。
- `k_cache`、`v_cache`和`k_scale_cache`的`BlockNum`和`BlockSize`必须一致。
- 三个cache均为4维正stride，最后一维stride为1；`k_cache`和`v_cache`前三维stride必须一致。
- legacy中`query_start_loc[0]`应为0，`query_start_loc[-1]`应等于`T`，`seq_lens`长度应等于`query_start_loc.shape[0]-1`，且`seq_lens[b] >= query_start_loc[b+1] - query_start_loc[b]`。
- `cos_sin`第一维需覆盖本次调用会访问的所有RoPE位置。
- `slot_mapping`取值范围应为`[0,BlockNum*BlockSize-1]`。同一次调用内多个token写入同一slot时，最终写入顺序和结果未定义。
- 资源边界约束：`Nq+Nk <= 128`，`Nq+Nk+Nv <= 160`，`Nv <= 80`，`ceil_align(Nq,16)+ceil_align(Nk,16) <= 256`。
- <term>Ascend 950PR/Ascend 950DT</term>：支持该接口。

## 确定性计算

默认支持确定性计算。

## 调用说明

- legacy原地调用：

  新参数全部有默认值，因此既有调用表达式保持不变。

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  T, Nq, Nk, Nv, D = 4, 16, 2, 2, 128
  block_num, block_size = 1, 16
  head_nums = [Nq, Nk, Nv]

  qkv = torch.randn(T, Nq + Nk + Nv, D, device="npu", dtype=torch.bfloat16)
  q_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  k_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  cos_sin = torch.zeros(16, D, device="npu", dtype=torch.float32)
  cos_sin[:, : D // 2] = 1.0
  slot_mapping = torch.arange(T, device="npu", dtype=torch.int32)
  k_cache = torch.empty(block_num, Nk, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  v_cache = torch.empty(block_num, Nv, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  k_scale_cache = torch.empty(block_num, Nk, block_size, 1, device="npu", dtype=torch.float32)
  query_start_loc = torch.tensor([0, T], device="npu", dtype=torch.int32)
  seq_lens = torch.tensor([T], device="npu", dtype=torch.int32)
  rotation = torch.eye(D, device="npu", dtype=torch.bfloat16)
  v_scale = torch.ones(Nv, device="npu", dtype=torch.float32)

  q_out, q_scale = cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_(
      qkv,
      q_gamma,
      k_gamma,
      cos_sin,
      slot_mapping,
      k_cache,
      v_cache,
      k_scale_cache,
      query_start_loc,
      seq_lens,
      head_nums,
      rotation=rotation,
      v_scale=v_scale,
  )

  print(q_out.shape, q_out.dtype, q_scale.shape, q_scale.dtype)
  ```

- M-RoPE函数式调用：

  M-RoPE在`query_start_loc`和`seq_lens`的固定位置显式传`None`，并在keyword-only参数中传入完整M-RoPE合同。`q_scale`返回`None`。

  ```python
  import torch
  import torch_npu
  import cann_ops_transformer

  torch_npu.npu.set_device(0)

  T, Nq, Nk, Nv, D = 5, 16, 2, 2, 128
  block_num, block_size = 1, 16
  head_nums = [Nq, Nk, Nv]

  qkv = torch.randn(T, Nq + Nk + Nv, D, device="npu", dtype=torch.bfloat16)
  q_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  k_gamma = torch.ones(D, device="npu", dtype=torch.float32)
  cos_sin = torch.randn(32, D, device="npu", dtype=torch.float32)
  slot_mapping = torch.arange(T, device="npu", dtype=torch.int32)
  k_cache = torch.empty(block_num, Nk, block_size, D, device="npu", dtype=torch.int8)
  v_cache = torch.empty(block_num, Nv, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
  k_scale_cache = torch.empty(block_num, Nk, block_size, 1, device="npu", dtype=torch.float32)
  rotation = torch.eye(D, device="npu", dtype=torch.bfloat16)
  v_scale = torch.ones(Nv, device="npu", dtype=torch.float32)
  mrope_position = torch.tensor(
      [[0, 3, 7, 7, 2], [1, 5, 4, 9, 6], [2, 8, 3, 6, 1]],
      device="npu",
      dtype=torch.int32,
  )

  q_out, q_scale, k_cache_out, v_cache_out, k_scale_cache_out = (
      cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale(
          qkv,
          q_gamma,
          k_gamma,
          cos_sin,
          slot_mapping,
          k_cache,
          v_cache,
          k_scale_cache,
          None,
          None,
          head_nums,
          rotation=rotation,
          v_scale=v_scale,
          layout_qkv="TND",
          layout_q_out="TND",
          mrope_position=mrope_position,
          mrope_section=[22, 12, 10],
          q_quant_mode="NoQuant",
          q_out_dtype=27,  # ge::DT_BF16
      )
  )

  assert q_out.dtype == torch.bfloat16
  assert q_scale is None
  print(q_out.shape, k_cache_out.dtype, v_cache_out.dtype, k_scale_cache_out.shape)
  ```

- TorchAir图模式调用：

  暂不支持TorchAir图模式调用。
