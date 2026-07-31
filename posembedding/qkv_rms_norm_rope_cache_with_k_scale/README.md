# QkvRmsNormRopeCacheWithKScale

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：输入Q/K/V融合张量`qkv`，按`head_nums=[Nq, Nk, Nv]`拆分出Q、K、V。RoPE场景对Q/K执行RMSNorm、RoPE和共享`rotation`矩阵乘，再按token/head动态量化为FP8 E4M3FN；M-RoPE场景使用`mrope_position[T,3]`提供T/H/W位置，Q保持BF16、K量化为INT8。两种场景都按`slot_mapping`更新K/V Cache和K scale cache，V分支按`v_scale`缩放后量化为FP8 E4M3FN。
- 使用场景：推理场景下PagedAttention KV Cache更新，当前仅支持<term>Ascend 950PR/Ascend 950DT</term>上
  `D=128`、`0<Nq<=64`、`Nq=8*Nk`、`Nk=Nv`的实现。RoPE支持三种布局实例；M-RoPE仅支持
  `TND->TND`。
- 计算公式：

  按`head_nums=[Nq, Nk, Nv]`从融合输入中拆分Q、K、V：

  $$
  q, k, v = split(qkv, [Nq, Nk, Nv])
  $$

  两种场景的Q/K分支分别使用`q_gamma`和`k_gamma`做RMSNorm：

  $$
  y = \frac{x}{\sqrt{mean(x^2) + epsilon}} * gamma
  $$

  RoPE场景的Q/K分支执行普通RoPE，`cos_sin[..., :D/2]`为cos，`cos_sin[..., D/2:]`为sin；V分支不执行RoPE：

  $$
  y_{rope} = concat(y_{low} * cos - y_{high} * sin,\ y_{high} * cos + y_{low} * sin)
  $$

  M-RoPE场景从`mrope_position[u,0/1/2]`分别读取T/H/W三路`cos_sin`行，按`mrope_section=[t,h,w]`选择交织half lane；`mrope_section[0]`只参与值域和总和校验，不参与lane选源。

  两种场景的Q/K共享`rotation`矩阵：

  $$
  q_{rot} = q_{rope} @ rotation,\quad k_{rot} = k_{rope} @ rotation
  $$

  RoPE场景的Q/K按每个token和head做动态量化，FP8 E4M3FN最大有限值使用`FP8_E4M3FN_MAX`，该值为448：

  $$
  scale = max(abs(x)) / FP8\_E4M3FN\_MAX,\quad x_{fp8} = cast(x / scale)
  $$

  V分支按`v_scale`缩放后量化：

  $$
  v_{fp8} = cast(v * v\_scale)
  $$

  Cache写回位置由`slot_mapping`决定：

  $$
  blockId = slot\_mapping[t] / BlockSize,\quad blockOffset = slot\_mapping[t]\ \%\ BlockSize
  $$

  $$
  k\_cache[blockId, nk, blockOffset, :] = k_{quant}[t, nk, :]
  $$

  $$
  v\_cache[blockId, nv, blockOffset, :] = v_{fp8}[t, nv, :]
  $$

  $$
  k\_scale\_cache[blockId, nk, blockOffset, 0] = k\_scale[t, nk]
  $$

  其中`k_quant`在RoPE场景为FP8 E4M3FN，在M-RoPE场景为INT8；M-RoPE的Q直接输出BF16且不生成`q_scale`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| qkv | 输入 | Q/K/V融合输入。`layout_qkv="TND"`时shape为`[T, Nq+Nk+Nv, D]`，`layout_qkv="NTD"`时shape为`[Nq+Nk+Nv, T, D]`。 | BFLOAT16 | ND |
| q_gamma | 输入 | Q分支RMSNorm权重，shape为`[D]`。 | FLOAT | ND |
| k_gamma | 输入 | K分支RMSNorm权重，shape为`[D]`。 | FLOAT | ND |
| cos_sin | 输入 | RoPE/M-RoPE位置编码表，shape为`[MaxSeqLen, D]`。前`D/2`列为cos，后`D/2`列为sin。 | FLOAT | ND |
| slot_mapping | 输入 | 每个token写入cache的slot索引，shape为`[T]`。 | INT32 | ND |
| k_cache | 输入/输出 | K Cache，输入输出同地址复用，shape为`[BlockNum, Nk, BlockSize, D]`，支持非连续Tensor。RoPE为FP8 E4M3FN，M-RoPE为INT8。 | FLOAT8_E4M3FN / INT8 | ND |
| v_cache | 输入/输出 | V Cache，输入输出同地址复用，shape为`[BlockNum, Nv, BlockSize, D]`，支持非连续Tensor。 | FLOAT8_E4M3FN | ND |
| k_scale_cache | 输入/输出 | K动态量化scale cache，输入输出同地址复用，shape为`[BlockNum, Nk, BlockSize, 1]`，支持非连续Tensor。 | FLOAT | ND |
| query_start_loc | 可选输入 | RoPE场景当前调用内各batch token数的前缀和，shape为`[Batch+1]`；M-RoPE场景不传。 | INT32 | ND |
| seq_lens | 可选输入 | RoPE场景每个batch追加当前token后的实际序列长度，shape为`[Batch]`；M-RoPE场景不传。 | INT32 | ND |
| rotation | 可选输入 | Q/K共享矩阵乘权重，shape为`[D, D]`。当前不支持不传或空Tensor。 | BFLOAT16 | ND |
| v_scale | 可选输入 | V分支量化缩放因子，shape为`[Nv]`。当前不支持不传或空Tensor。 | FLOAT | ND |
| mrope_position | 可选输入 | M-RoPE场景token-major位置索引，shape为`[T,3]`，三列依次为T/H/W；RoPE场景不传。 | INT32 | ND |
| q_out | 输出 | Q输出。RoPE为FP8 E4M3FN，M-RoPE为BF16；RoPE按`layout_q_out`为`[T,Nq,D]`或`[Nq,T,D]`，M-RoPE固定`[T,Nq,D]`。 | FLOAT8_E4M3FN / BFLOAT16 | ND |
| q_scale | 输出 | Q动态量化scale。RoPE场景返回FP32 scale；M-RoPE场景在ACLNN/Torch接口中分别为`nullptr`/`None`。 | FLOAT | ND |
| head_nums | 属性 | Q/K/V头数，按`[Nq, Nk, Nv]`传入；两个场景均要求`0<Nq<=64`、`Nq=8*Nk`、`Nk=Nv`。 | INT64数组 | - |
| layout_qkv | 可选属性 | `qkv`的N/T轴布局，默认值为`"TND"`。仅支持大小写敏感的`"TND"`和`"NTD"`。 | STRING | - |
| layout_q_out | 可选属性 | `q_out`和`q_scale`的N/T轴布局，默认值为`"NTD"`。仅支持大小写敏感的`"TND"`和`"NTD"`。 | STRING | - |
| epsilon | 可选属性 | RMSNorm防除零参数，默认值为`1e-6`。 | FLOAT | - |
| mrope_section | 可选属性 | M-RoPE场景`[t,h,w]`，非空时长度必须为3；空列表等价于未传。 | INT64数组 | - |
| q_quant_mode | 可选属性 | Q量化模式。RoPE为`PerTokenPerHead`，M-RoPE为`NoQuant`。 | STRING | - |
| q_out_dtype | 可选属性 | Q输出类型。RoPE为FP8 E4M3FN，M-RoPE为BF16。 | INT64 | - |

## 约束说明

- 输入shape限制：
  - 当前实现仅支持`D=128`，RoPE和M-RoPE均要求`0<Nq<=64`、`Nq=8*Nk`、`Nk=Nv`。
  - RoPE场景必须同时传`query_start_loc/seq_lens`，不传`mrope_position`且`mrope_section`为空或未传；
    `q_quant_mode=PerTokenPerHead`、`q_out_dtype=FP8 E4M3FN`、`q_scale`非空，K Cache为FP8。
  - M-RoPE场景必须同时传`mrope_position[T,3]`和非空`mrope_section=[t,h,w]`，不传`query_start_loc/seq_lens`；
    `q_quant_mode=NoQuant`、`q_out_dtype=BF16`、公开`q_scale=None`，K Cache为INT8，且仅支持
    `layout_qkv="TND"`、`layout_q_out="TND"`。两组位置输入不能同时存在或同时缺失。
  - M-RoPE的`mrope_section`长度必须为3。D128时三项分别受`[22,21,21]`上限约束；这些互不重叠的lane容量使三项总量自然不超过`64`，第0项
    只参与值域校验，不参与lane选源。`mrope_position`必须为INT32、rank 2、shape `[T,3]`。
  - `layout_qkv`控制`qkv`的N/T轴布局，默认值为`"TND"`；`layout_q_out`控制`q_out`和`q_scale`的N/T轴布局，默认值为`"NTD"`：
    - `layout_qkv="TND"`，`layout_q_out="TND"`：`qkv=[T, Nq+Nk+Nv, D]`，`q_out=[T, Nq, D]`，`q_scale=[T, Nq]`。
    - `layout_qkv="TND"`，`layout_q_out="NTD"`：`qkv=[T, Nq+Nk+Nv, D]`，`q_out=[Nq, T, D]`，`q_scale=[Nq, T]`。
    - `layout_qkv="NTD"`，`layout_q_out="NTD"`：`qkv=[Nq+Nk+Nv, T, D]`，`q_out=[Nq, T, D]`，`q_scale=[Nq, T]`。
  - `cos_sin`第二维必须为`D`，第一维必须覆盖本次调用会访问的RoPE/M-RoPE位置。
  - `k_cache`、`v_cache`和`k_scale_cache`支持非连续Tensor，需符合以下约束：
    - `k_cache`和`v_cache`均为4维正stride，最后一维stride为1，head维和token维stride均不小于`D=128`；
      K Cache dtype 按场景分别为FP8或INT8，V Cache始终为FP8。
    - `k_scale_cache`为4维正stride，最后一维stride为1。
    - `k_cache`和`v_cache`前三维stride必须一致。
- 输入值域限制：
  - RoPE场景中，`query_start_loc`表示当前调用内token的batch前缀和，`query_start_loc[0]`应为0，`query_start_loc[Batch]`应等于`T`。
  - RoPE场景中，`seq_lens[b]`表示第`b`个batch追加本次token后的实际序列长度。对该batch内第`i`个token，RoPE位置由`seq_lens[b] - (query_start_loc[b+1] - query_start_loc[b]) + i`得到；调用方需保证`seq_lens[b] >= query_start_loc[b+1] - query_start_loc[b]`。若`seq_lens[b]`小于该batch本次调用的token数，行为未定义。
  - `slot_mapping`取值范围应为`[0, BlockNum * BlockSize - 1]`。同一次调用内多个token写入同一slot会对同一cache地址产生重复写，最终写入顺序和结果未定义。
  - 资源边界约束：`Nq+Nk <= 128`，`Nq+Nk+Nv <= 160`，`Nv <= 80`，`ceil_align(Nq,16)+ceil_align(Nk,16) <= 256`。
- 输入属性限制：
  - `head_nums`必须包含3个正整数，顺序为`[Nq, Nk, Nv]`，并满足`Nq<=64`、`Nq=8*Nk`、
    `Nk=Nv`。
  - `layout_qkv`和`layout_q_out`大小写敏感，仅支持`"TND"`和`"NTD"`，且当前不支持`layout_qkv="NTD"`、`layout_q_out="TND"`。
- 输入数据类型限制：
  - 各输入的数据类型和数据格式需符合参数说明，不支持隐式类型转换。
  - 输入均为ND格式，不支持私有格式。
- 其他限制：
  - `rotation`和`v_scale`为可选输入，当前不支持不传或空Tensor。
  - 非可选输入和输出不支持空Tensor；可选位置输入按场景合同使用`None`，空`mrope_section=[]`等价于未传。

## 调用说明

<term>Ascend 950PR/Ascend 950DT</term>

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| aclnn API | [test_aclnn_qkv_rms_norm_rope_cache_with_k_scale](examples/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp) | 通过[aclnnQkvRmsNormRopeCacheWithKScale](docs/aclnnQkvRmsNormRopeCacheWithKScale.md)接口方式调用QkvRmsNormRopeCacheWithKScale算子。 |
| PyTorch API | - | 通过[qkv_rms_norm_rope_cache_with_k_scale](../../torch_extension/cann_ops_transformer/docs/zh/qkv_rms_norm_rope_cache_with_k_scale.md)接口方式调用QkvRmsNormRopeCacheWithKScale算子。 |
