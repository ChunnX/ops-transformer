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

- 算子功能：输入Q/K/V融合张量`qkv`，按`head_nums=[Nq, Nk, Nv]`拆分出Q、K、V；对Q/K执行RMSNorm、Half-and-Half RoPE和共享`rotation`矩阵乘，再动态量化为FP8 E4M3FN；Q分支输出`q_out`和`q_scale`，K分支按`slot_mapping`更新`k_cache`与`k_scale_cache`；V分支按`v_scale`缩放后量化为FP8 E4M3FN，并更新`v_cache`。
- 本算子主要用于推理场景下PagedAttention KV Cache更新，当前仅支持Ascend 950上`D=128`的实现。

- 计算公式：

  (1) Split:

  $$
  q, k, v = split(qkv, [Nq, Nk, Nv])
  $$

  (2) RMSNorm，Q/K分支分别使用`q_gamma`和`k_gamma`：

  $$
  y = \frac{x}{\sqrt{mean(x^2) + epsilon}} * gamma
  $$

  (3) Q/K分支RoPE，`cos_sin[..., :D/2]`为cos，`cos_sin[..., D/2:]`为sin；V分支不执行RoPE：

  $$
  y_{rope} = concat(y_{low} * cos - y_{high} * sin,\ y_{high} * cos + y_{low} * sin)
  $$

  (4) Q/K共享矩阵乘：

  $$
  q_{rot} = q_{rope} @ rotation
  $$

  $$
  k_{rot} = k_{rope} @ rotation
  $$

  (5) Q/K动态量化，FP8 E4M3FN最大有限值使用`FP8_E4M3FN_MAX`，该值为448：

  $$
  scale = max(abs(x)) / FP8\_E4M3FN\_MAX
  $$

  $$
  x_{fp8} = cast(x / scale)
  $$

  (6) V分支静态缩放量化：

  $$
  v_{fp8} = cast(v * v\_scale)
  $$

  (7) Cache写回：

  $$
  blockId = slot\_mapping[t] / BlockSize
  $$

  $$
  blockOffset = slot\_mapping[t]\ \%\ BlockSize
  $$

  $$
  k\_cache[blockId, nk, blockOffset, :] = k_{fp8}[t, nk, :]
  $$

  $$
  v\_cache[blockId, nv, blockOffset, :] = v_{fp8}[t, nv, :]
  $$

  $$
  k\_scale\_cache[blockId, nk, blockOffset, 0] = k\_scale[t, nk]
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| qkv | 输入 | Q/K/V融合输入。`qkv_layout="TND"`时shape为`[T, Nq+Nk+Nv, D]`，`qkv_layout="NTD"`时shape为`[Nq+Nk+Nv, T, D]`。 | BFLOAT16 | ND |
| q_gamma | 输入 | Q分支RMSNorm权重，shape为`[D]`。 | FLOAT | ND |
| k_gamma | 输入 | K分支RMSNorm权重，shape为`[D]`。 | FLOAT | ND |
| cos_sin | 输入 | RoPE位置编码表，shape为`[MaxSeqLen, D]`。前`D/2`列为cos，后`D/2`列为sin。 | FLOAT | ND |
| slot_mapping | 输入 | 每个token写入cache的slot索引，shape为`[T]`。 | INT32 | ND |
| k_cache | 输入/输出 | K Cache，输入输出同地址复用，shape为`[BlockNum, Nk, BlockSize, D]`，支持连续和符合约束的非连续Tensor。 | FLOAT8_E4M3FN | ND |
| v_cache | 输入/输出 | V Cache，输入输出同地址复用，shape为`[BlockNum, Nv, BlockSize, D]`，支持连续和符合约束的非连续Tensor。 | FLOAT8_E4M3FN | ND |
| k_scale_cache | 输入/输出 | K动态量化scale cache，输入输出同地址复用，shape为`[BlockNum, Nk, BlockSize, 1]`，支持连续和符合约束的非连续Tensor。 | FLOAT | ND |
| cu_seq_len | 输入 | 当前调用内各batch token数的前缀和，shape为`[Batch+1]`。 | INT32 | ND |
| actual_seq_lens | 输入 | 每个batch追加当前token后的实际序列长度，shape为`[Batch]`。 | INT32 | ND |
| rotation | 输入 | Q/K共享矩阵乘权重，shape为`[D, D]`，当前必须传入非空Tensor。 | BFLOAT16 | ND |
| v_scale | 输入 | V分支量化缩放因子，shape为`[Nv]`，当前必须传入非空Tensor。 | FLOAT | ND |
| head_nums | 属性 | Q/K/V头数，按`[Nq, Nk, Nv]`传入。 | INT64数组 | - |
| qkv_layout | 属性 | `qkv`、`q_out`和`q_scale`的N/T轴布局。仅支持大小写敏感的`"TND"`和`"NTD"`。 | STRING | - |
| epsilon | 可选属性 | RMSNorm防除零参数，默认值为`1e-6`。 | FLOAT | - |
| q_out | 输出 | Q分支FP8 E4M3FN量化输出。`"TND"`时shape为`[T, Nq, D]`，`"NTD"`时shape为`[Nq, T, D]`。 | FLOAT8_E4M3FN | ND |
| q_scale | 输出 | Q分支动态量化scale。`"TND"`时shape为`[T, Nq]`，`"NTD"`时shape为`[Nq, T]`。 | FLOAT | ND |

## 约束说明

- 仅支持<term>Ascend 950PR/Ascend 950DT</term>。
- 当前实现仅支持`D=128`，且`Nv=Nk`。
- `head_nums`必须包含3个正整数，顺序为`[Nq, Nk, Nv]`。
- `qkv_layout`大小写敏感，仅支持`"TND"`和`"NTD"`。
- `qkv`、`q_out`、`q_scale`保持相同N/T布局：
  - `qkv_layout="TND"`：`qkv=[T, Nq+Nk+Nv, D]`，`q_out=[T, Nq, D]`，`q_scale=[T, Nq]`。
  - `qkv_layout="NTD"`：`qkv=[Nq+Nk+Nv, T, D]`，`q_out=[Nq, T, D]`，`q_scale=[Nq, T]`。
- `cu_seq_len`表示当前调用内token的batch前缀和，`cu_seq_len[0]`应为0，`cu_seq_len[Batch]`应等于`T`。
- `actual_seq_lens[b]`表示第`b`个batch追加本次token后的实际序列长度。对该batch内第`i`个token，RoPE位置由`actual_seq_lens[b] - (cu_seq_len[b+1] - cu_seq_len[b]) + i`得到；调用方需保证`actual_seq_lens[b] >= cu_seq_len[b+1] - cu_seq_len[b]`且`cos_sin`覆盖对应位置。若`actual_seq_lens[b]`小于该batch本次调用的token数，行为未定义。
- `slot_mapping`取值范围应为`[0, BlockNum * BlockSize - 1]`。同一次调用内多个token写入同一slot会对同一cache地址产生重复写，最终写入顺序和结果未定义。
- `rotation`和`v_scale`当前必须传入非空Tensor。
- 输入和输出均为ND格式，不支持私有格式；不支持空Tensor。
- `k_cache`、`v_cache`和`k_scale_cache`支持连续Tensor，也支持符合以下约束的非连续Tensor：
  - `k_cache`、`v_cache`和`k_scale_cache`均为4维正stride，最后一维stride为1。
  - `k_cache`和`v_cache`前三维stride必须一致。
  - `k_scale_cache`前三维stride作为block/head/token写回stride使用，输入和原地更新后的Tensor必须保持同一组前三维stride。
- 资源边界约束：`Nq+Nk <= 128`，`Nq+Nk+Nv <= 160`，`Nv <= 80`，`ceil_align(Nq,16)+ceil_align(Nk,16) <= 256`。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---|---|---|
| aclnn接口 | [test_aclnn_qkv_rms_norm_rope_cache_with_k_scale](examples/test_aclnn_qkv_rms_norm_rope_cache_with_k_scale.cpp) | 通过[aclnnQkvRmsNormRopeCacheWithKScale](docs/aclnn_qkv_rms_norm_rope_cache_with_k_scale.md)接口方式调用本算子。 |
