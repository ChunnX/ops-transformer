# cann_ops_transformer.qkv\_rms\_norm\_rope\_cache\_with\_k\_scale

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

    `qkv_rms_norm_rope_cache_with_k_scale`封装`aclnnQkvRmsNormRopeCacheWithKScale`，用于大语言模型推理场景下的Q/K/V预处理与PagedAttention KV Cache更新。接口从融合输入`qkv`中拆分Q、K、V分量，对Q/K执行RMSNorm、Half-and-Half RoPE和共享`rotation`矩阵乘，再动态量化为FP8 E4M3FN；Q分支返回`q_out`和`q_scale`，K分支原地更新`k_cache`和`k_scale_cache`，V分支按`v_scale`缩放后量化并原地更新`v_cache`。

- 语义说明：

    `qkv_rms_norm_rope_cache_with_k_scale`为原地cache更新接口，会修改调用方传入的`k_cache`、`v_cache`和`k_scale_cache`，返回值仅包含`q_out`和`q_scale`。

    `qkv_rms_norm_rope_cache_with_k_scale_functional`为函数式变体，内部先拷贝cache，再对副本执行更新，返回`q_out`、`q_scale`和更新后的三个cache；调用方传入的cache保持不变。

- 计算公式：

    Q/K分支RMSNorm：

    $$
    y = \frac{x}{\sqrt{mean(x^2) + epsilon}} * gamma
    $$

    Q/K分支Half-and-Half RoPE，`cos_sin[..., :D/2]`为cos，`cos_sin[..., D/2:]`为sin；V分支不执行RoPE：

    $$
    y_{rope} = concat(y_{low} * cos - y_{high} * sin,\ y_{high} * cos + y_{low} * sin)
    $$

    Q/K共享矩阵乘与动态量化，FP8 E4M3FN最大有限值使用`FP8_E4M3FN_MAX`，该值为448：

    $$
    q_{rot} = q_{rope} @ rotation,\quad k_{rot} = k_{rope} @ rotation
    $$

    $$
    scale = max(abs(x)) / FP8\_E4M3FN\_MAX,\quad x_{fp8} = cast(x / scale)
    $$

    V分支缩放量化：

    $$
    v_{fp8} = cast(v * v\_scale)
    $$

    Cache写回：

    $$
    blockId = slot\_mapping[t] / BlockSize,\quad blockOffset = slot\_mapping[t]\ \%\ BlockSize
    $$

## 函数原型

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
    cu_seq_len,
    actual_seq_lens,
    head_nums,
    qkv_layout,
    *,
    rotation=None,
    v_scale=None,
    epsilon=0.000001,
) -> (Tensor, Tensor)

cann_ops_transformer.qkv_rms_norm_rope_cache_with_k_scale_functional(
    qkv,
    q_gamma,
    k_gamma,
    cos_sin,
    slot_mapping,
    k_cache,
    v_cache,
    k_scale_cache,
    cu_seq_len,
    actual_seq_lens,
    head_nums,
    qkv_layout,
    *,
    rotation=None,
    v_scale=None,
    epsilon=0.000001,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|---|
| qkv | Tensor | 必选 | Q/K/V融合输入。`qkv_layout="TND"`时shape为`(T, Nq+Nk+Nv, D)`，`qkv_layout="NTD"`时shape为`(Nq+Nk+Nv, T, D)`。 | torch.bfloat16 | 3维 |
| q_gamma | Tensor | 必选 | Q分支RMSNorm权重。 | torch.float32 | `(D,)` |
| k_gamma | Tensor | 必选 | K分支RMSNorm权重。 | torch.float32 | `(D,)` |
| cos_sin | Tensor | 必选 | RoPE位置编码表，前`D/2`列为cos，后`D/2`列为sin。 | torch.float32 | `(MaxSeqLen, D)` |
| slot_mapping | Tensor | 必选 | 每个token写入cache的slot索引。 | torch.int32 | `(T,)` |
| k_cache | Tensor | 必选 | K Cache。原地接口会直接更新该Tensor；functional接口会更新其副本并返回；支持连续和符合约束的非连续Tensor。 | torch.float8_e4m3fn | `(BlockNum, Nk, BlockSize, D)` |
| v_cache | Tensor | 必选 | V Cache。原地接口会直接更新该Tensor；functional接口会更新其副本并返回；支持连续和符合约束的非连续Tensor。 | torch.float8_e4m3fn | `(BlockNum, Nv, BlockSize, D)` |
| k_scale_cache | Tensor | 必选 | K动态量化scale cache。原地接口会直接更新该Tensor；functional接口会更新其副本并返回；支持连续和符合约束的非连续Tensor。 | torch.float32 | `(BlockNum, Nk, BlockSize, 1)` |
| cu_seq_len | Tensor | 必选 | 当前调用内各batch token数的前缀和。 | torch.int32 | `(Batch+1,)` |
| actual_seq_lens | Tensor | 必选 | 每个batch追加当前token后的实际序列长度，用于计算RoPE位置历史偏移。 | torch.int32 | `(Batch,)` |
| head_nums | List[int] | 必选 | Q/K/V头数，按`[Nq, Nk, Nv]`传入。 | int | 长度为3 |
| qkv_layout | str | 必选 | `qkv`、`q_out`和`q_scale`的N/T轴布局。仅支持大小写敏感的`"TND"`和`"NTD"`。 | str | - |
| rotation | Tensor | 可选 | Q/K共享矩阵乘权重。schema默认值为`None`，当前实现要求必须传入非空Tensor。 | torch.bfloat16 | `(D, D)` |
| v_scale | Tensor | 可选 | V分支量化缩放因子。schema默认值为`None`，当前实现要求必须传入非空Tensor。 | torch.float32 | `(Nv,)` |
| epsilon | float | 可选 | RMSNorm防除零参数，默认值为`1e-6`。 | float | - |

## 输出说明

### qkv_rms_norm_rope_cache_with_k_scale

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|
| q_out | Tensor | Q分支FP8 E4M3FN量化输出。`qkv_layout="TND"`时shape为`(T, Nq, D)`，`qkv_layout="NTD"`时shape为`(Nq, T, D)`。 | torch.float8_e4m3fn | 3维 |
| q_scale | Tensor | Q分支每个token/head对应的动态量化scale。`qkv_layout="TND"`时shape为`(T, Nq)`，`qkv_layout="NTD"`时shape为`(Nq, T)`。 | torch.float32 | 2维 |

### qkv_rms_norm_rope_cache_with_k_scale_functional

| 参数名 | 参数类型 | 描述 | 数据类型 | 维度(shape) |
|---|---|---|---|---|
| q_out | Tensor | Q分支FP8 E4M3FN量化输出。 | torch.float8_e4m3fn | 同原地接口 |
| q_scale | Tensor | Q分支动态量化scale。 | torch.float32 | 同原地接口 |
| k_cache_out | Tensor | 更新后的K Cache，shape和dtype与输入`k_cache`一致。 | torch.float8_e4m3fn | `(BlockNum, Nk, BlockSize, D)` |
| v_cache_out | Tensor | 更新后的V Cache，shape和dtype与输入`v_cache`一致。 | torch.float8_e4m3fn | `(BlockNum, Nv, BlockSize, D)` |
| k_scale_cache_out | Tensor | 更新后的K scale cache，shape和dtype与输入`k_scale_cache`一致。 | torch.float32 | `(BlockNum, Nk, BlockSize, 1)` |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口当前仅支持单算子模式调用；未注册torchair图模式converter。
- 仅支持<term>Ascend 950PR/Ascend 950DT</term>。
- 当前实现仅支持`D=128`，且`Nv`必须等于`Nk`。
- `head_nums`必须为长度为3的正整数列表，顺序为`[Nq, Nk, Nv]`。
- `qkv_layout`大小写敏感，仅支持`"TND"`和`"NTD"`。
- `qkv`、`q_out`、`q_scale`保持相同N/T布局：
    - `qkv_layout="TND"`：`qkv=(T, Nq+Nk+Nv, D)`，`q_out=(T, Nq, D)`，`q_scale=(T, Nq)`。
    - `qkv_layout="NTD"`：`qkv=(Nq+Nk+Nv, T, D)`，`q_out=(Nq, T, D)`，`q_scale=(Nq, T)`。
- `cu_seq_len[0]`应为0，`cu_seq_len[-1]`应等于`T`。`actual_seq_lens[b]`表示第`b`个batch追加本次token后的实际序列长度。
- 对第`b`个batch内第`i`个token，RoPE位置由`actual_seq_lens[b] - (cu_seq_len[b+1] - cu_seq_len[b]) + i`计算；调用方需保证`cos_sin`覆盖所有位置。
- `slot_mapping`取值范围应为`[0, BlockNum * BlockSize - 1]`。同一次调用内多个token写入同一slot会对同一cache地址产生重复写，最终写入顺序和结果未定义。
- `rotation`和`v_scale`虽然在schema中为可选参数，但当前仅支持传入非空Tensor。
- cache Tensor支持连续Tensor，也支持非连续Tensor；非连续场景需满足：
    - `k_cache`和`v_cache`均为4维正stride，最后一维stride为1。
    - `k_cache`和`v_cache`前三维stride必须一致。
    - `k_scale_cache`为4维正stride，最后一维stride为1。
- 资源边界约束：`Nq+Nk <= 128`，`Nq+Nk+Nv <= 160`，`Nv <= 80`，`ceil_align(Nq,16)+ceil_align(Nk,16) <= 256`。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import qkv_rms_norm_rope_cache_with_k_scale

    torch_npu.npu.set_device(0)

    T, Nq, Nk, Nv, D = 4, 16, 2, 2, 128
    batch, max_seq_len = 1, 16
    block_num, block_size = 1, 16
    head_nums = [Nq, Nk, Nv]
    qkv_layout = "TND"

    qkv = torch.randn(T, Nq + Nk + Nv, D, device="npu", dtype=torch.bfloat16)
    q_gamma = torch.ones(D, device="npu", dtype=torch.float32)
    k_gamma = torch.ones(D, device="npu", dtype=torch.float32)
    cos_sin = torch.zeros(max_seq_len, D, device="npu", dtype=torch.float32)
    cos_sin[:, : D // 2] = 1.0
    slot_mapping = torch.arange(T, device="npu", dtype=torch.int32)

    k_cache = torch.empty(block_num, Nk, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
    v_cache = torch.empty(block_num, Nv, block_size, D, device="npu", dtype=torch.float8_e4m3fn)
    k_scale_cache = torch.empty(block_num, Nk, block_size, 1, device="npu", dtype=torch.float32)
    cu_seq_len = torch.tensor([0, T], device="npu", dtype=torch.int32)
    actual_seq_lens = torch.tensor([T], device="npu", dtype=torch.int32)
    rotation = torch.eye(D, device="npu", dtype=torch.float32).to(torch.bfloat16)
    v_scale = torch.ones(Nv, device="npu", dtype=torch.float32)

    q_out, q_scale = qkv_rms_norm_rope_cache_with_k_scale(
        qkv,
        q_gamma,
        k_gamma,
        cos_sin,
        slot_mapping,
        k_cache,
        v_cache,
        k_scale_cache,
        cu_seq_len,
        actual_seq_lens,
        head_nums,
        qkv_layout,
        rotation=rotation,
        v_scale=v_scale,
        epsilon=1e-6,
    )

    print(q_out.shape, q_out.dtype, q_scale.shape, q_scale.dtype)
    ```

- 函数式调用

    ```python
    from cann_ops_transformer.ops import qkv_rms_norm_rope_cache_with_k_scale_functional

    q_out, q_scale, k_cache_out, v_cache_out, k_scale_cache_out = (
        qkv_rms_norm_rope_cache_with_k_scale_functional(
            qkv,
            q_gamma,
            k_gamma,
            cos_sin,
            slot_mapping,
            k_cache,
            v_cache,
            k_scale_cache,
            cu_seq_len,
            actual_seq_lens,
            head_nums,
            qkv_layout,
            rotation=rotation,
            v_scale=v_scale,
        )
    )
    ```

- 图模式调用

    暂不支持图模式调用。
