# flash\_mla\_with\_kvcache

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

  `flash_mla_with_kvcache`是基于`TorchNPU`的`cann_ops_transformer`扩展接口，用于调用`FlashMlaWithKvcache`算子完成DeepSeek MLA（Multi-head Latent Attention，非量化）注意力计算，支持prefill与decoder（flash-decode）场景。MLA将Key/Value压缩为低秩潜向量，注意力计算共享单一KV缓存：`k_cache`中同时存放Key与Value数据（**不单独传入v**），且**RoPE已合并进`q`与`k_cache`的最后64位**（q/k_cache最后维恒为576 = nope 512 + rope 64），接口不再有独立的`q_rope`/`k_rope`张量。

  `flash_mla_with_kvcache_metadata`是`flash_mla_with_kvcache`的元数据生成接口，用于在主算子执行前生成metadata。metadata记录AICore/AIVCore的任务切分结果（含多section调度信息），主算子可选择传入该metadata以优化调度。典型调用流程如下：

  1. 准备`q`、`k_cache`、`block_table`、`cache_seqlens`、`cu_seqlens_q`等输入。
  2. 调用`flash_mla_with_kvcache_metadata`生成`metadata`。
  3. 调用`flash_mla_with_kvcache`，将上一步得到的`metadata`传入主算子。

- **计算公式**：

  MLA前向计算由两阶段[（矩阵乘 + 注意力）]构成：先经低秩KV压缩得到潜向量，再基于潜向量执行缩放点积注意力。注意力部分的计算公式为：

  $$
  Attention(Q, K, V)=Softmax(\frac{QK^T}{\sqrt{d_k}})V
  $$

  其中$Q$由$q$（含RoPE）经线性变换得到，$K、V$由共享的$k\_cache$（含RoPE）经线性变换得到，$\sqrt{d_k}$为缩放因子（可通过`softmax_scale`显式覆盖）。

  开启**return_softmax_lse**之后，返回值softmax_lse计算逻辑如下所示：

  $$
  S = \frac{QK^T}{d_k^{1/2}}
  $$

  $$
  softmax\_max = max(S)
  $$

  $$
  softmax\_lse = log{\sum e^{S-softmax\_max}} + softmax\_max
  $$

> [!NOTE]
>
> 本接口仅支持有限布局组合（R1）：`layout_q="TND"` + `layout_kv="NZ"`（PA_NZ）+ `layout_out="NTD"`，且`num_heads_kv==1`（单latent KV头）。其中B（Batch）表示输入样本批量大小、Q_T表示所有query batch序列长度的累加和、Q_N表示heads_q、D表示头维（nope部分512，整体576含rope64）。

## 函数原型

调用flash_mla_with_kvcache接口之前，请先调用前置接口flash_mla_with_kvcache_metadata，完成flash_mla_with_kvcache负载均衡的计算。

```python
cann_ops_transformer.flash_mla_with_kvcache_metadata(
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    num_heads_q,
    num_heads_kv,
    head_dim,
    mask_mode=None,
    layout_q=None,
    layout_kv=None,
    layout_out=None,
    *,
    cu_seqlens_q=None,
    cache_seqlens=None,
    seqused_q=None
) -> Tensor
```

```python
cann_ops_transformer.flash_mla_with_kvcache(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="TND",
    layout_kv="NZ",
    layout_out="NTD",
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## 参数说明

### flash_mla_with_kvcache_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| batch_size | int | 可选 | batch大小。若未传入，则从cu_seqlens_q或seqused_q推导 | int32 | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - |
| max_seqlen_kv | int | 可选 | 指定KV缓存序列的长度上限 | int32 | - | - |
| num_heads_q | int | 必选 | Query head数 | int32 | - | - |
| num_heads_kv | int | 必选 | Key/Value head数，MLA场景固定为1 | int32 | - | - |
| head_dim | int | 必选 | 每个注意力头的维度（qk nope部分，512） | int32 | - | - |
| mask_mode | int | 可选 | 掩码模式，仅支持NO_MASK与RIGHT_DOWN两种取值（对应值0与3，取值与fused_infer_attention_score的sparse_mode约定一致），其余取值将被拒绝 | int32 | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式，仅支持"TND" | string | - | - |
| layout_kv | string | 可选 | 定义输入k_cache张量的布局格式，仅支持"NZ" | string | - | - |
| layout_out | string | 可选 | 定义输出张量的布局格式，仅支持"NTD" | string | - | - |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) |
| cache_seqlens | Tensor | 可选 | 每batch的KV缓存实际长度（seqused_kv语义） | int32 | ND | (B,) |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的查询序列长度，截断冗余运算 | int32 | ND | (B,) |

### flash_mla_with_kvcache

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的q（含RoPE，最后64维为rope） | bfloat16/float16 | ND | (Q_T, Q_N, 576) |
| k_cache | Tensor | 必选 | 共享KV缓存（Key与Value同一份输入，最后64维为rope） | bfloat16/float16 | ND | (num_blocks, 1, D//16, block_size, 16)、即PA_NZ布局，(num_blocks, 1, 块, 576) |
| block_table | Tensor | 可选 | 用于分块注意力计算中的块索引映射 | int32 | ND | (B, max_num_blocks_per_seq) |
| cache_seqlens | Tensor | 可选 | 每batch的KV缓存实际长度（seqused_kv语义） | int32 | ND | (B,) |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0；layout_q为TND时必传 | int32 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的查询序列长度，截断冗余运算 | int32 | ND | (B,) |
| attn_mask | Tensor | 可选 | 掩码矩阵（RIGHT_DOWN模式下配合使用） | int8 | ND | (2048, 2048) |
| metadata | Tensor | 可选 | `flash_mla_with_kvcache_metadata`生成的任务切分结果，传入后可优化调度 | int32 | ND | (max_schedule_size,) |
| head_dim_v | int | 可选 | nope（value）部分宽度，恒为512（=576−64，即 k_cache 最后维减去 rope 宽度）；传入其它值将被拒绝 | int32 | - | - |
| softmax_scale | float | 可选 | 可显式设置缩放因子，覆盖默认计算 | float32 | - | - |
| mask_mode | int | 可选 | 掩码模式，仅支持NO_MASK与RIGHT_DOWN两种取值（对应值0与3，取值与fused_infer_attention_score的sparse_mode约定一致），其余取值将被拒绝 | int32 | - | - |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - |
| max_seqlen_kv | int | 可选 | 指定KV缓存序列的长度上限 | int32 | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式，仅支持"TND" | string | - | - |
| layout_kv | string | 可选 | 定义输入k_cache张量的布局格式，仅支持"NZ" | string | - | - |
| layout_out | string | 可选 | 定义输出张量的布局格式，仅支持"NTD" | string | - | - |
| return_softmax_lse | bool | 可选 | 是否需要获取softmax的LSE结果 | bool | - | - |

## 返回值说明

### flash_mla_with_kvcache_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | flash_mla_with_kvcache的任务切分数据 | int32 | ND | shape根据batch_size动态计算，容量上界为((36+72)×batch_size×1+1)×16对齐4096 |

### flash_mla_with_kvcache

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attn_out | Tensor | 必选 | flash_mla_with_kvcache的计算输出 | bfloat16/float16 | ND | layout_out为"NTD"（默认）时：(Q_N, Q_T, D)；layout_out为"TND"时：(Q_T, Q_N, D)。D=head_dim_v |
| softmax_lse | Tensor | 可选 | softmax的LSE结果 | float32 | ND | (Q_N, Q_T) |

**说明**

- attn_out：输出D维度与`head_dim_v`保持一致（512）；维度0/1由layout_out决定：NTD为(Q_N, Q_T)，TND为(Q_T, Q_N)（NTD语义与fused_infer_attention_score的outLayout="NTD"一致）。
- softmax_lse：return_softmax_lse为True时，输出shape为(Q_N, Q_T)的Tensor；return_softmax_lse为False时，则输出shape为[1]的值为0的Tensor。

## 约束说明

- 参数cu_seqlens_q、cache_seqlens、seqused_q、block_table及attn_mask属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- flash_mla_with_kvcache_metadata和flash_mla_with_kvcache的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由客户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- metadata由`flash_mla_with_kvcache_metadata`生成后原样传入`flash_mla_with_kvcache`；用户自建过小的metadata缓冲区将被容量校验拒绝。
- 本接口的布局路由（R1）限定为：`layout_q="TND"`（需传cu_seqlens_q）、`layout_kv="NZ"`（PA_NZ，需传block_table）、`layout_out="NTD"`；其余布局组合（BSND/BNSD查询、连续KV、非NTD输出等）一律由算子校验拒绝。
- `num_heads_kv`必须为1（单latent KV头）；`head_dim_v`必须为512；`q`与`k_cache`最后维必须为576（= head_dim_v 512 + rope 64），不允许512宽输入。
- 掩码仅支持NO_MASK与RIGHT_DOWN（右下三角/因果类）两种模式，其余mask_mode取值将被拒绝。
- 空Tensor（必有输入的任一轴shape为0）将被全部拦截报错。

### 特性参数组

|      特性参数组      |    参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |       k_cache       |      INPUT      | Tensor |
|                      |      head_dim_v      | ATTR(OPTIONAL) |   int   |
|                      |      softmax_scale      | ATTR(OPTIONAL) |   float   |
|                      |      layout_q      | ATTR(OPTIONAL) |   string   |
|                      |      layout_kv      | ATTR(OPTIONAL) |   string   |
|                      |      layout_out      | ATTR(OPTIONAL) |   string   |
|                      |      metadata        |      INPUT(OPTIONAL)      | Tensor   |
|                      |     attn_out     |     OUTPUT     |   Tensor   |
|      Mask参数组      |       mask_mode       | ATTR(OPTIONAL) |   int   |
|                      |      attn_mask      | INPUT(OPTIONAL) |   Tensor   |
| SeqLens参数组  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  cache_seqlens  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_q  | INPUT(OPTIONAL) |  Tensor  |
|                      |  max_seqlen_q  | ATTR(OPTIONAL) |  int  |
|                      |  max_seqlen_kv  | ATTR(OPTIONAL) |  int  |
| Paged Attention参数组 |      block_table      | INPUT(OPTIONAL) |   Tensor   |
|   SoftmaxLSE参数组   |    return_softmax_lse    | ATTR(OPTIONAL) |    bool    |
|                      |      softmax_lse      |     OUTPUT     |   Tensor   |

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch,表示输入样本批量大小                |
|     Q_N     |        输入q tensor的头数，对应q shape中的N        |
|     Q_T     |          输入q tensor所有batch序列长度的累加和          |
|     D     |          输出attn_out隐藏层最小的单元尺寸headdim（=head_dim_v=512）         |

## 调用示例

- flash_mla_with_kvcache_metadata + flash_mla_with_kvcache联合调用示例（TND / PA_NZ / NTD，RIGHT_DOWN掩码）

    ```python
    import torch
    import torch_npu
    import cann_ops_transformer

    torch_npu.npu.set_device(0)

    dtype = torch.bfloat16
    B = 2            # batch
    Q_T = 8          # total query tokens (TND)
    Q_N = 4          # query heads
    HEAD_DIM_NOPE = 512
    ROPE_DIM = 64
    HEAD_DIM = HEAD_DIM_NOPE + ROPE_DIM  # 576，q/k_cache 最后维
    KV_N = 1         # MLA 单 latent KV 头
    BLOCK_SIZE = 64
    NUM_BLOCKS = 8
    S = 16           # max seqlen per batch

    q = torch.randn(Q_T, Q_N, HEAD_DIM, dtype=dtype, device="npu")
    # k_cache 为 PA_NZ 布局（数值上简化为 (NUM_BLOCKS, KV_N, BLOCK_SIZE, HEAD_DIM) 便于示例）
    k_cache = torch.randn(NUM_BLOCKS, KV_N, BLOCK_SIZE, HEAD_DIM, dtype=dtype, device="npu")
    cu_seqlens_q = torch.tensor([0, 4, 8], dtype=torch.int32, device="npu")
    cache_seqlens = torch.tensor([8, 8], dtype=torch.int32, device="npu")
    block_table = torch.zeros(B, 2, dtype=torch.int32, device="npu")

    metadata = cann_ops_transformer.ops.flash_mla_with_kvcache_metadata(
        batch_size=B,
        max_seqlen_q=S,
        max_seqlen_kv=S,
        num_heads_q=Q_N,
        num_heads_kv=KV_N,
        head_dim=HEAD_DIM_NOPE,
        mask_mode=3,  # RIGHT_DOWN/因果（与 fused_infer_attention_score 的 sparse_mode 约定一致）
        layout_q="TND",
        layout_kv="NZ",
        layout_out="NTD",
        cu_seqlens_q=cu_seqlens_q,
        cache_seqlens=cache_seqlens,
    )

    attn_out, softmax_lse = cann_ops_transformer.ops.flash_mla_with_kvcache(
        q,
        k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        metadata=metadata,
        head_dim_v=HEAD_DIM_NOPE,
        softmax_scale=1.0 / (HEAD_DIM_NOPE ** 0.5),
        mask_mode=3,  # RIGHT_DOWN/因果
        max_seqlen_q=S,
        max_seqlen_kv=S,
        layout_q="TND",
        layout_kv="NZ",
        layout_out="NTD",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == (Q_N, Q_T, HEAD_DIM_NOPE)
    assert attn_out.dtype == q.dtype
    assert torch.isfinite(attn_out.float()).all().item()
    ```

> [!NOTE]
>
> 图模式调用（暂不支持）：`flash_mla_with_kvcache`的图模式GE转换暂未支持，图模式下调用会报"GE not supported"。
