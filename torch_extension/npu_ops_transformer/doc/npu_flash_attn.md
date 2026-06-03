# npu_flash_attn<a name="ZH-CN_TOPIC_0000002309174912"></a>

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |


## 功能说明<a name="zh-cn_topic_0000002168254826_section14441124184110"></a>

-   API功能:

    支持Flash Attention训练推理归一化。
-   计算公式:
    
    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n \times d$的矩阵。

    self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$\sqrt{d}$进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n \times d$的矩阵。

    增加**sink**之后计算逻辑如下所示，主要修改相关softmax_max和softmax_sum逻辑计算部分。
    
    $$
    S = \frac{QK^T}{\sqrt{d}}
    $$
    
    $$
    m = max(sink, max(S))
    $$
    
    $$
    Attention = \frac{e^{S - m} \times V}{\sum e^{S-m} + e^{sink - m}}
    $$

    开启**return_softmax_lse**之后，返回值softmax_lse计算逻辑如下所示：

    $$
    S = \frac{QK^T}{\sqrt{d}}
    $$
    
    $$
    softmax\_max = max(S)
    $$

    $$
    softmax\_lse = log{\sum e^{S-softmax\_max}} + softmax\_max
    $$

**说明**
-   q、k、v数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、Q_T表示所有query Batch输入样本序列长度的累加和，KV_T表示所有k、v Batch输入样本序列长度的累加和。
-   Q_S表示seqlen_q，Q_N表示nheads_q，KV_S表示seqlen_kv，KV_N表示nheads_kv。

## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
npu_flash_attn(q, k, v, block_table=None, cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None, sinks=None, metadata=None, softmax_scale=1.0, mask_mode=0, attn_mask=None, win_left=-1, win_right=-1, max_seqlen_q=-1, max_seqlen_kv=-1, layout_q="BSND", layout_kv="BSND", layout_out="BSND", return_softmax_lse=0) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002168254826_section112637109429"></a>

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:---|:---|:---|:---|:---|:---|:---:|
| q | 必选输入 | 公式中的q | BFLOAT16/FLOAT16 | ND | (B, Q_S, Q_N, D) / (B, Q_N, Q_S, D) / (Q_T, Q_N, D) | ✓ |
| k | 必选输入 | 公式中的k | BFLOAT16/FLOAT16 | ND | (B, KV_S, KV_N, D) / (B, KV_N, KV_S, D) / (KV_T, KV_N, D) / (num_blocks, block_size, KV_N, D) / (num_blocks, KV_N, block_size, D) | × |
| v | 必选输入 | 公式中的v | BFLOAT16/FLOAT16 | ND | (B, KV_S, KV_N, D) / (B, KV_N, KV_S, D) / (KV_T, KV_N, D) / (num_blocks, block_size, KV_N, D) / (num_blocks, KV_N, block_size, D) | × |
| block_table | 可选输入 | 用于分块注意力计算中的块索引映射 | INT32 | ND | (B, max_num_blocks_per_seq) | ✓ |
| cu_seqlens_q | 可选输入 | 累积序列长度，用于处理变长序列 | INT32 | ND | (B+1,) | ✓ |
| cu_seqlens_kv | 可选输入 | 累积序列长度，用于处理变长序列 | INT32 | ND | (B+1,) | ✓ |
| seqused_q | 可选输入 | 指定每batch中实际使用的序列长度，截断冗余运算 | INT32 | ND | (B,) | ✓ |
| seqused_kv | 可选输入 | 指定每batch中实际使用的序列长度，截断冗余运算 | INT32 | ND | (B,) | ✓ |
| sinks | 可选输入 | 暂不支持 | FLOAT32 | ND | (Q_N,) | ✓ |
| metadata | 可选输入 | metadata | INT32 | ND | (max_schedule_size,) | ✓ |
| softmax_scale | 可选输入 | 可显式设置缩放因子，覆盖默认计算 | FLOAT32 | N/A | - | - |
| mask_mode | 可选输入 | 掩码模式 | INT | N/A | - | - |
| attn_mask | 可选输入 | 掩码矩阵 | INT8 | ND | (2048, 2048) | ✓ |
| win_left | 可选输入 | window左界限 | INT | - | - | - |
| win_right | 可选输入 | window右界限 | INT | - | - | - |
| max_seqlen_q | 可选输入 | 指定查询q序列的长度上限 | INT | - | - | - |
| max_seqlen_kv | 可选输入 | 指定键k和值v序列的长度上限 | INT | - | - | - |
| layout_q | 可选输入 | 定义输入q张量的布局格式 | STRING | N/A | - | - |
| layout_kv | 可选输入 | 定义输入k和v张量的布局格式 | STRING | N/A | - | - |
| layout_out | 可选输入 | 定义输出张量的布局格式 | STRING | N/A | - | - |
| return_softmax_lse | 可选输入 | 是否需要获取softmax的LSE结果 | INT | N/A | - | - |

### 基准信息说明

资料约束中，常见字段释义如下：

| 命名 | 含义 |
|:---:|:---|
| B | Batch, 表示输入样本批量大小 |
| Q_N | 输入q tensor的头数，对应q shape中的N |
| KV_N | 输入k/v tensor的头数，对应k/v shape中的N |
| Q_S | 输入q tensor的序列长度，对应q shape中的S |
| KV_S | 输入k/v tensor的序列长度，对应k/v shape中的S |
| Q_T | 输入q tensor所有batch序列长度的累加和 |
| KV_T | 输入k/v tensor所有batch序列长度的累加和 |
| D | 输入q/k/v tensor以及输出attn_out隐藏层最小的单元尺寸headdim |

## 返回值说明<a name="zh-cn_topic_0000002168254826_section22231435517"></a>

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:---|:---|:---|:---|:---|:---|:---:|
| attn_out | 固定输出 | flash_attn的计算输出。限制：该输出参数的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致 | BFLOAT16/FLOAT16 | ND | (B, Q_S, Q_N, D) / (B, Q_N, Q_S, D) / (Q_T, Q_N, D) | ✓ |
| softmax_lse | 可选输出 | softmax的LSE结果。对query乘key的结果先取max得到softmax_max，再减去softmax_max后取exp，最后取sum得到softmax_sum，对softmax_sum取log再加上softmax_max。return_softmax_lse为1时，一般情况下输出shape为(B, Q_N, Q_S)的Tensor，当input_q为TND时输出shape为(Q_N, Q_T)的Tensor；return_softmax_lse为0时，输出shape为[1]的值为0的Tensor | FLOAT32 | ND | (B, Q_N, Q_S) / (Q_N, Q_T) | ✓ |

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

- 声明
  - 参数 cu_seqlens_q、cu_seqlens_kv、seqused_q、seqused_kv、block_table 及 attn_mask 属于tensor。由于算子在 Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。

### 特性参数组

| 特性参数组 | 参数字段名称 | 字段分组 | 字段类型 |
|:---|:---|:---|:---|
| 公共参数组 | q | INPUT | Tensor |
| | k | INPUT | Tensor |
| | v | INPUT | Tensor |
| | softmax_scale | ATTR(OPTIONAL) | double |
| | layout_q | ATTR(OPTIONAL) | string |
| | layout_kv | ATTR(OPTIONAL) | string |
| | layout_out | ATTR(OPTIONAL) | string |
| | attn_out | OUTPUT | Tensor |
| Mask参数组 | mask_mode | ATTR(OPTIONAL) | int |
| | win_left | ATTR(OPTIONAL) | int |
| | win_right | ATTR(OPTIONAL) | int |
| | attn_mask | INPUT(OPTIONAL) | Tensor |
| SeqLens参数组 | cu_seqlens_q | INPUT(OPTIONAL) | Tensor |
| | cu_seqlens_kv | INPUT(OPTIONAL) | Tensor |
| | seqused_q | INPUT(OPTIONAL) | Tensor |
| | seqused_kv | INPUT(OPTIONAL) | Tensor |
| | max_seqlen_q | ATTR(OPTIONAL) | int |
| | max_seqlen_kv | ATTR(OPTIONAL) | int |
| Paged Attention参数组 | block_table | INPUT(OPTIONAL) | Tensor |
| Sinks参数组 | sinks | INPUT(OPTIONAL) | Tensor |
| SoftmaxLSE参数组 | return_softmax_lse | ATTR(OPTIONAL) | int |
| | softmax_lse | OUTPUT | Tensor |

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：
    - 空Tensor指必选输入和输出的shape size为0, 即有任意轴为0。
    - 触发空tensor的用例将全部拦截报错。
- q、k、v、attn_out校验

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
|:---|:---|:---|:---|:---|
| q | tensor_type支持BFLOAT16和FLOAT16；BNSD -> (B, N_Q, S_Q, D)；BSND -> (B, S_Q, N_Q, D)；TND -> (T_Q, N_Q, D) | 必须存在 | q、k、v、attn_out的数据类型需相同；k、v的shape需相同；Layout校验规则见layout匹配关系表 | 轴校验：65536 > B > 0；Q_T > 0；KV_T > 0；Q_N > 0；KV_N > 0；Q_S > 0；KV_S > 0；D = 128；Q_N % KV_N == 0 且 Q_N / KV_N > 0 |
| k | tensor_type支持BFLOAT16和FLOAT16；BNSD -> (B, N_KV, S_KV, D)；BSND -> (B, S_KV, N_KV, D)；TND -> (T_KV, N_KV, D)；PA_BBND -> (num_blocks, block_size, KV_N, D)；PA_BNBD -> (num_blocks, KV_N, block_size, D)；1024 >= block_size >= 16，block_size % 16 == 0 | 必须存在 | 同上 | 同上 |
| v | 同k | 必须存在 | 同上 | 同上 |
| attn_out | data_type支持BFLOAT16和FLOAT16；shape维度支持3、4 | 必须存在 | 同上 | 同上 |

layout匹配关系表：

| layout_q | layout_kv | layout_out | layout_softmax_lse |
|:---|:---|:---|:---|
| BNSD | BNSD / PA_BBND / PA_BNBD | BNSD / BSND | (B, Q_N, Q_S) |
| BSND | BSND / PA_BBND / PA_BNBD | BSND | (B, Q_N, Q_S) |
| TND | TND / PA_BBND / PA_BNBD | TND | (Q_N, Q_T) |

#### Mask参数组

mask_mode参数解释：
- mask_mode=0，全计算模式（默认值）
- mask_mode=3，Causal模式

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
|:---|:---|:---|:---|:---|
| mask_mode | data_type支持INT；支持输入范围仅为 0、3，默认值为 0 | 可选输入，如果不传该参数，默认值为0 | 当mask_mode为0时，win_left和win_right必须为-1以及不支持传入attn_mask；当mask_mode为3时，win_left和win_right必须为-1且必须传入attn_mask矩阵 | 无 |
| attn_mask | tensor_type支持INT8；tensor_shape维度为(2048, 2048)，该矩阵固定为2048*2048的下三角矩阵对角线全0 | 无 | 同上 | 无 |
| win_left | data_type支持INT；不传或者传入-1，代表该值为正无穷；值需要 >= -1 | 无 | 同上 | 无 |
| win_right | 同win_left | 无 | 同上 | 无 |

#### SeqLengths参数组

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
|:---|:---|:---|:---|:---|
| seqused_q | tensor_type支持INT32；tensor_shape为(B,)；仅支持非负整数；seqused_q中的值需小于等于Q_S | 可选参数 | 无 | 无 |
| seqused_kv | tensor_type支持INT32；tensor_shape为(B,)；仅支持非负整数；seqused_kv中的值需小于等于KV_S | 可选参数 | 无 | 无 |
| cu_seqlens_q | tensor_type支持INT32；tensor_shape为(B+1,)；值仅支持非负整数；其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于 Q_T | 可选参数 | 无 | 当layout_q为TND时，必须传入；当layout_q不为TND时，不支持传入 |
| cu_seqlens_kv | tensor_type支持INT32；tensor_shape为(B+1,)；值仅支持非负整数；其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于 KV_T | 可选参数 | 无 | 当layout_kv为TND时，必须传入；当layout_kv不为TND时，不支持传入 |
| max_seqlen_q | data_type支持INT；暂不生效，仅支持-1；默认值为-1 | 可选参数 | 无 | 暂不生效，仅支持传入-1 |
| max_seqlen_kv | 同max_seqlen_q | 可选参数 | 无 | 暂不生效，仅支持传入-1 |

#### Paged Attention参数组

当block_table不为空时，使能Paged Attention

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
|:---|:---|:---|:---|:---|
| block_table | tensor_type仅支持INT32；tensor_shape为(B, max_num_blocks_per_seq)；值只能为正整数 | 可选参数 | 无 | PagedAttention 使能情况下，必须传入 seqused_kv；PagedAttention 使能情况下，block_table必须不为空 |

#### Sinks参数组

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
|:---|:---|:---|:---|:---|
| sinks | 暂不支持 | 可选参数 | 无 | 暂不支持 |

#### SoftmaxLSE参数组

| 参数 | 单参数校验 | 存在性校验 | 一致性校验 | 特性交叉校验 |
|:---|:---|:---|:---|:---|
| return_softmax_lse | data_type仅支持INT；值仅支持0和1，1代表开启softmax_lse，0代表关闭softmax_lse | 可选属性，默认值为0 | 当return_softmax_lse为0时，输出shape为[1]的值为0的Tensor；当return_softmax_lse为1时，输出shape见layout匹配关系表 | 无 |
| softmax_lse | data_type仅支持FLOAT32 | 无 | 同上 | 无 |

## 调用示例<a name="zh-cn_topic_0000002168254826_section14459801435"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    from npu_ops_transformer.ops import npu_flash_attn, npu_flash_attn_metadata

    # 配置参数
    device = torch.device("npu:0")
    dtype = torch.float16
    batch_size = 1
    num_heads_q = 1
    num_heads_kv = 1
    seq_len_q = 1
    seq_len_kv = 8192
    head_dim = 128
    mask_size = 2048
    layout = "BSND"
    mask_mode = 3  # Causal模式

    # 构造Causal掩码矩阵（上三角为1）
    attn_mask = torch.triu(
        torch.ones(mask_size, mask_size, dtype=torch.int8), diagonal=1
    ).to(device)

    # 构造输入数据
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)

    q = torch.randn(batch_size, seq_len_q, num_heads_q, head_dim,
                    dtype=dtype, generator=generator).to(device)
    k = torch.randn(batch_size, seq_len_kv, num_heads_kv, head_dim,
                    dtype=dtype, generator=generator).to(device)
    v = torch.randn(batch_size, seq_len_kv, num_heads_kv, head_dim,
                    dtype=dtype, generator=generator).to(device)

    # 构造metadata
    metadata = npu_flash_attn_metadata(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        batch_size=batch_size,
        max_seqlen_q=seq_len_q,
        max_seqlen_kv=seq_len_kv,
        mask_mode=mask_mode,
        win_left=-1,
        win_right=-1,
        layout_q=layout,
        layout_kv=layout,
        layout_out=layout,
    )

    # 调用算子
    softmax_scale = 1.0 / (head_dim ** 0.5)
    attn_out, softmax_lse = npu_flash_attn(
        q, k, v,
        softmax_scale=softmax_scale,
        mask_mode=mask_mode,
        attn_mask=attn_mask,
        metadata=metadata,
        layout_q=layout,
        layout_kv=layout,
        layout_out=layout,
    )

    torch.npu.synchronize()
    print("attn_out shape:", attn_out.shape)
    ```
