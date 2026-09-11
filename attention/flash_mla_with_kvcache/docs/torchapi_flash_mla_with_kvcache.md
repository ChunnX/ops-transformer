# flash_mla_with_kvcache

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
  O = \mathrm{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
  $$

  其中 $Q$ 由 $q$（含RoPE）经线性变换得到，$K$ 与 $V$ 均来自同一份 $k\_cache$：$K$ 使用 $k\_cache$ 的全部 576 维（nope+rope），$V$ 使用 $k\_cache$ 的前 512 维（nope 部分，即 $\mathit{head\_dim\_v}$）。**MLA 复用单一 KV 缓存，接口不单独传入 $V$**。$\sqrt{d_k}$ 为缩放因子（可通过 `softmax_scale` 显式覆盖）。

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
> query（q）支持TND、BNSD、BSND三种布局；KV缓存（k_cache）仅支持Paged Attention（PA）分页布局（PA_NZ、PA_BBND、PA_BNBD），不支持连续KV；输出（attn_out）默认布局为BSND，TND输入支持NTD输出（转置），其余布局必须与输入一致。其中B（Batch）表示输入样本批量大小、Q_S表示seqlen_q、Q_N表示heads_q、Q_T表示所有query batch序列长度的累加和、D表示头维（nope部分512，整体576含rope64）。

## 函数原型

调用flash_mla_with_kvcache接口之前，请先调用前置接口flash_mla_with_kvcache_metadata，完成flash_mla_with_kvcache负载均衡的计算。

```python
cann_ops_transformer.flash_mla_with_kvcache_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=None,
    layout_q="BSND",
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
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False
) -> (Tensor, Tensor)
```

## 参数说明

### flash_mla_with_kvcache_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cache_seqlens | Tensor | 必选 | 每batch的KV缓存实际长度（seqused_kv语义），batch大小由它的长度推导，不再单独传batch_size | int32 | ND | (B,) |
| num_heads_q | int | 必选 | Query head数 | int32 | - | - |
| num_heads_kv | int | 必选 | Key/Value head数，MLA场景固定为1 | int32 | - | - |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0 | int32 | ND | (B+1,) |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的查询序列长度，截断冗余运算 | int32 | ND | (B,) |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - |
| max_seqlen_kv | int | 可选 | 指定KV缓存序列的长度上限 | int32 | - | - |
| head_dim_qk | int | 可选 | 每个注意力头的qk维度（nope 512 + rope 64，默认576） | int32 | - | - |
| head_dim_v | int | 可选 | value（输出）宽度，默认512 | int32 | - | - |
| mask_mode | int | 可选 | 掩码模式，仅支持NO_MASK与RIGHT_DOWN两种取值（对应值0与3），其余取值将被拒绝 | int32 | - | - |
| layout_q | string | 可选 | 定义输入q张量的布局格式，支持"TND"/"BNSD"/"BSND"，默认为"BSND" | string | - | - |

### flash_mla_with_kvcache

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 | 默认值 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| q | Tensor | 必选 | 公式中的q（含RoPE，最后64维为rope） | bfloat16/float16 | ND | <ul><li>TND -> (Q_T, Q_N, 576)</li><li>BNSD -> (B, Q_N, Q_S, 576)</li><li>BSND -> (B, Q_S, Q_N, 576)</li></ul> | — |
| k_cache | Tensor | 必选 | 共享KV缓存（Key与Value同一份输入，最后64维为rope），仅支持Paged Attention（PA）分页布局 | bfloat16/float16 | ND | <ul><li>PA_NZ -> (num_blocks, KV_N, D//16, block_size, 16)</li><li>PA_BBND -> (num_blocks, block_size, KV_N, D)</li><li>PA_BNBD -> (num_blocks, KV_N, block_size, D)</li></ul> | — |
| block_table | Tensor | 可选 | 用于分块注意力计算中的块索引映射，k_cache为PA分页布局时必须传入 | int32 | ND | (B, max_num_blocks_per_seq) | `None` |
| cache_seqlens | Tensor | 可选 | 每batch的KV缓存实际长度（seqused_kv语义） | int32 | ND | (B,) | `None` |
| cu_seqlens_q | Tensor | 可选 | 累积序列长度，用于处理变长序列，第一个元素必须为0；layout_q为TND时必传 | int32 | ND | (B+1,) | `None` |
| seqused_q | Tensor | 可选 | 指定每batch中实际使用的查询序列长度，截断冗余运算 | int32 | ND | (B,) | `None` |
| attn_mask | Tensor | 可选 | 掩码矩阵（mask_mode为3时必传） | int8 | ND | (2048, 2048) | `None` |
| metadata | Tensor | 可选 | `flash_mla_with_kvcache_metadata`生成的任务切分结果，传入后可优化调度 | int32 | ND | (max_schedule_size,) | `None` |
| head_dim_v | int | 可选 | nope（value）部分宽度，仅支持512 | int32 | - | - | `512` |
| softmax_scale | float | 可选 | 默认1.0；显式传值时使用传入值 | float32 | - | - | `1.0` |
| mask_mode | int | 可选 | 掩码模式，仅支持NO_MASK与RIGHT_DOWN两种取值（对应值0与3），其余取值将被拒绝 | int32 | - | - | `0` |
| max_seqlen_q | int | 可选 | 指定查询q序列的长度上限 | int32 | - | - | `-1` |
| max_seqlen_kv | int | 可选 | 指定KV缓存序列的长度上限 | int32 | - | - | `-1` |
| layout_q | string | 可选 | 定义输入q张量的布局格式，支持"TND"/"BNSD"/"BSND" | string | - | - | `"BSND"` |
| layout_kv | string | 可选 | 定义输入k_cache张量的布局格式，仅支持PA分页布局"PA_NZ"/"PA_BBND"/"PA_BNBD" | string | - | - | `"PA_BBND"` |
| layout_out | string | 可选 | 定义输出张量的布局格式，默认为BSND；TND输入支持NTD输出（转置），其余布局必须与layout_q一致 | string | - | - | `"BSND"` |
| return_softmax_lse | bool | 可选 | 是否需要获取softmax的LSE结果 | bool | - | - | `False` |

默认值表示 Python 接口省略参数时的取值；`None` 不代表该参数在所有场景下都可省略，必传条件仍按下文约束执行。`softmax_scale` 默认使用 `1.0`，不会自动按头维度计算；如需缩放，请显式传值。当 `layout_q` 显式设为 TND 或 BNSD 时，也需显式设置匹配的 `layout_out`（TND 还支持 NTD）。

## 返回值说明

### flash_mla_with_kvcache_metadata

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | flash_mla_with_kvcache的任务切分数据 | int32 | ND | shape根据batch_size动态计算，容量上界为((aic_core_num+aiv_core_num)×batch_size×1+1)×16，按4096个INT32元素对齐（核数取当前设备实际值） |

### flash_mla_with_kvcache

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| attn_out | Tensor | 必选 | flash_mla_with_kvcache的计算输出，默认布局为BSND；TND输入支持NTD输出（转置，此时shape为(N, T, D)） | bfloat16/float16 | ND | <ul><li>TND -> (Q_T, Q_N, D)</li><li>NTD -> (Q_N, Q_T, D)</li><li>BNSD -> (B, Q_N, Q_S, D)</li><li>BSND -> (B, Q_S, Q_N, D)</li></ul>D=head_dim_v |
| softmax_lse | Tensor | 可选 | softmax的LSE结果 | float32 | ND | <ul><li>TND -> (Q_N, Q_T)</li><li>BNSD/BSND -> (B, Q_N, Q_S)</li></ul> |

**说明**

- attn_out：输出D维度与`head_dim_v`保持一致（512）；维度排布默认为BSND，TND输入支持NTD输出（转置）。
- softmax_lse：return_softmax_lse为True时，layout_q为TND输出shape为(Q_N, Q_T)，layout_q为BNSD/BSND时输出shape为(B, Q_N, Q_S)；return_softmax_lse为False时，则输出shape为[1]的值为0的Tensor。

## 约束说明

- 参数cu_seqlens_q、cache_seqlens、seqused_q、block_table及attn_mask属于设备侧Tensor。主算子的host tiling仅校验shape、dtype和属性等信息，不读取设备Tensor的元素值，也不校验这些值之间的大小关系。调用方必须保证下述值约束成立；非法值可能导致错误调度、精度异常或非法内存访问。
- metadata算子的AICPU会读取序列长度值用于调度，并进行部分值校验，但不接收q和k_cache，不能确认长度是否超出实际存储容量。metadata生成成功不代表主算子的所有输入值约束均已通过校验。
- flash_mla_with_kvcache_metadata和flash_mla_with_kvcache的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由客户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
- metadata由`flash_mla_with_kvcache_metadata`生成后原样传入`flash_mla_with_kvcache`；用户自建过小的metadata缓冲区将被容量校验拒绝。
- layout_q支持TND、BNSD、BSND三种布局；layout_out默认为BSND；TND输入支持NTD输出（转置），其余布局必须与layout_q一致。
- **KV缓存仅支持Paged Attention（PA）分页布局**：layout_kv仅支持"NZ"（PA_NZ）、"PA_BBND"、"PA_BNBD"，且必须传入`block_table`；不支持连续KV布局，其余布局一律由算子校验拒绝。
- **q与k_cache的最后维（D）必须为576**（= head_dim_v 512 + rope 64），不允许512宽输入。
- **head_dim_v必须为512**（nope/value部分宽度，即 k_cache 最后维576减去rope宽度64）；传入其它值将被拒绝。
- `num_heads_kv`必须为1（单latent KV头）。
- 掩码仅支持NO_MASK（0）与RIGHT_DOWN（3，右对齐因果）两种模式，其余mask_mode取值将被拒绝；具体约束与flash_attn一致（mask_mode为0时不支持传入attn_mask，mask_mode为3时必须传入attn_mask矩阵）。
- 空Tensor（必有输入的任一轴shape为0）将被全部拦截报错。

### 序列长度值约束（由调用方保证）

下表中的`b`表示batch下标，`Q_S`表示BSND/BNSD输入q的S维大小，`Q_T`表示TND输入q的T维大小。数组元素数属于shape信息，主算子host tiling可获取；数组中记录的实际序列长度属于设备数据，host tiling不读取。

| 参数/场景 | 值约束 |
| :--- | :--- |
| TND的cu_seqlens_q | 必传，shape为`(B+1,)`；首元素为0，单调不减，最后一个元素等于`Q_T`。第b个batch的存储长度为`cu_seqlens_q[b+1] - cu_seqlens_q[b]`。 |
| TND的seqused_q | 如传入，shape为`(B,)`，且`0 <= seqused_q[b] <= cu_seqlens_q[b+1] - cu_seqlens_q[b]`；不传时使用上述存储长度。当前接口不保证拦截超过该上界的值。 |
| BSND/BNSD的seqused_q | 如传入，shape为`(B,)`，且`0 <= seqused_q[b] <= Q_S`；不传时主算子使用`Q_S`。这两种布局不传cu_seqlens_q。当前接口不保证拦截超过该上界的值。 |
| BSND/BNSD的metadata长度来源 | 不传seqused_q时，生成metadata必须显式传入与主算子`Q_S`一致的max_seqlen_q，不能保留未指定值。 |
| cache_seqlens与block_table | `0 <= cache_seqlens[b] <= block_table.shape[1] * block_size`；实际使用的页表项必须满足`0 <= block_table[b, j] < num_blocks`，其中`j < ceil(cache_seqlens[b] / block_size)`。 |

例如，TND输入`cu_seqlens_q=[0, 4, 8]`表示两个batch各存储4个token，`seqused_q=[6, 4]`是不合法的：第一个batch要求使用的6个token超过其存储范围，可能读到下一batch的数据。调用方应在构造输入时保证长度合法，不应依赖主算子host tiling读取设备数据后拦截。

生成metadata和调用主算子时，必须使用一致的序列长度、布局、头数和mask模式；修改影响调度的输入后，应重新生成metadata。

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
|                      |        metadata        |      INPUT(OPTIONAL)      | Tensor   |
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
|     Q_S     |        输入q tensor的序列长度，对应q shape中的S（layout_q为BNSD/BSND时）        |
|     Q_T     |          输入q tensor所有batch序列长度的累加和（layout_q为TND时）          |
|     D     |          输出attn_out隐藏层最小的单元尺寸headdim（=head_dim_v=512）         |

### 参数组约束

#### 公共参数组

- 入参为空的场景处理：
  - 空Tensor指必选输入和输出的shape size为0,即有任意轴为0。
  - 触发空tensor的用例将全部拦截报错。

- q、k_cache、attn_out校验:

    <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 147px">
    <col style="width: 232px">
    <col style="width: 232px">
    <col style="width: 293px">
    <col style="width: 185px">
    </colgroup>
    <thead>
    <tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
    </tr>
    </thead>
    <tbody>
    <tr>
        <td>q</td>
        <td>
            <ul>
                <li>tensor_type支持bfloat16和float16</li>
                <li>TND -> (Q_T, Q_N, 576)</li>
                <li>BNSD -> (B, N_Q, S_Q, 576)</li>
                <li>BSND -> (B, S_Q, N_Q, 576)</li>
                <li>最后维（D）必须为576（head_dim_v 512 + rope 64），不允许512宽输入</li>
            </ul>
        </td>
        <td rowspan="3">
            必须存在
        </td>
        <td rowspan="3">
            <ul>
                <li>q、k_cache、attn_out的数据类型需相同</li>
                <li>q与k_cache的最后维（D）必须相等，且必须为576</li>
                <li>layout_out默认为BSND；TND输入支持NTD输出（转置），其余布局必须与layout_q一致</li>
                <li>Layout校验规则见layout匹配关系表</li>
            </ul>
        </td>
        <td rowspan="3">
            轴校验：
            <ul>
                <li>65536 > B > 0</li>
                <li>Q_T > 0</li>
                <li>Q_N > 0</li>
                <li>Q_S > 0</li>
                <li>num_heads_kv必须为1（单latent KV头）</li>
                <li>q与k_cache的D必须为576</li>
                <li>head_dim_v必须为512</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k_cache</td>
        <td>
            <ul>
                <li>tensor_type支持bfloat16和float16</li>
                <li>PA_NZ -> (num_blocks, KV_N, D//16, block_size, 16)，D=576</li>
                <li>PA_BBND -> (num_blocks, block_size, KV_N, D)，D=576</li>
                <li>PA_BNBD -> (num_blocks, KV_N, block_size, D)，D=576</li>
                <li>仅支持Paged Attention（PA）分页缓存布局，不支持连续KV布局</li>
                <li>KV_N恒为1（单latent KV头）</li>
                <li>1024 >= block_size >= 16，block_size % 16 == 0</li>
                <li>非连续Tensor约束：
                    <ul>
                        <li>PA_BBND：仅dim0支持非连续</li>
                        <li>PA_BNBD/PA_NZ：dim0或dim1支持非连续</li>
                    </ul>
                </li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>attn_out</td>
        <td>
            <ul>
                <li>data_type支持bfloat16和float16</li>
                <li>shape维度支持3、4</li>
                <li>输出D与head_dim_v保持一致（512）</li>
            </ul>
        </td>
    </tr>
    </tbody>
    </table>

- layout匹配关系表：

    <table style="undefined;table-layout: fixed; width:1625px"><colgroup>
    <col style="width: 247px">
    <col style="width: 232px">
    <col style="width: 232px">
    <col style="width: 293px">
    </colgroup>
    <thead>
    <tr>
        <th>layout_q</th>
        <th>layout_kv</th>
        <th>layout_out</th>
        <th>layout_softmax_lse</th>
    </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="2">TND</td>
            <td rowspan="4">
            <li>PA_NZ（"NZ"）</li>
            <li>PA_BBND</li>
            <li>PA_BNBD</li>
            </td>
            <td>TND</td>
            <td>(Q_N, Q_T)</td>
        </tr>
        <tr>
            <td>NTD（转置）</td>
            <td>(Q_N, Q_T)</td>
        </tr>
        <tr>
            <td>BNSD</td>
            <td>BNSD</td>
            <td>(B, Q_N, Q_S)</td>
        </tr>
        <tr>
            <td>BSND</td>
            <td>BSND</td>
            <td>(B, Q_N, Q_S)</td>
        </tr>
    </tbody>
    </table>

- metadata校验:

    <table style="undefined;table-layout: fixed; width:1625px">
        <colgroup>
            <col style="width: 147px">
            <col style="width: 232px">
            <col style="width: 232px">
            <col style="width: 293px">
            <col style="width: 185px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>单参数校验</th>
                <th>存在性校验</th>
                <th>一致性校验</th>
                <th>特性交叉校验</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>metadata</td>
                <td>
                    <ul>
                        <li>tensor_type仅支持int32</li>
                        <li>shape由flash_mla_with_kvcache_metadata动态计算</li>
                        <li>容量上界为((aic_core_num+aiv_core_num)×batch_size×1+1)×16，按4096个INT32元素对齐（核数取当前设备实际值）</li>
                    </ul>
                </td>
                <td>可选参数，不传入时不启用负载均衡调度优化</td>
                <td>无</td>
                <td>传入时需与flash_mla_with_kvcache_metadata生成的结果一致，用户自建过小的metadata缓冲区将被容量校验拒绝</td>
            </tr>
        </tbody>
    </table>

#### Mask参数组

mask_mode参数解释，具体约束与flash_attn一致
<ul>
    <li>mask_mode=0，NO_MASK模式（全计算，默认值）</li>
    <li>mask_mode=3，RIGHT_DOWN模式（右对齐因果）</li>
</ul>

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>mask_mode</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>仅支持0（NO_MASK）与3（RIGHT_DOWN）两种取值，默认值为0，其余取值将被拒绝</li>
                </ul>
            </td>
            <td>
                可选输入，如果不传该参数，默认值为0
            </td>
            <td rowspan="2">
                <ul>
                    <li>当mask_mode为0时，不支持传入attn_mask</li>
                    <li>当mask_mode为3时，必须传入attn_mask矩阵</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>无</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>attn_mask</td>
            <td>
                <ul>
                    <li>tensor_type支持int8</li>
                    <li>tensor_shape维度为(2048, 2048)，该矩阵固定为2048*2048，左下三角包括对角线为0（保留），其余位置为1（屏蔽）</li>
                </ul>
            </td>
            <td>
                当mask_mode为3时必传
            </td>
        </tr>
    </tbody>
</table>

#### SeqLengths参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>seqused_q</td>
            <td>
                <ul>
                    <li>tensor_type支持int32</li>
                    <li>tensor_shape为(B,)</li>
                    <li>仅支持非负整数</li>
                    <li>seqused_q中的值需小于等于max_seqlen_q</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td rowspan="4">无</td>
            <td rowspan="4">无</td>
        </tr>
        <tr>
            <td>cache_seqlens</td>
            <td>
                <ul>
                    <li>tensor_type支持int32</li>
                    <li>tensor_shape为(B,)</li>
                    <li>仅支持非负整数</li>
                    <li>表示每batch的KV缓存实际长度（seqused_kv语义）</li>
                    <li>cache_seqlens中的值需小于等于max_seqlen_kv</li>
                </ul>
            </td>
            <td>可选参数</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td>
                <ul>
                    <li>tensor_type支持int32</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列</li>
                    <li>第一个元素为0且最后一个元素等于Q_T</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时，cu_seqlens_q必须传入</li>
                    <li>当layout_q不为TND时，cu_seqlens_q不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td rowspan="2">
                <ul>
                    <li>data_type支持INT</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>可选参数</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>值必须大于等于-1；传入时必须等于实际的最大序列长度，否则行为未定义</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_kv</td>
        </tr>
    </tbody>
</table>

#### Paged Attention参数组

KV缓存仅支持Paged Attention（PA）分页布局，block_table不为空时开启Paged Attention。
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>block_table</td>
            <td>
                <ul>
                    <li>tensor_type仅支持int32</li>
                    <li>tensor_shape为(B, max_num_blocks_per_seq)</li>
                    <li>值只能为正整数</li>
                    <li>1024 >= block_size >= 16，block_size % 16 == 0</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>
                <ul>
                    <li>k_cache仅支持Paged Attention（PA）分页缓存布局（PA_NZ/PA_BBND/PA_BNBD），不支持连续KV布局</li>
                    <li>k_cache为PA分页布局时，block_table必须不为空</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

#### SoftmaxLSE参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>return_softmax_lse</td>
            <td>
                <ul>
                    <li>data_type仅支持BOOL</li>
                    <li>值仅支持True和False，True代表开启softmax_lse，False代表关闭softmax_lse</li>
                </ul>
            </td>
            <td>可选属性，默认值为False</td>
            <td rowspan="2">
                <ul>
                    <li>当return_softmax_lse为False时，输出shape为[1]的值为0的Tensor</li>
                    <li>当return_softmax_lse为True时，输出shape见layout匹配关系表</li>
                </ul>
            </td>
            <td>无</td>
        </tr>
        <tr>
            <td>softmax_lse</td>
            <td>
                <ul>
                    <li>data_type仅支持float32</li>
                </ul>
            </td>
            <td>无</td>
            <td>无</td>
        </tr>
    </tbody>
</table>

## 调用示例

- flash_mla_with_kvcache_metadata + flash_mla_with_kvcache联合调用示例（TND / PA_NZ / TND，RIGHT_DOWN掩码）

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
    # mask_mode=3（RIGHT_DOWN/右对齐因果）：1=屏蔽，0=保留
    # 固定为2048*2048，左下三角包括对角线为0（保留），其余位置为1（屏蔽）
    attn_mask = torch.triu(
        torch.ones(2048, 2048, dtype=torch.int8, device="npu"), diagonal=1
    )

    metadata = cann_ops_transformer.ops.flash_mla_with_kvcache_metadata(
        cache_seqlens=cache_seqlens,
        num_heads_q=Q_N,
        num_heads_kv=KV_N,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=S,
        max_seqlen_kv=S,
        head_dim_qk=HEAD_DIM_NOPE,
        mask_mode=3,  # RIGHT_DOWN/因果
        layout_q="TND",
    )

    attn_out, softmax_lse = cann_ops_transformer.ops.flash_mla_with_kvcache(
        q,
        k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        attn_mask=attn_mask,
        metadata=metadata,
        head_dim_v=HEAD_DIM_NOPE,
        softmax_scale=1.0 / (HEAD_DIM_NOPE ** 0.5),
        mask_mode=3,  # RIGHT_DOWN/因果
        max_seqlen_q=S,
        max_seqlen_kv=S,
        layout_q="TND",
        layout_kv="NZ",
        layout_out="TND",
        return_softmax_lse=False,
    )
    torch_npu.npu.synchronize()
    assert attn_out.shape == (Q_T, Q_N, HEAD_DIM_NOPE)
    assert attn_out.dtype == q.dtype
    assert torch.isfinite(attn_out.float()).all().item()
    ```

> [!NOTE]
>
> 图模式调用（暂不支持）：`flash_mla_with_kvcache`的图模式GE转换暂未支持，图模式下调用会报"GE not supported"。
