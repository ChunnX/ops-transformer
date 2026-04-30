# torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明<a name="zh-cn_topic_0000002236535552_section1023311522369"></a>

- 接口功能：融合了MLA（Multi-head Latent Attention）结构中RMSNorm归一化计算与RoPE（Rotary Position Embedding）位置编码以及更新KVCache的ScatterUpdate操作。

- 计算公式：
  首先对尾轴进行拆分得到kv_a部分和kv_pe部分，分别进行rms_norm和rope计算，然后进行量化(可选)并更新kvcache

  $$
  kv\_a=kv[...,:rms\_size]
  $$

  $$
  kv\_pe=kv[...,rms\_size:]
  $$

  $$
  y\_out=RmsNorm(kv\_a)
  $$

  $$
  k\_embed\_out = Rope(kv\_pe)
  $$

  $$
  k\_cache[index] = Quant(x = y\_out, scale = k\_scale, offset = k\_offset)
  $$

  $$
  ckv\_cache[index] = Quant(x = k\_embed\_out, scale = k\_scale, offset = k\_offset)
  $$

  (1) RmsNorm逻辑如下:

  $$
  \operatorname{RmsNorm}(x_i)=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  (2) Rope逻辑如下：

  当rotary_mode=half时：

  $$
  x1, x2 = torch.chunk(x, 2, dim=-1)
  $$

  当rotary_mode=interleave_half时：

  $$
  x1=x[...,::2],x2=x[...,1::2]
  $$

  然后进行拼接与旋转计算：

  $$
  x\_rotate = torch.cat((-x2, x1), dim=-1)
  $$

  $$
  x\_rope = x * cos + x\_rotate * sin
  $$

  (3) Quant逻辑如下：

  当quant_mode=static时：

  $$
  x = x * scale,\ if\ scale\ !=\ None
  $$

  $$
  x = x + offset,\ if\ offset\ !=\ None
  $$

  $$
  y = round(x).clamp(-128,127)
  $$

  当quant_mode=pertile128时：（每128个元素为一组进行动态量化）

  $$
  tile\_size = 128 \\
  x\_rng[i]=[(i-1)*tile\_size: i * tile\_size - 1]
  $$

  $$
  scaleOut[i]=row\_max(abs(x[x\_rng[i]]))/127
  $$

  $$
  yOut=round(\frac{x[x\_rng[i]]}{scaleOut[i]})
  $$

## 函数原型<a name="zh-cn_topic_0000002236535552_section123412524369"></a>

torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache(kv, gamma, cos, sin, index, k_cache, ckv_cache, k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None, epsilon=1e-5, cache_mode='Norm', rotary_mode='interleave-half', quant_mode='static', is_output_kv=False) -> (Tensor, Tensor, Tensor, Tensor)


## 参数说明<a name="zh-cn_topic_0000002236535552_section1723416525369"></a>

> - Tensor中shape使用的变量说明：
>   - batch\_size：batch的大小。
>   - seq\_len：sequence的长度。
>   - hidden\_size：表示MLA输入的向量长度，取值仅支持576。
>   - rms\_size：表示RMSNorm分支的向量长度，取值仅支持512。
>   - rope\_size：表示RoPE分支的向量长度，取值仅支持64。
>   - cache\_length：Norm模式下有效，表示KVCache支持的最大长度。
>   - block\_num：PagedAttention模式下有效，表示Block的个数。
>   - block\_size：PagedAttention模式下有效，表示Block的大小。

- **kv** (`Tensor`)：必选参数，表示输入的特征张量。数据类型支持`bfloat16`、`float16`，数据格式为$BNSD$，要求为4维张量，形状为\[batch\_size, 1, seq\_len, hidden\_size\]，其中hidden\_size=rms\_size\(RMS\)+rope\_size\(RoPE\)。
- **gamma** (`Tensor`)：必选参数，表示RMS归一化的缩放参数。数据类型支持`bfloat16`、`float16`，数据格式为$ND$，要求为1维张量，形状为\[rms\_size\]。
- **cos** (`Tensor`)：必选参数，表示RoPE旋转位置编码的余弦分量。数据类型支持`bfloat16`、`float16`，数据格式为$ND$，要求为4维张量，形状为\[batch\_size, 1, seq\_len, rope\_size\]。
- **sin** (`Tensor`)：必选参数，表示RoPE旋转位置编码的正弦分量。数据类型支持`bfloat16`、`float16`，数据格式为$ND$，要求为4维张量，形状为\[batch\_size, 1, seq\_len, rope\_size\]。
- **index** (`Tensor`)：必选参数，表示缓存索引张量，用于定位`k_cache`和`ckv_cache`的写入位置。数据类型支持`int64`，数据格式为$ND$。shape取决于`cache_mode`。
- **k\_cache** (`Tensor`)：必选参数，用于存储量化/非量化的键向量。数据类型支持`bfloat16`、`float16`、`int8`，数据格式为$ND$。shape取决于`cache_mode`。
- **ckv\_cache** (`Tensor`)：必选参数，用于存储量化/非量化的压缩后的kv向量。数据类型支持`bfloat16`、`float16`、`int8`，数据格式为$ND$。shape取决于`cache_mode`。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **k\_rope\_scale** (`Tensor`)：可选参数，默认值None，表示k旋转位置编码的量化缩放因子。数据类型支持`float32`，数据格式为$ND$，要求为1维张量，形状为\[rope\_size\]。量化模式下必填。
- **c\_kv\_scale** (`Tensor`)：可选参数，默认值None，表示压缩后kv的量化缩放因子。数据类型支持`float32`，数据格式为$ND$，要求为1维张量，形状为\[rms\_size\]。量化模式下必填。
- **k\_rope\_offset** (`Tensor`)：可选参数，默认值None，表示k旋转位置编码量化偏移量。数据类型支持`float32`，数据格式为$ND$，要求为1维张量，形状为\[rope\_size\]。量化模式下必填。
- **c\_kv\_offset** (`Tensor`)：可选参数，默认值None，表示压缩后kv的量化偏移量。数据类型支持`float32`，数据格式为$ND$，要求为1维张量，形状为\[rms\_size\]。量化模式下必填。
- **epsilon** (`float`)：可选参数，默认值1e-5，表示RMS归一化中的极小值，防止除以零。
- **cache\_mode** (`str`)：可选参数，默认值'Norm'，表示缓存模式，支持的模式如下：
  
  <a></a>
  
  <table><thead align="left"><tr id="zh-cn_topic_0000002236535552_row12998195743918"><th class="cellrowborder" valign="top" width="10.34%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002236535552_p1299819576394"><a></a><a></a>枚举值</p>
  </th>
  <th class="cellrowborder" valign="top" width="19.84%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002236535552_p93491748114015"><a></a><a></a>模式名</p>
  </th>
  <th class="cellrowborder" valign="top" width="69.82000000000001%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002236535552_p099810576395"><a></a><a></a>说明</p>
  </th>
  </tr>
  </thead>
  <tbody><tr id="zh-cn_topic_0000002236535552_row499835715398"><td class="cellrowborder" valign="top" width="10.34%"><p id="zh-cn_topic_0000002236535552_p09984572391"><a></a><a></a>Norm</p>
  </td>
  <td class="cellrowborder" valign="top" width="19.84%"><p id="zh-cn_topic_0000002236535552_p23491348164018"><a></a><a></a>KV-Cache更新模式</p>
  </td>
  <td class="cellrowborder" valign="top" width="69.82000000000001%"><p id="zh-cn_topic_0000002236535552_p1499812579399"><a></a><a></a>k_cache形状为[batch_size, 1, cache_length, rope_size]，ckv_cache形状为[batch_size, 1, cache_length, rms_size]。index形状为[batch_size, seq_len]，index里的值表示每个Batch下的偏移。</p>
  </td>
  </tr>
  <tr id="zh-cn_topic_0000002236535552_row1699810579398"><td class="cellrowborder" valign="top" width="10.34%"><p id="zh-cn_topic_0000002236535552_p1699810576392"><a></a><a></a>PA/PA_BNSD</p>
  </td>
  <td class="cellrowborder" valign="top" width="19.84%"><p id="zh-cn_topic_0000002236535552_p1934914484406"><a></a><a></a>PagedAttention模式</p>
  </td>
  <td class="cellrowborder" valign="top" width="69.82000000000001%"><p id="zh-cn_topic_0000002236535552_p19457162584119"><a></a><a></a>k_cache形状为[block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size*seq_len]，index里的值表示每个token的偏移。</p>
  </td>
  </tr>
  <tr id="zh-cn_topic_0000002236535552_row179981757193914"><td class="cellrowborder" valign="top" width="10.34%"><p id="zh-cn_topic_0000002236535552_p99981257113913"><a></a><a></a>PA_NZ</p>
  </td>
  <td class="cellrowborder" valign="top" width="19.84%"><p id="zh-cn_topic_0000002236535552_p6349174814019"><a></a><a></a>Cache数据格式为FRACTAL_NZ的PagedAttention模式</p>
  </td>
  <td class="cellrowborder" valign="top" width="69.82000000000001%"><p id="zh-cn_topic_0000002236535552_p15265174112368"><a></a><a></a>k_cache形状为 [block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size * seq_len]，index里的值表示每个token的偏移。</p>
  <p id="zh-cn_topic_0000002236535552_p10572135917168"><a></a><a></a>不同量化模式下数据排布不同：</p>
  <a></a><a></a><ul id="zh-cn_topic_0000002236535552_ul183421286166"><li>非量化模式下：k_cache数据排布为[block_num, rope_size//16, block_size, 1, 16]，ckv_cache数据排布为[block_num, rms_size//16, block_size, 1, 16]</li><li>量化模式下：k_cache数据排布为[block_num, rope_size//32, block_size, 1, 32]，ckv_cache数据排布为[block_num, rms_size//32, block_size, 1, 32]</li></ul>
  </td>
  </tr>
  <tr id="zh-cn_topic_0000002236535552_row428921319407"><td class="cellrowborder" valign="top" width="10.34%"><p id="zh-cn_topic_0000002236535552_p18290111311409"><a></a><a></a>PA_BLK_BNSD</p>
  </td>
  <td class="cellrowborder" valign="top" width="19.84%"><p id="zh-cn_topic_0000002236535552_p12349144844010"><a></a><a></a>特殊的PagedAttention模式</p>
  </td>
  <td class="cellrowborder" valign="top" width="69.82000000000001%"><p id="zh-cn_topic_0000002236535552_p329011315406"><a></a><a></a>k_cache形状为[block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size*Ceil(seq_len/block_size)]，index里的值表示每个block的起始偏移，不再和token一一对应。</p>
  </td>
  </tr>
  <tr id="zh-cn_topic_0000002236535552_row182905133408"><td class="cellrowborder" valign="top" width="10.34%"><p id="zh-cn_topic_0000002236535552_p13290313134017"><a></a><a></a>PA_BLK_NZ</p>
  </td>
  <td class="cellrowborder" valign="top" width="19.84%"><p id="zh-cn_topic_0000002236535552_p173491486403"><a></a><a></a>Cache数据格式为FRACTAL_NZ的特殊的PagedAttention模式</p>
  </td>
  <td class="cellrowborder" valign="top" width="69.82000000000001%"><p id="zh-cn_topic_0000002236535552_p637504423711"><a></a><a></a>k_cache形状为 [block_num, block_size, 1, rope_size]，ckv_cache形状为[block_num, block_size, 1, rms_size]。index形状为[batch_size * Ceil(seq_len / block_size)]，index里的值表示每个block的起始偏移，不再和token一一对应。</p>
  <p id="zh-cn_topic_0000002236535552_p916702812303"><a></a><a></a>不同量化模式下数据排布不同：</p>
  <a></a><a></a><ul id="zh-cn_topic_0000002236535552_ul20871102344210"><li>非量化模式下：k_cache数据排布为[block_num, rope_size//16, block_size, 1, 16]，ckv_cache数据排布为[block_num, rms_size//16, block_size, 1, 16]</li><li>量化模式下：k_cache数据排布为[block_num, rope_size//32, block_size, 1, 32]，ckv_cache数据排布为[block_num, rms_size//32, block_size, 1, 32]</li></ul>
  </td>
  </tr>
  </tbody>
  </table>
- **rotary\_mode** (`str`)：可选参数，默认值'interleave_half'，表示rope旋转的模式，支持的模式有'interleave_half'，'half_half'
- **quant\_mode** (`str`)：可选参数，默认值'static'，表示rope旋转的模式，支持的模式有'none', 'static'，'pertile128'。设置为'none'表示显式强调不使用任何量化模式，此时传入合法的量化缩放因子和量化偏移量，也都会被无视。
- **is\_output\_kv** (`bool`)：可选参数，表示是否输出处理后的`k_embed_out`和`y_out`（未量化的原始值），默认值False不输出，仅`cache_mode`在\(PA/PA\_BNSD/PA\_NZ/PA\_BLK\_BNSD/PA\_BLK\_NZ\)模式下有效。

## 返回值说明<a name="zh-cn_topic_0000002236535552_section3234185215368"></a>

- **k\_cache** (`Tensor`)：和输入`k_cache`的数据类型、维度、数据格式完全一致（本质in-place更新）。
- **ckv\_cache** (`Tensor`)：和输入`ckv_cache`的数据类型、维度、数据格式完全一致（本质in-place更新）。
- **k\_embed\_out** (`Tensor`)：仅当`is_output_kv`为True时，表示RoPE处理后的值。要求为4维张量，形状为\[batch\_size, 1, seq\_len, 64\]，数据类型和格式同输入`kv`一致。
- **y\_out** (`Tensor`)：仅当`is_output_kv`为True时，表示RMSNorm处理后的值。要求为4维张量，形状为\[batch\_size, 1, seq\_len, 512\]，数据类型和格式同输入`kv`一致。

## 约束说明<a name="zh-cn_topic_0000002236535552_section1523425283618"></a>

-   该接口支持推理场景下使用。
-   该接口支持图模式。
-   量化模式：当`k_rope_scale`和`c_kv_scale`非空时，`k_cache`和`ckv_cache`的dtype为`int8`，缓存形状的最后一个维度需要为32（Cache数据格式为FRACTAL\_NZ模式），`k_rope_scale`和`c_kv_scale`必须同时非空，`k_rope_offset`和`c_kv_offset`必须同时为None或非空。
-   非量化模式：当`k_rope_scale`和`c_kv_scale`为空时，`k_cache`和`ckv_cache`的dtype为`bfloat16`或`float16`。
-   索引映射：所有`cache_mode`缓存模式下，index的值不可以重复，如果传入的index值存在重复，算子的行为是未定义的且不可预知的。
    -   Norm：index的值表示每个Batch下的偏移。
    -   PA/PA\_BNSD/PA\_NZ：index的值表示全局的偏移。
    -   PA\_BLK\_BNSD/PA\_BLK\_NZ：index的值表示每个页的全局偏移；该场景假设cache更新是连续的，不支持非连续更新的cache。

-   Shape关联规则：不同的`cache_mode`缓存模式有不同的Shape规则。
    -   Norm：k\_cache形状为\[batch\_size, 1, cache\_length, rope\_size\]，ckv\_cache形状为\[batch\_size, 1, cache\_length, rms\_size\]，index形状为\[batch\_size, seq\_len\], cache\_length\>=seq\_len。
    -   非Norm模式\(PagedAttention相关模式\)：要求block\_num\>=Ceil\(seq\_len/block\_size\)\*batch\_size。

## 调用示例<a name="zh-cn_topic_0000002236535552_section3235105212365"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    import omni_custom_ops

    batch_size = 8
    seq_len = 1
    page_num = 8
    page_size = 128
    input_dtype = torch.bfloat16

    # 构造输入
    kv = torch.randn(batch_size, 1, seq_len, 576, dtype=input_dtype).npu()
    gamma = torch.randn(512, dtype=input_dtype).npu()
    cos = torch.randn(batch_size, 1, seq_len, 64, dtype=input_dtype).npu()
    sin = torch.randn(batch_size, 1, seq_len, 64, dtype=input_dtype).npu()
    index = torch.arange(batch_size * seq_len, dtype=torch.int64).npu()

    # 构造cache
    k_cache = torch.zeros(page_num, page_size, 1, 64, dtype=input_dtype).npu()
    ckv_cache = torch.zeros(page_num, page_size, 1, 512, dtype=input_dtype).npu()

    # 调用算子
    k_rope_out, c_kv_out = torch.ops.custom.npu_ai_infra_kv_rmsnorm_rope_cache_v2(
        kv, gamma, cos, sin, index, k_cache, ckv_cache,
        k_rope_scale=None,
        c_kv_scale=None,
        k_rope_offset=None,
        c_kv_offset=None,
        v=None,
        epsilon=1e-5,
        cache_mode="PA_BNSD",
        rotary_mode="half",
        quant_mode="none",
        is_output_kv=True,
    )
    ```