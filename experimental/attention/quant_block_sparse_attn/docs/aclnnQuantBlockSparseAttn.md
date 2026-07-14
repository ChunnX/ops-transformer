# aclnnQuantBlockSparseAttn

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

QuantBlockSparseAttn（QBSA）用于 FP8 量化场景下的分块稀疏注意力计算。算子根据 `sparse_indices` 和 `sparse_seq_len` 指定的稀疏块索引，只对每个 Query block 选中的 KV block 执行注意力计算，并支持 PagedAttention 形式的 KV Cache 存储。

该算子面向大序列推理或预填充场景，通过块级稀疏选择降低 QK、PV 两次矩阵乘的计算量；同时结合 `q_descale`、`k_descale`、`v_descale` 与 `p_scale` 完成 FP8 量化数据的反量化与 softmax 后再量化计算。

计算语义如下：

$$
P = \text{softmax}((QK^T) \times q\_descale \times k\_descale \times softmax\_scale, mask)
$$

$$
O = (quant(P / p\_scale) V) \times p\_scale \times v\_descale
$$

其中 `K`、`V` 由 `block_table` 和 `sparse_indices` 从 PageAttention KV Cache 中按块寻址获得。`mask_mode=0` 表示不加 mask，`mask_mode=3` 表示 causal mask。

## 接口说明

该算子提供底层 ACLNN 接口，并通过 PyTorch 扩展注册为 `torch.ops.custom.npu_quant_block_sparse_attn`。PyTorch 入口会创建输出 Tensor，并调用底层 `aclnnQuantBlockSparseAttn`。

### PyTorch 接口原型

```python
torch.ops.custom.npu_quant_block_sparse_attn(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_descale: Tensor,
    k_descale: Tensor,
    v_descale: Tensor,
    p_scale: Tensor,
    sparse_indices: Tensor,
    sparse_seq_len: Tensor,
    atten_mask: Tensor,
    softmax_scale: float,
    sparse_q_block_size: int,
    sparse_kv_block_size: int,
    *,
    cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_kv: Optional[Tensor] = None,
    seqused_q: Optional[Tensor] = None,
    seqused_kv: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    metadata: Optional[Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_kv: int = 0,
    pa_block_stride: int = 0,
    layout_kv: str = "PA_BNSD",
    layout_q: str = "TND",
    layout_sparse_indices: str = "B_N_Qb_Kb",
    layout_out: str = "TND",
    quant_mode: int = 1,
    mask_mode: int = 3,
    return_softmax_lse: bool = False,
) -> Tuple[Tensor, Tensor]
```


## 参数说明

维度符号说明：

- `B`：Batch size。
- `S1`：单个 batch 的 Query 最大序列长度。
- `S2`：单个 batch 的 KV 最大序列长度。
- `T1`：所有 batch 的 Query 有效 token 数之和。
- `N1`：Query head 数。
- `N2`：KV head 数。
- `G`：GQA 分组数，`G = N1 / N2`。
- `D`：Q/K head dim，当前实现固定为 128。
- `D_v`：V head dim，当前实现固定为 128。
- `max_Qb`：Query block 最大数量，`max_Qb = ceil(S1 / sparse_q_block_size)`。
- `max_Kb`：KV block 最大数量，`max_Kb = ceil(S2 / sparse_kv_block_size)`。
- `block_num`：PageAttention KV Cache 物理 block 总数。

<table style="table-layout: auto; width: 100%;">
  <colgroup>
    <col style="white-space: nowrap;">
    <col style="white-space: nowrap; width: 96px;">
    <col>
    <col style="white-space: nowrap;">
    <col style="white-space: nowrap; width: 80px;">
    <col>
    <col>
  </colgroup>
  <thead>
    <tr>
      <th style="white-space: nowrap;">参数名</th>
      <th style="white-space: nowrap;">输入/输出</th>
      <th>描述</th>
      <th style="white-space: nowrap;">数据类型</th>
      <th style="white-space: nowrap;">数据格式</th>
      <th>维度(shape)</th>
      <th>说明</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="white-space: nowrap;"><code>query</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>Query 输入。</td>
      <td style="white-space: nowrap;"><code>FLOAT8_E4M3FN</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>layout_q="TND"</code> 时为 <code>(T1,N1,D)</code>；<code>layout_q="NTD"</code> 时为 <code>(N1,T1,D)</code></td>
      <td>不支持空 Tensor。PyTorch 接入层会将该输入转为连续 Tensor。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>key</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>PageAttention KV Cache 中的 Key。</td>
      <td style="white-space: nowrap;"><code>FLOAT8_E4M3FN</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td>常规 PA 形态为 <code>(block_num,N2,sparse_kv_block_size,D)</code>；组合 KV 存储形态可为一维尾部存储。</td>
      <td><code>layout_kv</code> 当前仅支持 <code>PA_BNSD</code>。常规 PA 形态要求 stride 为 <code>[paBlockStride, blockSize * D, D, 1]</code>。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>value</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>PageAttention KV Cache 中的 Value。</td>
      <td style="white-space: nowrap;"><code>FLOAT8_E4M3FN</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td>常规 PA 形态为 <code>(block_num,N2,sparse_kv_block_size,D_v)</code>；组合 KV 存储形态可为一维尾部存储。</td>
      <td>常规 PA 形态要求 <code>value.stride(0) == key.stride(0)</code>，内部输出 <code>attention_out</code> 为 BF16。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>q_descale</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>Query 反量化缩放因子。</td>
      <td style="white-space: nowrap;"><code>FLOAT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>TND</code> 场景通常为 <code>(T1,N1)</code>；<code>NTD</code> 场景通常为 <code>(N1,T1)</code>。</td>
      <td>与 Query token 和 Query head 对应，用于 QK 结果缩放。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>k_descale</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>Key 反量化缩放因子。</td>
      <td style="white-space: nowrap;"><code>FLOAT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td>常规 PA 形态通常为 <code>(block_num,sparse_kv_block_size,N2)</code> 或与分段 PA 存储匹配的等价布局。</td>
      <td>PyTorch 接入层要求常规 PA 形态下 <code>k_descale.stride(0) * 4 == key.stride(0)</code>，且 stride 后两维为 <code>[blockSize,1]</code>。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>v_descale</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>Value 反量化缩放因子。</td>
      <td style="white-space: nowrap;"><code>FLOAT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>(N2)</code></td>
      <td>按 KV head 生效。组合 KV 存储形态下通过该输入第 0 维推导 <code>N2</code>。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>p_scale</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>softmax 概率 FP8 量化缩放因子。</td>
      <td style="white-space: nowrap;"><code>FLOAT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>(1)</code></td>
      <td>当前为 per-tensor 静态缩放。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>sparse_indices</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>稀疏 KV block 索引。</td>
      <td style="white-space: nowrap;"><code>INT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>(B,N1,max_Qb,max_Kb)</code></td>
      <td><code>layout_sparse_indices</code> 当前仅支持 <code>B_N_Qb_Kb</code>。有效元素表示逻辑 KV block id，取值范围为 <code>[0, max_Kb - 1]</code>。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>sparse_seq_len</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>每个 Query block 对应的有效 KV block 数。</td>
      <td style="white-space: nowrap;"><code>INT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>(B,N1,max_Qb)</code></td>
      <td>每个值应在 <code>[0, max_Kb]</code> 范围内；值为 0 时对应稀疏任务输出置零，LSE 置为 <code>-FLT_MAX</code>。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>atten_mask</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>Attention mask。</td>
      <td style="white-space: nowrap;"><code>UINT8</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td>二维 Tensor，通常为 <code>(sparse_q_block_size,sparse_kv_block_size)</code> 或与实际 mask 窗口匹配的二维形状。</td>
      <td>当前 host 侧校验要求二维；<code>mask_mode=0</code> 时不使用 mask，<code>mask_mode=3</code> 时按 causal 语义使用。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>cu_seqlens_q</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>Query 累积序列长度。</td>
      <td style="white-space: nowrap;"><code>INT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>(B+1)</code></td>
      <td><code>TND/NTD</code> 变长场景建议传入。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>cu_seqlens_kv</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>KV 累积序列长度。</td>
      <td style="white-space: nowrap;"><code>INT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>(B+1)</code></td>
      <td><code>TND/NTD</code> 变长场景建议传入。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>seqused_q</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>每个 batch 的 Query 实际使用长度。</td>
      <td style="white-space: nowrap;"><code>INT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>(B)</code></td>
      <td>可选输入；用于实际长度与最大长度不同的场景。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>seqused_kv</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>每个 batch 的 KV 实际使用长度。</td>
      <td style="white-space: nowrap;"><code>INT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>(B)</code></td>
      <td>可选输入；用于实际长度与最大长度不同的场景。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>block_table</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>PageAttention block 映射表。</td>
      <td style="white-space: nowrap;"><code>INT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>(B,maxBlockNumPerBatch)</code></td>
      <td>可选输入。第 0 维必须等于 <code>B</code>。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>metadata</code></td>
      <td style="white-space: nowrap;">输入</td>
      <td>负载均衡元数据。</td>
      <td style="white-space: nowrap;"><code>INT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td>一维 Tensor，典型值为 <code>(2048)</code></td>
      <td>可选输入，可由 <code>npu_quant_block_sparse_attn_metadata</code> 生成。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>softmax_scale</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>QK 结果缩放因子。</td>
      <td style="white-space: nowrap;"><code>FLOAT32</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td>常用值为 <code>1 / sqrt(D)</code>。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>sparse_q_block_size</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>Query 方向稀疏 block 大小。</td>
      <td style="white-space: nowrap;"><code>INT64</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td>当前实现仅支持 128。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>sparse_kv_block_size</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>KV 方向稀疏 block 大小。</td>
      <td style="white-space: nowrap;"><code>INT64</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td>当前实现仅支持 128。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>max_seqlen_q</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>单 batch Query 最大长度。</td>
      <td style="white-space: nowrap;"><code>INT64</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td><code>TND/NTD</code> 场景下用于推导 <code>S1</code> 和 <code>max_Qb</code>；传 0 时按 <code>query</code> 中 token 维长度推导。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>max_seqlen_kv</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>单 batch KV 最大长度。</td>
      <td style="white-space: nowrap;"><code>INT64</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td>用于推导 <code>S2</code> 和 <code>max_Kb</code>；传 0 时默认按 <code>sparse_kv_block_size</code> 处理。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>pa_block_stride</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>PA 物理 block 外步长。</td>
      <td style="white-space: nowrap;"><code>INT64</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td>常规 4D PA 输入由 <code>key.stride(0)</code> 自动推导；一维组合 KV 存储形态必须显式传入，且大于 0。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>layout_kv</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>KV 数据布局。</td>
      <td style="white-space: nowrap;"><code>STRING</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td>当前仅支持 <code>"PA_BNSD"</code>。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>layout_q</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>Query 数据布局。</td>
      <td style="white-space: nowrap;"><code>STRING</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td>支持 <code>"TND"</code>、<code>"NTD"</code>。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>layout_sparse_indices</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>稀疏索引布局。</td>
      <td style="white-space: nowrap;"><code>STRING</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td>当前仅支持 <code>"B_N_Qb_Kb"</code>。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>layout_out</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>输出布局。</td>
      <td style="white-space: nowrap;"><code>STRING</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td>当前默认 <code>"TND"</code>。<code>NTD</code> 输入在 host infer shape 中会将 <code>attention_out</code> 转为 TND 排布。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>quant_mode</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>量化模式。</td>
      <td style="white-space: nowrap;"><code>INT64</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td>当前仅支持 1，表示 <code>Q_PERTOKEN_PERHEAD_K_PERTOKEN_PERHEAD_V_PERHEAD</code>。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>mask_mode</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>mask 模式。</td>
      <td style="white-space: nowrap;"><code>INT64</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td>当前执行路径支持 0 和 3，0 表示无 mask，3 表示 causal。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>return_softmax_lse</code></td>
      <td style="white-space: nowrap;">输入属性</td>
      <td>是否返回 softmax LSE。</td>
      <td style="white-space: nowrap;"><code>BOOL</code></td>
      <td style="white-space: nowrap;">-</td>
      <td>-</td>
      <td><code>True</code> 时返回有效 LSE；<code>False</code> 时不返回有效 LSE，PyTorch 接入层返回占位 Tensor。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>attention_out</code></td>
      <td style="white-space: nowrap;">输出</td>
      <td>Attention 输出。</td>
      <td style="white-space: nowrap;"><code>BFLOAT16</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td><code>query</code> 为 <code>TND</code> 时输出 <code>(T1,N1,D_v)</code>；<code>query</code> 为 <code>NTD</code> 时 host infer shape 输出 <code>(T1,N1,D_v)</code></td>
      <td>PyTorch 接口的第 1 个返回值。</td>
    </tr>
    <tr>
      <td style="white-space: nowrap;"><code>softmax_lse</code></td>
      <td style="white-space: nowrap;">输出</td>
      <td>softmax log-sum-exp。</td>
      <td style="white-space: nowrap;"><code>FLOAT32</code></td>
      <td style="white-space: nowrap;">ND</td>
      <td>有效 LSE 的逻辑维度为 Query head 和 Query token。PyTorch 接入层当前构造为 <code>(N1,T1)</code>；底层 ACLNN host infer shape 为 <code>(T1,N1)</code>；<code>return_softmax_lse=False</code> 时为占位 Tensor</td>
      <td>PyTorch 接口的第 2 个返回值。</td>
    </tr>
  </tbody>
</table>

## 返回值

PyTorch 接口返回 `(attention_out, softmax_lse)`：

- `attention_out`：BF16 Tensor，最后一维为 `D_v`。PyTorch 接入层按 TND 语义返回 `(T1,N1,D_v)`；`layout_q="NTD"` 输入不会使输出保持 NTD。
- `softmax_lse`：FLOAT32 Tensor。`return_softmax_lse=True` 时，PyTorch 接入层返回 `(N1,T1)`；底层 ACLNN host infer shape 为 `(T1,N1)`。`return_softmax_lse=False` 时返回无有效 LSE 的占位 Tensor。

底层 ACLNN 接口返回 `aclnnStatus`。第一段 GetWorkspaceSize 接口完成参数校验，必选输入为空、数据类型不支持、shape 与属性不匹配时返回参数错误。

## 约束说明

### 约束类型说明

QuantBlockSparseAttn 算子约束分为 4 个档位，按约束复杂程度递增分为单参数约束、存在性约束、一致性约束和特性交叉约束，各档位约束内容如下：

- 单参数约束：对于单个接口参数的约束，包含 Tensor 和 Attribute。
  - 对于 Tensor，单参数约束包含 shape 维度、每一维度取值、dtype、format、是否为空 Tensor、是否连续或 stride 形态等校验。
  - 对于 Attribute，单参数约束包含属性取值范围和默认值语义。
- 存在性约束：约束特定场景下，特性参数组内必须传入某参数，或不支持传入某参数。
- 一致性约束：特性参数组内，各个参数间的 shape、dtype、layout、head 数、序列长度、block 数等一致性约束。
- 特性交叉约束：涉及多个参数组，不同参数组间的交叉约束。例如稀疏索引参数组中的逻辑 KV block id 需要能被 Paged Attention 参数组中的 `block_table` 映射到合法物理 block。

### 特性参数组

<table><thead>
  <tr>
    <th>特性参数组</th>
    <th>参数字段名称</th>
    <th>字段分组</th>
    <th>字段类型</th>
  </tr></thead>
<tbody>
  <tr>
    <td rowspan="12">公共参数组</td>
    <td>query</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>key</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>value</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>softmax_scale</td>
    <td>ATTR</td>
    <td>float</td>
  </tr>
  <tr>
    <td>sparse_q_block_size</td>
    <td>ATTR</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>sparse_kv_block_size</td>
    <td>ATTR</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>max_seqlen_q</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>max_seqlen_kv</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>layout_q</td>
    <td>ATTR</td>
    <td>string</td>
  </tr>
  <tr>
    <td>layout_out</td>
    <td>ATTR</td>
    <td>string</td>
  </tr>
  <tr>
    <td>quant_mode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>attention_out</td>
    <td>OUTPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="4">量化参数组</td>
    <td>q_descale</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>k_descale</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>v_descale</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>p_scale</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="3">稀疏索引参数组</td>
    <td>sparse_indices</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>sparse_seq_len</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>layout_sparse_indices</td>
    <td>ATTR</td>
    <td>string</td>
  </tr>
  <tr>
    <td rowspan="3">Paged Attention参数组</td>
    <td>block_table</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>pa_block_stride</td>
    <td>ATTR</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>layout_kv</td>
    <td>ATTR</td>
    <td>string</td>
  </tr>
  <tr>
    <td rowspan="4">ActualSeqLen参数组</td>
    <td>cu_seqlens_q</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>cu_seqlens_kv</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>seqused_q</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>seqused_kv</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="2">Attention Mask参数组</td>
    <td>atten_mask</td>
    <td>INPUT</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td>mask_mode</td>
    <td>ATTR(OPTIONAL)</td>
    <td>int64</td>
  </tr>
  <tr>
    <td>Metadata参数组</td>
    <td>metadata</td>
    <td>INPUT(OPTIONAL)</td>
    <td>Tensor</td>
  </tr>
  <tr>
    <td rowspan="2">SoftmaxLSE参数组</td>
    <td>return_softmax_lse</td>
    <td>ATTR(OPTIONAL)</td>
    <td>bool</td>
  </tr>
  <tr>
    <td>softmax_lse</td>
    <td>OUTPUT</td>
    <td>Tensor</td>
  </tr>
</tbody></table>

### 基准信息说明

资料约束中，常见字段释义如下：

| 命名 | 含义 |
| :---: | :--- |
| FP8全量化 | `query`、`key`、`value` 为 `FLOAT8_E4M3FN`，`q_descale`、`k_descale`、`v_descale`、`p_scale` 参与反量化/再量化缩放的场景。 |
| PA_BNSD | Paged Attention KV Cache 排布，逻辑形态为 `[PageBlockNum, KeyValueNumHead, BlockSize, HeadDim]`。 |
| 1D组合KV存储 | 将每个物理 PA block 的 Key、Value、`k_descale` 分段打包到同一段一维存储中，接口传入 `key`、`value`、`k_descale` 的一维尾部视图。 |
| BatchSize | Batch 数，对应 `sparse_indices`、`sparse_seq_len`、`block_table` 的第 0 维。 |
| QueryTokenNum | 所有 batch 的 Query 有效 token 数之和，对应 `query` 的 T 轴。 |
| QueryMaxSeqLen | 单个 batch 的 Query 最大序列长度，对应 `max_seqlen_q` 或由 `query` T 轴推导。 |
| KeyValueMaxSeqLen | 单个 batch 的 KV 最大序列长度，对应 `max_seqlen_kv` 或默认的 `sparse_kv_block_size`。 |
| QueryNumHead | Query head 数，对应 `query` 的 N 轴和 `sparse_indices` 的第 1 维。 |
| KeyValueNumHead | KV head 数，对应 `key`、`value` 的 N 轴，组合 KV 存储下由 `v_descale` 第 0 维推导。 |
| GroupNum | GQA 分组数，`GroupNum = QueryNumHead / KeyValueNumHead`。 |
| QueryHeadDim | Query/Key head dim，当前固定为 128。 |
| ValueHeadDim | Value head dim，当前固定为 128。 |
| QueryBlockNumMax | Query block 最大数量，`ceil(QueryMaxSeqLen / sparse_q_block_size)`。 |
| KeyValueBlockNumMax | KV block 最大数量，`ceil(KeyValueMaxSeqLen / sparse_kv_block_size)`。 |
| max_Kb | 每个 Query block 最多关联的 KV block 数，对应 `sparse_indices` 第 3 维，与 `KeyValueBlockNumMax` 含义一致。 |
| PageBlockNum | PA KV Cache 物理 block 总数，对应 4D `key` 第 0 维，或 1D 组合 KV 存储长度除以 `pa_block_stride`。 |
| MaxBlockNumPerBatch | 每个 batch 在 `block_table` 中可索引的最大逻辑 block 数，对应 `block_table` 第 1 维。 |
| PaBlockStride | 相邻物理 PA block 的外步长；4D PA 输入下由 PyTorch 接入层使用 `key.stride(0)` 传给底层，1D 组合 KV 存储下必须由调用者显式传入。 |

### 参数组约束

#### 公共参数组（CommonChecker）

- 单参数约束

  - 当前算子配置支持 `ascend950`。
  - `query`、`key`、`value` 数据类型仅支持 `FLOAT8_E4M3FN`，数据格式仅支持 ND。
  - `query` 仅支持 3D Tensor：
    - `layout_q="TND"` 时，shape 为 `(QueryTokenNum, QueryNumHead, QueryHeadDim)`。
    - `layout_q="NTD"` 时，shape 为 `(QueryNumHead, QueryTokenNum, QueryHeadDim)`。
  - `query` 不支持空 Tensor，且 `QueryHeadDim` 当前固定为 128。
  - `key` 支持 4D PA 形态或 1D 组合 KV 存储尾部视图；4D PA 形态 shape 为 `(PageBlockNum, KeyValueNumHead, sparse_kv_block_size, QueryHeadDim)`，其中 `QueryHeadDim` 固定为 128。
  - `value` 支持与 `key` 对应的 4D PA 形态或 1D 组合 KV 存储尾部视图，`ValueHeadDim` 固定为 128。
  - `attention_out` 数据类型为 `BFLOAT16`，数据格式为 ND。`layout_q="TND"` 时输出 shape 为 `(QueryTokenNum, QueryNumHead, ValueHeadDim)`；`layout_q="NTD"` 时 PyTorch 接入层输出仍为 TND 语义的 `(QueryTokenNum, QueryNumHead, ValueHeadDim)`。
  - `sparse_q_block_size` 和 `sparse_kv_block_size` 当前均仅支持 128。
  - `layout_q` 当前仅支持 `TND`、`NTD`。
  - `layout_out` 当前按 `TND` 输出语义维护；`NTD` 输入不会使输出保持 NTD。
  - `quant_mode` 当前主算子仅支持 1，表示 `Q_PERTOKEN_PERHEAD_K_PERTOKEN_PERHEAD_V_PERHEAD`。
  - `softmax_scale` 为 float 属性，常用值为 `1 / sqrt(QueryHeadDim)`。

- 存在性约束

  - `query`、`key`、`value`、`q_descale`、`k_descale`、`v_descale`、`p_scale`、`sparse_indices`、`sparse_seq_len`、`atten_mask` 为必选输入。
  - 当前算子不支持非量化、伪量化、后量化输出、PSE、ROPE、公共前缀、左 padding、dropout、tensorlist KV 输入等扩展场景。

- 一致性约束

  - `BatchSize`、`QueryNumHead`、`KeyValueNumHead`、`GroupNum` 均必须大于 0。
  - `QueryNumHead` 必须能被 `KeyValueNumHead` 整除。
  - `QueryMaxSeqLen`、`KeyValueMaxSeqLen`、`QueryBlockNumMax`、`KeyValueBlockNumMax` 均必须大于 0。
  - `max_seqlen_q` 传入正数时用于推导 `QueryMaxSeqLen` 和 `QueryBlockNumMax`；小于等于 0 时按 `query` 的 token 维长度推导。
  - `max_seqlen_kv` 传入正数时用于推导 `KeyValueMaxSeqLen` 和 `KeyValueBlockNumMax`；小于等于 0 时按 `sparse_kv_block_size` 推导。
  - 总任务数 `BatchSize * QueryNumHead * QueryBlockNumMax` 必须在 `[1, UINT32_MAX]` 范围内。

- 特性交叉约束

  - 当前执行路径仅支持 FP8 全量化场景；`query`、`key`、`value` 的 head dim 必须与量化参数、稀疏 block 参数和输出 head dim 保持一致。
  - `layout_q` 会影响 `query`、`q_descale`、`attention_out`、`softmax_lse` 的轴解释；其余参数中的 Batch、Head、Block 维度需与该解释保持一致。

#### 量化参数组（QuantChecker）

- 单参数约束

  - `q_descale`、`k_descale`、`v_descale`、`p_scale` 数据类型仅支持 `FLOAT32`，数据格式仅支持 ND。
  - `q_descale` 表示 Query per-token-per-head 反量化缩放，`layout_q="TND"` 时通常为 `(QueryTokenNum, QueryNumHead)`，`layout_q="NTD"` 时通常为 `(QueryNumHead, QueryTokenNum)`。
  - `k_descale` 表示 Key per-token-per-head 反量化缩放，需与 PA KV Cache 的物理 block、KV head 和 block 内 token 对应。
  - `v_descale` 表示 Value per-head 反量化缩放，shape 为 `(KeyValueNumHead)`。
  - `p_scale` 表示 softmax 概率 per-tensor 静态量化缩放，shape 为 `(1)`。

- 存在性约束

  - FP8 全量化场景下 `q_descale`、`k_descale`、`v_descale`、`p_scale` 均必须传入。

- 一致性约束

  - `q_descale` 的 token/head 维度必须与 `query` 的 QueryTokenNum/QueryNumHead 对齐。
  - `k_descale` 的 PA block、KV head、block 内 token 维度必须与 `key`、`block_table`、`sparse_kv_block_size` 对齐。
  - `v_descale` 第 0 维必须等于 `KeyValueNumHead`；1D 组合 KV 存储形态下，底层会通过 `v_descale` 第 0 维推导 `KeyValueNumHead`。
  - `p_scale` 数值应大于 0；Tiling 阶段无法读取 Tensor 数值，该数值合法性由调用者保证。

- 特性交叉约束

  - `quant_mode=1` 时，量化粒度固定为 Query per-token-per-head、Key per-token-per-head、Value per-head、P per-tensor。
  - 常规 4D PA 输入下，`k_descale` 必须满足 Paged Attention 参数组中的分段 PA stride 约束；1D 组合 KV 存储下，`k_descale` 必须是一维尾部视图。

#### 稀疏索引参数组（SparseIndexChecker）

- 单参数约束

  - `sparse_indices`、`sparse_seq_len` 数据类型仅支持 `INT32`，数据格式仅支持 ND。
  - `layout_sparse_indices` 当前仅支持 `B_N_Qb_Kb`。
  - `sparse_indices` 必须为 4D Tensor，shape 为 `(BatchSize, QueryNumHead, QueryBlockNumMax, max_Kb)`，其中 `max_Kb` 必须大于 0。
  - `sparse_seq_len` 必须为 3D Tensor，shape 为 `(BatchSize, QueryNumHead, QueryBlockNumMax)`。

- 存在性约束

  - `sparse_indices` 和 `sparse_seq_len` 均为必选输入。

- 一致性约束

  - `sparse_indices` 的 Batch、Query head、Query block 维度必须分别等于 `BatchSize`、`QueryNumHead`、`QueryBlockNumMax`。
  - `sparse_seq_len` 的 Batch、Query head、Query block 维度必须分别等于 `BatchSize`、`QueryNumHead`、`QueryBlockNumMax`。
  - `sparse_seq_len[b,n,qb]` 表示对应 Query block 的有效 KV block 数，有效范围为 `[0, max_Kb]`；值为 0 时对应稀疏任务输出置零，LSE 置为 `-FLT_MAX`。
  - `sparse_indices` 的有效元素表示逻辑 KV block id，应在 `[0, KeyValueBlockNumMax - 1]` 范围内。

- 特性交叉约束

  - `sparse_indices` 中的逻辑 KV block id 需要能通过 `block_table` 映射到合法 PA 物理 block。
  - Tiling 阶段无法读取 Tensor 数值，`sparse_indices`、`sparse_seq_len` 的数值合法性由调用者保证。

#### Paged Attention参数组（PagedAttentionChecker）

- 单参数约束

  - `layout_kv` 当前仅支持 `PA_BNSD`。
  - `block_table` 数据类型仅支持 `INT32`，数据格式仅支持 ND。当前 PA 执行路径依赖 `block_table` 做逻辑 block 到物理 block 的映射，建议始终传入有效 `block_table`。
  - 传入 `block_table` 时，shape 必须为 `(BatchSize, MaxBlockNumPerBatch)`，第 0 维必须等于 `BatchSize`。
  - `pa_block_stride` 为非负整数属性。常规 4D PA 输入下由 PyTorch 接入层使用 `key.stride(0)` 传给底层；1D 组合 KV 存储下必须显式传入，且必须大于 0。

- 存在性约束

  - 当前算子面向 Paged Attention KV Cache 场景，不支持普通连续 KV 输入。
  - 需要使用逻辑 KV block 到物理 block 映射时，必须传入 `block_table`。

- 一致性约束

  - 常规 4D PA 输入下：
    - `key` stride 必须满足 `[paBlockStride, sparse_kv_block_size * QueryHeadDim, QueryHeadDim, 1]`。
    - `value` stride 必须满足 `[paBlockStride, sparse_kv_block_size * ValueHeadDim, ValueHeadDim, 1]`。
    - `value.stride(0)` 必须等于 `key.stride(0)`。
    - `k_descale.stride(0) * 4` 必须等于 `key.stride(0)`，且 `k_descale` 后两维 stride 必须为 `[sparse_kv_block_size, 1]`。
  - 1D 组合 KV 存储下：
    - `key`、`value`、`k_descale` 必须均为一维尾部视图。
    - `key` 存储长度必须为正数，且必须能被 `pa_block_stride` 整除。
    - `PageBlockNum = key.numel() / pa_block_stride`。
    - 典型打包顺序为每个物理 block 内 `Key block | Value block | k_descale block`，且 `pa_block_stride` 以 FP8 元素数/字节数计。
  - `MaxBlockNumPerBatch` 应大于等于 `KeyValueBlockNumMax`；`block_table` 的有效值必须在 `[0, PageBlockNum - 1]` 范围内。

- 特性交叉约束

  - `block_table` 的 Batch 维度必须与 `sparse_indices`、`sparse_seq_len`、ActualSeqLen 参数组中的 BatchSize 一致。
  - `sparse_indices` 的有效逻辑 block id 必须小于 `MaxBlockNumPerBatch`，否则 `block_table` 映射越界。
  - `block_table` 数值合法性由调用者保证。

#### ActualSeqLen参数组（ActualSeqLenChecker）

- 单参数约束

  - `cu_seqlens_q`、`cu_seqlens_kv`、`seqused_q`、`seqused_kv` 数据类型仅支持 `INT32`，数据格式仅支持 ND。
  - `cu_seqlens_q`、`cu_seqlens_kv` 通常为一维 Tensor，shape 为 `(BatchSize + 1)`。
  - `seqused_q`、`seqused_kv` 通常为一维 Tensor，shape 为 `(BatchSize)`。

- 存在性约束

  - ActualSeqLen 参数均为可选输入。变长 TND/NTD 场景建议传入 `cu_seqlens_q`、`cu_seqlens_kv`、`seqused_q`、`seqused_kv`。

- 一致性约束

  - `cu_seqlens_q` 和 `cu_seqlens_kv` 应从 0 开始单调非降，末尾值分别等于 QueryTokenNum 和实际 KV token 总数。
  - `seqused_q` 每个元素应在 `[0, QueryMaxSeqLen]` 范围内，`seqused_kv` 每个元素应在 `[0, KeyValueMaxSeqLen]` 范围内。
  - `seqused_*`、`cu_seqlens_*` 的 Batch 语义必须与 `sparse_indices`、`sparse_seq_len`、`block_table` 保持一致。

- 特性交叉约束

  - Tiling 阶段无法读取 Tensor 数值，`cu_seqlens_*` 和 `seqused_*` 的数值合法性由调用者保证。
  - `max_seqlen_q`、`max_seqlen_kv` 应分别不小于 `seqused_q`、`seqused_kv` 中的最大有效长度。

#### Attention Mask参数组（MaskChecker）

- 单参数约束

  - `atten_mask` 数据类型仅支持 `UINT8`，数据格式仅支持 ND。
  - `atten_mask` 必须为二维 Tensor。
  - `mask_mode` 当前执行路径仅支持 0 和 3：0 表示无 mask，3 表示 causal mask。

- 存在性约束

  - `atten_mask` 为必选输入。即使 `mask_mode=0`，也需要传入满足 dtype 和维度要求的 Tensor。

- 一致性约束

  - `mask_mode=0` 时，内核按无 mask 语义执行，不使用 `atten_mask` 的数值。
  - `mask_mode=3` 时，`atten_mask` 需与当前 Query/KV block 的 causal 访问窗口匹配。当前测试与推荐输入为二维 `(2048, 2048)` UINT8 mask。

- 特性交叉约束

  - `mask_mode=3` 时，`sparse_indices` 中被选中的 KV block 仍需满足 causal 语义下的有效访问范围。
  - Tiling 阶段无法读取 `atten_mask` 数值，mask 矩阵内容合法性由调用者保证。

#### Metadata参数组（MetadataChecker）

- 单参数约束

  - `metadata` 数据类型仅支持 `INT32`，数据格式仅支持 ND。
  - 有效 `metadata` 为一维 Tensor，shape 为 `(2048)`。

- 存在性约束

  - `metadata` 在主算子 schema 中为可选输入，但当前内核执行依赖 metadata 中的分核调度信息；推荐使用 `npu_quant_block_sparse_attn_metadata` 生成并传入。
  - PyTorch 测试封装在 `metadata` 为 `None`、长度不是 2048 或全 0 时会调用 `npu_quant_block_sparse_attn_metadata` 重新生成；直接调用主算子时需由调用者保证传入有效 metadata。

- 一致性约束

  - `metadata` 必须由与主算子相同的 `sparse_seq_len`、`num_heads_q`、`num_heads_kv`、`head_dim`、`sparse_block_size_q`、`sparse_block_size_k`、`quant_mode`、`mask_mode`、`max_seqlen_q`、`max_seqlen_kv`、`layout_q`、`layout_kv`、`layout_sparse_indices` 生成。
  - 用于主算子的 metadata 生成参数中，`head_dim`、`sparse_block_size_q`、`sparse_block_size_k` 均应为 128，`quant_mode` 应为 1，`mask_mode` 应为 0 或 3，`layout_sparse_indices` 应为 `B_N_Qb_Kb`。

- 特性交叉约束

  - metadata 与 `sparse_seq_len` 或 ActualSeqLen 参数不匹配时，分核调度范围可能与实际稀疏任务不一致，结果不保证正确。

#### SoftmaxLSE参数组（SoftmaxLSEChecker）

- 单参数约束

  - `return_softmax_lse` 为 bool 属性。
  - `softmax_lse` 输出数据类型为 `FLOAT32`。

- 存在性约束

  - `return_softmax_lse=False` 时，PyTorch 接入层返回无有效 LSE 的占位 `softmax_lse` Tensor。
  - `return_softmax_lse=True` 时，返回有效 `softmax_lse` Tensor。

- 一致性约束

  - `softmax_lse` 的 token/head 维度需与 `query` 的 QueryTokenNum/QueryNumHead 对齐。
  - 当 `sparse_seq_len` 对应行没有有效 KV block 时，该行 LSE 置为 `-FLT_MAX`。

- 特性交叉约束

  - `return_softmax_lse` 只影响 `softmax_lse` 是否返回有效结果，不影响 `attention_out` 的 shape、dtype 和计算语义。

## 调用示例

```python
import torch
import torch_npu
import custom_ops

attn_out, softmax_lse = torch.ops.custom.npu_quant_block_sparse_attn(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    sparse_indices,
    sparse_seq_len,
    atten_mask,
    softmax_scale,
    128,
    128,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_kv=cu_seqlens_kv,
    seqused_q=seqused_q,
    seqused_kv=seqused_kv,
    block_table=block_table,
    metadata=metadata,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_kv=max_seqlen_kv,
    layout_kv="PA_BNSD",
    layout_q="TND",
    layout_sparse_indices="B_N_Qb_Kb",
    layout_out="TND",
    quant_mode=1,
    mask_mode=3,
    return_softmax_lse=True,
)
```

## 相关接口

`npu_quant_block_sparse_attn_metadata` 可用于生成 QBSA 负载均衡元数据，其输出作为本算子的 `metadata` 输入。典型调用形式如下：

```python
metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
    sparse_seq_len,
    num_heads_q,
    num_heads_kv,
    head_dim,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_kv=cu_seqlens_kv,
    seqused_q=seqused_q,
    seqused_kv=seqused_kv,
    batch_size=batch_size,
    sparse_block_size_q=128,
    sparse_block_size_k=128,
    quant_mode=1,
    mask_mode=3,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_kv=max_seqlen_kv,
    layout_q="TND",
    layout_kv="PA_BNSD",
    layout_sparse_indices="B_N_Qb_Kb",
)
```
