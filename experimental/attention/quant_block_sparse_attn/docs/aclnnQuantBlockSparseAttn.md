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

> [!NOTE]
> 生成的 ACLNN 头文件原型以编译产物为准。当前 PyTorch 接入层按算子 IR 顺序调用 `aclnnQuantBlockSparseAttn`，输入、属性和输出与本文参数语义一致。

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
      <td><code>layout_sparse_indices</code> 当前仅支持 <code>B_N_Qb_Kb</code>。每个元素表示逻辑 KV block id。</td>
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
      <td><code>True</code> 时返回有效 LSE；<code>False</code> 时 PyTorch 接入层返回空 Tensor。</td>
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
      <td><code>TND</code> 时为 <code>(N1,T1)</code>；<code>NTD</code> 时当前 host infer shape 为 <code>(T1,N1)</code>；<code>return_softmax_lse=False</code> 时为空 Tensor</td>
      <td>PyTorch 接口的第 2 个返回值。</td>
    </tr>
  </tbody>
</table>

## 返回值

PyTorch 接口返回 `(attention_out, softmax_lse)`：

- `attention_out`：BF16 Tensor，最后一维为 `D_v`。`TND` 场景 shape 跟随输入布局，`NTD` 场景 host infer shape 转为 TND 输出。
- `softmax_lse`：FLOAT32 Tensor。`return_softmax_lse=False` 时返回空 Tensor。

底层 ACLNN 接口返回 `aclnnStatus`。第一段 GetWorkspaceSize 接口完成参数校验，必选输入为空、数据类型不支持、shape 与属性不匹配时返回参数错误。

## 约束说明

- 当前算子配置支持 `ascend950`。
- `query`、`key`、`value` 数据类型为 `FLOAT8_E4M3FN`。
- `q_descale`、`k_descale`、`v_descale`、`p_scale` 数据类型为 `FLOAT32`。
- `sparse_indices`、`sparse_seq_len`、`block_table`、`metadata`、`cu_seqlens_q`、`cu_seqlens_kv`、`seqused_q`、`seqused_kv` 数据类型为 `INT32`。
- `atten_mask` 数据类型为 `UINT8`，且必须为二维 Tensor。
- `D` 与 `D_v` 当前固定为 128。
- `sparse_q_block_size` 与 `sparse_kv_block_size` 当前固定为 128。
- `N1` 和 `N2` 均必须大于 0，且 `N1` 必须能被 `N2` 整除。
- `S1`、`S2`、`max_Qb`、`max_Kb` 均必须大于 0。
- `layout_q` 当前仅支持 `TND` 和 `NTD`，`layout_kv` 当前仅支持 `PA_BNSD`，`layout_sparse_indices` 当前仅支持 `B_N_Qb_Kb`，`quant_mode` 当前仅支持 1。
- 当前执行路径支持 `mask_mode=0` 和 `mask_mode=3`。
- 常规 4D PA 输入下，`key`、`value`、`k_descale` 必须满足分段 PA stride 约束；组合 KV 存储形态下，`key`、`value`、`k_descale` 为一维尾部存储，且 `pa_block_stride` 必须显式传入。
- Tiling 阶段无法读取 Tensor 数值，`sparse_indices`、`sparse_seq_len`、`cu_seqlens_*`、`seqused_*` 与 `block_table` 的数值合法性由调用者保证。

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
