# msa\_index\_score

## 产品支持情况

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id3 -->

## 功能说明

- **接口功能**：

  `msa_index_score` 是基于 `torch_npu` 的 `cann_ops_transformer` 扩展接口，封装 `aclnnMsaIndexScore`，
  用于 MiniMax-M3 / DeepSeek 类稀疏注意力（MSA）Lightning Indexer 的 block 打分：
  对 index Q 与分页 index K cache 做点乘，并在每个 sparse block 内取最大值，产出供下游 `topk` 选块使用的 score。
  Prefill 与 Decode 由同一接口承载。

  支持两条路径：

  - **非量化**：`query` / `key_cache` 同为 float16 或 bfloat16，**不得**传入 `scale`。
  - **int8 前融合**：`key_cache` 为 int8，`scale` 必选，shape `[NP, N_kv, P]`，用于 per-token(+per-head) 反量化；
    当前要求 `query` 为 float16。

- **计算公式**：

$$
q_{\text{pos}}(t) = \texttt{start\_loc}[b] + \bigl(t - \texttt{actual\_seq\_lengths\_query}[b]\bigr)
$$

$$
\texttt{score}[h, t, \beta] =
\max_{\substack{j \in [\beta P,\ \beta P + P) \\ j \le q_{\text{pos}}(t),\ j < \texttt{actual\_seq\_lengths\_key}[b]}}
\bigl\langle \mathbf{q}_{t,h},\ \tilde{\mathbf{k}}_{j} \bigr\rangle
$$

  非量化：$\tilde{\mathbf{k}}_j = \texttt{key\_cache}[\ldots]$；
  int8：$\tilde{\mathbf{k}}_j = \mathrm{cast}(\texttt{key\_cache}[\ldots])\cdot\texttt{scale}[\mathrm{page}, h_{kv}, j\bmod P]$。
  窗口内无可见 $j$ 时输出 $-\infty$（`fill_invalid_tail=True`）。输出止于 block score，**不含** topk。

## 函数原型

```python
cann_ops_transformer.msa_index_score(
    query,
    key_cache,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    start_loc,
    *,
    scale=None,
    block_size=128,
    fill_invalid_tail=True,
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| query | Tensor | 必选 | index query，varlen 扁平布局 | float16、bfloat16 | `[T, Hq, D]` |
| key_cache | Tensor | 必选 | 分页 index key cache；非量化与 query 同 dtype，量化为 int8 | float16、bfloat16、int8 | `[NP, P, Hkv, D]` |
| block_table | Tensor | 必选 | 逻辑 block → 物理 page | int32 | `[B, MB]` |
| actual_seq_lengths_query | Tensor | 必选 | query 长度前缀和（cumsum），首元素为 0 | int32 | `[B+1]` |
| actual_seq_lengths_key | Tensor | 必选 | 每请求可见 kv 长度 | int32 | `[B]` |
| start_loc | Tensor | 必选 | 首个 query token 在自身 kv 序列中的位置 | int32 | `[B]` |
| scale | Tensor | 可选 | int8 反量化系数；非量化必须为 `None` | float32 | `[NP, N_kv, P]` |
| block_size | int | 可选 | sparse block / page token 数，默认 128，当前仅支持 128 | int | - |
| fill_invalid_tail | bool | 可选 | 不可见 block 是否写 `-inf`，默认 True | bool | - |

数据格式均为 `$ND$`。

## 输出说明

| 参数名 | 类型 | 描述 | 数据类型 | 维度(shape) |
|--------|------|------|----------|-------------|
| score | Tensor | block score，末维为对齐后的 block 数 | float32 | `[Hq, T, RoundUp(MB, 16)]` |

## 约束说明

- `block_size` 当前仅支持 **128**，且须等于 `key_cache` 第 1 维。
- `num_kv_heads`（`key_cache` 第 2 维）当前仅支持 **1**。
- `head_dim` 须为 16 的正整数倍，且不超过 128。
- 非量化：`key_cache` dtype 与 `query` 相同，且 `scale` 必须为 `None`。
- int8 前融合：`key_cache` 为 int8，`scale` 必选；当前要求 `query` 为 float16。
- 本接口**不包含** `topk`。

## 调用示例

- 单算子模式调用（非量化）

```python
import torch
import torch_npu
import cann_ops_transformer

block_size = 128
hq, d = 2, 16
q_len, kv_len, start = 2, 5, 3
num_pages, max_blocks = 2, 1
total_q = q_len

query = torch.randn(total_q, hq, d, dtype=torch.float16).npu()
key_cache = torch.randn(num_pages, block_size, 1, d, dtype=torch.float16).npu()
block_table = torch.tensor([[0]], dtype=torch.int32).npu()
cu_seq_q = torch.tensor([0, q_len], dtype=torch.int32).npu()
kv_len_t = torch.tensor([kv_len], dtype=torch.int32).npu()
start_loc = torch.tensor([start], dtype=torch.int32).npu()

score = cann_ops_transformer.msa_index_score(
    query, key_cache, block_table, cu_seq_q, kv_len_t, start_loc,
    block_size=block_size, fill_invalid_tail=True)
print(score.shape, score.dtype)  # [Hq, T, RoundUp(MB,16)], float32
```

- int8 前融合

```python
key_i8 = torch.randint(-64, 64, (num_pages, block_size, 1, d), dtype=torch.int8).npu()
dequant_scale = torch.rand(num_pages, 1, block_size, dtype=torch.float32).npu() * 0.02 + 0.01
query_fp16 = torch.randn(total_q, hq, d, dtype=torch.float16).npu()

score_q = cann_ops_transformer.msa_index_score(
    query_fp16, key_i8, block_table, cu_seq_q, kv_len_t, start_loc,
    scale=dequant_scale, block_size=block_size)
```

- 图模式调用

  经 `graph_convert_msa_index_score` 注册的 GE Converter 下沉为 `MsaIndexScore` 算子。需安装 torchair。

```python
import torchair

class MsaIndexScoreModel(torch.nn.Module):
    def forward(self, query, key_cache, block_table, cu_seq_q, kv_len, start_loc):
        return cann_ops_transformer.msa_index_score(
            query, key_cache, block_table, cu_seq_q, kv_len, start_loc,
            block_size=128, fill_invalid_tail=True)

model = torch.compile(MsaIndexScoreModel().npu(), backend=torchair.get_npu_backend(), dynamic=False)
out = model(query, key_cache, block_table, cu_seq_q, kv_len_t, start_loc)
```
