# generic\_lightning\_indexer

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

  `generic_lightning_indexer`封装 `aclnnGenericLightningIndexer`，对已经完成前处理的 Index Query / Index Key 做 Matmul、激活、沿 head 规约、掩码、TopK 以及可选的 Expand / 追加尾 token，得到每一个 token 对应的 Top-$k$ 位置。

  不包含 RoPE、RMSNorm、AvgPool、KeyPool、CSA 压缩。压缩后的 Key 通过 `cmp_ratio` 消费。该接口用于收编 `lightning_indexer`、`lightning_indexer_v2`、`quant_lightning_indexer`、`quant_lightning_indexer_v2`、`pool_key_indexer` 以及 Qwen Compressed Lightweight Indexer。Step 1 不做独立 `metadata` 算子，`metadata` 张量可选传入。

- **计算公式**：

  $$
  idx = \mathrm{Post}\Bigl(\mathrm{Top}\text{-}k\bigl(\mathrm{Reduce}_{h}(\phi(\alpha\, Q K^{\top}))\odot M\bigr)\Bigr)
  $$

  `topk` 表示打分轴选出的个数。旧 PKI 的 token 预算需先除以 `pool_size` 再传入。

## 函数原型

```python
cann_ops_transformer.generic_lightning_indexer(
    query,
    key,
    *,
    weights=None,
    actual_seq_q=None,
    actual_seq_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    q_descale=None,
    k_descale=None,
    output_idx_offset=None,
    metadata=None,
    start_loc=None,
    topk=2048,
    cmp_ratio=1,
    layout_q="BSND",
    layout_k="BSND",
    sparse_mode=0,
    act_type=0,
    has_weighted_sum=True,
    expand_indices=False,
    append_tail=False,
    quant_mode=-1,
    return_value=False,
    select_mode=0,
    key_reduce_mode=0,
    score_scale=1.0,
    key_stride0=-1,
    k_descale_stride0=-1,
    max_seqlen_q=-1,
    init_blocks=0,
    local_blocks=0,
) -> (Tensor, Tensor)
```

## 参数说明

> **说明：**
>
> B 表示 Batch Size、S1 / S2 表示 Sequence Length（S2 为压缩后长度）、N1 / N2 表示 Head Num、D 表示 Head Dim（仅支持 128）、T1 / T2 表示 Total Tokens。N2 仅支持 1。

| 参数名 | 可选/必选 | 描述 | dtype | shape |
|--------|-----------|------|-------|-------|
| query | 必选 | Indexer Q | fp16 / bf16；950 另支持 int8 / fp8_e4m3fn / hifloat8 / fp4_e2m1 / fp8_e5m2 | BSND `[B,S1,N1,D]`；TND `[T1,N1,D]` |
| key | 必选 | 已压缩或未压缩的 Indexer K | 同 query；MSA 可为 int8（query=fp16） | BSND `[B,S2,1,D]`；TND `[T2,1,D]`；PA `[BN,BS,1,D]` |
| weights | has_weighted_sum=true 必选 | 多头权重 | fp16 / bf16 / fp32 | BSND `[B,S1,N1]`；TND `[T1,N1]` |
| actual_seq_q | TND 必选 | Query 有效长度 | int32 / int64 | `[B]` exclusive 或 `[B+1]` cu_seqlens |
| actual_seq_k | TND / PA 必选 | Key 压缩轴长度 | int32 / int64 | 同上 |
| seqused_q | 可选 | Query 真实长度（非前缀和） | int32 / int64 | `[B]` |
| seqused_k | 可选 | Key 真实长度 | int32 / int64 | `[B]` |
| cmp_residual_k | r>1 且 sparse_mode!=0 必选 | 压缩残余 | int32 / int64 | `[B]` |
| block_table | PA 必选 | page 映射 | int32 | `[B, maxBlockNum]` |
| q_descale | 量化必选 | Q 反量化系数 | fp32 / fp16 / fp8_e8m0 | 随 quant_mode |
| k_descale | 量化或 INT8 key 必选 | K 反量化系数 | 同 q_descale；MSA FP16×INT8 只要求本输入 | 随 quant_mode |
| output_idx_offset | 可选 | 输出索引偏移 | int32 | `[B]` |
| metadata | 可选 | 预留负载均衡表 | int32 | `[1024]` |
| start_loc | 可选 | MSA：query 所在逻辑 block | int32 / int64 | `[B]` |
| topk | 可选 | 打分轴 k | int | 默认 2048 |
| cmp_ratio | 可选 | 压缩比 | int | 默认 1 |
| layout_q | 可选 | Query 布局 | str | `BSND` / `TND` |
| layout_k | 可选 | Key 布局 | str | `BSND` / `TND` / `PA_BBND` / `PA_BSND` / `BBND` / `BNBD` |
| sparse_mode | 可选 | mask | int | 0 / 3 / 4 |
| act_type | 可选 | 激活 | int | 0 ReLU / 1 Identity |
| has_weighted_sum | 可选 | 是否加权和 | bool | 默认 True |
| expand_indices | 可选 | 是否展开 | bool | 默认 False |
| append_tail | 可选 | 是否追加尾 token | bool | 默认 False |
| quant_mode | 可选 | 量化模式 | int | -1 / 0 / 1 / 2 / 3 / 4 / 5 |
| return_value | 可选 | 是否输出分数 | bool | 默认 False |
| select_mode | 可选 | 选择模式 | int | 0 TopK / 1 none |
| key_reduce_mode | 可选 | Key 轴规约 | int | 0 none / 1 maxpool |
| score_scale | 可选 | QK 缩放 | float | 默认 1.0 |
| key_stride0 | 可选 | PA key 0 轴 stride | int | 默认 -1 |
| k_descale_stride0 | 可选 | PA k_descale 0 轴 stride | int | 默认 -1 |
| max_seqlen_q | 可选 | TND 最大 Q 长度 | int | 默认 -1 |
| init_blocks | 可选 | MSA local_mask sink 块数 | int | 默认 0 |
| local_blocks | 可选 | MSA local_mask 窗口块数 | int | 默认 0 |
| sparse_indices | 输出 | 选中下标，无效 -1；MSA 为 `(0,)` | int32 | BSND `[B,S1,L_out]`；TND `[T1,L_out]` |
| sparse_values | 输出 | TopK 分数；return_value=False 时 `(0,)`；MSA 为 keep-heads Maxpool | float32 | BSND `[B,S1,topk]`；TND `[T1,topk]`；MSA BSND `[B,S1,N1,nOut]`；MSA TND `[N1,T1,nOut]` |

## 约束说明

与 [aclnnGenericLightningIndexer](./aclnnGenericLightningIndexer.md) 相同。`PA_BSND` / `BBND` 视为 `PA_BBND` 别名；`BNBD` 为 `[BN,N2,BS,D]`（N2=1 时与 BBND 等价）。序列张量支持 int32 与 int64。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_transformer

query = torch.randn(1, 32, 8, 128, dtype=torch.float16, device="npu")
key = torch.randn(1, 1024, 1, 128, dtype=torch.float16, device="npu")
weights = torch.randn(1, 32, 8, dtype=torch.float16, device="npu")
indices, values = cann_ops_transformer.generic_lightning_indexer(
    query,
    key,
    weights=weights,
    topk=2048,
    cmp_ratio=1,
    layout_q="BSND",
    layout_k="BSND",
    sparse_mode=3,
    has_weighted_sum=True,
    return_value=True,
)
```
