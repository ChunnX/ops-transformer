# msa\_index\_score

## 产品支持情况

<!-- npu="910b" id1 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="950" id3 -->
- <term>Ascend 950PR/Ascend 950DT</term>：文档标注支持；torch_extension / A2 实现暂未落地 FP8
<!-- end id3 -->

## 功能说明

- **接口功能**：封装 `aclnnMsaIndexScore`，计算 MSA Index Branch 的 block score。
- **公式**：

$$
score = Maxpool[(scale\cdot)Q_{idx}@K_{idx}^{T}+atten\_mask]+local\_mask
$$

- `sparse_mode=0`：无因果；`sparse_mode=3`：rightDownCausal（须传 `[2048,2048]` `atten_mask`）。
- `start_loc[B]`：当前 query 所在逻辑 block 索引，用于 `local_mask`。

## 函数原型

```python
cann_ops_transformer.msa_index_score(
    query,
    key,
    start_loc,
    *,
    block_table=None,
    scale=None,
    atten_mask=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    sparse_mode=3,
) -> Tensor
```

## 参数说明

| 参数名 | 可选/必选 | 描述 | dtype | shape |
|--------|-----------|------|-------|-------|
| query | 必选 | $Q_{idx}$ TND | fp16/bf16 | `[T1,N1,D]` |
| key | 必选 | $K_{idx}$ PA BBND | 同 query 或 int8 | `[NP,P,N2,D]` |
| start_loc | 必选 | query 所在逻辑 block | int32 | `[B]` |
| block_table | PA 必选 | 逻辑→物理 page | int32 | `[B,MB]` |
| scale | 量化必选 | 反量化 | float32 | `[NP,N2,P]` |
| atten_mask | mode=3 必选 | 压缩下三角模板 | int8 | `[2048,2048]` |
| actual_seq_qlen | TND 必选 | query 前缀和 | int32 | `[B+1]` |
| actual_seq_klen | PA 必选 | 各请求 S2 | int32 | `[B]` |
| sparse_mode | 可选 | 0 / 3，默认 3 | int | - |

## 输出

`[N1, T1, RoundUp(MB,16)]` float32
