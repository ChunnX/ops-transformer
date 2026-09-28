# msa\_index\_score

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

- **接口功能**：封装 `aclnnMsaIndexScore`，计算 MSA Index Branch 的 block score。对每个 query token 与每个 KV sparse block，取该 block 内因果可见 token 的 $Q_{idx}$ 与 $K_{idx}$ 做 matmul 后按 block Maxpool，得到逐 block 重要性分数，作为后续 TopK 的输入。

- **计算公式**：

$$
score = Maxpool[(scale\cdot)Q_{idx}@K_{idx}^{T}+atten\_mask]+local\_mask
$$

- **主要计算过程**：

  1. 将每个 query token 的 $Q_{idx}$ 与 KV sparse block 内因果可见的 $K_{idx}$ 做矩阵乘，得到 token 级相关性；int8 场景先按 $scale$ 反量化。
  2. 按 `sparse_mode` 叠加 $atten\_mask$：`sparse_mode=0` 不做因果截断，`sparse_mode=3` 按 rightDownCausal 屏蔽不可见位置。
  3. 在每个长度为 $block\_size$ 的 sparse block 上做 Maxpool，得到逐 block 的重要性分数。
  4. 用 `start_loc`、`init_blocks`、`local_blocks` 生成 $local\_mask$，对强制保留的 block 写入高分，得到输出 `score`。本接口不包含 TopK。

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
    layout_key="BBND",
    sparse_mode=3,
    init_blocks=0,
    local_blocks=1,
) -> Tensor
```

## 参数说明

>**说明：**<br>
>
>- B 表示 Batch Size，S1 / S2 分别表示 query / key 的序列长度，T1 / T2 分别表示 query / key 的 token 累加和，N1 / N2 分别表示 query / key 的 Head Num，D 表示 Head Dim。PageAttention 场景下 $block\_num$ 为物理 block 总数，$block\_size$ 为每个 block 的 token 数。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| query | Tensor | 必选 | 公式中的 $Q_{idx}$，TND。数据格式为 ND。 | bfloat16、float16；<term>Ascend 950PR/Ascend 950DT</term> 另支持 float8_e4m3fn、float8_e5m2、hifloat8 | (T1, N1, D) |
| key | Tensor | 必选 | 公式中的 $K_{idx}$。PA 为 BBND / BNBD，或 TND packed。A2/A3 与 950 上 PA 允许 dim0 非连续；size>1 的非首轴必须连续。TND 不允许非连续。数据格式为 ND。 | 与 query 相同，或 int8（须配 float16 query） | BBND 为 (block_num, block_size, N2, D)；BNBD 为 (block_num, N2, block_size, D)；TND 为 (T2, N2, D) |
| start_loc | Tensor | 必选 | 当前 query 所在逻辑 block 索引，用于生成 $local\_mask$。数据格式为 ND。 | int32 | (B,) |
| block_table | Tensor | 可选 | PageAttention 逻辑 block 到物理 page 的映射。`layout_key` 为 BBND / BNBD 时必传，TND 不传。数据格式为 ND。 | int32 | (B, maxBlockNumPerSeq) |
| scale | Tensor | 可选 | 反量化系数。非量化必须为空；int8 量化必传。数据格式为 ND。 | float32 | PA 为 (block_num, N2, block_size)；TND 为 (T2, N2) |
| atten_mask | Tensor | 可选 | 压缩下三角模板。`sparse_mode=3` 时必传，`sparse_mode=0` 时不传。数据格式为 ND。 | int8 | (2048, 2048) |
| actual_seq_qlen | Tensor | 可选 | query 为 TND 时必传，单调不减前缀和。数据格式为 ND。 | int32 | (B+1,) |
| actual_seq_klen | Tensor | 必选 | PA 为各请求可见 S2；TND 为 key 前缀和。数据格式为 ND。 | int32 | PA 为 (B,)；TND 为 (B+1,) |
| layout_key | str | 可选 | key 布局，支持 TND、BBND、BNBD，默认值为 BBND。 | string | - |
| sparse_mode | int | 可选 | 0 表示 defaultMask，3 表示 rightDownCausal，默认值为 3。 | int64 | - |
| init_blocks | int | 可选 | $local\_mask$ 头部强制块数，默认值为 0。 | int64 | - |
| local_blocks | int | 可选 | $local\_mask$ 局部窗口长度，默认值为 1。与 Triton raw score 对齐时置 0。 | int64 | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| score | Tensor | 必选 | 公式中的 $score$，逐 block 重要性分数。不可见 block 填 -inf。数据格式为 ND。 | float32 | (N1, T1, RoundUp(maxBlockNumPerSeq, 16)) |

## 约束说明

- 该接口支持推理场景下使用。
- 当前 $block\_size$ 仅支持 128。
- `layout_key` 必须与 `key` 的实际 shape 一致，取值为 `"BBND"`、`"BNBD"` 或 `"TND"`。
- PageAttention（`layout_key` 为 `"BBND"` / `"BNBD"`）必须传入 `block_table`；TND 不得传入 `block_table`，`actual_seq_klen` 为 `(B+1,)` 前缀和。
- 非量化场景下 `key` 的数据类型与 `query` 相同，`scale` 必须为空。量化场景仅支持 int8 key 搭配 float16 query，且必须传入 `scale`。
- `sparse_mode` 仅支持 0 和 3。为 3 时必须传入 shape 为 `(2048, 2048)` 的 `atten_mask`。
- `init_blocks`、`local_blocks` 必须大于等于 0，且不超过逻辑 block 数。两者均为 0 时跳过 $local\_mask$。
- `q_len` / `kv_len` 允许为 0（含整 batch）。对应请求跳过 QK；空 KV 的 score 为 -inf。
- PageAttention 的 `key` 仅允许 dim0 非连续，须保持为 view；TND 的 `key` 必须连续。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - `query`、`key` 额外支持同型 float8_e4m3fn、float8_e5m2、hifloat8，且不得传入 `scale`。
  - 短 M 长 S 时按可见 KV stile 切分 `kvChunks`。score 末维超过 256 列时按窗写回。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - 不支持 float8_e4m3fn、float8_e5m2、hifloat8。

## 确定性计算

默认支持确定性计算

## 调用示例

- PageAttention BBND：

```python
import torch
import torch_npu
import cann_ops_transformer

T1, N1, N2, D, P = 32, 8, 1, 128, 128
B, NP, MB = 1, 8, 2
query = torch.randn(T1, N1, D, dtype=torch.float16).npu()
key_bbnd = torch.randn(NP, P, N2, D, dtype=torch.float16).npu()
block_table = torch.arange(B * MB, dtype=torch.int32).view(B, MB).npu()
actual_seq_qlen = torch.tensor([0, T1], dtype=torch.int32).npu()
actual_seq_klen = torch.tensor([256], dtype=torch.int32).npu()
start_loc = torch.tensor([1], dtype=torch.int32).npu()
atten_mask = torch.zeros(2048, 2048, dtype=torch.int8).npu()
score = cann_ops_transformer.msa_index_score(
    query, key_bbnd, start_loc,
    block_table=block_table, atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen,
    layout_key="BBND")
```

- TND packed key（`layout_key="TND"`，不传 `block_table`，`actual_seq_klen` 为 `[B+1]` 前缀和）：

```python
T2 = 256
key_tnd = torch.randn(T2, N2, D, dtype=torch.float16).npu()
actual_seq_klen_tnd = torch.tensor([0, T2], dtype=torch.int32).npu()
score_tnd = cann_ops_transformer.msa_index_score(
    query, key_tnd, start_loc,
    atten_mask=atten_mask,
    actual_seq_qlen=actual_seq_qlen, actual_seq_klen=actual_seq_klen_tnd,
    layout_key="TND")
```
