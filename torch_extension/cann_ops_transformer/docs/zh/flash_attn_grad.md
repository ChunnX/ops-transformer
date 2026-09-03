# flash_attn_grad

## 功能说明

Flash Attention 反向梯度计算。根据前向注意力计算的中间结果（softmax_lse、attn_out）和上游梯度（do），计算 Query、Key、Value 的梯度（dq、dk、dv）。

## 接口说明

```python
cann_ops_transformer.flash_attn_grad(
    q, k, v, do, attn_out, softmax_lse,
    cu_seqlens_q=None, cu_seqlens_kv=None,
    seqused_q=None, seqused_kv=None,
    sinks=None, attn_mask=None, metadata=None,
    softmax_scale=0.0, mask_mode=0, win_left=-1, win_right=-1,
    max_seqlen_q=-1, max_seqlen_kv=-1,
    layout_q="BSND", layout_kv="BSND", layout_out="BSND"
) -> (dq, dk, dv)
```

## 参数说明

### 输入参数

| 参数名 | 类型 | 必选 | 说明 |
|--------|------|------|------|
| q | Tensor(BF16/FP16) | 是 | Query tensor |
| k | Tensor(BF16/FP16) | 是 | Key tensor |
| v | Tensor(BF16/FP16) | 是 | Value tensor |
| do | Tensor(BF16/FP16) | 是 | 上游梯度 |
| attn_out | Tensor(BF16/FP16) | 是 | 前向注意力输出 |
| softmax_lse | Tensor(FP32) | 是 | 前向 softmax LSE |
| cu_seqlens_q | Tensor(INT32) | 否 | TND layout Q 累积序列长度 |
| cu_seqlens_kv | Tensor(INT32) | 否 | TND layout KV 累积序列长度 |
| seqused_q | Tensor(INT32) | 否 | Q 实际使用的序列长度 |
| seqused_kv | Tensor(INT32) | 否 | KV 实际使用的序列长度 |
| sinks | Tensor(FP32) | 否 | Sink tensor |
| attn_mask | Tensor(INT8) | 否 | Attention mask（mask_mode 为 3 或 4 时需要） |
| metadata | Tensor(INT32) | 否 | 预计算的 metadata tensor（由 flash_attn_metadata 生成） |

### 属性参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| softmax_scale | float | 0.0 | Softmax 缩放因子，0.0 表示 1/sqrt(head_dim) |
| mask_mode | int | 0 | 0: 全计算, 3: causal, 4: window |
| win_left | int | -1 | Window mask 左窗口大小，-1 表示无穷 |
| win_right | int | -1 | Window mask 右窗口大小，-1 表示无穷 |
| max_seqlen_q | int | -1 | Q 最大序列长度 |
| max_seqlen_kv | int | -1 | KV 最大序列长度 |
| layout_q | str | "BSND" | Q 布局：BSND/TND/BNSD |
| layout_kv | str | "BSND" | KV 布局：BSND/TND/BNSD |
| layout_out | str | "BSND" | 输出布局：BSND/TND/BNSD |

### 输出

| 输出 | 类型 | 说明 |
|------|------|------|
| dq | Tensor(BF16/FP16) | Query 梯度，shape 与 q 相同 |
| dk | Tensor(BF16/FP16) | Key 梯度，shape 与 k 相同 |
| dv | Tensor(BF16/FP16) | Value 梯度，shape 与 v 相同 |

## 约束说明

- `softmax_scale` 为 0.0 时自动使用 `1/sqrt(head_dim)`。
- `mask_mode` 为 0 或 3 时，`win_left` 和 `win_right` 必须为 -1。
- `mask_mode` 为 4 时，`win_left` 和 `win_right` 有效，-1 表示无穷。
- TND layout 时需要提供 `cu_seqlens_q` 和 `cu_seqlens_kv`。
- `metadata` 参数为可选，由 `flash_attn_metadata` 预计算生成，包含 FAG 分核调度信息。

## 调用示例

```python
import torch
import torch_npu
import cann_ops_transformer

# 前向计算
attn_out, softmax_lse = cann_ops_transformer.flash_attn(q, k, v, ...)

# 反向梯度计算
dq, dk, dv = cann_ops_transformer.flash_attn_grad(
    q, k, v, do, attn_out, softmax_lse,
    mask_mode=0,
    layout_q="BSND", layout_kv="BSND", layout_out="BSND",
)
```
