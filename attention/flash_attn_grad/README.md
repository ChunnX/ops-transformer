# FlashAttnGrad

## 功能说明

Flash Attention 反向梯度计算算子。根据前向注意力计算的中间结果（softmax_lse、attn_out）和上游梯度（do），计算 Q/K/V 的梯度（dq/dk/dv）。

## 输入输出

### 输入

| 名称 | 类型 | 必选 | 说明 |
|------|------|------|------|
| q | BF16/FP16 | 是 | Query tensor |
| k | BF16/FP16 | 是 | Key tensor |
| v | BF16/FP16 | 是 | Value tensor |
| do | BF16/FP16 | 是 | 上游梯度 |
| attn_out | BF16/FP16 | 是 | 前向注意力输出 |
| softmax_lse | FP32 | 是 | 前向 softmax LSE |
| cu_seqlens_q | INT32 | 否 | TND layout 累积序列长度 |
| cu_seqlens_kv | INT32 | 否 | TND layout 累积序列长度 |
| seqused_q | INT32 | 否 | 实际使用的 Q 序列长度 |
| seqused_kv | INT32 | 否 | 实际使用的 KV 序列长度 |
| sinks | FP32 | 否 | Sink tensor |
| attn_mask | INT8 | 否 | Attention mask (mask_mode=3/4) |
| metadata | INT32 | 否 | FAG metadata tensor |

### 输出

| 名称 | 类型 | 说明 |
|------|------|------|
| dq | BF16/FP16 | Query 梯度 |
| dk | BF16/FP16 | Key 梯度 |
| dv | BF16/FP16 | Value 梯度 |

### 属性

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| softmax_scale | Float | 0.0 | softmax 缩放因子，0.0 表示 1/sqrt(d) |
| mask_mode | Int | 0 | 0:全计算, 3:causal, 4:window |
| win_left | Int | -1 | window mask 左窗口 |
| win_right | Int | -1 | window mask 右窗口 |
| max_seqlen_q | Int | -1 | Q 最大序列长度 |
| max_seqlen_kv | Int | -1 | KV 最大序列长度 |
| layout_q | String | "BSND" | Q 布局 |
| layout_kv | String | "BSND" | KV 布局 |
| layout_out | String | "BSND" | 输出布局 |

## 当前状态

骨架阶段，kernel 为空实现。
