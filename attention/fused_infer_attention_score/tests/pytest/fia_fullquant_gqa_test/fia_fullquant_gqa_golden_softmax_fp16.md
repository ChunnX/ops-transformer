# fia_fullquant_gqa_golden.py softmax 计算过程 fp32→fp16 修改说明

## 修改文件

`attention/fused_infer_attention_score/tests/pytest/fia_fullquant_gqa_test/fia_fullquant_gqa_golden.py`

函数：`cpu_fp8_fullquant_gqa_golden`

## 修改背景

NPU 侧 `vf_mul_sel_softmaxflashv2_cast_nz_dn.h` 已将 softmax 计算类型从 fp32 改为 fp16（详见 `vf_mul_sel_softmaxflashv2_cast_nz_dn_softmax_fp16.md`）。为使 golden 参考实现与 NPU 实际行为对齐，需将 golden 中的 softmax 统计量（max、exp、sum）计算过程同步改为 fp16，而 attention 输出累加仍保持 fp32 以保证精度。

## 修改内容

### 1. softmax 统计量初始化类型 fp32→fp16

```python
# 修改前
o_sum = torch.zeros(q_tensor.shape[:-1], dtype=torch.float32)[..., None].contiguous()
minValue = torch.tensor(-3.402823466e+38, dtype=torch.float32)
o_max = torch.full(q_tensor.shape[:-1], minValue.item(), dtype=torch.float32)[..., None].contiguous()
```

改为：

```python
# 修改后
o_sum = torch.zeros(q_tensor.shape[:-1], dtype=torch.float16)[..., None].contiguous()
minValue = torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float16)
o_max = torch.full(q_tensor.shape[:-1], minValue.item(), dtype=torch.float16)[..., None].contiguous()
```

`o_sum`（softmax sum）、`o_max`（softmax max）从 fp32 改为 fp16，`minValue` 从 `-FLT_MAX`（-3.4e+38）改为 `torch.finfo(torch.float16).min`（-65504），与 NPU 侧 half 最小值一致。

### 2. ln_p_scale 类型 fp32→fp16

```python
# 修改前
ln_p_scale = torch.tensor([math.log(p_scale.item())], dtype=torch.float32).contiguous()
```

改为：

```python
# 修改后
ln_p_scale = torch.tensor([math.log(p_scale.item())], dtype=torch.float16).contiguous()
```

pScale 的对数值与 max 做减法，需保持相同类型。

### 3. softmax 核心计算基于 fp16

```python
# 修改前
m_block, _ = torch.max(sij, dim=-1, keepdims=True)
...
pij = torch.where(all_masked_block,
                  torch.zeros_like(sij),
                  torch.exp(sij - mi_new))
s_block = torch.sum(pij, dim=-1, keepdims=True)
```

改为：

```python
# 修改后
sij_fp16 = sij.to(torch.float16)
m_block, _ = torch.max(sij_fp16, dim=-1, keepdims=True)
...
pij = torch.where(all_masked_block,
                  torch.zeros_like(sij_fp16),
                  torch.exp(sij_fp16 - mi_new))
s_block = torch.sum(pij, dim=-1, keepdims=True)
```

得分矩阵 `sij` 先 cast 为 fp16，后续求 max、exp、sum 均在 fp16 下计算，与 NPU 侧 softmax 运算类型一致。

### 4. pij_drop 量化路径调整

```python
# 修改前
pij_drop = pij.to(FP8_DTYPE).to(torch.float32)
```

改为：

```python
# 修改后
pij_drop = pij.to(torch.float32).to(FP8_DTYPE).to(torch.float32)
```

`pij` 此时为 fp16，需先转回 fp32 再量化为 FP8，避免 fp16→FP8 直接转换的精度问题。

### 5. attention 输出累加显式转 fp32

```python
# 修改前
o_blocks[i] = (si * torch.exp(mi - mi_new) * oi + pij_v) / (si_new + EPSILON)
```

改为：

```python
# 修改后
o_blocks[i] = (si.to(torch.float32) * torch.exp(mi - mi_new).to(torch.float32) * oi + pij_v) / (si_new.to(torch.float32) + EPSILON)
```

`si`（sum）、`mi`（max）、`si_new` 此时为 fp16，输出累加时显式转回 fp32，保证 attention 输出精度。

### 6. LSE 输出转回 fp32

```python
# 修改前
lse = out_max + torch.log(out_sum + EPSILON).contiguous()
...
lse = torch.where(all_masked,
                  torch.full_like(out_max, float('inf')),
                  out_max + torch.log(out_sum + EPSILON)).contiguous()
```

改为：

```python
# 修改后
out_sum_f32 = out_sum.to(torch.float32)
out_max_f32 = out_max.to(torch.float32)
lse = out_max_f32 + torch.log(out_sum_f32 + EPSILON).contiguous()
...
lse = torch.where(all_masked,
                  torch.full_like(out_max_f32, float('inf')),
                  out_max_f32 + torch.log(out_sum_f32 + EPSILON)).contiguous()
```

LSE 最终输出为 fp32，`log` 计算在 fp32 下完成避免精度损失。

## 未修改部分

| 项目 | 说明 |
|---|---|
| `out` 初始化 | attention 输出 tensor 保持 fp32 |
| `sij = torch.matmul(qi, kj_T)` | QK^T 矩阵乘法保持 fp32 |
| `pij_v = torch.matmul(pij_drop, vj)` | PV 矩阵乘法保持 fp32 |
| `result` 输出 | fp32 |
| `scale` / `si_new` 中间计算 | 跟随 `mi`/`si` 类型为 fp16 |
| 空 KV 场景 | `result`/`lse` 保持 fp32 不变 |

## 设计原则

1. **softmax 统计量对齐 NPU**：max、exp、sum 在 fp16 下计算，与 NPU 侧 `vf_mul_sel_softmaxflashv2_cast_nz_dn.h` 的 softmax 计算类型一致。
2. **矩阵乘法保持 fp32**：QK^T 和 PV 的 matmul 仍用 fp32，因为这些是 cube 计算，NPU 侧也是高精度计算。
3. **输出累加回 fp32**：attention 输出 `o_blocks` 的累加在 fp32 下完成，避免 fp16 累加误差。
4. **LSE 输出 fp32**：`log` 计算对精度敏感，转回 fp32 后计算。
