# QBSA MXFP8 Golden 优化总结

对比文件：
- 原始版本：`wangke/QBSA-MXFP8/.../qbsa_mxfp8_golden.py`
- 优化版本：`ops-transformer/.../qbsa_mxfp8_golden.py`

---

## 一、MXFP8 量化公式修复（精度修复，核心改动）

### 1.1 descale 计算公式从 `floor(log2(max)) - emax` 改为 `ceil(log2(max / fp8_max))`

**位置**：`quantize_mxfp8_qk` / `quantize_mxfp8_v` 函数

**原始代码**：
```python
emax_elem = _EMAX_MAP[fp8_dtype]  # = 8
shared_exp = torch.floor(torch.log2(max_vals)) - emax_elem
descale = torch.where(all_zero_mask, torch.ones_like(shared_exp), 2 ** shared_exp)
```

**优化代码**：
```python
fp8_max = torch.finfo(fp8_dtype).max  # = 448.0
shared_exp = torch.ceil(torch.log2(max_vals / fp8_max))
descale = torch.where(all_zero_mask, torch.ones_like(shared_exp), 2 ** shared_exp)
```

**优化原因**：
- `floor(log2(max)) - 8` 只考虑了指数位，将 group max 映射到 `[256, 512)` 区间
- 但 e4m3fn 最大可表示值是 **448**（= 1.75 × 2^8），不是 256
- 当 group max 超过 448 时，`max / descale` 超过 448，量化后饱和，反量化值最大误差达 12.5%
- `ceil(log2(max / 448))` 确保所有值除以 descale 后 ≤ 448，**无饱和**
- 大 data_range 下（如 128），几乎每个 group 的 max 都会触发饱和，是精度问题的根因之一

### 1.2 量化函数从"仅返回 scale"改为"返回 fp8 + descale 配套"

**原始代码**：
```python
def get_mxfp8_per_token_group_quant_scale(tensor, ...):
    ...
    return descale  # 只返回 scale，fp8 数据由外部 _rand_fp8 单独生成

# 调用方：
query = _rand_fp8(shape, generator, DATA_RANGE_Q)
q_descale = get_mxfp8_per_token_group_quant_scale(query.unsqueeze(0), ...)
```

**优化代码**：
```python
def quantize_mxfp8_qk(tensor_fp32, ...):
    ...
    quantized = (tensor_fp32 / descale_expanded).clamp(-448.0, 448.0).to(fp8_dtype)
    return quantized, descale  # 返回配套的 fp8 + descale

# 调用方：
q_fp32 = _rand_float(shape, generator, DATA_RANGE_Q)
query, q_descale = quantize_mxfp8_qk(q_fp32.unsqueeze(0), ...)
```

**优化原因**：
- 原始方式先随机生成 FP8，再从 FP8 数据反推 descale，`fp8 × descale` 不一定等于原始值
- 优化方式从 FP32 出发，先算 descale 再量化 FP8，保证 `fp8 × descale ≈ 原始 FP32`
- 数据生成流程与 NPU kernel 的 `MxMatmulFull` 随路反量化路径一致

### 1.3 函数泛化：支持任意维度输入

**原始代码**：
```python
def get_mxfp8_per_token_group_quant_scale(tensor, ...):
    dim1, dim2, dim3, dim4 = tensor.shape  # 只支持 4D
    grouped = tensor.reshape(dim1, dim2, dim3, num_groups, group_size)
```

**优化代码**：
```python
def quantize_mxfp8_qk(tensor_fp32, ...):
    last_dim = tensor_fp32.shape[-1]  # 支持任意维度
    grouped = tensor_fp32.reshape(*tensor_fp32.shape[:-1], num_groups, group_size)
```

**优化原因**：Q 是 3D（TND/NTD），K 是 4D（BSND），原始代码需要 `unsqueeze` 补维度，优化后直接支持。

---

## 二、softmax 路径对齐 NPU kernel（精度修复）

### 2.1 max 对齐与减 ln_p_scale 的顺序调整

**原始代码**：
```python
local_max = _align_up_to_ln2(masked_scores.max(dim=-1).values)
# ... 合并 max ...
m_candidate = _align_up_to_ln2(m_candidate)
# P = exp(score - max + ln_p_scale)
p_subloop = torch.exp(s_subloop - safe_m_subloop.view(nq, 1) + ln_p_scale)
```

**优化代码**：
```python
local_max = masked_scores.max(dim=-1).values
local_max = _align_up_to_ln2(local_max - ln_p_scale)  # 先减 ln_p_scale 再对齐
# ... 合并 max (不再二次对齐) ...
# P = exp(score - max), max 已含 -ln_p_scale
p_subloop = torch.exp(s_subloop - safe_m_subloop.view(nq, 1))
```

**优化原因**：
- NPU kernel 顺序：`align(max * dScale)` → `Sub(ln_p_scale)` → `exp(score - max)`
- 原始 golden 顺序：`align(max)` → 合并 → `align(合并结果)` → `exp(score - max + ln_p_scale)`
- 优化后对齐 NPU：`align(max - ln_p_scale)` → 合并 → `exp(score - max)`
- 消除了二次 `_align_up_to_ln2` 调用，与 NPU 的 max 处理路径一致

### 2.2 LSE 恢复公式修复

**原始代码**：
```python
lse = torch.log(safe_l) + m_run  # m_run 未减 ln_p_scale
```

**优化代码**：
```python
lse = torch.log(safe_l) + m_run + ln_p_scale  # m_run 已减 ln_p_scale，需加回
```

**优化原因**：对齐 NPU 后，`m_run` 存储的是 `aligned_max - ln_p_scale`，计算 LSE 时需要加回 `ln_p_scale` 恢复原始语义。

### 2.3 L 累加改用 FP8 P（对齐 NPU e4m3fn 路径）

**原始代码**：
```python
round_sum += p_subloop.sum(dim=-1) * subloop_rescale  # FP32 P
```

**优化代码**：
```python
round_sum += p_quant_subloop.sum(dim=-1) * subloop_rescale  # FP8→FP32 P
```

**优化原因**：
- NPU kernel 在 e4m3fn 路径下，L 累加用的是 FP8 cast 回 FP32 的值（不是原始 FP32 exp 值）
- 原始 golden 用 FP32 P 累加 L，导致 LSE 与 NPU 不一致
- 优化后用 `p_quant_subloop`（FP8→FP32），与 NPU 的 `Cast<fp8_e4m3fn> → Cast<fp32> → Add(sum)` 路径一致

### 2.4 输出 bf16 截断（对齐 NPU 输出 dtype）

**原始代码**：
```python
attention_out[out_idx, head_idx] = attn[local_idx]  # float32
```

**优化代码**：
```python
attention_out[out_idx, head_idx] = attn[local_idx].to(torch.bfloat16)  # 模拟 NPU bf16 输出
```

**优化原因**：
- NPU kernel 的 `CopyOutAttentionOut` 通过 `Cast<bfloat16_t, CAST_ROUND>` 输出 bf16
- 原始 golden 输出 FP32，缺少 bf16 量化损失
- 优化后模拟 NPU 的 bf16 截断，使两边的量化误差对齐

---

## 三、NPU 调用接口适配

### 3.1 删除已废弃的算子参数

**原始代码**：
```python
metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
    ...,
    max_seqlen_q=max_sq,
    max_seqlen_kv=max_skv,
)
output = torch.ops.custom.npu_quant_block_sparse_attn(
    ...,
    max_seqlen_q=max_sq,
    max_seqlen_kv=max_skv,
    pa_block_stride=0,
)
```

**优化代码**：
```python
metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
    ...,
    # 删除 max_seqlen_q, max_seqlen_kv
)
output = torch.ops.custom.npu_quant_block_sparse_attn(
    ...,
    # 删除 max_seqlen_q, max_seqlen_kv, pa_block_stride
)
```

**优化原因**：`3a97295fa`（QBSA算子检视意见整改）删除了 `max_seqlen_q`、`max_seqlen_kv`、`pa_block_stride` 三个属性，golden 需同步适配。

### 3.2 empty_actual_seq 场景修复

**原始代码**：
```python
if CASE.get("empty_actual_seq", False):
    seqused_q = torch.empty((0,), dtype=torch.int32)
    seqused_kv = torch.empty((0,), dtype=torch.int32)  # 空 tensor
```

**优化代码**：
```python
if CASE.get("empty_actual_seq", False):
    seqused_q = torch.empty((0,), dtype=torch.int32)
    seqused_kv = torch.tensor(kv_lengths, dtype=torch.int32)  # 传真实值
```

**优化原因**：`3a97295fa` 将 `seqused_kv` 从可选改为必需（不可为空），且 op_host check 要求 `dim[0] == B`。空 tensor 的 `dim[0]=0 ≠ B=1`，导致 `call aclnnQuantBlockSparseAttn failed`。

---

## 四、代码格式化（非功能性改动）

大量改动是 Python 代码格式化（black 风格），不涉及逻辑变更：

| 格式化类型 | 示例 |
|---|---|
| 集合多行展开 | `{"a", "b"}` → `{"a",\n "b",\n}` |
| 函数参数多行 | `f(a, b, c)` → `f(\n a,\n b,\n c,\n)` |
| 字符串引号统一 | `'...'` → `"..."` |
| 切片空格 | `a:b+c` → `a : b + c` |
| 链式调用换行 | `.a().b().c()` → `.a()\n.b()\n.c()` |
| global 多行 | `global A, B, C` → `global \\\n A, \\\n B, \\\n C` |

---

## 五、优化效果总结

| 优化项 | 类型 | 影响范围 | 精度影响 |
|---|---|---|---|
| descale 公式 `ceil(log2(max/448))` | 精度修复 | Q/K/V 量化 | 消除大 data_range 下的饱和误差 |
| 配套生成 fp8+descale | 精度修复 | 数据生成 | 保证反量化值正确 |
| max 对齐顺序 | 精度修复 | softmax | 对齐 NPU kernel 的 max 处理 |
| LSE 恢复公式 | 精度修复 | LSE 输出 | 修正 LSE 的 ln_p_scale 偏移 |
| L 累加用 FP8 P | 精度修复 | LSE 输出 | 对齐 NPU e4m3fn 的 L 累加路径 |
| 输出 bf16 截断 | 精度修复 | attention_out | 对齐 NPU 的 bf16 输出量化损失 |
| 删除废弃参数 | 接口适配 | NPU 调用 | 适配检视整改后的算子 schema |
| seqused_kv 修复 | 接口适配 | empty_actual_seq | 修复算子调用失败 |
| 代码格式化 | 非功能 | 全文件 | 无 |
