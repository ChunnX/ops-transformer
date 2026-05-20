# Python端Metadata计算集成指南

## 目标

使用纯 Python 实现替换 `torch_npu.npu_sparse_attn_sharedkv_metadata`，避免 D2H 数据传输开销。

---

## 集成方式

### 方式一：直接替换（推荐）

在 `sparse_attn_sharedkv_process.py` 中，直接使用 Python 实现：

```python
# 原代码（第147-166行）
metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
    num_heads_q=N1,
    num_heads_kv=N2,
    head_dim=D,
    cu_seqlens_q=cu_seqlens_q,
    ...
)

# 替换为
from sparse_attn_metadata_compute import compute_metadata_python

# 注意：cu_seqlens_q 和 seqused_kv 在 CPU 上创建，还未 .npu() 时计算
metadata = compute_metadata_python(
    num_heads_q=N1,
    num_heads_kv=N2,
    head_dim=D,
    cu_seqlens_q=tensor_input['cu_seqlens_q'],  # CPU tensor
    seqused_kv=tensor_input['seqused_kv'],      # CPU tensor
    batch_size=B,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_kv=ori_max_s2,
    cmp_topk=cmp_topk,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=ori_mask_mode,
    cmp_mask_mode=cmp_mask_mode,
    ori_win_left=ori_win_left,
    ori_win_right=ori_win_right,
    layout_q=layout_q,
    layout_kv=layout_kv,
    has_ori_kv=ori_k_in_pa_shape != None,
    has_cmp_kv=cmp_k_in_pa_shape != None,
    device="npu"  # 返回 NPU tensor
)
```

### 方式二：数据流优化

当前代码的数据流：
```
tensor_input['cu_seqlens_q'] (CPU) → cu_seqlens_q.npu() (NPU) → torch_npu.npu_sparse_attn_sharedkv_metadata (AICPU) → metadata (NPU)
```

优化后的数据流：
```
tensor_input['cu_seqlens_q'] (CPU) → compute_metadata_python (CPU) → metadata.to("npu") (NPU)
```

**关键点**：在 `cu_seqlens_q = cu_seqlens_q.npu()` 之前计算 metadata。

---

## 修改位置

### `sparse_attn_sharedkv_process.py` 修改点

#### 位置 1：SWA 模式（template_idx == 0）

**原代码**（第147-166行）：
```python
metadata = torch_npu.npu_sparse_attn_sharedkv_metadata(
    num_heads_q=N1,
    num_heads_kv=N2,
    head_dim=D,
    cu_seqlens_q=cu_seqlens_q,  # 已经是 NPU tensor
    ...
)
```

**修改后**：
```python
from sparse_attn_metadata_compute import compute_metadata_python

metadata = compute_metadata_python(
    num_heads_q=N1,
    num_heads_kv=N2,
    head_dim=D,
    cu_seqlens_q=tensor_input['cu_seqlens_q'],  # 使用 CPU tensor
    seqused_q=tensor_input['seqused_q'],        # 使用 CPU tensor
    seqused_kv=tensor_input['seqused_kv'],      # 使用 CPU tensor
    batch_size=B,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_kv=ori_max_s2,
    ori_mask_mode=ori_mask_mode,
    ori_win_left=ori_win_left,
    ori_win_right=ori_win_right,
    layout_q=layout_q,
    layout_kv=layout_kv,
    has_ori_kv=ori_k_in_pa_shape != None,
    has_cmp_kv=False,
    device="npu"
)
```

#### 位置 2：CFA 模式（template_idx == 1）

**原代码**（第182-203行）：

**修改后**：
```python
metadata = compute_metadata_python(
    num_heads_q=N1,
    num_heads_kv=N2,
    head_dim=D,
    cu_seqlens_q=tensor_input['cu_seqlens_q'],
    seqused_q=tensor_input['seqused_q'],
    seqused_kv=tensor_input['seqused_kv'],
    batch_size=B,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_kv=ori_max_s2,
    cmp_ratio=cmp_ratio,
    ori_mask_mode=ori_mask_mode,
    cmp_mask_mode=cmp_mask_mode,
    ori_win_left=ori_win_left,
    ori_win_right=ori_win_right,
    layout_q=layout_q,
    layout_kv=layout_kv,
    has_ori_kv=ori_k_in_pa_shape != None,
    has_cmp_kv=cmp_k_in_pa_shape != None,
    device="npu"
)
```

#### 位置 3：SCFA 模式（template_idx == 2）

同 CFA 模式，额外添加 `cmp_topk` 参数。

---

## 性能对比

| 方案 | 数据流 | 延迟估算 |
|------|--------|----------|
| **AICPU原始** | CPU → NPU → AICPU → NPU | ~10-50μs + D2H(~50-100μs) |
| **Python优化** | CPU → Python → NPU | ~20-100μs (计算+H2D) |

**收益**：避免了 D2H 传输开销（~50-100μs）。

---

## 参数对照表

| torch_npu 参数 | compute_metadata_python 参数 | 说明 |
|---------------|------------------------------|------|
| num_heads_q | num_heads_q | Q头数，固定64 |
| num_heads_kv | num_heads_kv | KV头数，固定1 |
| head_dim | head_dim | 头维度，512 |
| cu_seqlens_q | cu_seqlens_q | Q序列前缀和 |
| cu_seqlens_ori_kv | cu_seqlens_ori_kv | 暂不使用 |
| cu_seqlens_cmp_kv | cu_seqlens_cmp_kv | 暂不使用 |
| seqused_q | seqused_q | Q实际长度 |
| seqused_kv | seqused_kv | KV实际长度 |
| batch_size | batch_size | 批次大小 |
| max_seqlen_q | max_seqlen_q | Q最大序列长度 |
| max_seqlen_kv | max_seqlen_kv | KV最大序列长度 |
| cmp_topk | cmp_topk | SCFA模式稀疏token数 |
| cmp_ratio | cmp_ratio | CFA/SCFA压缩率 |
| ori_mask_mode | ori_mask_mode | 原始KV mask模式 |
| cmp_mask_mode | cmp_mask_mode | 压缩KV mask模式 |
| ori_win_left | ori_win_left | 滑动窗口左边界 |
| ori_win_right | ori_win_right | 滑动窗口右边界 |
| layout_q | layout_q | Q数据排布格式 |
| layout_kv | layout_kv | KV数据排布格式 |
| has_ori_kv | has_ori_kv | 是否有原始KV |
| has_cmp_kv | has_cmp_kv | 是否有压缩KV |
| device | device | **新增**：返回tensor的设备 |

---

## 注意事项

1. **数据时机**：必须在 `cu_seqlens_q.npu()` 之前调用，使用 CPU tensor
2. **设备参数**：`device="npu"` 时自动执行 H2D 传输
3. **返回类型**：`torch.Tensor [1024] int32`，与原接口一致
4. **三种模式**：
   - SWA: `cmp_ratio=None, cmp_topk=None`
   - CFA: `cmp_ratio有值, cmp_topk=None`
   - SCFA: `cmp_ratio有值, cmp_topk有值`

---

## 文件清单

- `sparse_attn_metadata_compute.py` - Python实现（NumPy优化）
- `test_metadata_compute.py` - 测试脚本
- `integration_guide.md` - 本文档