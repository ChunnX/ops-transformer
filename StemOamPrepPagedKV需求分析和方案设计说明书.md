# stem_oam_prep_paged_kv 需求分析和方案设计说明书

## 1 需求分析

### 1.1 需求背景

Stem OAM (Output-Aware Metric) 是大模型推理中的动态稀疏注意力机制，为 block-sparse-attention 的前置评分模块。Stem Prep Paged KV 算子负责从 Paged KV Cache 中提取 K/V 数据，经 K Processing (per-token K-scale × group sum + anti-diagonal flip) 和 V Processing (per-head vScale × L2 Norm → Log → Global Normalize → ReLU → Block Average) 计算，输出 kFlat 和 vBias 供 Stem OAM score computation 使用。

- **需求来源**: Stem 论文 https://arxiv.org/pdf/2603.06274
- **基线对齐**: 用户给定 KV Preparation 分阶段公式（见 1.2 基准分析）
- **论文公式**: Stem OAM block-sparse attention (block_size=128, stride=16)

| 项目 | 内容 |
|-----|------|
| 算子名称 | stem_oam_prep_paged_kv |
| 算子类别 | Reduction + Conversion (复合型) |
| 支持数据类型 | FP8_E4M3FN (输入) → BF16 (kFlat) + FP32 (vBias) |
| **目标芯片** | Ascend950DT / Ascend950PR |
| **目标架构** | arch35 (DAV_3510) |

**适用场景**: 融合算子场景（De-page Paged KV + Per-token K-scale × Group Sum + Anti-diag Flip + Per-head V-scale × L2 Norm + Global Normalize + Block Average 多步组合，仅 FP8 输入路径）

**模型结构分析**: Stem OAM attention 中 KV 侧预处理将 paged KV Cache 从分页存储格式转化为按 stem block（128 token）分组的 kFlat/vBias 向量，用于与 qFlat 计算稀疏 attention score。与 Q 侧预处理的关键区别在于 K 侧翻转 stride 维度（anti-diagonal flip），且 V 侧需要全局 Normalize（μ/σ over ALL k_down_len）。

**设计演进趋势**: 从标准 attention 的 K/V 直接参与 dot-product，演进为 block-sparse 场景下 KV 需先做 weighted group sum + global normalize 再参与 OAM score 计算。

### 1.2 基准分析

**基准接口**: test_stem_scale.py Triton kernel (KV Prep path, QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_PERHEAD_VPERHEAD)  

```python
def _gather_per_token_kscale_one_req(
    kscale_fp32, block_ids, kv_len, k_padded, kv_block_size,
):
    """Materialize per-token K-scale for one request: shape [k_padded, num_head_kv].

    Padding rows (>= kv_len) get zero.  All ops vectorized — O(k_padded) memory,
    no per-token Python loop.
    """
    device = kscale_fp32.device
    num_head_kv = kscale_fp32.shape[2]
    num_kv_blocks = block_ids.shape[0]

    token_ids = torch.arange(k_padded, device=device, dtype=torch.long)
    valid_mask = token_ids < kv_len
    page_idx = (token_ids // kv_block_size).clamp(max=num_kv_blocks - 1)
    row_in_page = token_ids % kv_block_size
    phys_blocks = block_ids[page_idx].to(torch.long)
    row_chunk = row_in_page // 32
    row_offset = row_in_page % 32
    # Advanced indexing: kscale_fp32[phys_block, row_chunk, :, row_offset]
    # → shape [k_padded, num_head_kv]
    gathered = kscale_fp32[phys_blocks, row_chunk, :, row_offset]
    gathered = torch.where(
        valid_mask.unsqueeze(-1), gathered, torch.zeros_like(gathered),
    )
    return gathered


def _prep_kv_reference(
    kcache_fp8: torch.Tensor,
    vcache_fp8: torch.Tensor,
    kscale_fp32: torch.Tensor,
    vscale: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    kv_block_size: int,
    lambda_mag: float = 0.3,
    stem_block_size: int = 128,
    stem_stride: int = 16,
):
    """Pure PyTorch reference matching CUDA stem_prep_paged_kv_qkpertoken_perhead_vperhead_kernel.

    K processing: per-KV-token K-scale folded into FP32 group sum (per-row scaling).
    V processing: per-head V-scale element-wise applied before L2 norm.
    Vbias path is shared with legacy (log normalize is invariant to per-head scale).

    Returns:
        kflat: [num_batch, num_head_kv, max_Kb, stem_stride * dim_qk] BF16
        v_bias: [num_batch, num_head_kv, max_Kb] FP32
    """
    device = kcache_fp8.device
    num_batch = kv_seq_lens.shape[0]
    num_head_kv = kcache_fp8.shape[2]
    dim_qk = kcache_fp8.shape[3]
    dim_v = vcache_fp8.shape[3]
    R = stem_block_size // stem_stride

    k_padded_lens = (
        (kv_seq_lens.to(torch.int64) + stem_block_size - 1)
        // stem_block_size * stem_block_size
    ).to(torch.int32)
    max_k_padded = k_padded_lens.max().item()
    max_Kb = max_k_padded // stem_block_size

    kflat_out = torch.zeros(
        num_batch, num_head_kv, max_Kb, stem_stride * dim_qk,
        dtype=torch.bfloat16, device=device,
    )
    v_bias_out = torch.zeros(
        num_batch, num_head_kv, max_Kb,
        dtype=torch.float32, device=device,
    )

    for b in range(num_batch):
        kv_len = kv_seq_lens[b].item()
        k_padded = k_padded_lens[b].item()
        num_stem_blocks = k_padded // stem_block_size
        k_down_len = k_padded // stem_stride

        # De-page K and V into dense FP32 (zero-padded beyond kv_len)
        num_kv_blocks = (kv_len + kv_block_size - 1) // kv_block_size
        block_ids = kv_indices[b, :num_kv_blocks]
        k_paged = kcache_fp8[block_ids].float()
        v_paged = vcache_fp8[block_ids].float()
        k_dense = torch.zeros(k_padded, num_head_kv, dim_qk, dtype=torch.float32, device=device)
        v_dense = torch.zeros(k_padded, num_head_kv, dim_v, dtype=torch.float32, device=device)
        actual_rows = min(kv_len, num_kv_blocks * kv_block_size)
        k_dense[:actual_rows] = k_paged.reshape(-1, num_head_kv, dim_qk)[:actual_rows]
        v_dense[:actual_rows] = v_paged.reshape(-1, num_head_kv, dim_v)[:actual_rows]

        # Per-token K-scale gather: [k_padded, num_head_kv]
        kscale_per_row = _gather_per_token_kscale_one_req(
            kscale_fp32, block_ids, kv_len, k_padded, kv_block_size,
        )

        # Apply per-token K-scale to K (broadcast on dim) before group sum
        k_scaled = (
            k_dense[:num_stem_blocks * stem_block_size]
            * kscale_per_row[:num_stem_blocks * stem_block_size].unsqueeze(-1)
        )

        for h in range(num_head_kv):
            # ---- Kflat ----
            k_h = k_scaled[:, h, :]  # [k_padded, dim_qk]
            # row index within stem block = r * stem_stride + g
            k_blocks = k_h.reshape(num_stem_blocks, R, stem_stride, dim_qk)
            k_group_sum = k_blocks.sum(dim=1)  # [num_stem_blocks, stride, dim_qk]
            k_group_rev = k_group_sum.flip(1)
            kflat_out[b, h, :num_stem_blocks] = k_group_rev.reshape(
                num_stem_blocks, stem_stride * dim_qk,
            ).to(torch.bfloat16)

            # ---- V L2 norm with per-head V-scale ----
            vscale_h = vscale[h].item()
            v_h = v_dense[:num_stem_blocks * stem_block_size, h, :] * vscale_h
            v_rows = v_h.reshape(k_down_len, stem_stride, dim_v)
            norms = torch.sqrt((v_rows ** 2).sum(dim=-1))  # [k_down_len, stride]
            row_ids = torch.arange(
                num_stem_blocks * stem_block_size, device=device,
            ).reshape(k_down_len, stem_stride)
            norms = torch.where(row_ids < kv_len, norms, torch.zeros_like(norms))
            v_norm_down = norms.max(dim=-1).values

            # ---- v_bias (FP32 throughout, matches kernel + scale-invariant)
            if k_down_len > 0:
                log_vals = torch.log(v_norm_down + 1e-6)
                v_mean = log_vals.mean()
                if k_down_len > 1:
                    v_std = log_vals.std(unbiased=True)
                else:
                    v_std = torch.tensor(0.0, device=device)
                inv_std = 1.0 / (v_std + 1e-6)
                normalized = (log_vals - v_mean) * inv_std
                v_final = lambda_mag * torch.relu(normalized)
                v_final_blocks = v_final[:num_stem_blocks * R].reshape(num_stem_blocks, R)
                v_bias_out[b, h, :num_stem_blocks] = v_final_blocks.mean(dim=1)

    return kflat_out, v_bias_out

```
**功能逻辑**: paged KV de-page → per-token kScale × group sum (anti-diag flip) → kFlat; per-head vScale × L2 norm → max → log → global normalize → ReLU → block average → vBias
#### 1.2.1 KV Cache 布局

kCache/vCache 支持两种 layout，由 `kvLayout` 属性指定(当前迭代可以仅支持layout B):

| Layout | kvLayout | Shape | 说明 |
|--------|----------|-------|------|
| **Layout A** (interleaved) | 0 | `[total_blocks, kv_block_size, H_kv, D]` | rows from different heads interleaved within a block |
| **Layout B** (contiguous) | 1 | `[total_blocks, H_kv, kv_block_size, D]` | all rows for a head contiguous within a block |

`kv_block_size` ∈ {64, 128} , `D = 128` (当前仅支持128).

**kScaleCache layout**: 随 kvLayout 变化, 与 kCache/vCache 布局一致 (仅 D=1 代替 D=128):
- Layout A: `[total_blocks, kv_block_size, H_kv, 1]`
- Layout B: `[total_blocks, H_kv, kv_block_size, 1]`

**De-page 偏移计算**:  
  

这里需要支持kvcache是非连续的，即使用stride来计算不同headDim的位置（最后一维HeadDim是连续的）（对于layoutB来说，除了total_blocks是非连续的 其他轴是连续的，这种情况可以提升性能）
| Layout | kcache/vcache offset (head=h, row=r, block=phys) |
|--------|------|
| Layout A | `phys × kv_block_size × H_kv × D + r × H_kv × D + h × D` |
| Layout B | `phys × H_kv × kv_block_size × D + h × kv_block_size × D + r × D` |

**kScaleCache 偏移计算** (布局一致, D=1):

同kvcache，需要支持非连续场景（场景和kvcache一样） 
| Layout | kScaleCache offset |
|--------|------|
| Layout A | `kScaleCache[phys, row, h, 0]` → `phys × kv_block_size × H_kv + row × H_kv + h` |
| Layout B | `kScaleCache[phys, h, row, 0]` → `phys × H_kv × kv_block_size + h × kv_block_size + row` |

#### 1.2.2 Scale 布局

| Scale | Layout A Shape | Layout B Shape | Dtype | 语义 |
|-------|---------------|---------------|-------|------|
| kScaleCache | `[total_blocks, kv_block_size, H_kv, 1]` | `[total_blocks, H_kv, kv_block_size, 1]` | FP32 | per-token per-head (与 kCache 同布局, D=1) |
| vScale | `[H_kv]` | `[H_kv]` | FP32 | per-head |

kScaleCache 索引:
- Layout A: `kScaleCache[physical_block, row_in_block, head_idx, 0]`
- Layout B: `kScaleCache[physical_block, head_idx, row_in_block, 0]`
#### 1.2.3 **数学公式**:

**阶段 0: 输入参数**

| 参数 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| kCache | `[total_blocks, kv_block_size, H_kv, D=128]` 或 `[total_blocks, H_kv, kv_block_size, D=128]` | FP8_e4m3fn | Paged K cache (两布局) |
| vCache | 同 kCache shape/dtype/layout | FP8_e4m3fn | Paged V cache |
| kvIndices | `[batch, max_kv_blocks]` | INT32 | Block index 数组 |
| kvSeqLens | `[batch]` | INT32 | 每 batch KV 序列长度 |
| kScaleCache | Layout A `[total_blocks, kvBlockSize, H_kv, 1]` / Layout B `[total_blocks, H_kv, kvBlockSize, 1]` | FP32 | Per-token per-head K scale (必填, 随 kvLayout 变化) |
| vScale | `[H_kv]` | FP32 | Per-head V scale (必填) |
| lambdaMag | ATTR | FP32 | 默认 0.3 (必填) |
| kvLayout | ATTR | INT64 | 0=Layout A (interleaved), 1=Layout B (contiguous) |
| kvBlockSize | ATTR | INT64 | 64 或 128 |
| stemBlockSize (B) | ATTR | INT64 | %32==0, ≤256 |
| stemStride (S) | ATTR | INT64 | %16==0, ≤64, S≤B |

**典型参数**:  
total_tokens = 1024k  
batch = 8 
H_kv = 2   
stemBlockSize = 128  
stemStride = 16  
max_seqlen = 256k  
kv_block_size = 128  
total_blocks = 9649
max_kv_blocks = 2k(最大)

**阶段 1: De-page + Cast FP32 + kScaleCache Gather**

$$
K\_dense[b], V\_dense[b], kscale\_per\_row[b] = Cast\big(gather(kcache[kv\_indices[b]]),\ FP32\big),\ Cast\big(gather(vcache[kv\_indices[b]]),\ FP32\big),\ gather(kScaleCache[kv\_indices[b]])
$$

以上三个 gather 使用相同的 `kv_indices[b]` 和相同的 layout-aware 偏移计算（kScaleCache 仅 D=1 替代 D=128）。

$$
k\_padded[b] = \lceil kv\_len[b] / B \rceil \times B, \quad num\_Kb[b] = k\_padded[b] / B, \quad R = B / S
$$

$$
K\_dense, V\_dense, kscale\_per\_row: \text{zero-pad beyond } kv\_len[b]
$$

$$
K\_scaled[b, h] = K\_dense[b, h, :] \times kscale\_per\_row[b, h, :] \quad (\text{per-token per-head, broadcast on dim})
$$

**数据节点 (阶段 1)**:

| 节点 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| K_dense | [kv_len, H_kv, D] | FP32 | De-page K + Cast, zero-pad to k_padded |
| V_dense | [kv_len, H_kv, D] | FP32 | De-page V + Cast, zero-pad to k_padded |
| kscale_per_row | [kv_len, H_kv] | FP32 | De-page kScaleCache via kv_indices (与 kCache 同逻辑, D=1) |
| K_scaled | [k_padded, H_kv, D] | FP32 | K_dense × kscale_per_row, per-token per-head broadcast |

**阶段 2: K Processing (Weighted Group Sum + Anti-diagonal Flip)**

$$
K\_blocks[b,h] = reshape\big(K\_scaled[:num\_Kb \times B, h, :],\ [num\_Kb,\ R,\ S,\ D]\big)
$$

$$
K\_group\_sum[b,h,kb,g,:] = \sum_{r=0}^{R-1} K\_blocks[b,h,kb,r,g,:]
$$

$$
K\_group\_rev[b,h,kb,g',:] = K\_group\_sum[b,h,kb,\ S-1-g',:] \quad (\text{anti-diagonal flip})
$$

$$
kFlat[b,h,kb,:] = reshape(K\_group\_rev[b,h,kb,:,:],\ [kflat\_dim = S \times D]) \xrightarrow{Cast} BF16
$$

**数据节点 (阶段 2)**:

| 节点 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| K_blocks | [num_Kb, R, S, D] | FP32 | reshape K_scaled |
| K_group_sum | [num_Kb, S, D] | FP32 | Σ over R, per (kb,g) |
| K_group_rev | [num_Kb, S, D] | FP32 | anti-diag flip: g'=S-1-g |
| kFlat | [batch, H_kv, max_Kb, S×D] | BF16 | Cast FP32→BF16, GM output |

**阶段 3: V Processing (Per-head vScale + Global Normalize → vBias)**

**Step 3a: L2 Norm + Max Pool + Log**（R=8,S=16）

$$
V\_rows[b,h] = reshape(V\_dense[:k\_down\_len \times S, h, :],\ [k\_down\_len,\ S,\ D]), \quad k\_down\_len = num\_Kb \times R
$$

$$
norms[b,h,idx,s,:] = \| V\_rows[b,h,idx,s,:] \times vScale[h] \|_2 =  \sqrt{\sum_d (V\_rows[idx,s,d] \times vScale[h])^2}
$$

$$
v\_norm\_down[b,h,idx] = \max_{s} norms[b,h,idx,s,:] \quad (\text{zero beyond } kv\_len)
$$

$$
log\_vals[b,h,idx] = \log(v\_norm\_down[b,h,idx] + \epsilon), \quad \epsilon = 10^{-6}
$$

**Step 3b: Global Normalize (μ/σ over ALL k_down_len values)**

$$
\mu[b,h] = \frac{1}{k\_down\_len} \sum_{idx=0}^{k\_down\_len-1} log\_vals[b,h,idx]
$$

$$
\sigma[b,h] = \sqrt{\frac{\sum(log\_vals - \mu)^2}{k\_down\_len - 1}} \quad (k\_down\_len > 1);\quad \sigma = 0 \quad (k\_down\_len = 1)
$$

**Step 3c: Normalize + ReLU + Block Average**

$$
normalized[b,h,idx] = \frac{log\_vals[b,h,idx] - \mu[b,h]}{\sigma[b,h] + \epsilon}
$$

$$
v\_final[b,h,idx] = \lambda \times ReLU(normalized[b,h,idx])
$$

$$
vBias[b,h,kb] = \frac{1}{R} \sum_{r=0}^{R-1} v\_final[b,h,\ kb \times R + r]
$$

**数据节点 (阶段 3)**:

| 节点 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| V_rows | [k_down_len, S, D] | FP32 | reshape V_dense, k_down_len=num_Kb×R |
| norms | [k_down_len, S] | FP32 | L2 norm per (idx,s), ‖V_rows×vScale[h]‖₂ |
| v_norm_down | [k_down_len] | FP32 | max over S per idx, zero beyond kv_len |
| log_vals | [k_down_len] | FP32 | log(v_norm_down+ε) |
| μ | scalar | FP32 | mean(log_vals) over ALL k_down_len |
| σ | scalar | FP32 | std(log_vals), σ=0 if k_down_len=1 |
| inv_std | scalar | FP32 | 1/(σ+ε) |
| normalized | [k_down_len] | FP32 | (log_vals-μ)/inv_std |
| v_final | [k_down_len] | FP32 | λ×ReLU(normalized) |
| vBias | [batch, H_kv, max_Kb] | FP32 | mean(v_final,R) per kb, GM output |

**阶段 4: 输出**

| 输出 | Shape | Dtype | 说明 |
|------|-------|-------|------|
| kFlat | `[batch, H_kv, max_Kb, S×D]` | BF16 | K per-token-scaled group sum + flip |
| vBias | `[batch, H_kv, max_Kb]` | FP32 | V per-head-scaled block bias |

### 1.3 竞争分析

#### 1.3.1 接口

不涉及

#### 1.3.2 实现分析

不涉及

#### 1.3.3 优劣势

不涉及

### 1.4 需求分析总结

#### 1.4.1 aclnn API定义

每个算子分为两段式接口，必须先调用"aclnnStemOamPrepPagedKVGetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnStemOamPrepPagedKV"接口执行计算。

**函数原型**:

```cpp
aclnnStatus aclnnStemOamPrepPagedKVGetWorkspaceSize(
  const aclTensor  *kCache,
  const aclTensor  *vCache,
  const aclTensor  *kvIndices,
  const aclTensor  *kvSeqLens,
  const aclTensor  *kScaleCache,
  const aclTensor  *vScale,
  double             lambdaMag,
  int64_t            kvLayout,
  int64_t            kvBlockSize,
  int64_t            stemBlockSize,
  int64_t            stemStride,
  aclTensor         *kFlat,
  aclTensor         *vBias,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnStemOamPrepPagedKV(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

**参数说明**:

| 参数名 | 输入/输出 | 描述 | 数据类型 | 维度(shape) |
|-------|---------|------|---------|-----------|
| kCache | 输入 | Paged K cache, 两布局 | FLOAT8_E4M3FN | Layout A: [total_blocks, kvBlockSize, H_kv, 128]; Layout B: [total_blocks, H_kv, kvBlockSize, 128] |
| vCache | 输入 | Paged V cache, shape/layout 与 kCache 一致 | FLOAT8_E4M3FN | 同 kCache |
| kvIndices | 输入 | Block index 数组 | INT32 | [batch, max_kv_blocks] |
| kvSeqLens | 输入 | 每 batch KV 序列长度 | INT32 | [batch] |
| kScaleCache | 输入(OPTIONAL→FP8时REQUIRED) | Per-token per-head K scale, 随 kvLayout 变化 | FLOAT | Layout A: [total_blocks, kvBlockSize, H_kv, 1]; Layout B: [total_blocks, H_kv, kvBlockSize, 1] |
| vScale | 输入(OPTIONAL→FP8时REQUIRED) | Per-head V scale | FLOAT | [H_kv] |
| lambdaMag | ATTR | 默认 0.3, 必填 | DOUBLE | scalar |
| cacheLayout | ATTR | 0=Layout A (interleaved), 1=Layout B (contiguous) | INT64 | scalar |
| kvBlockSize | ATTR | 64 或 128 | INT64 | scalar |
| stemBlockSize | ATTR | stem block 大小, 推荐 128 | INT64 | scalar |
| stemStride | ATTR | stem stride 大小, 推荐 16 | INT64 | scalar |
| kFlat | 输出 | K group sum + flip | BFLOAT16 | [batch, H_kv, max_Kb, kflat_dim] |
| vBias | 输出 | V block bias | FLOAT | [batch, H_kv, max_Kb] |

**返回值错误码**:

| 返回值 | 错误码 | 描述 |
|-------|------|------|
| ACLNN_ERR_PARAM_NULLPTR | 161001 | kCache、vCache、kFlat、vBias 存在空指针。kScaleCache、vScale、lambdaMag 在 FP8 输入时不允许空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | kCache dtype 不为 FP8_E4M3FN；kScaleCache dtype 不为 FLOAT；kScaleCache shape 不随 kvLayout 变化；vScale 不为 1D [H_kv]；kvLayout/kvBlockSize/stemBlockSize/stemStride 不满足约束 |
| ACLNN_ERR_INNER | 361001 | 内部错误，如 Tiling 计算失败或 Kernel 查找失败 |

**约束说明**:
- 确定性说明: 模式0 默认确定性（单核内 FP32 累加）；模式1 聚合顺序由 SyncAll 保证一致性,待确认
- 仅支持 arch35 架构（Ascend 950PR/Ascend 950DT、Atlas A3 系列），不支持 arch22
- kCache/vCache dtype 必须为 FP8_E4M3FN（仅 FP8 路径）
- kScaleCache/vScale 必填, 不允许 nullptr
- kScaleCache shape 随 cacheLayout 变化: Layout A `[total_blocks, kvBlockSize, H_kv, 1]`, Layout B `[total_blocks, H_kv, kvBlockSize, 1]`
- vScale shape: `[H_kv]`
- kvBlockSize ∈ {64, 128}
- stemBlockSize % 32 == 0, ≤256; stemStride % 16 == 0, ≤64, S≤B
- 派生值: R = stemBlockSize / stemStride, kflat_dim = stemStride × 128, k_down_len = num_Kb × R
- 边界: kvSeqLens[b]=0 时输出全零; kScaleCache padding rows → zero

#### 1.4.2 图模式IR

| 类别 | 名称 | 类型 | 类型范围 | 默认值 | Shape | 说明 |
|------|------|------|---------|--------|-------|------|
| INPUT | kCache | Tensor | FP8_E4M3FN | NA | [total_blocks, kvBlockSize, H_kv, 128] 或 [total_blocks, H_kv, kvBlockSize, 128] | Paged K cache |
| INPUT | vCache | Tensor | FP8_E4M3FN | NA | 同 kCache | Paged V cache |
| INPUT | kvIndices | Tensor | INT32 | NA | [batch, max_kv_blocks] | Block index |
| INPUT | kvSeqLens | Tensor | INT32 | NA | [batch] | KV 序列长度 |
| OPTIONAL_INPUT | kScaleCache | Tensor | FP32 | NA | 随 kvLayout 变化 | Per-token per-head K scale, FP8必填 |
| OPTIONAL_INPUT | vScale | Tensor | FP32 | NA | [H_kv] | Per-head V scale, FP8必填 |
| ATTR | lambdaMag | DOUBLE | {0~1} | NA | NA | 默认 0.3 |
| ATTR | cacheLayout | INT64 | {0,1} | 0 | NA | 0=Layout A, 1=Layout B |
| ATTR | kvBlockSize | INT64 | {64,128} | 64 | NA | Paged KV block size |
| ATTR | stemBlockSize | INT64 | {32,...,256} | 128 | NA | Stem block 大小 |
| ATTR | stemStride | INT64 | {16,...,64} | 16 | NA | Stride 大小 |
| OUTPUT | kFlat | Tensor | BF16 | NA | [batch, H_kv, max_Kb, kflat_dim] | K group sum + flip |
| OUTPUT | vBias | Tensor | FP32 | NA | [batch, H_kv, max_Kb] | V block bias |

### 1.5 约束与要求



**精度要求**:
- FP8 → FP32 → BF16: rtol=1e-3, atol=1e-3

**边界情况处理**:
- kvSeqLens 中存在 0 值 batch 时，该 batch 对应的 kFlat/vBias 填零
- kScaleCache padding rows (beyond kv_len) → zero

**确定性计算**: 模式0 每个 (batch, head) pair 由单一 AI Core 处理，FP32 累加在单核内完成，无并行累加歧义。模式1 聚合由 local_id==0 的 core 完成，SyncAll 保证数据一致性。

---

## 2 方案设计

### 2.1 API实现设计

**OpDef 注册调用方式**:
- 本算子采用标准 OpDef 注册调用方式，通过 aclnn 框架调度 kernel
- 不使用 <<<>>> 内核调用语法
- 调用流程:
  1. aclnnStemOamPrepPagedKVGetWorkspaceSize() — 计算 workspace 大小
  2. aclnnStemOamPrepPagedKV() — 执行 kernel

**OpDef 定义** (`op_host/stem_oam_prep_paged_kv_def.cpp`):

```cpp
namespace ops {
class StemOamPrepPagedKV : public OpDef {
 public:
  explicit StemOamPrepPagedKV(const char* name) : OpDef(name) {
    this->Input("kCache")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT8_E4M3FN})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("vCache")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT8_E4M3FN})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("kvIndices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND});
    this->Input("kvSeqLens")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32})
        .Format({ge::FORMAT_ND});
    this->Input("kScaleCache")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("vScale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("kFlat")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BFLOAT16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("vBias")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("lambdaMag")
        .AttrType(OPTIONAL)
        .Float(0.3);
    this->Attr("cacheLayout")
        .AttrType(OPTIONAL)
        .Int(0);
    this->Attr("kvBlockSize")
        .AttrType(OPTIONAL)
        .Int(64);
    this->Attr("stemBlockSize")
        .AttrType(OPTIONAL)
        .Int(128);
    this->Attr("stemStride")
        .AttrType(OPTIONAL)
        .Int(16);
    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(false)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("opFile.value", "stem_oam_prep_paged_kv_apt");
    this->AICore().AddConfig("ascend950", aicore_config);
  }
};
OP_ADD(StemOamPrepPagedKV);
}  // namespace ops
```

**REG_OP 定义**:

```cpp
REG_OP(StemOamPrepPagedKV)
    .INPUT(kCache, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(vCache, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(kvIndices, TensorType::IndexNumberType())
    .INPUT(kvSeqLens, TensorType::IndexNumberType())
    .INPUT(kScaleCache, TensorType({DT_FLOAT}))
    .INPUT(vScale, TensorType({DT_FLOAT}))
    .ATTR(lambdaMag, Float, 0))
    .ATTR(cacheLayout, Int, 0)
    .ATTR(kvBlockSize, Int, 64)
    .ATTR(stemBlockSize, Int, 128)
    .ATTR(stemStride, Int, 16)
    .OUTPUT(kFlat, TensorType({DT_BFLOAT16}))
    .OUTPUT(vBias, TensorType({DT_FLOAT}))
    .END();
```

**模块职责表**:

| 模块 | 聣责 | 核心文件 |
|------|------|---------|
| **op_host** | 算子定义(OpDef) + Tiling 计算 + Shape 推导 | `_def.cpp`, `_infershape.cpp`, `arch35/_tiling.cpp`, `arch35/_tiling.h` |
| **op_kernel** | Kernel 实现 + TilingKey + TilingData | `arch35/.h`, `arch35/_tiling_key.h`, `arch35/_tiling_data.h`, `_arch35.cpp` |
| **op_api** | aclnn 接口封装 (GetWorkspaceSize + Execute) | `aclnn_stem_oam_prep_paged_kv.h`, `aclnn_stem_oam_prep_paged_kv.cpp`, `stem_oam_prep_paged_kv.h`, `stem_oam_prep_paged_kv.cpp` |

**模块依赖**:

```
op_api (ACLNN接口层)
├── aclnnStemOamPrepPagedKVGetWorkspaceSize()
│     ├──▶ op_host/_def.cpp: 获取算子定义
│     ├──▶ op_host/arch35/_tiling.cpp: 计算 TilingData (dtype→tilingKey=0, batch×H_kv→split_mode)
│     └──▶ 返回 workspaceSize
└── aclnnStemOamPrepPagedKV()
      ├──▶ 加载 Kernel 二进制 (tilingKey=0 fixed)
      ├──▶ 传递 TilingData 到 Device
      └──▶ NPU Device 执行
```

**执行流程**:

```
aclnnStemOamPrepPagedKVGetWorkspaceSize()
     │
     ├──▶ op_host/_def.cpp: 获取算子定义
     ├──▶ op_host/_tiling.cpp: 计算 TilingData (tilingKey=0, split_mode判定, cores_per_bh)
     │
     ▼
aclnnStemOamPrepPagedKV()
     │
     ├──▶ 加载 Kernel 二进制
     ├──▶ 传递 TilingData 到 Device
     │
     ▼
NPU Device 执行 (arch35 only)
     ├──▶ REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT
     ├──▶ GetBlockIdx() → split_mode 分支:
     │     ├── 模式0: task_id → (batch, head) 全 kb
     │     ├── 模式1: task_id → (group_id, local_id) → kb_range
     │     └── 模式2: flat_idx → (BatchIdx, HeadNumIdx, stemBlockIdx, stemBlockNum)
     ├──▶ GM → UB: De-page K/V + scales
     │     ├── 模式0/1: 逐行 DataCopyPad
     │     └── 模式2: kv_block级 bulk gather (stemBlockSize/kvBlockSize 个 kv_block)
     ├──▶ K: Cast → ×kScaleCache → Group Sum (anti-diag flip) → Cast BF16
     ├──▶ V: ×vScale → L2 Norm → Max → Log → [模式0: 单核μ/σ | 模式1/2: SyncAll聚合]
     ├──▶ Normalize → ReLU → ×λ → Block Average → vBias
     └──▶ UB → GM: kFlat, vBias
```

**InferShape 逻辑**:

```
if kvLayout == 0: H_kv = kCache.dim(2)
if kvLayout == 1: H_kv = kCache.dim(1)

max_kv_len = max(kvSeqLens)
max_Kb = ceil(max_kv_len / stemBlockSize)
kflat_dim = stemStride × 128
vbias_stride = (max_Kb < 8) ? 8 : max_Kb

kFlat shape = [batch, H_kv, max_Kb, kflat_dim]  // BF16
vBias shape = [batch, H_kv, vbias_stride]  // FP32 (padded, output slices to max_Kb)
```

### 2.2 图模式设计

**图模式算子原型定义**:

OpProto 定义文件: `op_graph/stem_oam_prep_paged_kv_proto.h`

```cpp
#ifndef OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_PAGED_KV_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_STEM_OAM_PREP_PAGED_KV_PROTO_H_
#include "graph/operator_reg.h"

namespace ge {
REG_OP(StemOamPrepPagedKV)
    .INPUT(kCache, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(vCache, TensorType({DT_FLOAT8_E4M3FN}))
    .INPUT(kvIndices, TensorType::IndexNumberType())
    .INPUT(kvSeqLens, TensorType::IndexNumberType())
    .INPUT(kScaleCache, TensorType({DT_FLOAT}))
    .INPUT(vScale, TensorType({DT_FLOAT}))
    .ATTR(lambdaMag, Float, 0.3)
    .ATTR(cacheLayout, Int, 0)
    .ATTR(kvBlockSize, Int, 64)
    .ATTR(stemBlockSize, Int, 128)
    .ATTR(stemStride, Int, 16)
    .OUTPUT(kFlat, TensorType({DT_BFLOAT16}))
    .OUTPUT(vBias, TensorType({DT_FLOAT}))
    .END();
}  // namespace ge
#endif
```

**InferDtype 逻辑**:

```
kFlat dtype = BF16 (FP8 input)
vBias dtype = FP32 (always)
```

### 2.3 总体规格设计


**TilingKey 机制**: 单路径 FP8, tilingKey = 0

**模板参数定义**: 无模板参数（仅 FP8 路径，非模板类）

**模板划分表**:

| 模板 | tilingKey | split_mode | 触发条件 | 适用场景 |
|-----|----------|-----------|---------|---------|
| FP8 模式0 | 0 | 0 | batch×H_kv ≥ AICore数 | 1核/(batch,head)，零SyncAll |
| FP8 模式1 | 0 | 1 | batch×H_kv < AICore数 | kb_range分核(group/local)，2SyncAll |
| FP8 模式2 (Template 3) | 0 | 2 | stemBlockSize ≥ kvBlockSize 且 total_stem_blocks > 0 | stem_block级分核 + bulk kv_block gather，显式per-core参数 |

**FP8 路径三模式统一框架**: tilingKey=0 固定，split_mode ∈ {0,1,2} 由 Host Tiling 根据运行时 shape 和 ATTR 判定。模式0/1 为行级 De-page，模式2 为 kv_block 级 bulk gather。

**分支差异对比** (仅 FP8 路径):

| 操作 | FP8 路径 |
|------|---------|
| De-page K/V | DataCopyPad\<fp8_e4m3fn_t\> (layout-aware offset) |
| Cast K/V→FP32 | Cast\<fp8_e4m3fn_t, float\> |
| K-scale | Muls(denseRow × kScaleCache_val) per-token |
| V-scale | Muls(denseRow × vScale[h]) per-head |
| Group Sum | Add\<float\> (anti-diag flip: offset=S-1-g) |
| Cast 输出 | Cast\<float, bfloat16_t\> (kFlat) |

**分核策略**: 数据依赖驱动的双模式分核

**数据依赖分析**:

| 操作 | 最小独立粒度 | 跨粒度依赖 |
|------|-------------|-----------|
| DepageRow(K) | (b,h, row) | **无** — 每 K 行独立从 GM 读取 |
| DepageRow(V) | (b,h, row) | **无** — 每 V 行独立从 GM 读取 |
| Cast+×kScaleCache | (b,h, row) | **无** — 单行 Muls |
| Group Sum 累加 | (b,h,kb) | **无** — 累加仅当前 kb 内 R×S 行 |
| kFlat write | (b,h,kb) | **无** — GM 写入独立偏移 |
| V L2 Norm | (b,h, row) | **无** — 单行 Mul+ReduceSum+Sqrt |
| V Max Pool | (b,h,kb,idx) | **无** — 单 idx 内 ReduceMax(S_) |
| V Log | (b,h,idx) | **无** — 单值 Log |
| **V Global Normalize (μ/σ)** | **(b,h, ALL k_down_len)** | **有** — μ/σ 需全部 k_down_len 的 Σ(log)/Σ(log²) |
| V Normalize+ReLU+×λ | (b,h,idx) | **依赖 μ/σ** — 必须先获得 μ/σ |
| vBias write | (b,h,kb) | **依赖 μ/σ** — 必须先获得 μ/σ |

**结论**: K Processing 的每个 (b,h,kb) 完全独立。V Processing 的 L2 Norm/Max/Log 各行独立，但 **Global Normalize (μ/σ) 依赖全部 k_down_len**，这与 Q Prep 的零依赖形成鲜明对比。因此 KV Prep 模式1 **需要 2 SyncAll**。

**模式 0 (batch×H_kv ≥ AICore数)**: 1 核/(batch,head)，零 SyncAll

- task_id = GetBlockIdx()
- batch_idx = task_id / num_kv_heads, head_idx = task_id % num_kv_heads
- total_tasks = batch_size × num_kv_heads
- 每个 (b,h) 核处理完整 kb 0..num_Kb-1 序列
- μ/σ 在单核内计算，无需同步
- 超出 total_tasks 的核直接 return（空核）

**模式 1 (batch×H_kv < AICore数)**: 按 (batch,head,kb_range) 分核，2 SyncAll

- 每个 (b,h) 分配 cores_per_bh 个核
- K Processing: 各核独立处理 kb_range，零依赖
- V Phase A: 各核独立累积局部 sumLog/sumLogSq → GM 写入
- SyncAll #1: 等所有核完成局部统计写入
- V Phase B (聚合): 仅 local_id==0 的核读取所有核的局部统计量 → 计算 μ/σ → GM 写入
- SyncAll #2: 等指定核完成 μ/inv_std 写入
- V Phase C: 各核从 GM 读 μ/inv_std → Normalize+ReLU+×λ → Block Average → vBias GM 写

**模式判定 (tiling.cpp)**:

```
AICore数 = GetCoreNumAiv()  // Ascend950DT ≈ 40

total_bh = batch × H_kv

if total_bh >= AICore数:
    split_mode = 0
    cores_per_bh = 1
    total_tasks = total_bh
    used_core_num = min(total_bh, AICore数)
else:
    split_mode = 1
    cores_per_bh = AICore数 / total_bh   // 均匀分配
    total_tasks = total_bh × cores_per_bh
    used_core_num = AICore数
```

**模式 0 分核示意 (batch=8, H_kv=8, AICore=40)**:

```
total_bh = 64 ≥ 40 → 模式0

┌─────────────────────────────────────────────────────────────────────┐
│                    模式 0: AI Core 分核示意图                        │
│                                                                     │
│  task_id = 0        task_id = 1        ...   task_id = 39           │
│  ┌──────────┐      ┌──────────┐             ┌──────────┐           │
│  │ (b=0,h=0)│      │ (b=0,h=1)│             │ (b=4,h=7)│           │
│  │ 全 kb    │      │ 全 kb    │             │ 全 kb    │           │
│  │ sumLog   │      │ sumLog   │             │ sumLog   │           │
│  │ μ/σ本核  │      │ μ/σ本核  │             │ μ/σ本核  │           │
│  └──────────┘      └──────────┘             └──────────┘           │
│                                                                     │
│  task_id 40..63 → 闲置 (≥total_tasks → return)                     │
│  batch_idx = task_id / H_kv                                        │
│  head_idx  = task_id % H_kv                                        │
│  无 SyncAll, μ/σ 在单核内计算                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**模式 1 分核示意 (batch=1, H_kv=4, AICore=40)**:

```
total_bh = 4 << 40 → 模式1
cores_per_bh = 40/4 = 10, total_tasks = 40

┌─ Group 0 (b=0, h=0): 10 cores ──────────────────────────────────┐
│ Core 0:  kb[0..7]    Core 1:  kb[8..15]   Core 2:  kb[16..23] │
│ Core 3:  kb[24..31]  Core 4:  kb[32..39]  Core 5:  kb[40..47] │
│ Core 6:  kb[48..55]  Core 7:  kb[56..63]  Core 8:  kb[64..71] │
│ Core 9:  kb[72..79]                                             │
├─ Group 1 (b=0, h=1): 10 cores ──────────────────────────────────┤
│ Core 10: kb[0..7]    ...       Core 19: kb[72..79]             │
├─ Group 2 (b=0, h=2): 10 cores ──────────────────────────────────┤
│ Core 20: kb[0..7]    ...       Core 29: kb[72..79]             │
├─ Group 3 (b=0, h=3): 10 cores ──────────────────────────────────┤
│ Core 30: kb[0..7]    ...       Core 39: kb[72..79]             │
└─────────────────────────────────────────────────────────────────┘
```

**模式 1 分核示意 (batch=1, H_kv=1, AICore=40)**:

```
total_bh = 1 << 40 → 模式1
cores_per_bh = 40/1 = 40, total_tasks = 40

┌─ Group 0 (b=0, h=0): 全部 40 cores ─────────────────────────────┐
│ Core 0:  kb[0..1]    Core 1:  kb[2..3]    ...                   │
│ Core 39: kb[78..79]                                             │
│                                                                   │
│ 40 核并行处理同一 (b,h) 的 kb → 2 SyncAll 聚合 μ/σ              │
│ 各核仅处理 ~2 kb, 负载极均衡                                     │
└─────────────────────────────────────────────────────────────────┘
```

**模式 2 (Template 3): stem_block 级分核 + bulk kv_block gather**

- **核心思想**: 将所有 (b,h) 的 stem_block 视为 flat 工作池，按 stem_block 总个数均匀分核
- 每个 core 接收显式 per-core 参数: BatchIdx, HeadNumIdx, stemBlockIdx, stemBlockNum
- 不使用 group_id/local_id 结构，内核逻辑更简洁
- De-page 阶段采用 kv_block 级 bulk gather: 每个 stem_block 一次搬运 `stemBlockSize / kvBlockSize` 个 kv_block
- V Processing Global Normalize 依赖 μ/σ: 同一 (b,h) 的 stem_block 可能跨多个 core → 需要 2 SyncAll

**前提条件**: stemBlockSize ≥ kvBlockSize (保证 stem_block 包含整数个 kv_block)

**分核参数计算**:

```
kv_blocks_per_stem_block = stemBlockSize / kvBlockSize   // e.g. 128/64=2
total_stem_blocks = Σ(b,h) num_Kb[b,h]                   // 实际 stem_block 总数
  简化: total_stem_blocks = batch × H_kv × max_Kb         // 用 max_Kb 均匀化

N = GetCoreNumAiv()
stem_blocks_per_core = ceil(total_stem_blocks / N)

每个 core_id 的 flat 范围:
  start_flat = core_id × stem_blocks_per_core
  end_flat   = min(start_flat + stem_blocks_per_core, total_stem_blocks)

flat_idx → (BatchIdx, HeadNumIdx, stemBlockIdx) 映射:
  BatchIdx      = flat_idx / (H_kv × max_Kb)
  HeadNumIdx    = (flat_idx % (H_kv × max_Kb)) / max_Kb
  stemBlockIdx  = flat_idx % max_Kb
```

**模式2 分核示意 (batch=2, H_kv=4, max_Kb=10, AICore=40, stemBlockSize=128, kvBlockSize=64)**:

```
total_stem_blocks = 2 × 4 × 10 = 80
stem_blocks_per_core = ceil(80/40) = 2
kv_blocks_per_stem_block = 128/64 = 2  → 每个 stem_block 搬运 2 个 kv_block

┌─ Core 0 ─────────────────────────────────────────────────────────┐
│ flat[0,1] → (b=0,h=0,kb=0), (b=0,h=0,kb=1)                     │
│ BatchIdx=0, HeadNumIdx=0, stemBlockIdx=0, stemBlockNum=2         │
│ 每个 kb: bulk gather 2 kv_blocks (128 rows FP8) → K/V/kScale    │
└──────────────────────────────────────────────────────────────────┘

┌─ Core 1 ─────────────────────────────────────────────────────────┐
│ flat[2,3] → (b=0,h=0,kb=2), (b=0,h=0,kb=3)                     │
│ BatchIdx=0, HeadNumIdx=0, stemBlockIdx=2, stemBlockNum=2         │
└──────────────────────────────────────────────────────────────────┘

... (Core 2..9 处理 b=0,h=0, kb=4..9 → 10 个 stem_block 均分到 10 core)

┌─ Core 10 ────────────────────────────────────────────────────────┐
│ flat[20,21] → (b=0,h=1,kb=0), (b=0,h=1,kb=1)                   │
│ BatchIdx=0, HeadNumIdx=1, stemBlockIdx=0, stemBlockNum=2         │
└──────────────────────────────────────────────────────────────────┘

... (Core 10..19 处理 b=0,h=1 的 kb=0..9)

... (Core 20..29 处理 b=0,h=2 的 kb=0..9)

... (Core 30..39 处理 b=0,h=3 的 kb=0..9)

──────────────────────────────────────────────────────────────────
  V Processing: 同一 (b,h) 的 stem_block 分布在多个 core
  → Phase A: 各 core 独立累积局部 sumLog/sumLogSq → GM 写入
  → SyncAll #1: 等所有 core 完成
  → Phase B: 仅 (b,h) 组内 local_id==0 的 core 聚合 μ/σ
  → SyncAll #2: 等聚合完成
  → Phase C: 各 core Normalize + 写 vBias
```

**模式2 分核示意 (batch=1, H_kv=1, max_Kb=80, AICore=40)**:

```
total_stem_blocks = 80
stem_blocks_per_core = ceil(80/40) = 2

┌─ Core 0  ─ (b=0,h=0,kb=0,1)   BatchIdx=0, HeadNumIdx=0, stemBlockIdx=0,  stemBlockNum=2 ─┐
┌─ Core 1  ─ (b=0,h=0,kb=2,3)   BatchIdx=0, HeadNumIdx=0, stemBlockIdx=2,  stemBlockNum=2 ─┐
┌─ Core 39 ─ (b=0,h=0,kb=78,79) BatchIdx=0, HeadNumIdx=0, stemBlockIdx=78, stemBlockNum=2 ─┐

  40 核并行处理同一 (b,h) 的 stem_block → 2 SyncAll 聚合 μ/σ
  每个 core 处理 2 个 stem_block, 负载极均衡
  每个 stem_block bulk gather 2 个 kv_block (stemBlockSize/kvBlockSize=2)
```

**模式2 core 跨 (b,h) 边界场景 (batch=4, H_kv=8, max_Kb=2, AICore=40)**:

```
total_stem_blocks = 4 × 8 × 2 = 64
stem_blocks_per_core = ceil(64/40) = 2

┌─ Core 0 ─────────────────────────────────────────────────────────┐
│ flat[0,1] → (b=0,h=0,kb=0), (b=0,h=0,kb=1)                     │
│ BatchIdx=0, HeadNumIdx=0, stemBlockIdx=0, stemBlockNum=2         │
│ V Processing: 仅与同 (b=0,h=0) 的 core 需要 SyncAll              │
└──────────────────────────────────────────────────────────────────┘

┌─ Core 31 ────────────────────────────────────────────────────────┐
│ flat[62,63] → (b=3,h=7,kb=0), (b=3,h=7,kb=1)                   │
│ BatchIdx=3, HeadNumIdx=7, stemBlockIdx=0, stemBlockNum=2         │
│ 每个 core 处理 2 stem_block, 恰好在 (b,h) 边界内                 │
│ (max_Kb=2 → 每个 core 恰好处理一个 (b,h) 的全部 kb)              │
└──────────────────────────────────────────────────────────────────┘

┌─ Core 32..39 ────────────────────────────────────────────────────┐
│ flat[64..79] → 超出 total_stem_blocks → return (空核)            │
└──────────────────────────────────────────────────────────────────┘
```

**注意**: 当 stem_blocks_per_core > max_Kb 时，单个 core 会跨 (b,h) 边界处理多个 pair 的 stem_block。此时 V Processing 需为每个 (b,h) pair 分别 SyncAll，逻辑复杂度增加。推荐 stem_blocks_per_core ≤ max_Kb 时使用模式2 (Host Tiling 可通过条件判断控制)。

**TilingData 字段与分核的关系**:

| TilingData 字段 | 分核用途 | 计算方式 |
|---|---|---|
| split_mode | 选择分核粒度 | total_bh≥N→0, total_bh<N→1, stemBlockSize≥kvBlockSize→2 |
| cores_per_bh | 模式1: 每(b,h)核数 | N/total_bh |
| total_tasks | 决定核数 | 模式0=total_bh, 模式1=total_bh×cores_per_bh, 模式2=N |
| used_core_num | 实际使用核数 | 模式0=min(total_bh,N), 模式1=N, 模式2=min(N,total_stem_blocks) |
| batch_size, num_kv_heads | flat_idx→(b,h)映射 | flat_idx/(H_kv×max_Kb), flat_idx%(H_kv×max_Kb)/max_Kb |
| max_Kb | stem_block 范围 | flat_idx%max_Kb |
| total_stem_blocks | 模式2: stem_block总数 | batch×H_kv×max_Kb |
| stem_blocks_per_core | 模式2: 每核stem_block数 | ceil(total_stem_blocks/used_core_num) |
| kv_blocks_per_stem_block | 模式2: 每stem_block的kv_block数 | stemBlockSize/kvBlockSize |
| kv_block_size, kv_layout | De-page/kScaleCache 偏移计算 | ATTR 传入 |
| stem_block_size, stem_stride, R_val | Stem Block 结构参数 | ATTR 传入 + 派生 |
| kflat_dim | 输出每块宽度 | S × D |
| lambda_mag | λ 乘数 | lambdaMag tensor[0] |
| epsilon | Normalize ε | 固定 1e-6 |

**TilingData 结构体**:

```cpp
constexpr uint32_t DIM_QK = 128;
constexpr uint32_t KV_INDICES_BATCH_SIZE = 64;
constexpr uint32_t MAX_STEM_BLOCK_SIZE = 256;
constexpr uint32_t MAX_STEM_STRIDE = 64;
constexpr uint32_t MAX_R_VAL = MAX_STEM_BLOCK_SIZE / 16;
constexpr uint32_t MAX_KFLAT_DIM = MAX_STEM_STRIDE * DIM_QK;
constexpr uint32_t DENSE_ROW_ELEM = DIM_QK;
constexpr uint32_t REDUCE_TMP_SIZE = 9216;

struct StemPrepKVTilingData {
    uint32_t batch_size = 0;
    uint32_t num_kv_heads = 0;
    uint32_t dim_qk = DIM_QK;
    uint32_t dim_v = DIM_QK;
    uint32_t stem_block_size = 128;
    uint32_t stem_stride = 16;
    uint32_t R_val = 8;
    uint32_t kflat_dim = 2048;
    uint32_t kv_block_size = 64;
    uint32_t kv_layout = 0;
    uint32_t total_blocks = 0;
    uint32_t max_k_padded = 0;
    uint32_t max_Kb = 0;
    uint32_t max_kv_blocks = 0;
    uint32_t vbias_stride = 0;
    float lambda_mag = 0.3f;
    float epsilon = 1e-6f;
    uint32_t total_tasks = 0;
    uint32_t used_core_num = 0;
    uint32_t split_mode = 0;
    uint32_t cores_per_bh = 1;
    uint32_t total_stem_blocks = 0;
    uint32_t stem_blocks_per_core = 1;
    uint32_t kv_blocks_per_stem_block = 1;
};
```

### 2.4 FP8 路径设计

**触发条件**: kCache.dtype == FP8_e4m3fn, tilingKey = 0

#### 2.4.1 计算流程图 — 模式0 (粗粒度, 零 SyncAll)

```
┌────────── INPUT ───────────────────────────────────────────────────────┐
│ kCache FP8, vCache FP8, kvIndices INT32, kvSeqLens INT32               │
│ kScaleCache FP32, vScale FP32, lambdaMag FP32                          │
│ ATTR: kvLayout, kvBlockSize, B, S                                       │
│ 派生: R=B/S, kflat_dim=S×128, k_down_len=num_Kb×R                     │
│ task_id → batch_idx, head_idx, kv_len, num_Kb                          │
│ Init: vScale[h], lambdaMag 读入                                         │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────── K Processing (逐 kb, 无跨核依赖) ───────────────────────────────┐
│                                                                          │
│  for kb in [0, num_Kb):                                                  │
│    Duplicate(groupSumBuf, 0.0f, kflat_dim)  ← 清零累加器               │
│    for r in [0, R):                                                      │
│      for g in [0, S):                                                    │
│        row_idx = kb×B + r×S + g                                          │
│        ┌───────────┐                                                      │
│        │ De-page K │ DataCopyPad<fp8> (layout-aware offset)             │
│        │ + Cast    │ Cast<fp8→fp32> → denseRow [128] FP32              │
│        └───────────┘                                                      │
│             │                                                             │
│             │  ReadKScaleCache(row_idx) → kscale_val scalar FP32         │
│             │                                                             │
│        ┌───────────┐                                                      │
│        │ Muls      │ denseRow × kscale_val → K_scaled [128] FP32        │
│        └───────────┘                                                      │
│             │                                                             │
│        ┌───────────┐                                                      │
│        │ Add       │ groupSumBuf[(S-1-g)×D] += K_scaled                  │
│        │ (flip)    │ anti-diagonal flip: offset=S-1-g                   │
│        └───────────┘                                                      │
│                                                                          │
│    ┌───────────┐                                                          │
│    │ Cast      │ groupSumBuf[kflat_dim] FP32 → kflatBuf BF16            │
│    │ + Write   │ DataCopyPad → kFlatGM[kb×kflat_dim]                    │
│    └───────────┘                                                          │
│                                                                          │
│  数据节点:                                                                │
│    groupSumBuf [kflat_dim]  FP32  (UB, 每kb清零重用)                     │
│    kFlat      [max_Kb, kflat_dim]  BF16  (GM, 逐kb写)                   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────── V Phase A: 累积全局统计量 (单 core 内, 无同步) ───────────────┐
│                                                                          │
│  sumLog = 0, sumLogSq = 0                                                │
│                                                                          │
│  for kb in [0, num_Kb):                                                  │
│    for idx_off in [0, R):                                                 │
│      for s in [0, S):                                                    │
│        row_idx = kb×B + idx_off×S + s                                     │
│        ┌───────────┐                                                      │
│        │ De-page V │ DataCopyPad<fp8> + Cast<fp8→fp32>                  │
│        │ + ×vScale│ Muls(denseRow, vScale[h]) → V_scaled [128] FP32    │
│        └───────────┘                                                      │
│             │                                                             │
│        ┌───────────┐                                                      │
│        │ L2 Norm    │ Mul(V_scaled²) → ReduceSum → Sqrt                  │
│        │            │ → norms[idx_off×S+s] scalar FP32                   │
│        └───────────┘                                                      │
│             │                                                             │
│    ┌───────────┐                                                          │
│    │ Max Pool │ ReduceMax(norms[idx_off×S_0..S_0+S], S_)                 │
│    │          │ → vNormDown[idx_off] scalar FP32                          │
│    └───────────┘                                                          │
│             │                                                             │
│    ┌───────────┐                                                          │
│    │ Log       │ Adds(vNormDown, ε) → Log → logVals[idx_off] FP32        │
│    └───────────┘                                                          │
│             │                                                             │
│    sumLog  += Σ logVals (该 kb 的 R 个值)                                │
│    sumLogSq += Σ logVals²                                                │
│                                                                          │
│  数据节点:                                                                │
│    sumLog    scalar  FP32  (单core内累积)                                 │
│    sumLogSq  scalar  FP32  (单core内累积)                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
         │
         │  (全部 kb 遍历完成后, sumLog/sumLogSq 覆盖所有 k_down_len)
         ▼
┌─────── V Phase B: 计算 μ/σ (单 core, 无 SyncAll) ─────────────────────┐
│                                                                          │
│  μ = sumLog / k_down_len                                                 │
│  variance = (sumLogSq - sumLog²/k_down_len) / (k_down_len - 1)          │
│  if variance < 0 → variance = 0                                          │
│  σ = Sqrt(variance)                                                       │
│  inv_std = 1.0 / (σ + ε)                                                 │
│                                                                          │
│  数据节点:                                                                │
│    μ         scalar  FP32    [1]                                          │
│    inv_std   scalar  FP32    [1]                                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────── V Phase C: Normalize + ReLU + 写 vBias (逐 kb) ────────────────┐
│                                                                          │
│  for kb in [0, num_Kb):                                                  │
│    (重新 De-page V + L2 Norm + Max + Log → logVals[R] FP32)             │
│             │                                                             │
│    ┌───────────┐                                                          │
│    │ Normalize │ Subs(logVals, μ) → Muls(×inv_std)                       │
│    │           │ → normalized[R] FP32                                     │
│    └───────────┘                                                          │
│             │                                                             │
│    ┌───────────┐                                                          │
│    │ ReLU      │ Relu(normalized) → relu_vals[R] FP32                    │
│    │ + ×λ     │ Muls(relu_vals, lambda_mag) → v_final[R] FP32           │
│    └───────────┘                                                          │
│             │                                                             │
│    ┌───────────┐                                                          │
│    │ BlockAvg │ ReduceSum(v_final) / R → v_bias_val scalar FP32          │
│    │ + Write  │ ((__gm__ float*)vBiasGM)[offset + kb] = v_bias_val       │
│    └───────────┘                                                          │
│                                                                          │
│  数据节点:                                                                │
│    logVals     [R]  FP32  (逐kb重算)                                      │
│    normalized  [R]  FP32  ((logVals-μ)×inv_std)                           │
│    v_final     [R]  FP32  (λ×ReLU(normalized))                            │
│    vBias       [max_Kb]  FP32  (GM, 逐kb直接写)                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────── OUTPUT ───────────────────────────────────────────────┐
│ kFlat [batch, H_kv, max_Kb, kflat_dim]  BF16                           │
│ vBias [batch, H_kv, max_Kb]              FP32                           │
└──────────────────────────────────────────────────────────────────────────┘
```

#### 2.4.2 计算流程图 — 模式1 (细粒度, 2 SyncAll)

```
┌────────── INPUT ───────────────────────────────────────────────────────┐
│ (同模式0)                                                               │
│ task_id → group_id, local_id, batch_idx, head_idx, kb_start, kb_end    │
│ Init: vScale[h], lambdaMag 读入                                         │
│ GM stats buffer: gm_stats[group_id][local_id] = {sumLog, sumLogSq}     │
│ GM shared buffer: gm_shared[group_id] = {μ, inv_std}                   │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Phase 1: K Processing (各 core 独立, 无同步) ──────────────────┐
│                                                                   │
│ for kb in [kb_start, kb_end):                                     │
│     Duplicate(groupSumBuf, 0.0f)                                  │
│     for r,g: DepageRow(K) → Cast → ×kScaleCache → Add(flip)     │
│     Cast → DataCopyPad → kFlatGM[kb×kflat_dim]                   │
│                                                                   │
│ 数据节点: kFlat[b,h,kb,kflat_dim] BF16 ← 各 core 写各自 kb 区间 │
└───────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Phase 2: V Phase A — 局部统计量 (各 core 独立) ─────────────────┐
│                                                                   │
│ local_sumLog = 0, local_sumLogSq = 0                              │
│ for kb in [kb_start, kb_end):                                     │
│     for idx_off, s:                                                │
│         DepageRow(V) → ×vScale → L2 Norm → Max → Log             │
│         local_sumLog  += Σ logVals                                │
│         local_sumLogSq += Σ logVals²                              │
│                                                                   │
│ ┌───────────────────┐                                             │
│ │ GM Write Stats    │ gm_stats[g][lid] = {local_sumLog,           │
│ │                   │                local_sumLogSq}              │
│ └───────────────────┘                                             │
│                                                                   │
│ 数据节点: gm_stats [total_bh × cores_per_bh × 2] FP32 GM        │
└───────────────────────────────────────────────────────────────────┘
         │
    ┌─────────────┐
    │ SyncAll #1  │ ← 等所有 core 完成局部统计写入
    └─────────────┘
         │
         ▼
┌─ Phase 3: 聚合 μ/σ (仅 local_id==0 的 core 执行) ──────────────┐
│                                                                   │
│ if local_id == 0:                                                 │
│     ┌───────────────────┐                                         │
│     │ GM Read + ReduceSum│                                        │
│     │                    │ global_sumLog   = Σ gm_stats[g][i].sumLog │
│     │                    │ global_sumLogSq = Σ gm_stats[g][i].sumLogSq │
│     └───────────────────┘                                         │
│             │                                                     │
│     ┌───────────┐                                                 │
│     │ μ         │ μ = global_sumLog / k_down_len                  │
│     │ σ         │ σ = √((global_sumLogSq - global_sumLog²/       │
│     │           │     k_down_len) / (k_down_len-1))              │
│     │ inv_std   │ inv_std = 1/(σ + ε)                            │
│     └───────────┘                                                 │
│             │                                                     │
│     ┌───────────────────┐                                         │
│     │ GM Write Shared   │ gm_shared[g] = {μ, inv_std}            │
│     └───────────────────┘                                         │
│                                                                   │
│ else: (local_id != 0 → 空等)                                     │
│                                                                   │
│ 数据节点: μ scalar FP32, inv_std scalar FP32                     │
│           gm_shared [total_bh × 2] FP32 GM                       │
└───────────────────────────────────────────────────────────────────┘
         │
    ┌─────────────┐
    │ SyncAll #2  │ ← 等指定 core 完成 μ/inv_std 写入
    └─────────────┘
         │
         ▼
┌─ Phase 4: V Phase C — Normalize + 写 vBias (各 core 独立) ───────┐
│                                                                   │
│ μ      = gm_shared[group_id].μ                                    │
│ inv_std = gm_shared[group_id].inv_std                             │
│                                                                   │
│ for kb in [kb_start, kb_end):                                     │
│     (重新 Depage V + L2 Norm + Max + Log → logVals[R])           │
│     ┌───────────┐                                                 │
│     │ Normalize │ Subs(logVals, μ) → Muls(×inv_std)              │
│     └───────────┘                                                 │
│     ┌───────────┐                                                 │
│     │ ReLU+λ   │ Relu → Muls(×lambda_mag) → v_final[R]          │
│     └───────────┘                                                 │
│     ┌───────────┐                                                 │
│     │ BlockAvg │ ReduceSum(v_final)/R → v_bias_val               │
│     │ + GM Write│ vBiasGM[offset+kb] = v_bias_val                │
│     └───────────┘                                                 │
│                                                                   │
│ 数据节点: vBias[b,h,kb] FP32 ← 各 core 写各自 kb 区间           │
└───────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────── OUTPUT ───────────────────────────────────────────────┐
│ kFlat [batch, H_kv, max_Kb, kflat_dim]  BF16                           │
│ vBias [batch, H_kv, max_Kb]              FP32                           │
└──────────────────────────────────────────────────────────────────────────┘
```

**GM 额外空间 (模式1)**:

| Buffer | Shape | Dtype | 大小 (batch=1,H_kv=4) | 用途 |
|--------|-------|-------|------------------------|------|
| gm_stats | [total_bh × cores_per_bh × 2] | FP32 | 4×10×2×4B = 320B | 各 core 局部 sumLog/sumLogSq |
| gm_shared | [total_bh × 2] | FP32 | 4×2×4B = 32B | 聚合后 μ/inv_std |

总额外 GM: 极小 (数百字节), 通过 workspace 分配。

#### 2.4.3 计算流程图 — 模式2 (Template 3, stem_block级分核 + bulk kv_block gather, 2 SyncAll)

```
┌────────── INPUT ───────────────────────────────────────────────────────┐
│ kCache FP8, vCache FP8, kvIndices INT32, kvSeqLens INT32               │
│ kScaleCache FP32, vScale FP32, lambdaMag FP32                          │
│ ATTR: kvLayout, kvBlockSize, B, S                                       │
│ 派生: R=B/S, kflat_dim=S×128, k_down_len=num_Kb×R                     │
│ kv_blocks_per_stem_block = B / kvBlockSize                              │
│                                                                          │
│ core_id → flat 范围 → (BatchIdx, HeadNumIdx, stemBlockIdx, stemBlockNum)│
│ Init: vScale[HeadNumIdx], lambdaMag 读入                                │
│ GM stats buffer: gm_stats[flat_idx] = {local_sumLog, local_sumLogSq}   │
│ GM shared buffer: gm_shared[BatchIdx, HeadNumIdx] = {μ, inv_std}       │
└──────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Phase 1: K Processing (bulk kv_block gather, 各 core 独立) ─────────┐
│                                                                         │
│ for kb in [stemBlockIdx, stemBlockIdx + stemBlockNum):                  │
│     Duplicate(groupSumBuf, 0.0f)                                        │
│     for r in [0, R):                                                    │
│       for g in [0, S):                                                  │
│         row_idx = kb×B + r×S + g                                        │
│         ┌───────────────────────────────────────────┐                    │
│         │ Bulk Gather K                              │                    │
│         │ logical_kv_start = kb × kv_blocks_per_stem │                    │
│         │   _block + r×S+g / kvBlockSize             │                    │
│         │ physical_block = kvIndices[BatchIdx,        │                    │
│         │   logical_kv_start]                         │                    │
│         │ DataCopyPad<fp8> (layout-aware offset)     │                    │
│         │ Cast<fp8→fp32> → denseRow [128] FP32       │                    │
│         └───────────────────────────────────────────┘                    │
│              │                                                           │
│              │  ReadKScaleCache(row_idx) → kscale_val scalar FP32        │
│              │                                                           │
│         ┌───────────┐                                                    │
│         │ Muls      │ denseRow × kscale_val → K_scaled [128] FP32       │
│         └───────────┘                                                    │
│              │                                                           │
│         ┌───────────┐                                                    │
│         │ Add       │ groupSumBuf[(S-1-g)×D] += K_scaled                │
│         │ (flip)    │ anti-diagonal flip: offset=S-1-g                  │
│         └───────────┘                                                    │
│                                                                         │
│     ┌───────────┐                                                        │
│     │ Cast      │ groupSumBuf[kflat_dim] FP32 → kflatBuf BF16           │
│     │ + Write   │ DataCopyPad → kFlatGM[BatchIdx,HeadNumIdx,kb×kflat_dim]│
│     └───────────┘                                                        │
│                                                                         │
│ 数据节点: kFlat[b,h,kb,kflat_dim] BF16 ← 各 core 写各自 stem_block    │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ Phase 2: V Phase A — 局部统计量 (bulk kv_block gather) ─────────────┐
│                                                                         │
│ local_sumLog = 0, local_sumLogSq = 0                                    │
│ for kb in [stemBlockIdx, stemBlockIdx + stemBlockNum):                  │
│     for idx_off in [0, R):                                               │
│       for s in [0, S):                                                  │
│         row_idx = kb×B + idx_off×S + s                                   │
│         ┌───────────────────────────────────────────┐                    │
│         │ Bulk Gather V                              │                    │
│         │ 同 K Processing 的 kv_block gather 逻辑   │                    │
│         │ DataCopyPad<fp8> + Cast<fp8→fp32>         │                    │
│         │ Muls(denseRow, vScale[HeadNumIdx])         │                    │
│         │ → V_scaled [128] FP32                      │                    │
│         └───────────────────────────────────────────┘                    │
│              │                                                           │
│         ┌───────────┐                                                    │
│         │ L2 Norm    │ Mul(V_scaled²) → ReduceSum → Sqrt                 │
│         │            │ → norms[idx_off×S+s] scalar FP32                  │
│         └───────────┘                                                    │
│              │                                                           │
│     ┌───────────┐                                                        │
│     │ Max Pool │ ReduceMax(norms[idx_off×S_0..S_0+S], S_)               │
│     │          │ → vNormDown[idx_off] scalar FP32                        │
│     └───────────┘                                                        │
│              │                                                           │
│     ┌───────────┐                                                        │
│     │ Log       │ Adds(vNormDown, ε) → Log → logVals[idx_off] FP32       │
│     └───────────┘                                                        │
│              │                                                           │
│     local_sumLog  += Σ logVals (该 kb 的 R 个值)                        │
│     local_sumLogSq += Σ logVals²                                        │
│                                                                         │
│ ┌───────────────────┐                                                    │
│ │ GM Write Stats    │ gm_stats[flat_idx] = {local_sumLog,                │
│ │                   │                local_sumLogSq}                      │
│ └───────────────────┘                                                    │
│                                                                         │
│ 数据节点: gm_stats [used_core_num × 2] FP32 GM                          │
└─────────────────────────────────────────────────────────────────────────┘
         │
     ┌─────────────┐
     │ SyncAll #1  │ ← 等所有 core 完成局部统计写入
     └─────────────┘
         │
         ▼
┌─ Phase 3: 聚合 μ/σ ────────────────────────────────────────────────┐
│                                                                       │
│ 本 core 判断是否为 (BatchIdx, HeadNumIdx) 组内的第一个处理该 pair 的 │
│ core (通过 flat_idx 判断: flat_idx % max_Kb == 0 → 组头)             │
│                                                                       │
│ if 组头:                                                              │
│     ┌───────────────────┐                                             │
│     │ GM Read + ReduceSum│                                            │
│     │                    │ global_sumLog   = Σ gm_stats[同(b,h) cores] │
│     │                    │ global_sumLogSq = Σ gm_stats[同(b,h) cores] │
│     └───────────────────┘                                             │
│             │                                                         │
│     ┌───────────┐                                                     │
│     │ μ         │ μ = global_sumLog / k_down_len                      │
│     │ σ         │ σ = √((global_sumLogSq - global_sumLog²/            │
│     │           │     k_down_len) / (k_down_len-1))                   │
│     │ inv_std   │ inv_std = 1/(σ + ε)                                │
│     └───────────┘                                                     │
│             │                                                         │
│     ┌───────────────────┐                                             │
│     │ GM Write Shared   │ gm_shared[BatchIdx, HeadNumIdx] = {μ, inv_std}│
│     └───────────────────┘                                             │
│                                                                       │
│ else: (非组头 → 空等)                                                 │
│                                                                       │
│ 数据节点: μ scalar FP32, inv_std scalar FP32                         │
│           gm_shared [total_bh × 2] FP32 GM                           │
└───────────────────────────────────────────────────────────────────────┘
         │
     ┌─────────────┐
     │ SyncAll #2  │ ← 等组头 core 完成 μ/inv_std 写入
     └─────────────┘
         │
         ▼
┌─ Phase 4: V Phase C — Normalize + 写 vBias (各 core 独立) ───────────┐
│                                                                       │
│ μ      = gm_shared[BatchIdx, HeadNumIdx].μ                            │
│ inv_std = gm_shared[BatchIdx, HeadNumIdx].inv_std                     │
│                                                                       │
│ for kb in [stemBlockIdx, stemBlockIdx + stemBlockNum):                │
│     (重新 Bulk Gather V + L2 Norm + Max + Log → logVals[R])          │
│     ┌───────────┐                                                     │
│     │ Normalize │ Subs(logVals, μ) → Muls(×inv_std)                  │
│     └───────────┘                                                     │
│     ┌───────────┐                                                     │
│     │ ReLU+λ   │ Relu → Muls(×lambda_mag) → v_final[R]              │
│     └───────────┘                                                     │
│     ┌───────────┐                                                     │
│     │ BlockAvg │ ReduceSum(v_final)/R → v_bias_val                   │
│     │ + GM Write│ vBiasGM[BatchIdx,HeadNumIdx,kb] = v_bias_val       │
│     └───────────┘                                                     │
│                                                                       │
│ 数据节点: vBias[b,h,kb] FP32 ← 各 core 写各自 stem_block            │
└───────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────── OUTPUT ───────────────────────────────────────────────┐
│ kFlat [batch, H_kv, max_Kb, kflat_dim]  BF16                           │
│ vBias [batch, H_kv, max_Kb]              FP32                           │
└──────────────────────────────────────────────────────────────────────────┘
```

**GM 额外空间 (模式2)**:

| Buffer | Shape | Dtype | 大小 (batch=2,H_kv=4,max_Kb=10,N=40) | 用途 |
|--------|-------|-------|---------------------------------------|------|
| gm_stats | [used_core_num × 2] | FP32 | 40×2×4B = 320B | 各 core 局部 sumLog/sumLogSq |
| gm_shared | [total_bh × 2] | FP32 | 8×2×4B = 64B | 聚合后 μ/inv_std per (b,h) |

总额外 GM: 极小 (数百字节), 通过 workspace 分配。

**模式2 kv_block Bulk Gather 策略详解**:

核心优化点: 当 stemBlockSize ≥ kvBlockSize 时，每个 stem_block 包含 `kv_blocks_per_stem_block = stemBlockSize / kvBlockSize` 个 kv_block。De-page 阶段不再逐行搬运，而是按 kv_block 为单位进行 bulk gather。

**kv_block Bulk Gather 偏移计算**:

对于 stem_block `kb` 中的行 `row_idx` (row_idx = kb × B + r × S + g):
- logical_kv_block = row_idx / kvBlockSize
- physical_block = kvIndices[BatchIdx, logical_kv_block]
- row_in_kv_block = row_idx % kvBlockSize

| Layout | K/V Cache bulk offset (head=h, block=phys, row=r_in_block) |
|--------|------|
| Layout A | `phys × kvBlockSize × H_kv × D + r_in_block × H_kv × D + h × D` |
| Layout B | `phys × H_kv × kvBlockSize × D + h × kvBlockSize × D + r_in_block × D` |

| Layout | kScaleCache bulk offset |
|--------|------|
| Layout A | `kScaleCache[phys, row_in_block, h, 0]` → `phys × kvBlockSize × H_kv + row_in_block × H_kv + h` |
| Layout B | `kScaleCache[phys, h, row_in_block, 0]` → `phys × H_kv × kvBlockSize + h × kvBlockSize + row_in_block` |

**Bulk Gather 可选策略**:

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| 策略A: 逐行仍按 kv_block 边界预取 | 逐行 DataCopyPad 但行内 physical_block 预计算不变 | 所有场景, 兼容现有逻辑 |
| 策略B: 按 kv_block 批量搬运 | 每个 kv_block 搬运 kvBlockSize 行连续数据 (Layout B: h维度连续) | Layout B, h 固定 |
| 策略C: 按 stem_block 整块搬运 | 一个 stem_block 的所有行一次性从多个 kv_block 拷入 | stem_block 行数少 (B≤128) + UB 空间足够 |

**推荐**: 初始迭代采用策略A (逐行但 kv_block 边界预计算不变), 后续优化迭代采用策略B/C。

**策略B 详细 (Layout B, kvBlockSize=64)**:

当 kvBlockSize=64, stemBlockSize=128 时，每个 stem_block 涉及 2 个 kv_block。对于 Layout B 中特定 head h:
- 一个 kv_block 的数据: `kCache[phys_block, h, 0:64, 0:128]` = 64行×128列, 连续 64×128=8192 FP8 元素
- 可用 2D DataCopyPad 拷入 `rawRowBuf[64, 128]` FP8 → Cast → `denseRowBuf[64, 128]` FP32
- 然后在 UB 内按 (r, g) 切片访问对应行

但 UB 空间限制: 64×128×4 = 32KB (FP32 denseRow), 加上 groupSumBuf 32KB, reduceTmp 36KB, 总计 > 100KB。需仔细规划 UB 穖间。

**模式2 与模式1 的 De-page 差异**:

| 维度 | 模式1 (逐行 De-page) | 模式2 (kv_block Bulk Gather) |
|------|---------------------|-------------------------------|
| DataCopyPad 粒度 | 1行 × D=128 FP8 | kvBlockSize行 × D=128 FP8 (或逐行但预计算优化) |
| 每行 GM 读次数 | 1 (1 scalar或1行) | 同上(策略A), 或1次批量(策略B/C) |
| physical_block 计算 | 逐行计算 offset | kv_block 级 offset (减少重复计算) |
| kScaleCache gather | 逐行 scalar read | kv_block 级 bulk read (可选优化) |
| UB peak | ~87KB (逐行) | ~87KB (策略A), 或更大 (策略B/C) |

#### 2.4.7 三种模式对比

| 维度 | 模式0 (粗粒度) | 模式1 (细粒度) | 模式2 (Template 3) |
|------|----------------|-----------------|---------------------|
| **触发条件** | batch×H_kv ≥ AICore数 | batch×H_kv < AICore数 | stemBlockSize ≥ kvBlockSize |
| **典型场景** | batch≥8, H_kv≥8 | batch=1, H_kv=1~4 | 变长序列, 需bulk gather |
| **task 粒度** | (b,h) 全部 kb | (b,h,kb_range) | flat stem_block 分配 |
| **task 标识** | (batch_idx, head_idx) | (group_id, local_id) | (BatchIdx, HeadNumIdx, stemBlockIdx, stemBlockNum) |
| **core 利用率** | 每核处理完整 (b,h) | 多核并行 kb per (b,h) | 按stem_block总数均匀分配 |
| **同步次数** | 0 | 2 SyncAll | 2 SyncAll |
| **μ/σ 计算** | 单核内直接算 | 聚合局部统计量 → 广播 | 聚合局部统计量 → 组头广播 |
| **kFlat 写入** | 单核逐 kb 写 | 各核独立写各自 kb | 各核独立写各自 stem_block |
| **vBias 写入** | 单核逐 kb 直接 GM 写 | 各核独立写各自 kb | 各核独立写各自 stem_block |
| **GM 额外空间** | 无 | ~320B (stats+shared) | ~320B (stats+shared) |
| **De-page 粒度** | 逐行 | 逐行 | kv_block级bulk gather |
| **V De-page 次数** | num_Kb×2 (Phase A+C) | num_Kb×2 (Phase 2+4) | stemBlockNum×2 (Phase 2+4) |
| **核心数** | min(tasks, AICore数) | ≈AICore数 | ≈AICore数 (或 min(N,total_stem_blocks)) |
| **单核 kb 数** | 全部 num_Kb | ceil(num_Kb/cores_per_bh) | stem_blocks_per_core |
| **负载均衡** | 长序列核压力大 | per (b,h)均匀 | 全局stem_block均匀 |
| **性能瓶颈** | seqLen 大时单核压力大 | SyncAll 开销 (~μs级) | SyncAll + GM stride |
| **确定性** | 单核累加，天然确定性 | SyncAll 保证一致性 | SyncAll 保证一致性 |
| **内核复杂度** | 低 | 中 (group/local) | 中低 (flat index映射) |

#### 2.4.4 Host 侧 Tiling

```cpp
#include "stem_oam_prep_paged_kv_tiling_key.h"

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto compile_info = static_cast<const StemPrepKVCompileInfo*>(context->GetCompileInfo());
    auto attrs = context->GetAttrs();
    int64_t kvLayout = attrs->GetAttrValue<int64_t>("kvLayout");
    int64_t kvBlockSize = attrs->GetAttrValue<int64_t>("kvBlockSize");
    int64_t stemBlockSize = attrs->GetAttrValue<int64_t>("stemBlockSize");
    int64_t stemStride = attrs->GetAttrValue<int64_t>("stemStride");

    context->SetTilingKey(0);
    ASCENDC_TPL_SEL_PARAM(context, 0);

    auto tiling = context->GetTilingData<StemPrepKVTilingData>();

    uint32_t H_kv;
    if (kvLayout == 0) H_kv = kcacheShape->GetStorageShape().GetDim(2);
    else                H_kv = kcacheShape->GetStorageShape().GetDim(1);

    uint32_t total_bh = batch * H_kv;
    uint32_t N = compile_info->core_num;

    tiling->kv_blocks_per_stem_block = static_cast<uint32_t>(stemBlockSize / kvBlockSize);
    tiling->total_stem_blocks = total_bh * tiling->max_Kb;

    if (total_bh >= N) {
        tiling->split_mode = 0;
        tiling->cores_per_bh = 1;
        tiling->total_tasks = total_bh;
        tiling->used_core_num = min(total_bh, N);
        tiling->stem_blocks_per_core = tiling->max_Kb;
    } else if (stemBlockSize >= kvBlockSize && tiling->total_stem_blocks >= N) {
        tiling->split_mode = 2;
        tiling->cores_per_bh = N / total_bh;
        tiling->total_tasks = N;
        tiling->used_core_num = min(N, tiling->total_stem_blocks);
        tiling->stem_blocks_per_core = (tiling->total_stem_blocks + tiling->used_core_num - 1) / tiling->used_core_num;
    } else {
        tiling->split_mode = 1;
        tiling->cores_per_bh = N / total_bh;
        tiling->total_tasks = total_bh * tiling->cores_per_bh;
        tiling->used_core_num = N;
        tiling->stem_blocks_per_core = (tiling->max_Kb + tiling->cores_per_bh - 1) / tiling->cores_per_bh;
    }

    // ... rest of tiling fields
    context->SetBlockDim(tiling->used_core_num);
    context->SetTilingData(tiling);
    return ge::GRAPH_SUCCESS;
}
```

#### 2.4.5 Kernel 侧实现

```cpp
#include "stem_oam_prep_paged_kv_tiling_key.h"

__global__ __aicore__ void stem_oam_prep_paged_kv_apt(
    GM_ADDR kcache, GM_ADDR vcache, GM_ADDR kv_indices,
    GM_ADDR kv_seq_lens, GM_ADDR kscale_cache, GM_ADDR vscale,
    GM_ADDR lambda_mag, GM_ADDR kflat, GM_ADDR v_bias,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(StemPrepKVTilingData);
    GET_TILING_DATA_WITH_STRUCT(StemPrepKVTilingData, tilingData, tiling);

    KernelStemPrepKV op;
    op.Init(kcache, vcache, kv_indices, kv_seq_lens,
            kscale_cache, vscale, lambda_mag,
            kflat, v_bias, workspace, tilingData);
    op.Process();
}
```

#### 2.4.6 API 映射

| 计算步骤 | Ascend C API | 参数签名 | 平台验证 | 约束说明 | 替代方案 |
|---------|-------------|---------|---------|---------|---------|
| De-page K/V (Layout A) | DataCopyPad\<fp8_e4m3fn_t\> | 1行D=128元素, layout-aware offset | ✅ DAV_3510 | offset=phys×kvBS×H_kv×D+row×H_kv×D+h×D | 2D DataCopyPad (GM stride) |
| De-page K/V (Layout B) | DataCopyPad\<fp8_e4m3fn_t\> | 1行D=128元素, layout-aware offset | ✅ DAV_3510 | offset=phys×H_kv×kvBS×D+h×kvBS×D+row×D | 2D DataCopyPad (GM stride) |
| FP8→FP32 Cast | Cast\<fp8_e4m3fn_t, float\> | (dst, src, D=128, CAST_ROUND) | ✅ DAV_3510 | 逐行 Cast | - |
| kScaleCache scalar read | DataCopyPad\<float\> | (dst, src, {1,4,0,0}) | ✅ DAV_3510 | layout-aware offset; 1 scalar per row | bulk read per kb (迭代二) |
| Per-token K-scale | Muls\<float\> | (dst, src, scalar, D) | ✅ DAV_3510 | denseRow × kScaleCache_val, D elements | Mul(dst, src1, src2) 逐元素 |
| Per-head V-scale | Muls\<float\> | (dst, src, scalar, D) | ✅ DAV_3510 | denseRow × vScale[h], D elements | - |
| Group Sum | Add\<float\> | (dst, src1, src2, D) | ✅ DAV_3510 | anti-diag flip: offset=(S-1-g)×D | ReduceSum (需 tmpBuf) |
| L2 Norm | Mul+ReduceSum+Sqrt | Mul→ReduceSum→Sqrt | ✅ DAV_3510 | 共享 reduceTmpBuf 36KB | - |
| Max Pool | ReduceMax\<float\> | Level2 Reduce | ✅ DAV_3510 | S_ elements | - |
| Log | Log\<float\> | (dst, src, count) | ✅ DAV_3510 | ε 保证输入>0 | - |
| Normalize | Subs+Muls | Subs(-μ)→Muls(×inv_std) | ✅ DAV_3510 | inv_std = 1/(σ+ε) | - |
| ReLU | Relu\<float\> | (dst, src, count) | ✅ DAV_3510 | - | - |
| FP32→BF16 | Cast\<float, bfloat16_t\> | (dst, src, kflat_dim, CAST_ROUND) | ✅ DAV_3510 | kFlat 输出 | - |
| kFlat write-back | DataCopyPad\<bfloat16_t\> | UB→GM | ✅ DAV_3510 | kflat_dim×2B per kb | - |
| vBias write-back | Direct GM scalar write | ((__gm__ float*)vBiasGM)[off+kb] | ✅ DAV_3510 | 1 scalar per kb | DataCopyPad (如需向量写) |
| kv_indices 加载 | EnsureKvIndicesForBlock | 分批 64 INT32 | ✅ DAV_3510 | UB 缓存 kvIndicesBuf | - |

**API 验证记录**:

| API 名称 | 官方文档路径 | 通配符搜索结果 | 验证状态 | 备注 |
|---------|-------------|---------------|---------|------|
| DataCopyPad | docs/api/context/DataCopyPad*.md | DataCopyPad.md, DataCopyPadExtParams.md | ✅ 已验证 | arch35 支持 FP8/BF16/FP16/FP32 全类型 |
| Cast | docs/api/context/Cast*.md | Cast.md | ✅ 已验证 | fp8_e4m3fn→float DAV_3510 原生支持 |
| Mul | docs/api/context/Mul*.md | Mul.md | ✅ 已验证 | float 类型支持 |
| Muls | docs/api/context/Muls*.md | Muls.md | ✅ 已验证 | scalar 广播乘法 |
| Add | docs/api/context/Add*.md | Add.md | ✅ 已验证 | float 类型累加 |
| ReduceSum | docs/api/context/ReduceSum*.md | ReduceSum.md | ✅ 已验证 | Level2 Reduce |
| ReduceMax | docs/api/context/ReduceMax*.md | ReduceMax.md | ✅ 已验证 | Level2 Reduce |
| Sqrt | docs/api/context/Sqrt*.md | Sqrt.md | ✅ 已验证 | scalar/vector |
| Log | docs/api/context/Log*.md | Log.md | ✅ 已验证 | float |
| Relu | docs/api/context/Relu*.md | Relu.md | ✅ 已验证 | float |
| Subs | docs/api/context/Subs*.md | Subs.md | ✅ 已验证 | scalar 广播减法 |
| Duplicate | docs/api/context/Duplicate*.md | Duplicate.md | ✅ 已验证 | UB 填零 |

**验证检查清单**:
- [x] 已用通配符搜索 API 所有变体文件
- [x] 已确认 API 在目标芯片架构（arch35）上可用
- [x] 已确认 API 支持所需的数据类型（FP8/BF16/FP32）
- [x] 已确认参数签名与官方文档一致
- [x] 已确认 tmpBuffer/对齐等约束条件
- [x] 如 API 不可用，已确定替代方案

**数据搬运详解**:

**GM 布局描述**:

| Tensor | Shape | 说明 |
|---|---|---|
| kCache/vCache | 两布局, FP8 | Paged KV, layout-aware offset |
| kScaleCache | 随 kvLayout, FP32 | 与 kCache 同布局 (D=1) |
| kvIndices | [batch, max_kv_blocks] INT32 | Block index, 分批64读入UB |
| kvSeqLens | [batch] INT32 | 序列长度, Init读 |
| vScale | [H_kv] FP32 | per-head, Init读 |
| lambdaMag | [1] FP32 | Init读 |
| kFlat | [batch, H_kv, max_Kb, kflat_dim] BF16 | 每(b,h,kb)连续区域 |
| vBias | [batch, H_kv, vbias_stride] FP32 | 逐kb scalar GM写 |

**De-page K/V 拷入详解**:

- 从 kvIndices GM 分批读入 kvIndicesBuf (每批64 INT32)
- 逐行 De-page: DataCopyPad 1行 D=128 FP8 → Cast FP32
- Layout-aware offset 计算 (见 2.4.6 API映射)
- kScaleCache 同布局偏移读取 (D=1)

**FP8 K Processing 数据搬运图**:

```
  ┌─────────────────────────────── GM ────────────────────────────────┐
  │ kCache[physical_block, row_in_block, head_idx, :] (D=128 FP8)    │
  │ kScaleCache[physical_block, row/head, head/row, 0] (1 FP32)      │
  │ kvIndices[batch_idx, logical_kv_block] (1 INT32)                 │
  └─────────────────── DataCopyPad ×1 row ──────────────────────────┤
  │  K offset per row = layout-aware (见 De-page 偏移)               │
  │  Scale offset per row = layout-aware (见 kScaleCache 偏移)       │
  ▼                                                                   │
  ┌─────────────────────────────── UB ────────────────────────────────┐
  │ Step 1: DepageRow(K)                                               │
  │   rawRowBuf: D=128 fp8_e4m3fn  ← 1次 DataCopyPad from GM         │
  │   denseRowBuf: D=128 FP32       ← Cast<fp8→fp32>                  │
  │                                                                     │
  │ Step 2: ReadKScaleCache(row_idx)                                    │
  │   vProcSmallBuf: 1 FP32         ← DataCopyPad from GM              │
  │   kscale_val = vProcSmall.GetValue(0)                               │
  │                                                                     │
  │ Step 3: Muls (K × kScaleCache)                                     │
  │   vComputeBuf: D=128 FP32       ← Muls(denseRow, kscale_val)      │
  │                                                                     │
  │ Step 4: Add (group sum, anti-diag flip)                             │
  │   groupSumBuf: kflat_dim FP32   ← Add(groupSum[(S-1-g)×D], vComp) │
  │                                                                     │
  │ Step 5: Cast + Write (per kb)                                       │
  │   kflatBuf: kflat_dim BF16      ← Cast(groupSum, BF16)            │
  │     ↓ DataCopyPad                                                  │
  └─────────────────────────────────────────────────────────────────────┘
  │                                                                     │
  ▼                                                                    │
  ┌─────────────────────────────── GM ────────────────────────────────┐
  │ kFlat[batch_idx, head_idx, kb, :]  (kflat_dim BF16 elements) │
  └───────────────────────────────────────────────────────────────────┘
```

**FP8 V Processing 数据搬运图**:

```
  ┌─────────────────────────────── GM ────────────────────────────────┐
  │ vCache[physical_block, row/head, head/row, :] (D=128 FP8)         │
  │ vScale[head_idx] (1 FP32, Init阶段读入)                            │
  │ lambdaMag[0] (1 FP32, Init阶段读入)                                │
  └─────────────────── per row ──────────────────────────────────────┤
  ▼                                                                   │
  ┌─────────────────────────────── UB ────────────────────────────────┐
  │ Step 1: DepageRow(V)                                               │
  │   rawRowBuf → denseRowBuf (FP8→FP32)                               │
  │                                                                     │
  │ Step 2: Muls (×vScale[h])                                          │
  │   vComputeBuf: V_scaled D=128 FP32                                 │
  │                                                                     │
  │ Step 3: L2 Norm                                                     │
  │   Mul(vCompute²) → ReduceSum → Sqrt → normsBuf scalar              │
  │                                                                     │
  │ Step 4: Max Pool per idx                                            │
  │   ReduceMax(norms[S_]) → vNormDownBuf scalar                       │
  │                                                                     │
  │ Step 5: Log                                                         │
  │   Adds(vNormDown+ε) → Log → logValsBuf[R] FP32                    │
  │                                                                     │
  │ Step 6: Normalize (after μ/σ available)                             │
  │   Subs(logVals-μ) → Muls(×inv_std) → normalizedBuf[R] FP32        │
  │   Relu → Muls(×lambda_mag) → v_final                               │
  │                                                                     │
  │ Step 7: Block Average                                               │
  │   ReduceSum(v_final)/R → v_bias_val scalar FP32                    │
  │     ↓ Direct GM write                                              │
  └─────────────────────────────────────────────────────────────────────┘
  │                                                                     │
  ▼                                                                    │
  ┌─────────────────────────────── GM ────────────────────────────────┐
  │ vBias[batch_idx, head_idx, kb]  (1 FP32 scalar per kb)            │
  └───────────────────────────────────────────────────────────────────┘
```

**Stem Block 处理时序图 (模式0)**:

```
  时间线 (单核内, for kb=0..num_Kb-1):

  kb=0 (K Processing):
    ┌── EnsureKvIndicesForBlock(kb) ──┐
    │ 分批加载 kvIndices               │
    └─────────────────────────────────┘
    ┌── ProcessKBlock(kb) ─────────────┐
    │ Duplicate(groupSum, 0)            │
    │ for r,g: DepageRow → ×kScale → Add(flip) │
    │ Cast → DataCopyPad → kFlatGM      │
    └──────────────────────────────────┘

  kb=0 (V Phase A):
    ┌── VAccumGlobalStats(kb) ─────────┐
    │ DepageRow(V) → ×vScale → L2Norm  │
    │ → Max → Log → sumLog += Σ        │
    └──────────────────────────────────┘

  ... (num_Kb 次 K+V Phase A)

  V Phase B:
    ┌── μ/σ 计算 ─────────────────────┐
    │ μ = sumLog/k_down_len             │
    │ inv_std = 1/(σ+ε)                 │
    └──────────────────────────────────┘

  kb=0 (V Phase C):
    ┌── VNormalizeAndBias(kb) ─────────┐
    │ (重算logVals)                     │
    │ Normalize → ReLU → ×λ → BlockAvg │
    │ → GM write vBias                  │
    └──────────────────────────────────┘

  ... (num_Kb 次 V Phase C)
```

**UB Buffer 生命周期表**:

| Buffer | 用途 | 生命周期 |
|---|---|---|
| rawRowBuf | De-page 1行 FP8 | 每次 DepageRow 写入, Cast 读取 |
| denseRowBuf | Cast→FP32 1行 | 每次 DepageRow Cast 写入, Muls 读取 |
| groupSumBuf | K group sum 累加器 | 每kb: Duplicate(0)初始化, 累加, Cast→kflatBuf |
| kflatBuf | kFlat BF16 输出 | 每kb Cast, 立即写GM |
| reduceTmpBuf | Reduce tmp (共享) | ReduceSum/ReduceMax 共享, 36KB |
| vComputeBuf | ×kScaleCache/×vScale | Muls 中间结果 |
| normsBuf | L2 norm 结果 | V Phase A/C 累积 |
| vNormDownBuf | Max norm 结果 | V Phase A/C ReduceMax dst |
| vProcSmallBuf | Reduce dst + scalar read | ReduceSum dst + kScaleCache/vScale/lambdaMag read |
| kvIndicesBuf | kv_indices 分批缓存 | EnsureKvIndicesForBlock 分批加载 |
| logValsBuf | log(v_norm+ε) | V Phase A/C Log 结果 |
| normalizedBuf | normalize 结果 | V Phase C Subs+Muls |

**内存管理**:

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| rawRowBuf | De-page 1行 FP8 | ((128×1+31)/32)×32 = 160B | VECCALC |
| denseRowBuf | Cast→FP32 1行 | 128×4 = 512B | VECCALC |
| groupSumBuf | K group sum 累加器 | MAX_KFLAT_DIM×4 = 64×128×4 = 32KB | VECCALC |
| kflatBuf | kFlat BF16 输出 | MAX_KFLAT_DIM×2 = 16KB | VECCALC |
| reduceTmpBuf | Reduce tmp (共享) | REDUCE_TMP_SIZE×4 = 36KB | VECCALC |
| vComputeBuf | ×kScaleCache/×vScale | 128×4 = 512B | VECCALC |
| normsBuf | L2 norm 结果 | MAX_STEM_BLOCK_SIZE×4 = 1KB | VECCALC |
| vNormDownBuf | Max norm 结果 | MAX_R_VAL×4 = 64B | VECCALC |
| vProcSmallBuf | Reduce dst + scalar read | 8×4 = 32B | VECCALC |
| kvIndicesBuf | kv_indices 分批缓存 | 64×4 = 256B | VECCALC |
| logValsBuf | log(v_norm+ε) | MAX_R_VAL×4 = 64B | VECCALC |
| normalizedBuf | normalize 结果 | MAX_R_VAL×4 = 64B | VECCALC |

**总 UB 使用量**: 160B + 512B + 32KB + 16KB + 36KB + 512B + 1KB + 64B + 32B + 256B + 64B + 64B ≈ 87KB ✅ (≤ 248KB)

**UB 容量验证**:

| 平台 | 总 UB | 向量可用 | 系统保留 | 验证公式 |
|------|-------|---------|---------|---------|
| DAV_3510 (Ascend950DT) | 248 KB | 248 KB | 0 KB | `sum(buffers) ≤ 248 * 1024` |

**UB Buffer Sizing Strategy**: stemBlockSize and stemStride are runtime parameters, not constexpr constants. UB buffers use MAX constants (MAX_STEM_BLOCK_SIZE=256, MAX_STEM_STRIDE=64) for allocation, runtime tiling values for computation. kv_block_size and kv_layout are runtime attrs reflected in tiling data.

#### 2.4.7 性能优化

**并行策略**: 双模式分核 (见 2.3 分核策略)

**kScaleCache 逐行读取开销**: 1 scalar per row, 布局感知偏移计算. 对 seqLen=1M ~2M scalar reads, ~20M cycles overhead.

**缓解策略**:
- 初版: 逐行 scalar read (迭代一)
- 优化: bulk read per kb (迭代二), 一次读取 kb×R×S 个 kScaleCache 值, UB 约 512B per kb

**V Processing 重算**: V Phase A 和 Phase C 各遍历一次 V rows (2× De-page V + L2 Norm + Max + Log). 无法缓存 logVals 因 UB 不足 (k_down_len 可达数万).

**模式1 SyncAll 开销**: 2 次 SyncAll ~μs 级, 对 num_Kb=80 场景约 2μs, 相比 De-page 开销可接受.

#### 2.4.8 风险评估

**API 风险**:
- **Varlen Paged KV offset 计算**: DataCopyPad 的 srcOffset 需精确计算 physical_block + layout-aware offset, offset 越界会导致数据错误
- **FP8 Cast 精度**: fp8_e4m3fn → float Cast 在 DAV_3510 上为原生支持，但精度范围有限 (FP8 动态范围比 FP32 小)
- **kScaleCache scatter read**: per-position kScaleCache 可能导致非连续 GM 读取，影响搬运效率；替代方案为 bulk row load per kb
- **SyncAll 可靠性**: 模式1 依赖 2 次 SyncAll 保证数据一致性，SyncAll 失败会导致 μ/σ 计算错误

**精度风险**:
- **FP32→BF16 Cast**: kFlat 输出从 FP32 Cast 为 BF16; BF16路径精度损失 rtol 约 1e-2
- **R 层累加精度**: FP32 累加 R=stemBlockSize/stemStride 层加权结果，数值稳定性依赖 K 和 kScaleCache 的值域范围
- **Global Normalize 精度**: sumLog/sumLogSq 双精度累积, 单精度 FP32 对极端长序列 (k_down_len>100K) 可能不够
- **Zero-pad 影响**: 对齐到 stemBlockSize 的 zero-pad 区域不参与计算，但 DataCopyPad 需正确设置 padVal=0

**应对措施**:
- Paged KV offset 计算通过 kvIndicesBuf GM 拷入 UB 后逐元素读取
- kScaleCache scatter read 改用 bulk DataCopyPad per kb (迭代二)
- FP32→BF16 使用 CAST_ROUND 模式减少截断误差
- Zero-pad 通过 DataCopyPad 的 padVal 参数设为 0.0f 保证正确性
- 模式1 SyncAll 使用 AscendC::SyncAll() 保证全核同步

### 2.5 PTA接口设计

**PTA (PyTorch Ascend) 接口设计**:

本算子通过 aclnn 框架接入 PyTorch，采用标准两段式调用：

```python
import torch_npu

kFlat, vBias = torch.ops.npu.stem_oam_prep_paged_kv(
    kCache,      # [total_blocks, kvBlockSize, H_kv, 128] or [total_blocks, H_kv, kvBlockSize, 128] FP8 NPU
    vCache,      # same shape/dtype/layout as kCache
    kvIndices,   # [batch, max_kv_blocks] INT32 NPU
    kvSeqLens,   # [batch] INT32 NPU
    kScaleCache, # Layout A: [total_blocks, kvBlockSize, H_kv, 1] / Layout B: [total_blocks, H_kv, kvBlockSize, 1] FP32 NPU
    vScale,      # [H_kv] FP32 NPU
    lambdaMag=0.3,
    kvLayout=0,     # 0=Layout A, 1=Layout B
    kvBlockSize=64, # 64 or 128
    stemBlockSize=128,
    stemStride=16
)
# Returns: kFlat [batch, H_kv, max_Kb, kflat_dim] BF16
#          vBias [batch, H_kv, max_Kb] FP32
```

**C++ 桥接实现**:

```cpp
// stem_oam_prep_paged_kv_torch.cpp
at::Tensor stem_oam_prep_paged_kv_torch(
    const at::Tensor& kCache,
    const at::Tensor& vCache,
    const at::Tensor& kvIndices,
    const at::Tensor& kvSeqLens,
    const at::Tensor& kScaleCache,
    const at::Tensor& vScale,
    double lambdaMag,
    int64_t kvLayout,
    int64_t kvBlockSize,
    int64_t stemBlockSize,
    int64_t stemStride)
{
    // 1. 输入校验 (dtype, device, shape)
    // 2. 计算 H_kv from kvLayout + kCache shape
    // 3. 计算 max_Kb, kflat_dim
    // 4. 创建输出 tensor
    // 5. lambdaMag float → tensor [1]
    // 6. 创建 aclTensor (at::Tensor → aclTensor)
    // 7. aclnnStemOamPrepPagedKVGetWorkspaceSize()
    // 8. 分配 workspace
    // 9. aclnnStemOamPrepPagedKV()
    // 10. 返回 kFlat, vBias
}
```

**register.cpp**:

```cpp
TORCH_LIBRARY(npu, m) {
    m.def("stem_oam_prep_paged_kv(Tensor kCache, Tensor vCache, Tensor kvIndices, Tensor kvSeqLens, Tensor kScaleCache, Tensor vScale, float lambdaMag=0.3, int kvLayout=0, int kvBlockSize=64, int stemBlockSize=128, int stemStride=16) -> (Tensor, Tensor)");
    m.impl("stem_oam_prep_paged_kv", &ascend_kernel::stem_oam_prep_paged_kv_torch);
}
```

### 2.6 代码结构

```
stem_oam_prep_paged_kv/
├── op_host/
│   ├── stem_oam_prep_paged_kv_def.cpp          # 算子定义 (FP8-only, kvLayout/kvBlockSize attrs)
│   ├── stem_oam_prep_paged_kv_infershape.cpp   # InferShape (H_kv 根据 kvLayout 提取)
│   └── arch35/
│       ├── stem_oam_prep_paged_kv_tiling.cpp    # Tiling (tilingKey=0 fixed, split_mode判定, kv_layout/kv_block_size)
│       ├── stem_oam_prep_paged_kv_tiling.h      # TilingData struct + 常量
├── op_kernel/
│   ├── stem_oam_prep_paged_kv_arch35.cpp        # Kernel 入口 (非模板, tilingKey=0)
│   └── arch35/
│       ├── stem_oam_prep_paged_kv.h             # KernelStemPrepKV (布局感知 De-page + kScaleCache + 双模式分核)
│       ├── stem_oam_prep_paged_kv_tiling_key.h  # tilingKey=0 (FP8 fixed)
│       ├── stem_oam_prep_paged_kv_tiling_data.h # TilingData (kv_layout/kv_block_size/split_mode/cores_per_bh)
├── op_api/
│   ├── aclnn_stem_oam_prep_paged_kv.h           # aclnn API (7 inputs + 4 attrs)
│   ├── aclnn_stem_oam_prep_paged_kv.cpp         # CheckParams (layout-aware, kScaleCache 4D)
│   ├── stem_oam_prep_paged_kv.h                 # Launch helper
│   └── stem_oam_prep_paged_kv.cpp               # Launch implementation
├── op_extension/
│   ├── stem_oam_prep_paged_kv_torch.cpp         # PyTorch 桥接
│   ├── register.cpp                             # TORCH_LIBRARY
│   ├── ops.h                                    # 函数声明
├── scripts/
│   ├── gen_data.py, golden.py, verify_result.py, test_torch.py
├── CMakeLists.txt                                # 标准自定义算子构建
└── tests/
    ├── ut/
    └── st/
└── docs/
    ├── aclnnStemOamPrepPagedKV.md
    ├── REQUIREMENTS.md
    ├── DESIGN.md
    ├── PLAN.md
```

**交付件清单**:

**必需**:
- `op_host/stem_oam_prep_paged_kv_def.cpp` — 算子定义 (OpDef 注册)
- `op_host/stem_oam_prep_paged_kv_infershape.cpp` — Shape 推导 (InferShape/InferDtype)
- `op_host/arch35/stem_oam_prep_paged_kv_tiling.cpp` — Tiling 计算 (split_mode判定 + tilingKey=0)
- `op_host/arch35/stem_oam_prep_paged_kv_tiling.h` — Tiling 常量 + TilingData 结构体
- `op_kernel/stem_oam_prep_paged_kv_arch35.cpp` — Kernel 入口 (REGISTER_TILING)
- `op_kernel/arch35/stem_oam_prep_paged_kv.h` — Kernel 类定义 + 双模式分核 + 计算逻辑
- `op_kernel/arch35/stem_oam_prep_paged_kv_tiling_key.h` — TilingKey 定义
- `op_kernel/arch35/stem_oam_prep_paged_kv_tiling_data.h` — TilingData 结构体
- `op_api/aclnn_stem_oam_prep_paged_kv.h` — aclnn API 头文件
- `op_api/aclnn_stem_oam_prep_paged_kv.cpp` — aclnn API 实现
- `op_api/stem_oam_prep_paged_kv.h` — 算子注册头文件
- `op_api/stem_oam_prep_paged_kv.cpp` — 算子注册实现
- `op_extension/stem_oam_prep_paged_kv_torch.cpp` — Torch 桥接层
- `op_extension/register.cpp` — TORCH_LIBRARY 注册
- `op_extension/ops.h` — 函数声明
- `scripts/gen_data.py` — 数据生成
- `scripts/golden.py` — Golden 参考实现
- `scripts/verify_result.py` — 结果校验
- `scripts/test_torch.py` — Torch 调用测试
- `CMakeLists.txt` — 标准自定义算子构建
- `tests/ut/` — 单元测试
- `tests/st/` — 系统测试

**可选**:
- `op_graph/` — 图模式适配

---

## 附录：竞品分析 (Block-Sparse-Attention)

| 维度 | Stem OAM Prep KV | MIT BSA |
|------|-----------------|---------|
| OAM score 管线 | 集成 kFlat/vBias → mask 管线 | 无 OAM score，mask 外部预计算 |
| kFlat/vBias | 独立概念, weighted group sum + global normalize 输出 | 无此类概念，K/V 直接用于 attention |
| FP8 支持 | FP8_e4m3fn + kScaleCache × vScale 反量化 | 仅 FP16/BF16 |
| Paged KV | 两布局 paged KV (Layout A/B) + kvBlockSize ∈ {64,128} | 仅标准 paged KV |
| Weighted Group Sum | K × kScaleCache per-token, Σ over R=stemBlockSize/stemStride 层 | K 直接用于 dot-product attention |
| Anti-diagonal Flip | K 侧 stride 维度翻转 (g' = S-1-g) | 无翻转概念 |
| Global Normalize | V 侧 μ/σ over ALL k_down_len → Normalize → ReLU → BlockAvg | 无此类概念 |
| Block size | SPARSE_SIZE=128 = stemBlockSize (runtime, 推荐 128) | SPARSE_SIZE=128 |
| Stride | stemStride=16 (runtime, 推荐), R=stemBlockSize/stemStride | 无 stride 概念 |

---

## 附录：迭代规划

| 迭代 | 目标 | 代码开发 | UT开发 | ST用例 |
|------|------|---------|--------|-------|
| 迭代一 | 骨架搭建 + 双模式分核 | tilingKey=0(FP8) + 模式0+模式1 + 布局感知De-page + kScaleCache + vScale + vBias | 核心路径用例 | L0标准用例(batch=1,head=1,tokens=128) |
| 迭代二 | 性能优化 | kScaleCache bulk read per kb + V Processing double buffer | 性能对比用例 | 大shape用例(seqLen=1M) |
| 迭代三 | 规格完整 | kvBlockSize=128 验证 + 边界处理(kv_len=0) + vBias stride padding | 全覆盖用例 | 全布局 + 边界 + 变长极端case |