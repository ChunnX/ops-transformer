
## 算子需求规范

### 1. 基础信息
- **算子名称**: dense_lightning_indexer_softmax_lse
- **算子分类**: custom

### 1.1 功能描述

DenseLightningIndexerSoftmaxLse 算子用于计算 Lightning Indexer 分支的 Softmax LogSumExp 值。该算子接收 query、key 和 weight 三个输入，先通过 BatchMatmul 计算 query 与 key 的内积得分，经 ReLU 激活后与 weight 逐元素加权，再对 head 维做 ReduceSum，最后通过数值稳定的 LogSumExp 算法输出每个 query 位置的 softmax_lse 值。

该算子支持 BSND 和 TND 两种数据布局，支持 defaultMask（maskMode=0）和 rightDownCausal（maskMode=3）两种 mask 模式，并可通过 cmpRatio 参数控制压缩比。可选传入 metadata 分核信息以优化核函数调度。

### 1.2 算法参数

| 参数名 | 类型 | 默认值 | 说明 | 优先级 |
|--------|------|--------|------|--------|
| layoutQ | STRING | BSND | query_index、weight 的数据布局，支持 BSND / TND | P0 |
| layoutK | STRING | BSND | key_index 的数据布局，支持 BSND / TND | P0 |
| maskMode | INT64 | 0 | mask 模式：0=defaultMask（不做 causal mask），3=rightDownCausal（右下三角 causal mask） | P0(0) / P1(3) |
| cmpRatio | INT64 | 1 | 压缩比，取值范围 1~128。maskMode=3 且 cmpRatio≠1 时需传入 cmpResidualKOptional | P1 |

### 1.3 数学公式

$$
\text{score}[b,n,s,t] = \sum_d Q[b,s,n,d] \cdot K[b,t,0,d]
$$

$$
\text{relu\_score}[b,n,s,t] = \max(0,\ \text{score}[b,n,s,t])
$$

$$
\text{weighted}[b,n,s,t] = \text{relu\_score}[b,n,s,t] \cdot W[b,s,n]
$$

$$
x[b,s,t] = \sum_n \text{weighted}[b,n,s,t]
$$

$$
m[b,s] = \max_t x[b,s,t]
$$

$$
\text{lse}[b,s] = m[b,s] + \log\left(\sum_t \exp\left(x[b,s,t] - m[b,s]\right)\right)
$$

> 当 maskMode=3 时，在 ReLU 之前对 score 应用 rightDownCausal mask（按 cmpRatio 划分），被 mask 的位置在 ReLU 后自然为 0。

### 2. 关键特性

| 特性 | 是否需要 | 置信度 | 实现说明 | 优先级 |
|------|----------|--------|----------|--------|
| BatchMatmul | ✓ 需要 | ✓ 高 | query_index 与 key_index 在 D 维做内积，key head=1 广播到 N1 | P0 |
| ReLU 激活 | ✓ 需要 | ✓ 高 | 对 matmul 结果做 ReLU | P0 |
| 加权广播乘法 | ✓ 需要 | ✓ 高 | weight [B,S1,N1] permute+broadcast 到 [B,N1,S1,1] 后逐元素乘 | P0 |
| 多头 Reduction | ✓ 需要 | ✓ 高 | 对 N1 维做 ReduceSum | P0 |
| 数值稳定 LSE | ✓ 需要 | ✓ 高 | max 减法 + exp + sum + log（safe softmax LSE） | P0 |
| 多 dtype 支持 | ✓ 需要 | ✓ 高 | 支持 bf16/fp16/fp32 | P0 |
| TND layout | ✓ 需要 | ⚠ 中 | 支持 TND 变长序列布局，需配合 cuSeqLensQ/K | P1 |
| Causal mask | ✓ 需要 | ⚠ 中 | maskMode=3 时 rightDownCausal mask | P1 |
| 压缩比 | ✓ 需要 | ⚠ 中 | cmpRatio 1~128，maskMode=3 且 cmpRatio≠1 时需 cmpResidualK | P1 |
| Metadata 分核 | ✓ 需要 | ⚠ 中 | 可选传入预计算的分核信息 | P2 |

### 3. 算法描述

```
Algorithm: DenseLightningIndexerSoftmaxLse (Forward)
────────────────────────────────────
输入: query_index, key_index, weight,
      layoutQ, layoutK, maskMode, cmpRatio,
      cuSeqLensQ/K(可选), seqUsedQ/K(可选),
      cmpResidualK(可选), metadata(可选)
输出: softmax_lse

1. 根据 layoutQ/layoutK 确定数据布局
   - BSND: 直接使用 B, S1, S2 维度
   - TND: 通过 cuSeqLensQ/K 恢复 batch 边界

2. 对每个 batch b:
   2.1 score[b,n,s,t] = Σ_d Q[b,s,n,d] · K[b,t,0,d]
       (BatchMatmul, key head=1 广播到 N1)

   2.2 如果 maskMode == 3 (rightDownCausal):
         根据 cmpRatio 对 score 应用 rightDownCausal mask
         (被 mask 的位置在 ReLU 后自然为 0)

   2.3 relu_score[b,n,s,t] = max(0, score[b,n,s,t])

   2.4 weighted[b,n,s,t] = relu_score[b,n,s,t] · W[b,s,n]
       (W 从 [B,S1,N1] permute+broadcast 到 [B,N1,S1,1])

   2.5 x[b,s,t] = Σ_n weighted[b,n,s,t]  (ReduceSum over N1)

   2.6 m[b,s] = max_t(x[b,s,t])  (ReduceMax over S2)

   2.7 shifted[b,s,t] = x[b,s,t] - m[b,s]

   2.8 sum_exp[b,s] = Σ_t exp(shifted[b,s,t])  (ReduceSum over S2)

   2.9 lse[b,s] = m[b,s] + log(sum_exp[b,s])

3. return lse
```

### 4. 数据流图

```
  query_index [B,S1,N1,D]     key_index [B,S2,1,D]    weight [B,S1,N1]
  (TND: [T1,N1,D])            (TND: [T2,1,D])         (TND: [T1,N1])
       │                          │                       │
       └──────────┬───────────────┘                       │
                  ▼                                       │
          ┌────────────────┐                              │
          │  BatchMatmul   │  (对 D 求内积, key head=1    │
          │  (over D)      │   广播到 N1)                 │
          └───────┬────────┘                              │
                  │                                       │
                  ▼                                       │
    ┌───────────────────────────┐                         │
    │  maskMode=3? → Causal Mask│  (rightDownCausal,      │
    │  maskMode=0? → 无 mask    │   按 cmpRatio 划分)      │
    └───────────┬───────────────┘                         │
                │                                         │
                ▼                                         │
          ┌────────────────┐                              │
          │     ReLU       │                              │
          └───────┬────────┘                              │
                  │                                       │
                  ▼                                       │
        relu_score [B, N1, S1, S2]  ←─── w_b [B,N1,S1,1] ◄┘
                  │                      (permute+broadcast)
                  ▼
          ┌────────────────┐
          │  Mul(broadcast)│
          └───────┬────────┘
                  │
                  ▼
         weighted [B, N1, S1, S2]
                  │
                  ▼
          ┌────────────────┐
          │ ReduceSum(N1)  │
          └───────┬────────┘
                  │
                  ▼
            x [B, S1, S2]
                  │
                  ├────────────────────────┐
                  ▼                        │
          ┌────────────────┐               │
          │ ReduceMax(S2)  │               │
          └───────┬────────┘               │
                  │                        │
                  ▼                        ▼
            m [B, S1]          ┌─────────────────────┐
                  │            │   x - m (broadcast)  │
                  ▼            └──────────┬──────────┘
            ┌──────────────────────────────┘
                  │
                  ▼
          shifted [B, S1, S2]
                  │
                  ▼
          ┌─────────────────────┐
          │ exp → ReduceSum(S2) │
          └─────────┬───────────┘
                    │
                    ▼
            sum_exp [B, S1]
                    │
                    ▼
          ┌─────────────────────┐
          │  m + log(sum_exp)    │
          └─────────┬───────────┘
                    │
                    ▼
          lse [B, S1]  (TND: [T1,])
```

### 5. 输入输出规格

**输入规格**:

| 变量 | Shape (BSND) | Shape (TND) | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------------|-------------|-------|--------|--------|------|
| query_index | [B, S1, N1, D] | [T1, N1, D] | bfloat16 / float16 / float32 | B, S1, N1, D | ✓ 高 | Lightning Indexer 分支的 query 输入，不支持空 Tensor |
| key_index | [B, S2, 1, D] | [T2, 1, D] | bfloat16 / float16 / float32 | B, S2, D | ✓ 高 | Lightning Indexer 分支的 key 输入，head 维固定为 1，不支持空 Tensor |
| weight | [B, S1, N1] | [T1, N1] | bfloat16 / float16 / float32 | B, S1, N1 | ✓ 高 | Indexer logits 的 head 权重，不支持空 Tensor |
| cuSeqLensQOptional | (B+1,) | (B+1,) | INT32 | B | ⚠ 中 | TND layout 下 q 的累积序列长度，首元素为 0，layoutQ=TND 时必须传入 |
| cuSeqLensKOptional | (B+1,) | (B+1,) | INT32 | B | ⚠ 中 | TND layout 下 k 的累积序列长度，首元素为 0，layoutK=TND 时必须传入 |
| seqUsedQOptional | (B,) | (B,) | INT32 | B | ⚠ 中 | 预留字段，表示每个 batch 实际使用的 q 长度 |
| seqUsedKOptional | (B,) | (B,) | INT32 | B | ⚠ 中 | 预留字段，表示每个 batch 实际使用的 k 长度 |
| cmpResidualKOptional | (B,) | (B,) | INT32 | B | ⚠ 中 | maskMode=3 且 cmpRatio≠1 时需要传入 |
| metadataOptional | (x,) | (x,) | INT32 | - | ⚠ 中 | 由 Metadata 算子生成的分核信息，传入后 kernel 将使用 metadata 中的分核切分 |
| layoutQ | - | - | STRING | - | ⚠ 中 | query_index、weight 的 layout，仅支持 BSND 和 TND |
| layoutK | - | - | STRING | - | ⚠ 中 | key_index 的 layout，仅支持 BSND 和 TND |
| maskMode | - | - | INT64 | - | ⚠ 中 | mask 模式：0=defaultMask，3=rightDownCausal，其余不支持 |
| cmpRatio | - | - | INT64 | - | ⚠ 中 | 压缩比，取值范围 1~128 |

**输出规格**:

| 变量 | Shape (BSND) | Shape (TND) | Dtype | 动态轴 | 置信度 | 说明 |
|------|-------------|-------------|-------|--------|--------|------|
| softmax_lse | [B, S1] | [T1,] | 与输入一致 | B, S1 | ✓ 高 | Softmax LogSumExp 输出 |

### 6. 数据类型支持

| Dtype | 支持 | atol | rtol | 备注 |
|-------|------|------|------|------|
| float32 | ✓ | 0.001 | 0.001 | 默认精度 |
| bfloat16 | ✓ | 0.1 | 0.005 | 训练常用精度 |
| float16 | ✓ | 0.1 | 0.005 | 训练常用精度 |

### 7. 精度要求
- **atol**: float32=0.001, bfloat16/float16=0.1
- **rtol**: float32=0.001, bfloat16/float16=0.005

### 8. 动态轴说明
- **动态轴**: ['B', 'S1', 'S2', 'N1', 'D']
- **轴含义**: B=batch size, S1=query 序列长度, S2=key 序列长度, N1=query head 数, D=head 维度
- **取值范围**: B=[1, 256], S1=[1, 8192], S2=[1, 524288], N1=[1, 128], D=[1, INT32_MAX]

### 9. 边界条件处理
- **零值**: normal（正常计算）
- **极值**: normal（正常计算，LSE 通过 max 减法保证数值稳定性）
- **NaN/Inf**: normal（正常计算）

### 10. 性能要求
- **性能目标**: 首跑精度成功性能的 2 倍

### 11. 参考信息
- **参考实现**: `test_dense_lightning_indexer_softmax_lse.py` 中的 `golden_compute` 函数
- **论文**: 无
- **类似算子**: aclnnSparseLightningIndexerKLLossGrad（参数接口参考）

### 12. 应用场景
- **目标模型**: Lightning Indexer 注意力机制相关模型
- **使用位置**: Attention 分支中的 Lightning Indexer 前向计算

**典型配置**（建议至少提供一个，用于下游 golden 验证和设计方案生成）:

| 配置名称 | 类型 | 优先级 | 参数 | 输入 Shape | 输出 Shape | 说明 |
|----------|------|--------|------|------------|------------|------|
| default_bf16_bsnd | 功能 | P0 | B=2,S1=128,S2=256,N1=8,D=64,dtype=bf16,layout=BSND,maskMode=0 | q[2,128,8,64] k[2,256,1,64] w[2,128,8] | lse[2,128] | 默认 bf16 BSND 用例 |
| default_fp16_bsnd | 功能 | P0 | B=2,S1=128,S2=256,N1=8,D=64,dtype=fp16,layout=BSND,maskMode=0 | q[2,128,8,64] k[2,256,1,64] w[2,128,8] | lse[2,128] | 默认 fp16 BSND 用例 |
| default_fp32_bsnd | 功能 | P0 | B=2,S1=128,S2=256,N1=8,D=64,dtype=fp32,layout=BSND,maskMode=0 | q[2,128,8,64] k[2,256,1,64] w[2,128,8] | lse[2,128] | 默认 fp32 BSND 用例 |
| causal_tnd_bf16 | 功能 | P1 | B=2,S1=128,S2=256,N1=8,D=64,dtype=bf16,layout=TND,maskMode=3,cmpRatio=4 | q[256,8,64] k[512,1,64] w[256,8] | lse[256] | TND+causal mask 用例 |

---
*生成时间: 2025-07-14*
*确认状态: 已确认*
*置信度说明: ✓ 高（自身知识库/框架知识） / ⚠ 中（外部材料提取，需确认）*