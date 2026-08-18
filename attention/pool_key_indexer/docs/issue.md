已获取算子全部资料（docs/aclnnKpoolIndexer.md、development_blueprint.md、op_host、op_kernel、tests/atk、torch_ops_extension）。下面给出参考 #4122 格式完善后的完整 issue 内容：

---

Thanks for sending an requirement! Please fill in the following template to help quickly solve your problem.

### Backgroud（背景信息）

GLM next 模型新增 pool 概念的稀疏注意力机制，用于基于 key pool 的稀疏注意力场景下的稀疏索引选择。需要新增融合算子，融合反量化、分数计算与选择、索引展开与追加等计算，提升昇腾上的训练、推理性能。

### Benefit / Necessity （价值/作用）

将 key 多个连续 token 打包成一个 pool（池）作为本算子的输入，以 pool 为单位计算注意力相关性分数并选取 top-k 位置，从而在保持稀疏注意力优势的同时减少索引开销。该算子适用于基于 key pool 的稀疏注意力场景下的稀疏索引选择。支持 FLOAT16、BFLOAT16 浮点输入，并支持配合反量化系数的 FLOAT8_E4M3FN 量化输入（FP8 per-token-head 或 mxFP8 量化模式）。

将「反量化 + 缩放点积 + ReLU + 多头权重聚合 + 可见性过滤 + Top-k + 索引展开与尾块追加」融合为单算子，中间结果驻留 UB/寄存器，减少 GM 往返；稀疏索引与分数一次写出，省掉多个小算子的 kernel launch 与中间落盘开销。

### Design（设计方案）

算子名：`PoolKeyIndexer`
aclnn 接口名：`aclnnPoolKeyIndexer`
torch 接口名：`cann_ops_transformer.pool_key_indexer`

实现路径：基于 AscendC Cube+Vector 协同编程模型开发，Cube 核完成 Q⊗K^T 的 Mmad，Vector 核完成 ReLU / 权重聚合 / 因果掩码 / TopK / 索引展开与尾块追加。arch35（Ascend 950）额外支持 FP8 反量化路径。支持 Ascend 950PR/950DT（DAV_3510, arch35）、Atlas A3 训练/推理系列（arch22）、Atlas A2 训练/推理系列（arch22）。

**计算过程：**

设：
- batch 索引 $b \in [0, B)$，query 位置 $i \in [0, S_1)$，pool 位置 $j \in [0, S_2)$；
- $L_{orig}[b] = S_2 \cdot pool\_size + poolTailK[b]$（原始 token 总数），$poolTailK[b] \in [0, pool\_size-1]$；
- $sparseCount = \lfloor topk / pool\_size \rfloor$（选中的 pool 数）。

完整计算流程分为输入反量化、分数计算与选择、索引展开与追加三个部分。

**1) 输入反量化（仅量化场景，quantMode 为 0 或 1）**

当 quantMode 为 -1（不量化）时，直接使用原始输入：

$$
Q = query,\quad K_{pool} = poolKey
$$

当 quantMode 为 0 或 1（量化）时，先对 FLOAT8_E4M3FN 输入结合反量化系数进行反量化：

$$
Q = \text{Dequant}(query, q\_descale)
$$

$$
K_{pool} = \text{Dequant}(poolKey, k\_descale)
$$

其中 quantMode 为 0 表示 FP8 per-token-head 量化模式，$Q[t,h,:] = query[t,h,:].toFloat() \times q\_descale[t,h]$，反量化系数为 FLOAT 类型；quantMode 为 1 表示 mxFP8 量化模式，每 32 元素共享一个 FLOAT8_E8M0 指数 scale，$Q = query \times 2^{scale-127}$，反量化系数为 FLOAT8_E8M0 类型。$K_{pool}$ 同理。

**2) 分数计算与选择**

1. 缩放点积：

$$
S[i,j] = Q \otimes K_{pool}^T \odot \frac{1}{\sqrt{D}} = \frac{1}{\sqrt{D}} \sum_{d=0}^{D-1} Q[i,:,d] \cdot K_{pool}[j,:,d]
$$

2. ReLU 激活，过滤负相关信号：

$$
S'[i,j] = \text{ReLU}(S[i,j]) = \max(0, S[i,j])
$$

3. 多头权重聚合：

$$
PoolScores[i,j] = W \otimes S' = \sum_{h=0}^{N_1-1} W[i,h] \cdot S'[i,h,j]
$$

4. 可见性过滤：根据因果性和 padding 状态标记有效候选 pool，将无效 pool 分数置为 $-\infty$。

- maskMode=0（defaultMask）：仅按 actualSeqQ/actualSeqK 过滤 padding，有效 pool 保留。
- maskMode=3（rightDownCausal）：因果掩码，pool $j$ 覆盖 token 范围 $[j \cdot pool\_size, (j+1)\cdot pool\_size - 1]$，对 query 位置 $i$ 的全局位置 $g_i = L_{orig}[b] - S_1 + i$，有效条件：

$$
(j+1) \cdot pool\_size - 1 \le g_i \quad \Leftrightarrow \quad j \le \left\lfloor \frac{g_i + 1}{pool\_size} \right\rfloor - 1
$$

最后一个 pool（含 poolTailK 尾部）的 end $= L_{orig}[b] - 1$。

5. Top-k 选择（pool 级别）：

$$
TopK\_pool[i] = \text{Top-}k\bigl(PoolScores[i,:]\bigr), \quad |TopK\_pool[i]| = sparseCount = \lfloor topk / pool\_size \rfloor
$$

即选取前 $sparseCount$ 个最大分数对应的 pool 索引。

**3) 索引展开与尾块追加**

将选中的 pool 索引展开为原始 token 索引，并追加不完整 pool 的尾部有效 token（先展开后追加）：

展开（位置 $[0, topk)$）：

$$
\text{Expand}: \quad tokenIndices[i, k \cdot P + p] = TopK\_pool[i,k] \cdot pool\_size + p, \quad p \in [0, pool\_size)
$$

有效展开数 $= validS2Len \cdot pool\_size$（validS2Len 为因果掩码后有效 pool 数），超出部分填 -1。

尾块追加（poolTailK > 0 时，位置 $[topk, topk+pool\_size-1)$）：

$$
visibleTailK = \begin{cases} poolTailK[b], & maskMode = 0 \\ \max\bigl(0,\ \min(poolTailK[b],\ g_i - topk + 1)\bigr), & maskMode = 3 \end{cases}
$$

$$
tokenIndices[i, topk + t] = L_{orig}[b] - poolTailK[b] + t, \quad t \in [0, visibleTailK)
$$

未被命中的位置保持 -1。

**4) 输出**

$$
sparseIndices = tokenIndices,\quad SparseValues = PoolScores[TopK\_pool]
$$

当 return_value 为 1 时输出 $SparseValues$（选中 pool 的分数值）；return_value 为 0 时不输出。

**核心运算 = FP8 反量化（Cast+Mul，仅量化）+ Cube Mmad 缩放点积 + Vector ReLU + 权重聚合 ReduceSum + 因果掩码过滤 + TopK 排序选择 + 索引展开（Muls/Brcb/Add 向量化）+ 尾块追加 + 分页/直通写出。**

符号说明如下：

- $SparseIndices$：输出的稀疏 token 索引。
- $SparseValues$：输出的选中 pool 分数（return_value 为 1 时输出）。
- $PoolScores$：每个 pool 的综合分数。
- $TopK\_pool$：被选中的 top-k pool 索引。
- $sparseCount$：选中的 pool 数，$sparseCount = \lfloor topk / pool\_size \rfloor$。
- $W$：多头权重投影矩阵。
- $\otimes$：矩阵乘法（matmul）。
- $\text{ReLU}$：激活函数 $ReLU(x) = \max(0, x)$。
- $Q$：Query 向量（量化场景下为反量化后的 Query）。
- $K_{pool}$：Pool 的加权聚合 key，$K_{pool}^T$ 为其最后两维转置（量化场景下为反量化后的 Key）。
- $\odot$：逐元素乘法（Hadamard 积）。
- $headDim$：头维度，缩放因子 $\frac{1}{\sqrt{headDim}}$ 防止点积过大。
- $L_{orig}$：原始 token 总数，$L_{orig} = S_2 \cdot pool\_size + poolTailK$。
- $poolTailK$：每个 batch 尾部不完整 pool 的有效 token 数。
- $validS2Len$：因果掩码后的有效 pool 数。
- $visibleTailK$：因果可见的尾块 token 数。
- $query$、$poolKey$：原始输入 tensor（量化场景下为 FLOAT8_E4M3FN）。
- $q\_descale$、$k\_descale$：Query 与 Key 的反量化系数。

### 算子参数说明

> $B$ = Batch Size，$S_1$ = query 的序列长度，$S_2$ = poolKey 的序列长度（pool 数），$T_1$/$T_2$ = query/poolKey 的 T 累加和，$N_1$ = query 头数，$N_2$ = poolKey 头数（固定 1），$D$ = headDim（固定 128），$blockNum$ = PA block 总数，$blockSize$ = 单 block token 数。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
|--------|----------------|------|----------|----------|-------------|
| query | 输入 | 公式中的输入 Q，不支持空 tensor | FLOAT16、BFLOAT16、FLOAT8_E4M3FN(仅950) | ND | BSND: $(B,S_1,N_1,D)$ / TND: $(T_1,N_1,D)$ |
| pool_key | 输入 | 公式中的输入 $K_{pool}$，支持 0 轴非连续，dtype 需与 query 一致 | FLOAT16、BFLOAT16、FLOAT8_E4M3FN(仅950) | ND | PA_BBND: $(blockNum,blockSize,N_2,D)$ / BSND: $(B,S_2,N_2,D)$ / TND: $(T_2,N_2,D)$ |
| weights | 输入 | 公式中的输入 W，不支持空 tensor，dtype 需与 query 一致 | FLOAT16、BFLOAT16 | ND | BSND: $(B,S_1,N_1)$ / TND: $(T_1,N_1)$ |
| pool_tail_k | 输入 | 每个 batch 尾部不完整 pool 的有效 token 数，取值 $[0, pool\_size-1]$，0 表示无尾部有效 token | INT32 | ND | $(B,)$ |
| actual_seq_q | 可选输入 | 每个 batch 的 Query 有效 token 数；TND 必传且为前缀和 | INT32 | ND | $(B,)$ |
| actual_seq_k | 可选输入 | 每个 batch 的 Key 有效 token 数；TND/PA 必传。TND 为前缀和，PA 为非前缀和 | INT32 | ND | $(B,)$ |
| block_table | 可选输入 | PageAttention block 映射表，PA 场景必传 | INT32 | ND | $(B, maxBlockNumPerSeq)$ |
| q_descale | 可选输入 | Query 反量化系数，量化场景必传。quantMode=0 为 FLOAT，quantMode=1 为 FLOAT8_E8M0 | FLOAT、FLOAT8_E8M0 | ND | BSND: $(B,S_1,N_1)$ / TND: $(T_1,N_1)$ |
| k_descale | 可选输入 | Key 反量化系数，量化场景必传 | FLOAT、FLOAT8_E8M0 | ND | BSND: $(B,S_2,N_2)$ / TND: $(T_2,N_2)$ / PA: $(blockNum,blockSize,N_2)$ |
| topk | 属性 | 展开后保留的 token 数量，需满足 $topk \bmod pool\_size = 0$ | INT | - | - |
| pool_size | 属性 | 每个 pool 包含的 token 数 | INT | - | - |
| layout_q | 属性 | Query 数据排布，当前支持 BSND、TND | STRING | - | - |
| layout_k | 属性 | Key 数据排布，当前支持 PA_BBND、BSND、TND | STRING | - | - |
| mask_mode | 属性 | mask 模式：0=defaultMask，3=rightDownCausal（右顶点下三角因果） | INT | - | - |
| quant_mode | 属性 | 量化模式：-1=不量化，0=FP8 per-token-head，1=mxFP8 | INT | - | - |
| return_value | 属性 | 是否输出 sparse_values | BOOL | - | - |
| sparse_indices | 输出 | 展开后的稀疏 token 索引，无效位置填 -1 | INT32 | ND | BSND: $(B,S_1,topk+pool\_size-1)$ / TND: $(T_1,topk+pool\_size-1)$ |
| sparse_values | 输出 | 选中 pool 的分数值，仅 return_value=true 时输出有效结果 | FLOAT | ND | BSND: $(B,S_1,topk/pool\_size)$ / TND: $(T_1,topk/pool\_size)$ |

### 当前支持范围

| 项 | 范围 |
|---|---|
| 芯片 | Ascend 950PR / Ascend 950DT（arch35）、Atlas A3 训练/推理系列（arch22）、Atlas A2 训练/推理系列（arch22） |
| headDim | 固定 128 |
| $N_1$（query 头数） | $\le 64$ |
| $N_2$（poolKey 头数） | 固定 1 |
| $(topk, pool\_size)$ | pool_size $\in [1,128]$；topk $\in [1,2048] \cup \{3072,4096,5120,6144,7168,8192\}$，且需满足 $topk \bmod pool\_size = 0$ |
| maskMode | 0（defaultMask）、3（rightDownCausal） |
| quantMode | -1（不量化）；0（FP8 per-token-head，仅 arch35）；1（mxFP8，仅 arch35） |
| 数据布局 | query: BSND / TND；poolKey: BSND / TND / PA_BBND |
| block_size（PA） | 16 的倍数，最大 1024 |
| FP8 量化 | 仅 Ascend 950PR/950DT 支持 FLOAT8_E4M3FN 输入与 FLOAT8_E8M0 系数；A2/A3 不支持量化 |
| 确定性 | 默认确定性实现 |
| 非连续 | poolKey 支持 0 轴非连续 |

### 约束说明

- query、poolKey、weights 的数据类型需保持一致；poolKey 的 D 必须与 query 的 headDim（128）一致。
- 不支持 query、poolKey 中对 $S_1$、$S_2$ pad 无效数据。
- 维度一致性：weights/qDescale 的各维度须与 query 对应维度一致；kDescale 的各维度须与 poolKey 对应维度一致；poolTailK、actualSeqK 的元素数须等于 B。
- 前缀和约束：TND 时 actualSeqQ 末值 $= T_1$，actualSeqK 末值 $= T_2$。
- 输出维度约束：sparseIndices/sparseValues 的 B、$S_1$（或 $T_1$）须与 query 对应维度一致。
- PA 场景：blockTable 第一维 $= B$，第二维 $\ge maxBlockNumPerSeq$（每 batch 最大 actualSeqK 对应的 block 数）。

### 接口原型

```
cann_ops_transformer.pool_key_indexer(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    q_descale=None,
    k_descale=None,
    layout_q="BSND",
    layout_k="BSND",
    topk=2048,
    pool_size=16,
    mask_mode=3,
    quant_mode=-1,
    return_value=False,
) -> (Tensor sparse_indices, Tensor sparse_values)
```

参考 aclnn 两段式接口：

```cpp
aclnnStatus aclnnPoolKeyIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *poolKey, const aclTensor *weights,
    const aclTensor *poolTailK,
    const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *qDescaleOptional, const aclTensor *kDescaleOptional,
    char *layoutQueryOptional, char *layoutKeyOptional,
    int64_t topk, int64_t poolSize, int64_t maskMode, int64_t quantMode, bool returnValue,
    const aclTensor *sparseIndicesOut, const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
```