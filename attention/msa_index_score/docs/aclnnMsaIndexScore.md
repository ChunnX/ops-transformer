# aclnnMsaIndexScore

## 产品支持情况

| 产品 | 是否支持 |
|------|:--------:|
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |

## 功能说明

- **算子功能**：计算 MSA（MiniMax Sparse Attention）Lightning Indexer 的 block score。
  对每个 query token 与每个 KV sparse block，取该 block 内所有因果可见 token 的
  `index_q · index_k` 点乘最大值，再乘以 per-token / per-head 的正缩放。
  输出供下游 `topk` 选块使用。Prefill 与 Decode 由同一接口承载。

- **计算公式**：

$$
\texttt{score}[h, t, \beta] = \sigma[t, h]\cdot
\max_{\substack{j \in [\beta P,\ \beta P + P) \\ j \le q_{\text{pos}}(t),\ j < \texttt{kv\_len}[b]}}
\bigl\langle \mathbf{q}_{t,h},\ \mathbf{k}_{j} \bigr\rangle,
\qquad
q_{\text{pos}}(t) = \texttt{start\_loc}[b] + \bigl(t - \texttt{cu\_seq\_q}[b]\bigr)
$$

## 函数原型

每个算子分为[两段式接口](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkflow)，
必须先调用 `aclnnMsaIndexScoreGetWorkspaceSize` 接口获取入参并计算所需 workspace 大小，
再调用 `aclnnMsaIndexScore` 接口执行计算。

```cpp
aclnnStatus aclnnMsaIndexScoreGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *keyCache,
    const aclTensor *blockTable,
    const aclTensor *scaleOptional,
    const aclTensor *actualSeqLengthsQuery,
    const aclTensor *actualSeqLengthsKey,
    const aclTensor *startLoc,
    int64_t          blockSize,
    bool             fillInvalidTail,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
```

```cpp
aclnnStatus aclnnMsaIndexScore(
    void           *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

## aclnnMsaIndexScoreGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度 |
|--------|-----------|------|----------|----------|----------|------|
| query | 输入 | index query | varlen 扁平布局，$T$ 为所有请求 query token 之和 | BFLOAT16, FLOAT16 | ND | 3（$[T, H_q, D]$）|
| keyCache | 输入 | 分页 index key cache | 与 query 同 dtype；$H_{kv}$ 当前必须为 1 | BFLOAT16, FLOAT16 | ND | 4（$[NP, P, H_{kv}, D]$）|
| blockTable | 输入 | 逻辑 block → 物理 page 映射 | 元素取值需落在 $[0, NP)$ | INT32 | ND | 2（$[B, MB]$）|
| scaleOptional | 输入 | per-token / per-head 缩放 | 可为 `nullptr`（等价全 1）；非空时须逐元素为正 | FLOAT | ND | 2（$[T, H_q]$）|
| actualSeqLengthsQuery | 输入 | query 长度前缀和 | 单调不减，首元素为 0，末元素为 $T$ | INT32 | ND | 1（$[B+1]$）|
| actualSeqLengthsKey | 输入 | 每请求可见 kv 长度 | 需满足 $\lceil \texttt{kv\_len}/P \rceil \le MB$ | INT32 | ND | 1（$[B]$）|
| startLoc | 输入 | 首个 query token 在 kv 序列中的位置 | Prefill 为 prefix 长度，Decode 为 `seq_len - q_len` | INT32 | ND | 1（$[B]$）|
| blockSize | 输入 | 一个 sparse block / page 的 token 数 | 当前仅支持 128，且须等于 keyCache 第 1 维 | INT64 | - | - |
| fillInvalidTail | 输入 | 是否填充不可见 block | true 时不可见位写 $-\infty$，可直接供 topk | BOOL | - | - |
| out | 输出 | block score | 末维为 block 数而非 kv token 数 | FLOAT | ND | 3（$[H_q, T, \text{RoundUp}(MB,16)]$）|
| workspaceSize | 输出 | 所需 workspace 字节数 | - | uint64_t | - | - |
| executor | 输出 | 算子执行器 | - | aclOpExecutor** | - | - |

### 返回值

| 返回码 | 错误码 | 说明 |
|--------|--------|------|
| ACLNN_SUCCESS | 0 | 执行成功 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选入参或出参为空指针 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 数据类型、数据格式、维度或取值不满足约束 |

## aclnnMsaIndexScore

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| workspace | 输入 | device 侧 workspace 地址 |
| workspaceSize | 输入 | workspace 字节数，由第一段接口返回 |
| executor | 输入 | 算子执行器，由第一段接口返回 |
| stream | 输入 | acl stream |

### 返回值

| 返回码 | 错误码 | 说明 |
|--------|--------|------|
| ACLNN_SUCCESS | 0 | 执行成功 |
| ACLNN_ERR_PARAM_INVALID | 161002 | 参数不合法 |

## 约束说明

- `blockSize` 当前仅支持 128。
- `keyCache` 的 `numKvHeads` 当前仅支持 1（MSA index K 为全 head 共享的单头）。
- `headDim` 必须是 16 的正整数倍，且当前不超过 128；不支持 `headDim == 1`。
- `numQHeads` 必须能被 `numKvHeads` 整除。
- `scaleOptional` 非空时必须逐元素为正。
- 输出 `out` 的末维必须等于 $\text{RoundUp}(MB, 16)$，与 InferShape 推导一致。
- 本接口输出止于 block score，**不包含** `topk`。

## 调用示例

完整可运行示例见 [test_aclnn_msa_index_score.cpp](../examples/test_aclnn_msa_index_score.cpp)，
关键调用片段：

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
void *workspaceAddr = nullptr;

auto ret = aclnnMsaIndexScoreGetWorkspaceSize(
    queryT, keyCacheT, blockTableT, scaleT,
    cuSeqQT, kvLenT, startLocT,
    /*blockSize=*/128, /*fillInvalidTail=*/true,
    scoreT, &workspaceSize, &executor);
if (ret != ACL_SUCCESS) { /* handle error */ }

if (workspaceSize > 0ULL) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

ret = aclnnMsaIndexScore(workspaceAddr, workspaceSize, executor, stream);
ret = aclrtSynchronizeStream(stream);
```
