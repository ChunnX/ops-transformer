# aclnnBlockAttnResPrepare

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/block_attn_res_prepare)

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持。
- 其他产品：不支持。

## 功能说明

`BlockAttnResPrepare` 完成 Attention Residuals 的第一阶段计算。`block_res` 形状为
`[T, N, D]`，`pseudo_query` 形状为 `[S, D]`。令
`V = clamp(valid_blocks[0], 0, N)`，当 `V > 0` 时：

```text
rms_inv[t,n] = 1 / sqrt(mean(block_res[t,n,:]^2) + eps)
logit[s,t,n] = dot(pseudo_query[s,:], block_res[t,n,:]) * rms_inv[t,n]
logit_max[s,t] = max(logit[s,t,0:V])
weight[s,t,n] = exp(logit[s,t,n] - logit_max[s,t])
exp_sum[s,t] = sum(weight[s,t,0:V])
numerator[s,t,:] = sum(weight[s,t,n] * block_res[t,n,:], n=0..V-1)
```

当 `V == 0` 时三个输出均置零。有效块数小于 0 时按 0 处理，大于 `N` 时按 `N` 处理。

## 函数原型

```cpp
aclnnStatus aclnnBlockAttnResPrepareGetWorkspaceSize(
    const aclTensor *blockRes, const aclTensor *validBlocks,
    const aclTensor *pseudoQuery, aclTensor *numerator,
    aclTensor *logitMax, aclTensor *expSum, float eps,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnBlockAttnResPrepare(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

接口采用两段式调用：先获取 workspace 大小和执行器，再下发执行。

## 参数说明

| 参数 | 输入/输出 | 类型 | 格式 | Shape | 说明 |
| --- | --- | --- | --- | --- | --- |
| `blockRes` | 输入 | FLOAT | ND | `[T,N,D]` | 残差块。|
| `validBlocks` | 输入 | INT32 | ND | `[1]` | 有效块数，裁剪至 `[0,N]`。|
| `pseudoQuery` | 输入 | FLOAT | ND | `[S,D]` | 伪查询。|
| `numerator` | 输出 | FLOAT | ND | `[S,T,D]` | softmax 加权分子。|
| `logitMax` | 输出 | FLOAT | ND | `[S,T]` | softmax 最大值。|
| `expSum` | 输出 | FLOAT | ND | `[S,T]` | softmax 指数和。|
| `eps` | 输入 | FLOAT | - | - | RMS 稳定项，必须为有限正数。|
| `workspaceSize` | 输出 | UINT64 | - | - | Device workspace 字节数。|
| `executor` | 输出 | `aclOpExecutor*` | - | - | 第二段接口的执行器。|
| `workspace` | 输入 | `void*` | - | - | Device workspace。|
| `stream` | 输入 | `aclrtStream` | - | - | 执行流。|

## 约束说明

- `T >= 0`，`S >= 0`，`1 <= N <= 64`，`1 <= D <= 8192`。
- `blockRes` 与 `pseudoQuery` 的最后一维必须相同。
- 输入输出支持非连续 Tensor，接口内部执行必要的连续化和 ViewCopy。
- `T == 0` 或 `S == 0` 时返回对应形状的空输出。
- 仅支持 Ascend 950。

## 返回值

| 返回值 | 说明 |
| --- | --- |
| `ACLNN_SUCCESS` | 成功。|
| `ACLNN_ERR_PARAM_NULLPTR` | 必选参数为空。|
| `ACLNN_ERR_PARAM_INVALID` | dtype、rank、shape 或 eps 不满足约束。|
| `ACLNN_ERR_RUNTIME_ERROR` | 平台不支持或执行失败。|

## 调用示例

完整示例见 [`examples/test_aclnn_block_attn_res_prepare.cpp`](../examples/test_aclnn_block_attn_res_prepare.cpp)。

```bash
bash build.sh --run_example block_attn_res_prepare eager cust --vendor_name=custom --soc=ascend950
```
