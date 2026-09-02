# aclnnSlidingTileAttention

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Atlas A3 训练/推理系列产品 | 是 |
| Atlas A2 训练系列产品、Atlas 800I A2 推理产品、A200I A2 Box 异构组件 | 是 |

## 功能说明

本接口执行 FastVideo 风格的三维滑动 tile attention。图像 query 只关注当前 head 对应窗口覆盖的图像 KV；
窗口在 tile 网格边界处 clamp。有 text 时，图像 query 同时关注全部 text KV，text query 关注完整有效序列。

```text
attentionOut = softmax((Q @ K^T) / sqrt(D), masked_by_sliding_tile_window) @ V
```

## 函数原型

算子采用两段式接口。先获取 workspace 和 executor，再执行算子。

```cpp
aclnnStatus aclnnSlidingTileAttentionGetWorkspaceSize(
    const aclTensor* q,
    const aclTensor* k,
    const aclTensor* v,
    const aclIntArray* windowSize,
    int64_t textLength,
    bool hasText,
    int64_t seqShape,
    aclTensor* output,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```cpp
aclnnStatus aclnnSlidingTileAttention(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## 参数说明

| 参数 | 输入/输出 | 说明 | 数据类型/格式 |
| --- | --- | --- | --- |
| `q` | 输入 | Query，rank-4 BNSD，shape 为 `[B,N,S,D]`；可非连续 | FLOAT16/BFLOAT16，ND |
| `k` | 输入 | Key，shape、dtype 与 `q` 一致；可非连续 | FLOAT16/BFLOAT16，ND |
| `v` | 输入 | Value，shape、dtype 与 `q` 一致；可非连续 | FLOAT16/BFLOAT16，ND |
| `windowSize` | 输入 | 一个 `[t,h,w]` 三元组，或按 N 个 head 展平的 `3*N` 个值；每个三元组的三个值均为正奇数 | INT64 数组 |
| `textLength` | 输入 | 序列尾部 text token 数；必须非负并与 `hasText` 一致 | INT64 |
| `hasText` | 输入 | 是否包含序列尾部 text token | BOOL |
| `seqShape` | 输入 | 画布枚举：1=`30x48x80`，2=`36x48x48`，3=`18x48x80` | INT64 |
| `output` | 输出 | 输出 Tensor，shape、dtype 与 `q` 一致 | FLOAT16/BFLOAT16，ND |
| `workspaceSize` | 输出 | 执行所需 workspace 字节数；空 Tensor 返回 0 | UINT64 |
| `executor` | 输出 | 封装当前计算流程的 ACLNN executor | `aclOpExecutor*` |
| `workspace` | 输入 | Device workspace；大小由第一段接口返回 | `void*` |
| `stream` | 输入 | ACL runtime stream | `aclrtStream` |

## 约束说明

- `q/k/v/output` 必须是相同 shape、dtype 的 rank-4 BNSD Tensor。
- dtype 仅支持 FLOAT16 和 BFLOAT16。
- 非空输入满足 `B > 0`、`1 <= N <= 128`、`D > 0` 且 `D % 8 == 0`。
- tile 固定为 `[6,8,8]`。`seqShape=1` 时，`S` 可包含任务书定义的 text storage padding：
  `0 <= S-image_tokens <= 512`、`0 <= textLength <= S-image_tokens` 且 `hasText=true`；padding query 产生
  输出，但不作为 K/V。`seqShape=2/3` 时 `S` 必须等于对应画布 token 数、`textLength=0`、`hasText=false`。
- `windowSize` 长度必须为 3 或 `3*N`；长度 3 广播到全部 head，每个三元组的三个值均须为正奇数。
- `textLength >= 0`，且 `hasText == (textLength > 0)`。
- ACLNN L2 对 `q/k/v` 显式调用 `Contiguous`。非连续输入会 materialize 为连续 Tensor，调用方无需预先拷贝。
- shape 一致的任意空 Tensor 在 `GetWorkspaceSize` 直接返回 `ACLNN_SUCCESS`，workspace 为 0，不进入 L0，
  不创建 AI Core launcher，也不发射 kernel。
- `dtype=BF16 && D=64` 选择 key 1 lean 变体；已证明的 FP16 shape family 选择 key 2/3/5/7/8/9；
  D256 padded-text partial/merge 选择 key 10/11；其余合法 dtype/D 选择 key 0 通用变体。变体内部可继续按
  runtime B/N/画布/window/capacity 选择固定 task decode、grouped/compact 或动态数据流。所有选择仅依赖
  运行时 shape/dtype/属性，不依赖 case id、输入值或已知 workload。
- 范围外 metadata 返回 `ACLNN_ERR_PARAM_INVALID`，不做 framework、vendor 或其他 backend fallback。

## 返回值

- `ACLNN_SUCCESS`：成功；空 Tensor 快返也使用该返回值。
- `ACLNN_ERR_PARAM_NULLPTR`：必需指针为空。
- `ACLNN_ERR_PARAM_INVALID`：dtype、rank、shape、window 或其他 metadata 不满足接口契约。
- 其他返回值：ACLNN executor 或 runtime 执行失败。

## 调用示例

完整 C++ 调用、任务书 50-case、原 18-case 保护集及动态/非连续/空 Tensor 代表性检查入口位于：

```text
experimental/attention/sliding_tile_attention/examples/
```

```bash
bash experimental/attention/sliding_tile_attention/examples/run_taskbook_cases.sh
bash experimental/attention/sliding_tile_attention/examples/run_22cases.sh
```

taskbook 入口执行原始 `all_sta.json` 的全部 50 个设备 case；reviewer 入口执行自验证表中的 22 个额外设备
case。两者都执行空 Tensor API 检查、解析精度采样和同输入 bitwise 确定性检查，不代表接口完整动态支持域。
当前源码摘要上的结果分别为 50/50 与 22/22；未列入相应表的 workload 不由该日志给出性能结论。任务书历史
性能数值和本地 CSV 延迟均不作为接口性能结论。原始证据见 [`docs/evidence`](evidence)。
