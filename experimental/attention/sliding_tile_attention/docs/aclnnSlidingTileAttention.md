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
| `q` | 输入 | Query，连续 BNSD，shape 为 `[B,8,S,64]` | FLOAT16/BFLOAT16，ND |
| `k` | 输入 | Key，shape、dtype 与 `q` 一致 | FLOAT16/BFLOAT16，ND |
| `v` | 输入 | Value，shape、dtype 与 `q` 一致 | FLOAT16/BFLOAT16，ND |
| `windowSize` | 输入 | 一个 `[t,h,w]` 三元组，或按 8 个 head 展平的 24 个整数 | INT64 数组 |
| `textLength` | 输入 | 固定 case 的 text 长度；有 text 场景为 128，无 text 场景为 0 | INT64 |
| `hasText` | 输入 | 是否包含序列尾部 text token | BOOL |
| `seqShape` | 输入 | 画布枚举：1=`30x48x80`，2=`36x48x48`，3=`18x48x80` | INT64 |
| `output` | 输出 | 输出 Tensor，shape、dtype 与 `q` 一致 | FLOAT16/BFLOAT16，ND |
| `workspaceSize` | 输出 | 执行所需 workspace 字节数 | UINT64 |
| `executor` | 输出 | 封装当前计算流程的 ACLNN executor | `aclOpExecutor*` |
| `workspace` | 输入 | Device workspace；大小由第一段接口返回 | `void*` |
| `stream` | 输入 | ACL runtime stream | `aclrtStream` |

## 约束说明

根据任务书验收范围，本次交付只支持 `docs/self_validation.md` 声明的 18 个固定 case：

- `q/k/v/output` 必须是相同 shape、dtype 的连续 rank-4 BNSD Tensor。
- dtype 仅支持 FLOAT16 和 BFLOAT16。
- head 数固定为 8，head dimension 固定为 64。
- tile 固定为 `[6,8,8]`。
- `windowSize` 只接受固定 case 中的窗口模式；单个 `[3,3,3]` 会在 L2 广播为 8 个 head。
- 有 text 的固定 case 使用 `seqShape=1`、`textLength=128`；无 text case 使用 `textLength=0`。
- 不支持动态扩展。范围外 metadata 返回 `ACLNN_ERR_PARAM_INVALID`，不做任何 runtime fallback。

## 返回值

- `ACLNN_SUCCESS`：成功。
- `ACLNN_ERR_PARAM_NULLPTR`：必需指针为空。
- `ACLNN_ERR_PARAM_INVALID`：dtype、shape、window 或其他 metadata 不在固定验收范围内。
- 其他返回值：ACLNN executor 或 runtime 执行失败。

## 调用示例

完整 C++ 调用和固定 18-case 复跑入口位于：

```text
experimental/attention/sliding_tile_attention/examples/
```

```bash
bash experimental/attention/sliding_tile_attention/examples/run_18cases.sh
```
