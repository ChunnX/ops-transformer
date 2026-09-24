# aclnnKvCacheTurboQuant

## 产品支持情况

| 产品 | 支持情况 |
|---|---|
| Atlas 800T A2 | √ |
| Atlas 800I A2 | √ |

## 功能说明

- **接口功能**：面向标准 MHA/GQA KV cache 的在线向量量化压缩编码（TurboQuant encode）。对每个 128 维 KV 向量独立完成「归一化 → 正交旋转 → Lloyd-Max 标量量化（mse_bits 位宽）→ 残差 QJL 1-bit 投影」并输出 bit-packed 编码与两个范数标量。
- **计算公式**：

```
x_norm = ||x||_2
u = x / max(x_norm, 1e-30)
y = H @ u
idx[j] = Σ_k (y[j] > b_k)          # b_k = (centroids[k] + centroids[k+1]) / 2
yhat = centroids[idx]
r = y - yhat
gamma = x_norm * ||r||_2
qjl[j] = (S @ r)[j] >= 0           # ≥0 记 1
```

## 函数原型

每个算子分为两段式接口，两段之间通过 `workspace` 和 `executor` 传递信息：

```
aclnnStatus aclnnKvCacheTurboQuantGetWorkspaceSize(
    const aclTensor *kvVectors,
    const aclTensor *rotationMatrix,
    const aclTensor *qjlMatrix,
    int64_t mseBits,
    int64_t outDtype,
    const aclTensor *quantIdx,
    const aclTensor *quantQjl,
    const aclTensor *quantNorm,
    const aclTensor *quantGamma,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)

aclnnStatus aclnnKvCacheTurboQuant(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnKvCacheTurboQuantGetWorkspaceSize

- **参数说明**：

| 参数 | 维度 | 数据类型 | 输入/输出 | 说明 |
|---|---|---|---|---|
| kvVectors | 3 | bfloat16 | 输入 | [num_tokens, num_kv_heads, 128]，ND 连续 |
| rotationMatrix | 2 | float32、bfloat16 | 输入 | [128, 128]，推理期固定 |
| qjlMatrix | 2 | float32、bfloat16（与 rotationMatrix 同 dtype） | 输入 | [128, 128]，推理期固定 |
| mseBits | - | int64_t | 输入属性 | 主量化位宽，仅支持 2/3/4 |
| outDtype | - | int64_t | 输入属性 | quantNorm/quantGamma 元素类型：0=bfloat16（默认）、1=float16 |
| quantIdx | 3 | uint8 | 输出 | [num_tokens, num_kv_heads, 128·mseBits/8] |
| quantQjl | 3 | uint8 | 输出 | [num_tokens, num_kv_heads, 16] |
| quantNorm | 2 | bfloat16、float16 | 输出 | [num_tokens, num_kv_heads]，由 outDtype 决定 |
| quantGamma | 2 | bfloat16、float16 | 输出 | [num_tokens, num_kv_heads]，由 outDtype 决定 |
| workspaceSize | - | uint64_t | 输出 | 需要用户申请的 workspace 大小 |
| executor | - | aclOpExecutor* | 输出 | executor 指针 |

- **返回值**：`ACLNN_SUCCESS` / `ACLNN_ERR_PARAM_NULLPTR` / `ACLNN_ERR_PARAM_INVALID` 等，见 aclnn 错误码说明。

## aclnnKvCacheTurboQuant

- **参数说明**：

| 参数 | 说明 |
|---|---|
| workspace | GetWorkspaceSize 申请的内存地址 |
| workspaceSize | GetWorkspaceSize 返回的大小 |
| executor | GetWorkspaceSize 返回的 executor |
| stream | 任务执行流 |

- **返回值**：`ACLNN_SUCCESS` / `ACLNN_ERR_INTERNAL_ERROR`。

## 约束说明

- head_dim 固定 128，mse_bits 仅支持 2/3/4；num_kv_heads 支持 4~32；
- rotation_matrix / qjl_matrix 支持 float32 或 bfloat16（两者必须同 dtype）；
- 仅支持 ND 连续 tensor；
- 不支持 MLA latent cache。

## 调用示例

```cpp
// 1. 申请 workspace 并获取 executor
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnKvCacheTurboQuantGetWorkspaceSize(kvVectors, rotationMatrix, qjlMatrix, 3, 0,
                                       quantIdx, quantQjl, quantNorm, quantGamma,
                                       &workspaceSize, &executor);
// 2. 按 workspaceSize 申请设备内存
void *workspace = nullptr;
aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL);
// 3. 下发执行
aclnnKvCacheTurboQuant(workspace, workspaceSize, executor, stream);
```
