# aclnnKvCacheTurboQuant

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：不支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：√
- <term>Atlas 200I/500 A2 推理产品</term>：×
- <term>Atlas 推理系列产品</term>：×
- <term>Atlas 训练系列产品</term>：×

## 功能说明

- 接口功能：参考 TurboQuant 论文，对标准 MHA/GQA KV cache 的 K/V 向量做在线向量量化压缩（encode）。两阶段量化：随机正交旋转 + 逐维 `mseBits` bit MSE 主量化（Lloyd-Max 质心），残差经 `||r||_2` 归一化后做高斯随机投影 + 1-bit 符号量化（QJL）。同时输出输入向量范数 `x_norm` 与原尺度残差范数 `gamma = x_norm * ||r||_2`。
- 仅支持标准 MHA/GQA KV cache，不支持 MLA latent cache。
- 计算公式（每个 `(token, kv_head)` 向量 `x ∈ R^128`）：

  ```python
  x_norm = ||x||_2
  u = x / x_norm                       # 零向量输出 0
  y = u @ rotation_matrix.T
  idx = (y[..., None] > boundaries).sum(-1)      # mseBits 主量化索引
  y_hat = centroids[idx]
  r = y - y_hat
  gamma = x_norm * ||r||_2
  qjl = (r / ||r||_2) @ qjl_matrix.T >= 0        # 1-bit 符号
  ```

- bit 打包：idx 每 8 维打包为 `mseBits` 个字节（组内第 `lane` 维占 `lane*mseBits` 起始 bit，多字节小端）；qjl 每 8 维打包为 1 字节（`lane` 维占第 `lane` bit）。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnKvCacheTurboQuantGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnKvCacheTurboQuant` 接口执行计算。

```cpp
aclnnStatus aclnnKvCacheTurboQuantGetWorkspaceSize(
    const aclTensor *kvVectors,
    const aclTensor *rotationMatrix,
    const aclTensor *qjlMatrix,
    int64_t          mseBits,
    aclTensor       *quantIdx,
    aclTensor       *quantQjl,
    aclTensor       *quantNorm,
    aclTensor       *quantGamma,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)

aclnnStatus aclnnKvCacheTurboQuant(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnKvCacheTurboQuantGetWorkspaceSize

### 参数说明

| 参数名         | 输入/输出 | 描述                                       | 使用说明                                                     | 数据类型  | 数据格式 | 维度(shape)                                  | 非连续Tensor |
| -------------- | --------- | ------------------------------------------ | ------------------------------------------------------------ | --------- | -------- | -------------------------------------------- | ------------ |
| kvVectors      | 输入      | 待量化的 K 或 V 向量。                      | 不支持空 Tensor。`num_kv_heads` 支持 4~32，`head_dim` 固定 128。 | BFLOAT16  | ND       | [num_tokens, num_kv_heads, 128]              | ×            |
| rotationMatrix | 输入      | 正交旋转矩阵 H（Hadamard 或随机正交矩阵）。 | 推理期间固定。                                               | FLOAT32   | ND       | [128, 128]                                   | ×            |
| qjlMatrix      | 输入      | 残差投影矩阵 S，元素服从固定高斯分布。      | 推理期间固定。`qjl_dim` 支持 (0, 128] 且为 8 的倍数。          | FLOAT32   | ND       | [qjl_dim, 128]                               | ×            |
| mseBits        | 输入      | MSE 主量化位宽。                            | 支持 2/3/4，默认 3。                                         | INT64     | -        | -                                            | -            |
| quantIdx       | 输出      | 主量化编码（bit-packed）。                  | 不支持空 Tensor。                                            | UINT8     | ND       | [num_tokens, num_kv_heads, 16*mseBits]       | ×            |
| quantQjl       | 输出      | 残差 QJL 符号编码（bit-packed）。           | 不支持空 Tensor。                                            | UINT8     | ND       | [num_tokens, num_kv_heads, qjl_dim/8]        | ×            |
| quantNorm      | 输出      | 输入向量 L2 范数。                          | 不支持空 Tensor。                                            | BFLOAT16  | ND       | [num_tokens, num_kv_heads]                   | ×            |
| quantGamma     | 输出      | 原尺度残差范数 `x_norm * ||r||_2`。         | 不支持空 Tensor。                                            | BFLOAT16  | ND       | [num_tokens, num_kv_heads]                   | ×            |
| workspaceSize  | 输出      | 返回需要在 Device 侧申请的 workspace 大小。 | 用于存放逐向量归一化残差（fp32，`num_tokens*num_kv_heads*128*4` 字节）。 | - | - | -                                     | -            |
| executor       | 输出      | 返回 op 执行器，包含了算子计算流程。        | -                                                            | -         | -        | -                                            | -            |

### 返回值

aclnnStatus：返回 0 表示成功，返回其他值表示失败。

## aclnnKvCacheTurboQuant

### 参数说明

| 参数名        | 输入/输出 | 描述                                  |
| ------------- | --------- | ------------------------------------- |
| workspace     | 输入      | 在 Device 侧申请的 workspace 内存地址。 |
| workspaceSize | 输入      | workspace 的大小，单位为字节。         |
| executor      | 输入      | op 执行器，包含了算子计算流程。        |
| stream        | 输入      | 指定执行计算的 stream。                |

### 返回值

aclnnStatus：返回 0 表示成功，返回其他值表示失败。

## 约束说明

- 仅支持标准 MHA/GQA KV cache，不支持 MLA latent cache。
- `head_dim` 首版固定 128；`num_kv_heads` 支持 4~32；`mseBits` 支持 2/3/4。
- 混合 3.5-bit 模式（outlier/non-outlier channel 分组）不纳入本版范围。

## 调用示例

示例代码详见 [test_aclnn_kv_cache_turbo_quant.cpp](../examples/test_aclnn_kv_cache_turbo_quant.cpp)。
