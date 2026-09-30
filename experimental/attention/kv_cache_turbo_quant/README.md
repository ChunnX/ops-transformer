# KvCacheTurboQuant

## 产品支持情况

| 产品                                                     | 是否支持 |
| -------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    x     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    x     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    ×     |
| <term>Atlas 训练系列产品</term>                          |    ×     |

## 功能说明

- 算子功能：参考 TurboQuant 论文，面向标准 MHA/GQA KV cache 的在线向量量化压缩算子（encode 方向）。首版采用统一 `mse_bits` bit MSE 主编码 + 1-bit QJL 残差编码；`head_dim=128`、`mse_bits=3` 时，计入向量范数和残差范数后物理存储为 4.25 bit/channel，理论显存压缩比约 3.76x（vs BF16）。
- 仅支持标准 MHA/GQA KV cache，不支持 MLA latent cache。
- 计算逻辑（对每个 `(token, kv_head)` 向量 `x ∈ R^128`）：

  1. `x_norm = ||x||_2`，`u = x / x_norm`（零向量安全处理）
  2. `y = H @ u`（`H` 为 Hadamard 矩阵或预生成随机正交矩阵，推理期间固定）
  3. `idx[j] = scalar_quantize(y[j], mse_bits)`：标准正态 Lloyd-Max 质心逐维标量量化，按 bit 打包
  4. `y_hat = dequant(idx)`，`r = y - y_hat`，`gamma = x_norm * ||r||_2`
  5. `qjl = sign(S @ (r / ||r||_2))`：高斯随机投影 + 1-bit 符号量化，打包为 uint8

- 等价参考实现（fp32）：

```python
x_norm = np.linalg.norm(x, axis=-1)                     # [T, H]
u = np.where(x_norm[..., None] > 0, x / np.maximum(x_norm, 1e-30)[..., None], 0)
y = u @ rotation_matrix.T                               # [T, H, 128]
idx = (y[..., None] > boundaries).sum(-1)               # Lloyd-Max 质心分界
y_hat = centroids[idx]
r = y - y_hat
gamma = x_norm * np.linalg.norm(r, axis=-1)             # [T, H]
r_hat = np.where(rn[..., None] > 0, r / np.maximum(rn, 1e-30)[..., None], 0)
qjl = (r_hat @ qjl_matrix.T >= 0)                       # [T, H, qjl_dim]
# idx 按 mse_bits bit 打包，qjl 按 1 bit 打包（每 8 维一组，低位在前，小端字节序）
```

- 存储格式（`head_dim=128`，`mse_bits=3`，`qjl_dim=128`）：

| 组件              | 摊销 bit/channel | 总 bit | 说明                        |
| ----------------- | ---------------: | -----: | --------------------------- |
| idx (主量化)      |            3.000 |    384 | 48 bytes（bit-packed）      |
| qjl (残差符号)    |            1.000 |    128 | 16 bytes（bit-packed）      |
| x_norm (向量范数) |            0.125 |     16 | 2 bytes（bf16 scalar）      |
| gamma (残差范数)  |            0.125 |     16 | 2 bytes（bf16 scalar）      |
| 合计              |            4.250 |    544 | 68 bytes/head vs 原始 256B  |

## 参数说明

| 参数名          | 输入/输出/属性 | 描述                                                              | 数据类型 | 数据格式 |
| --------------- | -------------- | ----------------------------------------------------------------- | -------- | -------- |
| kv_vectors      | 输入           | 待量化的 K 或 V 向量，三维 `[num_tokens, num_kv_heads, head_dim]` | BFLOAT16 | ND       |
| rotation_matrix | 输入           | 正交旋转矩阵 H，二维 `[head_dim, head_dim]`                        | FLOAT32  | ND       |
| qjl_matrix      | 输入           | 残差投影矩阵 S，二维 `[qjl_dim, head_dim]`，元素服从固定高斯分布   | FLOAT32  | ND       |
| mse_bits        | 可选属性       | MSE 主量化位宽，支持 2/3/4，默认 3                                 | INT      | -        |
| quant_idx       | 输出           | 主量化编码（bit-packed），`[num_tokens, num_kv_heads, 16*mse_bits]`| UINT8    | ND       |
| quant_qjl       | 输出           | 残差 QJL 符号编码（bit-packed），`[num_tokens, num_kv_heads, qjl_dim/8]` | UINT8 | ND |
| quant_norm      | 输出           | 输入向量 L2 范数，`[num_tokens, num_kv_heads]`                     | BFLOAT16 | ND       |
| quant_gamma     | 输出           | 原尺度残差范数 `x_norm * ||r||_2`，`[num_tokens, num_kv_heads]`    | BFLOAT16 | ND       |

## 约束说明

- 仅支持标准 MHA/GQA KV cache，不支持 MLA latent cache。
- `head_dim` 首版固定 128；`num_kv_heads` 支持 4~32；`qjl_dim` 支持 (0, 128] 且为 8 的倍数，通常等于 `head_dim`。
- `mse_bits` 支持 2/3/4，默认 3；加上 QJL 后编码位宽分别为 3/4/5 bit/channel。
- bit 打包顺序：每 8 维为一组，组内第 `lane` 维置于 `lane * bit_width` 起始的低位，多字节按小端字节序输出，与参考实现 `_pack_bits` 完全一致。
- `rotation_matrix`、`qjl_matrix` 在推理期间固定；量化过程在线完成，不成为 prefill 瓶颈。

## 调用说明

| 调用方式  | 调用样例                                                                            | 说明                                                                      |
| --------- | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| aclnn调用 | [test_aclnn_kv_cache_turbo_quant](./examples/test_aclnn_kv_cache_turbo_quant.cpp)   | 通过 [aclnnKvCacheTurboQuant](./docs/aclnnKvCacheTurboQuant.md) 接口方式调用该算子。 |
