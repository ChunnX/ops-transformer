# KvCacheTurboQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

面向标准 MHA/GQA KV cache 的在线向量量化压缩算子（TurboQuant encode）。对每个 128 维 KV 向量独立完成两阶段量化并输出 bit-packed 编码：

1. **MSE 主量化**：向量归一化到单位球面（`u = x / max(‖x‖, 1e-30)`）→ 正交旋转 `y = H·u` → 逐维 Lloyd-Max 标量量化（`mse_bits` 决定 4/8/16 质心，编译期固化）；
2. **残差 QJL 修正**：主重构残差 `r = y − ŷ` 做高斯投影 `S·r` 并按「≥0 记 1」输出 1-bit 符号 sketch。

等价参考实现：

```
norm = ||x||_2
u = x / max(norm, 1e-30)
y = H @ u
idx[j] = Σ_k (y[j] > b_k)            # b_k 为相邻质心中点
yhat = centroids[idx]
r = y - yhat
gamma = norm * ||r||_2
qjl[j] = (S @ r)[j] >= 0
```

默认 3-bit 配置下（编码 3 bit + QJL 1 bit + 两个 bf16 标量），head_dim=128 时物理存储 4.25 bit/channel，显存压缩比约 3.76x（vs BF16）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | shape |
|---|---|---|---|---|---|
| kv_vectors | 输入 | 待量化的 K 或 V 向量 | bfloat16 | ND | [num_tokens, num_kv_heads, 128] |
| rotation_matrix | 输入 | 正交旋转矩阵 H（推理期固定） | float32、bfloat16 | ND | [128, 128] |
| qjl_matrix | 输入 | 残差投影矩阵 S（推理期固定，与 rotation_matrix 同 dtype） | float32、bfloat16 | ND | [128, 128] |
| quant_idx | 输出 | 主量化编码（bit-packed：8 维/组、lane×mse_bits 位移、小端字节序） | uint8 | ND | [num_tokens, num_kv_heads, 128·mse_bits/8] |
| quant_qjl | 输出 | 残差 QJL 符号编码（bit-packed，projected ≥ 0 记 1） | uint8 | ND | [num_tokens, num_kv_heads, 16] |
| quant_norm | 输出 | 输入向量 L2 范数 ‖x‖₂ | bfloat16、float16 | ND | [num_tokens, num_kv_heads] |
| quant_gamma | 输出 | 原尺度残差范数 ‖x‖₂·‖r‖₂ | bfloat16、float16 | ND | [num_tokens, num_kv_heads] |
| mse_bits | 属性 | MSE 主量化位宽，可选 2/3/4，默认 3 | int | - | - |
| out_dtype | 属性 | quant_norm/quant_gamma 元素类型：0=bfloat16（默认）、1=float16 | int | - | - |

## 约束说明

- head_dim 固定 128（质心表维度相关），num_kv_heads 支持 4~32，num_tokens ≥ 1；
- rotation_matrix / qjl_matrix 支持 float32 或 bfloat16（两者必须同 dtype），为推理期固定输入，算子不生成随机矩阵；
- 仅支持 ND 连续 tensor；
- 仅实现 encode（量化），decode/内积估计由调用方使用编码结果完成；
- 不支持 MLA latent cache；
- mse_bits 仅支持 2/3/4，其余取值返回参数错误。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| aclnn 调用 | [aclnnKvCacheTurboQuant](docs/aclnnKvCacheTurboQuant.md) | 两段式接口 `aclnnKvCacheTurboQuantGetWorkspaceSize` / `aclnnKvCacheTurboQuant` |

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
|---|---|---|---|---|
| skytosky | 社区开发者 | KvCacheTurboQuant | 2026-09 | 算子设计、Ascend C 实现、精度/性能自验 |
