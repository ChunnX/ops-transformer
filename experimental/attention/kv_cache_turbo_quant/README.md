# KvCacheTurboQuant

## 产品支持情况

| 产品                                                     | 是否支持 |
| -------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    ×     |
| <term>Atlas 训练系列产品</term>                          |    ×     |

## 功能说明

- 算子功能：对标准 MHA/GQA 的 KV cache 做在线 TurboQuant 向量量化。每个 head 先做正交旋转和 Lloyd-Max 标量量化，再对残差做 QJL 1-bit 符号编码。首版不覆盖 MLA latent cache，也不做 outlier/non-outlier 混合位宽。

- 计算逻辑：把输入看成 `rows = num_tokens * num_kv_heads` 条 `head_dim = 128` 的向量。对每条向量

  1. `x_norm = ||x||_2`，`u = x / x_norm`（范数为 0 时 `u = 0`）
  2. `y = H @ u`，实现为 `unit @ rotation_matrix.T`
  3. 用 `mse_bits` 的 Lloyd-Max 中点边界做最近码字，`idx` 按 8 通道一组、低位在前打包成 uint8
  4. `r = y - y_hat`，`gamma = x_norm * ||r||_2`
  5. `qjl = sign(S @ r / ||r||_2)`，正数和 0 记为 1，按 1 bit、低位在前打包

- `head_dim = 128`、`mse_bits = 3` 时每 head 存储 68 字节，相对 BF16 的 256 字节压缩约 3.76 倍。

- 等价参考实现见 [tests/golden.py](./tests/golden.py)。设备侧自测见 [examples/test_aclnn_kv_cache_turbo_quant.cpp](./examples/test_aclnn_kv_cache_turbo_quant.cpp)。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| kv_vectors | 输入 | 待量化的 K 或 V | BF16 | ND，`[num_tokens, num_kv_heads, 128]` |
| rotation_matrix | 输入 | 正交旋转矩阵 H，推理期固定 | FLOAT32 | ND，`[128, 128]` |
| qjl_matrix | 输入 | 残差高斯投影 S，推理期固定 | FLOAT32 | ND，`[qjl_dim, 128]`，首版 `qjl_dim = 128` |
| mse_bits | 可选属性 | 主量化位宽，默认 3 | INT | 仅 2、3、4 |
| quant_idx | 输出 | 主量化码字 | UINT8 | ND，`[num_tokens, num_kv_heads, 128 * mse_bits / 8]` |
| quant_qjl | 输出 | 残差符号 | UINT8 | ND，`[num_tokens, num_kv_heads, qjl_dim / 8]` |
| quant_norm | 输出 | `\|\|x\|\|_2` | BF16 | ND，`[num_tokens, num_kv_heads]` |
| quant_gamma | 输出 | `\|\|x\|\|_2 * \|\|r\|\|_2` | BF16 | ND，`[num_tokens, num_kv_heads]` |

## 约束说明

- 只处理标准 MHA/GQA KV，不处理 MLA 的 `cKV` / `kR`。
- `head_dim` 固定 128。`num_kv_heads` 为 4 到 32。`qjl_dim` 首版等于 128。
- `mse_bits` 只支持 2、3、4，默认 3。
- 旋转矩阵和投影矩阵在一次推理中保持不变，由调用方传入。
- 编码在同一条 stream 上完成：BF16 转 FP32、`aclnn` 的 vector_norm 与 div、Cube `Y = U H^T`、Vector Lloyd-Max、残差再走同一套 vector_norm 与 div、`gamma` 乘法、BF16 写回、Cube `P = R S^T`、Vector 符号打包。范数、除法和 GEMM 与 golden 使用的 torch 实现逐位一致。单独的 mix kernel 在当前直调路径上 `SyncAll` 返回 507015，因此 host 显式编排 AIC 与 AIV。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| 直调 | [test_aclnn_kv_cache_turbo_quant](./examples/test_aclnn_kv_cache_turbo_quant.cpp) | `bash examples/build.sh` 编译并跑完全部自测。接口说明见 [aclnnKvCacheTurboQuant](./docs/aclnnKvCacheTurboQuant.md)。 |

Ascend 910B 与 910_93 均在算子定义里注册。核函数使用 `dav-c220-vec` 与 `dav-c220-cube`，与 A2/A3 的 C220 核一致。

## 代码与评审地址

- 个人仓：https://gitcode.com/Wilbert_666/kv_cache_turbo_quant ，分支 `main`，算子目录 `kv_cache_turbo_quant/`。已邀请 `Ascend-CANN` 作为开发者。
- 自测报告：个人仓 `task_submission/`。复现步骤见 [tests/README.md](./tests/README.md)。
- 设计文档 PR：https://gitcode.com/cann/cann-ops-competitions/merge_requests/1821
- 算子源码 PR：https://gitcode.com/cann/ops-transformer/merge_requests/12860 ，目录 `experimental/attention/kv_cache_turbo_quant`。
