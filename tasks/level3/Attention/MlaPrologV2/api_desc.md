# MlaPrologV2

## 功能描述

MLA（Multi-head Latent Attention）前向中的 **Prolog V2** 融合段：在 V1 能力基础上增加 **`dequant_scale_q_nope`** 输出；对 `token_x` 与多组权重做矩阵乘、RMSNorm、RoPE，并将当前步的 KV/KR 写入 page-attention 风格的 `kv_cache` / `kr_cache`（如 `PA_BSND` 等 `cache_mode`）。

## 主要输入输出

- **输入**：与 V1 相同：`token_x`、`weight_dq`（NZ）、`weight_uq_qr`（NZ）、`weight_uk`、`weight_dkv_kr`（NZ）、`rmsnorm_gamma_cq`、`rmsnorm_gamma_ckv`、`rope_sin`、`rope_cos`、`cache_index`、`kv_cache`、`kr_cache`；可选反量化/量化/smooth 标量张量见 `mla_prolog_v2_def.cpp`。
- **输出**：`query`、`query_rope`；`kv_cache` 与 `kr_cache` 为就地更新；**`dequant_scale_q_nope`**（FLOAT，非全量化场景 Infershape 为标量形 `(1,)`）。
- **属性**：`rmsnorm_epsilon_cq`、`rmsnorm_epsilon_ckv`、`cache_mode`（字符串，如 `PA_BSND`）。

## Python 验证

精度脚本通过 **`kernel_gen_ops.mla_prolog_v2`** 调用自定义 aclnn（底层 **`aclnnMlaPrologV2WeightNz`**）；CPU 金标准前四路与开源仓 `mla_prolog_v2` 单测中 BF16 + `PA_BSND` 参考实现一致，第五路在非全量化 BF16 路径与零初始化输出对齐。
