# 契约锁（对照任务书 op.json + golden.py）

| 项 | 值 |
|----|-----|
| Op | `KvCacheTurboQuant` |
| 输入 | `kv_vectors` bf16 `[T,H,128]`；`rotation_matrix` f32 `[128,128]`；`qjl_matrix` f32 `[128,128]` |
| 输出 | `quant_idx` u8 `[T,H,ceil(128*mse_bits/8)]`；`quant_qjl` u8 `[T,H,16]`；`quant_norm`/`quant_gamma` bf16 `[T,H]` |
| Attr | `mse_bits` int，默认 3，允许 2/3/4 |
| 精度 | 对齐 golden 打包；相对 MSE `<0.05`；内积相对误差 95 分位 `<0.1` |
| 性能 | 相对 PyTorch eager ≥10×（任务书五 case 基线） |

禁止：MLA、独立 decompress 算子、混合 3.5-bit 首版。
