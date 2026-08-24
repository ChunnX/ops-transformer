# M-RoPE MX SRS/STC 交付物

本目录保存 `QkvRmsNormRopeCacheWithKScale` 的 SRS、STC 和实现偏差说明。测试参数、生成器和执行 CSV 只保留在 `tests/st/`；文档直接引用这些文件，不再复制第二套 CSV。

## 文件索引

| 文件 | 用途 |
|---|---|
| [mrope_mx_srs.md](mrope_mx_srs.md) | 相对已有 M-RoPE 的接口、Host tiling 和 SIMD 设计 |
| [mrope_mx_stc.md](mrope_mx_stc.md) | 1200 条 ST 的生成方法、当前白盒分支和需求追踪 |
| [implementation_deviations.md](implementation_deviations.md) | 当前实现与完整验收目标之间尚未关闭的差距 |

## 唯一事实源和派生链

| 内容 | 唯一事实源 |
|---|---|
| 公开合同 | [mrope_mx_srs.md](mrope_mx_srs.md) |
| 当前实现说明 | [M-RoPE MX design](../design/mrope_mx/) |
| 数值 Golden | [qkv_rms_norm_rope_cache_with_k_scale_golden.py](../../tests/assets/qkv_rms_norm_rope_cache_with_k_scale_golden.py) |
| 参数、因子、约束、配额和白盒锚点 | [tests/st/design](../../tests/st/design/) |
| 生成器 | [generate_st_cases.py](../../tests/st/generate_st_cases.py) |
| 1200 条统一索引 | [07_因子值.csv](../../tests/st/07_因子值.csv) |
| ACLNN L0 | [ttk_aclnn...l0_st.csv](../../tests/st/arch35/ttk_aclnn_qkv_rms_norm_rope_cache_with_k_scale_l0_st.csv) |
| ACLNN L1 | [ttk_aclnn...l1_st.csv](../../tests/st/arch35/ttk_aclnn_qkv_rms_norm_rope_cache_with_k_scale_l1_st.csv) |
| ACLNN L2 | [negative_aclnn...l2_manifest.csv](../../tests/st/arch35/negative_aclnn_qkv_rms_norm_rope_cache_with_k_scale_l2_manifest.csv) |
| Kernel | [ttk_kernel...st.csv](../../tests/st/arch35/ttk_kernel_qkv_rms_norm_rope_cache_with_k_scale_st.csv) |
| Torch | [ttk_e2e...st.csv](../../tests/st/arch35/ttk_e2e_qkv_rms_norm_rope_cache_with_k_scale_st.csv) |

`03_参数定义.yaml` 到 `08_白盒分支覆盖矩阵.yaml` 是可编辑设计输入。生成器是唯一派生程序；五份执行文件和 `07_因子值.csv` 都是生成结果。维护人员不得手工修改生成结果。

## 当前生成身份

| 文件 | 数据行数 | SHA256 |
|---|---:|---|
| `07_因子值.csv` | 1200 | `17dab17764c0e55d67822cd3d4e9c33db7af7939af761aed019ad1e8eeedb8a0` |
| `ttk_aclnn_qkv_rms_norm_rope_cache_with_k_scale_l0_st.csv` | 200 | `2edf6c60c2f3dbb282de1267889446a7d0162b3985bba3198a6625583a0c4633` |
| `ttk_aclnn_qkv_rms_norm_rope_cache_with_k_scale_l1_st.csv` | 650 | `044e164a2df2a2cc8bcf503d6c2a0feaaf544fa08a2b26c8e3e5c5bf060985dd` |
| `negative_aclnn_qkv_rms_norm_rope_cache_with_k_scale_l2_manifest.csv` | 50 | `8a86f82123a75de8c4260140f26198fa5e4e5141a0545f63cc1c48a414b117b6` |
| `ttk_kernel_qkv_rms_norm_rope_cache_with_k_scale_st.csv` | 250 | `ea1388589d5a781b799286464364f616bd3bf4a5a25b8672b26f9891911b68f5` |
| `ttk_e2e_qkv_rms_norm_rope_cache_with_k_scale_st.csv` | 50 | `669146cd65ba2f69deb67bd20e6786112730ecab3e52eebd83770322aece316f` |

五份执行/负例文件合计 1200 行。`07_因子值.csv` 是同一批用例的索引，不重复计数。

## 同步规则

1. 修改合同或源码分支时，先修改 `tests/st/design/` 和生成器。
2. 执行 `generate_st_cases.py generate` 生成全部六份文件，再执行 `generate_st_cases.py check`。
3. `check` 必须确认数量、场景/优先级配额、全局唯一 case 名、ACLNN V1、16 个当前 SIMD 锚点、数值/物理 storage 覆盖和生成字节一致。
4. 接口、Golden、tiling key、VF 路由或 runner 物理 storage 语义变化后，旧设备结果自动失效；不能只改文档中的哈希。
5. 文档只引用执行事实源，不复制 CSV，避免两套文件独立演化。

## 2026-08-23 离线检查边界

- `generate` 和 `check` 都通过，重复生成保持上表字节身份；生成结果只调用直接增加 `kQuantMode` 的 ACLNN V1，没有 V2 API。
- Host tiling UT 共 134 条全部通过，其中新增八个 `Nq=8,16,...,64` 当前路由和 Q pair-tail 低 UB 回退检查。
- Golden 离线检查覆盖 TND/NTD 的 `scale_edge_qk`、`max_finite_qk` 和 `position_1024`，并验证未命中 cache 保持原始字节。
- 本轮没有在真实 NPU 上执行 1200 条用例。历史 1200/1200 是合并后的 V2 诊断证据，不是当前 V1 源码和当前 package 的正式验收结果。
