# grouped_matmul_swiglu_quant_v2 MxA8W4 单多单 Tiling 侧 UT 校验清单

---

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档标题 | grouped_matmul_swiglu_quant_v2 MxA8W4 单多单 Tiling 侧 UT 校验清单 |
| 来源 STC | `gmm/grouped_matmul_swiglu_quant_v2/docs/GMMSQ_V2_MxA8W4_SMS_STC.md` |
| 适用范围 | arch35 MxA8W4 weight quant 路径下的 op_host / op_tiling UT |
| 更新时间 | 2026-06-15 |

---

## 2. 筛选口径

本文档从 `GMMSQ_V2_MxA8W4_SMS_STC.md` 中筛选可在 Tiling 侧 UT 中校验的用例和检查点。

### 2.1 纳入范围

- TilingKey 校验。
- `tilingData` 字段校验。
- host tiling checker 校验。
- TensorList 长度、desc、dtype、format、empty tensor 校验。
- shape / 轴长 / groupNum / attr / optional input 校验。
- 合法输入下 tiling 成功路径。
- 非单多单路径的 TilingKey 回归。

### 2.2 不纳入范围

- kernel 地址切换验证。
- kernel golden / 数值精度对比。
- aclnn 端到端冒烟。
- 性能回归。

因此，STC 中以下用例不归入 Tiling 侧 UT：

| case_id | testcase | 原因 |
| --- | --- | --- |
| WB015 | sms_address_switch_g3 | kernel group 间切换 tensor list 地址，属于 kernel 侧验证 |
| WB016 | sms_address_switch_scale_g3 | kernel group 间切换 `weightScale` 地址，属于 kernel 侧验证 |

---

## 3. Tiling UT 通用断言

### 3.1 合法单多单用例通用断言

合法单多单 Tiling UT 建议统一断言：

1. tiling 返回成功。
2. `GetTilingKey()` / `GET_TPL_TILING_KEY(wTrans, 0, isSingleMultiSingle)` 中 `IS_SINGLE_MULTI_SINGLE == 1`。
3. `tilingData.groupNum == groupNum`。
4. `tilingData.kSize == K`。
5. `tilingData.nSize == N`。
6. `tilingData.groupListType == groupListType`。
7. `weightListSize == weightScaleListSize == groupNum`。
8. `weight[i].origin_shape == [N, K]`。
9. `weight[i].storage_shape == [ceil(K / 32), ceil(N / 16), 16, 32]`。
10. `weightScale[i].shape == [N, ceil(K / 64), 2]`。
11. `xScale.shape == [M, ceil(K / 64), 2]`。
12. `groupList.shape == [groupNum]`。

### 3.2 非单多单回归用例通用断言

非单多单 Tiling UT 建议统一断言：

1. tiling 返回成功。
2. `IS_SINGLE_MULTI_SINGLE == 0`。
3. 非单多单路径 `groupNum` 按既有 weight shape 逻辑获取。
4. `weight` / `weightScale` 单 tensor 路径行为保持兼容。

### 3.3 异常用例通用断言

异常 Tiling UT 建议统一断言：

1. tiling 返回失败。
2. 每条用例只违反一条规则。
3. 失败日志包含预期关键字。
4. 不依赖多个 checker 的先后命中结果。

---

## 4. 白盒 / 正向 / 回归 Tiling UT

| case_id | testcase | groupNum | groupListType | weight format | weightScale format | shape 摘要 | Tiling 侧校验点 |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| WB001 | sms_wq_nz_ws_nd_g2_glt0_min_k_n | 2 | 0 | FRACTAL_NZ | ND | M=16,K=32,N=128 | 单多单最小 K/N；`IS_SINGLE_MULTI_SINGLE=1`；`groupNum=2`；`groupListType=0` |
| WB002 | sms_wq_nz_ws_nd_g2_glt1_min_k_n | 2 | 1 | FRACTAL_NZ | ND | M=16,K=32,N=128 | 覆盖 `groupListType=1`；`IS_SINGLE_MULTI_SINGLE=1` |
| WB003 | sms_wq_nz_c032_ws_nd_g3 | 3 | 0 | FRACTAL_NZ_C0_32 | ND | M=33,K=64,N=256 | 覆盖 `weight format = FRACTAL_NZ_C0_32`；`groupNum=3` |
| WB004 | sms_wq_nz_ws_nchw_g3 | 3 | 0 | FRACTAL_NZ | NCHW | M=33,K=64,N=256 | 覆盖 `weightScale format = NCHW` |
| WB005 | sms_wq_nz_c032_ws_nchw_g4 | 4 | 1 | FRACTAL_NZ_C0_32 | NCHW | M=64,K=128,N=256 | 覆盖 `weight=FRACTAL_NZ_C0_32` + `weightScale=NCHW` 组合 |
| WB006 | sms_xscale_ncl_g2 | 2 | 0 | FRACTAL_NZ | ND | M=17,K=64,N=128 | 覆盖 `xScale format = NCL` |
| WB007 | sms_k64_scale_ceil | 2 | 0 | FRACTAL_NZ | ND | M=8,K=64,N=128 | 校验 `ceil(K / 64)` 边界，`K=64` |
| WB008 | sms_k96_scale_ceil | 2 | 0 | FRACTAL_NZ | ND | M=8,K=96,N=128 | 校验 `K=96` 时 `ceil(K / 64)=2`，且 `K` 满足 32 对齐 |
| WB009 | sms_n256_y_half | 2 | 1 | FRACTAL_NZ | ND | M=20,K=128,N=256 | 校验 `N=256` 下 `nSize` 和 SwiGLU `N/2` 相关 shape 约束 |
| WB010 | sms_groupnum_1024_upper | 1024 | 1 | FRACTAL_NZ | ND | M=1024,K=32,N=128 | 单多单 `groupNum=1024` 上界合法路径 |
| WB011 | non_sms_single_tensor_regression | 1 | 0 | FRACTAL_NZ | ND | weight view [E,N,K] | 非单多单单 tensor 回归；`IS_SINGLE_MULTI_SINGLE=0` |
| WB012 | non_sms_nz_c032_regression | 1 | 0 | FRACTAL_NZ_C0_32 | ND | weight view [E,N,K] | 非单多单 `FRACTAL_NZ_C0_32` 回归；`IS_SINGLE_MULTI_SINGLE=0` |
| WB013 | sms_tiling_key_whitebox | 2 | 0 | FRACTAL_NZ | ND | 合法小 shape | 白盒检查 `IS_SINGLE_MULTI_SINGLE=1` |
| WB014 | non_sms_tiling_key_whitebox | 1 | 0 | FRACTAL_NZ | ND | 合法小 shape | 白盒检查 `IS_SINGLE_MULTI_SINGLE=0` |

---

## 5. 泛化合法 Tiling UT

泛化用例可作为参数化合法用例，不要求 kernel golden，对 Tiling 侧重点验证合法 shape / format / groupNum 组合能够成功 tiling，并生成正确 TilingKey 和 `tilingData` 字段。

| case_id | testcase | groupNum | shape 摘要 | format 轮转 | Tiling 侧校验点 |
| --- | --- | ---: | --- | --- | --- |
| GEN001 | sms_random_seed_001_g2 | 2 | M=7,K=32,N=128 | NZ / ND / xScale ND | 小 M 合法 tiling；`IS_SINGLE_MULTI_SINGLE=1` |
| GEN002 | sms_random_seed_002_g4 | 4 | M=31,K=64,N=256 | NZ_C0_32 / ND / xScale NCL | 多 group；覆盖 `weight=FRACTAL_NZ_C0_32` 和 `xScale=NCL` |
| GEN003 | sms_random_seed_003_g7 | 7 | M=113,K=96,N=384 | NZ / NCHW / xScale ND | 非整齐 M；覆盖 `weightScale=NCHW` |
| GEN004 | sms_random_seed_004_g16 | 16 | M=257,K=128,N=512 | NZ_C0_32 / NCHW / xScale NCL | 中等规模；覆盖组合 format |
| GEN005 | sms_random_seed_005_g64 | 64 | M=1024,K=256,N=1024 | NZ / ND / xScale ND | 较大 groupNum 合法 tiling |
| GEN006 | sms_random_seed_006_g3 | 3 | M=65,K=512,N=128 | NZ_C0_32 / ND / xScale NCL | 大 K 小 N；`ceil(K/64)` 多段 |
| GEN007 | sms_random_seed_007_g5 | 5 | M=128,K=64,N=512 | NZ / NCHW / xScale ND | 大 N；覆盖 `weightScale=NCHW` |
| GEN008 | sms_random_seed_008_g8 | 8 | M=512,K=128,N=256 | NZ_C0_32 / NCHW / xScale NCL | 常见训练形态合法 tiling |
| GEN009 | sms_random_seed_009_g11 | 11 | M=233,K=96,N=256 | NZ / ND / xScale ND | 奇数 groupNum 合法 tiling |
| GEN010 | sms_random_seed_010_g32 | 32 | M=768,K=256,N=384 | NZ_C0_32 / ND / xScale NCL | 稳定性合法 tiling |
| GEN011 | sms_random_seed_011_g6 | 6 | M=96,K=32,N=512 | NZ / NCHW / xScale ND | 最小 K + 大 N 合法 tiling |
| GEN012 | sms_random_seed_012_g9 | 9 | M=999,K=128,N=128 | NZ_C0_32 / NCHW / xScale NCL | 非整齐大 M 合法 tiling |

---

## 6. 负向 Tiling UT / Host Checker 校验

| case_id | testcase | exception_type | 触发点 | 期望结果 |
| --- | --- | --- | --- | --- |
| NEG001 | sms_invalid_weight_scale_list_less | invalid_weight_scale_list_size_less | `weight` 长度 3，`weightScale` 长度 2 | tiling 失败；日志包含 `weight and weightScale tensor list size should be equal` |
| NEG002 | sms_invalid_weight_scale_list_more | invalid_weight_scale_list_size_more | `weight` 长度 2，`weightScale` 长度 3 | tiling 失败；日志包含 `weight and weightScale tensor list size should be equal` |
| NEG003 | sms_invalid_weight_desc_null | invalid_weight_desc_null | `weight[1]` desc 为空 | tiling 失败；日志包含 `Dynamic input desc of weight[1] is nullptr` |
| NEG004 | sms_invalid_weight_scale_desc_null | invalid_weight_scale_desc_null | `weightScale[1]` desc 为空 | tiling 失败；日志包含 `Dynamic input desc of weightScale[1] is nullptr` |
| NEG005 | sms_invalid_weight_dtype | invalid_weight_dtype | `weight[1]` dtype 非 `FLOAT4_E2M1` | tiling 失败；日志包含 `weight[1] dtype must be` |
| NEG006 | sms_invalid_weight_scale_dtype | invalid_weight_scale_dtype | `weightScale[1]` dtype 非 `FLOAT8_E8M0` | tiling 失败；日志包含 `weightScale[1] dtype must be` |
| NEG007 | sms_invalid_weight_format | invalid_weight_format | `weight[1]` format 非 NZ / NZ_C0_32 | tiling 失败；日志包含 `Weight[1] format is only support` |
| NEG008 | sms_invalid_weight_scale_format | invalid_weight_scale_format | `weightScale[1]` format 非 ND / NCHW | tiling 失败；日志包含 `weightScale[1] format must be ND or NCHW` |
| NEG009 | sms_invalid_weight_empty | invalid_weight_empty | `weight[1]` empty tensor | tiling 失败；日志包含 `Not yet support empty tensor for weight[1]` |
| NEG010 | sms_invalid_weight_scale_empty | invalid_weight_scale_empty | `weightScale[1]` empty tensor | tiling 失败；日志包含 `Not yet support empty tensor for weightScale[1]` |
| NEG011 | sms_invalid_weight_origin_dim | invalid_weight_origin_dim | `weight[0]` origin dim 非 2 | tiling 失败；日志包含 `input dimension of weight should be 2` |
| NEG012 | sms_invalid_weight_k_mismatch | invalid_weight_k_mismatch | `x.K != weight.K` | tiling 失败；日志包含 `Inputs dimension is not match` |
| NEG013 | sms_invalid_weight_storage_shape | invalid_weight_storage_shape | `weight[1]` storage shape 非期望 | tiling 失败；日志包含 `Check input weight shape failed` |
| NEG014 | sms_invalid_weight_scale_shape | invalid_weight_scale_shape | `weightScale[1]` shape 非期望 | tiling 失败；日志包含 `Check input weightScale shape failed` |
| NEG015 | sms_invalid_x_scale_shape | invalid_x_scale_shape | `xScale` shape 非期望 | tiling 失败；日志包含 `Check input xScale shape failed` |
| NEG016 | sms_invalid_group_list_shape | invalid_group_list_shape | `groupList` shape 非 `[groupNum]` | tiling 失败；日志包含 `Check input groupList shape failed` |
| NEG017 | sms_invalid_k_min | invalid_k_min | `K=16` | tiling 失败；日志包含 `K size` / `< min 32` |
| NEG018 | sms_invalid_k_align | invalid_k_align | `K=48` | tiling 失败；日志包含 `not 32 aligned` |
| NEG019 | sms_invalid_n_min | invalid_n_min | `N=64` | tiling 失败；日志包含 `N size` / `< min 128` |
| NEG020 | sms_invalid_n_align | invalid_n_align | `N=192` | tiling 失败；日志包含 `not 128 aligned` |
| NEG021 | sms_invalid_group_num_upper | invalid_group_num_range | `groupNum=1025` | tiling 失败；日志包含 `The input range for the group num is [1, 1024]` |
| NEG022 | sms_invalid_attrs | invalid_attrs | `dequantMode / quantMode / quantDtype / transposeWeight / groupListType` 非法 | tiling 失败；日志包含对应 attr 关键字，建议逐项拆分 |
| NEG023 | sms_invalid_weight_assist | invalid_optional_weight_assist | `weightAssistMatrix` 非空 | tiling 失败；日志包含 `weightAssistMatrix is not supported` |
| NEG024 | sms_invalid_bias_smooth | invalid_optional_bias_smooth | `bias` 或 `smoothScale` 非空 | tiling 失败；日志包含 `bias is not supported` / `smoothScale is not supported`，建议拆分为 bias 和 smoothScale 两条 UT |

---

## 7. 实现点与 Tiling UT 映射

| 实现点 | Tiling UT 覆盖 |
| --- | --- |
| 单多单识别：`weightListSize > 1` | WB001-WB010、WB013、GEN001-GEN012 |
| list 长度一致性：`weightListSize == weightScaleListSize` | NEG001、NEG002 |
| `groupNum` 设置：单多单来自 `weightListSize`，非单多单来自 weight shape | WB001-WB014、NEG016、NEG021 |
| `weight` 逐成员 desc / dtype / format | NEG003、NEG005、NEG007 |
| `weightScale` 逐成员 desc / dtype / format | NEG004、NEG006、NEG008 |
| 空 tensor 检查 | NEG009、NEG010 |
| 单多单 `weight` storage shape | WB001-WB010、WB013、GEN001-GEN012、NEG013 |
| 单多单 `weightScale` shape | WB001-WB010、WB013、GEN001-GEN012、NEG014 |
| `xScale` / `groupList` shape | NEG015、NEG016 |
| M/K/N/groupNum 范围 | NEG017-NEG021 |
| attr 校验 | NEG022 |
| optional 输入校验 | NEG023、NEG024 |
| TilingKey：`GET_TPL_TILING_KEY(wTrans, 0, isSingleMultiSingle)` | WB013、WB014 |

---

## 8. 数量汇总

| 类型 | 用例范围 | 数量 | 说明 |
| --- | --- | ---: | --- |
| 白盒 / 正向 / 回归 | WB001-WB014 | 14 | 包含单多单合法路径、非单多单回归、TilingKey 白盒 |
| 泛化合法 Tiling | GEN001-GEN012 | 12 | 合法参数化 shape / format / groupNum 组合 |
| 负向 Host Checker | NEG001-NEG024 | 24 | list / desc / dtype / format / empty / shape / attr / optional 校验 |
| 不纳入 Tiling UT | WB015-WB016 | 2 | kernel 地址切换验证 |
| **Tiling UT 合计** | WB001-WB014 + GEN001-GEN012 + NEG001-NEG024 | **50** | 推荐落入 op_host / op_tiling UT |

---

## 9. 建议落地方式

1. 将 WB001-WB014 作为 Tiling 白盒基础集。
2. 将 GEN001-GEN012 作为参数化合法输入集，可复用同一 UT helper 构造 mock context。
3. 将 NEG001-NEG024 作为异常 checker 集，每条只 mutation 一个字段。
4. 对 NEG022 建议拆分为多个 attr 子用例：
   - invalid_dequant_mode
   - invalid_quant_mode
   - invalid_quant_dtype
   - invalid_transpose_weight
   - invalid_group_list_type
5. 对 NEG024 建议拆分为两个 optional 子用例：
   - invalid_bias_non_empty
   - invalid_smooth_scale_non_empty
6. TilingKey 校验建议直接覆盖：
   - 单多单：`IS_SINGLE_MULTI_SINGLE == 1`
   - 非单多单：`IS_SINGLE_MULTI_SINGLE == 0`
7. 所有合法用例建议同时校验 `groupNum`、`kSize`、`nSize`、`groupListType`，避免只验证返回成功。
