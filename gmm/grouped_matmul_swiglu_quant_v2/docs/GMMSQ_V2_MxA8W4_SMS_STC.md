# grouped_matmul_swiglu_quant_v2 MxA8W4 单多单场景支持 STC

---

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| STC 标题 | grouped_matmul_swiglu_quant_v2 MxA8W4 单多单场景支持 STC |
| 关联 SRS | `gmm/grouped_matmul_swiglu_quant_v2/docs/GMMSQ_V2_MxA8W4_SMS_SRS.md` |
| 责任人 | 待定 |
| 更新时间 | 2026-06-12 |
| 版本 | v0.1 |

---

## 2. 测试目标与范围

### 2.1 目标

验证 `grouped_matmul_swiglu_quant_v2` 在 arch35 MxA8W4 weight quant 路径下新增“单多单”能力的正确性、异常校验完整性和既有路径兼容性：

1. **功能正确性**：`x` / `y` 为单 tensor，`weight` / `weightScale` 为 tensor list 时，kernel 按 group 从 tensor list 切换 `weight[i]` / `weightScale[i]` 地址。
2. **TilingKey 正确性**：单多单路径生成 `IS_SINGLE_MULTI_SINGLE == 1`，非单多单回归生成 `IS_SINGLE_MULTI_SINGLE == 0`。
3. **Host 校验完整性**：覆盖 `weight` / `weightScale` 多 tensor list 长度、逐成员 shape、desc、dtype、format、空 tensor 等校验。
4. **shape / attr 约束**：覆盖 M / K / N / groupNum、`xScale`、`groupList`、`weight`、`weightScale` 约束。
5. **兼容性回归**：非单多单 MxA8W4 单 tensor weight 路径行为不变。

### 2.2 不在范围

- 非 MxA8W4 数据流。
- 非 arch35 路径。
- 非 weight quant 路径。
- `weightAssistMatrix`、`bias`、`smoothScale` 的新增可用能力。
- API utils 大范围重构验证。
- 算子数学定义、输出布局、量化语义变更验证。

---

## 3. 环境、输入与限制（可与 SRS 第 5–6 节对照）

### 3.1 运行环境

- 硬件 / SOC / 驱动 / CANN 版本范围：支持 arch35 MxA8W4 的 Ascend 环境，CANN 版本与当前仓库编译环境一致。
- 执行形态：本地 UT / CI / aclnn 冒烟；白盒 TilingKey 可通过 tiling 单测或 mock context 校验。
- 入口：`aclnnGroupedMatmulSwigluQuantV2` 两段式接口及 arch35 tiling / kernel 路径。

### 3.2 输入输出限制摘要

- **dtype**：
  - `x`: `FLOAT8_E4M3FN`
  - `weight`: `FLOAT4_E2M1`
  - `xScale` / `weightScale` / `yScale`: `FLOAT8_E8M0`
  - `groupList`: `INT64`
  - `y`: `FLOAT8_E4M3FN`
- **format**：
  - `x`: `ND`
  - `xScale`: `ND` / `NCL`
  - `groupList`: `ND`
  - `weight`: `FRACTAL_NZ` / `FRACTAL_NZ_C0_32`
  - `weightScale`: `ND` / `NCHW`
  - `y` / `yScale`: `ND`
- **单多单输入组合**：
  - `x` 为单 tensor；
  - `weight` 为 tensor list，长度为 `groupNum`；
  - `weightScale` 为 tensor list，长度必须与 `weight` 完全一致；
  - `y` / `yScale` 为单 tensor。
- **shape**：
  - `x`: `[M, K]`
  - `xScale`: `[M, ceil(K / 64), 2]`
  - 单多单 `weight[g]` view shape: `[N, K]`
  - 单多单 `weight[g]` storage shape: `[ceil(K / 32), ceil(N / 16), 16, 32]`
  - 单多单 `weightScale[g]`: `[N, ceil(K / 64), 2]`
  - `groupList`: `[groupNum]`
  - `y`: `[M, N / 2]`
  - `yScale`: `[M, ceil(N / 2 / 64), 2]`
- **轴向门限**：
  - `M >= 1`
  - `K >= 32 && K % 32 == 0`
  - `N >= 128 && N % 128 == 0`
  - `groupNum in [1, 1024]`
- **属性**：
  - `dequantMode == 2`
  - `dequantDtype == DT_FLOAT`
  - `quantMode == 2`
  - `quantDtype == FLOAT8_E4M3FN`
  - `transposeWeight == true`
  - `groupListType in {0, 1}`
- **资源比 / 核数假设**：`aivNum == aicNum * 2`，且 `aicNum > 0`、`aivNum > 0`。

### 3.3 未在实现中显式文档化的假设

| 条目 | 当前结论 | 风险 |
| --- | --- | --- |
| TensorList 下标连续 | `GetDynamicInputTensorListSize` 通过从 0 开始连续读取 shape 计数，遇到首个 nullptr 停止 | 若框架允许中间空洞，则尾部 tensor 不会被计入；当前按框架 TensorList 连续性假设设计 |
| `groupList` 内容值 | host tiling 当前校验 shape `[groupNum]` 和 dtype，不在本修改中新增数值合法性检查 | STC 正常用例需保证 `groupList` 内容与 M 切分一致；异常仅覆盖 shape / dtype / attr |
| golden 精度 | 与既有 GMMSQ V2 MxA8W4 单 tensor weight 场景一致 | 若底层 MX 量化误差策略变化，需要同步更新容差 |
| `weightAssistMatrix` 空 list 判定 | 当前通过 `IsDynamicInputEmpty` 接受空指针或空 list | 非空 list 应作为异常用例覆盖 |

---

## 4. 测试策略

### 4.1 白盒 / 小组合（精确逻辑）

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 覆盖目标 | 单多单识别、TilingKey、地址更新、list size、逐成员校验 | 与实现分支一一对应 |
| 场景 | 非单多单、单多单 | 非单多单验证 `IS_SINGLE_MULTI_SINGLE == 0`；单多单验证 `IS_SINGLE_MULTI_SINGLE == 1` |
| `groupNum` | 1、2、3、1024 | 1 用于非单多单/边界；2/3 用于 list 取址；1024 用于上界 |
| `groupListType` | 0、1 | 分别覆盖累积结束位置和每组 M 大小 |
| `weight` format | `FRACTAL_NZ`、`FRACTAL_NZ_C0_32` | 覆盖 host format 分支 |
| `weightScale` format | `ND`、`NCHW` | 覆盖 host format 分支 |
| 轴长 | `K=32/64/128`，`N=128/256`，`M=1/多 group 累计` | 覆盖最小值、MX scale ceil、SwiGLU N/2 |
| 用例规模 | 16 条 | 见 4.5 用例矩阵 |

### 4.2 泛化 / 大空间（稳定性与边界）

| 维度 | 取值 | 说明 |
| --- | --- | --- |
| 随机 / 网格 | 固定 seed，生成合法 `M/K/N/groupNum` 网格 | 便于复现 |
| `M` | 1、7、32、257、1024 | 覆盖小 M、非整齐 M、较大 M |
| `K` | 32、64、96、128、256、512 | 均满足 32 对齐 |
| `N` | 128、256、384、512、1024 | 均满足 128 对齐 |
| `groupNum` | 2、4、7、16、64 | 覆盖多个 tensor list 长度 |
| format 组合 | `weight` 2 种 × `weightScale` 2 种 × `xScale` 2 种 | 合法组合轮转 |
| 用例规模 | 12 条 | 用于稳定性与生成器规则验证 |

### 4.3 异常 / 负向

| 类型 | 说明 | 期望 |
| --- | --- | --- |
| invalid_weight_scale_list_size_less | `weight` 多 tensor，`weightScale` 少一个 tensor | `weight and weightScale tensor list size should be equal` |
| invalid_weight_scale_list_size_more | `weight` 单 tensor 或较少 tensor，`weightScale` 多 tensor | `weight and weightScale tensor list size should be equal` |
| invalid_weight_desc_null | 单多单某个 `weight[i]` desc 为空 | `Dynamic input desc of weight[i] is nullptr` |
| invalid_weight_scale_desc_null | 单多单某个 `weightScale[i]` desc 为空 | `Dynamic input desc of weightScale[i] is nullptr` |
| invalid_weight_dtype | 某个 `weight[i]` dtype 非 `FLOAT4_E2M1` | `weight[i] dtype must be FLOAT4_E2M1` |
| invalid_weight_scale_dtype | 某个 `weightScale[i]` dtype 非 `FLOAT8_E8M0` | `weightScale[i] dtype must be FLOAT8_E8M0` |
| invalid_weight_format | 某个 `weight[i]` format 非 `FRACTAL_NZ` / `FRACTAL_NZ_C0_32` | `Weight[i] format is only support` |
| invalid_weight_scale_format | 某个 `weightScale[i]` format 非 `ND` / `NCHW` | `weightScale[i] format must be ND or NCHW` |
| invalid_weight_empty | 某个 `weight[i]` storage shape size 为 0 | `Not yet support empty tensor for weight[i]` |
| invalid_weight_scale_empty | 某个 `weightScale[i]` storage shape size 为 0 | `Not yet support empty tensor for weightScale[i]` |
| invalid_weight_origin_dim | 单多单 `weight[0]` origin dim 非 2 | `input dimension of weight should be 2` |
| invalid_weight_k_mismatch | `x.K != weight.K` | `Inputs dimension is not match` |
| invalid_weight_storage_shape | `weight[i]` storage shape 非 `[ceil(K/32), ceil(N/16), 16, 32]` | `Check input weight shape failed` |
| invalid_weight_scale_shape | `weightScale[i]` shape 非 `[N, ceil(K/64), 2]` | `Check input weightScale shape failed` |
| invalid_x_scale_shape | `xScale` shape 非 `[M, ceil(K/64), 2]` | `Check input xScale shape failed` |
| invalid_group_list_shape | `groupList` shape 非 `[groupNum]` | `Check input groupList shape failed` |
| invalid_k_min | `K < 32` | `K size ... < min 32` |
| invalid_k_align | `K % 32 != 0` | `K size ... not 32 aligned` |
| invalid_n_min | `N < 128` | `N size ... < min 128` |
| invalid_n_align | `N % 128 != 0` | `N size ... not 128 aligned` |
| invalid_group_num_range | `groupNum == 0` 或 `groupNum > 1024` | `The input range for the group num is [1, 1024]` |
| invalid_attrs | `dequantMode / quantMode / quantDtype / transposeWeight / groupListType` 非法 | 对应 attr 日志 |
| invalid_optional_weight_assist | `weightAssistMatrix` 非空 | `weightAssistMatrix is not supported` |
| invalid_optional_bias_smooth | `bias` 或 `smoothScale` 非空 | `bias is not supported` / `smoothScale is not supported` |

异常用例每次只违反一条规则，避免多个 checker 同时命中导致误判。

### 4.4 性能 / 兼容 / 通路

- 性能：本 STC 仅要求单多单通路无明显异常，不设置新性能门限；非单多单路径需通过既有性能回归。
- 兼容：非单多单 MxA8W4 单 tensor `weight` / `weightScale` 用例必须仍走连续 offset 推进路径。
- 通路：至少覆盖 aclnn 冒烟、tiling 白盒和 kernel 结果对比。

### 4.5 用例矩阵（建议落表）

#### 4.5.1 白盒 / 正常用例

| case_id | testcase | 目标 | groupNum | groupListType | weight format | weightScale format | shape 摘要 |
| --- | --- | --- | ---: | ---: | --- | --- | --- |
| WB001 | sms_wq_nz_ws_nd_g2_glt0_min_k_n | 单多单最小 K/N，TilingKey=1 | 2 | 0 | FRACTAL_NZ | ND | M=16,K=32,N=128 |
| WB002 | sms_wq_nz_ws_nd_g2_glt1_min_k_n | groupListType=1 | 2 | 1 | FRACTAL_NZ | ND | M=16,K=32,N=128 |
| WB003 | sms_wq_nz_c032_ws_nd_g3 | weight FRACTAL_NZ_C0_32 | 3 | 0 | FRACTAL_NZ_C0_32 | ND | M=33,K=64,N=256 |
| WB004 | sms_wq_nz_ws_nchw_g3 | weightScale NCHW | 3 | 0 | FRACTAL_NZ | NCHW | M=33,K=64,N=256 |
| WB005 | sms_wq_nz_c032_ws_nchw_g4 | format 组合覆盖 | 4 | 1 | FRACTAL_NZ_C0_32 | NCHW | M=64,K=128,N=256 |
| WB006 | sms_xscale_ncl_g2 | xScale NCL | 2 | 0 | FRACTAL_NZ | ND | M=17,K=64,N=128 |
| WB007 | sms_k64_scale_ceil | `ceil(K/64)` 边界 | 2 | 0 | FRACTAL_NZ | ND | M=8,K=64,N=128 |
| WB008 | sms_k96_scale_ceil | `ceil(K/64)=2` 非 64 整数倍但 32 对齐 | 2 | 0 | FRACTAL_NZ | ND | M=8,K=96,N=128 |
| WB009 | sms_n256_y_half | SwiGLU 输出 N/2 | 2 | 1 | FRACTAL_NZ | ND | M=20,K=128,N=256 |
| WB010 | sms_groupnum_1024_upper | groupNum 上界 | 1024 | 1 | FRACTAL_NZ | ND | M=1024,K=32,N=128 |
| WB011 | non_sms_single_tensor_regression | 非单多单回归，TilingKey=0 | 1 | 0 | FRACTAL_NZ | ND | weight view [E,N,K] |
| WB012 | non_sms_nz_c032_regression | 非单多单 C0_32 回归 | 1 | 0 | FRACTAL_NZ_C0_32 | ND | weight view [E,N,K] |
| WB013 | sms_tiling_key_whitebox | 白盒检查 `IS_SINGLE_MULTI_SINGLE=1` | 2 | 0 | FRACTAL_NZ | ND | 合法小 shape |
| WB014 | non_sms_tiling_key_whitebox | 白盒检查 `IS_SINGLE_MULTI_SINGLE=0` | 1 | 0 | FRACTAL_NZ | ND | 合法小 shape |
| WB015 | sms_address_switch_g3 | kernel group 间切换 tensor list 地址 | 3 | 0 | FRACTAL_NZ | ND | 每组 weight 填不同 pattern |
| WB016 | sms_address_switch_scale_g3 | kernel group 间切换 weightScale 地址 | 3 | 0 | FRACTAL_NZ | ND | 每组 weightScale 填不同 pattern |

#### 4.5.2 泛化用例

| case_id | testcase | groupNum | shape 摘要 | format 轮转 | 说明 |
| --- | --- | ---: | --- | --- | --- |
| GEN001 | sms_random_seed_001_g2 | 2 | M=7,K=32,N=128 | NZ / ND / xScale ND | 小 M |
| GEN002 | sms_random_seed_002_g4 | 4 | M=31,K=64,N=256 | NZ_C0_32 / ND / xScale NCL | 多 group |
| GEN003 | sms_random_seed_003_g7 | 7 | M=113,K=96,N=384 | NZ / NCHW / xScale ND | 非整齐 M |
| GEN004 | sms_random_seed_004_g16 | 16 | M=257,K=128,N=512 | NZ_C0_32 / NCHW / xScale NCL | 中等规模 |
| GEN005 | sms_random_seed_005_g64 | 64 | M=1024,K=256,N=1024 | NZ / ND / xScale ND | 较大 groupNum |
| GEN006 | sms_random_seed_006_g3 | 3 | M=65,K=512,N=128 | NZ_C0_32 / ND / xScale NCL | 大 K 小 N |
| GEN007 | sms_random_seed_007_g5 | 5 | M=128,K=64,N=512 | NZ / NCHW / xScale ND | 大 N |
| GEN008 | sms_random_seed_008_g8 | 8 | M=512,K=128,N=256 | NZ_C0_32 / NCHW / xScale NCL | 常见训练形态 |
| GEN009 | sms_random_seed_009_g11 | 11 | M=233,K=96,N=256 | NZ / ND / xScale ND | 奇数 groupNum |
| GEN010 | sms_random_seed_010_g32 | 32 | M=768,K=256,N=384 | NZ_C0_32 / ND / xScale NCL | 稳定性 |
| GEN011 | sms_random_seed_011_g6 | 6 | M=96,K=32,N=512 | NZ / NCHW / xScale ND | 最小 K 大 N |
| GEN012 | sms_random_seed_012_g9 | 9 | M=999,K=128,N=128 | NZ_C0_32 / NCHW / xScale NCL | 非整齐 M 大样本 |

#### 4.5.3 异常用例

| case_id | testcase | exception_type | 触发点 | 期望日志关键字 |
| --- | --- | --- | --- | --- |
| NEG001 | sms_invalid_weight_scale_list_less | invalid_weight_scale_list_size_less | `weight` 长度 3，`weightScale` 长度 2 | `weight and weightScale tensor list size should be equal` |
| NEG002 | sms_invalid_weight_scale_list_more | invalid_weight_scale_list_size_more | `weight` 长度 2，`weightScale` 长度 3 | `weight and weightScale tensor list size should be equal` |
| NEG003 | sms_invalid_weight_desc_null | invalid_weight_desc_null | `weight[1]` desc 为空 | `Dynamic input desc of weight[1] is nullptr` |
| NEG004 | sms_invalid_weight_scale_desc_null | invalid_weight_scale_desc_null | `weightScale[1]` desc 为空 | `Dynamic input desc of weightScale[1] is nullptr` |
| NEG005 | sms_invalid_weight_dtype | invalid_weight_dtype | `weight[1]` dtype 非 FLOAT4_E2M1 | `weight[1] dtype must be` |
| NEG006 | sms_invalid_weight_scale_dtype | invalid_weight_scale_dtype | `weightScale[1]` dtype 非 FLOAT8_E8M0 | `weightScale[1] dtype must be` |
| NEG007 | sms_invalid_weight_format | invalid_weight_format | `weight[1]` format 非 NZ / NZ_C0_32 | `Weight[1] format is only support` |
| NEG008 | sms_invalid_weight_scale_format | invalid_weight_scale_format | `weightScale[1]` format 非 ND / NCHW | `weightScale[1] format must be ND or NCHW` |
| NEG009 | sms_invalid_weight_empty | invalid_weight_empty | `weight[1]` empty tensor | `Not yet support empty tensor for weight[1]` |
| NEG010 | sms_invalid_weight_scale_empty | invalid_weight_scale_empty | `weightScale[1]` empty tensor | `Not yet support empty tensor for weightScale[1]` |
| NEG011 | sms_invalid_weight_origin_dim | invalid_weight_origin_dim | `weight[0]` origin dim 非 2 | `input dimension of weight should be 2` |
| NEG012 | sms_invalid_weight_k_mismatch | invalid_weight_k_mismatch | `x.K != weight.K` | `Inputs dimension is not match` |
| NEG013 | sms_invalid_weight_storage_shape | invalid_weight_storage_shape | `weight[1]` storage shape 非期望 | `Check input weight shape failed` |
| NEG014 | sms_invalid_weight_scale_shape | invalid_weight_scale_shape | `weightScale[1]` shape 非期望 | `Check input weightScale shape failed` |
| NEG015 | sms_invalid_x_scale_shape | invalid_x_scale_shape | `xScale` shape 非期望 | `Check input xScale shape failed` |
| NEG016 | sms_invalid_group_list_shape | invalid_group_list_shape | `groupList` shape 非 `[groupNum]` | `Check input groupList shape failed` |
| NEG017 | sms_invalid_k_min | invalid_k_min | `K=16` | `K size` / `< min 32` |
| NEG018 | sms_invalid_k_align | invalid_k_align | `K=48` | `not 32 aligned` |
| NEG019 | sms_invalid_n_min | invalid_n_min | `N=64` | `N size` / `< min 128` |
| NEG020 | sms_invalid_n_align | invalid_n_align | `N=192` | `not 128 aligned` |
| NEG021 | sms_invalid_group_num_upper | invalid_group_num_range | `groupNum=1025` | `The input range for the group num is [1, 1024]` |
| NEG022 | sms_invalid_attrs | invalid_attrs | attr 非法组合逐项单独落表 | `dequantMode must be` / `transposeWeight must be true` 等 |
| NEG023 | sms_invalid_weight_assist | invalid_optional_weight_assist | `weightAssistMatrix` 非空 | `weightAssistMatrix is not supported` |
| NEG024 | sms_invalid_bias_smooth | invalid_optional_bias_smooth | `bias` 或 `smoothScale` 非空 | `bias is not supported` / `smoothScale is not supported` |

---

## 5. 覆盖统计（可独立成 `test_coverage_report.md`）

### 5.1 总体

| 类型 | 条数 | 占比 |
| --- | ---: | ---: |
| 白盒 | 16 | 30.8% |
| 泛化 | 12 | 23.1% |
| 异常 | 24 | 46.1% |
| **合计** | 52 | 100% |

### 5.2 分维度表

#### dtype 分布

| dtype 组合 | 条数 | 说明 |
| --- | ---: | --- |
| 合法 MxA8W4 组合 | 28 | 16 白盒 + 12 泛化 |
| 非法 `weight[i]` dtype | 1 | NEG005 |
| 非法 `weightScale[i]` dtype | 1 | NEG006 |
| attr / 其他 dtype 非法 | 1+ | NEG022 可拆分多行 |

#### format 分布

| 维度 | 取值 | 覆盖 |
| --- | --- | --- |
| `weight` | `FRACTAL_NZ` | WB001/WB002/WB004/WB006-WB011/WB013-WB016、部分 GEN |
| `weight` | `FRACTAL_NZ_C0_32` | WB003/WB005/WB012、部分 GEN |
| `weightScale` | `ND` | WB001-WB003/WB006-WB016、部分 GEN |
| `weightScale` | `NCHW` | WB004/WB005、部分 GEN |
| `xScale` | `ND` / `NCL` | 白盒与泛化轮转覆盖 |

#### 约束 / 异常类型分布

| 约束类型 | 条数 | 对应用例 |
| --- | ---: | --- |
| list size / tensor 组合 | 2 | NEG001-NEG002 |
| desc / dtype / format / empty | 8 | NEG003-NEG010 |
| shape / 维度关系 | 6 | NEG011-NEG016 |
| M/K/N/groupNum 边界 | 5 | NEG017-NEG021 |
| attr / optional | 3 | NEG022-NEG024 |

---

## 6. 用例命名规范

推荐格式：

```text
{scene}_{dtype_tag}_{weight_format}_{scale_format}_{shape_tag}_{variant}
```

字段含义：

- `scene`: `sms` / `non_sms`
- `dtype_tag`: `mxa8w4`
- `weight_format`: `nz` / `nz_c032`
- `scale_format`: `ws_nd` / `ws_nchw`
- `shape_tag`: `g{groupNum}_m{M}_k{K}_n{N}` 或语义化标签
- `variant`: `glt0` / `glt1` / `invalid_xxx` / `tiling_key`

示例：

```text
sms_mxa8w4_nz_ws_nd_g2_m16_k32_n128_glt0
sms_mxa8w4_nz_c032_ws_nchw_g4_m64_k128_n256_glt1
non_sms_mxa8w4_nz_ws_nd_g1_regression
sms_mxa8w4_invalid_weight_scale_list_less
```

---

## 7. 约束关系（与实现对齐）

正常用例需满足：

1. `x.dtype == FLOAT8_E4M3FN`，`weight.dtype == FLOAT4_E2M1`，`xScale / weightScale / yScale.dtype == FLOAT8_E8M0`。
2. `x.format == ND`，`xScale.format in {ND, NCL}`，`groupList.format == ND`。
3. `weight.format in {FRACTAL_NZ, FRACTAL_NZ_C0_32}`，`weightScale.format in {ND, NCHW}`。
4. 单多单时 `weightListSize > 1`，且 `weightListSize == weightScaleListSize == groupNum`。
5. 单多单时每个 `weight[i]` / `weightScale[i]` 的 desc 非空、shape 非空、dtype / format 合法。
6. `x.shape == [M, K]`，`weight[i].origin_shape == [N, K]`。
7. `weight[i].storage_shape == [ceil(K / 32), ceil(N / 16), 16, 32]`。
8. `weightScale[i].origin_shape == [N, ceil(K / 64), 2]`。
9. `xScale.shape == [M, ceil(K / 64), 2]`。
10. `groupList.shape == [groupNum]`。
11. `M >= 1`，`K >= 32 && K % 32 == 0`，`N >= 128 && N % 128 == 0`。
12. `groupNum in [1, 1024]`。
13. `dequantMode == 2`，`dequantDtype == DT_FLOAT`，`quantMode == 2`，`quantDtype == FLOAT8_E4M3FN`，`transposeWeight == true`，`groupListType in {0, 1}`。
14. `weightAssistMatrix` 为空指针或空 list，`bias` / `smoothScale` 为空指针。

异常用例每次只违反一条规则，便于定位。

---

## 8. 数据文件契约

### 8.1 文件清单

| 文件 | 说明 |
| --- | --- |
| `GMMSQ_V2_MxA8W4_SMS_STC.md` | STC 说明文档 |
| `test_cases.csv` / `*.xlsx` | 表驱动用例，可后续由生成器输出 |
| `test_coverage_report.md` | 覆盖统计，可从表驱动数据生成 |
| `generate_gmmsq_mxa8w4_sms_cases.py` | 用例生成脚本，规则与 SRS / checker 同构 |

### 8.2 列定义 — 正常用例

| 列名 | 类型 | 说明 |
| --- | --- | --- |
| `case_id` | str | 唯一编号，如 `WB001` |
| `testcase` | str | 命名见第 6 节 |
| `case_type` | enum | `whitebox` / `generalized` |
| `scene` | enum | `sms` / `non_sms` |
| `group_num` | int | group 数；单多单时等于 `weight` / `weightScale` list 长度 |
| `group_list_type` | int | 0 或 1 |
| `m` / `k` / `n` | int | 轴长 |
| `x_dtype` / `weight_dtype` / `scale_dtype` / `y_dtype` | str | dtype 枚举名 |
| `x_format` / `x_scale_format` / `weight_format` / `weight_scale_format` / `y_format` | str | format 枚举名 |
| `x_shape` | str | `[M,K]` |
| `x_scale_shape` | str | `[M,ceil(K/64),2]` |
| `weight_origin_shapes` | str | tensor list 序列化，每项 `[N,K]` |
| `weight_storage_shapes` | str | tensor list 序列化，每项 `[ceil(K/32),ceil(N/16),16,32]` |
| `weight_scale_shapes` | str | tensor list 序列化，每项 `[N,ceil(K/64),2]` |
| `group_list_shape` | str | `[groupNum]` |
| `y_shape` / `y_scale_shape` | str | 输出 shape |
| `expected_tiling_key_sms` | int | 单多单期望 1，非单多单期望 0 |
| `expected_result` | enum | `success` |
| `golden_source` | str | 参考实现 / numpy / 既有 golden |

### 8.3 列定义 — 异常用例

| 列名 | 说明 |
| --- | --- |
| `exception_type` | 与 4.3 类型一致 |
| `mutated_field` | 本条异常唯一变异字段 |
| `error_description` | 可读说明 |
| `expected_error_substring` | 期望日志关键字 |
| `expected_result` | `tiling_failed` / `aclnn_failed` |

### 8.4 fuzz / aclnn 表

- 泛化行参数全部合法；异常行仅最少字段偏离。
- shape 字段建议使用 JSON 数组字符串，便于生成 tensor list。
- 对于单多单 tensor list，`weight_origin_shapes`、`weight_storage_shapes`、`weight_scale_shapes` 的长度必须等于 `group_num`。
- 异常日志关键字从 SRS 第 10 节或 host checker `OP_LOGE` 中摘取。

---

## 9. 脚本与自动化

| 脚本 | 入参 | 出参 | 说明 |
| --- | --- | --- | --- |
| `generate_gmmsq_mxa8w4_sms_cases.py` | seed、case_type、output_path | `test_cases.csv` | 生成白盒 / 泛化 / 异常表驱动数据 |
| `run_gmmsq_mxa8w4_sms_aclnn.py` | `test_cases.csv`、soc、case_filter | 运行日志、结果表 | aclnn 冒烟与精度对比 |
| `check_gmmsq_mxa8w4_sms_tiling.py` | `test_cases.csv`、case_filter | TilingKey / tilingData 检查结果 | 白盒验证 `IS_SINGLE_MULTI_SINGLE`、`groupNum`、`kSize`、`nSize` |

原则：生成逻辑与 SRS 约束、host checker 保持同构；变更约束时先改 SRS / checker，再改生成器和 STC。

---

## 10. 交付通路验证

| 通路 | 验证方式 | 结论 |
| --- | --- | --- |
| aclnn | 正常用例 + 异常用例子集，检查返回状态和日志关键字 | 正常成功，异常按预期失败 |
| Tiling 白盒 | 检查 `GetTilingKey()` 第 3 参数、`tilingData.groupNum/kSize/nSize/groupListType` | 单多单 / 非单多单 TilingKey 与 SRS 一致 |
| Kernel | 正常单多单用例与 reference / golden 对比 | 结果在既有容差内 |
| 非单多单回归 | WB011 / WB012 和既有回归集 | 行为不变，`IS_SINGLE_MULTI_SINGLE == 0` |

---

## 11. 实现/Tiling 对照（评审与测试共用）

| 实现点 | 代码逻辑 | STC 覆盖 |
| --- | --- | --- |
| 单多单识别 | `weightListSize > 1` | WB001-WB010、WB013、WB015-WB016、GEN 全部 |
| list 长度一致性 | `weightListSize == weightScaleListSize`，只要任一侧为多 tensor 就校验 | NEG001-NEG002 |
| groupNum 设置 | 单多单 `groupNum = weightListSize`；非单多单 `groupNum = wShape.GetDim(0)` | WB001-WB014、NEG016、NEG021 |
| `weight` 逐成员 desc / dtype / format | `CheckDynamicInputDesc` + `CheckWeightFormat` | NEG003、NEG005、NEG007 |
| `weightScale` 逐成员 desc / dtype / format | `CheckDynamicInputDesc` + `CheckWeightScaleFormat` | NEG004、NEG006、NEG008 |
| 空 tensor | `CheckDynamicInputShapeNotEmpty(context, index, tensorIndex, name)` | NEG009-NEG010 |
| 单多单 weight shape | storage `[ceil(K/32), ceil(N/16), 16, 32]` | WB / GEN 正常、NEG013 |
| 单多单 weightScale shape | `[N, ceil(K/64), 2]` | WB / GEN 正常、NEG014 |
| `xScale` / `groupList` shape | `[M, ceil(K/64), 2]` / `[groupNum]` | NEG015-NEG016 |
| M/K/N/groupNum 范围 | `IsInputShapeValid` | NEG017-NEG021 |
| attr 校验 | `CheckAttrs` | NEG022 |
| optional 输入 | `CheckContext` | NEG023-NEG024 |
| TilingKey | `GET_TPL_TILING_KEY(wTrans, 0, isSingleMultiSingle)` | WB013-WB014 |
| kernel 地址切换 | `if constexpr (isSingleMultiSingle)` 下使用 `GetTensorAddr(nextGroupIdx, ...)` | WB015-WB016 |

---

## 12. 风险与未覆盖

| 项 | 说明 | 跟踪方式 |
| --- | --- | --- |
| TensorList 中间空洞 | 当前计数逻辑依赖 TensorList 连续性，异常用例主要覆盖尾部缺失 / 长度不一致 | 若框架支持空洞，需新增专门 checker 和 STC |
| 超大规模性能 | 本 STC 不设置性能门限，仅覆盖功能与通路 | 由性能专项或既有回归集跟踪 |
| `groupList` 数值非法 | 本修改未新增 groupList 数值校验，仅覆盖 shape / dtype / groupListType | 由既有 GMM groupList 校验策略或后续需求跟踪 |
| 多 SOC 差异 | 当前面向 arch35 MxA8W4 | 其他 SOC 不在本 STC 范围 |

---

## 附录 A：与 SRS 的一致性检查

- [x] MxA8W4 单多单 happy path：x 单 tensor、weight 多 tensor、weightScale 多 tensor、y 单 tensor。
- [x] 非单多单 MxA8W4 回归：`IS_SINGLE_MULTI_SINGLE == 0`，weight / weightScale 连续 offset 推进行为不变。
- [x] 单多单 TilingKey 白盒检查：`IS_SINGLE_MULTI_SINGLE == 1`。
- [x] 非单多单 TilingKey 白盒检查：`IS_SINGLE_MULTI_SINGLE == 0`。
- [x] `weight` / `weightScale` tensor list 长度不一致时报错。
- [x] `weight[i]` / `weightScale[i]` desc 缺失时报错。
- [x] `weight[i]` / `weightScale[i]` dtype 非法时报错。
- [x] `weight[i]` / `weightScale[i]` format 非法时报错。
- [x] `weight[i]` / `weightScale[i]` empty tensor 报错。
- [x] `weight[i]` storage shape 非法时报错。
- [x] `weightScale[i]` shape 非法时报错。
- [x] `groupList` 长度与 `groupNum` 不一致时报错。
- [x] `weightAssistMatrix` 非空时报错。
- [x] `bias` 非空时报错。
- [x] `smoothScale` 非空时报错。
