# PR 代码概要

PR: #6883 | 算子: fused_infer_attention_score | 侧别: 混合（Kernel + Tiling + 测试）

## 变更文件概览

| # | 文件 | 侧别 | 变更类型 | 说明 |
|---|------|------|---------|------|
| 1 | attention/common/op_host/arch35/fia_tiling_nonquant_gqa.cpp | Tiling | 修改 | 新模板 TilingData 填充 keyStrides/valueStrides/kRopeStrides |
| 2 | attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp | Tiling | 修改 | 老模板 inputParams 填充 keyStrides/valueStrides/kRopeStrides |
| 3 | attention/fused_infer_attention_score/op_host/checkers/common_checker.cpp | Tiling | 修改 | 新增 CheckKVContiguous 校验函数 |
| 4 | attention/fused_infer_attention_score/op_host/checkers/common_checker.h | Tiling | 修改 | 声明 CheckKVContiguous |
| 5 | attention/common/op_kernel/fia_tiling_data_public.h | Kernel | 修改 | stridesParams 新增 getter/setter 方法 |
| 6 | attention/common/op_kernel/CopyInL1.h | Kernel | 修改 | PAShape 新增 pageStride/n2Stride；GmCopyInToL1HasRopePA 和 GmCopyInToL1PA 新增 stride 查表逻辑 |
| 7 | attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h | Kernel | 修改 | 新增 StridesParams 类；InputParamsRegbase 新增 keyStrides/valueStrides/kRopeStrides 成员 |
| 8 | attention/common/op_kernel/arch35/util_regbase.h | Kernel | 修改 | COMMON_CONST_INFO/CVSharedParams<true,true> 新增 6 个 stride 字段 |
| 9 | attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h | Kernel | 修改 | offsetCalculator.Init 传入 keyStrides.bnStride/n2Stride |
| 10 | attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h | Kernel | 修改 | fiaBaseParams→constInfo 拷贝 keyStrides/valueStrides |
| 11 | attention/common/op_kernel/arch35/flash_attention_block_cube_noquant_mla.h | Kernel | 修改 | Bmm1 迭代中 shape/ropeShape 设置 pageStride/n2Stride |
| 12 | attention/common/op_kernel/arch35/flash_attention_kernel_noquant_mla.h | Kernel | 修改 | inputParamsRegbase→constInfo 拷贝 6 个 stride 字段 |
| 13 | attention/common/op_kernel/arch35/flash_attention_noquant_block_cube.h | Kernel | 修改 | 9 个迭代函数设置 shape.pageStride/n2Stride |
| 14 | attention/common/op_kernel/arch35/flash_attention_noquant_block_vec_infer.h | Kernel | 修改 | inputParamsRegbase→sharedParams 拷贝 6 个 stride 字段（PA 条件） |
| 15 | attention/common/op_kernel/arch35/flash_attention_noquant_kernel_infer.h | Kernel | 修改 | sharedParams→constInfo 拷贝 6 个 stride 字段（PA 条件） |
| 16 | attention/common/op_kernel/memcopy/offset_calculator_v2.h | Kernel | 修改 | GM_KV_PA_BNBD/NZ 的 Init 新增 bn2Stride/n2Stride 参数 |
| 17 | attention/fused_infer_attention_score/tests/pytest/test_fia_pa_noncontiguous.py | 测试 | 新增 | PA 非连续 KV cache stride 传递测试（300行） |

---

## 文件1: attention/common/op_host/arch35/fia_tiling_nonquant_gqa.cpp（Tiling侧）

### 变更概要
新模板（FiaTilingNonQuantArch35）的 SetFATilingData 函数，在 `!isTensorV1` 条件下，将外部传入的 gert::Stride（keyStrides/valueStrides/kRopeStrides）赋给 tilingData_.baseTiling.fiaBaseParams 的对应 stridesParams 成员。

### 代码脉络
**入口**: `FiaTilingNonQuantArch35::SetFATilingData()` → 被 Tiling 调度框架调用 → 触发条件：非量化的新模板架构（arch35）

**数据流**:
`fiaInfo_->keyStrides->GetStride(0/1)` → `tilingData_.baseTiling.fiaBaseParams.keyStrides.bnStride/n2Stride`
`fiaInfo_->valueStrides->GetStride(0/1)` → `tilingData_.baseTiling.fiaBaseParams.valueStrides.bnStride/n2Stride`
`fiaInfo_->kRopeStrides->GetStride(0/1)` → `tilingData_.baseTiling.fiaBaseParams.kRopeStrides.bnStride/n2Stride`（非空时）

**分支覆盖**:
| 分支条件 | 位置(行) | 触发场景 | 处理逻辑 | 涉及 API |
|---------|---------|---------|---------|----------|
| `!fiaInfo_->isTensorV1` | 967 | 非 V1 Tensor 格式（V2+ 或新框架） | 填充 strides，V1 时跳过（return success） | gert::Stride::GetStride |
| `fiaInfo_->kRopeStrides != nullptr` | 972 | 存在 kRope 输入 | 填充 kRopeStrides；为 nullptr 时跳过 | gert::Stride::GetStride |

**关键变量流转**:
| 变量 | 来源 | 用途 | 流转路径 |
|------|------|------|----------|
| fiaInfo_->keyStrides | 外部输入 gert::Stride | key BM 维度的非连续 stride | TilingInfo → tilingData_.baseTiling.fiaBaseParams.keyStrides → Kernel constInfo |
| fiaInfo_->valueStrides | 外部输入 gert::Stride | value BM 维度的非连续 stride | TilingInfo → tilingData_.baseTiling.fiaBaseParams.valueStrides → Kernel constInfo |
| fiaInfo_->kRopeStrides | 外部输入 gert::Stride | keyRope BM 维度的非连续 stride | TilingInfo → tilingData_.baseTiling.fiaBaseParams.kRopeStrides → Kernel constInfo |
| fiaInfo_->isTensorV1 | 框架/环境变量 | Tensor 格式版本标识 | V1 时跳过 stride 设置（兼容旧格式） |

**核心 API**: `gert::Stride::GetStride(dimIndex)` — 获取指定维度的 stride 值

---

## 文件2: attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp（Tiling侧）

### 变更概要
老模板（FusedInferAttentionScoreTilingImpl）的 SetFATilingData 函数，在 `!isTensorV1` 条件下，通过 inputParams.keyStrides/valueStrides/kRopeStrides 的 setter 方法填充 stride 值。

### 代码脉络
**入口**: `FusedInferAttentionScoreTilingImpl::SetFATilingData(const FiaTilingInfo &fiaInfo)`

**数据流**:
`fiaInfo.keyStrides->GetStride(0)` → `inputParams.keyStrides.set_bnStride(...)`
`fiaInfo.keyStrides->GetStride(1)` → `inputParams.keyStrides.set_n2Stride(...)`
（valueStrides、kRopeStrides 同理）

**分支覆盖**:
| 分支条件 | 位置(行) | 触发场景 | 处理逻辑 |
|---------|---------|---------|---------|
| `!fiaInfo.isTensorV1` | 1868 | 非 V1 Tensor 格式 | 填充 strides，V1 时跳过 |
| `fiaInfo.kRopeStrides != nullptr` | 1873 | 存在 kRope 输入 | 填充 kRopeStrides；为 nullptr 时跳过 |

**关键变量**:
| 变量 | 来源 | 用途 |
|------|------|------|
| inputParams.keyStrides | InputParamsRegbase 成员（StridesParams 类型） | 存储 key 的 bnStride/n2Stride |
| inputParams.valueStrides | InputParamsRegbase 成员 | 存储 value 的 bnStride/n2Stride |
| inputParams.kRopeStrides | InputParamsRegbase 成员 | 存储 kRope 的 bnStride/n2Stride |

### 变量溯源
| 变量 | 声明(文件:行) | 初始化(文件:行) | 校验(文件:行) | 来源类型 |
|------|-------------|----------------|-------------|---------|
| fiaInfo.keyStrides | fia_tiling_info.h:235 | 外部赋值(nullptr或gert::Stride*) | CheckKVContiguous(common_checker.cpp:1384) | 外部输入 |
| fiaInfo.isTensorV1 | fia_tiling_info.h:240 | 默认 false | CheckKVContiguous 中以 isTensorV1 为前置条件 | 外部输入/框架标识 |
| inputParams | FlashAttentionScoreSimplifiedTilingData 成员 | faRunTilingAdapter_ 构造时 | - | TilingData 结构 |

---

## 文件3: attention/fused_infer_attention_score/op_host/checkers/common_checker.cpp（Tiling侧）

### 变更概要
新增 `CommonChecker::CheckKVContiguous` 校验函数，在 `CheckMultiParaConsistency` 中被调用。该函数按不同存储模式/Layout 验证 key/value 张量的"非连续维度"是否在允许范围内。

### 代码脉络
**入口**: `CommonChecker::CheckKVContiguous(const FiaTilingInfo &fiaInfo) const` → 被 `CheckMultiParaConsistency` 调用（行1498）→ 触发条件：非 910B 平台 && enableNonQuant_ == true

**数据流**:
`fiaInfo.opParamInfo.key.shape->GetStorageShape()` → `CheckTensorContiguous(keyDimNum, keyShape, keyStrides, dimIndex)` → `dimIndex`（首个非连续维度索引）→ 按 Layout 规则判定

**分支覆盖**:
| 分支条件 | 位置(行) | 触发场景 | 处理逻辑 |
|---------|---------|---------|---------|
| `fiaInfo.isTensorV1` | 1386 | V1 Tensor 格式 | 直接返回 GRAPH_SUCCESS（跳过检查） |
| `fiaInfo.kvStorageMode == TENSOR_LIST` | 1389 | TensorList 存储模式 | 直接返回 GRAPH_SUCCESS（跳过检查） |
| `!fiaInfo.pageAttentionFlag` | 1398 | 非 PA 场景 | key/value 必须全部连续，否则报错 "must be contiguous" |
| `fiaInfo.kvLayout == BnBsH` (BBND) | 1409 | PA BBND 布局 | 只允许 dim0 非连续，否则报错 "only supports non-contiguous tensors in dimension 0" |
| `fiaInfo.kvLayout == BnNBsD or NZ` | 1430 | PA BNBD/NZ 布局 | 只允许 dim0 或 dim1 非连续，否则报错 "only supports non-contiguous tensors in dimensions 0 or 1" |
| `fiaInfo.socVersion != ASCEND910B && enableNonQuant_ && CheckKVContiguous != SUCCESS` | 1498-1501 | 新平台+非量化 | 校验失败则返回 GRAPH_FAILED |

**关键变量流转**:
| 变量 | 来源 | 用途 |
|------|------|------|
| fiaInfo.keyStrides / valueStrides | gert::Stride 外部输入 | 传入 CheckTensorContiguous 判断各维度连续性 |
| dimIndex | CheckTensorContiguous 输出参数 | 首个非连续维度的索引 |
| fiaInfo.kvLayout | FiaLayout 枚举 | 决定允许的非连续维度范围 |
| fiaInfo.pageAttentionFlag / kvStorageMode | 算子属性 | 决定校验策略（PA/非PA/TensorList） |

### 变量溯源
| 变量 | 声明(文件:行) | 初始化(文件:行) | 校验(文件:行) | 来源类型 |
|------|-------------|----------------|-------------|---------|
| fiaInfo.isTensorV1 | fia_tiling_info.h:240 | 默认 false | CheckKVContiguous 前置判断 | 外部输入/框架标识 |
| fiaInfo.keyStrides | fia_tiling_info.h:235 | 输入 tensor 的 stride | CheckKVContiguous 中通过 CheckTensorContiguous 验证 | 外部输入 |
| fiaInfo.kvLayout | FiaTilingInfo 成员 | 框架填充 | CheckKVContiguous 中按值分支 | 算子属性 |
| fiaInfo.pageAttentionFlag | fia_tiling_info.h:247 | 算子属性 | CheckKVContiguous 中按 bool 分支 | 算子属性 |

---

## 文件5: attention/common/op_kernel/fia_tiling_data_public.h（Kernel侧 — 公共数据结构）

### 变更概要
为 `stridesParams` 结构体新增 getter/setter 方法（set_bnStride/get_bnStride/set_n2Stride/get_n2Stride），供 Tiling 侧填充和 Kernel 侧读取使用。**注意**：`FiaBaseParams` 中已有 `keyStrides/valueStrides/kRopeStrides` 成员（stridesParams 类型），本次变更不影响结构体布局，仅新增方法。

### 代码脉络
**入口**: Tiling 侧通过 `tilingData_.baseTiling.fiaBaseParams.keyStrides.bnStride = value` 赋值 → Kernel 侧通过 `fiaBaseParams.keyStrides.bnStride` 读取

**关键变量**:
| 变量 | 来源 | 用途 |
|------|------|------|
| stridesParams::bnStride | Tiling 侧赋值（GmLayout 的 blockNum stride） | 描述 block 维度（BN）的非连续 stride |
| stridesParams::n2Stride | Tiling 侧赋值（GmLayout 的 n2 stride） | 描述 head 维度（N2）的非连续 stride |

---

## 文件6: attention/common/op_kernel/CopyInL1.h（Kernel侧 — 数据搬运）

### 变更概要
1. `PAShape` 结构体新增 `pageStride` 和 `n2Stride` 字段（默认值 0）
2. `GmCopyInToL1HasRopePA` 函数：offset 计算从固定公式 `idInBlockTable * B * H * D` 改为条件查表（pageStride > 0 时用查表 offset，n2Stride > 0 时用查表 n2offset）
3. `GmCopyInToL1PA` 函数：同上逻辑，offset 从固定公式改为条件查表

### 代码脉络
**入口**: `GmCopyInToL1HasRopePA` / `GmCopyInToL1PA` → 被各 block_cube 迭代函数调用 → 触发条件：PA 模式下搬运 KV 数据到 L1

**数据流**:
`PAShape.pageStride` (来自 constInfo keyBnStride/valueBnStride) → `offset = idInBlockTable * pageStride`（非连续公式）
`PAShape.n2Stride` (来自 constInfo keyN2Stride/valueN2Stride) → `offset += n2Idx * n2Stride`（非连续 n2 偏移）

**分支覆盖**（GmCopyInToL1HasRopePA，行227-314）:
| 分支条件 | 位置(行) | 触发场景 | 处理逻辑 |
|---------|---------|---------|---------|
| `shape.pageStride > 0` | 249 | PA 非连续 kv（stride 不为0） | offset = idInBlockTable * pageStride |
| else (pageStride == 0) | 250 | PA 连续 kv（stride=0） | offset = idInBlockTable * B * H * D（默认公式） |
| `ropeShape.pageStride > 0` | 251 | PA 非连续 kRope | keyRopeOffset = idInBlockTable * pageStride |
| `kvLayout == NZ` | 253 | NZ 布局 | DataCopyGmNZToL1；offset 后缀用 n2Stride（条件） |
| `kvLayout == BBH` | 276 | BBH 布局 | n2Stride 不适用（不改 offset 结构） |
| else (BNBD) | 283 | BNBD 布局 | n2Stride > 0 时用查表 offset，否则默认公式 |
| `shape.n2Stride > 0` | 254, 284 | n2 维度非连续 | n2Idx * n2Stride 替换 n2Idx * blockSize * headDim |
| `ropeShape.n2Stride > 0` | 260, 291 | kRope n2 非连续 | n2Idx * ropeShape.n2Stride 替换默认公式 |

**关键变量流转**:
| 变量 | 来源 | 用途 | 流转路径 |
|------|------|------|----------|
| shape.pageStride | constInfo.keyBnStride/valueBnStride（来自 tiling） | 非连续 PA 的 block 维度 stride | Tiling → constInfo → shape → GmCopyInToL1PA |
| shape.n2Stride | constInfo.keyN2Stride/valueN2Stride（来自 tiling） | 非连续 PA 的 head 维度 stride | Tiling → constInfo → shape → GmCopyInToL1PA |
| startPos.n2Idx | 调用者传入的 Position 参数 | 当前 head 索引 | 循环迭代产生 |

**核心 API**: `DataCopyGmNZToL1`（NZ→L1块搬）、`DataCopyGmNDToL1`（ND→L1块搬）

---

## 文件7: attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h（Kernel侧 — Tiling 数据结构）

### 变更概要
1. 新增 `StridesParams` 类（bnStride + n2Stride + getter/setter）
2. `InputParamsRegbase` 新增 `keyStrides/valueStrides/kRopeStrides` 成员及配套 getter/setter

### 代码脉络
**入口**: Tiling 侧通过 `inputParams.set_keyStrides(...)` 赋值 → Kernel 侧通过 `inputParamsRegbase.keyStrides.bnStride` 读取

**关键变量**:
| 变量 | 声明(行) | 用途 |
|------|---------|------|
| InputParamsRegbase::keyStrides | 154 | key 的 stride 参数传递（strides 字段） |
| InputParamsRegbase::valueStrides | 155 | value 的 stride 参数传递 |
| InputParamsRegbase::kRopeStrides | 156 | kRope 的 stride 参数传递 |

---

## 文件8: attention/common/op_kernel/arch35/util_regbase.h（Kernel侧 — 运行时常量结构体）

### 变更概要
1. `COMMON_CONST_INFO` 宏新增 6 个 uint64_t stride 字段：keyBnStride/keyN2Stride/valueBnStride/valueN2Stride/keyRopeBnStride/keyRopeN2Stride
2. `CVSharedParams<true, true>`（推理+PA 模板）新增相同的 6 个字段

**注意**：`CVSharedParams<true, false>`（推理+非PA）未新增 stride 字段，因为 stride 仅在 PA 场景使用。

### 关键变量:
| 变量 | 所属结构体 | 用途 |
|------|--------|------|
| constInfo.keyBnStride/N2Stride | ConstInfo (COMMON_CONST_INFO) | PA 场景 key 的非连续 block/head stride |
| constInfo.valueBnStride/N2Stride | ConstInfo | PA 场景 value 的非连续 block/head stride |
| constInfo.keyRopeBnStride/N2Stride | ConstInfo | PA 场景 kRope 的非连续 block/head stride |
| sharedParams.keyBnStride... | CVSharedParams<true,true> | 跨 cube/vec 核共享 stride |

---

## 文件9: attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h（Kernel侧）

### 变更概要
`InitKVBuffer` 函数中，PA BNBD/NZ 布局的 `offsetCalculator.Init` 调用新增 `constInfo.keyStrides.bnStride` 和 `constInfo.keyStrides.n2Stride` 参数。

### 代码脉络
**入口**: `FiaBlockCubeNoquantGqa::InitKVBuffer` → 被 Kernel Init 阶段调用

**分支覆盖**:
| 分支条件 | 位置(行) | 处理逻辑 |
|---------|---------|---------|
| `GM_KV_PA_BNBD` | 274-277 | Init(n2Size, blockSize, headDim, blockTable, maxBlockNum, bnStride, n2Stride) |
| `GM_KV_PA_NZ` | 278-283 | Init(n2Size, blockSize, d1, d0, blockTable, maxBlockNum, bnStride, n2Stride) |
| 其他（BNSD/TND） | 284-289 | 无需 stride 参数（非 PA） |

**核心 API**: `offsetCalculator.Init(...)` — 传递 stride 给 GmLayout 用于后续 GetOffset 计算

---

## 文件10: attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h（Kernel侧）

### 变更概要
`ComputeConstInfo` 阶段从 `fiaBaseParams.keyStrides/valueStrides` 拷贝到 `constInfo.keyStrides/valueStrides`（值拷贝）。

### 代码脉络
**入口**: `FiaKernelNoquantGqa::ComputeConstInfo` → 一次性读取 tiling 数据到运行时 constInfo

**关键变量流转**: `fiaBaseParams.keyStrides.bnStride`（来自 TilingData）→ `constInfo.keyStrides.bnStride`（运行时使用）

---

## 文件11-12: flash_attention_block_cube_noquant_mla.h / flash_attention_kernel_noquant_mla.h（Kernel侧）

### 变更概要
- `flash_attention_kernel_noquant_mla.h`（行373-381）：从 `inputParamsRegbase.keyStrides/valueStrides/kRopeStrides` 拷贝 6 个 stride 到 constInfo
- `flash_attention_block_cube_noquant_mla.h`（行238-244）：Bmm1 迭代中设置 `shape.pageStride/n2Stride` 和 `ropeShape.pageStride/n2Stride`

**数据流**: `inputParamsRegbase.keyStrides.bnStride` → `constInfo.keyBnStride` → `shape.pageStride` → `GmCopyInToL1HasRopePA`

---

## 文件13: attention/common/op_kernel/arch35/flash_attention_noquant_block_cube.h（Kernel侧 — 变更量最大）

### 变更概要
在 9 个迭代函数中添加 `shape.pageStride = constInfo.keyBnStride` / `shape.n2Stride = constInfo.keyN2Stride`（或 value 对应字段）赋值：

| 函数 | 行号 | 设置字段 | 数据类型 |
|------|------|---------|---------|
| IterateBmm2L1SplitN | 617-618 | valueBnStride/valueN2Stride | GmCopyInToL1PA |
| IterateBmm2 | 744-745 | valueBnStride/valueN2Stride | GmCopyInToL1PA |
| IterateBmm1NdL0Split | 1022-1023 | keyBnStride/keyN2Stride (+rope) | GmCopyInToL1HasRopePA（有rope） |
| IterateBmm1DnSplitK | 1220-1221 | keyBnStride/keyN2Stride | GmCopyInToL1PA |
| IterateBmm1Nd | 1616-1617 | keyBnStride/keyN2Stride | GmCopyInToL1PA |
| IterateBmm1NdL1SplitK | 1805-1806 | keyBnStride/keyN2Stride | GmCopyInToL1PA |
| IterateBmm1Dn | 1956-1957 | keyBnStride/keyN2Stride | GmCopyInToL1PA |
| IterateBmm1MLAFullQuant | 2101-2102 / 2108-2109 | keyBnStride/keyN2Stride (+rope) | GmCopyInToL1PA（nope）+ GmCopyInToL1HasRopePA（rope） |

---

## 文件14-15: flash_attention_noquant_block_vec_infer.h / flash_attention_noquant_kernel_infer.h（Kernel侧）

### 变更概要
- `flash_attention_noquant_block_vec_infer.h`（行205-210）：在 `isPa` 条件编译下，从 `inputParamsRegbase.keyStrides/valueStrides/kRopeStrides` 拷贝到 sharedParams
- `flash_attention_noquant_kernel_infer.h`（行79-84）：在 `isPa` 条件编译下，从 sharedParams 拷贝到 this->constInfo

**数据流链**: `InputParamsRegbase` → `sharedParams` → `constInfo` → `PAShape` → `GmCopyInToL1PA`

---

## 文件16: attention/common/op_kernel/memcopy/offset_calculator_v2.h（Kernel侧）

### 变更概要
`OffsetCalculatorImpl<GM_KV_PA_BNBD>` 和 `OffsetCalculatorImpl<GM_KV_PA_NZ>` 的 `Init` 方法新增可选参数 `bn2Stride` 和 `n2Stride`（默认值 0）。在 310/310R6 架构上传递给 `GmLayout::MakeLayout`。

---

## 文件17: attention/fused_infer_attention_score/tests/pytest/test_fia_pa_noncontiguous.py（测试）

### 变更概要
新增 300 行 pytest 测试文件，覆盖：
- **新模板正例**（10个）：BnNBsD/BnBsH/NZ 的连续/dim0非连续/dim01非连续、ROPE_COMBINE (D=192) 非连续
- **新模板异常**（1个）：BnBsH dim01 非连续应报错
- **老模板正例**（11个）：ROPE_SPLIT/D≠128/D=256/D=64/NZ/非PA 等各种场景的 stride 测试
- **老模板异常**（4个）：BnBsH dim01、非PA dim0/dim01、非PA kRope dim0 应报错

---

## 跨文件关联

### Tiling → Kernel 数据流

**整体数据流**:
```
外部 tensor stride (gert::Stride)
    ↓
FiaTilingInfo.keyStrides/valueStrides/kRopeStrides (fia_tiling_info.h:235-239)
    ↓ ┌─ 新模板: fia_tiling_nonquant_gqa.cpp → tilingData_.baseTiling.fiaBaseParams.keyStrides.valueStrides.kRopeStrides
    ├─ 老模板: fused_infer_attention_score_tiling_impl.cpp → inputParams.keyStrides.valueStrides.kRopeStrides
    └─ 校验: common_checker.cpp CheckKVContiguous → 合法非连续维度规则
    ↓
Kernel侧读取:
    ├─ GQA路径: fia_kernel_noquant_gqa.h → constInfo.keyStrides/valueStrides → fia_block_cube_noquant_gqa.h → offsetCalculator.Init(..., bnStride, n2Stride)
    ├─ regbase路径: InputParamsRegbase → flash_attention_noquant_block_vec_infer.h → sharedParams → flash_attention_noquant_kernel_infer.h → constInfo
    └─ MLA路径: InputParamsRegbase → flash_attention_kernel_noquant_mla.h → constInfo.keyBnStride
    ↓
使用端:
    ├─ flash_attention_noquant_block_cube.h → shape.pageStride/n2Stride = constInfo.keyBnStride/valueBnStride
    ├─ flash_attention_block_cube_noquant_mla.h → shape/ropeShape.pageStride/n2Stride
    └─ CopyInL1.h: GmCopyInToL1HasRopePA / GmCopyInToL1PA → offset = shape.pageStride > 0 ? pageStride倍乘 : 默认公式
```

**TilingData 结构名** 中本次涉及的字段（Tiling 填充 → Kernel 使用）:
| TilingData 字段 | Tiling 填充位置 | Kernel 使用位置 | 用途 |
|----------|----------|----------|------|
| fiaBaseParams.keyStrides.bnStride | fia_tiling_nonquant_gqa.cpp:968 | fia_kernel_noquant_gqa.h:252 / fia_block_cube_noquant_gqa.h:277 | key 的 block 维度 stride |
| fiaBaseParams.keyStrides.n2Stride | fia_tiling_nonquant_gqa.cpp:969 | fia_kernel_noquant_gqa.h:253 / fia_block_cube_noquant_gqa.h:277 | key 的 head 维度 stride |
| fiaBaseParams.valueStrides.bnStride | fia_tiling_nonquant_gqa.cpp:970 | fia_kernel_noquant_gqa.h:254 | value 的 block 维度 stride |
| fiaBaseParams.valueStrides.n2Stride | fia_tiling_nonquant_gqa.cpp:971 | fia_kernel_noquant_gqa.h:255 | value 的 head 维度 stride |
| inputParamsRegbase.keyStrides.bnStride | fused_infer_attention_score_tiling_impl.cpp:1869 | flash_attention_kernel_noquant_mla.h:376 | key 的 block 维度 stride（regbase路径） |
| inputParamsRegbase.keyStrides.n2Stride | fused_infer_attention_score_tiling_impl.cpp:1870 | flash_attention_kernel_noquant_mla.h:377 | key 的 head 维度 stride（regbase路径） |
| inputParamsRegbase.kRopeStrides.bnStride | fused_infer_attention_score_tiling_impl.cpp:1874 | flash_attention_kernel_noquant_mla.h:380 | kRope 的 block 维度 stride |
| inputParamsRegbase.kRopeStrides.n2Stride | fused_infer_attention_score_tiling_impl.cpp:1875 | flash_attention_kernel_noquant_mla.h:381 | kRope 的 head 维度 stride |

### Include / 调用链
| 调用方 | 被调用方 | 关系 |
|--------|---------|------|
| flash_attention_noquant_block_cube.h | CopyInL1.h (GmCopyInToL1PA, GmCopyInToL1HasRopePA) | 包含+调用 |
| flash_attention_block_cube_noquant_mla.h | CopyInL1.h (GmCopyInToL1HasRopePA) | 包含+调用 |
| fia_block_cube_noquant_gqa.h | offset_calculator_v2.h (OffsetCalculator.Init) | 包含+调用 |
| flash_attention_noquant_block_vec_infer.h | flash_attention_score_tiling_regbase.h | 包含 InputParamsRegbase |
| flash_attention_noquant_kernel_infer.h | util_regbase.h (ConstInfo, CVSharedParams) | 包含 struct 定义 |
| common_checker.cpp | fia_tiling_info.h (FiaTilingInfo) | 校验使用 |

### 校验链
| 校验函数 | 位置 | 校验内容 | 何时调用 |
|---------|------|---------|---------|
| CheckKVContiguous | common_checker.cpp:1384 | key/value 非连续维度的合法性 | CheckMultiParaConsistency 中，非910B且enableNonQuant_时 |
| CheckTensorContiguous | （已有函数） | 返回首个非连续维度索引 dimIndex | CheckKVContiguous 内部调用 |
| isTensorV1 前置判断 | common_checker.cpp:1386 | 跳过 V1 Tensor 场景 | CheckKVContiguous 入口 |

### 同步标志/常量关联
| 定义位置 | 值 | 引用位置 | 影响范围 |
|---------|-----|---------|---------|
| PAShape::pageStride (默认0) | CopyInL1.h:40 | GmCopyInToL1HasRopePA/GmCopyInToL1PA | ==0 时走默认连续公式；>0 时走查表 stride 公式 |
| PAShape::n2Stride (默认0) | CopyInL1.h:41 | GmCopyInToL1HasRopePA/GmCopyInToL1PA | ==0 时走默认 n2*blockSize*headDim；>0 时走 n2*stride |
| isTensorV1 | fia_tiling_info.h:240 | Tiling端 SetFATilingData + common_checker | true 时跳过 stride 赋值和校验 |
| fiaInfo.socVersion != ASCEND910B | common_checker.cpp:1498 | CheckKVContiguous 调用判断 | 910B平台跳过非连续检查 |

---

## 高性能设计（Kernel 侧）

**流水线模式**: 混合（Cube+Vec 协同，同步/异步交织）

**流水线设计**:
| 机制 | 状态 | 设计意图 |
|------|------|----------|
| stride 查表开关（pageStride/n2Stride > 0） | 新增 | 运行时零开销分支，连续场景走默认快速路径，非连续场景走 stride 查表 |
| offsetCalculator stride 传递 | 新增 | 仅在 310/310R6 架构启用 GmLayout::MakeLayout(n2, bs, d, bn2Stride, n2Stride) |

**切分策略**:
- 多核切分：按 BN 维度切分（原有逻辑未变）
- KV cache 切分：按 PA block 粒度（未变），非连续场景通过 stride 支持跨 block 的 gap

**Buffer 规划**（未变，与 stride 无关，stride 仅影响地址计算）:
| Buffer | 类型 | 用途 |
|--------|------|------|
| mm1ATensor/mm1BTensor | L1 TBuf | QK matmul 输入 |
| mm2BTensor | L1 TBuf | PV matmul 输入 |
| l1Tensor (CopyInL1) | L1 TBuf | GM→L1 KV 搬运目标 |

---

## 设计要点总结

1. **向后兼容**：所有新增 stride 字段默认值为 0，连续场景行为与既有一致
2. **条件启用**：通过 `isTensorV1` 标志位和 `pageStride > 0` / `n2Stride > 0` 运行时条件实现新旧路径切换
3. **平台限制**：checker 中 stride 校验在 910B 平台上跳过（socVersion != ASCEND910B）
4. **多路径覆盖**：两套模板（新模板 FiaTilingNonQuantArch35 + 老模板 FusedInferAttentionScoreTilingImpl）均已适配
5. **copy 函数间隙**：`GmScaleCopyInToL1PAForND` 和 `GmScaleCopyInToL1PAForDN` 未适配 stride（后量化场景），因为后量化 scale 不走 stride 路径
6. **CacheLine 约束**：CVSharedParams<true,true> 新增 6 个 uint64_t（共 48 字节），注释要求 ≤128 Bytes，检查是否仍有余量
