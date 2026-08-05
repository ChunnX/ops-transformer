# QuantFlashAttn HIF8 Per-Tensor 量化模式实现文档

## 1. 背景

### 1.1 HIF8 Per-Tensor 量化背景

QuantFlashAttn 算子此前仅支持 MXFP8（`quant_compute_mode=1`）量化模式，该模式采用 **per-block group 量化**（每 32 个元素一组，使用 `FLOAT8_E8M0` 类型的 block scale）。随着大模型推理场景对精度和性能的需求演进，`HIFLOAT8`（HiFloat8）数据类型被引入，其核心优势在于：

- **Per-Tensor 量化**：整个 Tensor 共享一个 `FLOAT32` 标量 descale，无需 per-block scale 数据，简化了数据准备和内存访问
- **更高精度**：HIFLOAT8 采用 1 位符号位 + 4 位指数位 + 3 位尾数位格式，相较 FP8_E4M3FN 在部分场景下精度更优
- **更少内存开销**：descale tensor 从多维 block scale 数组简化为 `shape=(1,)` 的标量，显著减少 descale 数据的内存占用

### 1.2 特性特点分析

| 维度 | MXFP8 (quant_compute_mode=1) | HIF8 (quant_compute_mode=7) |
|---|---|---|
| **输入数据类型** | `DT_FLOAT8_E4M3FN` | `DT_HIFLOAT8` |
| **Descale 数据类型** | `DT_FLOAT8_E8M0` (block scale) | `DT_FLOAT` (per-tensor 标量) |
| **Descale Shape** | 4D/5D/6D (per-block) | 1D `(1,)` |
| **Descale 读取方式** | Cube 侧 `InitQScaleBuffer`/`InitKScaleBuffer`/`InitVScaleBuffer` 初始化 GM | Kernel 侧直接读取 GM 标量 `*((__gm__ float *)dequantScaleQuery)` |
| **Descale 应用位置** | Cube 侧 `MatmulFullMX` 传 scale tensor 到 L0 | Vec 侧 `ProcessVec2OnUb` 中 `Muls(vec2Res, vScaleValue)` 标量乘 |
| **Q/K descale 应用** | Cube 侧 `MatmulFullMX` 的 scale tensor 参数 | Vec 侧 `ProcessVec1Dn` 中 `descaleQK` 乘入 `dScale`（`dScale = scale * deScaleQK`） |
| **V descale 应用** | Cube 侧 `MatmulFullMX` 的 v_scale tensor | Vec 侧 `ProcessVec2OnUb` 中 `Muls(vec2ResUb, vec2ResUb, vScaleValue_)` |
| **基本块大小** | S1=128, S2=512 (DN模式 S2 对半切为256) | S1=128, S2=256 (不对半切) |
| **SubLoop** | 有（S2 对半切分两次处理） | 无（S2=256 整块处理） |
| **Layout 支持** | TND / PA_BNBD / PA_NZ | TND only (prefill DN 模式) |
| **HeadDim 约束** | 64/128/256/512/576 | ≤128 |
| **Mask 处理** | `maskLine` 标量方式（`ComputeMaskLineDN` + `Arange` + `Compare GE`） | `maskLine` 标量方式（与 MXFP8 对齐） |
| **Kernel 路径** | DN + ND | DN only |
| **VF 函数** | `ProcessVec1VfDnMxfp8`（带 subLoop、pScaleSubLoop0 block scale） | `ProcessVec1VfDnHif8`（无 subLoop、无 pScaleSubLoop0，带 maskLine） |
| **Cast 指令** | `castTraitRintZero/One/Two/Three`（FP8_E4M3/E5M2 舍入模式） | `castTraitZero/One/Two/Three`（HIF8 截断模式） |
| **Cube scale 处理** | `SCALE_T = fp8_e8m0_t`，`MatmulFullMX` 传 scale tensor | 无 `SCALE_T`，`MatmulFullMX` 传 `dummyScale`（空 tensor） |

## 2. 主要方案

### 2.1 方案大纲与实现流程

```
┌──────────────────────────────────────────────────────────────┐
│                      Host 侧 (Tiling)                         │
│                                                               │
│  1. GetQuantMode(): 校验 quant_compute_mode ∈ {1, 7}          │
│  2. ParseAxisInfo(): HIF8 时 early return（跳过 4D/5D 校验）   │
│  3. Checker 校验:                                             │
│     ├── Q/K/V dtype = DT_HIFLOAT8                             │
│     ├── descale dtype = DT_FLOAT, shape = (1,), 1D            │
│     ├── layout: kv=TND only, out∈{TND,BSND,BNSD}             │
│     └── headDim ≤ 128                                         │
│  4. Tiling 配置:                                              │
│     ├── SplitPolicy: SINNER_256（非 SINNER_512）               │
│     ├── Config: 2/3 (S2Aligned256) + HIF8 fallback            │
│     └── QuantMode: QFA_HIF8_FP32_PREFILL = 7                  │
│  5. 生成 TilingKey                                            │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                    Kernel 侧 (Entry)                          │
│                                                               │
│  #elif (ORIG_DTYPE_Q == DT_HIFLOAT8)                          │
│    → quant_flash_attn_hif8<...>()                             │
│      ├── INPUT_T = hifloat8_t                                 │
│      ├── OUT_T = bfloat16_t                                   │
│      ├── useDn = true (仅 DN 路径)                            │
│      ├── CubeBlock = QuantFlashAttnBlockCubeHif8              │
│      ├── VecFaBlock = QuantFlashAttnBlockVecHif8              │
│      ├── VecFdBlock = QuantFlashAttnBlockVecFlashDecode       │
│      └── Kernel = QuantFlashAttnKernelHif8                    │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                 Kernel.Init() 执行流程                        │
│                                                               │
│  1. 从 GM 读取 per-tensor descale 标量:                       │
│     ├── qScale = *((__gm__ float *)dequantScaleQuery)         │
│     ├── kScale = *((__gm__ float *)dequantScaleKey)           │
│     └── vScale = *((__gm__ float *)dequantScaleValue)         │
│                                                               │
│  2. 传递给 Vec block:                                         │
│     ├── vecFaBlock.SetDescaleQK(qScale, kScale)               │
│     │   └── descaleQKValue_ = qScale * kScale                 │
│     └── vecFaBlock.SetVScale(vScale)                          │
│         └── vScaleValue_ = vScale                             │
│                                                               │
│  3. Cube block 不处理 descale:                                │
│     └── InitCubeBlock() 接受 descale 参数但不初始化 scale GM  │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                 Kernel.Process() 执行流程                     │
│                                                               │
│  ┌─ Cube 侧 (AIC) ────────────────────────────────────────┐  │
│  │                                                        │  │
│  │  IterateBmm1Dn:                                        │  │
│  │    ├── CopyQueryTile: L1 加载 Q (hifloat8_t)           │  │
│  │    ├── CopyKeyTile: L1 加载 K (hifloat8_t)             │  │
│  │    ├── s2BaseSizeCur = s2BaseSize (不切半)              │  │
│  │    ├── MatmulFullMX<Q_T, KV_T, T, 256, 128, dBaseSize, │  │
│  │    │   ..., float, float, INPUT_T, INPUT_T>(            │  │
│  │    │   ..., dummyScale, dummyScale)  ← 无 scale         │  │
│  │    └── FixpipeMm1Dn: L0C→UB                           │  │
│  │                                                        │  │
│  │  IterateBmm2Dn:                                        │  │
│  │    ├── CopyValueSlice: L1 加载 V (hifloat8_t)          │  │
│  │    ├── MatmulFullMX<..., dummyScale, dummyScale>       │  │
│  │    └── FixpipeMm2: L0C→UB                             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─ Vec 侧 (AIV) ─────────────────────────────────────────┐  │
│  │                                                        │  │
│  │  ProcessVec1Dn:                                        │  │
│  │    ├── ComputeMaskLineDN: Arange+Compare 生成 mask     │  │
│  │    ├── descaleQK = 1.0 (局部变量，不使用 descaleQKValue_)│  │
│  │    ├── dScale = scale * deScaleQK (scale 已含 Q/K descale)│
│  │    ├── ProcessVec1VfDnHif8:                            │  │
│  │    │   ├── castTraitZero/One/Two/Three (hif8 4路 cast) │  │
│  │    │   ├── Or + Gather: 合并重排 cast 结果              │  │
│  │    │   └── maskLine: Arange+Compare GE (无需 mask tensor)│  │
│  │    └── SoftmaxDataCopyOut: sum/max 输出                │  │
│  │                                                        │  │
│  │  ProcessVec2OnUb:                                      │  │
│  │    ├── FlashUpdateNew/FlashUpdateLastNew: softmax 更新  │  │
│  │    ├── Muls(vec2ResUb, vec2ResUb, vScaleValue_): V descale│
│  │    ├── Cast(attenOut, vec2ResUb, CAST_ROUND): fp32→bf16│  │
│  │    └── Bmm2DataCopyOutTrans: 输出到 GM                  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 2.2 Tiling 侧与 MXFP8 详细对比

| Tiling 项目 | MXFP8 | HIF8 | 差异原因 |
|---|---|---|---|
| `IsCapable()` | `==MXFP8` | `==MXFP8 \|\| ==HIF8` | 新增 HIF8 支持 |
| `SplitPolicy()` | `SINNER_512` | `SINNER_256` | HIF8 基本块 S2=256，不切半 |
| `UpdateTilingKeyConfig()` | Config 0/1 (S2=512) | Config 2/3 (S2=256) + HIF8 fallback | 不同基本块大小 |
| `UpdateTilingKeyQuantMode()` | `QFA_MXFP8_FP32_PREFILL/DECODE` | `QFA_HIF8_FP32_PREFILL` | 新增 tiling key 值 7 |
| `CheckSingleParaQuantMode` | `{1}` | `{1, 7}` | 新增枚举值 |
| `CheckSingleParaQDescale` | dtype: E8M0/FLOAT, dim: 4D/5D | dtype: FLOAT only, dim: 1D only | per-tensor 标量 |
| `CheckSingleParaKDescale` | dtype: E8M0/FLOAT, dim: 4D/5D/6D | dtype: FLOAT only, dim: 1D only | per-tensor 标量 |
| `CheckSingleParaVDescale` | dtype: E8M0/FLOAT, dim: 4D/5D/6D | dtype: FLOAT only, dim: 1D only | per-tensor 标量 |
| `CheckQDescaleShape` | 4D/5D per-block shape | `(1,)` per-tensor early return | 不同 descale 布局 |
| `CheckKDescaleShape` | 4D/5D/6D per-block shape | `(1,)` per-tensor early return | 不同 descale 布局 |
| `CheckVDescaleShape` | 4D/5D/6D per-block shape | `(1,)` per-tensor early return | 不同 descale 布局 |
| `CheckDescaleDtype` | E8M0 (if MXFP8) | FLOAT (if HIF8) + E8M0 (if MXFP8) | 两个独立 lambda 分支 |
| `CheckLayoutConstraint` | kv: TND/PA_BNBD/PA_NZ, out: TND | kv: TND only, out: TND/BSND/BNSD | HIF8 仅 TND |
| `CheckHeadDimConstraint` | 无 | ≤128 | 模板实例化仅覆盖 D/DV ∈ {64, 128} |
| `ParseAxisInfo()` | 4D/5D dim 校验 | HIF8 early return（跳过 dim 校验） | per-tensor 无需 dim 校验 |
| `GetQuantMode()` | 仅 mode=1 | mode=1 或 7 | 新增枚举校验 |
| `quant_flash_attn_def.cpp` | 单 dtype 列表 | 双 dtype 列表（E4M3/HIFLOAT8、E8M0/FLOAT） | 支持两种量化模式 |
| `mm1ResultSize` | `s2BaseSize * ... / 2` | `s2BaseSize * ...`（不除2） | 不切半 |

### 2.3 Kernel Cube 侧与 MXFP8 详细对比

| Cube 项目 | MXFP8 | HIF8 | 差异原因 |
|---|---|---|---|
| **SCALE_T** | `fp8_e8m0_t` | 无 | HIF8 无 block scale |
| **MXFP 常量** | `MXFP_GROUP_SIZE=32`/`MXFP_DIVISOR_SIZE=64`/`MXFP_MULTI_BASE_SIZE=2` | 无 | 无 block scale 计算 |
| **scale 格式** | `Q_SCALE_FORMAT`/`K_SCALE_FORMAT`/`V_SCALE_FORMAT` | 无 | 无 scale GM 格式 |
| **scale GM 变量** | `queryScaleGm_`/`keyScaleGm_`/`valueScaleGm_` | 无 | per-tensor 不需要 |
| **scale copy 对象** | `copyQueryScaleGmToL1_`/`copyKeyScaleGmToL1_`/`copyValueScaleGmToL1_` | 无 | 无 scale 数据拷贝 |
| **scale 类型选择** | `KeyScaleGmToL1Sel`/`ValueScaleGmToL1Sel` 结构体 | 无 | 无 scale copy 策略 |
| **InitQ/K/VScaleBuffer** | 有（初始化 scale GM offset calculator） | 无 | 无 scale 初始化 |
| **CopyQ/K/VScaleSlice/Tile** | 有（拷贝 scale 数据到 L1） | 无 | 无 scale 拷贝 |
| **InitBuffers** | `mm1QSize + mm1QScaleSize` | `mm1QSize`（无 scale size） | 无 scale buffer |
| **InitCubeInput** | 初始化 scale GM（3 个 InitXScaleBuffer 调用） | 不处理 descale | descale 在 vec 侧 |
| **IterateBmm1Nd** | 有（ND 路径） | 无 | HIF8 仅 DN 路径 |
| **IterateBmm1Dn** | `s2BaseSizeCur = s2BaseSize >> 1` | `s2BaseSizeCur = s2BaseSize` | HIF8 不对半切 |
| **MatmulFullMX (Bmm1)** | 传 `mm1AScaleTensor`/`mm1BScaleTensor` | 传 `dummyScale`（空 tensor） | 无 block scale |
| **IterateBmm2Nd** | 有（ND 路径） | 无 | HIF8 仅 DN 路径 |
| **MatmulFullMX (Bmm2)** | 传 `mm2AScaleFakeTensor`/`mm2BScaleTensor` | 传 `dummyScale`/`dummyScale` | 无 block scale |
| **InitValueL1BufferTrans** | 有（死代码，未调用） | 无 | 清理 |
| **GetKVGmFormat** | `GetKVGmFormat` | `GetKVGmFormatHif8`（同名不同函数） | 避免重复定义 |
| **成员变量后缀** | `_` 后缀 | `_` 后缀（对齐） | 命名一致 |
| **PScale L1 布局** | `pScaleL1Offset`/`pScaleDataLen`/`pScaleDstStride` + DataCopy | 无 | HIF8 无 block scale |

### 2.4 Kernel Vec 侧与 MXFP8 详细对比

| Vec 项目 | MXFP8 | HIF8 | 差异原因 |
|---|---|---|---|
| **descaleQK** | 局部变量 `= 1.0` | 局部变量 `= 1.0`（对齐） | Q/K descale 在 dScale 中处理 |
| **descaleQKValue_** | 无 | 有（类成员，`SetDescaleQK` 设置） | kernel 读取 GM 传给 vec |
| **vScaleValue_** | 无 | 有（类成员，`SetVScale` 设置） | per-tensor v_scale 标量 |
| **SetDescaleQK/SetVScale** | 无 | 有 | kernel 读取 GM descale 传给 vec |
| **s2BaseSizeCur** | `s2BaseSize >> 1` | `s2BaseSize` | HIF8 不对半切 |
| **actSingleLoopS2SizeAlign** | `/ 2` | 不除 2 | 对应不切半 |
| **ProcessVec1Nd** | 有 | 无 | HIF8 仅 DN 路径 |
| **VF 函数** | `ProcessVec1VfDnMxfp8` | `ProcessVec1VfDnHif8` | 专用函数 |
| **VF subLoop 参数** | 有（`int32_t subLoop`） | 无 | HIF8 不切半 |
| **VF pScaleSubLoop0** | 有（`fp8_e8m0_t` block scale tensor） | 无 | HIF8 无 block scale |
| **VF maskLine** | 有 | 有（对齐） | mask 标量优化 |
| **VF castTrait** | `castTraitRintZero/One/Two/Three`（else 分支） | `castTraitZero/One/Two/Three`（仅 hif8 路径） | HIF8 用截断模式 |
| **VF 非hif8分支** | 有（bfloat16/half/fp8_e5m2 分支） | 无（全部清除） | HIF8 专用函数 |
| **PScale L1 布局** | 有 | 无 | HIF8 无 block scale |
| **ProcessVec2OnUb** | `1.0f, 1.0f`（scale 参数） | `vScaleValue_, 1.0f` | per-tensor v_scale 乘 |
| **SoftmaxLseCopyOut** | `USE_DN ? ... : ...` | 直接用 `actVecMSize` | HIF8 仅 DN，简化 |
| **pScaleGm_** | 有 | 有（对齐，isFp8 读取标量） | p_scale 标量读取 |
| **isFp8** | `IsSameType<...>` 判断 | `true`（硬编码） | HIF8 恒为 fp8 |
| **POST_QUANT** | `IsSameType` 判断 | `false`（硬编码） | HIF8 输出恒为 BF16 |
| **Bmm2ResBuffSel** | `Bmm2ResBuffSel` | `Bmm2ResBuffSelHif8` | 避免重复定义 |

### 2.5 VF 层 ProcessVec1VfDnHif8 实现细节

新增 `ProcessVec1VfDnHif8` 函数链，参考老接口 `ProcessVec1VfDn` 实现，清除所有非 hif8 分支，加入 maskLine 优化：

```
ProcessVec1VfDnHif8 (入口, ubN=256)
├── isUpdate=false → ProcessVec1DnNoUpdateHif8 (中间层)
│   └── ProcessVec1DnNoUpdateHif8VF (VF函数)
└── isUpdate=true  → ProcessVec1DnUpdateHif8 (中间层)
    └── ProcessVec1DnUpdateHif8VF (VF函数)
```

**VF 函数相比老接口的关键改动：**

| 改动项 | 老接口 ProcessVec1DnNoUpdateVF | HIF8 ProcessVec1DnNoUpdateHif8VF |
|---|---|---|
| **maskUb 参数** | `__ubuf__ uint32_t *maskUb` | 无（删除） |
| **needAtten 参数** | `bool needAtten` | 无（删除） |
| **maskLine 参数** | 无 | `int64_t maskLine`（新增） |
| **mask 实现** | `LoadAlign<uint32_t>(mask_ub)` 加载 mask tensor | `Arange` + `Compare GE` 计算有效行 |
| **sinkValue 参数** | `const float sinkValue = 0.0f` | 无（硬编码 `0.0f`） |
| **hasSink 模板参数** | 有 | 有（对齐） |
| **IsSameType 分支** | `if constexpr (fp8_e5m2\|\|fp8_e4m3\|\|hifloat8) { fp8路径 } else { half/bf16路径 }` | 无（恒为 fp8 路径，else 分支已删除） |
| **castTrait** | `if constexpr (hifloat8) { castTraitZero } else { castTraitRint }` | `castTraitZero`（直接使用，无 if constexpr） |
| **half/bf16 变量** | `RegTensor<half>`/`RegTensor<bfloat16_t>` 声明 | 无（已删除） |
| **DeInterleave** | 有（half/bf16 cast 后交织重排） | 无（hif8 用 Gather 重排） |
| **dScale 计算** | `if constexpr (fp8) { dScale = scale * deScaleQK } else { dScale = scale }` | `dScale = scale * deScaleQK`（直接赋值） |
| **blockStride/repeatStride** | `if constexpr (fp8) { ubN>>2\|1, 2 } else { ubN>>1\|1, 1 }` | `ubN>>2|1, 2`（直接赋值） |
| **loopNum** | `if constexpr (fp8) { ubN/8 } else { ubN/4 }` | `ubN/8`（直接赋值） |

## 3. 主要工作

### 3.1 Host 侧 Tiling 修改

| 文件 | 修改内容 |
|---|---|
| `qfa_tiling_info.h` | 新增 `A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32 = 7` 枚举；`DTYPE_SUPPORT_MAP` Q/K/V 加 `DT_HIFLOAT8`，attn_out 加 `DT_FLOAT16` |
| `qfa_tiling_info.cpp` | 新增 HIF8 枚举的字符串映射 |
| `qfa_tiling_info_parser.cpp` | `GetQuantMode()` 加 mode=7 校验；`ParseAxisInfo()` 加 HIF8 early return；`Parse()` 中 `GetQuantMode()` 提前到 `ParseAxisInfo` 之前 |
| `quant_flash_attn_def.cpp` | 所有输入/输出 `DataType` 和 `FormatList` duplicate 为双条目（支持 E4M3/HIFLOAT8 两种 dtype 组合） |
| `checkers/quant_checker.cpp` | `CheckSingleParaQuantMode` 加 mode=7；Q/K/V descale 加 HIF8 dtype(FLOAT)/shape(1D) early return；`CheckQDescaleShape` 等加 HIF8 `(1,)` early return；`CheckDescaleDtype` 加 HIF8 FLOAT 分支；`CheckLayoutConstraint` 加 HIF8 layout 表；新增 `CheckHeadDimConstraint` (≤128) |
| `checkers/quant_checker.h` | 新增 `CheckHeadDimConstraint` 声明 |
| `checkers/common_checker.cpp` | `CheckSingleParaDtype` 加 HIF8 分支 (Q/K/V → `DT_HIFLOAT8`) |
| `arch35/quant_flash_attn_tiling_mxfp8.cpp` | `IsCapable` 加 HIF8；`SplitPolicy` 加 SINNER_256；`UpdateTilingKeyConfig` 加 Config 2/3 + HIF8 fallback；`UpdateTilingKeyQuantMode` 加 `QFA_HIF8_FP32_PREFILL`；`mm1ResultSize` 去掉 `/ 2` |
| `qfa_adjust_sinner_souter.h` | 新增 `SINNER_256` 和 `QUANT_MODE_HIF8_FP32 = 7` 常量 |

### 3.2 Kernel 侧修改

| 文件 | 修改内容 |
|---|---|
| `quant_flash_attn_common_def.h` | 新增 Config 2/3 (S2Aligned256)；新增 `QFA_HIF8_FP32_PREFILL = 7`；新增 `FA_BYTE_BLOCK` 常量 |
| `quant_flash_attn_template_tiling_key.h` | Config/QuantMode 选择列表加 HIF8 条目 |
| `quant_flash_attn.cpp` | 新增 `quant_flash_attn_hif8` 函数 (hifloat8_t, DN-only)；`#elif (ORIG_DTYPE_Q == DT_HIFLOAT8)` 分支 |
| `quant_flash_attn_block_cube_hif8.h` (新增) | 从 mxfp8 复制裁剪：删除 scale 类型/GM/copy/init 函数；删除 ND 路径；`MatmulFullMX` 用 dummyScale；`s2BaseSizeCur = s2BaseSize`；`GetKVGmFormatHif8` 避免重复定义 |
| `quant_flash_attn_block_vec_hif8.h` (新增) | 从 mxfp8 复制裁剪：删除 ND 路径；保留 `descaleQKValue_`/`vScaleValue_`/`SetDescaleQK`/`SetVScale`；`ProcessVec2OnUb` 用 `vScaleValue_`；`s2BaseSizeCur = s2BaseSize`；调用 `ProcessVec1VfDnHif8` |
| `quant_flash_attn_kernel_hif8.h` (新增) | 从 mxfp8 复制裁剪：新增 per-tensor descale 读取 (GM → float)；`SetDescaleQK`/`SetVScale` 传给 vec block；`ClipSInnerTokenHif8` 避免重复定义 |
| `vf_mul_sel_softmaxflashv2_cast_nz_dn.h` (公共) | 新增 `ProcessVec1DnNoUpdateHif8VF`/`ProcessVec1DnUpdateHif8VF`/`ProcessVec1DnNoUpdateHif8`/`ProcessVec1DnUpdateHif8`/`ProcessVec1VfDnHif8` 共 5 个函数，基于老接口 VF 实现 + maskLine 优化，清除所有非 hif8 分支（IsSameType/bfloat16/half/castTraitRint/DeInterleave），老接口零改动 |

### 3.3 Metadata 修改

| 文件 | 修改内容 |
|---|---|
| `quant_flash_attn_metadata_check.h` | `quantModeSet` 加 `QUANT_MODE_HIF8 = 7` |
| `quant_flash_attn_metadata_aicpu.cpp` | `queryType`/`kvType` 条件加 `quantMode_ == 7`；`s2BaseSize_ = 256U` (HIF8)；v_descale 校验跳过 `quantMode_ == 7` |

### 3.4 Torch Extension 修改

| 文件 | 修改内容 |
|---|---|
| `torch_extension/cann_ops_transformer/ops/csrc/quant_flash_attn.cpp` | 新增 `QUANT_MODE_HIF8 = 7`、`HIFLOAT8 = 34` 常量；`make_wrapper` 加 HIFLOAT8 dtype 分支；新增 `quant_mode == QUANT_MODE_HIF8` 的 wrapper 调用路径 |

### 3.5 测试文件 (新增)

| 目录 | 内容 |
|---|---|
| `tests/pytest/fia_fullquant_hif8_test/` | 完整测试套件 (16 个文件) |
| - `common/generate_hifloat8_data.py` | HIFLOAT8 数据生成 |
| - `common/quant_flash_attn_golden.py` | Golden 计算逻辑 |
| - `common/test_runner.py` | 测试执行框架 |
| - `common/result_compare_method.py` | 结果对比方法 |
| - `quant_flash_attn_paramset_func_rdv.py` | 功能测试参数集 |
| - `test_quant_flash_attn_func_rdv.py` | 功能测试入口 |
| - `conftest.py` / `pytest.ini` | pytest 配置 |
