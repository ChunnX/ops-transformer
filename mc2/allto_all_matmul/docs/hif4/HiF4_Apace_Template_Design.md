# AlltoAllQuantMatmul HiF4 全量化 Apace 模板详细设计文档

| 项目 | 内容 |
| :--- | :--- |
| 特性名称 | alltoallmatmul 新增 HiF4 全量化特性 |
| 代码基线 | 47b981b42（`alltoallmatmul新增hif4全量化特性`，基点 a9cb9856d） |
| 目标芯片 | Ascend 960DT（NpuArch::DAV_9201，A5/35 代架构） |
| 复用架构 | Ascend 950（NpuArch::DAV_3510）arch35 路径 |
| 交付通路 | 仅 Apace Hcomm 通路（纯 C 核通信，CCU/AICPU 引擎） |
| 关联文档 | HiF4_UT_Case_Design.md、HiF4_ST_Case_Design.md |

## 1. 特性概述

### 1.1 背景与目标

`AlltoAllQuantMatmul` 是 MoE EP（Expert Parallelism）场景下的通算融合算子：将 A2All 通信（token 在卡间按 M 轴散射 + K 轴聚集）与本地量化 Matmul 融合为单个 kernel，消除通信与计算之间的访存与同步开销。

本特性在既有 MX 量化（E4M3FN/E5M2/FP4-E2M1 + E8M0 scale）通路之上，新增 **HiF4（HiFloat4）全量化**通路：

- x1 / x2 均为 4-bit `hifloat4`（S1P2 编码，2 元素打包 1 字节）；
- scale 为 4 字节 `hif4_scale` 元数据（E6M2 + E1_8 + E1_16 三级缩放），每 64 个数据元素共享 1 个 scale 单元；
- 计算侧由 Blaze（devkit）的 HiF4 BlockMmad 分发完成反量化累乘，本仓负责通信编排、tiling 与融合框架接入；
- 支持转置（DN）/非转置（ND）两种 B 矩阵布局；
- 仅支持 Hcomm（Apace 纯 C 通信）通路，`all2all_out` 非空或 rankDim>8 场景直接报错。

### 1.2 量化模式定义

`QuantModeType`（matmul_allto_all_util.h）新增枚举：

```cpp
HIFLOAT4_QUANT = 8   // HiF4 全量化：x1QuantMode = x2QuantMode = 8
```

场景标识：`HIF4_SCENE = "HIF4 quantization(x1QuantMode=8, x2QuantMode=8)"`。

## 2. HiF4 数据格式定义

### 2.1 数据元素（S1P2，4-bit）

- 每个元素 4 bit：低 3 bit 为幅值码值，bit3 为符号位；
- 幅值码值表（8 档）：`{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}`；
- 数值 = `(sign==0 ? +1 : -1) × 码值`。

### 2.2 打包格式（hifloat4x2_t）

与 `fp4x2_e2m1_t` 打包规则一致：**偶数下标元素占低 4 bit，奇数下标元素占高 4 bit**，1 字节携带 2 个元素：

```
byte = (code[2i+1] << 4) | code[2i]
```

### 2.3 缩放元数据（hif4_scale，uint32，64 元素/组）

每 64 个连续数据元素（沿 K 轴）共享 1 个 4 字节 scale 单元，little-endian 字节布局：

| 字节 | 域 | 位宽 | 作用 |
| :---: | :---: | :---: | :--- |
| byte0 | E6M2 | 8 | 组级全局 scale，fp8-e6m2 编码（S1+E6+M2），本场景符号位恒 0 |
| byte1 | E1_8 | 8 | 8-bit 共享指数，bit[i/8] 作用于组内第 8i..8i+7 个元素 |
| byte2-3 | E1_16 | 16 | 16-bit 共享指数，bit[i/4] 作用于组内第 4i..4i+3 个元素 |

反量化公式（组内第 i 个元素）：

```
V[i] = E6M2 × 2^(E1_8[i/8] + E1_16[i/4]) × S1P2[i]
E6M2 解码: em = e6m2 & 0x7F;  value = 2^(floor(em/4) - 31) × (1 + (em mod 4)/4)
```

### 2.4 与 MX 量化格式对比

| 维度 | MX（E4M3FN/E5M2/FP4） | HiF4 |
| :--- | :--- | :--- |
| 数据 dtype | DT_FLOAT8_E4M3FN / DT_FLOAT8_E5M2 / DT_FLOAT4_E2M1 | DT_HIFLOAT4 |
| scale dtype | DT_FLOAT8_E8M0（1 字节） | DT_HIFLOAT4_SCALE（4 字节） |
| scale 组大小 | 32 元素/组（相邻 2 组构成 64 元素对齐窗口） | 64 元素/组 |
| scale shape | [M, K/64, 2]（每 64 元素 2 个 E8M0） | [M, K/64, 1]（每 64 元素 1 个 hif4_scale） |
| scale framing C0 | 2（2 个 E8M0 单元为一 framing 单元） | 1（单个 4 字节单元即完整 framing） |
| 每 64 元素 scale 字节 | 2 B | 4 B |
| scale HCCL 通信类型 | HCCL_DATA_TYPE_FP8E8M0 | HCCL_DATA_TYPE_UINT32 |
| 数据 HCCL 通信类型 | FP8 原生 / FP4 按 UINT8 | UINT8（2:1 打包） |
| groupSize attr | (1, 1, 32)，打包值 4295032864 | (1, 1, 64)，打包值 **4295032896** |
| B 矩阵布局 | 仅转置（DN，x2=[N,K]） | 转置（DN）与非转置（ND，x2=[K,N]）均可 |
| Ka 对齐约束 | CeilDiv(Ka,64) 组数为偶数（等价 Ka%128==0 由 tiling 保证） | Ka % 128 == 0（显式校验） |
| kernel 通路 | 主线 pipeline 与 Apace Hcomm | **仅 Apace Hcomm** |
| 输出精度容差（ST 验收） | 1e-2 | 1e-1（4-bit 三级缩放，动态范围更大） |

> groupSize 打包值说明：`groupSize = (groupSizeM << 32) | (groupSizeN << 16) | groupSizeK`，HiF4 为 `(1<<32)|(1<<16)|64 = 4295032896`。

## 3. 接口规格

### 3.1 aclnn 接口参数（HiF4 场景）

| 参数 | 方向 | 规格 |
| :--- | :---: | :--- |
| x1 | 输入 | `[M_total, Ka]`，DT_HIFLOAT4，ND 格式，连续；`Ka = K_total / rankDim`，**Ka % 128 == 0** |
| x2 | 输入 | 转置：`[N, K_total]`（DN）；非转置：`[K_total, N]`（ND）；DT_HIFLOAT4；允许非连续（自动转连续并置 transposeX2=true） |
| bias | 输入(可选) | `[N]`，DT_FLOAT |
| x1_scale | 输入 | `[M_total, CeilDiv(Ka,64), 1]`，DT_HIFLOAT4_SCALE，**必填非空** |
| x2_scale | 输入 | 转置：`[N, CeilDiv(K,64), 1]`；非转置：`[CeilDiv(K,64), N, 1]`；DT_HIFLOAT4_SCALE |
| y | 输出 | `[m, N]`（m = M_total/rankDim），DT_FLOAT16 / DT_BFLOAT16 / DT_FLOAT |
| all2all_out | 输出(可选) | **HiF4 场景必须为空**（非空报错），dtype 需与 x1 一致（DT_HIFLOAT4） |
| 关键 attrs | - | `x1QuantMode=8`、`x2QuantMode=8`、`groupSize=4295032896`、`transposeX1=false`、`transposeX2∈{true,false}`、`comm_mode∈{ccu,aicpu}` |

### 3.2 约束汇总

| 约束 | 校验层 | 错误类型 |
| :--- | :--- | :--- |
| x1_scale 必填非空 | op_api（CheckNotNull） | ACLNN_ERR_PARAM_NULLPTR |
| Ka % 128 == 0 | op_api（CheckScaleShape）+ tiling（CheckGroupSize） | PARAM_INVALID / GRAPH_FAILED |
| x1/x2 dtype 均为 DT_HIFLOAT4 且一致 | op_api + tiling | PARAM_INVALID / GRAPH_FAILED |
| scale dtype = DT_HIFLOAT4_SCALE | op_api + tiling | PARAM_INVALID / GRAPH_FAILED |
| scale shape 末维 = 1，dim1 = CeilDiv(K,64) | op_api + tiling | PARAM_INVALID / GRAPH_FAILED |
| y dtype ∈ {FP16, BF16, FP32} | op_api + tiling | PARAM_INVALID / GRAPH_FAILED |
| bias dtype = FP32 | op_api + tiling | PARAM_INVALID / GRAPH_FAILED |
| all2all_out 必须为空 | tiling（InitTilingContextParameters） | GRAPH_FAILED |
| rankDim ≤ 8 | tiling | GRAPH_FAILED |
| K_total % rankDim == 0（Ka 整除） | 通用 tiling | GRAPH_FAILED |
| groupSize = (1,1,64) | tiling（CheckGroupSize） | GRAPH_FAILED |
| MX 模式下 x1/x2 不允许 hifloat4 | tiling（CheckMxQuantTensorDataType） | GRAPH_FAILED |

### 3.3 计算语义

设 rankDim = R，rank d 的输出：

```
y[d] = A_perm[d] × B
A_perm[d][i, s*Ka:(s+1)*Ka] = A_s[d*m + i, :]        (s = 0..R-1)
B[k, n] 由 x2（DN/ND 布局）与 x2_scale 反量化得到
A 反量化: A[r, j] = dequant_hif4(x1[r, j], x1_scale[r, j/64])
```

即：每卡先将自己持有的 A 分片（M_total × Ka）按 M 轴 m 行一组散射到各卡；各卡收到所有卡数据后沿 K 轴拼接成 [m, K]，与本地完整 B 相乘。

## 4. 总体设计

### 4.1 软件分层与调用链

```
aclnnAlltoAllQuantMatmulBaseGetWorkspaceSize (op_api)
  ├─ IsDavidArch() = (DAV_3510 || DAV_9201)                     ← 新增 9201 判定
  ├─ CheckNotNull / CheckScaleShape(HiF4 分支) / CheckHiF4QuantDtypesValidA5   ← 新增校验
  └─ Inner → GE/FE
        ↓
AlltoAllMatmulTilingFunc (allto_all_matmul_tiling.cpp)
  └─ DAV_9201 → TilingRegistryArch（与 DAV_3510 同注册表）        ← 新增路由
        ↓
AllToAllMxQuantMatmulTilingBase (arch35, 注册 {DAV_3510, DAV_9201})
  ├─ IsCapable: x1QuantMode==8 && x2QuantMode==8 → isHiF4_
  ├─ CheckOpInputInfo（校验链，见 §6.2）
  ├─ usingApaceImpl_ = (!all2allOutFlag) && rankDim<=8；HiF4 强制为真
  ├─ TileCommAndCompute → GetCutResOfCommAndCompute（M 轴切分，256 对齐）
  ├─ DoMxQuantMMTiling → DoHcommMxQuantMMTiling                  ← HiF4 分支
  │     └─ QuantMatmulTilingSwat<DT_HIFLOAT4, DT_HIFLOAT4, DT_FLOAT>
  ├─ SetHcclTiling（4-bit → UINT8 通信 dtype，CCU/AICPU 算法选择）
  ├─ SetUserWorkSpace（commLen 2:1 打包 / commScaleLen 4B 每组）
  └─ GetTilingKey（HIF4_PREGROUP_QUANT_MODE）
        ↓
allto_all_matmul_apt.cpp (op_kernel, ORIG_DTYPE_X1==DT_HIFLOAT4 分支)   ← 新增分支
  └─ USING_APACE_IMPL==1:
       AllToAllMxQuantMatmulHcommImpl<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_X1,
                                      hif4_scale, TilingData, ServerType,
                                      /*IsMxFp4=*/false, /*TransB=*/X2TRANSPOSE>
         ├─ Init: HCCL 批量 AlltoAll 下发（scale + dataHead + dataTail）
         └─ Run:  MatmulProcess(REMOTE) → AllToAllQbmmMxKernel（Blaze BlockMmad HiF4 分发）
```

### 4.2 平台接入（DAV_9201）

| 接入点 | 变更 |
| :--- | :--- |
| 算子定义 | `allto_all_matmul_def.cpp` 新增 `aicoreConfig_960dt`：34 组 dtype 组合（前 31 组复用 950 组合，追加 3 组 HiF4：x1/x2=HIFLOAT4，scale=HIFLOAT4_SCALE，y∈{FP16,BF16,FP32}），`AddConfig("ascend960dt", ...)`；opFile=allto_all_matmul_apt，jitCompile=static_false（动态 shape 复用二进制） |
| tiling 路由 | `allto_all_matmul_tiling.cpp`：DAV_9201 走 `TilingRegistryArch`（与 950 相同 arch tiling 注册表） |
| tiling 注册 | `REGISTER_TILING_TEMPLATE_WITH_ARCH(AlltoAllMatmul, ..., {DAV_3510, DAV_9201}, 2)` |
| 编译配置 | `op_host/CMakeLists.txt`：ascend960dt 分支（dcci-before-kernel-end=false + ENABLE_CV_COMM_VIA_SSBUF） |
| 二进制配置 | `config/ascend960dt/allto_all_matmul_binary.json`（op_list 空，动态 shape 不预编译）+ `allto_all_matmul_simplified_key.ini`（default=0） |
| op_api 判定 | `IsDavidArch()` 统一 950/960dt 校验与处理路径（A5 代共路径） |

### 4.3 TilingKey 设计

`allto_all_matmul_tiling_key.h` 新增模板枚举与选择分支：

```cpp
#define HIF4_PREGROUP_QUANT_MODE 4
// QUANTMODE 列表追加 HIF4_PREGROUP_QUANT_MODE
```

HiF4 tilingKey 字段组合（`GET_TPL_TILING_KEY` 编码）：

| 字段 | 取值 | 说明 |
| :--- | :--- | :--- |
| KERNEL_TYPE | AIC_ONLY | 纯 C 核（Hcomm 无 AIV 参与） |
| QUANTMODE | 4（HIF4_PREGROUP_QUANT_MODE） | 区别于 MX(3) |
| X2TRANSPOSE | 0 / 1 | ND / DN 两种 B 布局 |
| DTYPEBIAS | 1（DTYPE_BIAS_FP32） | HiF4 恒 FP32（含无 bias） |
| ISSMALLK | 0 | 恒 0 |
| COMMTYPE | 0(CCU) / 1(AICPU) | 由 comm_mode attr 决定 |
| USING_APACE_IMPL | 1 | 恒 1 |
| ADDBIASSPLITMODE | 0（ADDBIAS_OFF） | 恒 0 |

即 HiF4 共 4 个二进制变体（2 布局 × 2 通信引擎）。MX 场景保持原 key（QUANTMODE=3，X2TRANSPOSE 恒 1），互不影响。

## 5. op_api 层详细设计

文件：`mc2/allto_all_matmul/op_api/allto_all_quant_matmul_base.cpp`。

### 5.1 平台判定

```cpp
static bool IsDavidArch() {
    NpuArch npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return npuArch == NpuArch::DAV_3510 || npuArch == NpuArch::DAV_9201;
}
```

原 `GetCurNpuArch() == DAV_3510` 的 5 处判定（shape 校验入口、dtype 校验入口、commQuantDtype 默认值、x2 非连续处理、HcclServerType 设置）统一替换为 `IsDavidArch()`。

### 5.2 非空校验（CheckNotNull）

- MX + David 架构：x1ScaleOptional 必填（既有）；
- **新增**：HiF4（x1QuantMode==8）+ DAV_9201：x1ScaleOptional 必填非空，否则 `ACLNN_ERR_PARAM_NULLPTR`。

### 5.3 scale shape 校验（CheckScaleShape → Check3DScaleShape 泛化）

`Check3DScaleShape` 增加 `expectedScaleLastDim` 参数（MX=2，HiF4=1）与 `x1` 参数，并新增 K 分组一致性校验：

```
x1Scale.shape  == [M_total, CeilDiv(Ka,64), expectedScaleLastDim]
x2Scale.shape  == transposeX2 ? [N, CeilDiv(K,64), expectedScaleLastDim]
                              : [CeilDiv(K,64), N, expectedScaleLastDim]
```

HiF4 分支前置校验（进入 Check3DScaleShape 之前）：

```
Ka % 128 == 0        // 统一对齐规则：保证 Ka%64==0 且 scale 组数为偶数
```

> 设计说明：`Ka%128` 同时蕴含 `Ka%64==0` 与 `CeilDiv(Ka,64)%2==0`（组数为偶数），与 MX 路径 tiling 侧的组数偶数约束统一；HiF4 将该约束显式前置到 op_api 层，尽早报错。

### 5.4 dtype 校验（CheckHiF4QuantDtypesValidA5，新增）

```
x1, x2          ∈ {DT_HIFLOAT4}，且 x1.dtype == x2.dtype
x1Scale, x2Scale ∈ {DT_HIFLOAT4_SCALE}
y               ∈ {DT_FLOAT16, DT_BFLOAT16, DT_FLOAT}
bias(可选)       ∈ {DT_FLOAT}
all2allOut(可选)  dtype == x1.dtype
```

所有错误日志带 `HIF4_SCENE` 场景前缀，与 MX/KC 场景日志区分。

### 5.5 x2 非连续处理（IsDavidArch 分支）

沿用 David 架构既有逻辑，HiF4 自动继承：

- x2 为转置视图（`IsTransposeLastTwoDims`）且用户 `transposeX2=false`：自动转连续，`transposeX2` 置 true（**ND 输入经 .t() 视图进入时会被归一化为 DN**）；
- x2 非连续且用户 `transposeX2=true`：报错（双重转置误用）。

## 6. op_host tiling 层详细设计

文件：`mc2/allto_all_matmul/op_host/op_tiling/arch35/allto_all_mx_quant_matmul_tiling_base.{h,cpp}`。

### 6.1 准入判定（IsCapable）

```cpp
isHiF4_ = (x1QuantMode == 8 && x2QuantMode == 8);
if ((x1QuantMode == 6 && x2QuantMode == 6) || isHiF4_) return true;   // MX 或 HiF4
```

常量定义（头文件新增）：

```cpp
constexpr size_t HIF4_QUANTMODE_VALUES  = 8;     // HIFLOAT4_QUANT
constexpr uint64_t HIF4_SCALE_BLOCK_K   = 64;    // groupSizeK 期望值
constexpr uint64_t HIF4_SCALE_LAST_DIM  = 1;     // scale shape 末维
constexpr uint64_t HIF4_SCALE_UNIT_BYTES = 4;    // sizeof(hif4_scale)
constexpr uint64_t HIF4_KA_ALIGN_SIZE   = 128;   // Ka 对齐约束
```

### 6.2 校验链（CheckOpInputInfo）

| 顺序 | 校验函数 | HiF4 行为 |
| :---: | :--- | :--- |
| 1 | CheckAttrsInfo（公共） | group 非空、worldSize 校验、transposeX1 拒绝等（继承） |
| 2 | CheckMxTensorFormat | scale format 必须 ND（继承） |
| 3 | CheckX2Transpose | **HiF4 放行 false**（MX 强制 true）；isTransX2 attr 缺失报错 |
| 4 | CheckMatrixMulShapes（公共） | m/k/n 乘法形状一致性（ND/DN 均支持） |
| 5 | CheckMxQuantTensorDataType | x1/x2 允许列表追加 DT_HIFLOAT4；**isHiF4_ 时 x1/x2 强制 DT_HIFLOAT4、scale 强制 DT_HIFLOAT4_SCALE**；MX 模式下出现 hifloat4 仍按列表通过但被 HiF4 强制校验隔离（见 §10 已知问题） |
| 6 | CheckMxQuantShapeInfo | x1Dim1%64、x2Dim1%64；x1Scale=[M, Ka/64, 末维]、x2Scale=[x2Dim0, x2Dim1/64, 末维]；末维 isHiF4_?1:2 |
| 7 | CheckAlltoAllOut | all2allOut 与输出一致性（继承） |
| 8 | CheckGroupSize | **HiF4：Ka%128 前置校验** + groupSize==(1,1,64)；shapeInfo.isHif4 传入 InferGroupSize |

`InferGroupSize`（mc2_tiling_utils）泛化：`Mc2MatmulShapeInfo` 新增 `isHif4` 字段（尾部追加，保持既有聚合初始化兼容）；mxfp 分支下 scale 末维 `isHif4 ? 1 : 2`：

```
scaleKValue = scaleShape.dim(-2) × (isHif4 ? 1 : 2)    // [M, K/64, 末维] → 等效 K 元素数
groupSizeK = kValue / scaleKValue × 64 → HiF4 恒 64
```

### 6.3 通路判定（InitTilingContextParameters）

```cpp
usingApaceImpl_ = (!allToAllOutFlag) && (rankDim <= 8);
if (isHiF4_ && !usingApaceImpl_) → 报错 "HIFLOAT4 quant mode only supports the hcomm path..."
```

HiF4 无主线 pipeline 实现，此校验保证非法组合在 tiling 阶段拦截（对应 kernel 侧 static_assert 双保险）。

### 6.4 通信切分与 SWAT tiling（DoHcommMxQuantMMTiling）

通信切分（GetCutResOfCommAndCompute → AlignCutResForKernel）沿用 apace 路径：longTile 按 256 向上对齐，shortTile 置尾。

量化 matmul tiling 按 dtype 分支实例化：

```cpp
if (isHiF4_) {
    QuantMatmulTilingSwat<mm::DataType::DT_HIFLOAT4, mm::DataType::DT_HIFLOAT4,
                          mm::BiasDataType::DT_FLOAT> tilingSwat;
    // SetPlatformInfoPtr / SetBiasInfo(hasBias)
    tilingSwat.GetTilingData(perRankM, nValue, perRankK, /*transA=*/false, transB,
                             hcommTilingData_.tileQbmmTilingData);
}
```

其中 `perRankM = mValue`（每卡输出 M），`perRankK = orgKValue = Ka`，`nValue = N`。

SWAT tiling 的 HiF4 特化（详见 §7.6）由 `aDataType/bDataType = DT_HIFLOAT4` 模板参数驱动：

| 决策项 | HiF4 取值 | 来源 |
| :--- | :--- | :--- |
| baseM | min(m,256) 对齐 16 | 通用 |
| baseN | transB: min(n,256) 对齐 16；ND: min(n,256) 对齐 64（GetShapeWithDataType\<HIF4\>(L1_ALIGN_SIZE=32)×2） | dtype 特化 |
| baseK | min(Ka, **256**) 对齐 64（4-bit 与 FP4 同档，8-bit 为 128） | dtype 特化 |
| scale 块大小 | `Align(CeilDiv(baseK,64), 1) × 4B × baseM/baseN`（MX 为 `Align(·,2)×1B`） | GetPerGroupSize/GetScaleUnitPerGroup/GetAlignSize |
| biasL1 | hasBias ? 2×Align(baseN,16)×4B : 0 | 通用 |
| depthA1/B1, stepK | L1 预算搜索，stepK ≤ 4 | 通用 |
| scaleKL1 | min(scaleFactorA×stepKa×baseK, scaleFactorB×stepKb×baseK) | 通用 |
| dbL0c | baseM×baseN×4×2 ≤ l0cSize ? 2 : 1 | 通用 |
| usedCoreNum | 全块>1 或无尾块 → aicNum；否则 尾块数×mTailTile×nTailTile | 通用 |

tiling 数据结构（`Apace::hcommAllToAllMatmulTilingData`）：

```cpp
struct hcommAllToAllMatmulTilingData {
    Mc2InitTiling mc2InitTiling;          // HCCL init 参数（SetHcclTiling 生成）
    Mc2CcTiling mc2CcTiling;              // HCCL CC 侧参数（SetCcTilingV2 偏移）
    CommTilingData commTilingData;        // splitAxisTileSize/Cnt, splitAxisTailSize/Cnt, nonSplitAxisSize
    QuantMatmulTilingData tileQbmmTilingData;  // SWAT 切分结果（baseM/N/K、尾块、stepK、scaleKL1…）
    uint32_t localMatmul{0};              // apace 路径恒 0（REMOTE 融合模式）
};
```

### 6.5 HCCL tiling 与 workspace

**SetHcclTiling**：

- 通信数据 dtype：`isMxFp4_ || isHiF4_` → `ge::DT_UINT8`（4-bit 2:1 打包载体），其余用 x1 dtype；
- reduce type / comm engine / 算法选择沿用主线（`Mc2CommAlgoSelector`，按 commDataBytes 与 rankDim 选算法）；
- `mc2InitTiling / mc2CcTiling` 写入 hcommTilingData_。

**SetUserWorkSpace**：

```
commLen      = AlignUp(CeilDiv(m × K_total × 1B, 2), 512)        // 4-bit 打包，2 元素/字节
commScaleLen = AlignUp(M_total × CeilDiv(Ka,64) × (isHiF4 ? 4 : 2)B, 512)
permuteLen = commLen（all2allOut 为空时）；permuteScaleLen = commScaleLen
workspace = libApiWS + commLen + permuteLen + commScaleLen + permuteScaleLen
```

> 注：workspace 总量按主线口径（含 permute 重排段）申请；apace 路径实际只使用前段 `[scale 收包区 | 数据收包区]`（§7.2），permute 段预留。

## 7. Apace kernel 层详细设计（核心）

### 7.1 模板签名与泛化参数

`AllToAllMxQuantMatmulHcommImpl` 模板参数扩展（+2）：

```cpp
template <typename X1Type, typename X2Type, typename YType,
          typename CommDataTypeX1,
          typename ScaleType,                                    // ← 新增：fp8_e8m0_t | hif4_scale
          typename AlltoAllMatmulTilingDataType,
          AscendC::HcclServerType ServerType,
          bool IsMxFp4,
          bool TransB>                                           // ← 新增：B 布局开关
```

- **ScaleType**：由算子入口按量化模式显式指定（HiF4 → `hif4_scale`，devkit/blaze 提供；MX → `AscendC::fp8_e8m0_t`，MX 分支同步改造为显式传入）；
- **TransB**：`LayoutB = conditional_t<TransB, DNExtLayoutPtn, NDExtLayoutPtn>`；MX 场景恒 true（行为不变），HiF4 两种布局均可。

派生常量（编译期）：

```cpp
mxScaleHcclDataType_ = IsHif4<X1Type>() ? HCCL_DATA_TYPE_UINT32 : HCCL_DATA_TYPE_FP8E8M0
isHiF4Scale_         = IsHif4<X1Type>()                          // Blaze::Gemm::Kernel::IsHif4
x1HcclDataType_      = GetX1HcclDataType<hifloat4x2_t>() = HCCL_DATA_TYPE_UINT8
```

> `IsHif4<X1Type>()` 由 devkit/blaze 侧提供（hifloat4x2_t 类型分发）。

### 7.2 通信设计（Init）

**workspace 布局**（apace 视角，从 workspaceGM 起始）：

```
┌───────────────────────────────┬──────────────────────────────────────┐
│ scale 收包区                    │ 数据收包区 commOutGM_                 │
│ x1ScaleLen = AlignUp(          │ commDataLen = M_total × Ka / 2 (B)   │
│   M_total × Ka/64 × 4B, 512)  │（all2all_out 非空时直接用 all2all_out, │
│                               │  HiF4 场景恒为空 → workspace 内分配）   │
└───────────────────────────────┴──────────────────────────────────────┘
```

**通信任务下发**（3 类 handle，批量一次下发、kernel 内逐 tile 等待）：

维度约定：`splitAxis`（splitAxisTileSize×Cnt + TailSize×Cnt）覆盖 **per-rank m**（每目标卡分片），非 M_total——依据：host `AlignCutResForKernel` 以 mValue=m 封顶切分；AllToAll 目的块间距 `splitAxis×Ka/2 B` 必须等于目的分片间距 `m×Ka/2 B`；URMA 实现同以 `axisM=m` 传入 LocalParams。

HCCL `AlltoAll<true>(sendBuf, recvBuf, dataCount, dataType, strideCount, repeat)` 语义：sendBuf 中第 j 块（偏移 j×strideCount、大小 dataCount）发往 rank j；recvBuf 中来自 rank i 的块落于偏移 i×strideCount。send 侧块 j = x1 行 `[j×m + t×tileSize, +tileSize)`（目的 j 的 m 分片按 tile 逐段发送）；recv 侧布局 `[R, m, Ka]`（源 i 贡献在行 `i×m` 起），与 kernel `layoutA=(R×m, Ka)`、`actualMPos=rank×m+mPos` 精确对应。

| 任务 | 触发条件 | dataCount（块大小） | strideCount（目的块间距） | repeat |
| :--- | :--- | :--- | :--- | :--- |
| scale AlltoAll | 恒下发 | `m×Ka/64` 个 hif4_scale（代码传 `scaleSendCount×sizeof(ScaleType)` 字节，单位约定存疑见 R5） | `m×Ka/64`（scale 元素） | 1 |
| dataHead AlltoAll | `splitAxisTileCnt > 0` | `tileSize×Ka/2` B（UINT8，2:1 打包，`rankForComm = CeilDiv(Ka,2)`） | `m×Ka/2` B | tileCnt |
| dataTail AlltoAll | `splitAxisTailCnt > 0` | `tailSize×Ka/2` B（src 偏移 `tileCnt×tileSize×Ka/2` B） | `m×Ka/2` B | tailCnt |

关键泛化点（相对 MX）：

1. **scale 单元数**：`MxScaleKLen<hif4_scale>(k) = CeilDiv(k, 64)`（MX 为 `CeilDiv(k,64) × 2`）——scale shape [M, K/64, 1] 与 [M, K/64, 2] 的统一抽象；
2. **scale 通信宽度**：HiF4 按 UINT32 传输完整 4 字节元数据（HCCL 无专用类型）；
3. **数据 2:1 打包**：`isHiF4Scale_ || IsMxFp4` 均按 2 元素/字节折算通信量（HiF4 不再依赖 IsMxFp4 模板参数，由数据类型判定）。

**HCCL 上下文**：`hccl_.InitV2(GetHcclContext<0>(), &mc2InitTiling)` + `SetCcTilingV2(offsetof(..., mc2CcTiling))`，AIC-only（kernel 类型 KERNEL_TYPE_MIX_AIC_1_1 / AIC_ONLY tilingKey 约束一致）。

### 7.3 计算设计（Run → MatmulProcess(REMOTE)）

```
Run():
  if (localMatmul != 0) MatmulProcess(LOCAL);   // apace 主线恒 0，保留能力
  MatmulProcess(REMOTE);                        // 融合 REMOTE：通信+计算单遍
  SyncAll(); hccl_.Finalize();
```

`MatmulProcess` → `SetupParams` 组装 `AllToAllQbmmMxKernel::Params`：

| 参数 | 取值 |
| :--- | :--- |
| problemShape | (m, N, K_total, 1) |
| mmadParams | A=commOutGM_（通信收包区），scaleA=scale 收包区，B=x2（本卡 GM），scaleB=x2_scale，C=y，bias |
| splitKNum | localMatmul==0 → rankDim（REMOTE 遍历全部 R 份 K 分片） |
| l1Params | stepK×baseK, scaleKL1, nBufferNum=2 |
| schedulerParams | baseM/N + 尾块四元组（mTailTile/nTailTile/mBaseTailSplitCnt/nBaseTailSplitCnt/mTailMain/nTailMain） |
| qbmmParams | baseM/N/K, dbL0c, isBias |

### 7.4 Qbmm kernel 处理流程（ProcessSingleBatch，REMOTE 分支）

```
for each 调度块 (mPos, nPos):
    singleShape = 调度器块形状（含 M/N 尾块与尾块再分裂）
    if (M块或N块 ≤ 0) return;                        // 空块保护
    dependTileIdx = min((mPos+blockM-1)/headTileSize, totalTiles-1)
    while (readyTileIdx < dependTileIdx):            // 通信等待去重
        WaitTile(++readyTileIdx)
    for rank in [0, R):
        actualMPos = rank × m + mPos
        A 块   = (rank==self) ? gmALocal[actualMPos]    // 本卡数据直接读原 GM
                              : gmA[actualMPos]          // 远端数据读通信收包区
        scaleA = 同源切换（localScaleA / commScaleA）
        B 块   = gmB[rank×K, nPos]；scaleB = gmScaleB[rank×scaleKLen, nPos]
        mmadOp_(A块, B块, scaleA块, scaleB块, bias块, C块, singleShape, remoteRankCnt++)   // L0C 累加
// 尾核兜底：核心调度块不足时等待剩余通信 tile，防止 Finalize 提前
while (readyTileIdx < totalTiles-1): WaitTile(++readyTileIdx)
```

`WaitTile(tileIdx)`（HcommCommWaitPolicy）：

```
tileIdx == 0            → hccl.Wait(scaleHandle)     // scale 仅等一次，掩盖 matmul 头开销
tileIdx < headTileCnt   → hccl.Wait(dataHeadHandle)
else                    → hccl.Wait(dataTailHandle)
```

保留分支（本特性不激活，模板能力保留）：LOCAL 先行（localMatmul==2，GM 累加+REMOTE 原子累加）、DEFERRED_SYNC（L0C 驻留 + wait_flag）、localMatmul==2 时 REMOTE 跳过 self rank。apace 主线 `localMatmul=0`：self rank 数据也经 AlltoAll 收包后从 commOutGM_ 读取（保证数据通路一致性）。

> 说明：REMOTE 分支中 `rank==rankId && localMatmul!=2` 时，本卡的 A 数据与 scaleA **均**切换到输入 GM（gmALocal/gmScaleALocal）直接读取，减少一次数据搬运；远端 rank 数据从收包区读取。

### 7.5 scale 泛化与 L2 cache 策略（all_to_all_qbmm_mx_kernel.h）

**泛化函数族**（Blaze::Gemm::Kernel 命名空间，MX/HiF4 统一）：

```cpp
IsHiF4Scale<ScaleType>()            // hif4_scale 判定
MxScaleKLen<ScaleType>(k)           // K 方向 scale 单元数：HiF4=⌈k/64⌉，MX=⌈k/64⌉×2
MxScaleBytesPerDivisor<ScaleType>() // 每 64 元素 scale 字节：HiF4=4，MX=2
MxScaleC0<ScaleType>()              // scale framing C0：HiF4=1，MX=2
```

- scale 元素类型从 Blaze BlockMmad 导出（`ScaleAType/ScaleBType`），GM 指针类型随之泛化（原硬编码 `fp8_e8m0_t`）；
- scale 布局 framing：`FrameLayoutFormat<ScaleA/B{DN,ND}LayoutPtn, Int<SCALE_C0_A/B>>`，HiF4 C0=1；
- kCacheLineAlignMask：`IsFp4<AType>() ? 0xff : 0x7f`——HiF4（hifloat4x2_t 非 fp4x2_e2m1_t）取 **0x7f（128B）**。

**SetScaleL2Cache / SetL2Cache**（仅 fullMTile 且 B 维=1 时生效）：

| 布局 | 对齐判据（128B） | 分支 |
| :--- | :--- | :--- |
| transB（DN） | `(K/64)×4B % 128 == 0` 且 `(scaleKL1 组)×4B % 128 == 0`，即 K % 2048 == 0 | 对齐 → CACHE_MODE_DISABLE（流式，不污染 cache）；否则 NORMAL |
| 非转置（ND） | `N×4B % 128 == 0` 且 `baseN×4B % 128 == 0`，即 N % 32 == 0 | 同上 |
| B 矩阵 | transB：K%128==0；ND：N%128==0 且 baseN%128==0 | 同上 |

### 7.6 tiling 泛化（apace_common_utils.h / apace_constant.h）

```cpp
enum class DataType { ..., DT_HIFLOAT4 };          // mm::DataType HiF4 枚举（kernel 侧）
GetShapeWithDataType<DT_HIFLOAT4>(s) = s × 2       // 4-bit：shape(元素数) = 字节×2
GetSizeWithDataType<DT_HIFLOAT4>(b)  = ⌈b/2⌉
GetPerGroupSize<DT_HIFLOAT4>()       = 64          // HIF4_GROUP_SIZE
GetScaleUnitPerGroup<DT_HIFLOAT4>()  = 4           // HIF4_SCALE_UNIT（字节/组）
GetAlignSize<DT_HIFLOAT4>()          = 1           // TILING_HIF4_MULTI_BASE_SIZE（MX=2）
```

SWAT L1 预算中 scale 块大小按上述泛化：

```
baseScaleASize = Align(⌈baseK/64⌉, 1) × 4B × baseM     // HiF4
               （MX: Align(⌈baseK/32⌉, 2) × 1B × baseM）
```

### 7.7 算子入口（allto_all_matmul_apt.cpp）

```cpp
#elif ((ORIG_DTYPE_X1 == DT_HIFLOAT4) && (ORIG_DTYPE_X2 == DT_HIFLOAT4))
    if constexpr (USING_APACE_IMPL) {
        // ScaleType=hif4_scale；IsMxFp4=false；TransB=X2TRANSPOSE
        Apace::AllToAllMxQuantMatmulHcommImpl<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_X1, hif4_scale,
                                              Apace::hcommAllToAllMatmulTilingData, hcclServerType,
                                              false, X2TRANSPOSE> op(&tilingData);
        op.Init(x1, x2, bias, y, all2all_out, x1_scale, x2_scale, workspaceGM);
        op.Run();
    } else {
        static_assert(sizeof(DTYPE_X1) == 0, "HiF4 only supports the APACE Hcomm path...");
    }
```

MX FP8/FP4 分支同步改造：显式传 `AscendC::fp8_e8m0_t` 作为 ScaleType（+尾部 true 模板参数），行为不变。

## 8. 精度设计

- **golden 公式**（ST CPU 参考实现，gen_data.py hif4 模式）：

```
A_deq[r,j] = E6M2(r, j/64) × 2^(E1_8(j mod 64 /8) + E1_16(j mod 64 /4)) × S1P2(j)
y = (AllToAll 重排后的 A_deq) × B_deq → BF16
```

- **数据分布**：E6M2 指数限定 [30,32]（数值 0.5~8，动态范围适中）；E1_8 每 bit 伯努利(1/2)；E1_16 每 bit 置位概率 1/4；S1P2 码值/符号均匀随机；`np.random.seed(42)` 固定。
- **验收容差**：`rtol = atol = 1e-1`（4-bit 输入 + 三级缩放的固有量化误差远大于 MX；对齐低精度算子通用放宽门槛）。
- 输出 dtype：ST 直调工程固定 BF16；aclnn 主线 FP16/BF16/FP32（FP32 输出时容差同 1e-1）。

## 9. 性能设计要点

1. **纯 C 通信**：HCCL 任务在 AIC 核内下发/等待，无 AIV↔AIC 跨核 flag 同步（AIC_ONLY tilingKey）；
2. **批量通信**：scale（整段一次）+ data（head/tail 分段批量）在 Init 阶段一次下发，计算侧按 tile 依赖去重等待，通信与计算流水重叠；
3. **scale 单次全量通信**：HiF4 scale 通信量为 M_total×Ka/64×4B（数据量的 1/32），单 handle 全量下发；
4. **L2 cache 策略**：仅整 M 块（fullMTile）场景按 128B 对齐判据关闭 scale/B 的 L2 缓存（流式访问，避免污染）；
5. **2:1 打包通信**：数据通信量减半（相对 8-bit）；
6. **本卡短路**：REMOTE 模式 self rank 数据直接读输入 GM（不要求经收包区中转）。

## 10. 已知问题与风险

| 编号 | 问题 | 影响 | 建议 |
| :--- | :--- | :--- | :--- |
| R1 | **host tiling `CheckMxQuantShapeInfo` 对 ND 布局 x2Scale 的校验与 op_api/kernel 约定不一致**：该函数假定 x2=[N,K]（校验 `x2ScaleDim0==x2Dim0`、`x2ScaleDim1==x2Dim1/64`、`x2Dim1%64`），而 HiF4 非转置布局为 x2=[K,N]、x2Scale=[K/64,N,1]（op_api Check3DScaleShape 与 kernel ScaleBNDLayoutPtn 均按此约定），合法 ND 输入会在 tiling 阶段被误拒（报 "shape[0] of x2Scale must be K"） | **transB=false 的 aclnn 主线通路不可用**（ST 直调通路正常，因其绕过 host tiling 自行构造 tilingData） | 修复 CheckMxQuantShapeInfo 按 isBTrans 区分布局校验；修复前 ND 用例仅在 ST 直调框架执行，aclnn ND 正向用例暂标记预期 FAILED（见 UT 设计 UT-TH-02） |
| R2 | `IsHif4<X1Type>()`、`hif4_scale` 类型与 Blaze BlockMmad HiF4 分发依赖 devkit 就位，本仓代码在 devkit 未更新时无法编译 HiF4 目标（ST CMake 注释已声明） | 编译/CI 依赖 | devkit 版本对齐后执行编译与 ST 验证 |
| R3 | MX 模式 dtype 允许列表加入 DT_HIFLOAT4 后，"MX 模式 + hifloat4 输入" 在 tiling 侧由 `isHiF4_` 互斥校验拦截（x1QuantMode=6 时 isHiF4_=false，但 dtype 校验仍过列表）→ 实际会被 CheckGroupSize 的 groupSize 校验或后续流程拦截，但错误信息不直接 | 错误信息可读性 | 补充显式互斥校验："hifloat4 dtype requires HIFLOAT4 quant mode" |
| R4 | HiF4 仅 4 个二进制变体（2 布局×2 引擎），动态 shape（jitCompile=static_false）依赖 FE 按需编译，首图编译耗时 | 首次执行延迟 | 预热或后续补预编译二进制 |
| **R5** | **scale AllToAll dataCount 单位歧义**（hcomm_impl.h:214-216）：HiF4 传 `scaleSendCount×sizeof(hif4_scale)`（字节口径）作 dataCount、`scaleSendCount`（元素口径）作 strideCount。HCCL 头文件 fp16 示例暗示 dataCount 按元素计数，若按元素则 HiF4 4 倍超发；若按字节则正确（strideCount 需按元素）。MX 路径 E8M0 为 1 字节元素，两种口径等价，存量用例无法区分 | 多卡 scale 通路错乱 → 精度失败或越界 | 与 HCCL 确认单位约定；以 ST 定向用例（WB-SCALE，rank 标签 scale）实测；若按元素需修正为 `scaleSendCount` |
| **R6** | **ST main_hif4.cpp 通信 tiling 单位错误**：`headTileCnt = mTotal/headMSize`、`tailMSize = mTotal%headMSize` 使用 M_total，与主线 splitAxis 覆盖 per-rank m 的约定不一致（AllToAll stride=`splitAxis×Ka/2` 必须=`m×Ka/2`），splitAxis=M_total 时目的块 j≥1 send 偏移越界、recv 侧 2 倍溢出 | ST 直调多卡执行必失败（devkit 依赖未就位，当前静态交付未暴露） | 修复为 `m/headMSize`、`m%headMSize`（ST 文档 F1）；修复后 WB-03/04/05 通信形态断言方成立 |
| **R7** | **Blaze `IsFp4()` 未包含 hifloat4 的布局风险**：qbmm kernel 的 `C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8` 与 `kCacheLineAlignMask = IsFp4 ? 0xff : 0x7f` 均以 IsFp4 判定。hifloat4x2_t 非 fp4x2_e2m1_t，若 Blaze 侧未按上游设计（v1.1 §4.5.4"IsFp4() 增加 hifloat4，已提需求"）扩展，HiF4 的 A/B 数据 framing 将取 C0_SIZE_B8（32 元素）而非 4-bit 所需的 64 元素档，布局错配 | Kernel 计算布局错乱（Blaze HiF4 分发就位时需一并确认） | 与 Blaze 确认 IsFp4/HiF4 帧档位；必要时仓内为 HiF4 显式取 C0_SIZE_B4（与 mask 0x7f 的现状一并列出复核） |
| **R8** | **HiF4 在 950 上被误使能**：上游设计 v1.1 §7.1 明确 arch35(950) 不支持 HiF4（仅 arch36/960DT 支持），但实现的 `IsDavidArch()=DAV_3510||DAV_9201`、tiling 注册 {DAV_3510, DAV_9201}、MX dtype 列表追加 DT_HIFLOAT4 均对 950 同样生效——950 上 HiF4 请求会通过 host 校验并路由进 HiF4 tiling（def 原型 950 无对应 dtype 组合，最终被 FE 拒绝，但错误发生在图阶段而非参数校验阶段） | 950 上错误拦截层级不当；若 950 侧后续误配 def 组合则产生不可用路径 | op_api/tiling 层将 HiF4 分支显式限定 `NpuArch::DAV_9201`（用户已确认 HiF4 仅 A6 支持） |

## 11. 外部依赖

| 依赖 | 提供方 | 用途 |
| :--- | :--- | :--- |
| `hifloat4x2_t` / `hif4_scale` 类型 | devkit（kernel 侧基础类型） | 4-bit 数据与 scale 元素类型 |
| `Blaze::Gemm::Kernel::IsHif4<X1Type>()` | devkit/blaze | HiF4 类型分发 |
| Blaze BlockMmad HiF4 分发（ScaleAType/ScaleBType 导出、反量化 mmad） | devkit/blaze | 计算核 |
| `DT_HIFLOAT4 / DT_HIFLOAT4_SCALE` GE dtype | devkit（ge 枚举） | 算子定义/op_api/tiling |
| HCCL UINT32 AlltoAll | HCCL | hif4_scale 4 字节通信 |

## 12. 附录：关键常量速查

| 常量 | 值 | 位置 |
| :--- | :--- | :--- |
| HIFLOAT4_QUANT | 8 | matmul_allto_all_util.h |
| HIF4_PREGROUP_QUANT_MODE | 4 | all_to_all_matmul_tiling_key.h |
| HIF4_QUANTMODE_VALUES | 8 | allto_all_mx_quant_matmul_tiling_base.h |
| HIF4_SCALE_BLOCK_K / groupSizeK | 64 | 同上（groupSize 打包 4295032896） |
| HIF4_SCALE_LAST_DIM | 1 | 同上 |
| HIF4_SCALE_UNIT_BYTES | 4 | 同上 |
| HIF4_KA_ALIGN_SIZE | 128 | 同上 / op_api / ST |
| HIF4_GROUP_SIZE | 64 | apace_constant.h |
| HIF4_SCALE_UNIT | 4 | apace_constant.h |
| TILING_HIF4_MULTI_BASE_SIZE | 1 | apace_constant.h |
| APACE_IMPL_MAX_RANK_DIM | 8 | tiling / ST |
| ALIGN_NUM（workspace 对齐） | 512 | hcomm impl / ST |
| MXFP_DATA_NUM_PER_BYTE | 2 | hcomm impl / ST |

## 13. 与上游设计文档（dingwei002 v1.1）对照

> 对照对象：`hifloat4_apace_allto_all_matmul_design_v1.1.md`（dingwei002/SuperHub，参考代码基线即本特性 47b981b42 的源仓库）。
> 本文档（§1-§12）以**代码实现（47b981b42）为真值源**编写；本章给出与 v1.1 的逐项比对结论。

### 13.1 一致项（实现与 v1.1 及本文档三方一致）

| 项 | v1.1 章节 | 实现位置 |
| :--- | :--- | :--- |
| HIFLOAT4_QUANT=8 量化模式 | §3.1 | matmul_allto_all_util.h |
| TilingKey：HIF4_PREGROUP_QUANT_MODE=4、X2TRANSPOSE 0/1、AIC_ONLY、DTYPEBIAS 恒 FP32、ADDBIAS_OFF、COMMTYPE CCU/AICPU | §4.2.3 | all_to_all_matmul_tiling_key.h |
| hif4_scale 32-bit 三级元数据布局（byte0 E6M2 / byte1 E1_8 / byte2-3 E1_16，little-endian） | §1.1.2.4 | gen_data.py / §2.3 |
| 反量化公式 V=E6M2×2^(E1_8[⌊i/8⌋]+E1_16[⌊i/4⌋])×S1P2[i] | §1.1.2.3 | gen_data.py hif4_dequant |
| scale shape [M, Ceil(Ka/64), 1]（每元素 4 字节） | §1.1.2.4 / §3.1 | HIF4_SCALE_LAST_DIM=1 |
| 2:1 打包（hifloat4x2_t，与 fp4x2_e2m1_t 打包一致） | §1.1.2.1 | §2.2 |
| 仅 Hcomm（AIC-only），不设计 URMA 通路 | §1/§7.5 | apt.cpp static_assert |
| group_size=(1,1,64) | §1.3 | CheckGroupSize |
| scale 通信 UINT32 / 数据通信 UINT8 | §4.5.2 | hcomm_impl 常量 |
| TransB 模板参数 + LayoutB conditional（DN/ND） | §3.2/§4.5.1 | hcomm_impl 模板 |
| SWAT 泛化扩展点 1-4（mm::DT_HIFLOAT4 枚举、GetShape/GetSizeWithDataType、baseK 4-bit 256 档、GetPerGroupSize/GetScaleUnitPerGroup/GetAlignSize + baseScaleSize 公式） | §4.2.4 | quant_matmul_tiling_swat.h / apace_common_utils.h（逐条一致） |
| hcommAllToAllMatmulTilingData 结构与字段语义 | §4.2.2 | all_to_all_matmul_tiling_data.h |
| workspace 布局 [scale 收包区 | 数据收包区 (+DFX 64KB)] | §4.2.5 | hcomm_impl Init / main_hif4.cpp |
| ascend960dt def 原型（950 组合末尾追加 HiF4 dtype 组合） | §3.2 | allto_all_matmul_def.cpp |
| MxScaleKLen / scaleKRowBytes / SCALE_C0 泛化 | §4.5.3 | all_to_all_qbmm_mx_kernel.h |
| ST 框架结构与 7 条用例表 | §8 | cases_hif4.csv（逐条一致） |
| 精度阈值 rel_err<0.1（BF16 级） | §8.6 | verify_result.py HIF4_ERROR_TOL=1e-1 |
| localMatmul 三模式能力保留、当前不可切换（恒 0） | §6.3 | DoHcommMxQuantMMTiling localMatmulDisabled=0 |
| x2Scale ND 布局 shape [Ceil(K,64), N, 1] | §3.1 | op_api Check3DScaleShape |

### 13.2 出入项（实现 ≠ v1.1）

| 编号 | 项 | v1.1 说法 | 实现/本文档说法 | 判定 |
| :--- | :--- | :--- | :--- | :--- |
| D1 | Impl 类组织 | §1/§3.2：新增独立类 `AllToAllHiF4QuantMatmulHcommImpl`（"kernel 部分不能与 MXFP4 共用"），模板签名 6 参数 | 复用泛化 `AllToAllMxQuantMatmulHcommImpl`（+ScaleType/+TransB，9 参数），未新建类 | 设计演进：实现以模板参数封装差异，符合 v1.1 §2"复用 MXFP4、差异接口封装"总原则；v1.1 类名/签名过时 |
| **D2** | **S1P2 码值表** | §1.1.2.1：S/I/F 定点，码值 {0, 0.25, …, 1.75}（0.25 步进，最大 1.75）；但 §1.3 BLAZE 规格又写 "hif4(e1m2)"，**自相矛盾** | 码值 {0, 0.5, 1, 1.5, 2, 3, 4, 6}（E2M1 值域，gen_data.py 注释 "S1P2(E2M1)"） | **高风险待确认**：若硬件按论文 S1P2 定点表（0.25 步进），ST golden 与硬件不一致。组内动态范围 v1.1 称 0.25~7.0，E2M1 值域下为 0.5~6 |
| **D3** | **E6M2 指数偏置** | §1.1.2.2：bias=48，范围 2^-48 ~ 2^15×1.50 | bias=31（2^(⌊em/4⌋-31)×(1+M/4)，标准 fp8-e6m2 解码，gen_data.py） | **高风险待确认**：golden 解码与 v1.1 规格不一致，需对齐论文/硬件定义 |
| D4 | bias 支持 | §1.3 交付规格："bias：不支持"（BLAZE 表"暂无需求"） | 支持可选 FP32 bias（op_api/tiling/kernel 全链路） | 实现超出 v1.1 交付范围（后补能力），本文档按实现描述 |
| D5 | y 输出 dtype | §1.3：仅 FP16/BF16 两种 | FP16/BF16/**FP32** 三种（def 960dt y 列表与 op_api 支持列表均含 FLOAT） | 实现超出 v1.1；本文档含 FP32 |
| D6 | scale shape dim2 | §7.3："dim2 == 4"（与自身 §1.1.2.4/§3.1 的 1 矛盾） | dim2==1，每元素 4 字节 | v1.1 §7.3 笔误（混淆字节数与元素数）；实现/本文档正确 |
| D7 | Ka 对齐约束 | §7.2："ka 是 64 倍数" 与 §8.4 "CeilDiv(k/rank_num,64)%2==0"（=Ka%128）自相矛盾 | Ka%128==0（op_api+tiling 显式校验） | 实现取更严的 128 对齐，与 v1.1 §8.4 一致；§7.2 说法过时 |
| D8 | 芯片支持范围 | §7.1：arch36(960DT)✅，arch35(950)❌ 明确不支持 | `IsDavidArch()=3510||9201`、tiling 注册 {3510,9201}、MX dtype 列表加 DT_HIFLOAT4——**950 上 HiF4 校验/tiling 路径同样打开**（最终被 950 def 无 dtype 组合挡在 FE 层） | 出入 → 风险 R8：建议 host 侧显式限定 DAV_9201（用户已确认 HiF4 仅 A6 支持） |
| D9 | IsFp4 扩展 | §4.5.4："IsFp4() 增加 hifloat4（已提需求）" | 仓内未见对应扩展（Blaze 侧待办）；qbmm 的 C0_SIZE/mask 以 IsFp4 判定，HiF4 当前走 B8/0x7f 分支 | 依赖缺口 → 风险 R7 |
| D10 | ST launcher 细节 | §8.2：ServerType=AICPU、无 transB/workspace 参数 | CCU + 运行时 transB 双分支 + workspaceGM 参数 | 实现更完整；v1.1 示例为草案 |
| D11 | PTA / DFX | §4.4 PTA(torchair/lingqu 私仓)、§5 DFX 补齐要求 | 本文档范围仅 CANN 仓，未覆盖 | 范围差异（非冲突）；PTA/DFX 由对应仓/团队跟踪 |
| D12 | 通信切分单位 | §8.3 仅给 `splitAxisTileSize=headMSize` 等赋值，未定义 headMSize 语义 | splitAxis 覆盖 per-rank m（推导见 §7.2）；ST main_hif4.cpp 误用 M_total → R6 | v1.1 未写清单位，实现(ST 侧)引入缺陷 R6 |
| D13 | 9201 枚举/`Ascend960DT` | §9：Host 侧 NpuArch::DAV_9201、Kernel 侧 __NPU__ARCH__==9201 | 一致（DAV_9201=9201） | 一致项（补充：A5 包无 DT_HIFLOAT4_SCALE/ACL_HIFLOAT4，见 R2 与 UT 文档前置依赖） |

### 13.3 对照结论

- **架构/接口/tilingKey/数据格式框架层**：本文档与 v1.1 高度一致（§13.1 的 21 项），v1.1 的 SWAT 扩展点 1-4 与实现逐条对应。
- **需上游确认的实质出入**：D2（S1P2 码值表）、D3（E6M2 bias）两处直接影响 golden 正确性，建议在 A6 devkit 就位前与 HiF4 论文/硬件团队对齐；D8（950 误使能）、D9（IsFp4 依赖）已列入风险 R7/R8。
- **实现超出 v1.1 的部分**（bias、FP32 输出、ST transB/CCU）以实现为准，本文档已按实现描述。
- **v1.1 自身矛盾处**（D2/D6/D7）以实现与 v1.1 交付规格章节为准修正。
