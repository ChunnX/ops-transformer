# AlltoAllMatmul MX 量化场景右矩阵非转置支持（Apace 模板）详细设计

| 项目 | 内容 |
| ---- | ---- |
| 算子 | AlltoAllMatmul（mc2/allto_all_matmul） |
| 场景 | MX 量化（x1QuantMode=6, x2QuantMode=6），Ascend 950（DAV_3510 / arch35） |
| 特性 | 右矩阵 x2 支持非转置（transpose_x2=False） |
| 实现范围 | **仅 Apace 模板（hcomm 纯 C 核路线，USING_APACE_IMPL=1）**；主线 pipeline（USING_APACE_IMPL=0）保持"必须转置"约束不变 |
| 分支 | feat/a2amm-mx-nt-x2 |

## 1. 需求背景与现状分析

### 1.1 现状

当前 AlltoAllMatmul 在 MX 量化（mxfp8/mxfp4）场景下，右矩阵 x2 **必须转置**（transpose_x2=True，x2 shape 为 (N, H*rankSize)），非转置输入在多层被拒绝：

| 层 | 位置 | 现状逻辑 |
| ---- | ---- | ---- |
| aclnn API 层 | `op_api/allto_all_quant_matmul_base.cpp` `CheckScaleShape()` | MX 场景硬性校验 `transposeX2==true`，否则报 `In the mx quantization scenario, x2 must be transposed.` |
| Tiling 层 | `op_host/op_tiling/arch35/allto_all_mx_quant_matmul_tiling_base.cpp` `CheckX2Transpose()` | 同样硬性要求 `*isTransX2 == true` |
| Tiling 层 | 同文件 `CheckMxQuantShapeInfo()` | x2 的 K 轴、x2Scale 的 shape 校验均按"转置布局"写死 |
| Kernel 层 | `op_kernel/arch35/allto_all_matmul_apt.cpp` | `AllToAllMxQuantMatmulHcommImpl`（Apace 模板）模板参数未包含 TransB，`LayoutB` 硬编码 `dn_ext_layout_ptn`（转置布局） |

### 1.2 目标

MX 量化场景下，当满足 Apace 路线准入条件（all2all_out 为空 且 rankDim ≤ 8，即 `usingApaceImpl_ == true`）时，支持右矩阵非转置输入：

- `transpose_x2=False`，x2 shape 为 `(H*rankSize, N)`（K 主序，ND 行主序存储）；
- x2Scale shape 为 `(ceil(H*rankSize/64), N, 2)`（与 kernel 侧 `scaleb_nd_layout_ptn` 物理排布一致）。

不满足 Apace 准入条件（all2all_out 非空或 rankDim>8，走主线 pipeline）时，仍要求 x2 必须转置并报错。

### 1.3 已具备的基础设施（无需改动）

- **Blaze BlockMmad / AllToAllQbmmMxKernel**（`mc2/common/op_kernel/apace/kernel/matmul/quant_batch_matmul/all_to_all_qbmm_mx_kernel.h`）：已通过 `BlockMmad::TRANS_B = IsTrans<LayoutB>` 支持两种布局；`MakeLayoutScaleB` 按 `transB` 条件选择 `scaleb_dn_layout_ptn` / `scaleb_nd_layout_ptn`；`SetL2Cache` / `SetScaleL2Cache` 已有 transB 分支。
- **Apace tiling SWAT**（`mc2/common/op_kernel/apace/tiling/quant_matmul_tiling_swat.h`）：`GetTilingData(m, n, k, transA, transB, ...)` 已支持 transB=false（baseN 对齐、尾块切分等已有分支）。
- **URMA 参考实现**（`all_to_all_mx_quant_matmul_urma_impl.h`）：已采用 `conditional_t<TransB, dn_ext_layout_ptn, nd_ext_layout_ptn>` 模板范式，本设计照搬该范式。
- **TilingKey 声明**（`op_kernel/arch35/allto_all_matmul_tiling_key.h`）：MX+APACE 组合已声明 `X2TRANSPOSE(0,1)` 双值，无需修改。
- **tiling 现有传参**：`DoHcommMxQuantMMTiling()` 已把 `isBTrans` 传给 SWAT tiling；`GetTilingKey()` 已按 `isBTrans` 编码 X2TRANSPOSE 位。
- **infershape / CheckShapeAAMM / InferGroupSize / CheckMatrixMulShapes / SetAlltoAllMatmulShapeInfo**：均按 `isBTrans` 双分支实现，天然兼容非转置。

### 1.4 布局约定（关键结论）

Blaze kernel 内部视图（`all_to_all_qbmm_mx_kernel.h` ProcessSingleBatch）：

```
layoutB      = MakeLayoutB{}(rankSize * K, N)        // K_total x N
layoutScaleB = MakeLayoutScaleB{}(rankSize * scaleKLen, N)  // scaleK_total x N, scaleKLen = CeilDiv(K,64)*2
```

| x2 布局 | LayoutB 类型 | x2 物理排布（用户视角） | x2Scale 物理排布（用户视角 3D shape） |
| ---- | ---- | ---- | ---- |
| 转置（现状） | `dn_ext_layout_ptn`（列主序） | (N, K_total) C-order | (N, K_total/64, 2) C-order |
| 非转置（新增） | `nd_ext_layout_ptn`（行主序） | (K_total, N) C-order | (K_total/64, N, 2) C-order |

`scaleb_nd_layout_ptn` 的物理排布（`layout_pattern.h`）：offset(kScaleIdx, n) = 2*(g*N + n) + s，其中 kScaleIdx=2g+s —— 即 (K/64, N, 2) C-order，与上表一致。该约定与 aclnn 层 `Check3DScaleShape`（`x2ScaleNVal = transposeX2 ? dim0 : dim1`）、`InferGroupSize`（`nIdx = isBTrans ? 0 : 1`）的既有双分支逻辑吻合。

## 2. 总体方案

数据流不变：AIC 核先下发 x1/x1Scale 的 AlltoAll 通信，再按 tile 等待通信完成并做 Blaze MX 量化 MatMul。**本次改动只影响 B 矩阵（x2）与 scaleB（x2Scale）的读取布局**，通信、调度、workspace 布局均不涉及 x2，无需变更。

```
                     ┌─────────────────────────────────────────────────┐
                     │ aclnn API 层（CheckScaleShape）                    │
                     │  放开 MX 场景 transposeX2=false 硬校验（A5）        │
                     └─────────────────────────────────────────────────┘
                                          ↓
                     ┌─────────────────────────────────────────────────┐
                     │ Tiling 层（arch35 allto_all_mx_quant_matmul）      │
                     │ 1. CheckX2Transpose: 仅 Apace 准入时允许 false      │
                     │ 2. CheckMxQuantShapeInfo: x2 K 轴/x2Scale 双分支    │
                     └─────────────────────────────────────────────────┘
                                          ↓ tilingKey(X2TRANSPOSE=0, USING_APACE_IMPL=1)
                     ┌─────────────────────────────────────────────────┐
                     │ Kernel 层                                          │
                     │ 1. HcommImpl 增加 TransB 模板参数（默认 true）      │
                     │ 2. LayoutB 按 TransB 条件选择 nd/dn                 │
                     │ 3. apt.cpp 实例化传 X2TRANSPOSE                    │
                     └─────────────────────────────────────────────────┘
```

## 3. 详细设计

### 3.1 Kernel 层

#### 3.1.1 `mc2/common/op_kernel/apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_hcomm_impl.h`

1. 类模板参数追加 `bool TransB = true`（带默认值，兼容既有测试代码显式全参实例化）：

```cpp
template <typename X1Type, typename X2Type, typename YType, typename CommDataTypeX1,
          typename AlltoAllMatmulTilingDataType, AscendC::HcclServerType ServerType, bool IsMxFp4,
          bool TransB = true>
class AllToAllMxQuantMatmulHcommImpl { ... };
```

2. `LayoutB` 改为条件类型（与 URMA 实现保持一致）：

```cpp
using LayoutB =
    typename AscendC::Std::conditional_t<TransB, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
```

3. 所有成员函数的模板参数列表同步追加 `TransB`。

**不需要改动的部分**（论证）：
- `SetupParams` 中 `bGmAddr = x2_`、`scaleBGmAddr = x2Scale_` 直接透传用户地址，布局解释由 Blaze 层 LayoutB/LayoutScaleB 承担；
- 通信下发（`Init` 中 AlltoAll 提交）只针对 x1/x1Scale，与 x2 无关；
- `BlockSchedulerQuantBatchMatmulV3` / `MatmulWithScaleMx` / `BlockMmad` 均以 LayoutB 为模板参数派生 TRANS_B，Blaze 层已支持。

#### 3.1.2 `mc2/allto_all_matmul/op_kernel/arch35/allto_all_matmul_apt.cpp`

FP8 与 FP4 两处 `AllToAllMxQuantMatmulHcommImpl` 实例化均追加模板实参 `X2TRANSPOSE`：

```cpp
Apace::AllToAllMxQuantMatmulHcommImpl<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_X1,
                                      Apace::hcommAllToAllMatmulTilingData, hcclServerType, false, X2TRANSPOSE>
```

（FP8 分支 IsMxFp4=false，FP4 分支 IsMxFp4=true。）

`X2TRANSPOSE` 由 tilingKey 模板实参机制注入：tiling 侧对 MX+APACE+非转置产出 key（X2TRANSPOSE=0），自动选择 `ASCENDC_TPL_AIC_ONLY + X2TRANSPOSE=0` 的实例化，tiling_key.h 声明表已覆盖，无需改动。

### 3.2 Tiling 层

文件：`mc2/allto_all_matmul/op_host/op_tiling/arch35/allto_all_mx_quant_matmul_tiling_base.cpp`（及 `.h`）

#### 3.2.1 `CheckX2Transpose()` —— 按实现路线放行

`CheckOpInputInfo()` 在 `InitTilingContextParameters()`（设置 `usingApaceImpl_`）之前执行，因此在 `CheckX2Transpose` 内部就地判定 Apace 准入条件（与 `InitTilingContextParameters` 中 `usingApaceImpl_ = (!allToAllOutFlag) && (rankDim <= 8)` 完全一致的条件）：

```cpp
ge::graphStatus AllToAllMxQuantMatmulTilingBase::CheckX2Transpose(const gert::TilingContext *context,
                                                                   const char *opName,
                                                                   const OpAttrIndexSchema &indexSchema)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const bool *isTransX2 = attrs->GetAttrPointer<bool>(indexSchema.x2Transpose);
    OP_TILING_CHECK(isTransX2 == nullptr, OP_LOGE_WITH_INVALID_INPUT(opName, "isTransX2"), return ge::GRAPH_FAILED);
    if (*isTransX2) {
        return ge::GRAPH_SUCCESS;
    }
    // 非转置仅 Apace(hcomm) 路线支持: all2all_out 为空 且 卡数 <= 8
    constexpr uint64_t APACE_IMPL_MAX_RANK_DIM = 8;
    const bool *allToAllOutFlag = attrs->GetAttrPointer<bool>(ALLTOALLMATMUL_ATTR_ALLTO_ALL_OUT_FLAG_INDEX);
    bool hasAllToAllOut = (allToAllOutFlag != nullptr) && (*allToAllOutFlag);
    int64_t rankDim = 0;
    if (MatmulAlltoAllTilingUtil::GetAndValidateRankSize(context, opName,
            attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX), rankDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (hasAllToAllOut || (static_cast<uint64_t>(rankDim) > APACE_IMPL_MAX_RANK_DIM)) {
        OP_LOGE_FOR_INVALID_VALUE(opName, "x2Transpose", "false",
            "true (x2 non-transpose in mx quant scenario is only supported by the apace impl, "
            "which requires empty all2all_out and rankSize <= 8)");
        return ge::GRAPH_FAILED;
    }
    OP_LOGI(opName, "Mx quant uses non-transposed x2 with apace impl.");
    return ge::GRAPH_SUCCESS;
}
```

注意：`rankDim` 解析失败时直接 FAILED（与公共校验保持一致的严格语义）；本函数在 `CheckAttrsInfo` 之后执行，group/world_size 属性已保证非空。

#### 3.2.2 `CheckMxQuantShapeInfo()` —— x2/x2Scale shape 校验双分支

现校验按转置布局写死（`x2Dim1` 为 K、`x2ScaleDim0==x2Dim0` 且 `x2ScaleDim1==x2Dim1/64`）。改为按 `isTransX2` 双分支：

```cpp
// K 轴（H*rankSize）: 转置时为 dim1，非转置时为 dim0
uint64_t x2KDim = x2IsTrans ? x2Dim1 : x2Dim0;
OP_TILING_CHECK((x2KDim % MX_SCALE_ALIGN != 0), ... "The dim(k) of x2 must be divisible by 64.");

// x2Scale shape:
//   转置:    (N, K/64, 2)      → dim0 == x2Dim0, dim1 == x2Dim1 / 64
//   非转置:  (K/64, N, 2)      → dim0 == x2Dim0 / 64, dim1 == x2Dim1
if (x2IsTrans) {
    OP_TILING_CHECK((x2ScaleDim0 != x2Dim0) || (x2ScaleDim1 != x2Dim1 / MX_SCALE_ALIGN) || (x2ScaleDim2 != DIM_TWO), ...);
} else {
    OP_TILING_CHECK((x2ScaleDim0 != x2Dim0 / MX_SCALE_ALIGN) || (x2ScaleDim1 != x2Dim1) || (x2ScaleDim2 != DIM_TWO), ...);
}
```

x1/x1Scale 校验不变（x1 布局与 x2 转置无关）。`CheckMatrixMulShapes`、`CheckGroupSize`、`SetAlltoAllMatmulShapeInfo`、`SetUserWorkSpace`、`BuildWorkspaceLayoutApace` 均为双分支/与 x2 布局无关，不动。

> 说明：函数内已有 `attrs->GetAttrPointer<bool>(indexSchema.x2Transpose)` 的读取（现用于 `x2IsTransFlag` 相关分支之外的逻辑），复用该变量即可。

### 3.3 aclnn API 层

文件：`mc2/allto_all_matmul/op_api/allto_all_quant_matmul_base.cpp`

`CheckScaleShape()` MX 分支删除 `!transposeX2` 硬校验。放行后由 tiling 层（3.2.1）做实现路线维度的精确裁决：

- Apace 准入（all2all_out 空 且 ≤8 卡）→ tiling 通过，走 Apace NT kernel；
- 否则 → tiling `CheckX2Transpose` 报错（错误信息明确指引"非转置仅 Apace 路线支持"）。

`Check3DScaleShape()` 已按 `transposeX2` 双分支校验 N 轴（`x2ScaleNVal = transposeX2 ? dim0 : dim1`）与 last-dim==2，天然兼容，无需改动。`CheckShapeAAMM`、非连续处理（`IsTransposeLastTwoDims`/`TransX2Tensor`）均为通用逻辑，无需改动。

### 3.4 TilingKey 与二进制

- `GET_TPL_TILING_KEY(MX_QUANT_MODE, x2TransposeFlag, ...)` 已按 `isBTrans` 编码，非转置自动产出 X2TRANSPOSE=0 的 key；
- key 编码（4bit×UINT、1bit×BOOL 顺序拼位）：MX+APACE+CCU+转置=16435（既有 UT 已验证），**MX+APACE+CCU+非转置=16419**；ai_cpu 对应 17459/17443；
- `allto_all_matmul_tiling_key.h` SEL 表已声明 `X2TRANSPOSE(0,1)`（MX+USING_APACE_IMPL=1 组合），编译期自动实例化新增模板组合，无需改声明。

### 3.5 文档

1. `mc2/allto_all_matmul/README.md` 约束说明：x2 必须转置的表述追加 Apace 路线非转置例外及 x2Scale 非转置 shape 说明。
2. `mc2/allto_all_matmul/docs/aclnnAlltoAllQuantMatmul.md` 维度约束：同步更新（注明仅 Apace 实现路线——all2all_out 为空且 rankSize≤8——支持非转置，及两种布局的 x2/x2Scale shape 约定）。

## 4. 测试设计

### 4.1 UT（tiling，arch35）

文件：`mc2/allto_all_matmul/tests/ut/op_host/arch35/test_allto_all_matmul_tiling.csv`

| 用例 | 关键参数 | 期望 |
| ---- | ---- | ---- |
| `alltoall_matmul_mx_apace_case_legal_nt_nobias_e4m3_e4m3_2p` | x1=(57086,1536) E4M3；x2=(3072,9216)（K_total,N）；x2Scale=(48,9216,2)；worldSize=2；transposeX2=false；ccu | SUCCESS，tilingKey=16419，workspace 同转置用例 197691392 |
| `alltoall_matmul_mx_apace_case_legal_nt_bias_e4m3_e4m3_4p` | 同上 4p 规模，带 bias | SUCCESS，tilingKey=16419 |
| `alltoall_matmul_mx_apace_case_legal_nt_bias_fp4_fp4_2p` | FP4 组合，非转置 | SUCCESS |
| `alltoall_matmul_mx_apace_case_commode_aicpu_nt` | 非转置 + ai_cpu | SUCCESS，tilingKey=17443 |
| `alltoall_matmul_mx_mainline_case_illegal_x2trans_false`（由原 `..._apace_case_illegal_x2trans_false` 改造） | 非转置 + alltoalloutFlag=true（强制主线） | FAILED |
| `alltoall_matmul_mx_mainline_case_illegal_x2trans_false_16p` | 非转置 + 16 卡（超 Apace 上限） | FAILED |
| `alltoall_matmul_mx_apace_case_illegal_nt_x2scale_shape` | 非转置但 x2Scale 仍按转置布局 (N,K/64,2) 传 | FAILED（校验 3.2.2 生效） |

### 4.2 UT（op_api）

文件：`mc2/allto_all_matmul/tests/ut/op_api/test_aclnn_allto_all_quant_matmul.csv`

新增 MX 非转置 SUCCESS 用例（x2=(K_total,N)、x2Scale=(K/64,N,2)、transpose_x2=0）与 x2Scale 布局错误用例。

### 4.3 编译验证

```
./build.sh --ops=allto_all_matmul --soc=ascend950 --ophost --opapi --opkernel
```

重点确认：
- arch35 kernel 新增 `X2TRANSPOSE=0, USING_APACE_IMPL=1` 模板实例（AIC_ONLY）编译通过；
- Blaze `nd_ext_layout_ptn` B 路径（CopyBToL1/scaleb_nd）实例化成功；
- ophost/opapi 编译通过，UT（`--ophost_test --ops=allto_all_matmul --soc=ascend950`）通过。

## 5. 影响面与兼容性

| 项 | 影响 |
| ---- | ---- |
| 既有转置场景 | 无行为变化（TransB=true 路径、既有 tilingKey、既有校验逻辑均不变） |
| 主线 pipeline MX 场景 | 不放行非转置（CheckX2Transpose 按 Apace 准入条件精确拦截） |
| AlltoAllMatmulV2 / arch22 | 不涉及（V2 tiling key 固定 x2Transpose=1；arch22 无 MX Apace 路线） |
| 接口签名 | aclnn 接口无签名变化；HcommImpl 模板参数追加带默认值，不影响既有显式全参调用 |
| 二进制体积 | arch35 新增 MX+APACE+X2TRANSPOSE=0 的 kernel 实例（FP8/FP4 × CCU/AICPU 组合由 SEL 表决定） |

## 6. 交付清单

| 文件 | 变更 |
| ---- | ---- |
| `mc2/common/op_kernel/apace/kernel/fusions/all_to_all_quant_matmul/all_to_all_mx_quant_matmul_hcomm_impl.h` | 增加 TransB 模板参数 + LayoutB 条件类型 |
| `mc2/allto_all_matmul/op_kernel/arch35/allto_all_matmul_apt.cpp` | 2 处实例化传入 X2TRANSPOSE |
| `mc2/allto_all_matmul/op_host/op_tiling/arch35/allto_all_mx_quant_matmul_tiling_base.cpp` | CheckX2Transpose 按路线放行；CheckMxQuantShapeInfo 双分支 |
| `mc2/allto_all_matmul/op_api/allto_all_quant_matmul_base.cpp` | CheckScaleShape 删除 MX 非转置硬校验 |
| `mc2/allto_all_matmul/tests/ut/op_host/arch35/test_allto_all_matmul_tiling.csv` | 用例增改（见 4.1） |
| `mc2/allto_all_matmul/tests/ut/op_api/test_aclnn_allto_all_quant_matmul.csv` | 用例新增（见 4.2） |
| `mc2/allto_all_matmul/README.md`、`docs/aclnnAlltoAllQuantMatmul.md` | 约束说明更新（并修正文档中转置 shape 描述的既有笔误） |

## 7. 验证结果（Ascend 950）

| 验证项 | 命令 | 结果 |
| ---- | ---- | ---- |
| ophost 编译 | `build.sh --ops=allto_all_matmul --soc=ascend950 --ophost` | 通过 |
| opapi 编译 | `build.sh --ops=allto_all_matmul --soc=ascend950 --opapi` | 通过 |
| opkernel 编译 | `build.sh --ops=allto_all_matmul --soc=ascend950 --opkernel` | 通过；binary_info 中新增 tilingKey 16419（CCU+NT）/17443（AICPU+NT） |
| ophost UT | `build.sh --ophost_test --ops=allto_all_matmul --soc=ascend950` | 119/119 通过，含 9 个新增 NT 用例（6 合法 + 3 非法） |
| opapi UT | `build.sh --opapi_test --ops=allto_all_matmul --soc=ascend950 --noexec` | 69/69 通过，含 3 个新增 NT 用例 |

> 精度/性能验收（ST）需 NPU 环境，本环境无设备，待上板执行：可复用 `tests/st/aclnnAlltoAllMatmul` 执行器构造 transposeX2=False 用例。
