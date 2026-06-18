## 1. 基本规格

### 1.1 MXFP4数据格式定义

MXFP4 (Microscaling FP4) 的核心规格：

| 参数 | MXFP8 | MXFP4 | 说明 |
|------|-------|-------|------|
| x1/x2  | FP8_E4M3FN / FP8_E5M2 | **FP4_E2M1** (DT_FLOAT4_E2M1) | fp4为2bit指数+1bit尾数，4bit总宽度 |
| x1/x2 shape | [M,K],[N,K]（仅支持转置） | **[M,K],[N,K]（x2仅支持转置）**  | |
| Scale dtype | DT_FLOAT8_E8M0 | **DT_FLOAT8_E8M0** | 与MXFP8共享，纯指数格式 |
| Scale dim | 3 | **3** | 与MXFP8共享3维结构 |
| groupSize | [1, 1, 32] | **[1, 1, 32]** | 与MXFP8完全相同的分组策略 |
| x1Scale/x2Scale shape | (M, ceilDiv(K,64), 2),(N, ceilDiv(K,64), 2)   | **(M, ceilDiv(K,64), 2),(N, ceilDiv(K,64), 2) ** | 与MXFP8完全相同 |
| bias dtype | DT_FLOAT | **DT_FLOAT** | 与MXFP8共享 |
| output dtype | FP16/BF16/FLOAT | **FP16/BF16/FLOAT** | 与MXFP8共享 |

### 1.2 MXFP4的补充约束

1. **x1/x2必须类型相同**：MXFP4场景下x1和x2都必须为DT_FLOAT4_E2M1
2. **K和ceilDiv(K,32)必须为偶数**：因为fp4打包因子为2，K轴元素数和K分组数必须为偶数
3. **x2仅支持转置（transposeB=true）**：MX量化场景强制要求

### 1.3 输入/输出dtype组合规格

MXFP4新增组合（在MXFP8的12组基础上新增3组）：

| # | x1 dtype | x2 dtype | x1Scale dtype | x2Scale dtype | output dtype | 量化模式 |
|---|----------|----------|---------------|---------------|-------------|---------|
| 29 | **FP4_E2M1** | **FP4_E2M1** | FLOAT8_E8M0 | FLOAT8_E8M0 | FP16 | MXFP4 |
| 30 | **FP4_E2M1** | **FP4_E2M1** | FLOAT8_E8M0 | FLOAT8_E8M0 | BF16 | MXFP4 |
| 31 | **FP4_E2M1** | **FP4_E2M1** | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT | MXFP4 |

---

## 2. 当前A5 MXFP8架构分析

### 2.1 校验链路

MXFP8校验分两层：

- **L1 API层** (`aclnn_matmul_reduce_scatter_v2.cpp`): dtype范围检查 → CaseOption判定(3D+E8M0 → LOW_ACCURACY_PER_TENSOR_WITH_QUANT_AMAX) → Scale转置处理
- **L2 Tiling层** (`quant_bmm_reduce_scatter_tiling.cpp`): SetScene(dim=3+E8M0 → MXFP_MODE) → CheckMxScaleDim → CheckGroupSize([1,1,32]) → MxfpSceneParamCheck(x1/x2∈{E4M3FN,E5M2})

### 2.2 Tiling数据流

```
DoOpTiling()
  ├── CheckHCCLSize()                    // 检查通信约束（单次≤256MB）
  ├── CheckInput()                       // 参数校验 + SetScene() 识别量化模式
  ├── SetMc2Hcomm()                      // 设置 HCCL 通信参数
  ├── SetRcsTilingData()                 // 设置基础 RCS tiling 参数
  ├── DoSplitMTiling()                   // ★ M 维公式化tiling切分
  ├── AdjustHCCLLimit()                  // 调整 HCCL 256MB 限制
  ├── DoAdaptSlidWindowTiling()          // MatMul 滑窗 tiling
  └── SetTilingResult()                  // 设置最终 tiling 结果

```

### 2.3 Kernel

kernel入口：

![image](./ff655bc5-678e-4291-b90f-eb09789651ba.png)

使用的matmul模版：`Mc2MatmulV3::Mc2QuantBatchMatmulASWKernel`

![image](./cb8079cb-6b80-4815-b17e-f2046fe23d61.png)

两条路径：

- **ReduceScatter路径** (`QuantBMMReduceScatter`): 纯AIC
- **A2A路径** (`QuantBmmA2AVecReduceFP8HiF8`): AIC-AIV

**假设有多轮切分，其中一轮的过程如下：**
![image](./bdf9a052-ae25-4cfb-9618-e0ef4816c2fe.png)

![image](./fb343da3-188d-4874-8579-c5a667ac6717.png)

---

## 3. MXFP4继承架构设计

### 3.1 设计原则：最小侵入

**核心策略**：MXFP4与MXFP8在**量化语义**上完全相同（groupSize=[1,1,32], Scale=E8M0, Scale shape 3D），区别仅在**数据存储格式**（FP4打包 vs FP8独立）。因此：

1. MXFP4应**复用MXFP_MODE枚举值**，可通过新增`isMxfp4_`标志位区分存储差异
2. MXFP4应**共享MXFP8的全部校验逻辑**，仅在FP4特有约束处添加分支
3. MXFP4应**共享MXFP8的全部Tiling流程**，仅在地址偏移计算处适配packFactor=2
4. MXFP4应**共享MXFP8的通信归约路径**，ReduceScatter和A2A路径均不变

### 3.2 架构继承图

```
QuantBmmReduceScatterTiling (现有MXFP8 Tiling类)
    │
    ├── quantMode_: Mc2QuantMode::MXFP_MODE  ← 复用，不新增枚举
    ├── isMxfp4_: bool = false               ← 新增标志位
    │
    ├── SetScene()                           ← 扩展：FP4_E2M1+3D+E8M0 → MXFP_MODE + isMxfp4_=true
    ├── CheckInput()                         ← 扩展：MXFP4分支添加同类型+K偶数约束
    ├── CheckMxScaleDim()                    ← 复用：Scale shape规则完全相同
    ├── CheckGroupSize()                     ← 复用：[1,1,32]完全相同
    ├── DoOpTiling()                         ← 复用：公式化切分+isPerTensor=true
    ├── DoAdaptSlidWindowTiling()            ← 扩展：groupSizeK=32复用，inputParams新增packFactor
    ├── GetTilingKey()                       ← 扩展：需区分FP4 vs FP8的input dtype
    │
QuantBMMReduceScatter / QuantBmmA2AVecReduceFP8HiF8 (现有MXFP8 Kernel类)
    │
    ├── QuantBatchMatmul计算中aGM的偏移需要除以2

```

### 3.3 修改点清单（按层级）

#### L1: API层（`aclnn_matmul_reduce_scatter_v2.cpp`）

| 修改点 | 类型 | 说明 |
|--------|------|------|
| DTYPE_SUPPORT_LIST | 扩展 | 新增DT_FLOAT4_E2M1 |
| INPUT_SUPPORT_TYPE_LOW_ACCURACY | 扩展 | 新增DT_FLOAT4_E2M1 |
| CheckLowAccuracyCase | 扩展 | MXFP4(x1/x2均为FP4_E2M1) → 同类型校验+K偶数校验 |
| 算子定义(`matmul_reduce_scatter_v2_def.cpp`) | 扩展 | 新增FP4_E2M1组合3组 |

#### L2: Tiling层（`quant_bmm_reduce_scatter_tiling.cpp`）

| 修改点 | 类型 | 说明 |
|--------|------|------|
| MXFP8DTYPE_SUPPORT_LIST → MXFPXDTYPE_SUPPORT_LIST | 扩展 | 新增DT_FLOAT4_E2M1 |
| SetScene() | 扩展 | 3D+E8M0场景：若x1/x2为FP4_E2M1 → isMxfp4_=true |
| MxfpSceneParamCheck() | 扩展 | 改名为MxfpSceneParamCheck，支持FP4_E2M1；MXFP4时强制x1==x2 |
| QuantBmmReduceScatterTiling | 扩展 | 新增isMxfp4_成员变量 |
| CheckMxScaleDim | 扩展 | MXFP4时添加K偶数约束 |
| GetTilingKey() | 扩展 | 新增INPUT_TYPE_IS_FP4枚举值 |
| DoAdaptSlidWindowTiling | 扩展 | GetShapeAttrsInfo中设置packFactor相关参数 |

#### Kernel层（`matmul_reduce_scatter_v2_apt.cpp` + kernel headers）

| 修改点 | 类型 | 说明 |
|--------|------|------|
| PackFactor模板特化 | 新增 | `PackFactor<fp4x2_e2m1_t>::value = 2` |
| ComputeByteOffset/ComputeGMAddr | 新增 | 地址偏移计算辅助函数 |
| INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP4_REDUCE_SCATTER_OP_IMPL | 新增 | DTYPE_X1=fp4x2_e2m1_t的kernel入口宏 |
| APT入口 | 扩展 | 新增FP4_E2M1条件分支 |
| quant_bmm_reduce_scatter_fp8_hif8.h | 扩展 | Init/Process中地址偏移使用ComputeByteOffset |
| quant_bmm_a2a_vec_reduce_fp8_hif8.h | 扩展 | Init/Process中地址偏移使用ComputeByteOffset |
| Tiling Key | 扩展 | 新增TPL_INPUT的FP4枚举值 |

---

## 4. MXFP4的关键实现方案

### 4.1 host相关修改

#### aclnn层场景判定扩展

在 `CheckLowAccuracyCase`（:284-311）中扩展MXFP4判定：

```cpp
// 现有MXFP8判定（:299-301）
if ((x1Scale->GetViewShape().GetDimNum() == DIM_NUM_THREE) &&
    (x1Scale->GetDataType() == ge::DT_FLOAT8_E8M0)) {
    return CaseOption::LOW_ACCURACY_PER_TENSOR_WITH_QUANT_AMAX;
}

// MXFP4扩展：无需修改此判定逻辑
// MXFP4(x1=FP4_E2M1, x2=FP4_E2M1, Scale=3D+E8M0) 同样走此分支
// 因为CaseOption只区分精度等级，不区分具体数据格式
```

**关键修改**：在判定后增加MXFP4特化校验：

```cpp
// 新增：MXFP4时x1/x2必须同类型（参考matmul_allto_all :195-201）
if (x1->GetDataType() == op::DataType::DT_FLOAT4_E2M1 || 
    x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1) {
    OP_CHECK_DTYPE_NOT_SAME(x1, x2, return CaseOption::INVALID); // MXFP4强制同类型
}
```

#### Tiling层

在 `SetScene()`（:403-432）中，MXFP4与MXFP8走同一条判定路径：

```
x1Scale.dim=3 + x1Scale.dtype=DT_FLOAT8_E8M0 → MXFP_MODE
```

新增`isMxfp4_`标志位：

```cpp
// SetScene() 扩展
if (quantMode_ == mc2tiling::Mc2QuantMode::MXFP_MODE) {
    if (args_.geAType == ge::DataType::DT_FLOAT4_E2M1) {
        isMxfp4_ = true;  // 新增标志
    }
}
```

现有校验（:281-285）限制x1/x2∈{E4M3FN, E5M2}，需扩展支持FP4_E2M1：

```cpp
// 现有
OP_TILING_CHECK(!mc2tiling::CheckDataTypeVaild(args_.geAType, mc2tiling::MXFP8DTYPE_SUPPORT_LIST) || ...);

// 扩展为 MXFPXDTYPE_SUPPORT_LIST = {E4M3FN, E5M2, FP4_E2M1}
// MXFP4时额外约束：x1/x2必须同类型
if (isMxfp4_) {
    OP_TILING_CHECK(args_.geAType != args_.geBType, ...); // 强制同类型
}
```

MXFP4特有约束：K和ceilDiv(K,32)必须为偶数（参考matmul_allto_all :480-486）：

```cpp
// CheckMxScaleDim 扩展
if (isMxfp4_) {
    uint64_t kValue = x1Dim1; // K轴值
    uint64_t kDivGroupSize = CeilDiv(kValue, MXFP8_SIZE); // ceilDiv(K, 32)
    OP_TILING_CHECK((kValue % 2 != 0) || (kDivGroupSize % 2 != 0),
        OP_LOGE(..., "in mxfp4 quant mode, k and ceilDiv(k,32) must be even"),
        return ge::GRAPH_FAILED);
}
```

**原因**：fp4打包因子为2，每2个fp4元素组成1个存储单元。如果K为奇数，最后一个存储单元只包含1个fp4值，破坏了2:1打包比例。ceilDiv(K,32)为偶数确保Scale的E8M0对齐存储。

### 4.2 Kernel层APT入口扩展

参考mxfp8的入口，增加FP4_E2M1判断条件：

```cpp
#if (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_HIFLOAT8))) || \
     (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))) || \
     ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)))
```

### 4.7 Kernel层地址偏移适配

在现有kernel中，所有涉及A矩阵（x1）,B矩阵（x2）的GM地址计算需要考虑除以2；

```
### mc2_quant_batch_matmul_asw_block.h
在所有A,B偏移计算的地方，需要使得偏移减半
eg:uint64_t tmp1 = mc2MRest * blockBaseMDotRankK_ + params_.mSplitAddrOffset * rankK_/2(针对mxfp4);
```

**ReduceScatter路径**（`quant_bmm_reduce_scatter_fp8_hif8.h`）：
x1，x2偏移在MatMul内部通过`UpdateSlice`处理，需确保MatMul计算类正确处理fp4x2_e2m1_t的打包格式。

```
Mc2QuantBatchMatmulASWKernel::Init()          // mc2_quant_batch_matmul.h:81
  ├─ block_.Init(tilingData, blockIdx)         // 设置 mCnt, nCnt, totalCnt 等
  ├─ block_.InitForMC2(cfg, isTail, ...)       // 设置 rankK_, rankM_, headSliceM_, mSliceCnt_, blockBaseMDotRankK_
  └─ UpdateGlobalAddr(aGM, ...)                // 设置 aGlobal_ 指向 aGM

Mc2QuantBatchMatmulASWKernel::UpdateSlice(idx, isTail)  // mc2_quant_batch_matmul.h:156
  └─ block_.UpdateOffset<ScaleType>(idx, isTail)        // 设置 offsetsA_[], offsetsC_[]

Mc2QuantBatchMatmulASWKernel::Process(isLast)           // mc2_quant_batch_matmul.h:163
  └─ ComputeBasicOptiLoop(isLast)                       // mc2_quant_batch_matmul.h:177
       for each round:
         ├─ block_.UpdateBasicIndex(roundIdx, newBlockIdx)  // 设置 params_.mIndex
         ├─ block_.UpdateBlockParams(roundIdx, isLast)       // 设置 params_.mSplitAddrOffset
         ├─ block_.CalcGMOffset<...>()                       // 计算 offset_.offsetA
         └─ SetMMParaAndCompute()                            // 使用 aGlobal_[offset_.offsetA]
```

**A2A路径**（`quant_bmm_a2a_vec_reduce_fp8_hif8.h`）：

但A2A路径中x1矩阵的轮间偏移需要适配：

```cpp
// 现有（MXFP8）
currentAPtr = aGM_ + i * singleSliceM * rankK * sizeof(DTYPE_X1);

// MXFP4适配
currentAPtr = ComputeGMAddr<DTYPE_X1>(aGM_, i * singleSliceM * rankK);
```

Scale偏移不受影响（Scale仍为fp8_e8m0_t，packFactor=1）。

---## 5. Tiling策略设计及偏移实现

### 5.1 Tiling Key编码扩展

现有Tiling Key 8参数模板（`matmul_reduce_scatter_v2_apt_tiling_key.h`）：

| 参数 | 当前值域 | MXFP4扩展 |
|------|---------|----------|
| TPL_ISPERBLOCK | {0, 1} | 不变，MXFP4=0 |
| TPL_TRANSA | {0} | 不变 |
| TPL_TRANSB | {0, 1} | 不变，MXFP4强制=1 |
| TPL_INPUT | {0=FP8, 1=FP16_BF16} | **新增2=FP4** |
| TPL_OUTPUTDTYPE | {0=FP16_BF16, 2=FLOAT} | 不变 |
| TPL_SCALETYPE | {0=FP32, 1=E8M0} | 不变，MXFP4=1 |
| TPL_COMMALG | {0=RS, 1=A2A} | 不变 |
| TPL_COMM_MODE | {0=CCU, 1=AICPU} | 不变 |

**新增组合**（参考matmul_allto_all的MX_QUANT_MODE=2）：

```
MXFP4 RS路径:  isPerBlock=0, transA=0, transB=1, input=2(FP4), output=0/2, scaleType=1(E8M0), commAlg=0(RS), commMode=0/1
MXFP4 A2A路径: isPerBlock=0, transA=0, transB=1, input=2(FP4), output=0/2, scaleType=1(E8M0), commAlg=1(A2A), commMode=0
```

### 5.2 Tiling数据结构

MXFP4**复用** `QuantBatchMatmulV3ReduceScatterTilingData` 结构，无需新增结构体。仅在MatMul tiling参数中通过`isMxfp4_`标志影响以下字段：

```cpp
// DoAdaptSlidWindowTiling 中 GetShapeAttrsInfo 扩展
inputParams_.isPerTensor = true;        // 复用MXFP8逻辑
inputParams_.isPerBlock = false;
inputParams_.isDoubleScale = true;
inputParams_.groupSizeK = MXFP8_SIZE;   // = 32, 与MXFP8相同
inputParams_.groupSizeM = MXFP8_SIZE_M; // = 1
inputParams_.groupSizeN = MXFP8_SIZE_N; // = 1

// MXFP4特有：MatMul内部通过DTYPE_X1=fp4x2_e2m1_t自动处理打包
// 无需额外tiling参数传递packFactor
```

### 5.3 公式化切分策略

MXFP4**完全复用**MXFP8的公式化切分策略：

- `DoSplitMTiling` → 与MXFP8/Pertensor共享（isPerTensor=true）
- `GetTilingResult` → 4P/8P走A2A，其他走RS（与MXFP8相同决策）

**唯一差异**：MXFP4输入为4bit(fp4)，L1占用更小（仅为fp8的1/2），可考虑更激进的L1 Tile大小。但为最小侵入原则，初期**不做此优化**，完全复用MXFP8的切分参数。

### 5.4 workspace计算

MXFP4的workspace计算与MXFP8**完全相同**：

- CCU RS路径：`workspaceSize = libApiWorkSpaceSize + cToFloatLen × 1`
- A2A路径：`workspaceSize = cToFloatLen × 3`

原因：workspace存储的是反量化后的输出数据（fp16/bf16/float），不存储fp4原始数据。

### 5.5 HCCL通信配置

MXFP4的HCCL通信配置与MXFP8**完全相同**：

- 通信数据类型为输出dtype（fp16/bf16/float）
- 通信原语：RS路径用ReduceScatter，A2A路径用AlltoAll
- 通信引擎：CCU或AICPU

---

## 6. 修改文件清单与影响评估

### 6.1 必须修改的文件（仅参考）

| 文件 | 修改类型 | 影响范围 |
|------|---------|---------|
| `op_api/aclnn_matmul_reduce_scatter_v2.cpp` | 扩展dtype列表+MXFP4校验 | API层 |
| `op_host/matmul_reduce_scatter_v2_def.cpp` | 新增FP4_E2M1 dtype组合 | 算子定义 |
| `op_host/op_tiling/arch35/quant_bmm_reduce_scatter_tiling.cpp` | 新增isMxfp4_+扩展校验 | Tiling层核心 |
| `op_host/op_tiling/arch35/quant_bmm_reduce_scatter_tiling.h` | 新增isMxfp4_成员 | Tiling层声明 |
| `mc2/common/op_host/op_tiling/mc2_tiling_utils.h` | 扩展MXFP8DTYPE_SUPPORT_LIST→含FP4_E2M1 | 公共Tiling工具 |
| `op_kernel/arch35/matmul_reduce_scatter_v2_apt.cpp` | 新增FP4_E2M1 kernel分支 | Kernel入口 |
| `op_kernel/arch35/quant_bmm_reduce_scatter_fp8_hif8.h` | 地址偏移适配 | RS Kernel |
| `op_kernel/arch35/quant_bmm_a2a_vec_reduce_fp8_hif8.h` | 地址偏移适配 | A2A Kernel |
| `op_kernel/matmul_reduce_scatter_v2_apt_tiling_key.h` | 新增TPL_INPUT=FP4枚举 | Tiling Key |

### 6.2 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| FP4打包导致MatMul计算类不兼容 | 高 | 需确认MatMulASWKernel是否已支持fp4x2_e2m1_t输入 |
| K偶数约束遗漏 | 中 | 在CheckMxScaleDim中显式校验 |
| MXFP4误入PerBlock路径 | 低 | isMxfp4_=true时强制isPerTensor=true |
| dtype映射遗漏 | 低 | 算子定义中完整新增3组FP4组合 |
| 通信精度 | 低 | 通信传输反量化后数据，不受FP4影响 |

---

## 7. 实施优先级建议

### Phase 1: 基础规格打通（API + Def + Tiling校验）

1. 扩展dtype支持列表（API层 + Tiling公共工具）
2. 算子定义新增FP4_E2M1组合
3. Tiling层新增isMxfp4_标志 + MXFP4校验分支

### Phase 2: Kernel层打通

4. Tiling Key新增FP4枚举
5. APT入口新增FP4_E2M1条件分支
6. Kernel地址偏移适配

### Phase 3: 测试与验证

7. UT测试新增MXFP4用例
8. 精度验证（对比MXFP8的数值结果）
9. 性能基准测试


