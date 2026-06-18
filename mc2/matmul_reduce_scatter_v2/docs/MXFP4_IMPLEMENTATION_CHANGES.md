# MXFP4量化场景功能实现修改说明

## 概述

本文档记录了为 matmul_reduce_scatter_v2 算子添加 MXFP4 量化场景支持的所有修改内容。MXFP4 与 MXFP8 在量化语义上完全相同,主要差异在于数据存储格式(FP4打包 vs FP8独立)。

**设计依据**: 参考《MatmulReduceScatterMXFP4详细设计.md》

**参考实现**: 参考 matmul_allto_all 算子的 MXFP4 实现

---

## 1. API层修改

### 1.1 aclnn_matmul_reduce_scatter_v2.cpp

**修改内容**:
- 扩展 DTYPE_SUPPORT_LIST,新增 DT_FLOAT4_E2M1
- 扩展 INPUT_SUPPORT_TYPE_LOW_ACCURACY,新增 DT_FLOAT4_E2M1
- CheckLowAccuracyCase 函数增加 MXFP4 特化校验

**依据**: 详细设计书第3.3节 L1: API层修改

**关键代码**:

```cpp
// Line 124-127: 扩展dtype支持列表
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_BF16, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2,
    op::DataType::DT_HIFLOAT8, op::DataType::DT_FLOAT4_E2M1};  // 新增DT_FLOAT4_E2M1

// Line 131-133: 扩展低精度输入支持列表
static const std::initializer_list<op::DataType> INPUT_SUPPORT_TYPE_LOW_ACCURACY = {
    op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_HIFLOAT8,
    op::DataType::DT_FLOAT4_E2M1};  // 新增DT_FLOAT4_E2M1

// Line 284-311: CheckLowAccuracyCase 增加MXFP4校验
static enum CaseOption CheckLowAccuracyCase(const aclTensor* x1, const aclTensor* x2,
                                            const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    // ... 现有校验逻辑 ...
    
    // 新增：MXFP4场景校验（详细设计书1.2节）
    if (x1->GetDataType() == op::DataType::DT_FLOAT4_E2M1 ||
        x2->GetDataType() == op::DataType::DT_FLOAT4_E2M1) {
        OP_CHECK_DTYPE_NOT_SAME(x1, x2, return CaseOption::INVALID);
        // MXFP4场景：K和ceilDiv(K,32)必须为偶数
        int64_t kVal = x1->GetViewShape().GetDim(1);
        int64_t kDivGroupSize = ((kVal + 31) / 32);
        if ((kVal % 2 != 0) || (kDivGroupSize % 2 != 0)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(...);
            return CaseOption::INVALID;
        }
    }
    // ...
}
```

---

## 2. 算子定义修改

### 2.1 matmul_reduce_scatter_v2_def.cpp

**修改内容**:
- 新增 3 组 FP4_E2M1 dtype组合(组合29-31)
- 所有输入输出列表增加对应的dtype

**依据**: 详细设计书第1.3节 表

**新增dtype组合**:

| # | x1 dtype | x2 dtype | x1Scale dtype | x2Scale dtype | output dtype |
|---|----------|----------|---------------|---------------|-------------|
| 29 | FP4_E2M1 | FP4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | FP16 |
| 30 | FP4_E2M1 | FP4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | BF16 |
| 31 | FP4_E2M1 | FP4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT |

**关键代码**:

```cpp
// Line 23-31: x1输入扩展
.DataType({ge::DT_BF16, ge::DT_FLOAT16, ..., ge::DT_FLOAT8_E4M3FN,
           ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1})  // 新增3个FP4_E2M1

// Line 33-42: x2输入扩展(同上)
.DataType({... , ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1, ge::DT_FLOAT4_E2M1})

// Line 45-55: bias扩展(新增3个DT_FLOAT)
.DataType({... , ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})

// Line 57-67: x1_scale扩展(新增3个DT_FLOAT8_E8M0)
.DataType({... , ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0})

// Line 69-79: x2_scale扩展(同上)

// Line 85-95: output扩展(新增FP16/BF16/FLOAT)
.DataType({... , ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
```

---

## 3. Tiling层修改

### 3.1 mc2_tiling_utils.h (公共工具)

**修改内容**:
- 扩展 MXFP8DTYPE_SUPPORT_LIST → MXFPXDTYPE_SUPPORT_LIST
- 新增 DT_FLOAT4_E2M1

**依据**: 详细设计书第3.3节 L2: Tiling层修改

**关键代码**:

```cpp
// Line 130-132: 扩展MXFP dtype支持列表
constexpr std::initializer_list<ge::DataType> MXFPXDTYPE_SUPPORT_LIST = {
    ge::DataType::DT_FLOAT8_E4M3FN, ge::DataType::DT_FLOAT8_E5M2,
    ge::DataType::DT_FLOAT4_E2M1};  // 新增FP4_E2M1
```

### 3.2 quant_bmm_reduce_scatter_tiling.h

**修改内容**:
- 新增 `isMxfp4_` 成员变量(bool)

**依据**: 详细设计书第3.3节

**关键代码**:

```cpp
// Line 99: 新增成员变量
private:
    ...
    bool isMxfp4_{false};  // MXFP4场景标志位
```

### 3.3 quant_bmm_reduce_scatter_tiling.cpp

**修改内容**:
1. SetScene(): 识别 FP4_E2M1 场景,设置 `isMxfp4_=true`
2. MxfpSceneParamCheck(): 扩展支持 FP4_E2M1,添加强制 x1==x2 校验
3. CheckMxScaleDim(): 添加 K 和 ceilDiv(K,32) 必须为偶数的约束
4. GetTilingKey(): MXFP4场景设置 inputType=INPUT_TYPE_IS_FP4

**依据**: 详细设计书第4.1节

**关键代码**:

```cpp
// Line 403-432: SetScene() 扩展
void QuantBmmReduceScatterTiling::SetScene()
{
    quantMode_ = mc2tiling::Mc2QuantMode::INVALID_MODE;
    isMxfp4_ = false;  // 初始化为false
    // ... 现有判定逻辑 ...
    
    // MXFP4场景识别（详细设计书4.1节）
    if (quantMode_ == mc2tiling::Mc2QuantMode::MXFP_MODE) {
        if (args_.geAType == ge::DataType::DT_FLOAT4_E2M1) {
            isMxfp4_ = true;
            OP_LOGI(opName_, "MXFP4 quant scene detected: x1/x2 dtype is DT_FLOAT4_E2M1");
        }
    }
}

// Line 278-289: MxfpSceneParamCheck() 扩展
bool QuantBmmReduceScatterTiling::MxfpSceneParamCheck(...)
{
    // 扩展dtype校验列表
    OP_TILING_CHECK(!mc2tiling::CheckDataTypeVaild(args_.geAType, mc2tiling::MXFPXDTYPE_SUPPORT_LIST) ||
                    !mc2tiling::CheckDataTypeVaild(args_.geBType, mc2tiling::MXFPXDTYPE_SUPPORT_LIST), ...);
    
    // MXFP4强制同类型校验
    if (isMxfp4_) {
        OP_TILING_CHECK(args_.geAType != args_.geBType,
                        OP_LOGE(..., "In MXFP4 quant mode, the dtype of x1 and x2 must be the same (DT_FLOAT4_E2M1)"),
                        return false);
    }
}

// Line 166-213: CheckMxScaleDim() 扩展
ge::graphStatus QuantBmmReduceScatterTiling::CheckMxScaleDim(...) const
{
    // MXFP4特有约束：K和ceilDiv(K,32)必须为偶数（详细设计书1.2节）
    if (isMxfp4_) {
        uint64_t kValue = x1Dim1;
        uint64_t kDivGroupSize = ((kValue + MXFP8_SIZE_K - 1) / MXFP8_SIZE_K);
        OP_TILING_CHECK((kValue % 2 != 0) || (kDivGroupSize % 2 != 0),
            OP_LOGE(..., "In MXFP4 quant mode, K and ceilDiv(K,32) must be even"),
            return ge::GRAPH_FAILED);
    }
    // ... 现有校验逻辑 ...
}

// Line 487-519: GetTilingKey() 扩展
uint64_t QuantBmmReduceScatterTiling::GetTilingKey() const
{
    // MXFP4场景设置inputType为INPUT_TYPE_IS_FP4（详细设计书5.1节）
    uint8_t inputType = INPUT_TYPE_IS_FP8;
    if (isMxfp4_) {
        inputType = INPUT_TYPE_IS_FP4;
        OP_LOGD(opName_, "MXFP4 scene detected, setting inputType to INPUT_TYPE_IS_FP4 (2)");
    }
    
    uint64_t tilingKey = GET_TPL_TILING_KEY(
        isPerBlock, args_.isATrans, args_.isBTrans, inputType, outputType, scaleType, commAlg, commMode_);
    return tilingKey;
}
```

---

## 4. Kernel层修改

### 4.1 matmul_reduce_scatter_v2_apt_tiling_key.h

**修改内容**:
- 新增 INPUT_TYPE_IS_FP4 枚举值(值为2)
- 扩展 TPL_INPUT 参数值域
- 扩展 ASCENDC_TPL_SEL 组合

**依据**: 详细设计书第5.1节

**关键代码**:

```cpp
// Line 24-26: 新增INPUT_TYPE枚举
#define INPUT_TYPE_IS_FP8 0
#define INPUT_TYPE_IS_FP16_BF16 1
#define INPUT_TYPE_IS_FP4 2  // 新增FP4枚举

// Line 50-51: 扩展TPL_INPUT参数值域
ASCENDC_TPL_UINT_DECL(TPL_INPUT, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST, \
                        INPUT_TYPE_IS_FP8, INPUT_TYPE_IS_FP16_BF16, INPUT_TYPE_IS_FP4),

// Line 96-109: 扩展模板组合(scaletype=1时支持FP4)
ASCENDC_TPL_ARGS_SEL(
    ...
    ASCENDC_TPL_UINT_SEL(TPL_INPUT, ASCENDC_TPL_UI_LIST, INPUT_TYPE_IS_FP8, INPUT_TYPE_IS_FP4),
    ...
)
```

### 4.2 matmul_reduce_scatter_v2_apt.cpp

**修改内容**:
- 扩展 #if 条件判断,新增 FP4_E2M1 条件分支
- 新增 MXFP4 kernel入口宏调用

**依据**: 详细设计书第4.2节

**关键代码**:

```cpp
// Line 25-31: 扩展条件判断
#if (((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && ((ORIG_DTYPE_X1 == DT_HIFLOAT8))) || \
     (((ORIG_DTYPE_X1 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X1 == DT_FLOAT8_E5M2)) && \
      ((ORIG_DTYPE_X2 == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_X2 == DT_FLOAT8_E5M2))) || \
     ((ORIG_DTYPE_X1 == ORIG_DTYPE_X2) && (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)))  // 新增FP4条件

// Line 110-137: Kernel入口扩展
#elif (...)
    // MXFP4场景使用与MXFP8相同的实现路径（详细设计书4.2节）
    #if (ORIG_DTYPE_X1 == DT_FLOAT4_E2M1)
        if constexpr (TPL_SCALETYPE == TPL_X1_X2_DTYPE_IS_FP8E8M0) {
            if constexpr (TPL_COMMALG == TPL_CCU_ALL2ALL_VEC_REDUCE) {
                INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(QuantBmmA2AVecReduceFP8HiF8, false, TPL_TRANSB);
            } else {
                INVOKE_QUANT_BATCHMM_PERTENSOR_MXFP8_REDUCE_SCATTER_OP_IMPL(QuantBMMReduceScatter, false, TPL_TRANSB);
            }
        }
    #endif
```

### 4.3 quant_bmm_reduce_scatter_fp8_hif8.h（RS路径Kernel）

**修改内容**:
1. 新增 PackFactor 模板特化定义
2. InnerProcess() 尾块A矩阵地址偏移适配
3. MatMulComputReduceScatterPerblock() A矩阵偏移适配

**依据**: 详细设计书第4.7节；参考 MatmulAlltoAll 的 PackFactor 和地址计算方式

**关键代码**:

```cpp
// Line 33-47: 新增PackFactor定义
template <typename T>
struct PackFactor {
    static constexpr uint64_t value = 1UL; // 默认：每个元素包含1个标量（如fp8）
};

template <>
struct PackFactor<fp4x2_e2m1_t> {
    static constexpr uint64_t value = 2UL; // fp4x2：2个fp4打包成1个元素
};

// Line 192: InnerProcess()尾块A矩阵地址计算
// MXFP4场景适配：A矩阵地址偏移需除以packFactor（fp4为2，fp8为1）
auto aGMTail = aGM_ + aSize * sizeof(AType) / PackFactor<AType>::value * static_cast<uint64_t>(cfg.tileCnt);
// Scale矩阵地址偏移保持不变（fp8_e8m0 packFactor=1）
auto x1ScaleTail = x1ScaleGM_ + ... * sizeof(ScaleType) * static_cast<uint64_t>(cfg.tileCnt);

// Line 267-269: MatMulComputReduceScatterPerblock() A矩阵偏移
// MXFP4场景适配：A矩阵偏移需除以packFactor（fp4为2，fp8为1）
auto aOffset = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.Ka) * sizeof(AType) / PackFactor<AType>::value;
```

**修改说明**:
- PackFactor 定义放在 namespace MatmulReduceScatterV2Impl 开头（:32行之后）
- **只修改 A 矩阵地址偏移，Scale 保持不变**
- 原因：FP4 的 A 矩阵存储时 2个fp4打包成1个元素（sizeof(fp4x2_e2m1_t)包含2个fp4），而 Scale 仍为 fp8_e8m0（单元素存储）

---

### 4.4 quant_bmm_a2a_vec_reduce_fp8_hif8.h（A2A路径Kernel）

**修改内容**:
1. 新增 PackFactor 模板特化定义
2. InnerProcess() 尾块A矩阵地址偏移适配
3. MatMulComputReduceScatterPerblock() A矩阵Tile偏移适配

**依据**: 详细设计书第4.7节；参考 MatmulAlltoAll 的 PackFactor 和地址计算方式

**关键代码**:

```cpp
// Line 33-47: 新增PackFactor定义（与RS路径相同）
template <typename T>
struct PackFactor {
    static constexpr uint64_t value = 1UL;
};

template <>
struct PackFactor<fp4x2_e2m1_t> {
    static constexpr uint64_t value = 2UL;
};

// Line 285-291: InnerProcess()尾块A矩阵地址计算
// MXFP4场景适配：A矩阵地址偏移需除以packFactor（fp4为2，fp8为1）
auto aGMTail = aGM_ + aSize * sizeof(AType) / PackFactor<AType>::value * static_cast<uint64_t>(cfg.tileCnt);
// Scale矩阵地址偏移保持不变（fp8_e8m0 packFactor=1）
auto x1ScaleTail = x1ScaleGM_ + ... * sizeof(ScaleType) * static_cast<uint64_t>(cfg.tileCnt);

// Line 489-493: MatMulComputReduceScatterPerblock() A矩阵Tile偏移
// MXFP4场景适配：A矩阵单个Tile的字节偏移需除以packFactor（fp4为2，fp8为1）
const uint64_t aTileBytes = static_cast<uint64_t>(tiling.M) * static_cast<uint64_t>(tiling.Ka) * sizeof(AType) / PackFactor<AType>::value;
// Scale 矩阵单个Tile的字节偏移保持不变（fp8_e8m0 packFactor=1）
const uint64_t scaleTileBytes = scaleBlockM * scaleBlockK * sizeof(ScaleType);
```

**修改说明**:
- PackFactor 定义放在 namespace MatmulReduceScatterV2Impl 开头（:33行之后）
- **只修改 A 矩阵地址偏移，Scale 保持不变**
- A2A 路径中，A 矩阵有轮间偏移（for循环中的 currAPtr += aTileBytes），需要适配
- Scale 矩阵也有轮间偏移（currScalePtr += scaleTileBytes），但无需修改（packFactor=1）

---

## 5. 地址偏移适配原理说明

### 5.1 为什么需要除以 PackFactor

**核心原因**：FP4 的存储格式与 FP8 不同

| dtype | 存储方式 | sizeof() | packFactor | 地址偏移计算 |
|-------|---------|----------|-----------|-------------|
| fp8_e4m3fn | 单元素独立存储 | 1字节 | 1 | `elemCount * sizeof(fp8) / 1` |
| fp8_e5m2 | 单元素独立存储 | 1字节 | 1 | `elemCount * sizeof(fp8) / 1` |
| fp4x2_e2m1 | 2个fp4打包成1个元素 | 1字节（含2个fp4） | 2 | `elemCount * sizeof(fp4x2) / 2` |

**关键点**：
- `sizeof(fp4x2_e2m1_t)` = 1字节，但这个字节里打包了 **2个 fp4 标量**
- 当我们说"A矩阵有 M*K 个 fp4 元素"时，实际存储只需要 `M*K/2` 个字节
- 因此地址偏移公式：`字节偏移 = 元素数量 * sizeof(fp4x2) / 2`

**对比 FP8**：
- `sizeof(fp8)` = 1字节，包含1个fp8标量
- 地址偏移公式：`字节偏移 = 元素数量 * sizeof(fp8) / 1`

### 5.2 为什么 Scale 不需要修改

**原因**：MXFP4 场景中 Scale 仍为 fp8_e8m0，packFactor=1

- MXFP4 的量化语义：A矩阵(fp4) + Scale(fp8_e8m0) → 恢复为高精度值
- Scale 是 fp8_e8m0 类型，每个元素独立存储，sizeof(fp8_e8m0) = 1字节
- 因此 Scale 的地址偏移计算不变：`元素数量 * sizeof(fp8_e8m0)`

**MatmulAlltoAll 参考实现验证**：
```cpp
// matmul_allto_all/op_kernel/arch35/mx_quant_matmul_allto_all_arch35.h:160-161
pipeLineContext_.computationContext->additionalData.x1ScaleOffset = ComputeByteOffset<AscendC::fp8_e8m0_t>(
    (uint64_t)mc2Tiling_.tileM * CeilDiv(mc2Tiling_.rankK, MXFP_GROUP_SIZE) * NUM_TWO);
// 注意：这里明确使用 fp8_e8m0_t 类型调用 ComputeByteOffset，packFactor=1
```

### 5.3 参考实现方式

**MatmulAlltoAll 的实现方式**：

1. **PackFactor 模板特化**：
```cpp
// mx_quant_matmul_allto_all_arch35.h:24-32
template <typename T>
struct PackFactor {
    static constexpr uint64_t value = 1UL;
};

template <>
struct PackFactor<fp4x2_e2m1_t> {
    static constexpr uint64_t value = 2UL;
};
```

2. **封装辅助函数**（可选，RS/A2A 直接除以 PackFactor 更简洁）：
```cpp
// mx_quant_matmul_allto_all_arch35.h:39-42
template <typename T>
__aicore__ inline uint64_t ComputeByteOffset(uint64_t elementCount)
{
    return elementCount * sizeof(T) / PackFactor<T>::value;
}
```

3. **实际应用**：
```cpp
// mx_quant_matmul_allto_all_arch35.h:153-154
pipeLineContext_.computationContext->baseData.aOffset =
    ComputeByteOffset<DTYPE_X1>((uint64_t)mc2Tiling_.tileM * mc2Tiling_.rankK);
// x1为fp4x2时，自动除以2；为fp8时，除以1
```

---

## 6. 未完成项（需后续实现）

---

### 6.1 UT测试用例设计

**建议新增的UT测试用例**:

1. **MXFP4 dtype组合测试**:
   - 测试组合29: FP4_E2M1 + FP4_E2M1 + E8M0 + FP16
   - 测试组合30: FP4_E2M1 + FP4_E2M1 + E8M0 + BF16
   - 测试组合31: FP4_E2M1 + FP4_E2M1 + E8M0 + FLOAT

2. **K偶数约束测试**:
   - 正常场景: K=256 (偶数), ceilDiv(K,32)=8 (偶数)
   - 异常场景: K=255 (奇数), 预期校验失败
   - 异常场景: K=64 (偶数), ceilDiv(64,32)=2 (偶数) ✓
   - 异常场景: K=96 (偶数), ceilDiv(96,32)=3 (奇数), 预期校验失败

3. **类型约束测试**:
   - 正常场景: x1=FP4_E2M1, x2=FP4_E2M1
   - 异常场景: x1=FP4_E2M1, x2=FP8_E4M3FN, 预期校验失败

**依据**: 详细设计书第1.2节和第7节Phase 3

---

## 7. 实施总结

### 7.1 完成的修改项

✅ **API层**:
- 扩展 DTYPE_SUPPORT_LIST 和 INPUT_SUPPORT_TYPE_LOW_ACCURACY
- CheckLowAccuracyCase 增加 MXFP4 特化校验

✅ **算子定义**:
- 新增 3 组 FP4_E2M1 dtype组合

✅ **Tiling层**:
- 新增 isMxfp4_ 成员变量
- SetScene() 识别 FP4_E2M1 场景
- MxfpSceneParamCheck() 扩展 dtype校验和强制同类型校验
- CheckMxScaleDim() 添加 K偶数约束
- GetTilingKey() 设置正确的 inputType
- 扩展 MXFPXDTYPE_SUPPORT_LIST

✅ **Kernel层**:
- **Tiling Key**: 新增 INPUT_TYPE_IS_FP4
- **APT入口**: 新增 FP4_E2M1 条件分支
- **地址偏移适配**: RS路径和A2A路径全部完成
  - 新增 PackFactor 模板特化定义
  - InnerProcess() 尾块A矩阵地址偏移适配
  - MatMulComputReduceScatterPerblock() A矩阵偏移适配
  - Scale 地址偏移保持不变（符合预期）

### 7.2 待完成项

⚠️ **UT测试**: 需要编写具体的测试用例代码
- MXFP4 dtype组合测试
- K偶数约束测试
- 类型约束测试

⚠️ **编译验证**: 需要实际编译并验证无语法错误

⚠️ **精度验证**: 需要在 NPU 上实际运行并对比 MXFP8 的数值结果

### 7.3 遇到的问题及解决

**问题1**: MXFP4与MXFP8的差异点识别
- **解决**: 通过阅读详细设计书和参考 matmul_allto_all 实现,明确了差异点在于packFactor=2和K偶数约束

**问题2**: Tiling Key 的扩展方式
- **解决**: 参考 matmul_allto_all 的 Tiling Key 定义,新增 INPUT_TYPE_IS_FP4 枚举值

**问题3**: MXFP4与MXFP8的代码复用策略
- **解决**: 采用最小侵入原则,复用 MXFP8 的校验逻辑和 Tiling 流程,仅在必要处添加分支判断

**问题4**: 地址偏移适配的核心原理理解
- **解决**: 通过分析 PackFactor 定义和参考实现,理解了 FP4 打包存储的本质：sizeof(fp4x2)包含2个fp4标量,因此地址偏移需除以2

**问题5**: Scale 是否需要修改的判断
- **解决**: 通过阅读 MatmulAlltoAll 的实现（明确使用 fp8_e8m0_t 类型调用 ComputeByteOffset）和 MXFP4 量化语义,确认 Scale 保持 fp8_e8m0,packFactor=1,无需修改

### 7.4 验证结果

**当前状态**: Phase 1、Phase 2 和 Kernel层地址偏移适配全部完成,Phase 3 UT测试待实现。

**修改文件清单**:
1. `aclnn_matmul_reduce_scatter_v2.cpp` - API层扩展
2. `matmul_reduce_scatter_v2_def.cpp` - 算子定义扩展
3. `mc2_tiling_utils.h` - Tiling工具扩展
4. `quant_bmm_reduce_scatter_tiling.h/cpp` - Tiling逻辑扩展
5. `matmul_reduce_scatter_v2_apt_tiling_key.h` - Tiling Key扩展
6. `matmul_reduce_scatter_v2_apt.cpp` - Kernel入口扩展
7. `quant_bmm_reduce_scatter_fp8_hif8.h` - RS路径地址偏移适配 ✅
8. `quant_bmm_a2a_vec_reduce_fp8_hif8.h` - A2A路径地址偏移适配 ✅

**下一步建议**:
1. 编写并执行 UT 测试用例
2. 在 NPU 上实际编译和运行验证
3. 对比 MXFP4 与 MXFP8 的数值结果,确保精度符合预期
4. 验证不同 shape 下的地址偏移计算正确性（特别是尾块场景）

---

## 8. 参考文档

- 《MatmulReduceScatterMXFP4详细设计.md》
- matmul_allto_all 算子 MXFP4 实现:
  - `mc2/matmul_allto_all/op_host/matmul_allto_all_def.cpp`
  - `mc2/matmul_allto_all/op_host/op_tiling/arch35/mx_quant_matmul_allto_all_tiling_base.cpp`
  - `mc2/matmul_allto_all/op_kernel/arch35/matmul_allto_all_apt.cpp`
  - `mc2/matmul_allto_all/op_kernel/arch35/mx_quant_matmul_allto_all_arch35.h` **（地址偏移适配核心参考）**

---

**修改日期**: 2026-06-11
**修改人**: CANNBot
**版本**: v1.1（新增Kernel层地址偏移适配）