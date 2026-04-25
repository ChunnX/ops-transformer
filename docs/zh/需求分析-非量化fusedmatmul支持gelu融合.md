# 需求分析：非量化 fusedmatmul支持 mm加gelu（Erf和Tanh）融合（基础模板）

## 一、需求拆解

### 1.1 关键词解析

| 关键词 | 含义 | 具体说明 |
|-------|------|---------|
| **非量化** | 不使用量化数据类型 | 数据类型为 FLOAT16/BFLOAT16/FLOAT32，不支持INT8/INT4等量化类型 |
| **fusedmatmul** | 融合矩阵乘法算子 | 指 `grouped_matmul` 算子（分组矩阵乘） |
| **mm** | Matmul | 标准矩阵乘法：[M,K] × [K,N] = [M,N] |
| **gelu融合** | 激活函数融合 | 在Matmul计算后直接应用GELU激活函数，减少中间数据搬运 |
| **Erf** | Error Function | GELU的一种精确实现：$GELU(x) = 0.5x(1+erf(x/\sqrt{2}))$ |
| **Tanh** | Hyperbolic Tangent | GELU的一种近似实现：$GELU(x) \approx 0.5x(1+tanh[\sqrt{2/\pi}(x+0.044715x^3)])$ |
| **基础模板** | 基础Kernel模板 | 为新功能添加Ascend C Kernel模板定义，支持编译和执行 |

### 1.2 需求目标

**核心目标**：为 `grouped_matmul` 算子的非量化场景，扩展支持 GELU 激活函数融合：
1. **支持 GELU-Erf 实现** - actType=3
2. **支持 GELU-Tanh 实现** - actType=2
3. **添加基础模板** - 在 `grouped_matmul_tiling_key.h` 中新增模板定义

---

## 二、当前实现分析

### 2.1 grouped_matmul 算子 actType 参数

#### actType 定义（当前支持）

```cpp
// gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md 第298-304行
actType（int64_t）取值范围0-5：
  0：GMM_ACT_TYPE_NONE      // 无激活函数
  1：GMM_ACT_TYPE_RELU      // ReLU激活
  2：GMM_ACT_TYPE_GELU_TANH // GELU-Tanh近似
  3：GMM_ACT_TYPE_GELU_ERR_FUNC // GELU-Erf精确
  4：GMM_ACT_TYPE_FAST_GELU // FastGELU
  5：GMM_ACT_TYPE_SILU      // SiLU/Swish激活
```

#### 当前约束（关键问题）

**非量化场景约束**（第741行）：
```
支持GroupType=-1、0、2，actType=0，groupListType=0/1
```

**Ascend950约束**（第799行）：
```
在伪量化和非量化场景下，actType仅支持0
```

**结论**：
- ✓ actType参数已定义GELU类型（actType=2/3）
- ✗ **非量化场景只支持actType=0**（无激活函数）
- ✗ **需要扩展非量化场景支持actType=2/3**

### 2.2 GELU 激活函数两种实现

#### GELU-Erf 精确实现（actType=3）

**公式**：
```
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

**特点**：
- 精确度高，误差<1e-6
- 计算复杂，涉及erf函数调用
- 适用于精度敏感场景（如训练）

**实现难点**：
- Ascend C需要支持erf函数
- Vector单元计算erf的效率

#### GELU-Tanh 近似实现（actType=2）

**公式**：
```
GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
```

**特点**：
- 近似实现，误差约1e-3
- 计算简单，仅需tanh和多项式
- 性能优于Erf版本
- 适用于推理场景

**实现要点**：
- Ascend C支持tanh函数
- 多项式计算优化

### 2.3 融合计算流程对比

#### 当前流程（无融合）

```
1. grouped_matmul计算
   y = x × weight + bias
   
2. 拷贝y到activation算子输入
   
3. activation计算（单独算子）
   y_activated = gelu(y)
```

**缺点**：
- 两次算子调用，增加调度开销
- 中间数据y需要Global Memory搬运
- 总耗时 = Matmul时间 + 搬运时间 + GELU时间

#### 融合流程（需求目标）

```
1. grouped_matmul + gelu融合计算
   y_activated = gelu(x × weight + bias)
   （直接在L0C/L1缓冲区应用GELU）
```

**优点**：
- 单次算子调用，减少调度开销
- 无中间数据搬运，节省Global Memory带宽
- 总耗时 ≈ Matmul时间（GELU在Vector单元并行执行）
- 性能提升预期：10-20%

### 2.4 量化场景的GELU支持情况

#### 量化场景actType支持

| 场景 | actType支持 | 说明 |
|------|-------------|------|
| **全量化-A8W8** | 1/2/3/4/5 | ✓ 支持GELU |
| **全量化-A4W4** | 0 | ✗ 不支持激活 |
| **伪量化-A8W4** | 0 | ✗ 不支持激活 |
| **伪量化-A16W8** | 0 | ✗ 不支持激活 |
| **伪量化-A16W4** | 0 | ✗ 不支持激活 |
| **非量化** | 0 | ✗ **不支持激活（需扩展）** |

**结论**：
- 量化场景已支持GELU融合（actType=2/3）
- **非量化场景缺失GELU支持（需求目标）**

---

## 三、模板分析

### 3.1 当前非量化模板定义

#### 模板宏定义（grouped_matmul_tiling_key.h 第92-104行）

```cpp
#define SET_GMM_NO_QUANT_TPL_ARGS(kernel_type, dtype, trans_a, trans_b, aiv_aic_ratio)
    ASCENDC_TPL_KERNEL_TYPE_SEL(kernel_type),      // kernel类型
    ASCENDC_TPL_DTYPE_SEL(D_T_A, dtype),           // 输入x数据类型
    ASCENDC_TPL_DTYPE_SEL(D_T_B, dtype),           // 输入weight数据类型
    ASCENDC_TPL_DTYPE_SEL(D_T_Y, dtype),           // 输出数据类型
    ASCENDC_TPL_BOOL_SEL(TRANS_A, trans_a),        // x是否转置
    ASCENDC_TPL_BOOL_SEL(TRANS_B, trans_b),        // weight是否转置
    ASCENDC_TPL_UINT_SEL(GROUP_LIST_TYPE, ...),    // 分组列表类型
    ASCENDC_TPL_BOOL_SEL(IS_STATIC_TILING_API, 0), // Tiling API类型
    ASCENDC_TPL_UINT_SEL(A8W4_KERNEL_TEMPLATE, NONE), // 量化模板（非量化为NONE）
    ASCENDC_TPL_UINT_SEL(A16W8_KERNEL_TEMPLATE, NONE), // 量化模板（非量化为NONE）
    ASCENDC_TPL_UINT_SEL(AIV_AIC_RATIO, ...),      // Vector:Core比例
    ASCENDC_TPL_BOOL_SEL(IS_ENABLE_FIXED_AXIS, 0)  // 固定轴算法
```

#### 当前模板组合（GMM_FLOAT场景，第342-366行）

**Ascend910B/A3（__CCE_AICORE__ == 220）**：
```cpp
// 非量化场景，actType=0（无激活）
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AIC_1_0, BF16, 0, 0, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AIC_1_0, BF16, 0, 1, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AIC_1_0, FLOAT16, 0, 0, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AIC_1_0, FLOAT16, 0, 1, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AIC_1_0, FLOAT, 0, 0, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AIC_1_0, FLOAT, 0, 1, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AIC_1_1, BF16, 1, 0, AIV_AIC_RATIO_1)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AIC_1_1, FLOAT16, 1, 0, AIV_AIC_RATIO_1)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AIC_1_1, FLOAT, 1, 0, AIV_AIC_RATIO_1)),
```

**Ascend950（__CCE_AICORE__ == 200）**：
```cpp
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AICORE, BF16, 0, 0, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AICORE, BF16, 0, 1, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AICORE, FLOAT16, 0, 0, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AICORE, FLOAT16, 0, 1, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AICORE, FLOAT, 0, 0, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AICORE, FLOAT, 0, 1, CUBE_ONLY)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(MIX_VECTOR_CORE, BF16, 1, 0, AIV_AIC_RATIO_1)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(MIX_VECTOR_CORE, FLOAT16, 1, 0, AIV_AIC_RATIO_1)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(MIX_VECTOR_CORE, FLOAT, 1, 0, AIV_AIC_RATIO_1)),
```

### 3.2 缺失模板分析

#### 缺失的激活函数模板参数

**当前问题**：
- 模板宏中没有 `ACT_TYPE` 参数
- 所有非量化模板都默认actType=0（无激活）
- 需要新增 `ACT_TYPE` 模板参数支持actType=2/3

#### 需新增的模板组合矩阵

| dtype | kernel_type | aiv_aic_ratio | actType | 数量 |
|-------|-------------|---------------|---------|------|
| BF16 | AIC_1_1 | AIV_AIC_RATIO_1 | 2（Tanh） | 1 |
| BF16 | AIC_1_1 | AIV_AIC_RATIO_1 | 3（Erf） | 1 |
| FLOAT16 | AIC_1_1 | AIV_AIC_RATIO_1 | 2（Tanh） | 1 |
| FLOAT16 | AIC_1_1 | AIV_AIC_RATIO_1 | 3（Erf） | 1 |
| FLOAT | AIC_1_1 | AIV_AIC_RATIO_1 | 2（Tanh） | 1 |
| FLOAT | AIC_1_1 | AIV_AIC_RATIO_1 | 3（Erf） | 1 |
| **Ascend910B合计** | - | - | - | **6** |
| BF16 | MIX_VECTOR_CORE | AIV_AIC_RATIO_1 | 2（Tanh） | 1 |
| BF16 | MIX_VECTOR_CORE | AIV_AIC_RATIO_1 | 3（Erf） | 1 |
| FLOAT16 | MIX_VECTOR_CORE | AIV_AIC_RATIO_1 | 2（Tanh） | 1 |
| FLOAT16 | MIX_VECTOR_CORE | AIV_AIC_RATIO_1 | 3（Erf） | 1 |
| FLOAT | MIX_VECTOR_CORE | AIV_AIC_RATIO_1 | 2（Tanh） | 1 |
| FLOAT | MIX_VECTOR_CORE | AIV_AIC_RATIO_1 | 3（Erf） | 1 |
| **Ascend950合计** | - | - | - | **6** |
| **总模板数** | - | - | - | **12** |

---

## 四、具体开发任务清单

### 4.1 模板扩展任务

#### 任务1：模板宏扩展（新增ACT_TYPE参数）

**修改文件**：
- `gmm/grouped_matmul/op_kernel/grouped_matmul_tiling_key.h`

**修改内容**：

```cpp
// 第120-174行，ASCENDC_TPL_ARGS_DECL中新增ACT_TYPE声明
ASCENDC_TPL_ARGS_DECL(
    GroupedMatmul,
    ...
    // 新增：激活函数类型
    ASCENDC_TPL_UINT_DECL(
        ACT_TYPE, ASCENDC_TPL_3_BW, ASCENDC_TPL_UI_LIST,
        GMM_ACT_TYPE_NONE,       // 0: 无激活
        GMM_ACT_TYPE_GELU_TANH,  // 2: GELU-Tanh
        GMM_ACT_TYPE_GELU_ERR    // 3: GELU-Erf
    ),
    ...
);

// 第92-104行，SET_GMM_NO_QUANT_TPL_ARGS宏新增ACT_TYPE参数
#define SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(kernel_type, dtype, trans_a, trans_b, aiv_aic_ratio, act_type)
    ASCENDC_TPL_KERNEL_TYPE_SEL(kernel_type),
    ASCENDC_TPL_DTYPE_SEL(D_T_A, dtype),
    ASCENDC_TPL_DTYPE_SEL(D_T_B, dtype),
    ASCENDC_TPL_DTYPE_SEL(D_T_Y, dtype),
    ASCENDC_TPL_BOOL_SEL(TRANS_A, trans_a),
    ASCENDC_TPL_BOOL_SEL(TRANS_B, trans_b),
    ASCENDC_TPL_UINT_SEL(GROUP_LIST_TYPE, ...),
    ASCENDC_TPL_BOOL_SEL(IS_STATIC_TILING_API, 0),
    ASCENDC_TPL_UINT_SEL(A8W4_KERNEL_TEMPLATE, NONE),
    ASCENDC_TPL_UINT_SEL(A16W8_KERNEL_TEMPLATE, NONE),
    ASCENDC_TPL_UINT_SEL(AIV_AIC_RATIO, aiv_aic_ratio),
    ASCENDC_TPL_UINT_SEL(ACT_TYPE, act_type),  // 新增
    ASCENDC_TPL_BOOL_SEL(IS_ENABLE_FIXED_AXIS, 0)
```

**工作量**：约1天

#### 任务2：新增GELU模板组合

**修改文件**：
- `gmm/grouped_matmul/op_kernel/grouped_matmul_tiling_key.h`

**新增模板**（GMM_FLOAT场景，第342行后插入）：

```cpp
#elif defined(GMM_FLOAT)
    // NO_QUANT（原有）
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AIC_1_0, BF16, 0, 0, CUBE_ONLY)),
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(AIC_1_0, BF16, 0, 1, CUBE_ONLY)),
    ... // 原有模板
    
    // NO_QUANT_WITH_GELU（新增）
    // Ascend910B/A3: GELU-Tanh
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(AIC_1_1, BF16, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_TANH)),
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(AIC_1_1, FLOAT16, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_TANH)),
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(AIC_1_1, FLOAT, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_TANH)),
    
    // Ascend910B/A3: GELU-Erf
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(AIC_1_1, BF16, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_ERR)),
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(AIC_1_1, FLOAT16, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_ERR)),
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(AIC_1_1, FLOAT, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_ERR)),
#endif

// Ascend950（第356行后插入）
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 200
#if defined(GMM_FLOAT)
    // NO_QUANT（原有）
    ...
    
    // NO_QUANT_WITH_GELU（新增）
    // GELU-Tanh
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(MIX_VECTOR_CORE, BF16, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_TANH)),
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(MIX_VECTOR_CORE, FLOAT16, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_TANH)),
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(MIX_VECTOR_CORE, FLOAT, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_TANH)),
    
    // GELU-Erf
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(MIX_VECTOR_CORE, BF16, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_ERR)),
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(MIX_VECTOR_CORE, FLOAT16, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_ERR)),
    ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS_WITH_ACT(MIX_VECTOR_CORE, FLOAT, 1, 0, AIV_AIC_RATIO_1, GMM_ACT_TYPE_GELU_ERR)),
#endif
#endif
```

**工作量**：约0.5天

### 4.2 Kernel实现任务

#### 任务3：GELU-Tanh Kernel实现

**修改文件**：
- `gmm/grouped_matmul/op_kernel/grouped_matmul.h` 新增GELU计算函数
- `gmm/grouped_matmul/op_kernel/grouped_matmul.cpp` 新增actType分支

**新增GELU-Tanh计算函数**：

```cpp
// grouped_matmul.h新增
template<typename T>
__aicore__ inline void ComputeGeluTanh(const LocalTensor<T>& src, const LocalTensor<T>& dst, uint32_t elementNum) {
    // Tanh近似公式：GELU(x) ≈ 0.5*x*(1+tanh(sqrt(2/π)*(x+0.044715*x^3)))
    
    // 1. 计算x^3
    LocalTensor<T> x3 = src;  // UB临时空间
    Mul(x3, src, src, elementNum);  // x^2
    Mul(x3, x3, src, elementNum);   // x^3
    
    // 2. 计算 0.044715 * x^3
    T coeff = static_cast<T>(0.044715);
    LocalTensor<T> tmp;
    Muls(tmp, x3, coeff, elementNum);
    
    // 3. 计算 x + 0.044715*x^3
    Add(tmp, src, tmp, elementNum);
    
    // 4. 计算 sqrt(2/π) * (x + 0.044715*x^3)
    T sqrtCoeff = static_cast<T>(0.7978845608);  // sqrt(2/π)
    Muls(tmp, tmp, sqrtCoeff, elementNum);
    
    // 5. 计算 tanh(...)
    Tanh(dst, tmp, elementNum);
    
    // 6. 计算 1 + tanh(...)
    T one = static_cast<T>(1.0);
    Adds(dst, dst, one, elementNum);
    
    // 7. 计算 0.5 * x * (1+tanh(...))
    Mul(dst, src, dst, elementNum);
    T half = static_cast<T>(0.5);
    Muls(dst, dst, half, elementNum);
}
```

**新增actType分支**：

```cpp
// grouped_matmul.cpp Process函数中新增actType判断
template<typename T>
__aicore__ void GroupedMatmul<T>::Process() {
    // ... Matmul计算
    
    // 根据actType应用激活函数
    if constexpr (ACT_TYPE == GMM_ACT_TYPE_GELU_TANH) {
        ComputeGeluTanh<T>(mmOutLocal, actOutLocal, elementNum);
    } else if constexpr (ACT_TYPE == GMM_ACT_TYPE_GELU_ERR) {
        ComputeGeluErf<T>(mmOutLocal, actOutLocal, elementNum);
    } else {
        // ACT_TYPE == GMM_ACT_TYPE_NONE: 无激活，直接拷贝输出
        DataCopy(mmOutGlobal, mmOutLocal, elementNum);
    }
}
```

**工作量**：约1天

#### 任务4：GELU-Erf Kernel实现

**新增GELU-Erf计算函数**：

```cpp
// grouped_matmul.h新增
template<typename T>
__aicore__ inline void ComputeGeluErf(const LocalTensor<T>& src, const LocalTensor<T>& dst, uint32_t elementNum) {
    // Erf精确公式：GELU(x) = 0.5*x*(1+erf(x/sqrt(2)))
    
    // 1. 计算 x / sqrt(2)
    T sqrt2 = static_cast<T>(1.414213562);  // sqrt(2)
    LocalTensor<T> tmp;
    Divs(tmp, src, sqrt2, elementNum);
    
    // 2. 计算 erf(x / sqrt(2))
    Erf(dst, tmp, elementNum);  // Ascend C erf API
    
    // 3. 计算 1 + erf(...)
    T one = static_cast<T>(1.0);
    Adds(dst, dst, one, elementNum);
    
    // 4. 计算 0.5 * x * (1+erf(...))
    Mul(dst, src, dst, elementNum);
    T half = static_cast<T>(0.5);
    Muls(dst, dst, half, elementNum);
}
```

**关键依赖**：
- Ascend C Vector API中的 `Erf` 函数支持
- 需确认 Ascend C SDK是否提供erf实现

**工作量**：约1天

#### 任务5：Tiling策略适配

**修改文件**：
- `gmm/grouped_matmul/op_host/grouped_matmul_tiling.cpp`

**新增actType Tiling逻辑**：

```cpp
// grouped_matmul_tiling.cpp SetTilingKey函数新增actType分支
void GroupedMatmulTiling::SetTilingKey() {
    // ... 其他Tiling参数
    
    // actType设置
    if (actType_ == 0) {
        tilingData.set_actType(GMM_ACT_TYPE_NONE);
    } else if (actType_ == 2) {
        tilingData.set_actType(GMM_ACT_TYPE_GELU_TANH);
        // 需调整UB空间分配：为GELU计算预留额外UB
        ubSizeForActivation_ = elementNum * sizeof(DTYPE_X);
    } else if (actType_ == 3) {
        tilingData.set_actType(GMM_ACT_TYPE_GELU_ERR);
        ubSizeForActivation_ = elementNum * sizeof(DTYPE_X);
    }
}
```

**UB空间重新分配**：

```cpp
// ComputeTiling函数新增UB分配逻辑
void GroupedMatmulTiling::ComputeTiling() {
    // 总UB空间
    uint32_t totalUbSize = 64 * 1024;  // 64KB
    
    // Matmul占用UB
    uint32_t ubForMatmul = ...;  // 根据M/N/K计算
    
    // 激活函数占用UB（新增）
    uint32_t ubForActivation = 0;
    if (actType_ != 0) {
        ubForActivation = baseN * sizeof(DTYPE_X) * 4;  // 为GELU预留4倍baseN空间
    }
    
    // 校验UB是否足够
    if (ubForMatmul + ubForActivation > totalUbSize) {
        // 调整baseN/baseM，减小单次计算块大小
        AdjustBlockSize(ubForMatmul + ubForActivation, totalUbSize);
    }
}
```

**工作量**：约1天

### 4.3 Host侧任务

#### 任务6：actType参数传递

**修改文件**：
- `gmm/grouped_matmul/op_host/grouped_matmul_def.cpp`

**新增actType属性定义**：

```cpp
// grouped_matmul_def.cpp
class FlashAttentionScore : public OpDef {
public:
    explicit FlashAttentionScore(const char* name) : OpDef(name) {
        // ... 输入输出定义
        
        // 新增actType属性
        this->Attr("actType")
            .AttrType(REQUIRED_INT)
            .DefaultValue(0)
            .ValidateRange(0, 5);  // 0-5
    }
};
```

**工作量**：约0.5天

#### 任务7：Shape推导适配

**修改文件**：
- `gmm/grouped_matmul/op_host/grouped_matmul_infershape.cpp`

**逻辑**（actType不影响输出Shape）：
```cpp
// GELU激活不改变Shape，沿用原逻辑
ge::graphStatus InferShape(gert::TilingContext* context) {
    // actType对Shape无影响，输出Shape与Matmul输出一致
    auto outShape = context->GetOutputShape(0);
    outShape = xShape;  // [M,N]
    
    // 无需特殊处理actType
    return ge::GRAPH_SUCCESS;
}
```

**工作量**：约0.5天

### 4.4 API接口任务

#### 任务8：aclnn接口文档更新

**修改文件**：
- `gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md`

**更新内容**：

1. **非量化场景约束更新**（第741行）：
```markdown
除公共约束外，非量化场景其余约束如下：
- 支持GroupType=-1、0、2
- **支持actType=0/2/3**（新增：2=GELU-Tanh, 3=GELU-Erf）
- 支持groupListType=0/1
```

2. **数据类型表格更新**（第732行）：
```markdown
| x | weight | bias | actType支持 |
| FLOAT | FLOAT | FLOAT/null | 0/2/3 |
| FLOAT16 | FLOAT16 | FLOAT16/null | 0/2/3 |
| BFLOAT16 | BFLOAT16 | FLOAT/null | 0/2/3 |
```

3. **新增GELU计算公式说明**：
```markdown
- **非量化场景 + GELU激活**：

  - actType=2（GELU-Tanh近似）：
    $$
    y_i = 0.5 * (x_i \times weight_i + bias_i) * (1 + tanh(\sqrt{2/\pi} * ((x_i \times weight_i + bias_i) + 0.044715 * (x_i \times weight_i + bias_i)^3)))
    $$

  - actType=3（GELU-Erf精确）：
    $$
    y_i = 0.5 * (x_i \times weight_i + bias_i) * (1 + erf((x_i \times weight_i + bias_i) / \sqrt{2}))
    $$
```

**工作量**：约0.5天

#### 任务9：调用示例更新

**修改文件**：
- `gmm/grouped_matmul/examples/test_aclnn_grouped_matmul.cpp`

**新增GELU调用示例**：

```cpp
// 示例1：非量化 + GELU-Tanh
int64_t actType = 2;  // GELU-Tanh
aclnnGroupedMatmulV4GetWorkspaceSize(
    x, weight, bias, scale, offset, antiquantScale, antiquantOffset, perTokenScale,
    groupList, activationInput, activationQuantScale, activationQuantOffset,
    splitItem, groupType, groupListType, actType,  // actType=2
    out, activationFeatureOut, dynQuantScaleOut,
    &workspaceSize, &executor);

// 示例2：非量化 + GELU-Erf
int64_t actType = 3;  // GELU-Erf
aclnnGroupedMatmulV4GetWorkspaceSize(..., actType, ...);
```

**工作量**：约0.5天

### 4.5 测试任务

#### 任务10：UT测试用例

**新增文件**：
- `gmm/grouped_matmul/tests/ut/op_host/test_grouped_matmul_gelu.cpp`

**测试覆盖**：
```cpp
// 测试用例设计
TEST_F(GroupedMatmulGelu, test_no_quant_gelu_tanh_bf16) {
    // 输入：BFLOAT16, actType=2
    TestGroupedMatmulWithGelu(BF16, GMM_ACT_TYPE_GELU_TANH);
}

TEST_F(GroupedMatmulGelu, test_no_quant_gelu_err_float16) {
    // 输入：FLOAT16, actType=3
    TestGroupedMatmulWithGelu(FLOAT16, GMM_ACT_TYPE_GELU_ERR);
}

TEST_F(GroupedMatmulGelu, test_no_quant_gelu_different_mnk) {
    // 不同M/N/K组合
    TestGroupedMatmulWithGelu({M=1024, N=4096, K=7168}, GMM_ACT_TYPE_GELU_TANH);
}
```

**工作量**：约1天

#### 任务11：pytest集成测试

**新增文件**：
- `gmm/grouped_matmul/tests/pytest/test_grouped_matmul_gelu.py`

**测试覆盖**：
```python
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("act_type", [2, 3])  # GELU-Tanh, GELU-Erf
@pytest.mark.parametrize("m,n,k", [(1024,4096,7168), (2048,7168,2048)])
def test_grouped_matmul_gelu(dtype, act_type, m, n, k):
    # CPU基准计算
    x_cpu = torch.randn(m, k, dtype=dtype)
    w_cpu = torch.randn(k, n, dtype=dtype)
    mm_out_cpu = torch.matmul(x_cpu, w_cpu)
    
    # 根据act_type选择GELU实现
    if act_type == 2:
        y_cpu = gelu_tanh_approx(mm_out_cpu)
    else:
        y_cpu = gelu_erf_exact(mm_out_cpu)
    
    # NPU算子调用
    x_npu = x_cpu.npu()
    w_npu = w_cpu.npu()
    y_npu = aclnn_grouped_matmul(x_npu, w_npu, act_type=act_type)
    
    # 精度对比
    assert torch.allclose(y_cpu, y_npu.cpu(), rtol=1e-3 if act_type==2 else 1e-6)
```

**工作量**：约1天

---

## 五、开发优先级与工作量评估

### 5.1 开发优先级排序

| 优先级 | 任务 | 工作量 | 紧迫性 |
|-------|------|-------|--------|
| **P0** | 任务1：模板宏扩展（ACT_TYPE参数） | 1天 | **立即启动** |
| **P0** | 任务2：新增GELU模板组合 | 0.5天 | **立即启动** |
| **P1** | 任务3：GELU-Tanh Kernel实现 | 1天 | 第2周启动 |
| **P1** | 任务4：GELU-Erf Kernel实现 | 1天 | 第2周启动 |
| **P1** | 任务5：Tiling策略适配 | 1天 | 第2周启动 |
| **P2** | 任务6-7：Host侧参数+Shape推导 | 1天 | 第3周启动 |
| **P2** | 任务8-9：API文档+示例更新 | 1天 | 第3周启动 |
| **P2** | 任务10-11：UT+pytest测试 | 2天 | 第4周启动 |

### 5.2 总工作量评估

- **模板扩展**：1.5天
- **Kernel实现**：2天
- **Tiling适配**：1天
- **Host侧+API**：1.5天
- **测试**：2天
- **总计**：约 **8天**（1.6周）

### 5.3 风险点

| 风险项 | 风险等级 | 应对措施 |
|-------|---------|---------|
| Ascend C erf API不支持 | 高 | 先实现Tanh版本，Erf版本待SDK支持 |
| UB空间不足 | 中 | 调整Tiling策略，减小单次计算块 |
| GELU计算精度 | 低 | pytest充分验证精度 |
| 编译时间增加 | 低 | 分层次编译，仅编译必要模板 |

---

## 六、验收标准

### 6.1 功能验收标准

1. **模板覆盖**：
   - ✓ 新增12个GELU模板（Ascend910B: 6个, Ascend950: 6个）
   - ✓ 支持 actType=2（GELU-Tanh）和 actType=3（GELU-Erf）

2. **算子接口**：
   - ✓ 非量化场景支持 actType参数（0/2/3）
   - ✓ aclnn接口参数校验通过

3. **计算精度**：
   - ✓ GELU-Tanh精度误差 < 1e-3
   - ✓ GELU-Erf精度误差 < 1e-6

### 6.2 性能验收标准

1. **融合性能提升**：
   - Matmul+GELU融合算子性能 ≥ 分离算子性能的 **110%**
   - 减少Global Memory搬运次数：从2次 → 0次

2. **编译时间**：
   - 新增12个模板后，编译时间增加 < 10%

### 6.3 测试验收标准

1. **测试覆盖率**：
   - UT测试用例 > 10个
   - pytest测试用例 > 20个

2. **测试场景覆盖**：
   - 不同dtype：BF16/FLOAT16/FLOAT
   - 不同actType：2/3
   - 不同M/N/K组合：小/中/大shape

---

## 七、参考资料

### 7.1 代码仓相关文件

- `gmm/grouped_matmul/op_kernel/grouped_matmul_tiling_key.h` - 模板定义
- `gmm/grouped_matmul/op_kernel/grouped_matmul.h` - Kernel实现
- `gmm/grouped_matmul/op_kernel/grouped_matmul_vector.h` - Vector计算辅助
- `gmm/grouped_matmul/docs/aclnnGroupedMatmulV4.md` - API文档

### 7.2 技术文档

- [Ascend C Vector API文档](https://hiascend.com/document/detail/ascendc)
- [GELU激活函数论文](https://arxiv.org/abs/1606.08415)
- [GELU精确vs近似对比](https://ml-explained.com/blog/gelu-explained)

### 7.3 GELU数学定义

**Erf精确版本**：
$$
GELU(x) = 0.5x(1 + erf(\frac{x}{\sqrt{2}}))
$$

其中：
$$
erf(x) = \frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2}dt
$$

**Tanh近似版本**：
$$
GELU(x) \approx 0.5x(1 + tanh[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)])
$$

---

## 八、总结

### 需求核心内容

**"非量化 fusedmatmul支持 mm加gelu（Erf和Tanh）融合（基础模板）"** 的具体任务为：

1. **为 grouped_matmul 算子扩展非量化场景的GELU激活函数融合**：
   - 当前：非量化场景仅支持 actType=0（无激活）
   - 需求：扩展支持 actType=2（GELU-Tanh）和 actType=3（GELU-Erf）

2. **新增基础模板定义**：
   - 新增 ACT_TYPE模板参数
   - 新增12个GELU模板组合（Ascend910B: 6个, Ascend950: 6个）

3. **实现GELU Kernel计算**：
   - GELU-Tanh：基于Tanh和多项式计算
   - GELU-Erf：基于erf函数计算

### 开发路线

- **第1周**：完成模板扩展任务（任务1-2）
- **第2周**：完成Kernel实现+Tiling适配（任务3-5）
- **第3周**：完成Host侧+API文档任务（任务6-9）
- **第4周**：完成测试任务（任务10-11）

### 预期成果

- 非量化场景 grouped_matmul 支持 **GELU融合**
- 融合算子性能提升 **10-20%**
- 减少 **中间数据搬运**
- 为后续激活函数融合（ReLU/SiLU等）奠定模板基础

---

**文档版本**：v1.0  
**生成日期**：2026-04-24  
**适用范围**：grouped_matmul 非量化场景 GELU融合扩展需求