# 需求分析：非量化 fusedmatmul add mul支持mm bmm全模板

## 一、需求拆解

### 1.1 关键词解析

| 关键词 | 含义 | 具体说明 |
|-------|------|---------|
| **非量化** | 不使用量化数据类型 | 数据类型为 FLOAT16/BFLOAT16/FLOAT，不支持 INT8/INT4 等量化类型 |
| **fusedmatmul** | 融合矩阵乘法算子 | 指 `grouped_matmul` 或 `grouped_matmul_add` 算子 |
| **add** | 加法融合操作 | 算子公式：`y_i = x_i × weight_i + y_i` |
| **mul** | 矩阵乘法操作 | Matmul 计算：`x_i × weight_i` |
| **mm** | Matmul | 标准矩阵乘法：[M,K] × [K,N] = [M,N] |
| **bmm** | Batch Matmul | 批量矩阵乘法：[B,M,K] × [B,K,N] = [B,M,N]，B为Batch维度 |
| **全模板** | 支持所有模板组合 | 包括 kernel_type、dtype、transpose、aiv_aic_ratio 等所有组合 |

### 1.2 需求目标

**核心目标**：为 `grouped_matmul_add` 算子的非量化场景，扩展支持：
1. **Batch Matmul (bmm)** - 支持Batch维度计算
2. **全模板覆盖** - 支持所有模板组合

---

## 二、当前实现分析

### 2.1 现有算子概览

#### grouped_matmul 算子

**算子功能**：
- 分组矩阵乘：`y_i[m_i,n_i] = x_i[m_i,k_i] × weight_i[k_i,n_i]`
- 支持 K轴分组、M轴分组
- 支持量化/非量化/伪量化多种模式

**当前支持的维度**：
- **仅支持 2D 矩阵乘（mm）**：[M,K] × [K,N] = [M,N]
- **不支持 Batch 维度（bmm）**：无 [B,M,K] × [B,K,N] 支持

#### grouped_matmul_add 算子

**算子功能**：
- 融合矩阵乘 + 加法：`yRef_i = x_i × weight_i + y_i`
- 仅支持非量化场景（FLOAT16/BFLOAT16 输入，FLOAT32 输出）

**当前限制**：
- **仅支持 K轴分组**
- **不支持 Batch 维度**
- **模板覆盖不全**

### 2.2 当前模板分析

#### 非量化模板定义（GMM_FLOAT）

```cpp
// grouped_matmul_tiling_key.h 第92-104行
#define SET_GMM_NO_QUANT_TPL_ARGS(kernel_type, dtype, trans_a, trans_b, aiv_aic_ratio)
    ASCENDC_TPL_KERNEL_TYPE_SEL(kernel_type),      // kernel类型
    ASCENDC_TPL_DTYPE_SEL(D_T_A, dtype),           // 输入数据类型
    ASCENDC_TPL_DTYPE_SEL(D_T_B, dtype),           // 权重数据类型
    ASCENDC_TPL_DTYPE_SEL(D_T_Y, dtype),           // 输出数据类型
    ASCENDC_TPL_BOOL_SEL(TRANS_A, trans_a),        // 输入转置
    ASCENDC_TPL_BOOL_SEL(TRANS_B, trans_b),        // 权重转置
    ASCENDC_TPL_UINT_SEL(GROUP_LIST_TYPE, ...),    // 分组类型
    ASCENDC_TPL_BOOL_SEL(IS_STATIC_TILING_API, 0), // Tiling API类型
    ASCENDC_TPL_UINT_SEL(A8W4_KERNEL_TEMPLATE, ...), // 量化模板（非量化场景为NONE）
    ASCENDC_TPL_UINT_SEL(A16W8_KERNEL_TEMPLATE, ...), // 量化模板（非量化场景为NONE）
    ASCENDC_TPL_UINT_SEL(AIV_AIC_RATIO, ...),      // AI Vector:AI Core比例
    ASCENDC_TPL_BOOL_SEL(IS_ENABLE_FIXED_AXIS, 0)  // 固定轴算法
```

#### 当前模板组合（342-366行）

**Ascend910B/A3（__CCE_AICORE__ == 220）**：

| dtype | kernel_type | trans_a | trans_b | aiv_aic_ratio |
|-------|-------------|---------|---------|---------------|
| BF16  | AIC_1_0     | 0       | 0       | CUBE_ONLY     |
| BF16  | AIC_1_0     | 0       | 1       | CUBE_ONLY     |
| FLOAT16 | AIC_1_0   | 0       | 0       | CUBE_ONLY     |
| FLOAT16 | AIC_1_0   | 0       | 1       | CUBE_ONLY     |
| FLOAT | AIC_1_0     | 0       | 0       | CUBE_ONLY     |
| FLOAT | AIC_1_0     | 0       | 1       | CUBE_ONLY     |
| BF16  | AIC_1_1     | 1       | 0       | AIV_AIC_RATIO_1 |
| FLOAT16 | AIC_1_1   | 1       | 0       | AIV_AIC_RATIO_1 |
| FLOAT | AIC_1_1     | 1       | 0       | AIV_AIV_RATIO_1 |

**Ascend950（__CCE_AICORE__ == 200）**：

| dtype | kernel_type | trans_a | trans_b | aiv_aic_ratio |
|-------|-------------|---------|---------|---------------|
| BF16  | AICORE      | 0       | 0       | CUBE_ONLY     |
| BF16  | AICORE      | 0       | 1       | CUBE_ONLY     |
| FLOAT16 | AICORE    | 0       | 0       | CUBE_ONLY     |
| FLOAT16 | AICORE    | 0       | 1       | CUBE_ONLY     |
| FLOAT | AICORE      | 0       | 0       | CUBE_ONLY     |
| FLOAT | AICORE      | 0       | 1       | CUBE_ONLY     |
| BF16  | MIX_VECTOR_CORE | 1 | 0    | AIV_AIC_RATIO_1 |
| FLOAT16 | MIX_VECTOR_CORE | 1 | 0 | AIV_AIC_RATIO_1 |
| FLOAT | MIX_VECTOR_CORE | 1 | 0   | AIV_AIC_RATIO_1 |

### 2.3 缺失模板组合

#### trans_a × trans_b 组合矩阵

| trans_a | trans_b | 当前支持 | 需要新增 |
|---------|---------|---------|---------|
| 0 | 0 | ✓ | - |
| 0 | 1 | ✓ | - |
| 1 | 0 | ✓ | - |
| 1 | 1 | ✗ | **需新增** |

#### kernel_type × aiv_aic_ratio 组合

| kernel_type | aiv_aic_ratio | 当前支持 | 需要新增 |
|-------------|---------------|---------|---------|
| AIC_1_0     | CUBE_ONLY     | ✓ | - |
| AIC_1_0     | AIV_AIC_RATIO_1 | ✗ | **需新增** |
| AIC_1_1     | AIV_AIC_RATIO_1 | ✓ | - |
| AIC_1_1     | AIV_AIC_RATIO_2 | ✗ | **需新增** |
| AIC_1_2     | AIV_AIC_RATIO_2 | ✗ | **需新增** |

#### dtype覆盖

- ✓ BF16
- ✓ FLOAT16
- ✓ FLOAT
- ✗ **缺少混合精度场景**（如 x=FLOAT16, weight=BF16）

---

## 三、需要新增的Batch Matmul (bmm)支持

### 3.1 Batch Matmul 定义

**计算公式**：
```
y[b,m,n] = x[b,m,k] × weight[b,k,n] + bias[b,n]
```

**维度规格**：
- **Batch维度**：B，表示批量个数
- **M维度**：Token数量（行）
- **K维度**：隐藏层维度（内积维度）
- **N维度**：输出维度（列）

**典型场景**：
- MoE 专家并行：Batch=专家数，每专家独立计算
- 多头注意力：Batch=头数，每头独立计算

### 3.2 Batch Matmul vs Grouped Matmul

| 特性 | Grouped Matmul | Batch Matmul |
|-----|---------------|--------------|
| 分组方式 | K轴或M轴分段 | Batch维度独立 |
| 数据布局 | 多个tensor拼接 | 单tensor含Batch维度 |
| Shape变化 | 每组维度可不同 | 所有Batch维度相同 |
| 性能优化 | 分组调度 | Batch并行调度 |
| 适用场景 | MoE路由计算 | MoE专家并行计算 |

### 3.3 实现方案对比

#### 方案1：扩展 grouped_matmul_add 算子

**优点**：
- 统一算子接口，减少算子数量
- 复用现有Tiling和Kernel框架

**缺点**：
- 算子复杂度增加，维护困难
- 接口参数膨胀（需新增 batch_dim 参数）
- Shape推导逻辑复杂

#### 方案2：新增 batch_grouped_matmul_add 算子

**优点**：
- 算子职责清晰，易于维护
- 专项优化，性能更优
- 接口简洁，易于使用

**缺点**：
- 算子数量增加
- 部分代码重复

#### 方案3：参考 mc2 模块的 batch_matmul 实现

**参考算子**：
- `batch_mat_mul_reduce_scatter_allto_all`（mc2模块）
- 已有Batch Matmul基础设施（gmm_infra/gemm目录）

**借鉴点**：
- Batch维度切分策略
- Batch并行调度逻辑
- Batch Shape推导

---

## 四、具体开发任务清单

### 4.1 模板扩展任务

#### 任务1：新增 trans_a=1, trans_b=1 模板

**修改文件**：
- `gmm/grouped_matmul/op_kernel/grouped_matmul_tiling_key.h`

**新增模板**：
```cpp
// 非量化场景，trans_a=1, trans_b=1
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(
    ASCENDC_TPL_MIX_AIC_1_1, GMM_TPL_BF16, 1, 1, GROUPED_MATMUL_AIV_AIC_RATIO_1)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(
    ASCENDC_TPL_MIX_AIC_1_1, GMM_TPL_FLOAT16, 1, 1, GROUPED_MATMUL_AIV_AIC_RATIO_1)),
ASCENDC_TPL_ARGS_SEL(SET_GMM_NO_QUANT_TPL_ARGS(
    ASCENDC_TPL_MIX_AIC_1_1, GMM_TPL_FLOAT, 1, 1, GROUPED_MATMUL_AIV_AIC_RATIO_1)),
```

**工作量**：约0.5天

#### 任务2：新增 kernel_type × aiv_aic_ratio 组合

**新增组合**：
- `AIC_1_0 × AIV_AIC_RATIO_1`
- `AIC_1_1 × AIV_AIC_RATIO_2`
- `AIC_1_2 × AIV_AIC_RATIO_2`

**工作量**：约0.5天

#### 任务3：新增 grouped_matmul_add 模板定义

**修改文件**：
- `gmm/grouped_matmul_add/op_kernel/` 新增 `grouped_matmul_add_tiling_key.h`

**内容**：
- 定义 grouped_matmul_add 专用模板宏
- 支持非量化场景的所有组合

**工作量**：约1天

### 4.2 Batch Matmul支持任务

#### 任务4：Batch Matmul原型定义

**修改文件**：
- 方案1：扩展 `gmm/grouped_matmul_add/op_host/grouped_matmul_add_def.cpp`
- 方案2：新增 `gmm/batch_grouped_matmul_add/` 目录

**新增内容**：
- 输入张量支持Batch维度：`[B,M,K]`、`[B,K,N]`
- 输出张量：`[B,M,N]`
- 新增属性：`batch_dim`（Batch维度索引）

**工作量**：约1天

#### 任务5：Batch Matmul Shape推导

**实现逻辑**：
```cpp
// InferShape逻辑
if (has_batch_dim) {
    // Batch Matmul场景
    batch_size = x_shape[batch_dim];
    m = x_shape[batch_dim + 1];
    k = x_shape[batch_dim + 2];
    n = weight_shape[batch_dim + 2];
    out_shape = {batch_size, m, n};
} else {
    // Grouped Matmul场景（原有逻辑）
    ...
}
```

**工作量**：约1天

#### 任务6：Batch Matmul Tiling策略

**核心挑战**：
- Batch维度切分：如何分配Batch到不同AI Core
- M/N/K维度切分：Batch内的矩阵乘切分
- 内存布局：Batch数据的连续性处理

**实现策略**：
```cpp
// Batch Tiling策略
struct BatchTilingConfig {
    uint32_t batch_size;        // Batch总数
    uint32_t batch_per_core;    // 每核处理Batch数
    uint32_t batch_block_size;  // Batch块大小
    uint32_t m_block_size;      // M维度块大小
    uint32_t n_block_size;      // N维度块大小
    uint32_t k_block_size;      // K维度块大小
};
```

**参考**：
- `mc2/batch_mat_mul_reduce_scatter_allto_all/op_host/op_tiling/`

**工作量**：约2天

#### 任务7：Batch Matmul Kernel实现

**Kernel类设计**：
```cpp
template<typename T>
class BatchGroupedMatmulAdd {
public:
    void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR y, 
              GM_ADDR yRef, BatchTilingData* tiling);
    void Process();
private:
    void ProcessBatchBlock(uint32_t batch_idx);  // 处理单个Batch块
    void ProcessMatmulBlock(uint32_t m_idx, uint32_t n_idx); // Batch内矩阵乘
};
```

**关键实现点**：
1. **Batch循环调度**：`for (batch_idx = 0; batch_idx < batch_per_core; batch_idx++)`
2. **Batch内Matmul**：复用现有 `grouped_matmul.h` 的 `MNBlockIdxCompute` 等函数
3. **数据搬运**：Batch维度的 Global Memory → UB 搬运优化

**工作量**：约3天

#### 任务8：Batch Matmul测试用例

**测试覆盖**：
- 不同Batch大小：B=1, 2, 8, 16, 32, 64, 128
- 不同M/N/K维度组合
- 不同dtype：BF16, FLOAT16, FLOAT
- 不同transpose组合

**工作量**：约1天

### 4.3 API接口任务

#### 任务9：aclnn接口扩展

**新增参数**：
```cpp
aclnnStatus aclnnGroupedMatmulAddV3GetWorkspaceSize(
    const aclTensorList *x,         // [B,M,K] 或 [M,K]
    const aclTensorList *weight,    // [B,K,N] 或 [K,N]
    const aclTensorList *y,         // [B,M,N] 或 [M,N]
    const aclTensor     *groupListOptional,
    int64_t              batchDim,  // 新增：Batch维度索引（-1表示无Batch）
    int64_t              groupType,
    int64_t              groupListType,
    aclTensorList       *yRef,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor);
```

**工作量**：约1天

#### 任务10：文档与示例

**新增文档**：
- `docs/aclnnGroupedMatmulAddV3.md`
- 新增Batch Matmul使用示例

**工作量**：约0.5天

---

## 五、开发优先级与工作量评估

### 5.1 开发优先级排序

| 优先级 | 任务 | 工作量 | 紧迫性 |
|-------|------|-------|--------|
| P0 | 任务3：grouped_matmul_add模板定义 | 1天 | **立即启动** |
| P0 | 任务1-2：模板组合补全 | 1天 | **立即启动** |
| P1 | 任务4-5：Batch原型+Shape推导 | 2天 | 第2周启动 |
| P1 | 任务6：Batch Tiling策略 | 2天 | 第3周启动 |
| P1 | 任务7：Batch Kernel实现 | 3天 | 第4周启动 |
| P2 | 任务8：测试用例 | 1天 | 第5周启动 |
| P2 | 任务9-10：API+文档 | 1.5天 | 第5周启动 |

### 5.2 总工作量评估

- **模板扩展**：2天
- **Batch Matmul开发**：7天
- **测试+文档**：2.5天
- **总计**：约11.5天（2.3周）

### 5.3 风险点

| 风险项 | 风险等级 | 应对措施 |
|-------|---------|---------|
| Batch Tiling策略复杂 | 中 | 参考mc2模块batch_matmul实现 |
| Kernel性能优化难度大 | 中 | 先实现功能，后续性能迭代 |
| 内存布局处理复杂 | 中 | 设计清晰的Batch数据搬运函数 |
| 模板数量爆炸 | 低 | 分层次编译，避免全量编译 |

---

## 六、验收标准

### 6.1 功能验收标准

1. **模板覆盖**：
   - ✓ 支持所有 trans_a × trans_b 组合（4种）
   - ✓ 支持所有 kernel_type × aiv_aic_ratio 组合（6种）
   - ✓ 支持 BF16/FLOAT16/FLOAT 全覆盖

2. **Batch Matmul支持**：
   - ✓ 支持Batch维度计算
   - ✓ Batch维度范围：1-128
   - ✓ 支持非量化场景（FLOAT16/BFLOAT16）

3. **算子接口**：
   - ✓ aclnn接口支持Batch参数
   - ✓ Shape推导正确
   - ✓ 调用示例完备

### 6.2 性能验收标准

1. **性能基准**：
   - Batch Matmul性能 ≥ PyTorch批量化矩阵乘性能的90%
   - 不同Batch大小的性能稳定性（方差<10%）

2. **内存效率**：
   - Batch数据搬运效率优化（减少重复搬运）
   - UB利用率 > 80%

### 6.3 测试验收标准

1. **测试覆盖率**：
   - UT测试覆盖率 > 80%
   - pytest测试用例 > 50个

2. **精度验证**：
   - CPU vs NPU精度误差 < 1e-3
   - 边界场景测试（Batch=1, 128等）

---

## 七、参考资料

### 7.1 代码仓相关文件

- `gmm/grouped_matmul/op_kernel/grouped_matmul_tiling_key.h` - 模板定义
- `gmm/grouped_matmul_add/README.md` - 算子文档
- `mc2/batch_mat_mul_reduce_scatter_allto_all/` - Batch Matmul参考实现
- `gmm/grouped_matmul/op_kernel/gmm_infra/` - 基础设施

### 7.2 技术文档

- [Ascend C Matmul API文档](https://hiascend.com/document/detail/ascendc)
- [Batch Matmul算法介绍](https://pytorch.org/docs/stable/generated/torch.bmm.html)
- [MoE专家并行计算论文](https://arxiv.org/abs/2006.16668)

---

## 八、总结

### 需求核心内容

**"非量化 fusedmatmul add mul支持mm bmm全模板"** 的具体任务为：

1. **为 grouped_matmul_add 算子扩展全模板支持**：
   - 新增缺失的 transpose 组合（trans_a=1, trans_b=1）
   - 新增缺失的 kernel_type × aiv_aic_ratio 组合
   - 新增 grouped_matmul_add 专用模板定义文件

2. **为 grouped_matmul_add 算子新增 Batch Matmul (bmm) 支持**：
   - 支持Batch维度计算：[B,M,K] × [B,K,N] = [B,M,N]
   - 实现Batch维度切分策略
   - 实现Batch Kernel并行调度
   - 扩展aclnn接口支持Batch参数

### 开发路线

- **第1周**：完成模板扩展任务（任务1-3）
- **第2-4周**：完成Batch Matmul开发任务（任务4-7）
- **第5周**：完成测试+文档任务（任务8-10）

### 预期成果

- grouped_matmul_add 算子支持 **mm + bmm** 双模式
- 非量化场景模板 **100%覆盖**
- Batch Matmul性能优化，支持 **MoE专家并行场景**

---

**文档版本**：v1.0  
**生成日期**：2026-04-24  
**适用范围**：grouped_matmul_add 算子 Batch Matmul + 全模板扩展需求