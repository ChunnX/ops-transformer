# AIC生成初始矩阵方案

**日期**: 2026-04-28  
**目标**: 将(I-L)初始矩阵生成从AIV移至AIC，实现纯Cube MMA Neumann求逆

---

## 问题背景

### 当前架构问题

| 参数 | 当前值 | 问题 |
|---|---|---|
| chunkSize | 16 | ✓ 已统一 |
| TASK_RATIO | 2 | 导致halfChunkSize=8 |
| InverseCompute | AIV处理16×8半块 | **不完整** |

**数据流矛盾**:
- AIV的InverseCompute只写16×8半块（两个subBlock各写8行）
- NeumannLpowCompute需要完整16×16矩阵做L²=L×L

### 原版vs Neumann对比

| 原版chunk_gated_delta_rule | Neumann版 |
|---|---|
| chunkSize=64, INVERSE_SHAPE=32 | chunkSize=16, INVERSE_SHAPE=16 |
| InverseAIV处理32×32块 | InverseCompute处理16×8半块 |
| 两个subBlock各处理一个32×32块 | **试图拼成16×16，但workspace布局冲突** |

---

## 方案详解

### 方案3: AIC生成初始矩阵

**核心思路**: 将(I-L)矩阵生成从AIV Vector操作改为AIC Cube操作

#### 1. 初始矩阵(I-L)的结构

```
(I-L) for chunk=16:
     [ 1  -L₀₁  -L₀₂  ...  -L₀₁₅ ]
     [ 0    1   -L₁₂  ...  -L₁₁₅ ]
     [ 0    0     1   ...  -L₂₁₅ ]
     ...
     [ 0    0     0   ...    1   ]

L是下三角矩阵，(I-L)构造方式：
- 对角线设为1
- 下三角区域设为-L对应元素
- 上三角区域为0
```

#### 2. AIC生成方案

**方案3.1: 直接用AIC初始化（推荐）**

```cpp
// AIC分支新增函数 InitInverseMatrix()
__aicore__ inline void InitInverseMatrix() {
    // 步骤1: 从kkWsGm_读取L矩阵（已由key@key.T计算得到）
    // 步骤2: 构造I-L写入INVWsGm_
    
    // 方法：利用Cube的SetBias功能
    // - MatMul模板支持SetBias设置对角线
    // - 或者使用L1Buffer构造I-L
    
    // 简化实现：
    // 1. 用Muls将L矩阵乘-1得到-L
    // 2. 用SetBias或SetValue设对角线为1
}
```

**方案3.2: 分离数据流**

```
原流程:
  AIV InverseCompute -> 写INVWsGm_(16×8半块)
  AIC NeumannLpowCompute -> 读INVWsGm_(完整16×16)

修改流程:
  AIC InitInverseMatrix -> 写完整INVWsGm_(16×16)
  AIC NeumannLpowCompute -> 读INVWsGm_(16×16)
  （AIV不再参与初始化，只做后续GBKCompute等）
```

#### 3. 具体修改位置

| 文件 | 函数 | 修改内容 |
|---|---|---|
| `stage1.h` | ProcessOneChunk AIC分支 | 新增InitInverseMatrix()调用 |
| `stage1.h` | ProcessOneChunk AIV分支 | 移除InverseCompute()调用或简化 |
| `stage1.h` | 新增 InitInverseMatrix() | AIC生成完整(I-L)矩阵 |
| `stage1.h` | NeumannLpowCompute | 保持不变，读取完整INVWsGm_ |

---

## 实施步骤

### Step 1: 验证workspace布局
- 检查INVWsGm_、L_powWsGm_、tmpWsGm_分配
- 确保每个core有3个完整16×16矩阵空间
- 当前tiling: `neumannWsSz = sizeHigh * aiCoreNum * c * c * 3`

### Step 2: AIC初始化矩阵
- 在AIC分支的ProcessOneChunk中，同步点之后添加InitInverseMatrix()
- 从kkWsGm_提取L矩阵数据
- 生成(I-L)写入INVWsGm_和L_powWsGm_

### Step 3: 移除AIV InverseCompute依赖
- AIV的InverseCompute改为只做mask处理
- 或者完全移除，让AIV专注于GBKCompute/VBetaCompute

### Step 4: 同步流程调整
```
原同步:
  AIC等待AIV InverseCompute完成(flag 0x7)

新同步:
  AIC完成InitInverseMatrix后通知AIV
  AIV等待flag后继续GBKCompute
```

### Step 5: 验证精度
- 运行pytest单算子测试
- 对比golden结果

---

## 技术细节

### AIC SetBias用法

```cpp
// Matmul模板中设置Bias矩阵
mm.SetBias(true);  // 启用Bias
mm.SetBiasType(...);  // Bias类型

// 或者在L1中构造Bias矩阵后加载
```

### workspace初始化

```cpp
// 在SetGlobalTensors中
// 当前INVWsGm_按coreIdx分配：
INVWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(ws + offset + 
    coreIdx_ * chunkSize_ * chunkSize_ * sizeof(float)));

// chunkSize_=16时，每个core一个16×16=256元素
```

---

## 预期结果

| 指标 | 目标 |
|---|---|---|
| Kernel执行 | 成功，无挂起 |
| 精度 | max_abs_err < 1e-3 |
| 性能 | 6次16×16 MMA + 3次FP16 Add |

---

## 下一步行动

1. 先验证当前workspace大小是否足够
2. 实现AIC版InitInverseMatrix
3. 调整同步流程
4. 测试验证