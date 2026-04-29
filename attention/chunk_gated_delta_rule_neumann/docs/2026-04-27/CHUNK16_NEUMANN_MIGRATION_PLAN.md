# ChunkGatedDeltaRuleNeumann CHUNK=16 + Neumann 求逆迁移计划

## 一、修改目标

将 `chunk_gated_delta_rule_neumann` 算子的两个核心参数修改：

| 参数 | 当前值 | 目标值 | 原因 |
|------|--------|--------|------|
| **chunkSize** | 64 | **16** | decay 值在 bf16 动态范围内更安全；与 FlashKDA 对齐 |
| **求逆方法** | 逐列前代法 (32×32 对角块 + Cube 块间乘) | **Neumann 级数展开** (16×16 FP16 MMA) | 利用 Cube 加速，减少 Vector 循环 |

---

## 二、当前实现分析

### 2.1 chunkSize 定义位置

```
op_host/chunk_gated_delta_rule_neumann_tiling.cpp:
  Line 113: int64_t c = 64;    // chunk size 取 64
  
op_kernel/chunk_gated_delta_rule_neumann_tiling_data.h:
  Line 33: int64_t chunkSize;

op_kernel/chunk_gated_delta_rule_neumann_stage1.h:
  Line 32: constexpr uint64_t INVERSE_SHAPE = 32;  // 对角块边长
  Line 33: constexpr uint64_t INVERSE_COUNT = 5;   // 求逆所需空间
```

### 2.2 当前求逆流程 (CHUNK=64)

**分块策略**：64×64 矩阵拆成 2×2 块矩阵

```
原矩阵 (I+L):                          逆矩阵 (I+L)⁻¹:
┌─────────┬─────────┐ (32×32)          ┌────────────────────┬──────────────┐
│ I+L_UL  │    0    │                  │   (I+L_UL)⁻¹      │      0       │
├─────────┼─────────┤                  ├────────────────────┼──────────────┤
│  L_LL   │ I+L_LR  │                  │  非对角块 (乘法)    │  (I+L_LR)⁻¹ │
└─────────┴─────────┘                  └────────────────────┴──────────────┘

步骤1 [AIV]: 前代法分别求 (I+L_UL)⁻¹ 和 (I+L_LR)⁻¹
          - InverseAIV(offset=0, inverseVecLen=32)  // 上半块，31步循环
          - InverseAIV(offset=32, inverseVecLen=32) // 下半块，31步循环
          
步骤2 [AIC]: 非对角块 = -(I+L_LR)⁻¹ × L_LL × (I+L_UL)⁻¹
          - AttnInverseMMCompute(INVERSE_SHAPE=32)
          - 两次 32×32 fp32 Cube matmul
```

**前代法核心循环** (InverseAIV, stage1.h:458-499):

```cpp
for (int i = 1; i < inverseVecLen; ++i) {  // 31 步循环
    Gather(col, attnUbFloat_[...], ...);       // 取列向量
    Broadcast(col, ...);                        // 列广播
    Broadcast(row, inverseLocal_[...], ...);   // 行广播
    MulAddDst(yLocal[...], col, row, ...);     // 外积累加
    Sub(inverseLocal_[...], ei, yLocal[...]);  // 得到第 i 行
}
```

---

## 三、Neumann 级数求逆原理 (CHUNK=16)

### 3.1 数学推导

对于严格下三角矩阵 L（对角线为 0），L 是幂零矩阵，L^C = 0：

```
(I + L)^{-1} = I - L + L² - L³ + ... + (-L)^{C-1} = Σ_{k=0}^{C-1} (-L)^k
```

**倍增法**：初始 `INV₀ = I - L` 含前两项，只需 log₂(C) - 1 = **3 步** doubling：

```
初始:  INV₀ = I - L                          [16,16] FP16  (含 (-L)⁰ + (-L)¹)

步骤1: L²  = L × L                           [16,16] FP16 MMA
       INV₁ = INV₀ + INV₀ × L²               含 (-L)⁰ ~ (-L)³ 项
       展开: (I-L)(I+L²) = I - L + L² - L³

步骤2: L⁴  = L² × L²                         [16,16] FP16 MMA
       INV₂ = INV₁ + INV₁ × L⁴               含 (-L)⁰ ~ (-L)⁷ 项

步骤3: L⁸  = L⁴ × L⁴                         [16,16] FP16 MMA
       INV₃ = INV₂ + INV₂ × L⁸               含 (-L)⁰ ~ (-L)¹⁵ = 精确解 (I+L)⁻¹
```

**总计**：6 次 16×16 FP16 MMA + 3 次 FP16 Add

### 3.2 与前代法对比

| 维度 | 前代法 (CHUNK=64) | Neumann (CHUNK=16) |
|------|------------------|-------------------|
| **矩阵大小** | 64×64 (拆成 2×32×32) | **单块 16×16** |
| **算法复杂度** | 31 步 Vector 循环 × 2 + 2 次 Cube | **6 次 Cube MMA + 3 次 Add** |
| **精度** | FP32 全程 | **FP16 MMA + FP16 加法** |
| **硬件利用** | Vector ALU (对角块) + Cube (块间) | **Cube MMA 全程** |
| **收敛保证** | 精确（前代法） | **精确（幂零矩阵有限级数）** |

---

## 四、修改方案

### 4.0 正确的 Neumann 架构设计 🔴 关键

**问题发现**: 原 NeumannInverseCompute 在 AIV 分支调用 AICProcess 是错误的！

**正确架构** (利用 Cube L0C 累加能力):

```
数据流设计:
  GM(L FP32) → L1 → L0A/L0B → Cube(MMA FP32) → L0C(FP32累加) → GM(FP32)
                                                         ↓
                                            GM(FP32) → L1 → L0A → 下次 MMA

核心优化:
  1. L0C 可以直接累加 (无需 Vector Add)
  2. 初始 INV₀ = I - L 可以通过 SetBias 实现
  3. 每次迭代: L0C += INV × L_pow (累加模式)
  4. FP32 全程在 Cube 流水线
```

**L0C 累加机制**:

```cpp
// Ascend C matmul API 支持 L0C bias
mm.SetOrgShape(M, N, K);
mm.SetSingleShape(M, N, K);
mm.SetTensorA(A_gm);      // L0A 输入
mm.SetTensorB(B_gm);      // L0B 输入
mm.SetBias(bias_gm);      // ← 关键：L0C 初始值 = bias
mm.IterateAll(C_gm);      // 输出 = A×B + bias (L0C 累加)
mm.End();
```

**Neumann 倍增法正确实现**:

```
步骤0: INV₀ = I - L
  - bias = I (单位矩阵，GM buffer)
  - A = L, B = L (但需要 -L × L = -L²)
  - 或直接: bias = 0, A = -L, 输出到 INV_Gm_

步骤1-3: INV_k = INV_{k-1} + INV_{k-1} × L_pow_k
  - bias = INV_{k-1} (L0C 初始值)
  - A = INV_{k-1}, B = L_pow_k
  - 输出 = INV × L_pow + INV = INV_k (L0C 累加)
```

**AIC 分支执行** (必须):

```cpp
if ASCEND_IS_AIC {
    // Neumann 求逆在 AIC 分支执行
    if (INVERSE_SHAPE == chunkSize_) {
        NeumannMatmulCompute();  // 利用 Cube L0C 累加
    }
}
if ASCEND_IS_AIV {
    // AIV 仅准备 L 数据，不参与求逆计算
    // mask/gamma 操作完成后等待 AIC 求逆结果
}
```

---

### 4.1 Stage1 修改项

#### 修改点 1: Tiling 层 chunkSize

**文件**: `op_host/chunk_gated_delta_rule_neumann_tiling.cpp`

```cpp
// Line 113 修改
- int64_t c = 64;    // chunk size 取 64
+ int64_t c = 16;    // chunk size 取 16 (与 FlashKDA 对齐)
```

**影响分析**:
- `tilingData_.chunkSize = c` → 传递到 kernel
- `tilingData_.maxGroupLength = p * tilingData_.aiCoreNum * c` → 变小
- workspace 计算 `interWorkspaceSz` 需重新评估

#### 修改点 2: Kernel 层常量

**文件**: `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`

```cpp
// Line 32-33 修改
- constexpr uint64_t INVERSE_SHAPE = 32;          // 对角块边长
- constexpr uint64_t INVERSE_COUNT = 5;           // 求逆所需空间
+ constexpr uint64_t INVERSE_SHAPE = 16;          // 整块边长 (CHUNK=16)
+ constexpr uint64_t INVERSE_COUNT = 4;           // Neumann: L, L², L⁴, L⁸, INV, tmp (6个)
```

#### 修改点 3: 删除分块逻辑

**文件**: `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`

```cpp
// 删除以下函数或使其空实现:
// - AttnInverseMMCompute() // 块间乘法不再需要
// - InverseAIV() 的 offset 参数逻辑 // CHUNK=16 无需分块

// InverseCompute() 简化为:
__aicore__ inline void InverseCompute()
{
    // 构造 I - L (初始 INV₀)
    inverseLocal_ = fp32OutQueue_.AllocTensor<float>();
    Muls(inverseLocal_, attnUbFloat_, static_cast<float>(-1.0), chunkSize_ * chunkSize_);
    // 对角线设为 1.0
    for (int i = 0; i < chunkSize_; ++i) {
        inverseLocal_.SetValue(i * chunkSize_ + i, static_cast<float>(1.0));
    }
    
    // Neumann 3 步倍增
    NeumannInverseCompute();  // 新增函数
    
    fp32OutQueue_.EnQue(inverseLocal_);
    DataCopyOutFp32(chunkSize_, chunkSize_, chunkSize_, attnWsGm_[0]);
}
```

#### 修改点 4: 新增 Neumann 求逆函数

**文件**: `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`

```cpp
__aicore__ inline void NeumannInverseCompute()
{
    // 需要的 buffer:
    // - L_pow: 存储 L², L⁴, L⁸
    // - tmp: 存储 INV × L_pow 的临时结果
    
    auto L_pow = inverseUbFloat_[0];               // L², L⁴, L⁸ 复用同一 buffer
    auto tmp = inverseUbFloat_[chunkSize_ * chunkSize_];
    
    // ===== 步骤1: L² = L × L =====
    // AIC: Cube matmul (FP16 输入 → FP32 累加 → 截断 FP16 或保留 FP32)
    // 由于 Ascend C 无 FP16 MMA 原生接口，使用 FP32 Cube + 手动截断
    mmFp32_->SetOrgShape(chunkSize_, chunkSize_, chunkSize_);
    mmFp32_->SetSingleShape(chunkSize_, chunkSize_, chunkSize_);
    mmFp32_->SetTensorA(attnWsGm_[0]);              // L [16,16]
    mmFp32_->SetTensorB(attnWsGm_[0]);              // L [16,16]
    mmFp32_->IterateAll(L_powGm_);
    mmFp32_->End();
    SyncAll<false>();
    
    // INV₁ = INV₀ + INV₀ × L²
    mmFp32_->SetTensorA(inverseLocalGm_);
    mmFp32_->SetTensorB(L_powGm_);
    mmFp32_->IterateAll(tmpGm_);
    mmFp32_->End();
    SyncAll<false>();
    
    // FP32 Add: INV += tmp
    Add(inverseLocal_, inverseLocal_, tmp, chunkSize_ * chunkSize_);
    PipeBarrier<PIPE_V>();
    
    // ===== 步骤2: L⁴ = L² × L² =====
    mmFp32_->SetTensorA(L_powGm_);
    mmFp32_->SetTensorB(L_powGm_);
    mmFp32_->IterateAll(L_powGm_);
    mmFp32_->End();
    SyncAll<false>();
    
    // INV₂ = INV₁ + INV₁ × L⁴
    mmFp32_->SetTensorA(inverseLocalGm_);
    mmFp32_->SetTensorB(L_powGm_);
    mmFp32_->IterateAll(tmpGm_);
    mmFp32_->End();
    SyncAll<false>();
    
    Add(inverseLocal_, inverseLocal_, tmp, chunkSize_ * chunkSize_);
    PipeBarrier<PIPE_V>();
    
    // ===== 步骤3: L⁸ = L⁴ × L⁴ =====
    mmFp32_->SetTensorA(L_powGm_);
    mmFp32_->SetTensorB(L_powGm_);
    mmFp32_->IterateAll(L_powGm_);
    mmFp32_->End();
    SyncAll<false>();
    
    // INV₃ = INV₂ + INV₂ × L⁸ = (I+L)⁻¹ 精确解
    mmFp32_->SetTensorA(inverseLocalGm_);
    mmFp32_->SetTensorB(L_powGm_);
    mmFp32_->IterateAll(tmpGm_);
    mmFp2_->End();
    SyncAll<false>();
    
    Add(inverseLocal_, inverseLocal_, tmp, chunkSize_ * chunkSize_);
    PipeBarrier<PIPE_V>();
    
    // 至此 INV = Σ_{k=0}^{15} (-L)^k = (I+L)⁻¹
}
```

#### 修改点 5: UB Buffer 重新分配

**文件**: `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`

```cpp
// Line 158-163 修改
// 原: halfChunkSize_ * halfChunkSize_ * INVERSE_COUNT (32×32×5 = 5120 FP32)
// 新: chunkSize_ * chunkSize_ * INVERSE_COUNT (16×16×6 = 1536 FP32)

- inverseUbFloat_ = tmpBuff_.GetWithOffset<float>(
-     static_cast<uint32_t>(halfChunkSize_ * halfChunkSize_ * INVERSE_COUNT), buffOffset);
- buffOffset += halfChunkSize_ * halfChunkSize_ * INVERSE_COUNT * sizeof(float);

+ // Neumann 求逆 buffer: L_pow[16×16] + tmp[16×16] = 2 块
+ constexpr uint64_t NEUMANN_BUFFER_COUNT = 2;  // L_pow + tmp
+ inverseUbFloat_ = tmpBuff_.GetWithOffset<float>(
+     static_cast<uint32_t>(chunkSize_ * chunkSize_ * NEUMANN_BUFFER_COUNT), buffOffset);
+ buffOffset += chunkSize_ * chunkSize_ * NEUMANN_BUFFER_COUNT * sizeof(float);
```

**Workspace 修改**:

```cpp
// Line 103-109 需新增 L_pow 和 tmp 的 GM 地址
+ uint64_t L_powWsAddr_;
+ uint64_t tmpWsAddr_;

// Init() 中:
+ L_powWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + offset));
+ offset += coreNum_ * chunkSize_ * chunkSize_ * sizeof(float);
+ tmpWsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(initParams.ws + offset));
+ offset += coreNum_ * chunkSize_ * chunkSize_ * sizeof(float);
```

---

### 4.2 Tiling 层 workspace 重新计算

**文件**: `op_host/chunk_gated_delta_rule_neumann_tiling.cpp`

```cpp
// Line 126-134 修改 interWorkspaceSz 计算
// 原: 基于 maxGroupLength (与 chunkSize 无关) 和 b * nv * dv * dk (highState)

// 需新增 Neumann 求逆 workspace:
int64_t neumannWsSz = sizeHigh * nv * tilingData_.aiCoreNum * chunkSize_ * chunkSize_; // L_pow
neumannWsSz += sizeHigh * nv * tilingData_.aiCoreNum * chunkSize_ * chunkSize_;       // tmp

tilingData_.interWorkspaceSz += neumannWsSz;
```

---

### 4.3 Stage2/Stage3 影响分析

| Stage | chunkSize 影响 | 需修改 |
|-------|---------------|--------|
| **Stage2** | chunkNum 变化：`(seqLen+63)/64` → `(seqLen+15)/16` | chunk 循环次数增加，但逻辑不变 |
| **Stage3** | 同上 | chunk 循环次数增加，但逻辑不变 |

**关键发现**: Stage2/Stage3 的核心逻辑（state update + output）与 chunk 大小无关，只是**处理次数变多**。

```cpp
// Stage2 Process() 核心循环 (stage2.h:99)
for (int64_t cId = 0; cId < chunkNum_; cId++) {
    // chunkNum_ 变化，但内部逻辑不变
    // curChunkSize_ 仍为 chunkSize_ 或尾块大小
}
```

**结论**: Stage2/Stage3 **无需修改代码逻辑**，但性能会受影响（更多 chunk → 更多 sync）。

---

## 五、精度策略选择

### 5.1 FP16 vs FP32

参考文档指出两种策略：

| 策略 | 精度 | 优点 | 缺点 |
|------|------|------|------|
| **A: 纯 FP16 MMA + FP16 加法** | 10bit 尾数 | 与 FlashKDA 对齐；寄存器内完成 | Ascend C 无原生 FP16 MMA 接口 |
| **B: FP32 Cube + FP32 加法** | 23bit 尾数 | Ascend C 原生支持；精度更高 | 比 FlashKDA 精度好，但计算量大 |

**决策**: 采用 **策略 A (FP16 MMA + FP16 加法)** - 与 FlashKDA 对齐

理由：
1. 与 FlashKDA 实现完全对齐，便于验证和对比
2. 16×16 矩阵小，FP16 精度在可控范围内
3. Ascend Cube 支持 FP16 MMA，通过 matmul API 配置实现

**实现细节**：
- L 输入需要 Cast 到 FP16（从原 FP32）
- MMA 结果为 FP32 累加，需截断回 FP16 存储
- Add 操作在 FP16 下完成

### 5.2 L 的计算精度

当前 L 计算：
```
KKT = K [C,Dk] fp32 × K.T [Dk,C] fp32  → fp32 Cube
L = KKT × β(fp32) × Γ(fp32)            → fp32 Vector
```

CHUNK=16 后 L 计算不变，仍为 FP32 输出。

---

## 六、实施步骤

### Step 1: Tiling 层修改 (op_host)

| 文件 | 修改内容 |
|------|---------|
| `chunk_gated_delta_rule_neumann_tiling.cpp` | Line 113: c = 64 → 16 |
| `chunk_gated_delta_rule_neumann_tiling.cpp` | Line 126-134: 新增 neumannWsSz 计算 |

### Step 2: Kernel 常量修改 (op_kernel)

| 文件 | 修改内容 |
|------|---------|
| `chunk_gated_delta_rule_neumann_stage1.h` | INVERSE_SHAPE = 32 → 16 |
| `chunk_gated_delta_rule_neumann_stage1.h` | INVERSE_COUNT 重新定义 |
| `chunk_gated_delta_rule_neumann_stage1.h` | UB buffer 重新分配 |

### Step 3: Stage1 求逆函数重构 (op_kernel)

| 文件 | 修改内容 |
|------|---------|
| `chunk_gated_delta_rule_neumann_stage1.h` | 删除/空化 AttnInverseMMCompute |
| `chunk_gated_delta_rule_neumann_stage1.h` | 新增 NeumannInverseCompute |
| `chunk_gened_delta_rule_neumann_stage1.h` | 修改 InverseCompute 调用 Neumann |
| `chunk_gated_delta_rule_neumann_stage1.h` | 新增 L_powWsGm_, tmpWsGm_ 成员变量 |

### Step 4: Golden 函数同步修改 (tests)

| 文件 | 修改内容 |
|------|---------|
| `chunk_gated_delta_rule_neumann_golden.py` | chunk_size 参数 64 → 16 |
| `generate_test_inputs.py` | seqlen 测试参数调整 |

### Step 5: 编译验证

```bash
# 编译
bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_neumann -j16

# 安装
bash build.sh --install --soc=ascend910_93

# 测试
bash build.sh --run_example chunk_gated_delta_rule_neumann eager cust
```

---

## 七、风险评估

| 风险点 | 严重程度 | 缓解方案 |
|--------|---------|---------|
| **chunkSize=16 导致 chunk 数增加** | 中 | 性能下降，但精度提升；需实测对比 |
| **Neumann 求逆 Cube 调用开销** | 低 | 16×16 矩阵 Cube 效率低于 64×64，但步数少 |
| **FP32 vs FP16 精度差异** | 低 | 采用 FP16 与 FlashKDA 完全对齐 |
| **Workspace 地址管理复杂化** | 低 | 新增 L_powWsAddr 和 tmpWsAddr，需同步管理 |
| **AIC/AIV 同步点增加** | 中 | 每次 Cube 后需 SyncAll，流水线中断 |

---

## 八、验证方案

### 8.1 单元测试

```python
# tests/pytest/chunk_gated_delta_rule_neumann_golden.py
# 验证 CHUNK=16 下的 Neumann 求逆精度

def test_chunk_16_neumann_inverse():
    chunk_size = 16
    # ... 构造测试数据 ...
    # 验证 INV = (I+L)⁻¹ 精度
```

### 8.2 端到端测试

```bash
# B=2, seqlen=64 (共 128 tokens, 8 chunks)
python3 generate_test_inputs.py  # chunk_size=16
./test_aclnn_chunk_gated_delta_rule_neumann
# 验证 max_abs_err < threshold
```

### 8.3 性能对比

| 配置 | CHUNK | 求逆方法 | 预期耗时 |
|------|-------|---------|---------|
| 原版 | 64 | 前代法 (31步×2 + 2 Cube) | 基线 |
| 新版 | 16 | Neumann (6 Cube + 3 Add) | 待测 |

---

## 九、参考文件路径

| 目的 | 文件路径 |
|------|---------|
| FlashKDA Neumann 实现 | `/mnt/workspace/gitCode/refresh/FlashKDA/csrc/smxx/utils.cuh` |
| 当前 Stage1 | `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` |
| 当前 Tiling | `op_host/chunk_gated_delta_rule_neumann_tiling.cpp` |
| 当前 Golden | `tests/pytest/chunk_gated_delta_rule_neumann_golden.py` |
| 方案分析文档 | `/mnt/workspace/gitCode/refresh/KDA/FlashKDA_方案分析与950适配规划.md` |

---

## 十、总结

| 维度 | 变化 |
|------|------|
| **chunkSize** | 64 → **16** |
| **求逆方法** | 前代法 → **Neumann 级数 3 步倍增** |
| **Stage2/3** | 无需修改逻辑，chunk 循环次数增加 |
| **精度策略** | FP16 MMA + FP16 加法 (与 FlashKDA 对齐) |
| **核心修改文件** | `*_tiling.cpp`, `*_stage1.h`, `*_golden.py` |

**预期收益**：
1. decay 值 bf16 安全范围增大
2. 求逆计算从 Vector 循环改为 Cube 加速
3. 与 FlashKDA 架构对齐，便于后续 per-dimension g 扩展

**预期代价**：
1. chunk 数增加 → 更多 SyncAll
2. 16×16 Cube 效率低于 64×64
3. 需重新验证精度和性能