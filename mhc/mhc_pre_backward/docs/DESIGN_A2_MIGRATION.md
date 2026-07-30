# MhcPreBackward A2 迁移与 Phi Shape 变更设计文档

## 1. 背景与目标

### 1.1 现状

| 算子 | 平台支持 | fusionSize | res 段含义 |
|------|---------|-----------|-----------|
| `mhc_pre_sinkhorn_backward` | ascend910b (A2) + ascend950 (A3) | `N^2 + 2N` | N×N Sinkhorn 归一化矩阵 |
| `mhc_pre_backward` | **仅 ascend950 (A3)** | `N^2 + 2N` | N×N 矩阵（无 Sinkhorn） |

### 1.2 目标

1. **A2 平台迁移**：将 `mhc_pre_sinkhorn_backward` 的 A2 (arch22) 实现迁移到 `mhc_pre_backward`
2. **删除 Sinkhorn 代码**：移除 `SinkhornGrad`、`sum_out`/`norm_out` 输入、`skIterCount` 等 Sinkhorn 相关逻辑
3. **Phi shape 变更**：`fusionSize` 从 `N^2 + 2N` 变为 `N! + 2N`，res 段从 N×N 矩阵变为 N! 排列向量

### 1.3 变更后预期

| 算子 | 平台支持 | fusionSize (N=4) | res 段含义 |
|------|---------|-----------------|-----------|
| `mhc_pre_backward` | **ascend910b (A2)** + ascend950 (A3) | `4! + 2×4 = 32` | N! 排列向量（无 Sinkhorn） |

---

## 2. 两个算子接口对比

### 2.1 输入差异

| 输入 | mhc_pre_backward | mhc_pre_sinkhorn_backward (A2) | 迁移后 mhc_pre_backward (A2) |
|------|-----------------|-------------------------------|------------------------------|
| x | ✅ | ✅ | ✅ |
| phi | ✅ | ✅ | ✅ |
| alpha | ✅ | ✅ | ✅ |
| grad_h_in | ✅ | ✅ (grad_hin) | ✅ |
| grad_h_post | ✅ | ✅ | ✅ |
| grad_h_res | ✅ | ✅ | ✅ |
| inv_rms | ✅ | ✅ | ✅ |
| h_mix | ✅ | ❌ | ✅ (对应 sinkhorn 的 hc_before_norm) |
| h_pre | ✅ | ✅ | ✅ |
| h_post | ✅ | ❌ | ✅ |
| gamma | ✅ (optional) | ❌ | ✅ (optional) |
| grad_x_post | ✅ (optional) | ❌ | ✅ (optional) |
| bias | ❌ | ✅ | ❌ (A2 kernel 内部不使用 bias 输入) |
| hc_before_norm | ❌ | ✅ | ❌ (用 h_mix 替代) |
| sum_out | ❌ | ✅ | ❌ (**删除**) |
| norm_out | ❌ | ✅ | ❌ (**删除**) |

### 2.2 输出差异

| 输出 | mhc_pre_backward | mhc_pre_sinkhorn_backward (A2) | 迁移后 |
|------|-----------------|-------------------------------|--------|
| grad_x | ✅ | ✅ | ✅ |
| grad_phi | ✅ | ✅ | ✅ |
| grad_alpha | ✅ | ✅ | ✅ |
| grad_bias | ✅ | ✅ | ✅ |
| grad_gamma | ✅ (optional) | ❌ | ✅ (optional) |

### 2.3 参数映射关系

迁移时 A2 kernel 的参数映射：

```
// sinkhorn_backward A2 Init 参数        →  mhc_pre_backward A2 Init 参数
x           → x                           x           → x
hc_fn       → weightGlobal_ (phi)         phi         → weightGlobal_ (phi)
pre         → preGlobal_ (h_pre)          h_pre       → preGlobal_ (h_pre)
grad_y      → gradYGlobal_ (grad_h_in)    grad_h_in   → gradYGlobal_
grad_post   → gradPostGlobal_             grad_h_post → gradPostGlobal_
grad_comb   → gradHResGlobal_ (grad_h_res) grad_h_res → gradHResGlobal_
hc_scale    → hcScaleGlobal_ (alpha)      alpha       → hcScaleGlobal_
hc_base     → hcBaseGlobal_ (bias)        h_mix       → h2Global_ (hc_before_norm)
h_hat2      → h2Global_ (hc_before_norm)  inv_rms     → rsqrtGlobal_
rsqrt       → rsqrtGlobal_ (inv_rms)
sum_out     → skSumGlobal_  (**删除**)
norm_out    → skNormGlobal_ (**删除**)
```

---

## 3. Sinkhorn 代码删除策略

### 3.1 需要完全删除的代码

| 组件 | 位置 | 删除内容 |
|------|------|---------|
| **SinkhornGrad 函数** | kernel L374-561 | 整个函数体（~190 行） |
| **SKInQueue** | kernel L113 | Sinkhorn 数据输入队列 |
| **skNormGlobal_** | kernel L142 | norm_out GlobalTensor |
| **skSumGlobal_** | kernel L143 | sum_out GlobalTensor |
| **skIterCount_** | kernel L99 | Sinkhorn 迭代次数 |
| **BSNN/BSN** | kernel L164-165 | Sinkhorn 相关偏移量 |
| **MASK_PRE/MASK_POST** | kernel L33-34 | Sinkhorn BlockReduceSum 掩码 |
| **gradHResTempLocal_/gradHResTempLocal2_** | kernel L312-315 | Sinkhorn 临时 buffer |
| **tilingData.skIterCount** | tiling data L33 | 迭代次数字段 |
| **sum_out/norm_out** | def.cpp, tiling.cpp | 输入定义和 shape 校验 |

### 3.2 ComputeGradHHat2 中的变更

**原流程**（sinkhorn_backward A2）：
```
1. SinkhornGrad(taskOffset, tileTaskCount)  → gradHResLocal_ 经过 Sinkhorn 反向变换
2. Copy(dhatLocal_[res段], gradHResLocal_)  → 将变换结果放入 dhatLocal_
3. ... 后续 RMSNorm/Sigmoid 反向计算
```

**新流程**（mhc_pre_backward A2，无 Sinkhorn）：
```
1. DataCopyPad(gradHResLocal_, gradHResGlobal_[taskOffset * N!])
   → 直接加载 grad_h_res，无需变换
2. Copy(dhatLocal_[res段], gradHResLocal_)
   → 将 grad_h_res 直接放入 dhatLocal_ 的 res 段
3. ... 后续 RMSNorm/Sigmoid 反向计算（保持不变）
```

### 3.3 res 段梯度计算的简化

在 sinkhorn 版本中，res 段的前向计算为：
```
z_res → exp/softmax → Sinkhorn(20轮) → h_res
```
反向为：
```
grad_h_res → SinkhornGrad → ExpGrad → grad_z_res
```

在 mhc_pre_backward（无 Sinkhorn）中，res 段的前向计算为：
```
z_res → (直接输出) → h_res
```
反向为：
```
grad_h_res → (恒等变换) → grad_z_res
```

因此 `grad_z_res = grad_h_res`，直接拷贝即可。

---

## 4. Phi Shape 变更：N^2 → N!

### 4.1 变更定义

```cpp
// 变更前
fusionSize = 2 * N + N * N;

// 变更后
fusionSize = 2 * N + Factorial(N);
```

### 4.2 N=4 时的数值变化

| 属性 | 变更前 (N^2) | 变更后 (N!) |
|------|-------------|-------------|
| res 段大小 | 4×4 = 16 | 4! = 24 |
| pre 段大小 | 4 (padded 8) | 4 (padded 8) |
| post 段大小 | 4 (padded 8) | 4 (padded 8) |
| fusionSize | 24 | 32 |
| UB 块数 (8 elem/block) | 3 | 4 |
| phi shape | (24, N·C) | (32, N·C) |

### 4.3 数据布局对比 (N=4)

```
变更前 (fusionSize=24, 3 blocks):
┌──────────┬──────────┬──────────────────────┐
│  pre (8) │ post (8) │      res (16)        │
│  [0..7]  │  [8..15] │     [16..23]          │
└──────────┴──────────┴──────────────────────┘
  block 0    block 1    block 2

变更后 (fusionSize=32, 4 blocks):
┌──────────┬──────────┬──────────────────────────────────┐
│  pre (8) │ post (8) │           res (24)                │
│  [0..7]  │  [8..15] │          [16..31]                  │
└──────────┴──────────┴──────────────────────────────────┘
  block 0    block 1    block 2    block 3
```

### 4.4 影响范围

#### 4.4.1 全局影响（所有平台）

| 文件 | 变更点 |
|------|--------|
| `op_host/mhc_pre_backward_infershape.cpp` L117 | `(2*N) + (N*N)` → `(2*N) + Factorial(N)` |
| `op_host/op_tiling/arch35/mhc_pre_backward_tiling.cpp` L220 | `fusionSize_ = (2*N_) + (N_*N_)` → `(2*N_) + Factorial(N_)` |
| `op_kernel/arch35/mhc_pre_backward_kernel.h` L289 | `gradHResBufLen_ = vecDealChunk_ * N_ * N_` → `vecDealChunk_ * Factorial(N_)` |
| `op_kernel/arch35/mhc_pre_backward_utils.h` L124 | `fusionSize = 0; // 2N + N^2` → `// 2N + N!` |
| UT 测试文件 | `fusionSize = n * n + 2 * n` → `Factorial(n) + 2 * n` |

#### 4.4.2 A2 平台特有影响

A2 kernel 中所有使用 `n_ * n_` 或 `n_ * n_ + PRE_POST_NUM * n_` 的位置需要变更为 `Factorial(n_)` 或 `Factorial(n_) + PRE_POST_NUM * n_`。

**关键硬编码值变更**：

| 硬编码位置 | 变更前 (N=4) | 变更后 (N=4) |
|-----------|-------------|-------------|
| repeat stride `{3, 0, 3, 24, 0, 24}` | 3 blocks, stride 24 | 4 blocks, stride 32 |
| `MASK_PRE` / `MASK_POST` | Sinkhorn 专用，删除 | 删除 |
| Buffer 大小 `tileCoreBS_ * (n_*n_ + 2*n_)` | `tileCoreBS_ * 24` | `tileCoreBS_ * 32` |
| DataCopy 长度 `(n_*n_ + 2*n_) * sizeof(T)` | `24 * sizeof(T)` | `32 * sizeof(T)` |
| Matmul mm1K / mm2M | `N^2 + 2N = 24` | `N! + 2N = 32` |

### 4.5 N 值约束

| N | N^2 | N! | 可行性 |
|---|-----|----|--------|
| 4 | 16 | 24 | ✅ 可行 (A2 + A3) |
| 6 | 36 | 720 | ⚠️ 仅 A3 理论可行，UB 可能不足 |
| 8 | 64 | 40320 | ❌ 不可行 |

**决策**：A2 平台仅支持 N=4（与 sinkhorn_backward A2 一致）。A3 平台保留 N=4,6,8 支持，但 N=6,8 的 N! 值过大时由 tiling 层报错拦截。

### 4.6 Factorial 函数定义

在 `common.h` 中添加：

```cpp
inline int64_t Factorial(int64_t n)
{
    if (n <= 1) return 1;
    int64_t result = 1;
    for (int64_t i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}
```

---

## 5. A2 实现详细设计

### 5.1 新增文件清单

```
op_kernel/arch22/
├── mhc_pre_backward_kernel_arch22.h     # A2 kernel 实现（从 mhc_pre_grad_kernel.h 迁移改造）
├── mhc_pre_backward_data_arch22.h       # tiling data 结构体
└── mhc_pre_backward_key_arch22.h        # tiling key 模板参数

op_host/op_tiling/arch22/
├── common.h                              # 通用工具函数（含 Factorial）
├── mhc_pre_backward_arch22_tiling.h      # tiling 函数声明
└── mhc_pre_backward_arch22_tiling.cpp    # tiling 函数实现
```

### 5.2 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `op_host/mhc_pre_backward_def.cpp` | 添加 `ascend910b` AICore 配置 |
| `op_kernel/mhc_pre_backward_apt.cpp` | 添加 `__CCE_AICORE__ == 220` 条件分支，引入 A2 kernel |
| `op_host/mhc_pre_backward_infershape.cpp` | `N*N` → `Factorial(N)` |
| `op_host/op_tiling/arch35/mhc_pre_backward_tiling.cpp` | `fusionSize_` 计算改为 `Factorial(N)` |
| `op_kernel/arch35/mhc_pre_backward_kernel.h` | `N_*N_` → `Factorial(N_)` |
| `op_kernel/arch35/mhc_pre_backward_utils.h` | 注释更新 |
| UT 测试文件 | `fusionSize` 计算更新 |

### 5.3 A2 Kernel 类设计

```cpp
template <typename TYPE_X, typename T, bool DETERMINISTIC>
class MhcPreBackwardKernelArch22 {
public:
    // Matmul 类型定义（与 sinkhorn 版本一致）
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using CType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<A0Type, BType, CType, CType, MHC_PRE_GRAD_MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, CType, CType, MHC_PRE_GRAD_MM2_CFG> mm2_;

    // Init 签名（去除 sum_out, norm_out）
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR h_pre, GM_ADDR grad_h_in, GM_ADDR grad_h_post,
        GM_ADDR grad_h_res, GM_ADDR alpha, GM_ADDR h_mix, GM_ADDR inv_rms,
        GM_ADDR grad_x, GM_ADDR grad_phi, GM_ADDR grad_alpha, GM_ADDR grad_bias,
        GM_ADDR workspace, const MhcPreBackwardArch22TilingData *tilingData, TPipe *pipe);

    __aicore__ inline void Process();

protected:
    // === 保留的成员函数 ===
    __aicore__ inline void ComputeGradPre(...);      // grad_h_pre 计算
    __aicore__ inline void ComputeGradHHat2(...);    // HHat2 梯度（去除 SinkhornGrad 调用）
    __aicore__ inline void ComputeGradX1(...);       // grad_x 计算
    __aicore__ inline void GetHcScaleAndHcBase();    // 加载 alpha 和 bias
    __aicore__ inline void ProcessMatmul1(...);      // mm1: grad_h2 @ phi^T → grad_x_cube
    __aicore__ inline void ProcessMatmul2(...);      // mm2: grad_h2^T @ x_workspace → grad_phi
    __aicore__ inline void ComputeScaleBias();       // alpha/bias 梯度归约
    __aicore__ inline void ComputeDeterministic(...);// deterministic 归约

    // === 删除的成员函数 ===
    // __aicore__ inline void SinkhornGrad(...);  ← 完全删除

    // === 删除的成员变量 ===
    // GlobalTensor<T> skNormGlobal_, skSumGlobal_;
    // TQue<...> SKInQueue;
    // int64_t skIterCount_, BSNN, BSN;
    // LocalTensor<T> gradHResTempLocal_, gradHResTempLocal2_;
};
```

### 5.4 Tiling Data 结构体

```cpp
struct MhcPreBackwardArch22TilingData {
    int64_t batchSize = 0;
    int64_t seqLength = 0;
    int64_t c = 0;
    int64_t n = 0;
    int64_t c0 = 0;
    int64_t c1 = 0;
    int64_t cTail = 0;
    int64_t aivNum = 0;
    int64_t tileGradY = 0;
    int64_t tileHHat2 = 0;
    int64_t tileSize = 0;
    // int64_t skIterCount = 0;  ← 删除
    int64_t ubSize = 0;
    float eps = 0;
    TCubeTiling mm1TilingData;
    TCubeTiling mm2TilingData;
};
```

### 5.5 Tiling 计算变更

```cpp
// 变更前
int64_t mm1K = n * n + 2 * n;        // 24
int64_t mm2M = n * n + 2 * n;        // 24

// 变更后
int64_t factN = Factorial(n);         // 24 (for n=4)
int64_t mm1K = factN + 2 * n;         // 32
int64_t mm2M = factN + 2 * n;         // 32
```

### 5.6 Shape 校验变更

```cpp
// 变更前
OP_CHECK_IF(phiShape.GetDim(0) != n * n + 2 * n || phiShape.GetDim(1) != n * c, ...);

// 变更后
int64_t factN = Factorial(n);
OP_CHECK_IF(phiShape.GetDim(0) != factN + 2 * n || phiShape.GetDim(1) != n * c, ...);

// grad_h_res shape 变更
// 变更前: (B, S, N, N) 或 (B, S, N*N)
// 变更后: (B, S, N!)  — 一维排列向量
OP_CHECK_IF(gradHResShape.GetDim(2) != factN, ...);
```

### 5.7 Kernel 入口文件变更

当前 `mhc_pre_backward_apt.cpp` 只有 arch35 路径。需要新增 arch22 入口。

**方案**：将 `mhc_pre_backward_apt.cpp` 改为根据 `__CCE_AICORE__` 宏条件编译：

```cpp
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
// arch35 路径（现有代码不变）
#include "arch35/mhc_pre_backward_kernel.h"
// ...

#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
// arch22 路径（新增）
#include "arch22/mhc_pre_backward_kernel_arch22.h"
#include "arch22/mhc_pre_backward_data_arch22.h"
#include "arch22/mhc_pre_backward_key_arch22.h"

template<bool TILINGKEY>
__global__ __aicore__ void mhc_pre_backward(
    GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR grad_h_in,
    GM_ADDR grad_h_post, GM_ADDR grad_h_res, GM_ADDR inv_rms,
    GM_ADDR h_mix, GM_ADDR h_pre, GM_ADDR h_post, GM_ADDR gamma,
    GM_ADDR grad_x_post_optional, GM_ADDR grad_x, GM_ADDR grad_phi,
    GM_ADDR grad_alpha, GM_ADDR grad_bias, GM_ADDR grad_gamma,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // A2 kernel 入口逻辑
    // ...
}
#endif
```

### 5.8 Def 文件变更

```cpp
// 新增 ascend910b 配置
OpAICoreConfig aicore_Config_910b;
aicore_Config_910b.DynamicCompileStaticFlag(true)
    .DynamicFormatFlag(false)
    .DynamicRankSupportFlag(true)
    .DynamicShapeSupportFlag(true)
    .NeedCheckSupportFlag(false)
    .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
    .ExtendCfgInfo("opFile.value", "mhc_pre_backward");
this->AICore().AddConfig("ascend910b", aicore_Config_910b);
```

### 5.9 ComputeGradHHat2 改造详细

**删除的步骤**：
- `SinkhornGrad(taskOffset, tileTaskCount)` 调用
- Sinkhorn 结果到 dhatLocal_ res 段的 Copy（使用 N×N 矩阵重排）

**新增的步骤**：
- 直接 DataCopyPad 加载 grad_h_res（一维 N! 向量）
- 直接 Copy 到 dhatLocal_ 的 res 段（无需矩阵重排）

**代码对比**：

```cpp
// ===== 变更前（sinkhorn_backward A2）=====
SinkhornGrad(taskOffset, tileTaskCount);
// 将 N×N 矩阵结果重排到 dhatLocal_ res 段
for (int32_t loopIdN = 0; loopIdN < 2; loopIdN++) {
    Copy(dhatLocal_[ELEMENTS_SIZE_PER_BLOCK + loopIdN * ELEMENTS_SIZE_PER_BLOCK],
         gradHResLocal_[loopIdN * ELEMENTS_SIZE_PER_BLOCK], ...);
}

// ===== 变更后（mhc_pre_backward A2）=====
// 直接加载 grad_h_res（一维 N! 向量，无需 Sinkhorn 变换）
DataCopyPad(gradHResLocal_, gradHResGlobal_[taskOffset * factN_],
            {static_cast<uint16_t>(1),
             static_cast<uint32_t>(tileTaskCount * factN_ * sizeof(T)), 0, 0, 0},
            {false, 0, 0, 0});
// 直接 Copy 到 res 段（block 2 和 block 3）
// res 段从 block 2 开始（offset = 2 * 8 = 16 elements）
Copy(dhatLocal_[2 * ELEMENTS_SIZE_PER_BLOCK],
     gradHResLocal_, factN_ * tileTaskCount, 1, {1, 1, 8, 8});
```

### 5.10 Process 流程对比

```
变更前（sinkhorn A2 Process）:
┌─────────────────────────────────────────────────┐
│ AIV:                                             │
│   GetHcScaleAndHcBase()                          │
│   for each task:                                 │
│     ComputeGradPre()         ← grad_h_pre       │
│     ComputeGradHHat2():                          │
│       ├─ SinkhornGrad()      ← ★ 删除            │
│       ├─ 加载 grad_h_post                        │
│       ├─ Sigmoid 反向                             │
│       ├─ RMSNorm 反向                             │
│       ├─ Kahan 累加 grad_bias/grad_alpha          │
│       └─ 输出 dhat2 → GM                         │
│     CrossCoreSetFlag → AIC                       │
│     ComputeGradX1()          ← grad_x            │
│   ComputeScaleBias()         ← grad_alpha/bias   │
│                                                  │
│ AIC:                                             │
│   for each task:                                 │
│     CrossCoreWaitFlag                            │
│     ProcessMatmul1()          ← mm1              │
│     ProcessMatmul2()          ← mm2              │
│     CrossCoreSetFlag → AIV                       │
└─────────────────────────────────────────────────┘

变更后（mhc_pre_backward A2 Process）:
┌─────────────────────────────────────────────────┐
│ AIV:                                             │
│   GetHcScaleAndHcBase()                          │
│   for each task:                                 │
│     ComputeGradPre()         ← grad_h_pre       │
│     ComputeGradHHat2():                          │
│       ├─ 直接加载 grad_h_res（无 Sinkhorn）       │
│       ├─ 加载 grad_h_post                        │
│       ├─ Sigmoid 反向                             │
│       ├─ RMSNorm 反向                             │
│       ├─ Kahan 累加 grad_bias/grad_alpha          │
│       └─ 输出 dhat2 → GM                         │
│     CrossCoreSetFlag → AIC                       │
│     ComputeGradX1()          ← grad_x            │
│   ComputeScaleBias()         ← grad_alpha/bias   │
│                                                  │
│ AIC:                                             │
│   for each task:                                 │
│     CrossCoreWaitFlag                            │
│     ProcessMatmul1()          ← mm1              │
│     ProcessMatmul2()          ← mm2              │
│     CrossCoreSetFlag → AIV                       │
└─────────────────────────────────────────────────┘
```

---

## 6. UB 内存布局变更 (N=4)

### 6.1 Buffer 大小变更

| Buffer | 变更前 (N^2+2N=24) | 变更后 (N!+2N=32) |
|--------|-------------------|-------------------|
| gradRsqrtBuf_ | `tile * 24 * sizeof(float) * 2` | `tile * 32 * sizeof(float) * 2` |
| gradBiasBuf_ | `2 * 2 * tile * 24 * sizeof(float)` | `2 * 2 * tile * 32 * sizeof(float)` |
| hcBaseBuf_ | `24 * sizeof(float)` | `32 * sizeof(float)` |
| dhatLocal_ | `tile * 24 * 2` | `tile * 32 * 2` |
| rsqrtbrcbLocal_ | `brcbAlign * 8 * 24` | `brcbAlign * 8 * 32` |
| hat2Scale | `tile * 24` | `tile * 32` |
| dhatBeforeNormLocal | `tile * 24` | `tile * 32` |
| hatLocal | `tile * 24` | `tile * 32` |
| rsqrtTempLocal_ | `tile * 24` | `tile * 32` |

### 6.2 Repeat Stride 变更

所有使用 `{3, 0, 3, 24, 0, 24}` 的 repeat stride 参数变更为 `{4, 0, 4, 32, 0, 32}`：
- `3` → `4`：block 数从 3 变为 4
- `24` → `32`：fusionSize 从 24 变为 32

涉及函数：`ComputeGradHHat2` 中的 `Mul`、`Add`、`Muls` 调用。

---

## 7. Matmul 形状变更

| 参数 | 变更前 | 变更后 (N=4) |
|------|--------|-------------|
| mm1M | `tile * 2` | `tile * 2` (不变) |
| mm1N | `N * C` | `N * C` (不变) |
| mm1K | `N^2 + 2N = 24` | `N! + 2N = 32` |
| mm2M | `N^2 + 2N = 24` | `N! + 2N = 32` |
| mm2N | `N * C` | `N * C` (不变) |
| mm2K | `B * S` | `B * S` (不变) |

---

## 8. 风险与约束

### 8.1 UB 容量风险

fusionSize 从 24 增大到 32（+33%），所有相关 UB buffer 同比增大。需验证 tile=32 时 UB 是否充足。

A2 (ascend910b) UB 大小通常为 192KB。变更后主要 buffer 占用估算（tile=32, N=4, fusionSize=32）：
- gradBiasBuf_: `2 * 2 * 32 * 32 * 4 = 16KB`
- gradRsqrtBuf_: `32 * 32 * 4 * 2 = 8KB`
- dhatLocal_ 等 tempBuf_ 子分配: 约 80KB
- 其他 buffer: 约 40KB
- **总计约 144KB < 192KB** ✅ 可行

### 8.2 N 值约束

- A2 平台：仅支持 N=4（N!=24，fusionSize=32）
- A3 平台：N=4 可行；N=6 (N!=720) 和 N=8 (N!=40320) 需 tiling 层报错拦截

### 8.3 grad_h_res shape 变更

- 变更前：`(B, S, N, N)` 或 `(B, S, N^2)` — 4D 或 3D
- 变更后：`(B, S, N!)` — 3D（一维排列向量，无矩阵结构）
- Shape 校验需更新对应逻辑

---

## 9. 实施计划

### 阶段 1：A2 代码迁移与 Sinkhorn 删除

1. 创建 `op_kernel/arch22/` 目录及三个头文件
2. 从 `mhc_pre_grad_kernel.h` 迁移代码，删除 Sinkhorn 相关部分
3. 修改 `ComputeGradHHat2` 中的 res 段加载逻辑
4. 创建 `op_host/op_tiling/arch22/` 目录及 tiling 文件

### 阶段 2：Phi Shape 变更

1. 在 `common.h` 中添加 `Factorial` 函数
2. 修改 `infershape.cpp` 中的 fusionSize 计算
3. 修改 arch35 tiling 和 kernel 中的 `N*N` → `Factorial(N)`
4. 修改 A2 tiling 和 kernel 中的 `n_*n_` → `Factorial(n_)`
5. 修改所有硬编码的 stride/size 值

### 阶段 3：入口与注册

1. 修改 `def.cpp` 添加 `ascend910b` 配置
2. 修改 `apt.cpp` 添加 arch22 条件分支
3. 更新 UT 测试中的 fusionSize 计算

---

## 10. 验证标准

- [ ] A2 平台编译通过（`ascend910b` 配置）
- [ ] A3 平台编译通过（`ascend950` 配置，无回归）
- [ ] UT 测试通过（infershape + tiling）
- [ ] phi shape 正确变为 `(N! + 2N, N*C)`
- [ ] Sinkhorn 相关代码完全清除（无 sum_out/norm_out/SinkhornGrad 残留）
