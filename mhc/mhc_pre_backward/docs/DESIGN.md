# MhcPreBackward 设计文档

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | MhcPreBackward |
| 算子类别 | Fused Backward (MatMul + Elementwise + Reduction) |
| 支持数据类型 | x/gradHIn/gradX/gradXPostOptional: BF16/F16; 其余: F32 |
| **目标芯片** | Ascend910B (A2), Ascend910_93 (A3), Ascend950 (A5) |
| **目标架构** | arch22 (DAV_2201, A2/A3), arch35 (DAV_3510, A5) |

### 1.2 算子功能

MhcPreBackward 是 MhcPre 的反向算子，计算 mHC（Manifold-Constrained Hyper-Connections）架构中各参数的梯度。正向计算通过 phi 矩阵投影、RMSNorm 归一化、Sigmoid 门控和残差连接得到 $H^{in}$、$H^{pre}$、$H^{post}$、$H^{res}$；反向计算需要求解 $\text{grad\_x}$、$\text{grad\_phi}$、$\text{grad\_alpha}$、$\text{grad\_bias}$、$\text{grad\_gamma}$ 五个梯度。

### 1.3 数学公式

详见 [aclnnMhcPreBackward.md](aclnnMhcPreBackward.md)。核心反向计算公式：

**输出组合梯度**：
$$H\_pre\_grad = \text{Reduce}(H\_in\_grad \odot x, \text{dim}=-1), \quad x\_grad\_vec3 = H\_in\_grad \times H\_pre$$

**Sigmoid 门控反向**：
$$H\_pre\_2\_grad = H\_pre\_grad \odot s \odot (1-s), \quad H\_post\_2\_grad = H\_post\_grad \odot (H\_post \odot (1-H\_post/2))$$

**RMSNorm Fusion 反向**：
$$H\_mix\_grad = H\_mix\_tmp\_grad \cdot inv\_rms, \quad inv\_rms\_grad = \sum(H\_mix\_tmp\_grad \odot H\_mix)$$

**矩阵乘法反向**：
$$x\_rs\_grad = H\_mix\_grad @ \phi, \quad \phi\_grad = G^T @ X$$

**RMS 归一化梯度**：
$$x\_rs\_grad\_inv = -\frac{inv\_rms\_grad \cdot inv\_rms^3}{N \cdot D} \cdot x\_rs$$

**最终梯度组装**：
$$x\_grad = x\_grad\_vec3 + x\_grad\_vec1 + x\_grad\_mm$$

### 1.4 Shape 格式支持

| 格式 | 含义 | A2/A3 (arch22) | A5 (arch35) |
|------|------|:--------------:|:-----------:|
| BSNN | gradHRes=[B,S,N,N], factN=N² | ✓ | ✓ |
| BSN! | gradHRes=[B,S,N!], factN=N! | ✓ | ✗ |
| TNN | gradHRes=[T,N,N], factN=N² | ✓ | ✓ |
| TN! | gradHRes=[T,N!], factN=N! | ✓ | ✗ |

> fusionSize = factN + 2N，其中 factN=N²(BSNN/TNN) 或 N!(BSN!/TN!)

---

## 2. 架构设计

### 2.1 逻辑视图

**模块职责**：

| 模块 | 职责 | 核心文件 |
|------|------|---------|
| **op_api** | ACLNN 接口层：参数校验、Contiguous 转换、执行器构建 | `aclnn_mhc_pre_backward.cpp`, `mhc_pre_backward.h/cpp` |
| **op_host** | Host 侧逻辑：算子定义、Shape 推导、Tiling 切分 | `mhc_pre_backward_def.cpp`, `arch22/*_tiling.cpp`, `arch35/*_tiling.cpp` |
| **op_kernel** | Kernel 侧实现：AIV 向量计算 + AIC Cube 矩阵乘法 | `mhc_pre_backward.cpp`, `arch22/*_kernel*.h`, `arch35/*_kernel*.h` |

**模块依赖**：

```
aclnnMhcPreBackwardGetWorkspaceSize()
  ├──▶ op_api: 参数校验 (NotNull/Empty/Dim/Shape/Dtype/Format)
  ├──▶ op_api: Contiguous 转换
  ├──▶ op_host: l0op::MhcPreBackward() → TilingFunc()
  │     ├── arch22: TilingMhcPreBackwardArch22()
  │     └── arch35: TilingMhcPreBackwardArch35()
  └──▶ 返回 workspaceSize + executor

aclnnMhcPreBackward()
  └──▶ NPU Device 执行 Kernel
        ├── arch22: mhc_pre_backward<TILINGKEY>()
        └── arch35: mhc_pre_backward_apt<TILINGKEY>()
```

### 2.2 开发视图

```
mhc_pre_backward/
├── CMakeLists.txt
├── op_host/
│   ├── mhc_pre_backward_def.cpp          # 算子原型注册 (A2+A5)
│   ├── op_api/
│   │   ├── aclnn_mhc_pre_backward.h
│   │   ├── aclnn_mhc_pre_backward.cpp    # ACLNN 接口 + 参数校验
│   │   ├── mhc_pre_backward.h
│   │   └── mhc_pre_backward.cpp          # l0op::MhcPreBackward 实现
│   └── op_tiling/
│       ├── arch22/
│       │   ├── mhc_pre_backward_arch22_tiling.h
│       │   └── mhc_pre_backward_arch22_tiling.cpp  # A2/A3 Tiling
│       └── arch35/
│           ├── mhc_pre_backward_tiling.h
│           └── mhc_pre_backward_tiling.cpp         # A5 Tiling
├── op_kernel/
│   ├── mhc_pre_backward.cpp              # A2/A3 Kernel 入口
│   ├── mhc_pre_backward_apt.cpp          # A5 Kernel 入口
│   ├── arch22/
│   │   ├── mhc_pre_backward_kernel_arch22.h   # A2/A3 Kernel 主体
│   │   ├── mhc_pre_backward_data_arch22.h     # TilingData 结构体
│   │   └── mhc_pre_backward_key_arch22.h      # TilingKey 模板定义
│   └── arch35/
│       ├── mhc_pre_backward_kernel.h          # A5 Kernel 主体
│       └── mhc_pre_backward_utils.h           # A5 工具函数
└── docs/
    ├── aclnnMhcPreBackward.md            # API 接口文档
    ├── DESIGN.md                          # 本文档
    ├── DESIGN_A2_MIGRATION.md            # A2 迁移设计文档
    └── env.md                             # 环境配置
```

### 2.3 运行视图

**AIV-AIC 双核并行数据流**（arch22）：

```
            AIV (Vector 核心)                           AIC (Cube 核心)
    ┌─────────────────────────┐               ┌─────────────────────────┐
    │ GetHcScaleAndHcBase()   │               │                         │
    │         ↓               │               │                         │
    │ ComputeGradPre()        │               │                         │
    │   → x*grad_y → WS       │               │                         │
    │         ↓               │               │                         │
    │ ComputeGradHHat2()      │               │                         │
    │   → H_mix_grad → WS     │               │                         │
    │         ↓               │               │                         │
    │ CrossCoreSetFlag<0x2> ──────────────────▶│ WaitFlag<0x2>           │
    │         ↓               │               │         ↓               │
    │ WaitFlag<0x2> ◀──────────────────────────│ ProcessMatmul1()        │
    │                         │               │   x_rs_grad = G @ phi   │
    │                         │               │         ↓               │
    │                         │               │ ProcessMatmul2()        │
    │                         │               │   phi_grad = G^T @ X    │
    │                         │               │         ↓               │
    │                         │               │ CrossCoreSetFlag<0x2> ──▶
    │         ↓               │               │                         │
    │ ComputeGradX1()         │               │                         │
    │   x_grad = vec3+vec1+mm │               │                         │
    │         ↓               │               │                         │
    │ ComputeScaleBias()      │               │                         │
    │   → grad_alpha, grad_bias│              │                         │
    └─────────────────────────┘               └─────────────────────────┘
```

**执行流程**：

```
aclnnMhcPreBackwardGetWorkspaceSize()
     │
     ├──▶ MhcGradCheckParams(): 5 步校验 (NotNull → Empty → Dims → Shape → Dtype → Format)
     ├──▶ MhcGradCovertDataContiguous(): 非连续 Tensor 转连续
     ├──▶ l0op::MhcPreBackward() → TilingFunc() → 生成 TilingData
     │
     ▼
aclnnMhcPreBackward()
     │
     ▼
NPU Device 执行 (MIX_AIC_1_2: 1 AIC + 2 AIV per block)
     ├── AIV: 向量计算 (公式 1~5, 8)
     └── AIC: Cube 矩阵乘法 (公式 6: mm1 + mm2)
```

### 2.4 用户视图

| 调用方式 | 说明 |
|---------|------|
| **ACLNN 调用** | 两段式接口：`aclnnMhcPreBackwardGetWorkspaceSize()` → `aclnnMhcPreBackward()` |
| **图模式调用** | 通过 GE 图引擎调用，算子已注册 IR 原型 |

---

## 3. 实现方案 (arch22)

### 3.1 模板划分总览

**TilingKey 机制**：

| 参数 | 类型 | 取值范围 | 说明 |
|-----|------|---------|------|
| TILINGKEY | BOOL | {0, 1} | 0=非确定性, 1=确定性计算 |
| KERNEL_TYPE | - | MIX_AIC_1_2 | 1 AIC + 2 AIV 混合核 |

**模板划分表**：

| 模板 | 触发条件 | 模板参数 | 适用场景 |
|-----|---------|---------|---------|
| 非确定性 | `context->GetDeterministic() != 1` | `TILINGKEY=0` | 默认模式，AtomicAdd 累加 |
| 确定性 | `context->GetDeterministic() == 1` | `TILINGKEY=1` | 确定性计算，Workspace 暂存 + 多核归约 |

### 3.2 TilingData 结构体

**文件位置**：`op_kernel/arch22/mhc_pre_backward_data_arch22.h`

```cpp
struct MhcPreBackwardArch22TilingData {
    int64_t batchSize;     // B
    int64_t seqLength;     // S
    int64_t c;             // D (dimension)
    int64_t n;             // N (numsResidual, 固定为 4)
    int64_t fusionSize;    // factN + 2N (A2: N!+2N 或 N²+2N)
    int64_t isBSNN;        // 1=BSNN(N²), 0=BSN!(N!)
    int64_t c0;            // D 维分块大小 (256)
    int64_t c1;            // c / c0
    int64_t cTail;         // c % c0
    int64_t aivNum;        // AIV 核数
    int64_t tileSize;      // 每个 tile 的 BS 数 (32)
    int64_t ubSize;        // UB 大小
    float   eps;           // sigmoid eps
    TCubeTiling mm1TilingData;  // MatMul1 tiling
    TCubeTiling mm2TilingData;  // MatMul2 tiling
};
```

### 3.3 Host 侧 Tiling

**文件**: `op_host/op_tiling/arch22/mhc_pre_backward_arch22_tiling.cpp`

**核心流程**：

1. 从 x shape 读取 B/S/N/D
2. 从 phi shape 动态读取 fusionSize，校验为 N!+2N 或 N²+2N
3. 从 gradHRes 维度判断 BSNN(4D)/BSN!(3D)，设置 isBSNN
4. 计算 mm1/mm2 的 M/N/K 并生成 TCubeTiling
5. 计算 workspace 大小

**Workspace 布局**：

| 偏移 | 大小 | 用途 |
|------|------|------|
| 0 | B×S×fusionSize×4 | gradH2Global_ (H_mix_grad) |
| +offset1 | B×S×N×D×4 | xWorkspaceGlobal_ (x*grad_y) |
| +offset2 | B×S×N×D×4 | gradXCubeGlobal_ (x_rs_grad, mm1 结果) |
| +offset3 | aicNum×fusionSize×N×D×4 | gradWeightWSGlobal_ (确定性模式: phi_grad 暂存) |
| +offset4 | aivNum×(fusionSize+8)×4 | gradHcScale/BaseWSGlobal_ (确定性模式: α/bias 暂存) |

**MatMul 参数**：

| 参数 | mm1 (x_rs_grad = G @ phi) | mm2 (phi_grad = G^T @ X) |
|-----|--------------------------|--------------------------|
| M | tileSize×2 | fusionSize |
| N | N×D | N×D |
| K | fusionSize | B×S |

### 3.4 Kernel 侧实现

**文件**: `op_kernel/mhc_pre_backward.cpp` + `op_kernel/arch22/mhc_pre_backward_kernel_arch22.h`

#### 3.4.1 Kernel 模板实例化

```cpp
template<bool TILINGKEY>
__global__ __aicore__ void mhc_pre_backward(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 1 AIC + 2 AIV
    MhcPreBackwardKernelArch22<DTYPE_X, DTYPE_PHI, true> op;
    op.mm1_.Init(&tilingData.mm1TilingData, &pipe);
    op.mm2_.Init(&tilingData.mm2TilingData, &pipe);
    op.Init(...);
    op.Process();
}
```

#### 3.4.2 API 映射

| 计算步骤 | Ascend C API | 参数签名 | 平台验证 | 约束说明 |
|---------|-------------|---------|---------|---------|
| 数据搬入 (GM→UB) | DataCopyPad\<T\> | (dst, src, DataCopyExtParams, DataCopyPadExtParams) | ✅ DAV_2201 | 支持 padding；32B 对齐 |
| Cast (BF16→F32) | Cast\<T, TYPE_X\> | (dst, src, RoundMode, count) | ✅ DAV_2201 | CAST_NONE / CAST_RINT |
| 逐元素乘 | Mul\<T\> | (dst, src0, src1, count, repeatTimes, params) | ✅ DAV_2201 | 8 elem/block 对齐 |
| 逐元素加 | Add\<T\> | (dst, src0, src1, count) | ✅ DAV_2201 | - |
| Axpy | Axpy\<T\> | (dst, src, alpha, count) | ✅ DAV_2201 | dst += src * alpha |
| MulAddDst | MulAddDst\<T\> | (dst, src0, src1, mask, repeatTimes, params) | ✅ DAV_2201 | dst += src0 * src1 |
| 归约求和 | WholeReduceSum\<T\> | (dst, src, reduceSize, count, srcStride, dstStride, repStride) | ✅ DAV_2201 | 按 repeat 归约 |
| 广播 | Brcb\<T\> | (dst, src, count, params) | ✅ DAV_2201 | 1→8 广播 |
| 广播 (2D) | Broadcast\<T, 2, 1\> | (dst, src, dstShape, srcShape) | ✅ DAV_2201 | 任意维度广播 |
| 矩阵乘 | MatmulImpl::IterateAll\<true\> | (C) | ✅ DAV_2201 | MDL 配置, ND 格式 |
| 数据搬出 (UB→GM) | DataCopyPad\<T\> | (dst, src, DataCopyExtParams) | ✅ DAV_2201 | UB→GM |
| 原子加 | SetAtomicAdd\<T\> | () | ✅ DAV_2201 | 非确定性模式多核累加 |
| 跨核同步 | CrossCoreSetFlag/WaitFlag | (flagId) | ✅ DAV_2201 | AIV↔AIC 通信 |

#### 3.4.3 数据流设计

**AIV 侧计算流**：

```
1. GetHcScaleAndHcBase()
   GM(alpha) → UB(scaleLocal_)    构建 [α_pre, α_post, α_res] 块布局

2. ComputeGradPre(taskOffset, tileTaskCount, innerId)    [公式1]
   GM(x, grad_h_in) → UB → Cast → Mul → ReduceSum → UB(dPrePostTempLocal_)
   同时: x*grad_y → GM(xWorkspaceGlobal_)    供 AIC mm2 使用

3. ComputeGradHHat2(taskOffset, tileTaskCount)    [公式2-5,8]
   GM(grad_h_res, grad_h_post, h_mix, inv_rms, h_pre, h_post) → UB
   → Sigmoid 反向 → RMSNorm 反向 → H_mix_grad → GM(gradH2Global_)
   同时: 累加 α_grad/bias_grad 中间结果 (Kahan 求和)

4. [等待 AIC mm1/mm2 完成]

5. ComputeGradX1(taskOffset, tileTaskCount, innerId)    [公式1,8,6]
   GM(x, grad_h_in, x_rs_grad) → UB → Cast
   → x_grad_vec3 = grad_y * H_pre
   → x_grad_vec1 = x * x_rs_grad_inv (MulAddDst)
   → x_grad = vec3 + vec1 + mm
   → Cast → GM(gradX)

6. ComputeScaleBias()    [公式2,3,4]
   树形归约 bias_grad/α_grad → GM(grad_bias, grad_alpha)
```

**AIC 侧计算流**：

```
1. [等待 AIV workspace 数据就绪]

2. ProcessMatmul1(taskOffset, mm1M)    [公式6]
   A = gradH2Global_ (H_mix_grad) [M, K]
   B = weightGlobal_ (phi)         [K, N]
   C = gradXCubeGlobal_ (x_rs_grad) [M, N]

3. ProcessMatmul2(taskOffset, mm2K)    [公式6]
   A = gradH2Global_^T (H_mix_grad) [M, K]  (转置)
   B = xWorkspaceGlobal_ (x*grad_y) [K, N]
   C = gradWeightGlobal_ (phi_grad) [M, N]

4. [通知 AIV matmul 完成]
```

#### 3.4.4 内存管理

**UB Buffer 分配** (arch22, DAV_2201, 192KB UB)：

| Buffer | 大小计算 | 用途 |
|--------|---------|------|
| fusedGradHPre2AndGradHPost2Buf_ | tileSize×N×2×4 | H_pre_grad + H_post_grad 合并 |
| gradRsqrtBuf_ | tileSize×fusionSize×4×2 | inv_rms_grad (双缓冲) |
| gradBiasBuf_ | 2×2×tileSize×fusionSize×4 | bias_grad + α_grad (Kahan 双缓冲) |
| onesBuf_ | tileSize×N×2×4 | Ones 向量 |
| hcBaseBuf_ | fusionSize×4 | bias 参数 |
| ScaleBuf_ | 32×2×4 | α 参数 (block 布局) |
| inputXInQueue | 2 × tileSize×N×c0×4/4 | 输入数据队列 |
| inputGradQueue | 1 × tileSize×8×4 | 梯度输入队列 |
| OutQueue | 2 × tileSize×N×c0×4/8 | 输出数据队列 |
| tempBuf_ | 剩余空间 | 临时缓冲区 (dhat, rsqrt, preBrcb 等) |

#### 3.4.5 UB 容量验证

| 平台 | 总 UB | 系统保留 | 可用 UB | 验证 |
|------|-------|---------|---------|------|
| DAV_2201 (A2/A3) | 192 KB | 8 KB | 184 KB | `sum(buffers) ≤ 184 * 1024` ✅ |

---

## 4. 性能优化

### 4.1 AIV-AIC 双核并行

采用 `MIX_AIC_1_2` 混合核模式，每个 block 包含 1 个 AIC + 2 个 AIV：

- **AIV** 负责逐元素计算和归约（Sigmoid 反向、RMSNorm 反向、梯度组装）
- **AIC** 负责两个 Cube 矩阵乘法（x_rs_grad = G@phi, phi_grad = G^T@X）
- 通过 `CrossCoreSetFlag` / `CrossCoreWaitFlag` 实现 AIV↔AIC 同步

### 4.2 Kahan 求和

跨 tile 的 `α_grad` / `bias_grad` 累加使用 Kahan 求和算法（`kahanCustom`），补偿浮点累加误差，提升精度。

### 4.3 树形归约

`ComputeScaleBias()` 中对 `bias_grad` 和 `α_grad` 的跨 batch*seq 归约使用树形归约（二分法累加），减少串行归约步数。

### 4.4 数据布局优化

`scaleLocal_` 按 block(8 elem) 组织：`[α_pre(N), α_post(N), α_res_block0(8), α_res_block1(8), ...]`，使得 res 区段的多个 block 可以通过循环统一处理，支持任意 N! 值。

### 4.5 D 维分块

D 维度按 `c0=256` 分块处理，支持 `cTail` 非 256 对齐的尾部处理。`ComputeGradPre` 中使用手动树形归约对 D 维做 reduce sum。

---

## 5. 参数校验设计

### 5.1 校验流程

```
MhcGradCheckParams()
  ├── 1. MhcGradCheckNotNull()        — 空指针校验
  ├── 2. MhcGradCheckEmptyTensor()     — 空 Tensor 校验
  ├── 3. MhcGradCheckInputOutDims()    — 维数校验 (CheckDimNum)
  ├── 4. MhcGradCheckInputOutShape()   — Shape 校验
  │     ├── MhcGradCheckBSNDShape()    — BSND 路径 (x 4D)
  │     │   ├── BSNN: gradHRes 4D [B,S,N,N] — 全平台
  │     │   └── BSN!: gradHRes 3D [B,S,N!] — 仅 A2/A3 (IsAscend910BPlatform)
  │     ├── MhcGradCheckTNDShape()     — TND 路径 (x 3D)
  │     │   ├── TNN: gradHRes 3D [T,N,N] — 全平台
  │     │   └── TN!: gradHRes 2D [T,N!]  — 仅 A2/A3
  │     └── MhcGradCheckCommonShape()  — 公共校验 (phi, alpha, gamma 等)
  ├── 5. MhcGradCheckDtypeValid()      — 数据类型校验
  └── 6. MhcGradCheckFormat()          — 格式校验 (必须 ND)
```

### 5.2 芯片判断

```cpp
bool IsAscend910BPlatform() {
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    return socVersion == SocVersion::ASCEND910B ||
           socVersion == SocVersion::ASCEND910_93;
}
```

| SocVersion | 架构代号 | 芯片系列 | BSN!/TN! 支持 |
|------------|---------|---------|:-------------:|
| ASCEND910B | DAV_2201 | Atlas A2 | ✓ |
| ASCEND910_93 | DAV_2201 | Atlas A3 | ✓ |
| ASCEND950 | DAV_3510 | Atlas A5 | ✗ |

### 5.3 fusionSize 校验

phi 第一维必须为 `N!+2N` 或 `N²+2N`：

```cpp
uint64_t expectedFusionSizeFactN = Factorial(N) + 2 * N;  // N!+2N
uint64_t expectedFusionSizeNsq = N * N + 2 * N;             // N²+2N
// fusionSize 必须等于两者之一
```

---

## 6. 风险评估

### 6.1 精度风险

| 风险 | 说明 | 应对措施 |
|------|------|---------|
| 浮点累加误差 | 跨 tile 的 α_grad/bias_grad 累加 | Kahan 求和补偿 |
| BF16 精度损失 | x/gradHIn 为 BF16，Cast 到 F32 计算 | 全程 F32 计算，仅最终结果 Cast 回 BF16 |
| AtomicAdd 非确定性 | 非确定性模式多核 AtomicAdd 顺序不确定 | 提供确定性模式 (TILINGKEY=1) |

### 6.2 性能风险

| 风险 | 说明 | 应对措施 |
|------|------|---------|
| AIV-AIC 串行等待 | AIV 和 AIC 之间有同步屏障 | 乒乓流水优化 (规划中) |
| 大 BS 场景 mm2 K 轴过大 | mm2 的 K=B×S，大场景下 K 很大 | matmul K 轴分块优化 (规划中) |

### 6.3 兼容性风险

| 风险 | 说明 | 应对措施 |
|------|------|---------|
| N! 溢出 | N=8 时 N!=40320，大 N 场景 factN 过大 | Tiling 阶段校验 fusionSize 合法性 |
| D 不对齐 | D 非 128 对齐 | Tiling 阶段校验 `c % 128 == 0` |

---

## 7. 交付件清单

| 类型 | 文件 | 状态 |
|------|------|------|
| **op_host** | `mhc_pre_backward_def.cpp` | ✅ |
| **op_host** | `op_api/aclnn_mhc_pre_backward.cpp` | ✅ |
| **op_host** | `op_api/mhc_pre_backward.cpp` | ✅ |
| **op_host** | `op_tiling/arch22/mhc_pre_backward_arch22_tiling.cpp` | ✅ |
| **op_host** | `op_tiling/arch35/mhc_pre_backward_tiling.cpp` | ✅ |
| **op_kernel** | `mhc_pre_backward.cpp` (A2/A3 入口) | ✅ |
| **op_kernel** | `mhc_pre_backward_apt.cpp` (A5 入口) | ✅ |
| **op_kernel** | `arch22/mhc_pre_backward_kernel_arch22.h` | ✅ |
| **op_kernel** | `arch35/mhc_pre_backward_kernel.h` | ✅ |
| **文档** | `docs/aclnnMhcPreBackward.md` | ✅ |
| **文档** | `docs/DESIGN.md` (本文档) | ✅ |

---

## 8. 修订记录

| 版本 | 日期 | 修改人 | 修改内容 |
|------|------|--------|---------|
| 1.0 | 2026-07-23 | - | 初始版本，包含 arch22 (A2/A3) 实现设计 |
