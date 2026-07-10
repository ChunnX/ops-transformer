# AllToAll + Matmul 融合算子设计文档

## 一、场景描述

### 1.1 业务场景
K维度分布式矩阵乘法：
- 本地数据：每个rank持有所有rank的K片段，布局为 [rank0块][rank1块][rank2块][rank3块]，总大小 M * K
- 通过AllToAll PUT实现数据转置：每个rank收集所有其他rank发给自己的数据块
- Win区布局（方案1）：每个rank的Win区包含所有rank发给当前rank的数据（rankSize个chunk）
- FragmentTensor虚拟K轴重排：AssembledShape (M/rankSize, K*rankSize)，D1轴切分
- 执行本地Matmul计算：(M/rankSize, K*rankSize) × (K*rankSize, N) → (M/rankSize, N)

### 1.2 数据流
```
本地数据布局：
[rank0块(M/rankSize, K)][rank1块(M/rankSize, K)][rank2块(M/rankSize, K)][rank3块(M/rankSize, K)]
        ↓ AllToAll PUT（每个rank发送自己的所有块到目标rank）
Win区布局：
[rank0的Win区][rank1的Win区][rank2的Win区][rank3的Win区]
每个Win区包含：所有rank发给当前rank的数据（rankSize个chunk）
        ↓ FragmentTensor虚拟K轴重排（D1轴切分）
虚拟布局：(M/rankSize, K*rankSize) ← 拼接所有rank的K轴
        ↓ Matmul计算（通信计算解耦，流水线模式）
结果：   C(M/rankSize, N)

---

## 二、核心架构

### 2.1 流水线架构
```
┌─────────────────────────────────────────────────────┐
│  NPU芯片：AIV核 + AIC核并行执行                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  AIV核（通信引擎）              AIC核（计算引擎）       │
│  ├─ AllToAll PUT通信            ├─ FragmentTensor   │
│  ├─ PIPE_MTE3管道               ├─ BlockScheduler   │
│  └─ CrossCore Flag同步          ├─ BlockMmadFrag       │
│                                 └─ CrossCore Flag   │
│                                                      │
│  同步机制：CrossCore Flag（2级流水线）                 │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### 2.2 组件选型

| 组件 | 选择 | 来源 | 说明 |
|------|------|------|------|
| **通信引擎** | AivUdma::AllToAll PUT | dista/single_core/aiv_comm | 四段式编程，支持stride切分 |
| **地址管理** | FragmentTensor | dista/basic/fragment_tensor | 虚拟K轴重排，零搬运，编译期推导<br>**仅用于跨rank数据（A矩阵、ScaleA）** |
| **计算引擎** | BlockMmadFrag | dista/single_core/blaze_custom | 支持FragmentTensor（A矩阵、ScaleA）<br>本地数据直接Slice（B矩阵、ScaleB、Bias、C） |
| **切分引擎** | BlockScheduler | blaze/gemm/block | 自动切分问题空间，处理尾块 |
| **同步机制** | CrossCore Flag | AscendC原生API | 低延迟，支持流水线反压 |

**FragmentTensor使用原则**：
- ✅ **需要FragmentTensor**：跨rank通信后的数据（A矩阵、ScaleA）
- ❌ **不需要FragmentTensor**：本地连续数据（B矩阵、ScaleB、Bias、C）

---

## 三、FragmentTensor虚拟K轴重排

### 3.1 Win区物理布局（方案1）
```
Win区布局（每个rank的所有tile连续排列）：
────────────────────────────────────────────
[rank0的Win区: 所有rank发给rank0的数据]
  [rank0发给rank0: chunk_0_0]
  [rank1发给rank0: chunk_1_0]
  [rank2发给rank0: chunk_2_0]
  [rank3发给rank0: chunk_3_0]

[rank1的Win区: 所有rank发给rank1的数据]
  [rank0发给rank1: chunk_0_1]
  [rank1发给rank1: chunk_1_1]
  [rank2发给rank1: chunk_2_1]
  [rank3发给rank1: chunk_3_1]
...

Win区地址计算：
  winAddr + sourceRank * rankSize * chunkSize + myRankId * chunkSize + tileOffset
  
每个chunk大小：chunkSize = M/rankSize * K（包含所有tile）
```

### 3.2 FragmentTensor设计（D1轴切分）

```cpp
// AssembledShape: (M/rankSize, K*rankSize) ← 拼接所有rank的K轴
auto assembledShape = MakeShape(
    Std::Int<M/rankSize>{},           // D0: 每个rank计算M/rankSize行
    Std::Int<K*rankSize>{});          // D1: 拼接所有rank的K列

// FragmentLayout: (M/rankSize, K) ← 每个rank块的完整数据（包含所有tile）
auto fragmentLayout = MakeLayout(
    MakeShape(Std::Int<M/rankSize>{}, Std::Int<K>{}),
    MakeStride(Std::Int<K>{}, Std::Int<1>{}));

// addrList构建：每个rank块的起始地址（基于FragmentTilingData）
for (uint32_t rank = 0; rank < rankSize_; ++rank) {
    addrList[rank] = winAddr + rank * fragTensorTilingA.fragmentSize *
                     fragTensorTilingA.fragmentStride[0] * sizeof(AType);
}

// 编译期推导：
// - SplitAxis = 1（D1轴切分，K轴）
// - FragmentCount = rankSize
```

### 3.3 两级索引访问（BlockMmadFrag）

```cpp
// Block坐标：mPos（例如mPos=64，在tile[2]范围内）
// 依赖的tile：dependTileIdx = mPos / tileM

// Slice获取Block数据（从完整的FragmentTensor）
auto mmadBlockA = fragTensorA_.Slice(
    MakeCoord(mPos, 0),                // ← 全局mPos（不依赖tileIdx）
    MakeShape(baseM, axisKTotal_));    // ← Block大小
    
// FragmentTensor自动推导：
// 第一级：fragmentIdx = mPos / (M/rankSize) = rankIdx（根据行坐标确定rank）
// 第二级：fragmentCoord = (mPos % (M/rankSize), 0)（在rank块内的坐标）
```

### 3.4 FragmentTensor优势

| 特性 | 优势 |
|------|------|
| **零搬运** | 无需额外GM空间，无数据搬运开销 |
| **编译期推导** | SplitAxis和FragmentCount编译期计算，零运行态开销 |
| **统一抽象** | 封装不连续地址，提供统一的Tensor接口 |
| **ForEachFragment** | BlockMmadFrag内部遍历Fragment，分块拷贝到L1，适配空间限制 |

### 3.5 ForEachFragment流程（BlockMmadFrag内部）

BlockMmadFrag的CopyAInL1方法使用ForEachFragment遍历Fragment：

```cpp
gmA.ForEachFragment(kStart, kLength, [&](const auto& info) {
    auto fragment = gmA.GetFragment(info.fragmentIdx);  // GetFragment返回标准Tensor
    
    auto fragBlock = fragment.Slice(
        AscendC::Te::MakeCoord(0, 0),
        AscendC::Te::MakeShape(curM, info.usableSize));
    
    auto l1Block = tensorAL1.Slice(
        AscendC::Te::MakeCoord(0, info.offsetInSubRegion),
        AscendC::Te::MakeShape(curM, info.usableSize));
    
    AscendC::Te::Copy(copyGM2L1, l1Block, fragBlock);  // 逐Fragment拷贝到L1
});
```

**关键特性**：
- GetFragment返回标准Tensor，兼容Copy操作
- ForEachFragment遍历所有Fragment，累积偏移info.offsetInSubRegion
- 分块拷贝，适配L1空间限制（每次拷贝baseM*baseK）

---

## 四、Process流水线设计

### 4.1 统一Process流程

```cpp
// Phase 1: ScaleA通信（数据量小，一次性通信）
if ASCEND_IS_AIV {
    allToAllScale_.Commit();
    allToAllScale_.Wait();
    CrossCoreSetFlag<FLAG_SCALE_READY, PIPE_MTE3>(0);
}

if ASCEND_IS_AIC {
    CrossCoreWaitFlag<FLAG_SCALE_READY, PIPE_MTE2>(0);
}

// Phase 2: A矩阵Tile流水线
for (uint32_t tileIdx = 0; tileIdx < totalTiles; ++tileIdx) {
    
    // AIV核：AllToAll通信
    if ASCEND_IS_AIV {
        allToAll_.Commit();
        allToAll_.Wait();
        CrossCoreSetFlag<FLAG_TILE_READY, PIPE_MTE3>(tileIdx);
    }
    
    // AIC核：Matmul计算
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag<FLAG_TILE_READY, PIPE_MTE2>(tileIdx);
        
        // BlockScheduler动态调度
        while (sch.GetTileIdx(blockIdx)) {
            // WaitFlag优化：只在依赖tile变化时等待
            uint32_t dependTileIdx = CalcDependTileIdx(mOffset + mPos);
            if (dependTileIdx != lastTileIdx) {
                CrossCoreWaitFlag<FLAG_TILE_READY, PIPE_MTE2>(dependTileIdx);
                lastTileIdx = dependTileIdx;
            }
            
            // FragmentTensor Slice + BlockMmadFrag计算
            mmadOp(...);
        }
    }
}

// Phase 3: Finalize
if ASCEND_IS_AIV {
    allToAll_.Finalize();
    allToAllScale_.Finalize();
}
```

### 4.2 流水线时序图

```
时间轴：  ScaleA   tile[0]  tile[1]  tile[2]  tile[3]  ...  Finalize
──────────────────────────────────────────────────────────────────────
AIV核：   ScaleA   Comm0    Comm1    Comm2    Comm3    ...  Finalize
         ↓Set3    ↓Set2    ↓Set2    ↓Set2    ↓Set2
         Flag3[0] Flag2[0] Flag2[1] Flag2[2] Flag2[3]

──────────────────────────────────────────────────────────────────────
AIC核：   Wait3[0] Wait2[0] Wait2[1] Wait2[2] Wait2[3] ...
         ↑Wait3   ↑Wait2   ↑Wait2   ↑Wait2   ↑Wait2
         Flag3[0] Flag2[0] Flag2[1] Flag2[2] Flag2[3]
         
                  Compute0 Compute1 Compute2 Compute3 ...
                  (BlockScheduler动态调度，依赖等待优化)
```

### 4.3 Flag ID定义

```cpp
constexpr uint32_t FLAG_SCALE_READY = 0x3;  // ScaleA通信完成标志
constexpr uint32_t FLAG_TILE_READY  = 0x2;  // AIV通知AIC：tile数据就绪
```

### 4.4 WaitFlag优化

- **依赖计算**：`dependTileIdx = (mOffset + mPos) / tileM`
- **优化策略**：只在`dependTileIdx != lastTileIdx`时调用WaitFlag
- **效果**：减少75%的WaitFlag调用次数（假设每个tile有4个block）

### 4.5 通信计算解耦设计

**关键特性**：
- AIV顺序提交所有tile，无需等待AIC
- AIC通过依赖检查等待所需tile完成
- Blaze动态调度（BlockScheduler）+ 静态依赖检查
- 通信和计算在tile级别并行

---

## 五、Blaze组件复用策略

### 5.1 完全复用组件

| 组件 | 复用方式 | 修改 |
|------|---------|------|
| BlockMmadFrag | `mmadOp.Init() + mmadOp(tensor...)` | ✅ 无修改 |
| BlockScheduler | `sch.GetTileIdx() + sch.GetBlockShape()` | ✅ 无修改 |
| Tensor API | `Slice + MakeTensor` | ✅ 无修改 |

### 5.2 不使用GemmUniversal

**原因**：
- GemmUniversal封装了完整kernel流程（Init + Run + 循环）
- 分布式场景需要手动控制流水线（AIV通信 → AIC计算）
- 无法集成CrossCore Flag同步机制

**替代方案**：
- 拆解GemmUniversal，手动组合BlockMmadFrag + BlockScheduler
- 在for循环中集成FragmentTensor + CrossCore Flag
- 灵活控制rank遍历和tile切分

---

## 五、TilingData结构设计

### 5.1 引用CommTilingData设计

```cpp
struct AllToAllMatmulTilingData {
    // 问题规模
    uint64_t axisM;         // M维度总大小
    uint64_t axisK;         // K维度总大小（每个rank持有K片段）
    uint32_t axisN;         // N维度大小
    
    // 通信切分参数（引用CommTilingData）
    CommTilingData commTilingData;  // {tileSize, tileCnt, tailSize, tailCnt}
    
    // A矩阵Tile参数
    uint64_t tileM;         // tile的M大小（stride）
    uint64_t tailM;         // tail的M大小
    uint64_t totalTiles;   // 总tile数量
    
    // Blaze计算切分参数
    uint32_t baseM;         // Blaze基础块M大小
    uint32_t baseN;         // Blaze基础块N大小
    uint32_t baseK;         // Blaze基础块K大小
    uint32_t stepK;         // K维度分步加载次数
    uint32_t scaleKL1;      // Scale在L1的K大小
    uint32_t dbL0c;         // L0C double buffer数量
    
    // Blaze尾块参数
    uint32_t mTailTile;
    uint32_t nTailTile;
    uint32_t mBaseTailSplitCnt;
    uint32_t nBaseTailSplitCnt;
    uint32_t mTailMain;
    uint32_t nTailMain;
};
```

### 5.2 CommTilingData约束

```cpp
// CommTilingData定义
struct CommTilingData {
    uint64_t splitAxisTileSize;     // 切分轴头块大小（元素个数）
    uint64_t splitAxisTileCnt;      // 切分轴头块数量
    uint64_t splitAxisTailSize;     // 切分轴尾块大小（元素个数）
    uint64_t splitAxisTailCnt;      // 切分轴尾块数量
    uint64_t nonSplitAxisSize;      // 非切分轴数据块大小（元素个数）
};

// 约束关系
chunkSize = splitAxisTileSize * splitAxisTileCnt + splitAxisTailSize * splitAxisTailCnt
totalTiles = splitAxisTileCnt + splitAxisTailCnt
```

### 5.3 关键参数调整

由于K轴拼接，Blaze参数需要调整：

```cpp
// ProblemShape调整
ProblemShape problemShape{curM, axisN, axisK * rankSize, 1};  // ← K = axisK * rankSize

// baseK调整（需要支持完整的K*rankSize）
baseK = CalculateBaseK(axisK * rankSize);  // ← 根据拼接后的K计算

// L1空间约束
l1BufferSize = baseM * baseK * 2 + baseN * baseK * 2 + scale buffers;
需要确保：l1BufferSize < TOTAL_L1_SIZE
```

---

## 六、性能预估

| 特性 | 性能影响 | 预估提升 |
|------|---------|---------|
| FragmentTensor虚拟重排 | 零搬运 | 0% overhead |
| 编译期推导 | 零运行态开销 | 0% overhead |
| CrossCore Flag同步 | 低延迟 | ~10 cycles/flag |
| WaitFlag优化 | 减少75%调用 | 降低同步开销 |
| Blaze动态调度 | 自适应尾块 | 优化计算效率 |
| BlockMmadFrag MXFP8量化 | 矩阵加速 | ~4x throughput |

---

## 七、关键约束

### 7.1 AllToAll PUT线性语义
- 本地和Win区使用相同offset
- 无法实现不同的本地和Win区offset映射
- 必须依赖FragmentTensor虚拟重排

### 7.2 Win区空间约束
- Win区大小：rankSize * M * K * sizeof(fp8)
- 需要预留足够Win区空间

### 7.3 L1空间约束
- baseM * baseK需要支持完整的K*rankSize
- 可能需要调整baseM/baseK以适应L1空间

### 7.4 Blaze调度限制
- BlockScheduler使用动态调度（while GetTileIdx）
- 无法实现完全静态编排
- 解决方案：动态调度 + 静态依赖检查（WaitFlag优化）

---

## 八、Host端调用示例

```cpp
#include "dista/host/tiling/fusion_tiling.h"

// 1. 计算切分参数
CommTilingData commTiling = ComputeCommTiling(M, K, rankSize, sizeof(fp8_e4m3fn_t));

// 2. 构造TilingData
AllToAllMatmulTilingData kernelTiling;
kernelTiling.axisM = M;
kernelTiling.axisK = K;              // ← 每个rank持有的K片段
kernelTiling.axisN = N;
kernelTiling.commTilingData = commTiling;
kernelTiling.tileM = tileSize / (K * sizeof(fp8_e4m3fn_t));
kernelTiling.tailM = tailSize / (K * sizeof(fp8_e4m3fn_t));
kernelTiling.totalTiles = commTiling.tileCnt + commTiling.tailCnt;
kernelTiling.baseM = ComputeBaseM(M);
kernelTiling.baseN = ComputeBaseN(N);
kernelTiling.baseK = CalculateBaseK(K * rankSize);  // ← 调整baseK
// ... 其他参数

// 3. 调用kernel
AllToAllMatmulKernelEntry(ctx, aGM, aScaleGM, bGM, bScaleGM, cGM, kernelTiling);
```

---

## 九、实现文件

**文件路径**：`include/dista/kernel/all_to_all_matmul/all_to_all_matmul_kernel.h`

**实现清单**：
1. TilingData结构体定义（引用CommTilingData）
2. Kernel类框架（Init/Process）
3. Init函数（数据结构准备）
4. CreateFragmentTensor（D1轴切分，完整Layout）→ **仅用于跨rank数据（A矩阵）**
5. CreateScaleFragmentTensor → **仅用于跨rank数据（ScaleA）**
6. CreateTensors（B, ScaleB, C, Bias）→ **本地数据，直接MakeTensor，无需FragmentTensor**
7. Process函数
   - Phase 1: ScaleA通信
   - Phase 2: A矩阵Tile流水线
   - Phase 3: 统一Finalize
8. Helper函数（GetCurMSize, CalcMOffset, CalcDependTileIdx）
9. Kernel Entry（入口函数）

---

## 十、后续优化方向

1. **流水线深度优化**：探索AIV-AIC深度流水线，提升并行度
2. **L1空间优化**：动态调整baseM/baseK以最大化L1利用率
3. **尾块优化**：精细化尾块处理策略
4. **多核调度**：利用BlockScheduler的多核调度能力
5. **静态编排**：研究完全静态编排方案，消除动态调度开销