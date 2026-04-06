# MegaMoe GMM 详细设计与代码注释

> **目标**：从零讲清楚 MegaMoe 中的 GMM（Grouped Matrix Multiplication）是怎么实现的。

---

## 1. 什么是 GMM？

在 MoE（Mixture of Experts）模型中，每个 token 会被路由到不同的专家（Expert）。每个专家本质上是一个 FFN（前馈神经网络），包含两层矩阵乘法：

```
FFN(x) = (SwiGLU(x × W1)) × W2
```

**GMM（Grouped Matrix Multiplication）** 就是同时对多个专家做矩阵乘法。

### 1.1 普通矩阵乘 vs GMM

```
普通 GEMM:          C = A × B
                    A[M, K] × B[K, N] → C[M, N]

GMM:               C_i = A_i × B_i   (i = 0, 1, 2, ..., num_experts-1)
                   A_i[m_i, K] × B_i[K, N] → C_i[m_i, N]

注意：每个专家的 m_i 不同！这就是 GMM 的核心挑战。
```

### 1.2 MegaMoe 中的 GMM 流程

```
                    GMM1                    GMM2
x [m, k] ──────────────→ [m, n] ──→ SwiGLU ──→ [m, n/2] ──→ [m, k]
  (FP8)                  (BF16)     Quantize    (FP8)        (BF16)
                         ↑                    ↑
                    Weight1 [E, n, k]    Weight2 [E, k, n/2]
                    (FP8 + Scale)        (FP8 + Scale)
```

**关键点**：
- GMM1 输出 BF16，经过 SwiGLU 激活后量化为 FP8
- GMM2 接收量化后的 FP8 输入，输出 BF16
- 每个专家处理不同数量的 token（m_i 不同）

---

## 2. 整体架构

### 2.1 代码文件分工

```
mega_moe.h                          # 类型定义 + 参数组装 + Kernel 启动
    │
    ├─ 定义 GMM1 的 BlockMmad 类型
    ├─ 定义 GMM2 的 BlockMmad2 类型
    ├─ 定义 BlockEpilogue (SwiGLU + Quant)
    └─ 构造 KernelGmmSwiGluMixOnlineDynamic 并调用

kernel_gmm_swiglu_mxquant.h         # GMM 核心实现
    │
    ├─ InitParamsAndTensor()         # 初始化
    ├─ Dispatch()                    # 跨卡分发 token
    ├─ GroupMatmulSwigluQuant()      # GMM1 + SwiGLU + 量化
    ├─ GroupMatmul2()                # GMM2 + Combine（跨卡写回）
    └─ CrossRankSync...()            # 多卡同步
```

### 2.2 数据流全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                          MegaMoe Kernel                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  输入:                                                               │
│    x [m*topK, k] FP8          # 所有 token（已按专家分组）            │
│    weight1 [EP*E, n, k] FP8   # 所有专家的 W1                        │
│    weight2 [EP*E, k, n/2] FP8 # 所有专家的 W2                        │
│    probs [m, topK] BF16       # gating 概率                          │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Phase 1: InitRouting + CrossRank Sync                          │ │
│  │  - moe_init_routing_v3: 按 expertIdx 重排 token                │ │
│  │  - CrossRankSync: 多卡同步 tokenPerExpert，计算 cumsumMM       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Phase 2: GMM1 Loop (for each expert)                           │ │
│  │                                                                │ │
│  │  for groupIdx in expertPerRank:                                │ │
│  │    ┌──────────────────────────────────────────────────────┐   │ │
│  │    │ AIV (subBlockIdx==1): Dispatch                       │   │ │
│  │    │  - 从远端卡拉取分配给本地专家的 token                  │   │ │
│  │    │  - 写入 ptrA [m_i, k] FP8                           │   │ │
│  │    └──────────────────────────────────────────────────────┘   │ │
│  │    ┌──────────────────────────────────────────────────────┐   │ │
│  │    │ AIC (subBlockIdx==0): GMM1                           │   │ │
│  │    │  - blockMmad: A[m_i,k] × B[k,n] → C[m_i,n] BF16    │   │ │
│  │    │  - 循环遍历所有 L1 Tile                              │   │ │
│  │    │  - 对 SWIGLU_N_HALF=2 个权重块分别计算               │   │ │
│  │    └──────────────────────────────────────────────────────┘   │ │
│  │    ┌──────────────────────────────────────────────────────┐   │ │
│  │    │ AIV (subBlockIdx==0): SwiGLU + Quant                 │   │ │
│  │    │  - WaitForCube()                                     │   │ │
│  │    │  - epilogueOp: swish(x)*y + BF16→FP8                │   │ │
│  │    │  - 输出到 ptrA2 [m_i, n/2] FP8                      │   │ │
│  │    └──────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Phase 3: GMM2 Loop (for each expert)                           │ │
│  │                                                                │ │
│  │  for groupIdx in expertPerRank:                                │ │
│  │    ┌──────────────────────────────────────────────────────┐   │ │
│  │    │ AIC (subBlockIdx==0): GMM2                           │   │ │
│  │    │  - 等待 GMM1 完成 (groupFlagListGm)                  │   │ │
│  │    │  - blockMmad: A2[m_i,n/2] × B2[n/2,k] → C2[m_i,k]  │   │ │
│  │    │  - 使用 Ping-Pong 双缓冲                             │   │ │
│  │    └──────────────────────────────────────────────────────┘   │ │
│  │    ┌──────────────────────────────────────────────────────┐   │ │
│  │    │ AIV (subBlockIdx==0): Combine                        │   │ │
│  │    │  - WaitForCube()                                     │   │ │
│  │    │  - 按原始 rank 将结果写回远端 peermem                │   │ │
│  │    │  - 写入 ptrD [m_i, k] BF16                          │   │ │
│  │    └──────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Phase 4: Finalize                                              │ │
│  │  - moe_finalize_routing_v2: 按 probs 加权聚合                  │ │
│  │  - 输出 y [m, k] BF16                                          │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. mega_moe.h 详解：类型定义与参数组装

### 3.1 完整注释代码

```cpp
// mega_moe.h - MegaMoe 算子主类
// 职责：定义所有类型、组装参数、启动 Kernel

#ifndef MEGA_MOE_H
#define MEGA_MOE_H

#include "gmmsq_base.h"
#include "kernel_gmm_swiglu_mxquant.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "tile_copy.hpp"
#include "mega_moe_tiling_def.h"
#include "hccl_shmem.h"
#include "workspace_info.h"

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

// 引入 Catlass GEMM 库
#include "catlass/gemm/kernel/mx_matmul_tla.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

using namespace Catlass;
using namespace tla;

namespace MegaMoeImpl {

// HCCL 最大卡数
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024;

// MoE 上下文信息（跨卡通信所需）
struct Mc2MoeContext {
    uint32_t epRankId;                        // 当前 EP rank ID
    uint32_t epRankSize;                      // EP 总卡数
    uint64_t winSize;                         // 共享内存窗口大小
    uint64_t epHcclBuffer_[HCCL_MAX_RANK_SIZE]; // 每张卡的 HCCL buffer 地址
};

class MegaMoe {
public:
    __aicore__ inline MegaMoe(){};
    
    // Init: 定义类型、组装参数、启动 Kernel
    __aicore__ inline void Init(
        GM_ADDR mc2Context,     // MoE 上下文（包含跨卡通信信息）
        GM_ADDR x,              // 输入 token [m*topK, k] FP8
        GM_ADDR expertIdx,      // 专家索引 [m, topK] INT32
        GM_ADDR weight1,        // W1 [EP*E, n, k] FP8
        GM_ADDR weightScales1,  // W1 的 MX Scale
        GM_ADDR weight2,        // W2 [EP*E, k, n/2] FP8
        GM_ADDR weightScales2,  // W2 的 MX Scale
        GM_ADDR probs,          // gating 概率 [m, topK] BF16
        GM_ADDR y,              // 输出 [m, k] BF16
        GM_ADDR workspace,      // 本地工作区
        const MegaMoeTilingData *tilingData  // Tiling 参数
    );
    
    // Process: 当前为空（TODO：代码结构要改）
    __aicore__ inline void Process();
};

// ============================================================================
// Init 函数详解
// ============================================================================
__aicore__ inline void MegaMoe::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIdx, GM_ADDR weight1, GM_ADDR weightScales1,
    GM_ADDR weight2, GM_ADDR weightScales2, GM_ADDR probs, GM_ADDR y, GM_ADDR workspace,
    const MegaMoeTilingData *tilingData) 
{
    // ------------------------------------------------------------------------
    // Step 1: 提取 Tiling 参数
    // ------------------------------------------------------------------------
    uint32_t m = tilingData->m;       // token 总数
    uint32_t k = tilingData->k;       // 隐藏层大小
    uint32_t n = tilingData->n;       // FFN 中间层大小
    uint32_t expertPerRank = tilingData->expertPerRank;  // 每张卡的专家数
    uint8_t groupListType = tilingData->groupListType;   // 组列表类型
    
    // 计算 MX Scale 的 K 维度
    // MXFP 每 32 个元素共享 1 个 scale，每个 scale 占 2 字节
    uint32_t mxScaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;

    // ------------------------------------------------------------------------
    // Step 2: 定义 GMM1 的类型
    // ------------------------------------------------------------------------
    
    // 矩阵元素类型
    using AType = fp8_e4m3fn_t;   // A 矩阵（输入 token）使用 FP8 E4M3
    using BType = fp8_e4m3fn_t;   // B 矩阵（权重 W1）使用 FP8 E4M3
    using CType = fp8_e4m3fn_t;   // C 矩阵（GMM2 的输入）使用 FP8 E4M3
    
    // 矩阵布局
    using LayoutA = layout::RowMajor;        // A 是行优先 [m, k]
    using LayoutB = layout::ColumnMajor;     // B 是列优先 [k, n]（计算时转置）
    using LayoutMMadOut = layout::RowMajor;  // 输出是行优先 [m, n]
    
    // Scale 类型（MX 格式使用 E8M0）
    using xScaleType = AscendC::fp8_e8m0_t;      // 输入 token 的 scale
    using weightScaleType = AscendC::fp8_e8m0_t; // 权重的 scale
    
    // 输出类型（GMM1 输出是 BF16，用于 SwiGLU 计算）
    using MMadOutType = bfloat16_t;

    // 创建 MX Scale 的布局
    // layoutMxScaleA: [m, ceil(k/32)*2] E8M0
    //   - false 表示这是 ScaleA（行的 scale）
    auto layoutMxScaleA = tla::MakeMxScaleLayout<xScaleType, LayoutA, false>(m, mxScaleK);
    
    // layoutMxScaleB: [ceil(k/32)*2, n] E8M0
    //   - true 表示这是 ScaleB（列的 scale）
    auto layoutMxScaleB = tla::MakeMxScaleLayout<weightScaleType, LayoutB, true>(mxScaleK, n);

    // ------------------------------------------------------------------------
    // Step 3: 定义调度器（BlockScheduler）
    // ------------------------------------------------------------------------
    // GemmIdentityBlockSwizzle<3, 1>：
    //   - SwizzleOffset=3：每 3 个 Block 为一组进行调度优化
    //   - SwizzleDirection=1：沿 N 方向（列优先）进行 Swizzle
    // 作用：改变 Block 的处理顺序，提高 L2 Cache 命中率
    using BlockScheduler = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;

    // ------------------------------------------------------------------------
    // Step 4: 定义硬件架构和 Dispatch 策略
    // ------------------------------------------------------------------------
    using ArchTag = Arch::AtlasA5;  // 目标芯片：Atlas A5
    constexpr bool enableUnitFlag = true;  // 启用 Unit Flag（优化 L0C 写入）
    
    // MmadMx：支持 MX 格式（Microscaling）的矩阵乘策略
    //   - 自动处理 FP8 数据 + E8M0 Scale 的联合计算
    using DispatchPolicy = Gemm::MmadMx<ArchTag, enableUnitFlag>;

    // ------------------------------------------------------------------------
    // Step 5: 定义 Tile 形状（Tiling 策略）
    // ------------------------------------------------------------------------
    // L1 Tile: 从 GM 搬运到 L1 的块大小
    //   - M=256, N=256, K=448
    //   - L1 缓存约 512KB，这个大小可以充分利用 L1
    using L1TileShape = tla::Shape<Int<256>, Int<256>, Int<448>>;
    
    // L0 Tile: 从 L1 搬运到 L0 的块大小
    //   - M=256, N=256, K=128
    //   - L0 缓存约 64KB，K 更小以适应 L0 大小
    // L1 Tile 的 K=448 会被分成 4 批 (448/128≈4) 送入 L0 计算
    using L0TileShape = tla::Shape<Int<256>, Int<256>, Int<128>>;

    // ------------------------------------------------------------------------
    // Step 6: 定义 BlockEpilogue（GMM1 后的处理：SwiGLU + 量化）
    // ------------------------------------------------------------------------
    // 模板参数：
    //   - L1TileShape: tile 形状
    //   - CType: 输出数据类型 (FP8)
    //   - MMadOutType: 中间计算类型 (BF16)
    //   - weightScaleType: scale 类型
    //   - weightScaleType: scale 类型（X2）
    //   - true: 使用 TensorList 模式
    using BlockEpilogue = BlockEpilogueSwigluQuant<
        L1TileShape, CType, MMadOutType, 
        weightScaleType, weightScaleType, true>;
    
    // ------------------------------------------------------------------------
    // Step 7: 定义 TileCopy（数据搬运器）
    // ------------------------------------------------------------------------
    // PackedMxTileCopyTlaToUB：
    //   - 负责 GM → L1 → L0 的数据搬运
    //   - 支持 MX 格式（数据 + Scale 联合搬运）
    //   - 输出到 UB（Vector Buffer）供 Epilogue 使用
    using TileCopy = Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, 
        AType, LayoutA,           // A 的类型和布局
        BType, LayoutB,           // B 的类型和布局
        weightScaleType, decltype(layoutMxScaleA),  // ScaleA
        weightScaleType, decltype(layoutMxScaleB),  // ScaleB
        MMadOutType, LayoutMMadOut,  // 输出类型和布局
        void>;                    // 无 Bias

    // ------------------------------------------------------------------------
    // Step 8: 定义 BlockMmad（GMM1 的 Block 级矩阵乘）
    // ------------------------------------------------------------------------
    // BlockMmadTla 是 Catlass 提供的核心组件，封装了：
    //   - GM → L1 → L0 的流水线搬运
    //   - L0 上的 MMA 计算
    //   - L0 → UB 的输出搬运
    using BlockMmad = Gemm::Block::BlockMmadTla<
        DispatchPolicy,           // 策略（MX 矩阵乘）
        L1TileShape,              // L1 Tile 形状
        L0TileShape,              // L0 Tile 形状
        AType, BType, MMadOutType, // 数据类型
        void,                     // 无 Bias
        TileCopy>;                // 搬运器

    // ------------------------------------------------------------------------
    // Step 9: 定义 GMM2 的类型（与 GMM1 类似，但输入输出不同）
    // ------------------------------------------------------------------------
    
    // GMM2 的输入是 GMM1 的量化输出，形状为 [m, n/2]
    uint32_t mx2ScaleN = CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
    
    using B2Type = fp8_e4m3fn_t;           // W2 使用 FP8
    using MMadOut2Type = bfloat16_t;       // GMM2 输出使用 BF16
    using LayoutB2 = layout::ColumnMajor;  // W2 是列优先
    using LayoutMMadOut2 = layout::RowMajor; // 输出是行优先
    using x2ScaleType = AscendC::fp8_e8m0_t;   // 输入 scale
    using weight2ScaleType = AscendC::fp8_e8m0_t; // 权重 scale
    
    // GMM2 的 Scale 布局
    auto layoutMx2ScaleA = tla::MakeMxScaleLayout<x2ScaleType, LayoutMMadOut, false>(m, mx2ScaleN);
    auto layoutMx2ScaleB = tla::MakeMxScaleLayout<weight2ScaleType, LayoutB2, true>(mx2ScaleN, k);

    // GMM2 的 Tile 形状（与 GMM1 相同）
    using L1TileShape2 = tla::Shape<Int<256>, Int<256>, Int<448>>;
    using L0TileShape2 = tla::Shape<Int<256>, Int<256>, Int<128>>;

    // GMM2 的搬运器和 BlockMmad
    using TileCopy2 = Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, 
        CType, LayoutMMadOut,       // A2 的类型和布局（FP8，行优先）
        B2Type, LayoutB2,           // B2 的类型和布局
        weight2ScaleType, decltype(layoutMx2ScaleA),
        weight2ScaleType, decltype(layoutMx2ScaleB),
        MMadOut2Type, LayoutMMadOut2,
        void>;
    
    using BlockMmad2 = Gemm::Block::BlockMmadTla<
        DispatchPolicy, L1TileShape2, L0TileShape2, 
        CType, B2Type, MMadOut2Type, void, TileCopy2>;

    // ------------------------------------------------------------------------
    // Step 10: 定义 Kernel 类型并构造
    // ------------------------------------------------------------------------
    // KernelGmmSwiGluMixOnlineDynamic 是自定义的 Kernel 类，整合了：
    //   - GMM1 (BlockMmad)
    //   - GMM2 (BlockMmad2)
    //   - SwiGLU + 量化 (BlockEpilogue)
    //   - 调度器 (BlockScheduler)
    using QGmmKernel = KernelGmmSwiGluMixOnlineDynamic<
        BlockMmad, BlockMmad2, BlockEpilogue, BlockScheduler>;
    
    using Params = typename QGmmKernel::Params;

    // ------------------------------------------------------------------------
    // Step 11: 设置跨卡通信上下文
    // ------------------------------------------------------------------------
    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    mc2Context_ = (__gm__ Mc2MoeContext*)mc2Context;
    
    // 构建 HCCL buffer 数组
    GM_ADDR buff[HCCL_MAX_RANK_SIZE];
    for (int i = 0; i < tilingData->EP; i++) {
        buff[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
    }
    
    // ------------------------------------------------------------------------
    // Step 12: 组装 Params 并启动 Kernel
    // ------------------------------------------------------------------------
    Params params = {
        .aGmAddr = x,                              // 输入 token
        .expertIdxGmAddr = expertIdx,              // 专家索引
        .bGmAddr = weight1,                        // W1
        .bScaleGmAddr = weightScales1,             // W1 Scale
        .b2GmAddr = weight2,                       // W2
        .b2ScaleGmAddr = weightScales2,            // W2 Scale
        .y2GmAddr = y,                             // 输出
        .probsGmAddr = probs,                      // gating 概率
        .workspaceInfo = WorkspaceInfo(workspace, tilingData),  // 工作区
        .peermemInfo = PeermemInfo(buff[tilingData->rank], tilingData), // 远端内存
        .shmem = HcclShmem(buff, tilingData->rank, tilingData->EP),   // HCCL 通信
        .tilingData = tilingData                   // Tiling 参数
    };

    // 构造 Kernel 对象并执行
    QGmmKernel op;
    op(params);  // 调用 operator()
};

__aicore__ inline void MegaMoe::Process() {};
  // TODO: 代码结构要改

}  // namespace MegaMoeImpl
#endif
```

---

## 4. kernel_gmm_swiglu_mxquant.h 详解：GMM 核心实现

### 4.1 类定义与成员变量

```cpp
// ============================================================================
// KernelGmmSwiGluMixOnlineDynamic
// ============================================================================
// 这是 MegaMoe 的核心 Kernel 类，实现了完整的 GMM 流程。
//
// 模板参数：
//   - BlockMmad_:  GMM1 的 Block 级矩阵乘
//   - BlockMmad2_: GMM2 的 Block 级矩阵乘
//   - BlockEpilogue_: SwiGLU + 量化
//   - BlockScheduler_: 分块调度器
// ============================================================================
template <class BlockMmad_, class BlockMmad2_, class BlockEpilogue_, class BlockScheduler_>
class KernelGmmSwiGluMixOnlineDynamic {
public:
    CATLASS_DEVICE KernelGmmSwiGluMixOnlineDynamic() {}
    CATLASS_DEVICE ~KernelGmmSwiGluMixOnlineDynamic() {}

    // ------------------------------------------------------------------------
    // 类型别名：从模板参数中提取所有需要的类型
    // ------------------------------------------------------------------------
    using BlockEpilogue = BlockEpilogue_;
    using BlockScheduler = BlockScheduler_;
    using DataTypeOut = typename BlockEpilogue::DataTypeOut;
    
    // GMM1 相关类型
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;          // FP8
    using LayoutA = typename BlockMmad::LayoutA;            // RowMajor
    using ElementB = typename BlockMmad::ElementB;          // FP8
    using LayoutB = typename BlockMmad::LayoutB;            // ColumnMajor
    using ElementMxScaleA = typename BlockMmad::TileCopy::ElementMxScaleA;  // E8M0
    using ElementMxScaleB = typename BlockMmad::TileCopy::ElementMxScaleB;  // E8M0
    using ElementMMadOut = typename BlockMmad::ElementC;    // BF16
    using LayoutMMadOut = typename BlockMmad::LayoutC;      // RowMajor
    using LayoutTagA = typename BlockMmad::TileCopy::LayoutTagA;
    using LayoutTagB = typename BlockMmad::TileCopy::LayoutTagB;
    using LayoutTagC = typename BlockMmad::TileCopy::LayoutTagC;

    // L1 Tile 常量
    static constexpr uint32_t L1_TILE_M = tla::get<0>(L1TileShape{});  // 256
    static constexpr uint32_t L1_TILE_N = tla::get<1>(L1TileShape{});  // 256
    static constexpr uint32_t L1_TILE_K = tla::get<2>(L1TileShape{});  // 448

    // GMM2 相关类型（类似 GMM1）
    using BlockMmad2 = BlockMmad2_;
    using L1TileShape2 = typename BlockMmad2::L1TileShape;
    using ElementA2 = typename BlockMmad2::ElementA;          // FP8 (GMM1 输出)
    using LayoutA2 = typename BlockMmad2::LayoutA;            // RowMajor
    using ElementB2 = typename BlockMmad2::ElementB;          // FP8 (W2)
    using LayoutB2 = typename BlockMmad2::LayoutB;            // ColumnMajor
    using ElementMxScaleA2 = typename BlockMmad2::TileCopy::ElementMxScaleA;
    using ElementMxScaleB2 = typename BlockMmad2::TileCopy::ElementMxScaleB;
    using ElementMMadOut2 = typename BlockMmad2::ElementC;    // BF16
    using LayoutMMadOut2 = typename BlockMmad2::LayoutC;      // RowMajor
    using LayoutTagA2 = typename BlockMmad2::TileCopy::LayoutTagA;
    using LayoutTagB2 = typename BlockMmad2::TileCopy::LayoutTagB;
    using LayoutTagC2 = typename BlockMmad2::TileCopy::LayoutTagC;

    static constexpr uint32_t L1_TILE_M2 = tla::get<0>(L1TileShape2{});  // 256
    static constexpr uint32_t L1_TILE_N2 = tla::get<1>(L1TileShape2{});  // 256
    static constexpr uint32_t L1_TILE_K2 = tla::get<2>(L1TileShape2{});  // 448

    // ------------------------------------------------------------------------
    // 形状和偏移类型
    // ------------------------------------------------------------------------
    // TupleShape: {m, n, k, reserved}
    using TupleShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    
    // BlockOffset: 12 个偏移量，用于追踪每个 tensor 的当前位置
    // 索引含义：
    //   0: A 的偏移    1: B 的偏移     2: A Scale 的偏移
    //   3: B Scale 的偏移  4: Bias 的偏移  5: C 的偏移
    //   6: C Scale 的偏移  7: Group Flag  8: B2 的偏移
    //   9: B2 Scale 的偏移 10: Y2 的偏移   11: M 的偏移
    using BlockOffset = AscendC::Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                                       int64_t, int64_t, int64_t, int64_t, int64_t,
                                       int64_t, int64_t>;

    // ------------------------------------------------------------------------
    // 全局 Tensor（GM 内存）
    // ------------------------------------------------------------------------
    // GMM1 的输入和权重
    AscendC::GlobalTensor<ElementA> aGlobal;        // 输入 token [m, k] FP8
    AscendC::GlobalTensor<ElementB> bGlobal;        // 权重 W1 [k, n] FP8
    AscendC::GlobalTensor<ElementMxScaleA> aScaleGlobal;  // 输入 Scale
    AscendC::GlobalTensor<ElementMxScaleB> bScaleGlobal;  // 权重 Scale
    
    // 同步标志
    AscendC::GlobalTensor<int32_t> groupListGm;     // 组列表（从 cumsumMM 获取）
    AscendC::GlobalTensor<int32_t> groupFlagListGm; // GMM1 完成标志
    AscendC::GlobalTensor<int32_t> groupFlagListGm2;// Dispatch 完成标志

    // ------------------------------------------------------------------------
    // GMM1 的 UB 缓冲（输出到 Vector Buffer）
    // ------------------------------------------------------------------------
    // 双缓冲：两个 UB 区域交替使用，实现计算与搬运重叠
    // l0cOutUbFirst:  第一个 UB 区域，偏移 0，大小 MAX_SINGLE_MN
    // l0cOutUbSecond: 第二个 UB 区域，偏移 MAX_SINGLE_MN * sizeof(BF16)
    AscendC::LocalTensor<ElementMMadOut> l0cOutUbFirst{
        AscendC::TPosition::VECIN, 0, MAX_SINGLE_MN};
    AscendC::LocalTensor<ElementMMadOut> l0cOutUbSecond{
        AscendC::TPosition::VECIN, MAX_SINGLE_MN * sizeof(ElementMMadOut), MAX_SINGLE_MN};

    // ------------------------------------------------------------------------
    // GMM2 相关
    // ------------------------------------------------------------------------
    // Ping-Pong 索引：0 或 1，用于切换双缓冲
    uint16_t GMM2_PINGPONG_IDX = 0;
    static constexpr uint32_t GMM2_PINGPONG_COUNT = 2;  // 双缓冲数量
    static constexpr uint32_t GMM2_BLOCK_SIZE = L1_TILE_M2 * L1_TILE_N2;  // 256*256
    
    // GMM2 的输入（GMM1 的量化输出）
    AscendC::GlobalTensor<ElementA2> a2Global;      // [m, n/2] FP8
    AscendC::GlobalTensor<ElementB2> b2Global;      // W2 [n/2, k] FP8
    AscendC::GlobalTensor<ElementMxScaleA2> a2ScaleGlobal;
    AscendC::GlobalTensor<ElementMxScaleB2> b2ScaleGlobal;
    
    // GMM2 的 UB 缓冲（双缓冲）
    AscendC::LocalTensor<ElementMMadOut2> l0cOutUbGMM2First{
        AscendC::TPosition::VECIN, 0, GMM2_BLOCK_SIZE};
    AscendC::LocalTensor<ElementMMadOut2> l0cOutUbGMM2Second{
        AscendC::TPosition::VECIN, GMM2_BLOCK_SIZE * sizeof(ElementMMadOut2), GMM2_BLOCK_SIZE};
    
    // GMM2 输出
    AscendC::GlobalTensor<ElementMMadOut2> outputGm;  // [m, k] BF16

    // ------------------------------------------------------------------------
    // 跨卡通信相关
    // ------------------------------------------------------------------------
    AscendC::GlobalTensor<int32_t> preSumBeforeRank;  // 前缀和
    AscendC::GlobalTensor<int32_t> tokenPerExpert;    // 每个专家的 token 数
    AscendC::GlobalTensor<int32_t> cumsumMM;          // 累积和

    // ------------------------------------------------------------------------
    // Epilogue 和资源
    // ------------------------------------------------------------------------
    BlockEpilogue epilogueOp;  // SwiGLU + 量化操作
    Arch::Resource<ArchTag> resource;  // 硬件资源（L1/L0 缓存等）

    // ------------------------------------------------------------------------
    // 状态变量
    // ------------------------------------------------------------------------
    TupleShape problemShape{};  // 当前问题形状 {m, n, k, 0}
    BlockOffset baseOffset{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};  // 偏移
    Layout3D tokenPerExpertLayout;  // tokenPerExpert 的 3D 布局
    uint64_t preOffset = 0;     // 前一个偏移

    // 同步状态
    bool isVecSetSyncCom = false;  // GMM1 是否已设置同步
    uint32_t VecSetSyncCom2 = 0;   // GMM2 同步计数
    uint32_t startBlockIdx = 0;    // 起始 Block 索引
    uint32_t blockNum = AscendC::GetBlockNum();  // 总 Core 数
    uint32_t aicBlockNum = blockNum / AscendC::GetTaskRation();  // AIC 数量
    uint32_t blockIdx = AscendC::GetBlockIdx() / AscendC::GetTaskRation();  // 当前 Block ID
    uint32_t subBlockIdx = AscendC::GetSubBlockIdx();  // 子 Block ID (AIV 细分)
    uint32_t aivCoreIdx = AscendC::GetBlockIdx();  // AIV Core ID
    int32_t prevSum = 0;           // 前缀和
    uint32_t dispatchBlockNumPerEP = 2;  // 每 EP 的 dispatch 块数
```

### 4.2 同步机制

```cpp
    // ========================================================================
    // AIC/AIV 同步机制
    // ========================================================================
    // NPU 有两种核心：
    //   - AIC (AI Core): Cube 单元，专门做矩阵乘法
    //   - AIV (AI Vector): Vector 单元，做向量计算（SwiGLU、量化、搬运）
    //
    // 两者需要协同工作，通过 CrossCore Flag 进行同步：
    //   - AIC_SYNC_AIV_FLAG: AIC 通知 AIV
    //   - AIV_SYNC_AIC_FLAG: AIV 通知 AIC
    // ========================================================================

    // AIV 通知 AIC：AIV 完成了某项工作，AIC 可以继续
    CATLASS_DEVICE void NotifyCube(uint16_t value = 0)
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIV_SYNC_AIC_FLAG + value);
    }

    // AIC 等待 AIV：AIC 等待 AIV 完成
    CATLASS_DEVICE void WaitForVector(uint16_t value = 0)
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIV_SYNC_AIC_FLAG + value);
    }

    // AIC 通知 AIV：AIC 完成了 GEMM，AIV 可以做 Epilogue
    CATLASS_DEVICE void NotifyVector(uint16_t value = 0)
    {
        AscendC::CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_FIX>(AIC_SYNC_AIV_FLAG + value);
    }

    // AIV 等待 AIC：AIV 等待 AIC 完成 GEMM
    CATLASS_DEVICE void WaitForCube(uint16_t value = 0)
    {
        AscendC::CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_V>(AIC_SYNC_AIV_FLAG + value);
    }

    // GMM1 结束同步
    CATLASS_DEVICE void EndSync(bool isVecSetSync)
    {
        if (!isVecSetSync) return;
        if constexpr(g_coreType == AscendC::AIC) {
            WaitForVector();  // AIC 等待 AIV 完成最后的 Epilogue
        }
    }

    // GMM2 结束同步
    CATLASS_DEVICE void EndGMM2Sync(uint32_t syncNum)
    {
        if constexpr(g_coreType == AscendC::AIV) return;
        if (syncNum <= 0) return;
        else if (syncNum == 1) {
            WaitForVector();
        } else {
            // 等待两个 ping-pong 缓冲都完成
            WaitForVector(GMM2_PINGPONG_IDX);
            WaitForVector(1 - GMM2_PINGPONG_IDX);
        }
    }
```

### 4.3 动态 M 处理

```cpp
    // ========================================================================
    // 动态 M 处理：每个专家的 token 数不同
    // ========================================================================
    // GMM 的核心挑战：每个专家处理的 token 数 m_i 不同。
    // 解决方案：使用 groupList 记录每个专家的累积 token 数，
    //          每次处理一个专家时，从 groupList 获取 m_i。
    // ========================================================================

    // 从 groupList 获取当前专家的 token 数
    CATLASS_DEVICE int32_t GetSplitValueFromGroupList(uint32_t groupIdx, uint8_t groupListType)
    {
        int32_t splitValue = 0;
        if (groupListType == 0) {
            // 类型 0：groupList 存储的是累积值
            // splitValue = groupList[groupIdx] - groupList[groupIdx-1]
            int32_t offset = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
            splitValue = offset - preOffset;
            preOffset = offset;
        } else {
            // 类型 1：groupList 直接存储每个专家的 token 数
            splitValue = static_cast<uint64_t>(groupListGm.GetValue(groupIdx));
        }
        return splitValue;
    }

    // 更新全局 buffer 地址（根据 baseOffset 偏移）
    CATLASS_DEVICE void UpdateGlobalBuffer(const Params& params)
    {
        // A 和 A Scale 的偏移（两个核心都需要）
        aGlobal.SetGlobalBuffer(
            (__gm__ ElementA*)params.workspaceInfo.ptrA + Get<IDX_A_OFFSET>(baseOffset));
        aScaleGlobal.SetGlobalBuffer(
            (__gm__ ElementMxScaleA*)params.workspaceInfo.ptrAScale + Get<IDX_A_SCALE_OFFSET>(baseOffset));
        
        if constexpr(g_coreType == AscendC::AIC) {
            // AIC 需要权重和 GMM2 的输入
            bGlobal.SetGlobalBuffer(
                (__gm__ ElementB*)params.bGmAddr + Get<IDX_B_OFFSET>(baseOffset));
            bScaleGlobal.SetGlobalBuffer(
                (__gm__ ElementMxScaleB*)params.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset));
            a2Global.SetGlobalBuffer(
                (__gm__ ElementA2*)params.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset));
            a2ScaleGlobal.SetGlobalBuffer(
                (__gm__ ElementMxScaleA2*)params.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset));
            b2Global.SetGlobalBuffer(
                (__gm__ ElementB2*)params.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset));
            b2ScaleGlobal.SetGlobalBuffer(
                (__gm__ ElementMxScaleB2*)params.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset));
        }
        if constexpr(g_coreType == AscendC::AIV) {
            // AIV 需要 Epilogue 的地址
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset), Get<IDX_C_SCALE_OFFSET>(baseOffset),
                Get<IDX_FLAG_OFFSET>(baseOffset), 0L, 0L, 0L};
            epilogueOp.UpdateGlobalAddr(vecBaseOffset);
            outputGm.SetGlobalBuffer(
                (__gm__ ElementMMadOut2*)params.y2GmAddr + Get<IDX_Y2_OFFSET>(baseOffset));
        }
        // 更新同步标志地址
        groupFlagListGm.SetGlobalBuffer(
            (__gm__ int32_t*)params.workspaceInfo.ptrFlagList + 
            Get<IDX_FLAG_OFFSET>(baseOffset) * INT_CACHELINE * MAX_AICORE_NUM);
        groupFlagListGm2.SetGlobalBuffer(
            (__gm__ int32_t*)params.workspaceInfo.ptrFlagList2 + 
            Get<IDX_FLAG_OFFSET>(baseOffset) * INT_CACHELINE * MAX_AICORE_NUM);
    }

    // 更新 baseOffset（移动到下一个专家）
    CATLASS_DEVICE void UpdateOffset(uint32_t groupIdx)
    {
        // 第一个专家不需要偏移
        if (groupIdx == 0) return;
        
        uint64_t m = Get<M_VALUE>(problemShape);  // 当前专家的 token 数
        uint64_t n = Get<N_VALUE>(problemShape);  // FFN 中间层大小
        uint64_t k = Get<K_VALUE>(problemShape);  // 隐藏层大小
        
        // 计算每个 tensor 的偏移量
        Get<IDX_A_OFFSET>(baseOffset) += m * k;  // A: m*k
        Get<IDX_B_OFFSET>(baseOffset) += n * k;  // B: n*k
        
        // Scale 的偏移（MX 格式：每 32 个元素 1 个 scale，每个 scale 2 字节）
        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(baseOffset) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(baseOffset) += n * scaleK;
        
        // GMM1 输出（SwiGLU 后缩小为 n/2）
        Get<IDX_C_OFFSET>(baseOffset) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(baseOffset) +=
            m * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        
        // 标志位 +1
        Get<IDX_FLAG_OFFSET>(baseOffset) += 1;
        
        // GMM2 的权重和输出
        Get<IDX_B2_OFFSET>(baseOffset) += k * n / SWIGLU_N_HALF;
        Get<IDX_B2_SCALE_OFFSET>(baseOffset) += 
            k * CeilDiv(n / SWIGLU_N_HALF, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(baseOffset) += m * k;
        Get<IDX_M_OFFSET>(baseOffset) += m;
    }

    // 更新当前专家组的参数
    CATLASS_DEVICE bool UpdateGroupParams(const Params& params, uint32_t groupIdx)
    {
        UpdateOffset(groupIdx);  // 更新偏移
        
        // 获取当前专家的 token 数
        int32_t splitValue = GetSplitValueFromGroupList(groupIdx, params.tilingData->groupListType);
        Get<M_VALUE>(problemShape) = splitValue;
        
        // 如果 m=0，说明这个专家没有 token，跳过
        if (Get<M_VALUE>(problemShape) == 0) {
            return false;
        }
        return true;
    }
```

### 4.4 GMM1 实现

```cpp
    // ========================================================================
    // GroupMatmulSwigluQuant: GMM1 + SwiGLU + 量化
    // ========================================================================
    // 功能：
    //   1. GMM1: A[m, k] × B[k, n] → C[m, n] BF16
    //   2. SwiGLU: swish(x) * y → [m, n/2] BF16
    //   3. Quantize: BF16 → FP8 + Scale
    //
    // AIC/AIV 协同：
    //   - AIC: 执行 GMM1（矩阵乘法）
    //   - AIV: 执行 SwiGLU + 量化
    // ========================================================================
    CATLASS_DEVICE void GroupMatmulSwigluQuant(const Params& params)
    {
        // 获取当前专家的 M, N, K
        uint32_t m = Get<M_VALUE>(problemShape);
        uint32_t n = Get<N_VALUE>(problemShape);
        uint32_t k = Get<K_VALUE>(problemShape);
        uint32_t outputN = n / SWIGLU_N_HALF;  // SwiGLU 后输出维度减半
        
        // 更新 Epilogue 的问题形状
        epilogueOp.UpdateNextProblem({m, outputN, k, 0});

        // 创建 BlockMmad 实例（管理 L1/L0 缓存和计算）
        BlockMmad blockMmad(resource);

        // 计算 Scale 的维度
        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        
        // --------------------------------------------------------------------
        // 创建 Layout（描述数据在内存中的布局）
        // --------------------------------------------------------------------
        // layoutA: [m, k] RowMajor
        auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
        // layoutB: [k, n] ColumnMajor
        auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
        // layoutMxScaleA: [m, scaleK] MX 格式
        auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagA, false>(m, scaleK);
        // layoutMxScaleB: [scaleK, n] MX 格式
        auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(scaleK, n);
        // layoutC: [L1_TILE_M, L1_TILE_N] 输出布局
        auto layoutC = tla::MakeLayout<ElementMMadOut, LayoutTagC>(L1_TILE_M, L1_TILE_N);

        // --------------------------------------------------------------------
        // 创建 Tensor（封装 GM 地址 + Layout + 位置信息）
        // --------------------------------------------------------------------
        auto tensorA = tla::MakeTensor(aGlobal, layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(bGlobal, layoutB, Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(aScaleGlobal, layoutMxScaleA, Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(bScaleGlobal, layoutMxScaleB, Arch::PositionGM{});
        // 输出到 UB（Vector Buffer），供 Epilogue 使用
        auto tensorUbFirst = tla::MakeTensor(l0cOutUbFirst, layoutC, Arch::PositionUB{});
        auto tensorUbSecond = tla::MakeTensor(l0cOutUbSecond, layoutC, Arch::PositionUB{});

        // --------------------------------------------------------------------
        // 创建 BlockScheduler（分块调度器）
        // --------------------------------------------------------------------
        // 问题形状：{m, outputN, k}
        // L1 Tile 大小：{L1_TILE_M, L1_TILE_N}
        // BlockScheduler 会自动计算：
        //   - 总共有多少块 (coreLoops)
        //   - 每块的坐标 (blockCoord)
        //   - 每块的实际形状 (actualBlockShape，处理边界)
        BlockScheduler matmulBlockScheduler({m, outputN, k}, MakeCoord(L1_TILE_M, L1_TILE_N));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        
        // 计算当前 Core 的起始 Block 索引（多核负载均衡）
        uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;
        
        // --------------------------------------------------------------------
        // 主循环：遍历所有 Block
        // --------------------------------------------------------------------
        // loopIdx += blockNum 实现多核分发：
        //   Core 0: 处理 0, 32, 64, ...
        //   Core 1: 处理 1, 33, 65, ...
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            // 获取当前块的坐标和形状
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            
            // 计算当前块在 GM 中的位置
            uint32_t mLoc = blockCoord.m() * L1_TILE_M;
            uint32_t nLoc = blockCoord.n() * L1_TILE_N;
            uint32_t kLoc = blockCoord.k() * L1_TILE_K;

            // ====================================================================
            // AIC (Cube 核) 执行 GMM1
            // ====================================================================
            if constexpr(g_coreType == AscendC::AIC) {
                // ---- 等待 Dispatch 完成 ----
                if (loopIdx == startLoopIdx) {
                    // 等待 AIV 完成所有 token 的 Dispatch
                    uint32_t targetValue = params.tilingData->EP * dispatchBlockNumPerEP;
                    __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm2.GetPhyAddr();
                    while(targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                        // 忙等（带超时检测）
                        int64_t st = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - st < 100) {}
                    }
                }
                
                // 等待上一轮的 AIV 完成
                if (isVecSetSyncCom) {
                    WaitForVector();
                }
                
                // ---- 获取当前 Block 的 Tile 视图 ----
                // GetTile 不会搬运数据，只是创建一个"视图"
                // 告诉 BlockMmad 数据在 GM 中的哪个位置
                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(mLoc, kLoc),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockMxScaleA = GetTile(
                    tensorMxScaleA,
                    tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                    tla::MakeShape(actualBlockShape.m(), CeilDiv(actualBlockShape.k(), MXFP_SCALE_GROUP_NUM)));
                auto tensorBlockUbFirst = GetTile(
                    tensorUbFirst, tla::MakeCoord(0, 0),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                auto tensorBlockUbSecond = GetTile(
                    tensorUbSecond, tla::MakeCoord(0, 0),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));

                // ---- 对 SWIGLU_N_HALF=2 个权重块分别计算 ----
                // SwiGLU 需要将输出分成两部分：
                //   C[:, :n/2] = A × B[:, :n/2]
                //   C[:, n/2:] = A × B[:, n/2:]
                for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
                    auto tensorBlockB = GetTile(
                        tensorB,
                        tla::MakeCoord(kLoc, nLoc + weightBlock * outputN),
                        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                    auto tensorBlockMxScaleB = GetTile(
                        tensorBlockMxScaleB,
                        tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
                        tla::MakeShape(CeilDiv(actualBlockShape.k(), MXFP_SCALE_GROUP_NUM), actualBlockShape.n()));
                    
                    // 调用 BlockMmad 执行计算
                    // 内部自动完成：GM→L1→L0→MMA→L0→UB
                    blockMmad(tensorBlockA, tensorBlockB, 
                              weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond,
                              actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
                }
                
                // 通知 AIV：GMM1 完成，可以做 SwiGLU + 量化
                NotifyVector();
            }
            isVecSetSyncCom = true;
            
            // ====================================================================
            // AIV (Vector 核) 执行 SwiGLU + 量化
            // ====================================================================
            if constexpr(g_coreType == AscendC::AIV) {
                if (subBlockIdx == 0) {
                    // 计算 Epilogue 的形状和偏移
                    AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t> epilogueShape{
                        actualBlockShape.m(), actualBlockShape.n(), 0, 0};
                    AscendC::Std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> epilogueOffset{
                        mLoc * outputN + nLoc,  // yOffset
                        mLoc * CeilDiv(outputN, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE +
                            CeilDiv(nLoc, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE,  // yScaleOffset
                        0, 0, 0, 0
                    };
                    
                    // 等待 AIC：GMM1 完成
                    WaitForCube();
                    
                    // 执行 SwiGLU + 量化
                    epilogueOp(epilogueShape, epilogueOffset);
                    
                    // 通知 AIC：Epilogue 完成
                    NotifyCube();
                }
            }
        }
        // 更新起始 Block 索引（用于下一轮）
        startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
    }
```

### 4.5 GMM2 实现

```cpp
    // ========================================================================
    // GroupMatmul2: GMM2 + Combine（跨卡写回）
    // ========================================================================
    // 功能：
    //   1. GMM2: A2[m, n/2] × B2[n/2, k] → C2[m, k] BF16
    //   2. Combine: 将结果按原始 rank 写回远端 peermem
    //
    // 与 GMM1 的区别：
    //   - 使用 Ping-Pong 双缓冲（GMM2_PINGPONG_IDX）
    //   - 需要等待 GMM1 完成（通过 groupFlagListGm）
    //   - 输出需要按 rank 分散写回（Combine）
    // ========================================================================
    CATLASS_DEVICE void GroupMatmul2(const Params& params, uint32_t groupIdx)
    {
        // 注意：GMM2 的 M/N/K 与 GMM1 不同
        uint32_t m = Get<M_VALUE>(problemShape);    // 不变
        uint32_t n = Get<K_VALUE>(problemShape);    // GMM2 的 N 是 GMM1 的 K
        uint32_t k = Get<N_VALUE>(problemShape) / SWIGLU_N_HALF;  // GMM2 的 K 是 GMM1 的 N/2

        BlockMmad2 blockMmad(resource);

        auto scaleK = CeilDiv(k, MXFP_DIVISOR_SIZE) * MXFP_MULTI_BASE_SIZE;
        
        // 创建 Layout
        auto layoutA = tla::MakeLayout<ElementA2, LayoutTagA2>(m, k);
        auto layoutB = tla::MakeLayout<ElementB2, LayoutTagB2>(k, n);
        auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA2, LayoutTagA2, false>(m, scaleK);
        auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB2, LayoutTagB2, true>(scaleK, n);
        auto layoutC = tla::MakeLayout<ElementMMadOut2, LayoutTagC2>(L1_TILE_M2, L1_TILE_N2);

        // 创建 Tensor
        auto tensorA = tla::MakeTensor(a2Global, layoutA, Arch::PositionGM{});
        auto tensorB = tla::MakeTensor(b2Global, layoutB, Arch::PositionGM{});
        auto tensorMxScaleA = tla::MakeTensor(a2ScaleGlobal, layoutMxScaleA, Arch::PositionGM{});
        auto tensorMxScaleB = tla::MakeTensor(b2ScaleGlobal, layoutMxScaleB, Arch::PositionGM{});
        auto tensorUbFirst = tla::MakeTensor(l0cOutUbGMM2First, layoutC, Arch::PositionUB{});
        auto tensorUbSecond = tla::MakeTensor(l0cOutUbGMM2Second, layoutC, Arch::PositionUB{});

        // 创建 BlockScheduler
        BlockScheduler matmulBlockScheduler({m, n, k}, MakeCoord(L1_TILE_M2, L1_TILE_N2));
        uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();
        uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;
        uint32_t currentExpertM = cumsumMM((params.tilingData->EP - 1) * params.tilingData->expertPerRank + groupIdx);
        uint32_t preSrcExpertSum = 0;

        // 主循环
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
            GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
            uint32_t mLoc = blockCoord.m() * L1_TILE_M2;
            uint32_t nLoc = blockCoord.n() * L1_TILE_N2;
            uint32_t kLoc = blockCoord.k() * L1_TILE_K2;

            // 选择 Ping-Pong 缓冲
            auto tensorUb = GMM2_PINGPONG_IDX == 0 ? tensorUbFirst : tensorUbSecond;
            auto l0cOutUbGMM2 = GMM2_PINGPONG_IDX == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;
            
            // ====================================================================
            // AIC (Cube 核) 执行 GMM2
            // ====================================================================
            if constexpr(g_coreType == AscendC::AIC) {
                // ---- 等待 GMM1 完成 ----
                if (loopIdx == startLoopIdx) {
                    // 计算 GMM1 的总 Block 数
                    BlockScheduler gmmBlockScheduler({m, k, n}, MakeCoord(L1_TILE_M, L1_TILE_N));
                    uint32_t targetLoops = gmmBlockScheduler.GetCoreLoops();
                    __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm.GetPhyAddr();
                    while(targetLoops != AscendC::ReadGmByPassDCache(flagValueAddr)) {
                        int64_t st = AscendC::GetSystemCycle();
                        while (AscendC::GetSystemCycle() - st < 100) {}
                    }
                }
                
                // 等待上一轮 AIV 完成 Combine
                if (VecSetSyncCom2 >= GMM2_PINGPONG_COUNT) {
                    WaitForVector(GMM2_PINGPONG_IDX);
                }
                
                // 获取 Tile 视图
                auto tensorBlockA = GetTile(
                    tensorA, tla::MakeCoord(mLoc, kLoc),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
                auto tensorBlockMxScaleA = GetTile(
                    tensorMxScaleA,
                    tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
                    tla::MakeShape(actualBlockShape.m(), CeilDiv(actualBlockShape.k(), MXFP_SCALE_GROUP_NUM)));
                auto tensorBlockUb = GetTile(
                    tensorUb, tla::MakeCoord(0, 0),
                    tla::MakeShape(actualBlockShape.m(), actualBlockShape.n()));
                auto tensorBlockB = GetTile(
                    tensorB, tla::MakeCoord(kLoc, nLoc),
                    tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
                auto tensorBlockMxScaleB = GetTile(
                    tensorMxScaleB,
                    tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc),
                    tla::MakeShape(CeilDiv(actualBlockShape.k(), MXFP_SCALE_GROUP_NUM), actualBlockShape.n()));
                
                // 执行 GMM2
                blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb,
                          actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
                
                // 通知 AIV：GMM2 完成
                NotifyVector(GMM2_PINGPONG_IDX);
            }
            VecSetSyncCom2++;
            
            // ====================================================================
            // AIV (Vector 核) 执行 Combine（跨卡写回）
            // ====================================================================
            if constexpr(g_coreType == AscendC::AIV) {
                if (subBlockIdx == 0) {
                    // 等待 AIC：GMM2 完成
                    WaitForCube(GMM2_PINGPONG_IDX);
                    
                    // ----------------------------------------------------------------
                    // Combine 逻辑：将结果按原始 rank 写回远端 peermem
                    // ----------------------------------------------------------------
                    // 当前 Block 覆盖的 token 范围：[stTile, edTile)
                    int32_t lenTile = actualBlockShape.m();
                    int32_t stTile = mLoc;
                    int32_t edTile = stTile + lenTile;
                    int32_t preSumRankInExpert = 0;  // 累计前面 rank 的 token 数
                    int32_t tileOffset = 0;          // 当前 Block 内的偏移
                    
                    // 遍历所有 EP，找到与当前 Block 有交集的 rank
                    for (int32_t dstEpIdx = 0; dstEpIdx < params.tilingData->EP; dstEpIdx++) {
                        // 获取目标 rank 在本专家中的 token 数
                        uint32_t lenRankInExpert = tokenPerExpert(
                            tokenPerExpertLayout(dstEpIdx, params.tilingData->rank, groupIdx));
                        
                        // 目标 rank 在全局专家中的偏移
                        uint32_t dstExpertOffset = preSumBeforeRank(
                            dstEpIdx * params.tilingData->expertPerRank + groupIdx);
                        
                        // 目标 rank 在本地的 token 范围：[stRankInExpert, edRankInExpert)
                        uint32_t stRankInExpert = preSumRankInExpert;
                        uint32_t edRankInExpert = stRankInExpert + lenRankInExpert;
                        preSumRankInExpert += lenRankInExpert;

                        // 计算交集
                        if (stRankInExpert >= edTile) break;  // 无交集，后续也无
                        else if (edRankInExpert <= stTile) continue;  // 无交集，继续

                        // 交集范围：[stData, edData)
                        uint32_t stData = tla::max(stRankInExpert, stTile); 
                        uint32_t edData = tla::min(edRankInExpert, edTile);
                        uint32_t lenData = edData - stData;
                        if (lenData <= 0) continue;

                        // 计算目标 rank 中的偏移
                        uint32_t dstOffsetInExpert = 0;
                        if (stTile > stRankInExpert) {
                            dstOffsetInExpert = stTile - stRankInExpert;
                        }
                
                        // 设置远端地址
                        AscendC::GlobalTensor<ElementMMadOut2> gmRemoteD;
                        __gm__ void* dstPeermemPtr = params.shmem(
                            params.peermemInfo.ptrD, dstEpIdx);
                        gmRemoteD.SetGlobalBuffer(
                            reinterpret_cast<__gm__ ElementMMadOut2*>(dstPeermemPtr));
                        
                        // 计算远端偏移
                        uint64_t gmDstOffset = (dstOffsetInExpert + dstExpertOffset) * n + nLoc;
                        auto gmTileD = gmRemoteD[gmDstOffset];

                        // 写入远端（带 stride，处理不连续的行）
                        AscendC::DataCopyExtParams ub2GmParams{1, 0, 0, 0, 0};
                        ub2GmParams.blockCount = lenData;
                        ub2GmParams.blockLen = actualBlockShape.n() * sizeof(ElementMMadOut2);
                        ub2GmParams.dstStride = (n - actualBlockShape.n()) * sizeof(ElementMMadOut2);

                        AscendC::DataCopyPad(gmTileD, 
                            l0cOutUbGMM2[tileOffset * actualBlockShape.n()], ub2GmParams);
                        tileOffset += lenData;
                    }

                    // 通知 AIC：Combine 完成
                    NotifyCube(GMM2_PINGPONG_IDX);
                }
            }
            // 切换 Ping-Pong 缓冲
            GMM2_PINGPONG_IDX = 1 - GMM2_PINGPONG_IDX;
        }

        preSrcExpertSum += currentExpertM;
        startBlockIdx = (startBlockIdx + coreLoops) % blockNum;
    }
```

### 4.6 operator() 主流程

```cpp
    // ========================================================================
    // operator(): Kernel 入口
    // ========================================================================
    // 完整流程：
    //   1. 初始化
    //   2. InitRouting + CrossRank Sync
    //   3. GMM1 Loop (Dispatch + GMM1 + SwiGLU + Quant)
    //   4. GMM2 Loop (GMM2 + Combine)
    //   5. Finalize Routing
    // ========================================================================
    CATLASS_DEVICE void operator()(const Params& params)
    {
        // --------------------------------------------------------------------
        // Step 1: 初始化
        // --------------------------------------------------------------------
        InitParamsAndTensor(params);
        
        // 计算当前 rank 的 tokenPerExpert 地址
        GM_ADDR tokenPerExpertCurRank = params.peermemInfo.ptrTokenPerExpert + 
            tokenPerExpertLayout(params.tilingData->rank, 0, 0) * sizeof(int32_t);

        // 保存溢出模式（arch35 特殊处理）
        #if (__NPU_ARCH__ == 3510)
            int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
        #endif
        
        // --------------------------------------------------------------------
        // Step 2: InitRouting + CrossRank Sync
        // --------------------------------------------------------------------
        // InitRouting: 按 expertIdx 重排 token
        moe_init_routing_v3<bfloat16_t, fp8_e4m3fn_t>(
            params.aGmAddr, params.expertIdxGmAddr, nullptr, nullptr,
            params.peermemInfo.ptrA0, params.workspaceInfo.expandedRowIdx,
            tokenPerExpertCurRank, nullptr,
            &params.tilingData->moeInitRoutingTilingData, 
            params.tilingData->moeInitRoutingTilingData.tilingKey);

        AscendC::SyncAll<true>();
        
        // CrossRank Sync: 多卡同步 tokenPerExpert，计算 cumsumMM
        CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(params, tokenPerExpertCurRank);
        AivSyncAllForAic();  // AIV 同步所有 Core，通知 AIC 可以继续
        
        // --------------------------------------------------------------------
        // Step 3: GMM1 Loop
        // --------------------------------------------------------------------
        uint32_t expertPerRank = params.tilingData->expertPerRank;
        for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
            // 更新当前专家的参数
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;  // m=0，跳过
            }
            UpdateGlobalBuffer(params);
            
            // AIV subBlockIdx==1: 执行 Dispatch（跨卡拉取 token）
            if constexpr(g_coreType == AscendC::AIV) {
                if(subBlockIdx == 1) {
                    Dispatch(params, groupIdx);
                }
            }
            
            // subBlockIdx==0: 执行 GMM1 + SwiGLU + Quant
            if (subBlockIdx == 0) {
                GroupMatmulSwigluQuant(params);
            }
        }
        EndSync(isVecSetSyncCom);  // 等待 GMM1 全部完成
        
        // --------------------------------------------------------------------
        // Step 4: GMM2 Loop
        // --------------------------------------------------------------------
        // 重置偏移
        preOffset = 0;
        baseOffset = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        
        for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
            if (!UpdateGroupParams(params, groupIdx)) {
                continue;
            }
            UpdateGlobalBuffer(params);
            
            if (subBlockIdx == 0) {
                GroupMatmul2(params, groupIdx);
            }
        }

        EndGMM2Sync(VecSetSyncCom2);  // 等待 GMM2 全部完成
        AscendC::SyncAll<true>();
        
        // --------------------------------------------------------------------
        // Step 5: Finalize
        // --------------------------------------------------------------------
        if constexpr(g_coreType == AscendC::AIV) {
            ResetTokenPerExpert(params);  // 重置 tokenPerExpert
            params.shmem.CrossRankSync();  // 跨卡同步
        }
        AscendC::SyncAll<true>();

        // Finalize Routing: 按 probs 加权聚合
        moe_finalize_routing_v2<bfloat16_t>(
            params.peermemInfo.ptrD,              // expandedX
            params.workspaceInfo.expandedRowIdx,  // expandedRowIdx
            nullptr, nullptr, nullptr,            // x1, x2, bias (未使用)
            params.probsGmAddr,                   // scales (probs)
            params.expertIdxGmAddr,               // expertIdx
            params.y2GmAddr,                      // y (输出)
            params.workspaceInfo.ptrA,            // workspace
            &params.tilingData->moeFinalizeRoutingTilingData,
            params.tilingData->moeFinalizeRoutingTilingKey
        );

        // 恢复溢出模式（arch35）
        #if (__NPU_ARCH__ == 3510)
            SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
        #endif
    }
```

---

## 5. 关键设计总结

### 5.1 Tiling 策略

```
GMM1:
  L1 Tile: [256, 256, 448]  → 从 GM 搬运 256×448 的 A 和 448×256 的 B
  L0 Tile: [256, 256, 128]  → 分 4 批 (448/128) 送入 L0 计算

GMM2:
  L1 Tile: [256, 256, 448]  → 从 GM 搬运 256×448 的 A2 和 448×256 的 B2
  L0 Tile: [256, 256, 128]  → 分 4 批送入 L0 计算
```

### 5.2 AIC/AIV 协同

```
AIC (Cube):                    AIV (Vector):
  │                              │
  ├─ 等待 Dispatch 完成           ├─ 执行 Dispatch
  ├─ 执行 GMM1                    │  ├─ 跨卡拉取 token
  ├─ NotifyVector() ────────────→│  └─ AtomicAddFlag
  │                              │
  │                              ├─ WaitForCube()
  │                              ├─ 执行 SwiGLU + 量化
  │                              ├─ NotifyCube() ──────────→
  │                              │
  ├─ WaitForVector() ←───────────│
  ├─ 执行 GMM2                    │
  ├─ NotifyVector(GMM2_IDX) ────→│
  │                              ├─ WaitForCube(GMM2_IDX)
  │                              ├─ 执行 Combine (远端写回)
  │                              ├─ NotifyCube(GMM2_IDX) ──→
  │                              │
  ├─ WaitForVector() ←───────────│
```

### 5.3 动态 M 处理

```
问题：每个专家的 token 数 m_i 不同
解决：
  1. groupList 存储每个专家的累积 token 数
  2. UpdateGroupParams() 获取当前专家的 m_i
  3. UpdateOffset() 计算当前专家的数据偏移
  4. BlockScheduler 根据 m_i 自动切块
```

### 5.4 Ping-Pong 双缓冲

```
GMM2 使用双缓冲实现计算与通信重叠：

迭代 0 (PINGPONG_IDX=0):
  AIC: 计算 → l0cOutUbGMM2First
  AIV: 等待 → 处理 l0cOutUbGMM2First → 写回远端

迭代 1 (PINGPONG_IDX=1):
  AIC: 计算 → l0cOutUbGMM2Second
  AIV: 等待 → 处理 l0cOutUbGMM2Second → 写回远端

迭代 2 (PINGPONG_IDX=0):
  AIC: 计算 → l0cOutUbGMM2First (已空闲)
  AIV: 等待 → 处理 l0cOutUbGMM2First → 写回远端
```
我将详细展开这两个点。
这两个问题问得非常核心，直接触及了 MoE 算子在 NPU 上实现的两个关键难点：激活函数对计算图的重构 和 分布式 Token 的汇聚同步。
---
问题 1：tensorUbFirst 和 tensorUbSecond 是 Ping-Pong 吗？
结论：不是 Ping-Pong，而是 SwiGLU 激活函数的硬性要求。
1.1 为什么需要两个 Buffer？
在标准的 GEMM 中，我们通常只需要一个输出 Buffer。但在 MegaMoe 中，GMM1 的输出不是直接给下一个 GEMM，而是要经过 SwiGLU 激活函数。
SwiGLU 的数学公式是：
$$ \text{SwiGLU}(x) = \text{Swish}(xW_1) \otimes (xW_2) $$
注意这里有两个独立的矩阵乘法：
1. $C_1 = A \times B_1$ （用于 Swish 激活）
2. $C_2 = A \times B_2$ （用于 Gate 门控）
代码中的体现：
// SWIGLU_N_HALF = 2，意味着权重被切成了两半
for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
    // weightBlock == 0 时：计算 C1，存入 tensorBlockUbFirst
    // weightBlock == 1 时：计算 C2，存入 tensorBlockUbSecond
    blockMmad(tensorBlockA, tensorBlockB, 
              weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond, ...);
}
为什么不能复用同一个 Buffer？
因为 AIC（Cube核）计算 $C_2$ 的时候，$C_1$ 的数据必须保留。如果复用同一个 Buffer，$C_2$ 会覆盖 $C_1$，等 AIC 算完通知 AIV 时，AIV 拿到的就只有 $C_2$ 了，无法计算 $\text{Swish}(C_1) \times C_2$。
1.2 和 Ping-Pong 的区别
特性	这里的 First/Second	真正的 Ping-Pong (如 GMM2)
目的	同时保存两个计算结果供 AIV 消费	掩盖计算与通信的延迟
使用时机	同一个 Block 内，顺序写入，同时读取	不同 Block 迭代间，交替使用
数据依赖	AIV 需要同时访问 First 和 Second	AIV 只访问当前 Ping-Pong 索引的 Buffer
切换机制	不切换，固定对应 $C_1$ 和 $C_2$	GMM2_PINGPONG_IDX = 1 - GMM2_PINGPONG_IDX
1.3 AIC/AIV 协同流程
AIC (Cube)                          AIV (Vector)
  │                                   │
  ├─ 计算 C1 → l0cOutUbFirst          │
  ├─ 计算 C2 → l0cOutUbSecond         │
  ├─ NotifyVector() ────────────────→ │
  │                                   ├─ WaitForCube()
  │                                   ├─ 读取 First 和 Second
  │                                   ├─ 计算 Swish(First) × Second
  │                                   └─ 量化输出
  │                                   ├─ NotifyCube() ──────────→
  ├─ WaitForVector() ←───────────────│
  └─ (继续下一个 Block)
---
问题 2：targetValue = EP * dispatchBlockNumPerEP 是什么含义？
结论：这是 MoE 分布式计算的“数据汇聚屏障”。AIC 必须等待当前专家的所有 Token 从所有卡上完整拉取到本地，才能开始计算。
2.1 为什么不是“某个专家的 Token 到了”就开始？
你提到的“某个专家的 Token 都到了”其实是对的，但这个“都到了”在分布式环境下非常复杂。
在 MoE 中，一个专家（Expert）虽然物理上位于某张卡（Rank）上，但分配给它的 Token 可能来自所有卡。
假设：
- 总共有 EP = 8 张卡
- 当前卡（Rank 0）上有 4 个专家
- Token 0 在 Rank 3，但被路由到了 Rank 0 的 Expert 2
- Token 1 在 Rank 5，也被路由到了 Rank 0 的 Expert 2
那么 Rank 0 的 Expert 2 要开始计算，必须等待 Rank 0~7 所有卡 把属于 Expert 2 的 Token 都发送过来。
2.2 EP * dispatchBlockNumPerEP 的拆解
uint32_t targetValue = params.tilingData->EP * dispatchBlockNumPerEP;
- EP：总卡数（例如 8）。表示需要从 8 张卡上拉取数据。
- dispatchBlockNumPerEP：每张卡的任务被切分的块数（例如 2）。因为一张卡上可能有很多 Token 要给当前专家，一次搬不完，要分块搬。
总块数 = 8 张卡 × 每卡 2 块 = 16 块。
Dispatch 函数的工作就是处理这 16 块数据：
// Dispatch 循环遍历所有需要拉取的块
for(int32_t idx = blockIdx; idx < tilingData->EP * dispatchBlockNumPerEP; idx += blockNum) {
    uint32_t dstEpIdx = idx / dispatchBlockNumPerEP;  // 目标卡 ID (0~7)
    // ... 从 dstEpIdx 卡拉取数据 ...
    
    // 每拉完一块，原子增加标志位
    AtomicAddFlag((__gm__ int32_t*)groupFlagListGm2.GetPhyAddr(), ...);
}
2.3 AIC 的等待逻辑
if (loopIdx == startLoopIdx) {
    // 等待标志位等于总块数
    while(targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
        // 忙等...
    }
}
为什么只在 loopIdx == startLoopIdx 等待一次？
因为 GMM1 是按 Block 循环计算的，但数据依赖只在第一个 Block 开始前检查。
- 第一个 Block 开始前：确保所有 Token 都已到位。
- 后续 Block：数据已经在本地 GM 中，直接计算即可。
2.4 类比理解
想象你在组织一场全国联考（GMM1 计算）：
- 你是北京考点的负责人（当前 Rank）。
- 考场里有 4 个教室（4 个专家）。
- 每个教室的考生来自全国各地（Token 分布在所有 EP 张卡）。
规则：
1. 你不能等北京的考生到了就开始考试，必须等全国分配到这个教室的考生都到北京。
2. 全国有 8 个省份（EP=8），每个省份分 2 批大巴送考生（dispatchBlockNumPerEP=2）。
3. 总共要等 16 批大巴到达（targetValue = 16）。
4. 每到大巴，门卫就敲一下钟（AtomicAddFlag）。
5. 钟响 16 次后，监考老师（AIC）才能发卷开始考试（GMM1）。
这就是 targetValue = EP * dispatchBlockNumPerEP 的真实含义：分布式 Token 汇聚完成的信号。