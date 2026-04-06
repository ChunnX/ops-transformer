# Catlass 设计范式详解

## 概述

Catlass (CANN Template Linear Algebra Subroutine System) 是华为为 Ascend NPU 设计的高性能 GEMM 库，采用 C++ 模板元编程和 TLA (Tensor Layout Algebra) 抽象。

## 核心设计原则

### 1. 编译期多态

Catlass 大量使用模板参数和 `constexpr`，将计算决策移到编译期：

```cpp
template <class BlockMmad_, class BlockMmad2_, class BlockEpilogue_, class BlockScheduler_>
class KernelGmmSwiGluMixOnlineDynamic {
    // 所有类型在编译期确定
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    // ...
};
```

### 2. 层次化抽象

```
Kernel 层    ──→  编排整个计算流程
    │
Block 层     ──→  管理 L1 缓存的 tile 调度
    │
Tile 层      ──→  管理 L0 缓存的矩阵乘法
    │
硬件层       ──→  直接调用 AscendC API
```

### 3. TLA (Tensor Layout Algebra)

TLA 提供统一的张量布局抽象，支持：
- RowMajor / ColumnMajor
- zN / nZ (分形布局)
- MXFP Scale 布局

## 核心组件

### GemmCoord

**文件**: `catlass/gemm_coord.hpp`

```cpp
struct GemmCoord : public Coord<3, uint32_t> {
    static constexpr int M_INDEX = 0;  // 行
    static constexpr int N_INDEX = 1;  // 列
    static constexpr int K_INDEX = 2;  // 内维
    
    Index m() const { return this->At(M_INDEX); }
    Index n() const { return this->At(N_INDEX); }
    Index k() const { return this->At(K_INDEX); }
};
```

**用途**: 表示 GEMM 问题的 (M, N, K) 形状

### BlockScheduler

**文件**: `catlass/gemm/block/block_swizzle.hpp`

```cpp
template <uint32_t SwizzleOffset = 1, uint32_t SwizzleDirection = 0>
struct GemmIdentityBlockSwizzle {
    GemmCoord problemShape;   // 问题形状
    MatrixCoord tileMN;       // tile 大小
    MatrixCoord loopsMN;      // 循环次数
    
    // 获取总 tile 数
    CATLASS_DEVICE uint32_t GetCoreLoops() const {
        return loopsMN.row() * loopsMN.column();
    }
    
    // 根据 taskIdx 获取 tile 坐标
    CATLASS_DEVICE GemmCoord GetBlockCoord(uint32_t taskIdx);
    
    // 获取实际 tile 形状 (处理边界)
    CATLASS_DEVICE GemmCoord GetActualBlockShape(GemmCoord blockCoord);
};
```

**用途**: 将大矩阵分解为 tile，管理多核调度

### Resource

**文件**: `catlass/arch/resource.hpp`

```cpp
template<class ArchTag>
struct Resource {
    AscendC::TPipe pipe;
    
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A1> l1Buf;      // L1 缓存
    LocalTensorBuffer<ArchTag, AscendC::TPosition::A2> l0ABuf;     // L0A 缓存
    LocalTensorBuffer<ArchTag, AscendC::TPosition::B2> l0BBuf;     // L0B 缓存
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2> btBuf;      // BT 缓存
    LocalTensorBuffer<ArchTag, AscendC::TPosition::CO1> l0CBuf;    // L0C 缓存
    LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf; // UB 缓存
    LocalTensorBuffer<ArchTag, AscendC::TPosition::C2PIPE2GM> fpBuf; // FP 缓存
};
```

**用途**: 管理硬件资源（缓存、管道）

### Tensor

**文件**: `tla/tensor.hpp`

```cpp
template <class BuiltinTensor, class Layout_, class Coord_, AscendC::TPosition Position>
struct Tensor {
    using Element = typename BuiltinTensor::PrimType;
    using Layout = Layout_;
    using Coord = Coord_;
    static constexpr AscendC::TPosition position = Position;
    
    // 数据访问
    CATLASS_HOST_DEVICE constexpr decltype(auto) data() const { return get<0>(rep_); }
    
    // 布局访问
    CATLASS_HOST_DEVICE constexpr decltype(auto) layout() const { return get<1>(rep_); }
    
    // 坐标访问
    CATLASS_HOST_DEVICE constexpr decltype(auto) coord() const { return get<2>(rep_); }
    
    // 索引访问
    template <class CoordArg>
    CATLASS_HOST_DEVICE constexpr decltype(auto) operator()(CoordArg const& coord_arg) const;
    
    tla::tuple<BuiltinTensor, Layout, Coord> rep_;
};
```

**用途**: 封装张量数据、布局和坐标

### Layout

**文件**: `tla/layout.hpp`

```cpp
template <class Shape, class Stride, class OriginShape = detail::MakeOriginShapeType<Stride>>
struct Layout : private tla::tuple<Shape, Stride, OriginShape> {
    static constexpr int rank = rank_v<Stride>;
    static constexpr int depth = depth_v<Stride>;
    
    // 获取形状
    template <int... I> decltype(auto) shape() const;
    
    // 获取步长
    template <int... I> decltype(auto) stride() const;
    
    // 获取原始形状
    template <int... I> decltype(auto) originShape() const;
    
    // 坐标转偏移
    template <class Coord>
    CATLASS_HOST_DEVICE constexpr auto operator()(Coord const& coord) const {
        return crd2offset(coord, shape(), stride());
    }
};
```

**用途**: 描述张量在内存中的布局

## MegaMoe operator() 实现详解

### 代码结构

```cpp
CATLASS_DEVICE void operator()(const Params& params)
{
    // 1. 初始化
    InitParamsAndTensor(params);
    
    // 2. InitRouting
    moe_init_routing_v3<bfloat16_t, fp8_e4m3fn_t>(...);
    
    // 3. 跨 Rank 同步
    CrossRankSyncAndlocalTokenPerExpertAllGatherAndGetSumPreRankV2(params, ...);
    AivSyncAllForAic();
    
    // 4. 对每个专家组循环
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
        if (!UpdateGroupParams(params, groupIdx)) continue;
        UpdateGlobalBuffer(params);
        
        // Dispatch
        if constexpr(g_coreType == AscendC::AIV) {
            if(subBlockIdx == 1) {
                Dispatch(params, groupIdx);
            }
        }
        
        // GMM1 + SwiGLU + Quant
        if (subBlockIdx == 0) {
            GroupMatmulSwigluQuant(params);
        }
    }
    
    // 5. GMM2 + Combine
    for (uint32_t groupIdx = 0; groupIdx < expertPerRank; groupIdx++) {
        if (!UpdateGroupParams(params, groupIdx)) continue;
        UpdateGlobalBuffer(params);
        
        if (subBlockIdx == 0) {
            GroupMatmul2(params, groupIdx);
        }
    }
    
    // 6. Finalize
    moe_finalize_routing_v2<bfloat16_t>(...);
}
```

### 关键设计模式

#### 1. 编译期核心类型分发

```cpp
if constexpr(g_coreType == AscendC::AIC) {
    // Cube 核：执行 GEMM
    blockMmad(tensorBlockA, tensorBlockB, tensorBlockUb, ...);
} else if constexpr(g_coreType == AscendC::AIV) {
    // Vector 核：执行 SwiGLU + 量化
    epilogueOp(epilogueShape, epilogueOffset);
}
```

#### 2. BlockScheduler 调度

```cpp
// 创建 BlockScheduler
BlockScheduler matmulBlockScheduler({m, n, k}, MakeCoord(L1_TILE_M, L1_TILE_N));

// 计算总 tile 数
uint32_t coreLoops = matmulBlockScheduler.GetCoreLoops();

// 计算当前核的起始 tile
uint32_t startLoopIdx = (blockIdx < startBlockIdx ? blockIdx + blockNum : blockIdx) - startBlockIdx;

// 遍历当前核的 tile
for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += blockNum) {
    // 获取 tile 坐标
    GemmCoord blockCoord = matmulBlockScheduler.GetBlockCoord(loopIdx);
    GemmCoord actualBlockShape = matmulBlockScheduler.GetActualBlockShape(blockCoord);
    
    // 计算位置
    uint32_t mLoc = blockCoord.m() * L1_TILE_M;
    uint32_t nLoc = blockCoord.n() * L1_TILE_N;
    uint32_t kLoc = blockCoord.k() * L1_TILE_K;
}
```

#### 3. Tensor 和 Layout 使用

```cpp
// 创建布局
auto layoutA = tla::MakeLayout<ElementA, LayoutTagA>(m, k);
auto layoutB = tla::MakeLayout<ElementB, LayoutTagB>(k, n);
auto layoutMxScaleA = tla::MakeMxScaleLayout<ElementMxScaleA, LayoutTagA, false>(m, scaleK);
auto layoutMxScaleB = tla::MakeMxScaleLayout<ElementMxScaleB, LayoutTagB, true>(scaleK, n);
auto layoutC = tla::MakeLayout<ElementMMadOut, LayoutTagC>(L1_TILE_M, L1_TILE_N);

// 创建张量
auto tensorA = tla::MakeTensor(aGlobal, layoutA, Arch::PositionGM{});
auto tensorB = tla::MakeTensor(bGlobal, layoutB, Arch::PositionGM{});
auto tensorMxScaleA = tla::MakeTensor(aScaleGlobal, layoutMxScaleA, Arch::PositionGM{});
auto tensorMxScaleB = tla::MakeTensor(bScaleGlobal, layoutMxScaleB, Arch::PositionGM{});
auto tensorUbFirst = tla::MakeTensor(l0cOutUbFirst, layoutC, Arch::PositionUB{});
auto tensorUbSecond = tla::MakeTensor(l0cOutUbSecond, layoutC, Arch::PositionUB{});

// 获取 tile
auto tensorBlockA = GetTile(
    tensorA, tla::MakeCoord(mLoc, kLoc),
    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k()));
auto tensorBlockMxScaleA = GetTile(
    tensorMxScaleA,
    tla::MakeCoord(mLoc, kLoc / MXFP_SCALE_GROUP_NUM),
    tla::MakeShape(actualBlockShape.m(), CeilDiv(actualBlockShape.k(), MXFP_SCALE_GROUP_NUM)));
```

#### 4. BlockMmad 调用

```cpp
// 创建 BlockMmad
BlockMmad blockMmad(resource);

// 执行 GEMM
for (uint32_t weightBlock = 0; weightBlock < SWIGLU_N_HALF; ++weightBlock) {
    auto tensorBlockB = GetTile(
        tensorB,
        tla::MakeCoord(kLoc, nLoc + weightBlock * outputN),
        tla::MakeShape(actualBlockShape.k(), actualBlockShape.n()));
    auto tensorBlockMxScaleB = GetTile(
        tensorMxScaleB,
        tla::MakeCoord(kLoc / MXFP_SCALE_GROUP_NUM, nLoc + weightBlock * outputN),
        tla::MakeShape(CeilDiv(actualBlockShape.k(), MXFP_SCALE_GROUP_NUM), actualBlockShape.n()));
    
    blockMmad(tensorBlockA, tensorBlockB, 
              weightBlock == 0 ? tensorBlockUbFirst : tensorBlockUbSecond,
              actualBlockShape, tensorBlockMxScaleA, tensorBlockMxScaleB);
}
```

#### 5. AIC/AIV 同步

```cpp
// AIC 等待 Dispatch 完成
if (loopIdx == startLoopIdx) {
    uint32_t targetValue = params.tilingData->EP * dispatchBlockNumPerEP;
    __gm__ int32_t* flagValueAddr = (__gm__ int32_t*)groupFlagListGm2.GetPhyAddr();
    while(targetValue != AscendC::ReadGmByPassDCache(flagValueAddr)) {
        int64_t st = AscendC::GetSystemCycle();
        while (AscendC::GetSystemCycle() - st < 100) {}
    }
}

// AIC 等待 AIV
if (isVecSetSyncCom) {
    WaitForVector();
}

// AIC 执行 GEMM
blockMmad(...);

// AIC 通知 AIV
NotifyVector();

// AIV 等待 AIC
WaitForCube();

// AIV 执行 SwiGLU + 量化
epilogueOp(epilogueShape, epilogueOffset);

// AIV 通知 AIC
NotifyCube();
```

## TLA 关键概念

### Shape

描述张量的维度大小：

```cpp
// 1D Shape
auto shape1d = tla::MakeShape(1024);

// 2D Shape
auto shape2d = tla::MakeShape(256, 512);

// 嵌套 Shape (用于分形布局)
auto shapeFractal = tla::MakeShape(
    tla::MakeShape(Int<16>{}, Int<16>{}),  // [16, 16]
    tla::MakeShape(Int<16>{}, Int<32>{})   // [16, 32]
);
```

### Stride

描述张量在内存中的步长：

```cpp
// RowMajor: stride = (cols, 1)
auto strideRM = tla::MakeStride((int64_t)cols, Int<1>{});

// ColumnMajor: stride = (1, rows)
auto strideCM = tla::MakeStride(Int<1>{}, (int64_t)rows);
```

### Layout

组合 Shape 和 Stride：

```cpp
// RowMajor Layout
auto layoutRM = tla::MakeLayout(
    tla::MakeShape(rows, cols),           // shape
    tla::MakeStride((int64_t)cols, Int<1>{}),  // stride
    tla::MakeShape(rows, cols)            // originShape
);

// 坐标转偏移
auto offset = layoutRM(tla::MakeCoord(i, j));  // = i * cols + j
```

### GetTile

从张量中获取 tile：

```cpp
auto tensorBlockA = GetTile(
    tensorA,                                    // 源张量
    tla::MakeCoord(mLoc, kLoc),                // tile 起始坐标
    tla::MakeShape(actualBlockShape.m(), actualBlockShape.k())  // tile 形状
);
```

**实现原理**:
1. 计算新布局：使用 tile 形状作为 memory layout shape
2. 计算实际逻辑大小：处理边界情况 (tail tile)
3. 调整坐标：将源张量坐标偏移到 tile 起始位置

## 设计范式总结

### 1. 类型驱动设计

```cpp
// 所有类型在编译期确定
using BlockMmad = Gemm::Block::BlockMmadTla<DispatchPolicy, L1TileShape, L0TileShape, ...>;
using BlockEpilogue = BlockEpilogueSwigluQuant<L1TileShape, CType, MMadOutType, ...>;
using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
```

### 2. 层次化分解

```
问题 (M, N, K)
    ↓ BlockScheduler.GetBlockCoord()
Tile (m, n, k)
    ↓ GetTile()
Block (m_block, n_block, k_block)
    ↓ BlockMmad()
Fractal (16, 16, 16)
    ↓ 硬件指令
```

### 3. 资源管理

```cpp
// 使用 Resource 管理硬件资源
Arch::Resource<ArchTag> resource;

// 通过 Resource 访问缓存
resource.l1Buf   // L1 缓存
resource.l0ABuf  // L0A 缓存
resource.l0BBuf  // L0B 缓存
resource.l0CBuf  // L0C 缓存
resource.ubBuf   // UB 缓存
```

### 4. 同步机制

```cpp
// AIC/AIV 同步
NotifyVector()   → CrossCoreSetFlag<SYNC_AIC_AIV_MODE>(AIC_SYNC_AIV_FLAG)
WaitForCube()    → CrossCoreWaitFlag<SYNC_AIC_AIV_MODE>(AIC_SYNC_AIV_FLAG)

// 流水线同步
AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
```

### 5. 双缓冲

```cpp
// Ping-pong 缓冲
uint16_t GMM2_PINGPONG_IDX = 0;

// 使用
auto tensorUb = GMM2_PINGPONG_IDX == 0 ? tensorUbFirst : tensorUbSecond;
auto l0cOutUbGMM2 = GMM2_PINGPONG_IDX == 0 ? l0cOutUbGMM2First : l0cOutUbGMM2Second;

// 切换
GMM2_PINGPONG_IDX = 1 - GMM2_PINGPONG_IDX;
```

## 常用 API 速查

### 创建对象

```cpp
// 创建 Shape
auto shape = tla::MakeShape(m, n);

// 创建 Stride
auto stride = tla::MakeStride((int64_t)n, Int<1>{});

// 创建 Layout
auto layout = tla::MakeLayout<Element, LayoutTag>(rows, cols);

// 创建 Tensor
auto tensor = tla::MakeTensor(builtinTensor, layout, Arch::PositionGM{});

// 创建 Coord
auto coord = tla::MakeCoord(mIdx, nIdx);
```

### 访问元素

```cpp
// 访问 Shape 元素
auto m = tla::get<0>(shape);
auto n = tla::get<1>(shape);

// 访问 Layout
auto shape = layout.shape();
auto stride = layout.stride();

// 坐标转偏移
auto offset = layout(coord);

// 访问 Tensor
auto element = tensor(coord);
auto tile = GetTile(tensor, coord, tileShape);
```

### 类型操作

```cpp
// 获取类型
using Element = typename Tensor::Element;
using Layout = typename Tensor::Layout;
using ArchTag = typename BlockMmad::ArchTag;

// 类型判断
constexpr bool isRM = std::is_same_v<LayoutTag, Catlass::layout::RowMajor>;
constexpr bool isAIC = (g_coreType == AscendC::AIC);
```
