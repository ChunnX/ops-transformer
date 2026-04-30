/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_reduce_scatter_aiv_mode_block_epilogue_dequant.h
 * \brief BlockEpilogue的特化实现：在AIV核心上执行反量化（Dequant）的Epilogue操作
 *
 * 本文件是Catlass模板线性代数库中BlockEpilogue类的全特化版本，专门用于
 * MatmulReduceScatterV2算子的反量化场景。支持三种反量化模式：
 *   1. Per-Channel + Per-Token: C_int32 × perChannelScale × perTokenScale → D
 *   2. Per-Channel only: C_int32 × perChannelScale → D
 *   3. Per-Token only: D_half/bf16 × perTokenScale → D
 *
 * UB内存布局（双缓冲，UB_STAGES=2）：
 *   [C_0][Scale_0][PerTokenScale_0][D_0] | [C_1][Scale_1][PerTokenScale_1][D_1] | [CFp32][Mul][PerTokenScaleBrcb]
 *
 * 执行流水线（以Per-Channel+Per-Token为例）：
 *   1. GM→UB: 拷贝C、scale、perTokenScale
 *   2. Cast: C_int32 → C_fp32
 *   3. 广播乘法: C_fp32 × scale（行广播）→ mul
 *   4. 广播乘法: mul × perTokenScale（列广播）→ perTokenMul
 *   5. Cast: perTokenMul_fp32 → D_half/bf16
 *   6. UB→GM: 拷贝D到输出
 */
#ifndef MATMUL_REDUCE_SCATTER_AIV_MODE_BLOCK_EPILOGUE_DEQUANT_H
#define MATMUL_REDUCE_SCATTER_AIV_MODE_BLOCK_EPILOGUE_DEQUANT_H

#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/catlass.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/resource.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/dispatch_policy.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm_coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/gemm_type.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/matrix_coord.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/layout.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/detail/callback.hpp"
#include "../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/copy_gm_to_ub.hpp"

namespace Catlass::Epilogue::Block {

// BlockEpilogue全特化：用于MatmulReduceScatterV2的反量化Epilogue
// 模板参数：
//   ArchTag_: 硬件架构标签（AtlasA2）
//   CType_: int32累加结果类型
//   ScaleType_: per-channel scale类型（float向量）
//   PerTokenScaleType_: per-token scale类型（float向量）
//   DType_: 输出数据类型（half或bfloat16_t）
//   TileRowBroadcastMul_: 行广播乘法Tile操作
//   TileBroadcastOneBlk_: 单块广播Tile操作
//   TileOneBlkColumnBroadcastMul_: 列广播乘法Tile操作
//   TileCopy_: GM↔UB数据拷贝策略
//   EpilogueTileSwizzle_: Epilogue分块调度策略
template <
    class ArchTag_,
    class CType_,
    class ScaleType_,
    class PerTokenScaleType_,
    class DType_,
    class TileRowBroadcastMul_,
    class TileBroadcastOneBlk_,
    class TileOneBlkColumnBroadcastMul_,
    class TileCopy_,
    class EpilogueTileSwizzle_
>
class BlockEpilogue <
    ArchTag_,
    CType_,
    ScaleType_,
    PerTokenScaleType_,
    DType_,
    TileRowBroadcastMul_,
    TileBroadcastOneBlk_,
    TileOneBlkColumnBroadcastMul_,
    TileCopy_,
    EpilogueTileSwizzle_
> {
public:
    static constexpr uint32_t UB_STAGES = 2;  // UB双缓冲数量，实现计算与搬运重叠

    // Data infos - 数据类型定义
    using ArchTag = ArchTag_;
    using ElementC = typename CType_::Element;              // int32_t
    using LayoutC = typename CType_::Layout;                // RowMajor
    using ElementScale = typename ScaleType_::Element;       // float
    using LayoutScale = typename ScaleType_::Layout;         // VectorLayout
    using ElementPerTokenScale = typename PerTokenScaleType_::Element;  // float
    using LayoutPerTokenScale = typename PerTokenScaleType_::Layout;    // VectorLayout
    using ElementD = typename DType_::Element;               // half或bfloat16_t
    using LayoutD = typename DType_::Layout;                 // RowMajor

    // Check data infos - 编译期类型检查：确保C为int32、D为half/bf16、scale为float
    static_assert(
        std::is_same_v<ElementC, int32_t> && (std::is_same_v<ElementD, half> || std::is_same_v<ElementD, bfloat16_t>) &&
            std::is_same_v<ElementScale, float> && std::is_same_v<ElementPerTokenScale, float>,
        "The element type template parameters of BlockEpilogue are wrong"
    );
    // Check layout infos - 编译期布局检查：确保C/D为RowMajor、scale/perTokenScale为VectorLayout
    static_assert(
        std::is_same_v<LayoutC, layout::RowMajor> && std::is_same_v<LayoutScale, layout::VectorLayout> &&
            std::is_same_v<LayoutPerTokenScale, layout::VectorLayout> && std::is_same_v<LayoutD, layout::RowMajor>,
        "The layout template parameters of BlockEpilogue are wrong"
    );

    // Tile compute ops - 分块计算操作类型
    using TileRowBroadcastMul = TileRowBroadcastMul_;         // 行广播乘法：C_fp32 × scale[j]
    using TileBroadcastOneBlk = TileBroadcastOneBlk_;         // 单块广播：perTokenScale扩展到UB块
    using TileOneBlkColumnBroadcastMul = TileOneBlkColumnBroadcastMul_;  // 列广播乘法：result × perTokenScale[i]

    // Tile copy - GM↔UB数据拷贝操作类型
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;                    // GM→UB: 拷贝C（int32）
    using CopyGmToUbScale = typename TileCopy_::CopyGmToUbX;                // GM→UB: 拷贝scale（float）
    using CopyGmToUbPerTokenScale = typename TileCopy_::CopyGmToUbY;        // GM→UB: 拷贝perTokenScale（float）
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;                    // UB→GM: 拷贝D（half/bf16）
    using CopyGmToUbD = Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<ElementD, layout::RowMajor>>;  // GM→UB: 拷贝D（per-token场景的输入）

    using EpilogueTileSwizzle = EpilogueTileSwizzle_;  // Epilogue分块调度策略

    using TileShape = typename TileRowBroadcastMul::TileShape;  // 分块形状（64×128）

    static_assert(
        TileShape::ROW == TileBroadcastOneBlk::COMPUTE_LENGTH &&
        std::is_same_v<TileShape, typename TileOneBlkColumnBroadcastMul::TileShape>,
        "TileShape must be consistent for all tile compute ops"
    );

    // 编译期UB容量检查：确保双缓冲+临时缓冲区不超过UB大小
    static_assert(
        (UB_STAGES * (TileShape::COUNT * sizeof(ElementC) + TileShape::COLUMN * sizeof(ElementScale)
                + TileShape::ROW * sizeof(ElementPerTokenScale) + TileShape::COUNT * sizeof(ElementD))
            + (TileShape::COUNT + TileShape::COUNT) * sizeof(float) + TileShape::ROW * BYTE_PER_BLK)
        <= ArchTag::UB_SIZE,
        "TileShape is too large to fit in UB"
    );

    // 构造函数：分配UB缓冲区并初始化event flag
    // UB布局：[C_0][Scale_0][PerTokenScale_0][D_0] | [C_1][Scale_1][PerTokenScale_1][D_1] | [CFp32][Mul][PerTokenScaleBrcb]
    CATLASS_DEVICE
    BlockEpilogue(Arch::Resource<ArchTag> const &resource)
    {
        size_t ubOffset = 0;
        int32_t eventVMTE2 = 0;   // V→MTE2事件计数器
        int32_t eventMTE2V = 0;   // MTE2→V事件计数器
        int32_t eventMTE3V = 0;   // MTE3→V事件计数器
        int32_t eventVMTE3 = 0;   // V→MTE3事件计数器
        // 为每个双缓冲stage分配UB空间
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementC);
            ubScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementScale>(ubOffset);
            ubOffset += TileShape::COLUMN * sizeof(ElementScale);
            ubPerTokenScaleList[i] = resource.ubBuf.template GetBufferByByte<ElementPerTokenScale>(ubOffset);
            ubOffset += TileShape::ROW * sizeof(ElementPerTokenScale);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += TileShape::COUNT * sizeof(ElementD);

            eventUbCVMTE2List[i] = eventVMTE2++;
            eventUbCMTE2VList[i] = eventMTE2V++;
            eventUbScaleMTE2VList[i] = eventMTE2V++;
            eventUbPerTokenScaleMTE2VList[i] = eventMTE2V++;
            eventUbDMTE3VList[i] = eventMTE3V++;
            eventUbDVMTE3List[i] = eventVMTE3++;
        }
        // 共享临时缓冲区：用于Cast和广播乘法的中间结果
        ubCFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(float);
        ubMul = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::COUNT * sizeof(float);
        ubPerTokenScaleBrcb = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += TileShape::ROW * BYTE_PER_BLK;
        ubPerTokenMul = ubMul;  // Per-Token乘法结果复用ubMul缓冲区
    }

    // 等待所有双缓冲stage的event flag，确保所有异步操作完成
    CATLASS_DEVICE
    void WaitFlag()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    // 初始化所有双缓冲stage的event flag，允许第一次DMA操作
    CATLASS_DEVICE
    void InitFlag()
    {
        for (uint32_t i = 0; i < UB_STAGES; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[i]);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[i]);
        }
    }

    // operator()重载1：Per-Channel + Per-Token组合反量化
    // 计算：C_int32 × perChannelScale × perTokenScale → D
    // 流水线：GM→UB(C,scale,perTokenScale) → Cast(int32→fp32) → 行广播乘 → 列广播乘 → Cast(fp32→D) → UB→GM
    CATLASS_DEVICE
    void operator() (__gm__ ElementScale *ptrScale, LayoutScale layoutScale,
                     __gm__ ElementPerTokenScale *ptrPerTokenScale, LayoutPerTokenScale layoutPerTokenScale,
                     __gm__ ElementC *ptrC, LayoutC layoutC, __gm__ ElementD *ptrD, LayoutD layoutD,
                     GemmCoord problemShape)
    {
        // Calculate the offset of the current block
        MatrixCoord actualBlockShape = problemShape.GetCoordMN();

        gmScale.SetGlobalBuffer(ptrScale);
        gmPerTokenScale.SetGlobalBuffer(ptrPerTokenScale);
        gmC.SetGlobalBuffer(ptrC);
        gmD.SetGlobalBuffer(ptrD);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        InitFlag();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffset = tileCoord * tileShape;

            auto gmTileC = gmC[layoutC.GetOffset(tileOffset)];
            auto layoutGmTileC = layoutC.GetTileLayout(actualTileShape);

            auto &ubC = ubCList[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride};

            // 把 C 从GM拷贝到UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            auto scaleTileOffset = tileOffset.template GetCoordByAxis<1>();
            auto scaleTileShape = actualTileShape.template GetCoordByAxis<1>();

            auto gmTileScale = gmScale[layoutScale.GetOffset(scaleTileOffset)];
            auto layoutGmTileScale = layoutScale.GetTileLayout(scaleTileShape);

            auto &ubScale = ubScaleList[ubListId];
            auto layoutUbScale = LayoutScale::template MakeLayoutInUb<ElementScale>(scaleTileShape);

            // 把 scale 从GM拷贝到UB
            copyGmToUbScale(ubScale, gmTileScale, layoutUbScale, layoutGmTileScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);

            auto perTokenScaleTileOffset = tileOffset.template GetCoordByAxis<0>();
            auto perTokenScaleTileShape = actualTileShape.template GetCoordByAxis<0>();

            auto gmTilePerTokenScale = gmPerTokenScale[layoutPerTokenScale.GetOffset(perTokenScaleTileOffset)];
            auto layoutGmTilePerTokenScale = layoutPerTokenScale.GetTileLayout(perTokenScaleTileShape);

            auto &ubPerTokenScale = ubPerTokenScaleList[ubListId];
            auto layoutUbPerTokenScale = LayoutScale::template MakeLayoutInUb<ElementPerTokenScale>(
                perTokenScaleTileShape);

            // 把 perTokenScale 从GM拷贝到UB
            copyGmToUbPerTokenScale(ubPerTokenScale, gmTilePerTokenScale, layoutUbPerTokenScale,
                layoutGmTilePerTokenScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);

            // 在UB上把C cast到FP32
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);

            // 在UB上做广播乘法
            AscendC::PipeBarrier<PIPE_V>();
            tileRowBroadcastMul(ubMul, ubCFp32, ubScale);
            tileBroadcastOneBlk(ubPerTokenScaleBrcb, ubPerTokenScale);
            AscendC::PipeBarrier<PIPE_V>();
            tileOneBlkColumnBroadcastMul(ubPerTokenMul, ubMul, ubPerTokenScaleBrcb);
            AscendC::PipeBarrier<PIPE_V>();

            auto &ubD = ubDList[ubListId];
            LayoutD layoutUbD{actualTileShape, ubTileStride};

            // 将乘法结果从UB cast到D
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubD, ubPerTokenMul, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            auto gmTileD = gmD[layoutD.GetOffset(tileOffset)];
            auto layoutGmTileD = layoutD.GetTileLayout(actualTileShape);

            // 把乘法结果从UB拷贝到GM
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1) % UB_STAGES;
        }
        WaitFlag();
    }

    // operator()重载2：仅Per-Channel反量化
    // 计算：C_int32 × perChannelScale → D
    // 流水线：GM→UB(C,scale) → Cast(int32→fp32) → 行广播乘 → Cast(fp32→D) → UB→GM
    CATLASS_DEVICE
    void operator() (__gm__ ElementScale *ptrScale, LayoutScale layoutScale,
                     __gm__ ElementC *ptrC, LayoutC layoutC, __gm__ ElementD *ptrD, LayoutD layoutD,
                     GemmCoord problemShape)
    {
        // Calculate the offset of the current block
        MatrixCoord actualBlockShape = problemShape.GetCoordMN();

        gmScale.SetGlobalBuffer(ptrScale);
        gmC.SetGlobalBuffer(ptrC);
        gmD.SetGlobalBuffer(ptrD);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        InitFlag();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffset = tileCoord * tileShape;

            auto gmTileC = gmC[layoutC.GetOffset(tileOffset)];
            auto layoutGmTileC = layoutC.GetTileLayout(actualTileShape);

            auto &ubC = ubCList[ubListId];
            LayoutC layoutUbC{actualTileShape, ubTileStride};

            // 把 C 从GM拷贝到UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbC(ubC, gmTileC, layoutUbC, layoutGmTileC);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            auto scaleTileOffset = tileOffset.template GetCoordByAxis<1>();
            auto scaleTileShape = actualTileShape.template GetCoordByAxis<1>();

            auto gmTileScale = gmScale[layoutScale.GetOffset(scaleTileOffset)];
            auto layoutGmTileScale = layoutScale.GetTileLayout(scaleTileShape);

            auto &ubScale = ubScaleList[ubListId];
            auto layoutUbScale = LayoutScale::template MakeLayoutInUb<ElementScale>(scaleTileShape);

            // 把 scale 从GM拷贝到UB
            copyGmToUbScale(ubScale, gmTileScale, layoutUbScale, layoutGmTileScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);

            // 在UB上把C cast到FP32
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubC, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbScaleMTE2VList[ubListId]);

            // 在UB上做广播乘法
            AscendC::PipeBarrier<PIPE_V>();
            tileRowBroadcastMul(ubMul, ubCFp32, ubScale);
            AscendC::PipeBarrier<PIPE_V>();

            auto &ubD = ubDList[ubListId];
            LayoutD layoutUbD{actualTileShape, ubTileStride};

            // 将乘法结果从UB cast到D
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubD, ubMul, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            auto gmTileD = gmD[layoutD.GetOffset(tileOffset)];
            auto layoutGmTileD = layoutD.GetTileLayout(actualTileShape);

            // 把乘法结果从UB拷贝到GM
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmTileD, ubD, layoutGmTileD, layoutUbD);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1) % UB_STAGES;
        }
        WaitFlag();
    }

    // operator()重载3：仅Per-Token反量化
    // 计算：D_half/bf16 × perTokenScale → D
    // 流水线：GM→UB(D,perTokenScale) → Cast(half/bf16→fp32) → 列广播乘 → Cast(fp32→D) → UB→GM
    // 注意：此场景的输入D已经是half/bf16类型（由AIC fixpipe输出），不是int32
    CATLASS_DEVICE
    void operator() (__gm__ ElementPerTokenScale *ptrPerTokenScale, LayoutPerTokenScale layoutPerTokenScale,
                     __gm__ ElementD *ptrIn, LayoutD layoutIn, __gm__ ElementD *ptrOut, LayoutD layoutOut,
                     GemmCoord problemShape)
    {
        // Calculate the offset of the current block
        MatrixCoord actualBlockShape = problemShape.GetCoordMN();

        AscendC::GlobalTensor<ElementD> gmIn;
        AscendC::GlobalTensor<ElementD> gmOut;
        gmPerTokenScale.SetGlobalBuffer(ptrPerTokenScale);
        gmIn.SetGlobalBuffer(ptrIn);
        gmOut.SetGlobalBuffer(ptrOut);

        auto ubTileStride = MakeCoord(static_cast<int64_t>(TileShape::COLUMN), 1L);
        auto tileShape = TileShape::ToCoord();
        EpilogueTileSwizzle epilogueTileSwizzle(actualBlockShape, tileShape);
        uint32_t tileLoops = epilogueTileSwizzle.GetLoops();
        uint32_t subblockIdx = AscendC::GetSubBlockIdx();
        uint32_t subblockNum = AscendC::GetSubBlockNum();

        InitFlag();
        for (uint32_t loopIdx = subblockIdx; loopIdx < tileLoops; loopIdx += subblockNum) {
            auto tileCoord = epilogueTileSwizzle.GetTileCoord(loopIdx);
            auto actualTileShape = epilogueTileSwizzle.GetActualTileShape(tileCoord);
            auto tileOffset = tileCoord * tileShape;

            auto gmTileIn = gmIn[layoutIn.GetOffset(tileOffset)];
            auto gmTileOut = gmOut[layoutOut.GetOffset(tileOffset)];
            auto layoutGmTileIn = layoutIn.GetTileLayout(actualTileShape);
            auto layoutGmTileOut = layoutOut.GetTileLayout(actualTileShape);

            auto &ubIn = ubDList[ubListId];
            LayoutD layoutUbIn{actualTileShape, ubTileStride};

            // 把 D 从GM拷贝到UB
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);
            copyGmToUbD(ubIn, gmTileIn, layoutUbIn, layoutGmTileIn);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);

            auto perTokenScaleTileOffset = tileOffset.template GetCoordByAxis<0>();
            auto perTokenScaleTileShape = actualTileShape.template GetCoordByAxis<0>();

            auto gmTilePerTokenScale = gmPerTokenScale[layoutPerTokenScale.GetOffset(perTokenScaleTileOffset)];
            auto layoutGmTilePerTokenScale = layoutPerTokenScale.GetTileLayout(perTokenScaleTileShape);

            auto &ubPerTokenScale = ubPerTokenScaleList[ubListId];
            auto layoutUbPerTokenScale = LayoutScale::template MakeLayoutInUb<ElementPerTokenScale>(
                perTokenScaleTileShape);

            // 把 perTokenScale 从GM拷贝到UB
            copyGmToUbPerTokenScale(ubPerTokenScale, gmTilePerTokenScale, layoutUbPerTokenScale,
                layoutGmTilePerTokenScale);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);

            // 在UB上把D cast到FP32
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbCMTE2VList[ubListId]);
            AscendC::Cast(ubCFp32, ubIn, AscendC::RoundMode::CAST_NONE, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(eventUbCVMTE2List[ubListId]);

            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventUbPerTokenScaleMTE2VList[ubListId]);

            // 在UB上做广播乘法
            AscendC::PipeBarrier<PIPE_V>();
            tileBroadcastOneBlk(ubPerTokenScaleBrcb, ubPerTokenScale);
            AscendC::PipeBarrier<PIPE_V>();
            tileOneBlkColumnBroadcastMul(ubPerTokenMul, ubCFp32, ubPerTokenScaleBrcb);
            AscendC::PipeBarrier<PIPE_V>();

            // 将乘法结果从UB cast到D
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);
            AscendC::Cast(ubIn, ubPerTokenMul, AscendC::RoundMode::CAST_RINT, TileShape::COUNT);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);

            // 把乘法结果从UB拷贝到GM
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventUbDVMTE3List[ubListId]);
            copyUbToGmD(gmTileOut, ubIn, layoutGmTileOut, layoutUbIn);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventUbDMTE3VList[ubListId]);

            ubListId = (ubListId + 1) % UB_STAGES;
        }
        WaitFlag();
    }

private:
    // 双缓冲UB缓冲区：每个stage存储C/Scale/PerTokenScale/D的局部张量
    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];                    // int32累加结果C的双缓冲UB
    AscendC::LocalTensor<ElementScale> ubScaleList[UB_STAGES];            // per-channel scale的双缓冲UB
    AscendC::LocalTensor<ElementPerTokenScale> ubPerTokenScaleList[UB_STAGES];  // per-token scale的双缓冲UB
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];                    // 输出D的双缓冲UB

    // 双缓冲event flag：用于MTE2/MTE3/V流水线之间的同步
    int32_t eventUbCVMTE2List[UB_STAGES];      // V→MTE2事件：C的UB缓冲区可用
    int32_t eventUbCMTE2VList[UB_STAGES];      // MTE2→V事件：C的GM→UB拷贝完成
    int32_t eventUbScaleMTE2VList[UB_STAGES];  // MTE2→V事件：scale的GM→UB拷贝完成
    int32_t eventUbPerTokenScaleMTE2VList[UB_STAGES];  // MTE2→V事件：perTokenScale的GM→UB拷贝完成
    int32_t eventUbDMTE3VList[UB_STAGES];      // MTE3→V事件：D的UB→GM拷贝完成
    int32_t eventUbDVMTE3List[UB_STAGES];      // V→MTE3事件：D的Cast完成，可以写回GM

    uint32_t ubListId{0};  // 当前使用的双缓冲索引（0或1交替）

    // 共享临时缓冲区：用于Cast和广播乘法的中间结果
    AscendC::LocalTensor<float> ubCFp32;                // C cast到float32的临时缓冲区
    AscendC::LocalTensor<float> ubMul;                  // 行广播乘法结果的临时缓冲区
    AscendC::LocalTensor<float> ubPerTokenScaleBrcb;    // perTokenScale广播扩展后的临时缓冲区
    AscendC::LocalTensor<float> ubPerTokenMul;          // 列广播乘法结果的临时缓冲区（复用ubMul）

    // 分块计算操作实例
    TileRowBroadcastMul tileRowBroadcastMul;            // 行广播乘法：C_fp32 × scale[j]
    TileBroadcastOneBlk tileBroadcastOneBlk;            // 单块广播：perTokenScale扩展到UB块
    TileOneBlkColumnBroadcastMul tileOneBlkColumnBroadcastMul;  // 列广播乘法：result × perTokenScale[i]

    // GM↔UB数据拷贝操作实例
    CopyGmToUbC copyGmToUbC;                  // GM→UB: 拷贝C（int32）
    CopyGmToUbScale copyGmToUbScale;          // GM→UB: 拷贝scale（float）
    CopyGmToUbPerTokenScale copyGmToUbPerTokenScale;  // GM→UB: 拷贝perTokenScale（float）
    CopyUbToGmD copyUbToGmD;                  // UB→GM: 拷贝D（half/bf16）
    CopyGmToUbD copyGmToUbD;                  // GM→UB: 拷贝D（per-token场景的输入）

    // GM全局张量：用于设置各数据类型的全局缓冲区
    AscendC::GlobalTensor<ElementScale> gmScale;              // per-channel scale的GM张量
    AscendC::GlobalTensor<ElementPerTokenScale> gmPerTokenScale;  // per-token scale的GM张量
    AscendC::GlobalTensor<ElementC> gmC;                      // int32累加结果C的GM张量
    AscendC::GlobalTensor<ElementD> gmD;                      // 输出D的GM张量
};
}  // namespace Catlass::Epilogue::Block

#endif  // MATMUL_REDUCE_SCATTER_AIV_MODE_BLOCK_EPILOGUE_DEQUANT_H
