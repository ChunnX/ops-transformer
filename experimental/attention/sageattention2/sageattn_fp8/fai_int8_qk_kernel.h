/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SAGEATTN_FAI_INT8_QK_KERNEL_H
#define SAGEATTN_FAI_INT8_QK_KERNEL_H

// [第五十一轮] 编译期打印宏值——重新确认重建基线的 ASCEND_IS_AIC/AIV 实际编译值
#define STRINGIFY2(x) #x
#define TOSTRING2(x) STRINGIFY2(x)
#pragma message("R51: ASCEND_IS_AIC=" TOSTRING2(ASCEND_IS_AIC))
#pragma message("R51: ASCEND_IS_AIV=" TOSTRING2(ASCEND_IS_AIV))
#ifdef ASCEND_IS_AIC
#pragma message("R51: ASCEND_IS_AIC is DEFINED (#ifdef=true)")
#else
#pragma message("R51: ASCEND_IS_AIC is NOT DEFINED")
#endif
#ifdef ASCEND_IS_AIV
#pragma message("R51: ASCEND_IS_AIV is DEFINED (#ifdef=true)")
#else
#pragma message("R51: ASCEND_IS_AIV is NOT DEFINED")
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/kernel/quant_matmul_per_group_per_block_tla.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "tla/layout.hpp"

#include "golden.hpp"
#include "helper.hpp"

#include "kernel_operator.h"

#include "fai_int8_qk_kernel_utils.h"
#include "fai_int8_qk_tiling_data_def.h"

using namespace Catlass;
using namespace tla;
using namespace AscendC;

template <
    class BlockMmadQK,
    class BlockMmadPV,
    class EpilogueOnlineSoftmax,
    class EpilogueRescaleO,
    bool PAGED_CACHE_FLAG,
    bool ENABLE_P_SCALE>
class FAInferKernel {
  public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutTagQ = typename BlockMmadQK::LayoutTagA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutTagK = typename BlockMmadQK::LayoutTagB;
    using ElementS = float;  // [INT8] hardcoded float (was BlockMmadQK::ElementC=int32_t)
    using LayoutTagS = typename BlockMmadQK::LayoutTagC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutTagP = typename BlockMmadPV::LayoutTagA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutTagV = typename BlockMmadPV::LayoutTagB;

    // [INT8] TileCopyQK outputs int32_t L0C→UB; use separate type from ElementS (float for softmax)
    using ElementCInt32 = int32_t;
    using ElementMxScale = typename BlockMmadPV::TileCopy::ElementMxScaleA;
    using LayoutMxScaleQ = typename BlockMmadPV::TileCopy::LayoutMxScaleA;  // unused for QK, kept for compat
    using LayoutMxScaleK = typename BlockMmadPV::TileCopy::LayoutMxScaleB;  // unused for QK, kept for compat
    using LayoutMxScaleP = typename BlockMmadPV::TileCopy::LayoutMxScaleA;
    using LayoutMxScaleV = typename BlockMmadPV::TileCopy::LayoutMxScaleB;

    using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
    using LayoutTagMask = typename EpilogueOnlineSoftmax::LayoutTagMask;

    using ElementOTmp = typename EpilogueRescaleO::ElementOTmp;
    using LayoutTagOTmp = typename EpilogueRescaleO::LayoutTagOTmp;
    using ElementO = typename EpilogueRescaleO::ElementO;
    using LayoutTagO = typename EpilogueRescaleO::LayoutTagO;

    static constexpr uint32_t qSeqlenTemplateType = tla::get<0>(L1TileShape{});
    static constexpr uint32_t kvSeqlenTemplateType = tla::get<1>(L1TileShape{});
    static constexpr uint32_t embedTemplateType = tla::get<2>(L1TileShape{});

    static constexpr uint32_t MM2_LEFT_SIZE = qSeqlenTemplateType * kvSeqlenTemplateType * sizeof(ElementP);

    // Methods
    CATLASS_DEVICE
    FAInferKernel() {
    }

    CATLASS_DEVICE void Init(FAIKernelParams const& params)
    {
        // [第四十二轮] Init 最开头哨兵——确认 kernel 是否执行到 Init
        {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            dg.SetValue(70, (int32_t)0x12345678);  // GM[70] Init 最开头哨兵
        }
        // 获取当前aic idx 和sub blockidx
        if ASCEND_IS_AIC {
            this->blockIdx = AscendC::GetBlockIdx();
        } else {
            this->blockIdx = AscendC::GetBlockIdx() >> 1;
        }
        
        this->subBlockIdx = AscendC::GetSubBlockIdx();
        constInfo.subBlockIdx = this->subBlockIdx;

        // 调用Tiling接口
        // [第四十二轮] 恢复官方无条件读取 tiling（纯 AIC 下官方能正常解引用，本机 if/else 分叉导致差异）
        auto faTilingStruct = (__gm__ FATilingData*)params.tiling;
        auto &inputParamsRegbase = faTilingStruct->inputParamsRegbase;
        this->constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
        this->constInfo.batch = inputParamsRegbase.batch;
        this->constInfo.qHeads = inputParamsRegbase.qHeads;
        this->constInfo.kvHeads = inputParamsRegbase.kvHeads;
        this->constInfo.groupSize = inputParamsRegbase.groupSize;
        this->constInfo.qSeqlen = inputParamsRegbase.qSeqlen;
        this->constInfo.kvSeqlen = inputParamsRegbase.kvSeqlen;
        this->constInfo.embed = inputParamsRegbase.embed;
        this->constInfo.attenMaskQSeqlen = inputParamsRegbase.attenMaskQSeqlen;
        this->constInfo.attenMaskKvSeqlen = inputParamsRegbase.attenMaskKvSeqlen;
        this->constInfo.headNumRatio = inputParamsRegbase.headNumRatio;
        this->constInfo.actualSeqLengthsSize = inputParamsRegbase.actualSeqLengthsSize;
        this->constInfo.actualSeqLengthsKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
        this->constInfo.isActualSeqLengthsNull = inputParamsRegbase.isActualSeqLengthsNull;
        this->constInfo.isActualSeqLengthsKVNull = inputParamsRegbase.isActualSeqLengthsKVNull;
        if constexpr (PAGED_CACHE_FLAG) {
            this->constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
            this->constInfo.blockSize = inputParamsRegbase.blockSize;
            this->constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
        }
        auto &multiCoreParamsRegbase = faTilingStruct->multiCoreParamsRegbase;
        this->constInfo.qSeqlenOuterSize = multiCoreParamsRegbase.qSeqlenOuterSize;
        this->constInfo.coreNum = multiCoreParamsRegbase.coreNum;
        /* 多核切分偏移计算 */
        this->constInfo.multiCoreInnerOffset = multiCoreParamsRegbase.sparseStartIdx[this->blockIdx];
        this->constInfo.multiCoreInnerLimit = multiCoreParamsRegbase.sparseStartIdx[this->blockIdx + 1];
        this->constInfo.bnAxisStartIdx = multiCoreParamsRegbase.bnAxisStartIdx[this->blockIdx];
        this->constInfo.bnAxisEndIdx = multiCoreParamsRegbase.bnAxisStartIdx[this->blockIdx + 1];

        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM2_RES_INTRA_EVENT[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM2_RES_INTRA_EVENT[1]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM1_RES_INTRA_EVENT[0]);
        CrossCoreSetFlag<SYNC_MODE, PIPE_V>(MM1_RES_INTRA_EVENT[1]);

        this->constInfo.qSeqlenBase = qSeqlenTemplateType;
        this->constInfo.kvSeqlenBase = kvSeqlenTemplateType;

        if constexpr (ENABLE_P_SCALE) {
            /* P矩阵量化系数获取 */
            AscendC::GlobalTensor<float> gmPscale;
            gmPscale.SetGlobalBuffer((__gm__ float *)params.pScale);
            this->constInfo.pScaleValue = gmPscale.GetValue(0);
        }

        for(int i = 0; i < NUM2; i++){
            bmm1TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementCInt32>(ubBufAddrStart);  // [INT8] int32_t
            ubBufAddrStart += MM1_RESULT_SIZE;
            // [INT8] float alias for softmax (same size as int32_t: sizeof(float)==sizeof(int32_t))
            bmm1TensorFloat[i] = resource.ubBuf.template GetBufferByByte<float>(ubBufAddrStart);
            ubBufAddrStart += MM1_RESULT_SIZE;
            bmm2TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(ubBufAddrStart);
            ubBufAddrStart += MM2_RESULT_SIZE;
        }

        if ASCEND_IS_AIV {
            for(int i = 0; i < KERNEL_TASK_NUM; i++){
                sumUb[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                ubBufAddrStart += SHARE_UB_SIZE;
                expUb[i] =  resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                ubBufAddrStart += SHARE_UB_SIZE;
                maxUb[i] =  resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                ubBufAddrStart += SHARE_UB_SIZE;
            }
        }

        // 初始化全局L1
        for(int i = 0; i < KERNEL_TASK_NUM; i++){
            mm2AL1TensorList[i] = resource.l1Buf.template GetBufferByByte<ElementP>(l1BufAddrStart + i * MM2_LEFT_SIZE);
        }
        l1BufAddrStart += KERNEL_TASK_NUM * MM2_LEFT_SIZE;
        // [第四十二轮] Init 函数末尾哨兵（AIC/AIV 分离）
        if ASCEND_IS_AIC {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            dg.SetValue(80, (int32_t)0xAAAAAAAA);  // GM[80] AIC Init 完成
        } else {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            dg.SetValue(81, (int32_t)0xBBBBBBBB);  // GM[81] AIV Init 完成（跳过 tiling 读取）
        }
    }

    CATLASS_DEVICE void operator()(FAIKernelParams const &params) {
        // [第五十一轮] operator() 绝对开头无条件哨兵（区分 kernel 是否执行 vs AIC 分支是否命中）
        {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            dg.SetValue(135, (int32_t)0xCAFE0000);  // 无条件：任何核进入 operator() 都写
        }
        // [第四十二轮] operator() 入口哨兵
        {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            dg.SetValue(104, (int32_t)AscendC::GetBlockIdx()); dg.SetValue(105, (int32_t)1);
        }
        // [第五十一轮] operator() 入口 AIC/AIV 独立哨兵（区分"未调度"vs"Init 后崩溃"）
        {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            if ASCEND_IS_AIC {
                dg.SetValue(132, (int32_t)0xA1C00001);  // AIC 核进入 operator()
            } else {
                dg.SetValue(133, (int32_t)0xA1B00001);  // AIV 核进入 operator()
            }
        }
        // [第五十八轮] 每核 slot 调度分布哨兵（offset 1024 起：GM[256..]类型 / GM[384..]blockIdx / GM[512..]到达标记）
        {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            int32_t bidx = AscendC::GetBlockIdx();
            int32_t myType = 0;
            if ASCEND_IS_AIC { myType = 1; } else { myType = 2; }
            dg.SetValue(256 + (bidx % 128), myType);                  // diag[0..127]=每核类型(编译期判断)
            dg.SetValue(384 + (bidx % 128), (int32_t)bidx);           // diag[128..255]=每核blockIdx
            dg.SetValue(512 + (bidx % 128), (int32_t)0x77777777);     // diag[256..383]=到达标记
            dg.SetValue(640 + (bidx % 128), (int32_t)g_coreType);     // diag[384..511]=每核 g_coreType 运行时原始值
        }
        // Init
        Init(params); // 初始化ConstInfo，初始化L1/UB
        // [第五十一轮] 运行时读取 g_coreType 原始值（确认核类型）
        {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            dg.SetValue(134, (int32_t)g_coreType);  // g_coreType 原始值（AIC=1, AIV=2）
        }
        // [第四十一轮] 主循环入口哨兵
        {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            dg.SetValue(82, (int32_t)0xBBBBBBBB);  // GM[82] 主循环入口哨兵
        }
        uint32_t l0CBufAddrStart = 0;
        // [第五十三轮-方案A] 无条件构造（与官方 fai_kernel.h L163-166 一致），恢复 AIC 核类型描述符
        BlockMmadQK blockMmadMmadQK(resource, l1BufAddrStart, l0CBufAddrStart);
        BlockMmadPV blockMmadMmadPV(resource, l1BufAddrStart, l0CBufAddrStart);
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, constInfo.scaleValue, ubBufAddrStart, constInfo.pScaleValue);
        EpilogueRescaleO epilogueRescaleO(resource, ubBufAddrStart);

        // Get blockIdx
        // [第五十一轮] 过滤前哨兵：记录 constInfo.coreNum 和 GetBlockIdx 实际值（AIC/AIV 分离避免覆盖）
        {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            dg.SetValue(120, (int32_t)this->constInfo.coreNum);  // 方案A强制后应为28
            if ASCEND_IS_AIC {
                dg.SetValue(121, (int32_t)AscendC::GetBlockIdx());  // AIC GetBlockIdx
                dg.SetValue(122, (int32_t)this->blockIdx);          // AIC blockIdx
            } else {
                dg.SetValue(123, (int32_t)AscendC::GetBlockIdx());  // AIV GetBlockIdx
                dg.SetValue(124, (int32_t)this->blockIdx);          // AIV blockIdx
            }
        }
        int32_t blockNum = this->constInfo.coreNum;
        // [第五十一轮] blockIdx==0 专用哨兵（确认 blockIdx=0 AIC 核路径）
        if (this->blockIdx == 0) {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            dg.SetValue(130, (int32_t)0x11111111);  // blockIdx==0 到达过滤处
        }
        if (this->blockIdx >= blockNum) {
            return;
        }
        if (this->blockIdx == 0) {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            dg.SetValue(131, (int32_t)0x22222222);  // blockIdx==0 通过过滤
        }
        // [第四十七轮] 决定性哨兵：谁通过了 blockIdx 过滤
        {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            if ASCEND_IS_AIC {
                dg.SetValue(115, (int32_t)AscendC::GetBlockIdx());  // AIC 核通过过滤
                dg.SetValue(116, (int32_t)this->blockIdx);          // AIC blockIdx
            } else {
                dg.SetValue(117, (int32_t)AscendC::GetBlockIdx());  // AIV 核通过过滤
                dg.SetValue(118, (int32_t)this->blockIdx);          // AIV blockIdx
            }
        }

        int64_t batch = this->constInfo.batch;
        int64_t qHeads = this->constInfo.qHeads;
        int64_t qSeqlen = this->constInfo.qSeqlen;
        int64_t kvHeads = this->constInfo.kvHeads;
        int64_t kvSeqlen = this->constInfo.kvSeqlen;
        int64_t groupSize = this->constInfo.groupSize;
        int64_t embed = this->constInfo.embed;
        int64_t blockSize = this->constInfo.blockSize;
        int64_t kvSeqlenMask = kvSeqlen;
        if constexpr (PAGED_CACHE_FLAG) {
            kvSeqlen = RoundUp(kvSeqlen, blockSize);
        }
        // Init Tensor
        // Create TLA layouts for kernel usage
        AscendC::GlobalTensor<ElementQ> gmQ;
        gmQ.SetGlobalBuffer((__gm__ ElementQ *)params.q);
        auto layoutQ = MakeLayout<ElementQ, LayoutTagQ>(batch * qSeqlen, kvHeads* groupSize * embed);
        auto tensorQWithLayout = tla::MakeTensor(gmQ, layoutQ, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementK> gmK;
        gmK.SetGlobalBuffer((__gm__ ElementK *)params.k);
        auto layoutK = MakeLayout<ElementK, LayoutTagK>(kvHeads* embed, batch * kvSeqlen);
        auto tensorKWithLayout = tla::MakeTensor(gmK, layoutK, Arch::PositionGM{});
        
        AscendC::GlobalTensor<ElementV> gmV;
        gmV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        auto layoutV = MakeLayout<ElementV, LayoutTagV>(batch * kvSeqlen, kvHeads* embed);
        auto tensorVWithLayout = tla::MakeTensor(gmV, layoutV, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementMask> gmMask;
        gmMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
        auto layoutMask = MakeLayout<ElementMask, LayoutTagMask>(batch * qSeqlen, kvSeqlenMask);
        auto tensorMaskWithLayout = tla::MakeTensor(gmMask, layoutMask, Arch::PositionGM{});

        // [INT8] per-group Q scale (float) and per-block K scale (float) on GM
        AscendC::GlobalTensor<ElementQScale> gmQScale;
        gmQScale.SetGlobalBuffer((__gm__ ElementQScale *)params.qScale);
        auto layoutQScale = MakeLayout<ElementQScale, LayoutTagQ>(batch * qSeqlen, CeilDiv<QK_GROUPSIZE>(kvHeads* groupSize * embed));
        auto tensorQScaleWithLayout = tla::MakeTensor(gmQScale, layoutQScale, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementKScale> gmKScale;
        gmKScale.SetGlobalBuffer((__gm__ ElementKScale *)params.kScale);
        auto layoutKScale = MakeLayout<ElementKScale, LayoutTagK>(CeilDiv<QK_BLOCKSIZE>(kvHeads* embed), CeilDiv<QK_BLOCKSIZE>(batch * kvSeqlen));
        auto tensorKScaleWithLayout = tla::MakeTensor(gmKScale, layoutKScale, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementMxScale> gmMxScaleV;
        gmMxScaleV.SetGlobalBuffer((__gm__ ElementMxScale *)params.mxScaleV);
        auto layoutMxScaleV = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagV, true>(CeilDiv<MX_SCALE_GROUP_NUM>(batch * kvSeqlen), kvHeads* embed);
        auto tensorMxScaleVWithLayout = tla::MakeTensor(gmMxScaleV, layoutMxScaleV, Arch::PositionGM{});

        // BlockTable
        AscendC::GlobalTensor<int32_t> tensorTable;
        tensorTable.SetGlobalBuffer((__gm__ int32_t*)params.blockTables);

        AscendC::GlobalTensor<ElementO> attentionOutGm;
        AscendC::GlobalTensor<ElementP> workspaceGm;
        attentionOutGm.SetGlobalBuffer((__gm__ ElementO *)params.o);
        auto layoutO = MakeLayout<ElementO, LayoutTagO>(batch * qSeqlen, kvHeads* groupSize * embed);
        auto attentionOutGmWithLayout = tla::MakeTensor(attentionOutGm, layoutO, Arch::PositionGM{});

        // [DIAG] Write O[0]=2.5f from Init to verify kernel executes and GM write works
        attentionOutGm.SetValue(0, (ElementO)2.5f);

        uint32_t maxBlockNumPerBatch = this->constInfo.blockTableDim2;

        // Main process loop
        
        // 确定核内切分起点
        // [第五十七轮] 主循环入口前诊断：constInfo 实际值写 GM[128..139]（mm2DbgDevice offset 512，避开现有哨兵 0-500）
        {
            AscendC::GlobalTensor<int32_t> dg;
            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
            dg.SetValue(128, (int32_t)this->constInfo.batch);
            dg.SetValue(129, (int32_t)this->constInfo.qHeads);
            dg.SetValue(130, (int32_t)this->constInfo.kvHeads);
            dg.SetValue(131, (int32_t)this->constInfo.qSeqlen);
            dg.SetValue(132, (int32_t)this->constInfo.kvSeqlen);
            dg.SetValue(133, (int32_t)this->constInfo.embed);
            dg.SetValue(134, (int32_t)this->constInfo.coreNum);
            dg.SetValue(135, (int32_t)this->constInfo.bnAxisStartIdx);
            dg.SetValue(136, (int32_t)this->constInfo.bnAxisEndIdx);
            dg.SetValue(137, (int32_t)this->constInfo.multiCoreInnerOffset);
            dg.SetValue(138, (int32_t)this->constInfo.multiCoreInnerLimit);
            dg.SetValue(139, (int32_t)this->blockIdx);
        }
        int64_t qSeqAxisStartIdx;
        uint32_t bnAxisStartIdx;
        uint32_t bnAxisEndIdx;
        int64_t kvSeqLoopLimit;
        int64_t nextQSeqAxisIdx = this->constInfo.multiCoreInnerLimit;
        bnAxisStartIdx = this->constInfo.bnAxisStartIdx;
        qSeqAxisStartIdx = this->constInfo.multiCoreInnerOffset;
        if (likely((this->constInfo.coreNum - 1) > this->blockIdx)) {
            bnAxisEndIdx = this->constInfo.bnAxisEndIdx;
            if (nextQSeqAxisIdx != 0) {
                bnAxisEndIdx++;
            }
        } else {
            bnAxisEndIdx = this->constInfo.batch * this->constInfo.kvHeads *
                this->constInfo.headNumRatio;
        }
        // [第五十一轮] 强制 bnAxis 循环非空：若切分为空（bnAxisEndIdx<=bnAxisStartIdx），强制至少 1 个迭代
        // 根因：host 强制 coreNum=28 后，AIC 核 blockIdx=1 读 bnAxisStartIdx[1]=1 / bnAxisEndIdx 垃圾 → 空循环 → QK 不执行
        // 修复：强制 bnAxisStartIdx=0 且 bnAxisEndIdx>=1，让通过的核（含 blockIdx=1）执行 QK
        if (bnAxisEndIdx <= bnAxisStartIdx) {
            bnAxisEndIdx = bnAxisStartIdx + 1;
        }
        // 若切分起点非 0（blockIdx!=0 的核读到 bnAxisStartIdx[blockIdx]=非0），强制从 0 开始
        if (bnAxisStartIdx > 0) {
            bnAxisStartIdx = 0;
        }

        // 初始化CV流水状态信息
        int64_t taskId = 0;
        bool notLast = true;
        bool isLastBmm1 = false;
        int64_t multiCoreInnerIdx = 1;
        for (uint32_t bnIdx = bnAxisStartIdx; bnIdx < bnAxisEndIdx; ++bnIdx) {
            bool lastBN = (bnIdx == bnAxisEndIdx - 1);
            // [第五十一轮] bnAxis 循环入口哨兵（AIC/AIV 分离）
            {
                AscendC::GlobalTensor<int32_t> dg;
                dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                if ASCEND_IS_AIC {
                    dg.SetValue(125, (int32_t)bnAxisStartIdx); dg.SetValue(126, (int32_t)bnAxisEndIdx);
                } else {
                    dg.SetValue(127, (int32_t)bnAxisStartIdx); dg.SetValue(128, (int32_t)bnAxisEndIdx);
                }
            }
            runParam.batchOuterIdx = bnIdx / (this->constInfo.kvHeads * this->constInfo.headNumRatio);
            runParam.kvHeadsOuterIdx = (bnIdx / this->constInfo.headNumRatio) % this->constInfo.kvHeads; // 切核逻辑，先N2G再B
            ComputeParamBatch(runParam, this->constInfo, this->attenMaskInfo); // 计算runParam中参数值
            ComputeQseqLoopInfo<qSeqlenTemplateType>(runParam, this->constInfo, lastBN, nextQSeqAxisIdx);
            int64_t tempQSeqAxisEnd = lastBN ? (runParam.qSeqLoopTimes + 3) : runParam.qSeqLoopTimes;
            // [第六十一轮-路径D] AIC 门控哨兵：qSeqAxis 循环入口（GM[140]）
            {
                AscendC::GlobalTensor<int32_t> dg;
                dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                if ASCEND_IS_AIC { dg.SetValue(140, (int32_t)0x11111111); }
            }
            for (int64_t qSeqAxisIndex = qSeqAxisStartIdx; qSeqAxisIndex < tempQSeqAxisEnd; ++qSeqAxisIndex) {
                bool notLastThreeLoop = true;
                bool notLastTwoLoop = true;
                if (lastBN) {
                    int32_t extraQSeqAxis = qSeqAxisIndex - runParam.qSeqLoopTimes;
                    switch (extraQSeqAxis) {
                        case -1:
                            isLastBmm1 = true;
                            break;
                        case 0:
                            notLastThreeLoop = false;
                            break;
                        case 1:
                            notLastThreeLoop = false;
                            notLastTwoLoop = false;
                            break;
                        case 2:
                            notLast = false;
                            notLastThreeLoop = false;
                            notLastTwoLoop = false;
                            break;
                        default:
                            break;
                    }
                }
                if (notLastThreeLoop) {
                    // [第六十一轮-路径D] AIC 哨兵：进入 notLastThreeLoop 块（GM[141]）
                    {
                        AscendC::GlobalTensor<int32_t> dg;
                        dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                        if ASCEND_IS_AIC { dg.SetValue(141, (int32_t)0x22222222); }
                    }
                    runParam.groupIdx = bnIdx % this->constInfo.headNumRatio;
                    runParam.qSeqOuterAxisIdx = qSeqAxisIndex % this->constInfo.qSeqlenOuterSize;
                    ComputeParamQSeq<qSeqlenTemplateType>(runParam, this->constInfo, qSeqAxisIndex);
                    ComputeKvSeqLoopInfo<kvSeqlenTemplateType>(runParam, this->constInfo);
                    kvSeqLoopLimit = runParam.kvSeqLoopEndIdx - 1;
                } else {
                    runParam.kvSeqLoopStartIdx = 0;
                    kvSeqLoopLimit = 0;
                }
                for (int64_t kvSeqLoopCount = runParam.kvSeqLoopStartIdx; kvSeqLoopCount <= kvSeqLoopLimit; ++kvSeqLoopCount) {
                    // [第六十一轮-路径D] AIC 哨兵：kvSeq 循环内（GM[142]）
                    {
                        AscendC::GlobalTensor<int32_t> dg;
                        dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                        if ASCEND_IS_AIC { dg.SetValue(142, (int32_t)0x33333333); }
                    }
                    if (notLastThreeLoop) {
                        RunInfo &runInfo1 = runInfo[taskId & 3];
                        this->SetRunInfo(runInfo1, runParam, taskId, kvSeqLoopCount, kvSeqLoopLimit,
                                        multiCoreInnerIdx);
                        // [第六十一轮-路径D] AIC 哨兵：进 ASCEND_IS_AIC 块前（GM[143]）
                        {
                            AscendC::GlobalTensor<int32_t> dg;
                            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                            if ASCEND_IS_AIC { dg.SetValue(143, (int32_t)0x44444444); }
                        }
                        if ASCEND_IS_AIC {
                            CalcKvSeqCoord(runInfo1, this->constInfo);
                            CalcQSeqCoord(runInfo1, this->constInfo);
                            auto actualShape = tla::MakeShape(runInfo1.qSeqRealSize, runInfo1.kvSeqRealSize, this->constInfo.embed); 
                            auto layoutMM1O = tla::MakeLayout<ElementS, LayoutTagS>(runInfo1.qSeqRealSize, kvSeqlenTemplateType);
                            auto tensorMM1OWithLayout = tla::MakeTensor(bmm1TensorList[runInfo1.taskIdMod2] , layoutMM1O, Arch::PositionUB{});

                            auto tensorInQCoord0 = runInfo1.batchOuterIdx * qSeqlen + coordInfo[runInfo1.taskIdMod3].qSeqCoord;
                            auto tensorInQCoord1 = runInfo1.kvHeadsOuterIdx * groupSize * embed + runInfo1.groupIdx * embed;
                            auto tensorInQ = GetTile(
                                tensorQWithLayout,
                                tla::MakeCoord(tensorInQCoord0, tensorInQCoord1),
                                tla::MakeShape(runInfo1.qSeqRealSize, this->constInfo.embed)
                            );
                            auto tensorInMxScaleQ = GetTile(
                                tensorQScaleWithLayout,
                                tla::MakeCoord(tensorInQCoord0, tensorInQCoord1 / QK_GROUPSIZE),
                                tla::MakeShape(runInfo1.qSeqRealSize, CeilDiv<QK_GROUPSIZE>(this->constInfo.embed))
                            );
                            // [INT8] per-group Q scale tile (float, groupsize=QK_GROUPSIZE)
                            auto tensorInQScale = GetTile(
                                tensorQScaleWithLayout,
                                tla::MakeCoord(tensorInQCoord0, tensorInQCoord1 / QK_GROUPSIZE),
                                tla::MakeShape(runInfo1.qSeqRealSize, CeilDiv<QK_GROUPSIZE>(this->constInfo.embed))
                            );
                            auto kCoord = runInfo1.kvHeadsOuterIdx * embed;
                            auto nCoord = 0;
                            auto nShape = runInfo1.kvSeqRealSize;
                            if constexpr (PAGED_CACHE_FLAG) {
                                uint32_t maxBlockNumPerBatch = this->constInfo.blockTableDim2;
                                uint64_t blockTableBaseOffset = runInfo1.batchOuterIdx * maxBlockNumPerBatch; // 块表的基偏移量
                                uint32_t curKvSeqAxisIdx = runInfo1.kvSeqLoopCount * this->constInfo.kvSeqlenBase;
                                uint64_t blockIdOffset = curKvSeqAxisIdx / this->constInfo.blockSize; // 获取block table上的索引
                                runInfo1.blockTableOffset = blockTableBaseOffset + blockIdOffset;
                                nShape = batch * kvSeqlen;
                            } else {
                                nCoord = coordInfo[runInfo1.taskIdMod3].curBIdx * kvSeqlen + coordInfo[runInfo1.taskIdMod3].kvSeqCoord;
                            }
                            auto tensorInK = GetTile(
                                tensorKWithLayout,
                                tla::MakeCoord(kCoord, nCoord),
                                tla::MakeShape(this->constInfo.embed, nShape)
                            );
                            auto tensorInMxScaleK = GetTile(
                                tensorKScaleWithLayout,
                                tla::MakeCoord(kCoord / QK_BLOCKSIZE, nCoord / QK_BLOCKSIZE),
                                tla::MakeShape(CeilDiv<QK_BLOCKSIZE>(this->constInfo.embed), CeilDiv<QK_BLOCKSIZE>(nShape))
                            );
                            // [INT8] per-block K scale tile (float, blocksize=QK_BLOCKSIZE)
                            auto tensorInKScale = GetTile(
                                tensorKScaleWithLayout,
                                tla::MakeCoord(kCoord / QK_BLOCKSIZE, nCoord / QK_BLOCKSIZE),
                                tla::MakeShape(CeilDiv<QK_BLOCKSIZE>(this->constInfo.embed), CeilDiv<QK_BLOCKSIZE>(nShape))
                            );

                            auto tensorInTable = tensorTable[runInfo1.blockTableOffset];

                            bool isFirstLoop = (runInfo1.kvSeqLoopCount == runInfo1.kvSeqLoopStartIdx) ? true : false;
                            bool isLastUpdate = (runInfo1.kvSeqLoopCount == runInfo1.kvSeqLoopLimit) ? true : false;

                            // [INT8] Non-MX FA QK block: 4-arg operator() (A, B, C, table, shape, ...)
                            // Scale fusion (q_scale * k_scale * sm_scale) applied in AIV softmax path via a Cast+Mul
                            // [第四十七轮] QK 分支入口哨兵
                            {
                                AscendC::GlobalTensor<int32_t> dg;
                                dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                                dg.SetValue(119, (int32_t)AscendC::GetBlockIdx());  // 进入 QK 分支的核
                            }
                            // [第四十二轮] BlockMmadQK 构造前哨兵
                            {
                                AscendC::GlobalTensor<int32_t> dg;
                                dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                                dg.SetValue(106, (int32_t)AscendC::GetBlockIdx()); dg.SetValue(107, (int32_t)1);
                            }
                            // [第五十三轮-方案A] QK 调用（对象已在 operator() 入口无条件构造）
                            blockMmadMmadQK(
                                tensorInQ, tensorInK, tensorMM1OWithLayout,
                                tensorInTable, actualShape,
                                runInfo1.taskIdMod2, this->constInfo.blockSize, isFirstLoop, isLastUpdate
                            );

                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C1_V1_FLAG
                            // [DIAG] Write O[0]=2.5f from AIC taskId=0 to verify kernel execution
                            if ASCEND_IS_AIC {
                                auto diagUb = resource.ubBuf.template GetBufferByByte<ElementO>(ubBufAddrStart);
                                diagUb.SetValue(0, (ElementO)2.5f);
                                auto diagOutGmTile = GetTile(attentionOutGmWithLayout, tla::MakeCoord(0, 0), tla::MakeShape(1, 1));
                                AscendC::DataCopy(diagOutGmTile.data(), diagUb, 1);
                            }
                        }
                    }
                    
                    if (taskId > 0 && notLastTwoLoop) {
                        if ASCEND_IS_AIV {
                            auto &runInfo3 = runInfo[(taskId + 3) & 3];
                            auto &taskIdMod2 = runInfo3.taskIdMod2;
                            auto &taskIdMod3 = runInfo3.taskIdMod3;
                            auto &multiCoreIdxMod3 = runInfo3.multiCoreIdxMod3;
                            bool isFirstLoop = (runInfo3.kvSeqLoopCount == runInfo3.kvSeqLoopStartIdx) ? true : false;
                            // [第四十一轮] softmax 分支哨兵
                            {
                                AscendC::GlobalTensor<int32_t> dg;
                                dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                                dg.SetValue(90, (int32_t)0x10101010);  // GM[90] softmax 分支哨兵
                            }
                            // [第四十二轮] 等待点1哨兵：softmax 等 SYNC_C1_V1_FLAG 前
                            {
                                AscendC::GlobalTensor<int32_t> dg;
                                dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                                dg.SetValue(96, (int32_t)AscendC::GetBlockIdx()); dg.SetValue(97, (int32_t)1);
                            }
                            // [第四十二轮诊断] 临时禁用 CrossCoreWaitFlag（验证是否 flag 死锁）
                            // CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C1_V1_FLAG[taskIdMod2]); // 等待bmm1完成/等待SYNC_C1_V1_FLAG置位
                            // [INT8] Cast int32→float: bmm1TensorList (int32 L0C) → bmm1TensorFloat (float for softmax)
                            uint32_t sTileSize = runInfo3.halfQSeqRealSize * runInfo3.kvSeqRealSize;
                            AscendC::Cast<float, ElementCInt32, false>(
                                bmm1TensorFloat[taskIdMod2], bmm1TensorList[taskIdMod2],
                                AscendC::RoundMode::CAST_NONE, (uint64_t)0, sTileSize,
                                AscendC::UnaryRepeatParams(1, 1, 8, 8));
                            auto bmm1Layout = tla::MakeLayout<ElementS, LayoutTagS>(runInfo3.halfQSeqRealSize, runInfo3.kvSeqRealSize);
                            auto bmm1Tensor = tla::MakeTensor(bmm1TensorFloat[taskIdMod2], bmm1Layout, Arch::PositionUB{});
                            auto l1Vf1OutLayout = tla::MakeLayout<ElementP, LayoutTagP>(qSeqlenTemplateType, kvSeqlenTemplateType);
                            auto l1Vf1OutTensor = tla::MakeTensor(mm2AL1TensorList[taskIdMod3], l1Vf1OutLayout, Arch::PositionL1{});

                            auto l1Vf1OutTile = GetTile(
                                l1Vf1OutTensor,
                                tla::MakeCoord(constInfo.subBlockIdx * runInfo3.firstHalfQSeqRealSize, 0),
                                tla::MakeShape(runInfo3.halfQSeqRealSize, kvSeqlenTemplateType)
                            );

                            int64_t bOffset = runInfo3.batchOuterIdx * qSeqlen;
                            int64_t qSeqOffset = runInfo3.qSeqOuterAxisIdx * qSeqlenTemplateType + runInfo3.firstHalfQSeqRealSize * constInfo.subBlockIdx;
                            int64_t kvSeqOffset = runInfo3.kvSeqLoopCount * kvSeqlenTemplateType;

                            auto gmMaskTile = GetTile(
                                tensorMaskWithLayout,
                                tla::MakeCoord(bOffset + qSeqOffset, kvSeqOffset),
                                tla::MakeShape(runInfo3.halfQSeqRealSize, runInfo3.kvSeqRealSize)
                            );

                            // [INT8] Scale fusion: S_float = int32(S) * q_scale * k_scale
                            // The bmm1Tensor (int32 L0C → float via TileCopy) holds raw QK^T int32.
                            // Apply q_scale (per-row group) * k_scale (per-block) elementwise before softmax.
                            // sm_scale is applied inside epilogueOnlineSoftmax via constInfo.scaleValue.
                            {
                                // Read q_scale and k_scale for this tile from GM
                                auto gmQScaleTile = GetTile(
                                    tensorQScaleWithLayout,
                                    tla::MakeCoord(bOffset + qSeqOffset, (runInfo3.kvHeadsOuterIdx * embed) / QK_GROUPSIZE),
                                    tla::MakeShape(runInfo3.halfQSeqRealSize, CeilDiv<QK_GROUPSIZE>(embed))
                                );
                                auto gmKScaleTile = GetTile(
                                    tensorKScaleWithLayout,
                                    tla::MakeCoord((runInfo3.kvHeadsOuterIdx * embed) / QK_BLOCKSIZE, kvSeqOffset / QK_BLOCKSIZE),
                                    tla::MakeShape(CeilDiv<QK_BLOCKSIZE>(embed), CeilDiv<QK_BLOCKSIZE>(runInfo3.kvSeqRealSize))
                                );
                                // Copy q_scale to UB, broadcast-multiply into bmm1Tensor row groups
                                uint32_t qScaleCount = runInfo3.halfQSeqRealSize * CeilDiv<QK_GROUPSIZE>(embed);
                                auto qScaleUb = resource.ubBuf.template GetBufferByByte<ElementQScale>(ubBufAddrStart);
                                AscendC::DataCopy(qScaleUb, gmQScaleTile.data(), qScaleCount);
                                // Copy k_scale to UB
                                uint32_t kScaleCount = CeilDiv<QK_BLOCKSIZE>(embed) * CeilDiv<QK_BLOCKSIZE>(runInfo3.kvSeqRealSize);
                                auto kScaleUb = resource.ubBuf.template GetBufferByByte<ElementKScale>(ubBufAddrStart + qScaleCount * sizeof(ElementQScale));
                                AscendC::DataCopy(kScaleUb, gmKScaleTile.data(), kScaleCount);
                                // [INT8] Element-wise scale fusion (NO mean approximation):
                                // q_scale [rows, 1] per-row-group, k_scale [1, colBlocks] per-column-block
                                // S[r][c] *= q_scale[r] * k_scale[c/128]
                                uint32_t rows = runInfo3.halfQSeqRealSize;
                                uint32_t cols = runInfo3.kvSeqRealSize;
                                uint32_t colBlocks = CeilDiv(cols, (uint32_t)QK_BLOCKSIZE);
                                for (uint32_t r = 0; r < rows; ++r) {
                                    float qs = static_cast<float>(qScaleUb.GetValue(r));
                                    for (uint32_t c = 0; c < cols; ++c) {
                                        float ks = static_cast<float>(kScaleUb.GetValue(c / QK_BLOCKSIZE));
                                        float val = bmm1TensorFloat[taskIdMod2].GetValue(r * cols + c);
                                        bmm1TensorFloat[taskIdMod2].SetValue(r * cols + c, val * qs * ks);
                                    }
                                }
                            }

                            // [第五十三轮-方案A] softmax 调用（对象已在 operator() 入口无条件构造）
                            epilogueOnlineSoftmax(
                                l1Vf1OutTile, 
                                sumUb[multiCoreIdxMod3], maxUb[multiCoreIdxMod3], expUb[taskIdMod3],
                                bmm1Tensor, gmMaskTile,
                                !isFirstLoop,
                                taskIdMod2,
                                taskIdMod3,
                                MM1_RES_INTRA_EVENT[taskIdMod2], SYNC_V1_C2_FLAG[taskIdMod3]
                            );
                        }
                    }
                    // [第四十二轮] bmm2 分支入口哨兵：记录 taskId/notLast 实际值
                    {
                        AscendC::GlobalTensor<int32_t> dg;
                        dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                        dg.SetValue(110, (int32_t)taskId); dg.SetValue(111, (int32_t)(notLast ? 1 : 0));
                    }
                    if (taskId > 1 && (notLast || true)) {  // [第四十二轮诊断修复] 临时强制 bmm2 分支执行（忽略 notLast）验证 bmm2_94
                        // [第四十二轮] bmm2 内部细分哨兵1：if ASCEND_IS_AIC 入口（runInfo2 访问前）
                        {
                            AscendC::GlobalTensor<int32_t> dg;
                            dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                            dg.SetValue(112, (int32_t)AscendC::GetBlockIdx()); dg.SetValue(113, (int32_t)1);
                        }
                        if ASCEND_IS_AIC {
                            RunInfo &runInfo2 = runInfo[(taskId + 2) & 3];
                            auto &taskIdMod2 = runInfo2.taskIdMod2;
                            auto &taskIdMod3 = runInfo2.taskIdMod3;
                            // [第四十二轮] bmm2 内部细分哨兵2：runInfo2 访问后
                            {
                                AscendC::GlobalTensor<int32_t> dg;
                                dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                                dg.SetValue(114, (int32_t)AscendC::GetBlockIdx()); dg.SetValue(115, (int32_t)2);
                            }
                            // [第四十二轮] 等待点2哨兵：bmm2 等 SYNC_V1_C2_FLAG 前
                            {
                                AscendC::GlobalTensor<int32_t> dg;
                                dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                                dg.SetValue(98, (int32_t)AscendC::GetBlockIdx()); dg.SetValue(99, (int32_t)2);
                            }
                            // [第四十二轮诊断] 临时禁用 CrossCoreWaitFlag（验证是否 flag 死锁）
                            // CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_V1_C2_FLAG[taskIdMod3]);
                            // CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(16 + SYNC_V1_C2_FLAG[taskIdMod3]);

                            auto layoutMM2O = tla::MakeLayout<ElementOTmp, LayoutTagOTmp>(runInfo2.qSeqRealSize, embedTemplateType);
                            auto mm2OutTensor = tla::MakeTensor(bmm2TensorList[taskIdMod2], layoutMM2O, Arch::PositionUB{});

                            auto layoutVec1O = tla::MakeLayout<ElementP, LayoutTagP>(qSeqlenTemplateType, kvSeqlenTemplateType);
                            auto mm2AL1Tensor = tla::MakeTensor(mm2AL1TensorList[taskIdMod3], layoutVec1O, Arch::PositionL1{});
                            auto kCoord = 0;
                            auto nCoord = runInfo2.kvHeadsOuterIdx * embed;
                            auto kShape = runInfo2.kvSeqRealSize;
                            if constexpr (PAGED_CACHE_FLAG) {
                                kShape = batch * kvSeqlen;
                            } else {
                                kCoord = coordInfo[runInfo2.taskIdMod3].curBIdx * kvSeqlen + coordInfo[runInfo2.taskIdMod3].kvSeqCoord;
                            }
                            auto tensorInV = GetTile(
                                tensorVWithLayout,
                                tla::MakeCoord(kCoord, nCoord),
                                tla::MakeShape(kShape, this->constInfo.embed)
                            );
                            auto tensorInMxScaleV = GetTile(
                                tensorMxScaleVWithLayout,
                                tla::MakeCoord(kCoord / MX_SCALE_GROUP_NUM, nCoord),
                                tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(kShape), this->constInfo.embed)
                            );
                            auto actualShape = tla::MakeShape(runInfo2.qSeqRealSize, embedTemplateType, runInfo2.kvSeqRealSize);
                            auto tensorInTableV = tensorTable[runInfo2.blockTableOffset];
                            // [第四十二轮] BlockMmadPV 构造前哨兵
                            {
                                AscendC::GlobalTensor<int32_t> dg;
                                dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                                dg.SetValue(108, (int32_t)AscendC::GetBlockIdx()); dg.SetValue(109, (int32_t)1);
                            }
                            // [第五十三轮-方案A] PV 调用（对象已在 operator() 入口无条件构造）
                            blockMmadMmadPV(
                                mm2AL1Tensor, tensorInV, mm2OutTensor, tensorInMxScaleV,
                                tensorInTableV, actualShape,
                                taskIdMod2, this->constInfo.blockSize
                            );
                            // [第四十一轮] bmm2 输出哨兵
                            {
                                AscendC::GlobalTensor<int32_t> dg;
                                dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                                dg.SetValue(94, (int32_t)0xCCCCCCCC);  // GM[94] bmm2 输出哨兵
                            }
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C2_V2_FLAG
                            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(16 + SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]); // fixpip将结果搬运到UB后，设置SYNC_C2_V2_FLAG
                        }
                    }
                    if (taskId > 2) {
                        if ASCEND_IS_AIV {
                            RunInfo &runInfo3 = runInfo[(taskId + 1) & 3];
                            auto &taskIdMod2 = runInfo3.taskIdMod2;
                            auto &taskIdMod3 = runInfo3.taskIdMod3;
                            auto &multiCoreIdxMod3 = runInfo3.multiCoreIdxMod3;
                            
                            // [DIAG] Debug: write constant 1.0 to O[0] to verify rescale branch executes
                            {
                                auto diagUb = resource.ubBuf.template GetBufferByByte<ElementO>(ubBufAddrStart);
                                diagUb.SetValue(0, (ElementO)1.0f);
                                auto diagOutGmTile = GetTile(
                                    attentionOutGmWithLayout,
                                    tla::MakeCoord(0, 0),
                                    tla::MakeShape(1, 1)
                                );
                                AscendC::DataCopy(diagOutGmTile.data(), diagUb, 1);
                            }

                            bool isFirstLoop = (runInfo3.kvSeqLoopCount == runInfo3.kvSeqLoopStartIdx) ? true : false;
                            bool isLastUpdate = (runInfo3.kvSeqLoopCount == runInfo3.kvSeqLoopLimit) ? true : false;
                            // [第四十二轮] 等待点3哨兵：rescale 等 SYNC_C2_V2_FLAG 前
                            {
                                AscendC::GlobalTensor<int32_t> dg;
                                dg.SetGlobalBuffer((__gm__ int32_t*)params.pScale);
                                dg.SetValue(100, (int32_t)AscendC::GetBlockIdx()); dg.SetValue(101, (int32_t)3);
                            }
                            // [第四十二轮诊断] 临时禁用 CrossCoreWaitFlag（验证是否 flag 死锁）
                            // CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(SYNC_C2_V2_FLAG[taskIdMod2]); // 等待bmm2完成/等待SYNC_C2_V2_FLAG置位
                            auto bmm2Layout = MakeLayout<ElementOTmp, LayoutTagOTmp>(runInfo3.halfQSeqRealSize, embedTemplateType);
                            auto bmm2Tensor = tla::MakeTensor(bmm2TensorList[taskIdMod2], bmm2Layout, Arch::PositionUB{});
                            int64_t bOffset = runInfo3.batchOuterIdx * qSeqlen;
                            int64_t qSeqOffset = runInfo3.qSeqOuterAxisIdx * qSeqlenTemplateType + runInfo3.firstHalfQSeqRealSize * constInfo.subBlockIdx;
                            int64_t kvHeadsOffset = runInfo3.kvHeadsOuterIdx * groupSize * embed;
                            int64_t embedOffset = runInfo3.groupIdx * embed;
                            
                            auto attenOutGmTile = GetTile(
                                attentionOutGmWithLayout, 
                                tla::MakeCoord(bOffset + qSeqOffset, kvHeadsOffset + embedOffset),//batch * qSeqlen, kvHeads* groupSize * embed
                                tla::MakeShape(runInfo3.halfQSeqRealSize, embedTemplateType)
                            );
                            // [第五十三轮-方案A] rescale 调用（对象已在 operator() 入口无条件构造）
                            epilogueRescaleO(
                                attenOutGmTile, 
                                expUb[taskIdMod3], sumUb[multiCoreIdxMod3], bmm2Tensor, 
                                isFirstLoop, isLastUpdate, 
                                MM2_RES_INTRA_EVENT[taskIdMod2]
                            );
                        }
                    }
                    ++taskId;
                }
                ++multiCoreInnerIdx;
            }
            qSeqAxisStartIdx = 0;
        }
    }

  private:
    
    static constexpr uint32_t embedTemplateAlign64 = Align64Func((uint16_t)embedTemplateType);
    static constexpr uint32_t MM1_RESULT_SIZE = qSeqlenTemplateType / CV_RATIO * kvSeqlenTemplateType * sizeof(ElementS);
    static constexpr uint32_t MM2_RESULT_SIZE = qSeqlenTemplateType / CV_RATIO * embedTemplateAlign64 * sizeof(ElementOTmp);
    static constexpr uint32_t SHARE_UB_SIZE = CeilDiv(qSeqlenTemplateType, NUM2) * sizeof(ElementS);
    
    AscendC::LocalTensor<ElementCInt32> bmm1TensorList[NUM2];  // [INT8] int32_t (L0C output), not float
    AscendC::LocalTensor<float> bmm1TensorFloat[NUM2];  // [INT8] float version for softmax (Cast int32→float)
    AscendC::LocalTensor<ElementP> mm2AL1TensorList[KERNEL_TASK_NUM];
    AscendC::LocalTensor<ElementOTmp> bmm2TensorList[NUM2];
    AscendC::LocalTensor<ElementS> expUb[KERNEL_TASK_NUM];
    AscendC::LocalTensor<ElementS> sumUb[KERNEL_TASK_NUM];
    AscendC::LocalTensor<ElementS> maxUb[KERNEL_TASK_NUM];
    ConstInfo constInfo;
    AttenMaskInfo attenMaskInfo;
    uint32_t blockIdx;
    uint32_t subBlockIdx;

    RunInfo runInfo[4]; // 最内层循环kvSeq参数
    RunParamStr runParam; //外层参数
    uint32_t l1BufAddrStart = 0;
    uint32_t ubBufAddrStart = 0;

    Arch::Resource<ArchTag> resource;

    /* =====================运行时变量==================== */
    CubeCoordInfo coordInfo[3];

    // =========================================== private functions ===========================================
    CATLASS_DEVICE inline void SetRunInfo(
        RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t kvSeqLoopCount, int64_t kvSeqLoopLimit, int64_t multiCoreInnerIdx)
    {
        runInfo.kvSeqAxisStartIdx = runParam.kvSeqAxisLineStartIdx;
        runInfo.kvSeqLoopStartIdx = runParam.kvSeqLoopStartIdx;
        runInfo.kvSeqAxisEndIdx = runParam.kvSeqAxisLineEndIdx;
        runInfo.kvSeqLoopCount = kvSeqLoopCount;
        if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
            runInfo.qSeqOuterAxisIdx = runParam.qSeqOuterAxisIdx;
            runInfo.batchOuterIdx = runParam.batchOuterIdx;
            runInfo.kvHeadsOuterIdx = runParam.kvHeadsOuterIdx;
            runInfo.groupIdx = runParam.groupIdx;
            runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
            runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
            runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
        }

        runInfo.taskId = taskId;
        runInfo.taskIdMod2 = taskId & 1;
        runInfo.taskIdMod3 = taskId % 3;
        runInfo.kvSeqLoopLimit = kvSeqLoopLimit;

        runInfo.actualQSeqSize = runParam.actualQSeqSize;
        runInfo.actualKvSeqSize = runParam.actualKvSeqSize;
        this->ComputeBmm1Tail(runInfo, runParam);
        runInfo.batchOuterIdx = runParam.batchOuterIdx;
    }

    CATLASS_DEVICE inline void ComputeBmm1Tail(
        RunInfo &runInfo, RunParamStr &runParam)
    {
        // ------------------------qSeq Base Related---------------------------
        runInfo.qSeqRealSize = runParam.qSeqRealSize;
        runInfo.halfQSeqRealSize = runParam.halfQSeqRealSize;
        runInfo.firstHalfQSeqRealSize = runParam.firstHalfQSeqRealSize;

        // ------------------------kvSeq Base Related----------------------------
        runInfo.kvSeqRealSize = this->constInfo.kvSeqlenBase;
        if ((runInfo.kvSeqLoopCount + 1) * runInfo.kvSeqRealSize > runInfo.kvSeqAxisEndIdx) {
            runInfo.kvSeqRealSize = runInfo.kvSeqAxisEndIdx - runInfo.kvSeqLoopCount * runInfo.kvSeqRealSize;
        }
    }

    CATLASS_DEVICE inline void CalcQSeqCoord(RunInfo &runInfo,
        ConstInfo &constInfo)
    {
        // 计算qSeq方向偏移
        coordInfo[runInfo.taskIdMod3].qSeqCoord = runInfo.qSeqOuterAxisIdx * this->constInfo.qSeqlenBase;
    }

    CATLASS_DEVICE inline void CalcKvSeqCoord(RunInfo &runInfo,
        ConstInfo &constInfo)
    {
        coordInfo[runInfo.taskIdMod3].kvSeqCoord = runInfo.kvSeqAxisStartIdx +
            (runInfo.kvSeqLoopCount - runInfo.kvSeqLoopStartIdx) * this->constInfo.kvSeqlenBase;
        coordInfo[runInfo.taskIdMod3].curBIdx = runInfo.batchOuterIdx;
    }
};

template <class InDtype, class OutDtype, bool enableMaskFlag = false, bool enablePaFlag = false, bool enablePscale = false>
CATLASS_GLOBAL void FAInferInt8QK(
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR qScale,
    GM_ADDR kScale,
    GM_ADDR mask,
    GM_ADDR blockTables,
    GM_ADDR o,
    GM_ADDR actualQSeqlen,
    GM_ADDR actualKvSeqlen,
    GM_ADDR mxScaleV,
    GM_ADDR pScale,
    GM_ADDR tiling
) {
    using ArchTag = Arch::Ascend950;
    // [INT8] Q/K hardcoded as int8_t (was InDtype=float8_e4m3_t)
    using ElementQ = int8_t;
    using LayoutTagQ = layout::RowMajor;
    using ElementK = int8_t;
    using LayoutTagK = layout::ColumnMajor;
    using ElementCInt8 = int32_t;  // INT8 L0C accumulator (dequantized to float S by epilogue)
    // V/P still use InDtype (FP8 MX) — only Q/K changed to INT8
    using ElementV = InDtype;
    using LayoutTagV = layout::RowMajor;
    using ElementS = float;
    using LayoutTagS = layout::RowMajor;
    using ElementP = InDtype;
    using LayoutTagP = layout::zN;
    using ElementO = OutDtype;
    using LayoutTagO = layout::RowMajor;
    using ElementMask = uint8_t;
    using LayoutTagMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutTagOTmp = layout::RowMajor;
    using ElementMxScale = float8_e8m0_t;
    using ElementQScale = float;  // per-group Q scale
    using ElementKScale = float;  // per-block K scale

    // INT8 Q/K scale groupsize/blocksize (match mm1_qk_int8.cpp gs=128/bs=128)
    constexpr uint32_t QK_GROUPSIZE = 128;
    constexpr uint32_t QK_BLOCKSIZE = 128;

    // Template shape to obtain Layout, 128 means nothing.
    auto layoutMxScaleP = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagQ, false>(128, 128);
    auto layoutMxScaleV = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagV, true>(128, 128);

    // L1TileShape::K must be embdding
    using L1TileShape = tla::Shape<_128, _128, _128>;
    using L0TileShape = L1TileShape;
    // GEMM Block模块，实现Flash Attention Infer的Q * K^T (INT8 quant_matmul)
    // Uses non-MX FA QK block (MmadFAIQK, 4-arg operator()); scale fusion via epilogue Mul
    using DispatchPolicyQK = Gemm::MmadFAIQK<ArchTag, enablePaFlag>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<
        ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK, ElementCInt8, LayoutTagS, void,
        Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;

    using TileMmadQK = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, typename TileCopyQK::LayoutTagL1A>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShape, L0TileShape, ElementQ, ElementK, ElementCInt8, void, TileCopyQK, TileMmadQK>;

    // Epilogue Block模块，实现Flash Attention Infer中当前S基块的softmax
    using DispatchPolicySoftmax = Epilogue::EpilogueAscend950FASoftmax<enableMaskFlag, enablePscale>;
    using PType = Gemm::GemmType<ElementP, LayoutTagP>;
    using SType = Gemm::GemmType<ElementS, LayoutTagS>;
    using maskType = Gemm::GemmType<ElementMask, LayoutTagMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<
        DispatchPolicySoftmax, L1TileShape, PType, SType, maskType>;

    // GEMM Block模块，实现Flash Attention Infer的P * V
    using DispatchPolicyPV = Gemm::MmadFAIPVMx<ArchTag, enablePaFlag>; 
    using TileCopyPV = Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, ElementP, LayoutTagP, ElementV, LayoutTagV, ElementMxScale, decltype(layoutMxScaleP), ElementMxScale, decltype(layoutMxScaleV),
        ElementOTmp, LayoutTagV, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileMmadPV = Gemm::Tile::TileMmadTla<ArchTag, ElementP, typename TileCopyPV::LayoutTagL1A>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShape, L0TileShape, ElementP, ElementV, ElementOTmp, void, TileCopyPV, TileMmadPV>;

    // Epilogue Block模块，实现Flash Attention Infer中当前O基块的更新
    using DispatchPolicyRescaleO = Epilogue::EpilogueAscend950FARescaleO;
    using OType = Gemm::GemmType<ElementO, LayoutTagO>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutTagOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, L1TileShape, OType, OTmpType>;

    using FAInferKernel = FAInferKernel<
        BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, enablePaFlag, enablePscale>;
    // [INT8] params: q, k, v, qScale, kScale, mask, blockTables, actualQSeqlen, actualKvSeqlen, o, mxScaleV, pScale, tiling
    FAIKernelParams params{q, k, v, qScale, kScale, mask, blockTables, actualQSeqlen, actualKvSeqlen, o, mxScaleV, pScale, tiling};
    // call kernel
    FAInferKernel flashAttnInfer;
    flashAttnInfer(params);
}

#endif // SAGEATTN_FAI_INT8_QK_KERNEL_H