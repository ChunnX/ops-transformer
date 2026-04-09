/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_GEMM_KERNEL_HSTU_TLA_HPP
#define CATLASS_GEMM_KERNEL_HSTU_TLA_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/layout/layout.hpp"

#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/debug.hpp"

#include "tla/tensor.hpp"
#include "tla/layout.hpp"

namespace Catlass::Gemm::Kernel {

enum class LAYOUT {
    NTD = 0,
    TND = 1,
};

struct HSTUConstInfo {
    uint32_t numHeads = 0;
    uint32_t embeddingSize = 0;
    uint32_t kvHeads = 0;
    uint32_t batch = 0;
    uint32_t tileM = 0;
    uint32_t tileN = 0;
    uint32_t tileK = 0;
    uint32_t totalTaskNum = 0;
    float siluScale = 0.0;
    uint32_t layout = 0;

    uint64_t T1;
    uint64_t T2;
    
    uint64_t qkMatmulOutSize;
    uint64_t siluScaleOutSize;
    uint64_t pvMatmulOutSize;
};

struct HSTURunInfo {
    uint32_t loop;
    uint32_t actualQBlkSize;
    uint32_t actualKvseqlen;
    uint64_t gmOffsetQ;
    uint64_t gmOffsetK;
    uint64_t qNTOffset;
    uint64_t kNTOffset;
    uint64_t qTNOffset;
    uint64_t kTNOffset;
    uint64_t singleCoreOffset;
};

// Template for HSTU kernel.
template <
    class BlockMmadQK,
    class BlockMmadPV,
    class EpilogueSiluScale
>
class HSTUTla {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename BlockMmadQK::ElementC;

    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementO = typename BlockMmadPV::ElementC;
    using ElementQBlkOffset = uint32_t;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::RowMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;

    static constexpr uint32_t WS_DOUBLE = 2;
    static constexpr uint32_t MAX_SEQ_LEN = 20480;
    static constexpr uint32_t QK_READY_ID = 1;
    static constexpr uint32_t SILUSCALE_READY_ID = 2;
    static constexpr uint32_t PV_READY_ID = 3;
    static constexpr uint32_t CAST_READY_ID = 4;

    // Parameters structure
    struct Params {
        // Data members
        uint32_t numHeads;
        uint32_t embeddingSize;
        uint32_t kvHeads;
        uint32_t batch;
        uint32_t tileM;
        uint32_t tileN;
        uint32_t tileK;
        uint32_t totalTaskNum;
        float siluScale;
        uint32_t layout;

        GM_ADDR ptrQ;
        GM_ADDR ptrK;
        GM_ADDR ptrV;
        GM_ADDR ptrO;
        GM_ADDR ptrActualQseqlen;
        GM_ADDR ptrActualKvseqlen;
        GM_ADDR ptrWorkspace;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(uint32_t numHeads_, uint32_t embeddingSize_, uint32_t kvHeads_, uint32_t batch_, uint32_t tileM_,
               uint32_t tileN_, uint32_t tileK_, uint32_t totalTaskNum_, float siluScale_, uint32_t layout_,
               GM_ADDR ptrQ_, GM_ADDR ptrK_, GM_ADDR ptrV_, GM_ADDR ptrO_, GM_ADDR ptrActualQseqlen_,
               GM_ADDR ptrActualKvseqlen_, GM_ADDR ptrWorkspace_)
            : numHeads(numHeads_), embeddingSize(embeddingSize_), kvHeads(kvHeads_), batch(batch_), tileM(tileM_),
              tileN(tileN_), tileK(tileK_), totalTaskNum(totalTaskNum_), siluScale(siluScale_), layout(layout_),
              ptrQ(ptrQ_), ptrK(ptrK_), ptrV(ptrV_), ptrO(ptrO_), ptrActualQseqlen(ptrActualQseqlen_),
              ptrActualKvseqlen(ptrActualKvseqlen_), ptrWorkspace(ptrWorkspace_) {}
    };

    struct Arguments {
        uint32_t numHeads;
        uint32_t embeddingSize;
        uint32_t kvHeads;
        uint32_t batch;
        uint32_t tileM;
        uint32_t tileN;
        uint32_t tileK;
        uint32_t totalTaskNum;
        float siluScale;
        uint32_t coreNum;
        uint32_t layout;

        GM_ADDR ptrQ;
        GM_ADDR ptrK;
        GM_ADDR ptrV;
        GM_ADDR ptrO;
        GM_ADDR ptrActualQseqlen;
        GM_ADDR ptrActualKvseqlen;
    };

    static bool CanImplement(const Arguments &args)
    {
        return true;
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        uint64_t qBlkSize = (args.batch + 1) * sizeof(ElementQBlkOffset);
        uint64_t qkMatmulOutSize = args.coreNum * args.tileM * MAX_SEQ_LEN * sizeof(ElementS);
        uint64_t siluScaleOutSize = args.coreNum * args.tileM * MAX_SEQ_LEN * sizeof(ElementP);
        uint64_t pvMatmulOutSize = args.coreNum * args.tileM * args.embeddingSize * sizeof(ElementO);
        size_t minSpaceSize = 2 * 1024 * 1024;
        size_t workspaceSize = static_cast<size_t>(qBlkSize + qkMatmulOutSize * WS_DOUBLE +
            siluScaleOutSize * WS_DOUBLE + pvMatmulOutSize * WS_DOUBLE);
        return minSpaceSize > workspaceSize ? minSpaceSize : workspaceSize;
    }

    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        Params params{
            args.numHeads,
            args.embeddingSize,
            args.kvHeads,
            args.batch,
            args.tileM,
            args.tileN,
            args.tileK,
            args.totalTaskNum,
            args.siluScale,
            args.layout,
            args.ptrQ,
            args.ptrK,
            args.ptrV,
            args.ptrO,
            args.ptrActualQseqlen,
            args.ptrActualKvseqlen,
            workspace
        };
        return params;
    }

    // Methods
    CATLASS_DEVICE
    HSTUTla() {}

    CATLASS_DEVICE
    void Init(Params const &params)
    {
        // coreId
        tmpBlockIdx = AscendC::GetBlockIdx();
        coreNum = AscendC::GetBlockNum();
        if ASCEND_IS_AIC {
            coreId = tmpBlockIdx;
        }

        if ASCEND_IS_AIV {
            tmpBlockIdx = AscendC::GetBlockIdx();
            coreId = tmpBlockIdx / 2;
        }

        // init tiling data
        constInfo.batch = params.batch;
        constInfo.numHeads = params.numHeads;
        constInfo.embeddingSize = params.embeddingSize;
        constInfo.kvHeads = params.kvHeads;
        constInfo.tileM = params.tileM;
        constInfo.tileN = params.tileN;
        constInfo.tileK = params.tileK;
        constInfo.totalTaskNum = params.totalTaskNum;
        constInfo.siluScale = params.siluScale;
        constInfo.layout = params.layout;

        // global tensor
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.ptrQ);
        gK.SetGlobalBuffer((__gm__ ElementK *)params.ptrK);
        gV.SetGlobalBuffer((__gm__ ElementV *)params.ptrV);
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.ptrActualQseqlen);
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.ptrActualKvseqlen);
        gO.SetGlobalBuffer((__gm__ ElementQ *)params.ptrO);

        // other const info
        constInfo.T1 = gActualQseqlen.GetValue(constInfo.batch);
        constInfo.T2 = gActualKvseqlen.GetValue(constInfo.batch);

        // workspace
        uint64_t offset = 0;
        constInfo.qkMatmulOutSize = params.tileM * MAX_SEQ_LEN;
        gS.SetGlobalBuffer((__gm__ ElementS *)(params.ptrWorkspace + offset +
                                               coreId * constInfo.qkMatmulOutSize * WS_DOUBLE * sizeof(ElementS)));
        offset += coreNum * constInfo.qkMatmulOutSize * WS_DOUBLE * sizeof(ElementS);
        constInfo.siluScaleOutSize = params.tileM * MAX_SEQ_LEN;
        gP.SetGlobalBuffer((__gm__ ElementP *)(params.ptrWorkspace + offset +
                                               coreId * constInfo.siluScaleOutSize * WS_DOUBLE * sizeof(ElementP)));
        offset += coreNum * constInfo.siluScaleOutSize * WS_DOUBLE * sizeof(ElementP);
        constInfo.pvMatmulOutSize = params.tileM * params.embeddingSize;
        gOFp32.SetGlobalBuffer((__gm__ ElementO *)(params.ptrWorkspace + offset +
                                                   coreId * constInfo.pvMatmulOutSize * WS_DOUBLE * sizeof(ElementO)));
        offset += coreNum * constInfo.pvMatmulOutSize * WS_DOUBLE * sizeof(ElementO);
        gQBlkOffset.SetGlobalBuffer((__gm__ ElementQBlkOffset *)(params.ptrWorkspace + offset));
        offset += (params.batch + 1) * sizeof(ElementQBlkOffset);

        // init gQBlkOffset
        uint32_t curQBlkOffset = 0;
        gQBlkOffset.SetValue(0, 0);
        for (uint32_t bIdx = 0; bIdx < constInfo.batch; bIdx++) {
            curQBlkOffset += CeilDiv(gActualQseqlen.GetValue(bIdx + 1) - gActualQseqlen.GetValue(bIdx),
                             constInfo.tileM);
            gQBlkOffset.SetValue(bIdx + 1, curQBlkOffset);
        }
    }

    CATLASS_DEVICE
    void SplitCore()
    {
        uint32_t minTaskNumPerCore = constInfo.totalTaskNum / coreNum;
        uint32_t deal1MoreTaskCoreNum = constInfo.totalTaskNum % coreNum;
        uint32_t coreDealTaskCnt = coreId < deal1MoreTaskCoreNum ?
            minTaskNumPerCore + 1 : minTaskNumPerCore;
        
        uint32_t deal1MoreStartTaskId = coreId < deal1MoreTaskCoreNum ? (minTaskNumPerCore + 1) * coreId :
            (minTaskNumPerCore + 1) * deal1MoreTaskCoreNum +
            minTaskNumPerCore * (coreId - deal1MoreTaskCoreNum);

        startTaskId = deal1MoreTaskCoreNum == 0 ? (coreId * coreDealTaskCnt) : deal1MoreStartTaskId;
        endTaskId = startTaskId + coreDealTaskCnt - 1;
    }

    CATLASS_DEVICE
    void AllocEventId()
    {
        if ASCEND_IS_AIC {
            // qk
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            
            // pv
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        }

        if ASCEND_IS_AIV {
            Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(castReady);
        }
    }

    CATLASS_DEVICE
    void FreeEventId()
    {
        if ASCEND_IS_AIC {
            // qk
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            
            // pv
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);

            Arch::CrossCoreWaitFlag(castReady);
        }

        if ASCEND_IS_AIV {
        }
    }

    CATLASS_DEVICE
    void CalRunInfo(uint32_t taskId, uint32_t loop)
    {
        uint32_t qBlkNum = constInfo.totalTaskNum / constInfo.numHeads;
        uint32_t headIdx = taskId / qBlkNum;
        uint32_t batchIdx = 0;
        uint32_t qBlkIdx = 0;
        for (uint32_t bIdx = 0; bIdx < constInfo.batch; bIdx++) {
            uint32_t curQBlkOffset = gQBlkOffset.GetValue(bIdx + 1);
            uint32_t preQBlkOffset = gQBlkOffset.GetValue(bIdx);
            if (taskId % qBlkNum < curQBlkOffset) {
                batchIdx = bIdx;
                qBlkIdx = taskId % qBlkNum - preQBlkOffset;
                break;
            }
        }

        uint32_t accumKvseqlen = gActualKvseqlen.GetValue(batchIdx);
        uint32_t accumQseqlen = gActualQseqlen.GetValue(batchIdx);
        uint32_t curQseqlen = gActualQseqlen.GetValue(batchIdx + 1) - gActualQseqlen.GetValue(batchIdx);
        uint32_t curKvseqlen = gActualKvseqlen.GetValue(batchIdx + 1) - gActualKvseqlen.GetValue(batchIdx);
        runInfo.qNTOffset = headIdx * constInfo.T1 + accumQseqlen + qBlkIdx * constInfo.tileM;
        runInfo.qTNOffset = (accumQseqlen + qBlkIdx * constInfo.tileM) * constInfo.numHeads + headIdx;
        runInfo.kNTOffset = headIdx * constInfo.T2 + accumKvseqlen;
        runInfo.kTNOffset = accumKvseqlen * constInfo.numHeads + headIdx;
        runInfo.actualQBlkSize = (qBlkIdx + 1) * constInfo.tileM > curQseqlen ?
                                 curQseqlen - qBlkIdx * constInfo.tileM : constInfo.tileM;
        runInfo.actualKvseqlen = curKvseqlen;
        runInfo.loop = loop;
    }

    CATLASS_DEVICE
    void ProcessQkMatmul(BlockMmadQK blockMmadQK)
    {
        uint64_t qRowOffset = LAYOUT(constInfo.layout) == LAYOUT::NTD ? runInfo.qNTOffset : runInfo.qTNOffset;
        auto layoutQCube = tla::MakeLayout<ElementQ, LayoutQ>(constInfo.numHeads * constInfo.T1,
                                                              constInfo.embeddingSize);
        auto tensorQ = tla::MakeTensor(gQ, layoutQCube, Arch::PositionGM{});
        auto tensorTileQ = GetTile(tensorQ, tla::MakeCoord(qRowOffset, 0),
            tla::MakeShape(runInfo.actualQBlkSize, constInfo.embeddingSize));
        blockMmadQK.loadQGM(tensorTileQ, runInfo.loop, constInfo.numHeads, LAYOUT(constInfo.layout) == LAYOUT::NTD);

        uint64_t kvRowOffset = LAYOUT(constInfo.layout) == LAYOUT::NTD ? runInfo.kNTOffset : runInfo.kTNOffset;
        auto layoutKCube = tla::MakeLayout<ElementK, LayoutK>(constInfo.kvHeads * constInfo.T2,
                                                              constInfo.embeddingSize);
        auto tensorK = tla::MakeTensor(gK, layoutKCube, Arch::PositionGM{});
        auto tensorTileK = GetTile(tensorK, tla::MakeCoord(kvRowOffset, 0),
            tla::MakeShape(runInfo.actualKvseqlen, constInfo.embeddingSize));

        GemmCoord actualBlockShapeQK{runInfo.actualQBlkSize, runInfo.actualKvseqlen, constInfo.embeddingSize}; // (m,n,k)
        auto layoutSCube = tla::MakeLayout<ElementS, LayoutS>(constInfo.tileM, runInfo.actualKvseqlen);
        auto tensorS = tla::MakeTensor(gS[(runInfo.loop % WS_DOUBLE) * constInfo.qkMatmulOutSize],
                                       layoutSCube, Arch::PositionGM{});
        blockMmadQK(tensorTileQ, tensorTileK, tensorS, actualBlockShapeQK, constInfo.kvHeads,
                    LAYOUT(constInfo.layout) == LAYOUT::NTD);
    }

    CATLASS_DEVICE
    void ProcessSilu(EpilogueSiluScale epilogueSiluScale)
    {
        epilogueSiluScale(gS[(runInfo.loop % WS_DOUBLE) * constInfo.qkMatmulOutSize],
                          gP[(runInfo.loop % WS_DOUBLE) * constInfo.siluScaleOutSize],
                          runInfo.actualQBlkSize, runInfo.actualKvseqlen, tmpBlockIdx, constInfo.siluScale);
    }

    CATLASS_DEVICE
    void ProcessPvMatmul(BlockMmadPV blockMmadPV)
    {
        auto layoutPCube = tla::MakeLayout<ElementP, LayoutP>(runInfo.actualQBlkSize, runInfo.actualKvseqlen);
        auto tensorP = tla::MakeTensor(gP[(runInfo.loop % WS_DOUBLE) * constInfo.siluScaleOutSize],
                                       layoutPCube, Arch::PositionGM{});
        uint64_t kvRowOffset = LAYOUT(constInfo.layout) == LAYOUT::NTD ? runInfo.kNTOffset : runInfo.kTNOffset;

        auto layoutVCube = tla::MakeLayout<ElementV, LayoutV>(constInfo.kvHeads * constInfo.T2,
                                                              constInfo.embeddingSize);
        auto tensorV = tla::MakeTensor(gV, layoutVCube, Arch::PositionGM{});
        auto tensorTileV = GetTile(tensorV, tla::MakeCoord(kvRowOffset, 0),
            tla::MakeShape(runInfo.actualKvseqlen, constInfo.embeddingSize));
        
        GemmCoord actualBlockShapePV{runInfo.actualQBlkSize, constInfo.embeddingSize, runInfo.actualKvseqlen}; // (m,n,k)

        auto layoutOCube = tla::MakeLayout<ElementO, LayoutO>(runInfo.actualQBlkSize, constInfo.embeddingSize);
        auto tensorO = tla::MakeTensor(gOFp32[(runInfo.loop % WS_DOUBLE) * constInfo.pvMatmulOutSize],
                                       layoutOCube, Arch::PositionGM{});
        blockMmadPV(tensorP, tensorTileV, tensorO, actualBlockShapePV, constInfo.kvHeads,
                    LAYOUT(constInfo.layout) == LAYOUT::NTD);
    }

    CATLASS_DEVICE
    void ProcessCast(EpilogueSiluScale epilogueSiluScale)
    {
        uint64_t qRowOffset = LAYOUT(constInfo.layout) == LAYOUT::NTD ? runInfo.qNTOffset : runInfo.qTNOffset;
        epilogueSiluScale(gOFp32[(runInfo.loop % WS_DOUBLE) * constInfo.pvMatmulOutSize],
                          gO[qRowOffset * constInfo.embeddingSize],
                          runInfo.actualQBlkSize, constInfo.embeddingSize,
                          constInfo.numHeads, tmpBlockIdx, LAYOUT(constInfo.layout) == LAYOUT::NTD);
    }

    CATLASS_DEVICE
    void WorkspaceInit(EpilogueSiluScale epilogueSiluScale)
    {
        epilogueSiluScale(gOFp32[(runInfo.loop % WS_DOUBLE) * constInfo.pvMatmulOutSize], constInfo.pvMatmulOutSize,
                          0.f, tmpBlockIdx);
    }

    CATLASS_DEVICE
    void Process()
    {
        BlockMmadQK blockMmadQK(resource);
        static constexpr uint32_t L1_QK_SIZE =
            BlockMmadQK::L1_TILE_M * BlockMmadQK::L1_TILE_K * sizeof(ElementQ) +
            BlockMmadQK::L1_TILE_N * BlockMmadQK::L1_TILE_K * sizeof(ElementK) * 2;
        BlockMmadPV blockMmadPV(resource, L1_QK_SIZE);
        EpilogueSiluScale epilogueSiluScale(resource, LAYOUT(constInfo.layout) == LAYOUT::NTD);

        AllocEventId();
        uint32_t loop = 0;
        for (uint32_t taskId = startTaskId; taskId <= endTaskId; taskId++) {
            CalRunInfo(taskId, loop);
            if ASCEND_IS_AIC {
                Arch::CrossCoreWaitFlag(castReady);
                ProcessQkMatmul(blockMmadQK);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
            }

            if ASCEND_IS_AIV {
                Arch::CrossCoreWaitFlag(qkReady);
                ProcessSilu(epilogueSiluScale);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(siluScaleReady);
            }

            if ASCEND_IS_AIC {
                Arch::CrossCoreWaitFlag(siluScaleReady);
                ProcessPvMatmul(blockMmadPV);
                Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
            }

            if ASCEND_IS_AIV {
                Arch::CrossCoreWaitFlag(pvReady);
                ProcessCast(epilogueSiluScale);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(loop % 2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(loop % 2);
                WorkspaceInit(epilogueSiluScale);
                Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(castReady);
            }
            loop += 1;
        }
        FreeEventId();
    }

    CATLASS_DEVICE
    void operator()(Params const &params)
    {
        Init(params);
        SplitCore();
        Process();
    }

private:
    // aic、aiv核信息
    uint32_t tmpBlockIdx;
    uint32_t coreId;
    uint32_t coreNum;
    uint32_t startTaskId;
    uint32_t endTaskId;

    HSTUConstInfo constInfo{};
    HSTURunInfo runInfo{};

    Arch::Resource<ArchTag> resource;
    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag siluScaleReady{SILUSCALE_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
    Arch::CrossCoreFlag castReady{CAST_READY_ID};

    // global tensor
    AscendC::GlobalTensor<ElementQ> gQ;
    AscendC::GlobalTensor<ElementK> gK;
    AscendC::GlobalTensor<ElementV> gV;
    AscendC::GlobalTensor<int64_t> gActualQseqlen;
    AscendC::GlobalTensor<int64_t> gActualKvseqlen;
    AscendC::GlobalTensor<ElementQ> gO;
    AscendC::GlobalTensor<ElementQBlkOffset> gQBlkOffset;
    AscendC::GlobalTensor<ElementS> gS;
    AscendC::GlobalTensor<ElementP> gP;
    AscendC::GlobalTensor<ElementO> gOFp32;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_HSTU_TLA_HPP