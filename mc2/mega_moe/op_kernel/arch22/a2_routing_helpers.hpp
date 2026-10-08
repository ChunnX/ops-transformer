/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_A2_ROUTING_HELPERS_HPP
#define MEGA_MOE_A2_ROUTING_HELPERS_HPP
#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/matrix_coord.hpp"
#include "utils/layout3d.hpp"
#include "utils/hccl_shmem.hpp"
#include "utils/const_args.hpp"

namespace Catlass::Gemm::Kernel::Detail {
using namespace AscendC;

constexpr int32_t kA2UbMoveNum = 16 * 1024;

template <class ArchTag>
CATLASS_DEVICE void GetCumsumForMMAIVShared(Arch::Resource<ArchTag>& resource, int32_t paddedExpertNumAligned,
                                            AscendC::GlobalTensor<int32_t>& tokenPerExpert,
                                            AscendC::GlobalTensor<int32_t>& result, uint32_t expertPerRank, uint32_t EP)
{
    constexpr uint32_t int32PerBlock = 8;
    constexpr uint32_t ubReservedBytes = 32;
    uint32_t totalExpertNum = EP * expertPerRank;
    // Cumsum is independent for every global expert. Tile that dimension so the [EP, tileExpertNum]
    // working set always stays in the 192-KiB A2 UB.
    uint32_t maxExpertNumPerLoop =
        (ArchTag::UB_SIZE - ubReservedBytes) / sizeof(int32_t) / EP / int32PerBlock * int32PerBlock;
    AscendC::LocalTensor<int32_t> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int32_t>(0);
#define U16(x) static_cast<uint16_t>(x)

    for (uint32_t expertOffset = 0; expertOffset < totalExpertNum; expertOffset += maxExpertNumPerLoop) {
        uint32_t remainingExpertNum = totalExpertNum - expertOffset;
        uint32_t currentExpertNum = remainingExpertNum < maxExpertNumPerLoop ? remainingExpertNum : maxExpertNumPerLoop;
        uint32_t currentExpertNumAligned = (currentExpertNum + int32PerBlock - 1) / int32PerBlock * int32PerBlock;

        AscendC::DataCopyPad(tmpBuffer1, tokenPerExpert[expertOffset],
                             {U16(EP), U16(currentExpertNum * sizeof(int32_t)),
                              U16((paddedExpertNumAligned - currentExpertNum) * sizeof(int32_t)), 0},
                             {});

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);

        for (uint32_t i = 1; i < EP; ++i) {
            AscendC::Add(tmpBuffer1[i * currentExpertNumAligned], tmpBuffer1[i * currentExpertNumAligned],
                         tmpBuffer1[(i - 1) * currentExpertNumAligned], currentExpertNum);
            AscendC::PipeBarrier<PIPE_V>();
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

        AscendC::DataCopyPad(result[expertOffset], tmpBuffer1,
                             {U16(EP), U16(currentExpertNum * sizeof(int32_t)), 0,
                              U16((paddedExpertNumAligned - currentExpertNum) * sizeof(int32_t))});
        // The next loop reuses the same UB range for MTE2. Wait until MTE3 has consumed it.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    // cumsumMM 写出落盘后再返回，消费方（AIC 标量读 / 其他 AIV 经 SyncAll 读取）方可读到
    // （与 A3 GetCumsumForMMAIV 的修复方式一致）
    AscendC::PipeBarrier<PIPE_MTE3>();
}

CATLASS_DEVICE void ResetGmBufferShared(AscendC::GlobalTensor<int32_t>& dst, AscendC::LocalTensor<int32_t>& zeroBuffer,
                                        int32_t num)
{
    for (int32_t offset = 0; offset < num; offset += kA2UbMoveNum) {
        int32_t remainingNum = num - offset;
        int32_t currentNum = remainingNum < kA2UbMoveNum ? remainingNum : kA2UbMoveNum;
        AscendC::DataCopy(dst[offset], zeroBuffer, currentNum);
    }
}

template <class ArchTag, class PeerInfo>
CATLASS_DEVICE void ResetTokenPerExpertShared(Arch::Resource<ArchTag>& resource, HcclShmem<true>& shmem,
                                              const PeerInfo& peermemInfo,
                                              AscendC::GlobalTensor<int32_t>& tokenPerExpert, int32_t coreNum,
                                              int32_t coreIdx, int32_t EP, int32_t expertPerRank, int32_t num)
{
    if (coreNum == 0 || coreIdx != coreNum - 1 || num <= 0) {
        return;
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    AscendC::LocalTensor<int32_t> tmp = resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::Duplicate(tmp, 0, kA2UbMoveNum);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);

    // 1) tokenPerExpert winIn
    ResetGmBufferShared(tokenPerExpert, tmp, num);

    // 2) tokenPerExpert winOut
    AscendC::GlobalTensor<int32_t> tokenPerExpertWinOut;
    tokenPerExpertWinOut.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(shmem.windowsOutAddr() + peermemInfo.offsetPeerTokenPerExpert));
    ResetGmBufferShared(tokenPerExpertWinOut, tmp, num);

    // 3) Dispatch Flag winIn（按 64B 槽布局 = EP × expertPerRank × kFlagSlotI32 个 int32）
    int32_t dispatchFlagInts = EP * expertPerRank * PeerInfo::kFlagSlotI32;
    AscendC::GlobalTensor<int32_t> flagWinIn;
    flagWinIn.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetFlag));
    ResetGmBufferShared(flagWinIn, tmp, dispatchFlagInts);

    // 4) Dispatch Flag winOut
    AscendC::GlobalTensor<int32_t> flagWinOut;
    flagWinOut.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shmem.windowsOutAddr() + peermemInfo.offsetFlag));
    ResetGmBufferShared(flagWinOut, tmp, dispatchFlagInts);

    // 5) Allgather Flag winIn（EP × kFlagSlotI32 个 int32）
    int32_t allgatherFlagInts = EP * PeerInfo::kFlagSlotI32;
    AscendC::GlobalTensor<int32_t> agFlagWinIn;
    agFlagWinIn.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetAllgatherFlag));
    ResetGmBufferShared(agFlagWinIn, tmp, allgatherFlagInts);

    // 6) Allgather Flag winOut
    AscendC::GlobalTensor<int32_t> agFlagWinOut;
    agFlagWinOut.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(shmem.windowsOutAddr() + peermemInfo.offsetAllgatherFlag));
    ResetGmBufferShared(agFlagWinOut, tmp, allgatherFlagInts);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
}

template <class CopyUbToGm>
CATLASS_DEVICE void CrossRankAllGatherSelfSendShared(AscendC::GlobalTensor<int32_t>& dstAddress,
                                                     AscendC::LocalTensor<int32_t>& tmpBuffer, CopyUbToGm& copyUbToGm,
                                                     uint32_t numPerCore)
{
    copyUbToGm(dstAddress, tmpBuffer, layout::RowMajor{1, numPerCore}, layout::RowMajor{1, numPerCore});
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <class PeerInfo, class CopyUbToGm>
CATLASS_DEVICE void CrossRankAllGatherSameServerSendShared(HcclShmem<true>& shmem, const PeerInfo& peermemInfo,
                                                           int32_t rank, int32_t dstEpIdx, uint32_t numPerCore,
                                                           AscendC::GlobalTensor<int32_t>& dstAddress,
                                                           AscendC::LocalTensor<int32_t>& tmpBuffer,
                                                           CopyUbToGm& copyUbToGm)
{
    copyUbToGm(dstAddress, tmpBuffer, layout::RowMajor{1, numPerCore}, layout::RowMajor{1, numPerCore});
    AscendC::PipeBarrier<PIPE_ALL>();
    __gm__ int32_t* remoteAgFlag =
        reinterpret_cast<__gm__ int32_t*>(shmem(0, dstEpIdx) + peermemInfo.offsetAllgatherFlag) +
        rank * PeerInfo::kFlagSlotI32;
    gm_store(remoteAgFlag, FLAG_VALUE_MAGIC);
    gm_dcci((__gm__ uint8_t*)remoteAgFlag);
}

template <class PeerInfo, class CopyUbToGm>
CATLASS_DEVICE void CrossRankAllGatherRemoteSendShared(
    HcclShmem<true>& shmem, const PeerInfo& peermemInfo, __gm__ HcclAiRMAInfo* qp_info_, int32_t rank, int32_t dstEpIdx,
    uint32_t numPerCore, AscendC::GlobalTensor<int32_t>& dstAddress, AscendC::GlobalTensor<int32_t>& localDstAddress,
    AscendC::LocalTensor<int32_t>& tmpBuffer, CopyUbToGm& copyUbToGm, AscendC::LocalTensor<uint64_t>& rdmaUbLocal,
    AscendC::LocalTensor<uint32_t>& rdmaUbLocalHead, AscendC::LocalTensor<int32_t>& flagUb)
{
    // 路径 C：跨 server，先 RDMA 数据，再 RDMA 32B flag
    copyUbToGm(localDstAddress, tmpBuffer, layout::RowMajor{1, numPerCore}, layout::RowMajor{1, numPerCore});
    AscendC::PipeBarrier<PIPE_ALL>();
    AIVRDMAPostSend((GM_ADDR)localDstAddress.GetPhyAddr(), (GM_ADDR)dstAddress.GetPhyAddr(), dstEpIdx,
                    numPerCore * sizeof(int32_t), qp_info_, rdmaUbLocal, rdmaUbLocalHead);
    AscendC::PipeBarrier<PIPE_ALL>();

    // 准备 32B flag payload：[0]=magic，其余 7 个 i32 = 0
    AscendC::Duplicate<int32_t>(flagUb, 0, PeerInfo::kFlagPayloadI32);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    flagUb.SetValue(0, FLAG_VALUE_MAGIC);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

    // UB → 本地 winOut allgather flag 槽（32B 一次写）
    AscendC::GlobalTensor<int32_t> localAgFlagGm;
    localAgFlagGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(shmem.windowsOutAddr() + peermemInfo.offsetAllgatherFlag) +
        rank * PeerInfo::kFlagSlotI32);
    AscendC::DataCopy(localAgFlagGm, flagUb, PeerInfo::kFlagPayloadI32);
    AscendC::PipeBarrier<PIPE_ALL>();

    // RDMA 32B 到对端同槽
    GM_ADDR remoteAgFlagAddr = reinterpret_cast<GM_ADDR>(
        reinterpret_cast<uintptr_t>(shmem(0, dstEpIdx)) + static_cast<uintptr_t>(peermemInfo.offsetAllgatherFlag) +
        static_cast<uintptr_t>(rank) * static_cast<uintptr_t>(PeerInfo::kFlagSlotI32 * sizeof(int32_t)));
    AIVRDMAPostSend((GM_ADDR)localAgFlagGm.GetPhyAddr(), remoteAgFlagAddr, static_cast<uint64_t>(dstEpIdx),
                    PeerInfo::kFlagPayloadI32 * sizeof(int32_t), // 32B
                    qp_info_, rdmaUbLocal, rdmaUbLocalHead);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <class ArchTag, class PeerInfo, class ServerId>
CATLASS_DEVICE void CrossRankAllGatherSendShared(
    HcclShmem<true>& shmem, const PeerInfo& peermemInfo, Layout3D& tokenPerExpertLayout, __gm__ HcclAiRMAInfo* qp_info_,
    int64_t localTokenPerExpertOffset, int32_t rank, uint32_t coreIdx, uint32_t coreNum, ServerId serverId, int32_t EP,
    uint32_t numPerCore, AscendC::LocalTensor<int32_t>& tmpBuffer, AscendC::LocalTensor<uint64_t>& rdmaUbLocal,
    AscendC::LocalTensor<uint32_t>& rdmaUbLocalHead, AscendC::LocalTensor<int32_t>& flagUb)
{
    // Stage 1：发送（每 core 处理若干 dstEpIdx）
    for (int32_t dstEpIdx = coreIdx, dstServerId = 0; dstEpIdx < EP; dstEpIdx += coreNum) {
        dstServerId = dstEpIdx / SERVER_RANK_SIZE_A2;
        AscendC::GlobalTensor<int32_t> srcAddress;
        srcAddress.SetGlobalBuffer(
            reinterpret_cast<__gm__ int32_t*>(shmem.windowsOutAddr() + localTokenPerExpertOffset));
        AscendC::GlobalTensor<int32_t> dstAddress;
        __gm__ void* dstPeermemPtr = shmem(localTokenPerExpertOffset, dstEpIdx);

        dstAddress.SetGlobalBuffer((__gm__ int32_t*)dstPeermemPtr);
        AscendC::GlobalTensor<int32_t> localDstAddress;
        GM_ADDR localDstPeermemPtr = shmem.windowsOutAddr() + peermemInfo.offsetPeerTokenPerExpert +
                                     tokenPerExpertLayout(dstEpIdx, 0, 0) * sizeof(int32_t);
        localDstAddress.SetGlobalBuffer((__gm__ int32_t*)localDstPeermemPtr);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        using TType = Gemm::GemmType<int32_t, layout::RowMajor>;
        using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
        using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
        CopyGmToUb copyGmToUb;
        CopyUbToGm copyUbToGm;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);

        copyGmToUb(tmpBuffer, srcAddress[0], layout::RowMajor{1, numPerCore}, layout::RowMajor{1, numPerCore});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        if (dstEpIdx == static_cast<int32_t>(rank)) {
            // 路径 A：self-rank，本地搬运，无 flag
            CrossRankAllGatherSelfSendShared(dstAddress, tmpBuffer, copyUbToGm, numPerCore);
        } else if (dstServerId == serverId) {
            // 路径 B：同 server IPC 直写数据 + gm_store + gm_dcci 写 flag
            CrossRankAllGatherSameServerSendShared(shmem, peermemInfo, rank, dstEpIdx, numPerCore, dstAddress,
                                                   tmpBuffer, copyUbToGm);
        } else {
            CrossRankAllGatherRemoteSendShared(shmem, peermemInfo, qp_info_, rank, dstEpIdx, numPerCore, dstAddress,
                                               localDstAddress, tmpBuffer, copyUbToGm, rdmaUbLocal, rdmaUbLocalHead,
                                               flagUb);
        }
    }
}

template <class PeerInfo>
CATLASS_DEVICE void WaitAllGatherFlagShared(HcclShmem<true>& shmem, const PeerInfo& peermemInfo, int32_t dstEpIdx,
                                            int32_t rank)
{
    // self-rank 已在发送阶段本地搬运完毕，无 flag。
    if (dstEpIdx != rank) {
        __gm__ int32_t* flagAddr = reinterpret_cast<__gm__ int32_t*>(shmem() + peermemInfo.offsetAllgatherFlag) +
                                   dstEpIdx * PeerInfo::kFlagSlotI32;
        while (true) {
            gm_dcci(flagAddr);
            if (gm_load(flagAddr) == FLAG_VALUE_MAGIC) {
                break;
            }
        }
    }
}

template <class PeerInfo>
CATLASS_DEVICE void CrossRankAllGatherCollectShared(
    HcclShmem<true>& shmem, const PeerInfo& peermemInfo, Layout3D& tokenPerExpertLayout,
    AscendC::GlobalTensor<int32_t>& tokenPerExpert, AscendC::GlobalTensor<int32_t>& preSumBeforeRankForDispatch,
    AscendC::GlobalTensor<int32_t>& preSumBeforeRankForCombine, int32_t rank, uint32_t coreIdx, uint32_t coreNum,
    int32_t EP, int32_t expertPerRank, uint32_t numPerCore, AscendC::LocalTensor<int32_t>& tmpBuffer,
    AscendC::LocalTensor<int32_t>& prevSumBuf)
{
    // Stage 2：等 flag + 读 tokenPerExpert + 算 preSumBeforeRank
    for (int32_t dstEpIdx = coreIdx; dstEpIdx < EP; dstEpIdx += coreNum) {
        WaitAllGatherFlagShared(shmem, peermemInfo, dstEpIdx, rank);

        AscendC::DataCopy(tmpBuffer, tokenPerExpert[tokenPerExpertLayout(dstEpIdx, 0, 0)], numPerCore);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);

        // 只有 dstEpIdx == rank 的 core 才拥有本 rank 自身的 token 分布，
        // 只允许该 core 写 preSumBeforeRankForDispatch（避免写竞争）
        if (static_cast<uint32_t>(dstEpIdx) == rank) {
            uint32_t realExpertNum = EP * expertPerRank;
            for (uint32_t i = 0, currentSum = 0; i < realExpertNum; i++) {
                prevSumBuf(i) = currentSum;
                currentSum += tmpBuffer(i);
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(
                preSumBeforeRankForDispatch, prevSumBuf,
                AscendC::DataCopyParams{1, static_cast<uint16_t>(realExpertNum * sizeof(int32_t)), 0, 0});
        }
        for (int32_t i = 0, j = 0, prevSum = 0; i < (rank + 1) * expertPerRank; i++) {
            if (i >= rank * expertPerRank) {
                prevSumBuf(j) = prevSum;
                j++;
            }
            prevSum += tmpBuffer(i);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(preSumBeforeRankForCombine[dstEpIdx * expertPerRank], prevSumBuf,
                             AscendC::DataCopyParams{1, static_cast<uint16_t>(expertPerRank * sizeof(int32_t)), 0, 0});
    }
}

template <class ArchTag, class PeerInfo, class ServerId>
CATLASS_DEVICE void CrossRankAllGatherShared(Arch::Resource<ArchTag>& resource, HcclShmem<true>& shmem,
                                             const PeerInfo& peermemInfo, Layout3D& tokenPerExpertLayout,
                                             AscendC::GlobalTensor<int32_t>& tokenPerExpert,
                                             AscendC::GlobalTensor<int32_t>& preSumBeforeRankForDispatch,
                                             AscendC::GlobalTensor<int32_t>& preSumBeforeRankForCombine,
                                             __gm__ HcclAiRMAInfo* qp_info_, int64_t localTokenPerExpertOffset,
                                             int32_t rank, uint32_t coreIdx, uint32_t coreNum, ServerId serverId,
                                             int32_t paddedExpertNumAligned, int32_t EP, int32_t expertPerRank)
{
    uint32_t numPerCore = paddedExpertNumAligned;
    AscendC::LocalTensor<int32_t> tmpBuffer = resource.ubBuf.template GetBufferByByte<int32_t>(0);
    AscendC::LocalTensor<int32_t> prevSumBuf = tmpBuffer[numPerCore];
    uint64_t ubOffet = 2 * numPerCore * sizeof(uint32_t);
    AscendC::LocalTensor<uint64_t> rdmaUbLocal = resource.ubBuf.template GetBufferByByte<uint64_t>(ubOffet);
    AscendC::LocalTensor<uint32_t> rdmaUbLocalHead =
        resource.ubBuf.template GetBufferByByte<uint32_t>(ubOffet + UB_ALIGN);
    // flagUb：用于路径 C 拼 32B flag payload，放在 rdmaUbLocalHead 之后
    AscendC::LocalTensor<int32_t> flagUb = resource.ubBuf.template GetBufferByByte<int32_t>(ubOffet + 2 * UB_ALIGN);

    CrossRankAllGatherSendShared<ArchTag>(shmem, peermemInfo, tokenPerExpertLayout, qp_info_, localTokenPerExpertOffset,
                                          rank, coreIdx, coreNum, serverId, EP, numPerCore, tmpBuffer, rdmaUbLocal,
                                          rdmaUbLocalHead, flagUb);
    CrossRankAllGatherCollectShared(shmem, peermemInfo, tokenPerExpertLayout, tokenPerExpert,
                                    preSumBeforeRankForDispatch, preSumBeforeRankForCombine, rank, coreIdx, coreNum, EP,
                                    expertPerRank, numPerCore, tmpBuffer, prevSumBuf);

    AscendC::SyncAll<true>();
}
// ============================================================
// ResetTokenPerExpert：每轮使用前由末核分块清零下列 6 块区域
//   - tokenPerExpert winIn   : 长度 num
//   - tokenPerExpert winOut  : 长度 num
//   - Dispatch  Flag winIn   : 长度 EP × expertPerRank × kFlagSlotI32（每槽 16 个 int32 = 64B）
//   - Dispatch  Flag winOut  : 同上
//   - Allgather Flag winIn   : 长度 EP × kFlagSlotI32
//   - Allgather Flag winOut  : 同上
// ============================================================

template <class L1TileShape, class TileCoord, class LayoutA, class LayoutB, class LayoutC>
CATLASS_DEVICE void GetGmmTileOffsets(const TileCoord& blockCoord, LayoutA& layoutA, LayoutB& layoutB, LayoutC& layoutC,
                                      int64_t& gmOffsetA, int64_t& gmOffsetB, int64_t& gmOffsetC)
{
    MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
    MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
    MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
    gmOffsetA = layoutA.GetOffset(offsetA);
    gmOffsetB = layoutB.GetOffset(offsetB);
    gmOffsetC = layoutC.GetOffset(offsetC);
}

} // namespace Catlass::Gemm::Kernel::Detail

#endif // MEGA_MOE_A2_ROUTING_HELPERS_HPP
