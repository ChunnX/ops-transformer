/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_ARCH22_TILING_ADAPTER_H
#define FLASH_MLA_ARCH22_TILING_ADAPTER_H

#include "../flash_mla_metadata_protocol.h"

namespace FlashMla {
__aicore__ inline bool MlaIsBsh(MlaLayout layout)
{
    return layout == MlaLayout::BSH || layout == MlaLayout::BSND;
}

__aicore__ inline bool MlaStaticInstanceSupported(const MlaStaticLaunchInfo &launch)
{
    if (launch.queryDataType != kMlaFp16 && launch.queryDataType != kMlaBf16 && launch.queryDataType != kMlaInt8) {
        return false;
    }
    if (launch.family == MlaFamily::NO_SCHEDULE) {
        return launch.physicalBlockDim > 0 && launch.physicalBlockDim <= launch.aivNum;
    }
    if (launch.aicNum == 0 || launch.aicNum > kMlaMaxAicNum || launch.cvRatio < 1 || launch.cvRatio > 2 ||
        launch.pageAttention > 1 || launch.treeMask > 1 || launch.aivNum != launch.aicNum * launch.cvRatio ||
        launch.physicalBlockDim == 0 || launch.sysWorkspaceSize > launch.workspaceCap) {
        return false;
    }
    if (launch.family == MlaFamily::LEGACY_MLA) {
        return launch.queryDataType == kMlaInt8 && launch.pageAttention == 1 && launch.layoutKv == MlaLayout::NZ &&
               (MlaIsBsh(launch.layoutQ) || launch.layoutQ == MlaLayout::TND);
    }
    if (launch.family != MlaFamily::ARCH22_MLA || launch.queryDataType == kMlaInt8) {
        return false;
    }
    if (!MlaIsBsh(launch.layoutQ) && launch.layoutQ != MlaLayout::BNSD && launch.layoutQ != MlaLayout::TND) {
        return false;
    }
    if (launch.pageAttention != 0) {
        return MlaIsBsh(launch.layoutKv) || launch.layoutKv == MlaLayout::BNSD || launch.layoutKv == MlaLayout::NZ;
    }
    return (MlaIsBsh(launch.layoutQ) && MlaIsBsh(launch.layoutKv)) ||
           (launch.layoutQ == launch.layoutKv &&
            (launch.layoutQ == MlaLayout::BNSD || launch.layoutQ == MlaLayout::TND));
}

__aicore__ inline MlaStatus ValidateMlaLaunch(const MlaStaticLaunchInfo &launch, uint32_t blockNum)
{
    if (!MlaStaticInstanceSupported(launch) || launch.softmaxLseFlag > 1 ||
        launch.staticLaunchKey != MakeMlaStaticLaunchKey(launch.family, launch.queryDataType, launch.layoutQ,
                                                         launch.layoutKv, launch.pageAttention != 0,
                                                         launch.treeMask != 0, launch.cvRatio)) {
        return MlaStatus::INVALID_PATCH;
    }
    // GetBlockNum is the logical AIC count for MIX kernels. It also determines
    // every original per-core workspace offset, independently of usedCoreNum.
    if (blockNum != (launch.family == MlaFamily::NO_SCHEDULE ? launch.physicalBlockDim : launch.aicNum)) {
        return MlaStatus::INVALID_PATCH;
    }
    return MlaStatus::SUCCESS;
}

__aicore__ inline bool MlaAddWorkspaceRegion(uint64_t elements, uint64_t elementBytes, uint64_t copies, uint64_t &end)
{
    uint64_t bytes;
    return MlaCheckedMul(elements, elementBytes, bytes) && MlaCheckedMul(bytes, copies, bytes) &&
           MlaCheckedAdd(end, bytes, end);
}

__aicore__ inline MlaStatus MlaCheckWorkspace(const MlaStaticLaunchInfo &launch, const MlaMetadataHeader &metadata,
                                              uint64_t layout)
{
    if (launch.sysWorkspaceSize > launch.workspaceCap) {
        return MlaStatus::CAPACITY_EXCEEDED;
    }
    return layout <= launch.workspaceCap && metadata.workspaceNeed <= launch.workspaceCap - launch.sysWorkspaceSize ?
               MlaStatus::SUCCESS :
               MlaStatus::CAPACITY_EXCEEDED;
}

// maskDim1/maskDim2 hold the trailing Q/KV axes for ranks 3 and 4. Host
// independently validates the rank-4 singleton head axis.
__aicore__ inline MlaStatus FinalizeMlaMask(const MlaStaticLaunchInfo &launch, const MlaMetadataHeader &metadata,
                                            uint64_t batchSize, uint64_t s1, uint64_t s2, uint32_t maskFlag,
                                            uint32_t sparseMode)
{
    if ((maskFlag != 0) != (launch.maskRank != 0) || (sparseMode == 9) != (launch.treeMask != 0)) {
        return MlaStatus::INVALID_PATCH;
    }
    if (maskFlag == 0) {
        return MlaStatus::SUCCESS;
    }
    if (launch.treeMask != 0) {
        // The active FIA checker validates only rank; legacy IFA requires an
        // exact shape, with packed TND using the sum of per-batch Q squares.
        if (launch.family == MlaFamily::ARCH22_MLA) {
            return launch.maskRank == 1 || launch.maskRank == 3 ? MlaStatus::SUCCESS : MlaStatus::INVALID_PATCH;
        }
        if (launch.layoutQ == MlaLayout::TND) {
            return launch.maskRank == 1 && launch.maskDim0 == metadata.qSeqSquareSum ? MlaStatus::SUCCESS :
                                                                                       MlaStatus::INVALID_PATCH;
        }
        return launch.maskRank == 3 && launch.maskDim0 == batchSize && launch.maskDim1 == s1 && launch.maskDim2 == s1 ?
                   MlaStatus::SUCCESS :
                   MlaStatus::INVALID_PATCH;
    }
    if (launch.maskRank < 2 || launch.maskRank > 4) {
        return MlaStatus::INVALID_PATCH;
    }
    const uint64_t maskBatch = launch.maskRank == 2 ? 1 : launch.maskDim0;
    const uint64_t maskQ = launch.maskRank == 2 ? launch.maskDim0 : launch.maskDim1;
    const uint64_t maskKv = launch.maskRank == 2 ? launch.maskDim1 : launch.maskDim2;
    if (sparseMode == 0 || sparseMode == 1) {
        // FIA checks actual maximum KV for all dense layouts and the rounded
        // page extent for PA. Storage capacity alone cannot decide this on Host.
        const uint64_t requiredKv = launch.pageAttention != 0 ? s2 : metadata.maxKv;
        if ((maskBatch != 1 && maskBatch != batchSize) || maskQ < s1 || maskKv < requiredKv) {
            return MlaStatus::INVALID_PATCH;
        }
    } else if (sparseMode == 2 || sparseMode == 3 || sparseMode == 4) {
        if (launch.family == MlaFamily::LEGACY_MLA && launch.maskRank != 2) {
            return MlaStatus::INVALID_PATCH;
        }
        if (maskBatch != 1 || maskQ != 2048 || maskKv != 2048) {
            return MlaStatus::INVALID_PATCH;
        }
    } else {
        return MlaStatus::INVALID_PATCH;
    }
    return MlaStatus::SUCCESS;
}

__aicore__ inline MlaStatus ValidateMlaNonQuantFd(const MlaStaticLaunchInfo &launch, const MlaNonQuantTiling &tiling)
{
    const auto &fd = tiling.fdParams;
    if (fd.numOfFdHead == 0) {
        return fd.usedVecNumOfFd == 0 ? MlaStatus::SUCCESS : MlaStatus::INVALID_PATCH;
    }
    // FiaBlockVecFlashDecode reserves 16 KiB for each FP32 D512 row buffer.
    constexpr uint32_t maxFdRows = 16U * 1024U / (512U * sizeof(float));
    constexpr uint32_t lseBufferElements = 6U * 1024U / sizeof(float);
    if (fd.gS1BaseSizeOfFd == 0 || fd.gS1BaseSizeOfFd > maxFdRows || fd.usedVecNumOfFd == 0) {
        return MlaStatus::INVALID_PATCH;
    }
    const auto &base = tiling.baseParams;
    const auto &workspace = tiling.workspaceParams;
    const uint64_t mBaseSize = tiling.innerSplitParams.mBaseSize;
    const uint64_t queryRows = static_cast<uint64_t>(base.s1Size) * base.gSize;
    uint64_t precedingSplits = 0;
    for (uint32_t i = 0; i < fd.numOfFdHead; ++i) {
        const uint32_t splits = fd.s2SplitNumOfFdHead[i];
        const uint32_t rowSplits = fd.gS1SplitNumOfFdHead[i];
        const uint32_t tail = fd.gS1LastPartSizeOfFdHead[i];
        // The original FD loop preloads two KV parts before reducing them.
        if (fd.bN2IdxOfFdHead[i] >= base.bSize || splits < 2 || splits > launch.aicNum + 1 || rowSplits == 0 ||
            tail == 0 || tail > fd.gS1BaseSizeOfFd) {
            return MlaStatus::INVALID_PATCH;
        }
        // Each LSE buffer holds all KV parts, with eight FP32 elements per row.
        const uint32_t copyRows = rowSplits == 1 ? tail : fd.gS1BaseSizeOfFd;
        if (splits * copyRows * 8U > lseBufferElements) {
            return MlaStatus::INVALID_PATCH;
        }
        const uint64_t rows = static_cast<uint64_t>(rowSplits - 1) * fd.gS1BaseSizeOfFd + tail;
        const uint64_t outputEnd = static_cast<uint64_t>(fd.gS1IdxOfFdHead[i]) * mBaseSize + rows;
        if (rows > mBaseSize || outputEnd > queryRows) {
            return MlaStatus::INVALID_PATCH;
        }
        // CopyAccumOutIn and CopyLseIn share a prefix of fixed mBaseSize slots.
        // Check each source region independently of the total workspace cap.
        const uint64_t sourceEnd = (precedingSplits + splits - 1) * mBaseSize + rows;
        if (sourceEnd > workspace.fdAccumOutSize / 512U || sourceEnd > workspace.fdLogSumExpSize / 2U / 8U) {
            return MlaStatus::INVALID_PATCH;
        }
        precedingSplits += splits;
    }
    for (uint32_t i = 0; i < fd.usedVecNumOfFd; ++i) {
        const uint32_t task = fd.gS1IdxEndOfFdHead[i];
        const uint32_t split = fd.gS1IdxEndOfFdHeadSplit[i];
        // Both endpoints are inclusive in the original device FD loops.
        if (task >= fd.numOfFdHead || split >= fd.gS1SplitNumOfFdHead[task]) {
            return MlaStatus::INVALID_PATCH;
        }
        if (i != 0 && (task < fd.gS1IdxEndOfFdHead[i - 1] ||
                       (task == fd.gS1IdxEndOfFdHead[i - 1] && split <= fd.gS1IdxEndOfFdHeadSplit[i - 1]))) {
            return MlaStatus::INVALID_PATCH;
        }
    }
    const uint32_t lastVec = fd.usedVecNumOfFd - 1;
    const uint32_t lastTask = fd.numOfFdHead - 1;
    return fd.gS1IdxEndOfFdHead[lastVec] == lastTask &&
                   fd.gS1IdxEndOfFdHeadSplit[lastVec] == fd.gS1SplitNumOfFdHead[lastTask] - 1 ?
               MlaStatus::SUCCESS :
               MlaStatus::INVALID_PATCH;
}

__aicore__ inline MlaStatus FinalizeMlaTiling(const MlaStaticLaunchInfo &launch, const MlaMetadataHeader &metadata,
                                              MlaNonQuantTiling &tiling, uint32_t blockNum, uint64_t &workspaceLayout)
{
    auto status = ValidateMlaLaunch(launch, blockNum);
    if (status != MlaStatus::SUCCESS) {
        return status;
    }
    if (launch.family != MlaFamily::ARCH22_MLA || (metadata.dynamicFlags & ~kMlaFlagFd) != 0 ||
        metadata.maxQ > launch.qStorageSize || metadata.maxKv > launch.kvStorageSize ||
        metadata.originalBlockDim > launch.physicalBlockDim) {
        return MlaStatus::INVALID_PATCH;
    }
    const auto &base = tiling.baseParams;
    const auto &inner = tiling.innerSplitParams;
    const auto &fd = tiling.fdParams;
    const uint64_t expectedQ = launch.layoutQ == MlaLayout::TND ? metadata.maxQ : launch.qStorageSize;
    uint64_t expectedKv = launch.layoutKv == MlaLayout::TND ? metadata.maxKv : launch.kvStorageSize;
    if (launch.pageAttention != 0) {
        const uint64_t blockSize = tiling.pageAttenParams.blockSize;
        if (blockSize == 0) {
            return MlaStatus::INVALID_PATCH;
        }
        const uint64_t blocks = metadata.maxKv / blockSize + (metadata.maxKv % blockSize != 0);
        // s2Size tracks the actual KV extent, while the block-table row stride
        // remains its storage width, including unused capacity after seq shrinks.
        if (!MlaCheckedMul(blocks, blockSize, expectedKv) || launch.kvStorageSize % blockSize != 0 ||
            tiling.pageAttenParams.maxBlockNumPerBatch != launch.kvStorageSize / blockSize) {
            return MlaStatus::INVALID_PATCH;
        }
    }
    const uint64_t maskKvSize = expectedKv;
    // EmptyTensorProcess retains a 1024-element inner-size surrogate when the
    // true maximum KV is zero; Init/Process still use the original seq tensor.
    if (expectedKv == 0) {
        expectedKv = 1024;
    }
    if (base.s1Size != expectedQ || base.s2Size != expectedKv || base.n2Size != 1 || base.gSize == 0 ||
        base.headDim != 512 || base.headDimRope != 64 || base.usedCoreNum > launch.aicNum || base.needInit > 1 ||
        base.isEmptyBatchOverHalf > 1 || base.softmaxLseFlag != launch.softmaxLseFlag || inner.mBaseSize == 0 ||
        inner.s2BaseSize == 0 || fd.numOfFdHead > launch.aicNum || fd.usedVecNumOfFd > launch.aivNum) {
        return MlaStatus::INVALID_PATCH;
    }
    const bool hasFd = (metadata.dynamicFlags & kMlaFlagFd) != 0;
    if (hasFd != (fd.numOfFdHead != 0)) {
        return MlaStatus::INVALID_PATCH;
    }
    for (uint32_t i = 0; i < base.usedCoreNum; ++i) {
        if (tiling.outerSplitParams.bN2End[i] > base.bSize) {
            return MlaStatus::INVALID_PATCH;
        }
    }
    status = ValidateMlaNonQuantFd(launch, tiling);
    if (status != MlaStatus::SUCCESS) {
        return status;
    }
    status = FinalizeMlaMask(launch, metadata, base.bSize, base.s1Size, maskKvSize, tiling.maskParams.attenMaskFlag,
                             tiling.maskParams.sparseMode);
    if (status != MlaStatus::SUCCESS) {
        return status;
    }
    const uint64_t maskStride =
        launch.maskRank == 2 && base.s1Size != 1 ? launch.maskBatchStrideAlt : launch.maskBatchStride;
    if (maskStride > UINT32_MAX) {
        return MlaStatus::OVERFLOW;
    }
    tiling.maskParams.attenMaskBatchStride = static_cast<uint32_t>(maskStride);

    // Mirror every region in FiaKernelNonQuantMla::Init, in the same order.
    workspaceLayout = launch.sysWorkspaceSize;
    const uint64_t copies = static_cast<uint64_t>(blockNum) * 2;
    const auto &workspace = tiling.workspaceParams;
    if (!MlaAddWorkspaceRegion(workspace.mm1ResSize, 4, copies, workspaceLayout) ||
        !MlaAddWorkspaceRegion(workspace.mm1ResSize, 2, copies, workspaceLayout) ||
        !MlaAddWorkspaceRegion(workspace.mm2ResSize, 4, copies, workspaceLayout) ||
        !MlaAddWorkspaceRegion(workspace.mm2ResSize, 4, copies, workspaceLayout) ||
        (hasFd && (!MlaAddWorkspaceRegion(workspace.fdAccumOutSize, 4, 1, workspaceLayout) ||
                   !MlaAddWorkspaceRegion(workspace.fdLogSumExpSize, 4, 1, workspaceLayout)))) {
        return MlaStatus::OVERFLOW;
    }
    return MlaCheckWorkspace(launch, metadata, workspaceLayout);
}

__aicore__ inline MlaStatus ValidateMlaLegacyFd(const MlaStaticLaunchInfo &launch, const MlaLegacyTiling &tiling)
{
    const auto &fd = tiling.tndSplitCoreParams;
    if (fd.tndFDCoreArrLen == 0 || fd.usedVecNumOfFd == 0) {
        return MlaStatus::INVALID_PATCH;
    }
    // The balanced MLA consumer uses eight D512 FP32 rows and a 6 KiB LSE buffer.
    constexpr uint32_t fdBaseRows = 8;
    constexpr uint32_t lseBufferElements = 6U * 1024U / sizeof(float);
    const auto &base = tiling.baseParams;
    const auto &workspace = tiling.splitKVParams;
    const uint64_t mBaseSize =
        static_cast<uint64_t>(base.nNumOfQInOneGroup) * tiling.increFlashAttentionSingleCoreParams.s1SplitSize;
    const uint64_t queryRows = static_cast<uint64_t>(base.qSeqSize) * base.nNumOfQInOneGroup;
    uint64_t precedingSplits = 0;
    for (uint32_t i = 0; i < fd.tndFDCoreArrLen; ++i) {
        const uint32_t splits = fd.balanceFDCoreKVSplitArr[i];
        const uint32_t rowSplits = fd.gS1SplitNumOfFdHeadMla[i];
        const uint32_t tail = fd.gS1LastPartSizeOfFdHead[i];
        if (splits < 2 || splits > launch.aicNum || rowSplits == 0 || tail == 0 || tail > fdBaseRows) {
            return MlaStatus::INVALID_PATCH;
        }
        const uint32_t copyRows = rowSplits == 1 ? tail : fdBaseRows;
        if (splits * copyRows * 8U > lseBufferElements) {
            return MlaStatus::INVALID_PATCH;
        }
        const uint64_t rows = static_cast<uint64_t>(rowSplits - 1) * fdBaseRows + tail;
        uint64_t outputEnd;
        if (rows > mBaseSize || !MlaCheckedMul(fd.balanceFDCoreS1Arr[i], mBaseSize, outputEnd) ||
            !MlaCheckedAdd(outputEnd, rows, outputEnd) || outputEnd > queryRows) {
            return MlaStatus::INVALID_PATCH;
        }
        // Each task starts after all preceding KV parts, each with mBaseSize rows.
        uint64_t sourceEnd;
        if (!MlaCheckedMul(precedingSplits + splits - 1, mBaseSize, sourceEnd) ||
            !MlaCheckedAdd(sourceEnd, rows, sourceEnd) || sourceEnd > workspace.accumOutSize / 512U ||
            sourceEnd > workspace.logSumExpSize / 2U / 8U) {
            return MlaStatus::INVALID_PATCH;
        }
        precedingSplits += splits;
    }
    for (uint32_t i = 0; i < fd.usedVecNumOfFd; ++i) {
        const uint32_t task = fd.gS1IdxEndOfFdHead[i];
        const uint32_t split = fd.gS1IdxEndOfFdHeadSplit[i];
        // Both task and row-split endpoints are inclusive in the device loops.
        if (task >= fd.tndFDCoreArrLen || split >= fd.gS1SplitNumOfFdHeadMla[task]) {
            return MlaStatus::INVALID_PATCH;
        }
        if (i != 0 && (task < fd.gS1IdxEndOfFdHead[i - 1] ||
                       (task == fd.gS1IdxEndOfFdHead[i - 1] && split <= fd.gS1IdxEndOfFdHeadSplit[i - 1]))) {
            return MlaStatus::INVALID_PATCH;
        }
    }
    const uint32_t lastVec = fd.usedVecNumOfFd - 1;
    const uint32_t lastTask = fd.tndFDCoreArrLen - 1;
    return fd.gS1IdxEndOfFdHead[lastVec] == lastTask &&
                   fd.gS1IdxEndOfFdHeadSplit[lastVec] == fd.gS1SplitNumOfFdHeadMla[lastTask] - 1 ?
               MlaStatus::SUCCESS :
               MlaStatus::INVALID_PATCH;
}

__aicore__ inline MlaStatus FinalizeMlaTiling(const MlaStaticLaunchInfo &launch, const MlaMetadataHeader &metadata,
                                              MlaLegacyTiling &tiling, uint32_t blockNum, uint64_t &workspaceLayout)
{
    auto status = ValidateMlaLaunch(launch, blockNum);
    if (status != MlaStatus::SUCCESS) {
        return status;
    }
    if (launch.family != MlaFamily::LEGACY_MLA || (metadata.dynamicFlags & ~kMlaDynamicFlagsMask) != 0 ||
        metadata.maxQ > launch.qStorageSize || metadata.maxKv > launch.kvStorageSize ||
        metadata.originalBlockDim > launch.physicalBlockDim) {
        return MlaStatus::INVALID_PATCH;
    }
    const auto &base = tiling.baseParams;
    const auto &inner = tiling.increFlashAttentionSingleCoreParams;
    const auto &fd = tiling.tndSplitCoreParams;
    const bool hasFd = (metadata.dynamicFlags & kMlaFlagFd) != 0;
    const bool balance = (metadata.dynamicFlags & kMlaFlagBalance) != 0;
    const uint64_t expectedQ = launch.layoutQ == MlaLayout::TND ? metadata.maxQ : launch.qStorageSize;
    if (base.qSeqSize != expectedQ || base.seqSize != launch.kvStorageSize || base.blockSize == 0 ||
        static_cast<uint64_t>(base.maxBlockNumPerBatch) * base.blockSize != launch.kvStorageSize ||
        base.nNumOfQInOneGroup == 0 || inner.usedCoreNum > launch.aicNum || inner.groupSplitSize == 0 ||
        inner.s1SplitSize == 0 || inner.singleProcessSInnerSize == 0 || base.softmaxLseFlag != launch.softmaxLseFlag ||
        fd.tndFDCoreArrLen > 26 || fd.usedVecNumOfFd > launch.aivNum || fd.FdBalanceFlag > 1 ||
        (launch.layoutQ == MlaLayout::TND && !balance) || (fd.FdBalanceFlag != 0 && !balance) ||
        (balance && hasFd != (fd.tndFDCoreArrLen != 0))) {
        return MlaStatus::INVALID_PATCH;
    }
    // FdBalanceFlag selects a splitting strategy. With a small core quota it
    // can produce no cross-core FD tasks; only the resulting table selects FD.
    // Init runs on all physical cores, before Process's usedCoreNum guard. Its
    // non-balanced path reads coreSidxEnd[aiCoreIdx + 1] (the 26th sentinel).
    // Balanced Init reads the preceding batch even for an idle core.
    if (balance) {
        for (uint32_t i = 0; i + 1 < launch.aicNum; ++i) {
            if (tiling.increFlashAttentionCoreParams.coreBEnd[i] >= base.batchSize) {
                return MlaStatus::INVALID_PATCH;
            }
        }
    }
    for (uint32_t i = 0; i < fd.tndFDCoreArrLen; ++i) {
        if (fd.balanceFDCoreBArr[i] >= base.batchSize || fd.balanceFDCoreKVSplitArr[i] == 0 ||
            fd.balanceFDCoreKVSplitArr[i] > launch.aicNum) {
            return MlaStatus::INVALID_PATCH;
        }
    }
    if (hasFd && fd.FdBalanceFlag != 0) {
        status = ValidateMlaLegacyFd(launch, tiling);
        if (status != MlaStatus::SUCCESS) {
            return status;
        }
    }
    status = FinalizeMlaMask(launch, metadata, base.batchSize, base.qSeqSize, base.seqSize, base.attenMaskFlag,
                             base.sparseMode);
    if (status != MlaStatus::SUCCESS) {
        return status;
    }
    // Mirror all int8 / NORMAL-MLA regions in the legacy Init. The baseline
    // int8 instance defines QUANT_MM2_FP16: int32 MM1, int8 Vec1, half MM2
    // and half Vec2. Dispatch asserts those element widths against the class.
    workspaceLayout = launch.sysWorkspaceSize;
    const uint64_t copies = static_cast<uint64_t>(blockNum) * 2;
    const auto &workspace = tiling.increFlashAttentionSingleCoreTensorSize;
    if (!MlaAddWorkspaceRegion(workspace.mmResUbSize, 4, copies, workspaceLayout) ||
        !MlaAddWorkspaceRegion(workspace.mmResUbSize, 1, copies, workspaceLayout) ||
        !MlaAddWorkspaceRegion(workspace.bmm2ResUbSize, 2, copies, workspaceLayout) ||
        !MlaAddWorkspaceRegion(workspace.bmm2ResUbSize, 2, copies, workspaceLayout) ||
        (hasFd && (!MlaAddWorkspaceRegion(tiling.splitKVParams.accumOutSize, 4, 1, workspaceLayout) ||
                   !MlaAddWorkspaceRegion(tiling.splitKVParams.logSumExpSize, 4, 1, workspaceLayout)))) {
        return MlaStatus::OVERFLOW;
    }
    return MlaCheckWorkspace(launch, metadata, workspaceLayout);
}
} // namespace FlashMla
#endif
