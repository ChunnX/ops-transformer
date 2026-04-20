/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_tiling_impl.cpp
 * \brief
 */

#include "flash_attn_tiling_impl.h"
#include "../flash_attn_tiling_info_parser.h"

#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "../flash_attn_tiling_utils.h"
#include "../flash_attn_tiling_constants.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

// TODO
const uint32_t MASK_MODE_NO_MASK = 4;
const uint32_t MASK_MODE_ALL_MASK = 4;
const uint32_t MASK_MODE_LEFT_UP = 4;
const uint32_t MASK_MODE_RIGHT_DOWN = 4;
const uint32_t MASK_MODE_BAND = 3;
using namespace arch35FA;
ge::graphStatus FlashAttnTilingImpl::SetPlatMemoryInfo(gert::TilingContext *context, const FaTilingInfo &faInfo)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(faInfo.opName, "The platformInfoPtr is null!"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    platformInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    platformInfo_.coreNum = platformInfo_.aivNum;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, platformInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, platformInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, platformInfo_.l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, platformInfo_.l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, platformInfo_.l0bSize);
    platformInfo_.defaultSysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    OP_LOGI(faInfo.opName, "AIV:%u AIC:%u L0A:%lu L0B:%lu L0C:%lu UB:%lu L1:%lu", platformInfo_.aivNum,
            platformInfo_.aicNum, platformInfo_.l0aSize, platformInfo_.l0bSize, platformInfo_.l0cSize,
            platformInfo_.ubSize, platformInfo_.l1Size);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingImpl::SetEmptyTensor(gert::TilingContext *context, const FaTilingInfo &faInfo)
{
    auto &initOutputParams = faRunTilingAdapter_.initOutputParams;
    int64_t outSize = faInfo.opParamInfo.attnOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize = faInfo.softmaxLseFlag ? faInfo.opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    initOutputParams.set_singleCoreSize(singleCoreSize);
    initOutputParams.set_totalOutputSize(outSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(lseSize);
    initOutputParams.set_needInit(1);

    OP_CHECK_IF(GenTilingKey(context, faInfo) != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "GenTilingKey fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetBlockDim(context, faInfo) != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "SetBlockDim fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspace(context, faInfo) != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "GetWorkspace fail."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetTilingData(context, faInfo) != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "SetTilingData fail."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
bool FlashAttnTilingImpl::CheckTransposeLayout(const FaTilingInfo &faInfo)
{
    std::string layoutStr(faInfo.opParamInfo.layoutOut);
    if (layoutStr == "BNSD_BSND") {
        return true;
    }
    return false;
}

void FlashAttnTilingImpl::SetGSMerge(const FaTilingInfo &faInfo) // ？？？？？？？？？？
{
    if (faInfo.s1Size == 1) {
        gsMergeFlag_ = true;
        return;
    }
    if (faInfo.s1Size <= 16) {
        gsMergeFlag_ = true;
        return;
    }
}

void FlashAttnTilingImpl::InitImplParam(const FaTilingInfo &faInfo) // TODO 改到parse中处理，actual
{
    SetGSMerge(faInfo);
    nLoopTimes_ = (gsMergeFlag_) ? faInfo.n2Size : faInfo.n1Size;
    gsSize_ = (gsMergeFlag_) ? faInfo.gSize * faInfo.s1Size : faInfo.s1Size;

    OP_LOGI(faInfo.opName, "gsMergeFlag:%u nLoopTimes:%u gsSize:%u ", gsMergeFlag_, nLoopTimes_, gsSize_);

    headDimAlign_ = AlignUp(faInfo.qkHeadDim, NUM_32);
}

ge::graphStatus FlashAttnTilingImpl::AdjustSinnerAndSouter(gert::TilingContext *context, const FaTilingInfo &faInfo)
{
    uint32_t softmaxSOuterFactor = SOUTER_64;
    sOuterFactor_ = SOUTER_64;
    sInnerFactor_ = SINNER_128;
    OP_CHECK_IF(SetMM1TilingData(context, faInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(faInfo.opName, "set bmm1 tiling data fail."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetMM2TilingData(context, faInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(faInfo.opName, "set bmm2 tiling data fail."), return ge::GRAPH_FAILED);

    if (faInfo.vHeadDim <= DSIZE_128) {
        bool checkDtype = true;
        bool checkQueryAndValueS = faInfo.s1Size <= SOUTER_64 && faInfo.s2Size > SINNER_128;
        uint32_t maskMode = faInfo.maskMode;
        int32_t winLefts = faInfo.winLeft;
        int32_t winRights = faInfo.winRight;
        if (maskMode == 0) {
            winLefts = (winLefts > 0) ? 0 : winLefts;
        } else if (maskMode == 4) {
            winRights = (winRights > 0) ? 0 : winRights;
        }
        bool checkMaskMode = (maskMode != 2 && winLefts + winRights > 128);
        if (checkDtype && checkQueryAndValueS && checkMaskMode) {
            sOuterFactor_ = SOUTER_32;
            sInnerFactor_ = SINNER_256;
            softmaxSOuterFactor = SOUTER_32;
        } else if (((faInfo.qLayout == FaLayout::BSND) ||
                    (faInfo.qLayout == FaLayout::TND)) &&
                   pfaMergeFlag_) {
            sOuterFactor_ = SOUTER_32;
            sInnerFactor_ = SINNER_256;
        }
    } else if (faInfo.vHeadDim > DSIZE_128 && faInfo.s1Size != 1) {
        if (((faInfo.qLayout == FaLayout::BSND) ||
             (faInfo.qLayout == FaLayout::TND)) &&
            pfaMergeFlag_ && faInfo.vHeadDim <= DSIZE_256) { // 256 : D size
            sOuterFactor_ = SOUTER_32;
            sInnerFactor_ = SINNER_256;
        } else {
            sOuterFactor_ = SOUTER_64;
            sInnerFactor_ = SINNER_128;
        }
        softmaxSOuterFactor = SOUTER_32;
    } else if (
               (faInfo.s1Size == 1 && faInfo.vHeadDim > DSIZE_128)) { // IFA VD > 128
        if (false) {
            sOuterFactor_ = SOUTER_32;
        } else {
            sOuterFactor_ = SOUTER_64;
        }
        sInnerFactor_ = SINNER_128;
        softmaxSOuterFactor = SOUTER_64;
    }

    OP_LOGI(faInfo.opName, "Souter:%u SInner:%u softmaxSOuterFactor %u", sOuterFactor_, sInnerFactor_,
            softmaxSOuterFactor);

    return ge::GRAPH_SUCCESS;
}
// void FlashAttnTilingImpl::GetPrewinRightsLeftUp(const FaTilingInfo &faInfo, int64_t actualSeqLength,
//                                                  int64_t actualSeqLengthKV, int64_t &winLeftsLeftUp,
//                                                  int64_t &winRightsLeftUp)
// {
//     if (faInfo.maskMode == MASK_MODE_RIGHT_DOWN) {
//         winLeftsLeftUp = -1;
//         if (faInfo.vHeadDim == 512) {
//             if (faInfo.qLayout == FaLayout::BSND ||
//                 faInfo.qLayout == FaLayout::TND) {
//                 winRightsLeftUp = actualSeqLengthKV * faInfo.gSize - actualSeqLength;
//             } else { // BNSD场景下分核不做优化
//                 winRightsLeftUp = -1;
//             }
//             actualSeqLenQFlag_ = true;
//             actualSeqLenKVFlag_ = true;
//         } else {
//             winRightsLeftUp = actualSeqLengthKV - actualSeqLength;
//         }
//     } else if (faInfo.maskMode == MASK_MODE_BAND) {
//         winLeftsLeftUp = faInfo.winLeft - actualSeqLengthKV + actualSeqLength;
//         winRightsLeftUp = faInfo.winRight + actualSeqLengthKV - actualSeqLength;
//     } else {
//         winLeftsLeftUp = faInfo.winLeft;
//         winRightsLeftUp = faInfo.winRight;
//     }
// }

void FlashAttnTilingImpl::FixParamWithRowInvalid(const FaTilingInfo &faInfo, int64_t &actualSeqLength,
                                                 int64_t actualSeqLengthKV, int64_t &winLeftsLeftUp,
                                                 int64_t &winRightsLeftUp)
{
    // 若出现行无效，需要重新计算winRights，winLefts，actualseqlen，以便正确计算分核核数
    int64_t winRightsError = (winRightsLeftUp < 0) ? -winRightsLeftUp : 0;
    winRightsError = winRightsError > actualSeqLength ? actualSeqLength : winRightsError;
    int64_t winLeftsError = 0;
    if (false) {
        winLeftsError = (actualSeqLength > actualSeqLengthKV * faInfo.gSize + winLeftsLeftUp) ?
                             (actualSeqLength - actualSeqLengthKV * faInfo.gSize - winLeftsLeftUp) :
                             0;
    } else {
        winLeftsError = (actualSeqLength > actualSeqLengthKV + winLeftsLeftUp) ?
                             (actualSeqLength - actualSeqLengthKV - winLeftsLeftUp) :
                             0;
    }
    winLeftsError = winLeftsError > actualSeqLength ? actualSeqLength : winLeftsError;

    // 若出现上方行无效，需要重新计算winRights，winLefts，actualseqlen
    winRightsLeftUp += winRightsError;
    winLeftsLeftUp -= winRightsError;
    actualSeqLength -= winRightsError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= winLeftsError;
}

int64_t FlashAttnTilingImpl::GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner,
                                             int64_t sOuter, int64_t token)
{
    int64_t blockNums = 0;
    int64_t blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
    int64_t outDivIn = sOuter > sInner ? sOuter / sInner : 1;
    int64_t inDivOut = sInner > sOuter ? sInner / sOuter : 1;
    int64_t tolerance = 0;
    int64_t smallSize = 0;
    if (outDivIn >= 1) {
        tolerance = outDivIn;
        smallSize = sInner;
    } else {
        tolerance = inDivOut;
        smallSize = sOuter;
    }
    int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
    int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
    int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength - blockToken) / smallSize - tolerance;
    blockNums += (innerCutBlockNums > 0) ?
                     (innerCutBlockNums % tolerance + innerCutBlockNums) * (innerCutBlockNums / tolerance + 1) / 2 :
                     0; // 2: The denominator of the arithmetic sequence summation formula
    blockNums -= (innerCutBlockLeftNums > 0) ? (innerCutBlockLeftNums % tolerance + innerCutBlockLeftNums) *
                                                   (innerCutBlockLeftNums / tolerance + 1) / 2 :
                                               0; // 2: The denominator of the arithmetic sequence summation formula
    blockNums -= (innerCutBlockDownNums > 0) ? (innerCutBlockDownNums % tolerance + innerCutBlockDownNums) *
                                                   (innerCutBlockDownNums / tolerance + 1) / 2 :
                                               0; // 2: The denominator of the arithmetic sequence summation formula
    return blockNums;
}

int64_t FlashAttnTilingImpl::GetCalcBlockNumsOneHead(const FaTilingInfo &faInfo, int64_t actualSeqLength,
                                                     int64_t actualSeqLengthKV, uint32_t sOuterSize,
                                                     uint32_t sInnerSize, int64_t winLeftsLeftUp,
                                                     int64_t winRightsLeftUp)
{
    return 0;
    // if (!faInfo.attnMaskFlag) {
    //     int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
    //     int64_t innerBlockNums =
    //         (actualSeqLengthKV + sInnerSize - 1) / sInnerSize + (faInfo.systemPrefixLen + sInnerSize - 1) / sInnerSize;
    //     int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
    //     return toCalcBlockNums;
    // } else {
    //     int64_t innerBlockNums =
    //         (actualSeqLengthKV + static_cast<int64_t>(sInnerSize) - 1) / static_cast<int64_t>(sInnerSize) +
    //         (faInfo.systemPrefixLen + static_cast<int64_t>(sInnerSize) - 1) / static_cast<int64_t>(sInnerSize);
    //     int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(sInnerSize);
    //     int64_t outerBlockNums =
    //         (actualSeqLength + static_cast<int64_t>(sOuterSize) - 1) / static_cast<int64_t>(sOuterSize);
    //     int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(sOuterSize);
    //     int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
    //     // Must meet this condition : winLeft + winRight > 0
    //     toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
    //                                        static_cast<int64_t>(sOuterSize), winRightsLeftUp);
    //     toCalcBlockNums -=
    //         GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
    //                         static_cast<int64_t>(sOuterSize), blockSeqLengthKV - blockSeqLength + winLeftsLeftUp);
    //     return toCalcBlockNums;
    // }
}

int64_t FlashAttnTilingImpl::GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
                                                int64_t innerBlockNums)
{
    int64_t sInnerBlockNums = 0;
    if (sInnerIndexEnd < 0) {
        sInnerBlockNums = 0;
    } else if (sInnerIndexEnd < innerBlockNums) {
        sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    } else {
        sInnerBlockNums = (sInnerIndexStart < 0) ?
                              innerBlockNums :
                              (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
    }
    return sInnerBlockNums;
}

void FlashAttnTilingImpl::ComputeSplitNBSeq(const FaTilingInfo &faInfo, const size_t maxCoreNums, uint32_t sOuterSize,
                                            uint32_t sInnerSize, double coreWeightTarget, uint32_t &curCore)
{
    std::vector<uint32_t> coreSposEnd(maxCoreNums, 0U);
    std::vector<uint32_t> coreSposStart(maxCoreNums, 0U);
    std::vector<uint32_t> coreSidEnd(maxCoreNums, 0U);
    std::vector<uint32_t> coreSidStart(maxCoreNums, 0U);
    std::vector<uint32_t> coreNidEnd(maxCoreNums, 0U);
    std::vector<uint32_t> coreNidStart(maxCoreNums, 0U);
    std::vector<int64_t> maskStartIdx(maxCoreNums, 0L);
    std::vector<uint32_t> bnStartIdx(maxCoreNums, 0U);
    std::vector<int64_t> gS1StartIdx(maxCoreNums, 0L);
}

void FlashAttnTilingImpl::SplitNBSeq(const FaTilingInfo &faInfo)
{
}

bool FlashAttnTilingImpl::CheckFlashDecode(const FaTilingInfo &faInfo)
{
    float flashDecodeBNRatio = 0.4F; // 0.4, 经验值
    if (faInfo.s1Size != 1) {
        return false;
    }
    if (faInfo.s2Size < NUM_256) {
        return false;
    }
    uint64_t loopTimes = faInfo.bSize * faInfo.n2Size * (faInfo.gSize + sOuterFactor_ - 1) / sOuterFactor_;
    if (loopTimes < flashDecodeBNRatio * platformInfo_.aicNum && faInfo.gSize == 1) {
        return true;
    }
    if (loopTimes < flashDecodeBNRatio * platformInfo_.aicNum &&
        (faInfo.s2Size >= NUM_2048)) { // 2048, 在flash decode + gqa时的经验值
        return true;
    }
    return false;
}

ge::graphStatus FlashAttnTilingImpl::SplitS2(const FaTilingInfo &faInfo)
{
    uint64_t batchSize = faInfo.bSize;
    uint64_t headNumSize = faInfo.n1Size;
    uint32_t kvSplitLimit =
        sInnerFactor_ <= 256U ?
            256U :
            sInnerFactor_; // 256: 经验值 这里的门限值也需要调整，否则会分的很碎，每个核分到的还不到一个基本块
    uint64_t loopTimes = faInfo.bSize * faInfo.n2Size * (faInfo.gSize + sOuterFactor_ - 1) / sOuterFactor_;
    int64_t kvSplitPart = platformInfo_.aicNum / loopTimes;
    while (((faInfo.s2Size / kvSplitPart) < kvSplitLimit) && (kvSplitPart > 1)) {
        kvSplitPart--;
    }

    faRunTilingAdapter_.inputParamsRegbase.set_kvSplitPart(kvSplitPart);
    faRunTilingAdapter_.inputParamsRegbase.set_accumOutSize(batchSize * headNumSize * kvSplitPart * headDimAlign_);
    faRunTilingAdapter_.inputParamsRegbase.set_logSumExpSize(batchSize * headNumSize * kvSplitPart *
                                                             (BYTE_BLOCK / sizeof(float)));

    return ge::GRAPH_SUCCESS;
}

// 伪量化
void FlashAttnTilingImpl::SetDequantBaseSize(const FaTilingInfo &faInfo)
{
    sOuterFactor_ = NUM_16;
    if (faInfo.s1Size > 1 && gsMergeFlag_) { // pfa gs1合轴时 s1base=32
        sOuterFactor_ = NUM_32;
    }
    if (faInfo.qkHeadDim <= NUM_64) {
        sInnerFactor_ = NUM_1024;
        if (faInfo.s1Size > 1 && gsMergeFlag_) { // pfa gs1合轴 s1base=32 s2base=512 dbase=64
            sInnerFactor_ = NUM_512;
        }
    } else if (faInfo.qkHeadDim <= NUM_128) {
        sInnerFactor_ = NUM_512;
    } else if (faInfo.qkHeadDim <= NUM_256) {
        sInnerFactor_ = NUM_256;
    } else {
        sInnerFactor_ = NUM_128;
    }
}

ge::graphStatus FlashAttnTilingImpl::CalcInnerSize(const FaTilingInfo &faInfo, uint32_t seqSize)
{
    /**
     * sInnerFactor：s2的切分大小，直接决定了MM的singleN/K和vector的切块大小，但当前切分也并非适用所有case。
     * GQA场景-伪量化：vector比较重，尽量较少vector的循环次数,
     * 因此，cube发小块，期望vector尽量被cube的mte2掩盖。sInnerSize=1024
     */
    SetDequantBaseSize(faInfo);
    sInnerFactorSize_ = sInnerFactor_;
    if (sInnerFactor_ > seqSize) {
        sInnerFactor_ = seqSize;
    }
    sInnerSizeAlign_ = AlignUp(sInnerFactor_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    return ge::GRAPH_SUCCESS;
}

int64_t FlashAttnTilingImpl::GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
                                                     int64_t innerBlockNums)
{
    int64_t sInnerBlockNums = 0;
    if (sInnerIndexEnd < 0) {
        sInnerBlockNums = 0;
    } else if (sInnerIndexEnd < innerBlockNums) {
        sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    } else {
        sInnerBlockNums = (sInnerIndexStart < 0) ?
                              innerBlockNums :
                              (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
    }

    return sInnerBlockNums;
}

// void FlashAttnTilingImpl::GetAntiQuantPrewinRightsLeftUp(const FaTilingInfo &faInfo, int64_t actualSeqLength,
//                                                           int64_t actualSeqLengthKV, int64_t &winLeftsLeftUp,
//                                                           int64_t &winRightsLeftUp)
// {
//     winLeftsLeftUp = faInfo.winLeft;
//     winRightsLeftUp = faInfo.winRight;
//     if (faInfo.maskMode == MASK_MODE_RIGHT_DOWN) {
//         winRightsLeftUp = actualSeqLengthKV - actualSeqLength;
//     } else if (faInfo.maskMode == MASK_MODE_BAND) {
//         winLeftsLeftUp = faInfo.winLeft - actualSeqLengthKV + actualSeqLength;
//         winRightsLeftUp = faInfo.winRight + actualSeqLengthKV - actualSeqLength;
//     }
// }

// void FlashAttnTilingImpl::FixAntiQuantParamWithRowInvalid(const FaTilingInfo &faInfo, int64_t &actualSeqLength,
//                                                           int64_t actualSeqLengthKV, int64_t &winLeftsLeftUp,
//                                                           int64_t &winRightsLeftUp)
// {
//     // 若出现行无效，需要重新计算winRights，winLefts，actualseqlen，以便正确计算分核核数
//     int64_t winRightsError = (winRightsLeftUp < 0) ? -winRightsLeftUp : 0;
//     winRightsError = winRightsError > actualSeqLength ? actualSeqLength : winRightsError;

//     int64_t winLeftsError = (actualSeqLength > actualSeqLengthKV + winLeftsLeftUp) ?
//                                  (actualSeqLength - actualSeqLengthKV - winLeftsLeftUp) :
//                                  0;
//     winLeftsError = winLeftsError > actualSeqLength ? actualSeqLength : winLeftsError;

//     // 若出现上方行无效，需要重新计算winRights，winLefts，actualseqlen
//     winRightsLeftUp += winRightsError;
//     winLeftsLeftUp -= winRightsError;
//     actualSeqLength -= winRightsError;

//     // 若出现下方行无效，需要重新计算actualseqlen
//     actualSeqLength -= winLeftsError;
//     if (winRightsError != 0 || winLeftsError != 0) {
//         needInit_ = true;
//     }
// }


void FlashAttnTilingImpl::ComputeDequantSplitNBSeq(const FaTilingInfo &faInfo, std::vector<int64_t> sOuterLoopTimes,
                                                   std::vector<int64_t> sInnerLoopTimes, int64_t sInnerLoopTimesPrefix,
                                                   double coreWeightTarget, uint32_t &curCore,
                                                   const size_t tilingElementArrayLen)
{
    // std::vector<int64_t> maskStartIdx(tilingElementArrayLen, 0L);
    // std::vector<uint32_t> bnStartIdx(tilingElementArrayLen, 0U);
    // std::vector<int64_t> gS1StartIdx(tilingElementArrayLen, 0L);
    // // Temporary algorithm to be optimized
    // int64_t curWeight = 0;
    // uint32_t tmpCoreNidEnd = 0; // actual seq为0时不分配核
    // uint32_t tmpCoreSidEnd = 0;
    // uint32_t tmpCoreSposEnd = 0;
    // int64_t actualSeqLengthsTmp = 0;
    // int64_t actualSeqLengthsKVTmp = 0;
    // for (uint32_t bIdx = 0; bIdx < faInfo.bSize; bIdx++) {
    //     GetActualSeqLength(faInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp, bIdx);
    //     for (uint32_t headNum = 0; headNum < nLoopTimes_; headNum++) {
    //         int64_t winLeftsLeftUp = 0;
    //         int64_t winRightsLeftUp = 0;
    //         // GetAntiQuantPrewinRightsLeftUp(faInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp + faInfo.systemPrefixLen,
    //         //                                 winLeftsLeftUp, winRightsLeftUp);
    //         // FixAntiQuantParamWithRowInvalid(faInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp + faInfo.systemPrefixLen,
    //         //                                 winLeftsLeftUp, winRightsLeftUp);
    //         int64_t outerBlockNums = sOuterLoopTimes[bIdx];
    //         int64_t innerBlockNums = sInnerLoopTimes[bIdx];
    //         for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
    //             int64_t dif = static_cast<int64_t>(coreWeightTarget * double(curCore + 1)) - curWeight;
    //             // 非prefix部分计算，去除prefix影响
    //             int64_t winLeftsNoPrefix = winLeftsLeftUp + faInfo.systemPrefixLen;
    //             int64_t winRightsNoPrefix = winRightsLeftUp - faInfo.systemPrefixLen;
    //             int64_t sInnerIndexStart =
    //                 -(winLeftsNoPrefix > 0 ? (winLeftsNoPrefix + static_cast<int64_t>(sInnerFactor_) - 1) /
    //                                               static_cast<int64_t>(sInnerFactor_) :
    //                                           winLeftsNoPrefix / static_cast<int64_t>(sInnerFactor_));
    //             int64_t sInnerIndexEnd = winRightsNoPrefix > 0 ?
    //                                          (winRightsNoPrefix + static_cast<int64_t>(sInnerFactor_) - 1) /
    //                                              static_cast<int64_t>(sInnerFactor_) :
    //                                          winRightsNoPrefix / static_cast<int64_t>(sInnerFactor_);

    //             // prefix部分单独计算
    //             int64_t sInnerIndexStartPrefix =
    //                 -(winLeftsLeftUp > 0 ? (winLeftsLeftUp + static_cast<int64_t>(sInnerFactor_) - 1) /
    //                                             static_cast<int64_t>(sInnerFactor_) :
    //                                         winLeftsLeftUp / static_cast<int64_t>(sInnerFactor_));
    //             int64_t sInnerIndexEndPrefix = winRightsLeftUp > 0 ?
    //                                                (winRightsLeftUp + static_cast<int64_t>(sInnerFactor_) - 1) /
    //                                                    static_cast<int64_t>(sInnerFactor_) :
    //                                                winRightsLeftUp / static_cast<int64_t>(sInnerFactor_);

    //             // 当前这一行有多少基本块需要计算
    //             int64_t actualInnerBlockNums =
    //                 GetActualInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums) +
    //                 GetActualInnerBlockNums(sInnerIndexStartPrefix, sInnerIndexEndPrefix, sInnerLoopTimesPrefix);
    //             if (actualInnerBlockNums - dif > dif &&
    //                 !(tmpCoreNidEnd == 0 && tmpCoreSidEnd == 0 && tmpCoreSposEnd == 0)) {
    //                 curCore += 1;
    //                 bnStartIdx[curCore] = bIdx * nLoopTimes_ + headNum;
    //                 gS1StartIdx[curCore] = sOuterIndex;
    //             }
    //             tmpCoreNidEnd = headNum + 1;
    //             tmpCoreSidEnd = bIdx + 1;
    //             tmpCoreSposEnd = sOuterIndex + 1;

    //             curWeight += actualInnerBlockNums;
    //             winLeftsLeftUp -= sOuterFactor_;
    //             winRightsLeftUp += sOuterFactor_;
    //         }
    //     }
    // }
    // bnStartIdx[curCore + 1] = faInfo.bSize * nLoopTimes_;
    // gS1StartIdx[curCore + 1] = tmpCoreSposEnd;

    // faRunTilingAdapter_.multiCoreParamsRegbase.set_bnStartIdx(bnStartIdx.data());
    // faRunTilingAdapter_.multiCoreParamsRegbase.set_maskStartIdx(gS1StartIdx.data());
}

void FlashAttnTilingImpl::GetActualSeqLength(const FaTilingInfo &faInfo, int64_t &actualSeqLengths,
                                             int64_t &actualSeqLengthsKV, uint32_t bIdx)
{
    // uint32_t nNumOfQInOneGroup = 1;
    // nNumOfQInOneGroup = *(faInfo.opParamInfo.numHeads) / *(faInfo.opParamInfo.kvHeadNums);
    // if (faInfo.qLayout == FaLayout::TND) {
    //     actualSeqLengths = bIdx == 0 ? faInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[0] :
    //                                    faInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx] -
    //                                        faInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx - 1];
    //     if (gsMergeFlag_) {
    //         actualSeqLengths *= nNumOfQInOneGroup;
    //     }
    //     actualSeqLengthsKV = faInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[bIdx];
    //     if (!faInfo.pageAttentionFlag && bIdx > 0) {
    //         actualSeqLengthsKV -= faInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[bIdx - 1];
    //     }
    // } else {
    //     if (faInfo.actualSeqLenFlag && faInfo.actualLenDims > 0 &&
    //         faInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>() != nullptr) { // kvLengths
    //         actualSeqLengthsKV = faInfo.actualLenDims == NUM1 ?
    //                                  faInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[0] :
    //                                  faInfo.opParamInfo.actualSeqLengths.tensor->GetData<int64_t>()[bIdx];
    //     } else {
    //         actualSeqLengthsKV =
    //             faInfo.kvListSeqLens.size() == NUM1 ? faInfo.kvListSeqLens[0] : faInfo.kvListSeqLens[bIdx];
    //     }
    //     if (actualSeqLengthsKV < faInfo.s2Size) {
    //         needInit_ = true;
    //     }
    //     if (actualSeqLenQFlag_) { // qLengths
    //         actualSeqLengths = faInfo.actualLenDims == NUM1 ?
    //                                faInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[0] :
    //                                faInfo.opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>()[bIdx];
    //     } else {
    //         actualSeqLengths = faInfo.s1Size;
    //         if (gsMergeFlag_) {
    //             actualSeqLengths = faInfo.s1Size * nNumOfQInOneGroup;
    //         }
    //     }
    // }
}

int64_t FlashAttnTilingImpl::GetAntiQuantCalcBlockNumsOneHead(const FaTilingInfo &faInfo, int64_t outerBlockNums,
                                                              int64_t innerBlockNums, int64_t sInnerLoopTimesPrefix,
                                                              int64_t winLeftsLeftUp, int64_t winRightsLeftUp)
{
    // if (!faInfo.attnMaskFlag) {
    //     return (innerBlockNums + sInnerLoopTimesPrefix) * outerBlockNums;
    // } else {
    //     int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(sOuterFactor_);
    //     int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(sInnerFactor_);
    //     int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
    //     // 必须满足winLeft + winRight > 0，否则会减出小于0的块数，这里需要去除prefix影响
    //     toCalcBlockNums -=
    //         GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerFactor_),
    //                         static_cast<int64_t>(sOuterFactor_), winRightsLeftUp - faInfo.systemPrefixLen);
    //     toCalcBlockNums -= GetCutBlockNums(
    //         blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerFactor_), static_cast<int64_t>(sOuterFactor_),
    //         blockSeqLengthKV - blockSeqLength + winLeftsLeftUp + faInfo.systemPrefixLen);

    //     // prefix部分单独计算
    //     int64_t blockSharedPrefix = sInnerLoopTimesPrefix * static_cast<int64_t>(sInnerFactor_);
    //     toCalcBlockNums += sInnerLoopTimesPrefix * outerBlockNums;
    //     toCalcBlockNums -= GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(sInnerFactor_),
    //                                        static_cast<int64_t>(sOuterFactor_), winRightsLeftUp);
    //     toCalcBlockNums -=
    //         GetCutBlockNums(blockSharedPrefix, blockSeqLength, static_cast<int64_t>(sInnerFactor_),
    //                         static_cast<int64_t>(sOuterFactor_), blockSharedPrefix - blockSeqLength + winLeftsLeftUp);
    //     return toCalcBlockNums;
    // }
}

void FlashAttnTilingImpl::DequantCubeSplitBNSeq(const FaTilingInfo &faInfo) // 伪量化只用Cube视角分核
{
    // uint32_t curCoreNum = platformInfo_.aicNum;
    // int64_t totalBlockNumsOneHead = 0;
    // std::vector<int64_t> sOuterLoopTimes(faInfo.bSize, 0U);
    // std::vector<int64_t> sInnerLoopTimes(faInfo.bSize, 0U);
    // int64_t multiSmaxsInnerLoopTimes = 0U;
    // int64_t sInnerLoopTimesPrefix =
    //     (faInfo.systemPrefixLen + static_cast<int64_t>(sInnerFactor_) - 1) / static_cast<int64_t>(sInnerFactor_);
    // for (uint32_t bIdx = 0; bIdx < faInfo.bSize; bIdx++) {
    //     int64_t actualSeqLengthsTmp = 0;
    //     int64_t actualSeqLengthsKVTmp = 0;
    //     GetActualSeqLength(faInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp, bIdx);
    //     int64_t winLeftsLeftUp = 0;
    //     int64_t winRightsLeftUp = 0;
    //     // GetAntiQuantPrewinRightsLeftUp(faInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp + faInfo.systemPrefixLen,
    //     //                                 winLeftsLeftUp, winRightsLeftUp);
    //     // FixAntiQuantParamWithRowInvalid(faInfo, actualSeqLengthsTmp, actualSeqLengthsKVTmp + faInfo.systemPrefixLen,
    //     //                                 winLeftsLeftUp, winRightsLeftUp);

    //     sOuterLoopTimes[bIdx] =
    //         (actualSeqLengthsTmp + static_cast<int64_t>(sOuterFactor_) - 1) / static_cast<int64_t>(sOuterFactor_);
    //     sInnerLoopTimes[bIdx] =
    //         (actualSeqLengthsKVTmp + static_cast<int64_t>(sInnerFactor_) - 1) / static_cast<int64_t>(sInnerFactor_);
    //     multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[bIdx] + sInnerLoopTimesPrefix);

    //     totalBlockNumsOneHead +=
    //         GetAntiQuantCalcBlockNumsOneHead(faInfo, sOuterLoopTimes[bIdx], sInnerLoopTimes[bIdx],
    //                                          sInnerLoopTimesPrefix, winLeftsLeftUp, winRightsLeftUp);
    // }
    // double coreWeightTarget = (double(totalBlockNumsOneHead * nLoopTimes_) / double(curCoreNum));

    // int64_t s1OuterSize = (faInfo.s1Size + sOuterFactor_ - 1) / sOuterFactor_;
    // faRunTilingAdapter_.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);
    // const size_t tilingElementArrayLen =
    //     (static_cast<size_t>(curCoreNum) > 64UL) ? static_cast<size_t>(curCoreNum) : 64UL;
    // uint32_t curIndx = 0;
    // ComputeDequantSplitNBSeq(faInfo, sOuterLoopTimes, sInnerLoopTimes, sInnerLoopTimesPrefix, coreWeightTarget, curIndx,
    //                          tilingElementArrayLen);
    // int64_t sinnerBlocknum = (faInfo.maxActualseq + sInnerFactor_ - 1) / sInnerFactor_;
    // int64_t totalSize = (totalBlockNumsOneHead / sinnerBlocknum) * nLoopTimes_;
    // int64_t actualUsedCoreNum = static_cast<int64_t>((curIndx + 1));

    // faRunTilingAdapter_.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    // faRunTilingAdapter_.multiCoreParamsRegbase.set_totalSize(totalSize);
    // faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    // faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorTailSize(
    //     CalcTailSize(totalSize, faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorSize()));
}

void FlashAttnTilingImpl::SplitDequant(const FaTilingInfo &faInfo)
{
    // CalcInnerSize(faInfo, faInfo.maxActualseq);
    DequantCubeSplitBNSeq(faInfo);
    if (!isPFAFlag_) {
        if (CheckFlashDecode(faInfo)) {
            flashDecodeFlag_ = true;
            SplitS2(faInfo);
        }
    }
}

bool FlashAttnTilingImpl::CheckEnableDN(const FaTilingInfo &faInfo)
{
    constexpr uint32_t dLimitDN = DSIZE_128;
    constexpr uint32_t sOuterLimitDN = SOUTER_64;
    bool res = !faInfo.attnMaskFlag && !faInfo.pageAttentionFlag &&
               faInfo.qkHeadDim <= dLimitDN && faInfo.vHeadDim <= dLimitDN &&
               sOuterFactor_ * CV_RATIO > sOuterLimitDN;
    return res;
}

bool FlashAttnTilingImpl::CheckQKVActualSeqLengthsRight(const FaTilingInfo &faInfo)
{
    for (uint32_t bIdx = 0; bIdx < faInfo.bSize; bIdx++) {
        if ((actualSeqLengthsQ_[bIdx] % SOUTER_32 > 0) || (actualSeqLengthsKV_[bIdx] <= SINNER_128)) {
            return false;
        }
    }
    return true;
}

ge::graphStatus FlashAttnTilingImpl::SplitPolicy(gert::TilingContext *context, const FaTilingInfo &faInfo)
{
    OP_CHECK_IF(AdjustSinnerAndSouter(context, faInfo) != ge::GRAPH_SUCCESS, // 确认souter/sinner
                OP_LOGE(faInfo.opName, "Get sinner and souter fail!"), return ge::GRAPH_FAILED);

    dnFlag_ = CheckEnableDN(faInfo);
    isQKVActualSeqLengthsRightFlag_ = CheckQKVActualSeqLengthsRight(faInfo);

    if (dnFlag_ && isQKVActualSeqLengthsRightFlag_ && faInfo.qkHeadDim == faInfo.vHeadDim &&
        faInfo.qkHeadDim == DSIZE_64) {
        sInnerFactor_ = SINNER_256;
    }

    SplitNBSeq(faInfo);
    if (CheckFlashDecode(faInfo)) {
        flashDecodeFlag_ = true;
        OP_LOGI(faInfo.opName, "FlashDecode is enable.");
        SplitS2(faInfo);
    }
    return ge::GRAPH_SUCCESS;
}

void FlashAttnTilingImpl::UpdateTilingKeyConfig(const FaTilingInfo &faInfo)
{
    // Config: 0=正常计算, 1=空tensor场景
    tilingKeyInfo_.kernelTypeKey = faInfo.emptyTensorFlag;
}

void FlashAttnTilingImpl::UpdateTilingKeyLayout(const FaTilingInfo &faInfo)
{
    // Layout: None(BSND) = 0, LAYOUT_BSH = 1, LAYOUT_SBH = 2, LAYOUT_BNSD = 3, LAYOUT_TND = 4, (LAYOUT_NTD_TND = 5,
    // LAYOUT_NTD = 6)
    std::string layoutStr(faInfo.opParamInfo.layoutOut);
    if (layoutStr == "BSND") {
        tilingKeyInfo_.inputLayout = 0;
    } else if (layoutStr == "BSH") {
        tilingKeyInfo_.inputLayout = 1;
    } else if (layoutStr == "SBH") {
        tilingKeyInfo_.inputLayout = 2;
    } else if (layoutStr == "BNSD") {
        tilingKeyInfo_.inputLayout = 3;
    } else if (layoutStr == "TND") {
        tilingKeyInfo_.inputLayout = 4;
    } else {
        tilingKeyInfo_.inputLayout = 0;
    }
}

// S1TemplateType支持: 0, 64, 128
uint64_t UpdateTilingKeyS1Template(int64_t actualS1)
{
    if (actualS1 <= 64)
        return 64;
    return 128; // 默认返回128
}

// S2TemplateType支持: 0, 64, 128, 256
uint64_t UpdateTilingKeyS2Template(int64_t actualS2)
{
    if (actualS2 <= 64)
        return 64;
    if (actualS2 <= 128)
        return 128;
    return 128; // 默认返回256
}

// DTemplateType支持: 0, 64, 128, 192, 256, 768
uint64_t UpdateTilingKeyDTemplate(int64_t actualD)
{
    if (actualD <= 64)
        return 64;
    if (actualD <= 128)
        return 128;
    if (actualD <= 192)
        return 192;
    if (actualD <= 256)
        return 256;
    return 64; // 默认返回768
}

// DvTemplateType支持: 0, 64, 128, 192, 256 (0表示与D相同)
uint64_t UpdateTilingKeyDvTemplate(int64_t actualDv, int64_t actualD)
{
    if (actualDv == actualD)
        return 0; // dv与d相同时返回0
    if (actualDv <= 64)
        return 64;
    if (actualDv <= 128)
        return 128;
    if (actualDv <= 192)
        return 192;
    return 0; // 默认返回256
}

//==========================================================================================//

void FlashAttnTilingImpl::UpdateTilingKeyMaskMode(const FaTilingInfo &faInfo)
{
    // 是否有attention mask
    // tilingKeyInfo_.hasAttenMask = faInfo.attnMaskFlag;
    // tilingKeyInfo_.maskMode = (faInfo.opParamInfo.maskMode != nullptr) ? *faInfo.opParamInfo.maskMode : 0; // TOUSE:
    // 应该从faInfo获取
}

void FlashAttnTilingImpl::UpdateTilingKeyMatmulMode(const FaTilingInfo &faInfo)
{
    // tilingKeyInfo_.matmulMode = 0; // TODO: 写死为0，可能需要动态设置
}
//==========================================================================================//


ge::graphStatus FlashAttnTilingImpl::UpdateTilingKeyInfo(const FaTilingInfo &faInfo)
{
    UpdateTilingKeyConfig(faInfo);
    UpdateTilingKeyLayout(faInfo);
    UpdateTilingKeyMaskMode(faInfo);
    UpdateTilingKeyMatmulMode(faInfo);

    // 设置其他标志
    tilingKeyInfo_.isPa = faInfo.pageAttentionFlag;
    // tilingKeyInfo_.isFd = false;  // TODO: FlashDecode标志，写死为false，应该根据条件动态设置
    // tilingKeyInfo_.enableKvPrefix = faInfo.sysPrefixFlag;

    tilingKeyInfo_.s1TemplateType = UpdateTilingKeyS1Template(faInfo.s1Size);
    tilingKeyInfo_.s2TemplateType = UpdateTilingKeyS2Template(faInfo.s2Size);
    tilingKeyInfo_.dTemplateType = UpdateTilingKeyDTemplate(faInfo.qkHeadDim);
    tilingKeyInfo_.dvTemplateType = UpdateTilingKeyDvTemplate(faInfo.vHeadDim, faInfo.qkHeadDim);
    tilingKeyInfo_.hasAttn = faInfo.attnMaskFlag;
    tilingKeyInfo_.isSoftmaxLse = faInfo.softmaxLseFlag;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingImpl::GenTilingKey(gert::TilingContext *context, const FaTilingInfo &faInfo)
{
    UpdateTilingKeyInfo(faInfo);
    uint64_t genTilingkey = GET_TPL_TILING_KEY(tilingKeyInfo_.kernelTypeKey,        // KernelTypeKey
                                               tilingKeyInfo_.implMode,             // ImplMode  // 固定值
                                               tilingKeyInfo_.inputLayout,          // Layout
                                               tilingKeyInfo_.s1TemplateType,       // S1TemplateType (模板值)
                                               tilingKeyInfo_.s2TemplateType,       // S2TemplateType (模板值)
                                               tilingKeyInfo_.dTemplateType,        // DTemplateType (模板值)
                                               tilingKeyInfo_.dvTemplateType,       // DvTemplateType (模板值)
                                               tilingKeyInfo_.hasAttn ? 1 : 0,      // HasAtten
                                               tilingKeyInfo_.isPa ? 1 : 0,         // IsPA
                                               tilingKeyInfo_.isSoftmaxLse ? 1 : 0, // IsSoftmaxLse
                                               1                                    // Regbase  //固定值
    );

    context->SetTilingKey(genTilingkey);
    OP_LOGI(faInfo.opName, "The tilingkey is %llu.", genTilingkey);
    OP_LOGI(faInfo.opName,
            "The tilingkey param: kernelTypeKey=%llu, implMode=%llu, layout=%llu, "
            "s1Template=%llu, s2Template=%llu, dTemplate=%llu, dvTemplate=%llu, hasAtten=%llu, isPA=%llu, "
            "isSoftmaxLse=%llu, regbase=%llu.",
            tilingKeyInfo_.kernelTypeKey ? 1 : 0, tilingKeyInfo_.implMode, tilingKeyInfo_.inputLayout,
            tilingKeyInfo_.s1TemplateType, tilingKeyInfo_.s2TemplateType, tilingKeyInfo_.dTemplateType,
            tilingKeyInfo_.dvTemplateType, tilingKeyInfo_.hasAttn ? 1 : 0, tilingKeyInfo_.isPa ? 1 : 0,
            tilingKeyInfo_.isSoftmaxLse ? 1 : 0, 1ULL);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingImpl::SetBlockDim(gert::TilingContext *context, const FaTilingInfo &faInfo)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    blockDim_ = ascendcPlatform.CalcTschBlockDim(platformInfo_.aivNum, platformInfo_.aicNum, platformInfo_.aivNum);
    context->SetBlockDim(blockDim_);

    OP_LOGI(faInfo.opName, "blockDim: %d", blockDim_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingImpl::SetWorkspaceNormal(const FaTilingInfo &faInfo,
                                                        int64_t &curWorkspaceSize) // TODO 新patch
{
    size_t sysWorkspaceSize = platformInfo_.defaultSysWorkspaceSize;
    size_t accumOutSize = 0;
    size_t logSumExpSize = 0;
    if (flashDecodeFlag_) {
        auto batchSize = faInfo.bSize;
        auto headNumSize = nLoopTimes_;
        auto vHeadSize = faInfo.vHeadDim;
        uint32_t kvSplitPart = faRunTilingAdapter_.inputParamsRegbase.get_kvSplitPart();
        accumOutSize = batchSize * faInfo.gSize * headNumSize * kvSplitPart * vHeadSize * sizeof(float);
        logSumExpSize = batchSize * faInfo.gSize * headNumSize * kvSplitPart * BYTE_BLOCK * 2;
    }

    int64_t bmm2Bytes = 0;
    int64_t vec2Bytes = 0;
    uint32_t dSize = faInfo.vHeadDim;
    uint32_t dVBasicBlock = 0;
    if (dSize <= DSIZE_64) {
        dVBasicBlock = DSIZE_64;
    } else if (dSize <= DSIZE_128) {
        dVBasicBlock = DSIZE_128;
    } else if (dSize <= DSIZE_256) {
        dVBasicBlock = DSIZE_256;
    } else if (dSize <= DSIZE_512) {
        dVBasicBlock = DSIZE_512;
    }
    int64_t bmm2ResBlockSize = dVBasicBlock;
    if (dVBasicBlock > DSIZE_256) {
        bmm2ResBlockSize = DSIZE_512;
    }
    uint32_t s1BasicBlock = sOuterFactor_ * CV_RATIO;
    uint32_t calcTypeSize = sizeof(float);
    if ((!dnFlag_ && dSize > DSIZE_128) || (dnFlag_ && dSize > DSIZE_192)) {
        bmm2Bytes = s1BasicBlock * bmm2ResBlockSize * calcTypeSize;
        if (dVBasicBlock > DSIZE_256) {
            vec2Bytes = s1BasicBlock * dVBasicBlock * calcTypeSize;
        }
    }
    curWorkspaceSize = (bmm2Bytes + vec2Bytes) * 3 * platformInfo_.coreNum + // 3: perload 2次 需要2+1
                       sysWorkspaceSize + accumOutSize + logSumExpSize;

    if (faInfo.pageAttentionFlag) {
        // 2 bmm, db, ensure alignment of each structure 64B, dcci cacheline needs
        curWorkspaceSize += static_cast<uint64_t>(platformInfo_.coreNum) * 2 * 2 * 64;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingImpl::GetWorkspace(gert::TilingContext *context, const FaTilingInfo &faInfo)
{
    int64_t workspace = 0;
    if (faInfo.emptyTensorFlag) {
        workspace = platformInfo_.defaultSysWorkspaceSize;
    } else {
        OP_CHECK_IF(SetWorkspaceNormal(faInfo, workspace) != ge::GRAPH_SUCCESS,
                    OP_LOGE(faInfo.opName, "Get workspace failed ."), return ge::GRAPH_FAILED);
    }

    OP_LOGI(faInfo.opName, "Workspaces: %ld", workspace);
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = workspace;
    return ge::GRAPH_SUCCESS;
}

bool FlashAttnTilingImpl::EnableMTE2BmmPipe(const FaTilingInfo &faInfo, matmul_tiling::MatmulApiTiling &bmm,
                                            TCubeTiling &bmmTilingData)
{
    if (faInfo.s1Size > 16) {
        return true;
    }
    uint32_t baseK = 32U;
    uint32_t headSize = faInfo.qkHeadDim;
    if (headSize % baseK != 0) {
        return true;
    }

    uint32_t baseM = std::min(uint32_t(128), sOuterFactor_);
    uint32_t baseN = std::min(uint32_t(512), sInnerFactor_);
    if (faInfo.pageAttentionFlag) {
        baseN = 128; // 128
    }
    bool res = (bmm.SetFixSplit(baseM, baseN, baseK) == ge::GRAPH_SUCCESS);
    OP_CHECK_IF(!res, OP_LOGE(faInfo.opName, "set fix split fail"), return ge::GRAPH_FAILED);
    // check
    res = bmm.GetTiling(bmmTilingData) != -1;
    return res;
}
void FlashAttnTilingImpl::GetMatMulType(const FaTilingInfo &faInfo, matmul_tiling::DataType &mmInputType,
                                        matmul_tiling::DataType &mmOutputType)
{
    if (faInfo.inputQType == ge::DT_FLOAT16) {
        mmInputType = matmul_tiling::DataType::DT_FLOAT16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (faInfo.inputQType == ge::DT_BF16) {
        mmInputType = matmul_tiling::DataType::DT_BF16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (faInfo.inputQType == ge::DT_INT8) {
        mmInputType = matmul_tiling::DataType::DT_INT8;
        mmOutputType = matmul_tiling::DataType::DT_INT32;
    }
}

ge::graphStatus FlashAttnTilingImpl::SetMM1TilingData(gert::TilingContext *context, const FaTilingInfo &faInfo)
{
    /*
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);

    matmul_tiling::DataType bmm1InputType = matmul_tiling::DataType::DT_INT8;
    matmul_tiling::DataType bmm1OutputType = matmul_tiling::DataType::DT_INT32;
    GetMatMulType(faInfo, bmm1InputType, bmm1OutputType);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, bmm1OutputType);

    auto qkHeadDim = faInfo.qkHeadDim;
    if (faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512) {
        qkHeadDim = faInfo.qkHeadDim + faInfo.ropeHeadDim;
    }
    int32_t ret = bmm1.SetShape(sOuterFactor_, sInnerFactor_, qkHeadDim);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "Bmm1 set shape space failed"),
                return ge::GRAPH_FAILED);
    if ((faInfo.qLayout == FaLayout::BSH) || (faInfo.qLayout == FaLayout::BSND) ||
        (faInfo.qLayout == FaLayout::TND)) {
        if (faInfo.qLayout == FaLayout::TND && faInfo.pageAttentionFlag && faInfo.kvLayout == FaLayout::BnNBsD) {
            bmm1.SetOrgShape(gsSize_, faInfo.s2Size, qkHeadDim * nLoopTimes_, qkHeadDim);
        } else if (faInfo.mlaMode == MlaMode::ROPE_SPLIT_D512 || faInfo.s1Size == 1) {
            bmm1.SetOrgShape(gsSize_, faInfo.s2Size, qkHeadDim, faInfo.n2Size * qkHeadDim);
        } else {
            bmm1.SetOrgShape(gsSize_, faInfo.s2Size, qkHeadDim * nLoopTimes_,
                             faInfo.n2Size * qkHeadDim);
        }
    } else if (faInfo.qLayout == FaLayout::BNSD) {
        if (faInfo.pageAttentionFlag &&
            faInfo.kvLayout == FaLayout::BnBsH) {  // The left matrix of PA is BNSD, and the right matrix is BSH.
            bmm1.SetOrgShape(gsSize_, faInfo.s2Size, qkHeadDim, faInfo.n2Size * qkHeadDim);
        } else {
            bmm1.SetOrgShape(gsSize_, faInfo.s2Size, qkHeadDim);
        }
    }
    bmm1.SetBias(false);
    ret = bmm1.SetBufferSpace(platformInfo_.l1Size, platformInfo_.l0cSize);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "Bmm1 set buffer space failed"),
                return ge::GRAPH_FAILED);
    if (faInfo.ropeMode != RopeMode::ROPE_SPLIT && faInfo.pageAttentionFlag) {
        ret = bmm1.SetFixSplit(sOuterFactor_, SINNER_128);
    } else {
        ret = bmm1.SetFixSplit(sOuterFactor_, sInnerFactor_);
    }
    TCubeTiling &bmm1TilingData = pfaTilingData_.bmm1TilingDataRect;

    ret = bmm1.GetTiling(bmm1TilingData);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "Bmm1 get tiling failed, ret:%d", ret),
                return ge::GRAPH_FAILED);
    uint32_t baseN = std::min(SINNER_128, sInnerFactor_);
    if (faInfo.pageAttentionFlag) {
        baseN = SINNER_128;
    }
    if (ret != ge::GRAPH_SUCCESS) {
        ret = bmm1.SetFixSplit(sOuterFactor_, baseN, 64U);  // check
        ret = bmm1.GetTiling(bmm1TilingData);
    }
    // check ret
    bmm1TilingData.set_shareMode(0);
    bmm1TilingData.set_shareL1Size(platformInfo_.l1Size);
    bmm1TilingData.set_shareL0CSize(platformInfo_.l0cSize);
    bmm1TilingData.set_shareUbSize(0);

    if ((bmm1TilingData.get_depthA1() == 1) && (bmm1TilingData.get_depthB1() == 1)) {
        bmm1TilingData.set_depthA1(DOUBLE_BUFFER_NUM);
        bmm1TilingData.set_depthB1(DOUBLE_BUFFER_NUM);
    }

    OP_CHECK_IF(!EnableMTE2BmmPipe(faInfo, bmm1, bmm1TilingData), OP_LOGE(faInfo.opName, "Bmm1 mte2 bmm pipe fail"),
                return ge::GRAPH_FAILED);
    */
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingImpl::SetMM2TilingData(gert::TilingContext *context, const FaTilingInfo &faInfo)
{
    /*
    auto platformInfoPtr = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);

    matmul_tiling::DataType bmm2InputType = matmul_tiling::DataType::DT_INT8;
    matmul_tiling::DataType bmm2OutputType = matmul_tiling::DataType::DT_INT32;
    bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
    bmm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, bmm2OutputType);

    int32_t ret = bmm2.SetShape(sOuterFactor_, faInfo.vHeadDim, sInnerFactor_);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "Bmm2 set shape failed"), return ge::GRAPH_FAILED);

    if ((faInfo.qLayout == FaLayout::BSH) || (faInfo.qLayout == FaLayout::BSND) ||
        (faInfo.qLayout == FaLayout::TND)) {
        if (faInfo.qLayout == FaLayout::TND && faInfo.pageAttentionFlag && faInfo.kvLayout == FaLayout::BnNBsD) {
            bmm2.SetOrgShape(gsSize_, faInfo.vHeadDim, faInfo.s2Size);
        } else {
            bmm2.SetOrgShape(gsSize_, faInfo.vHeadDim * faInfo.n2Size, faInfo.s2Size);
        }
    } else if (faInfo.qLayout == FaLayout::BNSD) {
        if (faInfo.pageAttentionFlag && faInfo.kvLayout == FaLayout::BnBsH) {
            bmm2.SetOrgShape(gsSize_, faInfo.vHeadDim * faInfo.n2Size, faInfo.s2Size);
        } else {
            bmm2.SetOrgShape(gsSize_, faInfo.qkHeadDim, faInfo.s2Size);
        }
    }
    bmm2.SetBias(false);
    bmm2.SetBufferSpace(platformInfo_.l1Size, platformInfo_.l0cSize);
    ret = bmm2.GetTiling(pfaTilingData_.bmm2TilingDataRect);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "Bmm2 get tiling failed"), return ge::GRAPH_FAILED);

    pfaTilingData_.bmm2TilingDataRect.set_shareMode(0);
    pfaTilingData_.bmm2TilingDataRect.set_shareL1Size(platformInfo_.l1Size);
    pfaTilingData_.bmm2TilingDataRect.set_shareL0CSize(platformInfo_.l0cSize);
    pfaTilingData_.bmm2TilingDataRect.set_shareUbSize(0);
    */
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingImpl::SetFullQuantTilingData(const FaTilingInfo &faInfo)
{
    /*
    auto &baseParams = pfaTilingData_.promptAttentionBaseParams;
    baseParams.set_batchSize(faInfo.bSize);
    baseParams.set_headNumSize(faInfo.n2Size);
    baseParams.set_seqSize(gsSize_);
    baseParams.set_headSize(faInfo.qkHeadDim);
    baseParams.set_softmaxScale(faInfo.softmaxScale);
    baseParams.set_winLefts(faInfo.winLeft);
    baseParams.set_winRights(faInfo.winRight);
    baseParams.set_blockSize(faInfo.blockSize);
    uint32_t blockTableDim2 = 0;
    if (faInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        blockTableDim2 = faInfo.opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1);
    }
    baseParams.set_blockTableDim2(blockTableDim2);
    baseParams.set_PABlockNumSum(faInfo.totalBlockNum);
    baseParams.set_dimNumOfseq(faInfo.bSize);
    baseParams.set_seqInnerSize(faInfo.s2Size);
    baseParams.set_prefixSeqInnerSize(faInfo.systemPrefixLen);

    baseParams.set_useMask(faInfo.attnMaskFlag);
    baseParams.set_headNumRatio(gsMergeFlag_ ? 1 : faInfo.gSize);
    baseParams.set_typeByteNum(BYTE_BLOCK);
    baseParams.set_attenMaskElemType(faInfo.opParamInfo.attnMask.desc != nullptr ?
                                         faInfo.opParamInfo.attnMask.desc->GetDataType() :
                                         faInfo.inputQType);


    baseParams.set_maskTypeByteNum(BYTE_BLOCK);
    baseParams.set_outputTypeByteNum(faInfo.isOutQuantEnable ? BYTE_BLOCK : BYTE_BLOCK / (FLOAT16SIZE / INT8SIZE));
    baseParams.set_softmaxTypeByteNum(BYTE_BLOCK / sizeof(float));
    baseParams.set_sparseMode(faInfo.maskMode);
    baseParams.set_alignedHeadSize(AlignUp(faInfo.qkHeadDim, BYTE_BLOCK));
    baseParams.set_splitS2(0);
    baseParams.set_splitD(0);

    baseParams.set_isLayoutSH(0);
    baseParams.set_isActualSeqLengthsNull(!actualSeqLenQFlag_);
    baseParams.set_isActualSeqLengthsKVNull(!actualSeqLenKVFlag_);
    baseParams.set_actualSeqLengthsSize(faInfo.actualLenQDims);
    baseParams.set_actualSeqLengthsKVSize(faInfo.actualLenDims);


    baseParams.set_isRowInvalid((faInfo.innerPrecise >> 1) & 1);
    baseParams.set_softmaxOuterSize(sOuterFactor_);

    baseParams.set_isKvContinuous(faInfo.kvStorageMode != KvStorageMode::TENSOR_LIST);
    baseParams.set_fromFused(!fromPFA_);
    baseParams.set_isBSNDOut(faInfo.qLayout == FaLayout::BNSD && faInfo.outLayout == FaLayout::BSND);
    baseParams.set_isIFA(gsMergeFlag_);
    baseParams.set_isSoftMaxLseEnable(faInfo.softmaxLseFlag);
    baseParams.set_isActualSharedPrefixLenNull(!actualSharedPrefixLenFlag_);
    baseParams.set_isQHasLeftPadding(faInfo.qPaddingSizeFlag);
    baseParams.set_isKVHasLeftPadding(faInfo.kvPaddingSizeFlag);


    baseParams.set_isMsd(0);

    baseParams.set_ropeHeadSize(faInfo.ropeHeadDim);
    baseParams.set_qkHeadSize(faInfo.qkHeadDim);
    baseParams.set_vHeadSize(faInfo.vHeadDim);
    baseParams.set_gOfMla(gsMergeFlag_ ? faInfo.gSize : 1);

    auto &initOutputParams = pfaTilingData_.promptAttentionInitOutputParams;
    int64_t outSize = faInfo.opParamInfo.attnOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize = faInfo.softmaxLseFlag ? faInfo.opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    if (faInfo.isOutQuantEnable) {
        singleCoreSize = ((singleCoreSize + 1) / 2) * 2;
    }
    initOutputParams.set_singleCoreSize(singleCoreSize);
    initOutputParams.set_totalOutputSize(outSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(lseSize);
    initOutputParams.set_needInit(faInfo.needInit || needInit_);


    auto &singleCoreParams = pfaTilingData_.promptAttentionSingleCoreParams;
    singleCoreParams.set_singleProcessSInnerSize(sInnerFactor_);
    singleCoreParams.set_singleProcessSOuterSize(sOuterFactor_);

    singleCoreParams.set_kvAntiquantSInnerSize(0);
    return ge::GRAPH_SUCCESS;
    */
}

bool FlashAttnTilingImpl::GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType)
{
    static struct {
        ge::DataType a;
        matmul_tiling::DataType b;
    } typeTrans[] = {{ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
                     {ge::DT_BF16, matmul_tiling::DataType::DT_BF16},
                     {ge::DT_INT8, matmul_tiling::DataType::DT_INT8},
                     {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
                     {ge::DT_INT4, matmul_tiling::DataType::DT_INT8},
                     {ge::DT_HIFLOAT8, matmul_tiling::DataType::DT_INT8},
                     {ge::DT_FLOAT8_E4M3FN, matmul_tiling::DataType::DT_INT8},
                     {ge::DT_FLOAT4_E2M1, matmul_tiling::DataType::DT_INT8}};

    for (uint32_t i = 0; i < sizeof(typeTrans) / sizeof(typeTrans[0]); i++) {
        if (typeTrans[i].a == getype) {
            *mmType = typeTrans[i].b;
            return true;
        }
    }
    return false;
}

void FlashAttnTilingImpl::AdjustPABmm1Tiling(const FaTilingInfo &faInfo, uint32_t &bmm1BaseN)
{
    if (bmm1BaseN < static_cast<uint32_t>(faInfo.blockSize)) {
        while (static_cast<uint32_t>(faInfo.blockSize) % bmm1BaseN != 0) {
            bmm1BaseN /=
                2; // 2:不断减半，确保1个base块不会跨block拷贝。已校验过blockSize 16/32对齐，因此bmm1BaseN最小值为16/32
        }
    } else if (bmm1BaseN > static_cast<uint32_t>(faInfo.blockSize)) {
        // nd2nz拷贝时ndnum>1场景性能较差，通过设置baseN <= blocksize避免
        uint32_t tmpBaseN = increGcd(bmm1BaseN, static_cast<uint32_t>(faInfo.blockSize));
        bmm1BaseN = tmpBaseN;
    }
    OP_LOGD(faInfo.opName, "Page attention is enabled, blockSize is %d, bmm1 baseN is adjusted to %u.",
            faInfo.blockSize, bmm1BaseN);
}

void FlashAttnTilingImpl::AdjustPABmm2Tiling(const FaTilingInfo &faInfo)
{
    // uint32_t targetBaseK = 128;
    // if (targetBaseK < static_cast<uint32_t>(faInfo.blockSize)) {
    //     while ((faInfo.blockSize % targetBaseK != 0) ||
    //            (targetBaseK * ifaTilingData_.bmm2TilingData.get_baseN() * sizeof(float) > 64 * 1024)) {
    //         targetBaseK /=
    //             2;  //
    //             2:不断减半，确保1个base块不会跨block拷贝，已校验过blockSize_16/32对齐，因此targetBaseK最小值为16/32
    //     }
    // } else {
    //     uint32_t tmpBaseK = increGcd(targetBaseK, (uint32_t)faInfo.blockSize);
    //     while (tmpBaseK * ifaTilingData_.bmm2TilingData.get_baseN() * sizeof(float) > 64 * 1024) {
    //         tmpBaseK /= 2;  // 2: 不断减半，确保base块大小在LOB有效范围内
    //     }
    //     targetBaseK = tmpBaseK;
    // }
    // // mm api不支持通过 SetFixSplit 设置baseK，需要直接配置tiling结构体
    // ifaTilingData_.bmm2TilingData.set_baseK(targetBaseK);
    // OP_LOGD(faInfo.opName, "Page attention is enabled, blockSize is %u, bmm2 baseK is adjusted to %u.",
    //         faInfo.blockSize, targetBaseK);
}

ge::graphStatus FlashAttnTilingImpl::SetDequantMMTilingData(gert::TilingContext *context, const FaTilingInfo &faInfo)
{
    // auto platformInfoPtr = context->GetPlatformInfo();
    // auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    // matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
    // matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);

    // matmul_tiling::DataType qType;
    // matmul_tiling::DataType kvType;

    // OP_CHECK_IF((!GetMatmulType(faInfo.inputQType, &qType) || !GetMatmulType(faInfo.inputKvType, &kvType)),
    //             OP_LOGE(faInfo.opName, "Get matmul type error."), return ge::GRAPH_FAILED);

    // uint32_t baseN = 512;  // antiquant to split K;
    // uint32_t singleM;
    // if (isPFAFlag_) {
    //     singleM = sOuterFactor_;
    // } else {
    //     singleM = faInfo.gSize;
    // }
    // bmm1.SetShape(singleM, sInnerFactor_, faInfo.qkHeadDim);
    // bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, false);
    // bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, kvType, true);
    // bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, false);
    // bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, true);
    // bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
    // matmul_tiling::DataType::DT_FLOAT); bmm1.SetOrgShape(singleM, sInnerFactor_, faInfo.qkHeadDim, headDimAlign_);
    // bmm1.SetBias(false);

    // uint32_t bmm1BaseN = std::min(AlignUp(sInnerFactor_, NUM_16), baseN);
    // if (faInfo.pageAttentionFlag) {
    //     AdjustPABmm1Tiling(faInfo, bmm1BaseN);
    // }
    // // 向下对齐保证M*N不超过L0C，且由于bmm1BaseN有最大限制，L0C_SIZE / sizeof(float) / bmm1BaseN不会小于16
    // uint32_t bmm1MaxBaseM =
    //     AlignUp(static_cast<uint32_t>(platformInfo_.l0cSize / sizeof(float) / bmm1BaseN) - NUM_16, NUM_16);

    // OP_CHECK_IF((bmm1.SetFixSplit(std::min(AlignUp(singleM, NUM_16), bmm1MaxBaseM), AlignUp(bmm1BaseN, NUM_16)) ==
    // -1),
    //             OP_LOGE(faInfo.opName, "Bmm1 SetFixSplit fail."), return ge::GRAPH_FAILED);
    // OP_CHECK_IF((bmm1.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN) == -1),
    //             OP_LOGE(faInfo.opName, "Bmm1 SetTraverse fail."), return ge::GRAPH_FAILED);
    // OP_CHECK_IF((bmm1.GetTiling(ifaTilingData_.bmm1TilingData) == -1), OP_LOGE(faInfo.opName, "Bmm1 get tiling
    // fail."),
    //             return ge::GRAPH_FAILED);

    // bmm2.SetShape(singleM, faInfo.qkHeadDim, sInnerFactor_);
    // bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, false);
    // bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, qType, false);
    // bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND_ALIGN,
    // matmul_tiling::DataType::DT_FLOAT); bmm2.SetOrgShape(singleM, headDimAlign_, sInnerSizeAlign_, sInnerFactor_);
    // // 存在输入query是BNSD格式，但使能PA，需要按BSH SetOrgShape
    // bmm2.SetBias(false);
    // OP_CHECK_IF((bmm2.SetFixSplit(std::min(AlignUp(singleM, NUM_16), NUM_128)) == -1),
    //             OP_LOGE(faInfo.opName, "Bmm2 SetFixSplit fail."), return ge::GRAPH_FAILED);
    // OP_CHECK_IF((bmm2.GetTiling(ifaTilingData_.bmm2TilingData) == -1), OP_LOGE(faInfo.opName, "Bmm2 get tiling
    // fail."),
    //             return ge::GRAPH_FAILED);
    // if (faInfo.pageAttentionFlag) {
    //     AdjustPABmm2Tiling(faInfo);
    // }

    // return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingImpl::ComputeTilingData(const FaTilingInfo &faInfo)
{
    // 处理不能直接从faInfo赋值到tiling data
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    // auto &baseParams = pfaTilingData_.promptAttentionBaseParams;
    // auto &singleCoreParams = pfaTilingData_.promptAttentionSingleCoreParams;

    /*
     *  mask mode 相关tiling data
     */
    if (faInfo.attnMaskFlag) {
        uint64_t maskBatch = 1;
        uint64_t maskDimNum = faInfo.opParamInfo.attnMask.tensor->GetStorageShape().GetDimNum();
        uint64_t maskS1Size = 2048;
        uint64_t maskS2Size = 2048;
        if (maskDimNum != 2 || faInfo.s1Size == 1) {
            maskBatch = faInfo.opParamInfo.attnMask.tensor->GetStorageShape().GetDim(0);
        }
        inputParams.set_attenMaskShapeType(maskBatch > 1 ? 1 : 2);
        // singleCoreParams.set_attenMaskBatch(maskBatch);
        maskS2Size = faInfo.opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 1);
        maskS1Size = faInfo.opParamInfo.attnMask.tensor->GetStorageShape().GetDim(maskDimNum - 2);
        inputParams.set_attenMaskS1Size(maskS1Size);
        inputParams.set_attenMaskS2Size(maskS2Size);
        // baseParams.set_maskQsSize(maskS1Size);
        // baseParams.set_maskKVsSize(maskS2Size);
    } else {
        inputParams.set_attenMaskS1Size(0);
        inputParams.set_attenMaskS2Size(0);
        inputParams.set_attenMaskShapeType(0);
        // baseParams.set_maskQsSize(0);
        // baseParams.set_maskKVsSize(0);
    }

    static std::map<uint32_t, uint8_t> maskToCompressModeMap = {{MASK_MODE_NO_MASK, 0},
                                                                {MASK_MODE_ALL_MASK, 0},
                                                                {MASK_MODE_LEFT_UP, 1},
                                                                {MASK_MODE_RIGHT_DOWN, 2},
                                                                {MASK_MODE_BAND, 3}};
    auto itr = maskToCompressModeMap.find(faInfo.maskMode);
    if (itr == maskToCompressModeMap.end()) {
        inputParams.set_attenMaskCompressMode(0);
    } else {
        inputParams.set_attenMaskCompressMode(itr->second);
    }

    // if (!isPFAFlag_) {
    //     uint8_t maskType = 0;
    //     inputParams.set_sparseType(maskType);
    // } else {
    uint8_t maskType = 0;
    if (faInfo.maskMode == MASK_MODE_NO_MASK) {
        if (faInfo.winLeft >= faInfo.s1Size && faInfo.winRight == 0) {
            maskType = 3;
        } else if (faInfo.winLeft >= faInfo.s1Size && !faInfo.pageAttentionFlag && faInfo.winRight >= faInfo.s2Size) {
            maskType = 0;
        } else {
            maskType = 4;
        }
    } else if (faInfo.maskMode == MASK_MODE_ALL_MASK) {
        maskType = 0;
    } else if (faInfo.maskMode == MASK_MODE_LEFT_UP) {
        maskType = 3;
    } else if (faInfo.maskMode == MASK_MODE_RIGHT_DOWN) {
        if (!faInfo.pageAttentionFlag && faInfo.s1Size == faInfo.s2Size) {
            maskType = 3;
        } else {
            maskType = 4;
        }
    } else if (faInfo.maskMode == MASK_MODE_BAND) {
        maskType = 4;
    }
    inputParams.set_maskType(maskType);
    // }

    /*
     *  alibi 相关tiling data
     */
    // int64_t qStartIdx = 0;
    // int64_t kvStartIdx = 0;
    // auto qStartIdxTensor = nullptr;
    // auto kvStartIdxTensor = nullptr;
    // ;
    // if (qStartIdxTensor != nullptr && qStartIdxTensor->GetShapeSize() >= 1) {
    //     const int64_t *value = qStartIdxTensor->GetData<int64_t>();
    //     if (value != nullptr) {
    //         qStartIdx = value[0];
    //     }
    // }

    // if (kvStartIdxTensor != nullptr && kvStartIdxTensor->GetShapeSize() >= 1) {
    //     const int64_t *value = kvStartIdxTensor->GetData<int64_t>();
    //     if (value != nullptr) {
    //         kvStartIdx = value[0];
    //     }
    // }
    // inputParams.set_qStartIdx(qStartIdx);
    // inputParams.set_kvStartIdx(kvStartIdx);

    /*
     *  layout 相关tiling data
     */
    static std::map<FaLayout, uint8_t> layoutStrToLayoutTypeMap = {
        {FaLayout::TND, 4},
        {FaLayout::BSND, 1},
        {FaLayout::BNSD, 3},
    };
    // auto iter = layoutStrToLayoutTypeMap.find(faInfo.qLayout);
    // if (iter == layoutStrToLayoutTypeMap.end()) {
    //     inputParams.set_layoutType(0);
    //     // baseParams.set_layoutType(0);
    // } else {
    //     inputParams.set_layoutType(iter->second);
    //     // baseParams.set_layoutType(iter->second);
    // }

    if (faInfo.pageAttentionFlag) {
        uint32_t keyCacheDimNum = faInfo.opParamInfo.key.shape->GetStorageShape().GetDimNum();
        if (keyCacheDimNum == 3) {
            inputParams.set_paLayoutType(1);
        } else if (keyCacheDimNum == 4) {
            inputParams.set_paLayoutType(0);
        } else if (keyCacheDimNum == 5) {
            inputParams.set_paLayoutType(2);
        }
    }

    const std::map<std::string, uint32_t> transposeLayoutMp = {{"BNSD_BSND", 1}, {"BSND_BNSD", 2}, {"BSH_BNSD", 3},
                                                               {"BNSD_NBSD", 4}, {"BSND_NBSD", 5}, {"BSH_NBSD", 6}};
    string layout(faInfo.opParamInfo.layoutOut);
    if (transposeLayoutMp.find(layout) != transposeLayoutMp.end()) {
        inputParams.set_transposeLayout(transposeLayoutMp.at(layout));
    } else {
        inputParams.set_transposeLayout(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingImpl::SetFATilingData(const FaTilingInfo &faInfo)
{
    auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
    inputParams.set_bSize(faInfo.bSize);
    inputParams.set_t1Size(faInfo.qTSize);
    inputParams.set_t2Size(faInfo.kTSize);
    inputParams.set_n2Size(faInfo.n2Size);
    inputParams.set_gSize(faInfo.gSize);
    inputParams.set_s1Size(faInfo.s1Size);
    inputParams.set_s2Size(faInfo.s2Size);
    inputParams.set_dSize(faInfo.qkHeadDim);
    inputParams.set_dSizeV(faInfo.vHeadDim);
    inputParams.set_softmaxScale(faInfo.softmaxScale);
    // inputParams.set_winLefts(faInfo.winLeft);
    // inputParams.set_winRights(faInfo.winRight);
    // inputParams.set_actualSeqLengthsSize(faInfo.actualLenQDims);
    // inputParams.set_actualSeqLengthsKVSize(faInfo.actualLenDims);
    inputParams.set_isKvContinuous(1);
    inputParams.set_fromFused(!fromPFA_);
    inputParams.set_isBSNDOut(faInfo.qLayout == FaLayout::BNSD && faInfo.outLayout == FaLayout::BSND);
    // inputParams.set_isGqa(gsMergeFlag_ && faInfo.ropeMode != RopeMode::ROPE_SPLIT); // TOUSE: 原始逻辑
    inputParams.set_isGqa(gsMergeFlag_);
    inputParams.set_isSoftMaxLseEnable(faInfo.softmaxLseFlag);
    inputParams.set_headNumRatio(gsMergeFlag_ ? 1 : faInfo.gSize);
    inputParams.set_blockSize(faInfo.blockSize);
    // uint32_t blockTableDim2 = 0; // TOUSE: 原始动态计算逻辑
    // if (faInfo.kvStorageMode == KvStorageMode::PAGE_ATTENTION) { // TOUSE
    //     blockTableDim2 = faInfo.opParamInfo.blockTable.tensor->GetStorageShape().GetDim(1); // TOUSE
    // } // TOUSE
    // inputParams.set_blockTableDim2(blockTableDim2); // TOUSE: 应该动态计算
    inputParams.set_blockTableDim2(faInfo.maxBlockNumPerBatch);
    inputParams.set_paBlockNumSum(faInfo.totalBlockNum);
    // inputParams.set_isRowInvalid((faInfo.innerPrecise >> 1) & 1);
    inputParams.set_isActualSeqLengthsNull(!actualSeqLenQFlag_);
    inputParams.set_isActualSeqLengthsKVNull(!actualSeqLenKVFlag_);
    inputParams.set_isActualSharedPrefixLenNull(!actualSharedPrefixLenFlag_);
    inputParams.set_keepProb(0); // TODO: 默认值，dropout场景需要从faInfo获取
    // mask相关参数设置
    inputParams.set_bandIndex(0);         // band index, currently always 0
    inputParams.set_attenMaskDataType(1); // mask数据类型，默认1
    inputParams.set_s1MaskValidSize(0);   // s1 mask有效大小，默认0
    inputParams.set_s2MaskValidSize(0);   // s2 mask有效大小，默认0

    // 从 faInfo 读取参数用于多核调度计算
    int64_t bSize = faInfo.bSize;
    int64_t n2Size = faInfo.n2Size;
    int64_t gSize = faInfo.gSize;
    int64_t s1Size = faInfo.s1Size;
    int64_t s2Size = faInfo.s2Size;
    int64_t dSize = faInfo.qkHeadDim;
    int64_t dSizeV = faInfo.vHeadDim;

    // 设置对齐的 s2 大小
    inputParams.set_alignedS2(AlignUp(s2Size, static_cast<int64_t>(16)));
    // 设置布局类型：BNSD = 3
    inputParams.set_layoutType(static_cast<uint8_t>(faInfo.qLayout)); // inputParams 中layout是uint8, 枚举值是uint32

    int64_t dBasicBlock = this->tilingKeyInfo_.dTemplateType;
    int64_t dVBasicBlock = this->tilingKeyInfo_.dvTemplateType == 0 ? dBasicBlock : this->tilingKeyInfo_.dvTemplateType;
    int64_t s1BasicBlock = this->tilingKeyInfo_.s1TemplateType;
    int64_t s2BasicBlock = this->tilingKeyInfo_.s2TemplateType;
    int64_t effectiveS1 = s1Size;
    int64_t s1Outer = CeilDivision(effectiveS1, s1BasicBlock);
    faRunTilingAdapter_.multiCoreParamsRegbase.set_s1OuterSize(s1Outer);

    int64_t totalSz = bSize * n2Size * gSize * faRunTilingAdapter_.multiCoreParamsRegbase.get_s1OuterSize();

    int32_t usedCoreNum = static_cast<int32_t>(std::min(totalSz, static_cast<int64_t>(platformInfo_.aicNum)));
    faRunTilingAdapter_.multiCoreParamsRegbase.set_coreNum(usedCoreNum);
    faRunTilingAdapter_.multiCoreParamsRegbase.set_totalSize(totalSz);

    int64_t splitFactor = CeilDivision(totalSz, static_cast<int64_t>(usedCoreNum));
    int64_t splitFactorTail = totalSz - (static_cast<int64_t>(usedCoreNum) - 1LL) * splitFactor;
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorSize(splitFactor);
    faRunTilingAdapter_.multiCoreParamsRegbase.set_splitFactorTailSize(splitFactorTail);

    uint32_t bnStartIdxArr[48] = {};
    for (int32_t i = 0; i < usedCoreNum && i < 48; i++) {
        bnStartIdxArr[i] = static_cast<uint32_t>(static_cast<int64_t>(i) * splitFactor);
    }
    faRunTilingAdapter_.multiCoreParamsRegbase.set_bnStartIdx(bnStartIdxArr);
    // 根据mask模式设置firstFullLoadS1OuterIdx
    int64_t firstFullLoadS1OuterIdx = (faInfo.maskMode == MASK_MODE_NO_MASK) ? -1 : 0;
    faRunTilingAdapter_.multiCoreParamsRegbase.set_firstFullLoadS1OuterIdx(firstFullLoadS1OuterIdx);


    auto &initOutputParams = faRunTilingAdapter_.initOutputParams;
    int64_t outSize = faInfo.opParamInfo.attnOut.shape->GetStorageShape().GetShapeSize();
    int64_t lseSize = faInfo.softmaxLseFlag ? faInfo.opParamInfo.lseOut.shape->GetStorageShape().GetShapeSize() : 0;
    uint32_t singleCoreSize = (outSize + platformInfo_.aivNum - 1) / (platformInfo_.aivNum);
    // if (faInfo.isOutQuantEnable) {
    //     singleCoreSize = ((singleCoreSize + 1) / 2) * 2;
    // }

    initOutputParams.set_singleCoreSize(singleCoreSize);
    initOutputParams.set_totalOutputSize(outSize);
    initOutputParams.set_totalSoftMaxLseOutputSize(lseSize);
    initOutputParams.set_needInit(faInfo.needInit || needInit_);


    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttnTilingImpl::SetTilingData(gert::TilingContext *context, const FaTilingInfo &faInfo)
{
    if (!faInfo.emptyTensorFlag) {
        SetFATilingData(faInfo);
    }

    int64_t cap = context->GetRawTilingData()->GetCapacity();
    OP_LOGI(faInfo.opName, "Tiling Data context GetCapacity: %lu.", cap);
    PrintAllTilingData(faInfo);
    FlashAttnSimplifiedTilingData *tiling = context->GetTilingData<FlashAttnSimplifiedTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE(faInfo.opName, "The tiling data is nullptr"), return ge::GRAPH_FAILED);
    *tiling = faRunTilingAdapter_;
    return ge::GRAPH_SUCCESS;
}

void FlashAttnTilingImpl::PrintAllTilingData(const FaTilingInfo &faInfo)
{
    OP_LOGD(faInfo.opName, "bSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_bSize());
    OP_LOGD(faInfo.opName, "t1Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_t1Size());
    OP_LOGD(faInfo.opName, "t2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_t2Size());
    OP_LOGD(faInfo.opName, "n2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_n2Size());
    OP_LOGD(faInfo.opName, "gSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_gSize());
    OP_LOGD(faInfo.opName, "s1Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_s1Size());
    OP_LOGD(faInfo.opName, "s2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_s2Size());
    OP_LOGD(faInfo.opName, "alignedS2:%d", faRunTilingAdapter_.inputParamsRegbase.get_alignedS2());
    OP_LOGD(faInfo.opName, "dSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_dSize());
    OP_LOGD(faInfo.opName, "dSizeV:%d", faRunTilingAdapter_.inputParamsRegbase.get_dSizeV());
    OP_LOGD(faInfo.opName, "keepProb:%d", faRunTilingAdapter_.inputParamsRegbase.get_keepProb());
    OP_LOGD(faInfo.opName, "softmaxScale:%d", faRunTilingAdapter_.inputParamsRegbase.get_softmaxScale());
    // OP_LOGD(faInfo.opName, "winLefts:%d", faRunTilingAdapter_.inputParamsRegbase.get_winLefts());
    // OP_LOGD(faInfo.opName, "winRights:%d", faRunTilingAdapter_.inputParamsRegbase.get_winRights());
    OP_LOGD(faInfo.opName, "bandIndex:%d", faRunTilingAdapter_.inputParamsRegbase.get_bandIndex());
    OP_LOGD(faInfo.opName, "layoutType:%d", faRunTilingAdapter_.inputParamsRegbase.get_layoutType());

    OP_LOGD(faInfo.opName, "attenMaskShapeType:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskShapeType());
    OP_LOGD(faInfo.opName, "attenMaskDataType:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskDataType());
    OP_LOGD(faInfo.opName, "attenMaskCompressMode:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_attenMaskCompressMode());

    OP_LOGD(faInfo.opName, "maskType:%d", faRunTilingAdapter_.inputParamsRegbase.get_maskType()); //


    OP_LOGD(faInfo.opName, "attenMaskS2Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskS2Size());
    OP_LOGD(faInfo.opName, "qStartIdx:%d", faRunTilingAdapter_.inputParamsRegbase.get_qStartIdx());
    OP_LOGD(faInfo.opName, "kvStartIdx:%d", faRunTilingAdapter_.inputParamsRegbase.get_kvStartIdx());
    OP_LOGD(faInfo.opName, "s1MaskValidSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_s1MaskValidSize());
    OP_LOGD(faInfo.opName, "s2MaskValidSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_s2MaskValidSize());


    OP_LOGD(faInfo.opName, "isActualSeqLengthsNull:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_isActualSeqLengthsNull());
    OP_LOGD(faInfo.opName, "isActualSeqLengthsKVNull:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_isActualSeqLengthsKVNull());
    OP_LOGD(faInfo.opName, "actualSeqLengthsSize:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_actualSeqLengthsSize());
    OP_LOGD(faInfo.opName, "actualSeqLengthsKVSize:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_actualSeqLengthsKVSize());
    OP_LOGD(faInfo.opName, "isKvContinuous:%d", faRunTilingAdapter_.inputParamsRegbase.get_isKvContinuous());
    OP_LOGD(faInfo.opName, "fromFused:%d", faRunTilingAdapter_.inputParamsRegbase.get_fromFused());
    OP_LOGD(faInfo.opName, "isBSNDOut:%d", faRunTilingAdapter_.inputParamsRegbase.get_isBSNDOut());
    OP_LOGD(faInfo.opName, "transposeLayout:%d", faRunTilingAdapter_.inputParamsRegbase.get_transposeLayout());
    OP_LOGD(faInfo.opName, "isGqa:%d", faRunTilingAdapter_.inputParamsRegbase.get_isGqa());
    OP_LOGD(faInfo.opName, "isSoftMaxLseEnable:%d", faRunTilingAdapter_.inputParamsRegbase.get_isSoftMaxLseEnable());
    OP_LOGD(faInfo.opName, "isActualSharedPrefixLenNull:%d",
            faRunTilingAdapter_.inputParamsRegbase.get_isActualSharedPrefixLenNull());
    OP_LOGD(faInfo.opName, "isQHasLeftPadding:%d", faRunTilingAdapter_.inputParamsRegbase.get_isQHasLeftPadding());
    OP_LOGD(faInfo.opName, "isKVHasLeftPadding:%d", faRunTilingAdapter_.inputParamsRegbase.get_isKVHasLeftPadding());
    OP_LOGD(faInfo.opName, "prefixSeqInnerSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_prefixSeqInnerSize());
    OP_LOGD(faInfo.opName, "headNumRatio:%d", faRunTilingAdapter_.inputParamsRegbase.get_headNumRatio());
    OP_LOGD(faInfo.opName, "blockSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_blockSize());
    OP_LOGD(faInfo.opName, "blockTableDim2:%d", faRunTilingAdapter_.inputParamsRegbase.get_blockTableDim2());
    OP_LOGD(faInfo.opName, "paBlockNumSum:%d", faRunTilingAdapter_.inputParamsRegbase.get_paBlockNumSum());
    OP_LOGD(faInfo.opName, "paLayoutType:%d", faRunTilingAdapter_.inputParamsRegbase.get_paLayoutType());
    OP_LOGD(faInfo.opName, "attenMaskS1Size:%d", faRunTilingAdapter_.inputParamsRegbase.get_attenMaskS1Size());
    OP_LOGD(faInfo.opName, "isRowInvalid:%d", faRunTilingAdapter_.inputParamsRegbase.get_isRowInvalid());
    OP_LOGD(faInfo.opName, "kvSplitPart:%d", faRunTilingAdapter_.inputParamsRegbase.get_kvSplitPart());
    OP_LOGD(faInfo.opName, "accumOutSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_accumOutSize());
    OP_LOGD(faInfo.opName, "logSumExpSize:%d", faRunTilingAdapter_.inputParamsRegbase.get_logSumExpSize());


    OP_LOGD(faInfo.opName, "coreNum:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_coreNum());
    OP_LOGD(faInfo.opName, "totalSize:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_totalSize());
    OP_LOGD(faInfo.opName, "s1OuterSize:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_s1OuterSize());
    OP_LOGD(faInfo.opName, "splitFactorSize:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorSize());
    OP_LOGD(faInfo.opName, "splitFactorTailSize:%d",
            faRunTilingAdapter_.multiCoreParamsRegbase.get_splitFactorTailSize());
    for (uint32_t i = 0; i < 48; i++) {
        OP_LOGD(faInfo.opName, "bnStartIdx[%d]:%d", i,
                faRunTilingAdapter_.multiCoreParamsRegbase.get_bnStartIdxPtr()[i]);
    }
    for (uint32_t i = 0; i < 48; i++) {
        OP_LOGD(faInfo.opName, "maskStartIdx[%d]:%d", i,
                faRunTilingAdapter_.multiCoreParamsRegbase.get_maskStartIdxPtr()[i]);
    }
    OP_LOGD(faInfo.opName, "firstFullLoadS1OuterIdx:%d",
            faRunTilingAdapter_.multiCoreParamsRegbase.get_firstFullLoadS1OuterIdx());
    OP_LOGD(faInfo.opName, "splitCoreMode:%d", faRunTilingAdapter_.multiCoreParamsRegbase.get_splitCoreMode());


    OP_LOGD(faInfo.opName, "singleCoreSize:%d", faRunTilingAdapter_.initOutputParams.get_singleCoreSize());
    OP_LOGD(faInfo.opName, "isOneN:%d", faRunTilingAdapter_.initOutputParams.get_isOneN());
    OP_LOGD(faInfo.opName, "totalOutputSize:%d", faRunTilingAdapter_.initOutputParams.get_totalOutputSize());
    OP_LOGD(faInfo.opName, "totalSoftMaxLseOutputSize:%d",
            faRunTilingAdapter_.initOutputParams.get_totalSoftMaxLseOutputSize());
}
ge::graphStatus FlashAttnTilingImpl::DoOpTiling(gert::TilingContext *context,
                                                const FaTilingInfo &faInfo) // TODO 参考新的patch 再做一下更改
{
    OP_CHECK_IF(SetPlatMemoryInfo(context, faInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE(faInfo.opName, "Set plat memory info fail."), return ge::GRAPH_FAILED);

    if (faInfo.emptyTensorFlag) { // TODO 确认支持程度
        OP_CHECK_IF(SetEmptyTensor(context, faInfo) != ge::GRAPH_SUCCESS,
                    OP_LOGE(faInfo.opName, "Set emptyt ensor fail."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    InitImplParam(faInfo);

    OP_CHECK_IF(SplitPolicy(context, faInfo) != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "Excute split policy fail."),
                return ge::GRAPH_FAILED); // TODO 修改

    OP_CHECK_IF(ComputeTilingData(faInfo) != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "Compute tilingData fail."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GenTilingKey(context, faInfo) != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "Generate tilingKey fail."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetBlockDim(context, faInfo) != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "Set blockDim fail."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(GetWorkspace(context, faInfo) != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "Get workspace fail."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(SetTilingData(context, faInfo) != ge::GRAPH_SUCCESS, OP_LOGE(faInfo.opName, "Set tiling data fail."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;

    // REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttn, FlashAttnTilingImpl, static_cast<int32_t>(NpuArch::DAV_3510), 83);
}

} // namespace optiling