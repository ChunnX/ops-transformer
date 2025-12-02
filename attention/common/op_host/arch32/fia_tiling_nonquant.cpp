/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_tiling_nonquant.cpp
 * \brief
 */

#include "fia_tiling_nonquant.h"
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "../fia_tiling_templates_registry.h"
#include "../split_core_v1.h"
#include "../../../fused_infer_attention_score/op_kernel/fused_infer_attention_score_template_tilingkey.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

constexpr uint64_t PRE_LOAD_NUM_MLA = 2;
constexpr uint32_t BLOCK_TABLE_ELEM_BYTE = 4;

constexpr uint64_t FIA_TILINGKEYOFFSET = uint64_t(100000000000000000UL);          
constexpr uint64_t FIA_PERF_MODE_TILINGKEYOFFSET = uint64_t(1000000000000000UL); 

constexpr uint32_t G_SIZE_128 = 128;
constexpr uint32_t S1_SIZE_16 = 16;

constexpr uint32_t QK_HEAD_DIM_64 = 64;
constexpr uint32_t QK_HEAD_DIM_128 = 128;
constexpr uint32_t QK_HEAD_DIM_192 = 192;

constexpr uint32_t ROPE_HEAD_DIM_0 = 0;
constexpr uint32_t ROPE_HEAD_DIM_64 = 64;

constexpr uint32_t V_HEAD_DIM_64 = 64;
constexpr uint32_t V_HEAD_DIM_128 = 128;

constexpr uint32_t S_INNER_SIZE_512 = 512;
constexpr uint32_t S_INNER_SIZE_1024 = 1024;

constexpr uint32_t S_INNER_SIZE_ALIGN_512 = 512;
constexpr uint32_t S_INNER_SIZE_ALIGN_1024 = 1024;
constexpr uint32_t S_INNER_SIZE_ALIGN_2048 = 2048;
constexpr uint32_t S_INNER_SIZE_ALIGN_4096 = 4096;

constexpr int32_t SPARSE_MODE_2 = 2;
constexpr int32_t SPARSE_MODE_3 = 3;
constexpr int32_t SPARSE_MODE_4 = 4;

constexpr uint32_t M_BASE_SIZE_32 = 32;
constexpr uint32_t M_BASE_SIZE_64 = 64;
constexpr uint32_t M_BASE_SIZE_128 = 128;
constexpr uint32_t M_BASE_SIZE_256 = 256;
constexpr uint32_t M_BASE_SIZE_512 = 512;

constexpr uint8_t TILINGKEY_NUM_0 = 0;
constexpr uint8_t TILINGKEY_NUM_3 = 3;

constexpr int64_t MAX_ACTUAL_SEQUENCE = 2048;

constexpr uint32_t MAX_GSIZE = 64;

template <typename T> 
inline auto Align(T num, T rnd) -> T
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

constexpr uint64_t RecursiveSum()
{
    return 0;
}

template <typename T, typename... Args> 
constexpr uint64_t RecursiveSum(T templateId, Args... templateIds)
{
    return static_cast<uint64_t>(templateId) + 10U * RecursiveSum(templateIds...);
}

template <typename... Args> 
constexpr uint64_t FIA_GET_TILINGKEY(Args... templateIds)
{
    return RecursiveSum(templateIds...);
}

void FiaTilingNonQuant::InitTilingInfo(TilingInfo *tilingInfo)
{
    fiaInfo_ = static_cast<FiaTilingInfo *>(tilingInfo);
}

ge::graphStatus FiaTilingNonQuant::GetPlatformInfo()
{
    OP_CHECK_IF(fiaInfo_->platformInfo == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(fiaInfo_->opName, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0,
        OPS_REPORT_VECTOR_INNER_ERR(fiaInfo_->opName, "num of core obtained is 0."), return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool FiaTilingNonQuant::IsCapable()
{
    if (fiaInfo_ == nullptr) {
        return false;
    }

    ge::DataType qDataType = fiaInfo_->inputQType;
    ge::DataType kDataType = fiaInfo_->inputKvType;

    if ((qDataType == ge::DT_FLOAT16 || qDataType == ge::DT_BF16) && (qDataType == kDataType)) {
        if ((fiaInfo_->qkHeadDim  == QK_HEAD_DIM_128 && fiaInfo_->ropeHeadDim  == ROPE_HEAD_DIM_0 && fiaInfo_->vHeadDim == V_HEAD_DIM_128) || 
            (fiaInfo_->qkHeadDim  == QK_HEAD_DIM_64 && fiaInfo_->ropeHeadDim  == ROPE_HEAD_DIM_0 && fiaInfo_->vHeadDim == V_HEAD_DIM_64) ||
            (fiaInfo_->qkHeadDim  == QK_HEAD_DIM_192 && fiaInfo_->ropeHeadDim  == ROPE_HEAD_DIM_64 && fiaInfo_->vHeadDim == V_HEAD_DIM_128) ||
            (fiaInfo_->qkHeadDim  == QK_HEAD_DIM_128 && fiaInfo_->ropeHeadDim  == ROPE_HEAD_DIM_64 && fiaInfo_->vHeadDim == V_HEAD_DIM_128)) {
            return true;
        }
    }
    return false;
}

void FiaTilingNonQuant::GenTilingKey()
{
    uint8_t inputQVal{0}, inputKvVal{0}, outputVal{0};
    uint8_t softmaxBrcbFlagVal = static_cast<uint8_t>((softmaxWithBrcbFlag_) ? 1 * 4 : 0);
    uint64_t tilingNum_{0};

    const std::map<ge::DataType, uint8_t> typeMap = {
        {ge::DT_FLOAT16, 0U}, {ge::DT_BF16, 2U}, {ge::DT_INT8, 3U}, {ge::DT_INT4, 4U},
    };

    if (typeMap.find(fiaInfo_->inputQType) != typeMap.end()) {
        inputQVal = typeMap.at(fiaInfo_->inputQType);
    }
    if (typeMap.find(fiaInfo_->inputKvType) != typeMap.end()) {
        inputKvVal = typeMap.at(fiaInfo_->inputKvType);
    }
    if (typeMap.find(fiaInfo_->outputType) != typeMap.end()) {
        outputVal = typeMap.at(fiaInfo_->outputType);
    }

    bool isFlashDecode = (kvSplit_ > 0);
    bool isPageAttention = (fiaInfo_->pageAttentionFlag && fiaInfo_->s2Size != 0);
    tilingKey_ = GET_TPL_TILING_KEY(static_cast<uint8_t>(inputQVal), static_cast<uint8_t>(inputKvVal), static_cast<uint8_t>(outputVal), static_cast<uint8_t>(isPageAttention),
                                    static_cast<uint8_t>(fiaInfo_->inputLayout),
                                    static_cast<uint8_t>(fiaInfo_->inputKvLayout), static_cast<uint8_t>(isFlashDecode), static_cast<uint8_t>(fiaInfo_->sysPrefixFlag),
                                    TILINGKEY_NUM_0, TILINGKEY_NUM_0, TILINGKEY_NUM_0, TILINGKEY_NUM_0, TILINGKEY_NUM_3, TILINGKEY_NUM_3, TILINGKEY_NUM_0, softmaxBrcbFlagVal, TILINGKEY_NUM_0);
    tilingNum_ = tilingKey_;
    OP_LOGI(fiaInfo_->opName, "FIA tilingNum_: %lu.", tilingNum_);
}

bool FiaTilingNonQuant::IsFlashDecode(uint32_t coreNum)
{
    uint32_t tndFDCoreArrLen = tilingData_->fdParams.get_numOfFdHead();
    return tndFDCoreArrLen > static_cast<uint32_t>(0);

    if ((fiaInfo_->s1Size > static_cast<uint32_t>(1) || fiaInfo_->gSize > G_SIZE_128)) {
        return false;
    }

    float flashDecodeBNRatio = static_cast<float>(0.5); // 0.5, 经验值
    bool coreOkFlag = (static_cast<float>(fiaInfo_->bSize) * static_cast<float>(fiaInfo_->n2Size) <= flashDecodeBNRatio * static_cast<float>(coreNum));
    if (coreOkFlag && (fiaInfo_->gSize == static_cast<uint32_t>(1))) {
        OP_LOGD(fiaInfo_->opName, "flash decode split K/V tensors."); 
        return true;
    }
    if (coreOkFlag && (fiaInfo_->maxActualseq >= MAX_ACTUAL_SEQUENCE)) { 
        OP_LOGD(fiaInfo_->opName, "flash decode and GQA split K/V tensors.");
        return true;
    }

    return false;
}

bool FiaTilingNonQuant::DealSameSeqEachBatch()
{
    if (!fiaInfo_->batchContinuousFlag){
        if (fiaInfo_->actualSeqLenFlag){
            return fiaInfo_->isSameActualseq;
        } else {
            return fiaInfo_->isSameSeqAllKVTensor;
        }
    } else {
        return fiaInfo_->isSameActualseq;
    }
}

void FiaTilingNonQuant::ZeroTensorProcess()
{
    if (fiaInfo_->s2Size == 0) {
        /*
         * 1024，空tensor场景下，作为默认值完成后续计算
         * 避免matmal tiling  softmax tiling异常
         * kernel计算使用真实的seqSize=0, 与actuseq_len流程归一
         */
        fiaInfo_->s2Size = 1024;
    }
}

void FiaTilingNonQuant::InitParams()
{
    perfMode_ = IfaPerfMode::CUBE_VIEW_MM;
    coreNum_ = aicNum_;
    blockDim_ = aicNum_; // Tiling下沉首次Tiling也会校验blockDim_是否为0，为避免拦截报错，将blockDim_设置为aicNum_，实际不生效

    headDimAlign_ = Align(fiaInfo_->qkHeadDim, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    ZeroTensorProcess();
}

void FiaTilingNonQuant::CalcInnerSize(uint32_t s2Size)
{
    if (fiaInfo_->inputLayout == TilingKeyLayout::TND || fiaInfo_->inputLayout == TilingKeyLayout::NTD) {
        sInnerSize_ = S_INNER_SIZE_512;
    } else {
        if (fiaInfo_->s1Size <= S1_SIZE_16) {
            /**
            * V1阶段分配用于存放mm1结果的UB大小为32K, 当计算的数据类型为float时，其可以存放8192个元素.
            * 另外, 需要保证单次计算不会切分S2, 那么S2的内切大小最大为8192, 所以将默认值设置为8192
            */
            sInnerSize_ = MAX_SPLIT_SIZE; // 8192

            /** 当前版本限制workspace大小不超过32MB，否则会影响网络中前后算子性能，
            *  GQA场景下 nNumOfQInOneGroup和sInnerSize_切分大小直接影响workspace大小,
            *  具体计算参考CalcWorkSpace函数，这里根据nNumOfQInOneGroup将sInnerSize_
            *  分为8192，4096，2048三档，nNumOfQInOneGroup增大时减小sInnerSize_，
            *   保证最终workspace大小不超过32MB。
            */
            uint32_t sInnerSize[3U] = {8192U, 4096U, 2048U};
            uint32_t idx = std::min(fiaInfo_->gSize / 5U, 2U);
            sInnerSize_ = sInnerSize[idx];
        } else {
            bool highPreciseFlag = ((fiaInfo_->innerPrecise & 1) == 0) ? true : false;
            sInnerSize_ = ((highPreciseFlag && fiaInfo_->inputQType == ge::DT_FLOAT16) ||
                fiaInfo_->inputQType == ge::DT_BF16) ? S_INNER_SIZE_512 : S_INNER_SIZE_1024;
        }
    }
    if (fiaInfo_->attenMaskFlag && (fiaInfo_->sparseMode == SPARSE_MODE_2 || fiaInfo_->sparseMode == SPARSE_MODE_3 || fiaInfo_->sparseMode == SPARSE_MODE_4)) {
        sInnerSize_ = std::min(sInnerSize_, S_INNER_SIZE_1024); // attention mask压缩场景，基本块最大支持1024*1024
    }
    // PA特性泛化场景，blockSize可能为112等值，无法被sInnerSize_整除，当step*base跨block时，搬运处理复杂，通过向下对齐避免
    if (fiaInfo_->pageAttentionFlag && fiaInfo_->blockSize != 0) {
        uint32_t blockSize = static_cast<uint32_t>(fiaInfo_->blockSize);
        if ((sInnerSize_ > blockSize) && (sInnerSize_ % blockSize != 0)) {
            sInnerSize_ = (sInnerSize_ / blockSize) * blockSize;
        }
    }
    if (sInnerSize_ != 0) {
        sInnerLoopTimes_ = (s2Size + sInnerSize_ - static_cast<uint32_t>(1)) / sInnerSize_;
        sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - static_cast<uint32_t>(1)) * sInnerSize_;
    }
    // tiling下沉 && flash decoder场景时，sInnerSize_基块大小不按照真实值修改
    // 否则会导致 tiling下沉 && flash decoder 场景时开辟workspace空间大小小于真实运行时所需的workspace大小
    if (sInnerSize_ > s2Size) {
        sInnerSize_ = s2Size;
    }
    sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK); // 元素个数按照基本块大小对齐
    CalcMBaseSize();
}

void FiaTilingNonQuant::CalcMBaseSize()
{
    if (fiaInfo_->inputLayout == TilingKeyLayout::TND || fiaInfo_->inputLayout == TilingKeyLayout::NTD) {
        mBaseSize_ = M_BASE_SIZE_512;
    } else {
        if (fiaInfo_->s1Size <= S1_SIZE_16) {
            if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_512) {
                mBaseSize_ = M_BASE_SIZE_512;
            } else if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_1024) {
                mBaseSize_ = M_BASE_SIZE_256;
            } else if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_2048) {
                mBaseSize_ = M_BASE_SIZE_128;
            } else if (sInnerSizeAlign_ <= S_INNER_SIZE_ALIGN_4096) {
                mBaseSize_ = M_BASE_SIZE_64;
            } else { // sInnerSizeAlign_最大值为8192
                mBaseSize_ = M_BASE_SIZE_32;
            }
        } else {
            bool highPreciseFlag = (fiaInfo_->innerPrecise & 1 == 0) ? true : false;
            mBaseSize_ = ((highPreciseFlag && fiaInfo_->inputQType == ge::DT_FLOAT16) ||
                           fiaInfo_->inputQType == ge::DT_BF16) ? M_BASE_SIZE_256 : M_BASE_SIZE_512;
        }
    }
    softmaxWithBrcbFlag_ = (mBaseSize_ <= M_BASE_SIZE_128);

    OP_LOGI(fiaInfo_->opName, "FIA sInnerSize_:%u sInnerSizeAlign_:%u mBaseSize_:%u softmaxWithBrcbFlag_:%u.",
        sInnerSize_, sInnerSizeAlign_, mBaseSize_, softmaxWithBrcbFlag_);
}

void FiaTilingNonQuant::CreateSplitInput(BaseInfo &baseInfo)
{
    //构造分核输入参数
    baseInfo.bSize = fiaInfo_->bSize;
    baseInfo.n2Size = fiaInfo_->n2Size;
    baseInfo.gSize = fiaInfo_->gSize;
    baseInfo.s2Size = fiaInfo_->s2Size;
    baseInfo.s1Size = fiaInfo_->s1Size;
    baseInfo.actualLenQDims = fiaInfo_->actualLenQDims;
    baseInfo.actualLenKvDims = fiaInfo_->actualLenDims;

    if (fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor != nullptr) {
        baseInfo.actualSeqS1Size = fiaInfo_->opParamInfo.actualSeqLengthsQ.tensor->GetData<int64_t>();
        baseInfo.isAccumSeqS1 = fiaInfo_->isAccumQSeq;
    }
    if (fiaInfo_->opParamInfo.actualSeqLengths.tensor != nullptr) {
        baseInfo.actualSeqS2Size = fiaInfo_->opParamInfo.actualSeqLengths.tensor->GetData<int64_t>();
        baseInfo.isAccumSeqS2 = fiaInfo_->isAccumKVSeq;
    }
}

void FiaTilingNonQuant::CreateSplitOutput(OuterSplitParams &outerSplitParams, FlashDecodeParams &fDParams, SplitCoreRes &res)
{   
    outerSplitParams.bN2End = tilingData_->outerSplitParams.get_bN2End();
    outerSplitParams.gS1End = tilingData_->outerSplitParams.get_gS1End();
    outerSplitParams.s2End = tilingData_->outerSplitParams.get_s2End();
    
    tilingData_->fdParams.set_gS1BaseSizeOfFd(mFdBaseSize_);
    fDParams.bN2IdxOfFdHead = tilingData_->fdParams.get_bN2IdxOfFdHead();
    fDParams.gS1IdxOfFdHead = tilingData_->fdParams.get_gS1IdxOfFdHead();
    fDParams.s2SplitNumOfFdHead = tilingData_->fdParams.get_s2SplitNumOfFdHead();
    fDParams.s2SplitStartIdxOfCore = tilingData_->fdParams.get_s2SplitStartIdxOfCore();
    fDParams.gS1BaseSizeOfFd = tilingData_->fdParams.get_gS1BaseSizeOfFd();
    fDParams.gS1SplitNumOfFdHead = tilingData_->fdParams.get_gS1SplitNumOfFdHead();
    fDParams.gS1LastPartSizeOfFdHead = tilingData_->fdParams.get_gS1LastPartSizeOfFdHead();
    fDParams.gS1IdxEndOfFdHead = tilingData_->fdParams.get_gS1IdxEndOfFdHead();
    fDParams.gS1IdxEndOfFdHeadSplit = tilingData_->fdParams.get_gS1IdxEndOfFdHeadSplit();
    
    res.numOfFdHead = 0;
    res.maxS2SplitNum = 1;
    res.usedCoreNum = aicNum_;
}

void FiaTilingNonQuant::Split()
{
    uint32_t s2SizeInput = static_cast<uint32_t>(fiaInfo_->s2Size);
    CalcInnerSize(s2SizeInput);

    BaseInfo baseInfo;
    CreateSplitInput(baseInfo);

    InnerSplitParams innerSplitParams;
    innerSplitParams.s1GBaseSize = mBaseSize_;
    innerSplitParams.s2BaseSize = sInnerSize_;
    tilingData_->innerSplitParams.set_mBaseSize(innerSplitParams.s1GBaseSize);
    tilingData_->innerSplitParams.set_s2BaseSize(innerSplitParams.s2BaseSize);

    OuterSplitParams outerSplitParams;
    FlashDecodeParams fDParams;
    SplitCoreRes res;
    CreateSplitOutput(outerSplitParams, fDParams, res);

    SplitCore(baseInfo, innerSplitParams, aicNum_, outerSplitParams, fDParams, res);
    if (res.numOfFdHead > aicNum_ || res.usedCoreNum > aicNum_ || res.maxS2SplitNum > aicNum_ + 1) {
        OP_LOGE(fiaInfo_->opName, "used_core_num: %u, num_of_fd_head: %u, max_s2_split_num: %u, aic_num: %u", 
            res.usedCoreNum, res.numOfFdHead, res.maxS2SplitNum, aicNum_);
    }

    tilingData_->fdParams.set_numOfFdHead(res.numOfFdHead);
    usedCoreNum_ = res.usedCoreNum;

    if (IsFlashDecode(coreNum_)) {
        splitKVFlag_ = true;
        kvSplit_++;
        kvSplitPart_ = res.maxS2SplitNum; // kvSplitPart_, 用于lse out workspace计算
        SplitFD(res, fDParams, usedCoreNum_);
        tilingData_->fdParams.set_usedVecNumOfFd(res.usedVecNumOfFd);
    }
    CalcMmResSize();
}

uint32_t FiaTilingNonQuant::GetL2CacheOffFlag()
{
    uint64_t kvTypeSize = 2;
    uint64_t kvSize = 0;
    float l2CacheSizeCoeff = static_cast<float>(1.2);
    if (fiaInfo_->kvStorageMode == KvStorageMode::PAGE_ATTENTION) {
        kvSize = fiaInfo_->opParamInfo.key.shape->GetStorageShape().GetShapeSize();
    } else if (fiaInfo_->kvStorageMode == KvStorageMode::TENSOR_LIST) {
        for (int64_t size = 0; size < fiaInfo_->bSize; ++size) {
            auto keyTensorInList = fiaInfo_->kCache[size];
            kvSize += keyTensorInList->GetStorageShape().GetShapeSize();
        }
    } else {
        kvSize = fiaInfo_->opParamInfo.key.shape->GetStorageShape().GetShapeSize();
    }

    uint64_t l2CacheSize = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);

    // 之前路由到IFA的GQA场景才需要考虑关闭L2Cache
    if ((fiaInfo_->ropeMode == RopeMode::NO_ROPE) && (fiaInfo_->s1Size == 1) && (fiaInfo_->gSize <= MAX_GSIZE)) {
        // 1. 连续访存时, 即KV的layout为BNSD或者BnNBsD, 不涉及数据预取, 可以直接关闭L2Cache
        // 2. 考虑K和V数据的总大小超过一定值后, 关闭L2Cache, 当前系数确定为1.2
        if (fiaInfo_->kvLayout == FiaLayout::BNSD || fiaInfo_->kvLayout == FiaLayout::BnNBsD) {
            l2CacheOffFlag_ = 1U;
        } else if (static_cast<double>(kvSize) * kvTypeSize * 2.0f >= l2CacheSize * l2CacheSizeCoeff) {
            l2CacheOffFlag_ = 1U;
        } else {
            l2CacheOffFlag_ = 0;
        }
    } else {
        l2CacheOffFlag_ = 0;
    }

    OP_LOGD(fiaInfo_->opName, "l2CacheOffFlag_: %u, kvSize: %lu, kvTypeSize: %u, l2CacheSize: %lu",
            l2CacheOffFlag_, kvSize, kvTypeSize, l2CacheSize);
    return l2CacheOffFlag_;
}

void FiaTilingNonQuant::FillTilingBaseParams()
{
    tilingData_->baseParams.set_bSize(fiaInfo_->bSize);
    tilingData_->baseParams.set_s2Size(fiaInfo_->s2Size);
    tilingData_->baseParams.set_s1Size(fiaInfo_->s1Size);
    tilingData_->baseParams.set_n2Size(fiaInfo_->n2Size);
    tilingData_->baseParams.set_headDim(fiaInfo_->vHeadDim);
    tilingData_->baseParams.set_headDimRope(fiaInfo_->ropeHeadDim);
    tilingData_->baseParams.set_scaleValue(fiaInfo_->scaleValue);
    tilingData_->baseParams.set_gSize(fiaInfo_->n1Size / fiaInfo_->n2Size);
    tilingData_->baseParams.set_batchContinuous((fiaInfo_->kvStorageMode == KvStorageMode::TENSOR_LIST) ? 0 : 1);
    tilingData_->baseParams.set_actualSeqS1Dims(fiaInfo_->actualLenQDims);
    tilingData_->baseParams.set_actualSeqS2Dims(fiaInfo_->actualLenDims);
    tilingData_->baseParams.set_accumQSeqFlag(fiaInfo_->isAccumQSeq ? 1 : 0);
    tilingData_->baseParams.set_accumKVSeqFlag(fiaInfo_->isAccumKVSeq ? 1 : 0);
    tilingData_->baseParams.set_outputLayout(static_cast<uint32_t>(fiaInfo_->outputLayout));
    tilingData_->baseParams.set_softmaxLseFlag(fiaInfo_->softmaxLseFlag ? 1 : 0);
    tilingData_->baseParams.set_usedCoreNum(usedCoreNum_);
    l2CacheOffFlag_ = GetL2CacheOffFlag();
    tilingData_->baseParams.set_l2CacheOffFlag(l2CacheOffFlag_);
}
 
void FiaTilingNonQuant::FillTilingPageAttenParams()
{
    tilingData_->pageAttenParams.set_blockSize(fiaInfo_->blockSize);
    tilingData_->pageAttenParams.set_maxBlockNumPerBatch(fiaInfo_->maxBlockNumPerBatch);
}
 
void FiaTilingNonQuant::FillTilingMaskParams()
{
    tilingData_->maskParams.set_attenMaskFlag(fiaInfo_->attenMaskFlag ? 1 : 0);
    tilingData_->maskParams.set_attenMaskSize(fiaInfo_->attenMaskSize);
    tilingData_->maskParams.set_attenMaskStride(fiaInfo_->attenMaskStride);
    tilingData_->maskParams.set_sparseMode(fiaInfo_->sparseMode);
    tilingData_->maskParams.set_preToken(fiaInfo_->preToken);
    tilingData_->maskParams.set_nextToken(fiaInfo_->nextToken);
    uint32_t isRowInvalid = fiaInfo_->innerPrecise >> 1;
    tilingData_->maskParams.set_isRowInvalid(isRowInvalid);
}

// for flash decode
void FiaTilingNonQuant::FillTilingWorkspaceParams()
{
    uint32_t maxConventNum = 2;
    uint32_t numOfFdSumMax = 2;
    // 每个核可能有头规约和尾规约，一共两份规约信息
    tilingData_->workspaceParams.set_fdAccumOutSize(aicNum_ * maxConventNum * mBaseSize_ * headDimAlign_);
    // 每个核可能有头规约和尾规约，一共两份规约信息; 另外sum和max各一份
    tilingData_->workspaceParams.set_fdLogSumExpSize(numOfFdSumMax * aicNum_ * maxConventNum * mBaseSize_ * (BYTE_BLOCK / BLOCK_TABLE_ELEM_BYTE));
    tilingData_->workspaceParams.set_mm1ResSize(mm1ResSize_);
    tilingData_->workspaceParams.set_mm2ResSize(mm2ResSize_);
}

void FiaTilingNonQuant::CalcMmResSize()
{
    int64_t mSize = std::min(fiaInfo_->gSize * fiaInfo_->s1Size, mBaseSize_);
    mm1ResSize_ = static_cast<int64_t>(sInnerSizeAlign_) * mSize;
    mm2ResSize_ = static_cast<int64_t>(headDimAlign_) * mSize;
}

void FiaTilingNonQuant::CalcMaxMmResSize()
{
    mBaseSize_ = M_BASE_SIZE_512;
    mm1ResSize_ = 512 * 512; // mm1的结果最大为512*512个元素
    mm2ResSize_ = static_cast<int64_t>(headDimAlign_) * 512; // mBaseSize最大值为512
}

void FiaTilingNonQuant::FillTiling()
{
    FillTilingBaseParams();
    FillTilingPageAttenParams();
    FillTilingMaskParams();
    FillTilingWorkspaceParams();
}

uint32_t FiaTilingNonQuant::CalcFlashDecodeParamNums(const uint32_t coreNum) const
{
    return coreNum * 2U * mBaseSize_; // 每个核可能有头规约和尾规约，一共两份规约信息
}

uint64_t FiaTilingNonQuant::CalcNormalWorkspaceSize(uint32_t coreNum, int64_t mm1ResSize,
    int64_t mm2ResSize) const
{
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;      // 4: fp32
    constexpr uint32_t V1_RES_ELEM_SIZE = 2;       // 2: fp16/bf16
    constexpr uint32_t MM2_RES_ELEM_SIZE = 4;      // 4: fp32
    constexpr uint32_t V2_RES_ELEM_SIZE = 4;       // 4: fp32

    uint64_t workspaceSize = 0;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm1ResSize * MM1_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm1ResSize * V1_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm2ResSize * MM2_RES_ELEM_SIZE;
    workspaceSize += PRE_LOAD_NUM_MLA * coreNum * mm2ResSize * V2_RES_ELEM_SIZE;
    return workspaceSize;
}

uint64_t FiaTilingNonQuant::CalcFlashDecodeWorkspace(const uint32_t coreNum) const
{
    uint64_t flashDecodeParamNums = static_cast<uint64_t>(CalcFlashDecodeParamNums(coreNum));
    uint64_t accumOutSize = flashDecodeParamNums * static_cast<uint64_t>(headDimAlign_);
    uint64_t logSumExpSize = 2 * flashDecodeParamNums * (BYTE_BLOCK / static_cast<uint64_t>(BLOCK_TABLE_ELEM_BYTE)); // log和sum的存储空间一致，共需要2份内存
    uint64_t workspaceSize = (accumOutSize + logSumExpSize) * static_cast<uint64_t>(BLOCK_TABLE_ELEM_BYTE);
    return workspaceSize;
}

void FiaTilingNonQuant::CalcWorkspaceSize()
{
    workspaceSize_ = libapiSize_;
    workspaceSize_ += CalcNormalWorkspaceSize(coreNum_, mm1ResSize_, mm2ResSize_);
    if (splitKVFlag_) {
        workspaceSize_ += CalcFlashDecodeWorkspace(coreNum_);
    }
}

void FiaTilingNonQuant::CalcMaxWorkspaceSize()
{
    CalcMaxMmResSize();
    workspaceSize_ = libapiSize_;
    workspaceSize_ += CalcNormalWorkspaceSize(coreNum_, mm1ResSize_, mm2ResSize_);
    workspaceSize_ += CalcFlashDecodeWorkspace(aicNum_);
}

void FiaTilingNonQuant::CalcBlockDim(uint32_t coreNum)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(fiaInfo_->platformInfo);
    auto aicNum = coreNum;
    auto aivNum = 2U * coreNum;  // vec核数量是cube核数量的两倍

    blockDim_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum); 
    OP_LOGI(fiaInfo_->opName, "FIA block dim: %u aiv Num: %u aic Num: %u.", blockDim_, aivNum, aicNum);
}

void FiaTilingNonQuant::CalcScheduleMode()
{
    scheduleMode_ = ScheduleMode::BATCH_MODE;
    OP_LOGI(fiaInfo_->opName, "FIA schedule mode: %u.", static_cast<uint32_t>(scheduleMode_));
}

ge::graphStatus FiaTilingNonQuant::DoOpTiling()
{
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    InitParams();
    if (fiaInfo_->isMaxWorkspace) {
        CalcMaxWorkspaceSize();
        GenTilingKey();
    } else {
        Split();
        FillTiling();
        CalcBlockDim(usedCoreNum_);
        CalcScheduleMode();
        CalcWorkspaceSize();
        GenTilingKey();
    }

    if ((SetBlockDim(blockDim_) != ge::GRAPH_SUCCESS) ||
        (SetTilingKey(tilingKey_) != ge::GRAPH_SUCCESS) ||
        (SetWorkspaceSize(workspaceSize_) != ge::GRAPH_SUCCESS) ||
        (SetScheduleMode(scheduleMode_) != ge::GRAPH_SUCCESS)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// 值越小表示优先级越高. 对于FIA, 使用3位数表示优先级, 优先级编码含义为:
// 1. 百位代表非量化、伪量化、全量化等场景, 即: 0xx-非量化，1xx-伪量化, 2xx-全量化
// 2. 十位表示gqa、mla、泛化，即: x0x-mla, x1x-gpa, x2x-泛化
// 3. 个位代表特化模板到泛化模板的优先级排序
REGISTER_TILING_TEMPLATE_FIA(FusedInferAttentionScore, FiaTilingNonQuant,
    std::vector<int32_t>({(int32_t)platform_ascendc::SocVersion::ASCEND910B}), 19);
} // namespace optiling
