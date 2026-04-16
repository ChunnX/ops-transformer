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
 * \file flash_attn_tiling_impl.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTN_IMPL_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_FLASHATTN_IMPL_H_
#include "register/tilingdata_base.h"
#include "../fa_tiling_base.h"
#include "../fa_tiling_info.h"
#include "tiling/tiling_api.h"  //这个头文件顺序必须在手写的tiling data前
#include "../../op_kernel/arch35/flash_attn_tiling_regbase.h"
#include "../../op_kernel/arch35/flash_attn_template_tiling_key.h"

namespace optiling {

struct FaTilingKeyInfo {
    uint64_t kernelTypeKey = 0;
    uint64_t implMode = 0;
    uint64_t inputLayout = 2;
    uint64_t s1TemplateType = 128;
    uint64_t s2TemplateType = 128;
    uint64_t dTemplateType  = 64;
    uint64_t dvTemplateType = 0;
    bool hasAttn = 0;
    bool isPa = 0;
    bool isSoftmaxLse = 0;
    bool regbase = 1;
};

struct FaPlatFormInfo {
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t l0bSize = 0;
    uint64_t l0aSize = 0;
    uint32_t coreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t defaultSysWorkspaceSize = 0;
};

class FlashAttnTilingImpl : public FaTilingBase {
public:
    explicit FlashAttnTilingImpl(gert::TilingContext *context) : FaTilingBase(context) {}
    ~FlashAttnTilingImpl() override = default;
    void InitTilingInfo(TilingInfo *tilingInfo) {}
    bool IsCapable() {}
    ge::graphStatus DoOpTiling() {}
    ge::graphStatus DoOpTiling(gert::TilingContext *context, const FaTilingInfo &faInfo);

protected:
    ge::graphStatus SetPlatMemoryInfo(gert::TilingContext *context, const FaTilingInfo &faInfo);
    ge::graphStatus SetEmptyTensor(gert::TilingContext *context, const FaTilingInfo &faInfo);
    ge::graphStatus SplitPolicy(gert::TilingContext *context, const FaTilingInfo &faInfo);
    ge::graphStatus ComputeTilingData(const FaTilingInfo &faInfo);
    ge::graphStatus GenTilingKey(gert::TilingContext *context, const FaTilingInfo &faInfo);
    ge::graphStatus SetBlockDim(gert::TilingContext *context, const FaTilingInfo &faInfo);
    ge::graphStatus GetWorkspace(gert::TilingContext *context, const FaTilingInfo &faInfo);
    ge::graphStatus SetTilingData(gert::TilingContext *context, const FaTilingInfo &faInfo);
    ge::graphStatus SetFullQuantTilingData(const FaTilingInfo &faInfo);
    void UpdateTilingKeyConfig(const FaTilingInfo &faInfo);
    void UpdateTilingKeyLayout(const FaTilingInfo &faInfo);
    void UpdateTilingKeyPseMode(const FaTilingInfo &faInfo);
    void UpdateTilingKeyQuantMode(const FaTilingInfo &faInfo);
    void UpdateTilingKeyHasRope(const FaTilingInfo &faInfo);
    void UpdateTilingKeyMaskMode(const FaTilingInfo &faInfo);
    void UpdateTilingKeyMatmulMode(const FaTilingInfo &faInfo);
    ge::graphStatus UpdateTilingKeyInfo(const FaTilingInfo &faInfo);
    ge::graphStatus SetWorkspaceNormal(const FaTilingInfo &faInfo, int64_t &curWorkspaceSize);
    ge::graphStatus SetWorkspaceAntiQuant(const FaTilingInfo &faInfo, int64_t &workspaceSize_);
    ge::graphStatus SetWorkspacePTQuant(const FaTilingInfo &faInfo, int64_t &curWorkspaceSize);

    bool EnableMTE2BmmPipe(const FaTilingInfo &faInfo, matmul_tiling::MatmulApiTiling &bmm,
                           TCubeTiling &bmmTilingData);
    void GetMatMulType(const FaTilingInfo &faInfo, matmul_tiling::DataType &mmInputType,
                       matmul_tiling::DataType &mmOutputType);
    ge::graphStatus SetMM1TilingData(gert::TilingContext *context, const FaTilingInfo &faInfo);
    ge::graphStatus SetMM2TilingData(gert::TilingContext *context, const FaTilingInfo &faInfo);
    ge::graphStatus SetFATilingData(const FaTilingInfo &faInfo);

    ge::graphStatus AdjustSinnerAndSouter(gert::TilingContext *context, const FaTilingInfo &faInfo);
    void GetPreNextTokensLeftUp(const FaTilingInfo &faInfo, int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp);
    void FixParamWithRowInvalid(const FaTilingInfo &faInfo, int64_t &actualSeqLength, int64_t actualSeqLengthKV, 
                                int64_t &preTokensLeftUp, int64_t &nextTokensLeftUp);
    int64_t GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner, int64_t sOuter,
                            int64_t token);
    int64_t GetCalcBlockNumsOneHead(const FaTilingInfo &faInfo, int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                    uint32_t sOuterSize, uint32_t sInnerSize, int64_t preTokensLeftUp,
                                    int64_t nextTokensLeftUp);
    int64_t GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums);
    void ComputeSplitNBSeq(const FaTilingInfo &faInfo, const size_t tilingElementArrayLen, uint32_t sOuterSize,
                           uint32_t sInnerSize, double coreWeightTarget, uint32_t &curCore);
    void SplitNBSeq(const FaTilingInfo &faInfo);
    void InitImplParam(const FaTilingInfo &faInfo);
    void SetGSMerge(const FaTilingInfo &faInfo);
    bool CheckEnableDN(const FaTilingInfo &faInfo);
    bool CheckQKVActualSeqLengthsRight(const FaTilingInfo &faInfo);
    bool CheckFlashDecode(const FaTilingInfo &faInfo);
    ge::graphStatus SplitS2(const FaTilingInfo &faInfo);
    void SetDequantBaseSize(const FaTilingInfo &faInfo);
    ge::graphStatus CalcInnerSize(const FaTilingInfo &faInfo, uint32_t seqSize);
    void GetActualSeqLength(const FaTilingInfo &faInfo, int64_t &actualSeqLengths, int64_t &actualSeqLengthsKV, uint32_t bIdx);
    void ComputeDequantSplitNBSeq(const FaTilingInfo &faInfo, std::vector<int64_t> sOuterLoopTimes,
                                  std::vector<int64_t> sInnerLoopTimes, int64_t sInnerLoopTimesPrefix,
                                  double coreWeightTarget, uint32_t &curCore, const size_t tilingElementArrayLen);
    int64_t GetAntiQuantCalcBlockNumsOneHead( const FaTilingInfo &faInfo, int64_t outerBlockNums,
                                              int64_t innerBlockNums, int64_t sInnerLoopTimesPrefix,
                                              int64_t preTokensLeftUp, int64_t nextTokensLeftUp);
    void GetAntiQuantPreNextTokensLeftUp(const FaTilingInfo &faInfo, int64_t actualSeqLength,
                                         int64_t actualSeqLengthKV, int64_t &preTokensLeftUp,
                                         int64_t &nextTokensLeftUp);
    void FixAntiQuantParamWithRowInvalid(const FaTilingInfo &faInfo, int64_t &actualSeqLength, 
                                         int64_t actualSeqLengthKV, int64_t &preTokensLeftUp,
                                         int64_t &nextTokensLeftUp);
    void DequantCubeSplitBNSeq(const FaTilingInfo &faInfo);
    int64_t GetActualInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums);
    void SplitDequant(const FaTilingInfo &faInfo);
    ge::graphStatus SetDequantMMTilingData(gert::TilingContext *context, const FaTilingInfo &faInfo);
    bool GetMatmulType(ge::DataType getype, matmul_tiling::DataType *mmType);
    void AdjustPABmm1Tiling(const FaTilingInfo &faInfo, uint32_t &bmm1BaseN);
    void AdjustPABmm2Tiling(const FaTilingInfo &faInfo);
    bool CheckTransposeLayout(const FaTilingInfo &faInfo);
    void PrintAllTilingData(const FaTilingInfo &faInfo);

    FlashAttnSimplifiedTilingData faRunTilingAdapter_;
    FaTilingKeyInfo tilingKeyInfo_;
    FaPlatFormInfo platformInfo_;
    uint32_t nLoopTimes_;
    uint32_t gsSize_;
    uint32_t sOuterFactor_;
    uint32_t sInnerFactor_;
    uint32_t sInnerFactorSize_;
    uint32_t blockDim_;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t headDimAlign_ = 0;
    bool flashDecodeFlag_ = false;
    bool dnFlag_ = false;
    bool gsMergeFlag_ = false;
    bool pfaMergeFlag_ = false;
    bool isQKVActualSeqLengthsRightFlag_ = false;
    bool actualSeqLenQFlag_ = false;
    bool actualSeqLenKVFlag_ = false;
    bool actualSharedPrefixLenFlag_ = false;
    std::vector<int64_t> actualSeqLengthsQ_ = {};
    std::vector<int64_t> actualSeqLengthsKV_ = {};
    bool fromPFA_ = false;
    bool isPFAFlag_ = false;
    bool needInit_ = false;
};

}  // namespace optiling
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_FUSEDINFERATTENTIONSCORE_IMPL_H_