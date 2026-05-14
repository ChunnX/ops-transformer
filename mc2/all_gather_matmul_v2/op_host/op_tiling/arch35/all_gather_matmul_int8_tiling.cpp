/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the the software repository for the full text of the License.
 */

/*!
 * \file all_gather_matmul_int8_tiling.cpp
 * \brief INT8 Tiling for A5 (950PR) with A5-specific optimized parameters
 *         - Fixed Zn mode (swizzlDirect=0)
 *         - Optimized m0=128 (with m0=256 for large M)
 *         - INPUT_DTYPE=1 for INT8
 *         - swizzlCount=1 for simplicity
 */

#include "all_gather_matmul_int8_tiling.h"
#include "mc2_log.h"
#include "platform/platform_infos_def.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include <algorithm>

using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;
using namespace optiling;

namespace {

constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_IS_TRANS_X1 = 1;
constexpr uint32_t ATTR_IS_TRANS_X2 = 2;
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t X1_SCALE_INDEX = 3;
constexpr uint32_t X2_SCALE_INDEX = 4;
constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024;
constexpr uint32_t AIV_TYPE = 3;

// A5-specific constants
constexpr int32_t HALF_KBYTE = 512;
constexpr int32_t DEFAULT_ROW = 128;
constexpr int32_t DEFAULT_COL = 256;
constexpr int32_t LCAL_BUFF_BYTES = 204 * 1024 * 1024;
constexpr int32_t FLAG_BUFF_BYTES = 5 * 512 * 1024;
constexpr int32_t INPUT_DTYPE = 1; // INT8专用
constexpr int32_t MIN_P_VALUE = 1;
constexpr int32_t MAX_BLOCK_COUNT = 2;
constexpr int32_t NUMSIZE_ONE = 1;

const std::set<int> SUPPORT_RANKSIZE{2, 4, 8, 16, 32, 64, 128, 256};

// Tiling Code Mask (A5-specific)
constexpr int32_t COMMDATASPLIT_MASK = 0b11111;
constexpr int32_t COMMDATASPLIT_BNUM = 5;
constexpr int32_t COMMNPUSPLIT_MASK = 0b11111;
constexpr int32_t COMMNPUSPLIT_BNUM = 5;
constexpr int32_t COMMDIRECT_MASK = 0b1;
constexpr int32_t COMMDIRECT_BNUM = 1;
constexpr int32_t UBMOVENUM_MASK = 0b11111111;
constexpr int32_t UBMOVENUM_BNUM = 8;
constexpr int32_t PVALUE_MASK = 0b1111;
constexpr int32_t PVALUE_BNUM = 4;
constexpr int32_t SWIZZLCOUNT_MASK = 0b1111;
constexpr int32_t SWIZZLCOUNT_BNUM = 4;
constexpr int32_t SWIZZLDIRECT_MASK = 0b1;
constexpr int32_t SWIZZLDIRECT_BNUM = 1;
constexpr int32_t M0_MASK = 0b1;
constexpr int32_t M0_BNUM = 1;

// A5-specific mapping table: Fixed Zn mode, swizzlCount=1, commDataSplit=4
// M range segments with optimized parameters
// Code encoding: commDataSplit=4, swizzlDirect=0(Zn), swizzlCount=1, ubMoveNum=5120

// Helper function to encode tiling code (constexpr for compile-time evaluation)
constexpr int32_t EncodeA5TilingCode(int32_t pValue, int32_t m0Factor, int32_t commNpuSplit)
{
    int32_t code = 0;
    code |= 4; // commDataSplit = 4 (fixed)
    code |= (commNpuSplit & 0b11111) << COMMDATASPLIT_BNUM;
    code |= 1 << (COMMDATASPLIT_BNUM + COMMNPUSPLIT_BNUM);                    // commDirect = 1
    code |= 10 << (COMMDATASPLIT_BNUM + COMMNPUSPLIT_BNUM + COMMDIRECT_BNUM); // ubMoveNum = 10*512
    code |= (pValue & 0b1111) << (COMMDATASPLIT_BNUM + COMMNPUSPLIT_BNUM + COMMDIRECT_BNUM + UBMOVENUM_BNUM);
    code |=
        1 << (COMMDATASPLIT_BNUM + COMMNPUSPLIT_BNUM + COMMDIRECT_BNUM + UBMOVENUM_BNUM + PVALUE_BNUM); // swizzlCount=1
    code |= 0 << (COMMDATASPLIT_BNUM + COMMNPUSPLIT_BNUM + COMMDIRECT_BNUM + UBMOVENUM_BNUM + PVALUE_BNUM +
                  SWIZZLCOUNT_BNUM); // swizzlDirect=0(Zn)
    code |= (m0Factor & 0b1) << (COMMDATASPLIT_BNUM + COMMNPUSPLIT_BNUM + COMMDIRECT_BNUM + UBMOVENUM_BNUM +
                                 PVALUE_BNUM + SWIZZLCOUNT_BNUM + SWIZZLDIRECT_BNUM);
    return code;
}

// A5 INT8 2-Rank mapping table
static std::map<int, std::vector<std::vector<int>>> g_allGatherV2MatmulA5TwoRankINT8CodeMap = {
    {EncodeA5TilingCode(1, 0, 1), {{-1, 512, -1, 2147483647, -1, 2147483647}}},           // 极小M: m0=128, p=1
    {EncodeA5TilingCode(1, 0, 1), {{512, 1024, -1, 2147483647, -1, 2147483647}}},         // 小M: m0=128, p=1
    {EncodeA5TilingCode(2, 0, 1), {{1024, 2048, -1, 2147483647, -1, 2147483647}}},        // 中小M: m0=128, p=2
    {EncodeA5TilingCode(2, 0, 1), {{2048, 4096, -1, 2147483647, -1, 2147483647}}},        // 中M: m0=128, p=2
    {EncodeA5TilingCode(4, 0, 1), {{4096, 8192, -1, 2147483647, -1, 2147483647}}},        // 中大M: m0=128, p=4
    {EncodeA5TilingCode(4, 0, 1), {{8192, 16384, -1, 2147483647, -1, 2147483647}}},       // 大M: m0=128, p=4
    {EncodeA5TilingCode(8, 1, 1), {{16384, 2147483647, -1, 2147483647, -1, 2147483647}}}, // 极大M: m0=256, p=8
};

// A5 INT8 4-Rank mapping table
static std::map<int, std::vector<std::vector<int>>> g_allGatherV2MatmulA5FourRankINT8CodeMap = {
    {EncodeA5TilingCode(1, 0, 2), {{-1, 512, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(1, 0, 2), {{512, 1024, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(2, 0, 2), {{1024, 2048, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(2, 0, 2), {{2048, 4096, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(4, 0, 2), {{4096, 8192, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(4, 0, 2), {{8192, 16384, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(8, 1, 2), {{16384, 2147483647, -1, 2147483647, -1, 2147483647}}},
};

// A5 INT8 8-Rank mapping table
static std::map<int, std::vector<std::vector<int>>> g_allGatherV2MatmulA5EightRankINT8CodeMap = {
    {EncodeA5TilingCode(1, 0, 4), {{-1, 512, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(1, 0, 4), {{512, 1024, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(2, 0, 4), {{1024, 2048, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(2, 0, 4), {{2048, 4096, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(4, 0, 4), {{4096, 8192, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(4, 0, 4), {{8192, 16384, -1, 2147483647, -1, 2147483647}}},
    {EncodeA5TilingCode(8, 1, 4), {{16384, 2147483647, -1, 2147483647, -1, 2147483647}}},
};

// Default code values
constexpr int32_t ALLGATHERV2_MATMUL_A5_TWO_RANK_INT8_CODE_DEFAULT = EncodeA5TilingCode(2, 0, 1);
constexpr int32_t ALLGATHERV2_MATMUL_A5_FOUR_RANK_INT8_CODE_DEFAULT = EncodeA5TilingCode(2, 0, 2);
constexpr int32_t ALLGATHERV2_MATMUL_A5_EIGHT_RANK_INT8_CODE_DEFAULT = EncodeA5TilingCode(2, 0, 4);

struct TilingValue {
    int32_t value = -1;
    std::map<int, std::vector<std::vector<int>>> conditionMap = {};

    TilingValue() = default;
    TilingValue(int32_t v, std::map<int, std::vector<std::vector<int>>> m = {}) : value(v), conditionMap(std::move(m))
    {
    }
};

inline int32_t CeilDiv(int32_t num, int32_t div)
{
    return (div == 0) ? 0 : (num + div - 1) / div;
}

inline int32_t ClampValue(int32_t value, int32_t min, int32_t max)
{
    return std::max(min, std::min(value, max));
}

inline int32_t AlignUp(int32_t len, int32_t size)
{
    return (len + size - 1) & ~(size - 1);
}

inline bool IsMatrixAligned(const int64_t &m, const int64_t &n, const bool &transpose, int nElemAlign)
{
    return (transpose ? m : n) % nElemAlign == 0;
}

const std::map<ge::DataType, int64_t> D_TYPE_SIZE_MAP = {
    {ge::DT_BF16, 2}, {ge::DT_FLOAT16, 2}, {ge::DT_FLOAT, 4},
    {ge::DT_INT8, 1}, {ge::DT_INT32, 4},   {ge::DT_INT4, ge::GetSizeByDataType(ge::DT_INT4)},
};

void AllGatherV2DecodeTilingData(int32_t code, CoCTiling &tilingData)
{
    tilingData.commDataSplit = code & COMMDATASPLIT_MASK;
    code >>= COMMDATASPLIT_BNUM;

    tilingData.commNpuSplit = code & COMMNPUSPLIT_MASK;
    code >>= COMMNPUSPLIT_BNUM;

    tilingData.commDirect = code & COMMDIRECT_MASK;
    code >>= COMMDIRECT_BNUM;

    tilingData.ubMoveNum = (code & UBMOVENUM_MASK) * HALF_KBYTE;
    code >>= UBMOVENUM_BNUM;

    tilingData.pValue = code & PVALUE_MASK;
    code >>= PVALUE_BNUM;

    tilingData.swizzlCount = code & SWIZZLCOUNT_MASK;
    code >>= SWIZZLCOUNT_BNUM;

    tilingData.swizzlDirect = code & SWIZZLDIRECT_MASK;
    code >>= SWIZZLDIRECT_BNUM;

    int32_t m0Factor = code & M0_MASK;
    tilingData.m0 = m0Factor * DEFAULT_ROW + DEFAULT_ROW; // 0->128, 1->256
    tilingData.k0 = DEFAULT_COL;
    tilingData.n0 = tilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;

    tilingData.mLoop = CeilDiv(tilingData.m, tilingData.m0);
    tilingData.nLoop = CeilDiv(tilingData.n, tilingData.n0);
    tilingData.kLoop = CeilDiv(tilingData.k, tilingData.k0);
}

int32_t GetValueFromMKNConditionMap(int32_t m, int32_t k, int32_t n, int32_t defaultValue,
                                    std::map<int, std::vector<std::vector<int>>> conditionMap)
{
    constexpr int32_t CONDITION_M_ST = 0;
    constexpr int32_t CONDITION_M_END = 1;
    constexpr int32_t CONDITION_K_ST = 2;
    constexpr int32_t CONDITION_K_END = 3;
    constexpr int32_t CONDITION_N_ST = 4;
    constexpr int32_t CONDITION_N_END = 5;

    int32_t value = defaultValue;
    for (auto &item : conditionMap) {
        for (auto &condition : item.second) {
            bool inRange = m > condition[CONDITION_M_ST] && m <= condition[CONDITION_M_END] &&
                           k > condition[CONDITION_K_ST] && k <= condition[CONDITION_K_END] &&
                           n > condition[CONDITION_N_ST] && n <= condition[CONDITION_N_END];
            if (inRange) {
                return item.first;
            }
        }
    }
    return value;
}

void SetTilingParam(CoCTiling &cocTilingData, const std::map<int *, TilingValue> &TilingparamMap)
{
    int32_t m = cocTilingData.m;
    int32_t k = cocTilingData.k;
    int32_t n = cocTilingData.n;

    for (auto &item : TilingparamMap) {
        auto value = item.second.value;
        auto conditionMap = item.second.conditionMap;
        if (!conditionMap.empty()) {
            *item.first = GetValueFromMKNConditionMap(m, k, n, value, conditionMap);
        } else if (value != -1) {
            *item.first = value;
        }
    }

    cocTilingData.ubMoveNum = cocTilingData.ubMoveNum * HALF_KBYTE;
    if (cocTilingData.m0 >= DEFAULT_ROW) {
        cocTilingData.k0 = DEFAULT_COL;
        cocTilingData.n0 = cocTilingData.m0 == DEFAULT_ROW ? DEFAULT_COL : DEFAULT_ROW;
        cocTilingData.mLoop = CeilDiv(cocTilingData.m, cocTilingData.m0);
        cocTilingData.nLoop = CeilDiv(cocTilingData.n, cocTilingData.n0);
        cocTilingData.kLoop = CeilDiv(cocTilingData.k, cocTilingData.k0);
    }
}

void DealTilingParamByBuffSize(CoCTiling &cocTilingData)
{
    auto blockCount = MAX_BLOCK_COUNT;
    int maxPeerMemPerRank = (LCAL_BUFF_BYTES - FLAG_BUFF_BYTES) / INPUT_DTYPE / cocTilingData.rankSize / blockCount;
    int maxPValue = maxPeerMemPerRank / cocTilingData.m0 / cocTilingData.k0 / cocTilingData.kLoop;
    cocTilingData.pValue = ClampValue(cocTilingData.pValue, MIN_P_VALUE, maxPValue);

    // A5-specific: check if m0=256 is needed for very large M
    if (cocTilingData.m0 == DEFAULT_COL &&
        cocTilingData.pValue * cocTilingData.m0 * cocTilingData.k0 * cocTilingData.kLoop >= maxPeerMemPerRank) {
        cocTilingData.m0 = DEFAULT_ROW; // switch to m0=128
        cocTilingData.n0 = DEFAULT_COL;
        cocTilingData.mLoop = CeilDiv(cocTilingData.m, cocTilingData.m0);
        cocTilingData.nLoop = CeilDiv(cocTilingData.n, cocTilingData.n0);
    }
}

void AllGatherV2MatmulA5TwoRankINT8Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_A5_TWO_RANK_INT8_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingparamMap = {
        {&code, {ALLGATHERV2_MATMUL_A5_TWO_RANK_INT8_CODE_DEFAULT, g_allGatherV2MatmulA5TwoRankINT8CodeMap}}};
    SetTilingParam(cocTilingData, TilingparamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    cocTilingData.is91093 = 1; // A5 flag
    DealTilingParamByBuffSize(cocTilingData);
}

void AllGatherV2MatmulA5FourRankINT8Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_A5_FOUR_RANK_INT8_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingparamMap = {
        {&code, {ALLGATHERV2_MATMUL_A5_FOUR_RANK_INT8_CODE_DEFAULT, g_allGatherV2MatmulA5FourRankINT8CodeMap}}};
    SetTilingParam(cocTilingData, TilingparamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    cocTilingData.is91093 = 1;
    DealTilingParamByBuffSize(cocTilingData);
}

void AllGatherV2MatmulA5EightRankINT8Tiling(CoCTiling &cocTilingData)
{
    int32_t code = ALLGATHERV2_MATMUL_A5_EIGHT_RANK_INT8_CODE_DEFAULT;
    std::map<int *, TilingValue> TilingparamMap = {
        {&code, {ALLGATHERV2_MATMUL_A5_EIGHT_RANK_INT8_CODE_DEFAULT, g_allGatherV2MatmulA5EightRankINT8CodeMap}}};
    SetTilingParam(cocTilingData, TilingparamMap);

    AllGatherV2DecodeTilingData(code, cocTilingData);

    cocTilingData.lenPerLoop = cocTilingData.ubMoveNum * cocTilingData.commDataSplit;
    cocTilingData.is91093 = 1;
    DealTilingParamByBuffSize(cocTilingData);
}

void SetTilingDataInner(CoCTiling &cocTilingData, int64_t rankSize)
{
    cocTilingData.rankSize = static_cast<int32_t>(rankSize);

    if (rankSize == 2) {
        AllGatherV2MatmulA5TwoRankINT8Tiling(cocTilingData);
    } else if (rankSize == 4) {
        AllGatherV2MatmulA5FourRankINT8Tiling(cocTilingData);
    } else if (rankSize == 8) {
        AllGatherV2MatmulA5EightRankINT8Tiling(cocTilingData);
    } else {
        // Default fallback for other rank sizes
        AllGatherV2MatmulA5EightRankINT8Tiling(cocTilingData);
    }
}

void GetUsrWorkSpaceSize(uint32_t nElemAlign, uint32_t elementSize, uint64_t &userWorkSpaceSize, int64_t rankSize,
                         Mc2Tiling::AllGatherMatmulTilingDataInt8 *tilingData)
{
    auto &info = tilingData->allGatherMatmulInfo;
    const auto &cocTiling = tilingData->cocTiling;

    bool hasAAlign = (!IsMatrixAligned(info.M, info.K, info.isTransposeX1, nElemAlign) && info.M != NUMSIZE_ONE);
    bool hasBAlign = !IsMatrixAligned(info.K, info.N, info.isTransposeX2, nElemAlign);

    int32_t mAlign = AlignUp(info.M, nElemAlign);
    int32_t kAlign = AlignUp(info.K, nElemAlign);
    int32_t nAlign = AlignUp(info.N, nElemAlign);

    info.aAlignSize = 0;
    info.bAlignSize = 0;
    info.hasAAlign = hasAAlign;
    info.hasBAlign = hasBAlign;

    if (info.hasAAlign) {
        if (elementSize != 0) {
            info.aAlignSize =
                static_cast<uint64_t>((info.isTransposeX1 ? info.K * mAlign : info.M * kAlign) * elementSize);
        } else {
            info.aAlignSize = static_cast<uint64_t>((info.isTransposeX1 ? info.K * mAlign : info.M * kAlign) / 2);
        }
        userWorkSpaceSize += info.aAlignSize;
    }

    if (info.hasBAlign) {
        if (elementSize != 0) {
            info.bAlignSize =
                static_cast<uint64_t>((info.isTransposeX2 ? info.N * kAlign : info.K * nAlign) * elementSize);
        } else {
            info.bAlignSize = static_cast<uint64_t>((info.isTransposeX2 ? info.N * kAlign : info.K * nAlign) / 2);
        }
        userWorkSpaceSize += info.bAlignSize;
    }

    if (info.quantFlag) {
        info.accumWorkSpacePingPong = (info.M < MAX_BLOCK_COUNT * cocTiling.m0 * cocTiling.pValue);
        int32_t workspaceM = info.accumWorkSpacePingPong ? info.M : MAX_BLOCK_COUNT * cocTiling.m0 * cocTiling.pValue;
        userWorkSpaceSize += static_cast<uint64_t>(workspaceM * info.N * rankSize * sizeof(int32_t));
    }

    if (info.dequantType == DequantType::PER_TOKEN) {
        userWorkSpaceSize += static_cast<uint64_t>(info.M * rankSize * sizeof(float32_t));
    }
}

} // namespace

AllGatherMatmulInt8Tiling::AllGatherMatmulInt8Tiling(gert::TilingContext *context)
    : TilingBaseClass(context), opName_(context->GetNodeName())
{
}

bool AllGatherMatmulInt8Tiling::IsCapable()
{
    if (x1Type_ != ge::DT_INT8 || x2Type_ != ge::DT_INT8) {
        OP_LOGE(opName_, "x1Type=%s, x2Type=%s, Skip A5 INT8 tiling as dtype not support.",
                ge::TypeUtils::DataTypeToSerialString(x1Type_).c_str(),
                ge::TypeUtils::DataTypeToSerialString(x2Type_).c_str());
        return false;
    }
    return true;
}

ge::graphStatus AllGatherMatmulInt8Tiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "fail to get platform info"),
                    return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

    uint32_t batchMode = 1U;
    auto ret = context_->SetScheduleMode(batchMode);
    GE_ASSERT_GRAPH_SUCCESS(ret);

    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    numBlocks_ = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(numBlocks_);

    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    totalUbSize_ = static_cast<uint32_t>(ubSize);
    aivNum_ = static_cast<uint32_t>(aivNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInt8Tiling::GetShapeAttrsInfo()
{
    auto x1Tensor = context_->GetInputTensor(X1_INDEX);
    auto x2Tensor = context_->GetInputTensor(X2_INDEX);
    OP_TILING_CHECK(x1Tensor == nullptr || x2Tensor == nullptr,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "x1 or x2 tensor is nullptr"), return ge::GRAPH_FAILED);

    x1Type_ = x1Tensor->GetDataType();
    x2Type_ = x2Tensor->GetDataType();

    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "attrs is null"),
                    return ge::GRAPH_FAILED);

    auto groupPtr = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    OP_TILING_CHECK(groupPtr == nullptr || strlen(groupPtr) == 0,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "group is invalid"), return ge::GRAPH_FAILED);

    auto isTransposeX1 = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_X1);
    auto isTransposeX2 = attrs->GetAttrPointer<bool>(ATTR_IS_TRANS_X2);
    OP_TILING_CHECK(isTransposeX2 == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "isTransposeX2 is invalid"),
                    return ge::GRAPH_FAILED);

    group_ = groupPtr;
    isTransposeX1_ = (isTransposeX1 != nullptr && *isTransposeX1) ? *isTransposeX1 : false;
    isTransposeX2_ = *isTransposeX2;

    OP_TILING_CHECK(!mc2tiling::GetRankSize(opName_, group_, rankSize_),
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "GetRankSize failed"), return ge::GRAPH_FAILED);

    const auto aShape = context_->GetInputShape(X1_INDEX);
    const auto bShape = context_->GetInputShape(X2_INDEX);
    M_ = aShape->GetStorageShape().GetDim(0);
    K_ = aShape->GetStorageShape().GetDim(1);
    N_ = bShape->GetStorageShape().GetDim(1);

    if (isTransposeX2_) {
        N_ = bShape->GetStorageShape().GetDim(0);
    }

    ge::Format formatB =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(X2_INDEX)->GetStorageFormat()));
    if (formatB == ge::FORMAT_FRACTAL_NZ) {
        const auto cShape = context_->GetOutputShape(Y_INDEX);
        N_ = cShape->GetOriginShape().GetDim(1);
    }

    auto yTensorDesc = context_->GetOutputDesc(Y_INDEX);
    yType_ = yTensorDesc->GetDataType();

    elementSize_ = static_cast<uint32_t>(D_TYPE_SIZE_MAP.at(x1Type_));
    nElemAlign_ = HALF_KBYTE / elementSize_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInt8Tiling::CheckInput()
{
    OP_TILING_CHECK(x1Type_ != ge::DT_INT8 || x2Type_ != ge::DT_INT8,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "x1 dtype and x2 dtype must be INT8"),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SUPPORT_RANKSIZE.find(rankSize_) == SUPPORT_RANKSIZE.end(),
                    VECTOR_INNER_ERR_REPORT_TILING(
                        opName_, "Unsupported rankSize %ld. Supported: 2, 4, 8, 16, 32, 64, 128, 256", rankSize_),
                    return ge::GRAPH_FAILED);

    OP_TILING_CHECK(yType_ != ge::DT_BF16 && yType_ != ge::DT_FLOAT16,
                    VECTOR_INNER_ERR_REPORT_TILING(opName_, "Output dtype must be BF16/FP16 for INT8"),
                    return ge::GRAPH_FAILED);

    auto x1ScaleTensor = context_->GetInputTensor(X1_SCALE_INDEX);
    auto x2ScaleTensor = context_->GetInputTensor(X2_SCALE_INDEX);
    OP_TILING_CHECK(x2ScaleTensor == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "x2Scale is required for INT8"),
                    return ge::GRAPH_FAILED);

    auto x2ScaleType = x2ScaleTensor->GetDataType();
    if (x2ScaleType == ge::DT_INT64 && yType_ == ge::DT_FLOAT16) {
        isX2ScaleTypeInt64_ = true;
    } else if (x2ScaleType != ge::DT_FLOAT) {
        VECTOR_INNER_ERR_REPORT_TILING(opName_, "Invalid x2Scale dtype for INT8");
        return ge::GRAPH_FAILED;
    }

    dequantType_ = 0;
    if (x1ScaleTensor != nullptr && x1ScaleTensor->GetDataType() == ge::DT_FLOAT) {
        dequantType_ = 1;
    }

    const gert::StorageShape *matrixBias = context_->GetOptionalInputShape(BIAS_INDEX);
    OP_TILING_CHECK(matrixBias != nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "bias must be nullptr for INT8"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInt8Tiling::SetTilingData()
{
    auto rawTilingData = context_->GetRawTilingData();
    OP_TILING_CHECK(rawTilingData == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "Fail to get rawTilingData"),
                    return ge::GRAPH_FAILED);

    tilingData_ = context_->GetTilingData<Mc2Tiling::AllGatherMatmulTilingDataInt8>();
    OP_TILING_CHECK(tilingData_ == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "Fail to get tilingData"),
                    return ge::GRAPH_FAILED);

    auto &info = tilingData_->allGatherMatmulInfo;
    auto &cocTiling = tilingData_->cocTiling;

    info.M = M_;
    info.K = K_;
    info.N = N_;
    info.worldSize = static_cast<uint32_t>(rankSize_);
    info.rankId = 0;
    info.aivNum = aivNum_;
    info.totalUbSize = totalUbSize_;
    info.isTransposeX1 = isTransposeX1_;
    info.isTransposeX2 = isTransposeX2_;
    info.quantFlag = true;
    info.is910C = true;
    info.isX2ScaleTypeInt64 = isX2ScaleTypeInt64_;
    info.dequantType = static_cast<DequantType>(dequantType_);

    cocTiling.m = static_cast<int32_t>(M_);
    cocTiling.k = static_cast<int32_t>(K_);
    cocTiling.n = static_cast<int32_t>(N_);
    cocTiling.numBlocks = static_cast<int32_t>(numBlocks_);

    SetTilingDataInner(cocTiling, rankSize_);

    // Calculate alignment info
    bool hasAAlign = (!IsMatrixAligned(info.M, info.K, info.isTransposeX1, nElemAlign_) && info.M != NUMSIZE_ONE);
    bool hasBAlign = !IsMatrixAligned(info.K, info.N, info.isTransposeX2, nElemAlign_);

    int32_t mAlign = AlignUp(info.M, nElemAlign_);
    int32_t kAlign = AlignUp(info.K, nElemAlign_);
    int32_t nAlign = AlignUp(info.N, nElemAlign_);

    info.hasAAlign = hasAAlign;
    info.hasBAlign = hasBAlign;
    info.aAlignSize = 0;
    info.bAlignSize = 0;

    if (info.hasAAlign) {
        info.aAlignSize =
            static_cast<uint64_t>((info.isTransposeX1 ? info.K * mAlign : info.M * kAlign) * elementSize_);
    }
    if (info.hasBAlign) {
        info.bAlignSize =
            static_cast<uint64_t>((info.isTransposeX2 ? info.N * kAlign : info.K * nAlign) * elementSize_);
    }

    OP_LOGI(opName_, "A5 New Tiling: m0=%d, n0=%d, swizzlCount=%d, swizzlDirect=%d(Zn), is91093=%d", cocTiling.m0,
            cocTiling.n0, cocTiling.swizzlCount, cocTiling.swizzlDirect, cocTiling.is91093);

    rawTilingData->SetDataSize(sizeof(Mc2Tiling::AllGatherMatmulTilingDataInt8));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInt8Tiling::SetMc2Hcomm()
{
    uint32_t opType = 6;
    std::string algConfig = "AllGather=level0:fullmesh";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group_, opType, algConfig);
    mc2CcTilingConfig.SetCommEngine(AIV_TYPE);

    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData_->mc2InitTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig GetTiling mc2InitTiling failed"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(mc2CcTilingConfig.GetTiling(tilingData_->mc2CcTiling) != 0,
                    OP_LOGE(opName_, "mc2CcTilingConfig GetTiling mc2CcTiling failed"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInt8Tiling::DoOpTiling()
{
    GE_ASSERT_GRAPH_SUCCESS(CheckInput());
    GE_ASSERT_GRAPH_SUCCESS(SetTilingData());
    GE_ASSERT_GRAPH_SUCCESS(SetMc2Hcomm());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInt8Tiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus AllGatherMatmulInt8Tiling::GetWorkspaceSize()
{
    size_t *workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, VECTOR_INNER_ERR_REPORT_TILING(opName_, "get workspace failed"),
                    return ge::GRAPH_FAILED);

    uint64_t userWorkSpaceSize = 0;
    GetUsrWorkSpaceSize(nElemAlign_, elementSize_, userWorkSpaceSize, rankSize_, tilingData_);

    workspaces[0] = SYSTEM_NEED_WORKSPACE + userWorkSpaceSize;

    return ge::GRAPH_SUCCESS;
}

uint64_t AllGatherMatmulInt8Tiling::GetTilingKey() const
{
    bool inputIsBf16Fp16 = false;
    uint8_t outputType = static_cast<uint8_t>(0);
    uint8_t scaleType = static_cast<uint8_t>(2);
    uint8_t quanMmMode = static_cast<uint8_t>(2);
    uint64_t tilingKey = GET_TPL_TILING_KEY(inputIsBf16Fp16, isTransposeX2_, outputType, quanMmMode, scaleType);
    return tilingKey;
}

ge::graphStatus AllGatherMatmulInt8Tiling::PostTiling()
{
    context_->SetTilingKey(GetTilingKey());
    OP_LOGI(opName_, "A5New TilingKey=%lu", GetTilingKey());
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AllGatherMatmulInt8Tiling::PrintTilingData()
{
    if (tilingData_ == nullptr) {
        OP_LOGE(opName_, "PrintTilingData: tilingData_ is nullptr");
        return;
    }

    auto &info = tilingData_->allGatherMatmulInfo;
    auto &coc = tilingData_->cocTiling;

    OP_LOGD(opName_, "========== A5 New Tiling Parameters (Zn Mode Fixed) ==========");
    OP_LOGD(opName_, "  INPUT_DTYPE: %d (INT8专用)", INPUT_DTYPE);
    OP_LOGD(opName_, "  HALF_KBYTE: %d", HALF_KBYTE);
    OP_LOGD(opName_, "  elementSize_: %u", elementSize_);
    OP_LOGD(opName_, "  nElemAlign_: %u", nElemAlign_);

    OP_LOGD(opName_, "========== AllGatherMatmulAIVModeInfo Parameters ==========");
    OP_LOGD(opName_, "  M: %u, K: %u, N: %u", info.M, info.K, info.N);
    OP_LOGD(opName_, "  worldSize: %u, rankId: %u", info.worldSize, info.rankId);
    OP_LOGD(opName_, "  aivNum: %u, totalUbSize: %u", info.aivNum, info.totalUbSize);
    OP_LOGD(opName_, "  isTransposeX1: %d, isTransposeX2: %d", info.isTransposeX1, info.isTransposeX2);
    OP_LOGD(opName_, "  hasAAlign: %d, hasBAlign: %d", info.hasAAlign, info.hasBAlign);
    OP_LOGD(opName_, "  aAlignSize: %lu, bAlignSize: %lu", info.aAlignSize, info.bAlignSize);
    OP_LOGD(opName_, "  quantFlag: %d, is910C: %d", info.quantFlag, info.is910C);
    OP_LOGD(opName_, "  dequantType: %d, accumWorkSpacePingPong: %d", static_cast<int>(info.dequantType),
            info.accumWorkSpacePingPong);

    OP_LOGD(opName_, "========== CoCTiling Parameters ==========");
    OP_LOGD(opName_, "  m: %d, k: %d, n: %d", coc.m, coc.k, coc.n);
    OP_LOGD(opName_, "  numBlocks: %d, rankSize: %d", coc.numBlocks, coc.rankSize);
    OP_LOGD(opName_, "  m0: %d, k0: %d, n0: %d", coc.m0, coc.k0, coc.n0);
    OP_LOGD(opName_, "  mLoop: %d, kLoop: %d, nLoop: %d", coc.mLoop, coc.kLoop, coc.nLoop);
    OP_LOGD(opName_, "  swizzlCount: %d (固定), swizzlDirect: %d (Zn固定)", coc.swizzlCount, coc.swizzlDirect);
    OP_LOGD(opName_, "  pValue: %d, ubMoveNum: %d", coc.pValue, coc.ubMoveNum);
    OP_LOGD(opName_, "  commNpuSplit: %d, commDataSplit: %d", coc.commNpuSplit, coc.commDataSplit);
    OP_LOGD(opName_, "  lenPerLoop: %d, is91093: %d", coc.lenPerLoop, coc.is91093);
    OP_LOGD(opName_, "========== PrintTilingData End ==========");
}

REGISTER_TILING_TEMPLATE_WITH_ARCH(AllGatherMatmulV2, AllGatherMatmulInt8Tiling,
                                   static_cast<int32_t>(NpuArch::DAV_3510), 4); // Priority=4, higher than a5.cpp(3)