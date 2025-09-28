/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_flash_attention_score.h"
#include "flash_attention_score.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/pad.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/transpose.h"
#include "opdev/common_types.h"
#include "opdev/fast_vector.h"
#include "opdev/op_errno.h"
#include "opdev/op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const int64_t PAD_BASIC_BLOCK = 16;
static const int64_t PAD_LOWER_BOUND_196 = 196;
static const int64_t PAD_ALIGN_128 = 128;
static const int64_t PAD_ALIGN_SPL_SHAPE = 448;
static const int64_t MAX_STRIDE_S1 = 65535;
static const uint64_t DIM_NUM_4 = 4;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_2 = 2;
static const int64_t HEAD_DIM_MAX = 768;
static const int64_t PSE_TYPE_V1 = 1; // add and mul
static const int64_t PSE_INNER_MUL_ADD = 2;
static const int64_t PSE_INNER_MUL_ADD_SQRT = 3;
static const int64_t HEAD_DIM_64 = 64;
static const int64_t HEAD_DIM_72 = 72;
static const int64_t HEAD_DIM_80 = 80;
static const int64_t HEAD_DIM_88 = 88;
static const int64_t HEAD_DIM_128 = 88;
static const int64_t TND_UNPAD_MAX_S2 = 1024;
static const int64_t TND_UNPAD_MAX_S1_SUM = 160 * 1024;
static const int64_t TND_UNPAD_MAX_DDIM = 96;

static const int64_t FRACTAL_NUM = 16L;
static const int64_t D_SPECIFIC_SIZE_96 = 96L;
static const int64_t S2_REUSE_SIZE_512 = 512L;
static const int64_t s2sizeLimitMin = 1024;
static const int64_t SAMEAB_D_LIMIT_128 = 128L;
static const int64_t SAMEAB_D_LIMIT_196 = 196L;

static const int64_t MAX_VAR_LEN_SEQ_LEN = 20000;

struct AxesInfo {
    int64_t b;
    int64_t n1;
    int64_t n2;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dk;
    int64_t dv;
};

enum class InputLayout {
    BSND,
    SBH,
    BNSD,
    BSH,
    TND
};

struct FaShapeInfo {
    AxesInfo axes;

    InputLayout inputLayout;
    string l0InputLayoutStr;

    uint64_t dimNum = 0;
    uint64_t padNum = 0;
    uint64_t padNumv = 0;

    FVector<int64_t, DIM_NUM_4> perm_in;
    FVector<int64_t, DIM_NUM_4> perm_out;
    FVector<int64_t, DIM_NUM_4> reshapedQueryShape;
    FVector<int64_t, DIM_NUM_4> reshapedKeyShape;
    FVector<int64_t, DIM_NUM_4> reshapedValueBefore;

    bool needPad = false;
    bool needTranspose = false;
    bool needReshape = false;
    bool needPadValue = false;
};

static void AnalysisAxisForBsh(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BSH;
    shapeInfo.l0InputLayoutStr = "BSH";
    uint64_t dSize = qShape[2] / shapeInfo.axes.n1;
    shapeInfo.axes.d = dSize;
    if (dSize == 0) {
        return;
    }
    shapeInfo.axes.b = qShape[0];
    shapeInfo.axes.n2 = kShape[2] / dSize;
    shapeInfo.axes.s1 = qShape[1];
    shapeInfo.axes.s2 = kShape[1];
    shapeInfo.axes.dk = kShape[2] / shapeInfo.axes.n2;
    shapeInfo.axes.dv = vShape[2] / shapeInfo.axes.n2;
}

static void AnalysisAxisForBsnd(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BSND;
    shapeInfo.l0InputLayoutStr = "BSND";
    shapeInfo.axes.b = qShape[0];
    shapeInfo.axes.n2 = kShape[2];
    shapeInfo.axes.s1 = qShape[1];
    shapeInfo.axes.s2 = kShape[1];
    shapeInfo.axes.d = qShape[3];
    shapeInfo.axes.dk = kShape[3];
    shapeInfo.axes.dv = vShape[3];
}

static void AnalysisAxisForTnd(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::TND;
    shapeInfo.l0InputLayoutStr = "TND";
    shapeInfo.axes.n2 = kShape[1];
    shapeInfo.axes.d = qShape[DIM_NUM_2];
    shapeInfo.axes.dk = kShape[DIM_NUM_2];
    shapeInfo.axes.dv = vShape[DIM_NUM_2];
}

static void AnalysisAxisForSbh(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::SBH;
    shapeInfo.l0InputLayoutStr = "SBH";
    uint64_t dSize = qShape[2] / shapeInfo.axes.n1;
    shapeInfo.axes.d = dSize;
    if (dSize == 0) {
        return;
    }
    shapeInfo.axes.b = qShape[1];
    shapeInfo.axes.n2 = kShape[2] / dSize;
    shapeInfo.axes.s1 = qShape[0];
    shapeInfo.axes.s2 = kShape[0];
    shapeInfo.axes.dk = kShape[2] / shapeInfo.axes.n2;
    shapeInfo.axes.dv = vShape[2] / shapeInfo.axes.n2;
}

static void AnalysisAxisForBnsd(const Shape &qShape, const Shape &kShape, const Shape &vShape, FaShapeInfo &shapeInfo)
{
    shapeInfo.inputLayout = InputLayout::BNSD;
    shapeInfo.l0InputLayoutStr = "BNSD";
    shapeInfo.axes.b = qShape[0];
    shapeInfo.axes.n2 = kShape[1];
    shapeInfo.axes.s1 = qShape[2];
    shapeInfo.axes.s2 = kShape[2];
    shapeInfo.axes.d = qShape[3];
    shapeInfo.axes.dk = kShape[3];
    shapeInfo.axes.dv = vShape[3];
}

static aclnnStatus AnalysisAxis(const aclTensor *query, const aclTensor *key, const aclTensor *value, 
                                const char *inputLayout, int64_t headNum, FaShapeInfo &shapeInfo)
{
    Shape qShape = query->GetViewShape();
    Shape kShape = key->GetViewShape();
    Shape vShape = value->GetViewShape();
    shapeInfo.dimNum = qShape.GetDimNum();

    // 记录轴的长度 b, n2, g, s1, s2, d
    // H1等于N1*D, H2等于N2*D
    // N1等于g*N2
    shapeInfo.axes.n1 = headNum;
    std::string inputLayoutStr = op::ToString(inputLayout).GetString();
    if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "BSH") {
        // query: (B,S1,N1*D)
        // key/value: (B,S2,N2*D)
        AnalysisAxisForBsh(qShape, kShape, vShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_4 && inputLayoutStr == "BSND") {
        // query: (B,S1,N1,D)
        // key/value: (B,S2,N2,D)
        AnalysisAxisForBsnd(qShape, kShape, vShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "SBH") {
        // query: (S1,B,N1*D)
        // key/value: (S2,B,N2*D)
        AnalysisAxisForSbh(qShape, kShape, vShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_4 && inputLayoutStr == "BNSD") {
        // query: (B,N1,S1,D)
        // key/value: (B,N2,S2,D)
        AnalysisAxisForBnsd(qShape, kShape, vShape, shapeInfo);
    } else if (shapeInfo.dimNum == DIM_NUM_3 && inputLayoutStr == "TND") {
        // query: (T,N1,D)
        // key/value: (T,N2,D)
        AnalysisAxisForTnd(qShape, kShape, vShape, shapeInfo);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "not support input_layout %s with dim_num %lu", inputLayout, shapeInfo.dimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.d != shapeInfo.axes.dk) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "qD and kD should be same, but got qD=%ld kD=%ld", shapeInfo.axes.d, 
            shapeInfo.axes.dk);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.axes.d < shapeInfo.axes.dv) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only support kD >= vD, but got kD=%ld vD=%ld", shapeInfo.axes.d, 
            shapeInfo.axes.dv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static void SetShapeInfoForBshBsnd(int64_t alignedH1Size, FaShapeInfo &shapeInfo)
{
    if (alignedH1Size > MAX_STRIDE_S1) {
        shapeInfo.needTranspose = true;
        shapeInfo.needReshape = true;
        shapeInfo.l0InputLayoutStr = "BNSD";

        // B,S,N,D -> B,N,S,D
        shapeInfo.perm_in.assign({0, 2, 1, 3});
        // B,N,S,D -> B,S,N,D
        shapeInfo.perm_out.assign(shapeInfo.perm_in.cbegin(), shapeInfo.perm_in.cend());
    }
    if (shapeInfo.needPad) {
        shapeInfo.needReshape = true;
    }

    if (shapeInfo.inputLayout == InputLayout::BSND) {
        shapeInfo.needReshape = false;
    }
    if (shapeInfo.needReshape) {
        if (!shapeInfo.needTranspose) {
            shapeInfo.l0InputLayoutStr = "BSND";
        }
        shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.b, shapeInfo.axes.s1, shapeInfo.axes.n1, shapeInfo.axes.d});
        shapeInfo.reshapedKeyShape.assign(
            {shapeInfo.axes.b, shapeInfo.axes.s2, shapeInfo.axes.n2, shapeInfo.axes.d});
        shapeInfo.reshapedValueBefore.assign({shapeInfo.axes.b, shapeInfo.axes.s2, shapeInfo.axes.n2, shapeInfo.axes.dv});
    }
}

static void SetShapeInfoForSbh(int64_t alignedH1Size, FaShapeInfo &shapeInfo)
{
    if (shapeInfo.axes.b * alignedH1Size > MAX_STRIDE_S1) {
        shapeInfo.needTranspose = true;
        shapeInfo.needReshape = true;
        shapeInfo.l0InputLayoutStr = "BNSD";

        // S,B,N,D -> B,N,S,D
        shapeInfo.perm_in.assign({1, 2, 0, 3});
        // B,N,S,D -> S,B,N,D
        shapeInfo.perm_out.assign({2, 0, 1, 3});
    }
    if (shapeInfo.needPad) {
        shapeInfo.needReshape = true;
    }

    if (shapeInfo.needReshape) {
        if (!shapeInfo.needTranspose) {
            shapeInfo.l0InputLayoutStr = "SBH";
        }
        shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.s1, shapeInfo.axes.b, shapeInfo.axes.n1, shapeInfo.axes.d});
        shapeInfo.reshapedKeyShape.assign(
            {shapeInfo.axes.s2, shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.d});
        shapeInfo.reshapedValueBefore.assign({shapeInfo.axes.s2, shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.dv});
    }
}

static int64_t GetSumIntArrayMaxValue(const aclIntArray *intArrayValue)
{
    // 获取targetLengthsList中的最大值
    int64_t maxLength = 0;
    int64_t tmpMaxLength = 0;
    if (intArrayValue->Size() == 1) {
        maxLength = static_cast<int64_t>((*intArrayValue)[0]);
        return maxLength;
    }
    maxLength = static_cast<int64_t>((*intArrayValue)[0]);
    for (size_t i = 1; i < intArrayValue->Size(); ++i) {
        tmpMaxLength = static_cast<int64_t>((*intArrayValue)[i]) - static_cast<int64_t>((*intArrayValue)[i - 1]);
        if (tmpMaxLength > maxLength) {
            maxLength = tmpMaxLength;
        }
    }
    return maxLength;
}

static int64_t getSeqLenQSum(const aclIntArray *actualSeqQLenOptional) {
    if (actualSeqQLenOptional->Size() < 1) {
        return 0;
    }
    int64_t sQLenSum = 0;
    for (int64_t i = actualSeqQLenOptional->Size() - 1; i >= 0; --i) {
        sQLenSum = static_cast<int64_t>((*actualSeqQLenOptional)[i]);
        if (sQLenSum > 0) {
            break;
        }
    }
    return sQLenSum;
}

static bool IsNeedPad(FaShapeInfo &shapeInfo, const aclIntArray *actualSeqQLenOptional,
                      const aclIntArray *actualSeqKvLenOptional)
{
    if (shapeInfo.inputLayout != InputLayout::BNSD &&
         shapeInfo.inputLayout != InputLayout::TND && shapeInfo.axes.n1 == shapeInfo.axes.n2 &&
         shapeInfo.needTranspose == false) {
        if (shapeInfo.axes.d == HEAD_DIM_72 || shapeInfo.axes.d == HEAD_DIM_88) {
            shapeInfo.padNum = 0;
        }
        if (shapeInfo.axes.dv == HEAD_DIM_72 || shapeInfo.axes.dv == HEAD_DIM_88) {
            shapeInfo.padNumv = 0;
        }
        if ((shapeInfo.axes.d == HEAD_DIM_72 || shapeInfo.axes.d == HEAD_DIM_88) 
                && (shapeInfo.axes.dv == HEAD_DIM_72 || shapeInfo.axes.dv == HEAD_DIM_88)) {
            return false;
        }
        return true;
    }

    if (shapeInfo.inputLayout == InputLayout::TND) {
        if (shapeInfo.axes.d >= TND_UNPAD_MAX_DDIM) {
            return true;
        }
        int64_t sKvLenMax = 0;
        int64_t sQLenSum = 0;
        if (actualSeqQLenOptional != nullptr && actualSeqKvLenOptional != nullptr &&
            actualSeqQLenOptional->Size() == actualSeqKvLenOptional->Size()) {
            sKvLenMax = GetSumIntArrayMaxValue(actualSeqKvLenOptional);
            sQLenSum = getSeqLenQSum(actualSeqQLenOptional);
        }

        if (sKvLenMax == 0 || sQLenSum == 0) {
            // 走原来逻辑是否pad
            OP_LOGD("Fa aclnn TND case sKvLenMax(%ld) or sQLenSum(%ld) is 0.", sKvLenMax, sQLenSum);
            return true;
        }

        if ((sKvLenMax <= TND_UNPAD_MAX_S2) && (sQLenSum < TND_UNPAD_MAX_S1_SUM)) {
            // 去除pad
            OP_LOGD("Fa aclnn TND case do not do pad dimD operation.");
            shapeInfo.padNum = 0;
            shapeInfo.padNumv = 0;
            return false;
        }
    }
    return true;
}

static bool IsCapableSameAB(const FaShapeInfo &shapeInfo, const aclTensor *query)
{
    if (query->GetDataType() == op::DataType::DT_FLOAT) {
        return false;
    }
    if (((shapeInfo.axes.d + shapeInfo.padNum) % FRACTAL_NUM != 0 || shapeInfo.axes.d == D_SPECIFIC_SIZE_96) && (shapeInfo.axes.s2 >= S2_REUSE_SIZE_512)) {
        return true;
    }
    if (shapeInfo.axes.s2 > s2sizeLimitMin && shapeInfo.axes.d > SAMEAB_D_LIMIT_128 && shapeInfo.axes.d < SAMEAB_D_LIMIT_196) {
        return true;
    }
    return false;
}

static bool isSupportMLA(const FaShapeInfo &shapeInfo, const aclTensor *query)
{
    // TND模板支持MLA
    if(shapeInfo.inputLayout == InputLayout::TND) {
        return true;
    }
    // SameAB模板支持MLA
    if(IsCapableSameAB(shapeInfo, query)) {
        return true;
    }
    // S1S2模板支持MLA
    if (shapeInfo.axes.s2 > s2sizeLimitMin) {
        return true;
    }
    return false;
}

static aclnnStatus InputDtypeCheck(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                   const aclTensor *realShiftOptional, int64_t pseType)
{
    auto vDtype = value->GetDataType();
    auto kDtype = key->GetDataType();
    auto qDtype = query->GetDataType();
    if (qDtype != kDtype || kDtype != vDtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of query[%s], key[%s], value[%s] are not equal.",
                op::ToString(DataType(qDtype)).GetString(), op::ToString(DataType(kDtype)).GetString(),
                op::ToString(DataType(vDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (pseType == PSE_INNER_MUL_ADD || pseType == PSE_INNER_MUL_ADD_SQRT) {
        // Inner pse alibi, dtype must be fp32
        if (realShiftOptional == nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When pseType is 2 or 3, pseShape cannot be null.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        auto pseDtype = realShiftOptional->GetDataType();
        if (pseDtype != op::DataType::DT_FLOAT) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The data type %s of pse is not invalid in pse type 2 or 3 mode, It must be float32",
                    op::ToString(DataType(pseDtype)).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
    if (realShiftOptional != nullptr) {
        auto pseDtype = realShiftOptional->GetDataType();
        if (pseDtype != qDtype) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The data type %s of pse is not equal to the data type %s of query, key and value.",
                    op::ToString(DataType(pseDtype)).GetString(), op::ToString(DataType(qDtype)).GetString());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus AnalysisInput(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                 char *inputLayout, int64_t headNum, FaShapeInfo &shapeInfo,
                                 const aclIntArray *actualSeqQLenOptional = nullptr,
                                 const aclIntArray *actualSeqKvLenOptional = nullptr)
{
    if (headNum <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "head_num must > 0, but got %ld", headNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(
        AnalysisAxis(query, key, value, inputLayout, headNum, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (shapeInfo.axes.d > HEAD_DIM_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Head dim must <= 768, but got %ld", shapeInfo.axes.d);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if ((actualSeqQLenOptional != nullptr && actualSeqKvLenOptional != nullptr) && 
                    (actualSeqQLenOptional->Size() > MAX_VAR_LEN_SEQ_LEN || 
                    actualSeqKvLenOptional->Size() > MAX_VAR_LEN_SEQ_LEN)) {
        OP_LOGW("The input parameter exceeds the operator range, unknown risks exist: " 
               "actualSeqQLen size and actualSeqKvLen size must <= 20000, but got %lu.", actualSeqQLenOptional->Size());
    }

    if (shapeInfo.axes.n2 == 0 || shapeInfo.axes.d == 0) {
        return ACLNN_SUCCESS;
    }

    if (shapeInfo.inputLayout != InputLayout::TND &&
        (shapeInfo.axes.b == 0 || shapeInfo.axes.s1 == 0 || shapeInfo.axes.s2 == 0)) {
        return ACLNN_SUCCESS;
    }

    int64_t alignDim = (shapeInfo.axes.d < PAD_LOWER_BOUND_196 || shapeInfo.axes.d == PAD_ALIGN_SPL_SHAPE) ?
                        PAD_BASIC_BLOCK : PAD_ALIGN_128;
    if (shapeInfo.axes.d % alignDim != 0 || shapeInfo.axes.dv % alignDim != 0) {
        shapeInfo.needPad = true;
        shapeInfo.padNum = (shapeInfo.axes.d + alignDim - 1) / alignDim * alignDim - shapeInfo.axes.d;
        shapeInfo.padNumv = (shapeInfo.axes.dv + alignDim - 1) / alignDim * alignDim - shapeInfo.axes.dv;
    }

    // 硬件亲和dim适配
    if ((shapeInfo.axes.d != shapeInfo.axes.dv) &&
        (shapeInfo.axes.d == HEAD_DIM_128 || shapeInfo.axes.d == HEAD_DIM_64 || shapeInfo.axes.d == HEAD_DIM_80)) {
        shapeInfo.needPad = true;
        shapeInfo.padNumv = shapeInfo.padNum + shapeInfo.axes.d - shapeInfo.axes.dv;
    }

    int64_t alignedH1Size = shapeInfo.axes.n1 * (shapeInfo.axes.d + shapeInfo.padNum);
    if (shapeInfo.inputLayout == InputLayout::BSH || shapeInfo.inputLayout == InputLayout::BSND) {
        SetShapeInfoForBshBsnd(alignedH1Size, shapeInfo);
    } else if (shapeInfo.inputLayout == InputLayout::SBH) {
        SetShapeInfoForSbh(alignedH1Size, shapeInfo);
    }

    if (!IsNeedPad(shapeInfo, actualSeqQLenOptional, actualSeqKvLenOptional)) {
        shapeInfo.needPad = false;
        shapeInfo.needReshape = false;
        if (shapeInfo.inputLayout == InputLayout::BSH) {
            shapeInfo.l0InputLayoutStr = "BSH";
        }
    }

    if ((shapeInfo.axes.d != shapeInfo.axes.dv) && (!isSupportMLA(shapeInfo, query))) {
        shapeInfo.needPadValue = true;
        shapeInfo.needPad = true;
        if (shapeInfo.inputLayout == InputLayout::SBH || shapeInfo.inputLayout == InputLayout::BSH) {
            shapeInfo.needReshape = true;
        }
        if (shapeInfo.inputLayout == InputLayout::BSH && !shapeInfo.needTranspose) {
            shapeInfo.l0InputLayoutStr = "BSND";
        }
        shapeInfo.padNumv = shapeInfo.padNum + shapeInfo.axes.d - shapeInfo.axes.dv;
        if (shapeInfo.inputLayout == InputLayout::BSH) {
            shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.b, shapeInfo.axes.s1, shapeInfo.axes.n1, shapeInfo.axes.d});
            shapeInfo.reshapedKeyShape.assign({shapeInfo.axes.b, shapeInfo.axes.s2, shapeInfo.axes.n2, shapeInfo.axes.d});
            shapeInfo.reshapedValueBefore.assign({shapeInfo.axes.b, shapeInfo.axes.s2, shapeInfo.axes.n2, shapeInfo.axes.dv});
        }
        if (shapeInfo.inputLayout == InputLayout::SBH) {
            shapeInfo.reshapedQueryShape.assign({shapeInfo.axes.s1, shapeInfo.axes.b, shapeInfo.axes.n1, shapeInfo.axes.d});
            shapeInfo.reshapedKeyShape.assign({shapeInfo.axes.s2, shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.d});
            shapeInfo.reshapedValueBefore.assign({shapeInfo.axes.s2, shapeInfo.axes.b, shapeInfo.axes.n2, shapeInfo.axes.dv});
        }
    }

    OP_LOGD("Analysis input success. Analysis result: [needReshape]: %d, [needPad]: %d, [padNum]: %lu, [padNumv]: %lu,"
            "[needTranspose]: %d, [needPadValue]: %d.",
            shapeInfo.needReshape, shapeInfo.needPad, shapeInfo.padNum, shapeInfo.padNumv, shapeInfo.needTranspose, shapeInfo.needPadValue);
    return ACLNN_SUCCESS;
}

static inline const aclTensor *GeneratePaddings(int32_t dimNum, int32_t padNum, aclOpExecutor *executor)
{
    // 2代表每根轴的前后都可以补0
    FVector<int64_t> padVec(dimNum * 2, 0);
    padVec[padVec.size() - 1] = padNum;

    auto padArray = executor->AllocIntArray(padVec.data(), padVec.size());
    if (padArray == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Try alloc padVec failed");
        return nullptr;
    }

    auto padTensor = executor->ConvertToTensor(padArray, DataType::DT_INT64);
    return padTensor;
}

static aclnnStatus Contiguous(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                              const aclTensor *&realShiftOptional, const aclTensor *&dropMaskOptional,
                              const aclTensor *&paddingMaskOptional, const aclTensor *&attenMaskOptional,
                              const aclTensor *&queryRope, const aclTensor *&keyRope,
                              aclOpExecutor *executor)
{
    query = l0op::Contiguous(query, executor);
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    key = l0op::Contiguous(key, executor);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    value = l0op::Contiguous(value, executor);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    if (realShiftOptional) {
        realShiftOptional = l0op::Contiguous(realShiftOptional, executor);
        CHECK_RET(realShiftOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (dropMaskOptional) {
        dropMaskOptional = l0op::Contiguous(dropMaskOptional, executor);
        CHECK_RET(dropMaskOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (paddingMaskOptional) {
        paddingMaskOptional = l0op::Contiguous(paddingMaskOptional, executor);
        CHECK_RET(paddingMaskOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (attenMaskOptional) {
        attenMaskOptional = l0op::Contiguous(attenMaskOptional, executor);
        CHECK_RET(attenMaskOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (queryRope != nullptr) {
        queryRope = l0op::Contiguous(queryRope, executor);
        CHECK_RET(queryRope != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    if (keyRope != nullptr) {
        keyRope = l0op::Contiguous(keyRope, executor);
        CHECK_RET(keyRope != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus PreprocessQKV(const aclTensor *&query, const aclTensor *&key, const aclTensor *&value,
                                 const struct FaShapeInfo &shapeInfo, aclOpExecutor *executor)
{
    if (shapeInfo.needReshape) {
        query = l0op::Reshape(
            query, executor->AllocIntArray(shapeInfo.reshapedQueryShape.data(), shapeInfo.reshapedQueryShape.size()),
            executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        key = l0op::Reshape(
            key,
            executor->AllocIntArray(shapeInfo.reshapedKeyShape.data(), shapeInfo.reshapedKeyShape.size()),
            executor);
        CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        value = l0op::Reshape(
            value,
            executor->AllocIntArray(shapeInfo.reshapedValueBefore.data(), shapeInfo.reshapedValueBefore.size()),
            executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    if (shapeInfo.needPad) {
        int32_t dimNum = shapeInfo.inputLayout == InputLayout::TND ? DIM_NUM_3 : DIM_NUM_4;
        auto qkPaddings = GeneratePaddings(dimNum, shapeInfo.padNum, executor);
        auto vPaddings = GeneratePaddings(dimNum, shapeInfo.padNumv, executor);
        if (shapeInfo.padNum != 0) {
            query = l0op::Pad(query, qkPaddings, executor);
            CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
            key = l0op::Pad(key, qkPaddings, executor);
            CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        }
        if (shapeInfo.padNumv != 0) {
            value = l0op::Pad(value, vPaddings, executor);
            CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        }
    }
    if (shapeInfo.needTranspose) {
        // B,S,N,D -> B,N,S,D
        // S,B,N,D -> B,N,S,D
        auto perm = executor->AllocIntArray(shapeInfo.perm_in.data(), shapeInfo.perm_in.size());
        query = l0op::Transpose(query, perm, executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        key = l0op::Transpose(key, perm, executor);
        CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        value = l0op::Transpose(value, perm, executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    if (shapeInfo.inputLayout == InputLayout::SBH && shapeInfo.needPad && !shapeInfo.needTranspose) {
        // (S,B,N,D) -> (S,B,N*D)
        FVector<int64_t, DIM_NUM_3> queryShape{shapeInfo.axes.s1, shapeInfo.axes.b,
                                               shapeInfo.axes.n1 * (shapeInfo.axes.d +
                                               static_cast<int64_t>(shapeInfo.padNum))};
        FVector<int64_t, DIM_NUM_3> keyShape{shapeInfo.axes.s2, shapeInfo.axes.b,
                                                  shapeInfo.axes.n2 * (shapeInfo.axes.d +
                                                  static_cast<int64_t>(shapeInfo.padNum))};
        FVector<int64_t, DIM_NUM_3> ValueShape{shapeInfo.axes.s2, shapeInfo.axes.b,
                                                  shapeInfo.axes.n2 * (shapeInfo.axes.dv +
                                                  static_cast<int64_t>(shapeInfo.padNumv))};

        query = l0op::Reshape(query, executor->AllocIntArray(queryShape.data(), queryShape.size()), executor);
        CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        key = l0op::Reshape(key, executor->AllocIntArray(keyShape.data(), keyShape.size()), executor);
        CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        value = l0op::Reshape(value, executor->AllocIntArray(ValueShape.data(), ValueShape.size()), executor);
        CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus Postprocess(const aclTensor *&l0AttentionOutOut, const aclTensor *attentionOutOut,
                               struct FaShapeInfo &shapeInfo, aclOpExecutor *executor)
{
    if (shapeInfo.inputLayout == InputLayout::SBH && shapeInfo.needPad && !shapeInfo.needTranspose) {
        // (S,B,Hp) -> (S,B,N,Dp)
        FVector<int64_t, DIM_NUM_4> paddedSBNDShape{shapeInfo.axes.s1, shapeInfo.axes.b, shapeInfo.axes.n1,
                                                    shapeInfo.axes.dv + static_cast<int64_t>(shapeInfo.padNumv)};
        l0AttentionOutOut = l0op::Reshape(
            l0AttentionOutOut, executor->AllocIntArray(paddedSBNDShape.data(), paddedSBNDShape.size()), executor);
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    if (shapeInfo.needTranspose) {
        auto perm = executor->AllocIntArray(shapeInfo.perm_out.data(), shapeInfo.perm_out.size());
        l0AttentionOutOut = l0op::Transpose(l0AttentionOutOut, perm, executor);
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    if (shapeInfo.needPad && shapeInfo.padNumv !=0) {
        // (B,S,N,D)
        // (S,B,N,D)
        // (B,N,S,D)
        // (T,N,D)
        FVector<int64_t, MAX_DIM_NUM> sizeVec = ToShapeVector(l0AttentionOutOut->GetViewShape());
        sizeVec.back() -= shapeInfo.padNumv;
        if (shapeInfo.inputLayout == InputLayout::TND) {
            FVector<int64_t, DIM_NUM_3> offsetVec(DIM_NUM_3, 0);
            l0AttentionOutOut =
                l0op::Slice(l0AttentionOutOut, executor->AllocIntArray(offsetVec.data(), offsetVec.size()),
                            executor->AllocIntArray(sizeVec.data(), sizeVec.size()), executor);
        } else {
            FVector<int64_t, DIM_NUM_4> offsetVec(DIM_NUM_4, 0);
            l0AttentionOutOut =
                l0op::Slice(l0AttentionOutOut, executor->AllocIntArray(offsetVec.data(), offsetVec.size()),
                            executor->AllocIntArray(sizeVec.data(), sizeVec.size()), executor);
        }
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }

    if (shapeInfo.needReshape) {
        auto attentionOutOutShape = ToShapeVector(attentionOutOut->GetViewShape());
        l0AttentionOutOut =
            l0op::Reshape(l0AttentionOutOut,
                          executor->AllocIntArray(attentionOutOutShape.data(), attentionOutOutShape.size()), executor);
        CHECK_RET(l0AttentionOutOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFaParam(const aclTensor *query, const aclTensor *key, const aclTensor *value,
    const char *inputLayout, const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut,
    const aclTensor *attentionOutOut, const uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 必须的参数指针判空
    CHECK_RET(query != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(key != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(value != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(inputLayout != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxMaxOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(softmaxSumOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(attentionOutOut != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus isSupportMultiInput(const aclTensor *query, const aclTensor *queryRope,
                                       const aclTensor *key, const aclTensor *keyRope, const aclTensor *value, 
                                       const aclTensor *attenMaskOptional, const aclTensor *pseShiftOptional,
                                       const aclTensor *dropMaskOptional, double keepProb,
                                       const FaShapeInfo &faShape, int64_t sparseMode)
{
    CHECK_RET((queryRope == nullptr && keyRope == nullptr) || (queryRope != nullptr && keyRope != nullptr),
            ACLNN_ERR_PARAM_NULLPTR);
    auto vDtype = value->GetDataType();
    auto kDtype = key->GetDataType();
    auto qDtype = query->GetDataType();
    auto kRopeDtype = keyRope->GetDataType();
    auto qRopeDtype = queryRope->GetDataType();
    Shape qRopeShape = queryRope->GetViewShape();
    Shape kRopeShape = keyRope->GetViewShape();
    if (qRopeShape[DIM_NUM_2] > faShape.axes.d || kRopeShape[DIM_NUM_2] > faShape.axes.d) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, do not support query_rope and key_rope when"
                " the head-dim of query_rope or key_rope is larger than the head-dim of query.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (qDtype !=  ge::DataType::DT_BF16 || kDtype != ge::DataType::DT_BF16 || vDtype != ge::DataType::DT_BF16
        || qRopeDtype != ge::DataType::DT_BF16 || kRopeDtype != ge::DataType::DT_BF16) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The data type of query[%s], queryRope[%s], key[%s], keyRope[%s], value[%s]"
                " should be BFloat16.", op::ToString(DataType(qDtype)).GetString(),
                op::ToString(DataType(qRopeDtype)).GetString(), op::ToString(DataType(kDtype)).GetString(),
                op::ToString(DataType(kRopeDtype)).GetString(), op::ToString(DataType(vDtype)).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (sparseMode == 6) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, do not support query_rope and key_rope when sparseMode is 6.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (queryRope != nullptr) {
        if (attenMaskOptional == nullptr ||
            attenMaskOptional->GetViewShape().GetDimNum() == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "Invalid input, only support query_rope and key_rope when attentionMask is given.");
            return ACLNN_ERR_PARAM_NULLPTR;
        }
        if (pseShiftOptional != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, only support query_rope and key_rope when pseShift is nullptr.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (dropMaskOptional != nullptr) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, only support query_rope and key_rope when dropMask is nullptr.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (keepProb < 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, only support query_rope and key_rope when keepProb = 1.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (faShape.inputLayout != InputLayout::TND) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, only support query_rope and key_rope as input for layout TND.");
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (faShape.needPad || faShape.needTranspose || faShape.needReshape || faShape.needPadValue) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Invalid input, do not support query_rope and key_rope as input when shape is not aligned with 128 or other corner cases.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode, const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckFaParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
        workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnFlashAttentionScore,
                   DFX_IN(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, prefixOptional, scaleValue, keepProb, preTokens, nextTokens,
                          headNum, inputLayout, innerPrecise, sparseMode),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(InputDtypeCheck(query, key, value, realShiftOptional, PSE_TYPE_V1) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    FaShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(query, key, value, inputLayout, headNum, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    const aclTensor *queryRope = nullptr;
    const aclTensor *keyRope = nullptr;
    CHECK_RET(Contiguous(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, queryRope, keyRope,
                         l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    auto l0FlashAttentionScoreOuts = l0op::FlashAttentionScore(
        query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, prefixOptional,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, scaleValue, keepProb, preTokens, nextTokens,
        headNum, shapeInfo.l0InputLayoutStr.c_str(), innerPrecise, sparseMode, PSE_TYPE_V1, 0, 0, 0, "", l0Executor);

    CHECK_RET(l0FlashAttentionScoreOuts[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(l0FlashAttentionScoreOuts[1] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(l0FlashAttentionScoreOuts[DIM_NUM_3] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto l0SoftmaxMaxOut = l0FlashAttentionScoreOuts[0];
    auto l0SoftmaxSumOut = l0FlashAttentionScoreOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FlashAttentionScoreOuts[DIM_NUM_3];

    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionScore);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionVarLenScoreGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, double scaleValue, double keepProb, int64_t preTokens,
    int64_t nextTokens, int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckFaParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
        workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnFlashAttentionVarLenScore,
                   DFX_IN(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, scaleValue,
                          keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not TND, invalid shape, please check", inputLayout);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }
    FaShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(query, key, value, inputLayout, headNum, shapeInfo, actualSeqQLenOptional,
                            actualSeqKvLenOptional) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    const aclTensor *queryRope = nullptr;
    const aclTensor *keyRope = nullptr;
    CHECK_RET(Contiguous(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, queryRope, keyRope,
                         l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    auto l0FlashAttentionScoreOuts = l0op::FlashAttentionScore(
        query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, scaleValue,
        keepProb, preTokens, nextTokens, headNum, shapeInfo.l0InputLayoutStr.c_str(), innerPrecise, sparseMode,
        PSE_TYPE_V1, 0, 0, 0, "", l0Executor);

    auto l0SoftmaxMaxOut = l0FlashAttentionScoreOuts[0];
    auto l0SoftmaxSumOut = l0FlashAttentionScoreOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FlashAttentionScoreOuts[3];

    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    if (l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr || l0AttentionOutOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "l0SoftmaxMaxOut or l0SoftmaxSumOut or l0AttentionOutOut is null");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionVarLenScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionVarLenScore);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionScoreV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *qStartIdxOptional, const aclIntArray *kvStartIdxOptional,
    double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens, int64_t headNum, char *inputLayout,
    int64_t innerPrecise, int64_t sparseMode, int64_t pseType, const aclTensor *softmaxMaxOut,
    const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut, const aclTensor *attentionOutOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckFaParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
        workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnFlashAttentionScoreV2,
                   DFX_IN(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, prefixOptional, qStartIdxOptional, kvStartIdxOptional, scaleValue,
                          keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode, pseType),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_RET(InputDtypeCheck(query, key, value, realShiftOptional, pseType) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    FaShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(query, key, value, inputLayout, headNum, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    const aclTensor *queryRope = nullptr;
    const aclTensor *keyRope = nullptr;
    CHECK_RET(Contiguous(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, queryRope, keyRope,
                         l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    auto l0FlashAttentionScoreOuts = l0op::FlashAttentionScore(
        query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, prefixOptional,
        nullptr, nullptr, qStartIdxOptional, kvStartIdxOptional, nullptr, nullptr, nullptr, nullptr, nullptr, scaleValue, keepProb,
        preTokens, nextTokens, headNum, shapeInfo.l0InputLayoutStr.c_str(), innerPrecise, sparseMode,
        pseType, 0, 0, 0, "", l0Executor);

    auto l0SoftmaxMaxOut = l0FlashAttentionScoreOuts[0];
    auto l0SoftmaxSumOut = l0FlashAttentionScoreOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FlashAttentionScoreOuts[3];

    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScoreV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                       const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionScoreV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionVarLenScoreV2GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckFaParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
        workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnFlashAttentionVarLenScoreV2,
                   DFX_IN(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
                          qStartIdxOptional, kvStartIdxOptional, scaleValue, keepProb, preTokens, nextTokens,
                          headNum, inputLayout, innerPrecise, sparseMode, pseType),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not TND, invalid shape, please check", inputLayout);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }
    FaShapeInfo shapeInfo;
    CHECK_RET(InputDtypeCheck(query, key, value, realShiftOptional, pseType) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(AnalysisInput(query, key, value, inputLayout, headNum, shapeInfo, actualSeqQLenOptional,
                            actualSeqKvLenOptional) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    const aclTensor *queryRope = nullptr;
    const aclTensor *keyRope = nullptr;
    CHECK_RET(Contiguous(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, queryRope, keyRope,
                         l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    auto l0FlashAttentionScoreOuts = l0op::FlashAttentionScore(
        query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional, nullptr, nullptr, nullptr, nullptr, nullptr,
        scaleValue, keepProb, preTokens, nextTokens, headNum, shapeInfo.l0InputLayoutStr.c_str(), innerPrecise,
        sparseMode, pseType, 0, 0, 0, "", l0Executor);

    auto l0SoftmaxMaxOut = l0FlashAttentionScoreOuts[0];
    auto l0SoftmaxSumOut = l0FlashAttentionScoreOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FlashAttentionScoreOuts[3];

    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    if (l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr || l0AttentionOutOut == nullptr) {
      OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "l0SoftmaxMaxOut or l0SoftmaxSumOut or l0AttentionOutOut is null");
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionVarLenScoreV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionVarLenScoreV2);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize(
    const aclTensor *query, const aclTensor *queryRope, const aclTensor *key, const aclTensor *keyRope, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional, double scaleValue, double keepProb, int64_t preTokens, int64_t nextTokens,
    int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode, int64_t pseType,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckFaParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
        workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnFlashAttentionVarLenScoreV3,
                   DFX_IN(query, queryRope, key, keyRope, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional,
                          qStartIdxOptional, kvStartIdxOptional, scaleValue, keepProb, preTokens, nextTokens,
                          headNum, inputLayout, innerPrecise, sparseMode, pseType),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not TND, invalid shape, please check", inputLayout);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }
    FaShapeInfo shapeInfo;
    CHECK_RET(InputDtypeCheck(query, key, value, realShiftOptional, pseType) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(AnalysisInput(query, key, value, inputLayout, headNum, shapeInfo, actualSeqQLenOptional,
                            actualSeqKvLenOptional) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();

    CHECK_RET(Contiguous(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, queryRope, keyRope,
        l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(isSupportMultiInput(query, queryRope, key, keyRope, value, attenMaskOptional, realShiftOptional,
        dropMaskOptional, keepProb, shapeInfo, sparseMode) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    auto l0FlashAttentionScoreOuts = l0op::FlashAttentionScore(
        query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, qStartIdxOptional, kvStartIdxOptional, nullptr, nullptr, nullptr, queryRope, keyRope,
        scaleValue, keepProb, preTokens, nextTokens, headNum, shapeInfo.l0InputLayoutStr.c_str(), innerPrecise,
        sparseMode, pseType, 0, 0, 0, "", l0Executor);

    auto l0SoftmaxMaxOut = l0FlashAttentionScoreOuts[0];
    auto l0SoftmaxSumOut = l0FlashAttentionScoreOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FlashAttentionScoreOuts[3];

    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    if (l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr || l0AttentionOutOut == nullptr) {
      OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "l0SoftmaxMaxOut or l0SoftmaxSumOut or l0AttentionOutOut is null");
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionVarLenScoreV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                             const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionVarLenScoreV3);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnFlashAttentionVarLenScoreV4GetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *realShiftOptional,
    const aclTensor *dropMaskOptional, const aclTensor *paddingMaskOptional, const aclTensor *attenMaskOptional,
    const aclIntArray *prefixOptional, const aclIntArray *actualSeqQLenOptional,
    const aclIntArray *actualSeqKvLenOptional, double scaleValue, double keepProb, int64_t preTokens,
    int64_t nextTokens, int64_t headNum, char *inputLayout, int64_t innerPrecise, int64_t sparseMode, char *softmaxOutLayout,
    const aclTensor *softmaxMaxOut, const aclTensor *softmaxSumOut, const aclTensor *softmaxOutOut,
    const aclTensor *attentionOutOut, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    CHECK_RET(CheckFaParam(query, key, value, inputLayout, softmaxMaxOut, softmaxSumOut, attentionOutOut,
        workspaceSize, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    L2_DFX_PHASE_1(aclnnFlashAttentionVarLenScoreV4,
                   DFX_IN(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional,
                          attenMaskOptional, prefixOptional, actualSeqQLenOptional, actualSeqKvLenOptional, scaleValue,
                          keepProb, preTokens, nextTokens, headNum, inputLayout, innerPrecise, sparseMode, softmaxOutLayout),
                   DFX_OUT(softmaxMaxOut, softmaxSumOut, softmaxOutOut, attentionOutOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // b, n1, s1 为0时，不进行任何处理
    // n2, s2, d 为0时，直接调用l0接口处理
    if (softmaxMaxOut->IsEmpty() && softmaxSumOut->IsEmpty() && attentionOutOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (strcmp(inputLayout, "TND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Layout %s is not TND, invalid shape, please check", inputLayout);
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_INVALID;
    }
    FaShapeInfo shapeInfo;
    CHECK_RET(AnalysisInput(query, key, value, inputLayout, headNum, shapeInfo, actualSeqQLenOptional,
                            actualSeqKvLenOptional) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclOpExecutor *l0Executor = uniqueExecutor.get();
    const aclTensor *queryRope = nullptr;
    const aclTensor *keyRope = nullptr;
    CHECK_RET(Contiguous(query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, queryRope, keyRope,
                         l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    CHECK_RET(PreprocessQKV(query, key, value, shapeInfo, l0Executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);

    auto l0FlashAttentionScoreOuts = l0op::FlashAttentionScore(
        query, key, value, realShiftOptional, dropMaskOptional, paddingMaskOptional, attenMaskOptional, prefixOptional,
        actualSeqQLenOptional, actualSeqKvLenOptional, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, scaleValue,
        keepProb, preTokens, nextTokens, headNum, shapeInfo.l0InputLayoutStr.c_str(), innerPrecise, sparseMode,
        PSE_TYPE_V1, 0, 0, 0, softmaxOutLayout, l0Executor);

    auto l0SoftmaxMaxOut = l0FlashAttentionScoreOuts[0];
    auto l0SoftmaxSumOut = l0FlashAttentionScoreOuts[1];
    // l0SoftmaxOutOut not used now
    auto l0AttentionOutOut = l0FlashAttentionScoreOuts[3];

    CHECK_RET(Postprocess(l0AttentionOutOut, attentionOutOut, shapeInfo, l0Executor) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_NULLPTR);
    if (l0SoftmaxMaxOut == nullptr || l0SoftmaxSumOut == nullptr || l0AttentionOutOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "l0SoftmaxMaxOut or l0SoftmaxSumOut or l0AttentionOutOut is null");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto viewCopyResult0 = l0op::ViewCopy(l0SoftmaxMaxOut, softmaxMaxOut, l0Executor);
    CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto viewCopyResult1 = l0op::ViewCopy(l0SoftmaxSumOut, softmaxSumOut, l0Executor);
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    // l0SoftmaxOutOut not used now
    auto viewCopyResult3 = l0op::ViewCopy(l0AttentionOutOut, attentionOutOut, l0Executor);
    CHECK_RET(viewCopyResult3 != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionVarLenScoreV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                           const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnFlashAttentionVarLenScoreV4);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

}  // namespace

#ifdef __cplusplus
}
#endif
