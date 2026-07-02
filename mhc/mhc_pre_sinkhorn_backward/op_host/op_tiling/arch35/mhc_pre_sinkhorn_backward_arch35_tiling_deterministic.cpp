/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file mhc_pre_sinkhorn_backward_arch35_tiling_deterministic.cpp
 * \brief deterministic tiling — C-Short/C-Long 分场景 UB 预算
 */

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "mhc_pre_sinkhorn_backward_arch35_tiling_deterministic.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "util/math_util.h"
#include "util/platform_util.h"

using namespace AscendC;
namespace optiling {

// 输入/输出索引
constexpr uint8_t GRAD_HIN_IDX        = 0;
constexpr uint8_t GRAD_H_POST_IDX     = 1;
constexpr uint8_t GRAD_H_RES_IDX      = 2;
constexpr uint8_t INPUT_X_IDX         = 3;
constexpr uint8_t PHI_IDX             = 4;
constexpr uint8_t ALPHA_IDX           = 5;
constexpr uint8_t BIAS_IDX            = 6;
constexpr uint8_t H_PRE_IDX           = 7;
constexpr uint8_t HC_BEFORE_NORM_IDX  = 8;
constexpr uint8_t INV_RMS_IDX         = 9;
constexpr uint8_t SUM_OUT_IDX         = 10;
constexpr uint8_t NORM_OUT_IDX        = 11;
constexpr uint8_t GRAD_X_IDX          = 0;
constexpr uint8_t GRAD_PHI_IDX        = 1;
constexpr uint8_t GRAD_ALPHA_IDX      = 2;
constexpr uint8_t GRAD_BIAS_IDX       = 3;

// 维度索引
constexpr uint8_t BATCH_SIZE_DIM_IDX  = 0;
constexpr uint8_t SEQ_LENGTH_DIM_IDX  = 1;
constexpr uint8_t N_DIM_IDX           = 2;
constexpr uint8_t C_DIM_IDX           = 3;

// 常量
constexpr float    DEFAULT_EPS         = 1e-6f;
constexpr uint8_t  ITER_COUNT_IDX      = 0;
constexpr int64_t  DOUBLE_BUFFER       = 2;
constexpr int64_t  BF16_BYTE_SIZE      = 2;
constexpr int64_t  FP32_BYTE_SIZE      = 4;
constexpr uint64_t UB_RESERVE_SIZE     = 256;
constexpr int64_t  WS_BUFFER_INTERVAL  = 1024;
constexpr int64_t  UB_FACTOR_C_DEFAULT = 256;  // C-Long 时默认 C 切分粒度

// UB Buffer 计数（与 kernel TBuf 声明对应）
constexpr int64_t NUM_K_BUFS_FIXED    = 3;   // biasBuf_ + gradCalcBuf1_ + gradCalcBuf2_
constexpr int64_t NUM_NN_BUFS_SIGMOID = 3;   // gradZBuf_ + normOutForwardBuf_ + zBuf_
constexpr int64_t NUM_NN_BUFS_SK      = 8;   // tmpDiv×4 + tmpMul + zResMax + zResMaxIdx + tmp + isMax
constexpr int64_t NUM_SUM_OUT_BUFS    = 2;   // sumOutRowBuf_ + sumOutColBuf_
constexpr int64_t NUM_NORM_OUT_BUFS   = 1;   // normOutBuf_

template <typename T>
constexpr T AlignUp(T value, T align)
{
    return (value + align - 1) / align * align;
}

using namespace ge;
using namespace std;
using namespace AscendC;

// ================================================================
// IsCapable
// ================================================================
bool MhcPreSinkhornBackwardArch35DeterminiticTiling::IsCapable()
{
    return context_->GetDeterministic() == 1;
}

// ================================================================
// SetTilingData — MatMul tiling + 字段填充
// ================================================================
ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::SetTilingData()
{
    auto* tilingData = context_->GetTilingData<MhcPreSinkhornBackwardArch35DeterminiticTilingData>();
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfo);

    auto floatDataType = matmul_tiling::DataType::DT_FLOAT;

    int64_t kDim = n_ * n_ + 2 * n_;
    mm1K_ = kDim;
    mm1M_ = batchSize_ * seqLength_;
    mm1N_ = n_ * c_;
    mm2K_ = batchSize_ * seqLength_;
    mm2M_ = kDim;
    mm2N_ = n_ * c_;

    matmul_tiling::MatmulApiTiling mm1Tiling(ascendPlatformInfo);
    matmul_tiling::MatmulApiTiling mm2Tiling(ascendPlatformInfo);

    mm1Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm1Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm1Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm1Tiling.SetOrgShape(mm1M_, mm1N_, mm1K_);
    mm1Tiling.SetShape(mm1M_, mm1N_, mm1K_);
    mm1Tiling.SetBias(false);
    mm1Tiling.SetBufferSpace(-1, -1, -1);
    mm1Tiling.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN);
    if (mm1Tiling.GetTiling(tilingData->mm1TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    mm2Tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType, true);
    mm2Tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm2Tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, floatDataType);
    mm2Tiling.SetOrgShape(mm2M_, mm2N_, mm2K_);
    mm2Tiling.SetShape(mm2M_, mm2N_, mm2K_);
    mm2Tiling.SetBias(false);
    mm2Tiling.SetBufferSpace(-1, -1, -1);
    mm2Tiling.SetTraverse(matmul_tiling::MatrixTraverse::FIRSTN);
    if (mm2Tiling.GetTiling(tilingData->mm2TilingData) == -1) {
        return ge::GRAPH_FAILED;
    }

    tilingData->b             = batchSize_;
    tilingData->s             = seqLength_;
    tilingData->c             = c_;
    tilingData->n             = n_;
    tilingData->usedCoreNum   = usedCoreNum_;
    tilingData->aicCoreNum    = coreNumAic_;
    tilingData->ubFactorBS    = ubFactorBS_;
    tilingData->ubFactorC     = ubFactorC_;
    tilingData->blockFactorBS = blockFactorBS_;
    tilingData->blockTailBS   = blockTailBS_;
    tilingData->skIterCount   = skIterCount_;
    tilingData->eps           = hcEps_;

    return ge::GRAPH_SUCCESS;
}

// ================================================================
// DoUbTiling — C-Short/C-Long 分场景 UB 预算
//
// 策略:
//   1. 先算 C 全量时每 BS 的 UB 开销 (perBsCostC_full)
//   2. 如果 perBsCostC_full + perBsCostK ≤ availUb → C-Short: ubFactorC=c
//   3. 否则 → C-Long: ubFactorC=对齐值, 重算 perBsCostC
//   4. 最后: ubFactorBS = (availUb - fixedCost) / (perBsCostC + perBsCostK)
// ================================================================
void MhcPreSinkhornBackwardArch35DeterminiticTiling::DoUbTiling()
{
    int64_t totalBs = batchSize_ * seqLength_;
    int64_t kDim = n_ * n_ + 2 * n_;
    int64_t nn = n_ * n_;

    // --- BS 分核 ---
    blockFactorBS_ = Ops::Base::CeilDiv(totalBs, static_cast<int64_t>(coreNumAiv_));
    usedCoreNum_   = Ops::Base::CeilDiv(totalBs, blockFactorBS_);
    blockTailBS_   = totalBs - blockFactorBS_ * (usedCoreNum_ - 1);

    // --- UB 预算 ---
    ubBlockSize_ = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context_));
    int64_t availUb = static_cast<int64_t>(ubSize_) - UB_RESERVE_SIZE;

    auto AlignB = [&](int64_t bytes) {
        return static_cast<int64_t>(Ops::Base::CeilAlign(bytes, ubBlockSize_));
    };

    int64_t cFp16AlignFull = AlignB(c_ * BF16_BYTE_SIZE);
    int64_t cFp32AlignFull = AlignB(c_ * FP32_BYTE_SIZE);
    int64_t nAlign   = AlignB(n_ * FP32_BYTE_SIZE);
    int64_t nnAlign  = AlignB(nn * FP32_BYTE_SIZE);
    int64_t kAlign   = AlignB(kDim * FP32_BYTE_SIZE);

    // 固定开销 (与 BS/C 无关)
    int64_t fixedCost = kAlign * NUM_K_BUFS_FIXED
                      + nnAlign                              // gradAlphaAndBiasQue_
                      + ubBlockSize_;                        // gradAlphaSumQue_

    // 每 BS 的 C 无关开销 (K 维, N≤8)
    int64_t perBsCostK =
        DOUBLE_BUFFER * nAlign                             // hPreAndGradHPostQue_
      + DOUBLE_BUFFER * kAlign                              // hcBeforeNormQue_
      + DOUBLE_BUFFER * nnAlign                             // gradHResQue_
      + DOUBLE_BUFFER * AlignB(FP32_BYTE_SIZE)              // invRmsQue_
      + nAlign                                              // gradHPreBuf_
      + nnAlign * NUM_NN_BUFS_SIGMOID                       // gradZ + normOutFwd + z
      + nnAlign * NUM_NN_BUFS_SK                            // tmpDiv×4 + tmpMul + zResMax×2 + tmp + isMax
      + nAlign * NUM_SUM_OUT_BUFS * skIterCount_            // sumOutRow + sumOutCol
      + nnAlign * NUM_NORM_OUT_BUFS * skIterCount_;         // normOutBuf

    // 每 BS 的 C 相关开销 (C 全量)
    int64_t perBsCostC_full =
        DOUBLE_BUFFER * n_ * cFp16AlignFull                 // xQue_
      + DOUBLE_BUFFER * n_ * cFp32AlignFull * 2             // xFp32Que_ + gradXFromHinQue_
      + DOUBLE_BUFFER * cFp16AlignFull;                     // gradHinQue_

    // --- 场景判断 ---
    int64_t perBsCostC;
    if (perBsCostC_full + perBsCostK <= availUb - fixedCost) {
        // C-Short: 不切 C, 全量处理
        ubFactorC_ = c_;
        perBsCostC = perBsCostC_full;
    } else {
        // C-Long: 切 C, 用对齐粒度
        ubFactorC_ = AlignUp(UB_FACTOR_C_DEFAULT, int64_t(8));  // 对齐到 8 元素
        if (ubFactorC_ > c_) {
            ubFactorC_ = c_;
        }
        int64_t cFp16Align = AlignB(ubFactorC_ * BF16_BYTE_SIZE);
        int64_t cFp32Align = AlignB(ubFactorC_ * FP32_BYTE_SIZE);
        perBsCostC =
            DOUBLE_BUFFER * n_ * cFp16Align
          + DOUBLE_BUFFER * n_ * cFp32Align * 2
          + DOUBLE_BUFFER * cFp16Align;
    }

    // --- 计算 ubFactorBS ---
    int64_t perBsCost = perBsCostC + perBsCostK;
    if (perBsCost > 0) {
        ubFactorBS_ = (availUb - fixedCost) / perBsCost;
    } else {
        ubFactorBS_ = blockFactorBS_;
    }
    ubFactorBS_ = max(int64_t(1), min(ubFactorBS_, blockFactorBS_));
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::DoOpTiling()
{
    DoUbTiling();
    return SetTilingData();
}

uint64_t MhcPreSinkhornBackwardArch35DeterminiticTiling::GetTilingKey() const
{
    bool isDeterministic = true;
    return GET_TPL_TILING_KEY(isDeterministic);
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::PostTiling()
{
    context_->SetBlockDim(coreNumAic_);
    context_->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendPlatformInfo = platform_ascendc::PlatformAscendC(platformInfo);

    int64_t totalBs = batchSize_ * seqLength_;
    int64_t kDim = n_ * n_ + 2 * n_;

    // 新方案 WS 布局 (gradXFromRms 不落 WS, gradNormOut 不落 WS):
    // 1. xFp32[B*S, N, C]
    // 2. gradXFromHin[B*S, N, C]
    // 3. gradHcBeforeNorm[B*S, K]  (供 AIC mm1)
    // 4. gradXFromMatmul[B*S, N*C] (AIC mm1 输出)
    // 5. gradAlpha[3 * usedCoreNum] (跨核归约)
    // 6. gradBias[usedCoreNum * K]  (跨核归约)
    size_t userWorkspaceSize =
        totalBs * n_ * c_ * FP32_BYTE_SIZE + WS_BUFFER_INTERVAL +                      // xFp32
        totalBs * n_ * c_ * FP32_BYTE_SIZE + WS_BUFFER_INTERVAL +                      // gradXFromHin
        totalBs * kDim * FP32_BYTE_SIZE + WS_BUFFER_INTERVAL +                         // gradHcBeforeNorm
        totalBs * n_ * c_ * FP32_BYTE_SIZE + WS_BUFFER_INTERVAL +                      // gradXFromMatmul
        3 * usedCoreNum_ * FP32_BYTE_SIZE + WS_BUFFER_INTERVAL +                       // gradAlpha
        (2 * usedCoreNum_ * n_ + usedCoreNum_ * n_ * n_) * FP32_BYTE_SIZE + WS_BUFFER_INTERVAL; // gradBias

    size_t systemWorkspaceSize = ascendPlatformInfo.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE(opName, "fail to GetWorkspaceSizes"), return ge::GRAPH_FAILED);
    currentWorkspace[0] = systemWorkspaceSize + userWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

void MhcPreSinkhornBackwardArch35DeterminiticTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "b: " << batchSize_ << std::endl;
    info << "s: " << seqLength_ << std::endl;
    info << "c: " << c_ << std::endl;
    info << "n: " << n_ << std::endl;
    info << "usedCoreNum: " << usedCoreNum_ << std::endl;
    info << "ubFactorBS: " << ubFactorBS_ << std::endl;
    info << "ubFactorC: " << ubFactorC_ << std::endl;
    info << "blockFactorBS: " << blockFactorBS_ << std::endl;
    info << "blockTailBS: " << blockTailBS_ << std::endl;
    info << "skIterCount: " << skIterCount_ << std::endl;
    OP_LOGI(opName, "Tiling info is: %s", info.str().c_str());
}

ge::graphStatus MhcPreSinkhornBackwardArch35DeterminiticTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(MhcPreSinkhornBackward, MhcPreSinkhornBackwardArch35DeterminiticTiling, 0);
} // namespace optiling
