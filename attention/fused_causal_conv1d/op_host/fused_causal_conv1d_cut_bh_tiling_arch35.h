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
 * \file fused_causal_conv1d_cut_bh_tiling_arch35.h
 * \brief FusedCausalConv1dCutBH tiling implementation
 */
#ifndef FUSED_CAUSAL_CONV1D_CUT_BH_TILING_H
#define FUSED_CAUSAL_CONV1D_CUT_BH_TILING_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "../op_kernel/arch35/fused_causal_conv1d_cut_bh_struct.h"

namespace optiling {

// CompileInfo structure for platform information
struct FusedCausalConv1dCutBHCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

constexpr uint64_t TILING_KEY_BH_BF16 = 20000;
constexpr uint64_t TILING_KEY_BH_FP16 = 20001;

// Input tensor indices
constexpr int32_t X_INDEX = 0;
constexpr int32_t WEIGHT_INDEX = 1;
constexpr int32_t CONV_STATES_INDEX = 2;
constexpr int32_t QUERY_START_LOC_INDEX = 3;
constexpr int32_t CACHE_INDICES_INDEX = 4;
constexpr int32_t INITIAL_STATE_MODE_INDEX = 5;
constexpr int32_t BIAS_INDEX = 6;
constexpr int32_t NUM_ACCEPTED_TOKEN_INDEX = 7;

// Output tensor indices
constexpr int32_t Y_INDEX = 0;
constexpr int32_t OUTPUT_CONV_STATES_INDEX = 1;

// Attribute indices
constexpr int32_t ATTR_ACTIVATION_MODE_INDEX = 0;
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;
constexpr int32_t ATTR_RUN_MODE_INDEX = 2;
constexpr int32_t ATTR_RESIDUAL_CONNECTION_INDEX = 3;

// Constants for validation
constexpr int64_t DIM_ALIGN_ELEMENT = 128;  // 128 elements (256 bytes for fp16/bf16)
constexpr int64_t COARSE_GRAIN_THRESHOLD = 1024;  // dim threshold for coarse-grain tiling
constexpr int64_t COARSE_GRAIN_ELEMENT = 1024;    // coarse-grain: 1024 elements
constexpr int64_t COARSE_GRAIN_SIZE = COARSE_GRAIN_ELEMENT * 2;  // 2048 bytes for fp16/bf16
constexpr int64_t DIM_ALIGN_SIZE = 256;  // 256 bytes
constexpr int64_t ALIGN_BYTES = 32;
constexpr int64_t MIN_DIM = 128;
constexpr int64_t MAX_DIM = 16384;
constexpr int64_t MIN_BATCH = 1;
constexpr int64_t MAX_BATCH = 256;
constexpr int64_t MIN_M = 0;
constexpr int64_t MAX_M = 5;
constexpr int64_t DIM_0 = 0;
constexpr int64_t DIM_1 = 1;
constexpr int64_t DIM_2 = 2;
constexpr int64_t DIM_3 = 3;
constexpr int64_t DTYPE_SIZE = 2;  // bf16/fp16 size in bytes
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t SYSTEM_RESERVED_UB_SIZE = 8 * 1024;

// Input mode constants
constexpr int64_t X_INPUT_3D = 0;  // 3D input mode
constexpr int64_t X_INPUT_2D = 1;  // 2D input mode

class FusedCausalConv1dCutBHTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit FusedCausalConv1dCutBHTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    // Shape validation functions for each tensor
    ge::graphStatus ValidateXShape();
    ge::graphStatus ValidateWeightShape();
    ge::graphStatus ValidateConvStatesShape();
    ge::graphStatus ValidateQueryStartLocShape();
    ge::graphStatus ValidateCacheIndicesShape();
    ge::graphStatus ValidateNumAcceptedTokenShape();

    // Type validation functions for each tensor
    ge::graphStatus ValidateXType();
    ge::graphStatus ValidateWeightType();
    ge::graphStatus ValidateConvStatesType();
    ge::graphStatus ValidateQueryStartLocType();
    ge::graphStatus ValidateCacheIndicesType();
    ge::graphStatus ValidateNumAcceptedTokenType();

    // Overall validation
    ge::graphStatus CheckInputParams();

private:
    // Tiling calculation functions
    int64_t CalculateLimitedCoreNum();
    int64_t CalculateLimitedCoreNumCoarse();

    // Helpers function for DoOpTiling
    ge::graphStatus ComputeInterCoreSplit();    //核间切分
    ge::graphStatus ComputeIntraCoreUbTiling(); // 核内切分
    void ComputeUbFor(int64_t coreDimElems, int64_t coreBS, int64_t availableUbSize,
                      int64_t &outUbDim, int64_t &outUbBS,
                      int64_t &outLoopDim, int64_t &outLoopBS,
                      int64_t &outUbTailDim, int64_t &outUbTailBS);
    void ComputeUbForCoarse(int64_t coreDimElems, int64_t coreBS, int64_t availableUbSize,
                        int64_t &outUbDim, int64_t &outUbBS,
                        int64_t &outLoopDim, int64_t &outLoopBS,
                        int64_t &outUbTailDim, int64_t &outUbTailBS);

    // Coarse-grain tiling functions (for dim >= 4096)
    ge::graphStatus ComputeInterCoreSplitCoarse();
    ge::graphStatus ComputeIntraCoreUbTilingCoarse();

    // Helper splits for GetShapeAttrsInfo
    ge::graphStatus GetShapeInfo();
    ge::graphStatus GetTypeInfo();
    ge::graphStatus GetAttrInfo();
    ge::graphStatus GetStrideInfo();

    // Hardware information
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t ubBlockSize_ = 0;
    
    // Runtime information
    int64_t fixedUBSize = 0;
    
    // Input tensor shape information
    int64_t batchSize_ = 0;
    int64_t seqLen_ = 0;
    int64_t cuSeqLen_ = 0;  // For 2D input: first dimension of x, equals batch * seq_len
    int64_t dim_ = 0;
    int64_t kernelSize_ = 0;
    int64_t stateLen_ = 0;   // State length: second dimension of cacheState (K-1+m)

    // Data type information
    ge::DataType xDtype_;
    ge::DataType weightDtype_;
    ge::DataType convStatesDtype_;
    ge::DataType queryStartLocDtype_;
    ge::DataType cacheIndicesDtype_;
    ge::DataType numAcceptedTokenDtype_;
    size_t xDtypeSize_ = 0;

    // Attribute values
    int64_t activationMode_ = 0;
    int64_t xStride_ = 0;     
    int64_t cacheStride0_ = 0;
    int64_t cacheStride1_ = 0;
    int64_t padSlotId_ = -1;
    int64_t runMode_ = 0;
    int64_t xInputMode_ = 0;            // 0 for 3D [batch, seq_len, dim], 1 for 2D [cu_seq_len, dim]
    int64_t hasAcceptTokenNum_ = 0;     // Whether acceptTokenNum input is provided: 0 for false, 1 for true
    int64_t residualConnection_ = 0;    // Whether use residual connection: 0 for false, 1 for true

    // Inter-core tiling parameters (non-uniform split)
    int64_t limitedCoreNum_ = 0;      // Limited core number based on data size (for reference)
    int64_t usedCoreNum_ = 0;         // Actually used core number
    int64_t dimCoreCnt_ = 0;          // Number of cores for dim direction
    int64_t batchCoreCnt_ = 0;        // Number of cores for batch direction
    int64_t dimMainCoreCnt_ = 0;      // Number of big dim cores (base+1 blocks)
    int64_t dimTailCoreCnt_ = 0;      // Number of small dim cores (base blocks)
    int64_t mainCoredimLen_ = 0;        // Big core dim size ((base+1) * 128)
    int64_t tailCoredimLen_ = 0;         // Small core dim size (base * 128)
    int64_t batchMainCoreCnt_ = 0;    // Number of big batch cores
    int64_t batchTailCoreCnt_ = 0;    // Number of small batch cores
    int64_t mainCoreBatchNum_ = 0;        // Batch size for big cores
    int64_t tailCoreBatchNum_ = 0;    // Batch size for small cores

    // Intra-core tiling parameters UB loop
    int64_t loopNumBS_ = 0;                // Loops in BS direction for big cores
    int64_t loopNumDim_ = 0;               // Loops in Dim direction for big cores
    int64_t ubMainFactorBS_ = 0;               // UB BS factor for big cores
    int64_t ubTailFactorBS_ = 0;           // UB BS tail factor for big cores
    int64_t ubMainFactorDim_ = 0;              // UB Dim factor for big cores
    int64_t ubTailFactorDim_ = 0;          // UB Dim tail factor for big cores
    int64_t tailBlockloopNumBS_ = 0;       // Loops in BS direction for tail cores
    int64_t tailBlockloopNumDim_ = 0;      // Loops in Dim direction for tail cores
    int64_t tailBlockubFactorBS_ = 0;      // UB BS factor for tail cores
    int64_t tailBlockubTailFactorBS_ = 0;  // UB BS tail factor for tail cores
    int64_t tailBlockubFactorDim_ = 0;     // UB Dim factor for tail cores
    int64_t tailBlockubTailFactorDim_ = 0; // UB Dim tail factor for tail cores

    // TilingData object
    FusedCausalConv1dCutBHTilingData tilingData_;
};

} // namespace optiling

#endif // FUSED_CAUSAL_CONV1D_CUT_BH_TILING_H
