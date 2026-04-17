/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_APT_ANTIQUANT_H
#define GROUPED_MATMUL_APT_ANTIQUANT_H

#if defined(V310_GMM_ANTI_QUANT)
#include "arch35/weight_quant_basic_block/basic_block_config.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_weight_quant_basic_controller.h"
#include "arch35/weight_quant_basic_block/grouped_matmul_weight_quant_resplit_controller.h"
#include "arch35/weight_quant_basic_block/weight_quant_basic_block.h"
#include "arch35/weight_quant_basic_block/weight_quant_vcv_basic_block.h"
#include "arch35/weight_quant_basic_block/weight_quant_tiling_key.h"
using WeightQuantBatchMatmulV2::Arch35::QuantType;
using WeightQuantBatchMatmulV2::Arch35::A16MXF4_NZKN;
using WeightQuantBatchMatmulV2::Arch35::MXA8W4_NZNK;
using WeightQuantBatchMatmulV2::Arch35::S8S4_NZKN_G;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantMatmulBasicBlock;
using WeightQuantBatchMatmulV2::Arch35::WeightQuantVcvMatmulBasicBlock;
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_0 = {2, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_1 = {4, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_2 = {2, 1024};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_3 = {4, 256};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_4 = {3, 512};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_5 = {3, 384};
static constexpr VecAntiQuantConfig VEC_ANTIQUANT_CONFIG_DYNAMIC = {4, 0};
#if defined(DT_FLOAT) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_WEIGHT == DT_FLOAT
    #undef DTYPE_WEIGHT
    #define DTYPE_WEIGHT fp4x2_e2m1_t
#endif
#if defined(DT_INT32) && defined(ORIG_DTYPE_WEIGHT) && ORIG_DTYPE_WEIGHT == DT_INT32
    #undef DTYPE_WEIGHT
    #define DTYPE_WEIGHT AscendC::int4b_t
    #undef ORIG_DTYPE_WEIGHT
    #define ORIG_DTYPE_WEIGHT DT_INT4
#endif

#define INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(T_X, T_W, T_BIAS, T_Y, templateClass, ...)                   \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);                 \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                         \
        GET_TILING_DATA_MEMBER_ADDR(GMMWeightQuantTilingData, gmmArray, gmmArrayAddr_, tiling);                        \
        templateClass<T_X, T_W, T_BIAS, T_Y, __VA_ARGS__> op;                                                         \
        op.Init(x, weight, antiquantScale, antiquantOffset, bias, groupList, y, &gmmBaseParams_,                       \
                &mmTilingData_, tiling, gmmArrayAddr_, &tPipe);                                                        \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(T_X, T_W, T_AS, T_SC, T_BIAS, T_Y,                         \
                                                           templateClass, ...)                                         \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);                 \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                         \
        templateClass<T_X, T_W, T_AS, T_SC, float,                                                                    \
                      T_BIAS, T_Y, WeightQuantMatmulBasicBlock, __VA_ARGS__> op;                                       \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL(T_X, T_W, T_AS, T_SC, T_PTS, T_BIAS, T_Y,                  \
                                                           templateClass, ...)                                         \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);                 \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                         \
        templateClass<T_X, T_W, T_AS, T_SC, T_PTS,                                                                    \
                      T_BIAS, T_Y, WeightQuantMatmulBasicBlock, __VA_ARGS__> op;                                       \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(T_X, T_W, T_AS, T_SC, T_PTS, T_BIAS, T_Y,                     \
                                                        templateClass, ...)                                            \
    do {                                                                                                               \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, gmmWeightQuantParam, gmmBaseParams_, tiling);                 \
        GET_TILING_DATA_MEMBER(GMMWeightQuantTilingData, mmTilingData, mmTilingData_, tiling);                         \
        templateClass<T_X, T_W, T_AS, T_SC, T_PTS,                                                                    \
                      T_BIAS, T_Y, WeightQuantVcvMatmulBasicBlock, __VA_ARGS__> op;                                    \
        op.Init(x, weight, scale, antiquantScale, antiquantOffset, bias, groupList, perTokenScale, y, &gmmBaseParams_, \
                &mmTilingData_, tiling, &tPipe);                                                                       \
        op.Process();                                                                                                  \
    } while (0)

#define GROUPED_MATMUL_ANTIQUANT_IMPL_BODY(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeY, TDtypeScale, TDtypeAntiquantScale, \
                                           TDtypePerTokenScale, kOrigDtypeX, kOrigDtypeAntiquantScale,                  \
                                           W_TYPE, OFFSET_OR_BIAS_EXIT, C_QUANT_TYPE, W_QUANT_TYPE, WQ_B_TRANS,        \
                                           WQ_A_TRANS, TEMPLATE_CUSTOM_SC, ALGORITHM_SUB_CATEGORY, ALGORITHM_CATEGORY) \
    REGISTER_TILING_DEFAULT(GMMWeightQuantTilingData);                                                                 \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                                                 \
    if constexpr (kOrigDtypeX == DT_INT8) {                                                                            \
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST \
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_NO_TRANS         \
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_512_BUF_NUM_DEFAULT         \
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) { \
            INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(TDtypeX, TDtypeWeight, TDtypeAntiquantScale,                \
                TDtypeScale, TDtypePerTokenScale, TDtypeBias, TDtypeY,                                                 \
                GMMWeightQuantResplitController, S8S4_NZKN_G, VEC_ANTIQUANT_CONFIG_4);                                 \
        } else if constexpr (W_TYPE == WQGMM_FRACTAL_NZ &&                                                             \
            OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST &&                                  \
            C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL &&                                         \
            WQ_B_TRANS == WQGMM_NO_TRANS && WQ_A_TRANS == WQGMM_NO_TRANS &&                                            \
            TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_384_BUF_NUM_3 &&                                               \
            ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT &&                                                    \
            ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {                                                            \
            INVOKE_GMM_WEIGHT_QUANT_VCV_CONTROLLER_OP_IMPL(TDtypeX, TDtypeWeight, TDtypeAntiquantScale,                \
                TDtypeScale, TDtypePerTokenScale, TDtypeBias, TDtypeY,                                                 \
                GMMWeightQuantResplitController, S8S4_NZKN_G, VEC_ANTIQUANT_CONFIG_5);                                 \
        }                                                                                                              \
    } else if constexpr (kOrigDtypeX == DT_FLOAT8_E4M3FN) {                                                            \
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST \
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_MX && WQ_B_TRANS == WQGMM_TRANS                     \
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_DYNAMIC_BUF_NUM_4           \
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) { \
            INVOKE_GMM_WEIGHT_QUANT_MXA8W4_CONTROLLER_OP_IMPL(TDtypeX, TDtypeWeight, TDtypeAntiquantScale,             \
                TDtypeScale, TDtypePerTokenScale, TDtypeBias, TDtypeY,                                                 \
                GMMWeightQuantResplitController, MXA8W4_NZNK, VEC_ANTIQUANT_CONFIG_DYNAMIC);                           \
        }                                                                                                              \
    } else if constexpr (kOrigDtypeAntiquantScale == DT_FLOAT8_E8M0) {                                                 \
        if constexpr (W_TYPE == WQGMM_FRACTAL_NZ && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST \
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_MX && WQ_B_TRANS == WQGMM_NO_TRANS                  \
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4               \
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) { \
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(TDtypeX, TDtypeWeight, TDtypeAntiquantScale,            \
                TDtypeScale, TDtypeBias, TDtypeY,                                                                      \
                GMMWeightQuantResplitController, A16MXF4_NZKN, VEC_ANTIQUANT_CONFIG_3);                                \
        }                                                                                                              \
    } else {                                                                                                           \
        if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST     \
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS            \
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4               \
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) { \
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, false,                        \
                                                QuantType::NONE, CubeFormat::ND};                                     \
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(TDtypeX, TDtypeWeight, TDtypeAntiquantScale,            \
                TDtypeScale, TDtypeBias, TDtypeY,                                                                      \
                GMMWeightQuantResplitController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);                                     \
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST   \
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS            \
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4               \
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_TAIL_RESPLIT && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) { \
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, true,                         \
                                                   QuantType::NONE, CubeFormat::ND};                                  \
            INVOKE_GMM_WEIGHT_QUANT_RESPLIT_CONTROLLER_OP_IMPL(TDtypeX, TDtypeWeight, TDtypeAntiquantScale,            \
                TDtypeScale, TDtypeBias, TDtypeY,                                                                      \
                GMMWeightQuantResplitController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);                                     \
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST \
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS            \
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4               \
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {  \
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, false,                        \
                                                   QuantType::NONE, CubeFormat::ND};                                  \
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeY,               \
                GMMWeightQuantBasicController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);                                       \
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST   \
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_TRANS            \
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4               \
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {  \
            static constexpr WqmmConfig wqmmCfg = {false, true, QuantType::PER_CHANNEL, true,                         \
                                                   QuantType::NONE, CubeFormat::ND};                                  \
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeY,               \
                GMMWeightQuantBasicController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);                                       \
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_NOT_EXIST_BIAS_NOT_EXIST \
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_NO_TRANS         \
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4               \
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {  \
            static constexpr WqmmConfig wqmmCfg = {false, false, QuantType::PER_CHANNEL, false,                       \
                                                   QuantType::NONE, CubeFormat::ND};                                  \
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeY,               \
                GMMWeightQuantBasicController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);                                       \
        } else if constexpr (W_TYPE == WQGMM_ND && OFFSET_OR_BIAS_EXIT == WQGMM_ANTIQUANT_OFFSET_EXIST_BIAS_NOT_EXIST   \
            && C_QUANT_TYPE == WQGMM_NONE && W_QUANT_TYPE == WQGMM_PER_CHANNEL && WQ_B_TRANS == WQGMM_NO_TRANS         \
            && WQ_A_TRANS == WQGMM_NO_TRANS && TEMPLATE_CUSTOM_SC == WQGMM_MTE2_INNER_SIZE_256_BUF_NUM_4               \
            && ALGORITHM_SUB_CATEGORY == WQGMM_N_FIRST_BASIC_BLOCK && ALGORITHM_CATEGORY == WQGMM_VECTOR_ANTIQUANT) {  \
            static constexpr WqmmConfig wqmmCfg = {false, false, QuantType::PER_CHANNEL, true,                        \
                                                   QuantType::NONE, CubeFormat::ND};                                  \
            INVOKE_GMM_WEIGHT_QUANT_BASIC_CONTROLLER_OP_IMPL(TDtypeX, TDtypeWeight, TDtypeBias, TDtypeY,               \
                GMMWeightQuantBasicController, wqmmCfg, VEC_ANTIQUANT_CONFIG_3);                                       \
        }                                                                                                              \
    }

#endif
#endif