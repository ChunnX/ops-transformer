/**
 * Copyright (c) 2025 Huawei Technologies Co.,
    Ltd.
 * This program is free software,
    you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS,
    WITHOUT WARRANTIES OF ANY KIND,
    EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
    MERCHANTABILITY,
    OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_common_def.h
 * \brief
 */

#ifndef FLASH_ATTN_UTILS_H_
#define FLASH_ATTN_UTILS_H_

#include "util.h"
#include "kernel_operator_list_tensor_intf.h"
// #include "flash_attn_common_def.h"
// #include "../../../common/op_kernel/vector_common.h"
// #include "../../../common/op_kernel/memory_copy.h"

using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;


namespace fa_base_vector {
// // BLOCK和REPEAT的字节数
// constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
// // BLOCK和REPEAT的FP32元素数
// constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

}

static constexpr uint32_t FA_BYTE_BLOCK = 32;
static constexpr uint32_t FA_METADATA_HEADER_OFFSET = 16U * sizeof(uint32_t);


// namespace BaseApi {
// enum class GmFormat {
//     BSNGD = 0,
//     BNGSD = 1,
//     NGBSD = 2,
//     TNGD = 3,
//     NGTD = 4,
//     BSND = 5,
//     BNSD = 6,
//     TND = 7,
//     NTD = 8,
//     PA_BnBsND = 9,
//     PA_BnNBsD = 10,
//     PA_NZ = 11,
//     NGD = 12, // post_quant
//     ND = 13, //antiquant no PA
//     BS2 = 14,
//     BNS2 = 15,
//     PA_BnBs = 16, //antiquant PA
//     PA_BnNBs = 17,
//     BN2GS1S2 = 18, //PSE_GmFormat
//     SBNGD = 19,
//     SBND = 20
// };

// template <LayOutTypeEnum LAYOUT>
// __aicore__ inline constexpr GmFormat GetQueryGmFormat() {
//     if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
//         return GmFormat::BSNGD;
//     } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
//         return GmFormat::SBNGD;
//     } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
//         return GmFormat::BNGSD;
//     } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
//         return GmFormat::TNGD;
//     } else {
//         return GmFormat::NGTD;
//     }
// }

// template <LayOutTypeEnum LAYOUT, bool isPa=false>
// __aicore__ inline constexpr GmFormat GetKVGmFormat() {
//     // TODO: PA正确处理layout，区分BNBD、BBH、NZ
//     // return GmFormat::PA_BnNBsD;
//     if constexpr (isPa) {
//         return GmFormat::PA_BnNBsD;
//     }
//     if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
//         return GmFormat::BSND;
//     } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
//         return GmFormat::SBND;
//     } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
//         return GmFormat::BNSD;
//     } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
//         return GmFormat::TND;
//     } else {
//         return GmFormat::NTD;
//     }
// }

// /* ============确定Query的L1类型============= */
// template <typename INPUT_T, uint32_t dBaseSize>
// struct QL1BuffSel {
//     using Type = std::conditional_t<
//         std::is_same_v<INPUT_T, float> ||
//         (!(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
//            std::is_same_v<INPUT_T, fp8_e5m2_t> ||
//            std::is_same_v<INPUT_T, hifloat8_t>) && dBaseSize > 256),
//         BuffersPolicySingleBuffer<BufferType::L1>,
//         BuffersPolicyDB<BufferType::L1>>;
// };

// /* ============确定Key的L1类型============= */
// template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
// struct KVL1BuffSel {
//     constexpr static bool isFP8DType =  
//             std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
//            std::is_same_v<INPUT_T, fp8_e5m2_t> ||
//            std::is_same_v<INPUT_T, hifloat8_t>;
//     using Type = std::conditional_t<
//             (isFP8DType && s2BaseSize == 128 && dBaseSize == 576),
//             BuffersPolicy4buff<BufferType::L1>,
//             std::conditional_t<
//                 (!(isFP8DType) && s2BaseSize == 256 && dBaseSize > 128),
//                 BuffersPolicySingleBuffer<BufferType::L1>,
//                 BuffersPolicyDB<BufferType::L1>
//             >
//         >;
// };

// /* ============确定L0A的类型============= */
// template <typename INPUT_T>
// struct L0ABuffSel {
//     using Type = std::conditional_t<
//         std::is_same_v<INPUT_T, float>,
//         BuffersPolicySingleBuffer<BufferType::L0A>,
//         BuffersPolicyDB<BufferType::L0A>>;
// };
// /* ============确定L0B的类型============= */
// template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
// struct L0BBuffSel {
//     using Type = std::conditional_t<
//         std::is_same_v<INPUT_T, float> || (s2BaseSize == 256 && dBaseSize > 128 && 
//         !(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
//            std::is_same_v<INPUT_T, fp8_e5m2_t> ||
//            std::is_same_v<INPUT_T, hifloat8_t>)),
//         BuffersPolicySingleBuffer<BufferType::L0B>,
//         BuffersPolicyDB<BufferType::L0B>>;
// };
// /* ============确定L0C的类型============= */
// template <typename INPUT_T, uint32_t s1BaseSize, uint32_t s2BaseSize, uint32_t dVBaseSize>
// struct L0CBuffSel {
//     using Type = std::conditional_t<
//         (s1BaseSize * s2BaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4 && s1BaseSize * dVBaseSize * FLOAT_BYTES <= (L0C_SIZE * KB_TO_BYTES) / NUM_4),
//         BuffersPolicy4buff<BufferType::L0C>,
//         BuffersPolicyDB<BufferType::L0C>>;
// };
// }  // namespace BaseApi

#endif  // FLASH_ATTN_COMMON_DEF_H_