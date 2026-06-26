/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_block_cube_noquant_gqa_comm.h
 * \brief Cube Dn/Nd 共用辅助类型与函数（GetQueryGmFormat / GetKVGmFormat / L0ABuffSel / L0BBuffSel）。
 *        置于 op_kernel/utils，flash_attn.cpp 同时 include Dn、Nd 头文件时仅编译一份，避免重定义。
 */
#ifndef FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_COMM_H_
#define FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_COMM_H_

#include "../../../common/op_kernel/offset_calculator.h"
#include "../arch35/fa_static_buffer.h"
#include "../../../common/op_kernel/matmul.h"
#include "../../../common/op_kernel/FixpipeOut.h"
#include "../../../common/op_kernel/memory_copy_arch35.h"
#include "../../../common/op_kernel/arch35/infer_flash_attention_comm.h"
#include "../../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;

namespace BaseApi {

template <LayOutTypeEnum LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
        return GmFormat::BSNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
        return GmFormat::SBNGD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
        return GmFormat::BNGSD;
    } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
        return GmFormat::TNGD;
    } else {
        return GmFormat::NGTD;
    }
}

template <LayOutTypeEnum LAYOUT, uint8_t KvLayoutType = 0, bool isPa = false>
__aicore__ inline constexpr GmFormat GetKVGmFormat()
{
    if constexpr (KvLayoutType == 0) { // KvLayoutType_NO_PA
        if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BSH) {
            return GmFormat::BSND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_SBH) {
            return GmFormat::SBND;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_BNSD) {
            return GmFormat::BNSD;
        } else if constexpr (LAYOUT == LayOutTypeEnum::LAYOUT_TND) {
            return GmFormat::TND;
        } else {
            return GmFormat::NTD;
        }
    } else if constexpr (KvLayoutType == 1) { // KvLayoutType_PA_BBH
        return GmFormat::PA_BnBsND;
    } else if constexpr (KvLayoutType == 2) { // KvLayoutType_PA_BNBD
        return GmFormat::PA_BnNBsD;
    } else { // KvLayoutType_PA_NZ
        return GmFormat::PA_NZ;
    }
}

/* ============确定L0A的类型（仅供matmul使用）============= */
template <typename INPUT_T>
struct L0ABuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float>, BuffersPolicySingleBuffer<BufferType::L0A>,
                                    BuffersPolicyDB<BufferType::L0A>>;
};
/* ============确定L0B的类型（仅供matmul使用）============= */
template <typename INPUT_T, uint32_t s2BaseSize, uint32_t dBaseSize>
struct L0BBuffSel {
    using Type = std::conditional_t<std::is_same_v<INPUT_T, float> ||
                                        (s2BaseSize == 256 && dBaseSize > 128 &&
                                         !(std::is_same_v<INPUT_T, fp8_e4m3fn_t> ||
                                           std::is_same_v<INPUT_T, fp8_e5m2_t> || std::is_same_v<INPUT_T, hifloat8_t>)),
                                    BuffersPolicySingleBuffer<BufferType::L0B>, BuffersPolicyDB<BufferType::L0B>>;
};

} // namespace BaseApi

#endif // FLASH_ATTENTION_NOQUANT_GQA_BLOCK_CUBE_COMM_H_
