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
 * \file flash_mla_with_kvcache_gm_format.h
 * \brief GM 格式推导（Q/KV），与 fused_infer_attention_score/op_kernel/arch35/
 *        flash_attention_score_block_cube.h:30-70 逐字一致（只读参考）。
 *        四件套（flash_mla_with_kvcache_block_cube_noquant_mla.h:44-47）直接调用这两个模板；
 *        FIA 的 TU 经由其自家 block_cube_base 链自然携带，flash_mla_with_kvcache 的 kernel 拷贝树
 *        不含跨算子文件，故在本算子内部补一份（Todo 14 实测依赖）。
 */

#ifndef FLASH_MLA_WITH_KVCACHE_GM_FORMAT_H_
#define FLASH_MLA_WITH_KVCACHE_GM_FORMAT_H_

#include "kernel_operator_list_tensor_intf.h"
#include "../../../common/op_kernel/memcopy/gm_layout.h"

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

} // namespace BaseApi

#endif // FLASH_MLA_WITH_KVCACHE_GM_FORMAT_H_
