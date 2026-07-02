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
 * \file flash_attn.cpp
 * \brief FlashAttn Kernel 唯一入口；按 tiling key 编译期推导模板参数，分发 Dn / Nd 两套模板。
 *        参考 quant_flash_attn.cpp 单层 __global__ 风格。
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif

#include "arch35/flash_attn_template_tiling_key.h"
#include "arch35/flash_attn_tiling_data.h"
#include "arch35/fa_static_buffer.h"
#include "utils/flash_attn_utils.h"
#include "utils/flash_attn_common_def.h"
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase.h"
#include "arch35/flash_attn_kernel_noquant_gqa_dn.h"
#include "arch35/flash_attn_kernel_noquant_gqa_nd.h"

using namespace AscendC;

template <uint8_t inOutLayoutType>
__aicore__ inline constexpr FA_LAYOUT GetQueryLayout()
{
    static_assert((inOutLayoutType == InOutLayoutType_BSND) ||
                  (inOutLayoutType == InOutLayoutType_BNSD) ||
                  (inOutLayoutType == InOutLayoutType_BNSD_BSND) ||
                  (inOutLayoutType == InOutLayoutType_TND),
                  "Get Query Layout fail, inOutLayoutType is incorrect");
    if constexpr (inOutLayoutType == InOutLayoutType_BSND) {
        return FA_LAYOUT::BSND;
    } else if constexpr (inOutLayoutType == InOutLayoutType_BNSD || inOutLayoutType == InOutLayoutType_BNSD_BSND) {
        return FA_LAYOUT::BNSD;
    } else if constexpr (inOutLayoutType == InOutLayoutType_TND) {
        return FA_LAYOUT::TND;
    }
}
 
template <uint8_t inOutLayoutType>
__aicore__ inline constexpr FA_LAYOUT GetOutLayout()
{
    static_assert((inOutLayoutType == InOutLayoutType_BSND) ||
                  (inOutLayoutType == InOutLayoutType_BNSD) ||
                  (inOutLayoutType == InOutLayoutType_BNSD_BSND) ||
                  (inOutLayoutType == InOutLayoutType_TND),
                  "Get AttnOut Layout fail, inOutLayoutType is incorrect");
    if constexpr (inOutLayoutType == InOutLayoutType_BSND || inOutLayoutType == InOutLayoutType_BNSD_BSND) {
        return FA_LAYOUT::BSND;
    } else if constexpr (inOutLayoutType == InOutLayoutType_BNSD) {
        return FA_LAYOUT::BNSD;
    } else if constexpr (inOutLayoutType == InOutLayoutType_TND) {
        return FA_LAYOUT::TND;
    }
}
 
template <uint8_t inOutLayoutType, uint8_t KV_STORAGE_MODE>
__aicore__ inline constexpr FA_LAYOUT GetKvLayout()
{
    static_assert((KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) ||
                  (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) ||
                  (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD) ||
                  (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_NZ),
                  "Get Key/Value Layout fail, KV_STORAGE_MODE is incorrect");
    if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) {
        return GetQueryLayout<inOutLayoutType>();
    } else if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) {
        return FA_LAYOUT::BSND; // block内的格式类似于BSND
    } else if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD) {
        return FA_LAYOUT::BNSD; // block内的格式类似于BNSD
    } else if constexpr (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_NZ) {
        return FA_LAYOUT::NZ; // block内的格式类似于BNSD
    }
}

template <uint8_t KV_STORAGE_MODE>
__aicore__ inline constexpr bool IsPageAttention()
{
    static_assert((KV_STORAGE_MODE == KV_STORAGE_MODE_CONTINUE) ||
                  (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BSND) ||
                  (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_BNSD) ||
                  (KV_STORAGE_MODE == KV_STORAGE_MODE_PA_NZ),
                  "Get PAGE_ATTENTION flag fail, KV_STORAGE_MODE is incorrect");
    return (KV_STORAGE_MODE != KV_STORAGE_MODE_CONTINUE);
}

template <uint8_t inOutLayoutType, uint8_t KvLayoutType, bool hasAttenMask, uint8_t config>
__global__ __aicore__ void
flash_attn(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *blockTable,
           __gm__ uint8_t *cuSeqLensQ, __gm__ uint8_t *cuSeqLensKv, __gm__ uint8_t *sequsedQ, __gm__ uint8_t *sequsedKv,
           __gm__ uint8_t *sinks, __gm__ uint8_t *attnMask, __gm__ uint8_t *metadata, __gm__ uint8_t *attnOut,
           __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(optiling::FlashAttnTilingData);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (inOutLayoutType == InOutLayoutType_TND && KvLayoutType == KvLayoutType_NO_PA) {
        dc_preload(reinterpret_cast<__gm__ uint64_t *>(cuSeqLensQ), 0);
        dc_preload(reinterpret_cast<__gm__ uint64_t *>(cuSeqLensKv), 0);
    }

#if (ORIG_DTYPE_Q == DT_BF16)
    using INPUT_T = bfloat16_t;
    using OUT_T = bfloat16_t;
#elif (ORIG_DTYPE_Q == DT_FLOAT16)
    using INPUT_T = half;
    using OUT_T = half;
#endif

#if (ORIG_DTYPE_Q == DT_BF16) || (ORIG_DTYPE_Q == DT_FLOAT16)
    fa_base_matmul::idCounterNum = 0;

    constexpr FA_LAYOUT qLayout = GetQueryLayout<inOutLayoutType>();
    constexpr FA_LAYOUT outLayout = GetOutLayout<inOutLayoutType>();
    constexpr FA_LAYOUT kvLayout = GetKvLayout<inOutLayoutType, KvLayoutType>();
    constexpr bool pageAttention = IsPageAttention<KvLayoutType>();

    // config (1-bit), D=128 fixed
    //   config=0: sOuter=64, sInner=128 → S1=128, S2=128, D=128, DV=128
    //   config=1: sOuter=32, sInner=256 → S1=64,  S2=256, D=128, DV=128
    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(ConfigValue[config].d);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(ConfigValue[config].dv);

    constexpr bool hasRopeConst = false;
    constexpr TPosition bmm2OutPos =
        BaseApi::GetC2Position(dVTemplateType,
                               BaseApi::UbOutCondition<INPUT_T>(false, PseTypeEnum::PSE_NONE_TYPE, hasAttenMask, false,
                                                                false, (uint32_t)s1TemplateType == 64),
                               ((uint32_t)s2TemplateType == 256 && (uint32_t)s1TemplateType == 64), false);
    constexpr bool useDn =
        BaseApi::IsDn(false, false, PseTypeEnum::PSE_NONE_TYPE, hasAttenMask, false, (uint32_t)s1TemplateType == 64,
                      dTemplateType, hasRopeConst, false, true, IsSameType<INPUT_T, hifloat8_t>::value);
    constexpr bool bmm2Write2Ub = (bmm2OutPos == TPosition::VECCALC);
    constexpr bool splitD = ((uint16_t)dVTemplateType > (uint16_t)DTemplateType::Aligned256);

    size_t offset = (size_t)(&((FlashAttnTilingData *)0)->baseTiling);
    const __gm__ FlashAttnNoQuantTilingArch35 *__restrict tilingData =
        (const __gm__ FlashAttnNoQuantTilingArch35 *__restrict)(tiling + offset);

    InitSocState();
    AscendC::SyncAll();

    using FA_T = BaseApi::FAType<INPUT_T, OUT_T, pageAttention, qLayout, kvLayout, outLayout, s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType, hasAttenMask, bmm2Write2Ub, splitD>;

    if constexpr (useDn) {
        using CubeBlock = BaseApi::FANoQuantGqaBlockCubeDn<FA_T>;
        using VecFaBlock = BaseApi::FANoQuantGqaBlockVecDn<FA_T>;
        using VecFdBlock = BaseApi::FiaBlockVecFlashDecode<FA_T>;
        using VecDummy = BaseApi::FANoQuantGqaBlockVecDummyDn<FA_T>;
        using CubeDummy = BaseApi::FANoQuantGqaBlockCubeDummyDn<FA_T>;
#ifdef __DAV_C310_CUBE__
        using Kernel = BaseApi::FlashAttentionNoQuantGqaKernelDn<FA_T, CubeBlock, VecDummy, VecDummy>;
#else
        using Kernel = BaseApi::FlashAttentionNoQuantGqaKernelDn<FA_T, CubeDummy, VecFaBlock, VecFdBlock>;
#endif
        Kernel op;
        op.Init(query, key, value, blockTable, cuSeqLensQ, cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata,
                attnOut, softmaxLse, user, tilingData);
        op.Process();
    } else {
        using CubeBlock = BaseApi::FANoQuantGqaBlockCubeNd<FA_T>;
        using VecFaBlock = BaseApi::FANoQuantGqaBlockVecNd<FA_T>;
        using VecFdBlock = BaseApi::FiaBlockVecFlashDecode<FA_T>;
        using VecDummy = BaseApi::FANoQuantGqaBlockVecDummyNd<FA_T>;
        using CubeDummy = BaseApi::FANoQuantGqaBlockCubeDummyNd<FA_T>;
#ifdef __DAV_C310_CUBE__
        using Kernel = BaseApi::FlashAttentionNoQuantGqaKernelNd<FA_T, CubeBlock, VecDummy, VecDummy>;
#else
        using Kernel = BaseApi::FlashAttentionNoQuantGqaKernelNd<FA_T, CubeDummy, VecFaBlock, VecFdBlock>;
#endif
        Kernel op;
        op.Init(query, key, value, blockTable, cuSeqLensQ, cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata,
                attnOut, softmaxLse, user, tilingData);
        op.Process();
    }
#endif
}
