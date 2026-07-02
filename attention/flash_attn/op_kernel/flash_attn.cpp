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

    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

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

    if constexpr (useDn) {
        using CubeBlock = BaseApi::FANoQuantGqaBlockCubeDn<INPUT_T, float, inputLayoutType, s1TemplateType,
                                                           s2TemplateType, dTemplateType, dVTemplateType, KvLayoutType,
                                                           useDn, bmm2Write2Ub, splitD>;
        using VecFaBlock = BaseApi::FANoQuantGqaBlockVecDn<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                           s1TemplateType, s2TemplateType, dTemplateType,
                                                           dVTemplateType, hasAttenMask, false, KvLayoutType,
                                                           bmm2Write2Ub, splitD>;
        using VecFdBlock = BaseApi::FiaBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                         s1TemplateType, s2TemplateType, dTemplateType,
                                                         dVTemplateType, hasAttenMask, false, KvLayoutType, useDn,
                                                         bmm2Write2Ub, splitD>;
        using VecDummy = BaseApi::FANoQuantGqaBlockVecDummyDn<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                              s1TemplateType, s2TemplateType, dTemplateType,
                                                              dVTemplateType, hasAttenMask, false, KvLayoutType,
                                                              bmm2Write2Ub, splitD>;
        using CubeDummy = BaseApi::FANoQuantGqaBlockCubeDummyDn<INPUT_T, float, inputLayoutType, s1TemplateType,
                                                                s2TemplateType, dTemplateType, dVTemplateType,
                                                                KvLayoutType, useDn, bmm2Write2Ub, splitD>;
#ifdef __DAV_C310_CUBE__
        using Kernel = BaseApi::FlashAttentionNoQuantGqaKernelDn<CubeBlock, VecDummy, VecDummy>;
#else
        using Kernel = BaseApi::FlashAttentionNoQuantGqaKernelDn<CubeDummy, VecFaBlock, VecFdBlock>;
#endif
        Kernel op;
        op.Init(query, key, value, attnMask, cuSeqLensQ, cuSeqLensKv, blockTable, sinks, softmaxLse, attnOut, user,
                metadata, sequsedQ, sequsedKv, tilingData);
        op.Process();
    } else {
        using CubeBlock = BaseApi::FANoQuantGqaBlockCubeNd<INPUT_T, float, inputLayoutType, s1TemplateType,
                                                           s2TemplateType, dTemplateType, dVTemplateType, KvLayoutType,
                                                           useDn, bmm2Write2Ub, splitD>;
        using VecFaBlock = BaseApi::FANoQuantGqaBlockVecNd<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                           s1TemplateType, s2TemplateType, dTemplateType,
                                                           dVTemplateType, hasAttenMask, false, KvLayoutType,
                                                           bmm2Write2Ub, splitD>;
        using VecFdBlock = BaseApi::FiaBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                         s1TemplateType, s2TemplateType, dTemplateType,
                                                         dVTemplateType, hasAttenMask, false, KvLayoutType, useDn,
                                                         bmm2Write2Ub, splitD>;
        using VecDummy = BaseApi::FANoQuantGqaBlockVecDummyNd<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                              s1TemplateType, s2TemplateType, dTemplateType,
                                                              dVTemplateType, hasAttenMask, false, KvLayoutType,
                                                              bmm2Write2Ub, splitD>;
        using CubeDummy = BaseApi::FANoQuantGqaBlockCubeDummyNd<INPUT_T, float, inputLayoutType, s1TemplateType,
                                                                s2TemplateType, dTemplateType, dVTemplateType,
                                                                KvLayoutType, useDn, bmm2Write2Ub, splitD>;
#ifdef __DAV_C310_CUBE__
        using Kernel = BaseApi::FlashAttentionNoQuantGqaKernelNd<CubeBlock, VecDummy, VecDummy>;
#else
        using Kernel = BaseApi::FlashAttentionNoQuantGqaKernelNd<CubeDummy, VecFaBlock, VecFdBlock>;
#endif
        Kernel op;
        op.Init(query, key, value, attnMask, cuSeqLensQ, cuSeqLensKv, blockTable, sinks, softmaxLse, attnOut, user,
                metadata, sequsedQ, sequsedKv, tilingData);
        op.Process();
    }
#endif
}
