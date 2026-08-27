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
 * \file gated_activation.h
 * \brief A5（arch35）门控激活统一入口和运行时分派。
 */

#ifndef MEGA_MOE_ARCH35_ACTIVATION_GATED_ACTIVATION_H
#define MEGA_MOE_ARCH35_ACTIVATION_GATED_ACTIVATION_H

#if defined(__DAV_C310__)
#include "activation_common.h"
#include "situ_activation.h"
#include "swiglu_activation.h"
#include "swiglu_oai_activation.h"

namespace MegaMoeImpl {
namespace Activation {

template <typename InputType, bool FuseTopkWeight>
__aicore__ inline void RunGatedActivation(
    const GatedActivationArgs<InputType> &args, const ActivationParams &params)
{
    if (params.mode == MegaMoeActMode::SITU) {
        const float invBeta = params.beta != 0.0f ? 1.0f / params.beta : 1.0f;
        if (params.subMode == MegaMoeActSubMode::LINEAR) {
            const float invAlpha = params.alpha != 0.0f ? 1.0f / params.alpha : 1.0f;
            const SituParams situParams{params.clampLimit, params.beta, invBeta, params.alpha, invAlpha};
            RunSiTU<InputType, FuseTopkWeight, true>(args, situParams);
        } else {
            const SituParams situParams{params.clampLimit, params.beta, invBeta, 1.0f, 1.0f};
            RunSiTU<InputType, FuseTopkWeight, false>(args, situParams);
        }
    } else if (params.mode == MegaMoeActMode::SWIGLU_STEP) {
        const SwiGluParams swiGluParams{params.clampLimit};
        RunSwiGLU<InputType, FuseTopkWeight, true>(args, swiGluParams);
    } else if (params.mode == MegaMoeActMode::SWIGLU_OAI) {
        const SwiGluOaiParams swiGluOaiParams{params.clampLimit, params.alpha, params.beta};
        RunSwiGLUOai<InputType, FuseTopkWeight>(args, swiGluOaiParams);
    } else {
        const SwiGluParams swiGluParams{params.clampLimit};
        RunSwiGLU<InputType, FuseTopkWeight, false>(args, swiGluParams);
    }
}

} // namespace Activation
} // namespace MegaMoeImpl

#endif // defined(__DAV_C310__)
#endif // MEGA_MOE_ARCH35_ACTIVATION_GATED_ACTIVATION_H
