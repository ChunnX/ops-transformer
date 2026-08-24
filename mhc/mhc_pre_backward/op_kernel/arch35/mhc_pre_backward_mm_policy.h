/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * either express or implied, including but not limited to non-infringement, merchantability, or fitness for a particular purpose.
 */

/*!
 * \file mhc_pre_backward_mm_policy.h
 * \brief MHC C0 VECIN Split-N copy-out policy.
 */

#ifndef ASCENDC_MHC_PRE_BACKWARD_MM_POLICY_H
#define ASCENDC_MHC_PRE_BACKWARD_MM_POLICY_H

#include "mc2/3rd/mat_mul_v3/op_kernel/arch35/mm_extension_interface/mm_copy_cube_out.h"

namespace MhcPreBackwardUtils {

template <const auto &MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class MhcPreBackwardSplitNMatmulPolicy
    : public AscendC::Impl::Detail::MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using CopyCubeOut = MatmulCommon::Mc2MMCustomCopyCubeOut<
        IMPL, A_TYPE, B_TYPE, C_TYPE, MM_CFG, AscendC::McgShfMode::DUAL_DST_SPLIT_N>;
};

} // namespace MhcPreBackwardUtils

#endif // ASCENDC_MHC_PRE_BACKWARD_MM_POLICY_H
