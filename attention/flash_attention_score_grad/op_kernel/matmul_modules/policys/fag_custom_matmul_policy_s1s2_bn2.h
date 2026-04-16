/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fag_custom_matmul_policy_s1s2_bn2.h
 * \brief
 */

#ifndef FAG_CUSTOM_MATMUL_POLICY_S1S2_BN2_KERNEL_H_
#define FAG_CUSTOM_MATMUL_POLICY_S1S2_BN2_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "../fag_data_common.h"
#include "../copy_in_buffer/fag_copy_in_buffer_base.h"
#include "../copy_cube_in/fag_copy_cube_in_base.h"
#include "../copy_cube_in/fag_s1s2_bn2_sab/fag_copy_cube_in_mm_256_256_128.h"

namespace AscendC {
    
namespace Impl {
namespace Detail {

// ---------------------------------------------------policys-----------------------------------------------------------
/**
 * support: D<=80; singleS1/S2=512
 * limit: mm base块相同; mm-dyv/qk orderN; 
 */
template <const auto& MM_CFG, typename IMPL, typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class FagMatmulPolicy : public MatmulPolicy<MM_CFG, IMPL, A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> {
public:
    using UserDefDataType = FagFlagData;
    using CopyCubeInA = T12FagCopyCubeIn<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CopyCubeInB = T12FagCopyCubeIn<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferA = FagCubeInBuffer<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
    using CubeInBufferB = FagCubeInBuffer<IMPL, MatmulInputBType<B_TYPE, typename A_TYPE::T>, MM_CFG>;
    using Context = FagContext<IMPL, MatmulInputAType<A_TYPE, typename A_TYPE::T>, MM_CFG>;
};
}
}
}
#endif // FAG_CUSTOM_MATMUL_POLICY_S1S2_BN2_KERNEL_H_
