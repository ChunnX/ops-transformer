/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_FT_CATLASS_ARCH_RESOURCE_AIV_HPP
#define MATMUL_FT_CATLASS_ARCH_RESOURCE_AIV_HPP

#include "catlass/arch/resource.hpp"

namespace FTSelf {

template <class ArchTag>
struct ResourceAIV {
public:
    AscendC::TPipe pipe;
    Catlass::Arch::LocalTensorBuffer<ArchTag, AscendC::TPosition::VECCALC> ubBuf;

    CATLASS_DEVICE
    ResourceAIV()
    {
        // TPipe initialization inserts synchronization that may conflict with callers.
        pipe.Destroy();
    }
};

}  // namespace FTSelf

#endif  // MATMUL_FT_CATLASS_ARCH_RESOURCE_AIV_HPP
