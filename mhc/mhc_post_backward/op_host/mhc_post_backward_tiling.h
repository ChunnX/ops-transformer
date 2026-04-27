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
 * \file mhc_post_backward_tiling.h
 * \brief
 */

#ifndef MHC_POST_BACKWARD_TILING_H
#define MHC_POST_BACKWARD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MhcPostBackwardTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, coreUsed);
    TILING_DATA_FIELD_DEF(uint64_t, frontCore);
    TILING_DATA_FIELD_DEF(uint64_t, tailCore);
    TILING_DATA_FIELD_DEF(uint64_t, singleCoreBS);
    TILING_DATA_FIELD_DEF(uint64_t, tailBS);
    TILING_DATA_FIELD_DEF(uint64_t, dFPostResSize);
    TILING_DATA_FIELD_DEF(uint64_t, xSize);
    TILING_DATA_FIELD_DEF(uint64_t, hResSize);
    TILING_DATA_FIELD_DEF(uint64_t, hOutSize);
    TILING_DATA_FIELD_DEF(uint64_t, hPostSize);
    TILING_DATA_FIELD_DEF(uint64_t, channel);
    TILING_DATA_FIELD_DEF(uint64_t, blockChannel);
    TILING_DATA_FIELD_DEF(uint64_t, n);
    TILING_DATA_FIELD_DEF(uint64_t, alignN);
    TILING_DATA_FIELD_DEF(uint64_t, tileC);
    TILING_DATA_FIELD_DEF(uint64_t, tailC);
    TILING_DATA_FIELD_DEF(uint64_t, loopC);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MhcPostBackward, MhcPostBackwardTilingData)

struct MhcPostBackwardCompileInfo {
};

} // namespace optiling

#endif // MHC_POST_BACKWARD_TILING_H
