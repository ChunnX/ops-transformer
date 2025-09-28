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
 * \file grouped_mat_mul_allto_allv_tiling_A3.h
 * \brief
 */
#ifndef MC2_GROUPED_MATMUL_ALLTO_ALLV_TILING_A3_H
#define MC2_GROUPED_MATMUL_ALLTO_ALLV_TILING_A3_H

#include "grouped_mat_mul_allto_allv_tiling_base.h"

namespace optiling {
class GmmAlltoAllvTilingA3 : public GmmAlltoAllvTilingBase
{
public:
    explicit GmmAlltoAllvTilingA3(gert::TilingContext* context) : GmmAlltoAllvTilingBase(context){};

protected:
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    bool IsCapable() override;
};
} // namespace optiling
#endif