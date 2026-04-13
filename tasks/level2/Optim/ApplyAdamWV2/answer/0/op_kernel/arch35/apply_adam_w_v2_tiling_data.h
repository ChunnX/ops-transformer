/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APPLY_ADAM_W_V2_TILING_DATA_H
#define APPLY_ADAM_W_V2_TILING_DATA_H

/*!
 * \file apply_adam_w_v2_tiling_data.h
 * \brief
 */
#include <cstdint>
#include "atvoss/elewise/elewise_base_struct.h"
using namespace Ops::Base;
struct ApplyAdamWV2RegbaseTilingData {
    EleBaseTilingDataV2 elewiseTiling;
    float lr;
    float beta1;
    float beta2;
    float weightDecay;
    float eps;
    float gt;
};

#endif // APPLY_ADAM_W_V2_TILING_DATA_H
