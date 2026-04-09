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
* \file section_stream_k.cpp
* \brief
*/

#include "section_stream_k_impl.h"

namespace load_balance {

uint32_t SectionStreamK::Compute(const DeviceInfo &deviceInfo, const IBaseInfo &baseInfo,
    const SectionStreamKParam &param, SectionStreamKResult &result)
{
    static SectionStreamKImpl impl;
    impl.SetParam(param);
    impl.Compute(deviceInfo, baseInfo, result);
    return LOAD_BALANCE_SUCCESS;
}

}