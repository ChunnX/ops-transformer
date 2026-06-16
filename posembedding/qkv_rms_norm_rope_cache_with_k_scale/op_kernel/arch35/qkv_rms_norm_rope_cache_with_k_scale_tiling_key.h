/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_KEY_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_KEY_H_

#include "ascendc/host_api/tiling/template_argument.h"

namespace optiling {
namespace QkvRmsNormRopeCacheWithKScale {

#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TPL_HEAD_DIM_D128 128

ASCENDC_TPL_ARGS_DECL(
    QkvRmsNormRopeCacheWithKScale,
    ASCENDC_TPL_UINT_DECL(HEAD_DIM, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_LIST,
                          QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TPL_HEAD_DIM_D128));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(HEAD_DIM, ASCENDC_TPL_UI_LIST, QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TPL_HEAD_DIM_D128)));

} // namespace QkvRmsNormRopeCacheWithKScale
} // namespace optiling

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_TILING_KEY_H_
