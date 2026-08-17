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
 * \file moe_tutel_combine_x_tiling_key.h
 * \brief TilingKey 常量。dtype 不进 key，由 DTYPE_Y_GRAD 模板参数区分。
 *        与 op_host/moe_tutel_combine_x_tiling.cpp 的 TILING_KEY_MAIN 保持一致。
 */

#ifndef MOE_TUTEL_COMBINE_X_TILING_KEY_H_
#define MOE_TUTEL_COMBINE_X_TILING_KEY_H_

// 唯一主模板：token 独占输出 + K/H 分块 gather 累加。
#define MOE_TUTEL_COMBINE_X_TILING_KEY_MAIN 10000

#endif // MOE_TUTEL_COMBINE_X_TILING_KEY_H_
