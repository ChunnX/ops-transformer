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
 * \file causal_conv1d_grad_tiling_data.h
 * \brief TilingData 结构体定义（标准 C++ struct，Host/Kernel 共享）
 *
 * 切分策略：仅沿 D（通道）维切分，每个核独占一段通道，
 * dweight/dbias 在核内即可完整累加，无需跨核 reduce。
 */

#ifndef _CAUSAL_CONV1D_GRAD_TILING_DATA_H_
#define _CAUSAL_CONV1D_GRAD_TILING_DATA_H_

struct CausalConv1dGradTilingData {
    int64_t batch = 0;        // B
    int64_t seqLen = 0;       // T
    int64_t dim = 0;          // D（通道数，固定 12288）
    int64_t kW = 0;           // 卷积核宽度（固定 4）
    int64_t baseDim = 0;      // 单次处理的通道 tile 大小
    int64_t dimTileCnt = 0;   // 通道 tile 总数 = ceil(dim / baseDim)
    int64_t usedCores = 0;    // 实际使用核数 = min(coreNum, dimTileCnt)
};
#endif
