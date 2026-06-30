/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  dispatch_teardown_tiling.h —— DispatchTeardown(mxfp8 多核版) 的 TilingData 定义。
 *  字段顺序即 device 字节布局（kernel POD cast 必须同序）。mxfp8：token = fp8 payload(h) + e8m0 scale(h/32)，
 *  两者各为独立 GM tensor、随 token 一起重排。多核：按 expandX 输出行切分（kernel 内按实收行分配）。
 */
#ifndef DISPATCH_TEARDOWN_TILING_H
#define DISPATCH_TEARDOWN_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DispatchTeardownTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, bs);        // 本卡 token 数
    TILING_DATA_FIELD_DEF(uint32_t, topk);      // 每 token 选专家数 k
    TILING_DATA_FIELD_DEF(uint32_t, h);         // hidden size（fp8 payload 元素数/字节数）
    TILING_DATA_FIELD_DEF(uint32_t, scaleH);    // 每 token scale 数 = h/32（e8m0，1 字节/个）
    TILING_DATA_FIELD_DEF(uint32_t, E);         // expert_per_rank
    TILING_DATA_FIELD_DEF(uint32_t, R);         // rank_num
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);    // = SetBlockDim（参与重排的 AIV 核数）
    TILING_DATA_FIELD_DEF(uint32_t, A);         // expandX 行上限 = R*bs*min(topk,E)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DispatchTeardown, DispatchTeardownTilingData)
}  // namespace optiling

#endif  // DISPATCH_TEARDOWN_TILING_H
