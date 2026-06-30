/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  combine_teardown_tiling.h —— CombineTeardown(mxfp8 多核) 的 TilingData（字段顺序 = device 字节布局）。
 *  mxfp8：combine 数据区 token = fp8(e4m3fn) payload(h) + e8m0 scale(h/32)；gather 每 token 的 k 个复制体后
 *  逐组(group=32)反量化(fp8×2^e8m0→fp32)，乘 expert_scales fp32 累加 → cast bf16 → x[bs,h]。
 *  多核：按 (token, h-chunk) 切分，各核写 x 不相交列段，无 SyncAll。
 */
#ifndef COMBINE_TEARDOWN_TILING_H
#define COMBINE_TEARDOWN_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CombineTeardownTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, bs);             // token 数
    TILING_DATA_FIELD_DEF(uint32_t, topk);           // 每 token 复制体数 k
    TILING_DATA_FIELD_DEF(uint32_t, h);              // hidden size
    TILING_DATA_FIELD_DEF(uint32_t, scaleH);         // h/32（每 token e8m0 scale 数）
    TILING_DATA_FIELD_DEF(uint32_t, globalExperts);  // = R*E，combine 数据区块数
    TILING_DATA_FIELD_DEF(uint32_t, aivNum);         // = SetBlockDim
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CombineTeardown, CombineTeardownTilingData)
}  // namespace optiling

#endif  // COMBINE_TEARDOWN_TILING_H
