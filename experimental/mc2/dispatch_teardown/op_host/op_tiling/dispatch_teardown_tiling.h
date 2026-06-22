/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 *
 *  dispatch_teardown_tiling.h —— DispatchTeardown 的 TilingData 定义。
 *  字段顺序即 device 侧读到的字节布局（kernel 侧 POD cast 必须同序，见 op_kernel）。
 */
#ifndef DISPATCH_TEARDOWN_TILING_H
#define DISPATCH_TEARDOWN_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DispatchTeardownTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalCnt);    // 状态区 int32 槽数（单 die = 1）
    TILING_DATA_FIELD_DEF(uint32_t, expectVal);   // 就绪值（= 1）
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DispatchTeardown, DispatchTeardownTilingData)
}  // namespace optiling

#endif  // DISPATCH_TEARDOWN_TILING_H
