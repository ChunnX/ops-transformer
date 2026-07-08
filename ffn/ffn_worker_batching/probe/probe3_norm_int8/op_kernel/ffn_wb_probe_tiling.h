/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

/*!
 * \file ffn_wb_probe_tiling.h
 * \brief 直调穿刺用 TilingData 平铺 struct（POD），字段顺序与 op_host/ffn_worker_batching_tiling.h
 *        的 FfnWorkerBatchingTilingData 完全一致，供 kernel 侧头文件 (ffn_wb_get_schedule_context.h)
 *        以 tilingData->X 方式读取。定义在全局命名空间，供 namespace FfnWbBatching 内非限定查找命中。
 */

#ifndef OP_KERNEL_FFN_WB_PROBE_TILING_H
#define OP_KERNEL_FFN_WB_PROBE_TILING_H

#include <cstdint>

struct FfnWorkerBatchingTilingData {
    int64_t Y;
    int64_t H;
    int64_t tokenDtype;
    int64_t expertNum;
    int64_t coreNum;
    int64_t ubSize;
    int64_t sortLoopMaxElement;
    int64_t sortNumWorkSpace;
};

#endif  // OP_KERNEL_FFN_WB_PROBE_TILING_H
