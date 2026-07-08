/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/*!
 * \file ffn_wb_probe_tiling.h
 * \brief 模板穿刺(穿刺1 RECV fp16)用的 TilingData POD 结构。
 *        字段顺序/类型与 op_host/ffn_worker_batching_tiling.h 的
 *        FfnWorkerBatchingTilingData(BEGIN_TILING_DATA_DEF) 完全一致，
 *        以便 Kernel 直调时逐字段从 GM 读取。
 *        放在全局命名空间：kernel 头(namespace FfnWbBatching)内以非限定名引用。
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
