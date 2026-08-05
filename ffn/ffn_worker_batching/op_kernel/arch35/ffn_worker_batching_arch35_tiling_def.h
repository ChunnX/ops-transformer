/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_worker_batching_arch35_tiling_def.h
 * \brief arch35 (Ascend950 / DAV_3510) kernel 侧平铺 TilingData struct，host/kernel 共用 ABI。
 *        字段与 A2 FfnWorkerBatchingArch35TilingData 一一对应（对齐 moe_init_routing_v3_arch35_tiling_def.h）。
 *        host 经 context_->GetTilingData<FfnWorkerBatchingArch35TilingData>() 直写；
 *        kernel 经 GET_TILING_DATA_WITH_STRUCT(FfnWorkerBatchingArch35TilingData, ...) 读取。
 */
#ifndef FFN_WB_ARCH35_TILING_DEF_H
#define FFN_WB_ARCH35_TILING_DEF_H

struct FfnWorkerBatchingArch35TilingData {
    int64_t Y{0};          // A*BS*K
    int64_t H{0};          // hidden size（max_out_shape[3]）
    int64_t tokenDtype{0}; // 0:FP16 1:BF16 2:dynamic_quant_int8
    int64_t expertNum{0};
    int64_t coreNum{0};            // 运行时 GetCoreNumAiv 取值
    int64_t ubSize{0};             // 运行时 GetCoreMemSize(UB) 取值
    int64_t sortLoopMaxElement{0}; // 单次 ub 可排序 int32 数（由 ubSize 派生）
    int64_t sortNumWorkSpace{0};   // = Y
};

#endif // FFN_WB_ARCH35_TILING_DEF_H
