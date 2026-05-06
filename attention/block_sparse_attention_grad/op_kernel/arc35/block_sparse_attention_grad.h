/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "kernel_operator.h"
#include "common_header.h"
#include "addr_compute.h"
using namespace AscendC;

namespace BSA_ARC35 {

template <typename INPUT_TYPE, class TILING_CLASS, bool DETERMINISTIC_ENABLE>
struct BSA_TYPE {
    using input_type = INPUT_TYPE;
    using tiling_class = TILING_CLASS;
    static constexpr bool deterministic_enable = DETERMINISTIC_ENABLE;
};

template <typename BSA_TYPE>
class BlockSparseAttentionGradArch35 {
    using INPUT_TYPE = typename BSA_TYPE::input_type;
    using TILING_CLASS = typename BSA_TYPE::tiling_class;
    static constexpr bool DETERMINISTIC_ENABLE = BSA_TYPE::deterministic_enable;

private:
    RunTimeInfo runTimeInfo_;
    AddrComputeModule<BSA_TYPE> addr_;

public:
    __aicore__ inline BlockSparseAttentionGradArch35(){};

    __aicore__ inline void Process(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR softmaxLse,
                                   GM_ADDR blockSparseMask, GM_ADDR blockShape, GM_ADDR attentionMask,
                                   GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                                   GM_ADDR workspace, const TILING_CLASS *tilingData)
    {
        printf("BlockSparseAttentionGrad LAUNCHED!\n");

        addr_.Init(tilingData);

        if ASCEND_IS_AIC {
            CubeProcess(dout, q, k, v, out, softmaxLse, blockSparseMask, blockShape, attentionMask, actualQseqlen,
                        actualKvseqlen, dq, dk, dv, workspace, tilingData);
        }
        if ASCEND_IS_AIV {
            VectorProcess(dout, q, k, v, out, softmaxLse, blockSparseMask, blockShape, attentionMask, actualQseqlen,
                          actualKvseqlen, dq, dk, dv, workspace, tilingData);
        }
    }

    __aicore__ inline void CubeProcess(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR softmaxLse,
                                       GM_ADDR blockSparseMask, GM_ADDR blockShape, GM_ADDR attentionMask,
                                       GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR dq, GM_ADDR dk,
                                       GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS *tilingData)
    {
        printf("CubeProcess LAUNCHED!\n");

        uint32_t taskId = 0;
        while (true) {
            addr_.GetRunTimeInfo(&runTimeInfo_);
            break;
        }
    }

    __aicore__ inline void VectorProcess(GM_ADDR dout, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, GM_ADDR softmaxLse,
                                         GM_ADDR blockSparseMask, GM_ADDR blockShape, GM_ADDR attentionMask,
                                         GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR dq, GM_ADDR dk,
                                         GM_ADDR dv, GM_ADDR workspace, const TILING_CLASS *tilingData)
    {
        printf("VectorProcess LAUNCHED!\n");

        uint32_t taskId = 0;
        while (true) {
            addr_.GetRunTimeInfo(&runTimeInfo_);
            break;
        }
    }
};

} // namespace BSA_ARC35
