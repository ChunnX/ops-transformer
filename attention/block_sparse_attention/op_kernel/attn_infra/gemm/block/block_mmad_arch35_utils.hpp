/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_MMAD_ARCH35_UTILS_HPP
#define BLOCK_MMAD_ARCH35_UTILS_HPP

namespace NpuArch::Gemm::Block {

namespace MXFP4 {
    static constexpr uint32_t KB_SIZE = 1024;

    // L1 分配
    // pL1Tensor: 16K, 128*256*sizeof(e2m1), BUFCNT=20
    static constexpr uint32_t L1_P_BUF_CNT = 20;
    static constexpr uint32_t L1_P_BUF_SIZE = 16 * KB_SIZE;
    static constexpr uint32_t L1_P_BUF_OFFSET = 0;

    // pDescaleL1Tensor: 1.25K, 128*2*(256/64+1)*sizeof(e8m0), BUFCNT=20
    static constexpr uint32_t L1_P_SCALE_BUF_CNT = 20;
    static constexpr uint32_t L1_P_SCALE_BUF_SIZE = 1280;
    static constexpr uint32_t L1_P_SCALE_BUF_OFFSET = 320 * KB_SIZE;

    // qL1Tensor: 8K, 128*D*sizeof(e2m1), BUFCNT=2
    static constexpr uint32_t L1_Q_BUF_CNT = 2;
    static constexpr uint32_t L1_Q_BUF_SIZE = 8 * KB_SIZE;
    static constexpr uint32_t L1_Q_BUF_OFFSET = 345 * KB_SIZE;

    // qDescaleL1Tensor: 0.5K, 128*D/32*sizeof(e8m0), BUFCNT=2
    static constexpr uint32_t L1_Q_DESCALE_BUF_CNT = 2;
    static constexpr uint32_t L1_Q_DESCALE_BUF_SIZE = 512;
    static constexpr uint32_t L1_Q_DESCALE_BUF_OFFSET = 361 * KB_SIZE;

    // kvL1Tensor: 32K, 512*D*sizeof(e2m1), BUFCNT=4
    static constexpr uint32_t L1_KV_BUF_CNT = 4;
    static constexpr uint32_t L1_KV_BUF_SIZE = 32 * KB_SIZE;
    static constexpr uint32_t L1_KV_BUF_OFFSET = 362 * KB_SIZE;

    // kvDescaleL1Tensor: 2K, 512*D/32*sizeof(e8m0), BUFCNT=4
    static constexpr uint32_t L1_KV_DESCALE_BUF_CNT = 4;
    static constexpr uint32_t L1_KV_DESCALE_BUF_SIZE = 2 * KB_SIZE;
    static constexpr uint32_t L1_KV_DESCALE_BUF_OFFSET = 490 * KB_SIZE;

    // localGlobalMaxL1: 0.5K, 256*sizeof(half), BUFCNT=4
    static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_CNT = 4;
    static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_SIZE = 512;
    static constexpr uint32_t L1_LOCAL_GLOBAL_MAX_BUF_OFFSET = 498 * KB_SIZE;
}  // namespace MXFP4

}  // namespace NpuArch::Gemm::Block

#endif  // BLOCK_MMAD_ARCH35_UTILS_HPP
