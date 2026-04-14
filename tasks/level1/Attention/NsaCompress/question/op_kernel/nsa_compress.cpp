/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_compress.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "nsa_compress_seq_manager.h"
#include "nsa_compress_tiling.h"
#include "nsa_compress.h"


using namespace NASCompress;

namespace NASCompress {

extern "C" __global__ __aicore__ void nsa_compress(__gm__ uint8_t *input, __gm__ uint8_t *weight,
                                                   __gm__ uint8_t *actSeqLenOptional, __gm__ uint8_t *output,
                                                   __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
}

} // namespace NASCompress