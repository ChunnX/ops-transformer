/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * SIMT 公共常量（对齐 CUDA CTA / SIMT best-practices）。
 */
#ifndef K2Q_CSR_SIMT_COMMON_ARCH35_H
#define K2Q_CSR_SIMT_COMMON_ARCH35_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace k2q_csr_simt {

/** 搬运/不规则访存：1024；LAUNCH_BOUND 与 Dim3 必须同常量 */
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 256;
#else
constexpr uint32_t THREAD_NUM = 1024;
#endif

} // namespace k2q_csr_simt

#endif
