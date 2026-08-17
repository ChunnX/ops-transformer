/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_allto_all_matmul_v2.cpp
 * \brief kernel ut
 */

/*!
 * AlltoAllMatmulV2 是 MC2 通信 + Matmul 融合算子（apace URMA 通信路径），
 * 核函数需要多卡通信上下文（CommContext），无法在 op_kernel UT 单卡场景下独立运行，
 * 因此本模块不提供可执行的 kernel UT 用例，仅在 ST 中做端到端验证。
 */
