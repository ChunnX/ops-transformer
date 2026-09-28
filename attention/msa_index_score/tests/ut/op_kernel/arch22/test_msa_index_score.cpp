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
 * \file test_msa_index_score.cpp
 * \brief Atlas A2/A3（arch22）kernel UT。用例与 arch35 共用。
 *
 * ut.cmake 在 ARCH_DIRECTORY 含 arch35 时跳过非 arch35 目录，因此 A2/A3 与 950
 * 必须分目录注册，不能像 op_host 那样放在同一层。
 */

#include "../msa_index_score_kernel_ut.inc"
