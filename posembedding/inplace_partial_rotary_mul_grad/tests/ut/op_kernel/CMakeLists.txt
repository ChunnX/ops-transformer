/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

add_ut_test(TEST_TARGET inplace_partial_rotary_mul_grad_kernel_ut
            CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_inplace_partial_rotary_mul_grad_kernel.cpp
            DEP_LIBS gtest
            INSTALL_DST ${CMAKE_INSTALL_PREFIX}/tests/ut/op_kernel
            DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inplace_partial_rotary_mul_grad_data
            DATA_INSTALL_DST data)
