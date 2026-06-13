# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
add_ut_test(TEST_TARGET inplace_partial_rotary_mul_grad_kernel_ut
            CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/test_inplace_partial_rotary_mul_grad_kernel.cpp
            DEP_LIBS gtest
            INSTALL_DST ${CMAKE_INSTALL_PREFIX}/tests/ut/op_kernel
            DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/inplace_partial_rotary_mul_grad_data
            DATA_INSTALL_DST data)
