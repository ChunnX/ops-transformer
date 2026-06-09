# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT AND ENABLE_TESTS_UT_OPS_TEST)
    add_subdirectory(op_api)
    add_subdirectory(op_kernel)
    add_subdirectory(op_host)
endif()