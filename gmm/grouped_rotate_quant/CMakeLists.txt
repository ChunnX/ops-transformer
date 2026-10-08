# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is licensed under the CANN Open Software License Agreement Version 2.0.
add_subdirectory(op_host)

if(UT_TEST_ALL OR OP_API_UT)
    add_subdirectory(tests/ut/op_api)
endif()
if(UT_TEST_ALL OR OP_HOST_UT)
    add_subdirectory(tests/ut/op_host)
endif()
