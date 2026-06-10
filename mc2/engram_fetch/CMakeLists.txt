# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    add_subdirectory(op_host)
    add_subdirectory(op_graph)
    if(NOT BUILD_OPS_API_INDEPENDENT)
        add_subdirectory(op_api)
    endif()
    if(ENABLE_TESTS_UT_OPS_TEST)
        add_subdirectory(tests/ut)
    endif()
else()
    add_mc2_modules_sources(OPTYPE engram_fetch ACLNNTYPE aclnn_inner)
endif()