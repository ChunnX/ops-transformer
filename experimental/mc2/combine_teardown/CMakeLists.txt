# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# CombineTeardown：纯 AIV、aclnn 注册式自定义算子。结构对齐 dispatch_teardown：
#   仅 GLOB 收录带 CMakeLists 的子目录（op_host）；op_kernel/op_api 由 op_host add_modules_sources 带编。
file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

if (BUILD_OPEN_PROJECT)
    get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
endif()
