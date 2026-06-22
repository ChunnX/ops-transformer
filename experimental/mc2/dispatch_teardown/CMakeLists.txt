# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# DispatchTeardown：纯 AIV、aclnn 注册式自定义算子。结构对齐 all_gather_matmul_v2：
#   仅 GLOB 收录带 CMakeLists 的子目录（op_host）；op_kernel / op_api 由 op_host 的
#   add_modules_sources 按 OPTYPE 约定带编，无需各自 CMakeLists。
# 经 experimental/mc2 顶层 GLOB 自动 add_subdirectory；真正编译发生在
#   ops-transformer 注册构建（BUILD_OPEN_PROJECT）：build.sh --pkg --soc=ascend950 --ops=dispatch_teardown。

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

if (BUILD_OPEN_PROJECT)
    get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
endif()
