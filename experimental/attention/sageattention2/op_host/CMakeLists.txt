# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
# Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

# The repository directory keeps the public task name "sageattention2", while the generated kernel op_file is
# "sage_attention2". Export the op_file-to-source mapping expected by add_bin_compile_target so normal builds do
# not silently skip all generated kernel compile scripts.
get_filename_component(SAGE_ATTENTION2_OP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
set(sage_attention2_dir ${SAGE_ATTENTION2_OP_DIR} CACHE INTERNAL "SageAttention2 kernel source directory" FORCE)

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE sage_attention2_def.cpp)
endif()

add_ops_compile_options(
    OP_NAME SageAttention2
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -DCATLASS_ARCH=3510
            -I${CMAKE_SOURCE_DIR}/3rdparty/catlass/include
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

add_modules_sources(
    OP_API_INDEPENDENT ON
    OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    OPTYPE sage_attention2
    ACLNNTYPE aclnn
)

add_tiling_modules()
target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    sage_attention2_tiling.cpp
)
