# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_kda_fwd_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_kda_fwd ACLNNTYPE aclnn_exclude)

set(CHUNK_KDA_CATLASS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/3rdparty/catlass/include")
if(NOT EXISTS "${CHUNK_KDA_CATLASS_INCLUDE_DIR}/tla/tensor.hpp")
    message(FATAL_ERROR
        "ChunkKdaFwd requires the Catlass submodule. Run: "
        "git submodule update --init --recursive 3rdparty/catlass")
endif()

add_ops_compile_options(
    OP_NAME ChunkKdaFwd
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -I${CHUNK_KDA_CATLASS_INCLUDE_DIR}
)
if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME ChunkKdaFwd
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()
