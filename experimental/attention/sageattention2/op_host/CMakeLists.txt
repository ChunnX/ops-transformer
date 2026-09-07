# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE sage_attention2_stats_def.cpp sage_attention2_quant_def.cpp sage_attention2_core_def.cpp)
endif()
if(NOT DEFINED CATLASS_ROOT)
    set(CATLASS_ROOT "$ENV{CATLASS_ROOT}")
endif()
if(NOT CATLASS_ROOT)
    get_filename_component(CATLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../catlass" ABSOLUTE)
endif()
if(NOT EXISTS "${CATLASS_ROOT}/include/catlass/catlass.hpp")
    message(FATAL_ERROR "Set CATLASS_ROOT to a CATLASS checkout with Ascend950 support")
endif()
foreach(stage Stats Quant Core)
    add_ops_compile_options(OP_NAME SageAttention2${stage}
        OPTIONS -DCATLASS_ARCH=3510 -I${CATLASS_ROOT}/include --cce-auto-sync=on -O2 -DNDEBUG)
endforeach()
add_modules_sources(OP_API_INDEPENDENT ON OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    OPTYPE sage_attention2_stats sage_attention2_quant sage_attention2_core
    ACLNNTYPE aclnn_exclude aclnn_exclude aclnn_exclude)

foreach(stage stats quant core)
    set(sage_attention2_${stage}_depends "attention/sageattention2" CACHE INTERNAL "Sage shared kernels")
endforeach()
