# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Keep the upstream QGMM extensions private to MegaMoe. Explicitly qualified includes
# prevent these generated headers from shadowing the shared Blaze dependency.
set(MEGA_MOE_QGMM_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_mega_moe_qgmm_root "${MEGA_MOE_QGMM_INCLUDE_DIR}/mega_moe_qgmm")
set(_mega_moe_qgmm_patch "${CMAKE_CURRENT_LIST_DIR}/qgmm_concat.patch.in")
set(_mega_moe_qgmm_headers
    blaze/gemm/block/block_mmad_qgmm_mx.h
    blaze/gemm/tile/arch35/copy_gm_to_l1.h)
foreach(_mega_moe_header IN LISTS _mega_moe_qgmm_headers)
    configure_file("${OPTENSOR_SOURCE_PATH}/include/${_mega_moe_header}"
                   "${_mega_moe_qgmm_root}/${_mega_moe_header}" COPYONLY)
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_mega_moe_qgmm_patch}")
find_program(MEGA_MOE_PATCH_EXECUTABLE patch REQUIRED)
execute_process(
    COMMAND "${MEGA_MOE_PATCH_EXECUTABLE}" --batch --forward -p1
        -d "${_mega_moe_qgmm_root}" -i "${_mega_moe_qgmm_patch}"
    RESULT_VARIABLE _mega_moe_patch_result
    OUTPUT_VARIABLE _mega_moe_patch_output
    ERROR_VARIABLE _mega_moe_patch_error)
if(NOT _mega_moe_patch_result EQUAL 0)
    message(FATAL_ERROR "MegaMoe QGMM extension failed for the selected ops-tensor dependency: ${_mega_moe_patch_output}${_mega_moe_patch_error}")
endif()

set(_mega_moe_qgmm_block "${_mega_moe_qgmm_root}/blaze/gemm/block/block_mmad_qgmm_mx.h")
file(READ "${_mega_moe_qgmm_block}" _mega_moe_qgmm_source)
string(REPLACE "\"blaze/gemm/tile/copy_gm_to_l1.h\""
               "\"mega_moe_qgmm/blaze/gemm/tile/arch35/copy_gm_to_l1.h\""
               _mega_moe_qgmm_source "${_mega_moe_qgmm_source}")
file(WRITE "${_mega_moe_qgmm_block}" "${_mega_moe_qgmm_source}")
add_ops_compile_options(OP_NAME MegaMoe OPTIONS "-I${MEGA_MOE_QGMM_INCLUDE_DIR}")
