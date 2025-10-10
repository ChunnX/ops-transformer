# ----------------------------------------------------------------------------
 # Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 # This file is a part of the CANN Open Software.
 # Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 # Please refer to the License for details. You may not use this file except in compliance with the License.
 # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 # See LICENSE in the root of the software repository for the full text of the License.
 # ----------------------------------------------------------------------------

if(BUILD_OPEN_PROJECT)
    file(GLOB MC2_3RD_SRCS 
        # mm
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/mat_mul_v3_def.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/op_tiling/matmul_v3_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/op_tiling/matmul_v3_l2_cache.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp
        # bmm
        ${CMAKE_CURRENT_SOURCE_DIR}/batch_mat_mul_v3/op_host/batch_mat_mul_v3_def.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/batch_mat_mul_v3/op_host/op_tiling/batch_mat_mul_v3_base_tiling.cpp
        # qbmm
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling_base.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_batch_matmul_v3/op_host/op_tiling/adaptive_sliding_window_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_checker.cpp
        # wqbmm
        ${CMAKE_CURRENT_SOURCE_DIR}/weight_quant_batch_matmul_v2/op_host/op_tiling/*.cpp
        # mm common
        ${CMAKE_CURRENT_SOURCE_DIR}/common/op_host/op_tiling/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ops_legacy/op_tiling/op_cache_tiling.cpp
     )

     if(MC2_3RD_SRCS)
         add_tiling_modules()
         target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${MC2_3RD_SRCS})
     endif()

     #打桩文件
     file(GLOB MC2_DZ 
         ${CMAKE_CURRENT_SOURCE_DIR}/matmul_reduce_scatter_v2/op_host/op_api/aclnn_matmul_reduce_scatter_v2.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/all_gather_matmul_v2/op_host/op_api/aclnn_all_gather_matmul_v2.cpp
     )
     if(MC2_DZ)
         add_opapi_modules()
         target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${MC2_DZ})
     endif()
endif()

add_subdirectory(common/op_host)