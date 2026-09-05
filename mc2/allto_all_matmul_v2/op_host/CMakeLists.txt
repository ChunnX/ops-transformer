# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    # Core OpDef registration
    target_sources(op_host_aclnnInner PRIVATE
        allto_all_matmul_v2_def.cpp
    )

    # Main module: discovers src in op_host/, op_tiling/; also discovers proto/gen_task if under op_graph/
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OP_MC2_ENABLE ON
        OPTYPE allto_all_matmul_v2 allto_all_matmul_v2 ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(allto_all_matmul_v2_depends mc2/common mc2/3rd PARENT_SCOPE)
    set(allto_all_matmul_v2_apt_depends mc2/common mc2/3rd PARENT_SCOPE)
    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/ops-tensor/include/tensor_api/impl/c_api
                         ${CMAKE_SOURCE_DIR}/third_party/ops-tensor/include/tensor_api/include/c_api)

    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    if("${CONDITION_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME AlltoAllMatmulV2
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS --cce-auto-sync=off
                    -mllvm
                    -cce-aicore-dcci-before-kernel-end=false
                    -I${CMAKE_SOURCE_DIR}/mc2/common/op_kernel
                    -I${CMAKE_SOURCE_DIR}/mc2/common/op_kernel/apace/tests/st/.third_party/ops-tensor/include
        )
        if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/asc/include/adv_api/hcomm/dfx/hcomm_channel_dump.h)
            # shim 实现头会导出 hcomm_dfx_queue_read_kernel 等强符号，多个算子各自链接 shim .o
            # 会在 cust_opmaster 内重复定义；用全局属性保证全工程只编译链接一份（首个处理的算子接线）
            get_property(MC2_HCOMM_DFX_WIRED GLOBAL PROPERTY MC2_HCOMM_DFX_SHIM_WIRED SET)
            if(NOT MC2_HCOMM_DFX_WIRED)
                set_property(GLOBAL PROPERTY MC2_HCOMM_DFX_SHIM_WIRED TRUE)
                set(HCOMM_DFX_SHIM_OBJ ${CMAKE_CURRENT_BINARY_DIR}/hcomm_dfx_shim_dav_c310.o)
                add_custom_command(
                    OUTPUT ${HCOMM_DFX_SHIM_OBJ}
                    COMMAND ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/ccec_compiler/bin/bisheng
                            --asc-aicore-lang --cce-aicore-arch=dav-c310 -std=c++17 -fPIC
                            -I${ASCEND_CANN_PACKAGE_PATH}/asc/include
                            -I${ASCEND_CANN_PACKAGE_PATH}/asc/include/adv_api
                            -I${ASCEND_CANN_PACKAGE_PATH}/include
                            -I${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include
                            -c ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/hcomm_dfx_shim.asc
                            -o ${HCOMM_DFX_SHIM_OBJ}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/hcomm_dfx_shim.asc
                    COMMENT "Compiling hcomm dfx shim with bisheng (dav-c310)"
                )
                add_custom_target(allto_all_matmul_v2_hcomm_dfx_shim DEPENDS ${HCOMM_DFX_SHIM_OBJ})
                add_dependencies(cust_opmaster allto_all_matmul_v2_hcomm_dfx_shim)
                target_link_libraries(cust_opmaster PRIVATE
                    ${HCOMM_DFX_SHIM_OBJ}
                    ${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/lib64/libascendc_runtime.a
                )
                target_compile_definitions(${OPHOST_NAME}_tiling_obj PRIVATE MC2_HCOMM_DFX_SHIM=1)
            endif()
        endif()
    endif()

    # Include apace headers for tiling code
    # Use op_kernel (parent of apace/) so that "apace/..." includes resolve correctly.
    # Also add mc2/ as a root so that "common/utils/mc2_log.h" resolves.
    target_include_directories(op_host_aclnnInner PRIVATE
        ${CMAKE_SOURCE_DIR}/mc2/common/op_kernel
        ${CMAKE_SOURCE_DIR}/mc2
        ${ASCEND_CANN_PACKAGE_PATH}/asc/include
    )
    target_include_directories(ophost_transformer_tiling_obj PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/asc/include
        ${ASCEND_CANN_PACKAGE_PATH}/asc/include/basic_api
        ${ASCEND_CANN_PACKAGE_PATH}/asc
    )
else()
    add_mc2_modules_sources(OPTYPE allto_all_matmul_v2 allto_all_matmul_v2 ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
endif()
