# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        allto_allv_grouped_mat_mul_def.cpp
        allto_allv_grouped_mat_mul_shmem_def.cpp
    )
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OP_MC2_ENABLE ON
        OPTYPE allto_allv_grouped_mat_mul allto_allv_grouped_mat_mul allto_allv_grouped_mat_mul_shmem
        ACLNNTYPE aclnn aclnn_inner aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(allto_allv_grouped_mat_mul_depends mc2/common mc2/3rd mc2/allto_allv_quant_grouped_mat_mul PARENT_SCOPE)
    # The Ascend950 registration uses a separate op_file name while sharing this operator directory.
    # Export both aliases so add_bin_compile_target does not silently skip the SHMEM kernel and its dependencies.
    set(allto_allv_grouped_mat_mul_shmem_dir ${CMAKE_CURRENT_SOURCE_DIR}/.. PARENT_SCOPE)
    set(allto_allv_grouped_mat_mul_shmem_depends
        mc2/common mc2/3rd mc2/allto_allv_quant_grouped_mat_mul PARENT_SCOPE)

    # --cce-auto-sync=off：指定CCE编译器是否自动执行线程间或模块间的同步操作
    # -mllvm -cce-aicore-dcci-before-kernel-end=false：禁用在kernel函数结束前插入DCCI指令
    # -DAICORE_EXCEPTION_RESTART：启用AI Core在发生异常时的自动重启机制
    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    if("${CONDITION_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME AlltoAllvGroupedMatMulShmem
            OPTIONS --cce-auto-sync=off
                    -mllvm -cce-aicore-dcci-before-kernel-end=false
        )
        if(DEFINED SHMEM_HOME_PATH AND NOT "${SHMEM_HOME_PATH}" STREQUAL "")
            set(A2AV_GMM_SHMEM_ROOT "${SHMEM_HOME_PATH}")
        elseif(DEFINED ENV{SHMEM_HOME_PATH} AND NOT "$ENV{SHMEM_HOME_PATH}" STREQUAL "")
            set(A2AV_GMM_SHMEM_ROOT "$ENV{SHMEM_HOME_PATH}")
        elseif(DEFINED SHMEM_HOME AND NOT "${SHMEM_HOME}" STREQUAL "")
            set(A2AV_GMM_SHMEM_HCCL_ROOT "${SHMEM_HOME}")
        elseif(DEFINED ENV{SHMEM_HOME} AND NOT "$ENV{SHMEM_HOME}" STREQUAL "")
            set(A2AV_GMM_SHMEM_HCCL_ROOT "$ENV{SHMEM_HOME}")
        else()
            message(FATAL_ERROR
                "SHMEM_HOME_PATH (source SDK layout) or SHMEM_HOME (HCCL-integrated layout) is required "
                "to compile AlltoAllvGroupedMatMulShmem for ascend950.")
        endif()
        if(DEFINED A2AV_GMM_SHMEM_ROOT)
            add_ops_compile_options(
                OP_NAME AlltoAllvGroupedMatMulShmem
                OPTIONS -I${A2AV_GMM_SHMEM_ROOT}/shmem/include
                        -I${A2AV_GMM_SHMEM_ROOT}/shmem/src
                        -I${A2AV_GMM_SHMEM_ROOT}/shmem/src/device
                        -I${A2AV_GMM_SHMEM_ROOT}/memfabric_hybrid/include/smem/host
                        -I${A2AV_GMM_SHMEM_ROOT}/memfabric_hybrid/include/smem/device
            )
        else()
            add_ops_compile_options(
                OP_NAME AlltoAllvGroupedMatMulShmem
                OPTIONS -I${A2AV_GMM_SHMEM_HCCL_ROOT}/include
                        -I${A2AV_GMM_SHMEM_HCCL_ROOT}/src/device
            )
        endif()
    elseif("${CONDITION_UNIT}" STREQUAL "ascend910_93")
        add_ops_compile_options(
            OP_NAME AlltoAllvGroupedMatMul
            OPTIONS --cce-auto-sync=off
                    -DAICORE_EXCEPTION_RESTART
        )
    endif()
else() # 回黄host
    add_mc2_modules_sources(
        OPTYPE allto_allv_grouped_mat_mul allto_allv_grouped_mat_mul allto_allv_grouped_mat_mul_shmem
        ACLNNTYPE aclnn aclnn_inner aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
endif()
