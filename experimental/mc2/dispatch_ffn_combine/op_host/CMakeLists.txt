# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_op_to_compiled_list()

set(_DISPATCH_FFN_INC_OPTS)
if (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/arm64-linux/ascendc/include)
elseif (EXISTS ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
    list(APPEND _DISPATCH_FFN_INC_OPTS -I${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include)
endif()
if (NOT EXISTS ${CMAKE_SOURCE_DIR}/experimental/third_party/catlass/include)
    execute_process(
        COMMAND git submodule update --init experimental/third_party/catlass
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_SUBMODULE_RESULT
    )
endif()
list(APPEND _DISPATCH_FFN_INC_OPTS -I${CMAKE_SOURCE_DIR}/experimental/third_party/catlass/include)
list(APPEND _DISPATCH_FFN_INC_OPTS -I${CMAKE_CURRENT_SOURCE_DIR})


# Copy moe_distribute_base.h to op_kernel/utils and rename HcclOpResParam to HcclOpResParamCustom
set(_MOE_DISTRIBUTE_BASE_SRC "${CMAKE_SOURCE_DIR}/mc2/common/inc/kernel/moe_distribute_base.h")
set(_MOE_DISTRIBUTE_BASE_DST "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/utils/moe_distribute_base.h")
if(EXISTS ${_MOE_DISTRIBUTE_BASE_SRC})
    file(READ ${_MOE_DISTRIBUTE_BASE_SRC} _MOE_DISTRIBUTE_BASE_CONTENT)
    string(REPLACE "struct HcclOpResParam" "struct HcclOpResParamCustom" _MOE_DISTRIBUTE_BASE_CONTENT "${_MOE_DISTRIBUTE_BASE_CONTENT}")
    string(REPLACE "struct HcclRankRelationResV2" "struct HcclRankRelationResV2Custom" _MOE_DISTRIBUTE_BASE_CONTENT "${_MOE_DISTRIBUTE_BASE_CONTENT}")
    file(WRITE ${_MOE_DISTRIBUTE_BASE_DST} "${_MOE_DISTRIBUTE_BASE_CONTENT}")
    message(STATUS "Copied and modified moe_distribute_base.h to ${_MOE_DISTRIBUTE_BASE_DST}")
endif()


if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        dispatch_ffn_combine_def.cpp
    )
endif()

add_ops_compile_options(
        OP_NAME DispatchFFNCombine
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
                -DHCCL_COMM
                ${_DISPATCH_FFN_INC_OPTS}
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE dispatch_ffn_combine ACLNNTYPE aclnn_inner)
endif()

# target_sources(op_host_aclnnInner PRIVATE
#         dispatch_ffn_combine_def.cpp
# )

# target_sources(opapi PRIVATE
#         aclnn_dispatch_ffn_combine.cpp
# )

# if (NOT BUILD_OPEN_PROJECT)
#     target_sources(aclnn_ops_train PRIVATE
#         aclnn_dispatch_ffn_combine.cpp
#     )

#     target_sources(aclnn_ops_infer PRIVATE
#         aclnn_dispatch_ffn_combine.cpp
#     )
# endif ()

# target_sources(optiling PRIVATE
#         dispatch_ffn_combine_tiling.cpp
# )

# target_include_directories(optiling PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}
#         ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
# )

# target_sources(opsproto PRIVATE
#         dispatch_ffn_combine_proto.cpp
# )

# file(GLOB _GMM_Aclnn_header "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_dispatch_ffn_combine.h")

# install(FILES ${_GMM_Aclnn_header}
#         DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
# )
