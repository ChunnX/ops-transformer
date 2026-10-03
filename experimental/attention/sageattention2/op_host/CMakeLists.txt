include(${CMAKE_CURRENT_LIST_DIR}/../cmake/SageAttnDeps.cmake)

# Host 侧源码与共享 include 列表（本目录两个 target 的唯一定义点）
set(SAGE_ATTN_HOST_CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_def.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_infershape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/sage_attention2_tiling.cpp
)

set(SAGE_ATTN_QUANT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quant_int8.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quant_fp8.cpp
)

set(SAGE_ATTN_HOST_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../aclnn
    ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/ascendc
    ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/catlass
    ${CMAKE_CURRENT_SOURCE_DIR}/../quant
)

if (BUILD_OPEN_PROJECT)
    if(TARGET op_host_aclnnExc)
        target_sources(op_host_aclnnExc PRIVATE
            ${SAGE_ATTN_HOST_CORE_SOURCES}
            ${SAGE_ATTN_QUANT_SOURCES}
        )
        target_include_directories(op_host_aclnnExc PRIVATE
            ${SAGE_ATTN_HOST_INCLUDE_DIRS}
            # aclnn 错误码与轻量执行器头 (aclnn/opdev/op_errno.h) 位于 CANN include/aclnn 子树
            ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        )
        if(TARGET tiling_api)
            target_link_libraries(op_host_aclnnExc PRIVATE tiling_api)
        endif()
    endif()
endif()

if(COMMAND add_ops_compile_options)
    add_ops_compile_options(
        OP_NAME SageAttention2
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL AND COMMAND add_modules_sources)
    # op_api 与 op_host 同级布局
    add_modules_sources(OPTYPE sage_attention2 ACLNNTYPE aclnn_exclude
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
endif()

# 支持独立构建 Host 共享库以供本地单测与集成测试
if(NOT TARGET op_host_aclnn AND NOT TARGET sageattention2_host)
    add_library(sageattention2_host SHARED
        ${SAGE_ATTN_HOST_CORE_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../aclnn/sage_attn_host.cpp
        ${SAGE_ATTN_QUANT_SOURCES}
    )
    target_include_directories(sageattention2_host PUBLIC ${SAGE_ATTN_HOST_INCLUDE_DIRS})
    target_compile_features(sageattention2_host PUBLIC cxx_std_17)
endif()
