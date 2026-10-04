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
    ${SAGE_ATTN_CATLASS_INCLUDE_DIR}
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
                -DENABLE_CV_COMM_VIA_SSBUF=true
                --cce-no-dcache-flush
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL AND COMMAND add_modules_sources)
    # op_api 与 op_host 同级布局
    add_modules_sources(OPTYPE sage_attention2 ACLNNTYPE aclnn_exclude
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
endif()
# 独立构建 Host 共享库 fallback 已随 2026-10-04 死代码清扫删除：宿主单测统一经
# run.sh host 直连 BiSheng 宿主模式编译（tests/_cxx.py 唯一定义点），
# 全仓无 cmake 独立构建入口（容器-only BiSheng 纪律，无宿主编译器代偿路径）。
