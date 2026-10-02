# 模块说明：sageattention2 op_host 构建脚本：def 源注册 + tiling 源收集 + op_kernel 契约头包含路径。

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        sa2_def.cpp
    )
endif()

# 探测并解析 CATLASS 根目录与头文件包含路径
if (NOT DEFINED CATLASS_ROOT OR "${CATLASS_ROOT}" STREQUAL "")
    if (DEFINED ENV{CATLASS_ROOT} AND NOT "$ENV{CATLASS_ROOT}" STREQUAL "")
        set(CATLASS_ROOT $ENV{CATLASS_ROOT})
    elseif (DEFINED CATLASS AND EXISTS "${CATLASS}/include/catlass/catlass.hpp")
        set(CATLASS_ROOT "${CATLASS}")
    elseif (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/catlass/include/catlass/catlass.hpp")
        set(CATLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/catlass")
    elseif (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../catlass/include/catlass/catlass.hpp")
        set(CATLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../catlass")
    elseif (EXISTS "${ASCEND_DIR}/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/common/catlass")
        set(CATLASS_ROOT "${ASCEND_DIR}/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/common/catlass")
    else()
        set(CATLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/catlass")
    endif()
endif()

add_ops_compile_options(
    OP_NAME SageAttention2
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -I${CATLASS_ROOT}/include
            -I${CATLASS_ROOT}/include/tla
            -DCATLASS_ARCH=3510
            -DCATLASS_ARCH_A5_ENABLED
)

add_modules_sources(
    OP_API_INDEPENDENT ON
    OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    OPTYPE sage_attention2
    ACLNNTYPE aclnn
)

# sa2_tiling.cpp 引用 op_kernel 契约头（sa2_spec.h / sa2_template_key.h）与 CATLASS 头
if (TARGET ${OPHOST_NAME}_tiling_obj)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
        ${CATLASS_ROOT}/include
    )
    target_compile_definitions(${OPHOST_NAME}_tiling_obj PRIVATE
        CATLASS_ARCH=3510
        CATLASS_ARCH_A5_ENABLED
    )
endif()

if (BUILD_OPEN_PROJECT AND TARGET op_host_aclnn)
    target_include_directories(op_host_aclnn PRIVATE
        ${CATLASS_ROOT}/include
    )
    target_compile_definitions(op_host_aclnn PRIVATE
        CATLASS_ARCH=3510
        CATLASS_ARCH_A5_ENABLED
    )
endif()

