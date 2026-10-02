# 模块说明：sageattention2 op_host 构建脚本：def 源注册 + tiling 源收集 + op_kernel 契约头包含路径。

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        sa2_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME SageAttention2
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
)

add_modules_sources(
    OP_API_INDEPENDENT ON
    OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    OPTYPE sage_attention2
    ACLNNTYPE aclnn
)

# sa2_tiling.cpp 引用 op_kernel 契约头（sa2_spec.h / sa2_template_key.h）
if (TARGET ${OPHOST_NAME}_tiling_obj)
    target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
    )
endif()
