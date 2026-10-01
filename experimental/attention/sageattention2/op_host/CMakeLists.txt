# ==============================================================================
# SageAttention2 op_host 构建：
#   1. 解析 CATLASS checkout（环境变量覆盖 -> 仓内相对路径 -> 存在性守卫）；
#   2. 汇总算子编译选项为单一列表变量；
#   3. 注册算子并挂接 op_api 源码目录。
# ==============================================================================

# 由 CATLASS checkout 位置解析出供编译选项使用的 include 根；解析失败立即终止，
# 避免把缺失 Catlass 头文件的编译选项继续传递给下游算子构建宏。
function(sage_resolve_catlass_root out_root)
    if(NOT DEFINED CATLASS_ROOT)
        set(CATLASS_ROOT "$ENV{CATLASS_ROOT}")
    endif()
    if(NOT CATLASS_ROOT)
        get_filename_component(_sage_catlass_repo_default
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../catlass" ABSOLUTE)
        set(CATLASS_ROOT "${_sage_catlass_repo_default}")
    endif()
    if(NOT EXISTS "${CATLASS_ROOT}/include/catlass/catlass.hpp")
        message(FATAL_ERROR
            "SageAttention2: CATLASS_ROOT 必须指向具备 Ascend950 支持的 CATLASS checkout（当前值: ${CATLASS_ROOT}）")
    endif()
    set(${out_root} "${CATLASS_ROOT}" PARENT_SCOPE)
endfunction()

add_op_to_compiled_list()
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE sage_attention2_def.cpp)
endif()

sage_resolve_catlass_root(SAGE_CATLASS_ROOT)
set(SAGE_HOST_COMPILE_OPTIONS
    -DCATLASS_ARCH=3510
    -I${SAGE_CATLASS_ROOT}/include
    --cce-auto-sync=on
    -O2
    -DNDEBUG
    -mllvm --cce-vf-affine2vag=false)

add_ops_compile_options(OP_NAME SageAttention2
    OPTIONS ${SAGE_HOST_COMPILE_OPTIONS})
add_modules_sources(OP_API_INDEPENDENT ON OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    OPTYPE sage_attention2 ACLNNTYPE aclnn_exclude)
