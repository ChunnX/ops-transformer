# ============================================================
# CMakeLists.txt — QuantMatmulReduceScatter ST 构建配置
# ============================================================

set(APACE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")          # mc2/common/op_kernel/apace
set(OP_KERNEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")    # mc2/common/op_kernel

# 使用 CANN built-in 的 apace + blaze + tensor_api (与 cann_skills 仓一致)
set(CANN_COMMON "${ASCEND_DIR}/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common")
set(CANN_APACE "${CANN_COMMON}/apace")

# 覆盖 apace_blaze_api include 路径
set_property(TARGET apace_blaze_api PROPERTY INTERFACE_INCLUDE_DIRECTORIES 
    "${CANN_COMMON}"
    "${CANN_COMMON}/tensor_api/include"
    "${CANN_COMMON}/tensor_api"
)

# kernel 模式 include 路径 (完全对齐 cann_skills 仓的 CMakeLists.txt)
set(KERNEL_INCLUDES
    ${CANN_COMMON}                                                          # apace/ blaze/ tensor_api/ 根
    ${CANN_APACE}                                                            # apace/ 直接路径
    ${CANN_APACE}/kernel/all_to_all_quant_matmul                             # all_to_all 的 quant_matmul_mx_kernel.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../kernel/quant_matmul_reduce_scatter  # 本算子 kernel 目录
    ${CMAKE_CURRENT_SOURCE_DIR}/src                                         # kernel_launcher.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../utils                                    # utils.h
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/ascendc/include/highlevel_api
    ${ASCEND_DIR}/compiler/tikcpp/tikcfw
    ${ASCEND_DIR}/include
)

set(KERNEL_OPTS "-xasc" "--npu-arch=dav-3510" "-DASC_DEVKIT_MAJOR=9" "-w" "-O3")
set(COMMON_LIBS dl platform tiling_api ascendcl runtime hccl stdc++)

add_executable(quant_matmul_reduce_scatter
    src/main.cpp
)

set_source_files_properties(src/main.cpp PROPERTIES
    COMPILE_OPTIONS "${KERNEL_OPTS}"
    INCLUDE_DIRECTORIES "${KERNEL_INCLUDES}"
)

target_link_directories(quant_matmul_reduce_scatter PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
    ${ASCEND_DIR}/lib64
)

target_link_libraries(quant_matmul_reduce_scatter PRIVATE ${COMMON_LIBS} apace_blaze_api)
target_link_libraries(quant_matmul_reduce_scatter PRIVATE "-Wl,--no-as-needed" hccl_fwk "-Wl,--as-needed")
