# aclnn 骨架：统一 FP8 PV 入口（与 PyTorch sageattn_qk_int8_pv_fp8_asc 语义对齐）
set(ACLNN_PV_SOURCES
    op_host/sageattn_pv.cpp
    op_kernel/sageattn_pv_kernel.cpp
)
catlass_example_add_executable(aclnn_sageattn_pv host ${ACLNN_PV_SOURCES})
target_include_directories(aclnn_sageattn_pv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})