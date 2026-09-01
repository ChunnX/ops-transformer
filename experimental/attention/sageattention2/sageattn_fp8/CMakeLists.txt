# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SageAttention2 INT8 QK + FP8 PV kernel 构建文件（AscendC + CATLASS）。
# 用法（与 CATLASS 样例一致）：
#   bash scripts/build.sh sageattn_fp8 -DCATLASS_ARCH=3510
# 或在本目录下：
#   mkdir -p build && cd build && cmake .. -DCATLASS_ARCH=3510 && make -j32
# ----------------------------------------------------------------------------

set(FAI_SOURCES
    fai.cpp)

set_source_files_properties(${FAI_SOURCES} PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(sageattn_fp8 mix ${FAI_SOURCES})
target_include_directories(sageattn_fp8 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sageattn_fp8 PRIVATE m)

# ---- SageAttention2 独立量化 matmul kernel（多 kernel 串联方案） ----
# MM1: INT8 QK^T（int8 输入 + int32 累加 + per-group/per-block scale 反量化，绕开 Fixpipe 限制）
set(MM1_SOURCES
    mm1_qk_int8.cpp)

set_source_files_properties(${MM1_SOURCES} PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(mm1_qk_int8 mix ${MM1_SOURCES})
target_include_directories(mm1_qk_int8 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mm1_qk_int8 PRIVATE m)

# MM2: FP8 PV（MX FP8 E4M3 matmul）
set(MM2_SOURCES
    mm2_pv_fp8.cpp)

set_source_files_properties(${MM2_SOURCES} PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(mm2_pv_fp8 mix ${MM2_SOURCES})
target_include_directories(mm2_pv_fp8 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mm2_pv_fp8 PRIVATE m)

# Host 侧串联驱动：MM1(INT8 QK^T) -> softmax -> MM2(FP8 PV)
set(CHAIN_SOURCES
    sageattn_chain.cpp)

set_source_files_properties(${CHAIN_SOURCES} PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(sageattn_chain mix ${CHAIN_SOURCES})
target_include_directories(sageattn_chain PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sageattn_chain PRIVATE m)

# Pipeline：MM1(INT8 QK^T → S to GM) + FA(读 S→softmax→MM2→rescale)
set(PIPELINE_SOURCES
    fai_pipeline.cpp)

set_source_files_properties(${PIPELINE_SOURCES} PROPERTIES LANGUAGE ASC)
catlass_example_add_executable(fai_pipeline mix ${PIPELINE_SOURCES})
target_include_directories(fai_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fai_pipeline PRIVATE m)
