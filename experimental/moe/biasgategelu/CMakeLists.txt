# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "BUILD_TORCH_OPS ON in biasgategelu")
# BIAS_GATE_GELU operation sources
file(GLOB BIAS_GATE_GELU_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

set(BIAS_GATE_GELU_SOURCES ${BIAS_GATE_GELU_NPU_SOURCES})
# Mark .cpp files with special properties
set_source_files_properties(
    ${BIAS_GATE_GELU_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--cce-soc-version=Ascend910B1 --cce-soc-core-type=VecCore --cce-auto-sync -xcce"
)

# Create object library
add_library(bias_gate_gelu_objects OBJECT ${BIAS_GATE_GELU_SOURCES})

target_compile_options(bias_gate_gelu_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(bias_gate_gelu_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()
