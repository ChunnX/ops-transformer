# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "BUILD_TORCH_OPS ON in incre_flash_attention")

# incre_flash_attention operation sources
file(GLOB_RECURSE incre_flash_attention_NPU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
set(incre_flash_attention_SOURCES ${incre_flash_attention_NPU_SOURCES})

# Mark .cpp files with special properties
set_source_files_properties(
    ${incre_flash_attention_NPU_SOURCES} PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "--npu-arch=dav-2201 -xasc "
)

# Create object library
add_library(incre_flash_attention_objects OBJECT ${incre_flash_attention_SOURCES})

target_compile_options(incre_flash_attention_objects PRIVATE ${COMMON_COMPILE_OPTIONS})
target_include_directories(incre_flash_attention_objects PRIVATE ${COMMON_INCLUDE_DIRS})
return()