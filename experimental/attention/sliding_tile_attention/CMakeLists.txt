# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# opFile.value is sliding_tile_attention_fai while the operator directory is
# sliding_tile_attention. Register the source directory under both names so the
# generated binary compile script is not skipped by add_bin_compile_target.
set(sliding_tile_attention_fai_dir ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
set(sliding_tile_attention_fai_a2_dir ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")
set(sliding_tile_attention_fai_a3_dir ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "")

add_subdirectory(op_host)

if(ENABLE_TEST)
    add_subdirectory(tests)
endif()
