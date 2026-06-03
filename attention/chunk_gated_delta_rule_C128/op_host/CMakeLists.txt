# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

# The OpDef-generated opFile is the canonical CANN snake name
# `chunk_gated_delta_rule_c128`, while this experimental copy keeps the
# user-visible directory name `chunk_gated_delta_rule_C128`.  Binary target
# generation indexes directories by opFile, so provide the lowercase alias.
get_filename_component(CHUNK_GATED_DELTA_RULE_C128_OP_DIR "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
set(chunk_gated_delta_rule_c128_dir ${CHUNK_GATED_DELTA_RULE_C128_OP_DIR} CACHE INTERNAL
    "Binary compile dir alias for ChunkGatedDeltaRuleC128")
set(chunk_gated_delta_rule_c128_apt_dir ${CHUNK_GATED_DELTA_RULE_C128_OP_DIR} CACHE INTERNAL
    "APT binary compile dir alias for ChunkGatedDeltaRuleC128")

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        chunk_gated_delta_rule_C128_def.cpp
    )
endif()

add_modules_sources(OPTYPE chunk_gated_delta_rule_C128 ACLNNTYPE aclnn_exclude)
