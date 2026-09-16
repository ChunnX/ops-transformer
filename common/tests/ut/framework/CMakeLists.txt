# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_HOST_UT)
    target_include_directories(${OP_ONNX_MODULE_NAME}_cases_obj PRIVATE
        ${PROJECT_SOURCE_DIR}/common/include/framework
        ${PROJECT_SOURCE_DIR}/common/include/op_graph
        ${PROJECT_SOURCE_DIR}/common/stub/inc/framework
        ${JSON_INCLUDE_DIR})
    target_sources(${OP_ONNX_MODULE_NAME}_cases_obj PRIVATE
        test_npu_fused_attention_score_onnx_plugin.cpp
        test_npu_masked_softmax_with_relposbias_onnx_plugin.cpp
        test_npu_scaled_masked_softmax_onnx_plugin.cpp
        test_tfidf_vectorizer_onnx_plugin.cpp)

endif()
