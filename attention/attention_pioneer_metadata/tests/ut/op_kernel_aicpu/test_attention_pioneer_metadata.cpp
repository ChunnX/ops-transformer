/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#include "attention_pioneer_metadata_aicpu.h"
#include "../op_host/attention_pioneer_metadata.h"
#undef private
#undef protected
#include <string>
#include <cstring>
#include <iostream>
#include <vector>
using namespace std;
using namespace aicpu;
using namespace optiling;
class TEST_ATTENTION_PIONEER_METADATA_UT : public testing::Test {};

static constexpr int64_t METADATA_ELEM_COUNT = 1024;

// Helper: build cumulative Q seq lengths for TND layout
// e.g., per-batch [128, 256, 128, 256] -> cumulative [128, 384, 512, 768]
static std::vector<int64_t> BuildCumulativeSeqLens(const std::vector<int64_t>& perBatch) {
  std::vector<int64_t> cumulative(perBatch.size());
  int64_t sum = 0;
  for (size_t i = 0; i < perBatch.size(); i++) {
    sum += perBatch[i];
    cumulative[i] = sum;
  }
  return cumulative;
}

// Macro: create NodeDef with 2 inputs, 1 output, and all required/optional attributes
#define CREATE_NODEDEF(q_shape, kv_shape, out_shape, q_data, kv_data, out_data)                     \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                                \
    NodeDefBuilder(node_def.get(), "AttentionPioneerMetadata", "AttentionPioneerMetadata")           \
        .Input({"actual_seq_lengths_q", DT_INT64, q_shape, q_data})                                 \
        .Input({"actual_seq_lengths_kv", DT_INT64, kv_shape, kv_data})                              \
        .Output({"metaData_out", DT_INT32, out_shape, out_data})                                    \
        .Attr("num_heads_q", num_heads_q)                                                           \
        .Attr("num_heads_kv", num_heads_kv)                                                         \
        .Attr("head_dim_qk", head_dim_qk)                                                          \
        .Attr("head_dim_v", head_dim_v)                                                             \
        .Attr("sparse_mode", sparse_mode)                                                           \
        .Attr("pre_tokens", pre_tokens)                                                             \
        .Attr("next_tokens", next_tokens)                                                           \
        .Attr("input_layout", input_layout)                                                         \
        .Attr("k_sink_num", k_sink_num)                                                             \
        .Attr("rope_head_dim", rope_head_dim)                                                       \
        .Attr("block_size", block_size)                                                             \
        .Attr("soc_version", soc_version)                                                           \
        .Attr("aic_core_num", aic_core_num)                                                         \
        .Attr("aiv_core_num", aiv_core_num);

// ============================================================
// Test 1: P-side, sparse_mode=2 (LEFT_UP_CAUSAL), batch=4
// ============================================================
TEST_F(TEST_ATTENTION_PIONEER_METADATA_UT, PSIDE_LEFT_UP_CAUSAL_SUCC) {
  // Attributes
  int64_t num_heads_q = 32;
  int64_t num_heads_kv = 8;
  int64_t head_dim_qk = 128;
  int64_t head_dim_v = 128;
  int64_t sparse_mode = 2;
  int64_t pre_tokens = 2147483647;
  int64_t next_tokens = 0;
  string input_layout = "TND";
  int64_t k_sink_num = 0;
  int64_t rope_head_dim = 0;
  int64_t block_size = 0;
  string soc_version = "ascend950";
  int64_t aic_core_num = 36;
  int64_t aiv_core_num = 72;

  // Q cumulative seq lengths: per-batch [128, 256, 128, 256] -> [128, 384, 512, 768]
  auto qSeqPerBatch = std::vector<int64_t>{128, 256, 128, 256};
  auto qSeqCumulative = BuildCumulativeSeqLens(qSeqPerBatch);
  // KV cumulative seq lengths: per-batch [512, 1024, 512, 1024] -> [512, 1536, 2048, 3072]
  auto kvSeqPerBatch = std::vector<int64_t>{512, 1024, 512, 1024};
  auto kvSeqCumulative = BuildCumulativeSeqLens(kvSeqPerBatch);

  int32_t metaDataOut[METADATA_ELEM_COUNT] = {0};

  vector<int64_t> q_shape = {static_cast<int64_t>(qSeqCumulative.size())};
  vector<int64_t> kv_shape = {static_cast<int64_t>(kvSeqCumulative.size())};
  vector<int64_t> out_shape = {METADATA_ELEM_COUNT};

  CREATE_NODEDEF(q_shape, kv_shape, out_shape,
                 (void*)qSeqCumulative.data(), (void*)kvSeqCumulative.data(), (void*)metaDataOut);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  // Verify base metadata
  auto* meta = reinterpret_cast<detail::APMetaData*>(metaDataOut);
  EXPECT_GT(meta->baseMetadata[USED_CORE_NUM_INDEX], 0);
  EXPECT_EQ(meta->baseMetadata[M_BASE_SIZE_INDEX], 128);       // P-side sOuterSize
  EXPECT_EQ(meta->baseMetadata[S_INNER_SIZE_INDEX], 128);      // sInnerSize
  EXPECT_EQ(meta->baseMetadata[S1_SIZE_INDEX], 256);            // max Q seq len
  EXPECT_EQ(meta->baseMetadata[S2_SIZE_INDEX], 1024);           // max KV seq len
  EXPECT_EQ(meta->baseMetadata[ACTUAL_LEN_Q_DIM_INDEX], 4);    // batch size
  EXPECT_EQ(meta->baseMetadata[ACTUAL_LEN_KV_DIM_INDEX], 4);
  EXPECT_EQ(meta->baseMetadata[T1_SIZE_INDEX], 768);            // last cumulative Q
  EXPECT_EQ(meta->baseMetadata[T2_SIZE_INDEX], 3072);           // last cumulative KV
  // Core 0 should always be enabled
  EXPECT_EQ(meta->coreSlots[0].fields[AIC_CORE_ENABLE_INDEX], 1);
}

// ============================================================
// Test 2: D-side (MLA decode), sparse_mode=2, batch=4
// ============================================================
TEST_F(TEST_ATTENTION_PIONEER_METADATA_UT, DSIDE_LEFT_UP_CAUSAL_SUCC) {
  int64_t num_heads_q = 128;
  int64_t num_heads_kv = 1;
  int64_t head_dim_qk = 512;
  int64_t head_dim_v = 512;
  int64_t sparse_mode = 2;
  int64_t pre_tokens = 2147483647;
  int64_t next_tokens = 0;
  string input_layout = "TND";
  int64_t k_sink_num = 0;
  int64_t rope_head_dim = 64;
  int64_t block_size = 0;
  string soc_version = "ascend950";
  int64_t aic_core_num = 36;
  int64_t aiv_core_num = 72;

  // D-side: each batch has 1 token for Q
  auto qSeqCumulative = std::vector<int64_t>{1, 2, 3, 4};
  // KV: cumulative [256, 768, 1280, 2048]
  auto kvSeqPerBatch = std::vector<int64_t>{256, 512, 512, 768};
  auto kvSeqCumulative = BuildCumulativeSeqLens(kvSeqPerBatch);

  int32_t metaDataOut[METADATA_ELEM_COUNT] = {0};

  vector<int64_t> q_shape = {4};
  vector<int64_t> kv_shape = {4};
  vector<int64_t> out_shape = {METADATA_ELEM_COUNT};

  CREATE_NODEDEF(q_shape, kv_shape, out_shape,
                 (void*)qSeqCumulative.data(), (void*)kvSeqCumulative.data(), (void*)metaDataOut);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  auto* meta = reinterpret_cast<detail::APMetaData*>(metaDataOut);
  EXPECT_GT(meta->baseMetadata[USED_CORE_NUM_INDEX], 0);
  EXPECT_EQ(meta->baseMetadata[M_BASE_SIZE_INDEX], 64);        // D-side sOuterSize
  EXPECT_EQ(meta->baseMetadata[S_INNER_SIZE_INDEX], 128);
  EXPECT_EQ(meta->baseMetadata[T1_SIZE_INDEX], 4);
  EXPECT_EQ(meta->baseMetadata[T2_SIZE_INDEX], 2048);
}

// ============================================================
// Test 3: P-side, sparse_mode=4 (BAND), with pre/next tokens
// ============================================================
TEST_F(TEST_ATTENTION_PIONEER_METADATA_UT, PSIDE_BAND_SUCC) {
  int64_t num_heads_q = 32;
  int64_t num_heads_kv = 8;
  int64_t head_dim_qk = 128;
  int64_t head_dim_v = 128;
  int64_t sparse_mode = 4;
  int64_t pre_tokens = 512;
  int64_t next_tokens = 512;
  string input_layout = "TND";
  int64_t k_sink_num = 0;
  int64_t rope_head_dim = 0;
  int64_t block_size = 0;
  string soc_version = "ascend950";
  int64_t aic_core_num = 36;
  int64_t aiv_core_num = 72;

  auto qSeqPerBatch = std::vector<int64_t>{256, 256};
  auto qSeqCumulative = BuildCumulativeSeqLens(qSeqPerBatch);
  auto kvSeqPerBatch = std::vector<int64_t>{1024, 1024};
  auto kvSeqCumulative = BuildCumulativeSeqLens(kvSeqPerBatch);

  int32_t metaDataOut[METADATA_ELEM_COUNT] = {0};

  vector<int64_t> q_shape = {2};
  vector<int64_t> kv_shape = {2};
  vector<int64_t> out_shape = {METADATA_ELEM_COUNT};

  CREATE_NODEDEF(q_shape, kv_shape, out_shape,
                 (void*)qSeqCumulative.data(), (void*)kvSeqCumulative.data(), (void*)metaDataOut);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  auto* meta = reinterpret_cast<detail::APMetaData*>(metaDataOut);
  EXPECT_GT(meta->baseMetadata[USED_CORE_NUM_INDEX], 0);
  EXPECT_EQ(meta->baseMetadata[M_BASE_SIZE_INDEX], 128);
  EXPECT_EQ(meta->baseMetadata[ACTUAL_LEN_Q_DIM_INDEX], 2);
}

// ============================================================
// Test 4: P-side with k_sink_num=128
// ============================================================
TEST_F(TEST_ATTENTION_PIONEER_METADATA_UT, PSIDE_WITH_SINK_SUCC) {
  int64_t num_heads_q = 32;
  int64_t num_heads_kv = 8;
  int64_t head_dim_qk = 128;
  int64_t head_dim_v = 128;
  int64_t sparse_mode = 2;
  int64_t pre_tokens = 2147483647;
  int64_t next_tokens = 0;
  string input_layout = "TND";
  int64_t k_sink_num = 128;
  int64_t rope_head_dim = 0;
  int64_t block_size = 0;
  string soc_version = "ascend950";
  int64_t aic_core_num = 36;
  int64_t aiv_core_num = 72;

  auto qSeqCumulative = std::vector<int64_t>{128, 256};
  auto kvSeqPerBatch = std::vector<int64_t>{512, 512};
  auto kvSeqCumulative = BuildCumulativeSeqLens(kvSeqPerBatch);

  int32_t metaDataOut[METADATA_ELEM_COUNT] = {0};

  vector<int64_t> q_shape = {2};
  vector<int64_t> kv_shape = {2};
  vector<int64_t> out_shape = {METADATA_ELEM_COUNT};

  CREATE_NODEDEF(q_shape, kv_shape, out_shape,
                 (void*)qSeqCumulative.data(), (void*)kvSeqCumulative.data(), (void*)metaDataOut);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  auto* meta = reinterpret_cast<detail::APMetaData*>(metaDataOut);
  EXPECT_GT(meta->baseMetadata[USED_CORE_NUM_INDEX], 0);
}

// ============================================================
// Test 5: TND_NTD layout
// ============================================================
TEST_F(TEST_ATTENTION_PIONEER_METADATA_UT, PSIDE_TND_NTD_LAYOUT_SUCC) {
  int64_t num_heads_q = 32;
  int64_t num_heads_kv = 8;
  int64_t head_dim_qk = 128;
  int64_t head_dim_v = 128;
  int64_t sparse_mode = 2;
  int64_t pre_tokens = 2147483647;
  int64_t next_tokens = 0;
  string input_layout = "TND_NTD";
  int64_t k_sink_num = 0;
  int64_t rope_head_dim = 0;
  int64_t block_size = 0;
  string soc_version = "ascend950";
  int64_t aic_core_num = 36;
  int64_t aiv_core_num = 72;

  auto qSeqCumulative = std::vector<int64_t>{128, 384};
  auto kvSeqPerBatch = std::vector<int64_t>{512, 1024};
  auto kvSeqCumulative = BuildCumulativeSeqLens(kvSeqPerBatch);

  int32_t metaDataOut[METADATA_ELEM_COUNT] = {0};

  vector<int64_t> q_shape = {2};
  vector<int64_t> kv_shape = {2};
  vector<int64_t> out_shape = {METADATA_ELEM_COUNT};

  CREATE_NODEDEF(q_shape, kv_shape, out_shape,
                 (void*)qSeqCumulative.data(), (void*)kvSeqCumulative.data(), (void*)metaDataOut);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  auto* meta = reinterpret_cast<detail::APMetaData*>(metaDataOut);
  EXPECT_GT(meta->baseMetadata[USED_CORE_NUM_INDEX], 0);
}

// ============================================================
// Test 6: Invalid - numHeadsQ not divisible by numHeadsKv
// ============================================================
TEST_F(TEST_ATTENTION_PIONEER_METADATA_UT, INVALID_HEAD_NUM_NOT_DIVISIBLE) {
  int64_t num_heads_q = 32;
  int64_t num_heads_kv = 5;
  int64_t head_dim_qk = 128;
  int64_t head_dim_v = 128;
  int64_t sparse_mode = 2;
  int64_t pre_tokens = 2147483647;
  int64_t next_tokens = 0;
  string input_layout = "TND";
  int64_t k_sink_num = 0;
  int64_t rope_head_dim = 0;
  int64_t block_size = 0;
  string soc_version = "ascend950";
  int64_t aic_core_num = 36;
  int64_t aiv_core_num = 72;

  auto qSeqCumulative = std::vector<int64_t>{128};
  auto kvSeqCumulative = std::vector<int64_t>{512};

  int32_t metaDataOut[METADATA_ELEM_COUNT] = {0};

  vector<int64_t> q_shape = {1};
  vector<int64_t> kv_shape = {1};
  vector<int64_t> out_shape = {METADATA_ELEM_COUNT};

  CREATE_NODEDEF(q_shape, kv_shape, out_shape,
                 (void*)qSeqCumulative.data(), (void*)kvSeqCumulative.data(), (void*)metaDataOut);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ============================================================
// Test 7: Invalid - core num is zero
// ============================================================
TEST_F(TEST_ATTENTION_PIONEER_METADATA_UT, INVALID_CORE_NUM_ZERO) {
  int64_t num_heads_q = 32;
  int64_t num_heads_kv = 8;
  int64_t head_dim_qk = 128;
  int64_t head_dim_v = 128;
  int64_t sparse_mode = 2;
  int64_t pre_tokens = 2147483647;
  int64_t next_tokens = 0;
  string input_layout = "TND";
  int64_t k_sink_num = 0;
  int64_t rope_head_dim = 0;
  int64_t block_size = 0;
  string soc_version = "ascend950";
  int64_t aic_core_num = 0;
  int64_t aiv_core_num = 0;

  auto qSeqCumulative = std::vector<int64_t>{128};
  auto kvSeqCumulative = std::vector<int64_t>{512};

  int32_t metaDataOut[METADATA_ELEM_COUNT] = {0};

  vector<int64_t> q_shape = {1};
  vector<int64_t> kv_shape = {1};
  vector<int64_t> out_shape = {METADATA_ELEM_COUNT};

  CREATE_NODEDEF(q_shape, kv_shape, out_shape,
                 (void*)qSeqCumulative.data(), (void*)kvSeqCumulative.data(), (void*)metaDataOut);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ============================================================
// Test 8: Invalid - Q seq lengths decreasing (TND cumulative)
// ============================================================
TEST_F(TEST_ATTENTION_PIONEER_METADATA_UT, INVALID_Q_SEQ_DECREASING) {
  int64_t num_heads_q = 32;
  int64_t num_heads_kv = 8;
  int64_t head_dim_qk = 128;
  int64_t head_dim_v = 128;
  int64_t sparse_mode = 2;
  int64_t pre_tokens = 2147483647;
  int64_t next_tokens = 0;
  string input_layout = "TND";
  int64_t k_sink_num = 0;
  int64_t rope_head_dim = 0;
  int64_t block_size = 0;
  string soc_version = "ascend950";
  int64_t aic_core_num = 36;
  int64_t aiv_core_num = 72;

  // Decreasing cumulative Q: invalid
  auto qSeqCumulative = std::vector<int64_t>{256, 128};
  auto kvSeqCumulative = std::vector<int64_t>{512, 1024};

  int32_t metaDataOut[METADATA_ELEM_COUNT] = {0};

  vector<int64_t> q_shape = {2};
  vector<int64_t> kv_shape = {2};
  vector<int64_t> out_shape = {METADATA_ELEM_COUNT};

  CREATE_NODEDEF(q_shape, kv_shape, out_shape,
                 (void*)qSeqCumulative.data(), (void*)kvSeqCumulative.data(), (void*)metaDataOut);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

// ============================================================
// Test 9: Single batch, batch=1
// ============================================================
TEST_F(TEST_ATTENTION_PIONEER_METADATA_UT, PSIDE_SINGLE_BATCH_SUCC) {
  int64_t num_heads_q = 32;
  int64_t num_heads_kv = 8;
  int64_t head_dim_qk = 128;
  int64_t head_dim_v = 128;
  int64_t sparse_mode = 2;
  int64_t pre_tokens = 2147483647;
  int64_t next_tokens = 0;
  string input_layout = "TND";
  int64_t k_sink_num = 0;
  int64_t rope_head_dim = 0;
  int64_t block_size = 0;
  string soc_version = "ascend950";
  int64_t aic_core_num = 36;
  int64_t aiv_core_num = 72;

  auto qSeqCumulative = std::vector<int64_t>{512};
  auto kvSeqCumulative = std::vector<int64_t>{2048};

  int32_t metaDataOut[METADATA_ELEM_COUNT] = {0};

  vector<int64_t> q_shape = {1};
  vector<int64_t> kv_shape = {1};
  vector<int64_t> out_shape = {METADATA_ELEM_COUNT};

  CREATE_NODEDEF(q_shape, kv_shape, out_shape,
                 (void*)qSeqCumulative.data(), (void*)kvSeqCumulative.data(), (void*)metaDataOut);
  RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

  auto* meta = reinterpret_cast<detail::APMetaData*>(metaDataOut);
  EXPECT_GT(meta->baseMetadata[USED_CORE_NUM_INDEX], 0);
  EXPECT_EQ(meta->baseMetadata[S1_SIZE_INDEX], 512);
  EXPECT_EQ(meta->baseMetadata[S2_SIZE_INDEX], 2048);
  EXPECT_EQ(meta->baseMetadata[ACTUAL_LEN_Q_DIM_INDEX], 1);
  EXPECT_EQ(meta->baseMetadata[T1_SIZE_INDEX], 512);
  EXPECT_EQ(meta->baseMetadata[T2_SIZE_INDEX], 2048);
}
