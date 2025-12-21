/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
// #include "test_ring_attention_update.h"
#include "../../../op_host/ring_attention_update_tiling.h"
#include "../../../op_kernel/ring_attention_update.h"
#include "../../../op_kernel/ring_attention_update_tnd.h"
#include "../../../../../math/ring_attention_update/op_kernel/ring_attention_update.cpp"
#include "data_utils.h"


#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void ring_attention_update(GM_ADDR prevAttnOut, GM_ADDR prevSoftmaxMax, GM_ADDR prevSoftmaxSum,
                                                            GM_ADDR curAttnOut, GM_ADDR curSoftmaxMax, GM_ADDR curSoftmaxSum,
                                                            GM_ADDR actualSeqQlen,
                                                            GM_ADDR attnOut, GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                                                            GM_ADDR workspace, GM_ADDR tiling);

class ring_attention_update_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "ring_attention_update_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "ring_attention_update_test TearDown\n" << endl;
  }
};

TEST_F(ring_attention_update_test, test_case_0) {
  system(
    "cp -rf "
    "../../../../../../../ops/built-in/tests/ut/fast_op_test/ring_attention_update/gen_data "
    "./");
  system("chmod -R 755 ./gen_data/");
  system("cd ./gen_data/ && python3 gen_data.py 2 12 1024 128 float16");

  size_t attnByteSize = 2 * 12 * 1024 * 128 * sizeof(half);
  size_t softmaxByteSize = 2 * 12 * 1024 * 8 * sizeof(float);
  size_t tiling_data_size = sizeof(RingAttentionUpdateTilingData);

  uint8_t* prev_attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* prev_softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* prev_softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* cur_attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* cur_softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* cur_softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* actual_seq_qlen = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 1;

  char* path_ = get_current_dir_name();
  string path(path_);
  auto read_file_0 = ReadFile(path + "/gen_data/prev_attn_out.bin", attnByteSize, prev_attn_out, attnByteSize);
  auto read_file_1 = ReadFile(path + "/gen_data/prev_softmax_max.bin", softmaxByteSize, prev_softmax_max, softmaxByteSize);
  auto read_file_2 = ReadFile(path + "/gen_data/prev_softmax_sum.bin", softmaxByteSize, prev_softmax_sum, softmaxByteSize);
  auto read_file_3 = ReadFile(path + "/gen_data/cur_attn_out.bin", attnByteSize, cur_attn_out, attnByteSize);
  auto read_file_4 = ReadFile(path + "/gen_data/cur_softmax_max.bin", softmaxByteSize, cur_softmax_max, softmaxByteSize);
  auto read_file_5 = ReadFile(path + "/gen_data/cur_softmax_sum.bin", softmaxByteSize, cur_softmax_sum, softmaxByteSize);
  auto read_file_6 = ReadFile(path + "/gen_data/cur_softmax_sum.bin", softmaxByteSize, actual_seq_qlen, softmaxByteSize);

  RingAttentionUpdateTilingData* tilingDatafromBin = reinterpret_cast<RingAttentionUpdateTilingData*>(tiling);

  tilingDatafromBin->batchSize = 2;
  tilingDatafromBin->headNum = 12;
  tilingDatafromBin->seqNum = 1024;
  tilingDatafromBin->headDim = 128;
  tilingDatafromBin->softmaxTailSize = 8;

  tilingDatafromBin->coreNum = 48;
  tilingDatafromBin->coreNumGroup = 2;
  tilingDatafromBin->bnNumGroup = 1;
  tilingDatafromBin->seqNumCoreEach = 512;
  tilingDatafromBin->seqNumCoreTail = 512;
  tilingDatafromBin->seqNumLoopEach = 64;
  tilingDatafromBin->headDimLoopEach = 128;

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(ring_attention_update, blockDim,
              prev_attn_out, prev_softmax_max, prev_softmax_sum,
              cur_attn_out, cur_softmax_max, cur_softmax_sum,
              actual_seq_qlen,
              attn_out, softmax_max, softmax_sum,
              workspace, tiling);

  AscendC::GmFree(prev_attn_out);
  AscendC::GmFree(prev_softmax_max);
  AscendC::GmFree(prev_softmax_sum);
  AscendC::GmFree(cur_attn_out);
  AscendC::GmFree(cur_softmax_max);
  AscendC::GmFree(cur_softmax_sum);
  AscendC::GmFree(actual_seq_qlen);
  AscendC::GmFree(attn_out);
  AscendC::GmFree(softmax_max);
  AscendC::GmFree(softmax_sum);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(ring_attention_update_test, test_case_1) {
  system(
    "cp -rf "
    "../../../../../../../ops/built-in/tests/ut/fast_op_test/ring_attention_update/gen_data "
    "./");
  system("chmod -R 755 ./gen_data/");
  system("cd ./gen_data/ && python3 gen_data.py 2 12 1024 22 float");

  size_t attnByteSize = 2 * 12 * 1024 * 22 * sizeof(float);
  size_t softmaxByteSize = 2 * 12 * 1024 * 8 * sizeof(float);
  size_t tiling_data_size = sizeof(RingAttentionUpdateTilingData);

  uint8_t* prev_attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* prev_softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* prev_softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* cur_attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* cur_softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* cur_softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* actual_seq_qlen = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 1;

  char* path_ = get_current_dir_name();
  string path(path_);
  auto read_file_0 = ReadFile(path + "/gen_data/prev_attn_out.bin", attnByteSize, prev_attn_out, attnByteSize);
  auto read_file_1 = ReadFile(path + "/gen_data/prev_softmax_max.bin", softmaxByteSize, prev_softmax_max, softmaxByteSize);
  auto read_file_2 = ReadFile(path + "/gen_data/prev_softmax_sum.bin", softmaxByteSize, prev_softmax_sum, softmaxByteSize);
  auto read_file_3 = ReadFile(path + "/gen_data/cur_attn_out.bin", attnByteSize, cur_attn_out, attnByteSize);
  auto read_file_4 = ReadFile(path + "/gen_data/cur_softmax_max.bin", softmaxByteSize, cur_softmax_max, softmaxByteSize);
  auto read_file_5 = ReadFile(path + "/gen_data/cur_softmax_sum.bin", softmaxByteSize, cur_softmax_sum, softmaxByteSize);
  auto read_file_6 = ReadFile(path + "/gen_data/cur_softmax_sum.bin", softmaxByteSize, actual_seq_qlen, softmaxByteSize);

  RingAttentionUpdateTilingData* tilingDatafromBin = reinterpret_cast<RingAttentionUpdateTilingData*>(tiling);

  tilingDatafromBin->batchSize = 2;
  tilingDatafromBin->headNum = 12;
  tilingDatafromBin->seqNum = 1024;
  tilingDatafromBin->headDim = 128;
  tilingDatafromBin->softmaxTailSize = 8;

  tilingDatafromBin->coreNum = 48;
  tilingDatafromBin->coreNumGroup = 2;
  tilingDatafromBin->bnNumGroup = 1;
  tilingDatafromBin->seqNumCoreEach = 512;
  tilingDatafromBin->seqNumCoreTail = 512;
  tilingDatafromBin->seqNumLoopEach = 40;
  tilingDatafromBin->headDimLoopEach = 128;

  ICPU_SET_TILING_KEY(2);
  ICPU_RUN_KF(ring_attention_update, blockDim,
              prev_attn_out, prev_softmax_max, prev_softmax_sum,
              cur_attn_out, cur_softmax_max, cur_softmax_sum,
              actual_seq_qlen,
              attn_out, softmax_max, softmax_sum,
              workspace, tiling);

  AscendC::GmFree(prev_attn_out);
  AscendC::GmFree(prev_softmax_max);
  AscendC::GmFree(prev_softmax_sum);
  AscendC::GmFree(cur_attn_out);
  AscendC::GmFree(cur_softmax_max);
  AscendC::GmFree(cur_softmax_sum);
  AscendC::GmFree(actual_seq_qlen);
  AscendC::GmFree(attn_out);
  AscendC::GmFree(softmax_max);
  AscendC::GmFree(softmax_sum);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(ring_attention_update_test, test_case_2) {
  system(
    "cp -rf "
    "../../../../../../../ops/built-in/tests/ut/fast_op_test/ring_attention_update/gen_data "
    "./");
  system("chmod -R 755 ./gen_data/");
  system("cd ./gen_data/ && python3 gen_data.py 2 8 13 12816 float16");

  size_t attnByteSize = 2 * 8 * 13 * 12816 * sizeof(half);
  size_t softmaxByteSize = 2 * 8 * 13 * 8 * sizeof(float);
  size_t tiling_data_size = sizeof(RingAttentionUpdateTilingData);

  uint8_t* prev_attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* prev_softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* prev_softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* cur_attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* cur_softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* cur_softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* actual_seq_qlen = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 1;

  char* path_ = get_current_dir_name();
  string path(path_);
  auto read_file_0 = ReadFile(path + "/gen_data/prev_attn_out.bin", attnByteSize, prev_attn_out, attnByteSize);
  auto read_file_1 = ReadFile(path + "/gen_data/prev_softmax_max.bin", softmaxByteSize, prev_softmax_max, softmaxByteSize);
  auto read_file_2 = ReadFile(path + "/gen_data/prev_softmax_sum.bin", softmaxByteSize, prev_softmax_sum, softmaxByteSize);
  auto read_file_3 = ReadFile(path + "/gen_data/cur_attn_out.bin", attnByteSize, cur_attn_out, attnByteSize);
  auto read_file_4 = ReadFile(path + "/gen_data/cur_softmax_max.bin", softmaxByteSize, cur_softmax_max, softmaxByteSize);
  auto read_file_5 = ReadFile(path + "/gen_data/cur_softmax_sum.bin", softmaxByteSize, cur_softmax_sum, softmaxByteSize);
  auto read_file_6 = ReadFile(path + "/gen_data/cur_softmax_sum.bin", softmaxByteSize, actual_seq_qlen, softmaxByteSize);

  RingAttentionUpdateTilingData* tilingDatafromBin = reinterpret_cast<RingAttentionUpdateTilingData*>(tiling);

  tilingDatafromBin->batchSize = 2;
  tilingDatafromBin->headNum = 8;
  tilingDatafromBin->seqNum = 13;
  tilingDatafromBin->headDim = 12816;
  tilingDatafromBin->softmaxTailSize = 8;

  tilingDatafromBin->coreNum = 48;
  tilingDatafromBin->coreNumGroup = 3;
  tilingDatafromBin->bnNumGroup = 1;
  tilingDatafromBin->seqNumCoreEach = 5;
  tilingDatafromBin->seqNumCoreTail = 3;
  tilingDatafromBin->seqNumLoopEach = 8;
  tilingDatafromBin->headDimLoopEach = 704;

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(ring_attention_update, blockDim,
              prev_attn_out, prev_softmax_max, prev_softmax_sum,
              cur_attn_out, cur_softmax_max, cur_softmax_sum,
              actual_seq_qlen,
              attn_out, softmax_max, softmax_sum,
              workspace, tiling);

  AscendC::GmFree(prev_attn_out);
  AscendC::GmFree(prev_softmax_max);
  AscendC::GmFree(prev_softmax_sum);
  AscendC::GmFree(cur_attn_out);
  AscendC::GmFree(cur_softmax_max);
  AscendC::GmFree(cur_softmax_sum);
  AscendC::GmFree(actual_seq_qlen);
  AscendC::GmFree(attn_out);
  AscendC::GmFree(softmax_max);
  AscendC::GmFree(softmax_sum);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(ring_attention_update_test, test_case_3) {
  system(
    "cp -rf "
    "../../../../../../../ops/built-in/tests/ut/fast_op_test/ring_attention_update/gen_data "
    "./");
  system("chmod -R 755 ./gen_data/");
  system("cd ./gen_data/ && python3 gen_data.py 3072 12 0 128 float16");

  size_t attnByteSize = 3072 * 12 * 128 * sizeof(half);
  size_t softmaxByteSize = 3072 * 12 * 8 * sizeof(float);
  size_t qlenByteSize = (3072 + 1) * sizeof(int64_t);
  size_t tiling_data_size = sizeof(RingAttentionUpdateTilingData);

  uint8_t* prev_attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* prev_softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* prev_softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* cur_attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* cur_softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* cur_softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* actual_seq_qlen = (uint8_t*)AscendC::GmAlloc(qlenByteSize);

  uint8_t* attn_out = (uint8_t*)AscendC::GmAlloc(attnByteSize);
  uint8_t* softmax_max = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);
  uint8_t* softmax_sum = (uint8_t*)AscendC::GmAlloc(softmaxByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
  uint32_t blockDim = 1;

  char* path_ = get_current_dir_name();
  string path(path_);
  auto read_file_0 = ReadFile(path + "/gen_data/prev_attn_out.bin", attnByteSize, prev_attn_out, attnByteSize);
  auto read_file_1 = ReadFile(path + "/gen_data/prev_softmax_max.bin", softmaxByteSize, prev_softmax_max, softmaxByteSize);
  auto read_file_2 = ReadFile(path + "/gen_data/prev_softmax_sum.bin", softmaxByteSize, prev_softmax_sum, softmaxByteSize);
  auto read_file_3 = ReadFile(path + "/gen_data/cur_attn_out.bin", attnByteSize, cur_attn_out, attnByteSize);
  auto read_file_4 = ReadFile(path + "/gen_data/cur_softmax_max.bin", softmaxByteSize, cur_softmax_max, softmaxByteSize);
  auto read_file_5 = ReadFile(path + "/gen_data/cur_softmax_sum.bin", softmaxByteSize, cur_softmax_sum, softmaxByteSize);
  auto read_file_6 = ReadFile(path + "/gen_data/actual_seq_qlen.bin", qlenByteSize, actual_seq_qlen, qlenByteSize);

  RingAttentionUpdateTilingData* tilingDatafromBin = reinterpret_cast<RingAttentionUpdateTilingData*>(tiling);

  tilingDatafromBin->batchSize = 3072;
  tilingDatafromBin->headNum = 12;
  tilingDatafromBin->seqNum = 0;
  tilingDatafromBin->headDim = 128;
  tilingDatafromBin->softmaxTailSize = 8;

  tilingDatafromBin->coreNum = 48;
  tilingDatafromBin->coreNumGroup = 0;
  tilingDatafromBin->bnNumGroup = 0;
  tilingDatafromBin->seqNumCoreEach = 0;
  tilingDatafromBin->seqNumCoreTail = 0;
  tilingDatafromBin->seqNumLoopEach = 1;
  tilingDatafromBin->headNumLoopEach = 12;
  tilingDatafromBin->headDimLoopEach = 0;

  tilingDatafromBin->batchSizeCoreEach = 64;
  tilingDatafromBin->batchSizeCoreTail = 64;

  ICPU_SET_TILING_KEY(10);
  ICPU_RUN_KF(ring_attention_update, blockDim,
              prev_attn_out, prev_softmax_max, prev_softmax_sum,
              cur_attn_out, cur_softmax_max, cur_softmax_sum,
              actual_seq_qlen,
              attn_out, softmax_max, softmax_sum,
              workspace, tiling);

  AscendC::GmFree(prev_attn_out);
  AscendC::GmFree(prev_softmax_max);
  AscendC::GmFree(prev_softmax_sum);
  AscendC::GmFree(cur_attn_out);
  AscendC::GmFree(cur_softmax_max);
  AscendC::GmFree(cur_softmax_sum);
  AscendC::GmFree(actual_seq_qlen);
  AscendC::GmFree(attn_out);
  AscendC::GmFree(softmax_max);
  AscendC::GmFree(softmax_sum);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}