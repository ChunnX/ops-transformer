/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "moe_token_unpermute_with_routing_map_tiling.h"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void moe_token_unpermute_with_routing_map(GM_ADDR permutedTokens,
                                                                           GM_ADDR sortedIndices,
                                                                           GM_ADDR routingMap,
                                                                           GM_ADDR probs,
                                                                           GM_ADDR unpermutedTokens,
                                                                           GM_ADDR outIndex,
                                                                           GM_ADDR permuteTokenId,
                                                                           GM_ADDR permuteProbs,
                                                                           GM_ADDR workspace, GM_ADDR tiling);

class moe_token_unpermute_with_routing_map_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "moe_token_unpermute_with_routing_map_test SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "moe_token_unpermute_with_routing_map_test TearDown\n" << endl;
  }
};

TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_fp16) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(half);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(half);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(half);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(half);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, tiling);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_fp32) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(float);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(float);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(float);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(float);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, tiling);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_bf16) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(bfloat16_t);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(bfloat16_t);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(bfloat16_t);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  ICPU_SET_TILING_KEY(0);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, tiling);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}


TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_fp16_probs) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(half);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(half);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(half);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(half);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  ICPU_SET_TILING_KEY(1);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, tiling);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_fp32_probs) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(float);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(float);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(float);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(float);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  ICPU_SET_TILING_KEY(1);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, tiling);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_test, test_non_pad_bf16_probs) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 4096;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;
  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = numTokens * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * hidden * sizeof(bfloat16_t);
  // output
  size_t unpermutedTokensByteSize = numTokens * hidden * sizeof(bfloat16_t);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numTokens * topkNum * sizeof(bfloat16_t);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  ICPU_SET_TILING_KEY(1);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, tiling);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}


TEST_F(moe_token_unpermute_with_routing_map_test, test_pad_fp32) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 256;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;

  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(float);
  size_t sortedIndicesByteSize = numExperts * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * numExperts * sizeof(float);
  // output
  size_t unpermutedTokensByteSize = numExperts * hidden * sizeof(float);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numExperts * topkNum * sizeof(float);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  ICPU_SET_TILING_KEY(1000);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, tiling);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}

TEST_F(moe_token_unpermute_with_routing_map_test, test_pad_fp16) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 256;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;

  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(half);
  size_t sortedIndicesByteSize = numExperts * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * numExperts * sizeof(half);
  // output
  size_t unpermutedTokensByteSize = numExperts * hidden * sizeof(half);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numExperts * topkNum * sizeof(half);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  ICPU_SET_TILING_KEY(1000);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, tiling);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}


TEST_F(moe_token_unpermute_with_routing_map_test, test_pad_bf16) {
  uint64_t numTokens = 4096;
  uint64_t hidden = 7168;
  uint64_t numExperts = 256;
  uint64_t topkNum = 8;
  uint64_t numOutTokens = 4096;

  size_t permutedTokensByteSize = numTokens * topkNum * hidden * sizeof(bfloat16_t);
  size_t sortedIndicesByteSize = numExperts * topkNum * sizeof(int32_t);
  size_t routingMapByteSize = numTokens * numExperts * sizeof(bool);
  size_t probsByteSize = numTokens * numExperts * sizeof(bfloat16_t);
  // output
  size_t unpermutedTokensByteSize = numExperts * hidden * sizeof(bfloat16_t);
  size_t outIndexByteSize = 0;
  size_t permuteTokenIdByteSize = 0;
  size_t permuteProbsByteSize = numExperts * topkNum * sizeof(bfloat16_t);

  size_t tilingDataSize = sizeof(MoeTokenUnpermuteWithRoutingMapTilingData);

  uint8_t* permutedTokens = (uint8_t*)AscendC::GmAlloc(permutedTokensByteSize);
  uint8_t* sortedIndices = (uint8_t*)AscendC::GmAlloc(sortedIndicesByteSize);
  uint8_t* routingMap = (uint8_t*)AscendC::GmAlloc(routingMapByteSize);
  uint8_t* probs = (uint8_t*)AscendC::GmAlloc(probsByteSize);
  uint8_t* unpermutedTokens = (uint8_t*)AscendC::GmAlloc(unpermutedTokensByteSize);
  uint8_t* outIndex = (uint8_t*)AscendC::GmAlloc(outIndexByteSize);
  uint8_t* permuteTokenId = (uint8_t*)AscendC::GmAlloc(permuteTokenIdByteSize);
  uint8_t* permuteProbs = (uint8_t*)AscendC::GmAlloc(permuteProbsByteSize);

  uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
  uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
  uint32_t blockDim = 48;

  char* path_ = get_current_dir_name();
  string path(path_);

  MoeTokenUnpermuteWithRoutingMapTilingData* tilingDatafromBin = reinterpret_cast<MoeTokenUnpermuteWithRoutingMapTilingData*>(tiling);

  ICPU_SET_TILING_KEY(1000);
  ICPU_RUN_KF(moe_token_unpermute_with_routing_map, blockDim, permutedTokens, sortedIndices,
              routingMap, probs, unpermutedTokens, outIndex, permuteTokenId, permuteProbs, workspace, tiling);

  AscendC::GmFree(permutedTokens);
  AscendC::GmFree(sortedIndices);
  AscendC::GmFree(routingMap);
  AscendC::GmFree(probs);
  AscendC::GmFree(unpermutedTokens);
  AscendC::GmFree(outIndex);
  AscendC::GmFree(permuteTokenId);
  AscendC::GmFree(permuteProbs);
  AscendC::GmFree(workspace);
  AscendC::GmFree(tiling);
  free(path_);
}