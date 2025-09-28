/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file distribute_barrier_tiling.cc
 * \brief
 */

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <string>
#include <vector>

#include "../../op_kernel/distribute_barrier_tiling.h"
// #include "graph/utils/op_desc_utils.h"   // 依赖 ge
#include "graph/utils/type_utils.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_log.h"
#include "register/op_def_registry.h"
#include "tiling/mc2_tiling_utils.h"

using namespace AscendC;
using namespace ge;

namespace optiling {
constexpr uint64_t INIT_TILINGKEY = 10000UL;

constexpr uint32_t INPUT_X_INDEX = 0;

constexpr uint32_t ATTR_GROUP_INDEX = 0;
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1;

constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16U * 1024 * 1024;
constexpr uint64_t MB_SIZE = 1024UL * 1024;
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;
const char *A_INNER_DEBUG_BARRIER = "DistributeBarrier Tiling Debug";

const int MIN_WORLD_SIZE = 2;
const int MAX_WORLD_SIZE = 384;

static void PrintTilingDataInfo(DistributeBarrierTilingData &tilingData) {
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "worldSize is %u.",
            tilingData.distributeBarrierInfo.worldSize);
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "rankId is %u.",
            tilingData.distributeBarrierInfo.rankId);
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "aivNum is %u.",
            tilingData.distributeBarrierInfo.aivNum);
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "totalUbSize is %lu.",
            tilingData.distributeBarrierInfo.totalUbSize);
}

static bool CheckAndSetAttrs(const gert::TilingContext *context,
                             DistributeBarrierTilingData &tilingData,
                             std::string &group) {
  auto attrs = context->GetAttrs();
  OP_TILING_CHECK(
      attrs == nullptr,
      OPS_LOG_E(A_INNER_DEBUG_BARRIER, "GetAttrs returned nullptr!"),
      return false);

  auto groupPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
  auto worldSizePtr = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE_INDEX);

  // 当前仅对必选属性进行校空
  OP_TILING_CHECK(groupPtr == nullptr,
                  OPS_LOG_E(A_INNER_DEBUG_BARRIER, "groupPtr is null!"),
                  return false);
  OP_TILING_CHECK(worldSizePtr == nullptr,
                  OPS_LOG_E(A_INNER_DEBUG_BARRIER, "worldSizePtr is null!"),
                  return false);

  OP_TILING_CHECK(
      (*worldSizePtr < MIN_WORLD_SIZE) || (*worldSizePtr > MAX_WORLD_SIZE),
      OPS_LOG_E(
          A_INNER_DEBUG_BARRIER,
          "WorldSize is invalid, only support [%d, %d], but got worldSize=%d.",
          MIN_WORLD_SIZE, MAX_WORLD_SIZE, *worldSizePtr),
      return false);

  tilingData.distributeBarrierInfo.worldSize = *worldSizePtr;

  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "group = %s", groupPtr);
  group = string(groupPtr);

  return true;
}

static ge::graphStatus SetWorkSpace(gert::TilingContext *context) {
  size_t *workSpaces = context->GetWorkspaceSizes(1);
  OP_TILING_CHECK(workSpaces == nullptr,
                  OPS_LOG_E(A_INNER_DEBUG_BARRIER, "workSpaces is nullptr."),
                  return ge::GRAPH_FAILED);
  workSpaces[0] = SYSTEM_NEED_WORKSPACE;
  return ge::GRAPH_SUCCESS;
}

static void SetHcommCfg([[maybe_unused]] gert::TilingContext *context,
                        DistributeBarrierTilingData *tiling,
                        const std::string group) {
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "distributeBarrier group = %s",
            group.c_str());
  uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
  std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";

  AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType1,
                                               algConfigAllToAllStr);
  mc2CcTilingConfig.GetTiling(tiling->mc2InitTiling);
  mc2CcTilingConfig.GetTiling(tiling->mc2CcTiling1);
}

ge::graphStatus DistributeBarrierTilingFunc(gert::TilingContext *context) {
  const char *nodeName = context->GetNodeName();
  DistributeBarrierTilingData *tilingData =
      context->GetTilingData<DistributeBarrierTilingData>();
  OP_TILING_CHECK(tilingData == nullptr,
                  OPS_LOG_E(nodeName, "tilingData is nullptr."),
                  return ge::GRAPH_FAILED);
  std::string group = "";

  // Function that get check and set Attrs
  OP_TILING_CHECK(
      !CheckAndSetAttrs(context, *tilingData, group),
      OPS_LOG_E(A_INNER_DEBUG_BARRIER, "Check and set attributes failed!"),
      return ge::GRAPH_FAILED);

  // Set WorkSpace
  OP_TILING_CHECK(
      SetWorkSpace(context) != ge::GRAPH_SUCCESS,
      OPS_LOG_E(A_INNER_DEBUG_BARRIER, "Tiling set workspace failed."),
      return ge::GRAPH_FAILED);

  // Set HcommCfg
  SetHcommCfg(context, tilingData, group);

  // Set TilingKey
  uint64_t tilingKey = INIT_TILINGKEY;
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "cur case tilingKey is %lu", tilingKey);
  context->SetTilingKey(tilingKey);

  // Set blockDim
  uint32_t blockDim = 1U;
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
  uint64_t ubSize = 0UL;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
  context->SetBlockDim(blockDim);
  context->SetScheduleMode(1);  // 设置为batch mode模式，所有核同时启动
  tilingData->distributeBarrierInfo.totalUbSize = ubSize;
  tilingData->distributeBarrierInfo.aivNum = aivNum;
  OPS_LOG_D(A_INNER_DEBUG_BARRIER, "blockDim=%u, aivNum=%u, ubSize=%lu",
            blockDim, aivNum, ubSize);

  PrintTilingDataInfo(*tilingData);
  OPS_LOG_D("DistributeBarrier", "tiling process finished successfully!!!");
  return ge::GRAPH_SUCCESS;
}

struct DistributeBarrierCompileInfo {};
ge::graphStatus TilingParseForDistributeBarrier(
    gert::TilingParseContext *context) {
  const gert::TilingParseContext *const_context = context;
  //避免未使用变量警告
  (void)const_context;
  (void)context;
  return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DistributeBarrier)
    .Tiling(DistributeBarrierTilingFunc)
    .TilingParse<DistributeBarrierCompileInfo>(TilingParseForDistributeBarrier);
}  // end of namespace optiling