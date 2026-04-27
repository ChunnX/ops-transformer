/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attention_to_ffn_tiling.cpp
 * \brief
 */

#include <queue>
#include <vector>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <sys/types.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <string>
 
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "../../op_kernel/attention_to_ffn_tiling.h"
#include "../../op_kernel/attention_to_ffn_tiling_key.h"
 
using namespace AscendC;
using namespace ge;
using namespace Mc2Tiling;
 
namespace MC2Tiling {
constexpr char ATTN_TO_FFN_WIN_TYPE_ENV[] = "ASCEND_ATTN_TO_FFN_WIN_TYPE";

constexpr uint32_t INPUT_X_INDEX = 0;                    // 输入x的索引：token数据
constexpr uint32_t INPUT_SESSION_IDS_INDEX = 1;          // 输入sessionId的索引：会话ID
constexpr uint32_t INPUT_MICRO_BATCH_IDS_INDEX = 2;      // 输入microBatchId的索引：微批次ID
constexpr uint32_t INPUT_LAYER_IDS_INDEX = 3;            // 输入layerId的索引：层ID
constexpr uint32_t INPUT_EXPERT_IDS_INDEX = 4;           // 输入expertIds的索引：专家ID
constexpr uint32_t INPUT_EXPERT_RANK_TABLE_INDEX = 5;    // 输入expertRankTable的索引：专家排名表
constexpr uint32_t INPUT_SCALES_INDEX = 6;               // 可选输入scales的索引：量化平滑参数
constexpr uint32_t INPUT_ACTIVE_MASK_INDEX = 7;          // 可选输入activeMask的索引：激活掩码
 
constexpr uint32_t ATTR_GROUP_INDEX = 0;                 // 属性group的索引：通信域名称
constexpr uint32_t ATTR_WORLD_SIZE_INDEX = 1;            // 属性worldSize的索引：通信域大小
constexpr uint32_t ATTR_FFN_TOKEN_INFO_TABLE_SHAPE_INDEX = 2; // 属性ffnTokenInfoTableShape的索引
constexpr uint32_t ATTR_FFN_TOKEN_DATA_SHAPE = 3;        // 属性ffnTokenDataShape的索引
constexpr uint32_t ATTR_ATTN_TOKEN_INFO_TABLE_SHAPE_INDEX = 4; // 属性attnTokenInfoTableShape的索引
constexpr uint32_t ATTR_MOE_EXPERT_NUM_INDEX = 5;        // 属性moeExpertNum的索引
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 6;            // 属性quantMode的索引
constexpr uint32_t ATTR_SYNC_FLAG_INDEX = 7;             // 属性syncFlag的索引
constexpr uint32_t ATTR_FFN_START_RANK_INDEX = 8;        // 属性ffnStartRankId的索引
 
constexpr uint32_t INDEX_TWO = 2;
constexpr uint32_t SYSTEM_NEED_WORKSPACE = 16 * 1024 * 1024; // 系统预留workspace大小：16MB
constexpr uint32_t WORKSPACE_ELEMENT_OFFSET = 128;           // workspace元素偏移对齐粒度：128字节
constexpr uint64_t MB_SIZE = 1024 * 1024;                    // 1MB字节数
constexpr uint32_t OP_TYPE_ALL_TO_ALL = 8;                   // AlltoAll通信操作类型
const char* ATTN_FFN_INNER_DEBUG = "AttentionToFFN Tiling Debug";

constexpr uint32_t INFO_HS_INDEX = 4;                        // ffnTokenDataShape中HS的索引位置
constexpr uint32_t INFO_TABLE_LAST_DIM_NUM_INDEX = 2;        // ffnTokenInfoTableShape中最后一维的索引位置
constexpr uint32_t NO_QUANT_MODE_WIN_SIZE = 2;               // 非量化模式下窗口大小倍数（FP16/BF16占2字节）

constexpr uint32_t ONE_DIM = 1;
constexpr uint32_t TWO_DIMS = 2;
constexpr uint32_t THREE_DIMS = 3;
constexpr uint32_t DIM2 = 2;
constexpr uint32_t DIM3 = 3;
constexpr uint32_t DIM5 = 5;

constexpr uint32_t SYNC_MODE = 1;                            // 同步模式标志
constexpr uint32_t UNQUANT_MODE = 0;                         // 非量化模式
constexpr uint32_t STATIC_QUANT_MODE = 1;                    // 静态量化模式（暂不支持）
constexpr uint32_t DYNAMIC_QUANT_MODE = 2;                   // 动态量化模式
constexpr uint32_t NO_SCALES = 0;                            // 无scales输入
constexpr uint32_t STATIC_SCALES = 1;                        // 静态scales
constexpr uint32_t DYNAMIC_SCALES = 2;                       // 动态scales
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;                 // MoE专家最大数量
constexpr int64_t SHARED_EXPERT_MAX_NUM = 4;                 // 共享专家最大数量
constexpr int64_t BS_MAX = 512;                              // BS最大值
constexpr int64_t H_MIN = 1024;                              // H最小值
constexpr int64_t H_MAX = 8192;                              // H最大值
constexpr int64_t K_MAX = 16;                                // K最大值
constexpr int64_t MAX_WORLD_SIZE = 768L;                     // worldSize最大值（384 * 2）
constexpr int64_t MIN_WORLD_SIZE = 2;                        // worldSize最小值
 
static void PrintTilingDataInfo(AttentionToFFNTilingData &tilingData)
{
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "X is %u.", tilingData.attentionToFFNInfo.X);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "BS is %u.", tilingData.attentionToFFNInfo.BS);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "H is %u.", tilingData.attentionToFFNInfo.H);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "K is %u.", tilingData.attentionToFFNInfo.K);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "L is %u.", tilingData.attentionToFFNInfo.L);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "expertNum is %u.", tilingData.attentionToFFNInfo.expertNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "moeExpertNum is %u.", tilingData.attentionToFFNInfo.moeExpertNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "HS is %u.", tilingData.attentionToFFNInfo.HS);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "expRankTableM is %u.", tilingData.attentionToFFNInfo.expRankTableM);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "microBatchNum is %u.", tilingData.attentionToFFNInfo.microBatchNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "attentionWorkerNum is %u.", tilingData.attentionToFFNInfo.attentionWorkerNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "infoTableLastDimNum is %.", tilingData.attentionToFFNInfo.infoTableLastDimNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "aivNum is %u.", tilingData.attentionToFFNInfo.aivNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "worldSize is %u.", tilingData.attentionToFFNInfo.worldSize);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "syncFlag is %u.", tilingData.attentionToFFNInfo.syncFlag);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "quantMode is %u.", tilingData.attentionToFFNInfo.quantMode);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "isScales is %u.", tilingData.attentionToFFNInfo.isScales);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "windowType is %u.", tilingData.attentionToFFNInfo.windowType);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "totalUbSize is %lu.", tilingData.attentionToFFNInfo.totalUbSize);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "totalWinSize is %lu.", tilingData.attentionToFFNInfo.totalWinSize);
}
 
static bool CheckAndSetShapeAttrs(gert::TilingContext* context, AttentionToFFNTilingData &tilingData)
{
    auto attrs = context->GetAttrs();
    auto worldSizePtr = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE_INDEX);
    auto ffnTokenInfoTableDimNum = attrs->GetListInt(ATTR_FFN_TOKEN_INFO_TABLE_SHAPE_INDEX)->GetSize();
    auto ffnTokenDataDimNum = attrs->GetListInt(ATTR_FFN_TOKEN_DATA_SHAPE)->GetSize();
    auto attnTokenDataDimNum = attrs->GetListInt(ATTR_ATTN_TOKEN_INFO_TABLE_SHAPE_INDEX)->GetSize();
    auto ffnTokenInfoTableShape = attrs->GetListInt(ATTR_FFN_TOKEN_INFO_TABLE_SHAPE_INDEX)->GetData();
    auto ffnTokenDataShape = attrs->GetListInt(ATTR_FFN_TOKEN_DATA_SHAPE)->GetData();
    auto attnTokenDataShape = attrs->GetListInt(ATTR_ATTN_TOKEN_INFO_TABLE_SHAPE_INDEX)->GetData();
    const gert::StorageShape *xShape = context->GetInputShape(INPUT_X_INDEX);
    int32_t bs = xShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(ffnTokenInfoTableDimNum != DIM3, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "the input of ffn_token_info_table_shape dims is not equal 3!"), return false);
    OP_TILING_CHECK(ffnTokenDataDimNum != DIM5, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "the input of ffn_token_data_shape dims is not equal 5!"), return false);
    OP_TILING_CHECK(attnTokenDataDimNum != DIM3, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "the input of attn_token_data_shape dims is not equal 3!"), return false);

    OP_TILING_CHECK(ffnTokenInfoTableShape[0] != ffnTokenDataShape[0], OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "ffn_token_info_table_shape dims0=%ld is not equal ffn_token_data_shape dims0=%ld!",
        ffnTokenInfoTableShape[0], ffnTokenDataShape[0]), return false);
    OP_TILING_CHECK(ffnTokenInfoTableShape[1] != ffnTokenDataShape[1], OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "ffn_token_info_table_shape dims1=%ld is not equal ffn_token_data_shape dims1=%ld!",
        ffnTokenInfoTableShape[1], ffnTokenDataShape[1]), return false);
    OP_TILING_CHECK(ffnTokenDataShape[DIM2] != bs, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "ffn_token_data_shape dims2=%ld is not equal bs=%d!", ffnTokenDataShape[DIM2], bs), return false);
    OP_TILING_CHECK(attnTokenDataShape[0] != ffnTokenInfoTableShape[1], OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "attn_token_data_shape dims0=%ld is not equal ffn_token_data_shape dims1=%ld!",
        attnTokenDataShape[0], ffnTokenInfoTableShape[1]), return false);
    OP_TILING_CHECK(attnTokenDataShape[1] != bs, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "attn_token_data_shape dims1=%ld is not equal xDim1=%d!", attnTokenDataShape[1], bs), return false);
    OP_TILING_CHECK(attnTokenDataShape[DIM2] != ffnTokenDataShape[DIM3], OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "attn_token_data_shape dims2=%ld is not equal ffn_token_data_shape dims3=%ld!",
        attnTokenDataShape[DIM2], ffnTokenDataShape[DIM3]), return false);
    
    uint32_t attentionWorkerNum = ffnTokenDataShape[0];
    OP_TILING_CHECK(attentionWorkerNum <= 0 || attentionWorkerNum >= *worldSizePtr, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "attentionWorkerNum is invalid, only support (0, worldSize = %u), but got attentionWorkerNum=%u.",
        *worldSizePtr, attentionWorkerNum), return false);

    tilingData.attentionToFFNInfo.attentionWorkerNum = attentionWorkerNum;
    tilingData.attentionToFFNInfo.microBatchNum = ffnTokenDataShape[1];
    tilingData.attentionToFFNInfo.HS = ffnTokenDataShape[INFO_HS_INDEX];
    tilingData.attentionToFFNInfo.infoTableLastDimNum = ffnTokenInfoTableShape[INFO_TABLE_LAST_DIM_NUM_INDEX];
    return true;
}

static bool CheckAndSetAttrs(gert::TilingContext* context, AttentionToFFNTilingData &tilingData, std::string &group)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "GetAttrs returned nullptr!"), return false);

    auto groupPtr = attrs->GetAttrPointer<char>(ATTR_GROUP_INDEX);
    auto worldSizePtr = attrs->GetAttrPointer<int>(ATTR_WORLD_SIZE_INDEX);
    auto quantModePtr = attrs->GetAttrPointer<int>(ATTR_QUANT_MODE_INDEX);
    auto syncFlagPtr = attrs->GetAttrPointer<int>(ATTR_SYNC_FLAG_INDEX);
    auto ffnStartRankIdPtr = attrs->GetAttrPointer<int>(ATTR_FFN_START_RANK_INDEX);
    auto moeExpertNumPtr = attrs->GetAttrPointer<int>(ATTR_MOE_EXPERT_NUM_INDEX);
    uint32_t moeExpertNum = *moeExpertNumPtr;
    uint32_t worldSize = *worldSizePtr;

    OP_TILING_CHECK(groupPtr == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "group is nullptr!"), return false);
    OP_TILING_CHECK(worldSizePtr == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "worldSize is nullptr!"), return false);
    OP_TILING_CHECK(moeExpertNumPtr == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "moeExpertNum is nullptr!"), return false);
    OP_TILING_CHECK(quantModePtr == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "quantMode is nullptr!"), return false);
    OP_TILING_CHECK(syncFlagPtr == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "syncFlag is nullptr!"), return false);
    OP_TILING_CHECK(ffnStartRankIdPtr == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "ffnStartRankId is nullptr!"), return false);

    OP_TILING_CHECK((moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM), OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "moeExpertNum is invalid, only support (0, %u], but got moeExpertNum=%u.", MOE_EXPERT_MAX_NUM, moeExpertNum), return false);
    OP_TILING_CHECK((*quantModePtr < static_cast<int64_t>(NO_SCALES)) || (*quantModePtr > static_cast<int64_t>(DYNAMIC_SCALES)),
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "quantMode is invalid, only support [0, %u], but got quantMode=%u.",
        DYNAMIC_SCALES, *quantModePtr), return false);
    OP_TILING_CHECK((*syncFlagPtr != 0) && (*syncFlagPtr != 1), OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "syncFlag is invalid, only support 0 and 1, but got syncFlag=%u.", *syncFlagPtr), return false);
    OP_TILING_CHECK((*ffnStartRankIdPtr >= *worldSizePtr) || (*ffnStartRankIdPtr < 0), OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "ffnStartRankId is invalid, only support (0, %u], but ffn_start_rank_id=%d!", *worldSizePtr, *ffnStartRankIdPtr), return false);
    OP_TILING_CHECK((worldSize < MIN_WORLD_SIZE) || (worldSize > MAX_WORLD_SIZE), OP_LOGE(ATTN_FFN_INNER_DEBUG, 
        "worldSize is invalid, only support [%ld, %ld], but got worldSize=%ld.", MIN_WORLD_SIZE, MAX_WORLD_SIZE, worldSize), return false);

    OP_TILING_CHECK(!CheckAndSetShapeAttrs(context, tilingData),
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "Check and set shape list attrs failed!"), return false);

    tilingData.attentionToFFNInfo.moeExpertNum = static_cast<uint32_t>(moeExpertNum);
    tilingData.attentionToFFNInfo.quantMode = static_cast<uint32_t>(*quantModePtr);
    tilingData.attentionToFFNInfo.syncFlag = static_cast<uint32_t>(*syncFlagPtr);
    tilingData.attentionToFFNInfo.ffnStartRankId = static_cast<uint32_t>(*ffnStartRankIdPtr);
    tilingData.attentionToFFNInfo.worldSize = static_cast<uint32_t>(worldSize);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "group = %s", groupPtr);
    group = string(groupPtr);
 
    return true;
}
 
static bool CheckInputForScales(const gert::StorageShape* scalesShape, const int32_t expertRankTableDim0,
                                const int32_t expertRankTableDim1, const int32_t xDim2)
{
    // 校验数据各维度值
    int32_t scalesDim0 = scalesShape->GetStorageShape().GetDim(0);
    int32_t scalesDim1 = scalesShape->GetStorageShape().GetDim(1);
    int32_t scalesDim2 = scalesShape->GetStorageShape().GetDim(2);

    OP_TILING_CHECK(scalesDim0 != expertRankTableDim0, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "scales dim0 not equal to expertRankTable dim0, cur scalesDim0 is %d, expertRankTableDim0 is %d", scalesDim0, expertRankTableDim0), 
        return false);
    OP_TILING_CHECK(scalesDim1 != expertRankTableDim1, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "scales dim1 not equal to expertRankTable dim1, cur scalesDim1 is %d, expertRankTableDim1 is %d", scalesDim1, expertRankTableDim1), 
        return false);
    OP_TILING_CHECK(scalesDim2 != xDim2, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "scales dim2 not equal to x dim2, cur scalesDim2 is %d, xDim2 is %d", scalesDim2, xDim2), return false);

    return true;
}

static bool CheckInputForActiveMask(const gert::StorageShape* activeMaskShape, const int32_t xDim0, const int32_t xDim1)
{
    int32_t activeMaskDim0 = activeMaskShape->GetStorageShape().GetDim(0);
    int32_t activeMaskDim1 = activeMaskShape->GetStorageShape().GetDim(1);

    OP_TILING_CHECK(activeMaskDim0 != xDim0, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "active_mask dim0 not equal x dim0, cur activeMaskDim0 is %d, xDim0 is %d", activeMaskDim0, xDim0), return false);
    OP_TILING_CHECK(activeMaskDim1 != xDim1, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "active_mask dim1 not equal to x dim1, cur activeMaskDim1 is %d, xDim1 is %d", activeMaskDim1, xDim1), return false);

    return true;
}

static void InitTilingDataByXDim(AttentionToFFNTilingData &tilingData, const int32_t xDim0, const int32_t xDim1, const int32_t xDim2)
{
    tilingData.attentionToFFNInfo.X = xDim0;
    tilingData.attentionToFFNInfo.BS = xDim1;
    tilingData.attentionToFFNInfo.H = xDim2;
}

static void InitTilingDataByExpert(AttentionToFFNTilingData &tilingData, const int32_t expertIdsDim2,
    const int32_t expertRankTableDim0, const int32_t expertRankTableDim1, const int32_t expertRankTableDim2)
{
    tilingData.attentionToFFNInfo.K = expertIdsDim2;
    tilingData.attentionToFFNInfo.L = expertRankTableDim0;
    tilingData.attentionToFFNInfo.expertNum = expertRankTableDim1; // 所有专家总数，包含1个共享+所有moe专家
    tilingData.attentionToFFNInfo.expRankTableM = expertRankTableDim2;
    tilingData.attentionToFFNInfo.sharedExpertNum = expertRankTableDim1 - tilingData.attentionToFFNInfo.moeExpertNum;
}

static bool CheckTensorDataType(gert::TilingContext* context, const bool isScales, const bool isActiveMask)
{
    auto xDesc = context->GetInputDesc(INPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "xDesc is null."), return false);
    OP_TILING_CHECK((xDesc->GetDataType() != ge::DT_BF16) && (xDesc->GetDataType() != ge::DT_FLOAT16),
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "x dataType is invalid, dataType should be bf16 or float16, but is %s.",
        Ops::Base::ToString(xDesc->GetDataType()).c_str()), return false);
    
    auto sessionIdDesc = context->GetInputDesc(INPUT_SESSION_IDS_INDEX);
    OP_TILING_CHECK(sessionIdDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "sessionIdDesc is null."), return false);
    OP_TILING_CHECK(sessionIdDesc->GetDataType() != ge::DT_INT32, OP_LOGE(ATTN_FFN_INNER_DEBUG, 
        "sessionId dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(sessionIdDesc->GetDataType()).c_str()), return false);
    
    auto microBatchIdDesc = context->GetInputDesc(INPUT_MICRO_BATCH_IDS_INDEX);
    OP_TILING_CHECK(microBatchIdDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "microBatchIdDesc is null."), return false);
    OP_TILING_CHECK(microBatchIdDesc->GetDataType() != ge::DT_INT32, OP_LOGE(ATTN_FFN_INNER_DEBUG, 
        "microBatchId dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(microBatchIdDesc->GetDataType()).c_str()), return false);
    
    auto layerIdDesc = context->GetInputDesc(INPUT_LAYER_IDS_INDEX);
    OP_TILING_CHECK(layerIdDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "layerIdDesc is null."), return false);
    OP_TILING_CHECK(layerIdDesc->GetDataType() != ge::DT_INT32, OP_LOGE(ATTN_FFN_INNER_DEBUG, 
        "layerId dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(layerIdDesc->GetDataType()).c_str()), return false);

    auto expertIdDesc = context->GetInputDesc(INPUT_EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertIdDesc is null."), return false);
    OP_TILING_CHECK(expertIdDesc->GetDataType() != ge::DT_INT32, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "expertId dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(expertIdDesc->GetDataType()).c_str()), return false);
    
    auto expertRankTableDesc = context->GetInputDesc(INPUT_EXPERT_RANK_TABLE_INDEX);
    OP_TILING_CHECK(expertRankTableDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertRankTableDesc is null."), return false);
    OP_TILING_CHECK(expertRankTableDesc->GetDataType() != ge::DT_INT32, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "expertRankTable dataType is invalid, dataType should be int32, but is %s.",
        Ops::Base::ToString(expertRankTableDesc->GetDataType()).c_str()), return false);

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(INPUT_SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "scalesDesc is null."), return false);
        OP_TILING_CHECK(scalesDesc->GetDataType() != ge::DT_FLOAT, OP_LOGE(ATTN_FFN_INNER_DEBUG,
            "scales dataType is invalid, dataType should be float, but is %s.",
            Ops::Base::ToString(scalesDesc->GetDataType()).c_str()), return false);
    }

    if (isActiveMask) {
        auto activeMaskDesc = context->GetOptionalInputDesc(INPUT_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(activeMaskDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "activeMaskDesc is null."), return false);
        OP_TILING_CHECK(activeMaskDesc->GetDataType() != ge::DT_BOOL, OP_LOGE(ATTN_FFN_INNER_DEBUG,
            "activeMask dataType is invalid, dataType should be bool, but is %s.",
            Ops::Base::ToString(activeMaskDesc->GetDataType()).c_str()), return false);
    }
 
    return true;
}

static bool CheckTensorFormat(gert::TilingContext* context, const bool isScales, const bool isActiveMask)
{
    auto xDesc = context->GetInputDesc(INPUT_X_INDEX);
    OP_TILING_CHECK(xDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "xDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "x format is invalid."), return false);
    
    auto sessionIdDesc = context->GetInputDesc(INPUT_SESSION_IDS_INDEX);
    OP_TILING_CHECK(sessionIdDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "sessionIdDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(sessionIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "sessionId format is invalid."), return false);
    
    auto microBatchIdDesc = context->GetInputDesc(INPUT_MICRO_BATCH_IDS_INDEX);
    OP_TILING_CHECK(microBatchIdDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "microBatchIdDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(microBatchIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "microBatchId format is invalid."), return false);
    
    auto layerIdDesc = context->GetInputDesc(INPUT_LAYER_IDS_INDEX);
    OP_TILING_CHECK(layerIdDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "layerIdDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(layerIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ,
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "layerId format is invalid."), return false);

    auto expertIdDesc = context->GetInputDesc(INPUT_EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertIdDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertIdDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ, 
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertId format is invalid."), return false);
    
    auto expertRankTableDesc = context->GetInputDesc(INPUT_EXPERT_RANK_TABLE_INDEX);
    OP_TILING_CHECK(expertRankTableDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertRankTableDesc is null."), return false);
    OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(expertRankTableDesc->GetStorageFormat())) == ge::FORMAT_FRACTAL_NZ, 
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertRankTable format is invalid."), return false);

    if (isScales) {
        auto scalesDesc = context->GetOptionalInputDesc(INPUT_SCALES_INDEX);
        OP_TILING_CHECK(scalesDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "scalesDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(scalesDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(ATTN_FFN_INNER_DEBUG, "scales format is invalid."), return false);
    }

    if (isActiveMask) {
        auto activeMaskDesc = context->GetOptionalInputDesc(INPUT_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(activeMaskDesc == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "activeMaskDesc is null."), return false);
        OP_TILING_CHECK(static_cast<ge::Format>(ge::GetPrimaryFormat(activeMaskDesc->GetStorageFormat())) ==
            ge::FORMAT_FRACTAL_NZ, OP_LOGE(ATTN_FFN_INNER_DEBUG, "activeMask format is invalid."), return false);
    }
 
    return true;
}

static bool CheckTensorDim(gert::TilingContext* context, const bool isScales, const bool isActiveMask)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(INPUT_X_INDEX);
    OP_TILING_CHECK(xStorageShape == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "xShape is null."), return false);
    OP_TILING_CHECK(xStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS, OP_LOGE(ATTN_FFN_INNER_DEBUG, 
        "xShape dims must be 3, but current dim num is %lu.", xStorageShape->GetStorageShape().GetDimNum()), return false);

    const gert::StorageShape *sessionIdStorageShape = context->GetInputShape(INPUT_SESSION_IDS_INDEX);
    OP_TILING_CHECK(sessionIdStorageShape == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "sessionIdShape is null."), return false);
    OP_TILING_CHECK(sessionIdStorageShape->GetStorageShape().GetDimNum() != ONE_DIM, OP_LOGE(ATTN_FFN_INNER_DEBUG, 
        "sessionIdShape dim must be 1, but current dim num is %lu.", sessionIdStorageShape->GetStorageShape().GetDimNum()), return false);
    
    const gert::StorageShape *microBatchIdStorageShape = context->GetInputShape(INPUT_MICRO_BATCH_IDS_INDEX);
    OP_TILING_CHECK(microBatchIdStorageShape == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "microBatchIdShape is null."), return false);
    OP_TILING_CHECK(microBatchIdStorageShape->GetStorageShape().GetDimNum() != ONE_DIM, OP_LOGE(ATTN_FFN_INNER_DEBUG, 
        "microBatchIdShape dim must be 1, but current dim num is %lu.", microBatchIdStorageShape->GetStorageShape().GetDimNum()), return false);
    
    const gert::StorageShape *layerIdStorageShape = context->GetInputShape(INPUT_LAYER_IDS_INDEX);
    OP_TILING_CHECK(layerIdStorageShape == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "layerIdShape is null."), return false);
    OP_TILING_CHECK(layerIdStorageShape->GetStorageShape().GetDimNum() != ONE_DIM, OP_LOGE(ATTN_FFN_INNER_DEBUG, 
        "layerIdShape dim must be 1, but current dim num is %lu.", layerIdStorageShape->GetStorageShape().GetDimNum()), return false);
    
    const gert::StorageShape *expertIdStorageShape = context->GetInputShape(INPUT_EXPERT_IDS_INDEX);
    OP_TILING_CHECK(expertIdStorageShape == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertIdShape is null."), return false);
    OP_TILING_CHECK(expertIdStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "expertIdShape dims must be 3, but current dim num is %lu.", expertIdStorageShape->GetStorageShape().GetDimNum()), return false);
    
    const gert::StorageShape *expertRankTableStorageShape = context->GetInputShape(INPUT_EXPERT_RANK_TABLE_INDEX);
    OP_TILING_CHECK(expertRankTableStorageShape == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertRankTableShape is null."), return false);
    OP_TILING_CHECK(expertRankTableStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS,
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertRankTableShape dims must be 3, but current dim num is %lu.",
        expertRankTableStorageShape->GetStorageShape().GetDimNum()), return false);
    
    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(INPUT_SCALES_INDEX);
        OP_TILING_CHECK(scalesStorageShape == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "scalesShape is null."), return false);
        OP_TILING_CHECK(scalesStorageShape->GetStorageShape().GetDimNum() != THREE_DIMS, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "scale shape dims must be 3, but current dim num is %lu.", scalesStorageShape->GetStorageShape().GetDimNum()), return false);
    }

    if (isActiveMask) {
        const gert::StorageShape *activeMaskStorageShape = context->GetOptionalInputShape(INPUT_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(activeMaskStorageShape == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "activeMaskShape is null."), return false);
        OP_TILING_CHECK(activeMaskStorageShape->GetStorageShape().GetDimNum() != TWO_DIMS, OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "activeMask shape dims must be 2, but current dim num is %lu.", activeMaskStorageShape->GetStorageShape().GetDimNum()), return false);
    }

    return true;
}

static bool CheckTensorShapeAndSetTinglingData(gert::TilingContext* context, AttentionToFFNTilingData &tilingData,
                                                const bool isScales, const bool isActiveMask)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(INPUT_X_INDEX);
    const gert::StorageShape *sessionIdStorageShape = context->GetInputShape(INPUT_SESSION_IDS_INDEX);
    const gert::StorageShape *microBatchIdStorageShape = context->GetInputShape(INPUT_MICRO_BATCH_IDS_INDEX);
    const gert::StorageShape *layerIdStorageShape = context->GetInputShape(INPUT_LAYER_IDS_INDEX);
    const gert::StorageShape *expertIdsStorageShape = context->GetInputShape(INPUT_EXPERT_IDS_INDEX);
    const gert::StorageShape *expertRankTableStorageShape = context->GetInputShape(INPUT_EXPERT_RANK_TABLE_INDEX);
    const int32_t xDim0 = xStorageShape->GetStorageShape().GetDim(0);
    const int32_t xDim1 = xStorageShape->GetStorageShape().GetDim(1);
    const int32_t xDim2 = xStorageShape->GetStorageShape().GetDim(DIM2);
    const int32_t sessionIdDim0 = sessionIdStorageShape->GetStorageShape().GetDim(0);
    const int32_t microBatchIdDim0 = microBatchIdStorageShape->GetStorageShape().GetDim(0);
    const int32_t layerIdDim0 = layerIdStorageShape->GetStorageShape().GetDim(0);
    const int32_t expertIdsDim0 = expertIdsStorageShape->GetStorageShape().GetDim(0);
    const int32_t expertIdsDim1 = expertIdsStorageShape->GetStorageShape().GetDim(1);
    const int32_t expertIdsDim2 = expertIdsStorageShape->GetStorageShape().GetDim(DIM2);
    const int32_t expertRankTableDim0 = expertRankTableStorageShape->GetStorageShape().GetDim(0);
    const int32_t expertRankTableDim1 = expertRankTableStorageShape->GetStorageShape().GetDim(1);
    const int32_t expertRankTableDim2 = expertRankTableStorageShape->GetStorageShape().GetDim(DIM2);
    int64_t moeExpertNum = static_cast<int64_t>(tilingData.attentionToFFNInfo.moeExpertNum);
 
    OP_TILING_CHECK(xDim0 != 1, OP_LOGE(ATTN_FFN_INNER_DEBUG, "x's dims0(X) only support 1, cur is %d!", xDim0), return false);
    OP_TILING_CHECK((xDim1 <= 0) || (xDim1 > BS_MAX), OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "xShape dims1(BS) should be in (0, %ld], but got %ld.", BS_MAX, xDim1), return false);
    OP_TILING_CHECK((xDim2 < H_MIN) || (xDim2 > H_MAX), OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "xShape dims2(H) should be in [%ld, %ld], but got %ld.", H_MIN, H_MAX, xDim2), return false);
    OP_TILING_CHECK((xDim0 != sessionIdDim0) || (xDim0 != microBatchIdDim0) || (xDim0 != layerIdDim0) || (xDim0 != expertIdsDim0),
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "sessionId's dims0, microBatchId's dims0, layerId's dims0 and expertIds's dims0 only support 1,"
        "but cur is %d, %d, %d, %d!", sessionIdDim0, microBatchIdDim0, layerIdDim0, expertIdsDim0), return false);
    OP_TILING_CHECK(expertIdsDim1 != xDim1, OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertIdsDim1=%d not equal xDim1=%d!", expertIdsDim1, xDim1), return false);
    OP_TILING_CHECK((expertIdsDim2 <= 0) || (expertIdsDim2 > K_MAX) || (expertIdsDim2 > moeExpertNum), OP_LOGE(ATTN_FFN_INNER_DEBUG,
        "expertIdShape's dim2(k) should be in (0, min(%ld, moeExpertNum = %ld)], but got expertIdShape's dim2=%ld.",
        K_MAX, moeExpertNum, expertIdsDim2), return false);
    OP_TILING_CHECK(expertRankTableDim0 != 1, OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertRankTable dim0(L) only support 1, cur is %d!", expertRankTableDim0), return false);
    OP_TILING_CHECK((expertRankTableDim1 < moeExpertNum) || (expertRankTableDim1 > moeExpertNum + SHARED_EXPERT_MAX_NUM),
                    OP_LOGE(ATTN_FFN_INNER_DEBUG, "expertRankTable dim1 should be in [moeExpertNum = %ld, moeExpertNum + 4 = %ld],"
                    "but cur is %d!", moeExpertNum, moeExpertNum + 4, expertRankTableDim1), return false);

    if (isScales) {
        const gert::StorageShape *scalesStorageShape = context->GetOptionalInputShape(INPUT_SCALES_INDEX);
        OP_TILING_CHECK(!CheckInputForScales(scalesStorageShape, expertRankTableDim0, expertRankTableDim1, xDim2),
            OP_LOGE(ATTN_FFN_INNER_DEBUG, "Check input for scales shape failed!"), return false);
    }

    if (isActiveMask) {
        const gert::StorageShape *activeMaskStorageShape = context->GetOptionalInputShape(INPUT_ACTIVE_MASK_INDEX);
        OP_TILING_CHECK(!CheckInputForActiveMask(activeMaskStorageShape, xDim0, xDim1),
            OP_LOGE(ATTN_FFN_INNER_DEBUG, "Check input for activeMask shape failed!"), return false);
    }
    InitTilingDataByXDim(tilingData, xDim0, xDim1, xDim2);
    InitTilingDataByExpert(tilingData, expertIdsDim2, expertRankTableDim0, expertRankTableDim1, expertRankTableDim2);

    return true;
}

static bool CheckInputAndSetTilingData(gert::TilingContext* context, AttentionToFFNTilingData &tilingData)
{
    const gert::StorageShape *scalesShape = context->GetOptionalInputShape(INPUT_SCALES_INDEX);
    const gert::StorageShape *activeMaskShape = context->GetOptionalInputShape(INPUT_ACTIVE_MASK_INDEX);
    bool isScales = (scalesShape != nullptr);
    bool isActiveMask = (activeMaskShape != nullptr);
 
    // 校验quantMode和scale是否匹配
    uint32_t quantMode = tilingData.attentionToFFNInfo.quantMode;
    OP_TILING_CHECK(quantMode == STATIC_SCALES, OP_LOGE(ATTN_FFN_INNER_DEBUG, "cannot support static quant now."),
        return false);
    OP_TILING_CHECK((isScales && (quantMode == NO_SCALES)) || ((!isScales) && (quantMode == STATIC_SCALES)),
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "quant mode and scales not match, isScales is %d, quantMode is %u.",
        static_cast<int32_t>(isScales), quantMode), return false);
    
    // 校验输入数据dim、format、dataType
    OP_TILING_CHECK(!CheckTensorDim(context, isScales, isActiveMask), OP_LOGE(ATTN_FFN_INNER_DEBUG,
                    "input tensor dim is invalid."), return false);
    OP_TILING_CHECK(!CheckTensorDataType(context, isScales, isActiveMask), OP_LOGE(ATTN_FFN_INNER_DEBUG,
                    "input tensor dataType is invalid."), return false);
    OP_TILING_CHECK(!CheckTensorFormat(context, isScales, isActiveMask), OP_LOGE(ATTN_FFN_INNER_DEBUG,
                    "input tensor format is invalid."), return false);
    
    // 校验输入数据维度值
    OP_TILING_CHECK(!CheckTensorShapeAndSetTinglingData(context, tilingData, isScales, isActiveMask), OP_LOGE(ATTN_FFN_INNER_DEBUG,
                    "input tensor format is invalid."), return false);

    tilingData.attentionToFFNInfo.isScales = isScales;
    tilingData.attentionToFFNInfo.isActiveMask = isActiveMask;
 
    return true;
}
 
// 根据同步标志、量化模式和激活掩码标志计算tilingKey，用于选择对应的kernel实现
static uint64_t CalTilingKey(const uint32_t syncFlag, const uint32_t quantMode, const bool activeMaskFlag)
{
    bool isSync = false;
    bool isQuant = false;
    bool isActiveMask = false;

    // 判断是否为同步通信模式
    if (syncFlag == SYNC_MODE) {
        isSync = true;
    }
    // 判断是否为动态量化模式
    if (quantMode == DYNAMIC_QUANT_MODE) {
        isQuant = true;
    } 
    // 判断是否存在激活掩码
    if (activeMaskFlag) {
        isActiveMask = true;
    }

    // 根据量化、同步、激活掩码三个维度组合生成tilingKey
    const uint64_t tilingKey = GET_TPL_TILING_KEY(isQuant, isSync, isActiveMask, TILINGKEY_TPL_A3);
    return tilingKey;
}
 
// 设置workspace大小，包括系统预留的API workspace和FFN通信所需的额外workspace
static ge::graphStatus SetWorkSpace(gert::TilingContext *context, AttentionToFFNTilingData *tilingData)
{
    size_t *workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr, OP_LOGE(ATTN_FFN_INNER_DEBUG, "workSpaces is nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    // 计算FFN worker数量：总worldSize减去attention worker数量
    uint32_t ffnNum = tilingData->attentionToFFNInfo.worldSize - tilingData->attentionToFFNInfo.attentionWorkerNum;
    // 按WORKSPACE_ELEMENT_OFFSET对齐计算每个AIV核所需的FFN workspace大小
    uint32_t ffnNumAlignSize = (ffnNum * sizeof(int32_t) + WORKSPACE_ELEMENT_OFFSET - 1) / WORKSPACE_ELEMENT_OFFSET * WORKSPACE_ELEMENT_OFFSET;
    // 总workspace = 系统API workspace + 每个AIV核的FFN workspace之和
    workSpaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize() + static_cast<size_t>(ffnNumAlignSize * aivNum);
    return ge::GRAPH_SUCCESS;
}
 
// 计算通信窗口大小，包括token信息表和token数据所需的窗口大小，并与最大可用窗口大小进行比较
static void CalWinSize(AttentionToFFNTilingData &tilingData, const uint32_t quantMode, uint64_t &neededSize, uint64_t &viableWindowSize)
{
    // 获取tiling数据中的关键参数
    uint64_t BS = static_cast<uint64_t>(tilingData.attentionToFFNInfo.BS);
    uint64_t K = static_cast<uint64_t>(tilingData.attentionToFFNInfo.K);
    uint64_t HS = static_cast<uint64_t>(tilingData.attentionToFFNInfo.HS);
    uint64_t microBatchNum = static_cast<uint64_t>(tilingData.attentionToFFNInfo.microBatchNum);
    uint64_t attentionWorkerNum = static_cast<uint64_t>(tilingData.attentionToFFNInfo.attentionWorkerNum);
    uint64_t infoTableLastDimNum = static_cast<uint64_t>(tilingData.attentionToFFNInfo.infoTableLastDimNum);
    uint64_t sharedExpertNum = static_cast<uint64_t>(tilingData.attentionToFFNInfo.sharedExpertNum);
    
    // 计算token信息表所需的窗口大小（int32类型）
    uint64_t tokenInfoNeedWinSize = attentionWorkerNum * microBatchNum * infoTableLastDimNum * sizeof(int32_t);
    uint64_t tokenDataNeedWinSize = 0;
    // 根据量化模式计算token数据所需的窗口大小：量化模式为int8占1字节，非量化模式为FP16/BF16占2字节
    if (quantMode != 0) {
        tokenDataNeedWinSize = attentionWorkerNum * microBatchNum * BS * (K + sharedExpertNum) * HS;
    } else {
        tokenDataNeedWinSize = attentionWorkerNum * microBatchNum * BS * (K + sharedExpertNum) * HS * NO_QUANT_MODE_WIN_SIZE;
    }
    // 实际需要的总窗口大小 = token信息表 + token数据
    uint64_t actualNeedWinSize = tokenInfoNeedWinSize + tokenDataNeedWinSize;
    // 获取系统支持的最大通信窗口大小
    uint64_t maxWindowSize = mc2tiling::Mc2TilingUtils::GetMaxWindowSize();
    // 将大小转换为MB单位，neededSize向上取整加1
    neededSize = actualNeedWinSize / MB_SIZE + 1;
    viableWindowSize = maxWindowSize / MB_SIZE;
    tilingData.attentionToFFNInfo.totalWinSize = maxWindowSize;
 
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "BS is %lu K is %lu HS is %u microBatchNum is %lu attentionWorkerNum is %lu infoTableLastDimNum is %lu.",
            BS, K, HS, microBatchNum, attentionWorkerNum, infoTableLastDimNum);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "actualNeedWinSize is %lu tokenInfoNeedWinSize is %lu tokenDataNeedWinSize is %u neededSize is %lu.",
            actualNeedWinSize, tokenInfoNeedWinSize, tokenDataNeedWinSize, neededSize);
    return;
}
 
// 设置HCCL通信配置，包括AlltoAll通信算法配置，并获取对应的tiling数据
static ge::graphStatus SetHcommCfg(AttentionToFFNTilingData &tilingData, const std::string group)
{
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "attentionToFFN group = %s", group.c_str());
    // 设置通信操作类型为AlltoAll
    uint32_t opType1 = OP_TYPE_ALL_TO_ALL;
    // 配置AlltoAll算法：level0使用fullmesh，level1使用pairwise
    std::string algConfigAllToAllStr = "AlltoAll=level0:fullmesh;level1:pairwise";
    AscendC::Mc2CcTilingConfig mc2CcTilingConfig(group, opType1, algConfigAllToAllStr);

    // 获取HCCL初始化tiling数据和通信tiling数据
    mc2CcTilingConfig.GetTiling(tilingData.mc2InitTiling);
    mc2CcTilingConfig.GetTiling(tilingData.mc2CcTiling1);
 
    return ge::GRAPH_SUCCESS;
}
 
// AttentionToFFN算子的主tiling函数，负责校验输入、计算tiling参数并设置tiling数据
ge::graphStatus AttentionToFFNTilingFunc(gert::TilingContext* context)
{
    // 获取tiling数据结构体指针
    AttentionToFFNTilingData *tilingData = context->GetTilingData<AttentionToFFNTilingData>();
    std::string group = "";
 
    // 第一步：校验并设置算子属性（worldSize、quantMode、syncFlag等）
    OP_TILING_CHECK(!CheckAndSetAttrs(context, *tilingData, group),
                    OP_LOGE(ATTN_FFN_INNER_DEBUG, "Check and set attributes failed!"), return ge::GRAPH_FAILED);
 
    // 第二步：校验输入张量的维度、数据类型、格式，并设置tiling数据
    OP_TILING_CHECK(!CheckInputAndSetTilingData(context, *tilingData),
                    OP_LOGE(ATTN_FFN_INNER_DEBUG, "Check Inputs and Outputs failed!"), return ge::GRAPH_FAILED);
 
    // 第三步：计算通信窗口大小，校验所需窗口大小是否超过可用窗口大小
    uint64_t neededSize = 0;
    uint64_t viableWindowSize = 0;
    uint32_t quantMode = tilingData->attentionToFFNInfo.quantMode;
    bool isScales = tilingData->attentionToFFNInfo.isScales;
    CalWinSize(*tilingData, quantMode, neededSize, viableWindowSize);
    OP_TILING_CHECK(neededSize > viableWindowSize, OP_LOGE(ATTN_FFN_INNER_DEBUG,
                    "needed size:%lu is greater than viable window size:%lu.", neededSize, viableWindowSize),
                    return ge::GRAPH_FAILED);
 
    // 第四步：设置workspace大小
    OP_TILING_CHECK(SetWorkSpace(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(ATTN_FFN_INNER_DEBUG, "Tiling set workspace failed."), return ge::GRAPH_FAILED);
 
    // 第五步：设置HCCL通信配置（AlltoAll通信算法）
    OP_TILING_CHECK(SetHcommCfg(*tilingData, group) != ge::GRAPH_SUCCESS, OP_LOGE(ATTN_FFN_INNER_DEBUG, "setHcommCfg failed."),
        return ge::GRAPH_FAILED);
 
    // 第六步：根据同步标志、量化模式和激活掩码计算tilingKey，用于选择对应的kernel实现
    uint32_t syncFlag = tilingData->attentionToFFNInfo.syncFlag;
    bool activeMaskFlag = tilingData->attentionToFFNInfo.isActiveMask;
    uint64_t tilingKey = CalTilingKey(syncFlag, quantMode, activeMaskFlag);
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "cur case tilingKey is %lu", tilingKey);
    context->SetTilingKey(tilingKey);
 
    // 第七步：获取平台信息，计算并设置核数、UB大小等参数
    uint32_t numBlocks = 1U;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // 根据AIV核数计算调度块维度
    numBlocks = ascendcPlatform.CalcTschBlockDim(aivNum, 0, aivNum);
    context->SetBlockDim(numBlocks);
    tilingData->attentionToFFNInfo.totalUbSize = ubSize;
    tilingData->attentionToFFNInfo.aivNum = aivNum;
    OP_LOGD(ATTN_FFN_INNER_DEBUG, "numBlocks=%u, aivNum=%u, ubSize=%lu", numBlocks, aivNum, ubSize);
 
    PrintTilingDataInfo(*tilingData);
    OP_LOGD("AttentionToFFN", "tiling process finished successfully!!!");
    return ge::GRAPH_SUCCESS;
}
 
struct AttentionToFFNCompileInfo {};
ge::graphStatus TilingParseForAttentionToFFN(gert::TilingParseContext *context) { 
    (void)context;
	return ge::GRAPH_SUCCESS; 
}
 
IMPL_OP_OPTILING(AttentionToFFN)
    .Tiling(AttentionToFFNTilingFunc)
    .TilingParse<AttentionToFFNCompileInfo>(TilingParseForAttentionToFFN);
}  // end of namespace optiling