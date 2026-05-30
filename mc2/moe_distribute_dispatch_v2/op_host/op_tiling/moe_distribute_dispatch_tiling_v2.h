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
 * \file moe_distribute_dispatch_v2_tiling.h
 * \brief
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_TILING_V2
#define MOE_DISTRIBUTE_DISPATCH_TILING_V2

#include <cstdint>
#include <functional>
#include "op_host/op_tiling/mc2_opversion_manager.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "../../common/op_kernel/mc2_moe_context.h"
#include "../../op_kernel/moe_distribute_dispatch_tiling.h"
#include "../../op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "../../op_kernel/moe_distribute_dispatch_v2_tiling_key.h"
#include "mc2_hcom_topo_info.h"
#include "moe_distribute_check_win_size.h"
#include "mc2_exception_dump.h"
#include "moe_distribute_dispatch_tiling_helper.h"

namespace optiling {

// A3 V2 TilingCtx：阶段间中间值载体（原则2 + 原则6）
struct DispatchV2TilingCtx {
    std::string groupEp;
    std::string groupTp;
    bool isSetFullMeshV2 = false;
    bool isLayered = false;
    bool isScales = false;
    bool isActiveMask = false;
    bool hasElasticInfo = false;
    bool isPerformance = false;
    uint32_t quantMode = 0;
    uint32_t bs = 0;
    uint32_t k = 0;
    uint32_t epWorldSize = 0;
    uint32_t tpWorldSize = 0;
    uint32_t sharedExpertNum = 0;
    uint32_t sharedExpertRankNum = 0;
    uint32_t epRankId = 0;
    uint32_t moeExpertNum = 0;
    uint32_t h = 0;
    uint32_t localMoeExpertNum = 1;
    uint32_t a = 0;
    bool shouldHaveScales = false;
    uint32_t aivNum = 0;
    uint64_t ubSize = 0;
    uint32_t blockDim = 0;
};

struct DispatchV2Config {
    uint32_t contextIndex = 0U;             // 0: 根据dispatchV3算子原型标志位初始化context索引
    uint32_t xIndex = 0U;                   // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
    uint32_t expertIdsIndex = 1U;           // 1: 根据dispatchV2算子原型标志位初始化expertIds索引
    uint32_t scalesIndex = 2U;              // 2: 根据dispatchV2算子原型标志位初始化scales索引
    uint32_t xActiveMaskIndex = 3U;         // 3: 根据dispatchV2算子原型标志位初始化xActiveMask索引
    uint32_t expertScalesIndex = 4U;        // 4: 根据dispatchV2算子原型标志位初始化expertScales索引
    uint32_t elasticInfoIndex = 5U;         // 5: 根据dispatchV2算子原型标志位初始化elasticInfo索引
    uint32_t performanceInfoIndex = 6U;     // 6: 根据dispatchV2算子原型标志位初始化performanceInfo索引
    uint32_t attrGroupEpIndex = 0;          // 0: 根据dispatchV2算子原型标志位初始化groupEp索引
    uint32_t attrEpWorldSizeIndex = 1;      // 1: 根据dispatchV2算子原型标志位初始化epWorldSize索引
    uint32_t attrEpRankIdIndex = 2;         // 2: 根据dispatchV2算子原型标志位初始化epRankId索引
    uint32_t attrMoeExpertNumIndex = 3;     // 3: 根据dispatchV2算子原型标志位初始化moeExpertNum索引
    uint32_t attrCclBufferSizeIndex = 3;    // 3: 根据dispatchV3算子原型标志位初始化cclBufferSize索引
    uint32_t attrGroupTpIndex = 4;          // 4: 根据dispatchV2算子原型标志位初始化groupTp索引
    uint32_t attrTpWorldSizeIndex = 5;      // 5: 根据dispatchV2算子原型标志位初始化tpWorldSize索引
    uint32_t attrTpRankIdIndex = 6;         // 6: 根据dispatchV2算子原型标志位初始化tpRankId索引
    uint32_t attrExpertSharedTypeIndex = 7; // 7: 根据dispatchV2算子原型标志位初始化expertSharedType索引
    uint32_t attrSharedExpertNumIndex = 8;  // 8: 根据dispatchV2算子原型标志位初始化sharedExpertNum索引
    uint32_t attrSharedExpertRankNumIndex = 9; // 9: 根据dispatchV2算子原型标志位初始化sharedExpertRankNum索引
    uint32_t attrQuantModeIndex = 10;          // 10: 根据dispatchV2算子原型标志位初始化quantMode索引
    uint32_t attrGlobalBsIndex = 11;           // 11: 根据dispatchV2算子原型标志位初始化globalBs索引
    uint32_t attrExpertTokenNumsTypeIndex = 12; // 12: 根据dispatchV2算子原型标志位初始化expertTokenNumType索引
    uint32_t attrCommAlgIndex = 13;             // 13: 根据dispatchV2算子原型标志位初始化commAlg索引
    uint32_t attrZeroExpertNumIndex = 14;  // 14: 根据dispatchV2算子原型标志位初始化zeroExpertNumIndex索引
    uint32_t attrCopyExpertNumIndex = 15;  // 15: 根据dispatchV2算子原型标志位初始化copyExpertNumIndex索引
    uint32_t attrConstExpertNumIndex = 16; // 16: 根据dispatchV2算子原型标志位初始化constExpertNumIndex索引
    bool isMc2Context = false;
    uint32_t maxQuantMode = 0;  // A3: PERTOKEN_DYNAMIC_QUANT, A5: MX_QUANT
    uint32_t tilingKeyArch = 0; // TILINGKEY_TPL_A3 or TILINGKEY_TPL_A5
};

// DispatchV2Config 默认初始化（公共索引映射）
static inline void InitDispatchV2ConfigDefault(DispatchV2Config &config)
{
    config.xIndex = 0U;
    config.expertIdsIndex = 1U;
    config.scalesIndex = 2U;
    config.xActiveMaskIndex = 3U;
    config.expertScalesIndex = 4U;
    config.elasticInfoIndex = 5U;
    config.performanceInfoIndex = 6U;
    config.attrGroupEpIndex = 0;
    config.attrEpWorldSizeIndex = 1;
    config.attrEpRankIdIndex = 2;
    config.attrMoeExpertNumIndex = 3;
    config.attrGroupTpIndex = 4;
    config.attrTpWorldSizeIndex = 5;
    config.attrTpRankIdIndex = 6;
    config.attrExpertSharedTypeIndex = 7;
    config.attrSharedExpertNumIndex = 8;
    config.attrSharedExpertRankNumIndex = 9;
    config.attrQuantModeIndex = 10;
    config.attrGlobalBsIndex = 11;
    config.attrExpertTokenNumsTypeIndex = 12;
    config.attrCommAlgIndex = 13;
    config.attrZeroExpertNumIndex = 14;
    config.attrCopyExpertNumIndex = 15;
    config.attrConstExpertNumIndex = 16;
    config.isMc2Context = false;
}
using namespace Ops::Transformer::OpTiling;

static inline void PrintTilingDataInfo(const char *nodeName, MoeDistributeDispatchV2TilingData &tilingData)
{
    auto &info = tilingData.moeDistributeDispatchV2Info;
    OP_LOGD(nodeName, "epWorldSize is %u.", info.epWorldSize);
    OP_LOGD(nodeName, "tpWorldSize is %u.", info.tpWorldSize);
    OP_LOGD(nodeName, "epRankId is %u.", info.epRankId);
    OP_LOGD(nodeName, "tpRankId is %u.", info.tpRankId);
    OP_LOGD(nodeName, "expertShardType is %u.", info.expertShardType);
    OP_LOGD(nodeName, "sharedExpertNum is %u.", info.sharedExpertNum);
    OP_LOGD(nodeName, "sharedExpertRankNum is %u.", info.sharedExpertRankNum);
    OP_LOGD(nodeName, "moeExpertNum is %u.", info.moeExpertNum);
    OP_LOGD(nodeName, "quantMode is %u.", info.quantMode);
    OP_LOGD(nodeName, "globalBs is %u.", info.globalBs);
    OP_LOGD(nodeName, "bs is %u.", info.bs);
    OP_LOGD(nodeName, "k is %u.", info.k);
    OP_LOGD(nodeName, "h is %u.", info.h);
    OP_LOGD(nodeName, "aivNum is %u.", info.aivNum);
    OP_LOGD(nodeName, "totalUbSize is %lu.", info.totalUbSize);
    OP_LOGD(nodeName, "totalWinSizeEP is %lu.", info.totalWinSizeEp);
    OP_LOGD(nodeName, "totalWinSizeTP is %lu.", info.totalWinSizeTp);
    OP_LOGD(nodeName, "hasElastic is %d.", info.hasElasticInfo);
    OP_LOGD(nodeName, "isPerformance is %d.", info.isPerformance);
    OP_LOGD(nodeName, "zeroComputeExpertNum is %d.", info.zeroComputeExpertNum);
    OP_LOGD(nodeName, "expertTokenNumsType is %u.", info.expertTokenNumsType);
    OP_LOGD(nodeName, "scalesCol is %lu.", info.scalesCol);
    OP_LOGD(nodeName, "scalesRow is %lu.", info.scalesRow);
    OP_LOGD(nodeName, "scalesTypeSize is %u.", info.scalesTypeSize);
    OP_LOGD(nodeName, "scalesCount is %lu.", info.scalesCount);
}

// 架构差异回调类型（原则5：通过参数传递，而非继承）
using CheckTensorDataTypeFn =
    std::function<bool(const gert::TilingContext *, const char *, DispatchV2TilingCtx &, DispatchV2Config &)>;
using CalTilingKeyFn = std::function<uint64_t(const DispatchV2TilingCtx &)>;
using CheckQuantModeMatchScalesFn =
    std::function<ge::graphStatus(gert::TilingContext *, const char *, DispatchV2TilingCtx &, DispatchV2Config &)>;
using CheckCommAlgPtrFn = std::function<ge::graphStatus(const char *, const char *)>;
using CheckQuantModePtrFn = std::function<ge::graphStatus(const int64_t *, const char *)>;

// 公共验证/解析函数：提取为自由函数（原则2）
ge::graphStatus TilingCheckMoeDistributeDispatch(gert::TilingContext *context, const char *nodeName,
                                                 DispatchV2TilingCtx &tc, DispatchV2Config &config,
                                                 const CheckTensorDataTypeFn &checkTensorDataType);
bool CheckCommomOtherInputTensorDataType(const gert::TilingContext *context, const char *nodeName,
                                         DispatchV2TilingCtx &tc, DispatchV2Config &config);
bool CheckCommomOutputTensorDataType(const gert::TilingContext *context, const char *nodeName);
ge::graphStatus CheckOtherAttrParams(const gert::TilingContext *context, const char *nodeName, DispatchV2TilingCtx &tc,
                                     DispatchV2Config &config, const CheckQuantModePtrFn &checkQuantModePtr,
                                     const CheckCommAlgPtrFn &checkCommAlgPtr);
ge::graphStatus CheckCommAttrParams(const gert::TilingContext *context, const char *nodeName, DispatchV2TilingCtx &tc,
                                    DispatchV2Config &config, const CheckQuantModePtrFn &checkQuantModePtr,
                                    const CheckCommAlgPtrFn &checkCommAlgPtr);
ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext *context, const char *nodeName,
                                        MoeDistributeDispatchV2TilingData &tilingData, DispatchV2TilingCtx &tc,
                                        DispatchV2Config &config, const CheckQuantModePtrFn &checkQuantModePtr,
                                        const CheckCommAlgPtrFn &checkCommAlgPtr);
ge::graphStatus MoeDistributeDispatchV2TilingFuncImplPublicV2(
    gert::TilingContext *context, DispatchV2Config &config, DispatchV2TilingCtx &tc,
    const CheckQuantModeMatchScalesFn &checkQuantModeMatchScales, const CheckTensorDataTypeFn &checkTensorDataType,
    const CheckQuantModePtrFn &checkQuantModePtr, const CheckCommAlgPtrFn &checkCommAlgPtr);

ge::graphStatus MoeDistributeDispatchA3TilingFuncImplPublic(
    gert::TilingContext *context, DispatchV2Config &config,
    const CheckQuantModeMatchScalesFn &checkQuantModeMatchScales, const CalTilingKeyFn &calTilingKey,
    const CheckTensorDataTypeFn &checkTensorDataType, const CheckQuantModePtrFn &checkQuantModePtr,
    const CheckCommAlgPtrFn &checkCommAlgPtr);

// A2/V3 保留基类（A2 仍继承，V3 仍通过 A2A3 调用旧入口）
class MoeDistributeDispatchV2TilingFuncBase {
public:
    ge::graphStatus MoeDistributeDispatchA3TilingFuncImplPublic(gert::TilingContext *context, DispatchV2Config &config);
    ge::graphStatus TilingCheckMoeDistributeDispatch(gert::TilingContext *context, const char *nodeName,
                                                     const bool isActiveMask, const bool isScales,
                                                     const bool hasElasticInfo, const bool isPerformance,
                                                     const uint32_t quantMode, const bool isLayered,
                                                     DispatchV2Config &config);
    bool CheckCommomOtherInputTensorDataType(const gert::TilingContext *context, const char *nodeName,
                                             const bool isActiveMask, const bool hasElasticInfo,
                                             const bool isPerformance, DispatchV2Config &config);
    bool CheckCommomOutputTensorDataType(const gert::TilingContext *context, const char *nodeName);
    ge::graphStatus CheckOtherAttrParams(const gert::TilingContext *context, const char *nodeName,
                                         bool &isSetFullMeshV2, DispatchV2Config &config);
    ge::graphStatus CheckCommAttrParams(const gert::TilingContext *context, const char *nodeName, std::string &groupTp,
                                        bool &isSetFullMeshV2, bool &isLayered, DispatchV2Config &config);
    ge::graphStatus GetAttrAndSetTilingData(const gert::TilingContext *context, const char *nodeName,
                                            MoeDistributeDispatchV2TilingData &tilingData, std::string &groupEp,
                                            std::string &groupTp, bool &isSetFullMeshV2, bool &isLayered,
                                            DispatchV2Config &config);

    virtual ge::graphStatus MoeDistributeDispatchV2TilingFunc(gert::TilingContext *context) = 0;
    virtual ge::graphStatus MoeDistributeDispatchTilingFuncImpl(gert::TilingContext *context) = 0;
    virtual bool CheckTensorDataType(const gert::TilingContext *context, const char *nodeName, const bool isScales,
                                     const uint32_t quantMode, const bool isActiveMask, const bool hasElasticInfo,
                                     const bool isPerformance, DispatchV2Config &config) = 0;
    virtual uint64_t CalTilingKey(const bool isScales, const uint32_t quantMode, const uint32_t tpWorldSize,
                                  const bool isSetFullMeshV2, bool isLayered) = 0;
    virtual ge::graphStatus CheckQuantModeMatchScales(gert::TilingContext *context, const char *nodeName, bool isScales,
                                                      uint32_t quantMode, DispatchV2Config &config) = 0;
    virtual ge::graphStatus CheckCommAlgPtr(const char *commAlgPtr, const char *nodeName) = 0;
    virtual ge::graphStatus CheckQuantModePtr(const int64_t *quantModePtr, const char *nodeName) = 0;
};

} // namespace optiling

#endif