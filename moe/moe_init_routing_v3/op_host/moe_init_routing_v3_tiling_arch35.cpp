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
 * \file moe_init_routing_v3_tiling_arch35.cpp
 * \brief
 */
#include "moe_init_routing_v3_tiling.h"
#include "register/op_def_registry.h"

using Ops::Transformer::OpTiling::TilingBaseClass;

namespace optiling {
const static int64_t NUM_TWO = 2LL;
const static int64_t NUM_THREE = 3LL;
const static int64_t NUM_FOUR = 4LL;
const static int64_t MRG_LIST_NUM = 4LL;
const static int64_t SORT32_ALIGN_ELEMENT = 32LL;
const static int64_t ONE_BLOCK_BYTE = 32LL;
const static size_t DIM_ONE = 1ULL;
const static size_t DIM_TWO = 2ULL;
const static int32_t SIZE_16 = 16;
const static int32_t SIZE_31 = 31;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376LL;
const static int64_t ASSIST_NUM = 256LL;
const static int64_t SPLIT_K_THRESHOLD = 512LL;
const static int64_t KV_FACTOR = 2LL;
const static int64_t ONE_CORE_SORT_BUFFER = 6LL;
const static int64_t EXPERT_IDX_MAX = 10240LL;
const static int64_t KV_MODE_EXPERT_IDX_MAX = EXPERT_IDX_MAX / KV_FACTOR;

const static int64_t INPUT_X_INDEX = 0LL;
const static int64_t INPUT_EXPERT_IDX_INDEX = 1LL;
const static int64_t INPUT_SCALE_INDEX = 2LL;
const static int64_t INPUT_OFFSET_INDEX = 3LL;
const static int64_t OUTPUT_EXPANDED_X_INDEX = 0LL;
const static int64_t OUTPUT_EXPANDED_ROW_IDX_INDEX = 1LL;
const static int64_t OUTPUT_EXPERT_TOKENS_COUNT_INDEX = 2LL;
const static int64_t OUTPUT_EXPANDED_SCALE_INDEX = 3LL;
const static int64_t ATTR_ACTIVE_NUM_INDEX = 0LL;
const static int64_t ATTR_EXPERT_CAPACITY_INDEX = 1LL;
const static int64_t ATTR_EXPERT_NUM_INDEX = 2LL;
const static int64_t ATTR_DROP_PAD_MODE_INDEX = 3LL;
const static int64_t ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX = 4LL;
const static int64_t ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX = 5LL;
const static int64_t ATTR_QUANT_MODE_INDEX = 6LL;
const static int64_t ATTR_EXPERT_RANGE_INDEX = 7LL;
const static int64_t ATTR_ROW_IDX_TYPE_INDEX = 8LL;
const static int64_t ATTR_EXPERT_RANGE_DIM = 2LL;
const static int64_t KEY_VALUE_MODE_DIM0_NUM = 2LL;
const static int64_t ACTIVE_NUM_MIN_VALUE = static_cast<int64_t>(-1LL);

const static int64_t GATHER = 0LL;
const static int64_t SCATTER = 1LL;
const static int64_t UN_QUANT = static_cast<int64_t>(-1LL);
const static int64_t STATIC_QUANT = 0LL;
const static int64_t DYNAMIC_QUANT = 1LL;
const static int64_t COUNT = 1LL;
const static int64_t KEY_VALUE = 2;
const static int64_t CUMSUM = 0LL;
const static int64_t DROP_LESS = 0LL;
const static int64_t DROP_PAD = 1LL;
const static int64_t DYNAMIC_QUANT_COLS_BUFFER = 21LL;

const static uint64_t TILINGKEY_BASE = 1000000ULL;
const static uint64_t SORT_CORE_TILINGKEY_BASE = 100000ULL;
const static uint64_t QUANT_MODE_TILINGKEY_BASE = 10000ULL;
const static uint64_t DROP_MODE_TILINGKEY_BASE = 1000ULL;

inline static int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline static int64_t Align(int64_t elementNum, int64_t bytes)
{
    if (bytes == 0) {
        return 0;
    }
    return (elementNum * bytes + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE / bytes;
}

inline static int64_t AlignBytes(int64_t elementNum, int64_t bytes)
{
    return (elementNum * bytes + ONE_BLOCK_BYTE - 1) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
}

class MoeInitRountingV3RegbaseTiling : public TilingBaseClass {
public:
    explicit MoeInitRountingV3RegbaseTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeInitRountingV3RegbaseTiling() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    // 1、获取INPUT/OUTPUT/ATTR信息：DelayedGetShapeAttrsInfo()，延后到DoOpTiling内执行，以便先检查IsCapable()
    ge::graphStatus GetShapeAttrsInfo() override
    {
        OP_LOGI(context_, "TilingContext: %s", context_->GetNodeName());
        return ge::GRAPH_SUCCESS;
    }
    // 2、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 3、判断此Tiling模板是否适配当前SOC
    bool IsCapable() override
    {
        return socVersion_ == platform_ascendc::SocVersion::ASCEND910_95;
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    ge::graphStatus DelayedGetShapeAttrsInfo();
    ge::graphStatus CheckAttr();
    ge::graphStatus CheckOutShape();
    ge::graphStatus CheckInputShape();
    void Tiling4GatherOutCompute();
    void Tiling4SortOutCompute();
    void Tiling4VMSMiddleCompute();
    void Tiling4VBSCompute();
    void Tiling4ExpertTokensCountCompute();
    void ShowTilingData();
    void Tiling4VBSOneCoreCompute(MoeV3VBSComputeTilingData *tilingData);
    void Tiling4VBSMultiCoreCompute(MoeV3VBSComputeTilingData *tilingData);

    int64_t aivNum;
    int64_t sortLoopMaxElement = 0LL;
    int64_t mrgSortListMaxElement = 1024LL;
    int64_t totalLength_ = 0LL;
    int64_t n_ = 0LL;
    int64_t k_ = 0LL;
    int64_t cols_ = 0LL;
    int64_t inuptXDtypeSize_;

    int64_t expertStart_ = 0LL;
    int64_t expertEnd_ = 0LL;
    int64_t isInputScale_ = 0LL;
    int64_t isInputOffset_ = 0LL;

    int64_t sortMode_ = 0LL;
    int64_t rowIdxType_ = 0LL;
    int64_t activeNum_ = static_cast<int64_t>(-1LL);
    int64_t expertCapacity_ = static_cast<int64_t>(-1LL);
    int64_t expertNum_ = static_cast<int64_t>(-1LL);
    int64_t dropPadMode_ = static_cast<int64_t>(-1LL);
    int64_t expertTokensNumType_ = static_cast<int64_t>(-1LL);
    bool expertTokensNumFlag_ = false;
    int64_t quantMode_ = 0LL;

    const gert::StorageShape *xShapePtr_ = nullptr;
    const gert::StorageShape *expertIdxShapePtr_ = nullptr;
    const gert::StorageShape *scaleShapePtr_ = nullptr;
    const gert::StorageShape *offsetShapePtr_ = nullptr;

    const int64_t *activeNumPtr_ = nullptr;
    const int64_t *expertCapacityPtr_ = nullptr;
    const int64_t *expertNumPtr_ = nullptr;
    const int64_t *dropPadModePtr_ = nullptr;
    const int64_t *expertTokensNumTypePtr_ = nullptr;
    const bool *expertTokensNumFlagPtr_ = nullptr;
    const int64_t *quantModePtr_ = nullptr;
    const gert::ContinuousVector *activeExpertRangeListPtr_;
    const int64_t *rowIdxTypePtr_ = nullptr;

    const gert::StorageShape *expandedXShapePtr_ = nullptr;
    const gert::StorageShape *expandedRowIdxShapePtr_ = nullptr;
    const gert::StorageShape *expertTokensCountOrCumsumShapePtr_ = nullptr;
    const gert::StorageShape *expandedScaleShapePtr_ = nullptr;

    const char *opName = "";
    MoeInitRoutingV3TilingData moeInitRoutingV3TilingData;

    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
};

ge::graphStatus MoeInitRountingV3RegbaseTiling::DelayedGetShapeAttrsInfo()
{
    // 获取输入shape
    xShapePtr_ = context_->GetInputShape(INPUT_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShapePtr_);

    expertIdxShapePtr_ = context_->GetInputShape(INPUT_EXPERT_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertIdxShapePtr_);

    // 可选输入scale
    scaleShapePtr_ = context_->GetOptionalInputShape(INPUT_SCALE_INDEX);
    if (scaleShapePtr_ == nullptr) {
        OP_LOGI(context_, "optional input scale is null");
    } else {
        isInputScale_ = 1;
    }
    moeInitRoutingV3TilingData.set_isInputScale(isInputScale_);

    // 可选输入offset
    offsetShapePtr_ = context_->GetOptionalInputShape(INPUT_OFFSET_INDEX);
    if (offsetShapePtr_ == nullptr) {
        OP_LOGI(context_, "optional input offset is null");
    } else {
        isInputOffset_ = 1;
    }
    moeInitRoutingV3TilingData.set_isInputOffset(isInputOffset_);

    // 获取输出shape
    expandedXShapePtr_ = context_->GetOutputShape(OUTPUT_EXPANDED_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedXShapePtr_);
    expandedRowIdxShapePtr_ = context_->GetOutputShape(OUTPUT_EXPANDED_ROW_IDX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedRowIdxShapePtr_);
    expertTokensCountOrCumsumShapePtr_ = context_->GetOutputShape(OUTPUT_EXPERT_TOKENS_COUNT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokensCountOrCumsumShapePtr_);
    expandedScaleShapePtr_ = context_->GetOutputShape(OUTPUT_EXPANDED_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expandedScaleShapePtr_);

    // 获取属性
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    activeNumPtr_ = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVE_NUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activeNumPtr_);
    expertCapacityPtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_CAPACITY_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertCapacityPtr_);
    expertNumPtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_NUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertNumPtr_);
    dropPadModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_DROP_PAD_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dropPadModePtr_);
    expertTokensNumTypePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_EXPERT_TOKEN_NUM_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokensNumTypePtr_);
    expertTokensNumFlagPtr_ = attrs->GetAttrPointer<bool>(ATTR_EXPERT_TOKEN_NUM_FLAG_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, expertTokensNumFlagPtr_);
    quantModePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_QUANT_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantModePtr_);
    activeExpertRangeListPtr_ = attrs->GetAttrPointer<gert::ContinuousVector>(ATTR_EXPERT_RANGE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, activeExpertRangeListPtr_);
    rowIdxTypePtr_ = attrs->GetAttrPointer<int64_t>(ATTR_ROW_IDX_TYPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rowIdxTypePtr_);
    return ge::GRAPH_SUCCESS;
}

void MoeInitRountingV3RegbaseTiling::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus MoeInitRountingV3RegbaseTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr, OP_LOGE(opName, "fail to get platform info"), return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    socVersion_ = ascendcPlatform.GetSocVersion();
    aivNum = ascendcPlatform.GetCoreNumAiv();
    aicoreParams_.blockDim = aivNum;
    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    aicoreParams_.ubSize = ubSizePlatForm;
    moeInitRoutingV3TilingData.set_coreNum(aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3RegbaseTiling::CheckAttr()
{
    int64_t activeExpertRangeListLength = activeExpertRangeListPtr_->GetSize();
    OP_CHECK_IF(activeExpertRangeListLength != ATTR_EXPERT_RANGE_DIM,
                OP_LOGE(context_->GetNodeName(),
                        "The list length of active_expert_range should be %ld, current is %ld.", ATTR_EXPERT_RANGE_DIM,
                        activeExpertRangeListLength),
                return ge::GRAPH_FAILED);

    const int64_t *expertRangeList = reinterpret_cast<const int64_t *>(activeExpertRangeListPtr_->GetData());
    expertStart_ = expertRangeList[0];
    expertEnd_ = expertRangeList[1];
    moeInitRoutingV3TilingData.set_expertStart(expertStart_);
    moeInitRoutingV3TilingData.set_expertEnd(expertEnd_);
    moeInitRoutingV3TilingData.set_actualExpertNum(expertEnd_ - expertStart_);
    OP_LOGI(context_, "Extracted attrs expert_start is: %ld, expert_end is: %ld, actual_expert_num is: %ld",
            expertStart_, expertEnd_, expertEnd_ - expertStart_);

    quantMode_ = *quantModePtr_;
    moeInitRoutingV3TilingData.set_quantMode(quantMode_);
    OP_LOGI(context_, "Attr quant_mode is: %ld", quantMode_);

    rowIdxType_ = *rowIdxTypePtr_;
    moeInitRoutingV3TilingData.set_rowIdxType(rowIdxType_);
    OP_LOGI(context_, "Attr row_idx_type is: %ld", rowIdxType_);

    activeNum_ = *activeNumPtr_;
    OP_CHECK_IF(activeNum_ < ACTIVE_NUM_MIN_VALUE,
                OP_LOGE(context_->GetNodeName(), "Attr active_num should be equal or greater than %ld, current is %ld.",
                        ACTIVE_NUM_MIN_VALUE, activeNum_),
                return ge::GRAPH_FAILED);

    expertTokensNumType_ = *expertTokensNumTypePtr_;
    moeInitRoutingV3TilingData.set_expertTokensNumType(expertTokensNumType_);
    OP_CHECK_IF((expertTokensNumType_ != COUNT) && (expertTokensNumType_ != KEY_VALUE),
                OP_LOGE(context_->GetNodeName(),
                        "Attr expert_tokens_num_type currently supports: %ld or %ld, but got %ld.", COUNT, KEY_VALUE,
                        expertTokensNumType_),
                return ge::GRAPH_FAILED);

    expertNum_ = *expertNumPtr_;
    moeInitRoutingV3TilingData.set_expertNum(expertNum_);
    OP_CHECK_IF(
        expertNum_ <= 0,
        OP_LOGE(context_->GetNodeName(), "Attr expert_num should be greater than 0, current is %ld.", expertNum_),
        return ge::GRAPH_FAILED);
    if (expertTokensNumType_ == KEY_VALUE) {
        OP_CHECK_IF(expertNum_ > KV_MODE_EXPERT_IDX_MAX,
                    OP_LOGE(context_->GetNodeName(),
                            "Attr expert_num should be equal or less than %ld, current is %ld.", KV_MODE_EXPERT_IDX_MAX,
                            expertNum_),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(expertNum_ > EXPERT_IDX_MAX,
                    OP_LOGE(context_->GetNodeName(),
                            "Attr expert_num should be equal or less than %ld, current is %ld.", EXPERT_IDX_MAX,
                            expertNum_),
                    return ge::GRAPH_FAILED);
    }
    OP_LOGI(context_, "Attr expert_num is: %ld", expertNum_);

    dropPadMode_ = *dropPadModePtr_;
    OP_CHECK_IF(dropPadMode_ != DROP_LESS,
                OP_LOGE(context_->GetNodeName(), "Attr drop_pad_mode currently supports %ld, but got %ld.", DROP_LESS,
                        dropPadMode_),
                return ge::GRAPH_FAILED);

    expertTokensNumFlag_ = *expertTokensNumFlagPtr_;
    OP_CHECK_IF(!expertTokensNumFlag_,
                OP_LOGE(context_->GetNodeName(), "Attr expert_tokens_num_flag currently supports True, but got %s",
                        (expertTokensNumFlag_ ? "True" : "False")),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(expertStart_ < 0,
                OP_LOGE(context_->GetNodeName(),
                        "Extracted attr expert_start should be equal or greater than 0, current is %ld.", expertStart_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertStart_ >= expertEnd_,
                OP_LOGE(context_->GetNodeName(),
                        "Extracted attr expert_start should be less than expert_end, current [expert_start, "
                        "expert_end) is [%ld, %ld).",
                        expertStart_, expertEnd_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(expertEnd_ > expertNum_,
                OP_LOGE(context_->GetNodeName(),
                        "Extracted attr expert_end should equal or less than expert_num(%ld), current is %ld.",
                        expertNum_, expertEnd_),
                return ge::GRAPH_FAILED);
    if (expertTokensNumType_ == KEY_VALUE) {
        OP_CHECK_IF(
            expertEnd_ > KV_MODE_EXPERT_IDX_MAX,
            OP_LOGE(context_->GetNodeName(),
                    "Extracted attr expert_end should be equal or less than %ld in KEY_VALUE mode, current is %ld.",
                    KV_MODE_EXPERT_IDX_MAX, expertEnd_),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(expertEnd_ > EXPERT_IDX_MAX,
                    OP_LOGE(context_->GetNodeName(), "expert_end should be equal or less than %ld, current is %ld.",
                            EXPERT_IDX_MAX, expertEnd_),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(quantMode_ != UN_QUANT && quantMode_ != DYNAMIC_QUANT,
                OP_LOGE(context_->GetNodeName(), "Attr quant_mode currently supports %ld or %ld, but got %ld", UN_QUANT,
                        DYNAMIC_QUANT, quantMode_),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(rowIdxType_ != SCATTER && rowIdxType_ != GATHER,
                OP_LOGE(context_->GetNodeName(), "row_idx_type currently supports %ld or %ld, but got %ld.", SCATTER,
                        GATHER, rowIdxType_),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3RegbaseTiling::CheckInputShape()
{
    const gert::Shape xShape = xShapePtr_->GetStorageShape();
    OP_LOGI(context_, "Input x shape is: %s", Ops::Base::ToString(xShape).c_str());
    const gert::Shape expertIdxShape = expertIdxShapePtr_->GetStorageShape();
    OP_LOGI(context_, "Input expert_idx shape is: %s", Ops::Base::ToString(expertIdxShape).c_str());

    // 参数校验
    auto xDimNum = static_cast<int64_t>(xShape.GetDimNum());
    OP_CHECK_IF(xDimNum != 2,
                OP_LOGE(context_->GetNodeName(), "The dim number of x should be 2, current is %ld.", xDimNum),
                return ge::GRAPH_FAILED);
    auto expertIdxDimNum = static_cast<int64_t>(expertIdxShape.GetDimNum());
    OP_CHECK_IF(
        expertIdxDimNum != 2,
        OP_LOGE(context_->GetNodeName(), "The dim number of expert_idx should be 2, current is %ld.", expertIdxDimNum),
        return ge::GRAPH_FAILED);
    auto xShapeDim0 = static_cast<int64_t>(xShape.GetDim(0));
    auto expertIdxDim0 = static_cast<int64_t>(expertIdxShape.GetDim(0));
    OP_CHECK_IF(xShapeDim0 != expertIdxDim0,
                OP_LOGE(context_->GetNodeName(), "Input rows of x(%ld) mismatches with of input expert_idx(%ld).",
                        xShapeDim0, expertIdxDim0),
                return ge::GRAPH_FAILED);

    n_ = expertIdxDim0;
    k_ = expertIdxShape.GetDim(1);
    cols_ = xShape.GetDim(1);
    moeInitRoutingV3TilingData.set_n(n_);
    moeInitRoutingV3TilingData.set_k(k_);
    moeInitRoutingV3TilingData.set_cols(cols_);
    totalLength_ = n_ * k_;
    if (activeNum_ != ACTIVE_NUM_MIN_VALUE) {
        OP_CHECK_IF(activeNum_ != totalLength_,
                    OP_LOGE(context_->GetNodeName(), "Attr active_num should equal to bs*k(%ld), current is %ld.",
                            totalLength_, activeNum_),
                    return ge::GRAPH_FAILED);
    }

    inuptXDtypeSize_ =
        static_cast<int64_t>(ge::GetSizeByDataType(context_->GetInputDesc(INPUT_X_INDEX)->GetDataType()));
    OP_LOGI(context_, "Input x dtype size is: %ld.", inuptXDtypeSize_);

    if (quantMode_ == 0 && scaleShapePtr_ != nullptr) {
        auto scaleShape = scaleShapePtr_->GetStorageShape();
        OP_LOGI(context_, "Input scale shape is: %s", Ops::Base::ToString(scaleShape).c_str());
        auto scaleDimNum = static_cast<int64_t>(scaleShape.GetDimNum());
        OP_CHECK_IF(
            scaleDimNum != 1,
            OP_LOGE(context_->GetNodeName(), "The dim number of scale should be 1, current is %ld", scaleDimNum),
            return ge::GRAPH_FAILED);
        auto scaleDim0 = static_cast<int64_t>(scaleShape.GetDim(0));
        OP_CHECK_IF(
            scaleDim0 != n_,
            OP_LOGE(context_->GetNodeName(), "The first dim of scale should be %ld, current is %ld.", n_, scaleDim0),
            return ge::GRAPH_FAILED);
    }

    if (quantMode_ == DYNAMIC_QUANT && scaleShapePtr_ != nullptr) {
        auto scaleShape = scaleShapePtr_->GetStorageShape();
        OP_LOGI(context_, "Input scale shape is: %s", Ops::Base::ToString(scaleShape).c_str());
        auto scaleDimNum = static_cast<int64_t>(scaleShape.GetDimNum());
        OP_CHECK_IF(
            scaleDimNum != 2,
            OP_LOGE(context_->GetNodeName(), "The dim number of scale should be 2, current is %ld", scaleDimNum),
            return ge::GRAPH_FAILED);
        auto scaleDim0 = static_cast<int64_t>(scaleShape.GetDim(0));
        OP_CHECK_IF(scaleDim0 != (expertEnd_ - expertStart_),
                    OP_LOGE(context_->GetNodeName(), "The first dim of scale should be %ld, current is %ld",
                            (expertEnd_ - expertStart_), scaleDim0),
                    return ge::GRAPH_FAILED);
        auto scaleDim1 = static_cast<int64_t>(scaleShape.GetDim(1));
        OP_CHECK_IF(scaleDim1 != cols_,
                    OP_LOGE(context_->GetNodeName(), "The second dim of scale should be %ld, current is %ld.", cols_,
                            scaleDim1),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3RegbaseTiling::CheckOutShape()
{
    // 获取输入shape
    const auto *outputShape0Ptr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape0Ptr);
    const gert::Shape expandedXShape = outputShape0Ptr->GetStorageShape();
    OP_LOGI(context_, "Output expanded_x shape is: %s", Ops::Base::ToString(expandedXShape).c_str());

    const auto *outputShape1Ptr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape1Ptr);
    const gert::Shape expandedRowIdxShape = outputShape1Ptr->GetStorageShape();
    OP_LOGI(context_, "Output expanded_row_idx shape is: %s", Ops::Base::ToString(expandedRowIdxShape).c_str());

    const auto *outputShape2Ptr = context_->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputShape2Ptr);
    const gert::Shape expertTokensCountOrCumsumShape = outputShape2Ptr->GetStorageShape();
    OP_LOGI(context_, "Output expert_tokens_count_or_cumsum shape is: %s",
            Ops::Base::ToString(expertTokensCountOrCumsumShape).c_str());

    const auto expandedXDimNum = static_cast<int64_t>(expandedXShape.GetDimNum());
    OP_CHECK_IF(
        expandedXDimNum != 2,
        OP_LOGE(context_->GetNodeName(), "The dim number of expanded_x should be 2, current is %ld.", expandedXDimNum),
        return ge::GRAPH_FAILED);

    const auto expandedRowIdxDimNum = static_cast<int64_t>(expandedRowIdxShape.GetDimNum());
    OP_CHECK_IF(expandedRowIdxDimNum != 1,
                OP_LOGE(context_->GetNodeName(), "The dim number of expanded_row_idx should be 1, current is %ld.",
                        expandedRowIdxDimNum),
                return ge::GRAPH_FAILED);

    const auto expandedXDim0 = static_cast<int64_t>(expandedXShape.GetDim(0));
    OP_CHECK_IF(expandedXDim0 != totalLength_,
                OP_LOGE(context_->GetNodeName(), "The first dim of expanded_x should be %ld, current is %ld.",
                        totalLength_, expandedXDim0),
                return ge::GRAPH_FAILED);

    const auto expandedXDim1 = static_cast<int64_t>(expandedXShape.GetDim(1));
    OP_CHECK_IF(expandedXDim1 != cols_,
                OP_LOGE(context_->GetNodeName(), "The second dim of expanded_x should be %ld, current is %ld.", cols_,
                        expandedXDim1),
                return ge::GRAPH_FAILED);

    const auto expandedRowIdxDim0 = static_cast<int64_t>(expandedRowIdxShape.GetDim(0));
    OP_CHECK_IF(expandedRowIdxDim0 != totalLength_,
                OP_LOGE(context_->GetNodeName(), "The first dim of expanded_row_idx should be %ld, current is %ld.",
                        totalLength_, expandedRowIdxDim0),
                return ge::GRAPH_FAILED);

    const auto expertTokensCountOrCumsumDimNum = static_cast<int64_t>(expertTokensCountOrCumsumShape.GetDimNum());
    if (expertTokensNumType_ == KEY_VALUE) {
        OP_CHECK_IF(
            expertTokensCountOrCumsumDimNum != 2,
            OP_LOGE(
                context_->GetNodeName(),
                "The dim number of expert_tokens_count_or_cumsum should be 2 when in KEY_VALUE mode, current is %ld.",
                expertTokensCountOrCumsumDimNum),
            return ge::GRAPH_FAILED);

        const auto expertTokensCountOrCumsumDim0 = static_cast<int64_t>(expertTokensCountOrCumsumShape.GetDim(0));
        OP_CHECK_IF(expertTokensCountOrCumsumDim0 != expertNum_,
                    OP_LOGE(context_->GetNodeName(),
                            "The first dim of expert_tokens_count_or_cumsum should be %ld, current is %ld.", expertNum_,
                            expertTokensCountOrCumsumDim0),
                    return ge::GRAPH_FAILED);

        const auto expertTokensCountOrCumsumDim1 = static_cast<int64_t>(expertTokensCountOrCumsumShape.GetDim(1));
        OP_CHECK_IF(expertTokensCountOrCumsumDim1 != KEY_VALUE_MODE_DIM0_NUM,
                    OP_LOGE(context_->GetNodeName(),
                            "The second dim of expert_tokens_count_or_cumsum should be %ld, current is %ld.",
                            KEY_VALUE_MODE_DIM0_NUM, expertTokensCountOrCumsumDim1),
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(expertTokensCountOrCumsumDimNum != 1,
                    OP_LOGE(context_->GetNodeName(),
                            "The dim number of expert_tokens_count_or_cumsum should be 1 when not in KEY_VALUE mode, "
                            "current is %ld.",
                            expertTokensCountOrCumsumDimNum),
                    return ge::GRAPH_FAILED);

        const auto expertTokensCountOrCumsumDim0 = static_cast<int64_t>(expertTokensCountOrCumsumShape.GetDim(0));
        OP_CHECK_IF(expertTokensCountOrCumsumDim0 != (expertEnd_ - expertStart_),
                    OP_LOGE(context_->GetNodeName(),
                            "The first dim of expert_tokens_count_or_cumsum should be %ld, current is %ld.",
                            (expertEnd_ - expertStart_), expertTokensCountOrCumsumDim0),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

void MoeInitRountingV3RegbaseTiling::ShowTilingData()
{
}
ge::graphStatus MoeInitRountingV3RegbaseTiling::DoOpTiling()
{
    auto ret = DelayedGetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckAttr();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckInputShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = CheckOutShape();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    sortLoopMaxElement =
        aicoreParams_.ubSize / (NUM_FOUR * NUM_TWO * NUM_FOUR) / SORT32_ALIGN_ELEMENT * SORT32_ALIGN_ELEMENT;
    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4ExpertTokensCountCompute();
    Tiling4GatherOutCompute();
    ShowTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3RegbaseTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeInitRountingV3RegbaseTiling::GetTilingKey() const
{
    return static_cast<uint64_t>(TILINGKEY_BASE + static_cast<uint64_t>(sortMode_) * SORT_CORE_TILINGKEY_BASE +
                                 static_cast<uint64_t>(quantMode_ + 1) * QUANT_MODE_TILINGKEY_BASE +
                                 static_cast<uint64_t>(rowIdxType_) * DROP_MODE_TILINGKEY_BASE);
}

ge::graphStatus MoeInitRountingV3RegbaseTiling::GetWorkspaceSize()
{
    // 计算workspace大小
    size_t sortWorkspaceSize = static_cast<size_t>(totalLength_) * sizeof(float) * static_cast<size_t>(NUM_TWO) *
                               static_cast<size_t>(NUM_THREE); // 排序需要的空间
    size_t coreSyncWorkspaceSize = static_cast<size_t>(moeInitRoutingV3TilingData.get_coreNum()) *
                                   SORT32_ALIGN_ELEMENT * NUM_TWO; // 多核同步需要的空间
    size_t scatterWorkspaceSize = static_cast<size_t>(totalLength_) * sizeof(int32_t);
    size_t expertTokensCountWorkspaceSize = static_cast<size_t>((expertEnd_ - expertStart_)) * sizeof(int32_t);
    int64_t expertTokenTotalCountWorkspace = AlignBytes(1, static_cast<int64_t>(sizeof(int32_t)));
    int64_t quantTempWorkspaceSize = aivNum * cols_ * static_cast<int64_t>(sizeof(float));
    workspaceSize_ = sortWorkspaceSize + coreSyncWorkspaceSize + scatterWorkspaceSize + expertTokensCountWorkspaceSize +
                     expertTokenTotalCountWorkspace + SIZE_16 * LENGTH_1024 * LENGTH_1024;
    if (quantMode_ == DYNAMIC_QUANT) {
        workspaceSize_ += quantTempWorkspaceSize;
    }
    OP_LOGI(context_, "Allocate workspaceSize is: %ld", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeInitRountingV3RegbaseTiling::PostTiling()
{
    context_->SetBlockDim(aivNum);
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    moeInitRoutingV3TilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                            context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(moeInitRoutingV3TilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
void MoeInitRountingV3RegbaseTiling::Tiling4VBSOneCoreCompute(MoeV3VBSComputeTilingData *tilingData)
{
    tilingData->set_needCoreNum(1);
    tilingData->set_perCoreElements(totalLength_);
    tilingData->set_perCoreLoops(1);
    tilingData->set_perCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(1);
    tilingData->set_lastCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLastLoopElements(tilingData->get_perCoreElements());
}

void MoeInitRountingV3RegbaseTiling::Tiling4VBSMultiCoreCompute(MoeV3VBSComputeTilingData *tilingData)
{
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, sortLoopMaxElement); // 向上取整
    needCoreNum = static_cast<int64_t>(std::pow(4, CeilLog4(needCoreNum)));     // 用到多核时，核数最多是4^x
    needCoreNum = std::min(needCoreNum, aivNum);                                // 不能超过物理核数

    OP_CHECK_IF(needCoreNum == 0, OP_LOGE(opName, "Variale needCoreNum cannot be 0."), return;);
    int64_t perCoreElements = (needCoreNum == 0) ? 0 : (totalLength_ / needCoreNum);
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength_ - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    tilingData->set_needCoreNum(needCoreNum);
    do {
        tilingData->set_perCoreElements(perCoreElements);
        tilingData->set_perCoreLoops(
            Ops::Base::CeilDiv(tilingData->get_perCoreElements(), sortLoopMaxElement)); // 每个核处理的loop数
        tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));

        tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements() -
                                                (tilingData->get_perCoreLoops() - 1) *
                                                    tilingData->get_perCorePerLoopElements());

        tilingData->set_lastCoreElements(totalLength_ -
                                         (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
        tilingData->set_lastCoreLoops(tilingData->get_perCoreLoops());
        int64_t lastCorePerLoopElements =
            Ops::Base::CeilDiv(Ops::Base::CeilDiv(tilingData->get_lastCoreElements(), tilingData->get_lastCoreLoops()),
                               SORT32_ALIGN_ELEMENT) *
            SORT32_ALIGN_ELEMENT;
        tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
        tilingData->set_lastCoreLastLoopElements(tilingData->get_lastCoreElements() -
                                                 (tilingData->get_lastCoreLoops() - 1) *
                                                     tilingData->get_lastCorePerLoopElements());
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (tilingData->get_lastCoreLastLoopElements() <= 0 && perCoreElements > 0);
    OP_CHECK_IF(tilingData->get_lastCoreLastLoopElements() <= 0, OP_LOGE(opName, "vbs tiling failed"), ;);
}

void MoeInitRountingV3RegbaseTiling::Tiling4VBSCompute()
{
    if (totalLength_ <= sortLoopMaxElement) { // 排序只用到一个核排序
        sortMode_ = 0;
    } else {
        sortMode_ = 1;
    }

    auto tilingData = &moeInitRoutingV3TilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (sortMode_ == 0UL) { // 只用到一个核
        Tiling4VBSOneCoreCompute(tilingData);
        return;
    }
    Tiling4VBSMultiCoreCompute(tilingData);
}

void MoeInitRountingV3RegbaseTiling::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeInitRoutingV3TilingData.vbsComputeParamsOp;
    auto tilingData = &moeInitRoutingV3TilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) { // 队列数小于一次vms则没有中间归并
        tilingData->set_needCoreNum(0);                            // 需要的核数
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM);
    tilingData->set_needCoreNum(needCoreNum); // 需要的核数
}

void MoeInitRountingV3RegbaseTiling::Tiling4SortOutCompute()
{
    auto tilingData = &moeInitRoutingV3TilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}

void MoeInitRountingV3RegbaseTiling::Tiling4ExpertTokensCountCompute()
{
    auto tilingData = &moeInitRoutingV3TilingData.expertTokensCountTilingDataOp;
    int64_t totalElements = moeInitRoutingV3TilingData.get_n() * moeInitRoutingV3TilingData.get_k();
    int64_t perCoreElements = Ops::Base::CeilDiv(totalElements, aivNum);
    int64_t needCoreNum = Ops::Base::CeilDiv(totalElements, perCoreElements);
    int64_t lastCoreElements = totalElements - (needCoreNum - 1) * perCoreElements;
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreElements(perCoreElements);
    tilingData->set_lastCoreElements(lastCoreElements);

    int64_t expertNumElement = (moeInitRoutingV3TilingData.get_expertTokensNumType() != KEY_VALUE) ?
                                   moeInitRoutingV3TilingData.get_actualExpertNum() :
                                   (moeInitRoutingV3TilingData.get_actualExpertNum() + 1) * DIM_TWO;
    int64_t maxElementsPerLoop =
        (static_cast<int64_t>(aicoreParams_.ubSize) -
         Ops::Base::CeilAlign(expertNumElement, ONE_BLOCK_BYTE) *
             (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO + static_cast<int64_t>(sizeof(int64_t))) -
         ONE_BLOCK_BYTE) /
        static_cast<int64_t>(sizeof(int32_t));
    int64_t perCoreLoops = Ops::Base::CeilDiv(perCoreElements, maxElementsPerLoop);
    int64_t perCorePerLoopElements = Ops::Base::CeilDiv(perCoreElements, perCoreLoops);
    int64_t perCoreLastLoopElements = perCoreElements - (perCoreLoops - 1) * perCorePerLoopElements;

    tilingData->set_perCoreLoops(perCoreLoops);
    tilingData->set_perCorePerLoopElements(perCorePerLoopElements);
    tilingData->set_perCoreLastLoopElements(perCoreLastLoopElements);

    int64_t lastCoreLoops = Ops::Base::CeilDiv(lastCoreElements, maxElementsPerLoop);
    int64_t lastCorePerLoopElements = Ops::Base::CeilDiv(lastCoreElements, lastCoreLoops);
    int64_t lastCoreLastLoopElements = lastCoreElements - (lastCoreLoops - 1) * lastCorePerLoopElements;

    tilingData->set_lastCoreLoops(lastCoreLoops);
    tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
    tilingData->set_lastCoreLastLoopElements(lastCoreLastLoopElements);

    OP_LOGI(context_,
            "ExpertTokensCountCompute Tilingdata, needCoreNum is: %ld, perCoreElements is: %ld, lastCoreElements is: "
            "%ld, maxElementsPerLoop is: %ld, perCoreLoops is: %ld, perCorePerLoopElements is: %ld, "
            "perCoreLastLoopElements "
            "is: %ld, lastCoreLoops is: %ld, lastCorePerLoopElements is: %ld, lastCoreLastLoopElements is: %ld",
            needCoreNum, perCoreElements, lastCoreElements, maxElementsPerLoop, perCoreLoops, perCorePerLoopElements,
            perCoreLastLoopElements, lastCoreLoops, lastCorePerLoopElements, lastCoreLastLoopElements);
}

void MoeInitRountingV3RegbaseTiling::Tiling4GatherOutCompute()
{
    auto tilingData = &moeInitRoutingV3TilingData.gatherOutComputeParamsOp;
    int64_t perCoreIndicesElements = Ops::Base::CeilDiv(totalLength_, aivNum);
    if (perCoreIndicesElements <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalLength_, perCoreIndicesElements);
    int64_t lastCoreIndicesElements = totalLength_ - (needCoreNum - 1) * perCoreIndicesElements;

    int64_t perLoopCols = moeInitRoutingV3TilingData.get_cols();
    int64_t colMultiple = NUM_TWO * inuptXDtypeSize_;
    int64_t rowMultiple = NUM_TWO;
    if (quantMode_ == DYNAMIC_QUANT) {
        colMultiple = DYNAMIC_QUANT_COLS_BUFFER;
        rowMultiple = NUM_FOUR;
    }
    int64_t perLoopMaxIndicesElements =
        (static_cast<int64_t>(aicoreParams_.ubSize) - Align(perLoopCols, inuptXDtypeSize_) * colMultiple -
         ONE_BLOCK_BYTE * NUM_TWO) /
        rowMultiple / static_cast<int64_t>(sizeof(int32_t));
    while (perLoopMaxIndicesElements <= 0) {
        perLoopCols = Ops::Base::CeilDiv(perLoopCols, NUM_TWO);
        perLoopMaxIndicesElements = (static_cast<int64_t>(aicoreParams_.ubSize) -
                                     Align(perLoopCols, inuptXDtypeSize_) * colMultiple - ONE_BLOCK_BYTE * NUM_TWO) /
                                    rowMultiple / static_cast<int64_t>(sizeof(int32_t));
        OP_LOGI(context_, "perLoopCols is: %ld, perLoopMaxIndicesElements is: %ld", perLoopCols,
                perLoopMaxIndicesElements);
    }
    int64_t colsLoops = Ops::Base::CeilDiv(moeInitRoutingV3TilingData.get_cols(), perLoopCols);
    int64_t lastLoopCols = moeInitRoutingV3TilingData.get_cols() - (colsLoops - 1) * perLoopCols;
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreIndicesElements(perCoreIndicesElements);
    tilingData->set_lastCoreIndicesElements(lastCoreIndicesElements);
    tilingData->set_colsLoops(colsLoops);
    tilingData->set_perLoopCols(perLoopCols);
    tilingData->set_lastLoopCols(lastLoopCols);

    int64_t perCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, perCoreIndicesElements);
    int64_t perCoreIndicesLoops = Ops::Base::CeilDiv(perCoreIndicesElements, perCorePerLoopIndicesElements);
    int64_t perCoreLastLoopIndicesElements =
        perCoreIndicesElements - (perCoreIndicesLoops - 1) * perCorePerLoopIndicesElements;
    tilingData->set_perCoreIndicesLoops(perCoreIndicesLoops);
    tilingData->set_perCorePerLoopIndicesElements(perCorePerLoopIndicesElements);
    tilingData->set_perCoreLastLoopIndicesElements(perCoreLastLoopIndicesElements);

    int64_t lastCorePerLoopIndicesElements = std::min(perLoopMaxIndicesElements, lastCoreIndicesElements);
    int64_t lastCoreIndicesLoops = Ops::Base::CeilDiv(lastCoreIndicesElements, lastCorePerLoopIndicesElements);
    int64_t lastCoreLastLoopIndicesElements =
        lastCoreIndicesElements - (lastCoreIndicesLoops - 1) * lastCorePerLoopIndicesElements;
    tilingData->set_lastCoreIndicesLoops(lastCoreIndicesLoops);
    tilingData->set_lastCorePerLoopIndicesElements(lastCorePerLoopIndicesElements);
    tilingData->set_lastCoreLastLoopIndicesElements(lastCoreLastLoopIndicesElements);

    OP_LOGI(
        context_,
        "GatherOut Tilingdata, needCoreNum is: %ld, perCoreIndicesElements is: %ld, lastCoreIndicesElements is: %ld, "
        "colsLoops is: %ld, perLoopCols is: %ld, lastLoopCols is: %ld, perCoreIndicesLoops is: %ld, "
        "perCorePerLoopIndicesElements is: %ld, perCoreLastLoopIndicesElements is: %ld, lastCoreIndicesLoops is: "
        "%ld, lastCorePerLoopIndicesElements is: "
        "%ld, lastCoreLastLoopIndicesElements is: %ld",
        needCoreNum, perCoreIndicesElements, lastCoreIndicesElements, colsLoops, perLoopCols, lastLoopCols,
        perCoreIndicesLoops, perCorePerLoopIndicesElements, perCoreLastLoopIndicesElements, lastCoreIndicesLoops,
        lastCorePerLoopIndicesElements, lastCoreLastLoopIndicesElements);
}

REGISTER_TILING_TEMPLATE("MoeInitRoutingV3", MoeInitRountingV3RegbaseTiling,
                         1000); // If 910_95, use this tiling class.
} // namespace optiling
