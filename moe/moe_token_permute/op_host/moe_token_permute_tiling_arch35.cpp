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
 * \file moe_token_permute_tiling_arch35.cpp
 * \brief ascend950(arch35) 原生 tiling：排序(两阶段 argsort 逆映射)+gather，复用 moe_init_routing_v2 arch35 kernel。
 */
#include "moe_token_permute_tiling.h"
#include "op_host/tiling_util.h"

namespace {
const static int64_t NUM_TWO = 2;
const static int64_t NUM_THREE = 3;
const static int64_t NUM_FOUR = 4;
const static int64_t MRG_LIST_NUM = 4;
const static int64_t SORT32_ALIGN_ELEMENT = 32;
const static int64_t ONE_BLOCK_BYTE = 32;
const static size_t DIM_ONE = 1;
const static size_t DIM_TWO = 2;
const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;
const static int64_t MAX_COLS_ONE_LOOP = 16376;
const static int64_t SCATTER_MODE_MAX_K = 6;
const static int64_t SCATTER_FULL_MAX_K = 4;
const static int64_t SCATTER_FULL_MAX_ROW_BYTES = 512;
const static int64_t ASSIST_NUM = 256;
const static int64_t MAX_INDICES_NUM = 512;
const static int64_t MAX_INT8_INDICES_NUM = 10240;
const static int64_t SORT_LIMIT_LENGTH = 16777215;
const static int64_t SIMT_UB_SIZE_BYTE = 40960; // arch35 SIMT DCache 预留
const static int64_t MRG_SORT_LIST_MAX_ELEMENT = 2048;
const static uint64_t TILING_KEY_SORT_ONE_CORE = 10001UL;
const static uint64_t TILING_KEY_SORT_MULTI_CORE = 10002UL;

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

inline int64_t GetPerOrLastValue(int64_t x, int64_t y)
{
    if (x <= 0 || y <= 0) {
        return 0;
    }
    return (x - 1) % y + 1;
}

inline int64_t CeilLog4(int64_t x)
{
    return static_cast<int64_t>(std::ceil(std::log(x) / std::log(NUM_FOUR)));
}

inline int64_t UpAlign(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}
} // namespace

namespace optiling {
class MoeTokenPermuteArch35TilingClass : public Ops::Transformer::OpTiling::TilingBaseClass
{
public:
    explicit MoeTokenPermuteArch35TilingClass(gert::TilingContext* context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~MoeTokenPermuteArch35TilingClass() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return Ops::Transformer::OpTiling::IsRegbaseSocVersion(context_);
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void Reset();

private:
    ge::graphStatus CheckOutShape();
    void Tiling4VBSCompute();
    void Tiling4VBSOneCoreCompute(PermuteVBSComputeTilingData* tilingData);
    void Tiling4VBSMultiCoreCompute(PermuteVBSComputeTilingData* tilingData);
    void Tiling4VMSMiddleCompute();
    void Tiling4SortOutCompute();
    void Tiling4SrcToDstCompute();
    void Tiling4GatherOutCompute();
    bool CalculateScatterOutCoreAllocation(int64_t actualActivateRows,
                                           PermuteGatherOutComputeTilingData* tilingData);
    void CalculateScatterOutBufferParams(int64_t ubRows, int64_t cols,
                                         PermuteGatherOutComputeTilingData* tilingData);
    void SetGatherOutLoopParams(int64_t perCoreRows, int64_t lastCoreRows, int64_t rowSize, int64_t colSize);

    int64_t aivNum = 0;
    int64_t numOutTokens = 0;
    int64_t activateNum = 0;
    int64_t totalLength = 0;
    int64_t tokenBtypeSize = 0;
    int64_t indicesBtypeSize = 0;
    int64_t sortLoopMaxElement = 0;
    int64_t mrgSortListMaxElement = MRG_SORT_LIST_MAX_ELEMENT;
    bool paddedMode = false;
    const char* opName = nullptr;
    MoeTokenPermuteTilingData moeTokenPermuteTilingData;
};

void MoeTokenPermuteArch35TilingClass::Reset()
{
    opName = nullptr;
    return;
}

ge::graphStatus MoeTokenPermuteArch35TilingClass::GetPlatformInfo()
{
    opName = context_->GetNodeName();
    auto compileInfoPtr = reinterpret_cast<const MoeTokenPermuteCompileInfo*>(context_->GetCompileInfo());
    OP_CHECK_IF(
        compileInfoPtr == nullptr, OP_LOGE(opName, "fail to get compile info"), return ge::GRAPH_FAILED);
    aivNum = compileInfoPtr->aivNum;
    aicoreParams_.numBlocks = aivNum;
    aicoreParams_.ubSize = compileInfoPtr->ubSize - SIMT_UB_SIZE_BYTE;
    moeTokenPermuteTilingData.set_coreNum(aivNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteArch35TilingClass::CheckOutShape()
{
    const auto tokenOutput = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, tokenOutput);
    const gert::Shape tokensShape = tokenOutput->GetStorageShape();
    const auto indicesOutput = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesOutput);
    const gert::Shape IndicesShape = indicesOutput->GetStorageShape();

    size_t tokensDimNnum = tokensShape.GetDimNum();
    if (tokensDimNnum < DIM_TWO) {
        OP_LOGE(
            context_->GetNodeName(), "The dim number of Output permute_tokens should be greater than 1 but got [%lu].",
            tokensDimNnum);
        return ge::GRAPH_FAILED;
    }

    int64_t cols = 1;
    for (size_t i = 1; i < tokensDimNnum; i++) {
        cols *= tokensShape.GetDim(i);
    }

    size_t IndicesDimNnum = IndicesShape.GetDimNum();
    if (IndicesDimNnum != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "The dim number of Output sort_indices should be 1.");
        return ge::GRAPH_FAILED;
    }
    if (IndicesShape.GetDim(0) != totalLength) {
        OP_LOGE(context_->GetNodeName(), "The size of Output sort_indices should be [%ld].", totalLength);
        return ge::GRAPH_FAILED;
    }
    if (cols != moeTokenPermuteTilingData.get_cols()) {
        OP_LOGE(
            context_->GetNodeName(), "The hidden_size of output permuteTokens should be %ld but got %ld.",
            moeTokenPermuteTilingData.get_cols(), cols);
        return ge::GRAPH_FAILED;
    }
    if (tokensShape.GetDim(0) != numOutTokens) {
        OP_LOGE(
            context_->GetNodeName(), "The dim 0 of output permuteTokens should be %ld but got %ld.", numOutTokens,
            tokensShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteArch35TilingClass::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    const gert::Shape tokensShape = context_->GetInputShape(0)->GetStorageShape();
    const gert::Shape IndicesShape = context_->GetInputShape(1)->GetStorageShape();
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    int64_t id = 0;
    const int64_t* numOutTokensPtr = attrs->GetAttrPointer<int64_t>(id++);
    const bool* paddedModePtr = attrs->GetAttrPointer<bool>(id);
    numOutTokens = *numOutTokensPtr;
    paddedMode = *paddedModePtr;

    if (paddedMode == true) {
        OP_LOGE(context_->GetNodeName(), "Currently only support padded_mode is false");
        return ge::GRAPH_FAILED;
    }

    size_t TokensDimNnum = tokensShape.GetDimNum();
    if (TokensDimNnum < DIM_TWO) {
        OP_LOGE(
            context_->GetNodeName(), "The dim number of input tokens should be greater than 1 but got [%lu].",
            TokensDimNnum);
        return ge::GRAPH_FAILED;
    }
    int64_t cols = 1;
    for (size_t i = 1; i < TokensDimNnum; i++) {
        cols *= tokensShape.GetDim(i);
    }

    size_t indicesDimNum = IndicesShape.GetDimNum();
    if (indicesDimNum != DIM_TWO && indicesDimNum != DIM_ONE) {
        OP_LOGE(context_->GetNodeName(), "The dim number of indices should be 2 or 1 but got [%lu].", indicesDimNum);
        return ge::GRAPH_FAILED;
    }
    if (tokensShape.GetDim(0) != IndicesShape.GetDim(0)) {
        OP_LOGE(
            context_->GetNodeName(), "Input token's dim 0 [%ld] should be same with indices' dim 0 [%ld].",
            tokensShape.GetDim(0), IndicesShape.GetDim(0));
        return ge::GRAPH_FAILED;
    }

    tokenBtypeSize = ge::GetSizeByDataType(context_->GetInputDesc(0)->GetDataType());
    indicesBtypeSize = ge::GetSizeByDataType(context_->GetInputDesc(1)->GetDataType());

    moeTokenPermuteTilingData.set_cols(cols);
    auto tokenOneBlockNum = ONE_BLOCK_BYTE / tokenBtypeSize;
    auto colsAlign = CeilDiv(cols, tokenOneBlockNum) * tokenOneBlockNum;
    moeTokenPermuteTilingData.set_n(IndicesShape.GetDim(0));
    moeTokenPermuteTilingData.set_colsAlign(colsAlign);

    int64_t topK = (indicesDimNum == 1) ? 1 : IndicesShape.GetDim(1);
    if (topK > MAX_INDICES_NUM) {
        OP_LOGE(
            context_->GetNodeName(), "Input indices's dim 1 [%ld] should not large than max topK[%ld].", topK,
            MAX_INDICES_NUM);
        return ge::GRAPH_FAILED;
    }
    moeTokenPermuteTilingData.set_topK(topK);
    int64_t n = moeTokenPermuteTilingData.get_n();
    if (topK > 0 && n >= CeilDiv(SORT_LIMIT_LENGTH, topK)) {
        OP_LOGE(
            context_->GetNodeName(), "The elements num of indices should be less than [%ld].", SORT_LIMIT_LENGTH);
        return ge::GRAPH_FAILED;
    }
    totalLength = moeTokenPermuteTilingData.get_n() * moeTokenPermuteTilingData.get_topK();
    bool isInt8 = context_->GetInputDesc(0)->GetDataType() == ge::DataType::DT_INT8;
    if (isInt8 && totalLength > MAX_INT8_INDICES_NUM) {
        OP_LOGE(
            context_->GetNodeName(), "The elements num of indices [%ld] should not exceed [%ld] for INT8 tokens.",
            totalLength, MAX_INT8_INDICES_NUM);
        return ge::GRAPH_FAILED;
    }
    if (isInt8 && numOutTokens > MAX_INT8_INDICES_NUM) {
        OP_LOGE(
            context_->GetNodeName(), "num_out_tokens [%ld] should not exceed [%ld] for INT8 tokens.", numOutTokens,
            MAX_INT8_INDICES_NUM);
        return ge::GRAPH_FAILED;
    }

    if (numOutTokens == 0) {
        numOutTokens = totalLength;
    } else if (numOutTokens < 0) {
        numOutTokens = (numOutTokens <= -totalLength) ? 0 : numOutTokens + totalLength;
    } else {
        numOutTokens = std::min(numOutTokens, totalLength);
    }
    activateNum = numOutTokens;

    // permute 无量化/专家/丢弃语义，相关标量恒 0（arch35 复用的 v2 kernel 据此走 dropless 分支）
    moeTokenPermuteTilingData.set_expertCapacity(0);
    moeTokenPermuteTilingData.set_expertNum(0);
    moeTokenPermuteTilingData.set_dropPadMode(0);
    moeTokenPermuteTilingData.set_expertTokensCountOrCumsumFlag(0);
    moeTokenPermuteTilingData.set_expertTokensBeforeCapacityFlag(0);

    return CheckOutShape();
}

ge::graphStatus MoeTokenPermuteArch35TilingClass::DoOpTiling()
{
    bool isIdxLong = context_->GetInputDesc(1)->GetDataType() == ge::DataType::DT_INT64;
    sortLoopMaxElement = static_cast<int64_t>(aicoreParams_.ubSize) /
                         (static_cast<int64_t>(sizeof(int32_t)) * NUM_TWO * NUM_FOUR + isIdxLong) / SORT32_ALIGN_ELEMENT *
                         SORT32_ALIGN_ELEMENT;
    Tiling4VBSCompute();
    Tiling4VMSMiddleCompute();
    Tiling4SortOutCompute();
    Tiling4SrcToDstCompute();
    Tiling4GatherOutCompute();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteArch35TilingClass::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

uint64_t MoeTokenPermuteArch35TilingClass::GetTilingKey() const
{
    return (totalLength <= sortLoopMaxElement) ? TILING_KEY_SORT_ONE_CORE : TILING_KEY_SORT_MULTI_CORE;
}

ge::graphStatus MoeTokenPermuteArch35TilingClass::GetWorkspaceSize()
{
    size_t sortWorkspaceSize = static_cast<size_t>(totalLength) * sizeof(float) * NUM_TWO * NUM_THREE;
    size_t scatterWorkspaceSize = static_cast<size_t>(totalLength) * sizeof(int32_t) * NUM_TWO;
    size_t expertTokenFlagSize = static_cast<size_t>(aivNum) * NUM_TWO * sizeof(int32_t);
    workspaceSize_ = sortWorkspaceSize + scatterWorkspaceSize + expertTokenFlagSize +
                     static_cast<size_t>(SIZE_16) * LENGTH_1024 * LENGTH_1024;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeTokenPermuteArch35TilingClass::PostTiling()
{
    context_->SetBlockDim(aivNum);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    // arch35 SIMT 需显式下发可用 UB 大小；核间同步算子独占全核
    context_->SetLocalMemorySize(aicoreParams_.ubSize);
    context_->SetScheduleMode(1);
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    moeTokenPermuteTilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(moeTokenPermuteTilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

void MoeTokenPermuteArch35TilingClass::Tiling4VBSOneCoreCompute(PermuteVBSComputeTilingData* tilingData)
{
    int64_t needCoreNum = totalLength <= 0 ? 0 : 1;
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreElements(totalLength);
    tilingData->set_perCoreLoops(1);
    tilingData->set_perCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_perCoreLastLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLoops(1);
    tilingData->set_lastCorePerLoopElements(tilingData->get_perCoreElements());
    tilingData->set_lastCoreLastLoopElements(tilingData->get_perCoreElements());
}

void MoeTokenPermuteArch35TilingClass::Tiling4VBSMultiCoreCompute(PermuteVBSComputeTilingData* tilingData)
{
    int64_t needCoreNum = CeilDiv(totalLength, sortLoopMaxElement);
    needCoreNum = static_cast<int64_t>(std::pow(NUM_FOUR, CeilLog4(needCoreNum)));
    needCoreNum = std::min(needCoreNum, aivNum);
    if (needCoreNum == 0) {
        return;
    }
    int64_t perCoreElements = totalLength / needCoreNum;
    int64_t alineFloorPerCoreElements = perCoreElements - perCoreElements % SORT32_ALIGN_ELEMENT;
    int64_t lastCoreElement = totalLength - (needCoreNum - 1) * alineFloorPerCoreElements;
    int64_t alineCeilPerCoreElements = perCoreElements + SORT32_ALIGN_ELEMENT - perCoreElements % SORT32_ALIGN_ELEMENT;
    if (lastCoreElement > alineCeilPerCoreElements) {
        perCoreElements = alineCeilPerCoreElements;
        needCoreNum = CeilDiv(totalLength, perCoreElements);
    } else {
        perCoreElements = alineFloorPerCoreElements;
    }

    tilingData->set_needCoreNum(needCoreNum);
    do {
        tilingData->set_perCoreElements(perCoreElements);
        tilingData->set_perCoreLoops(CeilDiv(tilingData->get_perCoreElements(), sortLoopMaxElement));
        tilingData->set_perCorePerLoopElements(std::min(tilingData->get_perCoreElements(), sortLoopMaxElement));
        tilingData->set_perCoreLastLoopElements(
            tilingData->get_perCoreElements() -
            (tilingData->get_perCoreLoops() - 1) * tilingData->get_perCorePerLoopElements());

        tilingData->set_lastCoreElements(
            totalLength - (tilingData->get_needCoreNum() - 1) * tilingData->get_perCoreElements());
        tilingData->set_lastCoreLoops(tilingData->get_perCoreLoops());
        int64_t lastCorePerLoopElements =
            CeilDiv(CeilDiv(tilingData->get_lastCoreElements(), tilingData->get_lastCoreLoops()),
                    SORT32_ALIGN_ELEMENT) *
            SORT32_ALIGN_ELEMENT;
        tilingData->set_lastCorePerLoopElements(lastCorePerLoopElements);
        tilingData->set_lastCoreLastLoopElements(
            tilingData->get_lastCoreElements() -
            (tilingData->get_lastCoreLoops() - 1) * tilingData->get_lastCorePerLoopElements());
        perCoreElements -= SORT32_ALIGN_ELEMENT;
    } while (tilingData->get_lastCoreLastLoopElements() <= 0 && perCoreElements > 0);
}

void MoeTokenPermuteArch35TilingClass::Tiling4VBSCompute()
{
    auto tilingData = &moeTokenPermuteTilingData.vbsComputeParamsOp;
    tilingData->set_oneLoopMaxElements(sortLoopMaxElement);
    if (totalLength <= sortLoopMaxElement) {
        Tiling4VBSOneCoreCompute(tilingData);
        return;
    }
    Tiling4VBSMultiCoreCompute(tilingData);
}

void MoeTokenPermuteArch35TilingClass::Tiling4VMSMiddleCompute()
{
    auto vbsComputeTilingData = &moeTokenPermuteTilingData.vbsComputeParamsOp;
    auto tilingData = &moeTokenPermuteTilingData.vmsMiddleComputeParamsOp;
    if (vbsComputeTilingData->get_needCoreNum() <= MRG_LIST_NUM) {
        tilingData->set_needCoreNum(0);
        return;
    }
    tilingData->set_needCoreNum(CeilDiv(vbsComputeTilingData->get_needCoreNum(), MRG_LIST_NUM));
}

void MoeTokenPermuteArch35TilingClass::Tiling4SortOutCompute()
{
    auto tilingData = &moeTokenPermuteTilingData.sortOutComputeParamsOp;
    tilingData->set_oneLoopMaxElements(mrgSortListMaxElement);
}

void MoeTokenPermuteArch35TilingClass::Tiling4SrcToDstCompute()
{
    auto tilingData = &moeTokenPermuteTilingData.srcToDstComputeParamsOp;

    int64_t useCore = aivNum;
    int64_t perLoopMaxRows =
        (static_cast<int64_t>(aicoreParams_.ubSize) - ASSIST_NUM * static_cast<int64_t>(sizeof(float)) -
         useCore * SORT32_ALIGN_ELEMENT) /
        (SORT32_ALIGN_ELEMENT * NUM_TWO) / NUM_TWO;
    int64_t perCoreRows = CeilDiv(totalLength, useCore);
    if (perCoreRows <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
    tilingData->set_needCoreNum(needCoreNum);
    int64_t lastCoreNum = totalLength - perCoreRows * (needCoreNum - 1);
    tilingData->set_perCoreRows(perCoreRows);

    if (perLoopMaxRows >= perCoreRows) {
        tilingData->set_perCorePerLoopRows(perCoreRows);
        tilingData->set_perCoreLastLoopRows(perCoreRows);
    } else {
        tilingData->set_perCorePerLoopRows(perLoopMaxRows);
        tilingData->set_perCoreLastLoopRows(perCoreRows - (CeilDiv(perCoreRows, perLoopMaxRows) - 1) * perLoopMaxRows);
    }

    tilingData->set_lastCoreRows(lastCoreNum);
    if (perLoopMaxRows >= lastCoreNum) {
        tilingData->set_lastCorePerLoopRows(lastCoreNum);
        tilingData->set_lastCoreLastLoopRows(lastCoreNum);
    } else {
        tilingData->set_lastCorePerLoopRows(perLoopMaxRows);
        tilingData->set_lastCoreLastLoopRows(lastCoreNum - (CeilDiv(lastCoreNum, perLoopMaxRows) - 1) * perLoopMaxRows);
    }
}

void MoeTokenPermuteArch35TilingClass::SetGatherOutLoopParams(int64_t perCoreRows, int64_t lastCoreRows,
                                                              int64_t rowSize, int64_t colSize)
{
    auto tilingData = &moeTokenPermuteTilingData.gatherOutComputeParamsOp;
    int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO;
    int64_t cols = moeTokenPermuteTilingData.get_cols();

    int64_t ubRows = ubSize / (cols * tokenBtypeSize + static_cast<int64_t>(sizeof(int32_t)));
    bool colsAligned = cols * tokenBtypeSize % ONE_BLOCK_BYTE == 0;
    bool scatterHasNoExtraReads = activateNum <= moeTokenPermuteTilingData.get_n();
    bool smallRowScatter = moeTokenPermuteTilingData.get_topK() <= SCATTER_FULL_MAX_K &&
                           cols * tokenBtypeSize <= SCATTER_FULL_MAX_ROW_BYTES;
    bool useScatterMode = colsAligned && ubRows >= NUM_TWO &&
                          moeTokenPermuteTilingData.get_topK() <= SCATTER_MODE_MAX_K &&
                          (scatterHasNoExtraReads || smallRowScatter);
    // 仅在成功分核后才置 scatterMode=1，保证 kernel 侧 scatterMode==1 => needCoreNum>0 的不变式。
    // activateNum<=0（空输出）时分核失败，回落到 generic gather，由其自身 needCoreNum 统一处理空输出。
    if (useScatterMode && CalculateScatterOutCoreAllocation(tilingData->get_activateRows(), tilingData)) {
        tilingData->set_scatterMode(1);
        CalculateScatterOutBufferParams(ubRows, cols, tilingData);
        return;
    }
    tilingData->set_scatterMode(0);

    if (rowSize + colSize < ubSize) {
        tilingData->set_perCorePerLoopRows(perCoreRows);
        tilingData->set_perCoreLastLoopRows(perCoreRows);
        tilingData->set_lastCorePerLoopRows(lastCoreRows);
        tilingData->set_lastCoreLastLoopRows(lastCoreRows);
        tilingData->set_perCoreLoops(1);
        tilingData->set_lastCoreLoops(1);
        int64_t loopCols = (rowSize + colSize * NUM_TWO < ubSize) ? cols * NUM_TWO : cols;
        tilingData->set_perLoopCols(loopCols);
        tilingData->set_lastLoopCols(loopCols);
        tilingData->set_colLoops(1);
    } else {
        int64_t baseMaxCols = MAX_COLS_ONE_LOOP;
        int64_t baseMaxColsSize = UpAlign(baseMaxCols * tokenBtypeSize, ONE_BLOCK_BYTE);
        int64_t basePerLoopMaxRows =
            (ubSize - baseMaxColsSize) / static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        if (cols < MAX_COLS_ONE_LOOP) {
            basePerLoopMaxRows =
                (ubSize - colSize) / static_cast<int64_t>(sizeof(int32_t)) / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        } else if (perCoreRows < basePerLoopMaxRows) {
            baseMaxCols = (ubSize - rowSize) / tokenBtypeSize / ONE_BLOCK_BYTE * ONE_BLOCK_BYTE;
        }
        tilingData->set_perLoopCols(std::min(baseMaxCols, cols));
        tilingData->set_lastLoopCols(GetPerOrLastValue(cols, baseMaxCols));
        tilingData->set_colLoops(CeilDiv(cols, baseMaxCols));

        tilingData->set_perCorePerLoopRows(std::min(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, basePerLoopMaxRows));
        tilingData->set_perCoreLoops(CeilDiv(perCoreRows, basePerLoopMaxRows));

        tilingData->set_lastCorePerLoopRows(std::min(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, basePerLoopMaxRows));
        tilingData->set_lastCoreLoops(CeilDiv(lastCoreRows, basePerLoopMaxRows));
    }
}

bool MoeTokenPermuteArch35TilingClass::CalculateScatterOutCoreAllocation(
    int64_t actualActivateRows, PermuteGatherOutComputeTilingData* tilingData)
{
    if (actualActivateRows <= 0) {
        tilingData->set_needCoreNum(0);
        return false;
    }

    int64_t perCoreRows = CeilDiv(actualActivateRows, aivNum);
    int64_t needCoreNum = std::min(CeilDiv(actualActivateRows, perCoreRows), aivNum);
    perCoreRows = CeilDiv(actualActivateRows, needCoreNum);
    int64_t lastCoreRows = actualActivateRows - perCoreRows * (needCoreNum - 1);
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreRows(perCoreRows);
    tilingData->set_lastCoreRows(lastCoreRows);
    return true;
}

void MoeTokenPermuteArch35TilingClass::CalculateScatterOutBufferParams(
    int64_t ubRows, int64_t cols, PermuteGatherOutComputeTilingData* tilingData)
{
    int64_t ubSize = static_cast<int64_t>(aicoreParams_.ubSize) / NUM_TWO;
    int64_t outputBufferSize = UpAlign(ubRows * cols * tokenBtypeSize, ONE_BLOCK_BYTE);
    int64_t maxRowsPerLoop = std::max((ubSize - outputBufferSize) / static_cast<int64_t>(sizeof(int32_t)), ubRows);

    int64_t perCoreRows = tilingData->get_perCoreRows();
    int64_t lastCoreRows = tilingData->get_lastCoreRows();
    int64_t perCorePerLoopRows = std::min(perCoreRows, maxRowsPerLoop);
    int64_t lastCorePerLoopRows = std::min(lastCoreRows, maxRowsPerLoop);
    tilingData->set_perCorePerLoopRows(perCorePerLoopRows);
    tilingData->set_perCoreLastLoopRows(GetPerOrLastValue(perCoreRows, perCorePerLoopRows));
    tilingData->set_lastCorePerLoopRows(lastCorePerLoopRows);
    tilingData->set_lastCoreLastLoopRows(GetPerOrLastValue(lastCoreRows, lastCorePerLoopRows));
    tilingData->set_perCoreLoops(CeilDiv(perCoreRows, perCorePerLoopRows));
    tilingData->set_lastCoreLoops(CeilDiv(lastCoreRows, lastCorePerLoopRows));
    tilingData->set_perLoopCols(ubRows);
    tilingData->set_lastLoopCols(ubRows);
    tilingData->set_colLoops(1);
}

void MoeTokenPermuteArch35TilingClass::Tiling4GatherOutCompute()
{
    auto tilingData = &moeTokenPermuteTilingData.gatherOutComputeParamsOp;
    tilingData->set_activateRows(activateNum); // dropPadMode==0：activateRows = numOutTokens（越界目标行丢弃）
    int64_t perCoreRows = CeilDiv(totalLength, aivNum);
    int64_t cols = moeTokenPermuteTilingData.get_cols();
    if (perCoreRows <= 0 || cols <= 0) {
        tilingData->set_needCoreNum(0);
        return;
    }
    int64_t needCoreNum = CeilDiv(totalLength, perCoreRows);
    tilingData->set_needCoreNum(needCoreNum);
    tilingData->set_perCoreRows(perCoreRows);
    int64_t lastCoreRows = totalLength - perCoreRows * (needCoreNum - 1);
    tilingData->set_lastCoreRows(lastCoreRows);

    int64_t rowSize = UpAlign(perCoreRows * static_cast<int64_t>(sizeof(int32_t)), ONE_BLOCK_BYTE);
    int64_t colSize = UpAlign(cols * tokenBtypeSize, ONE_BLOCK_BYTE);
    SetGatherOutLoopParams(perCoreRows, lastCoreRows, rowSize, colSize);
}

REGISTER_OPS_TILING_TEMPLATE(MoeTokenPermute, MoeTokenPermuteArch35TilingClass, 1000);
} // namespace optiling
