/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kvquant_sparse_attn_sharedkv_tiling.h
 * \brief Tiling data definition for KvQuantSparseAttnSharedKv operator
 */

#ifndef KVQUANT_SPARSE_ATTN_SHAREDKV_TILING_H_
#define KVQUANT_SPARSE_ATTN_SHAREDKV_TILING_H_

#include <sstream>
#include <graph/utils/type_utils.h>
#include <exe_graph/runtime/tiling_context.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/soc_spec.h"

namespace optiling {

constexpr uint32_t QSAS_Q_INPUT_INDEX = 0;
constexpr uint32_t QSAS_KV_INPUT_INDEX = 1;
constexpr uint32_t QSAS_SPARSE_INDICES_INPUT_INDEX = 2;
constexpr uint32_t QSAS_BLOCK_TABLE_INPUT_INDEX = 3;
constexpr uint32_t QSAS_CU_SEQLENS_Q_INPUT_INDEX = 4;
constexpr uint32_t QSAS_CU_SEQLENS_KV_INPUT_INDEX = 5;
constexpr uint32_t QSAS_SEQUSED_Q_INPUT_INDEX = 6;
constexpr uint32_t QSAS_SEQUSED_KV_INPUT_INDEX = 7;
constexpr uint32_t QSAS_METADATA_INPUT_INDEX = 8;

constexpr uint32_t QSAS_OUTPUT_INDEX = 0;
constexpr uint32_t QSAS_ATTEN_OUT_INDEX = 0;
constexpr uint32_t QSAS_SOFTMAX_LSE_INDEX = 1;

constexpr uint32_t QSAS_SCALE_VALUE_ATTR_INDEX = 0;
constexpr uint32_t QSAS_KV_QUANT_MODE_ATTR_INDEX = 1;
constexpr uint32_t QSAS_SPARSE_BLOCK_SIZE_ATTR_INDEX = 2;
constexpr uint32_t QSAS_LAYOUT_QUERY_ATTR_INDEX = 3;
constexpr uint32_t QSAS_LAYOUT_KV_ATTR_INDEX = 4;
constexpr uint32_t QSAS_SPARSE_MODE_ATTR_INDEX = 5;
constexpr uint32_t QSAS_PRE_TOKENS_ATTR_INDEX = 6;
constexpr uint32_t QSAS_NEXT_TOKENS_ATTR_INDEX = 7;
constexpr uint32_t QSAS_ATTENTION_MODE_ATTR_INDEX = 8;
constexpr uint32_t QSAS_QUANT_SCALE_REPO_MODE_ATTR_INDEX = 9;
constexpr uint32_t QSAS_TILE_SIZE_ATTR_INDEX = 10;
constexpr uint32_t QSAS_ROPE_HEAD_DIM_ATTR_INDEX = 11;

constexpr size_t QSAS_DIM_NUM_TWO = 2;
constexpr size_t QSAS_DIM_NUM_THREE = 3;
constexpr size_t QSAS_DIM_NUM_FOUR = 4;

constexpr uint32_t QSAS_MAX_BLOCK_SIZE = 1024;
constexpr uint32_t QSAS_COPYND2NZ_SRC_STRIDE_LIMITATION = 65535;
constexpr uint32_t QSAS_NUM_BYTES_FLOAT = 4;
constexpr uint32_t QSAS_NUM_BYTES_FLOAT16 = 2;
constexpr uint32_t QSAS_NUM_BYTES_BF16 = 2;
constexpr uint32_t QSAS_METADATA_LIMIT = 1024;
constexpr uint32_t QSAS_BYTE_BLOCK = 32;
constexpr uint32_t QSAS_MAX_AIC_CORE_NUM = 26;

enum class QSASLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

struct QSASTilingShapeCompareParam {
    int64_t B = 1;
    int64_t S = 1;
    int64_t N = 1;
    int64_t D = 1;
    int64_t T = 1;
    int64_t Bs = 1;
    int64_t Bn = 1;
};

enum class QSASStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    PAGE_ATTENTION = 1
};

enum class QSASPerfMode : uint32_t {
    C_TEMPLATE_MODE = 0,
    V_TEMPLATE_MODE
};

enum class QSASAxis : uint32_t {
    B = 0,
    S = 1,
    N = 2,
    D = 3,
    K = 3,
    T = 5,
    Bn = 6,
    Bs = 7,
};

struct QSASRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct QSASOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct QSASParaInfo {
    QSASRequiredParaInfo query = {nullptr, nullptr};
    QSASRequiredParaInfo kv = {nullptr, nullptr};
    QSASRequiredParaInfo sparseIndices = {nullptr, nullptr};
    QSASOptionalParaInfo blockTable = {nullptr, nullptr};
    QSASOptionalParaInfo cuSeqlensQ = {nullptr, nullptr};
    QSASOptionalParaInfo cuSeqlensKv = {nullptr, nullptr};
    QSASOptionalParaInfo sequsedQ = {nullptr, nullptr};
    QSASOptionalParaInfo sequsedKv = {nullptr, nullptr};
    QSASOptionalParaInfo metadata = {nullptr, nullptr};
    QSASRequiredParaInfo attenOut = {nullptr, nullptr};
    QSASOptionalParaInfo softmaxLse = {nullptr, nullptr};

    const char *layoutQuery = nullptr;
    const char *layoutKV = nullptr;
    const int64_t *sparseBlockSize = nullptr;
    const uint32_t *sparseBlockCount = nullptr;
    const uint32_t *blockSize = nullptr;
    const float *scaleValue = nullptr;
    const int64_t *sparseMode = nullptr;
    const int64_t *attentionMode = nullptr;
    const int64_t *kvQuantMode = nullptr;
    const int64_t *quantScaleRepoMode = nullptr;
    const int64_t *tileSize = nullptr;
    const int64_t *ropeHeadDim = nullptr;
    const int64_t *preTokens = nullptr;
    const int64_t *nextTokens = nullptr;
};

struct QSASInnerSplitParams {
    uint32_t s1GBaseSize = 1;
    uint32_t s2BaseSize = 1;
};

BEGIN_TILING_DATA_DEF(KvQuantSparseAttnSharedKvBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, seqSize)
TILING_DATA_FIELD_DEF(uint32_t, qSeqSize)
TILING_DATA_FIELD_DEF(int64_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(float, scaleValue)
TILING_DATA_FIELD_DEF(uint32_t, nNumOfQInOneGroup)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsQ)
TILING_DATA_FIELD_DEF(uint32_t, actualLenDimsKV)
TILING_DATA_FIELD_DEF(uint32_t, outputLayout)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(int64_t, sparseBlockSize)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockCount)
TILING_DATA_FIELD_DEF(int64_t, dSizeVInput)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(KvQuantSparseAttnSharedKvBaseParamsOp, KvQuantSparseAttnSharedKvBaseParams)

BEGIN_TILING_DATA_DEF(KvQuantSparseAttnSharedKvSingleCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(KvQuantSparseAttnSharedKvSingleCoreParamsOp, KvQuantSparseAttnSharedKvSingleCoreParams)

BEGIN_TILING_DATA_DEF(KvQuantSparseAttnSharedKvSingleCoreTensorSize)
TILING_DATA_FIELD_DEF(uint32_t, mmResUbSize);
TILING_DATA_FIELD_DEF(uint32_t, bmm2ResUbSize);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(KvQuantSparseAttnSharedKvSingleCoreTensorSizeOp, KvQuantSparseAttnSharedKvSingleCoreTensorSize)

BEGIN_TILING_DATA_DEF(KvQuantSparseAttnSharedKvSplitKVParams)
TILING_DATA_FIELD_DEF(uint32_t, s2)
TILING_DATA_FIELD_DEF(uint32_t, accumOutSize)
TILING_DATA_FIELD_DEF(uint32_t, logSumExpSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(KvQuantSparseAttnSharedKvSplitKVParamsOp, KvQuantSparseAttnSharedKvSplitKVParams)

BEGIN_TILING_DATA_DEF(KvQuantSparseAttnSharedKvInnerSplitParams)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(KvQuantSparseAttnSharedKvInnerSplitParamsOp, KvQuantSparseAttnSharedKvInnerSplitParams)

BEGIN_TILING_DATA_DEF(KvQuantSparseAttnQSASTilingData)
TILING_DATA_FIELD_DEF_STRUCT(KvQuantSparseAttnSharedKvBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(KvQuantSparseAttnSharedKvSplitKVParams, splitKVParams);
TILING_DATA_FIELD_DEF_STRUCT(KvQuantSparseAttnSharedKvSingleCoreParams, singleCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(KvQuantSparseAttnSharedKvSingleCoreTensorSize, singleCoreTensorSize);
TILING_DATA_FIELD_DEF_STRUCT(KvQuantSparseAttnSharedKvInnerSplitParams, innerSplitParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(KvQuantSparseAttnSharedKv, KvQuantSparseAttnQSASTilingData)

template <typename T> inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

template <typename T>
std::string QSASShape2String(const T &shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

static std::string GetShapeStr(gert::Shape shape);
static std::string QSASDataTypeToSerialString(ge::DataType type);
std::string QSASTensorDesc2String(const gert::StorageShape *shape, const gert::CompileTimeTensorDesc *tensor);
std::string QSASDebugTilingContext(const gert::TilingContext *context);
std::string QSASLayoutToSerialString(QSASLayout layout);

struct QSASTilingInfo {
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    QSASParaInfo opParamInfo;

    NpuArch npuArch = NpuArch::DAV_2201;
    bool isA5 = false;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t n2Size = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qHeadDim = 0;
    uint32_t kvHeadDim = 0;
    uint32_t gSize = 0;
    uint32_t ropeHeadDim = 0;
    uint32_t qTSize = 0;
    uint32_t kvTSize = 0;
    float scaleValue = 0;
    uint32_t innerPrecise = 0;
    uint32_t l2CacheOffFlag = 0;
    int64_t sparseBlockSize = 0;
    int64_t sparseBlockCount = 0;

    bool pageAttentionFlag = false;
    int64_t blockSize = 0;
    uint32_t blockTypeSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t totalBlockNum = 0;

    uint32_t actualLenDimsQ = 0;
    uint32_t maxActualseq = 0;

    bool actualSeqLenFlag = false;
    bool isSameSeqAllKVTensor = true;
    bool isSameActualseq = true;
    uint32_t actualLenDimsKV = 0;
    std::vector<int64_t> kvListSeqLens {};

    uint32_t sparseMode = 0;

    int64_t attentionMode = 0;
    int64_t kvQuantMode = 0;
    int64_t quantScaleRepoMode = 0;
    int64_t tileSize = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;

    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_INT8;
    ge::DataType outputType = ge::DT_FLOAT16;

    QSASStorageMode kvStorageMode = QSASStorageMode::BATCH_CONTINUOUS;

    QSASLayout qLayout = QSASLayout::BSND;
    QSASLayout outLayout = QSASLayout::BSND;
    QSASLayout kvLayout = QSASLayout::BSND;

    uint64_t l2CacheSize = 0;
    int64_t dSizeVInput = 0;
};

class QSASTiling {
public:
    explicit QSASTiling(gert::TilingContext *context) : context_(context) {}
    ge::graphStatus DoOpTiling(QSASTilingInfo *qsasInfo);

private:
    ge::graphStatus SetBlockDim(uint32_t blockDim) const;
    ge::graphStatus SetTilingKey(uint64_t tilingKey) const;
    ge::graphStatus SetWorkspaceSize(uint64_t workspaceSize) const;
    ge::graphStatus SetTilingData(TilingDef &tilingData) const;
    gert::TilingContext *context_ = nullptr;
    ge::graphStatus GetPlatformInfo();
    void GenTilingKey();
    bool DealSameSeqEachBatch();

    void ZeroTensorProcess() const;
    void InitParams();

    void Split();
    bool IsBalanceSplitCore();

    void SplitBalanced();
    void CalcInnerSize(uint32_t s2Size);

    bool IsFlashDecode(uint32_t coreNum);

    void FillTilingBaseParams();
    void FillTilingSplitKV();

    void FillTilingSingleCoreParams();
    void FillTilingSingleCoreTensorSize();
    void FillTiling();

    void CalcUbBmm();
    void CheckUbSpace();
    void NormalCalcFDWorkSpace(const uint32_t actCoreNum);
    void CalcFDWorkSpace(const uint32_t actCoreNum);
    void GetWorkspaceSize();

    uint32_t CalcBalanceFDParamNums(const uint32_t actCoreNum) const;

    void CalcBlockDim();

    bool balanceModeFlag_ = false;
    bool splitKVFlag_ = false;

    uint32_t coreNum_ = 0;
    QSASPerfMode perfMode_ = QSASPerfMode::V_TEMPLATE_MODE;
    uint32_t kvSplitPart_ = 1;
    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    size_t qPreSize_ = 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 0;
    uint32_t sInnerSizeTail_ = 0;
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t kvSplit_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t formerCoreNum_ = 0;
    uint32_t blockSplitBn2Range_ = 0;
    uint32_t tailSplitedBatchRange_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;

    KvQuantSparseAttnQSASTilingData tilingData_;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 128;
    uint32_t mFdBaseSize_ = 8;

    QSASTilingInfo *qsasInfo_ = nullptr;
};

class QSASTilingCheck {
public:
    explicit QSASTilingCheck(const QSASTilingInfo &qsasInfo) : qsasInfo_(qsasInfo) {};
    ~QSASTilingCheck() = default;
    ge::graphStatus Process();
private:
    void Init();
    void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
        const ge::DataType &actualDtype, const std::string &name) const;
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
        const std::string &name) const;
    template <typename T> void LogErrorNumberSupport(const std::vector<T> &expectNumberList,
        const T &actualValue, const std::string &name, const std::string subName) const;
    template <typename T> void LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
        const T &actualValue, const std::string &name) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape,
        const std::vector<size_t> &expectDimNumList, const std::string &name) const;
    ge::graphStatus CheckDimNumInLayoutSupport(const QSASLayout &layout,
        const gert::StorageShape *shape, const std::string &name) const;
    void LogErrorLayoutSupport(const std::vector<QSASLayout> &expectLayoutList,
        const QSASLayout &actualLayout, const std::string &name) const;
    ge::graphStatus GetExpectedShape(gert::Shape &shapeExpected,
        const QSASTilingShapeCompareParam &param, const QSASLayout &layout) const;
    ge::graphStatus CompareShape(QSASTilingShapeCompareParam &param,
        const gert::Shape &shape, const QSASLayout &layout, const std::string &name) const;
    ge::graphStatus CheckLayoutSupport(const QSASLayout &actualLayout, const std::string &name) const;
    ge::graphStatus CheckSingleParaQuery() const;
    ge::graphStatus CheckSingleParaKv() const;
    ge::graphStatus CheckSingleParaSparseIndices() const;
    ge::graphStatus CheckSingleParaAttenOut() const;
    ge::graphStatus CheckSingleParaLayout() const;
    ge::graphStatus CheckSingleParaSparseMode() const;
    ge::graphStatus CheckSingleParaSparseBlockSize() const;
    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus CheckMultiParaConsistency() const;
    ge::graphStatus CheckDequantScaleNotExistence();
    template <typename T> ge::graphStatus CheckAttrValueByMap(
        std::map<std::string, std::pair<const T *, T>> &attrMap) const;
    ge::graphStatus CheckParaExistence() const;
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
        const QSASLayout &layout, const std::string &name) const;
    void SetQSASShapeCompare();
    ge::graphStatus CheckKvDType();
    ge::graphStatus CheckKvShapeForBatchContinuous();
    uint32_t GetTypeSize(ge::DataType dtype) const;
    ge::graphStatus CheckKvShapeForPageAttention();
    ge::graphStatus CheckKvShape();
    ge::graphStatus CheckKv();
    ge::graphStatus CheckBlockTable() const;
    ge::graphStatus CheckDTypeConsistency(const ge::DataType &actualDtype,
        const ge::DataType &expectDtype, const std::string &name) const;

    ge::graphStatus CheckAttenOut();
    ge::graphStatus CheckAttenOutShape();
    ge::graphStatus CheckActualSeqLensQ();
    ge::graphStatus CheckMetadata();
    ge::graphStatus CheckActualSeqLensQShape();
    ge::graphStatus CheckActualSeqLensQDType();
    ge::graphStatus CheckActualSeqLensKv();
    ge::graphStatus CheckActualSeqLensKvDType();
    ge::graphStatus CheckActualSeqLensKvShape();
    ge::graphStatus CheckMultiParaConsistencyForQSAS();

    ge::graphStatus CheckFeature() const;

private:
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    QSASParaInfo opParamInfo_;
    const QSASTilingInfo &qsasInfo_;

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qHeadDim_ = 0;
    uint32_t kvHeadDim_ = 0;
    uint32_t qTSize_ = 0;
    uint32_t kvTSize_ = 0;
    QSASStorageMode kvStorageMode_ = QSASStorageMode::BATCH_CONTINUOUS;
    uint32_t sparseBlockCount_ = 0;
    int64_t sparseBlockSize_ = 0;
    int32_t attentionMode_ = 0;
    int32_t kvQuantMode_ = 0;
    int32_t quantScaleRepoMode_ = 0;
    int64_t tileSize_ = 0;
    int64_t preTokens_ = 0;
    int64_t nextTokens_ = 0;
    int32_t ropeHeadDim_ = 0;

    QSASLayout qLayout_ = QSASLayout::BSND;
    QSASLayout outLayout_ = QSASLayout::BSND;
    QSASLayout kvLayout_ = QSASLayout::BSND;

    uint32_t maxBlockNumPerBatch_ = 0;
    int64_t blockSize_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    NpuArch npuArch_ = NpuArch::DAV_2201;
    bool isA5_ = false;
    uint64_t l2CacheSize_ = 0;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_INT8;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    gert::Shape queryShapeCmp_{};
    gert::Shape kvShapeCmp_{};
    gert::Shape sparseIndicesShapeCmp_{};
    gert::Shape attenOutShapeCmp_{};
};

class QSASInfoParser {
public:
    explicit QSASInfoParser(const gert::TilingContext *context) : context_(context) {}
    ~QSASInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
        QSASLayout &layout, const std::string &name) const;
    ge::graphStatus GetActualSeqLenQSize();
    ge::graphStatus GetActualSeqLenKvSize();
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    void GetOriKvParaInfo();
    void GetOriSparseIndicesParaInfo();
    void GetOriBlockTableParaInfo();
    void GetCuSeqlensQParaInfo();
    void GetCuSeqlensKvParaInfo();
    void GetSequsedQParaInfo();
    void GetSequsedKvParaInfo();
    void GetMetadataParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetKv();
    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetQTSize();
    ge::graphStatus GetKVTSize();
    ge::graphStatus GetQHeadDim();
    ge::graphStatus GetKvHeadDim();
    ge::graphStatus GetS1Size();
    ge::graphStatus GetKvStorageMode();
    ge::graphStatus GetKvLayout();
    void SetQSASShape();
    ge::graphStatus GetS2SizeForBatchContinuous();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetDSizeKV();
    ge::graphStatus GetRopeHeadDim();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetSparseBlockCount();
    ge::graphStatus GetActualseqInfo();
    void GenerateInfo(QSASTilingInfo &qsasInfo);
    ge::graphStatus Parse(QSASTilingInfo &qsasInfo);

public:
    bool HasAxis(const QSASAxis &axis, const QSASLayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const QSASAxis &axis, const QSASLayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const QSASAxis &axis, const QSASLayout &layout) const;

    const gert::TilingContext *context_ = nullptr;

    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    QSASParaInfo opParamInfo_;
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t qHeadDim_ = 0;
    uint32_t kvHeadDim_ = 0;
    int32_t ropeHeadDim_ = 0;
    int64_t dSizeKV_ = 0;
    uint32_t qTSize_ = 0;
    uint32_t kvTSize_ = 0;
    QSASStorageMode kvStorageMode_ = QSASStorageMode::BATCH_CONTINUOUS;
    uint32_t sparseBlockCount_ = 0;

    QSASLayout qLayout_ = QSASLayout::BSND;
    QSASLayout outLayout_ = QSASLayout::BSND;
    QSASLayout kvLayout_ = QSASLayout::BSND;

    uint32_t maxBlockNumPerBatch_ = 0;
    uint32_t blockSize_ = 0;

    NpuArch npuArch_ = NpuArch::DAV_2201;
    bool isA5_ = false;

    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKvType_ = ge::DT_INT8;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    uint64_t l2CacheSize_ = 0;

    bool isSameSeqAllKVTensor_ = true;
    bool isSameActualseq_ = true;
    uint32_t maxActualseq_ = 0;

    uint32_t actualLenDimsQ_ = 0;
    uint32_t actualLenDimsKV_ = 0;

    gert::Shape queryShape_{};
    gert::Shape kvShape_{};
    gert::Shape sparseIndicesShape_{};
    gert::Shape oriKvShape_{};
    gert::Shape oriSparseIndicesShape_{};
    gert::Shape oriBlockTableShape_{};
    gert::Shape cuSeqlensQShape_{};
    gert::Shape cuSeqlensKvShape_{};
    gert::Shape sequsedQShape_{};
    gert::Shape sequsedKvShape_{};
    gert::Shape metadataShape_{};
};

}  // namespace optiling

#endif  // KVQUANT_SPARSE_ATTN_SHAREDKV_TILING_H_