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
 * \file sparse_flash_mla_tiling.h
 * \brief A2/A3 (arch22) tiling class declarations.
 */
#ifndef SPARSE_FLASH_MLA_TILING_ARCH22_H
#define SPARSE_FLASH_MLA_TILING_ARCH22_H

#include "../sparse_flash_mla_tiling.h"

namespace optiling {
namespace arch22 {
using SMLATilingRequiredParaInfo = optiling::SMLATilingRequiredParaInfo;
using SMLATilingOptionalParaInfo = optiling::SMLATilingOptionalParaInfo;
using SMLALayout = optiling::SMLALayout;
using SMLAAxis = optiling::SMLAAxis;
using SMLATemplateMode = optiling::SMLATemplateMode;
using KvStorageMode = optiling::KvStorageMode;
using SMLAParaInfo = optiling::SMLAParaInfo;
using SMLATilingInfo = optiling::SMLATilingInfo;
using SparseFlashMlaSwaParams = optiling::SparseFlashMlaSwaParams;
using SparseFlashMlaCmpParams = optiling::SparseFlashMlaCmpParams;
using SparseFlashMlaTilingData = optiling::SparseFlashMlaTilingData;

using optiling::Q_INDEX;
using optiling::ORI_KV_INDEX;
using optiling::CMP_KV_INDEX;
using optiling::ORI_SPARSE_INDICES_INDEX;
using optiling::CMP_SPARSE_INDICES_INDEX;
using optiling::ORI_BLOCK_TABLE_INDEX;
using optiling::CMP_BLOCK_TABLE_INDEX;
using optiling::CU_SEQLENS_Q_INDEX;
using optiling::CU_SEQLENS_ORI_KV_INDEX;
using optiling::CU_SEQLENS_CMP_KV_INDEX;
using optiling::SEQUSED_Q_INDEX;
using optiling::SEQUSED_ORI_KV_INDEX;
using optiling::SEQUSED_CMP_KV_INDEX;
using optiling::CMP_RESIDUAL_KV_INDEX;
using optiling::ORI_TOPK_LENGTH_INDEX;
using optiling::CMP_TOPK_LENGTH_INDEX;
using optiling::SINKS_INDEX;
using optiling::METADATA_INDEX;
using optiling::ATTN_OUT_INDEX;
using optiling::ATTR_SOFTMAX_SCALE_INDEX;
using optiling::ATTR_CMP_RATIO_INDEX;
using optiling::ATTR_ORI_MASK_MODE_INDEX;
using optiling::ATTR_CMP_MASK_MODE_INDEX;
using optiling::ATTR_ORI_WIN_LEFT_INDEX;
using optiling::ATTR_ORI_WIN_RIGHT_INDEX;
using optiling::ATTR_LAYOUT_Q_INDEX;
using optiling::ATTR_LAYOUT_KV_INDEX;
using optiling::ATTR_TOPK_VALUE_MODE_INDEX;
using optiling::ATTR_ORI_KV_STRIDE_INDEX;
using optiling::ATTR_CMP_KV_STRIDE_INDEX;
using optiling::ATTR_RETURN_SOFTMAX_LSE_INDEX;
using optiling::DIM_IDX_ONE;
using optiling::DIM_IDX_TWO;
using optiling::DIM_IDX_THREE;
using optiling::DIM_IDX_FOUR;
using optiling::DIM_NUM_ONE;
using optiling::DIM_NUM_TWO;
using optiling::DIM_NUM_THREE;
using optiling::DIM_NUM_FOUR;
using optiling::MAX_BLOCK_SIZE;
using optiling::COPYND2NZ_SRC_STRIDE_LIMITATION;
using optiling::NUM_BYTES_FLOAT;
using optiling::NUM_BYTES_FLOAT16;
using optiling::NUM_BYTES_BF16;
using optiling::BYTE_BLOCK;
using optiling::HEAD_DIM_LIMIT;
using optiling::SPARSE_LIMIT;
using optiling::SPARSE_MODE_LOWER;
using optiling::METADATA_LIMIT;
using optiling::DIM_LIMIT;
using optiling::TOPK_LIMIT;
using optiling::BLOCK_SIZE_LIMIT;
using optiling::Align;

class SMLATilingCheck {
public:
    explicit SMLATilingCheck(const SMLATilingInfo &smlaInfo) : smlaInfo_(smlaInfo) {}
    ~SMLATilingCheck() = default;
    virtual ge::graphStatus Process();

private:
    void Init();

    void LogErrorDtypeSupport(const std::vector<ge::DataType> &expectDtypeList,
        const ge::DataType &actualDtype, const std::string &name) const;
    ge::graphStatus CheckLayoutSupport(const SMLALayout &actualLayout, const std::string &name) const;
    template <typename T>
    void LogErrorDimNumSupport(const std::vector<T> &expectNumberList,
        const T &actualValue, const std::string &name) const;
    template <typename T>
    void LogErrorNumberSupport(const std::vector<T> &expectNumberList,
        const T &actualValue, const std::string &name, const std::string subName) const;
    ge::graphStatus CheckDimNumSupport(const gert::StorageShape *shape,
        const std::vector<size_t> &expectDimNumList, const std::string &name) const;
    void LogErrorLayoutSupport(const std::vector<SMLALayout> &expectLayoutList,
        const SMLALayout &actualLayout, const std::string &name) const;
    ge::graphStatus CheckDimNumInLayoutSupport(const SMLALayout &layout,
        const gert::StorageShape *shape, const std::string &name) const;
    ge::graphStatus CheckDtypeSupport(const gert::CompileTimeTensorDesc *desc,
        const std::string &name) const;
    ge::graphStatus CheckSinglePara() const;
    ge::graphStatus CheckSingleParaQuery() const;
    ge::graphStatus CheckSingleParaOriKv() const;
    ge::graphStatus CheckSingleParaCmpKv() const;
    ge::graphStatus CheckSingleParaCuSeqLensOriKv() const; // A5
    ge::graphStatus CheckSingleParaCuSeqLensCmpKv() const; // A5
    ge::graphStatus CheckSingleParaNumHeads() const;
    ge::graphStatus CheckSingleParaKvHeadNums() const;
    ge::graphStatus CheckSingleParaOriSparseIndices() const;
    ge::graphStatus CheckSingleParaCmpSparseIndices() const;
    ge::graphStatus CheckSingleParaSinks() const;
    ge::graphStatus CheckSingleParaMetadata() const;
    ge::graphStatus CheckSingleParaCmpRatio() const;
    ge::graphStatus CheckSingleParaOriMaskMode() const;
    ge::graphStatus CheckSingleParaCmpMaskMode() const;
    ge::graphStatus CheckSingleParaOriKvStride0() const; // A2/A3
    ge::graphStatus CheckSingleParaCmpKvStride0() const; // A2/A3
    ge::graphStatus CheckSingleParaOriWinLeft() const;
    ge::graphStatus CheckSingleParaOriWinRight() const;
    ge::graphStatus CheckSingleParaOriBlockTable() const;
    ge::graphStatus CheckSingleParaCmpBlockTable() const;

    ge::graphStatus CheckParaExistence() const;
    ge::graphStatus CheckExists(const void *pointer, const std::string &name) const;
    ge::graphStatus CheckNotExists(const void *pointer, const std::string &name) const;
    ge::graphStatus CheckExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    ge::graphStatus CheckNotExistsByMap(const std::map<std::string, const void *> &paramMap) const;
    ge::graphStatus CheckExistenceByMap(std::map<std::string, const void *> &existMap,
        std::map<std::string, const void *> &notExistMap) const;

    ge::graphStatus CheckFeature() const;
    ge::graphStatus CheckFeatureShape() const;
    ge::graphStatus CheckFeatureLayout() const;
    ge::graphStatus CheckFeatureDtype() const;
    ge::graphStatus CheckFeaturePa() const;

    ge::graphStatus CheckMultiParaConsistency();
    void SetSMLAShapeCompare();
    ge::graphStatus CheckDTypeConsistency(const ge::DataType &actualDtype,
        const ge::DataType &expectDtype, const std::string &name) const;
    ge::graphStatus CheckOriAndCmpKv() const;
    ge::graphStatus CheckAttenOut() const;
    ge::graphStatus CheckActualSeqLensQ() const;
    ge::graphStatus CheckActualSeqLens() const;
    ge::graphStatus CheckBlockTable() const;

    gert::Shape queryShapeCmp_{};
    gert::Shape oriKvShapeCmp_{};
    gert::Shape cmpKvShapeCmp_{};
    gert::Shape oriKvSparseIndicesCmp_{};
    gert::Shape cmpKvSparseIndicesCmp_{};
    gert::Shape attenOutShapeCmp_{};

    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    SMLAParaInfo opParamInfo_;
    const SMLATilingInfo &smlaInfo_;

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t cmpS2Size_ = 0; // A5
    uint32_t qHeadDim_ = 0;
    uint32_t oriKvHeadDim_ = 0;
    uint32_t cmpKvHeadDim_ = 0;

    uint32_t qTSize_ = 0;
    uint32_t kvTSize_ = 0;
    int64_t cmpRatio_ = 1;
    KvStorageMode kvStorageMode_ = KvStorageMode::BATCH_CONTINUOUS;
    uint32_t oriSparseBlockCount_ = 0; // A5
    uint32_t cmpSparseBlockCount_ = 0; // A5
    uint32_t sparseBlockCount_ = 0;    // A2/A3
    int64_t oriWinLeft_ = 0;
    int64_t oriWinRight_ = 0;
    SMLALayout qLayout_ = SMLALayout::TND;
    SMLALayout cmpSparseIndicesLayout_ = SMLALayout::TND;
    SMLALayout oriSparseIndicesLayout_ = SMLALayout::TND;
    SMLALayout outLayout_ = SMLALayout::TND;
    SMLALayout kvLayout_ = SMLALayout::PA_BBND;

    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    int64_t blockSize_ = 0;
    int32_t oriBlockSize_ = 0;
    int32_t cmpBlockSize_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    uint64_t l2CacheSize_ = 0;

    bool isSameSeqAllKVTensor_ = true;
    bool isSameActualseq_ = true;
    uint32_t maxActualseq_ = 0;

    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType outputType_ = ge::DT_FLOAT16;
};

class SMLAInfoParser {
public:
    explicit SMLAInfoParser(gert::TilingContext *context) : context_(context) {}
    ~SMLAInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus CheckUnrequiredParaExistence() const;

    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
        SMLALayout &layout, const std::string &name) const;
    ge::graphStatus GetActualSeqLenQSize(uint32_t &size);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAttrParaInfo();
    ge::graphStatus GetKvCache();
    ge::graphStatus GetOpParaInfo();

    ge::graphStatus GetInOutDataType();
    ge::graphStatus GetQueryAndOutLayout();
    ge::graphStatus GetKvLayout();
    ge::graphStatus GetSMLATemplateMode(SMLATilingInfo &smlaInfo);
    void SetSMLAShape();
    ge::graphStatus GetN1Size();
    ge::graphStatus GetN2Size();
    ge::graphStatus GetGSize();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetQTSize();
    ge::graphStatus GetKVTSize(); // A2/A3
    ge::graphStatus GetS1Size();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2SizeForTND(); // A2/A3
    ge::graphStatus GetS2Size();
    ge::graphStatus GetMaxBlockNumPerBatch();
    ge::graphStatus GetBlockSize();
    ge::graphStatus GetQHeadDim();
    ge::graphStatus GetValueHeadDim();
    ge::graphStatus GetSparseBlockCount();
    ge::graphStatus GetActualseqInfo();
    ge::graphStatus GetSinks();
    void GenerateInfo(SMLATilingInfo &smlaInfo);
    ge::graphStatus Parse(SMLATilingInfo &smlaInfo);

    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    SMLAParaInfo opParamInfo_;

    bool HasAxis(const SMLAAxis &axis, const SMLALayout &layout, const gert::Shape &shape) const;
    size_t GetAxisIdx(const SMLAAxis &axis, const SMLALayout &layout) const;
    uint32_t GetAxisNum(const gert::Shape &shape, const SMLAAxis &axis, const SMLALayout &layout) const;
    static constexpr int64_t invalidDimValue_ = std::numeric_limits<int64_t>::min();

    uint32_t bSize_ = 0;
    uint32_t n1Size_ = 0;
    uint32_t n2Size_ = 0;
    uint32_t gSize_ = 0;
    uint32_t s1Size_ = 0;
    int64_t s2Size_ = 0;
    int64_t cmpS2Size_ = 0; // A5
    uint32_t headDim_ = 0;
    uint32_t qTSize_ = 0;
    uint32_t orikvTSize_ = 0; // A2/A3
    uint32_t cmpkvTSize_ = 0; // A2/A3
    uint32_t qHeadDim_ = 0;
    uint32_t oriKvHeadDim_ = 0;
    uint32_t cmpKvHeadDim_ = 0;
    int64_t sparseBlockSize_ = 0;
    int64_t oriSparseBlockCount_ = 0; // A5
    int64_t cmpSparseBlockCount_ = 0; // A5
    int64_t sparseBlockCount_ = 0;    // A2/A3
    int64_t oriWinLeft_ = 0;
    int64_t oriWinRight_ = 0;
    uint32_t maxActualseq_ = 0;
    bool isSameSeqAllKVTensor_ = true;
    uint32_t actualLenDimsKV_ = 0;
    uint32_t actualLenDimsQ_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;

    SMLALayout qLayout_ = SMLALayout::TND;
    SMLALayout cmpSparseIndicesLayout_ = SMLALayout::TND;
    SMLALayout oriSparseIndicesLayout_ = SMLALayout::TND;
    SMLALayout outLayout_ = SMLALayout::BSND;
    SMLALayout kvLayout_ = SMLALayout::PA_BBND;

    uint32_t oriMaxBlockNumPerBatch_ = 0;
    uint32_t cmpMaxBlockNumPerBatch_ = 0;
    int32_t oriBlockSize_ = 0;
    int32_t cmpBlockSize_ = 0;

    SMLATemplateMode perfMode_ = SMLATemplateMode::SWA_TEMPLATE_MODE;

    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    ge::DataType qType_ = ge::DT_FLOAT16;
    ge::DataType oriKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpKvType_ = ge::DT_FLOAT16;
    ge::DataType cmpSparseIndicesType_ = ge::DT_INT32;
    ge::DataType oriBlockTableType_ = ge::DT_INT32;
    ge::DataType cmpBlockTableType_ = ge::DT_INT32;
    ge::DataType cuSeqLensQType_ = ge::DT_INT32;
    ge::DataType seqsedKvType_ = ge::DT_INT32;
    ge::DataType sinksType_ = ge::DT_INT32;
    ge::DataType metadataType_ = ge::DT_INT32;
    ge::DataType outputType_ = ge::DT_FLOAT16;

    gert::Shape qShape_{};
    gert::Shape oriKvShape_{};
    gert::Shape cmpKvShape_{};
    gert::Shape oriSparseIndicesShape_{};
    gert::Shape cmpSparseIndicesShape_{};
};

class SparseFlashMlaTilingArch22 {
public:
    explicit SparseFlashMlaTilingArch22(gert::TilingContext *context) : context_(context) {}
    ge::graphStatus DoOpTiling(SMLATilingInfo *tilingInfo);

private:
    void SplitBalanced(SMLATilingInfo *tilingInfo);
    void CalcUbBmm(SMLATilingInfo *tilingInfo);
    gert::TilingContext *context_ = nullptr;
    SMLATemplateMode perfMode_ = SMLATemplateMode::SWA_TEMPLATE_MODE;
    SparseFlashMlaTilingData tilingData_;
    uint32_t blockDim_{0};
    uint64_t workspaceSize_{0};
    uint64_t tilingKey_{0};

    SMLATilingInfo *smlaInfo_ = nullptr;

    size_t mmResUbSize_ = 0;
    size_t bmm2ResUbSize_ = 0;
    uint32_t sInnerLoopTimes_ = 0;
    uint32_t sInnerSize_ = 512; // s2固定切分512
    uint32_t sInnerSizeAlign_ = 0;
    uint32_t usedCoreNum_ = 0;

    uint32_t headDimAlign_ = 0;
    uint32_t mBaseSize_ = 64;
};

} // namespace arch22
} // namespace optiling
#endif
