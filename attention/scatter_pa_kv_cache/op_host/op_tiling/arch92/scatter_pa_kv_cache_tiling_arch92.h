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
 * \file scatter_pa_kv_cache_tiling_arch92.h
 * \brief Tiling implementation of scatter_pa_kv_cache for Ascend 960DT (arch92)
 */

#ifndef SCATTER_PA_KV_CACHE_TILING_ARCH92_H
#define SCATTER_PA_KV_CACHE_TILING_ARCH92_H

#include "../../scatter_pa_kv_cache_tiling.h"

namespace optiling {

// Ascend 960DT (arch92) tiling实现，950/350沿用 scatter_pa_kv_cache_tiling_arch35.cpp
class ScatterPaKvCacheTiling92 : public TilingBaseClass {
public:
    explicit ScatterPaKvCacheTiling92(gert::TilingContext* context, int64_t inOutMode)
        : TilingBaseClass(context),
          inOutMode_(inOutMode)
    {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus TemplateNormal();
    ge::graphStatus TemplateRope();
    ge::graphStatus TemplateAlibi();
    ge::graphStatus TemplateOmni();
    ge::graphStatus TemplateNZ();
    ge::graphStatus GetIndexDtype();
    ge::graphStatus GetInputDtype();
    ge::graphStatus CheckDimValid();
    ge::graphStatus CheckNormal();
    ge::graphStatus CheckRope();
    ge::graphStatus CheckAlibi();
    ge::graphStatus CheckNz();
    ge::graphStatus GetTemplateType(int64_t inputKeyDimNum);
    ge::graphStatus GetContiguousTensorInfo(gert::Shape& shape, gert::Stride& stride, size_t idx);
    void GetNonContigousStrideInfo();
    ge::graphStatus GetTensorInfo(gert::Shape& shape, gert::Stride& stride, size_t idx);
    void SetInputPos();
    void GetCommonTilingInfo();
    ge::graphStatus CheckSlotMappingShape(int64_t requiredDimNum);
    int64_t RoundUp(int64_t x, int64_t dtypeSize);
    void GenTilingKey();
    bool IsNonContiguous();
    bool IsAxesContiguous(const gert::Stride& stride, const gert::Shape& shape, int64_t startAxis, int64_t endAxis);
    bool IsLastAxisContiguous(const gert::Stride& stride, const gert::Shape& shape);

private:
    platform_ascendc::SocVersion socVersion_;
    bool isAscend96_ = false;
    int64_t ubSize_ = 0;
    int64_t totalCoreNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t blockFactor_;
    int64_t tailBlockFactor_ = 0;
    int64_t kHandleNumPerCore_ = 0;
    int64_t vHandleNumPerCore_ = 0;
    int64_t kLoopNum_ = 0;
    int64_t vLoopNum_ = 0;
    int64_t kHandleNumPerLoop_ = 0;
    int64_t vHandleNumPerLoop_ = 0;
    int64_t kTailHandleNum_ = 0;
    int64_t vTailHandleNum_ = 0;
    int64_t numTokens_ = 0;
    int64_t dtypeByteSize_ = 0;
    int64_t valueDtypeByteSize_ = 0;
    int64_t keyStride0_ = 0;
    int64_t keyStride1_ = 0;
    int64_t keyStride2_ = 0;
    int64_t valueStride0_ = 0;
    int64_t valueStride1_ = 0;
    int64_t valueStride2_ = 0;
    int64_t keyCacheStride0_ = 0;
    int64_t keyCacheStride1_ = 0;
    int64_t keyCacheStride2_ = 0;
    int64_t keyCacheStride3_ = 0;
    int64_t valueCacheStride0_ = 0;
    int64_t valueCacheStride1_ = 0;
    int64_t valueCacheStride2_ = 0;
    int64_t valueCacheStride3_ = 0;
    int64_t kHeadSize_;
    int64_t vHeadSize_;
    int64_t batch_;
    int64_t numHead_;
    int64_t numBlocks_;
    int64_t blockSize_;
    int64_t seqLen_;
    uint64_t tilingKey_;
    int64_t templateType_ = 0;   // 1: normal, 2: rope, 3: alibi
    int64_t indexDtypeSize_ = 0; // int32 -> 4, int64 -> 8
    int64_t isFullyLoad_ = 0;    // 0: not fully load, 1: fully load
    int64_t inOutMode_;          // 1: single input and output，2: dual input and output
    int64_t inputKey_;
    int64_t inputKeyCacheIn_;
    int64_t inputSlotMapping_;
    int64_t inputValue_;
    int64_t inputValueCacheIn_;
    int64_t inputCompressLens_;
    int64_t inputCompressSeqOffset_;
    int64_t inputSeqLens_;
    int64_t kStride_ = 0;
    int64_t vStride_ = 0;
    int64_t kOffset_ = 0;
    int64_t vOffset_ = 0;
    int64_t offsetTypeSize_ = 0;
    int64_t ubBlockFactor_ = 0;

    gert::Shape inputKeyShape_;
    gert::Shape inputKeyCacheInShape_;
    gert::Shape slotMappingShape_;
    gert::Shape inputValueShape_;
    gert::Shape inputValueCacheInShape_;
    gert::Shape compressLensShape_;
    gert::Shape compressSeqOffsetShape_;
    gert::Shape seqLensShape_;

    gert::Stride keyStride_;
    gert::Stride valueStride_;
    gert::Stride keyCacheStride_;
    gert::Stride valueCacheStride_;

    ge::DataType inputDtype_;
    ge::DataType valueDtype_;
    ScatterPaKvCacheTilingData tilingData_;
};

} // namespace optiling

#endif // SCATTER_PA_KV_CACHE_TILING_ARCH92_H
