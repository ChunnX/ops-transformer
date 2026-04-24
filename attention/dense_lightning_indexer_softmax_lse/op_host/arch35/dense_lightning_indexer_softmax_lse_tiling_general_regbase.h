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
 * \file dense_lightning_indexer_softmax_lse_tiling_general_regbase.h
 * \brief Arch35 tiling implementation for dense_lightning_indexer_softmax_lse
 */

#ifndef DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TILING_GENERAL_REGBASE_H
#define DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TILING_GENERAL_REGBASE_H

#include <numeric>
#include <algorithm>
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_type.h"
#include "../dense_lightning_indexer_softmax_lse_tiling.h"
#include "../../op_kernel/arch35/dense_lightning_indexer_softmax_lse_template_tiling_key_regbase.h"
#include "../../op_kernel/arch35/dense_lightning_indexer_softmax_lse_tiling_regbase.h"
#include "err/ops_err.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

class DenseLightningIndexerSoftmaxLseTilingBaseRegbase : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit DenseLightningIndexerSoftmaxLseTilingBaseRegbase(gert::TilingContext *context) : TilingBaseClass(context)
    {
        Reset();
    }
    ~DenseLightningIndexerSoftmaxLseTilingBaseRegbase() override = default;

    void Reset(gert::TilingContext *context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    void Reset() {
        bSize = 0;
        n2Size = 0;
        gSize = 0;
        s1Size = 0;
        s2Size = 0;
        dSize = 128;
        kSize = 0;
        s1BaseSize = 8;
        s2BaseSize = 512;
        mBaseSize = 512;
        sparseMode = 3;
        preTokens = INT64_MAX;
        nextTokens = INT64_MAX;
        scaleValue = 1.0f;
        layoutType = 0;
    }

    [[nodiscard]] gert::TilingContext *GetContext()
    {
        return context_;
    }
    bool IsCapable()
    {
        return true;
    }

    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;

    void GetActualSeqLenData(int64_t inputIdx, std::vector<int64_t> &res, int64_t &actualLen) const;
    bool AnalyzeAttrs();
    bool AnalyzeDimLayout(const gert::Shape &queryShape, const gert::Shape &keyShape,
                         const gert::Shape &weightsShape, size_t layoutLen);
    bool AnalyzeDtype();
    bool AnalyzeLayout();
    int64_t GetS2RealSize(int32_t sparseMode, int32_t s1Size, int32_t s2Size, int32_t s1Idx);
    bool InitSparseValidArray(std::vector<int64_t> &sparseValidArray);
    bool BalanceLoad(const std::vector<int64_t> &sparseValidArray,
                     std::vector<int64_t> &localValue, std::vector<int64_t> &sparseStartIdx);
    bool SetSparseStartIdx(const std::vector<int64_t> &sparseValidArray, int64_t maxCoreNum);
    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum);
    int64_t CalcTotalSize();

    int32_t bSize;
    int32_t n2Size;
    int32_t gSize;
    int32_t s1Size;
    int32_t s2Size;
    int32_t dSize;
    int32_t kSize;
    int32_t s1BaseSize;
    int32_t s2BaseSize;
    int32_t mBaseSize;
    int32_t sparseMode;
    int64_t preTokens;
    int64_t nextTokens;
    float scaleValue;
    uint8_t layoutType;

    LayoutTypeRegbase tilingKeyLayout;
    uint32_t aivNum;
    uint32_t aicNum;
    uint64_t l2CacheSize;

    int64_t realT1Size;
    int64_t maxS1Val;
    int64_t maxS2Val;

    platform_ascendc::SocVersion socVersion;
    const char *opName;
    const char *inputLayout;

    std::vector<int64_t> actualSeqLenData;
    std::vector<int64_t> actualSeqLenKData;

    DenseLightningIndexerSoftmaxLseRegBaseTilingData *tilingData = 
        context_->GetTilingData<DenseLightningIndexerSoftmaxLseRegBaseTilingData>();
    DenseLISoftmaxLseBaseParamsRegbase *baseParams_ = &tilingData->baseParams;
    DenseLISoftmaxLseMultiCoreParamsRegbase *multiCoreParams_ = &tilingData->multiCoreParams;
};

struct DenseLightningIndexerSoftmaxLseCompileInfo {};

} // namespace optiling

#endif // DENSE_LIGHTNING_INDEXER_SOFTMAX_LSE_TILING_GENERAL_REGBASE_H