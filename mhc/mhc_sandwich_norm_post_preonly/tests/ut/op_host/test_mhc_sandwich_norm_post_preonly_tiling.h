/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file test_mhc_sandwich_norm_post_preonly_tiling.h
 * \brief UT test fixture for MhcSandwichNormPostPreonly tiling
 */

#include <iostream>
#include <cmath>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "../../../op_host/mhc_sandwich_norm_post_preonly_tiling.h"

using namespace std;

// Raw buffer layout matching SaveToBuffer output (no CANN class overhead).
// Field order must exactly match TILING_DATA_FIELD_DEF order in
// mhc_sandwich_norm_post_preonly_tiling.h:
// totalTokens, headCount, embedDim, usedCoreNum, rowsPerCore, rowsPerCoreSp,
// blockPivot, dimTile, dimLoop, lastDimTile, phiRows, phiCols,
// normEps, hcEps, invD, invK, useRmsMid, coresPerToken, myHeadCount,
// maxTokensPerPair, syncGmOffsetFloats, returnHInF32(int8_t)
struct RawTilingData {
    uint32_t totalTokens;
    uint32_t headCount;
    uint32_t embedDim;
    uint32_t usedCoreNum;
    uint32_t rowsPerCore;
    uint32_t rowsPerCoreSp;
    uint32_t blockPivot;
    uint32_t dimTile;
    uint32_t dimLoop;
    uint32_t lastDimTile;
    uint32_t phiRows;
    uint32_t phiCols;
    float normEps;
    float hcEps;
    float invD;
    float invK;
    uint32_t useRmsMid;
    uint32_t coresPerToken;
    uint32_t myHeadCount;
    uint32_t maxTokensPerPair;
    uint32_t syncGmOffsetFloats;
    int8_t returnHInF32;
};

class MhcSandwichNormPostPreonlyTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MhcSandwichNormPostPreonlyTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MhcSandwichNormPostPreonlyTiling TearDown" << std::endl;
    }

    /**
     * @brief Build TilingContextPara for TND (3D) format: h_out=[T,D], residual=[T,N,D], etc.
     */
    static gert::TilingContextPara BuildTilingContextTND(
        uint32_t T, uint32_t N, uint32_t D, ge::DataType xDtype, bool useGamma2 = true,
        float normEps = 1e-6f, float hcEps = 1e-6f, bool returnHInF32 = false)
    {
        static optiling::MhcSandwichNormPostPreonlyCompileInfo compileInfo = {};

        std::vector<gert::TilingContextPara::TensorDescription> inputs = {
            {{{T, D}, {T, D}}, xDtype, ge::FORMAT_ND},
            {{{T, N, D}, {T, N, D}}, xDtype, ge::FORMAT_ND},
            {{{T, N}, {T, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{T, N, N}, {T, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{N, (int64_t)N * D}, {N, (int64_t)N * D}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{D}, {D}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{D}, {D}}, ge::DT_FLOAT, ge::FORMAT_ND},
        };

        if (useGamma2) {
            inputs.push_back({{{D}, {D}}, ge::DT_FLOAT, ge::FORMAT_ND});
        } else {
            inputs.push_back({{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND});
        }

        return gert::TilingContextPara(
            "MhcSandwichNormPostPreonly",
            inputs,
            {
                {{{T, D}, {T, D}}, xDtype, ge::FORMAT_ND},
                {{{T, N, D}, {T, N, D}}, xDtype, ge::FORMAT_ND},
            },
            {
                {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
                {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)},
                {"return_h_in_f32", Ops::Transformer::AnyValue::CreateFrom<bool>(returnHInF32)},
            },
            &compileInfo);
    }

    /**
     * @brief Build TilingContextPara for BSND (4D) format: h_out=[B,S,D], residual=[B,S,N,D], etc.
     */
    static gert::TilingContextPara BuildTilingContextBSND(
        uint32_t B, uint32_t S, uint32_t N, uint32_t D, ge::DataType xDtype, bool useGamma2 = true,
        float normEps = 1e-6f, float hcEps = 1e-6f, bool returnHInF32 = false)
    {
        static optiling::MhcSandwichNormPostPreonlyCompileInfo compileInfo = {};

        std::vector<gert::TilingContextPara::TensorDescription> inputs = {
            {{{B, S, D}, {B, S, D}}, xDtype, ge::FORMAT_ND},
            {{{B, S, N, D}, {B, S, N, D}}, xDtype, ge::FORMAT_ND},
            {{{B, S, N}, {B, S, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{B, S, N, N}, {B, S, N, N}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{N, (int64_t)N * D}, {N, (int64_t)N * D}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{D}, {D}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{D}, {D}}, ge::DT_FLOAT, ge::FORMAT_ND},
        };

        if (useGamma2) {
            inputs.push_back({{{D}, {D}}, ge::DT_FLOAT, ge::FORMAT_ND});
        } else {
            inputs.push_back({{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND});
        }

        return gert::TilingContextPara(
            "MhcSandwichNormPostPreonly",
            inputs,
            {
                {{{B, S, D}, {B, S, D}}, xDtype, ge::FORMAT_ND},
                {{{B, S, N, D}, {B, S, N, D}}, xDtype, ge::FORMAT_ND},
            },
            {
                {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(normEps)},
                {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(hcEps)},
                {"return_h_in_f32", Ops::Transformer::AnyValue::CreateFrom<bool>(returnHInF32)},
            },
            &compileInfo);
    }

    /**
     * @brief Run tiling and return TilingData for field-level validation.
     * Returns nullptr if tiling fails (caller should check).
     */
    static const RawTilingData* RunTilingGetData(
        gert::TilingContextPara& para, TilingInfo& info)
    {
        bool ok = ExecuteTiling(para, info);
        if (!ok) return nullptr;
        return reinterpret_cast<const RawTilingData*>(info.tilingData.get());
    }

    /**
     * @brief Run a tiling test case (pass/fail only, no field validation)
     */
    static void RunTilingTest(gert::TilingContextPara& para, bool expectFail)
    {
        ge::graphStatus expectStatus = expectFail ? ge::GRAPH_FAILED : ge::GRAPH_SUCCESS;
        int64_t expectTilingKey = 0;
        std::string expectTilingDataStr = "";
        std::vector<size_t> expectWorkspaces = expectFail ? std::vector<size_t>{0} : std::vector<size_t>{};
        ExecuteTestCase(para, expectStatus, expectTilingKey, expectTilingDataStr, expectWorkspaces);
    }

    /**
     * @brief Build mixed-format context: h_out 2D [T,D] but residual 4D [B,S,N,D]
     */
    static gert::TilingContextPara BuildMixed_Hout2D_Res4D()
    {
        static optiling::MhcSandwichNormPostPreonlyCompileInfo compileInfo = {};
        // h_out=[4,2560] (2D) but residual=[1,4,4,2560] (4D) — mismatch
        return gert::TilingContextPara(
            "MhcSandwichNormPostPreonly",
            {
                {{{4, 2560}, {4, 2560}}, ge::DT_BF16, ge::FORMAT_ND},           // h_out: 2D
                {{{1, 4, 4, 2560}, {1, 4, 4, 2560}}, ge::DT_BF16, ge::FORMAT_ND}, // residual: 4D (should be 3D)
                {{{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{4, 4, 4}, {4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{4, 10240}, {4, 10240}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{2560}, {2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{2560}, {2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{2560}, {2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {
                {{{4, 2560}, {4, 2560}}, ge::DT_BF16, ge::FORMAT_ND},
                {{{4, 4, 2560}, {4, 4, 2560}}, ge::DT_BF16, ge::FORMAT_ND},
            },
            {
                {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
            },
            &compileInfo);
    }

    /**
     * @brief Build mixed-format context: h_out 3D [B,S,D] but residual 3D [T,N,D]
     */
    static gert::TilingContextPara BuildMixed_Hout3D_Res3D()
    {
        static optiling::MhcSandwichNormPostPreonlyCompileInfo compileInfo = {};
        // h_out=[1,4,2560] (3D) but residual=[4,4,2560] (3D) — mismatch, should be 4D
        return gert::TilingContextPara(
            "MhcSandwichNormPostPreonly",
            {
                {{{1, 4, 2560}, {1, 4, 2560}}, ge::DT_BF16, ge::FORMAT_ND},     // h_out: 3D
                {{{4, 4, 2560}, {4, 4, 2560}}, ge::DT_BF16, ge::FORMAT_ND},      // residual: 3D (should be 4D)
                {{{1, 4, 4}, {1, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{1, 4, 4, 4}, {1, 4, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{4, 10240}, {4, 10240}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{2560}, {2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{2560}, {2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
                {{{2560}, {2560}}, ge::DT_FLOAT, ge::FORMAT_ND},
            },
            {
                {{{1, 4, 2560}, {1, 4, 2560}}, ge::DT_BF16, ge::FORMAT_ND},
                {{{1, 4, 4, 2560}, {1, 4, 4, 2560}}, ge::DT_BF16, ge::FORMAT_ND},
            },
            {
                {"norm_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
                {"hc_eps", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
            },
            &compileInfo);
    }

    template <typename T>
    static string TilingData2Str(void* buf, size_t size)
    {
        string result;
        const T* data = reinterpret_cast<const T*>(buf);
        size_t len = size / sizeof(T);
        for (size_t i = 0; i < len; i++) {
            result += std::to_string(data[i]);
            result += " ";
        }
        return result;
    }
};
