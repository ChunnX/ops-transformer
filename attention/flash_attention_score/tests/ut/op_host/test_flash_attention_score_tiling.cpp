/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <iomanip>
#include <cstring>
#include <gtest/gtest.h>
#include "../../../op_host/flash_attention_score_tiling_common.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
using namespace std;

class FlashAttentionScoreTilingTrace : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "FlashAttentionScoreTilingTrace SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "FlashAttentionScoreTilingTrace TearDown" << std::endl;
    }
};

TEST_F(FlashAttentionScoreTilingTrace, ScaleValue_0_7_Trace)
{
    std::cout << "====== [SCALE_TRACE] TEST START: scaleValue=0.7 ======" << std::endl;
    float inputFloat = 0.7f;
    double inputDouble = 0.7;
    std::cout << "[SCALE_TRACE] User input: double scaleValue = " << std::setprecision(17) << inputDouble << std::endl;
    std::cout << "[SCALE_TRACE] After double->float cast: float = " << std::setprecision(17) << inputFloat << std::endl;
    uint32_t rawHex;
    std::memcpy(&rawHex, &inputFloat, sizeof(float));
    std::cout << "[SCALE_TRACE] Float hex representation: 0x" << std::hex << rawHex << std::dec << std::endl;
    std::cout << "[SCALE_TRACE] Float bits: ";
    for (int i = 31; i >= 0; i--) {
        std::cout << ((rawHex >> i) & 1);
        if (i == 31 || i == 23) std::cout << " ";
    }
    std::cout << std::endl;

    optiling::FlashAttentionScoreCompileInfo compileInfo = {
        24, 48, 196608, 524288, 131072, 201326592, platform_ascendc::SocVersion::ASCEND910B, NpuArch::DAV_2201};
    gert::TilingContextPara tilingContextPara(
        "FlashAttentionScore",
        {
        {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{256, 256}, {256, 256}}, ge::DT_UINT8, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_INT64, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
        {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1, 1, 256, 8}, {1, 1, 256, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{0, 0, 0, 0}, {0, 0, 0, 0}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{256, 1, 128}, {256, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}
        },
        {
        {"scale_value", Ops::Transformer::AnyValue::CreateFrom<float>(0.7f)},
        {"keep_prob", Ops::Transformer::AnyValue::CreateFrom<float>(1.0f)},
        {"pre_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"next_tockens", Ops::Transformer::AnyValue::CreateFrom<int64_t>(65536)},
        {"head_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"input_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("SBH")},
        {"inner_precise", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"sparse_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"pse_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"seed", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"offset", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"softmax_out_layout", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}
        },
                &compileInfo,"Ascend910B",64,262144,8192);

    TilingInfo tilingInfo;
    bool ret = ExecuteTiling(tilingContextPara, tilingInfo);
    std::cout << "\n====== [SCALE_TRACE] TILING RESULT ======" << std::endl;
    std::cout << "[SCALE_TRACE] ExecuteTiling: " << (ret ? "SUCCESS" : "FAILED") << std::endl;
    if (ret) {
        std::cout << "[SCALE_TRACE] tilingKey = " << tilingInfo.tilingKey << std::endl;
        std::cout << "[SCALE_TRACE] blockNum = " << tilingInfo.blockNum << std::endl;
        std::cout << "[SCALE_TRACE] tilingDataSize = " << tilingInfo.tilingDataSize << std::endl;
        for (size_t i = 0; i < tilingInfo.workspaceSizes.size(); i++) {
            std::cout << "[SCALE_TRACE] workspace[" << i << "] = " << tilingInfo.workspaceSizes[i] << std::endl;
        }

        auto *dataFloat = reinterpret_cast<float*>(tilingInfo.tilingData.get());
        size_t lenFloat = tilingInfo.tilingDataSize / sizeof(float);
        std::cout << "\n[SCALE_TRACE] Scanning tilingData for float values near 0.7:" << std::endl;
        for (size_t i = 0; i < lenFloat; i++) {
            if (dataFloat[i] > 0.69f && dataFloat[i] < 0.71f) {
                uint32_t hexVal; std::memcpy(&hexVal, &dataFloat[i], sizeof(float));
                std::cout << "[SCALE_TRACE]   Found at float-index " << i
                          << " (byte-offset " << i * sizeof(float) << ")"
                          << ": value=" << std::setprecision(17) << dataFloat[i]
                          << ", hex=0x" << std::hex << hexVal << std::dec
                          << std::endl;
            }
        }

        auto *data64 = reinterpret_cast<int64_t*>(tilingInfo.tilingData.get());
        size_t len64 = tilingInfo.tilingDataSize / sizeof(int64_t);
        std::cout << "\n[SCALE_TRACE] First 20 tilingData values (int64_t):" << std::endl;
        for (size_t i = 0; i < len64 && i < 20; i++) {
            std::cout << "[SCALE_TRACE]   [" << i << "] = " << data64[i] << std::endl;
        }
    }
    std::cout << "\n====== [SCALE_TRACE] TEST END ======" << std::endl;
    ASSERT_TRUE(ret);
}
