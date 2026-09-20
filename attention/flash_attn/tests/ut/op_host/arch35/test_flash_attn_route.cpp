/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../../../../op_host/fa_adjust_sinner_souter.h"

namespace FlashAttnUT {
namespace {
namespace tiling = optiling::flash_attn::fa_tiling_util;

TEST(FlashAttnRoute, EligibleDimensionsShareMaskAndNoMaskRoute)
{
    const uint32_t dimensions[][2] = {{64, 64}, {72, 72}, {128, 128}, {192, 128}};
    for (const auto &dims : dimensions) {
        for (bool hasMask : {false, true}) {
            EXPECT_EQ(tiling::GetFlashAttnTemplateId(dims[0], dims[1], 128, hasMask, true), 2U);
            EXPECT_EQ(tiling::GetFlashAttnTemplateId(dims[0], dims[1], 128, hasMask, false), hasMask ? 0U : 1U);
        }
    }
}

TEST(FlashAttnRoute, ExcludedBlocksKeepLegacyRouteWithGateOnOrOff)
{
    const uint32_t dimensions[][2] = {{64, 64}, {72, 72}, {128, 128}, {192, 128}, {256, 256}};
    for (const auto &dims : dimensions) {
        for (uint32_t cubeM : {64U, 128U}) {
            if (cubeM == 128 && dims[0] != 256) {
                continue;
            }
            for (bool gate : {false, true}) {
                for (bool hasMask : {false, true}) {
                    EXPECT_EQ(tiling::GetFlashAttnTemplateId(dims[0], dims[1], cubeM, hasMask, gate), 0U);
                }
            }
        }
    }
    EXPECT_EQ(tiling::GetFlashAttnTemplateId(64, 128, 128, false, true), 0U);
    EXPECT_EQ(tiling::GetFlashAttnTemplateId(192, 192, 128, true, true), 0U);
}

TEST(FlashAttnRoute, DefaultGateRemainsDisabled)
{
    EXPECT_FALSE(tiling::DN_UNMERGED_ROUTE_ENABLED);
    EXPECT_EQ(tiling::GetFlashAttnTemplateId(64, 64, 128, false), 1U);
    EXPECT_EQ(tiling::GetFlashAttnTemplateId(64, 64, 128, true), 0U);
}

TEST(FlashAttnRoute, CoreCapUsesUnmergedHeadTasksOnlyForTemplate2)
{
    EXPECT_EQ(tiling::GetMaxUsedAicCores(28, 1, 1, 8, 65, 256, 128, 256), 5U);
    EXPECT_EQ(tiling::GetMaxUsedAicCores(28, 1, 1, 8, 65, 256, 128, 256, true), 8U);
    for (bool unmerged : {false, true}) {
        EXPECT_EQ(tiling::GetMaxUsedAicCores(28, 1, 1, UINT32_MAX, INT64_MAX, INT64_MAX, 128, 256, unmerged), 28U);
        EXPECT_EQ(tiling::GetMaxUsedAicCores(28, 1, 1, 8, -1, 256, 128, 256, unmerged), 28U);
        EXPECT_EQ(tiling::GetMaxUsedAicCores(28, 1, 1, 8, 0, 256, 128, 256, unmerged), 28U);
    }
    for (uint32_t g : {1U, 4U, 8U, 32U}) {
        for (int64_t q : {1, 64, 65, 128, 129, 4096}) {
            for (int64_t kv : {1, 207, 208, 209, 223, 224, 225, 256, 257, 4096}) {
                for (uint32_t n : {128U, 208U, 224U, 256U}) {
                    const uint64_t expected = 2U * 3U * g * ((q + 127U) / 128U) * ((kv + n - 1U) / n);
                    EXPECT_EQ(tiling::GetMaxUsedAicCores(28, 2, 3, g, q, kv, 128, n, true),
                              expected < 28 ? expected : 28U);
                }
            }
        }
    }
}

struct RouteCase {
    uint32_t d;
    uint32_t dv;
    uint32_t group;
    int64_t q;
    int64_t kv;
    int32_t mask;
    int64_t left;
    int64_t right;
    uint32_t expectedM;
    uint32_t expectedN;
    uint32_t expectedTemplate;
};

TEST(FlashAttnRoute, MultimodalPrefillAndShortQueryBoundaries)
{
    const RouteCase cases[] = {
        {64, 64, 1, 197, 197, 0, -1, -1, 128, 128, 2},     // Vision tail, no mask.
        {72, 72, 4, 257, 513, 1, -1, -1, 128, 128, 2},     // Multimodal, actual D72.
        {128, 128, 8, 2048, 2048, 3, -1, -1, 128, 128, 2}, // Causal GQA prefill.
        {192, 128, 16, 129, 4096, 4, 255, 0, 128, 128, 2}, // Windowed MLA prefill.
        {64, 64, 1, 64, 129, 0, 0, 0, 64, 256, 0},         // Ignored no-mask windows.
        {64, 64, 8, 64, 129, 3, 0, 0, 64, 256, 0},         // Ignored causal windows.
        {64, 64, 8, 65, 129, 3, 0, 0, 128, 128, 2},
        {64, 64, 1, 64, 128, 0, -1, -1, 128, 128, 2},   // Short actual Q is not M64.
        {128, 128, 4, 1, 4096, 2, -1, -1, 128, 128, 2}, // Mask mode 2 retains M128.
        {128, 128, 4, 64, 256, 4, 128, 0, 128, 128, 2},
        {128, 128, 4, 64, 256, 4, 129, 0, 64, 256, 0},
        {128, 128, 4, 64, 256, 4, -1, 0, 64, 256, 0},
        {128, 128, 4, 64, 256, 1, -2, -1, 64, 256, 0}, // Signed windows.
        {128, 128, 4, -1, -1, 3, -1, -1, 128, 128, 2},
        {256, 256, 1, 63, 512, 0, -1, -1, 64, 256, 0},
        {256, 256, 1, 64, 512, 0, -1, -1, 128, 128, 0},
    };
    for (const auto &item : cases) {
        for (uint32_t layout : {tiling::LAYOUT_BSH, tiling::LAYOUT_BSND, tiling::LAYOUT_BNSD, tiling::LAYOUT_TND}) {
            SCOPED_TRACE(testing::Message() << "D=" << item.d << " Q=" << item.q << " mask=" << item.mask
                                            << " layout=" << layout << " left=" << item.left);
            uint32_t outer = 0;
            uint32_t inner = 0;
            tiling::AdjustSinnerAndSouter(item.dv, item.group, item.q, item.kv, item.mask, item.left, item.right,
                                          layout, outer, inner);
            EXPECT_EQ(outer * 2, item.expectedM);
            EXPECT_EQ(inner, item.expectedN);
            EXPECT_EQ(tiling::GetFlashAttnTemplateId(item.d, item.dv, outer * 2, item.mask != 0, true),
                      item.expectedTemplate);
        }
    }
}
} // namespace
} // namespace FlashAttnUT
