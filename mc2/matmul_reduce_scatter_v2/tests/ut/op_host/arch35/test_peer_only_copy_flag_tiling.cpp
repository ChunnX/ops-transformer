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
#include <algorithm>
#include <cstdlib>
#include <string>
#include "../matmul_reduce_scatter_v2_host_ut_param.h"
#include "../../../../op_host/op_tiling/arch35/matmul_reduce_scatter_v2_tiling.h"
#include "../../../../op_host/op_tiling/arch35/quant_bmm_reduce_scatter_tiling.h"
#include "mc2_tiling_case_executor.h"
#include "hccl/internal/hccl_tiling_msg.h"

namespace {
using namespace matmul_reduce_scatter_v2_ut;
class PeerOnlyCopyFlagTest : public ::testing::Test {
protected:
    std::string savedDebug;
    bool hadDebug = false;
    void SetUp() override
    {
        const char *value = std::getenv("ASCEND_MC2_DEBUG_MODE");
        hadDebug = value != nullptr;
        if (hadDebug) {
            savedDebug = value;
        }
        setenv("ASCEND_MC2_DEBUG_MODE", "0", 1);
    }
    void TearDown() override
    {
        if (hadDebug) {
            setenv("ASCEND_MC2_DEBUG_MODE", savedDebug.c_str(), 1);
        } else {
            unsetenv("ASCEND_MC2_DEBUG_MODE");
        }
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    }

    void Verify(const std::string &caseName, const std::string &mode, bool expected)
    {
        // Reuse valid shapes without editing the existing CSV or its expected hashes.
        const std::string file = __FILE__;
        const auto csv = file.substr(0, file.find_last_of("/\\") + 1) + "test_matmul_reduce_scatter_v2_tiling.csv";
        const auto cases = GetCasesFromCsv<MatmulReduceScatterV2TilingUtParam>(csv);
        auto found = std::find_if(cases.begin(), cases.end(), [&](const auto &p) { return p.case_name == caseName; });
        ASSERT_NE(found, cases.end());
        auto p = *found;
        p.comm_mode = mode;
        p.is_amax_out = false;
        struct CompileInfo {
        } compileInfo;
        gert::TilingContextPara para("MatmulReduceScatterV2",
                                     {p.x1, p.x2, p.bias, p.x1_scale, p.x2_scale, p.quant_scale}, {p.y, p.amax_out},
                                     {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(p.group)},
                                      {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(p.reduce_op)},
                                      {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(p.is_trans_a)},
                                      {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(p.is_trans_b)},
                                      {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.comm_turn)},
                                      {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.rank_size)},
                                      {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.block_size)},
                                      {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.group_size)},
                                      {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(p.is_amax_out)},
                                      {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(p.y_dtype)},
                                      {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(p.comm_mode)}},
                                     p.inputInstance, p.outputInstance, &compileInfo, p.soc, p.coreNum, p.ubsize);
        Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues({{"rankNum", p.rank_size}});
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(para, info)) << caseName << " mode=" << mode;
        ASSERT_GE(info.tilingDataSize, sizeof(Mc2InitTiling) + sizeof(Mc2CcTiling));
        // Both MMRS tiling layouts start with the same init/communication tiling prefix.
        const auto *cc =
            reinterpret_cast<const HcclApi::Mc2CcTilingInner *>(info.tilingData.get() + sizeof(Mc2InitTiling));
        EXPECT_EQ(cc->skipLocalRankCopy, expected ? 1U : 0U) << caseName << " mode=" << mode;
        if (expected) {
            EXPECT_STREQ(cc->algConfig, "CcuSchedReduceScatterSoleMeshPeerOnly");
        } else {
            EXPECT_STRNE(cc->algConfig, "CcuSchedReduceScatterSoleMeshPeerOnly");
        }
    }

    void Both(const std::string &mode, bool expected)
    {
        Verify("Test2pFullmesh", mode, expected);
        Verify("E4M3_Pertensor_Rank2", mode, expected);
    }
};
TEST_F(PeerOnlyCopyFlagTest, PairedDirectWriteSetsFlag)
{
    Both("ccu_peer_only", true);
}
TEST_F(PeerOnlyCopyFlagTest, OrdinaryCcuKeepsCopy)
{
    Both("ccu", false);
}
TEST_F(PeerOnlyCopyFlagTest, AicpuKeepsDefault)
{
    Both("ai_cpu", false);
}
TEST_F(PeerOnlyCopyFlagTest, OtherRankSizeKeepsCopy)
{
    Verify("BF16_Rank4", "ccu_peer_only", false);
    Verify("E5M2_Pertensor_Rank4", "ccu_peer_only", false);
}
TEST_F(PeerOnlyCopyFlagTest, DebugKeepsDefault)
{
    setenv("ASCEND_MC2_DEBUG_MODE", "1", 1);
    Both("ccu_peer_only", false);
}
TEST_F(PeerOnlyCopyFlagTest, PerBlockKeepsDefault)
{
    Verify("Perblock_E4M3_Rank2", "ccu_peer_only", false);
}
} // namespace
