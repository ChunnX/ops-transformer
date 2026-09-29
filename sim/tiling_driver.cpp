/**
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Appended to the current upstream host implementation by build.py.
// No duplicate Python struct or workspace/peermem size formulas.
#include <fstream>
#include <iostream>
#include <cstddef>
#include "mc2_moe_context.h"

int main(int argc, char** argv)
{
    if (argc != 14)
        return 2;
    MegaMoeTilingData t{};
    t.bs = std::stoul(argv[2]);
    t.h = std::stoul(argv[3]);
    t.hiddenDim = 2 * std::stoul(argv[4]);
    t.epWorldSize = std::stoul(argv[5]);
    t.moeExpertPerRank = std::stoul(argv[6]);
    t.sharedExpertNum = std::stoul(argv[7]);
    t.topK = std::stoul(argv[8]);
    t.aicNum = std::stoul(argv[9]);
    t.blockAivNum = 2 * t.aicNum;
    t.blockNumPerEP = std::max(1U, t.aicNum / t.epWorldSize);
    t.numMaxTokensPerRank = std::stoul(argv[10]);
    t.maxOutputSize = std::stoul(argv[11]);
    if (!t.maxOutputSize)
        t.maxOutputSize = t.numMaxTokensPerRank * t.epWorldSize * std::min(t.topK, t.moeExpertPerRank);
    uint32_t ubBytes = std::stoul(argv[12]);
    t.rankNumPerServer = t.epWorldSize;
    t.moeGmmMode = GMM_MODE_A8W4_NZ;
    t.sharedGmmMode = GMM_MODE_A8W8_ND;
    t.actMode = static_cast<uint8_t>(MegaMoeActMode::SWIGLU);
    t.actSubMode = static_cast<uint8_t>(MegaMoeActSubMode::DEFAULT);
    t.clampLimit = std::numeric_limits<float>::max();
    t.activationAlpha = t.activationBeta = 1.0f;
    t.isPerExpertWeightTensor = false;
    t.isSharedQuantIndependent = false;
    // Default MTE attributes: combine_comm_mode=0, dispatch dedup uses host gates.
    t.dedupMode = 2;
    if (optiling::ApplyDedupPrepassChunkGuard(&t, "mega-moe-sim") != ge::GRAPH_SUCCESS)
        return 6;
    optiling::ApplyDedupBenefitGate(&t, "mega-moe-sim");
    if (optiling::ApplyDedupCombineCapacityGate(&t, "mega-moe-sim") != ge::GRAPH_SUCCESS)
        return 7;
    optiling::SetPrefetchAndWaveParams(&t, t.aicNum);
    t.dispatchBufferConfig = optiling::CalcDispatchBufferConfig(&t, 1, ubBytes);
    t.combineSyncSlotCountPerExpert = optiling::CalcCombineSyncSlotCountPerExpert(&t);
    optiling::SetTopkValidIndexBufferConfigs(&t, 1, 1, false, ubBytes);
    const std::string topkDtype = argv[13];
    if (topkDtype != "float32" && topkDtype != "bfloat16")
        return 5;
    optiling::SetUnpermuteBufferConfigs(&t, topkDtype == "float32" ? ge::DT_FLOAT : ge::DT_BF16, ubBytes);
    t.a8w4L1Layout = optiling::CalcMegaMoeL1Layout(std::max(uint64_t{t.h}, uint64_t{t.hiddenDim / 2}));
    const uint64_t key = GET_TPL_TILING_KEY(C_DT_FLOAT4_E2M1, C_DT_FLOAT8_E4M3FN, 0, 4, 4, 0, 0, 0, 0);
    if (key != 730152 || !UseInt16TopkIndex(t.numMaxTokensPerRank, t.topK))
        return 3;
    PeermemSizeParams p{};
    p.numMaxTokensPerRank = t.numMaxTokensPerRank;
    p.topK = t.topK;
    p.h = t.h;
    p.moeExpertPerRank = t.moeExpertPerRank;
    p.epWorldSize = t.epWorldSize;
    p.yDtypeSize = 2;
    p.elemsPerByte = 1;
    p.topoType = TOPO_TYPE_MTE;
    p.serverNum = 1;
    std::ofstream file(argv[1], std::ios::binary);
    file.write(reinterpret_cast<char*>(&t), sizeof(t));
    if (!file)
        return 4;
    WorkspaceLayout w(&t);
    std::cout << "{\"tiling_key\":" << key << ",\"tiling_bytes\":" << sizeof(t)
              << ",\"workspace_bytes\":" << w.workspaceSize
              << ",\"window_bytes\":" << EXCEPTION_DUMP_REGION_SIZE + CalcPeermemLeastSize(p)
              << ",\"context_bytes\":" << sizeof(Mc2Aclnn::Mc2MoeContext)
              << ",\"context_windows_offset\":" << offsetof(Mc2Aclnn::Mc2MoeContext, epHcclBuffer_)
              << ",\"dedup_mode\":" << t.dedupMode << ",\"max_output_size\":" << t.maxOutputSize
              << ",\"m_groups_per_wave\":" << t.mGroupsPerWave
              << ",\"dispatch_buffers\":" << t.dispatchBufferConfig.bufferCount
              << ",\"l1_a_buffers\":" << t.a8w4L1Layout.aBufferNum << ",\"l1_b_buffers\":" << t.a8w4L1Layout.bBufferNum
              << ",\"l1_scale_k\":" << t.a8w4L1Layout.scaleK << "}" << std::endl;
}
