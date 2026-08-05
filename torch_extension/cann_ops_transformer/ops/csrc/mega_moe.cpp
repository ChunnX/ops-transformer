// -----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------

#include <torch/extension.h>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include "aclnn_common.h"

namespace op_api {
using npu_utils = at_npu::native::NpuUtils;
const int DIM_TWO = 2;

// ════════════════════════════════════════════════════════════════════════════════
// [PROFILER] torch 路径打点回读（子包路径，见 mc2/mega_moe/tests/profiler/README.md）
//   ACLNN_CMD 宏内部的 workspace 是宏作用域局部变量，宏结束后即释放，Python 层读不到 profBuf。
//   因此 MEGA_MOE_PROF_DUMP=1 时走下面手动两阶段调用，显式保留 workspace 直到 D2H 回读完成。
//
//   两阶段工作流：
//     ① 标定：MEGA_MOE_PROF_CALIB=1 跑一次，kernel(address-bridge) 把 workspace 基址写 yOut[0..8]，
//        host 读出 delta = kernelWsBase - hostWsBase，打印后按 delta 填 MEGA_MOE_PROF_DELTA。
//     ② 打点：MEGA_MOE_PROF_DUMP=1（不带 CALIB），用 MEGA_MOE_PROF_DELTA 从 workspace 尾部读 profBuf，
//        写 prof_rank{device}.bin。
//   PROF_N_CORES 须与 op_kernel/arch35/profiler_config.h 一致（Ascend 950 MIX_1_2 = 84）。
// ════════════════════════════════════════════════════════════════════════════════
constexpr uint32_t MEGA_MOE_PROF_N_CORES = 84;  // [MODIFY] 与 profiler_config.h PROF_N_CORES 一致
constexpr uint32_t MEGA_MOE_PROF_SLOTS_PER_CORE = 4096;
constexpr uint32_t MEGA_MOE_PROF_RECORD_BYTES = 16;
constexpr uint32_t MEGA_MOE_PROF_RINGIDX_ALIGN = 64;
constexpr size_t MEGA_MOE_PROF_BUF_TOTAL =
    (size_t)MEGA_MOE_PROF_N_CORES * MEGA_MOE_PROF_RINGIDX_ALIGN +
    (size_t)MEGA_MOE_PROF_N_CORES * MEGA_MOE_PROF_SLOTS_PER_CORE * MEGA_MOE_PROF_RECORD_BYTES;

// 从 workspace 尾部读 profBuf 写 prof_rank{deviceId}.bin（MIX_1_2 → layout_type=3）
inline void DumpMegaMoeProfBuf(void *workspaceAddr, uint64_t workspaceSize, int deviceId, uint64_t wsDelta)
{
    if (workspaceSize < MEGA_MOE_PROF_BUF_TOTAL) {
        ASCEND_LOGW("MegaMoe profBuf: workspaceSize %llu < PROF_BUF_TOTAL %zu, skip dump.",
                    (unsigned long long)workspaceSize, MEGA_MOE_PROF_BUF_TOTAL);
        return;
    }
    uint64_t profBufOff = workspaceSize - MEGA_MOE_PROF_BUF_TOTAL;
    uint8_t *profDevPtr = reinterpret_cast<uint8_t *>(workspaceAddr) + wsDelta + profBufOff;

    std::vector<uint8_t> hostBuf(MEGA_MOE_PROF_BUF_TOTAL);
    aclError ret = aclrtMemcpy(hostBuf.data(), MEGA_MOE_PROF_BUF_TOTAL, profDevPtr, MEGA_MOE_PROF_BUF_TOTAL,
                               ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        ASCEND_LOGW("MegaMoe profBuf D2H failed, ret=%d.", ret);
        return;
    }

    char fname[256];
    snprintf(fname, sizeof(fname), "prof_rank%d.bin", deviceId);
    std::ofstream ofs(fname, std::ios::binary);
    uint32_t header[4] = {0x50524F46u, (3u << 16) | 1u, MEGA_MOE_PROF_N_CORES, MEGA_MOE_PROF_RECORD_BYTES};
    ofs.write(reinterpret_cast<const char *>(header), sizeof(header));
    ofs.write(reinterpret_cast<const char *>(hostBuf.data()), MEGA_MOE_PROF_BUF_TOTAL);
    ofs.close();
    uint32_t firstWord = 0;
    memcpy(&firstWord, hostBuf.data(), 4);
    printf("[PROFILER] MegaMoe rank %d dumped to %s (first word 0x%08X)\n", deviceId, fname, firstWord);
}

// 手动两阶段调用：与 ACLNN_CMD 等价，但保留 workspace 直到回读完成
template <typename... Ts>
void LaunchMegaMoeWithProfDump(const at::Tensor &yOut, int deviceId, Ts &...args)
{
    auto device = DecodeDevice(args...);
    const c10::OptionalDeviceGuard device_guard(device);
    static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr("aclnnMegaMoeGetWorkspaceSize");
    static const auto opApiFuncAddr = GetOpApiFuncAddr("aclnnMegaMoe");
    static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");
    static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");
    static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");
    TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,
                "aclnnMegaMoe / aclnnMegaMoeGetWorkspaceSize not found.");
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
    uint64_t workspace_size = 0;
    uint64_t *workspace_size_addr = &workspace_size;
    aclOpExecutor *executor = nullptr;
    aclOpExecutor **executor_addr = &executor;
    InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);
    UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);
    ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);
    if (initMemFunc) {
        initMemFunc(nullptr, false);
    }
    ApplyDeterministicConfig();
    auto converted_params = ConvertTypes(args..., workspace_size_addr, executor_addr);
    static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);
    auto workspace_status = call(getWorkspaceSizeFunc, converted_params);
    if (workspace_status != 0) {
        ReleaseConvertTypes(converted_params);
        TORCH_CHECK(false, "call aclnnMegaMoeGetWorkspaceSize failed, detail:", aclGetRecentErrMsg());
    }
    at::Tensor workspace_tensor;
    void *workspace_addr = nullptr;
    if (workspace_size != 0) {
        at::TensorOptions options = at::TensorOptions(torch_npu::utils::get_npu_device_type());
        workspace_tensor = at::empty({workspace_size}, options.dtype(at::kByte));
        workspace_addr = const_cast<void *>(workspace_tensor.storage().data());
    }
    auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]() -> int {
        typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *, const aclrtStream);
        OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);
        auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);
        ReleaseConvertTypes(converted_params);
        if (releaseMemFunc) {
            releaseMemFunc(nullptr, false);
        }
        TORCH_CHECK(api_ret == 0, "call aclnnMegaMoe failed, detail:", aclGetRecentErrMsg());
        return api_ret;
    };
    at_npu::native::OpCommand cmd;
    cmd.Name("aclnnMegaMoe");
    cmd.SetCustomHandler(acl_call);
    cmd.Run();
    if (unInitMemFunc) {
        unInitMemFunc(nullptr, false);
    }

    // ── 同步后回读 ──
    (void)aclrtSynchronizeStreamWithTimeout(acl_stream, 30000);
    const char *calib = getenv("MEGA_MOE_PROF_CALIB");
    const char *dump = getenv("MEGA_MOE_PROF_DUMP");
    uint64_t hostWsBase = reinterpret_cast<uint64_t>(workspace_addr);
    if (calib != nullptr && std::strcmp(calib, "1") == 0) {
        // 标定：kernel(address-bridge) 把 workspace 基址写到 yOut[0..8]
        uint64_t kernelWsBase = 0;
        aclError ret = aclrtMemcpy(&kernelWsBase, 8, const_cast<void *>(yOut.storage().data()), 8,
                                   ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_ERROR_NONE) {
            ASCEND_LOGW("MegaMoe calib read bridge failed, ret=%d.", ret);
        } else {
            uint64_t delta = kernelWsBase - hostWsBase;
            printf("[PROFILER] MegaMoe rank %d calib: kernelWsBase=0x%llX hostWsBase=0x%llX delta=%llu (0x%llX)\n",
                   deviceId, (unsigned long long)kernelWsBase, (unsigned long long)hostWsBase,
                   (unsigned long long)delta, (unsigned long long)delta);
            printf("[PROFILER] Set MEGA_MOE_PROF_DELTA=%llu and rerun with MEGA_MOE_PROF_DUMP=1 (no CALIB).\n",
                   (unsigned long long)delta);
        }
    } else if (dump != nullptr && std::strcmp(dump, "1") == 0) {
        const char *deltaStr = getenv("MEGA_MOE_PROF_DELTA");
        uint64_t wsDelta = 0;
        if (deltaStr != nullptr) {
            wsDelta = static_cast<uint64_t>(std::strtoull(deltaStr, nullptr, 0));
        }
        DumpMegaMoeProfBuf(workspace_addr, workspace_size, deviceId, wsDelta);
    }
}

std::tuple<at::Tensor, at::Tensor>
NpuMegaMoe(const at::Tensor &context, const at::Tensor &x, const at::Tensor &topkIds, const at::Tensor &topkWeights,
           const std::vector<at::Tensor> &weight1, const std::vector<at::Tensor> &weight2, int64_t moeExpertNum,
           int64_t epWorldSize, int64_t cclBufferSize, const c10::optional<std::vector<at::Tensor>> &weightScales1,
           const c10::optional<std::vector<at::Tensor>> &weightScales2,
           const c10::optional<std::vector<at::Tensor>> &bias1, const c10::optional<std::vector<at::Tensor>> &bias2,
           const c10::optional<at::Tensor> &xActiveMask, const c10::optional<std::vector<at::Tensor>> &sharedWeight1,
           const c10::optional<std::vector<at::Tensor>> &sharedWeight2,
           const c10::optional<std::vector<at::Tensor>> &sharedWeightScales1,
           const c10::optional<std::vector<at::Tensor>> &sharedWeightScales2,
           const c10::optional<std::vector<at::Tensor>> &sharedBias1,
           const c10::optional<std::vector<at::Tensor>> &sharedBias2, int64_t maxRecvTokenNum,
           int64_t dispatchQuantMode, int64_t combineQuantMode, std::string commAlg, int64_t numMaxTokensPerRank,
           std::string activation, c10::optional<float> activationClamp, c10::optional<int64_t> dispatchQuantOutDtype,
           c10::optional<int64_t> weight1Type, c10::optional<int64_t> weight2Type, c10::optional<int64_t> topoType,
           c10::optional<int64_t> rankNumPerServer, int64_t topkWeightsType)
{
    TORCH_CHECK((epWorldSize > 0), "The ep_world_sizes should be greater than 0, current is: ", epWorldSize);
    TORCH_CHECK((x.dim() == DIM_TWO) && (topkIds.dim() == DIM_TWO), "The x and topk_ids should be 2D");
    TORCH_CHECK(((x.scalar_type() == at::kBFloat16) || (x.scalar_type() == at::kHalf)) &&
                    (topkIds.scalar_type() == at::kInt),
                "dtype of x should be bfloat16, float16, dtype of topk_ids should be int.");

    at::TensorList weight1Ref = weight1;
    at::TensorList weight2Ref = weight2;

    auto toTensorList = [](const c10::optional<std::vector<at::Tensor>> &opt) -> at::TensorList {
        return opt.has_value() ? at::TensorList(opt.value()) : at::TensorList();
    };
    at::TensorList weightScales1Ref = toTensorList(weightScales1);
    at::TensorList weightScales2Ref = toTensorList(weightScales2);
    at::TensorList bias1Ref = toTensorList(bias1);
    at::TensorList bias2Ref = toTensorList(bias2);

    aclDataType weight1RefDtype = weight1Type.has_value() ? GetAclDataType(weight1Type.value()) :
                                                            ConvertToAclDataType(weight1Ref[0].scalar_type());
    aclDataType weightScales1Dtype;
    if (weight1RefDtype == aclDataType::ACL_FLOAT8_E5M2 || weight1RefDtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight1RefDtype == aclDataType::ACL_FLOAT4_E2M1) {
        weightScales1Dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weightScales1Dtype = aclDataType::ACL_UINT64;
    }

    aclDataType weight2RefDtype = weight2Type.has_value() ? GetAclDataType(weight2Type.value()) :
                                                            ConvertToAclDataType(weight2Ref[0].scalar_type());
    aclDataType weightScales2Dtype;
    if (weight2RefDtype == aclDataType::ACL_FLOAT8_E5M2 || weight2RefDtype == aclDataType::ACL_FLOAT8_E4M3FN ||
        weight2RefDtype == aclDataType::ACL_FLOAT4_E2M1) {
        weightScales2Dtype = aclDataType::ACL_FLOAT8_E8M0;
    } else {
        weightScales2Dtype = aclDataType::ACL_UINT64;
    }

    auto xSize = x.sizes();
    auto topkIdsSize = topkIds.sizes();
    int64_t bs = xSize[0];
    int64_t h = xSize[1];
    int64_t k = topkIdsSize[1];

    if ((dispatchQuantOutDtype.has_value()) &&
        (dispatchQuantOutDtype.value() == static_cast<int64_t>(DType::FLOAT4_E2M1))) {
        TORCH_CHECK(h % 2 == 0, "The last dim input shape must be divisible by 2 if "
                                "dispatch quant output type is torch_npu.float4_e2m1");
    }

    int64_t localMoeExpertNum = 1;
    localMoeExpertNum = moeExpertNum / epWorldSize;
    at::Tensor expertTokenNums;
    expertTokenNums = at::empty({localMoeExpertNum}, x.options().dtype(at::kInt));

    std::string commAlgStr = std::string(commAlg);
    char *commAlgPtr = const_cast<char *>(commAlg.c_str());

    std::string activationStr = std::string(activation);
    char *activationPtr = const_cast<char *>(activationStr.c_str());

    float activationClampValue = activationClamp.value_or(std::numeric_limits<float>::max());
    int64_t topoTypeValue = topoType.value_or(0);
    int64_t rankNumPerServerValue = rankNumPerServer.value_or(2);

    int64_t dispatchQuantResultType =
        dispatchQuantOutDtype.has_value() ? static_cast<int64_t>(GetAclDataType(dispatchQuantOutDtype.value())) : 28;

    at::Tensor y;
    y = at::empty({bs, h}, topkIds.options().dtype(x.scalar_type()));

    TensorListWrapper weight1Wrapper = {weight1Ref, weight1RefDtype};
    TensorListWrapper weight2Wrapper = {weight2Ref, weight2RefDtype};
    TensorListWrapper weightScales1Wrapper = {weightScales1Ref, weightScales1Dtype};
    TensorListWrapper weightScales2Wrapper = {weightScales2Ref, weightScales2Dtype};
    TensorListWrapper bias1Wrapper = {bias1Ref, aclDataType::ACL_FLOAT};
    TensorListWrapper bias2Wrapper = {bias2Ref, aclDataType::ACL_FLOAT};

    at::TensorList sharedWeight1Ref = toTensorList(sharedWeight1);
    at::TensorList sharedWeight2Ref = toTensorList(sharedWeight2);
    at::TensorList sharedWeightScales1Ref = toTensorList(sharedWeightScales1);
    at::TensorList sharedWeightScales2Ref = toTensorList(sharedWeightScales2);
    at::TensorList sharedBias1Ref = toTensorList(sharedBias1);
    at::TensorList sharedBias2Ref = toTensorList(sharedBias2);

    TensorListWrapper sharedWeight1Wrapper = {sharedWeight1Ref, weight1RefDtype};
    TensorListWrapper sharedWeight2Wrapper = {sharedWeight2Ref, weight2RefDtype};
    TensorListWrapper sharedWeightScales1Wrapper = {sharedWeightScales1Ref, weightScales1Dtype};
    TensorListWrapper sharedWeightScales2Wrapper = {sharedWeightScales2Ref, weightScales2Dtype};
    TensorListWrapper sharedBias1Wrapper = {sharedBias1Ref, aclDataType::ACL_FLOAT};
    TensorListWrapper sharedBias2Wrapper = {sharedBias2Ref, aclDataType::ACL_FLOAT};

    const char *profDump = getenv("MEGA_MOE_PROF_DUMP");
    const char *profCalib = getenv("MEGA_MOE_PROF_CALIB");
    bool profEnabled = (profDump != nullptr && std::strcmp(profDump, "1") == 0) ||
                       (profCalib != nullptr && std::strcmp(profCalib, "1") == 0);
    if (profEnabled) {
        int deviceId = 0;
        (void)aclrtGetDevice(&deviceId);
        LaunchMegaMoeWithProfDump(y, deviceId, context, x, topkIds, topkWeights, weight1Wrapper, weight2Wrapper,
                                  weightScales1Wrapper, weightScales2Wrapper, bias1Wrapper, bias2Wrapper,
                                  xActiveMask, sharedWeight1Wrapper, sharedWeight2Wrapper, sharedWeightScales1Wrapper,
                                  sharedWeightScales2Wrapper, sharedBias1Wrapper, sharedBias2Wrapper, moeExpertNum,
                                  epWorldSize, cclBufferSize, maxRecvTokenNum, dispatchQuantMode,
                                  dispatchQuantResultType, combineQuantMode, commAlgPtr, numMaxTokensPerRank,
                                  activationPtr, activationClampValue, topoTypeValue, rankNumPerServerValue,
                                  topkWeightsType, y, expertTokenNums);
    } else {
        ACLNN_CMD(aclnnMegaMoe, context, x, topkIds, topkWeights, weight1Wrapper, weight2Wrapper,
                  weightScales1Wrapper, weightScales2Wrapper, bias1Wrapper, bias2Wrapper, xActiveMask,
                  sharedWeight1Wrapper, sharedWeight2Wrapper, sharedWeightScales1Wrapper, sharedWeightScales2Wrapper,
                  sharedBias1Wrapper, sharedBias2Wrapper, moeExpertNum, epWorldSize, cclBufferSize, maxRecvTokenNum,
                  dispatchQuantMode, dispatchQuantResultType, combineQuantMode, commAlgPtr, numMaxTokensPerRank,
                  activationPtr, activationClampValue, topoTypeValue, rankNumPerServerValue, topkWeightsType, y,
                  expertTokenNums);
    }

    return std::tie(y, expertTokenNums);
}
namespace {
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_512 = 512LL;
constexpr int64_t MB_SIZE = 1024LL * 1024LL;
constexpr int64_t RESERVED_SPACE_SIZE = 10LL * 1024 * 1024;
constexpr int64_t MAX_EXPERTS_PER_RANK_A2A3 = 128LL;
constexpr int64_t SYNC_STATE_RESERVED_SIZE = 512LL * 1024;

int64_t CeilAlign(int64_t val, int64_t align)
{
    return (val + align - 1) / align * align;
}

// A2 minimum buffer size (MB).
// Matches tiling_arch22.cpp CalcLeastCclBufferSize with isA3=false.
int64_t CalcLeastCclBufferSizeA2(int64_t maxRecvTokenNum, int64_t h, int64_t epWorldSize, bool isQuantRouting,
                                 int64_t bs, int64_t topK)
{
    // Data block 1: TokenPerExpert
    // EP × CeilAlign(EP × MAX_EXPERTS_PER_RANK_A2A3 + 1, 128) × 4B
    int64_t offsetTokenPerExpert = epWorldSize * CeilAlign(epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 + 1, ALIGN_128) *
                                   static_cast<int64_t>(sizeof(int32_t));

    // Data block 2: tensors
    // ===== winIn =====
    int64_t offsetAAfterDispatch =
        maxRecvTokenNum * (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
    int64_t offsetD = bs * topK * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t winInTensorSize = offsetAAfterDispatch + offsetD;

    // ===== winOut =====
    int64_t offsetA = bs * topK * (!isQuantRouting ? h * static_cast<int64_t>(sizeof(int16_t)) : (h + ALIGN_512));
    int64_t offsetC = maxRecvTokenNum * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t winOutTensorSize = offsetA + offsetC;
    int64_t offsetTensor = std::max(winInTensorSize, winOutTensorSize);
    if (isQuantRouting) {
        offsetTensor += maxRecvTokenNum * static_cast<int64_t>(sizeof(float));
    }

    // Data block 3: sync flags
    int64_t offsetFlag = epWorldSize * ALIGN_512;                 // CrossRankSync
    offsetFlag += epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 * 64LL; // DispatchFlag
    offsetFlag += epWorldSize * 64LL;                             // AllGatherFlag

    return (offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE + MB_SIZE) / MB_SIZE;
}

// A3 minimum buffer size (MB).
// Matches tiling_arch22.cpp CalcLeastCclBufferSize with isA3=true.
int64_t CalcLeastCclBufferSizeA3(int64_t h, int64_t epWorldSize, bool isQuantRouting, int64_t bs, int64_t topK)
{
    // Data block 1: TokenPerExpert
    // EP × CeilAlign(EP × MAX_EXPERTS_PER_RANK_A2A3 + 1, 128) × 4B
    int64_t offsetTokenPerExpert = epWorldSize * CeilAlign(epWorldSize * MAX_EXPERTS_PER_RANK_A2A3 + 1, ALIGN_128) *
                                   static_cast<int64_t>(sizeof(int32_t));

    // Data block 2: tensors (winIn only, no winOut)
    int64_t offsetAAfterDispatch =
        bs * topK * (isQuantRouting ? (h + ALIGN_512) : h * static_cast<int64_t>(sizeof(int16_t)));
    int64_t offsetD = bs * topK * h * static_cast<int64_t>(sizeof(int16_t));
    int64_t offsetTensor = offsetAAfterDispatch + offsetD;
    if (isQuantRouting) {
        offsetTensor += bs * topK * static_cast<int64_t>(sizeof(float));
    }

    // Data block 3: sync flags
    int64_t offsetFlag = std::max(epWorldSize * ALIGN_512, SYNC_STATE_RESERVED_SIZE);

    return (offsetTokenPerExpert + offsetTensor + offsetFlag + RESERVED_SPACE_SIZE + MB_SIZE) / MB_SIZE;
}

// A5 half-buffer minimum size (MB). Ported 1:1 from the original Python implementation.
int64_t CalcHalfBufferSizeMBA5(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                               int64_t hidden, int64_t topkWeightsType)
{
    int64_t expertPerRank = moeExpertNum / epWorldSize;

    // 全卡软同步使用 60KB
    int64_t peermemDataOffset = 60LL * 1024LL;

    // mask_recv_size
    int64_t compareCount = CeilAlign(numMaxTokensPerRank * numTopk * 4, 256) / 4;
    int64_t maskAlignSize = CeilAlign(compareCount / 8, 32);
    int64_t maskSlotSize = maskAlignSize + 32;
    int64_t maskRecvSize = CeilAlign(expertPerRank * epWorldSize * maskSlotSize, 512);

    // quant_token_scale_size
    int64_t mxScaleNum = (hidden + 31) / 32;
    int64_t dataBytes = CeilAlign(hidden, 256);
    int64_t tokenBytes = CeilAlign(dataBytes + mxScaleNum, 32);
    if (topkWeightsType == 1) {
        int64_t weightBytes = CeilAlign(numTopk * static_cast<int64_t>(sizeof(float)), 32);
        tokenBytes = CeilAlign(tokenBytes + weightBytes, 32);
    }
    int64_t quantTokenScaleSize = CeilAlign(numMaxTokensPerRank * tokenBytes, 512);

    // combine_send_size
    int64_t combineOut = CeilAlign(numMaxTokensPerRank * hidden * numTopk * 2, 512);

    int64_t totalBytes = peermemDataOffset + maskRecvSize + quantTokenScaleSize + combineOut;
    totalBytes = totalBytes * 2;
    return totalBytes;
}
} // namespace

int64_t GetMegaMoeCclBufferSize(int64_t epWorldSize, int64_t moeExpertNum, int64_t numMaxTokensPerRank, int64_t numTopk,
                                int64_t hidden, int64_t maxRecvTokenNum, int64_t dispatchQuantMode,
                                c10::optional<int64_t> dispatchQuantOutDtype, int64_t combineQuantMode,
                                std::string commAlg, int64_t topkWeightsType)
{
    const char *socName = aclrtGetSocName();
    bool isA2 = (socName != nullptr && std::strstr(socName, "Ascend910B") != nullptr);
    bool isA3 = (socName != nullptr && std::strstr(socName, "Ascend910_93") != nullptr);
    if (isA2 || isA3) {
        TORCH_CHECK(epWorldSize == 2 || epWorldSize == 4 || epWorldSize == 8 || epWorldSize == 16 || epWorldSize == 32,
                    "ep_world_size only support {2, 4, 8, 16, 32} on A2/A3, but got ", epWorldSize);
        TORCH_CHECK(hidden >= 1024 && hidden <= 8192 && hidden % 512 == 0,
                    "hidden only support [1024, 8192] and hidden % 512 == 0 on A2/A3, but got ", hidden);
        TORCH_CHECK(numMaxTokensPerRank >= 1 && numMaxTokensPerRank <= 4096,
                    "num_max_tokens_per_rank only support [1, 4096] on A2/A3, but got ", numMaxTokensPerRank);
        TORCH_CHECK(moeExpertNum >= 1 && moeExpertNum <= 2048,
                    "moe_expert_num only support [1, 2048] on A2/A3, but got ", moeExpertNum);
        TORCH_CHECK(numTopk >= 1 && numTopk <= 16, "num_topk only support [1, 16] on A2/A3, but got ", numTopk);
        TORCH_CHECK(dispatchQuantMode == 0 || dispatchQuantMode == 2 || dispatchQuantMode == 4,
                    "dispatch_quant_mode only support {0, 2, 4} on A2/A3, but got ", dispatchQuantMode);

        bool isQuantRouting = (dispatchQuantMode == 4);
        // max_recv_token_num 为 0 时自动计算为 bs * epWorldSize * min(topK, expertPerRank)，
        if (maxRecvTokenNum == 0) {
            int64_t expertPerRank = moeExpertNum / epWorldSize;
            maxRecvTokenNum = numMaxTokensPerRank * epWorldSize * std::min(numTopk, expertPerRank);
        }
        if (isA3) {
            return CalcLeastCclBufferSizeA3(hidden, epWorldSize, isQuantRouting, numMaxTokensPerRank, numTopk);
        }
        return CalcLeastCclBufferSizeA2(maxRecvTokenNum, hidden, epWorldSize, isQuantRouting, numMaxTokensPerRank,
                                        numTopk);
    }

    // A5 / 950 — 校验与原 Python get_mega_moe_ccl_buffer_size 对齐
    TORCH_CHECK(epWorldSize >= 2 && epWorldSize <= 1024, "ep_world_size only support in [2, 1024], but got ",
                epWorldSize);
    TORCH_CHECK(hidden >= 1024 && hidden <= 8192, "hidden only support in [1024, 8192], but got ", hidden);
    TORCH_CHECK(numMaxTokensPerRank >= 1, "num_max_tokens_per_rank should be >= 1, but got ", numMaxTokensPerRank);
    TORCH_CHECK(moeExpertNum >= 1 && moeExpertNum <= 2048, "moe_expert_num only support in [1, 2048], but got ",
                moeExpertNum);
    TORCH_CHECK(numTopk >= 1 && numTopk <= 32, "num_topk only support in [1, 32], but got ", numTopk);

    return CalcHalfBufferSizeMBA5(epWorldSize, moeExpertNum, numMaxTokensPerRank, numTopk, hidden, topkWeightsType);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("npu_mega_moe", &NpuMegaMoe, "npu_mega_moe");
    m.def("get_mega_moe_ccl_buffer_size", &GetMegaMoeCclBufferSize, "get_mega_moe_ccl_buffer_size");
}

} // namespace op_api
