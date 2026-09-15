/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_vec_intf.h"
#include "kernel_cube_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "arch22/flash_mla_tiling_adapter.h"
#include <limits>
#include <type_traits>

// The original calculation classes receive their exact baseline wire layout.
// The new operator deliberately has no dependency on generated FIA Host tilings.
using FusedInferAttentionScoreTilingData = FlashMla::MlaNonQuantTiling;
using IncreFlashAttentionTilingDataMla = FlashMla::MlaLegacyTiling;
struct IncreFlashAttentionTilingDataV2;
struct alignas(8) FusedInferAttentionScoreEmptyTensorTilingData {
    uint64_t totalOutputSize;
    uint64_t singleCoreSize;
    uint64_t totalLseSize;
    uint64_t singleCoreLseSize;
    uint32_t usedCoreNum;
    uint32_t softmaxLseFlag;
    uint32_t headDim;
};

#include "adv_api/activation/softmax.h"
#include "adv_api/activation/softmaxflashv2.h"
#include "adv_api/math/log.h"
#include "adv_api/quantization/ascend_quant.h"
#include "adv_api/select/selectwithbytesmask.h"
#if ORIG_DTYPE_QUERY == DT_INT8
#include "../../incre_flash_attention/op_kernel/arch22/incre_flash_attention_preload_mla.h"
#endif
// IFA's public constants must be parsed before the empty kernel imports
// AttentionCommon; the nonquant kernel needs the empty kernel's FLOAT_INF.
#include "../../fused_infer_attention_score/op_kernel/arch22/fia_kernel_empty_tensor.h"

#if ORIG_DTYPE_QUERY != DT_INT8
// These constants came from the enclosing IFA entry in the original operator.
constexpr float FLOAT_ZERO = 0;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
#include "../../fused_infer_attention_score/op_kernel/arch22/fia_kernel_nonquant_mla.h"
#endif

namespace FlashMla {
// FIA_LAYOUT and LAYOUT use the same names, but are distinct original enums.
// BSND shares the original BSH template.
template <typename Layout>
__aicore__ constexpr Layout ToOriginalMlaLayout(MlaLayout layout)
{
    switch (layout) {
        case MlaLayout::BNSD:
            return Layout::BNSD;
        case MlaLayout::TND:
            return Layout::TND;
        case MlaLayout::NZ:
            return Layout::NZ;
        default:
            return Layout::BSH;
    }
}

struct MlaKernelInputs {
    __gm__ uint8_t *query;
    __gm__ uint8_t *key;
    __gm__ uint8_t *value;
    __gm__ uint8_t *queryRope;
    __gm__ uint8_t *keyRope;
    __gm__ uint8_t *attenMask;
    __gm__ uint8_t *seqQ;
    __gm__ uint8_t *seqKv;
    __gm__ uint8_t *blockTable;
    __gm__ uint8_t *dequantScaleQuery;
    __gm__ uint8_t *keyAntiquantScale;
    __gm__ uint8_t *valueAntiquantScale;
    __gm__ uint8_t *attentionOut;
    __gm__ uint8_t *softmaxLse;
};

#if ORIG_DTYPE_QUERY != DT_INT8
using MlaKernelTiling = MlaNonQuantTiling;

// Corresponds to INVOKE_FIA_OP_GENERAL_IMPL in the original FIA entry:
// select the original cube/vector/FD blocks, then call Init and Process.
template <typename T, bool PA, bool FD, AttentionCommon::FIA_LAYOUT Q, AttentionCommon::FIA_LAYOUT KV, bool TREE>
__aicore__ inline void RunMlaNonQuant(const MlaKernelInputs &in, const MlaNonQuantTiling &tiling,
                                      __gm__ uint8_t *userWorkspace, AscendC::TPipe &pipe)
{
    using Type = AttentionCommon::FIAType<T, T, T, T, PA, FD, Q, 0, false, KV, false, TREE>;
    using Cube = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockCubeNonQuantMla<Type>,
                                           FiaBlockCubeNonQuantMlaDummy<Type>>::type;
    using Vec = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecNonQuantMlaDummy<Type>,
                                          FiaBlockVecNonQuantMla<Type>>::type;
    using Decode = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecFlashDecodeDummy<Type>,
                                             FiaBlockVecFlashDecode<Type>>::type;
    FiaKernelNonQuantMla<Type, Cube, Vec, Decode> op;
    op.Init(in.query, in.key, in.value, nullptr, in.attenMask, in.seqQ, in.seqKv, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, in.blockTable, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, in.queryRope, in.keyRope, nullptr, nullptr, in.attentionOut, in.softmaxLse, userWorkspace,
            &tiling, nullptr, &pipe);
    op.Process();
}

#else
using MlaKernelTiling = MlaLegacyTiling;

template <typename Type>
class MlaLegacyKernel : public IncreFlashAttentionAttenPreloadMla<Type> {
public:
    __aicore__ inline void PrepareSeqBuffers(const MlaKernelInputs &in, const MlaLegacyTiling &tiling)
    {
        if constexpr (Type::isBalance) {
            // The baseline Init calls InitCalcParamsEach before InitActualSeqLen.
            // Balanced cores after core zero read the preceding batch there.
            this->tilingData = &tiling;
            this->InitActualSeqLen(in.seqQ, in.seqKv);
        }
    }
};

// Corresponds to INVOKE_IFA_NO_KFC_MLA_OP_IMPL in the original IFA entry.
// The only extra step is the BALANCE sequence prebinding described above.
template <bool FD, bool BALANCE, LAYOUT Q, bool TREE>
__aicore__ inline void RunMlaLegacy(const MlaKernelInputs &in, const MlaLegacyTiling &tiling,
                                    __gm__ uint8_t *userWorkspace, AscendC::TPipe &pipe)
{
    using Type = IFAType<int8_t, int8_t, bfloat16_t, bfloat16_t, true, FD, Q, 0, false, LAYOUT::NZ, AMLAMODE::NORMAL,
                         BALANCE, TREE>;
    static_assert(sizeof(typename IncreFlashAttentionAttenPreloadMla<Type>::MM1_OUT_T) == 4 &&
                      sizeof(typename IncreFlashAttentionAttenPreloadMla<Type>::KV_T) == 1 &&
                      sizeof(typename IncreFlashAttentionAttenPreloadMla<Type>::MM2_OUT_T) == 2 &&
                      sizeof(typename IncreFlashAttentionAttenPreloadMla<Type>::UPDATE_T) == 2,
                  "legacy workspace element widths must match FinalizeMlaTiling");
    MlaLegacyKernel<Type> op;
    op.PrepareSeqBuffers(in, tiling);
    op.Init(in.query, in.key, in.value, nullptr, in.attenMask, in.seqQ, in.seqKv, in.blockTable, nullptr, in.queryRope,
            in.keyRope, in.attentionOut, in.softmaxLse, userWorkspace, &tiling, nullptr, &pipe);
    op.InitQuant(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, in.keyAntiquantScale, nullptr,
                 in.valueAntiquantScale, nullptr, nullptr, in.dequantScaleQuery, userWorkspace);
    op.Process();
}

#endif

template <typename T>
__aicore__ inline void RunMlaEmpty(const MlaStaticLaunchInfo &launch, const MlaKernelInputs &in, AscendC::TPipe &pipe)
{
    // The original empty family partitions output by AIV, in elements.
    const uint64_t cores = launch.physicalBlockDim;
    FusedInferAttentionScoreEmptyTensorTilingData tiling{};
    tiling.totalOutputSize = launch.outputElements;
    tiling.singleCoreSize = launch.outputElements / cores + (launch.outputElements % cores != 0);
    tiling.totalLseSize = launch.lseElements;
    tiling.singleCoreLseSize = launch.lseElements / cores + (launch.lseElements % cores != 0);
    tiling.usedCoreNum = static_cast<uint32_t>(cores);
    tiling.softmaxLseFlag = launch.softmaxLseFlag;
    tiling.headDim = 512;
    FiaKernelEmptyTensor<T> op;
    op.Init(in.attentionOut, in.softmaxLse, &tiling, &pipe);
    op.Process();
}
} // namespace FlashMla

using namespace AscendC;
using namespace FlashMla;

extern "C" __global__ __aicore__ void flash_mla(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                                __gm__ uint8_t *attenMask, __gm__ uint8_t *seqQ, __gm__ uint8_t *seqKv,
                                                __gm__ uint8_t *blockTable, __gm__ uint8_t *keyAntiquantScale,
                                                __gm__ uint8_t *valueAntiquantScale, __gm__ uint8_t *queryRope,
                                                __gm__ uint8_t *keyRope, __gm__ uint8_t *dequantScaleQuery,
                                                __gm__ uint8_t *metadata, __gm__ uint8_t *attentionOut,
                                                __gm__ uint8_t *softmaxLse, __gm__ uint8_t *workspace,
                                                __gm__ uint8_t *staticTiling)
{
#include "arch22/flash_mla_kernel_registration.h"

    if (staticTiling == nullptr || metadata == nullptr ||
        (reinterpret_cast<uintptr_t>(metadata) & (kMlaMetadataAlignment - 1U)) != 0) {
        AscendC::Trap();
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(MlaStaticTiling, staticData, staticTiling);
    const auto &launch = staticData.launch;
    auto status = ValidateMlaLaunch(launch, GetBlockNum());
    // TILING_KEY_LIST emits CV1/CV2 entry symbols from one compilation of the
    // group's first key. Only the CV bit may differ from TILING_KEY_VAR.
    if ((launch.staticLaunchKey & ~UINT64_C(1)) != (TILING_KEY_VAR & ~UINT64_C(1))) {
        status = MlaStatus::INVALID_PATCH;
    }
    if (status != MlaStatus::SUCCESS) {
        AscendC::Trap();
        return;
    }
    if ((launch.outputElements != 0 && attentionOut == nullptr) ||
        (launch.softmaxLseFlag != 0 && launch.lseElements != 0 && softmaxLse == nullptr)) {
        AscendC::Trap();
        return;
    }
    MlaKernelInputs in{query,        key,       value,      queryRope,         keyRope,           attenMask,
                       seqQ,         seqKv,     blockTable, dequantScaleQuery, keyAntiquantScale, valueAntiquantScale,
                       attentionOut, softmaxLse};

    // The key's family field is zero only for static empty input. Keep this
    // compile-time exclusion: the empty entry runs on AIV without cube work.
#if ((TILING_KEY_VAR >> 16) & 15) == 0
    // Static input-empty dispatch requires no producer schedule.
    TPipe pipe;
    RunMlaEmpty<DTYPE_ATTENTION_OUT>(launch, in, pipe);
#else
    if ASCEND_IS_AIV {
        if (GetSubBlockNum() != launch.cvRatio) {
            AscendC::Trap();
            return;
        }
    }
    if (workspace == nullptr || query == nullptr || key == nullptr || value == nullptr || queryRope == nullptr ||
        keyRope == nullptr || attentionOut == nullptr || (launch.softmaxLseFlag != 0 && softmaxLse == nullptr) ||
        (launch.pageAttention != 0 && blockTable == nullptr)) {
        AscendC::Trap();
        return;
    }
    if (ORIG_DTYPE_QUERY == DT_INT8 &&
        (dequantScaleQuery == nullptr || keyAntiquantScale == nullptr || valueAntiquantScale == nullptr)) {
        AscendC::Trap();
        return;
    }
    // Bulk-copy before constructing TPipe, using the baseline tiling-copy ABI.
    GET_TILING_DATA_WITH_STRUCT(MlaMetadata, runtimeMetadata, metadata);
    MlaKernelTiling merged;
    MlaCopyBytes(&merged, staticData.payload, sizeof(merged));
    status = ApplyMlaValuePatch(runtimeMetadata, launch, &merged, sizeof(merged));
    uint64_t workspaceLayout = 0;
    if (status == MlaStatus::SUCCESS) {
        status = FinalizeMlaTiling(launch, runtimeMetadata.header, merged, GetBlockNum(), workspaceLayout);
    }
    if (status != MlaStatus::SUCCESS) {
        AscendC::Trap();
        return;
    }
    auto *userWorkspace = GetUserWorkspace(workspace);
    TPipe pipe;
#if ORIG_DTYPE_QUERY != DT_INT8
    constexpr auto queryLayout =
        ToOriginalMlaLayout<AttentionCommon::FIA_LAYOUT>(static_cast<MlaLayout>((TILING_KEY_VAR >> 8) & 15));
    constexpr auto kvLayout =
        ToOriginalMlaLayout<AttentionCommon::FIA_LAYOUT>(static_cast<MlaLayout>((TILING_KEY_VAR >> 4) & 15));
    constexpr bool pageAttention = ((TILING_KEY_VAR >> 3) & 1) != 0;
    constexpr bool treeMask = ((TILING_KEY_VAR >> 2) & 1) != 0;
    // FD depends on this invocation's sequence values. Both original template
    // instances are compiled into this static entry.
    if ((runtimeMetadata.header.dynamicFlags & kMlaFlagFd) != 0) {
        RunMlaNonQuant<DTYPE_QUERY, pageAttention, true, queryLayout, kvLayout, treeMask>(in, merged, userWorkspace,
                                                                                          pipe);
    } else {
        RunMlaNonQuant<DTYPE_QUERY, pageAttention, false, queryLayout, kvLayout, treeMask>(in, merged, userWorkspace,
                                                                                           pipe);
    }
#else
    constexpr auto queryLayout = ToOriginalMlaLayout<LAYOUT>(static_cast<MlaLayout>((TILING_KEY_VAR >> 8) & 15));
    constexpr bool treeMask = ((TILING_KEY_VAR >> 2) & 1) != 0;
    const bool flashDecode = (runtimeMetadata.header.dynamicFlags & kMlaFlagFd) != 0;
    if ((runtimeMetadata.header.dynamicFlags & kMlaFlagBalance) != 0) {
        if (flashDecode) {
            RunMlaLegacy<true, true, queryLayout, treeMask>(in, merged, userWorkspace, pipe);
        } else {
            RunMlaLegacy<false, true, queryLayout, treeMask>(in, merged, userWorkspace, pipe);
        }
    } else if constexpr (queryLayout != LAYOUT::TND) {
        if (flashDecode) {
            RunMlaLegacy<true, false, queryLayout, treeMask>(in, merged, userWorkspace, pipe);
        } else {
            RunMlaLegacy<false, false, queryLayout, treeMask>(in, merged, userWorkspace, pipe);
        }
    }
#endif
#endif
}
