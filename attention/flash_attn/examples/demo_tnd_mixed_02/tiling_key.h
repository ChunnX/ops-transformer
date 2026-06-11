#ifndef TILING_KEY_H_
#define TILING_KEY_H_

#include "flash_attn_def.h"
#include "split_policy.h"

// Replicates GET_TPL_TILING_KEY macro from AscendC SDK
// Based on ASCENDC_TPL_ARGS_DECL in op_kernel/arch35/flash_attn_template_tiling_key.h
// Bit packing:
//   bits [7:0]   InOutLayoutType (8-bit)
//   bits [15:8]  KvLayoutType    (8-bit)
//   bit  [16]    HasAttenMask    (1-bit)
//   bit  [17]    Config          (1-bit)
inline uint64_t ComputeTilingKey(uint64_t inputLayout, uint64_t kvLayoutType, bool hasAttenMask, uint64_t config)
{
    return inputLayout | (kvLayoutType << 8) | (static_cast<uint64_t>(hasAttenMask) << 16) | (config << 17);
}

// Extracted from: op_host/arch35/flash_attn_tiling.cpp (UpdateTilingKeyLayout)
inline void UpdateTilingKeyLayout(const CaseParams &params, FaTilingKeyInfo &keyInfo)
{
    if (params.outLayout == FaLayout::BSND && params.qLayout == FaLayout::BNSD) {
        keyInfo.inputLayout = InOutLayoutType_BNSD_BSND;
    } else if (params.outLayout == FaLayout::BNSD) {
        keyInfo.inputLayout = InOutLayoutType_BNSD;
    } else if (params.outLayout == FaLayout::TND) {
        keyInfo.inputLayout = InOutLayoutType_TND;
    } else {
        keyInfo.inputLayout = InOutLayoutType_BSND;
    }
}

// Extracted from: op_host/arch35/flash_attn_tiling.cpp (UpdateTilingKeyConfig)
inline void UpdateTilingKeyConfig(uint32_t sOuterFactor, FaTilingKeyInfo &keyInfo)
{
    keyInfo.config = (sOuterFactor == SOUTER_32) ? 1 : 0;
}

// Extracted from: op_host/arch35/flash_attn_tiling.cpp (UpdateTilingKeyKvLayout)
inline void UpdateTilingKeyKvLayout(const CaseParams &params, FaTilingKeyInfo &keyInfo)
{
    if (params.kvLayout == FaLayout::PA_BBND) {
        keyInfo.kvLayoutType = KvLayoutType_PA_BBH;
    } else if (params.kvLayout == FaLayout::PA_BNBD) {
        keyInfo.kvLayoutType = KvLayoutType_PA_BNBD;
    } else if (params.kvLayout == FaLayout::PA_Nz) {
        keyInfo.kvLayoutType = KvLayoutType_PA_NZ;
    }
}

// Extracted from: op_host/arch35/flash_attn_tiling.cpp (GenTilingKey + SplitPolicy)
// Full tiling key generation flow for a given case
inline uint64_t GenTilingKey(const CaseParams &params, FaTilingKeyInfo &keyInfo, uint32_t &sOuterOut,
                             uint32_t &sInnerOut)
{
    // Step 1: SplitPolicy — determine sOuter/sInner
    int64_t winLeft = params.winLeft;
    int64_t winRight = params.winRight;
    if (params.maskMode == static_cast<int64_t>(MaskMode::NO_MASK) ||
        params.maskMode == static_cast<int64_t>(MaskMode::CAUSAL)) {
        winLeft = MASK_MODE_INT_MAX;
        winRight = MASK_MODE_INT_MAX;
    }

    uint32_t qLayoutVal = static_cast<uint32_t>(params.qLayout);
    AdjustSinnerAndSouter(static_cast<uint32_t>(params.vHeadDim), static_cast<uint32_t>(params.s1Size), params.s2Size,
                          static_cast<int32_t>(params.maskMode), winLeft, winRight, qLayoutVal, sOuterOut, sInnerOut);

    // Step 2: UpdateTilingKeyInfo
    UpdateTilingKeyLayout(params, keyInfo);
    UpdateTilingKeyConfig(sOuterOut, keyInfo);
    UpdateTilingKeyKvLayout(params, keyInfo);
    keyInfo.hasAttenMask = (params.maskMode == 3) ? 1 : 0;

    // Step 3: Compute final tiling key
    return ComputeTilingKey(keyInfo.inputLayout, keyInfo.kvLayoutType, keyInfo.hasAttenMask, keyInfo.config);
}

#endif // TILING_KEY_H_
