/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Local copy of the FaVectorApi ReduceFinalRes_VF family for
 * sparse_attention_score_prefill Phase2 combine.
 *
 * Rationale: this operator's Phase1 does not divide accumOut by the softmax
 * sum, so Phase2 must perform an extra per-row normalization by the original
 * Phase1 rowSum before the cross-split scale-and-reduce. Instead of a separate
 * UB-level AscendC::Divs pre-pass, the per-row normalize is a Mul here: CopyLseIn
 * pre-inverts the compact per-row rowSum in place (AscendC::Reciprocal on
 * lseSumTmpBuf), so this kernel Mul-multiplies vregAccumOut by the broadcast
 * 1/rowSum (equivalent to the old Div, cheaper) before the existing
 * Mul(…, vregLse, …) / Add.
 *
 * This header is a near-verbatim copy of the ReduceFinalRes_* family in
 * attention/common/op_kernel/arch35/vf/vf_flash_decode.h, with:
 *   - an extra rowSumUb / rowSumLocal parameter (per-row original rowSum,
 *     pre-inverted to 1/rowSum by CopyLseIn);
 *   - one MicroAPI::Mul (was Div) per z-block, multiplying vregAccumOut by the
 *     broadcast 1/rowSum before the existing Mul(…, vregLse, …) / Add.
 *
 * Include ordering: must be included AFTER
 *   ../../../../../common/op_kernel/arch35/vf/vf_flash_decode.h
 * which provides FLT_ZERO / FLT_MAX_NEW and the MicroAPI types.
 */
#ifndef VF_FLASH_DECODE_MSA_H
#define VF_FLASH_DECODE_MSA_H

namespace FaVectorApiPrefill {

// 处理循环splitKVIndex=0的场景，vregDst需要置0
template <typename T>
__simd_vf__ void ReduceFinalRes_0_VF(__ubuf__ T * dstUb, __ubuf__ T * lseUb, __ubuf__ T * accumOutUb,
                                      __ubuf__ T * rowSumUb, uint16_t k, uint16_t z,
                                      uint32_t dealNum1Reg, uint32_t repStride, const uint16_t floatRepSize,
                                      const uint16_t dLoops, uint32_t dealRowCount, uint32_t splitKVIndex)
{
    MicroAPI::RegTensor<T> vregDst;
    MicroAPI::RegTensor<T> vregLse;
    MicroAPI::RegTensor<T> vregAccumOut;
    MicroAPI::RegTensor<T> vregRowSum;
    uint32_t n = dealNum1Reg;
    MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(n);

    for (k = 0; k < static_cast<uint16_t>(dealRowCount); k++) {  // repeat g

        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BLK>(vregLse,
                                                            (__ubuf__ float*&)lseUb + splitKVIndex * dealRowCount * 8 + k * 8);
        // rowSum is per-row (constant across the D-dim z-blocks): broadcast-load once per row.
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vregRowSum, (__ubuf__ float*&)rowSumUb + k);
        for (z = 0; z < dLoops; z++) {
            // splitKVIndex=0的场景，vregDst不需要load，直接置0
            MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::ZEROING, float>(vregDst, FLT_ZERO, pregTailN);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(
                vregAccumOut, (__ubuf__ float*&)accumOutUb + k * repStride * 8 + z * floatRepSize);
            // Embedded regbase Mul: normalize Phase1 accumOut by its original per-row rowSum.
            // rowSum was pre-inverted to 1/rowSum by CopyLseIn's Reciprocal on the compact
            // lseSumTmpBuf, so this is a Mul (cheaper than Div). Broadcast-loaded via
            // DIST_BRC_B32 (one reciprocal per row).
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregAccumOut, vregAccumOut, vregRowSum, pregTailN);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregAccumOut, vregLse, vregAccumOut, pregTailN);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregDst, vregDst, vregAccumOut, pregTailN);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ float*&)dstUb + k * repStride * 8 + z * floatRepSize, vregDst, pregTailN);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceFinalRes_0(LocalTensor<T>& dstLocal, LocalTensor<T>& lseLocal,
                                           LocalTensor<T>& accumOutLocal, LocalTensor<T>& rowSumLocal,
                                           uint32_t dealRowCount,
                                           uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    __ubuf__ T * dstUb = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ T * lseUb = (__ubuf__ T *)lseLocal.GetPhyAddr();
    __ubuf__ T * accumOutUb = (__ubuf__ T *)accumOutLocal.GetPhyAddr();
    __ubuf__ T * rowSumUb = (__ubuf__ T *)rowSumLocal.GetPhyAddr();
    uint16_t k = 0;
    uint16_t z = 0;
    uint32_t dealNum1Reg = 256 / sizeof(float);
    uint32_t repStride = headDimAlignFp32 / 8;
    const uint16_t floatRepSize = 64;
    const uint16_t dLoops = headDimAlignFp32 / floatRepSize;

    ReduceFinalRes_0_VF<T>(dstUb, lseUb, accumOutUb, rowSumUb, k, z, dealNum1Reg, repStride, floatRepSize,
                        dLoops, dealRowCount, splitKVIndex);
}

// 处理循环splitKVIndex>0的场景，reg_dst需要先从dstUb中load之前的结果，再进行add
template <typename T>
__simd_vf__ void ReduceFinalRes_Rest_VF(__ubuf__ T * dstUb, __ubuf__ T * lseUb, __ubuf__ T * accumOutUb,
                                        __ubuf__ T * rowSumUb, uint16_t k, uint16_t z,
                                        uint32_t dealNum1Reg, uint32_t repStride, const uint16_t floatRepSize, const uint16_t dLoops,
                                        uint32_t dealRowCount, uint32_t splitKVIndex)
{
    MicroAPI::RegTensor<T> vregDst;
    MicroAPI::RegTensor<T> vregLse;
    MicroAPI::RegTensor<T> vregAccumOut;
    MicroAPI::RegTensor<T> vregRowSum;
    uint32_t n = dealNum1Reg;
    MicroAPI::MaskReg pregTailN = MicroAPI::UpdateMask<T>(n);
    uint32_t stride = (0x1 << 16) | 0x8;

    for (k = 0; k < static_cast<uint16_t>(dealRowCount); k++) {  // repeat g
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BLK>(vregLse,
                                                            (__ubuf__ float*&)lseUb + splitKVIndex * dealRowCount * 8 + k * 8);
        // rowSum is per-row (constant across the D-dim z-blocks): broadcast-load once per row.
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(vregRowSum, (__ubuf__ float*&)rowSumUb + k);
        for (z = 0; z < dLoops; z++) {
            // splitKVIndex>0的场景，reg_dst需要先从dstUb中load之前的结果，再进行add
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(
                vregDst, (__ubuf__ float*&)dstUb + k * repStride * 8 + z * floatRepSize);
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(
                vregAccumOut, (__ubuf__ float*&)accumOutUb + k * repStride * 8 + z * floatRepSize);
            // Embedded regbase Mul: normalize Phase1 accumOut by its original per-row rowSum.
            // rowSum was pre-inverted to 1/rowSum by CopyLseIn's Reciprocal on the compact
            // lseSumTmpBuf, so this is a Mul (cheaper than Div). Broadcast-loaded via
            // DIST_BRC_B32 (one reciprocal per row).
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregAccumOut, vregAccumOut, vregRowSum, pregTailN);
            MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING>(vregAccumOut, vregLse, vregAccumOut, pregTailN);
            MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(vregDst, vregDst, vregAccumOut, pregTailN);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(
                (__ubuf__ float*&)dstUb + k * repStride * 8 + z * floatRepSize, vregDst, pregTailN);
        }
    }
}

template <typename T>
__aicore__ inline void ReduceFinalRes_Rest(LocalTensor<T>& dstLocal, LocalTensor<T>& lseLocal,
                                                 LocalTensor<T>& accumOutLocal, LocalTensor<T>& rowSumLocal,
                                                 uint32_t dealRowCount,
                                                 uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    __ubuf__ T * dstUb = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ T * lseUb = (__ubuf__ T *)lseLocal.GetPhyAddr();
    __ubuf__ T * accumOutUb = (__ubuf__ T *)accumOutLocal.GetPhyAddr();
    __ubuf__ T * rowSumUb = (__ubuf__ T *)rowSumLocal.GetPhyAddr();
    uint16_t k = 0;
    uint16_t z = 0;
    uint32_t dealNum1Reg = 256 / sizeof(float);
    uint32_t repStride = headDimAlignFp32 / 8;
    const uint16_t floatRepSize = 64;
    const uint16_t dLoops = headDimAlignFp32 / floatRepSize;

    ReduceFinalRes_Rest_VF<T>(dstUb, lseUb, accumOutUb, rowSumUb, k, z, dealNum1Reg, repStride, floatRepSize, dLoops, dealRowCount, splitKVIndex);
}

template <typename T>
__aicore__ inline void ReduceFinalRes_VF(LocalTensor<T>& dstLocal, LocalTensor<T>& lseLocal, LocalTensor<T>& accumOutLocal,
    LocalTensor<T>& rowSumLocal,
    uint32_t dealRowCount, uint64_t headDimAlignFp32, uint32_t splitKVIndex)
{
    if (splitKVIndex == 0) {
        ReduceFinalRes_0(dstLocal, lseLocal, accumOutLocal, rowSumLocal, dealRowCount, headDimAlignFp32, splitKVIndex);
    } else {
        ReduceFinalRes_Rest(dstLocal, lseLocal, accumOutLocal, rowSumLocal, dealRowCount, headDimAlignFp32, splitKVIndex);
    }
}

} // namespace FaVectorApiPrefill

#endif // VF_FLASH_DECODE_MSA_H
