/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef SAGE_ATTENTION2_SOFTMAX_H_
#define SAGE_ATTENTION2_SOFTMAX_H_

namespace SageAttention2Kernel {

// SageAttention2 already owns the INT32 QK accumulator and both quantization scales. Combining score
// dequantization with the row-max pass removes one full UB read/write pass and releases the QK slot before exp.
template <bool PER_THREAD>
class FusedDequantSoftmax {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementP = float8_e4m3_t;

    __aicore__ inline FusedDequantSoftmax(Catlass::Arch::Resource<ArchTag> &resource, uint32_t &ubOffset)
    {
        expSumUb_ = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += HALF_ROW_BYTES;
        nowMaxUb_ = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += HALF_ROW_BYTES;
        for (uint32_t i = 0U; i < 2U; ++i) {
            pUb_[i] = resource.ubBuf.template GetBufferByByte<ElementP>(ubOffset);
            ubOffset += HALF_P_BYTES;
        }
        for (uint32_t i = 0U; i < 3U; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i);
        }
    }

    __aicore__ inline ~FusedDequantSoftmax()
    {
        for (uint32_t i = 0U; i < 3U; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i);
        }
    }

    template <bool APPLY_CAUSAL, class TensorDst>
    __aicore__ inline void operator()(TensorDst &pL1, AscendC::LocalTensor<float> &sum,
        AscendC::LocalTensor<float> &lastMax, AscendC::LocalTensor<float> &expMax,
        const AscendC::LocalTensor<half> &accum, const AscendC::LocalTensor<half> &score,
        const AscendC::LocalTensor<float> &kScale, const AscendC::LocalTensor<float> &qScale,
        uint32_t rows, uint32_t rowOffset, uint32_t cols, float smScale, int32_t validBase,
        bool update, uint32_t slot2, uint32_t slot3, uint64_t freeFlag, uint64_t readyFlag)
    {
        auto accumAddr = reinterpret_cast<__ubuf__ half *>(accum.GetPhyAddr());
        auto scoreAddr = reinterpret_cast<__ubuf__ half *>(score.GetPhyAddr());
        auto kScaleAddr = reinterpret_cast<__ubuf__ float *>(kScale.GetPhyAddr());
        auto qScaleAddr = reinterpret_cast<__ubuf__ float *>(qScale.GetPhyAddr());
        auto lastMaxAddr = reinterpret_cast<__ubuf__ float *>(lastMax.GetPhyAddr());
        auto nowMaxAddr = update ? reinterpret_cast<__ubuf__ float *>(nowMaxUb_.GetPhyAddr()) : lastMaxAddr;
        if (cols > 128U) {
            auto highMaxAddr = reinterpret_cast<__ubuf__ float *>(expSumUb_.GetPhyAddr());
            if (rows == Q_TILE / 2U) {
                FusedDequantMax<APPLY_CAUSAL, true>(accumAddr, scoreAddr, kScaleAddr, qScaleAddr, nowMaxAddr,
                    Q_TILE / 2U, static_cast<uint16_t>(rowOffset), 128U, smScale, validBase);
            } else {
                FusedDequantMax<APPLY_CAUSAL, false>(accumAddr, scoreAddr, kScaleAddr, qScaleAddr, nowMaxAddr,
                    static_cast<uint16_t>(rows), static_cast<uint16_t>(rowOffset), 128U, smScale, validBase);
            }
            if (rows == Q_TILE / 2U && cols == KV_TILE) {
                constexpr uint32_t highKScaleOffset = PER_THREAD ? 8U : 2U;
                FusedDequantMax<APPLY_CAUSAL, true>(accumAddr + 128U, scoreAddr + 128U,
                    kScaleAddr + highKScaleOffset,
                    qScaleAddr, highMaxAddr, Q_TILE / 2U, static_cast<uint16_t>(rowOffset), 128U,
                    smScale, validBase - 128);
            } else {
                constexpr uint32_t highKScaleOffset = PER_THREAD ? 8U : 2U;
                FusedDequantMax<APPLY_CAUSAL, false>(accumAddr + 128U, scoreAddr + 128U,
                    kScaleAddr + highKScaleOffset,
                    qScaleAddr, highMaxAddr, static_cast<uint16_t>(rows), static_cast<uint16_t>(rowOffset),
                    static_cast<uint16_t>(cols - 128U), smScale, validBase - 128);
            }
            if (update) {
                MergeTileMaxUpdate(nowMaxAddr, highMaxAddr, lastMaxAddr, rows);
            } else {
                MergeTileMax(nowMaxAddr, highMaxAddr, rows);
            }
        } else if (rows == Q_TILE / 2U && cols == 128U) {
            FusedDequantMax<APPLY_CAUSAL, true>(accumAddr, scoreAddr, kScaleAddr, qScaleAddr, nowMaxAddr,
                Q_TILE / 2U, static_cast<uint16_t>(rowOffset), 128U, smScale, validBase);
        } else {
            FusedDequantMax<APPLY_CAUSAL, false>(accumAddr, scoreAddr, kScaleAddr, qScaleAddr, nowMaxAddr,
                static_cast<uint16_t>(rows), static_cast<uint16_t>(rowOffset), static_cast<uint16_t>(cols),
                smScale, validBase);
        }
        // The scaled FP16 score has independent storage, so release the QK slot before max update and exp.
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(freeFlag);
        if (update && cols <= 128U) {
            UpdateMax(nowMaxAddr, lastMaxAddr, rows);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot3);

        const uint32_t blockStride = AscendC::CeilDivision(rows, 32U) * 32U;
        auto pAddr = reinterpret_cast<__ubuf__ ElementP *>(pUb_[slot2].GetPhyAddr());
        auto expSumAddr = update ? reinterpret_cast<__ubuf__ float *>(expSumUb_.GetPhyAddr()) :
            reinterpret_cast<__ubuf__ float *>(sum.GetPhyAddr());
        if (cols > 128U) {
            ComputeExpPackSum<4U>(pAddr, scoreAddr, nowMaxAddr, expSumAddr,
                static_cast<uint16_t>(rows), blockStride);
        } else if (cols > 64U) {
            ComputeExpPackSum<2U>(pAddr, scoreAddr, nowMaxAddr, expSumAddr,
                static_cast<uint16_t>(rows), blockStride);
        } else {
            ComputeExpPackSum<1U>(pAddr, scoreAddr, nowMaxAddr, expSumAddr,
                static_cast<uint16_t>(rows), blockStride);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot2);
        auto pUbLayout = tla::MakeLayout(
            tla::MakeShape(tla::MakeShape(rows, tla::Int<1>{}),
                tla::MakeShape(tla::Int<32>{}, tla::Int<4>{})),
            tla::MakeStride(tla::MakeStride(tla::Int<32>{}, tla::Int<32>{} * rows),
                tla::MakeStride(tla::Int<1>{}, blockStride * 32U)));
        auto pUbTensor = tla::MakeTensor(pUb_[slot2], pUbLayout, Catlass::Arch::PositionUB{});
        if (rows != 0U) {
            const uint32_t zero = 0U;
            const uint32_t lowCols = cols < 128U ? cols : 128U;
            auto pLowDst = GetTile(pL1, tla::MakeCoord(zero, zero), tla::MakeShape(rows, lowCols));
            using CopyPLow = Catlass::Epilogue::Tile::CopyUb2L1Tla<ArchTag, decltype(pUbTensor), decltype(pLowDst)>;
            CopyPLow copyPLow;
            copyPLow(pLowDst, pUbTensor);
            if (cols > 128U) {
                auto pHighUb = pUb_[slot2][blockStride * 128U];
                auto pHighUbTensor = tla::MakeTensor(pHighUb, pUbLayout, Catlass::Arch::PositionUB{});
                auto pHighDst = GetTile(
                    pL1, tla::MakeCoord(zero, 128U), tla::MakeShape(rows, cols - 128U));
                using CopyPHigh = Catlass::Epilogue::Tile::CopyUb2L1Tla<
                    ArchTag, decltype(pHighUbTensor), decltype(pHighDst)>;
                CopyPHigh copyPHigh;
                copyPHigh(pHighDst, pHighUbTensor);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot3);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(readyFlag);

        if (update) {
            UpdateSum(reinterpret_cast<__ubuf__ float *>(sum.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(expMax.GetPhyAddr()), lastMaxAddr,
                reinterpret_cast<__ubuf__ float *>(expSumUb_.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float *>(nowMaxUb_.GetPhyAddr()), rows);
        }
    }

private:
    template <bool APPLY_CAUSAL, bool FULL_TILE>
    __simd_vf__ static inline void FusedDequantMax(__ubuf__ half *src, __ubuf__ half *dst,
        __ubuf__ float *kScale, __ubuf__ float *qScale, __ubuf__ float *rowMax, uint16_t rows,
        uint16_t rowOffset, uint16_t cols, float smScale, int32_t validBase)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait H16_F32 = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait F32_H16 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg full16 = CreateMask<half, MaskPattern::ALL>();
        RegTensor<float> ks0;
        RegTensor<float> ks1;
        RegTensor<int32_t> lanes;
        Arange(lanes, 0);
        if constexpr (PER_THREAD) {
            RegTensor<int32_t> scaleIndex;
            RegTensor<int32_t> scaleMask;
            ShiftRights(scaleIndex, lanes, static_cast<int16_t>(1), full);
            Duplicate(scaleMask, 3, full);
            And(scaleIndex, scaleIndex, scaleMask, full);
            Gather(ks0, kScale, reinterpret_cast<RegTensor<uint32_t> &>(scaleIndex), full);
            if (cols > 64U) {
                Gather(ks1, kScale + 4U, reinterpret_cast<RegTensor<uint32_t> &>(scaleIndex), full);
            }
        } else {
            LoadAlign<float, LoadDist::DIST_BRC_B32>(ks0, kScale);
            if (cols > 64U) {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ks1, kScale + 1U);
            }
        }
        RegTensor<float> minReg;
        if constexpr (APPLY_CAUSAL || !FULL_TILE) {
            Duplicate(minReg, MIN_SCORE, full);
        }
        UnalignReg rowMaxStore;
        if constexpr (FULL_TILE) {
            static_assert((Q_TILE / 2U) % WARP_ROWS == 0U,
                "a vector sub-block must contain complete quantization warps");
            constexpr uint16_t FULL_WARPS = (Q_TILE / 2U) / WARP_ROWS;
            for (uint16_t warp = 0U; warp < FULL_WARPS; ++warp) {
              const uint16_t baseGroup = PER_THREAD ?
                  (rowOffset / WARP_ROWS + warp) * THREAD_ROWS : rowOffset / WARP_ROWS + warp;
              for (uint16_t group = 0U; group < THREAD_ROWS; ++group) {
                RegTensor<float> qs;
                RegTensor<float> scale0;
                RegTensor<float> scale1;
                const uint16_t qGroup = PER_THREAD ? baseGroup + group : baseGroup;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(qs, qScale + qGroup);
                if constexpr (!PER_THREAD) {
                    Muls(qs, qs, smScale, full);
                }
                Mul(scale0, ks0, qs, full);
                Mul(scale1, ks1, qs, full);
                uint16_t r = warp * WARP_ROWS + group;
#define SAGE_FUSED_DEQUANT_MAX_ROW() \
                { \
                    RegTensor<half> a0; \
                    RegTensor<half> a1; \
                    RegTensor<float> v0; \
                    RegTensor<float> v1; \
                    RegTensor<half> h0; \
                    RegTensor<half> h1; \
                    RegTensor<float> maxInput; \
                    RegTensor<float> reduced; \
                    LoadAlign<half, LoadDist::DIST_UNPACK_B16>( \
                        a0, src + static_cast<uint32_t>(r) * KV_TILE); \
                    LoadAlign<half, LoadDist::DIST_UNPACK_B16>( \
                        a1, src + static_cast<uint32_t>(r) * KV_TILE + 64U); \
                    Cast<float, half, H16_F32>(v0, a0, full16); \
                    Cast<float, half, H16_F32>(v1, a1, full16); \
                    Mul(v0, v0, scale0, full); \
                    Mul(v1, v1, scale1, full); \
                    if constexpr (APPLY_CAUSAL) { \
                        int32_t valid = validBase + r; \
                        int32_t valid0 = valid < 64 ? valid : 64; \
                        int32_t valid1 = valid > 64 ? valid - 64 : 0; \
                        valid1 = valid1 < 64 ? valid1 : 64; \
                        MaskReg mask0; \
                        MaskReg mask1; \
                        CompareScalar<int32_t, AscendC::CMPMODE::LT>(mask0, lanes, valid0, full); \
                        CompareScalar<int32_t, AscendC::CMPMODE::LT>(mask1, lanes, valid1, full); \
                        Select(v0, v0, minReg, mask0); \
                        Select(v1, v1, minReg, mask1); \
                    } \
                    Cast<half, float, F32_H16>(h0, v0, full); \
                    Cast<half, float, F32_H16>(h1, v1, full); \
                    StoreAlign<half, StoreDist::DIST_PACK_B32>( \
                        dst + static_cast<uint32_t>(r) * KV_TILE, h0, full); \
                    StoreAlign<half, StoreDist::DIST_PACK_B32>( \
                        dst + static_cast<uint32_t>(r) * KV_TILE + 64U, h1, full); \
                    Max(maxInput, v0, v1, full); \
                    ReduceMax(reduced, maxInput, full); \
                    StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(rowMax, reduced, rowMaxStore, 1U); \
                }
                SAGE_FUSED_DEQUANT_MAX_ROW();
                r += THREAD_ROWS;
                SAGE_FUSED_DEQUANT_MAX_ROW();
                r += THREAD_ROWS;
                SAGE_FUSED_DEQUANT_MAX_ROW();
                r += THREAD_ROWS;
                SAGE_FUSED_DEQUANT_MAX_ROW();
#undef SAGE_FUSED_DEQUANT_MAX_ROW
              }
            }
        } else {
          for (uint16_t r = 0U; r < rows; ++r) {
            const uint16_t qr = rowOffset + r;
            const uint16_t qg = PER_THREAD ?
                (qr / WARP_ROWS) * THREAD_ROWS + qr % THREAD_ROWS : qr / WARP_ROWS;
            RegTensor<float> qs;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(qs, qScale + qg);
            if constexpr (!PER_THREAD) {
                Muls(qs, qs, smScale, full);
            }
            RegTensor<float> scale0;
            RegTensor<float> scale1;
            Mul(scale0, ks0, qs, full);
            if (cols > 64U) {
                Mul(scale1, ks1, qs, full);
            }
            RegTensor<half> a0;
            RegTensor<half> h0;
            RegTensor<float> v0;
            LoadAlign<half, LoadDist::DIST_UNPACK_B16>(a0, src + static_cast<uint32_t>(r) * KV_TILE);
            Cast<float, half, H16_F32>(v0, a0, full16);
            Mul(v0, v0, scale0, full);
            uint32_t n0 = cols < 64U ? cols : 64U;
            MaskReg nMask0 = UpdateMask<float>(n0);
            MaskReg mask0 = nMask0;
            if constexpr (APPLY_CAUSAL) {
                int32_t valid = validBase + r;
                int32_t valid0 = valid < 64 ? valid : 64;
                CompareScalar<int32_t, AscendC::CMPMODE::LT>(mask0, lanes, valid0, full);
            }
            if (n0 < 64U || APPLY_CAUSAL) {
                Select(v0, v0, minReg, mask0);
            }
            Cast<half, float, F32_H16>(h0, v0, full);
            StoreAlign<half, StoreDist::DIST_PACK_B32>(
                dst + static_cast<uint32_t>(r) * KV_TILE, h0, nMask0);
            RegTensor<float> maxInput;
            maxInput = v0;
            if (cols > 64U) {
                RegTensor<half> a1;
                RegTensor<half> h1;
                RegTensor<float> v1;
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(
                    a1, src + static_cast<uint32_t>(r) * KV_TILE + 64U);
                Cast<float, half, H16_F32>(v1, a1, full16);
                Mul(v1, v1, scale1, full);
                uint32_t n1 = cols - 64U < 64U ? cols - 64U : 64U;
                MaskReg nMask1 = UpdateMask<float>(n1);
                MaskReg mask1 = nMask1;
                if constexpr (APPLY_CAUSAL) {
                    int32_t valid = validBase + r - 64;
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(mask1, lanes, valid, full);
                }
                if (n1 < 64U || APPLY_CAUSAL) {
                    Select(v1, v1, minReg, mask1);
                }
                Cast<half, float, F32_H16>(h1, v1, full);
                StoreAlign<half, StoreDist::DIST_PACK_B32>(
                    dst + static_cast<uint32_t>(r) * KV_TILE + 64U, h1, nMask1);
                Max(maxInput, v0, v1, full);
            }
            RegTensor<float> reduced;
            ReduceMax(reduced, maxInput, full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(rowMax, reduced, rowMaxStore, 1U);
          }
        }
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(rowMax, rowMaxStore, 0U);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __simd_vf__ static inline void UpdateMax(__ubuf__ float *nowMax, __ubuf__ float *lastMax, uint32_t rows)
    {
        using namespace AscendC::MicroAPI;
        MaskReg mask = UpdateMask<float>(rows);
        RegTensor<float> now;
        RegTensor<float> last;
        RegTensor<float> out;
        LoadAlign(now, nowMax);
        LoadAlign(last, lastMax);
        Max(out, now, last, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(nowMax, out, mask);
    }

    __simd_vf__ static inline void MergeTileMax(__ubuf__ float *lowMax, __ubuf__ float *highMax, uint32_t rows)
    {
        using namespace AscendC::MicroAPI;
        MaskReg mask = UpdateMask<float>(rows);
        RegTensor<float> low;
        RegTensor<float> high;
        RegTensor<float> out;
        LoadAlign(low, lowMax);
        LoadAlign(high, highMax);
        Max(out, low, high, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lowMax, out, mask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __simd_vf__ static inline void MergeTileMaxUpdate(
        __ubuf__ float *lowMax, __ubuf__ float *highMax, __ubuf__ float *lastMax, uint32_t rows)
    {
        using namespace AscendC::MicroAPI;
        MaskReg mask = UpdateMask<float>(rows);
        RegTensor<float> low;
        RegTensor<float> high;
        RegTensor<float> last;
        RegTensor<float> out;
        LoadAlign(low, lowMax);
        LoadAlign(high, highMax);
        LoadAlign(last, lastMax);
        Max(out, low, high, mask);
        Max(out, out, last, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lowMax, out, mask);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    template <uint32_t CHUNKS>
    __simd_vf__ static inline void ComputeExpPackSum(__ubuf__ ElementP *out, __ubuf__ half *score,
        __ubuf__ float *rowMax, __ubuf__ float *rowSum, uint16_t rows, uint32_t blockStride)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait H16_F32_ZERO = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait CAST_ZERO = {
            RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait CAST_TWO = {
            RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg full16 = CreateMask<half, MaskPattern::ALL>();
        MaskReg full8 = CreateMask<uint8_t, MaskPattern::ALL>();
        MaskReg half8 = CreateMask<uint8_t, MaskPattern::H>();
        MaskReg quarter8 = CreateMask<uint8_t, MaskPattern::Q>();
        __ubuf__ ElementP *outHigh = out + blockStride * 128U;
        UnalignReg sumStore;
        for (uint16_t r = 0U; r < rows; ++r) {
            RegTensor<float> maxReg;
            RegTensor<float> e0;
            RegTensor<float> sumReg;
            LoadAlign<float, LoadDist::DIST_BRC_B32>(maxReg, rowMax + r);
            if constexpr (CHUNKS >= 2U) {
                RegTensor<half> h0;
                RegTensor<half> h1;
                RegTensor<float> low;
                RegTensor<float> high;
                RegTensor<float> s0;
                RegTensor<float> s1;
                RegTensor<float> e1;
                RegTensor<float> e;
                RegTensor<ElementP> p0;
                RegTensor<ElementP> p1;
                RegTensor<ElementP> packed;
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(
                    h0, score + static_cast<uint32_t>(r) * KV_TILE);
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(
                    h1, score + static_cast<uint32_t>(r) * KV_TILE + 64U);
                Cast<float, half, H16_F32_ZERO>(low, h0, full16);
                Cast<float, half, H16_F32_ZERO>(high, h1, full16);
                DeInterleave(s0, s1, low, high);
                ExpSub(e0, s0, maxReg, full);
                ExpSub(e1, s1, maxReg, full);
                Add(e, e0, e1, full);
                Cast<ElementP, float, CAST_ZERO>(p0, e0, full);
                Cast<ElementP, float, CAST_TWO>(p1, e1, full);
                Or(reinterpret_cast<RegTensor<uint8_t> &>(packed), reinterpret_cast<RegTensor<uint8_t> &>(p0),
                    reinterpret_cast<RegTensor<uint8_t> &>(p1), full8);
                DeInterleave(p0, p1, packed, packed);
                StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    out, p0, blockStride, 1U, half8);
                if constexpr (CHUNKS == 4U) {
                    RegTensor<float> s2;
                    RegTensor<float> s3;
                    RegTensor<float> e2;
                    RegTensor<float> e3;
                    RegTensor<float> eHigh;
                    RegTensor<float> eAll;
                    RegTensor<ElementP> p2;
                    RegTensor<ElementP> p3;
                    RegTensor<ElementP> packedHigh;
                    RegTensor<half> h2;
                    RegTensor<half> h3;
                    RegTensor<float> lowHigh;
                    RegTensor<float> highHigh;
                    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(
                        h2, score + static_cast<uint32_t>(r) * KV_TILE + 128U);
                    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(
                        h3, score + static_cast<uint32_t>(r) * KV_TILE + 192U);
                    Cast<float, half, H16_F32_ZERO>(lowHigh, h2, full16);
                    Cast<float, half, H16_F32_ZERO>(highHigh, h3, full16);
                    DeInterleave(s2, s3, lowHigh, highHigh);
                    ExpSub(e2, s2, maxReg, full);
                    ExpSub(e3, s3, maxReg, full);
                    Add(eHigh, e2, e3, full);
                    Add(eAll, e, eHigh, full);
                    Cast<ElementP, float, CAST_ZERO>(p2, e2, full);
                    Cast<ElementP, float, CAST_TWO>(p3, e3, full);
                    Or(reinterpret_cast<RegTensor<uint8_t> &>(packedHigh),
                        reinterpret_cast<RegTensor<uint8_t> &>(p2),
                        reinterpret_cast<RegTensor<uint8_t> &>(p3), full8);
                    DeInterleave(p2, p3, packedHigh, packedHigh);
                    StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                        outHigh, p2, blockStride, 1U, half8);
                    ReduceSum(sumReg, eAll, full);
                } else {
                    ReduceSum(sumReg, e, full);
                }
            } else {
                RegTensor<half> h01;
                RegTensor<float> s0;
                RegTensor<ElementP> p;
                RegTensor<ElementP> p0;
                RegTensor<ElementP> p1;
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(
                    h01, score + static_cast<uint32_t>(r) * KV_TILE);
                Cast<float, half, H16_F32_ZERO>(s0, h01, full16);
                ExpSub(e0, s0, maxReg, full);
                Cast<ElementP, float, CAST_ZERO>(p, e0, full);
                DeInterleave(p0, p1, p, p);
                DeInterleave(p, p1, p0, p0);
                StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    out, p, blockStride, 1U, quarter8);
                ReduceSum(sumReg, e0, full);
            }
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(rowSum, sumReg, sumStore, 1U);
        }
        StoreUnAlignPost<float, PostLiteral::POST_MODE_UPDATE>(rowSum, sumStore, 0U);
    }

    __simd_vf__ static inline void UpdateSum(__ubuf__ float *sumAddr, __ubuf__ float *expAddr,
        __ubuf__ float *lastAddr, __ubuf__ float *curAddr, __ubuf__ float *nowAddr, uint32_t rows)
    {
        using namespace AscendC::MicroAPI;
        MaskReg mask = UpdateMask<float>(rows);
        RegTensor<float> last;
        RegTensor<float> now;
        RegTensor<float> factor;
        RegTensor<float> runningSum;
        RegTensor<float> curSum;
        LoadAlign(last, lastAddr);
        LoadAlign(now, nowAddr);
        ExpSub(factor, last, now, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expAddr, factor, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lastAddr, now, mask);
        LoadAlign(runningSum, sumAddr);
        LoadAlign(curSum, curAddr);
        MulDstAdd(factor, runningSum, curSum, mask);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(sumAddr, factor, mask);
    }

    static constexpr uint32_t HALF_ROW_BYTES = (Q_TILE / 2U) * sizeof(float);
    static constexpr uint32_t HALF_P_BYTES = (Q_TILE / 2U) * KV_TILE * sizeof(ElementP);
    static constexpr uint64_t SYNC_MODE = 4U;
    AscendC::LocalTensor<float> expSumUb_;
    AscendC::LocalTensor<float> nowMaxUb_;
    AscendC::LocalTensor<ElementP> pUb_[2];
};

// Product fullquant keeps Q rows in vector lanes and walks the K axis. This Dn specialization preserves
// Sage's per-thread Q/K scales while removing all horizontal row reductions and the intermediate score tile.
template <bool PER_THREAD, bool IS_CAUSAL>
class FusedDnSoftmax {
public:
    using ArchTag = Catlass::Arch::Ascend950;
    using ElementP = float8_e4m3_t;

    __aicore__ inline FusedDnSoftmax(Catlass::Arch::Resource<ArchTag> &resource, uint32_t &ubOffset)
    {
        expSumUb_ = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += HALF_ROW_BYTES;
        nowMaxUb_ = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += HALF_ROW_BYTES;
        for (uint32_t i = 0U; i < 2U; ++i) {
            pUb_[i] = resource.ubBuf.template GetBufferByByte<ElementP>(ubOffset);
            ubOffset += DN_PACK_BYTES;
        }
        dnIndexUb_ = resource.ubBuf.template GetBufferByByte<uint8_t>(ubOffset);
        ubOffset += DN_INDEX_BYTES;
        qIndexUb_ = resource.ubBuf.template GetBufferByByte<uint32_t>(ubOffset);
        ubOffset += Q_INDEX_BYTES;
        qScaleExpandedUb_ = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
        ubOffset += Q_SCALE_EXPANDED_BYTES;
        if ASCEND_IS_AIV {
            for (uint32_t i = 0U; i < 4U; ++i) {
                for (uint32_t j = 0U; j < 64U; ++j) {
                    dnIndexUb_.SetValue(i * 64U + j, static_cast<uint8_t>(i + (j << 2U)));
                }
            }
            for (uint32_t q = 0U; q < Q_TILE; ++q) {
                if constexpr (PER_THREAD) {
                    qIndexUb_.SetValue(q, (q / WARP_ROWS) * THREAD_ROWS + q % THREAD_ROWS);
                } else {
                    qIndexUb_.SetValue(q, q / WARP_ROWS);
                }
            }
            // SetValue is scalar-pipeline work. Make the lookup tables visible before the first VF load;
            // a vector-only barrier can otherwise reuse stale UB contents after a different kernel shape.
            AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID0);
        }
        for (uint32_t i = 0U; i < 3U; ++i) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i);
        }
    }

    __aicore__ inline void PrepareQScale(const AscendC::LocalTensor<float> &qScale, uint32_t rowOffset)
    {
        auto qScaleAddr = reinterpret_cast<__ubuf__ float *>(qScale.GetPhyAddr());
        auto qIndexAddr = reinterpret_cast<__ubuf__ uint32_t *>(qIndexUb_.GetPhyAddr()) + rowOffset;
        auto expandedAddr = reinterpret_cast<__ubuf__ float *>(qScaleExpandedUb_.GetPhyAddr());
        ExpandQScaleVf(qScaleAddr, qIndexAddr, expandedAddr);
    }

    __aicore__ inline ~FusedDnSoftmax()
    {
        for (uint32_t i = 0U; i < 3U; ++i) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i);
        }
    }

    __aicore__ inline void ProfileMaxOnly(AscendC::LocalTensor<float> &lastMax,
        AscendC::LocalTensor<float> &expMax, const AscendC::LocalTensor<half> &accum,
        const AscendC::LocalTensor<half> &score, const AscendC::LocalTensor<float> &kScale,
        uint32_t cols, int32_t validBase, bool update, uint64_t freeFlag)
    {
        auto accumAddr = reinterpret_cast<__ubuf__ half *>(accum.GetPhyAddr());
        auto scoreAddr = reinterpret_cast<__ubuf__ half *>(score.GetPhyAddr());
        auto maxAddr = reinterpret_cast<__ubuf__ float *>(lastMax.GetPhyAddr());
        auto expAddr = reinterpret_cast<__ubuf__ float *>(expMax.GetPhyAddr());
        auto tmpMaxAddr = reinterpret_cast<__ubuf__ float *>(nowMaxUb_.GetPhyAddr());
        auto kScaleAddr = reinterpret_cast<__ubuf__ float *>(kScale.GetPhyAddr());
        auto qScaleExpandedAddr = reinterpret_cast<__ubuf__ float *>(qScaleExpandedUb_.GetPhyAddr());

        // Probe-only cumulative cut: execute the exact dequant/cache/max prefix used by the full Dn softmax,
        // then close the QK producer-consumer token without publishing a P token.
        DnMaxCacheVf(accumAddr, scoreAddr, maxAddr, expAddr, tmpMaxAddr,
            kScaleAddr, qScaleExpandedAddr, cols, validBase, update);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(freeFlag);
    }

    template <class TensorDst>
    __aicore__ inline void operator()(TensorDst &pL1,
        AscendC::LocalTensor<float> &sum, AscendC::LocalTensor<float> &lastMax,
        AscendC::LocalTensor<float> &expMax, const AscendC::LocalTensor<half> &accum,
        const AscendC::LocalTensor<half> &score,
        const AscendC::LocalTensor<float> &kScale, uint32_t cols, int32_t validBase, bool update,
        uint32_t slot2, uint32_t slot3, uint32_t halfRows, uint32_t rowOffset,
        uint64_t freeFlag, uint64_t readyFlag)
    {
        auto accumAddr = reinterpret_cast<__ubuf__ half *>(accum.GetPhyAddr());
        auto scoreAddr = reinterpret_cast<__ubuf__ half *>(score.GetPhyAddr());
        auto pAddr = reinterpret_cast<__ubuf__ ElementP *>(pUb_[slot2].GetPhyAddr());
        auto sumAddr = reinterpret_cast<__ubuf__ float *>(sum.GetPhyAddr());
        auto maxAddr = reinterpret_cast<__ubuf__ float *>(lastMax.GetPhyAddr());
        auto expAddr = reinterpret_cast<__ubuf__ float *>(expMax.GetPhyAddr());
        auto tmpSumAddr = reinterpret_cast<__ubuf__ float *>(expSumUb_.GetPhyAddr());
        auto tmpMaxAddr = reinterpret_cast<__ubuf__ float *>(nowMaxUb_.GetPhyAddr());
        auto kScaleAddr = reinterpret_cast<__ubuf__ float *>(kScale.GetPhyAddr());
        auto dnIndexAddr = reinterpret_cast<__ubuf__ uint8_t *>(dnIndexUb_.GetPhyAddr());
        auto qScaleExpandedAddr = reinterpret_cast<__ubuf__ float *>(qScaleExpandedUb_.GetPhyAddr());

        DnMaxCacheVf(accumAddr, scoreAddr, maxAddr, expAddr, tmpMaxAddr,
            kScaleAddr, qScaleExpandedAddr, cols, validBase, update);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_V>(freeFlag);
        if (cols < KV_TILE) {
            const uint32_t tailElements = (KV_TILE - cols) * HALF_ROWS;
            AscendC::Duplicate(score[static_cast<uint64_t>(cols) * HALF_ROWS], static_cast<half>(MIN_SCORE),
                tailElements);
            AscendC::PipeBarrier<PIPE_V>();
        }
        DnExpPackVf(scoreAddr, pAddr, sumAddr, expAddr, tmpSumAddr, tmpMaxAddr,
            dnIndexAddr, KV_TILE, update);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot3);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(slot2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(slot2);
        if constexpr (SAGE_ATTENTION2_SOFTMAX_PROBE == 0U) {
            constexpr uint32_t alignedCols = KV_TILE;
            const uint32_t blocksPerHalf = alignedCols / 64U;
            // The 64-row Q tail is split into 32 valid rows on each AIV. Compact those two halves into
            // consecutive P rows so MM2 sees a dense qReal x KV_TILE matrix. For a full Q128 tile this is
            // identical to the previous 0/64-row placement.
            const uint32_t dstBase = rowOffset * alignedCols;
            AscendC::DataCopy(pL1.data()[dstBase], pUb_[slot2],
                {static_cast<uint16_t>(blocksPerHalf), 64U, 66U, 0U});
            if (halfRows > 32U) {
                AscendC::DataCopy(pL1.data()[dstBase + alignedCols * 32U], pUb_[slot2][65U << 5U],
                    {static_cast<uint16_t>(blocksPerHalf), 64U, 66U, 0U});
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot3);
        AscendC::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(readyFlag);
    }

private:
    __simd_vf__ static inline void ExpandQScaleVf(
        __ubuf__ float *qScale, __ubuf__ uint32_t *qIndex, __ubuf__ float *expanded)
    {
        using namespace AscendC::MicroAPI;
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        RegTensor<uint32_t> qIndexes;
        RegTensor<float> qScaleReg;
        LoadAlign(qIndexes, qIndex);
        Gather(qScaleReg, qScale, qIndexes, full);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(expanded, qScaleReg, full);
    }

    __simd_vf__ static inline void DnMaxCacheVf(__ubuf__ half *src, __ubuf__ half *score,
        __ubuf__ float *runningMaxAddr, __ubuf__ float *expMaxAddr, __ubuf__ float *tmpMaxAddr,
        __ubuf__ float *kScale, __ubuf__ float *qScaleExpanded, uint32_t cols, int32_t validBase, bool update)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait H16_F32 = {
            RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
        constexpr static CastTrait F32_H16 = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait F32_H16_ONE = {
            RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg full16 = CreateMask<half, MaskPattern::ALL>();
        RegTensor<float> qScaleReg;
        LoadAlign(qScaleReg, qScaleExpanded);
        RegTensor<int32_t> lanes;
        RegTensor<float> minReg;
        if constexpr (IS_CAUSAL) {
            Arange(lanes, 0);
            Duplicate(minReg, MIN_SCORE, full);
        }

        RegTensor<float> max0;
        RegTensor<float> max1;
        RegTensor<float> max2;
        RegTensor<float> max3;
        Duplicate(max0, MIN_SCORE, full);
        Duplicate(max1, MIN_SCORE, full);
        Duplicate(max2, MIN_SCORE, full);
        Duplicate(max3, MIN_SCORE, full);
        constexpr uint16_t GROUPS_PER_SCALE_BLOCK = K_QUANT_BLOCK / THREAD_ROWS;
        for (uint16_t scaleBlock = 0U; scaleBlock < cols / K_QUANT_BLOCK; ++scaleBlock) {
            RegTensor<float> ks0;
            RegTensor<float> ks1;
            RegTensor<float> ks2;
            RegTensor<float> ks3;
            if constexpr (PER_THREAD) {
                const uint32_t scaleBase = static_cast<uint32_t>(scaleBlock) * 4U;
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ks0, kScale + scaleBase);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ks1, kScale + scaleBase + 1U);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ks2, kScale + scaleBase + 2U);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ks3, kScale + scaleBase + 3U);
            } else {
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ks0, kScale + scaleBlock);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ks1, kScale + scaleBlock);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ks2, kScale + scaleBlock);
                LoadAlign<float, LoadDist::DIST_BRC_B32>(ks3, kScale + scaleBlock);
            }
            // qScale is invariant for the whole softmax tile. Fold it into each K-group scale once,
            // instead of multiplying every score by qScale and kScale separately in both repeats.
            Mul(ks0, ks0, qScaleReg, full);
            Mul(ks1, ks1, qScaleReg, full);
            Mul(ks2, ks2, qScaleReg, full);
            Mul(ks3, ks3, qScaleReg, full);
            for (uint16_t groupInBlock = 0U; groupInBlock < GROUPS_PER_SCALE_BLOCK; ++groupInBlock) {
                const uint32_t group = static_cast<uint32_t>(scaleBlock) * GROUPS_PER_SCALE_BLOCK + groupInBlock;
                const uint32_t k = group * THREAD_ROWS;
                RegTensor<half> h0;
                RegTensor<half> h1;
                RegTensor<half> h2;
                RegTensor<half> h3;
                RegTensor<float> v0;
                RegTensor<float> v1;
                RegTensor<float> v2;
                RegTensor<float> v3;
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(h0, src + (k + 0U) * HALF_ROWS);
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(h1, src + (k + 2U) * HALF_ROWS);
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(h2, src + (k + 4U) * HALF_ROWS);
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(h3, src + (k + 6U) * HALF_ROWS);
                Cast<float, half, H16_F32>(v0, h0, full16);
                Cast<float, half, H16_F32>(v1, h1, full16);
                Cast<float, half, H16_F32>(v2, h2, full16);
                Cast<float, half, H16_F32>(v3, h3, full16);
                Mul(v0, v0, ks0, full);
                Mul(v1, v1, ks1, full);
                Mul(v2, v2, ks2, full);
                Mul(v3, v3, ks3, full);
                if constexpr (IS_CAUSAL) {
                    MaskReg invalid0;
                    MaskReg invalid1;
                    MaskReg invalid2;
                    MaskReg invalid3;
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        invalid0, lanes, static_cast<int32_t>(k + 0U) - validBase + 1, full);
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        invalid1, lanes, static_cast<int32_t>(k + 2U) - validBase + 1, full);
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        invalid2, lanes, static_cast<int32_t>(k + 4U) - validBase + 1, full);
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        invalid3, lanes, static_cast<int32_t>(k + 6U) - validBase + 1, full);
                    Select(v0, minReg, v0, invalid0);
                    Select(v1, minReg, v1, invalid1);
                    Select(v2, minReg, v2, invalid2);
                    Select(v3, minReg, v3, invalid3);
                }
                Max(max0, max0, v0, full);
                Max(max1, max1, v1, full);
                Max(max2, max2, v2, full);
                Max(max3, max3, v3, full);
                RegTensor<half> score0;
                RegTensor<half> score1;
                RegTensor<half> score2;
                RegTensor<half> score3;
                Cast<half, float, F32_H16>(score0, v0, full16);
                Cast<half, float, F32_H16>(score1, v1, full16);
                Cast<half, float, F32_H16>(score2, v2, full16);
                Cast<half, float, F32_H16>(score3, v3, full16);

                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(h0, src + (k + 1U) * HALF_ROWS);
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(h1, src + (k + 3U) * HALF_ROWS);
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(h2, src + (k + 5U) * HALF_ROWS);
                LoadAlign<half, LoadDist::DIST_UNPACK_B16>(h3, src + (k + 7U) * HALF_ROWS);
                Cast<float, half, H16_F32>(v0, h0, full16);
                Cast<float, half, H16_F32>(v1, h1, full16);
                Cast<float, half, H16_F32>(v2, h2, full16);
                Cast<float, half, H16_F32>(v3, h3, full16);
                Mul(v0, v0, ks0, full);
                Mul(v1, v1, ks1, full);
                Mul(v2, v2, ks2, full);
                Mul(v3, v3, ks3, full);
                if constexpr (IS_CAUSAL) {
                    MaskReg invalid0;
                    MaskReg invalid1;
                    MaskReg invalid2;
                    MaskReg invalid3;
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        invalid0, lanes, static_cast<int32_t>(k + 1U) - validBase + 1, full);
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        invalid1, lanes, static_cast<int32_t>(k + 3U) - validBase + 1, full);
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        invalid2, lanes, static_cast<int32_t>(k + 5U) - validBase + 1, full);
                    CompareScalar<int32_t, AscendC::CMPMODE::LT>(
                        invalid3, lanes, static_cast<int32_t>(k + 7U) - validBase + 1, full);
                    Select(v0, minReg, v0, invalid0);
                    Select(v1, minReg, v1, invalid1);
                    Select(v2, minReg, v2, invalid2);
                    Select(v3, minReg, v3, invalid3);
                }
                Max(max0, max0, v0, full);
                Max(max1, max1, v1, full);
                Max(max2, max2, v2, full);
                Max(max3, max3, v3, full);
                RegTensor<half> scoreOdd0;
                RegTensor<half> scoreOdd1;
                RegTensor<half> scoreOdd2;
                RegTensor<half> scoreOdd3;
                // Preserve adjacent K rows in the ZERO/ONE halves consumed by mixed-type ExpSub. This
                // matches the product fullquant score contract and halves score-cache stores and loads.
                Cast<half, float, F32_H16_ONE>(scoreOdd0, v0, full16);
                Cast<half, float, F32_H16_ONE>(scoreOdd1, v1, full16);
                Cast<half, float, F32_H16_ONE>(scoreOdd2, v2, full16);
                Cast<half, float, F32_H16_ONE>(scoreOdd3, v3, full16);
                Or(reinterpret_cast<RegTensor<uint16_t> &>(score0),
                    reinterpret_cast<RegTensor<uint16_t> &>(score0),
                    reinterpret_cast<RegTensor<uint16_t> &>(scoreOdd0), full16);
                Or(reinterpret_cast<RegTensor<uint16_t> &>(score1),
                    reinterpret_cast<RegTensor<uint16_t> &>(score1),
                    reinterpret_cast<RegTensor<uint16_t> &>(scoreOdd1), full16);
                Or(reinterpret_cast<RegTensor<uint16_t> &>(score2),
                    reinterpret_cast<RegTensor<uint16_t> &>(score2),
                    reinterpret_cast<RegTensor<uint16_t> &>(scoreOdd2), full16);
                Or(reinterpret_cast<RegTensor<uint16_t> &>(score3),
                    reinterpret_cast<RegTensor<uint16_t> &>(score3),
                    reinterpret_cast<RegTensor<uint16_t> &>(scoreOdd3), full16);
                StoreAlign<half, StoreDist::DIST_NORM_B16>(score + (k + 0U) * HALF_ROWS, score0, full16);
                StoreAlign<half, StoreDist::DIST_NORM_B16>(score + (k + 2U) * HALF_ROWS, score1, full16);
                StoreAlign<half, StoreDist::DIST_NORM_B16>(score + (k + 4U) * HALF_ROWS, score2, full16);
                StoreAlign<half, StoreDist::DIST_NORM_B16>(score + (k + 6U) * HALF_ROWS, score3, full16);
            }
        }
        Max(max0, max0, max2, full);
        Max(max1, max1, max3, full);
        Max(max0, max0, max1, full);

        RegTensor<float> oldMax;
        RegTensor<float> expFactor;
        if (update) {
            LoadAlign(oldMax, runningMaxAddr);
            Max(max0, max0, oldMax, full);
            ExpSub(expFactor, oldMax, max0, full);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(expMaxAddr, expFactor, full);
        }
        StoreAlign<float, StoreDist::DIST_NORM_B32>(tmpMaxAddr, max0, full);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(runningMaxAddr, max0, full);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }

    __simd_vf__ static inline void DnExpPackVf(__ubuf__ half *score, __ubuf__ ElementP *out,
        __ubuf__ float *runningSumAddr, __ubuf__ float *expMaxAddr, __ubuf__ float *tmpSumAddr,
        __ubuf__ float *tmpMaxAddr, __ubuf__ uint8_t *dnIndex, uint32_t cols, bool update)
    {
        using namespace AscendC::MicroAPI;
        constexpr static CastTrait F32_F8_ZERO = {
            RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait F32_F8_ONE = {
            RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait F32_F8_TWO = {
            RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait F32_F8_THREE = {
            RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait F32_H16_ZERO = {
            RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        constexpr static CastTrait F32_H16_ONE = {
            RegLayout::ONE, SatMode::NO_SAT, MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        MaskReg full = CreateMask<float, MaskPattern::ALL>();
        MaskReg full16 = CreateMask<half, MaskPattern::ALL>();
        MaskReg full8 = CreateMask<uint8_t, MaskPattern::ALL>();
        RegTensor<uint8_t> packIndexes;
        RegTensor<float> max0;
        RegTensor<half> maxHalf0;
        RegTensor<half> maxHalf1;
        RegTensor<half> maxHalf;
        if constexpr (SAGE_ATTENTION2_SOFTMAX_PROBE < 2U) {
            LoadAlign(packIndexes, dnIndex);
        }
        LoadAlign(max0, tmpMaxAddr);
        Cast<half, float, F32_H16_ZERO>(maxHalf0, max0, full);
        Cast<half, float, F32_H16_ONE>(maxHalf1, max0, full);
        Or(reinterpret_cast<RegTensor<uint16_t> &>(maxHalf),
            reinterpret_cast<RegTensor<uint16_t> &>(maxHalf0),
            reinterpret_cast<RegTensor<uint16_t> &>(maxHalf1), full16);

        RegTensor<float> sum0;
        RegTensor<float> sum1;
        RegTensor<float> sum2;
        RegTensor<float> sum3;
        RegTensor<float> sum4;
        RegTensor<float> sum5;
        RegTensor<float> sum6;
        RegTensor<float> sum7;
        Duplicate(sum0, 0.0F, full);
        Duplicate(sum1, 0.0F, full);
        Duplicate(sum2, 0.0F, full);
        Duplicate(sum3, 0.0F, full);
        Duplicate(sum4, 0.0F, full);
        Duplicate(sum5, 0.0F, full);
        Duplicate(sum6, 0.0F, full);
        Duplicate(sum7, 0.0F, full);
        __ubuf__ ElementP *out1 = out + 32U;
        const uint32_t quarterCols = cols / 4U;
        for (uint16_t i = 0U; i < cols / 8U; ++i) {
            const uint32_t k0 = static_cast<uint32_t>(i) * 2U;
            const uint32_t k1 = quarterCols + k0;
            const uint32_t k2 = 2U * quarterCols + k0;
            const uint32_t k3 = 3U * quarterCols + k0;
            RegTensor<half> half04;
            RegTensor<half> half15;
            RegTensor<half> half26;
            RegTensor<half> half37;
            RegTensor<float> value0;
            RegTensor<float> value1;
            RegTensor<float> value2;
            RegTensor<float> value3;
            RegTensor<float> value4;
            RegTensor<float> value5;
            RegTensor<float> value6;
            RegTensor<float> value7;
            LoadAlign<half, LoadDist::DIST_NORM>(half04, score + k0 * HALF_ROWS);
            LoadAlign<half, LoadDist::DIST_NORM>(half15, score + k1 * HALF_ROWS);
            LoadAlign<half, LoadDist::DIST_NORM>(half26, score + k2 * HALF_ROWS);
            LoadAlign<half, LoadDist::DIST_NORM>(half37, score + k3 * HALF_ROWS);
            ExpSub<float, half, RegLayout::ZERO>(value0, half04, maxHalf, full16);
            ExpSub<float, half, RegLayout::ZERO>(value1, half15, maxHalf, full16);
            ExpSub<float, half, RegLayout::ZERO>(value2, half26, maxHalf, full16);
            ExpSub<float, half, RegLayout::ZERO>(value3, half37, maxHalf, full16);
            ExpSub<float, half, RegLayout::ONE>(value4, half04, maxHalf, full16);
            ExpSub<float, half, RegLayout::ONE>(value5, half15, maxHalf, full16);
            ExpSub<float, half, RegLayout::ONE>(value6, half26, maxHalf, full16);
            ExpSub<float, half, RegLayout::ONE>(value7, half37, maxHalf, full16);
            Add(sum0, sum0, value0, full);
            Add(sum1, sum1, value1, full);
            Add(sum2, sum2, value2, full);
            Add(sum3, sum3, value3, full);
            Add(sum4, sum4, value4, full);
            Add(sum5, sum5, value5, full);
            Add(sum6, sum6, value6, full);
            Add(sum7, sum7, value7, full);

            if constexpr (SAGE_ATTENTION2_SOFTMAX_PROBE < 2U) {
                RegTensor<ElementP> packed0;
                RegTensor<ElementP> packed1;
                RegTensor<ElementP> cast0;
                RegTensor<ElementP> cast1;
                Cast<ElementP, float, F32_F8_ZERO>(packed0, value0, full);
                Cast<ElementP, float, F32_F8_ONE>(cast0, value1, full);
                Cast<ElementP, float, F32_F8_TWO>(cast1, value2, full);
                Cast<ElementP, float, F32_F8_THREE>(packed1, value3, full);
                Or(reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(cast0), full8);
                Or(reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(cast1), full8);
                Or(reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(packed1), full8);
                Gather(packed1, packed0, packIndexes);
                StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    out, packed1, 65U, 2U, full8);

                Cast<ElementP, float, F32_F8_ZERO>(packed0, value4, full);
                Cast<ElementP, float, F32_F8_ONE>(cast0, value5, full);
                Cast<ElementP, float, F32_F8_TWO>(cast1, value6, full);
                Cast<ElementP, float, F32_F8_THREE>(packed1, value7, full);
                Or(reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(cast0), full8);
                Or(reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(cast1), full8);
                Or(reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(packed0),
                    reinterpret_cast<RegTensor<uint8_t> &>(packed1), full8);
                Gather(packed1, packed0, packIndexes);
                StoreAlign<ElementP, DataCopyMode::DATA_BLOCK_COPY, PostLiteral::POST_MODE_UPDATE>(
                    out1, packed1, 65U, 2U, full8);
            }
        }
        Add(sum0, sum0, sum1, full);
        Add(sum2, sum2, sum3, full);
        Add(sum4, sum4, sum5, full);
        Add(sum6, sum6, sum7, full);
        Add(sum0, sum0, sum2, full);
        Add(sum4, sum4, sum6, full);
        Add(sum0, sum0, sum4, full);
        StoreAlign<float, StoreDist::DIST_NORM_B32>(tmpSumAddr, sum0, full);
        if (update) {
            RegTensor<float> runningSum;
            RegTensor<float> expFactor;
            LoadAlign(expFactor, expMaxAddr);
            LoadAlign(runningSum, runningSumAddr);
            MulDstAdd(expFactor, runningSum, sum0, full);
            StoreAlign<float, StoreDist::DIST_NORM_B32>(runningSumAddr, expFactor, full);
        } else {
            StoreAlign<float, StoreDist::DIST_NORM_B32>(runningSumAddr, sum0, full);
        }
    }

    static constexpr uint32_t HALF_ROWS = Q_TILE / 2U;
    static constexpr uint32_t HALF_ROW_BYTES = HALF_ROWS * sizeof(float);
    static constexpr uint32_t HALF_P_BYTES = HALF_ROWS * KV_TILE * sizeof(ElementP);
    static constexpr uint32_t DN_PACK_BYTES = HALF_P_BYTES + 256U;
    static constexpr uint32_t DN_INDEX_BYTES = 256U;
    static constexpr uint32_t Q_INDEX_BYTES = Q_TILE * sizeof(uint32_t);
    static constexpr uint32_t Q_SCALE_EXPANDED_BYTES = HALF_ROWS * sizeof(float);
    static constexpr uint64_t SYNC_MODE = 4U;
    AscendC::LocalTensor<float> expSumUb_;
    AscendC::LocalTensor<float> nowMaxUb_;
    AscendC::LocalTensor<ElementP> pUb_[2];
    AscendC::LocalTensor<uint8_t> dnIndexUb_;
    AscendC::LocalTensor<uint32_t> qIndexUb_;
    AscendC::LocalTensor<float> qScaleExpandedUb_;
};

} // namespace SageAttention2Kernel

#endif // SAGE_ATTENTION2_SOFTMAX_H_
