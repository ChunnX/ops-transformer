/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epilogue_pertoken_v2.hpp
 * \brief
 */

#ifndef CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP
#define CATLASS_EPILOGUE_BLOCK_EPILOGUE_PER_TOKEN_V2_ONLY_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/matrix_coord.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/detail/callback.hpp"

#include "const_args.hpp"
#include "hccl_shmem.hpp"
#include "layout3d.hpp"
#include "per_token_v2_route_utils.hpp"
#include "dispatch_policy_custom.hpp"

namespace Catlass::Epilogue::Block {

template <bool IS_A2_>
struct IS_A2_T {
    static constexpr bool Value = IS_A2_;
};

namespace detail {

// A2 and A3 share the tile pipeline. Platform and dequantization choices are
// fixed at compile time, retaining each policy's event and copy sequence.
template <class DispatchPolicy_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_, bool IS_A2_,
          bool NEED_DEQUANT_>
class PerTokenV2Impl {
public:
    using DispatchPolicy = DispatchPolicy_;
    using ArchTag = typename DispatchPolicy::ArchTag;
    static constexpr uint32_t UB_STAGES = DispatchPolicy::UB_STAGES;
    static constexpr bool IS_A2 = IS_A2_;
    static constexpr bool NEED_DEQUANT = NEED_DEQUANT_;

    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;
    using ElementPerTokenScale = float;
    using LayoutPerTokenScale = LayoutPerTokenScale_;
    using ElementD = typename DType_::Element;
    using LayoutD = typename DType_::Layout;

    using CopyScaleGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, Gemm::GemmType<float, layout::VectorLayout>>;
    using CopyGmToUbC = typename TileCopy_::CopyGmToUbC;
    using CopyUbToGmD = typename TileCopy_::CopyUbToGmD;

    struct Params {
        __gm__ int32_t *ptrTokenPerExpert{nullptr};
        int32_t EP;
        int32_t expertPerRank;
        int32_t n2;
        LayoutC layoutC;
        int32_t n0;
        int32_t rank;
        HcclShmem<IS_A2> shmem;
        int64_t offsetD;
        int64_t offsetWinOutD{0};
        int32_t serverId{0};
        Layout3D tokenPerExpertLayout;
        bool useUbRouteCache{false};
        int32_t ubRouteStride{0};
        AscendC::LocalTensor<int32_t> ubTokenPerExpert;
        AscendC::LocalTensor<int32_t> ubPreSumBeforeRank;

        CATLASS_DEVICE
        Params() {};
        CATLASS_DEVICE
        Params(int32_t EP_, int32_t expertPerRank_, int32_t rank_, __gm__ int32_t *ptrTokenPerExpert_, LayoutC layoutC_,
               int32_t n2_, int32_t n0_, HcclShmem<IS_A2> &shmem_, int64_t offsetD_, Layout3D tokenPerExpertLayout_)
            : ptrTokenPerExpert(ptrTokenPerExpert_),
              EP(EP_),
              expertPerRank(expertPerRank_),
              rank(rank_),
              layoutC(layoutC_),
              n2(n2_),
              n0(n0_),
              shmem(shmem_),
              offsetD(offsetD_),
              tokenPerExpertLayout(tokenPerExpertLayout_)
        {}
        CATLASS_DEVICE
        Params(int32_t EP_, int32_t expertPerRank_, int32_t rank_, __gm__ int32_t *ptrTokenPerExpert_, LayoutC layoutC_,
               int32_t n2_, int32_t n0_, HcclShmem<IS_A2> &shmem_, int64_t offsetD_, int64_t offsetWinOutD_,
               int32_t serverId_, Layout3D tokenPerExpertLayout_)
            : ptrTokenPerExpert(ptrTokenPerExpert_),
              EP(EP_),
              expertPerRank(expertPerRank_),
              rank(rank_),
              layoutC(layoutC_),
              n2(n2_),
              n0(n0_),
              shmem(shmem_),
              offsetD(offsetD_),
              offsetWinOutD(offsetWinOutD_),
              serverId(serverId_),
              tokenPerExpertLayout(tokenPerExpertLayout_)
        {}
    };

    CATLASS_DEVICE
    PerTokenV2Impl(Arch::Resource<ArchTag> const &resource, Params const &params = Params{})
        : params(params)
    {
        n0 = params.n0;
        size_t ubOffset = 0;
        for (int32_t i = 0; i < 2; i++) {
            ubCList[i] = resource.ubBuf.template GetBufferByByte<ElementC>(ubOffset);
            ubOffset += max_len * sizeof(ElementC);
            ubDList[i] = resource.ubBuf.template GetBufferByByte<ElementD>(ubOffset);
            ubOffset += max_len * sizeof(ElementD);
            ubFp32List[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += max_len * sizeof(float);
            scaleUbList[i] = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
            ubOffset += (max_len / n0) * sizeof(float);
            source_scale_offset[i] = -1;
        }
        tokenPerExpert.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.ptrTokenPerExpert));
        tokenPerExpertLayout = params.tokenPerExpertLayout;
        is_ping = true;
    }

    CATLASS_DEVICE
    void SetFlag()
    {
        if constexpr (!IS_A2) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        }
    }

    CATLASS_DEVICE
    void InitFlag() {}

    CATLASS_DEVICE
    void Finalize()
    {
        if constexpr (IS_A2 && NEED_DEQUANT) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        } else if constexpr (IS_A2) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    CATLASS_DEVICE
    ~PerTokenV2Impl() {}

    CATLASS_DEVICE int32_t GetLenRankInExpert(int32_t dstEpIdx, int32_t groupIdx) const
    {
        return ::Catlass::Epilogue::Block::detail::GetLenRankInExpert(params, tokenPerExpert, tokenPerExpertLayout,
                                                                      dstEpIdx, groupIdx);
    }

    CATLASS_DEVICE int32_t GetDstExpertOffset(int32_t dstEpIdx, int32_t groupIdx,
                                              AscendC::GlobalTensor<int32_t> const &preSumBeforeRank) const
    {
        return ::Catlass::Epilogue::Block::detail::GetDstExpertOffset(params, preSumBeforeRank, dstEpIdx, groupIdx);
    }

    template <typename TileElement>
    CATLASS_DEVICE void CopyRoutedSlice(AscendC::LocalTensor<TileElement> &ubTile, GemmCoord &blockCoord,
                                        GemmCoord &actualBlockShape, int32_t preSrcExpertSum, int32_t dstEpIdx,
                                        int32_t stData, uint32_t lenData, uint32_t dstOffsetInExpert,
                                        int32_t dstExpertOffset, int32_t tileOffset)
    {
        AscendC::GlobalTensor<ElementD> gmRemotePeer;
        __gm__ void *dstPeermemPtr = params.shmem(params.offsetD, dstEpIdx);
        gmRemotePeer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD *>(dstPeermemPtr));
        MatrixCoord dstOffset{dstOffsetInExpert + dstExpertOffset, blockCoord.n()};
        int64_t gmDstOffset = params.layoutC.GetOffset(dstOffset);
        auto gmTileD = gmRemotePeer[gmDstOffset];
        LayoutC layoutGM2{lenData, actualBlockShape.n(), params.n2};
        LayoutC layoutUB2{lenData, actualBlockShape.n(), n0};
        if constexpr (IS_A2) {
            bool isCrossServer = (dstEpIdx / SERVER_RANK_SIZE_A2) != params.serverId;
            if (isCrossServer) {
                AscendC::GlobalTensor<ElementD> gmLocalWindowsOut;
                gmLocalWindowsOut.SetGlobalBuffer(
                    reinterpret_cast<__gm__ ElementD *>(params.shmem.windowsOutAddr() + params.offsetWinOutD));
                MatrixCoord srcOffset{(uint32_t)(stData + preSrcExpertSum), blockCoord.n()};
                int64_t gmSrcOffset = params.layoutC.GetOffset(srcOffset);
                auto gmTileLocal = gmLocalWindowsOut[gmSrcOffset];
                copyUbToGmD(gmTileLocal, ubTile[tileOffset * n0], layoutGM2, layoutUB2);
            } else {
                copyUbToGmD(gmTileD, ubTile[tileOffset * n0], layoutGM2, layoutUB2);
            }
        } else {
            copyUbToGmD(gmTileD, ubTile[tileOffset * n0], layoutGM2, layoutUB2);
        }
    }

    template <typename TileElement>
    CATLASS_DEVICE void CopyRoutedTile(AscendC::LocalTensor<TileElement> &ubTile, GemmCoord &blockCoord,
                                       GemmCoord &actualBlockShape, int32_t groupIdx, int32_t preSrcExpertSum,
                                       AscendC::GlobalTensor<int32_t> const &preSumBeforeRank, int32_t stTile,
                                       int32_t edTile, int32_t &preSumRankInExpert, int32_t &tileOffset)
    {
        for (int32_t dstEpIdx = 0; dstEpIdx < params.EP; dstEpIdx++) {
            int32_t lenRankInExpert = GetLenRankInExpert(dstEpIdx, groupIdx);
            int32_t dstExpertOffset = GetDstExpertOffset(dstEpIdx, groupIdx, preSumBeforeRank);
            auto routeRange =
                ::Catlass::Epilogue::Block::detail::GetRouteRange(lenRankInExpert, stTile, edTile, preSumRankInExpert);
            if (routeRange.action == ::Catlass::Epilogue::Block::detail::RouteAction::STOP) {
                break;
            }
            if (routeRange.action == ::Catlass::Epilogue::Block::detail::RouteAction::SKIP) {
                continue;
            }
            int32_t stData = routeRange.stData;
            uint32_t lenData = routeRange.lenData;
            uint32_t dstOffsetInExpert = routeRange.dstOffsetInExpert;
            CopyRoutedSlice(ubTile, blockCoord, actualBlockShape, preSrcExpertSum, dstEpIdx, stData, lenData,
                            dstOffsetInExpert, dstExpertOffset, tileOffset);
            tileOffset += lenData;
        }
    }

    template <typename EventId>
    CATLASS_DEVICE void ApplyTokenScale(AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale,
                                        GemmCoord &blockCoord, GemmCoord &actualBlockShape, int32_t preSrcExpertSum,
                                        EventId event_id, EventId event_id_2, AscendC::LocalTensor<float> &ubCFp32,
                                        AscendC::LocalTensor<ElementPerTokenScale> &scaleUb)
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE2>(event_id_2);

        int32_t gmScaleOffset = preSrcExpertSum + blockCoord.m();
        layout::VectorLayout scaleLayout{actualBlockShape.m()};
        if (source_scale_offset[event_id] != gmScaleOffset) {
            source_scale_offset[event_id] = gmScaleOffset;
            copyScaleGmToUb(scaleUb, gmPerTokenScale[gmScaleOffset], scaleLayout, scaleLayout);
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id_2);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id_2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(event_id_2);
        AscendC::PipeBarrier<PIPE_V>();
        for (int32_t row = 0; row < actualBlockShape.m(); ++row) {
            float scale = scaleUb(row);
            Muls<float, false>(ubCFp32[n0 * row], ubCFp32[n0 * row], scale, -1, (actualBlockShape.n() + 127) / 128 * 2,
                               {1, 1, 8, 8});
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const &gmC,
                    AscendC::GlobalTensor<ElementPerTokenScale> const &gmPerTokenScale, GemmCoord &blockCoord,
                    GemmCoord &actualBlockShape, int32_t groupIdx, int32_t preSrcExpertSum,
                    AscendC::GlobalTensor<int32_t> preSumBeforeRank)
    {
        if constexpr (!NEED_DEQUANT) {
            (void)gmPerTokenScale;
            operator()(gmC, blockCoord, actualBlockShape, groupIdx, preSrcExpertSum, preSumBeforeRank);
            return;
        }
        is_ping = !is_ping;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;
        auto event_id_2 = is_ping ? EVENT_ID2 : EVENT_ID3;

        auto &ubC = ubCList[is_ping];
        auto &ubD = ubDList[is_ping];
        using GmCOffset = std::conditional_t<IS_A2, int32_t, int64_t>;
        GmCOffset gmCOffset =
            static_cast<GmCOffset>(preSrcExpertSum) * params.n2 + blockCoord.m() * params.n2 + blockCoord.n();
        auto gmTileC = gmC[gmCOffset];
        auto &ubCFp32 = ubFp32List[is_ping];
        auto &scaleUb = scaleUbList[is_ping];

        LayoutC layoutGM{actualBlockShape.m(), actualBlockShape.n(), params.n2};
        LayoutC layoutUB{actualBlockShape.m(), actualBlockShape.n(), n0};

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event_id);
        copyGmToUbC(ubC, gmTileC, layoutUB, layoutGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event_id);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event_id);
        AscendC::Cast<float, ElementC, false>(ubCFp32, ubC, AscendC::RoundMode::CAST_NONE, -1, repeat, {1, 1, 8, 4});
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id);

        ApplyTokenScale(gmPerTokenScale, blockCoord, actualBlockShape, preSrcExpertSum, event_id, event_id_2, ubCFp32,
                        scaleUb);

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event_id);
        AscendC::Cast<ElementD, float, false>(ubD, ubCFp32, AscendC::RoundMode::CAST_RINT, -1, repeat, {1, 1, 4, 8});
        AscendC::SetFlag<AscendC::HardEvent::S_MTE2>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event_id_2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event_id);

        int32_t lenTile = actualBlockShape.m();
        int32_t stTile = blockCoord.m();
        int32_t edTile = stTile + lenTile;
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event_id);
        CopyRoutedTile(ubD, blockCoord, actualBlockShape, groupIdx, preSrcExpertSum, preSumBeforeRank, stTile, edTile,
                       preSumRankInExpert, tileOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event_id);
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> const &gmC, GemmCoord &blockCoord, GemmCoord &actualBlockShape,
                    int32_t groupIdx, int32_t preSrcExpertSum, AscendC::GlobalTensor<int32_t> preSumBeforeRank)
    {
        is_ping = !is_ping;
        auto event_id = is_ping ? EVENT_ID0 : EVENT_ID1;

        auto &ubC = ubCList[is_ping];
        int32_t gmCOffset = preSrcExpertSum * params.n2 + blockCoord.m() * params.n2 + blockCoord.n();
        auto gmTileC = gmC[gmCOffset];

        LayoutC layoutGM{actualBlockShape.m(), actualBlockShape.n(), params.n2};
        LayoutC layoutUB{actualBlockShape.m(), actualBlockShape.n(), n0};

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
        copyGmToUbC(ubC, gmTileC, layoutUB, layoutGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);

        int32_t lenTile = actualBlockShape.m();
        int32_t stTile = blockCoord.m();
        int32_t edTile = stTile + lenTile;
        int32_t preSumRankInExpert = 0;
        int32_t tileOffset = 0;

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event_id);
        CopyRoutedTile(ubC, blockCoord, actualBlockShape, groupIdx, preSrcExpertSum, preSumBeforeRank, stTile, edTile,
                       preSumRankInExpert, tileOffset);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event_id);
    }

private:
    Params params;
    AscendC::LocalTensor<ElementC> ubCList[UB_STAGES];
    AscendC::LocalTensor<ElementD> ubDList[UB_STAGES];
    AscendC::LocalTensor<float> ubFp32List[UB_STAGES];
    AscendC::LocalTensor<float> scaleUbList[UB_STAGES];
    int32_t source_scale_offset[UB_STAGES];

    int32_t max_len = 8 * 32 / 4 * 128;
    int32_t n0;
    bool is_ping = false;

    int32_t repeat = 128;

    CopyGmToUbC copyGmToUbC;
    CopyUbToGmD copyUbToGmD;

    CopyScaleGmToUb copyScaleGmToUb;
    AscendC::GlobalTensor<int32_t> tokenPerExpert;
    Layout3D tokenPerExpertLayout;
};

} // namespace detail

// A3 retains its IS_A2_T dispatch tag and its SetFlag/Finalize protocol.
template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_, bool IS_A2_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>, CType_, Gemm::GemmType<float, LayoutPerTokenScale_>,
                    DType_, TileCopy_, IS_A2_T<IS_A2_>>
    : public detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>, CType_, LayoutPerTokenScale_, DType_,
                                    TileCopy_, IS_A2_, true> {
    using Base = detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2<UB_STAGES_>, CType_, LayoutPerTokenScale_,
                                        DType_, TileCopy_, IS_A2_, true>;

public:
    using Base::Base;
};

template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantV2Int8<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileCopy_>
    : public detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2Int8<UB_STAGES_>, CType_, LayoutPerTokenScale_,
                                    DType_, TileCopy_, true, true> {
    using Base = detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2Int8<UB_STAGES_>, CType_, LayoutPerTokenScale_,
                                        DType_, TileCopy_, true, true>;

public:
    using Base::Base;
};

template <uint32_t UB_STAGES_, class CType_, class LayoutPerTokenScale_, class DType_, class TileCopy_>
class BlockEpilogue<EpilogueAtlasA2PerTokenDequantV2BF16<UB_STAGES_>, CType_,
                    Gemm::GemmType<float, LayoutPerTokenScale_>, DType_, TileCopy_>
    : public detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2BF16<UB_STAGES_>, CType_, LayoutPerTokenScale_,
                                    DType_, TileCopy_, true, false> {
    using Base = detail::PerTokenV2Impl<EpilogueAtlasA2PerTokenDequantV2BF16<UB_STAGES_>, CType_, LayoutPerTokenScale_,
                                        DType_, TileCopy_, true, false>;

public:
    using Base::Base;
};

} // namespace Catlass::Epilogue::Block
#endif
