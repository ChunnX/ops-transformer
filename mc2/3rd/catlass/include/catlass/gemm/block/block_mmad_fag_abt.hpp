/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

#ifndef CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_ABT_HPP
#define CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_ABT_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

////////////////////////////////////////////////////////////////////
namespace Catlass::Gemm::Block
{
    template <
        class L1TileShape_,
        class L0TileShape_,
        class ElementA_,
        class ElementB_,
        class ElementC_,
        class BiasType_,
        class TileCopy_,
        class TileMmad_>
    struct BlockMmadTla<
        MmadAscend950FAGABt,
        L1TileShape_,
        L0TileShape_,
        ElementA_,
        ElementB_,
        ElementC_,
        BiasType_,
        TileCopy_,
        TileMmad_>
    {
    public:
        using DispatchPolicy = MmadAscend950FAGABt;
        using ArchTag = typename DispatchPolicy::ArchTag;
        using L1TileShape = L1TileShape_;
        using L0TileShape = L0TileShape_;
        using ElementA = ElementA_;
        using ElementB = ElementB_;
        using ElementC = ElementC_;
        using TileMmad = TileMmad_;
        using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
        using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
        using TensorGM_C = tla::Tensor<AscendC::GlobalTensor<ElementC>, typename TileCopy_::LayoutC,
            tla::Coord<tla::_0, tla::_0>, AscendC::TPosition::GM>;
        using CopyL0CToGm = typename TileCopy_::template CopyL0CToDst<TensorGM_C>;
        using CubeAddrInfo = typename DispatchPolicy::CubeAddrInfo;
        using CubePingPongFlags = typename DispatchPolicy::CubePingPongFlags;

        static constexpr uint32_t C0_SIZE = 16;
        static constexpr uint32_t CUBE_TILE = 128;
        static constexpr uint32_t CUBE_M_BLOCK = 256;  // abt m-loop step
        static const uint32_t SIZE_ONE_K = 1024;
        static const uint32_t BLOCK_WORKSPACE = 16 * CUBE_TILE * CUBE_TILE;

        static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
        static constexpr uint32_t L1_PINGPONG_BUF_SIZE = CUBE_TILE * SIZE_ONE_K;
        static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = ArchTag::L0A_SIZE / STAGES;
        static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = ArchTag::L0B_SIZE / STAGES;
        static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = ArchTag::L0C_SIZE / STAGES;

        CATLASS_DEVICE
        BlockMmadTla(Arch::Resource<ArchTag> &resource)
        {
            abtCnt = 0;
            globalBlockOffset = AscendC::GetBlockIdx() * BLOCK_WORKSPACE;
            for (uint32_t i = 0; i < STAGES; i++) {
                l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(L1_PINGPONG_BUF_SIZE * i);
                l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(L1_PINGPONG_BUF_SIZE * (2 + i));
                l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i);
                l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i);
                l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementC>(L0C_PINGPONG_BUF_SIZE * i);
            }
        }

        CATLASS_DEVICE
        ~BlockMmadTla() {}

        CATLASS_DEVICE
        void abt_base_matmul(
            AscendC::LocalTensor<ElementA> &l1_a_tensor, AscendC::LocalTensor<ElementB> &l1_b_tensor,
            AscendC::GlobalTensor<ElementC> *gOut, CubePingPongFlags &flags,
            uint64_t headdim, int32_t l1_m_size, int32_t l1_n_size, bool upper_right_flag)
        {
            int32_t l1_m_size_ = l1_m_size;
            int32_t l1_n_size_ = l1_n_size;

            uint32_t m0_ = CUBE_TILE;
            uint32_t n0_ = CUBE_TILE;
            uint32_t k0_ = CUBE_TILE;

            uint32_t m_mad_ = CUBE_TILE;
            uint32_t n_mad_ = CUBE_TILE;
            uint32_t k_mad_ = CUBE_TILE;

            for (int n_offset = 0; n_offset < l1_n_size_; n_offset += CUBE_TILE) {
                n_mad_ = (l1_n_size_ - n_offset) < CUBE_TILE ? (l1_n_size_ - n_offset) : CUBE_TILE;
                n0_ = RoundUp<C0_SIZE>(n_mad_);

                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3 + flags.l0B + 2);
                auto layoutBInL1Full = tla::MakeLayout<ElementB, layout::nZ>(static_cast<uint32_t>(headdim), l1_n_size_);
                auto layoutBInL1Slice = tla::MakeLayoutTile(layoutBInL1Full, tla::MakeShape(k0_, n_mad_));
                auto layoutBInL0 = tla::MakeLayout<ElementB, layout::nZ>(k0_, n_mad_);
                auto tensorL1B = tla::MakeTensor(l1_b_tensor[n_offset * C0_SIZE], layoutBInL1Slice, tla::MakeCoord(0u, 0u), Arch::PositionL1{});
                auto tensorL0B = tla::MakeTensor(l0BTensor[flags.l0B], layoutBInL0, Arch::PositionL0B{});
                copyL1ToL0B(tensorL0B, tensorL1B);

                AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(flags.l0B + 2);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(flags.l0B + 2);

                for (int m_offset = 0; m_offset < l1_m_size_; m_offset += CUBE_TILE) {
                    m_mad_ = (l1_m_size_ - m_offset) < CUBE_TILE ? (l1_m_size_ - m_offset) : CUBE_TILE;
                    m0_ = RoundUp<C0_SIZE>(m_mad_);

                    bool l0_skip_flag = (upper_right_flag && m_offset == 0);

                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3 + flags.l0A);
                    if (!l0_skip_flag) {
                        auto layoutAInL1Full = tla::MakeLayout<ElementA, layout::zN>(l1_m_size_, static_cast<uint32_t>(headdim));
                        auto layoutAInL1Slice = tla::MakeLayoutTile(layoutAInL1Full, tla::MakeShape(m_mad_, k0_));
                        auto layoutAInL0 = tla::MakeLayout<ElementA, layout::zN>(m_mad_, k0_);
                        auto tensorL1A = tla::MakeTensor(l1_a_tensor[m_offset * C0_SIZE], layoutAInL1Slice, tla::MakeCoord(0u, 0u), Arch::PositionL1{});
                        auto tensorL0A = tla::MakeTensor(l0ATensor[flags.l0A], layoutAInL0, Arch::PositionL0A{});
                        copyL1ToL0A(tensorL0A, tensorL1A);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(flags.l0A);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(flags.l0A);
                    if (!l0_skip_flag) {
                        uint32_t m_modify = (m_mad_ == 1) ? 2 : m_mad_;
                        auto layoutAInL0 = tla::MakeLayout<ElementA, layout::zN>(m_mad_, k0_);
                        auto tensorL0A = tla::MakeTensor(l0ATensor[flags.l0A], layoutAInL0, Arch::PositionL0A{});
                        auto layoutInL0C = tla::MakeLayoutL0C(m_mad_, n_mad_);
                        auto tensorL0CTile = tla::MakeTensor(l0CTensor[flags.l0C], layoutInL0C, Arch::PositionL0C{});
                        tileMmad(tensorL0CTile, tensorL0A, tensorL0B, m_modify, n_mad_, k_mad_, true, 3);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3 + flags.l0A);
                    auto out_offset = abtCnt * CUBE_TILE * CUBE_TILE;

                    if (!l0_skip_flag) {
                        // GM 输出块 stride 固定为 CUBE_TILE（与原始 LayoutC(m,n,128) 一致）
                        auto layoutGM_C = tla::MakeLayout(tla::MakeShape(m_mad_, n_mad_), tla::MakeStride(static_cast<int64_t>(CUBE_TILE), tla::Int<1>{}));
                        auto tensorGM_C = tla::MakeTensor((*gOut)[out_offset], layoutGM_C, Arch::PositionGM{});
                        auto layoutInL0C = tla::MakeLayoutL0C(m_mad_, n_mad_);
                        auto tensorL0C = tla::MakeTensor(l0CTensor[flags.l0C], layoutInL0C, Arch::PositionL0C{});
                        copyL0CToGm(tensorGM_C, tensorL0C, 3);
                        abtCnt++;
                    }

                    flags.l0C = 1 - flags.l0C;
                    flags.l0A = 1 - flags.l0A;
                }

                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3 + flags.l0B + 2);
                flags.l0B = 1 - flags.l0B;
            }
        }

        CATLASS_DEVICE
        void operator()(const CubeAddrInfo &addrs,
                        AscendC::GlobalTensor<ElementA> &gA,
                        AscendC::GlobalTensor<ElementB> &gB,
                        AscendC::GlobalTensor<ElementC> &gC,
                        uint64_t nheads, uint64_t nheads_k, uint64_t headdim,
                        CubePingPongFlags &flags)
        {
            pingPongIdx = addrs.taskId % 2;
            globalBlockOffset =  AscendC::GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;
            for (uint32_t i = 0; i < addrs.blockLength; ++i) {
                abtCnt = 0;
                auto &shapeInfo = addrs.addrInfo[i];
                uint32_t km = shapeInfo.ky;
                uint32_t kn = shapeInfo.kx;

                uint32_t src_k_size_ = headdim;

                uint32_t l1_m_size_ = km;
                uint32_t l1_n_size_ = kn;
                uint32_t l1_k_size_ = src_k_size_;

                uint32_t m_loop = CeilDiv<CUBE_M_BLOCK>(km);
                uint32_t n_loop = CeilDiv<CUBE_TILE>(kn);

                bool upperRight = !shapeInfo.upperRight;
                bool lowerLeft = !shapeInfo.lowerLeft;

                for (int m_index = 0; m_index < m_loop; m_index++) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(flags.l1A + 2);
                    {
                        auto layoutTileATla = tla::MakeLayout(tla::MakeShape(km, static_cast<uint32_t>(headdim)),
                            tla::MakeStride(static_cast<int64_t>(nheads * headdim), tla::Int<1>{}));
                        auto layoutAInL1Tla = tla::MakeLayout<ElementA, layout::zN>(km, static_cast<uint32_t>(headdim));
                        auto tensorGM_A = tla::MakeTensor(gA[shapeInfo.left], layoutTileATla, Arch::PositionGM{});
                        auto tensorL1A = tla::MakeTensor(l1ATensor[flags.l1A], layoutAInL1Tla, Arch::PositionL1{});
                        Gemm::Tile::TileCopyTla<ArchTag, decltype(tensorGM_A), decltype(tensorL1A)> copyGmToL1ATla;
                        copyGmToL1ATla(tensorL1A, tensorGM_A);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(flags.l1A + 2);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(flags.l1A + 2);

                    for (int n_index = 0; n_index < n_loop; n_index++) {
                        l1_n_size_ = (n_index == n_loop - 1) ? (kn - n_index * CUBE_TILE) : CUBE_TILE;
                        bool upper_right_flag = (upperRight && n_index == n_loop - 1);

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(flags.l1B);
                        {
                            auto layoutTileBTla = tla::MakeLayout(tla::MakeShape(l1_k_size_, l1_n_size_),
                                tla::MakeStride(tla::Int<1>{}, static_cast<int64_t>(nheads_k * headdim)));
                            auto layoutBInL1Tla = tla::MakeLayout<ElementB, layout::nZ>(l1_k_size_, l1_n_size_);
                            auto tensorGM_B = tla::MakeTensor(gB[shapeInfo.right + n_index * CUBE_TILE * src_k_size_ * nheads_k], layoutTileBTla, Arch::PositionGM{});
                            auto tensorL1B = tla::MakeTensor(l1BTensor[flags.l1B], layoutBInL1Tla, Arch::PositionL1{});
                            Gemm::Tile::TileCopyTla<ArchTag, decltype(tensorGM_B), decltype(tensorL1B)> copyGmToL1BTla;
                            copyGmToL1BTla(tensorL1B, tensorGM_B);
                        }

                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(flags.l1B);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(flags.l1B);

                        AscendC::GlobalTensor<ElementC> gOutBlock = gC[shapeInfo.out + globalBlockOffset];
                        abt_base_matmul(l1ATensor[flags.l1A], l1BTensor[flags.l1B], &gOutBlock, flags, headdim, l1_m_size_, l1_n_size_, upper_right_flag);

                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(flags.l1B);
                        flags.l1B = 1 - flags.l1B;
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(flags.l1A + 2);
                    flags.l1A = 1 - flags.l1A;
                }
            }
        }

    protected:
        /// Data members
        TileMmad tileMmad;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToGm copyL0CToGm;

        uint32_t abtCnt = 0;
        AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
        AscendC::LocalTensor<ElementB> l1BTensor[STAGES];

        // L0A L0B L0C (pingpong arrays)
        AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
        AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
        AscendC::LocalTensor<ElementC> l0CTensor[STAGES];

        uint32_t pingPongIdx = 0;
        uint64_t globalBlockOffset = 0;
    };

} // namespace Catlass::Gemm::Block

#endif  // CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_ABT_HPP
