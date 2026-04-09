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

#ifndef CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_AB_HPP
#define CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_AB_HPP

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
        MmadAscend950FAGAB,
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
        using DispatchPolicy = MmadAscend950FAGAB;
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

        static const uint32_t C0_SIZE = 16;
        static const uint32_t CUBE_TILE = 128;
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
            globalBlockOffset = AscendC::GetBlockIdx() * 16 * CUBE_TILE * CUBE_TILE;
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
        void operator()(
            const CubeAddrInfo &addrs,
            AscendC::GlobalTensor<ElementA> &gLeftBase,
            AscendC::GlobalTensor<ElementB> &gRightBase,
            AscendC::GlobalTensor<ElementC> &gOutBase,
            uint64_t nheads, uint64_t nheads_k, uint64_t headdim,
            CubePingPongFlags &flags)
        {
            pingPongIdx = addrs.taskId % 2;
            globalBlockOffset =  AscendC::GetBlockIdx() * BLOCK_WORKSPACE * 2 + pingPongIdx * BLOCK_WORKSPACE;

            for (uint32_t i = 0; i < addrs.blockLength; ++i) {
                auto &shapeInfo = addrs.addrInfo[i];

                auto gLeft = gLeftBase[(shapeInfo.out + globalBlockOffset) * 2];
                auto gRight = gRightBase[shapeInfo.right];
                auto gOut = gOutBase[shapeInfo.left];

                uint32_t kn = shapeInfo.kx;
                uint32_t km = shapeInfo.ky;

                int32_t l1_m_size = km;
                int32_t l1_n_size = kn;

                int32_t l1_m_size_align = RoundUp<C0_SIZE>(l1_m_size);
                int32_t l1_n_size_align = RoundUp<C0_SIZE>(l1_n_size);
                int32_t l1_m_block_size_tail = (l1_m_size % CUBE_TILE) == 0 ? CUBE_TILE : (l1_m_size % CUBE_TILE);
                int32_t l1_n_block_size_tail = (l1_n_size % CUBE_TILE) == 0 ? CUBE_TILE : (l1_n_size % CUBE_TILE);
                int32_t l1_m_block_size_align_tail = (l1_m_size_align % CUBE_TILE) == 0 ? CUBE_TILE : (l1_m_size_align % CUBE_TILE);
                int32_t l1_n_block_size_align_tail = (l1_n_size_align % CUBE_TILE) == 0 ? CUBE_TILE : (l1_n_size_align % CUBE_TILE);

                int32_t m_loop = CeilDiv<CUBE_TILE>(km);
                int32_t n_loop = CeilDiv<CUBE_TILE>(kn);
                bool upperRight = !shapeInfo.upperRight;

                int32_t skip_num = 0;
                for (uint32_t n_loop_index = 0; n_loop_index < n_loop; n_loop_index++) {
                    int32_t n_remain = (n_loop_index == n_loop - 1) ? l1_n_block_size_tail : CUBE_TILE;
                    int32_t n_remain_align = (n_loop_index == n_loop - 1) ? l1_n_block_size_align_tail : CUBE_TILE;
                    bool l0_c_init_flag = (n_loop_index == 0);

                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(flags.l1B);

                    {
                        auto layoutTileBTla = tla::MakeLayout(tla::MakeShape(static_cast<uint32_t>(n_remain), static_cast<uint32_t>(headdim)),
                            tla::MakeStride(static_cast<int64_t>(nheads_k * headdim), tla::Int<1>{}));
                        auto layoutBInL1Tla = tla::MakeLayout<ElementB, layout::zN>(static_cast<uint32_t>(n_remain), static_cast<uint32_t>(headdim));
                        auto tensorGM_B = tla::MakeTensor(gRight[n_loop_index * nheads_k * CUBE_TILE * headdim], layoutTileBTla, Arch::PositionGM{});
                        auto tensorL1B = tla::MakeTensor(l1BTensor[flags.l1B], layoutBInL1Tla, Arch::PositionL1{});
                        Gemm::Tile::TileCopyTla<ArchTag, decltype(tensorGM_B), decltype(tensorL1B)> copyGmToL1B;
                        copyGmToL1B(tensorL1B, tensorGM_B);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(flags.l1B);

                    flags.l1A = 0;
                    for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
                        int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : CUBE_TILE;
                        bool is_skip = false;

                        if (n_loop_index == n_loop - 1 && m_loop_index == 0 && upperRight) {
                            skip_num++;
                            is_skip = true;
                        }

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(flags.l1A + 2);

                        if (!is_skip) {
                            auto layoutTileATla = tla::MakeLayout(tla::MakeShape(static_cast<uint32_t>(m_remain), static_cast<uint32_t>(n_remain)),
                                tla::MakeStride(static_cast<int64_t>(CUBE_TILE), tla::Int<1>{}));
                            auto layoutAInL1Tla = tla::MakeLayout<ElementA, layout::zN>(static_cast<uint32_t>(m_remain), static_cast<uint32_t>(n_remain));
                            auto tensorGM_A = tla::MakeTensor(gLeft[2 * (m_loop * n_loop_index + m_loop_index - skip_num) * CUBE_TILE * CUBE_TILE], layoutTileATla, Arch::PositionGM{});
                            auto tensorL1A = tla::MakeTensor(l1ATensor[flags.l1A], layoutAInL1Tla, Arch::PositionL1{});
                            Gemm::Tile::TileCopyFAGTla<ArchTag, decltype(tensorGM_A), decltype(tensorL1A)> copyGmToL1A;
                            copyGmToL1A(tensorL1A, tensorGM_A);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(flags.l1A + 2);
                        flags.l1A = 1 - flags.l1A;
                    }

                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(flags.l1B);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(flags.l0B + 2 + FLAG_SHIFT);

                    {
                        auto layoutBInL1Tla = tla::MakeLayout<ElementB, layout::zN>(static_cast<uint32_t>(n_remain), static_cast<uint32_t>(headdim));
                        auto layoutBInL0Tla = tla::MakeLayout<ElementB, layout::nZ>(static_cast<uint32_t>(n_remain), static_cast<uint32_t>(headdim));
                        auto tensorL1B = tla::MakeTensor(l1BTensor[flags.l1B], layoutBInL1Tla, tla::MakeCoord(0u, 0u), Arch::PositionL1{});
                        auto tensorL0B = tla::MakeTensor(l0BTensor[flags.l0B], layoutBInL0Tla, Arch::PositionL0B{});
                        copyL1ToL0B(tensorL0B, tensorL1B);
                    }

                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(flags.l0B + 2);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(flags.l1B);

                    flags.l1A = 0;
                    flags.l0A = 0;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(flags.l0B + 2);

                    for (uint32_t m_loop_index = 0; m_loop_index < m_loop; m_loop_index++) {
                        int32_t m_remain = (m_loop_index == m_loop - 1) ? l1_m_block_size_tail : CUBE_TILE;
                        int32_t m_remain_align = (m_loop_index == m_loop - 1) ? l1_m_block_size_align_tail : CUBE_TILE;
                        bool is_skip = false;

                        if (n_loop_index == n_loop - 1 && m_loop_index == 0 && upperRight) {
                            is_skip = true;
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(flags.l1A + 2);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(flags.l0A + FLAG_SHIFT);
                        if (!is_skip) {
                            auto layoutAInL1Tla = tla::MakeLayout<ElementA, layout::zN>(static_cast<uint32_t>(m_remain_align), static_cast<uint32_t>(n_remain_align));
                            auto layoutAInL0Tla = tla::MakeLayout<ElementA, layout::zN>(static_cast<uint32_t>(m_remain), static_cast<uint32_t>(n_remain));
                            auto tensorL1A = tla::MakeTensor(l1ATensor[flags.l1A], layoutAInL1Tla, tla::MakeCoord(0u, 0u), Arch::PositionL1{});
                            auto tensorL0A = tla::MakeTensor(l0ATensor[flags.l0A], layoutAInL0Tla, Arch::PositionL0A{});
                            copyL1ToL0A(tensorL0A, tensorL1A);
                        }
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(flags.l0A);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(flags.l1A + 2);

                        bool last_k = false;
                        last_k = (m_loop_index == 0 && upperRight) ? n_loop_index == n_loop - 2 : n_loop_index == n_loop - 1;

                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(flags.l0A);
                        if (!is_skip) {
                            uint16_t m_modify = (m_remain == 1) ? 2 : m_remain;

                            auto layoutAInL0Tla = tla::MakeLayout<ElementA, layout::zN>(static_cast<uint32_t>(m_remain), static_cast<uint32_t>(n_remain));
                            auto layoutBInL0Tla = tla::MakeLayout<ElementB, layout::nZ>(static_cast<uint32_t>(n_remain), static_cast<uint32_t>(headdim));
                            auto layoutInL0C = tla::MakeLayoutL0C(static_cast<uint32_t>(m_modify), static_cast<uint32_t>(headdim));
                            auto tensorL0A = tla::MakeTensor(l0ATensor[flags.l0A], layoutAInL0Tla, Arch::PositionL0A{});
                            auto tensorL0B = tla::MakeTensor(l0BTensor[flags.l0B], layoutBInL0Tla, Arch::PositionL0B{});
                            auto tensorL0CTile = tla::MakeTensor(l0CTensor[m_loop_index], layoutInL0C, Arch::PositionL0C{});
                            tileMmad(tensorL0CTile, tensorL0A, tensorL0B,
                                static_cast<uint32_t>(m_modify), static_cast<uint32_t>(headdim), static_cast<uint32_t>(n_remain),
                                l0_c_init_flag, static_cast<uint8_t>(last_k ? 3 : 2));
                        }
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(flags.l0A + FLAG_SHIFT);

                        if (!is_skip && last_k) {
                            AscendC::SetAtomicType<float>();
                            uint16_t m_modify = (m_remain == 1) ? 2 : m_remain;
                            auto layoutGM_C = tla::MakeLayout(tla::MakeShape(static_cast<uint32_t>(m_modify), static_cast<uint32_t>(headdim)),
                                tla::MakeStride(static_cast<int64_t>(nheads * headdim), tla::Int<1>{}));
                            auto tensorGM_C = tla::MakeTensor((gOut)[m_loop_index * nheads * CUBE_TILE * headdim], layoutGM_C, Arch::PositionGM{});
                            auto layoutInL0C = tla::MakeLayoutL0C(static_cast<uint32_t>(m_modify), static_cast<uint32_t>(headdim));
                            auto tensorL0C = tla::MakeTensor(l0CTensor[m_loop_index], layoutInL0C, Arch::PositionL0C{});
                            copyL0CToGm(tensorGM_C, tensorL0C, 3);
                            AscendC::SetAtomicNone();
                        }
                        flags.l1A = 1 - flags.l1A;
                        flags.l0A = 1 - flags.l0A;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(flags.l0B + 2 + FLAG_SHIFT);

                    flags.l0B = 1 - flags.l0B;
                    flags.l1B = 1 - flags.l1B;
                }
            }
        }

    protected:
        TileMmad tileMmad;
        CopyL1ToL0A copyL1ToL0A;
        CopyL1ToL0B copyL1ToL0B;
        CopyL0CToGm copyL0CToGm;

        AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
        AscendC::LocalTensor<ElementB> l1BTensor[STAGES];

        // L0A L0B L0C (pingpong arrays)
        AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
        AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
        AscendC::LocalTensor<float> l0CTensor[STAGES];

        uint32_t FLAG_SHIFT = 3;

        uint32_t pingPongIdx = 0;
        uint64_t globalBlockOffset = 0;
    };

} // namespace Catlass::Gemm::Block

#endif  // CATLASS_MATMUL_BLOCK_BLOCK_MMAD_FAG_AB_HPP
