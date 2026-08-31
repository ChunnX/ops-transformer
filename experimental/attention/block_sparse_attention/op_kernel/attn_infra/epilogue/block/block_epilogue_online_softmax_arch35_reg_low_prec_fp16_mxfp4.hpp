/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP4_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP4_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/epilogue/tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"
#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "mxfp4_vf/vf_init_nd2nz_indexes_duplicate_mxfp4.h"
#include "mxfp4_vf/vf_mask_invalid_rows_mxfp4.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_qs128_kvs256.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_qs64_kvs256.h"
#include "mxfp4_vf/vf_mm1_res_pre_padding_align_kvs32_multi.h"
#include "mxfp4_vf/vf_mm1_res_pre_padding_align_kvs32_multi_qs64.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_align_qs128_kvs32_multi.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_align_qs128_kvs32.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_align_qs64_kvs32_multi.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_align_qs64_kvs32.h"
#include "mxfp4_vf/vf_softmax_all_invalid.h"

namespace NpuArch::Epilogue::Block {

using namespace MXFP4Kernel;

template <MXQuantMode MX_QUANT_MODE_, class OutputType_, class LayoutS_, class ScaleType_>
class BlockEpilogue<EpilogueOnlineSoftmaxBsaMX<true, MX_QUANT_MODE_>, OutputType_, Gemm::GemmType<half, LayoutS_>,
                    ScaleType_> {
public:
    using DispatchPolicy = EpilogueOnlineSoftmaxBsaMX<true, MX_QUANT_MODE_>;
    static constexpr MXQuantMode MX_QUANT_MODE = DispatchPolicy::MX_QUANT_MODE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element; // P
    using ElementInput = half;                           // S
    using ElementMax = ElementInput;
    using ElementDisguiseP = uint8_t;
    using ElementPScale = typename ScaleType_::Element;
    using ElementIndex = uint8_t;
    using LayoutPL1 = typename OutputType_::Layout;
    using LayoutPUB = layout::RowMajor;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag> &resource, ElementInput NEG_LOG2_CX)
    {
        // FIXPIPE<->V 区
        for (uint32_t i = 0; i < MXFP4::UB_S_BUF_CNT; i++) {
            sUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(MXFP4::UB_S_BUF_OFFSET +
                                                                                 MXFP4::UB_S_INNER_BUF_OFFSET * i);
        }

        // 输出缓冲区
        for (uint32_t i = 0; i < MXFP4::UB_P_BUF_CNT; i++) {
            pUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDisguiseP>(MXFP4::UB_P_BUF_OFFSET +
                                                                                     MXFP4::UB_P_INNER_BUF_OFFSET * i);
        }

        // 常驻 buffer
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GROUP_MAX_CNT; i++) {
            localGroupMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_LOCAL_GROUP_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GROUP_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        indexUBTensor = resource.ubBuf.template GetBufferByByte<ElementIndex>(MXFP4::UB_INDEX_BUF_OFFSET);

        NEG_LOG2_CX_ = NEG_LOG2_CX;
        //  Init Tensor
        Mxfp4VF::InitIndexesAndDuplicateCallVF<ElementMax>(indexUBTensor, localGlobalMaxUBTensor[0]);
    }
    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorL1P>
    __aicore__ inline void operator()(TensorL1P &l1PTensorTla, GemmCoord &actualBlockShape, TileInfo const &tileInfo,
                                      TaskInfo const &taskInfo)
    {
        uint32_t spBuffIdx = GetSPBufferIdx(tileInfo.loop);
        // 跨 tile 同步: 每个 vec core 都需要一个跨 tile 的同步
        if (tileInfo.isKvsFirstTilePerCore) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_GMAX_UB_TO_L1_BUF0_FLAG + tileInfo.tileMaxIdx);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);

        const uint32_t actKvsTile = actualBlockShape.m();
        const uint32_t actQsTile = actualBlockShape.n();
        // 只有「任务首 tile」或「TileGroup 首 tile」时 global_max 缓冲才是未初始化/轮转残留,
        // 需要 SoftmaxAllInvalidCallVF 兜底写 -60000; 其余tile 一律不写, 保住跨 tile 累积的 running max
        const bool gmaxNeedInit = (tileInfo.isFirstKvsTile || tileInfo.isTileGoupFirstTile);
        // 槽位只在这两个时机切换(kernel:704-708), 新槽位是 4 个 TileGroup 之前的残留 -> 落一次安全初值。
        // 无效 tile 一律不再碰 global_max(见 SoftmaxAllInvalidCallVF 注释), running max 只由有效数据推进。
        if (gmaxNeedInit) {
            Mxfp4VF::InitGlobalMaxCallVF<ElementMax>(localGlobalMaxUBTensor[tileInfo.tileMaxIdx]);
            AscendC::PipeBarrier<PIPE_V>();
        }
        const uint32_t groupMaxIdx = GetLocalGroupMaxBufIdx(tileInfo.loop);
        if (actKvsTile == MXFP4::KVS_BASE_SIZE) {
            if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                // qs=128, kvs=256 按 chunk 分流: Y==0 -> allInvalid(P/LGM 兜底 + global_max 置极负),
                // Y!=0 -> -inf mask + 单 chunk softmax。
                // allInvalid 必须全部先于有效 chunk 的 softmax: 先给出 global_max 极负基值,
                // 有效 chunk softmax 从基值 max 上来; 否则 Y1>0&&Y2==0 时 chunk1 的 allInvalid
                // 会覆盖 chunk0 已写回的 global_max(反之亦然)。
                if (tileInfo.validRowsY1 == 0) {
                    Mxfp4VF::SoftmaxAllInvalidCallVF<ElementInput, true, 4, 1, 0, 0>(
                        pUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx]);
                }
                if (tileInfo.validRowsY2 == 0) {
                    Mxfp4VF::SoftmaxAllInvalidCallVF<ElementInput, true, 4, 1, 0, 1>(
                        pUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx]);
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (tileInfo.validRowsY1 != 0) {
                    Mxfp4VF::MaskInvalidRowsToMinValueCallVF<ElementInput, false>(
                        sUBTensor[spBuffIdx], 0, tileInfo.validRowsY1, MXFP4::KVS_BASE_SIZE / 2);
                    AscendC::PipeBarrier<PIPE_V>();
                    DbgDumpSFirstCol("qs128kvs256", tileInfo.validRowsY1, 0, MXFP4::KVS_BASE_SIZE / 2, 0,
                                     spBuffIdx, (uint32_t)tileInfo.curKvsTileLoopIdx);
                    Mxfp4VF::SoftmaxWithGroupMaxQs128Kvs256ChunkCallVF<MX_QUANT_MODE, true, ElementInput,
                        ElementDisguiseP, MXFP4::QS_BASE_SIZE, 0>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                        tileInfo.validRowsY1);
                }
                if (tileInfo.validRowsY2 != 0) {
                    Mxfp4VF::MaskInvalidRowsToMinValueCallVF<ElementInput, false>(
                        sUBTensor[spBuffIdx], 128, tileInfo.validRowsY2, MXFP4::KVS_BASE_SIZE / 2);
                    AscendC::PipeBarrier<PIPE_V>();
                    DbgDumpSFirstCol("qs128kvs256", tileInfo.validRowsY2, 0, MXFP4::KVS_BASE_SIZE / 2, 128,
                                     spBuffIdx, (uint32_t)tileInfo.curKvsTileLoopIdx);
                    Mxfp4VF::SoftmaxWithGroupMaxQs128Kvs256ChunkCallVF<MX_QUANT_MODE, true, ElementInput,
                        ElementDisguiseP, MXFP4::QS_BASE_SIZE, 1>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                        tileInfo.validRowsY2);
                }
            } else {
                // qs=64, kvs=256 按 chunk 分流, 同 qs=128 (每行仅 64 half, HAS_HIGH_OFF=false)。
                if (tileInfo.validRowsY1 == 0) {
                    Mxfp4VF::SoftmaxAllInvalidCallVF<ElementInput, false, 4, 1, 0, 0>(
                        pUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx]);
                }
                if (tileInfo.validRowsY2 == 0) {
                    Mxfp4VF::SoftmaxAllInvalidCallVF<ElementInput, false, 4, 1, 0, 1>(
                        pUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx]);
                }
                AscendC::PipeBarrier<PIPE_V>();
                if (tileInfo.validRowsY1 != 0) {
                    Mxfp4VF::MaskInvalidRowsToMinValueCallVF<ElementInput, true>(
                        sUBTensor[spBuffIdx], 0, tileInfo.validRowsY1, MXFP4::KVS_BASE_SIZE / 2);
                    AscendC::PipeBarrier<PIPE_V>();
                    DbgDumpSFirstCol("qs64kvs256", tileInfo.validRowsY1, 0, MXFP4::KVS_BASE_SIZE / 2, 0,
                                     spBuffIdx, (uint32_t)tileInfo.curKvsTileLoopIdx);
                    Mxfp4VF::SoftmaxWithGroupMaxQs64Kvs256ChunkCallVF<MX_QUANT_MODE, true, ElementInput,
                        ElementDisguiseP, MXFP4::QS_BASE_SIZE, 0>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                        tileInfo.validRowsY1);
                }
                if (tileInfo.validRowsY2 != 0) {
                    Mxfp4VF::MaskInvalidRowsToMinValueCallVF<ElementInput, true>(
                        sUBTensor[spBuffIdx], 128, tileInfo.validRowsY2, MXFP4::KVS_BASE_SIZE / 2);
                    AscendC::PipeBarrier<PIPE_V>();
                    DbgDumpSFirstCol("qs64kvs256", tileInfo.validRowsY2, 0, MXFP4::KVS_BASE_SIZE / 2, 128,
                                     spBuffIdx, (uint32_t)tileInfo.curKvsTileLoopIdx);
                    Mxfp4VF::SoftmaxWithGroupMaxQs64Kvs256ChunkCallVF<MX_QUANT_MODE, true, ElementInput,
                        ElementDisguiseP, MXFP4::QS_BASE_SIZE, 1>(
                        pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                        localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                        tileInfo.validRowsY2);
                }
            }

        } else {
            // kvs padding 32 multi
            if (actKvsTile != tileInfo.kvsActBaseTileAlign32) {
                if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                    Mxfp4VF::Mm1ResPrePaddingAlignKvs32MultiCallVF<ElementInput>(
                        sUBTensor[spBuffIdx], static_cast<uint16_t>(actKvsTile),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32));

                } else {
                    // qs=64
                    Mxfp4VF::Mm1ResPrePaddingAlignKvs32MultiQs64CallVF<ElementInput>(
                        sUBTensor[spBuffIdx], static_cast<uint16_t>(actKvsTile),
                        static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32));
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            // softmax
            if (tileInfo.kvsActBaseTileAlign32 == MXFP4::DATA_BLOCK_BYTE) {
                if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                    if (tileInfo.validRowsY1 == 0) {
                        // softmax padding 32 全0: 专用模板(LGM_GROUPS=2 对齐主 VF 的 groupMax padding 到 64)
                        Mxfp4VF::SoftmaxAllInvalidCallVF<ElementInput, true, 1, 1, 2>(
                            pUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx]);
                    } else {
                        // softmax padding 32, qs=128
                        // [DBG] mask 上移到 softmax VF 前的独立 VF: 行 [validRowsY1, min((ceil(Y1/32)+1)*32,32)) 置 MIN_VALUE, 确保对 VF 预计算可见
                        Mxfp4VF::MaskInvalidRowsToMinValueCallVF<ElementInput, false>(
                            sUBTensor[spBuffIdx], 0, tileInfo.validRowsY1, MXFP4::DATA_BLOCK_BYTE);
                        // MIN_VALUE 填充(独立 VF MaskInvalidRowsToMinValueCallVF) 与 softmax VF 的
                        // 预计算(LoadAlign 读 s) 之间是同一 V 流水的 RAW, VF 内无法插屏障,
                        // 必须在此显式 PipeBarrier<PIPE_V>() 后预计算才读得到 MIN_VALUE。
                        // 实测: 无同步时预计算读到 MM1 真实数据 -> group_gmax/global_max 被污染
                        //       -> pscale 饱和 255 -> 输出 NaN。
                        AscendC::PipeBarrier<PIPE_V>();
                        DbgDumpSFirstCol("qs128kvs32", tileInfo.validRowsY1, 0, MXFP4::DATA_BLOCK_BYTE, 0,
                                         spBuffIdx, (uint32_t)tileInfo.curKvsTileLoopIdx);
                        Mxfp4VF::SoftmaxWithGroupMaxAlignQs128Kvs32CallVF<MX_QUANT_MODE, true, ElementInput,
                                                                          ElementDisguiseP>(
                            pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                            localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                            tileInfo.validRowsY1);
                    }
                } else {
                    if (tileInfo.validRowsY1 == 0) {
                        // softmax padding 32 全0: 专用模板(LGM_GROUPS=2 对齐主 VF 的 groupMax padding 到 64)
                        Mxfp4VF::SoftmaxAllInvalidCallVF<ElementInput, false, 1, 1, 2>(
                            pUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx]);
                    } else {
                        // softmax padding 32, qs=64
                        // [DBG] mask 上移到 softmax VF 前的独立 VF: 行 [validRowsY1,32) 置 MIN_VALUE, 确保对 VF 预计算可见 (qs=64: 每行仅 64 half)
                        Mxfp4VF::MaskInvalidRowsToMinValueCallVF<ElementInput, true>(
                            sUBTensor[spBuffIdx], 0, tileInfo.validRowsY1, MXFP4::DATA_BLOCK_BYTE);
                        AscendC::PipeBarrier<PIPE_V>();
                        DbgDumpSFirstCol("qs64kvs32", tileInfo.validRowsY1, 0, MXFP4::DATA_BLOCK_BYTE, 0,
                                         spBuffIdx, (uint32_t)tileInfo.curKvsTileLoopIdx);
                        Mxfp4VF::SoftmaxWithGroupMaxAlignQs64Kvs32CallVF<MX_QUANT_MODE, true, ElementInput,
                                                                         ElementDisguiseP>(
                            pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                            localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                            tileInfo.validRowsY1);
                    }
                }
            } else {
                // ===== softmax padding 32 multi: 按 chunk 分流（与 kvs256 分支同构）=====
                //   Y1 == 0             -> chunk0 全无效: allInvalid（不读 s、不碰 global_max）
                //   否则                -> chunk0: -inf mask + PipeBarrier<PIPE_V>() + softmax
                //   kvs>128 && Y2 == 0  -> chunk1 全无效: allInvalid
                //   kvs>128             -> chunk1: -inf mask + PipeBarrier<PIPE_V>() + softmax
                // 【为何必须按 chunk 分发】chunk VF 内"本大块第一个分组最大值"的预计算**无条件执行**,
                //   前提是 validGroups>=1。整 chunk 无效（kvs>128 且 Y2==0）若仍进入 VF, 会用
                //   preg_all_16bit 读 s 的组 0（MM1 真实数据）, 污染 group_gmax/global_max。
                const uint16_t rows0 = static_cast<uint16_t>(actKvsTile < 128 ? actKvsTile : 128);
                const uint16_t rows1 = static_cast<uint16_t>(actKvsTile > 128 ? actKvsTile - 128 : 0);
                const uint32_t dbgLoop = (uint32_t)tileInfo.curKvsTileLoopIdx;
                if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                    // ---------- chunk0: 行 [0, rows0) ----------
                    if (tileInfo.validRowsY1 == 0) {
                        Mxfp4VF::SoftmaxAllInvalidCallVF<ElementInput, true, 4, 1, 0, 0>(
                            pUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx]);
                    } else {
                        Mxfp4VF::MaskInvalidRowsToMinValueCallVF<ElementInput, false>(
                            sUBTensor[spBuffIdx], 0, tileInfo.validRowsY1, rows0);
                        AscendC::PipeBarrier<PIPE_V>();
                        DbgDumpSFirstCol("qs128kvs32multi", tileInfo.validRowsY1, 0, rows0, 0, spBuffIdx, dbgLoop);
                        Mxfp4VF::SoftmaxWithGroupMaxAlignQs128Kvs32MultiChunkCallVF<MX_QUANT_MODE, true, ElementInput,
                                                                                   ElementDisguiseP,
                                                                                   MXFP4::QS_BASE_SIZE>(
                            pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                            localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                            0, tileInfo.validRowsY1, rows0);
                    }
                    // ---------- chunk1: 行 [128, actKvsTile) ----------
                    if (actKvsTile > 128 && tileInfo.validRowsY2 == 0) {
                        Mxfp4VF::SoftmaxAllInvalidCallVF<ElementInput, true, 4, 1, 0, 1>(
                            pUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx]);
                    } else if (actKvsTile > 128) {
                        Mxfp4VF::MaskInvalidRowsToMinValueCallVF<ElementInput, false>(
                            sUBTensor[spBuffIdx], 128, tileInfo.validRowsY2, rows1);
                        AscendC::PipeBarrier<PIPE_V>();
                        DbgDumpSFirstCol("qs128kvs32multi", tileInfo.validRowsY2, 0, rows1, 128, spBuffIdx, dbgLoop);
                        Mxfp4VF::SoftmaxWithGroupMaxAlignQs128Kvs32MultiChunkCallVF<MX_QUANT_MODE, true, ElementInput,
                                                                                   ElementDisguiseP,
                                                                                   MXFP4::QS_BASE_SIZE>(
                            pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                            localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                            1, tileInfo.validRowsY2, rows1);
                    }
                } else {
                    // ---------- qs=64（每行仅前 64 half 有效, HAS_HIGH_OFF=false）----------
                    if (tileInfo.validRowsY1 == 0) {
                        Mxfp4VF::SoftmaxAllInvalidCallVF<ElementInput, false, 4, 1, 0, 0>(
                            pUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx]);
                    } else {
                        Mxfp4VF::MaskInvalidRowsToMinValueCallVF<ElementInput, true>(
                            sUBTensor[spBuffIdx], 0, tileInfo.validRowsY1, rows0);
                        AscendC::PipeBarrier<PIPE_V>();
                        DbgDumpSFirstCol("qs64kvs32multi", tileInfo.validRowsY1, 0, rows0, 0, spBuffIdx, dbgLoop);
                        Mxfp4VF::SoftmaxWithGroupMaxAlignQs64Kvs32MultiChunkCallVF<MX_QUANT_MODE, true, ElementInput,
                                                                                  ElementDisguiseP,
                                                                                  MXFP4::QS_BASE_SIZE>(
                            pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                            localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                            0, tileInfo.validRowsY1, rows0);
                    }
                    if (actKvsTile > 128 && tileInfo.validRowsY2 == 0) {
                        Mxfp4VF::SoftmaxAllInvalidCallVF<ElementInput, false, 4, 1, 0, 1>(
                            pUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx]);
                    } else if (actKvsTile > 128) {
                        Mxfp4VF::MaskInvalidRowsToMinValueCallVF<ElementInput, true>(
                            sUBTensor[spBuffIdx], 128, tileInfo.validRowsY2, rows1);
                        AscendC::PipeBarrier<PIPE_V>();
                        DbgDumpSFirstCol("qs64kvs32multi", tileInfo.validRowsY2, 0, rows1, 128, spBuffIdx, dbgLoop);
                        Mxfp4VF::SoftmaxWithGroupMaxAlignQs64Kvs32MultiChunkCallVF<MX_QUANT_MODE, true, ElementInput,
                                                                                  ElementDisguiseP,
                                                                                  MXFP4::QS_BASE_SIZE>(
                            pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                            localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                            1, tileInfo.validRowsY2, rows1);
                    }
                }
            }
        }

#if 0  // DBG_DUMP_TMP
        // ===== [DBG] softmax 出口: 打印本 tile 的有效行数与 softmax 后 global_max / local_group_max =====
        AscendDbgDumpI(l1PTensorTla, tileInfo, actKvsTile, groupMaxIdx, spBuffIdx, (uint32_t)taskInfo.taskTileNum);
#endif

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);

        // DataCopy UB -> L1 搬运
        auto ubPLayoutTla =
            tla::MakeLayout<ElementDisguiseP, LayoutPUB>(MXFP4::KVS_BASE_SIZE / 4, MXFP4::QS_BASE_SIZE * 4);
        auto ubPTensorTla = tla::MakeTensor(pUBTensor[0], ubPLayoutTla, Arch::PositionUB{});

        CopyPUBToPL1(l1PTensorTla, ubPTensorTla, tileInfo, taskInfo, spBuffIdx);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyPUBToPL1(TensorDst const &l1PTensorTla, TensorSrc const &ubPTensorTla,
                                        TileInfo const &tileInfo, TaskInfo const &taskInfo, uint32_t spBuffIdx)
    {
        uint32_t pUBTlaMSize = tla::get<0>(ubPTensorTla.shape());
        uint32_t pUBTlaNSize = tla::get<1>(ubPTensorTla.shape());

        if (tileInfo.kvsActBaseTileAlign32 == MXFP4::KVS_BASE_SIZE) {
            // qs=128, kvs=256 整块搬运
            AscendC::DataCopyParams repeatParams;
            if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                repeatParams.blockCount = static_cast<uint16_t>(pUBTlaMSize); // UB的物理P一整行256B, 包含P的逻辑视图4行
            } else {
                repeatParams.blockCount =
                    static_cast<uint16_t>(pUBTlaMSize / 2); // UB的物理P一整行256B, 包含P的逻辑视图4行
            }
            repeatParams.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams.dstStride = 0;

            auto pUBTensorTlaTile =
                GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP4::UB_P_INNER_BUF_ELEMENT_OFFSET),
                        tla::MakeShape(pUBTlaMSize, pUBTlaNSize / 2));
            auto srcOffset = pUBTensorTlaTile.layout()(pUBTensorTlaTile.coord());

            AscendC::DataCopy(l1PTensorTla.data(), pUBTensorTlaTile.data()[srcOffset], repeatParams);

        } else {
            // 消去高位 低位中间的间隙搬运
            // 低位搬运
            auto pUBTensorTlaTile =
                GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP4::UB_P_INNER_BUF_ELEMENT_OFFSET),
                        tla::MakeShape(pUBTlaMSize, pUBTlaNSize / 2));
            auto srcOffset = pUBTensorTlaTile.layout()(pUBTensorTlaTile.coord());

            AscendC::DataCopyParams repeatParams1;
            repeatParams1.blockCount = static_cast<uint16_t>(
                tileInfo.kvsActBaseTileAlign32 /
                (pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE)); // UB的物理P一整行256B, 包含P的逻辑视图低位(高位)地址8行
            repeatParams1.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams1.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
            repeatParams1.dstStride = 0;

            auto pUBTensorTlaTile1 =
                tla::GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP4::UB_P_INNER_BUF_ELEMENT_OFFSET),
                             tla::MakeShape(pUBTlaMSize / 2, pUBTlaNSize / 2));
            auto srcOffset1 = pUBTensorTlaTile1.layout()(pUBTensorTlaTile1.coord());
            AscendC::DataCopy(l1PTensorTla.data(), ubPTensorTla.data()[srcOffset1], repeatParams1);

            if (taskInfo.qsActBaseTileAlign64 == MXFP4::QS_BASE_SIZE) {
                // 高位搬运
                AscendC::DataCopyParams repeatParams2;
                repeatParams2.blockCount = static_cast<uint16_t>(
                    tileInfo.kvsActBaseTileAlign32 /
                    (pUBTlaNSize / 2 /
                     MXFP4::DATA_BLOCK_BYTE)); // UB的物理P一整行256B, 包含P的逻辑视图高位(低位)地址8行
                repeatParams2.blockLen = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
                repeatParams2.srcStride = static_cast<uint16_t>(pUBTlaNSize / 2 / MXFP4::DATA_BLOCK_BYTE);
                repeatParams2.dstStride = 0;

                uint32_t pL1NHalfSize = tla::get<1, 0>(l1PTensorTla.shape());
                uint32_t xL1P = tileInfo.kvsActBaseTileAlign64;
                uint32_t yL1P = 0;
                if (tileInfo.kvsActBaseTileAlign64 == MXFP4::KVS_BASE_SIZE) {
                    xL1P = 0;
                    yL1P = pL1NHalfSize;
                }
                auto pUBTensorTlaTile2 = tla::GetTile(pUBTensorTlaTile1, tla::MakeCoord(pUBTlaMSize / 2, 0u),
                                                      tla::MakeShape(pUBTlaMSize / 2, pUBTlaNSize / 2)); // 遗留
                auto pl1TensorTlaTile = tla::GetTile(l1PTensorTla, tla::MakeCoord(xL1P, yL1P),
                                                     tla::MakeShape(MXFP4::KVS_BASE_SIZE, pL1NHalfSize));

                auto srcOffset2 = pUBTensorTlaTile2.layout()(pUBTensorTlaTile2.coord());
                auto dstOffset2 = pl1TensorTlaTile.layout()(pl1TensorTlaTile.coord());

                AscendC::DataCopy(pl1TensorTlaTile.data()[dstOffset2], pUBTensorTlaTile2.data()[srcOffset2],
                                  repeatParams2);
            }
        }
    }

    // buffer id 获取
    __aicore__ inline uint32_t GetSPBufferIdx(const uint32_t loop) { return loop / 2 % 2; }

    __aicore__ inline uint32_t GetLocalGroupMaxBufIdx(uint32_t loop)
    {
        return loop / 2 * 2 % MXFP4::UB_LOCAL_GROUP_MAX_CNT;
    }

#if 0  // DBG_DUMP_TMP
    // ===== [DBG] softmax 出口 dump: 本 tile 有效行数 + softmax 后 global_max / local_group_max =====
    // 只打印整数(避免 %f 触发 float->double 提升)。约定: 999999=+inf, -999999=-inf, 888888=NaN
    template <class TensorL1P>
    __aicore__ inline void AscendDbgDumpI(TensorL1P &l1PTensorTla, TileInfo const &tileInfo, uint32_t actKvsTile,
                                          uint32_t groupMaxIdx, uint32_t spBuffIdx, uint32_t taskTileNum)
    {
        (void)l1PTensorTla;
        uint32_t sub = AscendC::GetSubBlockIdx();
        AscendC::PRINTF(
            "[SMDBG] t=%u/%u tm=%u sub=%u sp=%u actKvs=%u Y1=%u Y2=%u | first=%d tgf=%d | gmax=%d | lm=%d,%d,%d,%d,%d,%d,%d,%d\n",
            (uint32_t)tileInfo.curKvsTileLoopIdx, taskTileNum, (uint32_t)tileInfo.tileMaxIdx, sub,
            (uint32_t)spBuffIdx, (uint32_t)actKvsTile, (uint32_t)tileInfo.validRowsY1, (uint32_t)tileInfo.validRowsY2,
            (int32_t)tileInfo.isFirstKvsTile, (int32_t)tileInfo.isTileGoupFirstTile,
            DbgToI((float)localGlobalMaxUBTensor[tileInfo.tileMaxIdx].GetValue(0)),
            DbgToI((float)localGroupMaxUBTensor[groupMaxIdx].GetValue(0)),
            DbgToI((float)localGroupMaxUBTensor[groupMaxIdx].GetValue(128)),
            DbgToI((float)localGroupMaxUBTensor[groupMaxIdx].GetValue(256)),
            DbgToI((float)localGroupMaxUBTensor[groupMaxIdx].GetValue(384)),
            DbgToI((float)localGroupMaxUBTensor[groupMaxIdx].GetValue(512)),
            DbgToI((float)localGroupMaxUBTensor[groupMaxIdx].GetValue(640)),
            DbgToI((float)localGroupMaxUBTensor[groupMaxIdx].GetValue(768)),
            DbgToI((float)localGroupMaxUBTensor[groupMaxIdx].GetValue(896)));
    }

    __aicore__ inline int32_t DbgToI(float fv)
    {
        if (fv != fv) {
            return 888888;  // NaN
        }
        if (fv > 1e30f) {
            return 999999;  // +inf
        }
        if (fv < -1e30f) {
            return -999999;  // -inf
        }
        return (int32_t)fv;
    }

#endif

#if 1  // DBG_DUMP_SBUF
    // ===== [DBG] 填充后 S buffer 每行首值 dump =====
    // 打印 sUBTensor[spBuffIdx] 中 [rowBase, rowBase+rows) 行每行第 0 个元素。
    // 注意: S 物理行宽 = QS_BASE_SIZE*2 = 256 half = 512B (行内两个 128-half ping-pong 槽, 槽 = mm1ResBufId = spBuffIdx)。
    //       故行 r 的 half 偏移 = r*256, 而不是 r*128 (此前 dump 用 r*128 会交替命中槽0/槽1 造成"污染"假象)。
    // 每次调用只有一个组循环 (rows/8 次迭代, 每组 8 个值排成一排便于查看; 单次 PRINTF 参数不超过 16 个)。
    // 值约定: -65504=MIN_VALUE(-inf 填充); 0=0 填充; |v|<1 的分数放大 1000 倍(如 -0.5 -> -500) 便于与 0 区分;
    //         999999=+inf, -999999=-inf, 888888=NaN。只打印整数(避免 %f 触发 float->double 提升)。
    // 单次 dump 输出约 1KB (rows=128 行), 远小于 kernel 1MB 打印缓冲。
    __aicore__ inline int32_t DbgSColVal(float fv)
    {
        if (fv != fv) {
            return 888888;  // NaN
        }
        if (fv > 1e30f) {
            return 999999;  // +inf
        }
        if (fv < -1e30f) {
            return -999999;  // -inf
        }
        if (fv != 0.0f && fv > -1.0f && fv < 1.0f) {
            return (int32_t)(fv * 1000.0f);  // 分数放大 1000 倍, 便于区分 0 填充
        }
        return (int32_t)fv;
    }

    __aicore__ inline void DbgDumpSFirstCol(const __gm__ char *tag, uint32_t effY1, uint32_t effY2, uint32_t rows,
                                            uint32_t rowBase, uint32_t spBuffIdx, uint32_t loop)
    {
        uint32_t sub = AscendC::GetSubBlockIdx();
        AscendC::PRINTF("[SDBG] %s effY=%u/%u rows=%u rb=%u t=%u sub=%u\n", tag, effY1, effY2, rows, rowBase,
                        loop, sub);
        // for (uint32_t g = 0; g < rows / 8; ++g) {
        //     uint32_t r = rowBase + g * 8;
        //     AscendC::PRINTF(
        //         "  r%u: %d %d %d %d %d %d %d %d\n", r,
        //         DbgSColVal((float)sUBTensor[spBuffIdx].GetValue((r + 0) * MXFP4::QS_BASE_SIZE * 2)),
        //         DbgSColVal((float)sUBTensor[spBuffIdx].GetValue((r + 1) * MXFP4::QS_BASE_SIZE * 2)),
        //         DbgSColVal((float)sUBTensor[spBuffIdx].GetValue((r + 2) * MXFP4::QS_BASE_SIZE * 2)),
        //         DbgSColVal((float)sUBTensor[spBuffIdx].GetValue((r + 3) * MXFP4::QS_BASE_SIZE * 2)),
        //         DbgSColVal((float)sUBTensor[spBuffIdx].GetValue((r + 4) * MXFP4::QS_BASE_SIZE * 2)),
        //         DbgSColVal((float)sUBTensor[spBuffIdx].GetValue((r + 5) * MXFP4::QS_BASE_SIZE * 2)),
        //         DbgSColVal((float)sUBTensor[spBuffIdx].GetValue((r + 6) * MXFP4::QS_BASE_SIZE * 2)),
        //         DbgSColVal((float)sUBTensor[spBuffIdx].GetValue((r + 7) * MXFP4::QS_BASE_SIZE * 2)));
        // }
    }
#endif

private:
    // UB tensors（FIXPIPE<->V 及常驻 buffer）
    AscendC::LocalTensor<ElementInput> sUBTensor[MXFP4::UB_S_BUF_CNT];                       // mm1Res(S)
    AscendC::LocalTensor<ElementDisguiseP> pUBTensor[MXFP4::UB_P_BUF_CNT];                   // vec1Res(P)
    AscendC::LocalTensor<ElementMax> localGroupMaxUBTensor[MXFP4::UB_LOCAL_GROUP_MAX_CNT];   // LocalGroupMax
    AscendC::LocalTensor<ElementMax> localGlobalMaxUBTensor[MXFP4::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
    AscendC::LocalTensor<ElementIndex> indexUBTensor;                                        // Index
    ElementInput NEG_LOG2_CX_;
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH35_REG_LOW_PREC_FP16_MXFP4_HPP
