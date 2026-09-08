/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../arch35/kernel_utils.hpp"
#include "../attn_infra/epilogue/block/block_epilogue_arch35_utils.hpp"
#include "../attn_infra/gemm/block/block_mmad_arch35_utils.hpp"

using namespace NpuArch;
using namespace tla;
using namespace MXFP8Kernel;

namespace EpMXFP8 = Epilogue::Block::MXFP8; // UB 常量别名
namespace GmMXFP8 = Gemm::Block::MXFP8;     // L1 常量别名

namespace BsaKernelArch35 {

#ifndef BSA_MXFP8_DUMP
#define BSA_MXFP8_DUMP 0
#endif
// P0b：Wait(V1_C1) 前预取 K。置 0 回到「先 Wait 再 gather」（operator() = PrefetchK+ComputeQK）。
#ifndef BSA_MXFP8_K_PREFETCH
#define BSA_MXFP8_K_PREFETCH 1
#endif
#if BSA_MXFP8_DUMP
#define BSA_MXFP8_DUMP_CORE() (AscendC::GetBlockIdx() == 0)
#endif

template <class EpilogueMask2Idx, class BlockMmadQK, class EpilogueOnlineSoftmax, class BlockMmadPV,
          class EpilogueRescaleO, class EpilogueComputePScale, class EpilogueCopyGlobalMaxUbToL1,
          class BlockMmadCopyGlobalMaxL1ToUB, Format qFormat, Format kvFormat, bool transposedMm1 = false>
class BsaMXFP8FullQuantKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementDisguiseP = typename EpilogueOnlineSoftmax::ElementDisguiseP;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename EpilogueRescaleO::ElementO;
    using ElementLse = typename EpilogueRescaleO::ElementLse;
    using ElementSparseMask = typename EpilogueMask2Idx::ElementSparseMask;
    using ElementSparseIdx = typename EpilogueMask2Idx::ElementSparseIdx;
    using ElementSparseCount = typename EpilogueMask2Idx::ElementSparseCount;
    using ElementGroupMax = typename EpilogueComputePScale::ElementGroupMax;
    using ElementPScale = typename EpilogueComputePScale::ElementPScale;
    using ElementLocalGlobalMax = typename EpilogueCopyGlobalMaxUbToL1::ElementLocalGlobalMax;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::RowMajor;
    using LayoutS = layout::RowMajor;
    using LayoutPL1 = typename EpilogueOnlineSoftmax::LayoutPL1;
    using LayoutV = layout::ColumnMajor;
    using LayoutO = layout::RowMajor;
    using LayoutLse = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutLocalGlobalMax = typename EpilogueCopyGlobalMaxUbToL1::LayoutLocalGlobalMax;
    using LayoutPScale = typename EpilogueComputePScale::LayoutPScale;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;

    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;

    // 核间同步
    static constexpr uint8_t SYNC_MODE_2 = 2;
    static constexpr uint8_t SYNC_MODE_4 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_V1_C1[2] = {9, 10};
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_L1_UB_BACK = 8;
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF1 = 7;
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 = 6;
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF1 = 5;
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 = 4;
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2_BUF1 = 3;
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2_BUF0 = 2;
    static constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 1;
    static constexpr uint16_t CROSS_CORE_SYNC_V2_C2 = 0;

    static constexpr uint32_t PV_RE_DELAY_N = 20; // G=16 → Dpv>=19，取 20
    static constexpr uint32_t DELAY_PSCALE_N = 3;

    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PV_RE_DELAY_N + 1;

    // Methods
    __aicore__ inline BsaMXFP8FullQuantKernelArch35() {}

    __aicore__ inline void operator()(BsaFullQuantKernelParamsArch35 const &params)
    {
        __gm__ BlockSparseAttentionTilingData *bsaTilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionTilingData *>(params.tiling);
        FetchBaseShapeInfo(bsaTilingData);
        // CalcOnChipBufTileInfo(bsaTilingData);
        // KV(V 数据) 与 V-scale 的 stride 预算（PV 稀疏 gather 用）
        // V 数据 fp4：sizeof(fp8_e4m3fn_t)=1，元素偏移即字节偏移，行 stride = embed_(D 个 fp4)
        // V-scale E8M0：GM [B,N2,S2//64,D,2]，每 64-S2 行 group 占 D*2 字节
        // O / Lse stride & batch offsets（同 BF16，在 while 循环中创建 task 时传入）
        InitLayoutStride();
        // global buffers
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.query);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.key);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.value);
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualSeqLengths);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualSeqLengthsKv);
        AscendC::GlobalTensor<uint8_t> gBlockSparseMask;
        gBlockSparseMask.SetGlobalBuffer((__gm__ uint8_t *)params.blockSparseMask);
        AscendC::GlobalTensor<uint8_t> gQDequantScale;
        gQDequantScale.SetGlobalBuffer((__gm__ uint8_t *)params.qDequantScale);
        AscendC::GlobalTensor<uint8_t> gKDequantScale;
        gKDequantScale.SetGlobalBuffer((__gm__ uint8_t *)params.kDequantScale);
        AscendC::GlobalTensor<uint8_t> gVDequantScale;
        gVDequantScale.SetGlobalBuffer((__gm__ uint8_t *)params.vDequantScale);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.attentionOut);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);
        AscendC::GlobalTensor<ElementSparseIdx> gSparseIdx;
        gSparseIdx.SetGlobalBuffer((__gm__ ElementSparseIdx *)params.workSpace);
        AscendC::GlobalTensor<ElementSparseCount> gSparseCount;
        gSparseCount.SetGlobalBuffer((__gm__ ElementSparseCount *)(params.workSpace + sparseIdxSize_));

        // cross core data move dst buffers
        AscendC::LocalTensor<ElementS> ubSTensor[EpMXFP8::UB_S_BUF_CNT];
        AscendC::LocalTensor<ElementDisguiseP> l1PTensor[GmMXFP8::L1_P_BUF_CNT];
        AscendC::LocalTensor<ElementPScale> l1PScaleTensor[GmMXFP8::L1_P_SCALE_BUF_CNT];
        AscendC::LocalTensor<ElementS> l1LocalGlobalMaxTensor[GmMXFP8::L1_LOCAL_GLOBAL_MAX_BUF_CNT];
        AscendC::LocalTensor<ElementS> ubPeerGlobalMaxTensor[EpMXFP8::UB_PEER_GLOBAL_MAX_CNT];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor;

        InitCrossCoreDstBuf(l1PTensor, l1PScaleTensor, l1LocalGlobalMaxTensor, ubSTensor, ubPeerGlobalMaxTensor,
                            ubOTmpTensor);
        // core idx
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        // set reverse sync flags
        InitSyncFlags();
        // mask preprocess
        if ASCEND_IS_AIV {
            EpilogueMask2Idx epilogueMask2Idx(resource);
            uint32_t totalRowNumBlockMask = batch_ * qHeads_ * xBlockNumAligned_;
            epilogueMask2Idx(gBlockSparseMask, gSparseIdx, gSparseCount, totalRowNumBlockMask, yBlockNumAligned_,
                             avgRowPerSubCore_, preActiveSubCoreNum_);
        }
        AscendC::SyncAll<false>();

        // tla define
        auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(EpMXFP8::KVS_BASE_SIZE, EpMXFP8::QS_BASE_SIZE * 2);
        auto ubSTensorTla = tla::MakeTensor(ubSTensor[0], ubSLayoutTla, Arch::PositionUB{});

        auto l1PScaleLayoutTla =
            tla::MakeLayout<ElementPScale, LayoutPScale>(GmMXFP8::L1_P_SCALE_BUF_CNT, GmMXFP8::L1_P_SCALE_BUF_SIZE);
        auto l1PScaleTensorTla = tla::MakeTensor(l1PScaleTensor[0], l1PScaleLayoutTla, Arch::PositionL1{});

#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        // [QK/PV 共享] kernel 层单一 kvBufId，传引用进两 block ctor（QK 先构造预置 4 路 KV event；
        //   PV 后构造、其 ctor 内 InitL0BufferForReduceSum 把 kvBufId 0→1，与 QFA cube 同构）
        uint32_t kvBufId = 0;
        uint64_t softmaxScale = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&scaleValue_));
        BlockMmadQK blockMmadQK(resource, kvBufId, softmaxScale);
        BlockMmadPV blockMmadPV(resource, kvBufId);
        BlockMmadCopyGlobalMaxL1ToUB blockGMaxL1ToUb(resource);
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        // host 侧已预算: log2Cx_ = -dstTypeMax, log2CxCeil_ = ceil(dstTypeMax)
        half NEG_LOG2_CX = static_cast<half>(-log2Cx_);
        half LOG2_CX_CEIL = static_cast<half>(log2CxCeil_);
        // if(coreIdx == 0) {
        //             AscendC::printf("NEG_LOG2_CX = %f, LOG2_CX_CEIL = %f\n",
        //         static_cast<float>(NEG_LOG2_CX),
        //         static_cast<float>(LOG2_CX_CEIL));
        // }

        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, NEG_LOG2_CX);
        EpilogueCopyGlobalMaxUbToL1 epilogueCopyGlobalMaxUbToL1(resource);
        EpilogueComputePScale epilogueComputePScale(resource, LOG2_CX_CEIL);
        EpilogueRescaleO epilogueRescaleO(resource);
#endif

        // runMainLoop
        uint64_t coreTileLoop = 0;

        uint64_t curTaskTileIdx = 0;
        uint64_t delay12TaskTileIdx = 0;
        uint64_t delay3TaskTileIdx = 0;

        uint32_t coreTaskId = 0;
        uint32_t delay12CoreTaskId = 0;
        uint32_t delay3CoreTaskId = 0;

        uint64_t preTaskTileSum = 0;
        uint64_t delay12preTaskTileSum = 0;
        uint64_t delay3preTaskTileSum = 0;

        uint64_t coreTaskNum = coreTaskNum_;
        coreTaskNum = (coreIdx < totalTaskNum_ % coreNum) ? (coreTaskNum + 1) : coreTaskNum;

        TaskInfo taskInfos[PRELOAD_TASK_CACHE_SIZE] = {};
        TileInfo tileInfos[PRELOAD_TASK_CACHE_SIZE] = {};

        // CreateTaskInfo 跨调用 batch 扫描状态：curTotalTaskNum 从首 batch task 数起步，
        // 后续调用在 while 循环内基于该状态续算，不再从 batch 0 重扫
        BatchOffsetInfo batchOffsetInfo;
        batchOffsetInfo.curTotalTaskNum = firstBatchTaskNum_;
        // [BSND/TND stride 修复] 头内 per-token/per-group gather stride 因子: BNSD=1(头内连续);
        //   BSND/TND=kvHeads_/qHeads_(头内 token 隔 N*D)。原 gather layout 烧死 embed_=D(仅 BNSD 对),
        //   BSND/TND 头内越界 507015(开发者 :236 TODO)。修法=「乘 extent」hack:
        //   ColumnMajor(rows)/RowMajor(cols)/MxScale(rows) 乘此因子, col/row/group-stride 自动变对,
        //   extent 超估对 GetTile(按真实 extent 取子块)无害。
        const uint32_t kvHeadMul = (kvFormat == Format::BNSD) ? 1u : kvHeads_;
        const uint32_t qHeadMul = (qFormat == Format::BNSD) ? 1u : qHeads_;

        while (coreTaskId < coreTaskNum || delay12CoreTaskId < coreTaskNum || delay3CoreTaskId < coreTaskNum) {
            // AscendC::PRINTF("loop:%llu \n", coreTileLoop);
            curTaskTileIdx = coreTileLoop - preTaskTileSum;
            delay12TaskTileIdx = coreTileLoop - PV_RE_DELAY_N - delay12preTaskTileSum;
            delay3TaskTileIdx = coreTileLoop - DELAY_PSCALE_N - delay3preTaskTileSum;

            TaskInfo &curTaskInfo = taskInfos[coreTaskId % PRELOAD_TASK_CACHE_SIZE];
            TaskInfo &delay12TaskInfo = taskInfos[delay12CoreTaskId % PRELOAD_TASK_CACHE_SIZE];
            TaskInfo &delay3TaskInfo = taskInfos[delay3CoreTaskId % PRELOAD_TASK_CACHE_SIZE];

            TileInfo &curTileInfo = tileInfos[coreTileLoop % PRELOAD_TASK_CACHE_SIZE];
            TileInfo &delay12TileInfo = tileInfos[(coreTileLoop - PV_RE_DELAY_N) % PRELOAD_TASK_CACHE_SIZE];
            TileInfo &delay3TileInfo = tileInfos[(coreTileLoop - DELAY_PSCALE_N) % PRELOAD_TASK_CACHE_SIZE];

            // 创建qs任务
            if (curTaskTileIdx == 0 && coreTaskId < coreTaskNum) {
                CreateTaskInfo(curTaskInfo, coreTaskId, coreIdx, coreNum, gActualQseqlen, gActualKvseqlen, gSparseCount,
                               gSparseIdx, batchOffsetInfo);
            }

            // main process
            if (coreTileLoop >= PV_RE_DELAY_N && delay12CoreTaskId < coreTaskNum) {
                if ASCEND_IS_AIC {
                    if (delay12TileInfo.isTileGoupFirstTile) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2_BUF0 +
                                                                           delay12TileInfo.pscaleNum);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2_BUF0 +
                                                                           delay12TileInfo.pscaleNum + 16);
                    }
                    if (delay12TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
                    }
                    // ComputeMm2(delay12TileInfo);

                    // [BSND/TND stride 修复] V GM col-stride(头内 token-stride): BNSD=embed_(D, 头内连续);
                    //   BSND/TND=kvHeadMul*embed_(=N*D, 头内 token 隔 N*D)。原 :236 todo 已修——
                    //   乘 kvHeadMul(BNSD=1) 使 col-stride 变为 N*D; extent 超估对 GetTile(读 embed_) 无害。
                    //   物理: V BSND row-major [S,N,D] 头内 token-stride=N*D; BNSD [N,S,D] 头内=D。
                    auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(kvHeadMul * embed_, kvBaseTile_);
                    auto gmVTensorTla =
                        tla::MakeTensor(gV[delay12TaskInfo.gmOffsetV], gmVLayoutTla, Arch::PositionGM{});

                    // [catlass化] UB Oᵀ 实际存储布局：rows=embed_+1（D 行 Oᵀ + 1 行 rowsum）、
                    //   cols=s1Align128=CeilDiv(qsActBaseTile,128)*128（对齐 128，供 fixpipe nSize）、
                    //   row-stride=64（固定 rowsum-fractal 宽，供 fixpipe dstStride）。
                    //   非标 RowMajor（stride≠cols），由 block 的 FixpipeMm2 经 CopyL0CToUBTla SPLIT_N 消费；
                    //   isRowMajor 只查 stride<1>==1 → 命中。block 直传此 tla（不再 .data() 重包）。
                    auto ubOTmpLayoutTla =
                        tla::MakeLayout(tla::MakeShape(GmMXFP8::ROW_SUM_NUM + 1, delay12TaskInfo.qsActBaseTileAlign64),
                                        tla::MakeStride(static_cast<int64_t>(64), tla::Int<1>{}));
                    auto ubOTmpTensorTla = tla::MakeTensor(ubOTmpTensor, ubOTmpLayoutTla, Arch::PositionUB{});

                    GemmCoord actualBlockShapePV{delay12TaskInfo.qsActBaseTile, embed_, delay12TileInfo.kvsActBaseTile};
                    // [改造] 传 base GM 张量(gVDequantScale/gSparseIdx) + delay12TaskInfo 整体，
                    //   block 内按 gmOffsetVScale/gmOffsetSparseIdx 取子缓冲、解析
                    //   kvSeqlen/yBlockNumAval/yBlockNumRsvd。 调用处不再出现 delay12TaskInfo.<field> 分字段传参。
                    blockMmadPV(gmVTensorTla, gVDequantScale, gSparseIdx, ubOTmpTensorTla, actualBlockShapePV,
                                delay12TileInfo, // gatheredKvSTileIdx
                                delay12TaskInfo, kvBaseTile_, blockShapeY_, kvHeadMul);

                    if (delay12TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + 16);
                    }
                } else {
                    if (delay12TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
#if BSA_MXFP8_DUMP
                        if (delay12CoreTaskId == 0 && delay12TileInfo.isLastKvsTile) {
                            AscendC::PipeBarrier<PIPE_ALL>();
                            AscendC::DumpTensor(ubOTmpTensor, 220, 32);
                            AscendC::printf(
                                "[MXFP8-DUMP] OTmp desc=220 blk=%u sub=%u otmp0=%f otmp50=%f otmp57=%f otmp63=%f "
                                "otmpRS50=%f otmpRS57=%f lastKvs=%d tile=%u kvsAct=%u qsAct=%u\n",
                                AscendC::GetBlockIdx(), AscendC::GetSubBlockIdx(),
                                static_cast<float>(ubOTmpTensor.GetValue(0)),
                                static_cast<float>(ubOTmpTensor.GetValue(50)),
                                static_cast<float>(ubOTmpTensor.GetValue(57)),
                                static_cast<float>(ubOTmpTensor.GetValue(63)),
                                static_cast<float>(ubOTmpTensor.GetValue(128 * 64 + 50)),
                                static_cast<float>(ubOTmpTensor.GetValue(128 * 64 + 57)),
                                static_cast<int>(delay12TileInfo.isLastKvsTile),
                                static_cast<uint32_t>(delay12TaskTileIdx), delay12TileInfo.kvsActBaseTile,
                                delay12TaskInfo.qsActBaseTile);
                        }
#endif
                        ComputeVec2(delay12TileInfo);
                        // [BSND root fix] O GM layout: cols=embed_(D, 每 token 元素数) +
                        // row-stride=oShapeCol(=strideOS_, token-stride)。
                        //   原 RowMajor(qBaseTile, oShapeCol) 把 cols-extent 与 row-stride 都设成
                        //   oShapeCol(=strideOS_): BNSD 二者都=D 过; BSND oShapeCol=strideOS_=N*D -> cols 放大 ->
                        //   epilogue get<1>(shape)=N*D
                        //   -> copyUbToGmO 每 token 写 N*D 元素(128 真+128 垃圾) -> 输出错乱 1.147%。
                        //   hand-build 分离: cols=embed_(D, extent 正确)、row-stride=oShapeCol(strideOS_, token-stride
                        //   正确); isRowMajor(stride<1>=1) 仍成立 -> dispatch 不变; epilogue get<1>(shape)
                        //   自然=embed_=D, 无需 epilogue 侧特判。
                        auto gmOLayoutTla = tla::MakeLayout(tla::MakeShape(qBaseTile_, embed_),
                                                            tla::MakeStride(delay12TaskInfo.oShapeCol, tla::Int<1>{}));
                        auto gmOTensorTla =
                            tla::MakeTensor(gO[delay12TaskInfo.gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
                        GemmCoord actualBlockShapeVPT{embed_, delay12TaskInfo.qsActBaseTile,
                                                      delay12TileInfo.kvsActBaseTile};
                        epilogueRescaleO(gmOTensorTla, actualBlockShapeVPT, delay12TaskInfo, delay12TileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
                    }
                }
            }
            if (coreTileLoop >= DELAY_PSCALE_N && delay3CoreTaskId < coreTaskNum) {
                // delay3TileInfo = tileInfos[(coreTileLoop - DELAY_PSCALE_N) % PRELOAD_TASK_CACHE_SIZE];
                if (delay3TileInfo.isUpdatePScale) {
                    if ASCEND_IS_AIC {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 +
                                                                           delay3TileInfo.tileMaxIdx / 2);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 +
                                                                           delay3TileInfo.tileMaxIdx / 2 + 16);
                        // mxfp4 这里用 KVS_BASE=256，实际是 2*QS（两个 AIV 的 gmax 各 128）。
                        auto l1GMaxLayoutTla =
                            tla::MakeLayout<ElementLocalGlobalMax, LayoutLocalGlobalMax>(1, EpMXFP8::QS_BASE_SIZE * 2);
                        auto l1GMaxTensorTla = tla::MakeTensor(ubPeerGlobalMaxTensor[delay3TileInfo.tileMaxIdx],
                                                               l1GMaxLayoutTla, Arch::PositionUB{});
                        CopyGMaxL1ToUb(delay3TileInfo);
                        blockGMaxL1ToUb(l1GMaxTensorTla, delay3TileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 +
                                                                          delay3TileInfo.tileMaxIdx / 2);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 +
                                                                          delay3TileInfo.tileMaxIdx / 2 + 16);
                        if (delay3TileInfo.tileMaxIdx == 3 ||
                            (delay3CoreTaskId == coreTaskNum - 1 &&
                             delay3TileInfo.isLastKvsTile)) { // tilemax最后一块 或者 单核上的最后一块
                            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);
                            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK + 16);
                        }
                    } else {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_GMAX_L1_TO_UB_BUF0 +
                                                                        delay3TileInfo.tileMaxIdx / 2);
                        epilogueComputePScale(l1PScaleTensorTla, delay3TileInfo, delay3TaskInfo);
                        // AscendC::DumpTensor(ubPeerGlobalMaxTensor[delay3TileInfo.tileMaxIdx], 168, 128);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_PSCALE_C2_BUF0 +
                                                                          delay3TileInfo.pscaleNum);
                    }
                }
            }
            if (coreTaskId < coreTaskNum) {
                CreateTileInfo(curTileInfo, coreTileLoop, static_cast<uint32_t>(curTaskTileIdx),
                               curTaskInfo); // 创建任务

                uint32_t mm1ResBufId = (curTileInfo.loop / 2) % 2;
                uint32_t subBlockIdx = curTileInfo.loop % 2;

                if ASCEND_IS_AIC {
                    auto gQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(
                        curTaskInfo.qsActBaseTile, // rows = S1 实际 tile
                        qHeadMul * embed_);        // cols → row-stride（BNSD 时 qHeadMul=1，即 embed_）
                    auto gQTensorTla = tla::MakeTensor(gQ[curTaskInfo.gmOffsetQ], gQLayoutTla, Arch::PositionGM{});
                    auto gKLayout = tla::MakeLayout<ElementK, LayoutK>(
                        curTaskInfo.kvSeqlen,
                        kvHeadMul * embed_); // [BSND stride 修复] row-stride=kvHeadMul*embed(BNSD=embed)
                    auto gKTensorTla = tla::MakeTensor(gK[curTaskInfo.gmOffsetK], gKLayout, Arch::PositionGM{});
                    auto curUBSTensorTla = GetTile(ubSTensorTla, MakeCoord(0, EpMXFP8::QS_BASE_SIZE * mm1ResBufId),
                                                   MakeShape(EpMXFP8::KVS_BASE_SIZE, EpMXFP8::QS_BASE_SIZE));

#ifdef __DAV_CUBE__
#if BSA_MXFP8_K_PREFETCH
                    // P0b：K 的 MTE2 在 Wait(V1_C1) 前发出；Wait 留在 kernel，不塞进 QK block。
                    blockMmadQK.PrefetchK(gQTensorTla, gKTensorTla, gQDequantScale[curTaskInfo.gmOffsetQScale],
                                          gKDequantScale[curTaskInfo.gmOffsetKScale],
                                          gSparseIdx[curTaskInfo.gmOffsetSparseIdx], curTaskInfo, curTileInfo, embed_,
                                          kvBaseTile_, blockShapeY_, kvHeadMul, qHeadMul);
#endif
#endif
                    AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] +
                                                                      subBlockIdx * 16);
                    ComputeMm1(curTileInfo);
#ifdef __DAV_CUBE__
#if BSA_MXFP8_K_PREFETCH
                    blockMmadQK.ComputeQK(curUBSTensorTla, curTaskInfo, curTileInfo, embed_, subBlockIdx);
#else
                    blockMmadQK(gQTensorTla, gKTensorTla, gQDequantScale[curTaskInfo.gmOffsetQScale],
                                gKDequantScale[curTaskInfo.gmOffsetKScale], gSparseIdx[curTaskInfo.gmOffsetSparseIdx],
                                curUBSTensorTla, curTaskInfo, curTileInfo, embed_, kvBaseTile_, blockShapeY_,
                                subBlockIdx, kvHeadMul, qHeadMul);
#endif
#endif
                    AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] +
                                                                     subBlockIdx * 16);
                } else {
#ifdef __DAV_VEC__
                    if (subBlockIdx == AscendC::GetSubBlockIdx()) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                        uint32_t l1PBufId = curTileInfo.loop % GmMXFP8::L1_P_BUF_CNT;
                        auto l1PLayoutTla =
                            tla::MakeLayout<ElementDisguiseP, LayoutPL1>(EpMXFP8::KVS_BASE_SIZE, EpMXFP8::QS_BASE_SIZE);
                        auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId], l1PLayoutTla, Arch::PositionL1{});
                        GemmCoord actualBlockShapeQK{curTileInfo.kvsActBaseTile, curTaskInfo.qsActBaseTile, embed_};
#if BSA_MXFP8_DUMP
                        if (coreTaskId == 0 && curTaskTileIdx < 2) {
                            AscendC::PipeBarrier<PIPE_ALL>();
                            AscendC::DumpTensor(ubSTensor[0], static_cast<uint32_t>(200 + curTaskTileIdx), 32);
                            AscendC::printf("[MXFP8-DUMP] S desc=%u blk=%u sub=%u s0=%f s57=%f s58=%f s63=%f s64=%f "
                                            "s114=%f s127=%f s256_57=%f "
                                            "loop=%u tile=%u kvsAct=%u qsAct=%u taskTiles=%u\n",
                                            static_cast<uint32_t>(200 + curTaskTileIdx), AscendC::GetBlockIdx(),
                                            subBlockIdx, static_cast<float>(ubSTensor[0].GetValue(0)),
                                            static_cast<float>(ubSTensor[0].GetValue(57)),
                                            static_cast<float>(ubSTensor[0].GetValue(58)),
                                            static_cast<float>(ubSTensor[0].GetValue(63)),
                                            static_cast<float>(ubSTensor[0].GetValue(64)),
                                            static_cast<float>(ubSTensor[0].GetValue(114)),
                                            static_cast<float>(ubSTensor[0].GetValue(127)),
                                            static_cast<float>(ubSTensor[0].GetValue(256 + 57)),
                                            static_cast<uint32_t>(coreTileLoop), static_cast<uint32_t>(curTaskTileIdx),
                                            curTileInfo.kvsActBaseTile, curTaskInfo.qsActBaseTile,
                                            curTaskInfo.taskTileNum);
                        }
#endif
                        epilogueOnlineSoftmax(l1PTensorTla, actualBlockShapeQK, curTileInfo, curTaskInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                    }
                    if (curTileInfo.isUpdatePScale) {
                        if (curTileInfo.tileMaxIdx == 0) {
                            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);
                        }
                        auto l1GMaxLayoutTla =
                            tla::MakeLayout<ElementLocalGlobalMax, LayoutLocalGlobalMax>(2, EpMXFP8::QS_BASE_SIZE);
                        auto l1GMaxTensorTla = tla::MakeTensor(l1LocalGlobalMaxTensor[curTileInfo.tileMaxIdx],
                                                               l1GMaxLayoutTla, Arch::PositionL1{});
                        // CopyGMaxUbToL1(curTileInfo);
                        epilogueCopyGlobalMaxUbToL1(l1GMaxTensorTla, curTileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_GMAX_UB_TO_L1_BUF0 +
                                                                          curTileInfo.tileMaxIdx / 2);
                    }
#endif
                }
            }

            // 是否要切换到下一个qs任务, 新的kvs
            if (curTaskTileIdx == curTaskInfo.taskTileNum - 1 && coreTaskId < coreTaskNum) {
                preTaskTileSum += curTaskInfo.taskTileNum;
                coreTaskId++;
            }
            if (coreTileLoop >= PV_RE_DELAY_N && delay12TaskTileIdx == delay12TaskInfo.taskTileNum - 1) {
                delay12preTaskTileSum += delay12TaskInfo.taskTileNum;
                delay12CoreTaskId++;
            }
            if (coreTileLoop >= DELAY_PSCALE_N && delay3TaskTileIdx == delay3TaskInfo.taskTileNum - 1 &&
                delay3CoreTaskId < coreTaskNum) {
                delay3preTaskTileSum += delay3TaskInfo.taskTileNum;
                delay3CoreTaskId++;
            }
            coreTileLoop++;
            // AscendC::PRINTF("-----------------------\n");
        }

        // release reverse sync flags
        ReleaseSyncFlags();
    }

    __aicore__ inline void ComputeMm1(TileInfo &runInfo)
    {
        // AscendC::PRINTF("C1 \n");
        // cubeBlock.ComputeMm1(runInfo);
    }

    __aicore__ inline void ComputeMm2(TileInfo &runInfo)
    {
        // AscendC::PRINTF("C2 \n");
        // cubeBlock.ComputeMm2(runInfo);
    }

    __aicore__ inline void ComputeVec1(TileInfo &runInfo)
    {
        // AscendC::PRINTF("V1 \n");
        // vectorBlock.ComputeVec1(runInfo);
    }

    __aicore__ inline void CopyGMaxUbToL1(TileInfo &runInfo)
    {
        // AscendC::PRINTF("CopyGMaxUbToL1 \n");
        // vectorBlock.CopyGMaxUbToL1(runInfo);
    }

    __aicore__ inline void CopyGMaxL1ToUb(TileInfo &runInfo)
    {
        // AscendC::PRINTF("CopyGMaxL1ToUb \n");
        // cubeBlock.CopyGMaxL1ToUb(runInfo);
    }

    __aicore__ inline void UpdatePScale(TileInfo &runInfo)
    {
        // AscendC::PRINTF("UpdatePScale \n");
        // vectorBlock.UpdatePScale(runInfo);
    }

    __aicore__ inline void ComputeVec2(TileInfo &runInfo)
    {
        // AscendC::PRINTF("V2  \n");
        // vectorBlock.ComputeVec2(runInfo);
    }

    __aicore__ inline void FetchBaseShapeInfo(__gm__ BlockSparseAttentionTilingData *bsaTilingData)
    {
        batch_ = bsaTilingData->batch;
        qHeads_ = bsaTilingData->numHeads;
        kvHeads_ = bsaTilingData->kvHeads;
        embed_ = bsaTilingData->embeddingSize;
        firstBatchTaskNum_ = bsaTilingData->firstBatchTaskNum;
        totalTaskNum_ = bsaTilingData->totalTaskNum;
        coreTaskNum_ = bsaTilingData->coreTaskNum;
        blockShapeX_ = bsaTilingData->blockShapeX;
        blockShapeY_ = bsaTilingData->blockShapeY;
        scaleValue_ = bsaTilingData->scaleValue;
        log2Cx_ = bsaTilingData->log2Cx;
        log2CxCeil_ = bsaTilingData->log2CxCeil;
        // mask2idx tile info
        xBlockNumAligned_ = bsaTilingData->BsaMask2IdxTileInfo.xBlockNumAligned;
        yBlockNumAligned_ = bsaTilingData->BsaMask2IdxTileInfo.yBlockNumAligned;
        avgRowPerSubCore_ = bsaTilingData->BsaMask2IdxTileInfo.avgRowPerSubCore;
        preActiveSubCoreNum_ = bsaTilingData->BsaMask2IdxTileInfo.preActiveSubCoreNum;
        // base tile info
        qBaseTile_ = bsaTilingData->BsaBaseTileInfo.qBaseTile;
        kvBaseTile_ = bsaTilingData->BsaBaseTileInfo.kvBaseTile;
        // whether actual seqlen is provided
        actSeqAval_ = (!bsaTilingData->useUniformQSeqlen) && (!bsaTilingData->useUniformKvSeqlen);
        sparseIdxSize_ = bsaTilingData->selectIdxSize;
        // aligned seqlen q & kv
        qSeqlenAligned_ = bsaTilingData->maxQSeqlen;
        kvSeqlenAligned_ = bsaTilingData->maxKvSeqlen;
    }

    __aicore__ inline void CalcOnChipBufTileInfo(__gm__ BlockSparseAttentionTilingData *bsaTilingData)
    {
        mm1L1TileM_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileM;
        mm1L1TileN_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileN;
        mm1L1TileKLeft_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileKLeft;
        mm1L1TileKRight_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileKRight;
        mm2L1TileM_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileM;
        mm2L1TileN_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileN;
        mm2L1TileKLeft_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileKLeft;
        mm2L1TileKRight_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileKRight;
        qL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.qL1BufNum;
        kL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.kL1BufNum;
        vL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.vL1BufNum;
        pL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.pL1BufNum;
        Gemm::Block::Mm1L1TileHelper mm1L1TileHelper(mm1L1TileM_, mm1L1TileN_, mm1L1TileKLeft_, mm1L1TileKRight_,
                                                     qL1BufNum_, kL1BufNum_);
        mm1L1TileHelper_ = mm1L1TileHelper;
        Gemm::Block::Mm2L1TileHelper mm2L1TileHelper(mm2L1TileM_, mm2L1TileN_, mm2L1TileKLeft_, mm2L1TileKRight_,
                                                     pL1BufNum_, vL1BufNum_);
        mm2L1TileHelper_ = mm2L1TileHelper;
        mm2L1AddrStart_ = mm1L1TileM_ * mm1L1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
                          mm1L1TileKRight_ * mm1L1TileN_ * kL1BufNum_ * sizeof(ElementK);
        mm1L0ATotalStages_ = (qBaseTile_ / BlockMmadQK::L0_TILE_M) * (embed_ / BlockMmadQK::L0_TILE_K);
        mm1L0BTotalStages_ = (kvBaseTile_ / BlockMmadQK::L0_TILE_N) * (embed_ / BlockMmadQK::L0_TILE_K);
        mm2L0ATotalStages_ = (qBaseTile_ / BlockMmadPV::L0_TILE_M) * (kvBaseTile_ / BlockMmadPV::L0_TILE_K);
        mm2L0BTotalStages_ = (kvBaseTile_ / BlockMmadPV::L0_TILE_K) * (embed_ / BlockMmadPV::L0_TILE_N);
    }

    __aicore__ inline void InitLayoutStride()
    {
        // O stride
        if constexpr (qFormat == Format::TND) {
            strideO_ = static_cast<int64_t>(qHeads_) * embed_;
        } else if constexpr (qFormat == Format::BNSD) {
            strideOB_ = static_cast<int64_t>(qHeads_) * qSeqlenAligned_ * embed_;
            strideON_ = qSeqlenAligned_ * embed_;
            strideOS_ = embed_;
        } else if constexpr (qFormat == Format::BSND) {
            strideOB_ = qSeqlenAligned_ * static_cast<int64_t>(qHeads_) * embed_;
            strideOS_ = static_cast<int64_t>(qHeads_) * embed_;
            strideON_ = embed_;
        }
        // V / V-scale / K-scale stride
        const int64_t kScaleBytesPerToken = static_cast<int64_t>(embed_ / 64) * 2;
        if constexpr (kvFormat == Format::TND) {
            strideKVData_ = static_cast<int64_t>(kvHeads_) * embed_;
            vScaleStrideN_ = static_cast<int64_t>(embed_) * 2;
            kScaleStrideN_ = kScaleBytesPerToken;
        } else if constexpr (kvFormat == Format::BNSD) {
            strideKVDataB_ = static_cast<int64_t>(kvHeads_) * kvSeqlenAligned_ * embed_;
            strideKVDataN_ = kvSeqlenAligned_ * embed_;
            vScaleStrideB_ =
                static_cast<int64_t>(kvHeads_) * CeilDiv(kvSeqlenAligned_, static_cast<int64_t>(64)) * embed_ * 2;
            vScaleStrideN_ = CeilDiv(kvSeqlenAligned_, static_cast<int64_t>(64)) * embed_ * 2;
            kScaleStrideB_ = static_cast<int64_t>(kvHeads_) * kvSeqlenAligned_ * kScaleBytesPerToken;
            kScaleStrideN_ = kvSeqlenAligned_ * kScaleBytesPerToken;
        } else if constexpr (kvFormat == Format::BSND) {
            strideKVDataB_ = kvSeqlenAligned_ * static_cast<int64_t>(kvHeads_) * embed_;
            strideKVDataN_ = embed_;
            vScaleStrideB_ =
                CeilDiv(kvSeqlenAligned_, static_cast<int64_t>(64)) * static_cast<int64_t>(kvHeads_) * embed_ * 2;
            vScaleStrideN_ = static_cast<int64_t>(embed_) * 2;
            kScaleStrideB_ = kvSeqlenAligned_ * static_cast<int64_t>(kvHeads_) * kScaleBytesPerToken;
            kScaleStrideN_ = kScaleBytesPerToken;
        }
        // Q 数据 / Q-scale stride（Host 沿 D）
        const int64_t qScaleBytesPerToken = static_cast<int64_t>(embed_ / 64) * 2;
        if constexpr (qFormat == Format::TND) {
            strideQData_ = static_cast<int64_t>(qHeads_) * embed_;
            qScaleStrideN_ = qScaleBytesPerToken;
        } else if constexpr (qFormat == Format::BNSD) {
            strideQDataB_ = static_cast<int64_t>(qHeads_) * qSeqlenAligned_ * embed_;
            strideQDataN_ = qSeqlenAligned_ * embed_;
            qScaleStrideB_ = static_cast<int64_t>(qHeads_) * qSeqlenAligned_ * qScaleBytesPerToken;
            qScaleStrideN_ = qSeqlenAligned_ * qScaleBytesPerToken;
        } else if constexpr (qFormat == Format::BSND) {
            strideQDataB_ = qSeqlenAligned_ * static_cast<int64_t>(qHeads_) * embed_;
            strideQDataN_ = embed_;
            qScaleStrideB_ = qSeqlenAligned_ * static_cast<int64_t>(qHeads_) * qScaleBytesPerToken;
            qScaleStrideN_ = qScaleBytesPerToken;
        }
    }

    __aicore__ inline void InitCrossCoreDstBuf(
        AscendC::LocalTensor<ElementDisguiseP> (&l1PTensor)[GmMXFP8::L1_P_BUF_CNT],
        AscendC::LocalTensor<ElementPScale> (&l1PScaleTensor)[GmMXFP8::L1_P_SCALE_BUF_CNT],
        AscendC::LocalTensor<ElementS> (&l1LocalGlobalMax)[GmMXFP8::L1_LOCAL_GLOBAL_MAX_BUF_CNT],
        AscendC::LocalTensor<ElementS> (&ubSTensor)[EpMXFP8::UB_S_BUF_CNT],
        AscendC::LocalTensor<ElementS> (&ubPeerGlobalMax)[EpMXFP8::UB_PEER_GLOBAL_MAX_CNT],
        AscendC::LocalTensor<ElementOTmp> &ubOTmpTensor)

    {
        // L1 侧 buffer 初始化：OFFSET + i * SIZE 递增
        for (uint32_t i = 0; i < GmMXFP8::L1_P_BUF_CNT; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementDisguiseP>(GmMXFP8::L1_P_BUF_OFFSET +
                                                                                     GmMXFP8::L1_P_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < GmMXFP8::L1_P_SCALE_BUF_CNT; i++) {
            l1PScaleTensor[i] = resource.l1Buf.template GetBufferByByte<ElementPScale>(
                GmMXFP8::L1_P_SCALE_BUF_OFFSET + GmMXFP8::L1_P_SCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < GmMXFP8::L1_LOCAL_GLOBAL_MAX_BUF_CNT; i++) {
            l1LocalGlobalMax[i] = resource.l1Buf.template GetBufferByByte<ElementS>(
                GmMXFP8::L1_LOCAL_GLOBAL_MAX_BUF_OFFSET + GmMXFP8::L1_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        // UB 侧 buffer 初始化
        for (uint32_t i = 0; i < EpMXFP8::UB_S_BUF_CNT; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(EpMXFP8::UB_S_BUF_OFFSET +
                                                                             EpMXFP8::UB_S_INNER_BUF_OFFSET * i);
        }
        for (uint32_t i = 0; i < EpMXFP8::UB_PEER_GLOBAL_MAX_CNT; i++) {
            ubPeerGlobalMax[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                EpMXFP8::UB_PEER_GLOBAL_MAX_BUF_OFFSET + EpMXFP8::UB_PEER_GLOBAL_MAX_BUF_SIZE * i);
        }
        // OTmp 单 buffer
        ubOTmpTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(EpMXFP8::UB_OTMP_BUF_OFFSET);
    }

    // 由 coreTaskId (本核上的 task 序号) 反推 globalTaskId 并填充 TaskInfo
    // 仅在 task 起始 tile (taskTileId == 0) 调用，避免每个 tile 重复计算
    // coreIdx / coreNum 由调用方 Process 传入, 避免重复调用 GetBlockIdx/GetBlockNum
    __aicore__ inline void CreateTaskInfo(TaskInfo &info, uint32_t coreTaskId, uint32_t coreIdx, uint32_t coreNum,
                                          AscendC::GlobalTensor<int64_t> &gActualQseqlen,
                                          AscendC::GlobalTensor<int64_t> &gActualKvseqlen,
                                          AscendC::GlobalTensor<ElementSparseCount> &gSparseCount,
                                          AscendC::GlobalTensor<ElementSparseIdx> &gSparseIdx,
                                          BatchOffsetInfo &batchInfo)
    {
        uint32_t taskIdx = coreTaskId * coreNum + coreIdx; // 反推全局 task 序号

        uint32_t groupSize = qHeads_ / kvHeads_;
        uint32_t qSTileNumPerFullXBlock = CeilDiv(blockShapeX_, qBaseTile_);

        // 反推 taskIdx 所在的 batch：同核 taskIdx 单调递增，
        // 从 batchInfo 保存的上次扫描位置续算，不再从 batch 0 重扫
        int64_t qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(batchInfo.curBatch) : qSeqlenAligned_;
        int64_t kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(batchInfo.curBatch) : kvSeqlenAligned_;
        uint32_t curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
        while (taskIdx >= batchInfo.curTotalTaskNum) {
            ++batchInfo.curBatch;
            batchInfo.preTotalTaskNum = batchInfo.curTotalTaskNum;
            // TND: 跨 batch 时累加上一 batch 实际长度（此时 kvSeqlen/qSeqlen 仍为上一 batch 值）
            if constexpr (qFormat == Format::TND) {
                batchInfo.oBOffset += qSeqlen * strideO_;
                batchInfo.qBOffset += qSeqlen * strideQData_;
                batchInfo.qScaleBOffset += qSeqlen * static_cast<int64_t>(qHeads_) * qScaleStrideN_;
            }
            if constexpr (kvFormat == Format::TND) {
                batchInfo.vBOffset += kvSeqlen * strideKVData_;
                batchInfo.vScaleBOffset +=
                    CeilDiv(kvSeqlen, static_cast<int64_t>(64)) * static_cast<int64_t>(kvHeads_) * embed_ * 2;
                batchInfo.kScaleBOffset += kvSeqlen * static_cast<int64_t>(kvHeads_) * kScaleStrideN_;
            }
            qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(batchInfo.curBatch) : qSeqlenAligned_;
            kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(batchInfo.curBatch) : kvSeqlenAligned_;
            curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
            batchInfo.curTotalTaskNum += curQSTileNum * qHeads_;
        }
        uint32_t curBatch = batchInfo.curBatch;

        // 拆解 taskIdx 得到 head / qSTile / xBlock 索引
        uint32_t taskIdxCurBatch = taskIdx - batchInfo.preTotalTaskNum;
        uint32_t qHeadIdx = taskIdxCurBatch / curQSTileNum;
        uint32_t kvHeadIdx = qHeadIdx / groupSize;
        uint32_t qSTileIdx = taskIdxCurBatch - qHeadIdx * curQSTileNum;
        uint32_t xBlockIdx = qSTileIdx / qSTileNumPerFullXBlock;
        uint32_t qSTileIdxCurXBlock = qSTileIdx - xBlockIdx * qSTileNumPerFullXBlock;

        // 实际 Q tile 大小 (最后一个 tile 可能不足 qBaseTile_)
        uint32_t xBlockNumAval = static_cast<uint32_t>(CeilDiv(qSeqlen, static_cast<int64_t>(blockShapeX_)));
        uint32_t xBlockSize =
            (xBlockIdx == xBlockNumAval - 1) ? static_cast<uint32_t>(qSeqlen - xBlockIdx * blockShapeX_) : blockShapeX_;
        uint32_t qSTileNumCurXBlock = CeilDiv(xBlockSize, qBaseTile_);
        uint32_t qsActBaseTile = (qSTileIdxCurXBlock == qSTileNumCurXBlock - 1) ?
                                     (xBlockSize - qSTileIdxCurXBlock * qBaseTile_) :
                                     qBaseTile_;

        // 从 sparse mask 计算 gatheredKvSeqlen
        uint32_t gmOffsetSparseCount =
            curBatch * qHeads_ * xBlockNumAligned_ + qHeadIdx * xBlockNumAligned_ + xBlockIdx;
        uint32_t yBlockNumRsvd = gSparseCount.GetValue(gmOffsetSparseCount);
        uint32_t gmOffsetSparseIdx = gmOffsetSparseCount * yBlockNumAligned_;
        uint32_t lastIdxOffset = gmOffsetSparseIdx + yBlockNumRsvd - 1;
        uint32_t lastSparseIdx = gSparseIdx.GetValue(lastIdxOffset);
        uint32_t yBlockNumAval = static_cast<uint32_t>(CeilDiv(kvSeqlen, static_cast<int64_t>(blockShapeY_)));
        uint32_t lastYBlockSize = (lastSparseIdx == yBlockNumAval - 1) ?
                                      static_cast<uint32_t>(kvSeqlen - lastSparseIdx * blockShapeY_) :
                                      blockShapeY_;
        int64_t gatheredKvSeqlen = (yBlockNumRsvd - 1) * blockShapeY_ + lastYBlockSize;

        // taskTileNum = ceil(gatheredKvSeqlen / kvBaseTile_)
        uint32_t taskTileNum = static_cast<uint32_t>(CeilDiv(gatheredKvSeqlen, static_cast<int64_t>(kvBaseTile_)));

        // O GM offset
        int64_t qSOffset = static_cast<int64_t>(xBlockIdx) * blockShapeX_ + qSTileIdxCurXBlock * qBaseTile_;
        if constexpr (qFormat == Format::TND) {
            info.gmOffsetO = batchInfo.oBOffset + qSOffset * strideO_ + qHeadIdx * embed_;
        } else if constexpr (qFormat == Format::BNSD) {
            int64_t oBOffset = static_cast<int64_t>(curBatch) * strideOB_;
            info.gmOffsetO = oBOffset + qHeadIdx * strideON_ + qSOffset * strideOS_;
        } else if constexpr (qFormat == Format::BSND) {
            int64_t oBOffset = static_cast<int64_t>(curBatch) * strideOB_;
            info.gmOffsetO = oBOffset + qSOffset * strideOS_ + qHeadIdx * strideON_;
        }
        // O shape col
        if constexpr (qFormat == Format::TND) {
            info.oShapeCol = static_cast<uint32_t>(strideO_);
        } else if constexpr (qFormat == Format::BNSD) {
            info.oShapeCol = static_cast<uint32_t>(strideOS_);
        } else if constexpr (qFormat == Format::BSND) {
            info.oShapeCol = static_cast<uint32_t>(strideOS_);
        }

        info.batchIdx = curBatch;
        info.qHeadIdx = qHeadIdx;
        info.kvHeadIdx = kvHeadIdx;
        info.qSTileIdx = qSTileIdx;
        info.xBlockIdx = xBlockIdx;
        info.gatheredKvSeqlen = gatheredKvSeqlen;
        info.qsActBaseTile = qsActBaseTile;

        info.qsActBaseTileAlign8 = CeilDiv(qsActBaseTile, 8u) * 8;
        info.qsActBaseTileAlign16 = CeilDiv(qsActBaseTile, 16u) * 16;
        info.qsActBaseTileAlign64 = CeilDiv(qsActBaseTile, 64u) * 64;
        info.qsActBaseTileAlign128 = CeilDiv(qsActBaseTile, 128u) * 128;

        info.qsBaseTile = qBaseTile_;
        info.kvsBaseTile = kvBaseTile_;
        info.taskTileNum = taskTileNum;
        // ===== PV 稀疏 gather 所需 =====
        info.gmOffsetSparseIdx = gmOffsetSparseIdx; // 本 task sparseIdx 起始
        info.kvSeqlen = kvSeqlen;
        info.yBlockNumAval = yBlockNumAval;
        info.yBlockNumRsvd = yBlockNumRsvd;
        // V 数据 batch+head 基址（seq 由 gather 的 oriStartOffset 处理，基址只含 batch+head）
        // V-scale batch+head 基址（字节单位，布局 [B,N2,S2//64,D,2]）
        if constexpr (kvFormat == Format::TND) {
            // TND V [T2,N2,D]：head 在 token 内 stride = embed_（≠ strideKVData_，后者是 batch-token 步长
            // kvHeads_*embed_）
            info.gmOffsetV = batchInfo.vBOffset + kvHeadIdx * embed_;
            info.gmOffsetVScale = batchInfo.vScaleBOffset + kvHeadIdx * vScaleStrideN_;
            info.gmOffsetK = info.gmOffsetV;
            info.gmOffsetKScale = batchInfo.kScaleBOffset + kvHeadIdx * kScaleStrideN_;
        } else if constexpr (kvFormat == Format::BNSD) {
            info.gmOffsetV = static_cast<int64_t>(curBatch) * strideKVDataB_ + kvHeadIdx * strideKVDataN_;
            info.gmOffsetVScale = static_cast<int64_t>(curBatch) * vScaleStrideB_ + kvHeadIdx * vScaleStrideN_;
            info.gmOffsetK = info.gmOffsetV;
            info.gmOffsetKScale = static_cast<int64_t>(curBatch) * kScaleStrideB_ + kvHeadIdx * kScaleStrideN_;
        } else { // BSND
            info.gmOffsetV = static_cast<int64_t>(curBatch) * strideKVDataB_ + kvHeadIdx * strideKVDataN_;
            info.gmOffsetVScale = static_cast<int64_t>(curBatch) * vScaleStrideB_ + kvHeadIdx * vScaleStrideN_;
            info.gmOffsetK = info.gmOffsetV;
            info.gmOffsetKScale = static_cast<int64_t>(curBatch) * kScaleStrideB_ + kvHeadIdx * kScaleStrideN_;
        }
        // Q / Q-scale：稠密搬需要带上本 task 的 Q seq 起点（与上方 O 的 qSOffset 一致）
        const int64_t qScaleBytesPerTok = static_cast<int64_t>(embed_ / 64) * 2;
        if constexpr (qFormat == Format::TND) {
            info.gmOffsetQ = batchInfo.qBOffset + qSOffset * strideQData_ + qHeadIdx * embed_;
            info.gmOffsetQScale = batchInfo.qScaleBOffset +
                                  qSOffset * static_cast<int64_t>(qHeads_) * qScaleBytesPerTok +
                                  qHeadIdx * qScaleStrideN_;
        } else if constexpr (qFormat == Format::BNSD) {
            info.gmOffsetQ =
                static_cast<int64_t>(curBatch) * strideQDataB_ + qHeadIdx * strideQDataN_ + qSOffset * embed_;
            info.gmOffsetQScale = static_cast<int64_t>(curBatch) * qScaleStrideB_ + qHeadIdx * qScaleStrideN_ +
                                  qSOffset * qScaleBytesPerTok;
        } else { // BSND
            info.gmOffsetQ = static_cast<int64_t>(curBatch) * strideQDataB_ +
                             qSOffset * static_cast<int64_t>(qHeads_) * embed_ + qHeadIdx * embed_;
            info.gmOffsetQScale = static_cast<int64_t>(curBatch) * qScaleStrideB_ +
                                  qSOffset * static_cast<int64_t>(qHeads_) * qScaleBytesPerTok +
                                  qHeadIdx * qScaleStrideN_;
        }
        // [PV-VERIFY] 打印本 task 算出的稀疏 gather 偏移，供测试逐行核对（补丁1/2 验证用）
        // AscendC::printf(
        //     "[PV-VERIFY] t=%u b=%u qh=%u kvh=%u xb=%u | gmOffV=%llu gmOffVSc=%llu spIdx=%u kvSeq=%llu yAval=%u
        //     yRsvd=%u\n", (unsigned)coreTaskId, (unsigned)info.batchIdx, (unsigned)info.qHeadIdx,
        //     (unsigned)info.kvHeadIdx, (unsigned)info.xBlockIdx, (unsigned long long)info.gmOffsetV, (unsigned long
        //     long)info.gmOffsetVScale, (unsigned)info.gmOffsetSparseIdx, (unsigned long long)info.kvSeqlen,
        //     (unsigned)info.yBlockNumAval, (unsigned)info.yBlockNumRsvd);
        // AscendC::printf(
        //     "[QK-VERIFY] t=%u qsAct=%u qSOff=%lld | gmOffQ=%llu gmOffK=%llu gmOffQSc=%llu gmOffKSc=%llu "
        //     "gathered=%llu taskTiles=%u\n",
        //     (unsigned)coreTaskId, (unsigned)info.qsActBaseTile, (long long)qSOffset,
        //     (unsigned long long)info.gmOffsetQ, (unsigned long long)info.gmOffsetK,
        //     (unsigned long long)info.gmOffsetQScale, (unsigned long long)info.gmOffsetKScale,
        //     (unsigned long long)info.gatheredKvSeqlen, (unsigned)info.taskTileNum);
    }

    // 每个 tile 调用一次，根据 coreTileLoop / curTaskTileIdx / taskInfo 填充 TileInfo
    __aicore__ inline void CreateTileInfo(TileInfo &info, uint64_t coreTileLoop, uint32_t curTaskTileIdx,
                                          const TaskInfo &taskInfo)
    {
        info.loop = static_cast<uint32_t>(coreTileLoop);
        info.curKvsTileLoopIdx = curTaskTileIdx;
        info.isFirstKvsTile = (curTaskTileIdx == 0);
        info.isLastKvsTile = (curTaskTileIdx + 1 == taskInfo.taskTileNum);
        info.isLastSecondKvsTile =
            ((curTaskTileIdx + 2 == taskInfo.taskTileNum) || (taskInfo.taskTileNum == 1)); // pv rowsum刷1，同 mxfp4

        info.isUpdatePScale = (info.isLastKvsTile || ((info.curKvsTileLoopIdx + 1) % MXFP8Kernel::TILE_GROUP_N == 0));
        info.isTileGoupFirstTile = (info.curKvsTileLoopIdx % MXFP8Kernel::TILE_GROUP_N == 0);
        if (info.isFirstKvsTile) {
            kvsFirstTileStartVecCore_ = static_cast<uint32_t>(coreTileLoop % 2);
        }
        info.kvsFirstTileStartVecCore = kvsFirstTileStartVecCore_;

        if (info.isTileGoupFirstTile && !info.isFirstKvsTile) { // 非第一个 kvs tileGroup，更新 dm id（同 mxfp4）
            updateScaleIdx_ = (updateScaleIdx_ + 1) % 4;
        }

        // task 第一个 tile 或 TileGroup 第一个 tile 时切换 buff (双缓冲 / 四缓冲)
        if (info.isFirstKvsTile || info.isTileGoupFirstTile) {
            tileLoopIdx_ = (tileLoopIdx_ + 1) % 2;
            tileMaxIdx_ = (tileMaxIdx_ + 1) % 4;
            pscaleNum_ = (pscaleNum_ + 1) % 20;
        }
        info.pscaleNum = static_cast<uint16_t>(pscaleNum_ / 10);
        info.tileBuffIdx = tileLoopIdx_;
        info.tileMaxIdx = tileMaxIdx_;
        info.updateScaleIdx = updateScaleIdx_;
        info.isKvsFirstTilePerCore = (info.curKvsTileLoopIdx % MXFP8Kernel::TILE_GROUP_N / 2 == 0);

        info.kvsActBaseTile =
            info.isLastKvsTile ? (taskInfo.gatheredKvSeqlen - (taskInfo.taskTileNum - 1) * kvBaseTile_) : kvBaseTile_;
        // PV 稀疏 gather 用：当前 tile 在 task 内 gather 后的 KV base tile 序号
        info.pvGatheredKvSTileIdx = curTaskTileIdx;
        info.kvsActBaseTileAlign16 = (info.kvsActBaseTile + 15) / 16 * 16;
        info.kvsActBaseTileAlign32 = (info.kvsActBaseTile + 31) / 32 * 32;
        info.kvsActBaseTileAlign64 = (info.kvsActBaseTile + 63) / 64 * 64;
    }

    __aicore__ inline void InitSyncFlags()
    {
        if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK + 16);
        } else {
            // 核间
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[0]);
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[1]);
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
            // mask2index
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);

            // vec
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_VEC1_RES_BUF0_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_VEC1_RES_BUF1_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_ATTN_BUF_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF0_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF1_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF2_FLAG);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF3_FLAG);
        }
    }

    __aicore__ inline void ReleaseSyncFlags()
    {
        if ASCEND_IS_AIC {
            // 核间
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0]);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0] + 16);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1]);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1] + 16);
        } else {
            // 核间
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_GMAX_L1_UB_BACK);

            // mask2index
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);

            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);

            // vec
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_VEC1_RES_BUF0_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_VEC1_RES_BUF1_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_ATTN_BUF_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF0_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF1_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF2_FLAG);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EpMXFP8::SYNC_GMAX_UB_TO_L1_BUF3_FLAG);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    /*
    tiling info, which are const in each kernel launch
    */
    // basic shape info
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t firstBatchTaskNum_;
    uint32_t totalTaskNum_;
    uint32_t coreTaskNum_; // 向下取整: totalTaskNum_ / blockDim
    uint32_t blockShapeX_;
    uint32_t blockShapeY_;
    float scaleValue_;
    float log2Cx_;
    float log2CxCeil_;
    // mask2idx tile info
    uint32_t xBlockNumAligned_;
    uint32_t yBlockNumAligned_;
    uint32_t avgRowPerSubCore_;
    uint32_t preActiveSubCoreNum_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    // whether actual seqlen is provided
    uint32_t actSeqAval_;
    // workspace size
    uint64_t sparseIdxSize_;
    // aligned seqlen q & kv
    int64_t qSeqlenAligned_;
    int64_t kvSeqlenAligned_;
    // L1 tile info
    uint32_t mm1L1TileM_;
    uint32_t mm1L1TileN_;
    uint32_t mm1L1TileKLeft_;
    uint32_t mm1L1TileKRight_;
    uint32_t mm2L1TileM_;
    uint32_t mm2L1TileN_;
    uint32_t mm2L1TileKLeft_;
    uint32_t mm2L1TileKRight_;
    uint32_t qL1BufNum_;
    uint32_t kL1BufNum_;
    uint32_t vL1BufNum_;
    uint32_t pL1BufNum_;
    uint32_t mm1L0ATotalStages_;
    uint32_t mm1L0BTotalStages_;
    uint32_t mm2L0ATotalStages_;
    uint32_t mm2L0BTotalStages_;
    uint32_t mm2L1AddrStart_ = 0;
    Gemm::Block::Mm1L1TileHelper mm1L1TileHelper_;
    Gemm::Block::Mm2L1TileHelper mm2L1TileHelper_;
    // ===== stride 族（O / V 数据 / V-scale）=====
    int64_t strideO_;
    int64_t strideOB_;
    int64_t strideON_;
    int64_t strideOS_;
    // ===== V 数据 / V-scale stride（PV 稀疏 gather 用，operator() 内预算一次）=====
    // V 数据为 fp4，sizeof(fp8_e4m3fn_t)=1，元素偏移即字节偏移
    int64_t strideKVData_ = 0;  // TND: 每 token 跨 kvHead 的步长（fp4 元素）
    int64_t strideKVDataB_ = 0; // BNSD/BSND: V 数据 batch 步长
    int64_t strideKVDataN_ = 0; // BNSD/BSND: V 数据 head 步长
    // V-scale 为 E8M0(1 字节)，GM 布局 [B,N2,S2//64,D,2]；B16 视图下每 64-S2 行 group 占 D 个 B16 = 2D 字节
    int64_t vScaleStrideB_ = 0; // V-scale batch 步长（字节）
    int64_t vScaleStrideN_ = 0; // V-scale head 步长（字节）
    // ===== K-scale / Q / Q-scale stride（QK Mm1 用；K-scale Host 沿 D）=====
    int64_t kScaleStrideB_ = 0;
    int64_t kScaleStrideN_ = 0;
    int64_t strideQData_ = 0;
    int64_t strideQDataB_ = 0;
    int64_t strideQDataN_ = 0;
    int64_t qScaleStrideB_ = 0;
    int64_t qScaleStrideN_ = 0;

    // ---- CreateTileInfo 持久状态 (跨 tileGroup 保持 ----
    uint32_t kvsFirstTileStartVecCore_ = 0; // task 首 tile 时切换 (loop % 2)
    uint32_t updateScaleIdx_ = 3;           // C2 Sync 时切换 (循环 0~3)// dm
    uint32_t tileLoopIdx_ = 1;              // 双缓冲索引 (0/1)
    uint32_t tileMaxIdx_ = 3;               // 四缓冲索引 (0~3)
    uint32_t pscaleNum_ = 19;               // 20 缓冲索引 (0~19)，Dpv=20
};

} // namespace BsaKernelArch35
