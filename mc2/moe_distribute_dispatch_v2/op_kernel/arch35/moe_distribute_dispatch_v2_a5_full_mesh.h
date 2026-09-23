/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_a5_full_mesh.h
 * \brief
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_A5_FULL_MESH_H
#define MOE_DISTRIBUTE_DISPATCH_V2_A5_FULL_MESH_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/hccl/hccl.h"
#include "simt_api/asc_simt.h"
#include "adv_api/math/cumsum.h"
#include "adv_api/reduce/sum.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../moe_distribute_dispatch_v2_tiling.h"
#include "../moe_distribute_dispatch_v2_quant.h"
#include "../../../common/op_kernel/mc2_moe_context.h"
#include "../moe_distribute_v2_base.h"
#include "../check_winsize.h"
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#include "../hccl_context_holder.h"
#include "../mc2_moe_context_holder.h"
#define FLOAT_OVERFLOW_MODE_CTRL 60
namespace MoeDistributeDispatchV2A5FullMeshImpl {
using namespace Mc2Kernel;
constexpr uint64_t FLAG_FIELD_OFFSET = 768UL * 1024UL; // 384 * 2，0/1标识区偏移
constexpr uint64_t SPLIT_BLOCK_SIZE = 512UL;
constexpr uint64_t SPLIT_BLOCK_COUNT = 128UL; // 128 = SPLIT_BLOCK_SIZE / sizeof(float)
constexpr int32_t FULL_MESH_MAX_UB_SIZE = 190 * 1024;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_DATA_COUNT = 120U; // 120 = SPLIT_BLOCK_DATA_SIZE / sizeof(float)
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr uint32_t CUMSUM_MAX_CORE_NUM = 8U;
// 按32核上限预留前缀和数据区（32*32*32=32KB）
constexpr uint64_t CUMSUM_WS_FLAG_OFFSET = 32UL * 1024UL;
// CCU 控制面软同步标志位于 per-AIV 状态字（512B 槽）空闲字：状态区启动时清零、跨 launch 持久
// （workspace 不做初始化清零，依赖零基线的标志不得放 workspace）；word0-7 为 InitWinState 块、8-9 为 HCCL DFX
// 前缀表就绪字：cumsum 核 scan+写回后置 1.0f、各核查后清；word16=槽内 64B 偏移，保 DataCopy 基址 32B 对齐
constexpr uint32_t CCU_READY_WORD_POS = 16U;
// CCU 控制面前缀 scan 的 SIMT 线程数（sortlib 先例 THREAD_DIM_NUM=1024；矩阵规模 W×localExpertNum
// 常见 64~1024，取 256 时每线程 1~4 元素；LAUNCH_BOUND 与 dim3 传值保持一致，同 sortlib 用法）
constexpr uint32_t CCU_SIMT_THREAD_NUM = 256U;
// SIMT 单线程 chunk 元素上界：totalNum=Pe×W ≤ LOCAL_EXPERT_MAX_SIZE(2048)（host tiling 校验）⇒
// chunkSize = Ceil(totalNum, 256) ≤ 8——直写 GM 版的线程私有暂存数组按此定长
constexpr uint32_t CCU_SIMT_CHUNK_MAX = 8U;
// count 直方图化：复用 common.h GetExpertFreq（u8 输入 × u16 桶 × 256 桶寄存器直方图，跨 repeat 累加）
constexpr uint32_t CCU_HIST_BIN_NUM = 256U;
// 分组直方图每组可读桶数：bin 255 恒为组外/-1 收集桶（只落不读），专家数超 255 时分多趟建组
constexpr uint32_t CCU_HIST_GROUP_SIZE = 255U;
constexpr uint32_t RUNPOS_CALCUMSUM = 2U;
constexpr uint32_t RUNPOS_CUMSUMFLAG = 3U;
constexpr uint32_t RUNPOS_ARRIVECNT = 4U;
constexpr uint8_t VALID_EVENT_FLAG_NUM = 8U;
constexpr uint8_t LOCAL_COPY_BUFFER_NUM = 2U;
constexpr uint32_t DATA_COPY_MAX_BLOCK_COUNT = 4095U;
constexpr uint8_t UB_ALIGN_DATA_COUNT = 8U; // 8 = UB_ALIGN / sizeof(float) = UB_ALIGN / sizeof(int32_t)
constexpr uint32_t DURATION_OFFSET = sizeof(int64_t) / sizeof(int32_t);

#define TemplateMC2A5FullMeshTypeClass \
    typename ContextHolder, typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist
#define TemplateMC2A5FullMeshTypeFunc ContextHolder, XType, ExpandXOutType, QuantMode, IsSmoothScaleExist

using namespace AscendC;
using namespace MoeDistributeV2Base;
using namespace Mc2Aclnn;

// CCU 控制面：count 矩阵 (src 主序) → (expert 主序 × src 次序) 展平全局前缀——SIMT block-scan。
// 标量双循环在 vector 核标量单元上过慢（W×localMoe 次读改写）；SIMT 每线程独立寻址、跨步免惩罚，
// 链式依赖按经典两阶段分解：各线程扫自己 chunk（含转置寻址 s*segWords+j）→ syncthreads →
// 全线程 Hillis-Steele 就地把 chunk 小计扫成含端前缀（对数深度并行，取代线程 0 的 T 长串行链）→
// 各线程取前邻槽为排他偏移写出（输出连续）；再一次 syncthreads 后阶段四就地产出 expertTokenNums
// （专家末源前缀/count 差分，int64 直写，免 GM 跨步取数）。
// 整数加法结合律保证结果与串行标量版逐位一致；asc_vf_call 阻塞调用核直至线程完成（sortlib 先例）。
template <typename CountT>
__simt_vf__ LAUNCH_BOUND(CCU_SIMT_THREAD_NUM) __aicore__
    void CcuPrefixScanSimt(__ubuf__ CountT *mat, __ubuf__ CountT *out, __ubuf__ CountT *chunkOffsets,
                           __ubuf__ int64_t *tokenNums, uint32_t totalNum, uint32_t worldSize, uint32_t segWords,
                           uint32_t expertNum, uint32_t tokenNumsMode)
{
    uint32_t tid = threadIdx.x;
    uint32_t chunkSize = (totalNum + CCU_SIMT_THREAD_NUM - 1U) / CCU_SIMT_THREAD_NUM;
    uint32_t begin = tid * chunkSize;
    uint32_t end = begin + chunkSize;
    if (end > totalNum) {
        end = totalNum;
    }
    CountT runSum = 0;
    for (uint32_t i = begin; i < end; ++i) {
        // 展平序 i = j*W + s（expert 主序 × src 次序）→ src 主序矩阵地址 s*segWords + j
        uint32_t j = i / worldSize;
        uint32_t s = i - j * worldSize;
        runSum += mat[s * segWords + j];
        out[i] = runSum; // chunk 内局部前缀，阶段三统一加偏移
    }
    chunkOffsets[tid] = (begin < totalNum) ? runSum : 0;
    asc_syncthreads();
    // 阶段二（并行）：Hillis-Steele 就地把 chunk 小计扫成含端前缀，取代线程 0 的 256 长串行标量链
    // （A5 弱标量，该链是小矩阵场景的 scan 主导项）。步序"共读 → 屏障 → 各写己槽 → 屏障"：跨线程读
    // 全部落在屏障前、写只触本线程槽位，跨 warp 无读写交错；整数加法结合律保证与串行版逐位一致。
    // scanWidth=有效 chunk 数（空 chunk 小计恒 0，尾部不参与扫描），步数 ≤ log2(256)=8
    uint32_t scanWidth = (totalNum + chunkSize - 1U) / chunkSize;
    for (uint32_t d = 1U; d < scanWidth; d <<= 1) {
        CountT add = (tid >= d && tid < scanWidth) ? chunkOffsets[tid - d] : 0;
        asc_syncthreads();
        chunkOffsets[tid] += add;
        asc_syncthreads();
    }
    // 含端 → 排他：offset = Σ 小计[0..tid-1]；tid ≥ scanWidth 的空 chunk 线程不消费 offset
    CountT offset = (tid == 0U) ? 0 : chunkOffsets[tid - 1U];
    for (uint32_t i = begin; i < end; ++i) {
        out[i] += offset;
    }
    // 阶段四：全部线程落定后就地产出 expertTokenNums——T[j] = 专家末源含端前缀 P[(j+1)·W-1]，count 模式
    // 差分 T[j]-T[j-1] 直写 int64；免去 GM 跨步取数（基址 (W-1)·4 非 32B 对齐）与 Sub/Cast 向量链
    asc_syncthreads();
    for (uint32_t j = tid; j < expertNum; j += CCU_SIMT_THREAD_NUM) {
        CountT expertTotal = out[(j + 1U) * worldSize - 1U];
        if (tokenNumsMode != 0U) {
            expertTotal -= (j == 0U) ? 0 : out[j * worldSize - 1U];
        }
        tokenNums[j] = static_cast<int64_t>(expertTotal);
    }
}

// CCU 控制面直写 GM 版（A/B 备选，tiling useCcuScanGm 门控，默认关）：与上方 UB 版同算法，差异仅在
// 数据落点与访存形态——阶段零只读（转置读 GM 矩阵入线程私有暂存，CCU 引擎刚写完天然 L2 热）→ 两阶段
// block-scan → 阶段三只写（in-place 覆写 GM 前缀表并双写 sendCounts）→ 阶段四读回产出 tokenNums。
// 省 MTE2 矩阵搬入/MTE3 写回与 2×matWords scan 段 UB，且只写 [0,totalNum)。
// in-place 安全的真正前提：mat≡out 同缓冲且转置置换非恒等——若阶段一在屏障前写 out[i]，会与其它线程
// 未完成的转置读（跨 warp 独立调度）交错致数据错乱。故阶段零【只读不写】（私有数组 vals 暂存，cnt ≤
// CCU_SIMT_CHUNK_MAX）、阶段三【只写不读 mat】，全部读与全部写被两次 syncthreads 严格隔离。
// __gm__ volatile 为 SIMT 直写 GM 的仓内惯例（gather_v2_simt 先例）；消费侧协议与 UB 版完全一致。
template <typename CountT>
__simt_vf__ LAUNCH_BOUND(CCU_SIMT_THREAD_NUM) __aicore__
    void CcuPrefixScanSimtGm(__gm__ volatile CountT *mat, __gm__ volatile CountT *out, __ubuf__ CountT *chunkOffsets,
                             __gm__ volatile int64_t *tokenNums, __gm__ volatile int32_t *sendCounts, uint32_t totalNum,
                             uint32_t worldSize, uint32_t segWords, uint32_t expertNum, uint32_t tokenNumsMode)
{
    uint32_t tid = threadIdx.x;
    uint32_t chunkSize = (totalNum + CCU_SIMT_THREAD_NUM - 1U) / CCU_SIMT_THREAD_NUM;
    uint32_t begin = tid * chunkSize;
    uint32_t end = begin + chunkSize;
    if (end > totalNum) {
        end = totalNum;
    }
    uint32_t cnt = (begin < totalNum) ? (end - begin) : 0U; // 空 chunk 防无符号下溢
    // 阶段零（只读）：展平序 i = j*W + s（expert 主序 × src 次序）→ src 主序矩阵地址 s*segWords + j，
    // 暂存线程私有数组——不出屏障不写 mat/out，in-place 竞争的根除点
    CountT vals[CCU_SIMT_CHUNK_MAX];
    CountT runSum = 0;
    for (uint32_t k = 0; k < cnt; ++k) {
        uint32_t i = begin + k;
        uint32_t j = i / worldSize;
        uint32_t s = i - j * worldSize;
        vals[k] = mat[s * segWords + j]; // GM/L2 载入
        runSum += vals[k];
    }
    chunkOffsets[tid] = (begin < totalNum) ? runSum : 0;
    asc_syncthreads();
    // 阶段二改并行（同 UB 版）：Hillis-Steele 对数深度扫 chunk 小计，取代线程 0 串行链；步内
    // "共读-屏障-各写己槽-屏障"隔离跨 warp 读写交错，scanWidth 收窄步数与写集
    uint32_t scanWidth = (totalNum + chunkSize - 1U) / chunkSize;
    for (uint32_t d = 1U; d < scanWidth; d <<= 1) {
        CountT add = (tid >= d && tid < scanWidth) ? chunkOffsets[tid - d] : 0;
        asc_syncthreads();
        chunkOffsets[tid] += add;
        asc_syncthreads();
    }
    // 阶段三（只写）：chunk 内局部前缀（重放 vals）+ 排他偏移 = 含端前缀，落 out 并双写 sendCounts
    // （rscvStatusNum_ ≡ totalNum）；全程不回读 mat
    CountT offset = (tid == 0U) ? 0 : chunkOffsets[tid - 1U];
    CountT localSum = 0;
    for (uint32_t k = 0; k < cnt; ++k) {
        localSum += vals[k];
        CountT prefix = localSum + offset;
        out[begin + k] = prefix;
        sendCounts[begin + k] = static_cast<int32_t>(prefix);
    }
    // 阶段四：全部线程落定后就地产出 expertTokenNums——T[j] = 专家末源含端前缀 P[(j+1)·W-1]，count 模式
    // 差分 T[j]-T[j-1] 直写 GM int64（读回 out 为 L2 命中）
    asc_syncthreads();
    for (uint32_t j = tid; j < expertNum; j += CCU_SIMT_THREAD_NUM) {
        CountT expertTotal = out[(j + 1U) * worldSize - 1U];
        if (tokenNumsMode != 0U) {
            expertTotal -= (j == 0U) ? 0 : out[j * worldSize - 1U];
        }
        tokenNums[j] = static_cast<int64_t>(expertTotal);
    }
}
template <TemplateMC2A5FullMeshTypeClass>
class MoeDistributeDispatchV2A5FullMesh {
public:
    using XInType = typename std::conditional<
        (Std::IsSame<XType, fp4x2_e2m1_t>::value) || (Std::IsSame<XType, fp4x2_e1m2_t>::value), uint8_t, XType>::type;
    using XOutType = typename std::conditional<(Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value) ||
                                                   (Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value),
                                               uint8_t, ExpandXOutType>::type;
    __aicore__ inline MoeDistributeDispatchV2A5FullMesh(){};
    __aicore__ inline void Init(GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask,
                                GM_ADDR expertScales, GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut,
                                GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut, GM_ADDR expandScalesOut,
                                GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
                                const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ExpIdsCopyAndMaskCal();
    __aicore__ inline void TokenActiveMaskCal();
    __aicore__ inline void InitElasticInfo();
    __aicore__ inline void SetDataStatus();
    __aicore__ inline void SetTilingData(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void CalValidBSCnt(LocalTensor<bool> maskStrideTensor);
    __aicore__ inline void CalValidExpIdx(LocalTensor<bool> maskInputTensor);
    __aicore__ inline void GenerateGatherMaskTensor(uint32_t maskCnt);
    __aicore__ inline void MaskZeroComputeExpert(uint32_t maskCnt);
    __aicore__ inline void ZeroComputeExpertMaskCal();
    __aicore__ inline void SetTilingDataAndCal(const MoeDistributeDispatchV2TilingData *tilingData);
    __aicore__ inline void SendToSharedExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf);
    __aicore__ inline void SendToMoeExpert(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf);
    __aicore__ inline void SendToMoeExpertLoop(TQue<QuePosition::VECIN, 1> inQueue, uint32_t validTokenNum,
                                               bool writeExpertScale);
    __aicore__ inline void ExpertActiveMaskInit();
    __aicore__ inline void ExpertActiveMaskCal();
    __aicore__ inline void CalcSendTokenBufNum(TBuf<> &outBuf);
    __aicore__ inline void AllToAllDispatchA5(TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf);
    __aicore__ inline void AllToAllDispatch();
    __aicore__ inline void CalCumSum();
    __aicore__ inline void WaitCumSumFlag();
    __aicore__ inline void CcuWaitMatrixReady(); // CCU 控制面：自旋等待唯一 cumsum 核（最后核）的 Wait 完成
    __aicore__ inline int32_t CalExpertCnt(uint32_t expertId, uint32_t maskCnt);
    __aicore__ inline void CalAndSendCntByExp();
    __aicore__ inline void CalAndSendCntByRankAndExpert();
    __aicore__ inline void SendExpertCntRange(uint32_t startExpertId, uint32_t endExpertId);
    __aicore__ inline void CalCntPackAndIssueCcu(); // CCU 控制面：count 面编排（计数→AlltoAll→前缀→扇出）
    __aicore__ inline void CcuCalCountsToSegBuf(); // CCU 控制面：专家计数（分组直方图）构造并落入 sendBuf 段
    __aicore__ inline void CcuScanAndPublishPrefix(); // CCU 控制面：全表前缀 scan、写回、两统计输出+就绪字广播
    __aicore__ inline void CcuScanAndPublishPrefixGm(); // CCU 控制面：直写 GM 版 scan（A/B 备选，env 门控）
    __aicore__ inline void BuildSendCntFromCcuMatrix(); // CCU 控制面：各核拷自己前缀表切片
    __aicore__ inline void BufferInit();
    __aicore__ inline void WaitDispatchClearStatus();
    __aicore__ inline void GatherSumRecvCnt(LocalTensor<float> &gatherMaskOutTensor,
                                            LocalTensor<uint32_t> &gatherTmpTensor,
                                            LocalTensor<float> &statusSumOutTensor);
    __aicore__ inline void CalRecvAndSetFlag();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void GetCumSum(LocalTensor<int32_t> &outLocal, uint32_t newAivId);
    __aicore__ inline void RunPosRecord(const uint32_t runPos);
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void SetValidExpertInfo(uint32_t expInfoSize, uint32_t &validNum);
    __aicore__ inline uint32_t CheckDataArriveWithFlag(uint32_t srcExpDataIdx, int32_t beginIdx, int32_t copyCnt);
    __aicore__ inline void CopyInAndOut(LocalTensor<XOutType> xTmpTensor, GM_ADDR wAddr, uint32_t index,
                                        uint32_t dstPosition, uint32_t arriveCount, uint32_t dataBufferId,
                                        bool isDataBufferReused);
    __aicore__ inline void WaitAndFormatOutput(TBuf<> tBuf, uint32_t validNum);
    __aicore__ inline uint32_t GetLocalWindowSrcDataBlockIdx(uint32_t srcExpertId, uint32_t localExpertNum);
    __aicore__ inline void ClearLocalWindowDataFlags(TBuf<> tBuf, uint32_t validNum, uint32_t localExpertNum);
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId,
                                       uint32_t &endTokenId, uint32_t &sendTokenNum, bool isFront = true);
    __aicore__ inline void FillTriple(LocalTensor<XOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k,
                                      bool writeExpertScale);
    __aicore__ inline void CalTokenSendExpertCnt(uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt);
    __aicore__ inline void TokenToExpertInQuant(GlobalTensor<XOutType> dstWinGMTensor,
                                                TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex,
                                                uint32_t toExpertId, uint32_t toExpertIndex, bool writeExpertScale);
    __aicore__ inline void TokenToExpert(GlobalTensor<XOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue,
                                         uint32_t srcTokenIndex, uint32_t toExpertIndex, bool writeExpertScale);
    __aicore__ inline void RecordRankCommDuration(LocalTensor<int32_t> &performanceInfoTensor, uint64_t startTime);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindAddrByRankId(rankId, epRankIdOriginal_) + winDataSizeOffset_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return ctx_.GetWindStateAddrByRankId(rankId, epRankIdOriginal_) + dataState_ * WIN_STATE_OFFSET;
    }

    // per-AIV 状态字（FLAG_FIELD_OFFSET 起每 AIV 一个 512B 槽，启动清零、跨 launch 持久）的 float 视图：
    // CCU 软同步标志专用（word10+ 空闲段；就绪字取 word16 保 32B 对齐），勿放 workspace——首轮脏值可能假命中
    __aicore__ inline GlobalTensor<float> GetAivStatusFlagTensor(const uint32_t coreId)
    {
        GlobalTensor<uint32_t> statusWordTensor;
        statusWordTensor.SetGlobalBuffer((__gm__ uint32_t *)(statusDataSpaceGM_ + FLAG_FIELD_OFFSET +
                                                             static_cast<uint64_t>(coreId) * WIN_ADDR_ALIGN));
        return statusWordTensor.ReinterpretCast<float>();
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XInType> xGMTensor_;
    GlobalTensor<int32_t> expertIdsGMTensor_;
    GlobalTensor<float> scalesGMTensor_;
    GlobalTensor<float> expertScalesGMTensor_;
    GlobalTensor<uint8_t> dynamicScalesOutGMTensor_;
    GlobalTensor<int64_t> expertTokenNumsOutGMTensor_;
    GlobalTensor<float> windowInstatusFp32Tensor_;
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;
    GlobalTensor<int32_t> performanceInfoGMTensor_;
    GlobalTensor<float> selfRankWinInGMTensor_;
    GlobalTensor<float> cumsumWsGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    LocalTensor<int32_t> statusTensor_;
    LocalTensor<int32_t> waitStatusTensor_;
    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> validExpertIdsTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<int32_t> performanceInfoTensor_;
    LocalTensor<int32_t> performanceFlagTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<float> expertScalesTensor_;
    LocalTensor<float> statusCleanFp32Tensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<XOutType> outTensor_;
    LocalTensor<XOutType> tempTensor_;
    LocalTensor<float> floatLocalTemp_;
    LocalTensor<uint32_t> expertMapTensor_;
    LocalTensor<uint32_t> expertFinishNumTensor_;
    LocalTensor<uint32_t> expertLeftNumTensor_;
    LocalTensor<uint32_t> dataStateLocalTensor_;

    LocalTensor<float> flagReduceWorkTensor_;
    LocalTensor<float> flagRecvTensor_;

    TBuf<> statusBuf_;
    TBuf<> tokenNumBuf_;
    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> elasticInfoBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> gatherMaskOutBuf_;
    TBuf<> sumCoreBuf_;
    TBuf<> sumLocalBuf_;
    TBuf<> sumContinueBuf_;
    TBuf<> scalarBuf_;
    TBuf<> validExpertIndexBuf_;
    TBuf<> validBsIndexTBuf_;
    TBuf<> performanceInfoBuf_;
    TBuf<> performanceFlagBuf_;
    TBuf<> calBeginBuf_;
    TBuf<> calEndBuf_;
    GM_ADDR expandXOutGM_;
    GM_ADDR expandScalesOutGM_;
    GM_ADDR sendCountsOutGM_;
    GM_ADDR statusSpaceGM_;
    GM_ADDR windowGM_;
    GM_ADDR recvCntWorkspaceGM_;
    GM_ADDR statusDataSpaceGM_;

    // tiling侧已确保数据上限，相乘不会越界，因此统一采用uint32_t进行处理
    uint32_t syncFlagId_{0};
    uint8_t sendTokenBufNum_{0};
    uint32_t axisBS_{0};
    uint32_t axisMaxBS_{0};
    uint32_t axisH_{0};
    uint32_t axisK_{0};
    uint32_t aivNum_{0};
    uint32_t sharedUsedAivNum_{0};
    uint32_t moeUsedAivNum_{0};
    uint32_t epWorldSize_{0};
    uint32_t epWorldSizeOriginal_{0};
    int32_t epRankId_{0};
    int32_t epRankIdOriginal_{0};
    uint32_t aivId_{0}; // aiv id
    uint32_t sharedExpertNum_{0};
    uint32_t sharedExpertRankNum_{0};    // 共享专家卡数
    uint32_t rankNumPerSharedExpert_{0}; // 部署单个共享专家所用的卡数
    uint32_t moeExpertNum_{0};
    uint32_t moeExpertRankNum_{0}; // moe专家卡数，等于epWorldSize_ - sharedExpertRankNum_
    uint32_t moeExpertNumPerRank_{0};
    uint32_t totalExpertNum_{0};
    uint32_t hOutSize_{0};
    uint32_t hOutSizeAlign_{0};
    uint32_t hAlignSize_{0};
    uint32_t scaleInBytes_{0};
    uint32_t scaleOutBytes_{0};
    uint32_t scalesCount_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t startId_;
    uint32_t endId_;
    uint32_t sendNum_;
    uint32_t statusCntAlign_;
    uint32_t dataState_{0};
    uint32_t tBufRealSize_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t activeMaskBsCnt_{0};
    uint64_t sendToMoeExpTokenCnt_{0};
    bool isTokenMaskFlag_ = false;
    bool isExpertMaskFlag_ = false;
    bool hasElasticInfoFlag_ = false;
    bool hasExpertScalesFlag_ = false;
    bool isPerformanceFlag_ = false;
    bool isShareExpertRankFlag_ = false;
    bool isScalingDownFlag_ = false;
    uint64_t totalWinSize_{0};
    uint32_t expertTokenNumsType_{1};
    int32_t expertIdsCnt_{0};
    int32_t tokenQuantAlign_{0};
    uint32_t expertScaleAlign_{0};
    int32_t zeroComputeExpertNum_{0};
    uint32_t blockCntPerToken_{0};
    uint32_t axisHCommu_{0};
    uint32_t hCommuSize_{0};
    uint32_t expertIdsBufSize_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t endStatusIndex_{0};
    uint32_t recStatusNumPerCore_{0};
    uint32_t aivUsedCumSum_{0};
    uint32_t aivUsedAllToAll_{0};
    uint32_t maxSize_{0};
    uint32_t expertIdsSize_{0};
    uint32_t globalBS_{0};
    uint32_t copyInAxisH_{0};
    uint32_t copyOutAxisH_{0};
    uint64_t totalUbSize_{0};

    // 统一上下文持有器：v2 走 HcclOpParam，v3 走 Mc2MoeContext
    ContextHolder ctx_;

    // ==================== CCU 控制面（useCcuCtrl_ 时激活，count 交换走 AlltoAll）====================
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> ccuHccl_; // 控制面实例，与数据面 context 共存
    AscendC::HcclHandle ccuCntHandle_{0};
    GM_ADDR ccuSendBufGM_{nullptr}; // workspace 尾部：本 rank 的 W 段等长行 [counts...]
    GM_ADDR ccuRecvBufGM_{nullptr}; // workspace 尾部：src 主序全矩阵，Wait 后各核自读
    uint32_t ccuSegWords_{0};       // 段 u32 字数 = Ceil(moeExpertNumPerRank_, 8) * 8，与 host 同公式
    bool useCcuCtrl_{false};
    bool useCcuScanGm_{false}; // CCU 前缀 scan 直写 GM 路径（A/B 验证，默认 false=UB 中转版）

    DataCopyParams hCopyParams_;
    DataCopyParams dataStateParams_{1U, sizeof(uint32_t), 0U, 0U};

    MoeDistributeDispatchV2Quant<XInType, ExpandXOutType, XOutType, QuantMode, IsSmoothScaleExist> quantInst_;
};

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::InitElasticInfo()
{
    uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t);
    uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(elasticInfoBuf_, elasticInfoSizeAlign);
    elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
    DataCopyExtParams elasticInfoParams = {
        1U, static_cast<uint32_t>((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t)), 0U,
        0U, 0U};
    DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(elasticInfoTensor_, elasticInfoGMTensor_, elasticInfoParams, elasticInfoCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    isScalingDownFlag_ = elasticInfoTensor_.GetValue(0);
    if (isScalingDownFlag_) {
        epWorldSize_ = elasticInfoTensor_.GetValue(EP_WORLD_SIZE_IDX);
        sharedExpertRankNum_ = elasticInfoTensor_.GetValue(SHARE_RANK_NUM_IDX);
        moeExpertNum_ = elasticInfoTensor_.GetValue(MOE_NUM_IDX);
        epRankId_ = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epRankId_);
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SetTilingData(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    epWorldSizeOriginal_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    epRankIdOriginal_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    hasElasticInfoFlag_ = tilingData->moeDistributeDispatchV2Info.hasElasticInfo;
    hasExpertScalesFlag_ = tilingData->moeDistributeDispatchV2Info.hasExpertScales;
    isPerformanceFlag_ = tilingData->moeDistributeDispatchV2Info.isPerformance;
    epRankId_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    globalBS_ = tilingData->moeDistributeDispatchV2Info.globalBs;
    epWorldSize_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    sharedExpertRankNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertRankNum;
    moeExpertNum_ = tilingData->moeDistributeDispatchV2Info.moeExpertNum;
    sharedExpertNum_ = tilingData->moeDistributeDispatchV2Info.sharedExpertNum;
    expertTokenNumsType_ = tilingData->moeDistributeDispatchV2Info.expertTokenNumsType;
    zeroComputeExpertNum_ = tilingData->moeDistributeDispatchV2Info.zeroComputeExpertNum;
    isTokenMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isTokenMask;
    isExpertMaskFlag_ = tilingData->moeDistributeDispatchV2Info.isExpertMask;
    axisK_ = tilingData->moeDistributeDispatchV2Info.k;
    aivNum_ = tilingData->moeDistributeDispatchV2Info.aivNum;
    scaleInBytes_ =
        tilingData->moeDistributeDispatchV2Info.scalesCol * tilingData->moeDistributeDispatchV2Info.scalesTypeSize;
    scalesCount_ = tilingData->moeDistributeDispatchV2Info.scalesCount;
    axisMaxBS_ = globalBS_ / epWorldSizeOriginal_;
    maxSize_ = tilingData->moeDistributeDispatchV2Info.maxSizeForUbBuffer;
    totalUbSize_ = tilingData->moeDistributeDispatchV2Info.totalUbSize;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SetTilingDataAndCal(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    SetTilingData(tilingData);
    copyInAxisH_ = axisH_;
    copyOutAxisH_ = axisH_;
    if constexpr (Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value ||
                  Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value) {
        copyOutAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE);
    }
    if constexpr (Std::IsSame<XType, fp4x2_e2m1_t>::value || Std::IsSame<XType, fp4x2_e1m2_t>::value) {
        copyInAxisH_ = Ceil(axisH_, FP4_ELEMS_PER_BYTE);
    }
    if (hasElasticInfoFlag_) {
        InitElasticInfo();
    }
    isShareExpertRankFlag_ = (epRankId_ < sharedExpertRankNum_);
    if (sharedExpertNum_ > 0) {
        rankNumPerSharedExpert_ = sharedExpertRankNum_ / sharedExpertNum_;
    }
    moeExpertRankNum_ = epWorldSize_ - sharedExpertRankNum_;
    moeExpertNumPerRank_ = moeExpertNum_ / moeExpertRankNum_;
    expertIdsCnt_ = axisBS_ * axisK_;
    hOutSize_ = copyOutAxisH_ * sizeof(XOutType);
    quantInst_.QuantInit(hAlignSize_, hOutSize_, scaleInBytes_, tokenQuantAlign_, hOutSizeAlign_, scaleOutBytes_,
                         axisH_);
    // 因为与V2中预留给三元组的大小不一致，需要重新计算
    hOutSizeAlign_ = tokenQuantAlign_ * sizeof(int32_t) + UB_ALIGN;
    if (hasExpertScalesFlag_) {
        expertScaleAlign_ = tokenQuantAlign_ + EXPAND_IDX_INFO;
    }
    blockCntPerToken_ = Ceil(hOutSizeAlign_, SPLIT_BLOCK_DATA_SIZE);
    hCommuSize_ = blockCntPerToken_ * SPLIT_BLOCK_SIZE;
    axisHCommu_ = hCommuSize_ / sizeof(XOutType);
    expertPerSizeOnWin_ = axisMaxBS_ * hCommuSize_;
    rscvStatusNum_ = isShareExpertRankFlag_ ? epWorldSize_ : (epWorldSize_ * moeExpertNumPerRank_);
    totalExpertNum_ = sharedExpertRankNum_ + moeExpertNum_;
    statusCntAlign_ = Ceil(totalExpertNum_, UB_ALIGN_DATA_COUNT) * UB_ALIGN_DATA_COUNT;
    aivUsedCumSum_ = totalExpertNum_ / 16; // 单核处理16个专家cnt发送
    aivUsedCumSum_ = (aivUsedCumSum_ == 0) ? 1 : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= (aivNum_ / 2)) ? (aivNum_ / 2) : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= CUMSUM_MAX_CORE_NUM) ? CUMSUM_MAX_CORE_NUM : aivUsedCumSum_;
    // 确保每个核至少处理一个状态
    aivUsedCumSum_ = (aivUsedCumSum_ >= rscvStatusNum_) ? rscvStatusNum_ : aivUsedCumSum_;
    // CCU 控制面：count 面（计算+组包+issue+Wait）收敛单核（唯一 cumsum 核），其余核全部还给 token
    // 发送；单核串行免去 pack 软同步，且与前台各核的 token 发送天然并行（MTE 路径仍按上方钳位多核）。
    // V2/V3 入口均经 TryEnableCcuCtrl 显式写入该字段（FinalizeDispatchTiling 兜底默认关），无脏值
    if (tilingData->moeDistributeDispatchV2Info.useCcuCtrl && !hasElasticInfoFlag_) {
        aivUsedCumSum_ = 1U;
        // 临时打桩（流程确认用，验证后删除）：仅 0 核打印，避免 aivNum_ 份刷屏
        if (GetBlockIdx() == 0) {
            AscendC::printf("[CCU_CTRL] useCcuCtrl branch hit: aivNum_=%u, cumsum core id=%u\n", aivNum_, aivNum_ - 1U);
        }
    }
    aivUsedAllToAll_ = aivNum_ - aivUsedCumSum_;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivUsedAllToAll_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    moeUsedAivNum_ = aivUsedAllToAll_ - sharedUsedAivNum_;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SetDataStatus()
{
    statusDataSpaceGM_ = ctx_.GetStatusDataSpaceGm();
    uint32_t epRankIdHccl = ctx_.GetEpRankId();
    uint32_t epWorldSizeHccl = ctx_.GetEpWorldSize();
    selfDataStatusGMTensor_.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusDataSpaceGM_ + FLAG_FIELD_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeHccl, epRankIdOriginal_, moeExpertNum_,
                              epWorldSizeOriginal_, globalBS_, dataStateBuf);
    uint64_t hSizeAlignCombine = Ceil(axisH_ * COMBINE_IN_DATA_SIZE, SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_SIZE;
    winDataSizeOffset_ =
        dataState_ * (totalWinSize_ / BUFFER_NUM) + axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::Init(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
    GM_ADDR expandScalesOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    tpipe_ = pipe;
    tpipe_->InitBuffer(calBeginBuf_, UB_ALIGN);
    aivId_ = GetBlockIdx();
    totalWinSize_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
    ctx_.InitAndCheck(mc2Context, tilingData->moeDistributeDispatchV2Info.epWorldSize, totalWinSize_, tpipe_,
                      expandXOut);
    xGMTensor_.SetGlobalBuffer((__gm__ XInType *)x);
    xActiveMaskGMTensor_.SetGlobalBuffer((__gm__ bool *)xActiveMask);
    expertIdsGMTensor_.SetGlobalBuffer((__gm__ int32_t *)expertIds);
    dynamicScalesOutGMTensor_.SetGlobalBuffer((__gm__ uint8_t *)dynamicScalesOut);
    expertTokenNumsOutGMTensor_.SetGlobalBuffer((__gm__ int64_t *)expertTokenNumsOut);
    expandIdxGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(expandIdxOut));
    elasticInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(elasticInfo));
    scalesGMTensor_.SetGlobalBuffer((__gm__ float *)scales);
    quantInst_.dynamicScalesOutGMTensor_ = dynamicScalesOutGMTensor_;
    SetTilingDataAndCal(tilingData);
    // 临时诊断（必须位于 SetDataStatus 之前——那是首笔窗口派生地址的解引用，表项无效时 fault 会
    // 发生在本打印之前；验证后与 [CCU_CTRL] 打桩同批删除）。方案 C 下槽 0 = CCU 上下文（单槽覆写 +
    // CCU 会话），预期签名同三层一致版实测：ctx 有效、selfIn/in0/in1 全 0（CCU 视图无 windowsIn，
    // 数据面不读它——V3 走张量）、cuOut0 有效、winSize≈2×HCCL_BUFFSIZE 量级。两行独立打印按
    // 风险递增（任一行 fault 不殃及前行）；cuOut1/cuOut2 非 0 且 != cuOut0 = CCU 视图携带远端
    // rank 窗口映射（MTE flag 面候选，读数组固定槽位无越界风险）
    if (tilingData->moeDistributeDispatchV2Info.useCcuCtrl && (GetBlockIdx() == 0)) {
        __gm__ HcclCombineOpParam *slot0Ctx = (__gm__ HcclCombineOpParam *)(GetHcclContext<HCCL_GROUP_ID_0>());
        if ((unsigned long long)(uintptr_t)slot0Ctx != 0ULL) {
            uint32_t ccuSelfIdx = slot0Ctx->rankId & 63U; // 63 掩码防异常 rankId 越界读表（容量 64）
            AscendC::printf("[CCU_CTX0] ctx=0x%llx selfIn=0x%llx in0=0x%llx in1=0x%llx\n",
                            (unsigned long long)(uintptr_t)slot0Ctx,
                            (unsigned long long)slot0Ctx->windowsIn[ccuSelfIdx],
                            (unsigned long long)slot0Ctx->windowsIn[0], (unsigned long long)slot0Ctx->windowsIn[1]);
            AscendC::printf("[CCU_CTX1] cuOut0=0x%llx cuOut1=0x%llx cuOut2=0x%llx winSize=%llu rank=%u\n",
                            (unsigned long long)slot0Ctx->windowsOut[0], (unsigned long long)slot0Ctx->windowsOut[1],
                            (unsigned long long)slot0Ctx->windowsOut[2], (unsigned long long)slot0Ctx->winSize,
                            slot0Ctx->rankId);
        } else {
            AscendC::printf("[CCU_CTX0] ctx=NULL\n");
        }
    }
    if (hasExpertScalesFlag_) {
        expertScalesGMTensor_.SetGlobalBuffer((__gm__ float *)expertScales);
        expandScalesOutGM_ = expandScalesOut;
    }
    if (isPerformanceFlag_) {
        performanceInfoGMTensor_.SetGlobalBuffer((__gm__ int32_t *)(performanceInfo));
    }
    SetDataStatus();
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut;
    recvCntWorkspaceGM_ = AscendC::GetUserWorkspace(workspaceGM);
    statusSpaceGM_ = GetWindStateAddrByRankId(epRankIdOriginal_);
    windowInstatusFp32Tensor_.SetGlobalBuffer((__gm__ float *)(statusSpaceGM_));
    selfRankWinInGMTensor_.SetGlobalBuffer((__gm__ float *)(statusDataSpaceGM_));
    uint64_t cumsumWsBaseOffset = static_cast<uint64_t>(WORKSPACE_ELEMENT_OFFSET) * aivNum_ * aivNum_;
    cumsumWsGMTensor_.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t *)(recvCntWorkspaceGM_) + cumsumWsBaseOffset));
    windowGM_ = GetWindAddrByRankId(epRankIdOriginal_);
    hCopyParams_ = {1U, static_cast<uint32_t>(copyInAxisH_ * sizeof(XInType)), 0U, 0U};
    dataStateParams_ = {1U, sizeof(uint32_t), 0U, 0U};
    expertIdsSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;

    // CCU 控制面初始化（仅 A5 且 tiling 显式开启；弹性降缩路径不使能，维持原 MTE count 面）。
    // V2/V3（Mc2MoeContext）入口均经 TryEnableCcuCtrl 显式写入 useCcuCtrl（阶段 1 起同款 CCU 卸载）
    useCcuCtrl_ = tilingData->moeDistributeDispatchV2Info.useCcuCtrl && (!hasElasticInfoFlag_);
    // 与 useCcuCtrl_ 相与：弹性路径/脏 tiling 缓冲（host 未写该字段时）都不会误入直写 GM 分支
    useCcuScanGm_ = useCcuCtrl_ && tilingData->moeDistributeDispatchV2Info.useCcuScanGm;
    if (useCcuCtrl_) {
        // 方案 C：CCU 会话（op_api env 门控 ServerType=CCU）+ 单槽 CCU 覆写（matmul 生产配方）——
        // 槽 0 即 CCU 上下文（CCU 槽位注入仅在 CCU 会话发生）；数据面 V3 路径读 mc2Context 张量不受影响
        __gm__ HcclCombineOpParam *ccuContext = (__gm__ HcclCombineOpParam *)(GetHcclContext<HCCL_GROUP_ID_0>());
        ccuHccl_.InitV2((GM_ADDR)ccuContext, tilingData);
        ccuHccl_.SetCcTilingV2(offsetof(MoeDistributeDispatchV2TilingData, mc2CcTiling1));
        ccuSegWords_ = ((moeExpertNumPerRank_ + 7U) / 8U) * 8U;
        // CCU 缓冲自系统预留区之后（GetUserWorkspace）起步：host 侧 SetCcuCtrlTiling 保留 MTE 容量公式
        // 并在其后追加 CCU 缓冲（布局与是否 CCU 卸载无关——防 tiling key 复用下新旧二进制错配越界）；
        // CCU 模式下 [16MB, 16MB+MTE区) 无消费者、纯冗余分配。
        // [sendBuf: W*segWords*4B][recvBuf 半区0][recvBuf 半区1]——recvBuf 跟随 dataState_ 双缓冲，
        // 防止相邻两轮 launch 间对端引擎写入与本端读取踩踏（矩阵新鲜度由 Wait 契约保证，双半区防跨轮踩踏）
        ccuSendBufGM_ = recvCntWorkspaceGM_;
        uint64_t ccuRecvSize = static_cast<uint64_t>(epWorldSize_) * ccuSegWords_ * sizeof(uint32_t);
        ccuRecvBufGM_ = ccuSendBufGM_ + ccuRecvSize + dataState_ * ccuRecvSize;
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::FillTriple(
    LocalTensor<XOutType> &xOutTensor, uint32_t tokenIndex, uint32_t k, bool writeExpertScale)
{
    LocalTensor<int32_t> xOutTint32 = xOutTensor.template ReinterpretCast<int32_t>();
    xOutTint32(tokenQuantAlign_) = epRankId_;      // 0:epRankId index
    xOutTint32(tokenQuantAlign_ + 1) = tokenIndex; // 1:token index
    xOutTint32(tokenQuantAlign_ + 2) = k;          // 2:topK value index
    if ((k < axisK_) && writeExpertScale) {
        LocalTensor<float> xOutTfloat = xOutTensor.template ReinterpretCast<float>();
        xOutTfloat(expertScaleAlign_) = expertScalesTensor_.GetValue(tokenIndex * axisK_ + k);
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::TokenToExpertInQuant(
    GlobalTensor<XOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex,
    uint32_t fillExpertIdx, uint32_t quantExpertIdx, bool writeExpertScale)
{
    DataCopyPadParams copyPadParams{true, 0U, 0U, 0U};
    LocalTensor<XInType> xInTensor = inQueue.AllocTensor<XInType>();
    DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], hCopyParams_, copyPadParams);
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XInType>();
    if constexpr (QuantMode > UNQUANT) {
        quantInst_.QuantProcess(tempTensor_, xInTensor, quantExpertIdx, scalesCount_, scalesGMTensor_);
    } else { // 非量化直转hifp8
        Cast(floatLocalTemp_, xInTensor, RoundMode::CAST_NONE, axisH_);
        Cast(tempTensor_, floatLocalTemp_, RoundMode::CAST_ROUND, axisH_);
    }
    inQueue.FreeTensor<XInType>(xInTensor);
    SyncFunc<AscendC::HardEvent::V_S>();
    FillTriple(tempTensor_, srcTokenIndex, fillExpertIdx, writeExpertScale);
    SyncFunc<AscendC::HardEvent::S_V>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    LocalTensor<int32_t> tempTensorInt32 = tempTensor_.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> outTensorInt32 =
        (outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_]).template ReinterpretCast<int32_t>();
    // 64 = 256 / sizeof(int32_t) 一次操作字节数; 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32, tempTensorInt32, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    // 64：偏移前一次拷贝的256字节；56 = (480 - 256) / sizeof(int32_t)
    // 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[64], tempTensorInt32[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    DataCopy(dstWinGMTensor, outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_], axisHCommu_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    syncFlagId_++;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::TokenToExpert(
    GlobalTensor<XOutType> dstWinGMTensor, TQue<QuePosition::VECIN, 1> inQueue, uint32_t srcTokenIndex,
    uint32_t toExpertIndex, bool writeExpertScale)
{
    DataCopyPadParams copyPadParams{false, 0U, 0U, 0U};
    LocalTensor<XInType> xInTensor = inQueue.AllocTensor<XInType>();
    if constexpr (!IsSmoothScaleExist) {
        DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * axisH_], hCopyParams_, copyPadParams);
    } else {
        DataCopyParams scaleInParams = {1U, static_cast<uint16_t>(scaleInBytes_), 0U, 0U};
        DataCopyPadParams padParams = {true, 0, 0, 0};
        auto tmp = scalesGMTensor_.ReinterpretCast<uint8_t>();
        DataCopyPad(xInTensor, xGMTensor_[srcTokenIndex * copyInAxisH_], hCopyParams_, copyPadParams);
        DataCopyPad(xInTensor[Align32(copyInAxisH_)].template ReinterpretCast<uint8_t>(),
                    tmp[srcTokenIndex * scaleInBytes_], scaleInParams, padParams);
    }
    inQueue.EnQue(xInTensor);
    xInTensor = inQueue.DeQue<XInType>();
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    FillTriple(xInTensor, srcTokenIndex, toExpertIndex, writeExpertScale);
    SyncFunc<AscendC::HardEvent::S_V>();
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    LocalTensor<int32_t> xInTensorInt32 = xInTensor.template ReinterpretCast<int32_t>();
    LocalTensor<int32_t> outTensorInt32 =
        (outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_]).template ReinterpretCast<int32_t>();
    // 64 = 256 / sizeof(int32_t) 一次操作字节数; 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32, xInTensorInt32, uint64_t(64), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    // 64：偏移前一次拷贝的256字节；56 = (480 - 256) / sizeof(int32_t)
    // 16、15分别为dst、src相邻迭代间地址步长
    Copy(outTensorInt32[64], xInTensorInt32[64], uint64_t(56), uint8_t(blockCntPerToken_), {1, 1, 16, 15});
    inQueue.FreeTensor<XInType>(xInTensor);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(syncFlagId_ % sendTokenBufNum_);
    DataCopy(dstWinGMTensor, outTensor_[(syncFlagId_ % sendTokenBufNum_) * axisHCommu_], axisHCommu_);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(syncFlagId_ % sendTokenBufNum_);
    syncFlagId_++;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum,
    bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_;
    } else if (aivId_ >= aivUsedAllToAll_) { // aiv中后面aivUsedCumSum_个核给cusum计算使用
        newAivId = aivId_ - aivUsedAllToAll_;
    } else {
        newAivId = aivId_ - moeUsedAivNum_; // aivUsedAllToAll_中后面的核发送给共享专家
    }
    startTokenId = sendTokenNum * newAivId; // 每个aiv发送时的起始rankid
    if (newAivId < remainderTokenNum) {     // 前remainderRankNum个aiv需要多发1个卡的数据
        sendTokenNum += 1;
        startTokenId += newAivId;
    } else {
        startTokenId += remainderTokenNum;
    }
    endTokenId = startTokenId + sendTokenNum;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SendToSharedExpert(
    TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf)
{
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * sendTokenBufNum_ / sizeof(float));
    PipeBarrier<PIPE_V>();
    // 分核
    uint32_t startTokenId, endTokenId, sendTokenNum;
    // 参数 activeMaskBsCnt_、sharedExpertNum_、sharedUsedAivNum_
    uint32_t curSendCnt = activeMaskBsCnt_ * sharedExpertNum_;
    SplitToCore(curSendCnt, sharedUsedAivNum_, startTokenId, endTokenId, sendTokenNum, false);
    if (startTokenId >= curSendCnt) {
        return;
    }
    // 发送
    GlobalTensor<XOutType> dstWinGMTensor;
    // 计算目的共享专家卡在其所在共享专家组的id
    uint32_t idInSharedGroup = epRankId_ % rankNumPerSharedExpert_;
    syncFlagId_ = 0;
    for (int i = 0; i < sendTokenBufNum_; i++) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }
    for (uint32_t virtualTokenIndex = startTokenId; virtualTokenIndex < endTokenId; ++virtualTokenIndex) {
        uint32_t sendTokenIndex = virtualTokenIndex % activeMaskBsCnt_;
        uint32_t toSharedExpertIndex = virtualTokenIndex / activeMaskBsCnt_;
        int32_t toRankId = idInSharedGroup + toSharedExpertIndex * rankNumPerSharedExpert_;
        if (isScalingDownFlag_) {
            toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
        }
        dstWinGMTensor.SetGlobalBuffer((__gm__ XOutType *)(uint64_t(GetWindAddrByRankId(toRankId)) +
                                                           expertPerSizeOnWin_ * epRankId_ +
                                                           sendTokenIndex * hCommuSize_));
        uint32_t srcTokenIndex = sendTokenIndex;
        if (isExpertMaskFlag_) {
            srcTokenIndex = validBsIndexTensor_.GetValue(sendTokenIndex);
        }
        if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
            uint32_t fillExpertIdx = axisK_ + toSharedExpertIndex;
            uint32_t quantExpertIdx = toSharedExpertIndex;
            TokenToExpertInQuant(dstWinGMTensor, inQueue, srcTokenIndex, fillExpertIdx, quantExpertIdx, false);
        } else {
            TokenToExpert(dstWinGMTensor, inQueue, srcTokenIndex, axisK_ + toSharedExpertIndex, false);
        }
    }
    for (int i = 0; i < sendTokenBufNum_; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SendToMoeExpert(
    TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf)
{
    // 按Token分核
    LocalTensor<float> outTensorFp32 = outBuf.Get<float>();
    Duplicate<float>(outTensorFp32, float(1), hCommuSize_ * sendTokenBufNum_ / sizeof(float));
    uint32_t validTokenNum = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    syncFlagId_ = 0;

    for (int i = 0; i < sendTokenBufNum_; i++) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }

    // 在循环外区分有无专家权重，避免性能劣化
    if (hasExpertScalesFlag_) {
        SendToMoeExpertLoop(inQueue, validTokenNum, true);
    } else {
        SendToMoeExpertLoop(inQueue, validTokenNum, false);
    }

    for (int i = 0; i < sendTokenBufNum_; i++) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(i % sendTokenBufNum_);
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SendToMoeExpertLoop(
    TQue<QuePosition::VECIN, 1> inQueue, uint32_t validTokenNum, bool writeExpertScale)
{
    GlobalTensor<XOutType> dstWinGMTensor;
    int32_t dstTokenIdx = 0;

    for (int32_t index = aivId_; index < validTokenNum; index += moeUsedAivNum_) {
        int32_t tokenId = index / axisK_;
        int32_t topKId = index % axisK_;
        int32_t expertId = validExpertIdsTensor_(index);
        if (expertId >= moeExpertNum_ || expertId < 0)
            continue;
        int32_t toRankId = expertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
        if (isScalingDownFlag_) {
            toRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + toRankId);
        }

        CalTokenSendExpertCnt(expertId, index, dstTokenIdx);

        dstWinGMTensor.SetGlobalBuffer(
            (__gm__ XOutType *)(uint64_t(GetWindAddrByRankId(toRankId)) +
                                expertPerSizeOnWin_ * ((epRankId_ + toRankId) % epWorldSize_ * moeExpertNumPerRank_ +
                                                       expertId % moeExpertNumPerRank_) +
                                dstTokenIdx * hCommuSize_));
        if (hasElasticInfoFlag_) {
            dstWinGMTensor.SetGlobalBuffer((__gm__ XOutType *)(uint64_t(GetWindAddrByRankId(toRankId)) +
                                                               expertPerSizeOnWin_ * (epRankId_ * moeExpertNumPerRank_ +
                                                                                      expertId % moeExpertNumPerRank_) +
                                                               dstTokenIdx * hCommuSize_));
        }
        if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
            uint32_t quantExpertIdx = expertId + sharedExpertNum_;
            TokenToExpertInQuant(dstWinGMTensor, inQueue, tokenId, topKId, quantExpertIdx, writeExpertScale);
        } else {
            TokenToExpert(dstWinGMTensor, inQueue, tokenId, topKId, writeExpertScale);
        }
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalcSendTokenBufNum(
    TBuf<> &outBuf)
{
    tpipe_->InitBuffer(calEndBuf_, UB_ALIGN);
    uint64_t beiginUbAddr = (calBeginBuf_.Get<uint8_t>()).GetPhyAddr();
    uint64_t endUbAddr = (calEndBuf_.Get<uint8_t>()).GetPhyAddr();
    uint64_t remainUbSize = totalUbSize_ - (endUbAddr - beiginUbAddr + UB_ALIGN);

    sendTokenBufNum_ = remainUbSize / hCommuSize_;
    if (sendTokenBufNum_ > VALID_EVENT_FLAG_NUM)
        sendTokenBufNum_ = VALID_EVENT_FLAG_NUM;
    if (sendTokenBufNum_ == 0) {
        return;
    }

    tpipe_->InitBuffer(outBuf, hCommuSize_ * sendTokenBufNum_);
    outTensor_ = outBuf.Get<XOutType>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::AllToAllDispatch()
{
    // 使用的全局参数
    TQue<QuePosition::VECIN, 1> inQueue;
    TBuf<> tempBuf, outBuf, inQueueCleanBuf, calTempBuf;
    TBuf<> smoothScalesBuf, receiveDataCastFloatBuf;
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    AscendC::TBufPool<AscendC::TPosition::VECIN> tbufPool0, tbufPool1;
    tpipe_->InitBufPool(tbufPool0, BUFFER_NUM * hAlignSize_);
    tpipe_->InitBufPool(tbufPool1, BUFFER_NUM * hAlignSize_, tbufPool0);

    tbufPool0.InitBuffer(inQueue, BUFFER_NUM, hAlignSize_);
    tbufPool1.InitBuffer(inQueueCleanBuf, BUFFER_NUM * hAlignSize_);
    // MX以及PERGROUP量化在计算scales时每次搬入256字节数据，
    // 所以在token搬入前需要对空间填0，避免引入脏数据
    if constexpr ((QuantMode == MX_QUANT) || (QuantMode == PERGROUP_DYNAMIC_QUANT) || (QuantMode == MX_QUANT_CLIP)) {
        LocalTensor<uint8_t> inTensor8 = inQueueCleanBuf.Get<uint8_t>();
        Duplicate(inTensor8, QUANT_PADDING_VALUE, BUFFER_NUM * hAlignSize_);
    }
    uint32_t calTokenIdxBuffSize = Ceil(axisBS_ * axisK_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    if (hasExpertScalesFlag_) {
        tpipe_->InitBuffer(expertScalesBuf_, expertIdsBufSize_);
    }
    bool needMaskCalFlag = (isTokenMaskFlag_ || isExpertMaskFlag_ || zeroComputeExpertNum_ != 0);
    if (needMaskCalFlag) {
        tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);
    }
    if constexpr ((QuantMode > UNQUANT) || (QuantMode == UNQUANT && !Std::IsSame<ExpandXOutType, XType>::value)) {
        hOutAlignUbSize_ = Ceil(hOutSizeAlign_, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(tempBuf, hOutAlignUbSize_);
        tpipe_->InitBuffer(receiveDataCastFloatBuf, maxSize_);
        tpipe_->InitBuffer(smoothScalesBuf, maxSize_);
        tempTensor_ = tempBuf.Get<XOutType>();
        floatLocalTemp_ = receiveDataCastFloatBuf.Get<float>();
        smoothScalesTensor_ = smoothScalesBuf.Get<float>();
        dstExpBuf_ = receiveDataCastFloatBuf; // 内存复用
        subExpBuf_ = smoothScalesBuf;         // 内存复用
    } else if (needMaskCalFlag) {
        tpipe_->InitBuffer(dstExpBuf_, maxSize_);
        tpipe_->InitBuffer(subExpBuf_, maxSize_);
    } else {
        tpipe_->InitBuffer(dstExpBuf_, calTokenIdxBuffSize);
        tpipe_->InitBuffer(subExpBuf_, calTokenIdxBuffSize);
    }
    tpipe_->InitBuffer(calTempBuf, calTokenIdxBuffSize);
    workLocalTensor_ = calTempBuf.Get<float>();
    quantInst_.SetQuantInitParams(floatLocalTemp_, smoothScalesTensor_, smoothScalesBuf, dynamicScalesOutGMTensor_);
    ExpIdsCopyAndMaskCal();
    if (activeMaskBsCnt_ == 0) {
        return;
    }
    AllToAllDispatchA5(inQueue, outBuf);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::AllToAllDispatchA5(
    TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf)
{
    CalcSendTokenBufNum(outBuf);
    if ((aivId_ >= moeUsedAivNum_) && (sharedExpertRankNum_ != 0)) {
        SendToSharedExpert(inQueue, outBuf);
    } else {
        SendToMoeExpert(inQueue, outBuf);
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalTokenSendExpertCnt(
    uint32_t dstExpertId, int32_t calCnt, int32_t &curExpertCnt)
{
    if (calCnt < axisK_) {
        curExpertCnt = 0;
        return;
    }
    LocalTensor<int32_t> dstExpIdTensor = dstExpBuf_.Get<int32_t>();
    LocalTensor<int32_t> subExpIdTensor = subExpBuf_.Get<int32_t>();
    Duplicate<int32_t>(dstExpIdTensor, dstExpertId, calCnt);
    PipeBarrier<PIPE_V>();
    Sub(subExpIdTensor, validExpertIdsTensor_, dstExpIdTensor, calCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
    LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
    Abs(tmpoutFp32, tmpFp32, calCnt);
    PipeBarrier<PIPE_V>();
    Mins(subExpIdTensor, dstExpIdTensor, 1, calCnt);
    PipeBarrier<PIPE_V>();
    ReduceSum<float>(tmpoutFp32, tmpFp32, workLocalTensor_, calCnt);
    SyncFunc<AscendC::HardEvent::V_S>();
    int32_t curOtherExpertCnt = dstExpIdTensor(0);
    if (calCnt >= curOtherExpertCnt) {
        curExpertCnt = calCnt - curOtherExpertCnt;
    } else {
        curExpertCnt = 0;
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline int32_t MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalExpertCnt(
    uint32_t expertId, uint32_t maskCnt)
{
    int32_t expertCnt = 0;
    if ((expertId < sharedExpertRankNum_) && (activeMaskBsCnt_ > 0)) { // 当前处理专家id为共享专家
        if (expertId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_) {
            expertCnt = activeMaskBsCnt_;
        }
    } else if (sendToMoeExpTokenCnt_ > 0) { // 当前处理卡为moe专家卡
        int32_t moeExpertId = expertId - sharedExpertRankNum_;
        CalTokenSendExpertCnt(moeExpertId, maskCnt, expertCnt);
    }
    return expertCnt;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalAndSendCntByExp()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    uint32_t maskCnt = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    SplitToCore(totalExpertNum_, aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum, false);
    if (startExpertId >= totalExpertNum_) {
        return;
    }
    // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    uint64_t mask[2] = {0x101010101010101, 0};
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusCntAlign_ / 8, 1,
                       8); // 0x3F800000为float的1 8为一次操作8个block
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t curExpertId = startExpertId; curExpertId < endExpertId; ++curExpertId) {
        int32_t curExpertCnt = CalExpertCnt(curExpertId, maskCnt);
        // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
        int32_t cntPosIndex = (curExpertId - startExpertId) * 8 + 1;
        statusTensor_.SetValue(cntPosIndex, curExpertCnt);
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        uint32_t dstRankId = curExpertId;
        uint32_t offset = STATE_OFFSET * epRankId_;
        if (curExpertId >= sharedExpertRankNum_) {
            dstRankId = ((curExpertId - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset += ((curExpertId - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_ * STATE_OFFSET);
        }
        if (isScalingDownFlag_) {
            dstRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + dstRankId);
        }
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(dstRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(curExpertId - startExpertId) * UB_ALIGN_DATA_COUNT],
                          UB_ALIGN_DATA_COUNT);
    }
    // reset操作前需确保前面操作完成
    PipeBarrier<PIPE_ALL>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SendExpertCntRange(
    uint32_t startExpertId, uint32_t endExpertId)
{
    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t expertId = startExpertId; expertId < endExpertId;) {
        bool isSharedExpert = expertId < sharedExpertRankNum_;
        uint32_t moeExpertId = isSharedExpert ? 0U : expertId - sharedExpertRankNum_;
        uint32_t dstRankId = isSharedExpert ? expertId : moeExpertId / moeExpertNumPerRank_ + sharedExpertRankNum_;
        uint32_t localExpertId = isSharedExpert ? 0U : moeExpertId % moeExpertNumPerRank_;
        uint32_t sendExpertNum = 1U;
        if (!isSharedExpert) {
            sendExpertNum = endExpertId - expertId;
            uint32_t expertNumLeftInRank = moeExpertNumPerRank_ - localExpertId;
            sendExpertNum = sendExpertNum < expertNumLeftInRank ? sendExpertNum : expertNumLeftInRank;
        }

        uint32_t targetRankId = dstRankId;
        if (isScalingDownFlag_) {
            targetRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + dstRankId);
        }
        uint32_t offset = STATE_OFFSET * epRankId_ + localExpertId * epWorldSize_ * STATE_OFFSET;
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(targetRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
        DataCopyParams cntCopyParams = {static_cast<uint16_t>(sendExpertNum), 1U, 0U,
                                        static_cast<uint16_t>(epWorldSize_ - 1)};
        DataCopy<int32_t>(rankGMTensor, statusTensor_[expertId * UB_ALIGN_DATA_COUNT], cntCopyParams);
        expertId += sendExpertNum;
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CcuCalCountsToSegBuf()
{
    // 专家计数构造与落段：优先分组直方图（与专家数解耦，复用 common.h 的 GetExpertFreq）——G=Ceil(E,255)
    // 趟（E=1024 仅 4 趟），每趟 ids 平移到 [0,254]、组外与 -1 padding 钳到 bin 255（只落不读桶）。
    // A5 标量弱：不做逐专家查桶/组包循环——每趟一次向量 Cast 把桶值追加进 moeId 主序 counts 数组，
    // 再由两条 strided DataCopy 整写 sendBuf 段（MoE 段布局 [dst][local] 与 moeId 主序同构：
    // moeId=(dst−S)·Pe+local，段间仅隔 segWords−Pe 的 padding 字），run-length 组包机制不复存在。
    uint32_t maskCnt = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;

    // maskCnt==0（全 mask）时直方图无输入，走原路径（CalExpertCnt 对空区间返回 0）；
    // 桶为 u16 且单桶计数上界=maskCnt，maskCnt>65535 时会回绕，退回逐专家归约（int32 精确）。
    // 专家数无门控：分组机制自适应
    bool useHistCnt = (maskCnt > 0U) && (maskCnt <= 0xFFFFU);
    // counts 数组：[moeId 主序 MoE 计数 E 字（补齐到 32B 整块）][共享槽 pattern S 字]
    uint32_t moeWordsAlign = Ceil(moeExpertNum_, UB_ALIGN_DATA_COUNT) * UB_ALIGN_DATA_COUNT;
    TBuf<> ccuCountsBuf;
    tpipe_->InitBuffer(ccuCountsBuf, (moeWordsAlign + sharedExpertRankNum_) * sizeof(uint32_t));
    LocalTensor<int32_t> countsTensor = ccuCountsBuf.Get<int32_t>();
    LocalTensor<int32_t> sharedTensor =
        ccuCountsBuf.GetWithOffset<int32_t>(sharedExpertRankNum_, moeWordsAlign * sizeof(uint32_t));
    // int32 视图与 src tensor（countsTensor/sharedTensor 为 int32）保持 DataCopy 模板 T 一致；
    // GM 无类型、count 值域非负，接收侧（CcuScanAndPublishPrefix）的 uint32 视图不受影响
    GlobalTensor<int32_t> ccuSendGlobal;
    ccuSendGlobal.SetGlobalBuffer((__gm__ int32_t *)ccuSendBufGM_);

    if (useHistCnt) {
        TBuf<> ccuHistBuf;
        uint32_t idsU8Bytes = Ceil(maskCnt * sizeof(uint8_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(ccuHistBuf, idsU8Bytes + CCU_HIST_BIN_NUM * sizeof(uint16_t)); // [idsU8][hist]
        LocalTensor<uint8_t> idsU8Tensor = ccuHistBuf.Get<uint8_t>();
        LocalTensor<uint16_t> histTensor = ccuHistBuf.GetWithOffset<uint16_t>(CCU_HIST_BIN_NUM, idsU8Bytes);
        // 组直方图：每趟 ids-base 平移到 [0,254]，Mins/Maxs 双钳使组外与 -1 统一落 bin 255（不读桶）。
        // 平移中间量复用 CalCumSum 阶段 subExpBuf_/dstExpBuf_（hist 路径下逐专家归约不执行，空闲，
        // 容量按 maskCnt 已验证）；同 tensor 相邻 V op 间 PipeBarrier<PIPE_V> 对齐既有惯例
        LocalTensor<int32_t> baseTensor = dstExpBuf_.Get<int32_t>();
        LocalTensor<int32_t> shiftTensor = subExpBuf_.Get<int32_t>();
        uint32_t groupNum = (moeExpertNum_ + CCU_HIST_GROUP_SIZE - 1U) / CCU_HIST_GROUP_SIZE;
        for (uint32_t group = 0; group < groupNum; ++group) { // G ≤ Ceil(1024,255)=5，标量趟数 ≤16
            uint32_t base = group * CCU_HIST_GROUP_SIZE;
            uint32_t curGroupSize =
                (base + CCU_HIST_GROUP_SIZE > moeExpertNum_) ? (moeExpertNum_ - base) : CCU_HIST_GROUP_SIZE;
            Duplicate<int32_t>(baseTensor, static_cast<int32_t>(base), maskCnt);
            PipeBarrier<PIPE_V>();
            Sub(shiftTensor, validExpertIdsTensor_, baseTensor, maskCnt);
            PipeBarrier<PIPE_V>();
            Mins(shiftTensor, shiftTensor, 255, maskCnt);
            PipeBarrier<PIPE_V>();
            Maxs(shiftTensor, shiftTensor, -1, maskCnt);
            PipeBarrier<PIPE_V>();
            Cast(idsU8Tensor, shiftTensor, RoundMode::CAST_NONE, maskCnt);            // -1→0xFF 落 bin 255
            GetExpertFreq(histTensor, idsU8Tensor, maskCnt);                          // 内含尾块 mask 与 V 栅栏
            Cast(countsTensor[base], histTensor, RoundMode::CAST_NONE, curGroupSize); // 桶值→moeId 主序数组
        }
        SyncFunc<AscendC::HardEvent::V_MTE3>(); // counts(V 写) → 下方 strided DataCopy(MTE3 读)
    } else {
        // 退回路径：逐专家 CalExpertCnt（合法输入域 maskCnt≤6144 恒走直方图，此分支不可达，兜底保留）
        for (uint32_t moeId = 0; moeId < moeExpertNum_; ++moeId) {
            countsTensor.SetValue(moeId, CalExpertCnt(moeId + sharedExpertRankNum_, maskCnt));
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
    }

    // 共享段 word0 计数（S=sharedExpertRankNum_，MAX_SHARED_EXPERT_NUM=4 结构性小量）：命中槽 =
    // activeMaskBsCnt_、其余 0（与 CalExpertCnt 共享分支同义）；一条 strided 拷贝写各共享段 word0。
    // 单位契约：DataCopy(DataCopyParams) 的 blockLen/stride 单位是 32B datablock（基线先例 expFlagCopyParams
    // 以 SPLIT_BLOCK_DATA_SIZE/UB_ALIGN 表达 480B 间隔），无法表达"每段 1 字、段间隔 segWords*4B"——共享段/
    // MoE 段两处统一改 DataCopyExtParams 字节单位。重载形态：UB→GM 的 DataCopyPad 仅 3 参（框架自动
    // dummy-pad 且 GM 侧丢弃，基线 sendCounts/tokenNums/expandXOut 输出同款）；第 4 参仅 GM→UB 存在
    if (sharedExpertRankNum_ != 0U) {
        for (uint32_t expertId = 0; expertId < sharedExpertRankNum_; ++expertId) {
            sharedTensor.SetValue(expertId, CalExpertCnt(expertId, maskCnt));
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams ccuSharedParams{static_cast<uint16_t>(sharedExpertRankNum_), sizeof(int32_t), 0U,
                                          static_cast<uint32_t>((ccuSegWords_ - 1U) * sizeof(int32_t)), 0U};
        DataCopyPad(ccuSendGlobal[0], sharedTensor, ccuSharedParams);
    }
    // MoE 区整写：src 为 moeId 主序连续块（blockLen=Pe*4B、srcStride=0 紧邻取块），dst 段起点间隔 segWords*4B
    DataCopyExtParams ccuMoeParams{static_cast<uint16_t>(moeExpertRankNum_),
                                   static_cast<uint32_t>(moeExpertNumPerRank_ * sizeof(int32_t)), 0U,
                                   static_cast<uint32_t>((ccuSegWords_ - moeExpertNumPerRank_) * sizeof(int32_t)), 0U};
    DataCopyPad(ccuSendGlobal[sharedExpertRankNum_ * ccuSegWords_], countsTensor, ccuMoeParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(); // 本核全部段写完成且全局可见后，方可 issue（单核无跨核同步）
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalCntPackAndIssueCcu()
{
    // CCU 模式 aivUsedCumSum_==1：唯一 cumsum 核串行承担 count 面（与前台各核 token 发送并行）——
    // 计数落段 → 等长 AlltoAll + Wait → 全表前缀 scan 与统计输出 + 就绪字广播（scan 尾部单批合出）
    CcuCalCountsToSegBuf();
    // 等长 AlltoAll：每卡发给各对端 ccuSegWords_ 个 u32（纯 counts 段），Wait 后矩阵连续落 recvBuf
    ccuCntHandle_ =
        ccuHccl_.template AlltoAll<true>(ccuSendBufGM_, ccuRecvBufGM_, ccuSegWords_, AscendC::HCCL_DATA_TYPE_INT32, 0);
    ccuHccl_.Wait(ccuCntHandle_); // Wait 语义即"所有源的 count 已到"，替代原 WaitDispatch 轮询
    RunPosRecord(RUNPOS_CALCUMSUM); // 维测打点（对齐 MTE WaitDispatch 的 count 到齐点，仅 cumsum 核记录）
    if (useCcuScanGm_) {
        CcuScanAndPublishPrefixGm(); // 直写 GM 版（A/B 备选，env MOE_DISPATCH_CCU_SCAN_GM=1）
    } else {
        CcuScanAndPublishPrefix(); // UB 中转版（默认）
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CcuScanAndPublishPrefix()
{
    // 中心化前缀（本核不发 token、只承担 count 面，且 Wait 返回点最早拿到矩阵）：就地完成全表 scan，
    // 前缀表写回 recvBuf 当前半区（原地覆写矩阵——全量读回 UB 后才写；半区归当前 launch，消费即死），
    // 直接产出 sendCounts（全量）与 expertTokenNums，并广播前缀表就绪字（尾部单 MTE3 批次合出）——
    // 其余核随后只拷各自分核段切片，免去全量矩阵搬运 + 各自 scan（MTE 流量 ÷核数，切片互不相交无 incast）
    // UB 分段：计数段成员 buffer 已全部消费，Reset 释放（最坏 ~168KB）后 scan 段独立分配（≤74KB，含
    // tokenNums/就绪字缓冲），两段峰值分别 ≤179KB/≤74KB < 190KB（否则 W=1024 档 2×32KB 矩阵 + 成员区超限）
    tpipe_->Reset();
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1U : moeExpertNumPerRank_;
    uint32_t matWords = epWorldSize_ * ccuSegWords_;
    TBuf<> ccuMatBuf, ccuOutBuf, ccuScanBuf, ccuTokenNumBuf, ccuReadyBuf;
    tpipe_->InitBuffer(ccuMatBuf, matWords * sizeof(uint32_t));
    tpipe_->InitBuffer(ccuOutBuf, matWords * sizeof(uint32_t)); // segWords 按 8 字对齐，32B 整块可搬
    tpipe_->InitBuffer(ccuScanBuf, CCU_SIMT_THREAD_NUM * sizeof(uint32_t));
    tpipe_->InitBuffer(ccuTokenNumBuf, Ceil(localExpertNum * sizeof(int64_t), UB_ALIGN) * UB_ALIGN);
    tpipe_->InitBuffer(ccuReadyBuf, Ceil(aivNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN); // 就绪字广播 src
    LocalTensor<uint32_t> matTensor = ccuMatBuf.Get<uint32_t>();
    LocalTensor<uint32_t> outTensor = ccuOutBuf.Get<uint32_t>();
    LocalTensor<uint32_t> chunkOffsets = ccuScanBuf.Get<uint32_t>();
    LocalTensor<int64_t> tokenNumsTensor = ccuTokenNumBuf.Get<int64_t>();
    LocalTensor<float> ccuReadyTensor = ccuReadyBuf.Get<float>();
    GlobalTensor<uint32_t> ccuRecvGlobal;
    ccuRecvGlobal.SetGlobalBuffer((__gm__ uint32_t *)ccuRecvBufGM_);
    // blockLen/stride 单位=32B datablock（segWords 8 字对齐 ⇒ matWords 恰整块，基址 32B 对齐）
    DataCopyParams ccuMatParams{1U, static_cast<uint16_t>(matWords / UB_ALIGN_DATA_COUNT), 0U, 0U};
    DataCopy(matTensor, ccuRecvGlobal, ccuMatParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    uint32_t totalNum = localExpertNum * epWorldSize_;
    asc_vf_call<CcuPrefixScanSimt<uint32_t>>(dim3(CCU_SIMT_THREAD_NUM), (__ubuf__ uint32_t *)matTensor.GetPhyAddr(),
                                             (__ubuf__ uint32_t *)outTensor.GetPhyAddr(),
                                             (__ubuf__ uint32_t *)chunkOffsets.GetPhyAddr(),
                                             (__ubuf__ int64_t *)tokenNumsTensor.GetPhyAddr(), totalNum, epWorldSize_,
                                             ccuSegWords_, localExpertNum, (expertTokenNumsType_ != 0U) ? 1U : 0U);
    // 就绪字 1.0f 缓冲随 scan 顺势产出（V 已热；与 SIMT 输出共享同一次 V_MTE3）——A5 标量弱，广播收敛为
    // 一条 strided DataCopy（免 aivNum_ 次 issue 循环）：src 为 aivNum_ 份 1.0f 紧凑缓冲，dst 按各核 512B
    // 状态槽跨步写 word16（槽内 64B 偏移保 32B 对齐，每核 1 字、各占一行，单写单读）
    Duplicate<float>(ccuReadyTensor, float(1), aivNum_);
    SyncFunc<AscendC::HardEvent::V_MTE3>(); // scan(SIMT) + 就绪字缓冲(V) → 下方 MTE3 读

    // 单一 MTE3 批次（4 条，pipe 内按 issue 序 FIFO）：前缀表写回 recvBuf → sendCounts 全量输出 →
    // expertTokenNums 输出 → 就绪字广播；末尾唯一一次 MTE3_S 排空后，就绪字才可能全局可见——消费者自旋
    // 命中即前缀表与两统计输出均已落盘（合批省一次 MTE3 排空 + V↔MTE3 切换往返，全核关键路径 ~1μs）
    DataCopy(ccuRecvGlobal, outTensor, ccuMatParams);
    GlobalTensor<int32_t> sendCountsGlobal;
    sendCountsGlobal.SetGlobalBuffer((__gm__ int32_t *)sendCountsOutGM_);
    DataCopyExtParams ccuCntOutParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(sendCountsGlobal, outTensor.ReinterpretCast<int32_t>(), ccuCntOutParams);
    // expertTokenNums：scan 阶段四已就地产出（SIMT 直写 int64），此处仅输出
    DataCopyExtParams tokenNumsOutParams{1U, static_cast<uint32_t>(localExpertNum * sizeof(int64_t)), 0U, 0U, 0U};
    DataCopyPad(expertTokenNumsOutGMTensor_, tokenNumsTensor, tokenNumsOutParams);
    // 字节单位（DataCopyExtParams）：每核 1 字 4B、槽间隔 512B-4B——32B 块单位无法表达，改 strided Pad
    // （UB→GM 仅 3 参重载，框架自动 dummy-pad）
    DataCopyExtParams ccuReadyParams{static_cast<uint16_t>(aivNum_), sizeof(float), 0U,
                                     static_cast<uint32_t>(WIN_ADDR_ALIGN - sizeof(float)), 0U};
    DataCopyPad(GetAivStatusFlagTensor(0)[CCU_READY_WORD_POS], ccuReadyTensor, ccuReadyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CcuScanAndPublishPrefixGm()
{
    // 直写 GM 版（A/B 备选，useCcuScanGm_ 门控）：scan 全程不进 UB——无 MTE2 矩阵搬入/MTE3 写回与统计
    // DataCopyPad，scan 段 UB 仅剩 chunkOffsets(1KB)+就绪字缓冲；就绪字协议：阻塞 asc_vf_call（程序序
    // 保证 SIMT 写完成）→ Duplicate → V_MTE3 → 就绪字 MTE3 写（唯一一条 MTE3，MTE3_S 收尾）。
    // 消费侧协议与 UB 版一致（recvBuf 前缀表 + word16 就绪字自旋）；跨核可见性依赖 L2（volatile 直写）。
    tpipe_->Reset();
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1U : moeExpertNumPerRank_;
    TBuf<> ccuScanBuf, ccuReadyBuf;
    tpipe_->InitBuffer(ccuScanBuf, CCU_SIMT_THREAD_NUM * sizeof(uint32_t));
    tpipe_->InitBuffer(ccuReadyBuf, Ceil(aivNum_ * sizeof(float), UB_ALIGN) * UB_ALIGN);
    LocalTensor<uint32_t> chunkOffsets = ccuScanBuf.Get<uint32_t>();
    LocalTensor<float> ccuReadyTensor = ccuReadyBuf.Get<float>();
    uint32_t totalNum = localExpertNum * epWorldSize_;
    asc_vf_call<CcuPrefixScanSimtGm<uint32_t>>(dim3(CCU_SIMT_THREAD_NUM), (__gm__ volatile uint32_t *)ccuRecvBufGM_,
                                               (__gm__ volatile uint32_t *)ccuRecvBufGM_,
                                               (__ubuf__ uint32_t *)chunkOffsets.GetPhyAddr(),
                                               (__gm__ volatile int64_t *)expertTokenNumsOutGMTensor_.GetPhyAddr(),
                                               (__gm__ volatile int32_t *)sendCountsOutGM_, totalNum, epWorldSize_,
                                               ccuSegWords_, localExpertNum, (expertTokenNumsType_ != 0U) ? 1U : 0U);
    // 就绪字广播（与 UB 版 CcuScanAndPublishPrefix 尾部同款：strided DataCopy 扇出 word16）
    Duplicate<float>(ccuReadyTensor, float(1), aivNum_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    // 字节单位（DataCopyExtParams）：每核 1 字 4B、槽间隔 512B-4B——32B 块单位无法表达，改 strided Pad
    // （UB→GM 仅 3 参重载，框架自动 dummy-pad）
    DataCopyExtParams ccuReadyParams{static_cast<uint16_t>(aivNum_), sizeof(float), 0U,
                                     static_cast<uint32_t>(WIN_ADDR_ALIGN - sizeof(float)), 0U};
    DataCopyPad(GetAivStatusFlagTensor(0)[CCU_READY_WORD_POS], ccuReadyTensor, ccuReadyParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::BuildSendCntFromCcuMatrix()
{
    // CCU 控制面：唯一 cumsum 核已把全表前缀写回 recvBuf 当前半区并置就绪字（本函数之前的
    // CcuWaitMatrixReady 已自旋确认）——本核只拷自己分核段切片（各核切片互不相交，无 incast），
    // 替代原各核全量矩阵搬运 + 各自 SIMT scan；sendCounts/expertTokenNums 输出已由 cumsum 核完成。
    // 排布说明：前缀表第 i 字 = 含端前缀（expert 主序 × src 次序），i 即原状态槽序。
    GlobalTensor<int32_t> prefixGlobal;
    prefixGlobal.SetGlobalBuffer((__gm__ int32_t *)ccuRecvBufGM_);
    // 消费侧（expertMap/expertLeft 差分、WaitAndFormatOutput 的 dstPosition）按【绝对槽序】读写
    // sendCntTensor_，且切片首槽差分需读前一字——拷贝保持绝对落位；DataCopy 基址须 32B 对齐（契约，
    // MTE 路径全表拷贝基址天然对齐），故切片两端向 8 字取整：sliceStart 向下取整块、sliceEnd 向上取整
    // （≤ align8up(rscvStatusNum_) ≤ matWords 且 ≤ cumSumBuf 容量 rscvNumAlign；尾块多拷字无读者）
    uint32_t sliceFirst = (startId_ > 0U) ? (startId_ - 1U) : 0U;
    uint32_t sliceStart = sliceFirst / UB_ALIGN_DATA_COUNT * UB_ALIGN_DATA_COUNT; // 基址 32B 对齐
    uint32_t sliceEnd = Ceil(endId_, UB_ALIGN_DATA_COUNT) * UB_ALIGN_DATA_COUNT;  // 尾块对齐
    uint32_t sliceWords = sliceEnd - sliceStart;
    // blockLen 单位=32B datablock：sliceWords 两端 8 字取整（见上）恰为整块数
    DataCopyParams ccuSliceParams{1U, static_cast<uint16_t>(sliceWords / UB_ALIGN_DATA_COUNT), 0U, 0U};
    DataCopy(sendCntTensor_[sliceStart], prefixGlobal[sliceStart], ccuSliceParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalAndSendCntByRankAndExpert()
{
    uint32_t startExpertId, endExpertId, sendExpertNum;
    // 将(rank, localExpert)展平为expertId后分核，使同一rank可由多个核并行处理。
    SplitToCore(totalExpertNum_, aivUsedCumSum_, startExpertId, endExpertId, sendExpertNum, false);
    if (sendExpertNum == 0) {
        return;
    }

    uint32_t maskCnt = isTokenMaskFlag_ ? (activeMaskBsCnt_ * axisK_) : expertIdsCnt_;
    // 一次性操作256字节，也是64个int32_t，每8个数将首个设置为0x3F800000
    uint64_t mask[2] = {0x101010101010101, 0};
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusCntAlign_ / 8, 1,
                       8); // 0x3F800000为float的1 8为一次操作8个block
    SyncFunc<AscendC::HardEvent::V_S>();

    for (uint32_t expertId = startExpertId; expertId < endExpertId; ++expertId) {
        int32_t expertCnt = CalExpertCnt(expertId, maskCnt);
        // 一个block有8个int32的元素，第一个元素为flag位，第二个为发送token数
        int32_t cntPosIndex = expertId * UB_ALIGN_DATA_COUNT + 1;
        statusTensor_.SetValue(cntPosIndex, expertCnt);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SendExpertCntRange(startExpertId, endExpertId);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::BufferInit()
{
    uint32_t waitStatusBufSize = Ceil((recStatusNumPerCore_ * UB_ALIGN), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    uint64_t recvWinBlockNumSpace = epWorldSize_ * moeExpertNumPerRank_ * sizeof(float);
    uint64_t gatherMaskOutSize =
        (recStatusNumPerCoreSpace > recvWinBlockNumSpace) ? recStatusNumPerCoreSpace : recvWinBlockNumSpace;
    uint64_t sumContinueAlignSize = Ceil((aivNum_ * sizeof(float)), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, gatherMaskOutSize);  // recStatusNumPerCore_32对齐后大小  * 32B
    tpipe_->InitBuffer(sumCoreBuf_, aivNum_ * UB_ALIGN);       // 48 * 32B
    tpipe_->InitBuffer(sumLocalBuf_, aivNum_ * UB_ALIGN);      // 48 * 32B
    tpipe_->InitBuffer(sumContinueBuf_, sumContinueAlignSize); // 48 * 4B
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 3);              // 96 B
    if (isPerformanceFlag_) {
        uint32_t performanceFlagSize = recStatusNumPerCore_ * sizeof(int32_t);
        uint32_t performanceFlagSizeAlign = Ceil(performanceFlagSize, UB_ALIGN) * UB_ALIGN;
        uint32_t performanceInfoSize = epWorldSizeOriginal_ * sizeof(int64_t);
        uint32_t performanceInfoSizeAlign = Ceil(performanceInfoSize, UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(performanceInfoBuf_, performanceInfoSizeAlign);
        performanceInfoTensor_ = performanceInfoBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceInfoTensor_, 0, performanceInfoSizeAlign / sizeof(int32_t));
        tpipe_->InitBuffer(performanceFlagBuf_, performanceFlagSizeAlign);
        performanceFlagTensor_ = performanceFlagBuf_.Get<int32_t>();
        Duplicate<int32_t>(performanceFlagTensor_, 0, performanceFlagSizeAlign / sizeof(int32_t));
        if (isScalingDownFlag_) {
            uint32_t elasticInfoSize = (ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t);
            uint32_t elasticInfoSizeAlign = Ceil(elasticInfoSize, UB_ALIGN) * UB_ALIGN;
            tpipe_->InitBuffer(elasticInfoBuf_, elasticInfoSizeAlign);
            elasticInfoTensor_ = elasticInfoBuf_.Get<int32_t>();
            DataCopyExtParams elasticInfoParams = {
                1U,
                static_cast<uint32_t>((ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSizeOriginal_) * sizeof(int32_t)),
                0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> elasticInfoCopyPadParams{false, 0U, 0U, 0U};
            DataCopyPad(elasticInfoTensor_, elasticInfoGMTensor_, elasticInfoParams, elasticInfoCopyPadParams);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
        }
    }
    uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::WaitDispatchClearStatus()
{
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0}; // 一次操作256字节，每8个数将首个设置为0
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8); // 8 = 256 / 32
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::GatherSumRecvCnt(
    LocalTensor<float> &gatherMaskOutTensor, LocalTensor<uint32_t> &gatherTmpTensor,
    LocalTensor<float> &statusSumOutTensor)
{
    gatherTmpTensor.SetValue(0, 2); // 源操作数每个datablock取下标为1的元素
    uint32_t mask = 2;              // 源操作数每个datablock只需要处理两个元素
    SyncFunc<AscendC::HardEvent::S_V>();

    // 将当前核对应的专家 recvCnt 收集到gatherMaskOutTensor
    uint64_t recvCnt = 0;
    GatherMask(gatherMaskOutTensor, statusFp32Tensor_, gatherTmpTensor, true, mask,
               {1, (uint16_t)recStatusNumPerCore_, 1, 0}, recvCnt);
    PipeBarrier<PIPE_V>();

    // 对当前核对应的专家recv cnt求和，对inner要求32对齐
    uint32_t recStatusNumPerCoreInner = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, recStatusNumPerCoreInner, recStatusNumPerCore_};
    Sum(statusSumOutTensor, gatherMaskOutTensor, sumParams);
    SyncFunc<AscendC::HardEvent::V_S>();
    float sumOfRecvCnt = statusSumOutTensor.ReinterpretCast<float>().GetValue(0);

    // 把当前核的所有专家recv cnt之和写到状态区
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    // 每个核把sumOfRecvCnt重复写 aivUsedCumSum_ 份
    LocalTensor<float> sumCoreFP32Tensor = sumCoreBuf_.Get<float>();
    uint64_t maskArrayCount[2] = {0x0101010101010101, 0};
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8); // 8 = 256 / 32
    // 每次处理256字节，8个datablock，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFP32Tensor, sumOfRecvCnt, maskArrayCount, repeatTimes, 1, 8);
    uint64_t maskArrayFlag[2] = {0x0202020202020202, 0};
    Duplicate<float>(sumCoreFP32Tensor, static_cast<float>(1.0), maskArrayFlag, repeatTimes, 1, 8);
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(cumsumWsGMTensor_[(newAivId * aivUsedCumSum_ * UB_ALIGN) / sizeof(float)], sumCoreFP32Tensor,
             sumIntriParams);
    SyncFunc<AscendC::HardEvent::MTE3_V>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::GetCumSum(
    LocalTensor<int32_t> &outLocal, uint32_t newAivId)
{
    outLocal = gatherMaskOutBuf_.Get<int32_t>();
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivUsedCumSum_), 1, static_cast<uint16_t>(aivUsedCumSum_ - 1),
                                  0};
    LocalTensor<float> sumLocalTensor = sumLocalBuf_.Get<float>();
    LocalTensor<uint32_t> gatherSumPattern = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> sumContinueTensor = sumContinueBuf_.Get<float>();
    LocalTensor<float> recvCntSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);

    uint32_t mask = 2;
    uint64_t recvCnt = 0;
    uint32_t innerSumParams = Ceil(aivUsedCumSum_ * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
    SumParams sumParams{1, innerSumParams, aivUsedCumSum_};
    int32_t cumSumFlag = 0;
    gatherSumPattern.SetValue(0, 2);
    SyncFunc<AscendC::HardEvent::S_V>();

    // 获取状态区中每个核的recvCnt
    while (true) {
        DataCopy(sumLocalTensor, cumsumWsGMTensor_[(newAivId * UB_ALIGN) / sizeof(float)], sumIntriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask,
                   {1, static_cast<uint16_t>(aivUsedCumSum_), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        Sum(recvCntSumOutTensor, sumContinueTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = static_cast<int32_t>(recvCntSumOutTensor.GetValue(0));
        if (cumSumFlag == aivUsedCumSum_) {
            break;
        }
    }

    // 0核前面所有核recv cnt总和是0
    if (newAivId == 0) {
        outLocal.SetValue(0, 0);
    } else {
        mask = 1;
        recvCnt = 0;
        gatherSumPattern.SetValue(0, 1);
        SyncFunc<AscendC::HardEvent::S_V>();
        GatherMask(sumContinueTensor, sumLocalTensor, gatherSumPattern, true, mask,
                   {1, static_cast<uint16_t>(newAivId), 1, 0}, recvCnt);
        PipeBarrier<PIPE_V>();
        uint32_t innerCumSumParams = Ceil(newAivId * sizeof(float), UB_ALIGN) * UB_ALIGN / sizeof(float);
        SumParams cumSumParams{1, innerCumSumParams, newAivId};
        Sum(recvCntSumOutTensor, sumContinueTensor, cumSumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        outLocal.SetValue(0, recvCntSumOutTensor.ReinterpretCast<int32_t>().GetValue(0));
    }
    // 清除 flag 用于下次aivUsedCumSum_软同步
    LocalTensor<float> sumCoreFp32Tensor = sumLocalBuf_.Get<float>();
    // 一次处理256字节，8个datablock
    uint8_t repeatTimes = Ceil(aivUsedCumSum_, 8);
    // 64 = 256 / sizeof(float) 一次操作字节数，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<float>(sumCoreFp32Tensor, static_cast<float>(0), 64, repeatTimes, 1, 8);
    DataCopyParams cleanParams{static_cast<uint16_t>(aivUsedCumSum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(cumsumWsGMTensor_[(newAivId * UB_ALIGN) / sizeof(float)], sumCoreFp32Tensor, cleanParams);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::WaitDispatch()
{
    LocalTensor<float> gatherMaskOutTensor = gatherMaskOutBuf_.Get<float>();
    LocalTensor<uint32_t> gatherTmpTensor = scalarBuf_.GetWithOffset<uint32_t>(UB_ALIGN / sizeof(uint32_t), 0);
    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    uint32_t mask = 1;
    gatherTmpTensor.SetValue(0, 1);
    float compareTarget = static_cast<float>(1.0) * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    uint64_t performanceTimeStart = static_cast<uint64_t>(GetSystemCycle());
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInstatusFp32Tensor_[startStatusIndex_ * STATE_OFFSET / sizeof(float)],
                 intriParams);
        if (isPerformanceFlag_) {
            RecordRankCommDuration(performanceInfoTensor_, performanceTimeStart);
        }
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutTensor, mask, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }
    RunPosRecord(RUNPOS_CALCUMSUM); // 维测打点
    if (isPerformanceFlag_) {
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        SetAtomicMax<int32_t>();
        DataCopyExtParams performanceInfoCopyParams{1U, static_cast<uint32_t>(epWorldSizeOriginal_ * sizeof(int64_t)),
                                                    0U, 0U, 0U};
        DataCopyPad(performanceInfoGMTensor_, performanceInfoTensor_, performanceInfoCopyParams);
        SetAtomicNone();
    }
    // 清状态
    WaitDispatchClearStatus();
    GatherSumRecvCnt(gatherMaskOutTensor, gatherTmpTensor, statusSumOutTensor);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalRecvAndSetFlag()
{
    // check flag 用于 aivUsedCumSum_ 软同步并计算 aivUsedCumSum_ 个核各自的recvCount
    LocalTensor<int32_t> outCountLocal;
    uint32_t newAivId = aivId_ - aivUsedAllToAll_;
    GetCumSum(outCountLocal, newAivId);
    // 计算epRecvCnt
    uint32_t preSum = outCountLocal.GetValue(0);
    uint32_t curCnt = preSum;
    waitStatusTensor_ = waitStatusBuf_.Get<int32_t>();
    for (uint32_t index = startStatusIndex_; index < endStatusIndex_; index++) {
        uint32_t i = index - startStatusIndex_;
        uint32_t count = waitStatusTensor_.GetValue(i * UB_ALIGN_DATA_COUNT + 1);
        curCnt += count;
        outCountLocal.SetValue(i, curCnt);
    }
    SyncFunc<AscendC::HardEvent::S_V>();
    GM_ADDR wAddr = (__gm__ uint8_t *)(recvCntWorkspaceGM_);
    GlobalTensor<int32_t> sendCountsGlobal, workspaceGlobal;
    sendCountsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sendCountsOutGM_));
    workspaceGlobal.SetGlobalBuffer((__gm__ int32_t *)wAddr);
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(recStatusNumPerCore_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPad(sendCountsGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams);
    // 复制aivNum_份
    for (uint32_t index = 0; index < aivNum_; index++) {
        DataCopyPad(workspaceGlobal[index * rscvStatusNum_ + startStatusIndex_], outCountLocal, dataCopyOutParams);
    }
    uint8_t repeatTimes = Ceil(aivNum_, 8); // 一次处理256字节，8个datablock
    DataCopyParams sumIntriParams{static_cast<uint16_t>(aivNum_), 1, 0, static_cast<uint16_t>(aivUsedCumSum_ - 1)};
    LocalTensor<int32_t> syncOnCoreTensor = sumCoreBuf_.Get<int32_t>();
    LocalTensor<float> syncOnCoreFP32Tensor = sumCoreBuf_.Get<float>();
    // 每次处理256字节，1、8分别为dst、src相邻迭代间地址步长
    Duplicate<int32_t>(syncOnCoreTensor, static_cast<int32_t>(1), SIZE_ALIGN_256 / sizeof(int32_t), repeatTimes, 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    PipeBarrier<PIPE_MTE3>();
    DataCopy(cumsumWsGMTensor_[(CUMSUM_WS_FLAG_OFFSET + newAivId * UB_ALIGN) / sizeof(float)], syncOnCoreFP32Tensor,
             sumIntriParams); // 软同步
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::RecordRankCommDuration(
    LocalTensor<int32_t> &performanceInfoTensor, uint64_t startTime)
{
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    uint64_t endTime = static_cast<uint64_t>(GetSystemCycle());
    int32_t duration = static_cast<int32_t>((endTime - startTime) / CYCLES_PER_US);
    for (uint32_t i = 0; i < recStatusNumPerCore_; i++) {
        float statusFp32 = statusFp32Tensor_.GetValue(i * FLAG_OFFSET);
        int32_t performanceFlag = performanceFlagTensor_.GetValue(i);
        if (statusFp32 > float(0.5) && performanceFlag == 0) {
            performanceFlagTensor_.SetValue(i, 1);
            uint32_t fromLocalRankId = (startStatusIndex_ + i) % epWorldSize_;
            uint32_t fromRankId =
                isScalingDownFlag_ ?
                    elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + fromLocalRankId) :
                    fromLocalRankId;
            int32_t savedTime = performanceInfoTensor.GetValue(fromRankId * DURATION_OFFSET);
            int32_t newValue = (duration > savedTime) ? duration : savedTime;
            if (newValue != savedTime) {
                // 乘2 使用int32_t是因为atomicAdd不支持int64_t类型，这里只赋值到int64_t的低32位。
                performanceInfoTensor.SetValue(fromRankId * DURATION_OFFSET, duration);
            }
        }
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalCumSum()
{
    // 当前核计算并发送所负责专家的token总数
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    tpipe_->InitBuffer(dstExpBuf_, maxSize_);                                                   // BS * K * 4
    tpipe_->InitBuffer(subExpBuf_, maxSize_);                                                   // BS * K * 4
    tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);                                     // BS * K * 4
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    // statusBuf_（最坏 32KB）仅 MTE count 面消费（SendExpertCntRange/BufferInit 均在 MTE 分支内）：
    // CCU 模式门控掉，计数段 UB 余量从 ~3KB 放大到 ~35KB（最坏 2×maxSize_+2×ids+scales+mask链+hist/counts）
    if (!useCcuCtrl_) {
        tpipe_->InitBuffer(statusBuf_, statusCntAlign_ * UB_ALIGN);
        statusTensor_ = statusBuf_.Get<int32_t>();
    }
    if (hasExpertScalesFlag_) {
        tpipe_->InitBuffer(expertScalesBuf_, expertIdsBufSize_);
    }
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();
    ExpIdsCopyAndMaskCal();
    if (useCcuCtrl_) {
        // CCU 控制面：count 面收敛单核（aivUsedCumSum_==1），组包后一次 AlltoAll 下发并 Wait；接收侧由
        // 各核在 LocalWindowCopy 阶段自读矩阵，原 WaitDispatch/CalRecvAndSetFlag 的多 flag 轮询与
        // workspace 复制链跳过（就绪扇出改由唯一 cumsum 核 Wait 后就绪字广播 + 各核 CcuWaitMatrixReady 自旋）
        CalCntPackAndIssueCcu();
        return;
    }
    if (hasElasticInfoFlag_) {
        CalAndSendCntByExp();
    } else {
        CalAndSendCntByRankAndExpert();
    }
    SplitToCore(rscvStatusNum_, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_, false);
    BufferInit();
    WaitDispatch();
    CalRecvAndSetFlag();
    // 使用newAivId为0的核进行计算
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::WaitCumSumFlag()
{
    // Check cumsum is finished
    int32_t cumSumFlag = 0;
    int32_t targetFlag = aivUsedCumSum_ * UB_ALIGN_DATA_COUNT;
    uint32_t cumSumFlagOffset = (CUMSUM_WS_FLAG_OFFSET + aivId_ * aivUsedCumSum_ * UB_ALIGN) / sizeof(float);
    uint32_t innerSumParams = aivUsedCumSum_ * UB_ALIGN / sizeof(float);
    SumParams sumFlagParams{1, innerSumParams, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT};
    LocalTensor<float> statusSumOutTensor = scalarBuf_.Get<float>();

    while (true) {
        DataCopy(statusFp32Tensor_, cumsumWsGMTensor_[cumSumFlagOffset], aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusSumOutTensor, statusFp32Tensor_, sumFlagParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        cumSumFlag = statusSumOutTensor.ReinterpretCast<int32_t>().GetValue(0);
        if (cumSumFlag == targetFlag) {
            break;
        }
    }
    RunPosRecord(RUNPOS_CUMSUMFLAG); // 维测打点
    // Clean flag for next round
    Duplicate<float>(statusCleanFp32Tensor_, static_cast<float>(0), aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(cumsumWsGMTensor_[cumSumFlagOffset], statusCleanFp32Tensor_, aivUsedCumSum_ * UB_ALIGN_DATA_COUNT);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CcuWaitMatrixReady()
{
    // CCU 控制面就绪等待：唯一 cumsum 核（aivNum_-1，CCU 控制核）在 Wait 返回、完成全表 scan 并把前缀表
    // 写回 recvBuf 及 sendCounts/expertTokenNums 输出后，向各核状态字 word16 写 1.0f，本核自旋自己槽内
    // 该字——各核独占一行（控制核写、本核读后清），无共享 cacheline；命中即前缀表已落 recvBuf 且全局可见。
    // 标志在状态区（启动清零、跨 launch 持久）而非 workspace（首轮脏值可能假命中）。
    // PipeBarrier<PIPE_ALL> 仅排空本核管道、不跨核，故不能用 barrier 替代本自旋。
    GlobalTensor<float> readyWordTensor = GetAivStatusFlagTensor(aivId_);
    LocalTensor<float> readyTensor = scalarBuf_.Get<float>();
    DataCopyExtParams ccuReadyParams{1U, sizeof(float), 0U, 0U, 0U};
    DataCopyPadExtParams<float> ccuReadyPadParams{false, 0U, 0U, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    while (true) {
        DataCopyPad(readyTensor, readyWordTensor[CCU_READY_WORD_POS], ccuReadyParams, ccuReadyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        if (readyTensor.GetValue(0) == float(1)) {
            break;
        }
    }
    RunPosRecord(RUNPOS_CUMSUMFLAG); // 维测打点（对齐 MTE WaitCumSumFlag 的软同步完成点，全核记录）
    // 用后清零（防下一轮误判；仅本核读写自己字，清零无竞争）
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<float>(readyTensor, float(0), 1U);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(readyWordTensor[CCU_READY_WORD_POS], readyTensor, ccuReadyParams); // UB→GM 3 参
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::SetValidExpertInfo(
    uint32_t expInfoSize, uint32_t &validNum)
{
    // 获取cumSum
    if (useCcuCtrl_) {
        // CCU 控制面：各核只拷自己分核段的前缀表切片（前缀表由唯一 cumsum 核全表 scan 后写回 recvBuf）
        BuildSendCntFromCcuMatrix();
    } else {
        GlobalTensor<int32_t> workspaceGlobal;
        workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(recvCntWorkspaceGM_));
        DataCopyExtParams scalesCopyInParams{1U, static_cast<uint32_t>(rscvStatusNum_ * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> copyPadExtParams{false, 0U, 0U, 0U};
        DataCopyPad(sendCntTensor_, workspaceGlobal[aivId_ * rscvStatusNum_], scalesCopyInParams, copyPadExtParams);
    }
    PipeBarrier<PIPE_ALL>();

    // MTE 路径：expertTokenNums 由 aiv0 标量产出（CCU 路径已迁至 cumsum 核，scan 后向量化产出）
    if (!useCcuCtrl_ && (aivId_ == 0)) {
        uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
        int64_t lastVal = 0;
        uint32_t tokenNumBufSize = Ceil(moeExpertNumPerRank_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
        tpipe_->InitBuffer(tokenNumBuf_, tokenNumBufSize);
        LocalTensor<int64_t> expertTokenNumsLocalTensor = tokenNumBuf_.Get<int64_t>();
        for (uint32_t localExpertIdx = 0; localExpertIdx < localExpertNum; ++localExpertIdx) {
            if (expertTokenNumsType_ == 0) {
                expertTokenNumsLocalTensor(localExpertIdx) =
                    int64_t(sendCntTensor_(localExpertIdx * epWorldSize_ + epWorldSize_ - 1));
            } else {
                expertTokenNumsLocalTensor(localExpertIdx) =
                    int64_t(sendCntTensor_(localExpertIdx * epWorldSize_ + epWorldSize_ - 1)) - lastVal;
                lastVal = int64_t(sendCntTensor_(localExpertIdx * epWorldSize_ + epWorldSize_ - 1));
            }
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyExtParams expertTokenNumsCopyParams{1U, static_cast<uint32_t>(localExpertNum * sizeof(int64_t)), 0U, 0U,
                                                    0U};
        DataCopyPad(expertTokenNumsOutGMTensor_, expertTokenNumsLocalTensor, expertTokenNumsCopyParams);
    }

    Duplicate<uint32_t>(expertFinishNumTensor_, 0, expInfoSize / sizeof(uint32_t));
    // 从sendCnt中挑选当前有发送过来的卡的token数量
    for (uint32_t index = startId_; index < endId_; index++) {
        expertMapTensor_(validNum) = index;
        if (index == 0) {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index);
        } else {
            expertLeftNumTensor_(validNum) = sendCntTensor_(index) - sendCntTensor_(index - 1);
        }
        if (expertLeftNumTensor_(validNum) != 0) {
            validNum += 1;
        }
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CheckDataArriveWithFlag(
    uint32_t srcExpDataIdx, int32_t beginIdx, int32_t copyCnt)
{
    uint32_t flagNum = blockCntPerToken_ * uint32_t(copyCnt);
    DataCopyParams expFlagCopyParams{static_cast<uint16_t>(flagNum), 1U,
                                     static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE / UB_ALIGN), 0U};
    GlobalTensor<float> dataFlagGlobal;
    GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + srcExpDataIdx * expertPerSizeOnWin_ + beginIdx * hCommuSize_ +
                    SPLIT_BLOCK_DATA_SIZE;
    dataFlagGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));
    DataCopy(flagRecvTensor_, dataFlagGlobal, expFlagCopyParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<float> flagSumOutTensor = scalarBuf_.Get<float>();
    ReduceSum(flagSumOutTensor, flagRecvTensor_, flagReduceWorkTensor_, 1U, flagNum, 1U);
    SyncFunc<AscendC::HardEvent::V_S>();
    // aicore禁止uint->float的显式转换，借助浮点乘法隐式转换得到期望和
    float expectFlagSum = 1.0F * flagNum;
    return flagSumOutTensor.GetValue(0) == expectFlagSum ? static_cast<uint32_t>(copyCnt) : 0U;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CopyInAndOut(
    LocalTensor<XOutType> xTmpTensor, GM_ADDR wAddr, uint32_t index, uint32_t dstPosition, uint32_t arriveCount,
    uint32_t dataBufferId, bool isDataBufferReused)
{
    uint32_t hOutElemCount = hOutSize_ / sizeof(XOutType); // expandXOutGlobal申请每个token的GM Buffer空间大小
    GlobalTensor<XOutType> dataFlagGlobal, expandXOutGlobal;
    GlobalTensor<float> expertScaleInGlobal, expandScalesOutGlobal;
    dataFlagGlobal.SetGlobalBuffer((__gm__ XOutType *)(wAddr));
    expandXOutGlobal.SetGlobalBuffer((__gm__ XOutType *)(expandXOutGM_) + (dstPosition)*hOutElemCount);
    DataCopyParams srcTokenCopyParams{static_cast<uint16_t>(blockCntPerToken_ * arriveCount),
                                      static_cast<uint16_t>(SPLIT_BLOCK_DATA_SIZE), static_cast<uint16_t>(UB_ALIGN), 0};
    DataCopyExtParams scalesCopyParams{
        uint16_t(arriveCount), static_cast<uint32_t>(scaleOutBytes_),
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - scaleOutBytes_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams tokenCopyParams{
        uint16_t(arriveCount), hOutSize_,
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE - hOutSize_) / UB_ALIGN), 0U, 0U};
    DataCopyExtParams expandIdxCopyParams{
        uint16_t(arriveCount), EXPAND_IDX_INFO * sizeof(int32_t),
        static_cast<uint32_t>((blockCntPerToken_ * SPLIT_BLOCK_DATA_SIZE) / UB_ALIGN - 1), 0U, 0U};
    DataCopyPadParams srcTokenPadParams{false, 0U, 0U, 0U};
    LocalTensor<int32_t> xOutInt32Tensor = xTmpTensor.template ReinterpretCast<int32_t>();

    if (isDataBufferReused) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(dataBufferId);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(dataBufferId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(dataBufferId);
    }
    DataCopyPad(xTmpTensor, dataFlagGlobal[expertFinishNumTensor_(index) * hCommuSize_ / sizeof(XOutType)],
                srcTokenCopyParams, srcTokenPadParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(dataBufferId);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(dataBufferId);
    quantInst_.CopyScalesToOut(dstPosition, scaleOutBytes_, xTmpTensor, scalesCopyParams);
    DataCopyPad(expandXOutGlobal, xTmpTensor, tokenCopyParams);
    DataCopyPad(expandIdxGMTensor_[dstPosition * EXPAND_IDX_INFO], xOutInt32Tensor[tokenQuantAlign_],
                expandIdxCopyParams);
    if (hasExpertScalesFlag_) {
        LocalTensor<float> xOutFloatTensor = xTmpTensor.template ReinterpretCast<float>();
        uint32_t expertScaleOffset = expertScaleAlign_ * sizeof(float);
        uint32_t expertScaleWinOffset =
            expertScaleOffset / SPLIT_BLOCK_DATA_SIZE * SPLIT_BLOCK_SIZE + expertScaleOffset % SPLIT_BLOCK_DATA_SIZE;
        expertScaleInGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(
            wAddr + expertFinishNumTensor_(index) * hCommuSize_ + expertScaleWinOffset));
        expandScalesOutGlobal.SetGlobalBuffer((__gm__ float *)(expandScalesOutGM_) + dstPosition);
        DataCopyExtParams expertScaleCopyInParams{uint16_t(arriveCount), static_cast<uint32_t>(sizeof(float)),
                                                  static_cast<uint32_t>(hCommuSize_ - sizeof(float)), 0U, 0U};
        DataCopyPadExtParams<float> expertScalePadParams{false, 0U, 0U, 0.0f};
        DataCopyExtParams expertScaleCopyOutParams{uint16_t(arriveCount), static_cast<uint32_t>(sizeof(float)), 0U, 0U,
                                                   0U};
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(dataBufferId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(dataBufferId);
        DataCopyPad(xOutFloatTensor, expertScaleInGlobal, expertScaleCopyInParams, expertScalePadParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(dataBufferId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(dataBufferId);
        DataCopyPad(expandScalesOutGlobal, xOutFloatTensor, expertScaleCopyOutParams);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(dataBufferId);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline uint32_t
MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::GetLocalWindowSrcDataBlockIdx(uint32_t srcExpertId,
                                                                                                uint32_t localExpertNum)
{
    uint32_t srcDataBlockIdx = srcExpertId % epWorldSize_ * localExpertNum + srcExpertId / epWorldSize_;
    if (!(isShareExpertRankFlag_ || hasElasticInfoFlag_)) {
        srcDataBlockIdx = (srcExpertId + epRankId_) % epWorldSize_ * localExpertNum + srcExpertId / epWorldSize_;
    }
    return srcDataBlockIdx;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::ClearLocalWindowDataFlags(
    TBuf<> tBuf, uint32_t validNum, uint32_t localExpertNum)
{
    uint32_t maxCleanRecordNum = 0;
    for (uint32_t index = 0; index < validNum; ++index) {
        uint32_t cleanRecordNum = expertFinishNumTensor_(index) * blockCntPerToken_;
        maxCleanRecordNum = cleanRecordNum > maxCleanRecordNum ? cleanRecordNum : maxCleanRecordNum;
    }

    uint32_t cleanRecordCapacity = tBufRealSize_ / UB_ALIGN;
    cleanRecordCapacity =
        cleanRecordCapacity > DATA_COPY_MAX_BLOCK_COUNT ? DATA_COPY_MAX_BLOCK_COUNT : cleanRecordCapacity;
    uint32_t cleanRecordNum = maxCleanRecordNum > cleanRecordCapacity ? cleanRecordCapacity : maxCleanRecordNum;
    if (cleanRecordNum == 0) {
        return;
    }
    LocalTensor<float> cleanUpTensor = tBuf.Get<float>();
    Duplicate<float>(cleanUpTensor, float(0), cleanRecordNum * UB_ALIGN_DATA_COUNT);
    SyncFunc<AscendC::HardEvent::V_MTE3>();

    GlobalTensor<float> cleanGlobal;
    DataCopyParams cleanUpParams{1U, 1U, 0U, SPLIT_BLOCK_DATA_SIZE / UB_ALIGN};
    for (uint32_t index = 0; index < validNum; ++index) {
        uint32_t srcExpertId = expertMapTensor_(index);
        uint32_t srcDataBlockIdx = GetLocalWindowSrcDataBlockIdx(srcExpertId, localExpertNum);
        GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + srcDataBlockIdx * expertPerSizeOnWin_;
        cleanGlobal.SetGlobalBuffer((__gm__ float *)(wAddr));

        uint32_t leftRecordNum = expertFinishNumTensor_(index) * blockCntPerToken_;
        uint32_t recordOffset = 0;
        while (leftRecordNum > 0) {
            uint32_t curCleanRecordNum = leftRecordNum > cleanRecordNum ? cleanRecordNum : leftRecordNum;
            cleanUpParams.blockCount = static_cast<uint16_t>(curCleanRecordNum);
            uint32_t flagIndex = recordOffset * SPLIT_BLOCK_COUNT + SPLIT_BLOCK_DATA_COUNT;
            DataCopy(cleanGlobal[flagIndex], cleanUpTensor, cleanUpParams);
            recordOffset += curCleanRecordNum;
            leftRecordNum -= curCleanRecordNum;
        }
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::WaitAndFormatOutput(
    TBuf<> tBuf, uint32_t validNum)
{
    uint32_t index = 0;
    uint32_t finishNum = 0;
    uint32_t localExpertNum = isShareExpertRankFlag_ ? 1 : moeExpertNumPerRank_;
    uint32_t srcExpertId, dstPosition, arriveCount, copyCnt, srcDataBlockIdx;
    uint64_t packedTokenSize = static_cast<uint64_t>(blockCntPerToken_) * SPLIT_BLOCK_DATA_SIZE;
    uint64_t minUbSizePerToken =
        LOCAL_COPY_BUFFER_NUM * packedTokenSize + static_cast<uint64_t>(blockCntPerToken_) * (UB_ALIGN + sizeof(float));
    uint32_t maxCopyTokenCnt = static_cast<uint32_t>(tBufRealSize_ / minUbSizePerToken);
    uint32_t dataBufferSize = 0;
    uint32_t reduceWorkSize = 0;
    uint32_t flagRecvSize = 0;
    while (maxCopyTokenCnt > 0) {
        dataBufferSize = static_cast<uint32_t>(packedTokenSize * maxCopyTokenCnt);
        reduceWorkSize = Ceil(blockCntPerToken_ * maxCopyTokenCnt * sizeof(float), SIZE_ALIGN_256) * SIZE_ALIGN_256;
        flagRecvSize = blockCntPerToken_ * maxCopyTokenCnt * UB_ALIGN;
        uint64_t totalBufferSize =
            static_cast<uint64_t>(LOCAL_COPY_BUFFER_NUM) * dataBufferSize + reduceWorkSize + flagRecvSize;
        if (totalBufferSize <= tBufRealSize_) {
            break;
        }
        --maxCopyTokenCnt;
    }

    LocalTensor<XOutType> xTmpPingTensor = tBuf.GetWithOffset<XOutType>(dataBufferSize / sizeof(XOutType), 0);
    LocalTensor<XOutType> xTmpPongTensor =
        tBuf.GetWithOffset<XOutType>(dataBufferSize / sizeof(XOutType), dataBufferSize);
    uint32_t flagWorkOffset = LOCAL_COPY_BUFFER_NUM * dataBufferSize;
    flagReduceWorkTensor_ = tBuf.GetWithOffset<float>(reduceWorkSize / sizeof(float), flagWorkOffset);
    flagRecvTensor_ = tBuf.GetWithOffset<float>(flagRecvSize / sizeof(float), flagWorkOffset + reduceWorkSize);

    uint32_t copyBatchId = 0;
    while (true) {
        if (expertLeftNumTensor_(index) == 0) { // 当前核负责的不需要收集
            index = (index + 1) % validNum;     // 轮询查询每个有效的index
            continue;
        }
        srcExpertId = expertMapTensor_(index);
        copyCnt = expertLeftNumTensor_(index) > maxCopyTokenCnt ?
                      maxCopyTokenCnt :
                      expertLeftNumTensor_(index); // 按照ub大小一次搬入多个token
        srcDataBlockIdx = GetLocalWindowSrcDataBlockIdx(srcExpertId, localExpertNum);
        arriveCount = CheckDataArriveWithFlag(srcDataBlockIdx, expertFinishNumTensor_(index), copyCnt);
        if (arriveCount == copyCnt) {
            dstPosition = srcExpertId != 0 ? sendCntTensor_(srcExpertId - 1) : 0;
            dstPosition += expertFinishNumTensor_(index);
            GM_ADDR wAddr = (__gm__ uint8_t *)(windowGM_) + srcDataBlockIdx * expertPerSizeOnWin_;
            uint32_t dataBufferId = copyBatchId % LOCAL_COPY_BUFFER_NUM;
            LocalTensor<XOutType> xTmpTensor = xTmpPingTensor;
            if (dataBufferId != 0) {
                xTmpTensor = xTmpPongTensor;
            }
            CopyInAndOut(xTmpTensor, wAddr, index, dstPosition, arriveCount, dataBufferId,
                         copyBatchId >= LOCAL_COPY_BUFFER_NUM);
            ++copyBatchId;
            // Update progress here; data flags are cleared in batches after all outputs are issued.
            expertFinishNumTensor_(index) += arriveCount;
            expertLeftNumTensor_(index) -= arriveCount;
            if (expertLeftNumTensor_(index) == 0) {
                finishNum++;
            }
        } else {
            index = (index + 1) % validNum;
        }
        if (validNum == finishNum) {
            break;
        }
    }

    uint32_t usedDataBufferNum = copyBatchId > LOCAL_COPY_BUFFER_NUM ? LOCAL_COPY_BUFFER_NUM : copyBatchId;
    for (uint32_t dataBufferId = 0; dataBufferId < usedDataBufferNum; ++dataBufferId) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(dataBufferId);
    }
    ClearLocalWindowDataFlags(tBuf, validNum, localExpertNum);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::RunPosRecord(
    const uint32_t runPos)
{
    TBuf<> runPosBuf;
    tpipe_->InitBuffer(runPosBuf, UB_ALIGN);
    dataStateLocalTensor_ = runPosBuf.Get<uint32_t>();
    dataStateLocalTensor_.SetValue(0, runPos);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams_); // 维测打点
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::LocalWindowCopy()
{
    // 分核负责源专家数量
    tpipe_->Reset();
    TBuf<> cumSumBuf, statusWaitBuf, statusCleanBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    tpipe_->InitBuffer(statusWaitBuf, aivUsedCumSum_ * UB_ALIGN);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    tpipe_->InitBuffer(statusCleanBuf, aivUsedCumSum_ * UB_ALIGN);
    statusFp32Tensor_ = statusWaitBuf.Get<float>();
    statusCleanFp32Tensor_ = statusCleanBuf.Get<float>();
    sendCntTensor_ = cumSumBuf.Get<int32_t>();
    SplitToCore(rscvStatusNum_, aivNum_, startId_, endId_, sendNum_, true);
    if (useCcuCtrl_) {
        // 软同步：等唯一 cumsum 核（最后核）的 AlltoAll Wait 完成扇出（PipeBarrier 不跨核，就绪字自旋）
        CcuWaitMatrixReady();
    } else {
        WaitCumSumFlag();
    }
    if (sendNum_ == 0) {
        return;
    }
    // 连续化
    TBuf<> expertMapBuf, expertFinishBuf, expertLeftBuf, tBuf;
    uint32_t validNum = 0;
    uint32_t expInfoSize = Ceil(sendNum_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertMapBuf, expInfoSize);
    tpipe_->InitBuffer(expertFinishBuf, expInfoSize);
    tpipe_->InitBuffer(expertLeftBuf, expInfoSize);
    tBufRealSize_ = FULL_MESH_MAX_UB_SIZE - (UB_ALIGN + rscvNumAlign + 2 * aivUsedCumSum_ * UB_ALIGN) -
                    (expInfoSize * 3);       // 3为expInfoSize大小buffer申请个数
    tpipe_->InitBuffer(tBuf, tBufRealSize_); // 其余buffer空间统一申请
    expertMapTensor_ = expertMapBuf.Get<uint32_t>();
    expertFinishNumTensor_ = expertFinishBuf.Get<uint32_t>();
    expertLeftNumTensor_ = expertLeftBuf.Get<uint32_t>();
    SetValidExpertInfo(expInfoSize, validNum);
    if (validNum == 0) { // 本核负责的Expert对应rank收到数据
        return;
    }
    SyncFunc<AscendC::HardEvent::V_S>();
    WaitAndFormatOutput(tBuf, validNum);
    RunPosRecord(RUNPOS_ARRIVECNT); // 维测打点
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::TokenActiveMaskCal()
{
    // 搬运x_active_mask，当前仅用于计算有效token总数
    LocalTensor<half> maskTmpTensor;
    LocalTensor<half> sumOutTensor;
    LocalTensor<bool> maskInputTensor;
    uint32_t axisBsAlignSize = Ceil(axisBS_ * sizeof(bool), UB_ALIGN) * UB_ALIGN;
    maskInputTensor = dstExpBuf_.Get<bool>();
    maskTmpTensor = subExpBuf_.Get<half>();
    sumOutTensor = gatherMaskTBuf_.Get<half>();
    DataCopyExtParams maskParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(maskTmpTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, axisBS_);
    PipeBarrier<PIPE_V>();
    SumParams params{1, axisBsAlignSize, axisBS_};
    Sum(sumOutTensor, maskTmpTensor, params);
    SyncFunc<AscendC::HardEvent::V_S>();
    activeMaskBsCnt_ = static_cast<int32_t>(sumOutTensor.GetValue(0));
    sendToMoeExpTokenCnt_ = activeMaskBsCnt_ * axisK_;
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalValidBSCnt(
    LocalTensor<bool> maskStrideTensor)
{
    uint32_t mask = axisBS_;
    uint32_t activeMaskAlignSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN);
    uint32_t calCnt = Ceil(axisBS_ * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    uint32_t innerAlign = Ceil(axisK_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half) * BUFFER_NUM;
    LocalTensor<half> tempTensor = validExpertIndexBuf_.Get<half>();
    LocalTensor<half> maskTempTensor = expertIdsBuf_.Get<half>();
    LocalTensor<half> tokenTargetTensor = validBsIndexTBuf_.Get<half>();
    LocalTensor<uint8_t> maskTensor = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> bsIndexTensor = subExpBuf_.Get<int32_t>();
    LocalTensor<uint32_t> maskTensorInt32 = gatherMaskTBuf_.Get<uint32_t>();
    SumParams axisKSumParams{axisBS_, innerAlign, axisK_};
    SumParams axisBsSumParams{
        1, static_cast<uint32_t>(Ceil(axisBS_ * sizeof(half), UB_ALIGN) * UB_ALIGN / sizeof(half)), axisBS_};

    Duplicate<half>(maskTempTensor, (half)0, calCnt);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskStrideInt8Tensor = maskStrideTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskStrideInt8Tensor, RoundMode::CAST_NONE, activeMaskAlignSize);
    PipeBarrier<PIPE_V>();
    Sum(tokenTargetTensor, tempTensor, axisKSumParams);
    PipeBarrier<PIPE_V>();
    Mins(maskTempTensor, tokenTargetTensor, static_cast<half>(1), axisBS_);
    PipeBarrier<PIPE_V>();
    CompareScalar(maskTensor, maskTempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(bsIndexTensor, 0, axisBS_);
    PipeBarrier<PIPE_V>();
    GatherMask(validBsIndexTensor_, bsIndexTensor, maskTensorInt32, true, mask, {1, 1, 0, 0}, activeMaskBsCnt_);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::CalValidExpIdx(
    LocalTensor<bool> maskInputTensor)
{
    uint32_t mask = expertIdsCnt_;
    uint32_t curMaskCnt = axisBS_ * axisK_;
    uint32_t calCnt = Ceil(curMaskCnt * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    LocalTensor<int32_t> validExpertIndexTensor = validExpertIndexBuf_.Get<int32_t>();
    LocalTensor<half> tempTensor = subExpBuf_.Get<half>();
    LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTBuf_.Get<uint8_t>();
    LocalTensor<int32_t> expertsIndexTensor = expertIdsBuf_.Get<int32_t>();

    Duplicate<half>(tempTensor, (half)0, calCnt);
    PipeBarrier<PIPE_V>();
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    LocalTensor<int8_t> maskInputInt8Tensor = maskInputTensor.ReinterpretCast<int8_t>();
    Cast(tempTensor, maskInputInt8Tensor, RoundMode::CAST_NONE, curMaskCnt);
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0,
                        Ceil(expertIdsCnt_, SIZE_ALIGN_256) * SIZE_ALIGN_256 / BITS_PER_BYTE / sizeof(uint32_t));
    PipeBarrier<PIPE_V>();
    CompareScalar(gatherMaskTensorInt8, tempTensor, static_cast<half>(1), AscendC::CMPMODE::EQ, calCnt);
    CreateVecIndex(expertsIndexTensor, 0, curMaskCnt);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor_, true, mask, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::ExpertActiveMaskInit()
{
    uint32_t axisBSAlign = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    tpipe_->InitBuffer(validBsIndexTBuf_, axisBSAlign);
    uint32_t validBufferSize = expertIdsSize_ > xActivateMaskSize ? expertIdsSize_ : xActivateMaskSize;
    tpipe_->InitBuffer(validExpertIndexBuf_, validBufferSize);
    validBsIndexTensor_ = validBsIndexTBuf_.Get<int32_t>();
    gatherMaskTensor_ = gatherMaskTBuf_.Get<uint32_t>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::ExpertActiveMaskCal()
{
    // 计算当前有效bs数量, stride搬入xActiveMask进行sum计算, 用于moe专家发送
    LocalTensor<bool> maskStrideTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskStrideCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskStrideParams{static_cast<uint16_t>(axisBS_), static_cast<uint32_t>(axisK_ * sizeof(bool)), 0U,
                                       0U, 0U};
    DataCopyPad(maskStrideTensor, xActiveMaskGMTensor_, maskStrideParams, maskStrideCopyPadParams);
    CalValidBSCnt(maskStrideTensor);
    // 计算validExpIndexTensor, 连续搬入xActiveMask进行GatherMask计算, 用于moe专家的发送
    LocalTensor<bool> maskInputTensor = dstExpBuf_.Get<bool>();
    DataCopyPadExtParams<bool> maskCopyPadParams{false, 0U, 0U, 0U};
    DataCopyExtParams maskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(bool)), 0U, 0U, 0U};
    DataCopyPad(maskInputTensor, xActiveMaskGMTensor_, maskParams, maskCopyPadParams);
    CalValidExpIdx(maskInputTensor);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::MaskZeroComputeExpert(
    uint32_t maskCnt)
{
    LocalTensor<int32_t> expertsIndexTensor = dstExpBuf_.Get<int32_t>();
    int32_t maskTensorInt16Cnt = Ceil(expertIdsCnt_, UB_ALIGN / 2);
    LocalTensor<uint8_t> maskTensorInt8 = validExpertIndexBuf_.Get<uint8_t>();    // bs*k*1
    LocalTensor<uint32_t> maskTensorInt32 = validExpertIndexBuf_.Get<uint32_t>(); // bs*k*1
    LocalTensor<half> expertIdsTensorCast = subExpBuf_.Get<half>();               // bs*k*4
    LocalTensor<int32_t> validExpertIndexTensor = validExpertIndexBuf_.Get<int32_t>();
    int32_t moeExpertNumInt32 = static_cast<int32_t>(moeExpertNum_);

    DataCopyExtParams expertIdsCntParams = {1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    PipeBarrier<PIPE_V>();
    SetDeqScale((half)1.000000e+00f);
    PipeBarrier<PIPE_V>();
    Cast(expertIdsTensorCast, validExpertIdsTensor_, RoundMode::CAST_NONE, expertIdsCnt_);
    Duplicate<uint32_t>(maskTensorInt32, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    uint32_t calcCnt = Ceil(expertIdsCnt_ * sizeof(half), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(half);
    // 逐元素比较tensor中的元素和Scalar，比较结果为真时，输出对应比特位为1，
    // 否则为0，以筛掉零计算量专家
    CompareScalar(maskTensorInt8, expertIdsTensorCast, static_cast<half>(moeExpertNumInt32), AscendC::CMPMODE::LT,
                  calcCnt);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint16_t> maskTensorInt16 = validExpertIndexBuf_.Get<uint16_t>();  // 空间bs*k*1
    LocalTensor<uint16_t> gatherMaskTensorint16 = gatherMaskTBuf_.Get<uint16_t>(); // 空间bs*k*4
    /* 特殊专家的maskTensorInt16和之前的gatherMaskTensor_结果按位相与，AND 支持uint16，
     * gatherMaskTensor_和gatherMaskTensorint16是同一个地址 */
    And(gatherMaskTensorint16, gatherMaskTensorint16, maskTensorInt16, maskTensorInt16Cnt);
    PipeBarrier<PIPE_V>();
    // 再筛一次
    CreateVecIndex(expertsIndexTensor, 0, expertIdsCnt_);
    PipeBarrier<PIPE_V>();
    GatherMask(validExpertIndexTensor, expertsIndexTensor, gatherMaskTensor_, true, maskCnt, {1, 1, 0, 0},
               sendToMoeExpTokenCnt_);
    SyncFunc<AscendC::HardEvent::V_S>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::GenerateGatherMaskTensor(
    uint32_t maskCnt)
{
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0xFFFFFFFF, Ceil(maskCnt, UB_ALIGN));
    PipeBarrier<PIPE_V>();
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::ZeroComputeExpertMaskCal()
{
    uint32_t maskCnt = expertIdsCnt_;
    if (isTokenMaskFlag_) { // 一维
        maskCnt = activeMaskBsCnt_ * axisK_;
    }

    if (!isExpertMaskFlag_) { // 非二维要生成gatherMaskTensor_
        GenerateGatherMaskTensor(maskCnt);
    }

    // 零计算量专家剪枝
    MaskZeroComputeExpert(maskCnt);
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::ExpIdsCopyAndMaskCal()
{
    activeMaskBsCnt_ = axisBS_;
    sendToMoeExpTokenCnt_ = axisBS_ * axisK_;
    validExpertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) {
        ExpertActiveMaskInit();
    }

    if (isTokenMaskFlag_) {
        TokenActiveMaskCal();
    }

    if (isExpertMaskFlag_) {
        ExpertActiveMaskCal();
    }
    if (activeMaskBsCnt_ == 0) { // DataCopyPad不能拷贝0个
        return;
    }
    if (zeroComputeExpertNum_ != 0) {
        ZeroComputeExpertMaskCal();
    }

    Duplicate<int32_t>(validExpertIdsTensor_, -1, int32_t(expertIdsBufSize_ / sizeof(int32_t)));
    // 拷贝bs*k个专家id 到local  补齐到32 字节对齐  bs*k = 3*3 = 9 expertIdsAlignCnt= 16 补7个 填充-1
    if (isExpertMaskFlag_ || (zeroComputeExpertNum_ != 0)) {
        LocalTensor<int32_t> tmpExpertIdsTensor = subExpBuf_.Get<int32_t>();
        LocalTensor<float> tmpExpertIdsTensorFloat = subExpBuf_.Get<float>();
        LocalTensor<uint8_t> gatherMaskTensorInt8 = gatherMaskTensor_.ReinterpretCast<uint8_t>();
        DataCopyExtParams expertIdsMaskParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> expertIdsMaskCopyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tmpExpertIdsTensor, expertIdsGMTensor_, expertIdsMaskParams, expertIdsMaskCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        PipeBarrier<PIPE_V>();
        LocalTensor<float> validExpertIdsFloat = validExpertIdsTensor_.ReinterpretCast<float>();
        Select(validExpertIdsFloat, gatherMaskTensorInt8, tmpExpertIdsTensorFloat, static_cast<float>(-1),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, expertIdsCnt_);
        SyncFunc<AscendC::HardEvent::V_S>();
    } else {
        uint32_t expertIdsMask = activeMaskBsCnt_ * axisK_;
        uint32_t expertIdsAlignCnt = Ceil(expertIdsMask, BITS_PER_BYTE) * BITS_PER_BYTE;
        uint32_t rightPadding = expertIdsAlignCnt - expertIdsMask;
        // rightPadding字节数不能超过32，不能超过8个u32
        DataCopyPadExtParams<int32_t> expertIdsCntCopyPadParams{true, 0U, uint8_t(rightPadding), -1};
        DataCopyExtParams expertIdsCntParams{1U, static_cast<uint32_t>(expertIdsMask * sizeof(uint32_t)), 0U, 0U,
                                             0U}; // 第二个参数blockLen 范围[1, 2097151] 不能为0
        SyncFunc<AscendC::HardEvent::V_MTE2>();
        DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_, expertIdsCntParams, expertIdsCntCopyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
    if (hasExpertScalesFlag_) {
        expertScalesTensor_ = expertScalesBuf_.Get<float>();
        DataCopyExtParams expertScalesParams{1U, static_cast<uint32_t>(expertIdsCnt_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> expertScalesPadParams{false, 0U, 0U, 0U};
        DataCopyPad(expertScalesTensor_, expertScalesGMTensor_, expertScalesParams, expertScalesPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
    }
}

template <TemplateMC2A5FullMeshTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5FullMesh<TemplateMC2A5FullMeshTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        if (aivId_ < aivUsedAllToAll_) {
            AllToAllDispatch(); // 前面核all2all发送
        } else {
            CalCumSum(); // 后面核计算前缀和，输出epRecvCnt/exportTokenNums
        }
        // localWindowCopy中包含reset操作，需确保前面操作完成
        PipeBarrier<PIPE_ALL>();
        LocalWindowCopy(); // 本卡上专家数据连续化，输出expandX/scales/expandIdx
        if (useCcuCtrl_ && (aivId_ == aivUsedAllToAll_)) {
            ccuHccl_.Finalize(); // 唯一 cumsum 核（控制面 issue/Wait 核）负责释放 CCU 资源
        }
    }
}

} // namespace MoeDistributeDispatchV2A5FullMeshImpl
#endif // MOE_DISTRIBUTE_DISPATCH_V2_A5_FULL_MESH_H
