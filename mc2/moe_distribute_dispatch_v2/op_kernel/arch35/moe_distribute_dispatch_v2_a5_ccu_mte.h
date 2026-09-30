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
 * \file moe_distribute_dispatch_v2_a5_ccu_mte.h
 * \brief a5_ccu 配方（CCU 注册，V4→InnerV2 宿主）的双引擎混合内核：数据面 = CCU 注入 context
 *        （GetHcclContext<0>，参考 a5_ccu 纯 CCU 内核直取 windowsOut[0] 的用法）windowsIn[远端]/
 *        windowsOut[0][本端] IPC 窗口的 MTE 直写（480/32 inline flag），count 面 = 唯一核 CCU 等长
 *        AlltoAll。两面共用同一 CCU 注入 context，无额外 host 递送（graph 模式可用）；CCU 模式
 *        无条件进入本变体。
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_A5_CCU_MTE_H
#define MOE_DISTRIBUTE_DISPATCH_V2_A5_CCU_MTE_H

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
#include "../moe_distribute_v2_base.h"
#include "../check_winsize.h"
#include "../../../common/op_kernel/moe_distribute_base.h"
#include "../../../common/op_kernel/mc2_kernel_utils.h"
#define FLOAT_OVERFLOW_MODE_CTRL 60
namespace MoeDistributeDispatchV2A5CcuMteImpl {
using namespace Mc2Kernel;
constexpr uint64_t FLAG_FIELD_OFFSET = 768UL * 1024UL; // 384 * 2，0/1标识区偏移
constexpr uint64_t SPLIT_BLOCK_SIZE = 512UL;
constexpr uint64_t SPLIT_BLOCK_COUNT = 128UL; // 128 = SPLIT_BLOCK_SIZE / sizeof(float)
constexpr int32_t FULL_MESH_MAX_UB_SIZE = 190 * 1024;
constexpr uint32_t SPLIT_BLOCK_DATA_SIZE = 480U;
constexpr uint32_t SPLIT_BLOCK_DATA_COUNT = 120U; // 120 = SPLIT_BLOCK_DATA_SIZE / sizeof(float)
constexpr uint32_t SIZE_ALIGN_256 = 256U;
// CCU 控制面软同步标志位于 per-AIV 状态字（512B 槽）空闲字：状态区启动时清零、跨 launch 持久
// （workspace 不做初始化清零，依赖零基线的标志不得放 workspace）；word0-7 为 InitWinState 块、8-9 为 HCCL DFX。
// 512B=1 cacheline/核，各核独占一行，规避多核同 cacheline 伪共享。前缀表就绪字：固定 word16（槽内 64B 偏移，
// 32B 对齐）——cumsum 核 scan+写回后以单条 strided DataCopy（DataCopyParams 32B 块单位，full_mesh
// sumIntriParams 同型）向各核槽 word16 所在 32B 块广播 1.0f（word17-23 为连带写入的旁路字，无读者）；
// 各核自旋本槽 word16，命中后自清 0（用后清零，full_mesh WaitCumSumFlag 先例）供下轮发布——异常中止残留
// 由下轮广播整块覆写消除（中止场景本就要求 device reset，与基线同级风险）
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
// IPC 端到端验证死区（验证后删除）：[TEST] 写入/读回均落本端窗口状态区 [0,768KB) 死区内固定 +256KB
//（数据面不经此处，magic 可存活至 gather 期读回；单向验证：仅 rank 0 写 slot0，其余 rank 校验）
constexpr uint64_t CCU_IPC_TEST_OFFSET = 256UL * 1024UL;
constexpr uint32_t RUNPOS_CALCUMSUM = 2U;
constexpr uint32_t RUNPOS_CUMSUMFLAG = 3U;
constexpr uint32_t RUNPOS_ARRIVECNT = 4U;
constexpr uint8_t VALID_EVENT_FLAG_NUM = 8U;
constexpr uint8_t LOCAL_COPY_BUFFER_NUM = 2U;
constexpr uint32_t DATA_COPY_MAX_BLOCK_COUNT = 4095U;
constexpr uint8_t UB_ALIGN_DATA_COUNT = 8U; // 8 = UB_ALIGN / sizeof(float) = UB_ALIGN / sizeof(int32_t)
constexpr uint32_t DURATION_OFFSET = sizeof(int64_t) / sizeof(int32_t);

#define TemplateCcuMteTypeClass typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist
#define TemplateCcuMteTypeFunc XType, ExpandXOutType, QuantMode, IsSmoothScaleExist

using namespace AscendC;
using namespace MoeDistributeV2Base;

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

template <TemplateCcuMteTypeClass>
class MoeDistributeDispatchV2A5CcuMte {
public:
    using XInType = typename std::conditional<
        (Std::IsSame<XType, fp4x2_e2m1_t>::value) || (Std::IsSame<XType, fp4x2_e1m2_t>::value), uint8_t, XType>::type;
    using XOutType = typename std::conditional<(Std::IsSame<ExpandXOutType, fp4x2_e2m1_t>::value) ||
                                                   (Std::IsSame<ExpandXOutType, fp4x2_e1m2_t>::value),
                                               uint8_t, ExpandXOutType>::type;
    __aicore__ inline MoeDistributeDispatchV2A5CcuMte(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
                                GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut,
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
    __aicore__ inline void CcuWaitMatrixReady(); // CCU 控制面：自旋等待唯一 cumsum 核（最后核）的 Wait 完成
    __aicore__ inline int32_t CalExpertCnt(uint32_t expertId, uint32_t maskCnt);
    __aicore__ inline void CalCntPackAndIssueCcu(); // CCU 控制面：count 面编排（计数→AlltoAll→前缀→扇出）
    __aicore__ inline void CcuCalCountsToSegBuf(); // CCU 控制面：专家计数（分组直方图）构造并落入 sendBuf 段
    __aicore__ inline void CcuScanAndPublishPrefix(); // CCU 控制面：全表前缀 scan、写回、两统计输出+就绪字广播
    __aicore__ inline void BuildSendCntFromCcuMatrix(); // CCU 控制面：各核拷自己前缀表切片
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
    // 数据面窗口基址（CCU 注入 context，参考 a5_ccu 纯 CCU 内核直取 windowsOut[0] 的用法）：本端取
    // windowsOut[0]（hcomm CCU 配方不填 windowsIn[本端]，表项=0）；远端 r 取 windowsIn[r]（远端
    // cclBuffer 的 IPC 直写映射基址，整块无切片项，勿按 AIV 半区语义解读）。与 full_mesh 的
    // GetBaseWindAddrByRankId 同式叠加 per-rank 1KB（EP_RANK_OFFSET_STEP）偏移：打散各卡窗口低地址位，
    // 规避 A5 地址路由归一导致的同相位访问冲突（对性能影响明显）；收发两侧按窗口属主 rank 同式计算，对称
    __aicore__ inline GM_ADDR GetWinBaseByRankId(const int32_t rankId)
    {
        const uint64_t routeOffset = static_cast<uint64_t>(rankId) * EP_RANK_OFFSET_STEP;
        if (rankId == static_cast<int32_t>(ccuWinContext_->rankId)) {
            return (GM_ADDR)ccuWinContext_->windowsOut[0] + routeOffset;
        }
        return (GM_ADDR)ccuWinContext_->windowsIn[rankId] + routeOffset;
    }

    // 数据区基址在本端/远端窗口的 STATE_SIZE(1MB) 之后，与 full_mesh 窗口布局同语义
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return GetWinBaseByRankId(rankId) + Mc2Kernel::STATE_SIZE + winDataSizeOffset_;
    }

    // 状态区基址即窗口起始地址（叠加 dataState_ 双缓冲轮转偏移）
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return GetWinBaseByRankId(rankId) + dataState_ * WIN_STATE_OFFSET;
    }

    // per-AIV 状态字（FLAG_FIELD_OFFSET 起每 AIV 一个 512B 槽，启动清零、跨 launch 持久）的 float 视图：
    // CCU 软同步标志专用（word10+ 空闲段；就绪字取 word16/word17 按 dataState_ 键控，word16 保 32B 对齐），
    // 勿放 workspace——首轮脏值可能假命中
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
    GlobalTensor<bool> xActiveMaskGMTensor_;
    GlobalTensor<int32_t> expandIdxGMTensor_;
    GlobalTensor<int32_t> elasticInfoGMTensor_;
    GlobalTensor<float> selfRankWinInGMTensor_;
    GlobalTensor<uint32_t> selfDataStatusGMTensor_;

    LocalTensor<float> workLocalTensor_;
    LocalTensor<int32_t> validExpertIdsTensor_;
    LocalTensor<int32_t> elasticInfoTensor_;
    LocalTensor<int32_t> validBsIndexTensor_;
    LocalTensor<uint32_t> gatherMaskTensor_;
    LocalTensor<float> smoothScalesTensor_;
    LocalTensor<float> expertScalesTensor_;
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

    TBuf<> dstExpBuf_;
    TBuf<> subExpBuf_;
    TBuf<> gatherMaskTBuf_;
    TBuf<> expertIdsBuf_;
    TBuf<> expertScalesBuf_;
    TBuf<> elasticInfoBuf_;
    TBuf<> scalarBuf_;
    TBuf<> validExpertIndexBuf_;
    TBuf<> validBsIndexTBuf_;
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
    uint32_t aivUsedCumSum_{0};
    uint32_t aivUsedAllToAll_{0};
    uint32_t maxSize_{0};
    uint32_t expertIdsSize_{0};
    uint32_t globalBS_{0};
    uint32_t copyInAxisH_{0};
    uint32_t copyOutAxisH_{0};
    uint64_t totalUbSize_{0};

    // 数据面窗口表源：CCU 注入 context（GetHcclContext<0>），本端/远端基址解析见 GetWinBaseByRankId
    __gm__ HcclCombineOpParam *ccuWinContext_{nullptr};

    // ==================== CCU 控制面（恒开：本变体固定 count 交换走等长 AlltoAll）====================
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> ccuHccl_; // 控制面实例，与数据面 context 共存
    AscendC::HcclHandle ccuCntHandle_{0};
    GM_ADDR ccuSendBufGM_{nullptr}; // workspace 用户区头部：本 rank 的 W 段等长行 [counts...]
    GM_ADDR ccuRecvBufGM_{nullptr}; // workspace 用户区：src 主序全矩阵（双半区乒乓），Wait 后各核自读
    uint32_t ccuSegWords_{0};       // 段 u32 字数 = Ceil(moeExpertNumPerRank_, 8) * 8，与 host 同公式

    DataCopyParams hCopyParams_;
    DataCopyParams dataStateParams_{1U, sizeof(uint32_t), 0U, 0U};

    MoeDistributeDispatchV2Quant<XInType, ExpandXOutType, XOutType, QuantMode, IsSmoothScaleExist> quantInst_;
};

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::InitElasticInfo()
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::SetTilingData(
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    axisBS_ = tilingData->moeDistributeDispatchV2Info.bs;
    axisH_ = tilingData->moeDistributeDispatchV2Info.h;
    epWorldSizeOriginal_ = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    epRankIdOriginal_ = tilingData->moeDistributeDispatchV2Info.epRankId;
    hasElasticInfoFlag_ = tilingData->moeDistributeDispatchV2Info.hasElasticInfo;
    hasExpertScalesFlag_ = tilingData->moeDistributeDispatchV2Info.hasExpertScales;
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::SetTilingDataAndCal(
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
    // CCU 控制面：count 面（计算+组包+issue+Wait）收敛单核——固定核 0（CCU 客户端 owner：mc2_templates/
    // 基线 a5_ccu/apace 全部先例的通信控制核均为 blockIdx 0，SDK 客户端 prepare/notify 通路疑似核 0 亲和，
    // 末核 issue 会出现引擎完成而 Wait 不醒）；其余核（1..aivNum-1）全部还给 token 发送。单核串行免去
    // pack 软同步，且与前台各核的 token 发送天然并行；aivUsedCumSum_/aivUsedAllToAll_ 保留为数量语义
    aivUsedCumSum_ = 1U;
    aivUsedAllToAll_ = aivNum_ - aivUsedCumSum_;
    if (sharedExpertRankNum_ != 0U) {
        sharedUsedAivNum_ = (aivUsedAllToAll_ * sharedExpertNum_) / (axisK_ + sharedExpertNum_);
        if (sharedUsedAivNum_ == 0) {
            sharedUsedAivNum_ = 1;
        }
    }
    moeUsedAivNum_ = aivUsedAllToAll_ - sharedUsedAivNum_;
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::SetDataStatus()
{
    statusDataSpaceGM_ = GetWinBaseByRankId(static_cast<int32_t>(ccuWinContext_->rankId));
    uint32_t epRankIdHccl = ccuWinContext_->rankId;
    uint32_t epWorldSizeHccl = epWorldSizeOriginal_;
    selfDataStatusGMTensor_.SetGlobalBuffer(
        (__gm__ uint32_t *)(statusDataSpaceGM_ + FLAG_FIELD_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    dataState_ = InitWinState(selfDataStatusGMTensor_, epRankIdHccl, epWorldSizeHccl, epRankIdOriginal_, moeExpertNum_,
                              epWorldSizeOriginal_, globalBS_, dataStateBuf);
    // 联调打点（稳定后可删）：H3 相位 + H1b 域对拍——dump ZERONE 原始返回值（刻意暂不钳位，保留野值证据）：
    // 两卡对拍均 ∈{0,1} 且相等=同相；不等=错相（gather 挂死根因候选）；非{0,1}=窗口零基线破坏（回收显存）。
    // 同行附 tiling epRankIdOriginal_：与行首 r=（ctx rankId）不等即数据面窗口索引域错位（H1b）
    AscendC::printf("[CCU_MTE] r=%u core=%u phase: dataState=%u eprankOrig=%u\n", ccuWinContext_->rankId, aivId_,
                    dataState_, epRankIdOriginal_);
    uint64_t hSizeAlignCombine = Ceil(axisH_ * COMBINE_IN_DATA_SIZE, SPLIT_BLOCK_DATA_SIZE) * SPLIT_BLOCK_SIZE;
    winDataSizeOffset_ =
        dataState_ * (totalWinSize_ / BUFFER_NUM) + axisMaxBS_ * (axisK_ + sharedExpertNum_) * hSizeAlignCombine;
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR expandIdxOut,
    GM_ADDR expandScalesOut, GM_ADDR expertTokenNumsOut, GM_ADDR sendCountsOut, GM_ADDR workspaceGM, TPipe *pipe,
    const MoeDistributeDispatchV2TilingData *tilingData)
{
    if (GetBlockIdx() == 0) { // 联调打点（稳定后可删）：kernel 存活证明——无此条 = kernel 未被调度
        AscendC::printf("[CCU_MTE] Init enter\n");
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    tpipe_ = pipe;
    tpipe_->InitBuffer(calBeginBuf_, UB_ALIGN);
    aivId_ = GetBlockIdx();
    totalWinSize_ = static_cast<uint64_t>(tilingData->moeDistributeDispatchV2Info.totalWinSizeEp);
    // 数据面窗口源：CCU 注入 context（GetHcclContext<0>，graph 模式下同样可用），本端/远端基址解析见
    // GetWinBaseByRankId。容量校验：cclBuffer（winSize）需容纳路由偏移（rankNum*1KB 上界）+ 1MB 状态区
    // + 数据区，不足 fail-fast
    ccuWinContext_ = (__gm__ HcclCombineOpParam *)(GetHcclContext<HCCL_GROUP_ID_0>());
    if (GetBlockIdx() == 0) { // 联调打点（稳定后可删）：context 指针有效性，只打不解引用——若为空，
        // 让紧随的 winNeed_ 解引用自然触发 trap，本条已先落缓冲（挂死场景缓冲可能整批丢失，以 plog 互证）
        AscendC::printf("[CCU_MTE] r=%u ctx=0x%llx %s\n", ccuWinContext_->rankId, (unsigned long long)ccuWinContext_,
                        (ccuWinContext_ == nullptr) ? "NULL" : "ok");
    }
    const uint64_t winNeed_ =
        Mc2Kernel::STATE_SIZE + totalWinSize_ + static_cast<uint64_t>(ccuWinContext_->rankNum) * EP_RANK_OFFSET_STEP;
    // CCU 注入 ctx 路径不做 winsize 校验（对齐 mc2_moe_context_holder 的 no-op 先例）：check_winsize 的
    // 失败路径是故意越界触发 aicore 异常的 fail-fast；winSize 与 STATE_SIZE/totalWinSize_/路由偏移三项
    // 预算的契约，待按下方打印的实际数值重定后再评估恢复校验
    if (GetBlockIdx() == 0) { // 联调打点（稳定后可删）：容量数值 dump（无校验，供事后核定契约）
        AscendC::printf("[CCU_MTE] win: rank=%u rankNum=%u winSize=%llu need=%llu out0=0x%llx\n",
                        ccuWinContext_->rankId, ccuWinContext_->rankNum, (unsigned long long)ccuWinContext_->winSize,
                        (unsigned long long)winNeed_, (unsigned long long)ccuWinContext_->windowsOut[0]);
    }
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
    if (hasExpertScalesFlag_) {
        expertScalesGMTensor_.SetGlobalBuffer((__gm__ float *)expertScales);
        expandScalesOutGM_ = expandScalesOut;
    }
    SetDataStatus();
    if (GetBlockIdx() == 0) { // 联调打点（稳定后可删）：状态区初始化（含 InitWinState 对窗口的写入）完成
        AscendC::printf("[CCU_MTE] r=%u win state ok\n", ccuWinContext_->rankId);
    }
    expandXOutGM_ = expandXOut;
    sendCountsOutGM_ = sendCountsOut;
    recvCntWorkspaceGM_ = AscendC::GetUserWorkspace(workspaceGM);
    statusSpaceGM_ = GetWindStateAddrByRankId(epRankIdOriginal_);
    selfRankWinInGMTensor_.SetGlobalBuffer((__gm__ float *)(statusDataSpaceGM_));
    windowGM_ = GetWindAddrByRankId(epRankIdOriginal_);
    hCopyParams_ = {1U, static_cast<uint32_t>(copyInAxisH_ * sizeof(XInType)), 0U, 0U};
    dataStateParams_ = {1U, sizeof(uint32_t), 0U, 0U};
    expertIdsSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;

    // CCU 控制面初始化（恒开）：count 面固定走 CCU 等长 AlltoAll，与数据面共用同一注入 context；
    // 原 MTE count 面（多 flag 轮询）与直写 GM scan A/B 备选已随混合定型裁剪。InitV2/SetCcTilingV2 为
    // 纯本地状态初始化（无 RPC），按 mc2_templates/基线 a5_ccu 先例全核执行；带 RPC 的
    // AlltoAll/Wait/Finalize 才钉核 0（CCU 客户端 owner，notify 通路核 0 亲和）
    ccuHccl_.InitV2((GM_ADDR)ccuWinContext_, tilingData);
    ccuHccl_.SetCcTilingV2(offsetof(MoeDistributeDispatchV2TilingData, mc2CcTiling1));
    if (GetBlockIdx() == 0) { // 联调打点（稳定后可删）：Init 抵达 CCU 客户端初始化点
        AscendC::printf("[CCU_MTE] r=%u hccl client ok\n", ccuWinContext_->rankId);
    }
    ccuSegWords_ = ((moeExpertNumPerRank_ + 7U) / 8U) * 8U;
    // CCU 缓冲自系统预留区之后（GetUserWorkspace）起步：host 侧 tiling 混合块以 max(纯 CCU 公式,
    // full_mesh 容量+CCU 追加) 计算总量（只增不减——防同 tiling key 下旧二进制配新 tiling 越界）；
    // CCU count 缓冲占用户区头部（MTE count 面的 per-core recvCnt 表已裁剪，头区空闲）。
    // [sendBuf: W*segWords*4B][recvBuf 半区0][recvBuf 半区1]——recvBuf 跟随 dataState_ 双缓冲，
    // 防止相邻两轮 launch 间对端引擎写入与本端读取踩踏（矩阵新鲜度由 Wait 契约保证，双半区防跨轮踩踏）
    ccuSendBufGM_ = recvCntWorkspaceGM_;
    uint64_t ccuRecvSize = static_cast<uint64_t>(epWorldSize_) * ccuSegWords_ * sizeof(uint32_t);
    ccuRecvBufGM_ = ccuSendBufGM_ + ccuRecvSize + dataState_ * ccuRecvSize;

    // 诊断打印 + 直写测试（bring-up 用，验证后删）：仅 0 核输出 CCU 注入 context 的完整窗口表
    // （windowsIn/windowsOut 逐 rank，本端项 out=windowsOut[0]、in 不填）与容量对照；selfData/
    // selfState 为数据面自留区基址；ccuSnd/ccuRcv 为 workspace 内 count 缓冲
    if (GetBlockIdx() == 0) {
        AscendC::printf("[CCU_MTE][WIN] rank=%u rankNum=%u winSize=%llu need=%llu\n", ccuWinContext_->rankId,
                        ccuWinContext_->rankNum, (unsigned long long)ccuWinContext_->winSize,
                        (unsigned long long)winNeed_);
        uint32_t diagRankNum = (ccuWinContext_->rankNum < 64U) ? ccuWinContext_->rankNum : 64U;
        for (uint32_t i = 0; i < diagRankNum; ++i) {
            AscendC::printf("[CCU_MTE][WIN] r=%u in=0x%llx out=0x%llx\n", i,
                            (unsigned long long)ccuWinContext_->windowsIn[i],
                            (unsigned long long)ccuWinContext_->windowsOut[i]);
        }
        AscendC::printf("[CCU_MTE] r=%u selfData=0x%llx selfState=0x%llx ccuSnd=0x%llx ccuRcv=0x%llx segW=%u\n",
                        ccuWinContext_->rankId, (unsigned long long)windowGM_, (unsigned long long)statusSpaceGM_,
                        (unsigned long long)ccuSendBufGM_, (unsigned long long)ccuRecvBufGM_, ccuSegWords_);
        // ===== IPC 直写远端测试（验证后删除，单向：仅 rank 0 写）：经 GetWinBaseByRankId（与发送路径
        // 同源的基址派生，含路由偏移）向每个远端 rank 的状态区死区 +256KB 写 magic 槽（slot0=发送方
        // rankId 编码；选死区而非数据区——数据区起始会被真实 token 覆盖，magic 无法存活到 gather 期读回）。
        // 读回核对在 LocalWindowCopy 就绪字命中后（全 rank Init 届时必已完成）。UB 复用 calBeginBuf_（32B）
        if ((ccuWinContext_->rankId == 0U) && (ccuWinContext_->rankNum > 1U)) {
            LocalTensor<int32_t> testTensor = calBeginBuf_.Get<int32_t>();
            uint32_t testMagic = 0xA5CC0000U | ccuWinContext_->rankId;
            Duplicate<int32_t>(testTensor, static_cast<int32_t>(testMagic), 8U);
            SyncFunc<AscendC::HardEvent::V_S>();
            SyncFunc<AscendC::HardEvent::S_MTE3>();
            for (uint32_t rmt = 0U; rmt < ccuWinContext_->rankNum; ++rmt) {
                if (rmt == ccuWinContext_->rankId) {
                    continue;
                }
                if (ccuWinContext_->windowsIn[rmt] == 0U) {
                    AscendC::printf("[CCU_MTE][TEST] r=%u rmt=%u windowsIn=0 not mapped, skip\n",
                                    ccuWinContext_->rankId, rmt);
                    continue;
                }
                GlobalTensor<int32_t> testDstTensor;
                testDstTensor.SetGlobalBuffer(
                    (__gm__ int32_t*)(GetWinBaseByRankId(static_cast<int32_t>(rmt)) + CCU_IPC_TEST_OFFSET +
                                      static_cast<uint64_t>(ccuWinContext_->rankId) * sizeof(int32_t) * 8U));
                DataCopy(testDstTensor, testTensor, 8U);
                AscendC::printf(
                    "[CCU_MTE][TEST] r=%u wrote magic=0x%llx -> rmt=%u dst=0x%llx\n", ccuWinContext_->rankId,
                    (unsigned long long)testMagic, rmt,
                    (unsigned long long)(GetWinBaseByRankId(static_cast<int32_t>(rmt)) + CCU_IPC_TEST_OFFSET));
            }
            SyncFunc<AscendC::HardEvent::MTE3_S>();
        }
    }
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::FillTriple(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::TokenToExpertInQuant(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::TokenToExpert(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::SplitToCore(
    uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t &startTokenId, uint32_t &endTokenId, uint32_t &sendTokenNum,
    bool isFront)
{
    sendTokenNum = curSendCnt / curUseAivNum;               // 每个aiv需要发送的token数
    uint32_t remainderTokenNum = curSendCnt % curUseAivNum; // 余数
    uint32_t newAivId;
    if (isFront) {
        newAivId = aivId_; // gather 分核：全部核（含核 0）按绝对 aivId 分摊
    } else if (aivId_ == 0U) { // 核 0 为 cumsum 核不参与发送（防御分支，正常不会走到）
        newAivId = 0U;
    } else {
        newAivId = aivId_ - 1U - moeUsedAivNum_; // scatter 核（1..aivNum-1，逻辑号=aivId-1）中后面的核发送给共享专家
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::SendToSharedExpert(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::SendToMoeExpert(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::SendToMoeExpertLoop(
    TQue<QuePosition::VECIN, 1> inQueue, uint32_t validTokenNum, bool writeExpertScale)
{
    GlobalTensor<XOutType> dstWinGMTensor;
    int32_t dstTokenIdx = 0;
    int32_t scatterId = static_cast<int32_t>(aivId_) - 1; // 核 0 为 cumsum 核，scatter 逻辑号从核 1 起算

    for (int32_t index = scatterId; index < validTokenNum; index += moeUsedAivNum_) {
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CalcSendTokenBufNum(
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
    if (aivId_ == 1U) { // 联调打点（稳定后可删，首个 scatter 核）：发送乒乓缓冲数
        AscendC::printf("[CCU_MTE] r=%u send bufs: bufNum=%u hCommuSize=%u\n", ccuWinContext_->rankId, sendTokenBufNum_,
                        hCommuSize_);
    }

    tpipe_->InitBuffer(outBuf, hCommuSize_ * sendTokenBufNum_);
    outTensor_ = outBuf.Get<XOutType>();
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::AllToAllDispatch()
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
    if (aivId_ == 1U) { // 联调打点（稳定后可删，首个 scatter 核）：token 发送面入口
        AscendC::printf("[CCU_MTE] r=%u scatter: bs=%u k=%u h=%u activeBs=%llu moeAiv=%u sharedAiv=%u\n", axisBS_,
                        axisK_, axisH_, ccuWinContext_->rankId, (unsigned long long)activeMaskBsCnt_, moeUsedAivNum_,
                        sharedUsedAivNum_);
    }
    AllToAllDispatchA5(inQueue, outBuf);
    AscendC::printf("[CCU_MTE] r=%u scatter: core=%u send done\n", ccuWinContext_->rankId,
                    aivId_); // 联调打点（稳定后可删）：逐核发送完成
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::AllToAllDispatchA5(
    TQue<QuePosition::VECIN, 1> inQueue, TBuf<> outBuf)
{
    CalcSendTokenBufNum(outBuf);
    // scatter 逻辑号 = aivId-1（核 0 为 cumsum 核）：逻辑号 >= moeUsedAivNum_ 的核发送给共享专家
    if ((aivId_ > moeUsedAivNum_) && (sharedExpertRankNum_ != 0)) {
        SendToSharedExpert(inQueue, outBuf);
    } else {
        SendToMoeExpert(inQueue, outBuf);
    }
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CalTokenSendExpertCnt(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline int32_t MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CalExpertCnt(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CcuCalCountsToSegBuf()
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
    tpipe_->InitBuffer(ccuCountsBuf, (moeWordsAlign + sharedExpertRankNum_ * UB_ALIGN_DATA_COUNT) * sizeof(uint32_t));
    LocalTensor<int32_t> countsTensor = ccuCountsBuf.Get<int32_t>();
    LocalTensor<int32_t> sharedTensor = ccuCountsBuf.GetWithOffset<int32_t>(sharedExpertRankNum_ * UB_ALIGN_DATA_COUNT,
                                                                            moeWordsAlign * sizeof(uint32_t));
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
        // 组直方图：每趟 ids-base 平移后 Mins/Maxs 双钳到 [-1,255]，再 int32→u8 低字节直转（基线 a5_ccu
        // TokenScatter 出货同款 Cast 语义：-1→0xFF=组外桶，只落不读）。平移中间量复用 CalCumSum 阶段
        // subExpBuf_/dstExpBuf_（hist 路径下逐专家归约不执行，空闲，容量按 maskCnt 已验证）；同 tensor
        // 相邻 V op 间 PipeBarrier<PIPE_V> 对齐既有惯例
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
            // 低字节截断直转（基线 a5_ccu TokenScatter:831 出货同款 Cast(int32→u8, CAST_NONE)）：取低
            // 8 位，组外/-1 哨兵（0xFFFFFFFF）→ 0xFF=255 恰落组外桶（只落不读）——弃用此前 float NE/Select
            // 值域改写链（Select 掩码编码语义存疑，实测全量 token 疑似落 255 桶致 counts/send head 全零）
            Cast(idsU8Tensor, shiftTensor, RoundMode::CAST_NONE, maskCnt);
            Duplicate<uint16_t>(histTensor, 0, CCU_HIST_BIN_NUM); // 桶清零（基线 TokenScatter:817 同款）
            PipeBarrier<PIPE_V>();
            GetExpertFreq(histTensor, idsU8Tensor, maskCnt); // 内含跨 repeat 寄存器累加与 V 栅栏
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
    // 联调打点（稳定后可删）：计数构造链 UB 中间量 dump——与 send head(GM) 对拍定位断点：
    // ids 正常 + counts 异常 = 直方图/Cast 段问题；ids 异常 = mask 链/输入拷贝问题；
    // 两者正常 + GM 异常 = strided DataCopyPad 写路径问题（S=0 时 head[0..7] 应恰为 counts[0..7]）
    SyncFunc<AscendC::HardEvent::V_S>();
    AscendC::printf("[CCU_MTE] r=%u ids ub: %d %d %d %d %d %d %d %d\n", ccuWinContext_->rankId,
                    validExpertIdsTensor_(0), validExpertIdsTensor_(1), validExpertIdsTensor_(2),
                    validExpertIdsTensor_(3), validExpertIdsTensor_(4), validExpertIdsTensor_(5),
                    validExpertIdsTensor_(6), validExpertIdsTensor_(7));
    AscendC::printf("[CCU_MTE] r=%u counts ub: %u %u %u %u %u %u %u %u\n", ccuWinContext_->rankId,
                    countsTensor.GetValue(0), countsTensor.GetValue(1), countsTensor.GetValue(2),
                    countsTensor.GetValue(3), countsTensor.GetValue(4), countsTensor.GetValue(5),
                    countsTensor.GetValue(6), countsTensor.GetValue(7));

    // 共享段 word0 计数（S=sharedExpertRankNum_，MAX_SHARED_EXPERT_NUM=4 结构性小量）：命中槽 =
    // activeMaskBsCnt_、其余 0（与 CalExpertCnt 共享分支同义）。段间 pitch=segWords×4B 恒 32B 整块倍
    //（segWords 为 8 倍数），与就绪字广播同型走 DataCopy+DataCopyParams 块单位：每段写 word0 所在 32B 块
    //（word1-7 连带写 0，段内 padding 无读者）；源 UB 为 S×32B 块（word0=计数、其余 0）
    if (sharedExpertRankNum_ != 0U) {
        Duplicate<int32_t>(sharedTensor, 0, sharedExpertRankNum_ * UB_ALIGN_DATA_COUNT);
        PipeBarrier<PIPE_V>();
        for (uint32_t expertId = 0; expertId < sharedExpertRankNum_; ++expertId) {
            sharedTensor.SetValue(expertId * UB_ALIGN_DATA_COUNT, CalExpertCnt(expertId, maskCnt));
        }
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopyParams ccuSharedIntriParams{static_cast<uint16_t>(sharedExpertRankNum_), 1U, 0U,
                                            static_cast<uint16_t>(ccuSegWords_ * sizeof(uint32_t) / UB_ALIGN - 1U)};
        DataCopy(ccuSendGlobal[0], sharedTensor, ccuSharedIntriParams);
    }
    // MoE 区整写：src 为 moeId 主序连续块（blockLen=Pe*4B、srcStride=0 紧邻取块），dst 段起点间隔 segWords*4B
    DataCopyExtParams ccuMoeParams{static_cast<uint16_t>(moeExpertRankNum_),
                                   static_cast<uint32_t>(moeExpertNumPerRank_ * sizeof(int32_t)), 0U,
                                   static_cast<uint32_t>((ccuSegWords_ - moeExpertNumPerRank_) * sizeof(int32_t)), 0U};
    DataCopyPad(ccuSendGlobal[sharedExpertRankNum_ * ccuSegWords_], countsTensor, ccuMoeParams);
    PipeBarrier<PIPE_ALL>(); // 全流水排空（联调期由 MTE3_S 单事件同步加强）：段写 GM 全局可见后方可 issue
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CalCntPackAndIssueCcu()
{
    // CCU 模式 aivUsedCumSum_==1：唯一 cumsum 核串行承担 count 面（与前台各核 token 发送并行）——
    // 计数落段 → 等长 AlltoAll + Wait → 全表前缀 scan 与统计输出 + 就绪字广播（scan 尾部单批合出）
    CcuCalCountsToSegBuf();
    AscendC::printf("[CCU_MTE] r=%u cnt core=%u: segs packed (segW=%u)\n", ccuWinContext_->rankId, aivId_,
                    ccuSegWords_); // 联调打点
    // 等长 AlltoAll：每卡发给各对端 ccuSegWords_ 个 u32（纯 counts 段），Wait 后矩阵连续落 recvBuf。
    // 第 5/6 参 strideCount/repeat 与 apace A5-CCU 先例同款：紧凑布局显式传 stride=sendCount（勿传 0，
    // CCU 引擎对 stride=0 的字面语义未证实，可能致各 rank 段重叠、搬运错乱而死等），repeat=1
    ccuCntHandle_ = ccuHccl_.template AlltoAll<true>(ccuSendBufGM_, ccuRecvBufGM_, ccuSegWords_,
                                                     AscendC::HCCL_DATA_TYPE_INT32, ccuSegWords_, 1U);
    AscendC::printf("[CCU_MTE] r=%u cnt core=%u: AlltoAll issued (handle=%llu)\n", ccuWinContext_->rankId,
                    aivId_, // 联调打点
                    (unsigned long long)ccuCntHandle_);
    ccuHccl_.Wait(ccuCntHandle_); // Wait 语义即"所有源的 count 已到"，替代原 MTE count 面 flag 轮询
    AscendC::printf("[CCU_MTE] r=%u cnt core=%u: AlltoAll wait done\n", ccuWinContext_->rankId, aivId_); // 联调打点
    // 联调打点（稳定后可删）：H2 矩阵内容裁决——Wait 返回后、scan 覆写前 dump recvBuf 原始矩阵：首 8 字 +
    // 各源段行首（≤16 行），并附本端 sendBuf 首 8 字对照（本端段引擎回环正确性）。判读：某源段行首异常 =
    // 该段未落位（Wait 契约弱）；前段系统性 0 = core0 validNum 恒 0 的直接证据；recv[本端行]==send 头 =
    // 本端段回环正确。UB 复用 calBeginBuf_（32B，此刻无其他消费者；scan 段 Reset 在其后）
    {
        LocalTensor<uint32_t> dbgTensor = calBeginBuf_.Get<uint32_t>();
        GlobalTensor<uint32_t> dbgRecvGlobal;
        dbgRecvGlobal.SetGlobalBuffer((__gm__ uint32_t*)ccuRecvBufGM_);
        DataCopyExtParams dbgParams{1U, static_cast<uint32_t>(8U * sizeof(uint32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<uint32_t> dbgPad{false, 0U, 0U, 0};
        DataCopyPad(dbgTensor, dbgRecvGlobal, dbgParams, dbgPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        AscendC::printf("[CCU_MTE] r=%u matrix head: %u %u %u %u %u %u %u %u\n", ccuWinContext_->rankId,
                        dbgTensor.GetValue(0), dbgTensor.GetValue(1), dbgTensor.GetValue(2), dbgTensor.GetValue(3),
                        dbgTensor.GetValue(4), dbgTensor.GetValue(5), dbgTensor.GetValue(6), dbgTensor.GetValue(7));
        GlobalTensor<uint32_t> dbgSendGlobal;
        dbgSendGlobal.SetGlobalBuffer((__gm__ uint32_t*)ccuSendBufGM_);
        DataCopyPad(dbgTensor, dbgSendGlobal, dbgParams, dbgPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        AscendC::printf("[CCU_MTE] r=%u send head: %u %u %u %u %u %u %u %u\n", ccuWinContext_->rankId,
                        dbgTensor.GetValue(0), dbgTensor.GetValue(1), dbgTensor.GetValue(2), dbgTensor.GetValue(3),
                        dbgTensor.GetValue(4), dbgTensor.GetValue(5), dbgTensor.GetValue(6), dbgTensor.GetValue(7));
        uint32_t dbgRowNum = (epWorldSize_ < 16U) ? epWorldSize_ : 16U;
        DataCopyExtParams dbgRowParams{1U, sizeof(uint32_t), 0U, 0U, 0U};
        for (uint32_t s = 0U; s < dbgRowNum; ++s) {
            DataCopyPad(dbgTensor, dbgRecvGlobal[s * ccuSegWords_], dbgRowParams, dbgPad);
            SyncFunc<AscendC::HardEvent::MTE2_S>();
            AscendC::printf("[CCU_MTE] r=%u matrix row[%u] head=%u\n", ccuWinContext_->rankId, s,
                            dbgTensor.GetValue(0));
        }
    }
    RunPosRecord(RUNPOS_CALCUMSUM); // 维测打点（count 到齐点，仅 cumsum 核记录）
    CcuScanAndPublishPrefix(); // UB 中转版前缀 scan + 统计输出 + 就绪字广播（直写 GM 版 A/B 已随定型裁剪）
    AscendC::printf("[CCU_MTE] r=%u cnt core=%u: prefix published\n", ccuWinContext_->rankId, aivId_); // 联调打点
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CcuScanAndPublishPrefix()
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
    tpipe_->InitBuffer(ccuReadyBuf, static_cast<uint32_t>(Ceil(aivNum_, 8U)) * SIZE_ALIGN_256); // 就绪字广播 src
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
    // 就绪字广播缓冲随 scan 顺势产出（V 已热）：aivNum_×32B 块全 1.0f——full_mesh CalRecvAndSetFlag 的
    // Duplicate repeat 形态原样复刻（每趟 256B=8 块、repeatTimes=Ceil(aivNum_,8)，趟间连续）
    uint8_t ccuReadyRepTimes = static_cast<uint8_t>(Ceil(aivNum_, 8U));
    Duplicate<float>(ccuReadyTensor, float(1), SIZE_ALIGN_256 / sizeof(float), ccuReadyRepTimes, 1, 8);
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
    // 单条 strided 广播（DataCopyParams 全 32B 块单位，无 GM 侧字节 stride 歧义；full_mesh sumIntriParams
    // 同型）：blockCount=aivNum_、blockLen=1 块、srcStride=0 连续取块、dstStride=512/32−1=15 块——每核 512B
    // 槽（1 cacheline/核）内只写 word16 所在 32B 块；基址=窗口基+routeOffset(1KB 倍)+768KB+64B，32B 对齐成立
    DataCopyParams ccuReadyIntriParams{static_cast<uint16_t>(aivNum_), 1U, 0U,
                                       static_cast<uint16_t>(WIN_ADDR_ALIGN / UB_ALIGN - 1U)};
    DataCopy(GetAivStatusFlagTensor(0)[CCU_READY_WORD_POS], ccuReadyTensor, ccuReadyIntriParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::BuildSendCntFromCcuMatrix()
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CalCumSum()
{
    // 当前核计算并发送所负责专家的token总数
    AscendC::printf("[CCU_MTE] r=%u cumsum core=%u enter\n", ccuWinContext_->rankId, aivId_);
    expertIdsBufSize_ = Ceil(expertIdsCnt_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256; // 支持compareScalar
    tpipe_->InitBuffer(dstExpBuf_, maxSize_);                                                   // BS * K * 4
    tpipe_->InitBuffer(subExpBuf_, maxSize_);                                                   // BS * K * 4
    tpipe_->InitBuffer(gatherMaskTBuf_, expertIdsBufSize_);                                     // BS * K * 4
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsBufSize_);
    if (hasExpertScalesFlag_) {
        tpipe_->InitBuffer(expertScalesBuf_, expertIdsBufSize_);
    }
    workLocalTensor_ = gatherMaskTBuf_.Get<float>();
    ExpIdsCopyAndMaskCal();
    AscendC::printf("[CCU_MTE] r=%u cumsum core=%u: mask done\n", ccuWinContext_->rankId, aivId_); // 联调打点
    // CCU 控制面：count 面收敛单核（aivUsedCumSum_==1），组包后一次等长 AlltoAll 下发并 Wait；接收侧由
    // 各核在 LocalWindowCopy 阶段自读矩阵，就绪扇出由唯一 cumsum 核 Wait 后就绪字广播 + 各核自旋
    // （原 MTE count 面：CalAndSendCntByExp/WaitDispatch/CalRecvAndSetFlag 等多 flag 轮询链已随定型裁剪）
    CalCntPackAndIssueCcu();
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CcuWaitMatrixReady()
{
    // CCU 控制面就绪等待：唯一 cumsum 核（核 0，CCU 控制核）在 Wait 返回、完成全表 scan 并把前缀表
    // 写回 recvBuf 及 sendCounts/expertTokenNums 输出后，向各核状态字本轮 dataState_ 键控字（word16/word17）
    // 写 1.0f，本核自旋自己槽内该字——各核独占一行（控制核写、本核读），无共享 cacheline；命中即前缀表已落
    // recvBuf 且全局可见。标志在状态区（启动清零、跨 launch 持久）而非 workspace（首轮脏值可能假命中）。
    // PipeBarrier<PIPE_ALL> 仅排空本核管道、不跨核，故不能用 barrier 替代本自旋。
    // 联调打点（稳定后可删，全核）：就绪字等待起点——此前仅核 0 打印，非 0 核卡死于此无任何证据
    AscendC::printf("[CCU_MTE] r=%u core=%u wait ready word begin\n", ccuWinContext_->rankId, aivId_);
    GlobalTensor<float> readyWordTensor = GetAivStatusFlagTensor(aivId_);
    LocalTensor<float> readyTensor = scalarBuf_.Get<float>();
    DataCopyExtParams ccuReadyParams{1U, sizeof(float), 0U, 0U, 0U};
    DataCopyPadExtParams<float> ccuReadyPadParams{false, 0U, 0U, 0};
    uint32_t readyPollCnt = 0U; // 联调打点（稳定后可删）：自旋轮询计数
    while (true) {
        DataCopyPad(readyTensor, readyWordTensor[CCU_READY_WORD_POS], ccuReadyParams, ccuReadyPadParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        if (readyTensor.GetValue(0) == float(1)) {
            break;
        }
        ++readyPollCnt;
        if ((readyPollCnt % 2000000U) == 0U) { // 心跳（全核）：证明在自旋而非崩溃
            AscendC::printf("[CCU_MTE] r=%u core=%u still waiting ready word (polls=%u)\n", ccuWinContext_->rankId,
                            aivId_, readyPollCnt);
        }
    }
    // 联调打点（稳定后可删，全核）：就绪字命中（含等待轮数）——与 begin 行数差 = 卡在自旋的核
    AscendC::printf("[CCU_MTE] r=%u core=%u ready word hit (polls=%u)\n", ccuWinContext_->rankId, aivId_, readyPollCnt);
    // 用后自清（full_mesh WaitCumSumFlag 先例）：把本核 word16 复位 0，供下轮发布；单字 DataCopyPad
    // 形态（RunPosRecord 写 [1] 同款，实测已证）；仅本核读写自己槽，无竞争
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<float>(readyTensor, float(0), 1U);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopyPad(readyWordTensor[CCU_READY_WORD_POS], readyTensor, ccuReadyParams); // UB→GM 3 参
    SyncFunc<AscendC::HardEvent::MTE3_S>();
    RunPosRecord(RUNPOS_CUMSUMFLAG); // 维测打点（对齐 MTE WaitCumSumFlag 的软同步完成点，全核记录）
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::SetValidExpertInfo(
    uint32_t expInfoSize, uint32_t &validNum)
{
    // 获取cumSum
    // CCU 控制面：各核只拷自己分核段的前缀表切片（前缀表由唯一 cumsum 核全表 scan 后写回 recvBuf；
    // expertTokenNums 亦由 cumsum 核 scan 后产出，本函数不再产出——原 MTE 路径已随定型裁剪）
    BuildSendCntFromCcuMatrix();
    PipeBarrier<PIPE_ALL>();

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

template <TemplateCcuMteTypeClass>
__aicore__ inline uint32_t MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CheckDataArriveWithFlag(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CopyInAndOut(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline uint32_t
MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::GetLocalWindowSrcDataBlockIdx(uint32_t srcExpertId,
                                                                                                uint32_t localExpertNum)
{
    uint32_t srcDataBlockIdx = srcExpertId % epWorldSize_ * localExpertNum + srcExpertId / epWorldSize_;
    if (!(isShareExpertRankFlag_ || hasElasticInfoFlag_)) {
        srcDataBlockIdx = (srcExpertId + epRankId_) % epWorldSize_ * localExpertNum + srcExpertId / epWorldSize_;
    }
    return srcDataBlockIdx;
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::ClearLocalWindowDataFlags(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::WaitAndFormatOutput(
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
    uint32_t dataWaitCnt = 0U; // 联调打点（稳定后可删）：数据未到轮询计数
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
            ++dataWaitCnt;
            if ((aivId_ == 1U) && ((dataWaitCnt % 2000000U) == 0U)) { // 心跳+槽探针（原核0门控因 validNum==0
                // 早退永不触发，改核1）：读当前轮询槽首 token 的 inline flag 首字（u32 位型）——
                // 0x3f800000(1.0f)=对端已写而求和/数量算错；0=从未写入（IPC 映射或地址错位）；垃圾=槽位
                // 公式错位。附槽地址供与发送侧 dstWinGMTensor 对拍
                GM_ADDR probeAddr =
                    (__gm__ uint8_t *)(windowGM_) + srcDataBlockIdx * expertPerSizeOnWin_ + SPLIT_BLOCK_DATA_SIZE;
                GlobalTensor<float> probeGT;
                probeGT.SetGlobalBuffer((__gm__ float *)probeAddr);
                LocalTensor<uint32_t> probeLT = scalarBuf_.Get<uint32_t>();
                DataCopyExtParams probeParams{1U, sizeof(float), 0U, 0U, 0U};
                DataCopyPadExtParams<float> probePad{false, 0U, 0U, 0};
                DataCopyPad(probeLT.ReinterpretCast<float>(), probeGT, probeParams, probePad);
                SyncFunc<AscendC::HardEvent::MTE2_S>();
                AscendC::printf("[CCU_MTE] r=%u core=%u gather waiting data (polls=%u finish=%u/%u) "
                                "probe: exp=%u slot=%u flag0=0x%x addr=0x%llx\n",
                                ccuWinContext_->rankId, aivId_, dataWaitCnt, finishNum, validNum, srcExpertId,
                                srcDataBlockIdx, probeLT.GetValue(0), (unsigned long long)probeAddr);
            }
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::RunPosRecord(
    const uint32_t runPos)
{
    TBuf<> runPosBuf;
    tpipe_->InitBuffer(runPosBuf, UB_ALIGN);
    dataStateLocalTensor_ = runPosBuf.Get<uint32_t>();
    dataStateLocalTensor_.SetValue(0, runPos);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(selfDataStatusGMTensor_[1], dataStateLocalTensor_, dataStateParams_); // 维测打点
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::LocalWindowCopy()
{
    AscendC::printf("[CCU_MTE] r=%u core=%u localcopy enter\n", ccuWinContext_->rankId,
                    aivId_); // 联调打点（稳定后可删，全核）：函数入口，判所有核均进入本函数
    // 分核负责源专家数量
    tpipe_->Reset();
    TBuf<> cumSumBuf;
    uint32_t rscvNumAlign = Ceil(rscvStatusNum_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN);
    tpipe_->InitBuffer(cumSumBuf, rscvNumAlign);
    sendCntTensor_ = cumSumBuf.Get<int32_t>();
    SplitToCore(rscvStatusNum_, aivNum_, startId_, endId_, sendNum_, true);
    // 软同步：等唯一 cumsum 核（最后核）的 AlltoAll Wait 完成扇出（PipeBarrier 不跨核，就绪字自旋）
    CcuWaitMatrixReady();
    // IPC 单向读回验证（验证后删除，写/读均收敛 0 号核）：仅 rank 0 的核 0 在 Init 期写入，
    // 其余 rank 的核 0 读 slot0 校验 0xA5CC0000——先验证 rank0→其他 的单向 MTE/映射正确性，
    // 规避多写方混叠干扰判读；rank 0 本地无写入者，只打发送侧完成行、跳过读回
    if ((aivId_ == 0U) && (ccuWinContext_->rankId != 0U)) {
        GlobalTensor<int32_t> ipcGT;
        ipcGT.SetGlobalBuffer(
            (__gm__ int32_t*)(GetWinBaseByRankId(static_cast<int32_t>(ccuWinContext_->rankId)) + CCU_IPC_TEST_OFFSET));
        LocalTensor<int32_t> ipcLT = cumSumBuf.Get<int32_t>();
        DataCopyExtParams ipcParams{1U, static_cast<uint32_t>(8U * sizeof(int32_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> ipcPad{false, 0U, 0U, 0};
        DataCopyPad(ipcLT, ipcGT, ipcParams, ipcPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        if (ipcLT(0) == static_cast<int32_t>(0xA5CC0000U)) {
            AscendC::printf("[CCU_MTE][TEST] r=%u core=%u ipc readback: matched=1/1 (sender rank0)\n",
                            ccuWinContext_->rankId, aivId_);
        } else {
            AscendC::printf("[CCU_MTE][TEST] r=%u core=%u ipc miss: sender=0 got=0x%x\n", ccuWinContext_->rankId,
                            aivId_, ipcLT(0));
        }
    } else if (aivId_ == 0U) {
        AscendC::printf("[CCU_MTE][TEST] r=0 core=%u ipc sender only, skip readback\n", aivId_);
    }
    // 联调打点（稳定后可删，全核）：count 面就绪（就绪字已命中、IPC 读回完成）
    AscendC::printf("[CCU_MTE] r=%u core=%u gather: counts ready\n", ccuWinContext_->rankId, aivId_);
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
    AscendC::printf("[CCU_MTE] r=%u core=%u gather: validNum=%u sendNum=%u start=%u\n", ccuWinContext_->rankId, aivId_,
                    validNum, sendNum_, startId_); // 联调打点（稳定后可删）：逐核有效专家数（含分片起点）
    if (validNum == 0) {                           // 本核负责的Expert对应rank收到数据
        return;
    }
    SyncFunc<AscendC::HardEvent::V_S>();
    WaitAndFormatOutput(tBuf, validNum);
    // 联调打点（稳定后可删，全核）：接收/紧凑输出完成（validNum==0 早退的核不经过此处，由 local done 兜底）
    AscendC::printf("[CCU_MTE] r=%u core=%u gather: output done\n", ccuWinContext_->rankId, aivId_);
    RunPosRecord(RUNPOS_ARRIVECNT); // 维测打点
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::TokenActiveMaskCal()
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CalValidBSCnt(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::CalValidExpIdx(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::ExpertActiveMaskInit()
{
    uint32_t axisBSAlign = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t xActivateMaskSize = axisBS_ * (Ceil(axisK_ * sizeof(bool), UB_ALIGN) * UB_ALIGN) * sizeof(half);
    tpipe_->InitBuffer(validBsIndexTBuf_, axisBSAlign);
    uint32_t validBufferSize = expertIdsSize_ > xActivateMaskSize ? expertIdsSize_ : xActivateMaskSize;
    tpipe_->InitBuffer(validExpertIndexBuf_, validBufferSize);
    validBsIndexTensor_ = validBsIndexTBuf_.Get<int32_t>();
    gatherMaskTensor_ = gatherMaskTBuf_.Get<uint32_t>();
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::ExpertActiveMaskCal()
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::MaskZeroComputeExpert(
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::GenerateGatherMaskTensor(
    uint32_t maskCnt)
{
    Duplicate<uint32_t>(gatherMaskTensor_, 0, Ceil(expertIdsCnt_, UB_ALIGN));
    PipeBarrier<PIPE_V>();
    Duplicate<uint32_t>(gatherMaskTensor_, 0xFFFFFFFF, Ceil(maskCnt, UB_ALIGN));
    PipeBarrier<PIPE_V>();
}

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::ZeroComputeExpertMaskCal()
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::ExpIdsCopyAndMaskCal()
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

template <TemplateCcuMteTypeClass>
__aicore__ inline void MoeDistributeDispatchV2A5CcuMte<TemplateCcuMteTypeFunc>::Process()
{
    if ASCEND_IS_AIV { // 全aiv处理
        if (aivId_ == 0U) {
            CalCumSum(); // 核 0 承担 count 面（CCU 客户端 owner，见 SetTilingDataAndCal 注释）
        } else {
            AllToAllDispatch(); // 其余核（1..aivNum-1）all2all发送
        }
        // localWindowCopy中包含reset操作，需确保前面操作完成
        PipeBarrier<PIPE_ALL>();
        LocalWindowCopy(); // 本卡上专家数据连续化，输出expandX/scales/expandIdx
        AscendC::printf("[CCU_MTE] r=%u core=%u local done\n", ccuWinContext_->rankId,
                        aivId_); // 联调打点（稳定后可删）：逐核处理完成（覆盖 sendNum/validNum 早退路径）
        if (aivId_ == 0U) {
            ccuHccl_.Finalize(); // 唯一 cumsum 核（核 0，控制面 issue/Wait 核）负责释放 CCU 资源
            AscendC::printf("[CCU_MTE] r=%u cnt core=%u: ccu finalized\n", ccuWinContext_->rankId, aivId_); // 联调打点
        }
    }
}

} // namespace MoeDistributeDispatchV2A5CcuMteImpl
#endif // MOE_DISTRIBUTE_DISPATCH_V2_A5_CCU_MTE_H
