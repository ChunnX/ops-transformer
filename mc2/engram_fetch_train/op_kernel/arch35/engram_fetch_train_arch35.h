/*!
 * \file engram_fetch_train_arch35.h
 * \brief engram_fetch_train 算子 arch35 kernel 实现
 *
 * 整体流程（5 步）:
 *   Step 0: 桶排序 — 将 indices 按 owner rank 分组排序，记录原始位置，本地计算 sdispls
 *   Step 1: SendCountPhase — 交换 sendCounts
 *   Step 2: ExchangeIndices — URMA 交换 sortedIndices（quad-buffer 流控）
 *   Step 3: LocalReadTable — 按 recvIndices 从本地 engram 表读取 hidden vector
 *   Step 4: ExchangeToken — URMA 交换 hidden vectors（quad-buffer 流控，counts 互换）
 *   Step 5: ReorderAndSaveCtx — 重排到原始顺序 + 写 numRecvOut
 *
 * Win 区布局（分时复用，在 commBuffer[myRank] 中）:
 *   ├── 状态区 (stateOffset): numRanks × 512B — 每个 sender 一个 flag slot
 *   └── 数据区 (dataOffset): max(counts, indices, data) — 三轮分时复用
 *
 * 通信原语:
 *   发送方: WriteNbi(data → 远端数据区) → Drain → WriteNbi(flag=1 → 远端状态区) → Drain
 *   接收方: 轮询本 rank 状态区 → DataCopy → ReduceSum → 全到齐 → 读数据区 → 清零状态区
 */

#ifndef ENGRAM_FETCH_TRAIN_ARCH35_H
#define ENGRAM_FETCH_TRAIN_ARCH35_H

#include "basic_api/kernel_basic_intf.h"
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_train_tiling_data.h"
#include "../engram_fetch_train.h"
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"

namespace Mc2Kernel {

using namespace AscendC;

// 管线同步辅助函数：SetFlag + WaitFlag，保证指定管线的操作完成
template<AscendC::HardEvent event>
__aicore__ inline void EngramFetchTrainSyncFunc()
{
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

// WriteNbi 带返回值检查：0=成功，-1=失败
#define ENGRAM_WRITE_NBI(hcomm, handle, dst, src, len, tag) do { \
    int32_t ret = (hcomm).WriteNbi((handle), (dst), (src), (len)); \
    if (ret != 0) { \
        printf("[engram_fetch_train][ERROR] WriteNbi failed: rankId=%u, aivId=%u, ret=%d, tag=%s, dst=%lu, src=%lu, len=%lu\n", \
               rankId_, aivId_, ret, tag, (uint64_t)(dst), (uint64_t)(src), (uint64_t)(len)); \
    } \
} while (0)

class EngramFetchTrainArch35 {
public:
    __aicore__ inline EngramFetchTrainArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
                                GM_ADDR permOut, GM_ADDR sendCountsOut, GM_ADDR recvCountsOut,
                                GM_ADDR recvLocalEntryOut, GM_ADDR numRecvOut,
                                GM_ADDR workspaceGM, TPipe *pipe, const EngramFetchTrainTilingData *tilingData);

    __aicore__ inline void Process();

private:
    // ===== 桶排序：将 indices 按 owner rank 分组 =====
    // 输入: indicesLocal — 一批 indices（从 GM 拷到 UB）
    // 输出: rankCounts[r] — 发给 rank r 的条数（累加，支持多批次）
    //       rankOffsets[r] — rank r 数据在 tokenIdxInRank 中的起始偏移
    //       tokenIdxInRank — 按 rank 分组排列的原始位置（批次内 0~batchLen-1）
    __aicore__ inline void ScatterByRank(const LocalTensor<int32_t> &indicesLocal,
                                         const LocalTensor<uint32_t> &rankCounts,
                                         const LocalTensor<uint32_t> &rankOffsets,
                                         const LocalTensor<uint32_t> &tokenIdxInRank,
                                         uint32_t batchLen);

    // GM→UB→GM 的分块拷贝，用 ping-pong buffer 流水化
    __aicore__ inline void LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len);

    // 从 GM 拷一批 indices 到 UB
    __aicore__ inline void CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen);

    // ===== Step 1: sendCount 阶段（不分批，一次交换）=====
    __aicore__ inline void SendCountPhase();

    // ===== Step 2: exchangeIndices 阶段（quad-buffer 流控）=====
    __aicore__ inline void ExchangeIndices();

    // ===== Step 3: 本地读表 =====
    __aicore__ inline void LocalReadTable();

    // ===== Step 4: exchangeToken 阶段（quad-buffer 流控，counts 互换）=====
    __aicore__ inline void ExchangeToken();

    // ===== Step 5: 重排 + 写 numRecvOut =====
    __aicore__ inline void ReorderAndSaveCtx();

    // 轮询本 rank 的 win 状态区，直到所有 sender 的 flag 都到位
    __aicore__ inline void WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount);

    // 清零状态区（供下一轮通信复用）
    __aicore__ inline void ClearStatusFlags(GM_ADDR statusWinBase);

    // 获取远端 rank 的 win 地址: commBuffer[dstRank] + offset
    __aicore__ inline GM_ADDR GetRemoteWinAddr(uint32_t dstRank, uint64_t offset);

    // 获取到 dstRank 的 URMA channel handle
    __aicore__ inline uint64_t GetCommHandle(uint32_t dstRank);

    // 读远端写入本地的计数器（WriteNbi 写入绕过 cache，GetValue 读前需 invalidate）
    __aicore__ inline int32_t ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset, uint32_t srcRank);

    // 通过 WriteNbi 更新远端计数器
    __aicore__ inline void WriteRemoteCounter(uint32_t dstRank, uint64_t counterOffset, int32_t value);

    // ===== 基本资源 =====
    TPipe *tpipe_{nullptr};
    GM_ADDR indicesGM_{nullptr};       // 输入: indices tensor
    GM_ADDR fetchedGM_{nullptr};       // 输出: fetched tensor
    GM_ADDR permOutGM_{nullptr};       // 输出: perm (tokenIdxInRank)
    GM_ADDR sendCountsOutGM_{nullptr}; // 输出: sendCounts
    GM_ADDR recvCountsOutGM_{nullptr}; // 输出: recvCounts
    GM_ADDR recvLocalEntryOutGM_{nullptr}; // 输出: recvLocalEntry (recvIndices)
    GM_ADDR numRecvOutGM_{nullptr};    // 输出: numRecv (totalRecv)
    GM_ADDR workspaceGM_{nullptr};     // user workspace（已跳过 16MB system workspace）
    __gm__ EngramCommContext *ctxPtr_{nullptr};  // 通信上下文（含 commBuffer/hcommHandle）

    // ===== 运行时参数 =====
    uint32_t aivId_{0};           // 当前核 ID
    uint32_t totalBlocks_{1};     // 总核数
    uint32_t rankId_{0};          // 本 rank ID
    uint32_t numRanks_{0};        // 总 rank 数
    int32_t numEntriesPerRank_{0};// 每个 rank 的 entry 数量
    int64_t numTokens_{0};        // 输入 indices 的长度
    int64_t hiddenBytes_{0};      // 一个 hidden vector 的字节数
    int64_t hiddenDim_{0};        // hidden 维度
    uint64_t winSize_{0};         // win 区总大小
    uint64_t maxTokensPerRank_{0};// 每个 rank 的最大 token 数
    int64_t localStorageAddr_{0}; // 本地 engram 表地址（gather 源）

    // ===== Win 区 9 个区域偏移 =====
    uint64_t countFlagOffset_{0};     // sendCount 阶段 flag
    uint64_t indicesFlagOffset_{0};   // exchangeIndices 阶段完成 flag
    uint64_t tokenFlagOffset_{0};     // exchangeToken 阶段完成 flag
    uint64_t indicesWriteOffset_{0};  // 已发送 indices 量计数
    uint64_t indicesReadOffset_{0};   // 已读取 indices 量计数
    uint64_t tokenWriteOffset_{0};    // 已发送 token 量计数
    uint64_t tokenReadOffset_{0};     // 已读取 token 量计数
    uint64_t sendCountOffset_{0};     // 接收到的 sendCount
    uint64_t indicesDataOffset_{0};   // indices 数据区起始
    uint64_t tokenDataOffset_{0};     // token 数据区起始

    // ===== Win 区 slot 大小 =====
    uint64_t indicesSlotSize_{0};     // indices 区 per rank per slot 字节数
    uint64_t tokenSlotSize_{0};       // token 区 per rank per slot 字节数
    uint32_t maxIndicesPerSlot_{0};   // indices 区 per slot 可放 int32 个数
    uint32_t maxTokensPerSlot_{0};    // token 区 per slot 可放 hidden vector 个数

    // ===== 核分配 =====
    uint32_t numSendCores_{0};        // 发送核数量
    uint32_t numRecvCores_{0};        // 接收核数量
    bool isSender_{false};            // 当前核是否为发送核
    bool isReceiver_{false};          // 当前核是否为接收核
    bool isFlagCore_{false};          // 当前核是否为 flag 核

    uint64_t ubSize_{0};          // UB 总大小

    // ===== Workspace 中的 GM 地址（user workspace 内部布局）=====
    // 布局: sdispls | rdispls | sortedIndices | localData | recvData | counterScratch | flagScratch
    GM_ADDR sendCountsGM_{0};      // [numRanks] int32 — 桶排序后各 rank 的发送计数（32B spaced，直接写输出 tensor）
    GM_ADDR recvCountsGM_{0};      // [numRanks] int32 — SendCountPhase 输出（紧凑）
    GM_ADDR sdisplsGM_{0};         // [numRanks] int64 — send 前缀和（Step 0 本地计算，紧凑）
    GM_ADDR rdisplsGM_{0};         // [numRanks] int64 — recv 前缀和（Step 1 接收后计算，紧凑）
    GM_ADDR sortedIndicesGM_{0};   // [numTokens] int32 — 按 rank 分组排序的 indices
    GM_ADDR tokenIdxInRankGM_{0};  // [numTokens] uint32 — 原始位置记录（全局位置）
    GM_ADDR recvIndicesGM_{0};     // [totalRecvUB] int32 — 接收到的 indices（有重复）
    GM_ADDR localDataGM_{0};       // [totalRecvUB × hiddenBytes] — 本地读表输出
    GM_ADDR recvDataGM_{0};        // [numTokens × hiddenBytes] — 交换后的数据（按 rank 分组）
    GM_ADDR counterScratchGM_{0};  // [aivNum×32B] — WriteRemoteCounter per-core 32B 独占区
    GM_ADDR flagScratchGM_{0};     // [32B] float 1.0 — WriteNbi flag 的 GM 源地址

    // ===== UB Buffer =====
    TBuf<> pingBuf_;               // 32KB — GM→UB→GM 拷贝的 ping buffer
    TBuf<> pongBuf_;               // 32KB — GM→UB→GM 拷贝的 pong buffer
    int32_t ppEvtMte2ToMte3_[2] = {0, 0};  // ping-pong MTE2→MTE3 事件 ID
    int32_t ppEvtMte3ToMte2_[2] = {0, 0};  // ping-pong MTE3→MTE2 事件 ID
    TBuf<> indicesBuf_;            // 当前批次的 indices（从 GM 拷入）
    TBuf<> hcommBuf_;              // hcomm 通信上下文（512B）
    TBuf<> rankCountsBuf_;         // [numRanks] uint32 — 各 rank 的发送计数
    TBuf<> rankOffsetsBuf_;        // [numRanks] uint32 — 各 rank 的偏移（桶排序用）
    TBuf<> tokenIdxInRankBuf_;     // [batchSize] uint32 — 按 rank 分组的原始位置（批次内）
    TBuf<> rankIDsBuf_;            // [compareCntMax] int32 — owner rank 计算中间结果
    TBuf<> positionsBuf_;          // [compareCntMax] int32 — 位置序列 [0,1,2,...] / 复用做 sortedLocal
    TBuf<> divisorBuf_;            // [compareCntMax] int32 — 除数 numEntriesPerRank
    TBuf<> maskBuf_;               // [compareCntMax/8] uint8 — CompareScalar 的 mask
    TBuf<> statusBuf_;             // [numRanks×32B] — 轮询状态区（DataCopy 读 win flag 到 UB）/ prevAccum
    TBuf<> tempBuf_;               // 临时 buffer（复用：sumLocal / tmp64）

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;  // URMA 通信对象
    uint32_t indicesBatchSize_{0};  // 桶排序每批处理的 indices 数量（受 UB 容量限制）
};

// 获取远端 rank 的 win 地址
__aicore__ inline GM_ADDR EngramFetchTrainArch35::GetRemoteWinAddr(uint32_t dstRank, uint64_t offset)
{
    return (GM_ADDR)ctxPtr_->commBuffer[dstRank] + offset;
}

// 获取到 dstRank 的 URMA channel handle
// hcommHandle 数组直接用 rankId 作为索引，自身位置(rankId)未设置（跳过）
__aicore__ inline uint64_t EngramFetchTrainArch35::GetCommHandle(uint32_t dstRank)
{
    return ctxPtr_->hcommHandle[dstRank];
}

// ===== Init: 初始化所有资源和 buffer =====
__aicore__ inline void EngramFetchTrainArch35::Init(
    GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
    GM_ADDR permOut, GM_ADDR sendCountsOut, GM_ADDR recvCountsOut,
    GM_ADDR recvLocalEntryOut, GM_ADDR numRecvOut,
    GM_ADDR workspaceGM, TPipe *pipe, const EngramFetchTrainTilingData *tilingData)
{
    printf("[engram_fetch_train] Init start\n");
    tpipe_ = pipe;
    indicesGM_ = indices;
    fetchedGM_ = fetched;
    permOutGM_ = permOut;
    sendCountsOutGM_ = sendCountsOut;
    recvCountsOutGM_ = recvCountsOut;
    recvLocalEntryOutGM_ = recvLocalEntryOut;
    numRecvOutGM_ = numRecvOut;
    workspaceGM_ = workspaceGM;
    aivId_ = GetBlockIdx();
    totalBlocks_ = GetBlockNum();

    // 解析通信上下文
    ctxPtr_ = (__gm__ EngramCommContext *)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;
    printf("[engram_fetch_train] Init: rankId=%u, numRanks=%u, aivId=%u, totalBlocks=%u\n",
           rankId_, numRanks_, aivId_, totalBlocks_);

    // 从 tiling data 读取参数
    numEntriesPerRank_ = tilingData->numEntriesPerRank;
    numTokens_ = tilingData->numTokens;
    hiddenBytes_ = tilingData->hiddenBytes;
    hiddenDim_ = tilingData->hiddenDim;
    maxTokensPerRank_ = tilingData->numMaxTokensPerRank;
    ubSize_ = tilingData->ubSize;
    localStorageAddr_ = tilingData->localStorageAddr;

    winSize_ = tilingData->commBufferSize;

    // 计算 win 区 9 个区域偏移
    uint64_t fixedSize = 8U * numRanks_ * STATE_OFFSET;
    uint64_t remaining = (winSize_ > fixedSize) ? (winSize_ - fixedSize) : 0U;

    countFlagOffset_    = 0;
    indicesFlagOffset_  = numRanks_ * STATE_OFFSET;
    tokenFlagOffset_    = 2U * numRanks_ * STATE_OFFSET;
    indicesWriteOffset_ = 3U * numRanks_ * STATE_OFFSET;
    indicesReadOffset_  = 4U * numRanks_ * STATE_OFFSET;
    tokenWriteOffset_   = 5U * numRanks_ * STATE_OFFSET;
    tokenReadOffset_    = 6U * numRanks_ * STATE_OFFSET;
    sendCountOffset_    = 7U * numRanks_ * STATE_OFFSET;

    // indices 区 = remaining / 50，对齐到 32B
    uint64_t indicesAreaRaw = remaining / INDICES_RATIO;
    uint64_t indicesArea = Ceil(indicesAreaRaw, UB_ALIGN) * UB_ALIGN;
    // token 区 = remaining - indicesArea，对齐到 hiddenBytes
    uint64_t tokenArea = remaining - indicesArea;
    if (hiddenBytes_ > 0) {
        tokenArea = Ceil(tokenArea, static_cast<uint64_t>(hiddenBytes_)) * static_cast<uint64_t>(hiddenBytes_);
    }

    indicesDataOffset_ = sendCountOffset_ + numRanks_ * STATE_OFFSET;
    tokenDataOffset_ = indicesDataOffset_ + indicesArea;

    // per rank per slot
    indicesSlotSize_ = indicesArea / numRanks_ / NUM_SLOTS;
    tokenSlotSize_ = tokenArea / numRanks_ / NUM_SLOTS;
    maxIndicesPerSlot_ = static_cast<uint32_t>(indicesSlotSize_ / sizeof(int32_t));
    maxTokensPerSlot_ = static_cast<uint64_t>(tokenSlotSize_) >= static_cast<uint64_t>(hiddenBytes_)
                        ? static_cast<uint32_t>(tokenSlotSize_ / static_cast<uint64_t>(hiddenBytes_))
                        : 0U;

    // 核分配：前半为发送核，后半为接收核，最后一核为 flag 核
    numSendCores_ = totalBlocks_ / 2U;
    if (numSendCores_ == 0U) { numSendCores_ = 1U; }
    if (totalBlocks_ > 1U) {
        numRecvCores_ = totalBlocks_ - numSendCores_ - 1U;  // 排除 flag 核
    } else {
        numRecvCores_ = 1U;
    }
    if (numRecvCores_ == 0U) { numRecvCores_ = 1U; }  // 至少 1 个接收核
    isSender_ = (aivId_ < numSendCores_) || (totalBlocks_ <= 1U);
    isReceiver_ = (aivId_ >= numSendCores_ && aivId_ < totalBlocks_ - 1U) || (totalBlocks_ <= 1U);
    isFlagCore_ = (aivId_ == totalBlocks_ - 1U);

    printf("[engram_fetch_train] Init: winSize=%lu, indicesSlot=%lu, tokenSlot=%lu, maxIdxPerSlot=%u, maxTokPerSlot=%u, "
           "sendCores=%u, recvCores=%u, isSender=%d, isReceiver=%d, isFlag=%d\n",
           winSize_, indicesSlotSize_, tokenSlotSize_, maxIndicesPerSlot_, maxTokensPerSlot_,
           numSendCores_, numRecvCores_, isSender_, isReceiver_, isFlagCore_);

    if (maxIndicesPerSlot_ == 0U || maxTokensPerSlot_ == 0U) {
        printf("[engram_fetch_train][ERROR] Init: slot too small! maxIdxPerSlot=%u, maxTokPerSlot=%u, "
               "indicesSlotSize=%lu, tokenSlotSize=%lu, hiddenBytes=%ld, winSize=%lu, numRanks=%u\n",
               maxIndicesPerSlot_, maxTokensPerSlot_, indicesSlotSize_, tokenSlotSize_, hiddenBytes_, winSize_, numRanks_);
    }

    // 初始化 hcomm 通信对象
    printf("[engram_fetch_train] Init: before hcomm init\n");
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, HCOMM_INIT_SIZE);
    printf("[engram_fetch_train] Init: hcomm init done\n");

    // 初始化 ping-pong buffer（用于 GM→UB→GM 拷贝流水）
    tpipe_->InitBuffer(pingBuf_, TILE_BYTES);
    tpipe_->InitBuffer(pongBuf_, TILE_BYTES);
    ppEvtMte2ToMte3_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte2ToMte3_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    ppEvtMte3ToMte2_[0] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    ppEvtMte3ToMte2_[1] = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    printf("[engram_fetch_train] Init: pingpong buf done\n");

    // 直接写输出 tensor，不占 workspace:
    //   sendCountsGM_ → sendCountsOutGM_ (32B spaced)
    //   recvCountsGM_ → recvCountsOutGM_ (紧凑)
    //   tokenIdxInRankGM_ → permOutGM_ (紧凑)
    //   recvIndicesGM_ → recvLocalEntryOutGM_ (紧凑)
    sendCountsGM_ = sendCountsOutGM_;
    recvCountsGM_ = recvCountsOutGM_;
    tokenIdxInRankGM_ = permOutGM_;
    recvIndicesGM_ = recvLocalEntryOutGM_;

    // workspace 只保留 sdispls/rdispls/sortedIndices/localData/recvData/counterScratch/flagScratch
    uint64_t wsOffset = 0;
    sdisplsGM_ = workspaceGM_ + wsOffset;                             // [numRanks] int64, 32B spaced
    wsOffset += numRanks_ * UB_ALIGN;
    rdisplsGM_ = workspaceGM_ + wsOffset;                             // [numRanks] int64, 32B spaced
    wsOffset += numRanks_ * UB_ALIGN;
    sortedIndicesGM_ = workspaceGM_ + wsOffset;                       // [numTokens] int32
    wsOffset += static_cast<uint64_t>(numTokens_) * sizeof(int32_t);
    int64_t totalRecvUpperBound = static_cast<int64_t>(maxTokensPerRank_) * static_cast<int64_t>(numRanks_);
    localDataGM_ = workspaceGM_ + wsOffset;                           // [totalRecvUB × hiddenBytes]
    wsOffset += static_cast<uint64_t>(totalRecvUpperBound) * static_cast<uint64_t>(hiddenBytes_);
    recvDataGM_ = workspaceGM_ + wsOffset;                            // [numTokens × hiddenBytes]
    wsOffset += static_cast<uint64_t>(numTokens_) * static_cast<uint64_t>(hiddenBytes_);
    counterScratchGM_ = workspaceGM_ + wsOffset;                      // [aivNum×32B] per-core WriteNbi counter 源
    wsOffset += static_cast<uint64_t>(tilingData->aivNum) * UB_ALIGN;
    flagScratchGM_ = workspaceGM_ + wsOffset;                         // [32B] int32=1
    printf("[engram_fetch_train] Init: wsOffset=%lu, totalRecvUpperBound=%ld\n",
           wsOffset, totalRecvUpperBound);

    printf("[engram_fetch_train] Init: flagScratchGM_=%lu, skipping GM write in Init\n",
           (uint64_t)flagScratchGM_);
    printf("[engram_fetch_train] Init: flagScratch done\n");

    // 初始化桶排序/轮询用的 UB buffer
    uint32_t countsBufSize = Ceil(numRanks_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(rankCountsBuf_, countsBufSize);    // 各 rank 的发送计数
    tpipe_->InitBuffer(rankOffsetsBuf_, countsBufSize);   // 各 rank 的偏移
    // statusBuf_ 用于 WaitAllStatusFlags (DataCopy 读 numRanks×32B) 和 ClearStatusFlags
    uint32_t statusBufSize = Ceil(numRanks_ * STATE_OFFSET, UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    // tempBuf_ 用于 ReduceSum 输出等，至少 32B
    uint32_t tempBufSize = Ceil(numRanks_ * sizeof(int64_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tempBuf_, tempBufSize);
    printf("[engram_fetch_train] Init: counts/status/temp buf done, countsBufSize=%u, statusBufSize=%u\n",
           countsBufSize, statusBufSize);

    // 计算 indices 分批大小（受 UB 容量限制）
    constexpr uint64_t ubReserved = 8U * 1024U;
    uint32_t bytesPerIndice = sizeof(int32_t) + sizeof(uint32_t) + sizeof(int32_t) * 3U + 1U;
    uint64_t usedUb = HCOMM_INIT_SIZE + TILE_BYTES * 2U + countsBufSize * 2U
                    + statusBufSize
                    + tempBufSize;
    uint64_t availableUb = (ubSize_ > usedUb + ubReserved)
                           ? (ubSize_ - usedUb - ubReserved) : 0U;
    uint32_t maxBatchSize = static_cast<uint32_t>(availableUb / bytesPerIndice);
    uint32_t numTokens = static_cast<uint32_t>(numTokens_);
    indicesBatchSize_ = (numTokens <= maxBatchSize) ? numTokens : maxBatchSize;
    if (indicesBatchSize_ == 0U) {
        indicesBatchSize_ = 1U;
    }
    printf("[engram_fetch_train] Init: usedUb=%lu, availableUb=%lu, maxBatchSize=%u, indicesBatchSize=%u\n",
           usedUb, availableUb, maxBatchSize, indicesBatchSize_);

    // 初始化 indices/tokenIdx/rankIDs/positions/divisor/mask buffer
    uint32_t indicesBufSize = Ceil(indicesBatchSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(indicesBuf_, indicesBufSize);
    uint32_t tokenIdxInRankBufSize = Ceil(indicesBatchSize_ * sizeof(uint32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(tokenIdxInRankBuf_, tokenIdxInRankBufSize);
    printf("[engram_fetch_train] Init: indices/tokenIdx buf done\n");

    uint32_t compareCntMax = Ceil(indicesBatchSize_ * sizeof(int32_t), ALIGNED_LEN_256) *
                             ALIGNED_LEN_256 / sizeof(int32_t);
    uint32_t int32BufSize = compareCntMax * sizeof(int32_t);
    printf("[engram_fetch_train] Init: before rankIDs/positions/divisor/mask buf, compareCntMax=%u, int32BufSize=%u\n",
           compareCntMax, int32BufSize);
    tpipe_->InitBuffer(rankIDsBuf_, int32BufSize);    // owner rank 中间结果
    tpipe_->InitBuffer(positionsBuf_, int32BufSize);  // 位置序列 / 复用做 sortedLocal
    tpipe_->InitBuffer(divisorBuf_, int32BufSize);    // 除数
    uint32_t maskBufSize = Ceil(Ceil(compareCntMax, BITS_PER_BYTE), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(maskBuf_, maskBufSize);        // CompareScalar mask

    printf("[engram_fetch_train] Init done: indicesBatchSize=%u, compareCntMax=%u\n",
           indicesBatchSize_, compareCntMax);
}

// 从 GM 拷一批 indices 到 UB
__aicore__ inline void EngramFetchTrainArch35::CopyIndicesToUb(uint32_t indicesBatchStart, uint32_t indicesBatchLen)
{
    GlobalTensor<int32_t> indicesGlobal;
    indicesGlobal.SetGlobalBuffer((__gm__ int32_t *)indicesGM_);
    LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
    DataCopyExtParams params{1U, indicesBatchLen * static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    DataCopyPad(indicesLocal, indicesGlobal[indicesBatchStart], params, pad);
}

// ===== 桶排序: 将 indices 按 owner rank 分组 =====
// 用 Div 计算 ownerRank = index / numEntriesPerRank
// 用 CompareScalar + GatherMask 按 ownerRank 压缩 positions
__aicore__ inline void EngramFetchTrainArch35::ScatterByRank(
    const LocalTensor<int32_t> &indicesLocal,
    const LocalTensor<uint32_t> &rankCounts,
    const LocalTensor<uint32_t> &rankOffsets,
    const LocalTensor<uint32_t> &tokenIdxInRank,
    uint32_t batchLen)
{
    uint32_t compareCntMax = Ceil(batchLen * sizeof(int32_t), ALIGNED_LEN_256) * ALIGNED_LEN_256 / sizeof(int32_t);

    LocalTensor<int32_t> rankIDs = rankIDsBuf_.Get<int32_t>();     // owner rank = index / numEntriesPerRank
    LocalTensor<int32_t> positions = positionsBuf_.Get<int32_t>(); // 位置序列 [0,1,2,...,batchLen-1]
    LocalTensor<int32_t> divisor = divisorBuf_.Get<int32_t>();     // 除数 numEntriesPerRank
    LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();           // 比较结果 mask

    // 生成位置序列 [0, 1, 2, ..., batchLen-1]
    ArithProgression<int32_t>(positions, 0, 1, batchLen);
    // 填充除数
    Duplicate<int32_t>(divisor, numEntriesPerRank_, compareCntMax);

    // 计算 ownerRank = index / numEntriesPerRank
    Duplicate<int32_t>(rankIDs, static_cast<int32_t>(numRanks_), compareCntMax);
    PipeBarrier<PIPE_V>();
    Div<int32_t>(rankIDs, indicesLocal, divisor, batchLen);
    PipeBarrier<PIPE_V>();
    EngramFetchTrainSyncFunc<HardEvent::V_S>();

    // 每个核都遍历全部 ownerRank，确保 rankOffsets 是正确的前缀和
    uint32_t runningOffset = 0;
    for (uint32_t ownerRank = 0; ownerRank < numRanks_; ownerRank++) {
        // 生成 mask: rankIDs == ownerRank
        CompareScalar(mask, rankIDs, static_cast<int32_t>(ownerRank), AscendC::CMPMODE::EQ, compareCntMax);
        PipeBarrier<PIPE_V>();

        // 用 GatherMask 压缩 positions：只保留 ownerRank 匹配的位置
        LocalTensor<int32_t> dstRegion = tokenIdxInRank[runningOffset].ReinterpretCast<int32_t>();
        uint64_t rsvdCnt = 0;
        GatherMask(dstRegion, positions, mask.ReinterpretCast<uint32_t>(), true,
                   batchLen, {1, 1, 0, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();

        // 记录该 rank 的偏移和计数（累加，支持多批次）
        uint32_t count = static_cast<uint32_t>(rsvdCnt);
        rankOffsets.SetValue(ownerRank, runningOffset);
        rankCounts.SetValue(ownerRank, rankCounts.GetValue(ownerRank) + count);
        runningOffset += count;
    }

    EngramFetchTrainSyncFunc<HardEvent::V_S>();
}

// ===== 轮询状态区：等待所有 sender 的 flag 到齐 =====
// 单核读取（最后一核），用 GetValue + DataCacheCleanAndInvalid 刷 cache 即可
__aicore__ inline void EngramFetchTrainArch35::WaitAllStatusFlags(GM_ADDR statusWinBase, uint32_t expectCount)
{
    int32_t sumOfFlag = 0;
    int32_t compareFlag = static_cast<int32_t>(expectCount);

    uint32_t pollCount = 0;
    while (sumOfFlag != compareFlag) {
        sumOfFlag = 0;
        for (uint32_t i = 0; i < numRanks_; i++) {
            GlobalTensor<int32_t> slotGM;
            slotGM.SetGlobalBuffer((__gm__ int32_t *)(statusWinBase + i * STATE_OFFSET));
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(slotGM);
            int32_t flagVal = slotGM.GetValue(0);
            sumOfFlag += flagVal;
            if (pollCount <= 3U || pollCount % 100000U == 0) {
                printf("[engram_fetch_train] WaitAllStatusFlags rankId=%u, poll=%u, srcRank=%u, flag=%d\n",
                       rankId_, pollCount, i, flagVal);
            }
        }
        pollCount++;
        if (pollCount <= 3U || pollCount % 100000U == 0) {
            printf("[engram_fetch_train] WaitAllStatusFlags rankId=%u, poll=%u, sum=%d, expect=%d\n",
                   rankId_, pollCount, sumOfFlag, compareFlag);
        }
    }
}

// ===== 清零状态区（供下一轮通信复用）=====
// 使用 Duplicate + DataCopy(DMA 写) 清零，每个 sender slot 32B
__aicore__ inline void EngramFetchTrainArch35::ClearStatusFlags(GM_ADDR statusWinBase)
{
    LocalTensor<int32_t> cleanTensor = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(cleanTensor, 0, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();

    GlobalTensor<int32_t> statusGM;
    statusGM.SetGlobalBuffer((__gm__ int32_t *)statusWinBase);
    DataCopy(statusGM, cleanTensor, numRanks_ * STATE_OFFSET / sizeof(int32_t));
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
}

// ===== Step 1: sendCount 阶段（不分批，一次交换）=====
__aicore__ inline void EngramFetchTrainArch35::SendCountPhase()
{
    // --- 发送：所有核参与，按 dstRank 分核 ---
    for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += totalBlocks_) {
        // 读 sendCounts[dstRank]
        GM_ADDR sSlot = sendCountsGM_ + dstRank * UB_ALIGN;
        GlobalTensor<int32_t> sSlotGM;
        sSlotGM.SetGlobalBuffer((__gm__ int32_t *)sSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sSlotGM);
        int32_t countVal = sSlotGM.GetValue(0);

        GM_ADDR remoteSendCountAddr = GetRemoteWinAddr(dstRank, sendCountOffset_) + rankId_ * STATE_OFFSET;
        GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, countFlagOffset_) + rankId_ * STATE_OFFSET;

        if (dstRank == rankId_) {
            // 本地：直接写 sendCount区 + countFlag区
            LocalTensor<int32_t> valLocal = statusBuf_.Get<int32_t>();
            valLocal.SetValue(0, countVal);
            EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
            GlobalTensor<int32_t> sendCountGM;
            sendCountGM.SetGlobalBuffer((__gm__ int32_t *)remoteSendCountAddr);
            DataCopyParams valParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
            DataCopyPad(sendCountGM, valLocal, valParams);
            EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

            LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
            Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
            EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();
            GlobalTensor<int32_t> flagGM;
            flagGM.SetGlobalBuffer((__gm__ int32_t *)remoteFlagAddr);
            DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
            EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
        } else {
            // 远端：WriteNbi 发 count + flag
            // count 直接从 sendCountsGM_ 读取（已在 GM 中，WriteNbi 绕过 cache）
            uint64_t handle = GetCommHandle(dstRank);
            GM_ADDR countSrcAddr = sendCountsGM_ + dstRank * UB_ALIGN;

            ENGRAM_WRITE_NBI(hcomm_, handle, remoteSendCountAddr, countSrcAddr, sizeof(int32_t), "SendCount_data");
            hcomm_.Drain(handle);
            ENGRAM_WRITE_NBI(hcomm_, handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET, "SendCount_flag");
            hcomm_.Drain(handle);
        }
    }

    SyncAll<true>();

    // --- 接收：flag核等所有 flag 到齐，读 sendCount，构造 recvCounts/rdispls ---
    if (isFlagCore_) {
        GM_ADDR localFlagBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_] + countFlagOffset_;
        WaitAllStatusFlags(localFlagBase, numRanks_);

        int64_t rSum = 0;
        for (uint32_t r = 0; r < numRanks_; r++) {
            GM_ADDR countAddr = (GM_ADDR)ctxPtr_->commBuffer[rankId_] + sendCountOffset_ + r * STATE_OFFSET;
            GlobalTensor<int32_t> countGM;
            countGM.SetGlobalBuffer((__gm__ int32_t *)countAddr);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(countGM);
            int32_t recvCount = countGM.GetValue(0);

            // 写 recvCounts[r]
            GM_ADDR recvSlot = recvCountsGM_ + r * sizeof(int32_t);
            GlobalTensor<int32_t> recvSlotGM;
            recvSlotGM.SetGlobalBuffer((__gm__ int32_t *)recvSlot);
            recvSlotGM.SetValue(0, recvCount);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(recvSlotGM);

            // 写 rdispls[r]
            GM_ADDR rDisplSlot = rdisplsGM_ + r * UB_ALIGN;
            GlobalTensor<int64_t> rDisplSlotGM;
            rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
            rDisplSlotGM.SetValue(0, rSum);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);

            printf("[engram_fetch_train] SendCountPhase recvCounts[%u]=%d, rdispls=%ld\n", r, recvCount, rSum);
            rSum += static_cast<int64_t>(recvCount);
        }

        ClearStatusFlags(localFlagBase);
    }

    SyncAll<true>();
}

// ===== 辅助：读远端写入本地的计数器 =====
__aicore__ inline int32_t EngramFetchTrainArch35::ReadLocalCounter(GM_ADDR winBase, uint64_t counterOffset,
                                                                uint32_t srcRank)
{
    GM_ADDR counterAddr = winBase + counterOffset + srcRank * STATE_OFFSET;
    GlobalTensor<int32_t> counterGM;
    counterGM.SetGlobalBuffer((__gm__ int32_t *)counterAddr);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(counterGM);
    return counterGM.GetValue(0);
}

// ===== 辅助：通过 WriteNbi 更新远端计数器 =====
// 在 UB 中准备值，DataCopy(DMA 写绕过 cache) 到 GM 的 32B 间距位置，再 WriteNbi 读取
// 避免多核 SetValue + DataCacheCleanAndInvalid 的 cache line 踩踏
__aicore__ inline void EngramFetchTrainArch35::WriteRemoteCounter(uint32_t dstRank,
                                                                uint64_t counterOffset,
                                                                int32_t value)
{
    if (dstRank == rankId_) return;  // 本地不需要 WriteNbi

    // 在 UB 中准备值
    LocalTensor<int32_t> valLocal = statusBuf_.Get<int32_t>();
    valLocal.SetValue(0, value);
    EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();

    // DataCopy 写到 counterScratch 的 32B 间距位置（每核独占 32B，DMA 写绕过 cache）
    GM_ADDR srcAddr = counterScratchGM_ + static_cast<uint64_t>(aivId_) * UB_ALIGN;
    GlobalTensor<int32_t> srcGM;
    srcGM.SetGlobalBuffer((__gm__ int32_t *)srcAddr);
    DataCopyParams valParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
    DataCopyPad(srcGM, valLocal, valParams);
    EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

    // WriteNbi 从该 GM 位置发送到远端
    uint64_t handle = GetCommHandle(dstRank);
    GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, counterOffset) + rankId_ * STATE_OFFSET;
    ENGRAM_WRITE_NBI(hcomm_, handle, remoteCounterAddr, srcAddr, sizeof(int32_t), "WriteRemoteCounter");
    hcomm_.Drain(handle);
}

// ===== Step 2: exchangeIndices 阶段（quad-buffer 流控）=====
__aicore__ inline void EngramFetchTrainArch35::ExchangeIndices()
{
    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    GlobalTensor<int32_t> sortedIndicesGM;
    sortedIndicesGM.SetGlobalBuffer((__gm__ int32_t *)sortedIndicesGM_);
    GlobalTensor<int32_t> recvIndicesGM;
    recvIndicesGM.SetGlobalBuffer((__gm__ int32_t *)recvIndicesGM_);

    // --- 发送核：先处理本地，再处理远端 ---
    if (isSender_) {
        for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
            // 读 sendCounts[dstRank] 和 sdispls[dstRank]
            GM_ADDR sSlot = sendCountsGM_ + dstRank * UB_ALIGN;
            GlobalTensor<int32_t> sSlotGM;
            sSlotGM.SetGlobalBuffer((__gm__ int32_t *)sSlot);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sSlotGM);
            int32_t sendCount = sSlotGM.GetValue(0);

            GM_ADDR sDisplSlot = sdisplsGM_ + dstRank * UB_ALIGN;
            GlobalTensor<int64_t> sDisplSlotGM;
            sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
            int64_t sdispl = sDisplSlotGM.GetValue(0);

            printf("[engram_fetch_train] ExchangeIndices sender: rankId=%u, aivId=%u, dstRank=%u, sendCount=%d, sdispl=%ld\n",
                   rankId_, aivId_, dstRank, sendCount, sdispl);

            if (dstRank == rankId_) {
                // 本地：直接 DataCopy 到 recvIndicesGM
                if (sendCount > 0) {
                    // 读 rdispls[rankId] 确定写入位置
                    GM_ADDR rDisplSlot = rdisplsGM_ + rankId_ * UB_ALIGN;
                    GlobalTensor<int64_t> rDisplSlotGM;
                    rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
                    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
                    int64_t rdispl = rDisplSlotGM.GetValue(0);
                    printf("[engram_fetch_train] ExchangeIndices local copy: rankId=%u, rdispl=%ld, count=%d\n",
                           rankId_, rdispl, sendCount);
                    LocalCopySlice(recvIndicesGM_ + static_cast<uint64_t>(rdispl) * sizeof(int32_t),
                                  sortedIndicesGM_ + static_cast<uint64_t>(sdispl) * sizeof(int32_t),
                                  static_cast<uint64_t>(sendCount) * sizeof(int32_t));
                }
                // 本地 flag 预置 1
                GM_ADDR localFlagAddr = localWinBase + indicesFlagOffset_ + rankId_ * STATE_OFFSET;
                LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
                Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
                EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();
                GlobalTensor<int32_t> flagGM;
                flagGM.SetGlobalBuffer((__gm__ int32_t *)localFlagAddr);
                DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
                EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
                printf("[engram_fetch_train] ExchangeIndices local flag set: rankId=%u\n", rankId_);
                continue;
            }

            // 远端：发送数据
            uint64_t handle = GetCommHandle(dstRank);
            uint32_t totalSent = 0;
            uint32_t localWriteCnt = 0;

            printf("[engram_fetch_train] ExchangeIndices remote send start: rankId=%u, dstRank=%u, sendCount=%d, maxIdxPerSlot=%u\n",
                   rankId_, dstRank, sendCount, maxIndicesPerSlot_);

            while (totalSent < static_cast<uint32_t>(sendCount)) {
                // 流控：仅多核时等待
                if (totalBlocks_ > 1U) {
                    int32_t remoteReadCnt = ReadLocalCounter(localWinBase, indicesReadOffset_, dstRank);
                    uint32_t pollCnt = 0;
                    while (localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= NUM_SLOTS) {
                        remoteReadCnt = ReadLocalCounter(localWinBase, indicesReadOffset_, dstRank);
                        pollCnt++;
                        if (pollCnt <= 3U || pollCnt % 100000U == 0) {
                            printf("[engram_fetch_train] ExchangeIndices send wait: rankId=%u, dstRank=%u, writeCnt=%u, readCnt=%d\n",
                                   rankId_, dstRank, localWriteCnt, remoteReadCnt);
                        }
                    }
                }

                // 计算本块
                uint32_t remaining = static_cast<uint32_t>(sendCount) - totalSent;
                uint32_t chunkLen = (remaining > maxIndicesPerSlot_) ? maxIndicesPerSlot_ : remaining;
                if (chunkLen == 0U) {
                    printf("[engram_fetch_train][ERROR] ExchangeIndices send chunkLen=0: rankId=%u, dstRank=%u, "
                           "sendCount=%d, totalSent=%u, maxIdxPerSlot=%u\n",
                           rankId_, dstRank, sendCount, totalSent, maxIndicesPerSlot_);
                    break;
                }

                // 写入 slot[localWriteCnt % NUM_SLOTS]
                uint64_t slotOffset = indicesDataOffset_ + rankId_ * indicesSlotSize_
                                    + (localWriteCnt % NUM_SLOTS) * indicesSlotSize_;
                GM_ADDR remoteSlotAddr = GetRemoteWinAddr(dstRank, slotOffset);
                GM_ADDR srcAddr = sortedIndicesGM_ + (sdispl + totalSent) * sizeof(int32_t);

                printf("[engram_fetch_train] ExchangeIndices send chunk: rankId=%u, dstRank=%u, chunk=%u, totalSent=%u, slot=%u\n",
                       rankId_, dstRank, chunkLen, totalSent, localWriteCnt % NUM_SLOTS);

                ENGRAM_WRITE_NBI(hcomm_, handle, remoteSlotAddr, srcAddr, chunkLen * sizeof(int32_t), "ExIdx_data");
                hcomm_.Drain(handle);

                // 更新计数器
                localWriteCnt++;
                totalSent += chunkLen;
                WriteRemoteCounter(dstRank, indicesWriteOffset_, static_cast<int32_t>(localWriteCnt));
            }

            printf("[engram_fetch_train] ExchangeIndices send done: rankId=%u, dstRank=%u, totalSent=%u\n",
                   rankId_, dstRank, totalSent);

            // 所有数据发完，写完成 flag
            GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, indicesFlagOffset_) + rankId_ * STATE_OFFSET;
            ENGRAM_WRITE_NBI(hcomm_, handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET, "ExIdx_flag");
            hcomm_.Drain(handle);
        }
    }

    // --- 接收核：循环接收远端数据 ---
    if (isReceiver_) {
        uint32_t recvIdx = aivId_ - numSendCores_;
        for (uint32_t srcRank = recvIdx; srcRank < numRanks_; srcRank += numRecvCores_) {
            if (srcRank == rankId_) continue;  // 本地已由发送核处理

            // 读 recvCounts[srcRank] 和 rdispls[srcRank]
            GM_ADDR rSlot = recvCountsGM_ + srcRank * sizeof(int32_t);
            GlobalTensor<int32_t> rSlotGM;
            rSlotGM.SetGlobalBuffer((__gm__ int32_t *)rSlot);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rSlotGM);
            int32_t recvCount = rSlotGM.GetValue(0);

            GM_ADDR rDisplSlot = rdisplsGM_ + srcRank * UB_ALIGN;
            GlobalTensor<int64_t> rDisplSlotGM;
            rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
            int64_t rdispl = rDisplSlotGM.GetValue(0);

            printf("[engram_fetch_train] ExchangeIndices receiver: rankId=%u, aivId=%u, srcRank=%u, recvCount=%d, rdispl=%ld\n",
                   rankId_, aivId_, srcRank, recvCount, rdispl);

            uint32_t totalReceived = 0;
            uint32_t localReadCnt = 0;

            while (totalReceived < static_cast<uint32_t>(recvCount)) {
                // 流控：检查 write - read > 0
                int32_t remoteWriteCnt = ReadLocalCounter(localWinBase, indicesWriteOffset_, srcRank);
                uint32_t pollCnt = 0;
                while (static_cast<uint32_t>(remoteWriteCnt) - localReadCnt == 0) {
                    remoteWriteCnt = ReadLocalCounter(localWinBase, indicesWriteOffset_, srcRank);
                    pollCnt++;
                    if (pollCnt <= 3U || pollCnt % 100000U == 0) {
                        printf("[engram_fetch_train] ExchangeIndices recv wait: rankId=%u, srcRank=%u, writeCnt=%d, readCnt=%u\n",
                               rankId_, srcRank, remoteWriteCnt, localReadCnt);
                    }
                }

                // 计算本块
                uint32_t remaining = static_cast<uint32_t>(recvCount) - totalReceived;
                uint32_t chunkLen = (remaining > maxIndicesPerSlot_) ? maxIndicesPerSlot_ : remaining;

                printf("[engram_fetch_train] ExchangeIndices recv chunk: rankId=%u, srcRank=%u, chunk=%u, totalRecv=%u, slot=%u\n",
                       rankId_, srcRank, chunkLen, totalReceived, localReadCnt % NUM_SLOTS);

                // 从 slot[localReadCnt % NUM_SLOTS] 读
                uint64_t slotOffset = indicesDataOffset_ + srcRank * indicesSlotSize_
                                    + (localReadCnt % NUM_SLOTS) * indicesSlotSize_;
                GM_ADDR localSlotAddr = localWinBase + slotOffset;

                LocalCopySlice(recvIndicesGM_ + static_cast<uint64_t>(rdispl + totalReceived) * sizeof(int32_t),
                              localSlotAddr,
                              static_cast<uint64_t>(chunkLen) * sizeof(int32_t));

                // 更新计数器
                localReadCnt++;
                totalReceived += chunkLen;
                WriteRemoteCounter(srcRank, indicesReadOffset_, static_cast<int32_t>(localReadCnt));
            }
        }
    }

    // --- flag核：等所有发送完成 + 数据被读取 ---
    if (isFlagCore_) {
        printf("[engram_fetch_train] ExchangeIndices flagCore wait: rankId=%u, aivId=%u\n", rankId_, aivId_);
        GM_ADDR localFlagBase = localWinBase + indicesFlagOffset_;
        WaitAllStatusFlags(localFlagBase, numRanks_);
        printf("[engram_fetch_train] ExchangeIndices flagCore flags done: rankId=%u\n", rankId_);

        // 检查所有 rank 的 write == read（数据都被读取了）
        // for (uint32_t r = 0; r < numRanks_; r++) {
        //     if (r == rankId_) continue;
        //     int32_t writeCnt = ReadLocalCounter(localWinBase, indicesWriteOffset_, r);
        //     int32_t readCnt = ReadLocalCounter(localWinBase, indicesReadOffset_, r);
        //     printf("[engram_fetch_train] ExchangeIndices flagCore check: rankId=%u, r=%u, writeCnt=%d, readCnt=%d\n",
        //            rankId_, r, writeCnt, readCnt);
        //     while (writeCnt != readCnt) {
        //         writeCnt = ReadLocalCounter(localWinBase, indicesWriteOffset_, r);
        //         readCnt = ReadLocalCounter(localWinBase, indicesReadOffset_, r);
        //     }
        // }

        ClearStatusFlags(localFlagBase);
    }

    SyncAll<true>();
    printf("[engram_fetch_train] ExchangeIndices done: rankId=%u, aivId=%u\n", rankId_, aivId_);
}

// ===== Step 4: exchangeToken 阶段（quad-buffer 流控，counts 互换）=====
__aicore__ inline void EngramFetchTrainArch35::ExchangeToken()
{
    GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
    GlobalTensor<uint8_t> recvDataGM;
    recvDataGM.SetGlobalBuffer((__gm__ uint8_t *)recvDataGM_);

    // --- 发送核：先处理本地，再处理远端 ---
    if (isSender_) {
        for (uint32_t dstRank = aivId_; dstRank < numRanks_; dstRank += numSendCores_) {
            // counts 互换：发送用 recvCounts/rdispls
            GM_ADDR rSlot = recvCountsGM_ + dstRank * sizeof(int32_t);
            GlobalTensor<int32_t> rSlotGM;
            rSlotGM.SetGlobalBuffer((__gm__ int32_t *)rSlot);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rSlotGM);
            int32_t sendCount = rSlotGM.GetValue(0);

            GM_ADDR rDisplSlot = rdisplsGM_ + dstRank * UB_ALIGN;
            GlobalTensor<int64_t> rDisplSlotGM;
            rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)rDisplSlot);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
            int64_t rdispl = rDisplSlotGM.GetValue(0);

            printf("[engram_fetch_train] ExchangeToken sender: rankId=%u, aivId=%u, dstRank=%u, sendCount=%d, rdispl=%ld\n",
                   rankId_, aivId_, dstRank, sendCount, rdispl);

            if (dstRank == rankId_) {
                // 本地：直接 DataCopy 到 recvDataGM
                if (sendCount > 0) {
                    // counts 互换：接收用 sdispls
                    GM_ADDR sDisplSlot = sdisplsGM_ + rankId_ * UB_ALIGN;
                    GlobalTensor<int64_t> sDisplSlotGM;
                    sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
                    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
                    int64_t sdispl = sDisplSlotGM.GetValue(0);

                    uint64_t totalBytes = static_cast<uint64_t>(sendCount) * static_cast<uint64_t>(hiddenBytes_);
                    LocalCopySlice(recvDataGM_ + sdispl * hiddenBytes_,
                                  localDataGM_ + rdispl * hiddenBytes_, totalBytes);
                }
                // 本地 flag 预置 1
                GM_ADDR localFlagAddr = localWinBase + tokenFlagOffset_ + rankId_ * STATE_OFFSET;
                LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
                Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
                EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();
                GlobalTensor<int32_t> flagGM;
                flagGM.SetGlobalBuffer((__gm__ int32_t *)localFlagAddr);
                DataCopy(flagGM, flagLocal, STATE_OFFSET / sizeof(int32_t));
                EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
                continue;
            }

            // 远端：发送数据
            // 当 totalBlocks <= 1 时（单核串行），不使用流控
            uint64_t handle = GetCommHandle(dstRank);
            uint32_t totalSent = 0;
            uint32_t localWriteCnt = 0;

            printf("[engram_fetch_train] ExchangeToken remote send start: rankId=%u, dstRank=%u, sendCount=%d, maxTokPerSlot=%u\n",
                   rankId_, dstRank, sendCount, maxTokensPerSlot_);

            while (totalSent < static_cast<uint32_t>(sendCount)) {
                if (totalBlocks_ > 1U) {
                    int32_t remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank);
                    uint32_t pollCnt = 0;
                    while (localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= NUM_SLOTS) {
                        remoteReadCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, dstRank);
                        pollCnt++;
                        if (pollCnt <= 3U || pollCnt % 100000U == 0) {
                            printf("[engram_fetch_train] ExchangeToken send wait: rankId=%u, dstRank=%u, writeCnt=%u, readCnt=%d\n",
                                   rankId_, dstRank, localWriteCnt, remoteReadCnt);
                        }
                    }
                }

                uint32_t remaining = static_cast<uint32_t>(sendCount) - totalSent;
                uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;
                if (chunkLen == 0U) {
                    printf("[engram_fetch_train][ERROR] ExchangeToken send chunkLen=0: rankId=%u, dstRank=%u, "
                           "sendCount=%d, totalSent=%u, maxTokPerSlot=%u\n",
                           rankId_, dstRank, sendCount, totalSent, maxTokensPerSlot_);
                    break;
                }

                uint64_t slotOffset = tokenDataOffset_ + rankId_ * tokenSlotSize_
                                    + (localWriteCnt % NUM_SLOTS) * tokenSlotSize_;
                GM_ADDR remoteSlotAddr = GetRemoteWinAddr(dstRank, slotOffset);
                GM_ADDR srcAddr = localDataGM_ + (rdispl + totalSent) * hiddenBytes_;
                uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

                printf("[engram_fetch_train] ExchangeToken send chunk: rankId=%u, dstRank=%u, chunk=%u, totalSent=%u, slot=%u\n",
                       rankId_, dstRank, chunkLen, totalSent, localWriteCnt % NUM_SLOTS);

                ENGRAM_WRITE_NBI(hcomm_, handle, remoteSlotAddr, srcAddr, dataBytes, "ExTok_data");
                hcomm_.Drain(handle);

                localWriteCnt++;
                totalSent += chunkLen;
                WriteRemoteCounter(dstRank, tokenWriteOffset_, static_cast<int32_t>(localWriteCnt));
            }

            printf("[engram_fetch_train] ExchangeToken send done: rankId=%u, dstRank=%u, totalSent=%u\n",
                   rankId_, dstRank, totalSent);

            // 写完成 flag
            GM_ADDR remoteFlagAddr = GetRemoteWinAddr(dstRank, tokenFlagOffset_) + rankId_ * STATE_OFFSET;
            ENGRAM_WRITE_NBI(hcomm_, handle, remoteFlagAddr, flagScratchGM_, STATE_OFFSET, "ExTok_flag");
            hcomm_.Drain(handle);
        }
    }

    // --- 接收核 ---
    if (isReceiver_) {
        uint32_t recvIdx = aivId_ - numSendCores_;
        for (uint32_t srcRank = recvIdx; srcRank < numRanks_; srcRank += numRecvCores_) {
            if (srcRank == rankId_) continue;

            // counts 互换：接收用 sendCounts/sdispls
            GM_ADDR sSlot = sendCountsGM_ + srcRank * UB_ALIGN;
            GlobalTensor<int32_t> sSlotGM;
            sSlotGM.SetGlobalBuffer((__gm__ int32_t *)sSlot);
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sSlotGM);
            int32_t recvCount = sSlotGM.GetValue(0);

            GM_ADDR sDisplSlot = sdisplsGM_ + srcRank * UB_ALIGN;
            GlobalTensor<int64_t> sDisplSlotGM;
            sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sDisplSlot);
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
            int64_t sdispl = sDisplSlotGM.GetValue(0);

            printf("[engram_fetch_train] ExchangeToken receiver: rankId=%u, aivId=%u, srcRank=%u, recvCount=%d, sdispl=%ld\n",
                   rankId_, aivId_, srcRank, recvCount, sdispl);

            uint32_t totalReceived = 0;
            uint32_t localReadCnt = 0;

            while (totalReceived < static_cast<uint32_t>(recvCount)) {
                int32_t remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank);
                uint32_t pollCnt = 0;
                while (static_cast<uint32_t>(remoteWriteCnt) - localReadCnt == 0) {
                    remoteWriteCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, srcRank);
                    pollCnt++;
                    if (pollCnt <= 3U || pollCnt % 100000U == 0) {
                        printf("[engram_fetch_train] ExchangeToken recv wait: rankId=%u, srcRank=%u, writeCnt=%d, readCnt=%u\n",
                               rankId_, srcRank, remoteWriteCnt, localReadCnt);
                    }
                }

                uint32_t remaining = static_cast<uint32_t>(recvCount) - totalReceived;
                uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;

                printf("[engram_fetch_train] ExchangeToken recv chunk: rankId=%u, srcRank=%u, chunk=%u, totalRecv=%u, slot=%u\n",
                       rankId_, srcRank, chunkLen, totalReceived, localReadCnt % NUM_SLOTS);

                uint64_t slotOffset = tokenDataOffset_ + srcRank * tokenSlotSize_
                                    + (localReadCnt % NUM_SLOTS) * tokenSlotSize_;
                GM_ADDR localSlotAddr = localWinBase + slotOffset;
                uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * hiddenBytes_;

                LocalCopySlice(recvDataGM_ + (sdispl + totalReceived) * hiddenBytes_,
                              localSlotAddr, dataBytes);

                localReadCnt++;
                totalReceived += chunkLen;
                WriteRemoteCounter(srcRank, tokenReadOffset_, static_cast<int32_t>(localReadCnt));
            }
        }
    }

    // --- flag核 ---
    if (isFlagCore_) {
        printf("[engram_fetch_train] ExchangeToken flagCore wait: rankId=%u, aivId=%u\n", rankId_, aivId_);
        GM_ADDR localFlagBase = localWinBase + tokenFlagOffset_;
        WaitAllStatusFlags(localFlagBase, numRanks_);
        printf("[engram_fetch_train] ExchangeToken flagCore flags done: rankId=%u\n", rankId_);

        // for (uint32_t r = 0; r < numRanks_; r++) {
        //     if (r == rankId_) continue;
        //     int32_t writeCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, r);
        //     int32_t readCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, r);
        //     printf("[engram_fetch_train] ExchangeToken flagCore check: rankId=%u, r=%u, writeCnt=%d, readCnt=%d\n",
        //            rankId_, r, writeCnt, readCnt);
        //     while (writeCnt != readCnt) {
        //         writeCnt = ReadLocalCounter(localWinBase, tokenWriteOffset_, r);
        //         readCnt = ReadLocalCounter(localWinBase, tokenReadOffset_, r);
        //     }
        // }

        ClearStatusFlags(localFlagBase);
    }

    SyncAll<true>();
    printf("[engram_fetch_train] ExchangeToken done: rankId=%u, aivId=%u\n", rankId_, aivId_);
}

// ===== Step 4: 重排 + 保存 5 个独立输出 =====
__aicore__ inline void EngramFetchTrainArch35::ReorderAndSaveCtx()
{
    GlobalTensor<uint32_t> tokenIdxInRankGM;
    tokenIdxInRankGM.SetGlobalBuffer((__gm__ uint32_t *)tokenIdxInRankGM_);

    // --- 重排：按 tokenIdxInRank 将 recvData 恢复到 fetched 原始位置 ---
    // recvData[i] → fetched[tokenIdxInRank[i]]
    for (int64_t i = static_cast<int64_t>(aivId_); i < numTokens_; i += static_cast<int64_t>(totalBlocks_)) {
        uint32_t originalIdx = tokenIdxInRankGM.GetValue(static_cast<uint32_t>(i));
        GM_ADDR src = recvDataGM_ + static_cast<uint64_t>(i) * static_cast<uint64_t>(hiddenBytes_);
        GM_ADDR dst = fetchedGM_ + static_cast<uint64_t>(originalIdx) * static_cast<uint64_t>(hiddenBytes_);
        LocalCopySlice(dst, src, static_cast<uint64_t>(hiddenBytes_));
    }

    SyncAll<true>();

    // --- 保存 numRecvOut（仅核 0 执行）---
    // permOut/sendCountsOut/recvCountsOut/recvLocalEntryOut 已在各自步骤直接写入输出 tensor
    if (aivId_ == 0) {
        LocalTensor<int32_t> tmp32 = indicesBuf_.Get<int32_t>();

        // numRecvOut: totalRecv = rdispls[last] + recvCounts[last]
        GM_ADDR lastRDisplSlot = rdisplsGM_ + (numRanks_ - 1) * UB_ALIGN;
        GlobalTensor<int64_t> lastRDisplSlotGM;
        lastRDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)lastRDisplSlot);
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lastRDisplSlotGM);
        GM_ADDR lastRecvSlot = recvCountsGM_ + (numRanks_ - 1) * sizeof(int32_t);
        GlobalTensor<int32_t> lastRecvSlotGM;
        lastRecvSlotGM.SetGlobalBuffer((__gm__ int32_t *)lastRecvSlot);
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lastRecvSlotGM);
        int32_t totalRecv = static_cast<int32_t>(lastRDisplSlotGM.GetValue(0) + lastRecvSlotGM.GetValue(0));

        tmp32.SetValue(0, totalRecv);
        EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
        DataCopyParams gmParamsScalar = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
        GlobalTensor<int32_t> numRecvOutGM;
        numRecvOutGM.SetGlobalBuffer((__gm__ int32_t *)numRecvOutGM_);
        DataCopyPad(numRecvOutGM, tmp32, gmParamsScalar);
    }
}

// ===== Step 3: 本地读表 =====
// 按 recvIndices 从 commBuffer[rankId_]（本地 engram 表）读取 hidden vector → localData
__aicore__ inline void EngramFetchTrainArch35::LocalReadTable()
{
    if (numEntriesPerRank_ == 0 || numTokens_ == 0) {
        return;
    }

    GlobalTensor<int32_t> recvIndicesGM;
    recvIndicesGM.SetGlobalBuffer((__gm__ int32_t *)recvIndicesGM_);

    // 读最后一个 rank 的 rdispls 和 recvCounts 算 totalRecv
    GM_ADDR lastRDisplSlot = rdisplsGM_ + (numRanks_ - 1) * UB_ALIGN;
    GlobalTensor<int64_t> lastRDisplSlotGM;
    lastRDisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)lastRDisplSlot);
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lastRDisplSlotGM);
    GM_ADDR lastRecvSlot = recvCountsGM_ + (numRanks_ - 1) * sizeof(int32_t);
    GlobalTensor<int32_t> lastRecvSlotGM;
    lastRecvSlotGM.SetGlobalBuffer((__gm__ int32_t *)lastRecvSlot);
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(lastRecvSlotGM);
    int64_t totalRecv = lastRDisplSlotGM.GetValue(0) + lastRecvSlotGM.GetValue(0);
    printf("[engram_fetch_train] LocalReadTable: rankId=%u, totalRecv=%ld, aivId=%u\n",
           rankId_, totalRecv, aivId_);

    // 多核并行：每个核处理一部分 recvIndices
    for (int64_t i = static_cast<int64_t>(aivId_); i < totalRecv; i += static_cast<int64_t>(totalBlocks_)) {
        int32_t globalIdx = recvIndicesGM.GetValue(static_cast<uint32_t>(i));
        uint32_t localEntryIdx = static_cast<uint32_t>(globalIdx) - rankId_ * numEntriesPerRank_;
        GM_ADDR src = (GM_ADDR)localStorageAddr_
                      + static_cast<uint64_t>(localEntryIdx) * static_cast<uint64_t>(hiddenBytes_);
        GM_ADDR dst = localDataGM_ + static_cast<uint64_t>(i) * static_cast<uint64_t>(hiddenBytes_);
        LocalCopySlice(dst, src, static_cast<uint64_t>(hiddenBytes_));
    }

    SyncAll<true>();
}

// ===== GM→UB→GM 分块拷贝（ping-pong 流水）=====
// 用于 LocalReadTable 和 ReorderAndSaveCtx 中的 GM→GM 拷贝
__aicore__ inline void EngramFetchTrainArch35::LocalCopySlice(GM_ADDR dst, GM_ADDR src, uint64_t len)
{
    LocalTensor<uint8_t> tmp0 = pingBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> tmp1 = pongBuf_.Get<uint8_t>();
    GlobalTensor<uint8_t> srcGm;
    GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer((__gm__ uint8_t *)src);
    dstGm.SetGlobalBuffer((__gm__ uint8_t *)dst);

    uint32_t tileLen = TILE_BYTES;
    uint64_t off = 0;
    uint32_t tileIdx = 0;
    while (off < len) {
        uint64_t thisLen = (len - off > TILE_BYTES) ? tileLen : (len - off);
        uint32_t bufIdx = tileIdx % 2U;
        LocalTensor<uint8_t> tmp = (bufIdx == 0U) ? tmp0 : tmp1;

        // 从第 3 轮开始，需要等待上一轮的 MTE3→MTE2（UB buffer 被释放）
        if (tileIdx >= 2U) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);
        }

        // GM → UB (MTE2) — 尾块非 32B 对齐时用 DataCopyPad
        DataCopyExtParams mte2Params{1U, static_cast<uint32_t>(thisLen), 0U, 0U, 0U};
        DataCopyPadExtParams<uint8_t> mte2Pad{false, 0, 0, 0};
        DataCopyPad(tmp, srcGm[off], mte2Params, mte2Pad);
        AscendC::SetFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        AscendC::WaitFlag<HardEvent::MTE2_MTE3>(ppEvtMte2ToMte3_[bufIdx]);
        // UB → GM (MTE3) — 尾块非 32B 对齐时用 DataCopyPad
        DataCopyParams mte3Params{1U, static_cast<uint16_t>(thisLen), 0U, 0U};
        DataCopyPad(dstGm[off], tmp, mte3Params);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[bufIdx]);

        off += thisLen;
        tileIdx++;
    }

    // 等待最后两轮的 MTE3 完成
    if (tileIdx >= 1U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[(tileIdx - 1U) % 2U]);
    }
    if (tileIdx >= 2U) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(ppEvtMte3ToMte2_[tileIdx % 2U]);
    }
}

// ===== Process: 主流程 =====
// 调试宏：控制执行到哪一步，逐步验证
//   0 = 只执行 Step 0（桶排序）
//   1 = 执行到 Step 1（sendCount 阶段）
//   2 = 执行到 Step 2（exchangeIndices 阶段）
//   3 = 执行到 Step 3（LocalReadTable）
//   4 = 执行到 Step 4（exchangeToken 阶段）
//   5 = 执行全部步骤（含 ReorderAndSaveCtx）
#ifndef ENGRAM_FETCH_TRAIN_DEBUG_STEP
#define ENGRAM_FETCH_TRAIN_DEBUG_STEP 5
#endif

__aicore__ inline void EngramFetchTrainArch35::Process()
{
    if ASCEND_IS_AIV {
        printf("[engram_fetch_train] Process start: aivId=%u, numEntriesPerRank=%d, numTokens=%ld\n",
               aivId_, numEntriesPerRank_, numTokens_);
        if (numEntriesPerRank_ == 0 || numTokens_ == 0) {
            printf("[engram_fetch_train] Process: early return, numEntriesPerRank=%d, numTokens=%ld\n",
                   numEntriesPerRank_, numTokens_);
            return;
        }

        printf("[engram_fetch_train] Process: workspaceGM_=%lu, flagScratchGM_=%lu\n",
               (uint64_t)workspaceGM_, (uint64_t)flagScratchGM_);

        // 初始化 flagScratchGM_（32B，int32=1 填满，供 WriteNbi 发 flag）
        printf("[engram_fetch_train] Process: before flagScratch write\n");
        if (aivId_ == 0) {
            LocalTensor<int32_t> flagLocal = statusBuf_.Get<int32_t>();
            Duplicate<int32_t>(flagLocal, 1, STATE_OFFSET / sizeof(int32_t));
            EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();
            GlobalTensor<int32_t> flagInit;
            flagInit.SetGlobalBuffer((__gm__ int32_t *)flagScratchGM_);
            DataCopy(flagInit, flagLocal, STATE_OFFSET / sizeof(int32_t));
            EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
        }
        SyncAll<true>();
        printf("[engram_fetch_train] Process: flagScratch initialized\n");

        // 清零 win 区的 4 个计数器区域（indicesWrite/Read, tokenWrite/Read）
        // 放在 SendCountPhase 之前，利用 SendCountPhase 的跨卡 flag 交换作为 barrier，
        // 保证所有卡都清完才开始 ExchangeIndices/ExchangeToken 的通信
        if (aivId_ == 0) {
            LocalTensor<int32_t> cleanLocal = statusBuf_.Get<int32_t>();
            Duplicate<int32_t>(cleanLocal, 0, numRanks_ * STATE_OFFSET / sizeof(int32_t));
            EngramFetchTrainSyncFunc<HardEvent::V_MTE3>();

            uint64_t cleanSize = numRanks_ * STATE_OFFSET / sizeof(int32_t);
            GM_ADDR localWinBase = (GM_ADDR)ctxPtr_->commBuffer[rankId_];
            GlobalTensor<int32_t> writeGM, readGM;
            writeGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + indicesWriteOffset_));
            readGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + indicesReadOffset_));
            DataCopy(writeGM, cleanLocal, cleanSize);
            DataCopy(readGM, cleanLocal, cleanSize);
            GlobalTensor<int32_t> tWriteGM, tReadGM;
            tWriteGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenWriteOffset_));
            tReadGM.SetGlobalBuffer((__gm__ int32_t *)(localWinBase + tokenReadOffset_));
            DataCopy(tWriteGM, cleanLocal, cleanSize);
            DataCopy(tReadGM, cleanLocal, cleanSize);
            EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
        }
        SyncAll<true>();
        printf("[engram_fetch_train] Process: win counters cleared\n");

        // ===== Step 0: 桶排序（多核并行，按 ownerRank 分核）=====
        {
            LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
            LocalTensor<int32_t> rankIDs = rankIDsBuf_.Get<int32_t>();
            LocalTensor<int32_t> positions = positionsBuf_.Get<int32_t>();
            LocalTensor<int32_t> divisor = divisorBuf_.Get<int32_t>();
            LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
            LocalTensor<uint32_t> tokenIdxInRank = tokenIdxInRankBuf_.Get<uint32_t>();

            uint32_t compareCntMax = Ceil(indicesBatchSize_ * sizeof(int32_t), ALIGNED_LEN_256) *
                                     ALIGNED_LEN_256 / sizeof(int32_t);
            Duplicate<int32_t>(divisor, numEntriesPerRank_, compareCntMax);

            // ---- 阶段1: 各核累计自己负责的 ownerRank 的 count ----
            for (uint32_t ownerRank = aivId_; ownerRank < numRanks_; ownerRank += totalBlocks_) {
                uint32_t myCount = 0;

                uint32_t indicesBatchStart = 0;
                while (indicesBatchStart < static_cast<uint32_t>(numTokens_)) {
                    uint32_t indicesBatchLen = indicesBatchSize_;
                    if (indicesBatchStart + indicesBatchLen > static_cast<uint32_t>(numTokens_)) {
                        indicesBatchLen = static_cast<uint32_t>(numTokens_) - indicesBatchStart;
                    }
                    CopyIndicesToUb(indicesBatchStart, indicesBatchLen);
                    EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

                    uint32_t batchCompareCnt = Ceil(indicesBatchLen * sizeof(int32_t), ALIGNED_LEN_256) *
                                               ALIGNED_LEN_256 / sizeof(int32_t);

                    ArithProgression<int32_t>(positions, 0, 1, indicesBatchLen);
                    Duplicate<int32_t>(rankIDs, static_cast<int32_t>(numRanks_), compareCntMax);
                    PipeBarrier<PIPE_V>();
                    Div<int32_t>(rankIDs, indicesLocal, divisor, indicesBatchLen);
                    PipeBarrier<PIPE_V>();
                    EngramFetchTrainSyncFunc<HardEvent::V_S>();

                    CompareScalar(mask, rankIDs, static_cast<int32_t>(ownerRank), AscendC::CMPMODE::EQ,
                                  batchCompareCnt);
                    PipeBarrier<PIPE_V>();

                    uint64_t rsvdCnt = 0;
                    GatherMask(positions, positions, mask.ReinterpretCast<uint32_t>(), true,
                               indicesBatchLen, {1, 1, 0, 0}, rsvdCnt);
                    PipeBarrier<PIPE_V>();
                    EngramFetchTrainSyncFunc<HardEvent::V_S>();

                    myCount += static_cast<uint32_t>(rsvdCnt);
                    indicesBatchStart += indicesBatchLen;
                }

                LocalTensor<int32_t> countLocal = rankIDsBuf_.Get<int32_t>();
                countLocal.SetValue(0, static_cast<int32_t>(myCount));
                EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
                GM_ADDR sendCountSlot = sendCountsGM_ + ownerRank * UB_ALIGN;
                GlobalTensor<int32_t> sendCountsSlotGM;
                sendCountsSlotGM.SetGlobalBuffer((__gm__ int32_t *)sendCountSlot);
                DataCopyParams countParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
                DataCopyPad(sendCountsSlotGM, countLocal, countParams);
                EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();
                printf("[engram_fetch_train] step0 phase1: aivId=%u, ownerRank=%u, count=%u\n",
                       aivId_, ownerRank, myCount);
            }
        }
        SyncAll<true>();

        // core 0 读全量 sendCounts，本地算 sdispls 并写入 GM
        if (aivId_ == 0) {
            LocalTensor<int32_t> countsLocal = rankIDsBuf_.Get<int32_t>();

            GlobalTensor<int32_t> sendCountsGM;
            sendCountsGM.SetGlobalBuffer((__gm__ int32_t *)sendCountsGM_);
            DataCopyExtParams cpParams{1U, static_cast<uint32_t>(numRanks_ * UB_ALIGN), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            DataCopyPad(countsLocal, sendCountsGM, cpParams, cpPad);
            EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

            int64_t sdisplsAccum = 0;
            for (uint32_t r = 0; r < numRanks_; r++) {
                int32_t sCount = countsLocal.GetValue(r * UB_ALIGN / sizeof(int32_t));

                GlobalTensor<int64_t> sdisplsGM;
                sdisplsGM.SetGlobalBuffer((__gm__ int64_t *)sdisplsGM_);
                sdisplsGM.SetValue(r * UB_ALIGN / sizeof(int64_t), sdisplsAccum);
                DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sdisplsGM);

                sdisplsAccum += static_cast<int64_t>(sCount);
            }
        }
        SyncAll<true>();

        // ---- 阶段2: 各核提取自己负责的 ownerRank 数据写 GM ----
        {
            LocalTensor<int32_t> indicesLocal = indicesBuf_.Get<int32_t>();
            LocalTensor<int32_t> rankIDs = rankIDsBuf_.Get<int32_t>();
            LocalTensor<int32_t> positions = positionsBuf_.Get<int32_t>();
            LocalTensor<int32_t> divisor = divisorBuf_.Get<int32_t>();
            LocalTensor<uint8_t> mask = maskBuf_.Get<uint8_t>();
            LocalTensor<uint32_t> tokenIdxInRank = tokenIdxInRankBuf_.Get<uint32_t>();

            uint32_t compareCntMax = Ceil(indicesBatchSize_ * sizeof(int32_t), ALIGNED_LEN_256) *
                                     ALIGNED_LEN_256 / sizeof(int32_t);
            Duplicate<int32_t>(divisor, numEntriesPerRank_, compareCntMax);

            for (uint32_t ownerRank = aivId_; ownerRank < numRanks_; ownerRank += totalBlocks_) {
                GM_ADDR sdisplSlot = sdisplsGM_ + ownerRank * UB_ALIGN;
                GlobalTensor<int64_t> sdisplSlotGM;
                sdisplSlotGM.SetGlobalBuffer((__gm__ int64_t *)sdisplSlot);
                DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sdisplSlotGM);
                int64_t sdispl = sdisplSlotGM.GetValue(0);

                uint32_t prevAccum = 0;
                uint32_t indicesBatchStart = 0;
                while (indicesBatchStart < static_cast<uint32_t>(numTokens_)) {
                    uint32_t indicesBatchLen = indicesBatchSize_;
                    if (indicesBatchStart + indicesBatchLen > static_cast<uint32_t>(numTokens_)) {
                        indicesBatchLen = static_cast<uint32_t>(numTokens_) - indicesBatchStart;
                    }
                    CopyIndicesToUb(indicesBatchStart, indicesBatchLen);
                    EngramFetchTrainSyncFunc<HardEvent::MTE2_S>();

                    uint32_t batchCompareCnt = Ceil(indicesBatchLen * sizeof(int32_t), ALIGNED_LEN_256) *
                                               ALIGNED_LEN_256 / sizeof(int32_t);

                    ArithProgression<int32_t>(positions, 0, 1, indicesBatchLen);
                    Duplicate<int32_t>(rankIDs, static_cast<int32_t>(numRanks_), compareCntMax);
                    PipeBarrier<PIPE_V>();
                    Div<int32_t>(rankIDs, indicesLocal, divisor, indicesBatchLen);
                    PipeBarrier<PIPE_V>();
                    EngramFetchTrainSyncFunc<HardEvent::V_S>();

                    CompareScalar(mask, rankIDs, static_cast<int32_t>(ownerRank), AscendC::CMPMODE::EQ,
                                  batchCompareCnt);
                    PipeBarrier<PIPE_V>();

                    uint64_t rsvdCnt = 0;
                    GatherMask(positions, positions, mask.ReinterpretCast<uint32_t>(), true,
                               indicesBatchLen, {1, 1, 0, 0}, rsvdCnt);
                    PipeBarrier<PIPE_V>();
                    EngramFetchTrainSyncFunc<HardEvent::V_S>();

                    uint32_t batchCount = static_cast<uint32_t>(rsvdCnt);
                    if (batchCount > 0) {
                        uint32_t gmWriteOffset = static_cast<uint32_t>(sdispl) + prevAccum;

                        LocalTensor<uint32_t> tokenIdxLocal = rankIDsBuf_.Get<uint32_t>();
                        LocalTensor<int32_t> sortedLocal = positionsBuf_.Get<int32_t>();
                        for (uint32_t i = 0; i < batchCount; i++) {
                            uint32_t origPos = positions.GetValue(i);
                            tokenIdxLocal.SetValue(i, origPos + indicesBatchStart);
                            sortedLocal.SetValue(i, indicesLocal.GetValue(origPos));
                        }

                        EngramFetchTrainSyncFunc<HardEvent::S_MTE3>();
                        GlobalTensor<uint32_t> tokenIdxInRankGM;
                        tokenIdxInRankGM.SetGlobalBuffer((__gm__ uint32_t *)tokenIdxInRankGM_);
                        DataCopyParams tokenIdxGmParams = {1U, static_cast<uint16_t>(batchCount * sizeof(uint32_t)), 0U, 0U};
                        DataCopyPad(tokenIdxInRankGM[gmWriteOffset], tokenIdxLocal, tokenIdxGmParams);

                        GlobalTensor<int32_t> sortedIndicesGM;
                        sortedIndicesGM.SetGlobalBuffer((__gm__ int32_t *)sortedIndicesGM_);
                        DataCopyParams sortedGmParams = {1U, static_cast<uint16_t>(batchCount * sizeof(int32_t)), 0U, 0U};
                        DataCopyPad(sortedIndicesGM[gmWriteOffset], sortedLocal, sortedGmParams);
                        EngramFetchTrainSyncFunc<HardEvent::MTE3_S>();

                        prevAccum += batchCount;
                    }

                    indicesBatchStart += indicesBatchLen;
                }
            }
        }
        SyncAll<true>();

        // 打印 Step 0 结果
        if (aivId_ == 0) {
            GlobalTensor<int32_t> sortedIndicesGM;
            sortedIndicesGM.SetGlobalBuffer((__gm__ int32_t *)sortedIndicesGM_);
            GlobalTensor<uint32_t> tokenIdxInRankGM;
            tokenIdxInRankGM.SetGlobalBuffer((__gm__ uint32_t *)tokenIdxInRankGM_);
            for (uint32_t r = 0; r < numRanks_; r++) {
                GM_ADDR sSlot = sendCountsGM_ + r * UB_ALIGN;
                GlobalTensor<int32_t> sSlotGM;
                sSlotGM.SetGlobalBuffer((__gm__ int32_t *)sSlot);
                int32_t sc = sSlotGM.GetValue(0);
                printf("[engram_fetch_train] === step0 result: sendCounts[%u]=%d ===\n", r, sc);
            }
            for (int64_t i = 0; i < numTokens_; i++) {
                printf("[engram_fetch_train] === step0 result: sortedIndices[%ld]=%d, tokenIdxInRank[%ld]=%u ===\n",
                       i, sortedIndicesGM.GetValue(static_cast<uint32_t>(i)),
                       i, tokenIdxInRankGM.GetValue(static_cast<uint32_t>(i)));
            }
        }
        SyncAll<true>();

#if ENGRAM_FETCH_TRAIN_DEBUG_STEP >= 1
        // ===== Step 1: sendCount 阶段 =====
        printf("[engram_fetch_train] start SendCountPhase: rankId=%u, aivId=%u\n", rankId_, aivId_);
        SendCountPhase();
        printf("[engram_fetch_train] SendCountPhase done: rankId=%u, aivId=%u\n", rankId_, aivId_);
#endif

#if ENGRAM_FETCH_TRAIN_DEBUG_STEP >= 2
        // ===== Step 2: exchangeIndices 阶段 =====
        printf("[engram_fetch_train] start ExchangeIndices: rankId=%u, aivId=%u\n", rankId_, aivId_);
        ExchangeIndices();
        printf("[engram_fetch_train] ExchangeIndices done: rankId=%u, aivId=%u\n", rankId_, aivId_);
#endif

#if ENGRAM_FETCH_TRAIN_DEBUG_STEP >= 3
        // ===== Step 3: LocalReadTable =====
        printf("[engram_fetch_train] start LocalReadTable: rankId=%u, aivId=%u\n", rankId_, aivId_);
        LocalReadTable();
        printf("[engram_fetch_train] LocalReadTable done: rankId=%u, aivId=%u\n", rankId_, aivId_);
#endif

#if ENGRAM_FETCH_TRAIN_DEBUG_STEP >= 4
        // ===== Step 4: exchangeToken 阶段 =====
        printf("[engram_fetch_train] start ExchangeToken: rankId=%u, aivId=%u\n", rankId_, aivId_);
        ExchangeToken();
        printf("[engram_fetch_train] ExchangeToken done: rankId=%u, aivId=%u\n", rankId_, aivId_);
#endif

#if ENGRAM_FETCH_TRAIN_DEBUG_STEP >= 5
        // ===== Step 5: ReorderAndSaveCtx =====
        printf("[engram_fetch_train] start ReorderAndSaveCtx: rankId=%u, aivId=%u\n", rankId_, aivId_);
        ReorderAndSaveCtx();
        printf("[engram_fetch_train] ReorderAndSaveCtx done: rankId=%u, aivId=%u\n", rankId_, aivId_);
#endif
    }
}

} // namespace Mc2Kernel

#endif
