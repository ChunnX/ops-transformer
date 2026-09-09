/**
 * mgb_kts.h — 葛梦白 kernel timestamp 流水打点（通用探针头，算子无关）
 *
 * 设计目标（按优先级）：
 *  1. 真实准确：每个 span 的 end 时间戳必须落在「工作真完成」之后（异步流水经 X_S 桥回 Scalar
 *     再读时钟）；做不到的点位用 flags 里的边界质量位诚实标注为「下发时刻」。
 *  2. 极致细粒度：事件=带 (wave, expert, arg0, arg1) 参数的 begin/end span，逐核独占存储，
 *     等待与工作分离成独立 span（图上工作染色、等待留白）。
 *  3. 零扰动关闭：MGB_KTS_ENABLE=0 时本头文件不产生任何代码与数据（verify_off.sh 可机器证明）。
 *
 * 存储协议（全部小端 uint32，单位 word=4B）：
 *  运行期：每 (role, core) 在 workspace 尾部保留区（RESERVED_WORKSPACE_SIZE 50MiB）独占一片
 *          record 区，运行中仅本核追加写（WriteGmByPassDCache 绕 D-cache），零竞争零原子。
 *  收敛期（kernel 收尾统一汇合点）：各核把自己的 record 拷入载体（y 输出字节区），
 *          再由唯一发布者（AIC core0）写文件头，PipeBarrier<PIPE_ALL> 后最后写 magic。
 *          解析器读不到合法 magic 一律拒收（半成品不出图）。
 *
 *  文件头 32 words:
 *    [0] magic 0x4D4B5453 ('MKTS'，最后写入)   [1] version        [2] headerWords(32)
 *    [3] recordHeadWords(8) [4] eventWords(8)  [5] roleCount      [6] coreCount
 *    [7] eventCapacity      [8] clockBits(32)  [9] configHash     [10..17] opMeta[8]
 *    [18] cyclesPerUsHint   [19..31] 保留清零
 *  record（recordIndex = roleIndex * coreCount + coreIndex）:
 *    头 8 words: [0] processBegin [1] workEnd [2] eventCount [3] truncated [4..7] counters[4]
 *    事件数组 eventCapacity × 8 words:
 *    [0] pointId  [1] wave  [2] expert  [3] arg0  [4] arg1  [5] beginCycle  [6] endCycle  [7] flags
 *  flags: bit0-1 边界质量（0=causal：begin/end 都经完成边界桥接；1=end 为下发时刻；2=粗略）
 *
 * 时钟：GetSystemCycle() 低 32 位，同一 AI Block 内 AIC/AIV0/AIV1 共时钟可比；
 *       跨 Block 存在固定偏移，渲染层按 Block 分组、组内以本组 AIC processBegin 归零，
 *       禁止跨组横向比较（HTML 顶部明示）。差值一律按模 2^32 处理一次回绕。
 */

#ifndef MGB_KTS_H
#define MGB_KTS_H

#include "mgb_kts_switch.h"

#ifndef MGB_KTS_ENABLE
#define MGB_KTS_ENABLE 0
#endif

#if MGB_KTS_ENABLE

namespace MgbKts {

constexpr uint32_t KTS_MAGIC = 0x4D4B5453U;
constexpr uint32_t KTS_VERSION = 1U;
constexpr uint32_t KTS_HEADER_WORDS = 32U;
constexpr uint32_t KTS_RECORD_HEAD_WORDS = 8U;
constexpr uint32_t KTS_EVENT_WORDS = 8U;
constexpr uint32_t KTS_ROLE_COUNT = 3U; // 0=AIC 1=AIV0 2=AIV1
constexpr uint32_t KTS_MAX_EVENT_CAPACITY = 768U;
constexpr uint32_t KTS_COUNTER_COUNT = 4U;

constexpr uint32_t KTS_FLAG_BOUNDARY_CAUSAL = 0U; // end 经完成边界桥接，含义=工作真完成
constexpr uint32_t KTS_FLAG_BOUNDARY_ISSUE = 1U;  // end 为 Scalar 下发/观察时刻，可能早于流水完成
constexpr uint32_t KTS_FLAG_BOUNDARY_COARSE = 2U;

// 载体所需字节数（host 侧 python 以同一公式独立重算并互证，公式漂移当场炸）。
__aicore__ inline uint64_t CalcKtsRequiredBytes(uint32_t coreCount, uint32_t eventCapacity)
{
    uint64_t recordWords =
        static_cast<uint64_t>(KTS_RECORD_HEAD_WORDS) + static_cast<uint64_t>(eventCapacity) * KTS_EVENT_WORDS;
    return (static_cast<uint64_t>(KTS_HEADER_WORDS) +
            static_cast<uint64_t>(KTS_ROLE_COUNT) * coreCount * recordWords) *
           sizeof(uint32_t);
}

// 载体装得下的最大事件容量（三取小：载体、scratch、硬上限）。
__aicore__ inline uint32_t CalcKtsEventCapacity(uint64_t carrierBytes, uint64_t scratchBytes, uint32_t coreCount)
{
    uint64_t least = carrierBytes < scratchBytes ? carrierBytes : scratchBytes;
    uint64_t headBytes = static_cast<uint64_t>(KTS_HEADER_WORDS) * sizeof(uint32_t);
    if (least <= headBytes) {
        return 0U;
    }
    uint64_t perRecordBytes = (least - headBytes) / (static_cast<uint64_t>(KTS_ROLE_COUNT) * coreCount);
    uint64_t perRecordWords = perRecordBytes / sizeof(uint32_t);
    if (perRecordWords <= KTS_RECORD_HEAD_WORDS) {
        return 0U;
    }
    uint64_t capacity = (perRecordWords - KTS_RECORD_HEAD_WORDS) / KTS_EVENT_WORDS;
    return capacity > KTS_MAX_EVENT_CAPACITY ? KTS_MAX_EVENT_CAPACITY : static_cast<uint32_t>(capacity);
}

// 本核 (roleIndex, coreIndex)。dav_3510 上 AIC 与 AIV 的 GetBlockIdx 值域重叠
// （AIV = blockIdx*subBlockDim+subBlockId ∈ [0,2N)，AIC = blockIdx ∈ [0,N)），必须显式区分。
__aicore__ inline void GetKtsRoleCore(uint32_t &roleIndex, uint32_t &coreIndex)
{
    if constexpr (g_coreType == AscendC::AIC) {
        roleIndex = 0U;
        coreIndex = static_cast<uint32_t>(AscendC::GetBlockIdx());
    } else {
        roleIndex = AscendC::GetSubBlockIdx() == 0U ? 1U : 2U;
        coreIndex = static_cast<uint32_t>(AscendC::GetBlockIdx()) / 2U;
    }
}

__aicore__ inline uint32_t KtsNow()
{
    uint32_t cycle = static_cast<uint32_t>(AscendC::GetSystemCycle());
    return cycle == 0U ? 1U : cycle; // 0 保留为「未写」哨兵
}

/*
 * 每核状态存储。两条硬约束（memory: mc2-kernel-probe-methodology 实测）：
 *  1. aicore 函数里只有 GM_ADDR 型全局变量能编译（类类型、POD 整型数组全报
 *     "global variable is not allowed in aicore function"）→ 状态值借 GM_ADDR 槽存（值语义 cast）。
 *  2. 同 block 的 AIV0/AIV1 共享这份全局存储（AIC 独立）→ 槽按子核分半，否则两子核互踩。
 * 槽位（每半 13 个）：[0]=eventCount [1]=eventCapacity [2]=coreCount [3]=roleIndex [4]=coreIndex
 *                     [5]=processBegin [6]=truncated [7..10]=counters[4] [11]=valid [12]=recordBase(真指针)
 */
constexpr uint32_t KTS_ST_EVENT_COUNT = 0U;
constexpr uint32_t KTS_ST_CAPACITY = 1U;
constexpr uint32_t KTS_ST_CORE_COUNT = 2U;
constexpr uint32_t KTS_ST_ROLE = 3U;
constexpr uint32_t KTS_ST_CORE = 4U;
constexpr uint32_t KTS_ST_PROCESS_BEGIN = 5U;
constexpr uint32_t KTS_ST_TRUNCATED = 6U;
constexpr uint32_t KTS_ST_COUNTER0 = 7U;
constexpr uint32_t KTS_ST_VALID = 11U;
constexpr uint32_t KTS_ST_RECORD_BASE = 12U;
constexpr uint32_t KTS_SLOT_COUNT = 13U;
inline GM_ADDR g_ktsSlots[KTS_SLOT_COUNT * 2U];

__aicore__ inline uint32_t KtsSlotBase()
{
    if constexpr (g_coreType == AscendC::AIC) {
        return 0U; // AIC 存储独立，恒用前半
    } else {
        return static_cast<uint32_t>(AscendC::GetSubBlockIdx()) * KTS_SLOT_COUNT; // AIV0 前半 / AIV1 后半
    }
}

__aicore__ inline uint32_t KtsGet(uint32_t idx)
{
    return static_cast<uint32_t>(reinterpret_cast<uint64_t>(g_ktsSlots[KtsSlotBase() + idx]));
}

__aicore__ inline void KtsSet(uint32_t idx, uint32_t v)
{
    g_ktsSlots[KtsSlotBase() + idx] = reinterpret_cast<GM_ADDR>(static_cast<uint64_t>(v));
}

__aicore__ inline __gm__ uint32_t *KtsRecordPtr()
{
    return reinterpret_cast<__gm__ uint32_t *>(g_ktsSlots[KtsSlotBase() + KTS_ST_RECORD_BASE]);
}

/*
 * 初始化：scratchBase 指向 workspace 尾部保留区（各 record 直接就地写在 scratch 内，
 * 布局与最终载体完全一致，收敛期整片拷贝即可）。在 kernel 入口尽早调用（记录 processBegin）。
 */
__aicore__ inline void KtsInit(GM_ADDR scratchBase, uint32_t coreCount, uint64_t carrierBytes, uint64_t scratchBytes)
{
    KtsSet(KTS_ST_VALID, 0U);
    uint32_t capacity = CalcKtsEventCapacity(carrierBytes, scratchBytes, coreCount);
    if (capacity == 0U) {
        return; // 载体装不下：静默不打点（host 预检会拦，见 host/parse_kts.py）
    }
    KtsSet(KTS_ST_CORE_COUNT, coreCount);
    KtsSet(KTS_ST_CAPACITY, capacity);
    uint32_t roleIndex = 0U;
    uint32_t coreIndex = 0U;
    GetKtsRoleCore(roleIndex, coreIndex);
    KtsSet(KTS_ST_ROLE, roleIndex);
    KtsSet(KTS_ST_CORE, coreIndex);
    uint64_t recordWords = static_cast<uint64_t>(KTS_RECORD_HEAD_WORDS) +
                           static_cast<uint64_t>(capacity) * KTS_EVENT_WORDS;
    uint64_t recordIndex = static_cast<uint64_t>(roleIndex) * coreCount + coreIndex;
    g_ktsSlots[KtsSlotBase() + KTS_ST_RECORD_BASE] =
        scratchBase + (KTS_HEADER_WORDS + recordIndex * recordWords) * sizeof(uint32_t);
    KtsSet(KTS_ST_EVENT_COUNT, 0U);
    KtsSet(KTS_ST_TRUNCATED, 0U);
    for (uint32_t i = 0U; i < KTS_COUNTER_COUNT; ++i) {
        KtsSet(KTS_ST_COUNTER0 + i, 0U);
    }
    KtsSet(KTS_ST_PROCESS_BEGIN, KtsNow());
    KtsSet(KTS_ST_VALID, 1U);
}

// 追加一个 span。容量满只置 truncated，绝不越界写坏相邻 record。
__aicore__ inline void KtsSpan(uint32_t pointId, uint32_t wave, uint32_t expert, uint32_t arg0, uint32_t arg1,
                               uint32_t beginCycle, uint32_t endCycle, uint32_t flags = KTS_FLAG_BOUNDARY_CAUSAL)
{
    if (KtsGet(KTS_ST_VALID) == 0U) {
        return;
    }
    uint32_t count = KtsGet(KTS_ST_EVENT_COUNT);
    if (count >= KtsGet(KTS_ST_CAPACITY)) {
        KtsSet(KTS_ST_TRUNCATED, 1U);
        return;
    }
    __gm__ uint32_t *event = KtsRecordPtr() + KTS_RECORD_HEAD_WORDS + static_cast<uint64_t>(count) * KTS_EVENT_WORDS;
    AscendC::WriteGmByPassDCache(event + 0, pointId);
    AscendC::WriteGmByPassDCache(event + 1, wave);
    AscendC::WriteGmByPassDCache(event + 2, expert);
    AscendC::WriteGmByPassDCache(event + 3, arg0);
    AscendC::WriteGmByPassDCache(event + 4, arg1);
    AscendC::WriteGmByPassDCache(event + 5, beginCycle);
    AscendC::WriteGmByPassDCache(event + 6, endCycle);
    AscendC::WriteGmByPassDCache(event + 7, flags);
    KtsSet(KTS_ST_EVENT_COUNT, count + 1U);
}

// 即时点（duration=0 的标记）。
__aicore__ inline void KtsMark(uint32_t pointId, uint32_t wave, uint32_t expert, uint32_t arg0)
{
    uint32_t now = KtsNow();
    KtsSpan(pointId, wave, expert, arg0, 0U, now, now);
}

__aicore__ inline void KtsCounterAdd(uint32_t counterIdx, uint32_t delta)
{
    if (counterIdx < KTS_COUNTER_COUNT) {
        KtsSet(KTS_ST_COUNTER0 + counterIdx, KtsGet(KTS_ST_COUNTER0 + counterIdx) + delta);
    }
}

// 本核 record 头落盘（收敛点调用；workEnd = 调用时刻）。
__aicore__ inline void KtsSealRecord()
{
    if (KtsGet(KTS_ST_VALID) == 0U) {
        return;
    }
    __gm__ uint32_t *rec = KtsRecordPtr();
    AscendC::WriteGmByPassDCache(rec + 0, KtsGet(KTS_ST_PROCESS_BEGIN));
    AscendC::WriteGmByPassDCache(rec + 1, KtsNow());
    AscendC::WriteGmByPassDCache(rec + 2, KtsGet(KTS_ST_EVENT_COUNT));
    AscendC::WriteGmByPassDCache(rec + 3, KtsGet(KTS_ST_TRUNCATED));
    for (uint32_t i = 0U; i < KTS_COUNTER_COUNT; ++i) {
        AscendC::WriteGmByPassDCache(rec + 4 + i, KtsGet(KTS_ST_COUNTER0 + i));
    }
}

/*
 * 把 scratch 中本核 record 整片拷入载体（逐 word 标量拷贝，收敛期一次性，不在热路径）。
 * carrierBase 与 scratch 布局逐字节一致，各核只拷自己的 record 区，零竞争。
 */
__aicore__ inline void KtsCopyRecordToCarrier(GM_ADDR carrierBase)
{
    if (KtsGet(KTS_ST_VALID) == 0U) {
        return;
    }
    uint64_t recordWords = static_cast<uint64_t>(KTS_RECORD_HEAD_WORDS) +
                           static_cast<uint64_t>(KtsGet(KTS_ST_CAPACITY)) * KTS_EVENT_WORDS;
    uint64_t recordIndex = static_cast<uint64_t>(KtsGet(KTS_ST_ROLE)) * KtsGet(KTS_ST_CORE_COUNT) +
                           KtsGet(KTS_ST_CORE);
    uint64_t offsetWords = KTS_HEADER_WORDS + recordIndex * recordWords;
    __gm__ uint32_t *dst = reinterpret_cast<__gm__ uint32_t *>(carrierBase) + offsetWords;
    __gm__ uint32_t *src = KtsRecordPtr();
    // 只拷有效事件 + record 头（尾部无效槽位不搬，载体残值由 eventCount 界定不被解析）。
    uint64_t liveWords = KTS_RECORD_HEAD_WORDS + static_cast<uint64_t>(KtsGet(KTS_ST_EVENT_COUNT)) * KTS_EVENT_WORDS;
    for (uint64_t w = 0U; w < liveWords; ++w) {
        AscendC::WriteGmByPassDCache(dst + w, AscendC::ReadGmByPassDCache(src + w));
    }
}

/*
 * 唯一发布者（AIC core0）写文件头；调用方保证此前所有核已完成 KtsCopyRecordToCarrier
 * 并有全核同步；本函数内部 PipeBarrier 后最后写 magic。
 * opMeta：算子自定义 8 词（bs/h/topK/worldSize/dedupMode 等），configHash 由调用方给。
 */
__aicore__ inline void KtsPublishHeader(GM_ADDR carrierBase, uint32_t configHash,
                                        const uint32_t opMeta[8], uint32_t cyclesPerUsHint)
{
    if (KtsGet(KTS_ST_VALID) == 0U || KtsGet(KTS_ST_ROLE) != 0U || KtsGet(KTS_ST_CORE) != 0U) {
        return;
    }
    __gm__ uint32_t *h = reinterpret_cast<__gm__ uint32_t *>(carrierBase);
    AscendC::WriteGmByPassDCache(h + 1, KTS_VERSION);
    AscendC::WriteGmByPassDCache(h + 2, KTS_HEADER_WORDS);
    AscendC::WriteGmByPassDCache(h + 3, KTS_RECORD_HEAD_WORDS);
    AscendC::WriteGmByPassDCache(h + 4, KTS_EVENT_WORDS);
    AscendC::WriteGmByPassDCache(h + 5, KTS_ROLE_COUNT);
    AscendC::WriteGmByPassDCache(h + 6, KtsGet(KTS_ST_CORE_COUNT));
    AscendC::WriteGmByPassDCache(h + 7, KtsGet(KTS_ST_CAPACITY));
    AscendC::WriteGmByPassDCache(h + 8, 32U);
    AscendC::WriteGmByPassDCache(h + 9, configHash);
    for (uint32_t i = 0U; i < 8U; ++i) {
        AscendC::WriteGmByPassDCache(h + 10 + i, opMeta[i]);
    }
    AscendC::WriteGmByPassDCache(h + 18, cyclesPerUsHint);
    for (uint32_t i = 19U; i < KTS_HEADER_WORDS; ++i) {
        AscendC::WriteGmByPassDCache(h + i, 0U);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::WriteGmByPassDCache(h + 0, KTS_MAGIC); // magic 最后写：解析器无 magic 即拒收
}

} // namespace MgbKts

#endif // MGB_KTS_ENABLE
#endif // MGB_KTS_H
