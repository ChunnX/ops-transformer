// sa2_ub_plan.h — SageAttention2 v2 UB / L1 / L0 槽位计划（偏移链唯一来源）+ DeviceRes 资源封装 + Fence 助手。
//
// 推导依据：DESIGN_SPEC §5.1/§5.2/§3.3；槽深 = 2（支撑 QK(j+1)∥softmax(j) 与 PV(j)∥折叠(j-1) 两级 lookahead，
// 3 深以上收益递减且 UB 预算不允许）。偏移链终止锚点：per_warp=152448、per_thread=217984、L1 End=119424、
// L0C 四分区 0/32K/64K/96K（附录 C.2/C.3 口径，全部 static_assert 冻结）。
#ifndef SA2_UB_PLAN_H_
#define SA2_UB_PLAN_H_

#include <cstdint>

#include "sa2_events.h"
#include "sa2_spec.h"

#if __has_include("kernel_operator.h")
#include "kernel_operator.h"
#elif __has_include("sa2_ascendc_hoststub.h")
#include "sa2_ascendc_hoststub.h"
#endif

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __ubuf__
#define __ubuf__
#endif

namespace sa2 {

// ============================ UB 槽位（字节偏移；BR=64、BC=128、DP=128） ============================

// 槽大小（§5.1 逐项）：
constexpr int64_t kUbScoreWarpSlot = kStreamRows * kKvStep * 2;    // 16384：score FP16（§4.1 score 域 FP16）
constexpr int64_t kUbScoreThreadSlot = kStreamRows * kKvStep * 4;  // 32768：per_thread 的 S 为 FP32 组合域（R.2.2）
constexpr int64_t kUbPSlot = 32 * 272;                             // 8704（≈8.5K）：E4M3 NZ 16×16 分形 ×32 块，
                                                                   //   每块 256B + 16B pad 行（D-R6 防 bank conflict）
constexpr int64_t kUbPvSlot = kStreamRows * kMaxHeadDim * 4;       // 32768：PV 部分和 FP32（L0C→UB）
constexpr int64_t kUbOAcc = kStreamRows * kMaxHeadDim * 4;         // 32768：O 累加器 FP32
// D-R2 设计依据：L0C 无法原地乘标量、关 HF32 后 Cube 重缩放不可接受 ⇒ O 驻 AIV UB（每流一份），
// L0C 只承担块间累加；rescale 触发时序 = 先 copy-out、MMA 带清零标志重开（M2.4 要点）。
constexpr int64_t kUbSegSlot = kStreamRows * kKvStep * 2;          // 16384：per_thread 分段组合缓冲 FP16（仅 GRAN=THREAD），
                                                                   //   逐段接收 DEQF16(×2^-14) 偏部分和，VF 累进 FP32 score 槽

// 状态区（每流 1312B ×2 + 共合 scratch 1344B = 3968B）：
// D-R1 三状态 online softmax：m'_o（running max，S 域已含 s）+ denom + O；θ ≡ lnPDomain 触发 rescale。
constexpr int64_t kStateRowStatOff = 0;    // 512B：64 行 × 2 lane × fp32；lane0 = m'_o，lane1 = 旗标
                                           //   （D-R1：lane1 = “本块 Δmax > θ，需要 rescale”，float 1.0/0.0）
constexpr int64_t kStateDenomOff = 512;    // 256B：64 fp32，分母（累加反量化后 E4M3 码字，§4.5）
constexpr int64_t kStateCorrOff = 768;     // 256B：64 fp32，corr_i = q_i·μ（Q 量化顺路算出，LSE 校正 §4.6）
constexpr int64_t kStateAlphaOff = 1024;   // 288B：65 lane fp32（64 alpha + lane64 块级旗标），260B 取 32B 对齐
constexpr int64_t kStreamStateBytes = 1312;                       // 512+256+256+288
constexpr int64_t kUbSharedScratch = 1344;                        // 归约暂存 / 预除常数（D-R1 阈值比较 1/θ）/ QH 系数暂存
constexpr int64_t kUbStateBytes = 2 * kStreamStateBytes + kUbSharedScratch;  // 3968

// 链终锚点（附录 C.2）：
constexpr int64_t kUbDerivedEndPerWarp = 2 * kUbScoreWarpSlot + 2 * kUbPSlot + 2 * kUbPvSlot + kUbOAcc + kUbStateBytes;
constexpr int64_t kUbDerivedEndPerThread =
    2 * kUbScoreThreadSlot + 2 * kUbPSlot + 2 * kUbPvSlot + kUbOAcc + 2 * kUbSegSlot + kUbStateBytes;
static_assert(kUbDerivedEndPerWarp == kUbEndPerWarp, "per_warp derived UB matches spec constant 152448");
static_assert(kUbDerivedEndPerThread == kUbEndPerThread, "per_thread derived UB matches spec constant 217984");
static_assert(kUbScoreWarpSlot == 16384 && kUbScoreThreadSlot == 32768, "score 槽大小（FP16 / FP32 组合域）");
static_assert(kUbPSlot == 8704 && kUbPvSlot == 32768 && kUbOAcc == 32768 && kUbSegSlot == 16384,
              "P/PV/O/分段槽大小冻结");
static_assert(kStreamStateBytes == 1312 && kUbSharedScratch == 1344 && kUbStateBytes == 3968, "状态区 3968B 冻结");
static_assert(kUbEndPerWarp == 152448, "per_warp UB 终址 = 152448（附录 C.2 锚点）");
static_assert(kUbEndPerThread == 217984, "per_thread UB 终址 = 217984（附录 C.2 锚点；R.3.1 ≈217KB 口径）");
static_assert(kUbEndPerWarp <= kUbHighWater && kUbEndPerThread <= kUbHighWater, "UB 预算 ≤ 248KB 高水位");

// ============================ L1 槽位（AIC 侧共享；BR=64、BC=128、DP=128） ============================

constexpr int64_t kL1KStepSlot = kKvStep * kMaxHeadDim;  // 16384：K 步 tile int8（GM→L1 预取，MTE1 消费）
constexpr int64_t kL1VStepSlot = kKvStep * kMaxHeadDim;  // 16384：V 步 tile int8
constexpr int64_t kL1VScaleSlot = (kKvStep / kMxGroupTokens) * kMaxHeadDim * 2;  // 1024：未打包上界预留；
                                                                                 //   落槽按 D-R4 打包后 512B/步
constexpr int64_t kL1Q8Slot = kStreamRows * kMaxHeadDim;          // 8192：Q 常驻 int8（每流一份，任务级生命周期）
constexpr int64_t kL1QScaleSlot = kStreamRows * kSegsFull * 2;    // 1024：QH 因子（per_thread 64×8 fp16 恰满；
                                                                  //   per_warp 用前 128B）；AIV 经 L1 路径读写
constexpr int64_t kL1PNzSlot = (kStreamRows / 16) * (kKvStep / 16) * 16 * 32;  // 16384：P NZ 全 32B 分形行图像
                                                                              //   （32 块 × 16 行 × 32B，MTE1 取数粒度）
constexpr int64_t kL1KMeanSlot = kMaxHeadDim * 4;  // 512：μ_c fp32 暂存（Q 量化 corr 与 prep 共用入口）
constexpr int64_t kL1AuxSlot = 128;                // 128：32×fp32 共享标量（预除常数 / g·sign(s) 等，D-R1）

// 偏移链（附录 C.3，End = 119424）：
constexpr int64_t kL1KStep0 = 0;                    // 0 .. 16384
constexpr int64_t kL1KStep1 = 16384;                // 16384 .. 32768
constexpr int64_t kL1VStep0 = 32768;                // 32768 .. 49152
constexpr int64_t kL1VStep1 = 49152;                // 49152 .. 65536
constexpr int64_t kL1VScale0 = 65536;               // 65536 .. 66560
constexpr int64_t kL1VScale1 = 66560;               // 66560 .. 67584
constexpr int64_t kL1Q8S0 = 67584;                  // 67584 .. 75776（流 0 q8）
constexpr int64_t kL1QScale0 = 75776;               // 75776 .. 76800（流 0 QH 因子）
constexpr int64_t kL1Q8S1 = 76800;                  // 76800 .. 84992（流 1 q8）
constexpr int64_t kL1QScale1 = 84992;               // 84992 .. 86016（流 1 QH 因子）
constexpr int64_t kL1PNz0 = 86016;                  // 86016 .. 102400
constexpr int64_t kL1PNz1 = 102400;                 // 102400 .. 118784
constexpr int64_t kL1KMean = 118784;                // 118784 .. 119296
constexpr int64_t kL1Aux = 119296;                  // 119296 .. 119424
constexpr int64_t kL1End = kL1Aux + kL1AuxSlot;     // 119424

static_assert(kL1KStepSlot == 16384 && kL1VScaleSlot == 1024 && kL1Q8Slot == 8192 && kL1QScaleSlot == 1024 &&
                  kL1PNzSlot == 16384 && kL1KMeanSlot == 512 && kL1AuxSlot == 128,
              "L1 槽大小冻结（§5.2）");
static_assert(kL1KStep1 == kL1KStep0 + kL1KStepSlot && kL1VStep0 == kL1KStep1 + kL1KStepSlot &&
                  kL1VStep1 == kL1VStep0 + kL1VStepSlot && kL1VScale0 == kL1VStep1 + kL1VStepSlot &&
                  kL1VScale1 == kL1VScale0 + kL1VScaleSlot && kL1Q8S0 == kL1VScale1 + kL1VScaleSlot &&
                  kL1QScale0 == kL1Q8S0 + kL1Q8Slot && kL1Q8S1 == kL1QScale0 + kL1QScaleSlot &&
                  kL1QScale1 == kL1Q8S1 + kL1Q8Slot && kL1PNz0 == kL1QScale1 + kL1QScaleSlot &&
                  kL1PNz1 == kL1PNz0 + kL1PNzSlot && kL1KMean == kL1PNz1 + kL1PNzSlot &&
                  kL1Aux == kL1KMean + kL1KMeanSlot && kL1End == kL1Aux + kL1AuxSlot,
              "L1 偏移链连续无空洞");
static_assert(kL1End == 119424, "L1 End = 119424（附录 C.3 锚点）");
static_assert(kL1End <= kL1Size, "L1 预算 ≤ 512KB（Bc 升 256 余量充足，§5.2）");

// M2c-B4 V MX Scale L1 缓存区基址（在常规 L1 槽之后，128KB 边界对齐）
constexpr int64_t kL1MxScaleCacheBase = 131072;
static_assert(kL1MxScaleCacheBase >= kL1End, "MxScale L1 缓存区在常规 L1 槽之后");
static_assert(kL1MxScaleCacheBase + kL1MxScaleCacheBytes <= kL1Size, "MxScale L1 缓存区完全容纳在 512KB L1 内");

// P 的 L1 NZ 分形索引（冻结口径唯一算式；B-26：engine 文件只取此处，禁止自行书写偏移算式）。
// 图像 = kL1PNzSlot = (64/16 行块)×(128/16 列块)×16 行×32B：16×16 分形、行块主序
//（4 行块 × 8 列块 = 32 块，BC=kKvStep=128 即 8 个列块——非 per_warp 64 列的 4 块）；
// 分形行 32B = 16 个 E4M3 码 + 16B pad（D-R6 防 bank conflict，MTE1 取数粒度）。
// 行块项 row/16 不可省：缺此则 r≥16 的行写穿后续列块（跨块碰撞）。
// SA2_AICORE 标注：设备构建（-xcce 拆分编译）下被 AIV __simd_vf__ 函数运行期调用，
// 未标注会被视作 __host__ 函数产生 "call to __host__ function from __aicore__" 硬错误；
// static_assert 走常量求值不受影响（同 sa2_events.h Sa2FlagId 先例）。
SA2_AICORE constexpr int64_t PNzFractalIndex(int64_t row, int64_t col) {
  const int64_t rowBlk = row / 16;  // 64 行 → 4 行块（行块项 = row/16）
  const int64_t colBlk = col / 16;  // BC=128 → 8 个列块（16×16 分形）
  const int64_t inRow = row % 16;
  const int64_t inCol = col % 16;
  return ((rowBlk * (kKvStep / 16)) + colBlk) * (16 * 32) + inRow * 32 + inCol;
}
static_assert(PNzFractalIndex(0, 0) == 0 && PNzFractalIndex(15, 15) == 15 * 32 + 15,
              "P NZ 首分形：行内 32B 步长（16 码 + 16B pad）");
static_assert(PNzFractalIndex(16, 0) == 8 * 512 && PNzFractalIndex(0, 16) == 512,
              "P NZ 行块项 row/16 与列块项 col/16 均入高位（r≥16 不再跨块碰撞）");
static_assert(PNzFractalIndex(63, 127) == 31 * 512 + 15 * 32 + 15 && PNzFractalIndex(63, 127) == kL1PNzSlot - 17,
              "P NZ 终值 16367 紧贴槽尾 16384（pad 行保留在尾）");
constexpr int64_t kPNzFractalBlocks = (kStreamRows / 16) * (kKvStep / 16);  // 32：4 行块 × 8 列块
static_assert(kPNzFractalBlocks == 32, "P NZ 分形块总数冻结（BC=128 → 8 列块）");

// ---- M2c-B5 P 的 NZ 带 pad 行布局结构确认 ----
// P 矩阵在 L1 中采用 16×16 分形带 16B pad 布局存储：
// 每分形行 32B = 16 字节 FP8 E4M3 码字 + 16 字节 pad。
// Pad 的硬件目的：确保分形行 32 字节物理对齐（C0=32B），规避多核/多流水线并发访存时的 Bank 冲突。
constexpr int64_t kPNzFractalM = 16;           // 分形行数（16 行）
constexpr int64_t kPNzFractalN = 16;           // 分形列数（16 列）
constexpr int64_t kPNzFractalDataBytes = 16;   // 分形行内有效数据字节（16 个 FP8 码字）
constexpr int64_t kPNzFractalPadBytes = 16;    // 分形行内防 Bank 冲突 pad 字节
constexpr int64_t kPNzFractalRowStride = kPNzFractalDataBytes + kPNzFractalPadBytes; // 32B 分形行步长
constexpr int64_t kPNzFractalBlockBytes = kPNzFractalM * kPNzFractalRowStride;        // 512B 每 16x16 分形块
constexpr int64_t kPNzTotalDataBytes = kStreamRows * kKvStep;                         // 8192B 有效数据
constexpr int64_t kPNzTotalPadBytes = kPNzFractalBlocks * (kPNzFractalM * kPNzFractalPadBytes); // 8192B pad
constexpr int64_t kPNzTotalSlotBytes = kPNzFractalBlocks * kPNzFractalBlockBytes;     // 16384B 总槽位

static_assert(kPNzFractalRowStride == 32, "分形行严格为 32B 跨度 (16B 数据 + 16B pad)");
static_assert(kPNzFractalBlockBytes == 512, "单个 16x16 分形块占 512 字节");
static_assert(kPNzTotalSlotBytes == kL1PNzSlot, "P NZ 槽大小 16384 字节精确匹配 32 个分形块");
static_assert(kPNzTotalDataBytes + kPNzTotalPadBytes == kL1PNzSlot, "有效数据与 pad 各占 50% 且交替分布");

// P NZ 带 pad 行布局契约辅助器：编译期与运行期验证内存位置属于数据区还是 pad 区
struct Sa2PNzPadLayoutContract {
    static constexpr int64_t kSlotBytes = kL1PNzSlot;
    static constexpr int64_t kRowStride = kPNzFractalRowStride;
    static constexpr int64_t kDataBytes = kPNzFractalDataBytes;

    SA2_AICORE static constexpr bool IsDataByte(int64_t offset) {
        return (offset >= 0 && offset < kSlotBytes) && ((offset % kRowStride) < kDataBytes);
    }
    SA2_AICORE static constexpr bool IsPadByte(int64_t offset) {
        return (offset >= 0 && offset < kSlotBytes) && ((offset % kRowStride) >= kDataBytes);
    }
};

// ============================ L0 分区 ============================

// L0C 四分区（附录 C.3：0/32K/64K/96K）：流 s × 槽奇偶 p 各占一个 32KB 象限；
// QK(int32) 与 PV(fp32) 分时复用同一象限（QK 部分和经 fixpipe 排空后 PV 才入座，跨核旗号保序）。
constexpr int64_t kL0CQuadBytes = kStreamRows * kMaxHeadDim * 4;  // 32768
constexpr int64_t kL0CQuadTotal = 4 * kL0CQuadBytes;              // 131072 ≤ kL0CSize（半量驻留，余量给折半发射）

static_assert(kL0CQuadBytes == 32768 && kL0CQuadTotal == 131072 && kL0CQuadTotal <= kL0CSize, "L0C 四分区预算");
// L0A/L0B：Q/K/V/P 的 int8/e4m3 分形由 MTE1 按需装载（各 32KB），P 的 MX 单位 scale（2^0）init 驻 L0A（§5.2）。

// ============================ DeviceRes 资源封装 ============================

// 按模板粒度给出全部槽位基址；engine 文件只取此处偏移，禁止自行书写偏移算式（B-26 纪律）。
template <Gran G>
struct Sa2DeviceRes {
  static constexpr Gran kGran = G;
  static constexpr bool kIsThreadGran = (G == Gran::kPerThread);
  static constexpr int64_t kScoreSlotBytes = kIsThreadGran ? kUbScoreThreadSlot : kUbScoreWarpSlot;

  static constexpr int64_t kScoreBase0 = 0;                                // score 槽 ping
  static constexpr int64_t kScoreBase1 = kScoreSlotBytes;                  // score 槽 pong
  static constexpr int64_t kPBase0 = 2 * kScoreSlotBytes;                  // P 槽 ping
  static constexpr int64_t kPBase1 = kPBase0 + kUbPSlot;                   // P 槽 pong
  static constexpr int64_t kPvBase0 = kPBase1 + kUbPSlot;                  // PV 槽 ping
  static constexpr int64_t kPvBase1 = kPvBase0 + kUbPvSlot;                // PV 槽 pong
  static constexpr int64_t kOAccBase = kPvBase1 + kUbPvSlot;               // O 累加器（每流独立在流内偏移）
  static constexpr int64_t kSegBase0 = kIsThreadGran ? kOAccBase + kUbOAcc : -1;   // 分段缓冲 ping（仅 THREAD）
  static constexpr int64_t kSegBase1 = kIsThreadGran ? kSegBase0 + kUbSegSlot : -1;  // 分段缓冲 pong（仅 THREAD）
  static constexpr int64_t kStateBase =
      kIsThreadGran ? kSegBase1 + kUbSegSlot : kOAccBase + kUbOAcc;  // 状态区基址（两模式分支各自可达）
  static constexpr int64_t kUbEnd = kStateBase + kUbStateBytes;

  // 流内状态基址：stream ∈ {0,1}；流内字段偏移 = kStateRowStatOff/DenomOff/CorrOff/AlphaOff。
  static constexpr int64_t StreamStateBase(int64_t stream) { return kStateBase + stream * kStreamStateBytes; }
  // 共合 scratch 基址（两流状态之后）。
  static constexpr int64_t kScratchBase = kStateBase + 2 * kStreamStateBytes;

  // L1 槽位（模式无关，见上方链常量）；流 s 的 Q 槽：
  static constexpr int64_t L1Q8Base(int64_t stream) { return stream == 0 ? kL1Q8S0 : kL1Q8S1; }
  static constexpr int64_t L1QScaleBase(int64_t stream) { return stream == 0 ? kL1QScale0 : kL1QScale1; }
  // L0C 象限：流 s × 槽奇偶 p → 0 / 32K / 64K / 96K（附录 C.3）。
  static constexpr int64_t L0cQuadBase(int64_t stream, int64_t parity) {
    return (stream * 2 + parity) * kL0CQuadBytes;
  }

  // 槽位访问助手（奇偶槽索引：parity ∈ {0, 1}）
  static constexpr int64_t ScoreBase(int64_t parity) { return parity == 0 ? kScoreBase0 : kScoreBase1; }
  static constexpr int64_t PBase(int64_t parity) { return parity == 0 ? kPBase0 : kPBase1; }
  static constexpr int64_t PvBase(int64_t parity) { return parity == 0 ? kPvBase0 : kPvBase1; }
  static constexpr int64_t SegBase(int64_t parity) { return parity == 0 ? kSegBase0 : kSegBase1; }
  static constexpr int64_t L1KStepBase(int64_t parity) { return parity == 0 ? kL1KStep0 : kL1KStep1; }
  static constexpr int64_t L1VStepBase(int64_t parity) { return parity == 0 ? kL1VStep0 : kL1VStep1; }
  static constexpr int64_t L1VScaleBase(int64_t parity) { return parity == 0 ? kL1VScale0 : kL1VScale1; }
  static constexpr int64_t L1PNzBase(int64_t stream) { return stream == 0 ? kL1PNz0 : kL1PNz1; }

  // M2c-B3 K-scale 双缓冲暂存槽（每槽 128 half = 256B，共 512B，容纳于 scratch 内）
  static constexpr int64_t kKScaleStageSlotBytes = kKvStep * 2; // 256B
  static constexpr int64_t KScaleBase(int64_t parity) {
      return kScratchBase + 768 + (parity % 2) * kKScaleStageSlotBytes;
  }

  static_assert(kScoreSlotBytes == 16384 || kScoreSlotBytes == 32768, "score 槽二态");
  static_assert(kUbEnd == 152448 || kUbEnd == 217984, "UB 终址命中双锚点之一");
  static_assert(kUbEnd <= kUbHighWater, "模板实例化后仍 ≤ 248KB 高水位");
};

// ============================ 槽位缓冲区封装（设备与主机统一寻址） ============================
// Sa2UbBuffer 供 AIV MicroAPI（LoadAlign/StoreAlign）直接消费：真实 CANN 构建下 UB 裸指针
// 必须带 __ubuf__ 地址空间限定（hoststub 下展开为空，宿主仿真代码零改动）。L1/L0C 封装
// （Sa2L1Buffer/Sa2L0cBuffer）维持普通指针：仅作标量访问与 TBuf 无关的偏移运算载体，
// 真实张量构建（LocalTensor/GlobalTensor）一律由 attention_loop 经 TBuf::Get()[off] 完成。
template <Gran G>
struct Sa2UbBuffer {
  using Res = Sa2DeviceRes<G>;
  __ubuf__ uint8_t* base{nullptr};

  __aicore__ inline explicit Sa2UbBuffer(__ubuf__ uint8_t* b = nullptr) : base(b) {}
  __aicore__ inline void SetBase(__ubuf__ uint8_t* b) { base = b; }

  __aicore__ inline __ubuf__ half* GetScoreSlotWarp(int64_t parity) const {
    return reinterpret_cast<__ubuf__ half*>(base + Res::ScoreBase(parity));
  }
  __aicore__ inline __ubuf__ float* GetScoreSlotThread(int64_t parity) const {
    return reinterpret_cast<__ubuf__ float*>(base + Res::ScoreBase(parity));
  }
  __aicore__ inline __ubuf__ uint8_t* GetPSlot(int64_t parity) const {
    return base + Res::PBase(parity);
  }
  __aicore__ inline __ubuf__ float* GetPvSlot(int64_t parity) const {
    return reinterpret_cast<__ubuf__ float*>(base + Res::PvBase(parity));
  }
  __aicore__ inline __ubuf__ float* GetOAcc() const {
    return reinterpret_cast<__ubuf__ float*>(base + Res::kOAccBase);
  }
  __aicore__ inline __ubuf__ float* GetStateRowStat(int64_t stream) const {
    return reinterpret_cast<__ubuf__ float*>(base + Res::StreamStateBase(stream) + kStateRowStatOff);
  }
  __aicore__ inline __ubuf__ float* GetStateDenom(int64_t stream) const {
    return reinterpret_cast<__ubuf__ float*>(base + Res::StreamStateBase(stream) + kStateDenomOff);
  }
  __aicore__ inline __ubuf__ float* GetStateCorr(int64_t stream) const {
    return reinterpret_cast<__ubuf__ float*>(base + Res::StreamStateBase(stream) + kStateCorrOff);
  }
  __aicore__ inline __ubuf__ float* GetStateAlpha(int64_t stream, int64_t parity = 0) const {
    if (parity == 0) {
      return reinterpret_cast<__ubuf__ float*>(base + Res::StreamStateBase(stream) + kStateAlphaOff);
    } else {
      return reinterpret_cast<__ubuf__ float*>(base + Res::kScratchBase + stream * 256);
    }
  }
  __aicore__ inline __ubuf__ half* GetKScaleSlot(int64_t parity) const {
    return reinterpret_cast<__ubuf__ half*>(base + Res::KScaleBase(parity));
  }
  // 共合 scratch 中预留的 LSE m 值装配暂存（+768 起每流 256B；布局契约见
  // sa2_attention_loop.h VF 计算核段 scratch 表）：FinishTileVectorDevice 把跨步
  // rowStat 装配成连续行供 LoadAlign。
  __aicore__ inline __ubuf__ float* GetStateMStage(int64_t stream) const {
    return reinterpret_cast<__ubuf__ float*>(base + Res::kScratchBase + 768 + stream * 0);
  }
};

struct Sa2L1Buffer {
  uint8_t* base{nullptr};

  __aicore__ inline explicit Sa2L1Buffer(uint8_t* b = nullptr) : base(b) {}
  __aicore__ inline void SetBase(uint8_t* b) { base = b; }

  __aicore__ inline int8_t* GetKStepSlot(int64_t parity) const {
    return reinterpret_cast<int8_t*>(base + (parity == 0 ? kL1KStep0 : kL1KStep1));
  }
  __aicore__ inline int8_t* GetVStepSlot(int64_t parity) const {
    return reinterpret_cast<int8_t*>(base + (parity == 0 ? kL1VStep0 : kL1VStep1));
  }
  __aicore__ inline int16_t* GetVScaleSlot(int64_t parity) const {
    return reinterpret_cast<int16_t*>(base + (parity == 0 ? kL1VScale0 : kL1VScale1));
  }
  __aicore__ inline int8_t* GetQ8Slot(int64_t stream) const {
    return reinterpret_cast<int8_t*>(base + (stream == 0 ? kL1Q8S0 : kL1Q8S1));
  }
  __aicore__ inline half* GetQScaleSlot(int64_t stream) const {
    return reinterpret_cast<half*>(base + (stream == 0 ? kL1QScale0 : kL1QScale1));
  }
  __aicore__ inline uint8_t* GetPNzSlot(int64_t stream) const {
    return base + (stream == 0 ? kL1PNz0 : kL1PNz1);
  }
  __aicore__ inline float* GetKMeanSlot() const {
    return reinterpret_cast<float*>(base + kL1KMean);
  }
  __aicore__ inline uint8_t* GetMxScaleCacheBase() const {
    return base + kL1MxScaleCacheBase;
  }
};

struct Sa2L0cBuffer {
  uint8_t* base{nullptr};

  __aicore__ inline explicit Sa2L0cBuffer(uint8_t* b = nullptr) : base(b) {}
  __aicore__ inline void SetBase(uint8_t* b) { base = b; }

  __aicore__ inline void* GetQuadrant(int64_t stream, int64_t parity) const {
    return base + (stream * 2 + parity) * kL0CQuadBytes;
  }
};

using Sa2DeviceResWarp = Sa2DeviceRes<Gran::kPerWarp>;
using Sa2DeviceResThread = Sa2DeviceRes<Gran::kPerThread>;

static_assert(Sa2DeviceResWarp::kPBase0 == 32768 && Sa2DeviceResWarp::kPBase1 == 41472 &&
                  Sa2DeviceResWarp::kPvBase0 == 50176 && Sa2DeviceResWarp::kPvBase1 == 82944 &&
                  Sa2DeviceResWarp::kOAccBase == 115712 && Sa2DeviceResWarp::kStateBase == 148480 &&
                  Sa2DeviceResWarp::kUbEnd == 152448,
              "per_warp 链：score×2(32K) | P×2(17K) | PV×2(64K) | O(32K) | state(3968)");
static_assert(Sa2DeviceResThread::kPBase0 == 65536 && Sa2DeviceResThread::kPBase1 == 74240 &&
                  Sa2DeviceResThread::kPvBase0 == 82944 && Sa2DeviceResThread::kPvBase1 == 115712 &&
                  Sa2DeviceResThread::kOAccBase == 148480 && Sa2DeviceResThread::kSegBase0 == 181248 &&
                  Sa2DeviceResThread::kSegBase1 == 197632 && Sa2DeviceResThread::kStateBase == 214016 &&
                  Sa2DeviceResThread::kUbEnd == 217984,
              "per_thread 链：score×2(64K,FP32) | P×2(17K) | PV×2(64K) | O(32K) | seg×2(32K) | state(3968)");
static_assert(Sa2DeviceResThread::kScratchBase == 216640, "scratch = 217984 - 1344");
static_assert(Sa2DeviceResWarp::L0cQuadBase(0, 0) == 0 && Sa2DeviceResWarp::L0cQuadBase(0, 1) == 32768 &&
                  Sa2DeviceResWarp::L0cQuadBase(1, 0) == 65536 && Sa2DeviceResWarp::L0cQuadBase(1, 1) == 98304,
              "L0C 四分区 0/32K/64K/96K");

// ============================ Fence 助手 ============================

// 本地事件 ↔ 槽位配对（票号见 sa2_events.h；Set/Wait 边沿纪律随注释，engine 不得改配对）。
struct Sa2Fence {
  // K 步 tile：MTE1 读完 kL1KStep{parity} 入 L0A 后 Set(MTE1_MTE2, ev)；MTE2 预取下一 K tile 进同槽前 Wait。
  static constexpr int64_t KPrefetchEvent(int64_t parity) { return parity == 0 ? kEvKPrefetch0 : kEvKPrefetch1; }
  // V 步 tile：同 K，槽 kL1VStep{parity}，票号域 MTE1_MTE2。
  static constexpr int64_t VPrefetchEvent(int64_t parity) { return parity == 0 ? kEvVPrefetch0 : kEvVPrefetch1; }
  // scale 槽（vScale / 随步 kScale 侧数据）：AIV 读毕 Set(V_MTE2, ev)；MTE2 重取前 Wait。
  static constexpr int64_t ScaleEvent(int64_t parity) { return parity == 0 ? kEvScale0 : kEvScale1; }
};

static_assert(Sa2Fence::KPrefetchEvent(0) == 0 && Sa2Fence::KPrefetchEvent(1) == 1 &&
                  Sa2Fence::VPrefetchEvent(0) == 2 && Sa2Fence::VPrefetchEvent(1) == 3 &&
                  Sa2Fence::ScaleEvent(0) == 4 && Sa2Fence::ScaleEvent(1) == 5,
              "本地事件票号冻结：K 预取 0/1、V 预取 2/3、scale 4/5");

}  // namespace sa2

#endif  // SA2_UB_PLAN_H_
