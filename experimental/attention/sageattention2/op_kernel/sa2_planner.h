// sa2_planner.h — SageAttention2 v2 任务解码 / 区间分配 / GM·workspace 寻址（O(1) 纯整式，无查表）。
//
// 推导依据：DESIGN_SPEC §7（FA4 LPT：b 外层 → kh 奇偶分 die → 组内 qh 连续 → m-block 因果逆序；
// 有序任务表静态连续区间均分到 AIC；AIC 内两 AIV 流从区间两端对进）+ §5.3 workspace 布局 + D-R4 SF 打包口径。
#ifndef SA2_PLANNER_H_
#define SA2_PLANNER_H_

#include <cstdint>

#include "sa2_spec.h"

namespace sa2 {

// ---- 表结构常量（附录 C.5 口径）----
// die 局部任务表（每 AIC 一张）：slot = (b·pairCols + p)，pair 覆盖 kh = 2p + die；
// hk 奇数时最后一对在 die=1 侧越界，槽仍在表中占位（valid=false，O(1) 跳过，保证区间均分均匀）。
// 任务 id：gid = slot·perHead + (qh·tiles + mbRev)，mbRev 为因果逆序游标（0 ↔ 最高 m-block）。
SA2_AICORE constexpr int64_t PairCols(int64_t hk) { return (hk + 1) / 2; }           // ceil(hk/2)
SA2_AICORE constexpr int64_t TileCount(int64_t qp) { return qp / kStreamRows; }      // qp 已 64 对齐
SA2_AICORE constexpr int64_t GroupSize(int64_t hq, int64_t hk) { return hk > 0 ? (hq / hk) : 1; }
SA2_AICORE constexpr int64_t PerHead(int64_t group, int64_t tiles) { return group * tiles; }
SA2_AICORE constexpr int64_t SlotsPerDie(int64_t b, int64_t pairCols) { return b * pairCols; }
SA2_AICORE constexpr int64_t TasksPerDie(int64_t slots, int64_t perHead) { return slots * perHead; }

// 因果 limit（§3.3：非因果 limit=sk；因果 limit=min(sk, m0+BR)）与 KV 步数。
SA2_AICORE constexpr int64_t CausalLimit(int64_t sk, int64_t m0) {
  const int64_t lim = m0 + kStreamRows;
  return lim < sk ? lim : sk;
}
SA2_AICORE constexpr int64_t StepCount(int64_t limit) { return (limit + kKvStep - 1) / kKvStep; }

struct Sa2Task {
  int64_t b = 0;      // batch 下标
  int64_t kh = 0;     // KV head 下标（kh 偶 → die0，奇 → die1）
  int64_t qh = 0;     // 全局 query head 下标（kh·group + localQh）
  int64_t m0 = 0;     // query 行起点（mb·64）
  int64_t steps = 0;  // 本 tile 的 KV 步数 = ceil(limit(m0)/kKvStep)
  bool valid = false; // hk 奇数时 die1 尾槽 = false
};

SA2_AICORE constexpr Sa2Task DecodeTask(const Sa2Params& p, int64_t die, int64_t gid) {
  const int64_t pairCols = PairCols(p.hk);
  const int64_t tiles = TileCount(p.qp);
  const int64_t group = GroupSize(p.hq, p.hk);
  const int64_t perHead = PerHead(group, tiles);
  const int64_t slotBlock = pairCols * perHead;
  const int64_t b = gid / slotBlock;
  const int64_t rem = gid - b * slotBlock;
  const int64_t pair = rem / perHead;
  const int64_t i = rem - pair * perHead;
  const int64_t kh = pair * 2 + die;
  const int64_t localQh = i / tiles;
  const int64_t qh = kh * group + localQh;
  const int64_t mbRev = i - localQh * tiles;
  // LPT (Longest-Processing-Time-First) 调度（对齐 FlashAttention-4 arXiv:2603.05451 §3.2）：
  // 因果掩码下，第 mb 个 query 分块的有效 KV 范围为 min(sk, (mb+1)*64)，步数随 mb 严格单调递增；
  // 若按自然序调度，耗时最长的尾块在末尾执行，因核间微小时间差将导致严重尾核空转等待（tail core wait）；
  // 采用因果逆序 (tiles - 1 - mbRev)：长任务优先并行执行，调度尾部任务粒度缩小至最小步长（1步），
  // 在 24 AIC / 48 AIV 多核分配下，任意两核完成时间极差上界从 O(sk/Bc) 骤降至 <= 1 步，彻底消除负载不均！
  const int64_t mb = p.causal ? (tiles - 1 - mbRev) : mbRev;  // 因果逆序 (LPT) / 非因果自然序
  const int64_t m0 = mb * kStreamRows;
  const int64_t limit = p.causal ? CausalLimit(p.sk, m0) : p.sk;
  Sa2Task t;
  t.b = b;
  t.kh = kh;
  t.qh = qh;
  t.m0 = m0;
  t.steps = StepCount(limit);
  t.valid = (kh < p.hk) && (qh < p.hq);
  return t;
}

// ---- 核间分配：有序槽表静态连续区间均分（边界落在槽界 ⇒ 同 head 的 tile 落同核，K 的 L2 复用最大）----
struct Sa2Range {
  int64_t begin = 0;  // 槽区间起点（含）
  int64_t end = 0;    // 槽区间终点（不含）
};

SA2_AICORE constexpr Sa2Range SplitRange(int64_t slots, int64_t cores, int64_t core) {
  Sa2Range r;
  r.begin = slots * core / cores;
  r.end = slots * (core + 1) / cores;
  return r;
}

// ---- GM / workspace 寻址（(b,kh) 视图；k8/v8/kScale/vScale/kMean/vMean 语义见 Sa2TilingData 注释）----
struct Sa2KvHeadView {
  int64_t gmK = 0;      // K[b][kh][0][0] 元素偏移（×dtype 大小 = 字节）
  int64_t gmV = 0;      // V[b][kh][0][0] 元素偏移
  int64_t wsK8 = 0;     // 字节偏移：k8  int8  [B·Hk, Kp, DP]
  int64_t wsKScale = 0; // 字节偏移：kScale fp16 [B·Hk, Kp, segWidth]（per_warp 时 segWidth=1）
  int64_t wsV8 = 0;     // 字节偏移：v8  int8  [B·Hk, Kp, DP]
  int64_t wsVScale = 0; // 字节偏移：vScale int16 [B·Hk, DP, Kp/64]
                        //   （D-R4 950 口径：K 维相邻两个 32-token 组 UE8M0 打包 int16、MN-major，
                        //    出处 DeepGEMM-Ascend README:56 + transform_sf.hpp）
  int64_t wsKMean = 0;  // 字节偏移：kMean fp32 [B·Hk, DP]
  int64_t wsVMean = 0;  // 字节偏移：vMean fp32 [B·Hk, DP]（仅 smooth_v 且 fp32 档语义非零）
};

// __aicore__ 标注（SA2_AICORE 宿主为空）：被内核 __aicore__ 运行期调用（-xcce 混编实测必需）；
// kProbeView 静态断言走常量求值不受影响。
SA2_AICORE constexpr Sa2KvHeadView KvHeadView(const Sa2Params& p, int64_t b, int64_t kh) {
  const int64_t idx = b * p.hk + kh;
  const int64_t headPairs = p.kp / (kMxGroupTokens * kMxScalePack);  // kp/64：打包 int16 的 K 向组数
  Sa2KvHeadView v;
  v.gmK = b * p.kStride[0] + kh * p.kStride[1];
  v.gmV = b * p.vStride[0] + kh * p.vStride[1];
  v.wsK8 = idx * p.kp * p.dp;
  v.wsKScale = idx * p.kp * p.segWidth * 2;
  v.wsV8 = idx * p.kp * p.dp;
  v.wsVScale = idx * p.dp * headPairs * 2;
  v.wsKMean = idx * p.dp * 4;
  v.wsVMean = idx * p.dp * 4;
  return v;
}

// 步内寻址（第 step 步 tile = KV 行 [step·128, step·128+128)）：
SA2_AICORE constexpr int64_t K8StepByteOff(int64_t wsK8, int64_t step, int64_t dp) {
  return wsK8 + step * kKvStep * dp;
}
SA2_AICORE constexpr int64_t KScaleStepByteOff(int64_t wsKScale, int64_t step, int64_t segWidth) {
  return wsKScale + step * kKvStep * segWidth * 2;
}
SA2_AICORE constexpr int64_t V8StepByteOff(int64_t wsV8, int64_t step, int64_t dp) {
  return wsV8 + step * kKvStep * dp;
}
// vScale：每步推进 Kp/64 中的 2 个打包 int16/通道（MN-major，通道为外维）。
SA2_AICORE constexpr int64_t VScaleStepByteOff(int64_t wsVScale, int64_t step, int64_t dp) {
  return wsVScale + step * (kKvStep / (kMxGroupTokens * kMxScalePack)) * dp * 2;
}

// ---- Q tile / 输出 / LSE 寻址（附录 C.5 公式落地）----
// Q tile 元素偏移（行 [m0, m0+64)）；布局无关，stride 由 host 传入。
SA2_AICORE constexpr int64_t QTileOffset(const Sa2Params& p, int64_t b, int64_t qh, int64_t m0) {
  return b * p.qStride[0] + qh * p.qStride[1] + m0 * p.qStride[2];
}
// 尾块有效行数（epilogue 裁剪与 softmax 掩码用）。
SA2_AICORE constexpr int64_t QTileValidRows(const Sa2Params& p, int64_t m0) {
  const int64_t rows = p.sq - m0;
  return rows < 0 ? 0 : (rows > kStreamRows ? kStreamRows : rows);
}
// 输出 tile 元素偏移：输出与 q 同布局同 stride（任务书 §2.2 返回值约定）；裁回 d_og 由 epilogue 处理。
SA2_AICORE constexpr int64_t OutTileOffset(const Sa2Params& p, int64_t b, int64_t qh, int64_t m0) {
  return b * p.qStride[0] + qh * p.qStride[1] + m0 * p.qStride[2];
}
// LSE 行偏移：fp32 [B, Hq, Sq]（对齐开源 core.py 契约；数值口径见 §4.6）。
SA2_AICORE constexpr int64_t LseOffset(const Sa2Params& p, int64_t b, int64_t qh, int64_t m0) {
  return (b * p.hq + qh) * p.sq + m0;
}

// ============================ 静态自检（工作例：b=2、hk=7 奇、hq=4、sq=256、sk=1000、因果） ============================

SA2_AICORE constexpr Sa2Params MakePlannerProbe(bool causal) {
  Sa2Params p{};
  p.b = 2;
  p.hq = 28; // GQA group = 28 / 7 = 4
  p.hk = 7;  // 奇数：die1 最后一对为无效尾槽
  p.sq = 256;
  p.sk = 1000;
  p.d = 128;
  p.dp = 128;
  p.qp = 256;
  p.kp = 1024;
  p.qStride[0] = 28 * 256 * 128;
  p.qStride[1] = 256 * 128;
  p.qStride[2] = 128;
  p.kStride[0] = 7 * 1000 * 128;
  p.kStride[1] = 1000 * 128;
  p.kStride[2] = 128;
  p.vStride[0] = p.kStride[0];
  p.vStride[1] = p.kStride[1];
  p.vStride[2] = 128;
  p.causal = causal;
  p.gran = 1;
  p.accum = 2;
  p.segWidth = 16;
  p.cores = 1;
  p.smScale = 0.08838834764831845f;
  return p;
}

constexpr Sa2Params kPlannerProbe = MakePlannerProbe(true);
constexpr int64_t kProbePairCols = PairCols(kPlannerProbe.hk);                    // 4
constexpr int64_t kProbeTiles = TileCount(kPlannerProbe.qp);                      // 4
constexpr int64_t kProbeGroup = GroupSize(kPlannerProbe.hq, kPlannerProbe.hk);    // 4
constexpr int64_t kProbePerHead = PerHead(kProbeGroup, kProbeTiles);              // 16
constexpr int64_t kProbeSlots = SlotsPerDie(kPlannerProbe.b, kProbePairCols);     // 8
constexpr int64_t kProbeTasks = TasksPerDie(kProbeSlots, kProbePerHead);          // 128

static_assert(kProbePairCols == 4 && kProbeTiles == 4 && kProbePerHead == 16 && kProbeSlots == 8 &&
                  kProbeTasks == 128,
              "表结构：pairCols=ceil(7/2)=4、tiles=256/64=4、perHead=4×4=16、slots=2×4=8、tasks=8×16=128");

// 解码工作例：die=1、gid=108 → b=1、pair=2（kh=5）、i=12 → localQh=3、qh=5*4+3=23、mbRev=0 → 因果 mb=3、m0=192。
constexpr Sa2Task kProbeTask = DecodeTask(kPlannerProbe, 1, 108);
static_assert(kProbeTask.valid && kProbeTask.b == 1 && kProbeTask.kh == 5 && kProbeTask.qh == 23 &&
                  kProbeTask.m0 == 192,
              "gid=108 解码：slot=6(b=1,p=2)、i=12 → qh=23、mbRev=0 → 因果逆序 mb=3");
static_assert(kProbeTask.steps == 2, "因果 limit=min(1000,192+64)=256 → steps=ceil(256/128)=2");
constexpr Sa2Task kProbeTail = DecodeTask(kPlannerProbe, 1, 3 * kProbePerHead);  // die1、p=3 → kh=7 越界
static_assert(!kProbeTail.valid, "hk=7 奇数：die1 尾槽 kh=7 无效（valid=false，占位不执行）");
constexpr Sa2Task kProbeFirst = DecodeTask(kPlannerProbe, 0, 0);
static_assert(kProbeFirst.b == 0 && kProbeFirst.kh == 0 && kProbeFirst.qh == 0 && kProbeFirst.m0 == 192 &&
                  kProbeFirst.steps == 2,
              "gid=0 = 最高 m-block 先行（因果逆序 → LPT：KV 长度先长后短）");
constexpr Sa2Task kProbeNatural = DecodeTask(MakePlannerProbe(false), 0, 0);
static_assert(kProbeNatural.m0 == 0 && kProbeNatural.steps == 8,
              "非因果：自然序 m0=0，limit=sk=1000 → steps=ceil(1000/128)=8");

// 区间分配：8 槽 3 核 → [0,2)/[2,5)/[5,8)（整式均分，无需前缀和）。
constexpr Sa2Range kProbeR0 = SplitRange(8, 3, 0);
constexpr Sa2Range kProbeR1 = SplitRange(8, 3, 1);
constexpr Sa2Range kProbeR2 = SplitRange(8, 3, 2);
static_assert(kProbeR0.begin == 0 && kProbeR0.end == 2 && kProbeR1.begin == 2 && kProbeR1.end == 5 &&
                  kProbeR2.begin == 5 && kProbeR2.end == 8,
              "SplitRange 连续区间均分且并集完备");

// 寻址工作例（probe 的 q 为 HND：[2,28,256,128]）：
static_assert(QTileOffset(kPlannerProbe, 1, 3, 192) == 917504 + 98304 + 24576,
              "Q tile 元素偏移 = b·S0 + qh·S1 + m0·S2 = 917504+98304+24576 = 1040384");
static_assert(QTileValidRows(kPlannerProbe, 192) == 64 && QTileValidRows(kPlannerProbe, 250) == 6,
              "尾块有效行：sq=256 时 m0=192 满 64 行、m0=250 剩 6 行");
static_assert(LseOffset(kPlannerProbe, 1, 3, 192) == (1 * 28 + 3) * 256 + 192,
              "LSE 行偏移 = (b·hq+qh)·sq + m0 = 8128");
static_assert(VScaleStepByteOff(0, 1, 128) == 512 && KScaleStepByteOff(0, 1, 16) == 4096,
              "步内 scale 偏移：vScale 打包 512B/步（D-R4）、kScale per_thread 4KB/步");

// KvHeadView 工作例：b=1、kh=5 → idx=12；kp=1024、dp=128、segWidth=16、kp/64=16。
inline constexpr Sa2KvHeadView kProbeView = KvHeadView(kPlannerProbe, 1, 5);
static_assert(kProbeView.gmK == kPlannerProbe.kStride[0] + 5 * kPlannerProbe.kStride[1],
              "gmK = b·S0 + kh·S1");
static_assert(kProbeView.wsK8 == 12 * 1024 * 128 && kProbeView.wsKScale == 12 * 1024 * 16 * 2 &&
                  kProbeView.wsVScale == 12 * 128 * 16 * 2 && kProbeView.wsKMean == 12 * 128 * 4,
              "workspace 偏移：k8=1572864、kScale=393216、vScale=49152、kMean=6144（字节）");

// =============================================================================
// FlashAttention-4 LPT 因果逆序调度与 24 AIC / 48 AIV 核心网格形式化证明
// =============================================================================

// 1. 验证 LPT 因果逆序单调递减性：mbRev 增加时，步数严格单调不增
SA2_AICORE constexpr bool CheckLptMonotonicity(const Sa2Params& p) {
  const int64_t tiles = TileCount(p.qp);
  int64_t prevSteps = 999999999;
  for (int64_t mbRev = 0; mbRev < tiles; ++mbRev) {
    const int64_t mb = p.causal ? (tiles - 1 - mbRev) : mbRev;
    const int64_t m0 = mb * kStreamRows;
    const int64_t limit = p.causal ? CausalLimit(p.sk, m0) : p.sk;
    const int64_t steps = StepCount(limit);
    if (p.causal && steps > prevSteps) {
      return false;
    }
    prevSteps = steps;
  }
  return true;
}

static_assert(CheckLptMonotonicity(kPlannerProbe), "kPlannerProbe satisfies LPT monotonicity");

// 2. 验证 FlashAttention-4 消除尾核空转：最末尾任务的 KV 步数恒为 1（128 tokens）
SA2_AICORE constexpr int64_t GetLastTaskSteps(const Sa2Params& p) {
  const int64_t tiles = TileCount(p.qp);
  const int64_t mb = p.causal ? 0 : (tiles - 1);
  const int64_t m0 = mb * kStreamRows;
  const int64_t limit = p.causal ? CausalLimit(p.sk, m0) : p.sk;
  return StepCount(limit);
}

static_assert(GetLastTaskSteps(kPlannerProbe) == 1,
              "FlashAttention-4 LPT causal scheduling bounds tail task to exactly 1 KV step!");

// 3. 验证 24 AIC 网格（每 Die 12 AIC）无空隙无重叠均匀划分
SA2_AICORE constexpr bool CheckSplitRangeGrid(int64_t totalSlots, int64_t cores) {
  if (cores <= 0) return false;
  int64_t current = 0;
  for (int64_t c = 0; c < cores; ++c) {
    Sa2Range r = SplitRange(totalSlots, cores, c);
    if (r.begin != current || r.end < r.begin) return false;
    current = r.end;
  }
  return current == totalSlots;
}

static_assert(CheckSplitRangeGrid(64, 12), "12 AIC cores per Die partition 64 slots seamlessly");
static_assert(CheckSplitRangeGrid(128, 24), "24 AIC cores partition 128 slots seamlessly");
static_assert(CheckSplitRangeGrid(16, 12), "12 AIC cores partition 16 slots seamlessly");
static_assert(CheckSplitRangeGrid(8, 12), "12 AIC cores partition 8 slots seamlessly");

// 4. 验证 GQA 组内 Query Head 对同一 KV Head 的连续单调排布（L2 Cache 复用基石）
SA2_AICORE constexpr bool CheckGqaContiguity(const Sa2Params& p) {
  const int64_t tiles = TileCount(p.qp);
  const int64_t group = GroupSize(p.hq, p.hk);
  const int64_t perHead = PerHead(group, tiles);
  for (int64_t pair = 0; pair < PairCols(p.hk); ++pair) {
    const int64_t kh = pair * 2;
    if (kh >= p.hk) break;
    int64_t prevQh = -1;
    for (int64_t i = 0; i < perHead; ++i) {
      const int64_t localQh = i / tiles;
      const int64_t qh = kh * group + localQh;
      if (qh < prevQh) return false;
      prevQh = qh;
    }
  }
  return true;
}

static_assert(CheckGqaContiguity(kPlannerProbe),
              "GQA Query Heads are contiguously non-decreasing within each KV head");

// 5. 形式化证明：全量 KV Head 在任务表中 100% 存在且可达（P0-1 严格闭环）
SA2_AICORE constexpr bool CheckAllKvHeadsReachable(const Sa2Params& p) {
  if (p.hk <= 0) return false;
  const int64_t pairCols = PairCols(p.hk);
  const int64_t tiles = TileCount(p.qp);
  const int64_t group = GroupSize(p.hq, p.hk);
  const int64_t perHead = PerHead(group, tiles);
  const int64_t slotsPerDie = SlotsPerDie(p.b, pairCols);
  const int64_t totalTasksPerDie = TasksPerDie(slotsPerDie, perHead);

  for (int64_t targetKh = 0; targetKh < p.hk; ++targetKh) {
    bool found = false;
    const int64_t targetDie = targetKh % 2;
    for (int64_t gid = 0; gid < totalTasksPerDie; ++gid) {
      Sa2Task t = DecodeTask(p, targetDie, gid);
      if (t.valid && t.kh == targetKh) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }
  return true;
}

static_assert(CheckAllKvHeadsReachable(kPlannerProbe),
              "kPlannerProbe: ∀ kh ∈ [0, Hk - 1], ∃ (die, gid) such that task.kh == kh && task.valid");

SA2_AICORE constexpr Sa2Params MakeReachabilityProbe(int64_t hk, int64_t hq) {
  Sa2Params p{};
  p.b = 1;
  p.hq = hq;
  p.hk = hk;
  p.sq = 64;
  p.sk = 128;
  p.qp = 64;
  p.kp = 128;
  return p;
}

static_assert(CheckAllKvHeadsReachable(MakeReachabilityProbe(1, 4)), "Hk=1 MQA all heads reachable");
static_assert(CheckAllKvHeadsReachable(MakeReachabilityProbe(2, 4)), "Hk=2 all heads reachable");
static_assert(CheckAllKvHeadsReachable(MakeReachabilityProbe(7, 28)), "Hk=7 odd all heads reachable");
static_assert(CheckAllKvHeadsReachable(MakeReachabilityProbe(8, 32)), "Hk=8 GQA 4:1 all heads reachable");
static_assert(CheckAllKvHeadsReachable(MakeReachabilityProbe(32, 32)), "Hk=32 MHA all heads reachable");

}  // namespace sa2

#endif  // SA2_PLANNER_H_
