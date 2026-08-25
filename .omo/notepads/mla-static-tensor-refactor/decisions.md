# Decisions — mla-static-tensor-refactor

Architectural choices and rationales discovered during work on this plan.

_Auto-scaffolded by /start-work. Append new entries below - never overwrite._

---
## 2026-08-25 T2 前置（注 10）：vec 侧 HardEvent 类事件 ID 全额枚举

开工前对 vec 侧全部核内同步事件逐 HardEvent 类枚举固定 ID（替代旧 TPipe AllocEventID
动态分配 + TQue 内部事件）。**ID 空间按核隔离（Mutex 0-26），核间一律走 CROSSCORE 旗标。**

### MTE3_V 类（旧 TPipe 池 0..7）→ 新 Mutex ID（PIPE_V/PIPE_MTE3 成对）
| 旧动态事件 | 旧用途 | 新固定 ID | 新机制 |
|---|---|---|---|
| `mte3ToVId[0]` | vec2Res(64K@160K) 的 V→MTE3 门 | 0（UB_OUT_VEC2_RES_EVENT0） | Mutex::Lock/Unlock<PIPE_V>/<PIPE_MTE3>(0) |
| `mte3ToVId[1]` | Alloc 预置后仅 teardown 消费（无 mid-run 使用） | **删除**（单槽 vec2Res 由 ID0 覆盖） | — |
| `vToMte3Id[0]`（V_MTE3 类） | Bmm2ResCastAndCopyOut 内 V→MTE3 自同步 | **并入 0** | Mutex<MTE3>(0) 覆盖 DataCopy 段 |
| `vToMte3Id[1]`（V_MTE3 类） | Bmm2ResForFDCopyOut[loop%DB] | **并入 0** | 同上 |
| stage1OutQue（TQue<VECOUT,1> 内部 MTE3_V/V_MTE3） | stage1 cast→L1 P 写 | **2（UB_OUT_VEC1_RES_EVENT0）** | Mutex(2) |
| softmaxLseQueue（TQue 内部） | LSE 1K 写 GM | **4（UB_OUT_LSE_OUT_EVENT0）** | Mutex(4) |
| sumBrdcst（TQue 内部） | FD sum [32,8] 写 GM | **5（UB_BRDCST_SUM_EVENT）** | Mutex(5) |
| maxBrdcst（TQue 内部） | FD max [32,8] 写 GM | **8（UB_BRDCST_MAX_EVENT）** | Mutex(8) |
| `initOutputEventId`（旧 static 0U） | ClearOutput 单核刷无效行 | **1（EVENT_ID0）/3（EVENT_ID1）** | AttentionCommon::InitOutput ENABLE_LOCK=true（PIPE_V/PIPE_MTE3 Mutex） |

### MTE2_V / mask 类
MLA 的 `AttentionmaskCopyIn` 走 **common 头（attro common/op_kernel/arch35/attenmask_gs1.h）4 参自同步版本**
（内部 `GetTPipePtr()->FetchEventID(MTE2_V)` + SetFlag/WaitFlag + PipeBarrier<PIPE_V>），
**不接收显式 ID** → 与 FA 的 5 参（mte2ToVEventId 显式 Mutex）不同，MLA 不需要为 mask 单独
分配 Mutex ID。**表 2 的 UB_IN_MASK_EVENT0/1=6/7 仅在注释中保留为"与 FA 排布对齐的预留槽"，不实际使用**；
mask 槽 2×4K 由 `[loop%DB]` 静态索引轮转，同步由 common 自同步函数承担（语义 = 旧 TQue 深度 1 的
EnQue/DeQue 释放，见 learnings）。旧动态 `mte3ToVId[1]` 与 V_MTE3 双槽均因"单槽 vec2Res / 串行轮次"
收敛为单 Mutex ID（与 FA 单槽 vec2Res 同构）。

### FD 阶段旗标（保留原状，非 Mutex）
FD 的 SYNC_{LSE_MAX_SUM_BUF1/2,MM2RES_BUF1/2,FDOUTPUT,LSEOUTPUT,SINK_BUF1/2}_FLAG 为
**raw 字面量 SetFlag/WaitFlag（V_MTE2/MTE3_V/V_MTE3/MTE2_V 类）**，非 TPipe 事件、非 Mutex；
与 vec FA 阶段不重叠（分时）且机制独立，原样保留。

### 复核结论（表 2 冻结值与 FA 语义等价，本核内不冲突为硬约束）
- vec 核内 Mutex 最终占用：**0(vec2Res), 1(EVENT0), 2(vec1/stage1), 3(EVENT1), 4(lse), 5(sumBrdcst), 8(maxBrdcst)**；6/7 预留注释占位（mask 自同步）。无双重分配。
- cube 核内 Mutex：0/1(L0C), 2(Q 单槽), 4..6(KV/P 3 槽), 7/8(L0A), 9/10(L0B)。
- CROSSCORE（mode=4，AIV1 偏移 +16）：BMM1=0/1、BMM2=2/3（c1v1/c2v2 各 2 槽）、L1P=5/6/7。
