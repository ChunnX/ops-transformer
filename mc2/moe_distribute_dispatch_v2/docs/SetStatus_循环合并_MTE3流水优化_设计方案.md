# SetStatus 循环合并 + S→MTE3 流水优化设计方案

> 目标文件：`op_kernel/moe_distribute_dispatch_v2.h`
> 目标函数：`SetStatus`（定义 L1023–1066）
> 参考实现：
> - 本仓 `op_kernel/moe_distribute_dispatch_v2_full_mesh.h::CalAndSendCntByExp`（L1258–1304，**已验证的合并循环形态**）
> - step_split 仓 `moe_distribute_dispatch_v2_full_mesh.h::CalAndSendCntByExp`（旋转 event 流水）
> - step_split 设计方案 `docs/fullmesh_flag平面_MTE3事件流水优化_设计方案.md`
> 接口先例：本仓 full_mesh token 流水（`SetFlag/WaitFlag<MTE3_V>`，ring-indexed eventID）

---

## 1. 背景与目标

### 1.1 背景
`SetStatus` 负责把"本卡每个专家的 token 计数（count）"以 32B status block（8 int32：slot0=flag `0x3F800000`，slot1=count）的形式 `DataCopy` 到对端 rank 的 status 窗口，供接收侧 `WaitStatus` 轮询聚合。当前实现为**两循环 + 中间全局屏障**结构：

1. **Loop1**：逐专家算 count（`CalFlagSendExpertCnt`，Vector，O(BS·K)）+ 标量写 `statusTensor_` slot1；
2. `PipeBarrier<PIPE_ALL>()` + `SyncAll<true>()`；
3. **Loop2**：逐专家 `DataCopy` 8-int32 block 到对端 rank 窗口（MTE3，跨片远端写）。

### 1.2 现状问题
- Loop1（全量 Vector 计数计算）与 Loop2（MTE3 远端发送）**完全串行**，无重叠：所有专家的 count 都算完，才开始发第一个。
- 每笔 `DataCopy` 仅 32B，远小于互连有效传输粒度，单笔开销被建链/头开销主导；`sendExpertNum` 笔小远端写聚合，MTE3 段为潜在瓶颈。
- 中间 `SyncAll<true>()` 是全局跨核屏障，开销随核数增长。

### 1.3 目标
将两循环**合并为"算一个、发一个"的单循环**，让 **Vector 计数计算 ↔ MTE3 发送**重叠（计算专家 E+1 时，专家 E 的远端写在途）。**分阶段**推进：
- **Phase 1（本次）**：合并循环 + per-iter `SyncFunc<S_MTE3>` + `SyncAll` 上移到函数开头。对齐本仓 full_mesh `CalAndSendCntByExp` 的已验证形态，低风险、立即获得 V↔MTE3 重叠收益。
- **Phase 2（后续，打点驱动）**：若 MTE3 段仍为瓶颈，把 per-iter `SyncFunc<S_MTE3>` 升级为旋转 `SetFlag/WaitFlag<S_MTE3>`，让多笔远端写并发在途。

---

## 2. 现状正确性分析：中间 `SyncAll` 是否为硬性依赖？

**结论：非硬性跨核数据依赖，属过度保守/防御性屏障。** 拆解如下：

| 维度 | 事实 | 是否构成跨核依赖 |
|---|---|---|
| `statusTensor_` 归属 | 每核私有 UB（`statusBuf_`，L191；Init L455–465） | 否，无跨核共享 |
| Loop2 DataCopy 目的地 | 对端 rank 窗口，按 `expertIndex` 切分、各发送核写入互不重叠的 32B 槽 | 否，无写写冲突 |
| 接收侧 `WaitStatus`/`WaitStatusA5` | 轮询 flag 累加至 `compareTarget`（L1283–1307 / L1248–1279），晚到者仅延长轮询 | 否，对发送侧跨核顺序不敏感 |
| slot0 flag 来源 | Init 阶段 `Duplicate` 预置 `0x3F800000`（L465），SetStatus 运行时早已可见 | 否 |

⇒ 合并循环后，**唯一必须保留的依赖是每核内的 S→MTE3**（标量写 slot1 → 同块 DataCopy）。当前由中间 `PipeBarrier<PIPE_ALL>+SyncAll` 粗粒度覆盖；合并后改为 per-iter `SyncFunc<S_MTE3>` 细粒度门控即可。

> 备注：原方案把 `SyncAll` 放在两循环之间，主要起到**管控核间同步开销与时间差异**的作用（让所有核算完再统一发送）；提前到函数开头同样可行（见 §3.2），仅损失这部分差异管控，不影响正确性。

---

## 3. 实现方案

### 3.1 Phase 1：合并循环 + per-iter `SyncFunc<S_MTE3>`

目标函数体（替换 L1023–1066）：

```cpp
template <TemplateDispatchV2TypeClass>
__aicore__ inline void MoeDistributeDispatchV2<TemplateDispatchV2TypeFunc>::SetStatus()
{
    SplitToCore(totalExpertNum_, aivNum_, startExpertId_, endExpertId_, sendExpertNum_);
    SyncAll<true>();                 // 移到函数开头（防御性入口屏障）；须在早退 return 之前

    if (startExpertId_ >= totalExpertNum_) {
        return;
    }
    GlobalTensor<int32_t> rankGMTensor;

    for (uint32_t curExpertId = startExpertId_; curExpertId < endExpertId_; ++curExpertId) {
        int32_t curExpertCnt = 0;
        int32_t cntPosIndex = (curExpertId - startExpertId_) * 8 + 1;
        if ((curExpertId < sharedExpertRankNum_) && (activeMaskBsCnt_ > 0)) {        // shared 分支不变
            if (curExpertId % rankNumPerSharedExpert_ == epRankId_ % rankNumPerSharedExpert_) {
                curExpertCnt = activeMaskBsCnt_;
            }
        } else if (sendToMoeExpTokenCnt_ > 0) {                                      // MoE 分支不变
            int32_t curMoeExpertId = curExpertId - sharedExpertRankNum_;
            CalFlagSendExpertCnt(curMoeExpertId, sendToMoeExpTokenCnt_, curExpertCnt);
        }
        statusTensor_(cntPosIndex) = curExpertCnt;          // S 管写 slot1
        SyncFunc<AscendC::HardEvent::S_MTE3>();             // per-iter S→MTE3 门控（取代原中间 PipeBarrier+SyncAll）

        uint32_t dstRankId = curExpertId;                   // 地址计算与原 Loop2 完全一致
        uint32_t offset = stateOffset_ * epRankId_;
        if (curExpertId >= sharedExpertRankNum_) {
            dstRankId = ((curExpertId - sharedExpertRankNum_) / moeExpertNumPerRank_ + sharedExpertRankNum_);
            offset += ((curExpertId - sharedExpertRankNum_) % moeExpertNumPerRank_ * epWorldSize_ * stateOffset_);
        }
        if (isScalingDownFlag_) {
            dstRankId = elasticInfoTensor_.GetValue(ELASTIC_INFO_OFFSET + epWorldSizeOriginal_ + dstRankId);
        }
        GM_ADDR rankGM = (__gm__ uint8_t *)(GetWindStateAddrByRankId(dstRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t *)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(curExpertId - startExpertId_) * 8], 8UL);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();                 // 末尾 drain（保留原写法）
}
```

**每专家流水时序**：
```
CalFlagSendExpertCnt(V) → SetValue(S) → SyncFunc<S_MTE3> → DataCopy(MTE3, 异步) → 下一专家
```
`DataCopy(E)` 异步发出后立即推进下一专家；`CalFlagSendExpertCnt(E+1)`(V) 与 `DataCopy(E)`(MTE3) 重叠执行。

### 3.2 `SyncAll` 处理
- 上移到 `SplitToCore` 之后、早退 `return` 之前（**必须**保证所有核都到达，否则 `SyncAll<true>` 因部分核早退而死锁）。
- 作用：防御性入口屏障（确保所有核一起进入 / 上一阶段 `AlltoAllDispatch` 就绪）。
- 备选：完全移除（对齐本仓 full_mesh `CalAndSendCntByExp` 的无 `SyncAll` 形态），更省一次全局屏障——本次按"保留并上移"实施。

### 3.3 Phase 2（后续，打点驱动，不在本次）
若 Phase 1 后性能打点显示 MTE3 段仍为瓶颈，把循环内 `SyncFunc<S_MTE3>` 替换为旋转事件：

```cpp
// 循环前：uint32_t cntEvtId = 0;  （VALID_EVENT_FLAG_NUM = 8）
statusTensor_(cntPosIndex) = curExpertCnt;
AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(cntEvtId % VALID_EVENT_FLAG_NUM);
AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(cntEvtId % VALID_EVENT_FLAG_NUM);
... DataCopy ...
cntEvtId++;
```
末尾 `SyncFunc<MTE3_S>` 改为 `PipeBarrier<PIPE_ALL>()` 排空所有在途 MTE3。参照 step_split 设计与本仓 full_mesh token 流水的 ring-event 先例。

---

## 4. 涉及修改

| 位置 | 现状 | Phase 1 改为 |
|---|---|---|
| `SetStatus` L1026 后 | 直接进 Loop1 | `SplitToCore` 后插入 `SyncAll<true>()` |
| L1028–1042（Loop1） + L1050–1064（Loop2） | 两循环分离 | 合并为单循环，count 计算与地址计算/DataCopy 交错 |
| L1043–1044 | `PipeBarrier<PIPE_ALL>(); SyncAll<true>();` | 删除（SyncAll 已上移） |
| 合并循环内 SetValue 之后 | 无 | 新增 `SyncFunc<HardEvent::S_MTE3>()` |
| L1065 | `SyncFunc<MTE3_S>()` | 保留 |

> count 计算（`CalFlagSendExpertCnt`）、status 布局与初始化、地址计算（`dstRankId/offset/rankGM`）、`DataCopy` 载荷与目标**均不变**——仅同步粒度由"一次粗粒度全局屏障"改为"逐专家 S_MTE3"。

---

## 5. 数据结构

**无新增成员/缓冲**。复用：
- `statusTensor_`（`statusBuf_`）：每专家独占 8-int32 block（`cntPosIndex = (curExpertId - startExpertId_)*8 + 1` 互不重叠），尺寸/布局不变。
- `startExpertId_/endExpertId_/sendExpertNum_`：`SplitToCore` 切分结果。
- 常量：`STATE_OFFSET=32`、`WIN_STATE_OFFSET=384*1024`、`FLAG_OFFSET=8`（`op_kernel/moe_distribute_v2_constant.h`）。
- Phase 2 仅新增循环局部 `uint32_t cntEvtId`，复用 `VALID_EVENT_FLAG_NUM`（需在 v2.h 引入或与 full_mesh 对齐）。

---

## 6. 关键逻辑与正确性

### 6.1 S→MTE3 依赖等价
每专家 `SetValue`(S) → `SyncFunc<S_MTE3>` → `DataCopy`(MTE3)，等价原 `PipeBarrier<PIPE_ALL>+SyncAll` 对该依赖的覆盖：保证标量写入的 count 对本专家的 `DataCopy` 可见。

### 6.2 无缓冲复用 → 无回压
`statusTensor_` 每专家独占 block，`DataCopy(E)` 读 block E、`SetValue(E+1)` 写 block E+1，不同 offset 无冲突。故仅需单向 S→MTE3，无需 MTE3→S/V 回压，也无需循环前预置 flag。

### 6.3 slot0 flag 可见性
flag=1.0f 在 Init 阶段写入（L465），SetStatus 运行时早已可见，无需额外同步。

### 6.4 `SyncAll` 早退安全
早退 `return` 置于上移后的 `SyncAll` 之后，所有核必先到达 `SyncAll`，无死锁。

### 6.5 scratch 复用安全
`CalFlagSendExpertCnt` 内部用 `gatherMaskTBuf_/dstExpBuf_` 作 scratch，自带 `SyncFunc<V_S>/S_V` 边界；其 scratch 与 `DataCopy` 读取的 `statusTensor_` 无重叠，循环内多次调用无跨专家悬挂依赖。

### 6.6 接收侧不受影响
`WaitStatus`/`WaitStatusA5` → `SyncCntOnCore` → `WaitDispatchClearStatus` 均为轮询 + 管道级 `SyncFunc`，唯一的跨核 `SyncAll` 在 `WaitDispatch` 末尾（L1345）；发送侧合并循环与 `SyncAll` 上移不改变接收侧契约。

---

## 7. 与参考实现对照

| 维度 | 本仓 full_mesh `CalAndSendCntByExp` | step_split `CalAndSendCntByExp` | **本方案 Phase 1** |
|---|---|---|---|
| 循环结构 | 单循环（已合并） | 单循环（已合并） | 单循环（合并） |
| per-iter 同步 | `SyncFunc<S_MTE3>` | 旋转 `SetFlag/WaitFlag<S_MTE3>` | `SyncFunc<S_MTE3>` |
| `SyncAll` | 无 | 无 | 保留，上移到函数开头 |
| 末尾 drain | `PipeBarrier<PIPE_ALL>` | `PipeBarrier<PIPE_ALL>` | `SyncFunc<MTE3_S>`（保留原写法） |

Phase 1 = 本仓 full_mesh 的已验证形态 + 保留入口 `SyncAll`；Phase 2 = step_split 的旋转 event 升级。

---

## 8. 收益与风险

### 收益（Phase 1）
- **V 计算与 MTE3 发送重叠**：算 E+1 时发 E，掩盖 Vector 计数计算的耗时；专家数越多、BS·K 越大收益越显著。
- 消除中间全局 `SyncAll` 的跨核屏障等待（上移为一次性入口屏障）。

### 风险
1. **Phase 2 收益前提**：旋转 flag 的额外收益（MTE3 多笔并发）依赖"`SyncFunc<S_MTE3>` 是否串行化了 MTE3"。若 `SyncFunc<S_MTE3>` 仅约束 S→MTE3、不 drain MTE3，则 Phase 1 改前改后 MTE3 都已并发，Phase 2 收益有限——**必须 before/after 打点验证**（step_split 设计 §7 风险 1 同此）。若 Phase 1 后 MTE3 仍瓶颈且非串行所致，应转"同 rank 连续专家合并 strided DataCopy"减笔数。
2. **编译**：本机无 CANN 工具链，需 Ascend 构建环境确认 `SyncFunc<S_MTE3>`（Phase 1）/`SetFlag/WaitFlag<S_MTE3>`（Phase 2）编译通过。
3. **精度**：仅改同步方式、不改数据流，理论上对输出无影响；需回归确认接收侧 status 仍被正确聚合。

---

## 9. 测试方案

### 9.1 编译
Ascend 构建环境（A3/2201 目标）编译该 kernel，确认无告警。

### 9.2 精度
复用 moe_distribute_dispatch_v2 现有 UT，覆盖：
- 正常 BS×K、token mask（`isExpertMaskFlag_`）；
- shared + MoE 混合专家；
- 多 EP rank（flag 发往多远端 rank）、`moeExpertNum_ % aivNum_ != 0`；
- `isScalingDownFlag_`（elastic rank 重映射）；
- quant / non-quant。
断言 `expandXOut/expandIdxOut/expertTokenNumsOut` 与改前 allclose。

### 9.3 性能（核心）
before/after 打点 SetStatus 段（`PerformanceInfoCopyOut` / `GetSystemCycle`）：
- **该段耗时下降** → V↔MTE3 重叠生效；
- 关注大 EP（`sendExpertNum` 大、远端 rank 多）场景的收益放大；
- 记录数据供 Phase 2（旋转 flag）决策。
