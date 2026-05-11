# QAR/QRS 全阶段优化历程总结

> **算子**: QuantAllReduce (QAR) + QuantReduceScatter (QRS)
> **环境**: 4xAscend950PR (aivNum=56, AIV:AIC=2:1), CANN 9.0.0, world_size=4
> **测试**: 24 用例/算子, scene=one_hccl/one_hccl_npu

---

## 总览

```
baseline (硬编码1024)
  │
  ├─ stage1: UB最大化 → +5% 回退
  ├─ stage2: cap=2048 → -10% 改善 ✅
  ├─ stage3: TARGET_ITER公式 → -1.34% 改善 ✅
  ├─ stage4: MIN扫参 → 2048最优 ✅
  ├─ stage5: 全核ALIGN=256 → 劣化 ❌
  ├─ stage6: 读端DB/Leader/EnQue重排 → 不可行/持平 ❌
  └─ R-α: 多核分工 → 轻微退化 ❌

累计有效收益: baseline → -11%
```

---

## 第一阶段：分块策略探索

### baseline — 硬编码 xPerBlock=1024

xPerBlock 固定 1024 元素。简单但无法适应不同 shape。

### stage1 — UB 最大化

**做法**: 动态计算 UB 空间，允许 xPerBlock 跑到 ub_max≈7168。

**结果**: QAR +5.27%, QRS +2.14% 回退。

**失败原因**: xPerBlock 过大导致每核迭代数被压到 1-2。CopyDataToWin 中 TQueBind 的 DoubleBuffer 依赖 BUFFER_NUM=2 实现流水，迭代太少（1-2 轮）时 DB 无法满载——第一轮填缓冲，第二轮才开始重叠，没有第三轮来受益。同时活跃核数减少，负载不均衡。

### stage2 — cap=2048 硬上限

**做法**: kernel 内硬编码 `cap = 2048U` 限制最大 xPerBlock。

**结果**: QAR -9.95%, QRS -6.5~11.4% 改善。

**成功原因**: 每核 iter 回到 3-6，DoubleBuffer 流水满载。TQueBind 写端 DB（CopyDataToWin）在 ≥3 轮迭代时能有效重叠 MTE 搬运和计算。

**局限**: cap=2048 是经验值，不随 (xNums, aivNum) 自适应。

### stage3 — TARGET_ITER 公式

**做法**: host 侧 `xPerBlock = max(ceil(perCoreElem/TARGET_ITER), MIN_BLOCK)`，TARGET_ITER=3。

**结果**: QAR -1.34%, QRS 持平（-0.33%）。

**成功原因**: 公式自适应——QAR DSV3 系列 12 用例 xPerBlock 由 2048→4096（iter 6→3），减少同步开销 -0.9~3.6%。QAR qwen 系列公式自然落 2048，与 stage2 持平。QRS comm-bound，perCoreElem 太小，公式失效由 MIN=2048 兜底。

**关键发现**: QAR 和 QRS 物理特性不同——QAR compute-friendly（T_compute≈T_data，DB 重叠收益大），QRS comm-bound（T_compute<<T_data，DB 无重叠空间）。因此 QRS 少 iter 大块更优。

### stage4 — MIN 三点扫参

**做法**: QRS MIN 三点扫参（512/2048/4096），QAR MIN 1024→2048 统一。

**结果**:

| MIN | xPerBlock | DSV3活跃核 | qwen活跃核 | vs 2048 |
|-----|-----------|-----------|-----------|---------|
| 512 | 512 | 56/56 | 56/56 | +12.14% |
| 2048 | 2048 | 42/56 | 30/56 | 基线 |
| 4096 | 4096 | 21/56 | 15/56 | +14.35% |

**结论**: 2048 为局部最优。

**MIN=512 失败原因**: 块小 4 倍→iter 多 3 倍→通信 N*T_setup 开销线性增长。QRS comm-bound，减小 xPerBlock 启动的 DB 几乎无重叠收益，反而 setup 开销主导。

**MIN=4096 失败原因**: per-core 数据量翻倍（4096 vs 2048），CopyDataToWin + Execute 的每核耗时翻倍。T_var 翻倍直接加到 mc2_prof 上，减少核数节省的 ReadStatus 同步开销（~0.1us）远小于 per-core 耗时增加。

### stage5 — 全核启用 ALIGN=256

**做法**: 策略 A——全核公式，ALIGN 从 1024 降到 256。

**结果**: 劣化，不入主线。

**失败原因**: ALIGN 减小导致 xPerBlock 变小，和 stage4 MIN=512 类似——iter 增多带来的 setup 开销超过全核并行的收益。comm-bound 算子对 iter 数敏感，少 iter 大块是正确方向。

---

## 第二阶段：通信/同步优化

### DataCopyPad 鲁棒性提升（stage3）

**做法**: 将所有 scale 相关 DataCopy 替换为 DataCopyPad（4 处），支持 sub-32B 对齐。

**结果**: 鲁棒性提升，性能持平。为后续 sub-1024 对齐实验提供接口。

### WriteStatusToWin 复位外提（stage6）

**做法**: WriteStatusToWin 内每 rank 循环重复 DataCopy+SyncFunc 复位 statusTensor。改为循环外初始化一次，循环内写完 element 0 后再归零。省 4×(DataCopy+SyncFunc) 每核。

**结果**: 精度全过，性能持平。收益太小测不出，但代码逻辑更清晰。

### Leader 模式状态同步（stage6）

**做法**: Process 中 CopyDataToWin → WriteStatusToWin（全员写）→ SyncAll（AIV 全核同步）→ Execute。Execute 内 ReadStatus 改为 leader 模式：仅 core 0 跨卡自旋 readsStatus，确认后 SyncAll 广播，其余 55 核 SyncAll 等通知。

**结果**: 精度全过，性能持平（QRS +0.29%, QAR -0.42%）。

**持平原因**: 跨卡状态通信节省（ReadStatus 的 224→4 次跨卡读）被两次 SyncAll 屏障开销抵消。SyncAll 需等最慢核到达，在 QRS 这种负载不均（42/56 核活跃）的场景下，空闲核的 SyncAll 等待时间恰恰等于优化省下的时间。

**关键发现**: `SyncAll()` 不能加模板参数 `<false>` / `<true>`，裸调用才对。两种模板参数均导致 507035 kernel timeout。需配合 tiling 侧 `context->SetScheduleMode(1)`（batch mode）。纯 AIV 算子（AIC=0）无需 `ASCEND_IS_AIC` 保护。

---

## 第三阶段：读端 MTE+Vector 重叠探索

### TQueBind 读端 DoubleBuffer

**做法**: 将 `xInQueue_`/`scaleInQue` 从 `TQue<VECIN,1>` 改为 `TQueBind<VECIN,VECOUT,N>`（尝试 position 1/2）。拆分 ReadDataBlockReduceSum 为 PreFetchBlock（异步 EnQue）+ WaitAndCompute（DeQue+计算）。Execute 的 rank 循环改为预取流水——发起下一个 rank 的 Win→UB 搬运同时计算当前 rank。

**结果**: 精度全挂（0/192 PASS）。同步模式（不拆流水线，仅换 TQueBind）也全挂。

**失败原因**: 整个 ops-transformer 代码库中，TQueBind 的 DeQue 之后只能接 **MTE DataCopy**（写回 GM/Win），没有一例接纯 Vector 计算。TQueBind<VECIN,VECOUT> 的 VECOUT DeQue 在当前 TPipe 配置下与 Vector 流水线之间存在同步缺陷——DeQue 等的是 VECOUT 缓冲就绪信号，但该信号可能在 MTE 搬运真正完成前就触发了。

### x/scale EnQue 重排

**做法**: ReadDataBlockReduceSum 内把 x 和 scale 的 DataCopy 都发出后再一起 DeQue（EnQue x → EnQue scale → DeQue x → DeQue scale），替代原来的 EnQue x → DeQue x → EnQue scale → DeQue scale。期望 x 和 scale 的 MTE 并行搬运。

**结果**: 精度全过，但 QRS +12% 回退，QAR 持平。

**失败原因**: scale 数据量极小（2048 个 x 对应仅 64B scale），MTE 瞬间完成，重叠收益为零。额外 VECIN 操作（多一步 EnQue 入队）在 QRS 短耗时（~10us/case）场景下调度开销占比 ~10%。

### MOE 算子借鉴

**考察模式**:
- combine_v2: HCCL 异步通信+计算重叠 → QAR/QRS 走 Win Area 软同步，不走 HCCL，不适用
- dispatch quant: 双 TQue<VECIN>+TQue<VECOUT> 三级流水（读→算→写）→ 读端只有两级（读→算），结构不匹配
- weight_quant_bmm: 精细 SetFlag/WaitFlag 事件同步 → 复杂度高，收益不明

**结论**: 无可用模式。

---

## 第四阶段：架构探索

### R-α 多核分工

**做法**: AIV 主核/辅核分工——不同核负责不同 remote rank 的数据处理，分核策略尝试减少跨核竞争。

**结果**: 轻微退化。

**失败原因**: 增加的分工协调开销（跨核事件同步）超过并行收益。QAR/QRS 的 Win 区数据已按 aivId 分块，核间天然无竞争，强行分工反而破坏这一特性。

---

## 最终成绩

| 指标 | baseline | stage3 (最优) | 有效改善 |
|------|----------|--------------|---------|
| QAR 24 用例合计 | ~735 us | 661.93 us | **-10%** |
| QRS 24 用例合计 | ~260 us | 241.98 us | **-7%** |
| 精度 | PASS | 192 PASS | — |

### 有效优化清单

1. **stage2**: cap=2048 恢复 DoubleBuffer 流水（-10%）
2. **stage3**: TARGET_ITER 公式自适应 + DataCopyPad（-1.3%）
3. **stage4**: MIN 扫参确认 2048 最优
4. **stage6**: WriteStatus 复位外提 + Leader 模式（持平，代码优化）

### 不可行方向清单

1. UB 最大化（stage1）— iter 太少 DB 失效
2. xPerBlock 减小（stage4/5）— comm-bound setup 开销主导
3. TQueBind 读端 DB（stage6）— VECOUT→Vector 同步缺陷
4. EnQue 重排（stage6）— scale 太小无重叠收益
5. R-α 多核分工 — 协调开销 > 并行收益

---

## 遗留优化方向

1. **CrossCoreSetFlag/WaitFlag 替代 SyncAll**: 当前 Leader 模式两个 SyncAll 开销抵消了跨卡通信节省。事件标志比全屏障轻量，可能降低开销让 Leader 模式产生正收益。尚未实验。

2. **QRS 核利用率（并行 rank 读）**: qwen 形状 30/56 核空闲（54%）。已证伪：单纯减小 xPerBlock 激活更多核——MIN=512 劣化 +12%，MIN=1024 劣化 +25%，MIN=4096 劣化 +14%，2048 为局部最优。原因：comm-bound 算子块大小比核数更重要，小块 DB 效率损失 > 全核并行收益。替代思路：不改 xPerBlock，将 block 内 4 个 rank 的串行读拆成 4 核并行读，每核只负责 1 个 rank 的 Win→UB 搬运和反量化，最后跨核累加 partial sum。理论上 30×4=120 个任务分给 56 核可全核激活且不破坏块大小约束。实现量大（需改分块分配、跨核归并、同步机制），待评估优先级。
