# MegaMoe Dispatch 大 bs 适配设计文档（分块流式 / Chunked-Streaming Dispatch）

> 适用文件：
> - `mc2/mega_moe/op_kernel/arch35/mega_moe.h`
> - `mc2/mega_moe/op_kernel/arch35/mega_moe_base.h`
>
> 目标读者：算子内核维护者 / Reviewer。
> 一句话概括：把 dispatch 流水里所有按 `bs·topK` 整块分配在 UB 上的缓冲，改成按固定
> `CHUNK` 个 token **分批流式**处理，使 UB 占用与 `bs` 解耦，从而支持任意大的 `bs` 而不 OOM；
> 算法不变、dispatch 结果与改前**逐位一致**、win/HBM 布局与 host 侧分配**完全不变**。

---

## 1. 背景与问题

### 1.1 硬件约束
- 每个 AIV 核独立拥有 **256KB Unified Buffer (UB)**；aiv0 / aiv1 各自独立的 256KB。
- 所有 `LocalTensor` 都铺在这块 UB 上，是稀缺资源。

### 1.2 原实现的 UB 占用与 bs 耦合
记 `ST = sendTotalNum_ = bs · topK`。原 dispatch 流水里几乎每个关键 `LocalTensor` 都按 `ST` 整块分配：

| 缓冲 | 原大小 | 所在阶段 |
|---|---|---|
| `topkIdsTensor_`（本卡完整 topk） | `4·ST` 字节 | 源卡 SendMaskCal |
| `sendGatherOutTensor_`（GatherMask 计数废弃输出） | `4·ST` | 源卡 SendMaskCal |
| `gatherMaskTensor_`（本 expert 的 ws 个 mask 槽） | `ws · ST/8` | 接收端 SendInfoCal/Triple |
| `validTopkIndexTensor_`（命中 token 位置） | `4·ST` | 接收端 Triple |
| `topkIndexTensor_`（0..ST-1 索引） | `4·ST` | 接收端 Triple |
| `tripleTensor_`（动态，每核派发行的三元组） | `rows·32B`（∝ bs） | 接收端 Triple |
| `topKWeightsTensor_`（unpermute 权重） | `bs·topK·4` | 时段 C（§6.5 已改为每核段） |

当 `bs` 很大时这些之和远超 256KB → **OOM**。例如 `bs=10^5, topK=8` ⇒ `ST=8·10^5`，仅
`topkIdsTensor_` 就要 3.2MB。

### 1.3 关键观察：瓶颈是 UB，不是 win/HBM，也不是计数溢出
- **win / HBM 区**（`maskRecvPtr` / `quantTokenScalePtr` / `dispatchRevDataPtr` 等）由 host 按
  `tilingData->bs` / `maxOutputSize` 分配在 HBM（GB 级），天然容纳大 bs，**不是瓶颈**。
- 之前讨论的"`compareCount` 超 uint64"并不会发生：`compareCount`/`calCount` 都是 32 位语义，
  真正先爆的是 256KB UB。
- 因此这是**纯 UB 问题、纯 kernel 改动**，理想情况下不应触碰 host / win 布局。

---

## 2. 设计目标与约束

1. **UB 占用与 bs 解耦**：所有 `O(ST)` 缓冲降为 `O(CHUNK)`，循环 `numChunks_=ceil(ST/CHUNK)` 次处理。
2. **dispatch 结果逐位一致**：`dispatchRevData` 的 row→token 映射、`expertTokenNumsOut`、最终 `y`
   必须与改前完全相同（MoE dispatch 对行顺序敏感）。
3. **不改 win 布局 / host CCL buffer 大小**：`maskSlotSize_` 公式与改前数值一致，避免 host
   侧分配不匹配（详见 §4.1）。
4. **不引入性能劣化**：对原本能跑的 bs 规模保持中性；对新支持的大 bs 把分段开销降到最低。

---

## 3. 核心思路：分块流式（Chunked Streaming）

整条 dispatch 由三个被"跨卡同步"分隔的阶段组成，每个阶段内部都改成按 token 分批：

```
源卡 SendMaskCal      →  [跨卡 SyncAll]  →  接收端 SendInfoCal + TripleInfoCalAndDispatch
(算 mask 写到目标卡)                          (读 count 建 cumsum, 按 mask 取 token 派发)
```

> **通信机制说明**：本算子**AIV 直驱 MTE** 跨卡搬运——
> `DataCopyPad` / `DataCopy` 由 AIV 直接把数据**写到 / 读自对端卡的 win 内存（peer mem）**。
> 本文中"写对端 / 推送 / 搬运 / 跨卡写"均指这种 AIV 直驱 MTE 访存；
> 性能上关心的是**跨卡 MTE 搬运的次数**（指令/描述符数）与**跨卡访存延迟**，而非 RDMA。

- 分批粒度：`CK = chunkLenA_`(发)/`chunkLenB_`(收)（成员）**在 `Init` 运行时按 UB 余量动态算**，不再是编译期常量（见 §6.3）。
  `mega_moe_base.h` 仅保留 `DISPATCH_TRIPLE_TILE = 2048`（TT: `tripleTensor_` 每批最多行数，每行 32B）。
  ```cpp
  // Init 内: 发 chunkLenA_=min(ckMaxA, CeilAlign(ST,256)), 收 chunkLenB_=min(ckMaxB, CeilAlign(ST,256));
  //          各自向下取整到 256 倍数(发/收独立, 不取 min)。mask 切片字节 = CK/8（CK 为 256 倍数 ⇒ 天然 32B 对齐）
  constexpr int32_t DISPATCH_TRIPLE_TILE = 2048;
  ```
- `CK` 越大 ⇒ 大 bs 时分片数越少 ⇒ 源跨卡 MTE 搬运次数 & 接收端迭代越少；`CK=ST` 时单块跑（见 §6.3）。

### 3.1 整体调用骨架（修改前后**不变**）+ 数据流

dispatch 段在 `Process()` 里的调用骨架，以及卡间数据流，**修改前后完全相同**——变化只在
`SendMaskCal` / `SendInfoCal` / `TripleInfoCalAndDispatch` **内部**（把"一次处理全 ST"改成
"循环 `numChunks_` 次、每次处理 `CK`"）：

```
每张卡同时是"源卡"和"目标卡"。dispatch 段时间线（AIV）：

  BuffInit
     │
     ▼
  SendMaskCal ──── AIV 直驱 MTE 写对端 ───►  各目标卡 win 的 maskRecv 区
     │                                        布局 [localExpert][srcRank], 每槽 = [bit-mask | count]
  [PipeBarrier + SyncAll]   ← 确保所有卡的 mask 写都已落地
     │
  CrossRankSyncInWorldSize  ← 跨卡握手
     │
  FirstBuffInit
     │
     ▼   for localExpert in 0..expertPerRank:
            SendInfoCal              读本卡 maskRecv[le][*] 的 count → 建 cumsum 表
            UpdateGroupParams/Buffer
            TripleInfoCalAndDispatch 按 maskRecv[le][srcRank] 的 bit 取命中 token,
              │                       ── AIV 直驱 MTE 读对端 ──►  从源卡 win.quant 读量化 token
              ▼
            写本地 workspace.dispatchRevData ──►  GroupMatmulSwigluQuant (GMM1) ...
```

### 3.2 UB 占用：修改前 vs 修改后（**为什么要改**）

`ST = bs·topK`，`CK = 8192`，`TT = 2048`，`ws = worldSize`。核心问题：原实现的关键缓冲都 `∝ ST`。

```
              ┌─────────────────────────── 修改前（一次处理全 ST）───────────────────────────┐
 源 SendMaskCal│  topkIds[4·ST]  +  maskScratch[4·ST]                              ≈ 8·ST     │
 接收 Triple   │  validTopkIdx[4·ST] + topkIdx[4·ST] + gatherMask[ws·ST/8]                    │  全部 ∝ bs
              │  + tripleTensor[rows·32  (rows ∝ bs)]                                         │  ⇒ 大 bs 时
              └──────────────────────────────────────────────────────────────────────────────┘   累计 > 256KB
                                                                                                   ⇒ OOM
              ┌─────────────────────────── 修改后（每次处理 CK=8192）──────────────────────────┐
 源 SendMaskCal│  topkChunk[4·CK]  +  maskScratch[4·CK]                            ≈ 8·CK     │
 接收 Triple   │  validTopkIdx[4·CK] + topkIdx[4·CK] + maskChunk[CK/8]                         │  与 bs 无关
              │  + tripleTensor[≤ TT·32]                                                        │  (常数)
              └──────────────────────────────────────────────────────────────────────────────┘   ⇒ 任意 bs
                                                                                                   段A≈118KB
                                                                                                   段B≈179KB
```

> 直觉：把"一次把 `ST` 个 token 全摊在 UB 上"改成"每次只摊 `CK=8192` 个"。`CK` 是常数，
> 所以 UB 占用不再随 `bs` 涨；代价是同一份活儿循环 `numChunks_ = ceil(ST/CK)` 次做完。

### 3.3 三个关键函数：修改前 vs 修改后 伪代码

#### (a) `SendMaskCal`（源卡：算 mask 写到目标卡）

```text
# ───────── 修改前：outer = 专家，一次算全 ST ─────────
load topk[0:ST] → topkIds            # UB 4·ST 字节  ◄ O(ST)，大 bs 放不下
for g in 本核负责的全局专家:           # 外层 = 专家
    mask = (topkIds == g) over ST     # 全 ST 的 bit-mask
    cnt  = popcount(mask)             # 借 GatherMask，scratch 4·ST  ◄ O(ST)
    mask[countField] = cnt
    MTE 写对端: win[dst].slot[le][srcRank] ← (mask + count)   # 整槽一次写

# ───────── 修改后：outer = chunk / inner = 专家，分批 ─────────
cntAcc[numMyExperts] = 0
for c in 0 .. numChunks_:                          # ★新增外层 = chunk
    load topk[c·CK : c·CK+CK] → topkChunk          # UB 4·CK  ◄ O(CK)；尾部填 -1 哨兵
    for (e, g) in 本核负责的专家:                    # 内层 = 专家
        maskSlice = (topkChunk == g) over CK        # 本块 CK/8 字节切片
        cntAcc[e] += popcount(maskSlice) over validLen
        若 c==末块: 把 cntAcc[e] 折叠进切片尾部       # ★count 折叠(§6.1)
        MTE 写对端: win[dst].slot[le][srcRank] + c·CK/8 ← maskSlice(+末块的 count)
```
**关键差异**：① `topk` 从"全 ST 常驻"→"每块 `CK` 常驻"（O(ST)→O(CK)）；② 多一层 chunk 外循环；
③ mask 从"整槽一次写"→"按片写、末块折叠 count"；④ **写进 win 的内容/布局完全一致**（同一槽、同样的 bit、同一个 count）。

#### (b) `SendInfoCal`（目标卡：建 cumsum 表）

```text
# ───────── 修改前：把 ws 份 mask 全读进 UB ─────────
MTE 读本卡 win: maskRecv[le][0:ws] → gatherMask     # ws·slot ≈ ws·ST/8  ◄ O(ST)
for s in 0..ws: cnt = gatherMask[s].countField; cumsum += cnt    # 顺带为 Triple 把 mask 留驻 UB

# ───────── 修改后：只读 count，不读 mask ─────────
MTE 读本卡 win(带 stride): maskRecv[le][*].countField → cntTensor[ws]   # 仅 ws·4 字节  ◄ O(ws)
for s in 0..ws: cumsum += cntTensor[s]
# mask 不再常驻 → 由 Triple 按 (dstRank,chunk) 现读现用
```
**关键差异**：从"读 `ws·ST/8` 的全 mask"→"只读 `ws·4` 的 count"（O(ST)→O(ws)）；mask 的加载后移到 Triple。

#### (c) `TripleInfoCalAndDispatch`（目标卡：取命中 token 派发）

```text
# ───────── 修改前：每 (expert,dstRank) 一次全 ST GatherMask ─────────
for idx in 0 .. ws·blockNumPerRank:                 # dstRank-block 分核
    dstRank = idx / blockNumPerRank
    validTopkIdx = GatherMask(0..ST-1, mask[dstRank]) over ST    # in/out 各 4·ST  ◄ O(ST)
    cnt = popcount = sendToCurExpTokenCnt                        # 命中总数(扫全 ST 得来)
    [行→核划分: rowsNumInCore, rowSrcIdxInCore, rowDstOffset, maxOutput clamp]
    if rows > 0:
        tripleTensor = UB[ubUsed, rows·32]          # rows ∝ bs  ◄ O(ST)
        CopyGMToGMPerToken(rowDstOffset, dstRank, rowSrcIdxInCore, rows)  # MTE 读对端 token 派发
    wave-grain AtomicAdd(本核连续行区间)

# ───────── 修改后：单趟分块；命中总数由 cumsum 推出 ─────────
for idx in 0 .. ws·blockNumPerRank:
    dstRank = idx / blockNumPerRank
    cnt = cumsum[le, dstRank] - cumsum[le, dstRank-1]            # ★免去全 ST GatherMask(§4.2)
    [行→核划分: coreLo, coreHi, rowDstOffset —— 数学与改前逐字相同]
    matchedSoFar = 0; rowsThisCore = 0
    for c in 0 .. numChunks_  while matchedSoFar < coreHi:       # ★分块扫描
        MTE 读本卡 win: mask 切片[dstRank, c] → maskChunk        # CK/8  ◄ O(CK)
        idxVec = CreateVecIndex(c·CK .. c·CK+CK)                 # 本块全局 token 位置
        matchedInChunk = GatherMask(idxVec, maskChunk) over validLen   # in/out 各 4·CK  ◄ O(CK)
        [lo,hi) = [matchedSoFar, matchedSoFar+matchedInChunk) ∩ [coreLo, coreHi)   # 与本核区间求交
        if hi > lo:
            for off in 0 .. (hi-lo) step TT:                     # ★triple-tile 分批(§6.2)
                n = min(TT, hi-lo-off)
                tripleTensor = UB[ubUsed, n·32]                  # ≤ TT·32，与 CK/bs 解耦
                CopyGMToGMPerToken(rowStart+lo+off, dstRank, (lo-matchedSoFar)+off, n)
            rowsThisCore += (hi-lo)
        matchedSoFar += matchedInChunk
    wave-grain AtomicAdd(本核连续行区间)   # 循环后一次，桶算法与改前逐字相同
```
**关键差异**：① 命中总数由 **cumsum 推出**，省掉每 `(expert,dstRank)` 的全 ST GatherMask；
② mask **现读现用**，`validTopkIdx`/`topkIdx` 从 `4·ST`→`4·CK`（O(ST)→O(CK)）；
③ `tripleTensor` 从 `rows·32`(∝bs) → `≤ TT·32`（triple-tile，与 bs 解耦）；
④ **行→token 映射逐位一致**（chunk 升序扫 + GatherMask 升序压缩 ⇒ 命中顺序不变，§4.2）。

---

### 3.4 接收侧详解：mask 与 count 的角色，以及怎么"分批拿"

§3.3 给的是发送/接收三函数的伪代码对比；本节专门把**接收侧**讲透——先讲 win 里到底存了什么、
mask 和 count 各自干嘛，再讲两个接收函数（`SendInfoCal` / `TripleInfoCalAndDispatch`）改前改后
**分别怎么拿这两样东西**。读懂这一节，就能明白"为什么 win 布局不变、但接收端能不爆 UB"。

#### 3.4.1 win 槽位里存的两样东西

源卡 `SendMaskCal` 算完，往**目标专家所在卡**的 win 区写。布局 `[localExpert][srcRank]`，
每个 `(本卡某个本地专家, 某个源卡)` 组合占一个**槽位 slot**：

```
slot 布局（每个 [localExpert][srcRank] 一个，大小 maskSlotSize_）：
┌──────────────────────────────────────────────┬──────────┐
│  mask 区 (maskAlignSize_ 字节)                 │ count区  │
│  = bit-packed，每个 token 占 1 bit             │   32B    │
│  第 i 位 = 1 → 源卡第 i 个 token 要发给本专家  │ (命中总数)│
└──────────────────────────────────────────────┴──────────┘
   ←————— maskAlignSize_ —————→                    ←— 32B —→
```

两个字段作用完全不同，**这是理解整个接收侧的关键**：

| 字段 | 内容 | 回答的问题 | 给谁用 |
|------|------|-----------|--------|
| **count** | 一个标量：这个源卡发给这个专家**多少个** token | **"几个？"** | `SendInfoCal` → 建 cumsum（定位行区间） |
| **mask** | bit 位图：**具体是哪些** token（第几个 token = 第几个 bit） | **"是哪几个？"** | `TripleInfoCalAndDispatch` → 还原 token 实际下标去搬数 |

> 一句话：**count 回答"几个"，mask 回答"是哪几个"**。
> count 是 mask 里 1 的个数——源卡 `SendMaskCal` 做 `GatherMask` 时顺手数出来、直接写进 32B count 字段，
> 接收端**不用自己再扫 mask 去数**。这正是 §4.2"单趟派发"能成立的前提。

#### 3.4.2 两个接收函数的分工

```
对本卡每个 localExpert：
  ① SendInfoCal(localExpert)              ── 只需要 count，建 cumsum 表
  ② TripleInfoCalAndDispatch(localExpert) ── 需要 mask，还原 token 下标，搬数据
```

- **① SendInfoCal 只碰 count。** 它要回答"本专家从 ws 个源卡共收多少 token、每个源卡的边界在哪"，
  这只跟 count 有关，mask 一个 bit 都不看：

  ```
  for srcRank in 0..ws:
      cnt = count[localExpert][srcRank]
      累加 → cumsumInfoTensor_[le*ws + srcRank]      # 前缀和
  ```

  cumsum 表的含义：`cumsum[le*ws + r]` = 本专家从源卡 `0..r` 收到的 token 总数。
  于是源卡 `r` 的 token 落在 dispatchRevData 的行区间 `[cumsum[..r-1], cumsum[..r])`。
  **后面 Triple 全靠这张表定位行，不用再扫 mask 数数。**

- **② TripleInfoCalAndDispatch 必须读 mask。** cumsum 只告诉你"源卡 r 发了 N 个、占第几到第几行"，
  但**没告诉你这 N 个具体是源卡 token 流里的第几个**（是第 3、第 17、第 200 个？）。
  要把数据真正搬过来，必须知道 token 实际下标 → 对 mask 做 `GatherMask`：

  ```
  CreateVecIndex → [0,1,2,...]      # token 的全局下标
  GatherMask(下标, mask) → 只留 mask=1 的下标 = validTopkIndex
  # validTopkIndex[j] = 第 j 个命中 token 的实际下标，拿它去 win 搬 token 数据 + scale
  ```

#### 3.4.3 改前 vs 改后：差别只在 mask 怎么拿

**count 的处理两版几乎一样**（都是读 32B、建 cumsum）。真正的改动全在 **mask** 上——因为 mask 大小是
`ST/8` 字节、`GatherMask` 的下标表是 `ST` 个 int，这俩才是随 bs 爆 UB 的元凶。

| | **改前（全量）** | **改后（分块）** |
|--|------------------|------------------|
| **SendInfoCal 取 count** | `DataCopy` 把 ws 份 **整槽 mask** 全读进 UB 常驻，再从常驻副本里捎带读 count | 用 **stride 的 `DataCopyPad`**（blockCount=ws, blockLen=4B, srcStride=maskSlotSize_−4）**只挑出 ws 个 count 字段**，mask 一个字节都不读 |
| **Triple 取 mask** | `CreateVecIndex(0..ST)` 一次性 + 全 ST 一把 `GatherMask`，得到全部命中下标，再按 `[coreLo,coreHi)` 切本核行 | 升序扫 `numChunks_` 个 chunk，**每次只 `DataCopy` 一个 chunk 的 mask 切片**（CK/8 字节），块内 `GatherMask`，边扫边和本核区间求交、即时搬运 |
| **mask 是否常驻 UB** | 是，`gatherMaskTensor_` 占 **ws·ST/8** | 否，`maskChunk_` 只占 **CK/8**，现读现用 |
| **下标表 UB** | `topkIndex_`/`validTopkIndex_` 各 **4·ST** | 各 **4·CK** |
| **UB 量级** | **O(ST)** → 大 bs OOM | **O(CK)** 常数，与 bs 解耦 |

改后 Triple 的分块扫描，几个一眼容易绕的点：

1. **为什么还能定位行？** 本核负责区间 `[coreLo,coreHi)` 是 **SendInfoCal 的 cumsum 表**算出来的
   （`cnt = cumsum[le*ws+r] − cumsum[le*ws+r−1]`），**不依赖扫 mask**。改前是"先 GatherMask 全 ST 再切"，
   改后是"先用 cumsum 算好区间，再边扫 chunk 边对号入座"。
2. **`matchedSoFar` 是什么？** "扫到当前 chunk 之前、前面 chunk 一共命中了多少个"。用它把 chunk-local 的命中
   编号换算成全局命中编号，才能跟 cumsum 给的 `[coreLo,coreHi)` 对齐。`matchedSoFar ≥ coreHi` 即 break，
   本核活干完，后面 chunk 不再扫（高 `aivIdxInGroup` 核仍要空扫前导块，是已知、已接受的取舍，见 §6.4）。
3. **为什么结果 bit-identical？** 升序扫 chunk + `GatherMask` 块内升序压缩 ⇒ 各 chunk 命中首尾拼接的顺序，
   跟改前"全 ST 一把 GatherMask"逐一相同；第 j 个命中 token 仍落到第 j 行（§4.2）。

> **总览**：win 布局与最终内容两版**完全一致**（发送侧已确认）；接收端变的只是
> ——**count 改成 stride 直接挑出来读**（不再借道常驻 mask 副本），
> ——**mask 改成一段一段现读现用**（不再整块常驻 + 一把 GatherMask）。

---

## 4. 关键设计决策与"为什么"

### 4.1 win 槽位大小**保持不变**（末块 clamp，而非按 chunk 粒度扩大）

**问题**：mask 在 win 槽内是 bit-packed 的，槽 mask 区大小
`maskAlignSize_ = align32(compareCount/8)`（`compareCount = align256(ST·4)/4 ≈ ST` 向上取整到 64 的倍数）。
若源卡按"每块满 `CK/8`、共 `numChunks` 块"写，`numChunks·CK/8` 可能**大于** `maskAlignSize_`
（因为 `CK=8192` 把 ST 向上对齐到 2048 的倍数，而 `cc` 只对齐到 64），从而**越界写进 count 字段甚至越槽**。

**两种解法**：
1. ❌ 把 `maskAlignSize_` 改成 `numChunks·CK/8`（chunk 粒度）。→ 槽变大 → win mask 区变大 →
   把后面的 `quantTokenScalePtr`/`combineSendPtr` 往后挤。而 host 的 `leastCclBufferSize`
   校验（`mega_moe_tiling.cpp`）**根本没把 mask 区算进去**（用的是历史遗留的 `tokenPerExpert` 项），
   不会拦截这种增长 → **可能悄悄写爆 host 分配的 CCL buffer**。风险高且本地无法验证。
2. ✅ **保持 `maskSlotSize_` 与改前逐字一致**，源卡写 / 接收端读时对**最后一块 clamp**到
   `maskAlignSize_ - chunkStart/8` 字节，正好填满 mask 区、不越界 count。

**选定方案 2**。这让本次改动变成"同样大小的槽，只是改了填充方式"的**纯原地重构**，
win 布局 / host CCL buffer 大小**零变化**。

**clamp 不变量**（对 `ST=1..600000`、`CK∈{2048,8192}` 数值穷举验证全部成立）：
- A: `maskAlignSize_ ≥ ceil(ST/8)`（mask 区放得下全部 bit）；
- B: `maskAlignSize_ ≤ numChunks·CK/8`（所以末块需要、且可以安全 clamp）；
- C: 每块切片字节数 `>0` 且 **32B 对齐**（`DataCopy`/`DataCopyPad` 合法）；
- D: `chunkStart/8 + sliceBytes ≤ maskAlignSize_`（绝不写进 count 字段）；
- E: **只有最后一块**可能被 clamp，其余块都是满 `CK/8`；
- F: 每块读到的字节 `sliceBytes·8 ≥ validLen`（覆盖本块全部有效 bit）；
- H: **末块** `chunkStart/8 + sliceBytes == maskAlignSize_`（保证 §6.1 的 count 折叠落点正确）。

### 4.2 接收端**单趟**派发（用 cumsum 推总数，免去预扫）

原 `TripleInfoCalAndDispatch` 对每个 `(localExpert, dstRank)` 用一次**全 ST** 的 `GatherMask` 拿到
命中总数 `sendToCurExpTokenCnt`，再把这些行按 `blockNumPerRank_` 个核划分。

**关键观察**：这个总数其实**已经在 cumsum 表里**：
```
sendToCurExpTokenCnt = cumsumInfoTensor_[le·ws + dstRank] - cumsumInfoTensor_[le·ws + dstRank - 1]
```
（`cumsumInfoTensor_` 是 `SendInfoCal` 用各源卡 count 建的 [expert][srcRank] 前缀和。）

**因此分块后无需"先扫一遍拿总数再分核"的两趟方案**，可以**单趟**：行→核的划分数学
（`rowsNumInCore = CeilDiv(rowsCopyCnt, blockNumPerRank_)`、`rowSrcIdxInCore = aivIdxInGroup·rowsNumInCore`、
`maxOutputSize_` clamp 等）与改前**逐字相同**，只是把"对全 ST 的一次 GatherMask + 一次派发"换成
"逐 chunk 的 GatherMask + 与本核命中区间 `[coreLo, coreHi)` 求交后派发命中的 sub-slice"。

**为什么行映射仍逐位一致**：chunk 按 token 升序扫描，`GatherMask` 在块内也是升序压缩，
拼接各块命中列表 == 改前对全 ST 的一次命中列表（顺序相同）；第 `k` 个命中 token 落到第
`rowDstOffsetInCore + (k - coreLo)` 行，与改前完全一致。
（已用 C++ 仿真对 `CK∈{256,2048,8192}` × `TT∈{1,2048,∞}` × 各种密度/核数/maxOutput clamp 共
**26460 组**用例验证 row→token 映射逐位相等。）

### 4.3 源卡循环嵌套：`outer = chunk, inner = owned experts`

- 选 outer=chunk：本 chunk 的 topk **只 DataCopyPad 一次**，被本核负责的所有全局专家复用 →
  topk 的 MTE2 总量 = `ST`（每 token 载一次）。
- 反过来 outer=expert 会让每个专家都重载一遍 topk（`numMyExperts × ST`），更差。

---

## 5. 逐文件 / 逐函数改动说明

> 想先看"改前 vs 改后"的对比与伪代码，见 **§3.3**；本节是逐项**详细参考**（含同步/flag 细节）。

### 5.1 `mega_moe_base.h`

| 改动 | 说明 / 为什么 |
|---|---|
| **保留 `DISPATCH_TRIPLE_TILE`(2048)** | `CK` 改为 `Init` 运行时算的 `chunkLenA_`/`chunkLenB_`（§6.3），原 `DISPATCH_TOKEN_CHUNK`/`DISPATCH_MASK_SLICE_BYTES` 两个常量已删除。 |
| **新增 `DISPATCH_RESET_BATCH`(2048 int32=8KB)** | `resetTensor_` 清零分批粒度（§6.7），使 reset UB 与 bs 解耦。 |
| **`PeermemInfo` 的 `maskAlignSize` 公式** | **数值与改前一致**（`align32(compareCount/8)`），仅更新注释说明"分块不改槽位大小，末块 clamp"。→ win 布局 / host 分配零变化（§4.1）。 |

### 5.2 `mega_moe.h` — 成员与常量

| 改动 | 说明 |
|---|---|
| 删 `int32_t compareCount_` 成员 | 分块后不再需要常驻；`maskAlignSize_` 在 `Init` 用局部变量算。 |
| 新增 `int32_t numChunks_` | `= ceil(ST/CK)`，整条流水的分批数。 |
| 新增 `int32_t resetBatchLen_` | `= min(每核 flag 份额, DISPATCH_RESET_BATCH)`，`resetTensor_` 封顶份额 & `ResetFlagList` 分批步进（§6.7）。 |
| 删 `topkTensors_[6]`（早先 #1） | 旧版 SendInfoCal 跨卡读 topk 用，mask 前移源卡后已死代码，省 `6·4·cc`。 |
| 删 `gatherMaskTensor_` / `gatherMaskInt32Tensor_`（`ws·slot`） | 分块后 mask 不再常驻，接收端按 (dstRank,chunk) 现读现用。 |
| `topkIdsTensor_` → 重命名 `topkChunk_`（`4·CK`） | 语义从"全 ST topk"变为"当前 chunk 的 topk"。 |
| 新增 `maskChunk_`(u8) / `maskChunkU32_`(u32，同址别名，`CK/8`) | 接收端单个 (dstRank,chunk) 的 mask 切片；u8 给 DataCopy、u32 给 GatherMask。 |
| 新增 `sendCntTensor_`（`ws*32B`，每 rank 一个 32B 对齐 burst） | 接收端 `SendInfoCal` 用 stride **只读 count**，下标 `r*8`（§5.7）。 |
| 新增 `sendCntAcc_`（`numMyExperts`） | 源卡每核 per-owned-expert 跨 chunk 的 count 累加器。 |

### 5.3 `Init`
- 新增 `numChunks_ = CeilDiv(ST, CK)`。
- `maskAlignSize_ / maskSlotSize_` 用**与改前一致**的公式（局部 `compareCount`），保证与 `PeermemInfo` 数值一致。

### 5.4 `BuffInit`（时段 A，跨卡同步前，从 UB 地址 0 起）
所有 `O(ST)` 缓冲降为 `O(CK)`：
- `topkChunk_` `4·CK`；`sendMaskTensor_[2]` 每个 `CK/8 + 32B`（多留 32B 给 §6.1 的 count 折叠）；
  `sendGatherOutTensor_` `4·CK`；新增 `sendCntAcc_`（`O(ws·epr/核)`，与 bs 无关）。
- `mxTempTensor_` / `xOut*` / `xIn*`（量化缓冲，`O(H)`）不变。
- `resetTensor_` 封顶到 `DISPATCH_RESET_BATCH`(8KB)，`ResetFlagList` 改为分批清零（§6.7）——
  原先 `O(flag)∝bs` 的整核份额降为恒 ≤8KB。

### 5.5 `FirstBuffInit`（时段 B，dispatch+GMM1，覆盖时段 A 同片 UB）
- **删** `gatherMask` 的 `ws·slot` 大块；**加** `sendCntTensor_`(`ws·32B`) + `maskChunk_`/`maskChunkU32_`(`CK/8`)。
- `validTopkIndexTensor_` / `topkIndexTensor_` 各降为 `4·CK`；`topkIndexTensor_` 不再一次性
  `CreateVecIndex`，改为**每 chunk** `CreateVecIndex(chunkStart, CK)` 生成本块全局 token 位置。
- 尾部 `[ubBufferUsedAddr_, 256KB)` 留给 `tripleTensor_`，每批 `≤ TT·32B`（triple-tile，§6.2）。

### 5.6 `SendMaskCal`（源卡）—— 改为 outer=chunk / inner=expert + count 折叠
```text
numMyExperts = ceil((ws·epr - aivCoreIdx_) / blockAivNum_)   // 本核负责的全局专家数(扁平分核)
若 numMyExperts==0 直接 return                                 // 核数>专家数时的尾部核
Duplicate(sendCntAcc_, 0); V_S
for c in [0, numChunks_):
    sliceBytes = (末块) clamp 到 maskAlignSize_ - chunkStart/8, 否则 CK/8     // §4.1
    Duplicate(topkChunk_, -1, CK); 载本块有效 topk(DataCopyPad)               // -1 哨兵: 防 CompareScalar(g=0) 误命中
    for e in owned experts:   // g, dstRank=g/epr, le=g%epr
        CompareScalar(maskBuf, topkChunk_, g, EQ, CK)                          // 本块 mask 切片
        GatherMask(..., validLen, sendCnt)                                    // 本块命中数(mask=validLen 跳哨兵)
        sendCntAcc_[e] += sendCnt
        若末块: 把累计 count 写进 maskBuf 尾部(偏移 sliceBytes)                // §6.1 count 折叠
        DataCopyPad(win[slotBase(e)+chunkStart/8], maskBuf, pushBytes)        // 末块 pushBytes=sliceBytes+4
```
- 仍用原 `MTE3_V` ping-pong 软流水（init 2 / tail 2），只是迭代次数变多。
- **尾部哨兵 -1（关键）**：最后一块不足 `CK` 的部分填 `-1`（任意 `≥ ws·epr` 的值），否则
  `CompareScalar(g==0)` 会把哨兵误判为命中专家 0 → count 虚高 + 越界 token 位置。

### 5.7 `SendInfoCal`（接收端）—— 只读 count
- 把原"一次性 `DataCopy` 全 `ws·slot`（含 mask）"换成**带 stride 的 count-only 读**：
  `DataCopyPad`，`blockCount=ws`、`blockLen=4B`、`srcStride=maskSlotSize_-4`，把 ws 个 count
  读进 `sendCntTensor_`。
- **落位不是紧凑的（已修 bug）**：`DataCopyPad` GM→UB 每个 burst 在 UB 内 **32B 对齐落位**
  （4B count + 28B pad），所以 rank r 的 count 在 int 下标 `r*8`，不是 `r`。最初按 `GetValue(r)`
  紧凑读，r≥1 读到 pad 垃圾 → cumsum 错 → GMM1 等不到行数 → **卡死**（bs=1/ws=2 实测复现）。
  现 `sendCntTensor_` 按 `ws*32B` 分配、按 `GetValue(r*8)` 取数。
- cumsum 循环从 `sendCntTensor_` 取数；`totalSendCntInExp_` / `expertRevNumsGloablTensor_` 写出
  / `CrossCoreSetFlag` 通知 AIC **全不变**。mask 不再需要常驻（接收端按需现读）。

### 5.8 `TripleInfoCalAndDispatch`（接收端）—— 单趟分块 + triple-tile
```text
priorExpertCumsum, 行→核划分(coreLo/coreHi/rowDstOffsetInCore) 与改前逐字相同        // §4.2, 总数来自 cumsum
matchedSoFar=0; rowsThisCore=0
for c in [0, numChunks_) while matchedSoFar < coreHi:
    sliceBytes = 末块 clamp(同源卡)                                                   // §4.1
    DataCopy(maskChunk_, win[(le·ws+dstRank)·slot + chunkStart/8], sliceBytes)
    CreateVecIndex(topkIndexTensor_, chunkStart, CK)                                  // 本块全局 token 位置
    GatherMask(validTopkIndexTensor_, topkIndexTensor_, maskChunkU32_, validLen, matchedInChunk)
    V_S   // matchedInChunk/validTopkIndex 标量就绪 + 作下一轮 DataCopy 的程序序屏障(保护复用)
    [gLo,gHi) = [matchedSoFar, matchedSoFar+matchedInChunk)
    [lo,hi)   = 与 [coreLo,coreHi) 求交
    if hi>lo:                                                                         // 本核在本块有要派发的行
        subStart=lo-matchedSoFar; subNum=hi-lo; subDst=rowStartIdxInDst+lo
        for off in [0, subNum) step TT:                                              // triple-tile: §6.2
            n = min(TT, subNum-off)
            tripleTensor_ = LocalTensor(ubBufferUsedAddr_, n·32B)                     // 每批 ≤ TT·32B, 与 CK 解耦
            CopyGMToGMPerToken(subDst+off, dstRank, subStart+off, n)                  // 函数体不改
        rowsThisCore += subNum
    matchedSoFar = gHi
// wave-grain AtomicAdd: 循环后一次, 对连续行区间[rowDstOffsetInCore-priorExpertCumsum, +rowsThisCore)
// 拆 wave 桶 —— 数学与改前逐字相同(本核派发的行恒为一段连续 dispatchRevData 区间)
```
- `CopyGMToGMPerToken` **函数体不改**：`copyStartIdx` 现在指向 chunk-local 的 `validTopkIndexTensor_`，
  6-buffer 软流水自包含；每个 sub-slice / triple-tile 批次单独调用、互不干扰。
- **leading-chunk 扫描**：高 `aivIdxInGroup` 的核需要扫过前导块以推进 `matchedSoFar`；这些块
  `hi≤lo` 不派发，只做"本地 DataCopy + 向量 GatherMask"，**不在跨卡 token 搬运（MTE）关键路径上**。

### 5.9 不改的函数
`CopyGMToGMPerToken`（体）、`Process`、`UpdateGroupParams`（`totalSendCntInExp_` 路径）、
`UpdateGlobalBuffer`、`QuantProcessInRank`、`Unpermute`、`SecondBuffInit` 的核心逻辑、GMM 相关。

---

## 6. 性能适配（避免分段引入劣化）

分段本身会带来与 `numChunks` 成正比的额外开销。以下三项把它压到最低，且**都不改 win 布局**。

### 6.1 count 折叠（count fold-in）
- **问题**：分块后 mask 按片写对端，原本"mask+count 一次写"被拆成"mask 片 + 单独 count 写"，
  对每个 owned 专家多 1 次小的跨卡 MTE 写。
- **做法**：因 count 字段恰好紧跟 mask 区（偏移 `maskAlignSize_`，见不变量 H），把**最后一块**的
  mask 切片与累计 count 放进同一 buffer（`sendMaskTensor_` 多留 32B），用一次 `DataCopyPad`
  （`pushBytes = sliceBytes + 4`）一起推。非末块只推 mask 片。
- **收益**：单 chunk 场景恢复成与改前一样的"1 写/专家"，**消除常规 bs 的回退**。

### 6.2 triple-tile 解耦
- **问题**：`tripleTensor_` 随 `subNum` 增长，而 `subNum` 可达 `CK`；`CK` 调大后单批会撑爆 UB。
- **做法**：把一个 sub-slice 的派发按 `TT=2048` 行分批，`tripleTensor_` 每批 `≤ TT·32B`(64KB)，
  **与 `CK` 解耦**。这是"能把 `CK` 调大"的前提。
- **行距说明**：`CopyGMToGMPerToken` 写 triple 用 `tripleTensor_[i * INT32_PER_256B]`，
  `INT32_PER_256B = 8`（256 **bit** = 32B 内的 int32 数，命名易误读为 256 Byte）。
  即每行 32B，分配 `n*INT32_PER_256B` 个 int32 = n·32B，TT=2048 ⇒ 上限 64KB。

### 6.3 动态 CHUNK（发/收两段各按自己 UB 余量独立算 `chunkLenA_`/`chunkLenB_`，取代固定常量）
- **动机**：固定 `CK` 会把本可单块装下的中小 bs 也强行切块（如 bs=2048/topK=8、ST=16384 在 CK=8192 下切成
  2 块），白付每块固定开销（同步屏障、`CreateVecIndex`/`GatherMask` 重发、跨卡 push 的 RDMA setup、
  高 `aivIdxInGroup` 核扫前导块）。这些开销 ∝ `numChunks`。
- **发/收 chunk 解耦（不取 min）**：发(段 A：`SendAndQuantBuffInit`+`SendMaskCal`)和收(段 B：
  `DispatchBuffInit`+`TripleInfoCalAndDispatch`)是**时间隔离**的(中间 `SyncAll`)、UB 各自从地址 0 铺、
  两段 chunk 缓冲从不共存；且 win mask 是**整条 ST 的 bit 数组**，发按 `chunkLenA_` 切片写、收按
  `chunkLenB_` 切片读，**写/读怎么切互不相关**（只需各自覆盖完整 ST、边界 256 对齐 ⇒ `/8` 即 32B 对齐；
  count 固定落 `maskAlignSize_`、与 chunk 无关；收侧顺序由升序扫+`CreateVecIndex(chunkStartB, chunkB)` 保证）。
  ⇒ **无任何耦合，两段各按自己余量取最大 chunk**，不再被另一段拖小。
- **做法**：`Init` 里两段各算上限，各自 cap 到 `CeilAlign(ST,256)`：
  - 随 `CK` 线性增长的字节系数：段 A `8.25·CK`（topkChunk 4CK + sendMask CK/4 + sendGatherOut 4CK）→ `chunkLenA_`、
    段 B `8.125·CK`（maskChunk CK/8 + validTopkIndex 4CK + topkIndex 4CK）→ `chunkLenB_`；
  - 固定占用（段 A：reset(封顶≤8KB,§6.7)+mxTemp+2·xOut+2·xIn+sendCntAcc；段 B：cumsum+sendCnt+copyTmp48KB+
    expertTokenNumsOut + 尾部预留 `TT·32B`）与 `SendAndQuantBuffInit`/`DispatchBuffInit` 分配**逐项字节对应**；
  - `ckMaxA=(预算-fixedA)·4/33` → `chunkLenA_`、`ckMaxB=(预算-fixedB)·8/65` → `chunkLenB_`，各自向下取整到
    256 倍数（⇒ `CK/8` 天然 32B 对齐），预算 `=256KB-4KB安全余量`。
- **效果**：
  - **中小 bs（`ST ≤ CK上限`）两段各自单块跑**（`numChunksA_=numChunksB_=1`）= 完全回到原始无分块、零 chunk 开销；
  - 大 bs 时**发侧不再被收侧的较小余量拖累**：因段 B 多 `copyTmp48KB+triple64KB`，`ckMaxB<ckMaxA`，旧
    取 min 把发侧也压到 `ckMaxB`；拆开后发侧用 `ckMaxA`（H=7168 下约 +50%），发侧迭代/跨卡 mask 推送次数减约 1/3。收侧不变（本就是 binding）；
  - `CK` 上限随 `H`/`ws`/`epr` 自适应（`H` 越大固定占用越多 ⇒ `CK` 自动变小），不再依赖手调常量。
- **正确性**：固定项是分配的字节精确镜像，仅 `CK`-线性项因 `CeilAlign` 上取整被低估 ≤568B(A)/≤62B(B)，
  由 4KB 安全余量吸收；两段峰值都 `< 256KB`。**改 `SendAndQuantBuffInit`/`DispatchBuffInit` 布局时必须同步 `Init` 的 fixedA/fixedB**。

### 6.4 主动放弃的优化（per-chunk count 向量跳前导扫描）
- 可让接收端读"每块命中数向量"前缀和后直接跳到自己的首个重叠块，省掉 leading 扫描。
- **不做**：它要把 per-chunk count 写进 win 槽 → 槽变大 → 重新与 host CCL buffer 分配耦合（§4.1 的风险），
  而收益在跨卡关键路径之外，且 §6.3 已把 leading 扫描减少 4×。**风险 > 收益，本地又无法验证。**

### 6.5 时段 C（combine/unpermute）大 bs 适配：weights 只载本核段（实测 bs=7168 修复）
- **症状**：bs=2048 正常、bs=7168 报 "MTE write UB out of bounds"。dispatch 段已与 bs 解耦，
  炸的是时段 C：`SecondBuffInit` 把**全量** `topKWeightsTensor_`（`bs·topK·4`）和 bf16 临时区
  （`bs·topK·2`）整块驻留 UB —— bs=7168、topK=8 时仅这两项 ≈344KB > 256KB。
- **改法**：`Unpermute` 本就 `TilingByCore(m_)` 按核分 token，每核只需自己
  `[wCoreOffset_, +coreLen)` 段的 weights。`SecondBuffInit` 改为只分配/加载 `coreLen·topK`，
  `Unpermute` 取数下标改为 `(tokenIdx - wCoreOffset_)·topK + expId`。UB 从 O(bs) → O(bs/核数)，
  数值与改前逐位一致（各核只读不写、且只读自己的段）。
- **残留容量约束**：时段 C 仍 O(bs·topK/核)。预算 `coreLen·topK·6B + 18H·B ≤ 256KB`，
  H=7168、48 AIV 核 ⇒ `bs·topK ≲ 105 万`（topK=8 → bs≲13万；topK=32 → bs≲3.3万）。
  触线时把 weights 改成 Unpermute 内按 token tile 现取（带宽开销 ≈0.03%），暂不实施。

### 6.6 uint32 偏移回绕修复 + win 容量拦截（大 bs·topK 适配）
- 字节/元素偏移先在 32 位算完才升 64 位的乘法，`bs·topK·H ≥ 2^32` 时回绕（topK=32、H=7168 ⇒ bs≈18700 触发），已全部 cast：
  `combine_send.h:42`（写远端 win）、`mega_moe.h` Unpermute 读/写（:1076/:1097）、QuantProcessInRank（:573）、`impl.h:369`（tripleGm）。
- host `CheckAttrParams` 的 `leastCclBufferSize` 公式重写：mask 区改 `epr·ws·(align32(ST/8)+32)` 与
  `PeermemInfo` 逐项一致；quant 区去掉错误的 `×topK`。放不下直接拦截。workspace 全 int64 公式，超 HBM 由 GE 分配失败拦截。

### 6.7 `resetTensor_` 分批清零（reset UB 与 bs 解耦，支持任意大 bs 泛化）
- **动机**：`ResetFlagList` 把 workspace 的 `groupFlagList` 清零，flag 总数
  `= epr·(INT_CACHELINE + dispatchFlagSlotsPerExpert_)`，而 `dispatchFlagSlotsPerExpert_ =
  CeilAlign(CeilDiv(maxOutputSize,256),16)`、`maxOutputSize = bs·ws·min(topK,epr)`，所以 flag 总数 **∝ bs**。
  改前 `resetTensor_` 按**每核全量份额** `CeilDiv(flagTotal, blockAivNum)` 一次性开 UB 并 `Duplicate` 清零，
  再用**单次** `DataCopyPad` 推 GM —— 这块占用随 bs 线性膨胀。在 `ws·epr=2048`（专家总数上限）下最坏代入
  `bs=8192、min(topK,epr)=32`，per-core ≈ `8·bs = 65536 int32 = 256KB`（32 核时）/ `~171KB`（48 核时），
  成为段 A 最大的固定项，且核数偏少时会把整片 UB 占满 → 段 A OOM。它本质是 §6.3 动态 CHUNK 无法覆盖的
  **固定项**（不随 CK 变），CK 收到下限也救不了。
- **做法**：把 `resetTensor_` 封顶到 `resetBatchLen_ = min(每核 flag 份额, DISPATCH_RESET_BATCH=2048 int32=8KB)`
  （`Init` 算，`mega_moe_base.h` 定常量）。`Duplicate` 仍只清这块封顶零 UB 一次；`ResetFlagList` 把本核
  份额 `coreLen` 按 `resetBatchLen_` **分批循环** `DataCopyPad`，每批从**同一片零 UB**(只读)推一段到
  `groupFlagList[coreOffset + off]`。零 UB 批间不被改写 ⇒ 循环前一次 `V→MTE3` 同步即覆盖所有批，无写后读冒险；
  每批 `off` 按 8KB（32B 对齐）步进，保持 `coreOffset` 原有对齐特性，`DataCopyPad` 自处理尾批不足长度。
- **效果**：
  - **本核份额 ≤ 8KB（中小 bs / 小 epr）单批跑** = 完全退化为改前的一次 `DataCopyPad`，零额外开销、零额外同步；
  - 仅大 bs（flag 份额超封顶）才多批，最坏 256KB/8KB = 32 批，纯 init 期一次性 GM 写，相对整条 dispatch 可忽略；
  - 段 A 固定项里 reset 从"∝ bs 的 ~171~256KB"降到"恒 ≤8KB"，**reset 占用与 bs/maxOutputSize 彻底解耦** ⇒
    `bs=8192·topK=32` 等极端配置在任意核数下都装得下，且为后续更大 bs 泛化留足裕量。
- **正确性**：写到 GM 的总字节、落点 `[coreOffset, coreOffset+coreLen)` 与改前逐字节一致（只是拆成多次推、
  源都是零），清零语义不变；`Init` 的 `fixedA` 同步用封顶后的 `resetBytes`，与 `BuffInit` 分配逐项对应。

---

## 7. 正确性论证与验证（本地，无 NPU）

| 验证 | 方法 | 结果 |
|---|---|---|
| **行→token 映射逐位一致** | C++ 仿真：分块单趟(+triple-tile) vs 原全 ST，覆盖 `CK∈{256,2048,8192}`、`TT∈{1,2048,∞}`、随机命中密度、`blockNumPerRank∈{1,2,3,8}`、maxOutput clamp / 跳过 regime | **26460 组全过** ✅ |
| **clamp / count 落点字节不变量** | 对 `ST=1..600000`、`CK∈{2048,8192}` 穷举不变量 A–H | **全过** ✅ |
| 括号 / 符号一致性、无悬挂引用 | grep + 括号配平 | ✅ |
| AscendC API 用法 | 全部照搬同文件既有可用代码的写法 | 人工核对 |

> 注：本地有 CANN（`ccec`）但缺少算子完整构建脚手架（DTYPE 宏 / tiling key / soc target），
> 直接 `ccec` 只会在 CANN 头文件里报 frontend/arch flag 配置错误，**与本次改动无关**；
> 真正编译/运行走你们 NPU host 的 op-build / UT 流程。

---

## 8. 风险点与待硬件验证项

**实现/复核重点（已在代码注释标注）**：
1. **尾部哨兵 -1**：源卡末块必须填 `-1`，否则误命中专家 0（§5.6）。
2. **count 一致性**：单趟从 cumsum 推行区间、实际行来自逐块 GatherMask，二者须一致；`rowsThisCore`
   以实际 `subNum` 累加并守卫 wave-AtomicAdd（§5.8）。
3. **多次小跨卡写可见性**：`SendMaskCal` 现发更多小的跨卡 MTE 写，须被 `Process` 跨卡 signal 前的
   `PipeBarrier<PIPE_ALL>+SyncAll` drain（均 MTE3，已覆盖）。
4. **buffer 复用屏障**：`maskChunk_`/`validTopkIndex_` 每 chunk 复用，由 GatherMask 后的 `V_S`
   作程序序屏障保护（含未派发的前导块）。
5. **GMM1 wave-flag 时序**：wave-AtomicAdd 仍是循环末单批、对连续本地行区间，桶算法逐字不变。

**待在 NPU 上验证**：
1. 小 bs（单 chunk，`ST ≤ CK`）：与改前 **bit-exact** 对比 `dispatchRevData` / `expertTokenNumsOut` / `y`。
2. 跨 chunk 且 `ST` 非整除：触发尾部哨兵 + 末块 clamp + count 折叠 + triple-tile 边界。
3. 大 bs（改前会 OOM）：确认不 OOM 且数值正确。
4. 多卡（`ws>1`）：跨卡 mask 切片可见性 + cumsum 正确。
5. 小 `blockLen` `DataCopyPad`：`SendInfoCal` 的 4B stride count 读 / `SendMaskCal` 末块 count 折叠写。

**遗留事项（架构级，未解决）**：
- **GMM2 回送 combine 区 win 放不下**：combine 区 = `bs·topK·H·2B` 全量驻留 HCCL win，
  bs=128K/topK=32/H=7168 ⇒ 60GB，远超窗口容量；quant 区(bs·~7.4KB)/mask 区同样线性涨。
  现状靠 host `leastCclBufferSize` 拦截直接报错（§6.6）。候选方案：host 按 sub-batch 分轮跑整条
  流水复用 win（kernel 不改，sub-batch≲4K 时其余约束全部自然满足）；或 combine 流式分批回送+win 复用
  （重排同步，代价高）。待评审定方案。

---

## 9. 附录：最终 UB 布局（256KB / AIV，时段间整块复用）

记 `CK=chunkLen_`（`Init` 动态算，§6.3）`, TT=2048, ST=bs·topK, H=k_, ws/epr=worldSize/expertPerRank`。
三个 `BuffInit` 都从地址 0 铺平，时段间有 `PipeBarrier/SyncAll` 隔离，跨时段同址重叠是**有意复用**。
A/B 两段的 `CK` 系数（8.25 / 8.125）正是 `Init` 反解 `CK` 上限的依据：`fixed + 系数·CK ≤ 256KB-4KB`。

**时段 A（`BuffInit`：SendMaskCal / ResetFlagList / QuantProcessInRank）**
```
topkChunk_(4·CK) | resetTensor_(≤8KB封顶,§6.7) | mxTemp(2KB) | xOut1/2(~2(align256(H)+H/32)) |
xIn1/2(~2·align128(H)·sizeof(X)) | sendMaskTensor_[2](2·(CK/8+32)) | sendGatherOut(4·CK) | sendCntAcc_
合计 ≈ 8.25·CK + fixedA(2KB+~6.5·H+reset≤8KB+sendCntAcc)；CK=ckMaxA 时恰好填满预算 —— 与 bs 无关
```

**时段 B（`FirstBuffInit`：dispatch + GMM1）**
```
expertTokenCnt(32) | cumsumInfo(align32(ws·epr·4),驻留) | sendCnt(ws·4) |
maskChunk_/U32_(CK/8,别名) | validTopkIndex(4·CK) | topkIndex(4·CK) | copyTmp[6](48KB) |
expertTokenNumsOut(epr·4) | [ubBufferUsedAddr_) tripleTensor_(动态,≤TT·32B,triple-tile)
合计 ≈ 8.125·CK + fixedB(48KB+TT·32+小项)；CK=ckMaxB 时恰好填满预算 —— 与 bs 无关
```

**时段 C（`SecondBuffInit`：Unpermute）**：`dataRes(3H·2)` / `dataResFp32(3H·4)` /
`topKWeights(coreLen·topK·4)` / `temp(coreLen·topK·2)` —— 主体 `O(H)`，weights 按核分段（§6.5），
容量约束 `bs·topK ≲ 105 万`。死代码 `resetTokenTensor_` 已删。

---

## 10. 变更文件汇总

| 文件 | 改动 |
|---|---|
| `mc2/mega_moe/op_kernel/arch35/mega_moe.h` | 成员增删；`Init`/`BuffInit`/`FirstBuffInit`/`SendMaskCal`/`SendInfoCal`/`TripleInfoCalAndDispatch` 分块重写；count 折叠；triple-tile；`ResetFlagList` 分批清零（§6.7）。 |
| `mc2/mega_moe/op_kernel/arch35/mega_moe_base.h` | 保留 `DISPATCH_TRIPLE_TILE`、新增 `DISPATCH_RESET_BATCH`（§6.7）；`PeermemInfo` 槽位公式**数值不变**（仅注释）。 |
| `mc2/mega_moe/op_kernel/arch35/mega_moe_combine_send.h` | combine 写远端 win 偏移 uint64 cast（§6.6）。 |
| `mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h` | tripleGm 偏移 uint64 cast（§6.6）。 |
| `mc2/mega_moe/op_host/op_tiling/mega_moe_tiling.cpp` | `leastCclBufferSize` 对齐 PeermemInfo 真实公式（§6.6）。 |

> win 布局 / `_workspace_info.h` 未改动；tiling 仅改容量拦截公式，不改 tilingData。
