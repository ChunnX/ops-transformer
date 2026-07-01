# Dispatch 去冗余 Pull 设计文档(单表 firstFlag 版)

> 分支:`RmDuplicatedToken`(基于 master,**不含** `bsChunkSplit` 的分块流式 commit)。
> 目标文件:`mc2/mega_moe/op_kernel/arch35/mega_moe.h`(+ `mega_moe_base.h` win 区、host tiling)。
> 状态:设计草图 + 方案评估,待 microtest 后实现。

---

## 1. 背景与问题

### 1.1 当前 dispatch 数据流

一卡 `expertPerRank_` 个本地专家。`Process()`(`mega_moe.h:964`)按 `localExpertId = 0..expertPerRank_-1` **顺序**循环,每个专家调用 `TripleInfoCalAndDispatch`(`mega_moe.h:650`),它按 `(源卡 r, 子组)` 分核 (`for idx = blockIdx_; idx < worldSize_*blockNumPerRank_; idx += blockNum_`,`mega_moe.h:657`)。

> 命名注意:`TripleInfoCalAndDispatch` 里的 `dstRankIdx` 实际是**源卡 r**——mask 从本卡 `gatherMaskInt32Tensor_`(`SendCntCal` 已载入,见 §2)读,token 从 `remoteRankIdx = dstRankIdx` 的 win 拉(`CopyGMToGMPerToken`,`mega_moe.h:580`/`605`)。

对每个命中 token,`CopyGMToGMPerToken`(`mega_moe.h:560`)做两件事:

1. **远端读(瓶颈)**:`DataCopy(copyTmpTensors_, remoteRankGlobalTensor[tokenIndex*copyInNum])` (`mega_moe.h:605`/`630`)——跨卡走 HCCS。
2. **本地写**:`DataCopyPad` token + scale 到本卡 `dispatchRevData`/`dispatchRevScale` (`mega_moe.h:617`/`619`)。

### 1.2 冗余在哪

`tokenIndex = topkIndex / topK_`(`mega_moe.h:591`)。同一源 token `T`(来自源卡 `r`)被本卡多个本地专家命中时,量化是 **per-token** 的,从 `r` 的 win 拉到的 `copyInNum` 字节**完全相同**,却被每个命中专家各拉一次。复制比 `ρ ≈ topK_ / worldSize_`:`worldSize_` 越小、`topK_` 越大,冗余越严重(极端:一卡 16 专家、topK=16、小 worldSize 时,同一份数据可被跨卡拉十几次)。

**关键认知:冗余只在第 1 笔(远端读);本地写不能省**(GMM1 要求每个专家在分组布局里都有自己那份行)。目标:

> **把"同一 `(r,T)` 的 N 次跨卡读"压成 1 次跨卡读 + (N-1) 次本地拷贝。** 本地 HBM 带宽 ≫ 跨卡 HCCS。

### 1.3 为什么不能"循环内顺手去重"

"按专家顺序循环"是**每个核各自**的循环;一个专家的活被切给多个核并行(`mega_moe.h:657`)。token `T` 在专家 `le_a`、`le_b` 命中里的序号不同 ⇒ **`le_a` 时拉它的核与 `le_b` 时要它的核通常不是同一个**;而专家循环体里**专家之间没有 SyncAll**(GMM1 靠 wave-grain flag `mega_moe.h:691-705` 启动)。跨核 + 跨专家无序,任何"等前一专家写好再读本地"的同步式方案都要全核屏障,破坏 GMM1 overlap。

⇒ **本设计用"flag + 失败回退"消除屏障**:消费者不阻塞等待,数据没到就自己回退远端拉(等价当前行为),到了就本地拷。永不读错,最坏退化成现状。

---

## 2. 本分支结构要点(对去重有利)

- **无分块流式**:整份 topk 一次性载入 `topkIdsTensor_`(`mega_moe.h:404`),mask 全量。
- **mask 常驻**:`SendCntCal`(`mega_moe.h:519`)把当前 `localExpertId` 的全部 `worldSize_` 个 `[mask|count]` 槽位载入 `gatherMaskTensor_` 常驻(`mega_moe.h:529`),`TripleInfoCalAndDispatch` 直接在 UB 上 `GatherMask`(`mega_moe.h:660`)。
- **现有"算好再一把发"的范式**:`SendMaskCal`(`mega_moe.h:395`)就是源端在独立阶段(在 `QuantProcessInRank` 那条 token 量化搬运流水之外)算 mask/count 并推送。我们的 firstFlag 沿用同一范式。

---

## 3. 总体设计:单表 `firstFlag`

只引入**一张表** `firstFlag[r][T]`(int32),它同时承担两个职责:

| 职责 | 谁写 | 何时 |
|------|------|------|
| **first**:token `(r,T)` 命中本卡的**最小本地专家号**(区分"是不是第一次 pull") | 源卡 | 源端算好,**一把发**到接收端 win(发送阶段) |
| **flag+row**:冗余数据**是否就绪**及其落地行号 | 接收端生产者 | 运行时,生产者拉完后**原地改高位** |

- **不再有** count 表、独立 dedupSlot 表、接收端重算 `FirstExpertCal`。
- token 搬运流水(`CopyGMToGMPerToken`)里**零 V 操作**——所有计算都在源端发送阶段做完。

### 3.1 编码:高位即 ready

int32 单字,高位区分两态(`first < epr ≤ 256` 必为正;`row < maxOutputSize_ < 2³¹`):

```
未就绪 (w >= 0): 低位 = first(最小本地专家号);  不命中本卡的 token 填高位为0的哨兵 0x7fffffff
就绪   (w <  0): w = row | 0x80000000,  低31位 = dispatchRevData 绝对行号
```

### 3.2 生命周期

1. **源端发送阶段**:算 `first_c[T]`,一把发到目标卡 c 的 win,初值全部高位 0(=未就绪)。
2. **CrossRankSync**(`mega_moe.h:959`)后:表全卡可见,且把上一轮残留的就绪位全部覆盖清掉(源端整列重写)⇒ **无需单独 reset**。
3. **接收端 dispatch**,处理 `(le, r, T)`(只处理命中 le 的 token,故 `w` 要么是合法 `first ≤ le`,要么已就绪):
   - `w < 0`(就绪,**likely**)→ **本地拷**:源行 `row = w & 0x7fffffff`。
   - `w >= 0`(未就绪):`first = w`
     - `le == first` → **生产者**:远端拉 → 本核 MTE3 drain 后,写 `firstFlag[r][T] = row | 0x80000000`。
     - `le > first` → 消费者但生产者未完成 → **回退远端拉,不写**。

**不变量:唯一写者**(只有 `le==first` 改这个字,消费者永不写)⇒ 无并发写、无 torn read(对齐 int32 单次写原子)。

---

## 4. 接收端读取:逐 token `ReadGmByPassDCache`(已定方案)

消费者在 Phase 1 逐 token 用 `ReadGmByPassDCache` 直接读 `firstFlag[r][T]`(绕过 dcache,拿 HBM 最新值):

- 与生产者侧 `WriteGmByPassDCache` 成对,**不需要 `DataCacheCleanAndInvalid`**(见 §11,内存模型已确认)。
- 读到的总是最新就绪位(含此前各专家生产者写下的),命中率最高。
- Phase 1 本就是逐 token 标量循环(读 `validTopkIndex`、算 `tokenIndex`),顺手加一笔 bypass 读即可。

> 备选优化(遗留,见 §14):若实测 Phase 1 的逐 token HBM 读延迟成为瓶颈,可改为"每 (le,r) 一次 MTE2 整列快照进 UB,再逐 token 读 UB"——但需先确认 MTE2 `DataCopy` 能读到 scalar-bypass 写的就绪位(本设计默认走已确认的 bypass scalar 读/写路径,不依赖该优化)。

---

## 5. 内存与数据结构改动

### 5.1 win 区:新增 `firstFlagRecvPtr`(`mega_moe_base.h:40` `PeermemInfo`)

在 `maskRecvPtr` 区之后、`quantTokenScalePtr` 之前插一段(布局 `[srcRank][token]`,512 对齐):

```cpp
// PeermemInfo 构造里, maskRecvPtr 区 offset 推进之后:
firstFlagRecvPtr = base + offset;
offset += Ops::Base::CeilAlign(
    (int64_t)tilingData->epWorldSize * tilingData->bs * (int64_t)sizeof(int32_t), (int64_t)ALIGN_512);
// 之后才是 quantTokenScalePtr = base + offset;
```

- 大小 `ws*bs*4`,相比 `quantTokenScale`/`dispatchRevData` 极小。
- 插入会顺移 `quantTokenScalePtr`/`combineSendPtr` 偏移——它们由同一 `offset` 串算,所有卡一致,无需改读取逻辑;但 **win 总量增大,host 侧 win/workspace 预留需同步**。

> `Init` 里新增 `firstFlagWinOffset_ = firstFlagRecvPtr - rankSyncInWorldPtr`(类比 `maskWinOffset_` `mega_moe.h:193`),源端跨卡写、接收端本地读都用它。

### 5.2 类成员(`mega_moe.h:71-142`)

```cpp
uint64_t firstFlagWinOffset_ = 0;     // 源端跨卡写用 (firstFlagRecvPtr - rankSyncInWorldPtr)
LocalTensor<int32_t> firstCalTensor_; // 源端 SendFirstCal: per-token first 暂存, 大小 m_ 个 int32
```

- 源端写 firstFlag 用 `GetRankWinAddrWithOffset(c, firstFlagWinOffset_+...)`(MTE3 一把发)。
- 接收端读/写直接用裸指针 `params_.peermemInfo.firstFlagRecvPtr + r*m_ + T` 配 `Read/WriteGmByPassDCache`, **无需 GlobalTensor、无需 UB 快照缓冲**。
- `firstCalTensor_` 在 `SendAndQuantBuffInit`(`mega_moe.h:283`)申请,`m_*4`(bs=4096 约 16KB),布局有余量。

### 5.3 tripleTensor_ 复用空槽传递每-token决策

`tripleTensor_` 每 token 占 `INT32_PER_256B=8` 个 int32,现仅用 0/1/2 (`RANK_ID/TOKEN_ID/TOPK_INDEX`,`mega_moe_workspace_info.h:66-68`)。新增:

```cpp
constexpr uint32_t SRC_KIND = 3U;   // 0=远端拉, 1=本地拷, 2=生产者(远端拉+拉完写flag)
constexpr uint32_t SRC_ROW  = 4U;   // SRC_KIND==1 时本地源行
```

有 5/6/7 空槽,`tripleTensorSize` 不变(仍 `rowsNumInCore*ALIGN_32`,`mega_moe.h:681`)。

### 5.4 无需单独 reset

源端每轮整列重写 firstFlag(含哨兵),CrossRankSync 保证覆盖在任何生产者写之前完成(§3.2)。

---

## 6. 源端:`SendFirstCal()`(计算 + 一把发)

新增函数,在 `SendMaskCal()` 同阶段(`Process` `mega_moe.h:952` 附近,`QuantProcessInRank` 之前)调用。**按目标卡 c 分核**,每核负责若干 c;在该独立阶段允许 V 操作(不碰 token 搬运流水)。

语义:`first_c[T] = min{ le : 全局专家 (c*epr+le) ∈ token T 的 topK }`,不命中填哨兵 `0x7fffffff`。

```cpp
// topkIdsTensor_: sendTotalNum_=m_*topK_ 个 id, token T 占 [T*topK_, (T+1)*topK_)
GlobalTensor<int32_t> dstFirst;
for (int c = aivCoreIdx_; c < worldSize_; c += blockAivNum_) {
    Duplicate(firstCalTensor_, 0x7fffffff, m_);                 // 哨兵, 高位0
    // 降序 le: 小 le 最后写, 直接覆盖为最小, 省去比较
    for (int le = expertPerRank_ - 1; le >= 0; --le) {
        int g = c * expertPerRank_ + le;
        CompareScalar(bitMask, topkIdsTensor_, g, EQ, compareCount_);   // 命中 g 的 topk 入口
        GatherMask(hitEntry, idxRamp, bitMask, ..., sendTotalNum_, cnt);// 命中 entry 列表
        for (j=0; j<cnt; ++j) {
            int T = hitEntry[j] / topK_;
            firstCalTensor_.SetValue(T, le);                           // 覆盖 → 最终为最小 le
        }
    }
    // 一把发到目标卡 c 的 win: firstFlag[srcRank=rankId_][0..m_)
    uint64_t dstOff = firstFlagWinOffset_ +
        (uint64_t)rankId_ * m_ * sizeof(int32_t);
    dstFirst.SetGlobalBuffer((__gm__ int32_t*)GetRankWinAddrWithOffset(c, dstOff));
    DataCopyPad(dstFirst, firstCalTensor_, {1, (uint32_t)(m_*sizeof(int32_t)), 0,0,0});
}
```

> 注:firstFlag 的 win 布局 `[srcRank][token]`,各源卡写自己那一行 `[rankId_][*]`,互不重叠。
> V 操作(CompareScalar/GatherMask)+ 标量散写 + 一把 DataCopy,均在本独立阶段,token 搬运流水零影响。

---

## 7. 接收端:`CopyGMToGMPerToken` 改造(`mega_moe.h:560`)

`TripleInfoCalAndDispatch`(`mega_moe.h:650`)只需透传 `localExpertId`;读 firstFlag 在 Phase 1 逐 token 用 `ReadGmByPassDCache`(§4)。`r = remoteRankIdx`(`=dstRankIdx`)。

```cpp
__gm__ int32_t* firstFlagBase =
    (__gm__ int32_t*)params_.peermemInfo.firstFlagRecvPtr + (uint64_t)remoteRankIdx * m_;

// === Phase 1: 三元组 + 每-token 源决策 ===
for (int i = 0; i < copyNum; ++i) {
    int topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + i);
    int T = topkIndex / topK_;
    tripleTensor_[i*8].SetValue(RANK_ID, remoteRankIdx);
    tripleTensor_[i*8].SetValue(TOKEN_ID, T);
    tripleTensor_[i*8].SetValue(TOPK_INDEX, topkIndex % topK_);

    int w = AscendC::ReadGmByPassDCache(firstFlagBase + T);   // 绕 dcache, HBM 最新
    int srcKind, srcRow = 0;
    if (w < 0) {                          // 就绪 → 本地拷 (likely)
        srcKind = 1; srcRow = w & 0x7fffffff;
    } else if (localExpertId == w) {      // 我是第一个命中专家 → 生产者
        srcKind = 2;
    } else {                              // 消费者, 生产者未就绪 → 回退远端 (unlikely)
        srcKind = 0;
    }
    tripleTensor_[i*8].SetValue(SRC_KIND, srcKind);
    tripleTensor_[i*8].SetValue(SRC_ROW,  srcRow);
}

// === Phase 2 prime/steady: 按源发 MTE2 ===
//  srcKind 0/2 (远端): DataCopy(copyTmp, remoteRankGlobalTensor[T*copyInNum], copyInNum)  // 现有
//  srcKind 1 (本地): 两段读 (dispatchRev* 中 token/scale 分开紧密排列)
//     DataCopyPad(copyTmp[0],                     dispatchRevDataGm_[srcRow*widthA],      widthA字节)
//     DataCopyPad(copyTmp[mxFp8TokenAlignBytes_], dispatchRevScaleGm_[srcRow*widthAScale], widthAScale字节)
//  之后 MTE3 写出对两种源完全一致(token读copyTmp[0], scale读copyTmp[mxFp8TokenAlignBytes_],
//     mega_moe.h:614-620), 故写出代码不变, 6-buffer 软流水骨架不变, 仅 MTE2 发射处分叉。

// === Phase 2 drain (现有 mega_moe.h:634-637): 等齐本核所有 MTE3 → token/scale 已落 HBM ===
SyncFuncStatic<AscendC::HardEvent::MTE3_S, ...>();          // MTE3→S: 保证数据 MTE3 先于 flag 标量写

// === Phase 2.5 新增: 生产者写 firstFlag (drain + MTE3_S 之后) ===
for (int i = 0; i < copyNum; ++i) {
    if (tripleTensor_[i*8].GetValue(SRC_KIND) == 2) {       // 生产者
        int T   = tripleTensor_[i*8].GetValue(TOKEN_ID);
        int row = rowDstOffsetInCore + i;                  // 本 token 绝对 dispatchRevData 行
        AscendC::WriteGmByPassDCache(firstFlagBase + T, row | 0x80000000);
    }
}

// === Phase 3 (现有): triple 搬出 ===
```

> **内存序(已确认,见 §11)**:同一片 HBM 跨核可见;`MTE3(数据) → MTE3_S 同步 → 标量 WriteGmByPassDCache(flag)`,别的核 `ReadGmByPassDCache` 读到就绪位即可确保 token 数据 ready。**全程 bypass 读/写,不需要 `DataCacheCleanAndInvalid`。** drain 已等齐本核 MTE3,再补一个 MTE3_S 即满足"数据先于 flag"的程序序。

---

## 8. 各函数改动点总览

| 位置 | 改动 |
|------|------|
| `mega_moe_base.h:40` `PeermemInfo` | 加 `firstFlagRecvPtr` 区(ws\*bs\*4),顺移后续偏移 |
| host tiling | win 总量同步增大 |
| `mega_moe.h:148` `Init` | 算 `firstFlagWinOffset_`,`SetGlobalBuffer` |
| `SendAndQuantBuffInit(:283)` | 申请 `firstCalTensor_`(m_ 个 int32) |
| **新增** `SendFirstCal()` | §6:按目标卡分核算 first + 一把发 |
| `Process(:952` 附近) | `SendMaskCal()` 同阶段插 `SendFirstCal()` |
| `TripleInfoCalAndDispatch(:650)` | 透传 `localExpertId` 给 `CopyGMToGMPerToken` |
| `CopyGMToGMPerToken(:560)` | 签名加 `localExpertId`;Phase1 逐 token bypass 读 firstFlag + 源决策;prime/steady 按源发 MTE2;drain+MTE3_S 后生产者 bypass 写 flag |

---

## 9. 正确性论证

1. **每行都填满**:无论生产者/消费者本地拷/回退,消费者都把数据写进**自己的** `dispatchRevData` 行(写出路径不变),GMM1 看到的分组布局与改前逐字一致。
2. **无 false-ready**:就绪位只在生产者数据 MTE3 落 HBM(drain + MTE3_S 后)才置;同片 HBM 跨核可见(§11 已确认),消费者就绪 → 读到的是已写好的行。
3. **回退安全**:未就绪 → 远端拉,等价当前行为,永不读错。
4. **单写者**:`le==first` 唯一;消费者不写。无并发写;int32 对齐写原子,无 torn read。
5. **读最新值**:消费者逐 token `ReadGmByPassDCache` 绕 dcache,看到所有 `le'<le` 生产者已落地的就绪位;仅并发 race 时漏看 → 回退,正确。
6. **跨轮干净**:源端每轮整列重写 first(高位 0),CrossRankSync 屏障在任何生产者写之前 → 无上轮残留。
7. **不劣化下界**:极端(生产者全程慢)退化为"人人远端拉"=当前行为,**绝不会更差**。

---

## 10. 数据结构尺寸小结

| 名称 | 位置 | 大小 | 读写 |
|------|------|------|------|
| `firstFlag[r][T]` | win | ws\*bs\*4 | 源端跨卡写(first,MTE3);接收端逐 token `ReadGmByPassDCache` + 生产者 `WriteGmByPassDCache` |
| `firstCalTensor_` | UB | bs\*4 | 源端 `SendFirstCal` 暂存 |
| `tripleTensor_` SRC_KIND/SRC_ROW | UB(复用空槽) | 0 额外 | Phase1 写、Phase2 读 |

---

## 11. 内存序(已确认)

message-passing:生产者"写 DATA(MTE3)→写 FLAG",消费者"读 FLAG→读 DATA"。

**结论(已确认,无需 microtest / 无需 `DataCacheCleanAndInvalid`)**:

- 不同核访问的是**同一片 HBM**,跨核可见性可保证。
- 生产者:token/scale `MTE3` → 插 **`MTE3_S`** 同步 → 标量 `WriteGmByPassDCache(flag)`。这样别的核读到就绪位时,token 数据 MTE3 必已完成、ready。
- 消费者:`ReadGmByPassDCache(flag)`,绕 dcache 拿 HBM 最新。
- 生产者写/消费者读**全程 bypass**,故不需要 `DataCacheCleanAndInvalid`。

正确性单向:false "未就绪" 只损失收益(多拉一次,无害);上面的 `MTE3_S`+bypass 保证不会 false "就绪"。

---

## 12. 验证计划

1. **功能**:固定 topk 构造高复制比(小 worldSize、大 topK),对比开/关去重的 `dispatchRevData` 与最终 `yOut` 逐元素一致。
2. **性能**:dispatch 段耗时 + 跨卡读字节数(应降 ≈ `ρ` 倍),确认 GMM1 overlap 未退化。
3. **边界**:`maxOutputSize_` 截断行(`mega_moe.h:668`)、count==0 专家、单卡、bs 非 256 对齐。

---

## 13. 开启条件 / 门控

**已定:始终开**(视为纯收益)。门控策略列入遗留事项,后续若发现 `ρ≈1`(大 worldSize)场景有可测开销再补 `ρ`(`≈topK_/worldSize_`)阈值门控。

---

## 14. 遗留事项(Leftover)

1. **门控**:当前始终开;后续视需要补按 `ρ` 的门控开关(理论上纯收益,优先级低)。
2. **不等 bs 支持**:当前**所有 rank bs 一致**(已确认的前提);`maskSlotSize_`/`firstFlag` 尺寸均依赖 `ws*m_`。若将来需支持不等 bs,要改尺寸推导。
3. **读取优化**:默认逐 token `ReadGmByPassDCache`;若实测 Phase 1 读延迟成瓶颈,评估"每 (le,r) 一次 MTE2 整列快照进 UB"(需先确认 MTE2 能读到 bytes-bypass 写的就绪位)。
4. **count 优化(可选)**:当前生产者对所有第一笔都写 flag;若想省掉非冗余 token(count==1)的 flag 写,可由源端附带 count 表跳过——纯微优化,非必需。

---

## 15. 方案评估与备选设计

本节补充设计取舍,重点回答两个问题:

1. 现有单表 `firstFlag` 为什么选择"源端提前算第一个专家"。
2. 是否可以改成更直接的"谁先读到 token,谁就把表置位"。

下面统一用更直白的名字描述:

- `源卡 r`: token 原始所在 rank。
- `token T`: 原始 token 行号,不是 `topkIndex`。
- `本地专家 le`: 当前接收 rank 上的 local expert id。
- `接收行 row`: token 被写入本 rank `dispatchRevData/dispatchRevScale` 后的绝对行号。
- `去重表[r][T]`: 存在接收 rank 的 peermem/window 或 workspace 中,用于回答"这个源 token 是否已经被本 rank 拉回来过,如果拉过,在哪一行"。

### 15.1 当前代码对方案的约束

当前 dispatch 不是单线程顺序执行,而是有三层并发/无序:

1. `Process()` 按 `localExpertId = 0..expertPerRank_-1` 组织专家循环,但这是每个核自己的控制流,不是全局串行。
2. `TripleInfoCalAndDispatch()` 内部按 `idx = blockIdx_; idx < worldSize_ * blockNumPerRank_; idx += blockNum_` 分工,同一个源 rank 的 token 会被不同 AIV 分片处理。
3. 专家之间没有 `SyncAll`。dispatch 通过 `flagDispatchToGmm1Ptr` 按 256-row wave 解锁 GMM1,故不能为了去重在专家之间加全核屏障。

因此,同一个 `(r,T)` 在多个本地专家之间可能出现这些时序:

```text
理想顺序:
    le=2 先拉回并写表 -> le=3/5/7 读表,本地拷贝

实际可能:
    处理 le=3 的核先到 -> 表未就绪 -> 远端拉
    处理 le=2 的核后到 -> 也远端拉并写表
    后续 le=5/7 才可能受益
```

去重方案必须满足:

- 读到"已就绪"时不能读到未落地的数据。
- 读到"未就绪"时不能阻塞等待,否则破坏 dispatch/GMM1 overlap。
- 多核同时处理同一 token 时,即使发生竞争,最坏也只能多拉一次远端数据,不能读错。

### 15.2 方案 A: 预计算第一个专家(当前文档主方案)

核心做法:

1. dispatch 之前由源端计算 `first[r][T]`: token `T` 发给某个接收 rank 时,最小的本地专家号是多少。
2. 接收端 dispatch 时只允许 `le == first[r][T]` 的专家当生产者。
3. 生产者远端拉回 token 后,把表从"未就绪:first"改成"已就绪:row"。
4. 其他专家如果读到已就绪,走本地拷贝;如果读到未就绪,直接回退远端拉,不等待。

优点:

- **单写者清晰**:每个 `(r,T)` 只有固定的 `first` 专家会写 ready row,不需要 CAS/锁。
- **正确性容易证明**:消费者不写表,生产者 `MTE3(data) -> MTE3_S -> WriteGmByPassDCache(flag)` 后置 ready,不会 false-ready。
- **flag 写次数少**:理论上每个 `(r,T)` 最多一个生产者写 ready。
- **跨轮 reset 简单**:源端每轮整列重写 first,天然清掉上轮 ready 位。

缺点:

- **dispatch 前多一段计算表的时间**:`SendFirstCal()` 需要按目标 rank 和本地专家扫描 `topkIds`,做 `CompareScalar/GatherMask/标量散写/跨卡发表`。这段在真正 dispatch 前发生,会抬高阶段启动延迟。
- **固定生产者不一定最快**:`first` 是最小本地专家号,不是运行时最先完成远端拉的核。如果这个专家对应分片慢,后续专家会回退远端拉,收益下降。
- **与 `SendMaskCal` 重复扫描 topk**:现有 `SendMaskCal` 已经对每个全局专家做 `CompareScalar` 生成 mask/count。若 `SendFirstCal` 独立实现,会再次做类似扫描。若想复用 `SendMaskCal` 的 mask,需要跨核合并 first,实现复杂度会上升。
- **新增 peermem 空间和 host 校验**:需要在 `maskRecvPtr` 后增加 `epWorldSize * bs * sizeof(int32_t)` 并同步修改 `leastCclBufferSize`。

适用判断:

- 若 `topK/worldSize` 较大、重复率高,且前置计算能被隐藏或占比不大,方案 A 稳定可靠。
- 若 profiling 显示 dispatch 前置阶段已经明显拉长,方案 A 可能把省下的 HCCS 时间部分抵消。

### 15.3 方案 B: 机会式 ready-row 表(推荐优先 microtest)

这是更直接的思路:不提前算"第一个专家是谁",只维护一张"是否已经有可复用行"的表。

编码建议:

```text
去重表[r][T] = 0                  表示还没有可复用行
去重表[r][T] = row | 0x80000000   表示已有可复用行 row
```

接收端处理每个命中 token 时:

```text
w = ReadGmByPassDCache(table[r][T])

if w < 0:
        row = w & 0x7fffffff
        从本地 dispatchRevData/dispatchRevScale[row] 拷贝
else:
        从远端 quantTokenScalePtr 拉 token
        写到自己的 row
        MTE3(data) -> MTE3_S
        WriteGmByPassDCache(table[r][T], row | 0x80000000)
```

这个方案允许多个核在竞争窗口内都看到空表,于是都远端拉并写 ready row。这里的并发写是**良性竞争**:

- 写入的 row 可能不同,但每个 row 都是同一个 `(r,T)` 的完整 token+scale 拷贝。
- 后续消费者从任意一个已就绪 row 本地拷贝,数据语义相同。
- 竞争失败只损失收益,不会影响正确性。

优点:

- **没有 dispatch 前的 first 计算**:只需要把表清零,不需要源端按专家扫描 topk。
- **运行时谁先完成谁当生产者**:不固定为最小专家号,更符合当前多核无序执行模型。先到的核先把可复用行发布出去,后续专家即可复用。
- **实现更短**:不需要 `SendFirstCal()` 的 `CompareScalar/GatherMask` 逻辑,也不需要在表里存 first 专家号。
- **不需要 CAS**:因为多写者写的是等价数据行号,而不是互斥所有权。last-writer-wins 可接受。

缺点:

- **ready 写次数可能更多**:所有在表未就绪时回退远端拉的核,都会在完成后写一次 ready row。高竞争窗口里可能出现多次写同一 `(r,T)`。
- **收益不确定性更强**:若多个专家几乎同时到达同一 token,它们都会先远端拉,去重收益只体现在更晚到达的专家。
- **需要单独 reset 表**:每轮 dispatch 前必须把表清零。建议由接收端本地 AIV 按 `TilingByCore(epWorldSize * bs)` 清零,并在 dispatch 前 `SyncAll<true>()` 保证 reset 完成。
- **每个命中 token 多一次表读,每个远端拉 token 多一次表写**:这些是 dispatch 关键路径上的标量 GM 操作,需要 profiling 验证是否小于省下的跨卡 MTE2 成本。

安全性前提:

- ready 写必须在本核 MTE3 写 token/scale 完成之后,即 `MTE3(data) -> MTE3_S -> WriteGmByPassDCache(table)`。
- 消费者读表必须使用 `ReadGmByPassDCache`,避免 dcache 旧值。
- 表清零必须早于任何 dispatch 读表。
- `dispatchRevData/dispatchRevScale` 行在本 kernel 内不会被覆盖;GMM1 读取该行不改变其内容,所以其他专家把它作为本地拷贝源是安全的。

适用判断:

- 如果希望先做低复杂度试验,方案 B 更适合作为第一版 microtest。
- 如果实测多数重复 token 都能被较早发布 ready row,方案 B 可能比方案 A 更划算,因为它省掉了 `SendFirstCal()`。
- 如果实测表读写标量开销或多生产者远端拉过多,再回到方案 A 或增加门控。

### 15.4 方案 C: 抢占式 owner 表(CAS/原子占坑)

这是"读到空表就置位"的强一致版本。表里增加一个中间态:

```text
0                 empty
IN_PROGRESS       某个核已经抢到生产者身份,但数据还没 ready
row | READY_BIT    数据 ready
```

理想流程:

```text
读表 empty -> CAS(empty, IN_PROGRESS) 成功 -> 我远端拉,完成后写 ready row
读表 IN_PROGRESS -> 不等待,回退远端拉;或短等后再读
读表 ready -> 本地拷贝
```

优点:

- 每个 `(r,T)` 理论上只有一个生产者,比方案 B 少重复 ready 写和重复远端拉。
- 不需要源端预计算 first。

缺点:

- 依赖 GM 上可用且成本可接受的 CAS/原子交换语义;当前代码只使用 `AtomicAdd`、bypass read/write,没有现成的 GM CAS 模式。
- `IN_PROGRESS` 不能阻塞等待,否则会破坏 overlap;若选择回退远端拉,仍会有重复远端读,但实现比方案 B 复杂。
- CAS 成功后如果生产者所在核进度慢,后续消费者读到 `IN_PROGRESS` 要么回退、要么等待。等待不推荐,回退则收益接近方案 B 但复杂度更高。

结论:除非确认 GM CAS 可用、低成本,且 profiling 表明方案 B 的多生产者写入明显拖慢,否则不建议第一版采用。

### 15.5 方案 D: 严格按专家顺序本地复用(不推荐)

直观做法是让 `le+1` 等 `le` 把重复 token 拉回来后再本地拷贝。但当前代码的核心性能点正是 dispatch 与 GMM1 overlap、wave-grain 解锁。如果在每个专家之间插入 `SyncAll` 或 rank 内屏障,会带来:

- GMM1 不能边收边算,必须等整个 expert dispatch 完。
- 长尾 expert/source rank 会拖住全部核。
- 去重节省的 HCCS 读,很可能被屏障等待抵消。

结论:不建议。

### 15.6 对比表

| 方案 | 前置计算 | 并发安全 | 远端读节省上限 | 关键风险 | 实现复杂度 | 建议 |
|------|----------|----------|----------------|----------|------------|------|
| A. 预计算 first | 高:需额外扫描 topk 并发表 | 强:单写者 | 高,但依赖 fixed first 是否及时 ready | 前置延迟、重复扫描 topk | 中 | 稳健备选 |
| B. 机会式 ready-row | 低:只需清零表 | 强:多写者良性竞争 | 中到高,依赖运行时时序 | 多生产者导致收益波动;标量读写在关键路径 | 低到中 | **优先 microtest** |
| C. CAS owner | 低 | 强,但依赖 CAS | 高 | GM CAS 可用性/成本;IN_PROGRESS 处理 | 高 | 暂不推荐 |
| D. 专家间屏障 | 低 | 强 | 高 | 破坏 dispatch/GMM1 overlap | 中 | 不推荐 |

### 15.7 推荐路径

推荐先实现和验证 **方案 B: 机会式 ready-row 表**。

理由:

1. 它正面利用当前代码的真实并发:谁先把 token 拉回来,谁就发布可复用行,不强行指定最小专家为生产者。
2. 它避免 `SendFirstCal()` 的前置 `CompareScalar/GatherMask` 开销,更容易判断"去重本身"是否值得。
3. 它的并发写是良性竞争,不需要 CAS;最坏退化为当前远端拉行为,外加少量表读写。
4. 实现范围比方案 A 小:新增表区、reset、`CopyGMToGMPerToken` 的读表/本地拷/写表即可。

建议第一版 microtest 做两个开关:

- `DEDUP_MODE=0`: 当前基线,不读表。
- `DEDUP_MODE=1`: 方案 B,机会式 ready-row。

若方案 B 的收益不稳定或标量表读写开销偏大,再评估方案 A。方案 A 适合作为"确定生产者、减少 ready 写次数"的保守方案,但不应在未量化前置计算成本前直接定稿。

### 15.8 方案 B 需要补充的验证项

在 §12 的基础上增加:

1. **竞争构造**:构造同一 `(r,T)` 同时命中多个本地专家,观察多核同时 fallback 时最终 `dispatchRevData/yOut` 是否一致。
2. **收益统计**:增加计数器或临时统计,记录 `tableReadyHit`、`remoteFallback`、`readyWrite`。预期 `tableReadyHit` 越高,跨卡读节省越明显。
3. **时序安全**:强制在生产者写 ready 前后插入/去掉 `MTE3_S` 做对照 microtest,确认缺少同步时能暴露风险,保留同步后稳定。
4. **reset 安全**:连续多轮运行,确保上一轮 ready row 不会泄漏到下一轮。
5. **低重复场景**:`topK <= worldSize` 或专家分布均匀时,确认方案 B 的表读写开销不会明显拖慢。
