# BSA MXFP8 砍 `aic_mte2_time` 设计

日期：2026-09-03  
基线：`bsa_0827` / FusedExpSub qs128；几何仍是 `2026-08-25` 方案 A（kvs128、G=16、Dpv=20、L1 P 20×16KB）  
标尺：S=118800 keep=1.0 vs QFA MXFP8 dense（同 B=1 N=1 D=128）  
状态：**P0b 已在分支 `bsa_0827_k_prefetch_p0b` 落地（未上板）**；P0a/P0c/P1 未做

## 0. 目标与非目标

**目标：** 把 S=118800 的 `aic_mte2_time` 从 **15.12 ms** 往下砍。这是当前 Duration 下限（`max(cube 10.15, mte2 15.12, vec 10.87)`）。判据只用 `mte2_time` 和 `aic_mte2_active_bw`，不用 `mte2_ratio`。

**非目标（本文件不做）：**

- Q L0 双槽、K L0 双槽（叠的是 MTE1，不动 GM 字节/突发）
- 再抠 softmax VF / FusedExpSub
- 双 AIV float-S（只降墙钟分母，`mte2_time` 几乎不动）
- 真改 `kvBaseTile=256` 的 softmax/P 环/gmax（§6，独立几何重做，默认不做）

**成功线（S=118800 keep=1，warm 5 launch 10）：**

| 阶段 | `mte2_time` | `active_bw` | Duration 预期 |
| --- | --- | --- | --- |
| 现在 | 15.12 ms | 64 GB/s | 16.4 ms |
| P0 做完 | ≤ 13.5 ms | ≥ 70 GB/s | ~15.0–15.5（仍可能贴 MTE2） |
| P1 做完 | ≤ 12.0 ms | ≥ 78 GB/s | ~13.5–14.5 |
| 不要期望 | 10.65 ms / 91 GB/s | QFA 形态 | MXFP4 kvs256 仍是 ~63 GB/s |

精度：qs128 冒烟 + `./run.sh --reg`；sparse keep&lt;1 不得回退。

---

## 1. 现状（为什么 15.12 ms）

核均 AIC MTE2 字节：`64.1×15.12 ms ≈ 91.1×10.65 ms ≈ 0.97 GB`。与理论主流量一致：

```
K+V ≈ 2 × nQ × S × D / 28核 , nQ≈S/128
```

QFA 更短是 **同样字节、更高有效带宽**，不是少搬。

当前 QK 一拍：

```
Wait(V1_C1, PIPE_FIX)
→ Wait(MTE1_MTE2, l1K)
→ SparseKFused（标量 GetValue + Nd2Nz 数据 + Dn2Nz scale）
→ SetFlag(MTE2_MTE1) + 立刻 Wait(MTE2_MTE1)
→ LoadKToL0A + mmad + fixpipe
→ SetFlag(MTE1_MTE2)   // 在 mmad/fixpipe 之后才放 L1
→ SetFlag(V1_C1)
```

L1 KV 已是 **4×32KB**，QK/PV 共用 `kvBufId`。P 环 20×16KB，L1 只剩 ~6KB，**不能**把 P 槽改成 32KB 还不砍环深。

`blockShapeY` 默认 **128**。keep=1 时一个 kvs128 tile = 一个 y-block，`SparseKFused` 的 peek 已经拼成 **1 条数据 + 1 条 scale**。Dense 快路径 **不会**把 16KB 变成 32KB；它去掉的是每拍阻塞的 `GetValue` 和 while。突发加大是 P1。

D=128 时 scale 一条大约 **0.5KB**（`CopyGmToL1MxScaleDn2NzA5`），和 16KB 数据各发一次 MTE2。小 copy 几乎不增加字节，但加倍事务、拉低 `active_bw`。数据是 zN Nd2Nz，scale 是 Dn2Nz，**不能合成一条 DataCopy**。

---

## 2. 分期与推荐

| 阶段 | 改什么 | 动几何？ | 预期杠杆 |
| --- | --- | --- | --- |
| **P0a** | K/V dense 连续 copy（keep=1 / identity idx） | 否 | 去掉 GetValue；MTE2 连续双发 |
| **P0b** | 拆开 QK：预取 K，Wait 只在 LoadK 前；LoadK 后立刻放 L1 | 否 | 把 stall 从 `mte2_time` 里抠掉 |
| **P0c** | tile 级 scale：整 tile 一条；与数据背靠背发，中间无 scalar | 否 | 事务形状不变但不再被 GetValue 切开 |
| **P1** | **成对 gather**：偶 tile 一次搬 256 行填满 32KB 槽，奇 tile 复用 | 否（softmax/P/G/Dpv 仍 128） | 事务减半，突发 16→32KB |
| **P2** | `kvBaseTile=256` 真基块 | **是** | 默认不做，见 §6 |

推荐实现顺序：**P0b → P0a+P0c → 上板看 bw → 再 P1**。P0b 与 Q 双槽解绑（P0.5 回退原因）。P1 不碰 VF。P2 单独开分支。

---

## 3. P0a / P0c：dense 快路径 + tile 级 scale

### 3.1 判定

Host tiling 增字段 `kvGatherMode`（uint32）：

- `0 = SPARSE_FUSED`（默认，现网）
- `1 = DENSE_CONTIGUOUS`：keep=1 且 sparse idx 为 `0..yBlockNum-1` 递增（S=118800 标尺）

Kernel 仍做一次便宜校验（防 Host 标错）：

```
isDense = (kvGatherMode==1)
       && (gatheredKvSeqlen == kvSeqlen)
       && (yBlockNumRsvd == yBlockNumAval);
```

不在 kernel 里扫整段 idx。keep&lt;1、gap、置换 idx 全部走现网 `SparseKFused` / `SparseVFused`。

### 3.2 K（QK）

文件：`block_mmad_qk_arch35_mxfp8.hpp`

`SparseKFusedBaseTileL1FullLoad` 入口：

```
if (isDense) {
    DenseKBaseTileL1FullLoad(...);  // 新函数
    return;
}
// 现网 while + peek 不动
```

`DenseKBaseTileL1FullLoad`（D=128、kvs=128 时一条 16KB）：

```
oriStart = gatheredKvSTileIdx * kvSBaseTile;   // identity：gathered 行号 == GM 行号
nRows    = kvsActBaseTile;

gK tile  = GetTile(gK, (oriStart, 0), (nRows, embed));
l1K tile = GetTile(l1K, (0, 0), (nRows, embed));
copyGmToL1K(l1K, gK);                          // 一次 Nd2Nz

srcB16Off = oriStart * kvHeadMul * scaleRowHalf;
copyGmToL1MxScaleDn2Nz(l1Scale, gKScale, nRows, scaleK, srcB16Off, 0, kvHeadMul);
```

两条 MTE2 之间 **禁止** `GetValue` / 标量循环 / `printf`。不要在两条 copy 之间 `Wait`。

Q 仍只在 `isFirstKvsTile` 搬；dense 不改 Q。

### 3.3 V（PV）

文件：`block_mmad_pv_arch35_mxfp8.hpp`

对 `SparseVFusedBaseTileL1FullLoad` 同样分支。V 是 ColumnMajor `(embed, S)`，dense 一次 Nd2Nz 盖 `(embed, nRows)`，scale 一次沿 S 的 Nd2Nz（现网 `CopyGmToL1AScale`），长度为 `CeilDiv(nRows, MX_SCALE_GROUP_NUM)`。

### 3.4 不能做的「合并」

- 数据 + scale 合成一条 copy：layout 不同，硬件做不到。
- 改 Host scale 布局来迁就一条 copy：精度/GQA/BSND 雷区，本阶段不做。
- keep&lt;1 走 dense：地址不连续，会搬错。

P0c 在 dense 路径里已经是「整 tile 一条 scale」。sparse 路径保持「每段数据 + 每段 scale」；peek 合并连续段后 scale 已随段变大，不再另写一套。

---

## 4. P0b：K 预取（主排布）

### 4.1 要改的顺序

文件：`block_sparse_attention_kernel_arch35_mxfp8_full_quant.h` + `block_mmad_qk_arch35_mxfp8.hpp`

把 `BlockMmadQK::operator()` 拆成两个 AIC 可调用入口（PV 先不动）：

| 方法 | 做什么 | 不做什么 |
| --- | --- | --- |
| `PrefetchK(...)` | `Wait(MTE1_MTE2, kBufId)`；dense/sparse gather 数据+scale；`SetFlag(MTE2_MTE1)`；**不** `Wait(MTE2_MTE1)`；**不** `kBufId++` | 不碰 L0 / mmad / V1_C1 |
| `ComputeQK(...)` | `Wait(MTE2_MTE1, kBufId)`；`LoadQ/LoadK`；mmad；延后 fixpipe；`SetFlag(MTE1_MTE2)`；`kBufId++` | 不再发 GM→L1 |

Kernel 主循环 AIC（`coreTaskId < coreTaskNum` 段）：

```
// 与现在一样：先 PV(delay20)，再 delay3 gmax/pscale

PrefetchK(curTile);          // 用当前 kBufId 写 L1；与上一拍 AIV / 本拍 PV 重叠
Wait(V1_C1, PIPE_FIX);       // 只等 S 槽；K 已在飞或已在 L1
ComputeQK(curTile);          // 只等 MTE2 完成再 LoadK
SetFlag(V1_C1);
```

首 tile：循环外或 `loop==0` 时先 `PrefetchK`，此时 L1 KV event 已在 QK ctor 里 `SetFlag(MTE1_MTE2)` 预置，Wait 立即过。

**禁止**再把 `Wait(V1_C1)` 塞进 `PrefetchK` 和 `ComputeQK` 之间的 block 内部（那就是回退的 P0.5 三件套）。Wait 留在 kernel，gather 与 Wait 的相对位置由 kernel 决定。

### 4.2 LoadK 后立刻放 L1

`ComputeQK` 里：`LoadKToL0A` 一返回（`SetFlag(MTE1_M)` 之前或之后均可，但必须在 mmad/fixpipe **之前**）就 `SetFlag(MTE1_MTE2, l1KEventId)`。L1 槽只被 MTE1 占用到 LoadK 结束，mmad 用 L0。

这样下一拍 `PrefetchK` 的 `Wait(MTE1_MTE2)` 不会被本拍 Cube/fixpipe 挡住。

现在 `SetFlag(MTE1_MTE2)` 在 fixpipe 之后，L1 被白占 ~cube+fixpipe。

### 4.3 与 PV 共用 4 槽

同一 `kvBufId`：一拍 PV 消耗 1 槽（V），QK 消耗 1 槽（K）。4 槽 = 两拍 QK+PV 深度。P0b 预取是 **本拍 K 提前到 Wait(V1_C1) 前**，仍用即将 Compute 的那个 `kBufId`，**不**额外占用「下一 tile 的槽」。

不要在本拍结尾再预取 L+1（那要第 5 个逻辑槽，4 槽 + PV 同拍会打架）。若以后要跨 tile 预取，必须先画槽位表再加槽或让 PV 停一拍，本阶段不做。

### 4.4 明确不是 P0.5 回退件

| | 2026-08-20 P0.5 | 本 P0b |
| --- | --- | --- |
| Q 双槽 | 绑在一起 | **不绑** |
| wait 后移 | 塞进 QK operator | kernel 调用 `PrefetchK` 再 `Wait` 再 `ComputeQK` |
| 立刻 Wait MTE2 | 仍在 gather 后 | gather 只 Set；Wait 在 LoadK 前 |

---

## 5. P1：成对 gather（填满 32KB，不改 softmax）

### 5.1 想法

L1 KV 单槽已经是 32KB = **256×128×1B**。softmax / Fixpipe / L1 P / G / Dpv / VF **全部仍按 128**。只让 MTE2 每 **两个** kvs128 tile 发一次 256 行。

```
tile 2i  (even):  Wait(MTE1_MTE2, slot)
                  copy K[s : s+256] → L1[slot]   // 32KB + 一条 256 行 scale
                  SetFlag(MTE2_MTE1)
                  Wait(MTE2_MTE1); LoadK 行 [0,128); mmad 128
                  不放 L1、不 kBufId++

tile 2i+1 (odd):  不再 MTE2
                  Wait(MTE2_MTE1) 已在 even 满足；LoadK 行 [128,256); mmad 128
                  SetFlag(MTE1_MTE2); kBufId++
```

尾块 `remaining < 256`：even 只搬 `remaining`（可能仍是 128），odd 不存在。

PV 对 V 做同样的偶发奇用，否则 K 减半事务、V 仍每 128 打一次，总 MTE2 只能少 ~25%。

### 5.2 约束

- `LoadKToL0A` 已有 `nSubRowStart`。even 用 `0`，odd 用 `128`。现在 `nLoopNum=1` 且 `nSubRowStart=0`，P1 要传半槽偏移。
- L0A K 仍 16KB（128×128）。**不要**一次 mmad 128×256（L0C 128×256×4B=128KB，现 2×64KB 当两个 128×128 用，不要并成一块 256）。
- 偶 tile 的 L1 必须活过中间那次 softmax（奇 tile 的 `Wait(V1_C1)`）。这与「LoadK 后立刻放 L1」冲突：**P1 的 even 在第一次 LoadK 后不能放槽**，odd 第二次 LoadK 后再放。P0b 的早放只适用于「一槽一行 tile」。P1 覆盖 P0b 的早放语义：even 延迟到 odd。
- PrefetchK(even) 搬 256；PrefetchK(odd) 空操作。
- D=64：256×64=16KB，槽只用一半，逻辑仍按 256 行，避免再分 D 特例。
- qs64：K 行数仍是 kvs，与 qs 无关，P1 同样适用。

### 5.3 为何这不是 P2

`2026-08-25` §2 方案 B 否决的是 **softmax 基块 256 + P 槽 32KB**。P1 不改 `KVS_BASE_SIZE`、不改 `kvBaseTile_`（Host 仍 128）、不改 VF。只改 gather 步长。

### 5.4 预期

事务约减半（K+V 数据+scale 都是每 256 行一次）。突发 32KB，对标 QFA 单条 K。`active_bw` 可能到 75–85，**不要**按 91 写进验收。MXFP4 同是 BSA gather，kvs256 仍 ~63；P1 能超过 63 是因为 dense 连续 + 预取，不是因为「256」这个数字本身。

---

## 6. P2：真 kvs256（默认不做）

若要把 `kvBaseTile_` / `KVS_BASE_SIZE` 改成 256：

| 项 | 现在 | P2 需要 |
| --- | --- | --- |
| L1 P 槽 | 16KB = 128×128 | 32KB = 256×128 |
| P 环 20×32KB | 640KB | **超过 512KB** |
| 要保住 320KB P 预算 | 20×16 | 必须 **10×32** → Dpv/G 从 20/16 砍到 ~10/8 |
| 组窗口 | 16×128=2048 | 8×256=2048（窗口不变，组尾频率变） |
| L0A PV Vᵀ | 18KB 装 144×128 | 144×256=36KB → 必须切 S2 或加槽，现 L0A 已满 |
| UB S / VF | 128×128 | 新 VF 或两次 qs128 VF |
| Host | `kvBaseTile_=128` 写死 | tiling 改 256 |

L1 空闲只剩 ~6KB，没有「顺手把 P 改 32KB」的空间。P2 等于再做一次 0825 方案 B，**独立分支、独立精度矩阵**。本文件验收不依赖 P2。

---

## 7. 代码落点

| 文件 | 阶段 | 改动 |
| --- | --- | --- |
| `op_host/block_sparse_attention_tiling.h/.cpp` | P0a | `kvGatherMode`；keep=1 + iota idx → `DENSE_CONTIGUOUS` |
| `block_mmad_qk_arch35_mxfp8.hpp` | P0a/c, P0b, P1 | dense load；拆 `PrefetchK`/`ComputeQK`；早放 L1；P1 半槽 |
| `block_mmad_pv_arch35_mxfp8.hpp` | P0a/c, P1 | dense V；P1 成对 V gather |
| `block_sparse_attention_kernel_arch35_mxfp8_full_quant.h` | P0b | 调用序：PrefetchK → Wait V1_C1 → ComputeQK |
| `block_mmad_arch35_utils.hpp` | 无（P0/P1） | 不要改 `L1_P_*` / `L0A_QK_BUF_CNT` |
| VF / epilogue softmax | 无 | P0/P1 不碰 |

`TaskInfo` 需要带上 `kvGatherMode`（或从 tiling 拷到 kernel 成员）。

---

## 8. 流水（P0b + dense，kvs 仍 128）

```
     AIV:  ... softmax(L-1) ... | softmax(L) ...............
     AIC:  PrefetchK(L) | Wait V1_C1 | ComputeQK(L) | PV(L-20) ...
     MTE2: [K_L data+scale]              [V_{L-20} ...]
```

K 的 MTE2 与 `softmax(L-1)` / 本拍 PV 重叠，不再坐在 `Wait(V1_C1)` 后面。

P1 even：MTE2 窗口是 32KB，覆盖随后两次 ComputeQK。

---

## 9. 验证

每阶段单独上板，不要一次合四件。

**精度（每阶段）：**

- qs128 dense / qs64 / sparse kvs256 / GQA / 跨组 s2&gt;2048 / D=64
- `./run.sh --reg` 全过；失败例仍 ≥99.5%

**性能（S=118800 keep=1）：**

- 主看 `aic_mte2_time`、`aic_mte2_active_bw`
- 辅看 Duration、`aic_scalar_ratio`（P0b 应下降）、`aic_cube_time`（不应变差 &gt;2%）
- ratio 下降但 time/bw 不动 → 只是墙钟短了，阶段失败

**对照：** 同 commit 的 QFA 表；BSA MXFP4 只作 BW 形态参照，不是 Duration 目标。

**回退开关：**

- `kvGatherMode` 可强制 0
- kernel 宏 `BSA_MXFP8_K_PREFETCH=0` 回到「Wait 后 gather + 立刻 Wait MTE2」
- `BSA_MXFP8_K_PAIR256=0` 关闭 P1

---

## 10. 风险

| 风险 | 处理 |
| --- | --- |
| Prefetch 与 PV 抢同一 `kvBufId` | 本拍只预取即将 Compute 的槽；槽位表用 `KV_SHARE_DEBUG` 打前 40 tile |
| 早放 L1 时 MTE1 未写完 | `SetFlag(MTE1_MTE2)` 必须在 `LoadK` 的 MTE1 完成之后（已有 `SetFlag(MTE1_M)` 可作序） |
| dense 误判置换 idx | 只信 Host `kvGatherMode` + seqlen 校验，不猜 |
| P1 even 跨 softmax 占槽 | 4 槽在 Dpv=20 下 PV 仍每拍拿走 1 槽；上板若死锁，先关 P1 |
| P0.5 曾 -2% 混在 Q 双槽里 | 本设计强制不改 L0 Q/K 槽数 |

---

## 11. 决策（请审）

1. P0/P1 做，P2 不做。  
2. 突发加大 = **成对 256 gather**，不是改 `kvBaseTile`。  
3. 数据/scale 永远两条 copy；「减少条数」= 每 tile（P1 则每 256 行）各一条，中间无 scalar。  
4. 先合 P0b 单独上板，再 dense，再 P1。
