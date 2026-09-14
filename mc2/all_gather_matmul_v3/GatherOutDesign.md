# AllGatherMatmul V3 GatherOut 输出方案设计

## 1. 背景与目标

`allGatherMatmul`（V3，MX 量化 URMA 路径）目前把 AllGather 回来的数据放在 win 区（HCCL 通信 buffer），只供 AIC 侧的 matmul 消费，**没有把 gather 结果直接输出到 `gather_out`**。

本方案目标：在不影响现有 matmul 结果正确性的前提下，把 AllGather 回来的数据从 win 区搬到 `gather_out`（GM 输出 tensor），并利用 **多 buffer + 通信/拷贝流水** 尽量掩盖拷贝开销。

- 输出 `gather_out`：shape `(M * rankSize, K)`，dtype 与 `x1` 一致（FP8 或 FP4 packed），rank-major 排布，仅含 data，**不含 scale**（与 inferShape 及 `torchapi` 文档一致）。
- 由于 win 区数据段与 `gather_out` 在字节布局上完全一致（见 §2.1），拷贝本质上是一次「win → UB → gather_out」的等量字节搬运，无需任何重排/对齐填充。

---

## 2. 现状分析

### 2.1 通信数据布局（win 区）

URMA AllGather 使用 PUT 模式（`all_gather_udma_put.h`）：

- 源：`localAddr_`（本卡 `x1`）+ `tileByteOffset_`。
- 目的：`commBufferAddrs[targetRankId] + winOffset_ + rankId * chunkBytes_ + tileByteOffset_`。

其中：

- `dataBytesPerMRow_`（每 M 行字节数，无 K 方向 padding）：
  - FP8：`k * sizeof(AType) = k`
  - FP4：`k >> 1`（2 个 fp4 打包 1 字节）
- `chunkBytes_ = m * dataBytesPerMRow_`（每个 rank 一段的数据字节数）
- data 段 `winOffset_ = 0`，scale 段 `winOffset_ = dataRegionBytes_`。
- `dataRegionBytes_ = rankSize * m * dataBytesPerMRow_`。

因此本卡 win 区（`W = commBufferAddrs[rankId]`）data 段布局为 **rank-major、无 padding、连续**：

```
W + 0                    : rank0 的 m 行
W + m * rowBytes         : rank1 的 m 行
...
W + r * m * rowBytes     : rank r 的 m 行
...
W + dataRegionBytes_     : scale 段（本方案不拷贝）
```

> 注：barrier 标志位于独立的 2MB barrier buffer（`ubmemCtx->commBufferAddrs`），与 data/scale 的 `udmaCtx->commBufferAddrs`（HCCL 内置 buffer）不重叠，拷贝 data 段不会触碰 barrier 区。

### 2.2 当前 AIV 通信流水

`AllGatherMxMatmulUrmaImpl::AllGatherProcess()`（`all_gather_mx_matmul_urma_impl.h:269`）：

```cpp
CrossCoreSetFlag<0x2, PIPE_MTE3>(0);            // round0 自身数据就绪
for (round = 0; round < commTurn_; ++round) {
    if (GetBlockIdx() < rankSize_) {
        allGatherScale_.Commit();
        allGatherData_.Commit();
        allGatherData_.Wait<BARRIER_DEVICE>();  // Drain + TeamBarrier.CrossDevice
    }
    SyncAll<true>();                            // 全 AIV 核屏障
    CrossCoreSetFlag<0x2, PIPE_MTE3>(round + 1);// 通知 AIC 该轮数据就绪
}
```

要点：

- `commTurn_ = tileCnt_ + tailCnt_` 即 `loopCnt`。
- 每轮 `Commit` 搬 1 个 M 切片（head 为 `tileM` 行，tail 为 `tailM` 行），`Wait<BARRIER_DEVICE>` 保证本轮所有 rank 数据落位 win 区。
- `SyncAll<true>()` 同步**所有 AIV 核**（含不参与通信的核）。

### 2.3 `gather_out` 与 `y` 的 rank-major 关系

matmul 输出 `y` 与 win 区同为 rank-major（`cFragAddrs_[r] = cGM + r * mPerRank * cBytesPerM`）。`gather_out` 的 inferShape 也是 rank-major `(M*rankSize, K)`。因此：

- `gather_out` 的 rank r 段 = `G + r * m * rowBytes`，与 win 区 rank r 段**字节地址一一对应**。

---

## 3. 总体方案

### 3.1 核划分

设 AIV 核数 `blockNum = GetBlockNum()`，通信核数 `rankSize`，拷贝核数：

```
copyCoreNum = blockNum - rankSize          // 约定：恒有余量（copyCoreNum >= 1）
```

> 已确认：1:1（`KERNEL_TYPE_MIX_AIC_1_1`）模式下，`GetBlockNum()` 返回的就是 AIV 核数（`aivCoreNum`），`copyCoreNum` 计算正确。

- **通信核**：`blockIdx ∈ [blockNum - rankSize, blockNum)`，其 `jobIndex = blockIdx - (blockNum - rankSize)`，即「后 rankSize 个核」。
- **拷贝核**：`blockIdx ∈ [0, copyCoreNum)`，即「前 copyCoreNum 个核」。

说明：

- 现有的 `allGatherData_/allGatherScale_` 与 `teamBarrier_` 的 `Init` 均以 `GetBlockIdx()` 作为 `jobIndex`，需改为 `jobIndex = blockIdx - (blockNum - rankSize)`（仅在通信核上有效）。
- `teamBarrier_` 的 `Init` 对拷贝核仍可调用（`jobIndex >= totalJobs` 时 `CrossDevice` 直接 return），拷贝核不会触发 `CrossDevice`。
- 所有 AIV 核（通信核 + 拷贝核）都必须执行相同次数的 `SyncAll<true>()`，保证屏障配平。

### 3.2 流水线

V 核上采用任务给定伪代码（通信核执行 `allGather(i)`，拷贝核执行 `CopyDataFromWin(i-1)`）：

```cpp
// blockNum >= rankSize + 1 假设下
for (uint32_t i = 0; i < commTurn_; ++i) {
    if (IsCommCore()) {
        allGather(i);          // Commit + Wait<BARRIER_DEVICE>，round i 数据落位 win 区
    }
    if (IsCopyCore() && i > 0) {
        CopyDataFromWin(i - 1);// 把 round i-1 的 gather 数据 win -> UB -> gather_out
    }
    SyncAll<true>();
    if (IsCommCore()) {
        CrossCoreSetFlag<0x2, PIPE_MTE3>(i + 1);  // 保持与 AIC matmul 的原同步不变
    }
}
if (IsCopyCore()) {
    CopyDataFromWin(commTurn_ - 1);  // 收尾：最后一轮
}
SyncAll<true>();
allGatherScale_.Finalize();
allGatherData_.Finalize();
// dcci ... 保持原有收尾
```

流水正确性（数据依赖）：

- 第 i 轮（i>=1）拷贝的是 round `i-1`，其数据已在第 `i-1` 轮 `allGather(i-1)` 的 `Wait<BARRIER_DEVICE>` + `SyncAll<true>()` 中完成同步，因此拷贝核读取安全。
- 第 i 轮通信核写 round i 的 win 区切片，与拷贝核读 round i-1 的切片**位于不同 M 切片地址，无读写冲突**。
- 每轮末尾 `SyncAll<true>()` 保证：round i 数据就绪后，下一轮（i+1）拷贝核才开始读 round i。

### 3.3 同步语义小结

| 同步点 | 参与方 | 作用 |
| --- | --- | --- |
| `Wait<BARRIER_DEVICE>`（TeamBarrier.CrossDevice） | 通信核（跨 rank） | 保证本 round 所有 rank 数据已写入 win 区 |
| `SyncAll<true>()` | 全 AIV 核 | 通信核与拷贝核之间的轮次配平；拷贝核与通信核的先后顺序 |
| `CrossCoreSetFlag<0x2, PIPE_MTE3>` | 通信核 → AIC | 保持 matmul 的原有数据就绪通知，拷贝逻辑不影响它 |

---

## 4. 关键实现细节

### 4.1 `CopyDataFromWin(round)` 实现（多 buffer ping-pong）

参照 `MoveResultFromPeerMemToOut`（v2 `all_gather_matmul_aiv_mode.h:705`），但 v3 无 K 方向 padding，可简化为纯字节搬运。推荐在 v3 内使用 apace tensor copy（`asc::te::make_copy(copy_gm_to_ub/copy_ub_to_gm)`，与 `barrier_ubmem.h` 同源），或沿用 v2 的 `CopyGmToUbuf/CopyUbufToGmAlignB16` + `SetFlag/WaitFlag` 事件。

参数（每 round）：

```
round i:
  head (i < tileCnt_):  tileRows = tileM_
  tail (i == tileCnt_): tileRows = tailM_   (tailCnt_==1 时)
  roundTotalRows = rankSize_ * tileRows
  tileRowBytes   = tileRows * dataBytesPerMRow_
  // rank r 段的源/目的：
  src = W + r * m_ * dataBytesPerMRow_ + i * tileM_ * dataBytesPerMRow_
  dst = G + r * m_ * dataBytesPerMRow_ + i * tileM_ * dataBytesPerMRow_
  len = tileRowBytes
```

拷贝核 `c`（`c ∈ [0, copyCoreNum)`）按 **M 行均分** 本 round 的 `roundTotalRows` 行：

```
rowsPerCore = roundTotalRows / copyCoreNum
rem         = roundTotalRows % copyCoreNum
rowStart(c) = c * rowsPerCore + min(c, rem)
rowEnd(c)   = rowStart(c) + rowsPerCore + (c < rem ? 1 : 0)
```

行号 `row`（0-based，在本 round 内）到 (rank, rowInTile) 的映射：

```
rank      = row / tileRows
rowInTile = row % tileRows
```

由于 src/dst 布局相同（rank 段 stride = `m_ * rowBytes`，段内连续），拷贝核遍历自己负责的行区间所覆盖的 rank 段，逐段做连续字节搬运即可，天然对齐、无需跨 rank 拆行计算。

多 buffer（double buffer）伪代码：

```cpp
// 双 UB buffer：ubBuf[0]、ubBuf[1]，每个大小 = 2 * chunkBytes（或按 UB 容量切片）
// chunkBytes = 单次 MTE 搬运字节数（按 M 行 × rowBytes 对齐，如 tileRows/分块 × rowBytes）
for (seg in 本核负责的 rank 段子区间) {
    for (chunk in 段内分块) {
        evt = chunk & 1 ? EVT0 : EVT1;
        WaitFlag<MTE3_MTE2>(evt);
        CopyGmToUbuf(ub[chunk&1], src + off, 1, chunkBytes, 0, 0);   // win -> UB
        SetFlag<MTE2_MTE3>(evt);
        WaitFlag<MTE2_MTE3>(evt);
        CopyUbufToGmAlignB16(dst + off, ub[chunk&1], 1, chunkBytes, 0, 0); // UB -> gather_out
        SetFlag<MTE3_MTE2>(evt);
    }
}
WaitFlag<MTE3_MTE2>(EVT0);
WaitFlag<MTE3_MTE2>(EVT1);
```

- UB 缓冲分配：拷贝核独立于通信核的 `commBuf/commScaleBuf/barrierBuf`，在 AIV 的 UB 布局中预留一段用于拷贝（例如 `copyUbOffset`，大小按 2 个 buffer × 单次 chunk 计算）。若 UB 紧张，可减小 chunk 粒度（更多次 ping-pong）。

### 4.2 FP8 / FP4 处理

- 搬运全部以 **字节** 为单位（`uint8_t`/`int8_t` UB buffer），无需关心元素 dtype：
  - FP8：`dataBytesPerMRow_ = k`，`gather_out` 每行 k 字节。
  - FP4：`dataBytesPerMRow_ = k >> 1`，`gather_out` 以 uint8 打包存储、K 维逻辑长度为 k（实际 `k/2` 字节/行），字节量与 win 区一致。
- 无 K 方向 padding（v3 通信按 `dataBytesPerMRow_` 精确字节搬运），因此不需要 v2 里 `k_align vs k` 的差额修正逻辑。

### 4.3 尾块（tail）处理

- `commTurn_ = tileCnt_ + tailCnt_`。当 `tailCnt_ == 1`，最后一轮 `round = tileCnt_` 对应 `tileRows = tailM_`。
- `CopyDataFromWin` 内通过 round 索引区分 head/tail，取对应 `tileRows`，即可复用同一套均分逻辑。
- `tailM_ == 0`（M 恰好被 `tileM` 整除）时 `tailCnt_ == 0`，无尾块轮。

### 4.4 拷贝核读取 win 区的 cache 一致性

- 已确认：win 区地址连续，不同核读取的是不同的 cache line，**不存在 cache 冲突**，无需额外的 cache 失效/冲刷。
- 同步边界沿用 `Wait<BARRIER_DEVICE>`（Drain）+ `TeamBarrier.CrossDevice` 保证远端写入可见；收尾的 `dcci` 仍保留在通信核 `Finalize` 后执行。

---

## 5. 改动点清单

### 5.1 kernel 侧

1. `op_kernel/arch35/all_gather_matmul_v3_apt.cpp`
   - 去掉 `(void)gather_out;`，将 `gather_out` 传入 `impl.Init(...)`（新增参数 `GM_ADDR gatherOutGM`）。

2. `all_gather_mx_matmul_urma_impl.h`（`AllGatherMxMatmulUrmaImpl`）
   - 成员新增：`GM_ADDR gatherOutGM_{}`；`uint32_t blockNum_{}`、`copyCoreNum_{}`；UB 拷贝 buffer 偏移与大小。
   - `Init`：
     - 保存 `gatherOutGM_`；计算 `blockNum_ = GetBlockNum()`、`copyCoreNum_ = blockNum_ - rankSize_`。
     - 通信对象 `allGatherData_/allGatherScale_/teamBarrier_` 的 `jobIndex` 改为 `GetBlockIdx() - (blockNum_ - rankSize_)`（仅通信核有意义）。
     - 预留拷贝 UB 空间。
   - `AllGatherProcess`：
     - 按 §3.1 划分通信核/拷贝核；按 §3.2 流水执行。
     - 增加 `CopyDataFromWin(round)` 方法（§4.1），内部按 §4.2/§4.3 处理 FP4/FP8 与 tail。

### 5.2 host 侧

3. `op_host/op_tiling/all_gather_matmul_v3_tiling.cpp`
   - `CheckTensorPtrNullptr` 中移除/落实 `gather_out` 占位注释，改为校验 `gather_out` 非空、shape 为 `[m*rankSize, k]`、dtype 与 `x1` 一致（inferShape 已计算 shape，此处主要补 tiling 校验与日志）。
   - 无需新增 tiling 字段（`rankSize/m/k/tileM/tileCnt/tailM/tailCnt` 均已由 `commTile` + kernel 端计算得到）。

4. `op_host/all_gather_matmul_v3_def.cpp`
   - `gather_out` 已是 REQUIRED（当前声明无需改），确认其 `DataType` 列表与 `x1` 对齐（已对齐）。

5. `op_host/all_gather_matmul_v3_infershape.cpp`
   - 现状已输出 `gather_out = [m*worldSize, k]`，无需改动；如后续支持动态 shape 需补充 `-1` 分支（现状已有）。

---

## 6. 边界与约束

- **假设恒有余量**：`usedCoreNum > rankSize`（即 `copyCoreNum >= 1`）。若未来出现 `usedCoreNum == rankSize`，需增加兜底（通信核串行拷贝）或在 tiling 拦截，本期不处理。
- `rankSize ∈ {2,4,8,16}`，AIV 核数足够时均满足假设。
- 拷贝核数量可能小于 `rankSize`，M 行均分逻辑需正确处理 `rem`（余数行优先分给前 `rem` 个拷贝核）。
- `gather_out` 为 data-only 输出，不包含 `x1_scale`；若后续需求要求 scale 一并输出，需在 scale 段（`W + dataRegionBytes_` 起）增加对应拷贝。

---

## 7. 风险与待确认

1. ~~`GetBlockNum()` 语义~~（已确认：1:1 下即 AIV 核数）。
2. ~~cache 一致性~~（已确认：地址连续、不同核读不同 cache line，无冲突，无需冲刷）。
3. **UB 空间**：拷贝核新增双 buffer 是否与现有 UB 布局冲突，chunk 粒度需按实际 UB 余量调优。
4. **性能**：copyCoreNum 与 chunk 大小的取值对流水重叠率的影响，建议实现后做 perf 对比。
