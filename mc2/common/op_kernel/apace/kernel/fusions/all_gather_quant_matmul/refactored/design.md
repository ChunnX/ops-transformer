# AllGatherQuantMatmul (URMA) — 重构设计文档

## 1. 背景

本目录对 `kernel/fusions/all_gather_quant_matmul/` 下的 URMA 实现进行结构重构。
旧代码不动，消费者不切换。

## 2. 文件结构

```
apace/basic/
  buffer_channel.h                        BufferChannel (共享模块)

apace/core/aiv_comm/
  collective_comm_context.h               CommContext (共享模块)

refactored/
  all_gather_qbmm_mx_kernel.h             QuantMatmul kernel (Run(commCtx, td, addrs, tp))
  all_gather_mx_matmul_urma_impl.h        主体逻辑 (AllGatherProcess + 类型别名 + UB 分配)
  all_gather_mx_matmul_urma_kernel.h      __global__ kernel 入口
  design.md                               本文档
```

## 3. 参数分类

| 入参 | 分类 | 传递方式 |
|---|---|---|
| `CommContext*` | 独立（类似 tilingData） | `Run()` 独立参数 |
| `TensorAddrs` | Cat 1 动态 (6字段: GM 地址) | `Run()` 参数 |
| `TilingDataType*` | Cat 2 静态 | `Run()` 参数 |
| `TilingDerivedParams` | Cat 2 派生 (3字段) | `Run()` 参数 |
| `<AType, BType, CType>` | Cat 3 模板 | 模板参数 |

### TilingDerivedParams (2 字段)

- `commTurn` — impl AllGatherProcess 循环 + winBuffer Init
- `effBufferCount` — win buffer 复用深度，`commTurn <= effBufferCount` 时自然退化为非复用

scale 非切分轴大小 (`scaleNonSplitAxisSize`) 由 host 预算放入 commTilingData，impl/kernel 直接读取。

### kernel 私有派生成员 (Init() 从 td_ 计算)

- `rankId_`, `rankSize_` (从 tp_), win buffer 地址从 dataChannel_->GetBase() / scaleChannel_->GetBase() 获取
- `mPerRank_`, `paddedTailM_`, `headRows_` (从 commTile)
- `dataBytesPerMRow_`, `scaleBytesPerMRow_`, `cBytesPerM_` (依赖 sizeof(AType)/sizeof(CType))

tiling 原始字段 (tileCnt/tileM/tailCnt/tailM/k/n) kernel 直接从 `td_->commTile`/`td_->mmTile` 读取。

## 4. Win buffer 复用

### 布局计算 (impl Init 内联)

dataChannel_/scaleChannel_ 直接作为 impl 成员，effBufferCount 在 tp_ 中，scaleWinOffset 为 Init 局部变量。

```cpp
void Init(dataSlotSize, scaleSlotSize, commTurn, rawSlotNum) {
    effBufferCount = (rawSlotNum < commTurn) ? rawSlotNum : commTurn;
    scaleWinOffset = effBufferCount * dataSlotSize;
    dataChannel_.Init(dataSlotSize, effBufferCount);
    scaleChannel_.Init(scaleSlotSize, commTurn);
}
```

`dataSlotSize`/`scaleSlotSize` 由 host 预算放入 commTilingData。

### 复用退化

当 `effBufferCount >= commTurn` 时：
- AIV 等待 `round >= effBufferCount` 恒不成立 → 不等待 ✓
- AIC 释放 `releasedRoundIdx_ < commTurn - effBufferCount` 恒不成立 → 不释放 ✓

无需 `bufferCount > 0` 开关判断，架构归一。

### Host 端 slotNum 计算

```cpp
uint64_t winBufferBytes = hcclBufferSize - RESERVED;
uint64_t scaleRegionBytes = commTurn * scaleSlotSize;
uint64_t dataRegionBytes = winBufferBytes - scaleRegionBytes;
uint64_t slotNum = dataRegionBytes / dataSlotSize;
tilingData->commTile.slotNum = slotNum;
```

当 win buffer 足够大时 `slotNum >= commTurn` → 退化非复用。

## 5. impl 成员变量

```
commCtx_ (impl only), tilingData_, addrs_, tp_, dataChannel_, scaleChannel_,
allGatherData_, allGatherScale_, teamBarrier_,
  dataChannel_ + scaleChannel_ (impl 直接成员),
quantMatmulKernelImpl_ (public)
```

## 6. Run() 接口

```cpp
void Run(__gm__ CommContext *commCtx, const TilingDataType *td,
         const TensorAddrs &addrs, const TilingDerivedParams &tp);
```

## 7. 与旧代码的差异

| 维度 | 旧代码 | 重构后 |
|---|---|---|
| 文件数 | 1 (impl.h) | 4 (分层) |
| 参数分类 | 31字段混合 Params | TensorAddrs(6) + TilingDerivedParams(2) + CommCtx 独立 |
| 缓冲管理 | 裸 GM 地址 | BufferChannel 直接成员 + effBufferCount/scaleWinOffset |
| CommContext | 两个 fusion 重复定义 | 共享 collective_comm_context.h |
| 成员变量 | 25 个零散 | 8 个对象成员 |
| win 复用 | 无 | 默认开启，slotNum >= commTurn 自然退化 |
