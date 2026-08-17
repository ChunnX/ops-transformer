# mc2_native 流程带读 — 从 main 到 kernel

> 本文档带你从 `main()` 开始，逐步追踪 `--sqe_mode 3` (mc2_native) 的完整执行流程，
> 标注每一步的正确性检查点，方便排查问题。

## 0. 全局视角

```
main()                                    ← Host 主线程
  ├─ 解析命令行参数
  ├─ aclInit() + HcclGetRootInfo()
  └─ 每设备一个 std::thread → Sample()    ← Host 子线程 (rank0 / rank1)
       ├─ aclrtSetDevice + HcclCommInitRootInfoConfig
       ├─ setup_ctx()                      ← 6 步建立通信通道
       │    ├─ [1/6] aclrtMalloc workspace
       │    ├─ [2/6] HcclCommMemReg
       │    ├─ [3/6] build_channel_descs (枚举 UBC 链路)
       │    ├─ [4/6] HcclChannelAcquire (跨 rank 握手)
       │    ├─ [5/6] resolve channel ptrs (handle → device VA)
       │    └─ [6/6] alloc table+state GM
       ├─ barrier (等所有 rank 就绪)
       ├─ [仅 -m 3] build_hccl_context()   ← 构造 HcclCombinOpParam
       │    ├─ 从 device 读 ChannelEntity (D2H)
       │    ├─ 解码 SQ/CQ/buffer/token
       │    ├─ 填入 HcclAiRMAWQ / HcclAiRMACQ
       │    └─ 拷 HcclCombinOpParam 到 device
       └─ rank0: mc2_native_bench_launch → kernel 执行
          rank1: sleep (URMA 单向, 接收方不做事)
```

**关键设计**: 单进程多线程，rank0 发送，rank1 等待。URMA Write 是单向操作，接收方不需要做任何事。

---

## 1. main() — 入口与参数解析

**文件**: `host/test_urma_p2p_bench.cpp:598`

```cpp
int main(int argc, char *argv[])
```

### 1.1 参数解析

`main` 解析命令行参数到全局 `g_args` (BenchArgs 结构体):

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `--sqe_mode` (`-m`) | 1 | 3=mc2_native (本文档关注) |
| `--bench_mode` (`-o`) | 0 | 0=batch_write, 1=write_drain_loop |
| `--loop_count` (`-l`) | 100 | kernel 内循环次数 |
| `--sqe_bytes` (`-s`) | 65536 | 每个 SQE 传输的数据量 |
| `--buf_bytes` (`-b`) | 100MB | workspace 大小 |
| `--dev_ids` (`-i`) | 0,1 | 物理设备 ID 列表 |

### 1.2 ACL + HCCL 初始化

```cpp
aclInit(nullptr);           // 初始化 ACL 运行时
aclrtSetDevice(0);          // 主线程绑定 device 0 (HcclGetRootInfo 需要绑定 context)
HcclGetRootInfo(rootInfo);  // 生成通信域 rootInfo (含 magic number 等)
aclrtResetDevice(0);        // 释放主线程 device, 子线程各自 SetDevice
```

### 1.3 启动子线程

```cpp
for (uint32_t i = 0; i < g_args.devCount; i++) {
    args[i].rootInfo = rootInfo;
    args[i].device = i;           // rank = i (逻辑 rank, 非物理 device)
    args[i].devCount = g_args.devCount;
    threads[i] = std::thread(Sample, (void *)&args[i]);
}
```

每个设备一个线程，`rank = i` (逻辑 rank 编号)，物理设备通过 `args->devIds[rank]` 映射。

**正确性检查**:
- `g_args.devCount` 必须 ≥ 2 (P2P 至少 2 卡)
- `devIds` 列表的设备必须存在且支持 UBC 链路

---

## 2. Sample() — 每设备线程的 worker

**文件**: `host/test_urma_p2p_bench.cpp:441`

### 2.1 设备绑定与 HCCL 通信域初始化

```cpp
uint32_t physicalDev = (rank < args->devIds.size()) ? args->devIds[rank] : rank;
aclrtSetDevice(physicalDev);
aclrtCreateContext(&context, physicalDev);

HcclCommConfig config;
HcclCommConfigInit(&config);
config.hcclDeterministic = 1;
config.hcclBufferSize = 200;

HcclComm hcclComm;
HcclCommInitRootInfoConfig(devCount, rootInfo, rank, &config, &hcclComm);
```

- `physicalDev`: 物理设备 ID (通过 `--dev_ids` 映射)
- `HcclCommInitRootInfoConfig`: 所有 rank 用同一个 `rootInfo` 初始化通信域
  - **阻塞操作**: 两个 rank 必须同时到达才能完成

**正确性检查**:
- 两个 rank 的 `devCount` 必须一致
- 两个 rank 用同一个 `rootInfo` (主线程生成)
- 如果报 `HCCL_E_TIMEOUT`，检查网络/端口 (`HCCL_IF_BASE_PORT`)

### 2.2 peer 选择

```cpp
peerRank = (rank == 0) ? 1 : 0;  // 默认: rank0 ↔ rank1
if (devCount > 2) peerRank = (rank + 1) % devCount;  // 多卡: 环形
```

- rank0 的 peer = rank1, rank1 的 peer = rank0
- 可通过 `--peer` 覆盖

### 2.3 调用 setup_ctx

```cpp
BenchCtx *ctx = setup_ctx(hcclComm, rank, devCount, args->bufBytes);
```

详见下一节。

### 2.4 构造 tiling

```cpp
UrmaP2pBenchTiling tilingHost{};
tilingHost.peer         = peerRank;
tilingHost.numSqe       = args->numSqe;
tilingHost.sqeBytes     = args->sqeBytes;
tilingHost.bufBytes     = args->bufBytes;
tilingHost.dbInterval   = args->dbInterval;
tilingHost.channelSlot  = slot;
tilingHost.cqOwnerSlot  = slot;
tilingHost.stateStride  = stateStride;
tilingHost.loopCount    = args->loopCount;
tilingHost.benchMode    = args->benchMode;
```

**正确性检查**:
- `channelSlot` 必须对应 `channelPtrsHost[slot] != 0` (有可用通道)
- `loopCount * 2 < sqDepth` (通常 256)，否则 SQ 会溢出

### 2.5 barrier 同步

```cpp
barrier(devCount);  // 等所有 rank 的 setup_ctx 完成
```

**关键**: 必须在 `HcclChannelAcquire` 之后、kernel launch 之前 barrier。
因为 `HcclChannelAcquire` 是跨 rank 握手，两个 rank 必须都完成后才能 launch。

### 2.6 mc2_native 专属: 构造 HcclCombinOpParam

```cpp
if (args->sqeMode == 3) {
    build_hccl_context(ctx, rank, devCount, args->bufBytes);
}
```

详见第 5 节。

### 2.7 kernel launch

```cpp
bool isSender = (rank == 0);

if (isSender) {
    aclrtSynchronizeStream(stream);  // 确保 stream 就绪
    mc2_native_bench_launch(
        (uint32_t)BLOCK_NUM,   // 1
        stream,
        (uint64_t)ctx->hcclContextDev,  // HcclCombinOpParam GM 地址
        (uint64_t)ctx->workspaceDev,    // workspace GM 地址
        (uint64_t)timingGm,             // timing 输出 GM
        &tilingHost);
    aclrtSynchronizeStream(stream);  // 等 kernel 完成

    // 读取 timing 并打印
    int64_t timingHost[TIMING_SLOTS_SIZE] = {0};
    aclrtMemcpy(timingHost, sizeof(timingHost), timingGm, sizeof(timingHost),
                ACL_MEMCPY_DEVICE_TO_HOST);
    // ... 计算带宽并打印
} else {
    std::this_thread::sleep_for(std::chrono::seconds(10));
}
```

**正确性检查**:
- rank0 (sender) 必须 `aclrtSynchronizeStream` 等 kernel 完成
- rank1 (receiver) 只需 sleep (URMA 单向, 接收方不做事)
- timing 读取必须在 `aclrtSynchronizeStream` 之后

---

## 3. setup_ctx() — 6 步建立通信通道

**文件**: `host/test_urma_p2p_bench.cpp:166`

### [1/6] 分配 workspace

```cpp
aclrtMalloc(&ctx.workspaceDev, bufBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemset(ctx.workspaceDev, bufBytes, 0, bufBytes);
```

- workspace 是 URMA 通信的目标缓冲区 (对端写数据到这里)
- 默认 100MB, 用 HUGE_FIRST 分配大页

### [2/6] 注册 workspace 给 HCCL

```cpp
CommMem mem;
mem.type = COMM_MEM_TYPE_DEVICE;
mem.addr = ctx.workspaceDev;
mem.size = bufBytes;
HcclCommMemReg(comm, tag.c_str(), &mem, &ctx.workspaceMemHandle);
```

- `aclrtMalloc` 只是普通显存，对端看不到
- `HcclCommMemReg` 注册后生成 `memHandle`，后续 `HcclChannelAcquire` 时附到 channel desc 上
- HCCL 跨 rank 交换后，对端的 `ChannelEntity.remoteBufferAddr` 指向这块内存

### [3/6] 枚举 UBC 链路, 构建 ChannelDesc

```cpp
build_channel_descs(comm, rank, world, &ctx.workspaceMemHandle, descs, descSlots, ctx.numLayers);
```

`build_channel_descs` 内部:
1. `HcclRankGraphGetLayers` — 获取网络层数
2. 对每层每个 peer: `HcclRankGraphGetLinks` — 枚举物理链路
3. 过滤 `linkProtocol == 4 (UBC_CTP) || 5 (UBC_TP)` — 只选 UB fabric 链路
4. 构建 `HcclChannelDesc`:
   - `remoteRank = peer`
   - `channelProtocol = linkList[i].linkAttr.linkProtocol`
   - `localEndpoint / remoteEndpoint` = 链路两端描述
   - `notifyNum = 3`
   - `memHandles = &workspaceMemHandle` (附上 workspace 注册句柄)

**正确性检查**:
- 日志 `[INFO] rank 0: → created channel desc for layer=0 peer=1` 必须出现
- 如果报 `HCCL_E_NOT_FOUND`，说明没有 UBC 链路 (需要同机 NPU)

### [4/6] HcclChannelAcquire — 跨 rank 握手

```cpp
ctx.channels.resize(descs.size(), 0);
HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_AIV,
                   descs.data(), static_cast<uint32_t>(descs.size()),
                   ctx.channels.data());
```

- HCCL 内部完成跨 rank 的 buffer/endpoint 交换
- 返回 `ChannelHandle` 数组，指向 device 上的 `ChannelEntity`
- **阻塞操作**: 两个 rank 必须同时到达

**正确性检查**:
- 两个 rank 都必须打印 `[4/6] HcclChannelAcquire done`
- 如果卡住，说明 peer 没到达 (检查 barrier 或线程启动)

### [5/6] resolve channel ptrs — handle → device VA

```cpp
// 旧 ABI: HcommChannelGetPtrByHandle (dlsym 动态查找)
// 新 ABI (CANN 9.1.0+): handle 本身就是 device VA
auto getPtrByHandle = reinterpret_cast<GetPtrByHandleFn>(
    dlsym(RTLD_DEFAULT, "HcommChannelGetPtrByHandle"));

if (getPtrByHandle != nullptr) {
    // 旧 ABI: 翻译 handle → device VA
    getPtrByHandle(ctx.channels.data(), ..., ptrs.data());
    ctx.channelPtrsHost[descSlots[i]] = ptrs[i];
} else {
    // 新 ABI: handle 就是 device VA
    ctx.channelPtrsHost[descSlots[i]] = (ChannelPtr)ctx.channels[i];
}
```

- `channelPtrsHost[slot]` = `ChannelEntity` 的 device 虚拟地址
- 下标 `slot = peer` (2P 单 layer 场景)

**正确性检查**:
- 日志 `[5/6] slot[1]=0x...` 值不为 0
- 旧 ABI 时 `HcommChannelGetPtrByHandle` 返回 `HCOMM_SUCCESS_VALUE=0`

### [6/6] 持久化到 device GM

```cpp
// channelPtrTableDev: ChannelEntity VA 数组
aclrtMalloc(&ctx.channelPtrTableDev, tableBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(ctx.channelPtrTableDev, tableBytes, ctx.channelPtrsHost.data(),
            tableBytes, ACL_MEMCPY_HOST_TO_DEVICE);

// stateGmDev: SQ/CQ 环形索引持久化区 (仅 -m 0/1/2 用, -m 3 用 PICI)
aclrtMalloc(&ctx.stateGmDev, stateBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemset(ctx.stateGmDev, stateBytes, 0, stateBytes);
```

- `channelPtrTableDev`: -m 0/1/2 通过 `channelPtrTable_.GetValue(slot)` 读取
- `stateGmDev`: -m 0/1/2 的 SQ/CQ 状态持久化
- **-m 3 (mc2_native) 不使用这两个**，而是用 `HcclCombinOpParam` 中的 PICI 窗口

---

## 4. build_hccl_context() — 构造 HcclCombinOpParam (仅 -m 3)

**文件**: `host/test_urma_p2p_bench.cpp:278`

这是 mc2_native 模式的核心: 从 HCCL 返回的 `ChannelEntity` 解码出 SQ/CQ/buffer/token 信息，
填入 `HcclCombinOpParam` 结构体，拷到 device 供 kernel 使用。

### 4.1 整体流程

```
build_hccl_context():
  1. memset HcclCombinOpParam, 设 rankId/rankDim/workSpace
  2. 分配 PICI 窗口 GM (1MB 前缀 + 512B * world)
  3. windowsOut[myRank] = PICI 窗口 GM 地址
  4. for each peer (skip self):
     a. 从 device 读 ChannelEntity (256B) — aclrtMemcpy D2H
     b. 解码: localBufferPtr / remoteBufferPtr / sqContextPtr / cqContextPtr
     c. 从 device 读 SqContext (128B) — aclrtMemcpy D2H
     d. 从 device 读 CqContext (128B) — aclrtMemcpy D2H
     e. 从 device 读 remote RegedBuffer (64B) — aclrtMemcpy D2H
     f. 从 device 读 local RegedBuffer (64B) — aclrtMemcpy D2H
     g. 填 HcclAiRMAWQ (SQ 信息) ← SqContext + RegedBuffer
     h. 填 HcclAiRMACQ (CQ 信息) ← CqContext
     i. windowsIn[peer] ← remote RegedBuffer.addr (对端 workspace 地址)
  5. aclrtMemcpy H2D: HcclCombinOpParam → device GM
```

### 4.2 地址解码 — resolvePtr

```cpp
auto resolvePtr = [](uint64_t base, uint64_t stored) -> uint64_t {
    return (stored < base) ? (base + stored) : stored;
};
```

HCCL 内部有时存相对偏移 (stored < base)，有时存绝对地址 (stored >= base)。
这个 lambda 统一处理: 如果 stored < base，说明是相对偏移，返回 base + stored；否则直接用 stored。

**正确性检查**:
- 解码出的 `sqVA` / `cqVA` / `dbAddr` 应为合理的 device 地址 (非 0)
- `sqDepth` 通常 256, `wqeSize` 通常 64
- `windowsIn[peer]` 应等于对端的 workspace 地址

### 4.3 字段映射

```
SqContext (128B) → HcclAiRMAWQ:
  sqVA       ← sqU64[SQ_OFF_SQVA / 8]        // SQ ring 地址
  dbAddr     ← sqU64[SQ_OFF_DBVA / 8]        // JFS DoorBell 地址
  wqeSize    ← sqU32[SQ_OFF_WQESIZE / 4]     // 64B
  sqDepth    ← sqU32[SQ_OFF_SQDEPTH / 4]     // 256
  tp_id      ← sqU32[SQ_OFF_TPID / 4]        // Traffic Profile ID
  rmtEid[16] ← &sqU32[SQ_OFF_REMOTE_EID / 4] // 远端 EID (128bit)

CqContext (128B) → HcclAiRMACQ:
  cqVA       ← cqU64[CQ_OFF_CQVA / 8]        // CQ ring 地址
  cqeSize    ← cqU32[CQ_OFF_CQESIZE / 4]     // 64B
  cqDepth    ← cqU32[CQ_OFF_CQDEPTH / 4]     // 256
  dbAddr     ← cqU64[CQ_OFF_DBVA / 8]        // JFC DoorBell 地址

remote RegedBuffer (64B) → HcclAiRMAWQ + windowsIn:
  rmtObjId      ← rmU32[RB_OFF_UB_TOKEN_ID / 4]   // remoteTokenId
  rmtTokenValue ← rmU32[RB_OFF_UB_TOKEN_VAL / 4]  // remoteTokenValue
  windowsIn[peer] ← rmU64[RB_OFF_ADDR / 8]        // 对端 workspace 地址

local RegedBuffer (64B) → HcclAiRMAWQ:
  localTokenId ← lcU32[RB_OFF_UB_TOKEN_ID / 4]    // localTokenId
```

### 4.4 PICI 窗口分配

```cpp
// PICI 窗口: windowsOut[myRank] + 1MB + 512B * dstRankId
// 所以需要分配 1MB 前缀 + 512B * world
uint64_t winOutBytes = WIN_PICI_OFFSET + piciBytes;  // 1MB + 512B * world
aclrtMalloc(&winOutDev, winOutBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemset(winOutDev, winOutBytes, 0, winOutBytes);
param.windowsOut[rank] = (uint64_t)winOutDev;
```

**正确性检查**:
- `windowsOut[rank]` 必须指向有效 GM (1MB + 512B * world 大小)
- PICI 窗口必须清零 (首次 isFirstFlag=0 → 首次通信时触发 InvalidateCqeStatus)

### 4.5 拷到 device

```cpp
aclrtMalloc(&contextDev, sizeof(HcclCombinOpParam), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(contextDev, sizeof(HcclCombinOpParam), &param,
            sizeof(HcclCombinOpParam), ACL_MEMCPY_HOST_TO_DEVICE);
ctx->hcclContextDev = contextDev;
```

`HcclCombinOpParam` 大小约: 8+8+4+4 + 8 + 64*8 + 64*8 + 4*4 + 8*4 + 64*(8+4+4+8+8+8+8+4+16+4+4+4) + 64*(4+8+4+4+8+8+8) ≈ 20KB+

---

## 5. Kernel 执行流程

### 5.1 launch

**文件**: `kernel/mc2_native/mc2_native_launch.cpp:33`

```cpp
void mc2_native_bench_launch(uint32_t blockDim, void *stream,
                             uint64_t hcclContextGM, uint64_t workspace,
                             uint64_t timingGm, UrmaP2pBenchTiling *tiling)
{
    mc2_native_bench_kernel<<<blockDim, nullptr, stream>>>(
        (void*)hcclContextGM, (void*)workspace, (void*)timingGm,
        tiling->peer, tiling->numSqe, tiling->sqeBytes, tiling->bufBytes,
        tiling->dbInterval, tiling->channelSlot, tiling->cqOwnerSlot,
        tiling->stateStride, tiling->loopCount, tiling->benchMode);
}
```

- `blockDim = 1` (单核)
- tiling 字段被打散成标量传入 kernel，kernel 内部重新组装

### 5.2 kernel 入口

**文件**: `kernel/mc2_native/mc2_native_launch.cpp:9`

```cpp
extern "C" __global__ __aicore__ void mc2_native_bench_kernel(...)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);  // 仅 AIV
    TPipe pipe;
    UrmaP2pBenchTiling tiling;
    // 从标量参数重建 tiling
    ...
    Mc2NativeBenchOp op;
    op.Init(hcclContextGM, workspace, timingGm, &pipe, &tiling);
    op.Process();
}
```

### 5.3 Init — 绑定 GM + 分配 UB

**文件**: `kernel/mc2_native/mc2_native_kernel.h:40`

```
Init():
  hcclContext_ = (HcclCombinOpParam*)hcclContextGM   // 指向 GM 中的上下文
  epRankId_ = hcclContext_->rankId                   // 本 rank ID
  localUrmaWorkspace_ = workspace                    // 本端 workspace

  UB 缓冲区分配:
    sqInfoBuf_  (sizeof(HcclAiRMAWQ), 对齐 32B)     // SQ 信息
    cqInfoBuf_  (sizeof(HcclAiRMACQ), 对齐 32B)     // CQ 信息
    sqeBuf_     (96B)                                // SQE 模板
    cqeBuf_     (8KB = 32B*256)                      // CQE 轮询缓冲区
    jfsDoorBellBuf_ (32B)                            // JFS DoorBell (保留)
    jfcDoorBellBuf_ (32B)                            // JFC DoorBell (保留)
    timingBuf_  (64B = 8*8B)                         // timing 打点
```

### 5.4 Process — kernel 主逻辑

**文件**: `kernel/mc2_native/mc2_native_kernel.h:173`

```
Process():
  if (aivId_ != 0) return;        // 仅 aivId==0 执行

  InitRankComm(peer);             // 初始化通信上下文 (8 步)

  // 预计算地址
  localBase  = workspace + peer * bufBytes + 350KB    // 本端源数据地址
  remoteBase = windowsIn[peer] + 350KB                // 对端目的地址

  if (benchMode == 0):
    // batch_write: loopCount 次 Write → 一次 Drain
    t0 = GetSystemCycle()
    for (i = 0..loopCount-1):
      off = (i % numSlots) * sqeBytes
      Write(peer, localBase+off, remoteBase+off, sqeBytes)
    t1 = GetSystemCycle()
    Drain(peer)
    t2 = GetSystemCycle()
    // timing: [0]=assemble=t1-t0, [2]=wait=t2-t1, [3]=total=t2-t0

  else (benchMode == 1):
    // write_drain_loop: 循环 { Write → Drain }
    for (i = 0..loopCount-1):
      t0 = GetSystemCycle()
      Write(peer, ...)
      t1 = GetSystemCycle()
      Drain(peer)
      t2 = GetSystemCycle()
      sumWrite += t1-t0; sumDrain += t2-t1; sumTotal += t2-t0
    // timing: [0]=sumWrite, [2]=sumDrain, [3]=sumTotal (Host 除以 loopCount)

  DataCopy(timingGm, timingLocal)   // timing 拷到 GM
```

### 5.5 InitRankComm — 8 步初始化

**文件**: `kernel/mc2_native/mc2_native_kernel.h:82`

```
InitRankComm(dstRankId):
  步骤1: GetURMASqInfoTensor(sqInfoTensor_, hcclContext_, dstRankId)
         → 从 HcclCombinOpParam.sqs[dstRankId] 读 SQ 信息到 UB
         SyncFunc<MTE2_S>

  步骤2: GetURMACqInfoTensor(cqInfoTensor_, hcclContext_, dstRankId)
         → 从 HcclCombinOpParam.cqs[dstRankId] 读 CQ 信息到 UB
         SyncFunc<MTE2_S>

  步骤3: GetIsFirstInComm(hcclContext_, epRankId_, dstRankId, isFirstComm)
         → 读 PICI 窗口的 isFirstFlag
         isFirstComm_ = !isFirstComm  (取反: flag=0→首次, flag=1→非首次)

  步骤4: GetPICI(hcclContext_, epRankId_, dstRankId, ...)
         → 读 windowsOut[myRank]+1MB+512B*dstRankId 的 sqPi/sqCi/cqCi
         恢复上次 launch 的 SQ/CQ 索引

  步骤5: if (isFirstComm_):
           InvalidateCqeStatus(cqInfoTensor_, cqeTensor_)
           → CQ ring 全填 0xff (CQE 初始 status 必须是无效值)
           SyncFunc<MTE3_S>
           UpdateIsFirstInComm(hcclContext_, epRankId_, dstRankId, true)
           → 标记已初始化

  步骤6: PollCommCQUpdateSQCI(...)
         → 清理残余 CQE + 敲 JFC DoorBell
         (上次 launch 可能遗留未消费的 CQE)

  步骤7: GenerateCommWriteWithNotifySQE(sqeTensor_, 1)
         → 生成 SQE 模板 (opcode=0x5, 96B)
         填固定字段: odr=0b001, owner=1, jetty_type=0b01, token_en=1, sge_num=1

  步骤8: UpdateCommWriteWithNotifySQE(sqeTensor_, sqInfoTensor_)
         → 从 sqInfoTensor 读 tp_id/eid/token, 填入 SQE 模板
         此后模板的元数据固定, 只需 SetCommWriteWithNotifySQE 填数据地址

  curDstRankId_ = dstRankId
  pendingSqCnt_ = 0
```

**正确性检查**:
- 首次运行时 `isFirstComm_` 必须为 true (PICI 窗口清零 → flag=0 → 取反=true)
- 第二次 launch 时 `isFirstComm_` 应为 false (flag 已被设为 1)
- `outSqPi_` / `outSqCi_` 首次应为 0, 后续 launch 应为上次 Drain 后的值

### 5.6 Write — 提交 SQE (不敲 DoorBell)

**文件**: `kernel/mc2_native/mc2_native_kernel.h:125`

```
Write(dstRankId, localAddr, remoteAddr, length):
  1. SQ 溢出保护: if (pendingSqCnt_ >= 200): FlushPendingSqes()

  2. 计算 notify 地址:
     notifyAddr = windowsIn[dstRankId] + 0 * 350KB + 32 * epRankId_
                = 对端状态区 + 本 rank 偏移 (每个 rank 32B)
     notifyData = 0x3F800000  (float 1.0 的位模式)

  3. SetCommWriteWithNotifySQE(sqeTensor_, localAddr, remoteAddr, length,
                                notifyAddr, notifyData, cqe=0)
     → 填 dataAddr/rmtAddr/length/notifyAddr/notifyData (cqe=0: 不生成 CQE)

  4. PutCommNotifySQE(sqInfoTensor_, ..., sqeTensor_, ..., 1, ...)
     → 检查 SQ 可用空间 → DataCopy SQE 到 SQ ring → 更新 outSqPi
     (不敲 DoorBell!)

  5. pendingSqCnt_++
```

**关键**: Write 只把 SQE 拷到 SQ ring，不通知硬件。
硬件不知道有新 WQE，直到 Drain 调用 SendJFSDoorBell。
这是批量写模式的核心: 攒一批 SQE 后统一敲一次 DoorBell，减少 DoorBell 开销。

**正确性检查**:
- `cqe=0` (Write 阶段不生成 CQE，Drain 阶段不依赖单个 CQE)
- `notifyAddr` 指向对端状态区 (不是数据区)
- `PutCommNotifySQE` 中如果 SQ 空间不足会自动 `PollCommCQUpdateSQCI` 腾空间

### 5.7 Drain — 敲 DoorBell + 等完成 + 更新 PICI

**文件**: `kernel/mc2_native/mc2_native_kernel.h:147`

```
Drain(dstRankId):
  1. SendJFSDoorBell(jfsDoorBellTensor_, sqInfoTensor_, outSqPiLinear_)
     → st_dev(sqPiLinear, dbAddr, 0)
     → 通知硬件 "SQ 有新 WQE 了, 快来取"

  2. PollCommCQUpdateSQCI(sqInfoTensor_, cqInfoTensor_, cqeTensor_, ...)
     → 循环读 CQE:
        status=0xff → 未完成, break
        status=0    → 完成, 消费 CQE (设 status=0xff 写回), outCqCi++
     → 更新 outSqCi = (newestCompletedPi + 1) % sqDepth
     → SendJFCDoorBell(通知硬件 cqCi 更新)

  3. UpdatePICI(hcclContext_, epRankId_, dstRankId,
                outSqPi_, outSqCi_, outCqPi_, outCqCi_,
                outSqPiLinear_, outCqCiLinear_)
     → 写回 PICI 窗口, 供下次 launch 读取

  4. pendingSqCnt_ = 0
```

**正确性检查**:
- `SendJFSDoorBell` 传的是 `outSqPiLinear_` (线性计数，不取模)，硬件用此判断新增 WQE 数量
- `PollCommCQUpdateSQCI` 必须能读到 CQE (status=0)，否则会一直循环到 `cqDepth-1` 次后退出
- `UpdatePICI` 后，下次 `GetPICI` 应读出相同的 `outSqPi_` / `outSqCi_` 值

### 5.8 地址计算详解

**文件**: `kernel/mc2_native/mc2_native_kernel.h:261`

```
workspace 内存布局 (每个 rank):
  [0 .. 350KB-1]                    状态区 (notify flag 等)
  [350KB .. bufBytes-1]             数据区 (URMA 写入的数据)

GetLocalWinAddr(rankId, winSize, offset):
  = workspace + rankId * winSize + 350KB + offset
  → 本端源数据地址 (URMA 从这里读数据)

GetRemoteWinAddr(rankId, offset):
  = windowsIn[rankId] + 350KB + offset
  → 对端目的地址 (URMA 把数据写到这里)

GetRemoteWinStateAddr(rankId, dataState):
  = windowsIn[rankId] + dataState * 350KB
  → 对端状态区地址 (notify 写到这里)

notify 地址:
  = GetRemoteWinStateAddr(peer, 0) + 32 * myRank
  = windowsIn[peer] + 32 * myRank
  → 对端状态区内本 rank 的 32B 槽位
```

**正确性检查**:
- `localBase` 和 `remoteBase` 都跳过 350KB 状态区 (从数据区开始)
- `localBase = workspace + peer * bufBytes + 350KB`: 注意这里 `peer * bufBytes` 是多 rank 场景的偏移
  (2P 场景 peer=1, 所以 localBase = workspace + bufBytes + 350KB)
  **但这意味着 workspace 需要至少 `2 * bufBytes` 大小！**
  实际上 Host 侧 `aclrtMalloc(&ctx.workspaceDev, bufBytes, ...)` 只分配了 `bufBytes`，
  所以 `localBase = workspace + bufBytes + 350KB` 会越界！

  **这是一个潜在问题**: `GetLocalWinAddr` 的 `totalWinSizeEp * rankId` 在 2P 场景下
  会偏移到 workspace 之外。需要确认 workspace 是否足够大，或者 `totalWinSizeEp` 的语义。
  从代码看 `totalWinSizeEp = bufBytes`，`rankId = peer = 1`，
  所以 `localBase = workspace + bufBytes + 350KB`，确实超出了 `bufBytes` 大小的 workspace。

  **可能的原因**: 实际 URMA 写的数据源地址不一定要在 workspace 内。
  但这会导致读越界数据。需要验证 workspace 分配是否应该更大，或者 `totalWinSizeEp` 应为 0。

---

## 6. Timing 读取与带宽计算

**文件**: `host/test_urma_p2p_bench.cpp:547`

```cpp
int64_t timingHost[TIMING_SLOTS_SIZE] = {0};
aclrtMemcpy(timingHost, sizeof(timingHost), timingGm, sizeof(timingHost),
            ACL_MEMCPY_DEVICE_TO_HOST);

double divisor = (double)args->loopCount;
double avgAssembleNs = (double)timingHost[0] * NS_PER_CYCLE / divisor;
double avgWaitNs     = (double)timingHost[2] * NS_PER_CYCLE / divisor;
double avgTotalNs    = (double)timingHost[3] * NS_PER_CYCLE / divisor;

double avgBwGBs = (double)args->sqeBytes / avgTotalNs;
```

| timing slot | benchMode=0 | benchMode=1 |
|-------------|-------------|-------------|
| [0] assemble | 全部 Write 时间 (不除 loopCount) | 累计 Write 时间 (除 loopCount 得平均) |
| [2] wait | Drain 时间 (只 1 次) | 累计 Drain 时间 (除 loopCount) |
| [3] total | Write + Drain 总量 | 累计 total (除 loopCount) |

**带宽计算**:
- `avgBwGBs = sqeBytes / avgTotalNs` (单次数据量 / 单次时间)
- benchMode=0: `avgTotalNs` 已除 loopCount，所以是平均单次时间
- benchMode=1: `avgTotalNs` 也已除 loopCount

---

## 7. 完整数据流总结

```
Host build_hccl_context():
  ChannelEntity (device) ──aclrtMemcpy D2H──▶ Host buf
    解码: sqCtxVA, cqCtxVA, lcBufVA, rmBufVA
      SqContext (device) ──aclrtMemcpy D2H──▶ sqBuf
      CqContext (device) ──aclrtMemcpy D2H──▶ cqBuf
      RegedBuffer (device) ──aclrtMemcpy D2H──▶ rmBuf / lcBuf
        填入 HcclAiRMAWQ / HcclAiRMACQ / HcclCombinOpParam
          ──aclrtMemcpy H2D──▶ hcclContextDev (device GM)

Kernel Process():
  InitRankComm:
    hcclContext_ (GM) ──DataCopyPad──▶ sqInfoTensor_ (UB)
    hcclContext_ (GM) ──DataCopyPad──▶ cqInfoTensor_ (UB)
    PICI窗口 (GM) ──Scalar读──▶ outSqPi_ 等
    [首次] cqeTensor_ (UB) ──Duplicate 0xff──▶ CQ ring (GM)

  Write:
    SetCommWriteWithNotifySQE: 填 sqeTensor_ (UB) 的数据字段
    PutCommNotifySQE: sqeTensor_ (UB) ──DataCopy──▶ SQ ring (GM)

  Drain:
    SendJFSDoorBell: st_dev(sqPiLinear, dbAddr)  // Scalar 写 DoorBell
    PollCommCQUpdateSQCI: CQ ring (GM) ──DataCopyPad──▶ cqeTensor_ (UB) 读 status
    UpdatePICI: outSqPi_ 等 ──Scalar写──▶ PICI窗口 (GM)

  timing: timingLocal (UB) ──DataCopy──▶ timingGm (GM)

Host: timingGm (GM) ──aclrtMemcpy D2H──▶ timingHost ──▶ 打印
```

---

## 8. 正确性检查清单

### 8.1 Host 侧

- [ ] `devCount >= 2` (至少 2 卡)
- [ ] `devIds` 中的设备都存在
- [ ] 两个 rank 用同一个 `rootInfo`
- [ ] `HcclCommInitRootInfoConfig` 两个 rank 都返回 `HCCL_SUCCESS`
- [ ] `build_channel_descs` 找到 UBC_CTP/UBC_TP 链路
- [ ] `HcclChannelAcquire` 两个 rank 都完成
- [ ] `channelPtrsHost[slot]` 不为 0
- [ ] `build_hccl_context` 中 `sqVA` / `cqVA` / `dbAddr` 解码正确 (日志打印)
- [ ] `sqDepth == 256`, `wqeSize == 64`
- [ ] `windowsIn[peer]` 等于对端 workspace 地址
- [ ] `windowsOut[rank]` 指向有效 PICI GM (1MB + 512B * world)
- [ ] barrier 在 launch 之前
- [ ] rank0 是 sender, rank1 是 receiver (sleep)

### 8.2 Kernel 侧

- [ ] `aivId_ == 0` (单核执行)
- [ ] `hcclContext_` 的 `rankId` / `rankDim` 正确
- [ ] 首次运行 `isFirstComm_ == true` (触发 InvalidateCqeStatus)
- [ ] 第二次运行 `isFirstComm_ == false` (跳过初始化)
- [ ] `outSqPi_` / `outSqCi_` 首次为 0, 后续为上次 Drain 后的值
- [ ] Write 的 `cqe=0` (不生成单个 CQE)
- [ ] `pendingSqCnt_` 在 Drain 后归 0
- [ ] `SendJFSDoorBell` 传 `outSqPiLinear_` (线性计数，不取模)
- [ ] `PollCommCQUpdateSQCI` 能读到 `status=0` 的 CQE (否则会超时循环)
- [ ] `UpdatePICI` 写回的值与 `GetPICI` 下次读出的值一致

### 8.3 地址正确性

- [ ] `localBase` 在 workspace 范围内 (注意 `totalWinSizeEp * rankId` 偏移)
- [ ] `remoteBase` = `windowsIn[peer] + 350KB` (跳过状态区)
- [ ] `notifyAddr` = `windowsIn[peer] + 32 * myRank` (对端状态区内本 rank 槽位)
- [ ] `notifyData` = `0x3F800000` (float 1.0 的位模式)
- [ ] 每次 Write 的 `off = (i % numSlots) * sqeBytes` (循环使用不同偏移)

### 8.4 限制条件

- [ ] `loopCount * 2 < sqDepth` (WriteWithNotify 每个 SQE 占 2 个 WQEBB, 通常 256)
- [ ] `sqeBytes <= bufBytes` (单次传输不超过 workspace)
- [ ] `sqeBytes` 建议 32B 对齐
- [ ] 需要 UBC_CTP 或 UBC_TP 物理链路 (同机 NPU 间)
