# 第一阶段详细设计：x1 AllGather（CCU 通信 + AIV 验证拷贝）

> 隶属：CCU_ALLGATHER_MATMUL_PLAN.md 第一阶段
> 目标：CCU AllGather 通信跑通 + 精度验证（极简 case，不泛化）
> 同步架构：CCU 通信 + AIV 同步桥（三核协同的第一/二阶段形态，无 AIC）
> 参考：`02_allgather_add/main.asc`，通信侧整体复用

### 阶段定位

| 阶段 | 步骤 | 目标 | 核验内容 | 同步架构 |
|------|------|------|---------|---------|
| **一（本设计）** | Step 1 | CCU AllGather 通信跑通 | 不崩溃、不卡死（不关注精度） | CCU + AIV 同步桥 |
| **一（本设计）** | Step 2 | 通信精度验证 | recvBuf 拼接顺序与内容正确 | （同上） |
| 二 | — | AllGather + Matmul 串联端到端精度 | C = concat(A) @ B，BF16 rtol/atol=1e-3 | + CrossCoreSetFlag 唤醒 AIC |
| 三 | — | 通算并行流水 + AIC 直轮询 | 两次 allgather + 两次 matmul overlap | CCU + AIC 直轮询（去 AIV 桥） |

> 本设计覆盖第一阶段 Step 1 + Step 2。极简 case：2 卡，M=256, K=256, FP16。

---

## 1. 极简 case 定义

| 参数 | 值 | 说明 |
|------|-----|------|
| rankSize | 2 | 两卡 |
| M | 256 | 每 rank A 分片行数 |
| K | 256 | A 列数（= B 行数，第一阶段不涉及 B） |
| dtype | FP16 (half) | `sizeof = 2 bytes` |
| inputSize | `M*K*2 = 256*256*2 = 131072 bytes (128KB)` | 每 rank 发送字节数 |
| recvSize | `rankSize * inputSize = 256KB` | AllGather 聚合后总字节 |
| sendBuf | `aclrtMalloc(128KB)` | 本 rank 输入 |
| recvBuf | `aclrtMalloc(256KB)` | AllGather 拼接结果 |
| computeBuf | `aclrtMalloc(256KB)` | 验证拷贝输出（= recvBuf） |

**recvBuf rank-major 布局**（CCU AllGather 后）：
```
recvBuf: [rank0 的 128KB 数据 | rank1 的 128KB 数据]
          offset 0              offset 128KB (= inputSize * rankId)
```

---

## 2. 文件结构（对齐 02_allgather_add 独立工程）

```
apace/examples/ccu_all_gather_matmul/
├── ccu_all_gather_matmul_kernel.asc   # 第一阶段主体（CCU kernel + AIV verify_copy + Host 编排 + main，单文件仿 02_allgather_add/main.asc）
├── ccu_all_gather_matmul_tiling_data.h # Tiling POD（第一阶段简化）
├── CMakeLists.txt                      # 逐字节对齐 02_allgather_add/CMakeLists.txt
├── run.sh                              # 编译+运行+设置 CCU_SCHED 环境变量
├── verify_result.py                    # 可选：脚本化校验（进程内打印为主）
└── README.md
```

> 第一阶段单文件 `ccu_all_gather_matmul_kernel.asc` 仿 `02_allgather_add/main.asc`：CCU kernel + AIV verify_copy + Host 编排 + main 全在一个文件。数据在 main/Sample 内生成（rank i 填充 i+1），进程内打印校验，无需外部 gen_data.py。

---

## 3. CCU 通信 kernel（整体复用 02_allgather_add）

### 3.1 复用清单（逐函数）

| 函数 | 02_allgather_add 位置 | 是否改动 | 说明 |
|------|----------------------|---------|------|
| `CcuKernelArgAllGatherMesh1DMem2Mem` | L136 | 不变 | CCU kernel 参数结构 |
| `AllGatherMesh1DMem2MemContext` | L147 | 不变 | CCU 运行态上下文 |
| `CcuMappedAddrs` | L181 | 不变 | 跨流同步地址桥（eventVa[2]+startVa[10]） |
| `CalGoSize` | L330 | 不变 | 大块分片循环参数计算 |
| `InitGroupCopyResources` | L413 | 不变 | GroupCopy 资源初始化 |
| `GroupCopy` | L450 | 不变 | 本地拷贝（经 CCU Buffer 中转） |
| `ExecuteAllGatherTransfer` | L525 | 不变 | AllGather 核心：ccu::Write 远端 + GroupCopy 本地 + Event 同步 |
| `CcuAllGatherMesh1DMem2MemKernel` | L571 | 不变 | CCU kernel 入口（`__ccu_host__`） |
| `AcquireCcuChannels` | L259 | 不变 | 获取 CCU 通信通道 |
| `CreateCcuInsWithEnabledDies` | L216 | 不变 | 创建 CCU 实例 |

**结论**：CCU 通信 kernel **逐字节复用**，零改动。

### 3.2 CCU kernel 伪代码（复用，仅展示流程）

```cpp
__ccu_host__ void CcuAllGatherMesh1DMem2MemKernel(CcuKernelArg arg) {
    // 1. 解析参数：channels / rankId / rankSize / varHandle / eventHandle
    // 2. 从 channel 获取远端 output/token 变量
    // 3. EventWait(events[0])          ← 阻塞等 AIV 写好参数
    // 4. 从 Variable 槽读取本 rank 参数（vars[0..9]）
    //    vars[0]=sendBuf, vars[1]=recvBuf, vars[2]=token,
    //    vars[3]=inputOffset(0), vars[4]=outputOffset(inputSize*rankId),
    //    vars[5]=sliceSize(inputSize), vars[6..9]=goSize
    // 5. WriteVariableWithNotify + NotifyWait  ← 与远端 rank 交换 output/token
    // 6. ExecuteAllGatherTransfer():
    //    - ccu::Write 把本 rank 分片发到所有远端 rank 的 recvBuf 对应位置
    //    - GroupCopy 本 rank 分片拷到本地 recvBuf
    //    - EventRecord(event, 1<<rankId) + EventWait(event, totalMask) 同步
    // 7. post-sync (NotifyRecord/NotifyWait)
    // 8. EventRecord(events[1])         ← 通知 AIV 通信完成
}
```

---

## 4. AIV 验证拷贝 kernel（替换 vector_add）

### 4.1 职责
1. **参数桥**：写 taskArgs[0..9] → GM（通信参数送 CCU Variable 槽）
2. **唤醒 CCU**：写 ckeAddr0=1
3. **轮询等待**：读 ckeAddr1==1
4. **验证拷贝**：DataCopy(recvBuf → UB → computeBuf)，不做任何计算

### 4.2 伪代码

```cpp
template <typename T, uint64_t maxCount>
__global__ __vector__ void verify_copy(
    __gm__ void *src,               // 本 rank 输入 sendBuf（参数桥用，命名对齐参考样例）
    __gm__ void *recvBuf,           // AllGather 结果（拷贝源）
    __gm__ void *dstGm,             // 验证输出 computeBuf（拷贝目标）
    CcuMappedAddrs mappedAddrs,     // 跨流同步地址桥
    uint64_t token,
    uint64_t currentRankSliceInputOffset,   // = 0
    uint64_t currentRankSliceOutputOffset,  // = inputSize * rankId
    uint64_t sliceSize,                     // = inputSize
    uint64_t goSize0, uint64_t goSize1, uint64_t goSize2, uint64_t goSize3,
    uint64_t count)                         // = recvCount（元素数，非字节数）
{
    AscendC::InitSocState();

    // ---- 通信输出 buffer（命名对齐参考样例 vector_add L645-648）----
    AscendC::GlobalTensor<T> srcGlobal;   // recvBuf
    AscendC::GlobalTensor<T> dstGlobal;   // dstGm
    srcGlobal.SetGlobalBuffer((__gm__ T *)recvBuf);
    dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);

    // ---- UB buffer（验证拷贝用，仅 srcLocal，无 dstLocal）----
    AscendC::LocalMemAllocator<AscendC::Hardware::UB> ubAllocator;
    AscendC::LocalTensor<T> srcLocal = ubAllocator.Alloc<T, maxCount>();

    // ============ 参数桥：写 taskArgs[0..9] → GM（逐行对齐参考样例 L654-675）============
    __gm__ uint64_t *taskArgs0 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[0]);
    __gm__ uint64_t *taskArgs1 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[1]);
    __gm__ uint64_t *taskArgs2 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[2]);
    __gm__ uint64_t *taskArgs3 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[3]);
    __gm__ uint64_t *taskArgs4 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[4]);
    __gm__ uint64_t *taskArgs5 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[5]);
    __gm__ uint64_t *taskArgs6 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[6]);
    __gm__ uint64_t *taskArgs7 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[7]);
    __gm__ uint64_t *taskArgs8 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[8]);
    __gm__ uint64_t *taskArgs9 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.startVa[9]);

    AscendC::WriteGmByPassDCache<uint64_t>(taskArgs0, reinterpret_cast<uint64_t>(src));
    AscendC::WriteGmByPassDCache<uint64_t>(taskArgs1, reinterpret_cast<uint64_t>(recvBuf));
    AscendC::WriteGmByPassDCache<uint64_t>(taskArgs2, token);
    AscendC::WriteGmByPassDCache<uint64_t>(taskArgs3, currentRankSliceInputOffset);
    AscendC::WriteGmByPassDCache<uint64_t>(taskArgs4, currentRankSliceOutputOffset);
    AscendC::WriteGmByPassDCache<uint64_t>(taskArgs5, sliceSize);
    AscendC::WriteGmByPassDCache<uint64_t>(taskArgs6, goSize0);
    AscendC::WriteGmByPassDCache<uint64_t>(taskArgs7, goSize1);
    AscendC::WriteGmByPassDCache<uint64_t>(taskArgs8, goSize2);
    AscendC::WriteGmByPassDCache<uint64_t>(taskArgs9, goSize3);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();

    // ============ 唤醒 CCU：写 ckeAddr0 = 1 ============
    __gm__ uint64_t *ckeAddr0 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[0]);
    __gm__ uint64_t *ckeAddr1 = reinterpret_cast<__gm__ uint64_t *>(mappedAddrs.eventVa[1]);
    AscendC::WriteGmByPassDCache<uint64_t>(ckeAddr0, 1);
    AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();

    // ============ 轮询等待 CCU 通信完成 ============
    while (1) {
        uint64_t waitCke = AscendC::ReadGmByPassDCache(ckeAddr1);
        AscendC::DataSyncBarrier<AscendC::MemDsbT::ALL>();
        AscendC::Nop<800>();              // 空转 800 拍降总线压力
        if (waitCke == 1) {
            break;
        }
    }

    // ============ 验证拷贝：recvBuf → UB → computeBuf ============
    // 无计算（vector_add 此处是 Adds，本方案跳过），故无需 MTE2_V/V_MTE3 流水同步
    AscendC::DataCopy(srcLocal, srcGlobal, count);
    AscendC::DataCopy(dstGlobal, srcLocal, count);
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

### 4.3 相对 vector_add 的改动点
1. **类名/函数名**：`vector_add` → `verify_copy`
2. **去掉 Adds 计算**：删除 `T scalar = 8.0; AscendC::Adds(dstLocal, srcLocal, scalar, count);`
3. **去掉 dstLocal**：参考样例分配 dstLocal 是因为 Adds 输出到 dstLocal；本方案无计算，不需要 dstLocal，直接 `DataCopy(srcLocal, srcGlobal)` 后 `DataCopy(dstGlobal, srcLocal)`
4. **去掉 MTE2_V/V_MTE3 流水**：参考样例保留流水是因为 MTE2→V(Adds)→MTE3 三段需同步；本方案无 V 阶段计算，MTE2 直接到 MTE3 无需 flag 同步
5. **模板参数**：`T=half`（FP16），`maxCount=131072`（recvCount 元素数，= 2卡×256×256）

---

## 5. Host 编排（AllGatherCustom 入口）

### 5.1 伪代码

```cpp
HcclResult AllGatherCustom(
    void *computeBuf, void *sendBuf, void *recvBuf,
    uint64_t sendCount,                    // 每 rank 元素数 = M*K = 65536
    HcclDataType dataType,                 // HCCL_DATA_TYPE_FP16
    HcclComm comm,
    aclrtStream streamAiv, aclrtStream streamCcu)
{
    // ---- 1. 参数校验 ----
    CHK_PTR_NULL(streamAiv); CHK_PTR_NULL(streamCcu); CHK_PTR_NULL(comm);
    CHK_PTR_NULL(computeBuf); CHK_PTR_NULL(sendBuf); CHK_PTR_NULL(recvBuf);
    // 放开 FP32 硬校验，支持 FP16
    // 移除 sendCount != 256 硬校验

    uint32_t rankId, rankSize;
    HCCLCHECK(HcclGetRankId(comm, &rankId));
    HCCLCHECK(HcclGetRankSize(comm, &rankSize));

    uint32_t perDataSize = SIZE_TABLE[dataType];    // FP16 = 2
    uint64_t inputSize = sendCount * perDataSize;   // 65536 * 2 = 128KB
    if (sendCount == 0) return HCCL_SUCCESS;

    // ---- 2. 单 rank 退化处理（rankSize==1，逐字节复用参考样例 L729-737）----
    if (rankSize == 1) {
        ThreadHandle thread;
        uint32_t notifyNumOnMainThread = 0;
        uint32_t threadNum = 1;
        HCCLCHECK(HcclThreadAcquire(comm, CommEngine::COMM_ENGINE_CCU, threadNum, notifyNumOnMainThread, &thread));
        HCCLCHECK(static_cast<HcclResult>(HcommLocalCopyOnThread(thread, recvBuf, sendBuf, inputSize)));
        return HCCL_SUCCESS;
    }

    // ---- 3. 获取 CCU 通信通道 ----
    std::vector<ChannelHandle> kernelChannels;
    HCCLCHECK(AcquireCcuChannels(comm, rankId, rankSize, kernelChannels));

    // ---- 4. 创建 CCU 实例并绑定通信域 ----
    CcuInsHandle insHandle{0};
    HCCLCHECK(CreateCcuInsWithEnabledDies(insHandle));
    HcclResult assignRet = HcclCommAssignCcuIns(comm, insHandle);
    if (assignRet != HCCL_SUCCESS) {
        HcommCcuInsDestroy(insHandle);
        return assignRet;
    }

    // ---- 5. 分配 Variable（10 个）+ Event（2 个），获取 GM 地址 ----
    CcuMappedAddrs mappedAddrs{};
    CcuVariableHandle varHandle{0};
    HCCLCHECK(HcommCcuVariableAlloc(insHandle, DIE_ID, HCOMM_HOST_TASK_ARG_NUM, &varHandle));
    for (i = 0..9) HCCLCHECK(HcommCcuVariableGetAddr(varHandle, i, &mappedAddrs.startVa[i]));

    CcuEventHandle eventHandle{0};
    HCCLCHECK(HcommCcuEventAlloc(insHandle, DIE_ID, CKE_NUM, &eventHandle));
    for (i = 0..1) HCCLCHECK(HcommCcuEventGetAddr(eventHandle, i, &mappedAddrs.eventVa[i]));

    // ---- 6. 组装 kernelArg ----
    auto kernelArg = std::make_shared<CcuKernelArgAllGatherMesh1DMem2Mem>();
    kernelArg->rankSize = rankSize;
    kernelArg->rankId = rankId;
    kernelArg->channelCount = kernelChannels.size();
    kernelArg->varHandle = varHandle;
    kernelArg->varNum = HCOMM_HOST_TASK_ARG_NUM;
    kernelArg->eventHandle = eventHandle;
    kernelArg->eventNum = CKE_NUM;
    for (i...) kernelArg->channels[i] = kernelChannels[i];

    // ---- 7. 获取通信内存 token ----
    uint64_t token = 0;
    HCCLCHECK(HcommCcuGetMemToken(reinterpret_cast<uint64_t>(sendBuf), inputSize, &token));

    // ---- 8. 计算大块分片循环参数 ----
    LoopGroupConfig config{};
    config.msInterleave = CCU_MS_INTERLEAVE;
    config.loopCount = CCU_MS_LOCAL_COPY_LOOP_COUNT;
    config.memSlice = CCU_MS_SIZE * CCU_LOCAL_COPY_MS_PER_LOOP;
    auto goSize = CalGoSize(inputSize, config);

    // ---- 9. 组装 taskArgs（参数桥）----
    uint64_t taskArgs[10] = {
        reinterpret_cast<uint64_t>(sendBuf),    // [0] sendBuf 地址
        reinterpret_cast<uint64_t>(recvBuf),    // [1] recvBuf 地址
        token,                                  // [2] 通信内存 token
        0,                                      // [3] inputOffset = 0
        inputSize * rankId,                     // [4] outputOffset = inputSize * rankId
        inputSize,                              // [5] sliceSize = inputSize
        goSize[0], goSize[1], goSize[2], goSize[3]  // [6..9] goSize
    };

    // ---- 10. 下发 CCU kernel ----
    HCommCcuSchd schd = {1, 0, 0x01};    // {taskNum=1, dieId=0, dieMask=0x01}
    CcuAllGatherMesh1DMem2MemKernel<<<schd, insHandle, streamCcu>>>(kernelArg.get());

    // ---- 11. 下发 AIV verify_copy kernel ----
    const uint64_t recvCount = sendCount * rankSize;    // 65536 * 2 = 131072
    verify_copy<half, 131072><<<1, nullptr, streamAiv>>>(
        sendBuf, recvBuf, computeBuf, mappedAddrs,
        taskArgs[2], taskArgs[3], taskArgs[4], taskArgs[5],
        taskArgs[6], taskArgs[7], taskArgs[8], taskArgs[9],
        recvCount);

    return HCCL_SUCCESS;
}
```

### 5.2 相对 02_allgather_add 的改动点
1. 函数名 `AllGatherAddCustom` → `AllGatherCustom`（去掉 add 语义）
2. 去掉 `computeBuf` 的 add 用途，改为验证拷贝目标
3. 放开 `dataType != HCCL_DATA_TYPE_FP32` 和 `sendCount != 256` 硬校验
4. AIV kernel 调用 `vector_add` → `verify_copy`
5. `taskArgs` 组装逻辑**不变**（参数桥字段含义不变）

---

## 6. main() 多 rank 启动（复用 02_allgather_add thread 模式）

### 6.1 方案选择

**采用 `std::thread` 多线程模式**，与核心参考 `02_allgather_add/main.asc` 完全对齐。每设备一个线程，各线程 `aclrtSetDevice` + `HcclCommInitRootInfo`（同一 rootInfo 指针直接传递，无需 TCP 交换）。

| 模式 | 来源 | 选择 |
|------|------|------|
| `std::thread` 多线程 | 02_allgather_add L909 | **采用**（对齐核心参考，少一层 TCP 依赖） |
| `fork` 多进程 + TCP RootInfoExchanger | apace ST | 不采用 |

**采用 thread 模式理由**：核心参考是 02_allgather_add，第一阶段目标是跑通+对齐参考，直接复用其 main 骨架最稳妥；rootInfo 在同进程内通过指针共享，无需 TCP 交换；少一个外部依赖（RootInfoExchanger）。

### 6.2 伪代码（复用 02_allgather_add L811-943 骨架）

```cpp
struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
};

int Sample(void *arg) {
    ThreadContext *ctx = (ThreadContext *)arg;
    void *computeBuf = nullptr, *sendBuf = nullptr, *recvBuf = nullptr;
    uint32_t device = ctx->device;

    // ---- 极简 case 参数 ----
    constexpr uint64_t M = 256, K = 256;
    constexpr uint64_t sendCount = M * K;                 // 每 rank 元素数 = 65536
    const uint64_t recvCount = sendCount * ctx->devCount; // 2卡 = 131072
    size_t sendSize = sendCount * sizeof(half);           // 128KB
    size_t recvSize = recvCount * sizeof(half);           // 256KB

    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    // ---- 初始化集合通信域（同 rootInfo 指针）----
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    // ---- 创建双流 ----
    aclrtStream streamAiv, streamCcu;
    ACLCHECK(aclrtCreateStream(&streamAiv));
    ACLCHECK(aclrtCreateStream(&streamCcu));

    // ---- Device 内存 ----
    ACLCHECK(aclrtMalloc(&computeBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));

    // ---- Host 输入数据：rank i 填充 i+1（便于校验拼接顺序）----
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, sendSize));
    half *tmpHostBuff = static_cast<half *>(hostBuf);
    for (uint64_t i = 0; i < sendCount; ++i) {
        tmpHostBuff[i] = static_cast<half>(device) + 1;  // rank0=1.0, rank1=2.0
    }
    ACLCHECK(aclrtMemcpy(sendBuf, sendSize, hostBuf, sendSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostBuf));

    // ---- 调用 AllGatherCustom ----
    HCCLCHECK(AllGatherCustom(computeBuf, sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP16,
                              hcclComm, streamAiv, streamCcu));

    // ---- 同步 ----
    ACLCHECK(aclrtSynchronizeStream(streamAiv));
    ACLCHECK(aclrtSynchronizeStream(streamCcu));

    // ---- 读回结果 + 打印 ----
    std::this_thread::sleep_for(std::chrono::seconds(ctx->device));  // 错峰打印
    void *resultBuff;
    ACLCHECK(aclrtMallocHost(&resultBuff, recvSize));
    ACLCHECK(aclrtMemcpy(resultBuff, recvSize, computeBuf, recvSize, ACL_MEMCPY_DEVICE_TO_HOST));
    half *tmpResBuff = static_cast<half *>(resultBuff);
    std::cout << "rankId: " << device << ", computeBuf: [";
    for (uint32_t i = 0; i < recvCount; ++i) {
        std::cout << " " << tmpResBuff[i];
    }
    std::cout << " ]" << std::endl;
    ACLCHECK(aclrtFreeHost(resultBuff));

    // ---- 释放资源 ----
    HCCLCHECK(HcclCommDestroy(hcclComm));
    if (computeBuf) ACLCHECK(aclrtFree(computeBuf));
    if (sendBuf) ACLCHECK(aclrtFree(sendBuf));
    if (recvBuf) ACLCHECK(aclrtFree(recvBuf));
    ACLCHECK(aclrtDestroyStream(streamCcu));
    ACLCHECK(aclrtDestroyStream(streamAiv));
    ACLCHECK(aclrtResetDevice(device));
    return 0;
}

int main() {
    ACLCHECK(aclInit(NULL));
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    // rootInfo 生成（rootRank=0）
    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    // 多线程启动（每设备一线程）
    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    for (uint32_t i = 0; i < devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = devCount;
        threads[i] = std::thread(Sample, (void *)&args[i]);
    }
    for (uint32_t i = 0; i < devCount; i++) {
        threads[i].join();
    }

    ACLCHECK(aclrtFreeHost(rootInfoBuf));
    ACLCHECK(aclrtFinalize());
    return 0;
}
```

### 6.3 相对 02_allgather_add 的改动点
1. **数据类型**：`float` → `half`（FP16）
2. **sendCount**：固定 256 → `M*K = 65536`（动态，但第一阶段硬编码极简值）
3. **输入填充**：`device+1`（FP32）→ `static_cast<half>(device)+1`（FP16，语义不变）
4. **调用**：`AllGatherAddCustom` → `AllGatherCustom`
5. **ThreadContext**：不变（rootInfo/device/devCount 三字段）

---

## 7. 测试与验证（thread 模式，进程内生成+校验）

### 7.1 验证方式
thread 模式下数据在 Sample() 内生成（rank i 填充 i+1），结果在进程内打印。验证分两种：
- **Step 1（跑通）**：观察程序正常退出、打印输出不崩溃
- **Step 2（精度）**：人工/脚本检查打印的 computeBuf 前 65536 个 == 1.0，后 65536 个 == 2.0

### 7.2 可选 verify_result.py（若需脚本化校验）
```python
# 从程序 stdout 抓取 computeBuf 打印行，解析为数组后校验
# 或：程序内可选将 computeBuf dump 到文件，脚本读取校验
import numpy as np
M, K, RANK_NUM = 256, 256, 2
recvCount = M * K * RANK_NUM  # 131072
# 期望：前 65536 == 1.0，后 65536 == 2.0
expected = np.concatenate([np.full(M*K, r+1, dtype=np.float16) for r in range(RANK_NUM)])
# 读取 dump 文件（若程序支持 --dump 选项）
out = np.fromfile("computeBuf_dump.bin", dtype=np.float16)
print("[PASS]" if np.array_equal(out, expected) else "[FAIL]")
```

> 第一阶段极简 case，优先进程内打印人工校验；脚本化校验为可选增强。

### 7.3 run.sh
```bash
#!/bin/bash
set -e
# CCU 调度模式必须设置
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
source ${ASCEND_HOME_PATH}/set_env.sh

# 编译
mkdir -p build && cd build
cmake .. && make -j

# 运行（单进程，内部多线程起 2 rank）
./ccu_all_gather_matmul
```

---

## 8. CMake 构建要点（第一阶段，逐字节对齐 02_allgather_add）

第一阶段**不含 matmul**，无需 blaze/tensor_api 依赖。CMake 逐字节复用参考样例 `02_allgather_add/CMakeLists.txt`，仅改可执行文件名和源文件名：

```cmake
cmake_minimum_required(VERSION 3.16.0)
find_package(ASC REQUIRED)
project(ccu_all_gather_matmul LANGUAGES CXX ASC)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    elseif(DEFINED ENV{ASCEND_OPP_PATH})
        get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
    else()
        message(FATAL_ERROR "ASCEND_CANN_PACKAGE_PATH or ASCEND_HOME_PATH must be set")
    endif()
endif()

add_executable(ccu_all_gather_matmul
    ccu_all_gather_matmul_kernel.asc
)

target_include_directories(ccu_all_gather_matmul PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
)

target_compile_features(ccu_all_gather_matmul PRIVATE cxx_std_14)
target_compile_options(ccu_all_gather_matmul PRIVATE
    -Werror
    -fstack-protector-strong
    -fPIE
    -O2
)
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "CMAKE_ASC_ARCHITECTURES, e.g. dav-3510")
target_compile_options(ccu_all_gather_matmul PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_compile_definitions(ccu_all_gather_matmul PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)
target_link_options(ccu_all_gather_matmul PRIVATE
    -pie
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_link_directories(ccu_all_gather_matmul PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_link_libraries(ccu_all_gather_matmul PRIVATE
    hcomm
    asccomm_ccu_dataplane
    c_sec
    ascendcl
    acl_rt
    ascendc_runtime
)
```

> **对齐说明**：与参考样例 `02_allgather_add/CMakeLists.txt` 逐字节一致，仅 `demo`→`ccu_all_gather_matmul`、`main.asc`→`ccu_all_gather_matmul_kernel.asc`。include 路径 7 条、链接库 6 个、编译选项全部对齐。第二阶段引入 Blaze 时再追加 blaze/tensor_api include 和必要链接库，届时单独标注"第二阶段新增"。

---

## 9. Step 1 / Step 2 验收

### Step 1：跑通（不关注精度）
**验收标准**：2 rank 程序正常退出，不崩溃、不卡死。
- 关键观察点：
  - CCU kernel 是否执行（`HcclCommAssignCcuIns` 成功）
  - AIV 轮询是否退出（`ckeAddr1==1` 能读到，不死循环）
  - `aclrtSynchronizeStream` 不超时
- 输出可为任意值（全零也接受）

### Step 2：精度
**验收标准**：`computeBuf == [rank0 数据 | rank1 数据]`。
- rank0 sendBuf 全 1.0，rank1 sendBuf 全 2.0
- computeBuf 前 65536 元素 == 1.0，后 65536 元素 == 2.0
- `verify_result.py` 输出 `[PASS]`

---

## 10. 第一阶段风险点（具体）

| 风险 | 影响 | 缓解 |
|------|------|------|
| `HcommCcuGetMemToken` 对 FP16 sendBuf 地址/大小不匹配 | 通信数据错误 | 确认 inputSize 按字节传（= M*K*2） |
| `HCCL_OP_EXPANSION_MODE=CCU_SCHED` 环境变量未设 | CCU kernel 不执行 | run.sh 中 `export HCCL_OP_EXPANSION_MODE=CCU_SCHED` |
| AIV verify_copy 的 `count` 参数（元素数 vs 字节数） | DataCopy 量错误 | 确认 DataCopy 用元素数（recvCount=131072），非字节数 |
