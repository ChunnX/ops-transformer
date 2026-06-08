# FAKE_SEND_MODE 仿真适配说明

## 背景

本修改基于 PR #6382 的基础，目的是使 mega_moe 算子能够在**仿真软件**上运行，以采集各函数的运行速率。

仿真环境无法建立真实的 HCCL 通信域，因此引入 `FAKE_SEND_MODE` 机制，通过**虚假发送**和**虚假接收**的方式，将跨卡通信路径重定向到本地内存，使算子可以在单进程环境下完整执行。

## 修改概览

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `mega_moe_workspace_info.h` | C++ 头文件 | 新增 `FAKE_SEND_MODE` 宏定义及 fake win workspace 分配 |
| `mega_moe.h` | C++ 头文件 | 在 Init/SendInfoCal 等函数中适配虚假发送/接收逻辑 |
| `mega_moe.py` | Python 脚本 | 主测试脚本，构造满足虚假接收的 context |
| `mega_moe_only.py` | Python 脚本 | 单算子测试脚本，配合跑通算子 |
| `graph.py` | Python 脚本 | 图模型定义 |
| `torch_extension/.../mega_moe.py` | Python 接口 | `mega_moe()` 函数签名新增 `context` 参数直传 |

---

## 详细修改说明

### 1. mega_moe_workspace_info.h

#### 1.1 新增宏定义

```cpp
#define FAKE_SEND_MODE 1
```

#### 1.2 WorkspaceInfoWave 新增 fakeWinPtr 成员

```cpp
struct WorkspaceInfoWave {
    // ... 原有成员 ...
#if FAKE_SEND_MODE
    GM_ADDR fakeWinPtr;
#endif
```

#### 1.3 新增 fake win workspace 空间计算（逐行注释）

在 `WorkspaceInfoWave` 构造函数末尾追加一段 workspace，用于模拟所有 rank 的 win 区。
这块内存的布局必须与 `PeermemInfoWave` 构造函数中 `maskRecvPtr` / `quantTokenScalePtr` 的偏移计算**完全一致**，
因为 kernel 内部通过 `winRankAddr_[i] + offset` 访问时使用的是同一套偏移公式。

```cpp
#if FAKE_SEND_MODE
    // fakeWinPtr 指向 fake win 区的起始地址，紧接在正常 workspace 之后。
    // 仿真环境下所有 winRankAddr_[i] 都指向这块内存，替代真实的 epHcclBuffer。
    fakeWinPtr = base + workspaceSize;
    {
        // win 区前 60KB 是 rankSyncInWorld 同步区（PEERMEM_DATA_OFFSET = 60*1024），
        // 与 PeermemInfoWave 中 rankSyncInWorldPtr 的偏移一致。
        // 真正的数据从 PEERMEM_DATA_OFFSET 之后开始。
        int64_t winOffset = PEERMEM_DATA_OFFSET;

        // ---- mask 区大小计算 ----
        // 与 PeermemInfoWave::maskRecvPtr 之后的 mask 区完全对齐。
        // sendTotalNum = bs * topK，即本卡发出的总 token 数（每个 token 选 topK 个专家）。
        int64_t sendTotalNum = static_cast<int64_t>(tilingData->bs) * tilingData->topK;
        // compareCount: SendMaskCal 中 CompareScalar 处理的 int32 个数。
        // 需要将 sendTotalNum 个 int32 按 256B 对齐，再除以 sizeof(int32_t) 得到 int32 个数。
        // 对齐原因：CompareScalar 要求输入长度按 256B 对齐，尾部补 0 不影响结果。
        int64_t compareCount = Ops::Base::CeilAlign(sendTotalNum * static_cast<int64_t>(sizeof(int32_t)),
            static_cast<int64_t>(ALIGN_256)) / static_cast<int64_t>(sizeof(int32_t));
        // maskAlignSize: bit-packed mask 的字节数，按 32B 对齐。
        // compareCount 个 token 各占 1 bit，共 compareCount/8 字节，向上对齐到 32B。
        // 对齐原因：GatherMask 要求 mask 输入按 32B 对齐。
        int64_t maskAlignSize = Ops::Base::CeilAlign(compareCount / 8, static_cast<int64_t>(ALIGN_32));
        // maskSlotSize: 单个 [localExpert][srcRank] 槽位的总字节数。
        // = maskAlignSize（bit-packed mask）+ 32B（末尾存放 int32 count，源卡 SendMaskCal 同步算好）。
        // 32B 而非 4B 的原因：count 后面预留空间保证下一个槽位仍按 32B 对齐。
        int64_t maskSlotSize = maskAlignSize + static_cast<int64_t>(ALIGN_32);
        // mask 区总大小 = expertPerRank * epWorldSize 个槽位，按 512B 对齐。
        // expertPerRank 个本地专家，每个专家为 epWorldSize 个源卡各保留一个槽位。
        // 512B 对齐原因：保证后续 quantTokenScalePtr 仍 512B 对齐（CopyGMToGMPerToken 用普通 DataCopy 读）。
        winOffset += Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData->expertPerRank) * tilingData->epWorldSize * maskSlotSize,
            static_cast<int64_t>(ALIGN_512));

        // ---- quant token+scale 区大小计算 ----
        // 与 PeermemInfoWave::quantTokenScalePtr 的偏移计算一致。
        // QuantProcessInRank 将本卡 bs 个 token 量化后写入此区，供 CopyGMToGMPerToken 跨卡读取。
        // mxScaleNum: 每个 token 的 mxfp8 scale 个数 = CeilDiv(h, 32)，每 32 个元素共享一个 scale。
        uint32_t mxScaleNum = Ops::Base::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32));
        // dataBytes: 量化后 token 数据的字节数，按 256B 对齐（DataCopy 要求）。
        uint32_t dataBytes = Ops::Base::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_256)) * sizeof(int8_t);
        // scaleBytes: 每个 token 的 scale 字节数 = mxScaleNum * 1B（int8 scale）。
        uint32_t scaleBytes = mxScaleNum * sizeof(int8_t);
        // tokenBytes: 单个 token 的 data+scale 总字节数，按 32B 对齐，保证连续 token 间地址对齐。
        uint32_t tokenBytes = Ops::Base::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(ALIGN_32));
        // quant 区总大小 = bs 个 token，按 512B 对齐。
        winOffset += Ops::Base::CeilAlign(
            static_cast<int64_t>(tilingData->bs) * tokenBytes * sizeof(int8_t),
            static_cast<int64_t>(ALIGN_512));

        // fakeWinSize = PEERMEM_DATA_OFFSET + mask区 + quant区
        int64_t fakeWinSize = winOffset;
        workspaceSize += fakeWinSize;
    }
#endif
```

**fake win 区内存布局图**：

```
fakeWinPtr
├── [0, PEERMEM_DATA_OFFSET)          rankSyncInWorld 同步区（60KB，仿真下不使用）
├── [PEERMEM_DATA_OFFSET, +mask区)    mask 接收区
│   ├── [expert0][srcRank0]           maskSlotSize 字节: [bit-packed mask | count(4B)]
│   ├── [expert0][srcRank1]
│   ├── ...
│   ├── [expert0][srcRank_{ws-1}]
│   ├── [expert1][srcRank0]
│   ├── ...
│   └── [expert_{epr-1}][srcRank_{ws-1}]
│   总大小 = expertPerRank * epWorldSize * maskSlotSize，512B 对齐
├── [+mask区, +quant区)               量化 token+scale 区
│   ├── token0: [dataBytes | scaleBytes]  tokenBytes 字节
│   ├── token1: [dataBytes | scaleBytes]
│   ├── ...
│   └── token_{bs-1}
│   总大小 = bs * tokenBytes，512B 对齐
└── 结束
```

### 2. mega_moe.h

#### 2.1 新增成员变量

```cpp
#if FAKE_SEND_MODE
    GM_ADDR fakeMaskAddr_[1024];
#endif
```

`fakeMaskAddr_[le]` 存储第 `le` 个本地专家的虚假 mask 数据地址，在 Init 阶段从 `mc2Context_->epHcclBuffer[le]` 获取。

#### 2.2 Init() 函数适配

**rankId_ 写死为 0**（仿真只模拟单卡视角）：

```cpp
#if FAKE_SEND_MODE
    rankId_ = 0;
#else
    rankId_ = mc2Context_->epRankId;
#endif
```

**winRankAddr_ 全部指向 fakeWinPtr**（虚假发送：所有"远端卡"的 win 都是同一块本地内存）：

```cpp
#if FAKE_SEND_MODE
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = params_.workspaceInfo.fakeWinPtr;
    }
#else
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
#endif
```

**fakeMaskAddr_ 从 epHcclBuffer 取地址**（虚假接收：Python 脚本预构造的 mask 数据通过 epHcclBuffer 传入）：

```cpp
#if FAKE_SEND_MODE
    for (uint32_t le = 0; le < expertPerRank_; le++) {
        fakeMaskAddr_[le] = reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer[le]);
    }
#endif
```

#### 2.3 SendInfoCal() 虚假接收

从 `fakeMaskAddr_[localExpertId]` 读取预构造的 mask 数据，而非真实的跨卡 win 区：

```cpp
GlobalTensor<uint8_t> maskSrcGlobal;
#if FAKE_SEND_MODE
    maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(fakeMaskAddr_[localExpertId]));
#else
    maskSrcGlobal.SetGlobalBuffer((__gm__ uint8_t*)(params_.peermemInfo.maskRecvPtr +
        static_cast<uint64_t>(localExpertId) * worldSize_ * maskSlotSize_));
#endif
```

---

### 3. 虚假接收的数据格式与脚本构造方法

#### 3.1 C++ 侧期望的数据格式

`SendInfoCal` 从 `fakeMaskAddr_[localExpertId]` 读取 `worldSize_ * maskSlotSize_` 字节，按以下布局解析：

```
fakeMaskAddr_[localExpertId] 指向的内存（单个本地专家）：

偏移 0:                        [srcRank 0 的槽位]
  ├── [0, maskAlignSize)       bit-packed mask（compareCount 个 bit，每 bit 表示对应 token 是否发给本专家）
  └── [maskAlignSize, +32)     int32 count（该 srcRank 发给本专家的 token 总数）

偏移 maskSlotSize:             [srcRank 1 的槽位]
  ├── [0, maskAlignSize)       bit-packed mask
  └── [maskAlignSize, +32)     int32 count

...

偏移 (worldSize-1)*maskSlotSize: [srcRank worldSize-1 的槽位]
```

其中：
- `sendTotalNum = bs * topK`（本卡发出的总 token 数）
- `compareCount = CeilAlign(sendTotalNum * 4, 256) / 4`（256B 对齐后的 int32 个数）
- `maskAlignSize = CeilAlign(compareCount / 8, 32)`（bit-packed mask 字节数，32B 对齐）
- `maskSlotSize = maskAlignSize + 32`（单个槽位总大小）
- bit-packed mask 的第 `i` 位 = 1 表示 `topk_ids[srcRank][i / topK][i % topK] == localExpertId`
- count 为小端序 int32，值 = 该 srcRank 发给本专家的 token 数

#### 3.2 Python 脚本构造方法

**Step 1: `compute_fake_mask_data()` — 为每个本地专家构造 mask buffer**

```python
def compute_fake_mask_data(expert_ids_per_rank, bs, topK, expert_per_rank, world_size):
    send_total_num = bs * topK
    # 与 C++ 侧 compareCount 公式一致
    compare_count = ((send_total_num * 4 + 255) // 256 * 256) // 4
    # 与 C++ 侧 maskAlignSize 公式一致
    mask_align_size = ((compare_count // 8 + 31) // 32) * 32
    # 与 C++ 侧 maskSlotSize 公式一致
    mask_slot_size = mask_align_size + 32

    mask_buffers = []
    for le in range(expert_per_rank):
        # 每个专家一个 buffer，包含 world_size 个槽位
        buf = torch.zeros(world_size * mask_slot_size, dtype=torch.uint8).npu()
        for src_rank in range(world_size):
            # 取该 src_rank 的 topk_ids，展平为 [send_total_num] 数组
            topk_flat = expert_ids_per_rank[src_rank].flatten().cpu().numpy()
            # 逐位比较：该 token 是否发给专家 le
            mask_bool = (topk_flat == le).astype(np.uint8)
            # bit-pack：第 i 位 = mask_bool[i]
            mask_bytes = bytearray(mask_align_size)
            for i in range(send_total_num):
                if mask_bool[i]:
                    mask_bytes[i // 8] |= (1 << (i % 8))
            # count = 该 src_rank 发给专家 le 的 token 数
            count = int(mask_bool.sum())

            # 写入槽位：[mask_bytes | count(小端序 int32)]
            offset = src_rank * mask_slot_size
            buf[offset:offset + mask_align_size] = torch.tensor(
                list(mask_bytes), dtype=torch.uint8)
            count_bytes = struct.pack('<i', count)
            buf[offset + mask_align_size:offset + mask_align_size + 4] = torch.tensor(
                list(count_bytes), dtype=torch.uint8)

        mask_buffers.append(buf)  # mask_buffers[le] 在 NPU 上
    return mask_buffers
```

**Step 2: `build_fake_context()` — 将 mask buffer 地址填入 Mc2MoeContext**

```python
def build_fake_context(ep_world_size, expert_per_rank, mask_buffers):
    # Mc2MoeContext 结构体大小（int32 为单位）：
    #   epRankId(4B) + epRankSize(4B) + winSize(8B) + epHcclBuffer[1024](8192B) = 8208B = 2052 int32
    ctx_int32_count = (4 + 4 + 8 + 1024 * 8) // 4
    ctx_cpu = torch.zeros(ctx_int32_count, dtype=torch.int32)

    # epRankId = 0（仿真固定 rank 0）
    ctx_cpu[0] = 0
    # epRankSize = ep_world_size
    ctx_cpu[1] = ep_world_size

    # epHcclBuffer[le] 存储 mask_buffers[le] 的 NPU 地址（拆为低32位+高32位）
    # C++ 侧通过 mc2Context_->epHcclBuffer[le] 读取，赋值给 fakeMaskAddr_[le]
    for le in range(expert_per_rank):
        addr = mask_buffers[le].data_ptr()
        base = 4 + le * 2  # epHcclBuffer 起始偏移（int32 索引 4 起），每个地址占 2 个 int32
        ctx_cpu[base] = addr & 0xFFFFFFFF          # 低 32 位
        ctx_cpu[base + 1] = (addr >> 32) & 0xFFFFFFFF  # 高 32 位

    return ctx_cpu.npu()
```

**数据流总结**：

```
Python 脚本                                        C++ kernel
─────────────                                      ──────────
expert_ids_per_rank[src_rank]                      
    │                                              
    ▼                                              
compute_fake_mask_data()                           
    │ 为每个 le 构造 mask_buffer                    
    │ (bit-packed mask + count per src_rank)        
    ▼                                              
mask_buffers[le] → .npu() → NPU 内存               
    │                                              
    ▼                                              
build_fake_context()                               
    │ mask_buffers[le].data_ptr()                  
    │ 写入 ctx_cpu.epHcclBuffer[le]                 
    ▼                                              
ctx_cpu.npu() → fake_context                       
    │                                              
    ▼ 传入 mega_moe(context=fake_context, ...)     
                                                   Init():
                                                     fakeMaskAddr_[le] = epHcclBuffer[le]
                                                        ↓
                                                   SendInfoCal(localExpertId):
                                                     DataCopy(gatherMaskTensor_,
                                                       fakeMaskAddr_[localExpertId],
                                                       worldSize * maskSlotSize)
                                                        ↓
                                                   解析 [mask|count] → cumsum → dispatch
```

---

### 4. 单卡执行与多进程规避

#### 4.1 只起 rank 0，不启动多进程

原始脚本（`mega_moe_with_cascade.py` 等）使用 `torch.multiprocessing.Process` 为每个 rank 启动独立进程：

```python
# 原始多进程方式（仿真不可用）
for rank in rank_list:
    proc = Process(target=run_cascade_npu, kwargs={"queue": ..., "rank": rank, ...})
    proc.start()
```

`mega_moe_only.py` 改为**直接在主进程中调用 rank 0**：

```python
# FAKE_SEND_MODE：单进程，只跑 rank 0
print("FAKE_SEND_MODE: single rank only")
rank_output = run_cascade_npu(0, input_data)
```

#### 4.2 为什么不能起多进程

1. **仿真软件不支持多进程**：仿真环境只模拟单张 NPU 卡的行为，多进程会尝试创建多个 device context，导致仿真器报错或行为异常。
2. **无法建立 HCCL 通信域**：`dist.init_process_group(backend="hccl")` 需要真实的 NPU 设备间通信链路，仿真环境下不存在。FAKE_SEND_MODE 通过虚假发送/接收完全绕过了 HCCL 依赖。
3. **`run_cascade_npu` 内部不再调用 `init_hccl_comm_two_ep`**：FAKE_SEND_MODE 下所有跨卡通信都被重定向到本地内存，因此不需要初始化任何通信域。

#### 4.3 `run_cascade_npu` 的适配

`run_cascade_npu(rank=0, input_data)` 内部流程：

```python
def run_cascade_npu(rank, input_data):
    set_device(rank)  # 设置 device 0

    # 1. 构造虚假 mask 数据（所有 rank 的 topk 信息，在 Python 侧预计算）
    mask_buffers = compute_fake_mask_data(
        input_data.expert_ids, bs, k, local_moe_expert_num, ep_world_size)

    # 2. 构造虚假 context（将 mask buffer 地址填入 Mc2MoeContext）
    fake_context = build_fake_context(ep_world_size, local_moe_expert_num, mask_buffers)

    # 3. 构造 FakeSymBuffer（替代真实的 symm_buffer，不依赖 HCCL）
    distribute_buffer = FakeSymBuffer(...)

    # 4. 直接调用算子（无 HCCL 初始化、无多进程同步）
    mega_moe_model = MoeDispatchFFNCombineGraphModel(...)
    y, expert_token_nums = mega_moe_model.forward(**mega_moe_kwargs)
```

#### 4.4 torch_extension 接口改动

`torch_extension/npu_ops_transformer/ops/mega_moe.py` 中 `mega_moe()` 函数签名变更：

```python
def mega_moe(
    context: torch.Tensor,  # 新增：直接传入构造好的 context（含 epHcclBuffer）
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    # ... 其他参数
):
    return torch.ops.npu_ops_transformer.npu_mega_moe(
        context,  # 不再从 sym_buffer.context 取，由 Python 脚本直传
        x,
        # ...
    )
```

---

## 运行机制

```
┌──────────────────────────────────────────────────────────────────┐
│                    FAKE_SEND_MODE 运行流程                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Python 侧（主进程，rank 0 only）                                 │
│  ├── 1. compute_fake_mask_data()                                 │
│  │   └── 为每个本地专家构造 [mask|count] 槽位数据                  │
│  │       布局: expertPerRank 个 buffer，每个含 worldSize 个槽位    │
│  ├── 2. build_fake_context()                                     │
│  │   └── 将 mask buffer 的 NPU 地址填入 Mc2MoeContext.epHcclBuffer│
│  ├── 3. FakeSymBuffer 替代真实 symm_buffer                       │
│  └── 4. 调用 mega_moe(context=fake_context, ...)                 │
│                                                                  │
│  C++ Kernel 侧（仿真器执行）                                      │
│  ├── Init()                                                      │
│  │   ├── rankId_ = 0                                             │
│  │   ├── winRankAddr_[i] → fakeWinPtr（虚假发送目标）             │
│  │   └── fakeMaskAddr_[le] → epHcclBuffer[le]（虚假接收源）       │
│  │                                                               │
│  ├── SendMaskCal()                                               │
│  │   └── 计算 mask 并推送到 fakeWinPtr（本地内存，非真实跨卡）     │
│  │                                                               │
│  ├── CrossRankSyncInWorldSize()                                  │
│  │   └── 跨卡信号握手（仿真下写本地内存，不会真正阻塞）            │
│  │                                                               │
│  ├── SendInfoCal(localExpertId)                                  │
│  │   └── 从 fakeMaskAddr_[localExpertId] 读取预构造的 mask        │
│  │       解析 count → cumsum → 通知 AIC                          │
│  │                                                               │
│  ├── TripleInfoCalAndDispatch()                                  │
│  │   └── GatherMask + CopyGMToGMPerToken（正常执行）              │
│  │                                                               │
│  └── GroupMatmul / Unpermute 等后续流程正常执行                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## 使用方式

1. 在 `mega_moe_workspace_info.h` 中设置 `#define FAKE_SEND_MODE 1`
2. Python 脚本构造满足布局要求的 `context`（包含所有 token 的 mask 信息）
3. 调用 `mega_moe(context, ...)` 运行算子（单进程，rank 0）
4. 通过仿真软件采集各函数运行速率

## 注意事项

- `FAKE_SEND_MODE` 仅用于仿真调试，**不可用于生产环境**
- `mega_moe_with_cascade.py` 为废弃文件，可忽略
- 虚假接收要求 Python 脚本预构造的 mask 数据布局与 `PeermemInfoWave` 中 `maskRecvPtr` 完全一致
- 仿真环境只支持单进程单卡，禁止使用 `multiprocessing.Process` 启动多进程
- `run_cascade_npu` 内部不再调用 `init_hccl_comm_two_ep`，完全绕过 HCCL 初始化
