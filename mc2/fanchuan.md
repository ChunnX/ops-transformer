### 1.1 算子定位与核心功能

Q: EngramFetch 要解决什么问题？

大规模 Engram 表（100B~1T params）无法全部放入 NPU HBM。需要一种机制让每个 rank 将自己的表分片放在 host pinned DRAM 中，其他 rank 通过 RDMA one-sided Get 按全局索引远程抓取指定行到本 rank 的 device HBM。

Q: EngramFetch 与 EngramWrite 的关系是什么？

| 操作 | 方向 | 媒介 | 是否有独立 ACLNN | 本质 |
|------|------|------|-----------------|------|
| EngramWrite | host → host pinned 注册段 | H→H memcpy（推理场景） | 否，host runtime | 数据放置（本 rank 写入共享段） |
| EngramFetch | 远端 host pinned → 本端 device HBM | RDMA one-sided Get | 是，issue + wait 双段 | 数据获取（跨 rank 远程读取） |

两者共用同一个 `aclElasticBufferHandle` 和拓扑参数。Write 必须先于 Fetch 执行。

Q: 为什么 EngramWrite 没有独立的 device kernel / ACLNN 接口？

1. 计算密度 = 0，纯内存拷贝
2. 无需 tiling 切分（一次 memcpy 即可完成）
3. `aclrtMemcpyAsync` + `HcclBarrier` 已经是公开 API，封装 ACLNN 无额外价值
4. EngramFetch 需要 AIV 发起 RDMA Get，必须有 kernel；EngramWrite 不需要

### 1.2 与 DeepEP 的对标关系

Q: 我们在实现什么？对标的基线是哪个版本？

对齐 DeepEP commit `a422b5a` 的 `engram_fetch_impl` + `engram_fetch_wait_impl` 双 kernel 流水线，将其从 NVIDIA NCCL Gin backend 移植到华为 HCCL one-sided RMA backend，运行在 Ascend NPU 上。

Q: NPU 与 DeepEP 存在哪些必然差异？

| # | 差异点 | DeepEP | NPU | 影响 |
|---|--------|--------|-----|------|
| a | 数据搬运介质 | `engram_write` 走 `cudaMemcpyDeviceToDevice`，假设表分片先在 device HBM | 推理场景表分片常驻 host，`engram_write` 是 H→H memcpy | write 实现简化，device 不参与 |
| b | 模板实例化 | NVRTC JIT 运行时实例化模板 | TilingKey 预编译多模板组合（Ascend C 无 host-trigger JIT） | 需要预注册 ~30 个常用 TilingKey 组合 |
| c | QP 参数暴露 | `engram_fetch(indices, num_qps)` 可逐次覆盖 QP 数 | `num_qps` 在构造时固定，fetch 不暴露该入参 | 减少 tiling 维度，QP 池构造期注册 |
| d | 通信 backend | NCCL Gin (`gin.get` / `gin.flush_async` / `gin.wait`) | HCCL one-sided RMA (`HcclOneSidedGet` / `HcclOneSidedFlushAsync` / `HcclOneSidedWait`) | API 映射需 spike 验证 |

### 1.3 调用场景

Q: EngramFetch 在模型中的位置？

```
EngramAddress (产出 addrs/indices)
    → EngramFetch (按 indices 远程抓取行)    ← 独立 prefetch stream
    → EngramFuse (消费 fetched)              ← 主干 compute stream
    → Attention / FFN 计算                   ← 与 RDMA 拖尾重叠
```

运行在独立 prefetch stream 上，issue kernel 提前一个 layer 调度，wait kernel 在 compute stream 上紧挨 EngramFuse，实现通信-计算重叠。

---

## 二、接口设计反串讲

### 2.1 ACLNN 接口

Q: 有几个 ACLNN 接口？各自的职责？

四段式，分为 issue 和 wait 两组：

```
┌─ Issue 组 ──────────────────────────────────────────┐
│ aclnnEngramFetchGetWorkspaceSize                    │
│   - 解包 bufferHandle，校验 indices / fetched       │
│   - 选择 TilingKey（六维模板参数）                   │
│   - 计算 workspace = last_gin_requests buffer       │
│ aclnnEngramFetch                                    │
│   - 启动 issue kernel (grid=num_qps, threads=1024)  │
│   - 解码 indices → 发起 HcclOneSidedGet             │
│   - per-peer flush → 写 last_req_gm                 │
└──────────────────────────────────────────────────────┘
┌─ Wait 组 ──────────────────────────────────────────┐
│ aclnnEngramFetchWaitGetWorkspaceSize                │
│   - 计算 wait workspace（通常很小或为 0）            │
│ aclnnEngramFetchWait                                │
│   - 启动 wait kernel (grid=num_qps, threads=1024)   │
│   - 扫 last_req_gm → 对非零项调 HcclOneSidedWait    │
└──────────────────────────────────────────────────────┘
```

Q: 为什么拆 issue / wait 两次 launch？

这是设计的关键决策：
1. issue 立即返回，RDMA Get 异步发起
2. 调用方可利用 RDMA 传输期间做其他计算（与 attention/FFN 重叠）
3. wait 延迟到真正需要数据时才阻塞

DeepEP 同款设计，对齐 `engram_fetch_impl` + `engram_fetch_wait_impl` 双 kernel 模式。

Q: ACLNN 接口的入参和出参？

| 接口 | 入参 | 出参 | 说明 |
|------|------|------|------|
| `aclnnEngramFetchGetWorkspaceSize` | `bufferHandle, indices, fetched` | `workspaceSize, executor` | workspace = `numQps × numRdmaPeers × 16B + 4KB` |
| `aclnnEngramFetch` | `workspace, workspaceSize, executor, stream` | 无 | 启动 issue kernel |
| `aclnnEngramFetchWaitGetWorkspaceSize` | `bufferHandle` | `workspaceSize, executor` | wait 的 workspace |
| `aclnnEngramFetchWait` | `workspace, workspaceSize, executor, stream` | 无 | 启动 wait kernel |

其中：
• `bufferHandle: aclElasticBufferHandle` — 不透明 host 指针，持有 host pinned 区段地址、对端 mem desc、QP 池、last_request GM 槽、拓扑参数

• `indices: aclTensor*` — `(num_tokens,)` int32，全局索引 ∈ `[0, world_size × num_entries_per_rank)`

• `fetched: aclTensor*` — `(num_tokens, hidden)` bf16，fetch 落地 buffer，位于 ElasticBuffer GPU 段


注意：无 `numQps` 入参。QP 数在 `ElasticBuffer.__init__(num_allocated_qps)` 时已固定。

### 2.2 PyTorch 扩展接口

Q: 用户如何调用？给出完整的使用范式。

```python
# 1) 计算内存需求
g_bytes, c_bytes = ElasticBuffer.get_engram_storage_size_hint(
    num_entries=524288, hidden=128, num_max_tokens=4096, dtype=torch.bfloat16
)

# 2) 构造 buffer（一次性分配 + 注册）
buffer = ElasticBuffer(
    group=group,
    num_bytes=g_bytes + c_bytes,
    num_cpu_bytes=c_bytes,
    num_allocated_qps=32,        # QP 数在此固定
    allow_hybrid_mode=True,
    explicitly_destroy=True,
)

# 3) 写入本 rank 表分片到 host pinned 共享段
buffer.engram_write(storage)     # storage: (num_entries, hidden) bf16

# 4) 远程抓取（issue + wait 分离）
hook = buffer.engram_fetch(indices)   # indices: (num_tokens,) int32
# ... 此处可做其他计算（与 RDMA 传输重叠）...
fetched = hook()                      # (num_tokens, hidden) bf16

# 5) 释放资源
buffer.destroy()
```

Q: `engram_write` 的详细执行流程？

```
engram_write(storage):
    ① HeavyBarrier(with_cpu_sync=True)
       ├─ aclrtSynchronizeDevice()     ← 等本 device 所有 stream 静止
       ├─ HcclBarrier(comm, stream)     ← 跨 rank barrier kernel
       └─ aclrtSynchronizeDevice()      ← CPU 阻塞至 barrier 完成
    ② memcpy(pinned_dst + cpu_offset, host_src, nbytes)  ← H→H
       cpu_offset = hybrid ? scaleup_rank_idx × num_cpu_buffer_bytes : 0
    ③ HeavyBarrier(with_cpu_sync=True)   ← 同 ①
```

pre-barrier 的作用：保证在飞的 fetch 已读完 pinned 段（写前不干扰正在进行的 RDMA 读取）
post-barrier 的作用：保证全网写完才放行后续 fetch（写后对端可见）

Q: 如果权重直接加载到了已注册的 pinned 段呢？

memcpy 可跳过（copy=0），engram_write 退化为仅前后两道 barrier。

### 2.3 接口约定核验

Q: 逐条校验 REQUIREMENTS.md §1 接口约定表的 6 条语义约束：

| # | DeepEP 约定 | NPU 实现 | 是否满足 | 验证点 |
|---|------------|---------|---------|--------|
| 1 | fetch 不带 `entries/hidden`，靠 write 时记录 | `aclElasticBufferHandle` 持有 `num_entries_per_rank / hidden` | ✅ | write 调用时写入句柄字段，fetch 从句柄读取 |
| 2 | fetch 返回 hook 闭包，issue 立即返回、wait 延迟 | `engram_fetch` 返回 hook；ACLNN 拆 issue + wait | ✅ | hook 闭包持有 last_gin_requests tensor |
| 3 | write 无独立 kernel，纯 memcpy + 两道 barrier | NPU 同，H→H memcpy + HeavyBarrier | ✅ | 无 ACLNN 接口，host runtime 完成 |
| 4 | 同 buffer issue 后必须 wait 才能再 issue | 同句柄不允许并发 fetch | ✅ | last_request 槽位会被覆盖 |
| 5 | storage 注册全在构造时一次性完成 | `__init__` 一次性 `aclrtMallocHost + HcclRegisterMem + 交换 desc` | ✅ | write/fetch 控制面成本 ≈ 0 |
| 6 | DeepEP 带 `num_qps` 可逐次覆盖 | NPU 不暴露 `num_qps`，构造时固定 | ✅ (设计取舍) | QP 池构造期注册，逐次变更无收益 |

---

## 三、设计实现反串讲

### 3.1 模板与 Tiling 体系

Q: 为什么需要模板划分？有哪些维度？

Ascend C 无法像 CUDA NVRTC 那样运行时 JIT 实例化模板，需要在编译期通过 TilingKey 预注册多个模板组合。

六维 TilingKey：

```
(NUM_QPS_TIER, HIDDEN_TIER, NUM_PEERS_TIER, RANKS_PER_PEER_TIER, ENTRIES_TIER, HYBRID_MODE)
   × {1,4,8,16,32}  × {64..4096}  × {1,2,4,8,16}  × {1,2,4,8}  × {2^12..2^20}  × {0,1}
```

理论上限 7000 组合，实际只需注册 ~30 个常用组合。

Q: 三种模板的触发条件？

| 模板 | 触发条件 | 核心差异 | 场景 |
|------|---------|---------|------|
| 模板一 (flat) | `HYBRID_MODE=0` | `numRdmaPeers = total_ranks`, `numRanksPerPeer = 1` | 单节点 / 小规模 |
| 模板二 (hybrid) | `HYBRID_MODE=1` | `numRdmaPeers = scaleout_ranks`, `numRanksPerPeer = scaleup_ranks` | 跨节点训练主流 |
| 模板三 (大 hidden) | `HIDDEN ≥ 1024` | 每 warp 降为 1 token | 100B 表 + 大 hidden 推理 |

Q: TilingData 结构体的字段及其含义？

```cpp
struct EngramFetchTilingData {
    uint32_t numTokens;          // indices 数量
    uint32_t numEntriesPerRank;  // E，每 rank 表行数
    uint32_t hiddenBytes;        // hidden × 2 (bf16)
    uint32_t numQps;             // QP 数
    uint32_t numRdmaPeers;       // RDMA peer 数 (hybrid=节点数, flat=总 rank 数)
    uint32_t numRanksPerPeer;    // 每 peer 内 rank 数 (hybrid=scaleup_ranks, flat=1)
    uint64_t intraRankStride;    // = num_cpu_buffer_bytes (节点内 rank 间存储步幅)
    uint64_t lastReqGmOffset;    // workspace 内偏移
    uint32_t entriesBarrettMul;  // Barrett 乘法常量（高效实现 g/E 和 g%E）
    uint32_t entriesBarrettShr;  // Barrett 移位常量
};
```

Q: 切分逻辑？

```
每个 AIV 实例化为 1 QP
块内 32 warp × 32 lane = 1024 thread
token 按 round-robin 分到各 warp:
  tokens_per_warp = ⌈num_tokens / (num_qps × 32)⌉
```

### 3.2 数学公式与索引路由

Q: 给出完整的索引路由公式？

对每个全局索引 `g = indices[i]`：

```
Step 1: 找到 owner rank 和 local entry
  owner_rank = g / E                    (E = num_entries_per_rank)
  local_entry = g % E

Step 2 (hybrid only): 分解为 peer 和 intra-node rank
  peer = owner_rank / R_su              (R_su = num_scaleup_ranks)
  intra_rank = owner_rank % R_su

Step 2 (flat):
  peer = owner_rank                     (flat 模式无分层)
  intra_rank = 0

Step 3: 计算源偏移
  src_offset = intra_rank × stride_intra + local_entry × hidden_bytes
  (stride_intra = num_cpu_buffer_bytes)

Step 4: RDMA Get
  fetched[i, :] ← RDMA_READ(peer_storage[peer] + src_offset, hidden_bytes)
```

Q: 为什么 Barrett reduction？为什么不在 kernel 里做普通除法？

`E = num_entries_per_rank` 是编译期常量（通过 TilingKey 的 ENTRIES_TIER 传入），Barrett 乘法 + 移位比普通整数除法快约 3 倍。这是高性能 kernel 的标准优化手法。

### 3.3 Kernel 执行流程

Q: Issue kernel 的详细步骤？

```
engram_fetch_template (grid=num_qps, block=1024):

Per AIV(QP):
  1. 初始化 sent_to_peer[NUM_PEERS] = {false}
  2. gin = NCCLGin(dev_comm, window, qp_idx)
  3. __syncthreads()

  // Issue RDMA Gets
  4. for each warp (elect_one_sync):
       for i in [global_warp_idx; i < num_tokens; i += QPs × Warps]:
         g = indices[i]
         owner = Barrett(g, E), local_e = g - owner * E
         peer = HYBRID ? owner / R_su : owner
         intra_rk = HYBRID ? owner % R_su : 0
         src_off = intra_rk * STRIDE + local_e * HIDDEN_BYTES
         HcclOneSidedGet(peer_storage[peer] + src_off,
                         fetched[i * HIDDEN_BYTES],
                         HIDDEN_BYTES, qp_idx, AGGREGATE)
         sent_to_peer[peer] = true

  5. __syncthreads()

  // Per-peer flush
  6. for each warp (elect_one_sync):
       for p in [warp_idx; p < NUM_PEERS; p += Warps]:
         req_ptr = last_req[qp_idx * NUM_PEERS + p]
         if sent_to_peer[p]:
           req = HcclOneSidedFlushAsync(qp_idx, p)
           DataCopyPad UB(req) → GM(req_ptr)
         else:
           DataCopyPad UB(0) → GM(req_ptr)
```

Q: Wait kernel 的详细步骤？

```
engram_fetch_wait_template (grid=num_qps, block=1024):

Per AIV(QP):
  for p = thread_idx; p < NUM_PEERS; p += 1024:
    req = DataCopyPad GM(last_req[qp_idx * NUM_PEERS + p]) → UB
    if req != 0 (任一字段非零):
      HcclOneSidedWait(req)
```

Q: `last_gin_requests` 的形状？

`last_gin_requests[num_qps × num_rdma_peers]`，每个元素 16B（HCCL request 结构体）。

• hybrid: `num_rdma_peers = num_scaleout_ranks`（仅节点数）

• flat: `num_rdma_peers = num_scaleout_ranks × num_scaleup_ranks`（总 rank 数）


### 3.4 EngramWrite 协同流程

Q: EngramWrite 的完整流程？

```
engram_write(storage):
  // storage: (num_entries, hidden) bf16, host memory

  ① HeavyBarrier(with_cpu_sync=True)
     ├─ aclrtSynchronizeDevice()     ← 排空 device stream
     ├─ HcclBarrier(comm, stream)     ← 跨 rank barrier
     └─ aclrtSynchronizeDevice()      ← 等 barrier 完成

  ② cpu_offset = hybrid ? scaleup_rank_idx × num_cpu_buffer_bytes : 0
     memcpy(host_buf_base + cpu_offset, storage.data_ptr(), nbytes)
     // H→H: 源与目的都在 host

  ③ HeavyBarrier(with_cpu_sync=True)  ← 同 ①
```

Q: cpu_offset 的含义？为什么 hybrid 模式下非零？

hybrid 模式下，同一节点内的多个 rank 共享同一块 host pinned 内存段。每个 rank 在该段中的写入位置 = `scaleup_rank_idx × num_cpu_buffer_bytes`，互不重叠。fetch kernel 中的 `src_off = intra_rank × stride_intra + ...` 与此严格对应。

Q: Write 与 Fetch 的耦合约束？

1. Write 必须先于第一次 Fetch — 构造时 ElasticBuffer 默认空 storage
2. Write 进行中不允许有未完成的 Fetch — pre HeavyBarrier 的 `aclrtSynchronizeDevice` 是兜底
3. `cpu_offset` 与 kernel 内 `intra_rk` 索引必须严格匹配 — 由句柄字段保证

### 3.5 内存布局

Q: ElasticBuffer 的内存布局？

```
[Workspace | GPU buffer (fetch landing) | CPU buffer (host pinned storage)]
│←────────────── num_sym_bytes ──────────────────→│←── num_cpu_bytes ──→│
│←── aligned 2MB ──→│←── num_gpu_buffer_bytes ──→│                      │
                     │←──────── num_buffer_bytes ──────────────────→│
```

• GPU buffer: fetch 落地区，大小 = `align(per_entry × num_max_tokens, 2MB)`

• CPU buffer: host pinned storage（注册到 HCCL 域），大小 = `align(per_entry × num_entries, 2MB)`

• Workspace 在最前面，2MB 对齐


Q: workspace 的计算？

```
workspace = num_qps × num_rdma_peers × 16B + 4096B(系统预留)
```

存放 `last_gin_requests` 数组，每对 (QP, peer) 一个 16B 的 request 槽。

Q: UB 容量是否够？

以 `tokens_per_warp=32, NUM_PEERS=8` 为例：

| 项 | 字节 |
|---|---|
| idx_buf | 32 × 4 = 128B |
| sent_to_peer | 8B |
| req_buf | 16B |
| 合计 | < 1KB |

DAV_3510 UB = 248KB，充裕。

---

## 四、约束与限制反串讲

### 4.1 数据类型与精度

Q: 支持哪些数据类型？精度要求是什么？

| 数据 | 类型 | 说明 |
|------|------|------|
| fetched / storage | bfloat16 | 首版仅支持 BF16，FP8 留后续迭代 |
| indices | int32 | 全局索引 |

精度要求：二进制一致（bit-exact / byte-identical）

纯数据搬运，无浮点运算、无累加、无类型转换。输出与源必须逐字节相等，零容差。

验证方式：`torch.equal` / `memcmp`（不是 `allclose`）。

### 4.2 并发与生命周期约束

Q: 列出所有约束条件？

| 约束 | 说明 |
|------|------|
| 同句柄不允许并发 fetch | `last_gin_requests` 会被覆盖，下次 issue 前必须 wait |
| Wait 与 fetch 需在同一 stream | 保证 last_request 内存可见性 |
| Write 必须先于 Fetch | 构造时默认空 storage |
| Write 进行中不允许未完成的 Fetch | pre HeavyBarrier 兜底 |
| `indices` 越界值未定义 | 调用方负责保证范围 |
| `num_tokens = 0` 合法 | 立即返回 |
| 各 rank 构造时拓扑参数必须一致 | `num_scaleout_ranks × num_scaleup_ranks` |

### 4.3 运行环境

Q: 目标硬件和架构？

• 服务器：Atlas A3 训练系列、Ascend 950PR/950DT

• 芯片：Ascend910_93 / Ascend910PR / Ascend950DT

• 架构：DAV_3510（首版）

• 网络：要求 RoCE/IB RDMA

• 首版仅支持 `team = Rail` (hybrid) 或 `team = World` (flat)


---

## 五、关键风险与待确认项

### 5.1 API 风险（必须 iter1 spike 闭环）

| # | 风险项 | 状态 | 影响 | 降级方案 |
|---|--------|------|------|---------|
| R1 | HCCL device 侧 one-sided RMA API 是否存在？ | ⚠️ 待验证 | 若不存在，无法在 AIV kernel 内触发 RDMA Get | 退到 host 触发 + AIV 仅准备 metadata，性能上限下降 |
| R2 | AggregateRequests 等价 flag 是否存在？ | ⚠️ 待验证 | 若不存在，每条 Get 自动 DB，DB 数从 `num_peers` 涨到 `num_tokens/num_qps` | 性能下降但功能完整 |
| R3 | HCCL request 结构体大小？ | ⚠️ 待验证 | 决定 `last_req_gm` 槽位字节数 | 需确认后调整 TilingData |
| R4 | `aclrtMemcpyAsync(H→H)` 写入注册段后远端能否直接读？ | ⚠️ 待验证 | 若需额外 flush 则增加延迟 | pre/post barrier 仍可兜底 |

Q: spike 的最小验证范围？

iter1 必须先做 "单 QP × 单 peer × 32 tokens" 的 spike，闭环 R1-R3。不闭环时立即切 host 触发路径。

### 5.2 设计风险

| 风险 | 说明 | 缓解 |
|------|------|------|
| TilingKey 组合数爆炸 | 理论上限 7000，实际需 ~30 | 按需注册，常见组合优先 |
| 同句柄不支持并发 fetch | DeepEP 同款约束 | 并发场景需多句柄/独立 QP 子集 |

---

## 六、性能预期

Q: 目标性能指标？

| 指标 | 目标 | 条件 |
|------|------|------|
| AIV 占用 | ≤ `numQps` 核 × launch 期，其余时间 0 | — |
| 带宽 | ≥ 70% 线速 | RoCE 400Gbps, 4096 tokens, hidden=1024 |
| 端到端延迟 | ≤ 50 µs | 4096 tokens, hidden=128, world=8 |
| 性能基线 | ≥ 70% DeepEP 等效拓扑时间 | 对齐 `test_engram.py` |

Q: 并行策略？

• QP 维度并行：grid = num_qps，每 AIV 独立

• Token 维度并行：每 QP 内 32 warp，round-robin 分 token

• Peer 维度串行：仅 flush 阶段，NUM_PEERS ≤ 16


Q: 流水线设计？

推荐 prefetch stream + compute stream 两条 stream：
• issue 在 prefetch stream（提前一个 layer）

• wait 在 compute stream（紧挨 EngramFuse）


---

## 七、迭代规划

| 迭代 | 目标 | 代码开发 | ST 用例 |
|------|------|---------|---------|
| iter1 | spike + 骨架 | flat mode、单 TilingKey、host 触发兜底；issue+wait 双 launch 走通；engram_write 全链路打通 | `world=2, tokens=4096, hidden=128`，对比 all_gather 基线 |
| iter2 | hybrid + 多 TilingKey | HYBRID_MODE=1 + NUM_PEERS 多档 | 4节点×8卡 hybrid + 单节点8卡 flat |
| iter3 | 性能 + 大规格 | 大 hidden 模板、device 触发 RDMA | 对齐 DeepEP ≥ 60% 带宽 |
| iter4 | 反向/sparse (可选) | 反向梯度索引去重 fetch | 与 EngramScatterAdd 联调 |
| iter5 | scale-up 快速路径 (可选) | 同节点 PCIe direct DMA | 同节点比例 50% 时延迟降 ~40% |

---

## 八、开发视角的疑问清单

> 以下为开发过程中需逐项确认的疑点，需在 iter1 spike 前闭环。

### 8.1 必须闭环（阻塞 iter1）

| # | 疑点 | 需确认人 | 对应文档位置 |
|---|------|---------|-------------|
| D1 | HCCL 是否暴露 device 侧 `HcclOneSidedGet` / `HcclOneSidedFlushAsync` / `HcclOneSidedWait`？ | HCCL 团队 | DESIGN §3.3.4, §5.1 |
| D2 | HCCL 是否支持 AggregateRequests 等价语义（多条 Get 共享一个 DB/doorbell）？ | HCCL 团队 | DESIGN §5.1 |
| D3 | HCCL request 结构体大小是多少？ | HCCL 团队 | DESIGN §3.3.7 |
| D4 | 写入 HCCL 注册段后，远端是否可直接 RDMA 读取，无需额外 flush？ | HCCL 团队 / 平台团队 | DESIGN §3.6, §5.1 |