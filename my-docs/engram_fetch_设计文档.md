# engram_fetch 算子设计文档（含全量本 NPU 模式 / fetch_mode）

> 覆盖算子：`engram_fetch`（推理 / 训练前向）、`engram_fetch_wait`（RDMA 排空）、`engram_fetch_grad`（训练反向）。
> 本文前半部分为算子背景与 LLM 使用场景（含既有 **切片跨 NPU 通信** 模式的完整设计），后半部分为本次新增
> **fetch_mode 全量本 NPU（replicated / local）模式** 的需求背景、方案设计与接口/实现细节。

---

## 目录

1. [LLM 上下文背景：什么是 engram，为什么需要 engram_fetch](#1-llm-上下文背景什么是-engram为什么需要-engram_fetch)
2. [整体架构与数据面](#2-整体架构与数据面)
3. [通信底座与上下文（EngramCommContext）](#3-通信底座与上下文engramcommcontext)
4. [engram_fetch 既有设计：切片跨 NPU 通信模式](#4-engram_fetch-既有设计切片跨-npu-通信模式fetch_mode0-默认)
5. [engram_fetch_wait / engram_fetch_grad](#5-engram_fetch_wait--engram_fetch_grad)
6. [新需求：arch35 全量本 NPU 模式（fetch_mode=1）](#6-新需求arch35-全量本-npu-模式fetch_mode1)
7. [local 模式方案设计](#7-local-模式方案设计)
8. [接口变更（aclnn / PTA）](#8-接口变更aclnn--pta)
9. [PTA（torch_extension）适配](#9-ptatorch_extension适配)
10. [边界与约束](#10-边界与约束)
11. [验证计划](#11-验证计划)

---

## 1. LLM 上下文背景：什么是 engram，为什么需要 engram_fetch

### 1.1 大模型推理/训练中的“权重不常驻”问题

LLM（大语言模型）的参数量已达数百 GB（如 DeepSeek-V3 约 671B 参数，BF16 下 1.3+ TB），远超单卡 HBM
容量（950 单卡数十 GB 量级）。业界主流解法之一是 **权重流式加载（weight streaming）**：

- 权重不全部常驻 NPU HBM，而是存放在 **host 侧 pinned memory**（或更大的主机内存层级）；
- 每步计算前，仅把 **本步实际需要的权重行** 按索引拉到 NPU 上参与计算；
- 计算与加载异步重叠，用“时间换空间”跑超大模型。

在 Ascend 的 mc2 算子族中，一条权重行（宽 `hiddenSize`、128 对齐）称为一个 **engram（记忆条目）**。
engram 全局按行编号（global entry index），构成全局索引空间。

### 1.2 engram_fetch 三件套在 LLM 数据面的位置

以 PD 分离 / 专家并行推理为典型场景：

```
 indices(本步需要的权重行号, int32)
     │
     ▼
 engram_fetch(with_grad=0)   ── 推理：RDMA ReadNbi 异步单边读，不阻塞后续算子下发
     │ (fetched 异步填充中)
     ▼
 <其它算子继续下发>           ── 通算/搬数重叠
     │
     ▼
 engram_fetch_wait           ── Drain 排空 RDMA 读通道，fetched 数据保证可见
     │
     ▼
 计算（GEMM 等）消费 fetched
```

训练场景（with_grad=1）则记录反向上下文，反向阶段由 `engram_fetch_grad` 把梯度沿原路回传并聚合：

```
 前向: engram_fetch(with_grad=1) → fetched + (perm, sendCounts, recvCounts, recvLocalEntry, numRecv)
 反向: engram_fetch_grad(gradFetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv)
       → (gradUnique, uniqueLocalEntry, numUnique)   # 去重 + FP32 原子加聚合，供优化器更新权重表
```

### 1.3 既有部署形态：全局表按 rank 切片

- 全局 engram 表（numEntriesGlobal 行）按 `ownerRank = index / numEntriesPerRank` 均匀切片，
  每个 rank 本地只持有 `numEntriesPerRank × hiddenSize` 的分片（存于 host pinned + 设备映射内存）；
- 任一 rank 的 indices 可能落在任意 owner rank 上 → **取数必须跨 NPU 通信**：
  - 推理：本 rank 对远端 rank 的注册内存做 RDMA 单边读（Urma `ReadNbi`，攒批 `Commit`）；
  - 训练：通信窗口 a2a（先交换索引请求，再回传权重行），反向再做一次逆向 a2a。

该形态内存最省（每卡仅 1/N 权重），但 **每次取数都有跨卡带宽与时延开销**，且依赖卡间 RDMA 链路。

---

## 2. 整体架构与数据面

```
┌─────────────────────────── 每 rank ───────────────────────────┐
│  host pinned mem (2MB 对齐, num_cpu_bytes)                    │
│   └─ engram 分片表 (fetch_mode=0) / 全量表 (fetch_mode=1)      │
│        │ aclrtHostRegisterV2(ACL_HOST_REG_MAPPED) + HcclCommMemReg
│        ▼                                                      │
│  device 映射地址 (commBuffer[rankId] / localStorageAddr)       │
│                                                               │
│  comm_context (int32 tensor, 由 PTA 生成下刷)                  │
│   └─ EngramCommContext{rankId, rankSize,                       │
│        commBuffer[1024]   // 各 rank 表基址(设备映射地址)        │
│        hcommHandle[1024]  // 各 rank RDMA 通道句柄              │
│        channelsPerRank}                                        │
└───────────────────────────────────────────────────────────────┘
```

- kernel 侧通过 `comm_context` 拿到 rankId/rankSize/各 rank 表基址/通道句柄；
- 推理 local 命中读 `commBuffer[rankId]`；训练 local 读 `localStorageAddr`（int64 标量张量，同为
  host pinned 的设备映射地址）。

## 3. 通信底座与上下文（EngramCommContext）

```cpp
struct EngramCommContext {          // mc2/engram_fetch/op_kernel/engram_fetch_utils.h
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBuffer[HCCL_MAX_RANK_SIZE];    // 各 rank 权重表基址
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE];   // 各 rank 通道句柄(按 rank*channelsPerRank + ch)
    uint32_t channelsPerRank;
};
```

- 通信原语：`Hcomm<COMM_PROTOCOL_UBC_CTP>`（Urma 统一 RDMA 接口）；
  推理 `ReadNbi/Commit`；训练 `WriteNbi/WriteWithNotifyNbi/Drain`；
- 通道获取（PTA csrc）：`HcclRankGraphGetLinksFunc` 找 UBC_CTP/UBC_TP/UBG 链路 →
  `HcclChannelAcquireFunc`（AIV engine）→ `HcclChannelGetRemoteMemsFunc` 换远端表基址；
- 训练 a2a 窗口：HCCL 默认 buffer（`HcclGetHcclBufferFunc`），窗口内分区见 §4.3。

## 4. engram_fetch 既有设计：切片跨 NPU 通信模式（fetch_mode=0，默认）

### 4.1 接口

```
inputs : comm_context(int32) / indices(int32, (T,)) / local_storage_addr(int64, (1,), opt, 训练用)
outputs: fetched(T, hidden, bf16/fp16/fp32)
         perm_out(T,) / send_counts_out(W*8,) / recv_counts_out(W,) /   ┐ 训练(with_grad=1)
         recv_local_entry_out(R_max,) / num_recv_out(1,)               ┘ 反向上下文
attrs  : hidden_size(req) / num_entries_per_rank(req) /
         num_max_tokens_per_rank(opt) / comm_buffer_size(opt) / with_grad(opt)
```

### 4.2 推理内核（DEFAULT_MODE，tiling key 0）

`ownerRank = index / numEntriesPerRank`，AIV 按 rank 分组（GetCoreAssignment）：

- **本地命中**（ownerRank == rankId）：`LocalCopySlice`（relayQue 双缓冲 32KB，GM→UB→GM）
  从 `commBuffer[rankId] + (index - rankId*numEntriesPerRank)*hiddenBytes` 拷到 `fetched`；
- **远端命中**：`hcomm_.ReadNbi(channelHandle, dst=fetched行, src=远端表行, hiddenBytes)`
  异步单边读，攒批 `Commit`（首次 4 次、其后每 16 次）；
- indices 超过 UB 批大小（按 bytesPerIndice≈21B 估算的 maxBatchSize）时分批。

### 4.3 训练内核（TRAIN_MODE，tiling key 1）

五阶段（核内 SyncAll 同步，窗口槽位 + 状态位 + 计数器三件套做流控）：

1. `CountAndScatterPhase`：统计每 owner rank 的 token 数 → sendCounts（32B 步长布局）；
2. `ComputeSdispls`：send 侧偏移表；
3. `GatherAndSortPhase`：按 rank 分组重排 indices → sortedIndices + perm（分组位置→原始位置）；
4. `SendCountPhase` + `ExchangeIndices`：a2a 交换索引请求（4 槽位窗口 + 读写计数器背压，
   `WriteWithNotifyNbi` 带计数通知，5s 超时保护）→ recvLocalEntry / recvCounts；
5. `LocalReadTable` + `ExchangeToken`：本 rank 查本地表得权重行，再 a2a 回传 → recvData；
   `ReorderAndSaveCtx`：按 perm 还原到 fetched 原始 token 序，写 numRecv。

窗口分区（commBufferSize 内）：countFlag/indicesFlag/tokenFlag/indicesW/indicesR/tokenW/tokenR/
sendCount 各 `numRanks*32B`，余量按 1:49 分给 indices 区与 token 区（各 rank 4 槽）。

## 5. engram_fetch_wait / engram_fetch_grad

- **wait**：对每个远端 rank 读通道执行 `Drain`，排空未完成的 RDMA 读，保证 fetched 可见后才能消费
  （纯同步，不碰数据本体；核分配与 fetch 推理内核一致）。
- **grad**（反向，tiling key 单一 DEFAULT）：
  1. `UnsortGrad`：按 perm 把 gradFetched 还原成分组序 → gradSorted；
  2. `ExchangeGrad`：逆向 a2a（sendCounts↔recvCounts 语义互换）把梯度行送回 owner rank → recvGrad；
  3. `UniqueScatterAdd`：LSD 基数排序（256 桶，maxValue = numRanks*numEntriesPerRank）+
     相邻去重 + `localEntry = entry - rankId*numEntriesPerRank` + FP32 原子加聚合 →
     (gradUnique(K,H), uniqueLocalEntry(K,), numUnique)。

## 6. 新需求：arch35 全量本 NPU 模式（fetch_mode=1）

### 6.1 需求描述

arch35（ascend950）上支持**将全局数据整表放到每个 NPU 上**：

- 之前：全局数据切片分发到不同 NPU，取任意数据需走跨 NPU 通信（RDMA / a2a）；
- 现在：**每个 NPU 上都有全局全量表**，任意 index 均在本卡本地，**取数完全不需要跨 NPU 通信**。

### 6.2 动机与适用场景

| 维度 | fetch_mode=0（切片，默认） | fetch_mode=1（全量本 NPU） |
|---|---|---|
| host 内存占用/卡 | 全表 1/N | 全表（N 倍） |
| 取数路径 | 本地命中走 GM 拷贝；远端命中走 RDMA 读（推理）/ a2a（训练） | **纯本地 GM 拷贝** |
| 是否需要卡间 RDMA 链路 | 是（context 构建即获取通道） | **否**（不获取任何通道） |
| 时延 | 受跨卡带宽/背压/Drain 影响 | 本地带宽，时延最低 |
| 适用 | 权重表大到单卡 host 放不下 | 表可整表冗余（如 KV/LoRA/小表），追求低时延 |

### 6.3 关键约束

- **默认行为不变**：不传新参数时仍走跨 NPU 通信（fetch_mode=0），完全后向兼容；
- `engram_write` 语义随模式变化：mode=1 时写入的是**全量表**（storage.size(0) = 全局 entry 数，
  此时 `num_entries_per_rank` attr 承载"本 rank 表行数 = 全局行数"）；
- 训练反向：grad 聚合的 `localEntry = entry - rankId*numEntriesPerRank` 在 mode=1 下必须改为
  `localEntry = entry`，排序 maxValue 相应收缩，否则索引换算错误。

## 7. local 模式方案设计

### 7.1 总体思路

新增算子属性 `fetch_mode`（int，0=切片跨 NPU（默认）/ 1=全量本 NPU），经 tiling key 派生
4 个 fetch 内核实例 + 2 个 grad 内核实例：

| with_grad | fetch_mode | tiling key（EngramFetchMode） | 内核 |
|---|---|---|---|
| 0 | 0 | 0 ENGRAM_FETCH_DEFAULT_MODE | EngramFetchArch35（不变） |
| 1 | 0 | 1 ENGRAM_FETCH_TRAIN_MODE | EngramFetchTrainArch35（不变） |
| 0 | 1 | 2 ENGRAM_FETCH_LOCAL_MODE | **EngramFetchLocalArch35（新）** |
| 1 | 1 | 3 ENGRAM_FETCH_TRAIN_LOCAL_MODE | **EngramFetchTrainLocalArch35（新）** |

grad：`EngramFetchGradMode` 0=DEFAULT（不变）/ 1=**LOCAL（新）**。

### 7.2 推理 local 内核（EngramFetchLocalArch35）

```
for batch in indices (UB 容量分批):
    copy batch → UB
    for i in batch (按 aivId 交错并行, tokenPos = aivId, aivId+totalBlocks, ...):
        entry = indices[i]                     // 直接就是本表行号（无 rank 偏移）
        if 0 <= entry < numEntries(=attr num_entries_per_rank):
            LocalCopySlice(fetched + t*hiddenBytes, commBuffer[rankId] + entry*hiddenBytes, hiddenBytes)
```

- 表基址 = `commBuffer[rankId]`（host pinned 的设备映射地址，注册路径不变）；
- 不 Init Hcomm、不读通道、不调用 wait（kernel 完成即数据可见）；
- 越界 entry 跳过（沿用训练内核的防御性风格）。

### 7.3 训练 local 内核（EngramFetchTrainLocalArch35）

产出与 TRAIN_MODE **完全相同的 6 个输出**（PTA/反向契约不变），但全部本地生成、零通信：

| 输出 | local 模式取值 | 说明 |
|---|---|---|
| fetched | `table[indices[t]]` | 纯本地 gather（表基址 localStorageAddr） |
| perm | 恒等置换 `perm[t]=t` | 分组序 == 原始 token 序 |
| sendCounts | `[me]=T`，其余 0（32B 步长布局） | 我的 token 全部"发给我自己" |
| recvCounts | `[me]=T`，其余 0 | 我只服务我自己的请求 |
| recvLocalEntry | `= indices`（token 序） | 收到的请求即本卡 indices |
| numRecv | `= T` | |

- gather/perm/recvLocalEntry 按 token 区间分片到各核并行（ArithProgression 生成恒等 perm，
  indices 批量 DataCopy 直写 recvLocalEntry）；
- counts 与 numRecv 由 0 号核统一写（先零填再置位）；
- 无窗口布局、无状态位、无计数器、无 SyncAll 依赖（各输出写者唯一）。

### 7.4 反向 grad local 分支

grad 收到上述"退化上下文"后：

1. **跳过** InitFlagsAndDispls / ClearWinCounters / CrossRankBarrier / UnsortGrad / ExchangeGrad
   （零窗口访问、零跨核 rank 交互）；
2. `recvGradGM_ := gradFetchedGM_`（梯度已在"接收序"= token 序，免拷贝）；
3. `UniqueScatterAdd` 复用：entries = recvLocalEntry(=indices)，companion 对齐 token 序；
   - 排序 maxValue：`numEntriesPerRank`（mode=1 下即全表行数），不再乘 rankSize；
   - **`localEntry = entry`**（去掉 `rankId*numEntriesPerRank` 偏移）。

> 正确性：mode=1 下每卡持全量表副本，本卡梯度只聚合本卡 token 覆盖的 entry，产出稀疏
> (gradUnique, uniqueLocalEntry)；跨卡一致性由上层优化器按副本归约（与 DP 梯度 all-reduce 同语义），
> 不在本算子范围内。

### 7.5 tiling / workspace 差异

- fetch local（推理/训练）：workspace 仅 `SYSTEM_NEED_WORKSPACE(16MB)`（无 a2a 区域）；
- 其余校验（dtype/shape/format/hidden 128 对齐等）与既有路径一致；
- `fetch_mode` 越界（∉{0,1}）报参数错；
- grad local：workspace 保留原计算（gradSorted/recvGrad 等区域闲置但无害，避免 tiling 双路径）。

## 8. 接口变更（aclnn / PTA）

### 8.1 aclnn（attr 顺序追加在末尾，保持既有参数不动）

```
aclnnEngramFetchGetWorkspaceSize(..., int64_t withGrad, int64_t fetchMode, uint64_t*, aclOpExecutor**)
aclnnEngramFetchGradGetWorkspaceSize(..., int64_t commBufferSize, int64_t fetchMode, uint64_t*, aclOpExecutor**)
```

### 8.2 属性定义

```
EngramFetch      .Attr("fetch_mode").AttrType(OPTIONAL).Int()   // 0 sliced(默认) / 1 local
EngramFetchGrad  .Attr("fetch_mode").AttrType(OPTIONAL).Int()
```

## 9. PTA（torch_extension）适配

`ElasticBuffer` 新增 `fetch_mode` 参数，**默认 0 = 跨 NPU 通信**：

```python
ElasticBuffer.FETCH_MODE_SLICED = 0   # 切片 + 跨 NPU 通信（默认，行为与历史版本完全一致）
ElasticBuffer.FETCH_MODE_LOCAL  = 1   # 全量表冗余到每个 NPU，取数纯本地

buf = ElasticBuffer(group, num_cpu_bytes=get_engram_storage_size_hint(num_entries_full, hidden),
                    fetch_mode=ElasticBuffer.FETCH_MODE_LOCAL)
buf.engram_write(full_table)          # mode=1 写入全量表
w = buf.engram_fetch(indices)()       # 纯本地 gather；mode=1 下 wait 被短路（无 RDMA 在途）
# 训练： with_grad=True 时反向 engram_fetch_grad 自动走 local 分支
```

- op schema：`engram_fetch(..., int dtype, int fetch_mode)`、
  `engram_fetch_train(..., int rank_size, int fetch_mode)`（PrivateUse1 + Meta 同步适配）；
- csrc：构造函数追加 `fetchMode`（默认 0，pybind 关键字默认值保持向后兼容）；
  - mode=1 时 context 构建**跳过通道获取与远端地址交换**（不依赖任何卡间 RDMA 链路），
    仅保留 host pinned 注册 + 本地基址；with_grad 仍查询 HCCL buffer size（attr 校验需要）；
  - `EngramWrite` 的 INT32 上限校验按模式区分（mode=1：行数 ≤ INT32_MAX）；
  - 推理 mode=1 的 wait 调用在 Python 层短路，不发 aclnnEngramFetchWait。

## 10. 边界与约束

| 项 | 约束 |
|---|---|
| hidden_size | 128 对齐（两模式一致） |
| indices | int32 1D，NPU 张量；mode=1 取值域 [0, 全局 entry 数) |
| 训练 num_max_tokens_per_rank | ≥ numTokens（local 模式沿用） |
| sendCounts 32B 步长 | fetch↔grad 接口契约，local 模式保持同一布局 |
| 平台 | 仅 ascend950（arch35）；内核需 ASC_DEVKIT>9.0 且 HCOMM>9.0 |
| rankSize | [2, 1024]（local 模式同样要求组内 ≥2 卡，走 HCCL 组构建） |
| 兼容性 | 不传 fetch_mode ⇔ fetch_mode=0，aclnn/PTA 行为与历史完全一致 |

## 11. 验证计划与结果

### 11.1 本机环境结论

- 本机为 A2（910B2）服务器：engram_fetch 为 950 专属算子，**无法本机运行功能验证**（执行器无法匹配 950 kernel 二进制，UT success 用例在本机因 SoC 不匹配失败，属环境限制、与改动无关——已用 origin/master 原始代码复现同样失败确认）；
- 编译验证环境：CANN 9.1.0（ASC_DEVKIT 9.1 / HCOMM 9.1）+ **asc-comm 开发补丁**（`~/workspace/cann/asc-comm` 的 `install.sh --full`，提供 `MakeBatchHandle/BatchCommit` 批量 Hcomm API，origin/master 的既有 engram kernel 即依赖该 API）叠加到独立副本 CANN 根（`~/tmp/cann950`，set_env.sh 已改写为副本自身路径），不影响原安装。

### 11.2 已完成的编译/UT 验证

| 项 | 命令 | 结果 |
|---|---|---|
| arch35 全量出包 | `bash build.sh --pkg --soc=ascend950 --module=mc2 --ops=engram_fetch,engram_fetch_grad,engram_fetch_wait -j16` | ✅ exit=0，`build_out/cann-ops-transformer-custom_linux-aarch64.run` 生成 |
| kernel 实例覆盖 | 二进制解析 | ✅ EngramFetch 每 dtype 二进制含 **4 个 kernel 实例**（_0.._3 = DEFAULT/TRAIN/LOCAL/TRAIN_LOCAL）；EngramFetchGrad 含 **2 个实例**（DEFAULT/LOCAL）；5 dtype（bf16/fp16/fp32/fp8e4m3/fp8e5m2）全覆盖 |
| op_api UT 编译+运行 | `bash build.sh --opapi_test --noexec ...` 后运行 `transformer_op_api_ut --gtest_filter='*Engram*'` | ✅ 9 项通过（nullptr 校验/execute_entry/**invalid fetch_mode 拒绝**）；6 项 success 用例因 910B 无 950 二进制失败（环境限制，上游同样失败） |
| PTA schema | python 导入 + `torch.library.define` | ✅ 4 个 schema（含新增 `int fetch_mode`）全部解析注册成功；`FETCH_MODE_SLICED/LOCAL` 常量可用 |

### 11.3 用户在 arch35（950）环境的功能验证建议

1. mode=1 推理：W 卡各写同一全量表 → fetch 随机 indices → 与 CPU golden 逐行比对（含 sf 路径）；
2. mode=1 训练：fetched 前向比对 + gradFetched 回传 → gradUnique/uniqueLocalEntry 与参考
   scatter-add（按 indices 去重求和）比对；perm 应为恒等、sendCounts/recvCounts 仅 [me]=T、numRecv=T；
3. 回归：mode=0（不传 fetch_mode）行为与基线一致；
4. 边界：空 indices、单 entry 表、indices 全部越界（应全跳过）、大表（行数接近 INT32_MAX）。

### 11.4 关键环境备注（950 编译复现方法）

```bash
# 1) 独立 CANN 副本(避免污染原安装) + asc-comm 批量 API 补丁
cp -a ~/Ascend/cann-9.1.0/cann-9.1.0 ~/tmp/cann950
sed -i 's|version_dirpath="<原绝对路径>"|version_dirpath="$HOME/tmp/cann950"|' ~/tmp/cann950/set_env.sh
~/workspace/cann/asc-comm/build/package/asc-comm/install.sh --full --install-path=$HOME/tmp/cann950
# 2) 编译(本机需 g++-12 dev 头文件, bisheng clang 依赖其 C++ 标准库头)
source ~/Ascend/cann-9.1.0/cann-9.1.0/set_env.sh
export ASCEND_HOME_PATH=$HOME/tmp/cann950   # 及 ASCEND_AICPU/OPP/TOOLKIT_HOME 同步指向副本
bash build.sh --pkg --soc=ascend950 --module=mc2 --ops=engram_fetch,engram_fetch_grad,engram_fetch_wait -j16
```
