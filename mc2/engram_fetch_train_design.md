# EngramFetch 训练化综合设计文档

> 需求文档：`mc2/requriment.md` v4.0
> 推理基线：`mc2/engram_fetch_算子源码详解.md`、`mc2/ElasticBuffer_详解.md`
> 代码基线：PR 8541 `refactor: unify elastic buffer comm context handling`（已合并）

---

## 一、需求概述

### 1.1 核心原则

1. **前向不新造算子**：复用推理 `EngramFetch`（`aclnnEngramFetch` issue + `aclnnEngramFetchWait` wait）
2. **`with_grad` 在 `ElasticBuffer` 构造函数**：不在算子 attr 上
3. **save-for-backward 作为可选输出**：推理传 nullptr，训练态非空
4. **训练态内部走 all-to-all**：桶排序→a2a→gather→a2a→还原，融合在 issue/wait 内
5. **反向 `EngramFetchGrad` 是新算子**：单融合 kernel，3 步

### 1.2 推理 vs 训练对比

| | `with_grad=False`（推理） | `with_grad=True`（训练） |
|---|---|---|
| 内部数据路径 | one-sided GET 拉行（RDMA read） | 两侧 all-to-all（桶排序→a2a→gather→a2a→还原） |
| save-for-backward 可选输出 | 传 nullptr / 不产出 | 产出 `perm/sendCounts/recvCounts/recvLocalEntry/numRecv` |
| `fetched` 是否可反向 | 否 | 是（经 `EngramFetchGrad`） |
| GM（设备显存） | 无 | 每次 fetch 动态申请/释放（a2a 收发缓冲区） |
| wait | callable 内调 `aclnnEngramFetchWait` | 接口内调 `aclnnEngramFetchWait` |

### 1.3 数据流

#### 前向训练态（融合在 issue/wait 内）

```
indices[T]
   │
   ▼ ① 桶排序（issue kernel 内）
   │  owner = idx // E,  local = idx % E
   │  稳定排序 → perm[T], sendCounts[W], sortedLocalEntry[T]
   │
   ▼ ② device 触发 a2a(sortedLocalEntry)（issue kernel 内）
   │  → recvLocalEntry[R]    R = 本 rank 被全网请求行数
   │  → recvCounts[W]
   │
   ▼ ③ 本地 gather（issue kernel 内）
   │  storage[recvLocalEntry] → recvTokens(R, H)
   │
   ▼ ④ device 触发 a2a(recvTokens)（issue kernel 内）
   │  → gathered(T, H)  sorted 序
   │
   ▼ ⑤ 还原（wait kernel 内）
   │  fetched[perm] = gathered
   │
   ├─→ fetched (T, H) bf16
   └─→ save-for-backward: perm, sendCounts, recvCounts, recvLocalEntry, numRecv
```

#### 反向 EngramFetchGrad（新算子，3 步）

```
gradFetched(T,H) + perm/sendCounts/recvCounts/recvLocalEntry（前向 save-for-backward）
   │
   ▼ ① 按 perm 重排：grad_sorted = gradFetched[perm]
   │
   ▼ ② device 触发 a2a(grad_sorted; sendCounts↔recvCounts) → recvGrad(R,H)
   │
   ▼ ③ owner unique + fp32 scatter-add：对 (recvLocalEntry, recvGrad) 聚合
   │
   ├─→ gradUniqueOut (R_max, H) fp32     前 numUnique 有效
   ├─→ uniqueLocalEntryOut (R_max,) int32  1D sparse index
   └─→ numUniqueOut (1,) int32
```

---

## 二、通信域设计

### 2.1 PR 8541 架构基线

PR 8541 将通信域构建重构为 `EngramContextBuilder` 独立类 + 懒初始化：

```
ElasticBuffer 构造 (with_grad=True)
  └─ 首次 engram_write/fetch/barrier 时
       └─ EnsureEngramContext()
            └─ EngramContextBuilder.Build(groupName, numCpuBytes, withGrad)
                 ├─ AcquireHcclHandle
                 ├─ AllocateAndRegisterBuffer (host pinned)
                 ├─ BuildChannelDescs (HcclChannelDescInit, withGrad ? 2 : 1)
                 ├─ HcclChannelAcquire → hcommHandle[peer]
                 ├─ HcclChannelGetRemoteMems → virtualAddrList[peer]
                 └─ HcclEngineCtxCopy → contextTensor_
```

### 2.2 EngramCommContext（`engram_fetch.h`，kernel 侧）

```cpp
constexpr uint32_t MAX_QP_SIZE = 1024;
struct EngramCommContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBuffer[MAX_QP_SIZE];       // 各 rank host pinned 设备映射地址（推理+训练共用）
    uint64_t hcommHandle_[MAX_QP_SIZE];     // 各 rank channel 句柄（推理+训练共用）
    uint64_t gmCommBuffer[MAX_QP_SIZE];     // 各 rank GM 缓冲区地址（仅 with_grad a2a 使用）
};
```

### 2.3 CommContext（`elastic_buffer.cpp`，host 侧镜像）

```cpp
struct CommContext {
    uint32_t epRankId = 0;
    uint32_t rankSize = 0;
    uint64_t virtualAddrList[HCCL_MAX_RANK_SIZE] = {};   // → commBuffer
    uint64_t hcommHandle[HCCL_MAX_RANK_SIZE] = {};       // → hcommHandle_
    uint64_t gmAddrList[HCCL_MAX_RANK_SIZE] = {};        // → gmCommBuffer
};
```

### 2.4 channel memHandle 槽位

```cpp
HcclChannelDescInit(&desc, withGrad ? 2 : 1);
// withGrad=true: 预留 2 槽位（host pinned + GM），构造时 memHandleNum=1
// withGrad=false: 1 槽位（host pinned only），推理不受影响
```

GM 注册后通过 `HcclCommMemReg` 对通信域内所有 channel 自动可见，`HcclChannelGetRemoteMems` 可查到，无需重新 acquire。

### 2.5 GM 动态申请/释放（per-fetch）

GM 不在 builder 中预分配（builder 是 per-ElasticBuffer 生命周期），而是 per-fetch 动态管理：

#### AllocateAndExchangeGm

```
1. aclrtMalloc(HUGE_FIRST) → gmBufPtr_
2. HcclCommMemReg(tag="_gm_buffer_seqN") → gmMemHandle_
3. EngramBarrier(true)                     ← 所有 rank 注册完
4. HcclChannelGetRemoteMems → gmAddrList[peer]
5. UpdateContextTensor()                   ← 刷 contextTensor_ 的 gmCommBuffer
6. EngramBarrier(true)                     ← 所有 rank context 更新完
```

关键设计：
- **大页申请**：`ACL_MEM_MALLOC_HUGE_FIRST`
- **tag 带序号**：`static std::atomic<int64_t> gmSeq`，避免 HCCL tag 缓存冲突
- **两次 barrier**：保证所有 rank 注册完 + context 更新完

#### FreeGm

```
1. EngramBarrier(true)                     ← 等 kernel 完成
2. aclrtFree(gmBufPtr_)
3. 清空 gmAddrList
```

#### GM 大小计算

```cpp
// 前向：bf16 载荷
int64_t maxR = numTokens * rankSize;  // 保守上界
int64_t gmBytes = AlignTo(maxR * hiddenBytes, BUFFER_ALIGNMENT);

// 反向：fp32 聚合
int64_t gmBytes = AlignTo(maxR * fp32_hiddenBytes, BUFFER_ALIGNMENT);
```

---

## 三、Python 层设计

### 3.1 EngramFetchCtx

```python
@dataclass
class EngramFetchCtx:
    """训练态前向的 save-for-backward，传给 engram_fetch_grad"""
    perm: torch.Tensor            # (T,) int32 — 桶排列序索引
    sendCounts: torch.Tensor      # (W,) int32 — 每 rank 发送的 index 数
    recvCounts: torch.Tensor      # (W,) int32 — 每 rank 接收的 index 数
    recvLocalEntry: torch.Tensor  # (R_max,) int32 — 收到的 local entry
    numRecv: torch.Tensor         # (1,) int32 — 实际接收行数 R
```

### 3.2 ElasticBuffer 构造函数

```python
class ElasticBuffer:
    def __init__(self, group, *, num_cpu_bytes=0, with_grad=False, ...):
        self._with_grad = with_grad
        self._runtime = _elastic_buffer_ops.ElasticBuffer(
            self._group_name, self._num_cpu_bytes, self._with_grad)
```

### 3.3 engram_fetch

```python
def engram_fetch(self, indices: torch.Tensor) -> Callable:
    """
    推理/训练共用。返回 hook callable。
    - 推理(with_grad=False): hook() → fetched (T,H) bf16
    - 训练(with_grad=True):  hook() → (fetched, fetch_ctx)
    """
    # 推理: return self._runtime.engram_fetch(indices)
    # 训练: 分配 5 个 save-for-backward tensor → 调 C++ → 返回 train_hook
```

### 3.4 engram_fetch_grad

```python
def engram_fetch_grad(self, grad_fetched: torch.Tensor,
                          fetch_ctx: EngramFetchCtx) -> Tuple[Tensor, Tensor]:
    """
    返回 (grad_unique (K,H) fp32, unique_local_entry (K,) int32)
    grad_unique: 累加后的稀疏梯度
    unique_local_entry: 1D sparse index，标识梯度对应哪行 embedding
    """
    return self._runtime.engram_fetch_grad(
        grad_fetched, fetch_ctx.perm, fetch_ctx.sendCounts,
        fetch_ctx.recvCounts, fetch_ctx.recvLocalEntry)
```

### 3.5 用户使用示例

```python
# 构造（训练模式）
buf = ElasticBuffer(group, num_cpu_bytes=..., with_grad=True)
buf.engram_write(storage)

# 前向
hook = buf.engram_fetch(indices)
fetched, fetch_ctx = hook()              # (fetched (T,H), EngramFetchCtx)

# ... 使用 fetched 计算 loss, backward ...
grad_fetched = ...                       # 对 fetched 的梯度

# 反向
grad_unique, unique_local_entry = buf.engram_fetch_grad(grad_fetched, fetch_ctx)
# grad_unique (K,H) fp32 + unique_local_entry (K,) → 优化器更新 storage[unique_local_entry]
```

---

## 四、C++ 接口层设计（`elastic_buffer.cpp`）

### 4.1 EngramContextBuilder

```cpp
class EngramContextBuilder {
public:
    EngramContextResources Build(const std::string &groupName, int64_t numCpuBytes, bool withGrad = false);
private:
    static void BuildChannelDescs(..., bool withGrad, ...);  // HcclChannelDescInit(&desc, withGrad ? 2 : 1)
    static void GetHcclCommChannel(..., bool withGrad, ...);
    static void GetHcclCommResource(..., bool withGrad, ...);
    static void CreateContext(..., bool withGrad, ...);
};
```

### 4.2 ElasticBuffer 类

```cpp
class ElasticBuffer {
public:
    ElasticBuffer(const std::string &groupName, int64_t numCpuBytes, bool withGrad = false);

    // 前向（推理/训练共用）
    std::function<at::Tensor()> EngramFetch(
        const at::Tensor &indices,
        at::Tensor *permOut = nullptr,        // save-for-backward（训练态非空）
        at::Tensor *sendCountsOut = nullptr,
        at::Tensor *recvCountsOut = nullptr,
        at::Tensor *recvLocalEntryOut = nullptr,
        at::Tensor *numRecvOut = nullptr);

    // 反向（新算子）
    std::tuple<at::Tensor, at::Tensor> EngramFetchGrad(
        const at::Tensor &gradFetched,
        const at::Tensor &perm, const at::Tensor &sendCounts,
        const at::Tensor &recvCounts, const at::Tensor &recvLocalEntry);

private:
    void EnsureEngramContext();               // 懒初始化
    void AllocateAndExchangeGm(int64_t gmBytes);  // per-fetch GM 申请
    void FreeGm();                             // per-fetch GM 释放
    void UpdateContextTensor();               // 刷 contextTensor_

    bool withGrad_ = false;
    void *gmBufPtr_ = nullptr;                // per-fetch GM 指针
    HcclMemHandle gmMemHandle_ = nullptr;     // GM HCCL 注册句柄
    std::string gmBufferTag_;                 // GM tag（带序号）
};
```

### 4.3 EngramFetch 训练态流程

```
EngramFetch(indices, permOut, sendCountsOut, recvCountsOut, recvLocalEntryOut, numRecvOut)
  ├─ EnsureEngramContext()
  ├─ AllocateAndExchangeGm(gmBytes)           ← 申请 GM + 交换地址 + 更新 context
  ├─ ACLNN_CMD(aclnnEngramFetch, ..., perm, sendCounts, recvCounts, recvLocalEntry, numRecv)
  │    # issue: ① 桶排序 ② a2a(indices) ③ gather ④ a2a(embedding)
  ├─ ACLNN_CMD(aclnnEngramFetchWait, ...)
  │    # wait: ⑤ fetched[perm] = gathered
  ├─ FreeGm()                                 ← barrier + 释放 GM
  ├─ 输出 save-for-backward
  └─ return [fetched]() { return fetched; }
```

### 4.4 EngramFetchGrad 流程

```
EngramFetchGrad(gradFetched, perm, sendCounts, recvCounts, recvLocalEntry)
  ├─ EnsureEngramContext()
  ├─ TORCH_CHECK(withGrad_, ...)
  ├─ AllocateAndExchangeGm(gmBytes)           ← fp32 规模
  ├─ ACLNN_CMD(aclnnEngramFetchGrad, ...)
  │    # kernel: ① grad_sorted = gradFetched[perm]
  │    #         ② a2a(grad_sorted) → recvGrad
  │    #         ③ unique + fp32 scatter-add → gradUnique + uniqueLocalEntry
  ├─ FreeGm()
  ├─ actualK = numUnique.item<int32_t>()
  ├─ narrow 截取
  └─ return {gradUnique, uniqueLocalEntry}
```

### 4.5 pybind 绑定

```cpp
.def(pybind11::init<const std::string &, int64_t, bool>(),
     pybind11::arg("groupName"), pybind11::arg("numCpuBytes"),
     pybind11::arg("withGrad") = false)
.def("engram_fetch", ...,
     pybind11::arg("indices").noconvert(),
     pybind11::arg("permOut") = pybind11::none(),
     pybind11::arg("sendCountsOut") = pybind11::none(),
     pybind11::arg("recvCountsOut") = pybind11::none(),
     pybind11::arg("recvLocalEntryOut") = pybind11::none(),
     pybind11::arg("numRecvOut") = pybind11::none())
.def("engram_fetch_grad", ...,
     pybind11::arg("gradFetched").noconvert(),
     pybind11::arg("perm").noconvert(),
     pybind11::arg("sendCounts").noconvert(),
     pybind11::arg("recvCounts").noconvert(),
     pybind11::arg("recvLocalEntry").noconvert())
```

---

## 五、前向 Host 侧设计（`mc2/engram_fetch/`）

### 5.1 算子定义（`engram_fetch_def.cpp`）

```cpp
class EngramFetch : public OpDef {
    // Input: commContext (int32), indices (int32)
    // Output: fetched (bf16/fp16/fp32) REQUIRED
    //         permOut, sendCountsOut, recvCountsOut, recvLocalEntryOut, numRecvOut (OPTIONAL, int32)
    // Attr: hidden_size (REQUIRED), num_entries_per_rank (REQUIRED)
    // 无 with_grad attr、无 ep_world_size attr
    // 训练模式由可选输出是否非空决定
};
```

### 5.2 save-for-backward 输出（5 个可选输出）

| 输出 | shape | dtype | 说明 |
|------|-------|-------|------|
| `permOut` | `(T,)` | int32 | 桶排列序索引（原始 pos → 桶排 pos） |
| `sendCountsOut` | `(W,)` | int32 | 每 rank 发送的 index 数 |
| `recvCountsOut` | `(W,)` | int32 | 每 rank 接收的 index 数 |
| `recvLocalEntryOut` | `(R_max,)` | int32 | 收到的 local entry（前 numRecv 有效） |
| `numRecvOut` | `(1,)` | int32 | 实际接收行数 R |

推理态：5 个可选输出全传 nullptr。
训练态：5 个可选输出全非空。

### 5.3 TilingData（`engram_fetch_tiling_data.h`）

```cpp
struct EngramFetchTilingData {
    int64_t numTokens;
    int32_t numEntriesPerRank;
    int64_t hiddenDim;
    int64_t hiddenBytes;
    uint32_t aivNum;
    uint64_t ubSize;
    int32_t withGrad;   // kernel 据此选推理/训练分支
};
```

### 5.4 TilingKey（`engram_fetch_tiling_key.h`）

```cpp
#define ENGRAM_FETCH_INFER_MODE 0       // 推理
#define ENGRAM_FETCH_WITH_GRAD_MODE 1   // 训练
```

### 5.5 Tiling（`engram_fetch_tiling.cpp`）

训练模式判断：
```cpp
static bool GetWithGrad(const gert::TilingContext *context)
{
    auto permOutDesc = context->GetOutputDesc(PERM_OUT_INDEX);
    return permOutDesc != nullptr;  // 可选输出非空 → 训练
}
```

Workspace：
- 推理：固定 16MB
- 训练：`max(numTokens * sizeof(int32) * 4, numTokens * hiddenBytes)` + 16MB

### 5.6 aclnn 接口（`aclnn_engram_fetch.cpp`）

```cpp
aclnnStatus aclnnEngramFetchGetWorkspaceSize(
    const aclTensor *commContext, const aclTensor *indices,
    int32_t hiddenSize, int64_t numEntriesPerRank,
    aclTensor *fetched,
    aclTensor *permOut,               // 可选，nullptr=推理
    aclTensor *sendCountsOut,         // 可选
    aclTensor *recvCountsOut,         // 可选
    aclTensor *recvLocalEntryOut,     // 可选
    aclTensor *numRecvOut,            // 可选
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

---

## 六、反向 Host 侧设计（`mc2/engram_fetch_grad/`）

### 6.1 算子定义（`engram_fetch_grad_def.cpp`）

```cpp
class EngramFetchGrad : public OpDef {
    // Input 0: commContext (int32) — 通信上下文（含 gmCommBuffer）
    // Input 1: gradFetched (T, H) bf16/fp16/fp32 — 前向 fetched 的梯度
    // Input 2: perm (T,) int32 — 前向 save-for-backward
    // Input 3: sendCounts (W,) int32 — 前向 save-for-backward
    // Input 4: recvCounts (W,) int32 — 前向 save-for-backward
    // Input 5: recvLocalEntry (R,) int32 — 前向 save-for-backward

    // Output 0: gradUniqueOut (R_max, H) fp32 — 前 numUnique 有效，累加后的稀疏梯度
    // Output 1: uniqueLocalEntryOut (R_max,) int32 — 1D sparse index
    // Output 2: numUniqueOut (1,) int32 — 去重后的 unique local entry 个数

    // Attr: hidden_size (REQUIRED), num_entries_per_rank (REQUIRED)
    // 无 ep_world_size attr
};
```

### 6.2 InferShape（`engram_fetch_grad_infershape.cpp`）

```cpp
// gradUniqueOut = [-1, hiddenSize] fp32（动态 shape）
// uniqueLocalEntryOut = [-1] int32（1D sparse index）
// numUniqueOut = [1] int32
```

### 6.3 TilingData（`engram_fetch_grad_tiling_data.h`）

```cpp
struct EngramFetchGradTilingData {
    int64_t numTokens;             // gradFetched / perm dim0
    int32_t numEntriesPerRank;
    int64_t hiddenDim;
    int64_t hiddenBytes;           // hiddenDim * sizeof(float)（fp32 聚合）
    uint32_t aivNum;
    uint64_t ubSize;
};
```

### 6.4 Tiling（`engram_fetch_grad_tiling.cpp`）

- 校验 6 input + 3 output（dtype/shape/format）
- hiddenBytes 按 **fp32**（`DT_FLOAT`）计算
- rankSize 从 `sendCounts` dim0 获取
- Workspace：`max(numTokens * hiddenBytes * 2, numTokens * hiddenBytes)` + 16MB

### 6.5 aclnn 接口（`aclnn_engram_fetch_grad.cpp`）

```cpp
aclnnStatus aclnnEngramFetchGradGetWorkspaceSize(
    const aclTensor *commContext, const aclTensor *gradFetched,
    const aclTensor *perm, const aclTensor *sendCounts,
    const aclTensor *recvCounts, const aclTensor *recvLocalEntry,
    int32_t hiddenSize, int64_t numEntriesPerRank,
    aclTensor *gradUniqueOut,          // (R_max, H) fp32
    aclTensor *uniqueLocalEntryOut,    // (R_max,) int32 — 1D sparse index
    aclTensor *numUniqueOut,           // (1,) int32
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

---

## 七、端到端调用时序

### 7.1 训练前向

```
Python: buf.engram_fetch(indices)
  │
  ├─ C++: EngramFetch(indices, permOut, sendCountsOut, ...)
  │   ├─ EnsureEngramContext()                  ← 懒初始化（首次）
  │   ├─ AllocateAndExchangeGm(gmBytes)         ← GM 申请 + 地址交换
  │   ├─ ACLNN_CMD(aclnnEngramFetch, ...)       ← issue kernel
  │   │    # ① 桶排序 ② a2a(indices) ③ gather ④ a2a(embedding)
  │   ├─ ACLNN_CMD(aclnnEngramFetchWait, ...)   ← wait kernel
  │   │    # ⑤ fetched[perm] = gathered
  │   ├─ FreeGm()                               ← barrier + 释放 GM
  │   └─ return hook
  │
  └─ Python: hook() → (fetched, fetch_ctx)
```

### 7.2 训练反向

```
Python: buf.engram_fetch_grad(grad_fetched, fetch_ctx)
  │
  ├─ C++: EngramFetchGrad(gradFetched, perm, sendCounts, recvCounts, recvLocalEntry)
  │   ├─ EnsureEngramContext()
  │   ├─ AllocateAndExchangeGm(gmBytes)         ← fp32 规模 GM
  │   ├─ ACLNN_CMD(aclnnEngramFetchGrad, ...)
  │   │    # ① grad_sorted = gradFetched[perm]
  │   │    # ② a2a(grad_sorted) → recvGrad
  │   │    # ③ unique + fp32 scatter-add
  │   ├─ FreeGm()
  │   ├─ actualK = numUnique.item<int32_t>()
  │   ├─ narrow 截取
  │   └─ return (gradUnique, uniqueLocalEntry)
  │
  └─ Python: (grad_unique, unique_local_entry) = result
```

### 7.3 推理态（不变）

```
Python: buf.engram_fetch(indices)
  ├─ C++: EngramFetch(indices)  // 可选输出全 nullptr
  │   ├─ EnsureEngramContext()
  │   ├─ ACLNN_CMD(aclnnEngramFetch, ..., nullptr×5)
  │   └─ return callable
  └─ Python: fetched = hook()  // callable 内调 aclnnEngramFetchWait
```

---

## 八、修改文件清单

### 前向（修改现有文件）

| 文件 | 修改内容 |
|------|---------|
| `engram_fetch.h` | `EngramCommContext` 新增 `gmCommBuffer` |
| `engram_fetch_def.cpp` | 5 个可选输出，无 `is_training`/`ep_world_size` attr |
| `engram_fetch_tiling_data.h` | 新增 `withGrad` 字段 |
| `engram_fetch_tiling_key.h` | `ENGRAM_FETCH_INFER_MODE` / `ENGRAM_FETCH_WITH_GRAD_MODE` |
| `engram_fetch_tiling.cpp` | `GetWithGrad`，训练分支 workspace |
| `aclnn_engram_fetch.cpp` | 5 个可选输出参数 |
| `elastic_buffer.cpp` | `CommContext` + `gmAddrList`；`EngramContextBuilder` + `withGrad`；`ElasticBuffer` + GM 方法 + `EngramFetch` 5 输出 + `EngramFetchGrad`；pybind |
| `elastic_buffer.py` | `EngramFetchCtx`；`with_grad` 构造参数；`engram_fetch` 训练态；`engram_fetch_grad` |

### 反向（新建文件）

| 文件 | 内容 |
|------|------|
| `engram_fetch_grad_def.cpp` | `EngramFetchGrad` 算子定义（6 input + 3 output） |
| `engram_fetch_grad_infershape.cpp` | 动态 shape（gradUnique fp32 / uniqueLocalEntry int32 / numUnique int32） |
| `engram_fetch_grad_tiling_data.h` | TilingData（hiddenBytes 按 fp32） |
| `engram_fetch_grad_tiling_key.h` | 单一 `ENGRAM_FETCH_GRAD_DEFAULT_MODE` |
| `engram_fetch_grad_tiling.cpp` | 7 步 tiling，校验 6 input + 3 output |
| `aclnn_engram_fetch_grad.cpp` | `aclnnEngramFetchGrad` 薄封装 |

---

## 九、逻辑验证示例

### 9.1 初始设置

```
ep_world_size = 3,  numEntriesPerRank = 2,  hidden = 2

全局 index: rank0→[0,1], rank1→[2,3], rank2→[4,5]
各 rank embedding:
  rank0: emb_0 = [[e00,e01],[e10,e11]]
  rank1: emb_1 = [[e20,e21],[e30,e31]]
  rank2: emb_2 = [[e40,e41],[e50,e51]]

各 rank indices:
  rank0: [5,0,2,1], rank1: [0,4], rank2: [3,1,0]
```

### 9.2 前向（rank0 为例）

```
桶排: indices=[5,0,2,1] → owner=[2,0,1,0]
  perm=[3,0,2,1], sendCounts=[2,1,1]
  sortedLocalEntry=[0,1,0,1]

a2a #1 → recvLocalEntry=[0,1,0,1,0] (R=5)
gather → gathered=[emb_0[0],emb_0[1],emb_0[0],emb_0[1],emb_0[0]]
a2a #2 → recvEmb=[[e00,e01],[e10,e11],[e20,e21],[e50,e51]]
还原: fetched[perm] = recvEmb
  fetched = [[e50,e51],[e00,e01],[e20,e21],[e10,e11]] ✓
```

### 9.3 反向（rank0 为例，grad=fetched）

```
① grad_sorted = gradFetched[perm] = [[g00,g01],[g10,g11],[g20,g21],[g50,g51]]
② a2a(grad_sorted) → recvGrad = [[g00,g01],[g10,g11],[g00,g01],[g10,g11],[g00,g01]]
③ scatter-add by recvLocalEntry=[0,1,0,1,0]:
   local 0: g00×3, local 1: g10×2
   gradUnique = [[3g00,3g01],[2g10,2g11]]
   uniqueLocalEntry = [0,1]
   numUnique = 2

验证(grad=fetched时): gradUnique[k] = count_k × emb_0[uniqueLocalEntry[k]] ✓
```
