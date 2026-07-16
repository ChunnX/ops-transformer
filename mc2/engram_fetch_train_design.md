# EngramFetch 训练设计文档

> 需求文档：`mc2/requriment.md` v4.0
> 推理基线：`mc2/engram_fetch_算子源码详解.md`、`mc2/ElasticBuffer_详解.md`

---

## 一、接口变化与新增

### 1.1 Python API 层

**`ElasticBuffer.__init__`（修改）**

| 参数 | 变化 | 类型 | 测试关注 |
|------|------|------|---------|
| ★ `with_grad` | 新增 | bool=False | True→训练态，触发 GM 申请 + save-for-backward |

**`engram_fetch_grad`（★新增）**

| 参数 | 类型 | 测试关注 |
|------|------|---------|
| `grad_fetched` | (T,H) bf16/fp16/fp32 | 前向 fetched 的梯度 |
| `fetch_ctx` | EngramFetchCtx | 5 字段须与前向输出一致 |
| 返回 `grad_unique` | (K,H) fp32 | K=numUnique，前 actualK 有效 |
| 返回 `unique_local_entry` | (K,) int32 | 对应 local entry 索引 |

**`EngramFetchCtx`（★新增）**

| 字段 | shape | dtype | 测试关注 |
|------|-------|-------|---------|
| `perm` | (T,) | int32 | 桶排列序，反向 Input 2 |
| `sendCounts` | (W,) | int32 | 每 rank 发送数，dim0==rankSize |
| `recvCounts` | (W,) | int32 | 每 rank 接收数，dim0==rankSize |
| `recvLocalEntry` | (R_max,) | int32 | dim0==totalRecv |
| `numRecv` | (1,) | int32 | 实际接收行数 R |

### 1.2 aclnn 层

**训练 `aclnnEngramFetchTrainGetWorkspaceSize`（★新增）**

| 方向 | 参数 | dtype/shape | 测试关注 |
|------|------|-------------|---------|
| Input | commContext | int32 1D | REQUIRED |
| Input | indices | int32 (T,) | REQUIRED |
| Attr | hiddenSize | int32 | REQUIRED |
| Attr | numEntriesPerRank | int64 | REQUIRED |
| ★ Attr | localStorageAddr | int64 | host-pinned 设备映射地址（gather 源，即 `deviceBufPtr_`） |
| Output | fetched | bf16/fp16/fp32 (T,H) | REQUIRED |
| Output | permOut | int32 (T,) | REQUIRED |
| Output | sendCountsOut | int32 (W,) | REQUIRED, dim0==rankSize |
| Output | recvCountsOut | int32 (W,) | REQUIRED, dim0==rankSize |
| Output | recvLocalEntryOut | int32 (R_max,) | REQUIRED, dim0==totalRecv |
| Output | numRecvOut | int32 (1,) | REQUIRED |

**反向 `aclnnEngramFetchGradGetWorkspaceSize`（★新增）**

| 方向 | 参数 | dtype/shape | 测试关注 |
|------|------|-------------|---------|
| Input | commContext | int32 1D | REQUIRED |
| Input | gradFetched | bf16/fp16/fp32 (T,H) | REQUIRED |
| Input | perm | int32 (T,) | 来自前向 ctx |
| Input | sendCounts | int32 (W,) | dim0==rankSize |
| Input | recvCounts | int32 (W,) | dim0==rankSize |
| Input | recvLocalEntry | int32 (R_max,) | dim0==totalRecv |
| Input | numRecv | int32 (1,) | REQUIRED |
| Output | gradUniqueOut | fp32 (-1,H) | REQUIRED |
| Output | uniqueLocalEntryOut | int32 (-1,) | REQUIRED |
| Output | numUniqueOut | int32 (1,) | REQUIRED |
| Attr | hidden_size / num_entries_per_rank | int | REQUIRED |

### 1.3 TilingData 层

**训练 `EngramFetchTrainTilingData`（★新增）**：10 字段。关键：
- `rankSize` ← sendCountsOut.dim0
- `totalRecv` ← recvLocalEntryOut.dim0
- `localStorageAddr` ← attr（本地 storage GM 地址）
- `numMaxTokensPerRank` ← totalRecv / rankSize

**反向 `EngramFetchGradTilingData`（★新增）**：8 字段。关键交叉校验：
- `rankSize` ← sendCounts.dim0
- `totalRecv` ← recvLocalEntry.dim0

### 1.4 TilingKey 层

| 算子 | 变化 | 值 | 测试关注 |
|------|------|----|---------|
| 训练 engram_fetch_train | ★ 新增 | TRAIN_DEFAULT_MODE=0 | 单一模式 |
| 反向 engram_fetch_grad | ★ 新增 | GRAD_DEFAULT_MODE=0 | 单一模式 |

---

## 二、PTA

### 2.1 EngramFetchCtx

```python
@dataclass
class EngramFetchCtx:
    perm: torch.Tensor            # (T,) int32 — 桶排列序索引
    sendCounts: torch.Tensor      # (W,) int32 — 每 rank 发送的 index 数
    recvCounts: torch.Tensor      # (W,) int32 — 每 rank 接收的 index 数
    recvLocalEntry: torch.Tensor  # (R_max,) int32 — 收到的 local entry
    numRecv: torch.Tensor         # (1,) int32 — 实际接收行数 R
```

### 2.2 GM 生命周期

训练态数据存储与通信缓冲：

| 缓冲 | 大小 | 用途 | 位置 |
|------|------|------|------|
| host-pinned (`deviceBufPtr_`) | `numCpuBytes` | embedding 表存储（推理RDMA read源 / 训练gather源） | `EngramContextBuilder` 申请 |
| a2a GM (`epHcclBufferPtr_`) | 固定 200MB | all-to-all 收发缓冲（仅训练） | `EnsureEngramContext` 申请 |

storage 始终在 host-pinned，训练态 gather 从 `deviceBufPtr_`（host-pinned 设备映射地址）读取。
200MB a2a GM 仅做 all-to-all 通信的 send/recv 缓冲，不存 storage。
`deviceBufPtr_` 作为 `local_storage_addr` attr 透传给 `aclnnEngramFetchTrain`。

```
EnsureEngramContext :
  ├─ AllocateAndRegisterBuffer → hostBufPtr_ / deviceBufPtr_   ← host-pinned（推理+训练共用）
  ├─ AllocateAndExchangeGm(200MB)
  │   ├─ aclrtMalloc(HUGE_FIRST) → epHcclBufferPtr_
  │   ├─ HcclCommMemReg(tag="...engram_embedding_gm")
  │   ├─ EngramBarrier(true)
  │   ├─ HcclChannelGetRemoteMems → virtualAddrList[peer]  ← 覆盖 host-pinned 地址
  │   ├─ UpdateContextTensor()
  │   └─ EngramBarrier(true)

EngramWrite (推理+训练统一):
  ├─ memcpy_s(storage → hostBufPtr_)
  └─ EngramBarrier(true)

Destroy:
  ├─ aclrtFree(epHcclBufferPtr_)       ← a2a GM
  └─ aclrtFreeHost(hostBufPtr_)         ← host-pinned
```

### 2.3 通信Context

```cpp
struct EngramCommContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBuffer[1024];   // 推理=host-pinned地址 / 训练=a2a GM地址
    uint64_t hcommHandle[1024];  // UBC channel句柄（推理+训练共用）
};
```

> `epHcclBuffer[]` 已删除——训练态 a2a GM 地址复用 `commBuffer[]`（由 `AllocateAndExchangeGm` 覆盖）。
> `localStorageAddr` 作为 aclnn attr 参数传入，不放在 commContext 内。

### 2.4 pybind 绑定

```cpp
// 注册 EngramFetchCtx 为 Python class（属性名与 Python dataclass 一致, duck typing 兼容）
pybind11::class_<Mc2Api::EngramFetchCtx>(m, "EngramFetchCtx")
    .def_readwrite("perm", &Mc2Api::EngramFetchCtx::perm)
    .def_readwrite("sendCounts", &Mc2Api::EngramFetchCtx::sendCounts)
    .def_readwrite("recvCounts", &Mc2Api::EngramFetchCtx::recvCounts)
    .def_readwrite("recvLocalEntry", &Mc2Api::EngramFetchCtx::recvLocalEntry)
    .def_readwrite("numRecv", &Mc2Api::EngramFetchCtx::numRecv);

.def(pybind11::init<const string&, int64_t, int64_t, bool>(),
     pybind11::arg("groupName"), pybind11::arg("numCpuBytes"),
     pybind11::arg("numMaxTokensPerRank") = 0,
     pybind11::arg("withGrad") = false)
.def("engram_fetch",
    [](ElasticBuffer &self, const at::Tensor &indices) -> py::object {
        if (self.IsWithGrad()) {
            Mc2Api::EngramFetchCtx ctx;                    // 局部变量, 不存成员
            auto hook = self.EngramFetch(indices, &ctx);   // ctx 通过输出参数返回
            return py::cpp_function(
                [hook=move(hook), ctx=move(ctx)]() mutable {
                    return py::make_tuple(hook(), py::cast(ctx));   // 2-tuple
                });
        }
        return py::cast(self.EngramFetch(indices));        // 推理: callable 返回单 tensor
    }, pybind11::arg("indices").noconvert())
.def("engram_fetch_grad", &ElasticBuffer::EngramFetchGrad, ...)
```

> C++ `EngramFetch` 加可选输出参数 `EngramFetchCtx *ctxOut = nullptr`（推理 nullptr, 训练非空）。
> ctx 为局部变量, move 进 pybind lambda 捕获, 不存成员——多次 fetch 不覆盖。

### 2.5 使用示例

```python
buf = ElasticBuffer(group, num_cpu_bytes=..., num_max_tokens_per_rank=..., with_grad=True)
buf.engram_write(storage)

# 前向
fetched, fetch_ctx = buf.engram_fetch(indices)()

# ... loss = f(fetched); loss.backward() → grad_fetched ...

# 反向
grad_unique, unique_local_entry = buf.engram_fetch_grad(grad_fetched, fetch_ctx)
# 优化器更新 storage[unique_local_entry] -= lr * grad_unique
```

---

## 三、前向设计

### 3.1 Python 接口

```python
def engram_fetch(self, indices: Tensor) -> Callable
```

Python 无分支，`return self._runtime.engram_fetch(indices)`；C++ pybind lambda 根据 `IsWithGrad` 决定 callable 返回值：

| 模式 | `hook()` 返回 |
|------|--------------|
| 推理 | `fetched (T,H)` |
| 训练 | `(fetched, fetch_ctx)` — fetch_ctx 为 C++ bound EngramFetchCtx |

### 3.2 C++ EngramFetch 流程（`elastic_buffer.cpp`）

```
EngramFetch(indices, ctxOut=nullptr)
  ├─ if withGrad_:
  │   ├─ TORCH_CHECK(epHcclBufferPtr_ != nullptr)         ← a2a GM 已在 EnsureEngramContext 阶段申请
  │   ├─ opts = indices.options()                          ← 携带 device, int32
  │   ├─ EngramFetchCtx ctx{                               ← 局部变量, 不存成员
  │   │       at::empty({numTokens}, opts),                // perm
  │   │       at::empty({rankSize}, opts),                 // sendCounts
  │   │       at::empty({rankSize}, opts),                 // recvCounts
  │   │       at::empty({maxR}, opts),                     // recvLocalEntry, maxR = numTokens * rankSize
  │   │       at::empty({1}, opts)}                        // numRecv
  │   ├─ ACLNN_CMD(aclnnEngramFetchTrain, ..., deviceBufPtr_, ..., ctx.perm, ...)
  │   │       ↑ local_storage_addr = deviceBufPtr_（host-pinned 设备映射地址, gather 源）
  │   ├─ *ctxOut = std::move(ctx)                          ← 输出给调用方（pybind lambda 捕获）
  │   └─ return [fetched]() { return fetched; }
  │
  └─ else:
      ├─ ACLNN_CMD(aclnnEngramFetch, ..., fetched)     ← 异步
      └─ return [fetched]() { ACLNN_CMD(aclnnEngramFetchWait, ...); }
```

### 3.3 算子定义

**推理 `engram_fetch_def.cpp`（修改）**：2 input + 1 output + 2 attr

| 接口 | 方向 | dtype | shape | ParamType |
|------|------|-------|-------|-----------|
| commContext | Input 0 | int32 | 1D | REQUIRED |
| indices | Input 1 | int32 | (T,) | REQUIRED |
| fetched | Output 0 | bf16/fp16/fp32 | (T, H) | REQUIRED |
| hidden_size | Attr | int | — | REQUIRED |
| num_entries_per_rank | Attr | int | — | REQUIRED |

**训练 `engram_fetch_train_def.cpp`（★新增）**：2 input + 6 output + 3 attr

| 接口 | 方向 | dtype | shape | ParamType |
|------|------|-------|-------|-----------|
| commContext | Input 0 | int32 | 1D | REQUIRED |
| indices | Input 1 | int32 | (T,) | REQUIRED |
| fetched | Output 0 | bf16/fp16/fp32 | (T, H) | REQUIRED |
| permOut | Output 1 | int32 | (T,) | REQUIRED |
| sendCountsOut | Output 2 | int32 | (W,) | REQUIRED |
| recvCountsOut | Output 3 | int32 | (W,) | REQUIRED |
| recvLocalEntryOut | Output 4 | int32 | (R_max,) | REQUIRED |
| numRecvOut | Output 5 | int32 | (1,) | REQUIRED |
| hidden_size | Attr | int | — | REQUIRED |
| num_entries_per_rank | Attr | int | — | REQUIRED |
| ★ local_storage_addr | Attr | int | — | REQUIRED |

### 3.4 TilingData

**推理 `EngramFetchTilingData`（修改）**：纯推理，6 字段
```cpp
struct EngramFetchTilingData {
    int64_t numTokens;
    int32_t numEntriesPerRank;
    int64_t hiddenDim;
    int64_t hiddenBytes;
    uint32_t aivNum;
    uint64_t ubSize;
};
```

**训练 `EngramFetchTrainTilingData`（★新增）**：10 字段
```cpp
struct EngramFetchTrainTilingData {
    int64_t numTokens;
    int32_t numEntriesPerRank;
    int64_t hiddenDim;
    int64_t hiddenBytes;
    uint32_t aivNum;
    uint64_t ubSize;
    uint32_t rankSize;            // 从 sendCountsOut dim0 获取
    int64_t numMaxTokensPerRank;  // totalRecv / rankSize
    int64_t totalRecv;            // 从 recvLocalEntryOut dim0 获取
    int64_t localStorageAddr;   // 从 attr 获取（host-pinned 设备映射地址, gather 源）
};
```

### 3.5 TilingKey

```
engram_fetch:         ENGRAM_FETCH_INFER_MODE = 0         // 推理（单一模式）
engram_fetch_train:   ENGRAM_FETCH_TRAIN_DEFAULT_MODE = 0 // 训练（单一模式）
```

### 3.6 Workspace 布局（训练态）

```
workspace = sdispls + rdispls + sortedIndices + tokenIdxInRank
          + recvIndices + localData + recvData + flagScratch
          + 16MB系统保留
```

| 分项 | 大小 | 说明 |
|------|------|------|
| sdispls | rankSize × 8B | send 前缀和（int64, 中间计算） |
| rdispls | rankSize × 8B | recv 前缀和（int64, 中间计算） |
| sortedIndices | numTokens × 4B | 按 rank 分组的 indices |
| tokenIdxInRank | numTokens × 4B | 原始位置记录 |
| recvIndices | totalRecv × 4B | 接收到的 indices |
| localData | totalRecv × hiddenBytes | 本地读表输出 |
| recvData | numTokens × hiddenBytes | 交换后的数据（还原前） |
| flagScratch | 32B | WriteNbi flag 源地址 |

> `totalRecv = numMaxTokensPerRank × rankSize`
> 输出 tensor（fetched/permOut/sendCountsOut/recvCountsOut/recvLocalEntryOut/numRecvOut）

推理态：固定 16MB。

### 3.7 aclnn 接口

**推理（修改）**
```cpp
aclnnStatus aclnnEngramFetchGetWorkspaceSize(
    const aclTensor *commContext, const aclTensor *indices,
    int32_t hiddenSize, int64_t numEntriesPerRank,
    aclTensor *fetched,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

**训练（★新增）**
```cpp
aclnnStatus aclnnEngramFetchTrainGetWorkspaceSize(
    const aclTensor *commContext, const aclTensor *indices,
    int32_t hiddenSize, int64_t numEntriesPerRank, int64_t localStorageAddr,
    aclTensor *fetched, aclTensor *permOut, aclTensor *sendCountsOut,
    aclTensor *recvCountsOut, aclTensor *recvLocalEntryOut, aclTensor *numRecvOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

### 3.8 Kernel

**推理（修改，无模板）**
```cpp
__global__ __aicore__ void engram_fetch(
    GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
```

**训练（★新增，单一模板）**
```cpp
template<uint32_t EngramFetchTrainMode>
__global__ __aicore__ void engram_fetch_train(
    GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
    GM_ADDR permOut, GM_ADDR sendCountsOut,
    GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut,
    GM_ADDR numRecvOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
```

- 推理: RDMA read（`commBuffer[peer]` = host-pinned）
- 训练: all-to-all（`commBuffer[peer]` = a2a GM；`localStorageAddr` = gather 源）

---

## 四、反向设计

### 4.1 Python 接口

```python
def engram_fetch_grad(self, grad_fetched: Tensor, fetch_ctx: EngramFetchCtx
                      ) -> Tuple[Tensor, Tensor]
```

| 参数 | shape | dtype | 说明 |
|------|-------|-------|------|
| `grad_fetched` | (T,H) | bf16/fp16/fp32 | 前向 fetched 的梯度 |
| `fetch_ctx` | — | — | EngramFetchCtx（5 个 save-for-backward tensor） |
| 返回 `grad_unique` | (K,H) | fp32 | 累加后的稀疏梯度 |
| 返回 `unique_local_entry` | (K,) | int32 | 对应的 local entry 索引 |

### 4.2 C++ EngramFetchGrad 流程（`elastic_buffer.cpp`）

```
EngramFetchGrad(gradFetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv)
  ├─ ACLNN_CMD(aclnnEngramFetchGrad, ..., 3 outputs) 
  ├─ actualK = numUnique.item<int32_t>()               ← 有效行数
  ├─ gradUnique.narrow(0, 0, actualK)                  ← 截取有效部分
  └─ return {gradUnique, uniqueLocalEntry}
```

### 4.3 算子定义（`engram_fetch_grad_def.cpp`）

| 接口 | 方向 | dtype | shape | ParamType |
|------|------|-------|-------|-----------|
| commContext | Input 0 | int32 | 1D | REQUIRED |
| gradFetched | Input 1 | bf16/fp16/fp32 | (T, H) | REQUIRED |
| perm | Input 2 | int32 | (T,) | REQUIRED |
| sendCounts | Input 3 | int32 | (W,) | REQUIRED |
| recvCounts | Input 4 | int32 | (W,) | REQUIRED |
| recvLocalEntry | Input 5 | int32 | (R_max,) | REQUIRED |
| numRecv | Input 6 | int32 | (1,) | REQUIRED |
| gradUniqueOut | Output 0 | **fp32** | (-1, H) | REQUIRED |
| uniqueLocalEntryOut | Output 1 | int32 | (-1,) | REQUIRED |
| numUniqueOut | Output 2 | int32 | (1,) | REQUIRED |
| hidden_size | Attr | int | — | REQUIRED |
| num_entries_per_rank | Attr | int | — | REQUIRED |


### 4.4 TilingData

```cpp
struct EngramFetchGradTilingData {
    int64_t numTokens;          // gradFetched / perm dim0
    int32_t numEntriesPerRank;
    int64_t hiddenDim;
    int64_t hiddenBytes;        // hiddenDim * sizeof(float)（fp32 聚合）
    uint32_t aivNum;
    uint64_t ubSize;
    uint32_t rankSize;          // 从 sendCounts dim0 获取
    int64_t totalRecv;          // 从 recvLocalEntry dim0 获取
};
```

### 4.6 TilingKey

```
ENGRAM_FETCH_GRAD_DEFAULT_MODE = 0 
```

### 4.7 Workspace 布局

```
workspace = gradSorted + recvGrad + sdispls + rdispls
          + tokenIdxInRank + recvIndices + flagScratch
          + 16MB系统保留
```

| 分项 | 大小 | 用途 |
|------|------|------|
| gradSorted | numTokens × hiddenBytes | 步骤① gradFetched[perm] (fp32) |
| recvGrad | totalRecv × hiddenBytes | 步骤② a2a 接收梯度 (fp32) |
| sdispls | rankSize × 8B | send 前缀和 (int64, 中间计算) |
| rdispls | rankSize × 8B | recv 前缀和 (int64, 中间计算) |
| tokenIdxInRank | numTokens × 4B | 原始位置记录 (int32, 从 forwardCtx 解析) |
| recvIndices | totalRecv × 4B | 接收到的 indices (int32, 从 forwardCtx 解析) |
| flagScratch | 32B | WriteNbi flag 源地址 |

> 输入 tensor（sendCounts/recvCounts/perm/recvLocalEntry/numRecv）和输出 tensor（gradUniqueOut/uniqueLocalEntryOut/numUniqueOut）由 C++ `at::empty` 分配，不在 workspace 中。

### 4.8 aclnn 接口

```cpp
aclnnStatus aclnnEngramFetchGradGetWorkspaceSize(
    const aclTensor *commContext, const aclTensor *gradFetched,
    const aclTensor *perm, const aclTensor *sendCounts,
    const aclTensor *recvCounts, const aclTensor *recvLocalEntry,
    const aclTensor *numRecv,
    int32_t hiddenSize, int64_t numEntriesPerRank,
    aclTensor *gradUniqueOut, aclTensor *uniqueLocalEntryOut, aclTensor *numUniqueOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

### 4.9 Kernel

```cpp
template<uint32_t EngramFetchGradMode>
__global__ __aicore__ void engram_fetch_grad(
    GM_ADDR commContext, GM_ADDR gradFetched,
    GM_ADDR perm, GM_ADDR sendCounts, GM_ADDR recvCounts,
    GM_ADDR recvLocalEntry, GM_ADDR numRecv,
    GM_ADDR gradUniqueOut, GM_ADDR uniqueLocalEntryOut, GM_ADDR numUniqueOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM)
```

---

## 五、修改文件清单

### 推理 engram_fetch（修改现有文件）

| 文件 | 修改 |
|------|------|
| `engram_fetch.h` | `EngramCommContext` 删除 `epHcclBuffer[]`；`commBuffer` 语义双用 |
| `engram_fetch_def.cpp` | 删除 5 个 OPTIONAL 输出，回归 2 in + 1 out |
| `engram_fetch_tiling_data.h` | 删除 `withGrad`/`numMaxTokensPerRank`/`rankSize`，回归 6 字段 |
| `engram_fetch_tiling_key.h` | 删除 `WITH_GRAD_MODE`，单一 `INFER_MODE` |
| `engram_fetch_tiling.cpp` | 删除 `GetWithGrad`/withGrad 分支；纯推理 tiling；固定 16MB workspace |
| `aclnn_engram_fetch.cpp` | 删除 5 个可选输出参数 |
| `engram_fetch.cpp` kernel | 删除 template + `constexpr if`；纯推理 kernel |

### 训练 engram_fetch_train（★新建目录）

| 文件 | 内容 |
|------|------|
| `engram_fetch_train_def.cpp` | 2 in + 6 out + 3 attr（含 `local_storage_addr`） |
| `engram_fetch_train_infershape.cpp` | fetched(T,H) + permOut(T,) + sendCountsOut/recvCountsOut(W) + recvLocalEntryOut(R_max) + numRecvOut(1) |
| `engram_fetch_train_tiling.cpp` | 校验 2 in + 6 out；rankSize←sendCountsOut.dim0；totalRecv←recvLocalEntryOut.dim0；精确 workspace |
| `engram_fetch_train_tiling_data.h` | 10 字段（含 rankSize/totalRecv/localStorageAddr） |
| `engram_fetch_train_tiling_key.h` | `ENGRAM_FETCH_TRAIN_DEFAULT_MODE` |
| `aclnn_engram_fetch_train.cpp` | `aclnnEngramFetchTrain`（含 localStorageAddr 参数） |
| `engram_fetch_train.cpp` | kernel 入口（单一模板，stub） |

### C++ / Python（修改现有文件）

| 文件 | 修改 |
|------|------|
| `elastic_buffer.cpp` | `CommContext` 删 `epHcclBuffer[]`；`AllocateAndExchangeGm` 填 `virtualAddrList`；`EngramFetch` 训练态调 `aclnnEngramFetchTrain`（透传 `deviceBufPtr_` 作为 `local_storage_addr`）；`EngramWrite` 推理/训练统一（storage 拷 host-pinned）；200MB GM 仅做 a2a 收发缓冲 |
| `elastic_buffer.py` | 无签名变更（`engram_fetch` 通过 `IsWithGrad()` 分支，pybind lambda 路由两算子） |

### 反向 engram_fetch_grad（新建文件，不变）

| 文件 | 内容 |
|------|------|
| `engram_fetch_grad_def.cpp` | 7 input + 3 output |
| `engram_fetch_grad_infershape.cpp` | 动态 shape（-1, hiddenSize）fp32 |
| `engram_fetch_grad_tiling_data.h` | TilingData（含 rankSize/totalRecv, fp32） |
| `engram_fetch_grad_tiling_key.h` | `ENGRAM_FETCH_GRAD_DEFAULT_MODE` |
| `engram_fetch_grad_tiling.cpp` | 校验 7 input + 3 output；交叉校验 sendCounts==recvCounts；精确 workspace |
| `aclnn_engram_fetch_grad.cpp` | `aclnnEngramFetchGrad` 薄封装（含 numRecv） |
| `engram_fetch_grad.cpp` | kernel 入口（template, 含 numRecv 参数） |

---

## 六、逻辑验证示例

### 6.1 前向训练态（EngramFetchTrain，5 步融合 kernel）

```
indices[T]
   │ ① 桶排序: owner=idx//E, local=idx%E → perm[T], sendCounts[W], sortedLocalEntry[T]
   │ ② a2a(sortedLocalEntry) → recvLocalEntry[R], recvCounts[W]   (用 commBuffer[peer] = a2a GM)
   │ ③ 本地 gather: localStorageAddr[recvLocalEntry] → recvTokens(R,H)
   │ ④ a2a(recvTokens) → gathered(T,H) sorted 序                     (用 commBuffer[peer] = a2a GM)
   │ ⑤ 还原: fetched[perm] = gathered
   ├─→ fetched (T,H) bf16
   └─→ save-for-backward: perm, sendCounts, recvCounts, recvLocalEntry, numRecv
```

### 6.2 反向（EngramFetchGrad，3 步单融合 kernel）

```
gradFetched(T,H) + perm/sendCounts/recvCounts/recvLocalEntry/numRecv
   │ ① grad_sorted = gradFetched[perm]
   │ ② a2a(grad_sorted; sendCounts↔recvCounts) → recvGrad(R,H)   (用 commBuffer[peer] = a2a GM)
   │ ③ owner unique + fp32 scatter-add → gradUnique + uniqueLocalEntry + numUnique
   ├─→ gradUniqueOut (R_max,H) fp32     前 numUnique 有效
   ├─→ uniqueLocalEntryOut (R_max,) int32
   └─→ numUniqueOut (1,) int32
```

### 6.3 数值验证

```
ep_world_size=3, numEntriesPerRank=2, hidden=2
全局 index: rank0→[0,1], rank1→[2,3], rank2→[4,5]
rank0 indices=[5,0,2,1]

前向:
  桶排 → perm=[3,0,2,1], sendCounts=[2,1,1], sortedLocalEntry=[0,1,0,1]
  a2a#1 → recvLocalEntry=[0,1,0,1,0] (R=5)
  gather → gathered=[emb0[0],emb0[1],emb0[0],emb0[1],emb0[0]]
  a2a#2 → recvEmb=[[e50,e51],[e00,e01],[e20,e21],[e10,e11]]
  还原: fetched[perm]=recvEmb → fetched=[[e50,e51],[e00,e01],[e20,e21],[e10,e11]] ✓

反向 (grad=fetched):
  ① grad_sorted=gradFetched[perm]=[[g50,g51],[g00,g01],[g20,g21],[g10,g11]]
  ② a2a(grad_sorted) → recvGrad=[[g00,g01],[g10,g11],[g00,g01],[g10,g11],[g00,g01]]
  ③ scatter-add by recvLocalEntry=[0,1,0,1,0]:
     local 0: g00×3, local 1: g10×2
     gradUnique=[[3g00,3g01],[2g10,2g11]], uniqueLocalEntry=[0,1], numUnique=2 ✓
```
