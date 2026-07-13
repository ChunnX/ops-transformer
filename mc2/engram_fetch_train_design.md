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
| ★ `num_npu_bytes` | 新增 | int=0 | GM 字节数，由 `get_engram_storage_size_hint` 返回 |

**`engram_fetch`（修改，签名不变）**

| 项 | 变化 | 测试关注 |
|----|------|---------|
| ▽ 返回值 | 推理: `fetched`；训练: `(fetched, EngramFetchCtx)` | 同接口两形态，按 `IsWithGrad()` 分支；|

**`engram_fetch_grad`（★新增）**

| 参数 | 类型 | 测试关注 |
|------|------|---------|
| `grad_fetched` | (T,H) bf16/fp16/fp32 | 前向 fetched 的梯度 |
| `fetch_ctx` | EngramFetchCtx | 5 字段须与前向输出一致 |
| 返回 `grad_unique` | (K,H) fp32 | K=numUnique，前 actualK 有效 |
| 返回 `unique_local_entry` | (K,) int32 | 对应 local entry 索引 |

**`EngramFetchCtx`（★新增，dataclass + C++ bound class）**

| 字段 | shape | dtype | 测试关注 |
|------|-------|-------|---------|
| `perm` | (T,) | int32 | 桶排列序，反向 Input 2 |
| `sendCounts` | (W,) | int32 | 每 rank 发送数，dim0==rankSize |
| `recvCounts` | (W,) | int32 | 每 rank 接收数，dim0==rankSize |
| `recvLocalEntry` | (R_max,) | int32 | dim0==totalRecv |
| `numRecv` | (1,) | int32 | 实际接收行数 R |

**`get_engram_storage_size_hint`（修改）**

| 项 | 变化 | 测试关注 |
|----|------|---------|
| ★ `num_max_tokens_per_rank` | 新增第 4 参 | 训练态 GM 大小计算 |
| ▽ 返回类型 | `int → Tuple[int,int]` | (cpuBytes, npuBytes) |

### 1.2 aclnn 层

**`aclnnEngramFetchGetWorkspaceSize`（修改）**：新增 5 个可选输出（插在 `fetched` 与 `workspaceSize` 间）

| 参数 | 变化 | 类型 | 测试关注 |
|------|------|------|---------|
| ★ `permOut` | 新增 | aclTensor* | 训练非空 / 推理 nullOpt |
| ★ `sendCountsOut` | 新增 | aclTensor* | 同上 |
| ★ `recvCountsOut` | 新增 | aclTensor* | 同上 |
| ★ `recvLocalEntryOut` | 新增 | aclTensor* | 同上 |
| ★ `numRecvOut` | 新增 | aclTensor* | 同上 |

**`aclnnEngramFetchGradGetWorkspaceSize`（★新增）**

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

**前向 `EngramFetchTilingData`（修改）**

| 字段 | 变化 | 类型 | 测试关注 |
|------|------|------|---------|
| ★ `withGrad` | 新增 | int32 | 0=推理/1=训练，决定 TilingKey |
| ★ `numMaxTokensPerRank` | 新增 | int64 | workspace 计算用 |
| ★ `rankSize` | 新增 | uint32 | 从 sendCountsOut.dim0 获取 |

**反向 `EngramFetchGradTilingData`（★新增）**：8 字段全新增。关键交叉校验：
- `rankSize` ← sendCounts.dim0
- `totalRecv` ← recvLocalEntry.dim0

### 1.4 TilingKey 层

| 算子 | 变化 | 值 | 测试关注 |
|------|------|----|---------|
| 前向 | ▽ DEFAULT_MODE → 二值 | INFER_MODE=0 / WITH_GRAD_MODE=1 | 按 withGrad 选分支 |
| 反向 | ★ 新增 | GRAD_DEFAULT_MODE=0 | 单一模式 |

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

### 2.2 ElasticBuffer 构造函数
TO DO:疑问点1：num_max_tokens_per_rank --- MOE 和 num_max_tokens_per_rank -- engram  是否是一个
TO DO:疑问点2：num_max_tokens_per_rank 的大小限定问题
```python
ElasticBuffer(group, *, num_cpu_bytes=0, with_grad=False,
              num_max_tokens_per_rank=0, ...)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `group` | ProcessGroup | 分布式通信组 |
| `num_cpu_bytes` | int | CPU 锁页内存大小（2MB 对齐） |
| `with_grad` | bool | True=训练模式（all-to-all + save-for-backward） |
| `num_max_tokens_per_rank` | Optional[int] | 每 rank 最大 token 数（训练 GM 缓冲区 + MoE dispatch 共用） |

### 2.3 GM 生命周期

GM 在 `EngramWrite` 阶段申请，`Destroy` 时释放（前向/反向共用）。

GM 大小由 `GetEngramStorageSizeHint` 同时计算（返回 `(numCpuBytes, numNpuBytes)`），
经 Python 构造函数 `num_npu_bytes` 参数传入 C++ `numNpuBytes_` 成员：
```
numNpuBytes = AlignTo(numMaxTokensPerRank * hiddenSize * sizeof(float), 2MB)
```

```
EngramWrite :
  ├─ ... HCCL context 建立 ...
  ├─ EnsureEngramContext -> AllocateAndExchangeGm(numNpuBytes_)
  │   ├─ aclrtMalloc(HUGE_FIRST) → epHcclBufferPtr_
  │   ├─ HcclCommMemReg(tag="...engram_embedding_gm")
  │   ├─ EngramBarrier(true)
  │   ├─ HcclChannelGetRemoteMems → epHcclBuffer[peer]
  │   ├─ UpdateContextTensor()
  │   └─ EngramBarrier(true)

Destroy:
  └─ aclrtFree(epHcclBufferPtr_)
```

### 2.4 通信Context

```cpp
struct EngramCommContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBuffer[1024];      // host pinned 设备映射地址（推理+训练）
    uint64_t hcommHandle[1024];     // UBC channel 句柄（推理+训练）
    uint64_t epHcclBuffer[1024];    // GM 缓冲区地址（仅训练 a2a）
};
```

### 2.5 pybind 绑定

```cpp
// 注册 EngramFetchCtx 为 Python class（属性名与 Python dataclass 一致, duck typing 兼容）
pybind11::class_<Mc2Api::EngramFetchCtx>(m, "EngramFetchCtx")
    .def_readwrite("perm", &Mc2Api::EngramFetchCtx::perm)
    .def_readwrite("sendCounts", &Mc2Api::EngramFetchCtx::sendCounts)
    .def_readwrite("recvCounts", &Mc2Api::EngramFetchCtx::recvCounts)
    .def_readwrite("recvLocalEntry", &Mc2Api::EngramFetchCtx::recvLocalEntry)
    .def_readwrite("numRecv", &Mc2Api::EngramFetchCtx::numRecv);

.def(pybind11::init<const string&, int64_t, bool, int64_t, int64_t>(),
     pybind11::arg("groupName"), pybind11::arg("numCpuBytes"),
     pybind11::arg("withGrad") = false, pybind11::arg("numMaxTokensPerRank") = 0,
     pybind11::arg("numNpuBytes") = 0)
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

### 2.6 使用示例

```python
buf = ElasticBuffer(group, num_cpu_bytes=..., with_grad=True, num_max_tokens_per_rank=...)
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
  │   ├─ TORCH_CHECK(epHcclBufferPtr_ != nullptr)         ← GM 已在 EnsureEngramContext 阶段申请
  │   ├─ opts = indices.options()                          ← 携带 device, int32
  │   ├─ EngramFetchCtx ctx{                               ← 局部变量, 不存成员
  │   │       at::empty({numTokens}, opts),                // perm
  │   │       at::empty({rankSize}, opts),                 // sendCounts
  │   │       at::empty({rankSize}, opts),                 // recvCounts
  │   │       at::empty({maxR}, opts),                     // recvLocalEntry, maxR = numTokens * rankSize
  │   │       at::empty({1}, opts)}                        // numRecv
  │   ├─ ACLNN_CMD(aclnnEngramFetch, ..., ctx.perm, ctx.sendCounts, ...)
  │   ├─ *ctxOut = std::move(ctx)                          ← 输出给调用方（pybind lambda 捕获）
  │   └─ return [fetched]() { return fetched; }
  │
  └─ else:
      ├─ ACLNN_CMD(aclnnEngramFetch, ..., nullOpt×5)  ← 异步
      └─ return [fetched]() { ACLNN_CMD(aclnnEngramFetchWait, ...); }
```

### 3.3 算子定义（`engram_fetch_def.cpp`）

| 接口 | 方向 | dtype | shape | ParamType |
|------|------|-------|-------|-----------|
| commContext | Input 0 | int32 | 1D | REQUIRED |
| indices | Input 1 | int32 | (T,) | REQUIRED |
| fetched | Output 0 | bf16/fp16/fp32 | (T, H) | REQUIRED |
| permOut | Output 1 | int32 | (T,) | OPTIONAL |
| sendCountsOut | Output 2 | int32 | (W,) | OPTIONAL |
| recvCountsOut | Output 3 | int32 | (W,) | OPTIONAL |
| recvLocalEntryOut | Output 4 | int32 | (R_max,) | OPTIONAL |
| numRecvOut | Output 5 | int32 | (1,) | OPTIONAL |
| hidden_size | Attr | int | — | REQUIRED |
| num_entries_per_rank | Attr | int | — | REQUIRED |

训练模式判断：`GetWithGrad(context)` 检查 `permOut` 是否非空。

### 3.4 TilingData

```cpp
struct EngramFetchTilingData {
    int64_t numTokens;
    int32_t numEntriesPerRank;
    int64_t hiddenDim;
    int64_t hiddenBytes;          // hiddenDim * dtypeSize
    uint32_t aivNum;
    uint64_t ubSize;
    int32_t withGrad;             // 0=推理, 1=训练
    int64_t numMaxTokensPerRank;  // 训练态 workspace 计算用
    uint32_t rankSize;            // 从 sendCountsOut dim0 获取
};
```

### 3.5 TilingKey

```
ENGRAM_FETCH_INFER_MODE = 0     // 推理（RDMA read）
ENGRAM_FETCH_WITH_GRAD_MODE = 1 // 训练（all-to-all）
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

```cpp
aclnnStatus aclnnEngramFetchGetWorkspaceSize(
    const aclTensor *commContext, const aclTensor *indices,
    int32_t hiddenSize, int64_t numEntriesPerRank,
    aclTensor *fetched,
    aclTensor *permOut, aclTensor *sendCountsOut,
    aclTensor *recvCountsOut, aclTensor *recvLocalEntryOut,
    aclTensor *numRecvOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```

### 3.8 Kernel

```cpp
template<uint32_t EngramFetchMode>
__global__ __aicore__ void engram_fetch(
    GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched,
    GM_ADDR permOut, GM_ADDR sendCountsOut,
    GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut,
    GM_ADDR numRecvOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
```

- `INFER_MODE`: RDMA read（现有推理实现）
- `WITH_GRAD_MODE`: all-to-all

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

### 前向（修改现有文件）

| 文件 | 修改 |
|------|------|
| `engram_fetch.h` | `EngramCommContext` 新增 `epHcclBuffer` |
| `engram_fetch_def.cpp` | 5 个 OPTIONAL 输出 |
| `engram_fetch_tiling_data.h` | 新增 `withGrad`/`numMaxTokensPerRank`/`rankSize` |
| `engram_fetch_tiling_key.h` | `INFER_MODE` / `WITH_GRAD_MODE` |
| `engram_fetch_tiling.cpp` | `GetWithGrad`；训练态可选输出校验；精确 workspace |
| `aclnn_engram_fetch.cpp` | 5 个可选输出参数 |
| `elastic_buffer.cpp` | `CommContext`+`epHcclBuffer`；`EngramContextBuilder`+`withGrad_`成员；GM EnsureEngramContext 阶段申请/destroy 释放；`numNpuBytes_`成员；`GetEngramStorageSizeHint` 返回 tuple；新增 `EngramFetchCtx` 结构体；`EngramFetch` 加 `ctxOut` 输出参数（局部 ctx, 不存成员）；删 `savedXxx_`×5 + `GetFetchCtx`；pybind 注册 `EngramFetchCtx` class + 训练返回 2-tuple；`EngramFetchGrad`+numRecv |
| `elastic_buffer.py` | `EngramFetchCtx` dataclass（保留, duck typing 兼容 C++ bound class）；`with_grad`/`num_max_tokens_per_rank` 构造参数；`engram_fetch` 无分支（`return self._runtime.engram_fetch(indices)`）；`engram_fetch_grad` |

### 反向（新建文件）

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

### 6.1 前向训练态（EngramFetch，5 步融合在 issue kernel 内，无 wait kernel）

```
indices[T]
   │ ① 桶排序: owner=idx//E, local=idx%E → perm[T], sendCounts[W], sortedLocalEntry[T]
   │ ② a2a(sortedLocalEntry) → recvLocalEntry[R], recvCounts[W]
   │ ③ 本地 gather: storage[recvLocalEntry] → recvTokens(R,H)
   │ ④ a2a(recvTokens) → gathered(T,H) sorted 序
   │ ⑤ 还原: fetched[perm] = gathered
   ├─→ fetched (T,H) bf16
   └─→ save-for-backward: perm, sendCounts, recvCounts, recvLocalEntry, numRecv
```

### 6.2 反向（EngramFetchGrad，3 步单融合 kernel）

```
gradFetched(T,H) + perm/sendCounts/recvCounts/recvLocalEntry/numRecv
   │ ① grad_sorted = gradFetched[perm]
   │ ② a2a(grad_sorted; sendCounts↔recvCounts) → recvGrad(R,H)
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
