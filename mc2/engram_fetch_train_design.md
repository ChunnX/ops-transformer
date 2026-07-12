# EngramFetch 训练设计文档

> 需求文档：`mc2/requriment.md` v4.0
> 推理基线：`mc2/engram_fetch_算子源码详解.md`、`mc2/ElasticBuffer_详解.md`

---

## 一、数据流

### 1.1 前向训练态（EngramFetch，5 步融合在 issue kernel 内，无 wait kernel）

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

### 1.2 反向（EngramFetchGrad，3 步单融合 kernel）

```
gradFetched(T,H) + perm/sendCounts/recvCounts/recvLocalEntry/numRecv
   │ ① grad_sorted = gradFetched[perm]
   │ ② a2a(grad_sorted; sendCounts↔recvCounts) → recvGrad(R,H)
   │ ③ owner unique + fp32 scatter-add → gradUnique + uniqueLocalEntry + numUnique
   ├─→ gradUniqueOut (R_max,H) fp32     前 numUnique 有效
   ├─→ uniqueLocalEntryOut (R_max,) int32
   └─→ numUniqueOut (1,) int32
```

---

## 二、共享基础设施

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

```python
ElasticBuffer(group, *, num_cpu_bytes=0, with_grad=False,
              num_max_tokens_per_rank=0, ...)
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `group` | ProcessGroup | 分布式通信组 |
| `num_cpu_bytes` | int | CPU 锁页内存大小（2MB 对齐） |
| `with_grad` | bool | True=训练模式（all-to-all + save-for-backward） |
| `num_max_tokens_per_rank` | int | 每 rank 最大 token 数，决定 GM 通信缓冲区大小 |

### 2.3 GM 生命周期

GM 在 `EngramWrite` 末尾一次性申请，`Destroy` 时释放（前向/反向共用）：

```
EngramWrite 末尾 (训练态首次):
  ├─ gmBytes = AlignTo(maxR * hiddenSize * sizeof(float), 2MB)
  │   maxR = numMaxTokensPerRank * rankSize
  ├─ aclrtMalloc(HUGE_FIRST) → gmBufPtr_
  ├─ HcclCommMemReg(tag="...engram_embedding_gm")
  ├─ EngramBarrier(true)
  ├─ HcclChannelGetRemoteMems → gmAddrList[peer]
  ├─ UpdateContextTensor()
  └─ EngramBarrier(true)

Destroy:
  └─ aclrtFree(gmBufPtr_)
```

### 2.4 通信上下文（前向/反向共用）

```cpp
struct EngramCommContext {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commBuffer[1024];      // host pinned 设备映射地址（推理+训练）
    uint64_t hcommHandle_[1024];    // UBC channel 句柄（推理+训练）
    uint64_t gmCommBuffer[1024];    // GM 缓冲区地址（仅训练 a2a）
};
```

### 2.5 EngramContextBuilder

`withGrad` 作为成员变量，`Build()` 入口设置一次，下游方法通过 `withGrad_` 访问：

```cpp
class EngramContextBuilder {
public:
    EngramContextResources Build(..., bool withGrad = false);
private:
    bool withGrad_ = false;
    // BuildChannelDescs: HcclChannelDescInit(&desc, withGrad_ ? 2 : 1)
};
```

### 2.6 pybind 绑定

```cpp
.def(pybind11::init<const string&, int64_t, bool, int64_t>(),
     pybind11::arg("groupName"), pybind11::arg("numCpuBytes"),
     pybind11::arg("withGrad") = false, pybind11::arg("numMaxTokensPerRank") = 0)
.def("engram_fetch",
    [](ElasticBuffer &self, const at::Tensor &indices) -> py::object {
        auto hook = self.EngramFetch(indices);
        if (self.IsWithGrad()) {
            auto ctx = self.GetFetchCtx();
            return py::cpp_function([hook=move(hook), ctx=move(ctx)]() mutable {
                return py::make_tuple(hook(), get<0>(ctx), get<1>(ctx), get<2>(ctx), get<3>(ctx), get<4>(ctx));
            });
        }
        return py::cast(move(hook));
    }, pybind11::arg("indices").noconvert())
.def("engram_fetch_grad", &ElasticBuffer::EngramFetchGrad, ...)
```

### 2.7 使用示例

```python
buf = ElasticBuffer(group, num_cpu_bytes=..., with_grad=True, num_max_tokens_per_rank=...)
buf.engram_write(storage)

# 前向
hook = buf.engram_fetch(indices)
result = hook()
fetched, fetch_ctx = result[0], EngramFetchCtx(*result[1:])

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

Python 无分支，C++ pybind lambda 根据 `withGrad` 决定返回值：

| 模式 | `hook()` 返回 |
|------|--------------|
| 推理 | `fetched (T,H)` |
| 训练 | `(fetched, perm, sendCounts, recvCounts, recvLocalEntry, numRecv)` 6-tuple |

### 3.2 C++ EngramFetch 流程（`elastic_buffer.cpp`）

```
EngramFetch(indices)
  ├─ if withGrad_:
  │   ├─ TORCH_CHECK(gmBufPtr_ != nullptr)         ← GM 已在 write 阶段申请
  │   ├─ savedPerm_ = at::empty({numTokens}, ...)
  │   ├─ savedSendCounts_ = at::empty({rankSize}, ...)
  │   ├─ savedRecvCounts_ = at::empty({rankSize}, ...)
  │   ├─ savedRecvLocalEntry_ = at::empty({maxR}, ...)
  │   ├─ savedNumRecv_ = at::empty({1}, ...)
  │   ├─ ACLNN_CMD(aclnnEngramFetch, ..., 5 outputs)  ← 同步, 无 wait
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
workspace = sendCounts + recvCounts + sendPrefixSum + recvPrefixSum
          + sortedIndices + tokenIdxRank + recvIndices + localData
          + 16MB系统保留
```

| 分项 | 大小 |
|------|------|
| sendCounts | rankSize × 4B |
| recvCounts | rankSize × 4B |
| sendPrefixSum | rankSize × 8B |
| recvPrefixSum | rankSize × 8B |
| sortedIndices | numTokens × 4B |
| tokenIdxRank | numTokens × 4B |
| recvIndices | totalRecv × 4B |
| localData | totalRecv × hiddenBytes |

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
- `WITH_GRAD_MODE`: all-to-all（5 步融合, 含还原, 无需 wait kernel）

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
  ├─ ACLNN_CMD(aclnnEngramFetchGrad, ..., 3 outputs)  ← GM 已在 write 阶段申请
  ├─ actualK = numUnique.item<int32_t>()               ← D2H 读有效行数
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

### 4.4 InferShape

```
gradUniqueOut = [-1, hiddenSize] fp32（动态 shape, dim0 运行时由 numUnique 决定）
uniqueLocalEntryOut = [-1] int32
numUniqueOut = [1] int32
```

### 4.5 TilingData

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
ENGRAM_FETCH_GRAD_DEFAULT_MODE = 0   // 单一模式
```

### 4.7 Workspace 布局

```
workspace = sendCounts + recvCounts + sendPrefixSum + recvPrefixSum
          + gradSorted + recvGrad + uniqueEntry + gradAccum
          + 16MB系统保留
```

| 分项 | 大小 | 用途 |
|------|------|------|
| sendCounts | rankSize × 4B | a2a 发送计数 |
| recvCounts | rankSize × 4B | a2a 接收计数 |
| sendPrefixSum | rankSize × 8B | 发送前缀和 (int64) |
| recvPrefixSum | rankSize × 8B | 接收前缀和 (int64) |
| gradSorted | numTokens × hiddenBytes | 步骤① gradFetched[perm] (fp32) |
| recvGrad | totalRecv × hiddenBytes | 步骤② a2a 接收梯度 (fp32) |
| uniqueEntry | totalRecv × 4B | 步骤③ unique local entry 中间结果 (int32) |
| gradAccum | totalRecv × hiddenBytes | 步骤③ scatter-add 累加器 (fp32) |

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
| `engram_fetch.h` | `EngramCommContext` 新增 `gmCommBuffer` |
| `engram_fetch_def.cpp` | 5 个 OPTIONAL 输出 |
| `engram_fetch_tiling_data.h` | 新增 `withGrad`/`numMaxTokensPerRank`/`rankSize` |
| `engram_fetch_tiling_key.h` | `INFER_MODE` / `WITH_GRAD_MODE` |
| `engram_fetch_tiling.cpp` | `GetWithGrad`；训练态可选输出校验；精确 workspace |
| `aclnn_engram_fetch.cpp` | 5 个可选输出参数 |
| `elastic_buffer.cpp` | `CommContext`+`gmAddrList`；`EngramContextBuilder`+`withGrad_`成员；GM write 阶段申请/destroy 释放；`EngramFetch` 简化+savedPerm_；`EngramFetchGrad`+numRecv；pybind lambda |
| `elastic_buffer.py` | `EngramFetchCtx`；`with_grad`/`num_max_tokens_per_rank` 构造参数；`engram_fetch` 无分支；`engram_fetch_grad` |

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
