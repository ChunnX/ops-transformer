# MoeEpDispatchEpilogue hasTopkWeights / recvTopkWeights 可选化改动说明

## 1. 改动背景

`moe_ep_dispatch_epilogue` 原先默认认为 `recv_topk_weights` 一定存在：

- OpDef 中 `recv_topk_weights` 是 `REQUIRED` 输出；
- aclnn 第一段接口会对 `recvTopkWeights` 做非空检查；
- tiling 阶段无条件校验 `recvTopkWeights` 的 shape 和 dtype；
- kernel 无条件从 window metadata 的 topk weight 区域读取，并写入 `recvTopkWeights`；
- Python / torch extension 固定分配并返回 `recv_topk_weights`。

本次需求是：

> dispatch_epilogue 添加 `has_topkweight`，只有传了 `recvtopkweight` 才输出 `recvtopkweight`，并将 `recvtopkweight` 变为可选参数。

因此本次改动围绕一个统一语义展开：

- `recvTopkWeights != nullptr` / `recv_topk_weights is not None` 表示 `hasTopkWeights = true`；
- `recvTopkWeights == nullptr` / `recv_topk_weights is None` 表示不请求 topk weight 输出；
- 不请求时，host 侧不校验该输出，kernel 侧不读 topk weight、不写 `recvTopkWeights`。

## 2. 行为变化总览

| 场景 | 修改前 | 修改后 |
| --- | --- | --- |
| OpDef 输出定义 | `recv_topk_weights` 必选 | `recv_topk_weights` 可选 |
| aclnn 参数检查 | `recvTopkWeights == nullptr` 报错 | `recvTopkWeights == nullptr` 合法 |
| tiling 校验 | 无条件校验 `recvTopkWeights` | 仅存在时校验，并写入 `hasTopkWeights` |
| tiling key | 只区分 `cached` / `mxQuant` | 新增 `hasTopkWeights` 维度 |
| kernel | 无条件读取和输出 topk weight | `if constexpr (HasTopkWeights)` 控制读写 |
| torch schema | `Tensor recv_topk_weights` | `Tensor? recv_topk_weights` |
| torch 返回值 | 第二项固定 Tensor | 第二项变为 `Tensor?` |
| ElasticBuffer | 固定分配 `recv_topk_weights` | 仅有 `topk_weights` 输入时分配 |

## 3. 逐文件改动说明

### 3.1 `mc2/moe_ep_dispatch_epilogue/op_host/moe_ep_dispatch_epilogue_def.cpp`

**改动点：**

将输出 `recv_topk_weights` 的参数类型从必选改为可选：

```cpp
this->Output("recv_topk_weights")
    .ParamType(OPTIONAL)
    .DataTypeList({ge::DT_FLOAT})
    .FormatList({ge::FORMAT_ND});
```

**原因：**

OpDef 是图侧接口定义。如果这里仍然声明为 `REQUIRED`，上层即使传空也会被图构建/参数检查拦截，无法进入 tiling 和 kernel 的可选逻辑。

**影响：**

- 传入 `recv_topk_weights` 时行为保持原有输出逻辑；
- 不传 `recv_topk_weights` 时图侧允许该输出为空。

---

### 3.2 `mc2/moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue_tiling.h`

**改动点：**

在 `MoeEpDispatchEpilogueInfo` 中新增字段：

```cpp
uint32_t hasTopkWeights = 0;
```

**原因：**

tiling data 是 host tiling 和 kernel 之间的契约。新增字段用于记录当前执行实例是否请求了 topk weight 输出。

**影响：**

- host 侧根据输出是否存在设置该字段；
- 日志和后续调试可直接看到 `hasTopkWeights` 状态；
- 当前 kernel 主要通过 tiling key 的模板参数做编译期分支，该字段作为运行时 tiling 信息保留。

---

### 3.3 `mc2/moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue_tiling_key.h`

**改动点：**

在 tiling key 模板参数中新增 bool 维度：

```cpp
ASCENDC_TPL_BOOL_DECL(HAS_TOPK_WEIGHTS, 0, 1)
```

并在模板选择中同步增加：

```cpp
ASCENDC_TPL_BOOL_SEL(HAS_TOPK_WEIGHTS, 0, 1)
```

**原因：**

kernel 中希望通过 `if constexpr (HasTopkWeights)` 消除不需要的 topk weight 读写逻辑，因此 `hasTopkWeights` 必须进入 tiling key，作为 kernel 模板实例化参数。

**影响：**

原来的 tiling key 维度：

```text
ARCH_TAG, IS_CACHED, IS_MXQUANT
```

现在变为：

```text
ARCH_TAG, IS_CACHED, HAS_TOPK_WEIGHTS, IS_MXQUANT
```

也就是说，同一组 cached / mxQuant 场景下，会进一步按是否输出 topk weight 生成不同 kernel 分支。

---

### 3.4 `mc2/moe_ep_dispatch_epilogue/op_host/op_tiling/moe_ep_dispatch_epilogue_tiling.cpp`

#### 3.4.1 日志新增 `hasTopkWeights`

**改动点：**

`PrintTilingDataInfo` 中新增输出：

```cpp
hasTopkWeights=%u
```

**原因：**

便于从 tiling 日志确认本次走的是输出 topk weight 分支还是跳过分支。

#### 3.4.2 `recvTopkWeights` 从无条件校验改为条件校验

**修改前：**

- 无条件获取 `OUT_RECV_TOPK_WEIGHTS_INDEX` 的 shape；
- 无条件要求 shape 为 1D；
- 无条件要求 dim0 等于 `recvX` dim0；
- 无条件要求 dtype 为 `DT_FLOAT`。

**修改后：**

先判断输出是否存在：

```cpp
const bool hasTopkWeights = (context->GetOutputDesc(OUT_RECV_TOPK_WEIGHTS_INDEX) != nullptr);
info.hasTopkWeights = hasTopkWeights ? 1U : 0U;
```

仅当 `hasTopkWeights == true` 时才校验：

- `recvTopkWeights` shape 必须为 1D；
- dim0 必须等于 `recvX` dim0；
- dtype 必须为 `DT_FLOAT`。

**原因：**

`recvTopkWeights` 已经变成 optional output，不存在时不应该访问其 shape/desc，也不应该报错。

**影响：**

- `recvTopkWeights` 为空：tiling 继续执行，并设置 `info.hasTopkWeights = 0`；
- `recvTopkWeights` 非空：保持原有 shape/dtype 约束。

#### 3.4.3 tiling key 新增 `hasTopkWeights`

**修改前：**

```cpp
GET_TPL_TILING_KEY(TILINGKEY_TPL_A5, cached, isMxQuant)
```

**修改后：**

```cpp
GET_TPL_TILING_KEY(TILINGKEY_TPL_A5, cached, hasTopkWeights, isMxQuant)
```

**原因：**

kernel 模板新增了 `HasTopkWeights` 参数，tiling key 必须同步增加该维度，否则 host 侧选择的 key 与 kernel 模板签名不一致。

**影响：**

- 请求 topk weight 输出时，选择 `HasTopkWeights = true` 的 kernel 实例；
- 不请求 topk weight 输出时，选择 `HasTopkWeights = false` 的 kernel 实例。

---

### 3.5 `mc2/moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue.cpp`

**改动点：**

kernel 入口模板参数新增 `HasTopkWeights`：

```cpp
template<uint32_t ArchTag, bool IsCached, bool HasTopkWeights, bool IsMxQuant>
```

并将该参数传给实现类：

```cpp
KernelDispatchEpilogue<DTYPE_RECV_X, ScalesType, IsCached, HasTopkWeights> op;
```

**原因：**

这是 tiling key 到 kernel 实现类的桥接。`HasTopkWeights` 进入 kernel 后，才能在 `moe_ep_dispatch_epilogue.h` 中使用编译期分支控制 topk weight 相关逻辑。

**影响：**

kernel 实例化矩阵新增一维：

- `HasTopkWeights = true`：保留原有 topk weight 输出行为；
- `HasTopkWeights = false`：跳过 topk weight GM buffer、UB buffer、metadata 读取和输出写回。

---

### 3.6 `mc2/moe_ep_dispatch_epilogue/op_kernel/moe_ep_dispatch_epilogue.h`

这是本次 kernel 行为变化的核心文件。

#### 3.6.1 实现类模板新增参数

**改动点：**

```cpp
template <typename XType, typename ScalesType, uint32_t IsCached, uint32_t HasTopkWeights>
class KernelDispatchEpilogue { ... };
```

所有成员函数定义也同步补齐 `HasTopkWeights` 模板参数。

**原因：**

让 `HasTopkWeights` 成为编译期常量，使不需要 topk weight 的场景可以直接编译掉相关读写逻辑。

#### 3.6.2 `recvTopkWeightsGm_` 仅在需要时绑定

**修改前：**

```cpp
recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float*)recvTopkWeights);
```

**修改后：**

```cpp
if constexpr (HasTopkWeights) {
    recvTopkWeightsGm_.SetGlobalBuffer((__gm__ float*)recvTopkWeights);
}
```

**原因：**

当用户不传 `recvTopkWeights` 时，入参可能为空指针，不能再无条件绑定为 GM tensor。

#### 3.6.3 `ubStageWeights_` buffer 仅在需要时申请和获取

**修改前：**

无论是否需要输出 topk weight，都申请 `ubStageWeightsBuf_` 并获取 `ubStageWeights_`。

**修改后：**

```cpp
if constexpr (HasTopkWeights) {
    uint32_t ubStageWeightsBytes = ...;
    tpipe_->InitBuffer(ubStageWeightsBuf_, ubStageWeightsBytes);
}
```

以及：

```cpp
if constexpr (HasTopkWeights) {
    ubStageWeights_ = ubStageWeightsBuf_.Get<float>();
}
```

**原因：**

不输出 topk weight 时，该 UB 临时区完全无用，跳过可减少 UB 占用，并避免对未初始化 buffer 的误用。

#### 3.6.4 从 window metadata 读取 topk weight 改为条件读取

**修改前：**

```cpp
float weights = ubMeta_.ReinterpretCast<float>().GetValue(metaBase + axisKAlign_ + topkIdx);
ubStageWeights_.SetValue(stageIdx, weights);
```

**修改后：**

```cpp
if constexpr (HasTopkWeights) {
    float weights = ubMeta_.ReinterpretCast<float>().GetValue(metaBase + axisKAlign_ + topkIdx);
    ubStageWeights_.SetValue(stageIdx, weights);
}
```

**原因：**

当 dispatch 阶段没有写 topk weight，window 中对应区域可能没有有效数据。`HasTopkWeights = false` 时不应该读取该区域。

#### 3.6.5 写出 `recvTopkWeights` 改为条件写出

**修改前：**

```cpp
DataCopyPad(recvTopkWeightsGm_[expertRowStart], ubStageWeights_, weightOutParams);
```

**修改后：**

```cpp
if constexpr (HasTopkWeights) {
    DataCopyPad(recvTopkWeightsGm_[expertRowStart], ubStageWeights_, weightOutParams);
}
```

**原因：**

`recvTopkWeights` 可为空，不能无条件写出。

#### 3.6.6 未改变的逻辑

本次未改变以下逻辑：

- `recvX` 的拷贝逻辑；
- FP8 场景下 `recvScales` 的拷贝逻辑；
- `recvSrcMetadata` 的生成/拷贝逻辑；
- cached / non-cached 对 metadata 的处理方式；
- hit count、prefix sum、expert 排序逻辑。

也就是说，`HasTopkWeights = false` 只影响 topk weight 输出链路，不影响 token 数据和 metadata 的输出。

---

### 3.7 `mc2/moe_ep_dispatch_epilogue/op_api/aclnn_moe_ep_dispatch_epilogue.h`

**改动点：**

接口注释中将 `recvTopkWeights` 标记为可选：

```cpp
OPTIONAL [A_alloc] float; nullptr if topk weights are not requested
```

**原因：**

aclnn 头文件是调用方可见的接口契约，需要与实现同步。

**影响：**

调用方可以明确知道：

- 需要 topk weight 输出时传入 `[A_alloc]` float tensor；
- 不需要时传 `nullptr`。

---

### 3.8 `mc2/moe_ep_dispatch_epilogue/op_api/aclnn_moe_ep_dispatch_epilogue.cpp`

**改动点：**

`CheckNotNull` 不再检查 `recvTopkWeights` 非空。

**修改前：**

```cpp
CHECK_RET(recvTopkWeights != nullptr, ACLNN_ERR_PARAM_NULLPTR);
```

**修改后：**

该检查被移除，`CheckNotNull` 只保留必选参数检查：

- `context`
- `dstBufferSlotIdx`
- `numRecvTokensPerRank`
- `numRecvTokensPerExpert`
- `recvX`
- `recvSrcMetadata`

**原因：**

`recvTopkWeights` 已经变成 optional output，第一段接口不能再把空指针视为参数错误。

**影响：**

`recvTopkWeights == nullptr` 时，接口可以继续调用 inner aclnn 流程，由 OpDef/tiling/kernel 的 optional 逻辑处理。

---

### 3.9 `mc2/moe_ep_dispatch_epilogue/tests/ut/op_api/test_aclnn_moe_ep_dispatch_epilogue.cpp`

**改动点：**

新增用例：

```cpp
TestEpilogueWithoutTopkWeights
```

该用例：

- 不构造 `recvTopkWeightsDesc`；
- `OUTPUT` 中第二个输出传 `nullptr`；
- 期望 `aclnnMoeEpDispatchEpilogueGetWorkspaceSize` 返回 `ACLNN_SUCCESS`。

**原因：**

覆盖本次核心新增能力：`recvTopkWeights` 为空时，接口不应再报空指针错误。

**影响：**

后续如果有人重新把 `recvTopkWeights` 改回必选，或恢复 `CheckNotNull` 中的非空检查，该用例会失败。

---

### 3.10 `torch_extension/cann_ops_transformer/ops/csrc/moe_ep_dispatch_epilogue.cpp`

**改动点：**

#### 返回类型第二项改为 optional

**修改前：**

```cpp
using tensor_list = std::tuple<at::Tensor, at::Tensor, at::Tensor, c10::optional<at::Tensor>>;
```

**修改后：**

```cpp
using tensor_list = std::tuple<at::Tensor, c10::optional<at::Tensor>, at::Tensor, c10::optional<at::Tensor>>;
```

#### 输入参数 `recv_topk_weights` 改为 optional

**修改前：**

```cpp
at::Tensor &recv_topk_weights
```

**修改后：**

```cpp
const c10::optional<at::Tensor> &recv_topk_weights
```

#### 调 aclnn 时为空则传 empty tensor

新增：

```cpp
at::Tensor recv_topk_weights_tensor = recv_topk_weights.has_value() ? *recv_topk_weights : at::Tensor();
```

并将 `recv_topk_weights_tensor` 传给 `ACLNN_CMD`。

#### 返回值同步 optional

如果输入有 `recv_topk_weights`，返回该 tensor；否则返回 `None`。

**原因：**

torch extension 是 Python op 到 aclnn 的桥接层，必须允许 Python 侧传 `None`，并将该语义传递给 aclnn。

**影响：**

- Python 传 tensor：返回 tuple 第二项为 tensor；
- Python 传 `None`：返回 tuple 第二项为 `None`。

---

### 3.11 `torch_extension/cann_ops_transformer/ops/moe_ep_dispatch.py`

**改动点：**

#### schema 中 `recv_topk_weights` 改为 optional

**修改前：**

```text
Tensor recv_topk_weights
-> (Tensor, Tensor, Tensor, Tensor?)
```

**修改后：**

```text
Tensor? recv_topk_weights
-> (Tensor, Tensor?, Tensor, Tensor?)
```

#### meta 函数返回 optional topk weight

新增：

```python
recv_topk_weights_out = recv_topk_weights if recv_topk_weights is not None else None
```

并返回：

```python
(recv_x, recv_topk_weights_out, recv_src_metadata, recv_scales_out)
```

**原因：**

PyTorch dispatcher 的 schema 必须准确描述参数和返回值是否允许 `None`，否则 Python 调用层无法合法传入 `None`。

**影响：**

`torch.ops.cann_ops_transformer.npu_moe_ep_dispatch_epilogue` 支持：

```python
recv_topk_weights=None
```

且返回 tuple 第二项为 `None`。

---

### 3.12 `torch_extension/cann_ops_transformer/ops/elastic_buffer.py`

**改动点：**

`recv_topk_weights` 的分配从固定分配改为条件分配：

**修改前：**

```python
recv_topk_weights = torch.empty((actual_a,), dtype=torch.float32, device=x.device)
```

**修改后：**

```python
recv_topk_weights = torch.empty((actual_a,), dtype=torch.float32, device=x.device) if topk_weights is not None else None
```

**原因：**

ElasticBuffer 是上层使用入口。如果当前 dispatch 调用没有 `topk_weights` 输入，就不应再强制要求 epilogue 输出 `recv_topk_weights`。

**影响：**

- 非 cached 路径：当前代码仍要求 `topk_idx` 和 `topk_weights` 同时存在，因此会分配并返回 `recv_topk_weights`，行为与旧逻辑一致；
- cached 路径：默认没有新的 `topk_weights` 输入，因此不分配 `recv_topk_weights`，epilogue 返回 `None`，避免从 window 中读取无效 topk weight；
- 如果未来 cached 路径显式传入 `topk_weights`，该逻辑也会按存在 topk weight 的路径分配输出。

---

### 3.13 `mc2/moe_ep_dispatch_epilogue/docs/aclnnMoeEpDispatchEpilogue.md`

**改动点：**

- 功能公式中标注 `recvTopkWeights` 是 optional；
- 参数表中将 `recvTopkWeights` 的使用说明从“不支持空Tensor”改为“Optional参数，未请求topk权重输出时传入空指针；非空时shape为(A)”。

**原因：**

用户文档需要与新接口语义一致，避免调用方继续认为 `recvTopkWeights` 必须传入。

**影响：**

文档明确说明：不需要 topk weight 输出时，`recvTopkWeights` 可以为空。

## 4. hasTopkWeights 数据流

本次新增的数据流如下：

```text
调用方是否传 recvTopkWeights
        │
        ▼
op_host tiling: context->GetOutputDesc(OUT_RECV_TOPK_WEIGHTS_INDEX) != nullptr
        │
        ├── 写入 tilingData.moeEpDispatchEpilogueInfo.hasTopkWeights
        │
        └── 写入 tiling key 的 HAS_TOPK_WEIGHTS 维度
                │
                ▼
op_kernel.cpp: 模板参数 HasTopkWeights
                │
                ▼
op_kernel.h: if constexpr (HasTopkWeights)
        ├── true  : 绑定 recvTopkWeights GM、申请 UB、读取 weight、写出 weight
        └── false : 跳过上述全部 topk weight 逻辑
```

## 5. 兼容性说明

### 5.1 保持兼容的场景

如果调用方仍然传入 `recvTopkWeights`：

- shape / dtype 约束不变；
- kernel 仍然读取 window 中的 topk weight；
- 输出 `recvTopkWeights` 的行为不变；
- Python 返回 tuple 第二项仍然是 tensor。

### 5.2 新增支持的场景

如果调用方不传 `recvTopkWeights`：

- aclnn 不再报空指针；
- tiling 不再访问该输出的 desc/shape；
- kernel 不再读取 topk weight 区域；
- kernel 不再写出 `recvTopkWeights`；
- Python 返回 tuple 第二项为 `None`。

### 5.3 对 cached 路径的意义

`moe_ep_dispatch` 的 cached 路径可能没有新的 `topk_weights` 输入，因此 window 中的 topk weight 区域不一定有有效数据。旧 epilogue 无条件读取该区域，存在读取无效数据的风险。

本次改动后，cached 路径在不传 `recvTopkWeights` 时会走 `HasTopkWeights = false` 分支，直接跳过 topk weight 读取和输出。

## 6. 验证情况

已完成的检查：

- `git diff --check` 通过，无 whitespace error；
- Python 侧改动文件语法检查通过；
- 用户已在本地确认编译通过。

新增/更新的测试覆盖：

- 新增 `TestEpilogueWithoutTopkWeights`，验证 `recvTopkWeights` 输出为 `nullptr` 时第一段接口返回成功。

## 7. 注意事项

1. `recvTopkWeights` 可选只表示“不输出 topk weight”。它不改变 `recvX`、`recvSrcMetadata`、`recvScales` 的输出语义。
2. `HasTopkWeights = false` 时，kernel 不会读取 metadata 中的 weight 区域，因此可以规避 dispatch 阶段未写入 topk weight 时的无效读取。
3. 如果调用方后续需要 combine 阶段使用 topk weight，则仍应在 dispatch/epilogue 链路中传入并保留 topk weight。
4. 当前文档和 schema 已将返回 tuple 第二项改为 optional，Python 调用方需要处理 `None` 返回。
