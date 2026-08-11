# SFAG arch35 kvMerge 合并输出设计

## 1. 需求背景

SparseFlashAttentionGrad 在 K、V 共享同一个输入 tensor 的场景下，需要支持 dK 和 dV 的合并输出，以减少 dV workspace 清零、Scatter 写回、Post 读取和 dV GM 写回等开销。

kvMerge 的触发条件为：

```text
value == nullptr
```

此时语义不是把 dK 和 dV 拼接输出，而是表示 K/V 共享同一个原始输入，反向结果需要合并到 dK 输出：

```text
dKV = dK + dV
```

由于 arch35 中 dK 分支来自 mm4，且需要先乘 `scaleValue`，因此实际合并公式为：

```text
dKV = scaleValue * dK_mm4 + dV_mm5
```

注意不能写成：

```text
dKV = scaleValue * (dK_mm4 + dV_mm5)
```

## 2. 目标行为

### 2.1 非 kvMerge 场景

触发条件：

```text
value != nullptr
```

保持当前 arch35 行为不变：

```text
mm4 -> ScatterAdd -> dkWorkspace -> Post -> dkGm
mm5 -> ScatterAdd -> dvWorkspace -> Post -> dvGm
```

具体行为：

- `ScatterAdd` 和 `ScatterAddDeter` 分别将 dK、dV 写入 `dkWorkspace`、`dvWorkspace`。
- `Post` 分别处理 dQ、dK、dV 三路 workspace。
- dK 输出到 `dkGm`，dV 输出到 `dvGm`。
- tiling key 对旧场景保持兼容，`KvMerge = 0` 时不改变已有低位 key 语义。

### 2.2 kvMerge 场景

触发条件：

```text
value == nullptr
```

目标数据流：

```text
mm4 -> dK branch -> Muls(scaleValue) --+
                                       + -> dkWorkspace -> Post -> dkGm
mm5 -> dV branch ----------------------+

dvWorkspace 不清零
dvWorkspace 不写入
dvWorkspace 不读取
dvGm        不输出
```

具体行为：

- Host/Tiling 侧识别 `value == nullptr`，设置 `tmpData.kvMerge = true`。
- Host shape 推导中使用 `keyShape` 作为 `valueShape` fallback。
- Kernel 内部计算仍需要 V 的数值，因此 entry 侧应将 `valueIn` 指向 `key`。
- Scatter 阶段将 `scaleValue * dK_mm4 + dV_mm5` 合并到 `dkWorkspace`。
- Post 阶段只处理 dQ、dK 两路，不再处理 dV。

## 3. Host/Tiling 设计

### 3.1 `TempParams` 增加 `kvMerge`

文件：

```text
op_host/arch35/sparse_flash_attention_grad_tiling_bs1_regbase.h
```

在 `TempParams` 中新增：

```cpp
bool kvMerge = false;
```

建议放在：

```cpp
bool ropeEnable = false;
bool deterministic = false;
bool kvMerge = false;
```

### 3.2 `GetBaseShapeInfo()` 支持 V optional

文件：

```text
op_host/arch35/sparse_flash_attention_grad_tiling_bs1_regbase.cpp
```

当前 arch35 逻辑要求 `QUERY`、`KEY`、`VALUE` 都非空，并直接获取：

```cpp
const gert::Shape &valueShape =
    context_->GetInputShape(static_cast<size_t>(InputIndex::VALUE))->GetStorageShape();
```

需要改为：

```cpp
OP_CHECK_IF(
    context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY)) == nullptr ||
    context_->GetInputShape(static_cast<size_t>(InputIndex::KEY)) == nullptr,
    OPS_REPORT_VECTOR_INNER_ERR(opName, "InputShape of query or key is nullptr."),
    return ge::GRAPH_FAILED);

const gert::Shape &queryShape =
    context_->GetInputShape(static_cast<size_t>(InputIndex::QUERY))->GetStorageShape();
const gert::Shape &keyShape =
    context_->GetInputShape(static_cast<size_t>(InputIndex::KEY))->GetStorageShape();

const gert::StorageShape *valueStorageShape =
    context_->GetOptionalInputShape(static_cast<size_t>(InputIndex::VALUE));
tmpData.kvMerge = (valueStorageShape == nullptr);

const gert::Shape &valueShape =
    tmpData.kvMerge ? keyShape : valueStorageShape->GetStorageShape();
```

这样在 kvMerge 下：

```text
valueShape = keyShape
dimDv = dimDk
```

符合 K/V 共享输入的语义。

### 3.3 输出 shape 校验

当前 arch35 无条件获取并校验 `DV`：

```cpp
const gert::Shape &dvShape =
    context_->GetOutputShape(static_cast<size_t>(OutputIndex::DV))->GetStorageShape();

CheckOutShapeInfo(valueShape, "value", dvShape, inputLayout);
```

kvMerge 场景下 `dv` 不作为有效输出使用，因此不能无条件解引用 `DV`。建议结构：

```cpp
const gert::StorageShape *dvStorageShape =
    context_->GetOutputShape(static_cast<size_t>(OutputIndex::DV));

CheckOutShapeInfo(queryShape, "query", dqShape, inputLayout);
CheckOutShapeInfo(keyShape, "key", dkShape, inputLayout);

if (!tmpData.kvMerge) {
    OP_CHECK_IF(dvStorageShape == nullptr,
        OPS_REPORT_VECTOR_INNER_ERR(opName, "OutputShape of dv is nullptr."),
        return ge::GRAPH_FAILED);
    const gert::Shape &dvShape = dvStorageShape->GetStorageShape();
    CheckOutShapeInfo(valueShape, "value", dvShape, inputLayout);
}
```

校验关系：

```text
kvMerge=false:
    query <-> dq
    key   <-> dk
    value <-> dv

kvMerge=true:
    query <-> dq
    key   <-> dk
    不校验 value/dv
```

### 3.4 TilingKey 增加 KvMerge 模板 bit

文件：

```text
op_kernel/arch35/sparse_flash_attention_grad_template_tiling_key.h
```

当前 arch35 tiling key 已包含：

```text
InputDType
IsTnd
GTemplateNum
S2TemplateNum
DTemplateNum
IsRope
Deterministic
```

建议在最后追加 `KvMerge`，不改变已有字段的 bit 位置：

```cpp
// bit: 35 KvMerge
//      0: DISABLE
//      1: ENABLE
ASCENDC_TPL_BOOL_DECL(KvMerge, 0, 1)
```

selector 同步追加：

```cpp
ASCENDC_TPL_BOOL_SEL(KvMerge, 0, 1),
```

放在最后的原因：

```text
KvMerge = 0 时，旧场景 tiling key 低 35 bit 完全不变。
KvMerge = 1 时，仅新增 bit35 区分合并输出 kernel。
```

### 3.5 `GetTilingKey()` 下传 kvMerge

文件：

```text
op_host/arch35/sparse_flash_attention_grad_tiling_bs1_regbase.cpp
```

当前：

```cpp
GET_TPL_TILING_KEY(
    inputDtypeSize,
    isTnd,
    tmpData.singleM,
    tmpData.singleN,
    tmpData.d,
    tmpData.ropeEnable,
    tmpData.deterministic);
```

改为：

```cpp
GET_TPL_TILING_KEY(
    static_cast<uint8_t>(inputDtypeSize),
    static_cast<uint8_t>(isTnd),
    static_cast<uint16_t>(tmpData.singleM),
    static_cast<uint16_t>(tmpData.singleN),
    static_cast<uint16_t>(tmpData.d),
    static_cast<uint8_t>(tmpData.ropeEnable),
    static_cast<uint8_t>(tmpData.deterministic),
    static_cast<uint8_t>(tmpData.kvMerge));
```

日志同步增加 `KvMerge[%d]`，方便确认 host 已识别场景。

### 3.6 不新增 tilingData runtime 字段

arch35 本次建议只通过 tiling key 将 kvMerge 做成编译期模板参数：

```text
Host tmpData.kvMerge
    -> tiling key bit
    -> kernel template<bool KV_MERGE>
```

不建议在 `SparseFlashAttentionGradTilingDataRegbase` 中新增 runtime 字段，原因：

- 不修改 tiling struct layout/ABI。
- 不改变 workspace offset 和已有 tiling data 布局。
- `if constexpr (KV_MERGE)` 能在编译期消除非目标分支。

## 4. Kernel 模板参数传播

### 4.1 traits 增加 `KV_MERGE`

文件：

```text
op_kernel/arch35/sparse_flash_attention_grad_common.h
```

在 `CUBE_BLOCK_TRAITS_CONST_FIELDS` 中新增：

```cpp
X(KV_MERGE, bool, false)
```

建议位置：

```cpp
#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) \
    X(IS_DROP, bool, false)               \
    X(IS_TND, bool, false)                \
    X(IS_ROPE, bool, false)               \
    X(IS_DETER, bool, false)              \
    X(KV_MERGE, bool, false)              \
    X(s2TemplateType, S2TemplateType, S2TemplateType::Aligned128) \
    X(dTemplateType, DTemplateType, DTemplateType::Aligned128)
```

这样 `FAGBlockVec`、`FAGBlockCube`、kernel base 中均可通过模板常量访问 `KV_MERGE`。

### 4.2 顶层 kernel 模板增加 `kvMerge`

文件：

```text
op_kernel/sparse_flash_attention_grad.cpp
op_kernel/arch35/sparse_flash_attention_grad_entry_regbase.h
```

arch35 顶层模板从：

```cpp
template <
    uint8_t inputDType,
    bool isTnd,
    uint16_t gTemplateType,
    uint16_t s2TemplateType,
    uint16_t dTemplateType,
    bool isRope,
    bool deterministic>
```

扩展为：

```cpp
template <
    uint8_t inputDType,
    bool isTnd,
    uint16_t gTemplateType,
    uint16_t s2TemplateType,
    uint16_t dTemplateType,
    bool isRope,
    bool deterministic,
    bool kvMerge>
```

并在 `RegbaseSFAG` 调用中继续下传。

### 4.3 Entry 中将 `valueIn` 映射到 key

文件：

```text
op_kernel/arch35/sparse_flash_attention_grad_entry_regbase.h
```

kvMerge 下 `value` 指针为空，但 backward 前半段仍需要 V 的数值计算：

```text
dP = dO * V^T
```

因此不能将 null `value` 继续传入 kernel 主体。Entry 侧需要改为：

```cpp
__gm__ uint8_t *valueIn = KV_MERGE ? key : value;
```

后续 `op.Init()` 使用 `valueIn`：

```cpp
op.Init(query, key, valueIn, sparse_indices, d_out, out, ...);
```

这是 kvMerge 功能正确性的关键前置条件。

## 5. Pre 阶段设计

文件：

```text
op_kernel/arch35/sparse_flash_attention_grad_pre_regbase.h
```

当前 Pre 无条件清零：

```cpp
InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
InitOutput<float>(dkWorkSpaceGm[dkOffset], initdkSize, 0);
InitOutput<float>(dvWorkSpaceGm[dvOffset], initdvSize, 0);
```

kvMerge 下 `dvWorkspace` 后续不会被使用，建议模板增加 `KV_MERGE`：

```cpp
template <
    typename T1,
    typename T2,
    const uint32_t IS_TND = 0,
    const bool KV_MERGE = false>
class FlashAttentionGradPreRegbase;
```

Process 中：

```cpp
InitOutput<float>(dqWorkSpaceGm[dqOffset], initdqSize, 0);
InitOutput<float>(dkWorkSpaceGm[dkOffset], initdkSize, 0);

if constexpr (!KV_MERGE) {
    InitOutput<float>(dvWorkSpaceGm[dvOffset], initdvSize, 0);
}
```

## 6. Scatter 阶段设计

文件：

```text
op_kernel/arch35/sparse_flash_attention_grad_block_vec.h
```

需要同时修改：

```text
ScatterAdd
ScatterAddDeter
```

### 6.1 非 kvMerge 行为保持不变

```cpp
DataCopy(dkWorkSpaceGm[...], dkInTensor[...], HEAD_DIM_ALIGN);
DataCopy(dvWorkSpaceGm[...], dvInTensor[...], dSizeV);
```

### 6.2 kvMerge 合并顺序

正确顺序：

```text
1. DataCopy mm4 -> dkInTensor
2. Muls dkInTensor by scaleValue
3. DataCopy mm5 -> dvInTensor
4. Add dvInTensor into dkInTensor
5. DataCopy dkInTensor -> dkWorkspace
6. skip dvWorkspace write
```

伪代码：

```cpp
DataCopy(dkInTensor, mm4ResWorkSpaceGm[...], rows * HEAD_DIM_ALIGN);
Muls(dkInTensor, dkInTensor, (float)constInfo.scaleValue, rows * HEAD_DIM_ALIGN);

DataCopy(dvInTensor, mm5ResWorkSpaceGm[...], rows * dSizeV);

if constexpr (KV_MERGE) {
    for (int64_t row = 0; row < rows; row++) {
        Add(dkInTensor[row * HEAD_DIM_ALIGN],
            dkInTensor[row * HEAD_DIM_ALIGN],
            dvInTensor[row * dSizeV],
            dSizeV);
    }
}

DataCopy(dkOutGm[s2Idx * HEAD_DIM_ALIGN],
         dkInTensor[row * HEAD_DIM_ALIGN],
         HEAD_DIM_ALIGN);

if constexpr (!KV_MERGE) {
    DataCopy(dvOutGm[s2Idx * dSizeV],
             dvInTensor[row * dSizeV],
             dSizeV);
}
```

### 6.3 RoPE 场景 Add 长度

arch35 RoPE 场景下 dK workspace 行内布局为：

```text
| normal K gradient: 512 | K rope gradient: 64 |
```

dV 只有 normal value 维度：

```text
| V gradient: 512 |
```

kvMerge 只能将 dV 加到 normal K 区域，不能污染 rope 区域：

```text
| dK_normal + dV | dK_rope |
```

因此 Add 长度必须使用：

```cpp
constInfo.commonConstInfo.dSizeV
```

不能使用：

```cpp
HEAD_DIM_ALIGN
```

### 6.4 main loop 和 tail loop 都要改

`ScatterAdd` 和 `ScatterAddDeter` 都存在 main loop 和 tail loop。两处必须保持相同语义：

```text
Muls dK
Copy dV
KV_MERGE ? Add dV into dK
write dK
!KV_MERGE ? write dV
```

只改主循环会导致 selected block 数量非整块时尾块结果错误。

### 6.5 deterministic 路径同步不重排

`ScatterAddDeter` 中现有同步机制不应因 kvMerge 重排：

```text
SCATTER_SYNC_FLAG
SCATTER_CUBE_SYNC_FLAG
SCATTER_VEC_SYNC_FLAG
SetAtomicAdd
SetAtomicNone
```

本次只改变每次 atomic add 写入的内容：

```text
kvMerge=false: dK 写 dkWorkspace，dV 写 dvWorkspace
kvMerge=true : dK+dV 写 dkWorkspace，不写 dvWorkspace
```

## 7. Post 阶段设计

文件：

```text
op_kernel/arch35/sparse_flash_attention_grad_post_regbase.h
```

当前 Post 固定处理三路：

```cpp
for (int qkvIdx = 0; qkvIdx < 3; qkvIdx++) {
    // 0: dQ
    // 1: dK
    // 2: dV
}
```

增加 `KV_MERGE` 模板参数：

```cpp
template <
    typename T1,
    typename T2,
    typename OUTDTYPE = T1,
    const bool IS_ROPE = false,
    const bool IS_TND = false,
    const bool KV_MERGE = false>
class SparseFlashAttentionGradPostRegbase;
```

Process 中：

```cpp
constexpr int QKV_NUM = KV_MERGE ? 2 : 3;

for (int qkvIdx = 0; qkvIdx < QKV_NUM; qkvIdx++) {
    ...
}
```

Init 中建议也避免 kvMerge 下绑定 dV 输出：

```cpp
dqkv[0].SetGlobalBuffer((__gm__ OUTDTYPE *)dq);
dqkv[1].SetGlobalBuffer((__gm__ OUTDTYPE *)dk);

if constexpr (!KV_MERGE) {
    dqkv[2].SetGlobalBuffer((__gm__ OUTDTYPE *)dv);
}
```

workspace 同理：

```cpp
dqkvWorkspace[0].SetGlobalBuffer(... dqWorkSpaceOffset ...);
dqkvWorkspace[1].SetGlobalBuffer(... dkWorkSpaceOffset ...);

if constexpr (!KV_MERGE) {
    dqkvWorkspace[2].SetGlobalBuffer(... dvWorkSpaceOffset ...);
}
```

RoPE 场景下 Post 的 K/rope 拆分逻辑不需要重新设计。Scatter 合并完成后，`dkWorkspace` 已经包含：

```text
normal K 区域 = dK + dV
rope K 区域   = dK_rope
```

Post 继续按 `qkvIdx == 1` 输出 dK 和 dK rope 即可，只需要跳过 `qkvIdx == 2`。

## 8. Workspace 设计

第一版不建议收缩 workspace。

仍然保留：

```text
dqWorkspaceLen
dkWorkspaceLen
dvWorkspaceLen
```

并保持原有 offset 布局不变。

原因：

- 避免修改 tiling data layout。
- 避免影响 mm4/mm5、dq/dk/dv workspace offset。
- 降低首版回归风险。
- 本次性能收益主要来自不清零、不写、不读、不 post dV workspace，而不是减少 workspace 申请量。

第一版实际减少的开销：

```text
1. dvWorkspace 清零 GM 写
2. Scatter -> dvWorkspace GM 写
3. Post dvWorkspace GM 读
4. Post -> dvGm GM 写
5. dV cast/post 计算
```

后续如需进一步优化 workspace 占用，可单独设计第二阶段。

## 9. 修改文件清单

| 文件 | 改动 |
| --- | --- |
| `op_host/arch35/sparse_flash_attention_grad_tiling_bs1_regbase.h` | `TempParams` 新增 `kvMerge` |
| `op_host/arch35/sparse_flash_attention_grad_tiling_bs1_regbase.cpp` | V optional、shape fallback、DV 输出校验分支、tiling key 下传 kvMerge |
| `op_kernel/arch35/sparse_flash_attention_grad_template_tiling_key.h` | 新增 `KvMerge` bool template bit |
| `op_kernel/arch35/sparse_flash_attention_grad_common.h` | traits 增加 `KV_MERGE` |
| `op_kernel/sparse_flash_attention_grad.cpp` | arch35 顶层 kernel template 增加 `kvMerge` |
| `op_kernel/arch35/sparse_flash_attention_grad_entry_regbase.h` | 下传 `KV_MERGE`，kvMerge 下 `valueIn = key`，Pre/Post 传模板参数 |
| `op_kernel/arch35/sparse_flash_attention_grad_kernel_base.h` | kvMerge 下不绑定 dV 输出 GM |
| `op_kernel/arch35/sparse_flash_attention_grad_block_vec.h` | `ScatterAdd` 和 `ScatterAddDeter` 合并 dK/dV，只在非 kvMerge 下写 dV workspace |
| `op_kernel/arch35/sparse_flash_attention_grad_pre_regbase.h` | kvMerge 下跳过 dV workspace 清零 |
| `op_kernel/arch35/sparse_flash_attention_grad_post_regbase.h` | kvMerge 下只处理 dQ/dK，跳过 dV post |

不建议修改：

```text
op_kernel/arch35/sparse_flash_attention_grad_tiling_data_regbase.h
```

## 10. 实施顺序

### Phase 1：打通 Host/TilingKey/模板参数

目标：

```text
value == nullptr
-> tmpData.kvMerge = true
-> tiling key 携带 KvMerge bit
-> arch35 kernel template 收到 KV_MERGE
```

这一阶段不改数值计算，优先验证编译和 dispatch 链路。

验收：

- `value != nullptr` 时 `KvMerge=0`。
- `value == nullptr` 时 `KvMerge=1`。
- 两种模板实例均可编译。

### Phase 2：实现非确定性 Scatter 合并

目标：

```text
ScatterAdd:
dkWorkspace += scaleValue * dK_mm4 + dV_mm5
不写 dvWorkspace
```

先验证 `deterministic=false` 场景，方便定位地址和数值问题。

### Phase 3：实现 deterministic Scatter 合并

目标：

```text
ScatterAddDeter:
语义与 ScatterAdd 保持一致
deterministic 同步顺序保持不变
```

重点确认多次运行结果一致。

### Phase 4：Pre/Post 性能收尾

目标：

```text
Pre 跳过 dvWorkspace clear
Post 跳过 dvWorkspace read/cast
Post 不写 dvGm
```

## 11. 测试设计

### 11.1 数学关系测试

同一组输入跑两次：

Case A：普通模式

```text
value = key
kvMerge = false

得到：
dk_normal
dv_normal
```

Case B：合并模式

```text
value = nullptr
kvMerge = true

得到：
dk_merge
```

期望：

```text
dk_merge ~= dk_normal + dv_normal
```

该测试能同时验证：

- `valueIn = key` 是否生效。
- dK 先乘 `scaleValue` 再加 dV。
- Scatter offset 是否正确。
- Post 是否没有重复处理 dV。

### 11.2 回归组合

至少覆盖：

```text
BSND x non-deter x no-rope x kvMerge
BSND x deter     x no-rope x kvMerge
TND  x non-deter x no-rope x kvMerge
TND  x deter     x no-rope x kvMerge
BSND x non-deter x rope    x kvMerge
BSND x deter     x rope    x kvMerge
TND  x non-deter x rope    x kvMerge
TND  x deter     x rope    x kvMerge
```

每类至少保留一个 `kvMerge=false` 用例，验证旧行为不回退。

重点关注：

```text
TND + deterministic + kvMerge
```

覆盖 actualSeqKvLen offset、deterministic scatter 和 shared KV。

```text
RoPE + kvMerge
```

覆盖 dV 只加到 K normal 维度，不污染 K rope 维度。

## 12. 风险点

1. 漏掉 `valueIn = key`

`value == nullptr` 后，kernel 前半段仍需要 V 数值。如果继续传 null，前面计算会出错。

2. 合并顺序错误

必须先：

```text
Muls(dK, scaleValue)
```

再：

```text
Add(dK, dV)
```

3. RoPE Add 长度错误

kvMerge 下 Add 长度必须是 `dSizeV`，不能是 `HEAD_DIM_ALIGN`，否则会污染 dK rope 区域。

4. 只改 `ScatterAdd`，漏掉 `ScatterAddDeter`

arch35 deterministic 是独立路径，两者必须同时实现。

5. Host 仍拒绝 V null

如果 `GetBaseShapeInfo()` 仍然要求 `VALUE` 非空，kernel 侧改动不会被触发。

6. Post 仍访问 dV

kvMerge 下 `dv` 可能为空，Post 不能进入 `qkvIdx == 2` 分支，也不应绑定或读写 dV 相关 GM。

## 13. 验收标准

```text
1. value != nullptr
   -> KV_MERGE=false
   -> 原有行为、精度、tiling key 低位保持不变

2. value == nullptr
   -> KV_MERGE=true
   -> kernel 内部 value 实际指向 key

3. ScatterAdd
   -> dKWorkspace += scaleValue * dK_mm4 + dV_mm5
   -> 不写 dVWorkspace

4. ScatterAddDeter
   -> 与 ScatterAdd 语义一致
   -> deterministic 结果保持稳定

5. Pre
   -> 不清 dVWorkspace

6. Post
   -> 只处理 dQ/dK
   -> 不读取 dVWorkspace
   -> 不写 dVGm

7. RoPE
   -> dV 只加到 dK normal 部分
   -> dK rope 部分不参与 merge

8. TND
   -> valueShape 使用 keyShape fallback
   -> actualSeqKvLen 相关 offset 逻辑保持不变

9. Workspace
   -> 第一版保留 dvWorkspaceLen 和原 offset
   -> 不做 workspace layout 重构
```
