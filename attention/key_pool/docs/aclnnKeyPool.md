# aclnnKeyPool

## 产品支持情况

| 产品 | 是否支持 |
| --- | :---: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Ascend 950PR/Ascend 950DT（A5） | √ |

本版本的实机验证范围为 A2 和 A5。A3 不在本版本验证范围内。

## 功能说明

KeyPool 是推理场景下的 Key 压缩算子。算子对每个输入 token 分别执行 K
投影和 Gate 投影，按 `cmpRatio` 个 token 进行 Gate 加权池化，并将未完成
压缩组所需的中间状态写入分页 `stateCache`。

对当前输入的每个 token：

```text
K = hidden_states @ wk.T
G = hidden_states @ gate_weight.T
K_norm = LayerNorm(K, dim=-1)  # 提供 normWeight/normBias 时
```

对第 `p` 个压缩组：

```text
logits[r] = G[p * cmpRatio + r] + ape[r]
weight = softmax(logits, dim=0)
pooled_key[p] = sum(weight[r] * K_norm[p * cmpRatio + r], dim=0)
```

历史 token 通过 `cacheBlockTable` 从 `stateCache` 读取，当前输入 token 的投影
结果优先使用当前调用产生的数据。`ape` 不写入 `stateCache`。RoPE 参数入口
保留用于后续扩展，但当前版本不实现 RoPE；传入非空 `cos`/`sin` 会返回参数
错误。

LayerNorm 沿每个 token 的最后一维 `D` 独立计算，不跨 token 归约。

## 函数原型

算子采用两段式接口。必须先调用
`aclnnKeyPoolGetWorkspaceSize` 获取 workspace 大小和 executor，再调用
`aclnnKeyPool` 执行计算。

```cpp
aclnnStatus aclnnKeyPoolGetWorkspaceSize(
    const aclTensor *hiddenStates,
    const aclTensor *wk,
    const aclTensor *gateWeight,
    const aclTensor *ape,
    aclTensor       *stateCacheRef,
    const aclTensor *cacheBlockTable,
    const aclTensor *startPos,
    const aclTensor *normWeightOptional,
    const aclTensor *normBiasOptional,
    const aclTensor *cosOptional,
    const aclTensor *sinOptional,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *sequsedOptional,
    int64_t          cmpRatio,
    double           normEps,
    int64_t          rotaryMode,
    int64_t          stateCacheStrideDim0,
    aclTensor       *pooledKeyOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor);
```

```cpp
aclnnStatus aclnnKeyPool(
    void          *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor  *executor,
    aclrtStream     stream);
```

`stateCacheRef` 为原地更新输入，`pooledKeyOut` 是唯一输出。

## aclnnKeyPoolGetWorkspaceSize

### 参数说明

| 参数 | 输入/输出 | 数据类型 | Shape | 说明 |
| --- | --- | --- | --- | --- |
| `hiddenStates` | 输入 | BF16/FP16 | BSH `[B,S,H]` 或 TH `[T,H]` | 当前调用的 token |
| `wk` | 输入 | 与 `hiddenStates` 相同 | `[D,H]` | K 投影权重 |
| `gateWeight` | 输入 | 与 `hiddenStates` 相同 | `[D,H]` | Gate 投影权重 |
| `ape` | 输入 | FP32 | `[R,D]` | `R=cmpRatio` 的位置偏置 |
| `stateCacheRef` | 输入/输出 | FP32 | `[N,BS,2D]` | 前 `D` 列保存 K，后 `D` 列保存原始 Gate |
| `cacheBlockTable` | 输入 | INT32 | `[B,L]` | 逻辑 block 到物理 block 的映射 |
| `startPos` | 输入 | INT32 | `[B]` | 每个 batch 的逻辑起始位置 |
| `normWeightOptional` | 可选输入 | FP32 | `[D]` | 与 `normBiasOptional` 成对出现 |
| `normBiasOptional` | 可选输入 | FP32 | `[D]` | 与 `normWeightOptional` 成对出现 |
| `cosOptional` | 可选输入 | BF16 | 保留接口 | 当前版本必须为空 |
| `sinOptional` | 可选输入 | BF16 | 保留接口 | 当前版本必须为空 |
| `cuSeqlensOptional` | 可选输入 | INT32 | TH 时 `[B+1]` | BSH 时必须为空 |
| `sequsedOptional` | 可选输入 | INT32 | 保留接口 | 当前版本必须为空 |
| `cmpRatio` | 属性 | INT64 | 标量 | 压缩率 |
| `normEps` | 属性 | FLOAT | 标量 | LayerNorm epsilon |
| `rotaryMode` | 属性 | INT64 | 标量 | RoPE 模式入口 |
| `stateCacheStrideDim0` | 属性 | INT64 | 标量 | `stateCache` 第 0 轴 stride |
| `pooledKeyOut` | 输出 | 与 `hiddenStates` 相同 | `[B,Pcap,D]` | 固定容量池化结果 |

其中：

- `R = cmpRatio`；
- `N` 为物理 block 数；
- `BS` 为 block size；
- `L` 为每个 batch 的最大逻辑 block 数；
- `Pcap = ceil(L * BS / cmpRatio)`。

输入 tensor 使用 ND 格式。`stateCacheRef` 是原地输入，不应在两次算子调用
之间被释放或改作其他用途。

### 属性约束

| 属性 | 默认值 | 支持范围 |
| --- | --- | --- |
| `cmpRatio` | 4 | `2/4/8/16/32/64/128` |
| `normEps` | `1e-6` | 大于 0 |
| `rotaryMode` | 1 | `0/1` |
| `stateCacheStrideDim0` | 0 | 由调用方传入实际 stride |

### 返回值

返回 `aclnnStatus`。第一段接口负责参数、shape、dtype 和 workspace 校验，
常见错误码包括：

| 返回值 | 错误码 | 含义 |
| --- | --- | --- |
| `ACLNN_ERR_PARAM_NULLPTR` | 161001 | 必须提供的参数为空 |
| `ACLNN_ERR_PARAM_INVALID` | 161002 | shape、dtype、属性或布局不支持 |
| `ACLNN_ERR_RUNTIME_ERROR` | 361001 | Runtime 或内存申请失败 |
| `ACLNN_ERR_INNER_TILING_ERROR` | 561002 | Tiling 失败 |

## aclnnKeyPool

第二段接口的参数含义与其他 aclnn 两段式接口一致：

| 参数 | 说明 |
| --- | --- |
| `workspace` | Device 侧 workspace 地址 |
| `workspaceSize` | 第一段接口返回的 workspace 大小 |
| `executor` | 第一段接口生成的算子执行器 |
| `stream` | 执行算子的 ACL stream |

## Shape 和布局约束

### hiddenStates

- BSH：`hiddenStates` 为 `[B,S,H]`，`cuSeqlensOptional` 必须为空。
- TH：`hiddenStates` 为 `[T,H]`，`cuSeqlensOptional` 必须为 `[B+1]`。
- TH 下 `cuSeqlens[0] == 0`，元素单调不减，且
  `cuSeqlens[B] == T`。
- `cacheBlockTable` 和 `startPos` 的第 0 维必须等于 `B`。
- TH 输入允许存在长度为 0 的 batch。

### 投影、cache 和输出

- `wk.shape == gateWeight.shape == [D,H]`。
- `ape.shape == [cmpRatio,D]`。
- `stateCache.shape == [N,BS,2D]`，其中 `N > 0` 且 `BS` 在支持范围内。
- `pooledKeyOut.shape == [B,Pcap,D]`，输出 dtype 与 `hiddenStates` 相同。
- 输出容量由 cache table 和 cache block size 决定，而不是由当前调用的
  token 数动态决定。
- 每个 batch 的有效压缩结果写在输出前部；未形成完整压缩组的尾部和剩余
  容量保持为 0。

### 数据类型

- `hiddenStates`、`wk`、`gateWeight`、`pooledKeyOut` 支持 BF16 或 FP16，
  且四者 dtype 一致。
- `stateCacheRef`、`ape`、`normWeightOptional`、`normBiasOptional` 使用 FP32。
- `cacheBlockTable`、`startPos`、`cuSeqlensOptional` 使用 INT32。
- `cosOptional`、`sinOptional` 的接口 dtype 为 BF16，但当前必须为空。

### 形状范围

- `D` 支持 128 或 512。
- `H` 支持 `[1024,10240]`，并要求按 512 对齐。
- `BS` 支持 `[1,1024]`。
- `cmpRatio` 仅支持 `2/4/8/16/32/64/128`。
- `normWeightOptional` 和 `normBiasOptional` 必须同时提供或同时为空，
  shape 均为 `[D]`。

## 语义约束

- `stateCacheRef` 的前半部分保存 K，后半部分保存 Gate；`ape` 不进入 cache。
- 开启 LayerNorm 时，K 投影先沿 `D` 维归一化，再写入 cache 并参与池化。
- `sequsedOptional` 当前仅作接口占位，必须传空指针。
- `cosOptional` 和 `sinOptional` 当前仅作 RoPE 接口占位，必须同时为空。
- `stateCacheStrideDim0` 应与实际 `stateCacheRef` 的第 0 轴 stride 一致；
  当前验证路径使用连续 state cache。
- 当前版本不支持通过 `attention_mask` 挑选 token，调用方应保证输入 token
  已经是连续有效 token。

## PyTorch 调用

安装 torch extension 后，可使用：

```python
torch_npu.key_pool(
    hidden_states,
    wk,
    gate_weight,
    ape,
    state_cache,
    cache_block_table,
    start_pos,
    norm_weight=None,
    norm_bias=None,
    cos=None,
    sin=None,
    cu_seqlens=None,
    seqused=None,
    cmp_ratio=4,
    norm_eps=1e-6,
    rotary_mode=1,
)
```

测试脚本位于 `attention/key_pool/tests/pytest`，覆盖 CPU Golden、BSH/TH、
LayerNorm、历史 cache、跨 block、压缩率、确定性和长序列场景。
