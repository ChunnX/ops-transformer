# aclnnUndGenQkvRmsNormRopeCache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                 |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term> |      ×     |
| <term>Atlas 推理系列产品</term> |      ×     |
| <term>Atlas 训练系列产品</term> |      ×     |

## 功能说明

- **算子功能**：把 undecoded 段与 generated 段的 QKV 按 `catIndices` 拼接寻址，逐 token 完成
  Split + RMSNorm(Q/K) + MRoPE(Q/K) 融合计算；Q 作为独立输出，K/V 经 Cast float32→bf16 后按
  `slotMapping` 写入分页 KV Cache（原地更新）。

- **计算公式**：

  对每个输出位置 $out\_t$，令 $src\_t = catIndices[out\_t]$（未提供 `catIndices` 时 $src\_t = out\_t$），
  $isUnd = (src\_t < und\_len)$：

  $$
  x = \begin{cases} undQkv[src\_t] & isUnd \\ genQkv[src\_t - und\_len] & \text{otherwise} \end{cases}
  $$

  按 N 维拆分得到 $q, k, v$，其中 $q \in \mathbb{R}^{H_q \times D}$，$k \in \mathbb{R}^{H_k \times D}$，
  $v \in \mathbb{R}^{H_v \times D}$。

  RMSNorm：

  $$
  \mathrm{RmsNorm}(x, w) = \frac{x}{\sqrt{\frac{1}{D}\sum_{i=0}^{D-1} x_i^2 + \epsilon}} \odot w
  $$

  MRoPE：对 half 索引 $i \in [0, D/2)$，按 `mropeSection` 选轴

  $$
  axis(i) = \begin{cases}
    1 & i \bmod 3 = 1 \ \text{且}\ i < 3 \cdot sec_1 \\
    2 & i \bmod 3 = 2 \ \text{且}\ i < 3 \cdot sec_2 \\
    0 & \text{otherwise}
  \end{cases}
  $$

  取 $p = positions[axis(i), out\_t]$，$c_i = cosSinCache[p, i]$，$s_i = cosSinCache[p, i + D/2]$，则

  $$
  y_i = x_i c_i - x_{i + D/2} s_i, \qquad y_{i + D/2} = x_{i + D/2} c_i + x_i s_i
  $$

  输出：

  $$
  q[out\_t] = \mathrm{RoPE}(\mathrm{RmsNorm}(q, w_q)),\quad
  kCache[\lfloor slot/Bs \rfloor, slot \bmod Bs] = \mathrm{RoPE}(\mathrm{RmsNorm}(k, w_k)),\quad
  vCache[\lfloor slot/Bs \rfloor, slot \bmod Bs] = v
  $$

  其中 $slot = slotMapping[out\_t]$，$w_q, w_k$ 按 $isUnd$ 在 und/gen 两套权重间选择。

## 函数原型

每个算子分为[两段式接口](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkflow)，必须先调用
`aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize` 接口获取入参并根据计算流程所需workspace大小，再调用
`aclnnUndGenQkvRmsNormRopeCache` 接口执行计算。

```c++
aclnnStatus aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize(
    const aclTensor* undQkv,
    const aclTensor* undWeightsQ,
    const aclTensor* undWeightsK,
    const aclTensor* cosSinCache,
    aclTensor*       kCache,
    aclTensor*       vCache,
    const aclTensor* slotMapping,
    const aclTensor* positions,
    const aclTensor* genQkv,
    const aclTensor* genWeightsQ,
    const aclTensor* genWeightsK,
    const aclTensor* catIndices,
    int64_t          numHeadsQ,
    int64_t          numHeadsK,
    int64_t          numHeadsV,
    double           normEps,
    const aclIntArray* mropeSection,
    aclTensor*       q,
    aclTensor*       kCacheOut,
    aclTensor*       vCacheOut,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor);
```

```c++
aclnnStatus aclnnUndGenQkvRmsNormRopeCache(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream);
```

## aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize

- **参数说明**：

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度 |
| :----- | :-------- | :--- | :------- | :------- | :------- | :--- |
| undQkv | 输入 | undecoded 段 QKV | 必选 | BFLOAT16 | ND | [und_len, N, D] |
| undWeightsQ | 输入 | und 段 Q 的 RMSNorm 权重 | 必选 | BFLOAT16 | ND | [D] |
| undWeightsK | 输入 | und 段 K 的 RMSNorm 权重 | 必选 | BFLOAT16 | ND | [D] |
| cosSinCache | 输入 | cos/sin 缓存，前半 cos 后半 sin | 必选 | FLOAT32 | ND | [max_pos, D] |
| kCache | 输入/输出 | K Cache，调用方预分配、算子原地写入 | 必选，内存连续 | BFLOAT16 | ND | [Bn, Bs, Hk, D] |
| vCache | 输入/输出 | V Cache，调用方预分配、算子原地写入 | 必选，内存连续 | BFLOAT16 | ND | [Bn, Bs, Hv, D] |
| slotMapping | 输入 | 每个 token 的 cache 行偏移 | 必选，值唯一且 < Bn*Bs | INT64 | ND | [T] |
| positions | 输入 | MRoPE 三轴位置 | 必选，值 < max_pos | INT64 | ND | [3, T] |
| genQkv | 输入 | generated 段 QKV | **当前必填**，gen_len > 0 | BFLOAT16 | ND | [gen_len, N, D] |
| genWeightsQ | 输入 | gen 段 Q 的 RMSNorm 权重 | **当前必填** | BFLOAT16 | ND | [D] |
| genWeightsK | 输入 | gen 段 K 的 RMSNorm 权重 | **当前必填** | BFLOAT16 | ND | [D] |
| catIndices | 输入 | out_t → src_t 映射 | **当前必填** | INT64 | ND | [T] |
| numHeadsQ | 输入 | Q 头数 Hq | 必选，> 0 | INT64 | - | - |
| numHeadsK | 输入 | K 头数 Hk | 必选，> 0 | INT64 | - | - |
| numHeadsV | 输入 | V 头数 Hv | 必选，> 0 | INT64 | - | - |
| normEps | 输入 | RMSNorm epsilon | 必选，> 0，推荐 1e-6 | DOUBLE（内部转 FLOAT） | - | - |
| mropeSection | 输入 | MRoPE 三轴分段 | 可选，nullptr/空表示 [D/2, 0, 0] | INT64 数组 | - | [3] |
| q | 输出 | 处理后的 Q | - | BFLOAT16 | ND | [T, Hq, D] |
| kCacheOut | 输出 | 与 kCache 同地址 | - | BFLOAT16 | ND | 同 kCache |
| vCacheOut | 输出 | 与 vCache 同地址 | - | BFLOAT16 | ND | 同 vCache |
| workspaceSize | 输出 | 返回需要在 Device 侧申请的 workspace 大小 | - | uint64_t | - | - |
| executor | 输出 | 返回 op 执行器 | - | aclOpExecutor** | - | - |

其中 N = Hq + Hk + Hv，T = und_len + gen_len。

- **返回值**：

| 返回码 | 错误码 | 说明 |
| :----- | :----- | :--- |
| ACLNN_SUCCESS | 0 | 执行成功 |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选输入/输出为空指针；或提供 genQkv 但未提供 genWeightsQ/genWeightsK |
| ACLNN_ERR_PARAM_INVALID | 161002 | kCache/vCache 非连续；numHeadsQ/K/V 非正、normEps 非正、mropeSection 长度非 0/3、shape 不满足约束 |

## aclnnUndGenQkvRmsNormRopeCache

- **参数说明**：

| 参数名 | 输入/输出 | 描述 |
| :----- | :-------- | :--- |
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址 |
| workspaceSize | 输入 | 在 Device 侧申请的 workspace 大小，由第一段接口返回 |
| executor | 输入 | op 执行器，由第一段接口返回 |
| stream | 输入 | 指定执行任务的 stream |

- **返回值**：`aclnnStatus`，返回状态码。

## 约束说明

- 仅支持 <term>Ascend 950PR/Ascend 950DT</term>。
- headDim 固定为 128；(Hq, Hk, Hv) 支持 (8, 1, 1) 与 (16, 2, 2)；cache block_size 为 128。
- T = und_len + gen_len 不设上限，只要求为正；真实上限是 KV Cache 容量，要求 `Bn * Bs >= T`。上板已验证到 T = 64K。
- `numHeadsQ + numHeadsK + numHeadsV` 必须等于 `undQkv` 的 N 维；`kCache` 的 Hk、`vCache` 的 Hv 必须与属性一致。
- `slotMapping` 取值必须唯一，重复 slot 会造成多核写冲突、结果不确定。
- `kCache`/`vCache` 必须是连续的 BBND 布局，不支持非连续（BNBD）物理布局。
- `sum(mropeSection)` 不得超过 D/2。
- **当前实现要求 `genQkv`、`genWeightsQ`、`genWeightsK`、`catIndices` 全部传入且 `gen_len > 0`**；
  传 nullptr 的退化路径（纯 prefill、恒等映射）暂不支持，会在 tiling 阶段返回失败。
  接口保留可选语义，后续版本放开。

## 调用示例

详见 [test_aclnn_und_gen_qkv_rms_norm_rope_cache.cpp](../examples/test_aclnn_und_gen_qkv_rms_norm_rope_cache.cpp)。
