# PR7350 Low Latency Expertscale 适配详细设计

## 1. 背景

`MoeDistributeBuffer.low_latency_dispatch` 和 `MoeDistributeBuffer.low_latency_combine` 是 DeepEP 低时延 MoE 通信链路在 Python 侧暴露的成对接口。dispatch 阶段根据 `topk_idx` 将 token 分发到对应专家所在 rank，combine 阶段再按 dispatch 产生的辅助信息把专家输出原路回收，并对同一个 token 的 topK 专家结果做聚合。

原有链路里，dispatch 只发送 token 特征、量化 scale、路由辅助信息等通信必需数据；路由权重 `topk_weights` 不随 token 一起走 dispatch 通信。这样会带来两个问题：

1. combine 阶段如果需要做 `sum(topk_weight * expert_out)`，必须依赖调用者额外保存并传入原始 `topk_weights`。
2. token 已经在 dispatch 侧按专家和 rank 重排，权重如果仍停留在原始 `(BS, K)` 顺序，combine/kernel 侧还需要额外处理顺序映射，接口语义不够闭环。

PR7350 的目标是把 `topk_weights` 纳入低时延 dispatch/combine 链路：

- dispatch 使能 expertscale：发送 token 时同时携带该 token 对应 topK 专家的权重。
- dispatch 输出 `expand_scales`：权重按 dispatch 后的接收/重排顺序输出。
- combine 的 expertscale 可选：传入权重时做加权聚合；不传权重时不做权重缩放，等价 scale 全 1。

命名关系：

```text
Python API: topk_weights
底层算子:  expert_scales / expertScales
语义:      token -> topK expert 的路由权重
```

## 2. 需求拆解

### 2.1 功能需求

Dispatch 侧：

- `low_latency_dispatch(..., topk_weights=Tensor)` 支持传入 shape 为 `(BS, K)` 的 float32 权重。
- 底层 `npu_moe_distribute_dispatch` 将该 Tensor 作为 optional `expert_scales` 传给 ACLNN dispatch 算子。
- 算子把权重作为 token 伴随数据写入通信窗口，并随 token 发送到目标 rank。
- 接收侧按 dispatch 后顺序生成 `expand_scales`。

Combine 侧：

- `low_latency_combine(..., topk_weights=Tensor)` 保持加权求和能力。
- `low_latency_combine(..., topk_weights=None)` 合法，表示不使能 expertscale。
- Python 侧用空 Tensor 表示 `None`，避免底层 combine schema 从必选 Tensor 改成 optional Tensor，降低 ABI 变化风险。
- 算子侧识别空 Tensor 后不访问 `expert_scales` GM 地址，也不执行乘 1，直接对专家输出求和。

### 2.2 兼容需求

- 老调用不传 dispatch `topk_weights` 时，`expand_scales` 返回空 Tensor，不产生额外权重拷贝。
- combine 旧链路如果传真实 `topk_weights`，结果不变。
- combine 新链路如果传 `None`，不需要用户手工构造 dummy Tensor。
- 图模式、meta 模式、C++ eager wrapper 的 shape/dtype 推导必须一致，否则 FakeTensor、编译图、实际执行会出现输出 dtype 或 shape 不一致。

### 2.3 为什么本需求还要适配 quant_mode/dtype

这个 PR 不只是把 `expert_scales` 接上线。dispatch 侧要把 expertscale 放进现有低时延量化链路，而该链路同时支持非量化、静态量化、per-token/per-group 动态量化、MX/MX clip、低精度输入/输出等场景。因此 Python meta、C++ wrapper、graph convert 必须同步补齐 dtype 和 shape 规则。

如果只接 `topk_weights`，但不修正这些规则，会出现以下问题：

- `quant_mode=5` 的 MX clip 在 meta/C++ 侧没有按 MX 类量化处理，`dynamic_scales` dtype/shape 会推错。
- `x_dtype/y_dtype/x_smooth_scales_dtype` 只传到底层但 Python meta 不理解，FakeTensor 下输出 dtype 与真实执行不一致。
- float4 类 dtype 物理存储为 uint8，逻辑 H 和实际存储 H/2 不一致，如果 meta 不处理会导致输出 shape 错。
- graph 模式中低精度 dtype 需要通过 `Bitcast` 标注真实 GE dtype，否则图编译看到的是 uint8/默认 dtype，无法匹配算子原型。
- Ascend950 使用 channel 后端，`dynamic_scales` shape 规则与 A2/A3 KFC 路径不同，不能再只按 `tp_world_size` 分支推导。

所以 quant_mode/dtype 相关改动属于 expertscale 能力进入低时延 dispatch 后必须一起对齐的接口一致性修复。

## 3. 总体方案

总体采用“Python 语义适配 + 算子显式标志位 + kernel 共享主流程”的方案。

```text
用户
  |
  | low_latency_dispatch(topk_weights)
  v
Python deep_ep.py
  |  topk_weights -> expert_scales
  |  meta 推导 expand_scales / dynamic_scales / expand_x
  v
C++ torch extension wrapper
  |  分配输出 Tensor
  |  TensorWrapper 标注真实 ACL dtype
  v
ACLNN MoeDistributeDispatchV5
  |  tiling: hasExpertScales
  |  kernel: 权重随 token 写入通信窗口
  v
expand_x, assist_info_for_combine, ep_recv_counts, expand_scales

用户/FFN
  |
  | low_latency_combine(topk_weights 或 None)
  v
Python deep_ep.py
  |  None -> empty float32 Tensor
  v
ACLNN MoeDistributeCombineV5
  |  tiling: 空 Tensor -> hasExpertScales=false
  |  kernel: true 读取权重并加权，false 直接求和
  v
combined_x
```

为什么用空 Tensor 表达 combine None：

- combine 底层原型中 `expert_scales` 是必选 Tensor，直接改 optional 会影响 ACLNN 原型、fallback、图转换、已有调用方。
- 空 Tensor 可以用 shape 是否含 0 维统一判断“不存在有效数据”，不需要增加新 attribute。
- kernel 通过 `hasExpertScales` tiling 标志位选择加权聚合或直接求和路径。

## 4. Python 侧详细设计

涉及文件：

- `torch_extension/cann_ops_transformer/ops/deep_ep.py`
- `torch_extension/cann_ops_transformer/ops/graph_convert/graph_convert_moe_distribute_dispatch.py`

### 4.1 dispatch 参数映射

`low_latency_dispatch` 暴露参数名仍使用 Python 用户熟悉的 `topk_weights`，调用底层 op 时映射为 `expert_scales`：

```python
expert_scales=topk_weights
```

这么设计的原因：

- Python API 与 MoE 路由语义保持一致，用户理解的是 topK 权重。
- 底层算子历史命名为 `expert_scales`，沿用该名称可以减少 ACLNN/op_host/op_kernel 大范围重命名。
- 只在 Python wrapper 层做语义映射，避免上层接口和底层实现互相污染。

### 4.2 combine None 适配

`low_latency_combine` 中新增：

```python
topk_weights = _empty_topk_weights_like(x) if topk_weights is None else topk_weights
```

`_empty_topk_weights_like(x)` 返回：

```python
x.new_empty((0,), dtype=torch.float32)
```

原因：

- 空 Tensor 与真实 Tensor 一样在同一 device 上，底层不需要处理 Python `None`。
- dtype 固定为 float32，符合 expert_scales 的算子 dtype 约束。
- shape `(0,)` 能被 tiling 的 `IsPresentTensorShape` 判定为 false。
- combine kernel 在 `hasExpertScales=false` 时走直接求和路径，避免无意义的乘 1 计算。

同时 `assist_info_for_combine` 和 `ep_send_counts` 增加 Python 侧空值检查。原因是 combine 必须依赖 dispatch 的辅助信息原路回收 token；这两个 Tensor 缺失时继续下发会在更深层报错，错误信息不直观。

### 4.3 dtype enum 映射

`deep_ep.py` 新增 `TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP`，用于把传入的 dtype enum 转成 PyTorch dtype。

适配原因：

- 底层接口为了支持特殊低精度类型，通过 `y_dtype/x_dtype/x_smooth_scales_dtype` 传递逻辑 dtype。
- PyTorch Tensor 的物理 dtype 不总是等于算子逻辑 dtype。例如 `torch_npu.int4`、`hifloat8`、`float8_e8m0`、float4 打包类型在 Python Tensor 上可能以 uint8 承载。
- meta 实现必须知道逻辑 dtype，才能推导 `expand_x` 和 `dynamic_scales` 的 dtype。

如果不做 enum 映射，FakeTensor/meta 下可能把低精度输出推成默认 int8 或 uint8，和真实 ACLNN 执行不一致。

### 4.4 quant_mode 相关 meta 推导

#### 4.4.1 dynamic_scales dtype

`get_dispatch_dynamic_scales_dtype` 的规则：

- `quant_mode == 0` 且 `x` 不是 fp16/bf16，并且传入 `scales` 时：
  - 如果传了 `x_smooth_scales_dtype`，按该 enum 推导 dtype。
  - 否则按 `scales.dtype` 推导。
- `quant_mode in (4, 5)` 时，`dynamic_scales` 使用 uint8 物理承载，对应逻辑 `float8_e8m0`。
- 其他场景默认 float32。

为什么要这样：

- 非量化模式下低精度输入可能仍需要 `scales` 做平滑/反量化辅助，`dynamic_scales` 的 dtype 应与 scales 的真实逻辑 dtype 对齐。
- `quant_mode=5` 是 MX clip，与 `quant_mode=4` 一样产出 E8M0 scale，只是 scale 计算公式不同；从接口 shape/dtype 看二者应一致。
- `x_smooth_scales_dtype` 解决了 Tensor 物理 dtype 和算子逻辑 dtype 不一致的问题。

#### 4.4.2 dynamic_scales shape

shape 规则：

```text
quant_mode=0 且 scales 存在:       (A, scales.shape[1])
quant_mode=2:                     (A,)
quant_mode=3:                     (A, ceil(H / 128))
quant_mode=4 或 5:                (A, even_align(ceil(H / 32)))
其他:                             (A,)
```

其中 Ascend950 分支使用 `max(A, A * tp_world_size)` 作为 A 侧长度。

为什么要这样：

- 旧逻辑中 `quant_mode=0 + scales` 使用 `(A * scales.shape[1],)`，这会丢失二维语义；底层 C++ wrapper 已按二维 shape 分配，Python meta 必须对齐。
- per-group 动态量化需要按 128 hidden 分组保存 scale。
- MX/MX clip 以 32 hidden 为 group，且 E8M0 scale 存储需要偶数对齐。
- Ascend950 的 channel 后端输出布局按实际 dispatch/TP 展开后的 token 数推导，不能沿用 A2/A3 的 `tp_world_size == 0/else` 简化分支。

#### 4.4.3 expand_x dtype 与 packed float4 shape

dispatch meta 中新增：

- `y_dtype` 非空时，`expand_x` dtype 由 `y_dtype` 决定。
- `y_dtype` 为 packed float4 类型且未传 `scales` 时，要求 `H` 为偶数，`expand_x` 最后一维使用 `H / 2`。

原因：

- 量化 dispatch 的输出 dtype 不一定由 `quant_mode` 简单决定，用户可能通过 `y_dtype` 显式指定低精度输出。
- float4 是两个 4-bit 值打包进一个 uint8，逻辑 hidden 为 H，物理存储长度为 H/2。
- meta 如果仍返回 `(A, H)`，后续图编译和真实 kernel 写入都会错位。

### 4.5 x_dtype / x_smooth_scales_dtype 参数

`low_latency_dispatch` 透传新增参数：

```python
y_dtype=None
x_dtype=None
x_smooth_scales_dtype=None
```

为什么 dispatch Python API 需要暴露这些参数：

- `x_dtype`：告诉 C++ wrapper 和 graph convert 当前 `x` 的逻辑 ACL dtype。用于 hifloat8、float8、float4 等物理存储与逻辑 dtype 不一致的输入。
- `y_dtype`：告诉 wrapper 分配 `expand_x` 时使用什么逻辑输出 dtype。否则 wrapper 只能按 `quant_mode` 粗略推导。
- `x_smooth_scales_dtype`：替代原 `scales_dtype`，命名更明确，表示 `x_smooth_scale/scales` 的逻辑 dtype，而不是所有 scale 输出的 dtype。

这个改名的价值是避免把输入平滑 scale、动态量化输出 scale、expertscale 三类 scale 混为一个概念。

### 4.6 group name 与 backend 适配

新增 `_get_hccl_comm_name`，优先调用 `group.get_hccl_comm_name`，否则兼容旧的 `_get_backend(torch.device("npu")).get_hccl_comm_name`。

同时 `CommContextManager` 的 backend 从固定 `"kfc"` 改为按 SoC 选择：

```python
{
    "Ascend910B": "kfc",
    "Ascend910_93": "kfc",
    "Ascend950": "channel",
}
```

原因：

- 不同 torch.distributed group 实现暴露 HCCL comm name 的方式不完全一致。
- Ascend950 使用 channel 后端，与 A2/A3 的 KFC 后端不同；这会影响前文提到的 `dynamic_scales` shape 与实际执行路径。

## 5. Graph 模式详细设计

图模式适配集中在 `graph_convert_moe_distribute_dispatch.py`。

### 5.1 算子原型 dtype 范围扩展

dispatch GE 原型中：

- `x` 从只支持 bf16/fp16 扩展到 float8、hifloat8、float4 等低精度 dtype。
- `scales` 从 float 扩展到 float 与 float8_e8m0。
- `expand_x` 输出扩展到 float4 等低精度 dtype。
- `dynamic_scales` 支持 float 与 float8_e8m0。

原因：

- Python eager wrapper 已经支持通过 `x_dtype/y_dtype/x_smooth_scales_dtype` 传递低精度逻辑 dtype，图模式必须暴露相同 dtype 集合。
- 否则 eager 能跑、图模式注册原型不接收，编译会失败。

### 5.2 x bitcast

新增 `_convert_dispatch_x(x, x_dtype)`：

- `x_dtype is None`：不处理。
- 普通低精度 dtype：`ge.Bitcast(x, type=真实 GE dtype)`。
- packed float4：先 bitcast，再把最后一维乘 2，恢复逻辑 shape。

原因：

- float4 的物理 Tensor 是 uint8，每个元素承载两个逻辑值。图中如果不把 shape 还原为逻辑 H，算子的 shape 校验和 hidden 维计算都会错误。
- GE 图需要通过 desc dtype 明确告诉算子这是逻辑低精度输入，而不是普通 uint8。

### 5.3 scales bitcast

新增 `_convert_dispatch_scales(scales, x_smooth_scales_dtype)`。

原因：

- `scales` 的逻辑 dtype 可能是 E8M0，但 Python/GE 侧物理承载不一定直接体现。
- dispatch 算子需要按真实逻辑 dtype 做校验和执行。

### 5.4 expand_x 输出格式化

新增 `_format_expand_x(expand_x, expand_x_dtype, y_dtype)`。

对 packed float4：

- 先把逻辑 shape 转成打包 shape。
- bitcast 为 uint8。
- 再 reshape 到物理输出 shape。

原因：

- 算子内部按逻辑 dtype 执行，Python/GE 对外看到的 Tensor 需要符合物理存储约定。
- 这一步与 eager wrapper 中 `expand_x` 最后一维 H/2 的分配规则一致。

### 5.5 dynamic_scales dtype 回写

新增 `_set_dynamic_scales_dtype`。

原因：

- GE op 返回后，`dynamic_scales` desc dtype 需要按 quant_mode 修正。
- `quant_mode=0` 且低精度 scales 存在时，dynamic scales dtype 跟 scales。
- `quant_mode=4/5` 时，dynamic scales 是 E8M0。
- 其他场景是 float32。

这保证图模式输出 dtype 与 eager meta/C++ wrapper 一致。

## 6. C++ Torch Extension 详细设计

涉及文件：

- `torch_extension/cann_ops_transformer/ops/csrc/moe_distribute_dispatch.cpp`
- `torch_extension/cann_ops_transformer/ops/csrc/moe_distribute_combine.cpp`
- `torch_extension/cann_ops_transformer/common/inc/aclnn_common.h`

### 6.1 dispatch wrapper 输出分配

C++ wrapper 在真正调用 ACLNN 前先分配 7 个输出：

```text
expand_x
dynamic_scales
assist_info_for_combine
expert_token_nums
ep_recv_counts
tp_recv_counts
expand_scales
```

关键变化：

- `HasTensor(expertScales)` 判断是否启用 expertscale。
- `expandScales = empty({a})` 或 `empty({0})`。
- 非 Ascend950 且启用 expertscale 时，扩大 `epRecvCounts`。
- `yDtype` 决定 `expandX` 的 ACL dtype。
- `scalesDtype/xDtype` 通过 `TensorWrapper` 传给 ACLNN。

为什么 C++ wrapper 也要做与 Python meta 相同的逻辑：

- Python meta 只影响 FakeTensor/编译前推导。
- eager 真实执行时输出 Tensor 是 C++ wrapper 分配的。
- 两边任何一个 shape/dtype 不一致，都会导致 eager 与 compile 行为不同。

### 6.2 TensorWrapper 的必要性

dispatch 调用从直接传 Tensor 改成：

```cpp
TensorWrapper xWrapper = {x, xAclDtype};
TensorWrapper scalesWrapper = {scalesTensor, scalesAclDtype};
TensorWrapper expandXWrapper = {expandX, outputAclDtype};
TensorWrapper dynamicScalesWrapper = {dynamicScales, dynamicScalesAclDtype};
```

原因：

- 对特殊 dtype，PyTorch Tensor 的 storage dtype 与 ACLNN 期望 dtype 不一定相同。
- `TensorWrapper` 可以把同一个 Tensor 的地址与逻辑 ACL dtype 一起传给 ACLNN。
- 这样既不需要额外复制/转换物理内存，也能让底层按正确 dtype 解释数据。

### 6.3 dynamic_scales C++ 推导

C++ 新增 `GetDynamicScalesDtype` 与 `GetDynamicScalesShape`，规则与 Python meta 对齐。

原因：

- `quant_mode=5` 要与 `quant_mode=4` 一样推导 E8M0 scale。
- Ascend950 要走完整二维 shape 规则。
- `quant_mode=0 + scales` 要保留 `(A, scale_cols)` 二维 shape。

### 6.4 packed float4 输出分配

C++ wrapper 中如果 `yDtype` 是 packed float4 且没有 `scales`，则：

- 检查 `H % 2 == 0`。
- `expandX` 分配为 `(max(A, A * tpWorldSize), H / 2)`。

原因同 Python meta：float4 逻辑 H 对应 uint8 物理 H/2。

## 7. Dispatch 算子详细设计

涉及文件：

- `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp`
- `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_check_win_size.h`
- `mc2/moe_distribute_dispatch_v2/op_kernel/*`
- `mc2/moe_distribute_dispatch_v3/*`

### 7.1 hasExpertScales 判定

tiling 中新增 `IsPresentTensorShape`：

```text
storageShape == nullptr        -> false
任意维度为 0                  -> false
否则                           -> true
```

Dispatch 的 `expert_scales` 是 optional input，所以不存在和空 Tensor 都表示不启用。

为什么要检查“任意维度为 0”：

- Python combine 用空 Tensor 表达 None，dispatch graph/eager 中也可能出现空 optional Tensor。
- 只判断 nullptr 无法区分“传了一个空 Tensor”和“传了真实权重”。

### 7.2 expert_scales 输入校验

仅当 `hasExpertScales=true` 时校验：

- shape 必须 2D。
- dtype 必须 float32。
- format 必须 ND。
- dim0 等于 `BS`。
- dim1 等于 `K`。

原因：

- `expert_scales[token, topk]` 必须与 `expert_ids[token, topk]` 一一对应。
- dtype 固定 float32 可以简化通信窗口布局和 kernel UB 处理。
- 不启用 expertscale 时跳过校验，允许空 Tensor。

### 7.3 expand_scales 输出校验

启用 expertscale 时，`expand_scales`：

- shape 1D。
- dtype float32。
- format ND。
- dim0 至少为 `A`。

原因：

- dispatch 后每个接收 token 最多对应一个 expand 后位置，因此输出长度按 `A` 预留。
- `A` 是静态上界，实际有效 token 数由 `expert_token_nums/ep_recv_counts` 描述。

### 7.4 通信窗口大小适配

expertScale 复用三元组所在 32B 对齐块的剩余空间，依次排布为：

```text
rank_id | token_id | topk_id | expert_scale
```

因此 token 实际发送长度仍为：

```text
align(H * dtype_size, 32) + scale_expand_idx_buffer
```

启用 expertscale 不再增加每个 token 的通信窗口占用，因此 tiling 的窗口校验和 Python
`get_low_latency_ccl_buffer_size` 接口保持原有计算方式。

### 7.5 dispatch kernel 数据流

启用 expertscale 后，kernel 做以下事情：

1. 初始化时绑定 `expertScalesGMTensor_` 和 `expandScalesOutGM_`。
2. 将 `(BS, K)` 权重从 GM 拷贝到 UB。
3. 处理 token/topK 路由时，从 `expertScalesTensor_[token * K + topk]` 读取权重。
4. 将权重写入发送窗口中预留的 expert scale 槽位。
5. 接收端从窗口中取出权重，并按 expand 后位置写入 `expand_scales`。

未启用 expertscale 时：

- 不绑定 expertScales GM。
- 不初始化 expertScales UB buffer。
- 不写入/读取权重槽位。
- `expand_scales` 是空输出。

### 7.6 V3 同步适配

V3 dispatch 原型、infer shape、A3/A5 kernel 入口同步增加/透传 `expert_scales`。

原因：

- Python `npu_moe_distribute_dispatch` 最终面向 V5 ACLNN，但底层不同 SoC/arch 可能落到 V2/V3 相关实现。
- 只适配 V2 会导致部分架构路径编译或运行失败。

## 8. Combine 算子详细设计

涉及文件：

- `mc2/moe_distribute_combine_v2/op_host/op_tiling/moe_distribute_combine_v2_tiling.cpp`
- `mc2/moe_distribute_combine_v2/op_kernel/moe_distribute_combine_v2_tiling.h`
- `mc2/moe_distribute_combine_v2/op_kernel/arch22/moe_distribute_combine_v2.h`
- `mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h`

### 8.1 tiling 标志位

`MoeDistributeCombineV2Info` 中将 reserved 字段替换为：

```cpp
bool hasExpertScales;
```

tiling 根据 `expert_scales` shape 判断：

```text
真实 (BS, K) Tensor -> hasExpertScales=true
空 Tensor           -> hasExpertScales=false
```

原因：

- kernel 不能在设备侧可靠地通过 Tensor shape 做复杂判断，host tiling 应提前给出布尔标志。
- 复用 reserved 字段可以减少 tiling 结构体扩展带来的二进制布局影响。

### 8.2 combine 输入校验分支

combine 原先默认 `expert_scales` 一定存在且为 `(BS, K)`。本 PR 改为仅在 `hasExpertScales=true` 时校验：

- dim。
- dtype。
- format。
- shape 与 `expert_ids` 一致。

原因：

- Python 传 `None` 时底层实际收到 `(0,)` Tensor，如果继续强制 2D 校验会报错。
- 空 Tensor 是“不使能”的控制信号，不应该进入真实权重校验。

### 8.3 kernel scale 数据来源

`ExpertScaleCopy` 中新增分支：

```text
hasExpertScales=true:
    从 expertScalesGM_ 拷贝真实权重到 expertScalesLocal_

hasExpertScales=false:
    不分配或填充 expertScalesLocal_，直接对专家输出求和
```

直接求和路径的设计：

- 普通专家、copy expert、constant expert 均根据 `hasExpertScales` 选择加权或直接累加。
- A5 的 INT8/MXFP8 融合反量化微指令新增不带专家权重的直接累加入口。
- 空 Tensor 路径不读取权重、不填充 1.0，也不执行乘 1。

### 8.4 数学语义

传入 `topk_weights`：

```text
out[token] = sum_i(topk_weights[token, i] * expert_out[token, i])
```

传入 `None`：

```text
out[token] = sum_i(1.0 * expert_out[token, i])
```

共享专家仍按原逻辑处理，不属于 topK expertscale 权重控制范围。

## 9. 数据结构与内存布局

### 9.1 expert_scales

输入逻辑：

```text
shape: (BS, K)
dtype: float32
format: ND
index: expert_scales[token_id, topk_id]
```

该 Tensor 与 `expert_ids` 使用同样的 `(token, topK)` 坐标。

### 9.2 expand_scales

dispatch 输出：

```text
未启用 expertscale: shape (0,)
启用 expertscale:   shape (A,)
dtype:              float32
```

`expand_scales` 表示权重已经按 dispatch 后顺序重排。它的主要价值是给后续链路提供“与 expand_x 同序”的专家权重，不再需要使用原始 `(BS, K)` 做复杂索引映射。

### 9.3 assist_info_for_combine / ep_recv_counts 扩容

非 Ascend950 且启用 expertscale 时，需要额外空间承载 scale 随通信传递产生的元信息：

```text
expert_scales_recv_extra = global_bs_real * 2 * K * (ep_world_size / 8)
ep_recv_counts = ep_world_size * local_moe_expert_num + expert_scales_recv_extra
assist_info_for_combine_shape = max(old_shape, expert_scales_recv_extra)
```

原因：

- 非 Ascend950 的 fullmesh/KFC 路径需要额外 send count/meta 空间描述 expertscale 数据块。
- 该空间与 `global_bs`、`K`、节点内 rank 分组相关。
- Ascend950 channel 路径不走同样的扩容规则，因此分支排除 Ascend950。

## 10. 文件级适配清单

| 文件 | 适配点 | 为什么要适配 |
| --- | --- | --- |
| `torch_extension/cann_ops_transformer/ops/deep_ep.py` | `topk_weights -> expert_scales`；combine None 转空 Tensor；dtype enum；quant_mode 4/5；packed float4；Ascend950 backend | 保证 Python 用户语义、meta 推导、eager 执行入口一致 |
| `torch_extension/cann_ops_transformer/ops/graph_convert/graph_convert_moe_distribute_dispatch.py` | 扩展 dtype 原型；x/scales bitcast；float4 reshape；dynamic_scales dtype 回写 | 保证图模式与 eager 支持同样低精度和 quant_mode |
| `torch_extension/cann_ops_transformer/ops/csrc/moe_distribute_dispatch.cpp` | 按专家权重分配 `expandScales`；按 dtype enum 分配输出；TensorWrapper 传 ACL dtype | 保证真实执行输出 shape/dtype 与 Python meta 一致 |
| `torch_extension/cann_ops_transformer/ops/csrc/moe_distribute_combine.cpp` | 透传真实或空 `expertScales` Tensor | 保持 combine ABI，同时支持可选 expertscale |
| `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp` | `hasExpertScales` 判定；校验 `expert_scales/expand_scales` | 避免空 Tensor 被当作真实权重，保证真实权重 shape 合法 |
| `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_check_win_size.h` | 保持原有窗口计算，不额外计入权重 buffer | expertScale 复用三元组后的剩余空间，不增大通信窗口 |
| `mc2/moe_distribute_dispatch_v2/op_kernel/*` | 权重作为三元组后的第 4 项随 token 写窗口，接收后写 `expand_scales` | 实现 dispatch 携带 expertscale 且不增加 token 窗口占用 |
| `mc2/moe_distribute_dispatch_v3/*` | 原型/infer/kernel 入口同步 expert_scales | 覆盖不同 SoC/arch 的 dispatch 实现 |
| `mc2/moe_distribute_combine_v2/op_host/op_tiling/*` | combine `hasExpertScales`，条件化校验 expert_scales | 支持空 Tensor 表示不使能 |
| `mc2/moe_distribute_combine_v2/op_kernel/*` | 无 expertscale 时直接求和 | 避免访问空 Tensor和无意义的乘 1 计算 |
| `torch_extension/cann_ops_transformer/docs/zh/low_latency_dispatch.md` | 更新 `topk_weights` 与 quant/dtype 说明 | 对外 API 文档同步 |
| `torch_extension/cann_ops_transformer/docs/zh/low_latency_combine.md` | 更新 `topk_weights=None` 说明 | 对外 API 文档同步 |

## 11. 典型调用

### 11.1 启用 expertscale

```python
expand_x, dynamic_scales, assist_info, expert_token_nums, ep_recv_counts, expand_scales = \
    buffer.low_latency_dispatch(
        x,
        topk_idx,
        num_experts,
        quant_mode=quant_mode,
        x_smooth_scale=x_smooth_scale,
        topk_weights=topk_weights,
        num_max_dispatch_tokens_per_rank=max_bs,
        y_dtype=y_dtype,
        x_dtype=x_dtype,
        x_smooth_scales_dtype=x_smooth_scales_dtype,
    )

out = buffer.low_latency_combine(
    ffn_out,
    topk_idx,
    topk_weights,
    assist_info,
    ep_recv_counts,
    num_experts=num_experts,
    num_max_dispatch_tokens_per_rank=max_bs,
)
```

### 11.2 combine 不启用 expertscale

```python
out = buffer.low_latency_combine(
    ffn_out,
    topk_idx,
    None,
    assist_info,
    ep_recv_counts,
    num_experts=num_experts,
    num_max_dispatch_tokens_per_rank=max_bs,
)
```

该调用下：

```text
Python None -> empty float32 Tensor -> tiling hasExpertScales=false -> kernel 直接求和
```

## 12. 测试建议

Python/meta：

- `topk_weights=None` 时 dispatch meta 返回 `expand_scales.shape == (0,)`。
- `topk_weights=(BS,K)` 时 dispatch meta 返回 `expand_scales.shape == (A,)`。
- `quant_mode=4/5` 时 `dynamic_scales` dtype 为 E8M0 对应物理 uint8。
- `quant_mode=0 + x_smooth_scales_dtype` 时 `dynamic_scales` dtype 与逻辑 scales dtype 一致。
- `y_dtype` 为 packed float4 时检查 H 偶数约束与 H/2 输出 shape。

Graph convert：

- float8/hifloat8/float4 输入能完成 GE dtype 标注。
- packed float4 输入输出 reshape 后 shape 与 eager 一致。
- `dynamic_scales` desc dtype 在 quant_mode 0/4/5 下正确。

算子：

- dispatch 启用 expertscale 后，`expand_scales` 与参考重排权重一致。
- dispatch 未启用 expertscale 时不增加权重窗口读写。
- combine 传真实权重时与 `sum(weight * expert_out)` 一致。
- combine 传 None 时与 `sum(expert_out)` 一致。
- 特殊专家、`x_active_mask`、`comm_alg=fullmesh_v1/fullmesh_v2`、Ascend950 channel 后端分别覆盖。

回归：

- 不传 `topk_weights` 的旧 dispatch/combine 调用保持可用。
- 只使用 bf16/fp16、`quant_mode=0` 的旧模型输出不变。
- 图模式和 eager 模式输出 shape/dtype 一致。
