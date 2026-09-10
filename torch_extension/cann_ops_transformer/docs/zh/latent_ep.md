# latent\_ep

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

latent_ep接口用于MoE的Expert Parallelism（EP）并行部署，支持通过 [dispatch](#dispatch) 将token数据分发到对应专家卡，再通过 [combine](#combine) 将专家输出按原路由聚合回原始序列。

- MoE专家接收latent token，共享专家接收投影到latent空间之前的原始token。
- dispatch和combine的量化仅作用于MoE专家，共享专家使用原始数据类型通信。
- 通过 [get_ccl_buffer_size](#get_ccl_buffer_size) 计算CCL通信buffer大小，由MoeDistributeBuffer创建通信上下文。

## 函数说明

### dispatch

**功能**：需与 [combine](#combine) 配套使用，完成MoE的Expert Parallelism（EP）并行部署下的token dispatch。该接口根据每个token的topK专家索引，将latent token分发到对应的MoE专家卡上，并将原始token分发到共享专家卡上。

**函数原型**：

```python
latent_ep.dispatch(
    context, x, expert_ids,
    ep_world_size, ep_rank_id, moe_expert_num, ccl_buffer_size,
    *, shared_expert_input=None, scales=None, x_active_mask=None, expert_scales=None,
    elastic_info=None, performance_info=None,
    tp_world_size=0, tp_rank_id=0, expert_shard_type=0,
    shared_expert_num=1, shared_expert_rank_num=0,
    quant_mode=0, global_bs=0, expert_token_nums_type=1,
    comm_alg="", zero_expert_num=0, copy_expert_num=0, const_expert_num=0,
    y_dtype=None, x_dtype=None, x_smooth_scales_dtype=None,
) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
```

**输入参数**：

- **context** (`Tensor`)：必选参数，表示通信上下文，传入MoeDistributeBuffer实例的 `context`。
- **x** (`Tensor`)：必选参数，表示计算使用的latent token数据，需根据 `expert_ids` 来发送给其他卡。要求为2维张量，shape为 `(BS, H)`，数据类型支持 `bfloat16`、`float16`，数据格式为ND。
- **expert_ids** (`Tensor`)：必选参数，表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2维张量，shape为 `(BS, K)`，数据类型为 `int32`，数据格式为ND。对应 [combine](#combine) 的 `expert_ids` 输入，同一行中的K个value不能重复。
- **ep_world_size** (`int`)：必选参数，表示EP通信域的大小（即参与EP通信的卡数）。取值范围 `[2, 1024]`。
- **ep_rank_id** (`int`)：必选参数，表示当前卡在EP通信域中的rank编号。取值范围 `[0, ep_world_size)`。
- **moe_expert_num** (`int`)：必选参数，MoE专家数量，取值范围[1, 1024]，并且满足以下条件：moe_expert_num%(ep_world_size - shared_expert_rank_num)=0。
- **ccl_buffer_size** (`int`)：必选参数，表示CCL通信buffer大小，单位为字节，传入MoeDistributeBuffer实例的 `ccl_buffer_size`。
- <strong>*</strong>：其之前的变量是位置相关的；之后的变量是可选参数，需要使用键值对赋值，不赋值会使用默认值。
- **shared_expert_input** (`Tensor`)：可选参数，表示投影到latent空间之前的token数据。要求为2维张量，shape为 `(BS, 2H)`，数据类型、BS和设备与 `x` 一致，数据格式为ND。传入有效Tensor时使能独立共享专家，默认值为None。
- **scales** (`Tensor`)：可选参数，表示量化平滑参数或量化系数，数据类型为 `float32`，数据格式为ND。`quant_mode` 为1时必须传，为2或3时可传可不传，为0、4或5时不传。shape约束见“quant_mode相关约束”。默认值为None。
- **x_active_mask** (`Tensor`)：可选参数，表示token是否参与通信，要求是一个1维或者2维张量。当输入为1维时，shape为(BS, )；当输入为2维时，shape为(BS, K)。数据格式要求为ND，支持非连续的Tensor。当输入为1维时，参数为true表示对应的token参与通信，true必须排到false之前，例：{true, false, true}为非法输入；当输入为2D时，参数为true表示当前token对应的`expert_ids`参与通信，若当前token对应的K个`bool`值全为false，表示当前token不会参与通信。默认所有token都会参与通信。当每张卡的BS数量不一致时，所有token必须全部有效。数据类型为 `bool`。使能独立共享专家时，仅作用于MoE专家，共享专家接收全部token。默认值为None。
- **expert_scales** (`Tensor`)：可选参数，表示每个token的topK个专家权重。要求为2维张量，shape为 `(BS, K)`，数据类型为 `float32`，数据格式为ND。传入有效Tensor时，`expand_scales` 输出有效。默认值为None。
- **elastic_info** (`Tensor`)：可选参数，使能独立共享专家时不支持该参数，传默认值None即可。
- **performance_info** (`Tensor`)：可选参数，表示性能统计信息，默认值为None。
- **tp_world_size** (`int`)：可选参数，表示TP通信域的大小。使能独立共享专家时仅支持1，默认值为0。
- **tp_rank_id** (`int`)：可选参数，表示当前卡在TP通信域中的rank编号。使能独立共享专家时仅支持0，默认值为0。
- **expert_shard_type** (`int`)：可选参数，表示共享专家卡排布类型。当前仅支持0，表示共享专家卡排在MoE专家卡前面。默认值为0。
- **shared_expert_num** (`int`)：可选参数，表示共享专家数量，一个共享专家可以复制部署到多个卡上。取值范围[0, 4]，0表示无共享专家，默认值为1。
- **shared_expert_rank_num** (`int`)：可选参数，表示共享专家卡数量。取值范围[0, ep_world_size)。取0表示无共享专家，不取0需满足shared_expert_rank_num%shared_expert_num=0。默认值为0。
- **quant_mode** (`int`)：可选参数，表示量化模式，0表示非量化（默认），1表示静态量化，2表示pertoken动态量化，3表示pergroup动态量化，4表示mx动态量化，5表示mx clip动态量化。使能独立共享专家时，仅对MoE专家量化。
- **global_bs** (`int`)：可选参数，表示EP域的token容量，所有rank取值相同。各卡BS相同时可传0，按 `BS * ep_world_size` 计算；各卡BS不同时传入 `max_bs * ep_world_size`，其中max_bs为各卡BS的最大值。默认值为0。
- **expert_token_nums_type** (`int`)：可选参数，表示输出`expert_token_nums`的值类型，取值范围[0, 1]，0表示每个专家收到token数量的前缀和，1表示每个专家收到的token数量（默认）。
- **comm_alg** (`str`)：可选参数，表示通信亲和内存布局算法。使能独立共享专家时仅支持 `"fullmesh_v2"`。默认值为 `""`。
- **zero_expert_num** (`int`)：可选参数，表示零专家的数量。取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的零专家的ID值是[moe_expert_num, moe_expert_num+zero_expert_num)。默认值为0。
- **copy_expert_num** (`int`)：可选参数，表示拷贝专家的数量。取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的拷贝专家的ID值是[moe_expert_num+zero_expert_num, moe_expert_num+zero_expert_num+copy_expert_num)。默认值为0。
- **const_expert_num** (`int`)：可选参数，表示常量专家的数量。取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的常量专家的ID值是[moe_expert_num+zero_expert_num+copy_expert_num, moe_expert_num+zero_expert_num+copy_expert_num+const_expert_num)。默认值为0。
- **y_dtype** (`int`)：可选参数，表示MoE专家 `expand_x` 输出的逻辑数据类型，传入整数枚举，默认值为None。所有rank取值相同；共享专家卡的 `expand_x` 数据类型与 `shared_expert_input` 一致。支持的取值见“quant_mode相关约束”。
- **x_dtype** (`int`)：可选参数，表示输入 `x` 的逻辑数据类型，默认值为None，由 `x` 的PyTorch存储类型推导。使能独立共享专家时使用默认值。
- **x_smooth_scales_dtype** (`int`)：可选参数，表示 `scales` 的逻辑数据类型，默认值为None，由 `scales` 的PyTorch存储类型推导。使能独立共享专家时使用默认值。

**输出说明**：

- **expand_x** (`Tensor`)：表示本卡收到的token数据。MoE专家卡shape为 `(A, H)`，类型由 `quant_mode` 和 `y_dtype` 决定；共享专家卡shape为 `(A, 2H)`，数据类型与 `shared_expert_input` 一致，数据格式为ND。经专家网络处理后，作为 [combine](#combine) 的 `expand_x` 输入。
- **dynamic_scales** (`Tensor`)：表示动态量化系数，数据格式为ND。MoE专家卡的shape及数据类型见“quant_mode相关约束”；共享专家卡shape为 `(A,)`，数据类型为 `float32`，有效行的值为1。
- **assist_info_for_combine** (`Tensor`)：表示dispatch阶段生成的辅助信息。shape为 `(max(BS * K, A * 128),)`，数据类型为 `int32`，数据格式为ND，对应 [combine](#combine) 的同名输入。
- **expert_token_nums** (`Tensor`)：本卡每个专家实际收到的token数量。要求为1维张量，shape为 `(E,)`，数据类型为 `int64`，数据格式为ND，E为本卡专家数。值类型由 `expert_token_nums_type` 决定。
- **ep_recv_counts** (`Tensor`)：表示EP通信域各卡收到的token数，token数以前缀和的形式表示。shape为 `(ep_world_size * E,)`，数据类型为 `int32`，数据格式为ND，对应 [combine](#combine) 的 `ep_send_counts` 输入。
- **tp_recv_counts** (`Tensor`)：表示TP通信计数。shape为 `(1,)`，数据类型为 `int32`，数据格式为ND，对应 [combine](#combine) 的 `tp_send_counts` 输入。
- **expand_scales** (`Tensor`)：表示本卡收到的topK权重。当输入 `expert_scales` 不为None时，shape为 `(A,)`，数据类型为 `float32`，数据格式为ND，对应 [combine](#combine) 的 `expand_scales` 输入；否则返回shape为 `(0,)` 的空Tensor。

其中A表示本卡可能收到的最大token数。设 `G = global_bs`，传0时取 `BS * ep_world_size`：

- MoE专家卡：`E = moe_expert_num / (ep_world_size - shared_expert_rank_num)`，`A = G * min(E, K)`。
- 共享专家卡：`E = 1`，`R = shared_expert_rank_num / shared_expert_num`，`A = (G / ep_world_size) * ceil(ep_world_size / R)`。
- 有效token数由 `expert_token_nums` 指定，超过有效token数的输出数据无效。

### combine

**功能**：需与 [dispatch](#dispatch) 配套使用，相当于按dispatch算子收集数据的路径原路返回。该接口将专家处理后的token数据根据dispatch阶段记录的信息，通过逆向路由和加权聚合，将token数据组合还原为原始序列顺序。MoE专家和共享专家的结果分别输出。

**计算公式**：

$$x\_out_i = \sum_{k \in V_i} expert\_scales_{i,k} \times moe\_result_{i,k}$$

$$shared\_expert\_out_i = \sum_{e=0}^{shared\_expert\_num-1} shared\_result_{i,e}$$

其中，`V_i` 为token i的有效路由集合，`moe_result` 和 `shared_result` 为专家计算结果。共享专家不乘权重系数，直接相加。零、拷贝、常量专家的结果归入 `x_out`。

**函数原型**：

```python
latent_ep.combine(
    context, expand_x, expert_ids, assist_info_for_combine,
    ep_send_counts, expert_scales,
    ep_world_size, ep_rank_id, moe_expert_num, ccl_buffer_size,
    *, tp_send_counts=None, x_active_mask=None, expand_scales=None,
    shared_expert_x=None, elastic_info=None, ori_x=None,
    const_expert_alpha_1=None, const_expert_alpha_2=None, const_expert_v=None,
    performance_info=None, tp_world_size=0, tp_rank_id=0, expert_shard_type=0,
    shared_expert_num=1, shared_expert_rank_num=0, global_bs=0,
    comm_quant_mode=0, comm_alg="", zero_expert_num=0,
    copy_expert_num=0, const_expert_num=0,
    enable_independent_shared_expert=False,
) -> (Tensor, Tensor)
```

**输入参数**：

- **context** (`Tensor`)：必选参数，表示通信上下文，传入MoeDistributeBuffer实例的 `context`。
- **expand_x** (`Tensor`)：必选参数，表示经过专家计算后的token数据，即 [dispatch](#dispatch) 输出的 `expand_x` 经过专家网络处理后的结果。要求为2维张量，MoE专家卡shape为 `(A, H)`，共享专家卡shape为 `(A, 2H)`，数据类型支持 `bfloat16`、`float16`，数据格式为ND。
- **expert_ids** (`Tensor`)：必选参数，表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2维张量，shape为 `(BS, K)`，数据类型为 `int32`，数据格式为ND。对应 [dispatch](#dispatch) 的 `expert_ids` 输入，同一行中的K个value不能重复。
- **assist_info_for_combine** (`Tensor`)：必选参数，表示dispatch阶段生成的辅助信息，shape为 `(max(BS * K, A * 128),)`，数据类型为 `int32`，数据格式为ND。对应 [dispatch](#dispatch) 的 `assist_info_for_combine` 输出。
- **ep_send_counts** (`Tensor`)：必选参数，表示本卡每个专家发给EP（Expert Parallelism）域每个卡的token数（token数以前缀和的形式表示），数据格式为ND。对应[dispatch](#dispatch)的`ep_recv_counts`输出。shape为 `(ep_world_size * E,)`，数据类型为 `int32`，E为本卡专家数。
- **expert_scales** (`Tensor`)：必选参数，表示每个token的topK个专家的权重，其中共享专家不需要乘权重系数，直接相加即可。要求为2维张量，shape为 `(BS, K)`，数据类型为 `float32`，数据格式为ND。
- **ep_world_size** (`int`)：必选参数，表示EP通信域的大小（即参与EP通信的卡数）。取值范围 `[2, 1024]`。
- **ep_rank_id** (`int`)：必选参数，表示当前卡在EP通信域中的rank编号。取值范围 `[0, ep_world_size)`。
- **moe_expert_num** (`int`)：必选参数，MoE专家数量，取值范围[1, 1024]，并且满足以下条件：moe_expert_num%(ep_world_size - shared_expert_rank_num)=0。
- **ccl_buffer_size** (`int`)：必选参数，表示CCL通信buffer大小，单位为字节，传入MoeDistributeBuffer实例的 `ccl_buffer_size`。
- <strong>*</strong>：其之前的变量是位置相关的；之后的变量是可选参数，需要使用键值对赋值，不赋值会使用默认值。
- **tp_send_counts** (`Tensor`)：可选参数，表示TP通信计数，shape为 `(1,)`，数据类型为 `int32`，数据格式为ND。对应 [dispatch](#dispatch) 的 `tp_recv_counts` 输出。默认值为None。
- **x_active_mask** (`Tensor`)：可选参数，表示token是否参与通信，数据类型为 `bool`，shape为 `(BS,)` 或 `(BS, K)`，数据格式为ND。取值与 [dispatch](#dispatch) 的 `x_active_mask` 一致，仅作用于MoE专家。默认值为None。
- **expand_scales** (`Tensor`)：可选参数，表示随token分发的专家权重，shape为 `(A,)`，数据类型为 `float32`，数据格式为ND。对应 [dispatch](#dispatch) 的有效 `expand_scales` 输出。默认值为None。
- **shared_expert_x** (`Tensor`)：可选参数，表示combine后需要相加的共享专家token。使能独立共享专家时不支持该参数，传默认值None即可。
- **elastic_info** (`Tensor`)：可选参数，使能独立共享专家时不支持该参数，传默认值None即可。
- **ori_x** (`Tensor`)：可选参数，表示未经过专家计算的token数据，在`copy_expert_num`不为0或`const_expert_num`不为0的场景下需要本输入数据，数据类型需跟`expand_x`保持一致，数据格式要求为ND。可选择传入有效数据或填None，当`copy_expert_num`不为0或`const_expert_num`不为0时必须传入有效数据。默认值为None。shape为 `(BS, H)`，H为latent隐藏维度。
- **const_expert_alpha_1** (`Tensor`)：可选参数，在`const_expert_num`不为0的场景下需要输入的计算系数，数据类型需跟`expand_x`保持一致，数据格式要求为ND。可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入。默认值为None。shape为 `(const_expert_num, H)`。
- **const_expert_alpha_2** (`Tensor`)：可选参数，在`const_expert_num`不为0的场景下需要输入的计算系数，数据类型需跟`expand_x`保持一致，数据格式要求为ND。可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入。默认值为None。shape为 `(const_expert_num, H)`。
- **const_expert_v** (`Tensor`)：可选参数，在`const_expert_num`不为0的场景下需要输入的计算系数，数据类型需跟`expand_x`保持一致，数据格式要求为ND。可选择传入有效数据或填None，当`const_expert_num`不为0时必须传入有效输入。默认值为None。shape为 `(const_expert_num, H)`。
- **performance_info** (`Tensor`)：可选参数，表示性能统计信息，默认值为None。
- **tp_world_size** (`int`)：可选参数，表示TP通信域的大小。使能独立共享专家时仅支持1，默认值为0。
- **tp_rank_id** (`int`)：可选参数，表示当前卡在TP通信域中的rank编号。使能独立共享专家时仅支持0，默认值为0。
- **expert_shard_type** (`int`)：可选参数，表示共享专家卡排布类型。当前仅支持0，表示共享专家卡排在MoE专家卡前面。默认值为0。
- **shared_expert_num** (`int`)：可选参数，表示共享专家数量，一个共享专家可以复制部署到多个卡上。取值范围[0, 4]，0表示无共享专家，默认值为1。
- **shared_expert_rank_num** (`int`)：可选参数，表示共享专家卡数量。取值范围[0, ep_world_size)。取0表示无共享专家，不取0需满足shared_expert_rank_num%shared_expert_num=0。默认值为0。
- **global_bs** (`int`)：可选参数，表示EP域的token容量，所有rank取值相同。各卡BS相同时可传0，按 `BS * ep_world_size` 计算；各卡BS不同时传入 `max_bs * ep_world_size`，其中max_bs为各卡BS的最大值。默认值为0。
- **comm_quant_mode** (`int`)：可选参数，表示通信量化类型。支持取0、2和4，0表示通信时不量化，2表示通信时进行int8量化，4表示通信时进行mx量化，量化类型为 `float8_e4m3fn`。仅对MoE专家量化，默认值为0。
- **comm_alg** (`str`)：可选参数，表示通信亲和内存布局算法，当前版本仅支持""。默认值为""。
- **zero_expert_num** (`int`)：可选参数，表示零专家的数量。取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的零专家的ID值是[moe_expert_num, moe_expert_num+zero_expert_num)。默认值为0。
- **copy_expert_num** (`int`)：可选参数，表示拷贝专家的数量。取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的拷贝专家的ID值是[moe_expert_num+zero_expert_num, moe_expert_num+zero_expert_num+copy_expert_num)。默认值为0。
- **const_expert_num** (`int`)：可选参数，表示常量专家的数量。取值范围[0, MAX_int32)，MAX_int32 = 2^31 - 1，合法的常量专家的ID值是[moe_expert_num+zero_expert_num+copy_expert_num, moe_expert_num+zero_expert_num+copy_expert_num+const_expert_num)。默认值为0。
- **enable_independent_shared_expert** (`bool`)：可选参数，表示是否使能独立共享专家。设为True时，MoE专家和共享专家结果分别输出；设为False时，保持原有合并方式。默认值为False。

**输出说明**：

- **x_out** (`Tensor`)：表示MoE专家combine后的token数据，还原为原始序列顺序。要求为2维张量，shape为 `(BS, H)`，数据类型与 `expand_x` 一致（`bfloat16` 或 `float16`），数据格式为ND。使能独立共享专家时不包含共享专家部分；不使能独立共享专家时保持原有合并方式。
- **shared_expert_out** (`Tensor`)：表示共享专家combine后的token数据，还原为原始序列顺序。要求为2维张量，shape为 `(BS, 2H)`，数据类型与 `x_out` 一致，数据格式为ND。不使能独立共享专家时返回shape为 `(0,)` 的空Tensor。

### get_ccl_buffer_size

**功能**：返回独立共享专家场景所需的CCL通信buffer配置大小。

**函数原型**：

```python
latent_ep.get_ccl_buffer_size(
    ep_world_size, max_bs, hidden, moe_expert_num, topk,
    shared_expert_num, shared_expert_rank_num, quant_mode=0, y_dtype=None,
) -> int
```

**输入参数**：

- **ep_world_size** (`int`)：必选参数，EP通信域大小，范围 `[2, 1024]`。
- **max_bs** (`int`)：必选参数，所有rank的输入token数上界，范围 `[1, 512]`。
- **hidden** (`int`)：必选参数，表示latent隐藏维度H，取值范围 `[1024, 4096]`。
- **moe_expert_num** (`int`)：必选参数，MoE专家总数，取值及整除约束与Dispatch一致。
- **topk** (`int`)：必选参数，每个token的路由数K，范围 `[1, 16]`。
- **shared_expert_num** (`int`)：必选参数，逻辑共享专家数，范围 `[1, 4]`。
- **shared_expert_rank_num** (`int`)：必选参数，共享专家卡数，必须小于EP大小且为逻辑共享专家数的正整数倍。
- **quant_mode** (`int`)：可选参数，MoE分发量化模式，默认值为0，与Dispatch保持一致。
- **y_dtype** (`int`)：可选参数，MoE分发类型枚举，默认值为None；量化模式1/2/3必须指定，MX只支持E4M3FN，与Dispatch保持一致。

**输出说明**：

- **ccl_buffer_size** (`int`)：CCL通信buffer配置大小，单位为MiB，与 `MoeDistributeBuffer.get_low_latency_ccl_buffer_size` 的返回值单位一致。构造A5的MoeDistributeBuffer时，传入 `ccl_buffer_size * 2 * 1024 * 1024` 字节；dispatch/combine使用实例的 `buffer.ccl_buffer_size`。

## 约束说明

- 本文约束适用于独立共享专家场景。算子包需提供ACLNN V6接口，并与Torch Extension版本匹配。
- `H` 表示latent隐藏维度，取值范围为 `[1024, 4096]`，原始隐藏维度为2H，不超过8192；各卡最大BS范围为 `[1, 512]`。
- `K` 取值范围为 `[1, 16]`，且不超过MoE专家与特殊专家总数。K为13至16时，还需满足tiling的UB容量约束。
- `shared_expert_num` 取值范围为 `[1, 4]`；`shared_expert_rank_num > 0`，且满足 `shared_expert_rank_num % shared_expert_num = 0`。
- 每张MoE专家卡部署的专家数为 `L = moe_expert_num / (ep_world_size - shared_expert_rank_num)`，需满足 `L * ep_world_size <= 2048`。
- `tp_world_size` 仅支持1，`tp_rank_id` 和 `expert_shard_type` 仅支持0。
- 各rank的专家分布、量化模式、`y_dtype` 和通信容量配置需保持一致，并以相同顺序调用dispatch/combine。同一通信上下文可连续复用，容量不得小于用例需求。
- 所有rank均需提供 `x` 和 `shared_expert_input`。共享专家按来源rank分担token，每个token向每个逻辑共享专家的一个副本发送一次，各副本部署该专家的完整权重。
- dispatch传入 `shared_expert_input` 时，配对的combine需设置 `enable_independent_shared_expert=True`。不使能独立共享专家时，分别使用None和False。
- dispatch使用 `comm_alg="fullmesh_v2"`，combine使用 `comm_alg=""`。
- dispatch的 `assist_info_for_combine`、`ep_recv_counts` 和有效 `expand_scales` 需传给同轮combine的对应参数。
- 输入Tensor位于本rank的NPU，数据格式为ND。专家计算需保持dispatch输出的专家排列及token顺序。

### quant_mode相关约束

| `quant_mode` | 模式 | MoE `y_dtype` | `scales` |
| --- | --- | --- | --- |
| `0` | 非量化 | `None`，或与输入一致的 `5`（FP16）/`15`（BF16） | `None` |
| `1` | 静态量化 | 必填：`1`（INT8）或 `290`（HIFLOAT8） | 必填，形状规则见下文 |
| `2` | per-token 动态量化 | 必填：`1`、`24/292`（E4M3FN）或 `23/291`（E5M2） | 可选，原有规则不变 |
| `3` | per-group 动态量化，组大小 128 | 必填：`24/292` 或 `23/291` | 可选，原有规则不变 |
| `4` | MX 动态量化，组大小 32 | `None`（默认 E4M3FN）或 `24/292` | 必须为 `None` |
| `5` | MX clip 动态量化，组大小 32 | `None`（默认 E4M3FN）或 `24/292` | 必须为 `None` |

- `quant_mode=1` 时，INT8的 `scales` 支持shape为 `(1,)`、`(H,)` 或 `(shared_expert_num + moe_expert_num, H)`；HIFLOAT8的 `scales` shape为 `(1,)`。
- `quant_mode=2` 或3时，`scales` 可传可不传，专家维shape规则同静态量化。共享专家对应的系数不参与计算。
- `quant_mode=4` 或5时，`scales` 必须为None，量化类型仅支持 `float8_e4m3fn`。
- HIFLOAT8在PyTorch侧使用uint8存储；MX缩放因子的逻辑类型为E8M0，在PyTorch侧使用uint8存储。
- MoE专家卡的 `dynamic_scales` shape及数据类型如下，`ceil` 表示向上取整。

| quant_mode | shape | 数据类型 |
| --- | --- | --- |
| 0、1 | (A,)，输出不作为动态量化系数使用 | float32 |
| 2 | (A,) | float32 |
| 3 | (A, ceil(H / 128)) | float32 |
| 4、5 | (A, 2 * ceil(ceil(H / 32) / 2)) | uint8（逻辑类型E8M0） |

- combine的 `comm_quant_mode` 支持0、2、4，仅对MoE专家通信量化，输出类型与输入 `expand_x` 一致。
- 量化模式下，MoE专家侧需执行反量化或使用支持对应量化输入的专家计算，combine输入为专家计算后的bfloat16/float16结果。

## 调用示例

以2卡场景为例，rank 0部署1个共享专家，rank 1部署2个MoE专家，数据类型为bfloat16，采用非量化模式。示例中专家计算为恒等映射。

保存为 `latent_ep_example.py`，安装匹配的算子包和扩展包后执行：

```bash
torchrun --standalone --nproc_per_node=2 latent_ep_example.py
```

```python
import os
from datetime import timedelta
from pathlib import Path

# 设置JIT缓存路径。
cache_dir = Path(__file__).resolve().parent / "torch_extension_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TORCH_EXTENSIONS_DIR"] = str(cache_dir)

import torch
import torch.distributed as dist
import torch_npu
from cann_ops_transformer.ops import MoeDistributeBuffer, latent_ep


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    timeout = timedelta(seconds=300)
    dist.init_process_group("hccl", timeout=timeout)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise ValueError("This example requires exactly 2 ranks")

    # 初始化EP通信域。
    ep_group = dist.new_group(list(range(world)), backend="hccl", timeout=timeout)
    backend = (ep_group if hasattr(ep_group, "get_hccl_comm_name")
               else ep_group._get_backend(device))
    name = backend.get_hccl_comm_name(dist.get_rank(ep_group), init_comm=True)
    if not name:
        raise RuntimeError("HCCL communicator name is empty")

    bs, hidden, topk = 4, 1024, 2
    size_mib = latent_ep.get_ccl_buffer_size(
        world, bs, hidden, 2, topk, 1, 1, quant_mode=0,
    )
    buffer = MoeDistributeBuffer(ep_group, size_mib * 2 * 1024 * 1024)
    common = dict(
        ep_world_size=world, ep_rank_id=rank, moe_expert_num=2,
        ccl_buffer_size=buffer.ccl_buffer_size,
        tp_world_size=1, tp_rank_id=0, expert_shard_type=0,
        shared_expert_num=1, shared_expert_rank_num=1, global_bs=bs * world,
    )
    all_passed = True
    for step in range(2):
        # 构造latent token和原始token。
        x = torch.full((bs, hidden), rank + step + 1,
                       dtype=torch.bfloat16, device=device)
        original = torch.full((bs, hidden * 2), rank + step + 11,
                              dtype=torch.bfloat16, device=device)
        ids = torch.arange(topk, dtype=torch.int32, device=device).repeat(bs, 1)
        weights = torch.full((bs, topk), 0.5, dtype=torch.float32, device=device)

        expanded, scales, assist, token_nums, ep_counts, tp_counts, expand_scales = (
            latent_ep.dispatch(
                buffer.context, x, ids, **common,
                shared_expert_input=original,
                quant_mode=0, comm_alg="fullmesh_v2",
            )
        )
        # 恒等专家计算，保持token顺序。
        moe_out, shared_out = latent_ep.combine(
            buffer.context, expanded, ids, assist, ep_counts, weights,
            **common, tp_send_counts=tp_counts, comm_quant_mode=0,
            comm_alg="", enable_independent_shared_expert=True,
        )
        torch.npu.synchronize()
        local_pass = torch.equal(moe_out.cpu(), x.cpu()) and torch.equal(
            shared_out.cpu(), original.cpu()
        )
        passed = torch.tensor([int(local_pass)], dtype=torch.int32, device=device)
        dist.all_reduce(passed, op=dist.ReduceOp.MIN, group=ep_group)
        round_passed = bool(passed.item())
        all_passed = all_passed and round_passed
        if rank == 0:
            print(f"Round {step + 1}: {'PASS' if round_passed else 'FAIL'}", flush=True)

    if rank == 0:
        print(f"Summary: {'PASS' if all_passed else 'FAIL'}", flush=True)
    dist.destroy_process_group(ep_group)
    dist.destroy_process_group()
    if not all_passed:
        raise AssertionError("latent_ep output mismatch")


if __name__ == "__main__":
    main()
```
