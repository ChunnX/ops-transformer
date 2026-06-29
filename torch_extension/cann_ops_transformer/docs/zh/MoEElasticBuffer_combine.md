# MoEElasticBuffer.combine<a name="ZH-CN_TOPIC_0000002309174912"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A5 训练系列产品/Atlas A5 推理系列产品</term>            |    √    |

## 功能说明<a name="zh-cn_topic_0000002168254826_section14441124184110"></a>

-   API功能：

    需与[MoEElasticBuffer.dispatch](MoEElasticBuffer_dispatch.md)配套使用，相当于按dispatch算子收集数据的路径原路返回。该接口将专家处理后的token数据根据dispatch阶段记录的元数据信息，通过逆向路由和加权聚合，将token数据组合还原为原始序列顺序。
     - 支持带`topk_weights`加权聚合和纯累加两种模式；
     - 当前版本不支持bias参数。
-   计算公式：
    -   数据整合功能：

      当提供`topk_weights`时：

      $$combined\_x_i = \sum_{k=0}^{K-1} topk\_weights_{i,k} \times x_{slot(i,k)}$$

      当不提供`topk_weights`时（纯累加）：

      $$combined\_x_i = \sum_{k=0}^{K-1} x_{slot(i,k)}$$

## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
MoEElasticBuffer.combine(x, handle, *, topk_weights=None, bias=None) -> (Tensor, Tensor?)
```

## 参数说明<a name="zh-cn_topic_0000002168254826_section112637109429"></a>

-   **x** (`Tensor`)：必选参数，表示经过专家计算后的token数据，即[MoEElasticBuffer.dispatch](MoEElasticBuffer_dispatch.md)输出的`recv_x`经过专家网络处理后的结果。要求为2维张量，shape为(A, H)，数据类型仅支持`bfloat16`，数据格式为$ND$。

-   **handle** (`EPHandle`)：必选参数，表示[MoEElasticBuffer.dispatch](MoEElasticBuffer_dispatch.md)返回的handle对象，包含slot索引、接收元数据等信息。handle的属性参见[MoEElasticBuffer.dispatch](MoEElasticBuffer_dispatch.md)输出说明。

- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的；之后的变量是可选参数，需要使用键值对赋值，不赋值会使用默认值。

-   **topk\_weights** (`Tensor`)：可选参数，表示每个token对应的topK专家权重，用于加权聚合。要求为1维张量，shape为(A,)，数据类型支持`float32`，数据格式为$ND$，对应[MoEElasticBuffer.dispatch](MoEElasticBuffer_dispatch.md)的`recv_topk_weights`输出。若不提供，则进行纯累加combine，输出`combined_topk_weights`为None。

-   **bias** (`Tensor`或`Tuple[Tensor, Tensor]`)：可选参数，当前版本不支持bias，传入None即可。预留支持单个bias张量或bias\_0/bias\_1双张量模式。

## 返回值说明<a name="zh-cn_topic_0000002168254826_section22231435517"></a>

-   **combined\_x** (`Tensor`)：表示combine后的token数据，还原为原始序列顺序。要求为2维张量，shape为(BS, H)，数据类型为`bfloat16`，数据格式为$ND$，不支持非连续的Tensor。

-   **combined\_topk\_weights** (`Tensor`)：表示combine后的topK专家权重。当`topk_weights`输入不为None时，要求为2维张量，shape为(BS, K)，数据类型为`float32`，数据格式为$ND$；当`topk_weights`输入为None时，返回None。

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

-   该接口支持推理和训练场景。
-   `MoEElasticBuffer.dispatch`和`MoEElasticBuffer.combine`必须配套使用。
-   当前版本不支持bias参数，`bias`必须传入None。
-   调用接口过程中使用的`ep_world_size`、`num_experts`、`num_max_tokens_per_rank`参数需通过handle传递，与[MoEElasticBuffer.dispatch](MoEElasticBuffer_dispatch.md)保持一致。
-   参数里Shape使用的变量如下：
    -   A：表示本卡接收的最大token数量，A = ep\_world\_size \* num\_max\_tokens\_per\_rank \* MIN(K, num\_local\_experts)。
    -   H：表示hidden size隐藏层大小。取值范围为(0, 8192]。
    -   BS：表示batch sequence size，即本卡的原始token数量。
    -   K：表示选取topK个专家，取值范围为1 ≤ K ≤ 32。
    -   num\_local\_experts：表示本卡专家数量，num\_local\_experts = num\_experts / ep\_world\_size。
-   HCCL通信域缓存区大小：
    调用本接口前需检查通信域缓存区大小取值是否合理，单位MB，不配置时默认为200MB。
    -   通信域缓存区大小可通过调用[MoEElasticBuffer.get\_moe\_ep\_ccl\_buffer\_size](MoEElasticBuffer_get_moe_ep_ccl_buffer_size.md)接口计算。
    -   设置大小要求 >= 2 \* (ep\_world\_size \* num\_max\_tokens\_per\_rank \* Align512(Align32(2 \* H)))。
    -   其中 Align512(x) = ((x+512-1)/512)\*512, Align32(x) = ((x+32-1)/32)\*32。
-   通信域使用约束：
    -   一个模型中的`MoEElasticBuffer.dispatch`和`MoEElasticBuffer.combine`算子仅支持相同EP通信域，且该通信域中不允许有其他算子。

## 调用示例<a name="zh-cn_topic_0000002168254826_section14459801435"></a>

-   单算子模式调用

    ```python
    import os
    import torch
    import torch_npu
    import torch.distributed as dist
    from torch.multiprocessing import Process
    from npu_ops_transformer.ops.moe_elastic_buffer import MoEElasticBuffer

    master_ip = '127.0.0.1'
    world_size = 2

    def run_dispatch_combine(rank):
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                                 init_method=f'tcp://{master_ip}:50002')

        num_experts = world_size * 4
        num_max_tokens_per_rank = 128
        hidden = 4096
        top_k = 4

        group = dist.new_group(backend="hccl")
        buf = MoEElasticBuffer(
            group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden,
            num_topk=top_k,
            num_experts=num_experts,
        )

        num_tokens = 64
        x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=f"npu:{rank}")
        topk_idx = torch.randint(0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=f"npu:{rank}")
        topk_weights = torch.rand(num_tokens, top_k, dtype=torch.float32, device=f"npu:{rank}")

        # dispatch阶段
        recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(x, topk_idx=topk_idx, topk_weights=topk_weights)

        torch.npu.synchronize()

        # 专家计算（此处为示例，实际为专家网络forward）
        expert_output = recv_x

        # combine阶段：带topk_weights加权聚合
        combined_x, combined_topk_weights = buf.combine(expert_output, handle, topk_weights=recv_topk_weights)

        torch.npu.synchronize()
        print(f"[rank {rank}] combined_x shape={combined_x.shape}, expected ({num_tokens}, {hidden})")
        assert combined_x.shape == (num_tokens, hidden)
        print(f"[rank {rank}] dispatch_combine PASS")

        buf.destroy()
        dist.destroy_process_group()


    if __name__ == "__main__":
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        processes = []
        for rank in range(world_size):
            p = Process(target=run_dispatch_combine, args=(rank,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print("All processes finished.")
    ```