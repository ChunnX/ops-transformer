# MoEElasticBuffer.dispatch<a name="ZH-CN_TOPIC_0000002343094193"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A5 训练系列产品/Atlas A5 推理系列产品</term>            |    √    |

## 功能说明<a name="zh-cn_topic_0000002203575833_section14441124184110"></a>

-   API功能：需与[MoEElasticBuffer.combine](MoEElasticBuffer_combine.md)配套使用，完成MoE的Expert Parallelism（EP）并行部署下的token dispatch。该接口根据每个token的topK专家索引，将token数据通过EP域的alltoallv通信分发到对应的专家卡上。
     - 支持cached模式，即第二次dispatch时可复用第一次的handle，跳过slot分配阶段，实现更低延迟；
     - 支持指定`num_max_tokens_per_rank`控制接收buffer大小上限。
-   计算公式：

     $$alltoall\_x\_out = alltoallv(x)$$

     $$dst\_buffer\_slot\_idx = SlotAssignment(topk\_idx)$$

     $$recv\_src\_metadata = MetadataAssignment(alltoall\_x\_out, topk\_idx)$$

## 函数原型<a name="zh-cn_topic_0000002203575833_section45077510411"></a>

```
MoEElasticBuffer.dispatch(x, topk_idx=None, topk_weights=None, *, handle=None, num_experts=None, num_max_tokens_per_rank=None, expert_alignment=None, do_cpu_sync=None) -> (Tensor, Tensor, Tensor, EPHandle)
```

## 参数说明<a name="zh-cn_topic_0000002203575833_section112637109429"></a>

-   **x** (`Tensor`)：必选参数，表示计算使用的token数据，需根据`topk_idx`来发送给其他卡。要求为2维张量，shape为(BS, H)，数据类型支持`bfloat16`、`float16`、`float8_e5m2`、`float8_e4m3fn`，数据格式为$ND$。
-   **topk\_idx** (`Tensor`)：可选参数，表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2维张量，shape为(BS, K)，数据类型支持`int32`，数据格式为$ND$。张量里value取值范围为\[0, num\_experts\)。非cached模式下为必选参数，cached模式下必须为None。
-   **topk\_weights** (`Tensor`)：可选参数，表示每个token对应的topK专家权重。要求为2维张量，shape为(BS, K)，数据类型支持`float32`，数据格式为$ND$。非cached模式下为可选参数，cached模式下必须为None。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的；之后的变量是可选参数，需要使用键值对赋值，不赋值会使用默认值。
-   **handle** (`EPHandle`)：可选参数，表示上一次dispatch返回的handle对象，用于cached模式。传入handle时，`topk_idx`和`topk_weights`必须为None，`do_cpu_sync`必须为False。默认为None，即非cached模式。
-   **num\_experts** (`int`)：可选参数，MoE专家总数量，覆盖初始化时的值。取值范围\[2, 2048\]，且满足num\_experts % ep\_world\_size = 0。默认使用初始化时传入的值。
-   **num\_max\_tokens\_per\_rank** (`int`)：可选参数，表示每张卡上的最大token数量上限，覆盖初始化时的值。默认使用初始化时传入的值。
-   **expert\_alignment** (`int`)：可选参数，表示专家对齐数，覆盖初始化时的值。默认使用初始化时传入的值。
-   **do\_cpu\_sync** (`bool`)：可选参数，表示是否进行CPU同步等待。非cached模式默认为True，cached模式必须为False。

## 输出说明<a name="zh-cn_topic_0000002203575833_section22231435517"></a>

-   **recv\_x** (`Tensor`)：表示本卡收到的token数据。要求为2维张量，shape为(A, H)，数据类型为`bfloat16`，数据格式为$ND$。经专家网络处理后，作为[MoEElasticBuffer.combine](MoEElasticBuffer_combine.md)的`x`输入。
-   **recv\_topk\_idx** (`None`)：当前版本始终为None，预留参数。
-   **recv\_topk\_weights** (`Tensor | None`)：表示本卡收到的topk权重。仅当输入`topk_weights`不为None时返回，否则为None。要求为1维张量，shape为(A,)，数据类型为`float32`，数据格式为$ND$，作为[MoEElasticBuffer.combine](MoEElasticBuffer_combine.md)的`topk_weights`输入。
-   **handle** (`EPHandle`)：表示dispatch阶段生成的handle对象，包含slot索引、元数据等信息，需传递给[MoEElasticBuffer.combine](MoEElasticBuffer_combine.md)使用。handle的属性如下：
    -   **dst\_buffer\_slot\_idx** (`Tensor`)：slot索引，shape为(BS, K)，dtype为`int32`。
    -   **recv\_src\_metadata** (`Tensor`)：接收元数据，shape为(A, 3)，dtype为`int32`。
    -   **num\_recv\_tokens\_per\_rank** (`Tensor`)：各卡接收token数量，shape为(ep\_world\_size,)，dtype为`int32`。
    -   **num\_recv\_tokens\_per\_expert** (`Tensor`)：每个本地专家接收的token数量，shape为(num\_local\_experts,)，dtype为`int64`。
    -   **num\_experts** (`int`)：专家总数量。
    -   **expert\_alignment** (`int`)：专家对齐数。
    -   **num\_max\_tokens\_per\_rank** (`int`)：每张卡最大token数量上限。
    -   **topk\_idx** (`Tensor`)：原始topk索引。

## 约束说明<a name="zh-cn_topic_0000002203575833_section12345537164214"></a>

-   该接口支持推理和训练场景。
-   `MoEElasticBuffer.dispatch`和[MoEElasticBuffer.combine](MoEElasticBuffer_combine.md)必须配套使用。
-   调用接口过程中使用的`num_experts`、`num_max_tokens_per_rank`参数取值所有卡需保持一致，且和[MoEElasticBuffer.combine](MoEElasticBuffer_combine.md)对应参数也保持一致。
-   cached模式下，`topk_idx`和`topk_weights`必须为None，`do_cpu_sync`必须为False；非cached模式下，`topk_idx`为必选参数，`topk_weights`为可选参数。
-   参数里Shape使用的变量如下：
    -   A：表示本卡接收的最大token数量，A = ep\_world\_size \* num\_max\_tokens\_per\_rank \* MIN(K, num\_local\_experts)。
    -   H：表示hidden size隐藏层大小。取值范围为(0, 8192]。
    -   BS：表示batch sequence size，即本卡的token数量。
    -   K：表示选取topK个专家，取值范围为1 ≤ K ≤ 32。
    -   num\_local\_experts：表示本卡专家数量，num\_local\_experts = num\_experts / ep\_world\_size，应满足0 < num\_local\_experts \* ep\_world\_size ≤ 2048。
-   HCCL通信域缓存区大小：
    调用本接口前需检查HCCL\_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
    -   通信域缓存区大小可通过调用[MoEElasticBuffer.get\_moe\_ep\_ccl\_buffer\_size](MoEElasticBuffer_get_moe_ep_ccl_buffer_size.md)接口计算。
    -   设置大小要求 >= 2 \* (ep\_world\_size \* num\_max\_tokens\_per\_rank \* Align512(Align32(2 \* H)))。
    -   其中 Align512(x) = ((x+512-1)/512)\*512, Align32(x) = ((x+32-1)/32)\*32。
-   通信域使用约束：
    -   一个模型中的`MoEElasticBuffer.dispatch`和[MoEElasticBuffer.combine](MoEElasticBuffer_combine.md)算子仅支持相同EP通信域，且该通信域中不允许有其他算子。

## 调用示例<a name="zh-cn_topic_0000002203575833_section14459801435"></a>

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

    def run_dispatch(rank):
        torch_npu.npu.set_device(rank)
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size,
                                 init_method=f'tcp://{master_ip}:50001')

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

        recv_x, recv_topk_idx, recv_topk_weights, handle = buf.dispatch(x, topk_idx, topk_weights)

        torch.npu.synchronize()
        print(f"[rank {rank}] recv_x shape={recv_x.shape}")

        buf.destroy()
        dist.destroy_process_group()


    if __name__ == "__main__":
        os.environ['HCCL_WHITELIST_DISABLE'] = '1'
        processes = []
        for rank in range(world_size):
            p = Process(target=run_dispatch, args=(rank,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print("All processes finished.")
    ```