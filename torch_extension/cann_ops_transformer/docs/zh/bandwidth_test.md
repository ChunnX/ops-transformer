# bandwidth\_test

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √    |

## 功能说明

-   API功能：

    该算子实现MOE（Mixture of Experts）场景下的`AlltoAll Dispatch`通信带宽测试，用于测试NPU设备间的通信带宽性能。每个rank根据`dstrank_id`将token发送到对应的目标rank，实现数据的分布式分发。

    执行流程（mode=0 完整流程）：
    1. SendToMoeExpert：根据`dstrank_id`将token写入目标rank的通信窗口
    2. SetStatus：统计发送到各rank的token数量，写入状态区通知对端
    3. WaitDispatch：等待所有rank完成数据发送并写入状态
    4. LocalWindowCopy：从本地窗口读取接收到的数据，拷贝到输出tensor

    执行流程（mode=1 仅发送）：
    仅执行SendToMoeExpert，用于单向带宽测试。

-   计算公式：

    ```
    y = AlltoAllV(x, dstrank_id)
    receive_cnt = 统计从各rank接收的token数量
    ```

    其中：
    - 每个token根据`dstrank_id`指定的目标rank进行分发
    - 数据通过HCCL通信窗口在rank间传输
    - `receive_cnt`记录从每个rank接收到的token数量

## 函数原型

```
npu_bandwidth_test(x, dstrank_id, group, world_size, max_bs, mode, comm_alg='', aiv_num=-1) -> (Tensor, Tensor)
```

## 参数说明

-   **x** (`Tensor`)：必选参数，表示本卡发送的token数据。要求为2维张量，shape为(BS, H)，其中BS表示batch size（token数量），H表示hidden size（特征维度）。数据类型支持`float16`、`bfloat16`、`float32`，数据格式为ND，支持非连续的Tensor。
-   **dst\_rank\_id** (`Tensor`)：必选参数，表示每个token的目标rank索引。要求为1维张量，shape为(BS,)，每个元素表示对应token要发送到哪个rank，取值范围为[0, world_size)。数据类型支持`int32`，数据格式为ND，支持非连续的Tensor。
-   **group** (`String`)：必选参数，表示通信域名称（EP通信域）。字符串长度范围为[1, 128)。
-   **world\_size** (`int`)：必选参数，表示通信域大小（rank总数）。取值范围为[2, 768]。
-   **max\_bs** (`int`)：必选参数，表示最大batch size大小。
-   **mode** (`int`)：必选参数，表示执行模式。取值范围：
    - 0：精度模式（完整流程：发送+同步+接收）
    - 1：纯发模式（仅发送，用于单向带宽测试）
-   **comm\_alg** (`String`)：可选参数，表示通信算法。默认为空字符串。
-   **aiv\_num** (`int`)：可选参数，表示AIV核数量。默认为-1，表示自动获取。

## 输出说明

-   **y** (`Tensor`)：表示输出的token数据。要求为2维张量，shape为(max_recv_cnt, H)，其中max_recv_cnt = max_bs × world_size，表示预分配的最大接收token数量。数据类型与输入x一致，支持`float16`、`bfloat16`、`float32`，数据格式为ND，支持非连续的Tensor。
-   **receive\_cnt** (`Tensor`)：表示从各rank接收的token数量。要求为1维张量，shape为(world_size,)，每个元素表示从对应rank接收的token数量。数据类型支持`int32`，数据格式为ND，支持非连续的Tensor。

## 约束说明

- **参数一致性约束**：
    - 调用算子过程中使用的 `world_size`、`max_bs`、`HCCL_BUFFSIZE`等参数取值，所有卡需保持一致，网络中不同层中也需保持一致。

- **通信域使用约束**：
    - 仅支持 `EP`域，无 `TP` 域。
    - 一个通信域内的节点需在一个超节点内，不支持跨超节点。
    - 所有卡的 `world_size`、`max_bs` 参数取值需保持一致。

- **参数约束**：
    - world\_size 范围 [2, 768]。
    - dst\_rank\_id 中每个元素的取值范围必须为 [0, world\_size)。
    - mode 取值必须为 0 或 1。
    - group 字符串长度必须小于128。

- **数据类型约束**：
    - 输入 `x` 和输出 `y` 的数据类型必须一致，支持 `float16`。
    - 输入 `dstrank_id` 的数据类型必须为 `int32`。
    - 输出 `receive_cnt` 的数据类型必须为 `int32`。

- **环境变量约束**：
    - 调用本接口前需检查 `HCCL_BUFFSIZE` 环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB。
    - 设置大小要求：≥ 2 × (max_bs × world_size × H × sizeof(dtype) + 2MB)。

- **驱动约束**：
    - 算子通信域各节点的驱动版本应当相同。

## 调用示例

-   单算子模式调用

    ```python
    import os
    import torch
    import torch_npu
    from torch.multiprocessing import Process, Manager
    import torch.distributed as dist
    from npu_ops_transformer.ops import npu_bandwidth_test
    import torchair

    BS = 8
    H = 4096
    max_bs = BS
    world_size = 2
    mode = 0  # 0: 完整流程, 1: 仅发送模式

    server_num = 1
    rank_per_dev = 2
    total_world_size = server_num * rank_per_dev
    ep_ranks_list = [list(range(tp_id, total_world_size, 1)) for tp_id in range(1)]
    server_index = 0


    def set_device(rank):
        torch_npu.npu.set_device(rank % rank_per_dev)
        print(f"current device set: {torch_npu.npu.current_device()}")

    def init_hccl_comm(rank):
        print(f'[INFO] device_{rank} 创建HCCL通信链路')
        master_ip = '127.0.0.1'
        dist.init_process_group(backend="hccl", rank=rank, world_size=total_world_size, init_method=f'tcp://{master_ip}:50001')
        print(f"device_{rank} init_process_group success")
        
        print(f"device {rank} 初始化EP域")
        for ep_ranks in ep_ranks_list:
            tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group = tmp_group

        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        
        return ep_hcomm_info, ep_group

    def run_bandwidth_test_npu(queue, rank, x, dstrank_id):
        print(f"{os.getpid()=}{rank=}")
        set_device(rank)
        ep_hcomm_info, ep_group = init_hccl_comm(rank)
        
        print(f'[INFO] device_{rank} 构造bandwidth_test算子输入数据')
        x = x.to(torch.bfloat16).npu()
        dstrank_id = dstrank_id.to(torch.int32).npu()
        
        # 运行bandwidth_test
        y, receive_cnt = npu_bandwidth_test(
            x=x,
            dstrank_id=dstrank_id,
            group=ep_hcomm_info,
            world_size=world_size,
            max_bs=max_bs,
            mode=mode,
            comm_alg='',
            aiv_num=-1
        )

        torch.npu.synchronize()
        print(f"[INFO] device_{rank} finish\n")
        print(f"[INFO] device_{rank} receive_cnt: {receive_cnt.cpu()}\n")
        dist.destroy_process_group()
        print(f'rank {rank} npu finished! \n')
        
        queue.put([
            rank, 
            [
                y.cpu(), receive_cnt.cpu()
            ]
        ])

    def gen_npu(target_func, **server_kwargs):
        def parse_rank_input(target_func, result_queue, rank, server_kwargs):
            ep_id = rank // 1
            
            if target_func == run_bandwidth_test_npu:
                return {
                    "queue": result_queue,
                    "rank": rank,
                    "x": server_kwargs["x_list"][ep_id],
                    "dstrank_id": server_kwargs["dstrank_id_list"][ep_id]
                }

        print("single_server scene!!!!!")
        rank_list = list(range(total_world_size))
        print(f"rank list is: {rank_list}")

        proc_list = []
        manager = Manager()
        result_queue = manager.Queue()
        
        from torch.multiprocessing import mp
        mp.set_start_method("forkserver", force=True)
        
        for rank in rank_list:
            rank_kwargs = parse_rank_input(target_func, result_queue, rank, server_kwargs)
            proc = Process(target=target_func, kwargs=rank_kwargs)
            proc.start()
            proc_list.append(proc)

        rank_outputs = [None] * rank_per_dev
        for proc in proc_list:
            rank_id, rank_output = result_queue.get()
            local_rank_id = rank_id - server_index * rank_per_dev
            rank_outputs[local_rank_id] = rank_output

        for proc in proc_list:
            proc.join()

        if None in rank_outputs:
            print("[ERROR] Task failed! Please check the detailed error logs printed by the subprocesses.")
            exit(1)
        
        # 将各类输出放入同一个列表中，category_outputs存储各类输出的列表
        category_outputs = []
        category_num = len(rank_outputs[0])
        for category_id in range(category_num):
            specific_category_output = [rank_output[category_id] for rank_output in rank_outputs]
            category_outputs.append(specific_category_output)

        return category_outputs

    if __name__ == "__main__":
        x_shape = [BS, H]
        dstrank_id_shape = [BS]
        
        # 构造输入
        x = torch.randn(x_shape, dtype=torch.bfloat16)
        # 每个token发送到不同的rank（循环分配）
        dstrank_id = torch.tensor([i % world_size for i in range(BS)], dtype=torch.int32)

        golden_x_list = [x.clone() for _ in range(rank_per_dev)]
        golden_dstrank_id_list = [dstrank_id.clone() for _ in range(rank_per_dev)]

        [y_list, receive_cnt_list] = gen_npu(
            run_bandwidth_test_npu,
            x_list=golden_x_list,
            dstrank_id_list=golden_dstrank_id_list,
        )
        
        print(f"[INFO] bandwidth_test finished successfully!")
        for i, receive_cnt in enumerate(receive_cnt_list):
            print(f"[INFO] rank_{i} receive_cnt: {receive_cnt}")
    ```
