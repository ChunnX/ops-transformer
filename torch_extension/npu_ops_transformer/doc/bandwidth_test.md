# bandwidth\_test

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>            |    √    |

## 功能说明

-   API功能：

    实现 MOE（Mixture of Experts）场景下的 AlltoAll Dispatch 通信，用于测试 NPU 设备间的通信带宽。每个 rank 根据 `dst_rank_id` 将 token 发送到对应的目标 rank，实现数据的分布式分发。

    该算子支持两种运行模式：
    1. **mode=0（精度模式/完整流程）**：执行完整的通信流程，包括数据发送、状态同步、等待接收和数据拷贝输出，返回接收到的 token 数据和各 rank 接收的 token 数量。
    2. **mode=1（纯发模式）**：仅执行数据发送和状态同步，用于单向带宽测试，不产生有效输出数据。

-   计算公式：

    - 情形1：如果 mode=0（完整流程）：

    $$
    y = AllToAllV(x, dst\_rank\_id) \\
    receive\_cnt = CountTokensByRank(dst\_rank\_id)
    $$

    其中，`y` 为接收到的 token 数据，`receive_cnt` 为从各 rank 接收的 token 数量。

    - 情形2：如果 mode=1（纯发模式）：

    仅执行发送操作，不产生有效输出数据。

    其中，$x$ 表示参数 `x`，$dst\_rank\_id$ 表示参数 `dst_rank_id`，$y$ 表示参数 `y`，$receive\_cnt$ 表示参数 `receive_cnt`。

## 函数原型

```
npu_bandwidth_test(x, dst_rank_id, group, world_size, max_bs, mode, comm_alg) -> (Tensor, Tensor)
```

## 参数说明

-   **x** (`Tensor`)：必选参数，表示本卡发送的 token 数据，需根据 `dst_rank_id` 发送给其他卡。要求为 2 维张量，shape 为 (BS, H)，表示有 BS 个 token，数据类型支持 `float16`，数据格式为 $ND$，支持非连续的 Tensor。
-   **dst\_rank\_id** (`Tensor`)：必选参数，表示每个 token 的目标 rank 索引，决定每个 token 要发送到哪个 rank。要求为 1 维张量，shape 为 (BS,)，数据类型支持 `int32`，数据格式为 $ND$，支持非连续的 Tensor。张量里 value 取值范围为 [0, world_size)。
-   **group** (`String`)：必选参数，表示通信域名称。字符串长度范围为 [1, 128)，不能为空。
-   **world\_size** (`int`)：必选参数，表示通信域大小（参与通信的 rank 数量）。取值范围需大于 0。
-   **max\_bs** (`int`)：必选参数，表示最大 batch size 大小。取值范围需大于 0，用于确定输出 tensor 的空间大小。
-   **mode** (`int`)：必选参数，表示运行模式。取值范围 [0, 1]：
    - 0：精度模式（完整流程），执行发送、同步、接收和输出拷贝。
    - 1：纯发模式，仅执行发送和同步，用于单向带宽测试。
-   **comm\_alg** (`String`)：必选参数，表示通信算法。可选参数，用于指定通信算法类型。

## 输出说明

-   **y** (`Tensor`)：表示接收到的 token 数据，要求为 2 维张量，shape 为 (max_bs * world_size, H)，数据类型与 `x` 保持一致。数据格式为 $ND$，支持非连续的 Tensor。mode=0 时有有效输出；mode=1 时不产生有效数据。
-   **receive\_cnt** (`Tensor`)：表示从各 rank 接收的 token 数量，要求为 1 维张量，shape 为 (world_size,)，数据类型 `int32`，数据格式为 $ND$，支持非连续的 Tensor。mode=0 时有有效输出；mode=1 时不产生有效数据。

## 约束说明

- **产品支持约束**：
    - 仅支持 <term>Ascend 950PR/Ascend 950DT</term>。

- **参数一致性约束**：
    - 调用算子过程中使用的 `group`、`world_size`、`max_bs` 参数取值所有卡需保持一致。

- **通信域使用约束**：
    - 一个模型中的 `npu_bandwidth_test` 算子仅支持同一通信域，且该通信域中不允许有其他算子。
    - 一个通信域内的节点需在一个超节点内，不支持跨超节点。

- **参数约束**：
    - BS（x.dim0）范围 [1, max_bs]。
    - H（x.dim1）范围 [1, ∞)。
    - world_size 范围 [1, ∞)。
    - `dst_rank_id` 中的每个元素值需在 [0, world_size) 范围内，表示对应 token 发送到的目标 rank。
    - `y` 的数据类型需与 `x` 保持一致。

- **HCCL通信域缓存区大小**：
    调用本接口前需检查 HCCL_BUFFSIZE 环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位 MB。
    - <term>Ascend 950PR/Ascend 950DT</term>：设置大小要求 >= 2 且满足 >= 2 * (max_bs * world_size * Align512(Align32(2 * H) + 64))（`Align512(x) = ((x + 512 - 1) / 512) * 512`；`Align32(x) = ((x + 32 - 1) / 32) * 32`）。

- **驱动约束**：
    - 算子通信域各节点的驱动版本应当相同。

- 本文公式中的"/"表示整除。

## 调用示例

-   单算子模式调用

    ```python
    import os
    import torch
    import torch_npu
    from torch.multiprocessing import Process
    import torch.distributed as dist
    from npu_ops_transformer.ops import npu_bandwidth_test

    BS = 8
    H = 1024
    max_bs = BS
    world_size = 2
    mode = 0  # 精度模式（完整流程）

    server_num = 1
    rank_per_dev = 2
    port = 50001
    master_ip = '127.0.0.1'


    def set_device(rank):
        torch_npu.npu.set_device(rank % rank_per_dev)
        print(f"current device set: {torch_npu.npu.current_device()}")


    def init_hccl_comm(rank):
        print(f'[INFO] device_{rank} 创建HCCL通信链路')
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:{port}')
        print(f"device_{rank} init_process_group success")

        ep_ranks = list(range(world_size))
        ep_group = dist.new_group(backend="hccl", ranks=ep_ranks)
        
        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        
        return ep_hcomm_info, ep_group


    def run_bandwidth_test_npu(rank):
        print(f"{os.getpid()=}{rank=}")
        set_device(rank)
        ep_hcomm_info, ep_group = init_hccl_comm(rank)
        print(f'[INFO] device_{rank} 构造bandwidth_test算子输入数据')

        # 创建输入tensor
        x = torch.randn(BS, H, dtype=torch.float16).npu()
        # 每个token发送到rank (token_id % world_size)
        dst_rank_id = torch.tensor([i % world_size for i in range(BS)], dtype=torch.int32).npu()
        
        # 运行bandwidth_test
        y, receive_cnt = npu_bandwidth_test(
            x=x,
            dst_rank_id=dst_rank_id,
            group=ep_hcomm_info,
            world_size=world_size,
            max_bs=max_bs,
            mode=mode,
            comm_alg=""
        )

        torch.npu.synchronize()
        print(f"[INFO] device_{rank} finish")
        print(f"[INFO] device_{rank} receive_cnt: {receive_cnt.cpu()}")
        dist.destroy_process_group()
        print(f'rank {rank} npu finished! \n')


    if __name__ == "__main__":
        print(f"BS={BS}")
        print(f"H={H}")
        print(f"max_bs={max_bs}")
        print(f"world_size={world_size}")
        print(f"mode={mode}", flush=True)

        p_list = []
        for rank in range(rank_per_dev):
            p = Process(target=run_bandwidth_test_npu, args=(rank,))
            p_list.append(p)

        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

        print("run npu success.")
    ```

    -   图模式调用（暂不支持）