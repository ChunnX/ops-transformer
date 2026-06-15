# mega\_moe

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR/Ascend 950DT</term>            |    √    |

## 功能说明

-   API功能：

    需与[get\_symm\_buffer\_for\_mega\_moe](#get_symm_buffer_for_mega_moe接口说明)配套使用，该算子将`MoE（Mixture of Experts）`以及`FFN（Feed-Forward Network）`的完整计算流程融合为单个算子，实现`Dispatch + GroupMatmul1 + SwiGLUQuant + GroupMatmul2 + Combine`的端到端融合计算。
-   计算公式：

    MegaMoe是一个多阶段融合的MoE算子，在专家并行场景下依次完成Token路由分发与量化、分组矩阵乘法 + SwiGLU激活、二次分组矩阵乘法与跨Rank聚合、以及加权求和，完整计算流程可以分解为以下四个阶段。
 	 
 	   第一阶段对输入Token按专家分组收集后做MXFP8量化，生成各专家的量化输入与缩放因子：
 	 
 	   $$
 	   \hat{X}_e,\ S_{X,e} = \mathrm{Q}_{\text{MX}}\!\left(X[\mathcal{T}_e]\right), \quad e = 0, 1, \ldots, E_{\text{local}}-1
 	   $$
 	 
 	   说明：根据`topkIds`将Token按专家排序收集，$\mathcal{T}_e$ 为分配到专家 $e$ 的Token索引集合，$E_{local}$表示当前专家收到的最大token数，每个专家数值可能不同，$X[\mathcal{T}_e]$ 为对应的子矩阵。$\mathrm{Q}_{\text{MX}}$ 表示MX逐组量化（group size = 32），对每组32个元素提取共享指数后量化为FP8目标类型（FLOAT8_E5M2或FLOAT8_E4M3FN），同时输出FLOAT8_E8M0缩放因子。量化后的数据将作为GMM1的输入。
 	 
 	   第二阶段对每个专家执行GMM1矩阵乘法（将 $W_1$ 沿列方向分为两半分别计算）、SwiGLU激活和MX量化：
 	 
 	   $$
 	   Z_e^{(x)} = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(x)}, S_{1,e}^{(x)}), \quad Z_e^{(y)} = \mathrm{DQ}_{\text{MX}}(\hat{X}_e, S_{X,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{1,e}^{(y)}, S_{1,e}^{(y)})
 	   $$
 	 
 	   $$
 	   U_e = Z_e^{(x)} \odot \sigma\!\left(Z_e^{(x)}\right) \odot Z_e^{(y)}
 	   $$
 	 
 	   $$
 	   \hat{U}_e,\ S_{U,e} = \mathrm{Q}_{\text{MX}}(U_e)
 	   $$
 	 
 	   说明：将 $W_1$ 的前 $N/2$ 列 $W_{1,e}^{(x)}$ 和后 $N/2$ 列 $W_{1,e}^{(y)}$ 分别与MX反量化后的输入做矩阵乘法，得到Swish分支 $Z_e^{(x)}$ 和门控分支 $Z_e^{(y)}$。SwiGLU激活对两个分支做逐元素乘积 $x \cdot \sigma(x) \cdot y$，其中 $\sigma$ 为Sigmoid函数，将中间维度从 $N$ 减半为 $N/2$。随后对SwiGLU输出做MX量化，得到GMM2的量化输入 $\hat{U}_e$。
 	 
 	   第三阶段对每个专家执行GMM2矩阵乘法，并将结果按目标Rank分发：
 	 
 	   $$
 	   O_e = \mathrm{DQ}_{\text{MX}}(\hat{U}_e, S_{U,e}) \cdot \mathrm{DQ}_{\text{MX}}(W_{2,e}, S_{2,e})
 	   $$
 	 
 	   说明：将量化后的SwiGLU输出与第二组权重 $W_2$ 做MX反量化后的矩阵乘法，将 $N/2$ 维中间表示映射回 $H$ 维隐藏空间，得到每个专家的输出 $O_e$。计算完成后通过RDMA peermem将结果按目标Rank的专家偏移地址写入远端，实现跨Rank聚合。
 	 
 	   第四阶段对所有Token按路由权重加权求和，恢复为与输入相同形状的输出：
 	 
 	   $$
 	   Y[i] = \sum_{k=0}^{K-1} W[i,\, k] \cdot O[\pi(i,\, k)]
 	   $$
 	 
 	   说明：对每个Token $i$，根据排序后的路由索引 $\pi(i,k)$ 从聚合后的专家结果中取出对应行，按`topkWeights`中的权重逐元素加权累加，得到最终输出 $Y$。
 	 
 	   其中，$X$ 表示参数`x`，$W$ 表示参数`topkWeights`，$W_1$ 表示参数`weight1`，$W_2$ 表示参数`weight2`，$Y$ 表示参数`y`，$E_{\text{local}}$ 表示属性`moeExpertNum / epWorldSize`（每个Rank的专家数），$K$ 表示`topkIds`的第二维度（top-K值，取值6或8）。
 	 
 	   局部变量说明：
 	   - $\mathcal{T}_e$：被路由到专家 $e$ 的Token索引集合，由`topkIds`排序后确定。
 	   - $\hat{X}_e,\ S_{X,e}$：专家 $e$ 的量化输入及其MX缩放因子，第一阶段中间结果。
 	   - $W_{1,e}^{(x)}$、$W_{1,e}^{(y)}$：$W_1$ 对应专家 $e$ 的前 $N/2$ 列和后 $N/2$ 列子矩阵，由`weight1`按SwiGLU拆分推导。
 	   - $S_{1,e}^{(x)}$、$S_{1,e}^{(y)}$：$W_{1,e}^{(x)}$ 和 $W_{1,e}^{(y)}$ 对应的MX缩放因子，从`weightScales1`按维度截取。
 	   - $S_{2,e}$：$W_{2,e}$ 对应的MX缩放因子，来自参数`weightScales2`。
 	   - $Z_e^{(x)},\ Z_e^{(y)}$：GMM1的两路矩阵乘法输出（Swish分支和门控分支），中间结果。
 	   - $U_e$：SwiGLU激活输出，维度 $m_e \times N/2$，中间结果。
 	   - $\hat{U}_e,\ S_{U,e}$：量化后的SwiGLU输出及其MX缩放因子，中间结果。
 	   - $O_e$：GMM2的专家级输出，维度 $m_e \times H$，中间结果。
 	   - $\pi(i, k)$：Token $i$ 的第 $k$ 个top-k专家在展开排序后的行索引，由路由排序确定。
 	   - $\mathrm{Q}_{\text{MX}}(\cdot)$：MX逐组量化操作，block size = 32，输出FP8数据和E8M0缩放因子。
 	   - $\mathrm{DQ}_{\text{MX}}(\cdot)$：MX逐组反量化操作，在matmul内部隐式执行。
 	

## 函数原型

```
mega_moe(x, topk_idx, topk_weights, l1_weights, l2_weights, sym_buffer, *, scales=None, l1_weights_sf=None, l2_weights_sf=None, x_active_mask=None, weight1_type=28, weight2_type=None) -> (Tensor, Tensor)
```

## 参数说明

-   **x** (`Tensor`)：必选参数，表示计算使用的token数据，需根据`topk_idx`来发送给其他卡。要求为2维张量，shape为(BS, H)，表示有BS个token，数据类型支持`bfloat16`，数据格式为$ND$，支持非连续的Tensor。
-   **topk\_idx** (`Tensor`)：必选参数，表示每个token的topK个专家索引，决定每个token要发给哪些专家。要求为2维张量，shape为(BS, K)，数据类型支持`int32`，数据格式为$ND$，支持非连续的Tensor。张量里value取值范围为[0, num\_experts)，且同一行中的K个value不能重复。
-   **topk\_weights** (`Tensor`)：必选参数，表示每个token的topK个专家的权重，要求为2维张量，shape为(BS, K)，其中共享专家不需要乘权重系数，直接相加即可。数据类型支持`float`，数据格式为$ND$，支持非连续的Tensor。
-   **l1\_weights** (`TensorList`)：必选参数，`GroupMatmul1`计算的右矩阵，用于计算`SwiGLU`激活前的线性变换，支持3维张量，shape为(expertPerRank, N, H)。数据类型支持`float_e5m2`和`float_e4m3fn`，数据格式为$ND$，支持非连续的Tensor。
-   **l2\_weights** (`TensorList`)：必选参数，`GroupMatmul2`计算的右矩阵，用于计算`SwiGLU`激活后的线性变换，支持3维张量，shape为(expertPerRank, H, N/2)。数据类型支持`float_e5m2`和`float_e4m3fn`，数据格式为$ND$，支持非连续的Tensor。
-   **sym\_buffer** (`SymmBuffer`)：必选参数，由[get\_symm\_buffer\_for\_mega\_moe](#get_symm_buffer_for_mega_moe接口说明)接口创建的结构体，封装了以下入参：
    - `context`：本卡通信域信息，数据数据类型支持`int32`。
    - `ep_world_size`：专家并行通信域大小，数据数据类型支持`int64`。
    - `ccl_buffer_size`：CCL通信缓冲区大小，数据数据类型支持`int64`。
    - `num_experts`：MoE模型的总专家数量，数据数据类型支持`int64`。
    - `max_recv_token_num`：每个Rank最大可接收Token数，默认值为0表示自动计算，数据数据类型支持`int64`。
    - `num_max_tokens_per_rank`：每张卡上的token数量，当每个rank的BS不同时，最大的BS大小，数据数据类型支持`int64`。
    - `dispatch_quant_mode`：dispatch通信时量化模式，目前仅支持4（MX模式），数据数据类型支持`int64`。
    - `dispatch_quant_out_type`：dispatch量化后输出的数据类型，支持输入23（torch.float8_e5m2）、24（torch.float8_e4m3fn），数据数据类型支持`int64`。
    - `combine_quant_mode`：暂不支持该参数，使用默认值即可。
    - `comm_alg`：暂不支持该参数，使用默认值即可。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
-   **scales** (`Tensor`)：可选参数，表示每个专家的权重。暂不支持该参数，使用默认值即可。
-   **l1\_weights\_sf** (`TensorList`)：可选参数，量化场景需要，`GroupMatmul1`右矩阵反量化参数，当前支持4维张量，shape为(expertPerRank, N, CeilDiv(H, 64), 2)，其中CeilDiv(H, 64) = (H + 63) / 64。数据类型支持`float8_e8m0`，数据格式为$ND$，支持非连续的Tensor。
-   **l2\_weights\_sf** (`TensorList`)：可选参数，量化场景需要，`GroupMatmul2`右矩阵反量化参数，当前支持4维张量，shape为(expertPerRank, H, CeilDiv(N/2, 64), 2)，其中CeilDiv(N/2, 64) = (N/2 + 63) / 64。数据类型支持`float8_e8m0`，数据格式为$ND$，支持非连续的Tensor。
-   **x\_active\_mask** (`Tensor`)：可选参数，表示token是否参与通信。暂不支持该参数，使用默认值即可。
-   **weight1\_type** (`int`)：暂不支持该参数，使用默认值即可。
-   **weight2\_type** (`int`)：暂不支持该参数，使用默认值即可。

## 输出说明

-   **y\_out** (`Tensor`)：表示本卡收到的token数据，要求为2维张量，shape为(BS, H)，当前数据类型支持`bfloat16`。数据格式为$ND$，支持非连续的Tensor。
-   **expert\_token\_nums** (`Tensor`)：本卡每个专家实际收到的token数量，要求为1维张量，shape为(local\_expert\_num,)，数据类型`int32`，数据格式支持$ND$，支持非连续的Tensor。

## 约束说明

- **参数一致性约束**：
    - 调用算子过程中使用的`epWorldSize`、`globalBs`、`HCCL_BUFFSIZE`等参数取值，所有卡需保持一致，网络中不同层中也需保持一致。

- **通信域使用约束**：
    - 仅支持`EP`域，无`TP`域，不支持`groupTp`、`tpWorldSize`、`tpRankId`属性。
    - 所有卡的`moe_expert_num`、`ep_world_size`、`ccl_buffer_size`、`max_recv_token_num`、`dispatch_quant_mode`、`dispatch_quant_out_type`、`global_bs`参数取值需保持一致。

- **参数约束**：
    - BS（x.dim0）范围 [1, 512]。
    - H（x.dim1）仅支持4096、5120、7168。
    - topK（topkIds.dim1）仅支持6或8。
    - expertPerRank（weight1.dim0）范围 [1, 16]。
    - N（weight1.dim1）仅支持1024、2048、3072、4096、7168。
    - epWorldSize范围 [2, 768]。
    - moeExpertNum范围 [epWorldSize, 1024]，且moeExpertNum % epWorldSize == 0。
    - maxRecvTokenNum范围 [0, BS × epWorldSize × min(topK, expertPerRank)]。
    - dispatchQuantOutType仅支持23（FLOAT8_E5M2）或24（FLOAT8_E4M3FN）。
    - globalBs为0或满足BS × epWorldSize <= globalBs且globalBs % epWorldSize == 0。
    - 当前版本仅支持MXFP量化模式（dispatchQuantMode = 4），dispatch阶段使用MX逐组量化（group size = 32），量化缩放因子类型为FLOAT8_E8M0。
    - xActiveMask和scales参数当前版本必须传入空指针，不支持非空输入。
    - combineQuantMode必须为0，commAlg必须为空字符串""。
    - y的数据类型与x相同。
    - weight1的dim1（N）必须等于weight2的dim2的二倍，这是因为SwiGLU激活需要将中间维度从N减半为N/2。
    - weightScales1和weightScales2不可为空指针。
    - expertPerRank = moeExpertNum / epWorldSize，必须为整数且在 [1, 16] 范围内。
    - weightScales1和weightScales2不可为空指针。

- **MXFP量化场景约束**：
    - weight1 shape为(expertPerRank, N, H)，weight2 shape为(expertPerRank, H, N/2)。
    - weightScales1 shape为(expertPerRank, N, CeilDiv(H, 64), 2)，其中CeilDiv(H, 64) = (H + 63) / 64。
    - weightScales2 shape为(expertPerRank, H, CeilDiv(N/2, 64), 2)，其中CeilDiv(N/2, 64) = (N/2 + 63) / 64。
    - weightScales1的dim3和weightScales2的dim3必须等于2。
    - MXFP场景下，dispatchQuantOutType=23时weight1和weight2必须为FLOAT8_E5M2，dispatchQuantOutType=24时必须为FLOAT8_E4M3FN。
    - xActiveMask和scales必须为空指针。

## get\_symm\_buffer\_for\_mega\_moe接口说明
```
get_symm_buffer_for_mega_moe(group, num_experts: int, num_max_tokens_per_rank: int, num_topk: int, hidden: int, intermediate_hidden: int, *, max_recv_token_num: int = 0, dispatch_quant_mode: int = 0, dispatch_quant_out_type: int = 28, combine_quant_mode: int = 0, comm_alg: str = "") -> SymmBuffer:
```
## 参数说明

- **group**(`String`)：必选参数，EP通信域名称（专家并行通信域）。
- **num_experts**(`int`)：必选参数，MoE模型的总专家数量。
- **num_max_tokens_per_rank**(`int`)：每张卡上的token数量，当每个rank的BS不同时，最大的BS大小。
- **num_topk**(`int`)：必选参数，表示每个token发送的专家数。预留参数，暂不支持。
- **hidden**(`int`)：必选参数，表示每个token大小。预留参数，暂不支持。
- **intermediate_hidden**(`int`)：必选参数，表示中间层投影维度。预留参数，暂不支持。
- <strong>*</strong>：必选参数，代表其之前的变量是位置相关的，必须按照顺序输入；之后的变量是可选参数，位置无关，需要使用键值对赋值，不赋值会使用默认值。
- **max_recv_token_num**(`int`)：可选参数，每个Rank最大可接收Token数，默认值为0表示自动计算。
- **dispatch_quant_mode**(`int`)：可选参数，dispatch通信时量化模式，目前仅支持4（MX模式）。
- **dispatch_quant_out_type**(`int`)：dispatch量化后输出的数据类型，支持输入23（torch.float8_e5m2）、24（torch.float8_e4m3fn）。
- **combine_quant_mode**(`int`)：暂不支持该参数，使用默认值即可。
- **comm_alg**(`String`)：暂不支持该参数，使用默认值即可。

## 输出说明

-   **sym\_buffer** (`SymmBuffer`)：用于封装输入参数并生成`context`、`ep_world_size`和`ccl_buffer_size`。
    
## 调用示例

-   单算子模式调用

    ```python
    import os
    import torch
    import torch_npu
    from torch.multiprocessing import Process, Manager
    import torch.distributed as dist
    from torch.distributed import ReduceOp
    import torch.multiprocessing as mp
    from npu_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
    import torchair

    E = 4
    BS = 256
    H = 4096
    N = 1024
    topK = 6
    num_experts = 8

    server_num = 1
    rank_per_dev = 2
    world_size = server_num * rank_per_dev
    ep_ranks_list = [list(range(tp_id, world_size, 1)) for tp_id in range(1)]
    server_index = 0


    def ceil(a, b):
        return (a + b - 1) // b

    def set_device(rank):
        torch_npu.npu.set_device(rank % rank_per_dev)
        print(f"current device set: {torch_npu.npu.current_device()}")

    def init_hccl_comm(rank):
        print(f'[INFO] device_{rank} 创建HCCL通信链路')
        master_ip = '127.0.0.1'
        dist.init_process_group(backend="hccl", rank=rank, world_size=world_size, init_method=f'tcp://{master_ip}:50001')
        print(f"device_{rank} init_process_group success")
        
        print(f"device {rank} 初始化EP域")
        for ep_ranks in ep_ranks_list:
            tmp_group = dist.new_group(backend="hccl", ranks=ep_ranks)
            if rank in ep_ranks:
                ep_group = tmp_group

        ep_hcomm_info = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        
        return ep_hcomm_info, ep_group

    def get_megamoe_kwargs(
        x, expert_ids, weights1, weights_scales1, weights2, weights_scales2, expert_scales
    ):
        x = x.to(torch.bfloat16).npu()
        expert_ids = expert_ids.to(torch.int32).npu()
        weights1 = weights1.to(torch.float8_e5m2).npu()
        weights_scales1 = weights_scales1.to(torch.float8_e8m0fnu).npu()
        weights2 = weights2.to(torch.float8_e5m2).npu()
        weights_scales2 = weights_scales2.to(torch.float8_e8m0fnu).npu()
        expert_scales = expert_scales.to(torch.bfloat16).npu()

        return {
            'x': x,
            'topk_ids': expert_ids,
            'topk_weights': expert_scales,
            'l1_weights': [weights1],
            'l1_weights_sf': [weights_scales1],
            'l2_weights': [weights2],
            'l2_weights_sf': [weights_scales2],
        }

    def run_megamoe_npu(
        queue, rank, x, expert_ids, weights1, weights_scales1, weights2, weights_scales2, expert_scales
    ):
        print(f"{os.getpid()=}{rank=}")
        set_device(rank)
        ep_hcomm_info, ep_group = init_hccl_comm(rank)
        print(f'[INFO] device_{rank} 构造megamoe算子输入数据')
        megamoe_kwargs = get_megamoe_kwargs(
            x=x,
            expert_ids=expert_ids,
            weights1=weights1,
            weights_scales1=weights_scales1,
            weights2=weights2,
            weights_scales2=weights_scales2,
            expert_scales=expert_scales,
        )
        # 构造distribute_buffer
        distribute_buffer = get_symm_buffer_for_mega_moe(
            ep_group, num_experts=num_experts,
            num_max_tokens_per_rank=0, num_topk=topK,
            hidden=H, intermediate_hidden=0,
            dispatch_quant_mode=4, dispatch_quant_out_type=23
        )
        # 运行mega_moe
        y, expert_token_nums = mega_moe(**megamoe_kwargs, sym_buffer=distribute_buffer)

        torch.npu.synchronize()
        print(f"[INFO] device_{rank} finish\n")
        dist.destroy_process_group()
        print(f'rank {rank} epid {rank} npu finished! \n')
        
        queue.put([
            rank, 
            [
                y.cpu(), expert_token_nums.cpu()
            ]
        ])

    def gen_npu(target_func, **server_kwargs):
        def parse_rank_input(target_func, result_queue, rank, server_kwargs):
            
            ep_id = rank // 1
            
            if target_func == run_megamoe_npu:
                return {
                    "queue": result_queue,
                    "rank": rank,
                    "x": server_kwargs["x_list"][ep_id],
                    "expert_ids": server_kwargs["expert_ids_list"][ep_id],
                    "weights1": server_kwargs["weights1_list"][ep_id],
                    "weights_scales1": server_kwargs["weights_scales1_list"][ep_id],
                    "weights2": server_kwargs["weights2_list"][ep_id],
                    "weights_scales2": server_kwargs["weights_scales2_list"][ep_id],
                    "expert_scales": server_kwargs["expert_scales_list"][ep_id]
                }
        

        print("single_server scene!!!!!")
        rank_list = list(range(world_size))
        print(f"rank list is: {rank_list}")

        proc_list = []
        manager = Manager()
        result_queue = manager.Queue()
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
        expert_idx_shape = [BS, topK]
        weight_shape = [E, N, H]
        weight_scale_shape = [E, N, ceil(H, 64), 2]
        output_shape = [BS, N//2]
        weight2_shape = [E, H, N//2]
        weight2_scale_shape = [E, H, ceil(N//2, 64), 2]
        expert_scales_shape = [BS, topK]
        # 构造输入
        x = torch.randn(x_shape, dtype=torch.bfloat16)
        expert_scales = torch.randn(expert_scales_shape, dtype=torch.bfloat16)
        expert_ids = torch.stack(
            [torch.randperm(num_experts)[:topK] for _ in range(BS)]
        ).to(torch.int32)
        weight1 = torch.randn(weight_shape, dtype=torch.float32).to(torch.float8_e5m2)    
        weight_scales1 = torch.randint(125, 130, weight_scale_shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        weight2 = torch.randn(weight2_shape, dtype=torch.float32).to(torch.float8_e5m2)  
        weight_scales2 = torch.randint(125, 130, weight2_scale_shape, dtype=torch.uint8).view(torch.float8_e8m0fnu)

        golden_x_list = [x.clone() for _ in range(rank_per_dev)]
        golden_expert_ids_list = [expert_ids.clone() for _ in range(rank_per_dev)]
        golden_weights1_list = [weight1.clone() for _ in range(rank_per_dev)]
        golden_weights_scales1_list = [weight_scales1.clone() for _ in range(rank_per_dev)]
        golden_weights2_list = [weight2.clone() for _ in range(rank_per_dev)]
        golden_weights_scales2_list = [weight_scales2.clone() for _ in range(rank_per_dev)]
        golden_expert_scales_list = [expert_scales.clone() for _ in range(rank_per_dev)]

        [y, expert_token_nums] = gen_npu(
            run_megamoe_npu,
            x_list=golden_x_list,
            expert_ids_list=golden_expert_ids_list,
            weights1_list=golden_weights1_list,
            weights_scales1_list=golden_weights_scales1_list,
            weights2_list=golden_weights2_list,
            weights_scales2_list=golden_weights_scales2_list,
            expert_scales_list=golden_expert_scales_list,
        )
    ```

-   图模式调用（暂不支持）

