"""
A5 MTE MoE Distribute Dispatch/Combine fast_kernel_launch 测试
用法:
    torchrun --nproc_per_node=2 test_a5_mte.py
"""
import torch
import torch_npu
import torch.distributed as dist
import ascend_ops


def main():
    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch_npu.npu.set_device(rank)

    ep_world = world
    ep_rank = rank
    moe_num = 4
    bs, h, k = 4, 128, 2
    dtype = torch.float16

    # 创建 EP 通信组，获取 HCCL comm 名称
    ep_group = dist.new_group(backend="hccl", ranks=list(range(ep_world)))
    group_ep = ep_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)

    x = torch.randn(bs, h, dtype=dtype).npu()
    eid = torch.randint(0, moe_num, (bs, k), dtype=torch.int32).npu()

    if rank == 0:
        print(f"[A5] rank={rank} ep={ep_rank}/{ep_world} bs={bs} h={h} k={k}")

    # Step 1: 创建 mc2Context（获取真实HCCL buffer地址）
    mc2_ctx, buf_size = torch.ops.ascend_ops.updateContext(x, group_ep, ep_world)
    if rank == 0:
        print(f"[A5] updateContext done, buf_size={buf_size}")

    # Step 2: Dispatch（mc2Context 直接传给 kernel）
    out = torch.ops.ascend_ops.npu_moe_distribute_dispatch_v2_a5(
        x, eid, group_ep, ep_world, ep_rank, moe_num,
        mc2_ctx, buf_size,
    )
    expand_x, dyn_scales, assist, token_nums, ep_cnts, _ = out
    if rank == 0:
        print(f"[A5] Dispatch: expand_x={list(expand_x.shape)} tokens={token_nums.tolist()}")

    # Step 3: Combine
    x_out = torch.ops.ascend_ops.npu_moe_distribute_combine_v2_a5(
        expand_x, eid, assist, ep_cnts,
        dyn_scales, group_ep, ep_world, ep_rank, moe_num,
        mc2_ctx, buf_size,
    )
    if rank == 0:
        print(f"[A5] Combine: x_out={list(x_out.shape)}")
        print(f"[A5] PASS")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
