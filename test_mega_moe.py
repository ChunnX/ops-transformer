import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu

from npu_ops_transformer.ops.mega_moe import get_symm_buffer_for_mega_moe, mega_moe


BS = 32
H = 4096
N = 1024
TOPK = 6
NUM_EXPERTS = 8
# Change this single value to switch between 2-card and 4-card runs.
EP_WORLD_SIZE = 2
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 50123
RESULT_DIR = Path(f"/tmp/mega_moe_e2e_results_w{EP_WORLD_SIZE}")
SEED = 20260507
MAX_DIFF_THRESHOLD = 2.0
MEAN_DIFF_THRESHOLD = 0.2


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def local_experts() -> int:
    return NUM_EXPERTS // EP_WORLD_SIZE


def max_recv_token_num() -> int:
    return BS * EP_WORLD_SIZE * min(TOPK, local_experts())


def validate_config() -> None:
    if EP_WORLD_SIZE not in (2, 4):
        raise RuntimeError(f"EP_WORLD_SIZE only supports 2 or 4, got {EP_WORLD_SIZE}")
    if NUM_EXPERTS % EP_WORLD_SIZE != 0:
        raise RuntimeError(
            f"NUM_EXPERTS({NUM_EXPERTS}) must be divisible by EP_WORLD_SIZE({EP_WORLD_SIZE})"
        )


def cleanup_result_dir() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for path in RESULT_DIR.glob("rank_*"):
        path.unlink()
    for path in RESULT_DIR.glob("*_output.txt"):
        path.unlink()


def to_cpu_tensor(tensor):
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    return tensor


def format_tensor_dump(name: str, tensor) -> str:
    tensor = to_cpu_tensor(tensor)
    shape = tuple(tensor.shape) if hasattr(tensor, "shape") else "unknown"
    dtype = getattr(tensor, "dtype", "unknown")
    values = tensor.tolist() if hasattr(tensor, "tolist") else str(tensor)
    return f"{name}\nshape={shape}\ndtype={dtype}\nvalues={values}\n"


def build_golden_rank_outputs(topk_ids: torch.Tensor, golden_y: torch.Tensor):
    local_expert_num = local_experts()
    topk_id_rows = to_cpu_tensor(topk_ids).tolist()
    golden_rank_outputs = []

    for rank in range(EP_WORLD_SIZE):
        local_start = rank * local_expert_num
        local_end = local_start + local_expert_num
        counts = [0] * local_expert_num

        for expert_id_row in topk_id_rows:
            for expert_id in expert_id_row:
                if local_start <= expert_id < local_end:
                    counts[expert_id - local_start] += 1

        # Every rank uses the same synthetic routing inputs, so the global expected
        # token count is the per-rank histogram multiplied by EP world size.
        counts = [count * EP_WORLD_SIZE for count in counts]
        golden_rank_outputs.append(
            {
                "y": golden_y,
                "expert_token_nums": torch.tensor(counts, dtype=torch.int32),
            }
        )

    return golden_rank_outputs


def archive_outputs(rank_outputs, golden_rank_outputs, archive_dir: Path) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)

    for rank, rank_output in enumerate(rank_outputs):
        golden_output = golden_rank_outputs[rank]
        golden_text = "\n".join(
            [
                format_tensor_dump(f"rank_{rank}_golden_y", golden_output["y"]),
                format_tensor_dump(
                    f"rank_{rank}_golden_expert_token_nums",
                    golden_output["expert_token_nums"],
                ),
            ]
        )
        golden_path = archive_dir / f"rank_{rank}_golden_output.txt"
        golden_path.write_text(golden_text)
        print(golden_text)
        print(f"Archived golden rank {rank} output to: {golden_path}")

        rank_text = "\n".join(
            [
                format_tensor_dump(f"rank_{rank}_y", rank_output["y"]),
                format_tensor_dump(
                    f"rank_{rank}_expert_token_nums",
                    rank_output["expert_token_nums"],
                ),
            ]
        )
        rank_path = archive_dir / f"rank_{rank}_output.txt"
        rank_path.write_text(rank_text)
        print(rank_text)
        print(f"Archived rank {rank} output to: {rank_path}")


def build_inputs():
    torch.manual_seed(SEED)
    x = torch.randn(BS, H, dtype=torch.float32) * 0.05

    routing_scores = torch.softmax(torch.randn(BS, NUM_EXPERTS, dtype=torch.float32), dim=-1)
    topk_weights, topk_ids = torch.topk(routing_scores, k=TOPK, dim=-1, sorted=True)
    topk_ids = topk_ids.to(torch.int32)

    weight1_fp32 = torch.randn(NUM_EXPERTS, N, H, dtype=torch.float32) * 0.05
    weight2_fp32 = torch.randn(NUM_EXPERTS, H, N // 2, dtype=torch.float32) * 0.05

    weight1_q = weight1_fp32.to(torch.float8_e5m2)
    weight2_q = weight2_fp32.to(torch.float8_e5m2)

    weight1_sf = torch.ones(NUM_EXPERTS, N, ceil_div(H, 64), 2, dtype=torch.float32).to(torch.float8_e8m0fnu)
    weight2_sf = torch.ones(NUM_EXPERTS, H, ceil_div(N // 2, 64), 2, dtype=torch.float32).to(torch.float8_e8m0fnu)

    golden_y = golden_mega_moe(
        x=x,
        expert_ids=topk_ids,
        expert_weights=topk_weights,
        weight1=weight1_q.to(torch.float32),
        weight2=weight2_q.to(torch.float32),
    )

    return {
        "x": x,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "weight1_q": weight1_q,
        "weight2_q": weight2_q,
        "weight1_sf": weight1_sf,
        "weight2_sf": weight2_sf,
        "golden_y": golden_y,
    }


def golden_mega_moe(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
) -> torch.Tensor:
    out = torch.zeros((x.shape[0], x.shape[1]), dtype=torch.float32)
    x_ref = x.to(torch.bfloat16).to(torch.float32)

    for batch_idx in range(x.shape[0]):
        x_b = x_ref[batch_idx].unsqueeze(0)
        for topk_idx in range(expert_ids.shape[1]):
            expert_id = expert_ids[batch_idx, topk_idx].item()
            gate = expert_weights[batch_idx, topk_idx].item()

            h1 = torch.matmul(x_b, weight1[expert_id].t())
            h1_left, h1_right = torch.chunk(h1, 2, dim=-1)
            swiglu_out = F.silu(h1_left) * h1_right
            h2 = torch.matmul(swiglu_out, weight2[expert_id].t())

            out[batch_idx] += gate * h2.squeeze(0)

    return out.to(torch.bfloat16)


def get_rank_info():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != EP_WORLD_SIZE:
        raise RuntimeError(f"Expected WORLD_SIZE={EP_WORLD_SIZE}, got {world_size}")
    return rank, local_rank, world_size


def init_process():
    rank, local_rank, world_size = get_rank_info()
    torch_npu.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    ep_group = dist.new_group(ranks=list(range(world_size)), backend="hccl")
    return rank, local_rank, ep_group


def run_rank():
    rank, local_rank, ep_group = init_process()
    output_path = RESULT_DIR / f"rank_{rank}.pt"
    error_path = RESULT_DIR / f"rank_{rank}.err"

    try:
        if rank == 0:
            print(f"[rank0] start run_rank with EP_WORLD_SIZE={EP_WORLD_SIZE}", flush=True)
        inputs = build_inputs()
        if rank == 0:
            print("[rank0] build_inputs done", flush=True)

        x_npu = inputs["x"].to(torch.bfloat16).npu()
        topk_ids_npu = inputs["topk_ids"].to(torch.int32).npu()
        topk_weights_npu = inputs["topk_weights"].to(torch.float32).npu()
        if rank == 0:
            print("[rank0] input tensors moved to npu", flush=True)

        local_start = rank * local_experts()
        local_end = (rank + 1) * local_experts()

        weight1_npu = inputs["weight1_q"][local_start:local_end].npu()
        weight2_npu = inputs["weight2_q"][local_start:local_end].npu()
        weight1_sf_npu = inputs["weight1_sf"][local_start:local_end].npu()
        weight2_sf_npu = inputs["weight2_sf"][local_start:local_end].npu()
        if rank == 0:
            print("[rank0] local expert weights moved to npu", flush=True)

        backend = ep_group._get_backend(torch.device("npu"))
        backend.get_hccl_comm_name(dist.get_rank(ep_group), init_comm=True)
        if rank == 0:
            print("[rank0] hccl comm initialized", flush=True)

        # The installed wrapper forwards num_max_tokens_per_rank into global_bs.
        # Passing 0 matches the doc contract and avoids injecting a wrong value.
        sym_buffer = get_symm_buffer_for_mega_moe(
            ep_group,
            num_experts=NUM_EXPERTS,
            num_max_tokens_per_rank=0,
            num_topk=TOPK,
            hidden=H,
            intermediate_hidden=N,
            max_recv_token_num=max_recv_token_num(),
            dispatch_quant_mode=4,
            dispatch_quant_out_type=23,
        )
        if rank == 0:
            print("[rank0] sym_buffer ready", flush=True)

        y, expert_token_nums = mega_moe(
            x=x_npu,
            topk_ids=topk_ids_npu,
            topk_weights=topk_weights_npu,
            l1_weights=[weight1_npu],
            l2_weights=[weight2_npu],
            sym_buffer=sym_buffer,
            l1_weights_sf=[weight1_sf_npu],
            l2_weights_sf=[weight2_sf_npu],
        )
        if rank == 0:
            print("[rank0] mega_moe done", flush=True)

        torch.npu.synchronize()
        torch.save(
            {"y": y.cpu(), "expert_token_nums": expert_token_nums.cpu()},
            output_path,
        )
        if rank == 0:
            print("[rank0] rank outputs saved", flush=True)

        dist.barrier()
        if rank == 0:
            print("[rank0] validate_outputs start", flush=True)
            validate_outputs(inputs["golden_y"])
    except Exception:
        error_path.write_text(traceback.format_exc())
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def validate_outputs(golden_y: torch.Tensor) -> None:
    rank_outputs = []
    for rank in range(EP_WORLD_SIZE):
        error_path = RESULT_DIR / f"rank_{rank}.err"
        output_path = RESULT_DIR / f"rank_{rank}.pt"

        if error_path.exists():
            print(error_path.read_text())
            raise RuntimeError(f"rank {rank} failed")

        if not output_path.exists():
            raise RuntimeError(f"missing output from rank {rank}")

        rank_outputs.append(torch.load(output_path))

    golden_rank_outputs = build_golden_rank_outputs(
        topk_ids=build_inputs()["topk_ids"],
        golden_y=golden_y,
    )
    archive_outputs(
        rank_outputs=rank_outputs,
        golden_rank_outputs=golden_rank_outputs,
        archive_dir=RESULT_DIR,
    )

    max_diff = 0.0
    mean_diff = 0.0
    for rank, rank_output in enumerate(rank_outputs):
        golden_rank_output = golden_rank_outputs[rank]
        y_npu = rank_output["y"]
        diff = torch.abs(golden_rank_output["y"] - y_npu)
        max_diff = max(max_diff, diff.max().item())
        mean_diff = max(mean_diff, diff.mean().item())

        expert_token_nums = rank_output["expert_token_nums"].to(torch.int32)
        golden_expert_token_nums = golden_rank_output["expert_token_nums"].to(torch.int32)
        if not torch.equal(expert_token_nums, golden_expert_token_nums):
            raise RuntimeError(
                f"rank {rank} expert_token_nums mismatch: "
                f"got {expert_token_nums.tolist()}, expected {golden_expert_token_nums.tolist()}"
            )

    print(f"Max diff: {max_diff}, Mean diff: {mean_diff}")
    if max_diff < MAX_DIFF_THRESHOLD and mean_diff < MEAN_DIFF_THRESHOLD:
        print("Accuracy test passed!")
        return

    raise RuntimeError("Accuracy test failed!")


def launch_distributed() -> int:
    validate_config()
    cleanup_result_dir()

    launcher = shutil.which("torchrun")
    if launcher is None:
        launcher = sys.executable
        base_cmd = [launcher, "-m", "torch.distributed.run"]
    else:
        base_cmd = [launcher]

    cmd = base_cmd + [
        "--nproc_per_node",
        str(EP_WORLD_SIZE),
        "--master_addr",
        MASTER_ADDR,
        "--master_port",
        str(MASTER_PORT),
        __file__,
    ]
    result = subprocess.run(cmd, check=False)
    return result.returncode


if __name__ == "__main__":
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        run_rank()
    else:
        sys.exit(launch_distributed())
