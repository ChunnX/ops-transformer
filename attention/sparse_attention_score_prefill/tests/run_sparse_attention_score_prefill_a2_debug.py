"""Run one deterministic A2 debug probe and persist comparable tensors.

The A2 kernel's DEBUG_MODE selects which Phase1 workspace view is returned in
the normal attention output buffer.  This runner always writes the CPU Phase1
golden workspace and the NPU return tensor into one fresh directory, so the
matching comparator can report an element-wise result without relying on
device-side DumpTensor descriptor logs.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch_npu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", required=True)
    parser.add_argument("--q-len", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=512)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-precise", type=int, default=0)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def main():
    args = parse_args()
    dump_dir = Path(args.dump_dir).resolve()
    # The workflow wrapper creates the directory first so it can place the
    # live kernel log and metadata beside the tensor dumps.  Keep direct runs
    # idempotent for a freshly allocated directory while never deleting files.
    dump_dir.mkdir(parents=True, exist_ok=True)

    # The golden module reads these once at import time.
    os.environ["GOLDEN_DUMP"] = "1"
    os.environ["GOLDEN_DUMP_DIR"] = str(dump_dir)
    os.environ.setdefault("CANN_OPS_TRANSFORMER_LAZY_IMPORT", "1")

    this_dir = Path(__file__).resolve().parent
    repo_root = this_dir.parents[2]
    sys.path.insert(0, str(repo_root / "torch_extension"))
    sys.path.insert(0, str(this_dir))

    from test_sparse_attention_score_prefill_golden import (  # pylint: disable=import-outside-toplevel
        cpu_golden_prefill_phase1,
        cpu_golden_prefill_phase2,
        make_case,
    )
    import cann_ops_transformer.ops.sparse_attention_score_prefill  # pylint: disable=import-outside-toplevel,unused-import

    torch_npu.npu.set_device(args.device)
    case = make_case(
        1, [args.q_len], [args.kv_len], args.q_heads, args.kv_heads,
        args.head_dim, args.block_size, args.top_k, args.seed)
    torch.save({
        "query": case["query"],
        "key": case["key"],
        "value": case["value"],
        "block_table": case["block_table"],
        "k2q_row_ptr": case["k2q_row_ptr"],
        "k2q_q_indices": case["k2q_q_indices"],
        "k2q_slot_indices": case["k2q_slot_indices"],
        "actual_seq_lengths": case["actual_seq_lengths"],
        "actual_seq_lengths_kv": case["actual_seq_lengths_kv"],
    }, dump_dir / "case_inputs.pt")
    ws_o, ws_max, ws_sum = cpu_golden_prefill_phase1(
        case["query"], case["key"], case["value"],
        case["k2q_row_ptr"], case["k2q_q_indices"], case["k2q_slot_indices"],
        case["block_table"], case["actual_seq_lengths"], case["actual_seq_lengths_kv"],
        args.block_size, args.top_k, case["scale_value"], args.inner_precise)
    torch.save(ws_o, dump_dir / "phase1_ws_o.pt")
    torch.save(ws_max, dump_dir / "phase1_ws_max.pt")
    torch.save(ws_sum, dump_dir / "phase1_ws_sum.pt")
    golden_final = cpu_golden_prefill_phase2(
        ws_o, ws_max, ws_sum, args.top_k, args.kv_heads,
        args.q_heads // args.kv_heads, args.head_dim, dump_dir=dump_dir)
    torch.save(golden_final, dump_dir / "golden_final.pt")

    npu_out = torch.ops.cann_ops_transformer.npu_sparse_attention_score_prefill(
        case["query"].to(args.device), case["key"].to(args.device), case["value"].to(args.device),
        case["block_table"].to(args.device), case["k2q_row_ptr"].to(args.device),
        case["k2q_q_indices"].to(args.device), case["k2q_slot_indices"].to(args.device),
        case["actual_seq_lengths"].to(args.device), case["actual_seq_lengths_kv"].to(args.device),
        args.kv_heads, case["scale_value"], args.block_size, args.top_k, args.inner_precise)
    torch_npu.npu.synchronize()
    actual = npu_out.cpu()
    torch.save(actual, dump_dir / "npu_debug_output.pt")

    print(f"[a2-debug] dump_dir={dump_dir}")
    print(
        "[a2-debug] "
        f"q={args.q_len} kv={args.kv_len} q_heads={args.q_heads} "
        f"kv_heads={args.kv_heads} top_k={args.top_k} seed={args.seed} "
        f"inner_precise={args.inner_precise}")
    print(f"[a2-debug] golden_final={tuple(golden_final.shape)} {golden_final.dtype}")
    print(f"[a2-debug] golden_phase2_dir={dump_dir}")
    print(f"[a2-debug] npu_debug_output={tuple(actual.shape)} {actual.dtype}")
    print(
        f"[a2-debug] npu_range=({actual.float().min().item():.9g}, "
        f"{actual.float().max().item():.9g})")


if __name__ == "__main__":
    main()
