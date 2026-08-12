"""Print CSR/block metadata for one deterministic A2 debug case."""

import argparse

from tests.test_sparse_attention_score_prefill_golden import make_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q-len", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=512)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    case = make_case(
        1, [args.q_len], [args.kv_len], args.q_heads, args.kv_heads,
        args.head_dim, args.block_size, args.top_k, args.seed)
    for name in ("select_idx", "select_num_idx", "k2q_row_ptr", "k2q_q_indices", "k2q_slot_indices", "block_table"):
        if name not in case:
            continue
        print(f"{name}={case[name].tolist()}")


if __name__ == "__main__":
    main()
