"""Print per-task finite and error summaries for an A2 opartial debug run."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_dir")
    args = parser.parse_args()
    actual = torch.load(f"{args.dump_dir}/actual_opartial.pt", map_location="cpu").float()
    golden = torch.load(f"{args.dump_dir}/golden_opartial.pt", map_location="cpu").float()
    finite = torch.isfinite(actual)
    diff = (actual - golden).abs()
    finite_diff = torch.where(finite, diff, torch.zeros_like(diff))
    print(f"shape={tuple(actual.shape)}")
    print(f"nan_count={torch.isnan(actual).sum(dim=-1).tolist()}")
    print(f"finite_by_task={finite.all(dim=-1).tolist()}")
    print(f"max_diff_by_task={finite_diff.amax(dim=-1).tolist()}")
    print(f"mean_diff_by_task={torch.nanmean(diff, dim=-1).tolist()}")
    ws_max = torch.load(f"{args.dump_dir}/phase1_ws_max.pt", map_location="cpu").float()
    ws_o = torch.load(f"{args.dump_dir}/phase1_ws_o.pt", map_location="cpu").float()
    print(f"golden_max_q2_kv1={ws_max[2, 1].tolist()}")
    print(f"golden_o_q2_kv1_first={ws_o[2, 1, :, :, :8].tolist()}")


if __name__ == "__main__":
    main()
