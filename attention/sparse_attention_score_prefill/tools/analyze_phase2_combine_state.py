"""Compare the A2 Phase2 scale/destination dumps with the same run's golden data."""

import argparse
from pathlib import Path

import torch


def _load_dump(run_dir, desc, count):
    paths = sorted(run_dir.glob(f"dump_desc{desc}_*.txt"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one desc={desc} dump, found {len(paths)}")
    values = [float(item) for item in paths[0].read_text(encoding="utf-8").split(",") if item.strip()]
    if len(values) != count:
        raise RuntimeError(f"desc={desc} has {len(values)} values, expected {count}: {paths[0]}")
    return torch.tensor(values, dtype=torch.float32), paths[0]


def _load_optional_dump(run_dir, desc, count):
    paths = sorted(run_dir.glob(f"dump_desc{desc}_*.txt"))
    if not paths:
        return None, None
    if len(paths) != 1:
        raise RuntimeError(f"expected one desc={desc} dump, found {len(paths)}")
    values = [float(item) for item in paths[0].read_text(encoding="utf-8").split(",") if item.strip()]
    if len(values) != count:
        raise RuntimeError(f"desc={desc} has {len(values)} values, expected {count}: {paths[0]}")
    return torch.tensor(values, dtype=torch.float32), paths[0]


def _summary(actual, expected):
    delta = (actual - expected).abs()
    flat_index = int(delta.reshape(-1).argmax().item())
    actual_flat = actual.reshape(-1)
    expected_flat = expected.reshape(-1)
    return {
        "values": int(delta.numel()),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "first_max_index": flat_index,
        "actual_at_max": float(actual_flat[flat_index]),
        "expected_at_max": float(expected_flat[flat_index]),
    }


def _scale(max_values, sum_values):
    valid = torch.isfinite(max_values) & (sum_values > 0)
    masked_max = torch.where(valid, max_values, torch.full_like(max_values, -float("inf")))
    global_max = masked_max.max(dim=0).values
    numerator = torch.where(valid, sum_values * torch.exp(max_values - global_max), torch.zeros_like(sum_values))
    denominator = numerator.sum(dim=0)
    return torch.where(denominator.unsqueeze(0) > 0, numerator / denominator.unsqueeze(0), torch.zeros_like(numerator))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    group_size = args.q_heads // args.kv_heads
    lse_count = args.top_k * group_size
    lse_vec_count = lse_count * 8
    aligned_lse_count = (lse_count + 7) // 8 * 8

    max_dump, max_file = _load_dump(run_dir, 970, lse_vec_count)
    pre_sum_dump, pre_sum_file = _load_dump(run_dir, 969, lse_vec_count)
    scale_dump, scale_file = _load_dump(run_dir, 971, lse_vec_count)
    inv_sum_dump, inv_sum_file = _load_dump(run_dir, 972, aligned_lse_count)
    max_global_dump, max_global_file = _load_dump(run_dir, 974, group_size * 8)
    total_dump, total_file = _load_dump(run_dir, 975, group_size * 8)
    dst_dump, dst_file = _load_dump(run_dir, 973, group_size * args.head_dim)
    denominator_dump, denominator_file = _load_optional_dump(run_dir, 977, group_size * 8)
    numerator_dump, numerator_file = _load_optional_dump(
        run_dir, 978, args.top_k * group_size * 8)
    direct_max, direct_max_file = _load_optional_dump(run_dir, 964, aligned_lse_count)
    direct_sum, direct_sum_file = _load_optional_dump(run_dir, 965, aligned_lse_count)
    copied_max, copied_max_file = _load_optional_dump(run_dir, 966, aligned_lse_count)
    copied_sum, copied_sum_file = _load_optional_dump(run_dir, 967, aligned_lse_count)

    state_max = max_dump.reshape(args.top_k, group_size, 8)[..., 0]
    state_pre_sum = pre_sum_dump.reshape(args.top_k, group_size, 8)[..., 0]
    state_scale = scale_dump.reshape(args.top_k, group_size, 8)[..., 0]
    state_sum = inv_sum_dump[:lse_count].reshape(args.top_k, group_size).reciprocal()
    expected_state_scale = _scale(state_max, state_sum)
    expected_global_max = state_max.max(dim=0).values
    expected_total = (state_sum * torch.exp(state_max - expected_global_max)).sum(dim=0)

    golden_max = torch.load(run_dir / "golden_phase2_row_max.pt", map_location="cpu").float()[0, 0]
    golden_sum = torch.load(run_dir / "golden_phase2_row_sum.pt", map_location="cpu").float()[0, 0]
    golden_inv_sum = torch.load(
        run_dir / "golden_phase2_inv_row_sum.pt", map_location="cpu").float()[0, 0]
    golden_global_max = torch.load(
        run_dir / "golden_phase2_global_max.pt", map_location="cpu").float()[0, 0]
    golden_numerator = torch.load(
        run_dir / "golden_phase2_numerator.pt", map_location="cpu").float()[0, 0]
    golden_denominator = torch.load(
        run_dir / "golden_phase2_denominator.pt", map_location="cpu").float()[0, 0]
    golden_scale = torch.load(
        run_dir / "golden_phase2_scale.pt", map_location="cpu").float()[0, 0]
    golden_combined = torch.load(
        run_dir / "golden_phase2_combined_fp32.pt", map_location="cpu").float()[0, 0]
    golden_final = torch.load(run_dir / "golden_final.pt", map_location="cpu").float()[0, :group_size]
    actual = torch.load(run_dir / "npu_debug_output.pt", map_location="cpu").float()[0, :group_size]

    print(f"[phase2-state] run_dir={run_dir}")
    print(f"[phase2-state] kernel_log={run_dir / 'kernel.log'}")
    print(
        f"[phase2-state] dumps pre_sum={pre_sum_file} max={max_file} scale={scale_file} "
        f"inv_sum={inv_sum_file} max_global={max_global_file} total={total_file} dst={dst_file}")
    if direct_sum is not None:
        print(
            f"[phase2-state] lse_probe direct_max={direct_max_file} direct_sum={direct_sum_file} "
            f"copied_max={copied_max_file} copied_sum={copied_sum_file}")
        print(f"[phase2-state] gm_sum_vs_golden={_summary(direct_sum[:lse_count], golden_sum.flatten())}")
        print(f"[phase2-state] copied_sum_vs_gm={_summary(copied_sum[:lse_count], direct_sum[:lse_count])}")
        print(f"[phase2-state] copied_max_vs_gm={_summary(copied_max[:lse_count], direct_max[:lse_count])}")
    print(f"[phase2-state] broadcast_sum_vs_compact_sum={_summary(state_pre_sum, state_sum)}")
    print(f"[phase2-state] max_global_vs_formula={_summary(max_global_dump.reshape(group_size, 8)[:, 0], expected_global_max)}")
    print(f"[phase2-state] total_vs_formula={_summary(total_dump.reshape(group_size, 8)[:, 0], expected_total)}")
    print(f"[phase2-state] scale_vs_state_formula={_summary(state_scale, expected_state_scale)}")
    print(f"[phase2-state] max_vs_golden={_summary(state_max, golden_max)}")
    print(f"[phase2-state] sum_vs_golden={_summary(state_sum, golden_sum)}")
    print(f"[phase2-state] inv_sum_vs_golden={_summary(inv_sum_dump[:lse_count].reshape(args.top_k, group_size), golden_inv_sum)}")
    print(f"[phase2-state] global_max_vs_golden={_summary(max_global_dump.reshape(group_size, 8)[:, 0], golden_global_max)}")
    print(f"[phase2-state] denominator_vs_golden={_summary(total_dump.reshape(group_size, 8)[:, 0], golden_denominator)}")
    if denominator_dump is not None:
        print(
            f"[phase2-state] numerator_probe denominator={denominator_file} "
            f"numerator={numerator_file}")
        print(f"[phase2-state] pre_recompute_denominator_vs_golden={_summary(denominator_dump.reshape(group_size, 8)[:, 0], golden_denominator)}")
        print(f"[phase2-state] numerator_vs_golden={_summary(numerator_dump.reshape(args.top_k, group_size, 8)[..., 0], golden_numerator)}")
    print(f"[phase2-state] scale_vs_golden={_summary(state_scale, golden_scale)}")
    print(f"[phase2-state] dst_vs_golden_combined_fp32={_summary(dst_dump.reshape(group_size, args.head_dim), golden_combined)}")
    print(f"[phase2-state] dst_vs_output={_summary(dst_dump.reshape(group_size, args.head_dim), actual)}")
    print(f"[phase2-state] dst_vs_golden_final={_summary(dst_dump.reshape(group_size, args.head_dim), golden_final)}")


if __name__ == "__main__":
    main()
