#!/usr/bin/env python3
"""L0C Dump vs Golden O_partial 比对脚本

用法:
  # 在远程服务器上运行
  cd /opt/rrh/code3/ops-transformer/attention/sparse_attention_score_prefill
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python /tmp/compare_l0c_golden.py \
      --golden golden_dumps/run_XXX/pv_Opartial_q0_h0_t0.pt \
      --l0c-dump "4.649417, -0.075913, 1.414795, ..." \
      --M 1 --N 128

  # 或从文件读取 L0C dump 值
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 python /tmp/compare_l0c_golden.py \
      --golden golden_dumps/run_XXX/pv_Opartial_q0_h0_t0.pt \
      --l0c-file l0c_dump.txt \
      --M 1 --N 128

说明:
  L0C 使用 ZN fractal 布局 (MakeLayoutL0C):
    - fractal 大小: 16x16 (C0_NUM_PER_FRACTAL=16, ELE_NUM_PER_FRACTAL=256)
    - fractal 内: 行连续 (row stride=1, col stride=16)
    - 物理内存: [row0_col0-15][row1_col0-15]...[row15_col0-15][row0_col16-31]...
    - DumpTensor 返回物理内存顺序

  对于 M 行 N 列的 L0C:
    - 行数向上对齐到 16: M_aligned = RoundUp(M, 16)
    - 列数向上对齐到 16: N_aligned = RoundUp(N, 16)
    - 总元素数: M_aligned * N_aligned
    - 有效元素: M * N (前 M 行, 前 N 列)

  本脚本从物理 dump 中提取有效行 (row 0..M-1) 的所有列 (col 0..N-1),
  与 golden 做 1:1 比对。
"""

import argparse
import sys
import numpy as np

try:
    import torch
except ImportError:
    print("Error: torch not found. Install or use TORCH_DEVICE_BACKEND_AUTOLOAD=0")
    sys.exit(1)


def parse_l0c_from_string(s):
    """从逗号分隔字符串解析 L0C dump 值"""
    values = []
    for token in s.replace('\n', ',').split(','):
        token = token.strip()
        if token:
            values.append(float(token))
    return np.array(values, dtype=np.float32)


def parse_l0c_from_file(path):
    """从文件读取 L0C dump 值 (每行一个浮点数或逗号分隔)"""
    with open(path, 'r') as f:
        content = f.read()
    return parse_l0c_from_string(content)


def de_fractal_zn(dump_flat, M, N):
    """将 ZN fractal 物理内存布局转换为逻辑 [M, N] row-major

    ZN fractal 布局:
      - C0 = 16 (fractal 内行/列大小)
      - 物理 shape: (M_aligned//16, N_aligned//16, 16, 16)
        其中最后一维 (16) = 行 (连续), 倒数第二维 (16) = 列
      - 物理内存顺序: fractal_row_major → within fractal: row0_col0-15, row1_col0-15, ...

    提取逻辑 [M, N]:
      logical[r][c] = dump_flat[
          (r // 16) * (N_aligned // 16) * 256  # 跳过前面的 fractal 行
        + (c // 16) * 256                       # 跳过前面的 fractal 列
        + (r % 16) * 16                         # fractal 内的行偏移
        + (c % 16)                              # fractal 内的列偏移
      ]
    """
    C0 = 16
    M_aligned = ((M + C0 - 1) // C0) * C0
    N_aligned = ((N + C0 - 1) // C0) * C0
    total_expected = M_aligned * N_aligned
    total_dump = len(dump_flat)

    if total_dump < total_expected:
        print(f"WARNING: dump has {total_dump} elements, expected {total_expected} "
              f"(M_aligned={M_aligned} x N_aligned={N_aligned})")
        print(f"  Only first {total_dump} elements available. "
              f"Valid coverage: {total_dump}/{total_expected} = {total_dump/total_expected*100:.1f}%")

    # Reshape to (M_aligned//16, N_aligned//16, 16, 16)
    num_fractal_rows = M_aligned // C0
    num_fractal_cols = N_aligned // C0
    fractal_size = C0 * C0  # 256

    logical = np.zeros((M, N), dtype=np.float32)
    coverage = np.zeros((M, N), dtype=bool)

    for r in range(M):
        for c in range(N):
            fr = r // C0          # fractal row index
            fc = c // C0          # fractal col index
            ir = r % C0           # intra-fractal row
            ic = c % C0           # intra-fractal col
            phys_idx = fr * num_fractal_cols * fractal_size + fc * fractal_size + ir * C0 + ic
            if phys_idx < total_dump:
                logical[r][c] = dump_flat[phys_idx]
                coverage[r][c] = True

    return logical, coverage


def de_fractal_nz(dump_flat, M, N):
    """将 NZ fractal 物理内存布局转换为逻辑 [M, N] row-major

    NZ fractal 布局:
      - C0 = 16
      - fractal 内: 列连续 (col stride=1, row stride=16)
      - 物理内存: [col0_row0-15][col1_row0-15]...
      - logical[r][c] = dump_flat[fr*NC*256 + fc*256 + ic*16 + ir]
    """
    C0 = 16
    M_aligned = ((M + C0 - 1) // C0) * C0
    N_aligned = ((N + C0 - 1) // C0) * C0
    total_expected = M_aligned * N_aligned
    total_dump = len(dump_flat)

    if total_dump < total_expected:
        print(f"WARNING: dump has {total_dump} elements, expected {total_expected}")

    num_fractal_rows = M_aligned // C0
    num_fractal_cols = N_aligned // C0
    fractal_size = C0 * C0

    logical = np.zeros((M, N), dtype=np.float32)
    coverage = np.zeros((M, N), dtype=bool)

    for r in range(M):
        for c in range(N):
            fr = r // C0
            fc = c // C0
            ir = r % C0
            ic = c % C0
            # NZ: col stride=1, row stride=16 within fractal
            phys_idx = fr * num_fractal_cols * fractal_size + fc * fractal_size + ic * C0 + ir
            if phys_idx < total_dump:
                logical[r][c] = dump_flat[phys_idx]
                coverage[r][c] = True

    return logical, coverage


def compare(l0c_logical, golden_flat, coverage, atol=0.02, rtol=0.01):
    """逐元素比对"""
    M, N = l0c_logical.shape
    total = M * N
    covered = int(coverage.sum())
    matches = 0
    max_diff = 0.0
    worst = []

    for r in range(M):
        for c in range(N):
            if not coverage[r][c]:
                continue
            gidx = r * N + c
            if gidx >= len(golden_flat):
                continue
            diff = abs(float(l0c_logical[r][c]) - float(golden_flat[gidx]))
            rel_diff = diff / (abs(float(golden_flat[gidx])) + 1e-8)
            if diff < atol or rel_diff < rtol:
                matches += 1
            if diff > max_diff:
                max_diff = diff
            if diff >= atol and rel_diff >= rtol:
                worst.append((r, c, gidx, float(l0c_logical[r][c]),
                             float(golden_flat[gidx]), diff, rel_diff))

    return {
        'total': total,
        'covered': covered,
        'matches': matches,
        'max_diff': max_diff,
        'worst': sorted(worst, key=lambda x: -x[5])[:20],
    }


def main():
    parser = argparse.ArgumentParser(description='Compare L0C dump with golden O_partial')
    parser.add_argument('--golden', required=True, help='Path to golden .pt file')
    parser.add_argument('--l0c-dump', help='L0C dump values as comma-separated string')
    parser.add_argument('--l0c-file', help='File containing L0C dump values')
    parser.add_argument('--M', type=int, default=1, help='Logical M (rows)')
    parser.add_argument('--N', type=int, default=128, help='Logical N (cols)')
    parser.add_argument('--layout', choices=['zn', 'nz', 'auto'], default='auto',
                        help='L0C fractal layout (default: auto, tries both)')
    parser.add_argument('--atol', type=float, default=0.02, help='Absolute tolerance')
    parser.add_argument('--rtol', type=float, default=0.01, help='Relative tolerance')
    parser.add_argument('--show-all', action='store_true', help='Show all elements, not just mismatches')
    args = parser.parse_args()

    # Load golden
    golden = torch.load(args.golden, map_location='cpu', weights_only=False)
    golden_flat = golden.float().numpy().flatten()
    print(f"Golden: {args.golden}")
    print(f"  shape={golden.shape} dtype={golden.dtype} flatten_len={len(golden_flat)}")
    print()

    # Load L0C dump
    if args.l0c_file:
        dump = parse_l0c_from_file(args.l0c_file)
        print(f"L0C dump from file: {args.l0c_file}")
    elif args.l0c_dump:
        dump = parse_l0c_from_string(args.l0c_dump)
        print(f"L0C dump from command line string")
    else:
        print("Error: must provide --l0c-dump or --l0c-file")
        sys.exit(1)

    print(f"  dump_len={len(dump)}")
    print(f"  M={args.M} N={args.N}")
    C0 = 16
    M_aligned = ((args.M + C0 - 1) // C0) * C0
    N_aligned = ((args.N + C0 - 1) // C0) * C0
    print(f"  M_aligned={M_aligned} N_aligned={N_aligned} expected_total={M_aligned*N_aligned}")
    print()

    layouts_to_try = ['zn', 'nz'] if args.layout == 'auto' else [args.layout]

    best_result = None
    best_layout = None

    for layout in layouts_to_try:
        print(f"=== Trying {layout.upper()} layout ===")
        if layout == 'zn':
            logical, coverage = de_fractal_zn(dump, args.M, args.N)
        else:
            logical, coverage = de_fractal_nz(dump, args.M, args.N)

        result = compare(logical, golden_flat, coverage, args.atol, args.rtol)
        print(f"  Covered: {result['covered']}/{result['total']}")
        print(f"  Matches: {result['matches']}/{result['covered']} (atol={args.atol} rtol={args.rtol})")
        print(f"  Max diff: {result['max_diff']:.6f}")

        if result['worst']:
            print(f"  Worst {min(10, len(result['worst']))} mismatches:")
            print(f"    {'r':>3s} {'c':>3s} {'idx':>4s}  {'L0C':>12s}  {'Golden':>12s}  {'diff':>10s}  {'rel':>10s}")
            for r, c, gidx, l0c_val, g_val, diff, rel in result['worst'][:10]:
                print(f"    {r:3d} {c:3d} {gidx:4d}  {l0c_val:12.6f}  {g_val:12.6f}  {diff:10.6f}  {rel:10.6f}")

        if args.show_all and result['covered'] > 0:
            print(f"\n  All covered elements:")
            print(f"    {'r':>3s} {'c':>3s} {'idx':>4s}  {'L0C':>12s}  {'Golden':>12s}  {'diff':>10s}  {'match':>5s}")
            for r in range(args.M):
                for c in range(args.N):
                    if not coverage[r][c]:
                        continue
                    gidx = r * args.N + c
                    if gidx >= len(golden_flat):
                        continue
                    diff = abs(float(logical[r][c]) - float(golden_flat[gidx]))
                    m = 'OK' if diff < args.atol else 'FAIL'
                    print(f"    {r:3d} {c:3d} {gidx:4d}  {logical[r][c]:12.6f}  {golden_flat[gidx]:12.6f}  {diff:10.6f}  {m:>5s}")

        match_rate = result['matches'] / max(result['covered'], 1)
        if best_result is None or match_rate > best_result['matches'] / max(best_result['covered'], 1):
            best_result = result
            best_layout = layout
        print()

    print(f"=== Best layout: {best_layout.upper()} ===")
    print(f"  Matches: {best_result['matches']}/{best_result['covered']}")
    print(f"  Max diff: {best_result['max_diff']:.6f}")

    if best_result['matches'] == best_result['covered'] and best_result['covered'] == args.M * args.N:
        print("\n  VERDICT: PASS — all elements match within tolerance")
    elif best_result['matches'] / max(best_result['covered'], 1) > 0.9:
        print("\n  VERDICT: MOSTLY PASS — >90% match (check worst mismatches)")
    else:
        print("\n  VERDICT: FAIL — significant mismatches")


if __name__ == '__main__':
    main()
