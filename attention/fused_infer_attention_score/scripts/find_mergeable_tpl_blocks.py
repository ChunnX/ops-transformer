#!/usr/bin/env python3
"""
Find pairs of ASCENDC_TPL_ARGS_SEL blocks that differ in exactly one parameter
and can be safely merged. Merging is safe when two blocks' union of Cartesian
products equals the Cartesian product of the merged parameter lists.

This holds iff the blocks differ in exactly ONE parameter dimension.

Usage:
    python3 find_mergeable_tpl_blocks.py [path_to_template_tiling_key.h]
"""

import re
import sys
from pathlib import Path
from functools import reduce
from operator import mul
from itertools import combinations


def parse_tpl_args_sel_blocks(content: str) -> list[dict]:
    """Parse ASCENDC_TPL_ARGS_SEL blocks into structured data."""
    blocks = []
    # Track dtype context
    lines = content.split('\n')
    current_dtype = "unknown"
    block_texts = []

    # First pass: split into blocks with dtype context
    in_block = False
    block_start = 0
    depth = 0

    parts = content.split('ASCENDC_TPL_ARGS_SEL(')

    # Track dtype context per character offset
    dtype_at_offset = []
    offset = 0
    current_dtype = "unknown"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#if') and 'ORIG_DTYPE' in stripped:
            current_dtype = stripped
        for _ in line:
            dtype_at_offset.append(current_dtype)
            offset += 1
        dtype_at_offset.append(current_dtype)  # newline
        offset += 1

    for i, part in enumerate(parts[1:], 1):
        depth_count = 1
        end = 0
        for j, ch in enumerate(part):
            if ch == '(':
                depth_count += 1
            elif ch == ')':
                depth_count -= 1
                if depth_count == 0:
                    end = j
                    break
        if end == 0:
            continue
        block_text = part[:end]

        # Find offset of this block in original content
        search_text = 'ASCENDC_TPL_ARGS_SEL(' + block_text
        block_offset = content.find(search_text, block_start)
        block_start = block_offset + 1 if block_offset >= 0 else block_start

        dtype = dtype_at_offset[min(block_offset, len(dtype_at_offset)-1)] if block_offset >= 0 else "unknown"

        block = parse_single_block(block_text, i, dtype)
        if block:
            blocks.append(block)
    return blocks


def parse_single_block(text: str, block_idx: int, dtype: str) -> dict | None:
    """Parse a single ASCENDC_TPL_ARGS_SEL block."""
    params = {}

    # Ordered parameter names (must match declaration order)
    param_order = []

    for m in re.finditer(r'ASCENDC_TPL_UINT_SEL\(\s*(\w+)\s*,\s*ASCENDC_TPL_UI_LIST\s*,\s*([^)]+)\)', text):
        name = m.group(1)
        values = tuple(sorted(v.strip() for v in m.group(2).split(',') if v.strip()))
        params[name] = values
        param_order.append(name)

    for m in re.finditer(r'ASCENDC_TPL_BOOL_SEL\(\s*(\w+)\s*,\s*([^)]+)\)', text):
        name = m.group(1)
        values = tuple(sorted(v.strip() for v in m.group(2).split(',') if v.strip()))
        params[name] = values
        param_order.append(name)

    # Parse tiling struct
    m = re.search(r'ASCENDC_TPL_TILING_STRUCT_SEL\(\s*(\w+)\s*\)', text)
    tiling_struct = m.group(1) if m else "unknown"

    if not params:
        return None

    return {
        'index': block_idx,
        'params': params,
        'param_order': param_order,
        'dtype': dtype,
        'tiling_struct': tiling_struct,
        'raw': text,
    }


def count_instantiations(block: dict) -> int:
    counts = [len(v) for v in block['params'].values()]
    return reduce(mul, counts, 1)


def find_diff_params(a: dict, b: dict) -> list[str]:
    """Find parameters that differ between two blocks."""
    diffs = []
    all_params = set(a['params'].keys()) | set(b['params'].keys())
    for param in all_params:
        va = a['params'].get(param, ())
        vb = b['params'].get(param, ())
        if va != vb:
            diffs.append(param)
    return diffs


def merge_blocks(a: dict, b: dict, diff_param: str) -> dict:
    """Merge two blocks that differ in exactly one parameter."""
    merged_params = {}
    for param in a['params']:
        if param == diff_param:
            merged_params[param] = tuple(sorted(set(a['params'][param]) | set(b['params'][param])))
        else:
            merged_params[param] = a['params'][param]

    return {
        'index': f"{a['index']}+{b['index']}",
        'params': merged_params,
        'param_order': a['param_order'],
        'dtype': a['dtype'],
        'tiling_struct': a['tiling_struct'],
    }


def analyze(filepath: str):
    content = Path(filepath).read_text()
    blocks = parse_tpl_args_sel_blocks(content)

    print(f"Parsed {len(blocks)} ASCENDC_TPL_ARGS_SEL blocks\n")

    mergeable_pairs = []

    for a, b in combinations(blocks, 2):
        # Must be same dtype context and tiling struct
        if a['dtype'] != b['dtype']:
            continue
        if a['tiling_struct'] != b['tiling_struct']:
            continue
        # Must have same parameter set
        if set(a['params'].keys()) != set(b['params'].keys()):
            continue

        diffs = find_diff_params(a, b)
        if len(diffs) == 1:
            diff_param = diffs[0]
            merged = merge_blocks(a, b, diff_param)
            old_count = count_instantiations(a) + count_instantiations(b)
            new_count = count_instantiations(merged)
            # new_count should == old_count for single-param diff
            mergeable_pairs.append({
                'block_a': a['index'],
                'block_b': b['index'],
                'diff_param': diff_param,
                'a_vals': a['params'][diff_param],
                'b_vals': b['params'][diff_param],
                'merged_vals': merged['params'][diff_param],
                'old_count': old_count,
                'new_count': new_count,
                'savings': old_count - new_count,
                'dtype': a['dtype'][:80],
            })

    if not mergeable_pairs:
        print("No mergeable pairs found.")
        return

    # Sort by savings descending
    mergeable_pairs.sort(key=lambda x: -x['savings'])

    print(f"Found {len(mergeable_pairs)} mergeable pairs (differ in exactly 1 parameter):\n")
    print(f"{'Blocks':>10} | {'Diff Param':<20} | {'Old':>6} | {'New':>6} | {'Saved':>6} | Values")
    print(f"{'-'*10}-+-{'-'*20}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+{'-'*50}")

    total_savings = 0
    seen_blocks = set()
    greedy_savings = 0

    for pair in mergeable_pairs:
        blocks_key = f"{pair['block_a']}+{pair['block_b']}"
        print(f"{blocks_key:>10} | {pair['diff_param']:<20} | {pair['old_count']:>6} | {pair['new_count']:>6} | {pair['savings']:>6} | "
              f"{','.join(pair['a_vals'][:3])}... + {','.join(pair['b_vals'][:3])}...")

        # Greedy non-overlapping merge tracking
        if pair['block_a'] not in seen_blocks and pair['block_b'] not in seen_blocks:
            greedy_savings += pair['savings']
            seen_blocks.add(pair['block_a'])
            seen_blocks.add(pair['block_b'])

    total_before = sum(count_instantiations(b) for b in blocks)
    print(f"\nTotal instantiations before: {total_before}")
    print(f"Greedy non-overlapping merge savings: {greedy_savings} instantiations")
    print(f"Total after greedy merge: {total_before - greedy_savings}")
    print(f"Blocks reduced from {len(blocks)} to {len(blocks) - len(seen_blocks) // 2}")


if __name__ == '__main__':
    default_path = str(Path(__file__).parent.parent /
                       'op_kernel' / 'fused_infer_attention_score_template_tiling_key.h')
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_path
    analyze(filepath)
