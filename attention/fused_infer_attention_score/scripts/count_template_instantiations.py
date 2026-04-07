#!/usr/bin/env python3
"""
Analyze ASCENDC_TPL_ARGS_SEL blocks in the FIAS template tiling key header
to count the total number of template instantiations and identify which
template parameters contribute most to combinatorial explosion.

Usage:
    python3 count_template_instantiations.py [path_to_template_tiling_key.h]
"""

import re
import sys
from pathlib import Path
from functools import reduce
from operator import mul
from collections import defaultdict


def parse_tpl_args_sel_blocks(content: str) -> list[dict]:
    """Parse ASCENDC_TPL_ARGS_SEL blocks into structured data."""
    blocks = []
    # Match each ASCENDC_TPL_ARGS_SEL(...) block
    pattern = r'ASCENDC_TPL_ARGS_SEL\((.*?)\)(?:\s*,|\s*\))'
    # Use a simpler approach: split by ASCENDC_TPL_ARGS_SEL and parse each
    parts = content.split('ASCENDC_TPL_ARGS_SEL(')

    for i, part in enumerate(parts[1:], 1):  # skip the preamble
        # Find the matching closing paren - track nesting
        depth = 1
        end = 0
        for j, ch in enumerate(part):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == 0:
            continue
        block_text = part[:end]
        block = parse_single_block(block_text, i)
        if block:
            blocks.append(block)
    return blocks


def parse_single_block(text: str, block_idx: int) -> dict | None:
    """Parse a single ASCENDC_TPL_ARGS_SEL block into parameter -> values mapping."""
    params = {}

    # Parse UINT_SEL entries
    for m in re.finditer(r'ASCENDC_TPL_UINT_SEL\(\s*(\w+)\s*,\s*ASCENDC_TPL_UI_LIST\s*,\s*([^)]+)\)', text):
        name = m.group(1)
        values = [v.strip() for v in m.group(2).split(',') if v.strip()]
        params[name] = values

    # Parse BOOL_SEL entries
    for m in re.finditer(r'ASCENDC_TPL_BOOL_SEL\(\s*(\w+)\s*,\s*([^)]+)\)', text):
        name = m.group(1)
        values = [v.strip() for v in m.group(2).split(',') if v.strip()]
        params[name] = values

    if not params:
        return None

    return {'index': block_idx, 'params': params}


def count_instantiations(block: dict) -> int:
    """Count instantiations for a block (Cartesian product of all param values)."""
    counts = [len(v) for v in block['params'].values()]
    return reduce(mul, counts, 1)


def shorten_dtype(dtype_str: str) -> str:
    """Convert a #if ORIG_DTYPE guard to a compact Q=FP16 K=INT8 O=FP16 form."""
    if dtype_str == "unknown":
        return "unknown"
    abbrevs = {
        'DT_FLOAT16': 'FP16', 'DT_BF16': 'BF16', 'DT_INT8': 'INT8',
        'DT_INT4': 'INT4', 'DT_HIFLOAT8': 'HiFP8',
        'DT_FLOAT8_E4M3FN': 'FP8E4M3', 'DT_FLOAT4_E2M1': 'FP4E2M1',
    }
    parts = []
    for m in re.finditer(r'ORIG_DTYPE_(\w+)\s*==\s*(\w+)', dtype_str):
        role = {'QUERY': 'Q', 'KEY': 'K', 'ATTENTION_OUT': 'O'}.get(m.group(1), m.group(1))
        dtype = abbrevs.get(m.group(2), m.group(2))
        parts.append(f"{role}={dtype}")
    return ' '.join(parts) if parts else dtype_str


def find_dtype_context(content: str, blocks: list[dict]) -> list[dict]:
    """Associate blocks with their #if dtype context."""
    lines = content.split('\n')
    current_dtype = "unknown"
    block_idx = 0
    dtype_map = {}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#if') and 'ORIG_DTYPE' in stripped:
            current_dtype = stripped
        elif stripped.startswith('#endif'):
            current_dtype = "unknown"
        if 'ASCENDC_TPL_ARGS_SEL(' in stripped:
            block_idx += 1
            dtype_map[block_idx] = current_dtype

    for block in blocks:
        block['dtype_context'] = dtype_map.get(block['index'], 'unknown')
    return blocks


def analyze(filepath: str):
    content = Path(filepath).read_text()
    blocks = parse_tpl_args_sel_blocks(content)
    blocks = find_dtype_context(content, blocks)

    total = 0
    by_dtype = defaultdict(int)
    by_param = defaultdict(lambda: defaultdict(int))  # param -> value -> count of blocks using it

    print(f"{'='*80}")
    print(f"Template Instantiation Analysis: {Path(filepath).name}")
    print(f"{'='*80}\n")

    print(f"{'Block':>5} | {'Count':>6} | {'Dtype Context':<30} | Parameters")
    print(f"{'-'*5}-+-{'-'*6}-+-{'-'*30}-+{'-'*40}")

    for block in blocks:
        count = count_instantiations(block)
        total += count
        dtype_short = shorten_dtype(block['dtype_context'])

        # Compact param summary
        param_parts = []
        for name, values in block['params'].items():
            param_parts.append(f"{name}({len(values)})")
            for v in values:
                by_param[name][v] += 1

        by_dtype[block['dtype_context']] += count
        print(f"{block['index']:>5} | {count:>6} | {dtype_short:<30} | {' '.join(param_parts)}")

    print(f"\n{'='*80}")
    print(f"TOTAL INSTANTIATIONS: {total}")
    print(f"{'='*80}\n")

    # Per-dtype breakdown
    print("Per-dtype breakdown:")
    for dtype, count in sorted(by_dtype.items(), key=lambda x: -x[1]):
        print(f"  {count:>6} instantiations: {shorten_dtype(dtype)}")

    # Per-parameter contribution analysis
    print(f"\nParameter contribution (how many values each parameter takes across all blocks):")
    param_summary = []
    for name in ['InOutLayoutType', 'Config', 'PseMode', 'QuantMode',
                 'HasAttenMask', 'HasRope', 'IsPa', 'IsFd', 'EmptyTensor',
                 'PFAMask', 'PFAMatMulType', 'EnableKVPrefix', 'EnableS1OutSplit']:
        if name in by_param:
            values = by_param[name]
            max_vals = max(len(block['params'].get(name, [])) for block in blocks)
            avg_vals = sum(len(block['params'].get(name, [])) for block in blocks) / len(blocks)
            param_summary.append((name, max_vals, avg_vals, len(values)))

    print(f"  {'Parameter':<20} | {'Max vals':>8} | {'Avg vals':>8} | {'Distinct vals':>13}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*8}-+-{'-'*13}")
    for name, max_v, avg_v, distinct in param_summary:
        print(f"  {name:<20} | {max_v:>8} | {avg_v:>8.1f} | {distinct:>13}")

    # Boolean multiplier analysis
    print(f"\nBoolean parameter multiplier analysis:")
    bool_params = ['HasAttenMask', 'HasRope', 'IsPa', 'IsFd', 'EmptyTensor',
                   'EnableKVPrefix', 'EnableS1OutSplit']
    total_bool_multiplier = 1
    for name in bool_params:
        vals_per_block = [len(block['params'].get(name, ['false'])) for block in blocks]
        max_v = max(vals_per_block)
        blocks_with_2 = sum(1 for v in vals_per_block if v == 2)
        print(f"  {name:<20}: max {max_v} values, {blocks_with_2}/{len(blocks)} blocks have both true/false")
        if max_v == 2:
            total_bool_multiplier *= 2

    print(f"\n  Theoretical boolean multiplier (if all bools had both values): {total_bool_multiplier}x")
    print(f"  If all booleans were type-erased to runtime:")
    non_bool_total = sum(
        reduce(mul, [len(v) for k, v in block['params'].items() if k not in bool_params], 1)
        for block in blocks
    )
    print(f"    Instantiations would drop from {total} to ~{non_bool_total}")
    print(f"    Reduction: {total / max(non_bool_total, 1):.1f}x")


if __name__ == '__main__':
    default_path = str(Path(__file__).parent.parent /
                       'op_kernel' / 'fused_infer_attention_score_template_tiling_key.h')
    filepath = sys.argv[1] if len(sys.argv) > 1 else default_path
    analyze(filepath)
