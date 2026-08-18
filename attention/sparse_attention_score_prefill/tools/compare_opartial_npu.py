#!/usr/bin/env python3
"""Compare NPU desc=981 dumps (O_partial) with golden pv_Opartial_*.pt files."""
import re
import sys
import torch

TORCH_DEVICE_BACKEND_AUTOLOAD = "0"
BASE_ADDR = 0x42400000
ELEM_SIZE = 2  # bf16 = 2 bytes
GOLDEN_DIR = "/opt/rrh/code3/ops-transformer/attention/sparse_attention_score_prefill/golden_dumps/run_218057_1786434111141243750"
LOG_FILE = "/opt/rrh/code3/ops-transformer/attention/sparse_attention_score_prefill/o.txt"

# wsOOff = qToken * 3072 + kvHead * 1536 + slot * 512
def offset_to_tag(offset):
    per_q = 3072
    per_h = 1536
    per_s = 512
    q = offset // per_q
    rem = offset % per_q
    h = rem // per_h
    rem2 = rem % per_h
    s = rem2 // per_s
    if q * per_q + h * per_h + s * per_s != offset:
        return None
    return (q, h, s)

def parse_npu_dumps(filepath):
    """Parse desc=981 dumps from o.txt. Returns {offset: [values]}."""
    dumps = {}
    current_offset = None
    current_values = []
    in_data = False

    with open(filepath, 'r') as f:
        for line in f:
            # Match dump header
            m = re.search(r'\[AIC Block \d+\] DumpTensor: desc=981, addr=(0x[0-9a-fA-F]+)', line)
            if m:
                if current_offset is not None and current_values:
                    dumps[current_offset] = current_values
                addr = int(m.group(1), 16)
                current_offset = (addr - BASE_ADDR) // ELEM_SIZE
                current_values = []
                in_data = True
                continue

            # Match wsOOff line (alternative source of offset)
            m2 = re.search(r'\[AIC Block \d+/20\] wsOOff = (\d+)', line)
            if m2 and not in_data:
                # This is from scatter loop, not dump - skip
                continue

            if in_data:
                # Parse data values
                stripped = line.strip()
                if stripped.startswith('[') or stripped.startswith('-'):
                    # Remove brackets and parse floats
                    clean = stripped.strip('[]')
                    parts = re.split(r'[,\s]+', clean)
                    for p in parts:
                        p = p.strip()
                        if p and p != '-':
                            try:
                                current_values.append(float(p))
                            except ValueError:
                                pass
                elif stripped == '' or stripped.startswith('[') == False:
                    # End of data block
                    if current_values and len(current_values) > 10:
                        if current_offset is not None:
                            dumps[current_offset] = current_values
                    in_data = False
                    current_offset = None
                    current_values = []

    if current_offset is not None and current_values:
        dumps[current_offset] = current_values

    return dumps

def main():
    print(f"Parsing NPU dumps from {LOG_FILE}...")
    npu_dumps = parse_npu_dumps(LOG_FILE)
    print(f"Found {len(npu_dumps)} desc=981 dumps")

    results = []
    for offset, npu_vals in sorted(npu_dumps.items()):
        tag = offset_to_tag(offset)
        if tag is None:
            print(f"  offset={offset}: cannot map to (q,h,t), skipping")
            continue

        q, h, t = tag
        golden_name = f"pv_Opartial_q{q}_h{h}_t{t}.pt"
        golden_path = f"{GOLDEN_DIR}/{golden_name}"

        try:
            golden = torch.load(golden_path, map_location='cpu')
            golden = golden.to(torch.float32).flatten()
        except Exception as e:
            print(f"  offset={offset} q{q}_h{h}_t{t}: golden load failed: {e}")
            continue

        npu_tensor = torch.tensor(npu_vals, dtype=torch.float32)
        n = min(len(npu_tensor), len(golden))
        if n == 0:
            print(f"  offset={offset} q{q}_h{h}_t{t}: no data to compare")
            continue

        npu_sub = npu_tensor[:n]
        golden_sub = golden[:n]

        diff = (npu_sub - golden_sub).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        match_count = (diff < 0.5).sum().item()

        status = "OK" if max_diff < 1.0 else "FAIL"
        results.append((q, h, t, offset, n, max_diff, mean_diff, match_count, status))

        print(f"  [{status}] q{q}_h{h}_t{t} offset={offset} n={n} "
              f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
              f"match(<0.5)={match_count}/{n}")

        # Show first 8 values comparison
        print(f"    NPU:    {npu_sub[:8].tolist()}")
        print(f"    Golden: {golden_sub[:8].tolist()}")

    print("\n=== Summary ===")
    ok = sum(1 for r in results if r[8] == "OK")
    fail = sum(1 for r in results if r[8] == "FAIL")
    print(f"Total: {len(results)} (OK={ok}, FAIL={fail})")

    if fail > 0:
        print("\nFailed cases:")
        for r in results:
            if r[8] == "FAIL":
                print(f"  q{r[0]}_h{r[1]}_t{r[2]} offset={r[3]} max_diff={r[5]:.6f}")

if __name__ == "__main__":
    main()
