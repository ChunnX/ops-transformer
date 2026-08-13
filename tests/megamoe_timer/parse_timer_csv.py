import argparse
import csv
from pathlib import Path

from timer_path import resolve_timer_dir


FIXED_NAMES = [
    "KERNEL_TIMING",
    "AIV_INIT_PARAM",
    "AIV_INITROUTING_SORT",
    "AIV_INITROUTING_EXPERT_TOKEN_COUNT",
    "AIV_INITROUTING_GATHEROUT_QUANT",
    "AIV_ALLGATHER",
    "AIV_WAIT_COMB",
    "AIV_UNPERMUTE_WAIT",
    "AIV_UNPERMUTE",
    "AIC_WAIT_ALLGATHER",
]
DYNAMIC_NAMES = [
    "AIV_DISPATCH",
    "AIV_SWIGLU_WAIT",
    "AIV_SWIGLU",
    "AIV_COMBINE_WAIT",
    "AIV_COMBINE",
    "AIC_GMM1_WAIT",
    "AIC_GMM1",
    "AIC_GMM2_WAIT",
    "AIC_GMM2",
]
MAX_DYNAMIC_ITER = 1024
CORE_COUNT = 96
CORES_PER_GROUP = 3
ITEMS_PER_CORE = len(FIXED_NAMES) + len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER
COUNTERS_PER_CORE = ((ITEMS_PER_CORE * 2 + 15) // 16) * 16
COUNTER_COUNT = COUNTERS_PER_CORE * CORE_COUNT
DYNAMIC_ITER_STRIDE = ((len(DYNAMIC_NAMES) + 15) // 16) * 16
TOTAL_COUNT = COUNTER_COUNT + DYNAMIC_ITER_STRIDE * CORE_COUNT
CYCLES_PER_US = 1000.0


def load_values(path: Path):
    values = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            values.extend(int(item.strip()) for item in row if item.strip())
    if len(values) != TOTAL_COUNT:
        raise ValueError(f"{path}: got {len(values)} int64 values, expected {TOTAL_COUNT}")
    return values


def core_info(core_id):
    group_id, slot = divmod(core_id, CORES_PER_GROUP)
    core_type = "AIC" if slot == 0 else f"AIV{slot - 1}"
    return group_id, core_type, (0 if slot == 0 else slot - 1)


def event_names(iter_values):
    result = list(FIXED_NAMES)
    for type_id, name in enumerate(DYNAMIC_NAMES):
        actual = max(
            iter_values[core * DYNAMIC_ITER_STRIDE + type_id] for core in range(CORE_COUNT)
        )
        result.extend(f"{name}_{index}" for index in range(min(actual, MAX_DYNAMIC_ITER)))
    return result


def logical_index(name):
    if name in FIXED_NAMES:
        return FIXED_NAMES.index(name)
    prefix, iteration = name.rsplit("_", 1)
    return len(FIXED_NAMES) + DYNAMIC_NAMES.index(prefix) * MAX_DYNAMIC_ITER + int(iteration)


def parse_file(input_path: Path, output_path: Path):
    values = load_values(input_path)
    counters = values[:COUNTER_COUNT]
    iterations = values[COUNTER_COUNT:]
    names = event_names(iterations)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["core_id", "group_id", "core_type", "sub_id"]
        for name in names:
            header.extend((f"{name}_start_cycle", f"{name}_end_cycle", f"{name}_duration(us)"))
        writer.writerow(header)
        for core_id in range(CORE_COUNT):
            group_id, core_type, sub_id = core_info(core_id)
            base = core_id * COUNTERS_PER_CORE
            row = [core_id, group_id, core_type, sub_id]
            for name in names:
                offset = base + logical_index(name) * 2
                start, end = counters[offset:offset + 2]
                duration = max(end - start, 0) / CYCLES_PER_US
                row.extend((start, end, f"{duration:.3f}"))
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Parse MegaMoe AscendTimerV2 CSV files")
    parser.add_argument(
        "--dir",
        default=None,
        help=(
            "Timer case directory or timer root. By default, use "
            "tests/megamoe_w4a8/timer and select its latest case."
        ),
    )
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-end", type=int, default=7)
    args = parser.parse_args()
    timer_dir = resolve_timer_dir(args.dir, "rank_*_npu_time.csv")
    print(f"[INFO] timer directory: {timer_dir}")
    for rank in range(args.rank_start, args.rank_end + 1):
        src = timer_dir / f"rank_{rank}_npu_time.csv"
        if not src.exists():
            print(f"[SKIP] {src}")
            continue
        dst = timer_dir / f"rank_{rank}_npu_time_parsed.csv"
        parse_file(src, dst)
        print(f"[OK] {dst}")


if __name__ == "__main__":
    main()
