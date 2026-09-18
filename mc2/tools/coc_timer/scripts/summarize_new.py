"""Summarize final-invocation timer CSVs under a test output directory (ABI v6)."""

import argparse
import csv
import json
import statistics
from decimal import Decimal
from pathlib import Path

METRICS = [
    ("AIV_CNT_SEND_0_duration(us)", "cnt_send"),
    ("AIV_PAYLOAD_0_duration(us)", "payload"),
    ("AIV_CNT_CALC_0_duration(us)", "cnt_calc"),
    ("AIV_RECV_COPY_0_duration(us)", "recv_copy"),
    ("AIV_TAIL_0_duration(us)", "tail"),
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def card_spans(raw_rows, meta):
    """Cross-core spans assume comparable system counters within this rank only."""
    cores = meta["cores"]
    if len(raw_rows) != cores or sorted(int(r["group_id"]) for r in raw_rows) != list(
        range(cores)
    ):
        raise ValueError(f"rank {meta['rank']}: 原始时间戳 CSV 核记录缺失或重复")
    freq = meta.get("cycles_per_us", 50)
    if freq != 50:
        raise ValueError("不支持的 cycles_per_us")
    send_starts, send_ends, recv_ends, tail_ends = [], [], [], []
    for row in raw_rows:
        for name, starts, ends in (
            ("AIV_CNT_SEND_0", send_starts, send_ends),
            ("AIV_RECV_COPY_0", recv_ends, recv_ends),
            ("AIV_TAIL_0", tail_ends, tail_ends),
        ):
            start, end = int(row[f"{name}_start_cycle"]), int(row[f"{name}_end_cycle"])
            if bool(start) != bool(end) or end < start:
                raise ValueError(f"不完整的 {name} 时间戳")
            if start:
                if starts is not recv_ends:
                    starts.append(start)
                ends.append(end)
    first_send = min(send_starts) if send_starts else None
    last_send = max(send_ends) if send_ends else None
    last_recv = max(recv_ends) if recv_ends else None
    last_tail = max(tail_ends) if tail_ends else None

    def span(start, end):
        if start is None or end is None:
            return ""
        if end < start:
            raise ValueError("整卡时间戳逆序，请检查采集及同卡跨核时钟可比性")
        return (end - start) / freq

    return dict(
        rank=meta["rank"],
        tprs=meta["tprs"],
        iteration=meta["iteration"],
        arrival_only=meta.get("arrival_only", ""),
        clock_basis="assumes_same_rank_common_counter",
        sending_cores=len(send_starts),
        first_send_cycle=first_send if first_send is not None else "",
        last_send_cycle=last_send if last_send is not None else "",
        last_recv_copy_cycle=last_recv if last_recv is not None else "",
        last_tail_end_cycle=last_tail if last_tail is not None else "",
        card_send_span_us=span(first_send, last_send),
        card_send_to_all_recv_us=span(first_send, last_recv),
        card_post_recv_tail_us=span(last_recv, last_tail),
    )


def summarize(results, timer):
    metadata_paths = list(timer.glob("metadata_rank_*.json"))
    if not metadata_paths:
        raise ValueError(
            f"{timer}: 没有 metadata_rank_*.json，请先使用 --dump-timer 采集"
        )
    metadata = sorted(
        (json.loads(p.read_text()) for p in metadata_paths), key=lambda m: m["rank"]
    )
    world = metadata[0]["ep_world_size"]
    if [m["rank"] for m in metadata] != list(range(world)):
        raise ValueError(f"{timer}: rank 不完整或重复，预期 0..{world - 1}")
    summary, all_rows, last_times, card_rows = [], [], [], []
    iterations = metadata[0]["iteration"]
    for meta in metadata:
        rank = meta["rank"]
        if (
            meta["ep_world_size"] != world
            or meta["iteration"] != iterations
            or f"tprs_{meta['tprs']}" != timer.name
        ):
            raise ValueError(f"{timer}: rank {rank} 的场景/调用次数与其他 rank 不一致")
        rows = read_csv(timer / f"timer_rank_{rank}.csv")
        if len(rows) != meta["cores"] or sorted(
            int(r["group_id"]) for r in rows
        ) != list(range(meta["cores"])):
            raise ValueError(f"{timer}: rank {rank} 的 AIV 记录缺失或重复")
        all_rows.extend(rows)
        raw_rows = read_csv(timer / "data" / f"timer_rank_{rank}_timeline.csv")
        card_rows.append(card_spans(raw_rows, meta))
        profile = results / "profile" / timer.name / f"rank_{rank:03d}"
        captures = []
        for path in sorted(profile.rglob("op_summary*.csv")):
            samples = [
                r
                for r in read_csv(path)
                if r.get("OP Type") == "MoeDistributeDispatchV2"
                and r.get("Task Type") == "MIX_AIV"
            ]
            if samples:
                captures.append((path, samples))
        # Ignore unrelated/failed captures, but do not silently mix successful runs.
        if len(captures) != 1:
            raise ValueError(
                f"{profile}: 找到 {len(captures)} 份包含 dispatch 的 profiler 文件；"
                "需要唯一一份与打点对应的采集，请分开不同运行的输出目录"
            )
        path, samples = captures[0]
        if len(samples) != iterations:
            raise ValueError(
                f"{path}: dispatch 样本数 {len(samples)} != metadata 调用次数 {iterations}"
            )
        samples.sort(key=lambda r: Decimal(r["Task Start Time(us)"].strip()))
        last_us = float(samples[-1]["Task Duration(us)"])
        last_times.append(last_us)
        for kind in sorted({r["send_kind"] for r in rows}):
            subset = [r for r in rows if r["send_kind"] == kind]
            item = dict(
                rank=rank,
                tprs=meta["tprs"],
                send_kind=kind,
                cores=len(subset),
                iteration=iterations,
                profiler_last_us=last_us,
            )
            for field, label in METRICS:
                values = [float(r[field]) for r in subset]
                for stat, func in [
                    ("min", min),
                    ("median", statistics.median),
                    ("max", max),
                ]:
                    item[f"{label}_core_{stat}_us"] = func(values)
            summary.append(item)
    output = results / f"last_iteration_summary_{timer.name}.csv"
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(
        f"[{timer.name}] {world} ranks, {len(all_rows)} AIV records, iteration={iterations}"
    )
    print(
        "Last profiler min/median/max (us):",
        min(last_times),
        statistics.median(last_times),
        max(last_times),
    )
    for kind in sorted({r["send_kind"] for r in all_rows}):
        rows = [r for r in all_rows if r["send_kind"] == kind]
        print(f"  {kind}: {len(rows)} cores")
        for field, label in METRICS:
            values = [float(r[field]) for r in rows]
            print(
                f"    {label} min/median/max (us): {min(values):.3f} / "
                f"{statistics.median(values):.3f} / {max(values):.3f}; nonzero={sum(v > 0 for v in values)}"
            )
    print("Output:", output)
    card_output = results / f"card_span_summary_{timer.name}.csv"
    with card_output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(card_rows[0]))
        writer.writeheader()
        writer.writerows(card_rows)
    print("Card spans (same-rank common-counter assumption):", card_output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        nargs="?",
        default=Path("."),
        help="包含 timer/ 和 profile/ 的测试目录，默认当前目录",
    )
    parser.add_argument(
        "--tprs", type=int, help="仅汇总指定 token 数的场景；默认汇总所有有打点的场景"
    )
    args = parser.parse_args()
    results = args.results.resolve()
    timers = (
        [results / "timer" / f"tprs_{args.tprs}"]
        if args.tprs is not None
        else sorted(p for p in (results / "timer").glob("tprs_*") if p.is_dir())
    )
    if not timers:
        parser.error(f"{results}/timer 下没有 tprs_* 场景目录")
    try:
        for timer in timers:
            summarize(results, timer)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
