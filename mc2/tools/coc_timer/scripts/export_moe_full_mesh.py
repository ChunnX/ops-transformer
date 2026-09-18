#!/usr/bin/env python3
"""Decode the compact full-mesh assist tail into CocTimer CSVs (ABI v6)."""
import argparse
import csv
import struct
from pathlib import Path

MAGIC = 0x434F434D4F453031
# Flat serial stages per core, no containment. Send cores record
# INIT -> CNT_SEND -> PAYLOAD -> RECV_COPY -> TAIL; cumsum cores record
# INIT -> CNT_CALC -> RECV_COPY -> TAIL. Durations add up to the kernel span.
EVENTS = (
    "AIV_INIT_0", "AIV_CNT_SEND_0", "AIV_PAYLOAD_0", "AIV_CNT_CALC_0",
    "AIV_RECV_COPY_0", "AIV_TAIL_0",
)
VERSION = 6
HEADER_WORDS = 16
RECORD_WORDS = 64  # uint64 words (512 B) per AIV record


def decode(data):
    if not data or len(data) % 512:
        raise ValueError(
            "Expected the entire contiguous int32 assist tensor (A * 128 elements)."
        )
    capacity_a = len(data) // 512
    offset = (capacity_a * 12 + 127) // 128 * 128
    if offset + 128 > len(data):
        raise ValueError("Assist tail is too small even for the timer header.")
    header = struct.unpack_from("<16Q", data, offset)
    (
        magic,
        version,
        status,
        a,
        cores,
        stride,
        events,
        rank,
        send_cores,
        packed,
        payload,
        stored_offset,
        required,
        freq,
        send_layout,
        arrival_only,
    ) = header
    if magic != MAGIC or version != VERSION:
        raise ValueError(
            "Expected timer ABI v6. Rebuild with the timer enabled and dump after stream synchronization."
        )
    stride_bytes = stride * 8
    if status == 2:
        raise ValueError(
            f"Timer disabled: tail has {len(data) - offset} bytes, needs {required} bytes for {cores} AIVs."
        )
    if (
        arrival_only not in (0, 1)
        or send_layout not in (0, 1)
        or status != 1
        or a != capacity_a
        or stride != RECORD_WORDS
        or events != len(EVENTS)
        or stored_offset != offset
        or cores == 0
        or send_cores > cores
        or freq != 50
        or required != 128 + cores * stride_bytes
        or offset + required > len(data)
    ):
        raise ValueError("Invalid timer layout or truncated assist dump.")
    rows = []
    for core in range(cores):
        record = struct.unpack_from(f"<{stride}Q", data, offset + 128 + core * stride_bytes)
        if record[0] != MAGIC or record[1] != core:
            raise ValueError(
                f"Missing completed record for AIV {core}; synchronize the dispatch stream before dumping."
            )
        pairs = list(
            zip(record[8 : 8 + 2 * len(EVENTS) : 2], record[9 : 8 + 2 * len(EVENTS) : 2])
        )
        for name, (start, end) in zip(EVENTS, pairs):
            if (start == 0) != (end == 0) or end < start:
                raise ValueError(f"Incomplete/invalid event {name} on AIV {core}.")
        if any(record[2:5]):
            raise ValueError(f"Stale data-wait counters on AIV {core}.")
        init, cnt_send, payload_send, cnt_calc, recv_copy, tail = pairs
        if not init[0]:
            raise ValueError(f"Missing init stage on AIV {core}.")
        chain = [init]
        if core < send_cores:
            if cnt_calc[0]:
                raise ValueError(
                    f"Send-core AIV {core} must not record the cumsum stage."
                )
            if not (cnt_send[0] and payload_send[0]):
                raise ValueError(f"Missing send stages on AIV {core}.")
            chain += [cnt_send, payload_send]
        else:
            if cnt_send[0] or payload_send[0]:
                raise ValueError(
                    f"Receive-only AIV {core} must not record send stages."
                )
            if not cnt_calc[0]:
                raise ValueError(f"Missing cumsum stage on AIV {core}.")
            chain += [cnt_calc]
        if not (recv_copy[0] and tail[0]):
            raise ValueError(f"Missing recv/tail stage on AIV {core}.")
        chain += [recv_copy, tail]
        for prev, cur in zip(chain, chain[1:]):
            if prev[1] > cur[0]:
                raise ValueError(
                    f"Overlapping serial stages on AIV {core}: {prev} vs {cur}."
                )
        rows.append({"pairs": pairs})
    return {
        "timer_version": version,
        "rank": rank,
        "cores": cores,
        "send_cores": send_cores,
        "arrival_only": bool(arrival_only),
        "send_layout": send_layout,
        "packed_token_bytes": packed,
        "payload_token_bytes": payload,
        "cycles_per_us": freq,
    }, rows


def export(data, output):
    metadata, rows = decode(data)
    output = Path(output)
    (output / "data").mkdir(parents=True, exist_ok=True)
    rank = metadata["rank"]
    raw_path = output / "data" / f"timer_rank_{rank}_timeline.csv"
    duration_path = output / f"timer_rank_{rank}.csv"
    identity = ["group_id", "core_type", "sub_group_id", "send_kind"]
    with (
        raw_path.open("w", newline="") as raw,
        duration_path.open("w", newline="") as durations,
    ):
        raw_writer, duration_writer = csv.writer(raw), csv.writer(durations)
        raw_writer.writerow(
            identity
            + [
                f"{name}_{suffix}"
                for name in EVENTS
                for suffix in ("start_cycle", "end_cycle", "duration_us")
            ]
        )
        duration_writer.writerow(
            identity + [f"{name}_duration(us)" for name in EVENTS]
        )
        for core, row in enumerate(rows):
            # Logical AIV lanes only: do not infer mixed AIC/AIV physical groups from GetTaskRation.
            if metadata["send_layout"] == 1:
                send_kind = (
                    "payload_count" if core < metadata["send_cores"] else "receive_only"
                )
            else:
                send_kind = "payload" if core < metadata["send_cores"] else "count"
            lane = [core, "AIV", 0, send_kind]
            pairs = row["pairs"]
            times = [(end - start) / metadata["cycles_per_us"] for start, end in pairs]
            raw_writer.writerow(
                lane
                + [
                    value
                    for pair, duration in zip(pairs, times)
                    for value in (*pair, duration)
                ]
            )
            duration_writer.writerow(lane + times)
    return metadata, raw_path, duration_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dump",
        type=Path,
        help="Entire little-endian int32 assist_info_for_combine binary dump",
    )
    parser.add_argument("--output", type=Path, default=Path("output_timer"))
    args = parser.parse_args()
    try:
        metadata, raw_path, duration_path = export(args.dump.read_bytes(), args.output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(metadata)
    print(raw_path)
    print(duration_path)


if __name__ == "__main__":
    main()
