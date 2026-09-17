#!/usr/bin/env python3
"""Decode the compact full-mesh assist tail into PR #5122 CocTimer CSVs."""

import argparse
import csv
import struct
from pathlib import Path

MAGIC = 0x434F434D4F453031
EVENTS = (
    "AIV_STAGE_0",
    "AIV_SEND_0",
    "AIV_COMM_TOTAL_0",
    "AIV_TAIL_0",
    "AIV_INIT_0",
    "AIV_CNT_SEND_0",
    "AIV_PAYLOAD_0",
    "AIV_CNT_CALC_0",
    "AIV_RECV_COPY_0",
    "AIV_EXPID_COPY_0",
    "AIV_CNT_NOTIFY_0",
    "AIV_WAIT_DISPATCH_0",
    "AIV_CUMSUM_CAL_0",
    "AIV_RECV_WAIT_0",
)
VERSION_EVENTS = {2: 3, 3: 4, 4: 9, 5: 14}
VERSION_STRIDE = {2: 16, 3: 16, 4: 32, 5: 64}  # uint64 words per AIV record


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
    if magic != MAGIC or version not in VERSION_EVENTS:
        raise ValueError(
            "Expected timer ABI v2/v3/v4/v5. Rebuild with the timer enabled and dump after stream synchronization."
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
        or stride != VERSION_STRIDE[version]
        or events != VERSION_EVENTS[version]
        or stored_offset != offset
        or cores == 0
        or send_cores > cores
        or freq != 50
        or required != 128 + cores * stride_bytes
        or offset + required > len(data)
    ):
        raise ValueError("Invalid timer layout or truncated assist dump.")
    event_names = EVENTS[:events]
    rows = []
    for core in range(cores):
        record = struct.unpack_from(
            f"<{stride}Q", data, offset + 128 + core * stride_bytes
        )
        if record[0] != MAGIC or record[1] != core:
            raise ValueError(
                f"Missing completed record for AIV {core}; synchronize the dispatch stream before dumping."
            )
        pairs = list(
            zip(record[8 : 8 + 2 * events : 2], record[9 : 8 + 2 * events : 2])
        )
        for name, (start, end) in zip(event_names, pairs):
            if (start == 0) != (end == 0) or end < start:
                raise ValueError(f"Incomplete/invalid event {name} on AIV {core}.")
        if not pairs[0][0]:
            raise ValueError(f"Missing stage interval on AIV {core}.")
        stage, send, total = pairs[:3]
        if bool(send[0]) != bool(total[0]) or (
            send[0]
            and not (
                stage[0] <= send[0] <= send[1] <= stage[1]
                and send[0] <= total[0] <= send[1]
                and total[1] <= stage[1]
            )
        ):
            raise ValueError(f"Invalid send/total interval on AIV {core}.")
        wait_cycles, wait_episodes, batches = record[2:5]
        if (
            wait_cycles > stage[1] - stage[0]
            or wait_episodes > batches
            or (wait_cycles and not wait_episodes)
            or (send[0] and wait_cycles > total[1] - total[0])
        ):
            raise ValueError(f"Invalid data-wait counters on AIV {core}.")
        if 3 <= version < 5:
            tail = pairs[3]
            if not (stage[0] <= tail[0] <= stage[1] <= tail[1]):
                raise ValueError(f"Invalid tail interval on AIV {core}.")
            if send[0] and tail[0] < send[1]:
                raise ValueError(
                    f"Arrival precedes local send completion on AIV {core}."
                )
        if version == 4:
            init, cnt_send, payload_send, cnt_calc, recv_copy = pairs[4:9]
            if not (init[0] and init[1] <= stage[0]):
                raise ValueError(f"Invalid init interval on AIV {core}.")
            for interval in (cnt_send, payload_send, cnt_calc, recv_copy):
                if interval[0] and not (
                    stage[0] <= interval[0] and interval[1] <= tail[1]
                ):
                    raise ValueError(
                        f"Stage interval escapes the process span on AIV {core}."
                    )
            if core < send_cores:
                if not (
                    cnt_send[0]
                    and payload_send[0]
                    and send[0]
                    and cnt_send[1] <= payload_send[1] <= send[1] <= recv_copy[0]
                ):
                    raise ValueError(f"Invalid send-core stage chain on AIV {core}.")
                if cnt_calc[0]:
                    raise ValueError(
                        f"Send-core AIV {core} must not record the cumsum stage."
                    )
            elif cnt_send[0] or payload_send[0]:
                raise ValueError(
                    f"Receive-only AIV {core} must not record send stages."
                )
            if core >= send_cores and not cnt_calc[0]:
                raise ValueError(f"Missing cumsum stage on AIV {core}.")
            if not recv_copy[0]:
                raise ValueError(f"Missing recv-copy stage on AIV {core}.")
        if version == 5:
            init, cnt_send, payload_send, cnt_calc, recv_copy = pairs[4:9]
            expid_copy, cnt_notify, wait_dispatch, cumsum_cal, recv_wait = pairs[9:14]
            tail = pairs[3]
            if not (recv_copy[1] <= tail[0] <= tail[1] <= stage[1]):
                raise ValueError(f"Invalid tail interval on AIV {core}.")
            if not (init[0] and init[1] <= stage[0]):
                raise ValueError(f"Invalid init interval on AIV {core}.")
            for interval in (
                cnt_send,
                payload_send,
                cnt_calc,
                recv_copy,
                expid_copy,
                cnt_notify,
                wait_dispatch,
                cumsum_cal,
                recv_wait,
            ):
                if interval[0] and not (
                    stage[0] <= interval[0] and interval[1] <= stage[1]
                ):
                    raise ValueError(
                        f"Stage interval escapes the process span on AIV {core}."
                    )
            if core < send_cores:
                if not (
                    cnt_send[0]
                    and payload_send[0]
                    and send[0]
                    and cnt_send[1] <= payload_send[1] <= send[1] <= recv_copy[0]
                ):
                    raise ValueError(f"Invalid send-core stage chain on AIV {core}.")
                if cnt_calc[0] or wait_dispatch[0] or cumsum_cal[0]:
                    raise ValueError(
                        f"Send-core AIV {core} must not record the cumsum stage."
                    )
                if not expid_copy[0]:
                    raise ValueError(f"Missing expertIds stage on AIV {core}.")
                if not (cnt_send[0] <= expid_copy[0] and expid_copy[1] <= cnt_send[1]):
                    raise ValueError(
                        f"expertIds stage escapes cnt-send span on AIV {core}."
                    )
                if cnt_notify[0] and not (
                    send[0] <= cnt_notify[0] and cnt_notify[1] <= payload_send[0]
                ):
                    raise ValueError(f"Invalid cnt-notify interval on AIV {core}.")
            else:
                if cnt_send[0] or payload_send[0] or expid_copy[0] or cnt_notify[0]:
                    raise ValueError(
                        f"Receive-only AIV {core} must not record send stages."
                    )
                if not cnt_calc[0]:
                    raise ValueError(f"Missing cumsum stage on AIV {core}.")
                if not (wait_dispatch[0] and cumsum_cal[0]):
                    raise ValueError(f"Missing cumsum sub-stages on AIV {core}.")
                if not (
                    cnt_calc[0] <= wait_dispatch[0]
                    and wait_dispatch[1] <= cumsum_cal[0]
                    and cumsum_cal[1] <= cnt_calc[1]
                ):
                    raise ValueError(f"Invalid cumsum sub-stage chain on AIV {core}.")
            if not recv_copy[0]:
                raise ValueError(f"Missing recv-copy stage on AIV {core}.")
            if not recv_wait[0]:
                raise ValueError(f"Missing recv-wait stage on AIV {core}.")
            if not (recv_copy[0] <= recv_wait[0] and recv_wait[1] <= recv_copy[1]):
                raise ValueError(
                    f"recv-wait stage escapes recv-copy span on AIV {core}."
                )
        rows.append(
            {
                "pairs": pairs,
                "wait_cycles": wait_cycles,
                "wait_episodes": wait_episodes,
                "receive_batches": batches,
            }
        )
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
    event_names = EVENTS[: VERSION_EVENTS[metadata["timer_version"]]]
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
                for name in event_names
                for suffix in ("start_cycle", "end_cycle", "duration_us")
            ]
            + [
                "DATA_WAIT_cycles",
                "DATA_WAIT_duration_us",
                "DATA_WAIT_episodes",
                "receive_batches",
            ]
        )
        duration_writer.writerow(
            identity
            + [f"{name}_duration(us)" for name in event_names]
            + ["DATA_WAIT_duration(us)", "DATA_WAIT_episodes", "receive_batches"]
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
            wait = [
                row["wait_cycles"] / metadata["cycles_per_us"],
                row["wait_episodes"],
                row["receive_batches"],
            ]
            # Waiting is a sum of disjoint stalls: no invented start/end pair in the timeline.
            raw_writer.writerow(
                lane
                + [
                    value
                    for pair, duration in zip(pairs, times)
                    for value in (*pair, duration)
                ]
                + [row["wait_cycles"]]
                + wait
            )
            duration_writer.writerow(lane + times + wait)
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
