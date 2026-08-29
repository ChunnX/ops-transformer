#!/usr/bin/env python3
"""Decode arch92 GMM2/Combine counter set_cond diagnostics."""

import argparse
import re
from collections import defaultdict
from pathlib import Path


COND_RE = re.compile(r"MOV_SPR_XN\s+SPR:COND,\s+XN:[^=]+=0x([0-9a-fA-F]+)")
SOURCE_RE = re.compile(r"^(.*?instr_log\.dump):\d+:")
UINT_BIAS = 1_000_000

CONFIG_MARKER = 913000
RESET_MARKER_BASE = 913010
RESET_MARKER_LIMIT = RESET_MARKER_BASE + 256
NOTIFY_MAGIC = 0xD
OVERSHOOT_MAGIC = 0xE
TILE_N = 256
TILE_K = 128
CONFIG_FIELDS = (
    "get_block_num",
    "aic_num",
    "block_aiv_num",
    "sync_slots_per_expert",
    "expert_count",
    "token_hidden_dim",
)


def decode_uint(raw):
    if raw < UINT_BIAS:
        raise ValueError(f"expected biased uint, got 0x{raw:x}")
    return raw - UINT_BIAS


def read_streams(paths):
    streams = defaultdict(list)
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            match = COND_RE.search(line)
            if match is None:
                continue
            source_match = SOURCE_RE.match(line)
            source = source_match.group(1) if source_match else str(path)
            streams[source].append(int(match.group(1), 16))
    return streams


def decode_notify(raw, source):
    return {
        "source": source,
        "raw": raw,
        "expert": (raw >> 24) & 0xF,
        "group": (raw >> 19) & 0x1F,
        "counter_before": (raw >> 13) & 0x3F,
        "block": (raw >> 7) & 0x3F,
        "n_tile": (raw >> 2) & 0x1F,
        "k_tile": raw & 0x3,
    }


def decode_overshoot(raw, source):
    return {
        "source": source,
        "raw": raw,
        "expert": (raw >> 24) & 0xF,
        "group": (raw >> 19) & 0x1F,
        "observed": (raw >> 13) & 0x3F,
        "job": (raw >> 7) & 0x3F,
        "slot": raw & 0x7F,
    }


def scan_stream(source, values):
    result = {
        "configs": [],
        "resets": [],
        "notifications": [],
        "overshoots": [],
    }
    index = 0
    while index < len(values):
        value = values[index]

        if value == CONFIG_MARKER:
            end = index + 1 + len(CONFIG_FIELDS)
            payload = values[index + 1:end]
            if len(payload) != len(CONFIG_FIELDS):
                print(f"WARNING: incomplete {CONFIG_MARKER} record in {source}")
                break
            try:
                config = {
                    field: decode_uint(raw)
                    for field, raw in zip(CONFIG_FIELDS, payload)
                }
            except ValueError as error:
                print(f"WARNING: invalid {CONFIG_MARKER} record in {source}: {error}")
                index += 1
                continue
            config["source"] = source
            result["configs"].append(config)
            index = end
            continue

        if RESET_MARKER_BASE <= value < RESET_MARKER_LIMIT:
            payload = values[index + 1:index + 3]
            if len(payload) != 2:
                print(f"WARNING: incomplete reset record 0x{value:x} in {source}")
                break
            try:
                first_slot = decode_uint(payload[0])
                last_slot = decode_uint(payload[1])
            except ValueError as error:
                print(f"WARNING: invalid reset record 0x{value:x} in {source}: {error}")
                index += 1
                continue
            result["resets"].append({
                "source": source,
                "expert": value - RESET_MARKER_BASE,
                "first_slot": first_slot,
                "last_slot": last_slot,
            })
            index += 3
            continue

        magic = (value >> 28) & 0xF
        if magic == NOTIFY_MAGIC:
            result["notifications"].append(decode_notify(value, source))
        elif magic == OVERSHOOT_MAGIC:
            result["overshoots"].append(decode_overshoot(value, source))
        index += 1

    return result


def print_config(configs):
    expected_n_tiles = None
    slot_mismatch = False
    for config in configs:
        expected_n_tiles = (config["token_hidden_dim"] + TILE_N - 1) // TILE_N
        print(f"\n[{CONFIG_MARKER}] runtime configuration")
        print(f"source={config['source']}")
        for field in CONFIG_FIELDS:
            print(f"{field}={config[field]}")
        print(f"expected_n_tiles_per_group={expected_n_tiles}")
        if config["sync_slots_per_expert"] < config["get_block_num"]:
            slot_mismatch = True
            print(
                "ERROR: sync_slots_per_expert < get_block_num. "
                "Adjacent expert counter regions can overlap."
            )
    return expected_n_tiles, slot_mismatch


def print_resets(resets):
    reset_nonzero = False
    if not resets:
        print("\nNo reset snapshots found.")
        return reset_nonzero

    print("\nReset snapshots (before any GMM2):")
    for item in sorted(resets, key=lambda value: (value["expert"], value["source"])):
        ok = item["first_slot"] == 0 and item["last_slot"] == 0
        reset_nonzero |= not ok
        print(
            f"expert={item['expert']} first_slot={item['first_slot']} "
            f"last_slot={item['last_slot']} status={'OK' if ok else 'NONZERO'}"
        )
    return reset_nonzero


def print_notifications(notifications, expected_n_tiles):
    duplicate_found = False
    stale_or_alias_found = False
    k_split_found = False
    if not notifications:
        print("\nNo AIC notification records found.")
        return duplicate_found, stale_or_alias_found, k_split_found

    notifications.sort(
        key=lambda item: (
            item["expert"],
            item["group"],
            item["counter_before"],
            item["n_tile"],
            item["k_tile"],
            item["block"],
        )
    )
    grouped = defaultdict(list)

    print("\nAIC notification records:")
    for item in notifications:
        grouped[(item["expert"], item["group"])].append(item)
        print(
            f"raw=0x{item['raw']:08x} expert={item['expert']} group={item['group']} "
            f"before={item['counter_before']} block={item['block']} "
            f"n_tile={item['n_tile']} n_loc={item['n_tile'] * TILE_N} "
            f"k_tile={item['k_tile']} k_loc={item['k_tile'] * TILE_K}"
        )

    print("\nAIC notification summary:")
    for key in sorted(grouped):
        records = grouped[key]
        coordinates = defaultdict(list)
        for item in records:
            coordinates[(item["n_tile"], item["k_tile"])].append(item)

        duplicates = {
            coordinate: items
            for coordinate, items in coordinates.items()
            if len(items) > 1
        }
        duplicate_found |= bool(duplicates)
        before_values = [item["counter_before"] for item in records]
        blocks = sorted({item["block"] for item in records})
        n_tiles = sorted({item["n_tile"] for item in records})
        k_tiles = sorted({item["k_tile"] for item in records})
        complete_capture = key == (0, 0)

        print(
            f"expert={key[0]} group={key[1]} "
            f"capture={'complete' if complete_capture else 'target-window'} "
            f"records={len(records)} unique_coords={len(coordinates)} "
            f"before_range={min(before_values)}..{max(before_values)} "
            f"blocks={blocks} n_tiles={n_tiles} k_tiles={k_tiles}"
        )

        if complete_capture and expected_n_tiles is not None:
            if len(records) != expected_n_tiles:
                duplicate_found |= len(records) > expected_n_tiles
                print(
                    f"  ERROR: captured {len(records)} notifications; "
                    f"expected {expected_n_tiles}."
                )

        for coordinate, items in sorted(duplicates.items()):
            print(
                f"  DUPLICATE n_tile={coordinate[0]} k_tile={coordinate[1]} "
                f"blocks={[item['block'] for item in items]} "
                f"before={[item['counter_before'] for item in items]}"
            )

        target = expected_n_tiles if expected_n_tiles is not None else 16
        if min(before_values) >= target:
            stale_or_alias_found = True
            print(
                "  ERROR: this producer started from a counter already at/above target. "
                "This indicates stale state or cross-expert slot aliasing."
            )

        if any(k_tile != 0 for k_tile in k_tiles):
            k_split_found = True
            print(
                "  ERROR: non-zero K tile published completion. "
                "Verify notification happens only after the final K contribution."
            )

    return duplicate_found, stale_or_alias_found, k_split_found


def print_overshoots(overshoots):
    if not overshoots:
        print("\nNo AIV wait overshoot records found.")
        return False

    print("\nAIV wait overshoots:")
    for item in sorted(
        overshoots,
        key=lambda value: (
            value["expert"],
            value["group"],
            value["job"],
            value["slot"],
        ),
    ):
        print(
            f"raw=0x{item['raw']:08x} expert={item['expert']} group={item['group']} "
            f"observed={item['observed']} job={item['job']} slot={item['slot']}"
        )
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Decode arch92 GMM2/Combine counter diagnostics"
    )
    parser.add_argument("logs", nargs="+", help="Raw instruction log or grep output files")
    args = parser.parse_args()

    streams = read_streams(args.logs)
    results = [scan_stream(source, values) for source, values in streams.items()]
    configs = [item for result in results for item in result["configs"]]
    resets = [item for result in results for item in result["resets"]]
    notifications = [item for result in results for item in result["notifications"]]
    overshoots = [item for result in results for item in result["overshoots"]]

    if not (configs or resets or notifications or overshoots):
        raise SystemExit("No arch92 GMM2/Combine diagnostic markers were found")

    print("=== Arch92 GMM2/Combine counter diagnostics ===")
    expected_n_tiles, slot_mismatch = print_config(configs)
    reset_nonzero = print_resets(resets)
    duplicate_found, stale_or_alias_found, k_split_found = print_notifications(
        notifications, expected_n_tiles
    )
    overshoot_found = print_overshoots(overshoots)

    print("\nLikely cause:")
    if slot_mismatch:
        print(
            "HOST/KERNEL SLOT COUNT MISMATCH: the runtime allocates fewer slots per "
            "expert than the kernel indexes. This can make adjacent experts share counters."
        )
    elif reset_nonzero:
        print("RESET FAILURE: at least one counter was non-zero before GMM2 started.")
    elif duplicate_found:
        print("DUPLICATE NOTIFICATION: at least one logical GMM2 tile was published more than once.")
    elif k_split_found:
        print("EARLY K-SPLIT NOTIFICATION: a non-final K contribution published completion.")
    elif stale_or_alias_found:
        print("STALE OR ALIASED COUNTER: a producer started from a counter already at target.")
    elif overshoot_found:
        print("COUNTER OVERSHOOT CONFIRMED, but the producer-side cause needs the notification records.")
    else:
        print("No synchronization violation was identified in the captured records.")


if __name__ == "__main__":
    main()
