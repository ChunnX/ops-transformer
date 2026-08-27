#!/usr/bin/env python3
"""Decode, validate, summarize and compare MegaMoe A8W4 critical-path raw data."""

from __future__ import annotations

import argparse
import csv
import html
import json
import struct
import tempfile
import unittest
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

MAGIC = 0x41385452
VERSION = 1
HEADER_WORDS = 24
RECORD_HEADER_WORDS = 8
EVENT_WORDS = 13
ROLE_NAMES = {0: "AIC", 1: "AIV0", 2: "AIV1"}
NO_SCOPE = 0xFFFFFFFF
EVENT_NAMES = {
    1: "OP_PRE_MOE",
    2: "DISPATCH_WORK",
    3: "WAIT_FOR_DISPATCH",
    4: "GMM1_PROLOGUE",
    5: "GMM1_COMPUTE",
    6: "WAIT_FOR_GMM1",
    7: "ACTIVATION_WORK",
    8: "WAIT_FOR_ACTIVATION",
    9: "GMM2_PROLOGUE",
    10: "GMM2_COMPUTE",
    11: "WAIT_FOR_GMM2",
    12: "COMBINE_WORK",
    13: "FINAL_COMBINE_OR_DRAIN",
    14: "COUNT_EXPORT",
    15: "UNPERMUTE",
    16: "OP_POST_MOE",
    17: "CROSS_RANK_INPUT_SYNC",
    18: "CROSS_RANK_OUTPUT_SYNC",
    19: "SHARED_GMM1_ACTIVATION",
    20: "SHARED_GMM2",
}
FLAG_COMPLETE = 1
FLAG_SUBMIT = 2
FLAG_WAIT = 4
FLAG_TRACE_ONLY = 8


class TraceFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    source: str
    rank: int
    role: str
    core: int
    event_id: int
    name: str
    parent_id: int
    flags: int
    wave: int | None
    expert: int | None
    tile: int | None
    token_start: int
    token_count: int
    work_count: int
    begin: int
    end: int | None

    @property
    def duration(self) -> int | None:
        return None if self.end is None else self.end - self.begin


@dataclass
class RecordInfo:
    source: str
    rank: int
    role: str
    core: int
    process_begin: int
    process_end: int
    written_count: int
    dropped_count: int


@dataclass
class TraceData:
    source: str
    header: dict
    records: list[RecordInfo]
    events: list[Event]
    warnings: list[str]


def _u64(lo: int, hi: int) -> int:
    return lo | (hi << 32)


def _scope(value: int) -> int | None:
    return None if value == NO_SCOPE else value


def parse_raw(path: Path) -> TraceData:
    blob = path.read_bytes()
    if len(blob) % 4:
        raise TraceFormatError(f"{path}: byte size {len(blob)} is not uint32 aligned")
    if len(blob) < HEADER_WORDS * 4:
        raise TraceFormatError(f"{path}: raw is shorter than the protocol header")
    words = struct.unpack(f"<{len(blob) // 4}I", blob)
    if words[0] != MAGIC:
        raise TraceFormatError(
            f"{path}: bad magic 0x{words[0]:08x}; raw may be unpublished or stale"
        )
    if words[1] != VERSION:
        raise TraceFormatError(
            f"{path}: protocol version {words[1]} != supported {VERSION}"
        )
    header_words, record_words, event_words = words[2], words[3], words[4]
    record_count, core_count, capacity = words[5], words[6], words[7]
    if header_words != HEADER_WORDS or event_words != EVENT_WORDS:
        raise TraceFormatError(f"{path}: incompatible header/event word count")
    if core_count == 0 or record_count != core_count * len(ROLE_NAMES):
        raise TraceFormatError(
            f"{path}: invalid record/core count {record_count}/{core_count}"
        )
    expected_record_words = RECORD_HEADER_WORDS + capacity * EVENT_WORDS
    if record_words != expected_record_words:
        raise TraceFormatError(
            f"{path}: record_words={record_words}, expected {expected_record_words}"
        )
    required_words = header_words + record_count * record_words
    if required_words > len(words):
        raise TraceFormatError(
            f"{path}: records require {required_words * 4} bytes, raw has {len(blob)}"
        )

    header = {
        "version": words[1],
        "record_count": record_count,
        "core_count": core_count,
        "event_capacity": capacity,
        "rank": words[8],
        "bs": words[9],
        "h": words[10],
        "hidden_dim": words[11],
        "topk": words[12],
        "expert_count": words[13],
        "world_size": words[14],
        "wave_m_group_target": words[15],
        "clock_bits": words[16],
    }
    records: list[RecordInfo] = []
    events: list[Event] = []
    warnings: list[str] = []
    for record_index in range(record_count):
        base = header_words + record_index * record_words
        process_begin = _u64(words[base], words[base + 1])
        process_end = _u64(words[base + 2], words[base + 3])
        written, dropped, rank, packed_role_core = words[base + 4 : base + 8]
        role_id, core = packed_role_core >> 16, packed_role_core & 0xFFFF
        expected_role, expected_core = divmod(record_index, core_count)
        if (
            role_id not in ROLE_NAMES
            or role_id != expected_role
            or core != expected_core
        ):
            raise TraceFormatError(
                f"{path}: record {record_index} has invalid role/core {role_id}/{core}"
            )
        if rank != header["rank"]:
            raise TraceFormatError(
                f"{path}: record {record_index} rank {rank} != header rank {header['rank']}"
            )
        if written > capacity:
            raise TraceFormatError(
                f"{path}: record {record_index} written_count {written} > {capacity}"
            )
        if process_end < process_begin:
            raise TraceFormatError(
                f"{path}: record {record_index} process interval is reversed"
            )
        role = ROLE_NAMES[role_id]
        records.append(
            RecordInfo(
                str(path),
                rank,
                role,
                core,
                process_begin,
                process_end,
                written,
                dropped,
            )
        )
        if dropped:
            warnings.append(
                f"rank {rank} {role}{core}: truncated, dropped {dropped} event(s)"
            )
        for event_index in range(written):
            pos = base + RECORD_HEADER_WORDS + event_index * EVENT_WORDS
            fields = words[pos : pos + EVENT_WORDS]
            event_id, parent_id, flags = fields[:3]
            if event_id not in EVENT_NAMES:
                raise TraceFormatError(
                    f"{path}: record {record_index} event {event_index}: unknown id {event_id}"
                )
            begin, raw_end = _u64(fields[9], fields[10]), _u64(fields[11], fields[12])
            end = None if raw_end == 0 else raw_end
            if end is None:
                warnings.append(
                    f"rank {rank} {role}{core}: unpaired {EVENT_NAMES[event_id]} at {begin}"
                )
            elif end < begin:
                raise TraceFormatError(
                    f"{path}: rank {rank} {role}{core} {EVENT_NAMES[event_id]} end {end} < begin {begin}"
                )
            events.append(
                Event(
                    str(path),
                    rank,
                    role,
                    core,
                    event_id,
                    EVENT_NAMES[event_id],
                    parent_id,
                    flags,
                    _scope(fields[3]),
                    _scope(fields[4]),
                    _scope(fields[5]),
                    fields[6],
                    fields[7],
                    fields[8],
                    begin,
                    end,
                )
            )
    events.sort(
        key=lambda e: (e.rank, e.begin, e.role, e.core, e.event_id, e.wave or -1)
    )
    return TraceData(str(path), header, records, events, warnings)


def load_many(paths: Sequence[Path]) -> tuple[list[TraceData], list[Event], list[str]]:
    traces = [parse_raw(path) for path in paths]
    events = sorted(
        (event for trace in traces for event in trace.events),
        key=lambda e: (e.rank, e.begin, e.role, e.core, e.event_id),
    )
    warnings = [warning for trace in traces for warning in trace.warnings]
    ranks = [trace.header["rank"] for trace in traces]
    if len(ranks) != len(set(ranks)):
        warnings.append(
            "multiple raw files carry the same rank; kept as separate sources"
        )
    return traces, events, warnings


def _active_duration(event: Event, events: Sequence[Event]) -> int | None:
    if event.duration is None:
        return None
    waits = []
    for child in events:
        if (
            child.end is None
            or not (child.flags & FLAG_WAIT)
            or child.parent_id != event.event_id
        ):
            continue
        if (child.rank, child.role, child.core) != (event.rank, event.role, event.core):
            continue
        if child.begin < event.begin or child.end > event.end:
            continue
        if (
            event.wave is not None
            and child.wave is not None
            and event.wave != child.wave
        ):
            continue
        if (
            event.expert is not None
            and child.expert is not None
            and event.expert != child.expert
        ):
            continue
        waits.append((child.begin, child.end))
    covered = 0
    merged_end = -1
    for begin, end in sorted(waits):
        if begin > merged_end:
            covered += end - begin
            merged_end = end
        elif end > merged_end:
            covered += end - merged_end
            merged_end = end
    return event.duration - covered


def summarize(events: Sequence[Event]) -> list[dict]:
    groups: dict[tuple[int, str, str], list[tuple[int, int]]] = defaultdict(list)
    for event in events:
        if event.duration is not None:
            groups[(event.rank, event.role, event.name)].append(
                (event.duration, _active_duration(event, events))
            )
    rows = []
    for (rank, role, name), values in sorted(groups.items()):
        totals = [value[0] for value in values]
        active = [value[1] for value in values]
        rows.append(
            {
                "rank": rank,
                "role": role,
                "event": name,
                "count": len(values),
                "total_cycles": sum(totals),
                "active_cycles": sum(active),
                "wait_cycles": sum(totals) - sum(active),
                "mean_cycles": sum(totals) / len(totals),
                "max_cycles": max(totals),
                "min_cycles": min(totals),
            }
        )
    return rows


def operator_breakdown(
    traces: Sequence[TraceData], events: Sequence[Event]
) -> list[dict]:
    rows = []
    ranks = sorted({trace.header["rank"] for trace in traces})
    main_ids = set(range(2, 14))
    for rank in ranks:
        records = [
            record
            for trace in traces
            for record in trace.records
            if record.rank == rank
        ]
        main = [
            event
            for event in events
            if event.rank == rank
            and event.event_id in main_ids
            and event.end is not None
        ]
        if not records:
            continue
        op_begin = min(record.process_begin for record in records)
        op_end = max(record.process_end for record in records)
        row = {
            "rank": rank,
            "op_begin": op_begin,
            "op_end": op_end,
            "op_total_cycles": op_end - op_begin,
            "main_begin": None,
            "main_end": None,
            "main_cycles": None,
            "outside_main_cycles": None,
            "outside_main_percent": None,
        }
        if main:
            main_begin = min(event.begin for event in main)
            main_end = max(event.end for event in main if event.end is not None)
            outside = max(0, main_begin - op_begin) + max(0, op_end - main_end)
            row.update(
                {
                    "main_begin": main_begin,
                    "main_end": main_end,
                    "main_cycles": main_end - main_begin,
                    "outside_main_cycles": outside,
                    "outside_main_percent": 100.0 * outside / max(1, op_end - op_begin),
                }
            )
        rows.append(row)
    return rows


def write_events_csv(path: Path, events: Sequence[Event]) -> None:
    fields = (
        list(asdict(events[0]).keys()) + ["duration", "active_duration"]
        if events
        else ["source"]
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for event in events:
            row = asdict(event)
            row["duration"] = event.duration
            row["active_duration"] = _active_duration(event, events)
            writer.writerow(row)


def write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = [
        "rank",
        "role",
        "event",
        "count",
        "total_cycles",
        "active_cycles",
        "wait_cycles",
        "mean_cycles",
        "max_cycles",
        "min_cycles",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_timeline_html(path: Path, events: Sequence[Event]) -> None:
    paired = [event for event in events if event.end is not None]
    if not paired:
        path.write_text(
            "<html><body>No paired events.</body></html>\n", encoding="utf-8"
        )
        return
    rank_origins = {
        rank: min(event.begin for event in paired if event.rank == rank)
        for rank in {event.rank for event in paired}
    }
    span = max(
        1,
        max(
            event.end - rank_origins[event.rank]
            for event in paired
            if event.end is not None
        ),
    )
    lanes = sorted({(event.rank, event.role, event.core) for event in paired})
    colors = {
        name: f"hsl({(event_id * 47) % 360} 65% 58%)"
        for event_id, name in EVENT_NAMES.items()
    }
    parts = [
        "<!doctype html><meta charset='utf-8'><title>MegaMoe A8W4 trace</title>",
        "<style>body{font:12px sans-serif}.lane{height:24px;position:relative;border-bottom:1px solid #ddd;"
        "margin-left:130px}.label{position:absolute;left:-125px;width:120px;text-align:right}.e{position:absolute;"
        "height:18px;top:3px;min-width:1px;border-radius:2px}.e:hover:after{content:attr(data-tip);position:fixed;"
        "left:10px;bottom:10px;background:#111;color:#fff;padding:5px;z-index:2}</style>",
    ]
    by_lane: dict[tuple[int, str, int], list[Event]] = defaultdict(list)
    for event in paired:
        by_lane[(event.rank, event.role, event.core)].append(event)
    for lane in lanes:
        parts.append(
            f"<div class='lane'><span class='label'>rank{lane[0]} {lane[1]}{lane[2]}</span>"
        )
        for event in by_lane[lane]:
            left = (event.begin - rank_origins[event.rank]) * 100.0 / span
            width = max(0.08, (event.duration or 0) * 100.0 / span)
            tip = html.escape(
                f"{event.name} wave={event.wave} expert={event.expert} tile={event.tile} "
                f"cycles={event.duration} flags=0x{event.flags:x}",
                quote=True,
            )
            parts.append(
                f"<span class='e' data-tip='{tip}' style='left:{left:.5f}%;width:{width:.5f}%;"
                f"background:{colors[event.name]}'></span>"
            )
        parts.append("</div>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def aggregate_for_compare(events: Sequence[Event]) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for event in events:
        if event.duration is not None:
            grouped[(event.role, event.name)].append(
                (event.duration, _active_duration(event, events))
            )
    return {
        key: {
            "count": len(values),
            "total": sum(value[0] for value in values),
            "active": sum(value[1] for value in values),
            "wait": sum(value[0] - value[1] for value in values),
            "mean": sum(value[0] for value in values) / len(values),
            "max": max(value[0] for value in values),
        }
        for key, values in grouped.items()
    }


def compare(good: Sequence[Event], bad: Sequence[Event]) -> list[dict]:
    ga, ba = aggregate_for_compare(good), aggregate_for_compare(bad)
    rows = []
    for role, event in sorted(set(ga) | set(ba)):
        g, b = ga.get((role, event), {}), ba.get((role, event), {})
        row = {"role": role, "event": event}
        for metric in ("count", "total", "active", "wait", "mean", "max"):
            gv, bv = g.get(metric), b.get(metric)
            row[f"good_{metric}"] = gv
            row[f"bad_{metric}"] = bv
            row[f"delta_{metric}"] = None if gv is None or bv is None else bv - gv
        rows.append(row)
    return rows


def _write_words(path: Path, words: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(words)}I", *words))


def make_synthetic(
    path: Path, rank: int, specs: Sequence[dict], dropped: int = 0, core_count: int = 1
) -> None:
    capacity = max(8, len(specs))
    record_count = core_count * 3
    record_words = RECORD_HEADER_WORDS + capacity * EVENT_WORDS
    words = [0] * (HEADER_WORDS + record_count * record_words)
    words[:17] = [
        MAGIC,
        VERSION,
        HEADER_WORDS,
        record_words,
        EVENT_WORDS,
        record_count,
        core_count,
        capacity,
        rank,
        64,
        4096,
        7168,
        8,
        16,
        2,
        4,
        64,
    ]
    for record_index in range(record_count):
        base = HEADER_WORDS + record_index * record_words
        role, core = divmod(record_index, core_count)
        own = [
            spec
            for spec in specs
            if spec.get("role", 0) == role and spec.get("core", 0) == core
        ]
        begin = min((spec["begin"] for spec in own), default=100)
        ends = [spec.get("end", 0) for spec in own if spec.get("end", 0)]
        end = max(ends, default=begin)
        words[base : base + 8] = [
            begin & 0xFFFFFFFF,
            begin >> 32,
            end & 0xFFFFFFFF,
            end >> 32,
            len(own),
            dropped if record_index == 0 else 0,
            rank,
            role << 16 | core,
        ]
        for index, spec in enumerate(own):
            pos = base + RECORD_HEADER_WORDS + index * EVENT_WORDS
            event_begin, event_end = spec["begin"], spec.get("end", 0)
            words[pos : pos + EVENT_WORDS] = [
                spec["event"],
                spec.get("parent", 0),
                spec.get("flags", FLAG_COMPLETE),
                spec.get("wave", NO_SCOPE),
                spec.get("expert", NO_SCOPE),
                spec.get("tile", NO_SCOPE),
                spec.get("token_start", 0),
                spec.get("token_count", 0),
                spec.get("work_count", 0),
                event_begin & 0xFFFFFFFF,
                event_begin >> 32,
                event_end & 0xFFFFFFFF,
                event_end >> 32,
            ]
    _write_words(path, words)


class SyntheticTests(unittest.TestCase):
    def test_normal_and_sorting_and_repeat_wave(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "rank0.bin"
            make_synthetic(
                raw,
                0,
                [
                    {"event": 5, "begin": 300, "end": 360, "wave": 1, "expert": 2},
                    {"event": 5, "begin": 100, "end": 180, "wave": 0, "expert": 2},
                ],
            )
            parsed = parse_raw(raw)
            self.assertEqual([event.begin for event in parsed.events], [100, 300])
            self.assertEqual([event.wave for event in parsed.events], [0, 1])

    def test_unpaired_and_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "rank0.bin"
            make_synthetic(raw, 0, [{"event": 7, "begin": 100, "end": 0}], dropped=3)
            parsed = parse_raw(raw)
            self.assertIsNone(parsed.events[0].end)
            self.assertTrue(any("unpaired" in warning for warning in parsed.warnings))
            self.assertTrue(any("truncated" in warning for warning in parsed.warnings))

    def test_multi_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            p0, p1 = Path(directory) / "r0.bin", Path(directory) / "r1.bin"
            spec = [{"event": 2, "begin": 100, "end": 150, "role": 2}]
            make_synthetic(p0, 0, spec)
            make_synthetic(p1, 1, spec)
            traces, events, _ = load_many([p1, p0])
            self.assertEqual({trace.header["rank"] for trace in traces}, {0, 1})
            self.assertEqual([event.rank for event in events], [0, 1])

    def test_record_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "bad.bin"
            make_synthetic(raw, 0, [])
            raw.write_bytes(raw.read_bytes()[:-4])
            with self.assertRaises(TraceFormatError):
                parse_raw(raw)

    def test_active_time_subtracts_matching_wait(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "rank0.bin"
            make_synthetic(
                raw,
                0,
                [
                    {
                        "event": 5,
                        "begin": 100,
                        "end": 200,
                        "wave": 0,
                        "expert": 1,
                        "tile": 7,
                    },
                    {
                        "event": 3,
                        "parent": 5,
                        "flags": FLAG_COMPLETE | FLAG_WAIT,
                        "begin": 120,
                        "end": 150,
                        "wave": 0,
                        "expert": 1,
                        "tile": 7,
                    },
                ],
            )
            parsed = parse_raw(raw)
            compute = next(
                row
                for row in summarize(parsed.events)
                if row["event"] == "GMM1_COMPUTE"
            )
            self.assertEqual(compute["active_cycles"], 70)
            self.assertEqual(compute["wait_cycles"], 30)


def command_decode(args: argparse.Namespace) -> int:
    traces, events, warnings = load_many([Path(path) for path in args.raw])
    rows = summarize(events)
    if args.events_csv:
        write_events_csv(Path(args.events_csv), events)
    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), rows)
    if args.timeline:
        write_timeline_html(Path(args.timeline), events)
    result = {
        "headers": [trace.header for trace in traces],
        "warnings": warnings,
        "operator_breakdown": operator_breakdown(traces, events),
        "summary": rows,
        "event_count": len(events),
    }
    if args.json:
        Path(args.json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def command_compare(args: argparse.Namespace) -> int:
    good_traces, good, good_warnings = load_many([Path(path) for path in args.good])
    bad_traces, bad, bad_warnings = load_many([Path(path) for path in args.bad])
    rows = compare(good, bad)
    if args.csv:
        fields = list(rows[0]) if rows else ["role", "event"]
        with Path(args.csv).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "warnings": {"good": good_warnings, "bad": bad_warnings},
        "operator_breakdown": {
            "good": operator_breakdown(good_traces, good),
            "bad": operator_breakdown(bad_traces, bad),
        },
        "comparison": rows,
    }
    if args.json:
        Path(args.json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    decode = sub.add_parser(
        "decode", help="validate and decode one or more rank raw files"
    )
    decode.add_argument("raw", nargs="+")
    decode.add_argument("--events-csv")
    decode.add_argument("--summary-csv")
    decode.add_argument("--timeline")
    decode.add_argument("--json")
    decode.set_defaults(func=command_decode)
    diff = sub.add_parser(
        "compare", help="compare stage aggregates from good and bad raw files"
    )
    diff.add_argument("--good", nargs="+", required=True)
    diff.add_argument("--bad", nargs="+", required=True)
    diff.add_argument("--csv")
    diff.add_argument("--json")
    diff.set_defaults(func=command_compare)
    test = sub.add_parser("synthetic-test", help="run protocol/parser synthetic tests")
    test.set_defaults(
        func=lambda _args: 0
        if unittest.main(argv=[__file__], exit=False).result.wasSuccessful()
        else 1
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
