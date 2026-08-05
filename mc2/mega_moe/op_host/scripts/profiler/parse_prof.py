#!/usr/bin/env python3
"""Parse prof_rank*.bin to prof.csv + prof_paired.csv.

Usage:
    python3 parse_prof.py [options] <prof_rank0.bin> [prof_rank1.bin ...]

Options:
    --n-cores N           Total cores (0=auto-detect from file size)
    --aiv-cores N         AIV core count (0=auto from known configs)
    --pair-map FILE       (Required) JSON file: {"begin_id": end_id, ...} for explicit pairing
    --event-names-from F  Parse profiler.h enum ProfEventId for readable names
    --config FILE         Path to generated profiler_config.h to read buffer constants
    --slots N             PROF_SLOTS_PER_CORE (default: 4096)
    --rec-bytes N         PROF_RECORD_BYTES (default: 16)
    --align N             PROF_RINGIDX_ALIGN (default: 64)
"""

from __future__ import annotations

import sys
if sys.version_info < (3, 8):
    print(f"[ERROR] Python 3.8+ required, got {sys.version}", file=sys.stderr)
    sys.exit(1)

import numpy as np, csv, sys, os, argparse, json, re
from collections import deque

# Auto-detection heuristic — maps total core count to (aiv_cores, layout_label).
# NOT authoritative: actual core counts vary by chip model and layout.
# Always verify with `host_check_cores` at build time and pass --aiv-cores if needed.
KNOWN_CONFIGS = {
    24: (24, "AIV_ONLY"),       # 24-core AIV-only config
    56: (28, "MIX_AIC_1_1"),    # Ascend950PR PCIE (28 AIV + 28 AIC)
    64: (32, "MIX_AIC_1_1"),    # Ascend950PR Server (32 AIV + 32 AIC)
    84: (56, "MIX_AIC_1_2"),    # Ascend950PR MIX_1_2 (56 AIV + 28 AIC)
    48: (24, "MIX_AIC_1_1"),  # 48-core MIX_1_1 (24 AIV + 24 AIC). For AIV-only (48 AIV), pass --aiv-cores 48
    # Authoritative table: npu-arch skill → npu-hardware-params.md
    # Update this dict when new chip SKUs are added.
}


def detect_n_cores(path: str, slots: int, rec: int, align: int) -> tuple[int, int]:
    """Auto-detect total core count from binary file size. Returns (n_cores, hdr_layout)."""
    try:
        sz = os.path.getsize(path)
    except OSError as e:
        print("[ERROR] Cannot access file %s: %s" % (path, e))
        sys.exit(1)
    # Check for binary header (magic = 0x50524F46 = "PROF")
    try:
        with open(path, 'rb') as f:
            hdr = f.read(16)
        if len(hdr) >= 16:
            import struct
            magic, hdr_ver, hdr_ncores, hdr_rec = struct.unpack('<IIII', hdr[:16])
            if magic == 0x50524F46:
                if hdr_rec > 0 and hdr_rec != rec:
                    print(f"[WARN] Binary header says rec={hdr_rec} but using --rec-bytes={rec}. "
                          f"Results may be incorrect.", file=sys.stderr)
                # Version field encodes layout_type in upper 16 bits (since v1.0.0)
                # layout_type: 0=unknown, 1=AIV_ONLY, 2=MIX_1_1, 3=MIX_1_2
                hdr_layout = (hdr_ver >> 16) & 0xFFFF
                hdr_ver = hdr_ver & 0xFFFF  # actual version number
                return hdr_ncores, hdr_layout
    except OSError:
        pass
    per_core = align + slots * rec
    if sz < per_core:
        print("[ERROR] File %s too small (%d bytes, need >= %d for at least 1 core)" % (path, sz, per_core))
        sys.exit(1)
    # Try with 16-byte header first, then without
    for hdr_size in [16, 0]:
        data_sz = sz - hdr_size
        if data_sz > 0 and data_sz % per_core == 0:
            n = data_sz // per_core
            break
    else:
        n = sz // per_core  # last resort
    if n * per_core != sz and (sz - 16) % per_core != 0:
        print("[WARN] File size %d not divisible by per-core size %d, guessing N_CORES=%d" % (sz, per_core, n))
    return n, 0


def resolve_aiv_cores(n_cores: int, user_aiv: int) -> int:
    """Resolve AIV core count from user override or known chip configs."""
    if user_aiv > 0:
        return user_aiv
    if n_cores in KNOWN_CONFIGS:
        aiv, desc = KNOWN_CONFIGS[n_cores]
        print("[INFO] Auto-detected config: %s" % desc)
        return aiv
    aiv = n_cores // 2
    print("[WARN] Unknown core config %d, assuming %d AIV cores. Use --aiv-cores to override." % (n_cores, aiv))
    return aiv


def parse_event_names_from_header(header_path: str) -> dict[int, str]:
    """Parse EVT_* event name constants from a C/C++ header file."""
    names: dict[int, str] = {}
    with open(header_path) as f:
        text = f.read()
    for m in re.finditer(r'(EVT_\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+)', text):
        name = m.group(1)
        val_str = m.group(2)
        val = int(val_str, 16) if val_str.startswith('0x') else int(val_str)
        names[val] = name
    return names


def read_config_constants(config_path: str) -> tuple[int, int, int]:
    """Parse PROF_SLOTS_PER_CORE, PROF_RECORD_BYTES, PROF_RINGIDX_ALIGN from profiler_config.h."""
    slots, rec, align = 4096, 16, 64
    with open(config_path) as f:
        for line in f:
            m = re.match(r'#define\s+PROF_SLOTS_PER_CORE\s+(\d+)', line)
            if m: slots = int(m.group(1))
            m = re.match(r'#define\s+PROF_RECORD_BYTES\s+(\d+)', line)
            if m: rec = int(m.group(1))
            m = re.match(r'#define\s+PROF_RINGIDX_ALIGN\s+(\d+)', line)
            if m: align = int(m.group(1))
    return slots, rec, align


def parse_one(path: str, n_cores: int, aiv_cores: int, slots: int, rec: int, align: int,
              event_names: dict[int, str] | None, hdr_offset: int = 0) -> list[dict]:
    """Parse a single prof_rank*.bin binary file into a list of event row dicts."""
    with open(path, 'rb') as f:
        f.seek(hdr_offset)
        raw = f.read()
    ringidx_region = n_cores * align

    try:
        ring_counts = np.frombuffer(raw[:ringidx_region], dtype='<u4')[::(align // 4)]
    except ValueError:
        print("[ERROR] Binary file too small for ring index region (%d bytes needed)" % ringidx_region)
        sys.exit(1)

    try:
        arr = np.frombuffer(raw[ringidx_region:], dtype=np.dtype([
            ('eid', '<u4'), ('payload', '<u4'), ('cycle', '<u8')
        ])).reshape(n_cores, slots)
    except ValueError:
        print("[ERROR] Binary data region size mismatch (expected %d bytes)" % (n_cores * slots * rec))
        sys.exit(1)

    overflow_cores: list[int] = []
    fallback_cores: list[int] = []
    rows: list[dict] = []
    for cid in range(n_cores):
        cnt = int(ring_counts[cid])
        # Fallback: when ring index reads as 0 but data exists (DCCI not flushed
        # before host D2H), scan for contiguous non-zero event IDs.
        if cnt == 0:
            scan_cnt = 0
            for si in range(slots):
                if int(arr[cid][si]['eid']) != 0:
                    scan_cnt = si + 1
                else:
                    break
            if scan_cnt > 0:
                cnt = scan_cnt
                fallback_cores.append(cid)
        has_sentinel = (cnt >= slots and int(arr[cid][slots - 1]['eid']) == 0xFFFF)
        if has_sentinel:
            overflow_cores.append(cid)
        role = "AIV" if cid < aiv_cores else "AIC"
        lid = cid if cid < aiv_cores else cid - aiv_cores
        effective_slots = slots - 1 if has_sentinel else min(cnt, slots)
        for i in range(effective_slots):
            entry = arr[cid][i]
            eid = int(entry['eid'])
            if event_names and eid in event_names:
                name = event_names[eid]
            # Fallback event names per event-id-spec.md convention:
            # 0x0001 = KERNEL_BEGIN, 0x00FF = KERNEL_END (recommended IDs for all operators).
            # These are only used when --event-names-from is not provided.
            elif eid == 0x0001:
                name = "KERNEL_BEGIN"
            elif eid == 0x00FF:
                name = "KERNEL_END"
            else:
                name = "EVT_0x%04X" % eid
            rows.append(dict(
                core_id=cid, role=role, local_id=lid,
                event_id=eid,
                event_name=name,
                payload=int(entry['payload']),
                cycle=int(entry['cycle'])))

    if overflow_cores:
        print("[WARN] Ring buffer overflow on %d core(s): %s (events >= %d, data may be truncated)" % (
            len(overflow_cores), overflow_cores[:10], slots))

    if fallback_cores:
        print("[INFO] Ring index was 0 on %d core(s), recovered via data scan (DCCI not flushed before D2H)" % (
            len(fallback_cores)))

    return rows


def parse_int(s: str) -> int:
    """Parse integer from string, supporting both decimal and hex (0x prefix)."""
    s = str(s).strip()
    return int(s, 16) if s.startswith('0x') or s.startswith('0X') else int(s)


def build_pair_map(user_pair_map: dict[str, str] | None,
                   event_names: dict[int, str] | None) -> tuple[dict[int, int], dict[int, int]]:
    """Build begin->end pair map and reverse end->begin map from user-provided JSON mapping."""
    if user_pair_map is not None:
        if not isinstance(user_pair_map, dict):
            print("[ERROR] pair_map.json must be a JSON object (dict), got %s" % type(user_pair_map).__name__)
            sys.exit(1)
    pair_map: dict[int, int] = {}
    if user_pair_map:
        try:
            pair_map = {parse_int(k): parse_int(v) for k, v in user_pair_map.items()}
        except (ValueError, TypeError) as e:
            print("[ERROR] Invalid key/value in pair_map.json: %s" % e)
            sys.exit(1)
    end_to_begin: dict[int, int] = {e: b for b, e in pair_map.items()}
    return pair_map, end_to_begin


def pair_events(rows: list[dict], pair_map: dict[int, int],
                end_to_begin: dict[int, int]) -> tuple[list[dict], int, int]:
    """Pair begin/end events by core and event ID, returning (pairs, unmatched_begin, unmatched_end)."""
    open_begin: dict[tuple, deque] = {}
    pairs: list[dict] = []
    unmatched_end = 0
    for r in sorted(rows, key=lambda x: x['cycle']):
        eid = r['event_id']
        end_eid = pair_map.get(eid)
        if end_eid is not None:
            open_begin.setdefault((r['core_id'], end_eid), deque()).append(r)
        elif eid in end_to_begin:
            key = (r['core_id'], eid)
            if key in open_begin and open_begin[key]:
                b = open_begin[key].popleft()
                begin_name = b['event_name']
                event_name = begin_name.replace('_BEGIN', '') if begin_name.endswith('_BEGIN') else begin_name
                pairs.append(dict(
                    core_id=r['core_id'], role=r['role'], local_id=r['local_id'],
                    event=event_name,
                    begin_cycle=b['cycle'], end_cycle=r['cycle'],
                    duration_cycles=r['cycle'] - b['cycle'],
                    payload=b['payload']))
            else:
                unmatched_end += 1
    unmatched_begin = sum(len(v) for v in open_begin.values())
    return pairs, unmatched_begin, unmatched_end


def main() -> None:
    """CLI entry point: parse profiler binary files to CSV with event pairing."""
    parser = argparse.ArgumentParser(description="Parse profiler binary to CSV")
    parser.add_argument('files', nargs='+', help='prof_rank*.bin files')
    parser.add_argument('--n-cores', type=int, default=0, help='Total cores (0=auto-detect)')
    parser.add_argument('--aiv-cores', type=int, default=0, help='AIV core count (0=auto from known configs)')
    parser.add_argument('--pair-map', required=True, help='JSON file with explicit {begin_id: end_id} mapping')
    parser.add_argument('--event-names-from', default=None, help='Path to profiler.h to extract event names')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to generated profiler_config.h to read buffer constants (overrides --slots/--rec-bytes/--align)')
    parser.add_argument('--slots', type=int, default=4096,
                        help='PROF_SLOTS_PER_CORE (default: 4096, must match profiler_config.h)')
    parser.add_argument('--rec-bytes', type=int, default=16,
                        help='PROF_RECORD_BYTES (default: 16, must match profiler_config.h)')
    parser.add_argument('--align', type=int, default=64,
                        help='PROF_RINGIDX_ALIGN (default: 64, must match profiler_config.h)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for prof.csv and prof_paired.csv (default: current directory)')
    args = parser.parse_args()

    # Resolve buffer constants
    if args.config:
        if not os.path.isfile(args.config):
            print("[ERROR] --config file not found: %s" % args.config)
            sys.exit(1)
        slots, rec, align = read_config_constants(args.config)
        print("[INFO] Read constants from %s: SLOTS=%d, REC=%d, ALIGN=%d" % (args.config, slots, rec, align))
    else:
        slots, rec, align = args.slots, args.rec_bytes, args.align
        print("[INFO] Buffer constants: SLOTS=%d, REC=%d, ALIGN=%d" % (slots, rec, align))

    # Validate input files exist
    for fp in args.files:
        if not os.path.isfile(fp):
            print("[ERROR] Input file not found: %s" % fp)
            sys.exit(1)

    event_names: dict[int, str] = {}
    if args.event_names_from:
        if not os.path.isfile(args.event_names_from):
            print("[ERROR] --event-names-from file not found: %s" % args.event_names_from)
            sys.exit(1)
        event_names = parse_event_names_from_header(args.event_names_from)
        print("[INFO] Loaded %d event names from %s" % (len(event_names), args.event_names_from))

    if args.n_cores:
        n_cores, hdr_layout = args.n_cores, 0
    else:
        n_cores, hdr_layout = detect_n_cores(args.files[0], slots, rec, align)

    if hdr_layout > 0:
        layout_names = {1: 'AIV-only', 2: 'MIX_1_1', 3: 'MIX_1_2'}
        print(f"[INFO] Layout from binary header: {layout_names.get(hdr_layout, 'unknown')}")

    if args.aiv_cores > 0:
        aiv_cores = args.aiv_cores
    elif hdr_layout == 1:
        aiv_cores = n_cores
    elif hdr_layout == 2:
        aiv_cores = n_cores // 2
    elif hdr_layout == 3:
        aiv_cores = (n_cores * 2) // 3
    else:
        aiv_cores = resolve_aiv_cores(n_cores, 0)
    print("[INFO] N_CORES=%d, AIV_CORES=%d, AIC_CORES=%d" % (n_cores, aiv_cores, n_cores - aiv_cores))

    try:
        with open(args.pair_map) as f:
            user_pair_map = json.load(f)
    except FileNotFoundError:
        print("[ERROR] pair-map file not found: %s" % args.pair_map); sys.exit(1)
    except json.JSONDecodeError as e:
        print("[ERROR] Invalid JSON in %s: %s" % (args.pair_map, e)); sys.exit(1)
    print("[INFO] Loaded explicit pair map: %d pairs" % len(user_pair_map))

    # Cross-validate pair_map.json against profiler.h event names
    if event_names and user_pair_map:
        pair_map_keys: set[str] = set()
        for k in user_pair_map:
            try:
                pair_map_keys.add('0x%04X' % parse_int(k))
            except (ValueError, TypeError):
                pass
        name_to_id = {name: eid for eid, name in event_names.items()}
        begin_events = {name: eid for name, eid in name_to_id.items() if name.endswith('_BEGIN')}
        end_events = {name: eid for name, eid in name_to_id.items() if name.endswith('_END')}
        for bname in sorted(begin_events):
            prefix = bname[:-6]  # strip '_BEGIN'
            ename = prefix + '_END'
            if ename in end_events:
                beid = begin_events[bname]
                eeid = end_events[ename]
                beid_hex = '0x%04X' % beid
                if beid_hex not in pair_map_keys:
                    print("[WARN] Event pair %s/%s (0x%04X/0x%04X) found in header but missing from --pair-map" % (bname, ename, beid, eeid))

    all_rows: list[dict] = []
    all_pairs: list[dict] = []
    pair_map, end_to_begin = build_pair_map(user_pair_map, event_names)
    per_core = align + slots * rec
    for file_index, fp in enumerate(args.files):
        try:
            fp_size = os.path.getsize(fp)
        except OSError as e:
            print("[WARN] Cannot access %s: %s. Skipping." % (fp, e))
            continue
        hdr_offset = 0
        with open(fp, 'rb') as hf:
            hdr = hf.read(16)
        if len(hdr) >= 16:
            import struct
            magic = struct.unpack('<I', hdr[:4])[0]
            if magic == 0x50524F46:
                hdr_offset = 16
        expected_size = n_cores * per_core + hdr_offset
        if fp_size != expected_size:
            print("[WARN] %s: size %d != expected %d (n_cores=%d). Skipping." % (fp, fp_size, expected_size, n_cores))
            continue
        m = re.search(r'prof_rank(\d+)', os.path.basename(fp))
        rk = m.group(1) if m else str(file_index)
        rows = parse_one(fp, n_cores, aiv_cores, slots, rec, align, event_names, hdr_offset)
        for r in rows:
            r['rank'] = rk
        all_rows.extend(rows)
        pairs, unmatched_begin, unmatched_end = pair_events(rows, pair_map, end_to_begin)
        print("[INFO] Pairing: %d pairs, %d unmatched begin, %d unmatched end" % (
            len(pairs), unmatched_begin, unmatched_end))
        for p in pairs:
            p['rank'] = rk
        all_pairs.extend(pairs)

    if not all_rows:
        print("[ERROR] No profiling events found in any input file.")
        sys.exit(1)

    output_dir = args.output_dir or '.'
    os.makedirs(output_dir, exist_ok=True)

    for fn, data, fields in [
        ('prof.csv', all_rows, ['rank', 'core_id', 'role', 'local_id',
         'event_id', 'event_name', 'payload', 'cycle']),
        ('prof_paired.csv', all_pairs, ['rank', 'core_id', 'role', 'local_id',
         'event', 'begin_cycle', 'end_cycle', 'duration_cycles', 'payload'])]:
        try:
            out_path = os.path.join(output_dir, fn)
            with open(out_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(data)
            print("[INFO] %s: %d rows" % (out_path, len(data)))
        except OSError as e:
            print("[ERROR] Failed to write %s: %s" % (out_path, e))
            sys.exit(1)

    print("[INFO] Paired %d event types" % len(pair_map))


if __name__ == '__main__':
    main()
