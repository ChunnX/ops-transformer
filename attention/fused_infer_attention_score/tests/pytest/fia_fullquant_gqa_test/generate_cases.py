#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------
"""Generate comprehensive test cases for fia_fullquant_gqa.

Usage:
    python generate_cases.py            # write cases.json
    python generate_cases.py --dry-run  # print count only

Coverage dimensions:
    Phase 1 — Generalize B (Batch):
        Fixed N_q, N_kv, seq_q=128, seq_kv=256.
        G = N_q/N_kv in {1..64}, G=8 gets the most thorough B sweep.

    Phase 2 — Generalize N_q and N_kv:
        Fixed seq_q=128, seq_kv=256.  B swept at 3 density levels.

    Phase 3 — Cross-parameter sweep (seq_q, seq_kv, B, N_q, N_kv):
        qs in [1, 64K], kvs in [0, 256K].
        Multiple (B, N_q, N_kv) configs tested against curated seq pairs.
"""

import argparse
import json
import math
import os
import random
import re


# ==============================================================================
# Test case builder helpers
# ==============================================================================
CASES = []
_PHASE_COUNTS = {"p1": 0, "p2": 0, "p3": 0}


def _add(name, B, N_q, N_kv, actual_seq_q, actual_seq_kv, phase="", **kwargs):
    """Add a single test case.  Skips duplicates by name."""
    for existing in CASES:
        if existing["name"] == name:
            return
    case = {
        "name": name,
        "B": int(B),
        "N_q": int(N_q),
        "N_kv": int(N_kv),
        "D": 128,
        "actual_seq_q": [int(x) for x in actual_seq_q],
        "actual_seq_kv": [int(x) for x in actual_seq_kv],
        "enable_pa": kwargs.get("enable_pa", True),
        "block_size": kwargs.get("block_size", 128),
        "sparse_mode": kwargs.get("sparse_mode", 0),
        "kv_cache_layout": kwargs.get("kv_cache_layout", "BnNBsD"),
        "p_scale": kwargs.get("p_scale", 1.0),
    }
    CASES.append(case)
    if phase:
        _PHASE_COUNTS[phase] = _PHASE_COUNTS.get(phase, 0) + 1


def _name(sq, skv, n_q, n_kv, b, g):
    """Build a compact case name.

    Format: PA_NZ_S{skv}_Q{sq}_G{g}_Nq{nq}_Nkv{nkv}_B{b}
    """
    return f"PA_NZ_S{skv}_Q{sq}_G{g}_Nq{n_q}_Nkv{n_kv}_B{b}"


# ==============================================================================
# Phase 1: Generalize B (Batch)
#   Fixed: N_q, N_kv, actual_seq_q=128, actual_seq_kv=256
# ==============================================================================
def _build_phase1():
    SQ, SKV = 128, 256

    # ---- G=1 (MHA) ----
    for b in [1, 4, 16, 64, 128, 256, 1024]:
        _add(_name(SQ, SKV, 16, 16, b, 1), b, 16, 16, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 32, 32, b, 1), b, 32, 32, [SQ] * b, [SKV] * b, "p1")

    # ---- G=2 ----
    for b in [1, 4, 16, 64, 128, 256]:
        _add(_name(SQ, SKV, 16, 8, b, 2), b, 16, 8, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 32, 16, b, 2), b, 32, 16, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 64, 32, b, 2), b, 64, 32, [SQ] * b, [SKV] * b, "p1")

    # ---- G=3 ----
    for b in [1, 4, 16, 64, 128]:
        _add(_name(SQ, SKV, 12, 4, b, 3), b, 12, 4, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 24, 8, b, 3), b, 24, 8, [SQ] * b, [SKV] * b, "p1")

    # ---- G=4 ----
    for b in [1, 4, 16, 64, 128, 256]:
        _add(_name(SQ, SKV, 16, 4, b, 4), b, 16, 4, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 32, 8, b, 4), b, 32, 8, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 64, 16, b, 4), b, 64, 16, [SQ] * b, [SKV] * b, "p1")

    # ---- G=5 ----
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 20, 4, b, 5), b, 20, 4, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 40, 8, b, 5), b, 40, 8, [SQ] * b, [SKV] * b, "p1")

    # ---- G=6 ----
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 24, 4, b, 6), b, 24, 4, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 48, 8, b, 6), b, 48, 8, [SQ] * b, [SKV] * b, "p1")

    # ---- G=7 ----
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 28, 4, b, 7), b, 28, 4, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 56, 8, b, 7), b, 56, 8, [SQ] * b, [SKV] * b, "p1")

    # ---- G=8 (typical, most thorough B sweep) ----
    B_FULL_SWEEP = [
        1, 2, 3, 4, 5, 7, 8, 15, 16, 17,
        31, 32, 33, 63, 64, 65,
        127, 128, 129, 255, 256, 257,
        511, 512, 513, 1023, 1024, 1025,
    ]
    for b in B_FULL_SWEEP:
        _add(_name(SQ, SKV, 16, 2, b, 8), b, 16, 2, [SQ] * b, [SKV] * b, "p1")

    # G=8 with larger head counts
    for b in [1, 4, 16, 64, 128, 256]:
        _add(_name(SQ, SKV, 32, 4, b, 8), b, 32, 4, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16, 64, 128]:
        _add(_name(SQ, SKV, 64, 8, b, 8), b, 64, 8, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 128, 16, b, 8), b, 128, 16, [SQ] * b, [SKV] * b, "p1")

    # G=8 with irregular N_kv
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 24, 3, b, 8), b, 24, 3, [SQ] * b, [SKV] * b, "p1")
        _add(_name(SQ, SKV, 40, 5, b, 8), b, 40, 5, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 56, 7, b, 8), b, 56, 7, [SQ] * b, [SKV] * b, "p1")

    # ---- G=9 ----
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 36, 4, b, 9), b, 36, 4, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 72, 8, b, 9), b, 72, 8, [SQ] * b, [SKV] * b, "p1")

    # ---- G=12 ----
    for b in [1, 4, 16, 64, 128]:
        _add(_name(SQ, SKV, 48, 4, b, 12), b, 48, 4, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 96, 8, b, 12), b, 96, 8, [SQ] * b, [SKV] * b, "p1")

    # ---- G=16 ----
    for b in [1, 4, 16, 64, 128, 256]:
        _add(_name(SQ, SKV, 16, 1, b, 16), b, 16, 1, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16, 64, 128]:
        _add(_name(SQ, SKV, 32, 2, b, 16), b, 32, 2, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 64, 4, b, 16), b, 64, 4, [SQ] * b, [SKV] * b, "p1")

    # ---- G=24 ----
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 24, 1, b, 24), b, 24, 1, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 48, 2, b, 24), b, 48, 2, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4]:
        _add(_name(SQ, SKV, 96, 4, b, 24), b, 96, 4, [SQ] * b, [SKV] * b, "p1")

    # ---- G=32 ----
    for b in [1, 4, 16, 64, 128]:
        _add(_name(SQ, SKV, 32, 1, b, 32), b, 32, 1, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 64, 2, b, 32), b, 64, 2, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 128, 4, b, 32), b, 128, 4, [SQ] * b, [SKV] * b, "p1")

    # ---- G=48 ----
    for b in [1, 4, 16, 64]:
        _add(_name(SQ, SKV, 48, 1, b, 48), b, 48, 1, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 96, 2, b, 48), b, 96, 2, [SQ] * b, [SKV] * b, "p1")

    # ---- G=64 ----
    for b in [1, 4, 16, 64, 128]:
        _add(_name(SQ, SKV, 64, 1, b, 64), b, 64, 1, [SQ] * b, [SKV] * b, "p1")
    for b in [1, 4, 16]:
        _add(_name(SQ, SKV, 128, 2, b, 64), b, 128, 2, [SQ] * b, [SKV] * b, "p1")


# ==============================================================================
# Phase 2: Generalize N_q and N_kv  (N_q > N_kv, i.e. G >= 2)
#   Fixed: actual_seq_q=128, actual_seq_kv=256
# ==============================================================================
def _build_phase2():
    SQ, SKV = 128, 256
    B_LARGE = [1]
    B_MEDIUM = [4, 16]
    B_SPARSE = [32, 64, 65, 128]

    def _add_p2(b, n_q, n_kv):
        g = n_q // n_kv
        _add(_name(SQ, SKV, n_q, n_kv, b, g), b, n_q, n_kv,
             [SQ] * b, [SKV] * b, "p2")

    def _by_b(n_q, n_kv, large=True, medium=True, sparse=True):
        if large:
            for b in B_LARGE:
                _add_p2(b, n_q, n_kv)
        if medium:
            for b in B_MEDIUM:
                _add_p2(b, n_q, n_kv)
        if sparse:
            for b in B_SPARSE:
                _add_p2(b, n_q, n_kv)

    # ---- G=1 (MHA: N_q == N_kv) ----
    for nkv in [1, 2, 4, 8, 16, 32, 64, 128]:
        _by_b(nkv, nkv, large=True, medium=(nkv <= 64),
              sparse=(nkv in [1, 8, 16, 32, 64]))

    # ---- G=2 ----
    for nkv in [1, 2, 4, 8, 16, 32, 64]:
        _by_b(2 * nkv, nkv, large=True, medium=(nkv <= 32),
              sparse=(nkv in [1, 2, 8, 16, 32]))

    # ---- G=3 ----
    for nkv in [1, 2, 4, 8, 16, 32]:
        _by_b(3 * nkv, nkv, large=True, medium=(nkv <= 16),
              sparse=(nkv in [1, 4, 8]))

    # ---- G=4 ----
    for nkv in [1, 2, 4, 8, 16, 32]:
        _by_b(4 * nkv, nkv, large=True, medium=(nkv <= 16),
              sparse=(nkv in [1, 2, 4, 8]))

    # ---- G=5 ----
    for nkv in [1, 2, 4, 8, 16]:
        _by_b(5 * nkv, nkv, large=True, medium=(nkv <= 8),
              sparse=(nkv in [1, 4, 8]))

    # ---- G=6 ----
    for nkv in [1, 2, 4, 8, 16]:
        _by_b(6 * nkv, nkv, large=True, medium=(nkv <= 8),
              sparse=(nkv in [1, 4, 8]))

    # ---- G=7 ----
    for nkv in [1, 2, 4, 8, 16]:
        _by_b(7 * nkv, nkv, large=True, medium=(nkv <= 8),
              sparse=(nkv in [1, 4, 8]))

    # ---- G=8 (typical, most comprehensive) ----
    for nkv in [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 32, 64]:
        _by_b(8 * nkv, nkv, large=True,
              medium=(nkv <= 16),
              sparse=(nkv in [1, 2, 4, 8]))
    for nkv in [17, 19, 23, 29, 31]:
        _add_p2(1, 8 * nkv, nkv)

    # ---- G=9 ----
    for nkv in [1, 2, 4, 8]:
        _by_b(9 * nkv, nkv, large=True, medium=(nkv <= 4),
              sparse=(nkv in [1, 4]))

    # ---- G=12 ----
    for nkv in [1, 2, 4, 8]:
        _by_b(12 * nkv, nkv, large=True, medium=(nkv <= 4),
              sparse=(nkv in [1, 4, 8]))

    # ---- G=16 ----
    for nkv in [1, 2, 4, 8]:
        _by_b(16 * nkv, nkv, large=True, medium=(nkv <= 4),
              sparse=(nkv in [1, 4, 8]))

    # ---- G=24 ----
    for nkv in [1, 2, 4, 8]:
        _by_b(24 * nkv, nkv, large=True, medium=(nkv <= 4),
              sparse=(nkv in [1, 4]))

    # ---- G=32 ----
    for nkv in [1, 2, 4, 8]:
        _by_b(32 * nkv, nkv, large=True, medium=(nkv <= 4),
              sparse=(nkv in [1, 4]))

    # ---- G=48 ----
    for nkv in [1, 2, 4]:
        _by_b(48 * nkv, nkv, large=True, medium=(nkv <= 2),
              sparse=(nkv in [1, 2]))

    # ---- G=64 ----
    for nkv in [1, 2, 4]:
        _by_b(64 * nkv, nkv, large=True, medium=(nkv <= 2),
              sparse=(nkv in [1, 2]))


# ==============================================================================
# Phase 3: Cross-parameter sweep — B, N_q, N_kv, seq_q, seq_kv all vary.
#   qs in [1, 64K],  kvs in [0, 256K]
#
#   Seq values are generated randomly (fixed seed for reproducibility) so
#   they aren't all neat powers of 2 — they span the full range with
#   realistic, irregular values.
#
#   Strategy:
#     - Baseline config (B=1, G=8, N_q=16, N_kv=2): full random seq-pair sweep.
#     - Additional configs (other B, G, N_q, N_kv): core random subset.
#     - Batch variants: B>1 with randomly varying per-batch sequence lengths.
# ==============================================================================
_SEED = 42


def _rand_seq_pairs(rng, n, edge=True):
    """Generate *n* diverse (qs, kvs) pairs covering the full ranges.

    Mixes several generation strategies (decode-like, prefill-like,
    equal-ish, long-context, purely random) so the distribution is
    realistic.  Values are seeded and *not* constrained to powers of 2.
    """
    pairs = set()

    # Always include edge / boundary values for safety
    if edge:
        for qs in [1, 2, 65535, 65536]:
            for kvs in [0, 1, 2, 262143, 262144]:
                pairs.add((qs, kvs))
        for qs in [1, 2, 3, 65534, 65535, 65536]:
            for kvs in [128, 129, 255, 256, 257, 511, 512, 513]:
                pairs.add((qs, kvs))

    def _rand(lo, hi):
        """Random int in [lo, hi], mix of uniform and log-uniform."""
        if lo >= hi:
            return lo
        if lo < 1:
            lo = 1
        if rng.random() < 0.5:
            v = int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))
        else:
            v = rng.randint(lo, hi)
        return max(lo, min(hi, v))

    while len(pairs) < n:
        r = rng.random()
        if r < 0.30:
            qs = _rand(1, 32)
            kvs = _rand(0, 262144)
        elif r < 0.55:
            qs = _rand(64, 65536)
            kvs = _rand(0, max(256, qs))
        elif r < 0.70:
            qs = _rand(1, 65536)
            delta = rng.randint(-max(1, qs // 8), max(1, qs // 8))
            kvs = max(0, min(262144, qs + delta))
        elif r < 0.85:
            qs = _rand(1, 4096)
            kvs = _rand(max(qs + 1, 256), 262144)
        else:
            qs = _rand(1, 65536)
            kvs = _rand(0, 262144)
        pairs.add((qs, kvs))

    return list(pairs)[:n]


def _rand_seq_list(rng, b, lo, hi):
    """Generate a list of *b* random sequence lengths in [lo, hi]."""
    return [max(lo, min(hi, rng.randint(lo, hi))) for _ in range(b)]


def _build_phase3():
    rng = random.Random(_SEED)

    # ------------------------------------------------------------------
    # Baseline config — full random sweep
    # ------------------------------------------------------------------
    BASELINE = (1, 16, 2)   # B=1, G=8
    B_bl, Nq_bl, Nkv_bl = BASELINE
    G_bl = Nq_bl // Nkv_bl

    FULL_SEQ_PAIRS = _rand_seq_pairs(rng, 150, edge=True)

    for qs, kvs in FULL_SEQ_PAIRS:
        _add(_name(qs, kvs, Nq_bl, Nkv_bl, B_bl, G_bl),
             B_bl, Nq_bl, Nkv_bl, [qs], [kvs], "p3")

    # ------------------------------------------------------------------
    # Cross-parameter configs  (B, N_q, N_kv)
    # ------------------------------------------------------------------
    CROSS_CONFIGS = [
        (1, 16, 16),    # G=1  MHA
        (1, 16, 4),     # G=4
        (1, 16, 1),     # G=16
        (1, 32, 4),     # G=8, larger heads
        (1, 64, 8),     # G=8, even larger
        (1, 32, 1),     # G=32
        (1, 64, 1),     # G=64
        (2, 16, 2),     # G=8, B=2
        (3, 16, 2),     # G=8, B=3
        (4, 16, 2),     # G=8, B=4
        (4, 16, 4),     # G=4, B=4
        (4, 16, 1),     # G=16, B=4
        (8, 16, 2),     # G=8, B=8
        (8, 16, 4),     # G=4, B=8
        (16, 16, 2),    # G=8, B=16
        (16, 16, 4),    # G=4, B=16
        (32, 16, 2),    # G=8, B=32
        (65, 16, 2),    # G=8, B=65
    ]

    CORE_SEQ_PAIRS = _rand_seq_pairs(rng, 55, edge=True)

    for B, N_q, N_kv in CROSS_CONFIGS:
        G = N_q // N_kv
        for qs, kvs in CORE_SEQ_PAIRS:
            _add(_name(qs, kvs, N_q, N_kv, B, G),
                 B, N_q, N_kv, [qs] * B, [kvs] * B, "p3")

    # ------------------------------------------------------------------
    # Batch variants — randomly varying per-batch seq lengths
    # ------------------------------------------------------------------
    def _a3b(b, n_q, n_kv, qs_list, kvs_list):
        max_qs = max(qs_list)
        max_kvs = max(kvs_list)
        g = n_q // n_kv
        _add(_name(max_qs, max_kvs, n_q, n_kv, b, g),
             b, n_q, n_kv, qs_list, kvs_list, "p3")

    # G=8 random batch variants
    for _ in range(6):
        b = rng.choice([3, 4, 7, 8, 15, 16, 31, 32])
        qs_list = _rand_seq_list(rng, b, 1, 65536)
        kvs_list = _rand_seq_list(rng, b, 0, 262144)
        _a3b(b, 16, 2, qs_list, kvs_list)

    # G=4 random batch variants
    for _ in range(3):
        b = rng.choice([4, 8, 16])
        qs_list = _rand_seq_list(rng, b, 1, 65536)
        kvs_list = _rand_seq_list(rng, b, 0, 262144)
        _a3b(b, 16, 4, qs_list, kvs_list)

    # G=1 random batch variants
    for _ in range(2):
        b = rng.choice([4, 8])
        qs_list = _rand_seq_list(rng, b, 1, 65536)
        kvs_list = _rand_seq_list(rng, b, 0, 262144)
        _a3b(b, 16, 16, qs_list, kvs_list)

    # G=16 random batch variants
    for _ in range(2):
        b = rng.choice([4, 8])
        qs_list = _rand_seq_list(rng, b, 1, 65536)
        kvs_list = _rand_seq_list(rng, b, 0, 262144)
        _a3b(b, 16, 1, qs_list, kvs_list)


# ==============================================================================
# Build & write JSON
# ==============================================================================
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "cases.json")


def _build_all():
    """Run all phase builders and return the assembled CASES list."""
    global CASES, _PHASE_COUNTS
    CASES = []
    _PHASE_COUNTS = {"p1": 0, "p2": 0, "p3": 0}
    _build_phase1()
    _build_phase2()
    _build_phase3()
    # Dedup by name
    seen = set()
    unique = []
    skipped = 0
    for c in CASES:
        if c["name"] not in seen:
            seen.add(c["name"])
            unique.append(c)
        else:
            skipped += 1
    if skipped:
        print(f"  (skipped {skipped} duplicate case names)")
    CASES = unique
    return CASES


def _compact_int_arrays(json_str):
    """Post-process JSON: collapse multi-line integer arrays to single line."""
    lines = json_str.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.endswith('[') and '":' in stripped:
            j = i + 1
            items = []
            while j < len(lines):
                s = lines[j].strip()
                if s in (']', '],'):
                    break
                items.append(s.rstrip(','))
                j += 1
            if items and all(re.match(r'^-?\d+$', it.strip()) for it in items):
                indent = ' ' * (len(line) - len(line.lstrip()))
                key = stripped.split('"')[1]
                nums = ', '.join(it.strip() for it in items)
                close = lines[j].strip()
                trailing = ',' if close == '],' else ''
                out.append(f'{indent}"{key}": [{nums}]{trailing}')
                i = j + 1
                continue
        out.append(line)
        i += 1
    return '\n'.join(out)


def write_json(target_path=None):
    """Build cases and write to a JSON file."""
    if target_path is None:
        target_path = OUTPUT_FILE

    cases = _build_all()

    raw = json.dumps(cases, indent=2, ensure_ascii=False)
    compact = _compact_int_arrays(raw)

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(compact)
        f.write('\n')

    print(f"Generated {len(cases)} test cases -> {target_path}")
    print(f"  Phase 1 (B generalization):   {_PHASE_COUNTS['p1']}")
    print(f"  Phase 2 (N_q / N_kv):         {_PHASE_COUNTS['p2']}")
    print(f"  Phase 3 (cross-parameter):    {_PHASE_COUNTS['p3']}")
    print(f"  Total:                         {len(cases)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test cases for fia_fullquant_gqa")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print case count without writing")
    parser.add_argument("--output", default=None,
                        help="Path to output JSON file (default: cases.json)")
    args = parser.parse_args()

    if args.dry_run:
        cases = _build_all()
        print(f"Would generate {len(cases)} test cases (dry run)")
        if cases:
            print(f"  First: {cases[0]['name']}")
            print(f"  Last:  {cases[-1]['name']}")
        return

    write_json(args.output)


if __name__ == "__main__":
    main()
