# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tile protocol/UB layout CPU models; no simulation of Reg instructions, DMA or hardware SQ progress."""

from pathlib import Path
import random
import re
import unittest

from test_dispatch_count_histograms import count_contributions, merge_contributions, oracle, tile_ranges, tile_tokens


SENDERS = 32
ROOT = Path(__file__).resolve().parents[3]


def align(value, size):
    return (value + size - 1) // size * size


def scan_counts(counts, peers):
    """Mirror eight-column vector scan and its 128-row carry, independent of the serial oracle."""
    width = align(peers, 8)
    prefix = [[0] * width for _ in range(len(counts) + 1)]
    for peer in range(0, peers, 8):
        carry = [0] * 8
        for first in range(0, len(counts), 128):
            block = [(row + [0] * (width - peers))[peer:peer + 8] for row in counts[first:first + 128]]
            step = 1
            while step < len(block):
                previous = block
                block = [row[:] if i < step else [a + b for a, b in zip(row, previous[i - step])]
                         for i, row in enumerate(previous)]
                step *= 2
            block = [[a + b for a, b in zip(row, carry)] for row in block]
            for i, row in enumerate(block):
                prefix[first + i + 1][peer:peer + 8] = row
            carry = block[-1]
    return [row[:peers] for row in prefix]


def workspace_regions(bs, topk, peers, experts, cores, metadata=512):
    tiles = len(tile_ranges(bs, cores, topk, metadata))
    rank_stride, expert_stride = align(peers * 4, 512), align(experts * 4, 512)
    sizes = (tiles * rank_stride, rank_stride + expert_stride, (tiles + 1) * rank_stride,
             cores * expert_stride, align(tiles * 32, 512), align(bs * topk * 2, 512),
             peers * align(bs * 4, 512))
    offsets, cursor = [], 0
    for size in sizes:
        offsets.append((cursor, cursor + size))
        cursor += size
    return offsets, cursor + 16 * 1024 * 1024 + 32


class DispatchTilePipelineTest(unittest.TestCase):
    def test_parallel_scan_and_cross_batch_carry(self):
        rng = random.Random(129)
        for peers in (2, 8, 9, 33, 1024):
            for tiles in (1, 2, 7, 127, 128, 129, 257):
                counts = [[rng.randrange(33) for _ in range(peers)] for _ in range(tiles)]
                actual = scan_counts(counts, peers)
                expected = [[0] * peers]
                for row in counts:
                    expected.append([a + b for a, b in zip(expected[-1], row)])
                self.assertEqual(actual, expected)

    def prepare_counts(self, routes, peers, experts, cores, cached):
        expected_selection, expected_counts = oracle(routes, peers, experts, cores)
        slots, next_slot = None, [0] * peers
        if cached:
            slots = []
            for row in expected_selection:
                slots.append([next_slot[peer] if peer >= 0 else -1 for peer in row])
                for peer in row:
                    if peer >= 0:
                        next_slot[peer] += 1
        selected, partials = count_contributions(routes, peers, experts, cores, slots)
        actual_counts = merge_contributions(partials, len(expected_counts[0]), peers, experts)
        self.assertEqual(selected, expected_selection)
        self.assertEqual(actual_counts, expected_counts)
        return selected, actual_counts

    def check_pipeline(self, bs, cores, peers, experts, topk, rng, cached=False):
        routes = [[rng.randrange(-1, experts) for _ in range(topk)] for _ in range(bs)]
        selected, (counts, rank_totals, _) = self.prepare_counts(routes, peers, experts, cores, cached)
        tiles = tile_ranges(bs, cores, topk)
        prefix = scan_counts(counts, peers)
        source = [[None] * total for total in rank_totals]
        consumed = [[0] * total for total in rank_totals]
        ready, next_tile = set(), list(range(SENDERS))
        flags = [[0] * peers for _ in range(SENDERS)]
        producer_visits = [0] * len(tiles)
        # Non-sender owners retain a deterministic, disjoint work list; completion order may vary.
        pending = [list(range(owner, len(tiles), cores - SENDERS)) for owner in range(cores - SENDERS)]
        while any(pending):
            owner = rng.choice([i for i, queue in enumerate(pending) if queue])
            tile = pending[owner].pop(0)
            producer_visits[tile] += 1
            begin, end = tiles[tile]
            for peer in range(peers):
                indices = [token for token in range(begin, end) if peer in selected[token]]
                self.assertEqual(len(indices), prefix[tile + 1][peer] - prefix[tile][peer])
                source[peer][prefix[tile][peer]:prefix[tile + 1][peer]] = indices
            ready.add(tile)
            for sender in range(SENDERS):
                while next_tile[sender] in ready:
                    current = next_tile[sender]
                    for peer in range(peers):
                        for slot in range(prefix[current][peer], prefix[current + 1][peer]):
                            self.assertIsNotNone(source[peer][slot])
                            consumed[peer][slot] += 1
                    next_tile[sender] += SENDERS
                if next_tile[sender] >= len(tiles):
                    flags[sender] = [1] * peers
        self.assertEqual(producer_visits, [1] * len(tiles))
        self.assertTrue(all(all(row) for row in flags))
        for peer in range(peers):
            self.assertEqual(source[peer], [token for token, row in enumerate(selected) if peer in row])
            self.assertEqual(consumed[peer], [1] * rank_totals[peer])

    def test_ready_tiles_preserve_slots_under_out_of_order_producers(self):
        rng = random.Random(4096)
        for cores in (33, 48, 56, 64, 100):
            for bs, peers, experts, topk in ((1, 2, 2, 1), (31, 8, 256, 6), (129, 33, 264, 7)):
                self.check_pipeline(bs, cores, peers, experts, topk, rng)
        self.check_pipeline(4096, 56, 8, 256, 6, rng)

    def test_valid_cached_tiles_reach_sender_without_changing_slots(self):
        # Cache comes from a previous uncached dispatch; arbitrary positive slots are not valid cache inputs.
        rng = random.Random(32)
        for cores in (33, 48, 56, 64, 100):
            for bs, peers, experts, topk in ((1, 2, 2, 1), (57, 3, 96, 32), (129, 33, 264, 7)):
                self.check_pipeline(bs, cores, peers, experts, topk, rng, cached=True)
        self.check_pipeline(4096, 56, 8, 256, 6, rng, cached=True)

    def test_first_tile_can_build_wqes_before_other_tiles_ready(self):
        tiles = tile_ranges(4096, 56, 6)
        self.assertEqual(len(tiles), 128)
        ready = {0}
        self.assertIn(0, ready)
        self.assertNotIn(32, ready)  # Constructing tile 0 WQEs needs no later tile; submission depends on capacity.
        self.assertLess(len(ready), len(tiles))

    def test_metadata_scatter_fp8_weights_and_scratch_bounds(self):
        for topk in (1, 6, 7, 8, 9, 16, 32):
            for scales in (0, 1, 128, 256):
                header = align(scales, 32) // 4
                k_align = align(topk * 4, 32) // 4
                metadata_bytes = align((header + 2 * k_align) * 4 + 32, 512)
                tokens = tile_tokens(4096, 56, topk, metadata_bytes)
                count, stride = tokens * topk, metadata_bytes // 4
                raw = list(range(count))
                metadata = [None] * (tokens * stride)
                offsets = [(i + (i // topk) * (stride - topk)) * 4 for i in range(count)]
                self.assertLess(max(offsets), 32768)  # int16 address construction does not wrap.
                self.assertLessEqual(2 * align(count * 2, 256), 16384)
                self.assertLessEqual(16384 + count * 4, 32768)
                self.assertLessEqual(count * 4, 32768)  # Raw TopK cannot overlap divisor at UB+32 KiB.
                for value, offset in zip(raw, offsets):
                    metadata[header + offset // 4] = value
                for token in range(tokens):
                    base = token * stride
                    self.assertEqual(metadata[base + header:base + header + topk], raw[token * topk:(token + 1) * topk])
                    metadata[base + header + 2 * k_align:base + header + 2 * k_align + 2] = [3, token]
                    for enabled in (False, True):
                        if enabled:
                            metadata[base + header + k_align:base + header + k_align + topk] = raw[token * topk:(token + 1) * topk]
                        self.assertEqual(metadata[base + header + 2 * k_align:base + header + 2 * k_align + 2], [3, token])

    def test_workspace_regions_and_partial_rows_do_not_alias(self):
        for bs in (1, 57, 4096, 65537):
            for cores in (33, 56, 100):
                for peers, experts in ((2, 2), (3, 96), (8, 256), (33, 264), (1024, 2048)):
                    regions, size = workspace_regions(bs, 6, peers, experts, cores)
                    self.assertEqual(regions[0][0], 0)
                    for first, second in zip(regions, regions[1:]):
                        self.assertEqual(first[1], second[0])
                        self.assertEqual(second[0] % 512, 0)
                    self.assertLess(regions[-1][1], size)

    def test_typical_round_conservative_wqebb_bound(self):
        # Even one two-SGE WQE per routed token is bounded here, without assuming any NIC progress.
        # This is not a general BS/K/P flow-control proof or a multi-round lifetime proof.
        bs, cores, topk, peers = 4096, 56, 6, 8
        tiles = tile_ranges(bs, cores, topk)
        for sender in range(SENDERS):
            tokens = sum(end - begin for tile, (begin, end) in enumerate(tiles) if tile % SENDERS == sender)
            count_requests = (peers + SENDERS - 1) // SENDERS
            self.assertLess(2 * tokens * topk + peers - 1 + 2 * count_requests, 4096)

    def test_source_contract_dma_sizeof_fields_have_explicit_width(self):
        # Source guard for C++ list-initialization narrowing; this does not compile the CANN headers.
        source = (ROOT / "op_kernel/moe_ep_dispatch.h").read_text(encoding="utf-8")
        initializers = re.findall(r"DataCopy(?:Ext)?Params\s+\w+\s*(?:=\s*)?\{([^;]+)\};", source)
        fields = [field.strip() for init in initializers for field in init.split(",") if "sizeof(" in field]
        self.assertTrue(fields)
        for field in fields:
            with self.subTest(field=field):
                self.assertRegex(field, r"^static_cast<uint(?:16|32)_t>\([\s\S]+\)$")

    def test_source_contract_keeps_tile_protocol_and_async_completion(self):
        source = (ROOT / "op_kernel/moe_ep_dispatch.h").read_text(encoding="utf-8")
        # Guard duplicated device range formulas; this check does not execute either kernel helper.
        self.assertEqual(source.count("uint32_t start = tile * tileTokens_;"), 2)
        self.assertEqual(source.count("axisBS_ - start < tileTokens_ ? axisBS_ - start : tileTokens_;"), 2)
        for obsolete in ("GetTokenBoundary", "LowerBoundToken", "GetSlotStartNum", "SetAtomicAdd", "WaitPrepareReady"):
            self.assertNotIn(obsolete, source)
        self.assertIn("PrepareMetadata(tokenStart, tokenCount);", source)
        self.assertIn("tile += aivNum_", source)
        self.assertIn("tile += prepareCoreCount_", source)
        self.assertIn("tile += sendCoreCount_", source)
        self.assertIn("PublishSendCompletions();", source)
        self.assertNotIn(".Drain(", source)


if __name__ == "__main__":
    unittest.main()
